// @adai-status: stable        (TD-050 GPU incremental-cache forward added; TD-180 gated world-model/hippocampal cross-attention paths added, CPU forward/backward only — see class doc; TD-187 save()/load() now persist the gated paths, closing TD-180's own documented gap)
// @adai-version: 1.3.0
// @adai-reviewed: 2026-09-17

#include "DecoderBlock.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "Logger.hpp"
using adai::Logger;
#include "CrossAttention.hpp"

DecoderBlock::DecoderBlock(int d_model, int num_heads, int d_ff, float dropout,
                           bool enable_world_model, bool enable_hippocampal)
    : d_model(d_model), num_heads(num_heads), d_ff(d_ff), dropout_rate(dropout) {
    // Initialize self-attention (masked)
    self_attention = std::make_unique<MultiHeadAttention>(d_model, num_heads);

    // Initialize cross-attention (to encoder) - NEW: Uses CrossAttention class
    cross_attention = std::make_unique<CrossAttention>(d_model, num_heads);

    // Initialize feed-forward network
    feed_forward = std::make_unique<FeedForward>(d_model, d_ff);

    // Initialize layer normalization layers
    norm1 = std::make_unique<LayerNorm>(d_model);
    norm2 = std::make_unique<LayerNorm>(d_model);
    norm3 = std::make_unique<LayerNorm>(d_model);

    // Set learning rates for sub-components
    self_attention->learning_rate = learning_rate;
    cross_attention->learning_rate = learning_rate;
    feed_forward->learning_rate = learning_rate;
    norm1->learning_rate = learning_rate;
    norm2->learning_rate = learning_rate;
    norm3->learning_rate = learning_rate;

    // TD-180: gated world-model/hippocampal paths — allocated only when requested. Both flags
    // default to false (see header), so every pre-TD-180 call site (only one exists in
    // production code, LLMDecoder's own construction loop) is completely unaffected: these
    // unique_ptrs stay null, forward()'s gated branches are unreachable regardless of what's
    // passed to them, and no extra weight memory is allocated at all.
    if (enable_world_model) {
        world_model_cross_attention = std::make_unique<CrossAttention>(d_model, num_heads);
        norm_world = std::make_unique<LayerNorm>(d_model);
        world_model_cross_attention->learning_rate = learning_rate;
        norm_world->learning_rate = learning_rate;
    }
    if (enable_hippocampal) {
        hippocampal_cross_attention = std::make_unique<CrossAttention>(d_model, num_heads);
        norm_hippocampal = std::make_unique<LayerNorm>(d_model);
        hippocampal_cross_attention->learning_rate = learning_rate;
        norm_hippocampal->learning_rate = learning_rate;
    }

    Logger::info(
        "DecoderBlock initialized: d_model={} num_heads={} d_ff={} world_model={} hippocampal={}",
        d_model, num_heads, d_ff, enable_world_model, enable_hippocampal);
}

Matrix DecoderBlock::forward(const Matrix& input, const Matrix& encoder_output,
                             const Matrix& self_attn_mask, const Matrix* cross_attn_mask,
                             const Matrix* world_model_output, const Matrix* world_model_mask,
                             HippocampalMemory* memory, float repetition_alpha,
                             float repetition_decay) {
    // Cache input for backward pass
    cached_input = input;
    cached_encoder_output = encoder_output;

    // Step 1: Pre-attention layer normalization, then masked self-attention (causal)
    Matrix normed1 = norm1->forward(input);
    cached_normed1 = normed1;
    Matrix self_attn_out = self_attention->forward(normed1, &self_attn_mask);
    cached_self_attn_output = self_attn_out;

    // Step 2: First residual connection (input + self-attention output) —
    // stays unnormalized, matching Pre-LN's preserved residual stream.
    Matrix residual1(input.rows, input.cols);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            residual1(i, j) = input(i, j) + self_attn_out(i, j);
        }
    }
    cached_residual1 = residual1;

    // Step 3: Pre-cross-attention layer normalization, then cross-attention to
    // encoder output. Query from the normalized decoder stream, Key/Value from encoder.
    Matrix normed2 = norm2->forward(residual1);
    cached_normed2 = normed2;
    Matrix cross_attn_out = cross_attention->forward(normed2, encoder_output, cross_attn_mask);
    cached_cross_attn_output = cross_attn_out;

    // Step 4: Second residual connection (residual1 + cross-attention output)
    Matrix residual2(residual1.rows, residual1.cols);
    for (int i = 0; i < residual1.rows; ++i) {
        for (int j = 0; j < residual1.cols; ++j) {
            residual2(i, j) = residual1(i, j) + cross_attn_out(i, j);
        }
    }

    // Step 4a (TD-180): gated world-model cross-attention. No-op unless both a real input was
    // supplied AND this instance's own path was allocated at construction — either condition
    // alone reproduces pre-TD-180 output exactly, since `residual2` above is untouched
    // otherwise.
    world_model_path_active_ = (world_model_output != nullptr && world_model_cross_attention);
    if (world_model_path_active_) {
        Matrix normed_world = norm_world->forward(residual2);
        Matrix wm_attn =
            world_model_cross_attention->forward(normed_world, *world_model_output, world_model_mask);
        cached_wm_attn = wm_attn;
        const float gate_tanh = std::tanh(gate);
        for (int i = 0; i < residual2.rows; ++i) {
            for (int j = 0; j < residual2.cols; ++j) {
                residual2(i, j) += gate_tanh * wm_attn(i, j);
            }
        }
    }

    // Step 4b (TD-180): gated hippocampal cross-attention, repetition-penalized. No-op unless a
    // real, non-empty memory was supplied AND this instance's own path was allocated. Applied
    // after the world-model path, per the plan's own Component 6 ordering ("after Component 4's
    // world-model Add & Norm").
    hippocampal_path_active_ = (memory != nullptr && memory->size() > 0 && hippocampal_cross_attention);
    if (hippocampal_path_active_) {
        auto [keys, values] = memory->read_all();
        (void)values;  // see forward()'s own doc comment: keys double as CrossAttention's
                        // kv_input, so both K and V are derived from them — correct when a
                        // slot's value equals its key (the documented common case).
        const int n_slots = memory->size();

        Matrix normed_hippocampal = norm_hippocampal->forward(residual2);

        // score_bias[i][slot] = -repetition_alpha * coverage[slot], broadcast across every
        // query row — coverage is a per-slot quantity, not per-query-position (same convention
        // CrossAttention's own mask broadcasting already uses).
        std::deque<float>& coverage = memory->coverage_vector();
        Matrix score_bias(normed_hippocampal.rows, n_slots);
        for (int i = 0; i < normed_hippocampal.rows; ++i) {
            for (int slot = 0; slot < n_slots; ++slot) {
                score_bias(i, slot) = -repetition_alpha * coverage[slot];
            }
        }

        Matrix hm_attn =
            hippocampal_cross_attention->forward_with_scores(normed_hippocampal, keys, score_bias);
        cached_hm_attn = hm_attn;
        const float gate_h_tanh = std::tanh(gate_h);
        for (int i = 0; i < residual2.rows; ++i) {
            for (int j = 0; j < residual2.cols; ++j) {
                residual2(i, j) += gate_h_tanh * hm_attn(i, j);
            }
        }

        // Coverage update: decay-then-accumulate (per the plan's own Component 6 pseudocode).
        // Self-bounding by construction — decay_coverage() runs before accumulation, and the
        // per-slot mean attention weight added below is itself in [0, 1] (a softmax output), so
        // coverage[slot] converges to at most 1 / (1 - repetition_decay) regardless of how many
        // decode steps hammer the same slot. get_last_attention_weights() is
        // [query_rows, n_slots]; averaged over query rows to get one per-slot increment,
        // matching the plan's own "called once per decode step" framing generalized to a
        // multi-row (e.g. teacher-forced training) call.
        memory->decay_coverage(repetition_decay);
        const Matrix& attn_weights = hippocampal_cross_attention->get_last_attention_weights();
        const float inv_rows = 1.0f / static_cast<float>(attn_weights.rows);
        for (int slot = 0; slot < n_slots; ++slot) {
            float mean_weight = 0.0f;
            for (int i = 0; i < attn_weights.rows; ++i) {
                mean_weight += attn_weights(i, slot);
            }
            coverage[slot] += mean_weight * inv_rows;
        }
    }

    cached_residual2 = residual2;

    // Step 5: Pre-feedforward layer normalization, then feed-forward network
    Matrix normed3 = norm3->forward(residual2);
    Matrix ff_out = feed_forward->forward(normed3);
    cached_ff_output = ff_out;

    // Step 6: Third residual connection (residual2 + ff output) — the block's
    // output is unnormalized; LLMDecoder's final_norm normalizes the
    // accumulated residual stream once, after the last block.
    Matrix residual3(residual2.rows, residual2.cols);
    for (int i = 0; i < residual2.rows; ++i) {
        for (int j = 0; j < residual2.cols; ++j) {
            residual3(i, j) = residual2(i, j) + ff_out(i, j);
        }
    }
    cached_residual3 = residual3;

    return residual3;
}

Matrix DecoderBlock::forward_with_cache(const Matrix& input, const Matrix& encoder_output,
                                        const Matrix& self_attn_mask, KVCache* self_attn_cache,
                                        KVCache* cross_attn_cache, const Matrix* cross_attn_mask,
                                        bool use_cache) {
    // If no caching, fall back to regular forward
    if (!use_cache || self_attn_cache == nullptr) {
        return forward(input, encoder_output, self_attn_mask, cross_attn_mask);
    }

    // Cache input for backward pass (if needed for training)
    cached_input = input;
    cached_encoder_output = encoder_output;

    // Step 1: Pre-attention layer normalization, then masked self-attention with cache
    Matrix normed1 = norm1->forward(input);
    cached_normed1 = normed1;
    Matrix self_attn_out =
        self_attention->forward_with_cache(normed1, &self_attn_mask, self_attn_cache, use_cache);
    cached_self_attn_output = self_attn_out;

    // Step 2: First residual connection (input + self-attention output)
    Matrix residual1(input.rows, input.cols);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            residual1(i, j) = input(i, j) + self_attn_out(i, j);
        }
    }
    cached_residual1 = residual1;

    // Step 3: Pre-cross-attention layer normalization, then cross-attention with cache
    // For cross-attention, K/V are from encoder (constant), cached on first call
    Matrix normed2 = norm2->forward(residual1);
    cached_normed2 = normed2;
    Matrix cross_attn_out = cross_attention->forward_with_cache(
        normed2, encoder_output, cross_attn_mask, cross_attn_cache, use_cache);
    cached_cross_attn_output = cross_attn_out;

    // Step 4: Second residual connection (residual1 + cross-attention output)
    Matrix residual2(residual1.rows, residual1.cols);
    for (int i = 0; i < residual1.rows; ++i) {
        for (int j = 0; j < residual1.cols; ++j) {
            residual2(i, j) = residual1(i, j) + cross_attn_out(i, j);
        }
    }
    cached_residual2 = residual2;

    // Step 5: Pre-feedforward layer normalization, then feed-forward network (no caching needed)
    Matrix normed3 = norm3->forward(residual2);
    Matrix ff_out = feed_forward->forward(normed3);
    cached_ff_output = ff_out;

    // Step 6: Third residual connection (residual2 + ff output)
    Matrix residual3(residual2.rows, residual2.cols);
    for (int i = 0; i < residual2.rows; ++i) {
        for (int j = 0; j < residual2.cols; ++j) {
            residual3(i, j) = residual2(i, j) + ff_out(i, j);
        }
    }
    cached_residual3 = residual3;

    return residual3;
}

Matrix DecoderBlock::backward(const Matrix& grad_output) {
    Matrix unused_grad_encoder_output;
    return backward(grad_output, unused_grad_encoder_output);
}

Matrix DecoderBlock::backward(const Matrix& grad_output, Matrix& grad_encoder_output) {
    // Step 1: Gradient through third residual connection (output = residual2 + ff_output)
    // Gradient splits into two paths: directly to residual2, and through the
    // feed-forward branch (ff_output -> normed3 -> residual2)
    Matrix grad_residual2(grad_output.rows, grad_output.cols);
    Matrix grad_ff_output(grad_output.rows, grad_output.cols);

    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_residual2(i, j) = grad_output(i, j);
            grad_ff_output(i, j) = grad_output(i, j);
        }
    }

    // Step 2: Gradient through feed-forward network
    Matrix grad_normed3 = feed_forward->backward(grad_ff_output);

    // Step 3: Gradient through pre-feedforward layer norm
    Matrix grad_residual2_from_norm3 = norm3->backward(grad_normed3);

    // Step 4: Accumulate gradients from both paths into residual2
    for (int i = 0; i < grad_residual2.rows; ++i) {
        for (int j = 0; j < grad_residual2.cols; ++j) {
            grad_residual2(i, j) += grad_residual2_from_norm3(i, j);
        }
    }

    // Step 4a/4b (TD-180): backprop through the gated paths, in reverse of the order forward()
    // applied them (hippocampal was applied last, so it's unwound first). grad_residual2 at
    // this point is the gradient w.r.t. whatever value fed norm3 — the POST-gated-path residual2
    // if either path was active on the most recent forward() call. Both paths' own K/V-source
    // gradients (w.r.t. the hippocampal memory's keys / world_model_output) are computed,
    // since CrossAttention::backward()'s own signature always produces them, but discarded —
    // see this method's own doc comment in DecoderBlock.hpp for why.
    if (hippocampal_path_active_) {
        const float gate_h_tanh = std::tanh(gate_h);
        Matrix grad_hm_attn(grad_residual2.rows, grad_residual2.cols);
        float gate_h_dot = 0.0f;
        for (int i = 0; i < grad_residual2.rows; ++i) {
            for (int j = 0; j < grad_residual2.cols; ++j) {
                grad_hm_attn(i, j) = grad_residual2(i, j) * gate_h_tanh;
                gate_h_dot += grad_residual2(i, j) * cached_hm_attn(i, j);
            }
        }
        gate_h_grad += gate_h_dot * (1.0f - gate_h_tanh * gate_h_tanh);  // d(tanh)/dx = 1 - tanh^2

        Matrix grad_normed_hippocampal, grad_memory_keys_unused;
        hippocampal_cross_attention->backward(grad_hm_attn, grad_normed_hippocampal,
                                              grad_memory_keys_unused);
        Matrix grad_from_hippocampal = norm_hippocampal->backward(grad_normed_hippocampal);
        for (int i = 0; i < grad_residual2.rows; ++i) {
            for (int j = 0; j < grad_residual2.cols; ++j) {
                grad_residual2(i, j) += grad_from_hippocampal(i, j);
            }
        }
    }

    if (world_model_path_active_) {
        const float gate_tanh = std::tanh(gate);
        Matrix grad_wm_attn(grad_residual2.rows, grad_residual2.cols);
        float gate_dot = 0.0f;
        for (int i = 0; i < grad_residual2.rows; ++i) {
            for (int j = 0; j < grad_residual2.cols; ++j) {
                grad_wm_attn(i, j) = grad_residual2(i, j) * gate_tanh;
                gate_dot += grad_residual2(i, j) * cached_wm_attn(i, j);
            }
        }
        gate_grad += gate_dot * (1.0f - gate_tanh * gate_tanh);

        Matrix grad_normed_world, grad_world_model_output_unused;
        world_model_cross_attention->backward(grad_wm_attn, grad_normed_world,
                                              grad_world_model_output_unused);
        Matrix grad_from_world = norm_world->backward(grad_normed_world);
        for (int i = 0; i < grad_residual2.rows; ++i) {
            for (int j = 0; j < grad_residual2.cols; ++j) {
                grad_residual2(i, j) += grad_from_world(i, j);
            }
        }
    }

    // Step 5: Gradient through second residual connection (residual2 = residual1 + cross_attn_output)
    // Gradient splits into two paths: directly to residual1, and through the
    // cross-attention branch (cross_attn_output -> normed2 -> residual1)
    Matrix grad_residual1(grad_residual2.rows, grad_residual2.cols);
    Matrix grad_cross_attn_output(grad_residual2.rows, grad_residual2.cols);

    for (int i = 0; i < grad_residual2.rows; ++i) {
        for (int j = 0; j < grad_residual2.cols; ++j) {
            grad_residual1(i, j) = grad_residual2(i, j);
            grad_cross_attn_output(i, j) = grad_residual2(i, j);
        }
    }

    // Step 6: Gradient through cross-attention
    Matrix grad_normed2_from_cross;
    cross_attention->backward(grad_cross_attn_output, grad_normed2_from_cross,
                              grad_encoder_output);

    // Step 7: Gradient through pre-cross-attention layer norm
    Matrix grad_residual1_from_norm2 = norm2->backward(grad_normed2_from_cross);

    // Step 8: Accumulate gradients from both paths into residual1
    for (int i = 0; i < grad_residual1.rows; ++i) {
        for (int j = 0; j < grad_residual1.cols; ++j) {
            grad_residual1(i, j) += grad_residual1_from_norm2(i, j);
        }
    }

    // Step 9: Gradient through first residual connection (residual1 = input + self_attn_output)
    // Gradient splits into two paths: directly to input, and through the
    // self-attention branch (self_attn_output -> normed1 -> input)
    Matrix grad_input(grad_residual1.rows, grad_residual1.cols);
    Matrix grad_self_attn_output(grad_residual1.rows, grad_residual1.cols);

    for (int i = 0; i < grad_residual1.rows; ++i) {
        for (int j = 0; j < grad_residual1.cols; ++j) {
            grad_input(i, j) = grad_residual1(i, j);
            grad_self_attn_output(i, j) = grad_residual1(i, j);
        }
    }

    // Step 10: Gradient through self-attention
    Matrix grad_normed1 = self_attention->backward(grad_self_attn_output);

    // Step 11: Gradient through pre-attention layer norm
    Matrix grad_input_from_norm1 = norm1->backward(grad_normed1);

    // Step 12: Accumulate gradients from both paths into input
    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            grad_input(i, j) += grad_input_from_norm1(i, j);
        }
    }

    return grad_input;
}

void DecoderBlock::update_weights() {
    self_attention->update_weights();
    cross_attention->update_weights();
    feed_forward->update_weights();
    norm1->update_weights();
    norm2->update_weights();
    norm3->update_weights();

    // TD-180: gated paths' own sub-components follow the exact same pattern as the six above
    // (each already handles both the optimizer-registered and plain-SGD cases internally). gate/
    // gate_h have no Optimizer equivalent (see register_parameters_with_optimizer()'s own doc
    // comment), so they're always plain SGD here regardless of whether an optimizer is
    // registered elsewhere on this instance.
    if (world_model_cross_attention) {
        world_model_cross_attention->update_weights();
        norm_world->update_weights();
        gate -= learning_rate * gate_grad;
        gate_grad = 0.0f;
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->update_weights();
        norm_hippocampal->update_weights();
        gate_h -= learning_rate * gate_h_grad;
        gate_h_grad = 0.0f;
    }
}

void DecoderBlock::zero_grad() {
    self_attention->zero_grad();
    cross_attention->zero_grad();
    feed_forward->zero_grad();
    norm1->zero_grad();
    norm2->zero_grad();
    norm3->zero_grad();

    if (world_model_cross_attention) {
        world_model_cross_attention->zero_grad();
        norm_world->zero_grad();
        gate_grad = 0.0f;
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->zero_grad();
        norm_hippocampal->zero_grad();
        gate_h_grad = 0.0f;
    }
}

float DecoderBlock::get_gradient_norm() const {
    float norm_sq = 0.0f;

    // Accumulate squared norms from all components (combined in quadrature)
    float self_attn_norm = self_attention->get_gradient_norm();
    norm_sq += self_attn_norm * self_attn_norm;

    float cross_attn_norm = cross_attention->get_gradient_norm();
    norm_sq += cross_attn_norm * cross_attn_norm;

    float ff_norm = feed_forward->get_gradient_norm();
    norm_sq += ff_norm * ff_norm;

    // LayerNorm doesn't expose gradient norm method
    // Approximation: gradients handled internally during backward

    // TD-180: gated paths' own cross-attention gradient norms, plus the gate scalars
    // themselves — same quadrature combination as everything else here.
    if (world_model_cross_attention) {
        float wm_norm = world_model_cross_attention->get_gradient_norm();
        norm_sq += wm_norm * wm_norm + gate_grad * gate_grad;
    }
    if (hippocampal_cross_attention) {
        float hm_norm = hippocampal_cross_attention->get_gradient_norm();
        norm_sq += hm_norm * hm_norm + gate_h_grad * gate_h_grad;
    }

    return std::sqrt(norm_sq);
}

void DecoderBlock::set_learning_rate(float lr) {
    learning_rate = lr;
    self_attention->learning_rate = lr;
    cross_attention->learning_rate = lr;
    feed_forward->learning_rate = lr;
    norm1->learning_rate = lr;
    norm2->learning_rate = lr;
    norm3->learning_rate = lr;

    if (world_model_cross_attention) {
        world_model_cross_attention->learning_rate = lr;
        norm_world->learning_rate = lr;
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->learning_rate = lr;
        norm_hippocampal->learning_rate = lr;
    }
}

void DecoderBlock::save(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    // Save dimensions and hyperparameters
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(d_model));
    file.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
    file.write(reinterpret_cast<const char*>(&d_ff), sizeof(d_ff));
    file.write(reinterpret_cast<const char*>(&dropout_rate), sizeof(dropout_rate));
    file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));

    // Save LayerNorm parameters inline
    const Matrix& gamma1 = norm1->get_gamma();
    const Matrix& beta1 = norm1->get_beta();
    for (int j = 0; j < gamma1.cols; ++j) {
        file.write(reinterpret_cast<const char*>(&gamma1(0, j)), sizeof(float));
    }
    for (int j = 0; j < beta1.cols; ++j) {
        file.write(reinterpret_cast<const char*>(&beta1(0, j)), sizeof(float));
    }

    const Matrix& gamma2 = norm2->get_gamma();
    const Matrix& beta2 = norm2->get_beta();
    for (int j = 0; j < gamma2.cols; ++j) {
        file.write(reinterpret_cast<const char*>(&gamma2(0, j)), sizeof(float));
    }
    for (int j = 0; j < beta2.cols; ++j) {
        file.write(reinterpret_cast<const char*>(&beta2(0, j)), sizeof(float));
    }

    const Matrix& gamma3 = norm3->get_gamma();
    const Matrix& beta3 = norm3->get_beta();
    for (int j = 0; j < gamma3.cols; ++j) {
        file.write(reinterpret_cast<const char*>(&gamma3(0, j)), sizeof(float));
    }
    for (int j = 0; j < beta3.cols; ++j) {
        file.write(reinterpret_cast<const char*>(&beta3(0, j)), sizeof(float));
    }

    // TD-187: gated world-model path, only when this instance actually has one allocated. The
    // presence flag is written unconditionally so load() always knows whether a gate value
    // follows and whether a matching .world_model_cross_attn/.norm_world file pair exists,
    // without having to guess from this instance's own construction (which may differ from
    // whatever instance eventually calls load()).
    const bool has_world_model = (world_model_cross_attention != nullptr);
    file.write(reinterpret_cast<const char*>(&has_world_model), sizeof(has_world_model));
    if (has_world_model) {
        file.write(reinterpret_cast<const char*>(&gate), sizeof(gate));
    }

    // TD-187: gated hippocampal path, same convention as the world-model path above.
    const bool has_hippocampal = (hippocampal_cross_attention != nullptr);
    file.write(reinterpret_cast<const char*>(&has_hippocampal), sizeof(has_hippocampal));
    if (has_hippocampal) {
        file.write(reinterpret_cast<const char*>(&gate_h), sizeof(gate_h));
    }

    file.close();

    // Save sub-components to separate files
    self_attention->save_weights(filepath + ".self_attn");
    cross_attention->save(filepath + ".cross_attn");
    feed_forward->save_weights(filepath + ".ff");

    // TD-187: gated paths' own sub-components, only written when actually allocated (matching
    // the presence flags just written above).
    if (has_world_model) {
        world_model_cross_attention->save(filepath + ".world_model_cross_attn");
        norm_world->save_weights(filepath + ".norm_world");
    }
    if (has_hippocampal) {
        hippocampal_cross_attention->save(filepath + ".hippocampal_cross_attn");
        norm_hippocampal->save_weights(filepath + ".norm_hippocampal");
    }
}

void DecoderBlock::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    // Load dimensions and hyperparameters
    int loaded_d_model = 0, loaded_num_heads = 0, loaded_d_ff = 0;
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(loaded_d_model));
    file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(loaded_num_heads));
    file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(loaded_d_ff));
    file.read(reinterpret_cast<char*>(&dropout_rate), sizeof(dropout_rate));
    file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));

    if (loaded_d_model != d_model || loaded_num_heads != num_heads || loaded_d_ff != d_ff) {
        throw std::runtime_error("Dimension mismatch in saved model");
    }

    // Load LayerNorm parameters inline
    Matrix gamma1(1, d_model), beta1(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&gamma1(0, j)), sizeof(float));
    }
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&beta1(0, j)), sizeof(float));
    }
    norm1->set_gamma(gamma1);
    norm1->set_beta(beta1);

    Matrix gamma2(1, d_model), beta2(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&gamma2(0, j)), sizeof(float));
    }
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&beta2(0, j)), sizeof(float));
    }
    norm2->set_gamma(gamma2);
    norm2->set_beta(beta2);

    Matrix gamma3(1, d_model), beta3(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&gamma3(0, j)), sizeof(float));
    }
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&beta3(0, j)), sizeof(float));
    }
    norm3->set_gamma(gamma3);
    norm3->set_beta(beta3);

    // TD-187: gated world-model path presence flag — must match this instance's own
    // construction-time enable_world_model exactly (see load()'s own doc comment for why a
    // mismatch throws rather than silently dropping or fabricating trained state).
    bool loaded_has_world_model = false;
    file.read(reinterpret_cast<char*>(&loaded_has_world_model), sizeof(loaded_has_world_model));
    float loaded_gate = 0.0f;
    if (loaded_has_world_model) {
        file.read(reinterpret_cast<char*>(&loaded_gate), sizeof(loaded_gate));
    }
    const bool current_has_world_model = (world_model_cross_attention != nullptr);
    if (loaded_has_world_model != current_has_world_model) {
        throw std::runtime_error(
            std::string("DecoderBlock::load(): saved world-model gated path is ") +
            (loaded_has_world_model ? "present" : "absent") +
            " but this instance was constructed with enable_world_model=" +
            (current_has_world_model ? "true" : "false") +
            " — construct with a matching flag before loading");
    }

    // TD-187: gated hippocampal path presence flag, same convention as above.
    bool loaded_has_hippocampal = false;
    file.read(reinterpret_cast<char*>(&loaded_has_hippocampal), sizeof(loaded_has_hippocampal));
    float loaded_gate_h = 0.0f;
    if (loaded_has_hippocampal) {
        file.read(reinterpret_cast<char*>(&loaded_gate_h), sizeof(loaded_gate_h));
    }
    const bool current_has_hippocampal = (hippocampal_cross_attention != nullptr);
    if (loaded_has_hippocampal != current_has_hippocampal) {
        throw std::runtime_error(
            std::string("DecoderBlock::load(): saved hippocampal gated path is ") +
            (loaded_has_hippocampal ? "present" : "absent") +
            " but this instance was constructed with enable_hippocampal=" +
            (current_has_hippocampal ? "true" : "false") +
            " — construct with a matching flag before loading");
    }

    file.close();

    // Load sub-components
    self_attention->load_weights(filepath + ".self_attn");
    cross_attention->load(filepath + ".cross_attn");
    feed_forward->load_weights(filepath + ".ff");

    // Update learning rates
    self_attention->learning_rate = learning_rate;
    cross_attention->learning_rate = learning_rate;
    feed_forward->learning_rate = learning_rate;

    // TD-187: gated paths' own sub-components + gate scalars, only when actually present (the
    // mismatch checks above already guarantee current_has_* matches loaded_has_* exactly here).
    if (current_has_world_model) {
        gate = loaded_gate;
        world_model_cross_attention->load(filepath + ".world_model_cross_attn");
        norm_world->load_weights(filepath + ".norm_world");
        world_model_cross_attention->learning_rate = learning_rate;
        norm_world->learning_rate = learning_rate;
    }
    if (current_has_hippocampal) {
        gate_h = loaded_gate_h;
        hippocampal_cross_attention->load(filepath + ".hippocampal_cross_attn");
        norm_hippocampal->load_weights(filepath + ".norm_hippocampal");
        hippocampal_cross_attention->learning_rate = learning_rate;
        norm_hippocampal->learning_rate = learning_rate;
    }
}

void DecoderBlock::register_parameters_with_optimizer(Optimizer& optimizer) {
    // Register all sub-component parameters
    self_attention->set_optimizer(&optimizer);
    cross_attention->set_optimizer(&optimizer);
    feed_forward->set_optimizer(&optimizer);
    norm1->set_optimizer(&optimizer);
    norm2->set_optimizer(&optimizer);
    norm3->set_optimizer(&optimizer);

    // TD-180: gated paths' own CrossAttention/LayerNorm sub-components, when allocated. gate/
    // gate_h are NOT registered here — they're plain floats with no ParameterGroup equivalent
    // in Optimizer's Matrix-based registration API (see this method's own header doc comment).
    if (world_model_cross_attention) {
        world_model_cross_attention->set_optimizer(&optimizer);
        norm_world->set_optimizer(&optimizer);
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->set_optimizer(&optimizer);
        norm_hippocampal->set_optimizer(&optimizer);
    }
}

#ifdef ADAI_ENABLE_GPU
void DecoderBlock::gpu_upload_weights() {
    self_attention->gpu_upload_weights();
    cross_attention->gpu_upload_weights();
    feed_forward->gpu_upload_weights();
    norm1->gpu_upload_weights();
    norm2->gpu_upload_weights();
    norm3->gpu_upload_weights();
}

void DecoderBlock::gpu_download_grads() {
    self_attention->gpu_download_grads();
    cross_attention->gpu_download_grads();
    feed_forward->gpu_download_grads();
    norm1->gpu_download_grads();
    norm2->gpu_download_grads();
    norm3->gpu_download_grads();
}

void DecoderBlock::gpu_zero_grads() {
    self_attention->gpu_zero_grads();
    cross_attention->gpu_zero_grads();
    feed_forward->gpu_zero_grads();
    norm1->gpu_zero_grads();
    norm2->gpu_zero_grads();
    norm3->gpu_zero_grads();
}

adai::gpu::GPUMatrix DecoderBlock::gpu_forward(const adai::gpu::GPUMatrix& input,
                                               const adai::gpu::GPUMatrix& encoder_out,
                                               const adai::gpu::GPUMatrix* self_mask) {
    // 1. norm1 -> masked self-attention -> residual1 (unnormalized)
    adai::gpu::GPUMatrix normed1 = norm1->gpu_forward(input);
    adai::gpu::GPUMatrix self_attn = self_attention->gpu_forward(normed1, self_mask);
    adai::gpu::GPUMatrix res1 = input + self_attn;

    // 2. norm2 -> cross-attention -> residual2 (unnormalized)
    adai::gpu::GPUMatrix normed2 = norm2->gpu_forward(res1);
    adai::gpu::GPUMatrix cross_attn = cross_attention->gpu_forward(normed2, encoder_out);
    adai::gpu::GPUMatrix res2 = res1 + cross_attn;

    // 3. norm3 -> feed-forward -> residual3 (unnormalized — final_norm handles
    // normalization once, after the last block, at the LLMDecoder level)
    adai::gpu::GPUMatrix normed3 = norm3->gpu_forward(res2);
    adai::gpu::GPUMatrix ff_out = feed_forward->gpu_forward(normed3);
    return res2 + ff_out;
}

adai::gpu::GPUMatrix DecoderBlock::gpu_forward_with_cache(
    const adai::gpu::GPUMatrix& input, const adai::gpu::GPUMatrix& encoder_out,
    const adai::gpu::GPUMatrix& self_attn_mask, adai::gpu::GPUKVCache* self_attn_cache,
    adai::gpu::GPUKVCache* cross_attn_cache, bool use_cache) {
    // Identical structure to gpu_forward() above — only the two attention sub-layers change,
    // to their own *_with_cache() incremental equivalents. Feed-forward and every norm are
    // exactly as in gpu_forward(), since neither depends on sequence position.
    adai::gpu::GPUMatrix normed1 = norm1->gpu_forward(input);
    adai::gpu::GPUMatrix self_attn = self_attention->gpu_forward_with_cache(
        normed1, &self_attn_mask, self_attn_cache, use_cache);
    adai::gpu::GPUMatrix res1 = input + self_attn;

    adai::gpu::GPUMatrix normed2 = norm2->gpu_forward(res1);
    adai::gpu::GPUMatrix cross_attn = cross_attention->gpu_forward_with_cache(
        normed2, encoder_out, nullptr, cross_attn_cache, use_cache);
    adai::gpu::GPUMatrix res2 = res1 + cross_attn;

    adai::gpu::GPUMatrix normed3 = norm3->gpu_forward(res2);
    adai::gpu::GPUMatrix ff_out = feed_forward->gpu_forward(normed3);
    return res2 + ff_out;
}

std::pair<adai::gpu::GPUMatrix, adai::gpu::GPUMatrix> DecoderBlock::gpu_backward(
    const adai::gpu::GPUMatrix& dout) {
    // Residual3 split: d_res2 = dout (direct) + norm3/ff branch
    adai::gpu::GPUMatrix d_normed3 = feed_forward->gpu_backward(dout);
    adai::gpu::GPUMatrix d_res2_from_norm3 = norm3->gpu_backward(d_normed3);
    adai::gpu::GPUMatrix d_res2 = dout + d_res2_from_norm3;

    // Residual2 split: d_res1 = d_res2 (direct) + norm2/cross-attn branch
    // cross_attention backward returns {d_query=d_normed2, d_kv=d_encoder}
    auto [d_normed2, d_enc] = cross_attention->gpu_backward(d_res2);
    adai::gpu::GPUMatrix d_res1_from_norm2 = norm2->gpu_backward(d_normed2);
    adai::gpu::GPUMatrix d_res1 = d_res2 + d_res1_from_norm2;

    // Residual1 split: d_input = d_res1 (direct) + norm1/self-attn branch
    adai::gpu::GPUMatrix d_normed1 = self_attention->gpu_backward(d_res1);
    adai::gpu::GPUMatrix d_input_from_norm1 = norm1->gpu_backward(d_normed1);
    return {d_res1 + d_input_from_norm1, std::move(d_enc)};
}
#endif
