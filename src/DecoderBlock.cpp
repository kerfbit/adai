// @adai-status: stable        (TD-050 GPU incremental-cache forward added; TD-180 gated world-model/hippocampal cross-attention paths added, CPU forward/backward only — see class doc; TD-187 save()/load() now persist the gated paths, closing TD-180's own documented gap; TD-194 gated cross-reference pull + Hebbian association strengthening added to the hippocampal path)
// @adai-version: 1.8.1
// @adai-reviewed: 2026-10-26

#include "DecoderBlock.hpp"
#include "BinaryIO.hpp"
#include <cstdint>
#include "gpu/GPUParamSync.hpp"
#include <cmath>
#include <deque>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "Logger.hpp"
using adai::Logger;
#include "CrossAttention.hpp"

// TODO: See TD-344 in TECHNICAL_DEBT.md - dropout is stored but never applied
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
                             float repetition_decay, float cross_reference_alpha,
                             float association_decay) {
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

    apply_gated_paths(residual2, world_model_output, world_model_mask, memory, repetition_alpha,
                      repetition_decay, cross_reference_alpha, association_decay);

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

namespace {

// TD-342: the hippocampal path's per-slot score bias, shared by the CPU and GPU forwards (TD-180,
// TD-194). Rows are the attention's query rows.
Matrix hippocampal_score_bias(HippocampalMemory& memory_ref, int rows, float repetition_alpha,
                              float cross_reference_alpha) {
    HippocampalMemory* memory = &memory_ref;
    const int n_slots = memory->size();
    // score_bias[i][slot] = -repetition_alpha * coverage[slot]
    //                     + cross_reference_alpha * tanh(sum_j(association[slot][j] * coverage[j]))
    // (TD-194's own second term added below), broadcast across every query row — both terms
    // are per-slot quantities, not per-query-position (same convention CrossAttention's own
    // mask broadcasting already uses).
    std::deque<float>& coverage = memory->coverage_vector();
    std::deque<std::deque<float>>& association = memory->association_matrix();

    // TD-194 gated cross-reference pull: a slot strongly associated with other *currently
    // recently-used* slots (their own coverage, from prior decode steps — this step's own
    // attention hasn't happened yet) gets pulled toward attention this step too. tanh keeps
    // the pull bounded regardless of how large association/coverage grow, the same gating
    // idiom gate_h's own blend below already uses. Computed once per slot (not per query row)
    // since neither association nor coverage vary across query rows.
    std::vector<float> cross_ref_boost(n_slots, 0.0f);
    if (cross_reference_alpha != 0.0f) {
        for (int slot = 0; slot < n_slots; ++slot) {
            float signal = 0.0f;
            for (int other = 0; other < n_slots; ++other) {
                if (other != slot) {
                    signal += association[slot][other] * coverage[other];
                }
            }
            cross_ref_boost[slot] = cross_reference_alpha * std::tanh(signal);
        }
    }

    Matrix score_bias(rows, n_slots);
    for (int i = 0; i < rows; ++i) {
        for (int slot = 0; slot < n_slots; ++slot) {
            score_bias(i, slot) = -repetition_alpha * coverage[slot] + cross_ref_boost[slot];
        }
    }
    return score_bias;
}


}  // namespace

// Steps 4a/4b of forward() (TD-180), shared with forward_with_cache() (TD-342) so cached and
// uncached decoding apply the gated paths identically.
void DecoderBlock::apply_gated_paths(Matrix& residual2, const Matrix* world_model_output,
                                     const Matrix* world_model_mask, HippocampalMemory* memory,
                                     float repetition_alpha, float repetition_decay,
                                     float cross_reference_alpha, float association_decay) {
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
        const float gate_tanh = std::tanh(gate.data[0][0]);
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
    last_hippocampal_source_ = hippocampal_path_active_ ? 1 : 0;
    if (hippocampal_path_active_) {
        auto [keys, values] = memory->read_all();
        (void)values;  // see forward()'s own doc comment: keys double as CrossAttention's
                        // kv_input, so both K and V are derived from them — correct when a
                        // slot's value equals its key (the documented common case).
        Matrix normed_hippocampal = norm_hippocampal->forward(residual2);

        Matrix score_bias = hippocampal_score_bias(*memory, normed_hippocampal.rows,
                                                   repetition_alpha, cross_reference_alpha);

        Matrix hm_attn =
            hippocampal_cross_attention->forward_with_scores(normed_hippocampal, keys, score_bias);
        cached_hm_attn = hm_attn;
        const float gate_h_tanh = std::tanh(gate_h.data[0][0]);
        for (int i = 0; i < residual2.rows; ++i) {
            for (int j = 0; j < residual2.cols; ++j) {
                residual2(i, j) += gate_h_tanh * hm_attn(i, j);
            }
        }

        // TD-343: no memory update here. The caller commits one coverage/association update per
        // decode step (HippocampalMemory::commit_attention()) from last_hippocampal_attention().
        (void)repetition_decay;
        (void)association_decay;
    }
}

Matrix DecoderBlock::forward_with_cache(const Matrix& input, const Matrix& encoder_output,
                                        const Matrix& self_attn_mask, KVCache* self_attn_cache,
                                        KVCache* cross_attn_cache, const Matrix* cross_attn_mask,
                                        bool use_cache, const Matrix* world_model_output,
                                        const Matrix* world_model_mask, HippocampalMemory* memory,
                                        float repetition_alpha, float repetition_decay,
                                        float cross_reference_alpha, float association_decay) {
    // If no caching, fall back to regular forward
    if (!use_cache || self_attn_cache == nullptr) {
        return forward(input, encoder_output, self_attn_mask, cross_attn_mask, world_model_output,
                       world_model_mask, memory, repetition_alpha, repetition_decay,
                       cross_reference_alpha, association_decay);
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
    // TD-342: the gated paths, as in forward(). The world-model K/V are recomputed each decode
    // step (not cached like the encoder's); the world-model sequence is short.
    apply_gated_paths(residual2, world_model_output, world_model_mask, memory, repetition_alpha,
                      repetition_decay, cross_reference_alpha, association_decay);
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
        const float gate_h_tanh = std::tanh(gate_h.data[0][0]);
        Matrix grad_hm_attn(grad_residual2.rows, grad_residual2.cols);
        float gate_h_dot = 0.0f;
        for (int i = 0; i < grad_residual2.rows; ++i) {
            for (int j = 0; j < grad_residual2.cols; ++j) {
                grad_hm_attn(i, j) = grad_residual2(i, j) * gate_h_tanh;
                gate_h_dot += grad_residual2(i, j) * cached_hm_attn(i, j);
            }
        }
        gate_h_grad.data[0][0] += gate_h_dot * (1.0f - gate_h_tanh * gate_h_tanh);  // d(tanh)/dx = 1 - tanh^2

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
        const float gate_tanh = std::tanh(gate.data[0][0]);
        Matrix grad_wm_attn(grad_residual2.rows, grad_residual2.cols);
        float gate_dot = 0.0f;
        for (int i = 0; i < grad_residual2.rows; ++i) {
            for (int j = 0; j < grad_residual2.cols; ++j) {
                grad_wm_attn(i, j) = grad_residual2(i, j) * gate_tanh;
                gate_dot += grad_residual2(i, j) * cached_wm_attn(i, j);
            }
        }
        gate_grad.data[0][0] += gate_dot * (1.0f - gate_tanh * gate_tanh);

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

// TODO: See TD-297 in TECHNICAL_DEBT.md - with a shared optimizer, each sub-layer's update_weights() calls step() again
void DecoderBlock::update_weights() {
    self_attention->update_weights();
    cross_attention->update_weights();
    feed_forward->update_weights();
    norm1->update_weights();
    norm2->update_weights();
    norm3->update_weights();

    // TD-180: gated paths' own sub-components follow the exact same pattern as the six above
    // (each already handles both the optimizer-registered and plain-SGD cases internally). The
    // gates are optimizer parameters once registered (TD-341); plain SGD only without one.
    if (world_model_cross_attention) {
        world_model_cross_attention->update_weights();
        norm_world->update_weights();
        if (!gate_optimizer_) {
            gate.data[0][0] -= learning_rate * gate_grad.data[0][0];
        }
        gate_grad.data[0][0] = 0.0f;
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->update_weights();
        norm_hippocampal->update_weights();
        if (!gate_optimizer_) {
            gate_h.data[0][0] -= learning_rate * gate_h_grad.data[0][0];
        }
        gate_h_grad.data[0][0] = 0.0f;
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
        gate_grad.data[0][0] = 0.0f;
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->zero_grad();
        norm_hippocampal->zero_grad();
        gate_h_grad.data[0][0] = 0.0f;
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
        norm_sq += wm_norm * wm_norm + gate_grad.data[0][0] * gate_grad.data[0][0];
    }
    if (hippocampal_cross_attention) {
        float hm_norm = hippocampal_cross_attention->get_gradient_norm();
        norm_sq += hm_norm * hm_norm + gate_h_grad.data[0][0] * gate_h_grad.data[0][0];
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
    adai::binio::Writer file(filepath);  // TD-423/TD-345: checked writes

    // Save dimensions and hyperparameters
    file.value(d_model, "d_model");
    file.value(num_heads, "num_heads");
    file.value(d_ff, "d_ff");
    file.value(dropout_rate, "dropout_rate");
    file.value(learning_rate, "learning_rate");

    // Save LayerNorm parameters inline
    file.matrix(norm1->get_gamma(), "norm1 gamma");
    file.matrix(norm1->get_beta(), "norm1 beta");
    file.matrix(norm2->get_gamma(), "norm2 gamma");
    file.matrix(norm2->get_beta(), "norm2 beta");
    file.matrix(norm3->get_gamma(), "norm3 gamma");
    file.matrix(norm3->get_beta(), "norm3 beta");

    // TD-187: gated world-model path, only when this instance actually has one allocated. The
    // presence flag is written unconditionally so load() always knows whether a gate value
    // follows and whether a matching .world_model_cross_attn/.norm_world file pair exists,
    // without having to guess from this instance's own construction (which may differ from
    // whatever instance eventually calls load()).
    const bool has_world_model = (world_model_cross_attention != nullptr);
    file.value(static_cast<std::uint8_t>(has_world_model), "world-model flag");
    if (has_world_model) {
        file.value(gate.data[0][0], "world-model gate");
    }

    // TD-187: gated hippocampal path, same convention as the world-model path above.
    const bool has_hippocampal = (hippocampal_cross_attention != nullptr);
    file.value(static_cast<std::uint8_t>(has_hippocampal), "hippocampal flag");
    if (has_hippocampal) {
        file.value(gate_h.data[0][0], "hippocampal gate");
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
    // TD-423/TD-345: checked reads; a truncated file used to load garbage, and dropout_rate and
    // learning_rate were overwritten before the dimensions were checked. This file is read whole and
    // validated before anything changes; the sub-component files check themselves.
    adai::binio::Reader file(filepath);
    const int loaded_d_model = file.value<int>("d_model");
    const int loaded_num_heads = file.value<int>("num_heads");
    const int loaded_d_ff = file.value<int>("d_ff");
    const float loaded_dropout_rate = file.value<float>("dropout_rate");
    const float loaded_learning_rate = file.value<float>("learning_rate");

    if (loaded_d_model != d_model || loaded_num_heads != num_heads || loaded_d_ff != d_ff) {
        throw std::runtime_error("Dimension mismatch in saved model");
    }

    // LayerNorm parameters inline
    Matrix gamma1 = file.matrix(1, d_model, "norm1 gamma");
    Matrix beta1 = file.matrix(1, d_model, "norm1 beta");
    Matrix gamma2 = file.matrix(1, d_model, "norm2 gamma");
    Matrix beta2 = file.matrix(1, d_model, "norm2 beta");
    Matrix gamma3 = file.matrix(1, d_model, "norm3 gamma");
    Matrix beta3 = file.matrix(1, d_model, "norm3 beta");

    // TD-187: gated world-model path presence flag — must match this instance's own
    // construction-time enable_world_model exactly (see load()'s own doc comment for why a
    // mismatch throws rather than silently dropping or fabricating trained state).
    const bool loaded_has_world_model = file.value<std::uint8_t>("world-model flag") != 0;
    float loaded_gate = 0.0f;
    if (loaded_has_world_model) {
        loaded_gate = file.value<float>("world-model gate");
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
    const bool loaded_has_hippocampal = file.value<std::uint8_t>("hippocampal flag") != 0;
    float loaded_gate_h = 0.0f;
    if (loaded_has_hippocampal) {
        loaded_gate_h = file.value<float>("hippocampal gate");
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
    file.expect_end();

    dropout_rate = loaded_dropout_rate;
    learning_rate = loaded_learning_rate;
    norm1->set_gamma(gamma1);
    norm1->set_beta(beta1);
    norm2->set_gamma(gamma2);
    norm2->set_beta(beta2);
    norm3->set_gamma(gamma3);
    norm3->set_beta(beta3);

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
        gate.data[0][0] = loaded_gate;
        world_model_cross_attention->load(filepath + ".world_model_cross_attn");
        norm_world->load_weights(filepath + ".norm_world");
        world_model_cross_attention->learning_rate = learning_rate;
        norm_world->learning_rate = learning_rate;
    }
    if (current_has_hippocampal) {
        gate_h.data[0][0] = loaded_gate_h;
        hippocampal_cross_attention->load(filepath + ".hippocampal_cross_attn");
        norm_hippocampal->load_weights(filepath + ".norm_hippocampal");
        hippocampal_cross_attention->learning_rate = learning_rate;
        norm_hippocampal->learning_rate = learning_rate;
    }
}

Matrix DecoderBlock::last_hippocampal_attention() const {
    if (last_hippocampal_source_ == 1) {
        return hippocampal_cross_attention->get_last_attention_weights();
    }
#ifdef ADAI_ENABLE_GPU
    if (last_hippocampal_source_ == 2) {
        return hippocampal_cross_attention->gpu_last_mean_attention_weights();
    }
#endif
    return Matrix();
}

float DecoderBlock::current_gate() {
#ifdef ADAI_ENABLE_GPU
    if (gated_gpu_) {
        float value = 0.0f;
        gated_gpu_->gate.download(&value, 1);
        return value;
    }
#endif
    return get_gate();
}

float DecoderBlock::current_gate_h() {
#ifdef ADAI_ENABLE_GPU
    if (gated_gpu_) {
        float value = 0.0f;
        gated_gpu_->gate_h.download(&value, 1);
        return value;
    }
#endif
    return get_gate_h();
}

void DecoderBlock::register_parameters_with_optimizer(Optimizer& optimizer) {
    // Register all sub-component parameters
    self_attention->set_optimizer(&optimizer);
    cross_attention->set_optimizer(&optimizer);
    feed_forward->set_optimizer(&optimizer);
    norm1->set_optimizer(&optimizer);
    norm2->set_optimizer(&optimizer);
    norm3->set_optimizer(&optimizer);

    // TD-180: gated paths' own CrossAttention/LayerNorm sub-components, when allocated, and
    // (TD-341) the gates themselves as 1×1 groups, so ChatbotTrainer's optimizer->step() opens
    // them. They have no device mirror, so GPU training steps them as host groups.
    if (world_model_cross_attention) {
        world_model_cross_attention->set_optimizer(&optimizer);
        norm_world->set_optimizer(&optimizer);
        optimizer.add_parameter_group(&gate, &gate_grad);
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->set_optimizer(&optimizer);
        norm_hippocampal->set_optimizer(&optimizer);
        optimizer.add_parameter_group(&gate_h, &gate_h_grad);
    }
    gate_optimizer_ = &optimizer;
}

#ifdef ADAI_ENABLE_GPU
DecoderBlock::GatedGPUState& DecoderBlock::gated_gpu() {
    if (!gated_gpu_) {
        gated_gpu_ = std::make_unique<GatedGPUState>();
        gated_gpu_->gate.upload(&gate.data[0][0], 1);
        gated_gpu_->gate_h.upload(&gate_h.data[0][0], 1);
        gated_gpu_->gate_grad.zero();
        gated_gpu_->gate_h_grad.zero();
    }
    return *gated_gpu_;
}

// TD-342: the gated world-model/hippocampal sub-layers and their gates are device parameters
// too, so GPU training trains them (they used to stay host-side and never ran on the GPU).
void DecoderBlock::gpu_upload_weights() {
    self_attention->gpu_upload_weights();
    cross_attention->gpu_upload_weights();
    feed_forward->gpu_upload_weights();
    norm1->gpu_upload_weights();
    norm2->gpu_upload_weights();
    norm3->gpu_upload_weights();
    if (world_model_cross_attention) {
        world_model_cross_attention->gpu_upload_weights();
        norm_world->gpu_upload_weights();
        gated_gpu().gate.upload(&gate.data[0][0], 1);
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->gpu_upload_weights();
        norm_hippocampal->gpu_upload_weights();
        gated_gpu().gate_h.upload(&gate_h.data[0][0], 1);
    }
}

void DecoderBlock::gpu_collect_parameters(std::vector<adai::gpu::GPUParamRef>& out, int layer) {
    self_attention->gpu_collect_parameters(out, layer);
    cross_attention->gpu_collect_parameters(out, layer);
    feed_forward->gpu_collect_parameters(out, layer);
    norm1->gpu_collect_parameters(out, layer);
    norm2->gpu_collect_parameters(out, layer);
    norm3->gpu_collect_parameters(out, layer);
    if (world_model_cross_attention) {
        world_model_cross_attention->gpu_collect_parameters(out, layer);
        norm_world->gpu_collect_parameters(out, layer);
        adai::gpu::add_param(out, gate, gated_gpu().gate, gated_gpu().gate_grad, layer);
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->gpu_collect_parameters(out, layer);
        norm_hippocampal->gpu_collect_parameters(out, layer);
        adai::gpu::add_param(out, gate_h, gated_gpu().gate_h, gated_gpu().gate_h_grad, layer);
    }
}

void DecoderBlock::gpu_download_weights() {
    self_attention->gpu_download_weights();
    cross_attention->gpu_download_weights();
    feed_forward->gpu_download_weights();
    norm1->gpu_download_weights();
    norm2->gpu_download_weights();
    norm3->gpu_download_weights();
    if (world_model_cross_attention) {
        world_model_cross_attention->gpu_download_weights();
        norm_world->gpu_download_weights();
        if (gated_gpu_) {
            adai::gpu::download_into(gated_gpu_->gate, gate);
        }
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->gpu_download_weights();
        norm_hippocampal->gpu_download_weights();
        if (gated_gpu_) {
            adai::gpu::download_into(gated_gpu_->gate_h, gate_h);
        }
    }
}

void DecoderBlock::gpu_download_grads() {
    self_attention->gpu_download_grads();
    cross_attention->gpu_download_grads();
    feed_forward->gpu_download_grads();
    norm1->gpu_download_grads();
    norm2->gpu_download_grads();
    norm3->gpu_download_grads();
    if (world_model_cross_attention) {
        world_model_cross_attention->gpu_download_grads();
        norm_world->gpu_download_grads();
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->gpu_download_grads();
        norm_hippocampal->gpu_download_grads();
    }
    if (gated_gpu_) {
        float g = 0.0f;
        gated_gpu_->gate_grad.download(&g, 1);
        gate_grad.data[0][0] += g;
        gated_gpu_->gate_h_grad.download(&g, 1);
        gate_h_grad.data[0][0] += g;
    }
}

void DecoderBlock::gpu_zero_grads() {
    self_attention->gpu_zero_grads();
    cross_attention->gpu_zero_grads();
    feed_forward->gpu_zero_grads();
    norm1->gpu_zero_grads();
    norm2->gpu_zero_grads();
    norm3->gpu_zero_grads();
    if (world_model_cross_attention) {
        world_model_cross_attention->gpu_zero_grads();
        norm_world->gpu_zero_grads();
    }
    if (hippocampal_cross_attention) {
        hippocampal_cross_attention->gpu_zero_grads();
        norm_hippocampal->gpu_zero_grads();
    }
    if (gated_gpu_) {
        gated_gpu_->gate_grad.zero();
        gated_gpu_->gate_h_grad.zero();
    }
}

// Steps 4a/4b of forward(), on the device. The world-model path transfers nothing. The
// hippocampal path keeps its coverage/association state on the host (HippocampalMemory), so
// per call it uploads the memory keys and score bias (slots x d_model, rows x slots) and
// downloads the mean attention weights (rows x slots) for the update — small, and the
// algorithm's state, not training math.
void DecoderBlock::gpu_apply_gated_paths(adai::gpu::GPUMatrix& res2,
                                         const adai::gpu::GPUMatrix* world_model_out,
                                         HippocampalMemory* memory, float repetition_alpha,
                                         float repetition_decay, float cross_reference_alpha,
                                         float association_decay) {
    const bool wm = world_model_out != nullptr && world_model_out->rows > 0 &&
                    world_model_cross_attention != nullptr;
    const bool hm = memory != nullptr && memory->size() > 0 && hippocampal_cross_attention != nullptr;
    last_hippocampal_source_ = hm ? 2 : 0;
    if (!wm && !hm) {
        if (gated_gpu_) {
            gated_gpu_->wm_active = gated_gpu_->hm_active = false;
        }
        return;
    }
    GatedGPUState& g = gated_gpu();
    const int n = res2.rows * res2.cols;
    g.wm_active = wm;
    if (wm) {
        adai::gpu::GPUMatrix normed_world = norm_world->gpu_forward(res2);
        g.wm_attn = std::make_unique<adai::gpu::GPUMatrix>(
            world_model_cross_attention->gpu_forward(normed_world, *world_model_out));
        adai::gpu::gated_residual_add_gpu(res2.device_ptr(), g.wm_attn->device_ptr(),
                                          g.gate.device_ptr(), n);
    }
    g.hm_active = hm;
    if (hm) {
        auto [keys, values] = memory->read_all();
        (void)values;  // keys double as K/V, as in forward()
        adai::gpu::GPUMatrix keys_dev = keys.to_gpu();
        adai::gpu::GPUMatrix normed_hippocampal = norm_hippocampal->gpu_forward(res2);
        Matrix bias = hippocampal_score_bias(*memory, res2.rows, repetition_alpha,
                                             cross_reference_alpha);
        adai::gpu::GPUMatrix bias_dev = bias.to_gpu();
        g.hm_attn = std::make_unique<adai::gpu::GPUMatrix>(hippocampal_cross_attention->gpu_forward(
            normed_hippocampal, keys_dev, nullptr, &bias_dev));
        adai::gpu::gated_residual_add_gpu(res2.device_ptr(), g.hm_attn->device_ptr(),
                                          g.gate_h.device_ptr(), n);
        // TD-343: no memory update (or weight download) here; see the CPU path.
        (void)repetition_decay;
        (void)association_decay;
    }
}

adai::gpu::GPUMatrix DecoderBlock::gpu_forward(const adai::gpu::GPUMatrix& input,
                                               const adai::gpu::GPUMatrix& encoder_out,
                                               const adai::gpu::GPUMatrix* self_mask,
                                               const adai::gpu::GPUMatrix* world_model_out,
                                               HippocampalMemory* memory, float repetition_alpha,
                                               float repetition_decay, float cross_reference_alpha,
                                               float association_decay) {
    // 1. norm1 -> masked self-attention -> residual1 (unnormalized)
    adai::gpu::GPUMatrix normed1 = norm1->gpu_forward(input);
    adai::gpu::GPUMatrix self_attn = self_attention->gpu_forward(normed1, self_mask);
    adai::gpu::GPUMatrix res1 = input + self_attn;

    // 2. norm2 -> cross-attention -> residual2 (unnormalized)
    adai::gpu::GPUMatrix normed2 = norm2->gpu_forward(res1);
    adai::gpu::GPUMatrix cross_attn = cross_attention->gpu_forward(normed2, encoder_out);
    adai::gpu::GPUMatrix res2 = res1 + cross_attn;
    gpu_apply_gated_paths(res2, world_model_out, memory, repetition_alpha, repetition_decay,
                          cross_reference_alpha, association_decay);  // TD-342

    // 3. norm3 -> feed-forward -> residual3 (unnormalized — final_norm handles
    // normalization once, after the last block, at the LLMDecoder level)
    adai::gpu::GPUMatrix normed3 = norm3->gpu_forward(res2);
    adai::gpu::GPUMatrix ff_out = feed_forward->gpu_forward(normed3);
    return res2 + ff_out;
}

adai::gpu::GPUMatrix DecoderBlock::gpu_forward_with_cache(
    const adai::gpu::GPUMatrix& input, const adai::gpu::GPUMatrix& encoder_out,
    const adai::gpu::GPUMatrix& self_attn_mask, adai::gpu::GPUKVCache* self_attn_cache,
    adai::gpu::GPUKVCache* cross_attn_cache, bool use_cache,
    const adai::gpu::GPUMatrix* world_model_out, HippocampalMemory* memory, float repetition_alpha,
    float repetition_decay, float cross_reference_alpha, float association_decay) {
    // Identical structure to gpu_forward() above — only the two attention sub-layers change,
    // to their own *_with_cache() incremental equivalents. Feed-forward and every norm are
    // exactly as in gpu_forward(), since neither depends on sequence position. The gated paths
    // (TD-342) recompute their K/V each step, as the CPU forward_with_cache().
    adai::gpu::GPUMatrix normed1 = norm1->gpu_forward(input);
    adai::gpu::GPUMatrix self_attn = self_attention->gpu_forward_with_cache(
        normed1, &self_attn_mask, self_attn_cache, use_cache);
    adai::gpu::GPUMatrix res1 = input + self_attn;

    adai::gpu::GPUMatrix normed2 = norm2->gpu_forward(res1);
    adai::gpu::GPUMatrix cross_attn = cross_attention->gpu_forward_with_cache(
        normed2, encoder_out, nullptr, cross_attn_cache, use_cache);
    adai::gpu::GPUMatrix res2 = res1 + cross_attn;
    gpu_apply_gated_paths(res2, world_model_out, memory, repetition_alpha, repetition_decay,
                          cross_reference_alpha, association_decay);

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

    // TD-342: the gated paths, in reverse of the forward (hippocampal, then world model), as
    // backward(). Each adds its gate gradient on the device and its norm's input gradient into
    // d_res2; the world-model output and memory keys get no gradient (frozen inputs).
    if (gated_gpu_) {
        const int n = d_res2.rows * d_res2.cols;
        if (gated_gpu_->hm_active) {
            adai::gpu::GPUMatrix d_hm(d_res2.rows, d_res2.cols);
            adai::gpu::gated_residual_backward_gpu(d_res2.device_ptr(),
                                                   gated_gpu_->hm_attn->device_ptr(),
                                                   gated_gpu_->gate_h.device_ptr(),
                                                   d_hm.device_ptr(),
                                                   gated_gpu_->gate_h_grad.device_ptr(), n);
            auto [d_normed_h, d_keys_unused] = hippocampal_cross_attention->gpu_backward(d_hm);
            (void)d_keys_unused;
            d_res2 = d_res2 + norm_hippocampal->gpu_backward(d_normed_h);
        }
        if (gated_gpu_->wm_active) {
            adai::gpu::GPUMatrix d_wm(d_res2.rows, d_res2.cols);
            adai::gpu::gated_residual_backward_gpu(d_res2.device_ptr(),
                                                   gated_gpu_->wm_attn->device_ptr(),
                                                   gated_gpu_->gate.device_ptr(),
                                                   d_wm.device_ptr(),
                                                   gated_gpu_->gate_grad.device_ptr(), n);
            auto [d_normed_w, d_wm_unused] = world_model_cross_attention->gpu_backward(d_wm);
            (void)d_wm_unused;
            d_res2 = d_res2 + norm_world->gpu_backward(d_normed_w);
        }
    }

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
