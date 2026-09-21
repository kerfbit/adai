// @adai-status: beta        (capped by TD-050 — see TECHNICAL_DEBT.md; TD-038 LoRA support added; TD-050 GPU incremental-cache forward added; TD-195 backward() gradient members now accumulate, not overwrite, across calls)
// @adai-version: 0.12.1
// @adai-reviewed: 2026-09-18

#include "MultiHeadAttention.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "Activation.hpp"
#include "Optimizer.hpp"

#ifdef _OPENMP
#include <omp.h>
#include <cmath>
#endif

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : d_model(d_model),
      num_heads(num_heads),
      d_k(d_model / num_heads),
      W_q(d_model, d_model),
      W_k(d_model, d_model),
      W_v(d_model, d_model),
      W_o(d_model, d_model),
      W_q_grad(d_model, d_model),
      W_k_grad(d_model, d_model),
      W_v_grad(d_model, d_model),
      W_o_grad(d_model, d_model) {
    // Validate that d_model is divisible by num_heads
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model (" + std::to_string(d_model) +
                                    ") must be divisible by num_heads (" +
                                    std::to_string(num_heads) + ")");
    }

    // Xavier/He initialization for weight matrices
    // Scale factor based on the input dimension
    float scale = std::sqrt(2.0f / static_cast<float>(d_model));

    W_q.randomize(scale);
    W_k.randomize(scale);
    W_v.randomize(scale);
    W_o.randomize(scale);

    // Initialize gradients to zero
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            W_q_grad(i, j) = 0.0f;
            W_k_grad(i, j) = 0.0f;
            W_v_grad(i, j) = 0.0f;
            W_o_grad(i, j) = 0.0f;
        }
    }
}

// TD-059: shared helpers for slicing/scattering a head's [*, d_k] column range out of
// (or into) a full [*, d_model] matrix. Used by backward() to materialize per-head
// sub-matrices for Matrix's ordinary GEMM/transpose operators; forward_parallel() below
// indexes columns directly in its own raw loops instead (avoids constructing temporaries
// inside the OpenMP-parallelized per-head loop).
namespace {
Matrix slice_head_columns(const Matrix& m, int start, int width) {
    Matrix result(m.rows, width);
    for (int i = 0; i < m.rows; ++i) {
        for (int k = 0; k < width; ++k) {
            result(i, k) = m(i, start + k);
        }
    }
    return result;
}

void scatter_head_columns(Matrix& dst, int start, const Matrix& src) {
    for (int i = 0; i < src.rows; ++i) {
        for (int k = 0; k < src.cols; ++k) {
            dst(i, start + k) = src(i, k);
        }
    }
}
}  // namespace

Matrix MultiHeadAttention::forward(const Matrix& input, const Matrix* mask) {
    // TD-059: forward_parallel() holds the one real per-head implementation; forward()
    // just calls it with parallel head computation enabled.
    return forward_parallel(input, mask, true);
}

Matrix MultiHeadAttention::forward_parallel(const Matrix& input, const Matrix* mask,
                                            bool use_parallel) {
    // Cache input for backward pass
    cached_input = input;

    int seq_len = input.rows;

    // Validate input dimensions
    if (input.cols != d_model) {
        throw std::invalid_argument("Input dimension (" + std::to_string(input.cols) +
                                    ") must match d_model (" + std::to_string(d_model) + ")");
    }
    // TD-059: forward() used to validate mask dimensions before forward_parallel() existed
    // as its implementation; keep that check here now that forward() delegates to this
    // method, so an invalid mask still throws std::invalid_argument rather than the
    // std::out_of_range Matrix::operator() would throw once the per-head loop below
    // indexes past a too-small mask.
    if (mask != nullptr && (mask->rows != seq_len || mask->cols != seq_len)) {
        throw std::invalid_argument("Mask dimensions must match sequence length");
    }

    // Linear projections to get Q, K, V
    Matrix Q = input * W_q;
    Matrix K = input * W_k;
    Matrix V = input * W_v;

    // TD-038: each active LoRA adapter adds its own low-rank ΔW to the frozen base
    // projection's output; B starts at zero, so this is a no-op until the adapter is
    // actually trained (see enable_lora()'s own doc comment). Q/K/V below become "the" real
    // values used by every downstream computation in this method — the per-head math has no
    // idea whether they came from the base weight alone or base+LoRA.
    if (lora_q_) {
        Q = lora_q_->forward(input, Q);
    }
    if (lora_k_) {
        K = lora_k_->forward(input, K);
    }
    if (lora_v_) {
        V = lora_v_->forward(input, V);
    }

    cached_Q = Q;
    cached_K = K;
    cached_V = V;

    // Allocate output matrix once
    Matrix concatenated(seq_len, d_model);
    // TD-059: one real post-softmax weight matrix per head, cached for backward().
    std::vector<Matrix> head_weights(num_heads, Matrix(seq_len, seq_len));

    // Per-head scaled dot-product attention: Q_h, K_h, V_h are this head's own
    // [seq_len, d_k] column slice ([h*d_k, (h+1)*d_k)) of the full-width Q/K/V above —
    // this is the actual "split into num_heads" step the class-level doc describes.
    // Written as raw indexed loops (not Matrix::operator*) so each head's block below
    // can run in its own OpenMP thread without nesting into operator*'s own internal
    // #pragma omp parallel for.
    auto compute_head = [&](int h) {
        int start_dim = h * d_k;
        Matrix& scores_head = head_weights[h];

        // scores_head = Q_h * K_h^T
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < d_k; ++k) {
                    sum += Q(i, start_dim + k) * K(j, start_dim + k);
                }
                scores_head(i, j) = sum;
            }
        }

        // Scale by 1/sqrt(d_k) — correct for this width, since the contraction above is
        // genuinely over d_k dimensions now (TD-059's mismatched-scale half of the bug
        // only existed because the contraction used to be over the full d_model width).
        float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                scores_head(i, j) *= scale_factor;
            }
        }

        // Apply mask if provided (same mask shared across every head)
        if (mask != nullptr) {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    if ((*mask)(i, j) == 0.0f) {
                        scores_head(i, j) = -1e9f;
                    }
                }
            }
        }

        // Softmax row-wise, in place — scores_head now holds this head's real attention
        // weights (cached below for backward()).
        for (int i = 0; i < seq_len; ++i) {
            float max_score = scores_head(i, 0);
            for (int j = 1; j < seq_len; ++j) {
                max_score = std::max(max_score, scores_head(i, j));
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                scores_head(i, j) = std::exp(scores_head(i, j) - max_score);
                sum_exp += scores_head(i, j);
            }

            for (int j = 0; j < seq_len; ++j) {
                scores_head(i, j) /= sum_exp;
            }
        }

        // Apply attention to values: output_h = attention_weights_h * V_h, written
        // directly into this head's column slice of the concatenated output.
        for (int i = 0; i < seq_len; ++i) {
            for (int k = 0; k < d_k; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    sum += scores_head(i, j) * V(j, start_dim + k);
                }
                concatenated(i, start_dim + k) = sum;
            }
        }
    };

#ifdef _OPENMP
    if (use_parallel) {
#pragma omp parallel for schedule(static)
        for (int h = 0; h < num_heads; ++h) {
            compute_head(h);
        }
    } else {
        for (int h = 0; h < num_heads; ++h) {
            compute_head(h);
        }
    }
#else
    for (int h = 0; h < num_heads; ++h) {
        compute_head(h);
    }
#endif

    cached_head_weights_ = std::move(head_weights);
    cached_attention_output = concatenated;

    // cached_attention_weights: mean across heads, elementwise — a convex combination of
    // per-row probability distributions is itself a valid probability distribution (still
    // sums to 1, still non-negative, still ~0 at masked positions), so this stays a
    // faithful [seq_len, seq_len] summary for get_attention_weights() and the entropy
    // hook below, without claiming to be any single head's actual weights.
    Matrix avg_weights(seq_len, seq_len);
    float inv_num_heads = 1.0f / static_cast<float>(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        const Matrix& hw = cached_head_weights_[h];
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                avg_weights(i, j) += hw(i, j) * inv_num_heads;
            }
        }
    }
    cached_attention_weights = avg_weights;

    // Fire attention hook if registered (used for entropy tracking, TD-013). forward_parallel()
    // never fired this before (it was dead code with no caller); now that forward() delegates
    // here, preserve forward()'s existing hook contract.
    if (attention_hook_) {
        attention_hook_(cached_attention_weights);
    }

    // Final linear projection
    Matrix output = concatenated * W_o;
    if (lora_o_) {
        output = lora_o_->forward(concatenated, output);
    }

    return output;
}

Matrix MultiHeadAttention::forward_with_cache(const Matrix& input, const Matrix* mask,
                                              KVCache* kv_cache, bool use_cache) {
    // If no cache provided or cache disabled, fall back to regular forward
    if (!use_cache || kv_cache == nullptr) {
        return forward(input, mask);
    }

    // Cache input for backward pass (if needed for training)
    cached_input = input;

    int num_new_tokens = input.rows;

    // Validate input dimensions
    if (input.cols != d_model) {
        throw std::invalid_argument("Input dimension (" + std::to_string(input.cols) +
                                    ") must match d_model (" + std::to_string(d_model) + ")");
    }

    // Compute Q, K, V for NEW tokens only
    Matrix Q_new = input * W_q;
    Matrix K_new = input * W_k;
    Matrix V_new = input * W_v;

    // TD-038: same LoRA adapters as forward_parallel() above — Q_new/K_new/V_new below
    // become the real, LoRA-adjusted values cached into the KV cache and used downstream.
    if (lora_q_) {
        Q_new = lora_q_->forward(input, Q_new);
    }
    if (lora_k_) {
        K_new = lora_k_->forward(input, K_new);
    }
    if (lora_v_) {
        V_new = lora_v_->forward(input, V_new);
    }

    // Query is always from the new tokens
    cached_Q = Q_new;

    // Append new K, V to cache
    kv_cache->append(K_new, V_new);

    // Get full K, V from cache (includes all previous + new tokens)
    const Matrix& K_full = kv_cache->get_keys();
    const Matrix& V_full = kv_cache->get_values();

    cached_K = K_full;
    cached_V = V_full;

    int total_seq_len = K_full.rows;

    // Mask shape should be [num_new_tokens, total_seq_len], shared across every head
    if (mask != nullptr && (mask->rows != num_new_tokens || mask->cols != total_seq_len)) {
        throw std::invalid_argument(
            "Mask dimensions (" + std::to_string(mask->rows) + ", " + std::to_string(mask->cols) +
            ") must match [num_new_tokens=" + std::to_string(num_new_tokens) +
            ", total_seq_len=" + std::to_string(total_seq_len) + "]");
    }

    // TD-059: genuine per-head attention over the cached K/V, same as forward_parallel().
    // Q_new_h/K_full_h/V_full_h are this head's own [*, d_k] column slice.
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    Matrix concatenated(num_new_tokens, d_model);
    std::vector<Matrix> head_weights;
    head_weights.reserve(num_heads);

    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * d_k;
        Matrix Q_h = slice_head_columns(Q_new, start_dim, d_k);
        Matrix K_h = slice_head_columns(K_full, start_dim, d_k);
        Matrix V_h = slice_head_columns(V_full, start_dim, d_k);

        Matrix scores_h = Q_h * K_h.transpose();  // [num_new_tokens, total_seq_len]
        scores_h = scores_h.scale(scale_factor);

        if (mask != nullptr) {
            for (int i = 0; i < num_new_tokens; ++i) {
                for (int j = 0; j < total_seq_len; ++j) {
                    if ((*mask)(i, j) == 0.0f) {
                        scores_h(i, j) = -1e9f;
                    }
                }
            }
        }

        Matrix weights_h = Activation::softmax(scores_h);
        Matrix out_h = weights_h * V_h;  // [num_new_tokens, d_k]

        scatter_head_columns(concatenated, start_dim, out_h);
        head_weights.push_back(std::move(weights_h));
    }

    cached_head_weights_ = std::move(head_weights);
    cached_attention_output = concatenated;

    // Mean across heads for get_attention_weights() — see forward_parallel()'s identical
    // rationale. forward_with_cache() never fired attention_hook_ before this fix (only
    // forward() did); preserved as still not firing it here, matching this method's prior
    // observable behavior.
    Matrix avg_weights(num_new_tokens, total_seq_len);
    float inv_num_heads = 1.0f / static_cast<float>(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        const Matrix& hw = cached_head_weights_[h];
        for (int i = 0; i < num_new_tokens; ++i) {
            for (int j = 0; j < total_seq_len; ++j) {
                avg_weights(i, j) += hw(i, j) * inv_num_heads;
            }
        }
    }
    cached_attention_weights = avg_weights;

    // Final linear projection
    Matrix output = cached_attention_output * W_o;
    if (lora_o_) {
        output = lora_o_->forward(cached_attention_output, output);
    }

    return output;
}

Matrix MultiHeadAttention::backward(const Matrix& grad_output) {
    // Validate gradient dimensions
    if (grad_output.rows != cached_input.rows || grad_output.cols != d_model) {
        throw std::invalid_argument(
            "Gradient dimensions must match forward pass output dimensions");
    }

    // Gradient w.r.t. W_o
    // W_o_grad = attention_output^T * grad_output
    W_o_grad = W_o_grad + cached_attention_output.transpose() * grad_output;

    // Gradient w.r.t. attention output
    // grad_attn_out = grad_output * W_o^T
    Matrix grad_attn_out = grad_output * W_o.transpose();

    // TD-038: LoRA_o shares the same `concatenated` input as W_o (forward_parallel()'s
    // `output = concatenated*W_o [+ LoRA_o delta]`) — its own branch's gradient w.r.t. that
    // shared input adds directly onto grad_attn_out, same as any two branches feeding a sum.
    if (lora_o_) {
        Matrix grad_from_lora_o = lora_o_->backward(cached_attention_output, grad_output);
        for (int i = 0; i < grad_attn_out.rows; ++i) {
            for (int j = 0; j < grad_attn_out.cols; ++j) {
                grad_attn_out(i, j) += grad_from_lora_o(i, j);
            }
        }
    }

    // TD-059: differentiate through each head's own softmax separately (using the real
    // per-head weights cached_head_weights_[h] from the matching forward pass), instead of
    // one d_model-wide softmax gradient. Each head only touches its own [*, d_k] column
    // slice of Q/K/V, so dQ/dK/dV can be assembled by scattering each head's gradient into
    // the matching column range of a full-width zero matrix — no cross-head terms exist,
    // since a given column of Q/K/V feeds exactly one head.
    int q_rows = cached_Q.rows;
    int kv_rows = cached_K.rows;
    Matrix dQ(q_rows, d_model);
    Matrix dK(kv_rows, d_model);
    Matrix dV(kv_rows, d_model);
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));

    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * d_k;
        const Matrix& weights_h = cached_head_weights_[h];  // [q_rows, kv_rows]

        Matrix grad_out_h = slice_head_columns(grad_attn_out, start_dim, d_k);  // [q_rows, d_k]
        Matrix V_h = slice_head_columns(cached_V, start_dim, d_k);              // [kv_rows, d_k]

        // grad_V_h = weights_h^T * grad_out_h
        Matrix grad_V_h = weights_h.transpose() * grad_out_h;
        // grad_weights_h = grad_out_h * V_h^T
        Matrix grad_weights_h = grad_out_h * V_h.transpose();

        // Softmax backward, row-wise, for this head's own weights:
        // grad_scores_h(i,j) = weights_h(i,j) * (grad_weights_h(i,j) - sum_k weights_h(i,k)*grad_weights_h(i,k))
        Matrix grad_scores_h(q_rows, kv_rows);
        for (int i = 0; i < q_rows; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < kv_rows; ++k) {
                sum += weights_h(i, k) * grad_weights_h(i, k);
            }
            for (int j = 0; j < kv_rows; ++j) {
                grad_scores_h(i, j) = weights_h(i, j) * (grad_weights_h(i, j) - sum);
            }
        }
        grad_scores_h = grad_scores_h.scale(scale_factor);

        Matrix Q_h = slice_head_columns(cached_Q, start_dim, d_k);  // [q_rows, d_k]
        Matrix K_h = slice_head_columns(cached_K, start_dim, d_k);  // [kv_rows, d_k]

        Matrix grad_Q_h = grad_scores_h * K_h;                // [q_rows, d_k]
        Matrix grad_K_h = grad_scores_h.transpose() * Q_h;    // [kv_rows, d_k]

        scatter_head_columns(dQ, start_dim, grad_Q_h);
        scatter_head_columns(dK, start_dim, grad_K_h);
        scatter_head_columns(dV, start_dim, grad_V_h);
    }

    // Gradients w.r.t. projection weights
    // W_q_grad = input^T * dQ
    W_q_grad = W_q_grad + cached_input.transpose() * dQ;

    // W_k_grad = input^T * dK
    W_k_grad = W_k_grad + cached_input.transpose() * dK;

    // W_v_grad = input^T * dV
    W_v_grad = W_v_grad + cached_input.transpose() * dV;

    // Gradient w.r.t. input (sum gradients from all three projections)
    // grad_input = dQ * W_q^T + dK * W_k^T + dV * W_v^T
    Matrix grad_input = dQ * W_q.transpose();
    Matrix grad_from_K = dK * W_k.transpose();
    Matrix grad_from_V = dV * W_v.transpose();

    // Add gradients
    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            grad_input(i, j) += grad_from_K(i, j) + grad_from_V(i, j);
        }
    }

    // TD-038: LoRA_q/k/v each share the same `cached_input` as W_q/W_k/W_v — dQ/dK/dV above
    // are exactly the gradients w.r.t. their (post-adapter) outputs, the same "grad_output"
    // each adapter's own backward() needs; their branch's contribution to dL/d(cached_input)
    // adds directly onto grad_input, same rationale as the LoRA_o addition above.
    auto add_lora_grad = [&](LoRAAdapter* adapter, const Matrix& d_proj) {
        if (!adapter) {
            return;
        }
        Matrix g = adapter->backward(cached_input, d_proj);
        for (int i = 0; i < grad_input.rows; ++i) {
            for (int j = 0; j < grad_input.cols; ++j) {
                grad_input(i, j) += g(i, j);
            }
        }
    };
    add_lora_grad(lora_q_.get(), dQ);
    add_lora_grad(lora_k_.get(), dK);
    add_lora_grad(lora_v_.get(), dV);

    return grad_input;
}

void MultiHeadAttention::set_optimizer(Optimizer* opt) {
    optimizer = opt;
    if (optimizer) {
        register_parameters();
    }
}

void MultiHeadAttention::register_parameters() {
    if (!optimizer) {
        return;
    }

    // Register all weight matrices with the optimizer
    optimizer->add_parameter_group(&W_q, &W_q_grad);
    optimizer->add_parameter_group(&W_k, &W_k_grad);
    optimizer->add_parameter_group(&W_v, &W_v_grad);
    optimizer->add_parameter_group(&W_o, &W_o_grad);
}

void MultiHeadAttention::update_weights() {
    if (optimizer) {
        // Use optimizer for weight updates
        optimizer->step();
    } else {
        // Fallback to simple gradient descent
        W_q.apply_gradients(W_q_grad, learning_rate);
        W_k.apply_gradients(W_k_grad, learning_rate);
        W_v.apply_gradients(W_v_grad, learning_rate);
        W_o.apply_gradients(W_o_grad, learning_rate);
    }

    // Zero gradients after update
    zero_grad();
}

void MultiHeadAttention::zero_grad() {
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            W_q_grad(i, j) = 0.0f;
            W_k_grad(i, j) = 0.0f;
            W_v_grad(i, j) = 0.0f;
            W_o_grad(i, j) = 0.0f;
        }
    }
    // TD-038: also zero any active LoRA adapters' own gradients.
    if (lora_q_) {
        lora_q_->zero_grad();
    }
    if (lora_k_) {
        lora_k_->zero_grad();
    }
    if (lora_v_) {
        lora_v_->zero_grad();
    }
    if (lora_o_) {
        lora_o_->zero_grad();
    }
}

void MultiHeadAttention::enable_lora(const LoRAConfig& config) {
    // Square (d_model, d_model) for every projection here, since Q/K/V/O all map
    // d_model -> d_model. Each starts fresh (B=0), so re-calling this discards any
    // previously-trained adapters, per this method's own doc comment.
    lora_q_ = config.apply_to_query
                  ? std::make_unique<LoRAAdapter>(d_model, d_model, config.rank, config.alpha)
                  : nullptr;
    lora_k_ = config.apply_to_key
                  ? std::make_unique<LoRAAdapter>(d_model, d_model, config.rank, config.alpha)
                  : nullptr;
    lora_v_ = config.apply_to_value
                  ? std::make_unique<LoRAAdapter>(d_model, d_model, config.rank, config.alpha)
                  : nullptr;
    lora_o_ = config.apply_to_output
                  ? std::make_unique<LoRAAdapter>(d_model, d_model, config.rank, config.alpha)
                  : nullptr;
}

void MultiHeadAttention::register_lora_parameters(Optimizer& optimizer) {
    if (lora_q_) {
        lora_q_->register_parameters(optimizer);
    }
    if (lora_k_) {
        lora_k_->register_parameters(optimizer);
    }
    if (lora_v_) {
        lora_v_->register_parameters(optimizer);
    }
    if (lora_o_) {
        lora_o_->register_parameters(optimizer);
    }
}

void MultiHeadAttention::merge_lora() {
    if (lora_q_) {
        W_q = lora_q_->merge_with_base(W_q);
        lora_q_.reset();
    }
    if (lora_k_) {
        W_k = lora_k_->merge_with_base(W_k);
        lora_k_.reset();
    }
    if (lora_v_) {
        W_v = lora_v_->merge_with_base(W_v);
        lora_v_.reset();
    }
    if (lora_o_) {
        W_o = lora_o_->merge_with_base(W_o);
        lora_o_.reset();
    }
}

void MultiHeadAttention::print_config(const std::string& name) const {
    std::cout << name << " Configuration:" << '\n';
    std::cout << "  Model Dimension (d_model): " << d_model << '\n';
    std::cout << "  Number of Heads: " << num_heads << '\n';
    std::cout << "  Dimension per Head (d_k): " << d_k << '\n';
    std::cout << "  Total Parameters: " << (4 * d_model * d_model) << '\n';
    std::cout << "  Memory Usage: " << std::fixed << std::setprecision(2)
              << (static_cast<float>(4) * static_cast<float>(d_model) *
                  static_cast<float>(d_model) * sizeof(float)) /
                     1024.0f / 1024.0f
              << " MB" << '\n';
    std::cout << "  Learning Rate: " << learning_rate << '\n';
}

void MultiHeadAttention::save_weights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write dimensions
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));

    // Write weight matrices
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.write(reinterpret_cast<const char*>(&W_q(i, j)), sizeof(float));
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.write(reinterpret_cast<const char*>(&W_k(i, j)), sizeof(float));
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.write(reinterpret_cast<const char*>(&W_v(i, j)), sizeof(float));
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.write(reinterpret_cast<const char*>(&W_o(i, j)), sizeof(float));
        }
    }

    file.close();

    std::cout << "Saved MultiHeadAttention weights to " << filename << '\n';
}

void MultiHeadAttention::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read dimensions
    int saved_d_model = 0, saved_num_heads = 0;
    file.read(reinterpret_cast<char*>(&saved_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_num_heads), sizeof(int));

    // Verify dimensions match
    if (saved_d_model != d_model || saved_num_heads != num_heads) {
        throw std::runtime_error(
            "Dimension mismatch: file has d_model=" + std::to_string(saved_d_model) +
            ", num_heads=" + std::to_string(saved_num_heads) + ", expected d_model=" +
            std::to_string(d_model) + ", num_heads=" + std::to_string(num_heads));
    }

    // Read weight matrices
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.read(reinterpret_cast<char*>(&W_q(i, j)), sizeof(float));
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.read(reinterpret_cast<char*>(&W_k(i, j)), sizeof(float));
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.read(reinterpret_cast<char*>(&W_v(i, j)), sizeof(float));
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.read(reinterpret_cast<char*>(&W_o(i, j)), sizeof(float));
        }
    }

    file.close();

    std::cout << "Loaded MultiHeadAttention weights from " << filename << '\n';
}

float MultiHeadAttention::get_gradient_norm() const {
    float sum_squares = 0.0f;

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float grad = NAN;

            grad = W_q_grad(i, j);
            sum_squares += grad * grad;

            grad = W_k_grad(i, j);
            sum_squares += grad * grad;

            grad = W_v_grad(i, j);
            sum_squares += grad * grad;

            grad = W_o_grad(i, j);
            sum_squares += grad * grad;
        }
    }

    return std::sqrt(sum_squares);
}

void MultiHeadAttention::clip_gradients(float max_norm) {
    float norm = get_gradient_norm();

    if (norm > max_norm) {
        float scale = max_norm / norm;

        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_model; ++j) {
                W_q_grad(i, j) *= scale;
                W_k_grad(i, j) *= scale;
                W_v_grad(i, j) *= scale;
                W_o_grad(i, j) *= scale;
            }
        }
    }
}

#ifdef ADAI_ENABLE_GPU
static void upload_sq_matrix(const Matrix& m, adai::gpu::GPUMatrix& g) {
    int n = m.rows * m.cols;
    std::vector<float> tmp(n);
    int idx = 0;
    for (const auto& row : m.data)
        for (float v : row)
            tmp[idx++] = v;
    g.upload(tmp.data(), n);
}

void MultiHeadAttention::gpu_upload_weights() {
    if (!gpu_)
        gpu_ = std::make_unique<GPUState>(d_model);
    upload_sq_matrix(W_q, gpu_->Wq);
    upload_sq_matrix(W_k, gpu_->Wk);
    upload_sq_matrix(W_v, gpu_->Wv);
    upload_sq_matrix(W_o, gpu_->Wo);
}

void MultiHeadAttention::gpu_download_grads() {
    if (!gpu_)
        return;
    auto add_back = [&](const adai::gpu::GPUMatrix& src, Matrix& dst) {
        int n = dst.rows * dst.cols;
        std::vector<float> tmp(n);
        src.download(tmp.data(), n);
        int idx = 0;
        for (auto& row : dst.data)
            for (auto& v : row)
                v += tmp[idx++];
    };
    add_back(gpu_->dWq, W_q_grad);
    add_back(gpu_->dWk, W_k_grad);
    add_back(gpu_->dWv, W_v_grad);
    add_back(gpu_->dWo, W_o_grad);
}

void MultiHeadAttention::gpu_zero_grads() {
    if (!gpu_)
        return;
    gpu_->dWq.zero();
    gpu_->dWk.zero();
    gpu_->dWv.zero();
    gpu_->dWo.zero();
}

namespace {
// TD-059: GPU-side equivalents of slice_head_columns()/scatter_head_columns() above, built
// entirely from matrix_copy_device_to_device_gpu() — an existing, already-used primitive
// (see gpu_upload_weights()'s cached_input copy above) — with pointer-offset device-to-device
// copies. No new CUDA/SYCL kernels: correctness here rests on primitives already exercised
// elsewhere in this file, which matters because this environment can compile but not
// runtime-verify either GPU backend (no physical GPU device available) — see TD-059's
// writeup in TECHNICAL_DEBT_RESOLVED.md for that residual verification gap.
adai::gpu::GPUMatrix gpu_slice_head_columns(const adai::gpu::GPUMatrix& m, int start, int width) {
    adai::gpu::GPUMatrix result(m.rows, width);
    for (int i = 0; i < m.rows; ++i) {
        adai::gpu::matrix_copy_device_to_device_gpu(m.device_ptr() + i * m.cols + start,
                                                     result.device_ptr() + i * width, width);
    }
    return result;
}

void gpu_scatter_head_columns(adai::gpu::GPUMatrix& dst, int start,
                              const adai::gpu::GPUMatrix& src) {
    for (int i = 0; i < src.rows; ++i) {
        adai::gpu::matrix_copy_device_to_device_gpu(
            src.device_ptr() + i * src.cols, dst.device_ptr() + i * dst.cols + start, src.cols);
    }
}
}  // namespace

adai::gpu::GPUMatrix MultiHeadAttention::gpu_forward(const adai::gpu::GPUMatrix& input,
                                                     const adai::gpu::GPUMatrix* mask) {
    if (!gpu_)
        gpu_upload_weights();
    const int seq = input.rows;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Resize caches if seq changed. Must check .cols too, not just .rows -- see
    // FeedForward::gpu_forward()'s identical fix for the confirmed-crashing instance of this
    // same bug: GPUState's constructor sentinel-initializes these to (1, 1), and a rows-only
    // check coincidentally matches whenever seq == 1. This particular call site (plain
    // gpu_forward(), not gpu_forward_with_cache()) isn't reached by the GPU-resident decode
    // path today, so it hasn't actually crashed yet, but the same collision would trigger it
    // the same way for any single-token gpu_forward() call (e.g. training on a length-1
    // sequence) -- fixed proactively rather than waiting for that to happen too.
    if (gpu_->cached_input.rows != seq || gpu_->cached_input.cols != d_model) {
        gpu_->cached_input = adai::gpu::GPUMatrix(seq, d_model);
        gpu_->cached_Q = adai::gpu::GPUMatrix(seq, d_model);
        gpu_->cached_K = adai::gpu::GPUMatrix(seq, d_model);
        gpu_->cached_V = adai::gpu::GPUMatrix(seq, d_model);
        gpu_->cached_attn_out = adai::gpu::GPUMatrix(seq, d_model);
        gpu_->cached_head_weights.clear();
        gpu_->cached_head_weights.reserve(num_heads);
        for (int h = 0; h < num_heads; ++h) {
            gpu_->cached_head_weights.emplace_back(seq, seq);
        }
    }
    adai::gpu::matrix_copy_device_to_device_gpu(input.device_ptr(), gpu_->cached_input.device_ptr(),
                                                seq * d_model);

    // Q, K, V projections
    gpu_->cached_Q = input * gpu_->Wq;
    gpu_->cached_K = input * gpu_->Wk;
    gpu_->cached_V = input * gpu_->Wv;

    // TD-059: genuine per-head attention — each head operates on its own [seq, d_k] column
    // slice of Q/K/V, mirroring forward_parallel()'s CPU implementation exactly (same formula,
    // same scale, same shared mask per head).
    adai::gpu::GPUMatrix concatenated(seq, d_model);
    float total_entropy = 0.0f;
    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * d_k;
        adai::gpu::GPUMatrix Q_h = gpu_slice_head_columns(gpu_->cached_Q, start_dim, d_k);
        adai::gpu::GPUMatrix K_h = gpu_slice_head_columns(gpu_->cached_K, start_dim, d_k);
        adai::gpu::GPUMatrix V_h = gpu_slice_head_columns(gpu_->cached_V, start_dim, d_k);

        adai::gpu::GPUMatrix scores_h = Q_h * K_h.transpose();
        scores_h = scores_h.scale(scale);
        if (mask != nullptr) {
            scores_h.masked_fill_inplace(*mask, -1e9f);
        }
        scores_h.softmax_rows_inplace();

        if (gpu_attention_stats_hook_) {
            total_entropy += scores_h.row_entropy_avg();
        }

        adai::gpu::GPUMatrix out_h = scores_h * V_h;
        gpu_scatter_head_columns(concatenated, start_dim, out_h);
        gpu_->cached_head_weights[h] = std::move(scores_h);
    }

    // Average entropy across heads — see forward_parallel()'s identical rationale for why an
    // average is the right representative single-scalar summary here.
    if (gpu_attention_stats_hook_) {
        gpu_attention_stats_hook_(total_entropy / static_cast<float>(num_heads));
    }

    gpu_->cached_attn_out = std::move(concatenated);

    // output = attn_out * W_o
    return gpu_->cached_attn_out * gpu_->Wo;
}

adai::gpu::GPUMatrix MultiHeadAttention::gpu_forward_with_cache(const adai::gpu::GPUMatrix& input,
                                                                const adai::gpu::GPUMatrix* mask,
                                                                adai::gpu::GPUKVCache* kv_cache,
                                                                bool use_cache) {
    if (!use_cache || kv_cache == nullptr) {
        return gpu_forward(input, mask);
    }
    if (!gpu_)
        gpu_upload_weights();

    const int num_new = input.rows;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Q/K/V for the new token(s) only — mirrors forward_with_cache()'s CPU algorithm exactly.
    adai::gpu::GPUMatrix Q_new = input * gpu_->Wq;
    adai::gpu::GPUMatrix K_new = input * gpu_->Wk;
    adai::gpu::GPUMatrix V_new = input * gpu_->Wv;

    kv_cache->append(K_new, V_new);
    adai::gpu::GPUMatrix K_full = kv_cache->get_keys();    // [total_seq_len, d_model]
    adai::gpu::GPUMatrix V_full = kv_cache->get_values();  // [total_seq_len, d_model]
    const int total_seq_len = K_full.rows;

    // TD-059's own per-head primitives, unmodified — the only difference from gpu_forward()'s
    // loop is that K_h/V_h come from the cache (width total_seq_len) instead of from this same
    // call's own Q/K/V (width num_new). Attention shape is [num_new, total_seq_len], exactly
    // matching forward_with_cache()'s CPU shape.
    adai::gpu::GPUMatrix concatenated(num_new, d_model);
    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * d_k;
        adai::gpu::GPUMatrix Q_h = gpu_slice_head_columns(Q_new, start_dim, d_k);
        adai::gpu::GPUMatrix K_h = gpu_slice_head_columns(K_full, start_dim, d_k);
        adai::gpu::GPUMatrix V_h = gpu_slice_head_columns(V_full, start_dim, d_k);

        adai::gpu::GPUMatrix scores_h = Q_h * K_h.transpose();  // [num_new, total_seq_len]
        scores_h = scores_h.scale(scale);
        if (mask != nullptr) {
            scores_h.masked_fill_inplace(*mask, -1e9f);
        }
        scores_h.softmax_rows_inplace();

        adai::gpu::GPUMatrix out_h = scores_h * V_h;  // [num_new, d_k]
        gpu_scatter_head_columns(concatenated, start_dim, out_h);
    }

    return concatenated * gpu_->Wo;
}

adai::gpu::GPUMatrix MultiHeadAttention::gpu_backward(const adai::gpu::GPUMatrix& dout) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // dW_o += attn_out^T * dout
    gpu_->dWo.add_inplace(gpu_->cached_attn_out.transpose() * dout);

    // d_attn_out = dout * W_o^T
    adai::gpu::GPUMatrix d_attn_out = dout * gpu_->Wo.transpose();

    // TD-059: per-head backward, symmetric with the CPU implementation in backward() above —
    // see that function's comment for why no cross-head terms exist.
    const int seq = gpu_->cached_Q.rows;
    adai::gpu::GPUMatrix dQ(seq, d_model);
    adai::gpu::GPUMatrix dK(seq, d_model);
    adai::gpu::GPUMatrix dV(seq, d_model);

    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * d_k;
        const adai::gpu::GPUMatrix& weights_h = gpu_->cached_head_weights[h];

        adai::gpu::GPUMatrix d_out_h = gpu_slice_head_columns(d_attn_out, start_dim, d_k);
        adai::gpu::GPUMatrix V_h = gpu_slice_head_columns(gpu_->cached_V, start_dim, d_k);

        // dV_h = weights_h^T * d_out_h
        adai::gpu::GPUMatrix dV_h = weights_h.transpose() * d_out_h;
        // d_weights_h = d_out_h * V_h^T
        adai::gpu::GPUMatrix d_weights_h = d_out_h * V_h.transpose();

        // Softmax backward → d_scores_h, for this head's own weights only
        adai::gpu::GPUMatrix d_scores_h = weights_h.softmax_backward(d_weights_h);
        d_scores_h = d_scores_h.scale(scale);

        adai::gpu::GPUMatrix Q_h = gpu_slice_head_columns(gpu_->cached_Q, start_dim, d_k);
        adai::gpu::GPUMatrix K_h = gpu_slice_head_columns(gpu_->cached_K, start_dim, d_k);

        // dQ_h = d_scores_h * K_h,  dK_h = d_scores_h^T * Q_h
        adai::gpu::GPUMatrix dQ_h = d_scores_h * K_h;
        adai::gpu::GPUMatrix dK_h = d_scores_h.transpose() * Q_h;

        gpu_scatter_head_columns(dQ, start_dim, dQ_h);
        gpu_scatter_head_columns(dK, start_dim, dK_h);
        gpu_scatter_head_columns(dV, start_dim, dV_h);
    }

    // Weight grads
    adai::gpu::GPUMatrix in_T = gpu_->cached_input.transpose();
    gpu_->dWq.add_inplace(in_T * dQ);
    gpu_->dWk.add_inplace(in_T * dK);
    gpu_->dWv.add_inplace(in_T * dV);

    // d_input = dQ*Wq^T + dK*Wk^T + dV*Wv^T
    adai::gpu::GPUMatrix d_input = dQ * gpu_->Wq.transpose();
    d_input.add_inplace(dK * gpu_->Wk.transpose());
    d_input.add_inplace(dV * gpu_->Wv.transpose());
    return d_input;
}
#endif
