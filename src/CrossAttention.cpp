// @adai-status: beta        (capped by TD-050 — see TECHNICAL_DEBT.md)
// @adai-version: 0.10.0
// @adai-reviewed: 2026-09-13

#include "CrossAttention.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "Activation.hpp"

CrossAttention::CrossAttention(int d_model, int num_heads)
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

// TD-059: shared helpers for slicing/scattering a head's [*, d_k] column range out of (or
// into) a full [*, d_model] matrix — same rationale as MultiHeadAttention.cpp's identical
// helpers (kept as a separate copy here rather than shared, since Matrix has no existing
// shared-utility translation unit and this is the only other user).
namespace {
Matrix ca_slice_head_columns(const Matrix& m, int start, int width) {
    Matrix result(m.rows, width);
    for (int i = 0; i < m.rows; ++i) {
        for (int k = 0; k < width; ++k) {
            result(i, k) = m(i, start + k);
        }
    }
    return result;
}

void ca_scatter_head_columns(Matrix& dst, int start, const Matrix& src) {
    for (int i = 0; i < src.rows; ++i) {
        for (int k = 0; k < src.cols; ++k) {
            dst(i, start + k) = src(i, k);
        }
    }
}
}  // namespace

Matrix CrossAttention::forward(const Matrix& query_input, const Matrix& kv_input,
                               const Matrix* mask) {
    // Cache inputs for backward pass
    cached_query_input = query_input;
    cached_kv_input = kv_input;

    int tgt_len = query_input.rows;
    int src_len = kv_input.rows;

    // Validate dimensions
    if (query_input.cols != d_model) {
        throw std::invalid_argument("Query input dimension (" + std::to_string(query_input.cols) +
                                    ") must match d_model (" + std::to_string(d_model) + ")");
    }
    if (kv_input.cols != d_model) {
        throw std::invalid_argument("Key-Value input dimension (" + std::to_string(kv_input.cols) +
                                    ") must match d_model (" + std::to_string(d_model) + ")");
    }
    if (mask != nullptr && (mask->rows != tgt_len || mask->cols != src_len)) {
        throw std::invalid_argument(
            "Mask dimensions (" + std::to_string(mask->rows) + ", " + std::to_string(mask->cols) +
            ") must match attention dimensions (" + std::to_string(tgt_len) + ", " +
            std::to_string(src_len) + ")");
    }

    // Project to Q, K, V
    cached_Q = query_input * W_q;  // [tgt_len, d_model]
    cached_K = kv_input * W_k;     // [src_len, d_model]
    cached_V = kv_input * W_v;     // [src_len, d_model]

    // TD-059: genuine per-head cross-attention — Q_h/K_h/V_h are this head's own [*, d_k]
    // column slice ([h*d_k, (h+1)*d_k)); same mask shared across every head.
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    Matrix concatenated(tgt_len, d_model);
    std::vector<Matrix> head_weights;
    head_weights.reserve(num_heads);

    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * d_k;
        Matrix Q_h = ca_slice_head_columns(cached_Q, start_dim, d_k);
        Matrix K_h = ca_slice_head_columns(cached_K, start_dim, d_k);
        Matrix V_h = ca_slice_head_columns(cached_V, start_dim, d_k);

        Matrix scores_h = Q_h * K_h.transpose();  // [tgt_len, src_len]
        scores_h = scores_h.scale(scale_factor);

        if (mask != nullptr) {
            for (int i = 0; i < tgt_len; ++i) {
                for (int j = 0; j < src_len; ++j) {
                    if ((*mask)(i, j) == 0.0f) {
                        scores_h(i, j) = -1e9f;
                    }
                }
            }
        }

        Matrix weights_h = Activation::softmax(scores_h);
        Matrix out_h = weights_h * V_h;  // [tgt_len, d_k]

        ca_scatter_head_columns(concatenated, start_dim, out_h);
        head_weights.push_back(std::move(weights_h));
    }

    cached_head_weights_ = std::move(head_weights);
    cached_attention_output = concatenated;

    // Mean across heads, elementwise, for callers that just want a [tgt_len, src_len]
    // summary — still a valid probability distribution (see MultiHeadAttention.cpp's
    // identical rationale). backward() uses cached_head_weights_ directly, not this.
    Matrix avg_weights(tgt_len, src_len);
    float inv_num_heads = 1.0f / static_cast<float>(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        const Matrix& hw = cached_head_weights_[h];
        for (int i = 0; i < tgt_len; ++i) {
            for (int j = 0; j < src_len; ++j) {
                avg_weights(i, j) += hw(i, j) * inv_num_heads;
            }
        }
    }
    cached_attention_weights = avg_weights;

    // Apply output projection
    Matrix output = concatenated * W_o;

    return output;
}

Matrix CrossAttention::forward_with_cache(const Matrix& query_input, const Matrix& kv_input,
                                          const Matrix* mask, KVCache* kv_cache, bool use_cache) {
    // If no cache or caching disabled, use regular forward
    if (!use_cache || kv_cache == nullptr) {
        return forward(query_input, kv_input, mask);
    }

    // Cache query input for backward pass (if needed)
    cached_query_input = query_input;

    int num_new_tokens = query_input.rows;

    // Validate dimensions
    if (query_input.cols != d_model) {
        throw std::invalid_argument("Query input dimension (" + std::to_string(query_input.cols) +
                                    ") must match d_model (" + std::to_string(d_model) + ")");
    }

    // Project queries (always from new decoder tokens)
    cached_Q = query_input * W_q;  // [num_new_tokens, d_model]

    // For cross-attention, K and V from encoder are constant across all generation steps
    // Compute and cache them only once (on first call when cache is empty)
    if (kv_cache->is_empty()) {
        // First call: compute K, V from encoder and cache them
        if (kv_input.cols != d_model) {
            throw std::invalid_argument("Key-Value input dimension (" +
                                        std::to_string(kv_input.cols) + ") must match d_model (" +
                                        std::to_string(d_model) + ")");
        }

        cached_kv_input = kv_input;

        Matrix K_encoder = kv_input * W_k;  // [src_len, d_model]
        Matrix V_encoder = kv_input * W_v;  // [src_len, d_model]

        // Initialize cache with encoder K/V (these remain constant)
        kv_cache->append(K_encoder, V_encoder);
    }

    // Retrieve cached K, V from encoder
    const Matrix& K_full = kv_cache->get_keys();
    const Matrix& V_full = kv_cache->get_values();

    cached_K = K_full;
    cached_V = V_full;

    int src_len = K_full.rows;

    // Mask shape: [num_new_tokens, src_len], shared across every head
    if (mask != nullptr && (mask->rows != num_new_tokens || mask->cols != src_len)) {
        throw std::invalid_argument(
            "Mask dimensions (" + std::to_string(mask->rows) + ", " + std::to_string(mask->cols) +
            ") must match [num_new_tokens=" + std::to_string(num_new_tokens) +
            ", src_len=" + std::to_string(src_len) + "]");
    }

    // TD-059: genuine per-head cross-attention over the cached encoder K/V, same as forward().
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    Matrix concatenated(num_new_tokens, d_model);
    std::vector<Matrix> head_weights;
    head_weights.reserve(num_heads);

    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * d_k;
        Matrix Q_h = ca_slice_head_columns(cached_Q, start_dim, d_k);
        Matrix K_h = ca_slice_head_columns(K_full, start_dim, d_k);
        Matrix V_h = ca_slice_head_columns(V_full, start_dim, d_k);

        Matrix scores_h = Q_h * K_h.transpose();  // [num_new_tokens, src_len]
        scores_h = scores_h.scale(scale_factor);

        if (mask != nullptr) {
            for (int i = 0; i < num_new_tokens; ++i) {
                for (int j = 0; j < src_len; ++j) {
                    if ((*mask)(i, j) == 0.0f) {
                        scores_h(i, j) = -1e9f;
                    }
                }
            }
        }

        Matrix weights_h = Activation::softmax(scores_h);
        Matrix out_h = weights_h * V_h;  // [num_new_tokens, d_k]

        ca_scatter_head_columns(concatenated, start_dim, out_h);
        head_weights.push_back(std::move(weights_h));
    }

    cached_head_weights_ = std::move(head_weights);
    cached_attention_output = concatenated;

    Matrix avg_weights(num_new_tokens, src_len);
    float inv_num_heads = 1.0f / static_cast<float>(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        const Matrix& hw = cached_head_weights_[h];
        for (int i = 0; i < num_new_tokens; ++i) {
            for (int j = 0; j < src_len; ++j) {
                avg_weights(i, j) += hw(i, j) * inv_num_heads;
            }
        }
    }
    cached_attention_weights = avg_weights;

    // Output projection
    Matrix output = concatenated * W_o;

    return output;
}

void CrossAttention::backward(const Matrix& grad_output, Matrix& grad_query_input,
                              Matrix& grad_kv_input) {
    // Gradient through output projection
    Matrix grad_attention_output = grad_output * W_o.transpose();
    W_o_grad = W_o_grad + (cached_attention_output.transpose() * grad_output);

    // TD-059: differentiate through each head's own softmax separately, using the real
    // per-head weights cached_head_weights_[h] — see MultiHeadAttention::backward()'s
    // identical rationale for why no cross-head terms exist. Sized from cached_Q/cached_K
    // (not cached_query_input/cached_kv_input) so this also does the right thing shape-wise
    // if ever called after forward_with_cache(), where cached_kv_input only holds the
    // encoder input from the first cache-populating call, not every subsequent K/V source.
    int tgt_len = cached_Q.rows;
    int src_len = cached_K.rows;
    Matrix dQ(tgt_len, d_model);
    Matrix dK(src_len, d_model);
    Matrix dV(src_len, d_model);
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));

    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * d_k;
        const Matrix& weights_h = cached_head_weights_[h];  // [tgt_len, src_len]

        Matrix grad_out_h = ca_slice_head_columns(grad_attention_output, start_dim, d_k);
        Matrix V_h = ca_slice_head_columns(cached_V, start_dim, d_k);

        // grad_V_h = weights_h^T * grad_out_h
        Matrix grad_V_h = weights_h.transpose() * grad_out_h;
        // grad_weights_h = grad_out_h * V_h^T
        Matrix grad_weights_h = grad_out_h * V_h.transpose();

        // Softmax backward, row-wise, for this head's own weights
        Matrix grad_scores_h(tgt_len, src_len);
        for (int i = 0; i < tgt_len; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < src_len; ++k) {
                sum += weights_h(i, k) * grad_weights_h(i, k);
            }
            for (int j = 0; j < src_len; ++j) {
                grad_scores_h(i, j) = weights_h(i, j) * (grad_weights_h(i, j) - sum);
            }
        }
        grad_scores_h = grad_scores_h.scale(scale_factor);

        Matrix Q_h = ca_slice_head_columns(cached_Q, start_dim, d_k);
        Matrix K_h = ca_slice_head_columns(cached_K, start_dim, d_k);

        Matrix grad_Q_h = grad_scores_h * K_h;              // [tgt_len, d_k]
        Matrix grad_K_h = grad_scores_h.transpose() * Q_h;  // [src_len, d_k]

        ca_scatter_head_columns(dQ, start_dim, grad_Q_h);
        ca_scatter_head_columns(dK, start_dim, grad_K_h);
        ca_scatter_head_columns(dV, start_dim, grad_V_h);
    }

    // Gradient through projections
    grad_query_input = dQ * W_q.transpose();
    Matrix grad_kv_from_K = dK * W_k.transpose();
    Matrix grad_kv_from_V = dV * W_v.transpose();
    grad_kv_input = grad_kv_from_K + grad_kv_from_V;

    // Accumulate weight gradients
    W_q_grad = W_q_grad + (cached_query_input.transpose() * dQ);
    W_k_grad = W_k_grad + (cached_kv_input.transpose() * dK);
    W_v_grad = W_v_grad + (cached_kv_input.transpose() * dV);
}

void CrossAttention::set_optimizer(Optimizer* opt) {
    optimizer = opt;
    if (optimizer) {
        register_parameters();
    }
}

void CrossAttention::register_parameters() {
    if (!optimizer) {
        return;
    }

    // Register all four weight matrices with optimizer
    optimizer->add_parameter_group(&W_q, &W_q_grad);
    optimizer->add_parameter_group(&W_k, &W_k_grad);
    optimizer->add_parameter_group(&W_v, &W_v_grad);
    optimizer->add_parameter_group(&W_o, &W_o_grad);
}

void CrossAttention::update_weights() {
    if (optimizer) {
        // Use advanced optimization (Adam, AdamW, etc.)
        optimizer->step();
    } else {
        // Fallback to simple gradient descent for backward compatibility
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_model; ++j) {
                W_q(i, j) -= learning_rate * W_q_grad(i, j);
                W_k(i, j) -= learning_rate * W_k_grad(i, j);
                W_v(i, j) -= learning_rate * W_v_grad(i, j);
                W_o(i, j) -= learning_rate * W_o_grad(i, j);
            }
        }
    }

    // Zero gradients after update
    zero_grad();
}

void CrossAttention::zero_grad() {
    // Zero out all gradient matrices
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            W_q_grad(i, j) = 0.0f;
            W_k_grad(i, j) = 0.0f;
            W_v_grad(i, j) = 0.0f;
            W_o_grad(i, j) = 0.0f;
        }
    }
}

float CrossAttention::get_gradient_norm() const {
    float sum_squares = 0.0f;

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float grad = W_q_grad(i, j);
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

void CrossAttention::save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    // Save hyperparameters
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
    file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(float));

    // Save weight matrices (manually write matrix data)
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float val = W_q(i, j);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float val = W_k(i, j);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float val = W_v(i, j);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float val = W_o(i, j);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }

    file.close();
}

void CrossAttention::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    // Load hyperparameters
    int loaded_d_model = 0, loaded_num_heads = 0;
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&learning_rate), sizeof(float));

    // Validate dimensions
    if (loaded_d_model != d_model || loaded_num_heads != num_heads) {
        throw std::runtime_error("Dimension mismatch: expected d_model=" + std::to_string(d_model) +
                                 ", num_heads=" + std::to_string(num_heads) +
                                 " but got d_model=" + std::to_string(loaded_d_model) +
                                 ", num_heads=" + std::to_string(loaded_num_heads));
    }

    // Load weight matrices (manually read matrix data)
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float val = NAN;
            file.read(reinterpret_cast<char*>(&val), sizeof(float));
            W_q(i, j) = val;
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float val = NAN;
            file.read(reinterpret_cast<char*>(&val), sizeof(float));
            W_k(i, j) = val;
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float val = NAN;
            file.read(reinterpret_cast<char*>(&val), sizeof(float));
            W_v(i, j) = val;
        }
    }

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float val = NAN;
            file.read(reinterpret_cast<char*>(&val), sizeof(float));
            W_o(i, j) = val;
        }
    }

    file.close();
}

#ifdef ADAI_ENABLE_GPU
static void ca_upload_sq(const Matrix& m, adai::gpu::GPUMatrix& g) {
    int n = m.rows * m.cols;
    std::vector<float> tmp;
    tmp.reserve(n);
    for (const auto& row : m.data)
        for (float v : row)
            tmp.push_back(v);
    g.upload(tmp.data(), n);
}

void CrossAttention::gpu_upload_weights() {
    if (!gpu_)
        gpu_ = std::make_unique<GPUState>(d_model);
    ca_upload_sq(W_q, gpu_->Wq);
    ca_upload_sq(W_k, gpu_->Wk);
    ca_upload_sq(W_v, gpu_->Wv);
    ca_upload_sq(W_o, gpu_->Wo);
}

void CrossAttention::gpu_download_grads() {
    if (!gpu_)
        return;
    auto add_back = [](const adai::gpu::GPUMatrix& src, Matrix& dst) {
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

void CrossAttention::gpu_zero_grads() {
    if (!gpu_)
        return;
    gpu_->dWq.zero();
    gpu_->dWk.zero();
    gpu_->dWv.zero();
    gpu_->dWo.zero();
}

namespace {
// TD-059: GPU-side equivalents of ca_slice_head_columns()/ca_scatter_head_columns() above —
// see MultiHeadAttention.cpp's identical helpers for the full rationale (built entirely from
// the already-used matrix_copy_device_to_device_gpu() primitive, no new CUDA/SYCL kernels).
adai::gpu::GPUMatrix ca_gpu_slice_head_columns(const adai::gpu::GPUMatrix& m, int start,
                                               int width) {
    adai::gpu::GPUMatrix result(m.rows, width);
    for (int i = 0; i < m.rows; ++i) {
        adai::gpu::matrix_copy_device_to_device_gpu(m.device_ptr() + i * m.cols + start,
                                                     result.device_ptr() + i * width, width);
    }
    return result;
}

void ca_gpu_scatter_head_columns(adai::gpu::GPUMatrix& dst, int start,
                                 const adai::gpu::GPUMatrix& src) {
    for (int i = 0; i < src.rows; ++i) {
        adai::gpu::matrix_copy_device_to_device_gpu(
            src.device_ptr() + i * src.cols, dst.device_ptr() + i * dst.cols + start, src.cols);
    }
}
}  // namespace

adai::gpu::GPUMatrix CrossAttention::gpu_forward(const adai::gpu::GPUMatrix& query,
                                                 const adai::gpu::GPUMatrix& kv,
                                                 const adai::gpu::GPUMatrix* mask) {
    if (!gpu_)
        gpu_upload_weights();
    const int tgt = query.rows;
    const int src = kv.rows;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    if (gpu_->cached_query.rows != tgt || gpu_->cached_query.cols != d_model)
        gpu_->cached_query = adai::gpu::GPUMatrix(tgt, d_model);
    if (gpu_->cached_kv.rows != src || gpu_->cached_kv.cols != d_model)
        gpu_->cached_kv = adai::gpu::GPUMatrix(src, d_model);
    adai::gpu::matrix_copy_device_to_device_gpu(query.device_ptr(), gpu_->cached_query.device_ptr(),
                                                tgt * d_model);
    adai::gpu::matrix_copy_device_to_device_gpu(kv.device_ptr(), gpu_->cached_kv.device_ptr(),
                                                src * d_model);

    gpu_->cached_Q = query * gpu_->Wq;
    gpu_->cached_K = kv * gpu_->Wk;
    gpu_->cached_V = kv * gpu_->Wv;

    // TD-059: genuine per-head cross-attention — mirrors CrossAttention::forward()'s CPU
    // implementation exactly (same formula, same scale, same shared mask per head).
    if (gpu_->cached_head_weights.size() != static_cast<size_t>(num_heads) ||
        (num_heads > 0 &&
         (gpu_->cached_head_weights[0].rows != tgt || gpu_->cached_head_weights[0].cols != src))) {
        gpu_->cached_head_weights.clear();
        gpu_->cached_head_weights.reserve(num_heads);
        for (int h = 0; h < num_heads; ++h) {
            gpu_->cached_head_weights.emplace_back(tgt, src);
        }
    }

    adai::gpu::GPUMatrix concatenated(tgt, d_model);
    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * d_k;
        adai::gpu::GPUMatrix Q_h = ca_gpu_slice_head_columns(gpu_->cached_Q, start_dim, d_k);
        adai::gpu::GPUMatrix K_h = ca_gpu_slice_head_columns(gpu_->cached_K, start_dim, d_k);
        adai::gpu::GPUMatrix V_h = ca_gpu_slice_head_columns(gpu_->cached_V, start_dim, d_k);

        adai::gpu::GPUMatrix scores_h = Q_h * K_h.transpose();
        scores_h = scores_h.scale(scale);
        if (mask != nullptr) {
            scores_h.masked_fill_inplace(*mask, -1e9f);
        }
        scores_h.softmax_rows_inplace();

        adai::gpu::GPUMatrix out_h = scores_h * V_h;
        ca_gpu_scatter_head_columns(concatenated, start_dim, out_h);
        gpu_->cached_head_weights[h] = std::move(scores_h);
    }

    gpu_->cached_attn_out = std::move(concatenated);
    return gpu_->cached_attn_out * gpu_->Wo;
}

std::pair<adai::gpu::GPUMatrix, adai::gpu::GPUMatrix> CrossAttention::gpu_backward(
    const adai::gpu::GPUMatrix& dout) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    gpu_->dWo.add_inplace(gpu_->cached_attn_out.transpose() * dout);
    adai::gpu::GPUMatrix d_ao = dout * gpu_->Wo.transpose();

    // TD-059: per-head backward, symmetric with CrossAttention::backward()'s CPU
    // implementation above.
    const int tgt = gpu_->cached_Q.rows;
    const int src = gpu_->cached_K.rows;
    adai::gpu::GPUMatrix dQ(tgt, d_model);
    adai::gpu::GPUMatrix dK(src, d_model);
    adai::gpu::GPUMatrix dV(src, d_model);

    for (int h = 0; h < num_heads; ++h) {
        int start_dim = h * d_k;
        const adai::gpu::GPUMatrix& weights_h = gpu_->cached_head_weights[h];

        adai::gpu::GPUMatrix d_out_h = ca_gpu_slice_head_columns(d_ao, start_dim, d_k);
        adai::gpu::GPUMatrix V_h = ca_gpu_slice_head_columns(gpu_->cached_V, start_dim, d_k);

        adai::gpu::GPUMatrix dV_h = weights_h.transpose() * d_out_h;
        adai::gpu::GPUMatrix d_weights_h = d_out_h * V_h.transpose();

        adai::gpu::GPUMatrix d_scores_h = weights_h.softmax_backward(d_weights_h);
        d_scores_h = d_scores_h.scale(scale);

        adai::gpu::GPUMatrix Q_h = ca_gpu_slice_head_columns(gpu_->cached_Q, start_dim, d_k);
        adai::gpu::GPUMatrix K_h = ca_gpu_slice_head_columns(gpu_->cached_K, start_dim, d_k);

        adai::gpu::GPUMatrix dQ_h = d_scores_h * K_h;
        adai::gpu::GPUMatrix dK_h = d_scores_h.transpose() * Q_h;

        ca_gpu_scatter_head_columns(dQ, start_dim, dQ_h);
        ca_gpu_scatter_head_columns(dK, start_dim, dK_h);
        ca_gpu_scatter_head_columns(dV, start_dim, dV_h);
    }

    gpu_->dWq.add_inplace(gpu_->cached_query.transpose() * dQ);
    gpu_->dWk.add_inplace(gpu_->cached_kv.transpose() * dK);
    gpu_->dWv.add_inplace(gpu_->cached_kv.transpose() * dV);

    adai::gpu::GPUMatrix d_query = dQ * gpu_->Wq.transpose();
    adai::gpu::GPUMatrix d_kv = dK * gpu_->Wk.transpose();
    d_kv.add_inplace(dV * gpu_->Wv.transpose());

    return {std::move(d_query), std::move(d_kv)};
}
#endif
