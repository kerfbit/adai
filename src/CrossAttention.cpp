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

Matrix CrossAttention::scaled_dot_product_attention(const Matrix& Q, const Matrix& K,
                                                    const Matrix& V, const Matrix* mask) {
    int tgt_len = Q.rows;
    int src_len = K.rows;

    // Compute attention scores: QK^T
    Matrix scores = Q * K.transpose();

    // Scale by sqrt(d_k) to prevent softmax saturation
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    scores = scores.scale(scale_factor);

    // Cache scores for backward pass
    cached_scores = scores;

    // Apply mask if provided
    if (mask != nullptr) {
        // Validate mask dimensions
        if (mask->rows != tgt_len || mask->cols != src_len) {
            throw std::invalid_argument(
                "Mask dimensions (" + std::to_string(mask->rows) + ", " +
                std::to_string(mask->cols) + ") must match attention dimensions (" +
                std::to_string(tgt_len) + ", " + std::to_string(src_len) + ")");
        }

        // Apply mask: set masked positions to large negative value
        for (int i = 0; i < tgt_len; ++i) {
            for (int j = 0; j < src_len; ++j) {
                if ((*mask)(i, j) == 0.0f) {
                    scores(i, j) = -1e9f;  // Large negative value
                }
            }
        }
    }

    // Apply softmax to get attention weights
    Matrix attention_weights = Activation::softmax(scores);
    cached_attention_weights = attention_weights;

    // Apply attention weights to values
    Matrix output = attention_weights * V;

    return output;
}

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

    // Project to Q, K, V
    cached_Q = query_input * W_q;  // [tgt_len, d_model]
    cached_K = kv_input * W_k;     // [src_len, d_model]
    cached_V = kv_input * W_v;     // [src_len, d_model]

    // Compute scaled dot-product attention
    Matrix attention_output = scaled_dot_product_attention(cached_Q, cached_K, cached_V, mask);
    cached_attention_output = attention_output;

    // Apply output projection
    Matrix output = attention_output * W_o;

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

    // Compute attention scores: Q_new * K_encoder^T
    // Shape: [num_new_tokens, src_len]
    Matrix scores = cached_Q * K_full.transpose();

    // Scale by sqrt(d_k)
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    scores = scores.scale(scale_factor);

    cached_scores = scores;

    // Apply mask if provided
    // Mask shape: [num_new_tokens, src_len]
    if (mask != nullptr) {
        if (mask->rows != num_new_tokens || mask->cols != src_len) {
            throw std::invalid_argument(
                "Mask dimensions (" + std::to_string(mask->rows) + ", " +
                std::to_string(mask->cols) + ") must match [num_new_tokens=" +
                std::to_string(num_new_tokens) + ", src_len=" + std::to_string(src_len) + "]");
        }

        for (int i = 0; i < num_new_tokens; ++i) {
            for (int j = 0; j < src_len; ++j) {
                if ((*mask)(i, j) == 0.0f) {
                    scores(i, j) = -1e9f;
                }
            }
        }
    }

    // Apply softmax
    cached_attention_weights = Activation::softmax(scores);

    // Apply attention to values
    // [num_new_tokens, src_len] * [src_len, d_model] = [num_new_tokens, d_model]
    cached_attention_output = cached_attention_weights * V_full;

    // Output projection
    Matrix output = cached_attention_output * W_o;

    return output;
}

void CrossAttention::backward(const Matrix& grad_output, Matrix& grad_query_input,
                              Matrix& grad_kv_input) {
    int tgt_len = cached_query_input.rows;
    int src_len = cached_kv_input.rows;

    // Gradient through output projection
    Matrix grad_attention_output = grad_output * W_o.transpose();
    W_o_grad = W_o_grad + (cached_attention_output.transpose() * grad_output);

    // Gradient through attention output
    Matrix grad_attention_weights = grad_attention_output * cached_V.transpose();
    Matrix grad_V = cached_attention_weights.transpose() * grad_attention_output;

    // Gradient through softmax
    Matrix grad_scores(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < src_len; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < src_len; ++k) {
                if (k == j) {
                    sum += grad_attention_weights(i, k) * cached_attention_weights(i, j) *
                           (1.0f - cached_attention_weights(i, j));
                } else {
                    sum -= grad_attention_weights(i, k) * cached_attention_weights(i, j) *
                           cached_attention_weights(i, k);
                }
            }
            grad_scores(i, j) = sum;
        }
    }

    // Gradient through scaling
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    grad_scores = grad_scores.scale(scale_factor);

    // Gradient through QK^T
    Matrix grad_Q = grad_scores * cached_K;
    Matrix grad_K = grad_scores.transpose() * cached_Q;

    // Gradient through projections
    grad_query_input = grad_Q * W_q.transpose();
    Matrix grad_kv_from_K = grad_K * W_k.transpose();
    Matrix grad_kv_from_V = grad_V * W_v.transpose();
    grad_kv_input = grad_kv_from_K + grad_kv_from_V;

    // Accumulate weight gradients
    W_q_grad = W_q_grad + (cached_query_input.transpose() * grad_Q);
    W_k_grad = W_k_grad + (cached_kv_input.transpose() * grad_K);
    W_v_grad = W_v_grad + (cached_kv_input.transpose() * grad_V);
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

    adai::gpu::GPUMatrix scores = gpu_->cached_Q * gpu_->cached_K.transpose();
    scores = scores.scale(scale);
    if (mask != nullptr)
        scores.masked_fill_inplace(*mask, -1e9f);

    if (gpu_->cached_weights.rows != tgt || gpu_->cached_weights.cols != src)
        gpu_->cached_weights = adai::gpu::GPUMatrix(tgt, src);
    adai::gpu::matrix_copy_device_to_device_gpu(scores.device_ptr(),
                                                gpu_->cached_weights.device_ptr(), tgt * src);
    gpu_->cached_weights.softmax_rows_inplace();

    gpu_->cached_attn_out = gpu_->cached_weights * gpu_->cached_V;
    return gpu_->cached_attn_out * gpu_->Wo;
}

std::pair<adai::gpu::GPUMatrix, adai::gpu::GPUMatrix> CrossAttention::gpu_backward(
    const adai::gpu::GPUMatrix& dout) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    gpu_->dWo.add_inplace(gpu_->cached_attn_out.transpose() * dout);
    adai::gpu::GPUMatrix d_ao = dout * gpu_->Wo.transpose();

    adai::gpu::GPUMatrix dV = gpu_->cached_weights.transpose() * d_ao;
    adai::gpu::GPUMatrix d_wts = d_ao * gpu_->cached_V.transpose();

    adai::gpu::GPUMatrix d_scores = gpu_->cached_weights.softmax_backward(d_wts);
    d_scores = d_scores.scale(scale);

    adai::gpu::GPUMatrix dQ = d_scores * gpu_->cached_K;
    adai::gpu::GPUMatrix dK = d_scores.transpose() * gpu_->cached_Q;

    gpu_->dWq.add_inplace(gpu_->cached_query.transpose() * dQ);
    gpu_->dWk.add_inplace(gpu_->cached_kv.transpose() * dK);
    gpu_->dWv.add_inplace(gpu_->cached_kv.transpose() * dV);

    adai::gpu::GPUMatrix d_query = dQ * gpu_->Wq.transpose();
    adai::gpu::GPUMatrix d_kv = dK * gpu_->Wk.transpose();
    d_kv.add_inplace(dV * gpu_->Wv.transpose());

    return {std::move(d_query), std::move(d_kv)};
}
#endif
