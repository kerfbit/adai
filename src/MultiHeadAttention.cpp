#include "MultiHeadAttention.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "Activation.hpp"
#include "Optimizer.hpp"

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
      W_o_grad(d_model, d_model),
      optimizer(nullptr),
      learning_rate(0.001f) {
    // Validate that d_model is divisible by num_heads
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model (" + std::to_string(d_model) +
                                    ") must be divisible by num_heads (" +
                                    std::to_string(num_heads) + ")");
    }

    // Xavier/He initialization for weight matrices
    // Scale factor based on the input dimension
    float scale = std::sqrt(2.0f / d_model);

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

Matrix MultiHeadAttention::scaled_dot_product_attention(const Matrix& Q, const Matrix& K,
                                                        const Matrix& V, const Matrix* mask) {
    int seq_len = Q.rows;

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
        if (mask->rows != seq_len || mask->cols != seq_len) {
            throw std::invalid_argument("Mask dimensions (" + std::to_string(mask->rows) + ", " +
                                        std::to_string(mask->cols) +
                                        ") must match sequence length (" + std::to_string(seq_len) +
                                        ", " + std::to_string(seq_len) + ")");
        }

        // Apply mask: set masked positions to large negative value
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                if ((*mask)(i, j) == 0.0f) {
                    scores(i, j) = -1e9f;  // Large negative value
                }
            }
        }
    }

    // Apply softmax to get attention weights
    Matrix attention_weights = Activation::softmax(scores);

    // Apply attention weights to values
    Matrix output = attention_weights * V;

    return output;
}

Matrix MultiHeadAttention::forward(const Matrix& input, const Matrix* mask) {
    // Cache input for backward pass
    cached_input = input;

    int seq_len = input.rows;

    // Validate input dimensions
    if (input.cols != d_model) {
        throw std::invalid_argument("Input dimension (" + std::to_string(input.cols) +
                                    ") must match d_model (" + std::to_string(d_model) + ")");
    }

    // Linear projections to get Q, K, V
    cached_Q = input * W_q;
    cached_K = input * W_k;
    cached_V = input * W_v;

    // Compute scaled dot-product attention
    // This function also caches scores and computes attention_weights
    Matrix scores = cached_Q * cached_K.transpose();

    // Scale by sqrt(d_k)
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    scores = scores.scale(scale_factor);

    // Cache scores for backward pass
    cached_scores = scores;

    // Apply mask if provided
    if (mask != nullptr) {
        if (mask->rows != seq_len || mask->cols != seq_len) {
            throw std::invalid_argument("Mask dimensions must match sequence length");
        }

        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                if ((*mask)(i, j) == 0.0f) {
                    scores(i, j) = -1e9f;
                }
            }
        }
    }

    // Apply softmax to get attention weights
    cached_attention_weights = Activation::softmax(scores);

    // Apply attention to values
    cached_attention_output = cached_attention_weights * cached_V;

    // Final linear projection
    Matrix output = cached_attention_output * W_o;

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

    // Compute attention scores: Q_new * K_full^T
    // Shape: [num_new_tokens, total_seq_len]
    Matrix scores = Q_new * K_full.transpose();

    // Scale by sqrt(d_k)
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    scores = scores.scale(scale_factor);

    // Cache scores for backward pass
    cached_scores = scores;

    // Apply mask if provided
    // Mask shape should be [num_new_tokens, total_seq_len]
    if (mask != nullptr) {
        if (mask->rows != num_new_tokens || mask->cols != total_seq_len) {
            throw std::invalid_argument(
                "Mask dimensions (" + std::to_string(mask->rows) + ", " +
                std::to_string(mask->cols) + ") must match [num_new_tokens=" +
                std::to_string(num_new_tokens) + ", total_seq_len=" +
                std::to_string(total_seq_len) + "]");
        }

        for (int i = 0; i < num_new_tokens; ++i) {
            for (int j = 0; j < total_seq_len; ++j) {
                if ((*mask)(i, j) == 0.0f) {
                    scores(i, j) = -1e9f;
                }
            }
        }
    }

    // Apply softmax to get attention weights
    // Shape: [num_new_tokens, total_seq_len]
    cached_attention_weights = Activation::softmax(scores);

    // Apply attention to values
    // [num_new_tokens, total_seq_len] * [total_seq_len, d_model] = [num_new_tokens, d_model]
    cached_attention_output = cached_attention_weights * V_full;

    // Final linear projection
    Matrix output = cached_attention_output * W_o;

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
    W_o_grad = cached_attention_output.transpose() * grad_output;

    // Gradient w.r.t. attention output
    // grad_attn_out = grad_output * W_o^T
    Matrix grad_attn_out = grad_output * W_o.transpose();

    // Gradient w.r.t. V
    // grad_V = attention_weights^T * grad_attn_out
    Matrix grad_V = cached_attention_weights.transpose() * grad_attn_out;

    // Gradient w.r.t. attention weights
    // grad_attn_weights = grad_attn_out * V^T
    Matrix grad_attn_weights = grad_attn_out * cached_V.transpose();

    // Gradient through softmax
    // This is complex: for softmax, we need to compute the Jacobian
    // For simplicity, we use: grad_scores = attention_weights * (grad_attn_weights - sum)
    Matrix grad_scores(cached_scores.rows, cached_scores.cols);

    for (int i = 0; i < cached_scores.rows; ++i) {
        // Compute sum of (attention_weights * grad_attn_weights) for this row
        float sum = 0.0f;
        for (int k = 0; k < cached_scores.cols; ++k) {
            sum += cached_attention_weights(i, k) * grad_attn_weights(i, k);
        }

        // Compute gradient for each element in the row
        for (int j = 0; j < cached_scores.cols; ++j) {
            grad_scores(i, j) = cached_attention_weights(i, j) * (grad_attn_weights(i, j) - sum);
        }
    }

    // Scale gradient (derivative of scaling)
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    grad_scores = grad_scores.scale(scale_factor);

    // Gradient w.r.t. Q and K
    // grad_Q = grad_scores * K
    Matrix grad_Q = grad_scores * cached_K;

    // grad_K = grad_scores^T * Q
    Matrix grad_K = grad_scores.transpose() * cached_Q;

    // Gradients w.r.t. projection weights
    // W_q_grad = input^T * grad_Q
    W_q_grad = cached_input.transpose() * grad_Q;

    // W_k_grad = input^T * grad_K
    W_k_grad = cached_input.transpose() * grad_K;

    // W_v_grad = input^T * grad_V
    W_v_grad = cached_input.transpose() * grad_V;

    // Gradient w.r.t. input (sum gradients from all three projections)
    // grad_input = grad_Q * W_q^T + grad_K * W_k^T + grad_V * W_v^T
    Matrix grad_input = grad_Q * W_q.transpose();
    Matrix grad_from_K = grad_K * W_k.transpose();
    Matrix grad_from_V = grad_V * W_v.transpose();

    // Add gradients
    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            grad_input(i, j) += grad_from_K(i, j) + grad_from_V(i, j);
        }
    }

    return grad_input;
}

void MultiHeadAttention::set_optimizer(Optimizer* opt) {
    optimizer = opt;
    if (optimizer) {
        register_parameters();
    }
}

void MultiHeadAttention::register_parameters() {
    if (!optimizer)
        return;

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
}

void MultiHeadAttention::print_config(const std::string& name) const {
    std::cout << name << " Configuration:" << std::endl;
    std::cout << "  Model Dimension (d_model): " << d_model << std::endl;
    std::cout << "  Number of Heads: " << num_heads << std::endl;
    std::cout << "  Dimension per Head (d_k): " << d_k << std::endl;
    std::cout << "  Total Parameters: " << (4 * d_model * d_model) << std::endl;
    std::cout << "  Memory Usage: " << std::fixed << std::setprecision(2)
              << (4 * d_model * d_model * sizeof(float)) / 1024.0f / 1024.0f << " MB" << std::endl;
    std::cout << "  Learning Rate: " << learning_rate << std::endl;
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

    std::cout << "Saved MultiHeadAttention weights to " << filename << std::endl;
}

void MultiHeadAttention::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read dimensions
    int saved_d_model, saved_num_heads;
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

    std::cout << "Loaded MultiHeadAttention weights from " << filename << std::endl;
}

float MultiHeadAttention::get_gradient_norm() const {
    float sum_squares = 0.0f;

    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float grad;

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
