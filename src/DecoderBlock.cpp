// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

#include "DecoderBlock.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "Logger.hpp"
using adai::Logger;
#include "CrossAttention.hpp"

DecoderBlock::DecoderBlock(int d_model, int num_heads, int d_ff, float dropout)
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

    Logger::info("DecoderBlock initialized: d_model={} num_heads={} d_ff={}", d_model, num_heads,
                 d_ff);
}

Matrix DecoderBlock::forward(const Matrix& input, const Matrix& encoder_output,
                             const Matrix& self_attn_mask, const Matrix* cross_attn_mask) {
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
}

void DecoderBlock::zero_grad() {
    self_attention->zero_grad();
    cross_attention->zero_grad();
    feed_forward->zero_grad();
    norm1->zero_grad();
    norm2->zero_grad();
    norm3->zero_grad();
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

    file.close();

    // Save sub-components to separate files
    self_attention->save_weights(filepath + ".self_attn");
    cross_attention->save(filepath + ".cross_attn");
    feed_forward->save_weights(filepath + ".ff");
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

    file.close();

    // Load sub-components
    self_attention->load_weights(filepath + ".self_attn");
    cross_attention->load(filepath + ".cross_attn");
    feed_forward->load_weights(filepath + ".ff");

    // Update learning rates
    self_attention->learning_rate = learning_rate;
    cross_attention->learning_rate = learning_rate;
    feed_forward->learning_rate = learning_rate;
}

void DecoderBlock::register_parameters_with_optimizer(Optimizer& optimizer) {
    // Register all sub-component parameters
    self_attention->set_optimizer(&optimizer);
    cross_attention->set_optimizer(&optimizer);
    feed_forward->set_optimizer(&optimizer);
    norm1->set_optimizer(&optimizer);
    norm2->set_optimizer(&optimizer);
    norm3->set_optimizer(&optimizer);
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
