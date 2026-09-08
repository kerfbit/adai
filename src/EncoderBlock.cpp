// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

#include "EncoderBlock.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

EncoderBlock::EncoderBlock(int d_model, int num_heads, int d_ff, float dropout)
    : d_model(d_model), num_heads(num_heads), d_ff(d_ff), dropout_rate(dropout) {
    // Initialize multi-head attention
    attention = std::make_unique<MultiHeadAttention>(d_model, num_heads);

    // Initialize feed-forward network
    feed_forward = std::make_unique<FeedForward>(d_model, d_ff);

    // Initialize layer normalization layers
    norm1 = std::make_unique<LayerNorm>(d_model);
    norm2 = std::make_unique<LayerNorm>(d_model);

    // Set learning rates for sub-components
    attention->learning_rate = learning_rate;
    feed_forward->learning_rate = learning_rate;
    norm1->learning_rate = learning_rate;
    norm2->learning_rate = learning_rate;
}

Matrix EncoderBlock::forward(const Matrix& input, const Matrix* mask) {
    // Cache input for backward pass
    cached_input = input;

    // Step 1: Pre-attention layer normalization
    Matrix normed1 = norm1->forward(input);
    cached_normed1 = normed1;

    // Step 2: Multi-head self-attention on the normalized input
    Matrix attn_output = attention->forward(normed1, mask);
    cached_attn_output = attn_output;

    // Step 3: First residual connection (input + attention output) — stays
    // unnormalized; this raw residual stream is what Pre-LN preserves.
    Matrix residual1(input.rows, input.cols);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            residual1(i, j) = input(i, j) + attn_output(i, j);
        }
    }
    cached_residual1 = residual1;

    // Step 4: Pre-feedforward layer normalization
    Matrix normed2 = norm2->forward(residual1);

    // Step 5: Feed-forward network on the normalized residual
    Matrix ff_output = feed_forward->forward(normed2);
    cached_ff_output = ff_output;

    // Step 6: Second residual connection (residual1 + ff_output) — the block's
    // output is unnormalized; LLMEncoder's final_norm normalizes the
    // accumulated residual stream once, after the last block.
    Matrix residual2(residual1.rows, residual1.cols);
    for (int i = 0; i < residual1.rows; ++i) {
        for (int j = 0; j < residual1.cols; ++j) {
            residual2(i, j) = residual1(i, j) + ff_output(i, j);
        }
    }
    cached_residual2 = residual2;

    return residual2;
}

Matrix EncoderBlock::backward(const Matrix& grad_output) {
    // Step 1: Gradient through second residual connection (output = residual1 + ff_output)
    // Residual splits gradient into two paths:
    //   - One path goes directly to residual1
    //   - Other path goes through the feed-forward branch (ff_output -> normed2 -> residual1)
    Matrix grad_residual1(grad_output.rows, grad_output.cols);
    Matrix grad_ff_output(grad_output.rows, grad_output.cols);

    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_residual1(i, j) = grad_output(i, j);
            grad_ff_output(i, j) = grad_output(i, j);
        }
    }

    // Step 2: Gradient through feed-forward network
    Matrix grad_normed2 = feed_forward->backward(grad_ff_output);

    // Step 3: Gradient through pre-feedforward layer norm
    Matrix grad_residual1_from_norm2 = norm2->backward(grad_normed2);

    // Step 4: Accumulate gradients from both paths into residual1
    for (int i = 0; i < grad_residual1.rows; ++i) {
        for (int j = 0; j < grad_residual1.cols; ++j) {
            grad_residual1(i, j) += grad_residual1_from_norm2(i, j);
        }
    }

    // Step 5: Gradient through first residual connection (residual1 = input + attn_output)
    // Residual splits gradient into two paths:
    //   - One path goes directly to input
    //   - Other path goes through the attention branch (attn_output -> normed1 -> input)
    Matrix grad_input(grad_residual1.rows, grad_residual1.cols);
    Matrix grad_attn_output(grad_residual1.rows, grad_residual1.cols);

    for (int i = 0; i < grad_residual1.rows; ++i) {
        for (int j = 0; j < grad_residual1.cols; ++j) {
            grad_input(i, j) = grad_residual1(i, j);
            grad_attn_output(i, j) = grad_residual1(i, j);
        }
    }

    // Step 6: Gradient through multi-head attention
    Matrix grad_normed1 = attention->backward(grad_attn_output);

    // Step 7: Gradient through pre-attention layer norm
    Matrix grad_input_from_norm1 = norm1->backward(grad_normed1);

    // Step 8: Accumulate gradients from both paths into input
    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            grad_input(i, j) += grad_input_from_norm1(i, j);
        }
    }

    return grad_input;
}

void EncoderBlock::update_weights() {
    // Update all sub-components with current learning rate
    attention->learning_rate = learning_rate;
    feed_forward->learning_rate = learning_rate;
    norm1->learning_rate = learning_rate;
    norm2->learning_rate = learning_rate;

    // Apply gradient updates to all components
    attention->update_weights();
    feed_forward->update_weights();
    norm1->update_weights();
    norm2->update_weights();
}

void EncoderBlock::zero_grad() {
    attention->zero_grad();
    feed_forward->zero_grad();
    norm1->zero_grad();
    norm2->zero_grad();
}

float EncoderBlock::get_gradient_norm() const {
    float norm_sq = 0.0f;

    // Accumulate squared norms from all components
    float attn_norm = attention->get_gradient_norm();
    norm_sq += attn_norm * attn_norm;

    float ff_norm = feed_forward->get_gradient_norm();
    norm_sq += ff_norm * ff_norm;

    // LayerNorm doesn't expose gradient norm method
    // Approximation: gradients handled internally during backward

    return std::sqrt(norm_sq);
}

void EncoderBlock::clip_gradients(float max_norm) {
    float current_norm = get_gradient_norm();

    if (current_norm > max_norm) {
        float scale = max_norm / current_norm;

        // Clip gradients in components that support it
        attention->clip_gradients(attention->get_gradient_norm() * scale);
        feed_forward->clip_gradients(feed_forward->get_gradient_norm() * scale);

        // LayerNorm doesn't expose gradient clipping method
        // Gradients are small relative to other components
    }
}

void EncoderBlock::save_weights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write dimensions for validation on load
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_ff), sizeof(int));
    file.write(reinterpret_cast<const char*>(&dropout_rate), sizeof(float));

    // Write LayerNorm parameters directly
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

    file.close();

    // Save component weights to separate files
    std::string base = filename.substr(0, filename.find_last_of('.'));
    attention->save_weights(base + "_attention.bin");
    feed_forward->save_weights(base + "_feedforward.bin");

    std::cout << "Saved EncoderBlock weights to " << filename << '\n';
}

void EncoderBlock::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Read and validate dimensions
    int saved_d_model = 0, saved_num_heads = 0, saved_d_ff = 0;
    float saved_dropout_rate = NAN;

    file.read(reinterpret_cast<char*>(&saved_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_num_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_d_ff), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_dropout_rate), sizeof(float));

    if (saved_d_model != d_model || saved_num_heads != num_heads || saved_d_ff != d_ff) {
        throw std::runtime_error("Dimension mismatch: saved (" + std::to_string(saved_d_model) +
                                 "," + std::to_string(saved_num_heads) + "," +
                                 std::to_string(saved_d_ff) + ") vs current (" +
                                 std::to_string(d_model) + "," + std::to_string(num_heads) + "," +
                                 std::to_string(d_ff) + ")");
    }

    // Read LayerNorm parameters directly
    Matrix gamma1(1, d_model);
    Matrix beta1(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&gamma1(0, j)), sizeof(float));
    }
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&beta1(0, j)), sizeof(float));
    }
    norm1->set_gamma(gamma1);
    norm1->set_beta(beta1);

    Matrix gamma2(1, d_model);
    Matrix beta2(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&gamma2(0, j)), sizeof(float));
    }
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&beta2(0, j)), sizeof(float));
    }
    norm2->set_gamma(gamma2);
    norm2->set_beta(beta2);

    file.close();

    // Load component weights from separate files
    std::string base = filename.substr(0, filename.find_last_of('.'));
    attention->load_weights(base + "_attention.bin");
    feed_forward->load_weights(base + "_feedforward.bin");

    std::cout << "Loaded EncoderBlock weights from " << filename << '\n';
}

void EncoderBlock::print_config(const std::string& name) const {
    std::cout << "\n" << name << " Configuration:" << '\n';
    std::cout << "  Model Dimension (d_model): " << d_model << '\n';
    std::cout << "  Number of Heads: " << num_heads << '\n';
    std::cout << "  Feed-Forward Dimension (d_ff): " << d_ff << '\n';
    std::cout << "  Dropout Rate: " << dropout_rate << '\n';
    std::cout << "  Learning Rate: " << learning_rate << '\n';

    // Calculate parameter count
    int attn_params = d_model * d_model * 4;              // Q, K, V, output projection
    int ff_params = 2 * d_model * d_ff + d_ff + d_model;  // W1, W2, b1, b2
    int norm_params = 4 * d_model;                        // gamma and beta for both layer norms
    int total_params = attn_params + ff_params + norm_params;

    std::cout << "  Total Parameters: " << total_params << '\n';
    std::cout << "    - Attention: " << attn_params << '\n';
    std::cout << "    - Feed-Forward: " << ff_params << '\n';
    std::cout << "    - Layer Norm: " << norm_params << '\n';
}

void EncoderBlock::register_parameters_with_optimizer(Optimizer& optimizer) {
    // Register all sub-component parameters
    attention->set_optimizer(&optimizer);
    feed_forward->set_optimizer(&optimizer);
    norm1->set_optimizer(&optimizer);
    norm2->set_optimizer(&optimizer);
}

#ifdef ADAI_ENABLE_GPU
void EncoderBlock::gpu_upload_weights() {
    attention->gpu_upload_weights();
    feed_forward->gpu_upload_weights();
    norm1->gpu_upload_weights();
    norm2->gpu_upload_weights();
}

void EncoderBlock::gpu_download_grads() {
    attention->gpu_download_grads();
    feed_forward->gpu_download_grads();
    norm1->gpu_download_grads();
    norm2->gpu_download_grads();
}

void EncoderBlock::gpu_zero_grads() {
    attention->gpu_zero_grads();
    feed_forward->gpu_zero_grads();
    norm1->gpu_zero_grads();
    norm2->gpu_zero_grads();
}

adai::gpu::GPUMatrix EncoderBlock::gpu_forward(const adai::gpu::GPUMatrix& input,
                                               const adai::gpu::GPUMatrix* mask) {
    // LayerNorm1 + self-attention + residual1 (unnormalized)
    adai::gpu::GPUMatrix normed1 = norm1->gpu_forward(input);
    adai::gpu::GPUMatrix attn_out = attention->gpu_forward(normed1, mask);
    adai::gpu::GPUMatrix res1 = input + attn_out;

    // LayerNorm2 + FeedForward + residual2 (unnormalized — final_norm handles
    // normalization once, after the last block, at the LLMEncoder level)
    adai::gpu::GPUMatrix normed2 = norm2->gpu_forward(res1);
    adai::gpu::GPUMatrix ff_out = feed_forward->gpu_forward(normed2);
    return res1 + ff_out;
}

adai::gpu::GPUMatrix EncoderBlock::gpu_backward(const adai::gpu::GPUMatrix& dout) {
    // Residual2 split: d_res1 = dout (direct) + norm2/ff branch
    adai::gpu::GPUMatrix d_normed2 = feed_forward->gpu_backward(dout);
    adai::gpu::GPUMatrix d_res1_from_norm2 = norm2->gpu_backward(d_normed2);
    adai::gpu::GPUMatrix d_res1 = dout + d_res1_from_norm2;

    // Residual1 split: d_input = d_res1 (direct) + norm1/attn branch
    adai::gpu::GPUMatrix d_normed1 = attention->gpu_backward(d_res1);
    adai::gpu::GPUMatrix d_input_from_norm1 = norm1->gpu_backward(d_normed1);
    return d_res1 + d_input_from_norm1;
}
#endif
