#include "DecoderBlock.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "CrossAttention.hpp"

DecoderBlock::DecoderBlock(int d_model, int num_heads, int d_ff, float dropout)
    : d_model(d_model),
      num_heads(num_heads),
      d_ff(d_ff),
      dropout_rate(dropout),
      learning_rate(0.001f) {
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

    std::cout << "DecoderBlock initialized:" << std::endl;
    std::cout << "  Model dimension: " << d_model << std::endl;
    std::cout << "  Number of heads: " << num_heads << std::endl;
    std::cout << "  Feed-forward dimension: " << d_ff << std::endl;
}

Matrix DecoderBlock::forward(const Matrix& input, const Matrix& encoder_output,
                             const Matrix& self_attn_mask, const Matrix* cross_attn_mask) {
    // Cache input for backward pass
    cached_input = input;
    cached_encoder_output = encoder_output;

    // Step 1: Masked self-attention (causal)
    Matrix self_attn_out = self_attention->forward(input, &self_attn_mask);
    cached_self_attn_output = self_attn_out;

    // Step 2: First residual connection (input + self-attention output)
    Matrix residual1(input.rows, input.cols);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            residual1(i, j) = input(i, j) + self_attn_out(i, j);
        }
    }
    cached_residual1 = residual1;

    // Step 3: First layer normalization
    Matrix normed1 = norm1->forward(residual1);
    cached_normed1 = normed1;

    // Step 4: Cross-attention to encoder output
    // Query from decoder (normed1), Key and Value from encoder
    Matrix cross_attn_out = cross_attention->forward(normed1, encoder_output, cross_attn_mask);
    cached_cross_attn_output = cross_attn_out;

    // Step 5: Second residual connection (normed1 + cross-attention output)
    Matrix residual2(normed1.rows, normed1.cols);
    for (int i = 0; i < normed1.rows; ++i) {
        for (int j = 0; j < normed1.cols; ++j) {
            residual2(i, j) = normed1(i, j) + cross_attn_out(i, j);
        }
    }
    cached_residual2 = residual2;

    // Step 6: Second layer normalization
    Matrix normed2 = norm2->forward(residual2);
    cached_normed2 = normed2;

    // Step 7: Feed-forward network
    Matrix ff_out = feed_forward->forward(normed2);
    cached_ff_output = ff_out;

    // Step 8: Third residual connection (normed2 + ff output)
    Matrix residual3(normed2.rows, normed2.cols);
    for (int i = 0; i < normed2.rows; ++i) {
        for (int j = 0; j < normed2.cols; ++j) {
            residual3(i, j) = normed2(i, j) + ff_out(i, j);
        }
    }
    cached_residual3 = residual3;

    // Step 9: Third layer normalization
    Matrix normed3 = norm3->forward(residual3);

    return normed3;
}

Matrix DecoderBlock::backward(const Matrix& grad_output) {
    // Step 1: Gradient through third layer norm
    Matrix grad_residual3 = norm3->backward(grad_output);

    // Step 2: Gradient through third residual connection
    // Gradient splits into two paths: normed2 and ff_output
    Matrix grad_normed2(grad_residual3.rows, grad_residual3.cols);
    Matrix grad_ff_output(grad_residual3.rows, grad_residual3.cols);

    for (int i = 0; i < grad_residual3.rows; ++i) {
        for (int j = 0; j < grad_residual3.cols; ++j) {
            grad_normed2(i, j) = grad_residual3(i, j);
            grad_ff_output(i, j) = grad_residual3(i, j);
        }
    }

    // Step 3: Gradient through feed-forward network
    Matrix grad_normed2_from_ff = feed_forward->backward(grad_ff_output);

    // Step 4: Accumulate gradients from both paths
    for (int i = 0; i < grad_normed2.rows; ++i) {
        for (int j = 0; j < grad_normed2.cols; ++j) {
            grad_normed2(i, j) += grad_normed2_from_ff(i, j);
        }
    }

    // Step 5: Gradient through second layer norm
    Matrix grad_residual2 = norm2->backward(grad_normed2);

    // Step 6: Gradient through second residual connection
    // Gradient splits into two paths: normed1 and cross_attn_output
    Matrix grad_normed1(grad_residual2.rows, grad_residual2.cols);
    Matrix grad_cross_attn_output(grad_residual2.rows, grad_residual2.cols);

    for (int i = 0; i < grad_residual2.rows; ++i) {
        for (int j = 0; j < grad_residual2.cols; ++j) {
            grad_normed1(i, j) = grad_residual2(i, j);
            grad_cross_attn_output(i, j) = grad_residual2(i, j);
        }
    }

    // Step 7: Gradient through cross-attention
    Matrix grad_normed1_from_cross, grad_encoder_output;
    cross_attention->backward(grad_cross_attn_output, grad_normed1_from_cross, grad_encoder_output);

    // Step 8: Accumulate gradients from both paths
    for (int i = 0; i < grad_normed1.rows; ++i) {
        for (int j = 0; j < grad_normed1.cols; ++j) {
            grad_normed1(i, j) += grad_normed1_from_cross(i, j);
        }
    }

    // Step 9: Gradient through first layer norm
    Matrix grad_residual1 = norm1->backward(grad_normed1);

    // Step 10: Gradient through first residual connection
    // Gradient splits into two paths: input and self_attn_output
    Matrix grad_input(grad_residual1.rows, grad_residual1.cols);
    Matrix grad_self_attn_output(grad_residual1.rows, grad_residual1.cols);

    for (int i = 0; i < grad_residual1.rows; ++i) {
        for (int j = 0; j < grad_residual1.cols; ++j) {
            grad_input(i, j) = grad_residual1(i, j);
            grad_self_attn_output(i, j) = grad_residual1(i, j);
        }
    }

    // Step 11: Gradient through self-attention
    Matrix grad_input_from_self = self_attention->backward(grad_self_attn_output);

    // Step 12: Accumulate gradients from both paths
    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            grad_input(i, j) += grad_input_from_self(i, j);
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

    file.close();

    // Save sub-components to separate files
    self_attention->save_weights(filepath + ".self_attn");
    cross_attention->save(filepath + ".cross_attn");
    feed_forward->save_weights(filepath + ".ff");

    // Note: LayerNorm doesn't have save/load methods
    // It has minimal learnable params (gamma/beta) that will be reinitialized
}

void DecoderBlock::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    // Load dimensions and hyperparameters
    int loaded_d_model, loaded_num_heads, loaded_d_ff;
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(loaded_d_model));
    file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(loaded_num_heads));
    file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(loaded_d_ff));
    file.read(reinterpret_cast<char*>(&dropout_rate), sizeof(dropout_rate));
    file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));

    if (loaded_d_model != d_model || loaded_num_heads != num_heads || loaded_d_ff != d_ff) {
        throw std::runtime_error("Dimension mismatch in saved model");
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
}
