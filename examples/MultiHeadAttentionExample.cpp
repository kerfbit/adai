#include <iomanip>
#include <iostream>
#include "Matrix.hpp"
#include "MultiHeadAttention.hpp"

int main() {
    std::cout << "=== MultiHeadAttention Standalone Test ===" << std::endl;
    std::cout << std::endl;

    // Create a multi-head attention layer
    int d_model = 512;
    int num_heads = 8;

    MultiHeadAttention mha(d_model, num_heads);
    mha.learning_rate = 0.001f;

    // Print configuration
    mha.print_config();
    std::cout << std::endl;

    // Create sample input
    int seq_len = 10;
    Matrix input(seq_len, d_model);

    // Initialize with some pattern
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            input(i, j) = 0.01f * (i + j);
        }
    }

    std::cout << "Input shape: [" << input.rows << ", " << input.cols << "]" << std::endl;
    std::cout << std::endl;

    // Forward pass
    std::cout << "Performing forward pass..." << std::endl;
    Matrix output = mha.forward(input);

    std::cout << "Output shape: [" << output.rows << ", " << output.cols << "]" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Output sample (first 5 values): ";
    for (int i = 0; i < 5 && i < output.cols; ++i) {
        std::cout << output(0, i) << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Check attention weights
    const Matrix& attn_weights = mha.get_attention_weights();
    std::cout << "Attention weights shape: [" << attn_weights.rows << ", " << attn_weights.cols
              << "]" << std::endl;
    std::cout << "Attention weights (first row, first 5): ";
    for (int i = 0; i < 5 && i < attn_weights.cols; ++i) {
        std::cout << attn_weights(0, i) << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Create gradient for backward pass
    Matrix grad_output(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.001f;
        }
    }

    // Backward pass
    std::cout << "Performing backward pass..." << std::endl;
    Matrix grad_input = mha.backward(grad_output);

    std::cout << "Gradient w.r.t. input shape: [" << grad_input.rows << ", " << grad_input.cols
              << "]" << std::endl;
    std::cout << std::endl;

    // Check gradient norm
    float grad_norm = mha.get_gradient_norm();
    std::cout << "Gradient norm: " << grad_norm << std::endl;
    std::cout << std::endl;

    // Test gradient clipping
    std::cout << "Testing gradient clipping with max_norm=5.0..." << std::endl;
    mha.clip_gradients(5.0f);
    float clipped_norm = mha.get_gradient_norm();
    std::cout << "Gradient norm after clipping: " << clipped_norm << std::endl;
    std::cout << std::endl;

    // Update weights
    std::cout << "Updating weights..." << std::endl;
    mha.update_weights();
    std::cout << "Gradient norm after update: " << mha.get_gradient_norm() << std::endl;
    std::cout << std::endl;

    // Test save and load
    std::string filename = "test_mha_weights.bin";
    std::cout << "Testing save/load..." << std::endl;
    mha.save_weights(filename);

    MultiHeadAttention mha2(d_model, num_heads);
    mha2.load_weights(filename);
    std::cout << "Weights loaded successfully!" << std::endl;
    std::cout << std::endl;

    // Test with attention mask
    std::cout << "Testing with attention mask..." << std::endl;
    Matrix mask(seq_len, seq_len);
    // Create a lower triangular mask (causal attention)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            mask(i, j) = (j <= i) ? 1.0f : 0.0f;
        }
    }

    Matrix masked_output = mha.forward(input, &mask);
    std::cout << "Masked output shape: [" << masked_output.rows << ", " << masked_output.cols << "]"
              << std::endl;
    std::cout << std::endl;

    std::cout << "=== All tests completed successfully! ===" << std::endl;

    return 0;
}
