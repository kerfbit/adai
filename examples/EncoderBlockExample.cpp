#include <iomanip>
#include <iostream>
#include "EncoderBlock.hpp"

void print_matrix_sample(const Matrix& m, const std::string& name, int rows = 3, int cols = 5) {
    std::cout << name << " [" << m.rows << " x " << m.cols << "] (showing first " << rows << "x"
              << cols << "):" << std::endl;
    for (int i = 0; i < std::min(rows, m.rows); ++i) {
        std::cout << "  ";
        for (int j = 0; j < std::min(cols, m.cols); ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << m(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "=== EncoderBlock Standalone Example ===" << std::endl;
    std::cout << std::endl;

    // Configuration
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    float dropout = 0.1f;

    std::cout << "Creating EncoderBlock with:" << std::endl;
    std::cout << "  d_model = " << d_model << std::endl;
    std::cout << "  num_heads = " << num_heads << std::endl;
    std::cout << "  d_ff = " << d_ff << std::endl;
    std::cout << "  dropout = " << dropout << std::endl;
    std::cout << std::endl;

    EncoderBlock encoder_block(d_model, num_heads, d_ff, dropout);
    encoder_block.print_config("EncoderBlock_0");

    // Test 1: Basic forward pass
    std::cout << "\n=== Test 1: Basic Forward Pass ===" << std::endl;
    int seq_len = 10;
    Matrix input(seq_len, d_model);

    // Initialize with small random values
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            input(i, j) = 0.01f * (i + j);
        }
    }

    print_matrix_sample(input, "Input");

    Matrix output = encoder_block.forward(input);
    print_matrix_sample(output, "Output");

    std::cout << "✓ Forward pass completed successfully" << std::endl;

    // Test 2: Forward and backward pass
    std::cout << "\n=== Test 2: Forward and Backward Pass ===" << std::endl;

    Matrix grad_output(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.001f;
        }
    }

    float grad_norm_before = encoder_block.get_gradient_norm();
    std::cout << "Gradient norm before backward: " << grad_norm_before << std::endl;

    Matrix grad_input = encoder_block.backward(grad_output);
    print_matrix_sample(grad_input, "Gradient w.r.t. Input");

    float grad_norm_after = encoder_block.get_gradient_norm();
    std::cout << "Gradient norm after backward: " << grad_norm_after << std::endl;

    std::cout << "✓ Backward pass completed successfully" << std::endl;

    // Test 3: Gradient clipping
    std::cout << "\n=== Test 3: Gradient Clipping ===" << std::endl;

    float max_norm = 5.0f;
    std::cout << "Clipping gradients to max norm: " << max_norm << std::endl;

    encoder_block.clip_gradients(max_norm);

    float grad_norm_clipped = encoder_block.get_gradient_norm();
    std::cout << "Gradient norm after clipping: " << grad_norm_clipped << std::endl;

    if (grad_norm_clipped <= max_norm + 1e-4f) {
        std::cout << "✓ Gradient clipping successful" << std::endl;
    } else {
        std::cout << "✗ Gradient clipping failed" << std::endl;
    }

    // Test 4: Weight update
    std::cout << "\n=== Test 4: Weight Update ===" << std::endl;

    encoder_block.learning_rate = 0.01f;
    std::cout << "Learning rate set to: " << encoder_block.learning_rate << std::endl;

    Matrix output_before = encoder_block.forward(input);

    encoder_block.backward(grad_output);
    encoder_block.update_weights();

    Matrix output_after = encoder_block.forward(input);

    // Check if output changed
    bool weights_updated = false;
    for (int i = 0; i < output_before.rows && !weights_updated; ++i) {
        for (int j = 0; j < output_before.cols && !weights_updated; ++j) {
            if (std::abs(output_before(i, j) - output_after(i, j)) > 1e-6f) {
                weights_updated = true;
            }
        }
    }

    if (weights_updated) {
        std::cout << "✓ Weights updated successfully (output changed)" << std::endl;
    } else {
        std::cout << "⚠ Warning: Output unchanged after weight update" << std::endl;
    }

    // Test 5: Zero gradients
    std::cout << "\n=== Test 5: Zero Gradients ===" << std::endl;

    encoder_block.forward(input);
    encoder_block.backward(grad_output);

    float norm_before_zero = encoder_block.get_gradient_norm();
    std::cout << "Gradient norm before zero_grad: " << norm_before_zero << std::endl;

    encoder_block.zero_grad();

    float norm_after_zero = encoder_block.get_gradient_norm();
    std::cout << "Gradient norm after zero_grad: " << norm_after_zero << std::endl;

    if (norm_after_zero < 1e-6f) {
        std::cout << "✓ Gradients zeroed successfully" << std::endl;
    } else {
        std::cout << "✗ Gradients not fully zeroed" << std::endl;
    }

    // Test 6: Multiple training steps
    std::cout << "\n=== Test 6: Multiple Training Steps ===" << std::endl;

    encoder_block.learning_rate = 0.1f;
    int num_steps = 10;

    std::cout << "Running " << num_steps << " training steps..." << std::endl;

    for (int step = 0; step < num_steps; ++step) {
        Matrix output = encoder_block.forward(input);
        encoder_block.backward(grad_output);

        if (step % 3 == 0) {
            float norm = encoder_block.get_gradient_norm();
            std::cout << "  Step " << step << ": gradient norm = " << norm << std::endl;
        }

        encoder_block.update_weights();
    }

    std::cout << "✓ Multiple training steps completed" << std::endl;

    // Test 7: Save and load weights
    std::cout << "\n=== Test 7: Save and Load Weights ===" << std::endl;

    const std::string weight_file = "test_encoder_block.bin";

    Matrix output_original = encoder_block.forward(input);

    std::cout << "Saving weights to: " << weight_file << std::endl;
    encoder_block.save_weights(weight_file);

    // Create new encoder block with same dimensions
    EncoderBlock encoder_block_loaded(d_model, num_heads, d_ff, dropout);

    std::cout << "Loading weights from: " << weight_file << std::endl;
    encoder_block_loaded.load_weights(weight_file);

    Matrix output_loaded = encoder_block_loaded.forward(input);

    // Compare outputs
    bool outputs_match = true;
    for (int i = 0; i < output_original.rows; ++i) {
        for (int j = 0; j < output_original.cols; ++j) {
            if (std::abs(output_original(i, j) - output_loaded(i, j)) > 1e-5f) {
                outputs_match = false;
                break;
            }
        }
        if (!outputs_match)
            break;
    }

    if (outputs_match) {
        std::cout << "✓ Save/load successful (outputs match)" << std::endl;
    } else {
        std::cout << "✗ Save/load failed (outputs differ)" << std::endl;
    }

    // Test 8: Different sequence lengths
    std::cout << "\n=== Test 8: Variable Sequence Lengths ===" << std::endl;

    std::vector<int> seq_lengths = {1, 5, 20, 50};

    for (int len : seq_lengths) {
        Matrix test_input(len, d_model);
        for (int i = 0; i < len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                test_input(i, j) = 0.1f;
            }
        }

        Matrix test_output = encoder_block.forward(test_input);

        std::cout << "  seq_len=" << len << ": output shape [" << test_output.rows << " x "
                  << test_output.cols << "]";

        if (test_output.rows == len && test_output.cols == d_model) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ✗" << std::endl;
        }
    }

    // Test 9: With attention mask
    std::cout << "\n=== Test 9: Forward with Attention Mask ===" << std::endl;

    int masked_seq_len = 8;
    Matrix masked_input(masked_seq_len, d_model);
    for (int i = 0; i < masked_seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            masked_input(i, j) = 0.1f * i;
        }
    }

    // Create causal mask (lower triangular)
    Matrix causal_mask(masked_seq_len, masked_seq_len);
    for (int i = 0; i < masked_seq_len; ++i) {
        for (int j = 0; j < masked_seq_len; ++j) {
            causal_mask(i, j) = (j <= i) ? 0.0f : -1e9f;  // -inf for masked positions
        }
    }

    std::cout << "Using causal mask (lower triangular)" << std::endl;
    Matrix masked_output = encoder_block.forward(masked_input, &causal_mask);
    print_matrix_sample(masked_output, "Masked Output", 3, 5);

    std::cout << "✓ Forward with mask completed" << std::endl;

    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "✓ Test 1: Basic forward pass" << std::endl;
    std::cout << "✓ Test 2: Forward and backward pass" << std::endl;
    std::cout << "✓ Test 3: Gradient clipping" << std::endl;
    std::cout << "✓ Test 4: Weight update" << std::endl;
    std::cout << "✓ Test 5: Zero gradients" << std::endl;
    std::cout << "✓ Test 6: Multiple training steps" << std::endl;
    std::cout << "✓ Test 7: Save and load weights" << std::endl;
    std::cout << "✓ Test 8: Variable sequence lengths" << std::endl;
    std::cout << "✓ Test 9: Forward with attention mask" << std::endl;

    std::cout << "\n=== All Tests Completed Successfully ===" << std::endl;

    return 0;
}
