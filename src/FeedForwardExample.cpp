#include <iomanip>
#include <iostream>
#include "FeedForward.hpp"
#include "Matrix.hpp"

int main() {
    std::cout << "=== FeedForward Network Example ===" << std::endl;
    std::cout << std::endl;

    // Create a feed-forward network
    // Typical transformer uses d_ff = 4 * d_model
    int d_model = 64;
    int d_ff = 256;

    FeedForward ff(d_model, d_ff);
    ff.print_config("Example FeedForward");
    std::cout << std::endl;

    // Test 1: Basic forward pass
    std::cout << "Test 1: Basic Forward Pass" << std::endl;
    std::cout << "----------------------------" << std::endl;

    Matrix input(5, d_model);  // 5 tokens, 64 dimensions
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < d_model; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    std::cout << "Input shape: [" << input.rows << " x " << input.cols << "]" << std::endl;

    Matrix output = ff.forward(input);

    std::cout << "Output shape: [" << output.rows << " x " << output.cols << "]" << std::endl;
    std::cout << "Output sample (first 5 values): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(4) << output(0, i) << " ";
    }
    std::cout << std::endl << std::endl;

    // Test 2: Backward pass and gradient computation
    std::cout << "Test 2: Backward Pass" << std::endl;
    std::cout << "----------------------" << std::endl;

    Matrix grad_output(5, d_model);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    Matrix grad_input = ff.backward(grad_output);

    std::cout << "Gradient input shape: [" << grad_input.rows << " x " << grad_input.cols << "]"
              << std::endl;

    float grad_norm = ff.get_gradient_norm();
    std::cout << "Gradient norm before clipping: " << std::fixed << std::setprecision(4)
              << grad_norm << std::endl;

    // Test gradient clipping
    ff.clip_gradients(5.0f);
    grad_norm = ff.get_gradient_norm();
    std::cout << "Gradient norm after clipping to 5.0: " << grad_norm << std::endl;
    std::cout << std::endl;

    // Test 3: Training simulation
    std::cout << "Test 3: Training Simulation (5 steps)" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    ff.learning_rate = 0.01f;

    for (int step = 0; step < 5; ++step) {
        // Forward pass
        Matrix output = ff.forward(input);

        // Create dummy gradient
        Matrix grad(5, d_model);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < d_model; ++j) {
                grad(i, j) = 0.1f;
            }
        }

        // Backward pass
        ff.backward(grad);

        float norm = ff.get_gradient_norm();
        std::cout << "Step " << step + 1 << " - Gradient norm: " << std::fixed
                  << std::setprecision(4) << norm << std::endl;

        // Update weights
        ff.update_weights();
    }
    std::cout << std::endl;

    // Test 4: Save and load weights
    std::cout << "Test 4: Weight Persistence" << std::endl;
    std::cout << "--------------------------" << std::endl;

    // Get output before saving
    Matrix output_before = ff.forward(input);
    std::cout << "Output before save (first 3 values): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << std::fixed << std::setprecision(4) << output_before(0, i) << " ";
    }
    std::cout << std::endl;

    // Save weights
    std::string filename = "ff_weights_example.bin";
    ff.save_weights(filename);

    // Create new network and load weights
    FeedForward ff2(d_model, d_ff);
    ff2.load_weights(filename);

    // Get output after loading
    Matrix output_after = ff2.forward(input);
    std::cout << "Output after load (first 3 values): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << std::fixed << std::setprecision(4) << output_after(0, i) << " ";
    }
    std::cout << std::endl;

    // Verify they match
    bool match = true;
    for (int i = 0; i < output_before.rows; ++i) {
        for (int j = 0; j < output_before.cols; ++j) {
            if (std::abs(output_before(i, j) - output_after(i, j)) > 1e-5f) {
                match = false;
                break;
            }
        }
        if (!match)
            break;
    }

    std::cout << "Outputs match: " << (match ? "YES ✓" : "NO ✗") << std::endl;
    std::cout << std::endl;

    // Test 5: Different input sizes
    std::cout << "Test 5: Variable Sequence Lengths" << std::endl;
    std::cout << "----------------------------------" << std::endl;

    for (int seq_len : {1, 10, 50, 100}) {
        Matrix test_input(seq_len, d_model);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                test_input(i, j) = 0.1f;
            }
        }

        Matrix test_output = ff2.forward(test_input);
        std::cout << "Sequence length " << std::setw(3) << seq_len << " -> Output shape: ["
                  << test_output.rows << " x " << test_output.cols << "]" << std::endl;
    }
    std::cout << std::endl;

    // Test 6: Zero gradients
    std::cout << "Test 6: Gradient Zeroing" << std::endl;
    std::cout << "------------------------" << std::endl;

    ff2.forward(input);
    ff2.backward(grad_output);

    float norm_before = ff2.get_gradient_norm();
    std::cout << "Gradient norm before zeroing: " << std::fixed << std::setprecision(4)
              << norm_before << std::endl;

    ff2.zero_grad();
    float norm_after = ff2.get_gradient_norm();
    std::cout << "Gradient norm after zeroing: " << norm_after << std::endl;
    std::cout << std::endl;

    // Clean up temporary file
    std::remove(filename.c_str());

    std::cout << "=== All tests completed successfully! ===" << std::endl;

    return 0;
}
