#include <iomanip>
#include <iostream>
#include "DecoderBlock.hpp"
#include "Matrix.hpp"

/**
 * DecoderBlock Example
 *
 * Demonstrates the usage of the DecoderBlock class for transformer decoder.
 * Shows forward pass, backward pass, and gradient flow through all three
 * sub-layers: masked self-attention, cross-attention, and feed-forward.
 */

void print_matrix(const Matrix& m, const std::string& name) {
    std::cout << "\n" << name << " [" << m.rows << " x " << m.cols << "]:\n";
    for (int i = 0; i < std::min(3, m.rows); ++i) {
        for (int j = 0; j < std::min(5, m.cols); ++j) {
            std::cout << std::fixed << std::setprecision(4) << m(i, j) << " ";
        }
        if (m.cols > 5)
            std::cout << "...";
        std::cout << "\n";
    }
    if (m.rows > 3)
        std::cout << "...\n";
}

Matrix create_causal_mask(int seq_len) {
    Matrix mask(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            mask(i, j) = (j <= i) ? 1.0f : 0.0f;  // Only attend to current and past
        }
    }
    return mask;
}

int main() {
    std::cout << "=== Transformer DecoderBlock Example ===\n";
    std::cout << "This demonstrates masked self-attention + cross-attention + FFN\n";

    // Configuration
    const int d_model = 64;     // Model dimension
    const int num_heads = 4;    // Number of attention heads
    const int d_ff = 256;       // Feed-forward hidden dimension
    const int tgt_seq_len = 5;  // Target sequence length (decoder input)
    const int src_seq_len = 8;  // Source sequence length (encoder output)
    const float dropout = 0.1f;

    std::cout << "\nConfiguration:\n";
    std::cout << "  d_model: " << d_model << "\n";
    std::cout << "  num_heads: " << num_heads << "\n";
    std::cout << "  d_ff: " << d_ff << "\n";
    std::cout << "  target_seq_len: " << tgt_seq_len << "\n";
    std::cout << "  source_seq_len: " << src_seq_len << "\n";

    // Create decoder block
    std::cout << "\n1. Creating DecoderBlock...\n";
    DecoderBlock decoder_block(d_model, num_heads, d_ff, dropout);
    decoder_block.set_learning_rate(0.001f);

    // Create input matrices
    std::cout << "\n2. Creating input matrices...\n";

    // Decoder input (e.g., partially generated sequence)
    Matrix decoder_input(tgt_seq_len, d_model);
    decoder_input.randomize(0.1f);
    print_matrix(decoder_input, "Decoder Input");

    // Encoder output (from encoder, different sequence length)
    Matrix encoder_output(src_seq_len, d_model);
    encoder_output.randomize(0.1f);
    print_matrix(encoder_output, "Encoder Output");

    // Create causal mask for self-attention (prevent looking ahead)
    std::cout << "\n3. Creating causal mask for self-attention...\n";
    Matrix causal_mask = create_causal_mask(tgt_seq_len);
    std::cout << "Causal mask (1=attend, 0=mask):\n";
    for (int i = 0; i < tgt_seq_len; ++i) {
        for (int j = 0; j < tgt_seq_len; ++j) {
            std::cout << (int)causal_mask(i, j) << " ";
        }
        std::cout << "\n";
    }

    // No mask for cross-attention (can attend to all encoder positions)
    Matrix* cross_mask = nullptr;

    // Forward pass
    std::cout << "\n4. Forward pass through DecoderBlock...\n";
    Matrix output = decoder_block.forward(decoder_input, encoder_output,
                                          causal_mask,  // Pass by reference, not pointer
                                          cross_mask    // Still nullptr for cross-attention
    );
    print_matrix(output, "DecoderBlock Output");

    // Create gradient for backward pass
    std::cout << "\n5. Creating gradient signal...\n";
    Matrix grad_output(tgt_seq_len, d_model);
    grad_output.randomize(0.01f);
    print_matrix(grad_output, "Gradient from upstream");

    // Backward pass
    std::cout << "\n6. Backward pass through DecoderBlock...\n";
    Matrix grad_input = decoder_block.backward(grad_output);
    print_matrix(grad_input, "Gradient w.r.t. decoder input");

    // Update weights
    std::cout << "\n7. Updating weights...\n";
    decoder_block.update_weights();
    std::cout << "✓ Weights updated using accumulated gradients\n";

    // Demonstrate gradient flow
    std::cout << "\n8. Analyzing gradient flow...\n";
    float grad_input_norm = 0.0f;
    float grad_output_norm = 0.0f;

    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            grad_input_norm += grad_input(i, j) * grad_input(i, j);
            grad_output_norm += grad_output(i, j) * grad_output(i, j);
        }
    }

    grad_input_norm = std::sqrt(grad_input_norm);
    grad_output_norm = std::sqrt(grad_output_norm);

    std::cout << "  Gradient norm (output): " << grad_output_norm << "\n";
    std::cout << "  Gradient norm (input):  " << grad_input_norm << "\n";
    std::cout << "  Ratio (should be reasonable): " << grad_input_norm / grad_output_norm << "\n";

    // Save/Load demonstration
    std::cout << "\n9. Testing save/load functionality...\n";
    std::string save_path = "decoder_block_test.bin";
    decoder_block.save(save_path);
    std::cout << "✓ Saved to " << save_path << "\n";

    DecoderBlock loaded_block(d_model, num_heads, d_ff, dropout);
    loaded_block.load(save_path);
    std::cout << "✓ Loaded from " << save_path << "\n";

    // Verify loaded model produces same output
    Matrix loaded_output =
        loaded_block.forward(decoder_input, encoder_output, causal_mask, cross_mask);

    float diff = 0.0f;
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            float d = output(i, j) - loaded_output(i, j);
            diff += d * d;
        }
    }
    diff = std::sqrt(diff);

    std::cout << "  Output difference (should be ~0): " << diff << "\n";
    if (diff < 1e-5) {
        std::cout << "✓ Save/load working correctly!\n";
    } else {
        std::cout << "✗ Warning: Large difference after save/load\n";
    }

    // Summary
    std::cout << "\n=== Summary ===\n";
    std::cout << "DecoderBlock successfully demonstrated:\n";
    std::cout << "  ✓ Forward pass with masked self-attention\n";
    std::cout << "  ✓ Cross-attention to encoder output\n";
    std::cout << "  ✓ Feed-forward network\n";
    std::cout << "  ✓ Backward pass and gradient computation\n";
    std::cout << "  ✓ Weight updates\n";
    std::cout << "  ✓ Save/load functionality\n";
    std::cout << "\nKey differences from EncoderBlock:\n";
    std::cout << "  - Causal masking in self-attention (no future peeking)\n";
    std::cout << "  - Cross-attention to encoder (different seq lengths OK)\n";
    std::cout << "  - Two separate attention mechanisms\n";

    return 0;
}
