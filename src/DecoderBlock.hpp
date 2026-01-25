#pragma once

#include <memory>
#include <string>
#include "CrossAttention.hpp"
#include "FeedForward.hpp"
#include "LayerNorm.hpp"
#include "Matrix.hpp"
#include "MultiHeadAttention.hpp"

/**
 * Transformer Decoder Block
 *
 * Implements a single layer of the transformer decoder architecture,
 * combining masked multi-head self-attention, cross-attention to encoder,
 * and position-wise feed-forward networks with residual connections
 * and layer normalization.
 *
 * Architecture:
 *   Input
 *     ↓
 *   Masked Self-Attention (causal)
 *     ↓
 *   Add & Norm (residual + layer norm)
 *     ↓
 *   Cross-Attention (to encoder output)
 *     ↓
 *   Add & Norm (residual + layer norm)
 *     ↓
 *   Feed-Forward Network
 *     ↓
 *   Add & Norm (residual + layer norm)
 *     ↓
 *   Output
 *
 * Mathematical Operations:
 *   self_attn_output = MultiHeadAttention(input, input, input, causal_mask)
 *   residual1 = LayerNorm(input + self_attn_output)
 *   cross_attn_output = MultiHeadAttention(residual1, encoder, encoder, mask)
 *   residual2 = LayerNorm(residual1 + cross_attn_output)
 *   ff_output = FeedForward(residual2)
 *   output = LayerNorm(residual2 + ff_output)
 *
 * Features:
 *   - Causal self-attention (prevents attending to future)
 *   - Cross-attention to encoder output
 *   - Position-wise feed-forward transformation
 *   - Three residual connections for gradient flow
 *   - Three layer normalizations for training stability
 *   - Full backpropagation support
 */
class DecoderBlock {
   private:
    // Core components
    std::unique_ptr<MultiHeadAttention> self_attention;  // Masked self-attention
    std::unique_ptr<CrossAttention> cross_attention;     // Attend to encoder (new!)
    std::unique_ptr<FeedForward> feed_forward;
    std::unique_ptr<LayerNorm> norm1;  // After self-attention
    std::unique_ptr<LayerNorm> norm2;  // After cross-attention
    std::unique_ptr<LayerNorm> norm3;  // After feed-forward

    // Hyperparameters
    int d_model;
    int num_heads;
    int d_ff;
    float dropout_rate;

    // Cached values for backward pass
    Matrix cached_input;
    Matrix cached_self_attn_output;
    Matrix cached_residual1;
    Matrix cached_normed1;
    Matrix cached_cross_attn_output;
    Matrix cached_residual2;
    Matrix cached_normed2;
    Matrix cached_ff_output;
    Matrix cached_residual3;
    Matrix cached_encoder_output;

   public:
    float learning_rate;

    /**
     * Constructor
     *
     * @param d_model Model dimension (embedding size)
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward network hidden dimension
     * @param dropout Dropout rate for regularization (default: 0.1)
     */
    DecoderBlock(int d_model, int num_heads, int d_ff, float dropout = 0.1f);

    /**
     * Forward pass through decoder block
     *
     * Applies:
     *   1. Masked multi-head self-attention (causal)
     *   2. Residual connection and layer norm
     *   3. Cross-attention to encoder output
     *   4. Residual connection and layer norm
     *   5. Feed-forward network
     *   6. Residual connection and layer norm
     *
     * @param input Decoder input [seq_len, d_model]
     * @param encoder_output Encoder output for cross-attention [enc_seq_len, d_model]
     * @param self_attn_mask Causal mask for self-attention [seq_len, seq_len]
     * @param cross_attn_mask Optional padding mask for encoder [seq_len, enc_seq_len]
     * @return Output [seq_len, d_model]
     */
    Matrix forward(const Matrix& input, const Matrix& encoder_output, const Matrix& self_attn_mask,
                   const Matrix* cross_attn_mask = nullptr);

    /**
     * Backward pass through decoder block
     *
     * Computes gradients for all parameters and returns gradient w.r.t. input.
     * Gradients flow through:
     *   1. Third layer norm (backward)
     *   2. Third residual connection (split gradient)
     *   3. Feed-forward network (backward)
     *   4. Second layer norm (backward)
     *   5. Second residual connection (split gradient)
     *   6. Cross-attention (backward)
     *   7. First layer norm (backward)
     *   8. First residual connection (split gradient)
     *   9. Self-attention (backward)
     *
     * @param grad_output Gradient from next layer [seq_len, d_model]
     * @return Gradient w.r.t. input [seq_len, d_model]
     */
    Matrix backward(const Matrix& grad_output);

    /**
     * Update weights using accumulated gradients
     *
     * Updates all sub-components: attention layers, feed-forward, layer norms
     */
    void update_weights();

    /**
     * Zero accumulated gradients
     *
     * Clears gradients in all sub-components
     */
    void zero_grad();

    /**
     * Set learning rate for all sub-components
     *
     * @param lr New learning rate
     */
    void set_learning_rate(float lr);

    /**
     * Save decoder block parameters to file
     *
     * @param filepath Path to save file
     */
    void save(const std::string& filepath);

    /**
     * Load decoder block parameters from file
     *
     * @param filepath Path to load file
     */
    void load(const std::string& filepath);
};
