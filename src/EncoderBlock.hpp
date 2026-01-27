#pragma once

#include <memory>
#include <string>
#include "FeedForward.hpp"
#include "LayerNorm.hpp"
#include "Matrix.hpp"
#include "MultiHeadAttention.hpp"
#include "Optimizer.hpp"

/**
 * Transformer Encoder Block
 *
 * Implements a single layer of the transformer encoder architecture,
 * combining multi-head self-attention and position-wise feed-forward
 * networks with residual connections and layer normalization.
 *
 * Architecture:
 *   Input
 *     ↓
 *   Multi-Head Attention
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
 *   attn_output = MultiHeadAttention(input, input, input, mask)
 *   residual1 = LayerNorm(input + attn_output)
 *   ff_output = FeedForward(residual1)
 *   output = LayerNorm(residual1 + ff_output)
 *
 * Features:
 *   - Self-attention mechanism for capturing dependencies
 *   - Position-wise feed-forward transformation
 *   - Residual connections for gradient flow
 *   - Layer normalization for training stability
 *   - Optional dropout for regularization
 *   - Full backpropagation support
 *   - Gradient accumulation and clipping
 */
class EncoderBlock {
   private:
    // Core components
    std::unique_ptr<MultiHeadAttention> attention;
    std::unique_ptr<FeedForward> feed_forward;
    std::unique_ptr<LayerNorm> norm1;
    std::unique_ptr<LayerNorm> norm2;

    // Hyperparameters
    int d_model;
    int num_heads;
    int d_ff;
    float dropout_rate;

    // Cached values for backward pass
    Matrix cached_input;
    Matrix cached_attn_output;
    Matrix cached_residual1;
    Matrix cached_normed1;
    Matrix cached_ff_output;
    Matrix cached_residual2;

   public:
    float learning_rate;  // Learning rate for gradient updates

    /**
     * Constructor
     *
     * @param d_model Model dimension (embedding size)
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward network hidden dimension
     * @param dropout Dropout rate for regularization (default: 0.1)
     */
    EncoderBlock(int d_model, int num_heads, int d_ff, float dropout = 0.1f);

    /**
     * Forward pass through encoder block
     *
     * Applies:
     *   1. Multi-head self-attention
     *   2. Residual connection and layer norm
     *   3. Feed-forward network
     *   4. Residual connection and layer norm
     *
     * @param input Input matrix [seq_len x d_model]
     * @param mask Optional attention mask [seq_len x seq_len] or nullptr
     * @return Output matrix [seq_len x d_model]
     */
    Matrix forward(const Matrix& input, const Matrix* mask = nullptr);

    /**
     * Backward pass through encoder block
     *
     * Computes gradients for all parameters and returns gradient w.r.t. input.
     * Gradients flow through:
     *   1. Second layer norm (backward)
     *   2. Second residual connection (splits gradient)
     *   3. Feed-forward network (backward)
     *   4. First layer norm (backward)
     *   5. First residual connection (splits gradient)
     *   6. Multi-head attention (backward)
     *
     * @param grad_output Gradient from next layer [seq_len x d_model]
     * @return Gradient w.r.t. input [seq_len x d_model]
     */
    Matrix backward(const Matrix& grad_output);

    /**
     * Update weights using accumulated gradients
     *
     * Applies gradient descent to all learnable parameters:
     *   - Attention weights (Q, K, V, output projection)
     *   - Feed-forward weights (W1, W2, biases)
     *   - Layer norm parameters (gamma, beta)
     *
     * Automatically zeros gradients after update.
     */
    void update_weights();

    /**
     * Zero all accumulated gradients
     *
     * Resets gradients in:
     *   - Multi-head attention
     *   - Feed-forward network
     *   - Both layer normalization layers
     */
    void zero_grad();

    /**
     * Get the L2 norm of all gradients combined
     *
     * Useful for:
     *   - Monitoring gradient flow
     *   - Detecting vanishing/exploding gradients
     *   - Deciding when to apply gradient clipping
     *
     * @return L2 norm of concatenated gradients
     */
    float get_gradient_norm() const;

    /**
     * Clip gradients to prevent explosion
     *
     * If gradient norm exceeds max_norm, scales all gradients
     * to have norm equal to max_norm.
     *
     * @param max_norm Maximum allowed gradient norm
     */
    void clip_gradients(float max_norm);

    /**
     * Save encoder block weights to binary file
     *
     * Saves all learnable parameters from:
     *   - Multi-head attention
     *   - Feed-forward network
     *   - Layer normalization layers
     *
     * @param filename Path to output file
     */
    void save_weights(const std::string& filename) const;

    /**
     * Load encoder block weights from binary file
     *
     * @param filename Path to input file
     * @throws std::runtime_error if file doesn't exist or dimensions mismatch
     */
    void load_weights(const std::string& filename);

    /**
     * Print encoder block configuration
     *
     * @param name Optional name for the block (e.g., "EncoderBlock_0")
     */
    void print_config(const std::string& name = "EncoderBlock") const;

    /**
     * Get model dimension
     *
     * @return d_model
     */
    int get_d_model() const {
        return d_model;
    }

    /**
     * Get number of attention heads
     *
     * @return num_heads
     */
    int get_num_heads() const {
        return num_heads;
    }

    /**
     * Get feed-forward dimension
     *
     * @return d_ff
     */
    int get_d_ff() const {
        return d_ff;
    }

    /**
     * Get dropout rate
     *
     * @return dropout_rate
     */
    float get_dropout_rate() const {
        return dropout_rate;
    }

    /**
     * Register all encoder block parameters with optimizer
     *
     * @param optimizer Optimizer to register parameters with
     */
    void register_parameters_with_optimizer(class Optimizer& optimizer);
};
