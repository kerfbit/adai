#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "KVCache.hpp"
#include "Matrix.hpp"
#include "Optimizer.hpp"

/**
 * Multi-Head Self-Attention Mechanism
 *
 * Implements the multi-head attention mechanism from "Attention is All You Need".
 * This is a core component of transformer architectures, allowing the model to
 * jointly attend to information from different representation subspaces.
 *
 * Key Features:
 * - Parallel attention heads for diverse representation learning
 * - Scaled dot-product attention
 * - Linear projections for queries, keys, values, and output
 * - Gradient computation for backpropagation
 * - Optional attention masking (e.g., for padding or causal attention)
 *
 * Architecture:
 * 1. Linear projections: Q = XW_q, K = XW_k, V = XW_v
 * 2. Split into num_heads: Each head processes d_k dimensions
 * 3. Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
 * 4. Concatenate heads and apply output projection: Output = Concat(heads)W_o
 *
 * Mathematical Formulation:
 * For input X ∈ ℝ^(seq_len × d_model):
 * - Q = XW_q, K = XW_k, V = XW_v  (each ∈ ℝ^(seq_len × d_model))
 * - Split into h heads: Q_i, K_i, V_i ∈ ℝ^(seq_len × d_k) where d_k = d_model/h
 * - Attention_i = softmax((Q_iK_i^T)/√d_k)V_i
 * - Output = Concat(Attention_1, ..., Attention_h)W_o
 *
 * Implementation Note:
 * This implementation uses a simplified approach where we don't explicitly
 * split and concatenate heads, but instead process the full d_model dimension
 * and rely on the learned weight matrices to capture multi-head behavior.
 */
/// Callback invoked after softmax in every forward() pass, receiving the
/// attention weight matrix [seq_len × seq_len].
using AttentionHookFn = std::function<void(const Matrix&)>;

class MultiHeadAttention {
   private:
    // Model dimensions
    int d_model;    // Model dimension (embedding size)
    int num_heads;  // Number of attention heads
    int d_k;        // Dimension per head (d_model / num_heads)

    // Learnable weight matrices
    Matrix W_q;  // Query projection [d_model, d_model]
    Matrix W_k;  // Key projection [d_model, d_model]
    Matrix W_v;  // Value projection [d_model, d_model]
    Matrix W_o;  // Output projection [d_model, d_model]

    // Gradient matrices
    Matrix W_q_grad;  // Accumulated gradients for W_q
    Matrix W_k_grad;  // Accumulated gradients for W_k
    Matrix W_v_grad;  // Accumulated gradients for W_v
    Matrix W_o_grad;  // Accumulated gradients for W_o

    // Cached values for backward pass
    Matrix cached_input;              // Input to forward pass
    Matrix cached_Q;                  // Projected queries
    Matrix cached_K;                  // Projected keys
    Matrix cached_V;                  // Projected values
    Matrix cached_attention_weights;  // Softmax attention weights
    Matrix cached_attention_output;   // Output after applying attention to values
    Matrix cached_scores;             // Pre-softmax attention scores

    // Optimizer for weight updates
    Optimizer* optimizer;  // Pointer to optimizer (nullptr means use simple gradient descent)

    // Attention hook (for entropy tracking, TD-013)
    AttentionHookFn attention_hook_;

    // Helper function for scaled dot-product attention
    /**
     * Compute scaled dot-product attention
     *
     * @param Q Queries matrix [seq_len, d_model]
     * @param K Keys matrix [seq_len, d_model]
     * @param V Values matrix [seq_len, d_model]
     * @param mask Optional attention mask [seq_len, seq_len]
     * @return Attention output [seq_len, d_model]
     */
    Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                        const Matrix* mask);

   public:
    float learning_rate;  // Learning rate for weight updates

    /**
     * Constructor
     *
     * Initializes the multi-head attention layer with Xavier/He initialization.
     *
     * @param d_model Dimension of the model (must be divisible by num_heads)
     * @param num_heads Number of attention heads
     *
     * @throws std::invalid_argument if d_model is not divisible by num_heads
     */
    MultiHeadAttention(int d_model, int num_heads);

    /**
     * Forward pass through multi-head attention
     *
     * Computes multi-head self-attention on the input sequence.
     * Caches intermediate values for backward pass.
     *
     * @param input Input matrix [seq_len, d_model]
     * @param mask Optional attention mask [seq_len, seq_len]
     *             Values of 0 indicate positions to mask out (set to -inf before softmax)
     *             Values of 1 indicate positions to attend to
     * @return Attention output [seq_len, d_model]
     *
     * Process:
     * 1. Project input to Q, K, V using W_q, W_k, W_v
     * 2. Compute attention scores: scores = QK^T / √d_k
     * 3. Apply mask (optional)
     * 4. Apply softmax to get attention weights
     * 5. Apply attention weights to values: output = attention_weights * V
     * 6. Project through W_o
     */
    Matrix forward(const Matrix& input, const Matrix* mask = nullptr);

    /**
     * Forward pass with parallel attention head computation
     *
     * Properly splits input into num_heads and processes each head independently
     * using OpenMP parallelization. This provides 2-4x speedup for multi-head attention.
     *
     * @param input Input matrix [seq_len, d_model]
     * @param mask Optional attention mask [seq_len, seq_len]
     * @param use_parallel Enable/disable parallel computation (default: true)
     * @return Attention output [seq_len, d_model]
     *
     * Implementation:
     * 1. Project input to Q, K, V
     * 2. Split Q, K, V into num_heads parts (each of dimension d_k)
     * 3. Compute attention for each head IN PARALLEL using OpenMP
     * 4. Concatenate head outputs
     * 5. Apply output projection
     */
    Matrix forward_parallel(const Matrix& input, const Matrix* mask = nullptr,
                            bool use_parallel = true);

    /**
     * Forward pass with KV cache support (for inference optimization)
     *
     * Enables caching of key-value pairs during autoregressive generation.
     * In subsequent calls, only computes K/V for new tokens and reuses cached values.
     *
     * @param input Input matrix [num_new_tokens, d_model]
     * @param mask Optional attention mask [num_new_tokens, total_seq_len]
     * @param kv_cache Pointer to KVCache structure (nullptr = no caching)
     * @param use_cache If true and kv_cache provided, update cache with new K/V
     * @return Attention output [num_new_tokens, d_model]
     *
     * Cache Behavior:
     * - First call (empty cache): Compute K/V for all tokens, store in cache
     * - Subsequent calls: Compute K/V for new token only, concatenate with cache
     * - Attention computed over all tokens (cached + new)
     *
     * Performance: ~2-3x speedup for long sequences
     */
    Matrix forward_with_cache(const Matrix& input, const Matrix* mask = nullptr,
                              KVCache* kv_cache = nullptr, bool use_cache = true);

    /**
     * Backward pass through multi-head attention
     *
     * Computes gradients with respect to input and weight matrices.
     * Updates weights using accumulated gradients.
     *
     * @param grad_output Gradient from upstream [seq_len, d_model]
     * @return Gradient with respect to input [seq_len, d_model]
     *
     * Note: This function both accumulates gradients AND updates weights.
     * Call zero_grad() before the next forward pass.
     */
    Matrix backward(const Matrix& grad_output);

    /**
     * Set optimizer for weight updates
     *
     * Registers all weight matrices with the optimizer.
     * If optimizer is nullptr, falls back to simple gradient descent.
     *
     * @param opt Pointer to optimizer instance
     */
    void set_optimizer(Optimizer* opt);

    /**
     * Register all parameters with the optimizer
     *
     * Should be called after set_optimizer() or when optimizer changes.
     */
    void register_parameters();

    /**
     * Update weights using accumulated gradients
     *
     * Uses optimizer if available, otherwise falls back to simple gradient descent.
     * Zeros out gradients after update.
     */
    void update_weights();

    /**
     * Zero out all accumulated gradients
     *
     * Resets all gradient matrices to zero.
     * Should be called before each forward/backward pass in training.
     */
    void zero_grad();

    /**
     * Get the model dimension
     *
     * @return Model dimension (d_model)
     */
    int get_d_model() const {
        return d_model;
    }

    /**
     * Get the number of attention heads
     *
     * @return Number of heads
     */
    int get_num_heads() const {
        return num_heads;
    }

    /**
     * Get dimension per head
     *
     * @return Dimension per head (d_k = d_model / num_heads)
     */
    int get_d_k() const {
        return d_k;
    }

    /**
     * Get the last computed attention weights
     *
     * Useful for visualization and interpretation.
     *
     * @return Attention weights matrix [seq_len, seq_len]
     */
    const Matrix& get_attention_weights() const {
        return cached_attention_weights;
    }

    /**
     * Register a callback fired after softmax in every forward() pass.
     * The callback receives the attention weight matrix [seq_len × seq_len].
     * Replaces any previously registered hook.
     */
    void set_attention_hook(AttentionHookFn fn) {
        attention_hook_ = std::move(fn);
    }

    /**
     * Remove any registered attention hook.
     */
    void clear_attention_hook() {
        attention_hook_ = nullptr;
    }

    /**
     * Print configuration
     *
     * @param name Optional name for the layer
     */
    void print_config(const std::string& name = "MultiHeadAttention") const;

    /**
     * Save weights to file
     *
     * @param filename Path to save weights
     */
    void save_weights(const std::string& filename) const;

    /**
     * Load weights from file
     *
     * @param filename Path to load weights from
     * @throws std::runtime_error if file cannot be opened or dimensions mismatch
     */
    void load_weights(const std::string& filename);

    /**
     * Get gradient norm for monitoring
     *
     * @return L2 norm of all gradients
     */
    float get_gradient_norm() const;

    /**
     * Clip gradients by norm
     *
     * @param max_norm Maximum allowed gradient norm
     */
    void clip_gradients(float max_norm);
};
