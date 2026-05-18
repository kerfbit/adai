#pragma once

#include <memory>
#include <vector>
#include "KVCache.hpp"
#include "Matrix.hpp"
#include "Optimizer.hpp"

/**
 * Cross-Attention Mechanism for Transformer Decoder
 *
 * Extends multi-head attention to support cross-attention where queries come
 * from the decoder and keys/values come from the encoder. This is essential
 * for encoder-decoder architectures like machine translation models.
 *
 * Key Difference from Self-Attention:
 * - Self-attention: Q = K = V = input
 * - Cross-attention: Q = decoder_input, K = V = encoder_output
 *
 * This allows the decoder to attend to all positions in the encoder output,
 * enabling the model to focus on relevant parts of the input sequence when
 * generating each output token.
 *
 * Architecture:
 * 1. Query projection: Q = decoder_input * W_q
 * 2. Key projection: K = encoder_output * W_k
 * 3. Value projection: V = encoder_output * W_v
 * 4. Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
 * 5. Output projection: Output = Attention * W_o
 *
 * Mathematical Formulation:
 * For decoder input X_dec ∈ ℝ^(tgt_len × d_model) and
 * encoder output X_enc ∈ ℝ^(src_len × d_model):
 * - Q = X_dec * W_q ∈ ℝ^(tgt_len × d_model)
 * - K = X_enc * W_k ∈ ℝ^(src_len × d_model)
 * - V = X_enc * W_v ∈ ℝ^(src_len × d_model)
 * - Scores = (Q * K^T) / √d_k ∈ ℝ^(tgt_len × src_len)
 * - Attention = softmax(Scores) * V ∈ ℝ^(tgt_len × d_model)
 * - Output = Attention * W_o ∈ ℝ^(tgt_len × d_model)
 */
class CrossAttention {
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
    Matrix cached_query_input;        // Query input (from decoder)
    Matrix cached_kv_input;           // Key-Value input (from encoder)
    Matrix cached_Q;                  // Projected queries
    Matrix cached_K;                  // Projected keys
    Matrix cached_V;                  // Projected values
    Matrix cached_attention_weights;  // Softmax attention weights
    Matrix cached_attention_output;   // Output after applying attention to values
    Matrix cached_scores;             // Pre-softmax attention scores

    // Optimizer support
    Optimizer* optimizer;  // Optional optimizer (nullptr = simple gradient descent)

    // Helper function for scaled dot-product attention
    /**
     * Compute scaled dot-product attention with separate Q, K, V
     *
     * @param Q Queries matrix [tgt_len, d_model]
     * @param K Keys matrix [src_len, d_model]
     * @param V Values matrix [src_len, d_model]
     * @param mask Optional attention mask [tgt_len, src_len]
     * @return Attention output [tgt_len, d_model]
     */
    Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                        const Matrix* mask);

   public:
    float learning_rate;  // Learning rate for weight updates (used when optimizer is nullptr)

    /**
     * Constructor
     *
     * Initializes the cross-attention layer with Xavier/He initialization.
     *
     * @param d_model Dimension of the model (must be divisible by num_heads)
     * @param num_heads Number of attention heads
     *
     * @throws std::invalid_argument if d_model is not divisible by num_heads
     */
    CrossAttention(int d_model, int num_heads);

    /**
     * Forward pass through cross-attention
     *
     * Computes cross-attention between decoder queries and encoder key-values.
     * Caches intermediate values for backward pass.
     *
     * @param query_input Query input from decoder [tgt_len, d_model]
     * @param kv_input Key-Value input from encoder [src_len, d_model]
     * @param mask Optional attention mask [tgt_len, src_len]
     *             Values of 0 indicate positions to mask out (set to -inf before softmax)
     *             Values of 1 indicate positions to attend to
     * @return Attention output [tgt_len, d_model]
     *
     * Process:
     * 1. Project query_input to Q using W_q
     * 2. Project kv_input to K, V using W_k, W_v
     * 3. Compute attention scores: scores = QK^T / √d_k
     * 4. Apply mask (optional)
     * 5. Apply softmax to get attention weights
     * 6. Apply attention weights to values: output = attention_weights * V
     * 7. Project through W_o
     */
    Matrix forward(const Matrix& query_input, const Matrix& kv_input, const Matrix* mask = nullptr);

    /**
     * Forward pass with KV cache support (for inference optimization)
     *
     * For cross-attention, K/V from encoder are constant across all generation steps,
     * so we compute and cache them once on the first call.
     *
     * @param query_input Query input from decoder [num_new_tokens, d_model]
     * @param kv_input Key-Value input from encoder [src_len, d_model] (used only if cache empty)
     * @param mask Optional attention mask [num_new_tokens, src_len]
     * @param kv_cache Cache for encoder K/V pairs (computed once, reused)
     * @param use_cache If true, use/populate cache
     * @return Attention output [num_new_tokens, d_model]
     */
    Matrix forward_with_cache(const Matrix& query_input, const Matrix& kv_input,
                              const Matrix* mask = nullptr, KVCache* kv_cache = nullptr,
                              bool use_cache = true);

    /**
     * Backward pass through cross-attention
     *
     * Computes gradients with respect to both inputs and weight matrices.
     * Accumulates gradients for weight updates.
     *
     * @param grad_output Gradient from upstream [tgt_len, d_model]
     * @param grad_query_input Output: Gradient w.r.t. query input [tgt_len, d_model]
     * @param grad_kv_input Output: Gradient w.r.t. key-value input [src_len, d_model]
     *
     * Returns two gradients since cross-attention has two inputs.
     */
    void backward(const Matrix& grad_output, Matrix& grad_query_input, Matrix& grad_kv_input);

    /**
     * Set optimizer for advanced optimization algorithms
     *
     * Enables use of Adam, AdamW, or other optimizers instead of
     * simple gradient descent. When set, automatically registers
     * weight matrices with the optimizer.
     *
     * @param opt Pointer to optimizer (nullptr to use simple gradient descent)
     */
    void set_optimizer(Optimizer* opt);

    /**
     * Register weight matrices with optimizer
     *
     * Called automatically by set_optimizer().
     * Registers W_q, W_k, W_v, W_o and their gradients with the optimizer.
     */
    void register_parameters();

    /**
     * Update weights using accumulated gradients
     *
     * If optimizer is set, uses optimizer->step() for advanced optimization.
     * Otherwise, applies simple gradient descent: W -= learning_rate * grad_W
     * Should be called after backward pass(es) to apply weight updates.
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
     * @return Number of attention heads
     */
    int get_num_heads() const {
        return num_heads;
    }

    // ── SafeTensors accessor API ─────────────────────────────────────────────
    const Matrix& get_Wq() const { return W_q; }
    const Matrix& get_Wk() const { return W_k; }
    const Matrix& get_Wv() const { return W_v; }
    const Matrix& get_Wo() const { return W_o; }
    void set_Wq(const Matrix& m) { W_q = m; }
    void set_Wk(const Matrix& m) { W_k = m; }
    void set_Wv(const Matrix& m) { W_v = m; }
    void set_Wo(const Matrix& m) { W_o = m; }

    /**
     * Save model weights to file
     *
     * @param filepath Path to save file
     */
    void save(const std::string& filepath) const;

    /**
     * Load model weights from file
     *
     * @param filepath Path to load file
     */
    void load(const std::string& filepath);
};
