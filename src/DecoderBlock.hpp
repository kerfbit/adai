#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07


#include <memory>
#include <string>
#include <utility>
#include "CrossAttention.hpp"
#include "FeedForward.hpp"
#include "KVCache.hpp"
#include "LayerNorm.hpp"
#include "Matrix.hpp"
#include "MultiHeadAttention.hpp"
#include "Optimizer.hpp"
#ifdef ADAI_ENABLE_GPU
#include "gpu/MatrixGPU.hpp"
#endif

/**
 * Transformer Decoder Block
 *
 * Implements a single layer of the transformer decoder architecture,
 * combining masked multi-head self-attention, cross-attention to encoder,
 * and position-wise feed-forward networks with residual connections
 * and layer normalization.
 *
 * Architecture (Pre-LN — normalization happens before each sublayer, not
 * after; the residual stream itself is never normalized inside the block,
 * which is why LLMDecoder applies a final LayerNorm once after the last
 * block. This placement is more stable for deep stacks than Post-LN):
 *   Input
 *     ↓
 *   Norm -> Masked Self-Attention (causal) -> Add (residual)
 *     ↓
 *   Norm -> Cross-Attention (to encoder output) -> Add (residual)
 *     ↓
 *   Norm -> Feed-Forward Network -> Add (residual)
 *     ↓
 *   Output (unnormalized)
 *
 * Mathematical Operations:
 *   self_attn_output = MultiHeadAttention(LayerNorm(input), causal_mask)
 *   residual1 = input + self_attn_output
 *   cross_attn_output = CrossAttention(LayerNorm(residual1), encoder, mask)
 *   residual2 = residual1 + cross_attn_output
 *   ff_output = FeedForward(LayerNorm(residual2))
 *   output = residual2 + ff_output
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
    float learning_rate{0.001f};

    /**
     * Get the feed-forward sublayer (for hook registration, diagnostics, etc.)
     *
     * @return Pointer to the internal FeedForward instance
     */
    FeedForward* get_feed_forward() {
        return feed_forward.get();
    }

    /**
     * Get the masked self-attention sublayer (for hook registration, entropy tracking, etc.)
     *
     * @return Pointer to the internal MultiHeadAttention instance
     */
    MultiHeadAttention* get_self_attention() {
        return self_attention.get();
    }

    // ── SafeTensors accessor API ─────────────────────────────────────────────
    CrossAttention* get_cross_attention() {
        return cross_attention.get();
    }
    LayerNorm* get_norm1() {
        return norm1.get();
    }
    LayerNorm* get_norm2() {
        return norm2.get();
    }
    LayerNorm* get_norm3() {
        return norm3.get();
    }

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
     * Forward pass with KV cache support (for inference optimization)
     *
     * Same as forward() but with caching for autoregressive generation.
     *
     * @param input Decoder input (new tokens only) [num_new_tokens, d_model]
     * @param encoder_output Encoder output for cross-attention [enc_seq_len, d_model]
     * @param self_attn_mask Causal mask [num_new_tokens, total_seq_len]
     * @param self_attn_cache Cache for self-attention K/V pairs
     * @param cross_attn_cache Cache for cross-attention K/V pairs (computed once from encoder)
     * @param cross_attn_mask Optional padding mask for encoder
     * @param use_cache If true, update caches
     * @return Output [num_new_tokens, d_model]
     */
    Matrix forward_with_cache(const Matrix& input, const Matrix& encoder_output,
                              const Matrix& self_attn_mask, KVCache* self_attn_cache,
                              KVCache* cross_attn_cache, const Matrix* cross_attn_mask = nullptr,
                              bool use_cache = true);

    /**
     * Backward pass through decoder block
     *
     * Computes gradients for all parameters and returns gradient w.r.t. input.
     * Gradients flow through:
     *   1. Third residual connection (split gradient)
     *   2. Feed-forward network (backward)
     *   3. Third layer norm (backward)
     *   4. Second residual connection (split gradient)
     *   5. Cross-attention (backward)
     *   6. Second layer norm (backward)
     *   7. First residual connection (split gradient)
     *   8. Self-attention (backward)
     *   9. First layer norm (backward)
     *
     * @param grad_output Gradient from next layer [seq_len, d_model]
     * @param grad_encoder_output Out-param: gradient w.r.t. this block's cross-attention
     *   encoder input (K/V source), overwritten (not accumulated) by this call.
     * @return Gradient w.r.t. input [seq_len, d_model]
     */
    Matrix backward(const Matrix& grad_output, Matrix& grad_encoder_output);

    /** Convenience overload: discards the encoder-side gradient. */
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
     * Compute the L2 norm of accumulated gradients across self-attention,
     * cross-attention, and feed-forward sub-components (combined in
     * quadrature). LayerNorm gradients are omitted, matching
     * EncoderBlock::get_gradient_norm()'s approximation.
     *
     * @return sqrt(sum of squared per-component gradient norms)
     */
    float get_gradient_norm() const;

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

    /**
     * Register all decoder block parameters with optimizer
     *
     * @param optimizer Optimizer to register parameters with
     */
    void register_parameters_with_optimizer(class Optimizer& optimizer);

#ifdef ADAI_ENABLE_GPU
    void gpu_upload_weights();
    void gpu_download_grads();
    void gpu_zero_grads();
    // encoder_out: cached GPU encoder output [src_len, d_model]
    adai::gpu::GPUMatrix gpu_forward(const adai::gpu::GPUMatrix& input,
                                     const adai::gpu::GPUMatrix& encoder_out,
                                     const adai::gpu::GPUMatrix* self_mask = nullptr);
    // Returns {grad_input, grad_encoder_output} — gradient w.r.t. decoder input,
    // and gradient w.r.t. this block's cross-attention encoder input.
    std::pair<adai::gpu::GPUMatrix, adai::gpu::GPUMatrix> gpu_backward(
        const adai::gpu::GPUMatrix& dout);
#endif
};
