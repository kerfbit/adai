#pragma once

// @adai-status: beta        (capped by TD-050 — see TECHNICAL_DEBT.md)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-08


#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "DecoderBlock.hpp"
#ifdef ADAI_ENABLE_GPU
#include "gpu/MatrixGPU.hpp"
#endif
#include "KVCache.hpp"
#include "LayerNorm.hpp"
#include "Matrix.hpp"
#include "Optimizer.hpp"
#include "PositionalEncoding.hpp"
#include "TokenEmbedding.hpp"

/**
 * LLM Decoder for autoregressive text generation
 *
 * This decoder implements a transformer-based architecture for generating text
 * autoregressively, suitable for chatbot responses, machine translation, and
 * other sequence-to-sequence tasks.
 *
 * Features:
 * - Token embeddings with positional encoding
 * - Multi-layer transformer decoder with causal masking
 * - Masked self-attention for autoregressive generation
 * - Cross-attention to encoder outputs (optional for encoder-decoder models)
 * - Layer normalization and residual connections
 * - Support for both decoder-only and encoder-decoder architectures
 *
 * Architecture:
 * Input Tokens → Token Embedding → Positional Encoding →
 * Decoder Block 1 → ... → Decoder Block N → Final LayerNorm → Output
 *
 * Each DecoderBlock contains:
 * - Masked self-attention (causal)
 * - Cross-attention (if encoder outputs provided)
 * - Feed-forward network
 * - Residual connections and layer normalization
 */
class LLMDecoder {
   private:
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> positional_encoding;
    std::vector<std::unique_ptr<DecoderBlock>> decoder_blocks;
    std::unique_ptr<LayerNorm> final_norm;

    int vocab_size;
    int d_model;
    int num_layers;
    int num_heads;
    int d_ff;
    int max_seq_length;

    // Training state
    bool requires_grad{true};
    float learning_rate{0.001};

    // Cached values for backward pass
    std::vector<int> cached_token_ids;
    Matrix cached_embeddings;
    Matrix cached_pos_encoded;
    std::vector<Matrix> cached_decoder_outputs;
    Matrix cached_encoder_output;  // For cross-attention

    /**
     * Create causal mask for autoregressive generation
     * Prevents positions from attending to future positions
     *
     * @param seq_length Length of the sequence
     * @return Matrix [seq_length, seq_length] with 1s on/below diagonal, 0s above
     */
    static Matrix create_causal_mask(int seq_length);

   public:
    /**
     * Constructor for LLM Decoder
     *
     * @param vocab_size Size of the vocabulary
     * @param d_model Dimension of the model embeddings
     * @param num_layers Number of decoder layers
     * @param num_heads Number of attention heads
     * @param d_ff Dimension of feed-forward layer
     * @param max_seq_length Maximum sequence length
     */
    LLMDecoder(int vocab_size, int d_model = 512, int num_layers = 6, int num_heads = 8,
               int d_ff = 2048, int max_seq_length = 512);

    /**
     * Destructor
     */
    ~LLMDecoder();
    LLMDecoder(const LLMDecoder&) = delete;
    LLMDecoder& operator=(const LLMDecoder&) = delete;
    LLMDecoder(LLMDecoder&&) = delete;
    LLMDecoder& operator=(LLMDecoder&&) = delete;

    /**
     * Forward pass through decoder (decoder-only mode, no cross-attention)
     *
     * @param token_ids Vector of token IDs [sequence_length]
     * @return Matrix of shape [sequence_length, d_model]
     */
    Matrix forward(const std::vector<int>& token_ids);

    /**
     * Forward pass with encoder outputs (encoder-decoder mode)
     *
     * @param token_ids Vector of token IDs [sequence_length]
     * @param encoder_output Matrix from encoder [encoder_seq_len, d_model]
     * @return Matrix of shape [sequence_length, d_model]
     */
    Matrix forward_with_encoder(const std::vector<int>& token_ids, const Matrix& encoder_output);

    /**
     * Forward pass with custom causal mask
     *
     * @param token_ids Vector of token IDs
     * @param causal_mask Custom causal mask matrix
     * @param encoder_output Optional encoder output for cross-attention
     * @return Matrix of shape [sequence_length, d_model]
     */
    Matrix forward_with_mask(const std::vector<int>& token_ids, const Matrix& causal_mask,
                             const Matrix* encoder_output = nullptr);

    /**
     * Forward pass with KV cache support (for inference optimization)
     *
     * Optimized for autoregressive generation. Uses cached key-value pairs
     * from previous generation steps to avoid redundant computation.
     *
     * @param token_ids Vector of new token IDs [num_new_tokens] (typically 1 during generation)
     * @param kv_cache Multi-layer KV cache structure
     * @param encoder_output Optional encoder output for cross-attention
     * @param use_cache If true, update cache with new K/V pairs
     * @return Matrix of shape [num_new_tokens, d_model]
     *
     * Performance: ~2-3x speedup for long sequences
     * Usage:
     *   1. First call: cache is empty, computes all positions
     *   2. Subsequent calls: only computes new token, reuses cache
     *   3. Clear cache when starting new sequence
     */
    Matrix forward_with_cache(const std::vector<int>& token_ids, DecoderKVCache& kv_cache,
                              const Matrix* encoder_output = nullptr, bool use_cache = true);

    /**
     * Backward pass for training
     *
     * @param grad_output Gradient from loss function [seq_length, d_model]
     * @param grad_encoder_output Out-param: gradient w.r.t. encoder_output, summed
     *   across every decoder block's cross-attention (the same encoder_output feeds
     *   all of them).
     */
    void backward(const Matrix& grad_output, Matrix& grad_encoder_output);

    /** Convenience overload: discards the encoder-side gradient. */
    void backward(const Matrix& grad_output);

    /**
     * Update weights using accumulated gradients
     *
     * @param learning_rate Learning rate for gradient descent
     */
    void update_weights(float learning_rate);

    /**
     * Set training mode
     *
     * @param mode True for training, false for inference
     */
    void set_training(bool mode);

    /**
     * Set learning rate
     *
     * @param lr New learning rate
     */
    void set_learning_rate(float lr);

    /**
     * Get model dimension
     *
     * @return Model dimension d_model
     */
    int get_d_model() const {
        return d_model;
    }

    /**
     * Get vocabulary size
     *
     * @return Vocabulary size
     */
    int get_vocab_size() const {
        return vocab_size;
    }

    /**
     * Get number of layers
     *
     * @return Number of decoder layers
     */
    int get_num_layers() const {
        return num_layers;
    }

    /**
     * Get maximum sequence length
     *
     * @return Maximum sequence length
     */
    int get_max_seq_length() const {
        return max_seq_length;
    }

    /**
     * Save decoder weights to file
     *
     * @param filepath Path to save file
     */
    void save_weights(const std::string& filepath) const;

    /**
     * Load decoder weights from file
     *
     * @param filepath Path to load file
     */
    void load_weights(const std::string& filepath);

    /**
     * Get embedding layer (for weight sharing with output projection)
     *
     * @return Pointer to token embedding
     */
    TokenEmbedding* get_token_embedding() {
        return token_embedding.get();
    }

    /**
     * Get decoder block at specific layer
     *
     * @param layer Layer index
     * @return Pointer to decoder block
     */
    DecoderBlock* get_decoder_block(int layer) {
        if (layer < 0 || layer >= num_layers) {
            throw std::out_of_range("Layer index out of range");
        }
        return decoder_blocks[layer].get();
    }

    // ── SafeTensors accessor API ─────────────────────────────────────────────
    LayerNorm* get_final_norm() {
        return final_norm.get();
    }

    /**
     * Zero all gradients
     */
    void zero_grad();

    /**
     * Get the last cached decoder output (for debugging)
     *
     * @return Last decoder output matrix
     */
    Matrix get_last_output() const {
        if (cached_decoder_outputs.empty()) {
            return {0, 0};
        }
        return cached_decoder_outputs.back();
    }

    /**
     * Register all decoder parameters with optimizer
     *
     * @param optimizer Optimizer to register parameters with
     */
    void register_parameters_with_optimizer(Optimizer& optimizer);

#ifdef ADAI_ENABLE_GPU
    void gpu_upload_weights();
    void gpu_download_grads();
    void gpu_zero_grads();
    /**
     * GPU decode: embed + positional encode on CPU, then run all decoder blocks on GPU.
     * @param token_ids     Decoder input token IDs
     * @param encoder_out   GPU-resident encoder output [src_len, d_model]
     * @return GPU matrix [tgt_len, d_model] (decoder output, before lm_head)
     */
    adai::gpu::GPUMatrix gpu_decode(const std::vector<int>& token_ids,
                                    const adai::gpu::GPUMatrix& encoder_out);
    /**
     * GPU backward through decoder blocks (lm_head grad passed in as dout).
     * @param dout  Upstream gradient [tgt_len, d_model]
     * @return {grad w.r.t. decoder embedding output, grad w.r.t. encoder_output
     *   summed across every decoder block's cross-attention}
     */
    std::pair<adai::gpu::GPUMatrix, adai::gpu::GPUMatrix> gpu_backward(
        const adai::gpu::GPUMatrix& dout);

    LayerNorm* get_final_norm_dec() {
        return final_norm.get();
    }
#endif
};
