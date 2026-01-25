#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "Activation.hpp"
#include "BPETokenizer.hpp"
#include "EncoderBlock.hpp"
#include "FeedForward.hpp"
#include "LayerNorm.hpp"
#include "Matrix.hpp"
#include "MultiHeadAttention.hpp"
#include "PositionalEncoding.hpp"
#include "TokenEmbedding.hpp"

/**
 * Main LLM Encoder for chatbot applications
 *
 * This encoder implements a transformer-based architecture for encoding text
 * into contextualized representations suitable for chatbot and NLP tasks.
 *
 * Features:
 * - BPE tokenization for subword handling
 * - Token embeddings with positional encoding
 * - Multi-layer transformer encoder
 * - Multi-head self-attention
 * - Layer normalization and residual connections
 * - Sentence-level embedding pooling
 */
class LLMEncoder {
   private:
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> positional_encoding;
    std::vector<std::unique_ptr<EncoderBlock>> encoder_blocks;
    std::unique_ptr<LayerNorm> final_norm;

    int vocab_size;
    int d_model;
    int num_layers;
    int num_heads;
    int d_ff;
    int max_seq_length;

    // Training state
    bool requires_grad;
    float learning_rate;

    // Cached values for backward pass
    std::vector<int> cached_token_ids;
    Matrix cached_embeddings;
    Matrix cached_pos_encoded;
    std::vector<Matrix> cached_encoder_outputs;

   public:
    /**
     * Constructor for LLM Encoder
     *
     * @param vocab_size Size of the vocabulary
     * @param d_model Dimension of the model embeddings
     * @param num_layers Number of encoder layers
     * @param num_heads Number of attention heads
     * @param d_ff Dimension of feed-forward layer
     * @param max_seq_length Maximum sequence length
     */
    LLMEncoder(int vocab_size, int d_model = 512, int num_layers = 6, int num_heads = 8,
               int d_ff = 2048, int max_seq_length = 512);

    /**
     * Encode text to contextualized embeddings
     *
     * @param text Input text to encode
     * @return Matrix of shape [sequence_length, d_model]
     */
    Matrix encode(const std::string& text);

    /**
     * Encode with attention mask for padding
     *
     * @param token_ids Vector of token IDs
     * @param attention_mask Attention mask matrix
     * @return Matrix of shape [sequence_length, d_model]
     */
    Matrix encode_with_mask(const std::vector<int>& token_ids, const Matrix& attention_mask);

    /**
     * Get mean pooling of encoded sequence for sentence-level representation
     *
     * @param text Input text
     * @return Vector of size d_model representing the sentence
     */
    std::vector<float> get_sentence_embedding(const std::string& text);

    /**
     * Load tokenizer vocabulary from file
     *
     * @param vocab_file Path to vocabulary file
     */
    void load_tokenizer_vocab(const std::string& vocab_file);

    /**
     * Build tokenizer from text corpus
     *
     * @param corpus Vector of training texts
     * @param vocab_size Target vocabulary size
     */
    void build_tokenizer(const std::vector<std::string>& corpus, int vocab_size = 10000);

    /**
     * Print model configuration
     */
    void print_config() const;

    /**
     * Get the embedding dimension
     *
     * @return Dimension of embeddings (d_model)
     */
    int get_embedding_dim() const;

    /**
     * Save model weights to file
     *
     * @param filename Path to save weights
     */
    void save_weights(const std::string& filename);

    /**
     * Load model weights from file
     *
     * @param filename Path to load weights from
     */
    void load_weights(const std::string& filename);

    /**
     * Set whether gradients should be computed
     *
     * @param requires_grad If true, enables gradient computation
     */
    void set_requires_grad(bool requires_grad);

    /**
     * Set learning rate for gradient updates
     *
     * @param lr Learning rate
     */
    void set_learning_rate(float lr);

    /**
     * Backward pass through encoder
     *
     * @param grad_output Gradient of loss w.r.t. encoder output
     */
    void backward(const Matrix& grad_output);

    /**
     * Zero all gradients
     */
    void zero_grad();

    /**
     * Get sentence embedding with gradient support
     *
     * @param text Input text
     * @return Sentence embedding vector
     */
    std::vector<float> get_sentence_embedding_trainable(const std::string& text);

    /**
     * Backward pass for sentence embedding
     *
     * @param grad_output Gradient w.r.t. sentence embedding
     */
    void backward_sentence_embedding(const std::vector<float>& grad_output);
};
