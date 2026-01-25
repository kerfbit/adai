#pragma once

#include <string>
#include <vector>
#include "Matrix.hpp"
#include "Optimizer.hpp"

/**
 * Token Embedding Layer
 *
 * Converts discrete token IDs to continuous dense vector representations.
 * This is the first layer in most neural language models, mapping vocabulary
 * indices to learned embeddings in a high-dimensional space.
 *
 * Key Features:
 * - Learnable embedding matrix [vocab_size, d_model]
 * - Efficient lookup for batch of token IDs
 * - Gradient computation for backpropagation
 * - Support for pre-trained embeddings
 * - Xavier initialization for stable training
 *
 * Mathematical Operation:
 * For token ID t, the embedding is simply a row lookup:
 *   embedding(t) = embedding_matrix[t, :]
 *
 * For a sequence of tokens [t1, t2, ..., tn]:
 *   output = [embedding_matrix[t1, :],
 *             embedding_matrix[t2, :],
 *             ...,
 *             embedding_matrix[tn, :]]
 *
 * Gradient Update:
 * During backpropagation, gradients accumulate only for tokens present
 * in the input sequence:
 *   embedding_grad[t, :] += grad_output[i, :]  for token t at position i
 */
class TokenEmbedding {
   private:
    Matrix embedding_matrix;  // [vocab_size, d_model] - learnable embeddings
    Matrix embedding_grad;    // [vocab_size, d_model] - accumulated gradients
    int vocab_size;           // Size of vocabulary
    int d_model;              // Embedding dimension

    // Cache for backward pass
    std::vector<int> cached_token_ids;  // Stores input token IDs for gradient computation

    // Optimizer support
    Optimizer* optimizer;  // Optional optimizer (nullptr = simple gradient descent)

   public:
    float learning_rate;  // Learning rate for gradient updates (used when optimizer is nullptr)

    /**
     * Constructor
     *
     * Initializes embedding matrix with Xavier/Glorot initialization:
     *   scale = sqrt(1 / d_model)
     * This ensures gradients have appropriate magnitude during early training.
     *
     * @param vocab_size Size of the vocabulary
     * @param d_model Dimension of embedding vectors
     */
    TokenEmbedding(int vocab_size, int d_model);

    /**
     * Forward pass - convert token IDs to embeddings
     *
     * Performs lookup of embeddings for each token in the input sequence.
     * Caches token IDs for use in backward pass.
     *
     * @param token_ids Vector of token IDs [sequence_length]
     * @return Matrix of embeddings [sequence_length, d_model]
     *
     * Example:
     *   token_ids = [5, 12, 3]
     *   output = [embedding_matrix[5, :],
     *             embedding_matrix[12, :],
     *             embedding_matrix[3, :]]
     */
    Matrix forward(const std::vector<int>& token_ids);

    /**
     * Backward pass - accumulate gradients
     *
     * Computes gradients with respect to embedding matrix.
     * For each token in the sequence, adds gradient to corresponding row.
     *
     * @param token_ids Vector of token IDs (must match forward pass)
     * @param grad_output Gradient from upstream [sequence_length, d_model]
     *
     * Updates:
     *   For each position i with token t:
     *     embedding_grad[t, :] += grad_output[i, :]
     *
     * Note: Multiple occurrences of same token accumulate gradients
     */
    void backward(const std::vector<int>& token_ids, const Matrix& grad_output);

    /**
     * Set optimizer for advanced optimization algorithms
     *
     * Enables use of Adam, AdamW, or other optimizers instead of
     * simple gradient descent. When set, automatically registers
     * embedding parameters with the optimizer.
     *
     * @param opt Pointer to optimizer (nullptr to use simple gradient descent)
     */
    void set_optimizer(Optimizer* opt);

    /**
     * Register embedding parameters with optimizer
     *
     * Called automatically by set_optimizer().
     * Registers embedding_matrix and embedding_grad with the optimizer.
     */
    void register_parameters();

    /**
     * Apply gradients and update embeddings
     *
     * If optimizer is set, uses optimizer->step() for advanced optimization.
     * Otherwise, performs simple gradient descent:
     *   embedding_matrix -= learning_rate * embedding_grad
     *
     * Then zeros out gradients for next iteration.
     */
    void update_weights();

    /**
     * Zero out all gradients
     *
     * Resets gradient accumulator to zero.
     * Should be called before each forward/backward pass in training.
     */
    void zero_grad();

    /**
     * Get embedding for a single token
     *
     * Useful for inspection or when only one token embedding is needed.
     *
     * @param token_id Token ID to get embedding for
     * @return Vector of size d_model
     * @throws std::out_of_range if token_id >= vocab_size
     */
    std::vector<float> get_token_embedding(int token_id) const;

    /**
     * Get reference to embedding matrix
     *
     * Provides read-only access to the entire embedding matrix.
     * Useful for visualization or analysis.
     *
     * @return Const reference to embedding matrix [vocab_size, d_model]
     */
    const Matrix& get_embeddings() const;

    /**
     * Get vocabulary size
     *
     * @return Number of tokens in vocabulary
     */
    int get_vocab_size() const;

    /**
     * Get embedding dimension
     *
     * @return Dimension of embedding vectors (d_model)
     */
    int get_embedding_dim() const;

    /**
     * Load pre-trained embeddings from file
     *
     * File format (binary):
     *   - vocab_size (int)
     *   - d_model (int)
     *   - embedding_matrix data (vocab_size * d_model floats)
     *
     * @param filename Path to pre-trained embeddings file
     * @throws std::runtime_error if file cannot be opened or dimensions mismatch
     */
    void load_pretrained(const std::string& filename);

    /**
     * Save embeddings to file
     *
     * Saves current embedding matrix in binary format for later reuse.
     *
     * @param filename Path to save embeddings
     * @throws std::runtime_error if file cannot be opened
     */
    void save_embeddings(const std::string& filename) const;

    /**
     * Initialize embeddings with specific values
     *
     * Sets all embeddings to a constant value.
     * Useful for testing or specific initialization strategies.
     *
     * @param value Value to set all embeddings to
     */
    void initialize_constant(float value);

    /**
     * Reinitialize embeddings with Xavier initialization
     *
     * Resets embeddings to random values with Xavier scaling.
     * Useful for restarting training or testing initialization impact.
     */
    void reinitialize();

    /**
     * Print configuration summary
     *
     * Displays embedding layer configuration for debugging.
     *
     * @param name Optional name for the layer
     */
    void print_config(const std::string& name = "TokenEmbedding") const;

    /**
     * Get gradient norm (for monitoring training)
     *
     * Computes L2 norm of gradient matrix.
     * Useful for detecting vanishing/exploding gradients.
     *
     * @return L2 norm of embedding_grad
     */
    float get_gradient_norm() const;

    /**
     * Clip gradients by norm
     *
     * Prevents exploding gradients by scaling down if norm exceeds threshold.
     *
     * @param max_norm Maximum allowed gradient norm
     */
    void clip_gradients(float max_norm);
};
