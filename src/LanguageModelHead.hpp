#pragma once

#include <string>
#include <vector>
#include "Matrix.hpp"
#include "Optimizer.hpp"

/**
 * Language Model Head for Next-Token Prediction
 *
 * Projects decoder output to vocabulary logits for next-token prediction.
 * This is the final layer in a transformer decoder that maps from the
 * model dimension to vocabulary size.
 *
 * Mathematical Operation:
 *   logits = input * W_output + bias
 *
 * Where:
 *   - input: [seq_len, d_model] decoder output
 *   - W_output: [d_model, vocab_size] weight matrix
 *   - bias: [vocab_size] bias vector
 *   - logits: [seq_len, vocab_size] output scores
 *
 * Features:
 *   - Xavier initialization for weights
 *   - Zero initialization for biases
 *   - Gradient computation for backpropagation
 *   - Model persistence (save/load)
 */
class LanguageModelHead {
   private:
    int d_model;     // Input dimension
    int vocab_size;  // Output vocabulary size

    // Learnable parameters
    Matrix W_output;  // [d_model, vocab_size]
    Matrix bias;      // [1, vocab_size]

    // Gradients
    Matrix W_output_grad;
    Matrix bias_grad;

    // Cached for backward pass
    Matrix cached_input;

    // Optimizer integration (optional)
    Optimizer* optimizer;  // Pointer to optimizer (nullptr = simple gradient descent)

   public:
    float learning_rate;  // Used when optimizer is nullptr (backward compatibility)

    /**
     * Constructor
     *
     * Initializes weight matrix with Xavier initialization and
     * bias vector with zeros.
     *
     * @param d_model Model dimension (input size)
     * @param vocab_size Vocabulary size (output size)
     */
    LanguageModelHead(int d_model, int vocab_size);

    /**
     * Forward pass: Project to vocabulary logits
     *
     * Applies linear transformation: logits = input * W + b
     * Caches input for backward pass.
     *
     * @param input Decoder output [seq_len, d_model]
     * @return Logits [seq_len, vocab_size]
     */
    Matrix forward(const Matrix& input);

    /**
     * Get probability distribution for next token prediction
     *
     * Applies softmax to convert logits to probabilities.
     *
     * @param logits Output logits [vocab_size] (single position)
     * @return Probability distribution [vocab_size]
     */
    std::vector<float> get_probabilities(const std::vector<float>& logits);

    /**
     * Backward pass: Compute gradients
     *
     * Computes gradients for W_output and bias, and returns
     * gradient w.r.t. input.
     *
     * @param grad_output Gradient from loss [seq_len, vocab_size]
     * @return Gradient w.r.t. input [seq_len, d_model]
     */
    Matrix backward(const Matrix& grad_output);

    /**
     * Update weights using accumulated gradients
     *
     * If optimizer is set, uses optimizer->step().
     * Otherwise, applies simple gradient descent: W = W - lr * grad_W
     * Automatically zeros gradients after update.
     */
    void update_weights();

    /**
     * Zero accumulated gradients
     *
     * Resets gradient accumulators to zero.
     */
    void zero_grad();

    /**
     * Save parameters to file
     *
     * @param filepath Path to save file
     */
    void save(const std::string& filepath) const;

    /**
     * Load parameters from file
     *
     * @param filepath Path to load file
     */
    void load(const std::string& filepath);

    /**
     * Set optimizer for advanced optimization algorithms
     *
     * Automatically registers parameters when optimizer is set.
     * Pass nullptr to use simple gradient descent.
     *
     * @param opt Pointer to optimizer instance
     */
    void set_optimizer(Optimizer* opt);

    /**
     * Register parameters with optimizer
     *
     * Called automatically by set_optimizer().
     * Can be called manually if needed.
     */
    void register_parameters();

    // ── SafeTensors accessor API ─────────────────────────────────────────────
    const Matrix& get_W_output() const {
        return W_output;
    }
    const Matrix& get_bias() const {
        return bias;
    }
    void set_W_output(const Matrix& m) {
        W_output = m;
    }
    void set_bias(const Matrix& m) {
        bias = m;
    }

    /**
     * Save weights to file (consistent with other components)
     *
     * Wrapper around save() for API consistency.
     *
     * @param filename Path to save weights
     */
    void save_weights(const std::string& filename) const;

    /**
     * Load weights from file (consistent with other components)
     *
     * Wrapper around load() for API consistency.
     *
     * @param filename Path to load weights from
     */
    void load_weights(const std::string& filename);
};
