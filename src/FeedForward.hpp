#pragma once

#include <functional>
#include <string>
#include "Matrix.hpp"
#include "Optimizer.hpp"

/**
 * Position-wise Feed-Forward Network
 *
 * Implements a two-layer feed-forward neural network with GELU activation,
 * used in transformer architectures between attention layers.
 *
 * Architecture:
 *   input (d_model) -> Linear -> GELU -> Linear -> output (d_model)
 *
 * The intermediate dimension d_ff is typically 4x the model dimension.
 *
 * Mathematical Operations:
 *   hidden = GELU(input * W1 + b1)
 *   output = hidden * W2 + b2
 *
 * Where:
 *   - W1: [d_model x d_ff] weight matrix
 *   - b1: [d_ff] bias vector
 *   - W2: [d_ff x d_model] weight matrix
 *   - b2: [d_model] bias vector
 *   - GELU: Gaussian Error Linear Unit activation
 *
 * Features:
 *   - Xavier/He initialization for weights
 *   - Zero initialization for biases
 *   - Caching for efficient backpropagation
 *   - Gradient clipping support
 *   - Weight persistence (save/load)
 */
/**
 * Callback invoked immediately after the GELU activation in every forward pass.
 * Receives the post-GELU hidden matrix [seq_len × d_ff] so callers can
 * compute per-layer activation statistics (e.g. saturation ratio).
 */
using ActivationHookFn = std::function<void(const Matrix&)>;

class FeedForward {
   private:
    // Model dimensions
    int d_model;  // Input/output dimension
    int d_ff;     // Hidden layer dimension

    // Weights and biases
    Matrix W1;  // First layer weights [d_model x d_ff]
    Matrix W2;  // Second layer weights [d_ff x d_model]
    Matrix b1;  // First layer bias [1 x d_ff]
    Matrix b2;  // Second layer bias [1 x d_model]

    // Gradients
    Matrix W1_grad;
    Matrix W2_grad;
    Matrix b1_grad;
    Matrix b2_grad;

    // Cached values for backward pass
    Matrix cached_input;             // Input to forward pass
    Matrix cached_hidden;            // Hidden layer before activation
    Matrix cached_hidden_activated;  // Hidden layer after GELU

    // Optimizer support
    Optimizer* optimizer;  // Optional optimizer (nullptr = simple gradient descent)

    // Activation hook (optional, fired after GELU in forward())
    ActivationHookFn activation_hook_;

   public:
    float learning_rate;  // Learning rate for weight updates (used when optimizer is nullptr)

    /**
     * Constructor
     *
     * @param d_model Model dimension (input/output size)
     * @param d_ff Feed-forward dimension (hidden layer size)
     */
    FeedForward(int d_model, int d_ff);

    /**
     * Forward pass through the feed-forward network
     *
     * Applies two linear transformations with GELU activation:
     *   1. Linear: input -> hidden (d_model -> d_ff)
     *   2. GELU activation
     *   3. Linear: hidden -> output (d_ff -> d_model)
     *
     * @param input Input matrix [seq_len x d_model]
     * @return Output matrix [seq_len x d_model]
     */
    Matrix forward(const Matrix& input);

    /**
     * Backward pass through the feed-forward network
     *
     * Computes gradients for all weights and biases, and returns
     * the gradient w.r.t. the input.
     *
     * @param grad_output Gradient from next layer [seq_len x d_model]
     * @return Gradient w.r.t. input [seq_len x d_model]
     */
    Matrix backward(const Matrix& grad_output);

    /**
     * Set optimizer for advanced optimization algorithms
     *
     * Enables use of Adam, AdamW, or other optimizers instead of
     * simple gradient descent. When set, automatically registers
     * weight matrices and biases with the optimizer.
     *
     * @param opt Pointer to optimizer (nullptr to use simple gradient descent)
     */
    void set_optimizer(Optimizer* opt);

    /**
     * Register weight matrices and biases with optimizer
     *
     * Called automatically by set_optimizer().
     * Registers W1, W2, b1, b2 and their gradients with the optimizer.
     */
    void register_parameters();

    /**
     * Update weights using accumulated gradients
     *
     * If optimizer is set, uses optimizer->step() for advanced optimization.
     * Otherwise, applies simple gradient descent: W = W - lr * grad_W
     * Automatically zeros gradients after update.
     */
    void update_weights();

    /**
     * Zero all accumulated gradients
     */
    void zero_grad();

    /**
     * Get the L2 norm of all gradients
     *
     * Useful for monitoring gradient flow and applying gradient clipping.
     *
     * @return L2 norm of concatenated gradients
     */
    float get_gradient_norm() const;

    /**
     * Clip gradients to prevent exploding gradients
     *
     * If gradient norm exceeds max_norm, scales all gradients
     * to have norm equal to max_norm.
     *
     * @param max_norm Maximum allowed gradient norm
     */
    void clip_gradients(float max_norm);

    // ── SafeTensors accessor API ─────────────────────────────────────────────
    const Matrix& get_W1() const {
        return W1;
    }
    const Matrix& get_W2() const {
        return W2;
    }
    const Matrix& get_b1() const {
        return b1;
    }
    const Matrix& get_b2() const {
        return b2;
    }
    void set_W1(const Matrix& m) {
        W1 = m;
    }
    void set_W2(const Matrix& m) {
        W2 = m;
    }
    void set_b1(const Matrix& m) {
        b1 = m;
    }
    void set_b2(const Matrix& m) {
        b2 = m;
    }

    /**
     * Save weights and biases to binary file
     *
     * @param filename Path to output file
     */
    void save_weights(const std::string& filename) const;

    /**
     * Load weights and biases from binary file
     *
     * @param filename Path to input file
     * @throws std::runtime_error if file doesn't exist or dimensions mismatch
     */
    void load_weights(const std::string& filename);

    /**
     * Print network configuration
     *
     * @param name Optional name for the layer
     */
    void print_config(const std::string& name = "FeedForward") const;

    /**
     * Get model dimension
     *
     * @return d_model
     */
    int get_d_model() const {
        return d_model;
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
     * Register a callback invoked after GELU in every forward pass.
     * Pass nullptr / call clear_activation_hook() to disable.
     *
     * @param fn Callable receiving the post-GELU hidden matrix [seq_len × d_ff]
     */
    void set_activation_hook(ActivationHookFn fn) {
        activation_hook_ = std::move(fn);
    }

    /**
     * Remove any previously registered activation hook.
     */
    void clear_activation_hook() {
        activation_hook_ = nullptr;
    }
};
