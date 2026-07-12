#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include "Matrix.hpp"
#include "Optimizer.hpp"
#ifdef ADAI_ENABLE_GPU
#include "gpu/MatrixGPU.hpp"
#endif

/**
 * Layer Normalization for stabilizing neural network training
 *
 * Layer Normalization normalizes inputs across features (within each sample)
 * rather than across the batch dimension. This provides several benefits:
 * - Stabilizes training of deep networks
 * - Reduces internal covariate shift
 * - Works well with variable batch sizes and RNNs
 * - Enables faster convergence
 *
 * Mathematical Formula:
 * For each sample (row) in the input:
 *   mean = (1/d) * Σ x_i
 *   variance = (1/d) * Σ (x_i - mean)²
 *   normalized = (x - mean) / sqrt(variance + epsilon)
 *   output = gamma * normalized + beta
 *
 * where:
 *   - gamma and beta are learned affine parameters
 *   - epsilon prevents division by zero
 *   - d is the feature dimension
 */
class LayerNorm {
   private:
    Matrix gamma;       // Scale parameter [1, dim]
    Matrix beta;        // Shift parameter [1, dim]
    Matrix gamma_grad;  // Gradient for gamma
    Matrix beta_grad;   // Gradient for beta
    float eps;          // Small constant for numerical stability

    // Optimizer for weight updates
    Optimizer* optimizer{
        nullptr};  // Pointer to optimizer (nullptr means use simple gradient descent)

    // Cached values for backward pass
    Matrix cached_input;             // Original input
    Matrix cached_normalized;        // Normalized values (before affine transform)
    std::vector<float> cached_mean;  // Mean for each sample
    std::vector<float> cached_var;   // Variance for each sample

   public:
    float learning_rate;  // Learning rate for parameter updates

    /**
     * Constructor for LayerNorm
     *
     * @param dim Feature dimension (number of features to normalize)
     * @param epsilon Small constant for numerical stability (default: 1e-5)
     */
    LayerNorm(int dim, float epsilon = 1e-5f);

    /**
     * Forward pass: normalize input and apply affine transformation
     *
     * @param input Input matrix [batch_size, dim]
     * @return Normalized output [batch_size, dim]
     *
     * For each row (sample):
     *   1. Compute mean and variance across features
     *   2. Normalize: (x - mean) / sqrt(var + eps)
     *   3. Apply affine: gamma * normalized + beta
     */
    Matrix forward(const Matrix& input);

    /**
     * Backward pass: compute gradients w.r.t. input and parameters
     *
     * @param grad_output Gradient from upstream layer [batch_size, dim]
     * @return Gradient w.r.t. input [batch_size, dim]
     *
     * Computes:
     *   - Gradients for gamma and beta (accumulated)
     *   - Gradient w.r.t. input using chain rule
     *
     * Note: This method computes gradients but does NOT update parameters.
     * Call update_weights() separately after backward().
     */
    Matrix backward(const Matrix& grad_output);

    /**
     * Set optimizer for weight updates
     *
     * Registers gamma and beta parameters with the optimizer.
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
     * Zero out accumulated gradients
     *
     * Should be called before each new backward pass to prevent
     * gradient accumulation across batches.
     */
    void zero_grad();

    /**
     * Get gamma (scale) parameter
     *
     * @return Reference to gamma matrix [1, dim]
     */
    const Matrix& get_gamma() const {
        return gamma;
    }

    /**
     * Get beta (shift) parameter
     *
     * @return Reference to beta matrix [1, dim]
     */
    const Matrix& get_beta() const {
        return beta;
    }

    /**
     * Get epsilon value
     *
     * @return Epsilon for numerical stability
     */
    float get_epsilon() const {
        return eps;
    }

    /**
     * Set gamma parameter (for loading pretrained weights)
     *
     * @param new_gamma Matrix of gamma values [1, dim]
     */
    void set_gamma(const Matrix& new_gamma);

    /**
     * Set beta parameter (for loading pretrained weights)
     *
     * @param new_beta Matrix of beta values [1, dim]
     */
    void set_beta(const Matrix& new_beta);

    int get_dim() const {
        return gamma.cols;
    }

#ifdef ADAI_ENABLE_GPU
    // --- TD-003 GPU interface ---

    struct GPUState {
        adai::gpu::GPUMatrix gamma_g;  // GPU mirror of gamma [1, dim]
        adai::gpu::GPUMatrix beta_g;   // GPU mirror of beta  [1, dim]
        adai::gpu::GPUMatrix dgamma;   // GPU gradient accumulator [1, dim]
        adai::gpu::GPUMatrix dbeta;    // GPU gradient accumulator [1, dim]
        // Cached for backward (resized per forward call)
        adai::gpu::GPUMatrix normed;  // (x-mean)*rstd before affine [seq, dim]
        adai::gpu::GPUMatrix mean;    // per-row mean [seq, 1]
        adai::gpu::GPUMatrix rstd;    // per-row 1/std [seq, 1]

        explicit GPUState(int dim)
            : gamma_g(1, dim), beta_g(1, dim),
              dgamma(1, dim), dbeta(1, dim),
              normed(1, 1), mean(1, 1), rstd(1, 1) {}
    };
    std::unique_ptr<GPUState> gpu_;

    /** Upload CPU weights to GPU mirrors. */
    void gpu_upload_weights();

    /** Download GPU grad accumulators into CPU grad members. */
    void gpu_download_grads();

    /** Zero GPU gradient accumulators. */
    void gpu_zero_grads();

    /** GPU forward pass. Caches input_norm, mean, rstd for backward. */
    adai::gpu::GPUMatrix gpu_forward(const adai::gpu::GPUMatrix& input);

    /** GPU backward pass. Accumulates dgamma/dbeta. Returns d_input. */
    adai::gpu::GPUMatrix gpu_backward(const adai::gpu::GPUMatrix& dout);
#endif

    /**
     * Print layer configuration
     *
     * @param name Optional name for the layer
     */
    void print_config(const std::string& name = "LayerNorm") const;

    /**
     * Save weights (gamma and beta) to file
     *
     * @param filename Path to save weights
     */
    void save_weights(const std::string& filename) const;

    /**
     * Load weights (gamma and beta) from file
     *
     * @param filename Path to load weights from
     * @throws std::runtime_error if file cannot be opened or dimensions mismatch
     */
    void load_weights(const std::string& filename);
};
