#ifndef LORA_HPP
#define LORA_HPP

// @adai-status: beta        (capped by TD-038 — tested but not wired into any shipped binary)
// @adai-version: 0.7.0
// @adai-reviewed: 2026-09-07


#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "Matrix.hpp"

/**
 * @file LoRA.hpp
 * @brief Low-Rank Adaptation (LoRA) for Parameter-Efficient Fine-Tuning
 *
 * LoRA is a technique that freezes pretrained model weights and injects trainable
 * low-rank decomposition matrices into each layer. This dramatically reduces the
 * number of trainable parameters for fine-tuning.
 *
 * Key Concept:
 * Instead of updating W directly: W' = W + ΔW
 * LoRA decomposes ΔW as: ΔW = BA, where B is (d × r) and A is (r × k)
 *
 * Benefits:
 * - Reduces trainable parameters by orders of magnitude (often 10,000x)
 * - No additional inference latency (can merge adapters)
 * - Multiple task adapters can be swapped efficiently
 * - Maintains base model quality
 *
 * Typical ranks: r = 4, 8, 16, 32 (much smaller than d or k)
 *
 * @version 1.0
 * @date January 2026
 */

/**
 * @class LoRAAdapter
 * @brief Low-rank adapter for a single weight matrix
 *
 * Represents the decomposition ΔW = B * A where:
 * - A: (rank × input_dim) - Initialized with Gaussian noise
 * - B: (output_dim × rank) - Initialized to zeros
 *
 * This ensures ΔW = 0 at initialization, so training starts from pretrained weights.
 */
class LoRAAdapter {
   private:
    int input_dim_;   // Input dimension (k)
    int output_dim_;  // Output dimension (d)
    int rank_;        // LoRA rank (r)
    float alpha_;     // LoRA scaling factor

    Matrix A_;  // Low-rank matrix A: (rank × input_dim)
    Matrix B_;  // Low-rank matrix B: (output_dim × rank)

    Matrix grad_A_;  // Gradients for A
    Matrix grad_B_;  // Gradients for B

    /**
     * @brief Initialize LoRA matrices
     *
     * A ~ N(0, sigma^2) - Random initialization
     * B = 0             - Zero initialization (ensures ΔW = 0 initially)
     */
    void initialize() {
        // Initialize A with Gaussian noise
        float std_dev = 1.0f / std::sqrt(rank_);
        A_ = Matrix(rank_, input_dim_);
        for (int r = 0; r < rank_; r++) {
            for (int c = 0; c < input_dim_; c++) {
                A_(r, c) = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_dev;
            }
        }

        // Initialize B to zeros
        B_ = Matrix(output_dim_, rank_);
        for (int r = 0; r < output_dim_; r++) {
            for (int c = 0; c < rank_; c++) {
                B_(r, c) = 0.0f;
            }
        }

        // Initialize gradients
        grad_A_ = Matrix(rank_, input_dim_);
        grad_B_ = Matrix(output_dim_, rank_);
    }

   public:
    /**
     * @brief Construct LoRA adapter
     *
     * @param input_dim Input dimension of the weight matrix
     * @param output_dim Output dimension of the weight matrix
     * @param rank LoRA rank (r << min(input_dim, output_dim))
     * @param alpha Scaling factor (typically rank or 2*rank)
     */
    LoRAAdapter(int input_dim, int output_dim, int rank, float alpha = -1.0f)
        : input_dim_(input_dim),
          output_dim_(output_dim),
          rank_(rank),
          alpha_(alpha < 0 ? static_cast<float>(rank) : alpha) {
        if (rank <= 0 || rank > std::min(input_dim, output_dim)) {
            throw std::invalid_argument("Invalid LoRA rank");
        }

        initialize();
    }

    /**
     * @brief Apply LoRA adapter to input
     *
     * Computes: y = x * (W + (alpha/r) * B * A)
     *         = x * W + (alpha/r) * x * A^T * B^T
     *
     * @param x Input matrix (batch_size × input_dim)
     * @param W_output Output from frozen weight W (already computed)
     * @return Final output with LoRA adaptation
     */
    Matrix forward(const Matrix& x, const Matrix& W_output) {
        // Compute LoRA contribution: x * A^T
        Matrix A_T = A_.transpose();
        Matrix xA = x * A_T;

        // Multiply by B^T: (x * A^T) * B^T
        Matrix B_T = B_.transpose();
        Matrix xAB = xA * B_T;

        // Scale by alpha/rank
        float scale = alpha_ / rank_;
        Matrix delta_W_output(xAB.rows, xAB.cols);
        for (int r = 0; r < xAB.rows; r++) {
            for (int c = 0; c < xAB.cols; c++) {
                delta_W_output(r, c) = scale * xAB(r, c);
            }
        }

        // Add to base output: W_output + ΔW_output
        Matrix output(W_output.rows, W_output.cols);
        for (int r = 0; r < output.rows; r++) {
            for (int c = 0; c < output.cols; c++) {
                output(r, c) = W_output(r, c) + delta_W_output(r, c);
            }
        }

        return output;
    }

    /**
     * @brief Backward pass for LoRA adapter
     *
     * Computes gradients for A and B given output gradient.
     *
     * @param x Input to forward pass (cached)
     * @param grad_output Gradient flowing back from loss
     */
    void backward(const Matrix& x, const Matrix& grad_output) {
        float scale = alpha_ / rank_;

        // For ΔW = B * A where B is (output_dim, rank), A is (rank, input_dim)
        // Forward: y = x * W + scale * (x * A^T * B^T)
        // grad_output is dy/dloss

        // grad_B = scale * grad_output^T * (x * A^T)
        Matrix A_T = A_.transpose();
        Matrix xA_T = x * A_T;                    // (batch, rank)
        Matrix grad_T = grad_output.transpose();  // (output_dim, batch)
        grad_B_ = grad_T * xA_T;                  // (output_dim, rank)

        // Scale grad_B
        for (int r = 0; r < grad_B_.rows; r++) {
            for (int c = 0; c < grad_B_.cols; c++) {
                grad_B_(r, c) *= scale;
            }
        }

        // grad_A = scale * B^T * grad_output^T * x
        Matrix B_T = B_.transpose();     // (rank, output_dim)
        Matrix B_T_grad = B_T * grad_T;  // (rank, batch)
        grad_A_ = B_T_grad * x;          // (rank, input_dim)

        // Scale grad_A
        for (int r = 0; r < grad_A_.rows; r++) {
            for (int c = 0; c < grad_A_.cols; c++) {
                grad_A_(r, c) *= scale;
            }
        }
    }

    /**
     * @brief Update LoRA parameters with gradients
     *
     * @param learning_rate Learning rate
     */
    void update(float learning_rate) {
        // Update A
        for (int r = 0; r < rank_; r++) {
            for (int c = 0; c < input_dim_; c++) {
                A_(r, c) -= learning_rate * grad_A_(r, c);
            }
        }

        // Update B
        for (int r = 0; r < output_dim_; r++) {
            for (int c = 0; c < rank_; c++) {
                B_(r, c) -= learning_rate * grad_B_(r, c);
            }
        }
    }

    /**
     * @brief Zero gradients
     */
    void zero_grad() {
        for (int r = 0; r < grad_A_.rows; r++) {
            for (int c = 0; c < grad_A_.cols; c++) {
                grad_A_(r, c) = 0.0f;
            }
        }
        for (int r = 0; r < grad_B_.rows; r++) {
            for (int c = 0; c < grad_B_.cols; c++) {
                grad_B_(r, c) = 0.0f;
            }
        }
    }

    /**
     * @brief Merge LoRA adapter into base weight matrix
     *
     * Computes: W' = W + (alpha/r) * B * A
     *
     * This allows removing LoRA overhead during inference.
     *
     * @param W Base weight matrix
     * @return Merged weight matrix
     */
    Matrix merge_with_base(const Matrix& W) {
        if (W.rows != output_dim_ || W.cols != input_dim_) {
            throw std::invalid_argument("Weight matrix dimension mismatch");
        }

        // Compute ΔW = B * A
        Matrix delta_W = B_ * A_;

        // Scale by alpha/r
        float scale = alpha_ / rank_;
        for (int r = 0; r < delta_W.rows; r++) {
            for (int c = 0; c < delta_W.cols; c++) {
                delta_W(r, c) *= scale;
            }
        }

        // Add to base weights
        Matrix merged(output_dim_, input_dim_);
        for (int r = 0; r < output_dim_; r++) {
            for (int c = 0; c < input_dim_; c++) {
                merged(r, c) = W(r, c) + delta_W(r, c);
            }
        }

        return merged;
    }

    /**
     * @brief Get number of trainable parameters
     */
    int num_parameters() const {
        return rank_ * (input_dim_ + output_dim_);
    }

    /**
     * @brief Save LoRA adapter to file
     */
    void save(const std::string& filepath) const {
        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + filepath);
        }

        // Write metadata
        file.write(reinterpret_cast<const char*>(&input_dim_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&output_dim_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&rank_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&alpha_), sizeof(float));

        // Write A matrix
        for (int r = 0; r < rank_; r++) {
            for (int c = 0; c < input_dim_; c++) {
                float val = A_(r, c);
                file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }

        // Write B matrix
        for (int r = 0; r < output_dim_; r++) {
            for (int c = 0; c < rank_; c++) {
                float val = B_(r, c);
                file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
    }

    /**
     * @brief Load LoRA adapter from file
     */
    void load(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + filepath);
        }

        // Read metadata
        file.read(reinterpret_cast<char*>(&input_dim_), sizeof(int));
        file.read(reinterpret_cast<char*>(&output_dim_), sizeof(int));
        file.read(reinterpret_cast<char*>(&rank_), sizeof(int));
        file.read(reinterpret_cast<char*>(&alpha_), sizeof(float));

        // Reinitialize matrices
        A_ = Matrix(rank_, input_dim_);
        B_ = Matrix(output_dim_, rank_);
        grad_A_ = Matrix(rank_, input_dim_);
        grad_B_ = Matrix(output_dim_, rank_);

        // Read A matrix
        for (int r = 0; r < rank_; r++) {
            for (int c = 0; c < input_dim_; c++) {
                float val;
                file.read(reinterpret_cast<char*>(&val), sizeof(float));
                A_(r, c) = val;
            }
        }

        // Read B matrix
        for (int r = 0; r < output_dim_; r++) {
            for (int c = 0; c < rank_; c++) {
                float val;
                file.read(reinterpret_cast<char*>(&val), sizeof(float));
                B_(r, c) = val;
            }
        }
    }

    // Getters
    int get_rank() const {
        return rank_;
    }
    float get_alpha() const {
        return alpha_;
    }
    const Matrix& get_A() const {
        return A_;
    }
    const Matrix& get_B() const {
        return B_;
    }
};

/**
 * @class LoRAConfig
 * @brief Configuration for applying LoRA to a model
 */
struct LoRAConfig {
    int rank = 8;                 // LoRA rank (common: 4, 8, 16)
    float alpha = 16.0f;          // LoRA alpha (often 2*rank)
    float dropout = 0.0f;         // LoRA dropout (0.0 = no dropout)
    bool apply_to_query = true;   // Apply to query projection
    bool apply_to_key = true;     // Apply to key projection
    bool apply_to_value = true;   // Apply to value projection
    bool apply_to_output = true;  // Apply to output projection
    bool apply_to_ffn = false;    // Apply to feedforward layers

    LoRAConfig() = default;

    /**
     * @brief Calculate parameter reduction ratio
     *
     * @param original_params Number of original parameters
     * @param adapter_params Number of LoRA parameters
     * @return Reduction ratio (e.g., 10000 means 10000x fewer params)
     */
    static float reduction_ratio(int original_params, int adapter_params) {
        return static_cast<float>(original_params) / adapter_params;
    }
};

/**
 * @brief Calculate LoRA parameters for attention layer
 *
 * @param d_model Model dimension
 * @param config LoRA configuration
 * @return Number of trainable parameters
 */
inline int calculate_lora_params_attention(int d_model, const LoRAConfig& config) {
    int params_per_projection = config.rank * (d_model + d_model);
    int num_projections = 0;

    if (config.apply_to_query)
        num_projections++;
    if (config.apply_to_key)
        num_projections++;
    if (config.apply_to_value)
        num_projections++;
    if (config.apply_to_output)
        num_projections++;

    return num_projections * params_per_projection;
}

/**
 * @brief Example: Compare LoRA vs full fine-tuning parameters
 *
 * For a 768-dim model with 12 layers:
 * - Full fine-tuning: ~110M parameters
 * - LoRA (r=8): ~0.3M parameters (~370x reduction)
 * - LoRA (r=4): ~0.15M parameters (~730x reduction)
 */
inline void print_lora_statistics(int d_model, int num_layers, const LoRAConfig& config) {
    int original_params_per_layer = 4 * d_model * d_model;  // Q, K, V, O projections
    int original_total = original_params_per_layer * num_layers;

    int lora_params_per_layer = calculate_lora_params_attention(d_model, config);
    int lora_total = lora_params_per_layer * num_layers;

    float reduction = LoRAConfig::reduction_ratio(original_total, lora_total);

    std::cout << "=== LoRA Statistics ===\n";
    std::cout << "Original parameters: " << original_total << "\n";
    std::cout << "LoRA parameters: " << lora_total << "\n";
    std::cout << "Reduction: " << reduction << "x\n";
    std::cout << "Rank: " << config.rank << "\n";
    std::cout << "Alpha: " << config.alpha << "\n";
}

#endif  // LORA_HPP
