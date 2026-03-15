#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include "Matrix.hpp"

/**
 * @brief Optimization algorithm types
 */
enum class OptimizerType : std::uint8_t {
    SGD,           // Stochastic Gradient Descent
    SGD_MOMENTUM,  // SGD with momentum
    ADAM,          // Adaptive Moment Estimation
    ADAMW          // Adam with decoupled weight decay
};

/**
 * @brief Parameter group for optimization
 *
 * Holds a reference to a weight matrix and its gradient,
 * along with optimizer state (momentum, adaptive learning rates, etc.)
 */
struct ParameterGroup {
    Matrix* weights;    // Pointer to weight matrix
    Matrix* gradients;  // Pointer to gradient matrix

    // Optimizer state
    Matrix momentum;  // First moment (momentum/mean of gradients)
    Matrix velocity;  // Second moment (variance of gradients)
    int step{0};      // Number of updates performed

    ParameterGroup(Matrix* w, Matrix* g) : weights(w), gradients(g) {}
};

/**
 * @brief Centralized gradient optimizer
 *
 * Provides advanced optimization algorithms and gradient processing:
 * - Multiple optimization strategies (SGD, Adam, AdamW)
 * - Gradient clipping for training stability
 * - Weight decay / L2 regularization
 * - Momentum and adaptive learning rates
 * - Centralized gradient statistics
 *
 * Usage:
 *   Optimizer opt(OptimizerType::ADAM, learning_rate);
 *   opt.add_parameter_group(weights, gradients);
 *   // ... training loop ...
 *   opt.zero_grad();
 *   // ... forward/backward ...
 *   opt.clip_gradients(1.0f);  // Optional
 *   opt.step();
 */
class Optimizer {
   private:
    OptimizerType type;
    float learning_rate;

    // Hyperparameters
    float momentum_beta;  // Momentum coefficient (default: 0.9)
    float beta1;          // Adam first moment decay (default: 0.9)
    float beta2;          // Adam second moment decay (default: 0.999)
    float epsilon;        // Small constant for numerical stability (default: 1e-8)
    float weight_decay;   // L2 regularization / weight decay (default: 0.0)
    bool amsgrad;         // Use AMSGrad variant of Adam (default: false)

    // Gradient clipping
    float max_grad_norm;  // Maximum gradient norm (0 = no clipping)

    // Parameter groups
    std::vector<ParameterGroup> parameter_groups;

    // Global step counter
    int global_step;

    /**
     * @brief Update parameters using SGD
     */
    void step_sgd(ParameterGroup& param);

    /**
     * @brief Update parameters using SGD with momentum
     */
    void step_sgd_momentum(ParameterGroup& param);

    /**
     * @brief Update parameters using Adam
     */
    void step_adam(ParameterGroup& param);

    /**
     * @brief Update parameters using AdamW
     */
    void step_adamw(ParameterGroup& param);

   public:
    /**
     * @brief Constructor
     *
     * @param opt_type Optimization algorithm
     * @param lr Learning rate
     */
    Optimizer(OptimizerType opt_type = OptimizerType::ADAM, float lr = 0.001f);

    /**
     * @brief Add a parameter group to optimize
     *
     * @param weights Pointer to weight matrix
     * @param gradients Pointer to gradient matrix (same shape as weights)
     */
    void add_parameter_group(Matrix* weights, Matrix* gradients);

    /**
     * @brief Set learning rate
     *
     * @param lr New learning rate
     */
    void set_learning_rate(float lr);

    /**
     * @brief Get current learning rate
     */
    float get_learning_rate() const {
        return learning_rate;
    }

    /**
     * @brief Set momentum coefficient (for SGD with momentum)
     *
     * @param beta Momentum coefficient (typically 0.9)
     */
    void set_momentum(float beta);

    /**
     * @brief Set Adam beta parameters
     *
     * @param b1 First moment decay rate (default: 0.9)
     * @param b2 Second moment decay rate (default: 0.999)
     */
    void set_betas(float b1, float b2);

    /**
     * @brief Set weight decay coefficient
     *
     * @param wd Weight decay (L2 regularization)
     */
    void set_weight_decay(float wd);

    /**
     * @brief Set maximum gradient norm for clipping
     *
     * @param max_norm Maximum L2 norm (0 = no clipping)
     */
    void set_max_grad_norm(float max_norm);

    /**
     * @brief Zero all gradients
     *
     * Should be called at the start of each training iteration.
     */
    void zero_grad();

    /**
     * @brief Clip gradients by global norm
     *
     * Scales gradients if their global norm exceeds max_grad_norm.
     * Essential for training stability in transformers.
     *
     * @return Global gradient norm before clipping
     */
    float clip_gradients();

    /**
     * @brief Clip gradients by global norm with custom threshold
     *
     * @param max_norm Maximum gradient norm
     * @return Global gradient norm before clipping
     */
    float clip_gradients(float max_norm);

    /**
     * @brief Perform optimization step
     *
     * Updates all parameters based on their gradients using
     * the configured optimization algorithm.
     */
    void step();

    /**
     * @brief Get global gradient norm
     *
     * Useful for monitoring training stability.
     *
     * @return L2 norm of all gradients
     */
    float get_gradient_norm() const;

    /**
     * @brief Get global weight (parameter) L2 norm
     *
     * Used by TD-013 advanced metrics to compute the weight-update ratio.
     *
     * @return L2 norm of all weight parameters
     */
    float get_weight_norm() const;

    /**
     * @brief Get number of parameter groups
     */
    size_t num_parameters() const {
        return parameter_groups.size();
    }

    /**
     * @brief Get total number of trainable parameters
     */
    size_t total_parameters() const;

    /**
     * @brief Get optimizer type name
     */
    const char* get_optimizer_name() const;

    /**
     * @brief Reset optimizer state
     *
     * Clears momentum and velocity terms.
     * Useful when loading a model or changing optimization strategy.
     */
    void reset_state();
};
