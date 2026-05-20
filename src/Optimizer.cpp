#include "Optimizer.hpp"
#include <cmath>
#include <iostream>
#include <limits>

// Constructor
Optimizer::Optimizer(OptimizerType opt_type, float lr) : type(opt_type), learning_rate(lr) {}

// Add parameter group
void Optimizer::add_parameter_group(Matrix* weights, Matrix* gradients) {
    if (!weights || !gradients) {
        throw std::runtime_error("Cannot add null parameter group");
    }

    if (weights->rows != gradients->rows || weights->cols != gradients->cols) {
        throw std::runtime_error("Weights and gradients must have same shape");
    }

    parameter_groups.emplace_back(weights, gradients);

    // Initialize optimizer state matrices
    auto& param = parameter_groups.back();
    if (type == OptimizerType::SGD_MOMENTUM || type == OptimizerType::ADAM ||
        type == OptimizerType::ADAMW) {
        param.momentum = Matrix(weights->rows, weights->cols);
        param.momentum.fill(0.0f);
    }

    if (type == OptimizerType::ADAM || type == OptimizerType::ADAMW) {
        param.velocity = Matrix(weights->rows, weights->cols);
        param.velocity.fill(0.0f);
    }
}

// Set learning rate
void Optimizer::set_learning_rate(float lr) {
    learning_rate = lr;
}

// Set momentum
void Optimizer::set_momentum(float beta) {
    momentum_beta = beta;
}

// Set Adam betas
void Optimizer::set_betas(float b1, float b2) {
    beta1 = b1;
    beta2 = b2;
}

// Set weight decay
void Optimizer::set_weight_decay(float wd) {
    weight_decay = wd;
}

// Set max gradient norm
void Optimizer::set_max_grad_norm(float max_norm) {
    max_grad_norm = max_norm;
}

// Zero all gradients
void Optimizer::zero_grad() {
    for (auto& param : parameter_groups) {
        if (param.gradients) {
            param.gradients->fill(0.0f);
        }
    }
}

// NOLINTBEGIN(readability-convert-member-functions-to-static)
// Get gradient norm
float Optimizer::get_gradient_norm() const {
    float total_norm = 0.0f;

    for (const auto& param : parameter_groups) {
        if (param.gradients) {
            for (int i = 0; i < param.gradients->rows; i++) {
                for (int j = 0; j < param.gradients->cols; j++) {
                    float val = (*param.gradients)(i, j);
                    total_norm += val * val;
                }
            }
        }
    }

    return std::sqrt(total_norm);
}

// Get weight (parameter) L2 norm — used for weight-update ratio (TD-013)
float Optimizer::get_weight_norm() const {
    float total_norm = 0.0f;

    for (const auto& param : parameter_groups) {
        if (param.weights) {
            for (int i = 0; i < param.weights->rows; i++) {
                for (int j = 0; j < param.weights->cols; j++) {
                    float val = (*param.weights)(i, j);
                    total_norm += val * val;
                }
            }
        }
    }

    float result = std::sqrt(total_norm);
    // TD-013 debug: log first call to help diagnose zero weight norm
    static int debug_count = 0;
    if (debug_count < 3) {
        ++debug_count;
        fprintf(stderr, "[TD-013] get_weight_norm: groups=%zu total_sq=%.4f result=%.6f\n",
                parameter_groups.size(), total_norm, result);
    }
    return result;
}

// Clip gradients
float Optimizer::clip_gradients() {
    if (max_grad_norm <= 0.0f) {
        return 0.0f;
    }
    return clip_gradients(max_grad_norm);
}

float Optimizer::clip_gradients(float max_norm) {
    if (max_norm <= 0.0f) {
        return 0.0f;
    }

    // Compute global gradient norm
    float total_norm = get_gradient_norm();

    // Clip if necessary
    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);

        for (auto& param : parameter_groups) {
            if (param.gradients) {
                for (int i = 0; i < param.gradients->rows; i++) {
                    for (int j = 0; j < param.gradients->cols; j++) {
                        (*param.gradients)(i, j) *= clip_coef;
                    }
                }
            }
        }
    }

    return total_norm;
}

float Optimizer::clip_gradients(float max_norm, float precomputed_norm) {
    if (max_norm <= 0.0f || precomputed_norm <= max_norm) {
        return precomputed_norm;
    }

    float clip_coef = max_norm / (precomputed_norm + 1e-6f);

    for (auto& param : parameter_groups) {
        if (param.gradients) {
            for (int i = 0; i < param.gradients->rows; i++) {
                for (int j = 0; j < param.gradients->cols; j++) {
                    (*param.gradients)(i, j) *= clip_coef;
                }
            }
        }
    }

    return precomputed_norm;
}

// SGD update
void Optimizer::step_sgd(ParameterGroup& param) {
    for (int i = 0; i < param.weights->rows; i++) {
        for (int j = 0; j < param.weights->cols; j++) {
            float grad = (*param.gradients)(i, j);

            // Apply weight decay (L2 regularization)
            if (weight_decay > 0.0f) {
                grad += weight_decay * (*param.weights)(i, j);
            }

            // Update weights
            (*param.weights)(i, j) -= learning_rate * grad;
        }
    }
}

// SGD with momentum update
void Optimizer::step_sgd_momentum(ParameterGroup& param) {
    for (int i = 0; i < param.weights->rows; i++) {
        for (int j = 0; j < param.weights->cols; j++) {
            float grad = (*param.gradients)(i, j);

            // Apply weight decay (L2 regularization)
            if (weight_decay > 0.0f) {
                grad += weight_decay * (*param.weights)(i, j);
            }

            // Update momentum
            param.momentum(i, j) = momentum_beta * param.momentum(i, j) + grad;

            // Update weights
            (*param.weights)(i, j) -= learning_rate * param.momentum(i, j);
        }
    }
}

// Adam update
void Optimizer::step_adam(ParameterGroup& param) {
    param.step++;

    // Bias correction terms
    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(param.step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(param.step));

    for (int i = 0; i < param.weights->rows; i++) {
        for (int j = 0; j < param.weights->cols; j++) {
            float grad = (*param.gradients)(i, j);

            // Apply weight decay (L2 regularization - coupled with gradients)
            if (weight_decay > 0.0f) {
                grad += weight_decay * (*param.weights)(i, j);
            }

            // Update biased first moment estimate (momentum)
            param.momentum(i, j) = beta1 * param.momentum(i, j) + (1.0f - beta1) * grad;

            // Update biased second moment estimate (velocity)
            param.velocity(i, j) = beta2 * param.velocity(i, j) + (1.0f - beta2) * grad * grad;

            // Compute bias-corrected estimates
            float m_hat = param.momentum(i, j) / bias_correction1;
            float v_hat = param.velocity(i, j) / bias_correction2;

            // Update weights
            (*param.weights)(i, j) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
}

// AdamW update
void Optimizer::step_adamw(ParameterGroup& param) {
    param.step++;

    // Bias correction terms
    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(param.step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(param.step));

    for (int i = 0; i < param.weights->rows; i++) {
        for (int j = 0; j < param.weights->cols; j++) {
            float grad = (*param.gradients)(i, j);

            // Update biased first moment estimate (momentum)
            param.momentum(i, j) = beta1 * param.momentum(i, j) + (1.0f - beta1) * grad;

            // Update biased second moment estimate (velocity)
            param.velocity(i, j) = beta2 * param.velocity(i, j) + (1.0f - beta2) * grad * grad;

            // Compute bias-corrected estimates
            float m_hat = param.momentum(i, j) / bias_correction1;
            float v_hat = param.velocity(i, j) / bias_correction2;

            // Update weights with decoupled weight decay
            // AdamW applies weight decay directly to weights, not gradients
            (*param.weights)(i, j) -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon) +
                                                       weight_decay * (*param.weights)(i, j));
        }
    }
}

// Perform optimization step
void Optimizer::step() {
    global_step++;

    for (auto& param : parameter_groups) {
        switch (type) {
            case OptimizerType::SGD:
                step_sgd(param);
                break;

            case OptimizerType::SGD_MOMENTUM:
                step_sgd_momentum(param);
                break;

            case OptimizerType::ADAM:
                step_adam(param);
                break;

            case OptimizerType::ADAMW:
                step_adamw(param);
                break;

            default:
                throw std::runtime_error("Unknown optimizer type");
        }
    }
}

// Get total parameters
size_t Optimizer::total_parameters() const {
    size_t total = 0;
    for (const auto& param : parameter_groups) {
        if (param.weights) {
            total += param.weights->rows * param.weights->cols;
        }
    }
    return total;
}
// NOLINTEND(readability-convert-member-functions-to-static)

// Get optimizer name
const char* Optimizer::get_optimizer_name() const {
    switch (type) {
        case OptimizerType::SGD:
            return "SGD";
        case OptimizerType::SGD_MOMENTUM:
            return "SGD+Momentum";
        case OptimizerType::ADAM:
            return "Adam";
        case OptimizerType::ADAMW:
            return "AdamW";
        default:
            return "Unknown";
    }
}

// Reset optimizer state
void Optimizer::reset_state() {
    global_step = 0;

    for (auto& param : parameter_groups) {
        param.step = 0;

        if (param.momentum.rows > 0) {
            param.momentum.fill(0.0f);
        }

        if (param.velocity.rows > 0) {
            param.velocity.fill(0.0f);
        }
    }
}
