#include "FeedForward.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include "Activation.hpp"

FeedForward::FeedForward(int d_model, int d_ff)
    : d_model(d_model),
      d_ff(d_ff),
      learning_rate(0.001f),
      W1(d_model, d_ff),
      W2(d_ff, d_model),
      b1(1, d_ff),
      b2(1, d_model),
      W1_grad(d_model, d_ff),
      W2_grad(d_ff, d_model),
      b1_grad(1, d_ff),
      b2_grad(1, d_model),
      optimizer(nullptr) {
    // Xavier/He initialization for weights
    // Scale for GELU: sqrt(2 / d_model)
    float scale = std::sqrt(2.0f / d_model);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);

    // Initialize W1
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            W1(i, j) = dist(gen);
        }
    }

    // Initialize W2 with scale adjusted for d_ff
    float scale2 = std::sqrt(2.0f / d_ff);
    std::normal_distribution<float> dist2(0.0f, scale2);
    for (int i = 0; i < d_ff; ++i) {
        for (int j = 0; j < d_model; ++j) {
            W2(i, j) = dist2(gen);
        }
    }

    // Zero initialization for biases
    for (int i = 0; i < d_ff; ++i) {
        b1(0, i) = 0.0f;
    }
    for (int i = 0; i < d_model; ++i) {
        b2(0, i) = 0.0f;
    }

    // Zero gradients
    zero_grad();
}

Matrix FeedForward::forward(const Matrix& input) {
    // Validate input dimensions
    if (input.cols != d_model) {
        throw std::invalid_argument("Input dimension mismatch. Expected " +
                                    std::to_string(d_model) + " but got " +
                                    std::to_string(input.cols));
    }

    cached_input = input;

    // First linear transformation: input * W1
    Matrix hidden = input * W1;

    // Add bias b1 (broadcast across batch)
    for (int i = 0; i < hidden.rows; ++i) {
        for (int j = 0; j < hidden.cols; ++j) {
            hidden(i, j) += b1(0, j);
        }
    }

    cached_hidden = hidden;

    // Apply GELU activation
    Matrix hidden_activated = Activation::gelu(hidden);
    cached_hidden_activated = hidden_activated;

    // Second linear transformation: hidden * W2
    Matrix output = hidden_activated * W2;

    // Add bias b2 (broadcast across batch)
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output(i, j) += b2(0, j);
        }
    }

    return output;
}

Matrix FeedForward::backward(const Matrix& grad_output) {
    // Validate gradient dimensions
    if (grad_output.rows != cached_input.rows || grad_output.cols != d_model) {
        throw std::invalid_argument(
            "Gradient dimensions must match forward pass output dimensions");
    }

    // Gradient w.r.t. b2: sum over batch dimension
    for (int j = 0; j < d_model; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < grad_output.rows; ++i) {
            sum += grad_output(i, j);
        }
        b2_grad(0, j) = sum;
    }

    // Gradient w.r.t. W2: cached_hidden_activated^T * grad_output
    W2_grad = cached_hidden_activated.transpose() * grad_output;

    // Gradient w.r.t. hidden_activated: grad_output * W2^T
    Matrix grad_hidden_activated = grad_output * W2.transpose();

    // Gradient through GELU activation
    Matrix gelu_grad = Activation::gelu_derivative(cached_hidden);
    Matrix grad_hidden = grad_hidden_activated.hadamard(gelu_grad);

    // Gradient w.r.t. b1: sum over batch dimension
    for (int j = 0; j < d_ff; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < grad_hidden.rows; ++i) {
            sum += grad_hidden(i, j);
        }
        b1_grad(0, j) = sum;
    }

    // Gradient w.r.t. W1: cached_input^T * grad_hidden
    W1_grad = cached_input.transpose() * grad_hidden;

    // Gradient w.r.t. input: grad_hidden * W1^T
    Matrix grad_input = grad_hidden * W1.transpose();

    return grad_input;
}

void FeedForward::set_optimizer(Optimizer* opt) {
    optimizer = opt;
    if (optimizer) {
        register_parameters();
    }
}

void FeedForward::register_parameters() {
    if (!optimizer)
        return;

    // Register all weight matrices and biases with optimizer
    optimizer->add_parameter_group(&W1, &W1_grad);
    optimizer->add_parameter_group(&W2, &W2_grad);
    optimizer->add_parameter_group(&b1, &b1_grad);
    optimizer->add_parameter_group(&b2, &b2_grad);
}

void FeedForward::update_weights() {
    if (optimizer) {
        // Use advanced optimization (Adam, AdamW, etc.)
        optimizer->step();
    } else {
        // Fallback to simple gradient descent for backward compatibility
        // Apply gradients using Matrix's apply_gradients method
        W1.apply_gradients(W1_grad, learning_rate);
        W2.apply_gradients(W2_grad, learning_rate);

        // Update biases manually
        for (int i = 0; i < d_ff; ++i) {
            b1(0, i) -= learning_rate * b1_grad(0, i);
        }
        for (int i = 0; i < d_model; ++i) {
            b2(0, i) -= learning_rate * b2_grad(0, i);
        }
    }

    // Zero gradients after update
    zero_grad();
}

void FeedForward::zero_grad() {
    // Zero weight gradients
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            W1_grad(i, j) = 0.0f;
        }
    }

    for (int i = 0; i < d_ff; ++i) {
        for (int j = 0; j < d_model; ++j) {
            W2_grad(i, j) = 0.0f;
        }
    }

    // Zero bias gradients
    for (int i = 0; i < d_ff; ++i) {
        b1_grad(0, i) = 0.0f;
    }
    for (int i = 0; i < d_model; ++i) {
        b2_grad(0, i) = 0.0f;
    }
}

float FeedForward::get_gradient_norm() const {
    float norm = 0.0f;

    // W1 gradients
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            norm += W1_grad(i, j) * W1_grad(i, j);
        }
    }

    // W2 gradients
    for (int i = 0; i < d_ff; ++i) {
        for (int j = 0; j < d_model; ++j) {
            norm += W2_grad(i, j) * W2_grad(i, j);
        }
    }

    // b1 gradients
    for (int i = 0; i < d_ff; ++i) {
        norm += b1_grad(0, i) * b1_grad(0, i);
    }

    // b2 gradients
    for (int i = 0; i < d_model; ++i) {
        norm += b2_grad(0, i) * b2_grad(0, i);
    }

    return std::sqrt(norm);
}

void FeedForward::clip_gradients(float max_norm) {
    float norm = get_gradient_norm();

    if (norm > max_norm) {
        float scale = max_norm / norm;

        // Scale W1 gradients
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_ff; ++j) {
                W1_grad(i, j) *= scale;
            }
        }

        // Scale W2 gradients
        for (int i = 0; i < d_ff; ++i) {
            for (int j = 0; j < d_model; ++j) {
                W2_grad(i, j) *= scale;
            }
        }

        // Scale b1 gradients
        for (int i = 0; i < d_ff; ++i) {
            b1_grad(0, i) *= scale;
        }

        // Scale b2 gradients
        for (int i = 0; i < d_model; ++i) {
            b2_grad(0, i) *= scale;
        }
    }
}

void FeedForward::save_weights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write dimensions
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_ff), sizeof(int));

    // Write W1
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            file.write(reinterpret_cast<const char*>(&W1(i, j)), sizeof(float));
        }
    }

    // Write W2
    for (int i = 0; i < d_ff; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.write(reinterpret_cast<const char*>(&W2(i, j)), sizeof(float));
        }
    }

    // Write b1
    for (int i = 0; i < d_ff; ++i) {
        file.write(reinterpret_cast<const char*>(&b1(0, i)), sizeof(float));
    }

    // Write b2
    for (int i = 0; i < d_model; ++i) {
        file.write(reinterpret_cast<const char*>(&b2(0, i)), sizeof(float));
    }

    file.close();
    std::cout << "Saved FeedForward weights to " << filename << std::endl;
}

void FeedForward::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Read dimensions
    int loaded_d_model, loaded_d_ff;
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(int));

    // Validate dimensions
    if (loaded_d_model != d_model || loaded_d_ff != d_ff) {
        throw std::runtime_error("Dimension mismatch. Expected d_model=" + std::to_string(d_model) +
                                 ", d_ff=" + std::to_string(d_ff) +
                                 " but got d_model=" + std::to_string(loaded_d_model) +
                                 ", d_ff=" + std::to_string(loaded_d_ff));
    }

    // Read W1
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            file.read(reinterpret_cast<char*>(&W1(i, j)), sizeof(float));
        }
    }

    // Read W2
    for (int i = 0; i < d_ff; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.read(reinterpret_cast<char*>(&W2(i, j)), sizeof(float));
        }
    }

    // Read b1
    for (int i = 0; i < d_ff; ++i) {
        file.read(reinterpret_cast<char*>(&b1(0, i)), sizeof(float));
    }

    // Read b2
    for (int i = 0; i < d_model; ++i) {
        file.read(reinterpret_cast<char*>(&b2(0, i)), sizeof(float));
    }

    file.close();
    std::cout << "Loaded FeedForward weights from " << filename << std::endl;
}

void FeedForward::print_config(const std::string& name) const {
    std::cout << name << " Configuration:" << std::endl;
    std::cout << "  Model Dimension (d_model): " << d_model << std::endl;
    std::cout << "  Feed-Forward Dimension (d_ff): " << d_ff << std::endl;
    std::cout << "  Expansion Ratio: " << static_cast<float>(d_ff) / d_model << "x" << std::endl;
    std::cout << "  Total Parameters: " << (d_model * d_ff + d_ff + d_ff * d_model + d_model)
              << std::endl;
    std::cout << "  W1 Parameters: " << (d_model * d_ff) << std::endl;
    std::cout << "  W2 Parameters: " << (d_ff * d_model) << std::endl;
    std::cout << "  Bias Parameters: " << (d_ff + d_model) << std::endl;
    std::cout << "  Memory Usage: " << std::fixed << std::setprecision(2)
              << (d_model * d_ff + d_ff * d_model + d_ff + d_model) * sizeof(float) / 1024.0f /
                     1024.0f
              << " MB" << std::endl;
    std::cout << "  Learning Rate: " << learning_rate << std::endl;
}
