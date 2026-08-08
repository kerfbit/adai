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

      W1(d_model, d_ff),
      W2(d_ff, d_model),
      b1(1, d_ff),
      b2(1, d_model),
      W1_grad(d_model, d_ff),
      W2_grad(d_ff, d_model),
      b1_grad(1, d_ff),
      b2_grad(1, d_model) {
    // Xavier/He initialization for weights
    // Scale for GELU: sqrt(2 / d_model)
    float scale = std::sqrt(2.0f / static_cast<float>(d_model));

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
    float scale2 = std::sqrt(2.0f / static_cast<float>(d_ff));
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

    // Fire activation hook if registered (used for saturation tracking, TD-013)
    if (activation_hook_) {
        activation_hook_(cached_hidden_activated);
    }

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
    if (!optimizer) {
        return;
    }

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
    std::cout << "Saved FeedForward weights to " << filename << '\n';
}

void FeedForward::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Read dimensions
    int loaded_d_model = 0, loaded_d_ff = 0;
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
    std::cout << "Loaded FeedForward weights from " << filename << '\n';
}

void FeedForward::print_config(const std::string& name) const {
    std::cout << name << " Configuration:" << '\n';
    std::cout << "  Model Dimension (d_model): " << d_model << '\n';
    std::cout << "  Feed-Forward Dimension (d_ff): " << d_ff << '\n';
    std::cout << "  Expansion Ratio: " << static_cast<float>(d_ff) / static_cast<float>(d_model)
              << "x" << '\n';
    std::cout << "  Total Parameters: " << (d_model * d_ff + d_ff + d_ff * d_model + d_model)
              << '\n';
    std::cout << "  W1 Parameters: " << (d_model * d_ff) << '\n';
    std::cout << "  W2 Parameters: " << (d_ff * d_model) << '\n';
    std::cout << "  Bias Parameters: " << (d_ff + d_model) << '\n';
    std::cout << "  Memory Usage: " << std::fixed << std::setprecision(2)
              << static_cast<float>(d_model * d_ff + d_ff * d_model + d_ff + d_model) *
                     sizeof(float) / 1024.0f / 1024.0f
              << " MB" << '\n';
    std::cout << "  Learning Rate: " << learning_rate << '\n';
}

#ifdef ADAI_ENABLE_GPU
void FeedForward::gpu_upload_weights() {
    if (!gpu_)
        gpu_ = std::make_unique<GPUState>(d_model, d_ff);
    auto flat = [](const Matrix& m) {
        std::vector<float> v;
        v.reserve(m.rows * m.cols);
        for (const auto& row : m.data)
            v.insert(v.end(), row.begin(), row.end());
        return v;
    };
    auto f1 = flat(W1);
    gpu_->W1_g.upload(f1.data(), d_model * d_ff);
    auto f2 = flat(W2);
    gpu_->W2_g.upload(f2.data(), d_ff * d_model);
    auto fb1 = flat(b1);
    gpu_->b1_g.upload(fb1.data(), d_ff);
    auto fb2 = flat(b2);
    gpu_->b2_g.upload(fb2.data(), d_model);
}

void FeedForward::gpu_download_grads() {
    if (!gpu_)
        return;
    std::vector<float> tmp;
    auto add_back = [](const std::vector<float>& src, Matrix& dst) {
        int idx = 0;
        for (auto& row : dst.data)
            for (auto& v : row)
                v += src[idx++];
    };
    tmp.resize(d_model * d_ff);
    gpu_->dW1.download(tmp.data(), d_model * d_ff);
    add_back(tmp, W1_grad);
    tmp.resize(d_ff * d_model);
    gpu_->dW2.download(tmp.data(), d_ff * d_model);
    add_back(tmp, W2_grad);
    tmp.resize(d_ff);
    gpu_->db1.download(tmp.data(), d_ff);
    add_back(tmp, b1_grad);
    tmp.resize(d_model);
    gpu_->db2.download(tmp.data(), d_model);
    add_back(tmp, b2_grad);
}

void FeedForward::gpu_zero_grads() {
    if (!gpu_)
        return;
    gpu_->dW1.zero();
    gpu_->dW2.zero();
    gpu_->db1.zero();
    gpu_->db2.zero();
}

adai::gpu::GPUMatrix FeedForward::gpu_forward(const adai::gpu::GPUMatrix& input) {
    if (!gpu_)
        gpu_upload_weights();
    const int seq = input.rows;
    // Resize caches if seq length changed
    if (gpu_->cached_input.rows != seq) {
        gpu_->cached_input = adai::gpu::GPUMatrix(seq, d_model);
        gpu_->cached_hidden = adai::gpu::GPUMatrix(seq, d_ff);
        gpu_->cached_act = adai::gpu::GPUMatrix(seq, d_ff);
    }
    // Copy input for backward
    adai::gpu::matrix_copy_device_to_device_gpu(input.device_ptr(), gpu_->cached_input.device_ptr(),
                                                seq * d_model);

    // hidden = input * W1 + b1
    adai::gpu::GPUMatrix hidden = input * gpu_->W1_g;
    adai::gpu::matrix_add_row_bias_gpu(hidden.device_ptr(), gpu_->b1_g.device_ptr(),
                                       gpu_->cached_hidden.device_ptr(), seq, d_ff);
    // act = GELU(hidden)
    adai::gpu::matrix_copy_device_to_device_gpu(gpu_->cached_hidden.device_ptr(),
                                                gpu_->cached_act.device_ptr(), seq * d_ff);
    gpu_->cached_act.apply_activation_inplace(adai::gpu::ActivationType::GELU);
    if (gpu_activation_stats_hook_) {
        gpu_activation_stats_hook_(gpu_->cached_act.count_below_threshold(0.01f));
    }

    // output = act * W2 + b2
    adai::gpu::GPUMatrix out = gpu_->cached_act * gpu_->W2_g;
    adai::gpu::GPUMatrix result(seq, d_model);
    adai::gpu::matrix_add_row_bias_gpu(out.device_ptr(), gpu_->b2_g.device_ptr(),
                                       result.device_ptr(), seq, d_model);
    return result;
}

adai::gpu::GPUMatrix FeedForward::gpu_backward(const adai::gpu::GPUMatrix& dout) {
    const int seq = dout.rows;

    // dW2 += act^T * dout
    adai::gpu::GPUMatrix act_T = gpu_->cached_act.transpose();
    adai::gpu::GPUMatrix dW2_step = act_T * dout;
    gpu_->dW2.add_inplace(dW2_step);

    // db2 += sum_rows(dout)
    adai::gpu::GPUMatrix db2_step = dout.sum_rows();
    gpu_->db2.add_inplace(db2_step);

    // d_act = dout * W2^T
    adai::gpu::GPUMatrix W2_T = gpu_->W2_g.transpose();
    adai::gpu::GPUMatrix d_act = dout * W2_T;

    // d_hidden = GELU'(cached_hidden) * d_act
    adai::gpu::GPUMatrix d_hidden = gpu_->cached_hidden.gelu_backward(d_act);

    // dW1 += input^T * d_hidden
    adai::gpu::GPUMatrix in_T = gpu_->cached_input.transpose();
    adai::gpu::GPUMatrix dW1_step = in_T * d_hidden;
    gpu_->dW1.add_inplace(dW1_step);

    // db1 += sum_rows(d_hidden)
    adai::gpu::GPUMatrix db1_step = d_hidden.sum_rows();
    gpu_->db1.add_inplace(db1_step);

    // d_input = d_hidden * W1^T
    adai::gpu::GPUMatrix W1_T = gpu_->W1_g.transpose();
    return d_hidden * W1_T;
}
#endif
