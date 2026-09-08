// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

#include "LayerNorm.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "Optimizer.hpp"

LayerNorm::LayerNorm(int dim, float epsilon) : eps(epsilon) {
    // Initialize gamma to 1.0 (identity scale)
    gamma = Matrix(1, dim);
    for (int j = 0; j < dim; ++j) {
        gamma(0, j) = 1.0f;
    }

    // Initialize beta to 0.0 (no shift)
    beta = Matrix(1, dim);
    for (int j = 0; j < dim; ++j) {
        beta(0, j) = 0.0f;
    }

    // Initialize gradient matrices
    gamma_grad = Matrix(1, dim);
    beta_grad = Matrix(1, dim);

    // Set default learning rate
    learning_rate = 0.001f;
}

Matrix LayerNorm::forward(const Matrix& input) {
    int batch_size = input.rows;
    int dim = input.cols;

    // Cache input for backward pass
    cached_input = input;

    // Initialize output and normalized matrices
    Matrix output(batch_size, dim);
    cached_normalized = Matrix(batch_size, dim);

    // Resize cached mean and variance vectors
    cached_mean.resize(batch_size);
    cached_var.resize(batch_size);

    // Process each sample (row) independently
    for (int i = 0; i < batch_size; ++i) {
        // Compute mean for this sample
        float mean = 0.0f;
        for (int j = 0; j < dim; ++j) {
            mean += input(i, j);
        }
        mean /= static_cast<float>(dim);
        cached_mean[i] = mean;

        // Compute variance for this sample
        float var = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float diff = input(i, j) - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(dim);
        cached_var[i] = var;

        // Normalize and apply affine transformation
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < dim; ++j) {
            // Normalize
            float normalized = (input(i, j) - mean) * inv_std;
            cached_normalized(i, j) = normalized;

            // Apply affine transformation: gamma * normalized + beta
            output(i, j) = gamma(0, j) * normalized + beta(0, j);
        }
    }

    return output;
}

Matrix LayerNorm::backward(const Matrix& grad_output) {
    int batch_size = grad_output.rows;
    int dim = grad_output.cols;

    // Compute gradients for gamma and beta
    for (int j = 0; j < dim; ++j) {
        float gamma_g = 0.0f;
        float beta_g = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            gamma_g += grad_output(i, j) * cached_normalized(i, j);
            beta_g += grad_output(i, j);
        }
        gamma_grad(0, j) = gamma_g;
        beta_grad(0, j) = beta_g;
    }

    // Compute gradient w.r.t. input
    Matrix grad_input(batch_size, dim);

    for (int i = 0; i < batch_size; ++i) {
        float inv_std = 1.0f / std::sqrt(cached_var[i] + eps);

        // Compute gradient w.r.t. normalized values
        std::vector<float> grad_normalized(dim);
        for (int j = 0; j < dim; ++j) {
            grad_normalized[j] = grad_output(i, j) * gamma(0, j);
        }

        // Compute gradient w.r.t. variance
        float grad_var = 0.0f;
        for (int j = 0; j < dim; ++j) {
            grad_var += grad_normalized[j] * (cached_input(i, j) - cached_mean[i]);
        }
        grad_var *= -0.5f * inv_std * inv_std * inv_std;

        // Compute gradient w.r.t. mean
        float grad_mean = 0.0f;
        for (int j = 0; j < dim; ++j) {
            grad_mean += grad_normalized[j] * (-inv_std);
        }

        float sum_diff = 0.0f;
        for (int j = 0; j < dim; ++j) {
            sum_diff += (cached_input(i, j) - cached_mean[i]);
        }
        grad_mean += grad_var * (-2.0f / static_cast<float>(dim)) * sum_diff;

        // Compute gradient w.r.t. input
        for (int j = 0; j < dim; ++j) {
            grad_input(i, j) =
                grad_normalized[j] * inv_std +
                grad_var * 2.0f * (cached_input(i, j) - cached_mean[i]) / static_cast<float>(dim) +
                grad_mean / static_cast<float>(dim);
        }
    }

    return grad_input;
}

void LayerNorm::set_optimizer(Optimizer* opt) {
    optimizer = opt;
    if (optimizer) {
        register_parameters();
    }
}

void LayerNorm::register_parameters() {
    if (!optimizer) {
        return;
    }

    optimizer->add_parameter_group(&gamma, &gamma_grad);
    optimizer->add_parameter_group(&beta, &beta_grad);
}

void LayerNorm::update_weights() {
    if (optimizer) {
        // Use optimizer for updates
        optimizer->step();
    } else {
        // Fallback to simple gradient descent
        gamma.apply_gradients(gamma_grad, learning_rate);
        beta.apply_gradients(beta_grad, learning_rate);
    }
    zero_grad();
}

void LayerNorm::zero_grad() {
    // Zero out gradient matrices
    for (int j = 0; j < gamma_grad.cols; ++j) {
        gamma_grad(0, j) = 0.0f;
        beta_grad(0, j) = 0.0f;
    }
}

void LayerNorm::set_gamma(const Matrix& new_gamma) {
    if (new_gamma.rows != 1 || new_gamma.cols != gamma.cols) {
        std::cerr << "Error: gamma dimensions mismatch. Expected [1, " << gamma.cols << "], got ["
                  << new_gamma.rows << ", " << new_gamma.cols << "]" << '\n';
        return;
    }
    gamma = new_gamma;
}

void LayerNorm::set_beta(const Matrix& new_beta) {
    if (new_beta.rows != 1 || new_beta.cols != beta.cols) {
        std::cerr << "Error: beta dimensions mismatch. Expected [1, " << beta.cols << "], got ["
                  << new_beta.rows << ", " << new_beta.cols << "]" << '\n';
        return;
    }
    beta = new_beta;
}

void LayerNorm::print_config(const std::string& name) const {
    std::cout << "\n" << name << " Configuration:" << '\n';
    std::cout << "  Dimension: " << gamma.cols << '\n';
    std::cout << "  Epsilon: " << eps << '\n';
    std::cout << "  Learning Rate: " << learning_rate << '\n';
    std::cout << "  Gamma range: [";

    float min_gamma = gamma(0, 0);
    float max_gamma = gamma(0, 0);
    for (int j = 0; j < gamma.cols; ++j) {
        if (gamma(0, j) < min_gamma) {
            min_gamma = gamma(0, j);
        }
        if (gamma(0, j) > max_gamma) {
            max_gamma = gamma(0, j);
        }
    }
    std::cout << min_gamma << ", " << max_gamma << "]" << '\n';

    std::cout << "  Beta range: [";
    float min_beta = beta(0, 0);
    float max_beta = beta(0, 0);
    for (int j = 0; j < beta.cols; ++j) {
        if (beta(0, j) < min_beta) {
            min_beta = beta(0, j);
        }
        if (beta(0, j) > max_beta) {
            max_beta = beta(0, j);
        }
    }
    std::cout << min_beta << ", " << max_beta << "]" << '\n';
}

void LayerNorm::save_weights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write dimension
    int dim = gamma.cols;
    file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&eps), sizeof(float));

    // Write gamma
    for (int j = 0; j < gamma.cols; ++j) {
        file.write(reinterpret_cast<const char*>(&gamma(0, j)), sizeof(float));
    }

    // Write beta
    for (int j = 0; j < beta.cols; ++j) {
        file.write(reinterpret_cast<const char*>(&beta(0, j)), sizeof(float));
    }

    file.close();
    std::cout << "Saved LayerNorm weights to " << filename << '\n';
}

void LayerNorm::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Read and verify dimensions
    int loaded_dim = 0;
    float loaded_eps = NAN;
    file.read(reinterpret_cast<char*>(&loaded_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_eps), sizeof(float));

    if (loaded_dim != gamma.cols) {
        throw std::runtime_error("Dimension mismatch: file has " + std::to_string(loaded_dim) +
                                 ", expected " + std::to_string(gamma.cols));
    }

    // Read gamma
    for (int j = 0; j < gamma.cols; ++j) {
        file.read(reinterpret_cast<char*>(&gamma(0, j)), sizeof(float));
    }

    // Read beta
    for (int j = 0; j < beta.cols; ++j) {
        file.read(reinterpret_cast<char*>(&beta(0, j)), sizeof(float));
    }

    file.close();
    std::cout << "Loaded LayerNorm weights from " << filename << '\n';
}

#ifdef ADAI_ENABLE_GPU
void LayerNorm::gpu_upload_weights() {
    const int dim = gamma.cols;
    if (!gpu_)
        gpu_ = std::make_unique<GPUState>(dim);
    gpu_->gamma_g.upload(gamma.data[0].data(), dim);
    gpu_->beta_g.upload(beta.data[0].data(), dim);
}

void LayerNorm::gpu_download_grads() {
    if (!gpu_)
        return;
    const int dim = gamma.cols;
    std::vector<float> tmp(dim);
    gpu_->dgamma.download(tmp.data(), dim);
    for (int j = 0; j < dim; ++j)
        gamma_grad.data[0][j] += tmp[j];
    gpu_->dbeta.download(tmp.data(), dim);
    for (int j = 0; j < dim; ++j)
        beta_grad.data[0][j] += tmp[j];
}

void LayerNorm::gpu_zero_grads() {
    if (!gpu_)
        return;
    gpu_->dgamma.zero();
    gpu_->dbeta.zero();
}

adai::gpu::GPUMatrix LayerNorm::gpu_forward(const adai::gpu::GPUMatrix& input) {
    const int rows = input.rows;
    const int dim = input.cols;
    if (!gpu_ || gpu_->gamma_g.cols != dim)
        gpu_upload_weights();

    // Resize cached buffers if sequence length changed
    if (gpu_->normed.rows != rows || gpu_->normed.cols != dim) {
        gpu_->normed = adai::gpu::GPUMatrix(rows, dim);
        gpu_->mean = adai::gpu::GPUMatrix(rows, 1);
        gpu_->rstd = adai::gpu::GPUMatrix(rows, 1);
    }

    return input.layer_norm(gpu_->gamma_g, gpu_->beta_g, eps, gpu_->normed, gpu_->mean, gpu_->rstd);
}

adai::gpu::GPUMatrix LayerNorm::gpu_backward(const adai::gpu::GPUMatrix& dout) {
    return dout.layer_norm_backward(gpu_->normed, gpu_->gamma_g, gpu_->mean, gpu_->rstd,
                                    gpu_->dgamma, gpu_->dbeta);
}
#endif
