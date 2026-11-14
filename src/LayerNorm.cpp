// @adai-status: stable        (TD-195 — backward() gradient members now accumulate, not overwrite, across calls)
// @adai-version: 1.3.0
// @adai-reviewed: 2026-11-13

#include "LayerNorm.hpp"
#include "BinaryIO.hpp"
#include "Logger.hpp"
#include "gpu/GPUParamSync.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include "Optimizer.hpp"

namespace {

std::string shape(int rows, int cols) {
    return "[" + std::to_string(rows) + ", " + std::to_string(cols) + "]";
}

void require_epsilon(float epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        throw std::invalid_argument("LayerNorm: epsilon must be a positive finite number, got " +
                                    std::to_string(epsilon));
    }
}

}  // namespace

LayerNorm::LayerNorm(int dim, float epsilon) : eps(epsilon) {
    // TD-426: a zero or negative dimension used to build empty matrices that failed later.
    if (dim <= 0) {
        throw std::invalid_argument("LayerNorm: dim must be positive, got " + std::to_string(dim));
    }
    require_epsilon(epsilon);
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
    // TD-426: a width mismatch used to surface as Matrix's generic out_of_range.
    if (input.cols != gamma.cols) {
        throw std::invalid_argument("LayerNorm::forward: input " + shape(input.rows, input.cols) +
                                    " has " + std::to_string(input.cols) + " columns, expected " +
                                    std::to_string(gamma.cols));
    }
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
    // TD-426: the gradient must match the cached forward input.
    if (cached_mean.empty() || cached_input.rows != static_cast<int>(cached_mean.size())) {
        throw std::logic_error("LayerNorm::backward called without a forward pass");
    }
    if (grad_output.rows != cached_input.rows || grad_output.cols != cached_input.cols) {
        throw std::invalid_argument(
            "LayerNorm::backward: gradient " + shape(grad_output.rows, grad_output.cols) +
            " doesn't match the forward input " + shape(cached_input.rows, cached_input.cols));
    }
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
        gamma_grad(0, j) += gamma_g;
        beta_grad(0, j) += beta_g;
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

    // TD-469: norm gains and biases aren't weight-decayed (decay pulled the gains toward 0).
    optimizer->add_parameter_group(&gamma, &gamma_grad, /*apply_weight_decay=*/false);
    optimizer->add_parameter_group(&beta, &beta_grad, /*apply_weight_decay=*/false);
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

// TD-426: a shape mismatch throws (it used to print to std::cerr and keep the old values), and a
// device mirror, if there is one, gets the new values.
void LayerNorm::set_gamma(const Matrix& new_gamma) {
    if (new_gamma.rows != 1 || new_gamma.cols != gamma.cols) {
        throw std::invalid_argument("LayerNorm::set_gamma: expected " + shape(1, gamma.cols) +
                                    ", got " + shape(new_gamma.rows, new_gamma.cols));
    }
    gamma = new_gamma;
    refresh_gpu_mirror();
}

void LayerNorm::set_beta(const Matrix& new_beta) {
    if (new_beta.rows != 1 || new_beta.cols != beta.cols) {
        throw std::invalid_argument("LayerNorm::set_beta: expected " + shape(1, beta.cols) +
                                    ", got " + shape(new_beta.rows, new_beta.cols));
    }
    beta = new_beta;
    refresh_gpu_mirror();
}

void LayerNorm::refresh_gpu_mirror() {
#ifdef ADAI_ENABLE_GPU
    if (gpu_) {
        gpu_upload_weights();
    }
#endif
}

void LayerNorm::print_config(const std::string& name) const {
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
    adai::Logger::info("{}: dim {}, epsilon {}, learning rate {}, gamma [{}, {}], beta [{}, {}]",
                       name, gamma.cols, eps, learning_rate, min_gamma, max_gamma, min_beta,
                       max_beta);
}

void LayerNorm::save_weights(const std::string& filename) const {
    adai::binio::Writer file(filename);  // TD-423: checked writes
    file.value(static_cast<int>(gamma.cols), "dim");
    file.value(eps, "eps");
    file.matrix(gamma, "gamma");
    file.matrix(beta, "beta");
    file.close();
    adai::Logger::debug("Saved LayerNorm weights to {}", filename);
}

void LayerNorm::load_weights(const std::string& filename) {
    // TD-423/TD-426: checked reads; a truncated file used to keep the old values past the cut.
    // gamma, beta and epsilon change only after the whole file has been read; the file's epsilon is
    // the one the weights were trained with (it used to be read and discarded).
    adai::binio::Reader file(filename);
    const int loaded_dim = file.value<int>("dim");
    const float loaded_eps = file.value<float>("eps");
    if (!std::isfinite(loaded_eps) || loaded_eps <= 0.0f) {
        throw std::runtime_error(filename + ": epsilon " + std::to_string(loaded_eps) +
                                 " is not a positive finite number");
    }

    if (loaded_dim != gamma.cols) {
        throw std::runtime_error("Dimension mismatch: file has " + std::to_string(loaded_dim) +
                                 ", expected " + std::to_string(gamma.cols));
    }

    Matrix loaded_gamma = file.matrix(1, gamma.cols, "gamma");
    Matrix loaded_beta = file.matrix(1, beta.cols, "beta");
    file.expect_end();
    gamma = std::move(loaded_gamma);
    beta = std::move(loaded_beta);
    eps = loaded_eps;
    refresh_gpu_mirror();
    adai::Logger::debug("Loaded LayerNorm weights from {}", filename);
}

#ifdef ADAI_ENABLE_GPU
void LayerNorm::gpu_upload_weights() {
    const int dim = gamma.cols;
    if (!gpu_)
        gpu_ = std::make_unique<GPUState>(dim);
    gpu_->gamma_g.upload(gamma.data[0].data(), dim);
    gpu_->beta_g.upload(beta.data[0].data(), dim);
}

void LayerNorm::gpu_collect_parameters(std::vector<adai::gpu::GPUParamRef>& out, int layer) {
    if (!gpu_)
        return;
    adai::gpu::add_param(out, gamma, gpu_->gamma_g, gpu_->dgamma, layer);
    adai::gpu::add_param(out, beta, gpu_->beta_g, gpu_->dbeta, layer);
}

void LayerNorm::gpu_download_weights() {
    if (!gpu_)
        return;
    adai::gpu::download_into(gpu_->gamma_g, gamma);
    adai::gpu::download_into(gpu_->beta_g, beta);
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
    // TD-426: gpu_ is null before the first gpu_forward(), and dout must match its cached rows.
    if (!gpu_) {
        throw std::logic_error("LayerNorm::gpu_backward called without a gpu_forward pass");
    }
    if (dout.rows != gpu_->normed.rows || dout.cols != gpu_->normed.cols) {
        throw std::invalid_argument(
            "LayerNorm::gpu_backward: gradient " + shape(dout.rows, dout.cols) +
            " doesn't match the forward input " + shape(gpu_->normed.rows, gpu_->normed.cols));
    }
    return dout.layer_norm_backward(gpu_->normed, gpu_->gamma_g, gpu_->mean, gpu_->rstd,
                                    gpu_->dgamma, gpu_->dbeta);
}
#endif
