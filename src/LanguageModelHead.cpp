// @adai-status: stable
// @adai-version: 1.1.1
// @adai-reviewed: 2026-10-26

#include "LanguageModelHead.hpp"
#include "BinaryIO.hpp"
#include "gpu/GPUParamSync.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include "Activation.hpp"

LanguageModelHead::LanguageModelHead(int d_model, int vocab_size)
    : d_model(d_model),
      vocab_size(vocab_size),

      W_output(d_model, vocab_size),
      bias(1, vocab_size),
      W_output_grad(d_model, vocab_size),
      bias_grad(1, vocab_size) {
    // Xavier initialization for weights
    float scale = std::sqrt(2.0f / static_cast<float>(d_model + vocab_size));
    W_output.randomize(scale);

    // Zero initialization for bias
    bias.fill(0.0f);

    // Zero gradients
    W_output_grad.fill(0.0f);
    bias_grad.fill(0.0f);

    // TODO: See TD-425 in TECHNICAL_DEBT.md - std::cout in library code; int parameter count can overflow
    std::cout << "LanguageModelHead initialized:" << '\n';
    std::cout << "  Input dimension: " << d_model << '\n';
    std::cout << "  Vocabulary size: " << vocab_size << '\n';
    std::cout << "  Parameters: " << (d_model * vocab_size + vocab_size) << '\n';
}

Matrix LanguageModelHead::forward(const Matrix& input) {
    // Cache input for backward pass
    cached_input = input;

    // Linear projection: input * W_output
    Matrix output = input * W_output;

    // Add bias (broadcast across all positions)
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output(i, j) += bias(0, j);
        }
    }

    return output;
}

// TODO: See TD-425 in TECHNICAL_DEBT.md - no production caller
std::vector<float> LanguageModelHead::get_probabilities(const std::vector<float>& logits) {
    // Convert vector to matrix for softmax
    Matrix logits_matrix(1, static_cast<int>(logits.size()));
    for (size_t i = 0; i < logits.size(); ++i) {
        logits_matrix(0, static_cast<int>(i)) = logits[i];
    }

    // Apply softmax
    Matrix probs_matrix = Activation::softmax(logits_matrix);

    // Convert back to vector
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = probs_matrix(0, static_cast<int>(i));
    }

    return probs;
}

Matrix LanguageModelHead::backward(const Matrix& grad_output) {
    // Gradient w.r.t. W_output: input^T * grad_output
    Matrix grad_W = cached_input.transpose() * grad_output;

    // Accumulate gradients
    for (int i = 0; i < W_output_grad.rows; ++i) {
        for (int j = 0; j < W_output_grad.cols; ++j) {
            W_output_grad(i, j) += grad_W(i, j);
        }
    }

    // Gradient w.r.t. bias: sum over sequence dimension
    for (int j = 0; j < bias_grad.cols; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < grad_output.rows; ++i) {
            sum += grad_output(i, j);
        }
        bias_grad(0, j) += sum;
    }

    // Gradient w.r.t. input: grad_output * W_output^T
    Matrix grad_input = grad_output * W_output.transpose();

    return grad_input;
}

void LanguageModelHead::update_weights() {
    if (optimizer) {
        optimizer->step();
    } else {
        // Fallback to simple gradient descent
        W_output.apply_gradients(W_output_grad, learning_rate);
        bias.apply_gradients(bias_grad, learning_rate);
    }
    zero_grad();
}

void LanguageModelHead::zero_grad() {
    W_output_grad.fill(0.0f);
    bias_grad.fill(0.0f);
}

void LanguageModelHead::save(const std::string& filepath) const {
    // TD-423: checked writes; a failed write or flush throws instead of leaving a short file.
    adai::binio::Writer file(filepath);
    file.value(d_model, "d_model");
    file.value(vocab_size, "vocab_size");
    file.matrix(W_output, "W_output");
    file.matrix(bias, "bias");
    file.close();
}

void LanguageModelHead::load(const std::string& filepath) {
    // TD-423: a truncated .lm_head used to load with NaN weights (all-NaN logits). The whole file is
    // read and checked before anything changes, so a failed load leaves the head as it was.
    adai::binio::Reader file(filepath);
    const int loaded_d_model = file.value<int>("d_model");
    const int loaded_vocab_size = file.value<int>("vocab_size");
    if (loaded_d_model != d_model || loaded_vocab_size != vocab_size) {
        throw std::runtime_error("Dimension mismatch in saved model");
    }
    Matrix loaded_w = file.matrix(W_output.rows, W_output.cols, "W_output");
    Matrix loaded_bias = file.matrix(bias.rows, bias.cols, "bias");
    file.expect_end();
    W_output = std::move(loaded_w);
    bias = std::move(loaded_bias);
}

void LanguageModelHead::set_optimizer(Optimizer* opt) {
    optimizer = opt;
    if (optimizer) {
        register_parameters();
    }
}

void LanguageModelHead::register_parameters() {
    if (!optimizer) {
        return;
    }

    // Register weight matrix and bias
    optimizer->add_parameter_group(&W_output, &W_output_grad);
    optimizer->add_parameter_group(&bias, &bias_grad);
}

void LanguageModelHead::save_weights(const std::string& filename) const {
    // Wrapper for consistency with other components
    save(filename);
}

void LanguageModelHead::load_weights(const std::string& filename) {
    load(filename);
}

#ifdef ADAI_ENABLE_GPU
void LanguageModelHead::gpu_upload_weights() {
    if (!gpu_)
        gpu_ = std::make_unique<GPUState>(d_model, vocab_size);
    int n_w = d_model * vocab_size;
    std::vector<float> tmp;
    tmp.reserve(n_w);
    for (const auto& row : W_output.data)
        for (float v : row)
            tmp.push_back(v);
    gpu_->W_g.upload(tmp.data(), n_w);
    gpu_->b_g.upload(bias.data[0].data(), vocab_size);
}

void LanguageModelHead::gpu_collect_parameters(std::vector<adai::gpu::GPUParamRef>& out,
                                               int layer) {
    if (!gpu_)
        return;
    adai::gpu::add_param(out, W_output, gpu_->W_g, gpu_->dW, layer);
    adai::gpu::add_param(out, bias, gpu_->b_g, gpu_->db, layer);
}

void LanguageModelHead::gpu_download_weights() {
    if (!gpu_)
        return;
    adai::gpu::download_into(gpu_->W_g, W_output);
    adai::gpu::download_into(gpu_->b_g, bias);
}

void LanguageModelHead::gpu_download_grads() {
    if (!gpu_)
        return;
    {
        std::vector<float> tmp(d_model * vocab_size);
        gpu_->dW.download(tmp.data(), d_model * vocab_size);
        int idx = 0;
        for (auto& row : W_output_grad.data)
            for (auto& v : row)
                v += tmp[idx++];
    }
    {
        std::vector<float> tmp(vocab_size);
        gpu_->db.download(tmp.data(), vocab_size);
        for (int j = 0; j < vocab_size; ++j)
            bias_grad.data[0][j] += tmp[j];
    }
}

void LanguageModelHead::gpu_zero_grads() {
    if (!gpu_)
        return;
    gpu_->dW.zero();
    gpu_->db.zero();
}

adai::gpu::GPUMatrix LanguageModelHead::gpu_forward(const adai::gpu::GPUMatrix& input) {
    if (!gpu_)
        gpu_upload_weights();
    const int seq = input.rows;
    if (gpu_->cached_input.rows != seq || gpu_->cached_input.cols != d_model)
        gpu_->cached_input = adai::gpu::GPUMatrix(seq, d_model);
    adai::gpu::matrix_copy_device_to_device_gpu(input.device_ptr(), gpu_->cached_input.device_ptr(),
                                                seq * d_model);

    adai::gpu::GPUMatrix logits = input * gpu_->W_g;
    // In-place add bias row-broadcast
    adai::gpu::GPUMatrix result(seq, vocab_size);
    adai::gpu::matrix_add_row_bias_gpu(logits.device_ptr(), gpu_->b_g.device_ptr(),
                                       result.device_ptr(), seq, vocab_size);
    return result;
}

// TODO: See TD-425 in TECHNICAL_DEBT.md - dereferences gpu_ unchecked (null before any gpu_forward())
adai::gpu::GPUMatrix LanguageModelHead::gpu_backward(const adai::gpu::GPUMatrix& dout) {
    // dW += input^T * dout
    gpu_->dW.add_inplace(gpu_->cached_input.transpose() * dout);
    // db += sum_rows(dout)
    gpu_->db.add_inplace(dout.sum_rows());
    // d_input = dout * W^T
    return dout * gpu_->W_g.transpose();
}
#endif
