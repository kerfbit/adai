// @adai-status: stable
// @adai-version: 1.2.1
// @adai-reviewed: 2026-10-26

#include "TokenEmbedding.hpp"
#include "BinaryIO.hpp"
#include <algorithm>
#include "gpu/GPUParamSync.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

TokenEmbedding::TokenEmbedding(int vocab_size, int d_model)
    : embedding_matrix(vocab_size, d_model),
      embedding_grad(vocab_size, d_model),
      vocab_size(vocab_size),
      d_model(d_model) {
    // TD-517: N(0, 1) so each row has norm ~sqrt(d_model), comparable to a sinusoidal
    // positional row (sqrt(d_model / 2)). The old N(0, 1/d_model) gave norm ~1, so position
    // swamped content (two different tokens at one position: cosine 0.997 at d_model = 768).
    embedding_matrix.randomize(kInitStd);

    // Initialize gradients to zero
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < d_model; ++j) {
            embedding_grad(i, j) = 0.0f;
        }
    }
}

Matrix TokenEmbedding::forward(const std::vector<int>& token_ids) {
    int seq_len = static_cast<int>(token_ids.size());
    Matrix result(seq_len, d_model);

    // Cache token IDs for backward pass
    cached_token_ids = token_ids;

    // Lookup embeddings for each token. Deliberately unscaled (no sqrt(d_model) factor, TD-517):
    // the scale comes from the N(0, 1) initialization instead, so saved tables mean the same
    // thing to every loader and exporter.
    for (int i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];

        // Bounds checking
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::out_of_range("Token ID " + std::to_string(token_id) + " out of range [0, " +
                                    std::to_string(vocab_size) + ")");
        }

        // Copy embedding row
        for (int j = 0; j < d_model; ++j) {
            result(i, j) = embedding_matrix(token_id, j);
        }
    }

    return result;
}

void TokenEmbedding::backward(const std::vector<int>& token_ids, const Matrix& grad_output) {
    // Validate dimensions
    if (token_ids.size() != static_cast<size_t>(grad_output.rows)) {
        throw std::invalid_argument("Token IDs length must match gradient output rows");
    }

    if (grad_output.cols != d_model) {
        throw std::invalid_argument("Gradient output columns must match d_model");
    }

    // Accumulate gradients for each token
    for (size_t i = 0; i < token_ids.size(); ++i) {
        int token_id = token_ids[i];

        // Bounds checking
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::out_of_range("Token ID " + std::to_string(token_id) + " out of range [0, " +
                                    std::to_string(vocab_size) + ")");
        }

        // Accumulate gradient for this token's embedding
        for (int j = 0; j < d_model; ++j) {
            embedding_grad(token_id, j) += grad_output(static_cast<int>(i), j);
        }
    }
}

void TokenEmbedding::set_optimizer(Optimizer* opt) {
    optimizer = opt;
    if (optimizer) {
        register_parameters();
    }
}

void TokenEmbedding::register_parameters() {
    if (!optimizer) {
        return;
    }

    // Register embedding matrix with optimizer
    optimizer->add_parameter_group(&embedding_matrix, &embedding_grad);
}

void TokenEmbedding::update_weights() {
    if (optimizer) {
        // Use advanced optimization (Adam, AdamW, etc.)
        optimizer->step();
    } else {
        // Fallback to simple gradient descent for backward compatibility
        for (int i = 0; i < vocab_size; ++i) {
            for (int j = 0; j < d_model; ++j) {
                embedding_matrix(i, j) -= learning_rate * embedding_grad(i, j);
            }
        }
    }

    // Zero gradients after update
    zero_grad();
}

// TODO: See TD-518 in TECHNICAL_DEBT.md - full vocab x d_model pass every step; only touched rows need zeroing.
void TokenEmbedding::zero_grad() {
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < d_model; ++j) {
            embedding_grad(i, j) = 0.0f;
        }
    }
}

std::vector<float> TokenEmbedding::get_token_embedding(int token_id) const {
    if (token_id < 0 || token_id >= vocab_size) {
        throw std::out_of_range("Token ID " + std::to_string(token_id) + " out of range [0, " +
                                std::to_string(vocab_size) + ")");
    }

    std::vector<float> embedding(d_model);
    for (int j = 0; j < d_model; ++j) {
        embedding[j] = embedding_matrix(token_id, j);
    }

    return embedding;
}

const Matrix& TokenEmbedding::get_embeddings() const {
    return embedding_matrix;
}

int TokenEmbedding::get_vocab_size() const {
    return vocab_size;
}

int TokenEmbedding::get_embedding_dim() const {
    return d_model;
}

void TokenEmbedding::load_pretrained(const std::string& filename) {
    // TD-423/TD-518: checked reads; a truncated file used to load with its later rows left random.
    // The table changes only after the whole file has been read.
    adai::binio::Reader file(filename);
    const int saved_vocab_size = file.value<int>("vocab_size");
    const int saved_d_model = file.value<int>("d_model");

    // Verify dimensions match
    if (saved_vocab_size != vocab_size || saved_d_model != d_model) {
        throw std::runtime_error("Dimension mismatch: file has (" +
                                 std::to_string(saved_vocab_size) + ", " +
                                 std::to_string(saved_d_model) + "), expected (" +
                                 std::to_string(vocab_size) + ", " + std::to_string(d_model) + ")");
    }

    Matrix loaded = file.matrix(vocab_size, d_model, "embedding matrix");
    file.expect_end();
    embedding_matrix = std::move(loaded);

    std::cout << "Loaded pre-trained embeddings from " << filename << '\n';
}

void TokenEmbedding::save_embeddings(const std::string& filename) const {
    adai::binio::Writer file(filename);  // TD-423: checked writes
    file.value(vocab_size, "vocab_size");
    file.value(d_model, "d_model");
    file.matrix(embedding_matrix, "embedding matrix");
    file.close();
}

void TokenEmbedding::initialize_constant(float value) {
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < d_model; ++j) {
            embedding_matrix(i, j) = value;
        }
    }
}

void TokenEmbedding::reinitialize() {
    embedding_matrix.randomize(kInitStd);  // same as the constructor (TD-517)
    zero_grad();
}

void TokenEmbedding::print_config(const std::string& name) const {
    std::cout << name << " Configuration:" << '\n';
    std::cout << "  Vocabulary Size: " << vocab_size << '\n';
    std::cout << "  Embedding Dimension: " << d_model << '\n';
    std::cout << "  Total Parameters: " << (vocab_size * d_model) << '\n';
    std::cout << "  Memory Usage: " << std::fixed << std::setprecision(2)
              << (static_cast<float>(vocab_size) * static_cast<float>(d_model) * sizeof(float)) /
                     1024.0f / 1024.0f
              << " MB" << '\n';
    std::cout << "  Learning Rate: " << learning_rate << '\n';
}

float TokenEmbedding::get_gradient_norm() const {
    float sum_squares = 0.0f;

    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float grad = embedding_grad(i, j);
            sum_squares += grad * grad;
        }
    }

    return std::sqrt(sum_squares);
}

void TokenEmbedding::clip_gradients(float max_norm) {
    float norm = get_gradient_norm();

    if (norm > max_norm) {
        float scale = max_norm / norm;

        for (int i = 0; i < vocab_size; ++i) {
            for (int j = 0; j < d_model; ++j) {
                embedding_grad(i, j) *= scale;
            }
        }
    }
}

void TokenEmbedding::save_weights(const std::string& filename) const {
    // Wrapper for consistency with other components
    save_embeddings(filename);
}

void TokenEmbedding::load_weights(const std::string& filename) {
    // Wrapper for consistency with other components
    load_pretrained(filename);
}

#ifdef ADAI_ENABLE_GPU
// ============================================================================
// TD-547: device table
// ============================================================================

void TokenEmbedding::gpu_upload_weights() {
    if (!gpu_)
        gpu_ = std::make_unique<GPUState>(vocab_size, d_model);
    std::vector<float> flat;
    flat.reserve(static_cast<size_t>(vocab_size) * static_cast<size_t>(d_model));
    for (const auto& row : embedding_matrix.data)
        flat.insert(flat.end(), row.begin(), row.end());
    gpu_->W_g.upload(flat.data(), vocab_size * d_model);
}

void TokenEmbedding::gpu_download_weights() {
    if (gpu_)
        adai::gpu::download_into(gpu_->W_g, embedding_matrix);
}

void TokenEmbedding::gpu_zero_grads() {
    if (gpu_)
        gpu_->dW.zero();
}

void TokenEmbedding::gpu_download_grads() {
    if (!gpu_)
        return;
    std::vector<float> tmp(static_cast<size_t>(vocab_size) * static_cast<size_t>(d_model));
    gpu_->dW.download(tmp.data(), vocab_size * d_model);
    size_t idx = 0;
    for (auto& row : embedding_grad.data)
        for (auto& v : row)
            v += tmp[idx++];
}

void TokenEmbedding::gpu_collect_parameters(std::vector<adai::gpu::GPUParamRef>& out, int layer) {
    if (gpu_)
        adai::gpu::add_param(out, embedding_matrix, gpu_->W_g, gpu_->dW, layer);
}

adai::gpu::GPUMatrix TokenEmbedding::gpu_forward(const std::vector<int>& token_ids,
                                                 const adai::gpu::GPUMatrix* pe, int pe_rows,
                                                 int pos_offset) {
    if (!gpu_)
        gpu_upload_weights();
    const int n = static_cast<int>(token_ids.size());
    for (int id : token_ids) {
        if (id < 0 || id >= vocab_size) {
            throw std::out_of_range("Token ID " + std::to_string(id) + " out of range [0, " +
                                    std::to_string(vocab_size) + ")");
        }
    }
    if (n > gpu_->ids_capacity) {
        gpu_->ids = std::make_unique<adai::gpu::GPUMemory<int>>(static_cast<size_t>(n));
        gpu_->ids_capacity = n;
    }
    adai::gpu::GPUMatrix out(std::max(n, 1), d_model);
    if (n > 0) {
        gpu_->ids->copy_from_host(token_ids.data(), static_cast<size_t>(n));
        adai::gpu::embedding_gather_add_pe_gpu(gpu_->W_g.device_ptr(), gpu_->ids->get(),
                                               pe ? pe->device_ptr() : nullptr, pe_rows,
                                               pos_offset, out.device_ptr(), n, d_model);
    }
    gpu_->last_count = n;
    return out;
}

void TokenEmbedding::gpu_backward(const adai::gpu::GPUMatrix& dout) {
    if (!gpu_ || gpu_->last_count == 0)
        return;
    if (dout.rows != gpu_->last_count || dout.cols != d_model) {
        throw std::invalid_argument("TokenEmbedding::gpu_backward: gradient shape doesn't match "
                                    "the last gpu_forward()");
    }
    adai::gpu::embedding_scatter_add_gpu(dout.device_ptr(), gpu_->ids->get(),
                                         gpu_->dW.device_ptr(), gpu_->last_count, d_model);
}
#endif
