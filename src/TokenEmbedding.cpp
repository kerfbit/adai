// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-10-17

#include "TokenEmbedding.hpp"
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
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read dimensions
    int saved_vocab_size = 0, saved_d_model = 0;
    file.read(reinterpret_cast<char*>(&saved_vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_d_model), sizeof(int));

    // Verify dimensions match
    if (saved_vocab_size != vocab_size || saved_d_model != d_model) {
        throw std::runtime_error("Dimension mismatch: file has (" +
                                 std::to_string(saved_vocab_size) + ", " +
                                 std::to_string(saved_d_model) + "), expected (" +
                                 std::to_string(vocab_size) + ", " + std::to_string(d_model) + ")");
    }

    // TODO: See TD-518 in TECHNICAL_DEBT.md - reads unchecked: a truncated file loads with later rows left random.
    // Read embedding matrix
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.read(reinterpret_cast<char*>(&embedding_matrix(i, j)), sizeof(float));
        }
    }

    file.close();

    std::cout << "Loaded pre-trained embeddings from " << filename << '\n';
}

void TokenEmbedding::save_embeddings(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write dimensions
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));

    // Write embedding matrix
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < d_model; ++j) {
            file.write(reinterpret_cast<const char*>(&embedding_matrix(i, j)), sizeof(float));
        }
    }

    file.close();

    std::cout << "Saved embeddings to " << filename << '\n';
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
