// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-10-20

#include "Predictor.hpp"
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

// Same validate-and-pass-through helper TD-175's SIGReg introduced: called directly inside the
// member-initializer list so a non-positive dimension is rejected with a clear
// std::invalid_argument *before* it reaches FeedForward's own constructor (which has no such
// guard itself and would instead fail deep inside Matrix's allocation).
int require_positive(int value, const char* name) {
    if (value <= 0) {
        throw std::invalid_argument(std::string("Predictor: ") + name + " must be positive");
    }
    return value;
}

}  // namespace

Predictor::Predictor(int d_model, int hidden_dim)
    : net_(std::make_unique<FeedForward>(require_positive(d_model, "d_model"),
                                          require_positive(hidden_dim, "hidden_dim"))),
      d_model_(d_model),
      hidden_dim_(hidden_dim) {}

Matrix Predictor::forward(const Matrix& context_embedding) {
    if (context_embedding.cols != d_model_) {
        throw std::invalid_argument("Predictor::forward: context_embedding.cols must equal d_model");
    }
    return net_->forward(context_embedding);
}

Matrix Predictor::backward(const Matrix& grad_output) {
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("Predictor::backward: grad_output.cols must equal d_model");
    }
    return net_->backward(grad_output);
}

void Predictor::update_weights() {
    net_->update_weights();
}

void Predictor::zero_grad() {
    net_->zero_grad();
}

void Predictor::register_parameters_with_optimizer(Optimizer& optimizer) {
    net_->set_optimizer(&optimizer);
}

void Predictor::save(const std::string& filename) const {
    net_->save_weights(filename);
}

bool Predictor::load(const std::string& filename) {
    std::error_code ec;
    if (!std::filesystem::exists(filename, ec)) {
        return false;
    }
    // FeedForward::save_weights(): two int dims, then W1, W2, b1, b2.
    const auto expected = static_cast<std::uintmax_t>(
        2 * sizeof(int) +
        sizeof(float) * (2 * static_cast<std::size_t>(d_model_) * hidden_dim_ + hidden_dim_ + d_model_));
    const std::uintmax_t actual = std::filesystem::file_size(filename, ec);
    if (ec || actual != expected) {
        throw std::runtime_error("Predictor::load(): " + filename + " is " + std::to_string(actual) +
                                 " bytes, expected " + std::to_string(expected));
    }
    net_->load_weights(filename);  // throws on a d_model/hidden_dim mismatch
    return true;
}

