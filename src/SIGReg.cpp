// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-15

#include "SIGReg.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace {

// See SIGReg.hpp's "Design note": a fixed, small set of frequencies standing in for the
// continuous integral the LeJEPA paper's own SIGReg formulation uses. Chosen to span both small
// t (sensitive to low-order moments — mean/variance) and larger t (sensitive to the tails), so a
// distribution that only matches a standard normal's first couple of moments but not its overall
// shape still gets penalized. Shared across every SIGReg instance as fixed, non-learnable
// configuration — no reason for it to be per-instance state.
const std::vector<float> kFrequencies = {0.5f, 1.0f, 1.5f, 2.0f};

// Validates and passes through in one expression, used directly inside the member-initializer
// list below. A check in the constructor *body* runs too late here: `directions_(d_model,
// num_sketches)` is itself a member initializer, and member initializers run in declaration
// order before the body — a negative value would already have reached Matrix's own constructor
// (and std::vector's internal resize, throwing its own, far less useful std::length_error) before
// the body ever got a chance to reject it.
int require_positive(int value, const char* name) {
    if (value <= 0) {
        throw std::invalid_argument(std::string("SIGReg: ") + name + " must be positive");
    }
    return value;
}

}  // namespace

SIGReg::SIGReg(int d_model, int num_sketches)
    : d_model_(require_positive(d_model, "d_model")),
      num_sketches_(require_positive(num_sketches, "num_sketches")),
      directions_(d_model_, num_sketches_) {
    // Random directions, fixed for this instance's whole lifetime (see the class doc comment).
    // Sampling each component i.i.d. standard normal then normalizing to unit length gives a
    // direction uniformly distributed on the unit sphere in R^d_model.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int k = 0; k < num_sketches; ++k) {
        float norm_sq = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            float v = dist(gen);
            directions_.data[i][k] = v;
            norm_sq += v * v;
        }
        // A component-wise-zero draw is astronomically unlikely for d_model > 1 in practice, but
        // guard the divide regardless rather than ever emitting NaN directions.
        float norm = std::sqrt(norm_sq);
        if (norm < 1e-8f) {
            norm = 1e-8f;
        }
        for (int i = 0; i < d_model; ++i) {
            directions_.data[i][k] /= norm;
        }
    }
}

float SIGReg::compute_loss(const Matrix& embeddings) const {
    if (embeddings.cols != d_model_) {
        throw std::invalid_argument("SIGReg::compute_loss: embeddings.cols must equal d_model");
    }
    const int batch = embeddings.rows;
    if (batch == 0) {
        return 0.0f;
    }

    // [batch, num_sketches] — projections[i][k] = embeddings[i] . directions[:,k]
    Matrix projections = embeddings * directions_;

    float total_loss = 0.0f;
    for (int k = 0; k < num_sketches_; ++k) {
        for (float t : kFrequencies) {
            float mean_cos = 0.0f;
            float mean_sin = 0.0f;
            for (int i = 0; i < batch; ++i) {
                const float p = projections.data[i][k];
                mean_cos += std::cos(t * p);
                mean_sin += std::sin(t * p);
            }
            mean_cos /= static_cast<float>(batch);
            mean_sin /= static_cast<float>(batch);

            // Characteristic function of a standard normal: phi(t) = exp(-t^2/2), purely real
            // (imaginary part is exactly 0 by symmetry of N(0,1)) — so the target for mean_sin
            // is 0, not a separately-computed value.
            const float target_real = std::exp(-0.5f * t * t);
            const float d_real = mean_cos - target_real;
            const float d_imag = mean_sin;

            total_loss += d_real * d_real + d_imag * d_imag;
        }
    }

    return total_loss / static_cast<float>(num_sketches_ * static_cast<int>(kFrequencies.size()));
}

SIGReg::VarianceStats SIGReg::compute_variance_stats(const Matrix& embeddings) const {
    if (embeddings.cols != d_model_) {
        throw std::invalid_argument(
            "SIGReg::compute_variance_stats: embeddings.cols must equal d_model");
    }
    const int batch = embeddings.rows;
    if (batch == 0) {
        return {0.0f, 0.0f};
    }

    // Same projection compute_loss() forms internally — recomputed rather than cached, matching
    // this class's existing no-caching style (compute_loss()/backward() each independently
    // recompute it too).
    Matrix projections = embeddings * directions_;

    // Per-direction population variance (mean-of-squares minus square-of-mean — batch here is
    // typically small, a handful of rows per LeJEPAEncoder::train_step() call, so this is an
    // inherently noisy per-step estimate; averaging over many steps in a training epoch is doing
    // real statistical work, not just cosmetic smoothing).
    std::vector<float> per_direction_variance(num_sketches_, 0.0f);
    for (int k = 0; k < num_sketches_; ++k) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int i = 0; i < batch; ++i) {
            const float p = projections.data[i][k];
            sum += p;
            sum_sq += p * p;
        }
        const float mean = sum / static_cast<float>(batch);
        per_direction_variance[k] = (sum_sq / static_cast<float>(batch)) - (mean * mean);
    }

    float variance_mean = 0.0f;
    for (float v : per_direction_variance) {
        variance_mean += v;
    }
    variance_mean /= static_cast<float>(num_sketches_);

    float variance_sq_diff_sum = 0.0f;
    for (float v : per_direction_variance) {
        const float diff = v - variance_mean;
        variance_sq_diff_sum += diff * diff;
    }
    const float variance_stddev = std::sqrt(variance_sq_diff_sum / static_cast<float>(num_sketches_));

    return {variance_mean, variance_stddev};
}

Matrix SIGReg::backward(const Matrix& embeddings) const {
    if (embeddings.cols != d_model_) {
        throw std::invalid_argument("SIGReg::backward: embeddings.cols must equal d_model");
    }
    const int batch = embeddings.rows;
    if (batch == 0) {
        return Matrix(0, d_model_);
    }

    Matrix projections = embeddings * directions_;  // [batch, num_sketches]
    const float norm_factor =
        1.0f / static_cast<float>(num_sketches_ * static_cast<int>(kFrequencies.size()));

    // d(loss)/d(projections[i][k]) — accumulated across every frequency's own contribution to
    // the same (i, k) projection, then chain-ruled through the projection itself below.
    Matrix grad_projections(batch, num_sketches_);

    for (int k = 0; k < num_sketches_; ++k) {
        for (float t : kFrequencies) {
            float mean_cos = 0.0f;
            float mean_sin = 0.0f;
            for (int i = 0; i < batch; ++i) {
                const float p = projections.data[i][k];
                mean_cos += std::cos(t * p);
                mean_sin += std::sin(t * p);
            }
            mean_cos /= static_cast<float>(batch);
            mean_sin /= static_cast<float>(batch);

            const float target_real = std::exp(-0.5f * t * t);
            const float d_real = mean_cos - target_real;
            const float d_imag = mean_sin;

            // loss_kt = d_real^2 + d_imag^2
            // d(mean_cos)/d(p_i) = -t*sin(t*p_i) / batch ; d(mean_sin)/d(p_i) = t*cos(t*p_i) / batch
            for (int i = 0; i < batch; ++i) {
                const float p = projections.data[i][k];
                const float dcos_dp = -t * std::sin(t * p) / static_cast<float>(batch);
                const float dsin_dp = t * std::cos(t * p) / static_cast<float>(batch);
                grad_projections.data[i][k] +=
                    (2.0f * d_real * dcos_dp + 2.0f * d_imag * dsin_dp) * norm_factor;
            }
        }
    }

    // Chain rule through p_ik = embeddings[i] . directions[:,k]  =>
    //   d(p_ik)/d(embeddings[i][j]) = directions[j][k]
    // grad_embeddings = grad_projections @ directions_^T  — [batch, num_sketches] @
    // [num_sketches, d_model] = [batch, d_model].
    Matrix directions_T = directions_.transpose();
    return grad_projections * directions_T;
}
