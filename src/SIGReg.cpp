// @adai-status: beta        (capped by TD-504 — see TECHNICAL_DEBT.md)
// @adai-version: 0.3.0
// @adai-reviewed: 2026-11-22

#include "InitRng.hpp"
#include "SIGReg.hpp"
#include <algorithm>
#include <cmath>
#include "SIGRegMath.hpp"
#include <stdexcept>
#include <string>

namespace {

// See SIGReg.hpp's "Design note": a fixed, small set of frequencies standing in for the
// continuous integral the LeJEPA paper's own SIGReg formulation uses. Chosen to span both small
// t (sensitive to low-order moments — mean/variance) and larger t (sensitive to the tails), so a
// distribution that only matches a standard normal's first couple of moments but not its overall
// shape still gets penalized. Shared across every SIGReg instance as fixed, non-learnable
// configuration — no reason for it to be per-instance state.
// TD-586: the grid itself is adai::sigreg_frequency() in SIGRegMath.hpp, shared with the GPU kernels.
// TODO: See TD-504 in TECHNICAL_DEBT.md - unweighted 4-point grid instead of a Gaussian-weighted integral.
constexpr int kNumFrequencies = adai::kSigregNumFrequencies;

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
      seed_(adai::next_init_seed()),
      directions_(d_model_, num_sketches_) {
    resample_directions();
}

void SIGReg::set_seed(std::uint32_t seed) {
    seed_ = seed;
    draw_ = 0;
    resample_directions();
}

// TD-502: a fresh draw per training step, so over training the loss constrains every dimension
// rather than one fixed num_sketches-dimensional span. Each component i.i.d. standard normal,
// then normalized: a direction uniform on the unit sphere in R^d_model.
void SIGReg::resample_directions() {
    ++draw_;  // TD-586: the directions for this draw are built where they're needed
}

// Each component a standard normal (a pure function of seed, draw and position), then each column
// normalized: a direction uniform on the unit sphere in R^d_model. The GPU kernel computes the same.
void SIGReg::ensure_host_directions() const {
    if (host_draw_ == draw_) {
        return;
    }
    for (int k = 0; k < num_sketches_; ++k) {
        float norm_sq = 0.0f;
        for (int i = 0; i < d_model_; ++i) {
            const float v = adai::sigreg_direction_raw(
                seed_, draw_, static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(num_sketches_) + k);
            directions_.data[i][k] = v;
            norm_sq += v * v;
        }
        // Guard the divide rather than ever emitting NaN directions (an all-zero column can't happen
        // in practice).
        const float norm = std::max(std::sqrt(norm_sq), 1e-8f);
        for (int i = 0; i < d_model_; ++i) {
            directions_.data[i][k] /= norm;
        }
    }
    host_draw_ = draw_;
}

float SIGReg::compute_loss(const Matrix& embeddings) const {
    if (embeddings.cols != d_model_) {
        throw std::invalid_argument("SIGReg::compute_loss: embeddings.cols must equal d_model");
    }
    const int batch = embeddings.rows;
    if (batch < 2) {
        return 0.0f;  // the unbiased estimator needs two samples (TD-503)
    }
    const float n = static_cast<float>(batch);

    // [batch, num_sketches] — projections[i][k] = embeddings[i] . directions[:,k]
    ensure_host_directions();
    Matrix projections = embeddings * directions_;

    float total_loss = 0.0f;
    for (int k = 0; k < num_sketches_; ++k) {
        for (int fj = 0; fj < kNumFrequencies; ++fj) {
            const float t = adai::sigreg_frequency(fj);
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

            // TD-503: |phi_hat - phi|^2 plus the bias correction (|phi_hat|^2 - 1)/(n - 1), which
            // replaces |phi_hat|^2 with its unbiased U-statistic (n|phi_hat|^2 - 1)/(n - 1). The
            // plain gap had an expected floor of (1 - |phi_P|^2)/n.
            const float abs_sq = mean_cos * mean_cos + mean_sin * mean_sin;
            total_loss += d_real * d_real + d_imag * d_imag + (abs_sq - 1.0f) / (n - 1.0f);
        }
    }

    return total_loss / static_cast<float>(num_sketches_ * kNumFrequencies);
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
    ensure_host_directions();
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
    if (batch < 2) {
        return Matrix(batch, d_model_);  // compute_loss() is 0 here, so is its gradient (TD-503)
    }
    const float n = static_cast<float>(batch);
    const float unbiased_scale = n / (n - 1.0f);

    // TODO: See TD-504 in TECHNICAL_DEBT.md - third recomputation of X*D per step; no non-finite input check.
    ensure_host_directions();
    Matrix projections = embeddings * directions_;  // [batch, num_sketches]
    const float norm_factor =
        1.0f / static_cast<float>(num_sketches_ * kNumFrequencies);

    // d(loss)/d(projections[i][k]) — accumulated across every frequency's own contribution to
    // the same (i, k) projection, then chain-ruled through the projection itself below.
    Matrix grad_projections(batch, num_sketches_);

    for (int k = 0; k < num_sketches_; ++k) {
        for (int fj = 0; fj < kNumFrequencies; ++fj) {
            const float t = adai::sigreg_frequency(fj);
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

            // TD-503: loss_kt = n/(n-1) * (mean_cos^2 + mean_sin^2) - 1/(n-1)
            //                   - 2 * target * mean_cos + target^2
            // d(mean_cos)/d(p_i) = -t*sin(t*p_i) / batch ; d(mean_sin)/d(p_i) = t*cos(t*p_i) / batch
            for (int i = 0; i < batch; ++i) {
                const float p = projections.data[i][k];
                const float dcos_dp = -t * std::sin(t * p) / static_cast<float>(batch);
                const float dsin_dp = t * std::cos(t * p) / static_cast<float>(batch);
                grad_projections.data[i][k] +=
                    (2.0f * unbiased_scale * (mean_cos * dcos_dp + mean_sin * dsin_dp) -
                     2.0f * target_real * dcos_dp) *
                    norm_factor;
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

#ifdef ADAI_ENABLE_GPU
// TD-586: compute_loss(), compute_variance_stats() and backward() on the device, from one projection.
void SIGReg::gpu_compute(const adai::gpu::GPUMatrix& x, float* out_device, adai::gpu::GPUMatrix* grad) {
    if (x.cols != d_model_) {
        throw std::invalid_argument("SIGReg::gpu_compute: x.cols must equal d_model");
    }
    if (x.rows <= 0) {
        throw std::invalid_argument("SIGReg::gpu_compute: x has no rows");
    }
    if (!gpu_directions_ || gpu_draw_ != draw_) {
        if (!gpu_directions_) {
            gpu_directions_ = std::make_unique<adai::gpu::GPUMatrix>(d_model_, num_sketches_);
        }
        adai::gpu::sigreg_directions_gpu(gpu_directions_->device_ptr(), d_model_, num_sketches_, seed_, draw_);
        gpu_draw_ = draw_;
    }
    const int batch = x.rows;
    const adai::gpu::GPUMatrix projections = x * *gpu_directions_;  // [batch, num_sketches]
    adai::gpu::GPUMatrix loss_terms(1, num_sketches_);
    adai::gpu::GPUMatrix var_terms(1, num_sketches_);
    std::unique_ptr<adai::gpu::GPUMatrix> grad_projections;
    if (grad) {
        grad_projections = std::make_unique<adai::gpu::GPUMatrix>(batch, num_sketches_);
    }
    const float norm_factor = 1.0f / static_cast<float>(num_sketches_ * kNumFrequencies);
    adai::gpu::sigreg_terms_gpu(projections.device_ptr(), batch, num_sketches_, norm_factor, loss_terms.device_ptr(),
                                var_terms.device_ptr(), grad ? grad_projections->device_ptr() : nullptr);
    adai::gpu::sigreg_finish_gpu(loss_terms.device_ptr(), var_terms.device_ptr(), num_sketches_, norm_factor,
                                 out_device);
    if (grad) {
        // grad_x = grad_projections · directionsᵀ: [batch, num_sketches] x [num_sketches, d_model].
        adai::gpu::GPUMatrix g(batch, d_model_);
        adai::gpu::matrix_gemm_strided_gpu(false, true, batch, d_model_, num_sketches_, 1.0f,
                                           grad_projections->device_ptr(), num_sketches_,
                                           gpu_directions_->device_ptr(), num_sketches_, g.device_ptr(), d_model_);
        *grad = std::move(g);
    }
}
#endif
