// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-08

#include "Dropout.hpp"
#include <stdexcept>
#include <string>
#include "DropoutMask.hpp"

Dropout::Dropout() : rng_(std::random_device{}()) {}

void Dropout::set_rate(float rate) {
    if (!(rate >= 0.0f && rate < 1.0f)) {
        throw std::invalid_argument("dropout rate must be in [0, 1), got " + std::to_string(rate));
    }
    rate_ = rate;
}

void Dropout::forward(Matrix& x) {
    applied_ = dropping();
    if (!applied_) {
        return;
    }
    const float keep = 1.0f - rate_;
    const std::uint64_t seed = rng_();
    mask_ = Matrix(x.rows, x.cols);
    std::uint64_t index = 0;
    for (int i = 0; i < x.rows; ++i) {
        for (int j = 0; j < x.cols; ++j, ++index) {
            const float m = adai::dropout_mask_value(seed, index, keep);
            mask_.data[i][j] = m;
            x.data[i][j] *= m;
        }
    }
}

void Dropout::backward(Matrix& grad) const {
    if (!applied_) {
        return;
    }
    if (grad.rows != mask_.rows || grad.cols != mask_.cols) {
        throw std::invalid_argument("Dropout::backward: gradient shape differs from the forward's");
    }
    for (int i = 0; i < grad.rows; ++i) {
        for (int j = 0; j < grad.cols; ++j) {
            grad.data[i][j] *= mask_.data[i][j];
        }
    }
}

#ifdef ADAI_ENABLE_GPU
void Dropout::gpu_forward(adai::gpu::GPUMatrix& x) {
    gpu_applied_ = dropping();
    if (!gpu_applied_) {
        return;
    }
    if (!gpu_mask_ || gpu_mask_->rows != x.rows || gpu_mask_->cols != x.cols) {
        gpu_mask_ = std::make_unique<adai::gpu::GPUMatrix>(x.rows, x.cols);
    }
    // Only the seed crosses to the device; the mask is generated and kept there.
    adai::gpu::dropout_forward_gpu(x.device_ptr(), gpu_mask_->device_ptr(), x.rows * x.cols,
                                   1.0f - rate_, rng_());
}

adai::gpu::GPUMatrix Dropout::gpu_masked(const adai::gpu::GPUMatrix& grad) const {
    if (!gpu_applied_) {
        throw std::logic_error("Dropout::gpu_masked: the last gpu_forward() didn't drop");
    }
    return grad.hadamard(*gpu_mask_);  // throws on a shape mismatch
}
#endif
