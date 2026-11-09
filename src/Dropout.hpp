#pragma once

// @adai-status: experimental
// @adai-version: 0.1.1
// @adai-reviewed: 2026-11-09

#include <cstdint>
#include <memory>
#include <random>
#include "Matrix.hpp"
#ifdef ADAI_ENABLE_GPU
#include "gpu/MatrixGPU.hpp"
#endif

/**
 * TD-344: inverted dropout on one residual branch of a transformer block (the attention or
 * feed-forward output, before it is added to the residual stream).
 *
 * Off unless set_active(true) and rate > 0: EncoderDecoderModel turns it on only while
 * ChatbotTrainer trains (set_dropout() + set_training()), so validation, generation, serving and
 * the frozen world model never drop anything. forward() scales the kept elements by
 * 1 / (1 - rate) and remembers the mask; backward() applies the same mask to the gradient, and is
 * a no-op when the matching forward() didn't drop. Masks come from adai::dropout_mask_value(), so
 * a seed gives the same mask on the CPU and on either GPU backend; on the GPU the mask is generated
 * and kept on the device.
 */
class Dropout {
   public:
    Dropout();

    void set_rate(float rate);  ///< [0, 1); throws std::invalid_argument otherwise
    float rate() const {
        return rate_;
    }
    void set_active(bool active) {
        active_ = active;
    }
    bool dropping() const {
        return active_ && rate_ > 0.0f;
    }
    /// Seeds the per-call seed generator (tests; production draws from adai::next_init_seed()).
    void seed(std::uint64_t seed) {
        rng_.seed(seed);
    }

    void forward(Matrix& x);
    void backward(Matrix& grad) const;

#ifdef ADAI_ENABLE_GPU
    void gpu_forward(adai::gpu::GPUMatrix& x);
    /// Whether the last gpu_forward() dropped; only then does the branch gradient need
    /// gpu_masked().
    bool gpu_dropped() const {
        return gpu_applied_;
    }
    /// @p grad times the last gpu_forward()'s mask (GPUMatrix is move-only, so a new matrix).
    adai::gpu::GPUMatrix gpu_masked(const adai::gpu::GPUMatrix& grad) const;
#endif

   private:
    float rate_ = 0.0f;
    bool active_ = false;
    bool applied_ = false;  // the last forward() dropped, so backward() must mask
    std::mt19937_64 rng_;
    Matrix mask_;
#ifdef ADAI_ENABLE_GPU
    bool gpu_applied_ = false;
    std::unique_ptr<adai::gpu::GPUMatrix> gpu_mask_;
#endif
};
