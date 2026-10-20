#ifndef GPU_OPTIMIZER_KERNELS_HPP
#define GPU_OPTIMIZER_KERNELS_HPP

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-19

// TD-468: parameters of the device-resident optimizer update, shared by the CUDA (MatrixGPU.cu)
// and SYCL (sycl/MatrixGPU_SYCL.cpp) kernels so both backends apply exactly the update rules of
// the CPU Optimizer (Optimizer.cpp's step_sgd/step_sgd_momentum/step_adam/step_adamw).

namespace adai {
namespace gpu {

/// Mirrors OptimizerType's values (Optimizer.hpp), kept separate so the kernels don't include it.
enum class GPUOptimizerRule : int { SGD = 0, SGD_MOMENTUM = 1, ADAM = 2, ADAMW = 3 };

struct GPUOptimizerUpdate {
    GPUOptimizerRule rule = GPUOptimizerRule::ADAMW;
    float learning_rate = 0.0f;
    float momentum_beta = 0.9f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.0f;
    /// 1 - beta1^t and 1 - beta2^t for this group's step t (Adam/AdamW only).
    float bias_correction1 = 1.0f;
    float bias_correction2 = 1.0f;
    /// Applied to every gradient before the update: the TD-272 short-window rescale times the
    /// clip factor, so neither needs its own pass over the gradients.
    float grad_scale = 1.0f;
};

}  // namespace gpu
}  // namespace adai

#endif  // GPU_OPTIMIZER_KERNELS_HPP
