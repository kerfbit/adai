#ifndef GPU_OPTIMIZER_KERNELS_HPP
#define GPU_OPTIMIZER_KERNELS_HPP

// @adai-status: beta        (device tests pass on the oneAPI CPU device and ai-machine's Arc)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-11-13

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

/// One parameter's update: returns the new weight and updates m/v in place. Shared by the SYCL
/// kernel and its host version (TD-574), so both apply the same formulas in the same order as
/// Optimizer::step_sgd/step_sgd_momentum/step_adam/step_adamw. @p sqrt_fn is the backend's sqrt.
template <typename Sqrt>
inline float optimizer_update_element(const GPUOptimizerUpdate& u, float w, float grad, float& m,
                                      float& v, Sqrt sqrt_fn) {
    float g = grad * u.grad_scale;
    switch (u.rule) {
        case GPUOptimizerRule::SGD:
            if (u.weight_decay > 0.0f)
                g += u.weight_decay * w;
            w -= u.learning_rate * g;
            break;
        case GPUOptimizerRule::SGD_MOMENTUM: {
            if (u.weight_decay > 0.0f)
                g += u.weight_decay * w;
            const float mi = u.momentum_beta * m + g;
            m = mi;
            w -= u.learning_rate * mi;
            break;
        }
        case GPUOptimizerRule::ADAM: {
            if (u.weight_decay > 0.0f)
                g += u.weight_decay * w;
            const float mi = u.beta1 * m + (1.0f - u.beta1) * g;
            const float vi = u.beta2 * v + (1.0f - u.beta2) * g * g;
            m = mi;
            v = vi;
            const float m_hat = mi / u.bias_correction1;
            const float v_hat = vi / u.bias_correction2;
            w -= u.learning_rate * m_hat / (sqrt_fn(v_hat) + u.epsilon);
            break;
        }
        case GPUOptimizerRule::ADAMW: {
            const float mi = u.beta1 * m + (1.0f - u.beta1) * g;
            const float vi = u.beta2 * v + (1.0f - u.beta2) * g * g;
            m = mi;
            v = vi;
            const float m_hat = mi / u.bias_correction1;
            const float v_hat = vi / u.bias_correction2;
            w -= u.learning_rate * (m_hat / (sqrt_fn(v_hat) + u.epsilon) + u.weight_decay * w);
            break;
        }
    }
    return w;
}

}  // namespace gpu
}  // namespace adai

#endif  // GPU_OPTIMIZER_KERNELS_HPP
