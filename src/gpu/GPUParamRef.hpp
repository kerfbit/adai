#ifndef GPU_PARAM_REF_HPP
#define GPU_PARAM_REF_HPP

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-19

// TD-468: one trainable parameter's device buffers, reported by each layer's
// gpu_collect_parameters() so the device-resident optimizer (GPUOptimizer) can update the GPU
// mirrors directly instead of round-tripping every gradient and weight through the host.

class Matrix;

namespace adai {
namespace gpu {

class GPUMatrix;

struct GPUParamRef {
    /// The host weight matrix, as registered with the CPU Optimizer (its identity key).
    Matrix* host = nullptr;
    /// Row-major device mirror of `host` and its device gradient accumulator.
    GPUMatrix* weights = nullptr;
    GPUMatrix* grads = nullptr;
    /// Block the parameter belongs to, for per-layer gradient norms: encoder block l -> l,
    /// decoder block l -> kDecoderLayerBase + l; -1 for parameters outside a block.
    int layer = -1;
};

inline constexpr int kDecoderLayerBase = 1000;

}  // namespace gpu
}  // namespace adai

#endif  // GPU_PARAM_REF_HPP
