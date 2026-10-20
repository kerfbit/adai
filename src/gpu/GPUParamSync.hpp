#ifndef GPU_PARAM_SYNC_HPP
#define GPU_PARAM_SYNC_HPP

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-19

#ifdef ADAI_ENABLE_GPU

#include <vector>
#include "../Matrix.hpp"
#include "GPUParamRef.hpp"
#include "MatrixGPU.hpp"

namespace adai {
namespace gpu {

// TD-468: copy a row-major device mirror back into its host Matrix (same shape). Used by the
// layers' gpu_download_weights(), which run only when the host copy is needed (checkpoints, the
// per-epoch quality-metrics clone, the end of a pass), never per step.
inline void download_into(const GPUMatrix& device, Matrix& host) {
    const int n = host.rows * host.cols;
    std::vector<float> tmp(static_cast<size_t>(n));
    device.download(tmp.data(), n);
    int idx = 0;
    for (auto& row : host.data)
        for (auto& v : row)
            v = tmp[static_cast<size_t>(idx++)];
}

inline void add_param(std::vector<GPUParamRef>& out, Matrix& host, GPUMatrix& weights,
                      GPUMatrix& grads, int layer) {
    out.push_back({&host, &weights, &grads, layer});
}

}  // namespace gpu
}  // namespace adai

#endif  // ADAI_ENABLE_GPU
#endif  // GPU_PARAM_SYNC_HPP
