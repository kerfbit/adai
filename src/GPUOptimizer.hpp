#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-19

#ifdef ADAI_ENABLE_GPU

#include <cstddef>
#include <memory>
#include <vector>
#include "Optimizer.hpp"
#include "gpu/GPUParamRef.hpp"
#include "gpu/MatrixGPU.hpp"

/**
 * @brief TD-468: the optimizer step for GPU training, run on the device.
 *
 * GPU training used to download every gradient to the host, run `Optimizer::step()` on the CPU
 * and upload every weight back, every optimizer step. This class binds the CPU `Optimizer`'s
 * parameter groups to the layers' device mirrors (`EncoderDecoderModel::gpu_collect_parameters()`)
 * and keeps the first/second moments on the device, so a step moves no tensors:
 *
 *  - `gradient_sq_norms()` reduces every group's gradient on the device and downloads only one
 *    float per group (the clip, NaN and zero-gradient decisions need them);
 *  - `step(grad_scale)` applies the same update rules as the CPU `Optimizer`, with the TD-272
 *    window rescale and the clip factor folded into `grad_scale`.
 *
 * Groups with no device mirror (the token embeddings, and the gated world-model/hippocampal
 * sub-layers, TD-342) stay host groups: their gradients are already on the host, so they are
 * reduced and stepped there, with no transfer either way.
 *
 * The CPU `Optimizer` stays the owner of the hyperparameters, the step counts and (between
 * passes) the moments: `download_state()` copies the device moments into it before its state is
 * saved, and the constructor uploads them, so a restored optimizer continues on the device.
 * The device weights are the live ones while this exists; the model's
 * `gpu_download_weights()` brings them back to the host when needed.
 */
class GPUOptimizer {
   public:
    GPUOptimizer(Optimizer& host, const std::vector<adai::gpu::GPUParamRef>& params);

    std::size_t device_group_count() const { return device_.size(); }
    std::size_t host_group_count() const { return host_groups_.size(); }

    /// Squared L2 norm of every group's gradient, indexed like the CPU optimizer's groups.
    const std::vector<float>& gradient_sq_norms();

    /// One optimizer step with every gradient multiplied by @p grad_scale. Groups whose squared
    /// gradient norm (from the last gradient_sq_norms() call) is 0 are skipped when weight decay
    /// is 0, as `Optimizer::step()` skips all-zero gradients.
    void step(float grad_scale);

    /// L2 norm of every weight (device and host groups), for the weight-update ratio (TD-013).
    float weight_norm();

    /// sqrt of the summed squared gradient norms of the groups tagged @p layer, from the last
    /// gradient_sq_norms() call (per-layer gradient diagnostic).
    float layer_gradient_norm(int layer) const;

    /// Copy the device moments into the CPU optimizer's groups (before saving training state).
    void download_state();

   private:
    struct DeviceGroup {
        std::size_t index;  // into the CPU optimizer's parameter_groups
        adai::gpu::GPUMatrix* weights;
        adai::gpu::GPUMatrix* grads;
        std::unique_ptr<adai::gpu::GPUMatrix> m;  // null when the rule needs no first moment
        std::unique_ptr<adai::gpu::GPUMatrix> v;  // null when the rule needs no second moment
        int size;
        int layer;
    };

    void upload_state();
    adai::gpu::GPUOptimizerUpdate update_params(const ParameterGroup& group,
                                                float grad_scale) const;

    Optimizer& host_;
    std::vector<DeviceGroup> device_;
    std::vector<std::size_t> host_groups_;  // CPU optimizer group indices stepped on the host
    std::vector<int> layer_of_group_;       // per CPU optimizer group (-1 = none)
    std::unique_ptr<adai::gpu::GPUMatrix> sq_slots_;  // one float per device group
    std::vector<float> sq_norms_;                     // per CPU optimizer group
};

#endif  // ADAI_ENABLE_GPU
