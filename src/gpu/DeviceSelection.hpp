#pragma once

// @adai-status: beta        (device tests pass on the oneAPI CPU device and ai-machine's Arc)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace adai {
namespace gpu {

/// TD-391: what device selection needs to know about one SYCL device. No SYCL types, so the rule
/// is tested without a SYCL build.
struct DeviceCandidate {
    std::string backend;  ///< "level_zero", "opencl" or anything else
    std::string vendor;
    std::string name;
};

/// Level Zero first (the expected, faster path for oneMKL), OpenCL last.
inline int device_backend_rank(const std::string& backend) {
    if (backend == "level_zero") {
        return 0;
    }
    return backend == "opencl" ? 2 : 1;
}

/**
 * TD-391: which of @p candidates GPU_DEVICE_ID numbers, as indices into it. One card is listed once
 * per SYCL runtime, so best backend comes first, and a device is dropped when a better backend
 * lists a device of the same vendor and name that no other listing has been matched to (two
 * identical cards stay two). SYCL has no device identity that works across backends. Where the
 * names differ between runtimes a card stays listed twice, but id 0 is still the better backend.
 */
inline std::vector<std::size_t> select_devices(const std::vector<DeviceCandidate>& candidates) {
    std::vector<std::size_t> order(candidates.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return device_backend_rank(candidates[a].backend) < device_backend_rank(candidates[b].backend);
    });

    std::vector<std::size_t> kept;
    std::vector<bool> matched;  // per kept device: already stands for a listing on a worse backend
    for (std::size_t index : order) {
        const DeviceCandidate& device = candidates[index];
        bool duplicate = false;
        for (std::size_t k = 0; k < kept.size() && !duplicate; ++k) {
            const DeviceCandidate& other = candidates[kept[k]];
            if (!matched[k] && other.backend != device.backend && other.vendor == device.vendor &&
                other.name == device.name) {
                matched[k] = true;
                duplicate = true;
            }
        }
        if (!duplicate) {
            kept.push_back(index);
            matched.push_back(false);
        }
    }
    return kept;
}

}  // namespace gpu
}  // namespace adai
