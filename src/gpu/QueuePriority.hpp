#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-17

#include <cstdint>

namespace adai {
namespace gpu {

/**
 * @brief Scheduling priority of ADAI's GPU queue (SYCL) or stream (CUDA), chosen once at
 *        GPUManager::initialize() from GPU_STRATEGY. No GPU headers, so Config.hpp and CPU-only
 *        builds can name it.
 *
 * LOW    background: priority_low (SYCL) / the lowest stream priority (CUDA); yields to other
 *        GPU work.
 * NORMAL normal, the default: no priority hint (SYCL) / the default stream priority (CUDA, where
 *        the default is also the lowest). TD-574: on the Arc B60 the GuC resets low-priority
 *        compute contexts that a queue with no hint runs without a reset.
 * HIGH   full: priority_high (SYCL) / the highest stream priority (CUDA).
 */
enum class QueuePriority : std::uint8_t {
    LOW,
    NORMAL,
    HIGH,
};

}  // namespace gpu
}  // namespace adai
