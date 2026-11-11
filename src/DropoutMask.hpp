#pragma once

// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10

#include <cstdint>

#if defined(__CUDACC__)
#define ADAI_HOST_DEVICE __host__ __device__
#else
#define ADAI_HOST_DEVICE
#endif

namespace adai {

/**
 * TD-344: one dropout mask element, a pure function of (seed, index), so the CPU, the CUDA kernel
 * and the SYCL kernel draw identical masks for the same seed and the GPU needs no random state of
 * its own. A splitmix64 hash of the index gives a uniform u in [0, 1); the element is kept, scaled
 * by 1 / keep_prob (inverted dropout, so nothing changes at inference), when u < keep_prob.
 */
ADAI_HOST_DEVICE inline float dropout_mask_value(std::uint64_t seed, std::uint64_t index,
                                                 float keep_prob) {
    std::uint64_t z = seed + (index + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const float u = static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
    return u < keep_prob ? 1.0f / keep_prob : 0.0f;
}

}  // namespace adai
