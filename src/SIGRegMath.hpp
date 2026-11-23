#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-22

#include <cmath>
#include <cstdint>

#ifndef ADAI_HOST_DEVICE
#if defined(__CUDACC__)
#define ADAI_HOST_DEVICE __host__ __device__
#else
#define ADAI_HOST_DEVICE
#endif
#endif

namespace adai {

/**
 * TD-586: SIGReg's per-element math, shared by the host SIGReg, the CUDA kernels and the SYCL kernels so
 * every backend computes the same thing (up to float rounding in log/cos/sin).
 */

/// SIGReg's frequency grid: t = 0.5, 1.0, 1.5, 2.0 (see SIGReg.hpp's design note; TD-504 tracks
/// replacing it with a weighted quadrature).
inline constexpr int kSigregNumFrequencies = 4;
ADAI_HOST_DEVICE inline float sigreg_frequency(int j) {
    return 0.5f * static_cast<float>(j + 1);
}

/// splitmix64 of a counter: a well-mixed 64-bit value for (seed, stream, index).
ADAI_HOST_DEVICE inline std::uint64_t counter_hash(std::uint64_t seed, std::uint64_t stream, std::uint64_t index) {
    std::uint64_t z = seed + stream * 0xD1B54A32D192ED03ull + (index + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// A standard normal draw that is a pure function of (seed, stream, index): Box-Muller over two hashed
/// uniforms, so a GPU kernel draws the same value as the host without any random state.
ADAI_HOST_DEVICE inline float counter_normal(std::uint64_t seed, std::uint64_t stream, std::uint64_t index) {
    const std::uint64_t a = counter_hash(seed, stream, 2 * index);
    const std::uint64_t b = counter_hash(seed, stream, 2 * index + 1);
    const float u1 = static_cast<float>((a >> 40) + 1) * (1.0f / 16777216.0f);  // (0, 1]: log is finite
    const float u2 = static_cast<float>(b >> 40) * (1.0f / 16777216.0f);        // [0, 1)
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530717958647692f * u2);
}

/// Unnormalized element (row i, column k) of SIGReg's direction matrix [d_model, num_sketches] for one draw:
/// element index i * num_sketches + k. Each column is normalized to unit length afterwards.
ADAI_HOST_DEVICE inline float sigreg_direction_raw(std::uint64_t seed, std::uint64_t draw, std::uint64_t element) {
    return counter_normal(seed, draw, element);
}

}  // namespace adai
