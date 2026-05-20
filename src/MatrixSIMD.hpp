#pragma once
// ============================================================================
// SIMD platform detection and helper utilities for Matrix operations.
//
// Compile-time capability macros are set by the compiler when:
//   -mavx2 -mfma   (explicit ENABLE_SIMD=ON)
//   -march=native  (automatic in Release builds)
//
// Runtime detection uses CPUID (x86/x86_64) intrinsics so callers can
// verify that the ISA selected at compile-time is actually available on
// the executing CPU.
// ============================================================================

#include <cstdint>

// ===== Compile-time SIMD capability detection =====

#if defined(__AVX2__)
#define ADAI_SIMD_AVX2
#include <immintrin.h>
#if defined(__FMA__)
#define ADAI_SIMD_FMA
#endif
#elif defined(__SSE4_1__)
#define ADAI_SIMD_SSE4
#include <smmintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define ADAI_SIMD_NEON
#include <arm_neon.h>
#endif

namespace adai {
namespace simd {

// ===== Runtime CPU feature detection (x86/x86_64) =====

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#if defined(_MSC_VER)
#include <intrin.h>

// NOLINTBEGIN(modernize-avoid-c-arrays)
inline void cpuid_query(int info[4], int leaf, int subleaf = 0) {
    __cpuidex(info, leaf, subleaf);
}
// NOLINTEND(modernize-avoid-c-arrays)
#else
#include <cpuid.h>

// NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-init-variables)
inline void cpuid_query(int info[4], int leaf, int subleaf = 0) {
    unsigned int a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid_count(static_cast<unsigned int>(leaf), static_cast<unsigned int>(subleaf), &a,
                           &b, &c, &d)) {
        info[0] = info[1] = info[2] = info[3] = 0;
        return;
    }
    info[0] = static_cast<int>(a);
    info[1] = static_cast<int>(b);
    info[2] = static_cast<int>(c);
    info[3] = static_cast<int>(d);
}
// NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-init-variables)
#endif

/// Returns true when the executing CPU supports AVX2 (CPUID leaf 7, EBX bit 5).
inline bool has_avx2() {
    // NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-init-variables)
    int info0[4] = {};
    cpuid_query(info0, 0);
    if (info0[0] < 7) {
        return false;  // max leaf < 7 → no AVX2
    }

    int info7[4] = {};
    cpuid_query(info7, 7, 0);
    return (info7[1] >> 5) & 1;  // EBX bit 5 = AVX2
    // NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-init-variables)
}

/// Returns true when the executing CPU supports FMA3 (CPUID leaf 1, ECX bit 12).
inline bool has_fma() {
    // NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-init-variables)
    int info[4] = {};
    cpuid_query(info, 1);
    return (info[2] >> 12) & 1;  // ECX bit 12 = FMA
    // NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-init-variables)
}

#else  // Non-x86 architectures

inline bool has_avx2() {
    return false;
}
inline bool has_fma() {
    return false;
}

#endif  // x86 detection

// ===== Compile-time SIMD helper functions =====

#ifdef ADAI_SIMD_AVX2

/// Horizontal sum of 8 single-precision floats packed in a __m256.
inline float hsum256(__m256 v) {
    // Add upper 4 lanes to lower 4 lanes
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    // Two rounds of adjacent-pair horizontal add collapse 4 → 2 → 1
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

#endif  // ADAI_SIMD_AVX2

#ifdef ADAI_SIMD_NEON

/// Horizontal sum of 4 single-precision floats packed in a float32x4_t.
inline float hsum128(float32x4_t v) {
    return vaddvq_f32(v);
}

#endif  // ADAI_SIMD_NEON

}  // namespace simd
}  // namespace adai
