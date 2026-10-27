// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-26
#pragma once

/**
 * @file ActivationConstants.hpp
 * @brief Constants shared by the CPU activations and the CUDA/SYCL kernels (TD-216)
 *
 * Plain namespace-scope constexpr floats, usable from host C++, CUDA device code and SYCL kernels.
 * The GELU tanh-approximation constants used to be typed separately in Activation.hpp and in each
 * GPU backend's forward and backward kernels, so a change in one place would have left CPU and GPU
 * training computing different activations.
 */
namespace adai::activation {

/// GELU tanh approximation: 0.5 * x * (1 + tanh(kSqrt2OverPi * (x + kGeluCoef * x^3)))
constexpr float kGeluCoef = 0.044715f;
constexpr float kSqrt2OverPi = 0.7978845608f;  // sqrt(2 / pi)

}  // namespace adai::activation
