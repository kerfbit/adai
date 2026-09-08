package com.adai.ops.network.dto

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import kotlinx.serialization.Serializable

/** Every hand-rolled JSON error body across metrics/MNS/registry servers uses this shape. */
@Serializable
data class ErrorResponseDto(
    val error: String? = null,
)
