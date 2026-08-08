package com.adai.ops.network.dto

import kotlinx.serialization.Serializable

/** Every hand-rolled JSON error body across metrics/MNS/registry servers uses this shape. */
@Serializable
data class ErrorResponseDto(
    val error: String? = null,
)
