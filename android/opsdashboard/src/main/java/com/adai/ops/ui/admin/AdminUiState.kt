package com.adai.ops.ui.admin

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import com.adai.ops.network.dto.MetricsAdminConfigDto
import com.adai.ops.network.dto.MnsAdminConfigDto
import com.adai.ops.network.dto.RegistryAdminConfigDto

/**
 * One daemon's *_error is set independently of the others — a 403 on mns_server (admin
 * disabled) or a 404 on metrics_api_server (admin routes not registered, see
 * MetricsAdminConfigDto's doc comment) must not block the other two sections from
 * rendering their live config.
 */
data class AdminUiState(
    val mnsConfig: MnsAdminConfigDto? = null,
    val mnsError: String? = null,
    val registryConfig: RegistryAdminConfigDto? = null,
    val registryError: String? = null,
    val metricsConfig: MetricsAdminConfigDto? = null,
    val metricsError: String? = null,
    val isLoading: Boolean = true,
    val actionInProgress: Boolean = false,
    val actionMessage: String? = null,
)
