package com.adai.ops.ui.trainer

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import com.adai.ops.network.dto.TrainerAdminConfigDto
import com.adai.ops.network.dto.TrainerStatusDto

/**
 * Severity for [TrainerLogEvent] — deliberately the same four levels
 * `TrainerLogEntryDto.level` can carry (debug/info/warn/error, matching
 * adai::Logger::Level on the daemon side), not a bespoke UI taxonomy — see
 * [TrainerLogEvent]'s doc comment.
 */
enum class TrainerLogSeverity { DEBUG, INFO, WARN, ERROR }

/**
 * One entry in the Trainer screen's activity log — a real message from the
 * daemon's own log, fetched via GET /admin/logs (see TrainerRepository.getLogs()
 * and TrainerLogEntryDto). TrainerControlState::log() on the C++ side appends to
 * the same ring buffer this reads *and* emits through adai::Logger at the
 * matching level, so this is the same text that lands in the daemon's log
 * file/journal, not a client-side summary of status-poll transitions.
 */
data class TrainerLogEvent(
    val id: Long,
    val timestampMillis: Long,
    val severity: TrainerLogSeverity,
    val message: String,
)

/**
 * status/statusError and config/configError are independent, like AdminUiState's
 * per-daemon fields — a status poll failure (e.g. the admin port isn't reachable right
 * now) must not blank out an already-loaded config section, and vice versa.
 */
data class TrainerUiState(
    val status: TrainerStatusDto? = null,
    val statusError: String? = null,
    val config: TrainerAdminConfigDto? = null,
    val configError: String? = null,
    val isLoading: Boolean = true,
    val actionInProgress: Boolean = false,
    val actionMessage: String? = null,
    val events: List<TrainerLogEvent> = emptyList(),
)
