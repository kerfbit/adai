package com.adai.ops.network.dto

// @adai-status: beta
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import kotlinx.serialization.Serializable

@Serializable
data class TrainerHealthDto(
    val status: String? = null,
)

/**
 * GET/PUT /admin/config. Field names verified against TrainerAdminAPI.cpp's
 * handle_admin_get_config/handle_admin_put_config. Unlike the other three daemons, there
 * is no admin_enabled/allow_control field here — the whole admin port is opt-in
 * (TRAINER_ADMIN_ENABLED=false by default) and simply isn't listening at all when
 * disabled, so a connection failure (ApiResult.NetworkError), not a 403 body flag, is
 * the "disabled" signal for this screen. port/host/dir are immutable at runtime
 * (TRAINER_ADMIN_PORT/TRAINER_ADMIN_HOST/TRAINER_ADMIN_DIR, file/CLI-only) and are not
 * part of this DTO at all — PUT rejects them with 400 if sent. See CLAUDE.md
 * "Incremental trainer admin API".
 */
@Serializable
data class TrainerAdminConfigDto(
    val auto_save_enabled: Boolean = false,
    val auto_save_every_samples: Int = 0,
    val auto_save_every_minutes: Int = 0,
    val max_sessions_to_keep: Int = 0,
)

/**
 * GET /admin/status. Field names verified against TrainerAdminAPI.cpp's handle_status.
 * phase is one of: idle | loading_data | tokenizing | training | checkpointing | pausing
 * (TrainerControlState.hpp's TrainerPhase, see to_string()).
 */
@Serializable
data class TrainerStatusDto(
    val phase: String = "idle",
    val paused: Boolean = false,
    val run_id: String = "",
    val session_id: String = "",
    val model_name: String = "",
    val current_epoch: Int = 0,
    val total_epochs: Int = 0,
    val samples_trained_this_pass: Long = 0,
    val last_loss: Double = 0.0,
    val best_loss: Double = 0.0,
    val checkpoints_written: Long = 0,
    val last_checkpoint_path: String = "",
    val last_checkpoint_time_unix: Long = 0,
)

/**
 * POST /admin/checkpoint[?wait_ms=N]. 409 (Response inspected, not thrown) when the
 * trainer is idle — no active pass to checkpoint. completed is only true when wait_ms
 * was given and a new checkpoint was observed within that window; otherwise the request
 * was accepted but hasn't necessarily fired yet.
 */
@Serializable
data class TrainerCheckpointResultDto(
    val requested: Boolean = false,
    val completed: Boolean = false,
    val checkpoints_written: Long = 0,
    val checkpoint_path: String = "",
)

/** POST /admin/pause and POST /admin/resume both return this shape (always 202). */
@Serializable
data class TrainerPauseResultDto(
    val paused: Boolean = false,
)

/**
 * One entry in GET /admin/logs — field names verified against TrainerAdminAPI.cpp's
 * handle_logs and TrainerControlState.hpp's TrainerLogEntry. These are real messages
 * from the daemon's own log (see TrainerControlState::log(), which appends to this
 * ring buffer *and* emits through adai::Logger at the same level, so what this screen
 * shows is the same text that lands in the daemon's log file/journal), not a
 * client-fabricated summary. `level` is one of: debug | info | warn | error — the same
 * four levels as adai::Logger::Level.
 */
@Serializable
data class TrainerLogEntryDto(
    val id: Long = 0,
    val timestamp_unix_ms: Long = 0,
    val level: String = "info",
    val message: String = "",
)

@Serializable
data class TrainerLogsResponseDto(
    val entries: List<TrainerLogEntryDto> = emptyList(),
)
