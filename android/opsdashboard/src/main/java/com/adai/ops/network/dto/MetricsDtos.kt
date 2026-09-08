package com.adai.ops.network.dto

// @adai-status: beta
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import kotlinx.serialization.Serializable

/** Field names verified against docs/development/TRAINING_METRICS_API.md. */
@Serializable
data class SessionSummaryDto(
    val key: String,
    val session_id: Int = 0,
    val is_training: Boolean = false,
    val current_epoch: Int = 0,
    val total_epochs: Int = 0,
    val current_loss: Double = 0.0,
    val best_validation_loss: Double = 0.0,
    val session_start_time: Long = 0,
    val last_update_time: Long = 0,
    val metrics_url: String? = null,
)

@Serializable
data class SessionsResponseDto(
    val sessions: List<SessionSummaryDto> = emptyList(),
    val total: Int = 0,
    val live: Int = 0,
)

@Serializable
data class CurrentMetricsDto(
    val session_id: Int = 0,
    val is_training: Boolean = false,
    val current_epoch: Int = 0,
    val total_epochs: Int = 0,
    val current_sample: Int = 0,
    val total_samples: Int = 0,
    val current_loss: Double = 0.0,
    val current_validation_loss: Double = 0.0,
    val current_learning_rate: Double = 0.0,
    val current_gradient_norm: Double = 0.0,
    val current_perplexity: Double = 0.0,
    val running_loss: Double = 0.0,
    val running_validation_loss: Double = 0.0,
    val samples_per_second: Double = 0.0,
    val estimated_time_remaining_seconds: Double = 0.0,
    val best_validation_loss: Double = 0.0,
    val best_epoch: Int = 0,
    val total_samples_trained: Long = 0,
    val total_training_time_seconds: Double = 0.0,
    val gradient_variance: Double = 0.0,
    val compute_time_ratio: Double = 0.0,
    val weight_update_ratio: Double = 0.0,
    val activation_saturation_ratio: Double = 0.0,
    val current_validation_perplexity: Double = 0.0,
    val current_validation_accuracy: Double = -1.0,
    val current_bleu4: Double = -1.0,
    val current_rouge1: Double = -1.0,
    val current_rouge2: Double = -1.0,
    val current_rougeL: Double = -1.0,
)

@Serializable
data class SessionStatusDto(
    val is_training: Boolean = false,
    val session_id: Int = 0,
    val current_epoch: Int = 0,
    val total_epochs: Int = 0,
    val current_sample: Int = 0,
    val total_samples: Int = 0,
    val progress_percent: Double = 0.0,
    val samples_per_second: Double = 0.0,
    val estimated_time_remaining_seconds: Double = 0.0,
    val is_stale: Boolean = false,
    val seconds_since_last_update: Double = 0.0,
    val effective_is_training: Boolean = false,
)

@Serializable
data class EpochHistoryDto(
    val current_epoch: Int = 0,
    val total_epochs: Int = 0,
    val epoch_losses: List<Double> = emptyList(),
    val epoch_validation_losses: List<Double> = emptyList(),
    val epoch_learning_rates: List<Double> = emptyList(),
    val epoch_perplexities: List<Double> = emptyList(),
    val epoch_durations: List<Double> = emptyList(),
    val epoch_gradient_norms: List<Double> = emptyList(),
    val epoch_adaptive_clip_thresholds: List<Double> = emptyList(),
    val epoch_validation_perplexities: List<Double> = emptyList(),
    val epoch_validation_accuracies: List<Double> = emptyList(),
    val best_validation_loss: Double = 0.0,
    val best_epoch: Int = 0,
)

@Serializable
data class SampleRecordDto(
    val timestamp: String = "",
    val session_id: Int = 0,
    val epoch: Int = 0,
    val sample: Int = 0,
    val loss: Double = 0.0,
    val validation_loss: Double = 0.0,
    val learning_rate: Double = 0.0,
    val gradient_norm: Double = 0.0,
    val perplexity: Double = 0.0,
)

@Serializable
data class SampleHistoryDto(
    val records: List<SampleRecordDto> = emptyList(),
    val count: Int = 0,
)

/** Shape of `/metrics/db-history` — same per-record fields as [SampleRecordDto] minus
 * `session_id` (redundant with the path's session key), plus an echoed `session_key`. */
@Serializable
data class DbHistoryDto(
    val session_key: String = "",
    val records: List<SampleRecordDto> = emptyList(),
    val count: Int = 0,
)

@Serializable
data class GenerationQualityDto(
    val current_bleu4: Double = -1.0,
    val current_rouge1: Double = -1.0,
    val current_rouge2: Double = -1.0,
    val current_rougeL: Double = -1.0,
    val epoch_bleu4: List<Double> = emptyList(),
    val epoch_rouge1: List<Double> = emptyList(),
    val epoch_rouge2: List<Double> = emptyList(),
    val epoch_rougeL: List<Double> = emptyList(),
)

@Serializable
data class PaddingEfficiencyDto(
    val current_padding_efficiency: Double = 0.0,
    val epoch_padding_efficiencies: List<Double> = emptyList(),
)

@Serializable
data class AggregateSessionDto(
    val key: String,
    val epoch: Int = 0,
    val loss: Double = 0.0,
    val validation_loss: Double = 0.0,
)

@Serializable
data class AggregateMetricsDto(
    val live_sessions: Int = 0,
    val sessions: List<AggregateSessionDto> = emptyList(),
)

@Serializable
data class MetricsHealthDto(
    val status: String? = null,
    val service: String? = null,
    val is_training: Boolean = false,
    val any_stale: Boolean = false,
)

/** Generic {"status":"...","message":"..."} shape used by the session/control POST endpoints. */
@Serializable
data class SimpleStatusDto(
    val status: String? = null,
    val message: String? = null,
)

/**
 * GET/PUT /admin/config. Field names verified against TrainingMetricsAPI.cpp's
 * handle_admin_get_config/handle_admin_put_config. All fields except allow_control are
 * PUT-able (port, storage_backend, db_path, db_url, db_pool_size, and the *_file paths are
 * immutable at runtime). Unlike mns_server and registry_server, these routes are not
 * registered at all when allow_control is false at server startup — GET/PUT both 404
 * rather than 403; see CLAUDE.md "Daemon admin config API".
 */
@Serializable
data class MetricsAdminConfigDto(
    val max_live_sessions: Int = 0,
    val completed_ttl_seconds: Int = 0,
    val sweep_interval_seconds: Int = 0,
    val persist_every_samples: Int = 0,
    val persist_every_seconds: Int = 0,
    val max_records_in_memory: Int = 0,
    val max_records_on_disk: Int = 0,
    val enable_prometheus: Boolean = false,
    val staleness_threshold_seconds: Int = 0,
    val allow_control: Boolean = false,
)
