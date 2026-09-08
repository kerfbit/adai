package com.adai.ops.network.dto

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import kotlinx.serialization.Serializable

/** Field names verified against ModelNameService.cpp's serialize_record/resolve_json. */
@Serializable
data class ArtifactDto(
    val host: String = "",
    val path: String = "",
    val checksum: String = "",
    val format: String = "adai-native",
)

@Serializable
data class ArchDto(
    val d_model: Long = 0,
    val num_heads: Long = 0,
    val d_ff: Long = 0,
    val num_encoder_layers: Long = 0,
    val num_decoder_layers: Long = 0,
    val max_seq_length: Long = 0,
)

@Serializable
data class TrainingHistoryEntryDto(
    val run_id: String = "",
    val metrics_session_key: String = "",
    val dataset_group: String = "",
    val epochs: Int = 0,
    val final_loss: Double = 0.0,
    val started_utc: String = "",
    val finished_utc: String = "",
)

/** state is one of: initializing | training | candidate | production | retired. */
@Serializable
data class ModelRecordDto(
    val model_id: String = "",
    val model_name: String = "",
    val role: String = "",
    val state: String = "initializing",
    val run_id: String = "",
    val created_utc: String = "",
    val updated_utc: String = "",
    val artifact: ArtifactDto = ArtifactDto(),
    val arch: ArchDto = ArchDto(),
    val training_history: List<TrainingHistoryEntryDto> = emptyList(),
    val tags: Map<String, String> = emptyMap(),
)

@Serializable
data class ModelsResponseDto(
    val models: List<ModelRecordDto> = emptyList(),
)

/** Returned by GET /models/{name}/resolve and GET /roles/{role}/production. */
@Serializable
data class ResolvedModelDto(
    val model_id: String = "",
    val model_name: String = "",
    val state: String = "",
    val artifact: ArtifactDto = ArtifactDto(),
)

@Serializable
data class RoleDto(
    val role: String,
    val production_model: String = "",
)

@Serializable
data class RolesResponseDto(
    val roles: List<RoleDto> = emptyList(),
)

/**
 * state must be "training" | "candidate" | "retired" (server-validated state machine).
 * For the "clear stale lock" admin action, always send state="retired" with no run_id.
 */
@Serializable
data class SetStateRequestDto(
    val state: String,
    val run_id: String? = null,
)

@Serializable
data class PromoteRequestDto(
    val model_name: String,
)

@Serializable
data class PromoteResultDto(
    val promoted: String = "",
    val retired: String = "",
    val role: String = "",
)

@Serializable
data class MnsHealthDto(
    val status: String? = null,
    val model_count: Int = 0,
    val uptime_seconds: Long = 0,
)

/**
 * GET /admin/config. Field names verified against ModelNameService.cpp's
 * handle_admin_get_config. port/data_dir are immutable at runtime (shown read-only in
 * the UI); only registry_url/registry_group are PUT-able. See CLAUDE.md "Daemon admin
 * config API".
 */
@Serializable
data class MnsAdminConfigDto(
    val port: Int = 0,
    val data_dir: String = "",
    val registry_url: String = "",
    val registry_group: String = "",
    val admin_enabled: Boolean = false,
)
