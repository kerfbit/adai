package com.adai.ops.network.dto

// @adai-status: beta        (TD-196 — kind/connection fields + register/link-world-model DTOs added)
// @adai-version: 0.5.0
// @adai-reviewed: 2026-09-19


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

/**
 * TD-196: the "connection standard" between a chatbot's encoder/decoder and its (optional) world
 * model + hippocampal memory. Field names verified against ModelNameService.cpp's
 * serialize_connection/parse_connection. Which fields are meaningful depends on the owning
 * record's own [ModelRecordDto.kind]:
 * - encoder_name/decoder_name: chatbot-kind only, set at register time, immutable.
 * - world_model_name and everything else in the middle group: chatbot-kind only, mutable via
 *   POST /models/{name}/link-world-model. Empty world_model_name means no world model attached.
 * - sigreg_lambda/sigreg_num_sketches: world_model-kind only, set at register time.
 */
@Serializable
data class ConnectionDto(
    val encoder_name: String = "",
    val decoder_name: String = "",
    val world_model_name: String = "",
    val world_model_inject_every_n_layers: Long = 0,
    val hippocampal_memory_enabled: Boolean = false,
    val hippocampal_memory_capacity: Long = 512,
    val hippocampal_repetition_alpha: Float = 0.0f,
    val hippocampal_repetition_decay: Float = 0.95f,
    val hippocampal_cross_reference_alpha: Float = 0.0f,
    val hippocampal_association_decay: Float = 0.95f,
    val sigreg_lambda: Float = 1.0f,
    val sigreg_num_sketches: Long = 64,
)

/**
 * state is one of: initializing | training | candidate | production | retired.
 * TD-196: kind is one of: encoder | decoder | world_model | chatbot (default "chatbot" —
 * every pre-TD-196 record keeps the same shape). See [ConnectionDto]'s own doc comment for how
 * kind selects which of [arch]'s 6 inline fields apply and how [connection] is interpreted.
 */
@Serializable
data class ModelRecordDto(
    val model_id: String = "",
    val model_name: String = "",
    val role: String = "",
    val kind: String = "chatbot",
    val state: String = "initializing",
    val run_id: String = "",
    val created_utc: String = "",
    val updated_utc: String = "",
    val artifact: ArtifactDto = ArtifactDto(),
    val arch: ArchDto = ArchDto(),
    val connection: ConnectionDto = ConnectionDto(),
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
 *
 * TD-147 (fixed): the "clear stale lock" admin action (state="candidate", recovering a model
 * stuck in "training") must send the model's own current run_id — the server's ownership check
 * on that transition rejects a missing/mismatched run_id with 409, even though an empty run_id
 * is treated as an override on some other endpoints (e.g. DatasetRegistry's /release). See
 * ModelRepository.clearStaleTrainingLock() / ModelDetailViewModel.clearStaleTrainingLock().
 * retireCandidate() (state="retired") is the one that always sends no run_id.
 */
@Serializable
data class SetStateRequestDto(
    val state: String,
    val run_id: String? = null,
)

/**
 * TD-196: POST /models body. Unlike mns_cli's own C++ (which conditionally omits "arch"/
 * "connection" per kind), this always sends both in full — a zero-valued [arch] alongside
 * [connection]'s encoder_name/decoder_name is inert server-side (the chatbot-with-linked-encoder/
 * decoder validation path never reads the inline arch fields), and an empty-string [connection]
 * is likewise inert for encoder/decoder/legacy-chatbot registration, since those are exactly the
 * server-side ModelRecord's own defaults. See ModelRepository.registerModel().
 */
@Serializable
data class RegisterModelRequestDto(
    val model_name: String,
    val role: String = "",
    val kind: String = "chatbot",
    val run_group: String = "",
    val arch: ArchDto = ArchDto(),
    val connection: ConnectionDto = ConnectionDto(),
    val tags: Map<String, String> = emptyMap(),
)

/** {"model_id":"...","state":"initializing"} on 201. */
@Serializable
data class RegisterModelResultDto(
    val model_id: String = "",
    val state: String = "",
)

/**
 * TD-196: POST /models/{name}/link-world-model body — the connection fields relevant to a
 * world-model link, minus encoder_name/decoder_name (immutable, register-only). Empty
 * world_model_name detaches. Field names/defaults match ConnectionDto's own.
 */
@Serializable
data class LinkWorldModelRequestDto(
    val world_model_name: String,
    val world_model_inject_every_n_layers: Long = 0,
    val hippocampal_memory_enabled: Boolean = false,
    val hippocampal_memory_capacity: Long = 512,
    val hippocampal_repetition_alpha: Float = 0.0f,
    val hippocampal_repetition_decay: Float = 0.95f,
    val hippocampal_cross_reference_alpha: Float = 0.0f,
    val hippocampal_association_decay: Float = 0.95f,
)

/** {"status":"ok","world_model_name":"..."} on 200. */
@Serializable
data class LinkWorldModelResultDto(
    val status: String = "",
    val world_model_name: String = "",
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
