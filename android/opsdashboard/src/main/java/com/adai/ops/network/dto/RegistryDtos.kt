package com.adai.ops.network.dto

// @adai-status: beta
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import kotlinx.serialization.Serializable

/**
 * Field names verified against RegistryServer.cpp handlers. run_id has no
 * timestamp anywhere in the wire format, so there is no server-side notion of
 * "claimed since" — the UI must not fabricate one. added_utc (Phase 15) is a
 * separate concept: when the entry first entered the system, not when it was
 * claimed by a run.
 *
 * size_bytes/num_entries/checksum are best-effort: a plain /pending/add'd path
 * the registry can't read locally (e.g. one never actually staged under its
 * own data_dir) leaves size_bytes=0, num_entries=-1, checksum="". Anything
 * created via fetch/gutenberg, fetch/huggingface, or /upload is always
 * locally readable, so those are populated for real.
 */
@Serializable
data class QueueEntryDto(
    val path: String,
    val run_id: String = "",
    val model_name: String = "",
    val source: String = "",
    val added_utc: String = "",
    val size_bytes: Long = 0,
    val num_entries: Int = -1,
    val checksum: String = "",
)

@Serializable
data class QueueResponseDto(
    val entries: List<QueueEntryDto> = emptyList(),
)

/**
 * source/added_utc (Phase 15) are carried forward from the originating
 * QueueEntryDto at the moment a file was marked trained — added_utc reflects
 * when the file first entered the system, not when training finished. Both
 * are empty for pre-Phase-15 registry entries.
 */
@Serializable
data class RegistryEntryDto(
    val data_file: String,
    val checksum: String = "",
    val num_samples: Int = 0,
    val trained: Boolean = false,
    val added_utc: String = "",
    val source: String = "",
)

@Serializable
data class RegistryResponseDto(
    val entries: List<RegistryEntryDto> = emptyList(),
)

@Serializable
data class RunsResponseDto(
    val runs: Map<String, List<String>> = emptyMap(),
)

@Serializable
data class HistoryEntryDto(
    val data_file: String,
    val checksum: String = "",
    val num_samples: Int = 0,
    val trained: Boolean = false,
    val model_id: String = "",
)

@Serializable
data class HistoryResponseDto(
    val entries: List<HistoryEntryDto> = emptyList(),
)

/** For a force-release, send run_id = "" — the server bypasses the owner check on empty run_id. */
@Serializable
data class ReleaseRequestDto(
    val run_id: String,
    val files: List<String>,
)

@Serializable
data class ReleaseResponseDto(
    val released: Int = 0,
)

/** Empty/absent [paths] assigns every pending entry in the group, not just none. */
@Serializable
data class AssignRequestDto(
    val model_name: String,
    val paths: List<String> = emptyList(),
)

@Serializable
data class AssignResponseDto(
    val assigned: Int = 0,
)

@Serializable
data class FetchGutenbergRequestDto(
    val book_id: Int,
    val num_pairs: Int = 500,
    val model_name: String = "",
)

@Serializable
data class FetchHuggingfaceRequestDto(
    val dataset_id: String,
    val num_pairs: Int = 500,
    val split: String = "train",
    val input_field: String = "",
    val output_field: String = "",
    val model_name: String = "",
)

/**
 * Shared response shape for both fetch endpoints. On failure the server returns
 * a non-2xx status with added=false and a reason (e.g. "fetch_failed"); on
 * success served_from_row/next_row describe the rotating-slice window served
 * (see RegistryServer.cpp's Phase 12/13 per-model cursor tracking).
 */
@Serializable
data class FetchResponseDto(
    val added: Boolean = false,
    val path: String = "",
    val served_from_row: Int? = null,
    val next_row: Int? = null,
    val pairs_written: Int? = null,
    val reason: String? = null,
)

@Serializable
data class RegistryHealthDto(
    val status: String? = null,
)

/**
 * GET/PUT /admin/config. Field names verified against RegistryServer.cpp's
 * admin_config_json/handle_admin_put_config. port/data_dir/ftp_* connection settings are
 * immutable at runtime; only ftp_token_ttl_minutes/ftp_max_sessions_per_run are PUT-able.
 * ftp_max_sessions_per_run_applied is only present on the PUT response, reflecting whether
 * the new value took effect immediately (false if an FtpDataServer instance is already
 * running — it's picked up on the next restart instead). See CLAUDE.md "Daemon admin
 * config API".
 */
@Serializable
data class RegistryAdminConfigDto(
    val ftp_token_ttl_minutes: Int = 0,
    val ftp_max_sessions_per_run: Int = 0,
    val admin_enabled: Boolean = false,
    val ftp_max_sessions_per_run_applied: Boolean? = null,
)
