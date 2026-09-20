package com.adai.ops.network.dto

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.5.0
// @adai-reviewed: 2026-09-20


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
 *
 * TD-205: segment_start/segment_count default to -1 ("the whole file", reproducing every
 * pre-TD-205 entry byte-for-byte) — when segment_start >= 0 this entry is one row-range
 * segment of `path`'s own JSONL pairs, not the whole file; num_entries for a segment is its
 * own pair count (segment_count), never the whole physical file's.
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
    val segment_start: Int = -1,
    val segment_count: Int = -1,
) {
    /** True if this entry is a row-range segment of [path] rather than the whole file. */
    val isSegment: Boolean get() = segment_start >= 0
}

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

/**
 * TD-205: one entry in an explicit segment-targeting list, additive alongside a request's own
 * plain `paths`/`files` list — see RegistryTransport.hpp's `SegmentTarget` (server/C++ side)
 * for the exact same shape. `segment_start < 0` addresses the whole-file entry for [path],
 * same convention as [QueueEntryDto].
 */
@Serializable
data class SegmentTargetDto(
    val path: String,
    val segment_start: Int = -1,
    val segment_count: Int = -1,
)

/** For a force-release, send run_id = "" — the server bypasses the owner check on empty run_id. */
@Serializable
data class ReleaseRequestDto(
    val run_id: String,
    val files: List<String>,
    val segments: List<SegmentTargetDto> = emptyList(),
)

@Serializable
data class ReleaseResponseDto(
    val released: Int = 0,
)

/**
 * Empty/absent [paths] (and empty [segments]) assigns every pending entry in the group, not
 * just none. TD-205: non-empty [segments] targets exactly those segment entries, taking
 * precedence over [paths]/[count] — see handle_assign's own mode-priority doc comment.
 */
@Serializable
data class AssignRequestDto(
    val model_name: String,
    val paths: List<String> = emptyList(),
    val count: Int = 0,
    val segments: List<SegmentTargetDto> = emptyList(),
)

@Serializable
data class AssignResponseDto(
    val assigned: Int = 0,
    val paths: List<String> = emptyList(),
    val segments: List<SegmentTargetDto> = emptyList(),
)

/**
 * Reverses [AssignRequestDto]: clears model_name back to unassigned. Both [paths]/[segments]
 * empty + non-empty [model_name] is the bulk "clear everything assigned to this model" mode.
 */
@Serializable
data class UnassignRequestDto(
    val model_name: String = "",
    val paths: List<String> = emptyList(),
    val force: Boolean = false,
    val segments: List<SegmentTargetDto> = emptyList(),
)

@Serializable
data class UnassignResponseDto(
    val unassigned: Int = 0,
    val skipped: Int = 0,
    val paths: List<String> = emptyList(),
    val segments: List<SegmentTargetDto> = emptyList(),
)

/** At least one of [paths]/[segments] must be non-empty — there is no bulk "delete everything". */
@Serializable
data class DeleteRequestDto(
    val paths: List<String> = emptyList(),
    val force: Boolean = false,
    val delete_files: Boolean = false,
    val segments: List<SegmentTargetDto> = emptyList(),
)

@Serializable
data class DeleteDetailDto(
    val path: String,
    val status: String,
    val file_deleted: Boolean = false,
    val segment_start: Int = -1,
    val segment_count: Int = -1,
)

@Serializable
data class DeleteResponseDto(
    val deleted: Int = 0,
    val skipped: Int = 0,
    val not_found: Int = 0,
    val details: List<DeleteDetailDto> = emptyList(),
)

/**
 * POST /registry/{group}/pending/add {"path":...} — queues an already-existing path (one the
 * registry_server can read from its own data_dir) without fetching/uploading new bytes.
 * TD-205: segment_start/segment_count queue a specific row-range segment instead of the whole
 * file. Shares its response shape with the upload endpoint (see UploadResponseDto's own note).
 */
@Serializable
data class PendingAddRequestDto(
    val path: String,
    val segment_start: Int = -1,
    val segment_count: Int = -1,
)

/**
 * Shared response shape for both pending/add and upload — both endpoints return
 * {"added":bool} on success (upload also echoes back "path"; pending/add doesn't) and
 * {"added":false,"reason":"..."} on failure/already-pending.
 */
@Serializable
data class PendingAddResponseDto(
    val added: Boolean = false,
    val path: String = "",
    val reason: String? = null,
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
