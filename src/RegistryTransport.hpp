#pragma once

// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-11-12

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "SessionCounterStore.hpp"

// ============================================================================
// Data types shared between DatasetRegistry and all transport implementations
// ============================================================================

/**
 * @brief Tracks a single data file across training runs.
 *
 * Moved from DatasetRegistry.hpp (TD-028 Phase 8) so RegistryTransport
 * implementations can read and write records without coupling to DatasetRegistry.
 */
struct DataVersion {
    std::string data_file;
    std::string checksum;
    int num_samples = 0;
    bool trained = false;
    std::string model_id;  ///< Phase 2: MNS model UUID; empty for pre-MNS records
    /// Phase 15: ISO-8601 UTC, carried forward from the originating PendingEntry's
    /// added_utc at commit time (so it reflects when the file first entered the
    /// system, not when training completed). Empty for pre-Phase-15 records.
    std::string added_utc;
    /// Phase 15: "gutenberg" | "huggingface" | "upload" | "manual"; empty = unknown
    /// (pre-Phase-15 record, or the originating PendingEntry had none).
    std::string source;
    /// TD-333: the TD-205 segment this record covers; -1/-1 = the whole file (every pre-TD-333
    /// record). Two records for one path are distinct when their ranges differ, so segments
    /// trained in separate passes are each recorded with their own sample count.
    int segment_start = -1;
    int segment_count = -1;
};

/// TD-333: identity check for a trained record, the counterpart of segment_matches() below.
/// A whole-file query (segment_start < 0) matches only a whole-file record.
inline bool trained_matches(const DataVersion& dv, const std::string& path, int segment_start,
                            int segment_count) {
    if (dv.data_file != path)
        return false;
    if (segment_start < 0)
        return dv.segment_start < 0;
    return dv.segment_start == segment_start && dv.segment_count == segment_count;
}

/**
 * @brief A file in the pending training queue.
 *
 * run_id is the claim: set by acquire() (either transport) for the run training the entry, and
 * cleared by release(); empty = unclaimed.
 */
struct PendingEntry {
    std::string path;
    std::string run_id;      // the claiming run; empty = unclaimed
    std::string model_name;  // target model; empty = unassigned
    /// Phase 15: "gutenberg" | "huggingface" | "upload" | "manual"; empty = legacy/unknown.
    std::string source;
    /// Phase 15: ISO-8601 UTC, set once when the entry is first created.
    std::string added_utc;
    /// Phase 15: size on disk in bytes; 0 = unknown (path not locally readable
    /// by the registry at creation time).
    std::size_t size_bytes = 0;
    /// Phase 15: JSONL line/sample count; -1 = unknown/uncounted.
    int num_entries = -1;
    /// Phase 15: lightweight size+mtime fingerprint (same convention as
    /// FileToken::checksum) — not cryptographic, logging/display only.
    std::string checksum;
    /// TD-205: row-range segment within `path`'s own JSONL pairs; -1/-1 (default) means "the
    /// whole file" — reproducing every pre-TD-205 entry's behavior byte-for-byte. When set,
    /// `segment_start` is a 0-based pair index and `segment_count` is how many pairs from
    /// there this entry covers. Two entries sharing the same `path` are distinct, independent
    /// pending units as long as their (segment_start, segment_count) differ — see
    /// segment_matches() below, the identity check every dedup/lookup site now uses instead of
    /// a bare `path` comparison.
    int segment_start = -1;
    int segment_count = -1;
    /// TD-494: Unix time the current claim (run_id) was made or last renewed; 0 = unknown (no
    /// claim, or a claim made before leases existed). registry_server expires a claim this old
    /// plus its lease, so a crashed trainer's entries return to the pool.
    long long claimed_unix = 0;
};

/**
 * @brief TD-205: identity check for a PendingEntry, used everywhere a bare `e.path == path`
 *        comparison used to suffice before segments existed. A query with `segment_start < 0`
 *        ("give me the whole-file entry") matches only a whole-file entry (itself
 *        `segment_start < 0`) — never a specific segment of that same path, and vice versa.
 *        This is what lets multiple segments of one physical file coexist as independent
 *        pending entries without colliding with each other or with a legacy whole-file entry.
 */
inline bool segment_matches(const PendingEntry& e, const std::string& path, int segment_start,
                            int segment_count) {
    if (e.path != path) {
        return false;
    }
    if (segment_start < 0) {
        return e.segment_start < 0;
    }
    return e.segment_start == segment_start && e.segment_count == segment_count;
}

/** @brief TD-205: one entry in an explicit segment-targeting list (see assign()/unassign()/
 *         delete_paths()/release()'s own `segments` parameter, and AcquireResponse::file_ranges()).
 *         `segment_start < 0` addresses the whole-file entry for `path`, same convention as
 *         PendingEntry itself. */
struct SegmentTarget {
    std::string path;
    int segment_start = -1;
    int segment_count = -1;
};

// ============================================================================
// FTP transport data types (Phase 10: dataset transport)
// ============================================================================

/**
 * @brief TD-205: a trainable unit — a physical file, or a row-range segment of one.
 *
 * Replaces a bare `std::string` path wherever a caller needs to distinguish "read the whole
 * file" from "read only this range" — see IncrementalTrainer::train_on_files()/
 * retrain_on_files() and AcquireResponse::file_ranges().
 */
struct PendingFileRange {
    std::string path;
    int segment_start = -1;
    int segment_count = -1;
};

/**
 * @brief Per-file FTP credential bundle returned by RemoteTransport::acquire().
 *
 * Localhost (LocalTransport) callers only populate registry_path; all ftp_*
 * fields are empty, which signals direct filesystem access.
 */
struct FileToken {
    std::string registry_path;      ///< absolute path as known to the registry
    std::string ftp_path;           ///< path relative to FTP root (data_dir)
    std::string ftp_username;       ///< per-file virtual FTP username
    std::string ftp_password;       ///< per-file FTP password (random 32-byte hex)
    std::string checksum;           ///< size+mtime from DataVersion (logging only)
    std::size_t size_bytes = 0;     ///< expected byte count; used for size verification
    std::string token_expires_utc;  ///< ISO-8601 UTC expiry timestamp
    /// TD-205: carried forward from the originating PendingEntry so a trainer that acquired a
    /// segment (not a whole file) knows which row-range to actually read; -1/-1 = whole file.
    int segment_start = -1;
    int segment_count = -1;
};

/**
 * @brief TD-337: the file name a trainer gives its FTP download of @p registry_path.
 *
 * A 16-hex-digit FNV-1a hash of the full registry path, '_', then the path's basename (kept for
 * readability). Unlike the basename alone, it is unique per registry file: the TD-205 segments of
 * one file share one download, and same-named files in different directories (or kinds) don't
 * collide. StartupSweep computes the same name to match downloads to pending entries.
 */
inline std::string download_file_name(const std::string& registry_path) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const unsigned char c : registry_path) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    static const char* kHex = "0123456789abcdef";
    std::string name(16, '0');
    for (int i = 15; i >= 0; --i, h >>= 4)
        name[static_cast<std::size_t>(i)] = kHex[h & 0xF];
    const auto slash = registry_path.find_last_of("/\\");
    name += '_';
    name += slash == std::string::npos ? registry_path : registry_path.substr(slash + 1);
    return name;
}

/**
 * @brief TD-496: why @p path can't be queued, or an empty string when it can.
 *
 * A queued path must be non-empty, valid UTF-8 and free of control characters (tab, newline,
 * carriage return, ...): they have no use in a data file's name and break line-based tools and
 * logs. Spaces are allowed; the registry files store paths as JSON strings.
 */
std::string registry_path_problem(const std::string& path);

/// TD-338: suffix of the sidecar file holding the token checksum a download was made against.
inline constexpr const char* kDownloadChecksumSuffix = ".checksum";

/**
 * @brief Full response from an acquire call.
 *
 * When ftp_server_host is empty the trainer reads files directly by
 * registry_path (localhost behavior, unchanged).  When ftp_server_host is set
 * the trainer fetches each file via FTP using the per-file credentials.
 */
struct AcquireResponse {
    std::string run_id;
    std::string ftp_server_host;  ///< empty → direct path access
    int ftp_server_port = 2121;
    bool ftps_enabled = false;  ///< Phase 3: use FTPS (TLS) for downloads
    std::vector<FileToken> files;
    /// TD-523: true when the registry couldn't be asked (unreachable, HTTP error, lock failure),
    /// so an empty `files` means "unknown", not "nothing pending".
    bool failed = false;
    /// TD-494: the registry's claim lease in seconds; the claimer must renew within it
    /// (RegistryTransport::renew_claims()). 0 = claims don't expire, nothing to renew.
    int claim_lease_seconds = 0;

    /// Convenience: collect registry_paths for release/mark_trained calls.
    std::vector<std::string> registry_paths() const {
        std::vector<std::string> out;
        out.reserve(files.size());
        for (const auto& f : files)
            out.push_back(f.registry_path);
        return out;
    }

    /// TD-205: like registry_paths(), but also carries each file's segment range (if any) — use
    /// this instead of registry_paths() wherever the caller needs to read only the acquired
    /// row-range rather than the whole file (see IncrementalTrainer::train_on_files()).
    std::vector<PendingFileRange> file_ranges() const {
        std::vector<PendingFileRange> out;
        out.reserve(files.size());
        for (const auto& f : files)
            out.push_back({f.registry_path, f.segment_start, f.segment_count});
        return out;
    }
};

// ============================================================================
// Phase 16: dataset management result types (assign-by-count, unassign, delete)
// ============================================================================

/** @brief Result of RegistryTransport::assign(). */
struct AssignResult {
    int assigned = 0;
    std::vector<std::string> paths;  ///< exact whole-file paths touched, in every mode
    /// TD-205: exact segment-targeted entries touched (empty unless the call used @p segments
    /// or a bulk/count mode happened to also match some segments).
    std::vector<SegmentTarget> segments;
};

/** @brief Result of RegistryTransport::unassign(). */
struct UnassignResult {
    int unassigned = 0;
    /// Entries that matched but were left untouched because they're actively
    /// claimed by a run (non-empty run_id) and force was not set.
    int skipped = 0;
    std::vector<std::string> paths;  ///< exact whole-file paths touched
    std::vector<SegmentTarget> segments;  ///< TD-205: exact segment-targeted entries touched
};

/** @brief Result of RegistryTransport::delete_paths(). */
struct DeleteResult {
    struct Detail {
        std::string path;
        std::string status;       ///< "deleted" | "skipped_active_run" | "not_found"
        bool file_deleted = false;  ///< true only if delete_files was requested and it worked
        /// TD-205: -1/-1 (default) for a whole-file target; otherwise the segment range that
        /// was targeted, so a caller deleting several segments of one path can tell them apart.
        int segment_start = -1;
        int segment_count = -1;
    };
    int deleted = 0;
    int skipped = 0;
    int not_found = 0;
    std::vector<Detail> details;
};

/// Result of requeue_trained(): trained registry entries moved back to the pending queue.
struct RequeueResult {
    struct Detail {
        std::string path;
        std::string status;  ///< "requeued" | "already_pending" | "not_found"
    };
    int requeued = 0;
    int already_pending = 0;
    int not_found = 0;
    bool ok = true;  ///< false if the request itself failed (e.g. registry_server unreachable)
    std::vector<Detail> details;
};

/// Result of upload_file() (TD-552).
struct UploadResult {
    std::string path;       ///< the registry's path for the file; empty when nothing was stored
    bool replaced = false;  ///< overwrite replaced an unclaimed queued file (its entry refreshed)
    /// Why registry_server refused (HTTP 409): "claimed", "segmented", or without overwrite
    /// "trained" / "already_queued". Empty for any other failure.
    std::string refused;
    bool ok() const {
        return !path.empty();
    }
};

// ============================================================================
// Abstract transport interface
// ============================================================================

/**
 * @brief Abstract I/O interface for DatasetRegistry persistence.
 *
 * Decouples DatasetRegistry from storage specifics.  Implementations:
 *   - LocalTransport  — flat files in session_dir             (Phase 8)
 *   - RemoteTransport — HTTP calls to registry_server daemon  (Phase 9)
 *
 * All methods are non-const; implementations may buffer or cache state.
 */
class RegistryTransport {
   public:
    virtual ~RegistryTransport() = default;

    /** @brief Load all DataVersion entries into @p out.  Clears @p out first.
     *  @return true if the backing store was found and parsed. */
    virtual bool load_registry(std::vector<DataVersion>& out) = 0;

    /** @brief Persist @p entries to the backing store.
     *  @return true on success. */
    virtual bool save_registry(const std::vector<DataVersion>& entries) = 0;

    /** @brief Load all pending-queue entries into @p out.  Clears @p out first.
     *  @return true if the backing store was found and parsed. */
    virtual bool load_pending(std::vector<PendingEntry>& out) = 0;

    /** @brief Persist @p entries to the backing store.
     *  @return true on success. */
    virtual bool save_pending(const std::vector<PendingEntry>& entries) = 0;

    // ── Phase 9: distributed queue operations ─────────────────────────────

    /**
     * @brief Atomically acquire up to @p max_files entries for @p run_id.
     *
     * LocalTransport uses an advisory lock file and returns an AcquireResponse
     * with ftp_server_host empty (direct path access).  RemoteTransport uses a
     * single atomic POST /acquire request and returns FTP credentials when the
     * registry server is configured with an FTP server.
     *
     * @p model_name scopes eligibility: an entry is claimable iff it's
     * unassigned (PendingEntry::model_name empty) or assigned to this exact
     * @p model_name — never an entry assigned to a *different* model. An empty
     * @p model_name (no MNS/model identity configured) can only claim
     * unassigned entries. Default "" preserves the pre-assignment-aware
     * behavior for callers that never touch model assignment.
     */
    virtual AcquireResponse acquire(const std::string& run_id, int max_files,
                                    const std::string& model_name = "") = 0;

    /** @brief Release @p run_id's claims on the whole-file entries of @p paths and on exactly
     *  the segment-targeted entries in @p segments (TD-205), in one call (TD-333: it used to
     *  apply only one of the two lists). A whole-file path never releases a segment. */
    virtual void release(const std::string& run_id, const std::vector<std::string>& paths,
                         const std::vector<SegmentTarget>& segments = {}) = 0;

    /**
     * @brief TD-494: renew every claim @p run_id holds, restarting its lease.
     * @return Claims renewed (0 = none left: they expired and were taken by another run), or
     *         -1 when the registry couldn't be asked. Only RemoteTransport has leases; the
     *         default (LocalTransport, test doubles) has nothing to renew and returns 0.
     */
    virtual int renew_claims(const std::string& /*run_id*/) { return 0; }

    /**
     * @brief Atomically append @p new_entries to the registry and remove
     *        @p trained_paths from the pending queue (for @p run_id).
     *
     * LocalTransport writes both files under an advisory lock.
     * RemoteTransport issues a single POST /trained request.
     * Pass an empty @p run_id to release any pending entry regardless of
     * assignment (single-run mode).
     *
     * TD-333: with @p trained_segments non-empty (exact mode, used by DatasetRegistry for every
     * acquired batch), @p trained_paths removes only whole-file pending entries and each segment
     * removes exactly its own entry. With it empty, the pre-TD-333 behaviour applies: every
     * pending entry sharing a trained path is removed.
     */
    virtual void commit_trained(const std::string& run_id,
                                const std::vector<DataVersion>& new_entries,
                                const std::vector<std::string>& trained_paths,
                                const std::vector<SegmentTarget>& trained_segments = {}) = 0;

    /** @brief Atomically append a single @p path to the pending queue.
     *  No-op (returns false) if an entry with the same identity — see segment_matches() — is
     *  already present.
     *  TD-205: @p segment_start/@p segment_count (both -1 by default, meaning "the whole
     *  file") let this add a specific row-range segment instead. Adding two different ranges
     *  of the same @p path creates two independent pending entries, not a dedup conflict.
     *  @return true on success. */
    virtual bool add_pending(const std::string& path, int segment_start = -1,
                             int segment_count = -1) = 0;

    /**
     * @brief Allocate the next session number for (@p model_name, @p run_id),
     *        e.g. "session-01", "session-02". A @p run_id never seen before
     *        starts its counter at 1 — since run_id changes whenever MNS
     *        allocates a new run, this naturally resets per run with no
     *        separate reset signal needed. Persisted durably (survives a
     *        restart of the daemon/process backing this transport).
     */
    virtual std::string next_session(const std::string& model_name,
                                     const std::string& run_id) = 0;

    // ── Phase 16: dataset management (assign-by-count, unassign, delete) ──

    /**
     * @brief Set model_name on pending entries. Modes, checked in order:
     *          - non-empty @p segments     — assign exactly those segment-targeted entries
     *                                        (TD-205; @p paths/@p count ignored)
     *          - non-empty @p paths        — assign exactly those whole-file entries (count
     *                                        ignored)
     *          - empty @p paths, count > 0 — assign the first @p count
     *                                        currently-unassigned whole-file entries
     *          - empty @p paths, count 0   — assign every pending entry (whole-file and
     *                                        segments alike)
     *          - count < 0                 — refused; nothing is assigned (TD-321)
     */
    virtual AssignResult assign(const std::string& model_name,
                                const std::vector<std::string>& paths = {}, int count = 0,
                                const std::vector<SegmentTarget>& segments = {}) = 0;

    /**
     * @brief Clear model_name back to unassigned. If both @p paths and @p segments are empty,
     *        clears every entry currently assigned to @p model_name (bulk mode, requires
     *        non-empty @p model_name). Otherwise clears only the listed whole-file @p paths
     *        and/or segment-targeted @p segments (TD-205); a non-empty @p model_name
     *        additionally acts as an ownership filter. An entry actively claimed by a run
     *        (non-empty run_id) is left untouched unless @p force is true.
     */
    virtual UnassignResult unassign(const std::string& model_name,
                                    const std::vector<std::string>& paths, bool force,
                                    const std::vector<SegmentTarget>& segments = {}) = 0;

    /**
     * @brief Permanently purge entries matching @p paths (whole-file) and/or @p segments
     *        (TD-205, segment-targeted) from both the pending queue and the trained registry.
     *        At least one of @p paths/@p segments must be non-empty — there is no bulk
     *        "delete everything" mode.
     *
     * A pending entry actively claimed by a run (non-empty run_id) is left
     * untouched unless @p force is true; the trained registry has no run_id
     * concept and is always purged unconditionally on a match.
     *
     * @p delete_files additionally unlinks the physical file, but the two
     * implementations differ here: RemoteTransport (registry_server) only
     * unlinks files that resolve inside its own group data_dir (server-owned
     * fetches/uploads) — arbitrary external paths are never touched, since
     * the server has no business reaching outside its managed directory
     * tree. LocalTransport has no such distinction (the caller already has
     * full filesystem access in local mode) and will unlink any existing
     * path, except its own registry_path_/pending_path_ state files.
     */
    virtual DeleteResult delete_paths(const std::vector<std::string>& paths, bool force,
                                      bool delete_files,
                                      const std::vector<SegmentTarget>& segments = {}) = 0;

    // ── Phase 11: server-side dataset fetch ────────────────────────────────
    //
    // In distributed mode the registry_server (not the caller) performs the
    // actual download/storage so the bytes land under its own data_dir where
    // the FTP delivery pipeline (FtpDataServer + DataTransport) can serve
    // them out to trainers. LocalTransport has no server to delegate to, so
    // these are unsupported there — see DatasetRegistry::add_file() /
    // DataFetcher for the local-mode equivalent.

    /**
     * @brief Ask the registry to download a Project Gutenberg book into its
     *        own data_dir and enqueue it as pending.
     *
     * @param model_name Identifies the caller for per-model rotating-slice
     *                    tracking (Phase 13) — the registry serves a
     *                    different slice of the book to each distinct
     *                    model_name rather than always the same sentences.
     *                    Empty is valid and buckets into a shared cursor.
     * @return Registry-relative path of the newly queued file, or "" on failure.
     */
    virtual std::string fetch_gutenberg(int book_id, int num_pairs,
                                        const std::string& model_name) = 0;

    /**
     * @brief Ask the registry to download a HuggingFace dataset into its own
     *        data_dir and enqueue it as pending.
     *
     * @param model_name Identifies the caller for per-model rotating-slice
     *                    tracking (Phase 12) — the registry serves a
     *                    different slice of the dataset to each distinct
     *                    model_name rather than always the same rows. Empty
     *                    is valid and buckets into a shared cursor.
     * @return Registry-relative path of the newly queued file, or "" on failure.
     */
    virtual std::string fetch_huggingface(const std::string& dataset_id, int num_pairs,
                                          const std::string& split,
                                          const std::string& input_field,
                                          const std::string& output_field,
                                          const std::string& model_name) = 0;

    /**
     * @brief Upload a local file's bytes to the registry's own data_dir and
     *        enqueue it as pending.
     * @param local_path Path to a file readable on the caller's machine.
     * @param overwrite TD-552: replace a file of the same name that is already trained (it is
     *        queued again) or queued unclaimed as a whole file. A claimed or segmented file is
     *        refused either way (TD-492).
     * @return The stored path and whether it replaced a queued file, or the refusal reason.
     */
    virtual UploadResult upload_file(const std::string& local_path, bool overwrite) = 0;

    /**
     * @brief Move trained registry entries back to the pending queue, so the next passes
     *        train on them again (e.g. before retraining a model from scratch).
     *
     * Targets the trained records of @p paths, or every trained record when @p all is true.
     * Each one is removed from the trained registry and queued as the unit it records (the
     * whole file, or since TD-333 that segment), keeping its source/added time/checksum. If that
     * unit is already pending, only the trained record is removed ("already_pending"). A non-empty
     * @p model_name assigns the re-queued entries to that model; empty leaves them unassigned.
     * Works against registry_server too (POST .../requeue), so it re-queues where the data actually
     * lives.
     */
    virtual RequeueResult requeue_trained(const std::vector<std::string>& paths, bool all,
                                          const std::string& model_name = "") = 0;
};

// ============================================================================
// Local flat-file transport (Phase 8)
// ============================================================================

/**
 * @brief LocalTransport keeps the queue and the trained registry in two local files. Zero network
 *        dependency; registry_server stores every group through it too.
 *
 * File formats (TD-496): JSON Lines, a header line ("# adai data registry, JSON Lines v2" /
 * "# adai pending queue, JSON Lines v2") then one object per entry with every field named. A file
 * without the header is read in the older space-separated (registry) or tab-separated (queue)
 * form and is written as JSON Lines by the next save. Saves are atomic (TD-495).
 *
 * Every operation that changes the queue holds an advisory lock on pending_path + ".lock"
 * (flock(2), or _locking() on Windows) for its read-modify-write, serialising callers on one host.
 */
class LocalTransport : public RegistryTransport {
   public:
    /**
     * @param registry_path  Full path to the data_registry.txt file.
     * @param pending_path   Full path to the pending_files.txt file.
     */
    LocalTransport(std::string registry_path, std::string pending_path);

    bool load_registry(std::vector<DataVersion>& out) override;
    bool save_registry(const std::vector<DataVersion>& entries) override;
    bool load_pending(std::vector<PendingEntry>& out) override;
    bool save_pending(const std::vector<PendingEntry>& entries) override;
    AcquireResponse acquire(const std::string& run_id, int max_files,
                            const std::string& model_name = "") override;
    void release(const std::string& run_id, const std::vector<std::string>& paths,
                const std::vector<SegmentTarget>& segments = {}) override;
    void commit_trained(const std::string& run_id, const std::vector<DataVersion>& new_entries,
                        const std::vector<std::string>& trained_paths,
                        const std::vector<SegmentTarget>& trained_segments = {}) override;
    bool add_pending(const std::string& path, int segment_start = -1,
                     int segment_count = -1) override;
    std::string next_session(const std::string& model_name, const std::string& run_id) override;

    AssignResult assign(const std::string& model_name, const std::vector<std::string>& paths,
                        int count, const std::vector<SegmentTarget>& segments = {}) override;
    UnassignResult unassign(const std::string& model_name, const std::vector<std::string>& paths,
                            bool force, const std::vector<SegmentTarget>& segments = {}) override;
    DeleteResult delete_paths(const std::vector<std::string>& paths, bool force, bool delete_files,
                              const std::vector<SegmentTarget>& segments = {}) override;

    // Phase 11: not supported in local mode — logs a warning and returns "".
    std::string fetch_gutenberg(int book_id, int num_pairs,
                                const std::string& model_name) override;
    std::string fetch_huggingface(const std::string& dataset_id, int num_pairs,
                                  const std::string& split, const std::string& input_field,
                                  const std::string& output_field,
                                  const std::string& model_name) override;
    UploadResult upload_file(const std::string& local_path, bool overwrite) override;
    RequeueResult requeue_trained(const std::vector<std::string>& paths, bool all,
                                  const std::string& model_name = "") override;

   private:
    std::string registry_path_;
    std::string pending_path_;

    // Session-number counter (Phase 3), lazily opened next to pending_path_ as
    // "session_counters.db" on first next_session() call (TD-308: its own
    // SessionCounterStore, no longer a DaemonConfigStore).
    std::unique_ptr<adai::SessionCounterStore> session_store_;

    // Hold an exclusive flock on pending_path_ + ".lock" for the duration of
    // an acquire/release cycle and return the fd, or -1 on failure.
    int lock_pending() const;
    void unlock_pending(int lock_fd) const;
};

// ============================================================================
// Remote HTTP transport (Phase 9)
// ============================================================================

/**
 * @brief RemoteTransport delegates all registry I/O to a registry_server HTTP
 *        daemon, enabling multiple trainers on different machines to share a
 *        single pending queue without double-training.
 *
 * Only compiled when BUILD_METRICS_API_SERVER is defined (i.e. when
 * cpp-httplib is available; set by CMake alongside adai_core).  When httplib
 * is absent all methods log an error and return failure / empty results.
 *
 * URL format: "http://host[:port]"  (no trailing slash, no path component).
 * Group is the logical namespace segment: requests go to
 *   /registry/<group>/<endpoint>
 */
class RemoteTransport final : public RegistryTransport {
   public:
    /**
     * @param base_url     Scheme + host + optional port, e.g. "http://reg:8082"
     * @param run_group    Logical namespace for this project.
     * @param timeout_ms   HTTP connect/read timeout in milliseconds.
     * @param dataset_kind TD-202: selects a per-trainable-piece sub-pool within the group
     *                     ("encoder"/"decoder"/"world_model"/"chatbot", MNS's own kind
     *                     vocabulary). Empty (default) reproduces the pre-TD-202 single
     *                     shared pool, so every existing caller stays byte-for-byte
     *                     unaffected by this trailing parameter.
     */
    RemoteTransport(std::string base_url, std::string run_group, int timeout_ms = 5000,
                    std::string dataset_kind = "");

    bool load_registry(std::vector<DataVersion>& out) override;
    bool save_registry(const std::vector<DataVersion>& entries) override;
    bool load_pending(std::vector<PendingEntry>& out) override;
    bool save_pending(const std::vector<PendingEntry>& entries) override;
    AcquireResponse acquire(const std::string& run_id, int max_files,
                            const std::string& model_name = "") override;
    void release(const std::string& run_id, const std::vector<std::string>& paths,
                const std::vector<SegmentTarget>& segments = {}) override;
    void commit_trained(const std::string& run_id, const std::vector<DataVersion>& new_entries,
                        const std::vector<std::string>& trained_paths,
                        const std::vector<SegmentTarget>& trained_segments = {}) override;
    bool add_pending(const std::string& path, int segment_start = -1,
                     int segment_count = -1) override;
    std::string next_session(const std::string& model_name, const std::string& run_id) override;

    AssignResult assign(const std::string& model_name, const std::vector<std::string>& paths,
                        int count, const std::vector<SegmentTarget>& segments = {}) override;
    UnassignResult unassign(const std::string& model_name, const std::vector<std::string>& paths,
                            bool force, const std::vector<SegmentTarget>& segments = {}) override;
    DeleteResult delete_paths(const std::vector<std::string>& paths, bool force, bool delete_files,
                              const std::vector<SegmentTarget>& segments = {}) override;

    // Phase 11: delegate to the registry_server, which performs the fetch/
    // upload itself and stores the result under its own data_dir.
    std::string fetch_gutenberg(int book_id, int num_pairs,
                                const std::string& model_name) override;
    std::string fetch_huggingface(const std::string& dataset_id, int num_pairs,
                                  const std::string& split, const std::string& input_field,
                                  const std::string& output_field,
                                  const std::string& model_name) override;
    UploadResult upload_file(const std::string& local_path, bool overwrite) override;
    RequeueResult requeue_trained(const std::vector<std::string>& paths, bool all,
                                  const std::string& model_name = "") override;
    int renew_claims(const std::string& run_id) override;  // TD-494

   private:
    std::string host_;
    int port_;
    std::string group_prefix_;  ///< "/registry/<run_group>" or "/registry/<run_group>/<kind>" (TD-202)
    int timeout_ms_;
};
