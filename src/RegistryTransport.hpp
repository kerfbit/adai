#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include "DaemonConfigStore.hpp"

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
};

/**
 * @brief A file in the pending training queue.
 *
 * run_id is set by RemoteTransport::acquire_pending() (Phase 9) for atomic
 * ownership tracking.  LocalTransport always leaves it empty.
 */
struct PendingEntry {
    std::string path;
    std::string run_id;      // empty for LocalTransport
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
};

// ============================================================================
// FTP transport data types (Phase 10: dataset transport)
// ============================================================================

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
};

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

    /// Convenience: collect registry_paths for release/mark_trained calls.
    std::vector<std::string> registry_paths() const {
        std::vector<std::string> out;
        out.reserve(files.size());
        for (const auto& f : files)
            out.push_back(f.registry_path);
        return out;
    }
};

// ============================================================================
// Phase 16: dataset management result types (assign-by-count, unassign, delete)
// ============================================================================

/** @brief Result of RegistryTransport::assign(). */
struct AssignResult {
    int assigned = 0;
    std::vector<std::string> paths;  ///< exact paths touched, in every mode
};

/** @brief Result of RegistryTransport::unassign(). */
struct UnassignResult {
    int unassigned = 0;
    /// Entries that matched but were left untouched because they're actively
    /// claimed by a run (non-empty run_id) and force was not set.
    int skipped = 0;
    std::vector<std::string> paths;  ///< exact paths touched
};

/** @brief Result of RegistryTransport::delete_paths(). */
struct DeleteResult {
    struct Detail {
        std::string path;
        std::string status;       ///< "deleted" | "skipped_active_run" | "not_found"
        bool file_deleted = false;  ///< true only if delete_files was requested and it worked
    };
    int deleted = 0;
    int skipped = 0;
    int not_found = 0;
    std::vector<Detail> details;
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

    /** @brief Release @p paths assigned to @p run_id back to the unassigned pool. */
    virtual void release(const std::string& run_id, const std::vector<std::string>& paths) = 0;

    /**
     * @brief Atomically append @p new_entries to the registry and remove
     *        @p trained_paths from the pending queue (for @p run_id).
     *
     * LocalTransport writes both files under an advisory lock.
     * RemoteTransport issues a single POST /trained request.
     * Pass an empty @p run_id to release any pending entry regardless of
     * assignment (single-run mode).
     */
    virtual void commit_trained(const std::string& run_id,
                                const std::vector<DataVersion>& new_entries,
                                const std::vector<std::string>& trained_paths) = 0;

    /** @brief Atomically append a single @p path to the pending queue.
     *  No-op (returns true) if @p path is already present.
     *  @return true on success. */
    virtual bool add_pending(const std::string& path) = 0;

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
     * @brief Set model_name on pending entries. Three modes, checked in order:
     *          - non-empty @p paths        — assign exactly those (count ignored)
     *          - empty @p paths, count > 0 — assign the first @p count
     *                                        currently-unassigned entries
     *          - empty @p paths, count<=0  — assign every pending entry
     */
    virtual AssignResult assign(const std::string& model_name,
                                const std::vector<std::string>& paths = {}, int count = 0) = 0;

    /**
     * @brief Clear model_name back to unassigned. If @p paths is empty, clears
     *        every entry currently assigned to @p model_name (bulk mode,
     *        requires non-empty @p model_name). Otherwise clears only the
     *        listed paths; a non-empty @p model_name additionally acts as an
     *        ownership filter. An entry actively claimed by a run (non-empty
     *        run_id) is left untouched unless @p force is true.
     */
    virtual UnassignResult unassign(const std::string& model_name,
                                    const std::vector<std::string>& paths, bool force) = 0;

    /**
     * @brief Permanently purge entries matching @p paths from both the
     *        pending queue and the trained registry. @p paths must be
     *        non-empty — there is no bulk "delete everything" mode.
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
                                      bool delete_files) = 0;

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
     * @return Registry-relative path of the newly queued file, or "" on failure.
     */
    virtual std::string upload_file(const std::string& local_path) = 0;
};

// ============================================================================
// Local flat-file transport (Phase 8)
// ============================================================================

/**
 * @brief LocalTransport implements the same flat-file format previously
 *        inlined in DatasetRegistry::load/save_registry() and
 *        load/save_pending_list().  Zero network dependency.
 *
 * File formats
 * ------------
 * registry_path: one entry per line, format: <data_file> <checksum> <num_samples> <trained>
 *                Lines starting with '#' are comments and are skipped on load.
 * pending_path:  one file path per line; run_id is not persisted (always empty on load).
 */
/**
 * @brief LocalTransport implements the same flat-file format previously
 *        inlined in DatasetRegistry::load/save_registry() and
 *        load/save_pending_list().  Zero network dependency.
 *
 * Pending-file format (Phase 9 extension)
 * ----------------------------------------
 * Each line is either:
 *   /path/to/file                 — unassigned (run_id empty)
 *   /path/to/file\trun-id         — assigned to run-id (tab separator)
 *
 * Lines written by Phase 8 (no tab) are loaded as unassigned; backward
 * compatible.
 *
 * acquire() and release() hold an advisory lock file (pending_path + ".lock")
 * via flock(2) for the duration of the read-modify-write cycle, serialising
 * concurrent callers on the same host.
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
    void release(const std::string& run_id, const std::vector<std::string>& paths) override;
    void commit_trained(const std::string& run_id, const std::vector<DataVersion>& new_entries,
                        const std::vector<std::string>& trained_paths) override;
    bool add_pending(const std::string& path) override;
    std::string next_session(const std::string& model_name, const std::string& run_id) override;

    AssignResult assign(const std::string& model_name, const std::vector<std::string>& paths,
                        int count) override;
    UnassignResult unassign(const std::string& model_name, const std::vector<std::string>& paths,
                            bool force) override;
    DeleteResult delete_paths(const std::vector<std::string>& paths, bool force,
                              bool delete_files) override;

    // Phase 11: not supported in local mode — logs a warning and returns "".
    std::string fetch_gutenberg(int book_id, int num_pairs,
                                const std::string& model_name) override;
    std::string fetch_huggingface(const std::string& dataset_id, int num_pairs,
                                  const std::string& split, const std::string& input_field,
                                  const std::string& output_field,
                                  const std::string& model_name) override;
    std::string upload_file(const std::string& local_path) override;

   private:
    std::string registry_path_;
    std::string pending_path_;

    // Session-number counter (Phase 3), lazily opened next to pending_path_ as
    // "session_counters.db" on first next_session() call. Kept as a raw
    // pointer behind a small helper rather than <memory> to avoid pulling
    // DaemonConfigStore's sqlite3 forward-declare into every RegistryTransport
    // consumer; see RegistryTransport.cpp for the definition.
    std::unique_ptr<adai::DaemonConfigStore> session_store_;

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
     * @param base_url   Scheme + host + optional port, e.g. "http://reg:8082"
     * @param run_group  Logical namespace for this project.
     * @param timeout_ms HTTP connect/read timeout in milliseconds.
     */
    RemoteTransport(std::string base_url, std::string run_group, int timeout_ms = 5000);

    bool load_registry(std::vector<DataVersion>& out) override;
    bool save_registry(const std::vector<DataVersion>& entries) override;
    bool load_pending(std::vector<PendingEntry>& out) override;
    bool save_pending(const std::vector<PendingEntry>& entries) override;
    AcquireResponse acquire(const std::string& run_id, int max_files,
                            const std::string& model_name = "") override;
    void release(const std::string& run_id, const std::vector<std::string>& paths) override;
    void commit_trained(const std::string& run_id, const std::vector<DataVersion>& new_entries,
                        const std::vector<std::string>& trained_paths) override;
    bool add_pending(const std::string& path) override;
    std::string next_session(const std::string& model_name, const std::string& run_id) override;

    AssignResult assign(const std::string& model_name, const std::vector<std::string>& paths,
                        int count) override;
    UnassignResult unassign(const std::string& model_name, const std::vector<std::string>& paths,
                            bool force) override;
    DeleteResult delete_paths(const std::vector<std::string>& paths, bool force,
                              bool delete_files) override;

    // Phase 11: delegate to the registry_server, which performs the fetch/
    // upload itself and stores the result under its own data_dir.
    std::string fetch_gutenberg(int book_id, int num_pairs,
                                const std::string& model_name) override;
    std::string fetch_huggingface(const std::string& dataset_id, int num_pairs,
                                  const std::string& split, const std::string& input_field,
                                  const std::string& output_field,
                                  const std::string& model_name) override;
    std::string upload_file(const std::string& local_path) override;

   private:
    std::string host_;
    int port_;
    std::string group_prefix_;  ///< "/registry/<run_group>"
    int timeout_ms_;
};
