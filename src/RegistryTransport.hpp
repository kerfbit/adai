#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

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
    int         num_samples = 0;
    bool        trained     = false;
    std::string model_id;  ///< Phase 2: MNS model UUID; empty for pre-MNS records
    std::chrono::system_clock::time_point added_time;
};

/**
 * @brief A file in the pending training queue.
 *
 * run_id is set by RemoteTransport::acquire_pending() (Phase 9) for atomic
 * ownership tracking.  LocalTransport always leaves it empty.
 */
struct PendingEntry {
    std::string path;
    std::string run_id;  // empty for LocalTransport
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
    virtual bool load_registry(std::vector<DataVersion>& out)           = 0;

    /** @brief Persist @p entries to the backing store.
     *  @return true on success. */
    virtual bool save_registry(const std::vector<DataVersion>& entries) = 0;

    /** @brief Load all pending-queue entries into @p out.  Clears @p out first.
     *  @return true if the backing store was found and parsed. */
    virtual bool load_pending(std::vector<PendingEntry>& out)           = 0;

    /** @brief Persist @p entries to the backing store.
     *  @return true on success. */
    virtual bool save_pending(const std::vector<PendingEntry>& entries) = 0;

    // ── Phase 9: distributed queue operations ─────────────────────────────

    /**
     * @brief Atomically acquire up to @p max_files unassigned entries for @p run_id.
     *
     * LocalTransport uses an advisory lock file.  RemoteTransport uses a
     * single atomic POST /acquire request.  Returns acquired paths; empty
     * when none are available.
     */
    virtual std::vector<std::string> acquire(const std::string& run_id, int max_files) = 0;

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

    bool load_registry(std::vector<DataVersion>& out)                         override;
    bool save_registry(const std::vector<DataVersion>& entries)               override;
    bool load_pending(std::vector<PendingEntry>& out)                         override;
    bool save_pending(const std::vector<PendingEntry>& entries)               override;
    std::vector<std::string> acquire(const std::string& run_id, int max_files) override;
    void release(const std::string& run_id, const std::vector<std::string>& paths) override;
    void commit_trained(const std::string& run_id,
                        const std::vector<DataVersion>& new_entries,
                        const std::vector<std::string>& trained_paths)        override;
    bool add_pending(const std::string& path)                                 override;

private:
    std::string registry_path_;
    std::string pending_path_;

    // Hold an exclusive flock on pending_path_ + ".lock" for the duration of
    // an acquire/release cycle and return the fd, or -1 on failure.
    int  lock_pending() const;
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

    bool load_registry(std::vector<DataVersion>& out)                         override;
    bool save_registry(const std::vector<DataVersion>& entries)               override;
    bool load_pending(std::vector<PendingEntry>& out)                         override;
    bool save_pending(const std::vector<PendingEntry>& entries)               override;
    std::vector<std::string> acquire(const std::string& run_id, int max_files) override;
    void release(const std::string& run_id, const std::vector<std::string>& paths) override;
    void commit_trained(const std::string& run_id,
                        const std::vector<DataVersion>& new_entries,
                        const std::vector<std::string>& trained_paths)        override;
    bool add_pending(const std::string& path)                                 override;

private:
    std::string host_;
    int         port_;
    std::string group_prefix_;  ///< "/registry/<run_group>"
    int         timeout_ms_;
};
