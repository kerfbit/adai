#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "ChatbotTrainer.hpp"     // ConversationPair
#include "Config.hpp"             // adai::ServiceConfig
#include "RegistryTransport.hpp"  // DataVersion, PendingEntry, RegistryTransport, LocalTransport

/**
 * @brief Configuration for DatasetRegistry.
 *
 * Fields that were previously embedded in IncrementalConfig are collected
 * here so that data-management operations can be performed without
 * constructing a model or tokenizer.
 */
struct DatasetConfig {
    // ── Local state ────────────────────────────────────────────────────────
    std::string session_dir          = "training_sessions";
    std::string data_registry_file   = "data_registry.txt";
    bool        cache_tokenized_data = false;
    std::string tokenized_cache_dir  = "tokenized_cache";

    // ── Remote registry (TD-028 Phase 9; empty = local flat-file mode) ────
    /// URL of the registry_server daemon (e.g. "http://registry-host:8081")
    std::string registry_server_url;
    /// Logical namespace for multi-project server sharing; defaults to session_dir basename
    std::string run_group;
    /// Per-process run identifier; auto-derived from hostname+PID when empty
    std::string run_id;
    /// HTTP timeout in milliseconds for registry_server calls (default: 5000)
    int         registry_timeout_ms = 5000;
    /// Maximum pending files to acquire per run; 0 = claim all available (default: 0)
    int         max_files_per_run   = 0;

    // ── FTP dataset transport (Phase 10) ──────────────────────────────────
    /// Local directory for FTP downloads; created at startup if absent.
    std::string download_dir;
    /// Maximum concurrent FTP connections for fetch_all() (default: 4)
    int         max_parallel_downloads = 4;
    /// Log a warning for any file whose size exceeds this threshold in MB; 0 = disabled
    int         large_file_warn_threshold_mb = 500;
};

/**
 * @brief Manages the data registry (trained files) and the pending queue.
 *
 * Owns all local data state: the registry on disk, the pending queue,
 * checksums, and training-file parsing.  Has zero network dependency —
 * it only reads/writes local flat files.
 *
 * Intended usage pattern:
 * @code
 *   DatasetRegistry reg(DatasetRegistry::make_config(svc));
 *   reg.load_registry();
 *   reg.load_pending_list();
 *   reg.add_file("path/to/training.txt");
 *   // ... after a successful training run:
 *   reg.mark_trained(trained_files, sample_counts);
 * @endcode
 *
 * See docs/proposals/dataset_manager_separation.md (TD-028) for the full
 * design including the distributed-mode extensions planned for Phase 9.
 */
class DatasetRegistry {
public:
    explicit DatasetRegistry(DatasetConfig cfg = {});

    /**
     * @brief Constructor with explicit transport — used in tests and Phase 9
     *        to inject a RemoteTransport in place of LocalTransport.
     */
    DatasetRegistry(DatasetConfig cfg, std::unique_ptr<RegistryTransport> transport);

    /**
     * @brief Build a DatasetConfig from a parsed ServiceConfig.
     *
     * Translates the relevant session/data fields.  Call this when you
     * already have a ServiceConfig and need a DatasetConfig.
     */
    static DatasetConfig make_config(const adai::ServiceConfig& svc);

    // ── Pending queue ──────────────────────────────────────────────────────

    /**
     * @brief Add a single local file to the pending training queue.
     *
     * Skips files that do not exist on disk or that are already present
     * in the trained-files set.  Persists the updated queue immediately
     * via save_pending_list().
     *
     * @return true if the file was added; false if skipped or missing.
     */
    bool add_file(const std::string& path);

    /**
     * @brief Add multiple files to the pending queue.
     *
     * Calls add_file() for each path and returns true if at least one
     * file was successfully added.
     */
    bool add_files(const std::vector<std::string>& paths);

    /** @brief Discard the in-memory pending queue (does not write to disk). */
    void clear_pending();

    /**
     * @brief Remove a single file from the pending queue.
     *
     * Removes the entry from in-memory state and persists via save_pending_list().
     *
     * @return true if the file was found and removed.
     */
    bool remove_pending(const std::string& path);

    /**
     * @brief Assign pending files to a model by name.
     *
     * Sets the model_name field on matching pending entries and persists
     * via save_pending_list().  If @p paths is empty, assigns all pending.
     *
     * @return true if at least one entry was updated.
     */
    bool assign_model(const std::string& model_name,
                      const std::vector<std::string>& paths = {});

    /** @return Copy of the current in-memory pending-file paths. */
    std::vector<std::string> pending_files() const;

    /** @return Copy of the current in-memory pending entries (with model assignments). */
    std::vector<PendingEntry> pending_entries() const;

    /** @return Sorted vector of all file paths that have been trained. */
    std::vector<std::string> trained_files() const;

    /** @return true if @p path appears in the trained-files set. */
    bool is_trained(const std::string& path) const;

    // ── Mark trained ───────────────────────────────────────────────────────

    /**
     * @brief Record a set of files as successfully trained (single-run mode).
     *
     * Creates a DataVersion entry for each file that is not already in
     * the registry, then persists the updated registry via save_registry().
     *
     * @param paths         File paths that were trained.
     * @param sample_counts Number of training samples from each file
     *                      (parallel to @p paths; extra or missing counts
     *                      are treated as 0).
     */
    void mark_trained(const std::vector<std::string>& paths,
                      const std::vector<int>& sample_counts);

    // ── Multi-run API (TD-028 Phase 9) ─────────────────────────────────────

    /**
     * @brief Atomically claim pending files for @p run_id.
     *
     * In local mode an advisory file lock serialises concurrent callers on
     * the same host.  In remote mode the registry_server provides the
     * atomic guarantee via a single POST request.
     *
     * @param run_id    Unique identifier for this training process.
     * @param max_files Maximum files to claim; 0 claims all unassigned files.
     * @return AcquireResponse containing per-file tokens.  files is empty when
     *         none are available.  ftp_server_host is empty for local transport
     *         (caller reads files directly by registry_path).
     */
    AcquireResponse acquire_pending(const std::string& run_id, int max_files = 0);

    /**
     * @brief Return @p paths claimed by @p run_id back to the unassigned pool.
     *
     * Use on training failure or crash recovery.  Has no effect on files
     * that were not assigned to @p run_id.
     */
    void release_pending(const std::string& run_id, const std::vector<std::string>& paths);

    /**
     * @brief Mark trained files and release the run's reservation atomically.
     *
     * For RemoteTransport this is a single atomic POST /trained request.
     * For LocalTransport it updates the registry file and pending queue with
     * an advisory lock.
     *
     * Also updates the in-memory @c pending_ list.
     */
    void mark_trained(const std::string& run_id, const std::vector<std::string>& paths,
                      const std::vector<int>& sample_counts);

    /** @brief Print which files are currently assigned to which run_id. */
    void print_run_assignments();

    // ── Persistence ────────────────────────────────────────────────────────

    /**
     * @brief Load the data registry from disk into memory.
     * @return true if the file was found and parsed; false if absent or
     *         unreadable (registry_ is cleared on failure).
     */
    bool load_registry();

    /**
     * @brief Persist the in-memory registry to disk.
     * @return true on success; false if the file could not be opened.
     */
    bool save_registry();

    /**
     * @brief Load the pending-file list from disk into memory.
     * @return true if the file was found and parsed.
     */
    bool load_pending_list();

    /**
     * @brief Persist the in-memory pending-file list to disk.
     * @return true on success.
     */
    bool save_pending_list();

    // ── Reporting ──────────────────────────────────────────────────────────

    /** @brief Print the registry table to stdout (ANSI colour output). */
    void print_registry() const;

    /** @return Total number of trained samples across all registry entries. */
    int  total_samples_trained() const;

    // ── Static helpers ─────────────────────────────────────────────────────

    /**
     * @brief Parse a training file (JSONL or legacy INPUT:/RESPONSE:) into ConversationPairs.
     *
     * Format is auto-detected from the first non-empty line.  JSONL samples
     * carry optional SampleMeta (domain, task_type, quality, …).  Legacy files
     * are read unchanged for backward compatibility.
     *
     * Pure I/O: no network access, no model dependency.  Safe to call from
     * any thread.
     *
     * @param path  Path to the training file.
     * @param out   Output vector; pairs are appended (not replaced).
     * @return Number of pairs appended to @p out.
     */
    static int load_conversation_pairs(const std::string& path,
                                       std::vector<ConversationPair>& out);

    /**
     * @brief Compute a lightweight content fingerprint for @p path.
     *
     * Uses file size + last-write-time.  Returns "MISSING" if the file
     * does not exist.
     */
    static std::string compute_checksum(const std::string& path);

private:
    DatasetConfig                      config_;
    std::unique_ptr<RegistryTransport> transport_;  // Phase 8: injected I/O backend
    std::vector<DataVersion>           registry_;
    std::set<std::string>              trained_set_;
    std::vector<PendingEntry>           pending_;

    /** @return Full path to the registry flat file. */
    std::string registry_file_path() const;

    /** @return Full path to the pending-files flat file. */
    std::string pending_file_path() const;
};
