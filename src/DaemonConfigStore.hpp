#pragma once

// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-11-03


#include <map>
#include <string>

// Forward-declare rather than pull in <sqlite3.h> here — mirrors the pimpl-ish
// pattern ModelNameService uses to keep sqlite3.h out of consumers' headers.
struct sqlite3;

namespace adai {

/**
 * @brief Small generic SQLite-backed key-value store for daemon runtime configuration.
 *
 * Used identically by metrics_api_server, mns_server, and registry_server to persist
 * settings changed at runtime via each daemon's `PUT /admin/config` endpoint. Each
 * daemon opens its own dedicated `daemon_config.db` file, separate from any other
 * SQLite database it owns (e.g. mns_server's `models.db`, metrics_api_server's
 * `metrics.db`) — this keeps DaemonConfigStore fully decoupled from those components'
 * schemas/locking, and avoids a load-order paradox where a daemon would need to open
 * a database to find out which path to open it at (db_path/data_dir/port are
 * therefore never stored here — see CLAUDE.md "Daemon admin config API").
 *
 * On startup, a daemon should load its config.<service>.conf file first (the seed/
 * bootstrap values), then overlay load_all() on top — DB values represent live
 * admin-applied state and take precedence over the file. Thread safety: not
 * synchronized internally; callers serialize access. Each daemon's PUT /admin/config
 * holds its own admin mutex while it saves with set_many() and then applies the
 * change in memory (TD-305), so a failed save changes nothing and two PUTs can't
 * interleave.
 */
class DaemonConfigStore {
   public:
    /// TD-304: how long a write waits for another connection's write lock before failing.
    static constexpr int kDefaultBusyTimeoutMs = 5000;

    /// Opens (creating if necessary) the daemon_config table at db_path. A write that finds
    /// another connection writing waits up to busy_timeout_ms (TD-304; it used to fail at once).
    explicit DaemonConfigStore(const std::string& db_path, int busy_timeout_ms = kDefaultBusyTimeoutMs);
    ~DaemonConfigStore();

    DaemonConfigStore(const DaemonConfigStore&) = delete;
    DaemonConfigStore& operator=(const DaemonConfigStore&) = delete;

    /// Returns every stored key/value pair. Throws std::runtime_error if the table can't be read in
    /// full (TD-307: a read error used to end the scan silently, returning only the rows before it,
    /// so a daemon's startup overlay could drop saved settings without a word). Callers fall back
    /// to their file/CLI settings with a warning.
    std::map<std::string, std::string> load_all() const;

    /// TD-307: the journal mode SQLite actually set ("wal" normally). Anything else is logged at
    /// open, since without WAL a reader can be blocked by an admin write.
    const std::string& journal_mode() const { return journal_mode_; }

    /// Upserts a single key/value pair, stamping updated_utc.
    void set(const std::string& key, const std::string& value);

    /// TD-305: upserts every pair in one transaction, stamping updated_utc. Either all of them
    /// are stored or, when anything fails (which throws std::runtime_error), none are.
    void set_many(const std::map<std::string, std::string>& values);
    // TD-308: increment() (TD-304) moved to SessionCounterStore with the session counters.

   private:
    sqlite3* db_ = nullptr;
    std::string journal_mode_;
};

}  // namespace adai
