#pragma once

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
 * synchronized internally; callers are expected to serialize access (all three
 * daemons currently only touch this from the single request-handling context that
 * already guards config mutation).
 */
class DaemonConfigStore {
   public:
    /// Opens (creating if necessary) the daemon_config table at db_path.
    explicit DaemonConfigStore(const std::string& db_path);
    ~DaemonConfigStore();

    DaemonConfigStore(const DaemonConfigStore&) = delete;
    DaemonConfigStore& operator=(const DaemonConfigStore&) = delete;

    /// Returns every stored key/value pair.
    std::map<std::string, std::string> load_all() const;

    /// Upserts a single key/value pair, stamping updated_utc.
    void set(const std::string& key, const std::string& value);

   private:
    sqlite3* db_ = nullptr;
};

}  // namespace adai
