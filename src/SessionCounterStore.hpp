#pragma once

// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10


#include <string>

// Forward-declared to keep <sqlite3.h> out of consumers' headers, as DaemonConfigStore does.
struct sqlite3;

namespace adai {

/**
 * @brief TD-308: SQLite-backed per-key counters, used for run/session numbering.
 *
 * `registry_server` (`POST /registry/{group}/session/next`) and `LocalTransport::next_session()`
 * keep one counter per "model_name|run_id" in `session_counters.db`. They used to do it through
 * DaemonConfigStore, so the counters sat in a table named `daemon_config` next to the admin
 * settings' API. The counters now live in their own `session_counters` table with an integer
 * value; a file written before TD-308 has its `daemon_config` rows moved over when it's opened.
 *
 * Safe to share between processes: next() runs in one write transaction, and a write waits up to
 * busy_timeout_ms for another connection's lock (TD-304). Not synchronized within a process; callers
 * serialize their own use of one instance.
 */
class SessionCounterStore {
   public:
    static constexpr int kDefaultBusyTimeoutMs = 5000;

    /// Opens (creating if necessary) the session_counters table at db_path, moving any pre-TD-308
    /// counters out of a `daemon_config` table in the same file. Throws std::runtime_error on failure.
    explicit SessionCounterStore(const std::string& db_path, int busy_timeout_ms = kDefaultBusyTimeoutMs);
    ~SessionCounterStore();

    SessionCounterStore(const SessionCounterStore&) = delete;
    SessionCounterStore& operator=(const SessionCounterStore&) = delete;

    /// Adds 1 to the counter at key (a missing counter counts as 0, so the first call returns 1),
    /// stores it and returns it, as one transaction, so concurrent callers in any process each get a
    /// different number. A counter at INT_MAX (or below 0) starts again at 1. Throws
    /// std::runtime_error on failure, storing nothing.
    int next(const std::string& key);

    /// The counter's current value, 0 when it has never been incremented. Throws on a read error.
    int current(const std::string& key) const;

    /// How many pre-TD-308 counters this open moved out of `daemon_config` (0 for a new-style file).
    int migrated_count() const { return migrated_; }

   private:
    sqlite3* db_ = nullptr;
    int migrated_ = 0;
};

}  // namespace adai
