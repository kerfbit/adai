// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10

#include "SessionCounterStore.hpp"
#include <sqlite3.h>
#include <chrono>
#include <climits>
#include <ctime>
#include <stdexcept>
#include "Logger.hpp"
#include "PortableTime.hpp"

namespace {

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    adai::gmtime_utc(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

}  // namespace

namespace adai {

SessionCounterStore::SessionCounterStore(const std::string& db_path, int busy_timeout_ms) {
    if (sqlite3_open(db_path.c_str(), &db_) != SQLITE_OK) {
        const std::string err = db_ ? sqlite3_errmsg(db_) : "unknown error";
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
        throw std::runtime_error("SessionCounterStore: failed to open " + db_path + ": " + err);
    }
    sqlite3_busy_timeout(db_, busy_timeout_ms);
    sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);

    const auto fail = [this](const std::string& what) {
        const std::string err = "SessionCounterStore: " + what + ": " + sqlite3_errmsg(db_);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error(err);
    };

    // One write transaction, so two processes opening a pre-TD-308 file at once move its counters
    // exactly once.
    if (sqlite3_exec(db_, "BEGIN IMMEDIATE;", nullptr, nullptr, nullptr) != SQLITE_OK) fail("begin failed");
    static const char* kSchema =
        "CREATE TABLE IF NOT EXISTS session_counters ("
        "  key TEXT PRIMARY KEY,"
        "  value INTEGER NOT NULL,"
        "  updated_utc TEXT NOT NULL"
        ");";
    if (sqlite3_exec(db_, kSchema, nullptr, nullptr, nullptr) != SQLITE_OK) fail("failed to create schema");

    bool legacy = false;
    sqlite3_exec(
        db_, "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'daemon_config';",
        [](void* out, int, char**, char**) {
            *static_cast<bool*>(out) = true;
            return 0;
        },
        &legacy, nullptr);
    if (legacy) {
        // A key already in session_counters (written by a new binary while an old one still used the
        // file) keeps the larger value, so no number is handed out twice.
        static const char* kMove =
            "INSERT INTO session_counters (key, value, updated_utc) "
            "SELECT key, CAST(value AS INTEGER), updated_utc FROM daemon_config WHERE true "
            "ON CONFLICT(key) DO UPDATE SET value = max(session_counters.value, excluded.value);";
        if (sqlite3_exec(db_, kMove, nullptr, nullptr, nullptr) != SQLITE_OK) fail("failed to move legacy counters");
        migrated_ = sqlite3_changes(db_);
        if (sqlite3_exec(db_, "DROP TABLE daemon_config;", nullptr, nullptr, nullptr) != SQLITE_OK) {
            fail("failed to drop the legacy daemon_config table");
        }
    }
    if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) fail("commit failed");
    if (legacy) {
        Logger::info("SessionCounterStore: moved {} counter(s) in {} from daemon_config to session_counters",
                     migrated_, db_path);
    }
}

SessionCounterStore::~SessionCounterStore() {
    if (db_) {
        sqlite3_close(db_);
    }
}

int SessionCounterStore::next(const std::string& key) {
    if (sqlite3_exec(db_, "BEGIN IMMEDIATE;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error(std::string("SessionCounterStore::next: begin failed: ") + sqlite3_errmsg(db_));
    }
    const auto fail = [this](sqlite3_stmt* stmt, const char* what) {
        const std::string err = std::string("SessionCounterStore::next: ") + what + ": " + sqlite3_errmsg(db_);
        sqlite3_finalize(stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw std::runtime_error(err);
    };

    // One upsert reads and writes the one key (no RETURNING: it needs SQLite 3.35).
    static const char* kUpsert =
        "INSERT INTO session_counters (key, value, updated_utc) VALUES (?, 1, ?) "
        "ON CONFLICT(key) DO UPDATE SET "
        "  value = CASE WHEN value < 0 OR value >= 2147483647 THEN 1 ELSE value + 1 END,"
        "  updated_utc = excluded.updated_utc;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, kUpsert, -1, &stmt, nullptr) != SQLITE_OK) fail(stmt, "prepare failed");
    const std::string now = utc_now();
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, now.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) != SQLITE_DONE) fail(stmt, "write failed");
    sqlite3_finalize(stmt);
    stmt = nullptr;

    if (sqlite3_prepare_v2(db_, "SELECT value FROM session_counters WHERE key = ?;", -1, &stmt, nullptr) !=
        SQLITE_OK)
        fail(stmt, "prepare failed");
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) != SQLITE_ROW) fail(stmt, "read failed");
    const int value = static_cast<int>(sqlite3_column_int64(stmt, 0));
    sqlite3_finalize(stmt);
    if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) fail(nullptr, "commit failed");
    return value;
}

int SessionCounterStore::current(const std::string& key) const {
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT value FROM session_counters WHERE key = ?;", -1, &stmt, nullptr) !=
        SQLITE_OK) {
        throw std::runtime_error(std::string("SessionCounterStore::current: prepare failed: ") + sqlite3_errmsg(db_));
    }
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    const int rc = sqlite3_step(stmt);
    int value = 0;
    if (rc == SQLITE_ROW) {
        value = static_cast<int>(sqlite3_column_int64(stmt, 0));
    } else if (rc != SQLITE_DONE) {
        const std::string err = std::string("SessionCounterStore::current: read failed: ") + sqlite3_errmsg(db_);
        sqlite3_finalize(stmt);
        throw std::runtime_error(err);
    }
    sqlite3_finalize(stmt);
    return value;
}

}  // namespace adai
