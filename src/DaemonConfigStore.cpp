// @adai-status: stable
// @adai-version: 1.2.0
// @adai-reviewed: 2026-11-03

#include "DaemonConfigStore.hpp"
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

DaemonConfigStore::DaemonConfigStore(const std::string& db_path, int busy_timeout_ms) {
    if (sqlite3_open(db_path.c_str(), &db_) != SQLITE_OK) {
        const std::string err = db_ ? sqlite3_errmsg(db_) : "unknown error";
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
        throw std::runtime_error("DaemonConfigStore: failed to open " + db_path + ": " + err);
    }

    // TD-304: wait for another connection's write lock instead of failing at once with
    // "database is locked" (trainer_service and its pass share one daemon_config.db, and
    // standalone trainers share session_counters.db).
    sqlite3_busy_timeout(db_, busy_timeout_ms);

    // WAL mode: avoid blocking a long-running daemon's own reads on an admin write.
    // TODO: See TD-307 in TECHNICAL_DEBT.md - result ignored; without WAL, readers can be blocked
    sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);

    static const char* kSchema =
        "CREATE TABLE IF NOT EXISTS daemon_config ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT NOT NULL,"
        "  updated_utc TEXT NOT NULL"
        ");";
    char* err_msg = nullptr;
    if (sqlite3_exec(db_, kSchema, nullptr, nullptr, &err_msg) != SQLITE_OK) {
        const std::string err = err_msg ? err_msg : "unknown error";
        sqlite3_free(err_msg);
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("DaemonConfigStore: failed to create schema: " + err);
    }
}

DaemonConfigStore::~DaemonConfigStore() {
    if (db_) {
        sqlite3_close(db_);
    }
}

std::map<std::string, std::string> DaemonConfigStore::load_all() const {
    std::map<std::string, std::string> out;
    sqlite3_stmt* stmt = nullptr;
    static const char* kQuery = "SELECT key, value FROM daemon_config;";
    if (sqlite3_prepare_v2(db_, kQuery, -1, &stmt, nullptr) != SQLITE_OK) {
        Logger::warn("DaemonConfigStore::load_all: prepare failed: {}", sqlite3_errmsg(db_));
        return out;
    }
    // TODO: See TD-307 in TECHNICAL_DEBT.md - a step error ends the loop silently (partial map, no warning)
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const auto* key = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const auto* value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        if (key && value) {
            out[key] = value;
        }
    }
    sqlite3_finalize(stmt);
    return out;
}

void DaemonConfigStore::set(const std::string& key, const std::string& value) {
    static const char* kUpsert =
        "INSERT INTO daemon_config (key, value, updated_utc) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_utc = excluded.updated_utc;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, kUpsert, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error(std::string("DaemonConfigStore::set: prepare failed: ") +
                                 sqlite3_errmsg(db_));
    }
    const std::string now = utc_now();
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, value.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, now.c_str(), -1, SQLITE_TRANSIENT);
    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("DaemonConfigStore::set: step failed: ") +
                                 sqlite3_errmsg(db_));
    }
}

void DaemonConfigStore::set_many(const std::map<std::string, std::string>& values) {
    if (values.empty()) return;
    const auto fail = [this](const char* what) {
        const std::string err = std::string("DaemonConfigStore::set_many: ") + what + ": " + sqlite3_errmsg(db_);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw std::runtime_error(err);
    };
    if (sqlite3_exec(db_, "BEGIN IMMEDIATE;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error(std::string("DaemonConfigStore::set_many: begin failed: ") + sqlite3_errmsg(db_));
    }
    static const char* kUpsert =
        "INSERT INTO daemon_config (key, value, updated_utc) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_utc = excluded.updated_utc;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, kUpsert, -1, &stmt, nullptr) != SQLITE_OK) fail("prepare failed");
    const std::string now = utc_now();
    for (const auto& [key, value] : values) {
        sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, value.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, now.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) != SQLITE_DONE) {
            const std::string err = std::string("DaemonConfigStore::set_many: step failed: ") + sqlite3_errmsg(db_);
            sqlite3_finalize(stmt);
            sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
            throw std::runtime_error(err);
        }
    }
    sqlite3_finalize(stmt);
    if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) fail("commit failed");
}

int DaemonConfigStore::increment(const std::string& key) {
    if (sqlite3_exec(db_, "BEGIN IMMEDIATE;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error(std::string("DaemonConfigStore::increment: begin failed: ") + sqlite3_errmsg(db_));
    }
    const auto fail = [this](sqlite3_stmt* stmt, const char* what) {
        const std::string err = std::string("DaemonConfigStore::increment: ") + what + ": " + sqlite3_errmsg(db_);
        sqlite3_finalize(stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw std::runtime_error(err);
    };

    long long current = 0;
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT CAST(value AS INTEGER) FROM daemon_config WHERE key = ?;", -1, &stmt,
                           nullptr) != SQLITE_OK)
        fail(stmt, "prepare failed");
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    const int rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        current = sqlite3_column_int64(stmt, 0);
    } else if (rc != SQLITE_DONE) {
        fail(stmt, "read failed");
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
    if (current < 0 || current >= INT_MAX) current = 0;
    const int next = static_cast<int>(current) + 1;

    static const char* kUpsert =
        "INSERT INTO daemon_config (key, value, updated_utc) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_utc = excluded.updated_utc;";
    if (sqlite3_prepare_v2(db_, kUpsert, -1, &stmt, nullptr) != SQLITE_OK) fail(stmt, "prepare failed");
    const std::string value = std::to_string(next);
    const std::string now = utc_now();
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, value.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, now.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) != SQLITE_DONE) fail(stmt, "write failed");
    sqlite3_finalize(stmt);
    if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK) fail(nullptr, "commit failed");
    return next;
}

}  // namespace adai
