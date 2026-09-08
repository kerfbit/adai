// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

#include "DaemonConfigStore.hpp"
#include <sqlite3.h>
#include <chrono>
#include <ctime>
#include <stdexcept>
#include "Logger.hpp"

namespace {

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

}  // namespace

namespace adai {

DaemonConfigStore::DaemonConfigStore(const std::string& db_path) {
    if (sqlite3_open(db_path.c_str(), &db_) != SQLITE_OK) {
        const std::string err = db_ ? sqlite3_errmsg(db_) : "unknown error";
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
        throw std::runtime_error("DaemonConfigStore: failed to open " + db_path + ": " + err);
    }

    // WAL mode: avoid blocking a long-running daemon's own reads on an admin write.
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

}  // namespace adai
