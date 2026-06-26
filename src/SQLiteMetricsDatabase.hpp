#pragma once

#include "MetricsDatabase.hpp"
#include <mutex>
#include <string>

struct sqlite3;
struct sqlite3_stmt;

class SQLiteMetricsDatabase : public IMetricsDatabase {
public:
    explicit SQLiteMetricsDatabase(const std::string& db_path);
    ~SQLiteMetricsDatabase() override;

    SQLiteMetricsDatabase(const SQLiteMetricsDatabase&) = delete;
    SQLiteMetricsDatabase& operator=(const SQLiteMetricsDatabase&) = delete;

    void upsert_session(const SessionRecord& rec) override;
    void mark_session_ended(const std::string& key) override;

    void insert_metrics_record(
        const std::string& session_key,
        const PersistentMetricsRecord& rec) override;

    void insert_abnormal_sample(
        const std::string& session_key,
        const AbnormalSample& sample) override;

    void insert_generation_quality(
        const std::string& session_key,
        int epoch,
        const GenerationQualityScore& score) override;

    std::vector<PersistentMetricsRecord> query_history(
        const std::string& session_key,
        std::optional<std::chrono::system_clock::time_point> from,
        std::optional<std::chrono::system_clock::time_point> to,
        int limit) override;

    std::vector<SessionRecord> list_sessions(
        std::optional<bool> is_training_filter) override;

    std::optional<SessionRecord> get_session(const std::string& key) override;

private:
    void bootstrap_schema();
    void prepare_statements();
    void finalize_statements();
    static std::string format_timestamp(const std::chrono::system_clock::time_point& tp);
    static std::chrono::system_clock::time_point parse_timestamp(const std::string& s);

    std::mutex mutex_;
    sqlite3* db_ = nullptr;

    sqlite3_stmt* stmt_upsert_session_       = nullptr;
    sqlite3_stmt* stmt_mark_ended_           = nullptr;
    sqlite3_stmt* stmt_insert_metrics_       = nullptr;
    sqlite3_stmt* stmt_insert_abnormal_      = nullptr;
    sqlite3_stmt* stmt_insert_gen_quality_   = nullptr;
    sqlite3_stmt* stmt_get_session_          = nullptr;
};
