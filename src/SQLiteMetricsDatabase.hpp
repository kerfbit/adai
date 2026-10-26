#pragma once

// @adai-status: stable
// @adai-version: 1.2.0
// @adai-reviewed: 2026-10-25


#include <functional>
#include <mutex>
#include <string>
#include "MetricsDatabase.hpp"

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
    void archive_session(const std::string& key, const std::string& archived_key) override;

    void insert_metrics_record(const std::string& session_key,
                               const PersistentMetricsRecord& rec) override;

    void insert_gradient_variance_sample(const std::string& session_key, int step, int epoch,
                                         float value) override;

    void insert_metrics_records(const std::string& session_key,
                                const std::vector<PersistentMetricsRecord>& recs) override;
    void insert_gradient_variance_samples(const std::string& session_key,
                                          const std::vector<GradientVarianceRecord>& recs) override;

    void insert_abnormal_sample(const std::string& session_key,
                                const AbnormalSample& sample) override;

    void insert_generation_quality(const std::string& session_key, int epoch,
                                   const GenerationQualityScore& score) override;

    std::vector<PersistentMetricsRecord> query_history(
        const std::string& session_key, std::optional<std::chrono::system_clock::time_point> from,
        std::optional<std::chrono::system_clock::time_point> to, int limit) override;

    std::vector<SessionRecord> list_sessions(std::optional<bool> is_training_filter) override;

    std::optional<SessionRecord> get_session(const std::string& key) override;

    std::vector<GradientVarianceRecord> query_gradient_variance(const std::string& session_key,
                                                                int limit) override;
    std::vector<GenerationQualityRecord> query_generation_quality(
        const std::string& session_key) override;

    long long cap_session_rows(const std::string& session_key, int max_rows) override;
    long long delete_sessions_updated_before(
        std::chrono::system_clock::time_point cutoff) override;

   private:
    static constexpr int kBusyTimeoutMs = 5000;  // TD-511

    void bootstrap_schema();
    bool has_column(const char* table, const char* column);
    // Callers hold mutex_.
    void in_transaction(const std::function<void()>& body);
    void insert_metrics_record_locked(const std::string& session_key,
                                      const PersistentMetricsRecord& rec);
    void insert_gradient_variance_locked(const std::string& session_key,
                                         std::chrono::system_clock::time_point when, int step,
                                         int epoch, float value);
    void prepare_statements();
    void finalize_statements();
    static std::string format_timestamp(const std::chrono::system_clock::time_point& tp);
    static std::chrono::system_clock::time_point parse_timestamp(const std::string& s);

    std::mutex mutex_;
    sqlite3* db_ = nullptr;

    sqlite3_stmt* stmt_upsert_session_ = nullptr;
    sqlite3_stmt* stmt_mark_ended_ = nullptr;
    sqlite3_stmt* stmt_insert_metrics_ = nullptr;
    sqlite3_stmt* stmt_insert_gradient_variance_ = nullptr;
    sqlite3_stmt* stmt_insert_abnormal_ = nullptr;
    sqlite3_stmt* stmt_insert_gen_quality_ = nullptr;
    sqlite3_stmt* stmt_get_session_ = nullptr;
};
