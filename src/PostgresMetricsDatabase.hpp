#pragma once

#ifdef ADAI_ENABLE_POSTGRES

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "MetricsDatabase.hpp"

struct pg_conn;
typedef struct pg_conn PGconn;

class PostgresMetricsDatabase : public IMetricsDatabase {
   public:
    PostgresMetricsDatabase(const std::string& connection_url, int pool_size = 4);
    ~PostgresMetricsDatabase() override;

    PostgresMetricsDatabase(const PostgresMetricsDatabase&) = delete;
    PostgresMetricsDatabase& operator=(const PostgresMetricsDatabase&) = delete;

    void upsert_session(const SessionRecord& rec) override;
    void mark_session_ended(const std::string& key) override;
    void archive_session(const std::string& key, const std::string& archived_key) override;

    void insert_metrics_record(const std::string& session_key,
                               const PersistentMetricsRecord& rec) override;

    void insert_gradient_variance_sample(const std::string& session_key, int step, int epoch,
                                         float value) override;

    void insert_abnormal_sample(const std::string& session_key,
                                const AbnormalSample& sample) override;

    void insert_generation_quality(const std::string& session_key, int epoch,
                                   const GenerationQualityScore& score) override;

    std::vector<PersistentMetricsRecord> query_history(
        const std::string& session_key, std::optional<std::chrono::system_clock::time_point> from,
        std::optional<std::chrono::system_clock::time_point> to, int limit) override;

    std::vector<SessionRecord> list_sessions(std::optional<bool> is_training_filter) override;

    std::optional<SessionRecord> get_session(const std::string& key) override;

   private:
    struct PooledConnection {
        PGconn* conn = nullptr;
        bool in_use = false;
    };

    PGconn* acquire_connection();
    void release_connection(PGconn* conn);
    PGconn* create_connection();
    bool ensure_connection(PGconn*& conn);
    void bootstrap_schema(PGconn* conn);
    static std::string format_timestamp(const std::chrono::system_clock::time_point& tp);
    static std::chrono::system_clock::time_point parse_timestamp(const std::string& s);

    bool execute_with_retry(const char* context, std::function<bool(PGconn*)> operation);

    std::string connection_url_;
    int pool_size_;

    std::mutex pool_mutex_;
    std::condition_variable pool_cv_;
    std::vector<PooledConnection> pool_;
};

std::unique_ptr<IMetricsDatabase> create_postgres_metrics_database(const std::string& url,
                                                                   int pool_size);

#endif  // ADAI_ENABLE_POSTGRES
