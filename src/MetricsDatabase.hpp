#pragma once

#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct PersistentMetricsRecord;
struct AbnormalSample;
struct GenerationQualityScore;

struct SessionRecord {
    std::string key;
    int         session_id     = 0;
    std::string label;
    std::string config_json;
    bool        is_training    = false;
    std::chrono::system_clock::time_point created_at;
    std::optional<std::chrono::system_clock::time_point> ended_at;
    std::chrono::system_clock::time_point last_update_at;
    int   total_epochs  = 0;
    int   total_samples = 0;
    float best_validation_loss = std::numeric_limits<float>::max();
    int   best_epoch           = 0;
};

class IMetricsDatabase {
public:
    virtual ~IMetricsDatabase() = default;

    virtual void upsert_session(const SessionRecord& rec) = 0;
    virtual void mark_session_ended(const std::string& key) = 0;

    // Renames a session (and all of its metrics/samples/quality rows) from `key` to
    // `archived_key`, marking it not-training. Used when a live session goes stale/
    // completed and its key may be reused by a new session — archiving first prevents
    // the new session's history from being intermixed with the old one's under the
    // same key. No-op (not an error) if `key` does not currently exist.
    virtual void archive_session(const std::string& key, const std::string& archived_key) = 0;

    virtual void insert_metrics_record(
        const std::string& session_key,
        const PersistentMetricsRecord& rec) = 0;

    virtual void insert_abnormal_sample(
        const std::string& session_key,
        const AbnormalSample& sample) = 0;

    virtual void insert_generation_quality(
        const std::string& session_key,
        int epoch,
        const GenerationQualityScore& score) = 0;

    virtual std::vector<PersistentMetricsRecord> query_history(
        const std::string& session_key,
        std::optional<std::chrono::system_clock::time_point> from = std::nullopt,
        std::optional<std::chrono::system_clock::time_point> to   = std::nullopt,
        int limit = 0) = 0;

    virtual std::vector<SessionRecord> list_sessions(
        std::optional<bool> is_training_filter = std::nullopt) = 0;

    virtual std::optional<SessionRecord> get_session(
        const std::string& key) = 0;
};

class MetricsDatabaseFactory {
public:
    static std::unique_ptr<IMetricsDatabase> create(
        const std::string& backend,
        const std::string& db_path = "",
        const std::string& db_url  = "",
        int pool_size = 4);
};
