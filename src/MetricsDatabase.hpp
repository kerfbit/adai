#pragma once

// @adai-status: stable
// @adai-version: 1.3.0
// @adai-reviewed: 2026-10-25


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
    int session_id = 0;
    std::string label;
    std::string config_json;
    bool is_training = false;
    std::chrono::system_clock::time_point created_at;
    std::optional<std::chrono::system_clock::time_point> ended_at;
    std::chrono::system_clock::time_point last_update_at;
    int total_epochs = 0;
    int total_samples = 0;
    float best_validation_loss = std::numeric_limits<float>::max();
    int best_epoch = 0;
    // Last known training/validation loss as of the most recent persist. Unlike
    // best_validation_loss, these have no "carry forward across sessions" semantics —
    // they simply record where a session's live current_loss/current_validation_loss
    // stood when it was last persisted, so completed/archived sessions (which no
    // longer have a live TrainingMetricsSnapshot) still have a real final value to
    // show instead of the summary view's zero-initialized default.
    float final_loss = 0.0f;
    float final_validation_loss = 0.0f;
};

// TD-440: one gradient_variance_history row (see insert_gradient_variance_sample()).
struct GradientVarianceRecord {
    std::chrono::system_clock::time_point timestamp;
    int step = 0;
    int epoch = 0;
    float value = 0.0f;
};

// TD-440: one generation_quality row (see insert_generation_quality()).
struct GenerationQualityRecord {
    std::chrono::system_clock::time_point timestamp;
    int epoch = 0;
    float bleu1 = 0.0f;
    float bleu2 = 0.0f;
    float bleu4 = 0.0f;
    float rouge1 = 0.0f;
    float rouge2 = 0.0f;
    float rougeL = 0.0f;
};

// TD-440: how cap_session_rows() trims one table holding `count` rows of a session to at most
// `max_rows`. Nothing happens until the cap is exceeded; then the table is trimmed to 3/4 of the
// cap, so the work is done in batches rather than one row per persist. When it can, it thins:
// it deletes every second row among the oldest `thin_first` rows, which halves their density.
// Repeated batches make older history progressively sparser while the start of the run survives.
// Only when more than half the rows must go does it delete the oldest `delete_oldest` rows instead.
struct RowCapPlan {
    long long thin_first = 0;
    long long delete_oldest = 0;
};
RowCapPlan plan_row_cap(long long count, int max_rows);

// The storage behind metrics_api_server's history (TD-020), implemented by SQLiteMetricsDatabase and
// PostgresMetricsDatabase; created by MetricsDatabaseFactory (MetricsDatabase.cpp).
//
// Contract (TD-441):
// - Thread safety: every method may be called from any thread concurrently (SQLite serializes on one
//   connection and mutex; Postgres uses a connection pool).
// - Errors: SQLite methods throw std::runtime_error on failure. Postgres single-row writes and reads
//   log and return (an empty result for reads) once its connection retries are exhausted; its batch
//   inserts throw, so callers can keep the rows for another try. A statement rejected on a healthy
//   connection is never retried (TD-479).
// - Order: every query returns rows oldest first. A `limit` > 0 keeps the newest `limit` rows;
//   0 means all.
// - Sentinels: SessionRecord::best_validation_loss is FLT_MAX when no validation ran; a stored NaN
//   reads back as NaN (SQLite stores it as NULL, TD-510).
// - Retention: cap_session_rows() and delete_sessions_updated_before() delete rows (TD-440).
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

    virtual void insert_metrics_record(const std::string& session_key,
                                       const PersistentMetricsRecord& rec) = 0;

    // Dedicated, unthrottled per-optimizer-step gradient_variance history —
    // deliberately NOT folded into insert_metrics_record()/PersistentMetricsRecord,
    // which is only written once every persist_every_samples (default 100)
    // steps. gradient_variance is the most direct vanishing-gradient signal,
    // so it gets its own table with a row for every TrainingMetricsService::
    // update_advanced_epoch_metrics() call (buffered and written in batches through
    // insert_gradient_variance_samples() since TD-511). `step` is a server-side
    // counter (not the trainer's sample index), increasing within a session: a service
    // that attaches to a session with stored rows continues from max_gradient_variance_step()
    // (TD-441; it used to restart at 0, duplicating steps).
    virtual void insert_gradient_variance_sample(const std::string& session_key, int step,
                                                 int epoch, float value) = 0;

    // TD-511: batch versions, one transaction each, all or nothing; they throw when the batch couldn't
    // be stored. TrainingMetricsService writes its history and per-step gradient-variance rows through
    // these at each persist (one autocommit per row cost ~60x as much in SQLite).
    virtual void insert_metrics_records(const std::string& session_key,
                                        const std::vector<PersistentMetricsRecord>& recs) = 0;
    virtual void insert_gradient_variance_samples(
        const std::string& session_key, const std::vector<GradientVarianceRecord>& recs) = 0;

    virtual void insert_abnormal_sample(const std::string& session_key,
                                        const AbnormalSample& sample) = 0;

    virtual void insert_generation_quality(const std::string& session_key, int epoch,
                                           const GenerationQualityScore& score) = 0;

    virtual std::vector<PersistentMetricsRecord> query_history(
        const std::string& session_key,
        std::optional<std::chrono::system_clock::time_point> from = std::nullopt,
        std::optional<std::chrono::system_clock::time_point> to = std::nullopt, int limit = 0) = 0;

    virtual std::vector<SessionRecord> list_sessions(
        std::optional<bool> is_training_filter = std::nullopt) = 0;

    virtual std::optional<SessionRecord> get_session(const std::string& key) = 0;

    // TD-441: the highest stored gradient-variance step for the session (0 when none).
    virtual int max_gradient_variance_step(const std::string& session_key) = 0;

    // TD-440: readers for the two tables that used to be write-only. Both return rows oldest first;
    // `limit` > 0 keeps only the newest `limit` rows.
    virtual std::vector<GradientVarianceRecord> query_gradient_variance(
        const std::string& session_key, int limit = 0) = 0;
    virtual std::vector<GenerationQualityRecord> query_generation_quality(
        const std::string& session_key) = 0;

    // TD-440: applies plan_row_cap() to the session's metrics_history and gradient_variance_history
    // rows. Returns the number of rows deleted. max_rows <= 0 does nothing.
    virtual long long cap_session_rows(const std::string& session_key, int max_rows) = 0;

    // TD-440: deletes every session whose last_update_at is before `cutoff`, with all of its rows
    // (metrics, gradient variance, generation quality, abnormal samples). A live session updates its
    // row at every persist, so only dead or archived sessions are old enough. Returns the number of
    // sessions deleted.
    virtual long long delete_sessions_updated_before(
        std::chrono::system_clock::time_point cutoff) = 0;
};

class MetricsDatabaseFactory {
   public:
    static std::unique_ptr<IMetricsDatabase> create(const std::string& backend,
                                                    const std::string& db_path = "",
                                                    const std::string& db_url = "",
                                                    int pool_size = 4);
};
