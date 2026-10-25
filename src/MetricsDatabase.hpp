#pragma once

// @adai-status: stable
// @adai-version: 1.1.0
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

// TODO: See TD-441 in TECHNICAL_DEBT.md - contract (throws, thread safety, ASC order, limit 0 = all) undocumented
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
    // so it gets its own table written on every TrainingMetricsService::
    // update_advanced_epoch_metrics() call instead. `step` is a server-side
    // monotonic counter (not the trainer's sample index).
    // TODO: See TD-441 in TECHNICAL_DEBT.md - step restarts at 0 per TrainingMetricsService instance
    virtual void insert_gradient_variance_sample(const std::string& session_key, int step,
                                                 int epoch, float value) = 0;

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
