#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "TrainingMetricsService.hpp"

// ============================================================================
// Helpers
// ============================================================================

namespace {

MetricsServiceConfig make_no_persist_config(int staleness_threshold_seconds = 60) {
    MetricsServiceConfig cfg;
    cfg.enable_persistence = false;
    cfg.enable_push = false;
    cfg.enable_prometheus_format = false;
    cfg.staleness_threshold_seconds = staleness_threshold_seconds;
    return cfg;
}

}  // namespace

// ============================================================================
// Suite 1: Ingest timestamp preservation
// ============================================================================

// get_current_snapshot() must NOT overwrite last_update_time.
// The timestamp should reflect the last real metrics write, not the read time.
TEST(StaleDetectionIngestTimestamp, SnapshotPreservesLastUpdateTimeAfterIngest) {
    TrainingMetricsService svc(make_no_persist_config());
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);

    auto snap1 = svc.get_current_snapshot();
    auto t1 = snap1.last_update_time;

    // A small delay and a second read should NOT advance last_update_time
    // (there is no new ingest between the two reads).
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    auto snap2 = svc.get_current_snapshot();
    auto t2 = snap2.last_update_time;

    EXPECT_EQ(t1, t2)
        << "last_update_time should not advance between reads without a new ingest";
}

// A new ingest should update last_update_time.
TEST(StaleDetectionIngestTimestamp, NewIngestAdvancesLastUpdateTime) {
    TrainingMetricsService svc(make_no_persist_config());
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);
    auto snap1 = svc.get_current_snapshot();

    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    svc.update_sample_metrics(2, 0.9f, 0.4f, 0.001f);
    auto snap2 = svc.get_current_snapshot();

    EXPECT_GT(snap2.last_update_time, snap1.last_update_time)
        << "last_update_time must advance after a new ingest";
}

// ============================================================================
// Suite 2: seconds_since_last_update
// ============================================================================

TEST(StaleDetectionSecondsAgo, IsZeroOrSmallImmediatelyAfterIngest) {
    TrainingMetricsService svc(make_no_persist_config());
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);
    auto snap = svc.get_current_snapshot();

    // Right after an ingest the age should be very small (< 5 seconds even on a
    // heavily loaded CI machine).
    EXPECT_GE(snap.seconds_since_last_update, 0.0);
    EXPECT_LT(snap.seconds_since_last_update, 5.0);
}

TEST(StaleDetectionSecondsAgo, IncreasesWithoutNewIngest) {
    TrainingMetricsService svc(make_no_persist_config());
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);

    auto snap1 = svc.get_current_snapshot();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto snap2 = svc.get_current_snapshot();

    EXPECT_GE(snap2.seconds_since_last_update, snap1.seconds_since_last_update)
        << "seconds_since_last_update must not decrease without a new ingest";
}

// seconds_since_last_update must be non-negative even before any ingest.
TEST(StaleDetectionSecondsAgo, NonNegativeBeforeAnyIngest) {
    TrainingMetricsService svc(make_no_persist_config());
    svc.start_session(1, 5, 1000);
    auto snap = svc.get_current_snapshot();
    EXPECT_GE(snap.seconds_since_last_update, 0.0);
}

// ============================================================================
// Suite 3: is_stale transitions
// ============================================================================

// With a very short threshold (1 second), a session that has not received an
// ingest for longer than the threshold should be marked stale.
TEST(StaleDetectionIsStale, FalseImmediatelyAfterIngest) {
    // Threshold = 60 s — should never be stale right after update.
    TrainingMetricsService svc(make_no_persist_config(60));
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);
    auto snap = svc.get_current_snapshot();
    EXPECT_FALSE(snap.is_stale);
}

TEST(StaleDetectionIsStale, TrueAfterThresholdExceeded) {
    // Threshold = -1 seconds — any non-negative secs_since_update satisfies secs > -1.
    TrainingMetricsService svc(make_no_persist_config(-1));
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);

    // Sleep briefly so seconds_since_last_update > 0
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto snap = svc.get_current_snapshot();

    // With threshold=0, any positive seconds_since_last_update triggers stale.
    EXPECT_TRUE(snap.is_stale);
}

// is_stale must be false after end_session() (session is no longer "training").
TEST(StaleDetectionIsStale, FalseAfterSessionEnds) {
    TrainingMetricsService svc(make_no_persist_config(0));
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);
    svc.end_session();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto snap = svc.get_current_snapshot();

    // is_training is false → is_stale must also be false.
    EXPECT_FALSE(snap.is_training);
    EXPECT_FALSE(snap.is_stale);
}

// ============================================================================
// Suite 4: effective_is_training
// ============================================================================

TEST(StaleDetectionEffectiveIsTraining, TrueWhenActiveAndFresh) {
    TrainingMetricsService svc(make_no_persist_config(60));
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);
    auto snap = svc.get_current_snapshot();

    EXPECT_TRUE(snap.is_training);
    EXPECT_FALSE(snap.is_stale);
    EXPECT_TRUE(snap.effective_is_training);
}

TEST(StaleDetectionEffectiveIsTraining, FalseWhenStale) {
    // Threshold = -1 — immediately stale since secs_since_update >= 0 > -1.
    TrainingMetricsService svc(make_no_persist_config(-1));
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto snap = svc.get_current_snapshot();

    EXPECT_TRUE(snap.is_training);    // raw flag still true
    EXPECT_TRUE(snap.is_stale);
    EXPECT_FALSE(snap.effective_is_training);  // stale → not effectively training
}

TEST(StaleDetectionEffectiveIsTraining, FalseWhenSessionNotStarted) {
    TrainingMetricsService svc(make_no_persist_config(60));
    // No start_session() — is_training is false.
    auto snap = svc.get_current_snapshot();
    EXPECT_FALSE(snap.is_training);
    EXPECT_FALSE(snap.effective_is_training);
}

// ============================================================================
// Suite 5: to_json() emits stale-state fields
// ============================================================================

TEST(StaleDetectionJson, EmitsRequiredFields) {
    TrainingMetricsService svc(make_no_persist_config(60));
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);

    const std::string json = svc.to_json();

    EXPECT_NE(json.find("\"is_stale\""), std::string::npos)
        << "to_json() must include 'is_stale'";
    EXPECT_NE(json.find("\"seconds_since_last_update\""), std::string::npos)
        << "to_json() must include 'seconds_since_last_update'";
    EXPECT_NE(json.find("\"effective_is_training\""), std::string::npos)
        << "to_json() must include 'effective_is_training'";
}

TEST(StaleDetectionJson, IsStaleValueMatchesSnapshot) {
    // Threshold = -1 — immediately stale.
    TrainingMetricsService svc(make_no_persist_config(-1));
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const std::string json = svc.to_json();
    EXPECT_NE(json.find("\"is_stale\": true"), std::string::npos)
        << "to_json() should emit is_stale:true when threshold is exceeded";
    EXPECT_NE(json.find("\"effective_is_training\": false"), std::string::npos)
        << "to_json() should emit effective_is_training:false when stale";
}

TEST(StaleDetectionJson, FreshSessionEmitsFalseIsStale) {
    TrainingMetricsService svc(make_no_persist_config(60));
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);

    const std::string json = svc.to_json();
    EXPECT_NE(json.find("\"is_stale\": false"), std::string::npos)
        << "to_json() should emit is_stale:false for a fresh session";
    EXPECT_NE(json.find("\"effective_is_training\": true"), std::string::npos)
        << "to_json() should emit effective_is_training:true for a fresh session";
}

// ============================================================================
// Suite 6: staleness config threshold
// ============================================================================

TEST(StaleDetectionConfig, DefaultThresholdIs60Seconds) {
    MetricsServiceConfig cfg = make_no_persist_config();
    EXPECT_EQ(cfg.staleness_threshold_seconds, 60);
}

TEST(StaleDetectionConfig, CustomThresholdRespected) {
    // Threshold = 1000 s — a fresh ingest should never be stale.
    TrainingMetricsService svc(make_no_persist_config(1000));
    svc.start_session(1, 5, 1000);
    svc.update_sample_metrics(1, 1.0f, 0.5f, 0.001f);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto snap = svc.get_current_snapshot();
    EXPECT_FALSE(snap.is_stale)
        << "With threshold=1000s a brief sleep must not trigger stale state";
}
