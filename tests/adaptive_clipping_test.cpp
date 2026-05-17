/**
 * @file adaptive_clipping_test.cpp
 * @brief Unit tests for TD-017 adaptive gradient clipping (TrainingMetricsService).
 *
 * Tests cover:
 *  - update_adaptive_clip_metrics() stores threshold and spike count in snapshot
 *  - update_adaptive_clip_epoch() pushes avg threshold into epoch history
 *  - to_json() emits current_adaptive_clip_threshold and current_adaptive_clip_spikes
 *  - Default sentinel: current_adaptive_clip_threshold == -1.0f
 *  - Multiple epochs accumulate in epoch_adaptive_clip_thresholds vector
 */

#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include "TrainingMetricsService.hpp"

// ============================================================================
// Helper: create a service with persistence disabled
// ============================================================================
static MetricsServiceConfig no_persist_config() {
    MetricsServiceConfig cfg;
    cfg.enable_persistence = false;
    cfg.enable_push = false;
    cfg.enable_prometheus_format = false;
    return cfg;
}

// ============================================================================
// Defaults
// ============================================================================

TEST(AdaptiveClipping, DefaultThresholdIsSentinel) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_adaptive_clip_threshold, -1.0f);
}

TEST(AdaptiveClipping, DefaultSpikeCountIsZero) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);

    auto snap = svc.get_current_snapshot();
    EXPECT_EQ(snap.current_adaptive_clip_spikes, 0);
}

TEST(AdaptiveClipping, EpochHistoryInitiallyEmpty) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 2, 20);

    auto snap = svc.get_current_snapshot();
    EXPECT_TRUE(snap.epoch_adaptive_clip_thresholds.empty());
}

// ============================================================================
// update_adaptive_clip_metrics
// ============================================================================

TEST(AdaptiveClipping, UpdateStoresThreshold) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);

    svc.update_adaptive_clip_metrics(1.25f, 0);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_adaptive_clip_threshold, 1.25f);
}

TEST(AdaptiveClipping, UpdateStoresSpikeCount) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);

    svc.update_adaptive_clip_metrics(2.0f, 3);

    auto snap = svc.get_current_snapshot();
    EXPECT_EQ(snap.current_adaptive_clip_spikes, 3);
}

TEST(AdaptiveClipping, UpdateOverwritesPreviousValues) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 20);
    svc.start_epoch(1, 20);

    svc.update_adaptive_clip_metrics(0.5f, 1);
    svc.update_adaptive_clip_metrics(1.8f, 4);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_adaptive_clip_threshold, 1.8f);
    EXPECT_EQ(snap.current_adaptive_clip_spikes, 4);
}

// ============================================================================
// update_adaptive_clip_epoch
// ============================================================================

TEST(AdaptiveClipping, EpochUpdatePushesHistory) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 3, 30);

    svc.start_epoch(1, 10);
    svc.update_adaptive_clip_metrics(1.0f, 0);
    svc.update_adaptive_clip_epoch(1.0f, 0);

    auto snap = svc.get_current_snapshot();
    ASSERT_EQ(snap.epoch_adaptive_clip_thresholds.size(), 1u);
    EXPECT_FLOAT_EQ(snap.epoch_adaptive_clip_thresholds[0], 1.0f);
}

TEST(AdaptiveClipping, EpochUpdateAccumulatesAcrossEpochs) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 3, 30);

    svc.start_epoch(1, 10);
    svc.update_adaptive_clip_epoch(1.0f, 0);

    svc.start_epoch(2, 10);
    svc.update_adaptive_clip_epoch(1.5f, 2);

    svc.start_epoch(3, 10);
    svc.update_adaptive_clip_epoch(2.0f, 5);

    auto snap = svc.get_current_snapshot();
    ASSERT_EQ(snap.epoch_adaptive_clip_thresholds.size(), 3u);
    EXPECT_FLOAT_EQ(snap.epoch_adaptive_clip_thresholds[0], 1.0f);
    EXPECT_FLOAT_EQ(snap.epoch_adaptive_clip_thresholds[1], 1.5f);
    EXPECT_FLOAT_EQ(snap.epoch_adaptive_clip_thresholds[2], 2.0f);
}

TEST(AdaptiveClipping, EpochUpdateRefreshesCurrentThreshold) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);

    svc.update_adaptive_clip_epoch(2.5f, 1);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_adaptive_clip_threshold, 2.5f);
    EXPECT_EQ(snap.current_adaptive_clip_spikes, 1);
}

// ============================================================================
// to_json
// ============================================================================

TEST(AdaptiveClipping, ToJsonEmitsThresholdField) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);
    svc.update_adaptive_clip_metrics(1.75f, 0);

    std::string json = svc.to_json();
    EXPECT_NE(json.find("\"current_adaptive_clip_threshold\""), std::string::npos);
}

TEST(AdaptiveClipping, ToJsonEmitsSpikeField) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);
    svc.update_adaptive_clip_metrics(1.0f, 7);

    std::string json = svc.to_json();
    EXPECT_NE(json.find("\"current_adaptive_clip_spikes\""), std::string::npos);
}

TEST(AdaptiveClipping, ToJsonDefaultSentinelIsNegativeOne) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);

    std::string json = svc.to_json();
    // The field must be present and its value must start with -1 (sentinel)
    auto pos = json.find("\"current_adaptive_clip_threshold\"");
    ASSERT_NE(pos, std::string::npos);
    // Skip past the key, colon, and any surrounding whitespace to find the value
    auto colon = json.find(':', pos);
    ASSERT_NE(colon, std::string::npos);
    // First non-space char after the colon should be '-' (negative sentinel)
    auto val_start = json.find_first_not_of(" \t", colon + 1);
    ASSERT_NE(val_start, std::string::npos);
    EXPECT_EQ(json[val_start], '-');
}
