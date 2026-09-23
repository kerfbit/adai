/**
 * @file lejepa_metrics_test.cpp
 * @brief Unit tests for LeJEPA world-model pretraining metrics in TrainingMetricsService
 *        (TD-178's own "wire predictor_loss/sigreg_loss as two new TrainingMetricsService
 *        series" Action Item).
 *
 * update_lejepa_metrics() is deliberately not called from LeJEPAEncoder itself — see that
 * class's own doc comment — so these tests exercise the setter directly, the same way
 * padding_efficiency_test.cpp exercises update_padding_efficiency() directly.
 */

#include <gtest/gtest.h>
#include "TrainingMetricsService.hpp"

static MetricsServiceConfig no_persist_config() {
    MetricsServiceConfig cfg;
    cfg.enable_persistence = false;
    cfg.enable_push = false;
    cfg.enable_prometheus_format = false;
    return cfg;
}

TEST(LeJEPAMetrics, DefaultIsNegativeOne) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_predictor_loss, -1.0f);
    EXPECT_FLOAT_EQ(snap.current_sigreg_loss, -1.0f);
}

TEST(LeJEPAMetrics, UpdateStoresBothValues) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);

    svc.update_lejepa_metrics(0.42f, 0.13f);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_predictor_loss, 0.42f);
    EXPECT_FLOAT_EQ(snap.current_sigreg_loss, 0.13f);
}

TEST(LeJEPAMetrics, SubsequentUpdatesOverwritePreviousValues) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);

    svc.update_lejepa_metrics(1.0f, 2.0f);
    svc.update_lejepa_metrics(0.5f, 0.25f);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_predictor_loss, 0.5f);
    EXPECT_FLOAT_EQ(snap.current_sigreg_loss, 0.25f);
}

TEST(LeJEPAMetrics, IndependentOfOtherMetricFields) {
    // The two new fields must be purely additive — setting them must not disturb an unrelated
    // field already on the snapshot (same additive-only guarantee this session has applied to
    // every other metrics/JSON extension).
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);

    svc.update_padding_efficiency(0.9f);
    svc.update_lejepa_metrics(0.3f, 0.1f);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_padding_efficiency, 0.9f);
    EXPECT_FLOAT_EQ(snap.current_predictor_loss, 0.3f);
    EXPECT_FLOAT_EQ(snap.current_sigreg_loss, 0.1f);
}

// ============================================================================
// TD-178 (finished): end_epoch pushes both losses into per-epoch history —
// mirrors padding_efficiency_test.cpp's own EndEpochPushesHistory/
// EndEpochAccumulatesAcrossEpochs/NotComputedSentinelPreservedInHistory.
// ============================================================================

TEST(LeJEPAMetrics, EndEpochPushesHistory) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 3, 30);

    svc.start_epoch(1, 10);
    svc.update_lejepa_metrics(0.42f, 0.13f);
    svc.end_epoch(1, 1.0f, 1.1f, 0.001f);

    auto snap = svc.get_current_snapshot();
    ASSERT_EQ(snap.epoch_predictor_losses.size(), 1u);
    ASSERT_EQ(snap.epoch_sigreg_losses.size(), 1u);
    EXPECT_FLOAT_EQ(snap.epoch_predictor_losses[0], 0.42f);
    EXPECT_FLOAT_EQ(snap.epoch_sigreg_losses[0], 0.13f);
}

TEST(LeJEPAMetrics, EndEpochAccumulatesAcrossEpochs) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 3, 30);

    svc.start_epoch(1, 10);
    svc.update_lejepa_metrics(0.80f, 0.30f);
    svc.end_epoch(1, 1.0f, 1.1f, 0.001f);

    svc.start_epoch(2, 10);
    svc.update_lejepa_metrics(0.60f, 0.20f);
    svc.end_epoch(2, 0.9f, 1.0f, 0.0009f);

    svc.start_epoch(3, 10);
    svc.update_lejepa_metrics(0.40f, 0.10f);
    svc.end_epoch(3, 0.8f, 0.9f, 0.0008f);

    auto snap = svc.get_current_snapshot();
    ASSERT_EQ(snap.epoch_predictor_losses.size(), 3u);
    ASSERT_EQ(snap.epoch_sigreg_losses.size(), 3u);
    EXPECT_FLOAT_EQ(snap.epoch_predictor_losses[0], 0.80f);
    EXPECT_FLOAT_EQ(snap.epoch_predictor_losses[1], 0.60f);
    EXPECT_FLOAT_EQ(snap.epoch_predictor_losses[2], 0.40f);
    EXPECT_FLOAT_EQ(snap.epoch_sigreg_losses[0], 0.30f);
    EXPECT_FLOAT_EQ(snap.epoch_sigreg_losses[1], 0.20f);
    EXPECT_FLOAT_EQ(snap.epoch_sigreg_losses[2], 0.10f);
}

TEST(LeJEPAMetrics, NotComputedSentinelPreservedInHistory) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);
    // Never call update_lejepa_metrics — stays at -1
    svc.end_epoch(1, 1.0f, 1.1f, 0.001f);

    auto snap = svc.get_current_snapshot();
    ASSERT_EQ(snap.epoch_predictor_losses.size(), 1u);
    ASSERT_EQ(snap.epoch_sigreg_losses.size(), 1u);
    EXPECT_FLOAT_EQ(snap.epoch_predictor_losses[0], -1.0f);
    EXPECT_FLOAT_EQ(snap.epoch_sigreg_losses[0], -1.0f);
}

// ============================================================================
// to_json emits current_predictor_loss / current_sigreg_loss
// ============================================================================

TEST(LeJEPAMetrics, ToJsonContainsFields) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);
    svc.update_lejepa_metrics(0.55f, 0.22f);

    std::string json = svc.to_json();
    EXPECT_NE(json.find("\"current_predictor_loss\""), std::string::npos);
    EXPECT_NE(json.find("\"current_sigreg_loss\""), std::string::npos);
}

TEST(LeJEPAMetrics, ToJsonValuesAreCorrect) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);
    svc.update_lejepa_metrics(0.50f, 0.25f);

    std::string json = svc.to_json();
    EXPECT_NE(json.find("\"current_predictor_loss\": 0.500000"), std::string::npos);
    EXPECT_NE(json.find("\"current_sigreg_loss\": 0.250000"), std::string::npos);
}

TEST(LeJEPAMetrics, ToJsonDefaultSentinel) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);

    std::string json = svc.to_json();
    EXPECT_NE(json.find("\"current_predictor_loss\": -1.000000"), std::string::npos);
    EXPECT_NE(json.find("\"current_sigreg_loss\": -1.000000"), std::string::npos);
}
