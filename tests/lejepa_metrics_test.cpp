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
