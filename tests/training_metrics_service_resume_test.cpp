#include <gtest/gtest.h>
#include "TrainingMetricsService.hpp"

// TODO(TD-018): add multi-session registry tests (concurrent session keys,
// duplicate start conflict, TTL eviction, and aggregate endpoint snapshots).

static MetricsServiceConfig no_persist_resume_config() {
    MetricsServiceConfig cfg;
    cfg.enable_persistence = false;
    cfg.enable_push = false;
    cfg.enable_prometheus_format = false;
    return cfg;
}

TEST(TrainingMetricsServiceResume, StartSessionPreservesBestValidationState) {
    TrainingMetricsService svc(no_persist_resume_config());

    svc.start_session(1, 4, 1000);
    svc.update_best_metrics(4.25f, 3);

    auto before_restart = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(before_restart.best_validation_loss, 4.25f);
    EXPECT_EQ(before_restart.best_epoch, 3);

    svc.start_session(2, 4, 1000);

    auto after_restart = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(after_restart.best_validation_loss, 4.25f);
    EXPECT_EQ(after_restart.best_epoch, 3);
    EXPECT_EQ(after_restart.session_id, 2);
    EXPECT_TRUE(after_restart.is_training);
}