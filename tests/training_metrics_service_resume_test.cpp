#include <gtest/gtest.h>
#include <filesystem>
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

TEST(TrainingMetricsServiceConfig, CreatesConfiguredOutputFilesInCustomDirectories) {
    namespace fs = std::filesystem;

    const fs::path temp_root =
        fs::temp_directory_path() / "adai_training_metrics_phase1_custom_paths";
    fs::remove_all(temp_root);

    MetricsServiceConfig cfg;
    cfg.enable_persistence = true;
    cfg.enable_push = false;
    cfg.enable_prometheus_format = true;
    cfg.persist_every_samples = 1;
    cfg.metrics_file = (temp_root / "session-a" / "metrics.jsonl").string();
    cfg.summary_file = (temp_root / "session-a" / "summary.json").string();
    cfg.prometheus_file = (temp_root / "session-a" / "metrics.prom").string();
    cfg.abnormal_samples_file = (temp_root / "session-a" / "abnormal.json").string();

    {
        TrainingMetricsService svc(cfg);
        svc.start_session(7, 2, 10);
        svc.update_sample_metrics(1, 1.25f, 0.75f, 0.001f);

        AbnormalSample abnormal;
        abnormal.epoch = 1;
        abnormal.sample_id = 1;
        abnormal.loss = 1.25f;
        abnormal.grad_norm = 0.75f;
        abnormal.reason = "test";
        svc.flag_abnormal_sample(abnormal);

        svc.flush_to_disk();
    }

    EXPECT_TRUE(fs::exists(cfg.metrics_file));
    EXPECT_TRUE(fs::exists(cfg.summary_file));
    EXPECT_TRUE(fs::exists(cfg.prometheus_file));
    EXPECT_TRUE(fs::exists(cfg.abnormal_samples_file));

    fs::remove_all(temp_root);
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

TEST(GlobalMetricsServiceCompatibility, InstanceUsesStableDefaultSessionProxy) {
    GlobalMetricsService::shutdown();

    auto& svc_a = GlobalMetricsService::instance();
    auto& svc_b = GlobalMetricsService::instance();
    EXPECT_EQ(&svc_a, &svc_b);

    svc_a.start_session(101, 2, 20);
    const auto snap = svc_b.get_current_snapshot();
    EXPECT_EQ(snap.session_id, 101);
    EXPECT_TRUE(snap.is_training);

    GlobalMetricsService::shutdown();
}

TEST(GlobalMetricsServiceCompatibility, InitializeConfigAppliesToDefaultSession) {
    namespace fs = std::filesystem;

    GlobalMetricsService::shutdown();

    const fs::path temp_root =
        fs::temp_directory_path() / "adai_global_metrics_service_phase7";
    fs::remove_all(temp_root);

    MetricsServiceConfig cfg;
    cfg.enable_persistence = true;
    cfg.enable_push = false;
    cfg.enable_prometheus_format = false;
    cfg.persist_every_samples = 1;
    cfg.metrics_file = (temp_root / "metrics.jsonl").string();
    cfg.summary_file = (temp_root / "summary.json").string();
    cfg.abnormal_samples_file = (temp_root / "abnormal.json").string();

    GlobalMetricsService::initialize(cfg);

    auto& service = GlobalMetricsService::instance();
    service.start_session(202, 1, 10);
    service.update_sample_metrics(1, 1.5f, 0.4f, 0.001f);
    service.flush_to_disk();

    EXPECT_TRUE(fs::exists(cfg.metrics_file));
    EXPECT_TRUE(fs::exists(cfg.summary_file));

    GlobalMetricsService::shutdown();
    fs::remove_all(temp_root);
}