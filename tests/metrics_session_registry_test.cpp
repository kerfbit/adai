#include <gtest/gtest.h>
#include <filesystem>
#include <thread>
#include "MetricsSessionRegistry.hpp"

TEST(MetricsSessionRegistry, DerivesPerSessionPathsFromBaseConfig) {
    namespace fs = std::filesystem;

    const fs::path temp_root = fs::temp_directory_path() / "adai_metrics_registry_test_paths";
    fs::remove_all(temp_root);

    MetricsServiceConfig config;
    config.enable_persistence = true;
    config.enable_prometheus_format = true;
    config.persist_every_samples = 1;
    config.metrics_file = (temp_root / "metrics.jsonl").string();
    config.summary_file = (temp_root / "metrics_summary.json").string();
    config.prometheus_file = (temp_root / "metrics.prom").string();
    config.abnormal_samples_file = (temp_root / "abnormal_samples.json").string();

    MetricsSessionRegistry registry(config, 16, 3600);
    auto svc = registry.create_or_get_session("42-gpu0");
    ASSERT_NE(svc, nullptr);

    auto derived = svc->get_config();
    EXPECT_NE(derived.metrics_file, config.metrics_file);
    EXPECT_NE(derived.summary_file, config.summary_file);
    EXPECT_NE(derived.prometheus_file, config.prometheus_file);
    EXPECT_NE(derived.abnormal_samples_file, config.abnormal_samples_file);

    EXPECT_NE(derived.metrics_file.find("42-gpu0_metrics.jsonl"), std::string::npos);
    EXPECT_NE(derived.summary_file.find("42-gpu0_metrics_summary.json"), std::string::npos);
    EXPECT_NE(derived.prometheus_file.find("42-gpu0_metrics.prom"), std::string::npos);
    EXPECT_NE(derived.abnormal_samples_file.find("42-gpu0_abnormal_samples.json"),
              std::string::npos);

    fs::remove_all(temp_root);
}

TEST(MetricsSessionRegistry, KeepsLegacyPathsForDefaultKey) {
    MetricsServiceConfig config;
    config.metrics_file = "training_sessions/metrics.jsonl";
    config.summary_file = "training_sessions/metrics_summary.json";
    config.prometheus_file = "training_sessions/metrics.prom";
    config.abnormal_samples_file = "training_sessions/abnormal_samples.json";

    MetricsSessionRegistry registry(config, 16, 3600);
    auto svc = registry.create_or_get_session("0-default");
    ASSERT_NE(svc, nullptr);

    const auto derived = svc->get_config();
    EXPECT_EQ(derived.metrics_file, config.metrics_file);
    EXPECT_EQ(derived.summary_file, config.summary_file);
    EXPECT_EQ(derived.prometheus_file, config.prometheus_file);
    EXPECT_EQ(derived.abnormal_samples_file, config.abnormal_samples_file);
}

TEST(MetricsSessionRegistry, ReplacesCompletedSessionOnRecreate) {
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600);
    auto first = registry.create_or_get_session("7-finetune");
    ASSERT_NE(first, nullptr);

    first->start_session(7, 2, 20);
    first->end_session();

    auto second = registry.create_or_get_session("7-finetune");
    ASSERT_NE(second, nullptr);
    EXPECT_NE(second.get(), first.get());
}

TEST(MetricsSessionRegistry, EvictsCompletedSessionsAfterTtl) {
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 0);
    auto svc = registry.create_or_get_session("88-gpu1");
    ASSERT_NE(svc, nullptr);

    svc->start_session(88, 1, 1);
    svc->end_session();

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto evicted = registry.evict_completed_sessions(0);
    EXPECT_EQ(evicted, 1U);
    EXPECT_EQ(registry.size(), 0U);
}