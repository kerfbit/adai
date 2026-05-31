#include <gtest/gtest.h>
#include <atomic>
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

// Phase 8: concurrent session tests ------------------------------------------

TEST(MetricsSessionRegistry, ConcurrentSessionCreationIsThreadSafe) {
    // Eight threads each create a distinct session and start training in it.
    // Verifies no crash, no deadlock, and all sessions land in the registry.
    constexpr int kSessions = 8;
    MetricsSessionRegistry registry(MetricsServiceConfig(), 32, 3600);

    std::vector<std::thread> threads;
    threads.reserve(kSessions);
    for (int i = 0; i < kSessions; ++i) {
        threads.emplace_back([&registry, i]() {
            const std::string key = "conc-" + std::to_string(i);
            auto svc = registry.create_or_get_session(key);
            ASSERT_NE(svc, nullptr);
            svc->start_session(i + 1, 3, 100);
            svc->update_sample_metrics(1, static_cast<float>(i) * 0.1f, 0.5f, 0.001f);
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(registry.size(), static_cast<size_t>(kSessions));
}

TEST(MetricsSessionRegistry, ConcurrentReadsAndWritesDoNotDeadlock) {
    // Four writer threads create new sessions while four reader threads
    // concurrently call list_sessions() and get_session(). Passing without
    // deadlock or crash is the success condition.
    MetricsSessionRegistry registry(MetricsServiceConfig(), 32, 3600);

    // Pre-populate sessions so readers have something to iterate immediately.
    for (int i = 0; i < 4; ++i) {
        auto svc = registry.create_or_get_session("preload-" + std::to_string(i));
        ASSERT_NE(svc, nullptr);
        svc->start_session(i + 1, 5, 200);
    }

    std::atomic<bool> stop_readers{false};
    constexpr int kReaders = 4;
    constexpr int kWriters = 4;

    std::vector<std::thread> writers;
    for (int i = 0; i < kWriters; ++i) {
        writers.emplace_back([&registry, i]() {
            const std::string key = "writer-" + std::to_string(i);
            auto svc = registry.create_or_get_session(key);
            if (svc) {
                svc->start_session(100 + i, 2, 50);
                for (int s = 0; s < 10; ++s) {
                    svc->update_sample_metrics(s + 1, 0.5f, 0.4f, 0.001f);
                }
            }
        });
    }

    std::vector<std::thread> readers;
    for (int i = 0; i < kReaders; ++i) {
        readers.emplace_back([&registry, &stop_readers]() {
            while (!stop_readers.load(std::memory_order_relaxed)) {
                const auto summaries = registry.list_sessions();
                (void)summaries.size();
                const auto opt = registry.get_session("preload-0");
                (void)opt;
                std::this_thread::yield();
            }
        });
    }

    for (auto& t : writers) {
        t.join();
    }
    stop_readers.store(true, std::memory_order_relaxed);
    for (auto& t : readers) {
        t.join();
    }

    SUCCEED();
}

TEST(MetricsSessionRegistry, ConcurrentSessionsHaveIsolatedData) {
    // Two sessions update their own metrics from separate threads simultaneously.
    // Verifies each session sees only its own session_id and total_samples.
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600);

    auto svc_a = registry.create_or_get_session("iso-a");
    auto svc_b = registry.create_or_get_session("iso-b");
    ASSERT_NE(svc_a, nullptr);
    ASSERT_NE(svc_b, nullptr);

    std::thread ta([&svc_a]() {
        svc_a->start_session(10, 5, 500);
        for (int i = 0; i < 50; ++i) {
            svc_a->update_sample_metrics(i + 1, 0.8f, 0.6f, 0.001f);
        }
    });
    std::thread tb([&svc_b]() {
        svc_b->start_session(20, 3, 300);
        for (int i = 0; i < 30; ++i) {
            svc_b->update_sample_metrics(i + 1, 2.5f, 1.5f, 0.002f);
        }
    });
    ta.join();
    tb.join();

    const auto snap_a = svc_a->get_current_snapshot();
    const auto snap_b = svc_b->get_current_snapshot();

    EXPECT_EQ(snap_a.session_id, 10);
    EXPECT_EQ(snap_b.session_id, 20);
    EXPECT_EQ(snap_a.total_samples, 500);
    EXPECT_EQ(snap_b.total_samples, 300);
    EXPECT_NE(snap_a.session_id, snap_b.session_id);
    EXPECT_NE(snap_a.total_samples, snap_b.total_samples);
}