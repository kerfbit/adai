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

TEST(MetricsSessionRegistry, StartSessionOrConflictRefusesGenuinelyLiveSession) {
    // Default staleness threshold (60s) — a session that just received an update must
    // never be treated as reclaimable by a second /session/start for the same key.
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600);
    auto first = registry.create_or_get_session("live-key");
    ASSERT_NE(first, nullptr);
    first->start_session(1, 5, 100);

    auto outcome = registry.start_session_or_conflict("live-key");
    EXPECT_TRUE(outcome.conflict);
    EXPECT_EQ(outcome.service, nullptr);

    // The original session must be untouched — not archived, not replaced.
    auto still_there = registry.get_session("live-key");
    ASSERT_TRUE(still_there.has_value());
    EXPECT_EQ(still_there->get(), first.get());
    EXPECT_TRUE(still_there->get()->get_current_snapshot().is_training);
}

TEST(MetricsSessionRegistry, StartSessionOrConflictReclaimsStaleSession) {
    // staleness_threshold_seconds = -1 makes is_stale true immediately after any ingest
    // (secs_since_update >= 0 > -1), simulating a trainer that crashed without posting /end.
    MetricsServiceConfig config;
    config.staleness_threshold_seconds = -1;

    MetricsSessionRegistry registry(config, 16, 3600);
    auto first = registry.create_or_get_session("crashed-key");
    ASSERT_NE(first, nullptr);
    first->start_session(1, 5, 100);
    ASSERT_TRUE(first->get_current_snapshot().is_stale);

    auto outcome = registry.start_session_or_conflict("crashed-key");
    EXPECT_FALSE(outcome.conflict);
    ASSERT_NE(outcome.service, nullptr);
    EXPECT_NE(outcome.service.get(), first.get());
}

TEST(MetricsSessionRegistry, StartSessionOrConflictCreatesFreshSessionForNewKey) {
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600);
    auto outcome = registry.start_session_or_conflict("brand-new-key");
    EXPECT_FALSE(outcome.conflict);
    ASSERT_NE(outcome.service, nullptr);
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

TEST(MetricsSessionRegistry, ReplaceArchivesPreviousSessionInDatabase) {
    namespace fs = std::filesystem;
    const fs::path temp_root = fs::temp_directory_path() / "adai_metrics_registry_test_replace_db";
    fs::remove_all(temp_root);
    fs::create_directories(temp_root);
    const std::string db_path = (temp_root / "metrics.db").string();

    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600, 60, "sqlite", db_path);
    auto first = registry.create_or_get_session("archive-key");
    ASSERT_NE(first, nullptr);
    first->start_session(7, 2, 20);
    first->end_session();

    ASSERT_TRUE(registry.get_database()->get_session("archive-key").has_value());

    auto second = registry.create_or_get_session("archive-key");
    ASSERT_NE(second, nullptr);
    EXPECT_NE(second.get(), first.get());

    // The old row must be moved out from under the reused key immediately upon
    // replacement — not left in place for the new session to silently inherit.
    EXPECT_FALSE(registry.get_database()->get_session("archive-key").has_value());

    // The old data must survive somewhere, under a distinct archived key.
    bool found_archived = false;
    for (const auto& rec : registry.get_database()->list_sessions(std::nullopt)) {
        if (rec.key != "archive-key" && rec.key.rfind("archive-key_archived_", 0) == 0) {
            found_archived = true;
            EXPECT_EQ(rec.session_id, 7);
        }
    }
    EXPECT_TRUE(found_archived);

    second->start_session(9, 1, 5);
    second->end_session();
    auto fresh = registry.get_database()->get_session("archive-key");
    ASSERT_TRUE(fresh.has_value());
    EXPECT_EQ(fresh->session_id, 9);  // fresh data, not the archived run's

    fs::remove_all(temp_root);
}

TEST(MetricsSessionRegistry, ListSessionsExcludesArchivedRowsFromDbSupplement) {
    // Regression test: list_sessions() feeds the dashboard/session-picker. Every eviction
    // (whether via TTL sweep or replace-on-reuse) renames a session's DB row to a unique
    // "<key>_archived_<ts>_<n>" row that is kept forever as queryable history. Left
    // unfiltered, list_sessions()'s DB supplement piles up every one of these rows forever,
    // burying genuinely live sessions in a picker full of dead entries.
    namespace fs = std::filesystem;
    const fs::path temp_root =
        fs::temp_directory_path() / "adai_metrics_registry_test_list_excludes_archived";
    fs::remove_all(temp_root);
    fs::create_directories(temp_root);
    const std::string db_path = (temp_root / "metrics.db").string();

    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600, 60, "sqlite", db_path);

    // A session that has ended but is still within the live-map TTL window remains visible
    // via the live-session loop, same as before this fix — only archived residue is filtered.
    auto ended = registry.create_or_get_session("completed-key");
    ASSERT_NE(ended, nullptr);
    ended->start_session(1, 1, 1);
    ended->end_session();

    // Reuse "archive-key" so its prior run gets archived into a new DB row.
    auto first = registry.create_or_get_session("archive-key");
    ASSERT_NE(first, nullptr);
    first->start_session(7, 2, 20);
    first->end_session();
    auto second = registry.create_or_get_session("archive-key");
    ASSERT_NE(second, nullptr);

    // Sanity: the archived row really is in the DB (same assertion style as the test above).
    bool db_has_archived_row = false;
    for (const auto& rec : registry.get_database()->list_sessions(std::nullopt)) {
        if (rec.key.rfind("archive-key_archived_", 0) == 0) db_has_archived_row = true;
    }
    ASSERT_TRUE(db_has_archived_row);

    // The registry-level (dashboard-facing) listing must not surface the archived row, but
    // must still show the ended-but-still-live "completed-key" session.
    bool picker_has_archived_row = false;
    bool picker_has_completed_row = false;
    for (const auto& summary : registry.list_sessions()) {
        if (summary.key.find("_archived_") != std::string::npos) {
            picker_has_archived_row = true;
        }
        if (summary.key == "completed-key") {
            picker_has_completed_row = true;
        }
    }
    EXPECT_FALSE(picker_has_archived_row);
    EXPECT_TRUE(picker_has_completed_row);

    fs::remove_all(temp_root);
}

TEST(MetricsSessionRegistry, SweepEvictionArchivesSessionInDatabase) {
    namespace fs = std::filesystem;
    const fs::path temp_root = fs::temp_directory_path() / "adai_metrics_registry_test_evict_db";
    fs::remove_all(temp_root);
    fs::create_directories(temp_root);
    const std::string db_path = (temp_root / "metrics.db").string();

    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, /*ttl=*/0, /*sweep=*/0, "sqlite", db_path);
    auto svc = registry.create_or_get_session("evict-key");
    ASSERT_NE(svc, nullptr);
    svc->start_session(3, 1, 1);
    svc->end_session();

    const auto evicted = registry.evict_completed_sessions(0);
    EXPECT_EQ(evicted, 1U);

    EXPECT_FALSE(registry.get_database()->get_session("evict-key").has_value());

    bool found_archived = false;
    for (const auto& rec : registry.get_database()->list_sessions(std::nullopt)) {
        if (rec.key.rfind("evict-key_archived_", 0) == 0) {
            found_archived = true;
        }
    }
    EXPECT_TRUE(found_archived);

    fs::remove_all(temp_root);
}

TEST(MetricsSessionRegistry, EvictionRenamesPerSessionFilesOnDisk) {
    namespace fs = std::filesystem;
    const fs::path temp_root = fs::temp_directory_path() / "adai_metrics_registry_test_evict_files";
    fs::remove_all(temp_root);
    fs::create_directories(temp_root);

    MetricsServiceConfig config;
    config.enable_persistence = true;
    config.persist_every_samples = 1;
    config.metrics_file = (temp_root / "metrics.jsonl").string();
    config.summary_file = (temp_root / "metrics_summary.json").string();
    config.prometheus_file = (temp_root / "metrics.prom").string();
    config.abnormal_samples_file = (temp_root / "abnormal_samples.json").string();

    MetricsSessionRegistry registry(config, 16, /*ttl=*/0, /*sweep=*/0);
    auto svc = registry.create_or_get_session("file-key");
    ASSERT_NE(svc, nullptr);
    svc->start_session(1, 1, 1);
    svc->end_session();

    const auto derived = svc->get_config();
    ASSERT_TRUE(fs::exists(derived.summary_file));

    registry.evict_completed_sessions(0);

    EXPECT_FALSE(fs::exists(derived.summary_file));

    bool found_archived_summary = false;
    for (const auto& entry : fs::directory_iterator(temp_root)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("file-key_archived_", 0) == 0 && name.find("_metrics_summary.json") != std::string::npos) {
            found_archived_summary = true;
        }
    }
    EXPECT_TRUE(found_archived_summary);

    fs::remove_all(temp_root);
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

// Phase 13: label / config_snapshot field tests (TD-021 step 13) ---------------

TEST(MetricsSessionRegistry, ListSessionsPopulatesLabelFromStartSession) {
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600);
    auto svc = registry.create_or_get_session("label-test");
    ASSERT_NE(svc, nullptr);

    svc->start_session(5, 3, 60, "my-run-label");

    const auto summaries = registry.list_sessions();
    ASSERT_EQ(summaries.size(), 1U);
    EXPECT_EQ(summaries[0].label, "my-run-label");
}

TEST(MetricsSessionRegistry, ListSessionsPopulatesConfigSnapshotFromStartSession) {
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600);
    auto svc = registry.create_or_get_session("cfg-snap-test");
    ASSERT_NE(svc, nullptr);

    const std::string cfg = R"({"lr":0.001,"batch_size":32})";
    svc->start_session(6, 2, 40, "", cfg);

    const auto summaries = registry.list_sessions();
    ASSERT_EQ(summaries.size(), 1U);
    EXPECT_EQ(summaries[0].config_snapshot, cfg);
}

TEST(MetricsSessionRegistry, ListSessionsKeyMatchesRegisteredKey) {
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600);
    registry.create_or_get_session("alpha-key");
    registry.create_or_get_session("beta-key");

    const auto summaries = registry.list_sessions();
    ASSERT_EQ(summaries.size(), 2U);

    std::vector<std::string> keys;
    for (const auto& s : summaries) { keys.push_back(s.key); }
    std::sort(keys.begin(), keys.end());

    EXPECT_EQ(keys[0], "alpha-key");
    EXPECT_EQ(keys[1], "beta-key");
}

TEST(MetricsSessionRegistry, LabelAndConfigSnapshotAreBothStoredIndependently) {
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600);
    auto svc = registry.create_or_get_session("both-fields");
    ASSERT_NE(svc, nullptr);

    svc->start_session(7, 1, 10, "the-label", "{\"key\":\"val\"}");

    const auto summaries = registry.list_sessions();
    ASSERT_EQ(summaries.size(), 1U);
    EXPECT_EQ(summaries[0].label,           "the-label");
    EXPECT_EQ(summaries[0].config_snapshot, "{\"key\":\"val\"}");
}

// Phase 13: get_session() tests -----------------------------------------------

TEST(MetricsSessionRegistry, GetSessionReturnsValueForExistingKey) {
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600);
    auto created = registry.create_or_get_session("lookup-key");
    ASSERT_NE(created, nullptr);

    const auto opt = registry.get_session("lookup-key");
    ASSERT_TRUE(opt.has_value());
    EXPECT_EQ(opt->get(), created.get());
}

TEST(MetricsSessionRegistry, GetSessionReturnsNulloptForMissingKey) {
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600);
    const auto opt = registry.get_session("no-such-key");
    EXPECT_FALSE(opt.has_value());
}

TEST(MetricsSessionRegistry, GetSessionReturnsNulloptAfterSessionIsEvicted) {
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 0);
    auto svc = registry.create_or_get_session("evict-me");
    ASSERT_NE(svc, nullptr);
    svc->start_session(1, 1, 1);
    svc->end_session();
    svc.reset();  // release local reference

    registry.evict_completed_sessions(0);

    const auto opt = registry.get_session("evict-me");
    EXPECT_FALSE(opt.has_value());
}

// Phase 13: sweep-thread tests ------------------------------------------------

TEST(MetricsSessionRegistry, SweepIntervalZeroDoesNotStartBackgroundThread) {
    // With sweep_interval_seconds == 0 the sweep thread must NOT be spawned.
    // We verify this indirectly: construct and immediately destroy with no
    // completed sessions — the destructor must not deadlock or crash.
    {
        MetricsSessionRegistry registry(MetricsServiceConfig(), 16, 3600, 0);
        auto svc = registry.create_or_get_session("no-sweep");
        ASSERT_NE(svc, nullptr);
        svc->start_session(1, 1, 1);
        // Session is still active; no eviction should happen automatically.
    }
    // Destructor ran cleanly — no deadlock.
    SUCCEED();
}

TEST(MetricsSessionRegistry, SweepThreadEvictsCompletedSessionsAutomatically) {
    // sweep_interval_seconds = 1 so the background thread fires after ~1 s.
    // completed_ttl_seconds  = 0 so any finished session is immediately stale.
    MetricsSessionRegistry registry(MetricsServiceConfig(), 16, /*ttl=*/0, /*sweep=*/1);

    auto svc = registry.create_or_get_session("auto-sweep-session");
    ASSERT_NE(svc, nullptr);
    svc->start_session(99, 1, 1);
    svc->end_session();
    svc.reset();

    EXPECT_EQ(registry.size(), 1U);  // still present before sweep fires

    // Wait slightly longer than the sweep interval.
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));

    EXPECT_EQ(registry.size(), 0U);
}