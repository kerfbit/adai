#include "DaemonConfigStore.hpp"
#include <gtest/gtest.h>
#include <sqlite3.h>
#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace {

class DaemonConfigStoreTest : public ::testing::Test {
   protected:
    void SetUp() override {
        static std::atomic<int> counter{0};
        db_path_ = (std::filesystem::temp_directory_path() /
                    ("daemon_config_store_test_" + std::to_string(::getpid()) + "_" +
                     std::to_string(counter++) + ".db"))
                       .string();
        std::filesystem::remove(db_path_);
    }

    void TearDown() override { std::filesystem::remove(db_path_); }

    std::string db_path_;
};

TEST_F(DaemonConfigStoreTest, LoadAllOnFreshStoreIsEmpty) {
    adai::DaemonConfigStore store(db_path_);
    EXPECT_TRUE(store.load_all().empty());
}

TEST_F(DaemonConfigStoreTest, SetThenLoadAllRoundTrips) {
    adai::DaemonConfigStore store(db_path_);
    store.set("registry_group", "g2");
    store.set("allow_control", "true");

    auto all = store.load_all();
    ASSERT_EQ(all.size(), 2u);
    EXPECT_EQ(all.at("registry_group"), "g2");
    EXPECT_EQ(all.at("allow_control"), "true");
}

TEST_F(DaemonConfigStoreTest, SetTwiceUpsertsRatherThanDuplicating) {
    adai::DaemonConfigStore store(db_path_);
    store.set("registry_group", "g1");
    store.set("registry_group", "g2");

    auto all = store.load_all();
    ASSERT_EQ(all.size(), 1u);
    EXPECT_EQ(all.at("registry_group"), "g2");
}

TEST_F(DaemonConfigStoreTest, ValuesPersistAcrossReopen) {
    {
        adai::DaemonConfigStore store(db_path_);
        store.set("port", "8082");
    }
    adai::DaemonConfigStore reopened(db_path_);
    auto all = reopened.load_all();
    ASSERT_EQ(all.size(), 1u);
    EXPECT_EQ(all.at("port"), "8082");
}

// TD-305: set_many() stores every pair in one transaction.
TEST_F(DaemonConfigStoreTest, SetManyStoresEveryPair) {
    adai::DaemonConfigStore store(db_path_);
    store.set("a", "old");
    store.set_many({{"a", "1"}, {"b", "2"}, {"c", "3"}});
    const auto all = store.load_all();
    ASSERT_EQ(all.size(), 3u);
    EXPECT_EQ(all.at("a"), "1");
    EXPECT_EQ(all.at("b"), "2");
    EXPECT_EQ(all.at("c"), "3");
    store.set_many({});  // nothing to store is not an error
    EXPECT_EQ(store.load_all().size(), 3u);
}

// TD-305: a failure partway through stores none of the pairs. Per-key set() calls used to leave the
// keys before the failure stored and the rest not.
TEST_F(DaemonConfigStoreTest, SetManyFailurePartwayStoresNothing) {
    adai::DaemonConfigStore store(db_path_);
    store.set("a", "old");
    sqlite3* raw = nullptr;
    ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw), SQLITE_OK);
    ASSERT_EQ(sqlite3_exec(raw,
                           "CREATE TRIGGER reject_bad BEFORE INSERT ON daemon_config WHEN NEW.key = 'bad' "
                           "BEGIN SELECT RAISE(ABORT, 'rejected by test'); END;",
                           nullptr, nullptr, nullptr),
              SQLITE_OK);
    sqlite3_close(raw);

    // "a" and "aa" sort (and are written) before "bad"; "c" after it.
    EXPECT_THROW(store.set_many({{"a", "new"}, {"aa", "1"}, {"bad", "x"}, {"c", "3"}}), std::runtime_error);
    const auto all = store.load_all();
    ASSERT_EQ(all.size(), 1u);
    EXPECT_EQ(all.at("a"), "old");

    // The store still works afterwards (the failed transaction was rolled back).
    store.set_many({{"a", "newer"}, {"c", "3"}});
    EXPECT_EQ(store.load_all().at("a"), "newer");
}

// TD-305: when another connection keeps the write lock past the busy timeout, set_many() throws and
// stores nothing.
TEST_F(DaemonConfigStoreTest, SetManyThrowsWhileAnotherWriterHoldsTheLock) {
    adai::DaemonConfigStore store(db_path_, /*busy_timeout_ms=*/50);
    store.set("a", "old");
    sqlite3* raw = nullptr;
    ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw), SQLITE_OK);
    ASSERT_EQ(sqlite3_exec(raw, "BEGIN IMMEDIATE;", nullptr, nullptr, nullptr), SQLITE_OK);
    EXPECT_THROW(store.set_many({{"a", "new"}, {"b", "2"}}), std::runtime_error);
    sqlite3_exec(raw, "ROLLBACK;", nullptr, nullptr, nullptr);
    sqlite3_close(raw);
    const auto all = store.load_all();
    ASSERT_EQ(all.size(), 1u);
    EXPECT_EQ(all.at("a"), "old");
}

// TD-304: a write waits for another connection's write lock instead of failing at once. Without a
// busy timeout set() threw "database is locked" within a millisecond.
TEST_F(DaemonConfigStoreTest, WritesWaitForAnotherWriter) {
    adai::DaemonConfigStore store(db_path_);
    store.set("a", "1");
    sqlite3* raw = nullptr;
    ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw), SQLITE_OK);
    ASSERT_EQ(sqlite3_exec(raw, "BEGIN IMMEDIATE;", nullptr, nullptr, nullptr), SQLITE_OK);
    std::thread release([raw] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        sqlite3_exec(raw, "COMMIT;", nullptr, nullptr, nullptr);
    });
    EXPECT_NO_THROW(store.set("a", "2"));
    EXPECT_NO_THROW(store.set_many({{"b", "3"}}));
    EXPECT_NO_THROW(store.increment("n"));
    release.join();
    sqlite3_close(raw);
    const auto all = store.load_all();
    EXPECT_EQ(all.at("a"), "2");
    EXPECT_EQ(all.at("b"), "3");
    EXPECT_EQ(all.at("n"), "1");
}

// TD-304: increment() counts from 1, keeps keys apart, and treats a non-numeric value as 0.
TEST_F(DaemonConfigStoreTest, IncrementCountsPerKey) {
    adai::DaemonConfigStore store(db_path_);
    EXPECT_EQ(store.increment("m|run-01"), 1);
    EXPECT_EQ(store.increment("m|run-01"), 2);
    EXPECT_EQ(store.increment("m|run-02"), 1);
    EXPECT_EQ(store.increment("m|run-01"), 3);
    store.set("bad", "abc");
    EXPECT_EQ(store.increment("bad"), 1);
    EXPECT_EQ(store.load_all().at("m|run-01"), "3");
}

// TD-304: concurrent increments through separate connections (as separate processes would use)
// are never lost or duplicated. LocalTransport's load_all() + set() lost 3 of 400 and threw on 284.
TEST_F(DaemonConfigStoreTest, ConcurrentIncrementsAreNeverLostOrDuplicated) {
    { adai::DaemonConfigStore init(db_path_); }
    constexpr int kThreads = 4;
    constexpr int kPerThread = 100;
    std::mutex mu;
    std::vector<int> got;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            adai::DaemonConfigStore store(db_path_);
            for (int i = 0; i < kPerThread; ++i) {
                try {
                    const int n = store.increment("model|run-01");
                    std::lock_guard<std::mutex> lock(mu);
                    got.push_back(n);
                } catch (const std::exception&) {
                    ++failures;
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(failures.load(), 0);
    ASSERT_EQ(got.size(), static_cast<size_t>(kThreads * kPerThread));
    const std::set<int> unique(got.begin(), got.end());
    EXPECT_EQ(unique.size(), got.size()) << "a number was handed out twice";
    EXPECT_EQ(*unique.begin(), 1);
    EXPECT_EQ(*unique.rbegin(), kThreads * kPerThread);
    EXPECT_EQ(adai::DaemonConfigStore(db_path_).load_all().at("model|run-01"), std::to_string(kThreads * kPerThread));
}

}  // namespace
