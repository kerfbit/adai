#include "SessionCounterStore.hpp"
#include <gtest/gtest.h>
#include <sqlite3.h>
#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

class SessionCounterStoreTest : public ::testing::Test {
   protected:
    void SetUp() override {
        static std::atomic<int> counter{0};
        db_path_ = (std::filesystem::temp_directory_path() /
                    ("session_counter_store_test_" + std::to_string(::getpid()) + "_" +
                     std::to_string(counter++) + ".db"))
                       .string();
        remove_db();
    }

    void TearDown() override { remove_db(); }

    void remove_db() {
        for (const char* suffix : {"", "-wal", "-shm"}) std::filesystem::remove(db_path_ + suffix);
    }

    // Runs SQL on its own connection, as a pre-TD-308 binary or another process would.
    void exec(const std::string& sql) {
        sqlite3* raw = nullptr;
        ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw), SQLITE_OK);
        char* err = nullptr;
        const int rc = sqlite3_exec(raw, sql.c_str(), nullptr, nullptr, &err);
        const std::string msg = err ? err : "";
        sqlite3_free(err);
        sqlite3_close(raw);
        ASSERT_EQ(rc, SQLITE_OK) << msg;
    }

    // Writes the table a pre-TD-308 DaemonConfigStore kept the counters in.
    void write_legacy(const std::vector<std::pair<std::string, std::string>>& rows) {
        std::string sql =
            "CREATE TABLE daemon_config (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_utc TEXT NOT NULL);";
        for (const auto& [key, value] : rows) {
            sql += "INSERT INTO daemon_config VALUES ('" + key + "', '" + value + "', '2026-10-01T00:00:00Z');";
        }
        exec(sql);
    }

    bool has_table(const std::string& name) {
        sqlite3* raw = nullptr;
        if (sqlite3_open(db_path_.c_str(), &raw) != SQLITE_OK) return false;
        bool found = false;
        const std::string sql = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '" + name + "';";
        sqlite3_exec(
            raw, sql.c_str(),
            [](void* out, int, char**, char**) {
                *static_cast<bool*>(out) = true;
                return 0;
            },
            &found, nullptr);
        sqlite3_close(raw);
        return found;
    }

    std::string db_path_;
};

// next() counts from 1 and keeps keys apart.
TEST_F(SessionCounterStoreTest, NextCountsPerKey) {
    adai::SessionCounterStore store(db_path_);
    EXPECT_EQ(store.current("m|run-01"), 0);
    EXPECT_EQ(store.next("m|run-01"), 1);
    EXPECT_EQ(store.next("m|run-01"), 2);
    EXPECT_EQ(store.next("m|run-02"), 1);
    EXPECT_EQ(store.next("m|run-01"), 3);
    EXPECT_EQ(store.current("m|run-01"), 3);
    EXPECT_EQ(store.migrated_count(), 0);
}

TEST_F(SessionCounterStoreTest, CountersPersistAcrossReopen) {
    { adai::SessionCounterStore(db_path_).next("m|run-01"); }
    adai::SessionCounterStore store(db_path_);
    EXPECT_EQ(store.next("m|run-01"), 2);
}

// TD-308: the counters live in their own table; the file has no daemon_config table.
TEST_F(SessionCounterStoreTest, CountersHaveTheirOwnTable) {
    adai::SessionCounterStore(db_path_).next("m|run-01");
    EXPECT_TRUE(has_table("session_counters"));
    EXPECT_FALSE(has_table("daemon_config"));
}

// TD-308: a session_counters.db written before the change keeps counting where it was, so a
// restarted registry or trainer doesn't hand out session-01 again for a run it already numbered.
TEST_F(SessionCounterStoreTest, MovesPreTd308CountersAndContinues) {
    write_legacy({{"m|run-01", "7"}, {"m|run-02", "2"}, {"junk", "abc"}});
    adai::SessionCounterStore store(db_path_);
    EXPECT_EQ(store.migrated_count(), 3);
    EXPECT_FALSE(has_table("daemon_config"));
    EXPECT_EQ(store.next("m|run-01"), 8);
    EXPECT_EQ(store.next("m|run-02"), 3);
    EXPECT_EQ(store.next("junk"), 1);  // non-numeric counts as 0, as before
    EXPECT_EQ(adai::SessionCounterStore(db_path_).migrated_count(), 0);
}

// A key counted by both tables (an old binary still writing daemon_config next to a new one) keeps
// the larger value, so no number is reused.
TEST_F(SessionCounterStoreTest, MigrationKeepsTheLargerOfTwoCounters) {
    { adai::SessionCounterStore(db_path_).next("m|run-01"); }  // session_counters: 1
    write_legacy({{"m|run-01", "5"}, {"m|run-03", "4"}});
    exec("UPDATE session_counters SET value = 9 WHERE key = 'm|run-01';");
    exec("INSERT INTO session_counters VALUES ('m|run-03', 2, '2026-11-03T00:00:00Z');");
    adai::SessionCounterStore store(db_path_);
    EXPECT_EQ(store.next("m|run-01"), 10);  // 9 beats the legacy 5
    EXPECT_EQ(store.next("m|run-03"), 5);   // legacy 4 beats 2
}

// The concurrency tests check that nothing is lost or duplicated, not the busy timeout (tested below), so
// they wait far longer than the 5 s default: under heavy disk-sync load (a parallel ctest run) a commit can
// take seconds, and a waiter can be passed over for longer than 5 s.
constexpr int kPatientBusyTimeoutMs = 120000;

// Several processes opening a pre-TD-308 file at once move its counters exactly once.
TEST_F(SessionCounterStoreTest, ConcurrentOpensMigrateOnce) {
    write_legacy({{"m|run-01", "7"}});
    constexpr int kThreads = 4;
    std::atomic<int> moved{0};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            try {
                moved += adai::SessionCounterStore(db_path_, kPatientBusyTimeoutMs).migrated_count();
            } catch (const std::exception&) {
                ++failures;
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(failures.load(), 0);
    EXPECT_EQ(moved.load(), 1);
    EXPECT_EQ(adai::SessionCounterStore(db_path_).next("m|run-01"), 8);
}

// A counter that reached INT_MAX (or was stored negative) starts again at 1, as increment() did.
TEST_F(SessionCounterStoreTest, CounterAtIntMaxStartsAgain) {
    { adai::SessionCounterStore init(db_path_); }
    exec("INSERT INTO session_counters VALUES ('big', 2147483647, 'x'), ('neg', -4, 'x');");
    adai::SessionCounterStore store(db_path_);
    EXPECT_EQ(store.next("big"), 1);
    EXPECT_EQ(store.next("neg"), 1);
}

// TD-304: a write waits for another connection's write lock instead of failing at once, and fails
// (storing nothing) once the busy timeout passes.
TEST_F(SessionCounterStoreTest, NextWaitsForAnotherWriterThenGivesUp) {
    adai::SessionCounterStore store(db_path_, /*busy_timeout_ms=*/100);
    store.next("n");
    sqlite3* raw = nullptr;
    ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw), SQLITE_OK);
    ASSERT_EQ(sqlite3_exec(raw, "BEGIN IMMEDIATE;", nullptr, nullptr, nullptr), SQLITE_OK);
    EXPECT_THROW(store.next("n"), std::runtime_error);
    std::thread release([raw] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        sqlite3_exec(raw, "COMMIT;", nullptr, nullptr, nullptr);
    });
    adai::SessionCounterStore patient(db_path_);
    EXPECT_EQ(patient.next("n"), 2);
    release.join();
    sqlite3_close(raw);
}

// TD-304: concurrent next() calls through separate connections (as separate processes would use)
// are never lost or duplicated.
TEST_F(SessionCounterStoreTest, ConcurrentNextIsNeverLostOrDuplicated) {
    { adai::SessionCounterStore init(db_path_); }
    constexpr int kThreads = 4;
    constexpr int kPerThread = 100;
    std::mutex mu;
    std::vector<int> got;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            try {
                adai::SessionCounterStore store(db_path_, kPatientBusyTimeoutMs);
                for (int i = 0; i < kPerThread; ++i) {
                    try {
                        const int n = store.next("model|run-01");
                        std::lock_guard<std::mutex> lock(mu);
                        got.push_back(n);
                    } catch (const std::exception&) {
                        ++failures;
                    }
                }
            } catch (const std::exception&) {  // a throwing constructor must fail the test, not abort it
                failures += kPerThread;
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
    EXPECT_EQ(adai::SessionCounterStore(db_path_).current("model|run-01"), kThreads * kPerThread);
}

}  // namespace
