#include "DaemonConfigStore.hpp"
#include <gtest/gtest.h>
#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <filesystem>

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

}  // namespace
