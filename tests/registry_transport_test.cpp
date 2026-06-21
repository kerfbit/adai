#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "../src/RegistryTransport.hpp"

namespace fs = std::filesystem;

class LocalTransportTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "adai_registry_transport_test";
        fs::remove_all(tmp_dir_);
        fs::create_directories(tmp_dir_);
        reg_path_  = (tmp_dir_ / "registry.txt").string();
        pend_path_ = (tmp_dir_ / "pending.txt").string();
    }

    void TearDown() override { fs::remove_all(tmp_dir_); }

    fs::path    tmp_dir_;
    std::string reg_path_;
    std::string pend_path_;
};

// ============================================================================
// Missing-file behaviour
// ============================================================================

TEST_F(LocalTransportTest, LoadMissingRegistryReturnsFalse) {
    LocalTransport t(reg_path_, pend_path_);
    std::vector<DataVersion> out;
    EXPECT_FALSE(t.load_registry(out));
    EXPECT_TRUE(out.empty());
}

TEST_F(LocalTransportTest, LoadMissingPendingReturnsFalse) {
    LocalTransport t(reg_path_, pend_path_);
    std::vector<PendingEntry> out;
    EXPECT_FALSE(t.load_pending(out));
    EXPECT_TRUE(out.empty());
}

// ============================================================================
// Round-trip correctness
// ============================================================================

TEST_F(LocalTransportTest, RoundTripRegistry) {
    LocalTransport t(reg_path_, pend_path_);

    DataVersion dv1;
    dv1.data_file   = "/data/training.txt";
    dv1.checksum    = "abc_123";
    dv1.num_samples = 42;
    dv1.trained     = true;

    DataVersion dv2;
    dv2.data_file   = "/data/pending.txt";
    dv2.checksum    = "def_456";
    dv2.num_samples = 0;
    dv2.trained     = false;

    ASSERT_TRUE(t.save_registry({dv1, dv2}));

    std::vector<DataVersion> loaded;
    ASSERT_TRUE(t.load_registry(loaded));

    ASSERT_EQ(loaded.size(), 2u);
    EXPECT_EQ(loaded[0].data_file,   dv1.data_file);
    EXPECT_EQ(loaded[0].checksum,    dv1.checksum);
    EXPECT_EQ(loaded[0].num_samples, dv1.num_samples);
    EXPECT_TRUE(loaded[0].trained);

    EXPECT_EQ(loaded[1].data_file,   dv2.data_file);
    EXPECT_EQ(loaded[1].num_samples, 0);
    EXPECT_FALSE(loaded[1].trained);
}

TEST_F(LocalTransportTest, RoundTripPending) {
    LocalTransport t(reg_path_, pend_path_);

    std::vector<PendingEntry> entries = {
        {"/data/a.txt", ""},
        {"/data/b.txt", ""},
    };

    ASSERT_TRUE(t.save_pending(entries));

    std::vector<PendingEntry> loaded;
    ASSERT_TRUE(t.load_pending(loaded));

    ASSERT_EQ(loaded.size(), 2u);
    EXPECT_EQ(loaded[0].path, "/data/a.txt");
    EXPECT_EQ(loaded[1].path, "/data/b.txt");
    // LocalTransport never sets run_id
    EXPECT_TRUE(loaded[0].run_id.empty());
    EXPECT_TRUE(loaded[1].run_id.empty());
}

// ============================================================================
// Directory creation
// ============================================================================

TEST_F(LocalTransportTest, SaveCreatesParentDirectory) {
    std::string reg  = (tmp_dir_ / "deep/nested/registry.txt").string();
    std::string pend = (tmp_dir_ / "deep/nested/pending.txt").string();
    LocalTransport t(reg, pend);

    EXPECT_TRUE(t.save_registry({}));
    EXPECT_TRUE(fs::exists(reg));

    EXPECT_TRUE(t.save_pending({}));
    EXPECT_TRUE(fs::exists(pend));
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(LocalTransportTest, EmptyFilesRoundTrip) {
    LocalTransport t(reg_path_, pend_path_);

    ASSERT_TRUE(t.save_registry({}));
    ASSERT_TRUE(t.save_pending({}));

    std::vector<DataVersion> reg_out;
    EXPECT_TRUE(t.load_registry(reg_out));
    EXPECT_TRUE(reg_out.empty());

    std::vector<PendingEntry> pend_out;
    EXPECT_TRUE(t.load_pending(pend_out));
    EXPECT_TRUE(pend_out.empty());
}

TEST_F(LocalTransportTest, SkipsCommentAndBlankLinesInRegistry) {
    {
        std::ofstream f(reg_path_);
        f << "# Data Registry: data_file checksum num_samples trained\n";
        f << "\n";
        f << "/data/file.txt abc123 10 1\n";
        f << "\n";
        f << "# another comment\n";
    }

    LocalTransport t(reg_path_, pend_path_);
    std::vector<DataVersion> loaded;
    ASSERT_TRUE(t.load_registry(loaded));
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded[0].data_file,   "/data/file.txt");
    EXPECT_EQ(loaded[0].checksum,    "abc123");
    EXPECT_EQ(loaded[0].num_samples, 10);
    EXPECT_TRUE(loaded[0].trained);
}

TEST_F(LocalTransportTest, OverwriteOnSecondSave) {
    LocalTransport t(reg_path_, pend_path_);

    DataVersion dv;
    dv.data_file = "/data/x.txt";
    dv.checksum  = "csum";
    dv.trained   = false;

    ASSERT_TRUE(t.save_registry({dv}));

    // Overwrite with different content
    dv.trained = true;
    ASSERT_TRUE(t.save_registry({dv}));

    std::vector<DataVersion> loaded;
    ASSERT_TRUE(t.load_registry(loaded));
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_TRUE(loaded[0].trained);
}

// ============================================================================
// Phase 2: model_id field round-trip
// ============================================================================

TEST_F(LocalTransportTest, ModelIdRoundTrip) {
    LocalTransport t(reg_path_, pend_path_);

    DataVersion dv;
    dv.data_file   = "/data/mns_test.txt";
    dv.checksum    = "abc_def";
    dv.num_samples = 100;
    dv.trained     = true;
    dv.model_id    = "550e8400-e29b-41d4-a716-446655440000";

    ASSERT_TRUE(t.save_registry({dv}));

    std::vector<DataVersion> loaded;
    ASSERT_TRUE(t.load_registry(loaded));
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded[0].model_id, dv.model_id);
    EXPECT_EQ(loaded[0].data_file, dv.data_file);
    EXPECT_EQ(loaded[0].trained, true);
}

TEST_F(LocalTransportTest, ModelIdDefaultsToEmptyForOldFormat) {
    // Write a 4-column file (pre-Phase-2 format) and verify model_id loads as empty.
    {
        std::ofstream f(reg_path_);
        f << "# Data Registry: data_file checksum num_samples trained\n";
        f << "/data/legacy.txt abc123 50 1\n";
    }

    LocalTransport t(reg_path_, pend_path_);
    std::vector<DataVersion> loaded;
    ASSERT_TRUE(t.load_registry(loaded));
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_TRUE(loaded[0].model_id.empty())
        << "Old 4-column format should yield empty model_id, got: " << loaded[0].model_id;
    EXPECT_EQ(loaded[0].data_file, "/data/legacy.txt");
    EXPECT_TRUE(loaded[0].trained);
}
