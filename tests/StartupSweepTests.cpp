/**
 * StartupSweepTests — unit tests for startup_sweep() (Phase 2).
 *
 * startup_sweep is extracted from IncrementalTrainingTool.cpp into
 * src/StartupSweep.hpp so it can be tested directly here.
 *
 * Four conditions under test:
 *   D  — file already marked trained in registry → delete local copy
 *   G  — file has no registry reference → delete (orphaned)
 *   A  — file is in pending list AND is zero-byte → delete + release
 *   B/C — file is in pending list AND is non-zero → keep for DataTransport
 */
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "../src/DatasetRegistry.hpp"
#include "../src/StartupSweep.hpp"

namespace fs = std::filesystem;

// ============================================================================
// Helpers
// ============================================================================

namespace {

static fs::path make_base(const std::string& tag) {
    fs::path p = fs::temp_directory_path() / ("adai_sweep_test_" + tag);
    fs::remove_all(p);
    return p;
}

static void touch(const fs::path& path, const std::string& content = "placeholder") {
    fs::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f << content;
}

static void touch_empty(const fs::path& path) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    // write nothing — 0 bytes
}

static DatasetConfig make_cfg(const fs::path& sessions_dir) {
    DatasetConfig cfg;
    cfg.session_dir = sessions_dir.string();
    return cfg;
}

}  // namespace

// ============================================================================
// Condition D — trained file: local copy deleted
// ============================================================================

TEST(StartupSweepTest, ConditionD_TrainedFileIsDeleted) {
    const auto base = make_base("d");
    const fs::path sess = base / "sessions";
    const fs::path reg_dir = base / "registry";
    const fs::path dl_dir = base / "downloads";
    fs::create_directories(sess);
    fs::create_directories(reg_dir);
    fs::create_directories(dl_dir);

    // Registry source file (must exist for mark_trained's checksum call)
    const std::string reg_path = (reg_dir / "train.bin").string();
    touch(reg_path);

    // Local copy in download dir
    const fs::path local = dl_dir / "train.bin";
    touch(local);

    DatasetRegistry reg(make_cfg(sess));
    reg.mark_trained({reg_path}, {1});

    startup_sweep(reg, "test-run", dl_dir.string());

    EXPECT_FALSE(fs::exists(local));            // deleted
    EXPECT_FALSE(reg.trained_files().empty());  // registry entry kept

    fs::remove_all(base);
}

// ============================================================================
// Condition G — orphaned file: not in pending or trained → deleted
// ============================================================================

TEST(StartupSweepTest, ConditionG_OrphanedFileIsDeleted) {
    const auto base = make_base("g");
    const fs::path sess = base / "sessions";
    const fs::path dl = base / "downloads";
    fs::create_directories(sess);
    fs::create_directories(dl);

    // No files registered at all
    DatasetRegistry reg(make_cfg(sess));

    const fs::path orphan = dl / "orphan.bin";
    touch(orphan);

    startup_sweep(reg, "test-run", dl.string());

    EXPECT_FALSE(fs::exists(orphan));

    fs::remove_all(base);
}

// ============================================================================
// Condition A — zero-byte file in pending list → deleted + released
// ============================================================================

TEST(StartupSweepTest, ConditionA_ZeroByteDeletedAndReleased) {
    const auto base = make_base("a");
    const fs::path sess = base / "sessions";
    const fs::path reg_dir = base / "registry";
    const fs::path dl_dir = base / "downloads";
    fs::create_directories(sess);
    fs::create_directories(reg_dir);
    fs::create_directories(dl_dir);

    // Registry source file must exist for add_file() existence check
    const std::string reg_path = (reg_dir / "data.bin").string();
    touch(reg_path);

    // Zero-byte local copy in download dir (simulates a crashed download)
    const fs::path local = dl_dir / "data.bin";
    touch_empty(local);
    ASSERT_EQ(fs::file_size(local), 0u);

    DatasetRegistry reg(make_cfg(sess));
    ASSERT_TRUE(reg.add_file(reg_path));
    ASSERT_EQ(reg.pending_files().size(), 1u);

    startup_sweep(reg, "test-run", dl_dir.string());

    EXPECT_FALSE(fs::exists(local));           // file deleted
    EXPECT_TRUE(reg.pending_files().empty());  // released from pending

    fs::remove_all(base);
}

// ============================================================================
// Condition B/C — non-zero file in pending list → kept for DataTransport
// ============================================================================

TEST(StartupSweepTest, ConditionBC_NonZeroFileKept) {
    const auto base = make_base("bc");
    const fs::path sess = base / "sessions";
    const fs::path reg_dir = base / "registry";
    const fs::path dl_dir = base / "downloads";
    fs::create_directories(sess);
    fs::create_directories(reg_dir);
    fs::create_directories(dl_dir);

    const std::string reg_path = (reg_dir / "partial.bin").string();
    touch(reg_path);

    // Non-zero partial local copy
    const fs::path local = dl_dir / "partial.bin";
    touch(local, "some bytes here");
    ASSERT_GT(fs::file_size(local), 0u);

    DatasetRegistry reg(make_cfg(sess));
    ASSERT_TRUE(reg.add_file(reg_path));
    ASSERT_EQ(reg.pending_files().size(), 1u);

    startup_sweep(reg, "test-run", dl_dir.string());

    EXPECT_TRUE(fs::exists(local));             // file kept
    EXPECT_EQ(reg.pending_files().size(), 1u);  // still pending

    fs::remove_all(base);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(StartupSweepTest, EmptyDownloadDirIsNoop) {
    const auto base = make_base("empty");
    const fs::path sess = base / "sessions";
    const fs::path dl = base / "downloads";
    fs::create_directories(sess);
    fs::create_directories(dl);

    DatasetRegistry reg(make_cfg(sess));
    EXPECT_NO_THROW(startup_sweep(reg, "test-run", dl.string()));

    fs::remove_all(base);
}

TEST(StartupSweepTest, NonExistentDownloadDirIsNoop) {
    const auto base = make_base("nodir");
    const fs::path sess = base / "sessions";
    fs::create_directories(sess);

    DatasetRegistry reg(make_cfg(sess));
    EXPECT_NO_THROW(startup_sweep(reg, "test-run", (base / "nonexistent").string()));

    fs::remove_all(base);
}

TEST(StartupSweepTest, EmptyDownloadDirStringIsNoop) {
    const auto base = make_base("emptystr");
    const fs::path sess = base / "sessions";
    fs::create_directories(sess);

    DatasetRegistry reg(make_cfg(sess));
    EXPECT_NO_THROW(startup_sweep(reg, "test-run", ""));

    fs::remove_all(base);
}
