/**
 * StartupSweepTests — unit tests for startup_sweep() (Phase 2).
 *
 * startup_sweep is extracted from IncrementalTrainingTool.cpp into
 * src/StartupSweep.hpp so it can be tested directly here.
 *
 * Four conditions under test:
 *   D  — file already marked trained in registry → delete local copy
 *   G  — file has no registry reference → delete (orphaned)
 *   A  — file is in pending list AND is zero-byte → delete + release this run's claims (TD-513: every
 *        segment too; kept when another run holds a claim)
 *   B/C — file is in pending list AND is non-zero → keep for DataTransport
 */
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

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
    const fs::path local = dl_dir / download_file_name(reg_path);  // TD-337 download name
    touch(local);

    DatasetRegistry reg(make_cfg(sess));
    reg.mark_trained({reg_path}, {1});
    reg.save_pending_list();  // an existing (empty) pending list — TD-512: the sweep needs one

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

    // No files registered at all: an empty (but existing) pending list, so the sweep knows
    // nothing is pending (TD-512: with no pending list at all it skips deleting).
    DatasetRegistry reg(make_cfg(sess));
    reg.save_pending_list();

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
    const fs::path local = dl_dir / download_file_name(reg_path);
    touch_empty(local);
    ASSERT_EQ(fs::file_size(local), 0u);

    DatasetRegistry reg(make_cfg(sess));
    ASSERT_TRUE(reg.add_file(reg_path));
    ASSERT_EQ(reg.acquire_pending("test-run").files.size(), 1u);  // the crashed pass's claim

    startup_sweep(reg, "test-run", dl_dir.string());

    EXPECT_FALSE(fs::exists(local));  // file deleted
    DatasetRegistry after(make_cfg(sess));
    ASSERT_TRUE(after.load_pending_list());
    ASSERT_EQ(after.pending_entries().size(), 1u);
    EXPECT_TRUE(after.pending_entries()[0].run_id.empty()) << "released back to the pool";

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
    const fs::path local = dl_dir / download_file_name(reg_path);
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

// ============================================================================
// TD-512: production call sequence — a freshly constructed registry that has loaded only the
// trained registry (as every caller did). The sweep must load the pending list itself.
// ============================================================================

TEST(StartupSweepTest, FreshRegistryKeepsQueuedDownloads) {
    const auto base = make_base("fresh");
    const fs::path sess = base / "sessions";
    const fs::path src = base / "registry";
    const fs::path dl = base / "downloads";
    fs::create_directories(sess);
    const std::string queued_a = (src / "a.jsonl").string();
    const std::string queued_b = (src / "b.jsonl").string();
    touch(queued_a);
    touch(queued_b);
    {
        DatasetRegistry setup(make_cfg(sess));
        setup.load_pending_list();
        ASSERT_TRUE(setup.add_file(queued_a));
        ASSERT_TRUE(setup.add_file(queued_b));
    }
    const fs::path partial = dl / download_file_name(queued_a);   // partly downloaded
    const fs::path complete = dl / download_file_name(queued_b);  // fully downloaded
    touch(partial, "half");
    touch(complete, "the whole file");

    DatasetRegistry reg(make_cfg(sess));
    reg.load_registry();  // exactly what the call sites did, and nothing more
    startup_sweep(reg, "test-run", dl.string());

    EXPECT_TRUE(fs::exists(partial)) << "a queued partial download must be kept for resume";
    EXPECT_TRUE(fs::exists(complete)) << "a queued complete download must be kept for reuse";

    fs::remove_all(base);
}

TEST(StartupSweepTest, UnreadablePendingListDeletesNothing) {
    const auto base = make_base("unreadable");
    const fs::path sess = base / "sessions";
    const fs::path dl = base / "downloads";
    fs::create_directories(sess);  // no pending list on disk at all
    const fs::path file = dl / "something.jsonl";
    touch(file);

    DatasetRegistry reg(make_cfg(sess));
    startup_sweep(reg, "test-run", dl.string());

    EXPECT_TRUE(fs::exists(file)) << "no pending state: nothing may be called orphaned";
    fs::remove_all(base);
}

// ============================================================================
// TD-337/338: downloads are matched by their full-path names, and checksum sidecars go with them
// ============================================================================

TEST(StartupSweepTest, TrainedFileWithSameBasenameDoesNotDeleteAPendingDownload) {
    const auto base = make_base("samename");
    const fs::path sess = base / "sessions";
    const fs::path dl = base / "downloads";
    fs::create_directories(sess);
    const std::string trained = (base / "uploads" / "x.jsonl").string();
    const std::string pending = (base / "datasets" / "x.jsonl").string();
    touch(trained);
    touch(pending);
    DatasetRegistry reg(make_cfg(sess));
    reg.load_pending_list();
    reg.mark_trained({trained}, {1});
    ASSERT_TRUE(reg.add_file(pending));

    const fs::path kept = dl / download_file_name(pending);
    const fs::path stale = dl / download_file_name(trained);
    touch(kept, "pending bytes");
    touch(stale, "trained bytes");
    touch(fs::path(stale.string() + kDownloadChecksumSuffix), "1_1");
    const fs::path orphan_sidecar = dl / ("gone.jsonl" + std::string(kDownloadChecksumSuffix));
    touch(orphan_sidecar, "1_1");

    startup_sweep(reg, "test-run", dl.string());

    EXPECT_TRUE(fs::exists(kept)) << "a same-basename trained file used to delete this download";
    EXPECT_FALSE(fs::exists(stale));
    EXPECT_FALSE(fs::exists(stale.string() + kDownloadChecksumSuffix)) << "sidecar goes with it";
    EXPECT_FALSE(fs::exists(orphan_sidecar)) << "a sidecar with no download is removed";
    fs::remove_all(base);
}

// TD-513: a zero-byte download of a file this run claimed as TD-205 segments releases every segment
// (a path-only release skipped them, leaving them claimed until the lease ran out).
TEST(StartupSweepTest, ConditionA_ReleasesEverySegmentThisRunClaimed) {
    const auto base = make_base("a_segments");
    const fs::path sess = base / "sessions";
    const fs::path dl_dir = base / "downloads";
    fs::create_directories(sess);
    const std::string reg_path = (base / "registry" / "big.jsonl").string();
    touch(reg_path);
    const fs::path local = dl_dir / download_file_name(reg_path);
    touch_empty(local);

    DatasetRegistry reg(make_cfg(sess));
    ASSERT_TRUE(reg.add_file(reg_path, 0, 5));
    ASSERT_TRUE(reg.add_file(reg_path, 5, 5));
    ASSERT_EQ(reg.acquire_pending("run-x").files.size(), 2u);

    startup_sweep(reg, "run-x", dl_dir.string());

    EXPECT_FALSE(fs::exists(local));
    DatasetRegistry after(make_cfg(sess));
    ASSERT_TRUE(after.load_pending_list());
    ASSERT_EQ(after.pending_entries().size(), 2u);
    for (const auto& e : after.pending_entries())
        EXPECT_TRUE(e.run_id.empty()) << "segment " << e.segment_start << " still claimed by " << e.run_id;
    fs::remove_all(base);
}

// TD-513: another run's claim means its download may be starting right now: leave the file and the claim.
TEST(StartupSweepTest, ConditionA_LeavesAnotherRunsEmptyDownload) {
    const auto base = make_base("a_other_run");
    const fs::path sess = base / "sessions";
    const fs::path dl_dir = base / "downloads";
    fs::create_directories(sess);
    const std::string reg_path = (base / "registry" / "data.jsonl").string();
    touch(reg_path);
    const fs::path local = dl_dir / download_file_name(reg_path);
    touch_empty(local);

    DatasetRegistry reg(make_cfg(sess));
    ASSERT_TRUE(reg.add_file(reg_path));
    ASSERT_EQ(reg.acquire_pending("other-run").files.size(), 1u);

    startup_sweep(reg, "my-run", dl_dir.string());

    EXPECT_TRUE(fs::exists(local));
    DatasetRegistry after(make_cfg(sess));
    ASSERT_TRUE(after.load_pending_list());
    ASSERT_EQ(after.pending_entries().size(), 1u);
    EXPECT_EQ(after.pending_entries()[0].run_id, "other-run");
    fs::remove_all(base);
}

// TD-513: the filesystem calls use their error-code overloads, so a download_dir that can't be read
// is logged and skipped instead of throwing out of the pass.
TEST(StartupSweepTest, UnreadableDownloadDirIsSkippedWithoutThrowing) {
    if (::geteuid() == 0)
        GTEST_SKIP() << "root reads any directory";
    const auto base = make_base("unreadable");
    const fs::path sess = base / "sessions";
    const fs::path dl_dir = base / "downloads";
    fs::create_directories(sess);
    touch(dl_dir / "orphan.bin");
    const std::string queued = (base / "registry" / "q.jsonl").string();
    touch(queued);
    DatasetRegistry reg(make_cfg(sess));
    ASSERT_TRUE(reg.add_file(queued));  // a pending list, so the sweep gets as far as the directory
    fs::permissions(dl_dir, fs::perms::none);

    EXPECT_NO_THROW(startup_sweep(reg, "run", dl_dir.string()));

    fs::permissions(dl_dir, fs::perms::owner_all);
    EXPECT_TRUE(fs::exists(dl_dir / "orphan.bin")) << "nothing was deleted";
    fs::remove_all(base);
}
