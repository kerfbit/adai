/**
 * RegistryTransportTests — Phase 9 acquire / release / commit_trained tests
 *
 * Covers LocalTransport's distributed-queue operations.  These run entirely
 * on the local filesystem using temp directories; no registry_server needed.
 *
 * RemoteTransport integration tests (stub-server) are deferred until a
 * test-harness server is available; those will live in this file too.
 */
#include <gtest/gtest.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "../src/RegistryTransport.hpp"

namespace fs = std::filesystem;

// ============================================================================
// Test fixture
// ============================================================================

class LocalTransportPhase9Test : public ::testing::Test {
   protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "adai_registry_p9_test";
        fs::remove_all(tmp_dir_);
        fs::create_directories(tmp_dir_);
        reg_path_ = (tmp_dir_ / "registry.txt").string();
        pend_path_ = (tmp_dir_ / "pending.txt").string();
    }

    void TearDown() override {
        fs::remove_all(tmp_dir_);
    }

    // Write a pending file manually (path only — no run_id)
    void seed_pending(const std::vector<std::string>& paths) {
        std::ofstream f(pend_path_);
        for (const auto& p : paths)
            f << p << '\n';
    }

    // Write a pending file with run_id assignments (tab-separated format)
    void seed_pending_assigned(const std::vector<std::pair<std::string, std::string>>& entries) {
        std::ofstream f(pend_path_);
        for (const auto& [path, run] : entries) {
            if (run.empty())
                f << path << '\n';
            else
                f << path << '\t' << run << '\n';
        }
    }

    fs::path tmp_dir_;
    std::string reg_path_;
    std::string pend_path_;
};

// ============================================================================
// acquire()
// ============================================================================

TEST_F(LocalTransportPhase9Test, AcquireNoFileReturnsEmpty) {
    LocalTransport t(reg_path_, pend_path_);
    // No pending file at all
    auto resp = t.acquire("run-a", 0);
    EXPECT_TRUE(resp.files.empty());
}

TEST_F(LocalTransportPhase9Test, AcquireAllUnassigned) {
    seed_pending({"/data/a.txt", "/data/b.txt", "/data/c.txt"});
    LocalTransport t(reg_path_, pend_path_);

    auto resp = t.acquire("run-a", 0);  // 0 = all available
    ASSERT_EQ(resp.files.size(), 3u);
    EXPECT_EQ(resp.files[0].registry_path, "/data/a.txt");
    EXPECT_EQ(resp.files[1].registry_path, "/data/b.txt");
    EXPECT_EQ(resp.files[2].registry_path, "/data/c.txt");
    // LocalTransport: no FTP credentials
    EXPECT_TRUE(resp.ftp_server_host.empty());

    // Verify run_ids are written to file
    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    ASSERT_EQ(entries.size(), 3u);
    for (const auto& e : entries) {
        EXPECT_EQ(e.run_id, "run-a");
    }
}

TEST_F(LocalTransportPhase9Test, AcquireWithMaxFilesLimit) {
    seed_pending({"/data/a.txt", "/data/b.txt", "/data/c.txt", "/data/d.txt"});
    LocalTransport t(reg_path_, pend_path_);

    auto resp = t.acquire("run-b", 2);
    ASSERT_EQ(resp.files.size(), 2u);

    // Second acquire should get the remaining 2
    auto resp2 = t.acquire("run-c", 10);
    ASSERT_EQ(resp2.files.size(), 2u);

    // Third acquire should find nothing
    auto resp3 = t.acquire("run-d", 0);
    EXPECT_TRUE(resp3.files.empty());
}

TEST_F(LocalTransportPhase9Test, AcquireSkipsAlreadyAssigned) {
    seed_pending_assigned({
        {"/data/a.txt", "run-x"},  // already assigned
        {"/data/b.txt", ""},       // free
    });
    LocalTransport t(reg_path_, pend_path_);

    auto resp = t.acquire("run-y", 0);
    ASSERT_EQ(resp.files.size(), 1u);
    EXPECT_EQ(resp.files[0].registry_path, "/data/b.txt");
}

TEST_F(LocalTransportPhase9Test, AcquireWithModelNameTakesOwnAndUnassignedNeverOthers) {
    LocalTransport t(reg_path_, pend_path_);
    std::vector<PendingEntry> seeded = {
        {"/data/mine.txt", "", "model-a", "", "", 0, -1, ""},
        {"/data/theirs.txt", "", "model-b", "", "", 0, -1, ""},
        {"/data/free.txt", "", "", "", "", 0, -1, ""},
    };
    t.save_pending(seeded);

    auto resp = t.acquire("run-a", 0, "model-a");
    std::vector<std::string> paths;
    for (const auto& f : resp.files)
        paths.push_back(f.registry_path);
    std::sort(paths.begin(), paths.end());
    EXPECT_EQ(paths, (std::vector<std::string>{"/data/free.txt", "/data/mine.txt"}));
}

TEST_F(LocalTransportPhase9Test, AcquireWithEmptyModelNameOnlyTakesUnassigned) {
    LocalTransport t(reg_path_, pend_path_);
    std::vector<PendingEntry> seeded = {
        {"/data/assigned.txt", "", "model-a", "", "", 0, -1, ""},
        {"/data/free.txt", "", "", "", "", 0, -1, ""},
    };
    t.save_pending(seeded);

    auto resp = t.acquire("run-a", 0);  // no model_name — anonymous caller
    ASSERT_EQ(resp.files.size(), 1u);
    EXPECT_EQ(resp.files[0].registry_path, "/data/free.txt");
}

// ============================================================================
// release()
// ============================================================================

TEST_F(LocalTransportPhase9Test, ReleaseReturnsFilesToPool) {
    seed_pending({"/data/a.txt", "/data/b.txt"});
    LocalTransport t(reg_path_, pend_path_);

    t.acquire("run-a", 0);

    // Release /data/a.txt back to the pool
    t.release("run-a", {"/data/a.txt"});

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    ASSERT_EQ(entries.size(), 2u);

    // /data/a.txt should be unassigned; /data/b.txt should still be run-a
    for (const auto& e : entries) {
        if (e.path == "/data/a.txt") {
            EXPECT_TRUE(e.run_id.empty()) << "Expected unassigned after release";
        } else {
            EXPECT_EQ(e.run_id, "run-a");
        }
    }
}

TEST_F(LocalTransportPhase9Test, ReleaseIgnoresWrongRunId) {
    seed_pending({"/data/a.txt"});
    LocalTransport t(reg_path_, pend_path_);

    t.acquire("run-a", 0);

    // run-b tries to release run-a's file — should have no effect
    t.release("run-b", {"/data/a.txt"});

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].run_id, "run-a");  // still assigned to run-a
}

// ============================================================================
// commit_trained()
// ============================================================================

TEST_F(LocalTransportPhase9Test, CommitTrainedUpdatesRegistryAndRemovesPending) {
    seed_pending({"/data/a.txt", "/data/b.txt", "/data/c.txt"});
    LocalTransport t(reg_path_, pend_path_);

    t.acquire("run-a", 2);  // claims a and b

    // Simulate training completion for a and b
    DataVersion dv1;
    dv1.data_file = "/data/a.txt";
    dv1.checksum = "MISSING";
    dv1.trained = true;
    dv1.num_samples = 10;
    DataVersion dv2;
    dv2.data_file = "/data/b.txt";
    dv2.checksum = "MISSING";
    dv2.trained = true;
    dv2.num_samples = 20;

    t.commit_trained("run-a", {dv1, dv2}, {"/data/a.txt", "/data/b.txt"});

    // Registry should have 2 entries
    std::vector<DataVersion> reg;
    t.load_registry(reg);
    ASSERT_EQ(reg.size(), 2u);
    EXPECT_EQ(reg[0].data_file, "/data/a.txt");
    EXPECT_EQ(reg[0].num_samples, 10);
    EXPECT_TRUE(reg[0].trained);
    EXPECT_EQ(reg[1].data_file, "/data/b.txt");

    // Pending should only have c.txt remaining
    std::vector<PendingEntry> pending;
    t.load_pending(pending);
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].path, "/data/c.txt");
}

TEST_F(LocalTransportPhase9Test, CommitTrainedWithEmptyRunIdRemovesAny) {
    seed_pending({"/data/x.txt", "/data/y.txt"});
    LocalTransport t(reg_path_, pend_path_);

    t.acquire("run-a", 1);
    t.acquire("run-b", 1);

    // Commit with empty run_id: removes any matching path regardless of owner
    DataVersion dv;
    dv.data_file = "/data/x.txt";
    dv.checksum = "MISSING";
    dv.trained = true;
    t.commit_trained("", {dv}, {"/data/x.txt"});

    std::vector<PendingEntry> pending;
    t.load_pending(pending);
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].path, "/data/y.txt");
}

TEST_F(LocalTransportPhase9Test, CommitTrainedAppendsToExistingRegistry) {
    // Pre-populate the registry with one entry
    {
        DataVersion existing;
        existing.data_file = "/data/old.txt";
        existing.checksum = "MISSING";
        existing.trained = true;
        LocalTransport t(reg_path_, pend_path_);
        t.save_registry({existing});
    }

    seed_pending({"/data/new.txt"});
    LocalTransport t(reg_path_, pend_path_);
    t.acquire("run-a", 0);

    DataVersion dv;
    dv.data_file = "/data/new.txt";
    dv.checksum = "MISSING";
    dv.trained = true;
    dv.num_samples = 5;
    t.commit_trained("run-a", {dv}, {"/data/new.txt"});

    std::vector<DataVersion> reg;
    t.load_registry(reg);
    ASSERT_EQ(reg.size(), 2u);

    // Pending should be empty
    std::vector<PendingEntry> pending;
    t.load_pending(pending);
    EXPECT_TRUE(pending.empty());
}

// TD-493: only the run holding a unit's claim records it, and a unit trained again is recorded again.
TEST_F(LocalTransportPhase9Test, CommitTrainedRecordsOnlyUnitsTheRunClaimed) {
    seed_pending({"/data/claimed.txt"});
    LocalTransport t(reg_path_, pend_path_);
    t.acquire("run-a", 0);

    DataVersion dv;
    dv.data_file = "/data/claimed.txt";
    dv.checksum = "MISSING";
    dv.trained = true;
    dv.num_samples = 9;
    t.commit_trained("other-run", {dv}, {"/data/claimed.txt"});

    std::vector<DataVersion> reg;
    t.load_registry(reg);
    EXPECT_TRUE(reg.empty()) << "another run's report must not record run-a's claimed unit";
    std::vector<PendingEntry> pending;
    t.load_pending(pending);
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].run_id, "run-a");

    t.commit_trained("run-a", {dv}, {"/data/claimed.txt"});
    t.load_registry(reg);
    ASSERT_EQ(reg.size(), 1u);
    EXPECT_EQ(reg[0].num_samples, 9);
}

TEST_F(LocalTransportPhase9Test, CommitTrainedRecordsAUnitTrainedAgain) {
    seed_pending({"/data/again.txt"});
    LocalTransport t(reg_path_, pend_path_);
    DataVersion dv;
    dv.data_file = "/data/again.txt";
    dv.checksum = "MISSING";
    dv.trained = true;

    t.acquire("run-1", 0);
    dv.num_samples = 5;
    t.commit_trained("run-1", {dv}, {"/data/again.txt"});

    ASSERT_TRUE(t.add_pending("/data/again.txt"));  // re-queued
    t.acquire("run-2", 0);
    dv.num_samples = 7;
    t.commit_trained("run-2", {dv}, {"/data/again.txt"});

    std::vector<DataVersion> reg;
    t.load_registry(reg);
    ASSERT_EQ(reg.size(), 2u);
    EXPECT_EQ(reg[0].num_samples, 5);
    EXPECT_EQ(reg[1].num_samples, 7);
}

// ============================================================================
// Backward compatibility: Phase 8 format (no run_id column) still loads
// ============================================================================

TEST_F(LocalTransportPhase9Test, LoadPendingPhase8FormatBackwardCompat) {
    // Write old-style file (no tabs, no run_id)
    {
        std::ofstream f(pend_path_);
        f << "/data/legacy1.txt\n";
        f << "/data/legacy2.txt\n";
    }

    LocalTransport t(reg_path_, pend_path_);
    std::vector<PendingEntry> entries;
    ASSERT_TRUE(t.load_pending(entries));
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[0].path, "/data/legacy1.txt");
    EXPECT_TRUE(entries[0].run_id.empty());
    EXPECT_EQ(entries[1].path, "/data/legacy2.txt");
    EXPECT_TRUE(entries[1].run_id.empty());
}

// ============================================================================
// Phase 2: model_id preserved through commit_trained
// ============================================================================

TEST_F(LocalTransportPhase9Test, CommitTrainedPreservesModelId) {
    seed_pending({"/data/mns_file.txt"});
    LocalTransport t(reg_path_, pend_path_);
    t.acquire("run-mns", 0);

    DataVersion dv;
    dv.data_file = "/data/mns_file.txt";
    dv.checksum = "MISSING";
    dv.trained = true;
    dv.num_samples = 42;
    dv.model_id = "550e8400-e29b-41d4-a716-446655440000";

    t.commit_trained("run-mns", {dv}, {"/data/mns_file.txt"});

    std::vector<DataVersion> reg;
    t.load_registry(reg);
    ASSERT_EQ(reg.size(), 1u);
    EXPECT_EQ(reg[0].model_id, "550e8400-e29b-41d4-a716-446655440000");
    EXPECT_EQ(reg[0].data_file, "/data/mns_file.txt");
    EXPECT_TRUE(reg[0].trained);
}

// ============================================================================
// Phase 11: server-side dataset fetch — unsupported in local mode
// ============================================================================

TEST_F(LocalTransportPhase9Test, FetchGutenbergUnsupportedInLocalMode) {
    LocalTransport t(reg_path_, pend_path_);
    EXPECT_EQ(t.fetch_gutenberg(1342, 100, ""), "");
}

TEST_F(LocalTransportPhase9Test, FetchHuggingfaceUnsupportedInLocalMode) {
    LocalTransport t(reg_path_, pend_path_);
    EXPECT_EQ(t.fetch_huggingface("daily_dialog", 100, "train", "", "", ""), "");
}

TEST_F(LocalTransportPhase9Test, UploadFileUnsupportedInLocalMode) {
    LocalTransport t(reg_path_, pend_path_);
    EXPECT_FALSE(t.upload_file("/some/local/file.jsonl", false).ok());
}

TEST_F(LocalTransportPhase9Test, CommitTrainedEmptyModelIdRoundTrips) {
    seed_pending({"/data/no_mns.txt"});
    LocalTransport t(reg_path_, pend_path_);
    t.acquire("run-nomns", 0);

    DataVersion dv;
    dv.data_file = "/data/no_mns.txt";
    dv.checksum = "MISSING";
    dv.trained = true;
    dv.num_samples = 10;
    // model_id intentionally left empty

    t.commit_trained("run-nomns", {dv}, {"/data/no_mns.txt"});

    std::vector<DataVersion> reg;
    t.load_registry(reg);
    ASSERT_EQ(reg.size(), 1u);
    EXPECT_TRUE(reg[0].model_id.empty())
        << "Empty model_id should remain empty after round-trip, got: " << reg[0].model_id;
}

// ============================================================================
// Phase 16 — assign-by-count / unassign / delete
// ============================================================================

TEST_F(LocalTransportPhase9Test, AssignAllWhenPathsAndCountEmpty) {
    seed_pending({"/data/a.txt", "/data/b.txt"});
    LocalTransport t(reg_path_, pend_path_);

    auto result = t.assign("model-a", {}, 0);
    EXPECT_EQ(result.assigned, 2);
    EXPECT_EQ(result.paths.size(), 2u);

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    for (const auto& e : entries) {
        EXPECT_EQ(e.model_name, "model-a");
    }
}

TEST_F(LocalTransportPhase9Test, AssignExactPathsIgnoresCount) {
    seed_pending({"/data/a.txt", "/data/b.txt", "/data/c.txt"});
    LocalTransport t(reg_path_, pend_path_);

    auto result = t.assign("model-a", {"/data/b.txt"}, 99);
    EXPECT_EQ(result.assigned, 1);
    ASSERT_EQ(result.paths.size(), 1u);
    EXPECT_EQ(result.paths[0], "/data/b.txt");

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    for (const auto& e : entries) {
        if (e.path == "/data/b.txt")
            EXPECT_EQ(e.model_name, "model-a");
        else
            EXPECT_TRUE(e.model_name.empty());
    }
}

TEST_F(LocalTransportPhase9Test, AssignFirstNUnassignedWhenCountGiven) {
    seed_pending({"/data/a.txt", "/data/b.txt", "/data/c.txt"});
    LocalTransport t(reg_path_, pend_path_);

    auto result = t.assign("model-a", {}, 2);
    EXPECT_EQ(result.assigned, 2);
    ASSERT_EQ(result.paths.size(), 2u);
    EXPECT_EQ(result.paths[0], "/data/a.txt");
    EXPECT_EQ(result.paths[1], "/data/b.txt");

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    ASSERT_EQ(entries.size(), 3u);
    EXPECT_EQ(entries[0].model_name, "model-a");
    EXPECT_EQ(entries[1].model_name, "model-a");
    EXPECT_TRUE(entries[2].model_name.empty());
}

TEST_F(LocalTransportPhase9Test, AssignCountSkipsAlreadyAssignedEntries) {
    LocalTransport t(reg_path_, pend_path_);
    std::vector<PendingEntry> seeded = {
        {"/data/a.txt", "", "model-x", "", "", 0, -1, ""},
        {"/data/b.txt", "", "", "", "", 0, -1, ""},
        {"/data/c.txt", "", "", "", "", 0, -1, ""},
    };
    t.save_pending(seeded);

    auto result = t.assign("model-a", {}, 5);
    EXPECT_EQ(result.assigned, 2);  // only b and c were unassigned; a keeps model-x
}

TEST_F(LocalTransportPhase9Test, UnassignBulkByModelName) {
    seed_pending({"/data/a.txt", "/data/b.txt", "/data/c.txt"});
    LocalTransport t(reg_path_, pend_path_);
    t.assign("model-a", {"/data/a.txt", "/data/b.txt"}, 0);

    auto result = t.unassign("model-a", {}, false);
    EXPECT_EQ(result.unassigned, 2);
    EXPECT_EQ(result.skipped, 0);

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    for (const auto& e : entries) {
        EXPECT_TRUE(e.model_name.empty());
    }
}

TEST_F(LocalTransportPhase9Test, UnassignSpecificPathsWithOwnershipCheck) {
    seed_pending({"/data/a.txt", "/data/b.txt"});
    LocalTransport t(reg_path_, pend_path_);
    t.assign("model-a", {"/data/a.txt"}, 0);
    t.assign("model-b", {"/data/b.txt"}, 0);

    // model-a is not the owner of b.txt — should not match.
    auto result = t.unassign("model-a", {"/data/b.txt"}, false);
    EXPECT_EQ(result.unassigned, 0);

    // The actual owner clears successfully.
    auto result2 = t.unassign("model-b", {"/data/b.txt"}, false);
    EXPECT_EQ(result2.unassigned, 1);
}

TEST_F(LocalTransportPhase9Test, UnassignSkipsActiveClaimWithoutForce) {
    seed_pending_assigned({{"/data/a.txt", "run-a"}});
    LocalTransport t(reg_path_, pend_path_);
    t.assign("model-a", {"/data/a.txt"}, 0);

    auto result = t.unassign("model-a", {"/data/a.txt"}, false);
    EXPECT_EQ(result.unassigned, 0);
    EXPECT_EQ(result.skipped, 1);
}

TEST_F(LocalTransportPhase9Test, UnassignForceClearsActiveClaim) {
    seed_pending_assigned({{"/data/a.txt", "run-a"}});
    LocalTransport t(reg_path_, pend_path_);
    t.assign("model-a", {"/data/a.txt"}, 0);

    auto result = t.unassign("model-a", {"/data/a.txt"}, true);
    EXPECT_EQ(result.unassigned, 1);
    EXPECT_EQ(result.skipped, 0);
}

TEST_F(LocalTransportPhase9Test, UnassignBothEmptyIsNoOp) {
    seed_pending({"/data/a.txt"});
    LocalTransport t(reg_path_, pend_path_);
    t.assign("model-a", {"/data/a.txt"}, 0);

    auto result = t.unassign("", {}, false);
    EXPECT_EQ(result.unassigned, 0);
    EXPECT_EQ(result.skipped, 0);

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].model_name, "model-a");  // untouched
}

TEST_F(LocalTransportPhase9Test, DeletePendingEntryRemovesFromQueue) {
    seed_pending({"/data/a.txt", "/data/b.txt"});
    LocalTransport t(reg_path_, pend_path_);

    auto result = t.delete_paths({"/data/a.txt"}, false, false);
    EXPECT_EQ(result.deleted, 1);
    ASSERT_EQ(result.details.size(), 1u);
    EXPECT_EQ(result.details[0].status, "deleted");

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].path, "/data/b.txt");
}

TEST_F(LocalTransportPhase9Test, DeleteRegistryEntryRemovesFromRegistry) {
    LocalTransport t(reg_path_, pend_path_);
    DataVersion dv;
    dv.data_file = "/data/trained.txt";
    dv.checksum = "MISSING";
    dv.trained = true;
    dv.num_samples = 3;
    t.save_registry({dv});

    auto result = t.delete_paths({"/data/trained.txt"}, false, false);
    EXPECT_EQ(result.deleted, 1);

    std::vector<DataVersion> reg;
    t.load_registry(reg);
    EXPECT_TRUE(reg.empty());
}

TEST_F(LocalTransportPhase9Test, DeleteSkipsActiveClaimWithoutForce) {
    seed_pending_assigned({{"/data/a.txt", "run-a"}});
    LocalTransport t(reg_path_, pend_path_);

    auto result = t.delete_paths({"/data/a.txt"}, false, false);
    EXPECT_EQ(result.deleted, 0);
    EXPECT_EQ(result.skipped, 1);
    ASSERT_EQ(result.details.size(), 1u);
    EXPECT_EQ(result.details[0].status, "skipped_active_run");

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    ASSERT_EQ(entries.size(), 1u);  // still present
}

TEST_F(LocalTransportPhase9Test, DeleteForceRemovesActiveClaim) {
    seed_pending_assigned({{"/data/a.txt", "run-a"}});
    LocalTransport t(reg_path_, pend_path_);

    auto result = t.delete_paths({"/data/a.txt"}, true, false);
    EXPECT_EQ(result.deleted, 1);
    EXPECT_EQ(result.skipped, 0);
}

TEST_F(LocalTransportPhase9Test, DeleteNotFoundReportsCorrectStatus) {
    LocalTransport t(reg_path_, pend_path_);
    auto result = t.delete_paths({"/no/such/file.txt"}, false, false);
    EXPECT_EQ(result.deleted, 0);
    EXPECT_EQ(result.not_found, 1);
    ASSERT_EQ(result.details.size(), 1u);
    EXPECT_EQ(result.details[0].status, "not_found");
}

TEST_F(LocalTransportPhase9Test, DeleteFilesUnlinksWhenRequested) {
    const std::string real_file = (tmp_dir_ / "real.txt").string();
    {
        std::ofstream f(real_file);
        f << "hello";
    }
    seed_pending({real_file});
    LocalTransport t(reg_path_, pend_path_);

    auto result = t.delete_paths({real_file}, false, true);
    EXPECT_EQ(result.deleted, 1);
    ASSERT_EQ(result.details.size(), 1u);
    EXPECT_TRUE(result.details[0].file_deleted);
    EXPECT_FALSE(fs::exists(real_file));
}

TEST_F(LocalTransportPhase9Test, DeleteWithoutDeleteFilesLeavesFileOnDisk) {
    const std::string real_file = (tmp_dir_ / "real2.txt").string();
    {
        std::ofstream f(real_file);
        f << "hello";
    }
    seed_pending({real_file});
    LocalTransport t(reg_path_, pend_path_);

    auto result = t.delete_paths({real_file}, false, false);
    EXPECT_EQ(result.deleted, 1);
    ASSERT_EQ(result.details.size(), 1u);
    EXPECT_FALSE(result.details[0].file_deleted);
    EXPECT_TRUE(fs::exists(real_file));
}

TEST_F(LocalTransportPhase9Test, DeleteRefusesToUnlinkOwnRegistryOrPendingFile) {
    // Register the pending state file's own path as a (contrived) entry.
    seed_pending({pend_path_});
    LocalTransport t(reg_path_, pend_path_);

    auto result = t.delete_paths({pend_path_}, false, true);
    EXPECT_EQ(result.deleted, 1);                   // metadata entry still removed
    ASSERT_EQ(result.details.size(), 1u);
    EXPECT_FALSE(result.details[0].file_deleted);   // but the file itself is untouched
    EXPECT_TRUE(fs::exists(pend_path_));  // still there (rewritten by save_pending, not unlinked)
}

// ============================================================================
// TD-205: row-range segments — segment_matches(), add_pending/assign/unassign/
// delete_paths/release all keyed by (path, segment_start, segment_count) instead
// of bare path.
// ============================================================================

TEST(SegmentMatchesTest, EmptyQueryMatchesOnlyWholeFileEntry) {
    PendingEntry whole;
    whole.path = "/data/a.txt";
    PendingEntry segment;
    segment.path = "/data/a.txt";
    segment.segment_start = 0;
    segment.segment_count = 100;

    EXPECT_TRUE(segment_matches(whole, "/data/a.txt", -1, -1));
    EXPECT_FALSE(segment_matches(segment, "/data/a.txt", -1, -1));
}

TEST(SegmentMatchesTest, SegmentQueryMatchesOnlyExactRange) {
    PendingEntry segment;
    segment.path = "/data/a.txt";
    segment.segment_start = 0;
    segment.segment_count = 100;

    EXPECT_TRUE(segment_matches(segment, "/data/a.txt", 0, 100));
    EXPECT_FALSE(segment_matches(segment, "/data/a.txt", 100, 100));  // different range
    EXPECT_FALSE(segment_matches(segment, "/data/a.txt", -1, -1));   // "whole file" query
}

TEST(SegmentMatchesTest, DifferentPathNeverMatches) {
    PendingEntry e;
    e.path = "/data/a.txt";
    EXPECT_FALSE(segment_matches(e, "/data/b.txt", -1, -1));
}

TEST_F(LocalTransportPhase9Test, AddPendingTwoSegmentsOfSamePathAreIndependentEntries) {
    LocalTransport t(reg_path_, pend_path_);
    EXPECT_TRUE(t.add_pending("/data/big.jsonl", 0, 100));
    EXPECT_TRUE(t.add_pending("/data/big.jsonl", 100, 100));

    std::vector<PendingEntry> entries;
    ASSERT_TRUE(t.load_pending(entries));
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[0].path, "/data/big.jsonl");
    EXPECT_EQ(entries[0].segment_start, 0);
    EXPECT_EQ(entries[0].segment_count, 100);
    EXPECT_EQ(entries[1].segment_start, 100);
    EXPECT_EQ(entries[1].segment_count, 100);
}

TEST_F(LocalTransportPhase9Test, AddPendingRejectsDuplicateSegment) {
    LocalTransport t(reg_path_, pend_path_);
    ASSERT_TRUE(t.add_pending("/data/big.jsonl", 0, 100));
    EXPECT_TRUE(t.add_pending("/data/big.jsonl", 0, 100));  // no-op success, same as whole-file dedup

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    EXPECT_EQ(entries.size(), 1u);
}

TEST_F(LocalTransportPhase9Test, AddPendingSegmentDoesNotCollideWithWholeFileEntry) {
    LocalTransport t(reg_path_, pend_path_);
    ASSERT_TRUE(t.add_pending("/data/big.jsonl"));           // whole-file
    ASSERT_TRUE(t.add_pending("/data/big.jsonl", 0, 100));  // segment of the same path

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    ASSERT_EQ(entries.size(), 2u);
}

TEST_F(LocalTransportPhase9Test, AssignSegmentTargetsExactRangeOnly) {
    LocalTransport t(reg_path_, pend_path_);
    t.add_pending("/data/big.jsonl", 0, 100);
    t.add_pending("/data/big.jsonl", 100, 100);

    auto result = t.assign("model-a", {}, 0, {{"/data/big.jsonl", 0, 100}});
    EXPECT_EQ(result.assigned, 1);
    ASSERT_EQ(result.segments.size(), 1u);
    EXPECT_TRUE(result.paths.empty());

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    for (const auto& e : entries) {
        if (e.segment_start == 0)
            EXPECT_EQ(e.model_name, "model-a");
        else
            EXPECT_TRUE(e.model_name.empty());
    }
}

TEST_F(LocalTransportPhase9Test, AssignByWholeFilePathsNeverTouchesSegmentsOfThatPath) {
    LocalTransport t(reg_path_, pend_path_);
    t.add_pending("/data/big.jsonl");           // whole-file entry
    t.add_pending("/data/big.jsonl", 0, 100);  // segment of the same path

    auto result = t.assign("model-a", {"/data/big.jsonl"}, 0);
    EXPECT_EQ(result.assigned, 1);

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    for (const auto& e : entries) {
        if (e.segment_start < 0)
            EXPECT_EQ(e.model_name, "model-a");
        else
            EXPECT_TRUE(e.model_name.empty()) << "whole-file paths must not match a segment";
    }
}

TEST_F(LocalTransportPhase9Test, UnassignSegmentTargetsExactRangeOnly) {
    LocalTransport t(reg_path_, pend_path_);
    t.add_pending("/data/big.jsonl", 0, 100);
    t.add_pending("/data/big.jsonl", 100, 100);
    t.assign("model-a", {}, 0, {{"/data/big.jsonl", 0, 100}});
    t.assign("model-a", {}, 0, {{"/data/big.jsonl", 100, 100}});

    auto result = t.unassign("model-a", {}, false, {{"/data/big.jsonl", 0, 100}});
    EXPECT_EQ(result.unassigned, 1);
    ASSERT_EQ(result.segments.size(), 1u);

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    for (const auto& e : entries) {
        if (e.segment_start == 0)
            EXPECT_TRUE(e.model_name.empty());
        else
            EXPECT_EQ(e.model_name, "model-a");
    }
}

TEST_F(LocalTransportPhase9Test, DeleteSegmentTargetNeverUnlinksSharedPhysicalFile) {
    const std::string real_file = (tmp_dir_ / "shared.jsonl").string();
    {
        std::ofstream f(real_file);
        f << "hello";
    }
    LocalTransport t(reg_path_, pend_path_);
    t.add_pending(real_file, 0, 100);
    t.add_pending(real_file, 100, 100);

    // delete_files=true must still be ignored for a segment target — another segment
    // of the same physical file is still pending and needs the bytes to remain.
    auto result = t.delete_paths({}, false, true, {{real_file, 0, 100}});
    EXPECT_EQ(result.deleted, 1);
    ASSERT_EQ(result.details.size(), 1u);
    EXPECT_FALSE(result.details[0].file_deleted);
    EXPECT_TRUE(fs::exists(real_file));

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].segment_start, 100);  // the other segment survives
}

TEST_F(LocalTransportPhase9Test, DeleteByWholeFilePathsNeverTouchesSegmentsOfThatPath) {
    LocalTransport t(reg_path_, pend_path_);
    t.add_pending("/data/big.jsonl");
    t.add_pending("/data/big.jsonl", 0, 100);

    auto result = t.delete_paths({"/data/big.jsonl"}, false, false);
    EXPECT_EQ(result.deleted, 1);

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].segment_start, 0) << "the segment must survive a whole-file-path delete";
}

TEST_F(LocalTransportPhase9Test, ReleaseSegmentTargetOnlyReleasesThatSegment) {
    LocalTransport t(reg_path_, pend_path_);
    t.add_pending("/data/big.jsonl", 0, 100);
    t.add_pending("/data/big.jsonl", 100, 100);
    t.acquire("run-1", 0);  // claims both segments under run-1

    t.release("run-1", {}, {{"/data/big.jsonl", 0, 100}});

    std::vector<PendingEntry> entries;
    t.load_pending(entries);
    for (const auto& e : entries) {
        if (e.segment_start == 0)
            EXPECT_TRUE(e.run_id.empty());
        else
            EXPECT_EQ(e.run_id, "run-1") << "the other segment's claim must survive";
    }
}

// ============================================================================
// next_session() — Phase 3 run/session numbering
// ============================================================================

TEST_F(LocalTransportPhase9Test, NextSessionStartsAtOneForNewRunId) {
    LocalTransport t(reg_path_, pend_path_);
    EXPECT_EQ(t.next_session("model-a", "run-01"), "session-01");
}

TEST_F(LocalTransportPhase9Test, NextSessionIncrementsWithinSameRunId) {
    LocalTransport t(reg_path_, pend_path_);
    EXPECT_EQ(t.next_session("model-a", "run-01"), "session-01");
    EXPECT_EQ(t.next_session("model-a", "run-01"), "session-02");
    EXPECT_EQ(t.next_session("model-a", "run-01"), "session-03");
}

TEST_F(LocalTransportPhase9Test, NextSessionResetsForNewRunId) {
    LocalTransport t(reg_path_, pend_path_);
    EXPECT_EQ(t.next_session("model-a", "run-01"), "session-01");
    EXPECT_EQ(t.next_session("model-a", "run-01"), "session-02");
    // A different run_id starts its own counter at 1 — no explicit reset needed.
    EXPECT_EQ(t.next_session("model-a", "run-02"), "session-01");
}

TEST_F(LocalTransportPhase9Test, NextSessionIsPerModelWithinSameRunId) {
    LocalTransport t(reg_path_, pend_path_);
    EXPECT_EQ(t.next_session("model-a", "run-01"), "session-01");
    // A different model_name under the same run_id has an independent counter.
    EXPECT_EQ(t.next_session("model-b", "run-01"), "session-01");
    EXPECT_EQ(t.next_session("model-a", "run-01"), "session-02");
}

TEST_F(LocalTransportPhase9Test, NextSessionPersistsAcrossTransportInstances) {
    {
        LocalTransport t(reg_path_, pend_path_);
        EXPECT_EQ(t.next_session("model-a", "run-01"), "session-01");
    }
    // A fresh LocalTransport instance (simulating a process restart) reopens
    // the same session_counters.db next to pending_path_ and continues.
    LocalTransport t2(reg_path_, pend_path_);
    EXPECT_EQ(t2.next_session("model-a", "run-01"), "session-02");
}

// ============================================================================
// TD-495: a save that fails part-way (disk full) leaves the old queue whole
// ============================================================================

namespace {
// Runs `fn` in a child whose files may be at most 4 KiB (a full disk, in effect); returns its exit
// code.
template <typename Fn>
int run_with_4k_file_limit(Fn fn) {
    const pid_t child = ::fork();
    if (child == 0) {
        std::signal(SIGXFSZ, SIG_IGN);
        rlimit limit{4096, 4096};
        ::setrlimit(RLIMIT_FSIZE, &limit);
        ::_exit(fn());
    }
    int status = 0;
    ::waitpid(child, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
}  // namespace

// The verified case: rewriting a 2,001-entry queue under a 4 KiB limit used to return false and
// keep 114 entries. It must keep all 2,001, and a later load sees them.
TEST_F(LocalTransportPhase9Test, FailedSaveKeepsEveryPendingEntry) {
    std::vector<std::string> paths;
    for (int i = 0; i < 2001; ++i)
        paths.push_back("/data/queued_file_" + std::to_string(i) + ".jsonl");
    seed_pending(paths);
    LocalTransport t(reg_path_, pend_path_);

    const int code = run_with_4k_file_limit([&] {
        std::vector<PendingEntry> entries;
        t.load_pending(entries);
        entries[0].run_id = "run-01";            // any change rewrites the whole file
        return t.save_pending(entries) ? 1 : 0;  // 0 = the failure was reported
    });
    EXPECT_EQ(code, 0) << "save_pending() must report the failed write";

    std::vector<PendingEntry> after;
    ASSERT_TRUE(t.load_pending(after));
    EXPECT_EQ(after.size(), 2001u) << "the old queue survives whole";
    EXPECT_TRUE(after[0].run_id.empty()) << "and unchanged";
}

TEST_F(LocalTransportPhase9Test, FailedRegistrySaveKeepsTheOldRegistry) {
    LocalTransport t(reg_path_, pend_path_);
    std::vector<DataVersion> reg;
    for (int i = 0; i < 300; ++i) {
        DataVersion dv;
        dv.data_file = "/data/trained_" + std::to_string(i) + ".jsonl";
        dv.trained = true;
        reg.push_back(dv);
    }
    ASSERT_TRUE(t.save_registry(reg));
    const int code = run_with_4k_file_limit([&] {
        std::vector<DataVersion> more = reg;
        more.push_back(more.front());
        return t.save_registry(more) ? 1 : 0;
    });
    EXPECT_EQ(code, 0);
    std::vector<DataVersion> after;
    ASSERT_TRUE(t.load_registry(after));
    EXPECT_EQ(after.size(), 300u);
}

// add_pending() used to append in place, so a failed append could leave a cut-off line, which
// reloads as a wrong path; it now rewrites the queue atomically and reports the failure.
TEST_F(LocalTransportPhase9Test, FailedAddPendingLeavesNoPartialLine) {
    std::vector<std::string> paths;
    for (int i = 0; i < 150; ++i)
        paths.push_back("/data/f" + std::to_string(i) + ".jsonl");
    seed_pending(paths);
    LocalTransport t(reg_path_, pend_path_);
    const int code = run_with_4k_file_limit(
        [&] { return t.add_pending("/data/a_new_file_with_a_long_enough_name.jsonl") ? 1 : 0; });
    EXPECT_EQ(code, 0);
    std::vector<PendingEntry> after;
    ASSERT_TRUE(t.load_pending(after));
    EXPECT_EQ(after.size(), 150u);
    for (const auto& e : after)
        EXPECT_EQ(e.path.rfind("/data/f", 0), 0u) << e.path;

    ASSERT_TRUE(t.add_pending("/data/new.jsonl"));
    ASSERT_TRUE(t.load_pending(after));
    EXPECT_EQ(after.size(), 151u);
    EXPECT_EQ(after.back().path, "/data/new.jsonl");
}

// acquire() reports a claim it couldn't save as a failure and claims nothing.
TEST_F(LocalTransportPhase9Test, AcquireThatCannotSaveClaimsNothing) {
    std::vector<std::string> paths;
    for (int i = 0; i < 300; ++i)
        paths.push_back("/data/acq_" + std::to_string(i) + ".jsonl");
    seed_pending(paths);
    LocalTransport t(reg_path_, pend_path_);
    const int code = run_with_4k_file_limit([&] {
        const AcquireResponse r = t.acquire("run-01", 5);
        return (r.failed && r.files.empty()) ? 0 : 1;
    });
    EXPECT_EQ(code, 0);
    std::vector<PendingEntry> after;
    ASSERT_TRUE(t.load_pending(after));
    for (const auto& e : after)
        EXPECT_TRUE(e.run_id.empty());
}

// ============================================================================
// TD-496: JSON Lines files; spaces, tabs and newlines in paths round-trip
// ============================================================================

TEST_F(LocalTransportPhase9Test, TrainedRecordWithSpaceInPathRoundTrips) {
    LocalTransport t(reg_path_, pend_path_);
    DataVersion dv;
    dv.data_file = "/data/my file.jsonl";
    dv.checksum = "1234:5678";
    dv.num_samples = 42;
    dv.trained = true;
    dv.source = "upload";
    dv.segment_start = 10;
    dv.segment_count = 5;
    ASSERT_TRUE(t.save_registry({dv}));

    std::vector<DataVersion> loaded;
    ASSERT_TRUE(t.load_registry(loaded));
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded[0].data_file, "/data/my file.jsonl");
    EXPECT_EQ(loaded[0].checksum, "1234:5678");
    EXPECT_EQ(loaded[0].num_samples, 42);
    EXPECT_TRUE(loaded[0].trained);
    EXPECT_TRUE(loaded[0].model_id.empty());
    EXPECT_EQ(loaded[0].source, "upload");
    EXPECT_EQ(loaded[0].segment_start, 10);
    EXPECT_EQ(loaded[0].segment_count, 5);
}

TEST_F(LocalTransportPhase9Test, PendingEntriesWithSpacesTabsAndNewlinesRoundTrip) {
    // Paths like these can't be queued any more (registry_path_problem), but entries already in a
    // queue, or written by other code, must still survive a save.
    LocalTransport t(reg_path_, pend_path_);
    std::vector<PendingEntry> entries(3);
    entries[0].path = "/data/my file.jsonl";
    entries[0].model_name = "model a";
    entries[1].path = "/data/tab\there.jsonl";
    entries[1].run_id = "run-01";
    entries[1].claimed_unix = 1700000000;
    entries[2].path = "/data/new\nline.jsonl";
    entries[2].size_bytes = 9000000000ULL;
    entries[2].num_entries = 7;
    ASSERT_TRUE(t.save_pending(entries));

    std::vector<PendingEntry> loaded;
    ASSERT_TRUE(t.load_pending(loaded));
    ASSERT_EQ(loaded.size(), 3u);
    EXPECT_EQ(loaded[0].path, "/data/my file.jsonl");
    EXPECT_EQ(loaded[0].model_name, "model a");
    EXPECT_EQ(loaded[1].path, "/data/tab\there.jsonl");
    EXPECT_EQ(loaded[1].run_id, "run-01");
    EXPECT_EQ(loaded[1].claimed_unix, 1700000000);
    EXPECT_EQ(loaded[2].path, "/data/new\nline.jsonl");
    EXPECT_EQ(loaded[2].size_bytes, 9000000000ULL);
    EXPECT_EQ(loaded[2].num_entries, 7);
    EXPECT_EQ(loaded[2].segment_start, -1);
}

TEST_F(LocalTransportPhase9Test, OldFormatFilesLoadAndAreRewrittenAsJsonLines) {
    {
        std::ofstream reg(reg_path_);
        reg << "# Data Registry: data_file checksum num_samples trained model_id added_utc source "
               "segment_start segment_count\n";
        reg << "/data/a.jsonl - 12 1 m-1 2026-10-01T00:00:00Z gutenberg 0 6\n";
        reg << "/data/b.jsonl abc 3 0\n";
        std::ofstream pend(pend_path_);
        pend << "/data/c.jsonl\trun-02\tmodel-x\tupload\t2026-10-02T00:00:00Z\t100\t4\tck\t2\t2\t"
                "1700000001\n";
        pend << "/data/d.jsonl\n";
    }
    LocalTransport t(reg_path_, pend_path_);

    for (int pass = 0; pass < 2; ++pass) {  // pass 0: old format; pass 1: after a save
        SCOPED_TRACE(pass);
        std::vector<DataVersion> reg;
        ASSERT_TRUE(t.load_registry(reg));
        ASSERT_EQ(reg.size(), 2u);
        EXPECT_EQ(reg[0].data_file, "/data/a.jsonl");
        EXPECT_TRUE(reg[0].checksum.empty());
        EXPECT_EQ(reg[0].num_samples, 12);
        EXPECT_TRUE(reg[0].trained);
        EXPECT_EQ(reg[0].model_id, "m-1");
        EXPECT_EQ(reg[0].added_utc, "2026-10-01T00:00:00Z");
        EXPECT_EQ(reg[0].source, "gutenberg");
        EXPECT_EQ(reg[0].segment_start, 0);
        EXPECT_EQ(reg[0].segment_count, 6);
        EXPECT_EQ(reg[1].checksum, "abc");
        EXPECT_FALSE(reg[1].trained);
        EXPECT_EQ(reg[1].segment_start, -1);

        std::vector<PendingEntry> pend;
        ASSERT_TRUE(t.load_pending(pend));
        ASSERT_EQ(pend.size(), 2u);
        EXPECT_EQ(pend[0].path, "/data/c.jsonl");
        EXPECT_EQ(pend[0].run_id, "run-02");
        EXPECT_EQ(pend[0].model_name, "model-x");
        EXPECT_EQ(pend[0].source, "upload");
        EXPECT_EQ(pend[0].size_bytes, 100u);
        EXPECT_EQ(pend[0].num_entries, 4);
        EXPECT_EQ(pend[0].checksum, "ck");
        EXPECT_EQ(pend[0].segment_start, 2);
        EXPECT_EQ(pend[0].segment_count, 2);
        EXPECT_EQ(pend[0].claimed_unix, 1700000001);
        EXPECT_EQ(pend[1].path, "/data/d.jsonl");

        if (pass == 0) {
            ASSERT_TRUE(t.save_registry(reg));
            ASSERT_TRUE(t.save_pending(pend));
        }
    }

    std::ifstream reg(reg_path_);
    std::string first;
    std::getline(reg, first);
    EXPECT_EQ(first, "# adai data registry, JSON Lines v2");
    std::ifstream pend(pend_path_);
    std::getline(pend, first);
    EXPECT_EQ(first, "# adai pending queue, JSON Lines v2");
}

TEST_F(LocalTransportPhase9Test, UnreadableJsonLineIsSkippedOthersKept) {
    {
        std::ofstream pend(pend_path_);
        pend << "# adai pending queue, JSON Lines v2\n";
        pend << "{\"path\":\"/data/one.jsonl\"}\n";
        pend << "{\"path\":\"/data/cut\n";
        pend << "{\"path\":\"/data/two.jsonl\",\"num_entries\":\"not a number\"}\n";
    }
    LocalTransport t(reg_path_, pend_path_);
    std::vector<PendingEntry> pend;
    ASSERT_TRUE(t.load_pending(pend));
    ASSERT_EQ(pend.size(), 2u);
    EXPECT_EQ(pend[0].path, "/data/one.jsonl");
    EXPECT_EQ(pend[1].path, "/data/two.jsonl");
    EXPECT_EQ(pend[1].num_entries, -1);  // a mistyped field takes its default
}

TEST_F(LocalTransportPhase9Test, AddPendingRefusesControlCharactersButTakesSpaces) {
    LocalTransport t(reg_path_, pend_path_);
    EXPECT_TRUE(t.add_pending("/data/my file.jsonl"));
    EXPECT_FALSE(t.add_pending("/data/tab\tfile.jsonl"));
    EXPECT_FALSE(t.add_pending("/data/new\nline.jsonl"));
    EXPECT_FALSE(t.add_pending(std::string("/data/bad\xff.jsonl")));
    EXPECT_FALSE(t.add_pending(""));

    std::vector<PendingEntry> pend;
    ASSERT_TRUE(t.load_pending(pend));
    ASSERT_EQ(pend.size(), 1u);
    EXPECT_EQ(pend[0].path, "/data/my file.jsonl");
}

TEST(RegistryPathProblemTest, AcceptsOrdinaryAndNonAsciiPaths) {
    EXPECT_EQ(registry_path_problem("/data/a.jsonl"), "");
    EXPECT_EQ(registry_path_problem("/data/my file.jsonl"), "");
    EXPECT_EQ(registry_path_problem("/data/caf\xc3\xa9/\xe6\x97\xa5\xe6\x9c\xac.jsonl"), "");
    EXPECT_EQ(registry_path_problem("/data/\xf0\x9f\x98\x80.jsonl"), "");  // U+1F600
    EXPECT_EQ(registry_path_problem("C:\\data\\a.jsonl"), "");
}

TEST(RegistryPathProblemTest, RejectsControlCharactersAndInvalidUtf8) {
    EXPECT_NE(registry_path_problem(""), "");
    EXPECT_NE(registry_path_problem("a\tb"), "");
    EXPECT_NE(registry_path_problem("a\nb"), "");
    EXPECT_NE(registry_path_problem("a\rb"), "");
    EXPECT_NE(registry_path_problem(std::string("a\0b", 3)), "");
    EXPECT_NE(registry_path_problem("a\x7f"), "");
    EXPECT_NE(registry_path_problem("a\xc2\x85"), "");          // U+0085 NEXT LINE (C1 control)
    EXPECT_NE(registry_path_problem("a\xff"), "");              // not a lead byte
    EXPECT_NE(registry_path_problem("a\xc3"), "");              // cut-off sequence
    EXPECT_NE(registry_path_problem("a\xc0\xaf"), "");          // overlong '/'
    EXPECT_NE(registry_path_problem("a\xe0\x80\xaf"), "");      // overlong '/'
    EXPECT_NE(registry_path_problem("a\xed\xa0\x80"), "");      // surrogate U+D800
    EXPECT_NE(registry_path_problem("a\xf4\x90\x80\x80"), "");  // above U+10FFFF
}

// TD-321: a negative count used to mean "assign every pending entry", as 0 does.
TEST_F(LocalTransportPhase9Test, AssignRefusesNegativeCount) {
    seed_pending({"/data/a.jsonl", "/data/b.jsonl", "/data/c.jsonl"});
    LocalTransport t(reg_path_, pend_path_);
    const AssignResult result = t.assign("m", {}, -2);
    EXPECT_EQ(result.assigned, 0);
    std::vector<PendingEntry> pend;
    ASSERT_TRUE(t.load_pending(pend));
    for (const auto& e : pend)
        EXPECT_TRUE(e.model_name.empty()) << e.path;

    EXPECT_EQ(t.assign("m", {}, 2).assigned, 2);  // a positive count still takes the first N
}

