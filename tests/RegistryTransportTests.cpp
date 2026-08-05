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
    EXPECT_EQ(t.upload_file("/some/local/file.jsonl"), "");
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
