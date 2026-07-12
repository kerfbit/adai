/**
 * DatasetRegistryTests — unit tests for DatasetRegistry (TD-028 Phase 10)
 *
 * Covers:
 *   - DatasetConfig defaults and make_config()
 *   - Static helpers: compute_checksum(), load_conversation_pairs()
 *   - Pending queue: add_file(), add_files(), pending_files(), clear_pending()
 *   - Trained set: is_trained(), trained_files(), mark_trained() (single-run)
 *   - total_samples_trained()
 *   - Persistence: load_registry(), save_registry(), load_pending_list(), save_pending_list()
 *   - Multi-run API: acquire_pending(), release_pending(), mark_trained(run_id,…)
 *
 * All tests are fully offline (local filesystem only).
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "../src/DatasetRegistry.hpp"
#include "../src/Config.hpp"

namespace fs = std::filesystem;

// ============================================================================
// Helpers
// ============================================================================

namespace {

void write_file(const std::string& path, const std::string& content) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path);
    f << content;
}

const std::string kSimplePairs =
    "INPUT: hello\nRESPONSE: world\n\n"
    "INPUT: foo\nRESPONSE: bar\n";

const std::string kJsonlPairs =
    "{\"input\":\"hello\",\"response\":\"world\"}\n"
    "{\"input\":\"foo\",\"response\":\"bar\"}\n";

const std::string kJsonlPairsWithMeta =
    "{\"input\":\"what is ml?\",\"response\":\"machine learning\","
    "\"domain\":\"science\",\"task_type\":\"qa\",\"language\":\"en\","
    "\"quality\":0.9,\"weight\":1.5,\"token_count\":20}\n"
    "{\"input\":\"greet\",\"response\":\"hi\","
    "\"domain\":\"dialogue\",\"task_type\":\"chat\"}\n";

}  // namespace

// ============================================================================
// DatasetConfig defaults
// ============================================================================

TEST(DatasetConfigTest, DefaultValues) {
    DatasetConfig cfg;
    EXPECT_EQ(cfg.session_dir,        "training_sessions");
    EXPECT_EQ(cfg.data_registry_file, "data_registry.txt");
    EXPECT_FALSE(cfg.cache_tokenized_data);
    EXPECT_TRUE(cfg.registry_server_url.empty());
    EXPECT_TRUE(cfg.run_id.empty());
    EXPECT_EQ(cfg.registry_timeout_ms, 5000);
    EXPECT_EQ(cfg.max_files_per_run,   0);
}

TEST(DatasetConfigTest, MakeConfigCopiesFields) {
    adai::ServiceConfig svc;
    svc.session_dir          = "/tmp/adai_dr_test_sessions";
    svc.registry_server_url  = "http://reg:8082";
    svc.run_group            = "my-group";
    svc.run_id               = "host_1234";
    svc.registry_timeout_ms  = 3000;

    const DatasetConfig cfg = DatasetRegistry::make_config(svc);

    EXPECT_EQ(cfg.session_dir,         svc.session_dir);
    EXPECT_EQ(cfg.registry_server_url, svc.registry_server_url);
    EXPECT_EQ(cfg.run_group,           svc.run_group);
    EXPECT_EQ(cfg.run_id,              svc.run_id);
    EXPECT_EQ(cfg.registry_timeout_ms, svc.registry_timeout_ms);
}

TEST(DatasetConfigTest, MakeConfigEmptySessionDirKeepsDefault) {
    adai::ServiceConfig svc;
    svc.session_dir = "";
    const DatasetConfig cfg = DatasetRegistry::make_config(svc);
    EXPECT_EQ(cfg.session_dir, "training_sessions");
}

// ============================================================================
// compute_checksum
// ============================================================================

TEST(DatasetRegistryChecksumTest, ReturnsMissingForAbsentFile) {
    EXPECT_EQ(DatasetRegistry::compute_checksum("/nonexistent/adai_no_such_file_xyz.txt"),
              "MISSING");
}

TEST(DatasetRegistryChecksumTest, ReturnsNonEmptyForExistingFile) {
    const fs::path tmp = fs::temp_directory_path() / "adai_chksum_test.txt";
    write_file(tmp.string(), "hello");
    const std::string cs = DatasetRegistry::compute_checksum(tmp.string());
    EXPECT_FALSE(cs.empty());
    EXPECT_NE(cs, "MISSING");
    fs::remove(tmp);
}

TEST(DatasetRegistryChecksumTest, DifferentSizesProduceDifferentChecksums) {
    const fs::path a = fs::temp_directory_path() / "adai_chk_a.txt";
    const fs::path b = fs::temp_directory_path() / "adai_chk_b.txt";
    write_file(a.string(), "short");
    write_file(b.string(), "a longer string with significantly more content here");
    EXPECT_NE(DatasetRegistry::compute_checksum(a.string()),
              DatasetRegistry::compute_checksum(b.string()));
    fs::remove(a);
    fs::remove(b);
}

// ============================================================================
// load_conversation_pairs
// ============================================================================

TEST(DatasetRegistryParserTest, ReturnsZeroForMissingFile) {
    std::vector<ConversationPair> pairs;
    EXPECT_EQ(DatasetRegistry::load_conversation_pairs("/no/such/adai_file.txt", pairs), 0);
    EXPECT_TRUE(pairs.empty());
}

TEST(DatasetRegistryParserTest, ParsesInputResponsePairs) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_test.txt";
    write_file(tmp.string(), kSimplePairs);

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs);
    ASSERT_EQ(n, 2);
    EXPECT_EQ(pairs[0].input,    "hello");
    EXPECT_EQ(pairs[0].response, "world");
    EXPECT_EQ(pairs[1].input,    "foo");
    EXPECT_EQ(pairs[1].response, "bar");
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, ParsesPairsWithoutBlankSeparator) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_nosep.txt";
    write_file(tmp.string(),
        "INPUT: q1\nRESPONSE: a1\nINPUT: q2\nRESPONSE: a2\n");

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs);
    ASSERT_EQ(n, 2);
    EXPECT_EQ(pairs[0].input, "q1");
    EXPECT_EQ(pairs[1].input, "q2");
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, EmptyFileReturnsZeroPairs) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_empty.txt";
    write_file(tmp.string(), "");
    std::vector<ConversationPair> pairs;
    EXPECT_EQ(DatasetRegistry::load_conversation_pairs(tmp.string(), pairs), 0);
    fs::remove(tmp);
}

// ── JSONL format ─────────────────────────────────────────────────────────────

TEST(DatasetRegistryParserTest, ParsesJsonlPairs) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_jsonl.txt";
    write_file(tmp.string(), kJsonlPairs);

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs);
    ASSERT_EQ(n, 2);
    EXPECT_EQ(pairs[0].input,    "hello");
    EXPECT_EQ(pairs[0].response, "world");
    EXPECT_EQ(pairs[1].input,    "foo");
    EXPECT_EQ(pairs[1].response, "bar");
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, JsonlPairsPreserveMetadata) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_jsonl_meta.txt";
    write_file(tmp.string(), kJsonlPairsWithMeta);

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs);
    ASSERT_EQ(n, 2);

    // First pair: full metadata
    EXPECT_EQ(pairs[0].input,    "what is ml?");
    EXPECT_EQ(pairs[0].response, "machine learning");
    EXPECT_EQ(pairs[0].meta.domain,    "science");
    EXPECT_EQ(pairs[0].meta.task_type, "qa");
    EXPECT_EQ(pairs[0].meta.language,  "en");
    EXPECT_NEAR(pairs[0].meta.quality, 0.9f, 1e-4f);
    EXPECT_NEAR(pairs[0].meta.weight,  1.5f, 1e-4f);
    EXPECT_EQ(pairs[0].meta.token_count, 20);

    // Second pair: partial metadata — unset fields use sentinels
    EXPECT_EQ(pairs[1].meta.domain,    "dialogue");
    EXPECT_EQ(pairs[1].meta.task_type, "chat");
    EXPECT_LT(pairs[1].meta.quality, 0.0f);   // sentinel
    EXPECT_LT(pairs[1].meta.token_count, 0);   // sentinel
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, JsonlSkipsLinesWithoutInputField) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_jsonl_noinput.txt";
    write_file(tmp.string(),
        "{\"input\":\"valid\",\"response\":\"yes\"}\n"
        "{\"response\":\"no input key\"}\n"
        "{\"input\":\"also valid\",\"response\":\"yes\"}\n");

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs);
    EXPECT_EQ(n, 2);
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, LegacyMetaSentinelsSet) {
    // Legacy INPUT:/RESPONSE: format — meta fields should remain at sentinel defaults
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_legacy_meta.txt";
    write_file(tmp.string(), kSimplePairs);

    std::vector<ConversationPair> pairs;
    ASSERT_EQ(DatasetRegistry::load_conversation_pairs(tmp.string(), pairs), 2);
    EXPECT_LT(pairs[0].meta.quality, 0.0f);
    EXPECT_LT(pairs[0].meta.token_count, 0);
    EXPECT_TRUE(pairs[0].meta.domain.empty());
    fs::remove(tmp);
}

// ============================================================================
// Fixture
// ============================================================================

class DatasetRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_dir_     = fs::temp_directory_path() / "adai_registry_unit_test";
        fs::remove_all(tmp_dir_);
        fs::create_directories(tmp_dir_);
        session_dir_ = (tmp_dir_ / "sessions").string();
        data_file_   = (tmp_dir_ / "training.txt").string();
        write_file(data_file_, kSimplePairs);
    }

    void TearDown() override { fs::remove_all(tmp_dir_); }

    DatasetConfig make_cfg() const {
        DatasetConfig cfg;
        cfg.session_dir = session_dir_;
        return cfg;
    }

    fs::path    tmp_dir_;
    std::string session_dir_;
    std::string data_file_;
};

// ============================================================================
// Pending queue
// ============================================================================

TEST_F(DatasetRegistryTest, AddFileMissingPathReturnsFalse) {
    DatasetRegistry reg(make_cfg());
    EXPECT_FALSE(reg.add_file("/absolutely/not/a/real/adai_file.txt"));
    EXPECT_TRUE(reg.pending_files().empty());
}

TEST_F(DatasetRegistryTest, AddFileSucceedsForRealFile) {
    DatasetRegistry reg(make_cfg());
    EXPECT_TRUE(reg.add_file(data_file_));
    const auto pending = reg.pending_files();
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0], data_file_);
}

TEST_F(DatasetRegistryTest, AddFilePersistsToDiskOnSecondLoad) {
    {
        DatasetRegistry reg(make_cfg());
        ASSERT_TRUE(reg.add_file(data_file_));
    }
    DatasetRegistry reg2(make_cfg());
    ASSERT_TRUE(reg2.load_pending_list());
    const auto pending = reg2.pending_files();
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0], data_file_);
}

TEST_F(DatasetRegistryTest, AddFileSkipsAlreadyTrainedFile) {
    DatasetRegistry reg(make_cfg());
    reg.mark_trained({data_file_}, {10});
    EXPECT_FALSE(reg.add_file(data_file_));
    EXPECT_TRUE(reg.pending_files().empty());
}

TEST_F(DatasetRegistryTest, AddFilesReturnsTrueIfAtLeastOneAdded) {
    DatasetRegistry reg(make_cfg());
    EXPECT_TRUE(reg.add_files({"/missing/file.txt", data_file_}));
    EXPECT_EQ(reg.pending_files().size(), 1u);
}

TEST_F(DatasetRegistryTest, AddFilesReturnsFalseIfNoneAdded) {
    DatasetRegistry reg(make_cfg());
    EXPECT_FALSE(reg.add_files({"/missing1_adai.txt", "/missing2_adai.txt"}));
}

TEST_F(DatasetRegistryTest, ClearPendingEmptiesInMemoryList) {
    DatasetRegistry reg(make_cfg());
    reg.add_file(data_file_);
    ASSERT_FALSE(reg.pending_files().empty());
    reg.clear_pending();
    EXPECT_TRUE(reg.pending_files().empty());
}

// ============================================================================
// Trained set
// ============================================================================

TEST_F(DatasetRegistryTest, IsTrainedReturnsFalseInitially) {
    DatasetRegistry reg(make_cfg());
    EXPECT_FALSE(reg.is_trained(data_file_));
}

TEST_F(DatasetRegistryTest, IsTrainedTrueAfterMarkTrained) {
    DatasetRegistry reg(make_cfg());
    reg.mark_trained({data_file_}, {42});
    EXPECT_TRUE(reg.is_trained(data_file_));
}

TEST_F(DatasetRegistryTest, TrainedFilesContainsAllMarkedPaths) {
    const std::string file2 = (tmp_dir_ / "training2.txt").string();
    write_file(file2, "INPUT: x\nRESPONSE: y\n");

    DatasetRegistry reg(make_cfg());
    reg.mark_trained({data_file_, file2}, {5, 10});

    const auto trained = reg.trained_files();
    ASSERT_EQ(trained.size(), 2u);
    EXPECT_NE(std::find(trained.begin(), trained.end(), data_file_), trained.end());
    EXPECT_NE(std::find(trained.begin(), trained.end(), file2),      trained.end());
}

TEST_F(DatasetRegistryTest, MarkTrainedSkipsDuplicateEntry) {
    DatasetRegistry reg(make_cfg());
    reg.mark_trained({data_file_}, {10});
    reg.mark_trained({data_file_}, {20});  // second call — should be a no-op
    EXPECT_EQ(reg.trained_files().size(), 1u);
}

// ============================================================================
// total_samples_trained
// ============================================================================

TEST_F(DatasetRegistryTest, TotalSamplesTrainedZeroWithNoRegistry) {
    DatasetRegistry reg(make_cfg());
    EXPECT_EQ(reg.total_samples_trained(), 0);
}

TEST_F(DatasetRegistryTest, TotalSamplesTrainedSumsCorrectly) {
    const std::string file2 = (tmp_dir_ / "training2.txt").string();
    write_file(file2, "INPUT: x\nRESPONSE: y\n");

    DatasetRegistry reg(make_cfg());
    reg.mark_trained({data_file_, file2}, {15, 25});
    EXPECT_EQ(reg.total_samples_trained(), 40);
}

// ============================================================================
// Persistence round-trips
// ============================================================================

TEST_F(DatasetRegistryTest, LoadRegistryReturnsFalseWhenAbsent) {
    DatasetRegistry reg(make_cfg());
    EXPECT_FALSE(reg.load_registry());
}

TEST_F(DatasetRegistryTest, LoadPendingListReturnsFalseWhenAbsent) {
    DatasetRegistry reg(make_cfg());
    EXPECT_FALSE(reg.load_pending_list());
}

TEST_F(DatasetRegistryTest, LoadSaveRegistryRoundTrip) {
    {
        DatasetRegistry reg(make_cfg());
        reg.mark_trained({data_file_}, {99});
    }
    DatasetRegistry reg2(make_cfg());
    ASSERT_TRUE(reg2.load_registry());
    EXPECT_TRUE(reg2.is_trained(data_file_));
    EXPECT_EQ(reg2.total_samples_trained(), 99);
}

TEST_F(DatasetRegistryTest, LoadPendingListRoundTrip) {
    {
        DatasetRegistry reg(make_cfg());
        ASSERT_TRUE(reg.add_file(data_file_));
    }
    DatasetRegistry reg2(make_cfg());
    ASSERT_TRUE(reg2.load_pending_list());
    const auto pending = reg2.pending_files();
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0], data_file_);
}

// ============================================================================
// Multi-run API — acquire / release / mark_trained(run_id, …)
// ============================================================================

TEST_F(DatasetRegistryTest, AcquirePendingReturnsFilesAndUpdatesPending) {
    {
        DatasetRegistry reg(make_cfg());
        reg.add_file(data_file_);
    }
    DatasetRegistry reg2(make_cfg());
    auto resp = reg2.acquire_pending("run-a");
    ASSERT_EQ(resp.files.size(), 1u);
    EXPECT_EQ(resp.files[0].registry_path, data_file_);
    EXPECT_TRUE(resp.ftp_server_host.empty());  // LocalTransport: no FTP
    EXPECT_FALSE(reg2.pending_files().empty());  // reflected in-memory
}

TEST_F(DatasetRegistryTest, AcquirePendingEmptyWhenNoneAvailable) {
    DatasetRegistry reg(make_cfg());
    EXPECT_TRUE(reg.acquire_pending("run-a").files.empty());
}

TEST_F(DatasetRegistryTest, ReleasePendingRestoresFileToPool) {
    {
        DatasetRegistry reg(make_cfg());
        reg.add_file(data_file_);
    }
    DatasetRegistry reg2(make_cfg());
    auto resp = reg2.acquire_pending("run-a");
    ASSERT_EQ(resp.files.size(), 1u);

    // Release: in-memory pending_ is cleared
    reg2.release_pending("run-a", resp.registry_paths());
    EXPECT_TRUE(reg2.pending_files().empty());

    // A subsequent acquire by a different run finds the file again
    DatasetRegistry reg3(make_cfg());
    EXPECT_EQ(reg3.acquire_pending("run-b").files.size(), 1u);
}

TEST_F(DatasetRegistryTest, MarkTrainedWithRunIdCommitsAndClearsPending) {
    {
        DatasetRegistry reg(make_cfg());
        reg.add_file(data_file_);
    }
    DatasetRegistry reg2(make_cfg());
    auto resp = reg2.acquire_pending("run-a");
    ASSERT_FALSE(resp.files.empty());

    reg2.mark_trained("run-a", resp.registry_paths(), {7});

    // In-memory state
    EXPECT_TRUE(reg2.is_trained(data_file_));
    EXPECT_TRUE(reg2.pending_files().empty());

    // On-disk state
    DatasetRegistry reg3(make_cfg());
    ASSERT_TRUE(reg3.load_registry());
    EXPECT_TRUE(reg3.is_trained(data_file_));
    EXPECT_EQ(reg3.total_samples_trained(), 7);

    // Pending file should be empty — no new acquire possible
    DatasetRegistry reg4(make_cfg());
    EXPECT_TRUE(reg4.acquire_pending("run-c").files.empty());
}
