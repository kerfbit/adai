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
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "../src/Config.hpp"
#include "../src/DatasetRegistry.hpp"
#include "../src/RegistryTransport.hpp"

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

// Fake RegistryTransport (Phase 11) — wraps a real LocalTransport for the
// flat-file operations DatasetRegistry needs at construction/load time, but
// gives the test full control over the new server-side-fetch methods so the
// remote_fetch_*/remote_upload pass-throughs can be tested without a real
// registry_server.
class FakeFetchTransport : public RegistryTransport {
   public:
    FakeFetchTransport(std::string registry_path, std::string pending_path)
        : inner_(std::move(registry_path), std::move(pending_path)) {}

    bool load_registry(std::vector<DataVersion>& out) override {
        return inner_.load_registry(out);
    }
    bool save_registry(const std::vector<DataVersion>& entries) override {
        return inner_.save_registry(entries);
    }
    bool load_pending(std::vector<PendingEntry>& out) override {
        return inner_.load_pending(out);
    }
    bool save_pending(const std::vector<PendingEntry>& entries) override {
        return inner_.save_pending(entries);
    }
    AcquireResponse acquire(const std::string& run_id, int max_files,
                            const std::string& model_name = "") override {
        auto resp = inner_.acquire(run_id, max_files, model_name);
        resp.claim_lease_seconds = lease_seconds;  // TD-494: as a registry_server would report
        return resp;
    }
    // TD-494: count renewals (called from DatasetRegistry's renewer thread).
    int lease_seconds = 0;
    std::atomic<int> renew_calls{0};
    int renew_claims(const std::string& /*run_id*/) override {
        ++renew_calls;
        return 1;
    }
    void release(const std::string& run_id, const std::vector<std::string>& paths,
                const std::vector<SegmentTarget>& segments = {}) override {
        inner_.release(run_id, paths, segments);
    }
    void commit_trained(const std::string& run_id, const std::vector<DataVersion>& new_entries,
                        const std::vector<std::string>& trained_paths,
                        const std::vector<SegmentTarget>& trained_segments = {}) override {
        inner_.commit_trained(run_id, new_entries, trained_paths, trained_segments);
    }
    bool add_pending(const std::string& path, int segment_start = -1,
                     int segment_count = -1) override {
        return inner_.add_pending(path, segment_start, segment_count);
    }
    std::string next_session(const std::string& model_name, const std::string& run_id) override {
        return inner_.next_session(model_name, run_id);
    }
    AssignResult assign(const std::string& model_name, const std::vector<std::string>& paths,
                        int count, const std::vector<SegmentTarget>& segments = {}) override {
        return inner_.assign(model_name, paths, count, segments);
    }
    UnassignResult unassign(const std::string& model_name, const std::vector<std::string>& paths,
                            bool force, const std::vector<SegmentTarget>& segments = {}) override {
        return inner_.unassign(model_name, paths, force, segments);
    }
    DeleteResult delete_paths(const std::vector<std::string>& paths, bool force, bool delete_files,
                              const std::vector<SegmentTarget>& segments = {}) override {
        return inner_.delete_paths(paths, force, delete_files, segments);
    }
    RequeueResult requeue_trained(const std::vector<std::string>& paths, bool all,
                                  const std::string& model_name = "") override {
        return inner_.requeue_trained(paths, all, model_name);
    }

    // Controllable Phase 11 behaviour:
    std::string next_fetch_result;  // returned by fetch_gutenberg/fetch_huggingface
    std::string next_upload_result;
    int fetch_gutenberg_calls = 0;
    int fetch_huggingface_calls = 0;
    int upload_calls = 0;
    int last_book_id = -1;
    std::string last_dataset_id;
    std::string last_model_name;
    std::string last_upload_path;

    std::string fetch_gutenberg(int book_id, int /*num_pairs*/,
                                const std::string& model_name) override {
        ++fetch_gutenberg_calls;
        last_book_id = book_id;
        last_model_name = model_name;
        return next_fetch_result;
    }
    std::string fetch_huggingface(const std::string& dataset_id, int /*num_pairs*/,
                                  const std::string& /*split*/, const std::string& /*input_field*/,
                                  const std::string& /*output_field*/,
                                  const std::string& model_name) override {
        ++fetch_huggingface_calls;
        last_dataset_id = dataset_id;
        last_model_name = model_name;
        return next_fetch_result;
    }
    std::string upload_file(const std::string& local_path) override {
        ++upload_calls;
        last_upload_path = local_path;
        return next_upload_result;
    }

   private:
    LocalTransport inner_;
};

}  // namespace

// ============================================================================
// DatasetConfig defaults
// ============================================================================

TEST(DatasetConfigTest, DefaultValues) {
    DatasetConfig cfg;
    EXPECT_EQ(cfg.session_dir, "training_sessions");
    EXPECT_EQ(cfg.data_registry_file, "data_registry.txt");
    EXPECT_FALSE(cfg.cache_tokenized_data);
    EXPECT_TRUE(cfg.registry_server_url.empty());
    EXPECT_TRUE(cfg.run_id.empty());
    EXPECT_EQ(cfg.registry_timeout_ms, 5000);
    EXPECT_EQ(cfg.max_files_per_run, 0);
    EXPECT_TRUE(cfg.dataset_kind.empty());
}

TEST(DatasetConfigTest, MakeConfigCopiesFields) {
    adai::ServiceConfig svc;
    svc.session_dir = "/tmp/adai_dr_test_sessions";
    svc.registry_server_url = "http://reg:8082";
    svc.run_group = "my-group";
    svc.run_id = "host_1234";
    svc.registry_timeout_ms = 3000;
    svc.dataset_kind = "world_model";

    const DatasetConfig cfg = DatasetRegistry::make_config(svc);

    EXPECT_EQ(cfg.session_dir, svc.session_dir);
    EXPECT_EQ(cfg.registry_server_url, svc.registry_server_url);
    EXPECT_EQ(cfg.run_group, svc.run_group);
    EXPECT_EQ(cfg.run_id, svc.run_id);
    EXPECT_EQ(cfg.registry_timeout_ms, svc.registry_timeout_ms);
    EXPECT_EQ(cfg.dataset_kind, svc.dataset_kind);
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
    EXPECT_EQ(pairs[0].input, "hello");
    EXPECT_EQ(pairs[0].response, "world");
    EXPECT_EQ(pairs[1].input, "foo");
    EXPECT_EQ(pairs[1].response, "bar");
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, ParsesPairsWithoutBlankSeparator) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_nosep.txt";
    write_file(tmp.string(), "INPUT: q1\nRESPONSE: a1\nINPUT: q2\nRESPONSE: a2\n");

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
    EXPECT_EQ(pairs[0].input, "hello");
    EXPECT_EQ(pairs[0].response, "world");
    EXPECT_EQ(pairs[1].input, "foo");
    EXPECT_EQ(pairs[1].response, "bar");
    fs::remove(tmp);
}

// ── TD-205: row-range segment reads ────────────────────────────────────────

const std::string kFiveJsonlPairs =
    "{\"input\":\"q0\",\"response\":\"a0\"}\n"
    "{\"input\":\"q1\",\"response\":\"a1\"}\n"
    "{\"input\":\"q2\",\"response\":\"a2\"}\n"
    "{\"input\":\"q3\",\"response\":\"a3\"}\n"
    "{\"input\":\"q4\",\"response\":\"a4\"}\n";

TEST(DatasetRegistryParserTest, CountPairsReturnsTotalWithoutMaterializingThem) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_count.txt";
    write_file(tmp.string(), kFiveJsonlPairs);
    EXPECT_EQ(DatasetRegistry::count_pairs(tmp.string()), 5);
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, RangedReadReturnsExactSlice) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_ranged.txt";
    write_file(tmp.string(), kFiveJsonlPairs);

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs, 1, 2);
    ASSERT_EQ(n, 2);
    EXPECT_EQ(pairs[0].input, "q1");
    EXPECT_EQ(pairs[1].input, "q2");
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, RangedReadAtStartMatchesFirstNPairs) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_ranged_start.txt";
    write_file(tmp.string(), kFiveJsonlPairs);

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs, 0, 3);
    ASSERT_EQ(n, 3);
    EXPECT_EQ(pairs[0].input, "q0");
    EXPECT_EQ(pairs[2].input, "q2");
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, RangedReadPastEndOfFileReturnsFewerPairsNotWholeFile) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_ranged_oob.txt";
    write_file(tmp.string(), kFiveJsonlPairs);

    std::vector<ConversationPair> pairs;
    // Only 2 pairs exist from index 4 onward (just q4), even though 100 were requested —
    // must NOT silently fall back to loading the whole 5-pair file.
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs, 4, 100);
    ASSERT_EQ(n, 1);
    EXPECT_EQ(pairs[0].input, "q4");
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, RangedReadFullyPastEndReturnsZeroPairs) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_ranged_oob2.txt";
    write_file(tmp.string(), kFiveJsonlPairs);

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs, 100, 5);
    EXPECT_EQ(n, 0);
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, RangedReadOnLegacyFormatFallsBackToWholeFile) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_ranged_legacy.txt";
    write_file(tmp.string(), kSimplePairs);  // legacy INPUT:/RESPONSE: format, 2 pairs

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs, 1, 1);
    EXPECT_EQ(n, 2) << "legacy format can't be range-sliced — must fall back to the whole file";
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, NegativeSegmentStartBehavesLikeTheTwoArgOverload) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_ranged_negative.txt";
    write_file(tmp.string(), kFiveJsonlPairs);

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs, -1, -1);
    EXPECT_EQ(n, 5);
    fs::remove(tmp);
}

TEST(DatasetRegistryParserTest, JsonlPairsPreserveMetadata) {
    const fs::path tmp = fs::temp_directory_path() / "adai_parser_jsonl_meta.txt";
    write_file(tmp.string(), kJsonlPairsWithMeta);

    std::vector<ConversationPair> pairs;
    const int n = DatasetRegistry::load_conversation_pairs(tmp.string(), pairs);
    ASSERT_EQ(n, 2);

    // First pair: full metadata
    EXPECT_EQ(pairs[0].input, "what is ml?");
    EXPECT_EQ(pairs[0].response, "machine learning");
    EXPECT_EQ(pairs[0].meta.domain, "science");
    EXPECT_EQ(pairs[0].meta.task_type, "qa");
    EXPECT_EQ(pairs[0].meta.language, "en");
    EXPECT_NEAR(pairs[0].meta.quality, 0.9f, 1e-4f);
    EXPECT_NEAR(pairs[0].meta.weight, 1.5f, 1e-4f);
    EXPECT_EQ(pairs[0].meta.token_count, 20);

    // Second pair: partial metadata — unset fields use sentinels
    EXPECT_EQ(pairs[1].meta.domain, "dialogue");
    EXPECT_EQ(pairs[1].meta.task_type, "chat");
    EXPECT_LT(pairs[1].meta.quality, 0.0f);   // sentinel
    EXPECT_LT(pairs[1].meta.token_count, 0);  // sentinel
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
        tmp_dir_ = fs::temp_directory_path() / "adai_registry_unit_test";
        fs::remove_all(tmp_dir_);
        fs::create_directories(tmp_dir_);
        session_dir_ = (tmp_dir_ / "sessions").string();
        data_file_ = (tmp_dir_ / "training.txt").string();
        write_file(data_file_, kSimplePairs);
    }

    void TearDown() override {
        fs::remove_all(tmp_dir_);
    }

    DatasetConfig make_cfg() const {
        DatasetConfig cfg;
        cfg.session_dir = session_dir_;
        return cfg;
    }

    fs::path tmp_dir_;
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

// ============================================================================
// TD-202: dataset_kind sub-pools (LocalTransport)
// ============================================================================

TEST_F(DatasetRegistryTest, DatasetKindLandsPendingFileUnderKindSubdirectory) {
    DatasetConfig cfg = make_cfg();
    cfg.dataset_kind = "world_model";
    DatasetRegistry reg(cfg);
    ASSERT_TRUE(reg.add_file(data_file_));
    EXPECT_TRUE(fs::exists(fs::path(session_dir_) / "world_model" / "pending_files.txt"));
    EXPECT_FALSE(fs::exists(fs::path(session_dir_) / "pending_files.txt"))
        << "an empty dataset_kind's pool must not be touched by a kind-scoped registry";
}

TEST_F(DatasetRegistryTest, DatasetKindPoolsAreIsolatedFromEachOtherAndFromLegacy) {
    DatasetConfig legacy_cfg = make_cfg();  // dataset_kind left empty — today's shared pool
    DatasetConfig wm_cfg = make_cfg();
    wm_cfg.dataset_kind = "world_model";
    DatasetConfig chatbot_cfg = make_cfg();
    chatbot_cfg.dataset_kind = "chatbot";

    const std::string other_file = (tmp_dir_ / "other.txt").string();
    write_file(other_file, kSimplePairs);
    const std::string third_file = (tmp_dir_ / "third.txt").string();
    write_file(third_file, kSimplePairs);

    DatasetRegistry legacy(legacy_cfg);
    ASSERT_TRUE(legacy.add_file(data_file_));
    DatasetRegistry wm(wm_cfg);
    ASSERT_TRUE(wm.add_file(other_file));
    DatasetRegistry chatbot(chatbot_cfg);
    ASSERT_TRUE(chatbot.add_file(third_file));

    // Each pool sees only its own file — this is the actual TD-202 leakage fix.
    EXPECT_EQ(legacy.pending_files(), std::vector<std::string>{data_file_});
    EXPECT_EQ(wm.pending_files(), std::vector<std::string>{other_file});
    EXPECT_EQ(chatbot.pending_files(), std::vector<std::string>{third_file});
}

TEST_F(DatasetRegistryTest, AddPendingPathUncheckedSkipsFilesystemExistenceCheck) {
    DatasetConfig cfg = make_cfg();
    cfg.dataset_kind = "world_model";
    DatasetRegistry reg(cfg);
    // Deliberately a path that does not exist on this host's filesystem — add_file() would
    // reject it, but add_pending_path_unchecked() (used by dataset_manager migrate) must not,
    // since a migrated entry's path may only exist on a remote registry_server's own storage.
    EXPECT_TRUE(reg.add_pending_path_unchecked("/no/such/file/on/this/host.jsonl"));
    EXPECT_EQ(reg.pending_files(), std::vector<std::string>{"/no/such/file/on/this/host.jsonl"});
}

TEST_F(DatasetRegistryTest, AddPendingPathUncheckedRejectsDuplicate) {
    DatasetConfig cfg = make_cfg();
    DatasetRegistry reg(cfg);
    ASSERT_TRUE(reg.add_pending_path_unchecked(data_file_));
    EXPECT_FALSE(reg.add_pending_path_unchecked(data_file_));
    EXPECT_EQ(reg.pending_files().size(), 1u);
}

TEST_F(DatasetRegistryTest, AddPendingPathUncheckedRejectsAlreadyTrainedFile) {
    // Regression: add_pending_path_unchecked() must reject an already-trained path exactly
    // like add_file() does — this is what protects `dataset_manager migrate` from re-queuing
    // a file that's already trained in the destination kind's own trained registry.
    DatasetConfig cfg = make_cfg();
    DatasetRegistry reg(cfg);
    reg.mark_trained({data_file_}, {10});
    EXPECT_FALSE(reg.add_pending_path_unchecked(data_file_));
    EXPECT_TRUE(reg.pending_files().empty());
}

// requeue_trained(): trained entries go back to the pending queue (needed to retrain from
// scratch, and the operation `incremental_trainer reset --keep-data` performs).
TEST_F(DatasetRegistryTest, RequeueTrainedMovesTrainedEntryBackToPending) {
    DatasetRegistry reg(make_cfg());
    reg.mark_trained({data_file_}, {10});
    ASSERT_TRUE(reg.is_trained(data_file_));

    const RequeueResult r = reg.requeue_trained({data_file_}, false, "model-a");
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.requeued, 1);
    ASSERT_EQ(r.details.size(), 1u);
    EXPECT_EQ(r.details[0].status, "requeued");
    EXPECT_FALSE(reg.is_trained(data_file_));
    ASSERT_EQ(reg.pending_files().size(), 1u);
    EXPECT_EQ(reg.pending_files()[0], data_file_);
    EXPECT_EQ(reg.pending_entries()[0].model_name, "model-a");

    // A fresh registry sees it too (persisted, not just in memory), and can acquire it.
    DatasetRegistry fresh(make_cfg());
    fresh.load_registry();
    fresh.load_pending_list();
    EXPECT_FALSE(fresh.is_trained(data_file_));
    EXPECT_EQ(fresh.pending_files().size(), 1u);
}

TEST_F(DatasetRegistryTest, RequeueTrainedAllAndEdgeCases) {
    DatasetRegistry reg(make_cfg());
    EXPECT_EQ(reg.requeue_trained({}, false).requeued, 0) << "no paths and no --all is a no-op";

    reg.mark_trained({data_file_}, {10});
    const RequeueResult missing = reg.requeue_trained({"/not/trained.jsonl"}, false);
    EXPECT_EQ(missing.not_found, 1);
    EXPECT_TRUE(reg.is_trained(data_file_));

    const RequeueResult all = reg.requeue_trained({}, true);
    EXPECT_EQ(all.requeued, 1);
    EXPECT_FALSE(reg.is_trained(data_file_));
    EXPECT_EQ(reg.pending_entries()[0].model_name, "") << "no model given: unassigned";

}

// TD-333: one file split into three TD-205 segments. A failed pass releases exactly the claimed
// segments (bare paths used to release none of them), and segments trained in separate passes
// are each recorded with their own sample count.
TEST_F(DatasetRegistryTest, ReleasingAnAcquiredBatchFreesItsSegments) {
    DatasetRegistry reg(make_cfg());
    reg.load_pending_list();
    ASSERT_TRUE(reg.add_file(data_file_, 0, 10));
    ASSERT_TRUE(reg.add_file(data_file_, 10, 10));
    ASSERT_TRUE(reg.add_file(data_file_, 20, 10));

    const AcquireResponse a = reg.acquire_pending("run-A");
    ASSERT_EQ(a.files.size(), 3u);
    EXPECT_TRUE(reg.acquire_pending("run-B").files.empty()) << "all claimed by run-A";
    reg.release_pending("run-A", a.file_ranges());

    DatasetRegistry fresh(make_cfg());
    fresh.load_pending_list();
    EXPECT_EQ(fresh.acquire_pending("run-B").files.size(), 3u)
        << "a failed pass's segments must be claimable by the next run";
}

TEST_F(DatasetRegistryTest, SegmentsTrainedSeparatelyAreEachRecorded) {
    DatasetRegistry reg(make_cfg());
    reg.load_pending_list();
    ASSERT_TRUE(reg.add_file(data_file_, 0, 10));
    ASSERT_TRUE(reg.add_file(data_file_, 10, 10));
    ASSERT_TRUE(reg.add_file(data_file_, 20, 10));

    const AcquireResponse first = reg.acquire_pending("run-1", 1);
    ASSERT_EQ(first.files.size(), 1u);
    reg.mark_trained("run-1", first.file_ranges(), {10});
    const AcquireResponse second = reg.acquire_pending("run-1", 1);
    ASSERT_EQ(second.files.size(), 1u);
    reg.mark_trained("run-1", second.file_ranges(), {7});

    DatasetRegistry fresh(make_cfg());
    fresh.load_registry();
    fresh.load_pending_list();
    std::vector<DataVersion> records;
    LocalTransport(fs::path(make_cfg().session_dir) / "data_registry.txt",
                   fs::path(make_cfg().session_dir) / "pending_files.txt")
        .load_registry(records);
    ASSERT_EQ(records.size(), 2u) << "one record per trained segment";
    EXPECT_EQ(records[0].num_samples + records[1].num_samples, 17);
    EXPECT_TRUE(fresh.is_trained(data_file_, first.files[0].segment_start,
                                 first.files[0].segment_count));
    EXPECT_TRUE(fresh.is_trained(data_file_));
    EXPECT_EQ(fresh.pending_files().size(), 1u) << "the untrained segment is still pending";
    EXPECT_FALSE(fresh.is_trained(data_file_, fresh.pending_entries()[0].segment_start,
                                  fresh.pending_entries()[0].segment_count));

    // Re-queuing restores each trained segment as that segment.
    const RequeueResult rq = fresh.requeue_trained({data_file_}, false);
    EXPECT_EQ(rq.requeued, 2);
    EXPECT_EQ(fresh.pending_entries().size(), 3u);
    for (const auto& e : fresh.pending_entries()) EXPECT_GE(e.segment_start, 0);
}

TEST(LocalTransportSegmentRecordTest, RegistryFileKeepsSegmentsAndReadsLegacyLines) {
    const fs::path dir = fs::temp_directory_path() / "adai_registry_segments";
    fs::remove_all(dir);
    fs::create_directories(dir);
    {
        std::ofstream legacy(dir / "data_registry.txt");
        legacy << "# legacy\n/data/old.jsonl abc 5 1 - 2026-01-01T00:00:00Z manual\n";
    }
    LocalTransport t((dir / "data_registry.txt").string(), (dir / "pending_files.txt").string());
    std::vector<DataVersion> reg;
    ASSERT_TRUE(t.load_registry(reg));
    ASSERT_EQ(reg.size(), 1u);
    EXPECT_EQ(reg[0].segment_start, -1) << "a legacy line is a whole-file record";

    DataVersion seg;
    seg.data_file = "/data/new.jsonl";
    seg.trained = true;
    seg.segment_start = 20;
    seg.segment_count = 10;
    reg.push_back(seg);
    ASSERT_TRUE(t.save_registry(reg));
    std::vector<DataVersion> back;
    ASSERT_TRUE(t.load_registry(back));
    ASSERT_EQ(back.size(), 2u);
    EXPECT_EQ(back[1].segment_start, 20);
    EXPECT_EQ(back[1].segment_count, 10);
    EXPECT_EQ(back[0].source, "manual");
    fs::remove_all(dir);
}

// A file both trained and pending (e.g. re-added after training) only loses its trained record;
// it never gets a second pending entry.
TEST(LocalTransportRequeueTest, AlreadyPendingFileOnlyLosesItsTrainedRecord) {
    const fs::path dir = fs::temp_directory_path() / "adai_requeue_already_pending";
    fs::remove_all(dir);
    fs::create_directories(dir);
    LocalTransport t((dir / "data_registry.txt").string(), (dir / "pending_files.txt").string());
    DataVersion dv;
    dv.data_file = "/data/a.jsonl";
    dv.trained = true;
    ASSERT_TRUE(t.save_registry({dv}));
    PendingEntry pe;
    pe.path = "/data/a.jsonl";
    ASSERT_TRUE(t.save_pending({pe}));

    const RequeueResult r = t.requeue_trained({"/data/a.jsonl"}, false);
    EXPECT_EQ(r.already_pending, 1);
    EXPECT_EQ(r.requeued, 0);
    std::vector<DataVersion> reg;
    std::vector<PendingEntry> pending;
    t.load_registry(reg);
    t.load_pending(pending);
    EXPECT_TRUE(reg.empty());
    EXPECT_EQ(pending.size(), 1u);
    fs::remove_all(dir);
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
    EXPECT_NE(std::find(trained.begin(), trained.end(), file2), trained.end());
}

TEST_F(DatasetRegistryTest, MarkTrainedSkipsDuplicateEntry) {
    DatasetRegistry reg(make_cfg());
    reg.mark_trained({data_file_}, {10});
    reg.mark_trained({data_file_}, {20});  // second call — should be a no-op
    EXPECT_EQ(reg.trained_files().size(), 1u);
}

// TD-067 regression: the 2-arg mark_trained() overload used to leave a
// now-trained file sitting in pending_ — only the Phase 9 run_id overload
// cleared it. A file that's both pending and trained is a nonsensical state
// (add_file() itself refuses to re-add an already-trained file).
TEST_F(DatasetRegistryTest, MarkTrainedRemovesFileFromPending) {
    DatasetRegistry reg(make_cfg());
    ASSERT_TRUE(reg.add_file(data_file_));
    ASSERT_EQ(reg.pending_files().size(), 1u);

    reg.mark_trained({data_file_}, {10});

    EXPECT_TRUE(reg.is_trained(data_file_));
    EXPECT_TRUE(reg.pending_files().empty());
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
    EXPECT_TRUE(resp.ftp_server_host.empty());   // LocalTransport: no FTP
    EXPECT_FALSE(reg2.pending_files().empty());  // reflected in-memory
}

TEST_F(DatasetRegistryTest, AcquirePendingEmptyWhenNoneAvailable) {
    DatasetRegistry reg(make_cfg());
    EXPECT_TRUE(reg.acquire_pending("run-a").files.empty());
}

TEST_F(DatasetRegistryTest, AcquirePendingReclaimsOwnPriorClaim) {
    {
        DatasetRegistry reg(make_cfg());
        reg.add_file(data_file_);
    }
    DatasetRegistry reg2(make_cfg());
    auto first = reg2.acquire_pending("run-a");
    ASSERT_EQ(first.files.size(), 1u);

    // Same run_id, no release in between — simulates a crash-restart that
    // gets the same run_id back (e.g. MNS's new_run=false continuation).
    // Must reclaim its own prior claim rather than finding nothing pending.
    DatasetRegistry reg3(make_cfg());
    auto second = reg3.acquire_pending("run-a");
    ASSERT_EQ(second.files.size(), 1u);
    EXPECT_EQ(second.files[0].registry_path, data_file_);
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

// ============================================================================
// Phase 16 — assign-by-count / unassign / delete
// ============================================================================

TEST_F(DatasetRegistryTest, AssignModelReturnsCountAndPaths) {
    DatasetRegistry reg(make_cfg());
    reg.add_file(data_file_);

    auto result = reg.assign_model("model-a");
    EXPECT_EQ(result.assigned, 1);
    ASSERT_EQ(result.paths.size(), 1u);
    EXPECT_EQ(result.paths[0], data_file_);
    ASSERT_EQ(reg.pending_entries().size(), 1u);
    EXPECT_EQ(reg.pending_entries()[0].model_name, "model-a");
}

TEST_F(DatasetRegistryTest, AssignModelWithCountLimitsAssignment) {
    const std::string file2 = (tmp_dir_ / "training2.txt").string();
    const std::string file3 = (tmp_dir_ / "training3.txt").string();
    write_file(file2, "INPUT: x\nRESPONSE: y\n");
    write_file(file3, "INPUT: p\nRESPONSE: q\n");

    DatasetRegistry reg(make_cfg());
    reg.add_file(data_file_);
    reg.add_file(file2);
    reg.add_file(file3);

    auto result = reg.assign_model("model-a", {}, 2);
    EXPECT_EQ(result.assigned, 2);
    EXPECT_EQ(result.paths.size(), 2u);

    int assigned_count = 0;
    for (const auto& e : reg.pending_entries()) {
        if (e.model_name == "model-a")
            ++assigned_count;
    }
    EXPECT_EQ(assigned_count, 2);
}

TEST_F(DatasetRegistryTest, UnassignModelClearsInMemoryPending) {
    DatasetRegistry reg(make_cfg());
    reg.add_file(data_file_);
    reg.assign_model("model-a");

    auto result = reg.unassign_model("model-a");
    EXPECT_EQ(result.unassigned, 1);
    EXPECT_EQ(result.skipped, 0);
    ASSERT_EQ(reg.pending_entries().size(), 1u);
    EXPECT_TRUE(reg.pending_entries()[0].model_name.empty());
}

TEST_F(DatasetRegistryTest, UnassignModelSkipsActiveClaimWithoutForce) {
    // acquire_pending is assignment-aware (see RegistryTransport::acquire): a
    // caller can only claim entries assigned to itself or unassigned ones, so
    // this test's config must identify as "model-a" to claim the file it just
    // assigned to "model-a" below.
    DatasetConfig cfg = make_cfg();
    cfg.model_name = "model-a";
    DatasetRegistry reg(cfg);
    reg.add_file(data_file_);
    reg.assign_model("model-a");
    auto resp = reg.acquire_pending("run-a");
    ASSERT_FALSE(resp.files.empty());

    auto result = reg.unassign_model("model-a");
    EXPECT_EQ(result.unassigned, 0);
    EXPECT_EQ(result.skipped, 1);

    auto result2 = reg.unassign_model("model-a", {}, /*force=*/true);
    EXPECT_EQ(result2.unassigned, 1);
    EXPECT_EQ(result2.skipped, 0);
}

TEST_F(DatasetRegistryTest, DeleteEntriesRemovesFromPendingAndRegistry) {
    DatasetRegistry reg(make_cfg());
    reg.add_file(data_file_);

    auto result = reg.delete_entries({data_file_});
    EXPECT_EQ(result.deleted, 1);
    EXPECT_TRUE(reg.pending_files().empty());
}

TEST_F(DatasetRegistryTest, DeleteEntriesUpdatesTrainedSetAndIsTrained) {
    DatasetRegistry reg(make_cfg());
    reg.mark_trained({data_file_}, {50});
    ASSERT_TRUE(reg.is_trained(data_file_));
    ASSERT_EQ(reg.total_samples_trained(), 50);

    auto result = reg.delete_entries({data_file_});
    EXPECT_EQ(result.deleted, 1);
    EXPECT_FALSE(reg.is_trained(data_file_));
    EXPECT_EQ(reg.total_samples_trained(), 0);
}

TEST_F(DatasetRegistryTest, DeleteEntriesSkipsActiveClaim) {
    DatasetRegistry reg(make_cfg());
    reg.add_file(data_file_);
    auto resp = reg.acquire_pending("run-a");
    ASSERT_FALSE(resp.files.empty());

    auto result = reg.delete_entries({data_file_});
    EXPECT_EQ(result.deleted, 0);
    EXPECT_EQ(result.skipped, 1);

    auto result2 = reg.delete_entries({data_file_}, /*force=*/true);
    EXPECT_EQ(result2.deleted, 1);
    EXPECT_EQ(result2.skipped, 0);
}

// ============================================================================
// TD-205: DatasetRegistry-level segment support (add_file/assign_model/
// unassign_model/delete_entries with explicit SegmentTarget lists)
// ============================================================================

TEST_F(DatasetRegistryTest, AddFileWithSegmentCreatesIndependentEntry) {
    DatasetRegistry reg(make_cfg());
    ASSERT_TRUE(reg.add_file(data_file_, 0, 1));
    ASSERT_TRUE(reg.add_file(data_file_, 1, 1));

    ASSERT_EQ(reg.pending_entries().size(), 2u);
    EXPECT_EQ(reg.pending_entries()[0].segment_start, 0);
    EXPECT_EQ(reg.pending_entries()[1].segment_start, 1);
}

TEST_F(DatasetRegistryTest, AssignModelWithSegmentsTargetsExactRange) {
    DatasetRegistry reg(make_cfg());
    reg.add_file(data_file_, 0, 1);
    reg.add_file(data_file_, 1, 1);

    auto result = reg.assign_model("model-a", {}, 0, {{data_file_, 0, 1}});
    EXPECT_EQ(result.assigned, 1);
    ASSERT_EQ(result.segments.size(), 1u);

    for (const auto& e : reg.pending_entries()) {
        if (e.segment_start == 0)
            EXPECT_EQ(e.model_name, "model-a");
        else
            EXPECT_TRUE(e.model_name.empty());
    }
}

TEST_F(DatasetRegistryTest, UnassignModelWithSegmentsTargetsExactRange) {
    DatasetRegistry reg(make_cfg());
    reg.add_file(data_file_, 0, 1);
    reg.add_file(data_file_, 1, 1);
    reg.assign_model("model-a", {}, 0, {{data_file_, 0, 1}});
    reg.assign_model("model-a", {}, 0, {{data_file_, 1, 1}});

    auto result = reg.unassign_model("model-a", {}, false, {{data_file_, 0, 1}});
    EXPECT_EQ(result.unassigned, 1);

    for (const auto& e : reg.pending_entries()) {
        if (e.segment_start == 0)
            EXPECT_TRUE(e.model_name.empty());
        else
            EXPECT_EQ(e.model_name, "model-a");
    }
}

TEST_F(DatasetRegistryTest, DeleteEntriesWithSegmentsNeverUnlinksSharedFileAndKeepsOtherSegment) {
    DatasetRegistry reg(make_cfg());
    reg.add_file(data_file_, 0, 1);
    reg.add_file(data_file_, 1, 1);

    auto result = reg.delete_entries({}, false, true, {{data_file_, 0, 1}});
    EXPECT_EQ(result.deleted, 1);
    EXPECT_TRUE(fs::exists(data_file_)) << "a segment delete must never unlink the shared file";

    ASSERT_EQ(reg.pending_entries().size(), 1u);
    EXPECT_EQ(reg.pending_entries()[0].segment_start, 1);
}

// ============================================================================
// Server-side dataset fetch (Phase 11) — remote_fetch_*/remote_upload
// ============================================================================

class DatasetRegistryFetchTest : public ::testing::Test {
   protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "adai_registry_fetch_test";
        fs::remove_all(tmp_dir_);
        fs::create_directories(tmp_dir_);
    }

    void TearDown() override {
        fs::remove_all(tmp_dir_);
    }

    std::unique_ptr<DatasetRegistry> make_reg(FakeFetchTransport** out_fake) {
        auto transport = std::make_unique<FakeFetchTransport>(
            (tmp_dir_ / "registry.txt").string(), (tmp_dir_ / "pending.txt").string());
        *out_fake = transport.get();
        DatasetConfig cfg;
        cfg.session_dir = (tmp_dir_ / "sessions").string();
        return std::make_unique<DatasetRegistry>(cfg, std::move(transport));
    }

    fs::path tmp_dir_;
};

TEST_F(DatasetRegistryFetchTest, RemoteFetchGutenbergReturnsPathAndUpdatesPending) {
    FakeFetchTransport* fake = nullptr;
    auto reg = make_reg(&fake);
    fake->next_fetch_result = "/reg/data/gutenberg_1342_training.jsonl";

    const std::string path = reg->remote_fetch_gutenberg(1342, 100);
    EXPECT_EQ(path, "/reg/data/gutenberg_1342_training.jsonl");
    EXPECT_EQ(fake->fetch_gutenberg_calls, 1);
    EXPECT_EQ(fake->last_book_id, 1342);
    EXPECT_TRUE(fake->last_model_name.empty());
    ASSERT_EQ(reg->pending_files().size(), 1u);
    EXPECT_EQ(reg->pending_files()[0], path);
}

TEST_F(DatasetRegistryFetchTest, RemoteFetchGutenbergForwardsModelName) {
    FakeFetchTransport* fake = nullptr;
    auto reg = make_reg(&fake);
    fake->next_fetch_result = "/reg/data/gutenberg_1342_model-a_row0_training.jsonl";

    reg->remote_fetch_gutenberg(1342, 100, "model-a");
    EXPECT_EQ(fake->last_model_name, "model-a");
}

TEST_F(DatasetRegistryFetchTest, RemoteFetchGutenbergFailureLeavesPendingEmpty) {
    FakeFetchTransport* fake = nullptr;
    auto reg = make_reg(&fake);
    fake->next_fetch_result = "";  // simulated failure

    EXPECT_EQ(reg->remote_fetch_gutenberg(1342, 100), "");
    EXPECT_TRUE(reg->pending_files().empty());
}

TEST_F(DatasetRegistryFetchTest, RemoteFetchHuggingfaceReturnsPathAndUpdatesPending) {
    FakeFetchTransport* fake = nullptr;
    auto reg = make_reg(&fake);
    fake->next_fetch_result = "/reg/data/daily_dialog_train_training.jsonl";

    const std::string path =
        reg->remote_fetch_huggingface("daily_dialog", 200, "train", "", "");
    EXPECT_EQ(path, "/reg/data/daily_dialog_train_training.jsonl");
    EXPECT_EQ(fake->fetch_huggingface_calls, 1);
    EXPECT_EQ(fake->last_dataset_id, "daily_dialog");
    EXPECT_TRUE(fake->last_model_name.empty());
    ASSERT_EQ(reg->pending_files().size(), 1u);
}

TEST_F(DatasetRegistryFetchTest, RemoteFetchHuggingfaceForwardsModelName) {
    FakeFetchTransport* fake = nullptr;
    auto reg = make_reg(&fake);
    fake->next_fetch_result = "/reg/data/daily_dialog_train_model-a_row0_training.jsonl";

    reg->remote_fetch_huggingface("daily_dialog", 200, "train", "", "", "model-a");
    EXPECT_EQ(fake->last_model_name, "model-a");
}

TEST_F(DatasetRegistryFetchTest, RemoteUploadReturnsPathAndUpdatesPending) {
    FakeFetchTransport* fake = nullptr;
    auto reg = make_reg(&fake);
    fake->next_upload_result = "/reg/data/uploads/my_file.jsonl";

    const std::string path = reg->remote_upload("/local/my_file.jsonl");
    EXPECT_EQ(path, "/reg/data/uploads/my_file.jsonl");
    EXPECT_EQ(fake->upload_calls, 1);
    EXPECT_EQ(fake->last_upload_path, "/local/my_file.jsonl");
    ASSERT_EQ(reg->pending_files().size(), 1u);
    EXPECT_EQ(reg->pending_files()[0], path);
}

TEST_F(DatasetRegistryFetchTest, RemoteUploadFailureLeavesPendingEmpty) {
    FakeFetchTransport* fake = nullptr;
    auto reg = make_reg(&fake);
    fake->next_upload_result = "";  // simulated failure

    EXPECT_EQ(reg->remote_upload("/local/missing.jsonl"), "");
    EXPECT_TRUE(reg->pending_files().empty());
}

// TD-413: incremental_trainer's pre-check counts what this model can acquire from its own kind
// sub-pool (SESSION_DIR/<kind>/ locally), through DatasetRegistry, instead of reading
// SESSION_DIR/pending_files.txt directly.
TEST(DatasetRegistryCountAcquirableTest, CountsThisModelsEntriesInTheKindPool) {
    const fs::path base = fs::temp_directory_path() / "adai_count_acquirable";
    fs::remove_all(base);
    fs::create_directories(base / "data");
    const auto file = [&](const std::string& name) {
        const fs::path p = base / "data" / name;
        std::ofstream(p) << "{\"input\":\"a\",\"response\":\"b\"}\n";
        return p.string();
    };
    DatasetConfig cfg;
    cfg.session_dir = (base / "sess").string();
    cfg.dataset_kind = "chatbot";
    cfg.model_name = "m1";
    {
        DatasetRegistry setup(cfg);
        setup.load_pending_list();
        ASSERT_TRUE(setup.add_file(file("free.jsonl")));
        ASSERT_TRUE(setup.add_file(file("mine.jsonl")));
        ASSERT_TRUE(setup.add_file(file("theirs.jsonl")));
        setup.assign_model("m1", {(base / "data" / "mine.jsonl").string()});
        setup.assign_model("m2", {(base / "data" / "theirs.jsonl").string()});
    }
    EXPECT_TRUE(fs::exists(base / "sess" / "chatbot" / "pending_files.txt"));
    EXPECT_FALSE(fs::exists(base / "sess" / "pending_files.txt"))
        << "the pool is the kind's, which the old pre-check never read";

    DatasetRegistry reg(cfg);
    reg.load_pending_list();
    EXPECT_EQ(reg.pending_files().size(), 3u);
    EXPECT_EQ(reg.count_acquirable(), 2u) << "unassigned + assigned to m1, not m2's";

    cfg.model_name.clear();
    DatasetRegistry anonymous(cfg);
    anonymous.load_pending_list();
    EXPECT_EQ(anonymous.count_acquirable(), 1u) << "no model identity: unassigned only";
    fs::remove_all(base);
}

// TD-494: while a lease applies, DatasetRegistry renews its claims every third of the lease (at
// least every 5 s) from acquire_pending() until the pass marks its files trained.
TEST(DatasetRegistryClaimLeaseTest, RenewsClaimsDuringThePassAndStopsAfterIt) {
    const fs::path base = fs::temp_directory_path() / "adai_claim_lease";
    fs::remove_all(base);
    const std::string data = (base / "d.jsonl").string();
    write_file(data, kJsonlPairs);

    auto fake_owned = std::make_unique<FakeFetchTransport>((base / "reg.txt").string(),
                                                           (base / "pending.txt").string());
    FakeFetchTransport* fake = fake_owned.get();
    fake->lease_seconds = 15;  // renew every 5 s
    DatasetConfig cfg;
    cfg.session_dir = base.string();
    DatasetRegistry reg(cfg, std::move(fake_owned));
    reg.load_pending_list();
    ASSERT_TRUE(reg.add_file(data));

    const auto resp = reg.acquire_pending("run-lease");
    ASSERT_EQ(resp.files.size(), 1u);
    EXPECT_EQ(resp.claim_lease_seconds, 15);
    std::this_thread::sleep_for(std::chrono::milliseconds(5600));
    EXPECT_GE(fake->renew_calls.load(), 1) << "no renewal within the first interval";

    reg.mark_trained("run-lease", resp.file_ranges(), {2});
    const int after_trained = fake->renew_calls.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(5600));
    EXPECT_EQ(fake->renew_calls.load(), after_trained) << "kept renewing after the pass ended";
    fs::remove_all(base);
}

// Without a lease (LocalTransport, or a registry with leases off) nothing is renewed.
TEST(DatasetRegistryClaimLeaseTest, NoLeaseMeansNoRenewer) {
    const fs::path base = fs::temp_directory_path() / "adai_claim_nolease";
    fs::remove_all(base);
    const std::string data = (base / "d.jsonl").string();
    write_file(data, kJsonlPairs);
    auto fake_owned = std::make_unique<FakeFetchTransport>((base / "reg.txt").string(),
                                                           (base / "pending.txt").string());
    FakeFetchTransport* fake = fake_owned.get();
    DatasetConfig cfg;
    cfg.session_dir = base.string();
    DatasetRegistry reg(cfg, std::move(fake_owned));
    reg.load_pending_list();
    ASSERT_TRUE(reg.add_file(data));
    const auto resp = reg.acquire_pending("run-nolease");
    ASSERT_EQ(resp.files.size(), 1u);
    EXPECT_EQ(resp.claim_lease_seconds, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(5600));
    EXPECT_EQ(fake->renew_calls.load(), 0);
    fs::remove_all(base);
}
