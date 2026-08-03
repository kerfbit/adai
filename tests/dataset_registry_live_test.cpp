// Live-server integration tests for the registry_server daemon (TD-028 Phase 9).
//
// Exercises every HTTP endpoint exposed by RegistryServer.cpp against a running
// instance.  Tests auto-skip when the server is unreachable so the suite can
// be included in the normal build without requiring the daemon to be running.
//
// Each test creates an isolated group name derived from the PID, timestamp, and
// a monotonic counter so concurrent test runs on the same server cannot collide.
//
// Configuration (env vars, both optional):
//   REGISTRY_SERVER_HOST   default: 127.0.0.1
//   REGISTRY_SERVER_PORT   default: 8082
//
// The default host MUST be a loopback/clearly-local address, never a real LAN IP —
// see the identical note in tests/training_metrics_api_live_test.cpp for why (a
// previous default here pointed at a real LAN address and every normal `ctest` run
// silently mutated whatever server happened to live there). Point REGISTRY_SERVER_HOST
// at a real server explicitly and deliberately when you need to exercise one.

#include <gtest/gtest.h>

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>

#include "../src/RegistryTransport.hpp"

namespace {

// ---------------------------------------------------------------------------
// Connection helpers
// ---------------------------------------------------------------------------

std::string server_host() {
    const char* env = std::getenv("REGISTRY_SERVER_HOST");
    return env ? std::string(env) : "127.0.0.1";
}

int server_port() {
    const char* env = std::getenv("REGISTRY_SERVER_PORT");
    return env ? std::atoi(env) : 8082;
}

httplib::Client make_client() {
    httplib::Client c(server_host(), server_port());
    c.set_connection_timeout(std::chrono::seconds(5));
    c.set_read_timeout(std::chrono::seconds(5));
    c.set_write_timeout(std::chrono::seconds(5));
    return c;
}

bool server_reachable() {
    // Require an explicit opt-in via REGISTRY_SERVER_HOST rather than probing a
    // default host/port — see the identical note in
    // tests/training_metrics_api_live_test.cpp for why: this suite mutates real
    // registry state, and guessing at a default (even loopback) risks silently
    // hitting whatever unrelated service happens to already be listening there.
    if (!std::getenv("REGISTRY_SERVER_HOST")) {
        return false;
    }
    auto c = make_client();
    auto res = c.Get("/health");
    return res && res->status == 200;
}

// Generates a group name that is unique across process invocations.
//
// Uses millisecond resolution (not seconds) so that even when the OS recycles a
// PID within the same second, the two processes — which started at different
// milliseconds — produce different prefixes.  The per-call counter ensures
// uniqueness across multiple tests within the same process.
std::string make_group(const std::string& tag = "") {
    // Captured once at first call: PID + process-start time in milliseconds.
    static const std::string process_prefix = [] {
        const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
        std::ostringstream o;
        o << "lt" << static_cast<int>(::getpid()) << "m" << ms;
        return o.str();
    }();
    static std::atomic<int> counter{0};
    std::ostringstream oss;
    oss << process_prefix << "c" << counter.fetch_add(1);
    if (!tag.empty())
        oss << "-" << tag;
    return oss.str();
}

// Minimal JSON field extractors — mirrors the server's own helpers.
std::string json_string(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return {};
    const auto start = pos + needle.size();
    const auto end = body.find('"', start);
    if (end == std::string::npos)
        return {};
    return body.substr(start, end - start);
}

int json_int(const std::string& body, const std::string& key, int def = -1) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return def;
    try {
        return std::stoi(body.substr(pos + needle.size()));
    } catch (...) {
        return def;
    }
}

bool json_bool(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return false;
    const auto rest = body.substr(pos + needle.size());
    return rest.substr(0, 4) == "true";
}

// ---------------------------------------------------------------------------
// Per-test fixture — each test gets its own unique group namespace.
// ---------------------------------------------------------------------------

class LiveRegistryTest : public ::testing::Test {
   protected:
    // True when the server supports current-format features (history endpoint +
    // "REMOTE" checksum column in the registry flat file).  Set false when the
    // server is an older build that predates these additions, detected by probing
    // GET /registry/<group>/history: current → 200, legacy → 404.
    bool server_current_format_ = true;

    void SetUp() override {
        if (!server_reachable()) {
            GTEST_SKIP() << "registry_server not reachable at " << server_host() << ":"
                         << server_port();
        }
        client_ = std::make_unique<httplib::Client>(server_host(), server_port());
        client_->set_connection_timeout(std::chrono::seconds(5));
        client_->set_read_timeout(std::chrono::seconds(5));
        client_->set_write_timeout(std::chrono::seconds(5));

        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        group_ = make_group(info ? info->name() : "");

        // Probe the /history endpoint to detect the server binary version.
        // A current server returns 200 (empty entries) for any group.
        // A legacy server returns 404 (endpoint not implemented).
        // The same version gap that removed /history also omitted the REMOTE
        // checksum column, so this one probe covers both format checks.
        const auto probe = client_->Get(("/registry/" + group_ + "/history").c_str());
        server_current_format_ = (probe && probe->status == 200);
    }

    // Helper: POST /registry/<group_>/pending/add {"path":"<path>"}
    httplib::Result add_pending(const std::string& path) {
        const std::string body = "{\"path\":\"" + path + "\"}";
        return client_->Post(("/registry/" + group_ + "/pending/add").c_str(), body,
                             "application/json");
    }

    // Helper: GET /registry/<group_>/queue
    httplib::Result get_queue() {
        return client_->Get(("/registry/" + group_ + "/queue").c_str());
    }

    // Helper: POST /registry/<group_>/acquire {"run_id":"<rid>","max_files":<n>}
    httplib::Result acquire(const std::string& run_id, int max_files = 0) {
        std::ostringstream body;
        body << "{\"run_id\":\"" << run_id << "\",\"max_files\":" << max_files << "}";
        return client_->Post(("/registry/" + group_ + "/acquire").c_str(), body.str(),
                             "application/json");
    }

    // Helper: POST /registry/<group_>/release {"run_id":"<rid>","files":[...]}
    httplib::Result release(const std::string& run_id, const std::vector<std::string>& files) {
        std::ostringstream body;
        body << "{\"run_id\":\"" << run_id << "\",\"files\":[";
        for (std::size_t i = 0; i < files.size(); ++i) {
            if (i)
                body << ',';
            body << '"' << files[i] << '"';
        }
        body << "]}";
        return client_->Post(("/registry/" + group_ + "/release").c_str(), body.str(),
                             "application/json");
    }

    // Helper: POST /registry/<group_>/assign {"model_name":"<model>","paths":[...]}
    httplib::Result assign(const std::string& model_name_raw_json,
                           const std::vector<std::string>& paths = {}) {
        std::ostringstream body;
        body << "{\"model_name\":\"" << model_name_raw_json << "\",\"paths\":[";
        for (std::size_t i = 0; i < paths.size(); ++i) {
            if (i)
                body << ',';
            body << '"' << paths[i] << '"';
        }
        body << "]}";
        return client_->Post(("/registry/" + group_ + "/assign").c_str(), body.str(),
                             "application/json");
    }

    // Helper: POST /registry/<group_>/trained
    httplib::Result trained(const std::string& run_id, const std::vector<std::string>& files,
                            const std::vector<int>& samples = {}) {
        std::ostringstream body;
        body << "{\"run_id\":\"" << run_id << "\",\"files\":[";
        for (std::size_t i = 0; i < files.size(); ++i) {
            if (i)
                body << ',';
            body << '"' << files[i] << '"';
        }
        body << "],\"samples\":[";
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (i)
                body << ',';
            body << samples[i];
        }
        body << "]}";
        return client_->Post(("/registry/" + group_ + "/trained").c_str(), body.str(),
                             "application/json");
    }

    // Helper: GET /registry/<group_>/registry
    httplib::Result get_registry() {
        return client_->Get(("/registry/" + group_ + "/registry").c_str());
    }

    // Helper: GET /registry/<group_>/runs
    httplib::Result get_runs() {
        return client_->Get(("/registry/" + group_ + "/runs").c_str());
    }

    // Helper: POST /registry/<group_>/trained with optional model_id
    httplib::Result trained_with_model_id(const std::string& run_id,
                                          const std::vector<std::string>& files,
                                          const std::string& model_id,
                                          const std::vector<int>& samples = {}) {
        std::ostringstream body;
        body << "{\"run_id\":\"" << run_id << "\",\"model_id\":\"" << model_id << "\",\"files\":[";
        for (std::size_t i = 0; i < files.size(); ++i) {
            if (i)
                body << ',';
            body << '"' << files[i] << '"';
        }
        body << "],\"samples\":[";
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (i)
                body << ',';
            body << samples[i];
        }
        body << "]}";
        return client_->Post(("/registry/" + group_ + "/trained").c_str(), body.str(),
                             "application/json");
    }

    // Helper: GET /registry/<group_>/history[?model_id=<id>]
    httplib::Result get_history(const std::string& model_id_filter = "") {
        std::string path = "/registry/" + group_ + "/history";
        if (!model_id_filter.empty())
            path += "?model_id=" + model_id_filter;
        return client_->Get(path.c_str());
    }

    // Helper: POST /registry/<group_>/fetch/gutenberg {"book_id":N,"num_pairs":N,"model_name":"..."}
    httplib::Result fetch_gutenberg(int book_id, int num_pairs = 100,
                                    const std::string& model_name_raw_json = "") {
        std::ostringstream body;
        body << "{\"book_id\":" << book_id << ",\"num_pairs\":" << num_pairs
             << ",\"model_name\":\"" << model_name_raw_json << "\"}";
        return client_->Post(("/registry/" + group_ + "/fetch/gutenberg").c_str(), body.str(),
                             "application/json");
    }

    // Helper: POST /registry/<group_>/fetch/huggingface {"dataset_id":"...",...}
    httplib::Result fetch_huggingface(const std::string& dataset_id_raw_json,
                                      int num_pairs = 100,
                                      const std::string& model_name_raw_json = "") {
        std::ostringstream body;
        body << "{\"dataset_id\":\"" << dataset_id_raw_json << "\",\"num_pairs\":" << num_pairs
             << ",\"model_name\":\"" << model_name_raw_json << "\"}";
        return client_->Post(("/registry/" + group_ + "/fetch/huggingface").c_str(), body.str(),
                             "application/json");
    }

    // Helper: POST /registry/<group_>/upload?filename=<name>  (raw body)
    httplib::Result upload(const std::string& filename, const std::string& contents) {
        const std::string path =
            "/registry/" + group_ + "/upload?filename=" + filename;
        return client_->Post(path.c_str(), contents, "application/octet-stream");
    }

    httplib::Client& client() {
        return *client_;
    }
    const std::string& group() const {
        return group_;
    }

   private:
    std::unique_ptr<httplib::Client> client_;
    std::string group_;
};

// ===========================================================================
// Health
// ===========================================================================

TEST(LiveRegistryHealth, Returns200WithStatusOk) {
    if (!server_reachable())
        GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    auto res = c.Get("/health");
    ASSERT_TRUE(res) << "no response from /health";
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"status\":\"ok\""), std::string::npos) << res->body;
}

// ===========================================================================
// Queue — GET /registry/<group>/queue
// ===========================================================================

TEST_F(LiveRegistryTest, QueueIsInitiallyEmpty) {
    auto res = get_queue();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"entries\":[]"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, PendingAddAppearsInQueue) {
    const std::string path = "/data/" + group() + "/file1.txt";
    ASSERT_TRUE(add_pending(path));

    auto res = get_queue();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find(path), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, MultiplePendingFilesAllAppearInQueue) {
    const std::string p1 = "/data/" + group() + "/a.txt";
    const std::string p2 = "/data/" + group() + "/b.txt";
    const std::string p3 = "/data/" + group() + "/c.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));
    ASSERT_TRUE(add_pending(p3));

    auto res = get_queue();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find(p1), std::string::npos) << res->body;
    EXPECT_NE(res->body.find(p2), std::string::npos) << res->body;
    EXPECT_NE(res->body.find(p3), std::string::npos) << res->body;
}

// ===========================================================================
// Pending/add — POST /registry/<group>/pending/add
// ===========================================================================

TEST_F(LiveRegistryTest, PendingAddReturnsTrueOnFirstAdd) {
    const std::string path = "/data/" + group() + "/new.txt";
    auto res = add_pending(path);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"added\":true"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, PendingAddDeduplicates) {
    const std::string path = "/data/" + group() + "/dup.txt";
    ASSERT_TRUE(add_pending(path));

    auto res = add_pending(path);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"added\":false"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"reason\":\"already_pending\""), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, PendingAddRequiresPathField) {
    auto res =
        client().Post(("/registry/" + group() + "/pending/add").c_str(), "{}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
    EXPECT_NE(res->body.find("\"error\":"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, PendingAddEmptyBodyReturns400) {
    auto res =
        client().Post(("/registry/" + group() + "/pending/add").c_str(), "", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

// ===========================================================================
// Acquire — POST /registry/<group>/acquire
// ===========================================================================

TEST_F(LiveRegistryTest, AcquireRequiresRunId) {
    const std::string path = "/data/" + group() + "/f.txt";
    ASSERT_TRUE(add_pending(path));

    auto res = client().Post(("/registry/" + group() + "/acquire").c_str(), "{\"max_files\":0}",
                             "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
    EXPECT_NE(res->body.find("\"error\":"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, AcquireReturnsClaimedFiles) {
    const std::string p1 = "/data/" + group() + "/x.txt";
    const std::string p2 = "/data/" + group() + "/y.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));

    auto res = acquire("run-A");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"acquired\":"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find(p1), std::string::npos) << res->body;
    EXPECT_NE(res->body.find(p2), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, AcquireRespectsMaxFilesLimit) {
    const std::string p1 = "/data/" + group() + "/f1.txt";
    const std::string p2 = "/data/" + group() + "/f2.txt";
    const std::string p3 = "/data/" + group() + "/f3.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));
    ASSERT_TRUE(add_pending(p3));

    auto res = acquire("run-B", /*max_files=*/2);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    // Exactly 2 paths should be in the acquired array.
    const std::string& body = res->body;
    const auto arr_start = body.find("\"acquired\":[");
    ASSERT_NE(arr_start, std::string::npos);
    const auto arr_end = body.find(']', arr_start);
    ASSERT_NE(arr_end, std::string::npos);
    const std::string arr = body.substr(arr_start, arr_end - arr_start);
    int count = 0;
    for (std::size_t pos = 0; (pos = arr.find("/data/", pos)) != std::string::npos; ++pos)
        ++count;
    EXPECT_EQ(count, 2) << "Expected exactly 2 acquired paths, got: " << body;
}

TEST_F(LiveRegistryTest, AcquireEmptyWhenNoPendingFiles) {
    auto res = acquire("run-C");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"acquired\":[]"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, SecondRunCannotClaimFilesAssignedToFirst) {
    const std::string path = "/data/" + group() + "/claimed.txt";
    ASSERT_TRUE(add_pending(path));

    // First run acquires all available files.
    auto res1 = acquire("run-first");
    ASSERT_TRUE(res1);
    EXPECT_EQ(res1->status, 200);
    EXPECT_NE(res1->body.find(path), std::string::npos) << res1->body;

    // Second run should find nothing available.
    auto res2 = acquire("run-second");
    ASSERT_TRUE(res2);
    EXPECT_EQ(res2->status, 200);
    EXPECT_NE(res2->body.find("\"acquired\":[]"), std::string::npos) << res2->body;
}

TEST_F(LiveRegistryTest, AcquireShowsAssignedRunIdInQueue) {
    const std::string path = "/data/" + group() + "/assigned.txt";
    ASSERT_TRUE(add_pending(path));

    ASSERT_TRUE(acquire("run-owner"));

    auto res = get_queue();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("run-owner"), std::string::npos) << res->body;
}

// ===========================================================================
// Release — POST /registry/<group>/release
// ===========================================================================

TEST_F(LiveRegistryTest, ReleaseRestoresFilesToUnassignedPool) {
    const std::string path = "/data/" + group() + "/relpend.txt";
    ASSERT_TRUE(add_pending(path));

    ASSERT_TRUE(acquire("run-X"));

    auto rel_res = release("run-X", {path});
    ASSERT_TRUE(rel_res);
    EXPECT_EQ(rel_res->status, 200);
    EXPECT_EQ(json_int(rel_res->body, "released"), 1) << rel_res->body;

    // A different run can now claim the file.
    auto res2 = acquire("run-Y");
    ASSERT_TRUE(res2);
    EXPECT_EQ(res2->status, 200);
    EXPECT_NE(res2->body.find(path), std::string::npos) << res2->body;
}

TEST_F(LiveRegistryTest, ReleaseDoesNotAffectFilesOwnedByOtherRun) {
    const std::string p1 = "/data/" + group() + "/own1.txt";
    const std::string p2 = "/data/" + group() + "/own2.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));

    // run-P claims p1, run-Q claims p2.
    ASSERT_TRUE(acquire("run-P", 1));
    ASSERT_TRUE(acquire("run-Q", 1));

    // run-P tries to release p2 (which it doesn't own).
    auto rel_res = release("run-P", {p2});
    ASSERT_TRUE(rel_res);
    EXPECT_EQ(rel_res->status, 200);
    EXPECT_EQ(json_int(rel_res->body, "released"), 0) << rel_res->body;

    // p2 is still owned by run-Q and not available for new acquisition.
    auto res3 = acquire("run-Z");
    ASSERT_TRUE(res3);
    EXPECT_NE(res3->body.find("\"acquired\":[]"), std::string::npos) << res3->body;
}

TEST_F(LiveRegistryTest, ReleaseReturnsZeroForNonExistentFiles) {
    auto res = release("run-ghost", {"/no/such/file.txt"});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(json_int(res->body, "released"), 0) << res->body;
}

// ===========================================================================
// Assign — POST /registry/<group>/assign (Phase 14)
//
// Fills a gap that predates this endpoint: DatasetRegistry::assign_model()
// silently no-ops against RemoteTransport (save_pending() is a documented
// no-op in distributed mode), so dataset_manager's `assign` command never
// actually worked against a real registry_server before this existed.
// ===========================================================================

TEST_F(LiveRegistryTest, AssignSetsModelNameOnSpecifiedPath) {
    const std::string p1 = "/data/" + group() + "/assign1.txt";
    const std::string p2 = "/data/" + group() + "/assign2.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));

    auto res = assign("model-a", {p1});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(json_int(res->body, "assigned"), 1) << res->body;

    auto queue_res = get_queue();
    ASSERT_TRUE(queue_res);
    // p1's entry should carry the model_name; p2's should not.
    const auto p1_pos = queue_res->body.find(p1);
    ASSERT_NE(p1_pos, std::string::npos);
    EXPECT_NE(queue_res->body.find("\"model_name\":\"model-a\"", p1_pos), std::string::npos)
        << queue_res->body;
}

TEST_F(LiveRegistryTest, AssignWithEmptyPathsAssignsAllPending) {
    const std::string p1 = "/data/" + group() + "/assignall1.txt";
    const std::string p2 = "/data/" + group() + "/assignall2.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));

    auto res = assign("model-b");  // no paths → assign all
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(json_int(res->body, "assigned"), 2) << res->body;

    auto queue_res = get_queue();
    ASSERT_TRUE(queue_res);
    int occurrences = 0;
    std::size_t pos = 0;
    const std::string needle = "\"model_name\":\"model-b\"";
    while ((pos = queue_res->body.find(needle, pos)) != std::string::npos) {
        ++occurrences;
        pos += needle.size();
    }
    EXPECT_EQ(occurrences, 2) << queue_res->body;
}

TEST_F(LiveRegistryTest, AssignRejectsEmptyModelName) {
    auto res = assign("");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, AssignRejectsUnsafeModelName) {
    // model_name is written verbatim into the tab-separated pending-file
    // format, so a literal '/' (and by extension tabs/newlines, though those
    // aren't expressible in this JSON-string test helper) must be rejected.
    auto res = assign("../../etc/passwd");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, AssignReturnsZeroForNonExistentPath) {
    auto res = assign("model-c", {"/no/such/file.txt"});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(json_int(res->body, "assigned"), 0) << res->body;
}

// ===========================================================================
// Trained — POST /registry/<group>/trained
// ===========================================================================

TEST_F(LiveRegistryTest, TrainedCommitsFileToRegistry) {
    if (!server_current_format_)
        GTEST_SKIP() << "Server uses legacy registry format; rebuild and redeploy registry_server";
    const std::string path = "/data/" + group() + "/train.txt";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-T"));

    auto tr_res = trained("run-T", {path}, {42});
    ASSERT_TRUE(tr_res);
    EXPECT_EQ(tr_res->status, 200);
    EXPECT_EQ(json_int(tr_res->body, "trained"), 1) << tr_res->body;

    auto reg_res = get_registry();
    ASSERT_TRUE(reg_res);
    EXPECT_EQ(reg_res->status, 200);
    EXPECT_NE(reg_res->body.find(path), std::string::npos) << reg_res->body;
    EXPECT_NE(reg_res->body.find("\"trained\":true"), std::string::npos) << reg_res->body;
}

TEST_F(LiveRegistryTest, TrainedRemovesFileFromQueue) {
    const std::string path = "/data/" + group() + "/pop.txt";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-U"));
    ASSERT_TRUE(trained("run-U", {path}, {10}));

    auto res = get_queue();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"entries\":[]"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, TrainedDeduplicatesInRegistry) {
    const std::string path = "/data/" + group() + "/dedup.txt";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-V"));
    ASSERT_TRUE(trained("run-V", {path}, {5}));

    // Adding and committing the same file again under a fresh run should be a no-op.
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-V2"));
    auto tr_res2 = trained("run-V2", {path}, {99});
    ASSERT_TRUE(tr_res2);
    EXPECT_EQ(tr_res2->status, 200);
    EXPECT_EQ(json_int(tr_res2->body, "trained"), 0) << tr_res2->body;

    // Registry still has exactly one entry for this file.
    auto reg_res = get_registry();
    ASSERT_TRUE(reg_res);
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = reg_res->body.find(path, pos)) != std::string::npos) {
        ++count;
        ++pos;
    }
    EXPECT_EQ(count, 1u) << "Duplicate entry written: " << reg_res->body;
}

TEST_F(LiveRegistryTest, TrainedRecordsSampleCountsInRegistry) {
    if (!server_current_format_)
        GTEST_SKIP() << "Server uses legacy registry format; rebuild and redeploy registry_server";
    const std::string path = "/data/" + group() + "/samples.txt";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-W"));
    ASSERT_TRUE(trained("run-W", {path}, {73}));

    auto reg_res = get_registry();
    ASSERT_TRUE(reg_res);
    EXPECT_NE(reg_res->body.find("\"num_samples\":73"), std::string::npos) << reg_res->body;
}

TEST_F(LiveRegistryTest, TrainedMultipleFilesInOneCall) {
    const std::string p1 = "/data/" + group() + "/m1.txt";
    const std::string p2 = "/data/" + group() + "/m2.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));
    ASSERT_TRUE(acquire("run-M"));

    auto tr_res = trained("run-M", {p1, p2}, {10, 20});
    ASSERT_TRUE(tr_res);
    EXPECT_EQ(tr_res->status, 200);
    EXPECT_EQ(json_int(tr_res->body, "trained"), 2) << tr_res->body;

    auto reg_res = get_registry();
    ASSERT_TRUE(reg_res);
    EXPECT_NE(reg_res->body.find(p1), std::string::npos) << reg_res->body;
    EXPECT_NE(reg_res->body.find(p2), std::string::npos) << reg_res->body;
}

// ===========================================================================
// Registry view — GET /registry/<group>/registry
// ===========================================================================

TEST_F(LiveRegistryTest, RegistryInitiallyEmpty) {
    auto res = get_registry();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"entries\":[]"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, RegistryContainsEntriesAfterTrained) {
    if (!server_current_format_)
        GTEST_SKIP() << "Server uses legacy registry format; rebuild and redeploy registry_server";
    const std::string path = "/data/" + group() + "/reg_check.txt";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-R"));
    ASSERT_TRUE(trained("run-R", {path}, {15}));

    auto res = get_registry();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"entries\":[{"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find(path), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"num_samples\":15"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"trained\":true"), std::string::npos) << res->body;
}

// ===========================================================================
// Runs view — GET /registry/<group>/runs
// ===========================================================================

TEST_F(LiveRegistryTest, RunsEmptyWhenNothingAcquired) {
    auto res = get_runs();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"runs\":{}"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, RunsShowsAssignedFilesPerRunId) {
    const std::string path = "/data/" + group() + "/run_view.txt";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-owner-view"));

    auto res = get_runs();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("run-owner-view"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find(path), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, RunsClearedAfterTrainedCommit) {
    const std::string path = "/data/" + group() + "/run_clear.txt";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-done"));
    ASSERT_TRUE(trained("run-done", {path}, {8}));

    auto res = get_runs();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"runs\":{}"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, RunsShowsMultipleRunsSimultaneously) {
    const std::string p1 = "/data/" + group() + "/r1.txt";
    const std::string p2 = "/data/" + group() + "/r2.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));

    // Each run acquires one file.
    ASSERT_TRUE(acquire("run-alpha", 1));
    ASSERT_TRUE(acquire("run-beta", 1));

    auto res = get_runs();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("run-alpha"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("run-beta"), std::string::npos) << res->body;
}

// ===========================================================================
// Full end-to-end workflow
// ===========================================================================

TEST_F(LiveRegistryTest, FullWorkflow_AddAcquireTrainVerify) {
    if (!server_current_format_)
        GTEST_SKIP() << "Server uses legacy registry format; rebuild and redeploy registry_server";
    const std::string p1 = "/data/" + group() + "/e2e_a.txt";
    const std::string p2 = "/data/" + group() + "/e2e_b.txt";

    // 1. Add two files to the pending queue.
    {
        auto r = add_pending(p1);
        ASSERT_TRUE(r) << "no response from pending/add";
        EXPECT_EQ(r->status, 200);
        EXPECT_NE(r->body.find("\"added\":true"), std::string::npos) << r->body;
    }
    {
        auto r = add_pending(p2);
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200);
    }

    // 2. Confirm both appear in the queue.
    {
        auto r = get_queue();
        ASSERT_TRUE(r);
        EXPECT_NE(r->body.find(p1), std::string::npos) << r->body;
        EXPECT_NE(r->body.find(p2), std::string::npos) << r->body;
    }

    // 3. Acquire files for a run.
    {
        auto r = acquire("e2e-run");
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200);
        EXPECT_NE(r->body.find(p1), std::string::npos) << r->body;
        EXPECT_NE(r->body.find(p2), std::string::npos) << r->body;
    }

    // 4. Confirm runs endpoint shows the assignment.
    {
        auto r = get_runs();
        ASSERT_TRUE(r);
        EXPECT_NE(r->body.find("e2e-run"), std::string::npos) << r->body;
    }

    // 5. Mark both files as trained.
    {
        auto r = trained("e2e-run", {p1, p2}, {100, 200});
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200);
        EXPECT_EQ(json_int(r->body, "trained"), 2) << r->body;
    }

    // 6. Queue should now be empty.
    {
        auto r = get_queue();
        ASSERT_TRUE(r);
        EXPECT_NE(r->body.find("\"entries\":[]"), std::string::npos) << r->body;
    }

    // 7. Registry should list both files as trained with correct sample counts.
    {
        auto r = get_registry();
        ASSERT_TRUE(r);
        EXPECT_NE(r->body.find(p1), std::string::npos) << r->body;
        EXPECT_NE(r->body.find(p2), std::string::npos) << r->body;
        EXPECT_NE(r->body.find("\"num_samples\":100"), std::string::npos) << r->body;
        EXPECT_NE(r->body.find("\"num_samples\":200"), std::string::npos) << r->body;
    }

    // 8. Runs view should be empty after the commit.
    {
        auto r = get_runs();
        ASSERT_TRUE(r);
        EXPECT_NE(r->body.find("\"runs\":{}"), std::string::npos) << r->body;
    }

    // 9. A new acquire should find nothing in the queue.
    {
        auto r = acquire("e2e-run2");
        ASSERT_TRUE(r);
        EXPECT_NE(r->body.find("\"acquired\":[]"), std::string::npos) << r->body;
    }
}

// Regression test for a bug where RemoteTransport::load_registry() parsed
// "num_samples" and "trained" with json_string(), a helper that only handles
// quoted string values ("key":"value"). Both fields are bare JSON literals
// ("num_samples":100, "trained":true), so the lookup always found nothing and
// silently defaulted to num_samples=0 / trained=false — regardless of the
// server's actual response. The tests above only ever inspected the raw HTTP
// response body, never fed it through the client parsing path, so this went
// unnoticed. This test exercises RemoteTransport itself, the way the real
// trainer does, and would have caught it.
TEST_F(LiveRegistryTest, RemoteTransportParsesTrainedAndNumSamplesAsRawJsonLiterals) {
    if (!server_current_format_)
        GTEST_SKIP() << "Server uses legacy registry format; rebuild and redeploy registry_server";
    const std::string p = "/data/" + group() + "/client_parse_check.txt";

    ASSERT_TRUE(add_pending(p));
    ASSERT_TRUE(acquire("client-parse-run"));
    {
        auto r = trained("client-parse-run", {p}, {321});
        ASSERT_TRUE(r);
        ASSERT_EQ(r->status, 200);
    }

    RemoteTransport rt("http://" + server_host() + ":" + std::to_string(server_port()), group());
    std::vector<DataVersion> entries;
    ASSERT_TRUE(rt.load_registry(entries));

    auto it = std::find_if(entries.begin(), entries.end(),
                           [&](const DataVersion& dv) { return dv.data_file == p; });
    ASSERT_NE(it, entries.end()) << "file not found in client-parsed registry";
    EXPECT_TRUE(it->trained) << "trained:true from the server was not parsed as true";
    EXPECT_EQ(it->num_samples, 321) << "num_samples was not parsed as a bare JSON number";
}

// ===========================================================================
// History — GET /registry/<group>/history[?model_id=<uuid>]  (Phase 2)
// ===========================================================================

TEST_F(LiveRegistryTest, HistoryInitiallyEmpty) {
    auto res = get_history();
    ASSERT_TRUE(res);
    if (res->status == 404)
        GTEST_SKIP() << "/history endpoint not available on this server version";
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"entries\":[]"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, TrainedWithModelIdAppearsInHistory) {
    const std::string path = "/data/" + group() + "/hist1.txt";
    const std::string model_id = "aaaabbbb-0001-0001-0001-000000000001";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-hist1"));
    ASSERT_TRUE(trained_with_model_id("run-hist1", {path}, model_id, {10}));

    auto res = get_history();
    ASSERT_TRUE(res);
    if (res->status == 404)
        GTEST_SKIP() << "/history endpoint not available on this server version";
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find(path), std::string::npos) << res->body;
    EXPECT_NE(res->body.find(model_id), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, HistoryFilterByModelIdReturnsOnlyMatches) {
    const std::string p1 = "/data/" + group() + "/hfilt1.txt";
    const std::string p2 = "/data/" + group() + "/hfilt2.txt";
    const std::string id1 = "aaaabbbb-0002-0002-0002-000000000002";
    const std::string id2 = "ccccdddd-0003-0003-0003-000000000003";

    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));
    ASSERT_TRUE(acquire("run-hf1", 1));
    ASSERT_TRUE(acquire("run-hf2", 1));
    ASSERT_TRUE(trained_with_model_id("run-hf1", {p1}, id1, {5}));
    ASSERT_TRUE(trained_with_model_id("run-hf2", {p2}, id2, {7}));

    // Filter for id1 — should see p1 but not p2.
    auto res = get_history(id1);
    ASSERT_TRUE(res);
    if (res->status == 404)
        GTEST_SKIP() << "/history endpoint not available on this server version";
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find(p1), std::string::npos)
        << "Expected p1 in filtered history: " << res->body;
    EXPECT_EQ(res->body.find(p2), std::string::npos)
        << "Did not expect p2 in filtered history: " << res->body;
}

TEST_F(LiveRegistryTest, HistoryWithoutFilterReturnsAll) {
    const std::string p1 = "/data/" + group() + "/hall1.txt";
    const std::string p2 = "/data/" + group() + "/hall2.txt";
    const std::string id1 = "aaaabbbb-0004-0004-0004-000000000004";
    const std::string id2 = "eeeeeeee-0005-0005-0005-000000000005";

    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));
    ASSERT_TRUE(acquire("run-hall1", 1));
    ASSERT_TRUE(acquire("run-hall2", 1));
    ASSERT_TRUE(trained_with_model_id("run-hall1", {p1}, id1, {3}));
    ASSERT_TRUE(trained_with_model_id("run-hall2", {p2}, id2, {9}));

    auto res = get_history();
    ASSERT_TRUE(res);
    if (res->status == 404)
        GTEST_SKIP() << "/history endpoint not available on this server version";
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find(p1), std::string::npos) << res->body;
    EXPECT_NE(res->body.find(p2), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, HistoryUnknownModelIdReturnsEmpty) {
    // Commit one entry with a known id, then query with a different id.
    const std::string path = "/data/" + group() + "/hunk1.txt";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-hunk1"));
    ASSERT_TRUE(
        trained_with_model_id("run-hunk1", {path}, "ffffffff-0006-0006-0006-000000000006", {1}));

    auto res = get_history("00000000-9999-9999-9999-999999999999");
    ASSERT_TRUE(res);
    if (res->status == 404)
        GTEST_SKIP() << "/history endpoint not available on this server version";
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"entries\":[]"), std::string::npos) << res->body;
}

TEST_F(LiveRegistryTest, TrainedWithoutModelIdHasEmptyModelIdInRegistry) {
    // Calling the plain trained() helper (no model_id field) should store an
    // empty model_id, keeping the registry backward-compatible.
    const std::string path = "/data/" + group() + "/hnomid.txt";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-hnomid"));
    ASSERT_TRUE(trained("run-hnomid", {path}, {20}));

    auto reg_res = get_registry();
    ASSERT_TRUE(reg_res);
    EXPECT_EQ(reg_res->status, 200);
    EXPECT_NE(reg_res->body.find(path), std::string::npos) << reg_res->body;
    // model_id key should either be absent or be an empty string — not a UUID.
    const auto mid_pos = reg_res->body.find("\"model_id\":\"");
    if (mid_pos != std::string::npos) {
        const auto val_start = mid_pos + std::string("\"model_id\":\"").size();
        const auto val_end = reg_res->body.find('"', val_start);
        const std::string stored_id = reg_res->body.substr(val_start, val_end - val_start);
        EXPECT_TRUE(stored_id.empty())
            << "Expected empty model_id for entry trained without one, got: " << stored_id;
    }
}

// ===========================================================================
// Phase 11: server-side dataset fetch — validation (no real network calls)
//
// These deliberately avoid exercising the DataFetcher happy path (which
// would perform real downloads from gutenberg.org / huggingface.co — slow,
// flaky, and against this project's convention of keeping test suites
// offline; see DataFetcherTests.cpp). They cover the validation added
// specifically to make these endpoints safe to expose over the network:
// invalid input must be rejected with 400 before DataFetcher is ever
// invoked, and the /upload path must reject path-traversal filenames.
// ===========================================================================

TEST_F(LiveRegistryTest, FetchGutenbergRejectsNonPositiveBookId) {
    auto res = fetch_gutenberg(0);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, FetchGutenbergRejectsNegativeBookId) {
    auto res = fetch_gutenberg(-5);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, FetchGutenbergRejectsUnsafeModelName) {
    // model_name flows into a cursor-file path (Phase 13) — '/' must be
    // rejected before it ever reaches the filesystem.
    auto res = fetch_gutenberg(1342, 10, "../../etc/passwd");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, FetchHuggingfaceRejectsUnsafeDatasetId) {
    // Shell metacharacters (space, ';') are outside the allow-list regex —
    // DataFetcher interpolates dataset_id unescaped into a curl command, so
    // this must be rejected before it ever reaches DataFetcher.
    auto res = fetch_huggingface("evil; rm -rf ~");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, FetchHuggingfaceRejectsUnsafeSplit) {
    std::ostringstream body;
    body << "{\"dataset_id\":\"daily_dialog\",\"num_pairs\":10,"
            "\"split\":\"train; touch /tmp/pwned\"}";
    auto res = client().Post(("/registry/" + group() + "/fetch/huggingface").c_str(), body.str(),
                             "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, FetchHuggingfaceRejectsUnsafeModelName) {
    // model_name flows into a cursor-file path (Phase 12) — '/' must be
    // rejected before it ever reaches the filesystem, unlike dataset_id/split
    // which legitimately need '/' for "org/name" identifiers.
    auto res = fetch_huggingface("daily_dialog", 10, "../../etc/passwd");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, UploadRejectsPathTraversalFilename) {
    auto res = upload("..%2F..%2F..%2Fetc%2Fpasswd", "malicious contents");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, UploadRejectsBareParentDirFilename) {
    auto res = upload("..", "malicious contents");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, UploadRejectsEmptyBody) {
    auto res = upload("empty.jsonl", "");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, UploadRejectsMissingFilename) {
    auto res = client().Post(("/registry/" + group() + "/upload").c_str(), "some bytes",
                             "application/octet-stream");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, UploadSucceedsAndAppearsInQueue) {
    const std::string filename = "smoke_" + group() + ".jsonl";
    const std::string contents = "{\"input\":\"hi\",\"response\":\"hello\"}\n";

    auto res = upload(filename, contents);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("\"added\":true"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find(filename), std::string::npos) << res->body;

    auto queue_res = get_queue();
    ASSERT_TRUE(queue_res);
    EXPECT_EQ(queue_res->status, 200);
    EXPECT_NE(queue_res->body.find(filename), std::string::npos) << queue_res->body;
}

TEST_F(LiveRegistryTest, UploadDeduplicatesSamePath) {
    const std::string filename = "dup_" + group() + ".jsonl";
    ASSERT_TRUE(upload(filename, "first"));

    auto res = upload(filename, "second");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"added\":true"), std::string::npos) << res->body;

    // Second upload overwrote the file but must not create a duplicate queue entry.
    auto queue_res = get_queue();
    ASSERT_TRUE(queue_res);
    int occurrences = 0;
    std::size_t pos = 0;
    while ((pos = queue_res->body.find(filename, pos)) != std::string::npos) {
        ++occurrences;
        pos += filename.size();
    }
    EXPECT_EQ(occurrences, 1) << queue_res->body;
}

// ===========================================================================
// Dataset metadata (Phase 15) — source/added_utc/size_bytes/num_entries/checksum
// ===========================================================================

TEST_F(LiveRegistryTest, PendingAddSetsManualSourceAndAddedUtc) {
    const std::string path = "/data/" + group() + "/manual_meta.txt";
    ASSERT_TRUE(add_pending(path));

    auto res = get_queue();
    ASSERT_TRUE(res);
    const auto entry_pos = res->body.find(path);
    ASSERT_NE(entry_pos, std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"source\":\"manual\"", entry_pos), std::string::npos) << res->body;
    // added_utc must be present and non-empty — look for the key followed by a
    // non-'"' character (i.e. not immediately closed, meaning some content is there).
    const auto added_pos = res->body.find("\"added_utc\":\"", entry_pos);
    ASSERT_NE(added_pos, std::string::npos) << res->body;
    EXPECT_NE(res->body[added_pos + std::string("\"added_utc\":\"").size()], '"') << res->body;
}

TEST_F(LiveRegistryTest, UploadComputesRealSizeAndEntryCountAndChecksum) {
    const std::string filename = "meta_" + group() + ".jsonl";
    const std::string contents =
        "{\"input\":\"a\",\"response\":\"b\"}\n{\"input\":\"c\",\"response\":\"d\"}\n";
    ASSERT_TRUE(upload(filename, contents));

    auto res = get_queue();
    ASSERT_TRUE(res);
    const auto entry_pos = res->body.find(filename);
    ASSERT_NE(entry_pos, std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"source\":\"upload\"", entry_pos), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"size_bytes\":" + std::to_string(contents.size()), entry_pos),
             std::string::npos)
        << res->body;
    // Uploaded content has exactly 2 non-empty lines.
    EXPECT_NE(res->body.find("\"num_entries\":2", entry_pos), std::string::npos) << res->body;
    // checksum is the registry's own size+mtime fingerprint — non-empty, and not
    // the "unknown" empty-string case that a not-locally-readable path would get.
    const auto checksum_pos = res->body.find("\"checksum\":\"", entry_pos);
    ASSERT_NE(checksum_pos, std::string::npos) << res->body;
    EXPECT_NE(res->body[checksum_pos + std::string("\"checksum\":\"").size()], '"') << res->body;
}

TEST_F(LiveRegistryTest, TrainedCarriesForwardSourceAndAddedUtcFromPendingEntry) {
    if (!server_current_format_)
        GTEST_SKIP() << "Server uses legacy registry format; rebuild and redeploy registry_server";
    const std::string filename = "carry_" + group() + ".jsonl";
    ASSERT_TRUE(upload(filename, "{\"input\":\"a\",\"response\":\"b\"}\n"));

    // Recover the exact registry_path the server assigned (data_dir/... — not
    // something this test constructs itself), and its added_utc, to compare
    // against the trained registry entry afterwards.
    auto queue_before = get_queue();
    ASSERT_TRUE(queue_before);
    const auto path_key_pos = queue_before->body.find("\"path\":\"");
    ASSERT_NE(path_key_pos, std::string::npos) << queue_before->body;
    const auto path_start = path_key_pos + std::string("\"path\":\"").size();
    const auto path_end = queue_before->body.find('"', path_start);
    const std::string full_path = queue_before->body.substr(path_start, path_end - path_start);

    const auto added_key_pos = queue_before->body.find("\"added_utc\":\"", path_end);
    ASSERT_NE(added_key_pos, std::string::npos) << queue_before->body;
    const auto added_start = added_key_pos + std::string("\"added_utc\":\"").size();
    const auto added_end = queue_before->body.find('"', added_start);
    const std::string original_added_utc = queue_before->body.substr(added_start, added_end - added_start);
    ASSERT_FALSE(original_added_utc.empty());

    ASSERT_TRUE(acquire("run-carry"));
    ASSERT_TRUE(trained("run-carry", {full_path}, {1}));

    auto reg_res = get_registry();
    ASSERT_TRUE(reg_res);
    const auto entry_pos = reg_res->body.find(full_path);
    ASSERT_NE(entry_pos, std::string::npos) << reg_res->body;
    EXPECT_NE(reg_res->body.find("\"source\":\"upload\"", entry_pos), std::string::npos)
        << reg_res->body;
    EXPECT_NE(reg_res->body.find("\"added_utc\":\"" + original_added_utc + "\"", entry_pos),
             std::string::npos)
        << "expected the trained entry to carry forward the pending entry's original "
           "added_utc ('"
        << original_added_utc << "') rather than resetting it: " << reg_res->body;
    // The file is still locally readable by the registry (it staged the upload
    // itself), so the checksum must be a real fingerprint, not the "REMOTE" placeholder.
    EXPECT_EQ(reg_res->body.find("\"checksum\":\"REMOTE\"", entry_pos), std::string::npos)
        << reg_res->body;
}

}  // namespace
