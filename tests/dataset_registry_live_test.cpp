// Live-server integration tests for the registry_server daemon (TD-028 Phase 9).
//
// Exercises every HTTP endpoint exposed by RegistryServer.cpp against a running
// instance.  Tests auto-skip when the server is unreachable so the suite can
// be included in the normal build without requiring the daemon to be running.
//
// Each test creates an isolated group name derived from the PID, timestamp, and
// a monotonic counter so concurrent test runs on the same server cannot collide.
//
// Configuration (env vars, all optional):
//   REGISTRY_SERVER_HOST   default: 127.0.0.1
//   REGISTRY_SERVER_PORT   default: 8082
//   REGISTRY_SERVER_DATA_DIR  the server's --data-dir; the fetch-cursor tests (TD-491) seed a
//                          cached dataset there and skip when it is unset
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
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
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

    // Helper: POST /registry/<group_>/acquire
    // {"run_id":"<rid>","max_files":<n>,"model_name":"<model_name>"}
    // acquire() is assignment-aware (see RegistryTransport::acquire): an
    // anonymous caller (model_name empty) can only claim unassigned entries,
    // never one assigned to a specific model — pass model_name to claim an
    // entry this test has assigned to that model.
    httplib::Result acquire(const std::string& run_id, int max_files = 0,
                            const std::string& model_name = "") {
        std::ostringstream body;
        body << "{\"run_id\":\"" << run_id << "\",\"max_files\":" << max_files
             << ",\"model_name\":\"" << model_name << "\"}";
        return client_->Post(("/registry/" + group_ + "/acquire").c_str(), body.str(),
                             "application/json");
    }

    // Helper: POST /registry/<group_>/session/next {"model_name":..., "run_id":...}
    httplib::Result next_session(const std::string& model_name, const std::string& run_id) {
        const std::string body =
            "{\"model_name\":\"" + model_name + "\",\"run_id\":\"" + run_id + "\"}";
        return client_->Post(("/registry/" + group_ + "/session/next").c_str(), body,
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

    // Helper: POST /registry/<group_>/assign {"model_name":"<model>","paths":[...],"count":N}
    httplib::Result assign(const std::string& model_name_raw_json,
                           const std::vector<std::string>& paths = {}, int count = 0) {
        std::ostringstream body;
        body << "{\"model_name\":\"" << model_name_raw_json << "\",\"paths\":[";
        for (std::size_t i = 0; i < paths.size(); ++i) {
            if (i)
                body << ',';
            body << '"' << paths[i] << '"';
        }
        body << "],\"count\":" << count << "}";
        return client_->Post(("/registry/" + group_ + "/assign").c_str(), body.str(),
                             "application/json");
    }

    // Helper: POST /registry/<group_>/unassign {"model_name":"<model>","paths":[...],"force":bool}
    httplib::Result unassign(const std::string& model_name_raw_json,
                             const std::vector<std::string>& paths = {}, bool force = false) {
        std::ostringstream body;
        body << "{\"model_name\":\"" << model_name_raw_json << "\",\"paths\":[";
        for (std::size_t i = 0; i < paths.size(); ++i) {
            if (i)
                body << ',';
            body << '"' << paths[i] << '"';
        }
        body << "],\"force\":" << (force ? "true" : "false") << "}";
        return client_->Post(("/registry/" + group_ + "/unassign").c_str(), body.str(),
                             "application/json");
    }

    // Helper: POST /registry/<group_>/delete {"paths":[...],"force":bool,"delete_files":bool}
    httplib::Result delete_entries(const std::vector<std::string>& paths, bool force = false,
                                   bool delete_files = false) {
        std::ostringstream body;
        body << "{\"paths\":[";
        for (std::size_t i = 0; i < paths.size(); ++i) {
            if (i)
                body << ',';
            body << '"' << paths[i] << '"';
        }
        body << "],\"force\":" << (force ? "true" : "false")
            << ",\"delete_files\":" << (delete_files ? "true" : "false") << "}";
        return client_->Post(("/registry/" + group_ + "/delete").c_str(), body.str(),
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
// TD-202: dataset_kind sub-pools — /registry/<group>/<kind>/<endpoint>
// ===========================================================================
//
// The route regex's optional kind segment ((?:/(encoder|decoder|world_model|chatbot))?) is what
// makes a kind-scoped and a legacy (no kind) URL land in physically separate pools on the same
// server — this is the actual black-box confirmation of that routing, not just a unit test of
// the regex in isolation.

TEST_F(LiveRegistryTest, KindScopedQueueIsDisjointFromLegacyQueue) {
    const std::string legacy_path = "/data/" + group() + "/legacy.txt";
    const std::string wm_path = "/data/" + group() + "/wm.txt";

    ASSERT_TRUE(add_pending(legacy_path));
    ASSERT_TRUE(client().Post(("/registry/" + group() + "/world_model/pending/add").c_str(),
                              "{\"path\":\"" + wm_path + "\"}", "application/json"));

    auto legacy_queue = get_queue();
    ASSERT_TRUE(legacy_queue);
    EXPECT_NE(legacy_queue->body.find(legacy_path), std::string::npos) << legacy_queue->body;
    EXPECT_EQ(legacy_queue->body.find(wm_path), std::string::npos)
        << "world_model-pool file leaked into the legacy queue: " << legacy_queue->body;

    auto wm_queue = client().Get(("/registry/" + group() + "/world_model/queue").c_str());
    ASSERT_TRUE(wm_queue);
    EXPECT_NE(wm_queue->body.find(wm_path), std::string::npos) << wm_queue->body;
    EXPECT_EQ(wm_queue->body.find(legacy_path), std::string::npos)
        << "legacy-pool file leaked into the world_model queue: " << wm_queue->body;
}

TEST_F(LiveRegistryTest, DifferentKindsOfTheSameGroupAreMutuallyDisjoint) {
    const std::string enc_path = "/data/" + group() + "/enc.txt";
    const std::string dec_path = "/data/" + group() + "/dec.txt";

    ASSERT_TRUE(client().Post(("/registry/" + group() + "/encoder/pending/add").c_str(),
                              "{\"path\":\"" + enc_path + "\"}", "application/json"));
    ASSERT_TRUE(client().Post(("/registry/" + group() + "/decoder/pending/add").c_str(),
                              "{\"path\":\"" + dec_path + "\"}", "application/json"));

    auto enc_queue = client().Get(("/registry/" + group() + "/encoder/queue").c_str());
    ASSERT_TRUE(enc_queue);
    EXPECT_NE(enc_queue->body.find(enc_path), std::string::npos) << enc_queue->body;
    EXPECT_EQ(enc_queue->body.find(dec_path), std::string::npos) << enc_queue->body;

    auto dec_queue = client().Get(("/registry/" + group() + "/decoder/queue").c_str());
    ASSERT_TRUE(dec_queue);
    EXPECT_NE(dec_queue->body.find(dec_path), std::string::npos) << dec_queue->body;
    EXPECT_EQ(dec_queue->body.find(enc_path), std::string::npos) << dec_queue->body;
}

TEST_F(LiveRegistryTest, InvalidKindSegmentIs404NotAValidationError) {
    // TD-202 design: the route regex's closed alternation IS the kind validation — an
    // unrecognized kind value simply can't match any route at all, so this must 404 like any
    // other unknown path, not reach a handler and come back 400.
    auto res = client().Get(("/registry/" + group() + "/not_a_real_kind/queue").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);
}

TEST_F(LiveRegistryTest, KindScopedAcquireOnlyClaimsFromItsOwnPool) {
    const std::string wm_path = "/data/" + group() + "/wm_acquire.txt";
    ASSERT_TRUE(client().Post(("/registry/" + group() + "/world_model/pending/add").c_str(),
                              "{\"path\":\"" + wm_path + "\"}", "application/json"));

    // A legacy (unkinded) acquire must not see the world_model-pool file.
    auto legacy_acq = acquire("run-td202-legacy");
    ASSERT_TRUE(legacy_acq);
    EXPECT_EQ(legacy_acq->body.find(wm_path), std::string::npos) << legacy_acq->body;

    std::ostringstream body;
    body << "{\"run_id\":\"run-td202-wm\",\"max_files\":0,\"model_name\":\"\"}";
    auto wm_acq = client().Post(("/registry/" + group() + "/world_model/acquire").c_str(),
                                body.str(), "application/json");
    ASSERT_TRUE(wm_acq);
    EXPECT_NE(wm_acq->body.find(wm_path), std::string::npos) << wm_acq->body;
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

TEST_F(LiveRegistryTest, SameRunCanReclaimItsOwnFiles) {
    const std::string path = "/data/" + group() + "/reclaimed.txt";
    ASSERT_TRUE(add_pending(path));

    // First acquire claims it.
    auto res1 = acquire("run-same");
    ASSERT_TRUE(res1);
    EXPECT_EQ(res1->status, 200);
    EXPECT_NE(res1->body.find(path), std::string::npos) << res1->body;

    // Same run_id, no release in between (simulates a crashed process
    // restarting and getting the same run_id back) — must get the file back,
    // not an empty result.
    auto res2 = acquire("run-same");
    ASSERT_TRUE(res2);
    EXPECT_EQ(res2->status, 200);
    EXPECT_NE(res2->body.find(path), std::string::npos) << res2->body;
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

TEST_F(LiveRegistryTest, AssignWithCountAssignsFirstNUnassignedOnly) {
    const std::string p1 = "/data/" + group() + "/count1.txt";
    const std::string p2 = "/data/" + group() + "/count2.txt";
    const std::string p3 = "/data/" + group() + "/count3.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));
    ASSERT_TRUE(add_pending(p3));

    auto res = assign("model-d", {}, 2);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(json_int(res->body, "assigned"), 2) << res->body;

    auto queue_res = get_queue();
    ASSERT_TRUE(queue_res);
    int occurrences = 0;
    std::size_t pos = 0;
    const std::string needle = "\"model_name\":\"model-d\"";
    while ((pos = queue_res->body.find(needle, pos)) != std::string::npos) {
        ++occurrences;
        pos += needle.size();
    }
    EXPECT_EQ(occurrences, 2) << queue_res->body;
}

TEST_F(LiveRegistryTest, AssignWithCountSkipsAlreadyAssignedEntries) {
    const std::string p1 = "/data/" + group() + "/countskip1.txt";
    const std::string p2 = "/data/" + group() + "/countskip2.txt";
    const std::string p3 = "/data/" + group() + "/countskip3.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));
    ASSERT_TRUE(add_pending(p3));

    ASSERT_TRUE(assign("model-x", {p1}));  // pre-assign p1 to a different model

    auto res = assign("model-d", {}, 5);  // ask for more than remain unassigned
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "assigned"), 2) << res->body;  // only p2, p3
}

TEST_F(LiveRegistryTest, AssignWithZeroCountAndEmptyPathsAssignsAll) {
    const std::string p1 = "/data/" + group() + "/zerocount1.txt";
    const std::string p2 = "/data/" + group() + "/zerocount2.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));

    auto res = assign("model-b", {}, 0);  // explicit count=0 == today's "assign all" default
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "assigned"), 2) << res->body;
}

TEST_F(LiveRegistryTest, AssignResponseIncludesPathsArray) {
    const std::string p1 = "/data/" + group() + "/pathsarr1.txt";
    ASSERT_TRUE(add_pending(p1));

    auto res = assign("model-e", {p1});
    ASSERT_TRUE(res);
    EXPECT_NE(res->body.find("\"paths\":[\"" + p1 + "\"]"), std::string::npos) << res->body;
}

// ===========================================================================
// Unassign — POST /registry/<group>/unassign
// ===========================================================================

TEST_F(LiveRegistryTest, UnassignBulkByModelNameClearsAllMatchingEntries) {
    const std::string p1 = "/data/" + group() + "/unassignbulk1.txt";
    const std::string p2 = "/data/" + group() + "/unassignbulk2.txt";
    const std::string p3 = "/data/" + group() + "/unassignbulk3.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));
    ASSERT_TRUE(add_pending(p3));
    ASSERT_TRUE(assign("model-e", {p1, p2, p3}));

    auto res = unassign("model-e");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(json_int(res->body, "unassigned"), 3) << res->body;

    auto queue_res = get_queue();
    ASSERT_TRUE(queue_res);
    EXPECT_EQ(queue_res->body.find("\"model_name\":\"model-e\""), std::string::npos)
        << queue_res->body;
}

TEST_F(LiveRegistryTest, UnassignSpecificPathsOnlyClearsListed) {
    const std::string p1 = "/data/" + group() + "/unassignspecific1.txt";
    const std::string p2 = "/data/" + group() + "/unassignspecific2.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(add_pending(p2));
    ASSERT_TRUE(assign("model-f", {p1, p2}));

    auto res = unassign("model-f", {p1});
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "unassigned"), 1) << res->body;

    auto queue_res = get_queue();
    ASSERT_TRUE(queue_res);
    const auto p2_pos = queue_res->body.find(p2);
    ASSERT_NE(p2_pos, std::string::npos);
    EXPECT_NE(queue_res->body.find("\"model_name\":\"model-f\"", p2_pos), std::string::npos)
        << queue_res->body;
}

TEST_F(LiveRegistryTest, UnassignSkipsActivelyClaimedEntryWithoutForce) {
    const std::string p1 = "/data/" + group() + "/unassignclaimed1.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(assign("model-g", {p1}));
    ASSERT_TRUE(acquire("run-Z", 1, "model-g"));

    auto res = unassign("model-g", {p1});
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "unassigned"), 0) << res->body;
    EXPECT_EQ(json_int(res->body, "skipped"), 1) << res->body;

    auto queue_res = get_queue();
    ASSERT_TRUE(queue_res);
    const auto p1_pos = queue_res->body.find(p1);
    ASSERT_NE(p1_pos, std::string::npos);
    EXPECT_NE(queue_res->body.find("\"model_name\":\"model-g\"", p1_pos), std::string::npos)
        << queue_res->body;
}

TEST_F(LiveRegistryTest, UnassignForceOverridesActiveClaim) {
    const std::string p1 = "/data/" + group() + "/unassignforce1.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(assign("model-g", {p1}));
    ASSERT_TRUE(acquire("run-Z", 1, "model-g"));

    auto res = unassign("model-g", {p1}, /*force=*/true);
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "unassigned"), 1) << res->body;
    EXPECT_EQ(json_int(res->body, "skipped"), 0) << res->body;
}

TEST_F(LiveRegistryTest, UnassignRejectsWhenBothPathsAndModelNameEmpty) {
    auto res = unassign("");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(LiveRegistryTest, UnassignReturnsZeroForNonMatchingModelName) {
    auto res = unassign("no-such-model");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(json_int(res->body, "unassigned"), 0) << res->body;
    EXPECT_EQ(json_int(res->body, "skipped"), 0) << res->body;
}

TEST_F(LiveRegistryTest, UnassignOwnershipCheckSkipsPathAssignedToDifferentModel) {
    const std::string p1 = "/data/" + group() + "/unassignownership1.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(assign("model-h", {p1}));

    auto res = unassign("model-i", {p1});
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "unassigned"), 0) << res->body;
}

// ===========================================================================
// Delete — POST /registry/<group>/delete
// ===========================================================================

TEST_F(LiveRegistryTest, DeleteRemovesPendingEntryByPath) {
    const std::string p1 = "/data/" + group() + "/delete1.txt";
    ASSERT_TRUE(add_pending(p1));

    auto res = delete_entries({p1});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(json_int(res->body, "deleted"), 1) << res->body;

    auto queue_res = get_queue();
    ASSERT_TRUE(queue_res);
    EXPECT_EQ(queue_res->body.find(p1), std::string::npos) << queue_res->body;
}

TEST_F(LiveRegistryTest, DeleteRemovesTrainedRegistryEntryByPath) {
    if (!server_current_format_) {
        GTEST_SKIP() << "server predates current registry format";
    }
    const std::string p1 = "/data/" + group() + "/deletetrained1.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(acquire("run-del", 1));
    ASSERT_TRUE(trained("run-del", {p1}, {5}));

    auto res = delete_entries({p1});
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "deleted"), 1) << res->body;

    auto reg_res = get_registry();
    ASSERT_TRUE(reg_res);
    EXPECT_EQ(reg_res->body.find(p1), std::string::npos) << reg_res->body;
}

TEST_F(LiveRegistryTest, DeleteReturnsNotFoundForUnknownPath) {
    auto res = delete_entries({"/no/such/file-" + group() + ".txt"});
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "deleted"), 0) << res->body;
    EXPECT_EQ(json_int(res->body, "not_found"), 1) << res->body;
}

TEST_F(LiveRegistryTest, DeleteSkipsActivelyClaimedPendingEntryWithoutForce) {
    const std::string p1 = "/data/" + group() + "/deleteclaimed1.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(acquire("run-Z", 1));

    auto res = delete_entries({p1});
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "deleted"), 0) << res->body;
    EXPECT_EQ(json_int(res->body, "skipped"), 1) << res->body;

    auto queue_res = get_queue();
    ASSERT_TRUE(queue_res);
    EXPECT_NE(queue_res->body.find(p1), std::string::npos) << queue_res->body;
}

TEST_F(LiveRegistryTest, DeleteForceOverridesActiveClaim) {
    const std::string p1 = "/data/" + group() + "/deleteforce1.txt";
    ASSERT_TRUE(add_pending(p1));
    ASSERT_TRUE(acquire("run-Z", 1));

    auto res = delete_entries({p1}, /*force=*/true);
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "deleted"), 1) << res->body;
}

TEST_F(LiveRegistryTest, DeleteWithoutDeleteFilesLeavesFileDeletedFlagFalse) {
    const std::string filename = "delete_nofiles_" + group() + ".txt";
    auto upload_res = upload(filename, "hello from delete test");
    ASSERT_TRUE(upload_res);
    ASSERT_EQ(upload_res->status, 200);
    const std::string path = json_string(upload_res->body, "path");
    ASSERT_FALSE(path.empty());

    // delete_files defaults to false — assert only via the response field,
    // never fs::exists, since the live server under test may be remote.
    auto res = delete_entries({path});
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "deleted"), 1) << res->body;
    EXPECT_FALSE(json_bool(res->body, "file_deleted")) << res->body;
}

TEST_F(LiveRegistryTest, DeleteWithDeleteFilesUnlinksUploadedFile) {
    const std::string filename = "delete_withfiles_" + group() + ".txt";
    auto upload_res = upload(filename, "hello from delete test 2");
    ASSERT_TRUE(upload_res);
    ASSERT_EQ(upload_res->status, 200);
    const std::string path = json_string(upload_res->body, "path");
    ASSERT_FALSE(path.empty());

    auto res = delete_entries({path}, /*force=*/false, /*delete_files=*/true);
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "deleted"), 1) << res->body;
    EXPECT_TRUE(json_bool(res->body, "file_deleted")) << res->body;
}

TEST_F(LiveRegistryTest, DeleteWithDeleteFilesRefusesPathOutsideDataDir) {
    // Deliberately never created on disk, so this test is valid whether the
    // live server under test is local or remote — the containment check (or
    // simple non-existence) both yield file_deleted=false either way.
    const std::string path = "/tmp/adai_external_" + group() + ".txt";
    ASSERT_TRUE(add_pending(path));

    auto res = delete_entries({path}, /*force=*/false, /*delete_files=*/true);
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "deleted"), 1) << res->body;  // metadata still purged
    EXPECT_FALSE(json_bool(res->body, "file_deleted")) << res->body;
}

TEST_F(LiveRegistryTest, DeleteMultiplePathsMixedOutcomes) {
    const std::string p_ok = "/data/" + group() + "/mixed_ok.txt";
    const std::string p_claimed = "/data/" + group() + "/mixed_claimed.txt";
    const std::string p_missing = "/no/such/mixed-" + group() + ".txt";
    ASSERT_TRUE(add_pending(p_ok));
    ASSERT_TRUE(add_pending(p_claimed));
    ASSERT_TRUE(acquire("run-mixed", 2));       // claims both
    ASSERT_TRUE(release("run-mixed", {p_ok}));  // release only p_ok back to unclaimed

    auto res = delete_entries({p_ok, p_claimed, p_missing});
    ASSERT_TRUE(res);
    EXPECT_EQ(json_int(res->body, "deleted"), 1) << res->body;    // p_ok
    EXPECT_EQ(json_int(res->body, "skipped"), 1) << res->body;    // p_claimed
    EXPECT_EQ(json_int(res->body, "not_found"), 1) << res->body;  // p_missing
}

TEST_F(LiveRegistryTest, DeleteRejectsEmptyPaths) {
    auto res = delete_entries({});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
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

    // TD-493: a re-queued unit trained by a run that claimed it IS recorded again (see
    // ReTrainedUnitIsRecordedAgain); only a forced report (empty run_id) still skips a unit that's
    // already recorded.
    ASSERT_TRUE(add_pending(path));
    auto tr_res2 = trained("", {path}, {99});
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

// TD-085: unlike TrainedDeduplicatesInRegistry (which covers the same path
// committed across two separate /trained calls), this covers the same path
// appearing TWICE in a single call's "files" array — handle_trained's
// `existing` dedup set was built once from the pre-existing registry and
// never updated as new entries were pushed within the loop, so a duplicate
// *within* one request bypassed the check entirely.
TEST_F(LiveRegistryTest, TrainedDeduplicatesWithinSingleCall) {
    const std::string path = "/data/" + group() + "/dup-single-call.txt";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("run-DUP"));

    auto tr_res = trained("run-DUP", {path, path}, {5, 7});
    ASSERT_TRUE(tr_res);
    EXPECT_EQ(tr_res->status, 200);
    EXPECT_EQ(json_int(tr_res->body, "trained"), 1)
        << "A path repeated within one /trained call must only be committed once: "
        << tr_res->body;

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
// session/next — Phase 3 run/session numbering
// ===========================================================================

TEST_F(LiveRegistryTest, SessionNextStartsAtOneAndIncrements) {
    const std::string model = "sess-model-" + group();
    const std::string run_id = "run-sess-" + group();

    auto r1 = next_session(model, run_id);
    ASSERT_TRUE(r1);
    EXPECT_EQ(200, r1->status);
    EXPECT_EQ(json_string(r1->body, "session_id"), "session-01") << r1->body;

    auto r2 = next_session(model, run_id);
    ASSERT_TRUE(r2);
    EXPECT_EQ(json_string(r2->body, "session_id"), "session-02") << r2->body;
}

TEST_F(LiveRegistryTest, SessionNextResetsForDifferentRunId) {
    const std::string model = "sess-model2-" + group();

    auto r1 = next_session(model, "run-x-" + group());
    ASSERT_TRUE(r1);
    EXPECT_EQ(json_string(r1->body, "session_id"), "session-01") << r1->body;

    // A different run_id starts its own counter at 1.
    auto r2 = next_session(model, "run-y-" + group());
    ASSERT_TRUE(r2);
    EXPECT_EQ(json_string(r2->body, "session_id"), "session-01") << r2->body;
}

TEST_F(LiveRegistryTest, SessionNextRejectsMissingRunId) {
    auto res = next_session("some-model", "");
    ASSERT_TRUE(res);
    EXPECT_EQ(400, res->status);
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

// TD-333: RemoteTransport commits and releases exact segments: two segments trained in one batch
// get their own records, a third released one is claimable by another run.
TEST_F(LiveRegistryTest, RemoteTransportCommitsAndReleasesExactSegments) {
    const std::string p = "/data/" + group() + "/segmented.jsonl";
    RemoteTransport rt("http://" + server_host() + ":" + std::to_string(server_port()), group());
    ASSERT_TRUE(rt.add_pending(p, 0, 10));
    ASSERT_TRUE(rt.add_pending(p, 10, 10));
    ASSERT_TRUE(rt.add_pending(p, 20, 10));
    const AcquireResponse resp = rt.acquire("seg-run", 0);
    ASSERT_EQ(resp.files.size(), 3u);

    DataVersion a, b;
    a.data_file = b.data_file = p;
    a.trained = b.trained = true;
    a.segment_start = 0;
    a.segment_count = 10;
    a.num_samples = 10;
    b.segment_start = 10;
    b.segment_count = 10;
    b.num_samples = 7;
    rt.commit_trained("seg-run", {a, b}, {}, {{p, 0, 10}, {p, 10, 10}});
    rt.release("seg-run", {}, {{p, 20, 10}});

    std::vector<DataVersion> reg;
    ASSERT_TRUE(rt.load_registry(reg));
    int records = 0, samples = 0;
    for (const auto& dv : reg)
        if (dv.data_file == p) {
            ++records;
            samples += dv.num_samples;
            EXPECT_GE(dv.segment_start, 0);
        }
    EXPECT_EQ(records, 2);
    EXPECT_EQ(samples, 17);
    const AcquireResponse next = rt.acquire("other-run", 0);
    ASSERT_EQ(next.files.size(), 1u);
    EXPECT_EQ(next.files[0].segment_start, 20);
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

    // TD-492: a second upload of a queued name is refused (it used to overwrite the file and say
    // added:true); overwrite=true replaces it (UploadOverwriteReplacesAnUnclaimedQueuedFile).
    auto res = upload(filename, "second");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 409) << res->body;
    EXPECT_NE(res->body.find("\"reason\":\"already_queued\""), std::string::npos) << res->body;

    // Still exactly one queue entry.
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

// ===========================================================================
// TD-490: a group name is one plain path component under data_dir
// ===========================================================================

// "/registry/../..." and "/registry/%2e%2e/..." both reached handlers as group "..", which
// resolved to data_dir's parent (a delete with delete_files unlinked a file there), and "." to
// data_dir itself. Every route now answers 400 for them before touching the filesystem.
TEST(LiveRegistryGroupNames, TraversalGroupsAreRejected) {
    if (!server_reachable())
        GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    const std::string body = R"({"paths":["/nonexistent/x.jsonl"],"delete_files":true})";
    for (const std::string group : {"%2e%2e", "%2E%2E", ".", "%2e", ".hidden"}) {
        auto queue = c.Get("/registry/" + group + "/queue");
        ASSERT_TRUE(queue) << group;
        EXPECT_EQ(queue->status, 400) << group << ": " << queue->body;
        auto del = c.Post("/registry/" + group + "/delete", body, "application/json");
        ASSERT_TRUE(del) << group;
        EXPECT_NE(del->status, 200) << group << ": " << del->body;
        auto add = c.Post("/registry/" + group + "/pending/add",
                          R"({"path":"/nonexistent/x.jsonl"})", "application/json");
        ASSERT_TRUE(add) << group;
        EXPECT_NE(add->status, 200) << group << ": " << add->body;
    }
    // A kind-scoped route gets the same check.
    auto kinded = c.Get("/registry/%2e%2e/chatbot/queue");
    ASSERT_TRUE(kinded);
    EXPECT_EQ(kinded->status, 400) << kinded->body;
}

TEST(LiveRegistryGroupNames, OrdinaryGroupNamesStillWork) {
    if (!server_reachable())
        GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    for (const std::string group : {make_group("ok"), std::string("Chatbot_v1.2-") + make_group()}) {
        auto res = c.Get("/registry/" + group + "/queue");
        ASSERT_TRUE(res) << group;
        EXPECT_EQ(res->status, 200) << group << ": " << res->body;
    }
}

// TD-310: fetch/huggingface allow-listed dataset_id/split as [A-Za-z0-9._-/]+, which let ".."
// through, and split went into paths unflattened. Traversal values now get 400 before any fetch.
TEST(LiveRegistryHuggingface, TraversalInDatasetIdOrSplitIsRejected) {
    if (!server_reachable())
        GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    const std::string group = make_group("hf");
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"org/ds", "x/../../ESCAPED"}, {"org/ds", ".."},    {"org/ds", "."},
        {"org/ds", "a/b"},             {"../ds", "train"},  {"org/../ds", "train"},
        {"/org/ds", "train"},          {"org/ds/", "train"}, {"org//ds", "train"}};
    for (const auto& [dataset_id, split] : cases) {
        const std::string body = R"({"dataset_id":")" + dataset_id + R"(","split":")" + split +
                                 R"(","num_pairs":5})";
        auto res = c.Post("/registry/" + group + "/fetch/huggingface", body, "application/json");
        ASSERT_TRUE(res) << dataset_id << " / " << split;
        EXPECT_EQ(res->status, 400) << dataset_id << " / " << split << ": " << res->body;
    }
}

// ===========================================================================
// TD-491: fetch cursors. These need the server's data directory to seed a cached dataset (no network):
// set REGISTRY_SERVER_DATA_DIR to the --data-dir the server was started with.
// ===========================================================================

namespace {

std::string server_data_dir() {
    const char* env = std::getenv("REGISTRY_SERVER_DATA_DIR");
    return env ? std::string(env) : "";
}

// Writes <data_dir>/<group>/datasets/org_cursor_train/full_dataset.jsonl, which
// DataFetcher::ensure_huggingface_cached() takes as the cached "org/cursor" train split.
void seed_hf_cache(const std::string& group, int rows) {
    const std::filesystem::path dir =
        std::filesystem::path(server_data_dir()) / group / "datasets" / "org_cursor_train";
    std::filesystem::create_directories(dir);
    std::ofstream f(dir / "full_dataset.jsonl");
    for (int i = 0; i < rows; ++i)
        f << "{\"input\":\"question " << i << "\",\"output\":\"answer " << i << "\"}\n";
}

httplib::Result fetch_cursor_slice(httplib::Client& c, const std::string& group, int num_pairs) {
    const std::string body = R"({"dataset_id":"org/cursor","split":"train","num_pairs":)" +
                             std::to_string(num_pairs) +
                             R"(,"input_field":"input","output_field":"output"})";
    return c.Post("/registry/" + group + "/fetch/huggingface", body, "application/json");
}

std::string json_field(const std::string& body, const std::string& key) {
    std::smatch m;
    const std::regex re("\"" + key + "\":\"?([^\",}]*)");
    return std::regex_search(body, m, re) ? m[1].str() : "";
}

int count_occurrences(const std::string& haystack, const std::string& needle) {
    int n = 0;
    for (std::size_t pos = haystack.find(needle); pos != std::string::npos;
         pos = haystack.find(needle, pos + needle.size()))
        ++n;
    return n;
}

}  // namespace

// Eight concurrent fetches of 10 pairs used to all serve row 0 into one file, queued once, while each
// reported added:true.
TEST(LiveRegistryFetchCursor, ConcurrentFetchesServeDistinctSlices) {
    if (!server_reachable() || server_data_dir().empty())
        GTEST_SKIP() << "needs REGISTRY_SERVER_HOST and REGISTRY_SERVER_DATA_DIR";
    const std::string group = make_group("cursor");
    seed_hf_cache(group, 200);

    std::vector<std::string> bodies(8);
    std::vector<int> statuses(8, 0);
    std::vector<std::thread> threads;
    std::atomic<int> busy_answers{0};
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            auto c = make_client();
            c.set_read_timeout(std::chrono::seconds(30));
            // TD-494: the server runs at most --max-concurrent-fetches fetches at once and answers
            // 429 to the rest; retry as a client should (the test doesn't wait out Retry-After).
            auto res = fetch_cursor_slice(c, group, 10);
            for (int retry = 0; res && res->status == 429 && retry < 200; ++retry) {
                if (retry == 0)
                    ++busy_answers;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                res = fetch_cursor_slice(c, group, 10);
            }
            if (res) {
                statuses[t] = res->status;
                bodies[t] = res->body;
            }
        });
    }
    for (auto& th : threads)
        th.join();

    std::set<std::string> rows, paths;
    for (int t = 0; t < 8; ++t) {
        EXPECT_EQ(statuses[t], 200) << "request " << t << ": " << bodies[t];
        rows.insert(json_field(bodies[t], "served_from_row"));
        paths.insert(json_field(bodies[t], "path"));
    }
    EXPECT_EQ(rows, (std::set<std::string>{"0", "10", "20", "30", "40", "50", "60", "70"}));
    EXPECT_EQ(paths.size(), 8u);

    auto c = make_client();
    auto queue = c.Get("/registry/" + group + "/queue");
    ASSERT_TRUE(queue);
    EXPECT_EQ(count_occurrences(queue->body, "org_cursor_train__unassigned_row"), 8) << queue->body;
}

// Once the cursor wraps, a new slice must not rewrite (and re-report) an earlier one, which may already be
// pending, claimed or trained.
TEST(LiveRegistryFetchCursor, WrappedFetchesNeverReuseASliceFile) {
    if (!server_reachable() || server_data_dir().empty())
        GTEST_SKIP() << "needs REGISTRY_SERVER_HOST and REGISTRY_SERVER_DATA_DIR";
    const std::string group = make_group("cursorwrap");
    seed_hf_cache(group, 20);
    auto c = make_client();
    std::set<std::string> paths;
    for (int i = 0; i < 6; ++i) {
        auto res = fetch_cursor_slice(c, group, 10);
        ASSERT_TRUE(res);
        ASSERT_EQ(res->status, 200) << res->body;
        EXPECT_NE(res->body.find("\"added\":true"), std::string::npos) << res->body;
        paths.insert(json_field(res->body, "path"));
    }
    EXPECT_EQ(paths.size(), 6u);
    auto queue = c.Get("/registry/" + group + "/queue");
    ASSERT_TRUE(queue);
    EXPECT_EQ(count_occurrences(queue->body, "org_cursor_train__unassigned_row"), 6) << queue->body;
}

// ===========================================================================
// TD-492: an upload never replaces bytes something depends on
// ===========================================================================

namespace {

std::string read_server_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace

// Verified before the fix: re-uploading a claimed file returned added:true and replaced the bytes under
// the claim.
TEST_F(LiveRegistryTest, UploadOverAClaimedFileIsRefused) {
    const std::string filename = "claimed_" + group() + ".jsonl";
    auto first = upload(filename, "{\"input\":\"a\",\"output\":\"b\"}\n");
    ASSERT_TRUE(first);
    ASSERT_EQ(first->status, 200) << first->body;
    const std::string path = json_field(first->body, "path");
    ASSERT_TRUE(acquire("r1"));

    for (const char* query : {"", "&overwrite=true"}) {
        auto res = client().Post(
            ("/registry/" + group() + "/upload?filename=" + filename + query).c_str(),
            "{\"input\":\"x\",\"output\":\"y\"}\n", "application/octet-stream");
        ASSERT_TRUE(res);
        EXPECT_EQ(res->status, 409) << query << ": " << res->body;
        EXPECT_NE(res->body.find("\"reason\":\"claimed\""), std::string::npos) << res->body;
    }
    if (!server_data_dir().empty())
        EXPECT_EQ(read_server_file(path), "{\"input\":\"a\",\"output\":\"b\"}\n");
}

TEST_F(LiveRegistryTest, UploadOverwriteReplacesAnUnclaimedQueuedFile) {
    const std::string filename = "replace_" + group() + ".jsonl";
    ASSERT_TRUE(upload(filename, "{\"input\":\"a\",\"output\":\"b\"}\n"));
    const std::string bigger = "{\"input\":\"a\",\"output\":\"b\"}\n{\"input\":\"c\",\"output\":\"d\"}\n";
    auto res = client().Post(
        ("/registry/" + group() + "/upload?filename=" + filename + "&overwrite=true").c_str(), bigger,
        "application/octet-stream");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("\"replaced\":true"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"added\":false"), std::string::npos) << res->body;

    auto queue = get_queue();
    ASSERT_TRUE(queue);
    EXPECT_EQ(count_occurrences(queue->body, filename), 1) << queue->body;
    EXPECT_NE(queue->body.find("\"size_bytes\":" + std::to_string(bigger.size())), std::string::npos)
        << queue->body;
    EXPECT_NE(queue->body.find("\"num_entries\":2"), std::string::npos) << queue->body;
}

TEST_F(LiveRegistryTest, UploadOverATrainedFileNeedsOverwrite) {
    const std::string filename = "trained_" + group() + ".jsonl";
    auto first = upload(filename, "{\"input\":\"a\",\"output\":\"b\"}\n");
    ASSERT_TRUE(first);
    const std::string path = json_field(first->body, "path");
    ASSERT_TRUE(acquire("r1"));
    auto t = trained("r1", {path}, {1});
    ASSERT_TRUE(t);
    ASSERT_EQ(t->status, 200) << t->body;

    auto res = upload(filename, "{\"input\":\"new\",\"output\":\"version\"}\n");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 409) << res->body;
    EXPECT_NE(res->body.find("\"reason\":\"trained\""), std::string::npos) << res->body;

    res = client().Post(
        ("/registry/" + group() + "/upload?filename=" + filename + "&overwrite=true").c_str(),
        "{\"input\":\"new\",\"output\":\"version\"}\n", "application/octet-stream");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("\"added\":true"), std::string::npos) << res->body;
}

// ===========================================================================
// TD-493: trained records only what the reporting run claimed, and records a re-trained unit again
// ===========================================================================

// Verified before the fix: "other" recorded r1's claimed file (trained:1) and r1's own report then
// recorded nothing (trained:0), so r1's sample count was lost.
TEST_F(LiveRegistryTest, TrainedFromAnotherRunIsRejected) {
    const std::string path = "/data/" + group() + "/claimed.jsonl";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("r1"));

    auto res = trained("other", {path}, {3});
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("\"trained\":0"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"rejected\":[\"" + path + "\"]"), std::string::npos) << res->body;
    auto reg = get_registry();
    ASSERT_TRUE(reg);
    EXPECT_EQ(reg->body.find(path), std::string::npos) << reg->body;

    res = trained("r1", {path}, {11});
    ASSERT_TRUE(res);
    EXPECT_NE(res->body.find("\"trained\":1"), std::string::npos) << res->body;
    reg = get_registry();
    ASSERT_TRUE(reg);
    EXPECT_EQ(count_occurrences(reg->body, path), 1) << reg->body;
    EXPECT_NE(reg->body.find("\"num_samples\":11"), std::string::npos) << reg->body;
    auto queue = get_queue();
    ASSERT_TRUE(queue);
    EXPECT_EQ(queue->body.find(path), std::string::npos) << queue->body;
}

TEST_F(LiveRegistryTest, ReTrainedUnitIsRecordedAgain) {
    const std::string path = "/data/" + group() + "/again.jsonl";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("r1"));
    auto res = trained("r1", {path}, {5});
    ASSERT_TRUE(res);
    EXPECT_NE(res->body.find("\"trained\":1"), std::string::npos) << res->body;

    ASSERT_TRUE(add_pending(path));  // queued again (e.g. an overwritten upload, TD-492)
    ASSERT_TRUE(acquire("r2"));
    res = trained("r2", {path}, {7});
    ASSERT_TRUE(res);
    EXPECT_NE(res->body.find("\"trained\":1"), std::string::npos) << res->body;

    auto reg = get_registry();
    ASSERT_TRUE(reg);
    EXPECT_EQ(count_occurrences(reg->body, path), 2) << reg->body;
    EXPECT_NE(reg->body.find("\"num_samples\":5"), std::string::npos) << reg->body;
    EXPECT_NE(reg->body.find("\"num_samples\":7"), std::string::npos) << reg->body;
}

// ===========================================================================
// TD-494: request bodies are parsed as JSON (whitespace, escapes), session keys can't collide, and
// /health reports the session counter store
// ===========================================================================

// Verified before the fix: a space after the colon gave "400 run_id required".
TEST_F(LiveRegistryTest, BodiesWithWhitespaceAreParsed) {
    const std::string path = "/data/" + group() + "/spaced.jsonl";
    auto add = client().Post(("/registry/" + group() + "/pending/add").c_str(),
                             "{ \"path\" : \"" + path + "\" }", "application/json");
    ASSERT_TRUE(add);
    EXPECT_EQ(add->status, 200) << add->body;
    auto res = client().Post(("/registry/" + group() + "/acquire").c_str(),
                             "{\n  \"run_id\": \"r2\",\n  \"max_files\": 0\n}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find(path), std::string::npos) << res->body;
}

// é used to be stored literally (backslash and all); it's now the UTF-8 character.
TEST_F(LiveRegistryTest, EscapesInBodiesAreDecoded) {
    const std::string raw = "/data/" + group() + "/caf\\u00e9.jsonl";
    const std::string decoded = "/data/" + group() + "/caf\xc3\xa9.jsonl";
    auto add = client().Post(("/registry/" + group() + "/pending/add").c_str(),
                             "{\"path\":\"" + raw + "\"}", "application/json");
    ASSERT_TRUE(add);
    ASSERT_EQ(add->status, 200) << add->body;
    auto queue = get_queue();
    ASSERT_TRUE(queue);
    EXPECT_NE(queue->body.find(decoded), std::string::npos) << queue->body;
    EXPECT_EQ(queue->body.find("\\u00e9"), std::string::npos) << queue->body;
}

TEST_F(LiveRegistryTest, SessionKeysCantCollideThroughAPipe) {
    auto res = next_session("a|b", "c");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400) << res->body;
    res = next_session("a", "b|c");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400) << res->body;
    res = next_session("model-a", "run-01");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
}

TEST(LiveRegistryHealthDetail, ReportsTheSessionCounterStore) {
    if (!server_reachable())
        GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    auto res = c.Get("/health");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"session_counters\":true"), std::string::npos) << res->body;
}

// ===========================================================================
// TD-494: claim leases. The server re-reads pending_files.txt on every request, so a test can
// backdate a claim by rewriting its claimed_unix column (the 11th) instead of waiting out a lease.
// ===========================================================================

namespace {

// Sets claimed_unix of @p path's entry in @p group's pending file (needs REGISTRY_SERVER_DATA_DIR).
void set_claimed_unix(const std::string& group, const std::string& path, long long value) {
    const std::filesystem::path file =
        std::filesystem::path(server_data_dir()) / group / "pending_files.txt";
    std::ifstream in(file);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        if (line.compare(0, path.size() + 1, path + "\t") == 0) {
            std::vector<std::string> cols;
            std::size_t start = 0;
            for (std::size_t tab; (tab = line.find('\t', start)) != std::string::npos; start = tab + 1)
                cols.push_back(line.substr(start, tab - start));
            cols.push_back(line.substr(start));
            cols.resize(std::max<std::size_t>(cols.size(), 11));
            cols[10] = std::to_string(value);
            line.clear();
            for (std::size_t i = 0; i < cols.size(); ++i)
                line += (i ? "\t" : "") + cols[i];
        }
        lines.push_back(line);
    }
    in.close();
    std::ofstream out(file, std::ios::trunc);
    for (const auto& l : lines)
        out << l << '\n';
}

}  // namespace

TEST_F(LiveRegistryTest, AcquireReportsTheClaimLease) {
    ASSERT_TRUE(add_pending("/data/" + group() + "/lease.jsonl"));
    auto res = acquire("r1");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("\"claim_lease_seconds\":"), std::string::npos) << res->body;
    EXPECT_EQ(res->body.find("\"claim_lease_seconds\":0"), std::string::npos)
        << "leases are on by default: " << res->body;
}

// A crashed trainer's claim used to stay until released by hand; once its lease runs out, another
// run takes it, and the old run can neither renew nor record it.
TEST_F(LiveRegistryTest, ExpiredClaimIsTakenByAnotherRun) {
    if (server_data_dir().empty())
        GTEST_SKIP() << "needs REGISTRY_SERVER_DATA_DIR";
    const std::string path = "/data/" + group() + "/expired.jsonl";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("r1"));
    set_claimed_unix(group(), path, 1);  // claimed in 1970

    auto res = acquire("r2");
    ASSERT_TRUE(res);
    EXPECT_NE(res->body.find(path), std::string::npos) << res->body;
    auto renew = client().Post(("/registry/" + group() + "/renew").c_str(), "{\"run_id\":\"r1\"}",
                               "application/json");
    ASSERT_TRUE(renew);
    EXPECT_NE(renew->body.find("\"renewed\":0"), std::string::npos) << renew->body;
    auto t = trained("r1", {path}, {4});
    ASSERT_TRUE(t);
    EXPECT_NE(t->body.find("\"trained\":0"), std::string::npos) << t->body;
}

TEST_F(LiveRegistryTest, RenewKeepsAClaim) {
    if (server_data_dir().empty())
        GTEST_SKIP() << "needs REGISTRY_SERVER_DATA_DIR";
    const std::string path = "/data/" + group() + "/renewed.jsonl";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("r1"));
    set_claimed_unix(group(), path, 1);
    auto renew = client().Post(("/registry/" + group() + "/renew").c_str(), "{\"run_id\":\"r1\"}",
                               "application/json");
    ASSERT_TRUE(renew);
    EXPECT_NE(renew->body.find("\"renewed\":1"), std::string::npos) << renew->body;
    auto res = acquire("r2");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->body.find(path), std::string::npos) << res->body;
}

// A claim made before leases existed has no claimed_unix; its lease counts from server start, so
// upgrading the server never releases a running trainer's files at once.
TEST_F(LiveRegistryTest, ClaimWithoutATimeIsNotReleasedAtOnce) {
    if (server_data_dir().empty())
        GTEST_SKIP() << "needs REGISTRY_SERVER_DATA_DIR";
    const std::string path = "/data/" + group() + "/legacy.jsonl";
    ASSERT_TRUE(add_pending(path));
    ASSERT_TRUE(acquire("r1"));
    set_claimed_unix(group(), path, 0);
    auto res = acquire("r2");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->body.find(path), std::string::npos) << res->body;
}


// The trainer's side: RemoteTransport reads the lease from acquire and renews through /renew.
TEST_F(LiveRegistryTest, RemoteTransportReadsTheLeaseAndRenews) {
    const std::string path = "/data/" + group() + "/remote_renew.jsonl";
    ASSERT_TRUE(add_pending(path));
    RemoteTransport rt("http://" + server_host() + ":" + std::to_string(server_port()), group());
    const auto resp = rt.acquire("r-remote", 0);
    ASSERT_FALSE(resp.failed);
    ASSERT_EQ(resp.files.size(), 1u);
    EXPECT_GT(resp.claim_lease_seconds, 0);
    EXPECT_EQ(rt.renew_claims("r-remote"), 1);
    EXPECT_EQ(rt.renew_claims("nobody"), 0);
}
