// Live-server integration tests for the mns_server daemon.
//
// Exercises every HTTP endpoint against a running ModelNameService instance.
// Tests auto-skip when the server is unreachable so the suite runs in CI
// without requiring the daemon to be up.
//
// Each test generates unique model and role names derived from PID, timestamp,
// and a monotonic counter to prevent collisions across concurrent runs.
//
// Configuration (env vars, both optional):
//   MNS_SERVER_HOST   default: localhost
//   MNS_SERVER_PORT   default: 8083

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string>
#include <unistd.h>

#include <httplib.h>

namespace {

// ---------------------------------------------------------------------------
// Connection helpers
// ---------------------------------------------------------------------------

std::string server_host() {
    const char* env = std::getenv("MNS_SERVER_HOST");
    return env ? std::string(env) : "localhost";
}

int server_port() {
    const char* env = std::getenv("MNS_SERVER_PORT");
    return env ? std::atoi(env) : 8083;
}

httplib::Client make_client() {
    httplib::Client c(server_host(), server_port());
    c.set_connection_timeout(std::chrono::seconds(5));
    c.set_read_timeout(std::chrono::seconds(5));
    c.set_write_timeout(std::chrono::seconds(5));
    return c;
}

bool server_reachable() {
    auto c   = make_client();
    auto res = c.Get("/health");
    return res && res->status == 200;
}

// Unique model name: lt<pid>t<time>c<counter>[-tag]
std::string make_model(const std::string& tag = "") {
    static std::atomic<int> counter{0};
    std::ostringstream oss;
    oss << "lt" << static_cast<int>(::getpid())
        << "t" << static_cast<long>(std::time(nullptr))
        << "c" << counter.fetch_add(1);
    if (!tag.empty()) oss << "-" << tag;
    return oss.str();
}

// Unique role name (same scheme, different prefix so it doesn't collide with model names)
std::string make_role(const std::string& tag = "") {
    static std::atomic<int> counter{0};
    std::ostringstream oss;
    oss << "role" << static_cast<int>(::getpid())
        << "c" << counter.fetch_add(1);
    if (!tag.empty()) oss << "-" << tag;
    return oss.str();
}

// ---------------------------------------------------------------------------
// Minimal JSON helpers
// ---------------------------------------------------------------------------

std::string json_str(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    const auto pos = body.find(needle);
    if (pos == std::string::npos) return {};
    const auto start = pos + needle.size();
    const auto end   = body.find('"', start);
    if (end == std::string::npos) return {};
    return body.substr(start, end - start);
}

bool json_bool(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos) return false;
    const auto val_pos = pos + needle.size();
    return body.substr(val_pos, 4) == "true";
}

bool body_contains(const std::string& body, const std::string& substr) {
    return body.find(substr) != std::string::npos;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test fixture — skips all tests when server is unreachable
// ---------------------------------------------------------------------------

class MNSLiveTest : public ::testing::Test {
   protected:
    void SetUp() override {
        if (!server_reachable()) {
            GTEST_SKIP() << "mns_server not reachable at "
                         << server_host() << ":" << server_port()
                         << " — set MNS_SERVER_HOST / MNS_SERVER_PORT to override";
        }
    }
};

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, HealthReturnsOk) {
    auto c   = make_client();
    auto res = c.Get("/health");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_EQ("ok", json_str(res->body, "status"));
}

// ---------------------------------------------------------------------------
// Model registration
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, RegisterModel_ReturnsUUIDAndInitializingState) {
    const auto name = make_model("reg");
    auto c = make_client();
    const std::string body =
        "{\"model_name\":\"" + name + "\""
        ",\"role\":\"chatbot\""
        ",\"arch\":{\"d_model\":128,\"num_heads\":4,\"d_ff\":512"
        ",\"num_encoder_layers\":2,\"num_decoder_layers\":2,\"max_seq_length\":256}"
        ",\"tags\":{\"owner\":\"test\"}}";
    auto res = c.Post("/models", body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(201, res->status);
    EXPECT_FALSE(json_str(res->body, "model_id").empty());
    EXPECT_EQ("initializing", json_str(res->body, "state"));
}

TEST_F(MNSLiveTest, RegisterModel_DuplicateReturns409) {
    const auto name = make_model("dup");
    auto c = make_client();
    const std::string body = "{\"model_name\":\"" + name + "\"}";
    auto r1 = c.Post("/models", body, "application/json");
    ASSERT_TRUE(r1);
    EXPECT_EQ(201, r1->status);
    auto r2 = c.Post("/models", body, "application/json");
    ASSERT_TRUE(r2);
    EXPECT_EQ(409, r2->status);
}

TEST_F(MNSLiveTest, RegisterModel_InvalidNameReturns400) {
    auto c = make_client();
    const std::string body = "{\"model_name\":\"UPPER_CASE_NAME\"}";
    auto res = c.Post("/models", body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(400, res->status);
}

TEST_F(MNSLiveTest, RegisterModel_MissingNameReturns400) {
    auto c   = make_client();
    auto res = c.Post("/models", "{\"role\":\"chatbot\"}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(400, res->status);
}

// ---------------------------------------------------------------------------
// GET /models/{name} and GET /models
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, GetModel_ReturnsRecord) {
    const auto name = make_model("get");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");

    auto res = c.Get("/models/" + name);
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_EQ(name, json_str(res->body, "model_name"));
    EXPECT_EQ("initializing", json_str(res->body, "state"));
}

TEST_F(MNSLiveTest, GetModel_NotFoundReturns404) {
    auto c   = make_client();
    auto res = c.Get("/models/no-such-model-xyzzy");
    ASSERT_TRUE(res);
    EXPECT_EQ(404, res->status);
}

TEST_F(MNSLiveTest, ListModels_ContainsRegisteredModel) {
    const auto name = make_model("list");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");

    auto res = c.Get("/models");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_TRUE(body_contains(res->body, name));
}

// ---------------------------------------------------------------------------
// GET /models/{name}/resolve
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, ResolveModel_InitializingReturns409) {
    const auto name = make_model("resolv");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");

    auto res = c.Get("/models/" + name + "/resolve");
    ASSERT_TRUE(res);
    EXPECT_EQ(409, res->status);
}

// ---------------------------------------------------------------------------
// State transitions
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, StateTransition_InitializingToTraining) {
    const auto name = make_model("trans");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");

    const std::string body = "{\"state\":\"training\",\"run_id\":\"run-001\"}";
    auto res = c.Put("/models/" + name + "/state", body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_EQ("training", json_str(res->body, "state"));
}

TEST_F(MNSLiveTest, StateTransition_TrainingToCandidate_AttachesArtifact) {
    const auto name = make_model("cand");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"training\",\"run_id\":\"run-abc\"}", "application/json");

    const std::string body =
        "{\"state\":\"candidate\",\"run_id\":\"run-abc\""
        ",\"artifact\":{\"host\":\"testhost\",\"path\":\"/tmp/model.bin\""
        ",\"checksum\":\"abc123\",\"format\":\"adai-native\"}"
        ",\"training_summary\":{\"epochs\":\"5\",\"final_loss\":\"1.23\"}}";
    auto res = c.Put("/models/" + name + "/state", body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_EQ("candidate", json_str(res->body, "state"));
    EXPECT_TRUE(body_contains(res->body, "testhost"));
}

TEST_F(MNSLiveTest, StateTransition_InvalidReturns409) {
    const auto name = make_model("inv");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");

    // initializing -> production is not a valid transition
    auto res = c.Put("/models/" + name + "/state",
                     "{\"state\":\"production\"}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(400, res->status);  // "production" is not accepted directly
}

TEST_F(MNSLiveTest, StateTransition_TrainingLock_SameRunIdIdempotent) {
    const auto name = make_model("lock");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"training\",\"run_id\":\"run-1\"}", "application/json");
    // Same run_id — should be accepted
    auto res = c.Put("/models/" + name + "/state",
                     "{\"state\":\"training\",\"run_id\":\"run-1\"}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
}

TEST_F(MNSLiveTest, ExplicitRetire) {
    const auto name = make_model("retire");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"training\",\"run_id\":\"r1\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"candidate\",\"run_id\":\"r1\"}", "application/json");

    auto res = c.Put("/models/" + name + "/state",
                     "{\"state\":\"retired\"}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_EQ("retired", json_str(res->body, "state"));
}

// ---------------------------------------------------------------------------
// Role promotion
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, Promote_CandidateToProduction) {
    const auto name = make_model("prod");
    const auto role = make_role("chatbot");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\",\"role\":\"" + role + "\"}",
           "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"training\",\"run_id\":\"r1\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"candidate\",\"run_id\":\"r1\"}", "application/json");

    const std::string pbody = "{\"model_name\":\"" + name + "\"}";
    auto res = c.Put("/roles/" + role + "/production", pbody, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_EQ(name, json_str(res->body, "promoted"));
    EXPECT_EQ(role, json_str(res->body, "role"));
}

TEST_F(MNSLiveTest, Promote_NonCandidateReturns409) {
    const auto name = make_model("nonprod");
    const auto role = make_role("nr");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");
    // Still in initializing state — cannot promote

    const std::string pbody = "{\"model_name\":\"" + name + "\"}";
    auto res = c.Put("/roles/" + role + "/production", pbody, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(409, res->status);
}

TEST_F(MNSLiveTest, Promote_AutoRetiresPreviousProduction) {
    const auto name1 = make_model("v1");
    const auto name2 = make_model("v2");
    const auto role  = make_role("auto");
    auto c = make_client();

    // Register and promote v1
    auto promote_to_candidate = [&](const std::string& mname) {
        c.Post("/models", "{\"model_name\":\"" + mname + "\",\"role\":\"" + role + "\"}",
               "application/json");
        c.Put("/models/" + mname + "/state",
              "{\"state\":\"training\",\"run_id\":\"r\"}", "application/json");
        c.Put("/models/" + mname + "/state",
              "{\"state\":\"candidate\",\"run_id\":\"r\"}", "application/json");
    };
    promote_to_candidate(name1);
    c.Put("/roles/" + role + "/production",
          "{\"model_name\":\"" + name1 + "\"}", "application/json");

    // Register and promote v2 — v1 should be auto-retired
    promote_to_candidate(name2);
    auto res = c.Put("/roles/" + role + "/production",
                     "{\"model_name\":\"" + name2 + "\"}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_EQ(name1, json_str(res->body, "retired"));

    // Verify v1 is now retired
    auto v1res = c.Get("/models/" + name1);
    ASSERT_TRUE(v1res);
    EXPECT_EQ("retired", json_str(v1res->body, "state"));
}

// ---------------------------------------------------------------------------
// Role resolution
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, ResolveRole_ReturnsArtifact) {
    const auto name = make_model("rres");
    const auto role = make_role("rres");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\",\"role\":\"" + role + "\"}",
           "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"training\",\"run_id\":\"r\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"candidate\",\"run_id\":\"r\""
          ",\"artifact\":{\"host\":\"h\",\"path\":\"/p\",\"checksum\":\"c\",\"format\":\"adai-native\"}}",
          "application/json");
    c.Put("/roles/" + role + "/production",
          "{\"model_name\":\"" + name + "\"}", "application/json");

    auto res = c.Get("/roles/" + role + "/production");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_EQ(name, json_str(res->body, "model_name"));
    EXPECT_EQ("production", json_str(res->body, "state"));
}

TEST_F(MNSLiveTest, ResolveRole_NoProductionReturns404) {
    const auto role = make_role("noprod");
    auto c   = make_client();
    auto res = c.Get("/roles/" + role + "/production");
    ASSERT_TRUE(res);
    EXPECT_EQ(404, res->status);
}

TEST_F(MNSLiveTest, ListRoles_ContainsPromotedRole) {
    const auto name = make_model("rlst");
    const auto role = make_role("rlst");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\",\"role\":\"" + role + "\"}",
           "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"training\",\"run_id\":\"r\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"candidate\",\"run_id\":\"r\"}", "application/json");
    c.Put("/roles/" + role + "/production",
          "{\"model_name\":\"" + name + "\"}", "application/json");

    auto res = c.Get("/roles");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_TRUE(body_contains(res->body, role));
}

// ---------------------------------------------------------------------------
// Resolve model by name
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, ResolveModel_CandidateReturnsArtifact) {
    const auto name = make_model("mres");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"training\",\"run_id\":\"r\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"candidate\",\"run_id\":\"r\""
          ",\"artifact\":{\"host\":\"h\",\"path\":\"/mymodel.bin\""
          ",\"checksum\":\"chk\",\"format\":\"adai-native\"}}",
          "application/json");

    auto res = c.Get("/models/" + name + "/resolve");
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_EQ(name, json_str(res->body, "model_name"));
    EXPECT_EQ("/mymodel.bin", json_str(res->body, "path"));
}

// ---------------------------------------------------------------------------
// Hard delete
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, DeleteModel_InitializingSucceeds) {
    const auto name = make_model("del");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");

    auto res = c.Delete("/models/" + name);
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_TRUE(json_bool(res->body, "deleted"));

    // Verify gone
    auto res2 = c.Get("/models/" + name);
    ASSERT_TRUE(res2);
    EXPECT_EQ(404, res2->status);
}

TEST_F(MNSLiveTest, DeleteModel_ProductionReturns409) {
    const auto name = make_model("delprod");
    const auto role = make_role("delprod");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\",\"role\":\"" + role + "\"}",
           "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"training\",\"run_id\":\"r\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"candidate\",\"run_id\":\"r\"}", "application/json");
    c.Put("/roles/" + role + "/production",
          "{\"model_name\":\"" + name + "\"}", "application/json");

    auto res = c.Delete("/models/" + name);
    ASSERT_TRUE(res);
    EXPECT_EQ(409, res->status);
}

// ---------------------------------------------------------------------------
// Phase 2: /models/{name}/datasets proxy  (TD-028 Phase 2)
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, DatasetsEndpoint_Returns501WhenNoRegistryConfigured) {
    // The mns_server running during tests is not started with --registry-url,
    // so the datasets endpoint must return 501 Not Implemented.
    const auto name = make_model("ds501");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");

    auto res = c.Get("/models/" + name + "/datasets");
    ASSERT_TRUE(res);
    EXPECT_EQ(501, res->status);
    EXPECT_TRUE(body_contains(res->body, "error")) << res->body;
}

TEST_F(MNSLiveTest, DatasetsEndpoint_Returns404ForUnknownModel) {
    auto c = make_client();
    auto res = c.Get("/models/no-such-model-xyzzy-datasets/datasets");
    ASSERT_TRUE(res);
    // Either 404 (model not found) or 501 (no registry) are acceptable,
    // but it must not be a 500 or a crash.
    EXPECT_TRUE(res->status == 404 || res->status == 501)
        << "Unexpected status: " << res->status << " body: " << res->body;
}

// ---------------------------------------------------------------------------
// Phase 2: training_history stored and retrievable
// ---------------------------------------------------------------------------

TEST_F(MNSLiveTest, TrainingHistory_StoredAfterStateTransitions) {
    const auto name = make_model("hist");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");

    // Transition: initializing → training → candidate
    c.Put("/models/" + name + "/state",
          "{\"state\":\"training\",\"run_id\":\"run-hist-1\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"candidate\",\"run_id\":\"run-hist-1\""
          ",\"artifact\":{\"host\":\"h\",\"path\":\"/m.bin\",\"checksum\":\"c\""
          ",\"format\":\"adai-native\"}"
          ",\"training_summary\":{\"final_loss\":\"0.42\",\"epochs\":\"3\"}}",
          "application/json");

    // GET /models/{name} should include training_history with at least one entry.
    auto res = c.Get("/models/" + name);
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    EXPECT_TRUE(body_contains(res->body, "training_history"))
        << "Expected training_history in model response: " << res->body;
    EXPECT_TRUE(body_contains(res->body, "run-hist-1"))
        << "Expected run_id in training_history: " << res->body;
}

TEST_F(MNSLiveTest, TrainingHistory_PersistsAcrossGetModel) {
    const auto name = make_model("histp");
    auto c = make_client();
    c.Post("/models", "{\"model_name\":\"" + name + "\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"training\",\"run_id\":\"run-histp-1\"}", "application/json");
    c.Put("/models/" + name + "/state",
          "{\"state\":\"candidate\",\"run_id\":\"run-histp-1\"}", "application/json");

    // Two independent GET calls must both return the training_history.
    auto r1 = c.Get("/models/" + name);
    auto r2 = c.Get("/models/" + name);
    ASSERT_TRUE(r1);
    ASSERT_TRUE(r2);
    EXPECT_EQ(200, r1->status);
    EXPECT_EQ(200, r2->status);
    EXPECT_TRUE(body_contains(r1->body, "run-histp-1")) << r1->body;
    EXPECT_TRUE(body_contains(r2->body, "run-histp-1")) << r2->body;
}
