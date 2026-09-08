/**
 * Unit tests for ModelNameClient — the HTTP client side of the ModelNameService
 * protocol. Previously untested (the @adai-status note crediting coverage via
 * mns_manager_gui_test.cpp was wrong — that file never references
 * ModelNameClient at all).
 *
 * Exercises the client against a real httplib::Server standing in for
 * mns_server, verifying both the requests it sends (method, path, body) and
 * how it parses canned responses — the same style as metrics_push_client_test.cpp.
 *
 * Only compiled when BUILD_MNS_SERVER is defined (httplib + sqlite3 found),
 * matching ModelNameClient.cpp's own guard — the client is a no-op stub
 * otherwise and there is nothing meaningful to test.
 */
#include <gtest/gtest.h>
#include "Config.hpp"
#include "ModelNameClient.hpp"

#ifdef BUILD_MNS_SERVER
#include <httplib.h>
#include <chrono>
#include <thread>

// Port range 44800–44899 — reserved for ModelNameClientTests
static constexpr int kPort_RegisterModel = 44800;
static constexpr int kPort_SetTraining = 44801;
static constexpr int kPort_SetCandidate = 44802;
static constexpr int kPort_PushProgress = 44803;
static constexpr int kPort_ResolveModel = 44804;
static constexpr int kPort_GetArchitectureFound = 44805;
static constexpr int kPort_GetArchitectureNotFound = 44806;
static constexpr int kPort_ResolveRole = 44807;
static constexpr int kPort_ListModels = 44808;
static constexpr int kPort_Promote = 44809;
static constexpr int kPort_UpdateRunGroup = 44810;
static constexpr int kPort_ErrorStatus = 44811;
static constexpr int kPort_ConnectionFailure = 44812;

namespace {

/** Starts svr.listen() in a background thread; blocks until is_running(). */
std::thread start_server(httplib::Server& svr, int port) {
    std::thread t([&svr, port]() { svr.listen("127.0.0.1", port); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!svr.is_running()) {
        EXPECT_LT(std::chrono::steady_clock::now(), deadline) << "Server failed to start";
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return t;
}

void stop_server(httplib::Server& svr, std::thread& t) {
    svr.stop();
    if (t.joinable())
        t.join();
}

std::string base_url(int port) {
    return "http://127.0.0.1:" + std::to_string(port);
}

adai::ServiceConfig make_arch() {
    adai::ServiceConfig cfg;
    cfg.d_model = 128;
    cfg.num_heads = 4;
    cfg.d_ff = 512;
    cfg.num_encoder_layers = 2;
    cfg.num_decoder_layers = 2;
    cfg.max_seq_length = 256;
    cfg.run_group = "group-a";
    return cfg;
}

}  // namespace

// ============================================================================
// register_model
// ============================================================================

TEST(ModelNameClientTest, RegisterModelSendsExpectedFieldsAndParsesId) {
    httplib::Server svr;
    std::string captured_path, captured_body;
    svr.Post("/models", [&](const httplib::Request& req, httplib::Response& res) {
        captured_path = req.path;
        captured_body = req.body;
        res.status = 201;
        res.set_content(R"({"model_id":"uuid-123"})", "application/json");
    });
    auto t = start_server(svr, kPort_RegisterModel);

    adai::ModelNameClient client(base_url(kPort_RegisterModel));
    std::string id = client.register_model("adai-chatbot-v3", "chatbot", make_arch(),
                                            {{"env", "test"}});

    stop_server(svr, t);

    EXPECT_EQ(id, "uuid-123");
    EXPECT_EQ(captured_path, "/models");
    EXPECT_NE(captured_body.find("\"model_name\":\"adai-chatbot-v3\""), std::string::npos);
    EXPECT_NE(captured_body.find("\"role\":\"chatbot\""), std::string::npos);
    EXPECT_NE(captured_body.find("\"run_group\":\"group-a\""), std::string::npos);
    EXPECT_NE(captured_body.find("\"d_model\":128"), std::string::npos);
    EXPECT_NE(captured_body.find("\"env\":\"test\""), std::string::npos);
}

// ============================================================================
// set_training
// ============================================================================

TEST(ModelNameClientTest, SetTrainingSendsStateAndNewRunReturnsRunId) {
    httplib::Server svr;
    std::string captured_path, captured_body;
    svr.Put("/models/adai-chatbot-v3/state", [&](const httplib::Request& req, httplib::Response& res) {
        captured_path = req.path;
        captured_body = req.body;
        res.status = 200;
        res.set_content(R"({"run_id":"run-02"})", "application/json");
    });
    auto t = start_server(svr, kPort_SetTraining);

    adai::ModelNameClient client(base_url(kPort_SetTraining));
    std::string run_id = client.set_training("adai-chatbot-v3", /*new_run=*/true, "session-key-1");

    stop_server(svr, t);

    EXPECT_EQ(run_id, "run-02");
    EXPECT_EQ(captured_path, "/models/adai-chatbot-v3/state");
    EXPECT_NE(captured_body.find("\"state\":\"training\""), std::string::npos);
    EXPECT_NE(captured_body.find("\"new_run\":true"), std::string::npos);
    EXPECT_NE(captured_body.find("\"metrics_session_key\":\"session-key-1\""), std::string::npos);
}

TEST(ModelNameClientTest, SetTrainingOmitsSessionKeyWhenEmpty) {
    httplib::Server svr;
    std::string captured_body;
    svr.Put("/models/m/state", [&](const httplib::Request& req, httplib::Response& res) {
        captured_body = req.body;
        res.status = 200;
        res.set_content(R"({"run_id":"run-01"})", "application/json");
    });
    auto t = start_server(svr, kPort_SetTraining + 50);

    adai::ModelNameClient client(base_url(kPort_SetTraining + 50));
    client.set_training("m", /*new_run=*/false);

    stop_server(svr, t);

    EXPECT_EQ(captured_body.find("metrics_session_key"), std::string::npos);
    EXPECT_NE(captured_body.find("\"new_run\":false"), std::string::npos);
}

// ============================================================================
// set_candidate
// ============================================================================

TEST(ModelNameClientTest, SetCandidateSendsArtifactAndSummary) {
    httplib::Server svr;
    std::string captured_body;
    svr.Put("/models/m/state", [&](const httplib::Request& req, httplib::Response& res) {
        captured_body = req.body;
        res.status = 200;
        res.set_content("{}", "application/json");
    });
    auto t = start_server(svr, kPort_SetCandidate);

    adai::ModelNameClient client(base_url(kPort_SetCandidate));
    adai::ArtifactLocation artifact;
    artifact.host = "host1";
    artifact.path = "/weights/model.bin";
    artifact.checksum = "abc123";
    artifact.format = "adai-native";
    EXPECT_NO_THROW(
        client.set_candidate("m", "run-01", artifact, {{"final_loss", "0.42"}}));

    stop_server(svr, t);

    EXPECT_NE(captured_body.find("\"state\":\"candidate\""), std::string::npos);
    EXPECT_NE(captured_body.find("\"run_id\":\"run-01\""), std::string::npos);
    EXPECT_NE(captured_body.find("\"path\":\"/weights/model.bin\""), std::string::npos);
    EXPECT_NE(captured_body.find("\"final_loss\":\"0.42\""), std::string::npos);
}

// ============================================================================
// push_progress
// ============================================================================

TEST(ModelNameClientTest, PushProgressSendsEpochAndLossFields) {
    httplib::Server svr;
    std::string captured_path, captured_body;
    svr.Put("/models/m/progress", [&](const httplib::Request& req, httplib::Response& res) {
        captured_path = req.path;
        captured_body = req.body;
        res.status = 200;
        res.set_content("{}", "application/json");
    });
    auto t = start_server(svr, kPort_PushProgress);

    adai::ModelNameClient client(base_url(kPort_PushProgress));
    EXPECT_NO_THROW(client.push_progress("m", "run-01", "session-01", 5, 1.234, 1.100));

    stop_server(svr, t);

    EXPECT_EQ(captured_path, "/models/m/progress");
    EXPECT_NE(captured_body.find("\"epoch\":5"), std::string::npos);
    EXPECT_NE(captured_body.find("\"session_id\":\"session-01\""), std::string::npos);
}

// ============================================================================
// resolve_model
// ============================================================================

TEST(ModelNameClientTest, ResolveModelParsesAllFields) {
    httplib::Server svr;
    svr.Get("/models/m/resolve", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(
            R"({"model_id":"uuid-1","model_name":"m","state":"production","run_group":"g1",)"
            R"("host":"h1","path":"/p","checksum":"c1","format":"adai-native"})",
            "application/json");
    });
    auto t = start_server(svr, kPort_ResolveModel);

    adai::ModelNameClient client(base_url(kPort_ResolveModel));
    adai::ResolvedModel rm = client.resolve_model("m");

    stop_server(svr, t);

    EXPECT_EQ(rm.model_id, "uuid-1");
    EXPECT_EQ(rm.model_name, "m");
    EXPECT_EQ(rm.state, "production");
    EXPECT_EQ(rm.run_group, "g1");
    EXPECT_EQ(rm.artifact.host, "h1");
    EXPECT_EQ(rm.artifact.path, "/p");
    EXPECT_EQ(rm.artifact.checksum, "c1");
    EXPECT_EQ(rm.artifact.format, "adai-native");
}

// ============================================================================
// get_architecture
// ============================================================================

TEST(ModelNameClientTest, GetArchitectureParsesArchBlock) {
    httplib::Server svr;
    svr.Get("/models/m", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(
            R"({"model_name":"m","arch":{"d_model":256,"num_heads":8,"d_ff":1024,)"
            R"("num_encoder_layers":4,"num_decoder_layers":4,"max_seq_length":512}})",
            "application/json");
    });
    auto t = start_server(svr, kPort_GetArchitectureFound);

    adai::ModelNameClient client(base_url(kPort_GetArchitectureFound));
    auto arch = client.get_architecture("m");

    stop_server(svr, t);

    ASSERT_TRUE(arch.has_value());
    EXPECT_EQ(arch->d_model, 256u);
    EXPECT_EQ(arch->num_heads, 8u);
    EXPECT_EQ(arch->d_ff, 1024u);
    EXPECT_EQ(arch->num_encoder_layers, 4u);
    EXPECT_EQ(arch->num_decoder_layers, 4u);
    EXPECT_EQ(arch->max_seq_length, 512u);
}

TEST(ModelNameClientTest, GetArchitectureReturnsNulloptOn404) {
    httplib::Server svr;
    svr.Get("/models/missing", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 404;
        res.set_content("{}", "application/json");
    });
    auto t = start_server(svr, kPort_GetArchitectureNotFound);

    adai::ModelNameClient client(base_url(kPort_GetArchitectureNotFound));
    auto arch = client.get_architecture("missing");

    stop_server(svr, t);

    EXPECT_FALSE(arch.has_value());
}

// ============================================================================
// resolve_role
// ============================================================================

TEST(ModelNameClientTest, ResolveRoleHitsProductionEndpoint) {
    httplib::Server svr;
    std::string captured_path;
    svr.Get("/roles/chatbot/production", [&](const httplib::Request& req, httplib::Response& res) {
        captured_path = req.path;
        res.status = 200;
        res.set_content(
            R"({"model_id":"uuid-2","model_name":"prod-model","state":"production",)"
            R"("run_group":"","host":"","path":"","checksum":"","format":"adai-native"})",
            "application/json");
    });
    auto t = start_server(svr, kPort_ResolveRole);

    adai::ModelNameClient client(base_url(kPort_ResolveRole));
    adai::ResolvedModel rm = client.resolve_role("chatbot");

    stop_server(svr, t);

    EXPECT_EQ(captured_path, "/roles/chatbot/production");
    EXPECT_EQ(rm.model_name, "prod-model");
}

// ============================================================================
// list_models
// ============================================================================

TEST(ModelNameClientTest, ListModelsAppliesFiltersAndParsesMultipleRecords) {
    httplib::Server svr;
    std::string captured_path;
    svr.Get("/models", [&](const httplib::Request& req, httplib::Response& res) {
        captured_path = req.target;
        res.status = 200;
        res.set_content(
            R"({"models":[)"
            R"({"model_id":"1","model_name":"m1","state":"production","role":"chatbot","updated_utc":"t1"}},)"
            R"({"model_id":"2","model_name":"m2","state":"candidate","role":"chatbot","updated_utc":"t2"}}]})",
            "application/json");
    });
    auto t = start_server(svr, kPort_ListModels);

    adai::ModelNameClient client(base_url(kPort_ListModels));
    auto models = client.list_models("production", "chatbot", 10);

    stop_server(svr, t);

    EXPECT_NE(captured_path.find("limit=10"), std::string::npos);
    EXPECT_NE(captured_path.find("state=production"), std::string::npos);
    EXPECT_NE(captured_path.find("role=chatbot"), std::string::npos);
    ASSERT_EQ(models.size(), 2u);
    EXPECT_EQ(models[0].model_name, "m1");
    EXPECT_EQ(models[0].state, "production");
    EXPECT_EQ(models[1].model_name, "m2");
    EXPECT_EQ(models[1].state, "candidate");
}

// ============================================================================
// promote
// ============================================================================

TEST(ModelNameClientTest, PromoteSendsModelNameToRoleEndpoint) {
    httplib::Server svr;
    std::string captured_path, captured_body;
    svr.Put("/roles/chatbot/production", [&](const httplib::Request& req, httplib::Response& res) {
        captured_path = req.path;
        captured_body = req.body;
        res.status = 200;
        res.set_content("{}", "application/json");
    });
    auto t = start_server(svr, kPort_Promote);

    adai::ModelNameClient client(base_url(kPort_Promote));
    EXPECT_NO_THROW(client.promote("chatbot", "m2"));

    stop_server(svr, t);

    EXPECT_EQ(captured_path, "/roles/chatbot/production");
    EXPECT_NE(captured_body.find("\"model_name\":\"m2\""), std::string::npos);
}

// ============================================================================
// update_run_group
// ============================================================================

TEST(ModelNameClientTest, UpdateRunGroupSendsNewGroup) {
    httplib::Server svr;
    std::string captured_path, captured_body;
    svr.Put("/models/m/run_group", [&](const httplib::Request& req, httplib::Response& res) {
        captured_path = req.path;
        captured_body = req.body;
        res.status = 200;
        res.set_content("{}", "application/json");
    });
    auto t = start_server(svr, kPort_UpdateRunGroup);

    adai::ModelNameClient client(base_url(kPort_UpdateRunGroup));
    EXPECT_NO_THROW(client.update_run_group("m", "group-b"));

    stop_server(svr, t);

    EXPECT_EQ(captured_path, "/models/m/run_group");
    EXPECT_NE(captured_body.find("\"run_group\":\"group-b\""), std::string::npos);
}

// ============================================================================
// Error handling
// ============================================================================

TEST(ModelNameClientTest, NonSuccessStatusThrowsRuntimeError) {
    httplib::Server svr;
    svr.Get("/models/m/resolve", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 409;
        res.set_content(R"({"error":"conflict"})", "application/json");
    });
    auto t = start_server(svr, kPort_ErrorStatus);

    adai::ModelNameClient client(base_url(kPort_ErrorStatus));
    EXPECT_THROW(client.resolve_model("m"), std::runtime_error);

    stop_server(svr, t);
}

TEST(ModelNameClientTest, ConnectionFailureThrowsRuntimeError) {
    // Nothing listening on this port.
    adai::ModelNameClient client(base_url(kPort_ConnectionFailure), /*timeout_ms=*/200);
    EXPECT_THROW(client.resolve_model("m"), std::runtime_error);
}

#else  // !BUILD_MNS_SERVER

TEST(ModelNameClientTest, SkippedWithoutBuildMnsServer) {
    GTEST_SKIP() << "BUILD_MNS_SERVER not defined (httplib/sqlite3 not found) — "
                    "ModelNameClient's HTTP methods are no-op stubs, nothing to test";
}

#endif  // BUILD_MNS_SERVER
