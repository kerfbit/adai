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
static constexpr int kPort_GetArchitectureComposedFromEncoderDecoder = 44813;
static constexpr int kPort_GetArchitectureLegacyChatbotFallsBackToInlineArch = 44814;
static constexpr int kPort_GetConnection = 44815;
static constexpr int kPort_LinkWorldModelSuccess = 44816;
static constexpr int kPort_LinkWorldModelFailure = 44817;

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

// TD-196: a new-style chatbot (kind=="chatbot", connection.encoder_name/decoder_name both set,
// no meaningful inline "arch") composes its ModelArchitecture from 2 additional
// GET /models/{name} calls against the linked encoder/decoder records instead of reading its own
// (unused, all-zero) inline fields — d_model/num_heads/d_ff/max_seq_length from the encoder,
// num_encoder_layers from the encoder's own layer count, num_decoder_layers from the decoder's.
TEST(ModelNameClientTest, GetArchitectureComposesFromLinkedEncoderAndDecoder) {
    httplib::Server svr;
    svr.Get("/models/my-chatbot", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(
            R"({"model_name":"my-chatbot","kind":"chatbot",)"
            R"("connection":{"encoder_name":"my-enc","decoder_name":"my-dec",)"
            R"("world_model_name":"","world_model_inject_every_n_layers":0,)"
            R"("hippocampal_memory_enabled":false,"hippocampal_memory_capacity":512,)"
            R"("hippocampal_repetition_alpha":0,"hippocampal_repetition_decay":0.95,)"
            R"("hippocampal_cross_reference_alpha":0,"hippocampal_association_decay":0.95,)"
            R"("sigreg_lambda":1,"sigreg_num_sketches":64},)"
            R"("arch":{"d_model":0,"num_heads":0,"d_ff":0,"num_encoder_layers":0,)"
            R"("num_decoder_layers":0,"max_seq_length":0}})",
            "application/json");
    });
    svr.Get("/models/my-enc", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(
            R"({"model_name":"my-enc","kind":"encoder",)"
            R"("arch":{"d_model":128,"num_heads":4,"d_ff":512,)"
            R"("num_encoder_layers":3,"num_decoder_layers":0,"max_seq_length":256}})",
            "application/json");
    });
    svr.Get("/models/my-dec", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(
            R"({"model_name":"my-dec","kind":"decoder",)"
            R"("arch":{"d_model":128,"num_heads":4,"d_ff":512,)"
            R"("num_encoder_layers":0,"num_decoder_layers":5,"max_seq_length":256}})",
            "application/json");
    });
    auto t = start_server(svr, kPort_GetArchitectureComposedFromEncoderDecoder);

    adai::ModelNameClient client(base_url(kPort_GetArchitectureComposedFromEncoderDecoder));
    auto arch = client.get_architecture("my-chatbot");

    stop_server(svr, t);

    ASSERT_TRUE(arch.has_value());
    EXPECT_EQ(arch->d_model, 128u);
    EXPECT_EQ(arch->num_heads, 4u);
    EXPECT_EQ(arch->d_ff, 512u);
    EXPECT_EQ(arch->max_seq_length, 256u);
    EXPECT_EQ(arch->num_encoder_layers, 3u) << "must come from the encoder's own layer count";
    EXPECT_EQ(arch->num_decoder_layers, 5u) << "must come from the decoder's own layer count";
}

// TD-196: a legacy chatbot (kind=="chatbot" but connection.encoder_name/decoder_name empty)
// must fall back to its own inline "arch" fields exactly as before this feature existed — no
// extra GET calls, no change to the pre-TD-196 behavior this signature has always had.
TEST(ModelNameClientTest, GetArchitectureLegacyChatbotFallsBackToInlineArch) {
    httplib::Server svr;
    int request_count = 0;
    svr.Get("/models/legacy-chatbot", [&](const httplib::Request&, httplib::Response& res) {
        ++request_count;
        res.status = 200;
        res.set_content(
            R"({"model_name":"legacy-chatbot","kind":"chatbot",)"
            R"("connection":{"encoder_name":"","decoder_name":""},)"
            R"("arch":{"d_model":64,"num_heads":2,"d_ff":256,)"
            R"("num_encoder_layers":2,"num_decoder_layers":2,"max_seq_length":128}})",
            "application/json");
    });
    auto t = start_server(svr, kPort_GetArchitectureLegacyChatbotFallsBackToInlineArch);

    adai::ModelNameClient client(
        base_url(kPort_GetArchitectureLegacyChatbotFallsBackToInlineArch));
    auto arch = client.get_architecture("legacy-chatbot");

    stop_server(svr, t);

    ASSERT_TRUE(arch.has_value());
    EXPECT_EQ(arch->d_model, 64u);
    EXPECT_EQ(arch->num_encoder_layers, 2u);
    EXPECT_EQ(arch->num_decoder_layers, 2u);
    EXPECT_EQ(request_count, 1) << "empty encoder_name/decoder_name must skip the composed-"
                                   "resolution path entirely — exactly 1 GET, no more";
}

// ============================================================================
// get_connection (TD-196)
// ============================================================================

TEST(ModelNameClientTest, GetConnectionParsesAllFields) {
    httplib::Server svr;
    svr.Get("/models/my-chatbot", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(
            R"({"model_name":"my-chatbot","kind":"chatbot",)"
            R"("connection":{"encoder_name":"my-enc","decoder_name":"my-dec",)"
            R"("world_model_name":"my-wm","world_model_inject_every_n_layers":2,)"
            R"("hippocampal_memory_enabled":true,"hippocampal_memory_capacity":256,)"
            R"("hippocampal_repetition_alpha":0.1,"hippocampal_repetition_decay":0.9,)"
            R"("hippocampal_cross_reference_alpha":0.2,"hippocampal_association_decay":0.8,)"
            R"("sigreg_lambda":2.5,"sigreg_num_sketches":32}})",
            "application/json");
    });
    auto t = start_server(svr, kPort_GetConnection);

    adai::ModelNameClient client(base_url(kPort_GetConnection));
    auto conn = client.get_connection("my-chatbot");

    stop_server(svr, t);

    ASSERT_TRUE(conn.has_value());
    EXPECT_EQ(conn->encoder_name, "my-enc");
    EXPECT_EQ(conn->decoder_name, "my-dec");
    EXPECT_EQ(conn->world_model_name, "my-wm");
    EXPECT_EQ(conn->world_model_inject_every_n_layers, 2u);
    EXPECT_TRUE(conn->hippocampal_memory_enabled);
    EXPECT_EQ(conn->hippocampal_memory_capacity, 256u);
    EXPECT_FLOAT_EQ(conn->hippocampal_repetition_alpha, 0.1f);
    EXPECT_FLOAT_EQ(conn->hippocampal_repetition_decay, 0.9f);
    EXPECT_FLOAT_EQ(conn->hippocampal_cross_reference_alpha, 0.2f);
    EXPECT_FLOAT_EQ(conn->hippocampal_association_decay, 0.8f);
    EXPECT_FLOAT_EQ(conn->sigreg_lambda, 2.5f);
    EXPECT_EQ(conn->sigreg_num_sketches, 32u);
}

TEST(ModelNameClientTest, GetConnectionReturnsNulloptOn404) {
    httplib::Server svr;
    svr.Get("/models/missing", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 404;
        res.set_content("{}", "application/json");
    });
    auto t = start_server(svr, kPort_GetArchitectureNotFound + 100);

    adai::ModelNameClient client(base_url(kPort_GetArchitectureNotFound + 100));
    auto conn = client.get_connection("missing");

    stop_server(svr, t);

    EXPECT_FALSE(conn.has_value());
}

// ============================================================================
// link_world_model (TD-196)
// ============================================================================

TEST(ModelNameClientTest, LinkWorldModelSendsAllFieldsAndReturnsTrueOn200) {
    httplib::Server svr;
    std::string captured_path, captured_body;
    svr.Post("/models/my-chatbot/link-world-model",
             [&](const httplib::Request& req, httplib::Response& res) {
                 captured_path = req.path;
                 captured_body = req.body;
                 res.status = 200;
                 res.set_content(R"({"status":"ok","world_model_name":"my-wm"})",
                                 "application/json");
             });
    auto t = start_server(svr, kPort_LinkWorldModelSuccess);

    adai::ModelNameClient client(base_url(kPort_LinkWorldModelSuccess));
    const bool ok = client.link_world_model("my-chatbot", "my-wm", /*inject_every_n_layers=*/2,
                                            /*hippocampal_memory_enabled=*/true,
                                            /*hippocampal_memory_capacity=*/256, 0.1f, 0.9f, 0.2f,
                                            0.8f);

    stop_server(svr, t);

    EXPECT_TRUE(ok);
    EXPECT_EQ(captured_path, "/models/my-chatbot/link-world-model");
    EXPECT_NE(captured_body.find("\"world_model_name\":\"my-wm\""), std::string::npos);
    EXPECT_NE(captured_body.find("\"world_model_inject_every_n_layers\":2"), std::string::npos);
    EXPECT_NE(captured_body.find("\"hippocampal_memory_enabled\":true"), std::string::npos);
    EXPECT_NE(captured_body.find("\"hippocampal_memory_capacity\":256"), std::string::npos);
}

// A 409 (dimension mismatch) is an expected, recoverable outcome for a caller validating a
// proposed pairing — link_world_model() returns false rather than throwing, unlike most other
// ModelNameClient methods' check_status()-based contract.
TEST(ModelNameClientTest, LinkWorldModelReturnsFalseOn409WithoutThrowing) {
    httplib::Server svr;
    svr.Post("/models/my-chatbot/link-world-model",
             [&](const httplib::Request&, httplib::Response& res) {
                 res.status = 409;
                 res.set_content(R"({"error":"d_model mismatch"})", "application/json");
             });
    auto t = start_server(svr, kPort_LinkWorldModelFailure);

    adai::ModelNameClient client(base_url(kPort_LinkWorldModelFailure));
    bool ok = true;
    EXPECT_NO_THROW(ok = client.link_world_model("my-chatbot", "my-wm"));

    stop_server(svr, t);

    EXPECT_FALSE(ok);
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

// TD-070 regression: list_models() used to find each record's end by
// searching for the literal substring "}}" — a free-form field appearing
// before state/updated_utc in the wire format (run_group has no server-side
// format validation, unlike model_name) that happens to contain that
// two-character sequence used to truncate the record early, extracting
// empty strings for the fields that come after it.
TEST(ModelNameClientTest, ListModelsHandlesBraceLikeSequenceInsideFieldValue) {
    httplib::Server svr;
    svr.Get("/models", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(
            R"({"models":[)"
            R"({"model_id":"1","model_name":"m1","role":"chatbot","run_group":"grp }} weird",)"
            R"("state":"training","updated_utc":"t1","artifact":{"host":"","path":"",)"
            R"("checksum":"","format":""},"tags":{}},)"
            R"({"model_id":"2","model_name":"m2","role":"embedder","run_group":"",)"
            R"("state":"production","updated_utc":"t2","artifact":{"host":"","path":"",)"
            R"("checksum":"","format":""},"tags":{}}]})",
            "application/json");
    });
    auto t = start_server(svr, kPort_ListModels);

    adai::ModelNameClient client(base_url(kPort_ListModels));
    auto models = client.list_models();

    stop_server(svr, t);

    ASSERT_EQ(models.size(), 2u);
    EXPECT_EQ(models[0].model_name, "m1");
    EXPECT_EQ(models[0].state, "training");
    EXPECT_EQ(models[0].updated_utc, "t1");
    EXPECT_EQ(models[1].model_name, "m2");
    EXPECT_EQ(models[1].state, "production");
    EXPECT_EQ(models[1].updated_utc, "t2");
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
