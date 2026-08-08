#include <gtest/gtest.h>

#include <unistd.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include <httplib.h>

#include "MetricsSessionRegistry.hpp"
#include "TrainingMetricsAPI.hpp"

namespace {

int candidate_test_port(int attempt) {
    constexpr int kStartPort = 39080;
    constexpr int kPortSpan = 800;
    return kStartPort + (attempt % kPortSpan);
}

MetricsServiceConfig make_test_config(const std::string& test_name) {
    namespace fs = std::filesystem;

    MetricsServiceConfig cfg;
    cfg.enable_persistence = false;
    cfg.enable_push = false;
    cfg.enable_prometheus_format = false;
    cfg.persist_every_samples = 1;

    const fs::path root = fs::temp_directory_path() / ("adai_metrics_api_tests_" + test_name);
    fs::remove_all(root);

    cfg.metrics_file = (root / "metrics.jsonl").string();
    cfg.summary_file = (root / "metrics_summary.json").string();
    cfg.prometheus_file = (root / "metrics.prom").string();
    cfg.abnormal_samples_file = (root / "abnormal_samples.json").string();

    return cfg;
}

class TrainingMetricsAPIRoutesTest : public ::testing::Test {
   protected:
    virtual size_t max_live_sessions() const {
        return 16;
    }

    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NE(info, nullptr);

        MetricsServiceConfig cfg = make_test_config(info->name());
        registry_ = std::make_shared<MetricsSessionRegistry>(cfg, max_live_sessions());
        ASSERT_NE(registry_, nullptr);

        // Ensure legacy-default session exists for alias endpoint coverage.
        auto default_service = registry_->create_or_get_session("0-default");
        ASSERT_NE(default_service, nullptr);

        ASSERT_TRUE(start_server_with_retry());
    }

    void TearDown() override {
        if (api_) {
            api_->stop();
        }
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        api_.reset();
        registry_.reset();
    }

    bool start_server_with_retry() {
        constexpr int kMaxAttempts = 20;

        for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
            port_ = candidate_test_port(attempt + static_cast<int>(::getpid()));
            api_ = std::make_unique<TrainingMetricsAPI>(registry_, port_, true);

            server_exit_code_.store(-1);
            server_thread_ =
                std::thread([this]() { server_exit_code_.store(api_->start() ? 0 : 1); });

            if (wait_for_server_ready()) {
                return true;
            }

            if (api_) {
                api_->stop();
            }
            if (server_thread_.joinable()) {
                server_thread_.join();
            }
            api_.reset();
        }

        return false;
    }

    bool wait_for_server_ready() {
        httplib::Client client("127.0.0.1", port_);
        client.set_connection_timeout(std::chrono::seconds(1));
        client.set_read_timeout(std::chrono::seconds(1));
        client.set_write_timeout(std::chrono::seconds(1));

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            if (server_exit_code_.load() == 1) {
                return false;
            }

            auto res = client.Get("/health");
            if (res && res->status == 200) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        return false;
    }

    httplib::Client make_client() const {
        httplib::Client client("127.0.0.1", port_);
        client.set_connection_timeout(std::chrono::seconds(2));
        client.set_read_timeout(std::chrono::seconds(2));
        client.set_write_timeout(std::chrono::seconds(2));
        return client;
    }

    std::shared_ptr<MetricsSessionRegistry> registry_;
    std::unique_ptr<TrainingMetricsAPI> api_;
    std::thread server_thread_;
    std::atomic<int> server_exit_code_{-1};
    int port_ = 0;
};

class TrainingMetricsAPICapacityRoutesTest : public TrainingMetricsAPIRoutesTest {
   protected:
    size_t max_live_sessions() const override {
        return 1;
    }
};

TEST_F(TrainingMetricsAPIRoutesTest, SessionsEndpointListsKnownSessions) {
    auto alpha = registry_->create_or_get_session("alpha1");
    ASSERT_NE(alpha, nullptr);
    alpha->start_session(101, 5, 1000);
    alpha->update_sample_metrics(1, 0.75f, 0.33f, 0.001f);

    auto client = make_client();
    auto res = client.Get("/api/sessions");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"key\":\"0-default\""), std::string::npos);
    EXPECT_NE(res->body.find("\"key\":\"alpha1\""), std::string::npos);
    EXPECT_NE(res->body.find("/api/sessions/alpha1/metrics/current"), std::string::npos);
}

TEST_F(TrainingMetricsAPIRoutesTest, AggregateEndpointReportsLiveSessionsOnly) {
    auto alpha = registry_->create_or_get_session("alpha1");
    ASSERT_NE(alpha, nullptr);
    alpha->start_session(201, 4, 500);
    alpha->update_sample_metrics(1, 0.9f, 0.4f, 0.001f);
    alpha->update_validation_metrics(0.8f, 0.7f, 0.0f);

    auto beta = registry_->create_or_get_session("beta2");
    ASSERT_NE(beta, nullptr);
    beta->start_session(202, 4, 500);
    beta->end_session();

    auto client = make_client();
    auto res = client.Get("/api/metrics/aggregate");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"live_sessions\":1"), std::string::npos);
    EXPECT_NE(res->body.find("\"key\":\"alpha1\""), std::string::npos);
    EXPECT_EQ(res->body.find("\"key\":\"beta2\""), std::string::npos);
}

TEST_F(TrainingMetricsAPIRoutesTest,
       LegacyAliasRoutesMapToDefaultSessionAndExposeDeprecationHeaders) {
    auto default_service = registry_->create_or_get_session("0-default");
    ASSERT_NE(default_service, nullptr);
    default_service->start_session(303, 3, 120);
    default_service->update_sample_metrics(5, 1.23f, 0.66f, 0.0005f);

    auto client = make_client();
    auto legacy_res = client.Get("/api/metrics/current");
    auto session_res = client.Get("/api/sessions/0-default/metrics/current");

    ASSERT_TRUE(legacy_res);
    ASSERT_TRUE(session_res);

    EXPECT_EQ(legacy_res->status, 200);
    EXPECT_EQ(session_res->status, 200);
    // Verify both routes serve the same session by session_id rather than full body equality.
    // total_training_time_seconds is recomputed from wall clock on every read, so sequential
    // requests can legitimately differ by ~1ms, making byte-identical comparison a timing race.
    EXPECT_NE(legacy_res->body.find("\"session_id\": 303"), std::string::npos);
    EXPECT_NE(session_res->body.find("\"session_id\": 303"), std::string::npos);

    EXPECT_EQ(legacy_res->get_header_value("Deprecation"), "true");
    EXPECT_EQ(legacy_res->get_header_value("Link"), "/api/sessions/0-default/metrics/current");
}

TEST_F(TrainingMetricsAPIRoutesTest,
       LegacyPostStartAliasMapsToDefaultSessionAndExposesDeprecationHeaders) {
    auto client = make_client();

    const std::string start_body = R"({"session_id":404,"total_epochs":2,"total_samples":50})";

    auto legacy_start = client.Post("/api/session/start", start_body, "application/json");
    auto session_start =
        client.Post("/api/sessions/0-default/start", start_body, "application/json");
    auto legacy_current = client.Get("/api/metrics/current");
    auto session_current = client.Get("/api/sessions/0-default/metrics/current");

    ASSERT_TRUE(legacy_start);
    ASSERT_TRUE(session_start);
    ASSERT_TRUE(legacy_current);
    ASSERT_TRUE(session_current);

    EXPECT_EQ(legacy_start->status, 200);
    EXPECT_EQ(session_start->status, 409);
    EXPECT_EQ(legacy_start->get_header_value("Deprecation"), "true");
    EXPECT_EQ(legacy_start->get_header_value("Link"), "/api/sessions/0-default/start");

    EXPECT_EQ(legacy_current->status, 200);
    EXPECT_EQ(session_current->status, 200);
    // Both routes must serve the same session — verified by session_id, not full body equality.
    // Full body comparison is intentionally avoided: total_training_time_seconds is recomputed
    // from wall clock on every read, so sequential requests can legitimately differ by ~1ms.
    EXPECT_NE(legacy_current->body.find("\"session_id\": 404"), std::string::npos);
    EXPECT_NE(session_current->body.find("\"session_id\": 404"), std::string::npos);
    EXPECT_NE(legacy_current->body.find("\"is_training\":"), std::string::npos);
}

TEST_F(TrainingMetricsAPIRoutesTest, SessionStartReturnsConflictForActiveSessionKey) {
    auto client = make_client();

    const std::string start_body = R"({"session_id":505,"total_epochs":2,"total_samples":100})";

    auto first_start = client.Post("/api/sessions/conflict1/start", start_body, "application/json");
    auto second_start =
        client.Post("/api/sessions/conflict1/start", start_body, "application/json");

    ASSERT_TRUE(first_start);
    ASSERT_TRUE(second_start);

    EXPECT_EQ(first_start->status, 200);
    EXPECT_EQ(second_start->status, 409);
    EXPECT_NE(second_start->body.find("session already active"), std::string::npos);
}

TEST_F(TrainingMetricsAPIRoutesTest, SessionScopedGetReturns404ForUnknownSessionKey) {
    auto client = make_client();
    auto res = client.Get("/api/sessions/unknown9/metrics/current");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);
    EXPECT_NE(res->body.find("Unknown session key"), std::string::npos);
}

TEST_F(TrainingMetricsAPIRoutesTest, SessionScopedPostReturns404ForUnknownSessionKey) {
    auto client = make_client();
    const std::string sample_body =
        R"({"sample":1,"loss":0.42,"gradient_norm":0.12,"learning_rate":0.001})";

    auto res =
        client.Post("/api/sessions/unknown9/metrics/sample", sample_body, "application/json");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);
    EXPECT_NE(res->body.find("Unknown session key"), std::string::npos);
}

TEST_F(TrainingMetricsAPIRoutesTest, LayerGradientNormsRoundTripThroughPostAndGet) {
    auto client = make_client();

    const std::string start_body = R"({"session_id":606,"total_epochs":2,"total_samples":100})";
    auto start_res = client.Post("/api/sessions/layergrad1/start", start_body, "application/json");
    ASSERT_TRUE(start_res);
    ASSERT_EQ(start_res->status, 200);

    const std::string layer_grad_body =
        R"({"encoder_layer_grad_norms":[0.9,0.7,0.5],"decoder_layer_grad_norms":[1.1,0.8]})";
    auto post_res = client.Post("/api/sessions/layergrad1/metrics/layer-gradients",
                                layer_grad_body, "application/json");
    ASSERT_TRUE(post_res);
    EXPECT_EQ(post_res->status, 200);

    auto get_res = client.Get("/api/sessions/layergrad1/metrics/current");
    ASSERT_TRUE(get_res);
    EXPECT_EQ(get_res->status, 200);
    // to_json() serializes floats under the stream's ambient std::fixed <<
    // setprecision(6) (set earlier for weight_update_ratio's neighbors), so
    // arrays come out as "0.900000" rather than the compact "0.9" the client
    // sends — this asserts against what the endpoint actually emits, not what
    // was POSTed.
    EXPECT_NE(get_res->body.find(
                 "\"encoder_layer_grad_norms\": [0.900000,0.700000,0.500000]"),
             std::string::npos)
        << get_res->body;
    EXPECT_NE(get_res->body.find("\"decoder_layer_grad_norms\": [1.100000,0.800000]"),
             std::string::npos)
        << get_res->body;
}

TEST_F(TrainingMetricsAPICapacityRoutesTest, SessionStartReturns503WhenRegistryIsFull) {
    auto client = make_client();

    const std::string start_body = R"({"session_id":606,"total_epochs":2,"total_samples":100})";

    // With max_live_sessions=1 and 0-default pre-created in fixture setup,
    // creating any additional key should be rejected as at-capacity.
    auto res = client.Post("/api/sessions/captest1/start", start_body, "application/json");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 503);
    EXPECT_NE(res->body.find("metrics_server_full"), std::string::npos);
    EXPECT_NE(res->body.find("\"max_live_sessions\":1"), std::string::npos);
}

TEST_F(TrainingMetricsAPIRoutesTest, AggregateEndpointCountsAllLiveSessions) {
    // Start the default session and create two more, all simultaneously live.
    // Verifies live_sessions == 3 and both named keys appear in the response.
    auto default_opt = registry_->get_session("0-default");
    ASSERT_TRUE(default_opt);
    (*default_opt)->start_session(1, 2, 100);

    auto alpha = registry_->create_or_get_session("multi-a");
    ASSERT_NE(alpha, nullptr);
    alpha->start_session(301, 2, 200);
    alpha->update_sample_metrics(1, 0.7f, 0.5f, 0.001f);

    auto beta = registry_->create_or_get_session("multi-b");
    ASSERT_NE(beta, nullptr);
    beta->start_session(302, 4, 400);
    beta->update_sample_metrics(1, 1.1f, 0.8f, 0.002f);

    auto client = make_client();
    auto res = client.Get("/api/metrics/aggregate");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"live_sessions\":3"), std::string::npos);
    EXPECT_NE(res->body.find("\"key\":\"multi-a\""), std::string::npos);
    EXPECT_NE(res->body.find("\"key\":\"multi-b\""), std::string::npos);
}

// ---------------------------------------------------------------------------
// Phase 2: model_id injection into config_snapshot (TD-028 Phase 2)
// ---------------------------------------------------------------------------

TEST_F(TrainingMetricsAPIRoutesTest, SessionStartWithModelIdInjectsIntoConfigSnapshot) {
    auto client = make_client();

    const std::string model_uuid = "550e8400-e29b-41d4-a716-446655440099";
    const std::string start_body =
        "{\"session_id\":707,\"total_epochs\":3,\"total_samples\":150"
        ",\"model_id\":\"" +
        model_uuid + "\"}";

    auto res = client.Post("/api/sessions/mns-inject1/start", start_body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    // Access config_snapshot directly via the registry — it is stored in the
    // TrainingMetricsSnapshot even though it is not emitted by to_json().
    auto svc_opt = registry_->get_session("mns-inject1");
    ASSERT_TRUE(svc_opt) << "Session 'mns-inject1' not found in registry";
    const auto snapshot = (*svc_opt)->get_current_snapshot();
    EXPECT_NE(snapshot.config_snapshot.find(model_uuid), std::string::npos)
        << "model_id not found in config_snapshot: " << snapshot.config_snapshot;
}

TEST_F(TrainingMetricsAPIRoutesTest, SessionStartWithoutModelIdStillWorks) {
    auto client = make_client();

    const std::string start_body = R"({"session_id":808,"total_epochs":2,"total_samples":80})";

    auto res = client.Post("/api/sessions/mns-compat1/start", start_body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << "Backward compat: session start without model_id must succeed";

    auto metrics_res = client.Get("/api/sessions/mns-compat1/metrics/current");
    ASSERT_TRUE(metrics_res);
    EXPECT_EQ(metrics_res->status, 200);
    EXPECT_NE(metrics_res->body.find("\"session_id\": 808"), std::string::npos);
}

}  // namespace
