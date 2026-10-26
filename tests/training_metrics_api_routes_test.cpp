#include <gtest/gtest.h>

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include <httplib.h>

#include "GenerationQualityMetrics.hpp"
#include "MetricsPushClient.hpp"
#include "nlohmann/json.hpp"
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
    virtual std::string cors_allowed_origins() const {
        return "";
    }
    virtual bool use_sqlite_db() const {
        return false;
    }

    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NE(info, nullptr);

        MetricsServiceConfig cfg = make_test_config(info->name());
        if (use_sqlite_db()) {
            const auto dir = std::filesystem::path(cfg.metrics_file).parent_path();
            std::filesystem::create_directories(dir);
            registry_ = std::make_shared<MetricsSessionRegistry>(
                cfg, max_live_sessions(), 3600, 60, "sqlite", (dir / "metrics.db").string());
        } else {
            registry_ = std::make_shared<MetricsSessionRegistry>(cfg, max_live_sessions());
        }
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
            api_ = std::make_unique<TrainingMetricsAPI>(registry_, port_, true, "", "",
                                                        cors_allowed_origins());

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

// TD-081 regression: every legacy "0-default" alias route (e.g. GET
// /api/metrics/current, POST /api/metrics/sample) used to catch only
// `std::exception`, not the more specific `ApiRequestError` its modern
// /api/sessions/{key}/... counterpart catches — so when
// resolve_session_service() throws ApiRequestError(404, ...) because
// "0-default" doesn't exist yet, the legacy route's generic catch swallowed
// it and returned a hardcoded 500/400 instead of the correct 404. This
// fixture deliberately skips the base fixture's SetUp()-time
// create_or_get_session("0-default") call so that scenario is actually
// exercised.
class TrainingMetricsAPIRoutesTestNoDefaultSession : public TrainingMetricsAPIRoutesTest {
   protected:
    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NE(info, nullptr);

        MetricsServiceConfig cfg = make_test_config(info->name());
        registry_ = std::make_shared<MetricsSessionRegistry>(cfg, max_live_sessions());
        ASSERT_NE(registry_, nullptr);

        // Deliberately do NOT pre-create "0-default" here.

        ASSERT_TRUE(start_server_with_retry());
    }
};

TEST_F(TrainingMetricsAPIRoutesTestNoDefaultSession,
      LegacyGetAliasReturns404WhenDefaultSessionDoesNotExist) {
    auto client = make_client();
    auto res = client.Get("/api/metrics/current");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);
    EXPECT_NE(res->body.find("Unknown session key"), std::string::npos);
    // The deprecation headers should still be present on the error response,
    // matching the success-path behavior of every other legacy alias route.
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(TrainingMetricsAPIRoutesTestNoDefaultSession,
      LegacyPostAliasReturns404WhenDefaultSessionDoesNotExist) {
    auto client = make_client();
    const std::string sample_body =
        R"({"sample":1,"loss":0.42,"gradient_norm":0.12,"learning_rate":0.001})";
    auto res = client.Post("/api/metrics/sample", sample_body, "application/json");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);
    EXPECT_NE(res->body.find("Unknown session key"), std::string::npos);
}

TEST_F(TrainingMetricsAPIRoutesTestNoDefaultSession,
      LegacyControlAliasReturns404WhenDefaultSessionDoesNotExist) {
    auto client = make_client();
    auto res = client.Post("/api/control/flush", "", "application/json");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);
    EXPECT_NE(res->body.find("Unknown session key"), std::string::npos);
}

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

TEST_F(TrainingMetricsAPIRoutesTest, LejepaMetricsRoundTripThroughEpochEndAndGet) {
    auto client = make_client();

    const std::string start_body = R"({"session_id":707,"total_epochs":1,"total_samples":10})";
    auto start_res = client.Post("/api/sessions/lejepa1/start", start_body, "application/json");
    ASSERT_TRUE(start_res);
    ASSERT_EQ(start_res->status, 200);

    const std::string epoch_end_body =
        R"({"epoch":1,"loss":0,"validation_loss":0,"learning_rate":0.001,)"
        R"("predictor_loss":0.42,"sigreg_loss":0.13})";
    auto post_res =
        client.Post("/api/sessions/lejepa1/epoch/end", epoch_end_body, "application/json");
    ASSERT_TRUE(post_res);
    EXPECT_EQ(post_res->status, 200);

    auto lejepa_res = client.Get("/api/sessions/lejepa1/metrics/lejepa");
    ASSERT_TRUE(lejepa_res);
    EXPECT_EQ(lejepa_res->status, 200);
    EXPECT_NE(lejepa_res->body.find("\"current_predictor_loss\":0.420000"), std::string::npos)
        << lejepa_res->body;
    EXPECT_NE(lejepa_res->body.find("\"current_sigreg_loss\":0.130000"), std::string::npos)
        << lejepa_res->body;
    EXPECT_NE(lejepa_res->body.find("\"epoch_predictor_losses\":[0.420000]"), std::string::npos)
        << lejepa_res->body;
    EXPECT_NE(lejepa_res->body.find("\"epoch_sigreg_losses\":[0.130000]"), std::string::npos)
        << lejepa_res->body;

    auto current_res = client.Get("/api/sessions/lejepa1/metrics/current");
    ASSERT_TRUE(current_res);
    EXPECT_EQ(current_res->status, 200);
    EXPECT_NE(current_res->body.find("\"current_predictor_loss\": 0.420000"), std::string::npos)
        << current_res->body;
    EXPECT_NE(current_res->body.find("\"current_sigreg_loss\": 0.130000"), std::string::npos)
        << current_res->body;
}

// TD-503: sigreg_loss is an unbiased estimate and can be slightly negative; the epoch must still
// be recorded (the ingest used to require sigreg_loss >= 0).
TEST_F(TrainingMetricsAPIRoutesTest, NegativeSigregLossIsStillRecorded) {
    auto client = make_client();
    const std::string start_body = R"({"session_id":709,"total_epochs":1,"total_samples":10})";
    auto start_res = client.Post("/api/sessions/lejepa3/start", start_body, "application/json");
    ASSERT_TRUE(start_res);
    ASSERT_EQ(start_res->status, 200);

    const std::string epoch_end_body =
        R"({"epoch":1,"loss":0,"validation_loss":0,"learning_rate":0.001,)"
        R"("predictor_loss":0.42,"sigreg_loss":-0.002})";
    auto post_res =
        client.Post("/api/sessions/lejepa3/epoch/end", epoch_end_body, "application/json");
    ASSERT_TRUE(post_res);
    EXPECT_EQ(post_res->status, 200);

    auto lejepa_res = client.Get("/api/sessions/lejepa3/metrics/lejepa");
    ASSERT_TRUE(lejepa_res);
    EXPECT_NE(lejepa_res->body.find("\"epoch_sigreg_losses\":[-0.002000]"), std::string::npos)
        << lejepa_res->body;
}

TEST_F(TrainingMetricsAPIRoutesTest, LejepaAdvancedMetricsRoundTripThroughEpochEndAndGet) {
    auto client = make_client();

    const std::string start_body = R"({"session_id":708,"total_epochs":1,"total_samples":10})";
    auto start_res = client.Post("/api/sessions/lejepa2/start", start_body, "application/json");
    ASSERT_TRUE(start_res);
    ASSERT_EQ(start_res->status, 200);

    const std::string epoch_end_body =
        R"({"epoch":1,"loss":0,"validation_loss":0,"learning_rate":0.001,)"
        R"("predictor_loss":0.42,"sigreg_loss":0.13,"masking_ratio":0.25,)"
        R"("predictor_target_cosine_sim":0.7,"sigreg_variance_mean":1.02,)"
        R"("sigreg_variance_stddev":0.11})";
    auto post_res =
        client.Post("/api/sessions/lejepa2/epoch/end", epoch_end_body, "application/json");
    ASSERT_TRUE(post_res);
    EXPECT_EQ(post_res->status, 200);

    auto lejepa_res = client.Get("/api/sessions/lejepa2/metrics/lejepa");
    ASSERT_TRUE(lejepa_res);
    EXPECT_EQ(lejepa_res->status, 200);
    EXPECT_NE(lejepa_res->body.find("\"current_masking_ratio\":0.250000"), std::string::npos)
        << lejepa_res->body;
    EXPECT_NE(
        lejepa_res->body.find("\"current_predictor_target_cosine_sim\":0.700000"),
        std::string::npos)
        << lejepa_res->body;
    EXPECT_NE(lejepa_res->body.find("\"current_sigreg_variance_mean\":1.020000"),
             std::string::npos)
        << lejepa_res->body;
    EXPECT_NE(lejepa_res->body.find("\"current_sigreg_variance_stddev\":0.110000"),
             std::string::npos)
        << lejepa_res->body;
    EXPECT_NE(lejepa_res->body.find("\"epoch_masking_ratios\":[0.250000]"), std::string::npos)
        << lejepa_res->body;
    EXPECT_NE(lejepa_res->body.find("\"epoch_predictor_target_cosine_sims\":[0.700000]"),
             std::string::npos)
        << lejepa_res->body;
    EXPECT_NE(lejepa_res->body.find("\"epoch_sigreg_variance_means\":[1.020000]"),
             std::string::npos)
        << lejepa_res->body;
    EXPECT_NE(lejepa_res->body.find("\"epoch_sigreg_variance_stddevs\":[0.110000]"),
             std::string::npos)
        << lejepa_res->body;

    // Deliberate deviation from predictor_loss/sigreg_loss (which DO appear on
    // /metrics/current, per TD-208): these four fields must NOT be duplicated there.
    auto current_res = client.Get("/api/sessions/lejepa2/metrics/current");
    ASSERT_TRUE(current_res);
    EXPECT_EQ(current_res->status, 200);
    EXPECT_EQ(current_res->body.find("current_masking_ratio"), std::string::npos)
        << current_res->body;
    EXPECT_EQ(current_res->body.find("current_predictor_target_cosine_sim"), std::string::npos)
        << current_res->body;
    EXPECT_EQ(current_res->body.find("current_sigreg_variance_mean"), std::string::npos)
        << current_res->body;
    EXPECT_EQ(current_res->body.find("current_sigreg_variance_stddev"), std::string::npos)
        << current_res->body;
}

// Living regression-documentation test for the update_advanced_epoch_metrics() bundling quirk
// (see IncrementalTrainingTool.cpp's own comment at the LeJEPA update_advanced_epoch_metrics()
// call site): posting only compute_time_ratio (as LeJEPA does — gradient_variance/
// weight_update_ratio have no LeJEPA-side equivalent) still causes the OTHER two fields to be
// persisted/exposed as literal 0.0, not the -1 "not applicable" sentinel used elsewhere, because
// TrainingMetricsAPI's gating condition fires once ANY of the three is non-zero.
TEST_F(TrainingMetricsAPIRoutesTest, AdvancedEpochMetricsGatingQuirkSendsZeroForUnsetFields) {
    auto client = make_client();

    const std::string start_body = R"({"session_id":709,"total_epochs":1,"total_samples":10})";
    auto start_res = client.Post("/api/sessions/lejepa3/start", start_body, "application/json");
    ASSERT_TRUE(start_res);
    ASSERT_EQ(start_res->status, 200);

    const std::string epoch_end_body =
        R"({"epoch":1,"loss":0,"validation_loss":0,"learning_rate":0.001,)"
        R"("compute_time_ratio":0.85})";
    auto post_res =
        client.Post("/api/sessions/lejepa3/epoch/end", epoch_end_body, "application/json");
    ASSERT_TRUE(post_res);
    EXPECT_EQ(post_res->status, 200);

    auto current_res = client.Get("/api/sessions/lejepa3/metrics/current");
    ASSERT_TRUE(current_res);
    EXPECT_EQ(current_res->status, 200);
    EXPECT_NE(current_res->body.find("\"compute_time_ratio\": 0.85"), std::string::npos)
        << current_res->body;
    // gradient_variance is a plain 0, not -1 — the quirk being documented.
    EXPECT_NE(current_res->body.find("\"gradient_variance\": 0"), std::string::npos)
        << current_res->body;
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

// ============================================================================
// TD-526: cross-origin access. Browsers always send Origin on cross-origin requests and on POSTs;
// the trainer, the Android app and curl don't.
// ============================================================================

namespace {

const std::string kSampleBody = R"({"sample":7,"loss":99.5,"gradient_norm":0.1,"learning_rate":0.001})";

httplib::Headers origin_header(const std::string& origin) {
    return {{"Origin", origin}};
}

}  // namespace

// The verified attack: a web page's preflight-free text/plain POST used to inject a sample (200)
// and could clear history the same way.
TEST_F(TrainingMetricsAPIRoutesTest, CrossOriginTextPlainWriteIsRefusedAndNotApplied) {
    auto client = make_client();
    auto res = client.Post("/api/sessions/0-default/metrics/sample",
                           origin_header("http://evil.example"), kSampleBody, "text/plain");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 403);
    EXPECT_FALSE(res->has_header("Access-Control-Allow-Origin"));
    auto clear = client.Post("/api/sessions/0-default/control/clear",
                             origin_header("http://evil.example"), "", "text/plain");
    ASSERT_TRUE(clear);
    EXPECT_EQ(clear->status, 403);

    auto current = client.Get("/api/sessions/0-default/metrics/current");
    ASSERT_TRUE(current);
    EXPECT_EQ(current->body.find("99.5"), std::string::npos) << current->body;
}

// Even a listed-origin JSON body can't arrive cross-origin without a preflight, but a simple
// Content-Type must be refused outright, Origin or not.
TEST_F(TrainingMetricsAPIRoutesTest, SimpleContentTypeWriteIsRefusedWithoutOrigin) {
    auto client = make_client();
    auto res = client.Post("/api/sessions/0-default/metrics/sample", kSampleBody, "text/plain");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 415);
    res = client.Post("/api/sessions/0-default/metrics/sample", kSampleBody,
                      "application/x-www-form-urlencoded");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 415);
}

// The trainer's path (no Origin, application/json) is unchanged.
TEST_F(TrainingMetricsAPIRoutesTest, NonBrowserJsonWriteStillWorks) {
    auto client = make_client();
    auto res = client.Post("/api/sessions/0-default/metrics/sample", kSampleBody,
                           "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    auto current = client.Get("/api/sessions/0-default/metrics/current");
    ASSERT_TRUE(current);
    EXPECT_NE(current->body.find("99.5"), std::string::npos) << current->body;
}

// Default (empty allow-list): reads still answer, but without CORS headers, so a web page on
// another origin can't read them; preflights are refused.
TEST_F(TrainingMetricsAPIRoutesTest, DefaultGrantsNoCrossOriginReadsOrPreflights) {
    auto client = make_client();
    auto res = client.Get("/api/sessions/0-default/metrics/current",
                          origin_header("http://evil.example"));
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_FALSE(res->has_header("Access-Control-Allow-Origin"));
    EXPECT_EQ(res->get_header_value("Vary"), "Origin");

    res = client.Options("/api/sessions/0-default/metrics/sample",
                         {{"Origin", "http://evil.example"},
                          {"Access-Control-Request-Method", "POST"}});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 403);
    EXPECT_FALSE(res->has_header("Access-Control-Allow-Origin"));
}

class TrainingMetricsAPICorsAllowListTest : public TrainingMetricsAPIRoutesTest {
   protected:
    std::string cors_allowed_origins() const override {
        return " https://dash.example:8443/ , * ";
    }
};

TEST_F(TrainingMetricsAPICorsAllowListTest, ListedOriginReadsAndWrites) {
    auto client = make_client();
    auto res = client.Get("/api/sessions/0-default/metrics/current",
                          origin_header("https://dash.example:8443"));
    ASSERT_TRUE(res);
    EXPECT_EQ(res->get_header_value("Access-Control-Allow-Origin"), "https://dash.example:8443");

    res = client.Options("/api/sessions/0-default/metrics/sample",
                         {{"Origin", "https://dash.example:8443"},
                          {"Access-Control-Request-Method", "POST"}});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 204);
    EXPECT_EQ(res->get_header_value("Access-Control-Allow-Origin"), "https://dash.example:8443");

    res = client.Post("/api/sessions/0-default/metrics/sample",
                      origin_header("https://dash.example:8443"), kSampleBody, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Access-Control-Allow-Origin"), "https://dash.example:8443");
}

// "*" grants reads to any origin, never writes.
TEST_F(TrainingMetricsAPICorsAllowListTest, WildcardGrantsReadsButNotWrites) {
    auto client = make_client();
    auto res = client.Get("/api/sessions/0-default/metrics/current",
                          origin_header("http://other.example"));
    ASSERT_TRUE(res);
    EXPECT_EQ(res->get_header_value("Access-Control-Allow-Origin"), "*");

    res = client.Post("/api/sessions/0-default/metrics/sample",
                      origin_header("http://other.example"), kSampleBody, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 403);
    res = client.Options("/api/sessions/0-default/metrics/sample",
                         {{"Origin", "http://other.example"},
                          {"Access-Control-Request-Method", "POST"}});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 403);
}

// TD-555: the trainer pushes flagged samples with POST …/metrics/abnormal; the route used to be
// GET-only (404), so nothing was ever stored.
TEST_F(TrainingMetricsAPIRoutesTest, PushedAbnormalSampleIsStoredAndReadable) {
    auto client = make_client();
    const std::string long_text(10000, 'x');
    const std::string body = R"({"epoch":2,"sample_id":17,"loss":9.5,"grad_norm":3.25,)"
                             R"("reason":"loss_outlier","input_text":"hello","target_text":")" +
                             long_text + R"("})";
    auto res = client.Post("/api/sessions/0-default/metrics/abnormal", body, "application/json");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200) << res->body;

    auto get = client.Get("/api/sessions/0-default/metrics/abnormal");
    ASSERT_TRUE(get);
    EXPECT_NE(get->body.find("\"count\":1"), std::string::npos) << get->body.substr(0, 300);
    EXPECT_NE(get->body.find("loss_outlier"), std::string::npos);
    EXPECT_NE(get->body.find("hello"), std::string::npos);
    EXPECT_EQ(get->body.find(long_text), std::string::npos) << "target_text must be truncated";
    EXPECT_LT(get->body.size(), 6000u);

    res = client.Post("/api/sessions/0-default/metrics/abnormal", "[1,2]", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400) << res->body;
    res = client.Post("/api/sessions/nosuch1/metrics/abnormal", body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);
    EXPECT_NE(res->body.find("Unknown session key"), std::string::npos) << res->body;
}

// TD-555 end to end: the real push client flags a sample against the real API; it is stored.
TEST_F(TrainingMetricsAPIRoutesTest, PushClientAbnormalSampleReachesTheServer) {
    MetricsPushClient pc("http://127.0.0.1:" + std::to_string(port_) + "/api/sessions/pushed1", 2000);
    ASSERT_EQ(pc.start_session(1, 1, 10, "", ""), 200);
    AbnormalSample sample;
    sample.epoch = 1;
    sample.sample_id = 4;
    sample.loss = 7.0f;
    sample.reason = "grad_norm_outlier";
    sample.input_text = "in";
    sample.target_text = "out";
    pc.flag_abnormal_sample(sample);
    pc.end_session();  // drains the queue

    auto client = make_client();
    auto get = client.Get("/api/sessions/pushed1/metrics/abnormal");
    ASSERT_TRUE(get);
    EXPECT_NE(get->body.find("\"count\":1"), std::string::npos) << get->body;
    EXPECT_NE(get->body.find("grad_norm_outlier"), std::string::npos) << get->body;
}

// TD-555: every endpoint MetricsPushClient posts to must exist as a POST route on the real API, so a
// client/server mismatch fails here instead of turning into a log flood in training. The client is
// driven through every IMetricsReporter method against a recording server; each recorded request is
// then replayed against the real API.
TEST_F(TrainingMetricsAPIRoutesTest, EveryPushClientEndpointHasAPostRoute) {
    httplib::Server recorder;
    std::mutex mtx;
    std::vector<std::pair<std::string, std::string>> seen;  // (path after the session key, body)
    recorder.Post(R"(/api/sessions/[^/]+(/.*))", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mtx);
        seen.emplace_back(req.matches[1], req.body);
        res.status = 200;
    });
    const int rec_port = recorder.bind_to_any_port("127.0.0.1");
    ASSERT_GT(rec_port, 0);
    std::thread rec_thread([&] { recorder.listen_after_bind(); });
    recorder.wait_until_ready();
    {
        MetricsPushClient pc("http://127.0.0.1:" + std::to_string(rec_port) + "/api/sessions/rec1",
                             2000);
        ASSERT_EQ(pc.start_session(1, 1, 10, "label", "{}"), 200);
        pc.start_epoch(1, 10);
        pc.update_sample_metrics(1, 0.5f, 1.0f, 0.001f);
        pc.update_validation_metrics(0.6f, 0.5f, 1.8f);
        pc.update_best_metrics(0.6f, 1);
        pc.update_advanced_epoch_metrics(0.1f, 0.5f, 0.01f);
        AbnormalSample a;
        a.reason = "loss_outlier";
        pc.flag_abnormal_sample(a);
        pc.update_adaptive_clip_metrics(1.0f, 2);
        pc.update_adaptive_clip_epoch(1.0f, 2);
        pc.update_activation_saturation(0.1f);
        pc.update_attention_entropy(2.0f);
        pc.update_layer_gradient_norms({0.1f, 0.2f}, {0.3f});
        pc.update_padding_efficiency(0.9f);
        pc.update_generation_quality_metrics(0.1f, 0.2f, 0.3f, 0.4f);
        pc.update_lejepa_metrics(0.5f, 0.01f);
        pc.update_lejepa_advanced_metrics(0.3f, 0.9f, 1.0f, 0.1f);
        pc.end_epoch(1, 0.5f, 0.6f, 0.001f, 1.6f, 1.0f, 2.0);
        // The queue is FIFO: once /epoch/end (queued last) has arrived, every earlier event has been
        // sent. end_session() would otherwise drop step events still queued (TD-442).
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (std::any_of(seen.begin(), seen.end(),
                                [](const auto& p) { return p.first == "/epoch/end"; }))
                    break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        pc.end_session();
    }
    recorder.stop();
    rec_thread.join();

    auto client = make_client();
    ASSERT_TRUE(client.Post("/api/sessions/routes1/start", R"({"session_id":1,"total_epochs":1})",
                            "application/json"));
    std::set<std::string> endpoints;
    for (const auto& [path, body] : seen) {
        endpoints.insert(path);
        auto res = client.Post("/api/sessions/routes1" + path, body, "application/json");
        ASSERT_TRUE(res) << path;
        EXPECT_NE(res->status, 404) << "the API has no POST route for " << path << ": " << res->body;
    }
    // The recorder saw the lifecycle and the per-event endpoints, not just a few of them.
    for (const char* e : {"/start", "/epoch/start", "/metrics/sample", "/metrics/abnormal", "/end"})
        EXPECT_EQ(endpoints.count(e), 1u) << e << " was never posted";
}

// TD-443: JSON has no NaN/infinity; operator<< writes them as nan/inf. Every JSON response goes through
// sanitize_non_finite_json(), which turns such values (outside strings) into null.
TEST(TrainingMetricsAPISanitize, NonFiniteValuesBecomeNull) {
    EXPECT_EQ(sanitize_non_finite_json(
                  R"({"a":nan,"b":[inf,-nan,1.5,-inf, 1e-07],"s":"nan, inf","c": Infinity,"d":-1})"),
              R"({"a":null,"b":[null,null,1.5,null, 1e-07],"s":"nan, inf","c": null,"d":-1})");
    const std::string untouched = R"({"nan":"inf","x":[1,2],"esc":"a\"nan"})";
    EXPECT_EQ(sanitize_non_finite_json(untouched), untouched);
}

// TD-443 end to end: a diverged epoch (NaN loss, infinite validation loss), a tiny learning rate and
// adaptive clipping, pushed by the real client. /epochs used to read "epoch_losses":[nan,...] (invalid
// JSON), a 1e-7 learning rate as 0, and an empty epoch_adaptive_clip_thresholds.
TEST_F(TrainingMetricsAPIRoutesTest, NonFiniteAndTinyEpochValuesRoundTripAsValidJson) {
    MetricsPushClient pc("http://127.0.0.1:" + std::to_string(port_) + "/api/sessions/diverge1", 2000);
    ASSERT_EQ(pc.start_session(1, 2, 10, "", ""), 200);
    pc.start_epoch(1, 10);
    pc.update_adaptive_clip_epoch(0.5f, 3);
    pc.end_epoch(1, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                 1e-7f);
    pc.start_epoch(2, 10);
    pc.end_epoch(2, 2.5f, 2.6f, 1e-7f);
    pc.end_session();

    auto client = make_client();
    auto res = client.Get("/api/sessions/diverge1/epochs");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200);
    const auto j = nlohmann::json::parse(res->body, nullptr, /*allow_exceptions=*/false);
    ASSERT_FALSE(j.is_discarded()) << "not valid JSON: " << res->body;
    ASSERT_EQ(j["epoch_losses"].size(), 2u) << res->body;
    EXPECT_TRUE(j["epoch_losses"][0].is_null()) << res->body;
    EXPECT_FLOAT_EQ(j["epoch_losses"][1].get<float>(), 2.5f);
    EXPECT_TRUE(j["epoch_validation_losses"][0].is_null()) << res->body;
    ASSERT_EQ(j["epoch_learning_rates"].size(), 2u);
    EXPECT_NEAR(j["epoch_learning_rates"][0].get<double>(), 1e-7, 1e-12) << res->body;
    ASSERT_TRUE(j.contains("epoch_adaptive_clip_thresholds")) << res->body;
    ASSERT_FALSE(j["epoch_adaptive_clip_thresholds"].empty()) << res->body;
    EXPECT_FLOAT_EQ(j["epoch_adaptive_clip_thresholds"][0].get<float>(), 0.5f);

    auto current = client.Get("/api/sessions/diverge1/metrics/current");
    ASSERT_TRUE(current);
    EXPECT_FALSE(nlohmann::json::parse(current->body, nullptr, false).is_discarded()) << current->body;
}

// TD-440: the gradient variance and generation quality tables are readable over the API, and the
// retention settings are admin-mutable.
class TrainingMetricsAPIDbRoutesTest : public TrainingMetricsAPIRoutesTest {
   protected:
    bool use_sqlite_db() const override {
        return true;
    }
};

TEST_F(TrainingMetricsAPIDbRoutesTest, GradientVarianceAndGenerationQualityAreReadable) {
    auto* db = registry_->get_database();
    ASSERT_NE(db, nullptr);
    SessionRecord rec;
    rec.key = "gv-key";
    rec.session_id = 1;
    rec.created_at = rec.last_update_at = std::chrono::system_clock::now();
    db->upsert_session(rec);
    for (int i = 1; i <= 3; ++i)
        db->insert_gradient_variance_sample("gv-key", i, 1, 2.5e-7f * i);
    GenerationQualityScore score;
    score.bleu1 = 0.5f;
    score.bleu4 = 0.125f;
    score.rougeL = 0.75f;
    db->insert_generation_quality("gv-key", 2, score);

    auto client = make_client();
    auto res = client.Get("/api/sessions/gv-key/metrics/db-gradient-variance?limit=2");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200) << res->body;
    auto j = nlohmann::json::parse(res->body);
    ASSERT_EQ(j["count"], 2);
    EXPECT_EQ(j["records"][0]["step"], 2);
    EXPECT_EQ(j["records"][1]["step"], 3);
    EXPECT_FLOAT_EQ(j["records"][1]["value"].get<float>(), 7.5e-7f) << "small values keep digits";

    res = client.Get("/api/sessions/gv-key/metrics/db-generation-quality");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200) << res->body;
    j = nlohmann::json::parse(res->body);
    ASSERT_EQ(j["count"], 1);
    EXPECT_EQ(j["records"][0]["epoch"], 2);
    EXPECT_FLOAT_EQ(j["records"][0]["bleu4"].get<float>(), 0.125f);
    EXPECT_FLOAT_EQ(j["records"][0]["rougeL"].get<float>(), 0.75f);
}

TEST_F(TrainingMetricsAPIRoutesTest, RetentionSettingsAreAdminMutable) {
    auto client = make_client();
    auto res = client.Put(
        "/admin/config",
        R"({"retention_days": 7, "metrics_file_max_mb": 50, "metrics_file_backups": 1})",
        "application/json");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200) << res->body;
    auto j = nlohmann::json::parse(res->body);
    EXPECT_EQ(j["retention_days"], 7);
    EXPECT_EQ(j["metrics_file_max_mb"], 50);
    EXPECT_EQ(j["metrics_file_backups"], 1);
    const auto cfg = registry_->base_metrics_config();
    EXPECT_EQ(cfg.retention_days, 7);
    EXPECT_EQ(cfg.metrics_file_max_mb, 50);
    EXPECT_EQ(cfg.metrics_file_backups, 1);
}

// TD-379: BLEU-1/BLEU-2 travel from the push client through the API into the DB row. They used to
// be dropped on the way, so the bleu1/bleu2 columns always held -1.
TEST_F(TrainingMetricsAPIDbRoutesTest, PushedBleu1AndBleu2AreStored) {
    MetricsPushClient pc("http://127.0.0.1:" + std::to_string(port_) + "/api/sessions/gq1", 2000);
    ASSERT_EQ(pc.start_session(1, 1, 10, "", ""), 200);
    pc.start_epoch(1, 10);
    pc.update_generation_quality_metrics(0.25f, 0.6f, 0.3f, 0.55f, 0.75f, 0.5f);
    pc.end_session();  // drains the queue

    auto client = make_client();
    auto res = client.Get("/api/sessions/gq1/metrics/db-generation-quality");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200) << res->body;
    auto j = nlohmann::json::parse(res->body);
    ASSERT_EQ(j["count"], 1) << res->body;
    EXPECT_FLOAT_EQ(j["records"][0]["bleu1"].get<float>(), 0.75f);
    EXPECT_FLOAT_EQ(j["records"][0]["bleu2"].get<float>(), 0.5f);
    EXPECT_FLOAT_EQ(j["records"][0]["bleu4"].get<float>(), 0.25f);
    EXPECT_FLOAT_EQ(j["records"][0]["rougeL"].get<float>(), 0.55f);
}

// TD-441: compare is bounded; it used to read every listed session's whole history per request.
TEST_F(TrainingMetricsAPIDbRoutesTest, CompareRefusesTooManySessions) {
    std::string keys;
    for (int i = 0; i < 17; ++i)
        keys += (i ? "," : "") + std::string("k") + std::to_string(i);
    auto client = make_client();
    auto res = client.Get("/api/metrics/compare?keys=" + keys + "&metric=loss");
    ASSERT_TRUE(res);
    EXPECT_NE(res->body.find("At most 16 sessions"), std::string::npos) << res->body;

    res = client.Get("/api/metrics/compare?keys=k1,k2&metric=loss");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->body.find("error"), std::string::npos) << res->body;
}

// TD-441: export reports whether it was cut at the read bound; db-history's limit keeps the newest.
TEST_F(TrainingMetricsAPIDbRoutesTest, ExportReportsTruncationAndDbHistoryLimitKeepsNewest) {
    auto* db = registry_->get_database();
    SessionRecord rec;
    rec.key = "exp";
    rec.session_id = 1;
    rec.created_at = rec.last_update_at = std::chrono::system_clock::now();
    db->upsert_session(rec);
    std::vector<PersistentMetricsRecord> rows(30);
    for (int i = 0; i < 30; ++i) {
        rows[i].timestamp = rec.created_at + std::chrono::milliseconds(i);
        rows[i].sample = i;
    }
    db->insert_metrics_records("exp", rows);

    auto client = make_client();
    auto res = client.Get("/api/sessions/exp/metrics/export?format=json");
    ASSERT_TRUE(res);
    auto j = nlohmann::json::parse(res->body);
    EXPECT_EQ(j["count"], 30);
    EXPECT_EQ(j["truncated"], false);

    res = client.Get("/api/sessions/exp/metrics/db-history?limit=3");
    ASSERT_TRUE(res);
    j = nlohmann::json::parse(res->body);
    ASSERT_EQ(j["count"], 3);
    EXPECT_EQ(j["records"][0]["sample"], 27);
    EXPECT_EQ(j["records"][2]["sample"], 29);
}
