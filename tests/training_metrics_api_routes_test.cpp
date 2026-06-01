#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

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
            server_thread_ = std::thread([this]() {
                server_exit_code_.store(api_->start() ? 0 : 1);
            });

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

TEST_F(TrainingMetricsAPIRoutesTest, LegacyAliasRoutesMapToDefaultSessionAndExposeDeprecationHeaders) {
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
    EXPECT_EQ(legacy_res->body, session_res->body);

    EXPECT_EQ(legacy_res->get_header_value("Deprecation"), "true");
    EXPECT_EQ(legacy_res->get_header_value("Link"), "/api/sessions/0-default/metrics/current");
}

TEST_F(TrainingMetricsAPIRoutesTest, LegacyPostStartAliasMapsToDefaultSessionAndExposesDeprecationHeaders) {
    auto client = make_client();

    const std::string start_body =
        R"({"session_id":404,"total_epochs":2,"total_samples":50})";

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
    EXPECT_EQ(legacy_current->body, session_current->body);
    EXPECT_NE(legacy_current->body.find("\"is_training\":"), std::string::npos);
}

TEST_F(TrainingMetricsAPIRoutesTest, SessionStartReturnsConflictForActiveSessionKey) {
    auto client = make_client();

    const std::string start_body =
        R"({"session_id":505,"total_epochs":2,"total_samples":100})";

    auto first_start =
        client.Post("/api/sessions/conflict1/start", start_body, "application/json");
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

TEST_F(TrainingMetricsAPICapacityRoutesTest, SessionStartReturns503WhenRegistryIsFull) {
    auto client = make_client();

    const std::string start_body =
        R"({"session_id":606,"total_epochs":2,"total_samples":100})";

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

}  // namespace
