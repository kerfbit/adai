// Live-server integration tests for the TrainingMetricsAPI.
//
// Connects to a running metrics_api_server and exercises every GET and POST
// route.  Read-only GET tests share a single class-level session created once
// by SetUpTestSuite so the registry is not exhausted by per-test creates.
// Mutating tests (lifecycle, error cases) each create their own short-lived
// session and clean up in TearDown.
//
// Configuration (env vars, both optional):
//   METRICS_SERVER_HOST   default: 192.168.1.16
//   METRICS_SERVER_PORT   default: 8081

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
// Helpers
// ---------------------------------------------------------------------------

std::string server_host() {
    const char* env = std::getenv("METRICS_SERVER_HOST");
    return env ? std::string(env) : "192.168.1.16";
}

int server_port() {
    const char* env = std::getenv("METRICS_SERVER_PORT");
    return env ? std::atoi(env) : 8081;
}

httplib::Client make_client() {
    httplib::Client c(server_host(), server_port());
    c.set_connection_timeout(std::chrono::seconds(5));
    c.set_read_timeout(std::chrono::seconds(5));
    c.set_write_timeout(std::chrono::seconds(5));
    return c;
}

// Each call returns a key that is unique across this process run.
std::string make_test_key(const std::string& tag = "") {
    static std::atomic<int> counter{0};
    std::ostringstream oss;
    oss << "lt" << static_cast<int>(::getpid())
        << "t" << static_cast<long>(std::time(nullptr))
        << "c" << counter.fetch_add(1);
    if (!tag.empty()) {
        oss << "-" << tag;
    }
    return oss.str();
}

bool server_reachable() {
    auto c = make_client();
    auto res = c.Get("/health");
    return res && res->status == 200;
}

// ---------------------------------------------------------------------------
// Shared session fixture used by all read-only GET tests.
//
// Each test creates its own httplib::Client so every request within one test
// goes to the same backend — essential when the server address is behind a
// load balancer where reconnections can round-robin to a different instance.
// All tests reuse the same static key so the registry replaces the completed
// session rather than accumulating one entry per test.
// ---------------------------------------------------------------------------

class LiveServerSharedSession : public ::testing::Test {
   protected:
    // End any active session with shared_key_ then seed a fresh one on `c`.
    static bool seed_shared_session(httplib::Client& c) {
        const std::string kp = "/api/sessions/" + shared_key_;
        // Clear any active session so create_or_get_session can replace it.
        c.Post((kp + "/end").c_str(), "", "application/json");

        std::string body =
            R"({"session_id":7001,"total_epochs":3,"total_samples":300,"label":"live-test-shared"})";
        auto res = c.Post((kp + "/start").c_str(), body, "application/json");
        if (!res || res->status != 200) return false;

        c.Post((kp + "/epoch/start").c_str(), R"({"epoch":1,"total_samples":100})",
               "application/json");
        for (int i = 1; i <= 3; ++i) {
            std::ostringstream sb;
            sb << R"({"sample":)" << i * 30 << R"(,"loss":)" << (1.0f - i * 0.1f)
               << R"(,"gradient_norm":0.4,"learning_rate":0.001})";
            c.Post((kp + "/metrics/sample").c_str(), sb.str(), "application/json");
        }
        c.Post((kp + "/metrics/validation").c_str(),
               R"({"validation_loss":0.72,"validation_accuracy":0.68,"validation_perplexity":0.0})",
               "application/json");
        c.Post((kp + "/metrics/best").c_str(), R"({"validation_loss":0.72,"epoch":1})",
               "application/json");
        c.Post((kp + "/metrics/advanced").c_str(),
               R"({"gradient_variance":0.02,"compute_time_ratio":0.85,"weight_update_ratio":0.95})",
               "application/json");
        c.Post((kp + "/metrics/generation-quality").c_str(),
               R"({"bleu4":0.4,"rouge1":0.5,"rouge2":0.3,"rougeL":0.45})", "application/json");
        c.Post((kp + "/epoch/end").c_str(),
               R"({"epoch":1,"loss":0.7,"validation_loss":0.72,"learning_rate":0.001,"perplexity":2.0,"gradient_norm":0.4})",
               "application/json");
        return true;
    }

    static void SetUpTestSuite() { shared_key_ = make_test_key("shared"); }
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (!server_reachable()) {
            GTEST_SKIP() << "metrics_api_server not reachable at " << server_host() << ":"
                         << server_port();
        }
        client_ = std::make_unique<httplib::Client>(server_host(), server_port());
        client_->set_connection_timeout(std::chrono::seconds(5));
        client_->set_read_timeout(std::chrono::seconds(5));
        client_->set_write_timeout(std::chrono::seconds(5));
        if (!seed_shared_session(*client_)) {
            GTEST_SKIP() << "shared test session could not be created (server may be at capacity)";
        }
    }

    void TearDown() override {
        if (client_) {
            client_->Post(("/api/sessions/" + shared_key_ + "/end").c_str(), "",
                          "application/json");
            client_.reset();
        }
    }

    httplib::Client& client() { return *client_; }
    static const std::string& shared_key() { return shared_key_; }

    std::unique_ptr<httplib::Client> client_;
    static std::string shared_key_;
};

std::string LiveServerSharedSession::shared_key_;

// ---------------------------------------------------------------------------
// Base fixture for tests that create their own sessions
// ---------------------------------------------------------------------------

class LiveServerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        if (!server_reachable()) {
            GTEST_SKIP() << "metrics_api_server not reachable at "
                         << server_host() << ":" << server_port();
        }
        client_ = std::make_unique<httplib::Client>(server_host(), server_port());
        client_->set_connection_timeout(std::chrono::seconds(5));
        client_->set_read_timeout(std::chrono::seconds(5));
        client_->set_write_timeout(std::chrono::seconds(5));
        session_started_ = false;
        test_key_ = make_test_key();
    }

    void TearDown() override {
        if (session_started_) {
            client_->Post(("/api/sessions/" + test_key_ + "/end").c_str(),
                          "", "application/json");
        }
    }

    // POST /start. Returns false (and records failure or skip) when the session
    // could not be created, so callers can ASSERT_TRUE(start_test_session(...))
    // to abort the test body cleanly.
    bool start_test_session(int session_id = 1, int epochs = 3, int samples = 300) {
        std::ostringstream body;
        body << R"({"session_id":)" << session_id
             << R"(,"total_epochs":)" << epochs
             << R"(,"total_samples":)" << samples
             << R"(,"label":"live-test"})";
        auto res = client_->Post(
            ("/api/sessions/" + test_key_ + "/start").c_str(),
            body.str(), "application/json");
        if (!res) {
            ADD_FAILURE() << "POST /start: no response (connection error)";
            return false;
        }
        if (res->status != 200) {
            ADD_FAILURE() << "POST /start: expected 200, got " << res->status
                          << " body: " << res->body;
            return false;
        }
        session_started_ = true;
        return true;
    }

    httplib::Client& client() { return *client_; }
    const std::string& test_key() const { return test_key_; }

   protected:
    bool session_started_ = false;

   private:
    std::unique_ptr<httplib::Client> client_;
    std::string test_key_;
};

// ===========================================================================
// Health
// ===========================================================================

TEST(LiveServerHealth, Returns200WithRequiredFields) {
    if (!server_reachable()) GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    auto res = c.Get("/health");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"status\":"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"service\":"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"is_training\":"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"any_stale\":"), std::string::npos) << res->body;
}

// ===========================================================================
// Session list
// ===========================================================================

TEST(LiveServerSessionList, Returns200WithSessionsArray) {
    if (!server_reachable()) GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    auto res = c.Get("/api/sessions");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("\"sessions\":"), std::string::npos) << res->body;
}

TEST(LiveServerSessionList, ContainsDefaultSession) {
    if (!server_reachable()) GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    auto res = c.Get("/api/sessions");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    // 0-default is created lazily on first legacy trainer connect, not at startup.
    if (res->body.find("\"key\":\"0-default\"") == std::string::npos)
        GTEST_SKIP() << "0-default not present; no trainer has started yet";
    EXPECT_NE(res->body.find("\"key\":\"0-default\""), std::string::npos) << res->body;
}

// ===========================================================================
// Aggregate endpoints
// ===========================================================================

TEST(LiveServerAggregate, JsonReturns200WithLiveSessionsField) {
    if (!server_reachable()) GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    auto res = c.Get("/api/metrics/aggregate");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("\"live_sessions\":"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"sessions\":"), std::string::npos) << res->body;
}

TEST(LiveServerAggregate, PrometheusReturns200WithTextPlain) {
    if (!server_reachable()) GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    auto res = c.Get("/api/metrics/prometheus/aggregate");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->get_header_value("Content-Type").find("text/plain"), std::string::npos);
}

// ===========================================================================
// Error handling
// ===========================================================================

TEST(LiveServerErrors, Get_UnknownSessionKeyReturns404) {
    if (!server_reachable()) GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    auto res = c.Get("/api/sessions/no-such-session-xyz/metrics/current");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404) << res->body;
    EXPECT_NE(res->body.find("Unknown session key"), std::string::npos) << res->body;
}

TEST(LiveServerErrors, Get_InvalidSessionKeyFormatReturns400or404) {
    if (!server_reachable()) GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    // Key starting with '-' fails the regex pattern so route won't match.
    auto res = c.Get("/api/sessions/-bad/metrics/current");
    ASSERT_TRUE(res);
    EXPECT_TRUE(res->status == 400 || res->status == 404)
        << "expected 400 or 404 for invalid key, got " << res->status << " body: " << res->body;
}

TEST(LiveServerErrors, Post_ToUnknownSessionReturns404) {
    if (!server_reachable()) GTEST_SKIP() << "server not reachable";
    auto c = make_client();
    const std::string body = R"({"sample":1,"loss":0.5,"gradient_norm":0.1,"learning_rate":0.001})";
    auto res = c.Post("/api/sessions/no-such-session-xyz/metrics/sample",
                      body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404) << res->body;
    EXPECT_NE(res->body.find("Unknown session key"), std::string::npos) << res->body;
}

// ===========================================================================
// Per-session GET routes — use the class-level shared session
// ===========================================================================

TEST_F(LiveServerSharedSession, CurrentMetricsReturns200WithSessionId) {
    auto res = client().Get(("/api/sessions/" + shared_key() + "/metrics/current").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("\"session_id\": 7001"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"is_training\":"), std::string::npos) << res->body;
}

TEST_F(LiveServerSharedSession, SummaryReturns200WithSessionIdField) {
    auto res = client().Get(("/api/sessions/" + shared_key() + "/metrics/summary").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("session_id"), std::string::npos) << res->body;
}

TEST_F(LiveServerSharedSession, HistoryReturns200WithRecordsField) {
    auto res = client().Get(("/api/sessions/" + shared_key() + "/metrics/history").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("records"), std::string::npos) << res->body;
}

TEST_F(LiveServerSharedSession, HistoryWithMaxRecordsParamReturns200) {
    auto res = client().Get(
        ("/api/sessions/" + shared_key() + "/metrics/history?max_records=2").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
}

TEST_F(LiveServerSharedSession, PrometheusReturns200WithTextPlain) {
    auto res = client().Get(("/api/sessions/" + shared_key() + "/metrics/prometheus").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->get_header_value("Content-Type").find("text/plain"), std::string::npos);
}

TEST_F(LiveServerSharedSession, CsvReturns200WithTextCsv) {
    auto res = client().Get(("/api/sessions/" + shared_key() + "/metrics/csv").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->get_header_value("Content-Type").find("text/csv"), std::string::npos);
}

TEST_F(LiveServerSharedSession, StatusReturns200WithIsTrainingField) {
    auto res = client().Get(("/api/sessions/" + shared_key() + "/status").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("is_training"), std::string::npos) << res->body;
}

TEST_F(LiveServerSharedSession, EpochsReturns200) {
    auto res = client().Get(("/api/sessions/" + shared_key() + "/epochs").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
}

TEST_F(LiveServerSharedSession, AbnormalReturns200WithAbnormalSamplesField) {
    auto res = client().Get(("/api/sessions/" + shared_key() + "/metrics/abnormal").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_NE(res->body.find("abnormal_samples"), std::string::npos) << res->body;
}

TEST_F(LiveServerSharedSession, GenerationQualityReturns200) {
    auto res = client().Get(
        ("/api/sessions/" + shared_key() + "/metrics/generation-quality").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
}

TEST_F(LiveServerSharedSession, PaddingEfficiencyReturns200) {
    auto res = client().Get(
        ("/api/sessions/" + shared_key() + "/metrics/padding-efficiency").c_str());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
}

// ===========================================================================
// Aggregate reflects a live session
// ===========================================================================

TEST_F(LiveServerSharedSession, AggregateShowsLiveSession) {
    auto res = client().Get("/api/metrics/aggregate");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    // Session seeded in SetUp is still active here; TearDown calls /end after.
    EXPECT_NE(res->body.find("\"key\":\"" + shared_key() + "\""), std::string::npos)
        << "live session should appear in aggregate; body: " << res->body;
}

// ===========================================================================
// POST lifecycle — one independent session, full round-trip
// ===========================================================================

TEST_F(LiveServerTest, PostLifecycle_FullRoundTrip) {
    ASSERT_TRUE(start_test_session(9001, 2, 200));
    const std::string kp = "/api/sessions/" + test_key();

    // epoch/start
    {
        auto r = client().Post((kp + "/epoch/start").c_str(),
                               R"({"epoch":1,"total_samples":100})", "application/json");
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200) << r->body;
    }
    // metrics/sample
    {
        auto r = client().Post((kp + "/metrics/sample").c_str(),
                               R"({"sample":50,"loss":0.85,"gradient_norm":0.45,"learning_rate":0.001})",
                               "application/json");
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200) << r->body;
    }
    // metrics/validation
    {
        auto r = client().Post((kp + "/metrics/validation").c_str(),
                               R"({"validation_loss":0.78,"validation_accuracy":0.65,"validation_perplexity":0.0})",
                               "application/json");
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200) << r->body;
    }
    // metrics/best
    {
        auto r = client().Post((kp + "/metrics/best").c_str(),
                               R"({"validation_loss":0.78,"epoch":1})", "application/json");
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200) << r->body;
    }
    // metrics/advanced
    {
        auto r = client().Post((kp + "/metrics/advanced").c_str(),
                               R"({"gradient_variance":0.02,"compute_time_ratio":0.85,"weight_update_ratio":0.95})",
                               "application/json");
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200) << r->body;
    }
    // metrics/generation-quality
    {
        auto r = client().Post((kp + "/metrics/generation-quality").c_str(),
                               R"({"bleu4":0.42,"rouge1":0.55,"rouge2":0.31,"rougeL":0.48})",
                               "application/json");
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200) << r->body;
    }
    // epoch/end
    {
        auto r = client().Post((kp + "/epoch/end").c_str(),
                               R"({"epoch":1,"loss":0.72,"validation_loss":0.78,"learning_rate":0.001,"perplexity":2.1,"gradient_norm":0.38})",
                               "application/json");
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200) << r->body;
    }
    // end
    {
        auto r = client().Post((kp + "/end").c_str(), "", "application/json");
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200) << r->body;
        session_started_ = false;
    }
    // verify ended session is absent from aggregate
    {
        auto r = client().Get("/api/metrics/aggregate");
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 200) << r->body;
        EXPECT_EQ(r->body.find("\"key\":\"" + test_key() + "\""), std::string::npos)
            << "ended session must be absent from aggregate; body: " << r->body;
    }
}

// ===========================================================================
// Duplicate start → 409
// ===========================================================================

TEST_F(LiveServerTest, Post_DuplicateStartReturns409) {
    ASSERT_TRUE(start_test_session(9002, 2, 100));

    const std::string body = R"({"session_id":9002,"total_epochs":2,"total_samples":100})";
    auto res = client().Post(("/api/sessions/" + test_key() + "/start").c_str(),
                             body, "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 409) << res->body;
    EXPECT_NE(res->body.find("session already active"), std::string::npos) << res->body;
}

// ===========================================================================
// Empty body to start → 200 (all fields default to zero)
// ===========================================================================

TEST_F(LiveServerTest, Post_EmptyBodyToStartReturns200) {
    auto res = client().Post(("/api/sessions/" + test_key() + "/start").c_str(),
                             "{}", "application/json");
    ASSERT_TRUE(res);
    if (res->status == 503) GTEST_SKIP() << "server at capacity";
    EXPECT_EQ(res->status, 200) << res->body;
    session_started_ = true;
}

// ===========================================================================
// Control endpoints (may be disabled — accept 200 or 404)
// ===========================================================================

TEST_F(LiveServerTest, Control_FlushResponds200Or404) {
    ASSERT_TRUE(start_test_session(9010, 2, 100));
    auto res = client().Post(
        ("/api/sessions/" + test_key() + "/control/flush").c_str(), "", "application/json");
    ASSERT_TRUE(res);
    EXPECT_TRUE(res->status == 200 || res->status == 404)
        << "flush: unexpected " << res->status << " body: " << res->body;
}

TEST_F(LiveServerTest, Control_ClearResponds200Or404) {
    ASSERT_TRUE(start_test_session(9011, 2, 100));
    auto res = client().Post(
        ("/api/sessions/" + test_key() + "/control/clear").c_str(), "", "application/json");
    ASSERT_TRUE(res);
    EXPECT_TRUE(res->status == 200 || res->status == 404)
        << "clear: unexpected " << res->status << " body: " << res->body;
}

// ===========================================================================
// Legacy alias GET routes — target 0-default.
//
// A per-test fixture creates 0-default before each GET so all requests within
// one test hit the same backend (important when the server address is
// load-balanced across multiple instances).
// ===========================================================================

class LiveServerLegacyGet : public ::testing::Test {
   protected:
    void SetUp() override {
        if (!server_reachable()) {
            GTEST_SKIP() << "metrics_api_server not reachable at " << server_host() << ":"
                         << server_port();
        }
        client_ = std::make_unique<httplib::Client>(server_host(), server_port());
        client_->set_connection_timeout(std::chrono::seconds(5));
        client_->set_read_timeout(std::chrono::seconds(5));
        client_->set_write_timeout(std::chrono::seconds(5));
        // Ensure 0-default exists on this backend.  Returns 200 on creation or
        // 409 if a trainer already started it — either way the session is there.
        auto res = client_->Post("/api/session/start", "{}", "application/json");
        default_was_started_ = (res && res->status == 200);
    }

    void TearDown() override {
        if (client_ && default_was_started_) {
            client_->Post("/api/session/end", "", "application/json");
        }
        client_.reset();
    }

    httplib::Client& client() { return *client_; }

   private:
    std::unique_ptr<httplib::Client> client_;
    bool default_was_started_ = false;
};

TEST_F(LiveServerLegacyGet, CurrentReturns200WithDeprecationHeader) {
    auto res = client().Get("/api/metrics/current");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
    EXPECT_EQ(res->get_header_value("Link"), "/api/sessions/0-default/metrics/current");
}

TEST_F(LiveServerLegacyGet, HistoryReturns200WithDeprecationHeader) {
    auto res = client().Get("/api/metrics/history");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(LiveServerLegacyGet, SummaryReturns200WithDeprecationHeader) {
    auto res = client().Get("/api/metrics/summary");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(LiveServerLegacyGet, PrometheusReturns200) {
    auto res = client().Get("/api/metrics/prometheus");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
}

TEST_F(LiveServerLegacyGet, CsvReturns200) {
    auto res = client().Get("/api/metrics/csv");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
}

TEST_F(LiveServerLegacyGet, AbnormalReturns200) {
    auto res = client().Get("/api/metrics/abnormal");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
}

TEST_F(LiveServerLegacyGet, GenerationQualityReturns200) {
    auto res = client().Get("/api/metrics/generation-quality");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
}

TEST_F(LiveServerLegacyGet, PaddingEfficiencyReturns200) {
    auto res = client().Get("/api/metrics/padding-efficiency");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
}

TEST_F(LiveServerLegacyGet, SessionStatusReturns200WithDeprecationHeader) {
    auto res = client().Get("/api/session/status");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(LiveServerLegacyGet, SessionEpochsReturns200WithDeprecationHeader) {
    auto res = client().Get("/api/session/epochs");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

// ===========================================================================
// Legacy alias POST routes — target 0-default.
// If 0-default is already active (trainer running) the /start will 409;
// sub-tests are skipped gracefully in that case.
// ===========================================================================

class LiveServerLegacyPost : public ::testing::Test {
   protected:
    void SetUp() override {
        if (!server_reachable()) {
            GTEST_SKIP() << "server not reachable";
        }
        client_ = std::make_unique<httplib::Client>(server_host(), server_port());
        client_->set_connection_timeout(std::chrono::seconds(5));
        client_->set_read_timeout(std::chrono::seconds(5));
        client_->set_write_timeout(std::chrono::seconds(5));

        const std::string body = R"({"session_id":1,"total_epochs":2,"total_samples":100})";
        auto res = client_->Post("/api/session/start", body, "application/json");
        default_was_started_ = (res && res->status == 200);
    }

    void TearDown() override {
        if (default_was_started_) {
            client_->Post("/api/session/end", "", "application/json");
        }
    }

    httplib::Client& client() { return *client_; }
    bool default_was_started() const { return default_was_started_; }

   protected:
    bool default_was_started_ = false;

   private:
    std::unique_ptr<httplib::Client> client_;
};

TEST_F(LiveServerLegacyPost, StartRespondsWithDeprecationHeader) {
    // Already called in SetUp; confirm header on a cold retry.
    const std::string body = R"({"session_id":1,"total_epochs":2,"total_samples":100})";
    auto res = client().Post("/api/session/start", body, "application/json");
    ASSERT_TRUE(res);
    // 200 or 409 — header must be present either way.
    EXPECT_TRUE(res->status == 200 || res->status == 409)
        << "unexpected status " << res->status << " body: " << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
    EXPECT_EQ(res->get_header_value("Link"), "/api/sessions/0-default/start");
}

TEST_F(LiveServerLegacyPost, EpochStartReturns200WithDeprecationHeader) {
    if (!default_was_started()) GTEST_SKIP() << "0-default already active; skipping";
    auto res = client().Post("/api/epoch/start", R"({"epoch":1,"total_samples":50})",
                             "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(LiveServerLegacyPost, SampleMetricsReturns200WithDeprecationHeader) {
    if (!default_was_started()) GTEST_SKIP() << "0-default already active; skipping";
    auto res = client().Post("/api/metrics/sample",
                             R"({"sample":1,"loss":0.88,"gradient_norm":0.4,"learning_rate":0.001})",
                             "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(LiveServerLegacyPost, ValidationMetricsReturns200WithDeprecationHeader) {
    if (!default_was_started()) GTEST_SKIP() << "0-default already active; skipping";
    auto res = client().Post("/api/metrics/validation",
                             R"({"validation_loss":0.75,"validation_accuracy":0.71,"validation_perplexity":0.0})",
                             "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(LiveServerLegacyPost, BestMetricsReturns200WithDeprecationHeader) {
    if (!default_was_started()) GTEST_SKIP() << "0-default already active; skipping";
    auto res = client().Post("/api/metrics/best",
                             R"({"validation_loss":0.75,"epoch":1})", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(LiveServerLegacyPost, AdvancedMetricsReturns200WithDeprecationHeader) {
    if (!default_was_started()) GTEST_SKIP() << "0-default already active; skipping";
    auto res = client().Post("/api/metrics/advanced",
                             R"({"gradient_variance":0.01,"compute_time_ratio":0.9,"weight_update_ratio":0.95})",
                             "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(LiveServerLegacyPost, GenerationQualityReturns200WithDeprecationHeader) {
    if (!default_was_started()) GTEST_SKIP() << "0-default already active; skipping";
    auto res = client().Post("/api/metrics/generation-quality",
                             R"({"bleu4":0.38,"rouge1":0.51,"rouge2":0.28,"rougeL":0.44})",
                             "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(LiveServerLegacyPost, EpochEndReturns200WithDeprecationHeader) {
    if (!default_was_started()) GTEST_SKIP() << "0-default already active; skipping";
    auto res = client().Post("/api/epoch/end",
                             R"({"epoch":1,"loss":0.88,"validation_loss":0.75,"learning_rate":0.001,"perplexity":2.4,"gradient_norm":0.4})",
                             "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
}

TEST_F(LiveServerLegacyPost, SessionEndReturns200WithDeprecationHeader) {
    if (!default_was_started()) GTEST_SKIP() << "0-default already active; skipping";
    auto res = client().Post("/api/session/end", "", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200) << res->body;
    EXPECT_EQ(res->get_header_value("Deprecation"), "true");
    default_was_started_ = false;
}

}  // namespace
