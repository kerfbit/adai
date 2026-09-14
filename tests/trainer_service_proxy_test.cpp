/**
 * @file trainer_service_proxy_test.cpp
 * @brief Tests for TrainerServiceProxy (TD-172) — trainer_service's own admin HTTP listener,
 *        which proxies /admin/* requests to whichever incremental_trainer child is currently
 *        alive (or answers directly, idle-shaped, when none is).
 *
 * The "live child" side of the proxying tests uses a real TrainerAdminAPI instance (exactly the
 * class a real `incremental_trainer --admin-port` child hosts) rather than a hand-rolled fake
 * server — this exercises real interoperability between the two classes, not just that
 * TrainerServiceProxy can talk to *some* HTTP server. Same style as trainer_admin_api_test.cpp:
 * real httplib::Server instances on background threads, driven with a real httplib::Client.
 */

#include "TrainerServiceProxy.hpp"
#include <httplib.h>
#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <thread>
#include "TrainerAdminAPI.hpp"
#include "TrainerControlState.hpp"

namespace fs = std::filesystem;
using adai::TrainerAdminAPI;
using adai::TrainerControlState;
using adai::TrainerServiceProxy;

namespace {

// Distinct base/span from trainer_admin_api_test.cpp's own kBasePort=43700 (span 100) so the two
// suites' ports never collide when ctest runs them in parallel.
constexpr int kBasePort = 43900;
constexpr int kPortSpan = 200;

int pick_port(int seed) {
    return kBasePort + (seed % kPortSpan);
}

/// Polls GET /health on `port` until it responds or 3s elapse — mirrors
/// trainer_admin_api_test.cpp's start_admin_api() helper, generalized to any already-started
/// server (both TrainerServiceProxy and TrainerAdminAPI expose the same start()-blocks/GET
/// /health shape).
bool wait_until_reachable(int port) {
    httplib::Client probe("127.0.0.1", port);
    probe.set_connection_timeout(std::chrono::milliseconds(200));
    probe.set_read_timeout(std::chrono::milliseconds(200));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto res = probe.Get("/health"); res && res->status == 200) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}

}  // namespace

class TrainerServiceProxyTest : public ::testing::Test {
   protected:
    fs::path test_dir;
    int proxy_port = 0;
    int child_port = 0;

    void SetUp() override {
        test_dir = fs::temp_directory_path() /
                   ("trainer_service_proxy_test_" + std::to_string(::getpid()) + "_" +
                    std::to_string(reinterpret_cast<uintptr_t>(this)));
        fs::create_directories(test_dir);
        const int seed = static_cast<int>(::getpid()) + static_cast<int>(reinterpret_cast<uintptr_t>(this));
        proxy_port = pick_port(seed);
        child_port = pick_port(seed + 1);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(test_dir, ec);
    }

    std::unique_ptr<TrainerServiceProxy> make_proxy() {
        return std::make_unique<TrainerServiceProxy>("127.0.0.1", proxy_port, test_dir.string());
    }
};

// ============================================================================
// Idle behavior — no child ever set
// ============================================================================

TEST_F(TrainerServiceProxyTest, HealthNeverProxiedEvenWhenIdle) {
    auto proxy = make_proxy();
    std::thread thr([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    httplib::Client client("127.0.0.1", proxy_port);
    auto res = client.Get("/health");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"status\":\"ok\""), std::string::npos);

    proxy->stop();
    thr.join();
}

TEST_F(TrainerServiceProxyTest, StatusReportsIdleShapeWithNoChild) {
    auto proxy = make_proxy();
    std::thread thr([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    httplib::Client client("127.0.0.1", proxy_port);
    auto res = client.Get("/admin/status");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"phase\":\"idle\""), std::string::npos);
    EXPECT_NE(res->body.find("\"paused\":false"), std::string::npos);

    proxy->stop();
    thr.join();
}

TEST_F(TrainerServiceProxyTest, LogsAreEmptyWithNoChild) {
    auto proxy = make_proxy();
    std::thread thr([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    httplib::Client client("127.0.0.1", proxy_port);
    auto res = client.Get("/admin/logs");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"entries\":[]"), std::string::npos);

    proxy->stop();
    thr.join();
}

TEST_F(TrainerServiceProxyTest, CheckpointRejectedWithNoChild) {
    auto proxy = make_proxy();
    std::thread thr([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    httplib::Client client("127.0.0.1", proxy_port);
    auto res = client.Post("/admin/checkpoint", "", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 409);

    proxy->stop();
    thr.join();
}

TEST_F(TrainerServiceProxyTest, GetConfigReturnsDefaultsWithNoChild) {
    auto proxy = make_proxy();
    std::thread thr([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    httplib::Client client("127.0.0.1", proxy_port);
    auto res = client.Get("/admin/config");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"auto_save_enabled\":true"), std::string::npos);
    EXPECT_NE(res->body.find("\"auto_save_every_samples\":1000"), std::string::npos);
    EXPECT_NE(res->body.find("\"auto_save_every_minutes\":30"), std::string::npos);
    EXPECT_NE(res->body.find("\"max_sessions_to_keep\":50"), std::string::npos);

    proxy->stop();
    thr.join();
}

// The real payoff of reusing the shared daemon_config.db while idle: a config change made with
// no child running still applies the next time GET /admin/config is read, and (per
// handle_get_config_idle()'s own doc comment) would equally be picked up by the next child's own
// TrainerAdminAPI::start() overlay — not exercised end-to-end here (that would need a real child
// process), but the write/read round-trip through the same file is.
TEST_F(TrainerServiceProxyTest, PutConfigWhileIdlePersistsAndGetReflectsIt) {
    auto proxy = make_proxy();
    std::thread thr([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    httplib::Client client("127.0.0.1", proxy_port);
    auto put_res = client.Put("/admin/config", "{\"auto_save_every_samples\":250}", "application/json");
    ASSERT_TRUE(put_res);
    EXPECT_EQ(put_res->status, 200);
    EXPECT_NE(put_res->body.find("\"auto_save_every_samples\":250"), std::string::npos);

    auto get_res = client.Get("/admin/config");
    ASSERT_TRUE(get_res);
    EXPECT_NE(get_res->body.find("\"auto_save_every_samples\":250"), std::string::npos);

    proxy->stop();
    thr.join();
}

TEST_F(TrainerServiceProxyTest, PutConfigRejectsImmutableKeyWhileIdle) {
    auto proxy = make_proxy();
    std::thread thr([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    httplib::Client client("127.0.0.1", proxy_port);
    auto res = client.Put("/admin/config", "{\"port\":9999}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);

    proxy->stop();
    thr.join();
}

// ============================================================================
// Proxying to a live "child" — a real TrainerAdminAPI instance
// ============================================================================

TEST_F(TrainerServiceProxyTest, StatusProxiesToLiveChildVerbatim) {
    auto control = std::make_shared<TrainerControlState>();
    control->current_epoch = 3;
    control->last_loss = 0.42;
    control->set_model_name("test-model");

    auto child_api = std::make_unique<TrainerAdminAPI>(control, "127.0.0.1", child_port,
                                                        (test_dir / "child").string());
    std::thread child_thread([&child_api] { child_api->start(); });
    ASSERT_TRUE(wait_until_reachable(child_port));

    auto proxy = make_proxy();
    std::thread proxy_thread([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    proxy->set_child_port(child_port);

    httplib::Client client("127.0.0.1", proxy_port);
    auto res = client.Get("/admin/status");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    // Idle-shaped default wouldn't have current_epoch:3/model_name:test-model — asserting on
    // these specifically confirms the real child's live state came through the proxy, not a
    // coincidentally-similar idle fallback.
    EXPECT_NE(res->body.find("\"current_epoch\":3"), std::string::npos);
    EXPECT_NE(res->body.find("\"model_name\":\"test-model\""), std::string::npos);

    proxy->stop();
    proxy_thread.join();
    child_api->stop();
    child_thread.join();
}

TEST_F(TrainerServiceProxyTest, PutConfigProxiesToLiveChild) {
    auto control = std::make_shared<TrainerControlState>();
    auto child_api = std::make_unique<TrainerAdminAPI>(control, "127.0.0.1", child_port,
                                                        (test_dir / "child").string());
    std::thread child_thread([&child_api] { child_api->start(); });
    ASSERT_TRUE(wait_until_reachable(child_port));

    auto proxy = make_proxy();
    std::thread proxy_thread([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    proxy->set_child_port(child_port);

    httplib::Client client("127.0.0.1", proxy_port);
    auto res = client.Put("/admin/config", "{\"max_sessions_to_keep\":7}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"max_sessions_to_keep\":7"), std::string::npos);
    // Confirms this really landed on the CHILD's own control state, not a coincidental match.
    EXPECT_EQ(control->max_sessions_to_keep.load(), 7);

    proxy->stop();
    proxy_thread.join();
    child_api->stop();
    child_thread.join();
}

TEST_F(TrainerServiceProxyTest, RequestToUnreachableChildPortReturns503) {
    auto proxy = make_proxy();
    std::thread thr([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    // A port nothing is listening on — simulates the window between the supervisor launching a
    // child and that child's own admin API actually binding.
    proxy->set_child_port(pick_port(999999));

    httplib::Client client("127.0.0.1", proxy_port);
    auto res = client.Get("/admin/status");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 503);
    EXPECT_NE(res->body.find("starting or exiting"), std::string::npos);

    proxy->stop();
    thr.join();
}

TEST_F(TrainerServiceProxyTest, SetChildPortBackToZeroReturnsToIdleFallback) {
    // A freshly-constructed TrainerControlState also reports phase "idle" by default (it's
    // literally never done anything), so distinguish "proxied to the live child" from "the
    // proxy's own idle fallback" via a field the idle fallback always zeroes instead: phase
    // itself doesn't discriminate here, current_epoch does.
    auto control = std::make_shared<TrainerControlState>();
    control->current_epoch = 9;
    auto child_api = std::make_unique<TrainerAdminAPI>(control, "127.0.0.1", child_port,
                                                        (test_dir / "child").string());
    std::thread child_thread([&child_api] { child_api->start(); });
    ASSERT_TRUE(wait_until_reachable(child_port));

    auto proxy = make_proxy();
    std::thread proxy_thread([&proxy] { proxy->start(); });
    ASSERT_TRUE(wait_until_reachable(proxy_port));

    proxy->set_child_port(child_port);
    httplib::Client client("127.0.0.1", proxy_port);
    {
        auto res = client.Get("/admin/status");
        ASSERT_TRUE(res);
        EXPECT_NE(res->body.find("\"current_epoch\":9"), std::string::npos)
            << "expected the live child's real status to come through while attached";
    }

    proxy->set_child_port(0);
    {
        auto res = client.Get("/admin/status");
        ASSERT_TRUE(res);
        EXPECT_EQ(res->body.find("\"current_epoch\":9"), std::string::npos);
        EXPECT_NE(res->body.find("\"current_epoch\":0"), std::string::npos);
    }

    proxy->stop();
    proxy_thread.join();
    child_api->stop();
    child_thread.join();
}
