/**
 * @file trainer_admin_api_test.cpp
 * @brief Tests for TrainerAdminAPI — `incremental_trainer serve`'s always-on
 *        HTTP admin daemon. Exercised against a real TrainerAdminAPI bound to
 *        127.0.0.1 on a background thread, driven with a real httplib::Client
 *        — the same style RegistryServer/ModelNameService's own routes tests
 *        use, not a mocked HTTP layer.
 *
 * Only gated on HTTPLIB_INCLUDE_DIR in CMakeLists.txt (see BUILD_TRAINER_ADMIN)
 * since TrainerAdminAPI.cpp itself only compiles when cpp-httplib is found.
 * No IncrementalTrainer involved at all — these tests construct a bare
 * TrainerControlState directly, exactly like the real `serve` command loop
 * hands one to both an IncrementalTrainer and a TrainerAdminAPI.
 */

#include "TrainerAdminAPI.hpp"
#include <httplib.h>
#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <thread>
#include "TrainerControlState.hpp"

namespace fs = std::filesystem;
using adai::TrainerAdminAPI;
using adai::TrainerControlState;
using adai::TrainerPhase;

namespace {

constexpr int kBasePort = 43700;
constexpr int kPortSpan = 100;

int pick_port(int seed) {
    return kBasePort + (seed % kPortSpan);
}

/// Starts `api` on a background thread and polls GET /health until it
/// responds or 3s elapse. Mirrors incremental_trainer_registry_test.cpp's
/// start_mock_server() helper for the same class of "spin up a real server,
/// wait for it to be reachable" need.
bool start_admin_api(TrainerAdminAPI& api, int port, std::thread& thread) {
    thread = std::thread([&api] { api.start(); });

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

class TrainerAdminAPITest : public ::testing::Test {
   protected:
    fs::path test_dir;
    std::shared_ptr<TrainerControlState> control;
    int port = 0;

    void SetUp() override {
        test_dir = fs::temp_directory_path() /
                   ("trainer_admin_api_test_" + std::to_string(::getpid()) + "_" +
                    std::to_string(reinterpret_cast<uintptr_t>(this)));
        fs::create_directories(test_dir);
        control = std::make_shared<TrainerControlState>();
        port = pick_port(static_cast<int>(::getpid()) + static_cast<int>(reinterpret_cast<uintptr_t>(this)));
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(test_dir, ec);
    }

    std::unique_ptr<TrainerAdminAPI> make_api() {
        return std::make_unique<TrainerAdminAPI>(control, "127.0.0.1", port, test_dir.string());
    }
};

TEST_F(TrainerAdminAPITest, HealthReturnsOk) {
    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    auto res = client.Get("/health");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"status\":\"ok\""), std::string::npos);

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, GetConfigReturnsDefaults) {
    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    auto res = client.Get("/admin/config");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"auto_save_enabled\":true"), std::string::npos);
    EXPECT_NE(res->body.find("\"auto_save_every_samples\":1000"), std::string::npos);
    EXPECT_NE(res->body.find("\"auto_save_every_minutes\":30"), std::string::npos);
    EXPECT_NE(res->body.find("\"max_sessions_to_keep\":50"), std::string::npos);

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, PutConfigUpdatesTunablesAndGetReflectsThem) {
    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    auto put_res = client.Put(
        "/admin/config",
        "{\"auto_save_enabled\":false,\"auto_save_every_samples\":500,"
        "\"auto_save_every_minutes\":5,\"max_sessions_to_keep\":10}",
        "application/json");
    ASSERT_TRUE(put_res);
    EXPECT_EQ(put_res->status, 200);
    EXPECT_NE(put_res->body.find("\"auto_save_enabled\":false"), std::string::npos);

    EXPECT_FALSE(control->auto_save_enabled.load());
    EXPECT_EQ(control->auto_save_every_samples.load(), 500);
    EXPECT_EQ(control->auto_save_every_minutes.load(), 5);
    EXPECT_EQ(control->max_sessions_to_keep.load(), 10);

    auto get_res = client.Get("/admin/config");
    ASSERT_TRUE(get_res);
    EXPECT_NE(get_res->body.find("\"max_sessions_to_keep\":10"), std::string::npos);

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, PutConfigRejectsImmutableKeys) {
    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    for (const std::string key : {"port", "host", "dir", "enabled"}) {
        auto res = client.Put("/admin/config", "{\"" + key + "\":123}", "application/json");
        ASSERT_TRUE(res);
        EXPECT_EQ(res->status, 400) << "key=" << key;
        EXPECT_NE(res->body.find("immutable"), std::string::npos) << "key=" << key;
    }

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, PutConfigRejectsInvalidValues) {
    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);

    auto neg_samples = client.Put("/admin/config", "{\"auto_save_every_samples\":-1}",
                                  "application/json");
    ASSERT_TRUE(neg_samples);
    EXPECT_EQ(neg_samples->status, 400);

    auto neg_minutes = client.Put("/admin/config", "{\"auto_save_every_minutes\":-1}",
                                  "application/json");
    ASSERT_TRUE(neg_minutes);
    EXPECT_EQ(neg_minutes->status, 400);

    auto zero_sessions =
        client.Put("/admin/config", "{\"max_sessions_to_keep\":0}", "application/json");
    ASSERT_TRUE(zero_sessions);
    EXPECT_EQ(zero_sessions->status, 400);

    auto empty_body = client.Put("/admin/config", "{}", "application/json");
    ASSERT_TRUE(empty_body);
    EXPECT_EQ(empty_body->status, 400);
    EXPECT_NE(empty_body->body.find("no recognized mutable keys"), std::string::npos);

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, PutConfigPersistsAcrossRestart) {
    {
        auto api = make_api();
        std::thread thr;
        ASSERT_TRUE(start_admin_api(*api, port, thr));

        httplib::Client client("127.0.0.1", port);
        auto res =
            client.Put("/admin/config", "{\"max_sessions_to_keep\":7}", "application/json");
        ASSERT_TRUE(res);
        EXPECT_EQ(res->status, 200);

        api->stop();
        thr.join();
    }

    // Fresh control state + fresh TrainerAdminAPI instance pointed at the
    // same config_store_dir — mirrors what a real process restart looks
    // like (a new `serve` invocation starts with defaults, then start()
    // overlays whatever was persisted last time).
    control = std::make_shared<TrainerControlState>();
    ASSERT_EQ(control->max_sessions_to_keep.load(), 50);  // fresh default, not yet overlaid

    auto api2 = make_api();
    std::thread thr2;
    ASSERT_TRUE(start_admin_api(*api2, port, thr2));

    EXPECT_EQ(control->max_sessions_to_keep.load(), 7);

    httplib::Client client("127.0.0.1", port);
    auto get_res = client.Get("/admin/config");
    ASSERT_TRUE(get_res);
    EXPECT_NE(get_res->body.find("\"max_sessions_to_keep\":7"), std::string::npos);

    api2->stop();
    thr2.join();
}

TEST_F(TrainerAdminAPITest, StatusReflectsControlState) {
    control->phase = TrainerPhase::Training;
    control->current_epoch = 2;
    control->total_epochs = 5;
    control->samples_trained_this_pass = 1234;
    control->set_run_id("run-02");
    control->set_model_name("ambitious-aardvark");

    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    auto res = client.Get("/admin/status");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"phase\":\"training\""), std::string::npos);
    EXPECT_NE(res->body.find("\"current_epoch\":2"), std::string::npos);
    EXPECT_NE(res->body.find("\"total_epochs\":5"), std::string::npos);
    EXPECT_NE(res->body.find("\"samples_trained_this_pass\":1234"), std::string::npos);
    EXPECT_NE(res->body.find("\"run_id\":\"run-02\""), std::string::npos);
    EXPECT_NE(res->body.find("\"model_name\":\"ambitious-aardvark\""), std::string::npos);
    EXPECT_NE(res->body.find("\"paused\":false"), std::string::npos);

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, LogsIncludesTheStartupEntryOnceTheServerIsUp) {
    // start() itself logs "Admin API listening on ..." via control_->log()
    // before it starts accepting connections, so by the time any HTTP
    // request can reach the server, at least that one entry always exists —
    // there is no reachable "truly empty" state to assert on here.
    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    auto res = client.Get("/admin/logs");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"entries\":["), std::string::npos);
    EXPECT_NE(res->body.find("Admin API listening on"), std::string::npos);
    EXPECT_NE(res->body.find("\"level\":\"info\""), std::string::npos);

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, LogsReflectsEntriesWrittenDirectlyToControlState) {
    control->log(adai::TrainerLogLevel::Info, "hello from a test");
    control->log(adai::TrainerLogLevel::Error, "something bad happened");

    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    auto res = client.Get("/admin/logs");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"level\":\"info\""), std::string::npos);
    EXPECT_NE(res->body.find("\"message\":\"hello from a test\""), std::string::npos);
    EXPECT_NE(res->body.find("\"level\":\"error\""), std::string::npos);
    EXPECT_NE(res->body.find("\"message\":\"something bad happened\""), std::string::npos);

    api->stop();
    thr.join();
}

/**
 * The admin actions themselves (pause/resume/checkpoint/config PUT) should
 * each leave a matching entry behind — this is the "feed messages through
 * the logger" contract every handler routes through TrainerControlState::log()
 * for, verified end-to-end via GET /admin/logs rather than by re-asserting
 * each handler's internal call site individually.
 */
TEST_F(TrainerAdminAPITest, AdminActionsAppendMatchingLogEntries) {
    control->phase = TrainerPhase::Training;

    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    ASSERT_TRUE(client.Post("/admin/pause"));
    ASSERT_TRUE(client.Post("/admin/resume"));
    ASSERT_TRUE(client.Put("/admin/config", "{\"max_sessions_to_keep\":7}", "application/json"));

    auto res = client.Get("/admin/logs");
    ASSERT_TRUE(res);
    EXPECT_NE(res->body.find("Pause requested via admin API"), std::string::npos);
    EXPECT_NE(res->body.find("Resume requested via admin API"), std::string::npos);
    EXPECT_NE(res->body.find("Admin config updated via PUT /admin/config"), std::string::npos);

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, CheckpointReturns409WhenIdle) {
    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    auto res = client.Post("/admin/checkpoint");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 409);
    EXPECT_FALSE(control->checkpoint_requested.load());

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, CheckpointSetsFlagAndWaitMsObservesCompletion) {
    control->phase = TrainerPhase::Training;

    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    // Simulate what run_training()'s sample callback does: consume the
    // one-shot flag and bump the counters once it "writes" a checkpoint.
    std::atomic<bool> stop_worker{false};
    std::thread worker([&] {
        while (!stop_worker.load()) {
            if (control->checkpoint_requested.exchange(false)) {
                control->set_last_checkpoint_path("/tmp/auto_save_session_1.bin");
                control->checkpoints_written.fetch_add(1);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(std::chrono::seconds(2));
    auto res = client.Post("/admin/checkpoint?wait_ms=1000");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 202);
    EXPECT_NE(res->body.find("\"completed\":true"), std::string::npos);
    EXPECT_NE(res->body.find("auto_save_session_1.bin"), std::string::npos);

    stop_worker = true;
    worker.join();
    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, PauseSetsPausedAndIsIdempotent) {
    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    auto res1 = client.Post("/admin/pause");
    ASSERT_TRUE(res1);
    EXPECT_EQ(res1->status, 202);
    EXPECT_TRUE(control->paused.load());

    auto res2 = client.Post("/admin/pause");
    ASSERT_TRUE(res2);
    EXPECT_EQ(res2->status, 202);
    EXPECT_TRUE(control->paused.load());

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, ResumeClearsPausedAndWakesSleepingLoop) {
    control->paused = true;

    auto api = make_api();
    std::thread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    std::atomic<bool> awoke{false};
    std::thread sleeper([&] {
        control->interruptible_sleep(30);
        awoke = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    httplib::Client client("127.0.0.1", port);
    auto res = client.Post("/admin/resume");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 202);
    EXPECT_FALSE(control->paused.load());

    sleeper.join();
    EXPECT_TRUE(awoke.load());

    api->stop();
    thr.join();
}
