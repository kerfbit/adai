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
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <thread>
#include "DaemonConfigStore.hpp"
#include "TrainerControlState.hpp"

namespace fs = std::filesystem;
using adai::TrainerAdminAPI;
using adai::TrainerControlState;
using adai::TrainerLogLevel;
using adai::TrainerPhase;

namespace {

// A port the kernel says is free. It used to come from a fixed range (43700 + pid % 100) inside
// Linux's ephemeral range (32768-60999), where other tests' outgoing connections can hold it; with
// exclusive listening ports (TD-551) the bind then fails and the API never comes up.
int free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    ::close(fd);
    return ntohs(addr.sin_port);
}

/// The admin API's server thread. A failed ASSERT used to return with a raw std::thread still
/// joinable, and its destructor calls std::terminate, so the whole binary aborted ("Subprocess
/// aborted" under a parallel ctest) instead of one test failing. This stops and joins the API on
/// destruction.
class AdminThread {
   public:
    AdminThread() = default;
    ~AdminThread() {
        if (thread_.joinable()) {
            api_->stop();
            thread_.join();
        }
    }
    AdminThread(const AdminThread&) = delete;
    AdminThread& operator=(const AdminThread&) = delete;

    void start(TrainerAdminAPI& api) {
        api_ = &api;
        thread_ = std::thread([this] {
            ok_ = api_->start();
            finished_ = true;
        });
    }
    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    /// start() returned without serving (it returns false when the port can't be bound).
    bool start_failed() const {
        return finished_ && !ok_;
    }

   private:
    TrainerAdminAPI* api_ = nullptr;
    std::atomic<bool> ok_{false};
    std::atomic<bool> finished_{false};
    std::thread thread_;
};

/// Starts `api` on `thread` and polls GET /health until it responds, start() fails, or 10s elapse
/// (3s used to be too short under a loaded parallel ctest). Mirrors
/// incremental_trainer_registry_test.cpp's start_mock_server() helper.
bool start_admin_api(TrainerAdminAPI& api, int port, AdminThread& thread) {
    thread.start(api);

    httplib::Client probe("127.0.0.1", port);
    probe.set_connection_timeout(std::chrono::milliseconds(200));
    probe.set_read_timeout(std::chrono::milliseconds(200));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto res = probe.Get("/health"); res && res->status == 200) {
            return true;
        }
        if (thread.start_failed()) {
            ADD_FAILURE() << "the admin API on port " << port << " failed to start";
            return false;
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
        port = free_port();
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
    AdminThread thr;
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
    AdminThread thr;
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
    AdminThread thr;
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
    AdminThread thr;
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
    AdminThread thr;
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

// TD-519: a space after the colon (Python json.dumps' default, hand-typed curl) used to read
// `true` as false, turning auto-save off and persisting it.
TEST_F(TrainerAdminAPITest, PutConfigAcceptsSpacedAndPrettyPrintedJson) {
    control->auto_save_enabled = false;
    auto api = make_api();
    AdminThread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    auto res = client.Put("/admin/config", "{\"auto_save_enabled\": true}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_TRUE(control->auto_save_enabled.load());

    control->auto_save_enabled = false;
    res = client.Put("/admin/config",
                     "{\n  \"auto_save_enabled\" : true,\n  \"max_sessions_to_keep\" : 9\n}",
                     "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_TRUE(control->auto_save_enabled.load());
    EXPECT_EQ(control->max_sessions_to_keep.load(), 9);

    api->stop();
    thr.join();
}

// TD-519: non-boolean / non-integer values and malformed JSON are 400s, and a rejected body
// changes nothing, even keys before the bad one.
TEST_F(TrainerAdminAPITest, PutConfigRejectedBodyChangesNothing) {
    auto api = make_api();
    AdminThread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));
    const int samples_before = control->auto_save_every_samples.load();

    httplib::Client client("127.0.0.1", port);
    for (const std::string body :
         {"{\"auto_save_enabled\":\"true\"}", "{\"auto_save_enabled\":1}",
          "{\"auto_save_every_samples\":\"500\"}", "{\"auto_save_every_samples\":1.5}",
          "{\"auto_save_enabled\":false", "[]",
          "{\"auto_save_enabled\":false,\"auto_save_every_samples\":123,"
          "\"max_sessions_to_keep\":0}"}) {
        auto res = client.Put("/admin/config", body, "application/json");
        ASSERT_TRUE(res);
        EXPECT_EQ(res->status, 400) << body;
    }
    EXPECT_TRUE(control->auto_save_enabled.load());
    EXPECT_EQ(control->auto_save_every_samples.load(), samples_before);

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, PutConfigPersistsAcrossRestart) {
    {
        auto api = make_api();
        AdminThread thr;
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
    AdminThread thr2;
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
    AdminThread thr;
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

// TD-520: a diverged pass's NaN/inf loss used to be streamed as `nan`/`inf`, which made the whole
// status invalid JSON exactly when an operator needed it.
TEST_F(TrainerAdminAPITest, StatusIsValidJsonWhenTheLossIsNotFinite) {
    control->phase = TrainerPhase::Training;
    control->samples_trained_this_pass = 10;
    control->best_loss = 0.5;

    auto api = make_api();
    AdminThread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));
    httplib::Client client("127.0.0.1", port);

    for (const double bad : {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()}) {
        control->last_loss = bad;
        auto res = client.Get("/admin/status");
        ASSERT_TRUE(res);
        EXPECT_EQ(res->status, 200);
        nlohmann::json j;
        ASSERT_NO_THROW(j = nlohmann::json::parse(res->body)) << res->body;
        EXPECT_TRUE(j["last_loss"].is_null()) << res->body;
        EXPECT_TRUE(j["loss_diverged"].get<bool>());
        EXPECT_DOUBLE_EQ(j["best_loss"].get<double>(), 0.5);
        EXPECT_EQ(j["phase"], "training");
    }

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, StatusLossesAreNullUntilAPassReportsOne) {
    auto api = make_api();
    AdminThread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));
    httplib::Client client("127.0.0.1", port);

    {
        auto res = client.Get("/admin/status");
        ASSERT_TRUE(res);
        const auto j = nlohmann::json::parse(res->body);
        EXPECT_TRUE(j["last_loss"].is_null());
        EXPECT_TRUE(j["best_loss"].is_null()) << "an unset best loss must not read as a perfect 0";
        EXPECT_FALSE(j["loss_diverged"].get<bool>());
    }

    // Finite losses come through exactly.
    control->samples_trained_this_pass = 3;
    control->last_loss = 0.1234567890123;
    control->best_loss = 1e-7;
    {
        auto res = client.Get("/admin/status");
        ASSERT_TRUE(res);
        const auto j = nlohmann::json::parse(res->body);
        EXPECT_DOUBLE_EQ(j["last_loss"].get<double>(), 0.1234567890123);
        EXPECT_DOUBLE_EQ(j["best_loss"].get<double>(), 1e-7);
        EXPECT_FALSE(j["loss_diverged"].get<bool>());
    }

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, LogsIncludesTheStartupEntryOnceTheServerIsUp) {
    // start() itself logs "Admin API listening on ..." via control_->log()
    // before it starts accepting connections, so by the time any HTTP
    // request can reach the server, at least that one entry always exists —
    // there is no reachable "truly empty" state to assert on here.
    auto api = make_api();
    AdminThread thr;
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
    AdminThread thr;
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
    AdminThread thr;
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
    AdminThread thr;
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
    AdminThread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    // Simulate what run_training()'s sample callback does: consume the
    // one-shot flag and bump the counters once it "writes" a checkpoint.
    std::atomic<bool> stop_worker{false};
    std::thread worker([&] {
        while (!stop_worker.load()) {
            if (const long long ticket = control->take_checkpoint_request()) {
                control->set_last_checkpoint_path("/tmp/auto_save_session_1.bin");
                control->checkpoints_written.fetch_add(1);
                control->mark_checkpoint_served(ticket);
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

// TD-521: an auto-save landing during the wait isn't the requested checkpoint (completion used to be
// "checkpoints_written went up").
TEST_F(TrainerAdminAPITest, AnAutoSaveDuringTheWaitIsNotTheRequestedCheckpoint) {
    control->phase = TrainerPhase::Training;
    auto api = make_api();
    AdminThread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    std::thread auto_saver([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        control->checkpoints_written.fetch_add(1);  // an auto-save; the request isn't taken
    });

    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(std::chrono::seconds(5));
    auto res = client.Post("/admin/checkpoint?wait_ms=600");
    auto_saver.join();
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 202);
    EXPECT_NE(res->body.find("\"completed\":false"), std::string::npos) << res->body;
    EXPECT_TRUE(control->checkpoint_requested.load());  // still pending for the pass

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, CheckpointRejectsAMalformedOrOversizedWait) {
    control->phase = TrainerPhase::Training;
    auto api = make_api();
    AdminThread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(std::chrono::seconds(5));
    for (const char* bad : {"abc", "100x", "-1", "60001", "99999999999"}) {
        auto res = client.Post(std::string("/admin/checkpoint?wait_ms=") + bad);
        ASSERT_TRUE(res) << bad;
        EXPECT_EQ(res->status, 400) << bad;
        EXPECT_NE(res->body.find("wait_ms"), std::string::npos) << bad;
    }
    EXPECT_FALSE(control->checkpoint_requested.load());
    EXPECT_EQ(control->checkpoint_requests_made.load(), 0);

    api->stop();
    thr.join();
}

// TD-521: saved overrides used to be applied unchecked on load (only PUT validated); a bad one is now
// skipped and logged, keeping the config file's value.
TEST_F(TrainerAdminAPITest, InvalidSavedOverridesAreIgnoredOnLoad) {
    {
        adai::DaemonConfigStore store((test_dir / "daemon_config.db").string());
        store.set("auto_save_every_samples", "-5");
        store.set("max_sessions_to_keep", "0");
        store.set("auto_save_enabled", "maybe");
        store.set("auto_save_every_minutes", "12");
    }
    control->auto_save_every_samples = 250;
    control->max_sessions_to_keep = 9;
    control->auto_save_enabled = false;

    auto api = make_api();
    AdminThread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));

    EXPECT_EQ(control->auto_save_every_samples.load(), 250);
    EXPECT_EQ(control->max_sessions_to_keep.load(), 9);
    EXPECT_FALSE(control->auto_save_enabled.load());
    EXPECT_EQ(control->auto_save_every_minutes.load(), 12);

    httplib::Client client("127.0.0.1", port);
    auto res = client.Get("/admin/logs");
    ASSERT_TRUE(res);
    EXPECT_NE(res->body.find("auto_save_every_samples='-5'"), std::string::npos) << res->body;

    api->stop();
    thr.join();
}

TEST_F(TrainerAdminAPITest, PauseSetsPausedAndIsIdempotent) {
    auto api = make_api();
    AdminThread thr;
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
    EXPECT_EQ(control->phase.load(), TrainerPhase::Idle);  // nothing to drain

    api->stop();
    thr.join();
}

// TD-522: a pause used to leave the reported phase at `training` until the pass ended; `pausing`
// was listed but never set. /admin/resume no longer calls a wake() nothing waits on.
TEST_F(TrainerAdminAPITest, PauseReportsPausingAndResumeRestoresThePhase) {
    control->set_phase(TrainerPhase::Training);

    auto api = make_api();
    AdminThread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));
    httplib::Client client("127.0.0.1", port);

    ASSERT_TRUE(client.Post("/admin/pause"));
    auto status = client.Get("/admin/status");
    ASSERT_TRUE(status);
    EXPECT_EQ(nlohmann::json::parse(status->body)["phase"], "pausing") << status->body;

    auto res = client.Post("/admin/resume");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 202);
    EXPECT_FALSE(control->paused.load());
    status = client.Get("/admin/status");
    ASSERT_TRUE(status);
    EXPECT_EQ(nlohmann::json::parse(status->body)["phase"], "training") << status->body;

    api->stop();
    thr.join();
}

// TD-522: GET /admin/logs?limit=N returns the newest N entries; the limit used to be ignored.
TEST_F(TrainerAdminAPITest, LogsHonorsAndChecksTheLimit) {
    auto api = make_api();
    AdminThread thr;
    ASSERT_TRUE(start_admin_api(*api, port, thr));
    control->log(TrainerLogLevel::Info, "one");  // after the API's own startup entries
    control->log(TrainerLogLevel::Info, "two");
    control->log(TrainerLogLevel::Info, "three");
    httplib::Client client("127.0.0.1", port);

    auto res = client.Get("/admin/logs?limit=2");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200);
    const auto entries = nlohmann::json::parse(res->body)["entries"];
    ASSERT_EQ(entries.size(), 2u) << res->body;
    EXPECT_EQ(entries[0]["message"], "two");
    EXPECT_EQ(entries[1]["message"], "three");

    res = client.Get("/admin/logs?limit=0");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_TRUE(nlohmann::json::parse(res->body)["entries"].empty());

    for (const char* bad : {"abc", "-1", "201", "2x", ""}) {
        res = client.Get(std::string("/admin/logs?limit=") + bad);
        ASSERT_TRUE(res);
        EXPECT_EQ(res->status, 400) << "limit=" << bad;
    }

    api->stop();
    thr.join();
}
