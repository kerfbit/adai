// TD-283: SIGHUP used to terminate every binary except chatbot_api_server. Each daemon is started for
// real, sent SIGHUP once it answers /health, and must still be running and answering afterwards.

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "../src/SignalSetup.hpp"
#include "cpp-httplib/httplib.h"

namespace fs = std::filesystem;

namespace {

int free_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    ::close(fd);
    return ntohs(addr.sin_port);
}

class SighupTest : public ::testing::Test {
   protected:
    fs::path dir_;
    pid_t pid_ = -1;

    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("adai_td283_" + std::to_string(::getpid()) + "_" +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::remove_all(dir_);
        fs::create_directories(dir_);
    }

    void TearDown() override {
        if (pid_ > 0) {
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, nullptr, 0);
        }
        fs::remove_all(dir_);
    }

    fs::path write_config(const std::string& name, const std::string& text) {
        const fs::path p = dir_ / name;
        std::ofstream(p) << text;
        return p;
    }

    // Starts `binary` with `args` in dir_, with no service URLs inherited from the environment.
    void spawn(const std::string& binary, const std::vector<std::string>& args) {
        pid_ = ::fork();
        ASSERT_GE(pid_, 0);
        if (pid_ == 0) {
            if (::chdir(dir_.c_str()) != 0) _exit(126);
            for (const char* var : {"REGISTRY_SERVER_URL", "NAME_SERVICE_URL", "METRICS_SERVER_URL"})
                ::unsetenv(var);
            const int devnull = ::open("/dev/null", O_WRONLY);
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
            std::vector<char*> argv;
            argv.push_back(const_cast<char*>(binary.c_str()));
            for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);
            ::execv(binary.c_str(), argv.data());
            _exit(127);
        }
    }

    bool healthy(int port) {
        httplib::Client client("127.0.0.1", port);
        client.set_connection_timeout(std::chrono::milliseconds(300));
        client.set_read_timeout(std::chrono::seconds(2));
        auto res = client.Get("/health");
        return res && res->status == 200;
    }

    bool still_running() {
        int status = 0;
        return ::waitpid(pid_, &status, WNOHANG) == 0;
    }

    // Waits for /health, sends SIGHUP, and checks the daemon survives and still answers.
    void expect_survives_sighup(int port, const std::string& name) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!healthy(port) && std::chrono::steady_clock::now() < deadline) {
            ASSERT_TRUE(still_running()) << name << " exited during startup";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ASSERT_TRUE(healthy(port)) << name << " never answered /health";

        ASSERT_EQ(::kill(pid_, SIGHUP), 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        int status = 0;
        const pid_t done = ::waitpid(pid_, &status, WNOHANG);
        if (done == pid_) {
            pid_ = -1;
            FAIL() << name << " died on SIGHUP"
                   << (WIFSIGNALED(status) ? " (signal " + std::to_string(WTERMSIG(status)) + ")"
                                           : " (exit " + std::to_string(WEXITSTATUS(status)) + ")");
        }
        EXPECT_TRUE(healthy(port)) << name << " stopped answering after SIGHUP";

        // SIGTERM still stops it.
        ASSERT_EQ(::kill(pid_, SIGTERM), 0);
        const auto stop_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (::waitpid(pid_, &status, WNOHANG) == 0 &&
               std::chrono::steady_clock::now() < stop_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (::kill(pid_, 0) != 0) pid_ = -1;  // reaped
        EXPECT_EQ(pid_, -1) << name << " didn't stop on SIGTERM";
    }
};

}  // namespace

TEST_F(SighupTest, MetricsApiServerSurvivesSighup) {
    const int port = free_port();
    const auto cfg = write_config("config.metrics.conf", "");
    spawn(METRICS_API_SERVER_BINARY_PATH,
          {"--config", cfg.string(), "--port", std::to_string(port), "--db-path",
           (dir_ / "metrics.db").string()});
    expect_survives_sighup(port, "metrics_api_server");
}

TEST_F(SighupTest, MnsServerSurvivesSighup) {
    const int port = free_port();
    const auto cfg = write_config("config.mns.conf", "");
    spawn(MNS_SERVER_BINARY_PATH, {"--config", cfg.string(), "--port", std::to_string(port),
                                   "--data-dir", (dir_ / "mns").string()});
    expect_survives_sighup(port, "mns_server");
}

TEST_F(SighupTest, RegistryServerSurvivesSighup) {
    const int port = free_port();
    const auto cfg = write_config("config.registry.conf", "");
    spawn(REGISTRY_SERVER_BINARY_PATH, {"--config", cfg.string(), "--port", std::to_string(port),
                                        "--data-dir", (dir_ / "registry").string()});
    expect_survives_sighup(port, "registry_server");
}

TEST_F(SighupTest, TrainerServiceSurvivesSighup) {
    const int port = free_port();
    const int child_port = free_port();
    const auto cfg = write_config(
        "config.trainer.conf",
        "TRAINER_ADMIN_ENABLED=true\nTRAINER_ADMIN_HOST=127.0.0.1\nTRAINER_ADMIN_PORT=" +
            std::to_string(port) + "\nTRAINER_CHILD_ADMIN_PORT=" + std::to_string(child_port) +
            "\nTRAINER_ADMIN_DIR=" + (dir_ / "admin").string() + "\nSESSION_DIR=" +
            (dir_ / "sessions").string() + "\nLOG_FILE_PATH=" + (dir_ / "trainer.log").string() +
            "\nREGISTRY_SERVER_URL=\nNAME_SERVICE_URL=\nMETRICS_SERVER_URL=\nGPU_ENABLED=false\n"
            "D_MODEL=64\nNUM_HEADS=2\nD_FF=128\nNUM_ENCODER_LAYERS=1\nNUM_DECODER_LAYERS=1\n"
            "MAX_SEQ_LENGTH=16\n");
    spawn(TRAINER_SERVICE_BINARY_PATH, {"--config", cfg.string()});
    expect_survives_sighup(port, "trainer_service");
}

// incremental_trainer calls the same helper before a pass starts; a process that calls it ignores
// SIGHUP (and so does anything it execs, as trainer_service's children do).
TEST(IgnoreSighupTest, AProcessThatCallsItIgnoresSighup) {
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        adai::ignore_sighup("sighup_test");
        ::raise(SIGHUP);
        _exit(0);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
        << (WIFSIGNALED(status) ? "killed by signal " + std::to_string(WTERMSIG(status)) : "");
}

// TD-288: every daemon and incremental_trainer run validate() at startup and exit 1 on an invalid
// value instead of starting with it.
TEST_F(SighupTest, EveryBinaryRefusesAnInvalidConfigAtStartup) {
    struct Case {
        std::string name, binary, file;
        std::vector<std::string> extra_args;
    };
    const std::vector<Case> cases = {
        {"metrics_api_server", METRICS_API_SERVER_BINARY_PATH, "config.metrics.conf",
         {"--port", std::to_string(free_port()), "--db-path", (dir_ / "metrics.db").string()}},
        {"mns_server", MNS_SERVER_BINARY_PATH, "config.mns.conf",
         {"--port", std::to_string(free_port()), "--data-dir", (dir_ / "mns").string()}},
        {"registry_server", REGISTRY_SERVER_BINARY_PATH, "config.registry.conf",
         {"--port", std::to_string(free_port()), "--data-dir", (dir_ / "registry").string()}},
        {"trainer_service", TRAINER_SERVICE_BINARY_PATH, "config.trainer.conf", {}},
        {"incremental_trainer", INCREMENTAL_TRAINER_BINARY_PATH, "config.trainer.conf", {"status"}},
    };
    for (const auto& c : cases) {
        // gradient_accumulation_steps = 0 is a divisor in ChatbotTrainer; the old validate() let it through.
        const auto cfg = write_config(c.file, "LOG_FILE_PATH=" + (dir_ / "log.txt").string() +
                                                  "\nGRADIENT_ACCUMULATION_STEPS=0\n");
        std::vector<std::string> args = {"--config", cfg.string()};
        args.insert(args.end(), c.extra_args.begin(), c.extra_args.end());
        spawn(c.binary, args);
        int status = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        pid_t done = 0;
        while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ASSERT_EQ(done, pid_) << c.name << " started with an invalid config";
        pid_ = -1;
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 1) << c.name;
    }
}

// TD-438: metrics_api_server printed usage and exited 0 on a bad argument, so a mistyped flag in a
// unit's ExecStart looked like a clean stop and Restart=on-failure didn't retry. With --port abc it
// listened on a random port.
TEST_F(SighupTest, MetricsApiServerExitsOneOnABadFlag) {
    const auto cfg = write_config("config.metrics.conf", "");
    for (const std::vector<std::string>& bad : std::vector<std::vector<std::string>>{
             {"--port", "abc"}, {"--port", "99999"}, {"--bogus"}, {"--max-live-sessions"}}) {
        std::vector<std::string> args = {"--config", cfg.string(), "--db-path", (dir_ / "metrics.db").string()};
        args.insert(args.end(), bad.begin(), bad.end());
        spawn(METRICS_API_SERVER_BINARY_PATH, args);
        int status = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        pid_t done = 0;
        while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ASSERT_EQ(done, pid_) << bad[0] << " started the server";
        pid_ = -1;
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 1) << bad[0];
    }
}

// TD-456: mns_server took a mistyped --admin-enabled value as true; now it's refused. And
// MNS_ADMIN_ENABLED=false in config.mns.conf closes PUT /admin/config.
TEST_F(SighupTest, MnsServerRefusesAMistypedAdminEnabledAndHonoursTheConfigKey) {
    const auto cfg = write_config("config.mns.conf", "MNS_ADMIN_ENABLED=false\n");
    spawn(MNS_SERVER_BINARY_PATH, {"--config", cfg.string(), "--port", std::to_string(free_port()), "--data-dir",
                                   (dir_ / "mns").string(), "--admin-enabled", "flase"});
    int status = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    pid_t done = 0;
    while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_EQ(done, pid_) << "mns_server started with --admin-enabled flase";
    pid_ = -1;
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 1);

    const int port = free_port();
    spawn(MNS_SERVER_BINARY_PATH,
          {"--config", cfg.string(), "--port", std::to_string(port), "--data-dir", (dir_ / "mns").string()});
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!healthy(port) && std::chrono::steady_clock::now() < deadline) {
        ASSERT_TRUE(still_running()) << "mns_server exited during startup";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    httplib::Client client("127.0.0.1", port);
    auto res = client.Put("/admin/config", R"({"registry_group":"hijacked"})", "application/json");
    ASSERT_TRUE(res);
    EXPECT_NE(res->status, 200) << res->body;
}

// TD-488: registry_server aborted with a core dump on --port abc (exit 134), ignored unknown arguments
// (--portt 8091 started on 8082) and dropped a trailing --admin-enabled (leaving the admin endpoint
// open). Each is now an error and exit 1.
TEST_F(SighupTest, RegistryServerExitsOneOnABadFlag) {
    const auto cfg = write_config("config.registry.conf", "");
    for (const std::vector<std::string>& bad : std::vector<std::vector<std::string>>{
             {"--port", "abc"}, {"--port", "99999"}, {"--portt", "8091"}, {"--admin-enabled"}}) {
        std::vector<std::string> args = {"--config", cfg.string(), "--data-dir", (dir_ / "registry").string()};
        args.insert(args.end(), bad.begin(), bad.end());
        spawn(REGISTRY_SERVER_BINARY_PATH, args);
        int status = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        pid_t done = 0;
        while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ASSERT_EQ(done, pid_) << bad[0] << " started the server";
        pid_ = -1;
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 1)
            << bad[0] << (WIFSIGNALED(status) ? " killed by signal " + std::to_string(WTERMSIG(status)) : "");
    }
}

// TD-401: incremental_trainer aborted with SIGABRT on --admin-port abc (an uncaught std::stoi), and
// a mistyped --objective or a valueless --config ran anyway. Each is now an error and exit 1.
TEST_F(SighupTest, IncrementalTrainerExitsOneOnABadFlag) {
    const auto cfg = write_config("config.trainer.conf", "LOG_FILE_PATH=" + (dir_ / "log.txt").string() + "\n");
    for (const std::vector<std::string>& bad : std::vector<std::vector<std::string>>{
             {"--admin-port", "abc", "status"},
             {"--admin-port", "99999", "status"},
             {"--objective=lejpa", "status"},
             {"--dataset-kind", "../x", "status"},  // TD-402
             {"status", "--config"}}) {
        std::vector<std::string> args = {"--config", cfg.string()};
        args.insert(args.end(), bad.begin(), bad.end());
        spawn(INCREMENTAL_TRAINER_BINARY_PATH, args);
        int status = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        pid_t done = 0;
        while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ASSERT_EQ(done, pid_) << bad[0] << " did not exit";
        pid_ = -1;
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 1)
            << bad[0] << (WIFSIGNALED(status) ? " killed by signal " + std::to_string(WTERMSIG(status)) : "");
    }
}

// TD-557: trainer_service --help started the service (it never exited), and a bad flag or a stray
// argument was ignored.
TEST_F(SighupTest, TrainerServiceHelpExitsZeroAndBadArgumentsExitOne) {
    const auto cfg = write_config("config.trainer.conf", "LOG_FILE_PATH=" + (dir_ / "log.txt").string() + "\n");
    const std::vector<std::pair<std::vector<std::string>, int>> cases = {
        {{"--help"}, 0}, {{"-h"}, 0}, {{"resume"}, 1}, {{"--admin-port", "abc"}, 1}, {{"--config"}, 1}};
    for (const auto& [extra, expected] : cases) {
        std::vector<std::string> args = {"--config", cfg.string()};
        args.insert(args.end(), extra.begin(), extra.end());
        spawn(TRAINER_SERVICE_BINARY_PATH, args);
        int status = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        pid_t done = 0;
        while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ASSERT_EQ(done, pid_) << extra.front() << " started the service instead of exiting";
        pid_ = -1;
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == expected) << extra.front();
    }
}

// TD-489: REGISTRY_ADMIN_ENABLED=false in config.registry.conf closes PUT /admin/config. registry_server
// had no file key for it, so only --admin-enabled false could.
TEST_F(SighupTest, RegistryServerHonoursRegistryAdminEnabled) {
    const auto cfg = write_config("config.registry.conf", "REGISTRY_ADMIN_ENABLED=false\n");
    const int port = free_port();
    spawn(REGISTRY_SERVER_BINARY_PATH,
          {"--config", cfg.string(), "--port", std::to_string(port), "--data-dir", (dir_ / "registry").string()});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!healthy(port) && std::chrono::steady_clock::now() < deadline) {
        ASSERT_TRUE(still_running()) << "registry_server exited during startup";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    httplib::Client client("127.0.0.1", port);
    auto res = client.Put("/admin/config", R"({"ftp_token_ttl_minutes":5})", "application/json");
    ASSERT_TRUE(res);
    EXPECT_NE(res->status, 200) << res->body;
    auto get = client.Get("/admin/config");
    ASSERT_TRUE(get);
    EXPECT_EQ(get->body.find("\"ftp_token_ttl_minutes\":5"), std::string::npos) << get->body;
}

