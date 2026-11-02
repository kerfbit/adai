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

#include <algorithm>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "../src/BPETokenizer.hpp"
#include "../src/DatasetRegistry.hpp"
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
             {"train", "abc"},                       // TD-416: aborted (exit 134)
             {"retrain", "0"},                       // TD-416
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


// TD-403: trainer_service parsed --objective/--dataset-kind and dropped them, so the always-on service
// could only train the chatbot from the chatbot pool. A stub incremental_trainer beside a copy of the
// binary records the arguments each pass gets.
TEST_F(SighupTest, TrainerServicePassesObjectiveAndDatasetKindToEveryPass) {
    const fs::path bin = dir_ / "bin";
    fs::create_directories(bin);
    fs::copy_file(TRAINER_SERVICE_BINARY_PATH, bin / "trainer_service");
    const fs::path recorded = dir_ / "child_args.txt";
    {
        std::ofstream stub(bin / "incremental_trainer");
        stub << "#!/bin/sh\nprintf '%s\\n' \"$@\" > '" << recorded.string() << ".tmp'\n"
             << "mv '" << recorded.string() << ".tmp' '" << recorded.string() << "'\nexit 3\n";
    }
    fs::permissions(bin / "incremental_trainer", fs::perms::owner_all);
    const auto cfg = write_config("config.trainer.conf", "LOG_FILE_PATH=" + (dir_ / "log.txt").string() +
                                                             "\nTRAINER_CHILD_ADMIN_PORT=" +
                                                             std::to_string(free_port()) + "\n");
    const auto read_args = [&](const std::vector<std::string>& flags) {
        fs::remove(recorded);
        std::vector<std::string> args = {"--config", cfg.string()};
        args.insert(args.end(), flags.begin(), flags.end());
        spawn((bin / "trainer_service").string(), args);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        while (!fs::exists(recorded) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ::kill(pid_, SIGTERM);
        ::waitpid(pid_, nullptr, 0);
        pid_ = -1;
        std::vector<std::string> lines;
        std::ifstream in(recorded);
        for (std::string line; std::getline(in, line);) lines.push_back(line);
        return lines;
    };
    const auto has = [](const std::vector<std::string>& v, const std::vector<std::string>& seq) {
        return std::search(v.begin(), v.end(), seq.begin(), seq.end()) != v.end();
    };

    auto args = read_args({"--objective=lejepa"});
    ASSERT_FALSE(args.empty()) << "no pass was launched";
    EXPECT_TRUE(has(args, {"--objective=lejepa"}));
    EXPECT_TRUE(has(args, {"--dataset-kind", "world_model"}));  // the objective's default pool
    EXPECT_EQ(args.back(), "resume");

    args = read_args({"--dataset-kind", "encoder"});
    EXPECT_TRUE(has(args, {"--objective=chatbot"}));
    EXPECT_TRUE(has(args, {"--dataset-kind", "encoder"}));
}

// TD-403: resume, the command every trainer_service pass runs, ignored --objective and always ran
// chatbot training. With --objective=lejepa it now runs the world-model pass, with resume's exit codes.
TEST_F(SighupTest, LejepaResumeTrainsTheWorldModelAndThenReportsNothingPending) {
    BPETokenizer tokenizer;
    tokenizer.build_vocab({"hello world", "how are you", "good morning", "the cat sat"}, 60);
    tokenizer.save_vocab((dir_ / "vocab.txt").string());
    const fs::path data = dir_ / "pretrain.jsonl";
    std::ofstream(data) << "{\"input\":\"hello world\"}\n{\"input\":\"how are you\"}\n"
                        << "{\"input\":\"good morning\"}\n{\"input\":\"the cat sat\"}\n";
    const fs::path sessions = dir_ / "sessions";
    {
        DatasetConfig dcfg;
        dcfg.session_dir = sessions.string();
        dcfg.dataset_kind = "world_model";
        fs::create_directories(DatasetRegistry::local_pool_dir(dcfg));
        DatasetRegistry reg(dcfg);
        reg.load_registry();
        reg.load_pending_list();
        ASSERT_TRUE(reg.add_file(data.string()));
        ASSERT_TRUE(reg.save_pending_list());
    }
    const auto cfg = write_config(
        "config.trainer.conf",
        "LOG_FILE_PATH=" + (dir_ / "log.txt").string() + "\nSESSION_DIR=" + sessions.string() +
            "\nVOCAB_PATH=" + (dir_ / "vocab.txt").string() +
            "\nREGISTRY_SERVER_URL=\nNAME_SERVICE_URL=\nMETRICS_SERVER_URL=\nGPU_ENABLED=false\n"
            "NUM_EPOCHS=1\nMAX_SEQ_LENGTH=16\nWORLD_MODEL_ENABLED=true\nWORLD_MODEL_D_MODEL=64\n"
            "WORLD_MODEL_NUM_LAYERS=1\nWORLD_MODEL_NUM_HEADS=2\nWORLD_MODEL_D_FF=128\n"
            "WORLD_MODEL_SIGREG_BATCH_SIZE=2\n");
    const auto run_resume = [&]() {
        spawn(INCREMENTAL_TRAINER_BINARY_PATH, {"--config", cfg.string(), "--foreground", "--objective=lejepa", "resume"});
        int status = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        pid_t done = 0;
        while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (done != pid_) return -1;
        pid_ = -1;
        return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    };
    std::stringstream log;
    EXPECT_EQ(run_resume(), 0);
    log << std::ifstream(dir_ / "log.txt").rdbuf();
    EXPECT_TRUE(fs::exists(sessions / "world_model_checkpoint" / "config.bin"))
        << "no world-model checkpoint was written:\n" << log.str();
    EXPECT_EQ(run_resume(), 3) << "the file is trained, so the next pass has nothing pending";
}

// TD-415: the LeJEPA pass loaded the whole file for every claimed segment, and its PID-based run id left a
// crashed pass's claims stuck. A file queued as two 2-row segments, both already claimed by a "crashed"
// pass under the persisted run id: the next pass reclaims them and trains 2 rows per segment.
TEST_F(SighupTest, LejepaPassTrainsEachSegmentOnceAndReclaimsAfterACrash) {
    BPETokenizer tokenizer;
    tokenizer.build_vocab({"hello world", "how are you", "good morning", "the cat sat"}, 60);
    tokenizer.save_vocab((dir_ / "vocab.txt").string());
    const fs::path data = dir_ / "pretrain.jsonl";
    std::ofstream(data) << "{\"input\":\"hello world\"}\n{\"input\":\"how are you\"}\n"
                        << "{\"input\":\"good morning\"}\n{\"input\":\"the cat sat\"}\n";
    const fs::path sessions = dir_ / "sessions";
    const fs::path checkpoint = sessions / "world_model_checkpoint";
    fs::create_directories(checkpoint);
    std::ofstream(checkpoint / "lejepa_run_id") << "lejepa-crashed-pass\n";
    DatasetConfig dcfg;
    dcfg.session_dir = sessions.string();
    dcfg.dataset_kind = "world_model";
    fs::create_directories(DatasetRegistry::local_pool_dir(dcfg));
    {
        DatasetRegistry reg(dcfg);
        reg.load_registry();
        reg.load_pending_list();
        ASSERT_TRUE(reg.add_file(data.string(), 0, 2));
        ASSERT_TRUE(reg.add_file(data.string(), 2, 2));
        ASSERT_TRUE(reg.save_pending_list());
        ASSERT_EQ(reg.acquire_pending("lejepa-crashed-pass").files.size(), 2u);  // then the pass "crashed"
    }
    const auto cfg = write_config(
        "config.trainer.conf",
        "LOG_FILE_PATH=" + (dir_ / "log.txt").string() + "\nSESSION_DIR=" + sessions.string() +
            "\nVOCAB_PATH=" + (dir_ / "vocab.txt").string() +
            "\nREGISTRY_SERVER_URL=\nNAME_SERVICE_URL=\nMETRICS_SERVER_URL=\nRUN_ID=\nGPU_ENABLED=false\n"
            "NUM_EPOCHS=1\nMAX_SEQ_LENGTH=16\nWORLD_MODEL_ENABLED=true\nWORLD_MODEL_D_MODEL=64\n"
            "WORLD_MODEL_NUM_LAYERS=1\nWORLD_MODEL_NUM_HEADS=2\nWORLD_MODEL_D_FF=128\n"
            "WORLD_MODEL_SIGREG_BATCH_SIZE=2\n");
    spawn(INCREMENTAL_TRAINER_BINARY_PATH, {"--config", cfg.string(), "--foreground", "--objective=lejepa", "resume"});
    int status = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    pid_t done = 0;
    while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_EQ(done, pid_);
    pid_ = -1;
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "the crashed pass's claims weren't reclaimed";

    // One trained record per segment, each with its own 2 rows (the whole file's 4 each, before).
    std::ifstream registry(fs::path(DatasetRegistry::local_pool_dir(dcfg)) / dcfg.data_registry_file);
    std::vector<int> samples;
    for (std::string line; std::getline(registry, line);) {
        std::istringstream fields(line);
        std::string file, checksum;
        int n = -1;
        if (fields >> file >> checksum >> n) samples.push_back(n);
    }
    EXPECT_EQ(samples, (std::vector<int>{2, 2}));
}

// TD-403: the LeJEPA pass had no per-pass admin API, so while trainer_service ran one, /admin/status showed
// only the supervisor's state, /admin/checkpoint did nothing and a pause waited for the pass to end. It now
// reports progress, checkpoints on request, and drains on pause (checkpoint, release its files, exit 1).
TEST_F(SighupTest, LejepaPassAnswersTheAdminApiAndDrainsOnPause) {
    BPETokenizer tokenizer;
    tokenizer.build_vocab({"hello world", "how are you", "good morning", "the cat sat"}, 60);
    tokenizer.save_vocab((dir_ / "vocab.txt").string());
    const fs::path data = dir_ / "pretrain.jsonl";
    {
        std::ofstream out(data);
        const char* lines[] = {"hello world", "how are you", "good morning", "the cat sat"};
        for (int i = 0; i < 400; ++i) out << "{\"input\":\"" << lines[i % 4] << " " << lines[(i + 1) % 4] << "\"}\n";
    }
    const fs::path sessions = dir_ / "sessions";
    DatasetConfig dcfg;
    dcfg.session_dir = sessions.string();
    dcfg.dataset_kind = "world_model";
    fs::create_directories(DatasetRegistry::local_pool_dir(dcfg));
    {
        DatasetRegistry reg(dcfg);
        reg.load_registry();
        reg.load_pending_list();
        ASSERT_TRUE(reg.add_file(data.string()));
        ASSERT_TRUE(reg.save_pending_list());
    }
    const int admin_port = free_port();
    const auto cfg = write_config(
        "config.trainer.conf",
        "LOG_FILE_PATH=" + (dir_ / "log.txt").string() + "\nSESSION_DIR=" + sessions.string() +
            "\nVOCAB_PATH=" + (dir_ / "vocab.txt").string() +
            "\nREGISTRY_SERVER_URL=\nNAME_SERVICE_URL=\nMETRICS_SERVER_URL=\nRUN_ID=\nGPU_ENABLED=false\n"
            "AUTO_SAVE_ENABLED=false\nNUM_EPOCHS=1000\nMAX_SEQ_LENGTH=16\nWORLD_MODEL_ENABLED=true\n"
            "WORLD_MODEL_D_MODEL=64\nWORLD_MODEL_NUM_LAYERS=1\nWORLD_MODEL_NUM_HEADS=2\nWORLD_MODEL_D_FF=128\n"
            "WORLD_MODEL_SIGREG_BATCH_SIZE=2\n");
    spawn(INCREMENTAL_TRAINER_BINARY_PATH, {"--config", cfg.string(), "--foreground", "--admin-port",
                                            std::to_string(admin_port), "--objective=lejepa", "resume"});
    httplib::Client client("127.0.0.1", admin_port);
    client.set_connection_timeout(std::chrono::milliseconds(300));
    client.set_read_timeout(std::chrono::seconds(30));

    // Training, with the pass's run id and progress.
    std::string status;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        ASSERT_TRUE(still_running()) << "the pass ended before it was seen training";
        auto res = client.Get("/admin/status");
        if (res && res->status == 200 && res->body.find("\"phase\":\"training\"") != std::string::npos &&
            res->body.find("\"samples_trained_this_pass\":0,") == std::string::npos) {
            status = res->body;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_FALSE(status.empty()) << "/admin/status never reported training";
    EXPECT_NE(status.find("\"run_id\":\"lejepa-"), std::string::npos) << status;

    // A requested checkpoint is written.
    auto checkpoint = client.Post("/admin/checkpoint?wait_ms=20000", "", "application/json");
    ASSERT_TRUE(checkpoint);
    EXPECT_EQ(checkpoint->status, 202) << checkpoint->body;
    EXPECT_NE(checkpoint->body.find("\"completed\":true"), std::string::npos) << checkpoint->body;
    EXPECT_TRUE(fs::exists(sessions / "world_model_checkpoint" / "config.bin"));

    // Pause drains the pass: it exits 1 (as a paused chatbot pass does) and its file goes back to the pool.
    auto pause = client.Post("/admin/pause", "", "application/json");
    ASSERT_TRUE(pause);
    EXPECT_EQ(pause->status, 202);
    int exit_status = 0;
    pid_t done = 0;
    const auto stop_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while ((done = ::waitpid(pid_, &exit_status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < stop_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_EQ(done, pid_) << "the pass didn't stop on /admin/pause";
    pid_ = -1;
    EXPECT_TRUE(WIFEXITED(exit_status) && WEXITSTATUS(exit_status) == 1);
    DatasetRegistry reg(dcfg);
    reg.load_pending_list();
    const auto entries = reg.pending_entries();
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_TRUE(entries[0].run_id.empty()) << "the paused pass's file is still claimed by " << entries[0].run_id;
}

// TD-416: SIGTERM ran a std::exit(0) from the handler: exit 0, no checkpoint, and the claimed file stayed
// claimed. It now drains the pass like /admin/pause: checkpoint, release, exit 1.
TEST_F(SighupTest, SigtermDrainsALejepaPass) {
    BPETokenizer tokenizer;
    tokenizer.build_vocab({"hello world", "how are you", "good morning", "the cat sat"}, 60);
    tokenizer.save_vocab((dir_ / "vocab.txt").string());
    const fs::path data = dir_ / "pretrain.jsonl";
    {
        std::ofstream out(data);
        const char* lines[] = {"hello world", "how are you", "good morning", "the cat sat"};
        for (int i = 0; i < 400; ++i) out << "{\"input\":\"" << lines[i % 4] << " " << lines[(i + 1) % 4] << "\"}\n";
    }
    const fs::path sessions = dir_ / "sessions";
    DatasetConfig dcfg;
    dcfg.session_dir = sessions.string();
    dcfg.dataset_kind = "world_model";
    fs::create_directories(DatasetRegistry::local_pool_dir(dcfg));
    {
        DatasetRegistry reg(dcfg);
        reg.load_registry();
        reg.load_pending_list();
        ASSERT_TRUE(reg.add_file(data.string()));
        ASSERT_TRUE(reg.save_pending_list());
    }
    const fs::path log = dir_ / "log.txt";
    const auto cfg = write_config(
        "config.trainer.conf",
        "LOG_FILE_PATH=" + log.string() + "\nSESSION_DIR=" + sessions.string() +
            "\nVOCAB_PATH=" + (dir_ / "vocab.txt").string() +
            "\nREGISTRY_SERVER_URL=\nNAME_SERVICE_URL=\nMETRICS_SERVER_URL=\nRUN_ID=\nGPU_ENABLED=false\n"
            "AUTO_SAVE_ENABLED=false\nNUM_EPOCHS=1000\nMAX_SEQ_LENGTH=16\nWORLD_MODEL_ENABLED=true\n"
            "WORLD_MODEL_D_MODEL=64\nWORLD_MODEL_NUM_LAYERS=1\nWORLD_MODEL_NUM_HEADS=2\nWORLD_MODEL_D_FF=128\n"
            "WORLD_MODEL_SIGREG_BATCH_SIZE=2\n");
    spawn(INCREMENTAL_TRAINER_BINARY_PATH, {"--config", cfg.string(), "--foreground", "--objective=lejepa", "resume"});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    std::string text;
    while (text.find("Training the world model on") == std::string::npos &&
           std::chrono::steady_clock::now() < deadline) {
        ASSERT_TRUE(still_running()) << "the pass ended before it was training";
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::stringstream s;
        s << std::ifstream(log).rdbuf();
        text = s.str();
    }
    ASSERT_NE(text.find("Training the world model on"), std::string::npos) << "never saw training";
    std::this_thread::sleep_for(std::chrono::milliseconds(300));  // a few batches in

    ASSERT_EQ(::kill(pid_, SIGTERM), 0);
    int status = 0;
    pid_t done = 0;
    const auto stop_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < stop_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_EQ(done, pid_) << "SIGTERM didn't stop the pass";
    pid_ = -1;
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 1) << "a stopped pass isn't a successful one";
    EXPECT_TRUE(fs::exists(sessions / "world_model_checkpoint" / "config.bin")) << "no checkpoint on SIGTERM";
    DatasetRegistry reg(dcfg);
    reg.load_pending_list();
    const auto entries = reg.pending_entries();
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_TRUE(entries[0].run_id.empty()) << "the stopped pass's file is still claimed by " << entries[0].run_id;
}

// TD-416: status built a whole IncrementalTrainer (model, checkpoint load, the deleting
// cleanup_dead_sessions()) and, with no vocabulary yet, aborted (TD-414).
TEST_F(SighupTest, StatusAndHistoryAreReadOnlyAndNeedNoVocabulary) {
    const fs::path sessions = dir_ / "sessions";
    fs::create_directories(sessions);
    const fs::path stray = sessions / "session_7_checkpoint.bin";  // unreferenced: cleanup would delete it
    std::ofstream(stray) << "x";
    const auto cfg = write_config("config.trainer.conf",
                                  "LOG_FILE_PATH=" + (dir_ / "log.txt").string() + "\nSESSION_DIR=" +
                                      sessions.string() + "\nVOCAB_PATH=" + (dir_ / "missing_vocab.txt").string() +
                                      "\nREGISTRY_SERVER_URL=\nNAME_SERVICE_URL=\nGPU_ENABLED=false\n");
    for (const std::string command : {"status", "history"}) {
        spawn(INCREMENTAL_TRAINER_BINARY_PATH, {"--config", cfg.string(), command});
        int status = 0;
        pid_t done = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ASSERT_EQ(done, pid_) << command;
        pid_ = -1;
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
            << command << (WIFSIGNALED(status) ? " killed by signal " + std::to_string(WTERMSIG(status)) : "");
    }
    EXPECT_TRUE(fs::exists(stray)) << "a read-only command deleted a checkpoint file";
}

// TD-414: on a fresh SESSION_DIR with data queued and no vocabulary, `train` aborted with VocabularyFileError
// (exit 134): the first-run vocabulary bootstrap was only in a constructor incremental_trainer never used. And
// any exception from a command reached std::terminate.
TEST_F(SighupTest, AFirstTrainBuildsTheVocabularyAndExceptionsExitOne) {
    const fs::path data = dir_ / "pairs.jsonl";
    {
        std::ofstream out(data);
        const char* lines[] = {"hello world", "how are you", "good morning", "the cat sat"};
        for (int i = 0; i < 20; ++i)
            out << "{\"input\":\"" << lines[i % 4] << "\",\"response\":\"" << lines[(i + 1) % 4] << "\"}\n";
    }
    const fs::path sessions = dir_ / "sessions";
    DatasetConfig dcfg;
    dcfg.session_dir = sessions.string();
    dcfg.dataset_kind = "chatbot";
    fs::create_directories(DatasetRegistry::local_pool_dir(dcfg));
    {
        DatasetRegistry reg(dcfg);
        reg.load_registry();
        reg.load_pending_list();
        ASSERT_TRUE(reg.add_file(data.string()));
        ASSERT_TRUE(reg.save_pending_list());
    }
    const auto config_with_vocab = [&](const fs::path& vocab) {
        return write_config("config.trainer.conf",
                            "LOG_FILE_PATH=" + (dir_ / "log.txt").string() + "\nSESSION_DIR=" + sessions.string() +
                                "\nVOCAB_PATH=" + vocab.string() + "\nMODEL_PATH=" + (dir_ / "model.bin").string() +
                                "\nREGISTRY_SERVER_URL=\nNAME_SERVICE_URL=\nMETRICS_SERVER_URL=\nGPU_ENABLED=false\n"
                                "D_MODEL=64\nNUM_HEADS=2\nD_FF=128\nNUM_ENCODER_LAYERS=1\nNUM_DECODER_LAYERS=1\n"
                                "MAX_SEQ_LENGTH=16\nVALIDATION_SPLIT=0\nVOCAB_BUILD_SIZE=60\n");
    };
    const auto run = [&](const fs::path& cfg, const std::vector<std::string>& command) {
        std::vector<std::string> args = {"--config", cfg.string(), "--foreground"};
        args.insert(args.end(), command.begin(), command.end());
        spawn(INCREMENTAL_TRAINER_BINARY_PATH, args);
        int status = 0;
        pid_t done = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (done != pid_) return -1;
        pid_ = -1;
        return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    };

    const fs::path vocab = dir_ / "vocab.txt";
    EXPECT_EQ(run(config_with_vocab(vocab), {"train", "1"}), 0);
    EXPECT_TRUE(fs::exists(vocab)) << "the first pass should have built the vocabulary";

    // A vocabulary path that can't be read (a directory) throws; it ends in exit 1, not an abort.
    const fs::path unreadable = dir_ / "vocab_dir";
    fs::create_directories(unreadable);
    EXPECT_EQ(run(config_with_vocab(unreadable), {"init"}), 1);
}

// TD-404: without MNS, the chatbot pass's run id was hostname prefix + PID, new every run (and able to
// collide across hosts), so files a crashed pass had claimed stayed claimed. It's now kept in
// SESSION_DIR/chatbot_run_id; the next pass reclaims them.
TEST_F(SighupTest, AChatbotPassReclaimsACrashedPassesFilesWithoutMns) {
    const fs::path data = dir_ / "pairs.jsonl";
    {
        std::ofstream out(data);
        const char* lines[] = {"hello world", "how are you", "good morning", "the cat sat"};
        for (int i = 0; i < 20; ++i)
            out << "{\"input\":\"" << lines[i % 4] << "\",\"response\":\"" << lines[(i + 1) % 4] << "\"}\n";
    }
    const fs::path sessions = dir_ / "sessions";
    fs::create_directories(sessions);
    std::ofstream(sessions / "chatbot_run_id") << "chatbot-crashed-pass\n";
    DatasetConfig dcfg;
    dcfg.session_dir = sessions.string();
    dcfg.dataset_kind = "chatbot";
    fs::create_directories(DatasetRegistry::local_pool_dir(dcfg));
    {
        DatasetRegistry reg(dcfg);
        reg.load_registry();
        reg.load_pending_list();
        ASSERT_TRUE(reg.add_file(data.string()));
        ASSERT_TRUE(reg.save_pending_list());
        ASSERT_EQ(reg.acquire_pending("chatbot-crashed-pass").files.size(), 1u);  // then it "crashed"
    }
    const auto cfg = write_config(
        "config.trainer.conf",
        "LOG_FILE_PATH=" + (dir_ / "log.txt").string() + "\nSESSION_DIR=" + sessions.string() +
            "\nVOCAB_PATH=" + (dir_ / "vocab.txt").string() + "\nMODEL_PATH=" + (dir_ / "model.bin").string() +
            "\nREGISTRY_SERVER_URL=\nNAME_SERVICE_URL=\nMETRICS_SERVER_URL=\nRUN_ID=\nGPU_ENABLED=false\n"
            "D_MODEL=64\nNUM_HEADS=2\nD_FF=128\nNUM_ENCODER_LAYERS=1\nNUM_DECODER_LAYERS=1\nMAX_SEQ_LENGTH=16\n"
            "VALIDATION_SPLIT=0\nVOCAB_BUILD_SIZE=60\nNUM_EPOCHS=1\n");
    spawn(INCREMENTAL_TRAINER_BINARY_PATH, {"--config", cfg.string(), "--foreground", "resume"});
    int status = 0;
    pid_t done = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while ((done = ::waitpid(pid_, &status, WNOHANG)) == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_EQ(done, pid_);
    pid_ = -1;
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
        << "exit " << (WIFEXITED(status) ? WEXITSTATUS(status) : -1) << ": the crashed pass's file wasn't reclaimed";
    DatasetRegistry reg(dcfg);
    reg.load_registry();
    EXPECT_EQ(reg.trained_files().size(), 1u);
}
