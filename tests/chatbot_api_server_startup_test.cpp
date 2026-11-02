// TD-246: chatbot_api_server refuses to serve an untrained model.
//
// Spawns the real chatbot_api_server binary (as registry_ftp_confinement_test does for
// registry_server) with a config file this test writes. Without a loadable MODEL_PATH it must exit
// 1; with ALLOW_UNTRAINED_MODEL=true it starts, and /health reports "untrained" with HTTP 503 where
// it used to say "ok" about random weights.

#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>
#include "../src/BPETokenizer.hpp"
#include "../src/HippocampalMemory.hpp"
#include "../src/LeJEPAEncoder.hpp"
#include "httplib.h"

namespace fs = std::filesystem;

namespace {

std::string server_binary() {
    if (const char* p = std::getenv("ADAI_CHATBOT_API_SERVER_BINARY"); p && *p) {
        return p;
    }
#ifdef CHATBOT_API_SERVER_BINARY_PATH
    return CHATBOT_API_SERVER_BINARY_PATH;
#else
    return "";
#endif
}

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

class ChatbotApiServerStartupTest : public ::testing::Test {
   protected:
    fs::path dir_;
    std::string binary_;
    int port_ = 0;

    void SetUp() override {
        binary_ = server_binary();
        if (binary_.empty() || !fs::exists(binary_)) {
            GTEST_SKIP() << "chatbot_api_server binary not available ('" << binary_ << "')";
        }
        dir_ = fs::temp_directory_path() / ("adai_td246_" + std::to_string(::getpid()));
        fs::remove_all(dir_);
        fs::create_directories(dir_);
        BPETokenizer tokenizer;
        tokenizer.build_vocab({"hello world", "how are you", "good morning"}, 60);
        tokenizer.save_vocab((dir_ / "vocab.txt").string());
        port_ = free_port();
    }

    void TearDown() override {
        if (!dir_.empty()) {
            fs::remove_all(dir_);
        }
    }

    std::string write_config(const std::string& extra) {
        const fs::path path = dir_ / "config.chatbot.conf";
        std::ofstream out(path);
        out << "PORT=" << port_ << "\n"
            << "VOCAB_PATH=" << (dir_ / "vocab.txt").string() << "\n"
            << "LOG_FILE_PATH=" << (dir_ / "server.log").string() << "\n"
            << "D_MODEL=64\nNUM_HEADS=2\nD_FF=128\nNUM_ENCODER_LAYERS=1\nNUM_DECODER_LAYERS=1\n"
            << "MAX_SEQ_LENGTH=32\nGPU_ENABLED=false\n"
            << extra;
        return path.string();
    }

    // fork + execve with a minimal environment, so a developer's own MODEL_PATH/MNS variables
    // (which override the config file) can't leak into the server.
    pid_t spawn(const std::string& config_path, const std::vector<std::string>& extra_args = {}) {
        pid_t pid = ::fork();
        if (pid == 0) {
            const std::string log = (dir_ / "stdout.log").string();
            FILE* f = std::freopen(log.c_str(), "w", stdout);
            (void)f;
            ::dup2(::fileno(stdout), STDERR_FILENO);
            std::vector<std::string> env_strings = {"PATH=/usr/bin:/bin",
                                                    "HOME=" + dir_.string()};
            std::vector<char*> env;
            for (auto& e : env_strings) env.push_back(e.data());
            env.push_back(nullptr);
            std::vector<std::string> args = {binary_, "--config", config_path};
            args.insert(args.end(), extra_args.begin(), extra_args.end());
            std::vector<char*> argv;
            for (auto& a : args) argv.push_back(a.data());
            argv.push_back(nullptr);
            ::execve(args[0].c_str(), argv.data(), env.data());
            ::_exit(127);
        }
        return pid;
    }

    // Exit code, or -1 if still running after `timeout`.
    static int wait_exit(pid_t pid, std::chrono::seconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            int status = 0;
            if (::waitpid(pid, &status, WNOHANG) == pid) {
                return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return -1;
    }

    static void stop(pid_t pid) {
        ::kill(pid, SIGTERM);
        if (wait_exit(pid, std::chrono::seconds(15)) < 0) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
        }
    }
};

}  // namespace

TEST_F(ChatbotApiServerStartupTest, MissingModelFileExitsOne) {
    const pid_t pid = spawn(write_config("MODEL_PATH=" + (dir_ / "missing.bin").string() + "\n"));
    const int code = wait_exit(pid, std::chrono::seconds(60));
    if (code < 0) {
        stop(pid);
    }
    EXPECT_EQ(code, 1) << "it used to keep serving random weights";
}

TEST_F(ChatbotApiServerStartupTest, NoModelPathExitsOne) {
    const pid_t pid = spawn(write_config(""));
    const int code = wait_exit(pid, std::chrono::seconds(60));
    if (code < 0) {
        stop(pid);
    }
    EXPECT_EQ(code, 1);
}

TEST_F(ChatbotApiServerStartupTest, OptInServesButHealthSaysUntrained) {
    const pid_t pid = spawn(write_config("ALLOW_UNTRAINED_MODEL=true\n"));
    httplib::Client client("127.0.0.1", port_);
    client.set_connection_timeout(1);
    httplib::Result res;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        res = client.Get("/health");
        if (res) break;
        ASSERT_LT(wait_exit(pid, std::chrono::seconds(0)), 0) << "the server exited";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    ASSERT_TRUE(res) << "the server never answered /health";
    EXPECT_EQ(res->status, 503);
    EXPECT_NE(res->body.find("\"status\":\"untrained\""), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("\"model_loaded\":false"), std::string::npos) << res->body;
    stop(pid);
}

// TD-247: RAG embeds with the trained model's encoder, so it stays off without a trained model.
TEST_F(ChatbotApiServerStartupTest, RagIsNotEnabledWithoutATrainedModel) {
    fs::create_directories(dir_ / "docs");
    std::ofstream(dir_ / "docs" / "a.txt") << "hello world";
    const pid_t pid = spawn(write_config("ALLOW_UNTRAINED_MODEL=true\nRAG_ENABLED=true\nRAG_DOCS_PATH=" +
                                         (dir_ / "docs").string() + "\n"));
    httplib::Client client("127.0.0.1", port_);
    client.set_connection_timeout(1);
    bool up = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!up && std::chrono::steady_clock::now() < deadline) {
        up = static_cast<bool>(client.Get("/health"));
        if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    stop(pid);
    ASSERT_TRUE(up);
    std::string logs;
    for (const char* name : {"stdout.log", "server.log"}) {
        std::ifstream in(dir_ / name);
        logs.append(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    EXPECT_NE(logs.find("[RAG] Not enabled: no trained model is loaded"), std::string::npos);
    EXPECT_EQ(logs.find("Initializing RAG engine"), std::string::npos);
}

// TD-226/TD-228 through the real routes: malformed bodies and mistyped fields get HTTP 400, and a batch
// whose messages contain ']' is answered (it used to be rejected as "Missing or empty 'messages'").
TEST_F(ChatbotApiServerStartupTest, ChatRoutesParseJsonAndRejectMalformedBodies) {
    const pid_t pid = spawn(write_config("ALLOW_UNTRAINED_MODEL=true\nMAX_GEN_LENGTH=3\nMAX_BATCH_SIZE=3\n"));
    httplib::Client client("127.0.0.1", port_);
    client.set_connection_timeout(1);
    client.set_read_timeout(60);
    bool up = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!up && std::chrono::steady_clock::now() < deadline) {
        up = static_cast<bool>(client.Get("/health"));
        if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    ASSERT_TRUE(up);

    auto malformed = client.Post("/chat/batch", R"({"messages":["a", )", "application/json");
    auto mistyped = client.Post("/chat/session", R"({"session_id":123,"message":"hi"})",
                                "application/json");
    auto brackets =
        client.Post("/chat/batch", R"({"messages":["see [1] here","second"]})", "application/json");
    // TD-232: ids are server-issued; one the server never issued is 404 on every session route.
    std::vector<std::pair<std::string, int>> unknown;
    for (const char* route : {"/chat/session", "/clear-session", "/chat/session/export",
                              "/chat/session/import"}) {
        auto r = client.Post(route, R"({"session_id":"client-made","message":"hi","data":"x"})",
                             "application/json");
        unknown.emplace_back(route, r ? r->status : -1);
    }
    // TD-236: validation errors used to go out with HTTP 200.
    std::vector<std::pair<std::string, int>> invalid;
    for (const auto& [route, body] : std::vector<std::pair<std::string, std::string>>{
             {"/chat", "{}"},
             {"/chat/session", R"({"message":""})"},
             {"/clear-session", "{}"},
             {"/chat/session/export", "{}"},
             {"/chat/session/import", R"({"data":"not a conversation"})"},
             {"/chat/batch", R"({"messages":[]})"}}) {
        auto r = client.Post(route, body, "application/json");
        invalid.emplace_back(route, r ? r->status : -1);
    }
    // TD-235: MAX_BATCH_SIZE=3 in this test's config.
    auto too_big = client.Post("/chat/batch", R"({"messages":["a","b","c","d"]})", "application/json");
    stop(pid);

    for (const auto& [route, status] : unknown) EXPECT_EQ(status, 404) << route;
    for (const auto& [route, status] : invalid) EXPECT_EQ(status, 400) << route;
    ASSERT_TRUE(too_big);
    EXPECT_EQ(too_big->status, 413) << too_big->body;

    ASSERT_TRUE(malformed);
    EXPECT_EQ(malformed->status, 400) << malformed->body;
    ASSERT_TRUE(mistyped);
    EXPECT_EQ(mistyped->status, 400) << mistyped->body;
    EXPECT_NE(mistyped->body.find("'session_id' must be a string"), std::string::npos);
    ASSERT_TRUE(brackets);
    EXPECT_EQ(brackets->status, 200);
    EXPECT_NE(brackets->body.find("\"success\":true"), std::string::npos) << brackets->body;
}

// TD-241: a config validate() rejects used to start (and was refused only on the next SIGHUP).
TEST_F(ChatbotApiServerStartupTest, AnInvalidConfigExitsOneAtStartup) {
    const pid_t pid = spawn(write_config("ALLOW_UNTRAINED_MODEL=true\nSESSION_TIMEOUT=-5\n"));
    const int code = wait_exit(pid, std::chrono::seconds(30));
    if (code < 0) {
        stop(pid);
    }
    EXPECT_EQ(code, 1) << "it used to start with every session already expired";
    std::stringstream out;
    out << std::ifstream(dir_ / "stdout.log").rdbuf();
    EXPECT_NE(out.str().find("Invalid session_timeout: -5"), std::string::npos) << out.str();
}

// TD-242: a SIGHUP reload re-read only the file and environment, so a server started with
// --temperature 0.3 went back to the file's temperature on the first reload.
TEST_F(ChatbotApiServerStartupTest, ACommandLineSettingSurvivesAReload) {
    const std::string config = write_config("ALLOW_UNTRAINED_MODEL=true\nTEMPERATURE=0.7\nTOP_P=0.9\n");
    const pid_t pid = spawn(config, {"--temperature", "0.3"});
    httplib::Client client("127.0.0.1", port_);
    client.set_connection_timeout(1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool up = false;
    while (!up && std::chrono::steady_clock::now() < deadline) {
        up = static_cast<bool>(client.Get("/health"));
        ASSERT_LT(wait_exit(pid, std::chrono::seconds(0)), 0) << "the server exited";
        if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    ASSERT_TRUE(up) << "the server never answered /health";

    // The file changes both; only TOP_P has no flag holding it.
    write_config("ALLOW_UNTRAINED_MODEL=true\nTEMPERATURE=0.9\nTOP_P=0.5\n");
    ASSERT_EQ(::kill(pid, SIGHUP), 0);
    std::string log;
    const auto reload_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (log.find("Generation configuration updated") == std::string::npos &&
           std::chrono::steady_clock::now() < reload_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::stringstream s;
        s << std::ifstream(dir_ / "server.log").rdbuf();
        log = s.str();
    }
    stop(pid);
    EXPECT_NE(log.find("temperature=0.3 top_p=0.5"), std::string::npos) << log;
    EXPECT_NE(log.find("TEMPERATURE changed in the file but is held by a command-line flag"), std::string::npos)
        << log;
}

// TD-244: TOP_K and BEAM_WIDTH were parsed but never copied into ChatbotAPI's generation settings, at
// startup or on reload, so it always used 50 and 4.
TEST_F(ChatbotApiServerStartupTest, TopKAndBeamWidthReachTheServer) {
    const std::string config = write_config("ALLOW_UNTRAINED_MODEL=true\nTOP_K=7\nBEAM_WIDTH=3\n");
    const pid_t pid = spawn(config, {"--top-k", "9"});
    auto read_log = [&] {
        std::stringstream s;
        s << std::ifstream(dir_ / "server.log").rdbuf();
        return s.str();
    };
    std::string log;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (log.find("Generation configuration:") == std::string::npos && std::chrono::steady_clock::now() < deadline) {
        ASSERT_LT(wait_exit(pid, std::chrono::seconds(0)), 0) << "the server exited";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        log = read_log();
    }
    EXPECT_NE(log.find("top_k=9 strategy=nucleus beam_width=3"), std::string::npos) << log;  // flag beats file

    // A reload applies a changed BEAM_WIDTH; TOP_K stays held by --top-k.
    write_config("ALLOW_UNTRAINED_MODEL=true\nTOP_K=11\nBEAM_WIDTH=5\n");
    ASSERT_EQ(::kill(pid, SIGHUP), 0);
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (log.find("Generation configuration updated") == std::string::npos &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        log = read_log();
    }
    stop(pid);
    const auto updated = log.find("Generation configuration updated");
    ASSERT_NE(updated, std::string::npos) << log;
    EXPECT_NE(log.find("top_k=9 strategy=nucleus beam_width=5", updated), std::string::npos) << log;
}

// TD-248: SIGTERM while the server thread is starting used to be lost, because httplib ignores
// stop() until it's listening; the process then hung until SIGKILL. The server's output is read
// through a pipe so the signal lands right after "[4/4] Starting API server", the first line after
// the signal handlers are installed and a few lines before the server thread starts.
TEST_F(ChatbotApiServerStartupTest, SigtermDuringStartupExitsPromptly) {
    const std::string config = write_config("ALLOW_UNTRAINED_MODEL=true\n");
    int hung = 0;
    for (int trial = 0; trial < 10; ++trial) {
        int fds[2];
        ASSERT_EQ(::pipe(fds), 0);
        const pid_t pid = ::fork();
        if (pid == 0) {
            ::dup2(fds[1], STDOUT_FILENO);
            ::dup2(fds[1], STDERR_FILENO);
            ::close(fds[0]);
            ::close(fds[1]);
            std::string path = "PATH=/usr/bin:/bin", home = "HOME=" + dir_.string();
            char* env[] = {path.data(), home.data(), nullptr};
            std::string flag = "--config", conf = config;
            char* argv[] = {binary_.data(), flag.data(), conf.data(), nullptr};
            ::execve(binary_.c_str(), argv, env);
            ::_exit(127);
        }
        ::close(fds[1]);
        std::string seen;
        char buf[4096];
        while (seen.find("[4/4] Starting API server") == std::string::npos) {
            const ssize_t n = ::read(fds[0], buf, sizeof(buf));
            if (n <= 0) break;
            seen.append(buf, static_cast<size_t>(n));
        }
        ::kill(pid, SIGTERM);
        // Keep draining so a full pipe can't block the server's logging.
        std::thread drain([fd = fds[0]] {
            char sink[4096];
            while (::read(fd, sink, sizeof(sink)) > 0) {
            }
        });
        const int code = wait_exit(pid, std::chrono::seconds(10));
        if (code < 0) {
            ++hung;
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
        } else {
            EXPECT_EQ(code, 0) << "trial " << trial;
        }
        drain.join();
        ::close(fds[0]);
        ASSERT_NE(seen.find("[4/4] Starting API server"), std::string::npos) << seen;
    }
    EXPECT_EQ(hung, 0) << hung << " of 10 trials were still running 10 s after SIGTERM";
}

// TD-250: hippocampal memory was saved only at a graceful shutdown, so a crash lost everything the
// server had learned since it started. With a 1 s save interval, a SIGKILL after a request keeps it.
TEST_F(ChatbotApiServerStartupTest, HippocampalMemorySurvivesACrashAfterAPeriodicSave) {
    BPETokenizer tokenizer;
    tokenizer.load_vocab((dir_ / "vocab.txt").string());
    const fs::path world_model_dir = dir_ / "world_model";
    LeJEPAEncoder(static_cast<int>(tokenizer.get_vocab_size()), 64, 1, 2, 128, 32, 64).save(world_model_dir.string());
    const fs::path state = dir_ / "memory" / "hm";
    fs::create_directories(state.parent_path());
    const pid_t pid = spawn(write_config(
        "ALLOW_UNTRAINED_MODEL=true\nMAX_GEN_LENGTH=3\nWORLD_MODEL_ENABLED=true\nWORLD_MODEL_ARTIFACT_PATH=" +
        world_model_dir.string() +
        "\nWORLD_MODEL_D_MODEL=64\nWORLD_MODEL_NUM_LAYERS=1\nWORLD_MODEL_NUM_HEADS=2\nWORLD_MODEL_D_FF=128\n"
        "WORLD_MODEL_INJECT_EVERY_N_LAYERS=1\nHIPPOCAMPAL_MEMORY_ENABLED=true\nHIPPOCAMPAL_STATE_PATH=" +
        state.string() + "\nHIPPOCAMPAL_SAVE_INTERVAL_SECONDS=1\n"));
    httplib::Client client("127.0.0.1", port_);
    client.set_connection_timeout(std::chrono::milliseconds(200));
    client.set_read_timeout(std::chrono::seconds(30));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!client.Get("/health") && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto reply = client.Post("/chat", R"({"message":"hello world"})", "application/json");
    ASSERT_TRUE(reply);
    ASSERT_EQ(reply->status, 200) << reply->body;

    const auto save_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!fs::exists(state) && std::chrono::steady_clock::now() < save_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ::kill(pid, SIGKILL);  // a crash: no shutdown save
    ::waitpid(pid, nullptr, 0);

    std::ifstream log_in(dir_ / "stdout.log");
    const std::string log((std::istreambuf_iterator<char>(log_in)), std::istreambuf_iterator<char>());
    ASSERT_NE(log.find("Hippocampal memory attached"), std::string::npos) << log;
    ASSERT_TRUE(fs::exists(state)) << "no periodic save within 10 s:\n" << log;
    EXPECT_TRUE(fs::exists(state.string() + ".assoc"));
    HippocampalMemory restored(64, 512);
    restored.load(state.string());
    EXPECT_GE(restored.size(), 1) << "the request's episode should be in the saved memory";
}

// TD-250: with neither HIPPOCAMPAL_STATE_PATH nor MODEL_PATH, the memory used to be written to hidden
// files in the working directory; it isn't attached now.
TEST_F(ChatbotApiServerStartupTest, HippocampalMemoryIsNotAttachedWithoutAPath) {
    BPETokenizer tokenizer;
    tokenizer.load_vocab((dir_ / "vocab.txt").string());
    const fs::path world_model_dir = dir_ / "world_model";
    LeJEPAEncoder(static_cast<int>(tokenizer.get_vocab_size()), 64, 1, 2, 128, 32, 64).save(world_model_dir.string());
    const pid_t pid = spawn(write_config(
        "ALLOW_UNTRAINED_MODEL=true\nWORLD_MODEL_ENABLED=true\nWORLD_MODEL_ARTIFACT_PATH=" + world_model_dir.string() +
        "\nWORLD_MODEL_D_MODEL=64\nWORLD_MODEL_NUM_LAYERS=1\nWORLD_MODEL_NUM_HEADS=2\nWORLD_MODEL_D_FF=128\n"
        "WORLD_MODEL_INJECT_EVERY_N_LAYERS=1\nHIPPOCAMPAL_MEMORY_ENABLED=true\n"));
    const fs::path stray = fs::current_path() / ".hippocampal";  // the server runs in our directory
    const bool stray_before = fs::exists(stray);
    httplib::Client client("127.0.0.1", port_);
    client.set_connection_timeout(std::chrono::milliseconds(200));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!client.Get("/health") && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop(pid);
    std::ifstream log_in(dir_ / "stdout.log");
    const std::string log((std::istreambuf_iterator<char>(log_in)), std::istreambuf_iterator<char>());
    EXPECT_NE(log.find("neither HIPPOCAMPAL_STATE_PATH nor MODEL_PATH"), std::string::npos) << log;
    EXPECT_EQ(log.find("Hippocampal memory attached"), std::string::npos) << log;
    if (!stray_before) {
        EXPECT_FALSE(fs::exists(stray)) << "nothing in the working directory";
        fs::remove(stray);
    }
}


// TD-251: a reload applied only the log level and generation settings. MAX_BATCH_SIZE (like
// SESSION_TIMEOUT, MAX_SESSIONS, the RAG retrieval settings and the save interval) waited for a
// restart, and the reloaded config took the file's MODEL_PATH while the old model kept serving
// (MNS-resolved paths and architecture were lost the same way).
TEST_F(ChatbotApiServerStartupTest, AReloadAppliesLiveSettingsAndKeepsTheLoadedModel) {
    const std::string model_a = (dir_ / "model_a").string();
    const std::string model_b = (dir_ / "model_b").string();
    const std::string config =
        write_config("ALLOW_UNTRAINED_MODEL=true\nMAX_GEN_LENGTH=2\nMAX_BATCH_SIZE=4\nMODEL_PATH=" + model_a + "\n");
    const pid_t pid = spawn(config);
    httplib::Client client("127.0.0.1", port_);
    client.set_connection_timeout(std::chrono::milliseconds(200));
    client.set_read_timeout(std::chrono::seconds(60));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool up = false;
    while (!up && std::chrono::steady_clock::now() < deadline) {
        up = static_cast<bool>(client.Get("/health"));
        ASSERT_LT(wait_exit(pid, std::chrono::seconds(0)), 0) << "the server exited";
        if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_TRUE(up) << "the server never answered /health";
    const std::string three = R"({"messages":["hello","world","good"]})";
    auto before = client.Post("/chat/batch", three, "application/json");
    ASSERT_TRUE(before);
    EXPECT_EQ(before->status, 200) << before->body;

    write_config("ALLOW_UNTRAINED_MODEL=true\nMAX_GEN_LENGTH=2\nMAX_BATCH_SIZE=2\nMODEL_PATH=" + model_b + "\n");
    ASSERT_EQ(::kill(pid, SIGHUP), 0);
    auto read_log = [&] {
        std::stringstream s;
        s << std::ifstream(dir_ / "server.log").rdbuf();
        return s.str();
    };
    std::string log;
    const auto reload_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (log.find("Generation configuration updated") == std::string::npos &&
           std::chrono::steady_clock::now() < reload_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        log = read_log();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // the settings after it
    auto after = client.Post("/chat/batch", three, "application/json");
    ASSERT_TRUE(after);
    EXPECT_EQ(after->status, 413) << "MAX_BATCH_SIZE=2 should apply without a restart: " << after->body;
    stop(pid);
    log = read_log();
    EXPECT_NE(log.find("MODEL_PATH changed but takes effect only after a restart"), std::string::npos) << log;
    EXPECT_EQ(log.find("MAX_BATCH_SIZE changed but takes effect only after a restart"), std::string::npos) << log;
    EXPECT_NE(log.find("Model state: " + model_a + " (read-only)"), std::string::npos)
        << "the running model's path should survive the reload:\n" << log;
    EXPECT_EQ(log.find("Model state: " + model_b), std::string::npos) << log;
}

// TD-252: the startup banner was a hand-written list of 4 of the 9 routes (no batch, export or import
// routes, and /admin/profile only with --profile). It's now built from ChatbotAPI::routes().
TEST_F(ChatbotApiServerStartupTest, TheStartupBannerListsEveryServedRoute) {
    const pid_t pid = spawn(write_config("ALLOW_UNTRAINED_MODEL=true\nMAX_GEN_LENGTH=2\n"));
    httplib::Client client("127.0.0.1", port_);
    client.set_connection_timeout(std::chrono::milliseconds(200));
    client.set_read_timeout(std::chrono::seconds(30));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool up = false;
    while (!up && std::chrono::steady_clock::now() < deadline) {
        up = static_cast<bool>(client.Get("/health"));
        ASSERT_LT(wait_exit(pid, std::chrono::seconds(0)), 0) << "the server exited";
        if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_TRUE(up) << "the server never answered /health";
    std::stringstream s;
    s << std::ifstream(dir_ / "server.log").rdbuf();
    const std::string log = s.str();
    const std::vector<std::pair<std::string, std::string>> routes = {
        {"POST", "/chat"},          {"POST", "/chat/session"},        {"POST", "/clear-session"},
        {"POST", "/chat/session/export"}, {"POST", "/chat/session/import"}, {"GET", "/health"},
        {"GET", "/admin/profile"},  {"POST", "/chat/batch"},          {"POST", "/chat/batch-session"}};
    for (const auto& [method, path] : routes) {
        const std::string padded = path + std::string(path.size() < 22 ? 22 - path.size() : 0, ' ') + " - ";
        EXPECT_NE(log.find(method + std::string(6 - method.size() + 1, ' ') + padded), std::string::npos)
            << method << " " << path << " missing from the banner:\n" << log;
        // And the server really serves it (an unregistered path is 404 with an empty body).
        auto reply = method == "GET" ? client.Get(path) : client.Post(path, "{}", "application/json");
        ASSERT_TRUE(reply) << path;
        EXPECT_FALSE(reply->status == 404 && reply->body.empty()) << path << " is not served";
    }
    EXPECT_NE(log.find("/admin/profile         - generate_response() timing stats (enabled by --profile) [not enabled]"),
              std::string::npos)
        << log;
    stop(pid);
}
