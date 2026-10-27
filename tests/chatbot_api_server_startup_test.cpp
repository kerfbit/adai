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
#include <iterator>
#include <string>
#include <thread>
#include <vector>
#include "../src/BPETokenizer.hpp"
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
            << "D_MODEL=32\nNUM_HEADS=2\nD_FF=64\nNUM_ENCODER_LAYERS=1\nNUM_DECODER_LAYERS=1\n"
            << "MAX_SEQ_LENGTH=32\nGPU_ENABLED=false\n"
            << extra;
        return path.string();
    }

    // fork + execve with a minimal environment, so a developer's own MODEL_PATH/MNS variables
    // (which override the config file) can't leak into the server.
    pid_t spawn(const std::string& config_path) {
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
            std::string b = binary_, c = "--config", cp = config_path;
            char* argv[] = {b.data(), c.data(), cp.data(), nullptr};
            ::execve(b.c_str(), argv, env.data());
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
    const pid_t pid = spawn(write_config("ALLOW_UNTRAINED_MODEL=true\nMAX_GEN_LENGTH=3\n"));
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
    stop(pid);

    ASSERT_TRUE(malformed);
    EXPECT_EQ(malformed->status, 400) << malformed->body;
    ASSERT_TRUE(mistyped);
    EXPECT_EQ(mistyped->status, 400) << mistyped->body;
    EXPECT_NE(mistyped->body.find("'session_id' must be a string"), std::string::npos);
    ASSERT_TRUE(brackets);
    EXPECT_EQ(brackets->status, 200);
    EXPECT_NE(brackets->body.find("\"success\":true"), std::string::npos) << brackets->body;
}
