// TD-411: `incremental_trainer retrain` against a remote registry_server. A full retrain used to
// read the already-trained files by their registry paths, which only exist on the registry host,
// and downloaded only the pending ones; it now returns the trained units to the pending queue and
// claims the whole pool, so every file comes over FTP like a `train` pass's. Spawns the real
// registry_server (FTP on, logging to a file) and incremental_trainer, and counts the FTP server's
// completed downloads of the file: one for `train`, a second for `retrain`.

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"

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

// Starts `binary` in `cwd` with output to `log`, without the service URLs of the environment.
pid_t spawn(const std::string& binary, const std::vector<std::string>& args, const fs::path& cwd,
            const fs::path& log) {
    const pid_t pid = ::fork();
    if (pid == 0) {
        if (::chdir(cwd.c_str()) != 0)
            _exit(126);
        for (const char* var : {"REGISTRY_SERVER_URL", "NAME_SERVICE_URL", "METRICS_SERVER_URL"})
            ::unsetenv(var);
        const int fd = ::open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
            ::dup2(fd, STDOUT_FILENO);
            ::dup2(fd, STDERR_FILENO);
        }
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(binary.c_str()));
        for (const auto& a : args)
            argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        ::execv(binary.c_str(), argv.data());
        _exit(127);
    }
    return pid;
}

// The child's exit code, or -1 if it didn't exit within `limit` (it's then killed).
int wait_exit(pid_t pid, std::chrono::seconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    int status = 0;
    pid_t done = 0;
    while ((done = ::waitpid(pid, &status, WNOHANG)) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (done != pid) {
        ::kill(pid, SIGKILL);
        ::waitpid(pid, nullptr, 0);
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

int count(const std::string& text, const std::string& needle) {
    int n = 0;
    for (std::size_t at = text.find(needle); at != std::string::npos;
         at = text.find(needle, at + 1))
        ++n;
    return n;
}

class RetrainRemoteTest : public ::testing::Test {
   protected:
    fs::path dir_;
    pid_t server_ = -1;
    int http_port_ = 0;

    void SetUp() override {
        dir_ = fs::temp_directory_path() / ("adai_td411_" + std::to_string(::getpid()));
        fs::remove_all(dir_);
        fs::create_directories(dir_ / "registry" / "g" / "chatbot");
        fs::create_directories(dir_ / "trainer");
    }

    void TearDown() override {
        if (server_ > 0) {
            ::kill(server_, SIGTERM);
            if (wait_exit(server_, std::chrono::seconds(20)) < 0)
                server_ = -1;
        }
        fs::remove_all(dir_);
    }

    bool healthy() {
        httplib::Client client("127.0.0.1", http_port_);
        client.set_connection_timeout(std::chrono::milliseconds(300));
        auto res = client.Get("/health");
        return res && res->status == 200;
    }
};

}  // namespace

TEST_F(RetrainRemoteTest, RetrainDownloadsTheAlreadyTrainedFilesFromTheRegistry) {
    http_port_ = free_port();
    int ftp_port = free_port();
    while (ftp_port == http_port_)
        ftp_port = free_port();
    const fs::path server_log = dir_ / "server.log";
    server_ =
        spawn(REGISTRY_SERVER_BINARY_PATH,
              {"--port", std::to_string(http_port_), "--data-dir", (dir_ / "registry").string(),
               "--ftp-enabled", "--ftp-port", std::to_string(ftp_port), "--ftp-ip", "127.0.0.1"},
              dir_, server_log);
    ASSERT_GT(server_, 0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!healthy() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_TRUE(healthy()) << read_file(server_log);

    // One data file, inside the registry's data directory, queued in the chatbot pool.
    const fs::path data = dir_ / "registry" / "g" / "chatbot" / "pairs.jsonl";
    {
        std::ofstream out(data);
        for (int i = 0; i < 12; ++i)
            out << "{\"input\":\"hello number " << i << "\",\"response\":\"hi there " << i
                << "\"}\n";
    }
    {
        httplib::Client cli("127.0.0.1", http_port_);
        auto res = cli.Post("/registry/g/chatbot/pending/add",
                            "{\"path\":\"" + data.string() + "\"}", "application/json");
        ASSERT_TRUE(res);
        ASSERT_EQ(res->status, 200) << res->body;
    }

    const fs::path t = dir_ / "trainer";
    const fs::path cfg = t / "config.trainer.conf";
    std::ofstream(cfg)
        << "LOG_FILE_PATH=" << (t / "trainer.log").string()
        << "\nSESSION_DIR=" << (t / "sessions").string()
        << "\nDOWNLOAD_DIR=" << (t / "downloads").string()
        << "\nVOCAB_PATH=" << (t / "vocab.txt").string()
        << "\nMODEL_PATH=" << (t / "model.bin").string()
        << "\nREGISTRY_SERVER_URL=http://127.0.0.1:" << http_port_
        << "\nRUN_GROUP=g\nNAME_SERVICE_URL=\nMETRICS_SERVER_URL=\nGPU_ENABLED=false"
           "\nD_MODEL=64\nNUM_HEADS=2\nD_FF=128\nNUM_ENCODER_LAYERS=1\nNUM_DECODER_LAYERS=1"
           "\nMAX_SEQ_LENGTH=16\nVOCAB_BUILD_SIZE=60\nSEED=7\n";
    const fs::path trainer_out = t / "out.txt";
    const std::string retr = "[AUDIT] RETR complete";

    const pid_t train =
        spawn(INCREMENTAL_TRAINER_BINARY_PATH,
              {"--config", cfg.string(), "--foreground", "train", "1"}, t, trainer_out);
    ASSERT_EQ(wait_exit(train, std::chrono::seconds(180)), 0) << read_file(trainer_out);
    ASSERT_EQ(count(read_file(server_log), retr), 1) << "the first pass downloads the file once";

    const pid_t retrain =
        spawn(INCREMENTAL_TRAINER_BINARY_PATH,
              {"--config", cfg.string(), "--foreground", "retrain", "1"}, t, trainer_out);
    ASSERT_EQ(wait_exit(retrain, std::chrono::seconds(180)), 0) << read_file(trainer_out);
    EXPECT_EQ(count(read_file(server_log), retr), 2)
        << "retrain read the trained file by its registry path instead of downloading it";

    // Trained again, and nothing left claimed or pending.
    httplib::Client cli("127.0.0.1", http_port_);
    auto reg = cli.Get("/registry/g/chatbot/registry");
    ASSERT_TRUE(reg);
    EXPECT_NE(reg->body.find("pairs.jsonl"), std::string::npos) << reg->body;
    auto queue = cli.Get("/registry/g/chatbot/queue");
    ASSERT_TRUE(queue);
    EXPECT_EQ(queue->body.find("pairs.jsonl"), std::string::npos) << queue->body;
}
