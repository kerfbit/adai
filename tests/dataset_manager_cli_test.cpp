/**
 * @file dataset_manager_cli_test.cpp
 * @brief TD-331: runs the real `dataset_manager` binary, in local mode against a scratch SESSION_DIR
 * and in remote mode against a `registry_server` it starts, and checks the queues it leaves:
 * `segment`, `assign --count`, `delete`, `remove` and `migrate` with segments. TD-326 and TD-327 lived
 * in exactly this composition of DatasetRegistry calls, which the parser and DatasetRegistry tests
 * don't reach.
 *
 * The server is this test's own child, stopped by its recorded pid; nothing else is signalled.
 */
#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>

#include "../src/DatasetRegistry.hpp"
#include "../src/RegistryTransport.hpp"

namespace fs = std::filesystem;

namespace {

int free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    ::close(fd);
    return ntohs(addr.sin_port);
}

// Runs dataset_manager with @p args; returns its exit status (-1 if it didn't exit normally) and
// its stdout and stderr together in @p output.
int run_dm(const std::vector<std::string>& args, std::string& output) {
    int pipe_fds[2];
    if (::pipe(pipe_fds) != 0)
        return -1;
    std::vector<std::string> argv_store = {DATASET_MANAGER_BINARY_PATH};
    argv_store.insert(argv_store.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& a : argv_store)
        argv.push_back(a.data());
    argv.push_back(nullptr);
    const pid_t pid = ::fork();
    if (pid < 0)
        return -1;
    if (pid == 0) {
        ::dup2(pipe_fds[1], STDOUT_FILENO);
        ::dup2(pipe_fds[1], STDERR_FILENO);
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        ::execv(argv[0], argv.data());
        ::_exit(127);
    }
    ::close(pipe_fds[1]);
    output.clear();
    char buf[4096];
    long n;
    while ((n = ::read(pipe_fds[0], buf, sizeof(buf))) > 0)
        output.append(buf, static_cast<size_t>(n));
    ::close(pipe_fds[0]);
    int status = 0;
    if (::waitpid(pid, &status, 0) != pid || !WIFEXITED(status))
        return -1;
    return WEXITSTATUS(status);
}

void write_pairs(const fs::path& path, int n) {
    std::ofstream out(path);
    for (int i = 0; i < n; ++i)
        out << R"({"input":"question )" << i << R"(","response":"answer )" << i << "\"}\n";
}

bool is_segment(const PendingEntry& e, int start, int count) {
    return e.segment_start == start && e.segment_count == count;
}

class DatasetManagerCliTest : public ::testing::Test {
   protected:
    void SetUp() override {
        if (!fs::exists(DATASET_MANAGER_BINARY_PATH))
            GTEST_SKIP() << "dataset_manager not built";
        root_ = fs::temp_directory_path() /
                ("adai_dm_cli_" + std::to_string(::getpid()) + "_" +
                 ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::remove_all(root_);
        fs::create_directories(root_ / "sessions");
        config_ = root_ / "config.registry.conf";
    }

    void TearDown() override {
        if (server_pid_ > 0) {  // only this test's own server
            ::kill(server_pid_, SIGTERM);
            int status = 0;
            ::waitpid(server_pid_, &status, 0);
            server_pid_ = -1;
        }
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    void write_config(const std::string& registry_url) {
        std::ofstream(config_) << "REGISTRY_SERVER_URL=" << registry_url
                               << "\nRUN_GROUP=g1\nNAME_SERVICE_URL=\nMETRICS_SERVER_URL=\nSESSION_DIR="
                               << (root_ / "sessions").string() << "\nLOG_FILE_PATH="
                               << (root_ / "dm.log").string() << "\n";
    }

    // Runs `dataset_manager --config <config> [extra...]`, expecting @p expected_exit.
    void dm(const std::vector<std::string>& args, int expected_exit = 0) {
        std::vector<std::string> full = {"--config", config_.string()};
        full.insert(full.end(), args.begin(), args.end());
        std::string out;
        const int rc = run_dm(full, out);
        std::string joined;
        for (const auto& a : args)
            joined += a + " ";
        ASSERT_EQ(rc, expected_exit) << "dataset_manager " << joined << "\n" << out;
    }

    // The local pool for @p kind ("" = the legacy pool).
    std::vector<PendingEntry> local_pending(const std::string& kind = "") {
        DatasetConfig cfg;
        cfg.session_dir = (root_ / "sessions").string();
        cfg.dataset_kind = kind;
        DatasetRegistry reg(cfg);
        reg.load_pending_list();
        return reg.pending_entries();
    }

    // Starts registry_server on a free port with its data under root_/data.
    bool start_server() {
        const fs::path data = root_ / "data";
        fs::create_directories(data);
        port_ = free_port();
        server_pid_ = ::fork();
        if (server_pid_ == 0) {
            const std::string port = std::to_string(port_);
            const std::string data_dir = data.string();
            char* argv[] = {const_cast<char*>(REGISTRY_SERVER_BINARY_PATH), const_cast<char*>("--port"),
                            const_cast<char*>(port.c_str()), const_cast<char*>("--data-dir"),
                            const_cast<char*>(data_dir.c_str()), const_cast<char*>("--config"),
                            const_cast<char*>("/dev/null"), nullptr};
            const int devnull = ::open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                ::dup2(devnull, STDOUT_FILENO);
                ::dup2(devnull, STDERR_FILENO);
            }
            ::execv(argv[0], argv);
            ::_exit(127);
        }
        if (server_pid_ < 0)
            return false;
        httplib::Client probe("127.0.0.1", port_);
        for (int i = 0; i < 100; ++i) {
            if (auto res = probe.Get("/health"); res && res->status == 200)
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return false;
    }

    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }

    std::vector<PendingEntry> remote_pending(const std::string& kind = "") {
        RemoteTransport t(url(), "g1", 5000, kind);
        std::vector<PendingEntry> entries;
        EXPECT_TRUE(t.load_pending(entries));
        return entries;
    }

    fs::path root_;
    fs::path config_;
    pid_t server_pid_ = -1;
    int port_ = 0;
};

}  // namespace

// ============================================================================
// Local mode (no registry server)
// ============================================================================

// segment replaces an unclaimed whole-file entry and carries its model onto every segment (TD-542).
TEST_F(DatasetManagerCliTest, LocalSegmentReplacesTheWholeEntryAndKeepsItsModel) {
    write_config("");
    const fs::path data = root_ / "data.jsonl";
    write_pairs(data, 10);
    dm({"add", data.string()});
    dm({"assign", "m1"});
    dm({"segment", data.string(), "--count", "3"});

    const auto pending = local_pending();
    ASSERT_EQ(pending.size(), 3u);
    EXPECT_TRUE(is_segment(pending[0], 0, 4));
    EXPECT_TRUE(is_segment(pending[1], 4, 3));
    EXPECT_TRUE(is_segment(pending[2], 7, 3));
    for (const auto& e : pending) {
        EXPECT_EQ(e.path, fs::absolute(data).string());
        EXPECT_EQ(e.model_name, "m1");
    }
    dm({"segment", data.string(), "--ranges", "0-10"}, 1);  // past the last pair: nothing queued
    EXPECT_EQ(local_pending().size(), 3u);
}

TEST_F(DatasetManagerCliTest, LocalAssignCountDeleteAndRemove) {
    write_config("");
    std::vector<std::string> files;
    for (const char* name : {"a.jsonl", "b.jsonl", "c.jsonl", "d.jsonl"}) {
        write_pairs(root_ / name, 2);
        files.push_back(fs::absolute(root_ / name).string());
        dm({"add", files.back()});
    }

    dm({"assign", "m1", "--count", "2"});
    auto pending = local_pending();
    ASSERT_EQ(pending.size(), 4u);
    EXPECT_EQ(std::count_if(pending.begin(), pending.end(),
                            [](const PendingEntry& e) { return e.model_name == "m1"; }),
              2)
        << "--count 2 assigns exactly two files";

    dm({"delete", files[1]});
    dm({"remove", files[2]});
    dm({"remove", files[2]}, 1);  // no longer queued
    pending = local_pending();
    ASSERT_EQ(pending.size(), 2u);
    EXPECT_EQ(pending[0].path, files[0]);
    EXPECT_EQ(pending[1].path, files[3]);
}

// migrate moves each segment exactly, with its model, and leaves nothing behind (TD-326).
TEST_F(DatasetManagerCliTest, LocalMigrateMovesSegmentsWithTheirModel) {
    write_config("");
    const fs::path data = root_ / "data.jsonl";
    write_pairs(data, 10);
    dm({"add", data.string()});
    dm({"segment", data.string(), "--ranges", "0-4,5-9"});
    dm({"assign", "m1"});

    dm({"migrate", "chatbot", data.string()});

    EXPECT_TRUE(local_pending().empty()) << "nothing left in the legacy pool";
    const auto moved = local_pending("chatbot");
    ASSERT_EQ(moved.size(), 2u);
    EXPECT_TRUE(is_segment(moved[0], 0, 5));
    EXPECT_TRUE(is_segment(moved[1], 5, 5));
    for (const auto& e : moved)
        EXPECT_EQ(e.model_name, "m1");
}

// ============================================================================
// Remote mode (a real registry_server)
// ============================================================================

TEST_F(DatasetManagerCliTest, RemoteSegmentAssignCountMigrateAndDelete) {
    ASSERT_TRUE(start_server()) << "registry_server didn't start";
    write_config(url());
    const fs::path data = root_ / "remote.jsonl";
    write_pairs(data, 9);

    // A local file is uploaded first, then its server path is segmented (TD-543).
    dm({"segment", data.string(), "--count", "3"});
    auto pending = remote_pending();
    ASSERT_EQ(pending.size(), 3u);
    const std::string segmented_path = pending[0].path;
    EXPECT_NE(segmented_path, fs::absolute(data).string()) << "the segments name the uploaded copy";
    EXPECT_TRUE(is_segment(pending[0], 0, 3));
    EXPECT_TRUE(is_segment(pending[2], 6, 3));

    const fs::path other = root_ / "other.jsonl";
    write_pairs(other, 2);
    dm({"add", other.string()});  // uploaded as a whole file
    pending = remote_pending();
    ASSERT_EQ(pending.size(), 4u);
    const std::string whole_path = pending[3].path;

    // --count picks unassigned whole-file entries only (TD-205), so it takes the uploaded file.
    dm({"assign", "m1", "--count", "1"});
    pending = remote_pending();
    for (const auto& e : pending)
        EXPECT_EQ(e.model_name, e.path == whole_path ? "m1" : "") << e.path << " " << e.segment_start;

    // remove is local-only (TD-327): refused, nothing changes.
    dm({"remove", whole_path}, 1);
    EXPECT_EQ(remote_pending().size(), 4u);

    dm({"migrate", "world_model"});
    EXPECT_TRUE(remote_pending().empty());
    const auto moved = remote_pending("world_model");
    ASSERT_EQ(moved.size(), 4u);
    EXPECT_EQ(std::count_if(moved.begin(), moved.end(),
                            [&](const PendingEntry& e) { return e.segment_start >= 0; }),
              3)
        << "the segments kept their ranges";
    EXPECT_EQ(std::count_if(moved.begin(), moved.end(),
                            [&](const PendingEntry& e) { return e.model_name == "m1"; }),
              1)
        << "the assignment moved with its file";

    // A path names only a whole-file entry: the segmented file's path deletes nothing, the whole
    // file's path deletes it.
    dm({"--kind", "world_model", "delete", segmented_path}, 1);
    dm({"--kind", "world_model", "delete", whole_path});
    const auto left = remote_pending("world_model");
    ASSERT_EQ(left.size(), 3u);
    for (const auto& e : left)
        EXPECT_EQ(e.path, segmented_path);
}
