/**
 * @file registry_ftp_confinement_test.cpp
 * @brief Permanent regression test for TD-040: handle_acquire() must never
 *        mint a working FTP token for a pending entry that doesn't resolve
 *        under registry_server's --data-dir.
 *
 * Unlike dataset_registry_live_test.cpp (which connects to an externally-
 * managed, already-running instance via REGISTRY_SERVER_HOST/PORT and skips
 * if unreachable), this test spawns its own registry_server subprocess with
 * a --data-dir and --ftp-enabled config it fully controls, so it runs
 * automatically in any environment with no manual setup — and so it can
 * deliberately construct a pending entry it knows for certain lies outside
 * that data_dir, which an externally-managed instance's unknown
 * configuration can't guarantee.
 *
 * Exercises the actual compiled registry_server binary end to end (fork +
 * execv, real HTTP requests via httplib::Client) rather than a mocked or
 * in-process reconstruction of its handlers — RegistryServer.cpp's request
 * handlers are static functions closed over process-global state (main()'s
 * own locals), not yet extracted into an independently-instantiable class
 * the way TrainerAdminAPI is (see TD-035, tracked separately: that's the
 * broader "no dedicated unit test" gap for this and five other daemon/CLI
 * binaries). A real subprocess is the correct-fidelity way to test this
 * specific fix without that larger refactor.
 */

#include <fcntl.h>
#include <gtest/gtest.h>
#include <httplib.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

constexpr int kBasePort = 44100;
constexpr int kPortSpan = 100;

int pick_port(int seed, int offset) {
    return kBasePort + offset * kPortSpan + (seed % kPortSpan);
}

}  // namespace

class RegistryFtpConfinementTest : public ::testing::Test {
   protected:
    fs::path scratch_root_;
    fs::path data_dir_;
    fs::path outside_dir_;
    int http_port_ = 0;
    int ftp_port_ = 0;
    pid_t server_pid_ = -1;
    bool server_ready_ = false;

    void SetUp() override {
#ifndef REGISTRY_SERVER_BINARY_PATH
        GTEST_SKIP() << "REGISTRY_SERVER_BINARY_PATH not defined at build time — "
                        "registry_server target unavailable";
        return;
#else
        if (!fs::exists(REGISTRY_SERVER_BINARY_PATH)) {
            GTEST_SKIP() << "registry_server binary not found at " << REGISTRY_SERVER_BINARY_PATH
                         << " — build it first";
            return;
        }

        const auto seed =
            static_cast<int>(::getpid()) + static_cast<int>(reinterpret_cast<uintptr_t>(this));
        http_port_ = pick_port(seed, 0);
        ftp_port_ = pick_port(seed, 1);

        scratch_root_ =
            fs::temp_directory_path() / ("adai_td040_regress_" + std::to_string(::getpid()) + "_" +
                                         std::to_string(reinterpret_cast<uintptr_t>(this)));
        data_dir_ = scratch_root_ / "data";
        outside_dir_ = scratch_root_ / "outside";
        fs::create_directories(data_dir_);
        fs::create_directories(outside_dir_);

        // The file the vulnerability used to expose — same role as the
        // "/etc/passwd"-style target TD-040's writeup describes, just a
        // synthetic stand-in so the test doesn't depend on host filesystem
        // contents.
        {
            std::ofstream f(outside_dir_ / "secret.txt");
            f << "SENSITIVE_OUTSIDE_DATA_DIR_CONTENT";
        }
        {
            std::ofstream f(data_dir_ / "legit.jsonl");
            f << "legit training data\n";
        }

        server_pid_ = ::fork();
        ASSERT_GE(server_pid_, 0) << "fork() failed: " << strerror(errno);
        if (server_pid_ == 0) {
            // Child: exec the real registry_server with a fully test-owned
            // config. execv wants a mutable argv of char*; const_cast is
            // safe here since execv never actually modifies these strings.
            const std::string port_str = std::to_string(http_port_);
            const std::string ftp_port_str = std::to_string(ftp_port_);
            const std::string data_dir_str = data_dir_.string();
            char* argv[] = {
                const_cast<char*>(REGISTRY_SERVER_BINARY_PATH),
                const_cast<char*>("--port"),
                const_cast<char*>(port_str.c_str()),
                const_cast<char*>("--data-dir"),
                const_cast<char*>(data_dir_str.c_str()),
                const_cast<char*>("--ftp-enabled"),
                const_cast<char*>("--ftp-port"),
                const_cast<char*>(ftp_port_str.c_str()),
                nullptr,
            };
            // Quiet the child's stdout/stderr so test output isn't
            // interleaved with the server's own logging.
            const int devnull = ::open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                ::dup2(devnull, STDOUT_FILENO);
                ::dup2(devnull, STDERR_FILENO);
            }
            ::execv(REGISTRY_SERVER_BINARY_PATH, argv);
            ::_exit(127);  // execv only returns on failure
        }

        // Parent: poll /health until the server is reachable or 5s elapse.
        httplib::Client probe("127.0.0.1", http_port_);
        probe.set_connection_timeout(std::chrono::milliseconds(200));
        probe.set_read_timeout(std::chrono::milliseconds(200));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            if (auto res = probe.Get("/health"); res && res->status == 200) {
                server_ready_ = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!server_ready_) {
            TearDown();
            GTEST_SKIP() << "registry_server did not become ready within 5s";
        }
#endif
    }

    void TearDown() override {
        if (server_pid_ > 0) {
            ::kill(server_pid_, SIGKILL);
            int status = 0;
            ::waitpid(server_pid_, &status, 0);
            server_pid_ = -1;
        }
        std::error_code ec;
        if (!scratch_root_.empty()) {
            fs::remove_all(scratch_root_, ec);
        }
    }
};

TEST_F(RegistryFtpConfinementTest, OutOfTreeEntryExcludedFromAcquireResponse) {
    httplib::Client cli("127.0.0.1", http_port_);

    // Add one pending entry outside data_dir, one legitimate entry inside it.
    {
        std::ostringstream body;
        body << "{\"path\":\"" << (outside_dir_ / "secret.txt").string() << "\"}";
        auto res = cli.Post("/registry/td040/pending/add", body.str(), "application/json");
        ASSERT_TRUE(res);
        EXPECT_EQ(res->status, 200);
    }
    {
        std::ostringstream body;
        body << "{\"path\":\"" << (data_dir_ / "legit.jsonl").string() << "\"}";
        auto res = cli.Post("/registry/td040/pending/add", body.str(), "application/json");
        ASSERT_TRUE(res);
        EXPECT_EQ(res->status, 200);
    }

    auto acquire =
        cli.Post("/registry/td040/acquire", "{\"run_id\":\"td040-regression\",\"max_files\":10}",
                 "application/json");
    ASSERT_TRUE(acquire);
    ASSERT_EQ(acquire->status, 200);

    const std::string& body = acquire->body;

    // The out-of-tree file must never appear anywhere in the response —
    // no registry_path, no ftp_path, no token. This is the actual TD-040
    // regression assertion: before the fix, this file appeared here with a
    // working ftp_username/ftp_password and an ftp_path containing "../".
    EXPECT_EQ(body.find("secret.txt"), std::string::npos)
        << "out-of-tree entry must be completely excluded from the acquire "
           "response; got: "
        << body;

    // The legitimate file must still be acquired normally, with a real FTP
    // token — confirming the fix didn't break the intended, safe case.
    EXPECT_NE(body.find("legit.jsonl"), std::string::npos)
        << "legitimate in-tree entry should still be acquired; got: " << body;
    EXPECT_NE(body.find("\"ftp_username\""), std::string::npos)
        << "legitimate entry should still receive a working FTP token; got: " << body;
}

TEST_F(RegistryFtpConfinementTest, OutOfTreeEntryRemainsUnclaimedNotConsumed) {
    httplib::Client cli("127.0.0.1", http_port_);

    {
        std::ostringstream body;
        body << "{\"path\":\"" << (outside_dir_ / "secret.txt").string() << "\"}";
        auto res = cli.Post("/registry/td040/pending/add", body.str(), "application/json");
        ASSERT_TRUE(res);
    }

    auto acquire =
        cli.Post("/registry/td040/acquire", "{\"run_id\":\"td040-regression\",\"max_files\":10}",
                 "application/json");
    ASSERT_TRUE(acquire);
    ASSERT_EQ(acquire->status, 200);

    // The rejected entry must be left unclaimed (empty run_id), not
    // claimed-then-silently-dropped — so it stays visible and available
    // (for a legacy/non-FTP acquire, or operator cleanup) rather than
    // vanishing into a dangling claimed-but-never-delivered state.
    auto queue = cli.Get("/registry/td040/queue");
    ASSERT_TRUE(queue);
    ASSERT_EQ(queue->status, 200);
    EXPECT_NE(queue->body.find("secret.txt"), std::string::npos)
        << "rejected entry should still be visible in the pending queue; got: " << queue->body;
    EXPECT_NE(queue->body.find("\"run_id\":\"\""), std::string::npos)
        << "rejected entry's run_id must remain empty (unclaimed), not set to the "
           "requesting run_id with no way to ever deliver it; got: "
        << queue->body;
}

TEST_F(RegistryFtpConfinementTest, PendingAddStillAcceptsOutOfTreePathForDirectFilesystemUse) {
    // TD-040's handle_pending_add() deliberately still *accepts* an
    // out-of-tree path (logging a warning, not rejecting it) — rejecting
    // outright would break legitimate non-FTP, direct-filesystem-path
    // deployments that never route through FtpDataServer at all. This test
    // guards that deliberate behavior, distinct from the acquire-time
    // enforcement the other two tests cover.
    httplib::Client cli("127.0.0.1", http_port_);
    std::ostringstream body;
    body << "{\"path\":\"" << (outside_dir_ / "secret.txt").string() << "\"}";
    auto res = cli.Post("/registry/td040/pending/add", body.str(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("\"added\":true"), std::string::npos) << res->body;
}
