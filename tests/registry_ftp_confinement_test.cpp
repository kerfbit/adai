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
#include <sys/resource.h>
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
#include <vector>
#include "RegistryTransport.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int kBasePort = 44100;
constexpr int kPortSpan = 100;

int pick_port(int seed, int offset) {
    return kBasePort + offset * kPortSpan + (seed % kPortSpan);
}

// REGISTRY_SERVER_BINARY_PATH is baked in at compile time via CMake's
// $<TARGET_FILE:registry_server> -- an absolute path into *this machine's*
// build directory. That's correct for a normal ctest run (the build tree is
// always right there), but breaks the instant this test binary is packaged
// and run somewhere else without that exact path existing (confirmed on
// ai-machine's GPU test package, which ships a flat bin/ with no build
// tree at all). ADAI_REGISTRY_SERVER_BINARY lets a relocated package point
// this at wherever it actually placed the registry_server binary, without
// changing behavior for the normal in-tree ctest case.
std::string resolve_registry_server_binary() {
    if (const char* override_path = std::getenv("ADAI_REGISTRY_SERVER_BINARY");
        override_path != nullptr && *override_path != '\0') {
        return override_path;
    }
#ifdef REGISTRY_SERVER_BINARY_PATH
    return REGISTRY_SERVER_BINARY_PATH;
#else
    return "";
#endif
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
        const std::string registry_server_binary = resolve_registry_server_binary();
        if (registry_server_binary.empty()) {
            GTEST_SKIP() << "REGISTRY_SERVER_BINARY_PATH not defined at build time and "
                            "ADAI_REGISTRY_SERVER_BINARY not set — registry_server target "
                            "unavailable";
            return;
        }
        if (!fs::exists(registry_server_binary)) {
            GTEST_SKIP() << "registry_server binary not found at " << registry_server_binary
                         << " — build it first, or point ADAI_REGISTRY_SERVER_BINARY at it";
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
                const_cast<char*>(registry_server_binary.c_str()),
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
            ::execv(registry_server_binary.c_str(), argv);
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

// TD-040 (found during a completeness re-check, September 11, 2026):
// path_resolves_under()'s first cut canonicalized `path` but not `root`.
// data_dir defaults to (and is commonly configured as) a *relative* path
// ("registry_sessions") — lexically_relative() is purely lexical, so
// comparing an absolute canonicalized `path` against a relative,
// uncanonicalized `root` returns an empty relative path regardless of actual
// containment, making EVERY legitimate in-tree pending/add spuriously log
// the "does not resolve under data_dir" warning. Not a security hole (the
// real enforcement point, handle_acquire()'s ftp_deliverable, already
// canonicalized its own data_root correctly) but it defeated the operator
// visibility this warning exists to provide. Neither the fixture above (its
// --data-dir is always an absolute scratch path) nor the original manual
// live check exercised a relative --data-dir, which is how this slipped
// through — this test closes that gap directly, independent of the fixture.
TEST(RegistryPendingAddRelativeDataDirTest, InTreeAddWithRelativeDataDirLogsNoWarning) {
    const std::string registry_server_binary = resolve_registry_server_binary();
    if (registry_server_binary.empty()) {
        GTEST_SKIP() << "REGISTRY_SERVER_BINARY_PATH not defined at build time and "
                        "ADAI_REGISTRY_SERVER_BINARY not set";
        return;
    }
    if (!fs::exists(registry_server_binary)) {
        GTEST_SKIP() << "registry_server binary not found at " << registry_server_binary
                     << " — build it first, or point ADAI_REGISTRY_SERVER_BINARY at it";
        return;
    }

    const int http_port = pick_port(static_cast<int>(::getpid()), 2);
    const fs::path scratch_root =
        fs::temp_directory_path() / ("adai_td040_relroot_" + std::to_string(::getpid()));
    fs::create_directories(scratch_root / "data" / "g2");
    {
        std::ofstream f(scratch_root / "data" / "g2" / "legit.jsonl");
        f << "legit\n";
    }
    const fs::path log_path = scratch_root / "server.log";

    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0) << "fork() failed: " << strerror(errno);
    if (pid == 0) {
        if (::chdir(scratch_root.c_str()) != 0) {
            ::_exit(126);
        }
        const std::string port_str = std::to_string(http_port);
        // Deliberately relative --data-dir, matching the real default
        // ("registry_sessions") — the exact shape that exposed the bug.
        char* argv[] = {
            const_cast<char*>(registry_server_binary.c_str()),
            const_cast<char*>("--port"),
            const_cast<char*>(port_str.c_str()),
            const_cast<char*>("--data-dir"),
            const_cast<char*>("data"),
            nullptr,
        };
        const int log_fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (log_fd >= 0) {
            ::dup2(log_fd, STDOUT_FILENO);
            ::dup2(log_fd, STDERR_FILENO);
        }
        ::execv(registry_server_binary.c_str(), argv);
        ::_exit(127);
    }

    httplib::Client probe("127.0.0.1", http_port);
    probe.set_connection_timeout(std::chrono::milliseconds(200));
    probe.set_read_timeout(std::chrono::milliseconds(200));
    bool ready = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto res = probe.Get("/health"); res && res->status == 200) {
            ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!ready) {
        ::kill(pid, SIGKILL);
        int status = 0;
        ::waitpid(pid, &status, 0);
        std::error_code ec;
        fs::remove_all(scratch_root, ec);
        GTEST_SKIP() << "registry_server did not become ready within 5s";
        return;
    }

    httplib::Client cli("127.0.0.1", http_port);
    const std::string abs_legit =
        fs::absolute(scratch_root / "data" / "g2" / "legit.jsonl").string();
    std::ostringstream body;
    body << "{\"path\":\"" << abs_legit << "\"}";
    auto res = cli.Post("/registry/g2/pending/add", body.str(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    // Give the logger a moment to flush before reading the log back.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    ::kill(pid, SIGKILL);
    int status = 0;
    ::waitpid(pid, &status, 0);

    std::ifstream log_file(log_path);
    std::ostringstream log_contents;
    log_contents << log_file.rdbuf();
    EXPECT_EQ(log_contents.str().find("does not resolve under data_dir"), std::string::npos)
        << "pending/add spuriously warned about a genuinely in-tree path when --data-dir is "
           "relative; full server log:\n"
        << log_contents.str();

    std::error_code ec;
    fs::remove_all(scratch_root, ec);
}

// TD-495: the verified failure, through the real server. registry_server runs with a 4 KiB
// file-size limit (a full disk, in effect) over a queue already bigger than that. Adding a file
// used to rewrite the queue in place, keep only what fit, and answer 200; it must answer 500 and
// keep the whole queue.
TEST(RegistryDiskFullTest, FailedQueueSaveIsAnErrorAndKeepsTheWholeQueue) {
    const std::string registry_server_binary = resolve_registry_server_binary();
    if (registry_server_binary.empty() || !fs::exists(registry_server_binary)) {
        GTEST_SKIP() << "registry_server binary not available";
        return;
    }
    const int http_port = pick_port(static_cast<int>(::getpid()), 3);
    const fs::path root = fs::temp_directory_path() / ("adai_td495_" + std::to_string(::getpid()));
    fs::remove_all(root);
    const fs::path group_dir = root / "data" / "td495";
    fs::create_directories(group_dir);
    constexpr int kQueued = 300;
    {
        std::ofstream q(group_dir / "pending_files.txt");
        for (int i = 0; i < kQueued; ++i)
            q << (group_dir / ("queued_" + std::to_string(i) + ".jsonl")).string() << '\n';
    }
    ASSERT_GT(fs::file_size(group_dir / "pending_files.txt"), 4096u);
    {
        std::ofstream f(group_dir / "new.jsonl");
        f << "{\"input\":\"a\",\"response\":\"b\"}\n";
    }

    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        ::signal(SIGXFSZ,
                 SIG_IGN);  // writes past the limit fail with EFBIG instead of killing the server
        rlimit limit{4096, 4096};
        ::setrlimit(RLIMIT_FSIZE, &limit);
        const std::string port_str = std::to_string(http_port);
        const std::string data_str = (root / "data").string();
        char* argv[] = {const_cast<char*>(registry_server_binary.c_str()),
                        const_cast<char*>("--port"),
                        const_cast<char*>(port_str.c_str()),
                        const_cast<char*>("--data-dir"),
                        const_cast<char*>(data_str.c_str()),
                        nullptr};
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
        }
        ::execv(registry_server_binary.c_str(), argv);
        ::_exit(127);
    }
    auto stop = [&] {
        ::kill(pid, SIGKILL);  // our own child only
        int status = 0;
        ::waitpid(pid, &status, 0);
    };

    httplib::Client cli("127.0.0.1", http_port);
    cli.set_connection_timeout(std::chrono::milliseconds(200));
    bool ready = false;
    for (auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
         std::chrono::steady_clock::now() < deadline;) {
        if (auto r = cli.Get("/health"); r && r->status == 200) {
            ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!ready) {
        stop();
        fs::remove_all(root);
        GTEST_SKIP() << "registry_server did not become ready within 5s";
    }

    const std::string body = "{\"path\":\"" + (group_dir / "new.jsonl").string() + "\"}";
    auto res = cli.Post("/registry/td495/pending/add", body, "application/json");
    stop();

    int lines = 0;
    std::ifstream q(group_dir / "pending_files.txt");
    for (std::string line; std::getline(q, line);)
        if (!line.empty())
            ++lines;
    std::error_code ec;
    fs::remove_all(root, ec);

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 500) << res->body;
    EXPECT_NE(res->body.find("could not save the pending queue"), std::string::npos) << res->body;
    EXPECT_EQ(lines, kQueued) << "the queue must survive whole";
}

// TD-497: RemoteTransport parsed replies by cutting each object at the next '}' and reading strings
// to the next '"', with no unescaping. A path with '}' vanished from load_pending(), a '"' cut a
// path short and each backslash came back quadrupled. Every RemoteTransport call that carries a
// path must now round-trip these names exactly, including the FTP acquire reply (the fixture runs
// with FTP on).
TEST_F(RegistryFtpConfinementTest, RemoteTransportRoundTripsPathsWithJsonSyntax) {
    const std::vector<std::string> names = {"a}b.jsonl",
                                            "c{d.jsonl",
                                            "back\\slash.jsonl",
                                            "quo\"te.jsonl",
                                            "caf\xc3\xa9 \xe6\x97\xa5.jsonl",
                                            "plain.jsonl"};
    std::vector<std::string> paths;
    for (const auto& n : names) {
        paths.push_back((data_dir_ / n).string());
        std::ofstream(paths.back()) << "{\"input\":\"a\",\"response\":\"b\"}\n";
    }
    const std::string windows_path = "C:\\data\\w.jsonl";  // queued only (outside data_dir)

    RemoteTransport rt("http://127.0.0.1:" + std::to_string(http_port_), "td497");
    for (const auto& p : paths)
        ASSERT_TRUE(rt.add_pending(p)) << p;
    ASSERT_TRUE(rt.add_pending(windows_path));

    std::vector<PendingEntry> pending;
    ASSERT_TRUE(rt.load_pending(pending));
    std::vector<std::string> pending_paths;
    for (const auto& e : pending)
        pending_paths.push_back(e.path);
    std::vector<std::string> expected = paths;
    expected.push_back(windows_path);
    EXPECT_EQ(pending_paths, expected);

    const AcquireResponse acquired = rt.acquire("td497-run", 0);
    ASSERT_FALSE(acquired.failed);
    std::vector<std::string> acquired_paths;
    for (const auto& t : acquired.files) {
        acquired_paths.push_back(t.registry_path);
        EXPECT_FALSE(t.ftp_username.empty()) << "the FTP reply shape is under test";
    }
    EXPECT_EQ(acquired_paths, paths) << "every in-tree file the server claimed reaches the client";

    std::vector<DataVersion> trained;
    for (const auto& p : paths) {
        DataVersion dv;
        dv.data_file = p;
        dv.trained = true;
        dv.num_samples = 1;
        trained.push_back(dv);
    }
    rt.commit_trained("td497-run", trained, paths);
    std::vector<DataVersion> registry;
    ASSERT_TRUE(rt.load_registry(registry));
    std::vector<std::string> registry_paths;
    for (const auto& dv : registry) {
        registry_paths.push_back(dv.data_file);
        EXPECT_TRUE(dv.trained) << dv.data_file;
    }
    EXPECT_EQ(registry_paths, paths);

    const RequeueResult requeued = rt.requeue_trained({paths[0], paths[3]}, false, "");
    ASSERT_TRUE(requeued.ok);
    EXPECT_EQ(requeued.requeued, 2);
    ASSERT_EQ(requeued.details.size(), 2u);
    EXPECT_EQ(requeued.details[0].path, paths[0]);
    EXPECT_EQ(requeued.details[1].path, paths[3]);

    const DeleteResult deleted = rt.delete_paths({paths[2], windows_path}, false, false);
    EXPECT_EQ(deleted.deleted, 2);
    ASSERT_EQ(deleted.details.size(), 2u);
    EXPECT_EQ(deleted.details[0].path, paths[2]);
    EXPECT_EQ(deleted.details[1].path, windows_path);
}

// TD-497: the server's escaper handled only '"', '\\', \n, \r and \t, so any other control
// character or invalid UTF-8 in a stored path (queued before TD-496 refused them) made the whole
// reply invalid JSON. Such entries are seeded straight into the queue file; the reply must stay
// readable.
TEST_F(RegistryFtpConfinementTest, ReplyStaysValidJsonForOldUnusualPaths) {
    fs::create_directories(data_dir_ / "td497old");
    {
        std::ofstream q(data_dir_ / "td497old" / "pending_files.txt", std::ios::binary);
        q << "/data/bell\x07.jsonl\n/data/bad\xff.jsonl\n/data/ok.jsonl\n";  // pre-TD-496 format
    }
    RemoteTransport rt("http://127.0.0.1:" + std::to_string(http_port_), "td497old");
    std::vector<PendingEntry> pending;
    ASSERT_TRUE(rt.load_pending(pending));
    ASSERT_EQ(pending.size(), 3u);
    EXPECT_EQ(pending[0].path, "/data/bell\x07.jsonl");
    EXPECT_EQ(pending[1].path, "/data/bad\xef\xbf\xbd.jsonl");  // U+FFFD
    EXPECT_EQ(pending[2].path, "/data/ok.jsonl");
}

// TD-321: registry_server's assign treated any count <= 0 as "assign every pending entry", so a
// negative count from a typo assigned the whole pool. 0 (and an absent count) still mean all, as
// the Android dashboard sends; a negative count is a 400 that changes nothing.
TEST_F(RegistryFtpConfinementTest, AssignRefusesNegativeCount) {
    RemoteTransport rt("http://127.0.0.1:" + std::to_string(http_port_), "td321");
    ASSERT_TRUE(rt.add_pending("/data/a.jsonl"));
    ASSERT_TRUE(rt.add_pending("/data/b.jsonl"));

    httplib::Client cli("127.0.0.1", http_port_);
    auto res = cli.Post("/registry/td321/assign", "{\"model_name\":\"m\",\"count\":-2}",
                        "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400) << res->body;
    std::vector<PendingEntry> pending;
    ASSERT_TRUE(rt.load_pending(pending));
    for (const auto& e : pending)
        EXPECT_TRUE(e.model_name.empty()) << e.path;

    EXPECT_EQ(rt.assign("m", {}, 0).assigned, 2) << "count 0 still assigns every entry";
}

#ifdef DATASET_MANAGER_BINARY_PATH
namespace {
// Runs dataset_manager with @p args (output discarded) and returns its exit status, -1 if it
// didn't exit normally.
// Like run_dataset_manager(), but returns stdout and stderr together in @p output.
int run_dataset_manager_output(const std::vector<std::string>& args, std::string& output) {
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

int run_dataset_manager(const std::vector<std::string>& args, const std::string& cwd = "") {
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
        if (!cwd.empty() && ::chdir(cwd.c_str()) != 0)
            ::_exit(126);
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
        }
        ::execv(argv[0], argv.data());
        ::_exit(127);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) != pid || !WIFEXITED(status))
        return -1;
    return WEXITSTATUS(status);
}
}  // namespace

// TD-327: `remove` erased the entry in memory and "saved" through RemoteTransport::save_pending(),
// a no-op, then printed "Removed" and exited 0 while the file stayed queued and would still be
// trained. remove and clear-pending now refuse up front when REGISTRY_SERVER_URL is set.
TEST_F(RegistryFtpConfinementTest, DatasetManagerRemoveRefusesInRemoteMode) {
    if (!fs::exists(DATASET_MANAGER_BINARY_PATH))
        GTEST_SKIP() << "dataset_manager not built";
    RemoteTransport rt("http://127.0.0.1:" + std::to_string(http_port_), "td327");
    const std::string queued = "/data/td327/queued.jsonl";
    ASSERT_TRUE(rt.add_pending(queued));

    const fs::path config = scratch_root_ / "config.registry.conf";
    std::ofstream(config) << "REGISTRY_SERVER_URL=http://127.0.0.1:" << http_port_
                          << "\nRUN_GROUP=td327\nNAME_SERVICE_URL=\nSESSION_DIR="
                          << (scratch_root_ / "sessions").string() << "\n";

    EXPECT_EQ(run_dataset_manager({"--config", config.string(), "remove", queued}), 1);
    EXPECT_EQ(run_dataset_manager({"--config", config.string(), "clear-pending"}), 1);

    std::vector<PendingEntry> pending;
    ASSERT_TRUE(rt.load_pending(pending));
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].path, queued);
}

namespace {
// Stand-in for mns_server's two lookups: the full record (any state) and /resolve, which the real
// server refuses with 409 while a model is still "initializing".
class FakeNameService {
   public:
    FakeNameService() {
        server_.Get("/models/td329-model", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"model_id":"id-329","model_name":"td329-model",)"
                            R"("state":"initializing","run_group":"td329-mns","artifact":{}})",
                            "application/json");
        });
        server_.Get(
            "/models/td329-model/resolve", [](const httplib::Request&, httplib::Response& res) {
                res.status = 409;
                res.set_content(R"({"error":"model has no artifact yet"})", "application/json");
            });
        server_.Get(R"(/models/([^/]+))", [](const httplib::Request&, httplib::Response& res) {
            res.status = 404;
            res.set_content(R"({"error":"model not found"})", "application/json");
        });
        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ > 0)
            thread_ = std::thread([this] { server_.listen_after_bind(); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (port_ > 0 && !server_.is_running() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ~FakeNameService() {
        server_.stop();
        if (thread_.joinable())
            thread_.join();
    }
    bool running() const {
        return port_ > 0 && server_.is_running();
    }
    std::string url() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

   private:
    httplib::Server server_;
    std::thread thread_;
    int port_ = -1;
};

std::vector<PendingEntry> pending_in(int http_port, const std::string& group) {
    std::vector<PendingEntry> pending;
    RemoteTransport rt("http://127.0.0.1:" + std::to_string(http_port), group);
    EXPECT_TRUE(rt.load_pending(pending));
    return pending;
}
}  // namespace

// TD-329: only `assign` adopted the model's MNS run_group (and only through /resolve, which MNS
// refuses for a model not yet trained); every other command used the config's RUN_GROUP, so
// operators read and changed a different pool from the one assign wrote into. Now every command
// resolves the group once, from the model's full MNS record in any state.
TEST_F(RegistryFtpConfinementTest, DatasetManagerUsesTheModelsMnsRunGroupForEveryCommand) {
    if (!fs::exists(DATASET_MANAGER_BINARY_PATH))
        GTEST_SKIP() << "dataset_manager not built";
    FakeNameService mns;
    ASSERT_TRUE(mns.running());
    const std::string base = "http://127.0.0.1:" + std::to_string(http_port_);
    const std::string in_mns_group = "/data/td329/mns.jsonl";
    const std::string in_config_group = "/data/td329/config.jsonl";
    ASSERT_TRUE(RemoteTransport(base, "td329-mns").add_pending(in_mns_group));
    ASSERT_TRUE(RemoteTransport(base, "td329-config").add_pending(in_config_group));

    const auto write_config = [&](const fs::path& path, const std::string& extra) {
        std::ofstream(path) << "REGISTRY_SERVER_URL=" << base
                            << "\nRUN_GROUP=td329-config\nSESSION_DIR="
                            << (scratch_root_ / "sessions").string() << "\n"
                            << extra;
    };
    const fs::path with_mns = scratch_root_ / "td329-mns.conf";
    write_config(with_mns, "NAME_SERVICE_URL=" + mns.url() + "\nMODEL_NAME=td329-model\n");

    // assign: the model is still initializing; resolve_model() made this fail outright.
    ASSERT_EQ(run_dataset_manager({"--config", with_mns.string(), "assign", "td329-model"}), 0);
    auto pending = pending_in(http_port_, "td329-mns");
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].model_name, "td329-model");

    // unassign: used to look in td329-config, find nothing and exit 1.
    EXPECT_EQ(run_dataset_manager({"--config", with_mns.string(), "unassign", "td329-model"}), 0);
    pending = pending_in(http_port_, "td329-mns");
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].model_name, "");

    // delete: no model on the command line, so MODEL_NAME's group.
    EXPECT_EQ(run_dataset_manager({"--config", with_mns.string(), "delete", in_mns_group}), 0);
    EXPECT_TRUE(pending_in(http_port_, "td329-mns").empty());

    // The config group was never touched.
    pending = pending_in(http_port_, "td329-config");
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].path, in_config_group);

    // A model MNS doesn't know keeps the config group.
    const fs::path unknown = scratch_root_ / "td329-unknown.conf";
    write_config(unknown, "NAME_SERVICE_URL=" + mns.url() + "\nMODEL_NAME=not-registered\n");
    EXPECT_EQ(run_dataset_manager({"--config", unknown.string(), "delete", in_config_group}), 0);
    EXPECT_TRUE(pending_in(http_port_, "td329-config").empty());
}

// TD-329: when the name service doesn't answer, the command can't know its pool, so it changes
// nothing instead of falling back to RUN_GROUP.
TEST_F(RegistryFtpConfinementTest, DatasetManagerRefusesWhenTheNameServiceDoesNotAnswer) {
    if (!fs::exists(DATASET_MANAGER_BINARY_PATH))
        GTEST_SKIP() << "dataset_manager not built";
    const std::string base = "http://127.0.0.1:" + std::to_string(http_port_);
    const std::string queued = "/data/td329/down.jsonl";
    ASSERT_TRUE(RemoteTransport(base, "td329-down").add_pending(queued));

    // A port nothing listens on: bound, then released.
    httplib::Server closed;
    const int dead_port = closed.bind_to_any_port("127.0.0.1");
    ASSERT_GT(dead_port, 0);
    closed.stop();

    const fs::path config = scratch_root_ / "td329-down.conf";
    std::ofstream(config) << "REGISTRY_SERVER_URL=" << base << "\nRUN_GROUP=td329-down\n"
                          << "NAME_SERVICE_URL=http://127.0.0.1:" << dead_port
                          << "\nNAME_SERVICE_TIMEOUT_MS=1000\nMODEL_NAME=td329-model\nSESSION_DIR="
                          << (scratch_root_ / "sessions").string() << "\n";

    EXPECT_EQ(run_dataset_manager({"--config", config.string(), "delete", queued}), 1);
    const auto pending = pending_in(http_port_, "td329-down");
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].path, queued);
}

namespace {
// Five training pairs, plus a blank line and a JSON line that isn't a pair: 7 lines, 6 non-blank.
void write_five_pairs(const fs::path& path) {
    std::ofstream out(path);
    for (int i = 0; i < 5; ++i) {
        out << R"({"input":"question )" << i << R"(","output":"answer )" << i << "\"}\n";
        if (i == 2)
            out << "\n" << R"({"note":"not a pair"})" << "\n";
    }
}

std::string segment_config(const fs::path& path, int http_port, const std::string& group,
                           const fs::path& sessions) {
    std::ofstream(path) << "REGISTRY_SERVER_URL=http://127.0.0.1:" << http_port
                        << "\nRUN_GROUP=" << group << "\nNAME_SERVICE_URL=\nSESSION_DIR="
                        << sessions.string() << "\n";
    return path.string();
}
}  // namespace

// TD-543: against a remote registry, `segment` counted a local file's pairs and queued its client
// path, which the server can't read or serve. A local file is now uploaded first and its server
// path segmented.
TEST_F(RegistryFtpConfinementTest, DatasetManagerSegmentUploadsALocalFileFirst) {
    if (!fs::exists(DATASET_MANAGER_BINARY_PATH))
        GTEST_SKIP() << "dataset_manager not built";
    const fs::path local = scratch_root_ / "client-side" / "td543-local.jsonl";
    fs::create_directories(local.parent_path());
    write_five_pairs(local);
    const std::string config = segment_config(scratch_root_ / "td543-local.conf", http_port_,
                                              "td543-local", scratch_root_ / "sessions");

    ASSERT_EQ(run_dataset_manager({"--config", config, "segment", local.string(), "--count", "2"}),
              0);

    const auto pending = pending_in(http_port_, "td543-local");
    ASSERT_EQ(pending.size(), 2u);
    for (const auto& e : pending) {
        EXPECT_NE(e.path, local.string());
        EXPECT_NE(e.path.find("/uploads/td543-local.jsonl"), std::string::npos) << e.path;
        EXPECT_TRUE(e.run_id.empty());
    }
    EXPECT_EQ(pending[0].segment_start, 0);
    EXPECT_EQ(pending[0].segment_count, 3);
    EXPECT_EQ(pending[1].segment_start, 3);
    EXPECT_EQ(pending[1].segment_count, 2);
}

// TD-543: a path already queued on the server is segmented by the pair count the server recorded,
// which counted non-blank lines and so was too high for this file (6, not 5).
TEST_F(RegistryFtpConfinementTest, DatasetManagerSegmentsAServerPathByItsServerPairCount) {
    if (!fs::exists(DATASET_MANAGER_BINARY_PATH))
        GTEST_SKIP() << "dataset_manager not built";
    const fs::path local = scratch_root_ / "client-side" / "td543-server.jsonl";
    fs::create_directories(local.parent_path());
    write_five_pairs(local);
    RemoteTransport rt("http://127.0.0.1:" + std::to_string(http_port_), "td543-server");
    const std::string server_path = rt.upload_file(local.string());
    ASSERT_FALSE(server_path.empty());

    auto pending = pending_in(http_port_, "td543-server");
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].num_entries, 5);

    const std::string config = segment_config(scratch_root_ / "td543-server.conf", http_port_,
                                              "td543-server", scratch_root_ / "sessions");
    ASSERT_EQ(run_dataset_manager({"--config", config, "segment", server_path, "--count", "5"}), 0);

    pending = pending_in(http_port_, "td543-server");
    ASSERT_EQ(pending.size(), 5u);  // the whole entry was replaced
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(pending[i].path, server_path);
        EXPECT_EQ(pending[i].segment_start, i);
        EXPECT_EQ(pending[i].segment_count, 1);
    }
}

// TD-543: a path that is neither queued on the server nor a local file is refused.
TEST_F(RegistryFtpConfinementTest, DatasetManagerSegmentRefusesAnUnknownRemotePath) {
    if (!fs::exists(DATASET_MANAGER_BINARY_PATH))
        GTEST_SKIP() << "dataset_manager not built";
    const std::string config = segment_config(scratch_root_ / "td543-none.conf", http_port_,
                                              "td543-none", scratch_root_ / "sessions");
    EXPECT_EQ(run_dataset_manager({"--config", config, "segment",
                                   (scratch_root_ / "nowhere.jsonl").string(), "--count", "2"}),
              1);
    EXPECT_TRUE(pending_in(http_port_, "td543-none").empty());
}
// TD-328: in local mode `add data.jsonl` queued "data.jsonl" as given, which a trainer running in
// another directory can't find. It is queued absolute now, and a relative target finds it again.
TEST_F(RegistryFtpConfinementTest, DatasetManagerQueuesLocalFilesAbsolute) {
    if (!fs::exists(DATASET_MANAGER_BINARY_PATH))
        GTEST_SKIP() << "dataset_manager not built";
    const fs::path work = scratch_root_ / "td328-work";
    fs::create_directories(work);
    std::ofstream(work / "data.jsonl") << R"({"input":"q","output":"a"})" << "\n";
    const fs::path sessions = scratch_root_ / "td328-sessions";
    const fs::path config = scratch_root_ / "td328.conf";
    std::ofstream(config) << "REGISTRY_SERVER_URL=\nNAME_SERVICE_URL=\nSESSION_DIR="
                          << sessions.string() << "\n";

    ASSERT_EQ(run_dataset_manager({"--config", config.string(), "add", "data.jsonl"}, work.string()),
              0);
    LocalTransport local((sessions / "data_registry.txt").string(),
                         (sessions / "pending_files.txt").string());
    std::vector<PendingEntry> pending;
    ASSERT_TRUE(local.load_pending(pending));
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].path, (fs::absolute(work) / "data.jsonl").lexically_normal().string());

    // An entry queued relative before TD-328 is still found by the name it was queued under.
    ASSERT_TRUE(local.add_pending("old.jsonl"));
    EXPECT_EQ(run_dataset_manager({"--config", config.string(), "remove", "old.jsonl"},
                                  work.string()),
              0);
    // And the relative name of the new entry finds its absolute entry.
    EXPECT_EQ(run_dataset_manager({"--config", config.string(), "remove", "data.jsonl"},
                                  work.string()),
              0);
    pending.clear();
    ASSERT_TRUE(local.load_pending(pending));
    EXPECT_TRUE(pending.empty());
}
// TD-330: global options were taken from anywhere in argv (a value-less trailing one became an
// argument), commands without arguments ignored extra ones, and there was no --help.
TEST_F(RegistryFtpConfinementTest, DatasetManagerGlobalOptionsAndHelp) {
    if (!fs::exists(DATASET_MANAGER_BINARY_PATH))
        GTEST_SKIP() << "dataset_manager not built";
    const fs::path config = scratch_root_ / "td330.conf";
    std::ofstream(config) << "REGISTRY_SERVER_URL=\nNAME_SERVICE_URL=\nSESSION_DIR="
                          << (scratch_root_ / "td330-sessions").string() << "\n";
    std::string out;
    EXPECT_EQ(run_dataset_manager_output({"--help"}, out), 0);
    EXPECT_NE(out.find("Usage:"), std::string::npos) << out;
    EXPECT_EQ(run_dataset_manager_output({"help"}, out), 0);
    EXPECT_NE(out.find("Usage:"), std::string::npos) << out;
    EXPECT_EQ(run_dataset_manager_output({"--config", config.string(), "status", "--kind",
                                          "chatbot"},
                                         out),
              1);
    EXPECT_NE(out.find("takes no arguments"), std::string::npos) << out;
    EXPECT_EQ(run_dataset_manager_output({"--config"}, out), 1);
    EXPECT_NE(out.find("Missing value for --config"), std::string::npos) << out;
    EXPECT_EQ(run_dataset_manager_output({"--konfig", config.string(), "status"}, out), 1);
    EXPECT_NE(out.find("Unknown option --konfig"), std::string::npos) << out;
    // `models` used to try http://localhost:8083 when NAME_SERVICE_URL was unset.
    EXPECT_EQ(run_dataset_manager_output({"--config", config.string(), "models"}, out), 1);
    EXPECT_NE(out.find("NAME_SERVICE_URL is not set"), std::string::npos) << out;
}

// TD-330: the success line repeated the requested pair count. A cached dataset of 20 rows asked for 50
// pairs queues fewer (the server's slice wraps to the start once: 40), and that is what is reported
// now, matching the queue entry.
TEST_F(RegistryFtpConfinementTest, DatasetManagerReportsThePairsActuallyQueued) {
    if (!fs::exists(DATASET_MANAGER_BINARY_PATH))
        GTEST_SKIP() << "dataset_manager not built";
    const fs::path cache = data_dir_ / "td330" / "datasets" / "org_small_train";
    fs::create_directories(cache);
    {
        std::ofstream f(cache / "full_dataset.jsonl");
        for (int i = 0; i < 20; ++i)
            f << R"({"input":"question )" << i << R"(","output":"answer )" << i << "\"}\n";
    }
    std::ofstream(cache / "full_dataset.complete") << "seeded\n";
    const fs::path config = scratch_root_ / "td330-remote.conf";
    std::ofstream(config) << "REGISTRY_SERVER_URL=http://127.0.0.1:" << http_port_
                          << "\nRUN_GROUP=td330\nNAME_SERVICE_URL=\nSESSION_DIR="
                          << (scratch_root_ / "sessions").string() << "\n";
    std::string out;
    ASSERT_EQ(run_dataset_manager_output({"--config", config.string(), "huggingface", "org/small",
                                          "50", "--input-field", "input", "--output-field",
                                          "output"},
                                         out),
              0)
        << out;
    std::vector<PendingEntry> pending;
    ASSERT_TRUE(RemoteTransport("http://127.0.0.1:" + std::to_string(http_port_), "td330")
                    .load_pending(pending));
    ASSERT_EQ(pending.size(), 1u);
    const int queued = pending[0].num_entries;
    EXPECT_GT(queued, 0);
    EXPECT_LT(queued, 50);
    EXPECT_NE(out.find("(" + std::to_string(queued) + " pairs of 50 requested)"),
              std::string::npos)
        << out;
}
#endif
