/**
 * @file training_pass_files_test.cpp
 * @brief TD-572: resolve_pass_files() and cleanup_downloads() (TrainingPassFiles.cpp), the step
 * every training pass uses to turn its claimed files into local files and to clean up after itself.
 *
 * The FTP cases run a real FtpDataServer (plain FTP) on loopback, on kernel-chosen ports, serving a
 * real DataTransport (libcurl), as DataTransportFtpsTests do.
 */

#include "TrainingPassFiles.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../src/FtpDataServer.hpp"
#endif
#include "../src/DataTransport.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

class TrainingPassFilesTest : public ::testing::Test {
   protected:
    void SetUp() override {
        std::random_device rd;
        root_ = fs::temp_directory_path() /
                ("adai_pass_files_" + std::to_string(rd()) + std::to_string(rd()));
        data_dir_ = root_ / "registry";
        fs::create_directories(data_dir_ / "group");
        write(data_dir_ / "group" / "a.jsonl", "{\"input\":\"a\",\"response\":\"b\"}\n");
        write(data_dir_ / "group" / "b.jsonl",
              "{\"input\":\"c\",\"response\":\"d\"}\n{\"input\":\"e\",\"response\":\"f\"}\n");
        dcfg_.session_dir = (root_ / "sessions").string();
        dcfg_.download_dir = (root_ / "downloads").string();
        dcfg_.max_parallel_downloads = 1;  // one passive port is enough
    }
    void TearDown() override {
#ifndef _WIN32
        if (server_)
            server_->stop();
#endif
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    static void write(const fs::path& p, const std::string& text) {
        std::ofstream(p, std::ios::binary) << text;
    }
    static std::string read(const fs::path& p) {
        std::ifstream f(p, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }

    // A token for `rel` (relative to the registry's data directory) as an FTP registry hands it
    // out.
    FileToken token(const std::string& rel, int seg_start = -1, int seg_count = -1) {
        FileToken t;
        t.registry_path = (data_dir_ / rel).string();
        t.ftp_path = rel;
        t.size_bytes = fs::file_size(data_dir_ / rel);
        t.segment_start = seg_start;
        t.segment_count = seg_count;
#ifndef _WIN32
        if (server_) {
            const auto issued = server_->issue_token("run-td572", rel, 5);
            t.ftp_username = issued.username;
            t.ftp_password = issued.password;
            t.token_expires_utc = issued.token_expires_utc;
        }
#endif
        return t;
    }

#ifndef _WIN32
    static int free_port() {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = 0;
        ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        socklen_t len = sizeof(addr);
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
        const int port = ntohs(addr.sin_port);
        ::close(fd);
        return port;
    }
    static bool listening(int port) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        const bool ok = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
        ::close(fd);
        return ok;
    }
    void start_server() {
        port_ = free_port();
        const int pasv = free_port();
        server_ = std::make_unique<FtpDataServer>(data_dir_.string(), port_, pasv, pasv,
                                                  "127.0.0.1", "", false, "", "");
        server_->start();
        for (int i = 0; i < 200 && !listening(port_); ++i)
            std::this_thread::sleep_for(50ms);
    }
    AcquireResponse ftp_response(std::vector<FileToken> files) {
        AcquireResponse r;
        r.run_id = "run-td572";
        r.ftp_server_host = "127.0.0.1";
        r.ftp_server_port = port_;
        r.files = std::move(files);
        return r;
    }
    std::unique_ptr<FtpDataServer> server_;
    int port_ = 0;
#endif

    fs::path root_, data_dir_;
    DatasetConfig dcfg_;
};

}  // namespace

// Without an FTP host the registry's own paths are used, each with its segment range, and nothing
// is downloaded.
TEST_F(TrainingPassFilesTest, ALocalRegistryUsesItsOwnPaths) {
    AcquireResponse r;
    r.files = {token("group/a.jsonl"), token("group/b.jsonl", 0, 1), token("group/b.jsonl", 1, 1)};
    std::vector<PendingFileRange> ranges;
    std::vector<fs::path> downloaded;
    ASSERT_TRUE(adai::resolve_pass_files(r, dcfg_, ranges, downloaded));
    ASSERT_EQ(ranges.size(), 3u);
    EXPECT_EQ(ranges[0].path, (data_dir_ / "group/a.jsonl").string());
    EXPECT_EQ(ranges[0].segment_start, -1);
    EXPECT_EQ(ranges[1].path, (data_dir_ / "group/b.jsonl").string());
    EXPECT_EQ(ranges[1].segment_start, 0);
    EXPECT_EQ(ranges[2].segment_start, 1);
    EXPECT_EQ(ranges[2].segment_count, 1);
    EXPECT_TRUE(downloaded.empty());
    EXPECT_FALSE(fs::exists(dcfg_.download_dir)) << "nothing to download";
}

#ifndef _WIN32

// From an FTP registry, every claimed unit becomes a local download with its segment range, in
// order; two segments of one file share one download (TD-337).
TEST_F(TrainingPassFilesTest, AnFtpRegistrysFilesAreDownloadedWithTheirRanges) {
    start_server();
    const auto r = ftp_response(
        {token("group/a.jsonl"), token("group/b.jsonl", 0, 1), token("group/b.jsonl", 1, 1)});
    std::vector<PendingFileRange> ranges;
    std::vector<fs::path> downloaded;
    ASSERT_TRUE(adai::resolve_pass_files(r, dcfg_, ranges, downloaded));
    ASSERT_EQ(ranges.size(), 3u);
    ASSERT_EQ(downloaded.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(ranges[i].path, downloaded[i].string());
        EXPECT_EQ(fs::path(ranges[i].path).parent_path(), fs::path(dcfg_.download_dir));
    }
    EXPECT_EQ(read(ranges[0].path), read(data_dir_ / "group/a.jsonl"));
    EXPECT_EQ(read(ranges[1].path), read(data_dir_ / "group/b.jsonl"));
    EXPECT_EQ(ranges[1].path, ranges[2].path) << "one download for both segments";
    EXPECT_EQ(ranges[1].segment_start, 0);
    EXPECT_EQ(ranges[2].segment_start, 1);
}

// With DOWNLOAD_DIR unset, downloads go to <SESSION_DIR>/downloads (they used to fall back to the
// registry host's paths, which don't exist on a separate trainer).
TEST_F(TrainingPassFilesTest, AnUnsetDownloadDirectoryMeansTheSessionsDownloads) {
    start_server();
    dcfg_.download_dir.clear();
    const auto r = ftp_response({token("group/a.jsonl")});
    std::vector<PendingFileRange> ranges;
    std::vector<fs::path> downloaded;
    ASSERT_TRUE(adai::resolve_pass_files(r, dcfg_, ranges, downloaded));
    ASSERT_EQ(ranges.size(), 1u);
    EXPECT_EQ(fs::path(ranges[0].path).parent_path(), fs::path(dcfg_.session_dir) / "downloads");
    EXPECT_TRUE(fs::exists(ranges[0].path));
}

// A download that fails (here, credentials the server never issued) returns false, so the caller
// releases the claimed files.
TEST_F(TrainingPassFilesTest, AFailedDownloadReturnsFalse) {
    start_server();
    FileToken bad = token("group/a.jsonl");
    bad.ftp_password = "not-the-issued-password";
    std::vector<PendingFileRange> ranges;
    std::vector<fs::path> downloaded;
    EXPECT_FALSE(adai::resolve_pass_files(ftp_response({bad}), dcfg_, ranges, downloaded));
}

#endif  // !_WIN32

// Each download and its TD-338 checksum sidecar are removed; a path listed twice (segments of one
// file) and a path that is already gone are fine.
TEST_F(TrainingPassFilesTest, CleanupRemovesDownloadsAndSidecars) {
    fs::create_directories(dcfg_.download_dir);
    const fs::path one = fs::path(dcfg_.download_dir) / "one.jsonl";
    const fs::path two = fs::path(dcfg_.download_dir) / "two.jsonl";
    write(one, "x");
    write(DataTransport::checksum_path_for(one), "1_2");
    write(two, "y");
    const fs::path other = fs::path(dcfg_.download_dir) / "unrelated.txt";
    write(other, "keep");
    adai::cleanup_downloads(
        {one, two, two, fs::path(dcfg_.download_dir) / "gone.jsonl", fs::path()});
    EXPECT_FALSE(fs::exists(one));
    EXPECT_FALSE(fs::exists(DataTransport::checksum_path_for(one)));
    EXPECT_FALSE(fs::exists(two));
    EXPECT_TRUE(fs::exists(other)) << "only the listed downloads are removed";
}
