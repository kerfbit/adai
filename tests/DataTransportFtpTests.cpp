/**
 * DataTransportFtpTests — Phase 2 unit tests for DataTransport.
 *
 * Compiled WITH BUILD_FTP_TRANSPORT so the real libcurl implementation is
 * exercised.  All tests avoid making actual network connections by exploiting
 * the Phase 2 early-exit logic (Condition B) or by using empty AcquireResponse
 * objects.
 *
 * Requires: ENABLE_FTP_TRANSPORT=ON and libcurl present at configure time.
 * BUILD_FTP_TRANSPORT is supplied via target_compile_definitions in CMakeLists.
 */
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../src/DataTransport.hpp"
#include "../src/RegistryTransport.hpp"

namespace fs = std::filesystem;

// ============================================================================
// Helpers
// ============================================================================

static fs::path make_temp_dir(const std::string& suffix) {
    fs::path p = fs::temp_directory_path() / ("adai_dt_ftp_test_" + suffix);
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

static void write_bytes(const fs::path& p, std::size_t n_bytes, char fill = 'X') {
    std::ofstream f(p, std::ios::binary);
    std::string content(n_bytes, fill);
    f.write(content.data(), static_cast<std::streamsize>(n_bytes));
}

static FileToken make_token(const std::string& ftp_path, std::size_t size_bytes) {
    FileToken tok;
    tok.registry_path     = "/registry/" + ftp_path;
    tok.ftp_path          = ftp_path;
    tok.ftp_username      = "adai_deadbeef";
    tok.ftp_password      = std::string(64, 'f');
    tok.size_bytes        = size_bytes;
    tok.token_expires_utc = "2099-01-01T00:00:00Z";
    return tok;
}

// ============================================================================
// DataTransport constants
// ============================================================================

TEST(DataTransportConstantsTest, MaxRetriesIsThree) {
    EXPECT_EQ(DataTransport::kMaxRetries, 3);
}

TEST(DataTransportConstantsTest, BaseBackoffIsOneSecond) {
    EXPECT_EQ(DataTransport::kBaseBackoffMs, 1000);
}

// ============================================================================
// Condition B: fetch() skips re-download when file already matches size_bytes
// ============================================================================

TEST(DataTransportConditionBTest, SkipsDownloadForCompleteFile) {
    const auto dir = make_temp_dir("cond_b_skip");
    const std::string ftp_path  = "train.bin";
    const std::size_t file_size = 1024;

    // Pre-create a "complete" file matching the expected size
    write_bytes(dir / ftp_path, file_size);

    FileToken tok   = make_token(ftp_path, file_size);
    DataTransport dt;

    // host:port "127.0.0.1:19999" — nothing is listening there.
    // If Condition B is working, fetch() returns BEFORE making any connection.
    fs::path result;
    EXPECT_NO_THROW(result = dt.fetch(tok, "127.0.0.1", 19999, dir));
    EXPECT_EQ(result, dir / ftp_path);
    EXPECT_TRUE(fs::exists(result));
    EXPECT_EQ(fs::file_size(result), file_size);

    fs::remove_all(dir);
}

TEST(DataTransportConditionBTest, DoesNotSkipWhenSizeMismatch) {
    const auto dir = make_temp_dir("cond_b_mismatch");
    const std::string ftp_path  = "partial.bin";
    const std::size_t file_size = 1024;
    const std::size_t disk_size = 512;  // only half downloaded

    // Pre-create a partial file
    write_bytes(dir / ftp_path, disk_size);

    FileToken tok = make_token(ftp_path, file_size);
    DataTransport dt;

    // Condition B should NOT fire (sizes differ).
    // fetch() will try to make an FTP connection — which fails since nothing
    // is listening at 19999.  That error is expected.
    EXPECT_THROW(dt.fetch(tok, "127.0.0.1", 19999, dir), std::runtime_error);

    fs::remove_all(dir);
}

TEST(DataTransportConditionBTest, DoesNotSkipWhenSizeBytesIsZero) {
    // When size_bytes == 0 we cannot determine completeness: Condition B
    // must NOT fire regardless of what's on disk.
    const auto dir = make_temp_dir("cond_b_zerosize");
    const std::string ftp_path = "unknown_size.bin";

    write_bytes(dir / ftp_path, 1024);  // file exists, non-zero

    FileToken tok = make_token(ftp_path, 0 /* size unknown */);
    DataTransport dt;

    // Should attempt a connection (and fail) — not skip
    EXPECT_THROW(dt.fetch(tok, "127.0.0.1", 19999, dir), std::runtime_error);

    fs::remove_all(dir);
}

TEST(DataTransportConditionBTest, CreatesDownloadDirIfAbsent) {
    const auto base    = fs::temp_directory_path() / "adai_dt_ftp_test_mkdir";
    const auto dir     = base / "nested" / "download";
    fs::remove_all(base);

    const std::string ftp_path  = "f.bin";
    const std::size_t file_size = 256;

    // Manually create the file in the expected location so Condition B fires
    fs::create_directories(dir);
    write_bytes(dir / ftp_path, file_size);

    FileToken tok   = make_token(ftp_path, file_size);
    DataTransport dt;

    fs::path result;
    EXPECT_NO_THROW(result = dt.fetch(tok, "127.0.0.1", 19999, dir));
    EXPECT_TRUE(fs::exists(result));

    fs::remove_all(base);
}

// ============================================================================
// Condition B: multiple files — fetch_all returns all pre-downloaded paths
// ============================================================================

TEST(DataTransportFetchAllTest, FetchAllReturnsPreDownloadedPaths) {
    const auto dir = make_temp_dir("fetch_all_predown");

    AcquireResponse resp;
    resp.ftp_server_host = "127.0.0.1";
    resp.ftp_server_port = 19999;  // nothing listening

    const std::vector<std::pair<std::string, std::size_t>> files = {
        {"a.bin", 100}, {"b.bin", 200}, {"c.bin", 300}
    };

    for (const auto& [name, sz] : files) {
        write_bytes(dir / name, sz);
        resp.files.push_back(make_token(name, sz));
    }

    DataTransport dt;
    std::vector<fs::path> result;
    EXPECT_NO_THROW(result = dt.fetch_all(resp, dir));

    ASSERT_EQ(result.size(), 3u);
    for (std::size_t i = 0; i < files.size(); ++i) {
        EXPECT_EQ(result[i].filename().string(), files[i].first);
        EXPECT_TRUE(fs::exists(result[i]));
    }

    fs::remove_all(dir);
}

TEST(DataTransportFetchAllTest, FetchAllReturnsInOrder) {
    const auto dir = make_temp_dir("fetch_all_order");

    AcquireResponse resp;
    resp.ftp_server_host = "127.0.0.1";
    resp.ftp_server_port = 19999;

    for (int i = 0; i < 8; ++i) {
        const std::string name = "file" + std::to_string(i) + ".bin";
        write_bytes(dir / name, static_cast<std::size_t>((i + 1) * 128));
        resp.files.push_back(make_token(name, static_cast<std::size_t>((i + 1) * 128)));
    }

    DataTransport dt;
    std::vector<fs::path> result;
    EXPECT_NO_THROW(result = dt.fetch_all(resp, dir, /*max_parallel=*/4));

    ASSERT_EQ(result.size(), 8u);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(result[i].filename().string(), "file" + std::to_string(i) + ".bin");
    }

    fs::remove_all(dir);
}

TEST(DataTransportFetchAllTest, FetchAllEmptyResponseReturnsEmpty) {
    const auto dir = make_temp_dir("fetch_all_empty");
    AcquireResponse resp;

    DataTransport dt;
    EXPECT_TRUE(dt.fetch_all(resp, dir).empty());

    fs::remove_all(dir);
}

TEST(DataTransportFetchAllTest, FetchAllPropagatesFirstError) {
    const auto dir = make_temp_dir("fetch_all_err");

    AcquireResponse resp;
    resp.ftp_server_host = "127.0.0.1";
    resp.ftp_server_port = 19999;

    // One file is complete (Condition B will pass), one is missing
    // (will fail with a network error because nothing is at 19999).
    write_bytes(dir / "ok.bin", 100);
    resp.files.push_back(make_token("ok.bin", 100));

    // This file is NOT pre-created so fetch() will attempt a connection and fail
    resp.files.push_back(make_token("missing.bin", 200));

    DataTransport dt;
    EXPECT_THROW(dt.fetch_all(resp, dir), std::runtime_error);

    fs::remove_all(dir);
}

// ============================================================================
// Parallel fetch_all with max_parallel=1 behaves like sequential
// ============================================================================

TEST(DataTransportFetchAllTest, SingleThreadedFetchAllPreservesOrder) {
    const auto dir = make_temp_dir("fetch_all_seq");

    AcquireResponse resp;
    resp.ftp_server_host = "127.0.0.1";
    resp.ftp_server_port = 19999;

    for (int i = 0; i < 5; ++i) {
        const std::string name = "seq" + std::to_string(i) + ".bin";
        write_bytes(dir / name, static_cast<std::size_t>((i + 1) * 64));
        resp.files.push_back(make_token(name, static_cast<std::size_t>((i + 1) * 64)));
    }

    DataTransport dt;
    std::vector<fs::path> result;
    EXPECT_NO_THROW(result = dt.fetch_all(resp, dir, /*max_parallel=*/1));

    ASSERT_EQ(result.size(), 5u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(result[i].filename().string(), "seq" + std::to_string(i) + ".bin");
    }

    fs::remove_all(dir);
}

// ============================================================================
// fetch() download directory creation
// ============================================================================

TEST(DataTransportFetchTest, CreatesMissingDownloadDir) {
    const auto base = fs::temp_directory_path() / "adai_dt_ftp_mkdir";
    const auto dir  = base / "a" / "b" / "c";
    fs::remove_all(base);

    // Pre-create the file (Condition B) so no network connection is needed
    fs::create_directories(dir);
    write_bytes(dir / "f.bin", 512);

    FileToken tok = make_token("f.bin", 512);
    DataTransport dt;

    EXPECT_NO_THROW(dt.fetch(tok, "127.0.0.1", 19999, dir));
    EXPECT_TRUE(fs::exists(dir / "f.bin"));

    fs::remove_all(base);
}

// ============================================================================
// FileToken — ftp_path basename extraction
// ============================================================================

TEST(DataTransportFetchTest, UsesBasenameOfFtpPath) {
    const auto dir = make_temp_dir("basename_test");

    // ftp_path has directory components: only the basename ends up in download_dir
    FileToken tok       = make_token("subdir/train.csv", 256);
    const fs::path expected = dir / "train.csv";

    write_bytes(expected, 256);

    DataTransport dt;
    fs::path result;
    EXPECT_NO_THROW(result = dt.fetch(tok, "127.0.0.1", 19999, dir));
    EXPECT_EQ(result, expected);

    fs::remove_all(dir);
}

// ============================================================================
// dt_detail::is_transient() — CURLcode classification
//
// Transient errors are worth retrying (temporary network issues).
// Non-transient errors indicate a permanent failure (auth denied, server
// absent, bad URL, etc.) and must not be retried.
// ============================================================================

TEST(IsTransientTest, OperationTimedOutIsTransient) {
    EXPECT_TRUE(dt_detail::is_transient(CURLE_OPERATION_TIMEDOUT));
}

TEST(IsTransientTest, RecvErrorIsTransient) {
    EXPECT_TRUE(dt_detail::is_transient(CURLE_RECV_ERROR));
}

TEST(IsTransientTest, SendErrorIsTransient) {
    EXPECT_TRUE(dt_detail::is_transient(CURLE_SEND_ERROR));
}

TEST(IsTransientTest, GotNothingIsTransient) {
    EXPECT_TRUE(dt_detail::is_transient(CURLE_GOT_NOTHING));
}

TEST(IsTransientTest, PartialFileIsTransient) {
    EXPECT_TRUE(dt_detail::is_transient(CURLE_PARTIAL_FILE));
}

TEST(IsTransientTest, FtpAcceptTimeoutIsTransient) {
    EXPECT_TRUE(dt_detail::is_transient(CURLE_FTP_ACCEPT_TIMEOUT));
}

TEST(IsTransientTest, OkIsNotTransient) {
    EXPECT_FALSE(dt_detail::is_transient(CURLE_OK));
}

// Connection refused means the server is absent — not worth retrying.
TEST(IsTransientTest, CouldntConnectIsNotTransient) {
    EXPECT_FALSE(dt_detail::is_transient(CURLE_COULDNT_CONNECT));
}

TEST(IsTransientTest, LoginDeniedIsNotTransient) {
    EXPECT_FALSE(dt_detail::is_transient(CURLE_LOGIN_DENIED));
}

TEST(IsTransientTest, RemoteAccessDeniedIsNotTransient) {
    EXPECT_FALSE(dt_detail::is_transient(CURLE_REMOTE_ACCESS_DENIED));
}

TEST(IsTransientTest, UnsupportedProtocolIsNotTransient) {
    EXPECT_FALSE(dt_detail::is_transient(CURLE_UNSUPPORTED_PROTOCOL));
}

TEST(IsTransientTest, UrlMalformatIsNotTransient) {
    EXPECT_FALSE(dt_detail::is_transient(CURLE_URL_MALFORMAT));
}
