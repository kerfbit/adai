/**
 * DataTransportTests — unit tests for DataTransport (Phase 10: dataset transport)
 *
 * Compiled WITHOUT BUILD_FTP_TRANSPORT so only the stub implementations are
 * exercised.  The stubs exist to give a clear error when the trainer binary is
 * built without libcurl; both methods must throw std::runtime_error immediately.
 *
 * Also covers FileToken and AcquireResponse struct field defaults (which are
 * independent of the libcurl macro).
 */
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

// Do NOT define BUILD_FTP_TRANSPORT — stubs are what we're testing.
#include "../src/DataTransport.hpp"

// ============================================================================
// Stub behaviour — both methods must throw when libcurl is absent
// ============================================================================

#ifndef BUILD_FTP_TRANSPORT

TEST(DataTransportStubTest, FetchThrowsRuntimeError) {
    DataTransport dt;
    FileToken tok;
    tok.ftp_path = "data/train.bin";
    tok.ftp_username = "adai_deadbeef";
    tok.ftp_password = std::string(64, 'a');
    tok.size_bytes = 1024;

    EXPECT_THROW(dt.fetch(tok, "127.0.0.1", 2121, "/tmp/adai_dt_test"), std::runtime_error);
}

TEST(DataTransportStubTest, FetchErrorMessageMentionsBuildFlag) {
    DataTransport dt;
    FileToken tok;
    try {
        dt.fetch(tok, "127.0.0.1", 2121, "/tmp/adai_dt_test");
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("BUILD_FTP_TRANSPORT"), std::string::npos)
            << "Error should mention the missing macro; got: " << msg;
    }
}

TEST(DataTransportStubTest, FetchAllThrowsRuntimeError) {
    DataTransport dt;
    AcquireResponse resp;
    resp.ftp_server_host = "127.0.0.1";
    resp.ftp_server_port = 2121;
    FileToken tok;
    tok.ftp_path = "data/train.bin";
    resp.files.push_back(tok);

    EXPECT_THROW(dt.fetch_all(resp, "/tmp/adai_dt_test"), std::runtime_error);
}

TEST(DataTransportStubTest, FetchAllErrorMessageMentionsBuildFlag) {
    DataTransport dt;
    AcquireResponse resp;
    resp.ftp_server_host = "127.0.0.1";
    FileToken tok;
    resp.files.push_back(tok);
    try {
        dt.fetch_all(resp, "/tmp/adai_dt_test");
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("BUILD_FTP_TRANSPORT"), std::string::npos)
            << "Error should mention the missing macro; got: " << msg;
    }
}

#else  // BUILD_FTP_TRANSPORT is defined — test what we can without a real server

// When libcurl is present, fetch_all on an empty AcquireResponse must return
// an empty vector without making any network calls.
TEST(DataTransportTest, FetchAllEmptyResponseReturnsEmpty) {
    DataTransport dt;
    AcquireResponse resp;
    // Empty files list — no FTP calls are made.
    const auto result = dt.fetch_all(resp, "/tmp/adai_dt_test");
    EXPECT_TRUE(result.empty());
}

#endif  // BUILD_FTP_TRANSPORT

// ============================================================================
// FileToken struct — default field values
// ============================================================================

TEST(FileTokenStructTest, DefaultSizeBytesIsZero) {
    FileToken tok;
    EXPECT_EQ(tok.size_bytes, 0u);
}

TEST(FileTokenStructTest, DefaultStringsAreEmpty) {
    FileToken tok;
    EXPECT_TRUE(tok.registry_path.empty());
    EXPECT_TRUE(tok.ftp_path.empty());
    EXPECT_TRUE(tok.ftp_username.empty());
    EXPECT_TRUE(tok.ftp_password.empty());
    EXPECT_TRUE(tok.checksum.empty());
    EXPECT_TRUE(tok.token_expires_utc.empty());
}

TEST(FileTokenStructTest, AssignmentRoundTrips) {
    FileToken tok;
    tok.registry_path = "/srv/data/train.csv";
    tok.ftp_path = "train.csv";
    tok.ftp_username = "adai_deadbeef";
    tok.ftp_password = std::string(64, 'f');
    tok.checksum = "12345_1718900000";
    tok.size_bytes = 102400;
    tok.token_expires_utc = "2026-12-31T23:59:59Z";

    EXPECT_EQ(tok.registry_path, "/srv/data/train.csv");
    EXPECT_EQ(tok.ftp_path, "train.csv");
    EXPECT_EQ(tok.ftp_username, "adai_deadbeef");
    EXPECT_EQ(tok.ftp_password.size(), 64u);
    EXPECT_EQ(tok.size_bytes, 102400u);
    EXPECT_EQ(tok.token_expires_utc, "2026-12-31T23:59:59Z");
}

// ============================================================================
// AcquireResponse struct — default field values
// ============================================================================

TEST(AcquireResponseStructTest, DefaultPortIs2121) {
    AcquireResponse resp;
    EXPECT_EQ(resp.ftp_server_port, 2121);
}

TEST(AcquireResponseStructTest, DefaultHostAndRunIdAreEmpty) {
    AcquireResponse resp;
    EXPECT_TRUE(resp.ftp_server_host.empty());
    EXPECT_TRUE(resp.run_id.empty());
}

TEST(AcquireResponseStructTest, DefaultFilesIsEmpty) {
    AcquireResponse resp;
    EXPECT_TRUE(resp.files.empty());
}

TEST(AcquireResponseStructTest, RegistryPathsReturnsAllPaths) {
    AcquireResponse resp;
    resp.ftp_server_host = "ftp.example.com";
    resp.ftp_server_port = 2121;
    resp.run_id = "run-42";

    for (int i = 0; i < 3; ++i) {
        FileToken tok;
        tok.registry_path = "/data/f" + std::to_string(i) + ".bin";
        tok.ftp_path = "f" + std::to_string(i) + ".bin";
        resp.files.push_back(tok);
    }

    const auto paths = resp.registry_paths();
    ASSERT_EQ(paths.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(paths[i], "/data/f" + std::to_string(i) + ".bin");
    }
}

TEST(AcquireResponseStructTest, RegistryPathsOnEmptyReturnsEmpty) {
    AcquireResponse resp;
    EXPECT_TRUE(resp.registry_paths().empty());
}

// ============================================================================
// DataTransport — construction
// ============================================================================

TEST(DataTransportConstructionTest, DefaultConstructionDoesNotCrash) {
    EXPECT_NO_THROW({ DataTransport dt; });
}
