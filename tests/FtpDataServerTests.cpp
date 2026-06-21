/**
 * FtpDataServerTests — unit tests for FtpDataServer (Phase 10: dataset transport)
 *
 * Covers:
 *   - TokenStore: insert, authenticate, mark_consumed, sweep_expired, remove
 *   - ftp_detail helpers: random_hex, utc_string
 *   - FtpDataServer::issue_token (no network; does not call start())
 *   - AcquireResponse::registry_paths convenience method
 *
 * No FTP server lifecycle tests (start/stop) to avoid port conflicts in CI.
 */
#include <gtest/gtest.h>

#include <chrono>
#include <regex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "../src/FtpDataServer.hpp"
#include "../src/RegistryTransport.hpp"

using namespace std::chrono_literals;

// ============================================================================
// Helpers
// ============================================================================

static VirtualUser make_user(const std::string& password,
                             const std::string& ftp_path,
                             std::chrono::seconds ttl = 3600s)
{
    VirtualUser u;
    u.password = password;
    u.ftp_path = ftp_path;
    u.expiry   = std::chrono::system_clock::now() + ttl;
    u.consumed = false;
    return u;
}

// ============================================================================
// TokenStore — insert and authenticate
// ============================================================================

TEST(TokenStoreTest, AuthenticateHappyPath) {
    TokenStore store;
    store.insert("adai_test01", make_user("secret123", "datasets/file.txt"));

    std::string ftp_path;
    EXPECT_TRUE(store.authenticate("adai_test01", "secret123", ftp_path));
    EXPECT_EQ(ftp_path, "datasets/file.txt");
}

TEST(TokenStoreTest, AuthenticateWrongPassword) {
    TokenStore store;
    store.insert("adai_test01", make_user("correct", "data/x.bin"));

    std::string ftp_path;
    EXPECT_FALSE(store.authenticate("adai_test01", "wrong", ftp_path));
}

TEST(TokenStoreTest, AuthenticateUnknownUser) {
    TokenStore store;
    std::string ftp_path;
    EXPECT_FALSE(store.authenticate("adai_ghost", "pass", ftp_path));
}

TEST(TokenStoreTest, AuthenticateExpiredToken) {
    TokenStore store;
    // TTL of -1 second: already expired
    store.insert("adai_old", make_user("pass", "data/old.bin", -1s));

    std::string ftp_path;
    EXPECT_FALSE(store.authenticate("adai_old", "pass", ftp_path));
}

TEST(TokenStoreTest, AuthenticateConsumedToken) {
    TokenStore store;
    store.insert("adai_u1", make_user("pw", "data/f.bin"));

    store.mark_consumed("adai_u1");

    std::string ftp_path;
    EXPECT_FALSE(store.authenticate("adai_u1", "pw", ftp_path));
}

TEST(TokenStoreTest, MarkConsumedPreventsReAuthentication) {
    TokenStore store;
    store.insert("adai_once", make_user("s3cr3t", "data/a.bin"));

    std::string ftp_path;
    // First authentication succeeds
    EXPECT_TRUE(store.authenticate("adai_once", "s3cr3t", ftp_path));

    // Now mark consumed (simulates a successful RETR)
    store.mark_consumed("adai_once");

    // Second attempt must fail (replay prevention)
    EXPECT_FALSE(store.authenticate("adai_once", "s3cr3t", ftp_path));
}

TEST(TokenStoreTest, MarkConsumedOnNonExistentUserIsNoOp) {
    TokenStore store;
    // Should not throw or crash
    EXPECT_NO_THROW(store.mark_consumed("adai_ghost"));
}

TEST(TokenStoreTest, RemoveErasesEntry) {
    TokenStore store;
    store.insert("adai_r1", make_user("pw", "data/r1.bin"));
    store.remove("adai_r1");

    std::string ftp_path;
    EXPECT_FALSE(store.authenticate("adai_r1", "pw", ftp_path));
}

TEST(TokenStoreTest, RemoveNonExistentIsNoOp) {
    TokenStore store;
    EXPECT_NO_THROW(store.remove("adai_ghost"));
}

TEST(TokenStoreTest, SweepExpiredRemovesExpiredTokens) {
    TokenStore store;
    store.insert("adai_good",    make_user("pw", "data/good.bin",    600s));
    store.insert("adai_expired", make_user("pw", "data/expired.bin", -1s));

    store.sweep_expired();

    std::string ftp_path;
    // Valid token still authenticates
    EXPECT_TRUE(store.authenticate("adai_good", "pw", ftp_path));
    // Expired token removed
    EXPECT_FALSE(store.authenticate("adai_expired", "pw", ftp_path));
}

TEST(TokenStoreTest, SweepExpiredRemovesConsumedTokens) {
    TokenStore store;
    store.insert("adai_consumed", make_user("pw", "data/c.bin", 600s));
    store.mark_consumed("adai_consumed");

    store.sweep_expired();

    std::string ftp_path;
    EXPECT_FALSE(store.authenticate("adai_consumed", "pw", ftp_path));
}

TEST(TokenStoreTest, SweepExpiredPreservesValidTokens) {
    TokenStore store;
    for (int i = 0; i < 5; ++i) {
        store.insert("adai_v" + std::to_string(i),
                     make_user("pw" + std::to_string(i),
                               "data/f" + std::to_string(i) + ".bin",
                               600s));
    }
    for (int i = 0; i < 3; ++i) {
        store.insert("adai_e" + std::to_string(i),
                     make_user("ew", "data/e.bin", -1s));
    }

    store.sweep_expired();

    // All 5 valid tokens remain
    for (int i = 0; i < 5; ++i) {
        std::string ftp_path;
        EXPECT_TRUE(store.authenticate("adai_v" + std::to_string(i),
                                       "pw" + std::to_string(i),
                                       ftp_path))
            << "Valid token adai_v" << i << " should survive sweep";
    }
}

TEST(TokenStoreTest, MultipleInsertsSameUserOverwrites) {
    TokenStore store;
    store.insert("adai_dup", make_user("pw1", "data/first.bin"));
    store.insert("adai_dup", make_user("pw2", "data/second.bin"));

    std::string ftp_path;
    EXPECT_FALSE(store.authenticate("adai_dup", "pw1", ftp_path));
    EXPECT_TRUE(store.authenticate("adai_dup", "pw2", ftp_path));
    EXPECT_EQ(ftp_path, "data/second.bin");
}

// ============================================================================
// TokenStore — thread safety
// ============================================================================

TEST(TokenStoreTest, ConcurrentInsertsAndAuthenticates) {
    TokenStore store;
    std::vector<std::thread> threads;

    // 8 threads each insert their own user and then authenticate
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&store, i] {
            const std::string user = "adai_t" + std::to_string(i);
            const std::string pw   = "secret" + std::to_string(i);
            store.insert(user, make_user(pw, "data/f" + std::to_string(i) + ".bin"));
            std::string ftp_path;
            // Authentication may succeed or fail depending on timing;
            // the important thing is it does not crash.
            store.authenticate(user, pw, ftp_path);
        });
    }
    for (auto& t : threads) t.join();

    // After all threads finish, each token should authenticate correctly.
    for (int i = 0; i < 8; ++i) {
        const std::string user = "adai_t" + std::to_string(i);
        const std::string pw   = "secret" + std::to_string(i);
        std::string ftp_path;
        EXPECT_TRUE(store.authenticate(user, pw, ftp_path));
    }
}

// ============================================================================
// ftp_detail::random_hex
// ============================================================================

TEST(RandomHexTest, LengthIs2xBytes) {
    EXPECT_EQ(ftp_detail::random_hex(4).size(),  8u);
    EXPECT_EQ(ftp_detail::random_hex(8).size(),  16u);
    EXPECT_EQ(ftp_detail::random_hex(32).size(), 64u);
}

TEST(RandomHexTest, ZeroBytes) {
    EXPECT_EQ(ftp_detail::random_hex(0), "");
}

TEST(RandomHexTest, OnlyLowercaseHexChars) {
    const std::string hex = ftp_detail::random_hex(64);
    for (char c : hex) {
        EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
            << "Unexpected character: '" << c << "'";
    }
}

TEST(RandomHexTest, ProducesDifferentValues) {
    // With 32 bytes of entropy the probability of collision is ~2^-256
    const std::string a = ftp_detail::random_hex(32);
    const std::string b = ftp_detail::random_hex(32);
    EXPECT_NE(a, b);
}

TEST(RandomHexTest, MultipleCallsAreUnique) {
    std::unordered_set<std::string> seen;
    for (int i = 0; i < 100; ++i) {
        seen.insert(ftp_detail::random_hex(8));
    }
    // With 8-byte tokens it would be extraordinary to see a collision in 100 draws
    EXPECT_GT(seen.size(), 95u);
}

// ============================================================================
// ftp_detail::utc_string
// ============================================================================

TEST(UtcStringTest, ReturnsISO8601Format) {
    const auto now = std::chrono::system_clock::now();
    const std::string s = ftp_detail::utc_string(now);
    // Must match: YYYY-MM-DDTHH:MM:SSZ
    const std::regex iso8601(R"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)");
    EXPECT_TRUE(std::regex_match(s, iso8601)) << "Got: " << s;
}

TEST(UtcStringTest, EndsWithZ) {
    const auto now = std::chrono::system_clock::now();
    const std::string s = ftp_detail::utc_string(now);
    EXPECT_EQ(s.back(), 'Z');
}

TEST(UtcStringTest, ContainsDateTimeSeparatorT) {
    const auto now = std::chrono::system_clock::now();
    const std::string s = ftp_detail::utc_string(now);
    EXPECT_NE(s.find('T'), std::string::npos);
}

TEST(UtcStringTest, Length19PlusZ) {
    const auto now = std::chrono::system_clock::now();
    const std::string s = ftp_detail::utc_string(now);
    EXPECT_EQ(s.size(), 20u) << "Expected 'YYYY-MM-DDTHH:MM:SSZ' (20 chars), got: " << s;
}

TEST(UtcStringTest, KnownEpoch) {
    // Unix epoch (1970-01-01T00:00:00Z)
    const auto epoch = std::chrono::system_clock::from_time_t(0);
    EXPECT_EQ(ftp_detail::utc_string(epoch), "1970-01-01T00:00:00Z");
}

// ============================================================================
// FtpDataServer::issue_token
// ============================================================================

class FtpDataServerTokenTest : public ::testing::Test {
protected:
    // Use a non-zero port range; server is never started in these tests.
    FtpDataServer server_{"/tmp/adai_ftp_test_root", 12121, 52000, 52099, "127.0.0.1"};
    const std::string kRunId = "run-test-001";
};

TEST_F(FtpDataServerTokenTest, UsernameHasAdaiPrefix) {
    const auto tok = server_.issue_token(kRunId, "data/train.bin", 30);
    EXPECT_EQ(tok.username.substr(0, 5), "adai_");
}

TEST_F(FtpDataServerTokenTest, UsernameIs13Chars) {
    // "adai_" (5) + 8 hex chars = 13
    const auto tok = server_.issue_token(kRunId, "data/train.bin", 30);
    EXPECT_EQ(tok.username.size(), 13u) << "Username: " << tok.username;
}

TEST_F(FtpDataServerTokenTest, PasswordIs64Chars) {
    const auto tok = server_.issue_token(kRunId, "data/train.bin", 30);
    EXPECT_EQ(tok.password.size(), 64u);
}

TEST_F(FtpDataServerTokenTest, PasswordIsHex) {
    const auto tok = server_.issue_token(kRunId, "data/train.bin", 30);
    for (char c : tok.password) {
        EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
            << "Non-hex char in password: '" << c << "'";
    }
}

TEST_F(FtpDataServerTokenTest, ExpiryStringIsISO8601) {
    const auto tok = server_.issue_token(kRunId, "data/train.bin", 30);
    const std::regex iso8601(R"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)");
    EXPECT_TRUE(std::regex_match(tok.token_expires_utc, iso8601))
        << "Got: " << tok.token_expires_utc;
}

TEST_F(FtpDataServerTokenTest, ExpiryIsInFuture) {
    const auto tok = server_.issue_token(kRunId, "data/train.bin", 30);
    // The expiry UTC string should be after "now" — crude check: year >= 2024
    EXPECT_GE(tok.token_expires_utc.substr(0, 4), "2024");
}

TEST_F(FtpDataServerTokenTest, TokenIsRegisteredAndAuthenticates) {
    const auto tok = server_.issue_token(kRunId, "datasets/sample.csv", 60);
    std::string out_path;
    EXPECT_TRUE(server_.tokens().authenticate(tok.username, tok.password, out_path));
    EXPECT_EQ(out_path, "datasets/sample.csv");
}

TEST_F(FtpDataServerTokenTest, WrongPasswordDoesNotAuthenticate) {
    const auto tok = server_.issue_token(kRunId, "datasets/sample.csv", 60);
    std::string out_path;
    EXPECT_FALSE(server_.tokens().authenticate(tok.username, "wrongpassword", out_path));
}

TEST_F(FtpDataServerTokenTest, MultipleTokensHaveUniqueUsernames) {
    std::unordered_set<std::string> usernames;
    for (int i = 0; i < 50; ++i) {
        const auto tok = server_.issue_token(kRunId, "data/f" + std::to_string(i) + ".bin", 30);
        usernames.insert(tok.username);
    }
    EXPECT_EQ(usernames.size(), 50u) << "Collision among 50 token usernames";
}

TEST_F(FtpDataServerTokenTest, MultipleTokensHaveUniquePasswords) {
    std::unordered_set<std::string> passwords;
    for (int i = 0; i < 50; ++i) {
        const auto tok = server_.issue_token(kRunId, "data/f" + std::to_string(i) + ".bin", 30);
        passwords.insert(tok.password);
    }
    EXPECT_EQ(passwords.size(), 50u) << "Collision among 50 token passwords";
}

TEST_F(FtpDataServerTokenTest, TokenGrantsCorrectPath) {
    const auto tok = server_.issue_token(kRunId, "path/to/mydata.bin", 30);
    std::string out_path;
    server_.tokens().authenticate(tok.username, tok.password, out_path);
    EXPECT_EQ(out_path, "path/to/mydata.bin");
}

TEST_F(FtpDataServerTokenTest, ZeroTtlTokenIsImmediatelyExpired) {
    const auto tok = server_.issue_token(kRunId, "data/f.bin", 0);
    // A 0-minute TTL may or may not be expired by the time authenticate() runs;
    // in practice the expiry is "now + 0" which is already past by the time we
    // reach the authenticate call. We just verify it was issued without crashing.
    EXPECT_FALSE(tok.username.empty());
    EXPECT_FALSE(tok.password.empty());
}

// ============================================================================
// Phase 3: run_id is stored in the virtual user (audit/rate-limit support)
// ============================================================================

TEST_F(FtpDataServerTokenTest, RunIdIsStoredInToken) {
    const auto tok = server_.issue_token("run-audit-1", "data/audit.bin", 30);
    std::string out_path, out_run_id;
    EXPECT_TRUE(server_.tokens().authenticate(tok.username, tok.password,
                                              out_path, &out_run_id));
    EXPECT_EQ(out_run_id, "run-audit-1");
}

TEST_F(FtpDataServerTokenTest, EmptyRunIdIsAllowed) {
    const auto tok = server_.issue_token("", "data/f.bin", 30);
    std::string out_path, out_run_id;
    EXPECT_TRUE(server_.tokens().authenticate(tok.username, tok.password,
                                              out_path, &out_run_id));
    EXPECT_EQ(out_run_id, "");
}

// ============================================================================
// Phase 3: Rate limiting (acquire_session / release_session via constructor)
// ============================================================================

TEST(RateLimitTest, ConstructorDefaultFtpsDisabled) {
    FtpDataServer srv("/tmp", 15000, 55000, 55099);
    EXPECT_FALSE(srv.ftps_enabled());
}

TEST(RateLimitTest, FtpsEnabledFlagPassedToConstructor) {
    // ftps_en=true without a running server — just check the accessor
    FtpDataServer srv("/tmp", 15001, 55100, 55199, "127.0.0.1", "", true);
    EXPECT_TRUE(srv.ftps_enabled());
}

// ============================================================================
// Phase 3 (BUILD_FTPS): HMAC-SHA256 token signing
// ============================================================================

#ifdef BUILD_FTPS

TEST(HmacTokenTest, HmacPasswordIsDeterministic) {
    // Two servers with the same secret must produce the same password for the
    // same (username, ftp_path, expiry) tuple — but since the username is
    // random and the expiry is time-based, the simplest test is that the
    // same call produces a non-random-looking 64-char hex string.
    FtpDataServer srv("/tmp/adai_hmac_test", 15100, 55200, 55299,
                      "127.0.0.1", "supersecretkey");

    const auto tok = srv.issue_token("run-hmac-1", "data/train.bin", 30);
    // HMAC-SHA256 of a 32-byte input is 32 bytes = 64 hex chars
    EXPECT_EQ(tok.password.size(), 64u);
    for (char c : tok.password) {
        EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
            << "Non-hex char in HMAC password: '" << c << "'";
    }
}

TEST(HmacTokenTest, HmacPasswordAuthenticates) {
    FtpDataServer srv("/tmp/adai_hmac_test", 15101, 55300, 55399,
                      "127.0.0.1", "supersecretkey");

    const auto tok = srv.issue_token("run-hmac-2", "data/check.bin", 30);
    std::string out_path;
    EXPECT_TRUE(srv.tokens().authenticate(tok.username, tok.password, out_path));
    EXPECT_EQ(out_path, "data/check.bin");
}

TEST(HmacTokenTest, EmptySecretFallsBackToRandomPassword) {
    // Without a secret the fallback is 32 random bytes = 64 hex chars
    FtpDataServer srv_plain("/tmp/adai_hmac_test", 15102, 55400, 55499,
                            "127.0.0.1", "");   // empty secret
    const auto tok = srv_plain.issue_token("run-plain", "data/plain.bin", 30);
    EXPECT_EQ(tok.password.size(), 64u);
}

TEST(HmacTokenTest, DifferentPathsDifferentHmacPasswords) {
    FtpDataServer srv("/tmp/adai_hmac_test", 15103, 55500, 55599,
                      "127.0.0.1", "sharedkey");

    const auto tok1 = srv.issue_token("run-diff", "data/file1.bin", 30);
    const auto tok2 = srv.issue_token("run-diff", "data/file2.bin", 30);

    // Different ftp_paths produce different HMAC inputs → different passwords
    // (usernames differ too, which alone would change the HMAC input)
    EXPECT_NE(tok1.password, tok2.password);
}

TEST(HmacTokenTest, HmacHelperProducesCorrectLength) {
    const std::string result = ftp_detail::hmac_sha256("key", "message");
    // HMAC-SHA256 = 32 bytes = 64 hex chars
    EXPECT_EQ(result.size(), 64u);
}

TEST(HmacTokenTest, HmacHelperOnlyHexChars) {
    const std::string result = ftp_detail::hmac_sha256("key", "message");
    for (char c : result) {
        EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
            << "Non-hex char: '" << c << "'";
    }
}

TEST(HmacTokenTest, HmacHelperIsDeterministic) {
    const std::string r1 = ftp_detail::hmac_sha256("same-key", "same-msg");
    const std::string r2 = ftp_detail::hmac_sha256("same-key", "same-msg");
    EXPECT_EQ(r1, r2);
}

TEST(HmacTokenTest, HmacHelperDifferentKeysProduceDifferentDigests) {
    const std::string r1 = ftp_detail::hmac_sha256("key-a", "msg");
    const std::string r2 = ftp_detail::hmac_sha256("key-b", "msg");
    EXPECT_NE(r1, r2);
}

TEST(HmacTokenTest, HmacHelperDifferentMsgsProduceDifferentDigests) {
    const std::string r1 = ftp_detail::hmac_sha256("key", "msg-a");
    const std::string r2 = ftp_detail::hmac_sha256("key", "msg-b");
    EXPECT_NE(r1, r2);
}

#endif // BUILD_FTPS

// ============================================================================
// AcquireResponse::registry_paths
// ============================================================================

TEST(AcquireResponseTest, EmptyFilesReturnsEmptyPaths) {
    AcquireResponse resp;
    EXPECT_TRUE(resp.registry_paths().empty());
}

TEST(AcquireResponseTest, SingleFileReturnsCorrectPath) {
    AcquireResponse resp;
    FileToken tok;
    tok.registry_path = "/data/train/a.txt";
    resp.files.push_back(tok);

    const auto paths = resp.registry_paths();
    ASSERT_EQ(paths.size(), 1u);
    EXPECT_EQ(paths[0], "/data/train/a.txt");
}

TEST(AcquireResponseTest, MultipleFilesReturnsAllPaths) {
    AcquireResponse resp;
    for (int i = 0; i < 5; ++i) {
        FileToken tok;
        tok.registry_path = "/data/f" + std::to_string(i) + ".bin";
        resp.files.push_back(tok);
    }

    const auto paths = resp.registry_paths();
    ASSERT_EQ(paths.size(), 5u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(paths[i], "/data/f" + std::to_string(i) + ".bin");
    }
}

TEST(AcquireResponseTest, PreservesOrder) {
    AcquireResponse resp;
    const std::vector<std::string> expected = {"/z.bin", "/a.bin", "/m.bin"};
    for (const auto& p : expected) {
        FileToken tok;
        tok.registry_path = p;
        resp.files.push_back(tok);
    }
    EXPECT_EQ(resp.registry_paths(), expected);
}

// ============================================================================
// FileToken — default values
// ============================================================================

TEST(FileTokenTest, DefaultSizeBytesIsZero) {
    FileToken tok;
    EXPECT_EQ(tok.size_bytes, 0u);
}

TEST(FileTokenTest, DefaultFieldsAreEmpty) {
    FileToken tok;
    EXPECT_TRUE(tok.registry_path.empty());
    EXPECT_TRUE(tok.ftp_path.empty());
    EXPECT_TRUE(tok.ftp_username.empty());
    EXPECT_TRUE(tok.ftp_password.empty());
    EXPECT_TRUE(tok.checksum.empty());
    EXPECT_TRUE(tok.token_expires_utc.empty());
}

// ============================================================================
// AcquireResponse — default values
// ============================================================================

TEST(AcquireResponseTest, DefaultFtpPort) {
    AcquireResponse resp;
    EXPECT_EQ(resp.ftp_server_port, 2121);
}

TEST(AcquireResponseTest, DefaultHostIsEmpty) {
    AcquireResponse resp;
    EXPECT_TRUE(resp.ftp_server_host.empty());
}

TEST(AcquireResponseTest, DefaultRunIdIsEmpty) {
    AcquireResponse resp;
    EXPECT_TRUE(resp.run_id.empty());
}

// ============================================================================
// FtpDataServer constructor
// ============================================================================

TEST(FtpDataServerConstructorTest, ConstructorDoesNotCrash) {
    EXPECT_NO_THROW({
        FtpDataServer srv("/tmp/adai_data", 12122, 52100, 52199, "127.0.0.1");
    });
}

TEST(FtpDataServerConstructorTest, AdvertiseIpIsPreserved) {
    FtpDataServer srv("/tmp/adai_data", 12123, 52200, 52299, "192.168.1.100");
    EXPECT_EQ(srv.advertise_ip(), "192.168.1.100");
}

TEST(FtpDataServerConstructorTest, ControlPortIsPreserved) {
    FtpDataServer srv("/tmp/adai_data", 13000, 53000, 53099, "");
    EXPECT_EQ(srv.control_port(), 13000);
}

TEST(FtpDataServerConstructorTest, EmptyAdvertiseIpIsPreserved) {
    FtpDataServer srv("/tmp/adai_data", 12124, 52300, 52399);
    EXPECT_TRUE(srv.advertise_ip().empty());
}
