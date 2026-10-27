// @adai-status: beta        (TD-040 fully resolved — see below)
// @adai-version: 0.12.1
// @adai-reviewed: 2026-10-27

/**
 * FtpDataServer — embedded read-only FTP server for dataset delivery.
 *
 * Runs as a sub-service of registry_server on a configurable port (default 2121).
 * Each file acquired by a trainer receives an independent per-file virtual FTP
 * user that exists only for the token's TTL.  Only RETR is permitted; all write
 * commands are refused.  PASV mode is supported for NAT traversal.
 *
 * Phase 3 — Security hardening:
 *   - FTPS (explicit TLS via AUTH TLS): encrypt credentials and data in transit.
 *     Enabled when BUILD_FTPS is defined and ftps_enabled=true.  A self-signed
 *     certificate is generated at startup when no cert/key files are provided.
 *   - HMAC-SHA256 token signing: token passwords are derived as
 *     HMAC-SHA256(server_secret, username + ":" + ftp_path + ":" + expiry_utc)
 *     so that forged tokens are rejected even if the issuance algorithm is known.
 *   - Rate limiting: at most max_sessions_per_run concurrent authenticated
 *     FTP sessions per run_id.  Excess connections receive 421.
 *   - Audit log: every token issuance, FTP login, RETR start/complete, and
 *     token expiry is logged with run_id, ftp_path, and bytes transferred.
 *
 * TD-040 security review (September 8, 2026): cmd_retr()'s exact-match check
 * against st.allowed_path makes client-side path traversal via the RETR
 * argument impossible, credentials are never written to the audit log, and
 * token validation/consumption is race-free under TokenStore's mutex. The one
 * confirmed gap was upstream of this file, in the caller: RegistryServer.cpp's
 * handle_acquire() minted a token via issue_token() below for any acquired
 * entry with no containment check, so a pending entry referencing a file
 * outside the registry's data_dir got a token whose allowed_path legitimately
 * escaped data_dir via "../" segments. Reproduced end-to-end (September 11,
 * 2026): a real minted token for such an entry successfully RETR'd a file
 * outside data_dir over a live FTP session. Fixed in handle_acquire() itself
 * (an entry that doesn't resolve under data_dir is now never claimed for FTP
 * delivery in the first place — this class needed no change). TD-040 is now fully resolved:
 * random_hex() below draws from a CSPRNG (the OS's since TD-233), RegistryServer.cpp's handle_pending_add() logs
 * a defense-in-depth warning for an out-of-tree path, and tests/registry_ftp_confinement_test.cpp
 * is a permanent regression test for the fix — see TECHNICAL_DEBT_RESOLVED.md for the full
 * writeup.
 *
 * TD-161 (resolved September 13, 2026): this file's raw socket calls now go through
 * PortableSocket.hpp instead of assuming POSIX (`<arpa/inet.h>`, `int` file descriptors,
 * `::close()`) directly, so it — and the registry_server binary that embeds it — build and run
 * on Windows/MinGW. Verified with a real end-to-end FTP session (HTTP `pending/add` →
 * `acquire` → a genuine FTP `USER`/`PASS`/`PASV`/`RETR` exchange, byte-for-byte correct
 * transfer) against `registry_server.exe` cross-compiled for Windows and run under Wine — see
 * TECHNICAL_DEBT_RESOLVED.md for the full writeup.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Logger.hpp"
#include "PortableSocket.hpp"  // TD-161: socket_t, close_socket(), WinsockGuard, etc.
#include "PortableTime.hpp"
#include "SecureRandom.hpp"  // TD-233

// Phase 3: OpenSSL for FTPS (TLS) and HMAC-SHA256 token signing
#ifdef BUILD_FTPS
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#endif

namespace fs = std::filesystem;
using adai::Logger;

// ============================================================================
// Virtual user record
// ============================================================================

struct VirtualUser {
    std::string password;
    std::string ftp_path;  ///< path relative to data_dir that this user may RETR
    std::string run_id;    ///< Phase 3: owning run_id (for audit log and rate limiting)
    std::chrono::system_clock::time_point expiry;
    bool consumed = false;  ///< set true after a successful RETR to prevent replay
};

// ============================================================================
// Thread-safe in-memory token store
// ============================================================================

class TokenStore {
   public:
    // TODO: See TD-377 in TECHNICAL_DEBT.md - silently overwrites on a (32-bit) username collision
    void insert(const std::string& username, VirtualUser user) {
        std::lock_guard<std::mutex> lk(mtx_);
        users_[username] = std::move(user);
    }

    /**
     * Validate credentials.  Returns true when the token is valid, not expired,
     * and has not been consumed.  Fills @p out_ftp_path and optionally @p out_run_id.
     *
     * Uses constant-time comparison when BUILD_FTPS is defined to mitigate
     * timing-based password extraction.
     */
    bool authenticate(const std::string& username, const std::string& password,
                      std::string& out_ftp_path, std::string* out_run_id = nullptr) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = users_.find(username);
        if (it == users_.end())
            return false;
        VirtualUser& u = it->second;
        if (u.consumed)
            return false;
        // TODO: See TD-377 in TECHNICAL_DEBT.md - expiry is only checked here, at login; an open session
        // outlives it, and a consumed token can still RETR on its open connection
        if (std::chrono::system_clock::now() > u.expiry)
            return false;

            // Constant-time password comparison (Phase 3)
#ifdef BUILD_FTPS
        if (u.password.size() != password.size() ||
            CRYPTO_memcmp(u.password.data(), password.data(), u.password.size()) != 0) {
            return false;
        }
#else
        if (u.password != password)
            return false;
#endif

        out_ftp_path = u.ftp_path;
        if (out_run_id)
            *out_run_id = u.run_id;
        return true;
    }

    void mark_consumed(const std::string& username) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = users_.find(username);
        if (it != users_.end())
            it->second.consumed = true;
    }

    /** Sweep expired and consumed tokens; audit-logs each expiry. */
    void sweep_expired() {
        std::lock_guard<std::mutex> lk(mtx_);
        const auto now = std::chrono::system_clock::now();
        for (auto it = users_.begin(); it != users_.end();) {
            const bool expired = !it->second.consumed && (now > it->second.expiry);
            if (expired) {
                Logger::info("[AUDIT] Token expired: username='{}' ftp_path='{}' run_id='{}'",
                             it->first, it->second.ftp_path, it->second.run_id);
            }
            if (it->second.consumed || expired)
                it = users_.erase(it);
            else
                ++it;
        }
    }

    void remove(const std::string& username) {
        std::lock_guard<std::mutex> lk(mtx_);
        users_.erase(username);
    }

   private:
    std::mutex mtx_;
    std::unordered_map<std::string, VirtualUser> users_;
};

// ============================================================================
// Token generation / crypto helpers
// ============================================================================

namespace ftp_detail {

// TD-040 hardening: used for the FTP username's random suffix always, and for
// the *password* whenever --ftp-secret is left unconfigured (the default —
// see RegistryServer.cpp, ftp_server_secret starts empty). TD-233: drawn from
// the OS CSPRNG in every build. It used RAND_bytes() only with BUILD_FTPS and
// otherwise (or if RAND_bytes() failed) a std::mt19937, which is predictable
// for something used as a credential.
inline std::string random_hex(std::size_t bytes) {
    return adai::secure_random_hex(bytes);
}

// Format a time_point as ISO-8601 UTC string: "2026-06-20T14:30:00Z"
inline std::string utc_string(std::chrono::system_clock::time_point tp) {
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm_buf{};
    adai::gmtime_utc(&t, &tm_buf);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return buf;
}

inline std::string local_ip() {
    return "0.0.0.0";
}

#ifdef BUILD_FTPS
// Phase 3: HMAC-SHA256 over @p msg with @p key.  Returns lowercase hex string.
inline std::string hmac_sha256(const std::string& key, const std::string& msg) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
         reinterpret_cast<const unsigned char*>(msg.data()), msg.size(), digest, &len);
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (unsigned int i = 0; i < len; ++i) {
        out += hex[digest[i] >> 4];
        out += hex[digest[i] & 0xf];
    }
    return out;
}

// Phase 3: generate a self-signed RSA-2048 TLS certificate and install it in ctx.
// Returns true on success, false on any OpenSSL error.
inline bool generate_self_signed(SSL_CTX* ctx) {
    // Generate RSA key
    EVP_PKEY_CTX* pctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr);
    if (!pctx)
        return false;
    if (EVP_PKEY_keygen_init(pctx) != 1) {
        EVP_PKEY_CTX_free(pctx);
        return false;
    }
    if (EVP_PKEY_CTX_set_rsa_keygen_bits(pctx, 2048) != 1) {
        EVP_PKEY_CTX_free(pctx);
        return false;
    }
    EVP_PKEY* pkey = nullptr;
    if (EVP_PKEY_keygen(pctx, &pkey) != 1) {
        EVP_PKEY_CTX_free(pctx);
        return false;
    }
    EVP_PKEY_CTX_free(pctx);

    X509* cert = X509_new();
    X509_set_version(cert, 2);
    ASN1_INTEGER_set(X509_get_serialNumber(cert), 1);
    X509_gmtime_adj(X509_get_notBefore(cert), 0);
    X509_gmtime_adj(X509_get_notAfter(cert), 365LL * 24 * 3600);  // 1 year
    X509_set_pubkey(cert, pkey);

    X509_NAME* name = X509_get_subject_name(cert);
    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
                               reinterpret_cast<const unsigned char*>("ADAI-FTP-Server"), -1, -1,
                               0);
    X509_set_issuer_name(cert, name);  // self-signed: issuer == subject
    X509_sign(cert, pkey, EVP_sha256());

    const bool ok =
        (SSL_CTX_use_certificate(ctx, cert) == 1 && SSL_CTX_use_PrivateKey(ctx, pkey) == 1);
    X509_free(cert);
    EVP_PKEY_free(pkey);
    return ok;
}

// TD-339: curl's CURLOPT_PINNEDPUBLICKEY form of a certificate's public key,
// "sha256//" + base64(SHA-256(DER SubjectPublicKeyInfo)); empty on error.
inline std::string public_key_pin(X509* cert) {
    if (!cert)
        return "";
    unsigned char* der = nullptr;
    const int der_len = i2d_PUBKEY(X509_get0_pubkey(cert), &der);
    if (der_len <= 0)
        return "";
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    const bool hashed =
        EVP_Digest(der, static_cast<std::size_t>(der_len), digest, &digest_len, EVP_sha256(),
                   nullptr) == 1;
    OPENSSL_free(der);
    if (!hashed)
        return "";
    unsigned char b64[4 * ((EVP_MAX_MD_SIZE + 2) / 3) + 1];
    const int b64_len = EVP_EncodeBlock(b64, digest, static_cast<int>(digest_len));
    return "sha256//" + std::string(reinterpret_cast<const char*>(b64), static_cast<std::size_t>(b64_len));
}

// TD-549: binds a data connection's TLS session to the control connection that created it. Each
// control connection gets a random token, stored as the SSL object's app data before its handshake.
// The ticket callbacks below write it into every session ticket the server issues (TLS 1.3 and TLS
// 1.2 tickets), and it is also set on the control session itself for TLS 1.2 session-ID resumption
// (the cached SSL_SESSION object is the same one). A data connection must resume a session carrying
// its own control connection's token: a client that merely reaches the PASV port from the right IP,
// even one with a session of its own from another control connection, is refused.
inline int session_ticket_generate(SSL* s, void* /*arg*/) {
    const auto* token = static_cast<const std::string*>(SSL_get_app_data(s));
    if (token && !token->empty())
        SSL_SESSION_set1_ticket_appdata(SSL_get0_session(s), token->data(), token->size());
    return 1;
}

inline SSL_TICKET_RETURN session_ticket_decrypt(SSL* /*s*/, SSL_SESSION* /*ss*/,
                                                const unsigned char* /*keyname*/,
                                                size_t /*keyname_len*/, SSL_TICKET_STATUS status,
                                                void* /*arg*/) {
    switch (status) {
        case SSL_TICKET_SUCCESS:
            return SSL_TICKET_RETURN_USE;
        case SSL_TICKET_SUCCESS_RENEW:
            return SSL_TICKET_RETURN_USE_RENEW;
        case SSL_TICKET_EMPTY:
        case SSL_TICKET_NO_DECRYPT:
            return SSL_TICKET_RETURN_IGNORE_RENEW;
        default:
            return SSL_TICKET_RETURN_ABORT;
    }
}

// True when @p data_ssl resumed a session carrying @p token.
inline bool data_session_matches(SSL* data_ssl, const std::string& token) {
    if (token.empty() || !SSL_session_reused(data_ssl))
        return false;
    void* data = nullptr;
    size_t len = 0;
    if (SSL_SESSION_get0_ticket_appdata(SSL_get0_session(data_ssl), &data, &len) != 1)
        return false;
    return len == token.size() && CRYPTO_memcmp(data, token.data(), len) == 0;
}
#endif  // BUILD_FTPS

}  // namespace ftp_detail

// ============================================================================
// FtpDataServer
// ============================================================================

class FtpDataServer {
   public:
    /**
     * @param data_dir            FTP root directory (registry's data_dir).
     * @param control_port        FTP control connection port (default 2121).
     * @param pasv_min            Lower bound of PASV data port pool.
     * @param pasv_max            Upper bound of PASV data port pool.
     * @param advertise_ip        IP reported in PASV responses (empty = 127.0.0.1).
     * @param server_secret       Phase 3: HMAC key for token signing (empty = random pw).
     * @param ftps_enabled        Phase 3: enable FTPS (explicit TLS via AUTH TLS).
     * @param cert_file           Phase 3: PEM cert path (empty = generate self-signed).
     * @param key_file            Phase 3: PEM key path (empty = generate self-signed).
     * @param max_sessions_per_run Phase 3: rate limit per run_id (0 = unlimited).
     */
    FtpDataServer(std::string data_dir, int control_port, int pasv_min, int pasv_max,
                  std::string advertise_ip = "", std::string server_secret = "",
                  bool ftps_en = false, std::string cert_file = "", std::string key_file = "",
                  int max_sessions_per_run = 4)
        : data_dir_(std::move(data_dir)),
          control_port_(control_port),
          pasv_min_(pasv_min),
          pasv_max_(pasv_max),
          advertise_ip_(std::move(advertise_ip)),
          server_secret_(std::move(server_secret)),
          ftps_enabled_(ftps_en),
          cert_file_(std::move(cert_file)),
          key_file_(std::move(key_file)),
          max_sessions_per_run_(max_sessions_per_run) {
        for (int p = pasv_min_; p <= pasv_max_; ++p)
            available_pasv_.insert(p);
    }

    ~FtpDataServer() {
        stop();
    }

    TokenStore& tokens() {
        return tokens_;
    }
    std::string advertise_ip() const {
        return advertise_ip_;
    }
    int control_port() const {
        return control_port_;
    }
    bool ftps_enabled() const {
        return ftps_enabled_;
    }
    /// TD-339: the TLS certificate's public-key pin ("sha256//<base64>") once start() has set
    /// up FTPS; empty otherwise.
    std::string public_key_pin() const {
        return public_key_pin_;
    }

    // ── Issue a new token ─────────────────────────────────────────────────

    struct IssuedToken {
        std::string username;           // "adai_<8hex>"
        std::string password;           // 64-char hex OR HMAC-SHA256 hex (Phase 3)
        std::string token_expires_utc;  // ISO-8601 UTC
    };

    /**
     * @brief Mint a per-file token and register it in the virtual user table.
     *
     * Phase 3: when server_secret_ is non-empty, the password is
     * HMAC-SHA256(server_secret_, username + ":" + ftp_path + ":" + expiry_utc)
     * instead of a random value, binding the credential to this server instance.
     *
     * @param run_id   Owning trainer run identifier (for audit log and rate limiting).
     * @param ftp_path Path relative to data_dir that the token grants access to.
     * @param ttl_min  Token lifetime in minutes.
     */
    IssuedToken issue_token(const std::string& run_id, const std::string& ftp_path, int ttl_min) {
        const std::string token_id = ftp_detail::random_hex(4);  // 8 hex chars
        const std::string username = "adai_" + token_id;
        const auto expiry = std::chrono::system_clock::now() + std::chrono::minutes(ttl_min);
        const std::string expiry_utc = ftp_detail::utc_string(expiry);

        // Phase 3: HMAC-derived password when a server secret is configured.
        // Falls back to random hex when no secret is set (backward-compatible).
        std::string password;
#ifdef BUILD_FTPS
        if (!server_secret_.empty()) {
            const std::string msg = username + ":" + ftp_path + ":" + expiry_utc;
            password = ftp_detail::hmac_sha256(server_secret_, msg);
        } else {
            password = ftp_detail::random_hex(32);
        }
#else
        password = ftp_detail::random_hex(32);
#endif

        VirtualUser vu;
        vu.run_id = run_id;
        vu.password = password;
        vu.ftp_path = ftp_path;
        vu.expiry = expiry;
        tokens_.insert(username, std::move(vu));

        Logger::info(
            "[AUDIT] Token issued: run_id='{}' ftp_path='{}' "
            "username='{}' expires='{}'",
            run_id, ftp_path, username, expiry_utc);

        return {username, password, expiry_utc};
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    // TODO: See TD-376 in TECHNICAL_DEBT.md - failures only logged; no return value for the caller
    void start() {
        if (running_.exchange(true))
            return;
#ifndef _WIN32
        // OpenSSL writes TLS records with plain write(), not send(MSG_NOSIGNAL), so a client that
        // drops an FTPS session (e.g. curl rejecting the certificate, TD-339) would raise SIGPIPE
        // and kill the process. registry_server was only safe because httplib::Server's
        // constructor happens to ignore SIGPIPE; don't depend on the host doing that.
        std::signal(SIGPIPE, SIG_IGN);
#endif

#ifdef BUILD_FTPS
        if (ftps_enabled_) {
            ssl_ctx_ = SSL_CTX_new(TLS_server_method());
            if (!ssl_ctx_) {
                Logger::error("[FTP] Failed to create SSL context");
                running_.store(false);
                return;
            }
            SSL_CTX_set_min_proto_version(ssl_ctx_, TLS1_2_VERSION);
            // TD-549: sessions must be resumable by the data connection, and carry the control
            // connection's token (see ftp_detail::session_ticket_generate()).
            static const unsigned char kSessionContext[] = "adai-ftp";
            SSL_CTX_set_session_id_context(ssl_ctx_, kSessionContext, sizeof(kSessionContext) - 1);
            SSL_CTX_set_session_cache_mode(ssl_ctx_, SSL_SESS_CACHE_SERVER);
            SSL_CTX_set_session_ticket_cb(ssl_ctx_, ftp_detail::session_ticket_generate,
                                          ftp_detail::session_ticket_decrypt, nullptr);

            bool cert_ok = false;
            if (!cert_file_.empty() && !key_file_.empty()) {
                cert_ok = (SSL_CTX_use_certificate_file(ssl_ctx_, cert_file_.c_str(),
                                                        SSL_FILETYPE_PEM) == 1 &&
                           SSL_CTX_use_PrivateKey_file(ssl_ctx_, key_file_.c_str(),
                                                       SSL_FILETYPE_PEM) == 1);
                if (!cert_ok)
                    Logger::error("[FTP/TLS] Failed to load cert '{}' / key '{}'", cert_file_,
                                  key_file_);
            } else {
                cert_ok = ftp_detail::generate_self_signed(ssl_ctx_);
                if (cert_ok)
                    Logger::info("[FTP/TLS] Generated self-signed TLS certificate");
                else
                    Logger::error("[FTP/TLS] Failed to generate self-signed certificate");
            }

            if (!cert_ok) {
                SSL_CTX_free(ssl_ctx_);
                ssl_ctx_ = nullptr;
                running_.store(false);
                return;
            }
            public_key_pin_ = ftp_detail::public_key_pin(SSL_CTX_get0_certificate(ssl_ctx_));
            Logger::info("[FTP] FTPS (TLS) enabled — connections will use AUTH TLS");
            // TD-339: trainers verify this server with FTPS_PINNED_PUBLIC_KEY=<pin> (or with
            // FTPS_CA_FILE for a CA-issued certificate). A generated certificate's pin changes
            // on every restart, so pinned deployments should pass --ftp-cert/--ftp-key.
            Logger::info("[FTP/TLS] Certificate public-key pin (trainer FTPS_PINNED_PUBLIC_KEY): {}",
                         public_key_pin_);
        }
#endif

        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ == adai::kInvalidSocket) {
            Logger::error("[FTP] Failed to create listen socket: {}", adai::last_socket_error());
            running_.store(false);
            return;
        }
        adai::set_reuse_addr(listen_fd_);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(control_port_));

        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            Logger::error("[FTP] bind on port {} failed: {}", control_port_,
                          adai::last_socket_error());
            adai::close_socket(listen_fd_);
            listen_fd_ = adai::kInvalidSocket;
            running_.store(false);
            return;
        }
        if (::listen(listen_fd_, 16) < 0) {
            Logger::error("[FTP] listen failed: {}", adai::last_socket_error());
            adai::close_socket(listen_fd_);
            listen_fd_ = adai::kInvalidSocket;
            running_.store(false);
            return;
        }

        Logger::info("[FTP] FtpDataServer listening on port {} (root: {})", control_port_,
                     data_dir_);

        listener_thread_ = std::thread([this] { listener_loop(); });
        sweep_thread_ = std::thread([this] { sweep_loop(); });
    }

    void stop() {
        if (!running_.exchange(false))
            return;
        {
            std::lock_guard<std::mutex> lock(stop_mtx_);  // TD-376: wake the sweep thread now
        }
        stop_cv_.notify_all();
        if (listen_fd_ != adai::kInvalidSocket) {
            ::shutdown(listen_fd_, adai::kShutdownBoth);
            adai::close_socket(listen_fd_);
            listen_fd_ = adai::kInvalidSocket;
        }
        if (listener_thread_.joinable())
            listener_thread_.join();
        if (sweep_thread_.joinable())
            sweep_thread_.join();
#ifdef BUILD_FTPS
        if (ssl_ctx_) {
            SSL_CTX_free(ssl_ctx_);
            ssl_ctx_ = nullptr;
        }
#endif
    }

   private:
    // ── State ─────────────────────────────────────────────────────────────

    std::string data_dir_;
    int control_port_;
    int pasv_min_;
    int pasv_max_;
    std::string advertise_ip_;
    std::string server_secret_;  // Phase 3: HMAC key
    bool ftps_enabled_;          // Phase 3: FTPS mode
    std::string cert_file_;      // Phase 3: PEM cert path
    std::string key_file_;       // Phase 3: PEM key path
    std::string public_key_pin_;  // TD-339: set by start() when FTPS is enabled
    int max_sessions_per_run_;   // Phase 3: rate limit

    TokenStore tokens_;
    std::atomic<bool> running_{false};
    std::mutex stop_mtx_;               // TD-376: with stop_cv_, ends the sweep wait on stop()
    std::condition_variable stop_cv_;
    adai::socket_t listen_fd_ = adai::kInvalidSocket;
    std::thread listener_thread_;
    std::thread sweep_thread_;

    // TD-161: ensures Winsock is initialized before start()'s first ::socket() call and torn
    // down only after this object (and everything else touching listen_fd_/ConnState::fd) is
    // gone — see PortableSocket.hpp's own doc comment for why this doesn't need to coordinate
    // with cpp-httplib's own independent WSAStartup call elsewhere in the same process.
    adai::WinsockGuard winsock_guard_;

    // PASV port pool
    std::mutex pasv_mtx_;
    std::set<int> available_pasv_;

    // Phase 3: per-run_id session counter for rate limiting
    std::mutex session_mtx_;
    std::unordered_map<std::string, int> active_sessions_;  // run_id → count

#ifdef BUILD_FTPS
    SSL_CTX* ssl_ctx_ = nullptr;  // Phase 3: TLS server context
#endif

    // ── PASV port pool ────────────────────────────────────────────────────

    int allocate_pasv_port() {
        std::lock_guard<std::mutex> lk(pasv_mtx_);
        if (available_pasv_.empty())
            return -1;
        int p = *available_pasv_.begin();
        available_pasv_.erase(available_pasv_.begin());
        return p;
    }

    void release_pasv_port(int port) {
        std::lock_guard<std::mutex> lk(pasv_mtx_);
        available_pasv_.insert(port);
    }

    // ── Phase 3: Rate limiting ────────────────────────────────────────────

    /** @return true if a session slot was successfully claimed for run_id. */
    bool acquire_session(const std::string& run_id) {
        if (max_sessions_per_run_ <= 0)
            return true;  // unlimited
        std::lock_guard<std::mutex> lk(session_mtx_);
        if (active_sessions_[run_id] >= max_sessions_per_run_)
            return false;
        ++active_sessions_[run_id];
        return true;
    }

    void release_session(const std::string& run_id) {
        std::lock_guard<std::mutex> lk(session_mtx_);
        auto it = active_sessions_.find(run_id);
        if (it != active_sessions_.end() && it->second > 0) {
            if (--it->second == 0)
                active_sessions_.erase(it);
        }
    }

    // ── Listener loop ─────────────────────────────────────────────────────

    void listener_loop() {
        while (running_.load()) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            adai::socket_t conn_fd =
                ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
            if (conn_fd == adai::kInvalidSocket) {
                if (running_.load())
                    Logger::warn("[FTP] accept error: {}", adai::last_socket_error());
                break;
            }
            char ip_buf[INET_ADDRSTRLEN] = {};
            ::inet_ntop(AF_INET, &client_addr.sin_addr, ip_buf, sizeof(ip_buf));
            std::string client_ip = ip_buf;

            // TODO: See TD-375 in TECHNICAL_DEBT.md - unlimited detached threads; they outlive stop() and
            // use `this` after destruction
            std::thread([this, conn_fd, client_ip = std::move(client_ip)]() mutable {
                handle_connection(conn_fd, std::move(client_ip));
            }).detach();
        }
    }

    // Periodic expired-token sweep with audit logging
    void sweep_loop() {
        while (running_.load()) {
            // TD-376: an interruptible wait; stop() wakes it, instead of waiting out a 1-minute sleep.
            std::unique_lock<std::mutex> lock(stop_mtx_);
            if (stop_cv_.wait_for(lock, std::chrono::minutes(1), [this] { return !running_.load(); })) {
                break;
            }
            lock.unlock();
            tokens_.sweep_expired();
        }
    }

    // ── Per-connection state ──────────────────────────────────────────────

    struct ConnState {
        adai::socket_t fd;
        std::string client_ip;
        bool authenticated = false;
        std::string username;
        std::string run_id;        // Phase 3: set at login for audit/rate-limit
        std::string pending_user;  // set by USER, cleared by PASS
        std::string allowed_path;  // ftp_path granted after login
        adai::socket_t pasv_listen_fd = adai::kInvalidSocket;
        int pasv_port = -1;
        bool type_binary = false;
        std::size_t bytes_transferred = 0;  // Phase 3: audit log

#ifdef BUILD_FTPS
        SSL* ssl = nullptr;        // non-null after AUTH TLS handshake
        bool data_prot_p = false;  // true after PROT P
        std::string tls_token;     // TD-549: carried by this control connection's TLS sessions
#endif
    };

    // ── I/O helpers — TLS-aware ───────────────────────────────────────────

    static void send_reply(ConnState& st, int code, const std::string& msg) {
        std::string line = std::to_string(code) + " " + msg + "\r\n";
#ifdef BUILD_FTPS
        if (st.ssl) {
            SSL_write(st.ssl, line.c_str(), static_cast<int>(line.size()));
            return;
        }
#endif
        adai::send_bytes(st.fd, line.c_str(), line.size());
    }

    // TODO: See TD-375 in TECHNICAL_DEBT.md - no length limit and no idle timeout (pre-auth resource exhaustion)
    static std::string read_line(ConnState& st) {
        std::string line;
        char c;
        while (true) {
            long n;
#ifdef BUILD_FTPS
            if (st.ssl)
                n = SSL_read(st.ssl, &c, 1);
            else
#endif
                n = adai::recv_bytes(st.fd, &c, 1);
            if (n <= 0)
                break;
            if (c == '\n')
                break;
            if (c != '\r')
                line += c;
        }
        return line;
    }

    // ── Per-connection state machine ──────────────────────────────────────

    void handle_connection(adai::socket_t fd, std::string client_ip) {
        ConnState st;
        st.fd = fd;
        st.client_ip = std::move(client_ip);

        send_reply(st, 220, "ADAI FTP Data Server ready.");

        while (true) {
            std::string line = read_line(st);
            if (line.empty())
                break;

            std::string cmd, arg;
            const auto sp = line.find(' ');
            if (sp != std::string::npos) {
                cmd = line.substr(0, sp);
                arg = line.substr(sp + 1);
                while (!arg.empty() && (arg.back() == '\r' || arg.back() == ' '))
                    arg.pop_back();
            } else {
                cmd = line;
                while (!cmd.empty() && (cmd.back() == '\r' || cmd.back() == ' '))
                    cmd.pop_back();
            }
            std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::toupper);

            if (cmd == "QUIT") {
                send_reply(st, 221, "Goodbye.");
                break;
            } else if (cmd == "NOOP") {
                send_reply(st, 200, "OK.");
            } else if (cmd == "USER") {
                cmd_user(st, arg);
            } else if (cmd == "PASS") {
                cmd_pass(st, arg);
            } else if (cmd == "TYPE") {
                st.type_binary = (arg == "I" || arg == "i");
                send_reply(st, 200, "Type set.");
            } else if (cmd == "SYST") {
                send_reply(st, 215, "UNIX Type: L8");
#ifdef BUILD_FTPS
            } else if (cmd == "AUTH") {
                cmd_auth_tls(st, arg);
            } else if (cmd == "PBSZ") {
                cmd_pbsz(st, arg);
            } else if (cmd == "PROT") {
                cmd_prot(st, arg);
#endif
            } else if (cmd == "FEAT") {
                // Advertise AUTH TLS when FTPS is enabled
#ifdef BUILD_FTPS
                const bool has_tls = (ftps_enabled_ && ssl_ctx_ != nullptr);
                std::string resp =
                    has_tls ? "211-Features:\r\n PASV\r\n AUTH TLS\r\n PBSZ\r\n PROT\r\n211 End"
                            : "211-Features:\r\n PASV\r\n211 End";
#else
                std::string resp = "211-Features:\r\n PASV\r\n211 End";
#endif
                resp += "\r\n";
                // TODO: See TD-376 in TECHNICAL_DEBT.md - raw socket write corrupts the channel after AUTH TLS
                adai::send_bytes(st.fd, resp.c_str(), resp.size());
            } else if (cmd == "PWD") {
                send_reply(st, 257, "\"/\" is the current directory");
            } else if (!st.authenticated) {
                send_reply(st, 530, "Please login first.");
            } else if (cmd == "PASV") {
                cmd_pasv(st);
            } else if (cmd == "PORT") {
                send_reply(st, 502, "PORT not supported; use PASV.");
            } else if (cmd == "RETR") {
                cmd_retr(st, arg);
            } else if (cmd == "SIZE") {
                cmd_size(st, arg);
            } else if (cmd == "CWD" || cmd == "CDUP") {
                send_reply(st, 550, "Directory navigation not permitted.");
            } else if (cmd == "LIST" || cmd == "NLST" || cmd == "MLSD") {
                send_reply(st, 550, "Directory listing not permitted.");
            } else if (cmd == "STOR" || cmd == "STOU" || cmd == "APPE" || cmd == "DELE" ||
                       cmd == "MKD" || cmd == "RMD" || cmd == "RNFR" || cmd == "RNTO") {
                send_reply(st, 502, "Write commands not permitted.");
            } else {
                send_reply(st, 502, "Command not implemented.");
            }
        }

        // Phase 3: release rate-limit slot and emit session-end audit entry
        if (st.authenticated && !st.run_id.empty()) {
            release_session(st.run_id);
            Logger::info(
                "[AUDIT] Session ended: run_id='{}' username='{}' "
                "client_ip='{}' bytes_transferred={}",
                st.run_id, st.username, st.client_ip, st.bytes_transferred);
        }

        // Clean up passive socket if still open
        if (st.pasv_listen_fd != adai::kInvalidSocket) {
            adai::close_socket(st.pasv_listen_fd);
            release_pasv_port(st.pasv_port);
        }

#ifdef BUILD_FTPS
        if (st.ssl) {
            SSL_shutdown(st.ssl);
            SSL_free(st.ssl);
            st.ssl = nullptr;
        }
#endif

        adai::close_socket(fd);
    }

    // ── FTP command handlers ──────────────────────────────────────────────

    // TODO: See TD-374 in TECHNICAL_DEBT.md - re-login doesn't release the rate-limit slot already held
    // TD-373: with FTPS enabled, credentials only over TLS (RFC 4217: 530 before AUTH TLS).
    bool tls_required_but_missing(ConnState& st) {
#ifdef BUILD_FTPS
        if (ftps_enabled_ && st.ssl == nullptr) {
            Logger::warn("[AUDIT] Plaintext login refused (FTPS required): client_ip='{}'",
                         st.client_ip);
            send_reply(st, 530, "TLS required: send AUTH TLS before USER/PASS.");
            return true;
        }
#else
        (void)st;
#endif
        return false;
    }

    void cmd_user(ConnState& st, const std::string& arg) {
        if (tls_required_but_missing(st)) {
            return;
        }
        st.pending_user = arg;
        st.authenticated = false;
        st.username.clear();
        st.run_id.clear();
        st.allowed_path.clear();
        send_reply(st, 331, "Password required.");
    }

    void cmd_pass(ConnState& st, const std::string& arg) {
        if (tls_required_but_missing(st)) {
            st.pending_user.clear();
            return;
        }
        if (st.pending_user.empty()) {
            send_reply(st, 503, "Send USER first.");
            return;
        }
        std::string ftp_path, run_id;
        if (!tokens_.authenticate(st.pending_user, arg, ftp_path, &run_id)) {
            Logger::warn("[AUDIT] Login failed: username='{}' client_ip='{}'", st.pending_user,
                         st.client_ip);
            st.pending_user.clear();
            send_reply(st, 530, "Login incorrect.");
            return;
        }

        // Phase 3: rate limiting
        if (!acquire_session(run_id)) {
            Logger::warn("[FTP] Rate limit exceeded: run_id='{}' max_sessions={}", run_id,
                         max_sessions_per_run_);
            st.pending_user.clear();
            send_reply(st, 421, "Too many concurrent sessions for this run_id; try again later.");
            return;
        }

        st.authenticated = true;
        st.username = st.pending_user;
        st.run_id = run_id;
        st.allowed_path = ftp_path;
        st.pending_user.clear();

        Logger::info("[AUDIT] Login: run_id='{}' username='{}' ftp_path='{}' client_ip='{}'",
                     st.run_id, st.username, st.allowed_path, st.client_ip);
        send_reply(st, 230, "Login successful.");
    }

#ifdef BUILD_FTPS
    void cmd_auth_tls(ConnState& st, const std::string& arg) {
        if (arg != "TLS" && arg != "SSL") {
            send_reply(st, 504, "Only AUTH TLS is supported.");
            return;
        }
        if (!ftps_enabled_ || !ssl_ctx_) {
            send_reply(st, 502, "FTPS not enabled on this server.");
            return;
        }
        if (st.ssl) {
            send_reply(st, 503, "Already in TLS mode.");
            return;
        }

        send_reply(st, 234, "AUTH TLS OK.");

        SSL* ssl = SSL_new(ssl_ctx_);
        SSL_set_fd(ssl, static_cast<int>(st.fd));  // TD-161: SOCKET->int narrowing on Windows, OpenSSL's own API shape
        st.tls_token = ftp_detail::random_hex(16);  // TD-549
        SSL_set_app_data(ssl, &st.tls_token);        // read by session_ticket_generate()
        if (SSL_accept(ssl) != 1) {
            Logger::error("[FTP/TLS] TLS handshake failed for client '{}'", st.client_ip);
            SSL_free(ssl);
            return;  // close connection on handshake failure
        }
        // TLS 1.2 session-ID resumption reuses this very SSL_SESSION from the server cache.
        SSL_SESSION_set1_ticket_appdata(SSL_get0_session(ssl), st.tls_token.data(),
                                        st.tls_token.size());
        st.ssl = ssl;
        Logger::info("[FTP/TLS] TLS established with client '{}'", st.client_ip);
    }

    void cmd_pbsz(ConnState& st, const std::string& /*arg*/) {
        // PBSZ 0 required before PROT; we only support stream mode (size 0)
        send_reply(st, 200, "PBSZ=0");
    }

    void cmd_prot(ConnState& st, const std::string& arg) {
        if (arg == "P" || arg == "p") {
            st.data_prot_p = true;
            send_reply(st, 200, "Protection set to Private.");
        } else if (arg == "C" || arg == "c") {
            st.data_prot_p = false;
            send_reply(st, 200, "Protection set to Clear.");
        } else {
            send_reply(st, 504, "Only PROT C and PROT P are supported.");
        }
    }
#endif  // BUILD_FTPS

    void cmd_pasv(ConnState& st) {
        // Close any previous PASV listener
        if (st.pasv_listen_fd != adai::kInvalidSocket) {
            adai::close_socket(st.pasv_listen_fd);
            release_pasv_port(st.pasv_port);
            st.pasv_listen_fd = adai::kInvalidSocket;
            st.pasv_port = -1;
        }

        int port = allocate_pasv_port();
        if (port < 0) {
            send_reply(st, 425, "No PASV ports available; try later.");
            return;
        }

        adai::socket_t srv = ::socket(AF_INET, SOCK_STREAM, 0);
        if (srv == adai::kInvalidSocket) {
            release_pasv_port(port);
            send_reply(st, 425, "Cannot open data connection.");
            return;
        }
        adai::set_reuse_addr(srv);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(port));

        if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
            ::listen(srv, 4) < 0) {  // TD-372: room for the real client behind a squatter
            adai::close_socket(srv);
            release_pasv_port(port);
            send_reply(st, 425, "Cannot bind data port.");
            return;
        }

        st.pasv_listen_fd = srv;
        st.pasv_port = port;

        // TODO: See TD-377 in TECHNICAL_DEBT.md - defaults to 127.0.0.1, unusable by remote trainers
        std::string ip = advertise_ip_.empty() ? "127.0.0.1" : advertise_ip_;
        std::replace(ip.begin(), ip.end(), '.', ',');
        const int p1 = port / 256;
        const int p2 = port % 256;
        send_reply(st, 227,
                   "Entering Passive Mode (" + ip + "," + std::to_string(p1) + "," +
                       std::to_string(p2) + ").");
    }

    // TD-372: accepts on st's PASV socket until a connection from st.client_ip arrives or
    // `timeout_s` passes; any other peer is closed. Returns the accepted socket or kInvalidSocket.
    adai::socket_t accept_from_client(ConnState& st, int timeout_s) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
        while (true) {
            const auto left = std::chrono::duration_cast<std::chrono::seconds>(
                                  deadline - std::chrono::steady_clock::now())
                                  .count();
            if (left <= 0) {
                return adai::kInvalidSocket;
            }
            adai::set_recv_timeout(st.pasv_listen_fd, static_cast<int>(left));
            sockaddr_in peer{};
            socklen_t peer_len = sizeof(peer);
            adai::socket_t fd = ::accept(st.pasv_listen_fd, reinterpret_cast<sockaddr*>(&peer),
                                         &peer_len);
            if (fd == adai::kInvalidSocket) {
                return adai::kInvalidSocket;  // timeout or error
            }
            char ip_buf[INET_ADDRSTRLEN] = {};
            ::inet_ntop(AF_INET, &peer.sin_addr, ip_buf, sizeof(ip_buf));
            if (st.client_ip == ip_buf) {
                return fd;
            }
            Logger::warn("[AUDIT] Data connection refused: from '{}' but the client is '{}' "
                         "(run_id='{}' username='{}')",
                         ip_buf, st.client_ip, st.run_id, st.username);
            adai::close_socket(fd);
        }
    }

    void cmd_retr(ConnState& st, const std::string& arg) {
        std::string requested = arg;
        while (!requested.empty() && requested.front() == '/')
            requested.erase(0, 1);

        if (requested != st.allowed_path) {
            Logger::warn(
                "[AUDIT] RETR denied: run_id='{}' username='{}' "
                "requested='{}' allowed='{}'",
                st.run_id, st.username, requested, st.allowed_path);
            send_reply(st, 550, "Permission denied.");
            return;
        }

        const fs::path local_path = fs::path(data_dir_) / requested;
        if (!fs::exists(local_path) || !fs::is_regular_file(local_path)) {
            send_reply(st, 550, "File not found.");
            return;
        }

        if (st.pasv_listen_fd == adai::kInvalidSocket) {
            send_reply(st, 425, "Use PASV first.");
            return;
        }

#ifdef BUILD_FTPS
        // TD-373: with FTPS enabled, the data channel must be encrypted too (RFC 4217's 521).
        if (ftps_enabled_ && !st.data_prot_p) {
            Logger::warn("[AUDIT] RETR refused without PROT P: run_id='{}' username='{}'",
                         st.run_id, st.username);
            send_reply(st, 521, "Data connection requires PROT P on this server.");
            return;
        }
#endif

        const std::size_t file_size = fs::file_size(local_path);
        Logger::info(
            "[AUDIT] RETR start: run_id='{}' username='{}' "
            "ftp_path='{}' size_bytes={}",
            st.run_id, st.username, requested, file_size);

        send_reply(st, 150, "Opening data connection.");

        // TD-372: only the control connection's own client may take the data connection. The first
        // connection from any address used to get the file (under PROT P too), and the token was
        // consumed. Others are logged and closed while this keeps waiting, within the same 30 s.
        adai::socket_t data_fd = accept_from_client(st, 30);
        adai::close_socket(st.pasv_listen_fd);
        release_pasv_port(st.pasv_port);
        st.pasv_listen_fd = adai::kInvalidSocket;
        st.pasv_port = -1;

        if (data_fd == adai::kInvalidSocket) {
            send_reply(st, 425, "Could not open data connection.");
            return;
        }

        // Phase 3: wrap data connection in TLS when PROT P is active
#ifdef BUILD_FTPS
        SSL* data_ssl = nullptr;
        if (ssl_ctx_ && st.data_prot_p) {
            data_ssl = SSL_new(ssl_ctx_);
            SSL_set_fd(data_ssl, static_cast<int>(data_fd));  // TD-161: SOCKET->int narrowing on Windows
            SSL_set_num_tickets(data_ssl, 0);  // the data connection never needs to be resumed
            if (SSL_accept(data_ssl) != 1) {
                Logger::error("[FTP/TLS] Data-channel TLS handshake failed for '{}'", st.username);
                SSL_free(data_ssl);
                data_ssl = nullptr;
                adai::close_socket(data_fd);
                send_reply(st, 425, "TLS handshake on data connection failed.");
                return;
            }
            // TD-549: the data connection must resume this control connection's TLS session.
            if (!ftp_detail::data_session_matches(data_ssl, st.tls_token)) {
                Logger::warn("[FTP/TLS] Data connection for '{}' from '{}' did not resume the control "
                             "connection's TLS session; refused",
                             st.username, st.client_ip);
                SSL_shutdown(data_ssl);
                SSL_free(data_ssl);
                data_ssl = nullptr;
                adai::close_socket(data_fd);
                send_reply(st, 522, "Data connections must reuse the control connection's TLS session.");
                return;
            }
        }
#endif

        std::ifstream file(local_path, std::ios::binary);
        if (!file.is_open()) {
#ifdef BUILD_FTPS
            if (data_ssl) {
                SSL_shutdown(data_ssl);
                SSL_free(data_ssl);
            }
#endif
            adai::close_socket(data_fd);
            send_reply(st, 550, "Cannot read file.");
            return;
        }

        const std::size_t buf_size = 65536;
        std::vector<char> buf(buf_size);
        bool ok = true;
        std::size_t bytes_sent = 0;

        while (file) {
            file.read(buf.data(), static_cast<std::streamsize>(buf_size));
            const std::streamsize n = file.gcount();
            if (n <= 0)
                break;

#ifdef BUILD_FTPS
            if (data_ssl) {
                int w = SSL_write(data_ssl, buf.data(), static_cast<int>(n));
                if (w <= 0) {
                    ok = false;
                    break;
                }
            } else
#endif
            {
                // TODO: See TD-476 in TECHNICAL_DEBT.md - a short write (sent < n) is counted as the full chunk: silent truncation
                long sent = adai::send_bytes(data_fd, buf.data(), static_cast<std::size_t>(n));
                if (sent < 0) {
                    ok = false;
                    break;
                }
            }
            bytes_sent += static_cast<std::size_t>(n);
        }

#ifdef BUILD_FTPS
        if (data_ssl) {
            SSL_shutdown(data_ssl);
            SSL_free(data_ssl);
        }
#endif
        adai::close_socket(data_fd);

        st.bytes_transferred += bytes_sent;

        if (ok) {
            tokens_.mark_consumed(st.username);
            Logger::info(
                "[AUDIT] RETR complete: run_id='{}' username='{}' "
                "ftp_path='{}' bytes={}",
                st.run_id, st.username, requested, bytes_sent);
            send_reply(st, 226, "Transfer complete.");
        } else {
            Logger::error(
                "[AUDIT] RETR aborted: run_id='{}' username='{}' "
                "ftp_path='{}' bytes_sent={}",
                st.run_id, st.username, requested, bytes_sent);
            send_reply(st, 426, "Connection closed; transfer aborted.");
        }
    }

    void cmd_size(ConnState& st, const std::string& arg) {
        std::string requested = arg;
        while (!requested.empty() && requested.front() == '/')
            requested.erase(0, 1);
        if (requested != st.allowed_path) {
            send_reply(st, 550, "Permission denied.");
            return;
        }
        const fs::path local_path = fs::path(data_dir_) / requested;
        if (!fs::exists(local_path)) {
            send_reply(st, 550, "File not found.");
            return;
        }
        send_reply(st, 213, std::to_string(fs::file_size(local_path)));
    }
};
