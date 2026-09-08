// @adai-status: beta        (capped by TD-040 — hand-rolled auth/token/virtual-user logic not security-reviewed)
// @adai-version: 0.8.0
// @adai-reviewed: 2026-09-07

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
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
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

// POSIX networking
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "Logger.hpp"

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

inline std::string random_hex(std::size_t bytes) {
    static const char hex[] = "0123456789abcdef";
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<unsigned> dis(0, 255);
    std::string out;
    out.reserve(bytes * 2);
    for (std::size_t i = 0; i < bytes; ++i) {
        unsigned v = dis(gen);
        out += hex[v >> 4];
        out += hex[v & 0xf];
    }
    return out;
}

// Format a time_point as ISO-8601 UTC string: "2026-06-20T14:30:00Z"
inline std::string utc_string(std::chrono::system_clock::time_point tp) {
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
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

    void start() {
        if (running_.exchange(true))
            return;

#ifdef BUILD_FTPS
        if (ftps_enabled_) {
            ssl_ctx_ = SSL_CTX_new(TLS_server_method());
            if (!ssl_ctx_) {
                Logger::error("[FTP] Failed to create SSL context");
                running_.store(false);
                return;
            }
            SSL_CTX_set_min_proto_version(ssl_ctx_, TLS1_2_VERSION);

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
            Logger::info("[FTP] FTPS (TLS) enabled — connections will use AUTH TLS");
        }
#endif

        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) {
            Logger::error("[FTP] Failed to create listen socket: {}", std::strerror(errno));
            running_.store(false);
            return;
        }
        int opt = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(control_port_));

        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            Logger::error("[FTP] bind on port {} failed: {}", control_port_, std::strerror(errno));
            ::close(listen_fd_);
            listen_fd_ = -1;
            running_.store(false);
            return;
        }
        if (::listen(listen_fd_, 16) < 0) {
            Logger::error("[FTP] listen failed: {}", std::strerror(errno));
            ::close(listen_fd_);
            listen_fd_ = -1;
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
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
            listen_fd_ = -1;
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
    int max_sessions_per_run_;   // Phase 3: rate limit

    TokenStore tokens_;
    std::atomic<bool> running_{false};
    int listen_fd_ = -1;
    std::thread listener_thread_;
    std::thread sweep_thread_;

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
            int conn_fd =
                ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
            if (conn_fd < 0) {
                if (running_.load())
                    Logger::warn("[FTP] accept error: {}", std::strerror(errno));
                break;
            }
            char ip_buf[INET_ADDRSTRLEN] = {};
            ::inet_ntop(AF_INET, &client_addr.sin_addr, ip_buf, sizeof(ip_buf));
            std::string client_ip = ip_buf;

            std::thread([this, conn_fd, client_ip = std::move(client_ip)]() mutable {
                handle_connection(conn_fd, std::move(client_ip));
            }).detach();
        }
    }

    // Periodic expired-token sweep with audit logging
    void sweep_loop() {
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::minutes(1));
            tokens_.sweep_expired();
        }
    }

    // ── Per-connection state ──────────────────────────────────────────────

    struct ConnState {
        int fd;
        std::string client_ip;
        bool authenticated = false;
        std::string username;
        std::string run_id;        // Phase 3: set at login for audit/rate-limit
        std::string pending_user;  // set by USER, cleared by PASS
        std::string allowed_path;  // ftp_path granted after login
        int pasv_listen_fd = -1;
        int pasv_port = -1;
        bool type_binary = false;
        std::size_t bytes_transferred = 0;  // Phase 3: audit log

#ifdef BUILD_FTPS
        SSL* ssl = nullptr;        // non-null after AUTH TLS handshake
        bool data_prot_p = false;  // true after PROT P
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
        ::send(st.fd, line.c_str(), line.size(), MSG_NOSIGNAL);
    }

    static std::string read_line(ConnState& st) {
        std::string line;
        char c;
        while (true) {
            ssize_t n;
#ifdef BUILD_FTPS
            if (st.ssl)
                n = SSL_read(st.ssl, &c, 1);
            else
#endif
                n = ::recv(st.fd, &c, 1, 0);
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

    void handle_connection(int fd, std::string client_ip) {
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
                ::send(st.fd, resp.c_str(), resp.size(), MSG_NOSIGNAL);
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
        if (st.pasv_listen_fd >= 0) {
            ::close(st.pasv_listen_fd);
            release_pasv_port(st.pasv_port);
        }

#ifdef BUILD_FTPS
        if (st.ssl) {
            SSL_shutdown(st.ssl);
            SSL_free(st.ssl);
            st.ssl = nullptr;
        }
#endif

        ::close(fd);
    }

    // ── FTP command handlers ──────────────────────────────────────────────

    void cmd_user(ConnState& st, const std::string& arg) {
        st.pending_user = arg;
        st.authenticated = false;
        st.username.clear();
        st.run_id.clear();
        st.allowed_path.clear();
        send_reply(st, 331, "Password required.");
    }

    void cmd_pass(ConnState& st, const std::string& arg) {
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
        SSL_set_fd(ssl, st.fd);
        if (SSL_accept(ssl) != 1) {
            Logger::error("[FTP/TLS] TLS handshake failed for client '{}'", st.client_ip);
            SSL_free(ssl);
            return;  // close connection on handshake failure
        }
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
        if (st.pasv_listen_fd >= 0) {
            ::close(st.pasv_listen_fd);
            release_pasv_port(st.pasv_port);
            st.pasv_listen_fd = -1;
            st.pasv_port = -1;
        }

        int port = allocate_pasv_port();
        if (port < 0) {
            send_reply(st, 425, "No PASV ports available; try later.");
            return;
        }

        int srv = ::socket(AF_INET, SOCK_STREAM, 0);
        if (srv < 0) {
            release_pasv_port(port);
            send_reply(st, 425, "Cannot open data connection.");
            return;
        }
        int opt = 1;
        ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(port));

        if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
            ::listen(srv, 1) < 0) {
            ::close(srv);
            release_pasv_port(port);
            send_reply(st, 425, "Cannot bind data port.");
            return;
        }

        st.pasv_listen_fd = srv;
        st.pasv_port = port;

        std::string ip = advertise_ip_.empty() ? "127.0.0.1" : advertise_ip_;
        std::replace(ip.begin(), ip.end(), '.', ',');
        const int p1 = port / 256;
        const int p2 = port % 256;
        send_reply(st, 227,
                   "Entering Passive Mode (" + ip + "," + std::to_string(p1) + "," +
                       std::to_string(p2) + ").");
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

        if (st.pasv_listen_fd < 0) {
            send_reply(st, 425, "Use PASV first.");
            return;
        }

        const std::size_t file_size = fs::file_size(local_path);
        Logger::info(
            "[AUDIT] RETR start: run_id='{}' username='{}' "
            "ftp_path='{}' size_bytes={}",
            st.run_id, st.username, requested, file_size);

        send_reply(st, 150, "Opening data connection.");

        struct timeval tv {
            30, 0
        };
        ::setsockopt(st.pasv_listen_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        int data_fd = ::accept(st.pasv_listen_fd, nullptr, nullptr);
        ::close(st.pasv_listen_fd);
        release_pasv_port(st.pasv_port);
        st.pasv_listen_fd = -1;
        st.pasv_port = -1;

        if (data_fd < 0) {
            send_reply(st, 425, "Could not open data connection.");
            return;
        }

        // Phase 3: wrap data connection in TLS when PROT P is active
#ifdef BUILD_FTPS
        SSL* data_ssl = nullptr;
        if (ssl_ctx_ && st.data_prot_p) {
            data_ssl = SSL_new(ssl_ctx_);
            SSL_set_fd(data_ssl, data_fd);
            if (SSL_accept(data_ssl) != 1) {
                Logger::error("[FTP/TLS] Data-channel TLS handshake failed for '{}'", st.username);
                SSL_free(data_ssl);
                data_ssl = nullptr;
                ::close(data_fd);
                send_reply(st, 425, "TLS handshake on data connection failed.");
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
            ::close(data_fd);
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
                ssize_t sent =
                    ::send(data_fd, buf.data(), static_cast<std::size_t>(n), MSG_NOSIGNAL);
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
        ::close(data_fd);

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
