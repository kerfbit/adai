// @adai-status: stable
// @adai-version: 1.2.0
// @adai-reviewed: 2026-10-24

/**
 * DataTransport — trainer-side FTP download client.
 *
 * Phase 1: single-file fetch via libcurl with size verification and large-file
 *          warnings; sequential fetch_all.
 * Phase 2: resume partial downloads via FTP REST; automatic retry with
 *          exponential backoff on transient errors; parallel fetch_all via
 *          thread pool; skip already-complete files (Condition B).
 *
 * Only compiled when BUILD_FTP_TRANSPORT is defined (i.e. libcurl is available).
 * When the macro is absent all methods throw std::runtime_error immediately so
 * callers that check ftp_server_host.empty() before calling will never reach
 * the throw in the non-FTP (localhost) path.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "Logger.hpp"
#include "RegistryTransport.hpp"  // FileToken, AcquireResponse

#ifdef BUILD_FTP_TRANSPORT
#include <curl/curl.h>
#endif

// TODO: See TD-340 in TECHNICAL_DEBT.md - global using/namespace alias in a header leak into every includer
using adai::Logger;
namespace fs = std::filesystem;

/// TD-339: how fetch() authenticates the registry's FTPS server. With neither field set the
/// server's certificate isn't checked (a warning is logged once per process).
struct FtpsVerification {
    /// PEM CA bundle (or the server's own certificate) that must sign the server's certificate;
    /// enables peer and hostname verification. The certificate must name the FTP host the
    /// registry advertises (FTP_ADVERTISE_IP / --ftp-advertise-ip).
    std::string ca_file;
    /// curl CURLOPT_PINNEDPUBLICKEY value: "sha256//<base64>" (registry_server logs its pin at
    /// startup) or the path of a PEM/DER public key. Works with a self-signed certificate.
    std::string pinned_public_key;

    bool configured() const { return !ca_file.empty() || !pinned_public_key.empty(); }
};

class DataTransport {
   public:
    DataTransport() = default;
    explicit DataTransport(FtpsVerification verification)
        : verification_(std::move(verification)) {}
    ~DataTransport() = default;

    const FtpsVerification& ftps_verification() const { return verification_; }

    // Phase 2: retry configuration constants
    static constexpr int kMaxRetries = 3;        // attempts = 1 + kMaxRetries
    static constexpr int kBaseBackoffMs = 1000;  // 1 s → 2 s → 4 s

    /**
     * @brief TD-337: local path of @p token's download in @p download_dir, named by
     *        download_file_name() from the token's registry_path (ftp_path when that is empty).
     */
    static fs::path local_path_for(const FileToken& token, const fs::path& download_dir) {
        return download_dir / download_file_name(token.registry_path.empty() ? token.ftp_path
                                                                             : token.registry_path);
    }

    /// TD-338: the sidecar recording the token checksum @p local_path was downloaded against.
    static fs::path checksum_path_for(const fs::path& local_path) {
        return fs::path(local_path.string() + kDownloadChecksumSuffix);
    }

    /// Removes a download and its checksum sidecar (missing files are ignored).
    static void remove_download(const fs::path& local_path) {
        std::error_code ec;
        fs::remove(local_path, ec);
        fs::remove(checksum_path_for(local_path), ec);
    }

    /**
     * @brief Download one file via FTP (or FTPS) to download_dir using its per-file token.
     *
     * The local file is named by local_path_for() (TD-337). When the token has a checksum, the
     * checksum is written to a sidecar (checksum_path_for()) before the transfer, and an
     * existing file whose sidecar doesn't match is deleted and downloaded again (TD-338).
     *
     * Phase 2 additions:
     *   - Condition B: if local_path already exists, its size equals
     *     token.size_bytes and (TD-338) its sidecar matches the token checksum,
     *     returns immediately without making a network connection (logs "[CLEANUP-B] ...").
     *   - Resume: if local_path exists and is smaller than token.size_bytes,
     *     resumes from the current byte offset using the FTP REST command
     *     (CURLOPT_RESUME_FROM_LARGE), appending to the existing partial file.
     *   - Retry: on transient FTP errors (connection refused, timeout, etc.),
     *     retries up to kMaxRetries times with exponential backoff starting at
     *     kBaseBackoffMs.  Authentication failures are not retried.
     *
     * Phase 3 additions:
     *   - ftps_enabled: when true, enables FTPS (FTP over TLS with AUTH TLS)
     *     via CURLOPT_USE_SSL.
     *   - TD-339: the server is authenticated by the FtpsVerification passed to the
     *     constructor (CA file and/or pinned public key). When one is configured, a registry
     *     offering plain FTP is refused, since the credentials would cross unencrypted to an
     *     unverified server.
     *
     * @return Local filesystem path to the downloaded file.
     * @throws std::runtime_error on FTP error, size mismatch, or missing libcurl.
     */
    fs::path fetch(const FileToken& token, const std::string& ftp_host, int ftp_port,
                   const fs::path& download_dir, std::size_t large_file_warn_bytes = 0,
                   bool ftps_enabled = false);

    /**
     * @brief Download all files in the acquire response.
     *
     * Phase 2: downloads up to max_parallel files concurrently using a
     * thread-pool work-stealing pattern.  Results are returned in the same
     * order as resp.files. TD-337: tokens for the same file (TD-205 segments) are downloaded
     * once, and each gets that one local path.  The first exception encountered is re-thrown after
     * all threads have completed so no downloads are left dangling.
     *
     * @return Local paths in the same order as resp.files.
     * @throws std::runtime_error if any single fetch fails.
     */
    std::vector<fs::path> fetch_all(const AcquireResponse& resp, const fs::path& download_dir,
                                    int max_parallel = 4, std::size_t large_file_warn_bytes = 0);

   private:
    FtpsVerification verification_;
};

// ============================================================================
// Implementation (header-only for this single-TU consumer)
// ============================================================================

#ifdef BUILD_FTP_TRANSPORT

namespace dt_detail {

struct WriteCtx {
    std::ofstream* file;
    std::size_t bytes_written = 0;
};

static std::size_t curl_write(void* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
    auto* ctx = static_cast<WriteCtx*>(userdata);
    const std::size_t total = size * nmemb;
    ctx->file->write(static_cast<const char*>(ptr), static_cast<std::streamsize>(total));
    if (ctx->file->good()) {
        ctx->bytes_written += total;
        return total;
    }
    return 0;  // signal error to curl
}

// Percent-encode each '/'-separated segment of an FTP path independently, so
// the '/' separators themselves survive as path structure while everything
// else (spaces, '#', '?', '%', etc.) becomes transport-safe. Registry-side
// ftp_path values are derived directly from real on-disk dataset filenames
// (Gutenberg/HuggingFace downloads routinely contain spaces), and libcurl
// does not implicitly encode an FTP URL passed via CURLOPT_URL — an
// unencoded space or '#' makes curl_easy_perform fail immediately with
// CURLE_URL_MALFORMAT, which is not in the transient/retryable set above.
inline std::string url_encode_ftp_path(const std::string& path) {
    std::string encoded;
    std::size_t start = 0;
    while (true) {
        const std::size_t slash = path.find('/', start);
        const std::string segment = path.substr(start, slash - start);
        if (!segment.empty()) {
            char* escaped = curl_easy_escape(nullptr, segment.c_str(),
                                             static_cast<int>(segment.size()));
            if (escaped) {
                encoded += escaped;
                curl_free(escaped);
            } else {
                encoded += segment;  // escape failure: fall back to the raw segment
            }
        }
        if (slash == std::string::npos)
            break;
        encoded += '/';
        start = slash + 1;
    }
    return encoded;
}

// Returns true for errors that are worth retrying (transient network issues).
// Connection refused (CURLE_COULDNT_CONNECT) and auth failures are NOT
// transient — the server is either absent or rejected us deliberately.
// TODO: See TD-340 in TECHNICAL_DEBT.md - CURLE_COULDNT_CONNECT isn't retried (a registry restart fails the pass)
inline bool is_transient(CURLcode rc) {
    switch (rc) {
        case CURLE_OPERATION_TIMEDOUT:  // connect/transfer timed out — server may be recovering
        case CURLE_RECV_ERROR:          // recv() failure — TCP drop mid-transfer
        case CURLE_SEND_ERROR:          // send() failure — TCP drop mid-transfer
        case CURLE_GOT_NOTHING:         // empty response — server closed connection early
        case CURLE_PARTIAL_FILE:        // server closed data connection before EOF
        case CURLE_FTP_ACCEPT_TIMEOUT:  // PASV data connection timed out
            return true;
        default:
            return false;
    }
}

}  // namespace dt_detail

inline fs::path DataTransport::fetch(const FileToken& token, const std::string& ftp_host,
                                     int ftp_port, const fs::path& download_dir,
                                     std::size_t large_file_warn_bytes, bool ftps_enabled) {
    fs::create_directories(download_dir);
    // TD-337: named from the full registry path, not the basename, so segment tokens of one file
    // share one download and same-named files in different directories don't collide.
    const fs::path local_path = local_path_for(token, download_dir);
    const fs::path checksum_path = checksum_path_for(local_path);

    // TD-338: an existing file (complete or partial) is only reused when it was downloaded
    // against this token's checksum; a same-size stale copy used to be trained on. Tokens with no
    // checksum keep the size-only check.
    if (!token.checksum.empty() && fs::exists(local_path)) {
        std::string recorded;
        std::ifstream(checksum_path) >> recorded;
        if (recorded != token.checksum) {
            Logger::info(
                "[DataTransport] '{}' on disk doesn't match the registry's checksum ({}); "
                "downloading it again",
                token.ftp_path, recorded.empty() ? "no record" : "changed");
            remove_download(local_path);
        }
    }

    // ── Condition B: file already fully downloaded — skip network call ────
    if (token.size_bytes > 0 && fs::exists(local_path)) {
        const std::size_t on_disk = fs::file_size(local_path);
        if (on_disk == token.size_bytes) {
            Logger::info(
                "[CLEANUP-B] Found previously downloaded file '{}' ({} bytes) "
                "ready for training. Skipping re-download.",
                token.ftp_path, token.size_bytes);
            return local_path;
        }
    }

    // Large-file pre-transfer warning
    const bool is_large = (large_file_warn_bytes > 0 && token.size_bytes >= large_file_warn_bytes);
    if (is_large) {
        const double size_mb = static_cast<double>(token.size_bytes) / (1024.0 * 1024.0);
        Logger::warn(
            "[DataTransport] Large file transfer beginning: '{}' ({:.1f} MB). "
            "This may take a while.",
            token.ftp_path, size_mb);
    }

    // TD-339: refuse before connecting when the configured verification can't be applied.
    if (verification_.configured() && !ftps_enabled) {
        throw std::runtime_error(
            "[DataTransport] FTPS verification is configured (FTPS_CA_FILE / "
            "FTPS_PINNED_PUBLIC_KEY) but the registry offered plain FTP for '" + token.ftp_path +
            "'; refusing to send credentials to an unverified server");
    }
    if (!verification_.ca_file.empty() && !fs::is_regular_file(verification_.ca_file)) {
        throw std::runtime_error("[DataTransport] FTPS_CA_FILE '" + verification_.ca_file +
                                 "' is not a readable file");
    }
    if (ftps_enabled && !verification_.configured()) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            Logger::warn(
                "[DataTransport] FTPS is NOT verifying the registry's certificate: anyone who can "
                "intercept the connection can impersonate registry_server, serve arbitrary training "
                "data and collect download credentials. Set FTPS_PINNED_PUBLIC_KEY (registry_server "
                "logs its pin at startup) or FTPS_CA_FILE in config.trainer.conf.");
    }

    // Build FTP URL — use plain ftp:// with CURLOPT_USE_SSL for FTPS (AUTH TLS)
    // rather than ftps:// (which implies implicit TLS on port 990).
    // ftp_path is percent-encoded per path segment: it comes from real on-disk
    // filenames and routinely contains spaces (Gutenberg/HuggingFace dataset
    // names), which libcurl will not tolerate unencoded in a URL.
    const std::string url = "ftp://" + token.ftp_username + ":" + token.ftp_password + "@" +
                            ftp_host + ":" + std::to_string(ftp_port) + "/" +
                            dt_detail::url_encode_ftp_path(token.ftp_path);

    // TD-338: record which version of the file this download is, before any bytes land, so a
    // partial file is only resumed against the same checksum.
    if (!token.checksum.empty()) {
        std::ofstream(checksum_path, std::ios::trunc) << token.checksum;
    }

    const auto t_start = std::chrono::steady_clock::now();
    CURLcode rc = CURLE_OK;

    for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
        // ── Determine resume offset from current on-disk state ────────────
        std::size_t resume_from = 0;
        if (fs::exists(local_path)) {
            const std::size_t on_disk = fs::file_size(local_path);
            if (on_disk == 0) {
                // Zero-byte sentinel file: start fresh
                fs::remove(local_path);
            } else if (token.size_bytes > 0 && on_disk >= token.size_bytes) {
                // Unexpected overshoot (shouldn't reach here after Condition B check)
                fs::remove(local_path);
            } else {
                // Genuine partial file: resume from current offset
                resume_from = on_disk;
                Logger::info(
                    "[DataTransport] Partial file '{}' ({}/{} bytes). "
                    "Resuming from offset {} (attempt {}/{}).",
                    token.ftp_path, on_disk, token.size_bytes, resume_from, attempt + 1,
                    kMaxRetries + 1);
            }
        }

        // ── Open local file for writing ───────────────────────────────────
        std::ofstream outfile;
        if (resume_from > 0) {
            outfile.open(local_path, std::ios::binary | std::ios::app);
        } else {
            outfile.open(local_path, std::ios::binary | std::ios::trunc);
        }
        if (!outfile.is_open()) {
            throw std::runtime_error("[DataTransport] Cannot open local file for writing: " +
                                     local_path.string());
        }

        dt_detail::WriteCtx ctx{&outfile, 0};

        // TODO: See TD-340 in TECHNICAL_DEBT.md - no curl_global_init(); implicit init from parallel threads needs a thread-safe libcurl
        CURL* curl = curl_easy_init();
        if (!curl)
            throw std::runtime_error("[DataTransport] curl_easy_init() failed");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dt_detail::curl_write);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
        curl_easy_setopt(curl, CURLOPT_FTP_USE_EPSV, 0L);  // prefer PASV over EPSV
        // FtpDataServer (registry-side) rejects all CWD ("directory navigation not
        // permitted") and expects the full relative path in a single RETR — curl's
        // default MULTICWD method would CWD into each path segment first and get a
        // 550 before ever sending RETR. Phase 11 dataset fetches (gutenberg/
        // huggingface/upload) are stored under <group>/datasets or <group>/uploads,
        // so ftp_path routinely contains '/' — NOCWD is required for those to work.
        curl_easy_setopt(curl, CURLOPT_FTP_FILEMETHOD, (long)CURLFTPMETHOD_NOCWD);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);  // abort if < 1 byte/s for 60s
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
        // Phase 3: FTPS — explicit TLS via AUTH TLS on the control channel.
        // TD-339: a CA file turns on peer + hostname verification; a pin is checked on its own,
        // so it also authenticates a self-signed certificate. With neither, nothing is checked.
        if (ftps_enabled) {
            curl_easy_setopt(curl, CURLOPT_USE_SSL, (long)CURLUSESSL_ALL);
            const bool verify_chain = !verification_.ca_file.empty();
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, verify_chain ? 1L : 0L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, verify_chain ? 2L : 0L);
            if (verify_chain)
                curl_easy_setopt(curl, CURLOPT_CAINFO, verification_.ca_file.c_str());
            if (!verification_.pinned_public_key.empty())
                curl_easy_setopt(curl, CURLOPT_PINNEDPUBLICKEY,
                                 verification_.pinned_public_key.c_str());
        }
        if (resume_from > 0) {
            curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(resume_from));
        }

        rc = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        outfile.close();

        if (rc == CURLE_OK)
            break;  // success — fall through to verification

        const bool retryable = dt_detail::is_transient(rc) && (attempt < kMaxRetries);
        if (retryable) {
            const int backoff_ms = kBaseBackoffMs << attempt;  // 1 s, 2 s, 4 s
            Logger::warn(
                "[DataTransport] Transient error on attempt {}/{} for '{}': {}. "
                "Retrying in {} ms.",
                attempt + 1, kMaxRetries + 1, token.ftp_path, curl_easy_strerror(rc), backoff_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        } else {
            break;  // non-retryable or max retries exhausted
        }
    }

    // TODO: See TD-340 in TECHNICAL_DEBT.md - deletes the partial file, so resume can't help across failed passes
    if (rc != CURLE_OK) {
        remove_download(local_path);
        throw std::runtime_error(std::string("[DataTransport] FTP transfer failed for '") +
                                 token.ftp_path + "': " + curl_easy_strerror(rc));
    }

    // ── Size verification against full on-disk file ───────────────────────
    if (token.size_bytes > 0) {
        const std::size_t on_disk = fs::file_size(local_path);
        if (on_disk != token.size_bytes) {
            remove_download(local_path);
            throw std::runtime_error("[DataTransport] Size mismatch for '" + token.ftp_path +
                                     "': expected " + std::to_string(token.size_bytes) +
                                     " bytes, got " + std::to_string(on_disk) + " bytes.");
        }
    }

    if (is_large) {
        const auto elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        const double mb = static_cast<double>(token.size_bytes) / (1024.0 * 1024.0);
        const double mbps = (elapsed > 0.0) ? (mb / elapsed) : 0.0;
        Logger::info(
            "[DataTransport] Large file transfer complete: '{}' "
            "({:.1f} MB in {:.1f}s, {:.2f} MB/s).",
            token.ftp_path, mb, elapsed, mbps);
    }

    Logger::info("[DataTransport] Downloaded '{}' → '{}' ({} bytes)", token.ftp_path,
                 local_path.string(), token.size_bytes);
    return local_path;
}

inline std::vector<fs::path> DataTransport::fetch_all(const AcquireResponse& resp,
                                                      const fs::path& download_dir,
                                                      int max_parallel,
                                                      std::size_t large_file_warn_bytes) {
    const std::size_t n = resp.files.size();
    if (n == 0)
        return {};

    // TD-337: one download per local file. TD-205 segment tokens of one file used to download it
    // N times, concurrently, into the same path, where one thread's size check could delete
    // another's in-progress file. `unique` holds the first token index for each local path;
    // `owner[i]` is the slot in `unique` whose download token i uses.
    std::vector<std::size_t> unique;
    std::vector<std::size_t> owner(n);
    for (std::size_t i = 0; i < n; ++i) {
        const fs::path p = local_path_for(resp.files[i], download_dir);
        std::size_t slot = 0;
        while (slot < unique.size() && local_path_for(resp.files[unique[slot]], download_dir) != p)
            ++slot;
        if (slot == unique.size())
            unique.push_back(i);
        owner[i] = slot;
    }

    const std::size_t m = unique.size();
    std::vector<fs::path> unique_paths(m);
    std::vector<std::exception_ptr> errors(m);

    // Work-stealing thread pool: each thread atomically claims the next
    // unprocessed index.  Results land in-order even though processing may
    // be out-of-order.
    std::atomic<std::size_t> next_idx{0};
    const int nthreads = std::min(static_cast<int>(m), std::max(1, max_parallel));

    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back([&] {
            while (true) {
                const std::size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                if (idx >= m)
                    break;
                try {
                    unique_paths[idx] =
                        fetch(resp.files[unique[idx]], resp.ftp_server_host, resp.ftp_server_port,
                              download_dir, large_file_warn_bytes, resp.ftps_enabled);
                } catch (...) {
                    errors[idx] = std::current_exception();
                }
            }
        });
    }
    for (auto& t : threads)
        t.join();

    // Re-throw the first error so the caller can handle it
    for (const auto& e : errors) {
        if (e)
            std::rethrow_exception(e);
    }
    std::vector<fs::path> local_paths(n);
    for (std::size_t i = 0; i < n; ++i)
        local_paths[i] = unique_paths[owner[i]];
    return local_paths;
}

#else  // BUILD_FTP_TRANSPORT not defined

inline fs::path DataTransport::fetch(const FileToken&, const std::string&, int, const fs::path&,
                                     std::size_t, bool) {
    throw std::runtime_error(
        "[DataTransport] FTP transport not compiled (BUILD_FTP_TRANSPORT not set). "
        "Build with libcurl to enable remote dataset download.");
}

inline std::vector<fs::path> DataTransport::fetch_all(const AcquireResponse&, const fs::path&, int,
                                                      std::size_t) {
    throw std::runtime_error(
        "[DataTransport] FTP transport not compiled (BUILD_FTP_TRANSPORT not set). "
        "Build with libcurl to enable remote dataset download.");
}

#endif  // BUILD_FTP_TRANSPORT
