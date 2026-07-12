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
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "RegistryTransport.hpp"  // FileToken, AcquireResponse
#include "Logger.hpp"

#ifdef BUILD_FTP_TRANSPORT
#include <curl/curl.h>
#endif

using adai::Logger;
namespace fs = std::filesystem;

class DataTransport {
public:
    DataTransport()  = default;
    ~DataTransport() = default;

    // Phase 2: retry configuration constants
    static constexpr int kMaxRetries    = 3;        // attempts = 1 + kMaxRetries
    static constexpr int kBaseBackoffMs = 1000;     // 1 s → 2 s → 4 s

    /**
     * @brief Download one file via FTP (or FTPS) to download_dir using its per-file token.
     *
     * Phase 2 additions:
     *   - Condition B: if local_path already exists and its size equals
     *     token.size_bytes, returns immediately without making a network
     *     connection (logs "[CLEANUP-B] ...").
     *   - Resume: if local_path exists and is smaller than token.size_bytes,
     *     resumes from the current byte offset using the FTP REST command
     *     (CURLOPT_RESUME_FROM_LARGE), appending to the existing partial file.
     *   - Retry: on transient FTP errors (connection refused, timeout, etc.),
     *     retries up to kMaxRetries times with exponential backoff starting at
     *     kBaseBackoffMs.  Authentication failures are not retried.
     *
     * Phase 3 additions:
     *   - ftps_enabled: when true, enables FTPS (FTP over TLS with AUTH TLS)
     *     via CURLOPT_USE_SSL.  Peer certificate verification is skipped for
     *     self-signed server certificates (internal network deployment).
     *
     * @return Local filesystem path to the downloaded file.
     * @throws std::runtime_error on FTP error, size mismatch, or missing libcurl.
     */
    fs::path fetch(
        const FileToken&    token,
        const std::string&  ftp_host,
        int                 ftp_port,
        const fs::path&     download_dir,
        std::size_t         large_file_warn_bytes = 0,
        bool                ftps_enabled = false);

    /**
     * @brief Download all files in the acquire response.
     *
     * Phase 2: downloads up to max_parallel files concurrently using a
     * thread-pool work-stealing pattern.  Results are returned in the same
     * order as resp.files.  The first exception encountered is re-thrown after
     * all threads have completed so no downloads are left dangling.
     *
     * @return Local paths in the same order as resp.files.
     * @throws std::runtime_error if any single fetch fails.
     */
    std::vector<fs::path> fetch_all(
        const AcquireResponse& resp,
        const fs::path&        download_dir,
        int                    max_parallel = 4,
        std::size_t            large_file_warn_bytes = 0);
};

// ============================================================================
// Implementation (header-only for this single-TU consumer)
// ============================================================================

#ifdef BUILD_FTP_TRANSPORT

namespace dt_detail {

struct WriteCtx {
    std::ofstream* file;
    std::size_t    bytes_written = 0;
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

// Returns true for errors that are worth retrying (transient network issues).
// Connection refused (CURLE_COULDNT_CONNECT) and auth failures are NOT
// transient — the server is either absent or rejected us deliberately.
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

} // namespace dt_detail

inline fs::path DataTransport::fetch(
        const FileToken&   token,
        const std::string& ftp_host,
        int                ftp_port,
        const fs::path&    download_dir,
        std::size_t        large_file_warn_bytes,
        bool               ftps_enabled)
{
    fs::create_directories(download_dir);
    const fs::path local_path = download_dir / fs::path(token.ftp_path).filename();

    // ── Condition B: file already fully downloaded — skip network call ────
    if (token.size_bytes > 0 && fs::exists(local_path)) {
        const std::size_t on_disk = fs::file_size(local_path);
        if (on_disk == token.size_bytes) {
            Logger::info("[CLEANUP-B] Found previously downloaded file '{}' ({} bytes) "
                         "ready for training. Skipping re-download.",
                         token.ftp_path, token.size_bytes);
            return local_path;
        }
    }

    // Large-file pre-transfer warning
    const bool is_large = (large_file_warn_bytes > 0 &&
                           token.size_bytes >= large_file_warn_bytes);
    if (is_large) {
        const double size_mb = static_cast<double>(token.size_bytes) / (1024.0 * 1024.0);
        Logger::warn("[DataTransport] Large file transfer beginning: '{}' ({:.1f} MB). "
                     "This may take a while.",
                     token.ftp_path, size_mb);
    }

    // Build FTP URL — use plain ftp:// with CURLOPT_USE_SSL for FTPS (AUTH TLS)
    // rather than ftps:// (which implies implicit TLS on port 990).
    const std::string url = "ftp://"
        + token.ftp_username + ":" + token.ftp_password
        + "@" + ftp_host + ":" + std::to_string(ftp_port)
        + "/" + token.ftp_path;

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
                Logger::info("[DataTransport] Partial file '{}' ({}/{} bytes). "
                             "Resuming from offset {} (attempt {}/{}).",
                             token.ftp_path, on_disk, token.size_bytes,
                             resume_from, attempt + 1, kMaxRetries + 1);
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
            throw std::runtime_error("[DataTransport] Cannot open local file for writing: "
                                     + local_path.string());
        }

        dt_detail::WriteCtx ctx{&outfile, 0};

        CURL* curl = curl_easy_init();
        if (!curl) throw std::runtime_error("[DataTransport] curl_easy_init() failed");

        curl_easy_setopt(curl, CURLOPT_URL,            url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  dt_detail::curl_write);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA,      &ctx);
        curl_easy_setopt(curl, CURLOPT_FTP_USE_EPSV,   0L);   // prefer PASV over EPSV
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);   // abort if < 1 byte/s for 60s
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
        // Phase 3: FTPS — explicit TLS via AUTH TLS on the control channel
        if (ftps_enabled) {
            curl_easy_setopt(curl, CURLOPT_USE_SSL,       (long)CURLUSESSL_ALL);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L); // accept self-signed certs
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
        }
        if (resume_from > 0) {
            curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE,
                             static_cast<curl_off_t>(resume_from));
        }

        rc = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        outfile.close();

        if (rc == CURLE_OK) break;  // success — fall through to verification

        const bool retryable = dt_detail::is_transient(rc) && (attempt < kMaxRetries);
        if (retryable) {
            const int backoff_ms = kBaseBackoffMs << attempt;  // 1 s, 2 s, 4 s
            Logger::warn("[DataTransport] Transient error on attempt {}/{} for '{}': {}. "
                         "Retrying in {} ms.",
                         attempt + 1, kMaxRetries + 1, token.ftp_path,
                         curl_easy_strerror(rc), backoff_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        } else {
            break;  // non-retryable or max retries exhausted
        }
    }

    if (rc != CURLE_OK) {
        fs::remove(local_path);
        throw std::runtime_error(
            std::string("[DataTransport] FTP transfer failed for '") + token.ftp_path
            + "': " + curl_easy_strerror(rc));
    }

    // ── Size verification against full on-disk file ───────────────────────
    if (token.size_bytes > 0) {
        const std::size_t on_disk = fs::file_size(local_path);
        if (on_disk != token.size_bytes) {
            fs::remove(local_path);
            throw std::runtime_error(
                "[DataTransport] Size mismatch for '" + token.ftp_path + "': expected "
                + std::to_string(token.size_bytes) + " bytes, got "
                + std::to_string(on_disk) + " bytes.");
        }
    }

    if (is_large) {
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        const double mb   = static_cast<double>(token.size_bytes) / (1024.0 * 1024.0);
        const double mbps = (elapsed > 0.0) ? (mb / elapsed) : 0.0;
        Logger::info("[DataTransport] Large file transfer complete: '{}' "
                     "({:.1f} MB in {:.1f}s, {:.2f} MB/s).",
                     token.ftp_path, mb, elapsed, mbps);
    }

    Logger::info("[DataTransport] Downloaded '{}' → '{}' ({} bytes)",
                 token.ftp_path, local_path.string(), token.size_bytes);
    return local_path;
}

inline std::vector<fs::path> DataTransport::fetch_all(
        const AcquireResponse& resp,
        const fs::path&        download_dir,
        int                    max_parallel,
        std::size_t            large_file_warn_bytes)
{
    const std::size_t n = resp.files.size();
    if (n == 0) return {};

    std::vector<fs::path>           local_paths(n);
    std::vector<std::exception_ptr> errors(n);

    // Work-stealing thread pool: each thread atomically claims the next
    // unprocessed index.  Results land in-order even though processing may
    // be out-of-order.
    std::atomic<std::size_t> next_idx{0};
    const int nthreads = std::min(static_cast<int>(n), std::max(1, max_parallel));

    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back([&] {
            while (true) {
                const std::size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                if (idx >= n) break;
                try {
                    local_paths[idx] = fetch(
                        resp.files[idx], resp.ftp_server_host, resp.ftp_server_port,
                        download_dir, large_file_warn_bytes, resp.ftps_enabled);
                } catch (...) {
                    errors[idx] = std::current_exception();
                }
            }
        });
    }
    for (auto& t : threads) t.join();

    // Re-throw the first error so the caller can handle it
    for (const auto& e : errors) {
        if (e) std::rethrow_exception(e);
    }
    return local_paths;
}

#else // BUILD_FTP_TRANSPORT not defined

inline fs::path DataTransport::fetch(
        const FileToken&, const std::string&, int, const fs::path&, std::size_t, bool) {
    throw std::runtime_error(
        "[DataTransport] FTP transport not compiled (BUILD_FTP_TRANSPORT not set). "
        "Build with libcurl to enable remote dataset download.");
}

inline std::vector<fs::path> DataTransport::fetch_all(
        const AcquireResponse&, const fs::path&, int, std::size_t) {
    throw std::runtime_error(
        "[DataTransport] FTP transport not compiled (BUILD_FTP_TRANSPORT not set). "
        "Build with libcurl to enable remote dataset download.");
}

#endif // BUILD_FTP_TRANSPORT
