#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-07

#include <string>
#include <vector>

namespace adai {

/**
 * TD-312: one file download over HTTPS through libcurl, in process. `DataFetcher` used to run
 * `std::system("curl … \"<url>\"")`, so `$(…)` in an ID or a URL ran in a shell, `HF_TOKEN` was on
 * the command line (visible in `ps`), and a stalled server held a `registry_server` request thread
 * forever.
 */
struct HttpDownloadRequest {
    std::string url;
    std::string output_path;
    /// Sent as `Authorization: Bearer …` to the URL's own host only, never to a host a redirect
    /// leads to. Empty = no header.
    std::string bearer_token;
    /// Hosts the URL may name: an exact name ("huggingface.co"), or a leading dot for any subdomain
    /// (".huggingface.co"). Empty = any host. Redirects may lead elsewhere (download CDNs), over
    /// HTTPS only.
    std::vector<std::string> allowed_hosts;
    /// Tests against a local server only: allows `http://` for the URL and its redirects.
    bool allow_plain_http = false;
    long connect_timeout_seconds = 30;
    /// Give up when the transfer stays below `low_speed_limit_bytes` per second for
    /// `low_speed_time_seconds` (a stalled server).
    long low_speed_limit_bytes = 1024;
    long low_speed_time_seconds = 60;
    /// Upper bound on the whole transfer; Parquet parts can be hundreds of MB.
    long max_time_seconds = 3600;
};

struct HttpDownloadResult {
    bool ok = false;
    long http_status = 0;
    std::string error;  ///< Why it failed; never contains the token.
};

/**
 * Downloads @p request.url to @p request.output_path. The body goes to `<output_path>.part`, which
 * is renamed over the output only when the whole transfer succeeded with a 2xx status and a
 * non-empty body; on failure the `.part` file is removed and an existing output is left as it was.
 * The URL is refused before any connection when it isn't `https://` (see allow_plain_http) or its
 * host isn't allowed. Without libcurl in the build, every call fails with that reason.
 */
HttpDownloadResult http_download(const HttpDownloadRequest& request);

/// Percent-encodes @p value for one URL query parameter (RFC 3986 unreserved characters kept).
std::string url_encode_component(const std::string& value);

}  // namespace adai
