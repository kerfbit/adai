// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10

#include "HttpDownload.hpp"
#include <cstdio>
#include <filesystem>
#include <system_error>

#ifdef ADAI_HAVE_LIBCURL
#include <curl/curl.h>
#include <mutex>
#endif

namespace fs = std::filesystem;

namespace adai {

std::string url_encode_component(const std::string& value) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(value.size() * 3);
    for (const unsigned char c : value) {
        const bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                                (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
                                c == '~';
        if (unreserved) {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 0x0F];
        }
    }
    return out;
}

#ifdef ADAI_HAVE_LIBCURL
namespace {

bool host_allowed(const std::string& host, const std::vector<std::string>& allowed) {
    if (allowed.empty())
        return true;
    for (const auto& entry : allowed) {
        if (entry.empty())
            continue;
        if (entry.front() == '.') {
            if (host.size() > entry.size() &&
                host.compare(host.size() - entry.size(), entry.size(), entry) == 0)
                return true;
        } else if (host == entry) {
            return true;
        }
    }
    return false;
}

// Reads the scheme and host with libcurl's own URL parser, the one the transfer uses.
bool parse_scheme_and_host(const std::string& url, std::string& scheme, std::string& host) {
    CURLU* u = curl_url();
    if (!u)
        return false;
    bool ok = curl_url_set(u, CURLUPART_URL, url.c_str(), 0) == CURLUE_OK;
    char* part = nullptr;
    if (ok && curl_url_get(u, CURLUPART_SCHEME, &part, 0) == CURLUE_OK) {
        scheme = part;
        curl_free(part);
        part = nullptr;
    } else {
        ok = false;
    }
    if (ok && curl_url_get(u, CURLUPART_HOST, &part, 0) == CURLUE_OK) {
        host = part;
        curl_free(part);
    } else {
        ok = false;
    }
    curl_url_cleanup(u);
    return ok;
}

size_t write_to_file(char* data, size_t size, size_t count, void* file) {
    return std::fwrite(data, size, count, static_cast<std::FILE*>(file)) * size;
}

}  // namespace

HttpDownloadResult http_download(const HttpDownloadRequest& request) {
    HttpDownloadResult result;
    std::string scheme, host;
    if (!parse_scheme_and_host(request.url, scheme, host)) {
        result.error = "not a valid URL: " + request.url;
        return result;
    }
    if (scheme != "https" && !(request.allow_plain_http && scheme == "http")) {
        result.error = "only https:// URLs are downloaded, not " + scheme + "://";
        return result;
    }
    if (!host_allowed(host, request.allowed_hosts)) {
        result.error = "host '" + host + "' is not one this download may use";
        return result;
    }

    static std::once_flag global_init;
    std::call_once(global_init, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });

    const std::string part_path = request.output_path + ".part";
    std::FILE* out = std::fopen(part_path.c_str(), "wb");
    if (!out) {
        result.error = "cannot write " + part_path;
        return result;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::fclose(out);
        std::error_code ec;
        fs::remove(part_path, ec);
        result.error = "curl_easy_init failed";
        return result;
    }
    const char* protocols = request.allow_plain_http ? "http,https" : "https";
    char error_buffer[CURL_ERROR_SIZE] = {0};
    curl_slist* headers = nullptr;
    if (!request.bearer_token.empty()) {
        headers =
            curl_slist_append(headers, ("Authorization: Bearer " + request.bearer_token).c_str());
    }
    curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, protocols);
    curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, protocols);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    // The Authorization header stays with the original host (also libcurl's default).
    curl_easy_setopt(curl, CURLOPT_UNRESTRICTED_AUTH, 0L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, request.connect_timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, request.low_speed_limit_bytes);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, request.low_speed_time_seconds);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, request.max_time_seconds);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);

    const CURLcode code = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &result.http_status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    const bool closed = std::fclose(out) == 0;

    std::error_code ec;
    if (code != CURLE_OK) {
        result.error = error_buffer[0] ? error_buffer : curl_easy_strerror(code);
    } else if (!closed) {
        result.error = "cannot finish writing " + part_path;
    } else if (fs::file_size(part_path, ec) == 0 || ec) {
        result.error = "the server sent an empty body";
    } else {
        fs::rename(part_path, request.output_path, ec);
        if (ec) {
            result.error = "cannot move the download into place: " + ec.message();
        } else {
            result.ok = true;
            return result;
        }
    }
    fs::remove(part_path, ec);
    return result;
}

#else

HttpDownloadResult http_download(const HttpDownloadRequest&) {
    HttpDownloadResult result;
    result.error = "this build has no libcurl, so it can't download (install libcurl and rebuild)";
    return result;
}

#endif

}  // namespace adai
