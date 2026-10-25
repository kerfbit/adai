// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-24

#pragma once

#include <optional>
#include <set>
#include <string>
#include "Logger.hpp"

namespace adai {

/**
 * TD-526/TD-550: refuse writes a web page could make from an operator's browser.
 *
 * None of the HTTP daemons authenticate requests, and most parse request bodies whatever their
 * Content-Type. A page on any origin can therefore make the browser send a CORS "simple" request
 * (POST with text/plain, form-encoded or multipart, no preflight) that the server executes even
 * though the page can't read the reply. Browsers always send `Origin` on cross-origin requests and
 * on POSTs; the trainer, the CLIs, the Android app and curl don't. So a write is refused when:
 *   - it carries an `Origin` that isn't in @p allowed_origins (403), or
 *   - its Content-Type is one a browser may send without a preflight (415).
 * Reads (GET/HEAD/OPTIONS) are never refused here.
 */
struct BrowserWriteRefusal {
    int status;           ///< 403 or 415
    std::string message;  ///< for the JSON error body
};

bool is_write_method(const std::string& method);

/// text/plain, application/x-www-form-urlencoded or multipart/form-data (parameters ignored).
bool is_cors_simple_content_type(const std::string& content_type);

std::optional<BrowserWriteRefusal> browser_write_refusal(
    const std::string& method, const std::string& origin, const std::string& content_type,
    const std::set<std::string>& allowed_origins);

/// Minimal JSON error body: {"error":"<message>"} with `"` and `\` escaped.
std::string browser_write_refusal_json(const BrowserWriteRefusal& refusal);

/**
 * Installs the check as @p server's pre-routing handler (an httplib::Server; templated so this
 * header needn't include httplib.h). httplib keeps one pre-routing handler per server, so a server
 * that needs its own must call browser_write_refusal() from it instead.
 */
template <typename Server>
void install_browser_write_guard(Server& server, const std::string& service_name,
                                 std::set<std::string> allowed_origins = {}) {
    server.set_pre_routing_handler(
        [service_name, allowed = std::move(allowed_origins)](const auto& req, auto& res) {
            const auto refusal =
                browser_write_refusal(req.method, req.get_header_value("Origin"),
                                      req.get_header_value("Content-Type"), allowed);
            if (!refusal)
                return Server::HandlerResponse::Unhandled;
            Logger::warn("{}: refused {} {} (origin '{}', Content-Type '{}'): {}", service_name,
                         req.method, req.path, req.get_header_value("Origin"),
                         req.get_header_value("Content-Type"), refusal->message);
            res.status = refusal->status;
            res.set_content(browser_write_refusal_json(*refusal), "application/json");
            return Server::HandlerResponse::Handled;
        });
}

}  // namespace adai
