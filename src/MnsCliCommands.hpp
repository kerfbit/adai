#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

// TD-035: mns_cli's per-command argument parsing and JSON-body construction, pulled out of
// MnsCliTool.cpp so it's testable without a real mns_server to talk to — see
// MnsCliCommands_test.cpp. Each build_*_request() function is pure (no network I/O): it takes
// the command's own argv slice (and, for `register`, the loaded ServiceConfig for architecture
// defaults) and returns the HTTP method/path/body mns_cli would send, or an error message if the
// arguments don't parse. Actually sending the request and printing the response stays in
// MnsCliTool.cpp's cmd_*() wrappers, unchanged.

#include <map>
#include <optional>
#include <string>
#include <vector>
#include "Config.hpp"

namespace adai {

enum class HttpMethod { Get, Post, Put, Delete };

struct MnsCliRequest {
    HttpMethod method = HttpMethod::Get;
    std::string path;
    std::string body;  // empty for Get/Delete

    bool error = false;
    std::string error_message;  // already newline-free
};

// Same lightweight host:port splitter mns_cli has always used (kept independent of
// ModelNameClient::ParsedUrl, which lives in a different library this CLI doesn't otherwise
// need).
struct ParsedUrl {
    std::string host = "localhost";
    int port = 8083;
};
ParsedUrl parse_url(const std::string& raw);

std::string json_escape(const std::string& s);

MnsCliRequest build_list_request(const std::vector<std::string>& args);
MnsCliRequest build_get_request(const std::vector<std::string>& args);
MnsCliRequest build_register_request(const std::vector<std::string>& args,
                                     const ServiceConfig& cfg);
MnsCliRequest build_update_run_group_request(const std::vector<std::string>& args);
MnsCliRequest build_resolve_request(const std::vector<std::string>& args);
MnsCliRequest build_set_training_request(const std::vector<std::string>& args);
MnsCliRequest build_set_candidate_request(const std::vector<std::string>& args);
MnsCliRequest build_delete_request(const std::vector<std::string>& args);
MnsCliRequest build_roles_request();
MnsCliRequest build_resolve_role_request(const std::vector<std::string>& args);
MnsCliRequest build_promote_request(const std::vector<std::string>& args);
MnsCliRequest build_health_request();

}  // namespace adai
