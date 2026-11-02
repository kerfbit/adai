#pragma once

// @adai-status: experimental
// @adai-version: 0.5.0
// @adai-reviewed: 2026-11-01

// TD-035: mns_cli's per-command argument parsing and JSON-body construction, pulled out of
// MnsCliTool.cpp so it's testable without a real mns_server to talk to — see
// tests/mns_cli_commands_test.cpp. Each build_*_request() function is pure (no network I/O): it takes
// the command's own argv slice (and, for `register`, the loaded ServiceConfig for the run_group
// default) and returns the HTTP method/path/body mns_cli would send, or an error message if the
// arguments don't parse: an unknown option, a flag missing its value, a number that isn't whole
// and in range, or a stray argument (TD-448). Actually sending the request and printing the
// response stays in MnsCliTool.cpp's cmd_*() wrappers, unchanged.

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

    // Printed to stderr before the request is sent: register's architecture and where it came from
    // (TD-450). Empty for every other command.
    std::string notice;
};

// mns_cli's server URL (kept independent of ModelNameClient::ParsedUrl, which lives in a different
// library this CLI doesn't otherwise need). TD-451: http:// (default port 8083) or https:// (443);
// a path is kept as a prefix for every request; a bad scheme, host or port sets `error`.
struct ParsedUrl {
    std::string host = "localhost";
    int port = 8083;
    bool https = false;
    std::string base_path;  // "" or "/prefix", no trailing '/'
    std::string error;

    // scheme://host:port<base_path><path>
    std::string url_for(const std::string& path) const;
};
ParsedUrl parse_url(const std::string& raw);

// TD-449: control characters below 0x20 come out as \u00XX.
std::string json_escape(const std::string& s);
// TD-449: percent-encodes everything except RFC 3986 unreserved characters, for path segments and query values.
std::string url_encode(const std::string& s);

MnsCliRequest build_list_request(const std::vector<std::string>& args);
MnsCliRequest build_get_request(const std::vector<std::string>& args);
// TD-450: needs the whole architecture (unless it links --encoder/--decoder): every field from a
// flag, or the missing ones from --from-config <file> (the trainer's config, read like the trainer
// reads it, environment included). A missing field is an error; nothing comes from `cfg`, which only
// supplies the run_group default.
MnsCliRequest build_register_request(const std::vector<std::string>& args,
                                     const ServiceConfig& cfg);
MnsCliRequest build_update_run_group_request(const std::vector<std::string>& args);
MnsCliRequest build_resolve_request(const std::vector<std::string>& args);
MnsCliRequest build_set_training_request(const std::vector<std::string>& args);
MnsCliRequest build_set_candidate_request(const std::vector<std::string>& args);
MnsCliRequest build_delete_request(const std::vector<std::string>& args);
MnsCliRequest build_roles_request(const std::vector<std::string>& args = {});
MnsCliRequest build_resolve_role_request(const std::vector<std::string>& args);
MnsCliRequest build_promote_request(const std::vector<std::string>& args);
MnsCliRequest build_health_request(const std::vector<std::string>& args = {});
// TD-196: attach/replace/detach a chatbot-kind record's world-model connection.
MnsCliRequest build_link_world_model_request(const std::vector<std::string>& args);

}  // namespace adai
