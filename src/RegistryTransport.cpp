#include "RegistryTransport.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <sys/file.h>  // flock()
#include <fcntl.h>     // open(), O_RDWR
#include <unistd.h>    // close()
#include "Logger.hpp"

#ifdef BUILD_METRICS_API_SERVER
#include <httplib.h>
#endif

using adai::Logger;
namespace fs = std::filesystem;

// ============================================================================
// LocalTransport
// ============================================================================

LocalTransport::LocalTransport(std::string registry_path, std::string pending_path)
    : registry_path_(std::move(registry_path))
    , pending_path_(std::move(pending_path)) {}

bool LocalTransport::load_registry(std::vector<DataVersion>& out) {
    if (!fs::exists(registry_path_)) {
        return false;
    }

    std::ifstream file(registry_path_);
    if (!file.is_open()) {
        return false;
    }

    out.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        DataVersion dv;
        int trained_int = 0;
        iss >> dv.data_file >> dv.checksum >> dv.num_samples >> trained_int;
        dv.trained = (trained_int == 1);
        // model_id is optional 5th column; absent in pre-Phase-2 files
        std::string mid;
        if (iss >> mid) dv.model_id = mid;
        out.push_back(std::move(dv));
    }

    return true;
}

bool LocalTransport::save_registry(const std::vector<DataVersion>& entries) {
    fs::create_directories(fs::path(registry_path_).parent_path());

    std::ofstream file(registry_path_);
    if (!file.is_open()) {
        return false;
    }

    file << "# Data Registry: data_file checksum num_samples trained model_id\n";
    for (const auto& dv : entries) {
        file << dv.data_file << " " << dv.checksum << " " << dv.num_samples << " "
             << (dv.trained ? 1 : 0) << " " << dv.model_id << "\n";
    }

    return file.good();
}

bool LocalTransport::load_pending(std::vector<PendingEntry>& out) {
    if (!fs::exists(pending_path_)) {
        return false;
    }

    std::ifstream file(pending_path_);
    if (!file.is_open()) {
        return false;
    }

    out.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        // Format: "path" or "path\trun_id" (tab-separated; Phase 9 extension)
        const auto tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            out.push_back({line.substr(0, tab_pos), line.substr(tab_pos + 1)});
        } else {
            out.push_back({std::move(line), {}});
        }
    }

    return true;
}

bool LocalTransport::save_pending(const std::vector<PendingEntry>& entries) {
    fs::create_directories(fs::path(pending_path_).parent_path());

    std::ofstream file(pending_path_);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& e : entries) {
        if (e.run_id.empty()) {
            file << e.path << '\n';
        } else {
            file << e.path << '\t' << e.run_id << '\n';
        }
    }

    return file.good();
}

// ── Lock helpers ──────────────────────────────────────────────────────────────

int LocalTransport::lock_pending() const {
    const std::string lock_path = pending_path_ + ".lock";
    fs::create_directories(fs::path(pending_path_).parent_path());
    const int fd = open(lock_path.c_str(), O_CREAT | O_WRONLY, 0600);
    if (fd < 0) {
        return -1;
    }
    if (flock(fd, LOCK_EX) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

void LocalTransport::unlock_pending(int lock_fd) const {
    if (lock_fd >= 0) {
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
    }
}

// ── Phase 9: acquire / release / commit_trained ───────────────────────────────

AcquireResponse LocalTransport::acquire(const std::string& run_id, int max_files) {
    AcquireResponse resp;
    resp.run_id = run_id;
    // ftp_server_host left empty → caller uses direct filesystem paths

    if (!fs::exists(pending_path_)) {
        return resp;
    }

    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::acquire — failed to acquire pending lock");
        return resp;
    }

    std::vector<PendingEntry> entries;
    load_pending(entries);

    const int limit = (max_files > 0) ? max_files : std::numeric_limits<int>::max();

    for (auto& e : entries) {
        if (e.run_id.empty() && static_cast<int>(resp.files.size()) < limit) {
            e.run_id = run_id;
            FileToken tok;
            tok.registry_path = e.path;
            resp.files.push_back(std::move(tok));
        }
    }

    if (!resp.files.empty()) {
        save_pending(entries);
        Logger::info("Acquired {} pending files for run '{}'", resp.files.size(), run_id);
    }

    unlock_pending(lock_fd);
    return resp;
}

void LocalTransport::release(const std::string& run_id,
                              const std::vector<std::string>& paths) {
    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::release — failed to acquire pending lock");
        return;
    }

    std::vector<PendingEntry> entries;
    load_pending(entries);

    const std::set<std::string> to_release(paths.begin(), paths.end());
    for (auto& e : entries) {
        if (e.run_id == run_id && to_release.count(e.path)) {
            e.run_id.clear();
        }
    }

    save_pending(entries);
    Logger::info("Released {} files back to pending queue (run '{}')", paths.size(), run_id);

    unlock_pending(lock_fd);
}

void LocalTransport::commit_trained(const std::string& run_id,
                                     const std::vector<DataVersion>& new_entries,
                                     const std::vector<std::string>& trained_paths) {
    // ── Registry ──────────────────────────────────────────────────────────
    // Load existing registry, append new entries, save.
    std::vector<DataVersion> registry;
    load_registry(registry);
    for (const auto& e : new_entries) {
        registry.push_back(e);
    }
    save_registry(registry);

    // ── Pending ───────────────────────────────────────────────────────────
    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::commit_trained — failed to acquire pending lock");
        return;
    }

    std::vector<PendingEntry> pending;
    load_pending(pending);

    const std::set<std::string> trained_set(trained_paths.begin(), trained_paths.end());
    pending.erase(std::remove_if(pending.begin(), pending.end(),
        [&](const PendingEntry& e) {
            return trained_set.count(e.path) &&
                   (run_id.empty() || e.run_id == run_id);
        }), pending.end());

    save_pending(pending);
    unlock_pending(lock_fd);
}

bool LocalTransport::add_pending(const std::string& path) {
    const int lock_fd = lock_pending();
    if (lock_fd < 0) {
        Logger::error("LocalTransport::add_pending — failed to acquire pending lock");
        return false;
    }

    std::vector<PendingEntry> entries;
    load_pending(entries);
    for (const auto& e : entries) {
        if (e.path == path) {
            unlock_pending(lock_fd);
            return true;  // already queued
        }
    }

    fs::create_directories(fs::path(pending_path_).parent_path());
    std::ofstream file(pending_path_, std::ios::app);
    const bool ok = file.is_open() && (file << path << '\n') && file.good();
    unlock_pending(lock_fd);
    return ok;
}

// ============================================================================
// RemoteTransport
// ============================================================================

namespace {
// Minimal JSON helpers shared between RemoteTransport methods ----------------

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// Extract the value of a simple JSON string key: "key":"value"
// Returns empty string if not found.
std::string json_string(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    const auto pos = body.find(needle);
    if (pos == std::string::npos) return {};
    const auto start = pos + needle.size();
    const auto end   = body.find('"', start);
    if (end == std::string::npos) return {};
    return body.substr(start, end - start);
}

// Parse a JSON array of strings from body["key"]: ["a","b",...]
std::vector<std::string> json_string_array(const std::string& body, const std::string& key) {
    std::vector<std::string> result;
    const std::string needle = "\"" + key + "\":[";
    const auto pos = body.find(needle);
    if (pos == std::string::npos) return result;
    auto cur = pos + needle.size();
    while (cur < body.size()) {
        cur = body.find('"', cur);
        if (cur == std::string::npos) break;
        ++cur;
        const auto end = body.find('"', cur);
        if (end == std::string::npos) break;
        result.push_back(body.substr(cur, end - cur));
        cur = end + 1;
        // Stop at ]
        const auto next = body.find_first_of(",]", cur);
        if (next == std::string::npos || body[next] == ']') break;
        cur = next + 1;
    }
    return result;
}

// Build a JSON array of quoted strings: ["a","b",...]
std::string json_array(const std::vector<std::string>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ',';
        out += '"'; out += json_escape(v[i]); out += '"';
    }
    out += ']';
    return out;
}

// Parse URL "http://host[:port]" → {host, port}
std::pair<std::string, int> parse_host_port(const std::string& url) {
    std::string s = url;
    for (const char* prefix : {"http://", "https://"}) {
        const std::string p = prefix;
        if (s.rfind(p, 0) == 0) { s = s.substr(p.size()); break; }
    }
    // strip any trailing path
    const auto slash = s.find('/');
    if (slash != std::string::npos) s = s.substr(0, slash);

    const auto colon = s.rfind(':');
    if (colon != std::string::npos) {
        try {
            return {s.substr(0, colon), std::stoi(s.substr(colon + 1))};
        } catch (...) {}
    }
    return {s, 8082};
}

} // anonymous namespace

RemoteTransport::RemoteTransport(std::string base_url, std::string run_group, int timeout_ms)
    : timeout_ms_(timeout_ms) {
    auto [host, port] = parse_host_port(base_url);
    host_         = std::move(host);
    port_         = port;
    group_prefix_ = "/registry/" + run_group;
}

bool RemoteTransport::load_registry(std::vector<DataVersion>& out) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    const auto res = cli.Get((group_prefix_ + "/registry").c_str());
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::load_registry — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/registry");
        return false;
    }

    // Parse JSON array: {"entries":[{"data_file":"...","checksum":"...","num_samples":N,"trained":T},...]}
    out.clear();
    const std::string& body = res->body;
    auto entries_start = body.find("\"entries\":[");
    if (entries_start == std::string::npos) return false;
    auto cur = entries_start + 11; // len of "entries":[
    while (cur < body.size()) {
        auto obj_start = body.find('{', cur);
        if (obj_start == std::string::npos) break;
        auto obj_end = body.find('}', obj_start);
        if (obj_end == std::string::npos) break;
        const std::string obj = body.substr(obj_start, obj_end - obj_start + 1);
        DataVersion dv;
        dv.data_file   = json_string(obj, "data_file");
        dv.checksum    = json_string(obj, "checksum");
        const auto ns  = json_string(obj, "num_samples");
        dv.num_samples = ns.empty() ? 0 : std::stoi(ns);
        const auto tr  = json_string(obj, "trained");
        dv.trained     = (tr == "true" || tr == "1");
        if (!dv.data_file.empty()) out.push_back(std::move(dv));
        cur = obj_end + 1;
        if (body.find(']', cur) < body.find('{', cur)) break;
    }
    return true;
#else
    Logger::error("RemoteTransport::load_registry — not compiled (BUILD_METRICS_API_SERVER not set)");
    return false;
#endif
}

bool RemoteTransport::save_registry(const std::vector<DataVersion>& /*entries*/) {
    // In distributed mode the server owns the registry; clients use commit_trained().
    Logger::warn("RemoteTransport::save_registry — no-op; use mark_trained(run_id,...) in distributed mode");
    return false;
}

bool RemoteTransport::load_pending(std::vector<PendingEntry>& out) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    const auto res = cli.Get((group_prefix_ + "/queue").c_str());
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::load_pending — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/queue");
        return false;
    }

    // Parse {"entries":[{"path":"...","run_id":"..."},...]}
    out.clear();
    const std::string& body = res->body;
    auto cur = body.find("\"entries\":[");
    if (cur == std::string::npos) return false;
    cur += 11;
    while (cur < body.size()) {
        auto obj_start = body.find('{', cur);
        if (obj_start == std::string::npos) break;
        auto obj_end = body.find('}', obj_start);
        if (obj_end == std::string::npos) break;
        const std::string obj = body.substr(obj_start, obj_end - obj_start + 1);
        const std::string path   = json_string(obj, "path");
        const std::string run_id = json_string(obj, "run_id");
        if (!path.empty()) out.push_back({path, run_id});
        cur = obj_end + 1;
        if (body.find(']', cur) < body.find('{', cur)) break;
    }
    return true;
#else
    Logger::error("RemoteTransport::load_pending — not compiled");
    return false;
#endif
}

bool RemoteTransport::save_pending(const std::vector<PendingEntry>& /*entries*/) {
    Logger::warn("RemoteTransport::save_pending — no-op; use acquire/release/commit_trained in distributed mode");
    return false;
}

AcquireResponse RemoteTransport::acquire(const std::string& run_id, int max_files) {
    AcquireResponse resp;
    resp.run_id = run_id;
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"run_id\":\"" << json_escape(run_id) << "\","
         << "\"max_files\":" << max_files << "}";

    const auto res = cli.Post((group_prefix_ + "/acquire").c_str(),
                              body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::acquire — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/acquire");
        return resp;
    }

    const std::string& b = res->body;

    // Detect new structured response: has "files":[ array.
    // Old response format: {"acquired":["path",...]}
    const bool new_format = (b.find("\"files\":[") != std::string::npos);

    if (new_format) {
        // Parse top-level scalar fields
        resp.ftp_server_host = json_string(b, "ftp_server_host");
        // ftp_server_port is a JSON number, not a string
        const std::string port_needle = "\"ftp_server_port\":";
        const auto pp = b.find(port_needle);
        if (pp != std::string::npos) {
            try { resp.ftp_server_port = std::stoi(b.substr(pp + port_needle.size())); }
            catch (...) {}
        }
        // ftps_enabled is a JSON boolean (Phase 3)
        const std::string ftps_needle = "\"ftps_enabled\":";
        const auto fp = b.find(ftps_needle);
        if (fp != std::string::npos) {
            const std::string fval = b.substr(fp + ftps_needle.size(), 5);
            resp.ftps_enabled = (fval.compare(0, 4, "true") == 0);
        }

        // Parse each file object inside "files":[{...},{...}]
        auto files_pos = b.find("\"files\":[");
        if (files_pos != std::string::npos) {
            auto cur = files_pos + 9; // len("\"files\":[")
            while (cur < b.size()) {
                auto obj_start = b.find('{', cur);
                if (obj_start == std::string::npos) break;
                // Find matching closing brace (objects are flat, no nesting)
                auto obj_end = b.find('}', obj_start);
                if (obj_end == std::string::npos) break;
                const std::string obj = b.substr(obj_start, obj_end - obj_start + 1);

                FileToken tok;
                tok.registry_path     = json_string(obj, "registry_path");
                tok.ftp_path          = json_string(obj, "ftp_path");
                tok.ftp_username      = json_string(obj, "ftp_username");
                tok.ftp_password      = json_string(obj, "ftp_password");
                tok.checksum          = json_string(obj, "checksum");
                tok.token_expires_utc = json_string(obj, "token_expires_utc");
                // size_bytes is a JSON number
                const std::string sb_needle = "\"size_bytes\":";
                const auto sbp = obj.find(sb_needle);
                if (sbp != std::string::npos) {
                    try { tok.size_bytes = static_cast<std::size_t>(
                            std::stoull(obj.substr(sbp + sb_needle.size()))); }
                    catch (...) {}
                }
                if (!tok.registry_path.empty()) {
                    resp.files.push_back(std::move(tok));
                }
                cur = obj_end + 1;
                if (b.find(']', cur) < b.find('{', cur)) break;
            }
        }
    } else {
        // Legacy format: {"acquired":["path1","path2",...]}
        // Wrap each path in a minimal FileToken with ftp fields empty.
        auto paths = json_string_array(b, "acquired");
        for (auto& p : paths) {
            FileToken tok;
            tok.registry_path = std::move(p);
            resp.files.push_back(std::move(tok));
        }
    }

    Logger::info("RemoteTransport: acquired {} files for run '{}'", resp.files.size(), run_id);
#else
    Logger::error("RemoteTransport::acquire — not compiled");
#endif
    return resp;
}

void RemoteTransport::release(const std::string& run_id,
                               const std::vector<std::string>& paths) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"run_id\":\"" << json_escape(run_id) << "\","
         << "\"files\":" << json_array(paths) << "}";

    const auto res = cli.Post((group_prefix_ + "/release").c_str(),
                              body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::release — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/release");
    }
#else
    Logger::error("RemoteTransport::release — not compiled");
#endif
}

void RemoteTransport::commit_trained(const std::string& run_id,
                                      const std::vector<DataVersion>& new_entries,
                                      const std::vector<std::string>& trained_paths) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    // Build files and samples arrays from new_entries
    std::vector<std::string> files;
    std::vector<int> samples;
    for (const auto& e : new_entries) {
        files.push_back(e.data_file);
        samples.push_back(e.num_samples);
    }
    // Also include any trained_paths not in new_entries (already in registry)
    std::set<std::string> files_set(files.begin(), files.end());
    for (const auto& p : trained_paths) {
        if (!files_set.count(p)) {
            files.push_back(p);
            samples.push_back(0);
        }
    }

    std::ostringstream body;
    body << "{\"run_id\":\"" << json_escape(run_id) << "\","
         << "\"files\":" << json_array(files) << ","
         << "\"samples\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) body << ',';
        body << samples[i];
    }
    body << "]}";

    const auto res = cli.Post((group_prefix_ + "/trained").c_str(),
                              body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::commit_trained — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/trained");
    }
#else
    Logger::error("RemoteTransport::commit_trained — not compiled");
#endif
}

bool RemoteTransport::add_pending(const std::string& path) {
#ifdef BUILD_METRICS_API_SERVER
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(0, timeout_ms_ * 1000);
    cli.set_read_timeout(0, timeout_ms_ * 1000);

    std::ostringstream body;
    body << "{\"path\":\"" << json_escape(path) << "\"}";

    const auto res = cli.Post((group_prefix_ + "/pending/add").c_str(),
                              body.str(), "application/json");
    if (!res || res->status != 200) {
        Logger::error("RemoteTransport::add_pending — HTTP {} from {}:{}{}",
                      res ? res->status : -1, host_, port_, group_prefix_ + "/pending/add");
        return false;
    }
    if (res->body.find("\"added\":false") != std::string::npos) {
        Logger::warn("RemoteTransport::add_pending — '{}' already in remote queue", path);
        return false;
    }
    return true;
#else
    Logger::error("RemoteTransport::add_pending — not compiled");
    return false;
#endif
}
