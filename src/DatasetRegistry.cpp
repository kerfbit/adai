// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

#include "DatasetRegistry.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <utility>
#include "Logger.hpp"
#include "RegistryTransport.hpp"
#include "TrainingSampleMeta.hpp"

using adai::Logger;
namespace fs = std::filesystem;

// ANSI colour codes used by print_registry() for intentional TUI output.
#define COLOR_RESET "\033[0m"
#define COLOR_INFO "\033[1;36m"

namespace {
// ISO-8601 UTC timestamp, e.g. "2026-08-02T14:30:00Z" — same convention used
// server-side (RegistryServer.cpp's utc_now_string) for DataVersion::added_utc.
std::string utc_now_string() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
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
}  // namespace

// ============================================================================
// Transport factory (file-scope helper)
// ============================================================================

static std::unique_ptr<RegistryTransport> build_transport(const DatasetConfig& cfg) {
    if (!cfg.registry_server_url.empty()) {
#ifdef BUILD_METRICS_API_SERVER
        const std::string group =
            cfg.run_group.empty() ? fs::path(cfg.session_dir).filename().string() : cfg.run_group;
        return std::make_unique<RemoteTransport>(cfg.registry_server_url, group,
                                                 cfg.registry_timeout_ms);
#else
        // BUILD_METRICS_API_SERVER is reused here as a general "cpp-httplib was found at
        // configure time" signal (RemoteTransport is an HTTP client, unrelated to the
        // metrics dashboard) — named after the first feature that needed it. If this
        // fires, the actual missing dependency is cpp-httplib, not the metrics server.
        adai::Logger::warn(
            "REGISTRY_SERVER_URL is set but this binary was built without cpp-httplib "
            "(BUILD_METRICS_API_SERVER was OFF at configure time because HTTPLIB_INCLUDE_DIR "
            "was not found), so RemoteTransport is unavailable; falling back to LocalTransport, "
            "which reads only local flat files (data_registry.txt/pending_files.txt) and will "
            "not see files staged on the remote registry_server. Rebuild with cpp-httplib "
            "available (vendored at external/cpp-httplib) to fix this.");
#endif
    }
    const std::string reg_path = cfg.session_dir + "/" + cfg.data_registry_file;
    const std::string pend_path = cfg.session_dir + "/pending_files.txt";
    return std::make_unique<LocalTransport>(reg_path, pend_path);
}

// ============================================================================
// Constructor / factory
// ============================================================================

DatasetRegistry::DatasetRegistry(DatasetConfig cfg)
    : config_(std::move(cfg)), transport_(build_transport(config_)) {}

DatasetRegistry::DatasetRegistry(DatasetConfig cfg, std::unique_ptr<RegistryTransport> transport)
    : config_(std::move(cfg)), transport_(std::move(transport)) {}

/*static*/
DatasetConfig DatasetRegistry::make_config(const adai::ServiceConfig& svc) {
    DatasetConfig cfg;
    if (!svc.session_dir.empty()) {
        cfg.session_dir = svc.session_dir;
    }
    cfg.registry_server_url = svc.registry_server_url;
    cfg.run_group = svc.run_group;
    cfg.run_id = svc.run_id;
    cfg.cache_tokenized_data = svc.cache_tokenized_data;
    if (!svc.tokenized_cache_dir.empty()) {
        cfg.tokenized_cache_dir = svc.tokenized_cache_dir;
    }
    cfg.registry_timeout_ms = svc.registry_timeout_ms;
    cfg.model_name = svc.model_name;
    cfg.download_dir = svc.download_dir;
    cfg.max_parallel_downloads = svc.max_parallel_downloads;
    cfg.large_file_warn_threshold_mb = svc.large_file_warn_threshold_mb;
    return cfg;
}

// ============================================================================
// Private helpers
// ============================================================================

std::string DatasetRegistry::registry_file_path() const {
    return config_.session_dir + "/" + config_.data_registry_file;
}

std::string DatasetRegistry::pending_file_path() const {
    return config_.session_dir + "/pending_files.txt";
}

// ============================================================================
// Pending queue
// ============================================================================

bool DatasetRegistry::add_file(const std::string& path) {
    if (!fs::exists(path)) {
        Logger::error("Data file not found: {}", path);
        return false;
    }

    if (is_trained(path)) {
        Logger::warn("Data file already trained, skipping: {}", path);
        return false;
    }

    const bool already_pending = std::any_of(pending_.begin(), pending_.end(),
                                             [&](const PendingEntry& e) { return e.path == path; });
    if (already_pending) {
        Logger::warn("Data file already in pending queue: {}", path);
        return false;
    }

    if (!transport_->add_pending(path)) {
        Logger::error("Failed to persist pending entry for: {}", path);
        return false;
    }
    pending_.push_back({path, {}, {}});
    Logger::info("Added new data file: {}", path);
    return true;
}

bool DatasetRegistry::add_files(const std::vector<std::string>& paths) {
    int added = 0;
    for (const auto& file : paths) {
        if (add_file(file)) {
            ++added;
        }
    }
    Logger::info("Added {}/{} new data files", added, paths.size());
    return added > 0;
}

void DatasetRegistry::clear_pending() {
    pending_.clear();
}

bool DatasetRegistry::remove_pending(const std::string& path) {
    auto it = std::find_if(pending_.begin(), pending_.end(),
                           [&](const PendingEntry& e) { return e.path == path; });
    if (it == pending_.end()) {
        return false;
    }
    pending_.erase(it);
    save_pending_list();
    return true;
}

std::vector<std::string> DatasetRegistry::pending_files() const {
    std::vector<std::string> paths;
    paths.reserve(pending_.size());
    for (const auto& e : pending_)
        paths.push_back(e.path);
    return paths;
}

std::vector<PendingEntry> DatasetRegistry::pending_entries() const {
    return pending_;
}

AssignResult DatasetRegistry::assign_model(const std::string& model_name,
                                           const std::vector<std::string>& paths, int count) {
    auto result = transport_->assign(model_name, paths, count);
    if (!result.paths.empty()) {
        const std::set<std::string> touched(result.paths.begin(), result.paths.end());
        for (auto& e : pending_) {
            if (touched.count(e.path)) {
                e.model_name = model_name;
            }
        }
    }
    return result;
}

UnassignResult DatasetRegistry::unassign_model(const std::string& model_name,
                                               const std::vector<std::string>& paths, bool force) {
    auto result = transport_->unassign(model_name, paths, force);
    if (!result.paths.empty()) {
        const std::set<std::string> touched(result.paths.begin(), result.paths.end());
        for (auto& e : pending_) {
            if (touched.count(e.path)) {
                e.model_name.clear();
            }
        }
    }
    return result;
}

DeleteResult DatasetRegistry::delete_entries(const std::vector<std::string>& paths, bool force,
                                             bool delete_files) {
    auto result = transport_->delete_paths(paths, force, delete_files);
    for (const auto& d : result.details) {
        if (d.status != "deleted") {
            continue;
        }
        pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                      [&](const PendingEntry& e) { return e.path == d.path; }),
                       pending_.end());
        registry_.erase(
            std::remove_if(registry_.begin(), registry_.end(),
                           [&](const DataVersion& dv) { return dv.data_file == d.path; }),
            registry_.end());
        trained_set_.erase(d.path);
    }
    return result;
}

std::vector<std::string> DatasetRegistry::trained_files() const {
    return std::vector<std::string>(trained_set_.begin(), trained_set_.end());
}

bool DatasetRegistry::is_trained(const std::string& path) const {
    return trained_set_.find(path) != trained_set_.end();
}

// ============================================================================
// Mark trained (internal helper)
// ============================================================================

// Build new DataVersion entries for each untrained path; update in-memory state.
// Returns the newly created entries (does NOT persist — callers do that).
static std::vector<DataVersion> build_new_versions(const std::vector<std::string>& paths,
                                                   const std::vector<int>& sample_counts,
                                                   std::vector<DataVersion>& registry_,
                                                   std::set<std::string>& trained_set_) {
    std::vector<DataVersion> new_entries;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        const std::string& f = paths[i];
        if (trained_set_.count(f))
            continue;

        DataVersion dv;
        dv.data_file = f;
        dv.checksum = DatasetRegistry::compute_checksum(f);
        dv.num_samples = (i < sample_counts.size()) ? sample_counts[i] : 0;
        dv.added_utc = utc_now_string();
        dv.trained = true;

        new_entries.push_back(dv);
        registry_.push_back(dv);
        trained_set_.insert(f);
    }
    return new_entries;
}

// ============================================================================
// Mark trained — public API
// ============================================================================

void DatasetRegistry::mark_trained(const std::vector<std::string>& paths,
                                   const std::vector<int>& sample_counts) {
    build_new_versions(paths, sample_counts, registry_, trained_set_);
    save_registry();  // calls transport_->save_registry(registry_) for LocalTransport
}

// Phase 9 overload — atomic for both local (flock) and remote (POST /trained).
void DatasetRegistry::mark_trained(const std::string& run_id, const std::vector<std::string>& paths,
                                   const std::vector<int>& sample_counts) {
    const auto new_entries = build_new_versions(paths, sample_counts, registry_, trained_set_);
    transport_->commit_trained(run_id, new_entries, paths);

    // Update in-memory pending_ to remove trained files
    const std::set<std::string> trained_set(paths.begin(), paths.end());
    pending_.erase(
        std::remove_if(pending_.begin(), pending_.end(),
                       [&](const PendingEntry& e) { return trained_set.count(e.path) > 0; }),
        pending_.end());
}

// ============================================================================
// Multi-run API (Phase 9)
// ============================================================================

AcquireResponse DatasetRegistry::acquire_pending(const std::string& run_id, int max_files) {
    const int limit = (max_files > 0) ? max_files : config_.max_files_per_run;
    auto resp = transport_->acquire(run_id, limit, config_.model_name);

    // Reflect acquisition in in-memory pending_
    for (const auto& f : resp.files) {
        const bool already_in =
            std::any_of(pending_.begin(), pending_.end(),
                        [&](const PendingEntry& e) { return e.path == f.registry_path; });
        if (!already_in) {
            pending_.push_back({f.registry_path, run_id, {}});
        }
    }

    return resp;
}

std::string DatasetRegistry::next_session(const std::string& model_name,
                                          const std::string& run_id) {
    return transport_->next_session(model_name, run_id);
}

void DatasetRegistry::release_pending(const std::string& run_id,
                                      const std::vector<std::string>& paths) {
    transport_->release(run_id, paths);

    // Remove from in-memory pending_
    const std::set<std::string> rel_set(paths.begin(), paths.end());
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                  [&](const PendingEntry& e) { return rel_set.count(e.path) > 0; }),
                   pending_.end());
}

// ============================================================================
// Server-side dataset fetch (Phase 11)
// ============================================================================

std::string DatasetRegistry::remote_fetch_gutenberg(int book_id, int num_pairs,
                                                     const std::string& model_name) {
    const std::string path = transport_->fetch_gutenberg(book_id, num_pairs, model_name);
    if (!path.empty()) {
        pending_.push_back({path, {}, {}});
    }
    return path;
}

std::string DatasetRegistry::remote_fetch_huggingface(const std::string& dataset_id,
                                                       int num_pairs, const std::string& split,
                                                       const std::string& input_field,
                                                       const std::string& output_field,
                                                       const std::string& model_name) {
    const std::string path = transport_->fetch_huggingface(dataset_id, num_pairs, split,
                                                            input_field, output_field, model_name);
    if (!path.empty()) {
        pending_.push_back({path, {}, {}});
    }
    return path;
}

std::string DatasetRegistry::remote_upload(const std::string& local_path) {
    const std::string path = transport_->upload_file(local_path);
    if (!path.empty()) {
        pending_.push_back({path, {}, {}});
    }
    return path;
}

void DatasetRegistry::print_run_assignments() {
    std::vector<PendingEntry> entries;
    transport_->load_pending(entries);

    std::cout << COLOR_INFO << "\nRun Assignments:" << COLOR_RESET << '\n';
    std::cout << "Run ID                         | File\n";
    std::cout << "-------------------------------|-----\n";
    for (const auto& e : entries) {
        std::cout << std::setw(31) << (e.run_id.empty() ? "<unassigned>" : e.run_id) << " | "
                  << e.path << '\n';
    }
}

// ============================================================================
// Persistence
// ============================================================================

bool DatasetRegistry::load_registry() {
    std::vector<DataVersion> entries;
    if (!transport_->load_registry(entries)) {
        registry_.clear();
        trained_set_.clear();
        return false;
    }

    registry_.clear();
    trained_set_.clear();

    for (auto& dv : entries) {
        if (dv.trained) {
            trained_set_.insert(dv.data_file);
        }
        registry_.push_back(std::move(dv));
    }

    Logger::info("Loaded data registry: {} files ({} trained)", registry_.size(),
                 trained_set_.size());
    return true;
}

bool DatasetRegistry::save_registry() {
    if (!transport_->save_registry(registry_)) {
        Logger::error("Failed to save data registry");
        return false;
    }
    return true;
}

bool DatasetRegistry::load_pending_list() {
    std::vector<PendingEntry> entries;
    if (!transport_->load_pending(entries)) {
        return false;
    }

    pending_ = std::move(entries);

    if (!pending_.empty()) {
        Logger::info("Loaded {} pending data files", pending_.size());
    }

    return true;
}

bool DatasetRegistry::save_pending_list() {
    return transport_->save_pending(pending_);
}

// ============================================================================
// Reporting
// ============================================================================

void DatasetRegistry::print_registry() const {
    std::cout << COLOR_INFO << "\nData Registry:" << COLOR_RESET << '\n';
    std::cout << "Trained | Samples | Data File\n";
    std::cout << "--------|---------|----------\n";

    for (const auto& dv : registry_) {
        std::cout << std::setw(7) << (dv.trained ? "yes" : "no") << " | " << std::setw(7)
                  << dv.num_samples << " | " << dv.data_file << '\n';
    }
}

int DatasetRegistry::total_samples_trained() const {
    int total = 0;
    for (const auto& dv : registry_) {
        if (dv.trained) {
            total += dv.num_samples;
        }
    }
    return total;
}

// ============================================================================
// Training-file parser
// ============================================================================

/*static*/
int DatasetRegistry::load_conversation_pairs(const std::string& path,
                                             std::vector<ConversationPair>& pairs) {
    std::ifstream file(path);
    if (!file.is_open()) {
        Logger::error("Cannot open file: {}", path);
        return 0;
    }

    // Detect format from first non-empty line
    std::string first_line;
    while (std::getline(file, first_line)) {
        first_line.erase(0, first_line.find_first_not_of(" \t\r\n"));
        if (!first_line.empty())
            break;
    }
    file.seekg(0);

    int pair_count = 0;

    if (!first_line.empty() && first_line.front() == '{') {
        // JSONL training format
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line.front() != '{')
                continue;
            std::string in, resp;
            SampleMeta meta;
            if (parse_jsonl_sample(line, in, resp, meta)) {
                pairs.emplace_back(std::move(in), std::move(resp), std::move(meta));
                ++pair_count;
            }
        }
    } else {
        // Legacy INPUT:/RESPONSE: format
        std::string line, current_input, current_response;
        while (std::getline(file, line)) {
            line.erase(0, line.find_first_not_of(" \t\n\r"));
            line.erase(line.find_last_not_of(" \t\n\r") + 1);

            if (line.empty()) {
                if (!current_input.empty() && !current_response.empty()) {
                    pairs.emplace_back(current_input, current_response);
                    ++pair_count;
                    current_input.clear();
                    current_response.clear();
                }
                continue;
            }

            if (line.substr(0, 6) == "INPUT:") {
                if (!current_input.empty() && !current_response.empty()) {
                    pairs.emplace_back(current_input, current_response);
                    ++pair_count;
                    current_input.clear();
                    current_response.clear();
                }
                current_input = line.substr(6);
                current_input.erase(0, current_input.find_first_not_of(" \t"));
            } else if (line.substr(0, 9) == "RESPONSE:") {
                current_response = line.substr(9);
                current_response.erase(0, current_response.find_first_not_of(" \t"));
            }
        }
        if (!current_input.empty() && !current_response.empty()) {
            pairs.emplace_back(current_input, current_response);
            ++pair_count;
        }
    }

    file.close();
    Logger::info("Loaded {} pairs from: {}", pair_count, path);
    return pair_count;
}

// ============================================================================
// Checksum
// ============================================================================

/*static*/
std::string DatasetRegistry::compute_checksum(const std::string& path) {
    if (!fs::exists(path)) {
        return "MISSING";
    }

    auto size = fs::file_size(path);
    auto ftime = fs::last_write_time(path);

    std::ostringstream oss;
    // file_time_type::duration::rep is __int128 on macOS (libc++) which has no
    // operator<< overload — cast to long long to keep it portable.
    oss << size << "_" << static_cast<long long>(ftime.time_since_epoch().count());
    return oss.str();
}
