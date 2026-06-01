#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "TrainingMetricsService.hpp"

struct MetricsSessionSummary {
    std::string key;
    int session_id = 0;
    bool is_training = false;
    int current_epoch = 0;
    int total_epochs = 0;
    float current_loss = 0.0f;
    float current_validation_loss = 0.0f;
    float best_validation_loss = std::numeric_limits<float>::max();
    std::chrono::system_clock::time_point session_start_time;
    std::chrono::system_clock::time_point last_update_time;
    std::string metrics_url;
};

class MetricsSessionRegistry {
   public:
    explicit MetricsSessionRegistry(MetricsServiceConfig base_config = MetricsServiceConfig(),
                                    size_t max_live_sessions = 16,
                                    int completed_ttl_seconds = 3600)
        : base_config_(std::move(base_config)),
          max_live_sessions_(max_live_sessions),
          completed_ttl_seconds_(completed_ttl_seconds) {}

    std::shared_ptr<TrainingMetricsService> create_or_get_session(const std::string& key) {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);

        evict_completed_sessions_locked(completed_ttl_seconds_);

        auto existing = sessions_.find(key);
        if (existing != sessions_.end()) {
            if (should_replace_completed_session(existing->second.service)) {
                sessions_.erase(existing);
            } else {
                return existing->second.service;
            }
        }

        if (sessions_.size() >= max_live_sessions_) {
            return nullptr;
        }

        auto service = std::make_shared<TrainingMetricsService>(config_for_session(key));
        sessions_.emplace(key, SessionEntry{service, std::chrono::system_clock::now()});
        return service;
    }

    std::optional<std::shared_ptr<TrainingMetricsService>> get_session(const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);
        auto existing = sessions_.find(key);
        if (existing == sessions_.end()) {
            return std::nullopt;
        }
        return existing->second.service;
    }

    std::vector<MetricsSessionSummary> list_sessions() const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);

        std::vector<MetricsSessionSummary> summaries;
        summaries.reserve(sessions_.size());
        for (const auto& [key, entry] : sessions_) {
            auto snapshot = entry.service->get_current_snapshot();
            MetricsSessionSummary summary;
            summary.key = key;
            summary.session_id = snapshot.session_id;
            summary.is_training = snapshot.is_training;
            summary.current_epoch = snapshot.current_epoch;
            summary.total_epochs = snapshot.total_epochs;
            summary.current_loss = snapshot.current_loss;
            summary.current_validation_loss = snapshot.current_validation_loss;
            summary.best_validation_loss = snapshot.best_validation_loss;
            summary.session_start_time = snapshot.session_start_time;
            summary.last_update_time = snapshot.last_update_time;
            summary.metrics_url = "/api/sessions/" + key + "/metrics/current";
            summaries.push_back(std::move(summary));
        }
        return summaries;
    }

    size_t evict_completed_sessions(int max_age_seconds) {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        return evict_completed_sessions_locked(max_age_seconds);
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);
        return sessions_.size();
    }

    size_t max_live_sessions() const {
        return max_live_sessions_;
    }

   private:
    struct SessionEntry {
        std::shared_ptr<TrainingMetricsService> service;
        std::chrono::system_clock::time_point created_at;
    };

    MetricsServiceConfig config_for_session(const std::string& key) const {
        if (key == "0-default") {
            return base_config_;
        }

        MetricsServiceConfig config = base_config_;
        config.metrics_file = derive_session_file_path(base_config_.metrics_file, key, "_metrics");
        config.summary_file =
            derive_session_file_path(base_config_.summary_file, key, "_metrics_summary");
        config.prometheus_file =
            derive_session_file_path(base_config_.prometheus_file, key, "_metrics");
        config.abnormal_samples_file = derive_session_file_path(base_config_.abnormal_samples_file,
                                                                key, "_abnormal_samples");
        return config;
    }

    static std::string derive_session_file_path(const std::string& base_file,
                                                const std::string& key,
                                                const std::string& suffix) {
        namespace fs = std::filesystem;

        fs::path path(base_file);
        const auto extension = path.extension().string();
        const auto parent = path.parent_path();
        return (parent / (key + suffix + extension)).string();
    }

    static bool has_started(const TrainingMetricsSnapshot& snapshot) {
        return snapshot.session_id != 0 || snapshot.total_epochs != 0 || snapshot.total_samples != 0 ||
               snapshot.total_samples_trained != 0;
    }

    static bool should_replace_completed_session(
        const std::shared_ptr<TrainingMetricsService>& service) {
        const auto snapshot = service->get_current_snapshot();
        return has_started(snapshot) && !snapshot.is_training;
    }

    static bool is_completed_and_stale(const SessionEntry& entry, int max_age_seconds) {
        const auto snapshot = entry.service->get_current_snapshot();
        if (snapshot.is_training || !has_started(snapshot)) {
            return false;
        }

        const auto last_activity = snapshot.last_update_time.time_since_epoch().count() > 0
                                       ? snapshot.last_update_time
                                       : entry.created_at;
        const auto max_age = std::chrono::seconds(std::max(0, max_age_seconds));
        return (std::chrono::system_clock::now() - last_activity) >= max_age;
    }

    size_t evict_completed_sessions_locked(int max_age_seconds) {
        std::vector<std::string> expired_keys;
        expired_keys.reserve(sessions_.size());
        for (const auto& [key, entry] : sessions_) {
            if (is_completed_and_stale(entry, max_age_seconds)) {
                expired_keys.push_back(key);
            }
        }

        for (const auto& key : expired_keys) {
            sessions_.erase(key);
        }
        return expired_keys.size();
    }

    MetricsServiceConfig base_config_;
    mutable std::shared_mutex registry_mutex_;
    std::unordered_map<std::string, SessionEntry> sessions_;
    size_t max_live_sessions_;
    int completed_ttl_seconds_;
};