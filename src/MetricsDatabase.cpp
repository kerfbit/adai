// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-11-04

// TD-441: the IMetricsDatabase factory and the shared row-cap planner, moved out of
// SQLiteMetricsDatabase.cpp (the factory picks a backend, so it doesn't belong to one).

#include "MetricsDatabase.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include "IMetricsReporter.hpp"
#include "Logger.hpp"
#include "SQLiteMetricsDatabase.hpp"
#include "nlohmann/json.hpp"

std::string encode_low_quality_samples(const std::vector<LowQualitySample>& samples) {
    nlohmann::json out = nlohmann::json::array();
    for (const LowQualitySample& s : samples) {
        out.push_back(
            {{"quality", std::isfinite(s.quality) ? nlohmann::json(s.quality) : nlohmann::json()},
             {"input_text", s.input_text},
             {"target_text", s.target_text}});
    }
    return out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::vector<LowQualitySample> decode_low_quality_samples(const std::string& json) {
    std::vector<LowQualitySample> samples;
    const nlohmann::json in = nlohmann::json::parse(json, nullptr, false);
    if (!in.is_array()) {
        return samples;
    }
    for (const auto& item : in) {
        if (!item.is_object()) {
            continue;
        }
        LowQualitySample s;
        const auto q = item.find("quality");
        s.quality = (q != item.end() && q->is_number()) ? q->get<float>()
                                                        : std::numeric_limits<float>::quiet_NaN();
        const auto in_text = item.find("input_text");
        const auto target = item.find("target_text");
        s.input_text =
            (in_text != item.end() && in_text->is_string()) ? in_text->get<std::string>() : "";
        s.target_text =
            (target != item.end() && target->is_string()) ? target->get<std::string>() : "";
        samples.push_back(std::move(s));
    }
    return samples;
}

RowCapPlan plan_row_cap(long long count, int max_rows) {
    RowCapPlan plan;
    if (max_rows <= 0 || count <= max_rows)
        return plan;
    const long long target = std::max<long long>(1, max_rows - max_rows / 4);
    const long long excess = count - target;
    if (2 * excess <= count)
        plan.thin_first = 2 * excess;  // every second of these goes: exactly `excess` rows
    else
        plan.delete_oldest = excess;
    return plan;
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<IMetricsDatabase> MetricsDatabaseFactory::create(const std::string& backend,
                                                                 const std::string& db_path,
                                                                 const std::string& db_url,
                                                                 int pool_size) {
    // Strip "+file" suffix — the dual-write decision is handled by the caller
    std::string core_backend = backend;
    auto plus_pos = core_backend.find("+file");
    if (plus_pos != std::string::npos) {
        core_backend = core_backend.substr(0, plus_pos);
    }

    if (core_backend == "sqlite") {
        std::string path = db_path.empty() ? "training_sessions/metrics.db" : db_path;
        return std::make_unique<SQLiteMetricsDatabase>(path);
    }

#ifdef ADAI_ENABLE_POSTGRES
    if (core_backend == "postgres") {
        // Forward declaration — implemented in PostgresMetricsDatabase.cpp
        extern std::unique_ptr<IMetricsDatabase> create_postgres_metrics_database(
            const std::string& url, int pool_size);
        return create_postgres_metrics_database(db_url, pool_size);
    }
#endif

    if (core_backend == "file" || core_backend.empty()) {
        return nullptr;
    }

    adai::Logger::warn("[MetricsDatabaseFactory] Unknown backend '{}'; falling back to file-only",
                       backend);
    return nullptr;
}
