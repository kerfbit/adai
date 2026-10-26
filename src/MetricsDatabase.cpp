// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-25

// TD-441: the IMetricsDatabase factory and the shared row-cap planner, moved out of
// SQLiteMetricsDatabase.cpp (the factory picks a backend, so it doesn't belong to one).

#include "MetricsDatabase.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include "Logger.hpp"
#include "SQLiteMetricsDatabase.hpp"

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
