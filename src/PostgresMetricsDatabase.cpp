#ifdef ADAI_ENABLE_POSTGRES

#include "PostgresMetricsDatabase.hpp"
#include "TrainingMetricsService.hpp"
#include "IMetricsReporter.hpp"
#include "GenerationQualityMetrics.hpp"
#include "Logger.hpp"

#include <libpq-fe.h>
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

static constexpr int RETRY_COUNT = 3;
static constexpr int RETRY_DELAYS_MS[] = {100, 400, 1600};

PostgresMetricsDatabase::PostgresMetricsDatabase(
    const std::string& connection_url, int pool_size)
    : connection_url_(connection_url),
      pool_size_(pool_size > 0 ? pool_size : 4) {

    pool_.resize(static_cast<size_t>(pool_size_));
    for (auto& entry : pool_) {
        entry.conn = create_connection();
        entry.in_use = false;
    }

    if (!pool_.empty() && pool_[0].conn) {
        bootstrap_schema(pool_[0].conn);
    }

    adai::Logger::info("[PostgresMetricsDB] Connection pool initialized (size={})", pool_size_);
}

PostgresMetricsDatabase::~PostgresMetricsDatabase() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (auto& entry : pool_) {
        if (entry.conn) {
            PQfinish(entry.conn);
            entry.conn = nullptr;
        }
    }
}

PGconn* PostgresMetricsDatabase::create_connection() {
    PGconn* conn = PQconnectdb(connection_url_.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        adai::Logger::error("[PostgresMetricsDB] Connection failed: {}", PQerrorMessage(conn));
        PQfinish(conn);
        return nullptr;
    }
    return conn;
}

bool PostgresMetricsDatabase::ensure_connection(PGconn*& conn) {
    if (conn && PQstatus(conn) == CONNECTION_OK) return true;
    if (conn) {
        PQfinish(conn);
        conn = nullptr;
    }
    conn = create_connection();
    return conn != nullptr;
}

PGconn* PostgresMetricsDatabase::acquire_connection() {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    pool_cv_.wait(lock, [this] {
        for (auto& e : pool_) {
            if (!e.in_use) return true;
        }
        return false;
    });
    for (auto& entry : pool_) {
        if (!entry.in_use) {
            entry.in_use = true;
            ensure_connection(entry.conn);
            return entry.conn;
        }
    }
    return nullptr;
}

void PostgresMetricsDatabase::release_connection(PGconn* conn) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (auto& entry : pool_) {
        if (entry.conn == conn) {
            entry.in_use = false;
            break;
        }
    }
    pool_cv_.notify_one();
}

bool PostgresMetricsDatabase::execute_with_retry(
    const char* context,
    std::function<bool(PGconn*)> operation) {
    for (int attempt = 0; attempt < RETRY_COUNT; ++attempt) {
        PGconn* conn = acquire_connection();
        if (!conn) {
            adai::Logger::error("[PostgresMetricsDB] {} — no connection available (attempt {}/{})",
                                context, attempt + 1, RETRY_COUNT);
            if (attempt < RETRY_COUNT - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAYS_MS[attempt]));
            }
            continue;
        }

        bool success = false;
        try {
            success = operation(conn);
        } catch (const std::exception& e) {
            adai::Logger::error("[PostgresMetricsDB] {} — exception: {} (attempt {}/{})",
                                context, e.what(), attempt + 1, RETRY_COUNT);
        }

        if (!success && PQstatus(conn) != CONNECTION_OK) {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            for (auto& entry : pool_) {
                if (entry.conn == conn) {
                    PQfinish(entry.conn);
                    entry.conn = create_connection();
                    conn = entry.conn;
                    break;
                }
            }
        }

        release_connection(conn);
        if (success) return true;

        if (attempt < RETRY_COUNT - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAYS_MS[attempt]));
        }
    }
    adai::Logger::error("[PostgresMetricsDB] {} — all {} retries exhausted", context, RETRY_COUNT);
    return false;
}

// ============================================================================
// Schema
// ============================================================================

void PostgresMetricsDatabase::bootstrap_schema(PGconn* conn) {
    const char* schema_sql = R"SQL(
        CREATE TABLE IF NOT EXISTS schema_version (
            version    INTEGER NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            key                  TEXT    PRIMARY KEY,
            session_id           INTEGER NOT NULL,
            label                TEXT    NOT NULL DEFAULT '',
            config_json          TEXT,
            is_training          BOOLEAN NOT NULL DEFAULT TRUE,
            created_at           TIMESTAMPTZ NOT NULL,
            ended_at             TIMESTAMPTZ,
            last_update_at       TIMESTAMPTZ NOT NULL,
            total_epochs         INTEGER NOT NULL DEFAULT 0,
            total_samples        INTEGER NOT NULL DEFAULT 0,
            best_validation_loss REAL,
            best_epoch           INTEGER
        );

        CREATE TABLE IF NOT EXISTS metrics_history (
            id                          SERIAL PRIMARY KEY,
            session_key                 TEXT    NOT NULL REFERENCES sessions(key),
            recorded_at                 TIMESTAMPTZ NOT NULL,
            epoch                       INTEGER NOT NULL,
            sample                      INTEGER NOT NULL,
            loss                        REAL,
            validation_loss             REAL,
            learning_rate               REAL,
            gradient_norm               REAL,
            perplexity                  REAL
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_history_session_time
            ON metrics_history (session_key, recorded_at);

        CREATE TABLE IF NOT EXISTS generation_quality (
            id          SERIAL PRIMARY KEY,
            session_key TEXT    NOT NULL REFERENCES sessions(key),
            recorded_at TIMESTAMPTZ NOT NULL,
            epoch       INTEGER NOT NULL,
            bleu1       REAL,
            bleu2       REAL,
            bleu4       REAL,
            rouge1      REAL,
            rouge2      REAL,
            "rougeL"    REAL
        );

        CREATE INDEX IF NOT EXISTS idx_generation_quality_session_epoch
            ON generation_quality (session_key, epoch);

        CREATE TABLE IF NOT EXISTS abnormal_samples (
            id          SERIAL PRIMARY KEY,
            session_key TEXT    NOT NULL REFERENCES sessions(key),
            epoch       INTEGER NOT NULL,
            sample_id   INTEGER NOT NULL,
            loss        REAL,
            grad_norm   REAL,
            reason      TEXT,
            input_text  TEXT,
            target_text TEXT,
            recorded_at TIMESTAMPTZ NOT NULL
        );
    )SQL";

    PGresult* res = PQexec(conn, schema_sql);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::string err = PQerrorMessage(conn);
        PQclear(res);
        throw std::runtime_error("PostgreSQL schema bootstrap failed: " + err);
    }
    PQclear(res);

    // Insert schema_version row 1 if not present
    res = PQexec(conn, "SELECT COUNT(*) FROM schema_version WHERE version = 1");
    int count = 0;
    if (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0) {
        count = std::atoi(PQgetvalue(res, 0, 0));
    }
    PQclear(res);

    if (count == 0) {
        res = PQexec(conn, "INSERT INTO schema_version (version, applied_at) VALUES (1, NOW())");
        PQclear(res);
    }

    adai::Logger::info("[PostgresMetricsDB] Schema bootstrap complete");
}

// ============================================================================
// Timestamp Helpers
// ============================================================================

std::string PostgresMetricsDatabase::format_timestamp(
    const std::chrono::system_clock::time_point& tp) {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
    gmtime_r(&time_t, &tm);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        tp.time_since_epoch()) % 1000;
    oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "+00";
    return oss.str();
}

std::chrono::system_clock::time_point PostgresMetricsDatabase::parse_timestamp(
    const std::string& s) {
    if (s.empty()) return {};

    std::tm tm{};
    int ms = 0;
    if (auto* end = strptime(s.c_str(), "%Y-%m-%dT%H:%M:%S", &tm)) {
        if (*end == '.') {
            ms = std::atoi(end + 1);
        }
    } else if (auto* end2 = strptime(s.c_str(), "%Y-%m-%d %H:%M:%S", &tm)) {
        if (*end2 == '.') {
            ms = std::atoi(end2 + 1);
        }
    }

    auto tp = std::chrono::system_clock::from_time_t(timegm(&tm));
    tp += std::chrono::milliseconds(ms);
    return tp;
}

// ============================================================================
// Session Operations
// ============================================================================

void PostgresMetricsDatabase::upsert_session(const SessionRecord& rec) {
    execute_with_retry("upsert_session", [&](PGconn* conn) -> bool {
        auto ts_created = format_timestamp(rec.created_at);
        auto ts_updated = format_timestamp(rec.last_update_at);
        auto bv_loss = std::to_string(rec.best_validation_loss);
        auto sid = std::to_string(rec.session_id);
        auto is_t = std::string(rec.is_training ? "true" : "false");
        auto epochs = std::to_string(rec.total_epochs);
        auto samples = std::to_string(rec.total_samples);
        auto best_ep = std::to_string(rec.best_epoch);

        const char* params[] = {
            rec.key.c_str(), sid.c_str(), rec.label.c_str(),
            rec.config_json.c_str(), is_t.c_str(), ts_created.c_str(),
            ts_updated.c_str(), epochs.c_str(), samples.c_str(),
            bv_loss.c_str(), best_ep.c_str()
        };

        PGresult* res = PQexecParams(conn,
            "INSERT INTO sessions (key, session_id, label, config_json, is_training, "
            "created_at, last_update_at, total_epochs, total_samples, "
            "best_validation_loss, best_epoch) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) "
            "ON CONFLICT (key) DO UPDATE SET "
            "session_id = EXCLUDED.session_id, label = EXCLUDED.label, "
            "config_json = EXCLUDED.config_json, is_training = EXCLUDED.is_training, "
            "last_update_at = EXCLUDED.last_update_at, total_epochs = EXCLUDED.total_epochs, "
            "total_samples = EXCLUDED.total_samples, "
            "best_validation_loss = EXCLUDED.best_validation_loss, "
            "best_epoch = EXCLUDED.best_epoch",
            11, nullptr, params, nullptr, nullptr, 0);

        bool ok = PQresultStatus(res) == PGRES_COMMAND_OK;
        if (!ok) adai::Logger::error("[PostgresMetricsDB] upsert_session: {}", PQerrorMessage(conn));
        PQclear(res);
        return ok;
    });
}

void PostgresMetricsDatabase::mark_session_ended(const std::string& key) {
    execute_with_retry("mark_session_ended", [&](PGconn* conn) -> bool {
        auto now = format_timestamp(std::chrono::system_clock::now());
        const char* params[] = {key.c_str(), now.c_str()};

        PGresult* res = PQexecParams(conn,
            "UPDATE sessions SET is_training = false, ended_at = $2, last_update_at = $2 "
            "WHERE key = $1",
            2, nullptr, params, nullptr, nullptr, 0);

        bool ok = PQresultStatus(res) == PGRES_COMMAND_OK;
        if (!ok) adai::Logger::error("[PostgresMetricsDB] mark_session_ended: {}", PQerrorMessage(conn));
        PQclear(res);
        return ok;
    });
}

// ============================================================================
// Metrics Record Operations
// ============================================================================

void PostgresMetricsDatabase::insert_metrics_record(
    const std::string& session_key,
    const PersistentMetricsRecord& rec) {
    execute_with_retry("insert_metrics_record", [&](PGconn* conn) -> bool {
        auto ts = format_timestamp(rec.timestamp);
        auto epoch = std::to_string(rec.epoch);
        auto sample = std::to_string(rec.sample);
        auto loss = std::to_string(rec.loss);
        auto vloss = std::to_string(rec.validation_loss);
        auto lr = std::to_string(rec.learning_rate);
        auto gnorm = std::to_string(rec.gradient_norm);
        auto ppl = std::to_string(rec.perplexity);

        const char* params[] = {
            session_key.c_str(), ts.c_str(), epoch.c_str(), sample.c_str(),
            loss.c_str(), vloss.c_str(), lr.c_str(), gnorm.c_str(), ppl.c_str()
        };

        PGresult* res = PQexecParams(conn,
            "INSERT INTO metrics_history (session_key, recorded_at, epoch, sample, "
            "loss, validation_loss, learning_rate, gradient_norm, perplexity) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            9, nullptr, params, nullptr, nullptr, 0);

        bool ok = PQresultStatus(res) == PGRES_COMMAND_OK;
        if (!ok) adai::Logger::error("[PostgresMetricsDB] insert_metrics_record: {}", PQerrorMessage(conn));
        PQclear(res);
        return ok;
    });
}

void PostgresMetricsDatabase::insert_abnormal_sample(
    const std::string& session_key,
    const AbnormalSample& sample) {
    execute_with_retry("insert_abnormal_sample", [&](PGconn* conn) -> bool {
        auto ts = format_timestamp(sample.timestamp);
        auto epoch = std::to_string(sample.epoch);
        auto sid = std::to_string(sample.sample_id);
        auto loss = std::to_string(sample.loss);
        auto gnorm = std::to_string(sample.grad_norm);

        const char* params[] = {
            session_key.c_str(), epoch.c_str(), sid.c_str(),
            loss.c_str(), gnorm.c_str(), sample.reason.c_str(),
            sample.input_text.c_str(), sample.target_text.c_str(), ts.c_str()
        };

        PGresult* res = PQexecParams(conn,
            "INSERT INTO abnormal_samples (session_key, epoch, sample_id, loss, "
            "grad_norm, reason, input_text, target_text, recorded_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            9, nullptr, params, nullptr, nullptr, 0);

        bool ok = PQresultStatus(res) == PGRES_COMMAND_OK;
        if (!ok) adai::Logger::error("[PostgresMetricsDB] insert_abnormal_sample: {}", PQerrorMessage(conn));
        PQclear(res);
        return ok;
    });
}

void PostgresMetricsDatabase::insert_generation_quality(
    const std::string& session_key,
    int epoch,
    const GenerationQualityScore& score) {
    execute_with_retry("insert_generation_quality", [&](PGconn* conn) -> bool {
        auto ts = format_timestamp(std::chrono::system_clock::now());
        auto ep = std::to_string(epoch);
        auto b1 = std::to_string(score.bleu1);
        auto b2 = std::to_string(score.bleu2);
        auto b4 = std::to_string(score.bleu4);
        auto r1 = std::to_string(score.rouge1);
        auto r2 = std::to_string(score.rouge2);
        auto rL = std::to_string(score.rougeL);

        const char* params[] = {
            session_key.c_str(), ts.c_str(), ep.c_str(),
            b1.c_str(), b2.c_str(), b4.c_str(),
            r1.c_str(), r2.c_str(), rL.c_str()
        };

        PGresult* res = PQexecParams(conn,
            "INSERT INTO generation_quality (session_key, recorded_at, epoch, "
            "bleu1, bleu2, bleu4, rouge1, rouge2, \"rougeL\") "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            9, nullptr, params, nullptr, nullptr, 0);

        bool ok = PQresultStatus(res) == PGRES_COMMAND_OK;
        if (!ok) adai::Logger::error("[PostgresMetricsDB] insert_generation_quality: {}", PQerrorMessage(conn));
        PQclear(res);
        return ok;
    });
}

// ============================================================================
// Query Operations
// ============================================================================

std::vector<PersistentMetricsRecord> PostgresMetricsDatabase::query_history(
    const std::string& session_key,
    std::optional<std::chrono::system_clock::time_point> from,
    std::optional<std::chrono::system_clock::time_point> to,
    int limit) {

    std::vector<PersistentMetricsRecord> results;

    execute_with_retry("query_history", [&](PGconn* conn) -> bool {
        results.clear();

        std::string sql = "SELECT recorded_at, epoch, sample, loss, validation_loss, "
                          "learning_rate, gradient_norm, perplexity "
                          "FROM metrics_history WHERE session_key = $1";

        std::vector<std::string> param_strs;
        param_strs.push_back(session_key);

        int param_idx = 2;
        if (from) {
            auto from_str = format_timestamp(*from);
            sql += " AND recorded_at >= $" + std::to_string(param_idx++);
            param_strs.push_back(from_str);
        }
        if (to) {
            auto to_str = format_timestamp(*to);
            sql += " AND recorded_at <= $" + std::to_string(param_idx++);
            param_strs.push_back(to_str);
        }

        sql += " ORDER BY recorded_at ASC";
        if (limit > 0) {
            sql += " LIMIT " + std::to_string(limit);
        }

        std::vector<const char*> params;
        for (const auto& s : param_strs) params.push_back(s.c_str());

        PGresult* res = PQexecParams(conn, sql.c_str(),
            static_cast<int>(params.size()), nullptr,
            params.data(), nullptr, nullptr, 0);

        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            adai::Logger::error("[PostgresMetricsDB] query_history: {}", PQerrorMessage(conn));
            PQclear(res);
            return false;
        }

        int nrows = PQntuples(res);
        results.reserve(static_cast<size_t>(nrows));

        for (int i = 0; i < nrows; ++i) {
            PersistentMetricsRecord rec;
            rec.timestamp       = parse_timestamp(PQgetvalue(res, i, 0));
            rec.epoch           = std::atoi(PQgetvalue(res, i, 1));
            rec.sample          = std::atoi(PQgetvalue(res, i, 2));
            rec.loss            = std::stof(PQgetvalue(res, i, 3));
            rec.validation_loss = std::stof(PQgetvalue(res, i, 4));
            rec.learning_rate   = std::stof(PQgetvalue(res, i, 5));
            rec.gradient_norm   = std::stof(PQgetvalue(res, i, 6));
            rec.perplexity      = std::stof(PQgetvalue(res, i, 7));
            results.push_back(rec);
        }

        PQclear(res);
        return true;
    });

    return results;
}

std::vector<SessionRecord> PostgresMetricsDatabase::list_sessions(
    std::optional<bool> is_training_filter) {

    std::vector<SessionRecord> results;

    execute_with_retry("list_sessions", [&](PGconn* conn) -> bool {
        results.clear();

        std::string sql = "SELECT key, session_id, label, config_json, is_training, "
                          "created_at, ended_at, last_update_at, total_epochs, "
                          "total_samples, best_validation_loss, best_epoch "
                          "FROM sessions";

        std::vector<const char*> params;
        std::string filter_val;

        if (is_training_filter) {
            filter_val = *is_training_filter ? "true" : "false";
            sql += " WHERE is_training = $1";
            params.push_back(filter_val.c_str());
        }

        sql += " ORDER BY created_at DESC";

        PGresult* res = PQexecParams(conn, sql.c_str(),
            static_cast<int>(params.size()), nullptr,
            params.empty() ? nullptr : params.data(),
            nullptr, nullptr, 0);

        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            adai::Logger::error("[PostgresMetricsDB] list_sessions: {}", PQerrorMessage(conn));
            PQclear(res);
            return false;
        }

        int nrows = PQntuples(res);
        results.reserve(static_cast<size_t>(nrows));

        for (int i = 0; i < nrows; ++i) {
            SessionRecord rec;
            rec.key          = PQgetvalue(res, i, 0);
            rec.session_id   = std::atoi(PQgetvalue(res, i, 1));
            rec.label        = PQgetvalue(res, i, 2);
            rec.config_json  = PQgetvalue(res, i, 3);
            rec.is_training  = std::string(PQgetvalue(res, i, 4)) == "t";
            rec.created_at   = parse_timestamp(PQgetvalue(res, i, 5));
            auto ended_str   = std::string(PQgetvalue(res, i, 6));
            if (!ended_str.empty()) {
                rec.ended_at = parse_timestamp(ended_str);
            }
            rec.last_update_at       = parse_timestamp(PQgetvalue(res, i, 7));
            rec.total_epochs         = std::atoi(PQgetvalue(res, i, 8));
            rec.total_samples        = std::atoi(PQgetvalue(res, i, 9));
            rec.best_validation_loss = std::stof(PQgetvalue(res, i, 10));
            rec.best_epoch           = std::atoi(PQgetvalue(res, i, 11));
            results.push_back(std::move(rec));
        }

        PQclear(res);
        return true;
    });

    return results;
}

std::optional<SessionRecord> PostgresMetricsDatabase::get_session(const std::string& key) {
    std::optional<SessionRecord> result;

    execute_with_retry("get_session", [&](PGconn* conn) -> bool {
        result.reset();

        const char* params[] = {key.c_str()};
        PGresult* res = PQexecParams(conn,
            "SELECT key, session_id, label, config_json, is_training, "
            "created_at, ended_at, last_update_at, total_epochs, "
            "total_samples, best_validation_loss, best_epoch "
            "FROM sessions WHERE key = $1",
            1, nullptr, params, nullptr, nullptr, 0);

        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            adai::Logger::error("[PostgresMetricsDB] get_session: {}", PQerrorMessage(conn));
            PQclear(res);
            return false;
        }

        if (PQntuples(res) == 0) {
            PQclear(res);
            return true;
        }

        SessionRecord rec;
        rec.key          = PQgetvalue(res, 0, 0);
        rec.session_id   = std::atoi(PQgetvalue(res, 0, 1));
        rec.label        = PQgetvalue(res, 0, 2);
        rec.config_json  = PQgetvalue(res, 0, 3);
        rec.is_training  = std::string(PQgetvalue(res, 0, 4)) == "t";
        rec.created_at   = parse_timestamp(PQgetvalue(res, 0, 5));
        auto ended_str   = std::string(PQgetvalue(res, 0, 6));
        if (!ended_str.empty()) {
            rec.ended_at = parse_timestamp(ended_str);
        }
        rec.last_update_at       = parse_timestamp(PQgetvalue(res, 0, 7));
        rec.total_epochs         = std::atoi(PQgetvalue(res, 0, 8));
        rec.total_samples        = std::atoi(PQgetvalue(res, 0, 9));
        rec.best_validation_loss = std::stof(PQgetvalue(res, 0, 10));
        rec.best_epoch           = std::atoi(PQgetvalue(res, 0, 11));

        result = std::move(rec);
        PQclear(res);
        return true;
    });

    return result;
}

// ============================================================================
// Factory Helper
// ============================================================================

std::unique_ptr<IMetricsDatabase> create_postgres_metrics_database(
    const std::string& url, int pool_size) {
    return std::make_unique<PostgresMetricsDatabase>(url, pool_size);
}

#endif // ADAI_ENABLE_POSTGRES
