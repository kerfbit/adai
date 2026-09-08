// @adai-status: beta        (no dedicated unit test file (covered via MetricsDatabaseTest integration paths))
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-07

#include "SQLiteMetricsDatabase.hpp"
#include "GenerationQualityMetrics.hpp"
#include "IMetricsReporter.hpp"
#include "Logger.hpp"
#include "TrainingMetricsService.hpp"

#include <sqlite3.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

static void check_sqlite(int rc, sqlite3* db, const char* context) {
    if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW) {
        std::string msg = std::string(context) + ": " + sqlite3_errmsg(db);
        adai::Logger::error("[SQLiteMetricsDB] {}", msg);
        throw std::runtime_error(msg);
    }
}

SQLiteMetricsDatabase::SQLiteMetricsDatabase(const std::string& db_path) {
    int rc = sqlite3_open(db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::string msg = "Failed to open SQLite database: " + db_path;
        if (db_) {
            msg += " — ";
            msg += sqlite3_errmsg(db_);
            sqlite3_close(db_);
            db_ = nullptr;
        }
        throw std::runtime_error(msg);
    }

    char* err = nullptr;
    sqlite3_exec(db_, "PRAGMA journal_mode = WAL;", nullptr, nullptr, &err);
    if (err) {
        sqlite3_free(err);
    }

    sqlite3_exec(db_, "PRAGMA synchronous = NORMAL;", nullptr, nullptr, &err);
    if (err) {
        sqlite3_free(err);
    }

    bootstrap_schema();
    prepare_statements();

    adai::Logger::info("[SQLiteMetricsDB] Opened database: {}", db_path);
}

SQLiteMetricsDatabase::~SQLiteMetricsDatabase() {
    finalize_statements();
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

// ============================================================================
// Schema
// ============================================================================

void SQLiteMetricsDatabase::bootstrap_schema() {
    const char* schema_sql = R"SQL(
        CREATE TABLE IF NOT EXISTS schema_version (
            version    INTEGER NOT NULL,
            applied_at TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            key                  TEXT    PRIMARY KEY,
            session_id           INTEGER NOT NULL,
            label                TEXT    NOT NULL DEFAULT '',
            config_json          TEXT,
            is_training          INTEGER NOT NULL DEFAULT 1,
            created_at           TEXT    NOT NULL,
            ended_at             TEXT,
            last_update_at       TEXT    NOT NULL,
            total_epochs         INTEGER NOT NULL DEFAULT 0,
            total_samples        INTEGER NOT NULL DEFAULT 0,
            best_validation_loss REAL,
            best_epoch           INTEGER,
            final_loss            REAL,
            final_validation_loss REAL
        );

        CREATE TABLE IF NOT EXISTS metrics_history (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key                 TEXT    NOT NULL REFERENCES sessions(key),
            recorded_at                 TEXT    NOT NULL,
            epoch                       INTEGER NOT NULL,
            sample                      INTEGER NOT NULL,
            loss                        REAL,
            validation_loss             REAL,
            learning_rate               REAL,
            gradient_norm               REAL,
            perplexity                  REAL,
            compute_time_ratio          REAL,
            weight_update_ratio         REAL,
            activation_saturation_ratio REAL,
            attention_entropy           REAL,
            padding_efficiency          REAL,
            layer_gradient_norms_json   TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_history_session_time
            ON metrics_history (session_key, recorded_at);

        -- TD-013 extension: dedicated, unthrottled per-optimizer-step
        -- gradient_variance history (see IMetricsDatabase::insert_gradient_variance_sample
        -- doc comment — deliberately not folded into metrics_history, which is
        -- only written once every persist_every_samples steps).
        CREATE TABLE IF NOT EXISTS gradient_variance_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT    NOT NULL REFERENCES sessions(key),
            recorded_at TEXT    NOT NULL,
            step        INTEGER NOT NULL,
            epoch       INTEGER NOT NULL,
            value       REAL
        );

        CREATE INDEX IF NOT EXISTS idx_gradient_variance_history_session_step
            ON gradient_variance_history (session_key, step);

        CREATE TABLE IF NOT EXISTS generation_quality (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT    NOT NULL REFERENCES sessions(key),
            recorded_at TEXT    NOT NULL,
            epoch       INTEGER NOT NULL,
            bleu1       REAL,
            bleu2       REAL,
            bleu4       REAL,
            rouge1      REAL,
            rouge2      REAL,
            rougeL      REAL
        );

        CREATE INDEX IF NOT EXISTS idx_generation_quality_session_epoch
            ON generation_quality (session_key, epoch);

        CREATE TABLE IF NOT EXISTS abnormal_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT    NOT NULL REFERENCES sessions(key),
            epoch       INTEGER NOT NULL,
            sample_id   INTEGER NOT NULL,
            loss        REAL,
            grad_norm   REAL,
            reason      TEXT,
            input_text  TEXT,
            target_text TEXT,
            recorded_at TEXT    NOT NULL
        );
    )SQL";

    char* err = nullptr;
    int rc = sqlite3_exec(db_, schema_sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::string msg = std::string("Schema bootstrap failed: ") + (err ? err : "unknown");
        if (err)
            sqlite3_free(err);
        throw std::runtime_error(msg);
    }

    // Insert schema_version row 1 if not present
    const char* version_check = "SELECT COUNT(*) FROM schema_version WHERE version = 1;";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, version_check, -1, &stmt, nullptr);
    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);

    if (count == 0) {
        auto now = format_timestamp(std::chrono::system_clock::now());
        std::string insert =
            "INSERT INTO schema_version (version, applied_at) VALUES (1, '" + now + "');";
        sqlite3_exec(db_, insert.c_str(), nullptr, nullptr, nullptr);
    }

    // Migration: final_loss/final_validation_loss were added after the initial schema, so
    // a database created before this change won't have them yet — CREATE TABLE IF NOT EXISTS
    // above is a no-op against an existing table. Add them if missing (checked via
    // PRAGMA table_info rather than just trying ALTER TABLE and swallowing the "duplicate
    // column" error, so a genuinely unexpected failure here still surfaces).
    bool has_final_loss = false;
    sqlite3_stmt* pragma_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "PRAGMA table_info(sessions);", -1, &pragma_stmt, nullptr) ==
        SQLITE_OK) {
        while (sqlite3_step(pragma_stmt) == SQLITE_ROW) {
            const auto* col_name =
                reinterpret_cast<const char*>(sqlite3_column_text(pragma_stmt, 1));
            if (col_name && std::string(col_name) == "final_loss") {
                has_final_loss = true;
                break;
            }
        }
    }
    sqlite3_finalize(pragma_stmt);

    if (!has_final_loss) {
        adai::Logger::info(
            "[SQLiteMetricsDB] Migrating sessions table: adding final_loss / "
            "final_validation_loss columns");
        sqlite3_exec(db_, "ALTER TABLE sessions ADD COLUMN final_loss REAL;", nullptr, nullptr,
                     nullptr);
        sqlite3_exec(db_, "ALTER TABLE sessions ADD COLUMN final_validation_loss REAL;", nullptr,
                     nullptr, nullptr);
    }

    // Migration: TD-013 extension columns on metrics_history, added after the
    // initial schema — same PRAGMA table_info check pattern as above.
    bool has_compute_time_ratio = false;
    sqlite3_stmt* mh_pragma_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "PRAGMA table_info(metrics_history);", -1, &mh_pragma_stmt,
                           nullptr) == SQLITE_OK) {
        while (sqlite3_step(mh_pragma_stmt) == SQLITE_ROW) {
            const auto* col_name =
                reinterpret_cast<const char*>(sqlite3_column_text(mh_pragma_stmt, 1));
            if (col_name && std::string(col_name) == "compute_time_ratio") {
                has_compute_time_ratio = true;
                break;
            }
        }
    }
    sqlite3_finalize(mh_pragma_stmt);

    if (!has_compute_time_ratio) {
        adai::Logger::info(
            "[SQLiteMetricsDB] Migrating metrics_history table: adding TD-013 diagnostic columns");
        sqlite3_exec(db_, "ALTER TABLE metrics_history ADD COLUMN compute_time_ratio REAL;",
                     nullptr, nullptr, nullptr);
        sqlite3_exec(db_, "ALTER TABLE metrics_history ADD COLUMN weight_update_ratio REAL;",
                     nullptr, nullptr, nullptr);
        sqlite3_exec(db_,
                     "ALTER TABLE metrics_history ADD COLUMN activation_saturation_ratio REAL;",
                     nullptr, nullptr, nullptr);
        sqlite3_exec(db_, "ALTER TABLE metrics_history ADD COLUMN attention_entropy REAL;",
                     nullptr, nullptr, nullptr);
        sqlite3_exec(db_, "ALTER TABLE metrics_history ADD COLUMN padding_efficiency REAL;",
                     nullptr, nullptr, nullptr);
        sqlite3_exec(db_,
                     "ALTER TABLE metrics_history ADD COLUMN layer_gradient_norms_json TEXT;",
                     nullptr, nullptr, nullptr);
    }
}

// ============================================================================
// Prepared Statements
// ============================================================================

void SQLiteMetricsDatabase::prepare_statements() {
    auto prep = [this](const char* sql, sqlite3_stmt** stmt) {
        check_sqlite(sqlite3_prepare_v2(db_, sql, -1, stmt, nullptr), db_, sql);
    };

    prep(R"SQL(
        INSERT INTO sessions (key, session_id, label, config_json, is_training,
                              created_at, last_update_at, total_epochs, total_samples,
                              best_validation_loss, best_epoch, final_loss, final_validation_loss)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
        ON CONFLICT(key) DO UPDATE SET
            session_id           = excluded.session_id,
            label                = excluded.label,
            config_json          = excluded.config_json,
            is_training          = excluded.is_training,
            last_update_at       = excluded.last_update_at,
            total_epochs         = excluded.total_epochs,
            total_samples        = excluded.total_samples,
            best_validation_loss = excluded.best_validation_loss,
            best_epoch           = excluded.best_epoch,
            final_loss            = excluded.final_loss,
            final_validation_loss = excluded.final_validation_loss;
    )SQL",
         &stmt_upsert_session_);

    prep(R"SQL(
        UPDATE sessions SET is_training = 0, ended_at = ?2, last_update_at = ?2
        WHERE key = ?1;
    )SQL",
         &stmt_mark_ended_);

    prep(R"SQL(
        INSERT INTO metrics_history (session_key, recorded_at, epoch, sample,
                                     loss, validation_loss, learning_rate,
                                     gradient_norm, perplexity, compute_time_ratio,
                                     weight_update_ratio, activation_saturation_ratio,
                                     attention_entropy, padding_efficiency,
                                     layer_gradient_norms_json)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15);
    )SQL",
         &stmt_insert_metrics_);

    prep(R"SQL(
        INSERT INTO gradient_variance_history (session_key, recorded_at, step, epoch, value)
        VALUES (?1, ?2, ?3, ?4, ?5);
    )SQL",
         &stmt_insert_gradient_variance_);

    prep(R"SQL(
        INSERT INTO abnormal_samples (session_key, epoch, sample_id, loss,
                                      grad_norm, reason, input_text, target_text,
                                      recorded_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);
    )SQL",
         &stmt_insert_abnormal_);

    prep(R"SQL(
        INSERT INTO generation_quality (session_key, recorded_at, epoch,
                                        bleu1, bleu2, bleu4, rouge1, rouge2, rougeL)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);
    )SQL",
         &stmt_insert_gen_quality_);

    prep(R"SQL(
        SELECT key, session_id, label, config_json, is_training,
               created_at, ended_at, last_update_at, total_epochs,
               total_samples, best_validation_loss, best_epoch,
               final_loss, final_validation_loss
        FROM sessions WHERE key = ?1;
    )SQL",
         &stmt_get_session_);
}

void SQLiteMetricsDatabase::finalize_statements() {
    auto fin = [](sqlite3_stmt*& s) {
        if (s) {
            sqlite3_finalize(s);
            s = nullptr;
        }
    };
    fin(stmt_upsert_session_);
    fin(stmt_mark_ended_);
    fin(stmt_insert_metrics_);
    fin(stmt_insert_gradient_variance_);
    fin(stmt_insert_abnormal_);
    fin(stmt_insert_gen_quality_);
    fin(stmt_get_session_);
}

// ============================================================================
// Timestamp Helpers
// ============================================================================

std::string SQLiteMetricsDatabase::format_timestamp(
    const std::chrono::system_clock::time_point& tp) {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
    gmtime_r(&time_t, &tm);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()) % 1000;
    oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    return oss.str();
}

std::chrono::system_clock::time_point SQLiteMetricsDatabase::parse_timestamp(const std::string& s) {
    if (s.empty())
        return {};

    std::tm tm{};
    int ms = 0;

    // Parse "YYYY-MM-DDTHH:MM:SS.mmmZ" or "YYYY-MM-DDTHH:MM:SS"
    if (auto* end = strptime(s.c_str(), "%Y-%m-%dT%H:%M:%S", &tm)) {
        if (*end == '.') {
            ms = std::atoi(end + 1);
        }
    }

    auto tp = std::chrono::system_clock::from_time_t(timegm(&tm));
    tp += std::chrono::milliseconds(ms);
    return tp;
}

// ============================================================================
// Session Operations
// ============================================================================

void SQLiteMetricsDatabase::upsert_session(const SessionRecord& rec) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto ts_created = format_timestamp(rec.created_at);
    auto ts_updated = format_timestamp(rec.last_update_at);

    sqlite3_reset(stmt_upsert_session_);
    sqlite3_bind_text(stmt_upsert_session_, 1, rec.key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt_upsert_session_, 2, rec.session_id);
    sqlite3_bind_text(stmt_upsert_session_, 3, rec.label.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_upsert_session_, 4, rec.config_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt_upsert_session_, 5, rec.is_training ? 1 : 0);
    sqlite3_bind_text(stmt_upsert_session_, 6, ts_created.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_upsert_session_, 7, ts_updated.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt_upsert_session_, 8, rec.total_epochs);
    sqlite3_bind_int(stmt_upsert_session_, 9, rec.total_samples);
    sqlite3_bind_double(stmt_upsert_session_, 10, static_cast<double>(rec.best_validation_loss));
    sqlite3_bind_int(stmt_upsert_session_, 11, rec.best_epoch);
    sqlite3_bind_double(stmt_upsert_session_, 12, static_cast<double>(rec.final_loss));
    sqlite3_bind_double(stmt_upsert_session_, 13, static_cast<double>(rec.final_validation_loss));

    check_sqlite(sqlite3_step(stmt_upsert_session_), db_, "upsert_session");
}

void SQLiteMetricsDatabase::archive_session(const std::string& key,
                                            const std::string& archived_key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto run = [this](const char* sql, const std::vector<std::string>& params) {
        sqlite3_stmt* stmt = nullptr;
        check_sqlite(sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr), db_, sql);
        for (size_t i = 0; i < params.size(); ++i) {
            sqlite3_bind_text(stmt, static_cast<int>(i + 1), params[i].c_str(), -1,
                              SQLITE_TRANSIENT);
        }
        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_DONE) {
            throw std::runtime_error(std::string(sql) + ": " + sqlite3_errmsg(db_));
        }
    };

    check_sqlite(sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr), db_,
                 "archive_session:begin");
    try {
        // Copy the session row under the archived key (rather than UPDATE the key in place)
        // so that child rows can be repointed to a key that already exists, avoiding any
        // foreign-key ordering issue if constraint enforcement is ever turned on.
        run("INSERT INTO sessions (key, session_id, label, config_json, is_training, created_at, "
            "ended_at, last_update_at, total_epochs, total_samples, best_validation_loss, "
            "best_epoch, "
            "final_loss, final_validation_loss) "
            "SELECT ?1, session_id, label, config_json, 0, created_at, "
            "COALESCE(ended_at, last_update_at), last_update_at, total_epochs, total_samples, "
            "best_validation_loss, best_epoch, final_loss, final_validation_loss "
            "FROM sessions WHERE key = ?2;",
            {archived_key, key});
        run("UPDATE metrics_history SET session_key = ?1 WHERE session_key = ?2;",
            {archived_key, key});
        run("UPDATE gradient_variance_history SET session_key = ?1 WHERE session_key = ?2;",
            {archived_key, key});
        run("UPDATE generation_quality SET session_key = ?1 WHERE session_key = ?2;",
            {archived_key, key});
        run("UPDATE abnormal_samples SET session_key = ?1 WHERE session_key = ?2;",
            {archived_key, key});
        run("DELETE FROM sessions WHERE key = ?1;", {key});
    } catch (...) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw;
    }
    check_sqlite(sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr), db_,
                 "archive_session:commit");
}

void SQLiteMetricsDatabase::mark_session_ended(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = format_timestamp(std::chrono::system_clock::now());

    sqlite3_reset(stmt_mark_ended_);
    sqlite3_bind_text(stmt_mark_ended_, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_mark_ended_, 2, now.c_str(), -1, SQLITE_TRANSIENT);

    check_sqlite(sqlite3_step(stmt_mark_ended_), db_, "mark_session_ended");
}

// ============================================================================
// Metrics Record Operations
// ============================================================================

void SQLiteMetricsDatabase::insert_metrics_record(const std::string& session_key,
                                                  const PersistentMetricsRecord& rec) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto ts = format_timestamp(rec.timestamp);

    sqlite3_reset(stmt_insert_metrics_);
    sqlite3_bind_text(stmt_insert_metrics_, 1, session_key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_insert_metrics_, 2, ts.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt_insert_metrics_, 3, rec.epoch);
    sqlite3_bind_int(stmt_insert_metrics_, 4, rec.sample);
    sqlite3_bind_double(stmt_insert_metrics_, 5, static_cast<double>(rec.loss));
    sqlite3_bind_double(stmt_insert_metrics_, 6, static_cast<double>(rec.validation_loss));
    sqlite3_bind_double(stmt_insert_metrics_, 7, static_cast<double>(rec.learning_rate));
    sqlite3_bind_double(stmt_insert_metrics_, 8, static_cast<double>(rec.gradient_norm));
    sqlite3_bind_double(stmt_insert_metrics_, 9, static_cast<double>(rec.perplexity));
    sqlite3_bind_double(stmt_insert_metrics_, 10, static_cast<double>(rec.compute_time_ratio));
    sqlite3_bind_double(stmt_insert_metrics_, 11, static_cast<double>(rec.weight_update_ratio));
    sqlite3_bind_double(stmt_insert_metrics_, 12,
                        static_cast<double>(rec.activation_saturation_ratio));
    sqlite3_bind_double(stmt_insert_metrics_, 13, static_cast<double>(rec.attention_entropy));
    sqlite3_bind_double(stmt_insert_metrics_, 14, static_cast<double>(rec.padding_efficiency));
    if (rec.layer_gradient_norms_json.empty()) {
        sqlite3_bind_null(stmt_insert_metrics_, 15);
    } else {
        sqlite3_bind_text(stmt_insert_metrics_, 15, rec.layer_gradient_norms_json.c_str(), -1,
                          SQLITE_TRANSIENT);
    }

    check_sqlite(sqlite3_step(stmt_insert_metrics_), db_, "insert_metrics_record");
}

void SQLiteMetricsDatabase::insert_gradient_variance_sample(const std::string& session_key,
                                                             int step, int epoch, float value) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto ts = format_timestamp(std::chrono::system_clock::now());

    sqlite3_reset(stmt_insert_gradient_variance_);
    sqlite3_bind_text(stmt_insert_gradient_variance_, 1, session_key.c_str(), -1,
                      SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_insert_gradient_variance_, 2, ts.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt_insert_gradient_variance_, 3, step);
    sqlite3_bind_int(stmt_insert_gradient_variance_, 4, epoch);
    sqlite3_bind_double(stmt_insert_gradient_variance_, 5, static_cast<double>(value));

    check_sqlite(sqlite3_step(stmt_insert_gradient_variance_), db_,
                "insert_gradient_variance_sample");
}

void SQLiteMetricsDatabase::insert_abnormal_sample(const std::string& session_key,
                                                   const AbnormalSample& sample) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto ts = format_timestamp(sample.timestamp);

    sqlite3_reset(stmt_insert_abnormal_);
    sqlite3_bind_text(stmt_insert_abnormal_, 1, session_key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt_insert_abnormal_, 2, sample.epoch);
    sqlite3_bind_int(stmt_insert_abnormal_, 3, sample.sample_id);
    sqlite3_bind_double(stmt_insert_abnormal_, 4, static_cast<double>(sample.loss));
    sqlite3_bind_double(stmt_insert_abnormal_, 5, static_cast<double>(sample.grad_norm));
    sqlite3_bind_text(stmt_insert_abnormal_, 6, sample.reason.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_insert_abnormal_, 7, sample.input_text.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_insert_abnormal_, 8, sample.target_text.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_insert_abnormal_, 9, ts.c_str(), -1, SQLITE_TRANSIENT);

    check_sqlite(sqlite3_step(stmt_insert_abnormal_), db_, "insert_abnormal_sample");
}

void SQLiteMetricsDatabase::insert_generation_quality(const std::string& session_key, int epoch,
                                                      const GenerationQualityScore& score) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto ts = format_timestamp(std::chrono::system_clock::now());

    sqlite3_reset(stmt_insert_gen_quality_);
    sqlite3_bind_text(stmt_insert_gen_quality_, 1, session_key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_insert_gen_quality_, 2, ts.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt_insert_gen_quality_, 3, epoch);
    sqlite3_bind_double(stmt_insert_gen_quality_, 4, static_cast<double>(score.bleu1));
    sqlite3_bind_double(stmt_insert_gen_quality_, 5, static_cast<double>(score.bleu2));
    sqlite3_bind_double(stmt_insert_gen_quality_, 6, static_cast<double>(score.bleu4));
    sqlite3_bind_double(stmt_insert_gen_quality_, 7, static_cast<double>(score.rouge1));
    sqlite3_bind_double(stmt_insert_gen_quality_, 8, static_cast<double>(score.rouge2));
    sqlite3_bind_double(stmt_insert_gen_quality_, 9, static_cast<double>(score.rougeL));

    check_sqlite(sqlite3_step(stmt_insert_gen_quality_), db_, "insert_generation_quality");
}

// ============================================================================
// Query Operations
// ============================================================================

std::vector<PersistentMetricsRecord> SQLiteMetricsDatabase::query_history(
    const std::string& session_key, std::optional<std::chrono::system_clock::time_point> from,
    std::optional<std::chrono::system_clock::time_point> to, int limit) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string sql =
        "SELECT recorded_at, epoch, sample, loss, validation_loss, "
        "learning_rate, gradient_norm, perplexity, compute_time_ratio, "
        "weight_update_ratio, activation_saturation_ratio, attention_entropy, "
        "padding_efficiency, layer_gradient_norms_json "
        "FROM metrics_history WHERE session_key = ?";

    int param_idx = 2;
    std::string from_str, to_str;

    if (from) {
        from_str = format_timestamp(*from);
        sql += " AND recorded_at >= ?" + std::to_string(param_idx++);
    }
    if (to) {
        to_str = format_timestamp(*to);
        sql += " AND recorded_at <= ?" + std::to_string(param_idx++);
    }

    sql += " ORDER BY recorded_at ASC";

    if (limit > 0) {
        sql += " LIMIT " + std::to_string(limit);
    }

    sqlite3_stmt* stmt = nullptr;
    check_sqlite(sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr), db_,
                 "query_history prepare");

    int bind_idx = 1;
    sqlite3_bind_text(stmt, bind_idx++, session_key.c_str(), -1, SQLITE_TRANSIENT);
    if (from)
        sqlite3_bind_text(stmt, bind_idx++, from_str.c_str(), -1, SQLITE_TRANSIENT);
    if (to)
        sqlite3_bind_text(stmt, bind_idx++, to_str.c_str(), -1, SQLITE_TRANSIENT);

    std::vector<PersistentMetricsRecord> results;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        PersistentMetricsRecord rec;
        auto ts_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        rec.timestamp = parse_timestamp(ts_text ? ts_text : "");
        rec.epoch = sqlite3_column_int(stmt, 1);
        rec.sample = sqlite3_column_int(stmt, 2);
        rec.loss = static_cast<float>(sqlite3_column_double(stmt, 3));
        rec.validation_loss = static_cast<float>(sqlite3_column_double(stmt, 4));
        rec.learning_rate = static_cast<float>(sqlite3_column_double(stmt, 5));
        rec.gradient_norm = static_cast<float>(sqlite3_column_double(stmt, 6));
        rec.perplexity = static_cast<float>(sqlite3_column_double(stmt, 7));
        rec.compute_time_ratio = static_cast<float>(sqlite3_column_double(stmt, 8));
        rec.weight_update_ratio = static_cast<float>(sqlite3_column_double(stmt, 9));
        rec.activation_saturation_ratio = static_cast<float>(sqlite3_column_double(stmt, 10));
        rec.attention_entropy = static_cast<float>(sqlite3_column_double(stmt, 11));
        rec.padding_efficiency = static_cast<float>(sqlite3_column_double(stmt, 12));
        if (const auto* lg_text =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 13))) {
            rec.layer_gradient_norms_json = lg_text;
        }
        results.push_back(rec);
    }

    sqlite3_finalize(stmt);
    return results;
}

std::vector<SessionRecord> SQLiteMetricsDatabase::list_sessions(
    std::optional<bool> is_training_filter) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string sql =
        "SELECT key, session_id, label, config_json, is_training, "
        "created_at, ended_at, last_update_at, total_epochs, "
        "total_samples, best_validation_loss, best_epoch, "
        "final_loss, final_validation_loss "
        "FROM sessions";

    if (is_training_filter) {
        sql += " WHERE is_training = " + std::to_string(*is_training_filter ? 1 : 0);
    }

    sql += " ORDER BY created_at DESC";

    sqlite3_stmt* stmt = nullptr;
    check_sqlite(sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr), db_,
                 "list_sessions prepare");

    std::vector<SessionRecord> results;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        SessionRecord rec;
        auto col_text = [&](int col) -> std::string {
            auto* t = reinterpret_cast<const char*>(sqlite3_column_text(stmt, col));
            return t ? t : "";
        };
        rec.key = col_text(0);
        rec.session_id = sqlite3_column_int(stmt, 1);
        rec.label = col_text(2);
        rec.config_json = col_text(3);
        rec.is_training = sqlite3_column_int(stmt, 4) != 0;
        rec.created_at = parse_timestamp(col_text(5));
        auto ended_str = col_text(6);
        if (!ended_str.empty()) {
            rec.ended_at = parse_timestamp(ended_str);
        }
        rec.last_update_at = parse_timestamp(col_text(7));
        rec.total_epochs = sqlite3_column_int(stmt, 8);
        rec.total_samples = sqlite3_column_int(stmt, 9);
        rec.best_validation_loss = static_cast<float>(sqlite3_column_double(stmt, 10));
        rec.best_epoch = sqlite3_column_int(stmt, 11);
        rec.final_loss = static_cast<float>(sqlite3_column_double(stmt, 12));
        rec.final_validation_loss = static_cast<float>(sqlite3_column_double(stmt, 13));
        results.push_back(std::move(rec));
    }

    sqlite3_finalize(stmt);
    return results;
}

std::optional<SessionRecord> SQLiteMetricsDatabase::get_session(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    sqlite3_reset(stmt_get_session_);
    sqlite3_bind_text(stmt_get_session_, 1, key.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt_get_session_) != SQLITE_ROW) {
        return std::nullopt;
    }

    auto col_text = [&](int col) -> std::string {
        auto* t = reinterpret_cast<const char*>(sqlite3_column_text(stmt_get_session_, col));
        return t ? t : "";
    };

    SessionRecord rec;
    rec.key = col_text(0);
    rec.session_id = sqlite3_column_int(stmt_get_session_, 1);
    rec.label = col_text(2);
    rec.config_json = col_text(3);
    rec.is_training = sqlite3_column_int(stmt_get_session_, 4) != 0;
    rec.created_at = parse_timestamp(col_text(5));
    auto ended_str = col_text(6);
    if (!ended_str.empty()) {
        rec.ended_at = parse_timestamp(ended_str);
    }
    rec.last_update_at = parse_timestamp(col_text(7));
    rec.total_epochs = sqlite3_column_int(stmt_get_session_, 8);
    rec.total_samples = sqlite3_column_int(stmt_get_session_, 9);
    rec.best_validation_loss = static_cast<float>(sqlite3_column_double(stmt_get_session_, 10));
    rec.best_epoch = sqlite3_column_int(stmt_get_session_, 11);
    rec.final_loss = static_cast<float>(sqlite3_column_double(stmt_get_session_, 12));
    rec.final_validation_loss = static_cast<float>(sqlite3_column_double(stmt_get_session_, 13));

    return rec;
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
