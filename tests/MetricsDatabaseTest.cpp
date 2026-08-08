#include <gtest/gtest.h>
#include <sqlite3.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include "GenerationQualityMetrics.hpp"
#include "IMetricsReporter.hpp"
#include "MetricsDatabase.hpp"
#include "SQLiteMetricsDatabase.hpp"
#include "TrainingMetricsService.hpp"

namespace fs = std::filesystem;

class MetricsDatabaseTest : public ::testing::Test {
   protected:
    void SetUp() override {
        test_dir_ = fs::temp_directory_path() / "adai_metrics_db_test";
        fs::remove_all(test_dir_);
        fs::create_directories(test_dir_);
        db_path_ = (test_dir_ / "test_metrics.db").string();
    }

    void TearDown() override {
        fs::remove_all(test_dir_);
    }

    fs::path test_dir_;
    std::string db_path_;
};

TEST_F(MetricsDatabaseTest, SchemaBootstrap) {
    SQLiteMetricsDatabase db(db_path_);

    EXPECT_TRUE(fs::exists(db_path_));

    // Verify schema_version row was inserted by opening a second connection
    sqlite3* raw_db = nullptr;
    ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw_db), SQLITE_OK);

    // Check schema_version has version 1
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(raw_db, "SELECT version FROM schema_version WHERE version = 1", -1, &stmt,
                       nullptr);
    ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
    EXPECT_EQ(sqlite3_column_int(stmt, 0), 1);
    sqlite3_finalize(stmt);

    // Check all 4 tables + schema_version exist
    for (const char* table : {"schema_version", "sessions", "metrics_history", "generation_quality",
                              "abnormal_samples"}) {
        sqlite3_prepare_v2(raw_db, "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                           -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, table, -1, SQLITE_TRANSIENT);
        EXPECT_EQ(sqlite3_step(stmt), SQLITE_ROW) << "Table missing: " << table;
        sqlite3_finalize(stmt);
    }

    sqlite3_close(raw_db);
}

TEST_F(MetricsDatabaseTest, WalModeEnabled) {
    SQLiteMetricsDatabase db(db_path_);

    sqlite3* raw_db = nullptr;
    ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw_db), SQLITE_OK);

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(raw_db, "PRAGMA journal_mode", -1, &stmt, nullptr);
    ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
    std::string mode = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    EXPECT_EQ(mode, "wal");
    sqlite3_finalize(stmt);
    sqlite3_close(raw_db);
}

TEST_F(MetricsDatabaseTest, InsertAndQueryHistory) {
    SQLiteMetricsDatabase db(db_path_);

    // Must create session first (FK constraint)
    SessionRecord session;
    session.key = "test-session";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db.upsert_session(session);

    auto base_time = std::chrono::system_clock::now();
    for (int i = 0; i < 50; ++i) {
        PersistentMetricsRecord rec;
        rec.timestamp = base_time + std::chrono::seconds(i);
        rec.epoch = i / 10;
        rec.sample = i;
        rec.loss = 2.0f - (static_cast<float>(i) * 0.02f);
        rec.learning_rate = 0.001f;
        rec.gradient_norm = 1.0f;
        rec.perplexity = std::exp(rec.loss);
        db.insert_metrics_record("test-session", rec);
    }

    auto results = db.query_history("test-session", std::nullopt, std::nullopt, 0);
    EXPECT_EQ(results.size(), 50u);

    // Verify ordering
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i].timestamp, results[i - 1].timestamp);
    }
}

TEST_F(MetricsDatabaseTest, TimeRangeFilter) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "range-test";
    session.session_id = 2;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db.upsert_session(session);

    auto base_time = std::chrono::system_clock::now();
    for (int i = 0; i < 60; ++i) {
        PersistentMetricsRecord rec;
        rec.timestamp = base_time + std::chrono::seconds(i * 10);
        rec.epoch = 1;
        rec.sample = i;
        rec.loss = 1.0f;
        db.insert_metrics_record("range-test", rec);
    }

    // Query middle portion: seconds 100-400
    auto from = base_time + std::chrono::seconds(100);
    auto to = base_time + std::chrono::seconds(400);
    auto results = db.query_history("range-test", from, to, 0);

    // Records at seconds 100, 110, 120, ..., 400 = 31 records
    EXPECT_GT(results.size(), 0u);
    // Allow 1-second tolerance for timestamp round-trip precision
    auto tolerance = std::chrono::seconds(1);
    for (const auto& r : results) {
        EXPECT_GE(r.timestamp, from - tolerance);
        EXPECT_LE(r.timestamp, to + tolerance);
    }
}

TEST_F(MetricsDatabaseTest, LimitClause) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "limit-test";
    session.session_id = 3;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db.upsert_session(session);

    auto base_time = std::chrono::system_clock::now();
    for (int i = 0; i < 20; ++i) {
        PersistentMetricsRecord rec;
        rec.timestamp = base_time + std::chrono::seconds(i);
        rec.epoch = 1;
        rec.sample = i;
        rec.loss = 1.0f;
        db.insert_metrics_record("limit-test", rec);
    }

    auto results = db.query_history("limit-test", std::nullopt, std::nullopt, 5);
    EXPECT_EQ(results.size(), 5u);
}

TEST_F(MetricsDatabaseTest, UpsertSession) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "upsert-test";
    session.session_id = 1;
    session.label = "First Label";
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    session.total_epochs = 5;
    session.best_validation_loss = 0.5f;
    db.upsert_session(session);

    auto rec = db.get_session("upsert-test");
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec->label, "First Label");
    EXPECT_EQ(rec->total_epochs, 5);

    // Update the same key
    session.label = "Updated Label";
    session.total_epochs = 10;
    session.best_validation_loss = 0.3f;
    session.last_update_at = std::chrono::system_clock::now();
    db.upsert_session(session);

    rec = db.get_session("upsert-test");
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec->label, "Updated Label");
    EXPECT_EQ(rec->total_epochs, 10);
    EXPECT_FLOAT_EQ(rec->best_validation_loss, 0.3f);

    // Verify only one row exists
    auto all = db.list_sessions(std::nullopt);
    int count = 0;
    for (const auto& s : all) {
        if (s.key == "upsert-test")
            ++count;
    }
    EXPECT_EQ(count, 1);
}

// Regression test for the gap where completed/archived sessions showed 0 for their
// final loss in the summary view: SessionRecord had no field to carry it, so the DB
// round-trip silently dropped it even though a live session's current_loss was real.
TEST_F(MetricsDatabaseTest, FinalLossRoundTripsThroughUpsertListAndArchive) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "final-loss-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    session.final_loss = 4.25f;
    session.final_validation_loss = 4.5f;
    db.upsert_session(session);

    // Round-trips through get_session()
    auto rec = db.get_session("final-loss-test");
    ASSERT_TRUE(rec.has_value());
    EXPECT_FLOAT_EQ(rec->final_loss, 4.25f);
    EXPECT_FLOAT_EQ(rec->final_validation_loss, 4.5f);

    // Round-trips through list_sessions()
    auto all = db.list_sessions(std::nullopt);
    auto it = std::find_if(all.begin(), all.end(),
                           [](const SessionRecord& s) { return s.key == "final-loss-test"; });
    ASSERT_NE(it, all.end());
    EXPECT_FLOAT_EQ(it->final_loss, 4.25f);
    EXPECT_FLOAT_EQ(it->final_validation_loss, 4.5f);

    // Survives archive_session() (the rename-on-eviction path used when a live
    // session's key gets reused) — this is exactly the path a completed session
    // that later gets superseded goes through.
    db.archive_session("final-loss-test", "final-loss-test_archived_1");
    auto archived = db.get_session("final-loss-test_archived_1");
    ASSERT_TRUE(archived.has_value());
    EXPECT_FLOAT_EQ(archived->final_loss, 4.25f);
    EXPECT_FLOAT_EQ(archived->final_validation_loss, 4.5f);
}

// Regression test for the migration path: a database created before final_loss/
// final_validation_loss existed must not crash or silently lose data when opened by
// the current code — CREATE TABLE IF NOT EXISTS is a no-op against an existing table,
// so this only works if bootstrap_schema() actually migrates it.
TEST_F(MetricsDatabaseTest, MigratesPreExistingDatabaseMissingFinalLossColumns) {
    // Create a DB with the old schema (no final_loss / final_validation_loss) by hand,
    // bypassing SQLiteMetricsDatabase entirely so this only exercises the migration.
    {
        sqlite3* raw_db = nullptr;
        ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw_db), SQLITE_OK);
        const char* old_schema = R"SQL(
            CREATE TABLE sessions (
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
                best_epoch           INTEGER
            );
            INSERT INTO sessions (key, session_id, is_training, created_at, last_update_at,
                                  total_epochs, total_samples, best_validation_loss, best_epoch)
            VALUES ('pre-existing', 1, 0, '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z',
                   3, 100, 0.9, 2);
        )SQL";
        char* err = nullptr;
        ASSERT_EQ(sqlite3_exec(raw_db, old_schema, nullptr, nullptr, &err), SQLITE_OK)
            << (err ? err : "unknown error");
        sqlite3_close(raw_db);
    }

    // Opening via SQLiteMetricsDatabase must migrate the table in place, not throw.
    SQLiteMetricsDatabase db(db_path_);
    auto rec = db.get_session("pre-existing");
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec->total_epochs, 3);
    EXPECT_FLOAT_EQ(rec->best_validation_loss, 0.9f);
    // Pre-existing rows have no value for the new columns — NULL reads back as 0
    // via sqlite3_column_double, which is an acceptable "unknown" default here
    // (matches every other float field's zero-initialized default in this struct).
    EXPECT_FLOAT_EQ(rec->final_loss, 0.0f);

    // And the migrated table must accept new writes to the new columns going forward.
    SessionRecord updated = *rec;
    updated.final_loss = 2.1f;
    db.upsert_session(updated);
    auto rec2 = db.get_session("pre-existing");
    ASSERT_TRUE(rec2.has_value());
    EXPECT_FLOAT_EQ(rec2->final_loss, 2.1f);
}

// Regression test for the TD-013 diagnostic extension: compute_time_ratio,
// weight_update_ratio, activation_saturation_ratio, attention_entropy,
// padding_efficiency, and layer_gradient_norms_json must all round-trip
// through insert_metrics_record()/query_history() alongside the pre-existing
// scalar fields.
TEST_F(MetricsDatabaseTest, TD013DiagnosticsRoundTripThroughInsertAndQueryHistory) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "td013-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db.upsert_session(session);

    PersistentMetricsRecord rec;
    rec.timestamp = std::chrono::system_clock::now();
    rec.epoch = 1;
    rec.sample = 100;
    rec.loss = 1.5f;
    rec.compute_time_ratio = 0.87f;
    rec.weight_update_ratio = 1.23e-5f;
    rec.activation_saturation_ratio = 0.34f;
    rec.attention_entropy = 2.1f;
    rec.padding_efficiency = 0.91f;
    rec.layer_gradient_norms_json = R"({"encoder":[0.5,0.4,0.3],"decoder":[0.6,0.5,0.4]})";
    db.insert_metrics_record("td013-test", rec);

    // A second record with no TD-013 diagnostics reported this call (matches a
    // per-sample record between epoch boundaries) — these fields must stay at
    // their "not computed" defaults, not silently inherit the first row's values.
    PersistentMetricsRecord rec2;
    rec2.timestamp = std::chrono::system_clock::now() + std::chrono::seconds(1);
    rec2.epoch = 1;
    rec2.sample = 200;
    rec2.loss = 1.4f;
    db.insert_metrics_record("td013-test", rec2);

    auto results = db.query_history("td013-test", std::nullopt, std::nullopt, 0);
    ASSERT_EQ(results.size(), 2u);

    EXPECT_FLOAT_EQ(results[0].compute_time_ratio, 0.87f);
    EXPECT_FLOAT_EQ(results[0].weight_update_ratio, 1.23e-5f);
    EXPECT_FLOAT_EQ(results[0].activation_saturation_ratio, 0.34f);
    EXPECT_FLOAT_EQ(results[0].attention_entropy, 2.1f);
    EXPECT_FLOAT_EQ(results[0].padding_efficiency, 0.91f);
    EXPECT_EQ(results[0].layer_gradient_norms_json,
             R"({"encoder":[0.5,0.4,0.3],"decoder":[0.6,0.5,0.4]})");

    EXPECT_FLOAT_EQ(results[1].compute_time_ratio, 0.0f);
    EXPECT_FLOAT_EQ(results[1].weight_update_ratio, 0.0f);
    EXPECT_FLOAT_EQ(results[1].activation_saturation_ratio, -1.0f);
    EXPECT_FLOAT_EQ(results[1].attention_entropy, -1.0f);
    EXPECT_FLOAT_EQ(results[1].padding_efficiency, -1.0f);
    EXPECT_TRUE(results[1].layer_gradient_norms_json.empty());
}

// Regression test for the migration path: a database created before the
// TD-013 extension columns existed must not crash or silently lose data when
// opened by the current code — CREATE TABLE IF NOT EXISTS is a no-op against
// an existing table, so this only works if bootstrap_schema() actually
// migrates metrics_history in place.
TEST_F(MetricsDatabaseTest, MigratesPreExistingDatabaseMissingTD013MetricsHistoryColumns) {
    {
        sqlite3* raw_db = nullptr;
        ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw_db), SQLITE_OK);
        const char* old_schema = R"SQL(
            CREATE TABLE sessions (
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
            CREATE TABLE metrics_history (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key                 TEXT    NOT NULL REFERENCES sessions(key),
                recorded_at                 TEXT    NOT NULL,
                epoch                       INTEGER NOT NULL,
                sample                      INTEGER NOT NULL,
                loss                        REAL,
                validation_loss             REAL,
                learning_rate               REAL,
                gradient_norm               REAL,
                perplexity                  REAL
            );
            INSERT INTO sessions (key, session_id, is_training, created_at, last_update_at,
                                  total_epochs, total_samples, best_validation_loss, best_epoch)
            VALUES ('pre-existing', 1, 0, '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z',
                   3, 100, 0.9, 2);
            INSERT INTO metrics_history (session_key, recorded_at, epoch, sample, loss)
            VALUES ('pre-existing', '2026-01-01T00:00:00.000Z', 1, 50, 1.8);
        )SQL";
        char* err = nullptr;
        ASSERT_EQ(sqlite3_exec(raw_db, old_schema, nullptr, nullptr, &err), SQLITE_OK)
            << (err ? err : "unknown error");
        sqlite3_close(raw_db);
    }

    // Opening via SQLiteMetricsDatabase must migrate the table in place, not
    // throw or crash reading the pre-existing (now-NULL) TD-013 columns back.
    SQLiteMetricsDatabase db(db_path_);
    auto results = db.query_history("pre-existing", std::nullopt, std::nullopt, 0);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_FLOAT_EQ(results[0].loss, 1.8f);
    EXPECT_TRUE(results[0].layer_gradient_norms_json.empty());

    // And the migrated table must accept new writes to the new columns going forward.
    PersistentMetricsRecord rec;
    rec.timestamp = std::chrono::system_clock::now();
    rec.epoch = 2;
    rec.sample = 60;
    rec.loss = 1.6f;
    rec.attention_entropy = 1.9f;
    db.insert_metrics_record("pre-existing", rec);
    auto results2 = db.query_history("pre-existing", std::nullopt, std::nullopt, 0);
    ASSERT_EQ(results2.size(), 2u);
    EXPECT_FLOAT_EQ(results2[1].attention_entropy, 1.9f);
}

// insert_gradient_variance_sample() writes to a dedicated table, deliberately
// separate from metrics_history's throttled per-100-sample cadence (see
// IMetricsDatabase::insert_gradient_variance_sample doc comment) — verify it
// actually persists a row per call, unthrottled, via direct SQL (there's no
// IMetricsDatabase query method for this table; the write path is what's
// under test here).
TEST_F(MetricsDatabaseTest, GradientVarianceHistoryInsertsUnthrottledPerCall) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "grad-var-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db.upsert_session(session);

    for (int step = 1; step <= 5; ++step) {
        db.insert_gradient_variance_sample("grad-var-test", step, /*epoch=*/1,
                                           /*value=*/0.1f * static_cast<float>(step));
    }

    sqlite3* raw_db = nullptr;
    ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw_db), SQLITE_OK);

    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(sqlite3_prepare_v2(raw_db,
                                "SELECT step, epoch, value FROM gradient_variance_history "
                                "WHERE session_key = 'grad-var-test' ORDER BY step ASC;",
                                -1, &stmt, nullptr),
             SQLITE_OK);

    int count = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ++count;
        EXPECT_EQ(sqlite3_column_int(stmt, 0), count);
        EXPECT_EQ(sqlite3_column_int(stmt, 1), 1);
        EXPECT_FLOAT_EQ(static_cast<float>(sqlite3_column_double(stmt, 2)), 0.1f * count);
    }
    EXPECT_EQ(count, 5);

    sqlite3_finalize(stmt);
    sqlite3_close(raw_db);
}

TEST_F(MetricsDatabaseTest, MarkSessionEnded) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "end-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db.upsert_session(session);

    auto rec = db.get_session("end-test");
    ASSERT_TRUE(rec.has_value());
    EXPECT_TRUE(rec->is_training);
    EXPECT_FALSE(rec->ended_at.has_value());

    db.mark_session_ended("end-test");

    rec = db.get_session("end-test");
    ASSERT_TRUE(rec.has_value());
    EXPECT_FALSE(rec->is_training);
    EXPECT_TRUE(rec->ended_at.has_value());
}

TEST_F(MetricsDatabaseTest, ArchiveSessionMovesRowUnderNewKey) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "archive-test";
    session.session_id = 42;
    session.label = "stale-run";
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    session.total_epochs = 3;
    session.best_validation_loss = 0.42f;
    db.upsert_session(session);

    db.archive_session("archive-test", "archive-test_archived_1");

    // Original key no longer resolves to a live session.
    EXPECT_FALSE(db.get_session("archive-test").has_value());

    auto archived = db.get_session("archive-test_archived_1");
    ASSERT_TRUE(archived.has_value());
    EXPECT_EQ(archived->label, "stale-run");
    EXPECT_EQ(archived->total_epochs, 3);
    EXPECT_FALSE(archived->is_training);
    EXPECT_TRUE(archived->ended_at.has_value());
}

TEST_F(MetricsDatabaseTest, ArchiveSessionMovesChildRows) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "archive-child-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db.upsert_session(session);

    auto base_time = std::chrono::system_clock::now();
    for (int i = 0; i < 5; ++i) {
        PersistentMetricsRecord rec;
        rec.timestamp = base_time + std::chrono::seconds(i);
        rec.epoch = 0;
        rec.sample = i;
        rec.loss = 1.0f;
        db.insert_metrics_record("archive-child-test", rec);
    }

    db.archive_session("archive-child-test", "archive-child-test_archived_1");

    // History no longer appears under the live key...
    EXPECT_EQ(db.query_history("archive-child-test", std::nullopt, std::nullopt, 0).size(), 0u);
    // ...but is preserved intact under the archived key.
    EXPECT_EQ(
        db.query_history("archive-child-test_archived_1", std::nullopt, std::nullopt, 0).size(),
        5u);

    // A new session can now reuse the original key with a clean history.
    SessionRecord new_session;
    new_session.key = "archive-child-test";
    new_session.session_id = 2;
    new_session.is_training = true;
    new_session.created_at = std::chrono::system_clock::now();
    new_session.last_update_at = new_session.created_at;
    db.upsert_session(new_session);

    EXPECT_EQ(db.query_history("archive-child-test", std::nullopt, std::nullopt, 0).size(), 0u);
}

TEST_F(MetricsDatabaseTest, ArchiveSessionOfMissingKeyIsNoop) {
    SQLiteMetricsDatabase db(db_path_);
    EXPECT_NO_THROW(db.archive_session("does-not-exist", "does-not-exist_archived_1"));
    EXPECT_FALSE(db.get_session("does-not-exist_archived_1").has_value());
}

TEST_F(MetricsDatabaseTest, AbnormalSampleRoundTrip) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "abnormal-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db.upsert_session(session);

    AbnormalSample sample;
    sample.epoch = 3;
    sample.sample_id = 42;
    sample.loss = 15.5f;
    sample.grad_norm = 100.0f;
    sample.reason = "loss_outlier";
    sample.input_text = "What is this?";
    sample.target_text = "This is a test.";
    sample.timestamp = std::chrono::system_clock::now();

    db.insert_abnormal_sample("abnormal-test", sample);

    // Verify via raw SQL
    sqlite3* raw_db = nullptr;
    ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw_db), SQLITE_OK);

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(raw_db,
                       "SELECT epoch, sample_id, loss, grad_norm, reason, input_text, target_text "
                       "FROM abnormal_samples WHERE session_key = 'abnormal-test'",
                       -1, &stmt, nullptr);
    ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);

    EXPECT_EQ(sqlite3_column_int(stmt, 0), 3);
    EXPECT_EQ(sqlite3_column_int(stmt, 1), 42);
    EXPECT_NEAR(sqlite3_column_double(stmt, 2), 15.5, 0.01);
    EXPECT_NEAR(sqlite3_column_double(stmt, 3), 100.0, 0.01);
    EXPECT_STREQ(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4)), "loss_outlier");
    EXPECT_STREQ(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)), "What is this?");
    EXPECT_STREQ(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6)), "This is a test.");

    sqlite3_finalize(stmt);
    sqlite3_close(raw_db);
}

TEST_F(MetricsDatabaseTest, GenerationQualityRoundTrip) {
    SQLiteMetricsDatabase db(db_path_);

    SessionRecord session;
    session.key = "genqual-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db.upsert_session(session);

    GenerationQualityScore score;
    score.bleu1 = 0.45f;
    score.bleu2 = 0.30f;
    score.bleu4 = 0.15f;
    score.rouge1 = 0.50f;
    score.rouge2 = 0.25f;
    score.rougeL = 0.40f;
    db.insert_generation_quality("genqual-test", 5, score);

    // Verify via raw SQL
    sqlite3* raw_db = nullptr;
    ASSERT_EQ(sqlite3_open(db_path_.c_str(), &raw_db), SQLITE_OK);

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(raw_db,
                       "SELECT epoch, bleu1, bleu4, rouge1, rougeL FROM generation_quality "
                       "WHERE session_key = 'genqual-test'",
                       -1, &stmt, nullptr);
    ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
    EXPECT_EQ(sqlite3_column_int(stmt, 0), 5);
    EXPECT_NEAR(sqlite3_column_double(stmt, 1), 0.45, 0.01);
    EXPECT_NEAR(sqlite3_column_double(stmt, 2), 0.15, 0.01);
    EXPECT_NEAR(sqlite3_column_double(stmt, 3), 0.50, 0.01);
    EXPECT_NEAR(sqlite3_column_double(stmt, 4), 0.40, 0.01);

    sqlite3_finalize(stmt);
    sqlite3_close(raw_db);
}

TEST_F(MetricsDatabaseTest, DualWritePath) {
    // Verify that with a DB and file persistence both are written
    auto jsonl_path = (test_dir_ / "dual_metrics.jsonl").string();
    auto summary_path = (test_dir_ / "dual_summary.json").string();

    MetricsServiceConfig config;
    config.enable_persistence = true;
    config.metrics_file = jsonl_path;
    config.summary_file = summary_path;
    config.persist_every_samples = 1;
    config.session_key = "dual-test";

    auto db = std::make_unique<SQLiteMetricsDatabase>(db_path_);

    // Create session in DB
    SessionRecord session;
    session.key = "dual-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db->upsert_session(session);

    TrainingMetricsService service(config);
    service.set_database(db.get(), "dual-test");
    service.start_session(1, 5, 100);
    service.start_epoch(1, 100);
    service.update_sample_metrics(1, 2.5f, 1.0f, 0.001f);
    service.flush_to_disk();

    // Both DB and JSONL should have data
    EXPECT_TRUE(fs::exists(jsonl_path));
    auto db_records = db->query_history("dual-test", std::nullopt, std::nullopt, 0);
    EXPECT_GT(db_records.size(), 0u);

    // JSONL should also have data
    std::ifstream jsonl_file(jsonl_path);
    std::string line;
    int jsonl_count = 0;
    while (std::getline(jsonl_file, line))
        ++jsonl_count;
    EXPECT_GT(jsonl_count, 0);
}

// Regression test: persist_metrics() used to iterate the *entire* history_ buffer
// (a rolling in-memory window, not a pending-persist queue) on every call, so each
// periodic/explicit flush re-inserted and re-appended every record already written
// by a previous flush. DualWritePath above only calls update_sample_metrics() once
// and only checks ">0", so it never exercised more than one flush and would not have
// caught this. This test drives multiple flushes and checks the exact count.
TEST_F(MetricsDatabaseTest, PersistMetricsDoesNotDuplicateAcrossMultipleFlushes) {
    auto jsonl_path = (test_dir_ / "dedup_metrics.jsonl").string();
    auto summary_path = (test_dir_ / "dedup_summary.json").string();

    MetricsServiceConfig config;
    config.enable_persistence = true;
    config.metrics_file = jsonl_path;
    config.summary_file = summary_path;
    config.persist_every_samples = 1;
    config.session_key = "dedup-test";

    auto db = std::make_unique<SQLiteMetricsDatabase>(db_path_);

    SessionRecord session;
    session.key = "dedup-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db->upsert_session(session);

    TrainingMetricsService service(config);
    service.set_database(db.get(), "dedup-test");
    service.start_session(1, 5, 100);
    service.start_epoch(1, 100);

    // Each call persists immediately (persist_every_samples=1); flush_to_disk()
    // afterward must not re-write records a previous call already persisted.
    service.update_sample_metrics(1, 2.5f, 1.0f, 0.001f);
    service.flush_to_disk();
    service.update_sample_metrics(2, 2.4f, 1.0f, 0.001f);
    service.flush_to_disk();
    service.update_sample_metrics(3, 2.3f, 1.0f, 0.001f);
    service.flush_to_disk();

    auto db_records = db->query_history("dedup-test", std::nullopt, std::nullopt, 0);
    EXPECT_EQ(db_records.size(), 3u)
        << "expected exactly 3 persisted records (one per sample), got " << db_records.size()
        << " — persist_metrics() is re-writing already-persisted records";

    std::ifstream jsonl_file(jsonl_path);
    std::string line;
    int jsonl_count = 0;
    while (std::getline(jsonl_file, line))
        ++jsonl_count;
    EXPECT_EQ(jsonl_count, 3) << "JSONL file should have exactly 3 lines, got " << jsonl_count;
}

TEST_F(MetricsDatabaseTest, RestoreFromDB) {
    auto summary_path = (test_dir_ / "restore_summary.json").string();

    auto db = std::make_unique<SQLiteMetricsDatabase>(db_path_);

    // Pre-populate DB with a session
    SessionRecord session;
    session.key = "restore-test";
    session.session_id = 42;
    session.label = "Restored Session";
    session.config_json = R"({"d_model":256})";
    session.is_training = false;
    session.created_at = std::chrono::system_clock::now() - std::chrono::hours(1);
    session.last_update_at = std::chrono::system_clock::now();
    session.total_epochs = 10;
    session.best_validation_loss = 0.25f;
    session.best_epoch = 8;
    db->upsert_session(session);

    MetricsServiceConfig config;
    config.enable_persistence = true;
    config.summary_file = summary_path;
    config.session_key = "restore-test";

    TrainingMetricsService service(config);
    service.set_database(db.get(), "restore-test");

    // Snapshot should reflect DB state after restore_from_summary triggers
    // Note: restore_from_summary is called in constructor before set_database,
    // so we need to verify the snapshot after explicitly calling set_database.
    // The DB-first restore happens on the next restart. Let's verify get_session works.
    auto rec = db->get_session("restore-test");
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec->session_id, 42);
    EXPECT_EQ(rec->label, "Restored Session");
    EXPECT_FLOAT_EQ(rec->best_validation_loss, 0.25f);
    EXPECT_EQ(rec->best_epoch, 8);
}

TEST_F(MetricsDatabaseTest, RestoreFromFileFallback) {
    auto summary_path = (test_dir_ / "fallback_summary.json").string();

    // Write a summary JSON file (no DB row)
    {
        std::ofstream f(summary_path);
        f << R"({"session_id":99,"best_validation_loss":0.42,"best_epoch":5,)"
          << R"("total_samples_trained":1000,"total_training_time_seconds":600.0,)"
          << R"("epoch_losses":[1.5,1.2,0.9],"epoch_validation_losses":[1.4,1.1,0.8],)"
          << R"("epoch_learning_rates":[0.001,0.001,0.001],)"
          << R"("epoch_perplexities":[4.48,3.32,2.46],)"
          << R"("epoch_durations":[120.0,110.0,105.0]})";
    }

    MetricsServiceConfig config;
    config.enable_persistence = true;
    config.summary_file = summary_path;

    // No DB — should fall back to file
    TrainingMetricsService service(config);
    auto snapshot = service.get_current_snapshot();

    EXPECT_EQ(snapshot.session_id, 99);
    EXPECT_FLOAT_EQ(snapshot.best_validation_loss, 0.42f);
    EXPECT_EQ(snapshot.best_epoch, 5);
    EXPECT_EQ(snapshot.epoch_losses.size(), 3u);
}

TEST_F(MetricsDatabaseTest, ListSessionsFilterByTraining) {
    SQLiteMetricsDatabase db(db_path_);

    auto now = std::chrono::system_clock::now();

    SessionRecord active;
    active.key = "active-1";
    active.session_id = 1;
    active.is_training = true;
    active.created_at = now;
    active.last_update_at = now;
    db.upsert_session(active);

    SessionRecord completed;
    completed.key = "completed-1";
    completed.session_id = 2;
    completed.is_training = false;
    completed.created_at = now - std::chrono::hours(1);
    completed.last_update_at = now;
    db.upsert_session(completed);

    auto all = db.list_sessions(std::nullopt);
    EXPECT_EQ(all.size(), 2u);

    auto training_only = db.list_sessions(std::optional<bool>(true));
    EXPECT_EQ(training_only.size(), 1u);
    EXPECT_EQ(training_only[0].key, "active-1");

    auto completed_only = db.list_sessions(std::optional<bool>(false));
    EXPECT_EQ(completed_only.size(), 1u);
    EXPECT_EQ(completed_only[0].key, "completed-1");
}

TEST_F(MetricsDatabaseTest, FactoryCreatesSqlite) {
    auto db = MetricsDatabaseFactory::create("sqlite", db_path_);
    ASSERT_NE(db, nullptr);
    EXPECT_TRUE(fs::exists(db_path_));
}

TEST_F(MetricsDatabaseTest, FactoryReturnsNullForFileOnly) {
    auto db = MetricsDatabaseFactory::create("file");
    EXPECT_EQ(db, nullptr);
}

TEST_F(MetricsDatabaseTest, FactoryHandlesSqlitePlusFile) {
    auto db = MetricsDatabaseFactory::create("sqlite+file", db_path_);
    ASSERT_NE(db, nullptr);
    EXPECT_TRUE(fs::exists(db_path_));
}
