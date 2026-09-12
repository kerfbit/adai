// Tests for PostgresMetricsDatabase (TD-042).
//
// Only built when ENABLE_POSTGRES_METRICS is on (requires libpq — see src/CMakeLists.txt).
// Mirrors MetricsDatabaseTest.cpp's IMetricsDatabase-interface-level coverage against a real
// PostgreSQL server rather than a mock: a scratch instance is initdb'd and started once for the
// whole binary (SetUpTestSuite/TearDownTestSuite — a fresh initdb per test would be too slow), on
// a unix socket in a scratch directory with a PID-derived port so parallel `ctest -j` runs don't
// collide, matching TD-065's own reproduction approach rather than touching the system's real
// postgresql.service. Each test gets its own freshly-created, freshly-dropped database for
// isolation (CREATE DATABASE/DROP DATABASE around every test), so tests never see each other's
// rows the way the SQLite suite's separate temp files per test already ensure.
//
// If no usable local `initdb`/`pg_ctl` is found, every test in this file cleanly skips via
// GTEST_SKIP() rather than failing — a missing Postgres toolchain isn't a test failure, it's an
// environment gap this file works around (see TD-042's own "not built by default" framing).

#ifdef ADAI_ENABLE_POSTGRES

#include <gtest/gtest.h>
#include <libpq-fe.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <limits>
#include <string>
#include "GenerationQualityMetrics.hpp"
#include "IMetricsReporter.hpp"
#include "MetricsDatabase.hpp"
#include "PostgresMetricsDatabase.hpp"
#include "TrainingMetricsService.hpp"

namespace fs = std::filesystem;

namespace {

// Runs `cmd`, returning its exit code. Output is discarded on success and dumped on failure
// (stderr) so a broken environment (e.g. no initdb) is diagnosable from `ctest --output-on-failure`
// rather than just silently skipping.
//
// std::system() already runs `cmd` via `/bin/sh -c` on its own — this must NOT additionally wrap
// it in `bash -c '...'` (a real earlier bug here): pg_ctl's own `-o "... -h ''"` argument contains
// a single-quote pair, which would terminate an outer single-quoted wrapper early and silently
// mangle the command into something else entirely.
int run_shell(const std::string& cmd) {
    std::string full = cmd + " > /tmp/adai_pgtest_out.$$ 2>&1; ec=$?; cat /tmp/adai_pgtest_out.$$ "
                             ">&2; rm -f /tmp/adai_pgtest_out.$$; exit $ec";
    return WEXITSTATUS(std::system(full.c_str()));
}

std::string find_pg_bindir() {
    // pg_config is the canonical, version-agnostic way to find the installed server binaries —
    // preferred over guessing a versioned path like /usr/lib/postgresql/16/bin.
    FILE* p = popen("pg_config --bindir 2>/dev/null", "r");
    if (p) {
        char buf[512] = {0};
        if (fgets(buf, sizeof(buf), p)) {
            std::string s(buf);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
                s.pop_back();
            pclose(p);
            if (!s.empty() && fs::exists(s + "/initdb"))
                return s;
        } else {
            pclose(p);
        }
    }
    // Fallback: common Debian/Ubuntu versioned layout (pg_config not always on PATH even when
    // the server package is installed).
    if (fs::exists("/usr/lib/postgresql")) {
        for (const auto& entry : fs::directory_iterator("/usr/lib/postgresql")) {
            auto bindir = entry.path() / "bin";
            if (fs::exists(bindir / "initdb"))
                return bindir.string();
        }
    }
    return "";
}

}  // namespace

class PostgresMetricsDatabaseTest : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
        pg_bindir_ = find_pg_bindir();
        if (pg_bindir_.empty()) {
            pg_available_ = false;
            return;
        }

        scratch_dir_ = fs::temp_directory_path() / "adai_postgres_metrics_test";
        fs::remove_all(scratch_dir_);
        fs::create_directories(scratch_dir_);
        data_dir_ = scratch_dir_ / "data";
        // PID-derived, not fixed: two ctest -j workers each running this file would otherwise
        // both try to bind the same port.
        port_ = 55000 + (static_cast<int>(getpid()) % 1000);

        if (run_shell(pg_bindir_ + "/initdb -D " + data_dir_.string() +
                     " -U adai_test -A trust --no-sync") != 0) {
            pg_available_ = false;
            return;
        }

        std::string start_cmd = pg_bindir_ + "/pg_ctl -D " + data_dir_.string() +
                                " -o \"-p " + std::to_string(port_) + " -k " +
                                scratch_dir_.string() + " -h ''\" -l " +
                                (scratch_dir_ / "server.log").string() + " start";
        if (run_shell(start_cmd) != 0) {
            pg_available_ = false;
            return;
        }

        // Poll for real readiness (server start above returns once pg_ctl believes it's up, but
        // connect anyway with a short retry loop to absorb any last-mile scheduling delay).
        std::string conninfo = "host=" + scratch_dir_.string() + " port=" +
                               std::to_string(port_) + " dbname=postgres user=adai_test";
        bool ready = false;
        for (int i = 0; i < 50 && !ready; ++i) {
            PGconn* c = PQconnectdb(conninfo.c_str());
            if (PQstatus(c) == CONNECTION_OK)
                ready = true;
            PQfinish(c);
            if (!ready)
                usleep(100 * 1000);
        }
        pg_available_ = ready;
    }

    static void TearDownTestSuite() {
        if (!pg_bindir_.empty() && fs::exists(data_dir_)) {
            run_shell(pg_bindir_ + "/pg_ctl -D " + data_dir_.string() + " stop -m fast");
        }
        if (!scratch_dir_.empty()) {
            fs::remove_all(scratch_dir_);
        }
    }

    void SetUp() override {
        if (!pg_available_) {
            GTEST_SKIP() << "no local PostgreSQL initdb/pg_ctl found (or it failed to start) — "
                           "see server.log in the scratch dir for details";
        }

        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string name = info ? info->name() : "unknown";
        std::replace_if(
            name.begin(), name.end(), [](char c) { return !std::isalnum((unsigned char)c); }, '_');
        static std::atomic<int> counter{0};
        db_name_ = "adai_test_" + name + "_" + std::to_string(counter.fetch_add(1));
        db_name_.resize(std::min<size_t>(db_name_.size(), 63));  // Postgres identifier limit
        // Lowercase: an unquoted CREATE DATABASE identifier is folded to lowercase by Postgres,
        // but libpq's dbname= connection parameter is sent to the server literally, with no such
        // folding — a mixed-case name here (test names are e.g. "SchemaBootstrap") would create
        // "adai_test_schemabootstrap_0" while every later connection asked for
        // "adai_test_SchemaBootstrap_0" and got "FATAL: database ... does not exist". Reproduced
        // directly before adding this line.
        std::transform(db_name_.begin(), db_name_.end(), db_name_.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        std::string admin_conninfo = maintenance_conninfo();
        PGconn* conn = PQconnectdb(admin_conninfo.c_str());
        ASSERT_EQ(PQstatus(conn), CONNECTION_OK) << PQerrorMessage(conn);
        PGresult* res = PQexec(conn, ("CREATE DATABASE " + db_name_).c_str());
        ASSERT_EQ(PQresultStatus(res), PGRES_COMMAND_OK) << PQerrorMessage(conn);
        PQclear(res);
        PQfinish(conn);

        db_url_ = "host=" + scratch_dir_.string() + " port=" + std::to_string(port_) +
                  " dbname=" + db_name_ + " user=adai_test";
    }

    void TearDown() override {
        if (!pg_available_)
            return;
        PGconn* conn = PQconnectdb(maintenance_conninfo().c_str());
        if (PQstatus(conn) == CONNECTION_OK) {
            PGresult* res =
                PQexec(conn, ("DROP DATABASE IF EXISTS " + db_name_ + " WITH (FORCE)").c_str());
            PQclear(res);
        }
        PQfinish(conn);
    }

    std::string maintenance_conninfo() const {
        return "host=" + scratch_dir_.string() + " port=" + std::to_string(port_) +
              " dbname=postgres user=adai_test";
    }

    // Runs a raw query against this test's own database (not through PostgresMetricsDatabase) —
    // for verification, the same role sqlite3_prepare_v2/sqlite3_step play in the SQLite suite.
    PGresult* raw_query(const std::string& sql) {
        if (!raw_conn_) {
            raw_conn_ = PQconnectdb(db_url_.c_str());
        }
        return PQexec(raw_conn_, sql.c_str());
    }

    void TearDownRawConn() {
        if (raw_conn_) {
            PQfinish(raw_conn_);
            raw_conn_ = nullptr;
        }
    }

    std::string db_url_;
    std::string db_name_;
    PGconn* raw_conn_ = nullptr;

    static bool pg_available_;
    static std::string pg_bindir_;
    static fs::path scratch_dir_;
    static fs::path data_dir_;
    static int port_;
};

bool PostgresMetricsDatabaseTest::pg_available_ = false;
std::string PostgresMetricsDatabaseTest::pg_bindir_;
fs::path PostgresMetricsDatabaseTest::scratch_dir_;
fs::path PostgresMetricsDatabaseTest::data_dir_;
int PostgresMetricsDatabaseTest::port_ = 0;

TEST_F(PostgresMetricsDatabaseTest, SchemaBootstrap) {
    PostgresMetricsDatabase db(db_url_, /*pool_size=*/2);

    PGresult* res = raw_query(
        "SELECT version FROM schema_version WHERE version = 1");
    ASSERT_EQ(PQresultStatus(res), PGRES_TUPLES_OK) << PQerrorMessage(raw_conn_);
    EXPECT_EQ(PQntuples(res), 1);
    PQclear(res);

    for (const char* table : {"schema_version", "sessions", "metrics_history",
                              "gradient_variance_history", "generation_quality",
                              "abnormal_samples"}) {
        std::string sql = "SELECT to_regclass('" + std::string(table) + "') IS NOT NULL";
        res = raw_query(sql);
        ASSERT_EQ(PQresultStatus(res), PGRES_TUPLES_OK) << PQerrorMessage(raw_conn_);
        EXPECT_STREQ(PQgetvalue(res, 0, 0), "t") << "table missing: " << table;
        PQclear(res);
    }
    TearDownRawConn();
}

TEST_F(PostgresMetricsDatabaseTest, InsertAndQueryHistory) {
    PostgresMetricsDatabase db(db_url_, 2);

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
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i].timestamp, results[i - 1].timestamp);
    }
}

TEST_F(PostgresMetricsDatabaseTest, TimeRangeFilter) {
    PostgresMetricsDatabase db(db_url_, 2);

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

    auto from = base_time + std::chrono::seconds(100);
    auto to = base_time + std::chrono::seconds(400);
    auto results = db.query_history("range-test", from, to, 0);

    EXPECT_GT(results.size(), 0u);
    auto tolerance = std::chrono::seconds(1);
    for (const auto& r : results) {
        EXPECT_GE(r.timestamp, from - tolerance);
        EXPECT_LE(r.timestamp, to + tolerance);
    }
}

TEST_F(PostgresMetricsDatabaseTest, LimitClause) {
    PostgresMetricsDatabase db(db_url_, 2);

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

TEST_F(PostgresMetricsDatabaseTest, UpsertSession) {
    PostgresMetricsDatabase db(db_url_, 2);

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

    auto all = db.list_sessions(std::nullopt);
    int count = 0;
    for (const auto& s : all) {
        if (s.key == "upsert-test")
            ++count;
    }
    EXPECT_EQ(count, 1);
}

TEST_F(PostgresMetricsDatabaseTest, FinalLossRoundTripsThroughUpsertListAndArchive) {
    PostgresMetricsDatabase db(db_url_, 2);

    SessionRecord session;
    session.key = "final-loss-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    session.final_loss = 4.25f;
    session.final_validation_loss = 4.5f;
    db.upsert_session(session);

    auto rec = db.get_session("final-loss-test");
    ASSERT_TRUE(rec.has_value());
    EXPECT_FLOAT_EQ(rec->final_loss, 4.25f);
    EXPECT_FLOAT_EQ(rec->final_validation_loss, 4.5f);

    auto all = db.list_sessions(std::nullopt);
    auto it = std::find_if(all.begin(), all.end(),
                           [](const SessionRecord& s) { return s.key == "final-loss-test"; });
    ASSERT_NE(it, all.end());
    EXPECT_FLOAT_EQ(it->final_loss, 4.25f);
    EXPECT_FLOAT_EQ(it->final_validation_loss, 4.5f);

    db.archive_session("final-loss-test", "final-loss-test_archived_1");
    auto archived = db.get_session("final-loss-test_archived_1");
    ASSERT_TRUE(archived.has_value());
    EXPECT_FLOAT_EQ(archived->final_loss, 4.25f);
    EXPECT_FLOAT_EQ(archived->final_validation_loss, 4.5f);
}

// TD-065 regression test: a session row with NULL best_validation_loss/final_loss/
// final_validation_loss (exactly what every pre-migration row looked like, and what
// bootstrap_schema()'s ADD COLUMN IF NOT EXISTS with no DEFAULT still produces for any row
// nothing has re-upserted since) used to make list_sessions()/get_session() throw inside
// std::stof(""), which execute_with_retry() caught and treated as a fully failed operation —
// so list_sessions() silently returned an empty vector (dropping every other session in the
// same result too) and get_session() returned nullopt even for the exact key requested. Inserts
// the row via raw SQL (bypassing upsert_session(), which always supplies concrete values) to
// reproduce a genuinely NULL column, not just a zero one.
TEST_F(PostgresMetricsDatabaseTest, BestValidationLossNullReadsAsSentinelNotZero) {
    {
        PostgresMetricsDatabase bootstrap(db_url_, 2);  // create the schema first
    }

    PGresult* res = raw_query(
        "INSERT INTO sessions (key, session_id, is_training, created_at, last_update_at, "
        "total_epochs, total_samples, best_epoch) "
        "VALUES ('never-validated', 1, false, '2026-01-01T00:00:00+00', "
        "'2026-01-01T00:00:00+00', 3, 100, 2)");
    ASSERT_EQ(PQresultStatus(res), PGRES_COMMAND_OK) << PQerrorMessage(raw_conn_);
    PQclear(res);
    TearDownRawConn();

    PostgresMetricsDatabase db(db_url_, 2);

    auto rec = db.get_session("never-validated");
    ASSERT_TRUE(rec.has_value())
        << "get_session() must not silently fail on a NULL best_validation_loss/final_loss row";
    EXPECT_FLOAT_EQ(rec->best_validation_loss, std::numeric_limits<float>::max())
        << "best_validation_loss=NULL must read back as \"no data\", not 0.0";
    EXPECT_FLOAT_EQ(rec->final_loss, 0.0f);
    EXPECT_FLOAT_EQ(rec->final_validation_loss, 0.0f);

    auto all = db.list_sessions(std::nullopt);
    ASSERT_EQ(all.size(), 1u)
        << "the NULL-column row must not make list_sessions() drop every session in the result";
    EXPECT_FLOAT_EQ(all[0].best_validation_loss, std::numeric_limits<float>::max());
}

TEST_F(PostgresMetricsDatabaseTest, TD013DiagnosticsRoundTripThroughInsertAndQueryHistory) {
    PostgresMetricsDatabase db(db_url_, 2);

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

    // Second record with no TD-013 diagnostics this call — must read back at "not computed"
    // defaults, not NULL-guard failures or inherited values from the first row.
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

TEST_F(PostgresMetricsDatabaseTest, GradientVarianceHistoryInsertsUnthrottledPerCall) {
    PostgresMetricsDatabase db(db_url_, 2);

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

    PGresult* res = raw_query(
        "SELECT step, epoch, value FROM gradient_variance_history "
        "WHERE session_key = 'grad-var-test' ORDER BY step ASC");
    ASSERT_EQ(PQresultStatus(res), PGRES_TUPLES_OK) << PQerrorMessage(raw_conn_);
    int nrows = PQntuples(res);
    EXPECT_EQ(nrows, 5);
    for (int i = 0; i < nrows; ++i) {
        EXPECT_EQ(std::atoi(PQgetvalue(res, i, 0)), i + 1);
        EXPECT_EQ(std::atoi(PQgetvalue(res, i, 1)), 1);
        EXPECT_NEAR(std::atof(PQgetvalue(res, i, 2)), 0.1 * (i + 1), 0.01);
    }
    PQclear(res);
    TearDownRawConn();
}

TEST_F(PostgresMetricsDatabaseTest, MarkSessionEnded) {
    PostgresMetricsDatabase db(db_url_, 2);

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

TEST_F(PostgresMetricsDatabaseTest, ArchiveSessionMovesRowUnderNewKey) {
    PostgresMetricsDatabase db(db_url_, 2);

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

    EXPECT_FALSE(db.get_session("archive-test").has_value());

    auto archived = db.get_session("archive-test_archived_1");
    ASSERT_TRUE(archived.has_value());
    EXPECT_EQ(archived->label, "stale-run");
    EXPECT_EQ(archived->total_epochs, 3);
    EXPECT_FALSE(archived->is_training);
    EXPECT_TRUE(archived->ended_at.has_value());
}

TEST_F(PostgresMetricsDatabaseTest, ArchiveSessionMovesChildRows) {
    PostgresMetricsDatabase db(db_url_, 2);

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

    EXPECT_EQ(db.query_history("archive-child-test", std::nullopt, std::nullopt, 0).size(), 0u);
    EXPECT_EQ(
        db.query_history("archive-child-test_archived_1", std::nullopt, std::nullopt, 0).size(),
        5u);

    SessionRecord new_session;
    new_session.key = "archive-child-test";
    new_session.session_id = 2;
    new_session.is_training = true;
    new_session.created_at = std::chrono::system_clock::now();
    new_session.last_update_at = new_session.created_at;
    db.upsert_session(new_session);

    EXPECT_EQ(db.query_history("archive-child-test", std::nullopt, std::nullopt, 0).size(), 0u);
}

TEST_F(PostgresMetricsDatabaseTest, ArchiveSessionOfMissingKeyIsNoop) {
    PostgresMetricsDatabase db(db_url_, 2);
    EXPECT_NO_THROW(db.archive_session("does-not-exist", "does-not-exist_archived_1"));
    EXPECT_FALSE(db.get_session("does-not-exist_archived_1").has_value());
}

TEST_F(PostgresMetricsDatabaseTest, AbnormalSampleRoundTrip) {
    PostgresMetricsDatabase db(db_url_, 2);

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

    PGresult* res = raw_query(
        "SELECT epoch, sample_id, loss, grad_norm, reason, input_text, target_text "
        "FROM abnormal_samples WHERE session_key = 'abnormal-test'");
    ASSERT_EQ(PQresultStatus(res), PGRES_TUPLES_OK) << PQerrorMessage(raw_conn_);
    ASSERT_EQ(PQntuples(res), 1);
    EXPECT_EQ(std::atoi(PQgetvalue(res, 0, 0)), 3);
    EXPECT_EQ(std::atoi(PQgetvalue(res, 0, 1)), 42);
    EXPECT_NEAR(std::atof(PQgetvalue(res, 0, 2)), 15.5, 0.01);
    EXPECT_NEAR(std::atof(PQgetvalue(res, 0, 3)), 100.0, 0.01);
    EXPECT_STREQ(PQgetvalue(res, 0, 4), "loss_outlier");
    EXPECT_STREQ(PQgetvalue(res, 0, 5), "What is this?");
    EXPECT_STREQ(PQgetvalue(res, 0, 6), "This is a test.");
    PQclear(res);
    TearDownRawConn();
}

TEST_F(PostgresMetricsDatabaseTest, GenerationQualityRoundTrip) {
    PostgresMetricsDatabase db(db_url_, 2);

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

    // "rougeL" is a mixed-case column name — bootstrap_schema() double-quotes it precisely
    // because Postgres folds unquoted identifiers to lowercase; the same quoting is required here.
    PGresult* res = raw_query(
        "SELECT epoch, bleu1, bleu4, rouge1, \"rougeL\" FROM generation_quality "
        "WHERE session_key = 'genqual-test'");
    ASSERT_EQ(PQresultStatus(res), PGRES_TUPLES_OK) << PQerrorMessage(raw_conn_);
    ASSERT_EQ(PQntuples(res), 1);
    EXPECT_EQ(std::atoi(PQgetvalue(res, 0, 0)), 5);
    EXPECT_NEAR(std::atof(PQgetvalue(res, 0, 1)), 0.45, 0.01);
    EXPECT_NEAR(std::atof(PQgetvalue(res, 0, 2)), 0.15, 0.01);
    EXPECT_NEAR(std::atof(PQgetvalue(res, 0, 3)), 0.50, 0.01);
    EXPECT_NEAR(std::atof(PQgetvalue(res, 0, 4)), 0.40, 0.01);
    PQclear(res);
    TearDownRawConn();
}

TEST_F(PostgresMetricsDatabaseTest, ListSessionsFilterByTraining) {
    PostgresMetricsDatabase db(db_url_, 2);

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

TEST_F(PostgresMetricsDatabaseTest, FactoryCreatesPostgres) {
    auto db = MetricsDatabaseFactory::create("postgres", "", db_url_, 2);
    ASSERT_NE(db, nullptr);

    SessionRecord session;
    session.key = "factory-test";
    session.session_id = 1;
    session.is_training = true;
    session.created_at = std::chrono::system_clock::now();
    session.last_update_at = session.created_at;
    db->upsert_session(session);

    auto rec = db->get_session("factory-test");
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec->session_id, 1);
}

// Regression test for a real connection-pool deadlock found writing this suite: this test needs
// no live Postgres of its own (it deliberately points at one that doesn't exist), so it doesn't
// use db_url_/pg_available_ at all — SetUp() still runs (harmlessly creating and dropping an
// unused scratch database) since this fixture has no lighter-weight base to opt out into.
//
// acquire_connection() used to mark a pool slot in_use=true *before* attempting to (re)connect
// it, and left it that way forever if the connection failed — a caller that gets nullptr back
// never calls release_connection() (see execute_with_retry()'s `if (!conn) { ...; continue; }`),
// so a failed slot never opened up again. With RETRY_COUNT (3) >= a small pool_size, a single
// sustained connection outage burned through the *entire* pool within one execute_with_retry()
// call, and every acquire_connection() anywhere after that blocked in pool_cv_.wait() forever —
// a real client (any of chatbot_api_server/incremental_trainer/metrics_api_server configured
// with this backend) would need a full process restart to recover from what should have been a
// simple "retries exhausted, return false" failure.
//
// Verified via a bounded wait on a std::promise/future pair, deliberately NOT std::async: a
// future from std::async blocks in its own destructor until the task completes, which would
// just turn a real regression here into a different, equally-total hang the moment this test
// function returns — a detached std::thread has no such destructor and a promise's future has
// perfectly normal (non-blocking) destructor semantics whether or not it was ever fulfilled, so
// a still-broken pool leaves nothing behind but a harmless leaked thread that dies with the
// process. On the pre-fix code this reproduced as a genuine indefinite hang (confirmed directly
// via a separate, throwaway standalone repro, run as its own process under a shell `timeout`,
// before writing this test) — this asserts the operation actually returns at all within a
// generous bound, not that it returns quickly.
TEST_F(PostgresMetricsDatabaseTest, PoolDoesNotDeadlockOnSustainedConnectionFailure) {
    // No Postgres listens here — every connection attempt fails immediately, every time.
    auto db = std::make_shared<PostgresMetricsDatabase>(
        "host=/tmp port=1 dbname=nope user=nope connect_timeout=2", /*pool_size=*/2);

    SessionRecord rec;
    rec.key = "deadlock-check";
    rec.session_id = 1;
    rec.created_at = std::chrono::system_clock::now();
    rec.last_update_at = rec.created_at;

    auto call_bounded = [&](std::chrono::seconds bound) {
        auto promise = std::make_shared<std::promise<void>>();
        std::future<void> ready = promise->get_future();
        std::thread([db, rec, promise] {
            db->upsert_session(rec);
            promise->set_value();
        }).detach();
        return ready.wait_for(bound);
    };

    // First call alone is enough to exhaust a 2-slot pool at RETRY_COUNT=3 on the unfixed code.
    ASSERT_EQ(call_bounded(std::chrono::seconds(20)), std::future_status::ready)
        << "first call never returned — connection pool likely deadlocked";

    // A second call proves slots genuinely came back, not just that the first call happened to
    // grab a not-yet-poisoned slot.
    ASSERT_EQ(call_bounded(std::chrono::seconds(20)), std::future_status::ready)
        << "second call never returned — pool slots were not released after the first failure";
}

#endif  // ADAI_ENABLE_POSTGRES
