// Tests for src/MetricsApiServerArgs.{hpp,cpp} — metrics_api_server's config-precedence
// resolution (file < persisted admin overrides < CLI) and argv parsing (TD-035), extracted from
// TrainingMetricsAPIServer.cpp so it's testable without a real MetricsSessionRegistry/HTTP
// server. TrainingMetricsAPI's own request-handling is already covered by
// TrainingMetricsAPIRoutesTests/TrainingMetricsAPILiveTests.

#include <gtest/gtest.h>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "MetricsApiServerArgs.hpp"

using adai::apply_metrics_admin_overrides;
using adai::MetricsApiServerConfig;
using adai::parse_metrics_api_server_args;
using adai::seed_metrics_server_config_from_file;
using adai::ServiceConfig;

namespace {
std::vector<char*> make_argv(std::vector<std::string>& storage) {
    std::vector<char*> argv;
    for (auto& s : storage)
        argv.push_back(s.data());
    return argv;
}
}  // namespace

TEST(SeedMetricsServerConfigFromFile, CopiesEveryField) {
    ServiceConfig file_config;
    file_config.metrics_api_port = 9191;
    file_config.metrics_enable_persistence = false;
    file_config.metrics_file = "/custom/metrics.jsonl";
    file_config.metrics_summary_file = "/custom/summary.json";
    file_config.metrics_prometheus_file = "/custom/metrics.prom";
    file_config.metrics_persist_every_samples = 42;
    file_config.metrics_persist_every_seconds = 7;
    file_config.metrics_max_records_in_memory = 111;
    file_config.metrics_max_records_on_disk = 222;
    file_config.metrics_retention_days = 14;
    file_config.metrics_file_max_mb = 64;
    file_config.metrics_file_backups = 5;
    file_config.metrics_max_live_sessions = 33;
    file_config.metrics_completed_ttl_seconds = 999;
    file_config.metrics_sweep_interval_seconds = 15;
    file_config.metrics_staleness_threshold_seconds = 20;
    file_config.metrics_enable_prometheus = true;
    file_config.metrics_api_allow_control = false;
    file_config.metrics_api_cors_allowed_origins = "https://dash.example";
    file_config.name_service_url = "http://ns:9000";
    file_config.metrics_storage_backend = "postgres";
    file_config.metrics_db_path = "/custom/db.sqlite";
    file_config.metrics_db_url = "postgresql://x";
    file_config.metrics_db_pool_size = 9;

    auto config = seed_metrics_server_config_from_file(file_config);

    EXPECT_EQ(config.port, 9191);
    EXPECT_FALSE(config.enable_persistence);
    EXPECT_EQ(config.metrics_file, "/custom/metrics.jsonl");
    EXPECT_EQ(config.summary_file, "/custom/summary.json");
    EXPECT_EQ(config.prometheus_file, "/custom/metrics.prom");
    EXPECT_EQ(config.persist_every_samples, 42);
    EXPECT_EQ(config.persist_every_seconds, 7);
    EXPECT_EQ(config.max_records_in_memory, 111);
    EXPECT_EQ(config.max_records_on_disk, 222);
    EXPECT_EQ(config.retention_days, 14);
    EXPECT_EQ(config.metrics_file_max_mb, 64);
    EXPECT_EQ(config.metrics_file_backups, 5);
    EXPECT_EQ(config.max_live_sessions, 33u);
    EXPECT_EQ(config.completed_ttl_seconds, 999);
    EXPECT_EQ(config.sweep_interval_seconds, 15);
    EXPECT_EQ(config.staleness_threshold_seconds, 20);
    EXPECT_TRUE(config.enable_prometheus);
    EXPECT_FALSE(config.allow_control);
    EXPECT_EQ(config.cors_allowed_origins, "https://dash.example");
    EXPECT_EQ(config.name_service_url, "http://ns:9000");
    EXPECT_EQ(config.storage_backend, "postgres");
    EXPECT_EQ(config.db_path, "/custom/db.sqlite");
    EXPECT_EQ(config.db_url, "postgresql://x");
    EXPECT_EQ(config.db_pool_size, 9);
}

class ApplyMetricsAdminOverridesTest : public ::testing::Test {
   protected:
    MetricsApiServerConfig config;
};

TEST_F(ApplyMetricsAdminOverridesTest, EmptyOverridesLeaveConfigUntouched) {
    MetricsApiServerConfig before = config;
    apply_metrics_admin_overrides(config, {});
    EXPECT_EQ(config.max_live_sessions, before.max_live_sessions);
    EXPECT_EQ(config.completed_ttl_seconds, before.completed_ttl_seconds);
}

TEST_F(ApplyMetricsAdminOverridesTest, AppliesEveryDocumentedKey) {
    std::map<std::string, std::string> overrides = {
        {"max_live_sessions", "50"},        {"completed_ttl_seconds", "1200"},
        {"sweep_interval_seconds", "30"},   {"persist_every_samples", "10"},
        {"persist_every_seconds", "5"},     {"max_records_in_memory", "500"},
        {"max_records_on_disk", "5000"},    {"staleness_threshold_seconds", "45"},
        {"enable_prometheus", "true"},      {"retention_days", "30"},
        {"metrics_file_max_mb", "10"},      {"metrics_file_backups", "2"},
    };
    apply_metrics_admin_overrides(config, overrides);
    EXPECT_EQ(config.max_live_sessions, 50u);
    EXPECT_EQ(config.completed_ttl_seconds, 1200);
    EXPECT_EQ(config.sweep_interval_seconds, 30);
    EXPECT_EQ(config.persist_every_samples, 10);
    EXPECT_EQ(config.persist_every_seconds, 5);
    EXPECT_EQ(config.max_records_in_memory, 500);
    EXPECT_EQ(config.max_records_on_disk, 5000);
    EXPECT_EQ(config.staleness_threshold_seconds, 45);
    EXPECT_TRUE(config.enable_prometheus);
    EXPECT_EQ(config.retention_days, 30);
    EXPECT_EQ(config.metrics_file_max_mb, 10);
    EXPECT_EQ(config.metrics_file_backups, 2);
}

TEST_F(ApplyMetricsAdminOverridesTest, UnparseableNumericOverrideIsSkippedNotCrashed) {
    config.max_live_sessions = 16;
    apply_metrics_admin_overrides(config, {{"max_live_sessions", "not-a-number"}});
    EXPECT_EQ(config.max_live_sessions, 16u) << "bad override value must leave the field alone";
}

TEST_F(ApplyMetricsAdminOverridesTest, EnableePrometheusOnlyTrueOnLiteralStringTrue) {
    config.enable_prometheus = true;
    apply_metrics_admin_overrides(config, {{"enable_prometheus", "false"}});
    EXPECT_FALSE(config.enable_prometheus);

    apply_metrics_admin_overrides(config, {{"enable_prometheus", "garbage"}});
    EXPECT_FALSE(config.enable_prometheus) << "anything other than the literal \"true\" is false";
}

class ParseMetricsApiServerArgsTest : public ::testing::Test {
   protected:
    MetricsApiServerConfig config;
};

TEST_F(ParseMetricsApiServerArgsTest, NoArgsIsNotAnError) {
    std::vector<std::string> raw = {"metrics_api_server"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    EXPECT_FALSE(r.help);
    EXPECT_FALSE(r.error);
}

TEST_F(ParseMetricsApiServerArgsTest, HelpFlagShortCircuits) {
    std::vector<std::string> raw = {"metrics_api_server", "--port", "1234", "--help"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    EXPECT_TRUE(r.help);
}

TEST_F(ParseMetricsApiServerArgsTest, UnknownOptionIsAnError) {
    std::vector<std::string> raw = {"metrics_api_server", "--bogus"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    EXPECT_TRUE(r.error);
    EXPECT_NE(r.error_message.find("--bogus"), std::string::npos);
}

TEST_F(ParseMetricsApiServerArgsTest, ConfigFlagIsSkipped) {
    std::vector<std::string> raw = {"metrics_api_server", "--config", "/tmp/c.conf", "--port",
                                    "9090"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    EXPECT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(config.port, 9090);
}

TEST_F(ParseMetricsApiServerArgsTest, BooleanFlagsToggleCorrectly) {
    std::vector<std::string> raw = {"metrics_api_server", "--no-persistence",
                                    "--enable-prometheus", "--no-control", "--no-name-service"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error);
    EXPECT_FALSE(config.enable_persistence);
    EXPECT_TRUE(config.enable_prometheus);
    EXPECT_FALSE(config.allow_control);
    EXPECT_EQ(config.name_service_url, "");
}

// TD-526
TEST_F(ParseMetricsApiServerArgsTest, CorsAllowedOriginsFlagIsApplied) {
    std::vector<std::string> raw = {"metrics_api_server", "--cors-allowed-origins",
                                    "https://a.example,https://b.example"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(config.cors_allowed_origins, "https://a.example,https://b.example");
}

TEST_F(ParseMetricsApiServerArgsTest, StorageBackendAndDbFlagsAreApplied) {
    std::vector<std::string> raw = {
        "metrics_api_server", "--storage-backend", "postgres", "--db-path",  "/x.db",
        "--db-url",           "postgresql://y",    "--db-pool-size", "8",
    };
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error);
    EXPECT_EQ(config.storage_backend, "postgres");
    EXPECT_EQ(config.db_path, "/x.db");
    EXPECT_EQ(config.db_url, "postgresql://y");
    EXPECT_EQ(config.db_pool_size, 8);
}

TEST_F(ParseMetricsApiServerArgsTest, MaxLiveSessionsParsesAsUnsigned) {
    std::vector<std::string> raw = {"metrics_api_server", "--max-live-sessions", "64"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error);
    EXPECT_EQ(config.max_live_sessions, 64u);
}

TEST_F(ParseMetricsApiServerArgsTest, CliOverridesWhateverWasAlreadySet) {
    config.port = 8081;
    std::vector<std::string> raw = {"metrics_api_server", "--port", "1234"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error);
    EXPECT_EQ(config.port, 1234);
}

// TD-440: retention flags.
TEST_F(ParseMetricsApiServerArgsTest, RetentionFlagsAreApplied) {
    std::vector<std::string> raw = {"metrics_api_server",    "--retention-days", "0",
                                    "--metrics-file-max-mb", "25",               "--metrics-file-backups",
                                    "4"};
    auto argv = make_argv(raw);
    auto result =
        parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    EXPECT_FALSE(result.error) << result.error_message;
    EXPECT_EQ(config.retention_days, 0);
    EXPECT_EQ(config.metrics_file_max_mb, 25);
    EXPECT_EQ(config.metrics_file_backups, 4);
}

// ============================================================================
// TD-438: numbers are checked; a bad flag is an error
// ============================================================================

namespace {
adai::MetricsApiServerArgsResult parse(std::vector<std::string> raw, MetricsApiServerConfig& config) {
    raw.insert(raw.begin(), "metrics_api_server");
    auto argv = make_argv(raw);
    return parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
}

const std::vector<const char*> kNumericFlags = {
    "--port",          "--persist-samples",       "--persist-seconds",        "--max-memory-records",
    "--max-disk-records", "--retention-days",     "--metrics-file-max-mb",    "--metrics-file-backups",
    "--max-live-sessions", "--completed-ttl-seconds", "--sweep-interval-seconds",
    "--staleness-threshold-seconds", "--db-pool-size"};
}  // namespace

TEST_F(ParseMetricsApiServerArgsTest, EveryNumericFlagRefusesNonNumbersPartialsAndNegatives) {
    // atoi/strtoul: "abc" -> 0 (port 0 listened anywhere), "8081x" -> 8081, "1e3" -> 1, "-1" wrapped.
    for (const char* flag : kNumericFlags) {
        for (const char* value : {"abc", "8081x", "1e3", "-1", ""}) {
            MetricsApiServerConfig fresh;
            auto r = parse({flag, value}, fresh);
            EXPECT_TRUE(r.error) << flag << " " << value;
            EXPECT_NE(r.error_message.find(flag), std::string::npos) << r.error_message;
        }
    }
}

TEST_F(ParseMetricsApiServerArgsTest, OutOfRangeValuesAreRefused) {
    for (const auto& c : std::vector<std::pair<std::string, std::string>>{{"--port", "0"},
                                                                          {"--port", "99999"},
                                                                          {"--max-live-sessions", "0"},
                                                                          {"--max-memory-records", "0"},
                                                                          {"--staleness-threshold-seconds", "0"},
                                                                          {"--db-pool-size", "0"}}) {
        MetricsApiServerConfig fresh;
        auto r = parse({c.first, c.second}, fresh);
        EXPECT_TRUE(r.error) << c.first << " " << c.second;
    }
    auto r = parse({"--port", "99999"}, config);
    EXPECT_EQ(r.error_message, "Invalid value for --port: '99999' (expected 1-65535)");
    EXPECT_EQ(config.port, 8081);
}

TEST_F(ParseMetricsApiServerArgsTest, DocumentedZeroesAreAccepted) {
    auto r = parse({"--retention-days", "0", "--metrics-file-max-mb", "0", "--max-disk-records", "0",
                    "--sweep-interval-seconds", "0", "--completed-ttl-seconds", "0"},
                   config);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(config.sweep_interval_seconds, 0);
}

TEST_F(ParseMetricsApiServerArgsTest, AFlagMissingItsValueSaysSo) {
    for (const char* flag : kNumericFlags) {
        MetricsApiServerConfig fresh;
        EXPECT_EQ(parse({flag}, fresh).error_message, std::string("Missing value for ") + flag);
    }
    for (const char* flag : {"--config", "--metrics-file", "--db-url", "--storage-backend"}) {
        MetricsApiServerConfig fresh;
        EXPECT_EQ(parse({flag}, fresh).error_message, std::string("Missing value for ") + flag);
    }
    EXPECT_EQ(parse({"--db-path", "--port", "9000"}, config).error_message, "Missing value for --db-path");
}

TEST_F(ApplyMetricsAdminOverridesTest, OutOfRangeOrPartialSavedValuesAreSkipped) {
    // stoull("-1") wrapped to no session cap; stoi("50abc") applied 50; negative TTLs were applied.
    config.max_live_sessions = 16;
    config.completed_ttl_seconds = 3600;
    config.persist_every_samples = 100;
    config.staleness_threshold_seconds = 60;
    apply_metrics_admin_overrides(config, {{"max_live_sessions", "-1"},
                                           {"completed_ttl_seconds", "-5"},
                                           {"persist_every_samples", "50abc"},
                                           {"staleness_threshold_seconds", "0"}});
    EXPECT_EQ(config.max_live_sessions, 16u);
    EXPECT_EQ(config.completed_ttl_seconds, 3600);
    EXPECT_EQ(config.persist_every_samples, 100);
    EXPECT_EQ(config.staleness_threshold_seconds, 60);
}

// TD-439: --storage-backend used to accept any name; the server then ran file-only while its banner
// printed that name as the backend.
TEST_F(ParseMetricsApiServerArgsTest, StorageBackendNameIsCheckedInAnyCase) {
    std::vector<std::string> raw = {"metrics_api_server", "--storage-backend", "SQLite+File"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(config.storage_backend, "sqlite+file");

    std::vector<std::string> typo = {"metrics_api_server", "--storage-backend", "postgress"};
    auto argv2 = make_argv(typo);
    r = parse_metrics_api_server_args(static_cast<int>(argv2.size()), argv2.data(), config);
    EXPECT_TRUE(r.error);
    EXPECT_NE(r.error_message.find("postgress"), std::string::npos) << r.error_message;
    EXPECT_EQ(config.storage_backend, "sqlite+file");  // unchanged
}

TEST(CheckMetricsStorageBackend, PostgresNeedsABuildWithIt) {
    EXPECT_EQ(adai::supported_metrics_storage_backends(false),
              (std::vector<std::string>{"sqlite+file", "sqlite", "file"}));
    EXPECT_EQ(adai::supported_metrics_storage_backends(true),
              (std::vector<std::string>{"sqlite+file", "sqlite", "postgres+file", "postgres", "file"}));

    for (const char* ok : {"sqlite+file", "sqlite", "file"}) {
        EXPECT_FALSE(adai::check_metrics_storage_backend(ok, false)) << ok;
        EXPECT_FALSE(adai::check_metrics_storage_backend(ok, true)) << ok;
    }
    for (const char* pg : {"postgres", "postgres+file"}) {
        EXPECT_FALSE(adai::check_metrics_storage_backend(pg, true)) << pg;
        const auto error = adai::check_metrics_storage_backend(pg, false);
        ASSERT_TRUE(error) << pg;
        EXPECT_NE(error->find("ENABLE_POSTGRES_METRICS"), std::string::npos) << *error;
    }
    const auto unknown = adai::check_metrics_storage_backend("postgress", true);
    ASSERT_TRUE(unknown);
    EXPECT_NE(unknown->find("Unknown storage backend 'postgress'"), std::string::npos) << *unknown;
}

// TD-439: daemon_config.db went next to the file-configured db_path, ignoring --db-path.
TEST_F(ParseMetricsApiServerArgsTest, AdminStoreFollowsTheFinalDbPath) {
    config.db_path = "/opt/adai/training_sessions/metrics.db";  // from the config file
    std::vector<std::string> raw = {"metrics_api_server", "--db-path", "/var/lib/adai-metrics/metrics.db"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(adai::metrics_admin_store_dir(config), "/var/lib/adai-metrics");

    MetricsApiServerConfig bare;
    bare.db_path = "metrics.db";
    EXPECT_EQ(adai::metrics_admin_store_dir(bare), ".");
}

// TD-439: flags beat saved admin settings, so a flag in the unit's ExecStart silently undid every
// PUT /admin/config on restart; now each replaced value is reported.
TEST_F(ParseMetricsApiServerArgsTest, FlagsThatReplaceSavedAdminSettingsAreReported) {
    const std::map<std::string, std::string> saved = {
        {"retention_days", "30"}, {"max_live_sessions", "8"}, {"enable_prometheus", "false"},
        {"persist_every_seconds", "15"}};
    apply_metrics_admin_overrides(config, saved);
    std::vector<std::string> raw = {"metrics_api_server",  "--retention-days", "7", "--port", "9090",
                                    "--max-live-sessions", "8",                "--enable-prometheus"};
    auto argv = make_argv(raw);
    auto r = parse_metrics_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(r.admin_keys_from_flags,
              (std::set<std::string>{"retention_days", "max_live_sessions", "enable_prometheus"}));

    const auto warnings = adai::flag_overridden_admin_settings(config, saved, r.admin_keys_from_flags);
    ASSERT_EQ(warnings.size(), 2u);  // max_live_sessions is the same value; --port isn't admin-mutable
    EXPECT_NE(warnings[0].find("--retention-days 7 replaces the saved admin setting retention_days=30"),
              std::string::npos)
        << warnings[0];
    EXPECT_NE(warnings[1].find("--enable-prometheus replaces the saved admin setting enable_prometheus=false"),
              std::string::npos)
        << warnings[1];

    EXPECT_TRUE(adai::flag_overridden_admin_settings(config, {}, r.admin_keys_from_flags).empty());
}
