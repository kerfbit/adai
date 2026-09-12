// Tests for src/MetricsApiServerArgs.{hpp,cpp} — metrics_api_server's config-precedence
// resolution (file < persisted admin overrides < CLI) and argv parsing (TD-035), extracted from
// TrainingMetricsAPIServer.cpp so it's testable without a real MetricsSessionRegistry/HTTP
// server. TrainingMetricsAPI's own request-handling is already covered by
// TrainingMetricsAPIRoutesTests/TrainingMetricsAPILiveTests.

#include <gtest/gtest.h>
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
    file_config.metrics_max_live_sessions = 33;
    file_config.metrics_completed_ttl_seconds = 999;
    file_config.metrics_sweep_interval_seconds = 15;
    file_config.metrics_staleness_threshold_seconds = 20;
    file_config.metrics_enable_prometheus = true;
    file_config.metrics_api_allow_control = false;
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
    EXPECT_EQ(config.max_live_sessions, 33u);
    EXPECT_EQ(config.completed_ttl_seconds, 999);
    EXPECT_EQ(config.sweep_interval_seconds, 15);
    EXPECT_EQ(config.staleness_threshold_seconds, 20);
    EXPECT_TRUE(config.enable_prometheus);
    EXPECT_FALSE(config.allow_control);
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
        {"enable_prometheus", "true"},
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
