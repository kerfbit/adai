// Tests for src/RegistryServerArgs.{hpp,cpp} — registry_server's argv parsing and
// config-precedence resolution (TD-035, narrower scope — see this item's own note in
// TECHNICAL_DEBT.md / the doc comment on RegistryServerArgs.hpp), extracted from
// RegistryServer.cpp so that piece is testable without a real HTTP server. The actual request
// handlers are already covered live by DatasetRegistryLiveTests/TrainerAdminAPITests/
// RegistryFtpConfinementTests.

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "RegistryServerArgs.hpp"

using adai::parse_registry_server_args;
using adai::resolve_registry_server_config;
using adai::RegistryServerArgs;
using adai::ServiceConfig;

namespace {
std::vector<char*> make_argv(std::vector<std::string>& storage) {
    std::vector<char*> argv;
    for (auto& s : storage)
        argv.push_back(s.data());
    return argv;
}
}  // namespace

TEST(ParseRegistryServerArgs, NoArgsLeavesEverythingAtDefaults) {
    std::vector<std::string> raw = {"registry_server"};
    auto argv = make_argv(raw);
    auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_FALSE(r.help);
    EXPECT_FALSE(r.config_path.has_value());
    EXPECT_FALSE(r.port.has_value());
    EXPECT_FALSE(r.data_dir.has_value());
    EXPECT_FALSE(r.ftp_enabled);
    EXPECT_EQ(r.ftp_port, 2121);
    EXPECT_EQ(r.ftp_pasv_min, 50000);
    EXPECT_EQ(r.ftp_pasv_max, 50099);
    EXPECT_FALSE(r.ftps_enabled);
    EXPECT_FALSE(r.ftp_ttl_minutes.has_value());
    EXPECT_FALSE(r.ftp_max_sessions.has_value());
    EXPECT_FALSE(r.admin_enabled.has_value());
}

// TD-494: claim lease, upload cap and fetch concurrency — defaults, file, and CLI over file.
TEST(RegistryServerTd494Settings, DefaultsFileAndCliPrecedence) {
    std::vector<std::string> none = {"registry_server"};
    auto argv0 = make_argv(none);
    ServiceConfig file;
    auto cfg = resolve_registry_server_config(
        parse_registry_server_args(static_cast<int>(argv0.size()), argv0.data()), file, {});
    EXPECT_EQ(cfg.claim_lease_minutes, 30);
    EXPECT_EQ(cfg.max_upload_mb, 2048);
    EXPECT_EQ(cfg.max_concurrent_fetches, 2);

    file.registry_claim_lease_minutes = 0;
    file.registry_max_upload_mb = 512;
    file.registry_max_concurrent_fetches = 4;
    cfg = resolve_registry_server_config(
        parse_registry_server_args(static_cast<int>(argv0.size()), argv0.data()), file, {});
    EXPECT_EQ(cfg.claim_lease_minutes, 0);
    EXPECT_EQ(cfg.max_upload_mb, 512);
    EXPECT_EQ(cfg.max_concurrent_fetches, 4);

    std::vector<std::string> raw = {"registry_server", "--claim-lease-minutes", "45",
                                    "--max-upload-mb", "100", "--max-concurrent-fetches", "1"};
    auto argv = make_argv(raw);
    cfg = resolve_registry_server_config(
        parse_registry_server_args(static_cast<int>(argv.size()), argv.data()), file, {});
    EXPECT_EQ(cfg.claim_lease_minutes, 45);
    EXPECT_EQ(cfg.max_upload_mb, 100);
    EXPECT_EQ(cfg.max_concurrent_fetches, 1);

    std::vector<std::string> bad = {"registry_server", "--claim-lease-minutes", "-5",
                                    "--max-upload-mb", "0", "--max-concurrent-fetches", "0"};
    auto argvb = make_argv(bad);
    cfg = resolve_registry_server_config(
        parse_registry_server_args(static_cast<int>(argvb.size()), argvb.data()), file, {});
    EXPECT_EQ(cfg.claim_lease_minutes, 0) << "negative lease clamps to 'never expire'";
    EXPECT_EQ(cfg.max_upload_mb, 1);
    EXPECT_EQ(cfg.max_concurrent_fetches, 1);
}

TEST(ParseRegistryServerArgs, HelpFlagShortCircuits) {
    std::vector<std::string> raw = {"registry_server", "--port", "9999", "--help"};
    auto argv = make_argv(raw);
    auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(r.help);
}

TEST(ParseRegistryServerArgs, ShortDashPForPort) {
    std::vector<std::string> raw = {"registry_server", "-p", "1234"};
    auto argv = make_argv(raw);
    auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
    ASSERT_TRUE(r.port.has_value());
    EXPECT_EQ(*r.port, 1234);
}

TEST(ParseRegistryServerArgs, UnrecognizedArgumentIsSilentlyIgnored) {
    // Regression-pin for a real (if minor) quirk found extracting this: unlike every sibling
    // daemon's argv parser in this codebase, registry_server's has never reported "Unknown
    // argument" for a bad flag — it just silently ignores it and continues. Preserved exactly
    // rather than "fixed" as a drive-by, since a narrow args-extraction pass shouldn't also
    // change behavior; see TECHNICAL_DEBT_RESOLVED.md for the full note.
    std::vector<std::string> raw = {"registry_server", "--totally-bogus", "--port", "1234"};
    auto argv = make_argv(raw);
    auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_FALSE(r.help);
    ASSERT_TRUE(r.port.has_value());
    EXPECT_EQ(*r.port, 1234) << "parsing must continue past the unrecognized flag";
}

TEST(ParseRegistryServerArgs, FtpFlagsAllApply) {
    std::vector<std::string> raw = {
        "registry_server", "--ftp-enabled", "--ftp-port",  "3131",
        "--ftp-ip",        "1.2.3.4",       "--ftp-pasv-min", "60000",
        "--ftp-pasv-max",  "60099",         "--ftp-secret", "s3cr3t",
        "--ftps",          "--ftp-cert",    "/c.pem",       "--ftp-key",
        "/k.pem",
    };
    auto argv = make_argv(raw);
    auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(r.ftp_enabled);
    EXPECT_EQ(r.ftp_port, 3131);
    EXPECT_EQ(r.ftp_advertise_ip, "1.2.3.4");
    EXPECT_EQ(r.ftp_pasv_min, 60000);
    EXPECT_EQ(r.ftp_pasv_max, 60099);
    EXPECT_EQ(r.ftp_server_secret, "s3cr3t");
    EXPECT_TRUE(r.ftps_enabled);
    EXPECT_EQ(r.ftp_cert_file, "/c.pem");
    EXPECT_EQ(r.ftp_key_file, "/k.pem");
}

TEST(ParseRegistryServerArgs, FtpTtlAndMaxSessionsAreOptional) {
    std::vector<std::string> raw = {"registry_server", "--ftp-ttl", "60", "--ftp-max-sessions",
                                    "10"};
    auto argv = make_argv(raw);
    auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
    ASSERT_TRUE(r.ftp_ttl_minutes.has_value());
    EXPECT_EQ(*r.ftp_ttl_minutes, 60);
    ASSERT_TRUE(r.ftp_max_sessions.has_value());
    EXPECT_EQ(*r.ftp_max_sessions, 10);
}

TEST(ParseRegistryServerArgs, AdminEnabledParsesLenientBooleans) {
    for (const std::string& v : {"true", "1", "yes", "on", "TRUE", "On"}) {
        std::vector<std::string> raw = {"registry_server", "--admin-enabled", v};
        auto argv = make_argv(raw);
        auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(r.admin_enabled.has_value()) << "value: " << v;
        EXPECT_TRUE(*r.admin_enabled) << "value: " << v;
    }
    for (const std::string& v : {"false", "0", "no", "off", "garbage"}) {
        std::vector<std::string> raw = {"registry_server", "--admin-enabled", v};
        auto argv = make_argv(raw);
        auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(r.admin_enabled.has_value()) << "value: " << v;
        EXPECT_FALSE(*r.admin_enabled) << "value: " << v;
    }
}

class ResolveRegistryServerConfigTest : public ::testing::Test {
   protected:
    ServiceConfig file_config;
    std::map<std::string, std::string> no_overrides;

    void SetUp() override {
        file_config.registry_listen_port = 8082;
        file_config.registry_data_dir = "registry_sessions";
        file_config.ftp_token_ttl_minutes = 30;
        file_config.ftp_max_sessions_per_run = 4;
    }
};

TEST_F(ResolveRegistryServerConfigTest, FileConfigAloneIsTheBaseline) {
    RegistryServerArgs args;
    auto eff = resolve_registry_server_config(args, file_config, no_overrides);
    EXPECT_EQ(eff.port, 8082);
    EXPECT_EQ(eff.data_dir, "registry_sessions");
    EXPECT_EQ(eff.ftp_ttl_minutes, 30);
    EXPECT_EQ(eff.ftp_max_sessions, 4);
    EXPECT_TRUE(eff.admin_enabled);
}

TEST_F(ResolveRegistryServerConfigTest, AdminOverridesBeatFileConfig) {
    std::map<std::string, std::string> overrides = {
        {"ftp_token_ttl_minutes", "90"},
        {"ftp_max_sessions_per_run", "20"},
    };
    RegistryServerArgs args;
    auto eff = resolve_registry_server_config(args, file_config, overrides);
    EXPECT_EQ(eff.ftp_ttl_minutes, 90);
    EXPECT_EQ(eff.ftp_max_sessions, 20);
}

TEST_F(ResolveRegistryServerConfigTest, UnparseableOverrideIsSkippedNotCrashed) {
    std::map<std::string, std::string> overrides = {{"ftp_token_ttl_minutes", "not-a-number"}};
    RegistryServerArgs args;
    auto eff = resolve_registry_server_config(args, file_config, overrides);
    EXPECT_EQ(eff.ftp_ttl_minutes, 30) << "bad override value must leave the field at its prior value";
}

TEST_F(ResolveRegistryServerConfigTest, CliFlagsBeatAdminOverridesAndFileConfig) {
    std::map<std::string, std::string> overrides = {
        {"ftp_token_ttl_minutes", "90"},
        {"ftp_max_sessions_per_run", "20"},
    };
    RegistryServerArgs args;
    args.ftp_ttl_minutes = 15;
    args.ftp_max_sessions = 2;
    args.admin_enabled = false;
    args.port = 12345;
    args.data_dir = "/custom/dir";

    auto eff = resolve_registry_server_config(args, file_config, overrides);
    EXPECT_EQ(eff.ftp_ttl_minutes, 15);
    EXPECT_EQ(eff.ftp_max_sessions, 2);
    EXPECT_FALSE(eff.admin_enabled);
    EXPECT_EQ(eff.port, 12345);
    EXPECT_EQ(eff.data_dir, "/custom/dir");
}
