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
using adai::parse_registry_admin_config;
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
    // TD-284: FTP flags are optional, so the config file applies when they aren't passed.
    EXPECT_FALSE(r.ftp_enabled.has_value());
    EXPECT_FALSE(r.ftp_port.has_value());
    EXPECT_FALSE(r.ftp_pasv_min.has_value());
    EXPECT_FALSE(r.ftp_pasv_max.has_value());
    EXPECT_FALSE(r.ftps_enabled.has_value());
    EXPECT_FALSE(r.ftp_server_secret.has_value());
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

    // TD-488: out-of-range flags are errors now (they used to be clamped).
    for (const auto& bad : std::vector<std::vector<std::string>>{{"registry_server", "--claim-lease-minutes", "-5"},
                                                                 {"registry_server", "--max-upload-mb", "0"},
                                                                 {"registry_server", "--max-concurrent-fetches", "0"}}) {
        auto copy = bad;
        auto argvb = make_argv(copy);
        EXPECT_TRUE(parse_registry_server_args(static_cast<int>(argvb.size()), argvb.data()).error) << bad[1];
    }
}

TEST(ParseRegistryServerArgs, HelpFlagShortCircuits) {
    std::vector<std::string> raw = {"registry_server", "--port", "9999", "--help"};
    auto argv = make_argv(raw);
    auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(r.help);
}

// TD-489: --help used to count only if no bad argument came before it; a mistake earlier on the
// line printed an error instead of the help text asked for.
TEST(ParseRegistryServerArgs, HelpWinsOverAnEarlierBadArgument) {
    for (const std::vector<std::string>& raw_const : std::vector<std::vector<std::string>>{
             {"registry_server", "--port", "abc", "--help"},
             {"registry_server", "--portt", "8091", "-h"},
             {"registry_server", "--admin-enabled", "--help"}}) {
        auto raw = raw_const;
        auto argv = make_argv(raw);
        auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_TRUE(r.help) << raw[1];
        EXPECT_FALSE(r.error) << raw[1] << ": " << r.error_message;
    }
}

TEST(ParseRegistryServerArgs, ShortDashPForPort) {
    std::vector<std::string> raw = {"registry_server", "-p", "1234"};
    auto argv = make_argv(raw);
    auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
    ASSERT_TRUE(r.port.has_value());
    EXPECT_EQ(*r.port, 1234);
}

// TD-488: an unknown argument used to be ignored, so --portt 8091 started on 8082.
TEST(ParseRegistryServerArgs, AnUnknownArgumentIsAnError) {
    std::vector<std::string> raw = {"registry_server", "--portt", "8091"};
    auto argv = make_argv(raw);
    auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(r.error);
    EXPECT_EQ(r.error_message, "Unknown argument: --portt");
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
    EXPECT_EQ(r.ftp_enabled, true);
    EXPECT_EQ(r.ftp_port, 3131);
    EXPECT_EQ(r.ftp_advertise_ip, "1.2.3.4");
    EXPECT_EQ(r.ftp_pasv_min, 60000);
    EXPECT_EQ(r.ftp_pasv_max, 60099);
    EXPECT_EQ(r.ftp_server_secret, "s3cr3t");
    EXPECT_EQ(r.ftps_enabled, true);
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

TEST(ParseRegistryServerArgs, AdminEnabledTakesTheSharedBooleanWords) {
    for (const std::string& v : {"true", "1", "yes", "on", "TRUE", "On", "enabled"}) {
        std::vector<std::string> raw = {"registry_server", "--admin-enabled", v};
        auto argv = make_argv(raw);
        auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(r.admin_enabled.has_value()) << "value: " << v;
        EXPECT_TRUE(*r.admin_enabled) << "value: " << v;
    }
    for (const std::string& v : {"false", "0", "no", "off", "Disabled"}) {
        std::vector<std::string> raw = {"registry_server", "--admin-enabled", v};
        auto argv = make_argv(raw);
        auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(r.admin_enabled.has_value()) << "value: " << v;
        EXPECT_FALSE(*r.admin_enabled) << "value: " << v;
    }
    // TD-488: anything else is an error (it used to mean false), like mns_server since TD-456.
    std::vector<std::string> raw = {"registry_server", "--admin-enabled", "garbage"};
    auto argv = make_argv(raw);
    EXPECT_TRUE(parse_registry_server_args(static_cast<int>(argv.size()), argv.data()).error);
}

// TD-488: bare std::stoi aborted on "abc" (core dump), took "8090x" as 8090, and bound 34463 for 99999.
TEST(ParseRegistryServerArgs, EveryNumericFlagIsAWholeNumberInRange) {
    for (const char* flag : {"--port", "-p", "--ftp-port", "--ftp-pasv-min", "--ftp-pasv-max", "--ftp-ttl",
                             "--ftp-max-sessions", "--claim-lease-minutes", "--max-upload-mb",
                             "--max-concurrent-fetches"}) {
        for (const char* value : {"abc", "8090x", "1e3", "-1", ""}) {
            std::vector<std::string> raw = {"registry_server", flag, value};
            auto argv = make_argv(raw);
            auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
            EXPECT_TRUE(r.error) << flag << " " << value;
            EXPECT_NE(r.error_message.find(flag), std::string::npos) << r.error_message;
        }
    }
    for (const auto& c : std::vector<std::pair<std::string, std::string>>{
             {"--port", "99999"}, {"--port", "0"}, {"--ftp-port", "70000"}, {"--ftp-ttl", "0"}}) {
        std::vector<std::string> raw = {"registry_server", c.first, c.second};
        auto argv = make_argv(raw);
        EXPECT_TRUE(parse_registry_server_args(static_cast<int>(argv.size()), argv.data()).error)
            << c.first << " " << c.second;
    }
    // 0 means unlimited sessions / claims that never expire.
    std::vector<std::string> raw = {"registry_server", "--ftp-max-sessions", "0", "--claim-lease-minutes", "0"};
    auto argv = make_argv(raw);
    auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(r.ftp_max_sessions, 0);
}

// TD-488: a value flag with no value was dropped; a bare --admin-enabled left PUT /admin/config open.
TEST(ParseRegistryServerArgs, AFlagMissingItsValueSaysSo) {
    for (const char* flag : {"--config", "--port", "-p", "--data-dir", "--ftp-port", "--ftp-ip", "--ftp-pasv-min",
                             "--ftp-pasv-max", "--ftp-ttl", "--ftp-secret", "--ftp-cert", "--ftp-key",
                             "--claim-lease-minutes", "--max-upload-mb", "--max-concurrent-fetches",
                             "--ftp-max-sessions", "--admin-enabled"}) {
        std::vector<std::string> raw = {"registry_server", flag};
        auto argv = make_argv(raw);
        auto r = parse_registry_server_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_TRUE(r.error) << flag;
        EXPECT_EQ(r.error_message, std::string("Missing value for ") + flag);
    }
    std::vector<std::string> raw = {"registry_server", "--ftp-secret", "--ftps"};
    auto argv = make_argv(raw);
    EXPECT_EQ(parse_registry_server_args(static_cast<int>(argv.size()), argv.data()).error_message,
              "Missing value for --ftp-secret");
}

TEST(ParseRegistryServerArgs, PasvMinAbovePasvMaxIsAnError) {
    std::vector<std::string> raw = {"registry_server", "--ftp-pasv-min", "60100", "--ftp-pasv-max", "60000"};
    auto argv = make_argv(raw);
    EXPECT_TRUE(parse_registry_server_args(static_cast<int>(argv.size()), argv.data()).error);
}

class ResolveRegistryServerConfigTest : public ::testing::Test {
   protected:
    ServiceConfig file_config;
    std::map<std::string, std::string> no_overrides;
    RegistryServerArgs args;  // no flags; tests that need some set their own

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

// ============================================================================
// TD-284: FTP listener settings come from the config file, with CLI flags overriding
// ============================================================================

namespace {
bool has_warning(const adai::RegistryServerEffectiveConfig& eff, const std::string& text) {
    for (const auto& w : eff.warnings)
        if (w.find(text) != std::string::npos) return true;
    return false;
}
}  // namespace

TEST_F(ResolveRegistryServerConfigTest, FtpSettingsComeFromTheConfigFile) {
    // These used to be CLI-only: FTPS_ENABLED and a real secret in the file did nothing.
    file_config.ftp_server_enabled = true;
    file_config.ftp_server_port = 3121;
    file_config.ftp_advertise_ip = "10.0.0.5";
    file_config.ftp_pasv_port_min = 51000;
    file_config.ftp_pasv_port_max = 51009;
    file_config.ftp_data_server_secret = "real-secret";
    file_config.ftps_enabled = true;
    file_config.ftp_cert_file = "/etc/adai/ftp.crt";
    file_config.ftp_key_file = "/etc/adai/ftp.key";
    const auto eff = resolve_registry_server_config(args, file_config, no_overrides);
    EXPECT_TRUE(eff.ftp_enabled);
    EXPECT_EQ(eff.ftp_port, 3121);
    EXPECT_EQ(eff.ftp_advertise_ip, "10.0.0.5");
    EXPECT_EQ(eff.ftp_pasv_min, 51000);
    EXPECT_EQ(eff.ftp_pasv_max, 51009);
    EXPECT_EQ(eff.ftp_server_secret, "real-secret");
    EXPECT_TRUE(eff.ftps_enabled);
    EXPECT_EQ(eff.ftp_cert_file, "/etc/adai/ftp.crt");
    EXPECT_EQ(eff.ftp_key_file, "/etc/adai/ftp.key");
    EXPECT_TRUE(eff.warnings.empty());
}

TEST_F(ResolveRegistryServerConfigTest, FtpCliFlagsOverrideTheFileAndSaySo) {
    file_config.ftps_enabled = true;
    file_config.ftp_server_port = 3121;
    args.ftps_enabled = false;
    args.ftp_port = 4121;
    const auto eff = resolve_registry_server_config(args, file_config, no_overrides);
    EXPECT_FALSE(eff.ftps_enabled);
    EXPECT_EQ(eff.ftp_port, 4121);
    EXPECT_TRUE(has_warning(eff, "--ftps overrides FTPS_ENABLED"));
    EXPECT_TRUE(has_warning(eff, "--ftp-port overrides FTP_SERVER_PORT"));
}

TEST_F(ResolveRegistryServerConfigTest, ThePlaceholderSecretIsIgnored) {
    // A publicly known HMAC key would let anyone compute FTP token passwords.
    file_config.ftp_data_server_secret = "change-me-in-production";
    const auto eff = resolve_registry_server_config(args, file_config, no_overrides);
    EXPECT_EQ(eff.ftp_server_secret, "");
    EXPECT_TRUE(has_warning(eff, "placeholder"));
}

TEST_F(ResolveRegistryServerConfigTest, PlaintextFtpOnANetworkAddressWarns) {
    file_config.ftp_server_enabled = true;
    file_config.ftp_advertise_ip = "192.168.1.16";
    EXPECT_TRUE(has_warning(resolve_registry_server_config(args, file_config, no_overrides), "without FTPS"));
    file_config.ftp_advertise_ip = "127.0.0.1";
    EXPECT_FALSE(has_warning(resolve_registry_server_config(args, file_config, no_overrides), "without FTPS"));
    file_config.ftp_advertise_ip = "192.168.1.16";
    file_config.ftps_enabled = true;
    EXPECT_FALSE(has_warning(resolve_registry_server_config(args, file_config, no_overrides), "without FTPS"));
}

// TD-488: a PASV range emptied by the file and a flag together, and bad saved overrides.
TEST_F(ResolveRegistryServerConfigTest, AnEmptyPasvRangeFromFileAndFlagIsAnError) {
    file_config.ftp_pasv_port_min = 50000;
    file_config.ftp_pasv_port_max = 50099;
    args.ftp_pasv_min = 51000;
    const auto cfg = resolve_registry_server_config(args, file_config, no_overrides);
    ASSERT_EQ(cfg.errors.size(), 1u);
    EXPECT_NE(cfg.errors[0].find("51000"), std::string::npos) << cfg.errors[0];
}

TEST_F(ResolveRegistryServerConfigTest, BadSavedOverridesAreSkippedWithAWarning) {
    const auto cfg = resolve_registry_server_config(
        args, file_config, {{"ftp_token_ttl_minutes", "0"}, {"ftp_max_sessions_per_run", "50abc"}});
    EXPECT_EQ(cfg.ftp_ttl_minutes, 30);
    EXPECT_EQ(cfg.ftp_max_sessions, 4);
    EXPECT_EQ(cfg.override_warnings.size(), 2u);
}

// TD-489: REGISTRY_ADMIN_ENABLED in config.registry.conf sets the default; --admin-enabled overrides
// it. It used to be the flag, else true, with no way to close the endpoint from the file.
TEST_F(ResolveRegistryServerConfigTest, AdminEnabledComesFromTheConfigUnlessTheFlagIsGiven) {
    file_config.registry_admin_enabled = false;
    EXPECT_FALSE(resolve_registry_server_config(args, file_config, no_overrides).admin_enabled);
    args.admin_enabled = true;
    EXPECT_TRUE(resolve_registry_server_config(args, file_config, no_overrides).admin_enabled);
    file_config.registry_admin_enabled = true;
    args.admin_enabled = false;
    EXPECT_FALSE(resolve_registry_server_config(args, file_config, no_overrides).admin_enabled);
    args.admin_enabled.reset();
    EXPECT_TRUE(resolve_registry_server_config(args, file_config, no_overrides).admin_enabled);
}

// TD-489: with FTP disabled, an FTP flag on the command line did nothing, silently.
TEST_F(ResolveRegistryServerConfigTest, FtpFlagsWithFtpDisabledWarn) {
    args.ftps_enabled = true;
    args.ftp_cert_file = "/etc/adai/cert.pem";
    args.ftp_port = 3121;
    args.ftp_ttl_minutes = 10;
    auto eff = resolve_registry_server_config(args, file_config, no_overrides);
    EXPECT_FALSE(eff.ftp_enabled);
    for (const std::string flag : {"--ftps", "--ftp-cert", "--ftp-port", "--ftp-ttl"}) {
        EXPECT_TRUE(has_warning(eff, flag + " has no effect: FTP is disabled")) << flag;
    }
    EXPECT_FALSE(has_warning(eff, "--ftp-key has no effect"));

    // Enabled from the file or the flag: no such warning.
    file_config.ftp_server_enabled = true;
    eff = resolve_registry_server_config(args, file_config, no_overrides);
    EXPECT_FALSE(has_warning(eff, "has no effect"));
    file_config.ftp_server_enabled = false;
    args.ftp_enabled = true;
    EXPECT_FALSE(has_warning(resolve_registry_server_config(args, file_config, no_overrides), "has no effect"));

    // File values alone (config.registry.conf ships the FTP keys with FTP off) don't warn.
    RegistryServerArgs no_flags;
    file_config.ftps_enabled = true;
    file_config.ftp_server_port = 3121;
    EXPECT_FALSE(has_warning(resolve_registry_server_config(no_flags, file_config, no_overrides), "has no effect"));
}


// TD-306: PUT /admin/config bodies are checked as a whole against registry_admin_int_settings().
// Values used to be applied and persisted unchecked (a TTL of -5 issued already-expired FTP tokens).
TEST(ParseRegistryAdminConfig, ValidValuesAreAccepted) {
    auto u = parse_registry_admin_config(R"({"ftp_token_ttl_minutes" : 10, "ftp_max_sessions_per_run": 0})");
    EXPECT_EQ(u.error, "");
    EXPECT_EQ(u.ftp_token_ttl_minutes, 10);
    EXPECT_EQ(u.ftp_max_sessions_per_run, 0) << "0 means unlimited";

    u = parse_registry_admin_config(R"({"ftp_max_sessions_per_run": 8})");
    EXPECT_EQ(u.error, "");
    EXPECT_FALSE(u.ftp_token_ttl_minutes.has_value());
    EXPECT_EQ(u.ftp_max_sessions_per_run, 8);
}

TEST(ParseRegistryAdminConfig, OutOfRangeOrNonIntegerValuesAreRejectedWhole) {
    for (const char* body : {R"({"ftp_token_ttl_minutes": -5})", R"({"ftp_token_ttl_minutes": 0})",
                             R"({"ftp_max_sessions_per_run": -1})", R"({"ftp_token_ttl_minutes": "10"})",
                             R"({"ftp_token_ttl_minutes": 2.5})", R"({"ftp_token_ttl_minutes": 3000000000})",
                             R"({"ftp_max_sessions_per_run": 2, "ftp_token_ttl_minutes": -1})"}) {
        const auto u = parse_registry_admin_config(body);
        EXPECT_NE(u.error, "") << body;
        EXPECT_FALSE(u.ftp_token_ttl_minutes.has_value()) << body;
        EXPECT_FALSE(u.ftp_max_sessions_per_run.has_value()) << body << ": nothing applies when any key is bad";
    }
    EXPECT_NE(parse_registry_admin_config(R"({"ftp_token_ttl_minutes": -5})").error.find(">= 1"), std::string::npos);
}

TEST(ParseRegistryAdminConfig, ImmutableUnknownAndMalformedBodiesAreRejected) {
    EXPECT_NE(parse_registry_admin_config(R"({"port": 9000, "ftp_token_ttl_minutes": 10})").error.find("immutable"),
              std::string::npos);
    EXPECT_NE(parse_registry_admin_config(R"({"ftp_ttl": 10})").error.find("no recognized"), std::string::npos);
    EXPECT_NE(parse_registry_admin_config("not json").error, "");
    EXPECT_NE(parse_registry_admin_config("[1]").error, "");
}

// TD-306: a saved override is held to the same range as the PUT.
TEST_F(ResolveRegistryServerConfigTest, SavedOverridesUseTheAdminRanges) {
    const auto cfg = resolve_registry_server_config(
        args, file_config, {{"ftp_token_ttl_minutes", "-5"}, {"ftp_max_sessions_per_run", "0"}});
    EXPECT_EQ(cfg.ftp_ttl_minutes, 30);
    EXPECT_EQ(cfg.ftp_max_sessions, 0);
    ASSERT_EQ(cfg.override_warnings.size(), 1u);
    EXPECT_NE(cfg.override_warnings[0].find("a whole number >= 1"), std::string::npos) << cfg.override_warnings[0];
}
