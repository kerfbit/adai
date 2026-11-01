// Tests for src/MnsServerArgs.{hpp,cpp} — mns_server's argv parsing and config-precedence
// resolution (TD-035), extracted from ModelNameServiceServer.cpp so it's testable without
// standing up a real ModelNameService/HTTP server. That server-lifecycle behavior itself
// (endpoints, registry proxy, admin API) is already covered live by MNSLiveTests
// (mnslive_test.cpp) — this file only covers the main()-specific surface: parsing argv into
// MnsServerArgs, and resolving the effective startup config from file/admin-override/CLI
// precedence.

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>
#include "MnsServerArgs.hpp"

using adai::MnsServerArgs;
using adai::MnsServerEffectiveConfig;

namespace {

// Builds an argv-shaped array from a list of C strings, argv[0] filled in as the program name —
// parse_mns_server_args never consults argv[0] itself, but every real caller has one.
std::vector<char*> make_argv(std::vector<std::string>& storage) {
    std::vector<char*> argv;
    for (auto& s : storage)
        argv.push_back(s.data());
    return argv;
}

}  // namespace

TEST(ParseMnsServerArgs, AdminEnabledAcceptsTheSharedBooleanWords) {
    for (const char* yes : {"true", "1", "yes", "on", "TRUE", "Enabled"}) {
        std::vector<std::string> raw = {"mns_server", "--admin-enabled", yes};
        auto argv = make_argv(raw);
        auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_FALSE(args.error) << yes << ": " << args.error_message;
        EXPECT_EQ(args.admin_enabled, true) << yes;
    }
    for (const char* no : {"false", "0", "no", "off", "Disabled"}) {
        std::vector<std::string> raw = {"mns_server", "--admin-enabled", no};
        auto argv = make_argv(raw);
        auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_FALSE(args.error) << no;
        EXPECT_EQ(args.admin_enabled, false) << no;
    }
}

// TD-456: an unrecognised value used to fall back to true, so "--admin-enabled flase" left
// PUT /admin/config open.
TEST(ParseMnsServerArgs, AnUnrecognisedAdminEnabledValueIsAnError) {
    std::vector<std::string> raw = {"mns_server", "--admin-enabled", "flase"};
    auto argv = make_argv(raw);
    auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(args.error);
    EXPECT_NE(args.error_message.find("'flase'"), std::string::npos) << args.error_message;
    EXPECT_FALSE(args.admin_enabled.has_value());
}

TEST(ParseMnsServerArgs, NoArgsLeavesEverythingUnset) {
    std::vector<std::string> raw = {"mns_server"};
    auto argv = make_argv(raw);
    auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());

    EXPECT_FALSE(args.help);
    EXPECT_FALSE(args.error);
    EXPECT_FALSE(args.config_path.has_value());
    EXPECT_FALSE(args.port.has_value());
    EXPECT_FALSE(args.data_dir.has_value());
    EXPECT_FALSE(args.registry_url.has_value());
    EXPECT_FALSE(args.registry_group.has_value());
    EXPECT_FALSE(args.admin_enabled.has_value());
}

TEST(ParseMnsServerArgs, ParsesEveryFlag) {
    std::vector<std::string> raw = {
        "mns_server",       "--config",         "/tmp/config.mns.conf",
        "--port",           "9999",             "--data-dir",
        "/tmp/data",        "--registry-url",   "http://localhost:8082",
        "--registry-group", "prod",             "--admin-enabled",
        "false",
    };
    auto argv = make_argv(raw);
    auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());

    ASSERT_FALSE(args.error) << args.error_message;
    EXPECT_EQ(args.config_path, "/tmp/config.mns.conf");
    EXPECT_EQ(args.port, 9999);
    EXPECT_EQ(args.data_dir, "/tmp/data");
    EXPECT_EQ(args.registry_url, "http://localhost:8082");
    EXPECT_EQ(args.registry_group, "prod");
    ASSERT_TRUE(args.admin_enabled.has_value());
    EXPECT_FALSE(*args.admin_enabled);
}

TEST(ParseMnsServerArgs, ShortDashPForPort) {
    std::vector<std::string> raw = {"mns_server", "-p", "1234"};
    auto argv = make_argv(raw);
    auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());
    ASSERT_FALSE(args.error);
    EXPECT_EQ(args.port, 1234);
}

TEST(ParseMnsServerArgs, InvalidPortSetsError) {
    std::vector<std::string> raw = {"mns_server", "--port", "not-a-number"};
    auto argv = make_argv(raw);
    auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(args.error);
    EXPECT_NE(args.error_message.find("not-a-number"), std::string::npos);
}

TEST(ParseMnsServerArgs, HelpFlagShortCircuits) {
    std::vector<std::string> raw = {"mns_server", "--data-dir", "/tmp/x", "--help"};
    auto argv = make_argv(raw);
    auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(args.help);
    EXPECT_FALSE(args.error);
}

TEST(ParseMnsServerArgs, UnknownArgumentSetsError) {
    std::vector<std::string> raw = {"mns_server", "--bogus"};
    auto argv = make_argv(raw);
    auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(args.error);
    EXPECT_NE(args.error_message.find("--bogus"), std::string::npos);
}

// TD-456: a flag given no value was reported as "Unknown argument".
TEST(ParseMnsServerArgs, AFlagMissingItsValueSaysSo) {
    for (const char* flag : {"--config", "--port", "-p", "--data-dir", "--registry-url", "--registry-group",
                             "--admin-enabled"}) {
        std::vector<std::string> raw = {"mns_server", flag};
        auto argv = make_argv(raw);
        auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_TRUE(args.error) << flag;
        EXPECT_EQ(args.error_message, std::string("Missing value for ") + flag);
    }
    std::vector<std::string> raw = {"mns_server", "--data-dir", "--port", "9000"};
    auto argv = make_argv(raw);
    EXPECT_EQ(adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data()).error_message,
              "Missing value for --data-dir");
}

// TD-456: std::stoi took "8083x" as 8083, and --port 99999 listened on 34463 (mod 65536).
TEST(ParseMnsServerArgs, PortMustBeAWholeNumberInRange) {
    for (const char* value : {"8083x", "99999", "0", "-1", "65536", "1e3", ""}) {
        std::vector<std::string> raw = {"mns_server", "--port", value};
        auto argv = make_argv(raw);
        auto args = adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_TRUE(args.error) << value;
        EXPECT_FALSE(args.port.has_value()) << value;
    }
    std::vector<std::string> raw = {"mns_server", "--port", "65535"};
    auto argv = make_argv(raw);
    EXPECT_EQ(adai::parse_mns_server_args(static_cast<int>(argv.size()), argv.data()).port, 65535);
}

class ResolveMnsServerConfigTest : public ::testing::Test {
   protected:
    adai::ServiceConfig file_config;
    std::map<std::string, std::string> no_overrides;

    void SetUp() override {
        file_config.name_service_port = 8083;
        file_config.name_service_dir = "name_service";
        file_config.registry_server_url = "";
        file_config.run_group = "";
    }
};

TEST_F(ResolveMnsServerConfigTest, FileConfigAloneIsTheBaseline) {
    MnsServerArgs args;  // everything unset
    auto eff = adai::resolve_mns_server_config(args, file_config, no_overrides);

    EXPECT_EQ(eff.port, 8083);
    EXPECT_EQ(eff.data_dir, "name_service");
    EXPECT_EQ(eff.registry_url, "");
    EXPECT_EQ(eff.registry_group, "default") << "empty run_group must fall back to \"default\"";
    EXPECT_TRUE(eff.admin_enabled) << "admin_enabled defaults to true when nothing overrides it";
}

TEST_F(ResolveMnsServerConfigTest, RunGroupFromFileConfigIsUsedWhenNonEmpty) {
    file_config.run_group = "nightly";
    MnsServerArgs args;
    auto eff = adai::resolve_mns_server_config(args, file_config, no_overrides);
    EXPECT_EQ(eff.registry_group, "nightly");
}

TEST_F(ResolveMnsServerConfigTest, AdminOverridesBeatFileConfig) {
    file_config.registry_server_url = "http://file-configured:8082";
    file_config.run_group = "file-group";
    std::map<std::string, std::string> overrides = {
        {"registry_url", "http://overridden:8082"},
        {"registry_group", "overridden-group"},
    };
    MnsServerArgs args;
    auto eff = adai::resolve_mns_server_config(args, file_config, overrides);
    EXPECT_EQ(eff.registry_url, "http://overridden:8082");
    EXPECT_EQ(eff.registry_group, "overridden-group");
}

TEST_F(ResolveMnsServerConfigTest, RegistryGroupOverrideAppliesEvenWithNoUrlOverride) {
    // Regression coverage for the original code's own documented case: only registry_group was
    // ever persisted (e.g. changed via PUT /admin/config independently of the URL) — it must
    // still take effect even though registry_url has no matching override entry.
    std::map<std::string, std::string> overrides = {{"registry_group", "just-the-group"}};
    MnsServerArgs args;
    auto eff = adai::resolve_mns_server_config(args, file_config, overrides);
    EXPECT_EQ(eff.registry_url, "");
    EXPECT_EQ(eff.registry_group, "just-the-group");
}

TEST_F(ResolveMnsServerConfigTest, CliFlagsBeatAdminOverridesAndFileConfig) {
    file_config.registry_server_url = "http://file:8082";
    std::map<std::string, std::string> overrides = {
        {"registry_url", "http://override:8082"},
        {"registry_group", "override-group"},
    };
    MnsServerArgs args;
    args.registry_url = "http://cli:8082";
    args.registry_group = "cli-group";
    args.admin_enabled = false;

    auto eff = adai::resolve_mns_server_config(args, file_config, overrides);
    EXPECT_EQ(eff.registry_url, "http://cli:8082");
    EXPECT_EQ(eff.registry_group, "cli-group");
    EXPECT_FALSE(eff.admin_enabled);
}

TEST_F(ResolveMnsServerConfigTest, CliPortAndDataDirOverrideFileConfig) {
    MnsServerArgs args;
    args.port = 12345;
    args.data_dir = "/custom/data";
    auto eff = adai::resolve_mns_server_config(args, file_config, no_overrides);
    EXPECT_EQ(eff.port, 12345);
    EXPECT_EQ(eff.data_dir, "/custom/data");
}

// TD-456: MNS_ADMIN_ENABLED in config.mns.conf sets the default; --admin-enabled overrides it.
TEST_F(ResolveMnsServerConfigTest, AdminEnabledComesFromTheConfigUnlessTheFlagIsGiven) {
    adai::MnsServerArgs args;
    file_config.name_service_admin_enabled = false;
    EXPECT_FALSE(adai::resolve_mns_server_config(args, file_config, no_overrides).admin_enabled);
    args.admin_enabled = true;
    EXPECT_TRUE(adai::resolve_mns_server_config(args, file_config, no_overrides).admin_enabled);
    file_config.name_service_admin_enabled = true;
    args.admin_enabled.reset();
    EXPECT_TRUE(adai::resolve_mns_server_config(args, file_config, no_overrides).admin_enabled);
}
