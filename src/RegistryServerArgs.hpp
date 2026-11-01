#pragma once

// @adai-status: experimental
// @adai-version: 0.5.0
// @adai-reviewed: 2026-11-01

// TD-035 (narrower scope, per the item's own note): registry_server's argv parsing and
// config-precedence resolution, pulled out of RegistryServer.cpp so that piece is testable
// without a real HTTP server — see tests/registry_server_args_test.cpp. Everything else in
// RegistryServer.cpp (the actual request handlers, the FTP listener, the file-scope globals they
// share) stays untouched: it's already exercised live by DatasetRegistryLiveTests,
// TrainerAdminAPITests, and RegistryFtpConfinementTests (which spawns the real compiled binary),
// and extracting *that* into an in-process-testable class is a materially bigger job the TD
// explicitly defers.

#include <map>
#include <optional>
#include <string>
#include <vector>
#include "Config.hpp"

namespace adai {

struct RegistryServerArgs {
    std::optional<std::string> config_path;
    std::optional<int> port;
    std::optional<std::string> data_dir;

    // TD-284: FTP listener settings resolve as config file < CLI (resolve_registry_server_config());
    // unset means "this run didn't pass the flag". They used to be CLI-only, so the FTPS and secret
    // keys config.registry.conf ships did nothing. (Not admin-mutable: they're baked into the
    // listener at startup.) With FTP disabled they do nothing, and a flag passed for one of them
    // is listed in the resolved config's warnings (TD-489).
    std::optional<bool> ftp_enabled;
    std::optional<int> ftp_port;
    std::optional<std::string> ftp_advertise_ip;
    std::optional<int> ftp_pasv_min;
    std::optional<int> ftp_pasv_max;
    std::optional<std::string> ftp_server_secret;
    std::optional<bool> ftps_enabled;
    std::optional<std::string> ftp_cert_file;
    std::optional<std::string> ftp_key_file;

    // Admin-mutable FTP settings — DO get the full file/admin-override/CLI precedence treatment
    // (see resolve_registry_server_config() below), so these stay optional<>: unset means "this
    // run didn't pass the flag," not "use the CLI default."
    std::optional<int> ftp_ttl_minutes;
    std::optional<int> ftp_max_sessions;
    // --admin-enabled, else REGISTRY_ADMIN_ENABLED from the file (TD-489; it used to be the flag,
    // else true). Not admin-mutable: PUT /admin/config can't switch itself off.
    std::optional<bool> admin_enabled;

    // TD-494: file < CLI (not admin-mutable).
    std::optional<int> claim_lease_minutes;
    std::optional<int> max_upload_mb;
    std::optional<int> max_concurrent_fetches;

    bool help = false;
    // TD-488: set (with a message) on an unknown argument, a flag missing its value, a number that
    // isn't a whole number in range, a non-boolean --admin-enabled, or --ftp-pasv-min above
    // --ftp-pasv-max. main() prints it with the usage text and exits 1.
    bool error = false;
    std::string error_message;
};

// Parses argv[1..]. Never throws. --help/-h anywhere wins over every other argument, including a
// bad one (TD-489). Numbers must be whole and in range (ports 1-65535, --ftp-ttl,
// --max-upload-mb and --max-concurrent-fetches at least 1, --ftp-max-sessions and
// --claim-lease-minutes at least 0, where 0 means unlimited / never expire). An unknown argument
// is an error (TD-488; it used to be ignored, so --portt 8091 started on 8082).
RegistryServerArgs parse_registry_server_args(int argc, char* argv[]);

struct RegistryServerEffectiveConfig {
    int port = 0;
    std::string data_dir;
    int ftp_ttl_minutes = 30;
    int ftp_max_sessions = 4;
    bool admin_enabled = true;  // --admin-enabled, else REGISTRY_ADMIN_ENABLED (TD-489)
    int claim_lease_minutes = 30;     // TD-494
    int max_upload_mb = 2048;         // TD-494
    int max_concurrent_fetches = 2;   // TD-494

    // TD-284: the FTP listener (config file < CLI).
    bool ftp_enabled = false;
    int ftp_port = 2121;
    std::string ftp_advertise_ip;
    int ftp_pasv_min = 50000;
    int ftp_pasv_max = 50099;
    std::string ftp_server_secret;  // empty = a random password per token
    bool ftps_enabled = false;
    std::string ftp_cert_file;
    std::string ftp_key_file;
    // Things the operator should see at startup: a file value a CLI flag overrides, the shipped
    // placeholder secret (ignored), plaintext FTP on a non-loopback address, an FTP flag passed
    // while FTP is disabled (TD-489).
    std::vector<std::string> warnings;
    // TD-488: saved admin overrides that were skipped as out of range or not a number.
    std::vector<std::string> override_warnings;
    // TD-488: a combination that can't run (an empty FTP passive port range); main() exits 1.
    std::vector<std::string> errors;
};

// Resolves port/data_dir/ftp_ttl_minutes/ftp_max_sessions from (lowest to highest precedence):
// `file_config` < `admin_overrides` (persisted via PUT /admin/config — only
// ftp_token_ttl_minutes/ftp_max_sessions_per_run are consulted, matching kImmutableKeys) <
// `args`' explicit CLI flags. admin_enabled, the FTP listener and the TD-494 limits are
// `file_config` < CLI. An override value that isn't a whole number in range is skipped and
// listed in override_warnings (TD-488).
RegistryServerEffectiveConfig resolve_registry_server_config(
    const RegistryServerArgs& args, const ServiceConfig& file_config,
    const std::map<std::string, std::string>& admin_overrides);

}  // namespace adai
