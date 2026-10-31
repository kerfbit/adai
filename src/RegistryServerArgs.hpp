#pragma once

// @adai-status: experimental
// @adai-version: 0.3.0
// @adai-reviewed: 2026-10-31

// TD-035 (narrower scope, per the item's own note): registry_server's argv parsing and
// config-precedence resolution, pulled out of RegistryServer.cpp so that piece is testable
// without a real HTTP server — see RegistryServerArgs_test.cpp. Everything else in
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
    // listener at startup.)
    // TODO: See TD-489 in TECHNICAL_DEBT.md - FTP settings do nothing without FTP enabled; no warning.
    std::optional<bool> ftp_enabled;
    std::optional<int> ftp_port;
    std::optional<std::string> ftp_advertise_ip;
    std::optional<int> ftp_pasv_min;
    std::optional<int> ftp_pasv_max;
    std::optional<std::string> ftp_server_secret;
    std::optional<bool> ftps_enabled;
    std::optional<std::string> ftp_cert_file;
    std::optional<std::string> ftp_key_file;

    // TODO: See TD-489 in TECHNICAL_DEBT.md - admin_enabled has no file/admin layer (CLI, else true).
    // Admin-mutable FTP settings — DO get the full file/admin-override/CLI precedence treatment
    // (see resolve_registry_server_config() below), so these stay optional<>: unset means "this
    // run didn't pass the flag," not "use the CLI default."
    std::optional<int> ftp_ttl_minutes;
    std::optional<int> ftp_max_sessions;
    std::optional<bool> admin_enabled;

    // TD-494: file < CLI (not admin-mutable).
    std::optional<int> claim_lease_minutes;
    std::optional<int> max_upload_mb;
    std::optional<int> max_concurrent_fetches;

    bool help = false;
};

// TODO: See TD-488 in TECHNICAL_DEBT.md - silently ignoring unknown arguments hides typos (--portt -> 8082).
// Parses argv[1..]. Note this preserves the original inline loop's own behavior exactly,
// including one quirk worth flagging rather than silently changing: an unrecognized argument is
// simply ignored (no error, no message) — unlike every sibling daemon's argv parser in this
// codebase, which reports "Unknown argument"/"Unknown option" and exits non-zero. Changing that
// is outside a narrow args-extraction pass; noted in TECHNICAL_DEBT_RESOLVED.md instead.
RegistryServerArgs parse_registry_server_args(int argc, char* argv[]);

struct RegistryServerEffectiveConfig {
    int port = 0;
    std::string data_dir;
    int ftp_ttl_minutes = 30;
    int ftp_max_sessions = 4;
    bool admin_enabled = true;
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
    // placeholder secret (ignored), plaintext FTP on a non-loopback address.
    std::vector<std::string> warnings;
};

// Resolves port/data_dir/ftp_ttl_minutes/ftp_max_sessions/admin_enabled from (lowest to highest
// precedence): `file_config` < `admin_overrides` (persisted via PUT /admin/config — only
// ftp_token_ttl_minutes/ftp_max_sessions_per_run are consulted, matching kImmutableKeys) <
// `args`' explicit CLI flags. An unparseable override value is silently skipped, same as the
// original inline try/catch.
RegistryServerEffectiveConfig resolve_registry_server_config(
    const RegistryServerArgs& args, const ServiceConfig& file_config,
    const std::map<std::string, std::string>& admin_overrides);

}  // namespace adai
