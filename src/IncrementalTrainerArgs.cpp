// @adai-status: experimental
// @adai-version: 0.7.0
// @adai-reviewed: 2026-11-01

#include "IncrementalTrainerArgs.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <set>
#include "Config.hpp"

#ifdef _WIN32
// TD-159: _getpid() below is declared here, not in <cstdlib>/<unistd.h> — was previously called
// with no include providing it at all, confirmed via a real cross-compile error ("'_getpid' was
// not declared in this scope").
#include <process.h>
#else
#include <unistd.h>
#endif

namespace adai {

IncrementalTrainerGlobalArgs parse_incremental_trainer_global_args(int argc, char* argv[]) {
    IncrementalTrainerGlobalArgs result;
    auto fail = [&](const std::string& message) {
        result.error = true;
        result.error_message = message;
        return result;
    };

    // TD-401: a global flag that takes a value, given none (last in argv, or followed by another
    // "--" flag), is an error. It used to land in args as a positional, so `status --config` ran
    // with normal config discovery.
    static const std::set<std::string> kValueFlags = {"--config", "--gpu-strategy", "--model", "--admin-port",
                                                      "--dataset-kind"};

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (kValueFlags.count(a) != 0 && (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)) {
            return fail("Missing value for " + a);
        }
        if (a == "--config") {
            result.config_path = argv[++i];
        } else if (a == "--gpu-strategy") {
            result.gpu_strategy = argv[++i];
        } else if (a == "--model") {
            result.model_name = argv[++i];
        } else if (a == "--foreground") {
            result.foreground = true;
        } else if (a == "--admin-port") {
            // TD-401: std::stoi aborted the process on "abc" (no try in main()) and took "8084x" as 8084.
            const std::string value = argv[++i];
            const auto port = parse_strict_int(value);
            if (!port || *port < 1 || *port > 65535) {
                return fail("Invalid value for --admin-port: '" + value + "' (expected a port, 1-65535)");
            }
            result.admin_port = *port;
        } else if (a.rfind("--objective=", 0) == 0) {
            // TD-401: anything but "lejepa" (e.g. the typo "lejpa") used to mean chatbot training.
            const std::string value = a.substr(std::string("--objective=").size());
            if (value != "chatbot" && value != "lejepa") {
                return fail("Invalid value for --objective: '" + value + "' (expected chatbot or lejepa)");
            }
            result.objective = value;
        } else if (a == "--objective") {
            return fail("--objective takes its value after '=': --objective=chatbot or --objective=lejepa");
        } else if (a == "--dataset-kind") {
            // TODO: See TD-402 in TECHNICAL_DEBT.md - not validated against the four kinds; LocalTransport joins it
            // into a path (typo = new empty pool, ".." escapes session_dir)
            result.dataset_kind = argv[++i];
        } else {
            // The command and its own arguments, including command flags such as reset's --yes.
            result.args.push_back(a);
        }
    }

    return result;
}

bool incremental_trainer_command_defers_gpu_init(const std::string& command) {
    // TD-172: "serve" removed — incremental_trainer no longer has an always-on service command
    // (see trainer_service, which never links adai_models/GPU objects at all).
    return command == "train" || command == "retrain" || command == "resume";
}

// TODO: See TD-404 in TECHNICAL_DEBT.md - 8-char hostname prefix + pid % 10000 can collide across hosts; the
// LeJEPA path always uses this (never MNS), and run ids decide claimed-file ownership in a shared registry
std::string derive_run_id(const std::string& configured) {
    if (!configured.empty())
        return configured;
    std::string host = "host";
#ifdef _WIN32
    if (const char* env_host = std::getenv("COMPUTERNAME"))
        host = env_host;
    const int pid_tail = static_cast<int>(_getpid() % 10000);
#else
    std::array<char, 256> buf{};
    if (gethostname(buf.data(), buf.size() - 1) == 0)
        host = buf.data();
    const int pid_tail = static_cast<int>(getpid() % 10000);
#endif
    if (host.size() > 8)
        host = host.substr(0, 8);
    return host + "_" + std::to_string(pid_tail);
}

std::string resolve_dataset_kind(const std::string& objective,
                                 const std::optional<std::string>& cli_override) {
    if (cli_override) {
        return *cli_override;
    }
    return (objective == "lejepa") ? "world_model" : "chatbot";
}

ResetCommandArgs parse_reset_command_args(const std::vector<std::string>& args) {
    ResetCommandArgs result;
    for (const auto& a : args) {
        if (a == "--yes") {
            result.yes = true;
        } else if (a == "--keep-data") {
            result.keep_data = true;
        }
    }
    return result;
}

PassOutcome classify_pass_exit(int exit_code) {
    if (exit_code == kResumeExitTrained)
        return PassOutcome::DidWork;
    if (exit_code == kResumeExitNothingPending)
        return PassOutcome::Idle;
    if (exit_code >= 128)
        return PassOutcome::Crashed;
    return PassOutcome::Failed;
}

int failure_backoff_seconds(int consecutive_failures, int base_seconds, int max_seconds) {
    long long wait = base_seconds;
    for (int i = 1; i < consecutive_failures && wait < max_seconds; ++i)
        wait *= 2;
    return static_cast<int>(std::min<long long>(wait, max_seconds));
}

}  // namespace adai
