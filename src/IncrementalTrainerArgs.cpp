// @adai-status: experimental
// @adai-version: 0.11.0
// @adai-reviewed: 2026-11-01

#include "IncrementalTrainerArgs.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include "Config.hpp"
#include "Logger.hpp"

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
            // TD-402: it used to be taken as given. LocalTransport joins it into a path, so a typo
            // selected a new, empty pool and ".." put the registry files outside session_dir.
            const std::string value = argv[++i];
            if (!is_dataset_kind(value)) {
                return fail("Invalid value for --dataset-kind: '" + value +
                            "' (expected encoder, decoder, world_model or chatbot)");
            }
            result.dataset_kind = value;
        } else {
            // The command and its own arguments, including command flags such as reset's --yes.
            result.args.push_back(a);
        }
    }

    return result;
}

TrainerServiceArgs parse_trainer_service_args(int argc, char* argv[]) {
    TrainerServiceArgs result;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            result.help = true;
            return result;
        }
    }
    result.global = parse_incremental_trainer_global_args(argc, argv);
    auto& g = result.global;
    if (g.error) return result;
    if (g.foreground || g.admin_port) {
        g.error = true;
        g.error_message = std::string(g.foreground ? "--foreground" : "--admin-port") +
                          " is set by trainer_service on the passes it launches, not passed to it "
                          "(the pass's admin port is TRAINER_CHILD_ADMIN_PORT)";
    } else if (!g.args.empty()) {
        g.error = true;
        g.error_message = "Unexpected argument '" + g.args.front() + "' (trainer_service takes no command)";
    }
    return result;
}

std::vector<std::string> trainer_service_child_argv(const std::string& trainer_path,
                                                    const IncrementalTrainerGlobalArgs& cli, int admin_port) {
    std::vector<std::string> argv = {trainer_path};
    if (cli.config_path) argv.insert(argv.end(), {"--config", *cli.config_path});
    if (cli.model_name) argv.insert(argv.end(), {"--model", *cli.model_name});
    if (cli.gpu_strategy) argv.insert(argv.end(), {"--gpu-strategy", *cli.gpu_strategy});
    argv.push_back("--objective=" + cli.objective);
    argv.insert(argv.end(), {"--dataset-kind", resolve_dataset_kind(cli.objective, cli.dataset_kind)});
    argv.insert(argv.end(), {"--foreground", "--admin-port", std::to_string(admin_port), "resume"});
    return argv;
}

bool incremental_trainer_command_defers_gpu_init(const std::string& command) {
    // TD-172: "serve" removed — incremental_trainer no longer has an always-on service command
    // (see trainer_service, which never links adai_models/GPU objects at all).
    return command == "train" || command == "retrain" || command == "resume";
}

std::string persistent_run_id(const std::string& configured, const std::string& dir) {
    if (!configured.empty()) return configured;
    namespace fs = std::filesystem;
    const fs::path file = fs::path(dir) / "lejepa_run_id";
    {
        std::ifstream in(file);
        std::string id;
        if (in && std::getline(in, id)) {
            while (!id.empty() && (id.back() == '\r' || id.back() == ' ' || id.back() == '\t')) id.pop_back();
            if (!id.empty()) return id;
        }
    }
    std::random_device rd;
    static const char* kHex = "0123456789abcdef";
    std::string suffix;
    for (int i = 0; i < 8; ++i) suffix += kHex[rd() & 0xF];
    const std::string id = "lejepa-" + derive_run_id("") + "-" + suffix;
    std::error_code ec;
    fs::create_directories(dir, ec);
    const fs::path tmp = file.string() + ".tmp";
    bool written = false;
    {
        std::ofstream out(tmp, std::ios::trunc);
        out << id << "\n";
        out.flush();
        written = static_cast<bool>(out);
    }
    if (written) {
        fs::rename(tmp, file, ec);
        written = !ec;
    }
    if (!written) {
        fs::remove(tmp, ec);
        Logger::warn("Could not save the LeJEPA run id to {}; a restarted pass won't reclaim this pass's files",
                     file.string());
    }
    return id;
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
