// @adai-status: experimental
// @adai-version: 0.16.0
// @adai-reviewed: 2026-11-09

#include "IncrementalTrainerArgs.hpp"
#include <algorithm>
#include <array>
#include <cctype>
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
#include <windows.h>  // TD-524: GetModuleFileNameA
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
    static const std::set<std::string> kValueFlags = {
        "--config",       "--gpu-strategy", "--model", "--admin-port",
        "--dataset-kind", "--log-file",     "--seed"};

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
        } else if (a == "--seed") {
            const std::string value = argv[++i];
            const auto seed = parse_strict_int(value);
            if (!seed || *seed < 1) {
                return fail("Invalid value for --seed: '" + value +
                            "' (expected a whole number of at least 1)");
            }
            result.seed = *seed;
        } else if (a.rfind("--objective=", 0) == 0) {
            // TD-401: anything but "lejepa" (e.g. the typo "lejpa") used to mean chatbot training.
            const std::string value = a.substr(std::string("--objective=").size());
            if (value != "chatbot" && value != "lejepa") {
                return fail("Invalid value for --objective: '" + value + "' (expected chatbot or lejepa)");
            }
            result.objective = value;
        } else if (a == "--objective") {
            return fail("--objective takes its value after '=': --objective=chatbot or --objective=lejepa");
        } else if (a == "--log-file") {  // TD-431
            result.log_file = argv[++i];
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

std::string sibling_executable_path(const std::string& argv0, const std::string& name) {
    namespace fs = std::filesystem;
    std::error_code ec;
#ifdef _WIN32
    char module[MAX_PATH] = {};
    const DWORD n = ::GetModuleFileNameA(nullptr, module, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return (fs::path(module).parent_path() / name).string();
#elif defined(__linux__)
    const fs::path self = fs::read_symlink("/proc/self/exe", ec);
    if (!ec && self.has_parent_path()) return (self.parent_path() / name).string();
#endif
    const fs::path p(argv0);
    if (p.has_parent_path()) return (p.parent_path() / name).string();
    Logger::warn("Could not locate this executable's directory; launching '{}' from PATH", name);
    return name;
}

std::string pass_log_path(const std::string& log_path) {
    namespace fs = std::filesystem;
    const fs::path p(log_path);
    if (!p.has_extension()) return log_path + ".pass";
    return (p.parent_path() / (p.stem().string() + ".pass" + p.extension().string())).string();
}

std::string default_background_log_path() {
    namespace fs = std::filesystem;
    fs::path dir;
#ifdef _WIN32
    if (const char* local = std::getenv("LOCALAPPDATA"); local && *local) dir = fs::path(local) / "adai" / "Logs";
#else
    if (const char* state = std::getenv("XDG_STATE_HOME"); state && *state) {
        dir = fs::path(state) / "adai" / "log";
    } else if (const char* home = std::getenv("HOME"); home && *home) {
        dir = fs::path(home) / ".local" / "state" / "adai" / "log";
    }
#endif
    if (dir.empty()) return "incremental_trainer.log";  // no usable home: the working directory
    std::error_code ec;
    fs::create_directories(dir, ec);
    return (dir / "incremental_trainer.log").string();
}

std::optional<int> parse_epochs_arg(const std::vector<std::string>& args, int default_epochs, std::string& error) {
    if (args.size() < 2) return default_epochs;
    if (args.size() > 2) {
        error = "Unexpected argument '" + args[2] + "' (" + args[0] + " takes one epoch count)";
        return std::nullopt;
    }
    const auto epochs = parse_strict_int(args[1]);
    if (!epochs || *epochs < 1) {
        error = "Invalid epoch count '" + args[1] + "' (expected a whole number, at least 1)";
        return std::nullopt;
    }
    return *epochs;
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
    if (g.log_file) {  // TD-431: trainer_service picks each pass's file from LOG_FILE_PATH
        g.error = true;
        g.error_message = "--log-file is set by trainer_service on the passes it launches; set LOG_FILE_PATH instead";
    } else if (g.foreground || g.admin_port) {
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
                                                    const IncrementalTrainerGlobalArgs& cli,
                                                    std::optional<int> admin_port,
                                                    const std::string& pass_log_file) {
    std::vector<std::string> argv = {trainer_path};
    if (cli.config_path) argv.insert(argv.end(), {"--config", *cli.config_path});
    if (cli.model_name) argv.insert(argv.end(), {"--model", *cli.model_name});
    if (cli.gpu_strategy) argv.insert(argv.end(), {"--gpu-strategy", *cli.gpu_strategy});
    argv.push_back("--objective=" + cli.objective);
    argv.insert(argv.end(), {"--dataset-kind", resolve_dataset_kind(cli.objective, cli.dataset_kind)});
    if (!pass_log_file.empty()) argv.insert(argv.end(), {"--log-file", pass_log_file});  // TD-431
    argv.push_back("--foreground");
    if (admin_port) argv.insert(argv.end(), {"--admin-port", std::to_string(*admin_port)});
    argv.push_back("resume");
    return argv;
}

bool incremental_trainer_command_defers_gpu_init(const std::string& command) {
    // TD-172: "serve" removed — incremental_trainer no longer has an always-on service command
    // (see trainer_service, which never links adai_models/GPU objects at all).
    return command == "train" || command == "retrain" || command == "resume";
}

std::string persistent_run_id(const std::string& configured, const std::string& dir, const std::string& name) {
    if (!configured.empty()) return configured;
    namespace fs = std::filesystem;
    const fs::path file = fs::path(dir) / (name + "_run_id");
    {
        std::ifstream in(file);
        std::string id;
        if (in && std::getline(in, id)) {
            while (!id.empty() && (id.back() == '\r' || id.back() == ' ' || id.back() == '\t')) id.pop_back();
            if (!id.empty()) return id;
        }
    }
    const std::string id = name + "-" + derive_run_id("");  // TD-404: already unique per host and process
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
        Logger::warn("Could not save the run id to {}; a restarted pass won't reclaim this pass's files",
                     file.string());
    }
    return id;
}

// TD-404: the full hostname (sanitized), the full PID and 8 random hex digits. It used to be the first 8
// hostname characters plus pid % 10000, so two hosts sharing a prefix (ai-machine-1, ai-machine-2) could get
// the same id and, in a shared registry, own each other's claimed files.
std::string derive_run_id(const std::string& configured) {
    if (!configured.empty())
        return configured;
    std::string host = "host";
#ifdef _WIN32
    if (const char* env_host = std::getenv("COMPUTERNAME"))
        host = env_host;
    const long long pid = static_cast<long long>(_getpid());
#else
    std::array<char, 256> buf{};
    if (gethostname(buf.data(), buf.size() - 1) == 0 && buf[0] != '\0')
        host = buf.data();
    const long long pid = static_cast<long long>(getpid());
#endif
    for (char& c : host) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '.') c = '-';
    }
    std::random_device rd;
    static const char* kHex = "0123456789abcdef";
    std::string suffix;
    for (int i = 0; i < 8; ++i) suffix += kHex[rd() & 0xF];
    return host + "_" + std::to_string(pid) + "_" + suffix;
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
