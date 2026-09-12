// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

#include "IncrementalTrainerArgs.hpp"
#include <array>
#include <cstdlib>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace adai {

IncrementalTrainerGlobalArgs parse_incremental_trainer_global_args(int argc, char* argv[]) {
    IncrementalTrainerGlobalArgs result;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            result.config_path = argv[++i];
        } else if (a == "--gpu-strategy" && i + 1 < argc) {
            result.gpu_strategy = argv[++i];
        } else if (a == "--model" && i + 1 < argc) {
            result.model_name = argv[++i];
        } else if (a == "--foreground") {
            result.foreground = true;
        } else {
            result.args.push_back(a);
        }
    }

    return result;
}

bool incremental_trainer_command_defers_gpu_init(const std::string& command) {
    return command == "train" || command == "retrain" || command == "resume" ||
          command == "serve";
}

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

}  // namespace adai
