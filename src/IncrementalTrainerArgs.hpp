#pragma once

// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-14

// TD-035: incremental_trainer's global argv parsing and a few small pure helpers, pulled out of
// IncrementalTrainingTool.cpp so they're testable without touching a real IncrementalTrainer
// (real training, forking, GPU init) — see IncrementalTrainerArgs_test.cpp. IncrementalTrainer
// itself is already covered by IncrementalTrainerTests/IncrementalTrainerBackgroundTests/
// IncrementalTrainerControlTests/TrainerControlStateTests/TrainerAdminAPITests.

#include <optional>
#include <string>
#include <vector>

namespace adai {

struct IncrementalTrainerGlobalArgs {
    std::optional<std::string> config_path;
    std::optional<std::string> gpu_strategy;
    std::optional<std::string> model_name;
    bool foreground = false;
    // TD-172: set only by the process-supervisor binary when it launches a single-pass child
    // (`--foreground --admin-port <N> resume`). Starts a TrainerAdminAPI bound to 127.0.0.1:<N>
    // for the duration of that one pass — the supervisor's own admin listener proxies to it while
    // the child is alive. Absent for every interactive/manual invocation.
    std::optional<int> admin_port;
    // args[0] is the command (e.g. "train"), args[1..] are its own arguments — --config,
    // --gpu-strategy, --model, --foreground, and --admin-port are stripped out before this is
    // populated, exactly like the original inline loop.
    std::vector<std::string> args;
};

IncrementalTrainerGlobalArgs parse_incremental_trainer_global_args(int argc, char* argv[]);

// train/retrain/resume/serve defer GPU init until after their fork (CUDA contexts aren't
// fork-safe) or, for serve, until after Logger::init() runs — every other command initializes
// GPU immediately in main(). Pure classification, no I/O.
bool incremental_trainer_command_defers_gpu_init(const std::string& command);

// Derives a fallback run_id from hostname+pid when `configured` (RUN_ID from config/MNS) is
// empty. Pure aside from the hostname/pid syscalls — no file or network I/O.
std::string derive_run_id(const std::string& configured);

struct ResetCommandArgs {
    bool yes = false;
    bool keep_data = false;
};

// Parses `reset`'s own flags. `args` is the reset command's argument slice, NOT including "reset"
// itself (e.g. `incremental_trainer reset --yes --keep-data` -> {"--yes", "--keep-data"}).
ResetCommandArgs parse_reset_command_args(const std::vector<std::string>& args);

}  // namespace adai
