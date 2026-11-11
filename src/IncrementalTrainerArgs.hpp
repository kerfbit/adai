#pragma once

// @adai-status: beta
// @adai-version: 0.15.0
// @adai-reviewed: 2026-11-10

// TD-035: incremental_trainer's global argv parsing and a few small pure helpers, pulled out of
// IncrementalTrainingTool.cpp so they're testable without touching a real IncrementalTrainer
// (real training, forking, GPU init) — see tests/incremental_trainer_args_test.cpp. IncrementalTrainer
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
    // TD-368: `--seed <n>`, at least 1; overrides SEED.
    std::optional<int> seed;
    // TD-183: selects which training objective `train` runs. "chatbot" (default — the flag can
    // be omitted entirely, matching every existing invocation byte-for-byte) is the existing
    // teacher-forcing ChatbotTrainer/IncrementalTrainer pipeline; "lejepa" instead pretrains the
    // standalone LeJEPAEncoder world model via LeJEPAEncoder::train_step() on the same acquired
    // files, reusing the dataset registry/FTP machinery but none of ChatbotTrainer's own code
    // path. Parsed from `--objective=<value>` (an "=" form, not the space-separated
    // "--flag value" convention every other global flag here uses) — deliberately matching the
    // literal syntax named in the flag's own filed TD (TECHNICAL_DEBT.md's TD-183 title) rather
    // than silently normalizing it to this parser's own pre-existing convention.
    std::string objective = "chatbot";
    // TD-202: overrides the dataset registry's per-trainable-piece sub-pool that would otherwise
    // be picked automatically from `objective` ("lejepa" -> "world_model", else "chatbot") in
    // IncrementalTrainingTool.cpp's main(). Empty (unset) means "use the automatic default" —
    // explicitly passing e.g. "--dataset-kind chatbot" always wins over that default.
    std::optional<std::string> dataset_kind;
    // TD-431: --log-file <path> overrides LOG_FILE_PATH for this run. trainer_service uses it to give
    // each pass its own file (pass_log_path()); it's also usable by hand.
    std::optional<std::string> log_file;
    // args[0] is the command (e.g. "train"), args[1..] are its own arguments — --config,
    // --gpu-strategy, --model, --foreground, --admin-port, --objective, and --dataset-kind are
    // stripped out before this is populated, exactly like the original inline loop.
    std::vector<std::string> args;
    // TD-401: set (with a message) for a global flag missing its value, an --admin-port that isn't
    // a port (1-65535), or an --objective other than chatbot/lejepa; main() prints it with the
    // usage and exits 1.
    bool error = false;
    std::string error_message;
};

IncrementalTrainerGlobalArgs parse_incremental_trainer_global_args(int argc, char* argv[]);

// TD-557: trainer_service's command line. The same global flags as incremental_trainer, except that
// --foreground and --admin-port (what trainer_service passes to each pass it launches) and any
// positional argument (it has no commands) are errors; all three used to be silently ignored, as were
// the parser's own errors, and --help started the service. `help` is set for --help/-h anywhere.
struct TrainerServiceArgs {
    IncrementalTrainerGlobalArgs global;
    bool help = false;
};

TrainerServiceArgs parse_trainer_service_args(int argc, char* argv[]);

// TD-403: the command line trainer_service runs each single-pass child with: its --config, --model
// and --gpu-strategy when given, always --objective=<objective> and --dataset-kind <kind> (kind from
// resolve_dataset_kind()), then `--foreground [--admin-port <port>] resume`. The objective and kind used
// to be parsed and dropped, so the service could only train the chatbot on the chatbot pool. TD-524:
// --admin-port only when `admin_port` is set (TRAINER_ADMIN_ENABLED); it used to be passed always, so
// every pass opened a loopback port even with the admin API off.
std::vector<std::string> trainer_service_child_argv(const std::string& trainer_path,
                                                    const IncrementalTrainerGlobalArgs& cli,
                                                    std::optional<int> admin_port,
                                                    const std::string& pass_log_file = "");

// TD-431 (owner decision, November 2, 2026): the file each trainer_service pass logs to when LOG_FILE_PATH
// is set: the same name with ".pass" before the extension (trainer.log -> trainer.pass.log; trainer ->
// trainer.pass). trainer_service keeps LOG_FILE_PATH itself, so each file is rotated by one process; both
// used to write and rotate one file, losing about a quarter of the configured history.
std::string pass_log_path(const std::string& log_path);

// TD-431: where a detached (background) incremental_trainer run logs when LOG_FILE_PATH is empty, since its
// stdout/stderr go to /dev/null: LOG_FILE_STANDARDS.md's per-user location, $XDG_STATE_HOME (else
// ~/.local/state)/adai/log/incremental_trainer.log, or %LOCALAPPDATA%\adai\Logs on Windows. Creates the
// directory. A foreground run with LOG_FILE_PATH empty logs to the console only (the journal under systemd).
std::string default_background_log_path();

// train/retrain/resume/serve defer GPU init until after their fork (CUDA contexts aren't
// fork-safe) or, for serve, until after Logger::init() runs — every other command initializes
// GPU immediately in main(). Pure classification, no I/O.
bool incremental_trainer_command_defers_gpu_init(const std::string& command);

// Derives a fallback run_id when `configured` (RUN_ID from config/MNS) is empty: TD-404, the full
// hostname, the PID and 8 random hex digits, so two hosts or processes never share one. No file or
// network I/O.
std::string derive_run_id(const std::string& configured);

// TD-415/TD-404: a run id that survives restarts. `configured` (RUN_ID) when set; else the id kept in
// `<dir>/<name>_run_id`, created on first use as "<name>-" + derive_run_id("") (written to a temporary
// file and renamed). Registries hand a claim back to the run id that made it, so a pass restarted after a
// crash or kill reclaims its own files; with a fresh PID-based id each time they stayed claimed by an id no
// later process had. Used by the LeJEPA pass ("lejepa", the world-model directory) and, without MNS, by
// the chatbot passes ("chatbot", SESSION_DIR). If the file can't be written the new id is still returned
// (and a warning logged), so the pass runs; only reclaiming is lost.
std::string persistent_run_id(const std::string& configured, const std::string& dir,
                              const std::string& name = "lejepa");

// TD-416: `train`/`retrain`'s optional epoch count (args[1], after the command). A whole number of at
// least 1; absent means `default_epochs`. Anything else sets `error` and returns nullopt (`train abc`
// used to abort through an uncaught std::stoi, and `train 5x` trained 5 epochs).
// TD-524: `name` in the directory of the running executable (/proc/self/exe on Linux, the module path on
// Windows), so trainer_service launches the incremental_trainer installed beside it. Falls back to
// argv[0]'s directory, and to the bare name (a PATH search) only when neither is known. It used to use
// argv[0] alone, so a trainer_service started by bare name ran whichever incremental_trainer PATH found.
std::string sibling_executable_path(const std::string& argv0, const std::string& name);

std::optional<int> parse_epochs_arg(const std::vector<std::string>& args, int default_epochs, std::string& error);

// TD-202/TD-204: resolves the dataset registry sub-pool ("chatbot"/"world_model") for this
// invocation. Deliberately takes only `objective` and `cli_override` — NOT a pre-loaded
// ServiceConfig::dataset_kind — so a static config-file/env DATASET_KIND value structurally
// cannot leak into this decision. That's the actual fix for the TD-203 regression: a config
// value is a single static string that can't vary between a plain `train` and a
// `--objective=lejepa` run against the same config.trainer.conf, so letting it override the
// automatic default would silently reunite the two objectives' pools — the exact leak TD-202
// exists to prevent. Only a deliberate, per-invocation `--dataset-kind` flag (`cli_override`)
// may override the objective-based default.
std::string resolve_dataset_kind(const std::string& objective,
                                 const std::optional<std::string>& cli_override);

struct ResetCommandArgs {
    bool yes = false;
    bool keep_data = false;
};

// Parses `reset`'s own flags. `args` is the reset command's argument slice, NOT including "reset"
// itself (e.g. `incremental_trainer reset --yes --keep-data` -> {"--yes", "--keep-data"}).
ResetCommandArgs parse_reset_command_args(const std::vector<std::string>& args);

// TD-523: `incremental_trainer resume` exit codes, which trainer_service's supervisor reads to
// tell a pass that trained from an idle one and from a failing one.
inline constexpr int kResumeExitTrained = 0;
inline constexpr int kResumeExitFailed = 1;
inline constexpr int kResumeExitNothingPending = 3;

// How trainer_service classifies one child pass's exit code. Crashed is a signal death (128 and
// up, see ChildProcess::poll_exit()); every other code that isn't a resume outcome is Failed,
// including 127 (exec failure).
enum class PassOutcome { DidWork, Idle, Failed, Crashed };
PassOutcome classify_pass_exit(int exit_code);

// Seconds to wait before the next pass after @p consecutive_failures (>= 1) failed or crashed
// passes in a row: @p base_seconds doubled per extra failure, capped at @p max_seconds.
int failure_backoff_seconds(int consecutive_failures, int base_seconds, int max_seconds);

}  // namespace adai
