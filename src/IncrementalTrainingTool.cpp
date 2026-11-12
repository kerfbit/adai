// @adai-status: beta        (capped by TD-039; large, actively evolving — see TECHNICAL_DEBT.md)
// @adai-version: 0.37.0
// @adai-reviewed: 2026-11-12

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>
#include "Config.hpp"
#include "DataTransport.hpp"
#include "DatasetRegistry.hpp"
#include "IncrementalTrainer.hpp"
#include "IncrementalTrainerArgs.hpp"
#include "InitRng.hpp"
#include "LeJEPAEncoder.hpp"
#include "LeJEPATrainingPass.hpp"
#include "Logger.hpp"
#include "Matrix.hpp"
#include "ModelNameClient.hpp"
#include "PassHeartbeat.hpp"
#include "SignalSetup.hpp"
#include "StartupSweep.hpp"
#include "TrainerControlState.hpp"
#ifdef BUILD_TRAINER_ADMIN
#include "TrainerAdminAPI.hpp"
#endif
#include "TrainingPassFiles.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

// TD-566: the LeJEPA pass and the shared download step live in their own files now, so they can be
// tested.
using adai::cleanup_downloads;
using adai::resolve_pass_files;
using adai::run_lejepa_training_pass;

namespace {

static constexpr const char* COLOR_RESET = "\033[0m";
static constexpr const char* COLOR_INFO = "\033[1;36m";

// Graceful-stop signal handling — registered only for the actual worker process (--foreground under
// systemd, or the detached background child; see run_training_pipeline()). Not load-bearing for crash
// recovery: a real SIGSEGV from the GPU driver never reaches a handler, and a later pass reclaims its
// own claims by run id either way.
//
// TD-416: the handler only stores atomics (it used to log through spdlog and call std::exit(0), neither
// async-signal-safe, and exited 0 without a checkpoint or releasing the claimed files). The first
// SIGTERM/SIGINT asks the running pass (its TrainerControlState, registered by PassAdminApi) to drain as
// for /admin/pause: it stops at the next sample or batch, checkpoints, releases its files and exits 1.
// With no pass registered, or on a second signal, the process exits at once with status 1.
std::atomic<adai::TrainerControlState*> g_signal_control{nullptr};
std::atomic<int> g_stop_signals{0};

void signal_handler(int sig) {
    if (sig != SIGTERM && sig != SIGINT) return;
    adai::TrainerControlState* control = g_signal_control.load();
    if (g_stop_signals.fetch_add(1) == 0 && control != nullptr) {
        control->stop_signal.store(sig);
        control->paused.store(true);
        return;
    }
    ::_exit(1);
}

void print_session_history(const std::vector<TrainingSession>& sessions) {
    std::cout << COLOR_INFO << "\n📜 Session History:" << COLOR_RESET << '\n';
    std::cout << "Session | Samples | Epochs | Loss   | Val Loss | Checkpoint\n";
    std::cout << "--------|---------|--------|--------|----------|------------\n";
    for (const auto& s : sessions) {
        std::cout << std::setw(7) << s.session_id << " | " << std::setw(7) << s.samples_trained
                  << " | " << std::setw(6) << s.epochs_completed << " | " << std::setw(6)
                  << std::fixed << std::setprecision(3) << s.final_loss << " | " << std::setw(8);
        if (s.final_validation_loss == TrainingSession::kNoValidationLoss)
            std::cout << "n/a";  // TD-541: pass had no validation data
        else
            std::cout << s.final_validation_loss;
        std::cout << " | " << s.checkpoint_path << '\n';
    }
}

void print_data_registry(const adai::ServiceConfig& svc_config) {
    DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
    reg.load_registry();
    reg.load_pending_list();
    std::cout << COLOR_INFO << "\n📋 Data Registry:" << COLOR_RESET << '\n';
    std::cout << "Status  | Data File\n";
    std::cout << "--------|----------\n";
    for (const auto& f : reg.trained_files())
        std::cout << "trained | " << f << '\n';
    for (const auto& f : reg.pending_files())
        std::cout << "pending | " << f << '\n';
}

// Detaches the calling process into the background.
//
// POSIX: fork()s; the parent gets back the child PID (> 0) and should print
//        the startup banner then exit(0).  The child gets 0, calls setsid(),
//        and redirects stdin/stdout/stderr to /dev/null before returning so
//        training output flows only through the Logger file sink.
// Windows: re-launches the current command line via CreateProcess(DETACHED_PROCESS);
//          the launcher gets back the new process ID (> 0) and should print the
//          banner then exit(0).  The re-launched child detects the
//          ADAI_BACKGROUND_CHILD env var and returns 0.
//
// Returns -1 if the fork/CreateProcess call fails (caller falls back to
// running in the foreground).
long long launch_background(int argc, char* argv[]) {
#ifndef _WIN32
    (void)argc;
    (void)argv;
    pid_t pid = ::fork();
    if (pid < 0)
        return -1LL;  // fork failed
    if (pid > 0)
        return static_cast<long long>(pid);  // parent: return child PID

    // --- child ---
    ::setsid();  // become session leader; fully detach from controlling terminal

    // Redirect stdin to /dev/null
    int nr = ::open("/dev/null", O_RDONLY);
    if (nr >= 0) {
        ::dup2(nr, STDIN_FILENO);
        ::close(nr);
    }

    // Redirect stdout/stderr to /dev/null (training output goes through Logger)
    int nw = ::open("/dev/null", O_WRONLY);
    if (nw >= 0) {
        ::dup2(nw, STDOUT_FILENO);
        ::dup2(nw, STDERR_FILENO);
        ::close(nw);
    }
    return 0LL;

#else  // _WIN32
    // If this process was re-launched as the background child, clear the
    // sentinel env var and return 0 so the caller proceeds with training.
    if (::GetEnvironmentVariableA("ADAI_BACKGROUND_CHILD", nullptr, 0) > 0) {
        ::SetEnvironmentVariableA("ADAI_BACKGROUND_CHILD", nullptr);
        return 0LL;
    }

    // Set sentinel before re-launching so the child knows it is the worker.
    ::SetEnvironmentVariableA("ADAI_BACKGROUND_CHILD", "1");

    std::string cmd = ::GetCommandLineA();
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    if (!::CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE,
                          DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, nullptr, nullptr, &si,
                          &pi)) {
        ::SetEnvironmentVariableA("ADAI_BACKGROUND_CHILD", nullptr);
        return -1LL;
    }

    const long long child_pid = static_cast<long long>(::GetProcessId(pi.hProcess));
    ::CloseHandle(pi.hProcess);
    ::CloseHandle(pi.hThread);
    return child_pid;
#endif
}

// TD-172/TD-403: the per-pass admin API a trainer_service child hosts on 127.0.0.1:<--admin-port>
// for one pass; trainer_service's own listener proxies /admin/* to it while the child is alive. Shared
// by the chatbot and LeJEPA resume passes (the LeJEPA pass had none, so during it the supervisor could
// only answer from its own state). TD-416: every pass (train, retrain, resume) builds one; control() is
// always set, and the HTTP listener runs only with --admin-port in a build with the admin API. The
// destructor stops the listener.
class PassAdminApi {
   public:
    PassAdminApi(const std::optional<int>& admin_port, const adai::ServiceConfig& svc_config) {
        // TD-416: every pass has a control state, admin API or not, so a SIGTERM/SIGINT can drain it.
        control_ = std::make_shared<adai::TrainerControlState>();
        control_->auto_save_enabled = svc_config.auto_save_enabled;
        control_->auto_save_every_samples = svc_config.auto_save_every_samples;
        control_->auto_save_every_minutes = svc_config.auto_save_every_minutes;
        control_->max_sessions_to_keep = svc_config.max_sessions_to_keep;
        control_->set_model_name(svc_config.model_name);
        g_signal_control.store(control_.get());
#ifdef BUILD_TRAINER_ADMIN
        if (admin_port) {
            api_ = std::make_unique<adai::TrainerAdminAPI>(control_, "127.0.0.1", *admin_port,
                                                           svc_config.trainer_admin_dir);
            adai::TrainerAdminAPI* api = api_.get();
            thread_ = std::thread([api] {
                if (!api->start()) {
                    adai::Logger::error(
                        "TrainerAdminAPI failed to bind for this pass — the supervisor's "
                        "status/control requests will see this child as unreachable "
                        "until the next one starts");
                }
            });
            // Bounded wait for the listener to actually be up before starting the pass,
            // so a pass that finishes near-instantly (nothing pending) doesn't race
            // stop() against a start() that hasn't reached listen() yet.
            for (int waited_ms = 0; waited_ms < 2000 && !api->is_running(); waited_ms += 10) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } else if (svc_config.trainer_admin_enabled) {
            adai::Logger::debug(
                "resume: TRAINER_ADMIN_ENABLED is set but no --admin-port was passed "
                "(only the process-supervisor binary passes it) — running with no admin "
                "API for this pass");
        }
#else
        (void)svc_config;
        if (admin_port) {
            adai::Logger::warn(
                "--admin-port given but this binary was built without the admin API "
                "(cpp-httplib not found at configure time) — resume continues with no "
                "admin port");
        }
#endif
    }
    ~PassAdminApi() {
        g_signal_control.store(nullptr);
#ifdef BUILD_TRAINER_ADMIN
        if (api_) {
            api_->stop();
            if (thread_.joinable()) thread_.join();
        }
#endif
    }
    PassAdminApi(const PassAdminApi&) = delete;
    PassAdminApi& operator=(const PassAdminApi&) = delete;

    const std::shared_ptr<adai::TrainerControlState>& control() const { return control_; }

   private:
    std::shared_ptr<adai::TrainerControlState> control_;
#ifdef BUILD_TRAINER_ADMIN
    std::unique_ptr<adai::TrainerAdminAPI> api_;
    std::thread thread_;
#endif
};

// Resolve which model to train.
//
// Priority: --model CLI flag > MODEL_NAME in config > interactive pick from MNS.
// When MNS is configured (NAME_SERVICE_URL is set), the name service is the
// authoritative source of available models.  If no model can be determined,
// returns an empty string (caller should abort with an error).
#ifdef BUILD_MNS_SERVER
std::string resolve_model_name(const adai::ServiceConfig& svc_config,
                               const std::string& cli_model_name) {
    if (!cli_model_name.empty())
        return cli_model_name;
    if (!svc_config.model_name.empty())
        return svc_config.model_name;

    if (svc_config.name_service_url.empty())
        return {};

    adai::ModelNameClient mns(svc_config.name_service_url, svc_config.name_service_timeout_ms);
    std::vector<adai::ModelSummary> models;
    try {
        models = mns.list_models("", svc_config.model_role);
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to query name service at " << svc_config.name_service_url << ": "
                  << e.what() << "\n";
        return {};
    }

    if (models.empty()) {
        std::cerr << "❌ No models registered in the name service";
        if (!svc_config.model_role.empty())
            std::cerr << " for role '" << svc_config.model_role << "'";
        std::cerr << ".\n";
        return {};
    }

    // TD-416: the pick needs someone at a terminal. Under systemd or with piped input it used to read EOF
    // and fail with no reason given.
#ifndef _WIN32
    if (!::isatty(STDIN_FILENO)) {
        std::cerr << "❌ No model to train: set MODEL_NAME or pass --model <name> (the interactive pick "
                     "from the name service needs a terminal)\n";
        return {};
    }
#endif
    std::cout << "\nAvailable models from name service:\n";
    std::cout << "  #  | State        | Role       | Model Name\n";
    std::cout << "-----|--------------|------------|---------------------------\n";
    for (size_t i = 0; i < models.size(); ++i) {
        std::cout << "  " << std::setw(2) << (i + 1) << " | " << std::setw(12) << std::left
                  << models[i].state << " | " << std::setw(10) << std::left << models[i].role
                  << " | " << models[i].model_name << "\n";
    }
    std::cout << "\nSelect model [1-" << models.size() << "]: ";
    std::string input;
    std::getline(std::cin, input);
    if (input.empty())
        return {};

    const auto parsed = adai::parse_strict_int(input);
    const int choice = parsed.value_or(0);
    if (choice < 1 || choice > static_cast<int>(models.size())) {
        std::cerr << "Invalid selection.\n";
        return {};
    }
    return models[static_cast<size_t>(choice - 1)].model_name;
}
#endif

// Forks into the background, prints a startup banner in the parent, then
// initialises the logger in the child before running child_work(), which returns an exit code.
// TD-571: child_work initialises the GPU itself, after building its PassAdminApi, so the pass's
// admin API answers (and a SIGTERM drains) during GPU initialisation too.
// banner_extras are {label, value} pairs printed between Model and Log lines;
// labels are left-padded to 7 characters to align with the fixed lines.
template <typename WorkerFn>
int run_training_pipeline(int argc, char* argv[], const adai::ServiceConfig& svc_config,
                          const std::string& default_model, const std::string& log_path,
                          const std::string& title,
                          const std::vector<std::pair<std::string, std::string>>& banner_extras,
                          WorkerFn&& child_work, bool foreground = false) {
    // --foreground skips the fork entirely — child_pid == 0 is exactly the
    // existing "we are the worker" path (already used for the fork-failure
    // fallback below), so this reuses it with no new branches. Needed for
    // systemd Type=simple, which tracks/restarts whatever ExecStart's PID
    // actually is — that can't be a process that forks and exits immediately.
    // TD-431: a detached run's stdout/stderr go to /dev/null, so with no log file it needs one anyway.
    const std::string effective_log_path =
        (log_path.empty() && !foreground) ? adai::default_background_log_path() : log_path;
    const long long child_pid = foreground ? 0 : launch_background(argc, argv);
    if (child_pid > 0) {
        std::cout << "[ADAI] " << title << " — PID " << child_pid << "\n"
                  << "       Model  : " << default_model << "\n";
        for (auto [label, value] : banner_extras) {
            label.resize(7, ' ');
            std::cout << "       " << label << ": " << value << "\n";
        }
        std::cout << "       Log    : " << effective_log_path << "\n"
                  << "       Stop   : kill " << child_pid << "\n";
        return 0;
    }
    if (child_pid < 0)
        adai::Logger::warn("[background] fork failed — running in foreground");

    adai::Logger::init(adai::Logger::parse_level(svc_config.log_level),  // TD-431: it used to stay at INFO
                       {effective_log_path, svc_config.log_max_size_mb, svc_config.log_max_files}, "adai");
    adai::ignore_sighup("incremental_trainer");  // TD-283: SIGHUP used to terminate this process
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);
    return child_work();
}

int output_usage(char* argv[]) {
    std::cout << "Usage: " << argv[0] << " [--config <path>] <command> [options]\n\n";
    std::cout << "Global options:\n";
    std::cout << "  --config <path>              Path to config.trainer.conf\n";
    std::cout << "                               Search order: --config > ./config.trainer.conf > "
                 "/etc/adai/config.trainer.conf\n";
    std::cout << "                               > ./config.conf (legacy) > /etc/adai/config.conf "
                 "(legacy)\n";
    std::cout << "                               Sets model architecture, training params, "
                 "vocab/model paths\n";
    std::cout
        << "  --gpu-strategy <mode>        GPU scheduling strategy: background (default) or full\n";
    std::cout << "                               background: low-priority stream, yields to other "
                 "GPU work\n";
    std::cout << "                               full:       high-priority stream, maximises "
                 "throughput\n";
    std::cout << "                               Tip: pair 'full' with GPU_MEMORY_FRACTION=0.9\n";
    std::cout << "  --model <name>               Model name (overrides MODEL_NAME in config)\n";
    std::cout << "                               When NAME_SERVICE_URL is set and --model is\n";
    std::cout << "                               omitted, lists available models interactively\n";
    std::cout << "  --foreground                 train/retrain/resume: stay in the foreground\n";
    std::cout << "                               instead of forking+daemonizing. For process\n";
    std::cout << "                               supervisors (e.g. systemd Type=simple) that need\n";
    std::cout << "                               to track/restart the actual worker PID.\n";
    std::cout << "  --admin-port <N>             resume: host a TrainerAdminAPI on\n";
    std::cout << "                               127.0.0.1:<N> for the duration of this one pass.\n";
    std::cout << "                               For the trainer-service process supervisor only —\n";
    std::cout << "                               not meant for interactive/manual use.\n";
    std::cout << "  --objective=<name>           train: which objective to run — \"chatbot\"\n";
    std::cout << "                               (default) is the existing teacher-forcing\n";
    std::cout << "                               pipeline; \"lejepa\" pretrains the standalone\n";
    std::cout << "                               LeJEPAEncoder world model instead (requires\n";
    std::cout << "                               WORLD_MODEL_ENABLED=true in config.trainer.conf).\n";
    std::cout << "                               Note the \"=\" — unlike every other flag above.\n";
    std::cout << "                               Any other value is an error.\n";
    std::cout << "  --log-file <path>            Log to this file instead of LOG_FILE_PATH (empty LOG_FILE_PATH:\n"
                 "                               console only, or ~/.local/state/adai/log/incremental_trainer.log\n"
                 "                               when running in the background)\n";
    std::cout << "  --heartbeat-file <path>      Write this run's progress there for "
                 "trainer_service's stall\n"
                 "                               check (TRAINER_PASS_STALL_MINUTES; needs "
                 "--foreground)\n";
    std::cout << "  --seed <n>                   Replays initialization, shuffles and dropout "
                 "(overrides SEED)\n";
    std::cout << "  --dataset-kind <kind>        Overrides the dataset registry sub-pool "
                 "(encoder/decoder/\n";
    std::cout << "                               world_model/chatbot, TD-202) that would "
                 "otherwise be picked\n";
    std::cout << "                               automatically from --objective (lejepa -> "
                 "world_model,\n";
    std::cout << "                               else chatbot). This is the only override — "
                 "DATASET_KIND in\n";
    std::cout << "                               config.trainer.conf is ignored here (it only "
                 "matters to\n";
    std::cout << "                               dataset_manager); a per-invocation flag is "
                 "required so the\n";
    std::cout << "                               automatic pool separation can't be silently "
                 "defeated by a\n";
    std::cout << "                               stale config value shared across both "
                 "objectives.\n\n";
    std::cout << "Commands:\n";
    std::cout << "  init [vocab] [model]         Initialize incremental trainer\n";
    std::cout << "  train [epochs]               Train on pending data\n";
    std::cout << "  retrain [epochs]             Full retrain on all data\n";
    std::cout << "  reset                        Remove all checkpoints and rebuild model from "
                 "config\n";
    std::cout << "  resume                       Resume from last session\n";
    std::cout << "                               (TD-172: the always-on service is now a\n";
    std::cout << "                               separate binary, trainer_service, which\n";
    std::cout << "                               launches this command repeatedly as a\n";
    std::cout << "                               single-pass child — see its own --help and\n";
    std::cout << "                               TRAINER_ADMIN_*/TRAINER_CHILD_ADMIN_PORT in\n";
    std::cout << "                               config.trainer.conf.)\n";
    std::cout << "  status                       Show training status\n";
    std::cout << "  history                      Show session history\n";
    std::cout << "\nreset options:\n";
    std::cout << "  --yes                        Skip confirmation prompt\n";
    std::cout << "  --keep-data                  Preserve data registry (mark entries untrained)\n";
    std::cout << "\nExample workflow:\n";
    std::cout << "  # Initial training with custom config\n";
    std::cout << "  " << argv[0] << " --config config.trainer.conf init\n";
    std::cout << "  " << argv[0] << " --config config.trainer.conf train 5\n";
    return 1;
}

}  // namespace

static int run_main(int argc, char* argv[]) {
    // -----------------------------------------------------------------------
    // Strip the optional global flag  --config <path>  from argv so that the
    // rest of the command-dispatch logic sees a clean args list.
    // Usage:  incremental_trainer [--config <path>] <command> [args...]
    // -----------------------------------------------------------------------
    adai::IncrementalTrainerGlobalArgs cli = adai::parse_incremental_trainer_global_args(argc, argv);
    if (cli.error) {  // TD-401
        std::cerr << "Error: " << cli.error_message << "\n\n";
        return output_usage(argv);
    }
    // TD-569: report this pass's progress to trainer_service from the start, so a pass that hangs
    // even before training (GPU initialization, a first kernel compile) is seen too. Removed at
    // exit.
    std::unique_ptr<adai::PassHeartbeatWriter> heartbeat;
    if (cli.heartbeat_file)
        heartbeat = std::make_unique<adai::PassHeartbeatWriter>(*cli.heartbeat_file);
    const std::string cli_model_name = cli.model_name.value_or("");
    const bool foreground = cli.foreground;
    const std::optional<int> admin_port = cli.admin_port;
    std::vector<std::string>& args = cli.args;  // args[0] = command, args[1..] = its args

    // Load model architecture + training params from config file.
    // Priority: file < environment variables (ConfigLoader already handles this).
    // Discovery: --config > ./config.trainer.conf > /etc/adai/config.trainer.conf
    // > ./config.conf (legacy) > /etc/adai/config.conf (legacy).
    std::string config_path =
        adai::ConfigLoader::discover_config_path(cli.config_path.value_or(""), "config.trainer.conf");
    adai::ServiceConfig svc_config = adai::ConfigLoader::load(config_path);
    adai::ConfigLoader::validate_or_exit(svc_config, "incremental_trainer");  // TD-288

    // TD-368: before anything is initialized. Unset, the first use picks a seed and logs it.
    if (cli.seed) {
        svc_config.seed = *cli.seed;
    }
    if (svc_config.seed > 0) {
        adai::set_init_seed(svc_config.seed);
        adai::Logger::info("Seed {}: initialization and training choices will repeat",
                           svc_config.seed);
    }

    // TD-202/TD-204: pick the dataset registry's per-trainable-piece sub-pool for this
    // invocation — see adai::resolve_dataset_kind()'s own doc comment for why this is a pure
    // function of (objective, --dataset-kind) only, deliberately never consulting a pre-loaded
    // ServiceConfig::dataset_kind (a config-file/env DATASET_KIND is ignored here by design; it
    // remains meaningful for `dataset_manager`, which has no per-objective mapping to protect).
    svc_config.dataset_kind = adai::resolve_dataset_kind(cli.objective, cli.dataset_kind);

    // CLI --gpu-strategy overrides the config file value.
    if (cli.gpu_strategy) {
        svc_config.gpu_strategy = adai::gpu_strategy_from_string(*cli.gpu_strategy);
    }

    // GPU init is deferred for commands that fork (train/retrain/resume): the
    // child reinitialises after fork because CUDA contexts are not fork-safe.
    // For all other commands (chat, infer, status, …) we initialise here.
    const bool command_defers_init =
        !args.empty() && adai::incremental_trainer_command_defers_gpu_init(args[0]);

    auto init_gpu = [&]() {
        if (!svc_config.gpu_enabled)
            return;
#ifdef ADAI_ENABLE_GPU
        const bool low_priority = (svc_config.gpu_strategy == adai::GPUStrategy::BACKGROUND);
        if (Matrix::gpu_try_initialize(svc_config.gpu_device_id, svc_config.gpu_memory_fraction,
                                       low_priority)) {
            const char* mode =
                low_priority ? "background (low-priority stream)" : "full (high-priority stream)";
            adai::Logger::info("[GPU] GPU ready — strategy: {}. {}", mode, Matrix::gpu_info());
        } else {
#if defined(ADAI_GPU_BACKEND_SYCL)
            adai::Logger::warn(
                "[GPU] No Intel GPU device found or SYCL initialisation failed"
                " — running on CPU");
            adai::Logger::warn("{}", adai::gpu::GPUManager::probe_diagnostic());
#else
            adai::Logger::warn(
                "[GPU] No CUDA device found or initialisation failed"
                " — running on CPU");
#endif
        }
#else
        adai::Logger::warn(
            "[GPU] GPU_ENABLED is set but this binary was built without GPU support"
            " (rebuild with -DENABLE_GPU=ON for CUDA or -DENABLE_SYCL=ON for Intel Arc)");
#endif
    };

    if (!command_defers_init)
        init_gpu();

    if (args.empty()) {
        return output_usage(argv);
    }

    // -----------------------------------------------------------------------
    // Resolve model identity via name service (when configured).
    // Priority: --model CLI > MODEL_NAME config > interactive MNS selection.
    // On success, svc_config.model_name is set and (if the model has an
    // artifact) model_path is resolved from the name service.
    // -----------------------------------------------------------------------
#ifdef BUILD_MNS_SERVER
    if (!svc_config.name_service_url.empty()) {
        std::string resolved = resolve_model_name(svc_config, cli_model_name);
        if (resolved.empty()) {
            std::cerr << "❌ No model selected. Specify --model <name>, set MODEL_NAME in "
                         "config.trainer.conf, or select from the name service list.\n";
            return 1;
        }
        svc_config.model_name = resolved;

        try {
            adai::ModelNameClient mns(svc_config.name_service_url,
                                      svc_config.name_service_timeout_ms);
            auto rm = mns.resolve_model(resolved);
            if (!rm.artifact.path.empty()) {
                svc_config.model_path = rm.artifact.path;
            }
            // MNS is the authoritative source for run_group when the model has
            // been registered with one — falls back to local RUN_GROUP config /
            // SESSION_DIR-basename derivation (DatasetRegistry.cpp) otherwise.
            // Every DatasetRegistry::make_config(svc_config) call site downstream
            // picks this up automatically since they all read svc_config.run_group
            // fresh — see IncrementalTrainer::begin_run()/train_on_files()/etc.
            if (!rm.run_group.empty()) {
                svc_config.run_group = rm.run_group;
                adai::Logger::deferred(adai::Logger::Level::INFO,
                                       "[MNS] run_group resolved from MNS: '" + rm.run_group + "'");
            }
        } catch (const std::exception& e) {
            // TD-571: said why (it was a silent catch (...)). A model with no artifact yet lands
            // here too, and then trains from scratch; MNS unreachable means the local paths are
            // used.
            adai::Logger::deferred(adai::Logger::Level::WARN,
                                   std::string("[MNS] Could not resolve '") + resolved + "' (" +
                                       e.what() + "); using the local MODEL_PATH and RUN_GROUP");
        }

        // MNS is the authoritative source for architecture — see CLAUDE.md
        // "Configuration". Falls back to config.trainer.conf's D_MODEL etc. when
        // the lookup fails or the model isn't registered yet (bootstrap case).
        try {
            adai::ModelNameClient mns(svc_config.name_service_url,
                                      svc_config.name_service_timeout_ms);
            if (auto arch = mns.get_architecture(resolved)) {
                svc_config.d_model = arch->d_model;
                svc_config.num_heads = arch->num_heads;
                svc_config.d_ff = arch->d_ff;
                svc_config.num_encoder_layers = arch->num_encoder_layers;
                svc_config.num_decoder_layers = arch->num_decoder_layers;
                svc_config.max_seq_length = arch->max_seq_length;
                adai::Logger::deferred(
                    adai::Logger::Level::INFO,
                    "[MNS] Architecture resolved from MNS for model '" + resolved + "'");
            } else {
                adai::Logger::deferred(adai::Logger::Level::INFO,
                                       "[MNS] No architecture on record for '" + resolved +
                                           "'; using local config architecture");
            }
        } catch (const std::exception& e) {
            adai::Logger::deferred(adai::Logger::Level::WARN,
                                   std::string("[MNS] Architecture lookup failed (") + e.what() +
                                       "); using local config architecture");
        }

        // TD-196: MNS-first world-model/hippocampal resolution, closing the gap flagged in
        // TD-184 — previously WORLD_MODEL_*/HIPPOCAMPAL_* came exclusively from local config,
        // kept in sync with the chatbot server's own config file purely by operator discipline.
        // When this chatbot has a world model linked via POST /models/{name}/link-world-model,
        // its connection settings and the world model's own resolved architecture + artifact
        // path now override local config; falls back to local config entirely when unlinked or
        // MNS is unreachable.
        try {
            adai::ModelNameClient mns(svc_config.name_service_url,
                                      svc_config.name_service_timeout_ms);
            if (auto conn = mns.get_connection(resolved); conn && !conn->world_model_name.empty()) {
                const adai::ResolvedModel wm_resolved = mns.resolve_model(conn->world_model_name);
                const auto wm_arch = mns.get_architecture(conn->world_model_name);
                if (!wm_resolved.artifact.path.empty() && wm_arch) {
                    svc_config.world_model_enabled = true;
                    svc_config.world_model_artifact_path = wm_resolved.artifact.path;
                    svc_config.world_model_d_model = wm_arch->d_model;
                    svc_config.world_model_num_layers = wm_arch->num_encoder_layers;
                    svc_config.world_model_num_heads = wm_arch->num_heads;
                    svc_config.world_model_d_ff = wm_arch->d_ff;
                    svc_config.world_model_inject_every_n_layers =
                        conn->world_model_inject_every_n_layers;
                    svc_config.hippocampal_memory_enabled = conn->hippocampal_memory_enabled;
                    svc_config.hippocampal_memory_capacity = conn->hippocampal_memory_capacity;
                    svc_config.hippocampal_repetition_alpha = conn->hippocampal_repetition_alpha;
                    svc_config.hippocampal_repetition_decay = conn->hippocampal_repetition_decay;
                    svc_config.hippocampal_cross_reference_alpha =
                        conn->hippocampal_cross_reference_alpha;
                    svc_config.hippocampal_association_decay = conn->hippocampal_association_decay;
                    // Only sigreg_num_sketches carries over here — it's structural (shapes the
                    // SIGReg module's own internal sketch matrices, so it must match whatever the
                    // checkpoint being load()ed was saved with). sigreg_lambda is purely a
                    // training-time gradient weight (LeJEPAEncoder::train_step()) with no effect
                    // on a world model attached here frozen — IncrementalTrainer::
                    // maybe_attach_world_model() always calls set_requires_grad(false), and
                    // IncrementalConfig itself has no world_model_sigreg_lambda field to carry it
                    // to that call site even if it did matter. Fetching it here would be dead
                    // weight, not a bug fix.
                    if (auto wm_conn = mns.get_connection(conn->world_model_name)) {
                        svc_config.world_model_sigreg_num_sketches = wm_conn->sigreg_num_sketches;
                    }
                    adai::Logger::deferred(adai::Logger::Level::INFO,
                                           "[MNS] World model '" + conn->world_model_name +
                                               "' resolved from MNS link (artifact='" +
                                               wm_resolved.artifact.path + "')");
                } else {
                    adai::Logger::deferred(
                        adai::Logger::Level::INFO,
                        "[MNS] World model '" + conn->world_model_name +
                            "' linked but not yet resolvable (no artifact/architecture) — "
                            "using local config");
                }
            }
        } catch (const std::exception& e) {
            adai::Logger::deferred(adai::Logger::Level::WARN,
                                   std::string("[MNS] World model connection lookup failed (") +
                                       e.what() + "); using local config");
        }
    }
#endif

    // -----------------------------------------------------------------------
    // Resolve vocab/model paths: prefer ServiceConfig paths, fall back to
    // conventional local file names so the tool still works without a config.
    // -----------------------------------------------------------------------
    std::string default_vocab = svc_config.vocab_path.empty() ? "vocab.txt" : svc_config.vocab_path;
    std::string default_model =
        svc_config.model_path.empty() ? "chatbot_model.bin" : svc_config.model_path;

    std::string command = args[0];

    // TD-431: --log-file (what trainer_service gives each pass), else LOG_FILE_PATH. Empty means console only,
    // or the per-user default file for a detached run (run_training_pipeline()). It used to fall back to
    // chatbot_server.log in the working directory, the same file trainer_service wrote and rotated.
    const std::string training_log_path = cli.log_file.value_or(svc_config.log_file_path);

    // TD-404: without MNS, a chatbot pass's run id (claimed-file ownership) is kept in
    // SESSION_DIR/chatbot_run_id, unique per host and stable across restarts, so a restarted pass
    // reclaims what a crashed one claimed. It used to be hostname prefix + PID, new every run and able
    // to collide across hosts. RUN_ID still wins, and an MNS-allocated run id comes first in the passes.
    if (cli.objective != "lejepa" && svc_config.run_id.empty() &&
        (command == "train" || command == "retrain" || command == "resume")) {
        svc_config.run_id = adai::persistent_run_id("", svc_config.session_dir, "chatbot");
    }

    // TD-183/TD-403: what `--objective=lejepa` train and resume need. TD-548: the checkpoint goes to
    // WORLD_MODEL_ARTIFACT_PATH when set (it used to be ignored), else <session_dir>/world_model_checkpoint,
    // which no longer collides with the standalone `world_model` dataset pool.
    const std::string world_model_dir = !svc_config.world_model_artifact_path.empty()
                                            ? svc_config.world_model_artifact_path
                                            : LeJEPAEncoder::default_checkpoint_dir(svc_config.session_dir);
    const auto lejepa_enabled = [&svc_config]() {
        if (svc_config.world_model_enabled) return true;
        std::cerr << "❌ --objective=lejepa requires WORLD_MODEL_ENABLED=true in "
                     "config.trainer.conf (or the WORLD_MODEL_ENABLED env var)\n";
        return false;
    };

    if (command == "init") {
        // init <vocab> <model>  -- paths from args override config
        std::string vocab_path = (args.size() >= 2) ? args[1] : default_vocab;
        std::string model_path = (args.size() >= 3) ? args[2] : default_model;

        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        // TD-416: AUTO_SAVE_* as configured (init and retrain used to force 30 min / 1,000 samples).

        IncrementalTrainer trainer(vocab_path, model_path, config);

        std::cout << "✅ Incremental trainer initialized\n";
        std::cout << "📁 Session directory: " << config.session_dir << "\n";
        std::cout << "   d_model=" << config.base_config.d_model
                  << "  d_ff=" << config.base_config.d_ff
                  << "  heads=" << config.base_config.num_heads
                  << "  enc_layers=" << config.base_config.num_encoder_layers
                  << "  dec_layers=" << config.base_config.num_decoder_layers
                  << "  max_seq=" << config.base_config.max_seq_length << "\n";

    } else if (command == "train") {
        // TD-416: checked; `train abc` used to abort through an uncaught std::stoi (same in retrain).
        std::string epochs_error;
        const auto parsed_epochs = adai::parse_epochs_arg(args, svc_config.num_epochs, epochs_error);
        if (!parsed_epochs) {
            std::cerr << "❌ " << epochs_error << "\n";
            return 1;
        }
        const int epochs = *parsed_epochs;

        // TD-413: counted through DatasetRegistry in both modes, so it reads the same kind
        // sub-pool (SESSION_DIR/<kind>/ locally, TD-202) and model filter the pass will acquire
        // from. The standalone check used to read SESSION_DIR/pending_files.txt directly, so a
        // correctly queued `--kind chatbot` pool read as empty.
        size_t pre_fork_pending_count = 0;
        {
            DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
            reg.load_pending_list();
            pre_fork_pending_count = reg.count_acquirable();
        }
        if (pre_fork_pending_count == 0) {
            std::cout << "⚠️  No pending data for this model in the '"
                      << (svc_config.dataset_kind.empty() ? "legacy" : svc_config.dataset_kind)
                      << "' pool. Use dataset_manager to queue training data.\n";
            return 1;
        }

        const std::string log_path =
            training_log_path;  // TD-431

        // TD-183: --objective=lejepa reuses the pending-data check above and the same
        // run_training_pipeline() fork/background/logging wrapper, but trains the standalone
        // LeJEPAEncoder world model instead of the chatbot's own ChatbotTrainer/IncrementalTrainer
        // pipeline below. This is the ONLY change to the "train" command — every existing
        // invocation (no --objective, or --objective=chatbot) falls through to the unmodified
        // code beneath unchanged (Action Item 3).
        if (cli.objective == "lejepa") {
            if (!lejepa_enabled()) return 1;
            return run_training_pipeline(
                argc, argv, svc_config, world_model_dir, log_path,
                "LeJEPA world-model pretraining started in background",
                {{"Data", std::to_string(pre_fork_pending_count) + " pending file(s)"},
                 {"Epochs", std::to_string(epochs)}},
                [&]() -> int {
                    PassAdminApi pass(std::nullopt, svc_config);  // TD-416: a SIGTERM drains the pass
                    init_gpu();                                   // TD-571: after the admin API
                    return run_lejepa_training_pass(svc_config, default_vocab, world_model_dir, epochs,
                                                    pass.control());
                },
                foreground);
        }

        return run_training_pipeline(
            argc, argv, svc_config, default_model, log_path, "Training started in background",
            {{"Data", std::to_string(pre_fork_pending_count) + " pending file(s)"},
             {"Epochs", std::to_string(epochs)}},
            [&]() -> int {
                // TD-571: the control state (and any admin API) before GPU initialisation and the
                // model.
                PassAdminApi pass(std::nullopt, svc_config);  // TD-416: a SIGTERM drains the pass
                init_gpu();
                IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
                config.base_config.num_epochs = epochs;
                // TD-414: with no vocabulary yet, the first pass builds it from its data (VOCAB_BUILD_SIZE).
                IncrementalTrainer trainer(default_vocab, default_model, config);
                trainer.set_control_state(pass.control());

                // MNS is the definitive source for run_id (continuing the
                // model's current run — not a retrain); falls back to the
                // local hostname+pid/RUN_ID-derived value only when MNS isn't
                // configured. The same run_id is then used for dataset-registry
                // file ownership below, unifying both systems' notion of "run".
                std::string run_id = trainer.begin_run(/*is_retrain=*/false);
                if (run_id.empty())
                    run_id = adai::derive_run_id(svc_config.run_id);

                DatasetConfig dcfg = DatasetRegistry::make_config(svc_config);
                DatasetRegistry reg(dcfg);
                reg.load_registry();

                // Startup stale-file sweep (conditions A-expired, B, C, D, G)
                startup_sweep(reg, run_id, dcfg.download_dir);  // loads the pending list (TD-512)

                auto resp = reg.acquire_pending(run_id);
                if (resp.files.empty()) {
                    adai::Logger::warn("No pending data files to train on");
                    return 1;
                }

                // Resolve local file paths — FTP download or direct access (TD-205 ranges kept)
                std::vector<PendingFileRange> local_ranges;
                std::vector<fs::path> downloaded_paths;
                if (!resolve_pass_files(resp, dcfg, local_ranges, downloaded_paths)) {
                    reg.release_pending(run_id, resp.file_ranges());  // TD-333: exact units
                    return 1;
                }

                const bool ok = trainer.train_on_files(local_ranges, epochs);
                // TD-333: exact units (whole files and segments)
                const auto reg_paths = resp.file_ranges();

                if (ok) {
                    // TD-334: pairs loaded per file (local_ranges is 1:1 with reg_paths); these
                    // used to be all zeros.
                    reg.mark_trained(run_id, reg_paths, trainer.last_pass_pair_counts());
                    trainer.print_training_summary();
                } else {
                    reg.release_pending(run_id, reg_paths);
                }

                cleanup_downloads(downloaded_paths);
                return ok ? 0 : 1;
            },
            foreground);

    } else if (command == "retrain") {
        std::string epochs_error;  // TD-416
        const auto parsed_epochs = adai::parse_epochs_arg(args, svc_config.num_epochs, epochs_error);
        if (!parsed_epochs) {
            std::cerr << "❌ " << epochs_error << "\n";
            return 1;
        }
        const int epochs = *parsed_epochs;

        int data_file_count = 0;
        {
            // TD-413: through DatasetRegistry in both modes, so the kind sub-pool is used.
            DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
            reg.load_registry();
            reg.load_pending_list();
            data_file_count =
                static_cast<int>(reg.trained_files().size() + reg.count_acquirable());
        }

        const std::string log_path =
            training_log_path;  // TD-431

        return run_training_pipeline(
            argc, argv, svc_config, default_model, log_path, "Full retrain started in background",
            {{"Data", std::to_string(data_file_count) + " file(s)"},
             {"Epochs", std::to_string(epochs)}},
            [&]() -> int {
                PassAdminApi pass(std::nullopt, svc_config);  // TD-416: a SIGTERM drains the pass
                init_gpu();                                   // TD-571: after the control state
                IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
                config.base_config.num_epochs = epochs;
                // TD-416: AUTO_SAVE_* as configured (this used to force 30 min / 1,000 samples).

                IncrementalTrainer trainer(default_vocab, default_model, config);
                trainer.reset_model_for_config();
                trainer.set_control_state(pass.control());

                // MNS is the definitive source for run_id — retrain always
                // requests a fresh run (new_run=true), falling back to the
                // local hostname+pid/RUN_ID-derived value only when MNS isn't
                // configured. The same run_id is then used for dataset-registry
                // file ownership below, unifying both systems' notion of "run".
                std::string run_id = trainer.begin_run(/*is_retrain=*/true);
                if (run_id.empty())
                    run_id = adai::derive_run_id(svc_config.run_id);

                DatasetConfig dcfg = DatasetRegistry::make_config(svc_config);
                DatasetRegistry reg(dcfg);
                reg.load_registry();

                startup_sweep(reg, run_id, dcfg.download_dir);  // loads the pending list (TD-512)

                auto trained_units = reg.trained_units();  // TD-546
                auto pending_resp = reg.acquire_pending(run_id);

                // For retrain: already-trained files are read directly by path (they
                // live on the registry machine which already trained them, so no FTP
                // needed for the trained portion).  Only newly acquired pending files
                // may need FTP download.
                // Each trained unit is read as it was trained, so a partly trained segmented file's
                // pending segments, acquired below, aren't also read here (TD-546).
                // TODO: See TD-411 in TECHNICAL_DEBT.md - trained files are read by registry path, never downloaded, so
                // retrain against a remote FTP registry can't read them (only pending files are fetched)
                std::vector<PendingFileRange> local_ranges = std::move(trained_units);
                std::vector<fs::path> downloaded_paths;
                if (!pending_resp.files.empty()) {
                    std::vector<PendingFileRange> pending_ranges;
                    if (!resolve_pass_files(pending_resp, dcfg, pending_ranges,
                                            downloaded_paths)) {
                        reg.release_pending(run_id, pending_resp.file_ranges());  // TD-333
                        return 1;
                    }
                    local_ranges.insert(local_ranges.end(), pending_ranges.begin(),
                                        pending_ranges.end());
                }

                if (local_ranges.empty()) {
                    adai::Logger::warn("No data files to retrain on");
                    return 1;
                }
                adai::Logger::info("Starting full retrain on {} data file(s)", local_ranges.size());

                const bool ok = trainer.retrain_on_files(local_ranges, epochs);
                const auto pending_reg_paths = pending_resp.file_ranges();  // TD-333

                if (ok) {
                    if (!pending_reg_paths.empty()) {
                        // TD-334: local_ranges holds the trained files first, then the pending
                        // ones, so the pending files' counts are the trailing entries.
                        const auto& all_counts = trainer.last_pass_pair_counts();
                        std::vector<int> counts(pending_reg_paths.size(), 0);
                        const std::size_t offset = all_counts.size() >= counts.size()
                                                       ? all_counts.size() - counts.size()
                                                       : 0;
                        for (std::size_t i = 0; i < counts.size() && offset + i < all_counts.size();
                             ++i)
                            counts[i] = all_counts[offset + i];
                        reg.mark_trained(run_id, pending_reg_paths, counts);
                    }
                    trainer.print_training_summary();
                } else {
                    if (!pending_reg_paths.empty())
                        reg.release_pending(run_id, pending_reg_paths);
                    std::cerr << "❌ Full retrain failed\n";
                }

                cleanup_downloads(downloaded_paths);
                return ok ? 0 : 1;
            },
            foreground);

    } else if (command == "reset") {
        // Parse reset-specific flags from remaining args
        const std::vector<std::string> reset_flags(args.begin() + 1, args.end());
        const adai::ResetCommandArgs reset_args = adai::parse_reset_command_args(reset_flags);
        if (!reset_args.error.empty()) {  // TD-571: nothing is reset
            std::cerr << "❌ " << reset_args.error << "\n";
            return 1;
        }
        const bool auto_yes = reset_args.yes;
        const bool keep_data = reset_args.keep_data;

        // Show what will happen
        std::cout << "\n⚠️  RESET will:\n";
        std::cout << "   • Remove all checkpoint .bin files from the session directory\n";
        std::cout << "   • Back up '" << default_model << "' to '" << default_model << ".bak'\n";
        std::cout << "   • Clear session history and pending-data list\n";
        if (keep_data) {
            std::cout << "   • Preserve data registry (entries marked untrained for re-training)\n";
        } else {
            std::cout << "   • Delete the data registry\n";
        }
        std::cout << "   • Rebuild model architecture from config:\n";
        std::cout << "       d_model=" << svc_config.d_model << "  d_ff=" << svc_config.d_ff
                  << "  heads=" << svc_config.num_heads
                  << "  enc_layers=" << svc_config.num_encoder_layers
                  << "  dec_layers=" << svc_config.num_decoder_layers
                  << "  max_seq=" << svc_config.max_seq_length << "\n\n";

        if (!auto_yes) {
            std::cout << "Continue? [y/N] ";
            std::string answer;
            std::getline(std::cin, answer);
            if (answer.empty() || (answer[0] != 'y' && answer[0] != 'Y')) {
                std::cout << "Reset cancelled.\n";
                return 0;
            }
        }

        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);

        if (trainer.reset_all(keep_data)) {
            std::cout << "✅ Reset complete. Model rebuilt from config.\n";
            if (!keep_data) {
                std::cout << "   Use DatasetManager to queue new training data.\n";
            } else {
                std::cout << "   All previous data marked untrained — run 'retrain' to use it.\n";
            }
        } else {
            std::cerr << "❌ Reset failed\n";
            return 1;
        }

    } else if (command == "resume") {
        const std::string log_path =
            training_log_path;  // TD-431

        // TD-403: trainer_service now forwards --objective, so resume (every service pass) runs the
        // world-model pass too; it used to run chatbot training whatever the objective. One pass over
        // whatever is pending, EPOCHS epochs, with resume's exit codes (0 trained, 3 nothing pending,
        // 1 failed). It hosts the same per-pass admin API as the chatbot pass (status, checkpoint, pause).
        if (cli.objective == "lejepa") {
            if (!lejepa_enabled()) return adai::kResumeExitFailed;
            return run_training_pipeline(
                argc, argv, svc_config, world_model_dir, log_path,
                "LeJEPA world-model resume started in background", {},
                [&]() -> int {
                    PassAdminApi admin(admin_port, svc_config);  // TD-403: status, checkpoint, pause
                    init_gpu();  // TD-571: after the admin API, so trainer_service sees this pass
                    return run_lejepa_training_pass(svc_config, default_vocab, world_model_dir,
                                                    svc_config.num_epochs, admin.control());
                },
                foreground);
        }

        return run_training_pipeline(
            argc, argv, svc_config, default_model, log_path, "Resume started in background", {},
            [&]() -> int {
                // TD-172: --admin-port <N>, set only by trainer_service when it launches this as a
                // single-pass child (always with --foreground, so this process IS the worker),
                // hosts the per-pass admin API (PassAdminApi above) for the duration of this one
                // pass. TD-571: started first, so it answers during GPU initialisation and the
                // model's construction (trainer_service's /admin/status used to show only a 503
                // until then).
                PassAdminApi admin(admin_port, svc_config);
                init_gpu();
                IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
                IncrementalTrainer trainer(default_vocab, default_model, config);
                trainer.set_control_state(admin.control());

                // MNS is the definitive source for run_id — resume continues
                // the model's current run (not a retrain); falls back to the
                // local hostname+pid/RUN_ID-derived value inside
                // resume_last_session() when MNS isn't configured.
                const std::string resume_run_id = trainer.begin_run(/*is_retrain=*/false);
                // TD-411: resume_last_session() (every trainer_service pass) gets the same FTP
                // download step as train; it used to train on the registry host's own paths,
                // which a remote FTP registry's trainer can't open.
                const DatasetConfig resume_dcfg = DatasetRegistry::make_config(svc_config);
                {
                    // Same stale-download sweep train runs (TD-512), now that resume downloads.
                    DatasetRegistry sweep_reg(resume_dcfg);
                    startup_sweep(sweep_reg,
                                  resume_run_id.empty() ? adai::derive_run_id(svc_config.run_id)
                                                        : resume_run_id,
                                  resume_dcfg.download_dir);
                }
                std::vector<fs::path> resume_downloads;
                const auto result = trainer.resume_last_session(
                    [&](const AcquireResponse& resp, std::vector<PendingFileRange>& ranges) {
                        return resolve_pass_files(resp, resume_dcfg, ranges, resume_downloads);
                    });
                cleanup_downloads(resume_downloads);

                // TD-523: distinct exit codes, so trainer_service can tell an idle pass (nothing
                // pending) from a failing one; it used to see exit 1 for both.
                if (result == IncrementalTrainer::ResumeResult::NothingPending)
                    return adai::kResumeExitNothingPending;
                if (result == IncrementalTrainer::ResumeResult::Failed)
                    return adai::kResumeExitFailed;
                adai::Logger::info("Resumed from last session; latest checkpoint: {}",
                                   trainer.get_latest_checkpoint());
                return 0;
            },
            foreground);

    } else if (command == "status") {
        // TD-416: read-only. It used to build an IncrementalTrainer just to print, which loaded a checkpoint,
        // ran the deleting cleanup_dead_sessions() and (TD-414) aborted without a vocabulary (also history).
        const auto history = IncrementalTrainer::read_session_history(svc_config.session_dir).value_or(
            std::vector<TrainingSession>{});
        long long total_samples = 0;
        double total_hours = 0.0;
        for (const auto& s : history) {
            total_samples += s.samples_trained;
            total_hours += std::chrono::duration<double, std::ratio<3600>>(s.end_time - s.start_time).count();
        }
        std::cout << COLOR_INFO << "\n📊 Incremental Training Summary" << COLOR_RESET << '\n';
        std::cout << "  Session dir   : " << svc_config.session_dir << '\n';
        std::cout << "  Sessions      : " << history.size() << '\n';
        std::cout << "  Total samples : " << total_samples << '\n';
        std::cout << "  Total time    : " << std::fixed << std::setprecision(2) << total_hours << " h\n";
        if (!history.empty()) {
            std::cout << "  Last session  : #" << history.back().session_id << "  "
                      << history.back().checkpoint_path << '\n';
        }

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();
        std::cout << "\n📋 Pending data files:\n";
        for (const auto& file : reg.pending_files()) {
            std::cout << "  - " << file << "\n";
        }

    } else if (command == "history") {
        print_session_history(IncrementalTrainer::read_session_history(svc_config.session_dir).value_or(
            std::vector<TrainingSession>{}));
        print_data_registry(svc_config);

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        std::cerr << "Run '" << argv[0] << "' without arguments for help\n";
        return 1;
    }

    return 0;
}

// TD-414: an exception from any command (a missing or unreadable vocabulary, a bad checkpoint...) ends
// with a clear message and exit 1. It used to reach std::terminate, which printed "terminate called
// after throwing an instance of ..." and aborted with exit 134.
int main(int argc, char* argv[]) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& e) {
        adai::Logger::error("incremental_trainer: {}", e.what());
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
