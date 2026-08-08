#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>
#include "Config.hpp"
#include "DataTransport.hpp"
#include "DatasetRegistry.hpp"
#include "IncrementalTrainer.hpp"
#include "Logger.hpp"
#include "Matrix.hpp"
#include "ModelNameClient.hpp"
#include "StartupSweep.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

static constexpr const char* COLOR_RESET = "\033[0m";
static constexpr const char* COLOR_INFO = "\033[1;36m";

void print_session_history(const std::vector<TrainingSession>& sessions) {
    std::cout << COLOR_INFO << "\n📜 Session History:" << COLOR_RESET << '\n';
    std::cout << "Session | Samples | Epochs | Loss   | Val Loss | Checkpoint\n";
    std::cout << "--------|---------|--------|--------|----------|------------\n";
    for (const auto& s : sessions) {
        std::cout << std::setw(7) << s.session_id << " | " << std::setw(7) << s.samples_trained
                  << " | " << std::setw(6) << s.epochs_completed << " | " << std::setw(6)
                  << std::fixed << std::setprecision(3) << s.final_loss << " | " << std::setw(8)
                  << s.final_validation_loss << " | " << s.checkpoint_path << '\n';
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

// ── Download directory helpers ────────────────────────────────────────────────

static void cleanup_downloads(const std::vector<fs::path>& local_paths) {
    for (const auto& p : local_paths) {
        if (!p.empty() && fs::exists(p)) {
            std::error_code ec;
            fs::remove(p, ec);
            if (ec) {
                adai::Logger::warn("[cleanup] Failed to remove '{}': {}", p.string(), ec.message());
            }
        }
    }
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

    int choice = 0;
    try {
        choice = std::stoi(input);
    } catch (...) {
        return {};
    }
    if (choice < 1 || choice > static_cast<int>(models.size())) {
        std::cerr << "Invalid selection.\n";
        return {};
    }
    return models[static_cast<size_t>(choice - 1)].model_name;
}
#endif

// Forks into the background, prints a startup banner in the parent, then
// initialises the logger and GPU in the child before running child_work().
// init_gpu_fn and child_work are callables; child_work returns an exit code.
// banner_extras are {label, value} pairs printed between Model and Log lines;
// labels are left-padded to 7 characters to align with the fixed lines.
template <typename InitGpuFn, typename WorkerFn>
int run_training_pipeline(int argc, char* argv[], const adai::ServiceConfig& svc_config,
                          const std::string& default_model, const std::string& log_path,
                          const std::string& title,
                          const std::vector<std::pair<std::string, std::string>>& banner_extras,
                          InitGpuFn&& init_gpu_fn, WorkerFn&& child_work) {
    const long long child_pid = launch_background(argc, argv);
    if (child_pid > 0) {
        std::cout << "[ADAI] " << title << " — PID " << child_pid << "\n"
                  << "       Model  : " << default_model << "\n";
        for (auto [label, value] : banner_extras) {
            label.resize(7, ' ');
            std::cout << "       " << label << ": " << value << "\n";
        }
        std::cout << "       Log    : " << log_path << "\n"
                  << "       Stop   : kill " << child_pid << "\n";
        return 0;
    }
    if (child_pid < 0)
        adai::Logger::warn("[background] fork failed — running in foreground");

    adai::Logger::init(adai::Logger::Level::INFO,
                       {log_path, svc_config.log_max_size_mb, svc_config.log_max_files}, "adai");
    init_gpu_fn();
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
    std::cout << "                               omitted, lists available models interactively\n\n";
    std::cout << "Commands:\n";
    std::cout << "  init [vocab] [model]         Initialize incremental trainer\n";
    std::cout << "  train [epochs]               Train on pending data\n";
    std::cout << "  retrain [epochs]             Full retrain on all data\n";
    std::cout << "  reset                        Remove all checkpoints and rebuild model from "
                 "config\n";
    std::cout << "  resume                       Resume from last session\n";
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

int main(int argc, char* argv[]) {
    // -----------------------------------------------------------------------
    // Strip the optional global flag  --config <path>  from argv so that the
    // rest of the command-dispatch logic sees a clean args list.
    // Usage:  incremental_trainer [--config <path>] <command> [args...]
    // -----------------------------------------------------------------------
    std::string config_path;
    std::string gpu_strategy_override;
    std::string cli_model_name;
    std::vector<std::string> args;  // args[0] = command, args[1..] = its args

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (a == "--gpu-strategy" && i + 1 < argc) {
            gpu_strategy_override = argv[++i];
        } else if (a == "--model" && i + 1 < argc) {
            cli_model_name = argv[++i];
        } else {
            args.push_back(a);
        }
    }

    // Load model architecture + training params from config file.
    // Priority: file < environment variables (ConfigLoader already handles this).
    // Discovery: --config > ./config.trainer.conf > /etc/adai/config.trainer.conf
    // > ./config.conf (legacy) > /etc/adai/config.conf (legacy).
    config_path = adai::ConfigLoader::discover_config_path(config_path, "config.trainer.conf");
    adai::ServiceConfig svc_config = adai::ConfigLoader::load(config_path);

    // CLI --gpu-strategy overrides the config file value.
    if (!gpu_strategy_override.empty()) {
        svc_config.gpu_strategy = adai::gpu_strategy_from_string(gpu_strategy_override);
    }

    // GPU init is deferred for commands that fork (train/retrain/resume): the
    // child reinitialises after fork because CUDA contexts are not fork-safe.
    // For all other commands (chat, infer, status, …) we initialise here.
    const bool command_forks =
        (!args.empty() && (args[0] == "train" || args[0] == "retrain" || args[0] == "resume"));

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

    if (!command_forks)
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
        } catch (...) {
            // Model exists but has no artifact yet (initializing state) — train from scratch.
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
                std::cout << "[MNS] Architecture resolved from MNS for model '" << resolved
                          << "'\n";
            } else {
                std::cout << "[MNS] No architecture on record for '" << resolved
                          << "'; using local config architecture\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "[MNS] Architecture lookup failed (" << e.what()
                      << "); using local config architecture\n";
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

    if (command == "init") {
        // init <vocab> <model>  -- paths from args override config
        std::string vocab_path = (args.size() >= 2) ? args[1] : default_vocab;
        std::string model_path = (args.size() >= 3) ? args[2] : default_model;

        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
        config.auto_save_enabled = true;
        config.auto_save_every_minutes = 30;
        config.auto_save_every_samples = 1000;

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
        int epochs = (args.size() >= 2) ? std::stoi(args[1]) : svc_config.num_epochs;

        size_t pre_fork_pending_count = 0;
        if (!svc_config.registry_server_url.empty()) {
            DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
            reg.load_pending_list();
            pre_fork_pending_count = reg.pending_files().size();
        } else {
            std::ifstream pf(svc_config.session_dir + "/pending_files.txt");
            std::string line;
            while (std::getline(pf, line))
                if (!line.empty())
                    ++pre_fork_pending_count;
        }
        if (pre_fork_pending_count == 0) {
            std::cout << "⚠️  No pending data. Use DatasetManager to queue training data.\n";
            return 1;
        }

        const std::string log_path =
            svc_config.log_file_path.empty() ? "chatbot_server.log" : svc_config.log_file_path;

        return run_training_pipeline(
            argc, argv, svc_config, default_model, log_path, "Training started in background",
            {{"Data", std::to_string(pre_fork_pending_count) + " pending file(s)"},
             {"Epochs", std::to_string(epochs)}},
            init_gpu, [&]() -> int {
                IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
                config.base_config.num_epochs = epochs;
                config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
                IncrementalTrainer trainer(default_vocab, default_model, config);

                // MNS is the definitive source for run_id (continuing the
                // model's current run — not a retrain); falls back to the
                // local hostname+pid/RUN_ID-derived value only when MNS isn't
                // configured. The same run_id is then used for dataset-registry
                // file ownership below, unifying both systems' notion of "run".
                std::string run_id = trainer.begin_run(/*is_retrain=*/false);
                if (run_id.empty())
                    run_id = derive_run_id(svc_config.run_id);

                DatasetConfig dcfg = DatasetRegistry::make_config(svc_config);
                DatasetRegistry reg(dcfg);
                reg.load_registry();

                // Startup stale-file sweep (conditions A-expired, B, C, D, G)
                startup_sweep(reg, run_id, dcfg.download_dir);

                auto resp = reg.acquire_pending(run_id);
                if (resp.files.empty()) {
                    adai::Logger::warn("No pending data files to train on");
                    return 1;
                }

                // Resolve local file paths — FTP download or direct access
                std::vector<std::string> local_paths;
                std::vector<fs::path> downloaded_paths;
                const bool use_ftp = !resp.ftp_server_host.empty() && !dcfg.download_dir.empty();
                if (use_ftp) {
                    const std::size_t warn_bytes =
                        static_cast<std::size_t>(dcfg.large_file_warn_threshold_mb) * 1024ULL *
                        1024ULL;
                    try {
                        DataTransport dt;
                        downloaded_paths = dt.fetch_all(resp, dcfg.download_dir,
                                                        dcfg.max_parallel_downloads, warn_bytes);
                        for (const auto& p : downloaded_paths)
                            local_paths.push_back(p.string());
                    } catch (const std::exception& ex) {
                        adai::Logger::error("[DataTransport] Download failed: {}", ex.what());
                        reg.release_pending(run_id, resp.registry_paths());
                        return 1;
                    }
                } else {
                    for (const auto& f : resp.files)
                        local_paths.push_back(f.registry_path);
                }

                const bool ok = trainer.train_on_files(local_paths, epochs);
                const auto reg_paths = resp.registry_paths();

                if (ok) {
                    std::vector<int> counts(reg_paths.size(), 0);
                    reg.mark_trained(run_id, reg_paths, counts);
                    trainer.print_training_summary();
                } else {
                    reg.release_pending(run_id, reg_paths);
                }

                if (use_ftp)
                    cleanup_downloads(downloaded_paths);
                return ok ? 0 : 1;
            });

    } else if (command == "retrain") {
        int epochs = (args.size() >= 2) ? std::stoi(args[1]) : svc_config.num_epochs;

        int data_file_count = 0;
        if (!svc_config.registry_server_url.empty()) {
            DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
            reg.load_registry();
            reg.load_pending_list();
            data_file_count =
                static_cast<int>(reg.trained_files().size() + reg.pending_files().size());
        } else {
            std::ifstream pf(svc_config.session_dir + "/pending_files.txt");
            std::string line;
            while (std::getline(pf, line))
                if (!line.empty())
                    ++data_file_count;
        }

        const std::string log_path =
            svc_config.log_file_path.empty() ? "chatbot_server.log" : svc_config.log_file_path;

        return run_training_pipeline(
            argc, argv, svc_config, default_model, log_path, "Full retrain started in background",
            {{"Data", std::to_string(data_file_count) + " file(s)"},
             {"Epochs", std::to_string(epochs)}},
            init_gpu, [&]() -> int {
                IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
                config.base_config.num_epochs = epochs;
                config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
                config.auto_save_enabled = true;
                config.auto_save_every_minutes = 30;
                config.auto_save_every_samples = 1000;

                IncrementalTrainer trainer(default_vocab, default_model, config);
                trainer.reset_model_for_config();

                // MNS is the definitive source for run_id — retrain always
                // requests a fresh run (new_run=true), falling back to the
                // local hostname+pid/RUN_ID-derived value only when MNS isn't
                // configured. The same run_id is then used for dataset-registry
                // file ownership below, unifying both systems' notion of "run".
                std::string run_id = trainer.begin_run(/*is_retrain=*/true);
                if (run_id.empty())
                    run_id = derive_run_id(svc_config.run_id);

                DatasetConfig dcfg = DatasetRegistry::make_config(svc_config);
                DatasetRegistry reg(dcfg);
                reg.load_registry();

                startup_sweep(reg, run_id, dcfg.download_dir);

                auto trained_fs = reg.trained_files();
                auto pending_resp = reg.acquire_pending(run_id);

                // For retrain: already-trained files are read directly by path (they
                // live on the registry machine which already trained them, so no FTP
                // needed for the trained portion).  Only newly acquired pending files
                // may need FTP download.
                std::vector<std::string> local_paths(trained_fs.begin(), trained_fs.end());
                std::vector<fs::path> downloaded_paths;
                const bool use_ftp =
                    !pending_resp.ftp_server_host.empty() && !dcfg.download_dir.empty();

                if (!pending_resp.files.empty()) {
                    if (use_ftp) {
                        const std::size_t warn_bytes =
                            static_cast<std::size_t>(dcfg.large_file_warn_threshold_mb) * 1024ULL *
                            1024ULL;
                        try {
                            DataTransport dt;
                            downloaded_paths =
                                dt.fetch_all(pending_resp, dcfg.download_dir,
                                             dcfg.max_parallel_downloads, warn_bytes);
                            for (const auto& p : downloaded_paths)
                                local_paths.push_back(p.string());
                        } catch (const std::exception& ex) {
                            adai::Logger::error("[DataTransport] Download failed: {}", ex.what());
                            reg.release_pending(run_id, pending_resp.registry_paths());
                            return 1;
                        }
                    } else {
                        for (const auto& f : pending_resp.files)
                            local_paths.push_back(f.registry_path);
                    }
                }

                if (local_paths.empty()) {
                    adai::Logger::warn("No data files to retrain on");
                    return 1;
                }
                adai::Logger::info("Starting full retrain on {} data file(s)", local_paths.size());

                const bool ok = trainer.retrain_on_files(local_paths, epochs);
                const auto pending_reg_paths = pending_resp.registry_paths();

                if (ok) {
                    if (!pending_reg_paths.empty()) {
                        std::vector<int> counts(pending_reg_paths.size(), 0);
                        reg.mark_trained(run_id, pending_reg_paths, counts);
                    }
                    trainer.print_training_summary();
                } else {
                    if (!pending_reg_paths.empty())
                        reg.release_pending(run_id, pending_reg_paths);
                    std::cerr << "❌ Full retrain failed\n";
                }

                if (use_ftp)
                    cleanup_downloads(downloaded_paths);
                return ok ? 0 : 1;
            });

    } else if (command == "reset") {
        // Parse reset-specific flags from remaining args
        bool auto_yes = false;
        bool keep_data = false;
        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--yes") {
                auto_yes = true;
            } else if (args[i] == "--keep-data") {
                keep_data = true;
            }
        }

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
        config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
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
            svc_config.log_file_path.empty() ? "chatbot_server.log" : svc_config.log_file_path;

        return run_training_pipeline(
            argc, argv, svc_config, default_model, log_path, "Resume started in background", {},
            init_gpu, [&]() -> int {
                IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
                IncrementalTrainer trainer(default_vocab, default_model, config);
                // MNS is the definitive source for run_id — resume continues
                // the model's current run (not a retrain); falls back to the
                // local hostname+pid/RUN_ID-derived value inside
                // resume_last_session() when MNS isn't configured.
                trainer.begin_run(/*is_retrain=*/false);
                if (!trainer.resume_last_session())
                    return 1;
                adai::Logger::info("Resumed from last session; latest checkpoint: {}",
                                   trainer.get_latest_checkpoint());
                return 0;
            });

    } else if (command == "status") {
        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);
        trainer.print_training_summary();

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();
        std::cout << "\n📋 Pending data files:\n";
        for (const auto& file : reg.pending_files()) {
            std::cout << "  - " << file << "\n";
        }

    } else if (command == "history") {
        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);
        print_session_history(trainer.get_session_history());
        print_data_registry(svc_config);

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        std::cerr << "Run '" << argv[0] << "' without arguments for help\n";
        return 1;
    }

    return 0;
}
