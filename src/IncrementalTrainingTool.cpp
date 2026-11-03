// @adai-status: beta        (TD-035 resolved — argv/config parsing extracted and tested; still large and actively evolving, see TD-039; TD-172 serve command removed, --admin-port added to resume; TD-183 added --objective=lejepa; TD-184 made the lejepa pass resume from an existing checkpoint; TD-202 added automatic-by-objective dataset_kind wiring + --dataset-kind override; TD-204 reverted TD-203's config-file-DATASET_KIND-wins-over-objective change as a design mistake)
// @adai-version: 0.32.0
// @adai-reviewed: 2026-11-02

#include <array>
#include <atomic>
#include <chrono>
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
#include "SignalSetup.hpp"
#include "DataTransport.hpp"
#include "DatasetRegistry.hpp"
#include "IncrementalTrainer.hpp"
#include "IncrementalTrainerArgs.hpp"
#include "LeJEPAEncoder.hpp"
#include "Logger.hpp"
#include "Matrix.hpp"
#include "ModelNameClient.hpp"
#include "StartupSweep.hpp"
#include "TrainerControlState.hpp"
#ifdef BUILD_TRAINER_ADMIN
#include "TrainerAdminAPI.hpp"
#endif

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

// ── Download directory helpers ────────────────────────────────────────────────

// TD-411: the "acquired files -> local files" step shared by train, retrain and resume. When the
// registry hands out an FTP host (a remote registry_server), the files are downloaded into
// dcfg.download_dir; otherwise the registry's own paths are used directly. `ranges` gets one
// entry per resp.files entry, in order, with its TD-205 segment range; `downloaded` gets the
// files to delete after the pass. Returns false (logged) if a download failed; the caller then
// releases the files.
static bool resolve_pass_files(const AcquireResponse& resp, const DatasetConfig& dcfg,
                               std::vector<PendingFileRange>& ranges,
                               std::vector<fs::path>& downloaded) {
    std::vector<std::string> local_paths;
    // Remote first: whenever the registry serves files over FTP, download them. An unset
    // DOWNLOAD_DIR used to silently fall back to the registry host's paths, which don't exist
    // on a separate trainer host; use <SESSION_DIR>/downloads instead.
    const bool use_ftp = !resp.ftp_server_host.empty();
    if (use_ftp) {
        std::string download_dir = dcfg.download_dir;
        if (download_dir.empty()) {
            download_dir = (fs::path(dcfg.session_dir) / "downloads").string();
            adai::Logger::warn("DOWNLOAD_DIR is not set; downloading registry files to {}",
                               download_dir);
        }
        std::error_code mk_ec;
        fs::create_directories(download_dir, mk_ec);
        const std::size_t warn_bytes =
            static_cast<std::size_t>(dcfg.large_file_warn_threshold_mb) * 1024ULL * 1024ULL;
        try {
            DataTransport dt(FtpsVerification{dcfg.ftps_ca_file, dcfg.ftps_pinned_public_key});
            downloaded =
                dt.fetch_all(resp, download_dir, dcfg.max_parallel_downloads, warn_bytes);
            for (const auto& p : downloaded)
                local_paths.push_back(p.string());
        } catch (const std::exception& ex) {
            adai::Logger::error("[DataTransport] Download failed: {}", ex.what());
            return false;
        }
    } else {
        for (const auto& f : resp.files)
            local_paths.push_back(f.registry_path);
    }
    // local_paths[i] corresponds 1:1 with resp.files[i] in both branches (fetch_all() fills its
    // result by index) -- zip each with its entry's segment range.
    ranges.clear();
    ranges.reserve(local_paths.size());
    for (std::size_t i = 0; i < local_paths.size(); ++i) {
        PendingFileRange r;
        r.path = local_paths[i];
        if (i < resp.files.size()) {
            r.segment_start = resp.files[i].segment_start;
            r.segment_count = resp.files[i].segment_count;
        }
        ranges.push_back(std::move(r));
    }
    return true;
}

static void cleanup_downloads(const std::vector<fs::path>& local_paths) {
    for (const auto& p : local_paths) {
        // TD-337: segments of one file share a download, so a path can appear more than once.
        if (!p.empty() && fs::exists(p)) {
            std::error_code ec;
            fs::remove(p, ec);
            if (ec) {
                adai::Logger::warn("[cleanup] Failed to remove '{}': {}", p.string(), ec.message());
            }
        }
        if (!p.empty()) {
            std::error_code ec;
            fs::remove(DataTransport::checksum_path_for(p), ec);  // TD-338 sidecar
        }
    }
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

// ── LeJEPA world-model pretraining pass (TD-183) ────────────────────────────────
//
// Runs one `--objective=lejepa train` pass: acquires pending files via the SAME dataset
// registry/FTP-download machinery the chatbot objective uses (the "train" command's own
// pre-fork pending-count check, shared by both objectives — see the call site in main()), but
// trains a standalone LeJEPAEncoder world model via LeJEPAEncoder::train_step() instead of
// building a ChatbotTrainer/IncrementalTrainer at all. Resumes from `world_model_dir`'s own
// checkpoint if one already exists (TD-184) — genuinely incremental across separate invocations,
// the same way IncrementalTrainer's own constructor resumes the chatbot model. No MNS
// registration or IncrementalTrainer::begin_run() here — the world model is its own,
// independently MNS-registered artifact (via a manual `mns_cli register`, TD-184), not something
// this pass registers itself.
//
// Data format: reuses DatasetRegistry::load_conversation_pairs() — the only file-content parser
// this codebase has — even though LeJEPA's own objective needs no (input, target) pairing at all
// (see LeJEPAEncoder's class doc). Both a ConversationPair's `input` and `response` fields are
// fed to train_step() as independent, unpaired plain-text samples: the JSONL half of that parser
// only requires a non-empty "input" field (see parse_jsonl_sample in TrainingSampleMeta.hpp), so
// an operator queuing genuinely unpaired text for this objective can write bare `{"input":
// "..."}` lines with no "response" field at all — and any already-queued ordinary (input,
// response) chatbot data gets both of its sides reused for pretraining for free, with zero new
// file-format code.
// Compact JSON config snapshot for the LeJEPA metrics session — mirrors
// IncrementalTrainer.cpp's own build_config_snapshot(), but reads the
// WORLD_MODEL_* fields off ServiceConfig directly since this path has no
// IncrementalConfig/TrainingConfig of its own.
static std::string build_lejepa_config_snapshot(const adai::ServiceConfig& svc_config) {
    std::ostringstream json;
    json << std::fixed;
    json << "{" << "\"d_model\":" << svc_config.world_model_d_model
         << ",\"heads\":" << svc_config.world_model_num_heads
         << ",\"d_ff\":" << svc_config.world_model_d_ff
         << ",\"layers\":" << svc_config.world_model_num_layers
         << ",\"sigreg_lambda\":" << svc_config.world_model_sigreg_lambda
         << ",\"sigreg_sketches\":" << svc_config.world_model_sigreg_num_sketches
         << ",\"lr\":" << svc_config.learning_rate << "}";
    return json.str();
}

// TD-403: `control`, when given (a trainer_service pass), is the per-pass admin API's state: the pass
// reports its phase and progress there, writes a checkpoint on /admin/checkpoint, takes the auto-save
// settings live from it, and on /admin/pause stops at the next batch, checkpoints, releases its files and
// returns kResumeExitFailed (as an aborted chatbot pass does).
static int run_lejepa_training_pass(const adai::ServiceConfig& svc_config,
                                    const std::string& default_vocab,
                                    const std::string& world_model_dir, int epochs,
                                    const std::shared_ptr<adai::TrainerControlState>& control = nullptr) {
    // The admin API reports idle again whichever way the pass ends.
    struct IdleOnExit {
        const std::shared_ptr<adai::TrainerControlState>& control;
        ~IdleOnExit() {
            if (control) control->phase = adai::TrainerPhase::Idle;
        }
    } idle_on_exit{control};
    if (control) {
        control->phase = adai::TrainerPhase::LoadingData;
        control->log(adai::TrainerLogLevel::Info, "LeJEPA world-model pass starting");
    }

    // Determine vocab_size the same way IncrementalTrainer.cpp's own model-construction path
    // does: load the vocab file into a throwaway tokenizer first, then size the encoder's fixed
    // TokenEmbedding to match. LeJEPAEncoder::load_tokenizer_vocab() itself does not resize
    // anything (a pre-existing gap in that method, not introduced here — flagged separately),
    // so the constructor's own vocab_size must already be correct before that call.
    // TD-416: TOKENIZER_MODE, as for the chatbot (it used to be ASCII always); a vocabulary that records
    // its own mode overrides it on load.
    const TokenizerMode tokenizer_mode = svc_config.unicode_tokenizer ? TokenizerMode::UNICODE : TokenizerMode::ASCII;
    BPETokenizer vocab_probe(tokenizer_mode);
    vocab_probe.load_vocab(default_vocab);
    const int vocab_size = vocab_probe.get_vocab_size();

    LeJEPAEncoder world_model(vocab_size, static_cast<int>(svc_config.world_model_d_model),
                              static_cast<int>(svc_config.world_model_num_layers),
                              static_cast<int>(svc_config.world_model_num_heads),
                              static_cast<int>(svc_config.world_model_d_ff),
                              static_cast<int>(svc_config.max_seq_length),
                              static_cast<int>(svc_config.world_model_sigreg_num_sketches));
    world_model.set_tokenizer_mode(tokenizer_mode);
    world_model.load_tokenizer_vocab(default_vocab);

    // TD-184: genuinely incremental across separate `train` invocations — resume from a prior
    // pass's checkpoint if one exists, the same way IncrementalTrainer's own constructor resumes
    // the chatbot model from its best saved checkpoint before training on newly-acquired data
    // (see IncrementalTrainer.cpp's constructor). Without this, every pass would silently discard
    // all prior progress and retrain a fresh, randomly-initialized encoder — defeating the whole
    // point of feeding it new pending-data batches over time. A mismatched architecture (the
    // WORLD_MODEL_* config changed since the checkpoint was saved) is reported clearly and this
    // pass aborts rather than either silently discarding the old checkpoint or crashing uncaught.
    // TD-548: saves go to world_model_dir; the checkpoint to resume may still be in the
    // pre-TD-548 <session_dir>/world_model (resolve_checkpoint_dir() finds it).
    const std::string resume_dir = LeJEPAEncoder::resolve_checkpoint_dir(
        svc_config.session_dir, svc_config.world_model_artifact_path);
    if (LeJEPAEncoder::has_checkpoint(resume_dir)) {
        try {
            world_model.load(resume_dir);
            adai::Logger::info("[lejepa] Resumed world-model weights from '{}'", resume_dir);
            if (resume_dir != world_model_dir) {
                adai::Logger::info("[lejepa] Saving to '{}' from now on (TD-548)", world_model_dir);
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ [lejepa] Failed to load existing world-model checkpoint at '"
                      << resume_dir << "': " << e.what() << "\n"
                      << "   (WORLD_MODEL_* config must match the architecture the checkpoint "
                         "was saved with — or move/remove the old checkpoint directory to start "
                         "fresh)\n";
            return 1;
        }
    }

    world_model.set_sigreg_lambda(svc_config.world_model_sigreg_lambda);
    world_model.set_learning_rate(svc_config.learning_rate);

    Optimizer optimizer(OptimizerType::ADAM, svc_config.learning_rate);
    world_model.register_parameters_with_optimizer(optimizer);

    // TD-415: a run id kept with the world-model checkpoint, so a pass restarted after a crash, kill or
    // SIGTERM reclaims the files the last one claimed (registries hand a claim back to the same run id).
    // It used to be PID-based, so those files stayed claimed by an id no later process had.
    std::string run_id = adai::persistent_run_id(svc_config.run_id, world_model_dir);
    if (control) control->set_run_id(run_id);

    DatasetConfig dcfg = DatasetRegistry::make_config(svc_config);
    DatasetRegistry reg(dcfg);
    reg.load_registry();
    startup_sweep(reg, run_id, dcfg.download_dir);  // loads the pending list itself (TD-512)

    auto resp = reg.acquire_pending(run_id);
    // TD-403: the same exit codes as the chatbot resume (TD-523), now that trainer_service runs this
    // pass too: an unreachable registry is a failure, an empty pool means idle.
    if (resp.failed) {
        adai::Logger::error("[lejepa] Could not ask the dataset registry for pending files");
        return adai::kResumeExitFailed;
    }
    if (resp.files.empty()) {
        adai::Logger::warn("[lejepa] No pending data files to train on");
        return adai::kResumeExitNothingPending;
    }

    // TD-411: the same download step as train/retrain/resume. This pass used to download only
    // when DOWNLOAD_DIR was set, so against an FTP registry without it, it read registry-host
    // paths that don't exist on the trainer host.
    std::vector<PendingFileRange> local_ranges;
    std::vector<fs::path> downloaded_paths;
    if (!resolve_pass_files(resp, dcfg, local_ranges, downloaded_paths)) {
        reg.release_pending(run_id, resp.file_ranges());  // TD-333: exact units
        return 1;
    }
    std::vector<std::string> texts;
    std::vector<int> pair_counts;  // TD-334: per file, 1:1 with local_ranges / resp.files
    for (const auto& range : local_ranges) {
        std::vector<ConversationPair> pairs;
        // TD-415: only the claimed segment's rows (TD-205), as the chatbot pass loads them; every claimed
        // segment of a file used to load the whole file.
        DatasetRegistry::load_conversation_pairs(range.path, pairs, range.segment_start, range.segment_count);
        pair_counts.push_back(static_cast<int>(pairs.size()));
        for (const auto& pair : pairs) {
            if (!pair.input.empty())
                texts.push_back(pair.input);
            if (!pair.response.empty())
                texts.push_back(pair.response);
        }
    }

    // TD-333: exact units (whole files and segments)
    const auto reg_paths = resp.file_ranges();
    if (texts.empty()) {
        adai::Logger::warn("[lejepa] Acquired files contained no usable text samples");
        reg.release_pending(run_id, reg_paths);
        cleanup_downloads(downloaded_paths);  // empty unless files were downloaded
        return 1;
    }
    if (control) {
        control->total_epochs = epochs;
        control->current_epoch = 0;
        control->samples_trained_this_pass = 0;
        control->phase = adai::TrainerPhase::Training;
        control->log(adai::TrainerLogLevel::Info,
                     "Training the world model on " + std::to_string(texts.size()) + " text(s)");
    }

    // Metrics-dashboard reporting (previously entirely absent from this path — the only
    // signal during a potentially many-hour pass was two log lines total: one per epoch,
    // one at the very end). Mirrors IncrementalTrainer::run_training()'s MetricsPushClient
    // wiring (IncrementalTrainer.cpp), but with a deliberately STABLE session key —
    // "lejepa-<model_name>", not a per-invocation session_id+host key — so the dashboard
    // shows one continuous training-history curve across every separate `train` invocation
    // for this world model's lifetime, matching how the checkpoint itself is resumed above.
    // A 409 here means a prior pass crashed without ever reaching end_session() (this loop
    // has no periodic checkpoint, so a crash mid-pass — e.g. the real GPU engine-reset seen
    // on ai-machine — leaves the session marked "training" forever); MetricsSessionRegistry
    // only 409s a genuinely still-"training" session, never an already-ended one, so a short
    // retry on the SAME key (changing it would fragment the continuous curve) gives the
    // staleness sweep a chance to reclaim it.
    std::unique_ptr<MetricsPushClient> metrics_client;
    if (!svc_config.metrics_server_url.empty()) {
        const std::string session_key = sanitize_session_key(
            "lejepa-" + (svc_config.model_name.empty() ? std::string("world-model")
                                                        : svc_config.model_name));
        const std::string push_url =
            build_metrics_session_push_base(svc_config.metrics_server_url, session_key);
        const std::string label = svc_config.metrics_session_label.empty()
                                      // "world_model": the same label as before TD-548 moved
                                      // the default directory
                                      ? derive_metrics_session_label(1, "world_model")
                                      : svc_config.metrics_session_label;
        const std::string config_snapshot = build_lejepa_config_snapshot(svc_config);

        auto pc = std::make_unique<MetricsPushClient>(
            push_url, svc_config.metrics_push_timeout_ms, 1024,
            svc_config.metrics_heartbeat_interval_ms,
            parse_metrics_server_headers(svc_config.metrics_server_headers),
            svc_config.metrics_server_ca_file);
        int rc = 0;
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (attempt > 0) {
                adai::Logger::warn("[lejepa] Metrics session busy (409), retrying '{}'",
                                   session_key);
                std::this_thread::sleep_for(std::chrono::milliseconds(500 * attempt));
            }
            rc = pc->start_session(1, epochs, static_cast<int>(texts.size()), label,
                                   config_snapshot, /*reset_best=*/false);
            if (rc != 409)
                break;
        }
        if (rc >= 200 && rc < 300) {
            metrics_client = std::move(pc);
        } else {
            adai::Logger::error(
                "[lejepa] Metrics session/start failed for key '{}' (HTTP {}) — training will "
                "proceed WITHOUT metrics reporting for this pass",
                session_key, rc);
        }
    }

    // Mid-pass checkpointing: this loop previously only saved once, after every acquired
    // file had been fully processed — for a many-hour-to-many-day pass over the whole
    // pending pool, a crash (process kill, the real GPU engine-reset seen on ai-machine,
    // etc.) lost 100% of progress, not just the time since the last save. Reuses the same
    // AUTO_SAVE_ENABLED/AUTO_SAVE_EVERY_SAMPLES/AUTO_SAVE_EVERY_MINUTES config keys the
    // chatbot path already uses (same cross-objective concern, no new config surface) —
    // resuming from wherever this leaves off is already handled by the TD-184 load-on-
    // startup logic above, so a periodic world_model.save() here is the whole fix; nothing
    // else needs to change. A failed periodic save is logged and skipped rather than
    // aborting the pass — the resilience feature itself must not become a new crash source.
    int samples_since_last_save = 0;
    auto last_save_time = std::chrono::steady_clock::now();
    // One checkpoint write, reported to the admin API when there is one (TD-403).
    const auto save_checkpoint = [&](const std::string& why) {
        if (control) control->phase = adai::TrainerPhase::Checkpointing;
        bool saved = false;
        try {
            world_model.save(world_model_dir);
            saved = true;
            if (control) {
                control->checkpoints_written.fetch_add(1);
                control->last_checkpoint_time_unix = static_cast<long long>(std::time(nullptr));
                control->set_last_checkpoint_path(world_model_dir);
                control->log(adai::TrainerLogLevel::Info, "Checkpoint saved (" + why + "): " + world_model_dir);
            } else {
                adai::Logger::info("[lejepa] Checkpoint saved to '{}' ({})", world_model_dir, why);
            }
        } catch (const std::exception& e) {
            adai::Logger::warn("[lejepa] Checkpoint save failed, continuing: {}", e.what());
        }
        if (control) control->phase = adai::TrainerPhase::Training;
        samples_since_last_save = 0;
        last_save_time = std::chrono::steady_clock::now();
        return saved;
    };
    bool aborted = false;

    // Advanced diagnostics (activation saturation / attention entropy): mirrors
    // ChatbotTrainer.cpp's own accumulate-via-hook/reset-per-epoch/report-then-clear pattern,
    // adapted to this loop's flat per-sample structure (there's no gradient-accumulation window
    // here, so the epoch is the natural reset boundary). Registered once, outside the epoch
    // loop, since world_model's layer structure doesn't change across epochs; accumulators are
    // reset at the start of each epoch below.
    //
    // Caveat: LeJEPAEncoder::train_step() calls encode_tokens() 3x per sample internally
    // (context view, target view, re-encoded context view for its second internal update), so
    // these hooks fire 3x per sample, blending the partially-masked context view and the
    // unmasked target view indiscriminately — a legitimate first-cut blended signal, not
    // view-separated.
    float sat_sum = 0.0f;
    int sat_count = 0;
    float ent_sum = 0.0f;
    int ent_count = 0;
    auto saturation_hook = [&sat_sum, &sat_count](const Matrix& activated) {
        const int total = activated.rows * activated.cols;
        if (total <= 0) {
            return;
        }
        int sat = 0;
        for (int r = 0; r < activated.rows; ++r) {
            for (int c = 0; c < activated.cols; ++c) {
                if (std::abs(activated(r, c)) < 0.01f) {
                    ++sat;
                }
            }
        }
        sat_sum += static_cast<float>(sat) / static_cast<float>(total);
        ++sat_count;
    };
    auto entropy_hook = [&ent_sum, &ent_count](const Matrix& attn_weights) {
        const int seq_len = attn_weights.rows;
        if (seq_len <= 0 || attn_weights.cols <= 0) {
            return;
        }
        float layer_entropy = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            float row_entropy = 0.0f;
            for (int j = 0; j < attn_weights.cols; ++j) {
                float a = attn_weights(i, j);
                if (a > 0.0f) {
                    row_entropy -= a * std::log(a + 1e-10f);
                }
            }
            layer_entropy += row_entropy;
        }
        ent_sum += layer_entropy / static_cast<float>(seq_len);
        ++ent_count;
    };
    const int world_model_layers = world_model.get_num_layers();
    for (int l = 0; l < world_model_layers; ++l) {
        world_model.get_encoder_block(l)->get_feed_forward()->set_activation_hook(
            saturation_hook);
        world_model.get_encoder_block(l)->get_self_attention()->set_attention_hook(entropy_hook);
    }

    double predictor_loss_sum = 0.0;
    double sigreg_loss_sum = 0.0;
    size_t step_count = 0;
    for (int epoch = 0; epoch < epochs && !aborted; ++epoch) {
        if (metrics_client)
            metrics_client->start_epoch(epoch + 1, static_cast<int>(texts.size()));

        double epoch_predictor_sum = 0.0;
        double epoch_sigreg_sum = 0.0;
        size_t epoch_step_count = 0;
        sat_sum = 0.0f;
        sat_count = 0;
        ent_sum = 0.0f;
        ent_count = 0;
        double epoch_masking_ratio_sum = 0.0;
        double epoch_cosine_sim_sum = 0.0;
        double epoch_sigreg_variance_mean_sum = 0.0;
        double epoch_sigreg_variance_stddev_sum = 0.0;
        // compute_time_ratio here means "fraction of epoch wall time actually inside
        // train_step() calls" (surfaces checkpoint-save/metrics-push overhead between
        // samples) — train_step() is an indivisible forward+backward+both-optimizer-steps
        // black box, unlike the chatbot path's own forward+backward-vs-data-loading version
        // of this same metric.
        long long epoch_train_step_ns = 0;
        const auto epoch_wall_start = std::chrono::steady_clock::now();
        int sample_idx = 0;
        // SIGReg is applied across each batch's mean-pooled embeddings (LeJEPAEncoder::
        // train_batch(), as the LeJEPA paper does) rather than one text's tokens.
        const size_t batch_size = std::max<size_t>(2, svc_config.world_model_sigreg_batch_size);
        for (size_t first = 0; first < texts.size(); first += batch_size) {
            if (control && control->paused) {  // TD-403: /admin/pause drains at a batch boundary
                aborted = true;
                break;
            }
            if (control && control->checkpoint_requested.exchange(false)) {  // POST /admin/checkpoint
                save_checkpoint("requested");
            }
            const size_t last = std::min(texts.size(), first + batch_size);
            const std::vector<std::string> batch(texts.begin() + static_cast<std::ptrdiff_t>(first),
                                                 texts.begin() + static_cast<std::ptrdiff_t>(last));
            const auto step_start = std::chrono::steady_clock::now();
            LeJEPAEncoder::BatchStepResult result;
            try {
                result = world_model.train_batch(batch);
            } catch (const std::exception& e) {
                adai::Logger::warn("[lejepa] Skipping batch of {} ({})", batch.size(), e.what());
                sample_idx += static_cast<int>(batch.size());
                continue;
            }
            epoch_train_step_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - step_start)
                                       .count();
            if (result.trained.size() < batch.size()) {
                adai::Logger::warn("[lejepa] Skipped {} of {} samples in a batch (too short or "
                                   "untokenizable)",
                                   batch.size() - result.trained.size(), batch.size());
            }
            sample_idx += static_cast<int>(batch.size());
            for (size_t k = 0; k < result.trained.size(); ++k) {
                const float predictor_loss = result.predictor_losses[k];
                // The batch's SIGReg loss and diagnostics count once per trained text, so the
                // per-step averages below keep their meaning.
                const float sigreg_loss = result.sigreg_loss;
                predictor_loss_sum += predictor_loss;
                sigreg_loss_sum += sigreg_loss;
                epoch_predictor_sum += predictor_loss;
                epoch_sigreg_sum += sigreg_loss;
                epoch_masking_ratio_sum += world_model.get_last_masking_ratio();
                epoch_cosine_sim_sum += world_model.get_last_predictor_target_cosine_sim();
                epoch_sigreg_variance_mean_sum += world_model.get_last_sigreg_variance_mean();
                epoch_sigreg_variance_stddev_sum += world_model.get_last_sigreg_variance_stddev();
                ++step_count;
                ++epoch_step_count;
                if (metrics_client) {
                    // Live per-sample curve on the existing loss/perplexity chart —
                    // predictor_loss alone (the actual predictive objective, most
                    // interpretable). sigreg_loss gets its own dedicated dashboard
                    // channel via update_lejepa_metrics() below (TD-178), not folded
                    // in here.
                    metrics_client->update_sample_metrics(
                        static_cast<int>(first + result.trained[k] + 1),
                        static_cast<float>(predictor_loss), world_model.get_last_gradient_norm(),
                        static_cast<float>(svc_config.learning_rate));
                }

                if (control) {
                    control->samples_trained_this_pass = static_cast<long long>(step_count);
                    control->last_loss = static_cast<double>(predictor_loss);
                }

                ++samples_since_last_save;
                // TD-403: live from the admin API when there is one (PUT /admin/config).
                const bool auto_save = control ? control->auto_save_enabled.load() : svc_config.auto_save_enabled;
                const int every_samples =
                    control ? control->auto_save_every_samples.load() : svc_config.auto_save_every_samples;
                const int every_minutes =
                    control ? control->auto_save_every_minutes.load() : svc_config.auto_save_every_minutes;
                if (auto_save) {
                    const auto elapsed_minutes =
                        std::chrono::duration_cast<std::chrono::minutes>(
                            std::chrono::steady_clock::now() - last_save_time)
                            .count();
                    const bool due_by_samples = every_samples > 0 && samples_since_last_save >= every_samples;
                    const bool due_by_time = every_minutes > 0 && elapsed_minutes >= every_minutes;
                    if (due_by_samples || due_by_time) {
                        save_checkpoint(std::to_string(step_count) + " steps this pass so far");
                    }
                }
            }
        }
        if (aborted) break;
        adai::Logger::info("[lejepa] Epoch {}/{} complete ({} samples)", epoch + 1, epochs,
                           texts.size());
        if (control) {
            control->current_epoch = epoch + 1;
            if (epoch_step_count > 0) {
                const double avg = epoch_predictor_sum / static_cast<double>(epoch_step_count);
                if (epoch == 0 || avg < control->best_loss.load()) control->best_loss = avg;
            }
        }
        if (metrics_client && epoch_step_count > 0) {
            const float avg_predictor_loss =
                static_cast<float>(epoch_predictor_sum / static_cast<double>(epoch_step_count));
            const float avg_sigreg_loss =
                static_cast<float>(epoch_sigreg_sum / static_cast<double>(epoch_step_count));
            // These buffered-field setters must all run BEFORE end_epoch() below —
            // MetricsPushClient::end_epoch() reads its buffers synchronously while building the
            // /epoch/end JSON body, so a setter called after it fires would have its value land
            // in the FOLLOWING epoch's POST instead of this one (still at this epoch's
            // start_epoch()-reset sentinel when end_epoch() runs) — the same ordering lesson
            // TD-208 already learned once server-side in TrainingMetricsAPI::
            // handle_post_epoch_end(), just on the client side this time. Mirrors
            // ChatbotTrainer.cpp's own call order (update_advanced_epoch_metrics/
            // update_activation_saturation/update_attention_entropy all precede its end_epoch()
            // call).
            //
            // TD-178: predictor_loss/sigreg_loss get their own dedicated dashboard channel —
            // two independent series, not a train/validation pair.
            metrics_client->update_lejepa_metrics(avg_predictor_loss, avg_sigreg_loss);

            const auto epoch_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - epoch_wall_start)
                                          .count();
            const float compute_time_ratio =
                (epoch_wall_ns > 0)
                    ? static_cast<float>(static_cast<double>(epoch_train_step_ns) /
                                         static_cast<double>(epoch_wall_ns))
                    : 0.0f;
            // gradient_variance/weight_update_ratio have no LeJEPA-side equivalent (blocked on
            // how train_step()'s two sequential internal optimizer updates, TD-189, should
            // report gradients separately vs. combined — a deeper design question than this
            // change covers) — passed as 0.0f. Both already default to 0.0f in the snapshot, so
            // this isn't a new ambiguity; update_advanced_epoch_metrics() bundles all three
            // fields into one call, so sending compute_time_ratio at all means these two also
            // get persisted as literal 0.0 rather than the -1 "not applicable" sentinel used
            // elsewhere (a pre-existing API-shape wrinkle, not introduced here).
            metrics_client->update_advanced_epoch_metrics(0.0f, compute_time_ratio, 0.0f);

            const float avg_saturation =
                (sat_count > 0) ? (sat_sum / static_cast<float>(sat_count)) : -1.0f;
            metrics_client->update_activation_saturation(avg_saturation);
            const float avg_entropy =
                (ent_count > 0) ? (ent_sum / static_cast<float>(ent_count)) : -1.0f;
            metrics_client->update_attention_entropy(avg_entropy);

            const float avg_masking_ratio = static_cast<float>(
                epoch_masking_ratio_sum / static_cast<double>(epoch_step_count));
            const float avg_cosine_sim = static_cast<float>(
                epoch_cosine_sim_sum / static_cast<double>(epoch_step_count));
            const float avg_sigreg_variance_mean = static_cast<float>(
                epoch_sigreg_variance_mean_sum / static_cast<double>(epoch_step_count));
            const float avg_sigreg_variance_stddev = static_cast<float>(
                epoch_sigreg_variance_stddev_sum / static_cast<double>(epoch_step_count));
            metrics_client->update_lejepa_advanced_metrics(
                avg_masking_ratio, avg_cosine_sim, avg_sigreg_variance_mean,
                avg_sigreg_variance_stddev);

            // No real validation split for this objective — validation_loss stays 0.0f,
            // matching ChatbotTrainer's own convention of simply not reporting it when
            // there's no validation data (rather than an N/A sentinel).
            metrics_client->end_epoch(epoch + 1, avg_predictor_loss, /*validation_loss=*/0.0f,
                                      static_cast<float>(svc_config.learning_rate),
                                      /*perplexity=*/0.0f, world_model.get_last_gradient_norm());
        }
    }

    for (int l = 0; l < world_model_layers; ++l) {
        world_model.get_encoder_block(l)->get_feed_forward()->clear_activation_hook();
        world_model.get_encoder_block(l)->get_self_attention()->clear_attention_hook();
    }

    if (aborted) {
        // TD-403: like a paused chatbot pass, keep the progress but don't record the files as trained;
        // they go back to the pool for the next pass.
        if (control) {
            control->log(adai::TrainerLogLevel::Warn,
                         "LeJEPA pass paused via admin API after " + std::to_string(step_count) +
                             " step(s) — checkpointing and releasing claimed files back to pending");
        }
        if (step_count > 0) save_checkpoint("paused");
        reg.release_pending(run_id, reg_paths);
        if (metrics_client) metrics_client->end_session();
        cleanup_downloads(downloaded_paths);
        return adai::kResumeExitFailed;
    }

    const bool ok = step_count > 0;
    if (ok) {
        reg.mark_trained(run_id, reg_paths, pair_counts);  // TD-334: pairs loaded per file
        adai::Logger::info(
            "[lejepa] Pass complete — {} steps, avg predictor_loss={:.4f}, avg sigreg_loss={:.4f}",
            step_count, predictor_loss_sum / static_cast<double>(step_count),
            sigreg_loss_sum / static_cast<double>(step_count));
        world_model.save(world_model_dir);
        adai::Logger::info("[lejepa] World-model checkpoint saved to '{}'", world_model_dir);
    } else {
        reg.release_pending(run_id, reg_paths);
        adai::Logger::warn("[lejepa] No samples were long enough to train on — all skipped");
    }

    if (metrics_client)
        metrics_client->end_session();
    cleanup_downloads(downloaded_paths);  // empty unless files were downloaded
    return ok ? 0 : 1;
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
// initialises the logger and GPU in the child before running child_work().
// init_gpu_fn and child_work are callables; child_work returns an exit code.
// banner_extras are {label, value} pairs printed between Model and Log lines;
// labels are left-padded to 7 characters to align with the fixed lines.
template <typename InitGpuFn, typename WorkerFn>
int run_training_pipeline(int argc, char* argv[], const adai::ServiceConfig& svc_config,
                          const std::string& default_model, const std::string& log_path,
                          const std::string& title,
                          const std::vector<std::pair<std::string, std::string>>& banner_extras,
                          InitGpuFn&& init_gpu_fn, WorkerFn&& child_work, bool foreground = false) {
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
                std::cout << "[MNS] run_group resolved from MNS: '" << rm.run_group << "'\n";
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
                    std::cout << "[MNS] World model '" << conn->world_model_name
                              << "' resolved from MNS link (artifact='"
                              << wm_resolved.artifact.path << "')\n";
                } else {
                    std::cout << "[MNS] World model '" << conn->world_model_name
                              << "' linked but not yet resolvable (no artifact/architecture) — "
                                 "using local config\n";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[MNS] World model connection lookup failed (" << e.what()
                      << "); using local config\n";
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
        // TD-269: warm up once per run
        config.base_config.lr_schedule = LRSchedule::LINEAR_WARMUP;
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
                init_gpu,
                [&]() -> int {
                    PassAdminApi pass(std::nullopt, svc_config);  // TD-416: a SIGTERM drains the pass
                    return run_lejepa_training_pass(svc_config, default_vocab, world_model_dir, epochs,
                                                    pass.control());
                },
                foreground);
        }

        return run_training_pipeline(
            argc, argv, svc_config, default_model, log_path, "Training started in background",
            {{"Data", std::to_string(pre_fork_pending_count) + " pending file(s)"},
             {"Epochs", std::to_string(epochs)}},
            init_gpu, [&]() -> int {
                IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
                config.base_config.num_epochs = epochs;
                // TD-269: warm up once per run
                config.base_config.lr_schedule = LRSchedule::LINEAR_WARMUP;
                // TD-414: with no vocabulary yet, the first pass builds it from its data (VOCAB_BUILD_SIZE).
                IncrementalTrainer trainer(default_vocab, default_model, config);
                PassAdminApi pass(std::nullopt, svc_config);  // TD-416: a SIGTERM drains the pass
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
            init_gpu, [&]() -> int {
                IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
                config.base_config.num_epochs = epochs;
                // TD-269: warm up once per run
                config.base_config.lr_schedule = LRSchedule::LINEAR_WARMUP;
                // TD-416: AUTO_SAVE_* as configured (this used to force 30 min / 1,000 samples).

                IncrementalTrainer trainer(default_vocab, default_model, config);
                trainer.reset_model_for_config();
                PassAdminApi pass(std::nullopt, svc_config);  // TD-416: a SIGTERM drains the pass
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

                auto trained_fs = reg.trained_files();
                auto pending_resp = reg.acquire_pending(run_id);

                // For retrain: already-trained files are read directly by path (they
                // live on the registry machine which already trained them, so no FTP
                // needed for the trained portion).  Only newly acquired pending files
                // may need FTP download.
                // Trained files are retrained whole (trained_files() lists each path once; TD-333's
                // per-segment records aren't consulted).
                // TODO: See TD-546 in TECHNICAL_DEBT.md - a partly trained segmented file is retrained whole and its
                // pending segments again
                // TODO: See TD-411 in TECHNICAL_DEBT.md - trained files are read by registry path, never downloaded, so
                // retrain against a remote FTP registry can't read them (only pending files are fetched)
                std::vector<PendingFileRange> local_ranges;
                for (const auto& path : trained_fs) {
                    PendingFileRange r;
                    r.path = path;
                    local_ranges.push_back(std::move(r));
                }
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
        // TD-269: warm up once per run
        config.base_config.lr_schedule = LRSchedule::LINEAR_WARMUP;
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
                "LeJEPA world-model resume started in background", {}, init_gpu,
                [&]() -> int {
                    PassAdminApi admin(admin_port, svc_config);  // TD-403: status, checkpoint, pause
                    return run_lejepa_training_pass(svc_config, default_vocab, world_model_dir,
                                                    svc_config.num_epochs, admin.control());
                },
                foreground);
        }

        return run_training_pipeline(
            argc, argv, svc_config, default_model, log_path, "Resume started in background", {},
            init_gpu, [&]() -> int {
                IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
                IncrementalTrainer trainer(default_vocab, default_model, config);

                // TD-172: --admin-port <N>, set only by trainer_service when it launches this as a
                // single-pass child (always with --foreground, so this process IS the worker), hosts
                // the per-pass admin API (PassAdminApi above) for the duration of this one pass.
                PassAdminApi admin(admin_port, svc_config);
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
