// @adai-status: experimental
// @adai-version: 0.9.0
// @adai-reviewed: 2026-11-02

// TD-172: trainer_service — the thin process-supervisor binary that replaces
// `incremental_trainer serve`. Deliberately minimal: it does not link adai_models/adai_nlp and
// never touches GPU/CUDA/SYCL objects at all — every actual training pass runs inside a
// single-pass `incremental_trainer --foreground --admin-port <N> resume` child this process
// launches and monitors (see ChildProcess.hpp). This binary's own job is just:
//   1. Launch a child, wait for it to exit, note whether it trained (exit 0), found nothing
//      pending (exit 3) or failed (anything else; TD-523), backing off on repeated failures.
//   2. If it did work, launch another one immediately (more may be pending); otherwise sleep the
//      poll interval first — the same shape `serve`'s old in-process loop already used.
//   3. Host an always-on admin HTTP listener (TrainerServiceProxy) that proxies /admin/* requests
//      to whichever child is currently alive, on the private port assigned via --admin-port.
//   4. Forward a graceful-stop request (SIGTERM/SIGINT) to whatever child is currently running,
//      escalating to a force kill if it doesn't exit in time (see ChildProcess::stop_and_wait()).
//
// TrainerControlState/TrainerAdminAPI are used completely unchanged — by the *child* process,
// exactly as `serve` already used them — see TrainerServiceProxy.hpp's own doc comment for why
// this design needs no changes to either class.
//
// TD-173: TrainerControlState's own `paused` only ever affected one pass — it's reconstructed
// fresh for every child, so nothing previously stopped this loop from immediately launching a new
// (unpaused) child right after a paused pass drained. TrainerServiceControlState (own header) is
// the process-lifetime piece that was missing: `paused`/`stop_requested` live here, checked by
// this loop before every launch, and set/cleared by TrainerServiceProxy's admin handlers.

#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "ChildProcess.hpp"
#include "Config.hpp"
#include "SignalSetup.hpp"
#include "IncrementalTrainerArgs.hpp"
#include "Logger.hpp"
#include "TrainerServiceControlState.hpp"
#include "TrainerServiceProxy.hpp"

namespace fs = std::filesystem;

namespace {

// Set once in main() before std::signal() registration — a signal handler can't capture, so it
// needs a plain pointer to reach shared state (same idiom IncrementalTrainingTool.cpp's own
// signal_handler() already uses for its TrainerControlState).
adai::TrainerServiceControlState* g_control = nullptr;

// Only an async-signal-safe atomic store here — see TrainerServiceControlState::wake()'s own doc
// comment for why this must NOT call wake() directly. interruptible_sleep()'s once-per-second
// poll (below, inside TrainerServiceControlState) picks up stop_requested within ~1s regardless.
void signal_handler(int sig) {
    if ((sig == SIGTERM || sig == SIGINT) && g_control != nullptr) {
        g_control->stop_requested = true;
    }
}

// Same `_unix` convention as TrainerControlState's own timestamp fields (e.g.
// last_checkpoint_time_unix, set in IncrementalTrainer.cpp) — whole seconds since the epoch.
std::int64_t unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace

static void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--config <path>] [--model <name>] [--gpu-strategy <strategy>]\n"
              << "       [--objective=chatbot|lejepa] [--dataset-kind <kind>]\n"
              << "\n"
              << "Runs the always-on training service: launches `incremental_trainer resume` as a\n"
              << "single-pass child, again and again, and serves the admin API (TRAINER_ADMIN_ENABLED,\n"
              << "TRAINER_ADMIN_PORT) for whichever pass is running. Stop it with SIGTERM.\n"
              << "\n"
              << "Options (passed on to every pass):\n"
              << "  --config <path>          config.trainer.conf to use (default: ./config.trainer.conf,\n"
              << "                           then /etc/adai/config.trainer.conf)\n"
              << "  --model <name>           MODEL_NAME override\n"
              << "  --gpu-strategy <s>       GPU_STRATEGY override: background or full\n"
              << "  --objective=<name>       chatbot (default) or lejepa (world-model pretraining;\n"
              << "                           needs WORLD_MODEL_ENABLED=true)\n"
              << "  --dataset-kind <kind>    pool to train from: encoder, decoder, world_model or chatbot\n"
              << "                           (default: world_model for lejepa, else chatbot)\n"
              << "  --help, -h               Show this help\n";
}

int main(int argc, char* argv[]) {
    // Reuses incremental_trainer's own global-flag parser (IncrementalTrainerArgs.hpp):
    // trainer_service accepts the same --config/--model/--gpu-strategy flags, passed straight
    // through to every child it launches. TD-557: --help prints the usage (it used to start the
    // service), and --foreground/--admin-port (what THIS process passes to each child), a
    // positional argument (trainer_service has no commands) or a flag error exits 1; all were ignored.
    const adai::TrainerServiceArgs parsed = adai::parse_trainer_service_args(argc, argv);
    if (parsed.help) {
        print_usage(argv[0]);
        return 0;
    }
    if (parsed.global.error) {
        std::cerr << parsed.global.error_message << "\n\n";
        print_usage(argv[0]);
        return 1;
    }
    const adai::IncrementalTrainerGlobalArgs& cli = parsed.global;

    const std::string config_path = adai::ConfigLoader::discover_config_path(
        cli.config_path.value_or(""), "config.trainer.conf");
    const adai::ServiceConfig svc_config = adai::ConfigLoader::load(config_path);

    const std::string log_path =
        svc_config.log_file_path.empty() ? "chatbot_server.log" : svc_config.log_file_path;
    // TODO: See TD-431 in TECHNICAL_DEBT.md - every child incremental_trainer pass falls back to the SAME file, so two
    // processes rotate it (~27% less retained history, verified); empty LOG_FILE_PATH should mean console-only
    adai::Logger::init(adai::Logger::Level::INFO,
                       {log_path, svc_config.log_max_size_mb, svc_config.log_max_files}, "adai");
    adai::ConfigLoader::validate_or_exit(svc_config, "trainer_service");  // TD-288

    auto control = std::make_shared<adai::TrainerServiceControlState>();
    control->service_started_unix = unix_seconds();
    g_control = control.get();
    adai::ignore_sighup("trainer_service");  // TD-283: SIGHUP used to terminate this process
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);

#ifdef _WIN32
    const std::string trainer_path = adai::sibling_executable_path(argv[0], "incremental_trainer.exe");
#else
    const std::string trainer_path = adai::sibling_executable_path(argv[0], "incremental_trainer");
#endif

    // Fixed prefix for every child this process ever launches — only --admin-port's value could
    // vary in principle, but this design always uses the one fixed trainer_child_admin_port
    // (only one child runs at a time, so no per-launch port negotiation is needed). TD-403: every
    // pass gets --objective and --dataset-kind, so the service trains any objective and pool.
    // TD-524: --admin-port only with the admin API on (each pass used to open the port regardless).
    const std::optional<int> child_admin_port =
        svc_config.trainer_admin_enabled ? std::optional<int>(svc_config.trainer_child_admin_port) : std::nullopt;
    const std::vector<std::string> child_argv = adai::trainer_service_child_argv(trainer_path, cli, child_admin_port);
    adai::Logger::info("trainer_service: objective {}, dataset pool '{}'", cli.objective,
                       adai::resolve_dataset_kind(cli.objective, cli.dataset_kind));

    adai::TrainerServiceProxy proxy(svc_config.trainer_admin_host, svc_config.trainer_admin_port,
                                    svc_config.trainer_admin_dir, control);
    std::thread proxy_thread;
    if (svc_config.trainer_admin_enabled) {
        adai::TrainerServiceProxy* proxy_ptr = &proxy;
        proxy_thread = std::thread([proxy_ptr] {
            if (!proxy_ptr->start()) {
                adai::Logger::error(
                    "trainer_service: admin proxy failed to bind — continuing with no admin "
                    "port");
            }
        });
        adai::Logger::info("trainer_service: admin API enabled on {}:{}",
                           svc_config.trainer_admin_host, svc_config.trainer_admin_port);
    } else {
        adai::Logger::info("trainer_service: admin API disabled (TRAINER_ADMIN_ENABLED=false)");
    }

    constexpr int kPollIntervalSeconds = 45;  // matches serve's own previous cadence
    // TD-523: consecutive failures double the wait from the poll interval up to this cap.
    constexpr int kMaxFailureBackoffSeconds = 900;
    adai::Logger::info(
        "trainer_service: supervisory loop starting (poll interval {}s); launching {}",
        kPollIntervalSeconds, trainer_path);

    // TD-173: how long a stop request waits for the current child to exit gracefully (SIGTERM)
    // before escalating to a force kill — see ChildProcess::stop_and_wait(). Inside
    // scripts/adai-trainer.service's own TimeoutStopSec=30, so systemd's external hard-kill
    // should never actually be needed; this is what makes that true even off systemd. TD-416: 25 s
    // (was 10), since a stopped pass now writes a checkpoint and releases its files before exiting.
    constexpr int kGracefulStopTimeoutMs = 25000;

    adai::ChildProcess child;
    while (!control->stop_requested.load()) {
        // TD-173: the piece TD-172 was missing — without this check, a paused pass draining and
        // exiting would immediately be followed by launching a brand-new, unpaused child. This is
        // exactly the same check `serve`'s own in-process loop used to make against its single,
        // process-lifetime TrainerControlState.
        if (control->paused.load()) {
            control->interruptible_sleep(kPollIntervalSeconds);
            continue;
        }

        if (!child.start(child_argv)) {
            // TD-524: backed off like a failed pass (it retried every poll interval forever).
            control->last_failure_unix = unix_seconds();
            const int failures = control->consecutive_failures.fetch_add(1) + 1;
            const int wait =
                adai::failure_backoff_seconds(failures, kPollIntervalSeconds, kMaxFailureBackoffSeconds);
            adai::Logger::error("trainer_service: failed to launch {} ({} in a row); next attempt in {}s",
                                trainer_path, failures, wait);
            control->interruptible_sleep(wait);
            continue;
        }
        control->total_passes_launched.fetch_add(1);
        control->last_launch_unix = unix_seconds();
        control->current_child_pid = child.pid();
        if (child_admin_port) proxy.set_child_port(*child_admin_port);

        int exit_code = 1;
        bool exited = false;
        while (!exited) {
            if (control->stop_requested.load()) {
                // Guaranteed to return (force-kills after kGracefulStopTimeoutMs if the child
                // doesn't exit on its own) — no more re-sending request_stop() in a tight loop
                // with no give-up point.
                child.stop_and_wait(kGracefulStopTimeoutMs, &exit_code);
                exited = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            exited = child.poll_exit(&exit_code);
        }
        control->current_child_pid = 0;
        control->last_exit_unix = unix_seconds();
        control->last_exit_code = exit_code;
        proxy.set_child_port(0);

        if (control->stop_requested.load()) {
            break;
        }

        // TD-523: `resume` exits 0 (trained), 3 (nothing pending) or 1 (failed). A failure used to
        // share exit 1 with "nothing pending", so a broken trainer looked idle and retried every
        // poll interval unreported, claiming and releasing files each time.
        switch (adai::classify_pass_exit(exit_code)) {
            case adai::PassOutcome::DidWork:
                control->total_passes_did_work.fetch_add(1);
                control->consecutive_failures = 0;
                continue;  // more work may already be pending — don't sleep
            case adai::PassOutcome::Idle:
                control->consecutive_failures = 0;
                control->interruptible_sleep(kPollIntervalSeconds);
                break;
            case adai::PassOutcome::Failed:
            case adai::PassOutcome::Crashed: {
                const bool crashed =
                    adai::classify_pass_exit(exit_code) == adai::PassOutcome::Crashed;
                (crashed ? control->total_passes_crashed : control->total_passes_failed)
                    .fetch_add(1);
                control->last_failure_unix = unix_seconds();
                const int failures = control->consecutive_failures.fetch_add(1) + 1;
                const int wait = adai::failure_backoff_seconds(failures, kPollIntervalSeconds,
                                                               kMaxFailureBackoffSeconds);
                adai::Logger::error(
                    "trainer_service: training pass {} with exit code {} ({} in a row); next "
                    "attempt in {}s — see the trainer log for the cause",
                    crashed ? "crashed" : "failed", exit_code, failures, wait);
                control->interruptible_sleep(wait);
                break;
            }
        }
    }

    adai::Logger::info("trainer_service: stop requested — shutting down");
    if (proxy_thread.joinable()) {
        proxy.stop();
        proxy_thread.join();
    }
    return 0;
}
