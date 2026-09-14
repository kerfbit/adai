// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-14

// TD-172: trainer_service — the thin process-supervisor binary that replaces
// `incremental_trainer serve`. Deliberately minimal: it does not link adai_models/adai_nlp and
// never touches GPU/CUDA/SYCL objects at all — every actual training pass runs inside a
// single-pass `incremental_trainer --foreground --admin-port <N> resume` child this process
// launches and monitors (see ChildProcess.hpp). This binary's own job is just:
//   1. Launch a child, wait for it to exit, note whether it did work (exit 0) or not (exit 1).
//   2. If it did work, launch another one immediately (more may be pending); otherwise sleep the
//      poll interval first — the same shape `serve`'s old in-process loop already used.
//   3. Host an always-on admin HTTP listener (TrainerServiceProxy) that proxies /admin/* requests
//      to whichever child is currently alive, on the private port assigned via --admin-port.
//   4. Forward a graceful-stop request (SIGTERM/SIGINT) to whatever child is currently running.
//
// TrainerControlState/TrainerAdminAPI are used completely unchanged — by the *child* process,
// exactly as `serve` already used them — see TrainerServiceProxy.hpp's own doc comment for why
// this design needs no changes to either class.

#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
#include "ChildProcess.hpp"
#include "Config.hpp"
#include "IncrementalTrainerArgs.hpp"
#include "Logger.hpp"
#include "TrainerServiceProxy.hpp"

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_stop_requested{false};

// Only async-signal-safe work here (matches IncrementalTrainingTool.cpp's own signal_handler()
// doc comment on why this doesn't log directly) — the main loop below observes the flag and logs
// the graceful-shutdown message itself, outside signal context.
void signal_handler(int sig) {
    if (sig == SIGTERM || sig == SIGINT) {
        g_stop_requested = true;
    }
}

// Sleeps up to `seconds`, checking g_stop_requested once per second so a shutdown request is
// noticed promptly instead of only after the full interval — deliberately simple (no condition
// variable) since trainer_service owns no other thread that needs to wake this one early the way
// TrainerControlState::wake() does for admin-triggered resume; a stop request is the only thing
// that should ever cut this short.
void interruptible_sleep(int seconds) {
    for (int waited = 0; waited < seconds && !g_stop_requested.load(); ++waited) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// Resolves the incremental_trainer binary this process launches per pass. Both binaries are
// installed side by side (see scripts/install_incremental_trainer.sh) — if argv[0] carries a
// directory component, reuse it directly rather than relying on PATH; otherwise fall back to the
// bare name, letting execvp()/CreateProcessA's own PATH search find it (matches how this process
// itself was presumably invoked, e.g. via a systemd unit's absolute ExecStart or a bare name on
// an operator's PATH).
std::string resolve_incremental_trainer_path(const std::string& argv0) {
#ifdef _WIN32
    const std::string name = "incremental_trainer.exe";
#else
    const std::string name = "incremental_trainer";
#endif
    const fs::path p(argv0);
    if (p.has_parent_path()) {
        return (p.parent_path() / name).string();
    }
    return name;
}

}  // namespace

int main(int argc, char* argv[]) {
    // Reuses incremental_trainer's own global-flag parser (IncrementalTrainerArgs.hpp):
    // trainer_service accepts the same --config/--model/--gpu-strategy flags, passed straight
    // through to every child it launches. --foreground/--admin-port are meaningless for
    // trainer_service's own invocation (they're what THIS process passes to each child, not
    // something an operator passes to trainer_service itself); any leftover positional args are
    // ignored — trainer_service has no "commands" of its own, unlike incremental_trainer.
    const adai::IncrementalTrainerGlobalArgs cli =
        adai::parse_incremental_trainer_global_args(argc, argv);

    const std::string config_path = adai::ConfigLoader::discover_config_path(
        cli.config_path.value_or(""), "config.trainer.conf");
    const adai::ServiceConfig svc_config = adai::ConfigLoader::load(config_path);

    const std::string log_path =
        svc_config.log_file_path.empty() ? "chatbot_server.log" : svc_config.log_file_path;
    adai::Logger::init(adai::Logger::Level::INFO,
                       {log_path, svc_config.log_max_size_mb, svc_config.log_max_files}, "adai");

    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);

    const std::string trainer_path = resolve_incremental_trainer_path(argv[0]);
    const std::string admin_port_str = std::to_string(svc_config.trainer_child_admin_port);

    // Fixed prefix for every child this process ever launches — only --admin-port's value could
    // vary in principle, but this design always uses the one fixed trainer_child_admin_port
    // (only one child runs at a time, so no per-launch port negotiation is needed).
    std::vector<std::string> child_argv = {trainer_path};
    if (cli.config_path) {
        child_argv.push_back("--config");
        child_argv.push_back(*cli.config_path);
    }
    if (cli.model_name) {
        child_argv.push_back("--model");
        child_argv.push_back(*cli.model_name);
    }
    if (cli.gpu_strategy) {
        child_argv.push_back("--gpu-strategy");
        child_argv.push_back(*cli.gpu_strategy);
    }
    child_argv.push_back("--foreground");
    child_argv.push_back("--admin-port");
    child_argv.push_back(admin_port_str);
    child_argv.push_back("resume");

    adai::TrainerServiceProxy proxy(svc_config.trainer_admin_host, svc_config.trainer_admin_port,
                                    svc_config.trainer_admin_dir);
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
    adai::Logger::info(
        "trainer_service: supervisory loop starting (poll interval {}s); launching {}",
        kPollIntervalSeconds, trainer_path);

    adai::ChildProcess child;
    while (!g_stop_requested.load()) {
        if (!child.start(child_argv)) {
            adai::Logger::error(
                "trainer_service: failed to launch incremental_trainer — retrying after poll "
                "interval");
            interruptible_sleep(kPollIntervalSeconds);
            continue;
        }
        proxy.set_child_port(svc_config.trainer_child_admin_port);

        int exit_code = 1;
        bool exited = false;
        while (!exited) {
            if (g_stop_requested.load()) {
                child.request_stop();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            exited = child.poll_exit(&exit_code);
        }
        proxy.set_child_port(0);

        if (g_stop_requested.load()) {
            break;
        }

        const bool did_work = (exit_code == 0);
        if (did_work) {
            continue;  // more work may already be pending — don't sleep
        }
        interruptible_sleep(kPollIntervalSeconds);
    }

    adai::Logger::info("trainer_service: stop requested — shutting down");
    if (proxy_thread.joinable()) {
        proxy.stop();
        proxy_thread.join();
    }
    return 0;
}
