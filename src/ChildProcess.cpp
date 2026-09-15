// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-14

#include "ChildProcess.hpp"
#include <chrono>
#include <thread>
#include "Logger.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#endif

namespace adai {

ChildProcess::~ChildProcess() {
    if (!running_) {
        return;
    }
    // TD-173: bounded, not the old unconditional blocking waitpid() (POSIX) / untimed handle leak
    // risk (Windows) — a child that never exits (this host's own documented GPU-driver *hang*,
    // not just crash, is exactly this failure mode) must not be able to hang this destructor
    // forever. Reuses stop_and_wait()'s own graceful-then-forceful escalation with a short 5s
    // grace period — implicit cleanup doesn't need the full 10s a deliberate shutdown gets.
    int exit_code = 0;
    stop_and_wait(5000, &exit_code);
}

bool ChildProcess::start(const std::vector<std::string>& argv) {
    if (running_) {
        Logger::warn("ChildProcess::start() called while a child is already running — ignored");
        return false;
    }
    if (argv.empty()) {
        Logger::error("ChildProcess::start() called with an empty argv");
        return false;
    }

#ifdef _WIN32
    // Build one command line string, quoting each argument. Minimal quoting (wrap in double
    // quotes, escape embedded quotes) — sufficient for the config paths/model names/port numbers
    // this class is actually launched with; not a general Windows command-line-escaping library.
    std::string cmdline;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i > 0)
            cmdline += ' ';
        cmdline += '"';
        for (char c : argv[i]) {
            if (c == '"')
                cmdline += '\\';
            cmdline += c;
        }
        cmdline += '"';
    }

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    // CREATE_NEW_PROCESS_GROUP: required for GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, ...) in
    // request_stop()/stop_and_wait() below to target this child specifically rather than this
    // process's own group (which would also affect trainer_service itself). Best-effort graceful
    // stop, not a guaranteed one — see request_stop()'s own doc comment.
    if (!::CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, FALSE,
                          CREATE_NEW_PROCESS_GROUP, nullptr, nullptr, &si, &pi)) {
        Logger::error("ChildProcess::start(): CreateProcessA failed (error {})",
                      static_cast<unsigned long>(::GetLastError()));
        return false;
    }
    ::CloseHandle(pi.hThread);
    process_handle_ = pi.hProcess;
    running_ = true;
    return true;
#else
    std::vector<char*> c_argv;
    c_argv.reserve(argv.size() + 1);
    for (const auto& a : argv) {
        c_argv.push_back(const_cast<char*>(a.c_str()));
    }
    c_argv.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0) {
        Logger::error("ChildProcess::start(): fork() failed");
        return false;
    }
    if (pid == 0) {
        // Child: replace this process image entirely. execvp() only returns on failure.
        ::execvp(c_argv[0], c_argv.data());
        // If we get here, exec failed — this is a forked copy of the supervisor, not the
        // supervisor itself, so exit immediately rather than returning into shared logic twice.
        _exit(127);
    }
    pid_ = pid;
    running_ = true;
    return true;
#endif
}

bool ChildProcess::poll_exit(int* exit_code) {
    if (!running_) {
        return false;
    }

#ifdef _WIN32
    HANDLE h = static_cast<HANDLE>(process_handle_);
    DWORD code = 0;
    if (!::GetExitCodeProcess(h, &code)) {
        Logger::error("ChildProcess::poll_exit(): GetExitCodeProcess failed");
        return false;
    }
    if (code == STILL_ACTIVE) {
        return false;
    }
    ::CloseHandle(h);
    process_handle_ = nullptr;
    running_ = false;
    if (exit_code)
        *exit_code = static_cast<int>(code);
    return true;
#else
    int status = 0;
    const pid_t result = ::waitpid(static_cast<pid_t>(pid_), &status, WNOHANG);
    if (result == 0) {
        return false;  // still running
    }
    if (result < 0) {
        // ECHILD etc. — treat as "no longer trackable", surface as exited with an
        // unmistakably-not-a-real-exit-code sentinel so callers don't mistake it for success.
        running_ = false;
        pid_ = -1;
        if (exit_code)
            *exit_code = -1;
        return true;
    }
    running_ = false;
    pid_ = -1;
    if (exit_code) {
        *exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128;
    }
    return true;
#endif
}

void ChildProcess::request_stop() {
    if (!running_) {
        return;
    }
#ifdef _WIN32
    // Best-effort: a console-control event reaches the child's own default handler (or a
    // registered one) rather than force-killing it outright, but — unlike POSIX SIGTERM — there
    // is no guarantee the child does anything graceful with it; requires CREATE_NEW_PROCESS_GROUP
    // at launch (see start()) to target this child specifically. Fall back to an immediate
    // TerminateProcess() only if even sending the event itself fails (e.g. group setup failed).
    if (!::GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT,
                                    static_cast<DWORD>(::GetProcessId(
                                        static_cast<HANDLE>(process_handle_))))) {
        Logger::warn(
            "ChildProcess::request_stop(): GenerateConsoleCtrlEvent failed (error {}) — falling "
            "back to TerminateProcess",
            static_cast<unsigned long>(::GetLastError()));
        ::TerminateProcess(static_cast<HANDLE>(process_handle_), 1);
    }
#else
    ::kill(static_cast<pid_t>(pid_), SIGTERM);
#endif
}

bool ChildProcess::stop_and_wait(int timeout_ms, int* exit_code) {
    if (!running_) {
        return false;
    }

    request_stop();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (poll_exit(exit_code)) {
            return true;  // exited gracefully within the grace period
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Still running after the grace period — escalate to an unconditional force kill.
    Logger::warn(
        "ChildProcess::stop_and_wait(): child did not exit within {}ms of the graceful stop "
        "request — force-killing",
        timeout_ms);
#ifdef _WIN32
    ::TerminateProcess(static_cast<HANDLE>(process_handle_), 1);
#else
    ::kill(static_cast<pid_t>(pid_), SIGKILL);
#endif

    // SIGKILL/TerminateProcess cannot be blocked or ignored, so this final wait is bounded in
    // practice by however long the OS takes to actually tear the process down (milliseconds,
    // barring an extremely rare kernel-level uninterruptible-sleep edge case genuinely out of
    // this class's reach either way) — a short poll loop rather than an unbounded blocking wait,
    // for exactly the same "never let a caller hang forever" reasoning as everywhere else here.
    const auto kill_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < kill_deadline) {
        if (poll_exit(exit_code)) {
            if (exit_code)
                *exit_code = -2;  // distinguish "we had to force-kill it" from a real exit code
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    Logger::error(
        "ChildProcess::stop_and_wait(): child still not reaped 5s after a force kill — giving up "
        "(this should not happen outside a wedged kernel-level process state)");
    // Force our own bookkeeping to match what we're telling the caller (gone), even though the
    // OS-level process may technically still be tearing down — a caller that gets `true` back
    // needs to be able to move on (e.g. launch a replacement child) rather than being stuck
    // forever believing this one is still "running" after every avenue to confirm otherwise has
    // been exhausted.
#ifdef _WIN32
    if (process_handle_) {
        ::CloseHandle(static_cast<HANDLE>(process_handle_));
        process_handle_ = nullptr;
    }
#else
    pid_ = -1;
#endif
    running_ = false;
    if (exit_code)
        *exit_code = -2;
    return true;
}

long long ChildProcess::pid() const {
    if (!running_) {
        return 0;
    }
#ifdef _WIN32
    return static_cast<long long>(::GetProcessId(static_cast<HANDLE>(process_handle_)));
#else
    return pid_;
#endif
}

}  // namespace adai
