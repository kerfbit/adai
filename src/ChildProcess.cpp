// @adai-status: beta
// @adai-version: 0.8.0
// @adai-reviewed: 2026-11-10

#include "ChildProcess.hpp"
#include <cerrno>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "Logger.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#ifdef __linux__
#include <linux/close_range.h>
#endif
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

#ifndef _WIN32
// TD-281: execvp()'s PATH search, done before fork() so the child only calls execve(). A name with a
// '/' is used as given; otherwise the first executable regular file in PATH (default /usr/bin:/bin).
// Empty when there's none.
static std::string resolve_program(const std::string& name) {
    if (name.find('/') != std::string::npos) return name;
    const char* env_path = std::getenv("PATH");
    const std::string search = (env_path && *env_path) ? env_path : "/usr/bin:/bin";
    std::size_t start = 0;
    while (start <= search.size()) {
        const std::size_t end = std::min(search.find(':', start), search.size());
        const std::string dir = end > start ? search.substr(start, end - start) : ".";
        const std::string candidate = dir + "/" + name;
        struct stat st{};
        if (::stat(candidate.c_str(), &st) == 0 && S_ISREG(st.st_mode) && ::access(candidate.c_str(), X_OK) == 0) {
            return candidate;
        }
        start = end + 1;
    }
    return {};
}
#endif

// TD-282: the quoting CommandLineToArgvW and the Microsoft C runtime undo. An argument is quoted when
// it's empty or has a space, tab, newline, vertical tab or quote; inside quotes, backslashes are literal
// except before a quote (2n+1 backslashes and the quote for n backslashes and a literal quote) and before
// the closing quote (doubled). The old code escaped quotes only, so an argument ending in a backslash
// (C:\data\) swallowed the closing quote and merged with the next argument.
std::string ChildProcess::windows_command_line(const std::vector<std::string>& argv) {
    std::string cmdline;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (i > 0) cmdline += ' ';
        const std::string& arg = argv[i];
        if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) {
            cmdline += arg;
            continue;
        }
        cmdline += '"';
        std::size_t backslashes = 0;
        for (char c : arg) {
            if (c == '\\') {
                ++backslashes;
                continue;
            }
            if (c == '"') {
                cmdline.append(backslashes * 2 + 1, '\\');
            } else {
                cmdline.append(backslashes, '\\');
            }
            backslashes = 0;
            cmdline += c;
        }
        cmdline.append(backslashes * 2, '\\');
        cmdline += '"';
    }
    return cmdline;
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
    // TD-282: quoted so the child's CommandLineToArgvW/CRT parsing gets argv back exactly.
    std::string cmdline = windows_command_line(argv);

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

    // TD-281: the child of a multithreaded fork() may only make async-signal-safe calls. execvp()'s PATH
    // search isn't one (it may allocate), so the program is resolved here, in the parent, and the child
    // calls execve(). posix_spawn() would avoid fork() altogether but can't set the TD-280 parent-death
    // signal between fork and exec.
    const std::string program = resolve_program(argv[0]);
    if (program.empty()) {
        Logger::error("ChildProcess::start(): could not execute '{}': {}", argv[0], std::strerror(ENOENT));
        return false;
    }
    // Highest descriptor to mark close-on-exec if close_range() isn't available; getrlimit() isn't
    // async-signal-safe, so it's read here.
    rlimit fd_limit{};
    const int max_fd = (::getrlimit(RLIMIT_NOFILE, &fd_limit) == 0 && fd_limit.rlim_cur != RLIM_INFINITY)
                           ? static_cast<int>(std::min<rlim_t>(fd_limit.rlim_cur, 65536))
                           : 65536;
    // TD-278: a close-on-exec pipe reports a failed exec to start(). A successful exec closes the write
    // end, so the parent reads EOF; a failed one writes errno first. A failed exec used to look like a
    // launch (start() returned true) and surfaced only later as exit code 127.
    int exec_pipe[2];
    if (::pipe2(exec_pipe, O_CLOEXEC) != 0) {
        Logger::error("ChildProcess::start(): pipe2() failed: {}", std::strerror(errno));
        return false;
    }
    const pid_t parent = ::getpid();  // TD-280: for the child's parent-death check below
    const pid_t pid = ::fork();
    if (pid < 0) {
        Logger::error("ChildProcess::start(): fork() failed: {}", std::strerror(errno));
        ::close(exec_pipe[0]);
        ::close(exec_pipe[1]);
        return false;
    }
    if (pid == 0) {
        // Child: replace this process image entirely. execvp() only returns on failure.
        ::close(exec_pipe[0]);
#ifdef __linux__
        // TD-280: SIGTERM this process when the thread that started it exits, so a hard-killed
        // supervisor (kill -9, a crash, outside systemd's cgroup kill) doesn't leave the pass running,
        // holding its claims and the GPU, beside the pass a restarted supervisor launches. SIGTERM lets
        // the pass drain (TD-416). The setting survives exec. If the parent died before prctl(), we've
        // already been reparented: exit instead of running unsupervised. Note that it follows the
        // forking *thread*: trainer_service calls start() from its main thread, which lives as long as
        // the process. No process group is used: signalling one means kill() with a negative pid (see
        // TD-279).
        if (::prctl(PR_SET_PDEATHSIG, SIGTERM) != 0) {
            const int prctl_errno = errno;  // reported like a failed exec (TD-278)
            ssize_t ignored = ::write(exec_pipe[1], &prctl_errno, sizeof(prctl_errno));
            (void)ignored;
            _exit(127);
        }
        if (::getppid() != parent) _exit(127);  // nobody is left to read the pipe
#endif
        // TD-281: nothing past stdin/stdout/stderr survives the exec. Accepted admin connections (httplib
        // uses plain accept()) and anything else open without O_CLOEXEC used to stay open in the pass.
        // exec_pipe[1] is already close-on-exec, so a failed exec can still report through it.
        bool marked = false;
#if defined(__linux__) && defined(SYS_close_range)
        marked = ::syscall(SYS_close_range, 3U, ~0U, CLOSE_RANGE_CLOEXEC) == 0;
#endif
        if (!marked) {
            for (int fd = 3; fd < max_fd; ++fd) {
                const int flags = ::fcntl(fd, F_GETFD);
                if (flags >= 0) ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
            }
        }
        ::execve(program.c_str(), c_argv.data(), environ);
        // If we get here, exec failed — this is a forked copy of the supervisor, not the
        // supervisor itself, so report errno and exit immediately rather than returning into
        // shared logic twice. Only async-signal-safe calls here.
        const int exec_errno = errno;
        ssize_t ignored = ::write(exec_pipe[1], &exec_errno, sizeof(exec_errno));
        (void)ignored;
        _exit(127);
    }
    ::close(exec_pipe[1]);
    int exec_errno = 0;
    ssize_t n = 0;
    do {
        n = ::read(exec_pipe[0], &exec_errno, sizeof(exec_errno));
    } while (n < 0 && errno == EINTR);
    ::close(exec_pipe[0]);
    if (n == static_cast<ssize_t>(sizeof(exec_errno))) {
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
        Logger::error("ChildProcess::start(): could not execute '{}': {}", argv[0], std::strerror(exec_errno));
        return false;
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
    HANDLE h = static_cast<HANDLE>(process_handle_.load());
    // TD-282: the handle is signalled when the process ends. Comparing the exit code with STILL_ACTIVE
    // (259) used to keep a child that really exited with 259 "running" forever.
    if (::WaitForSingleObject(h, 0) != WAIT_OBJECT_0) {
        return false;
    }
    DWORD code = 0;
    if (!::GetExitCodeProcess(h, &code)) {
        Logger::error("ChildProcess::poll_exit(): GetExitCodeProcess failed (error {})",
                      static_cast<unsigned long>(::GetLastError()));
        code = static_cast<DWORD>(-1);
    }
    running_ = false;  // TD-279: before the handle
    process_handle_ = nullptr;
    ::CloseHandle(h);
    if (exit_code)
        *exit_code = static_cast<int>(code);
    return true;
#else
    int status = 0;
    const long long tracked = pid_.load();
    // TD-279: waitpid(-1, ...) would reap any child of this process, not ours.
    const pid_t result = tracked > 0 ? ::waitpid(static_cast<pid_t>(tracked), &status, WNOHANG) : -1;
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
    const long long reaped_pid = tracked;
    running_ = false;
    pid_ = -1;
    // TD-277: a signal death is 128 + the signal number (139 for SIGSEGV, 137 for SIGKILL), the shell
    // convention. Every signal death used to be 128, so a driver segfault and an OOM kill looked alike.
    int code = 0;
    if (WIFEXITED(status)) {
        code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        const int sig = WTERMSIG(status);
        code = 128 + sig;
#ifdef WCOREDUMP
        const char* core = WCOREDUMP(status) ? ", core dumped" : "";
#else
        const char* core = "";
#endif
        // SIGTERM is the graceful stop this class sends (request_stop()), so only other signals warn.
        if (sig == SIGTERM) {
            Logger::info("ChildProcess: pid {} was stopped by signal {} ({}){}", reaped_pid, sig, ::strsignal(sig), core);
        } else {
            Logger::warn("ChildProcess: pid {} was killed by signal {} ({}){}", reaped_pid, sig, ::strsignal(sig), core);
        }
    }
    if (exit_code) *exit_code = code;
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
    HANDLE handle = static_cast<HANDLE>(process_handle_.load());
    if (handle == nullptr) return;  // TD-279: stopped meanwhile
    if (!::GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, static_cast<DWORD>(::GetProcessId(handle)))) {
        Logger::warn(
            "ChildProcess::request_stop(): GenerateConsoleCtrlEvent failed (error {}) — falling "
            "back to TerminateProcess",
            static_cast<unsigned long>(::GetLastError()));
        ::TerminateProcess(handle, 1);
    }
#else
    // TD-279: read once and only a real pid; kill(-1, SIGTERM) would signal every process the user can.
    const long long pid = pid_.load();
    if (pid > 0) ::kill(static_cast<pid_t>(pid), SIGTERM);
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
    if (HANDLE handle = static_cast<HANDLE>(process_handle_.load())) ::TerminateProcess(handle, 1);
#else
    if (const long long pid = pid_.load(); pid > 0) ::kill(static_cast<pid_t>(pid), SIGKILL);  // TD-279
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
    running_ = false;  // TD-279: cleared first, so a concurrent request_stop() returns early
#ifdef _WIN32
    if (void* handle = process_handle_.exchange(nullptr)) {
        ::CloseHandle(static_cast<HANDLE>(handle));
    }
#else
    pid_ = -1;
#endif
    if (exit_code)
        *exit_code = -2;
    return true;
}

long long ChildProcess::pid() const {
    if (!running_) {
        return 0;
    }
#ifdef _WIN32
    return static_cast<long long>(::GetProcessId(static_cast<HANDLE>(process_handle_.load())));
#else
    const long long pid = pid_.load();
    return pid > 0 ? pid : 0;
#endif
}

}  // namespace adai
