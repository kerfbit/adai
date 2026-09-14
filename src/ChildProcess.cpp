// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-14

#include "ChildProcess.hpp"
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
    // Best-effort cleanup: ask it to stop, then reap so we don't leave a zombie (POSIX) or a
    // dangling handle (Windows) behind if the owner is destroyed while a child is still alive.
    request_stop();
#ifdef _WIN32
    if (process_handle_) {
        ::WaitForSingleObject(static_cast<HANDLE>(process_handle_), 5000);
        ::CloseHandle(static_cast<HANDLE>(process_handle_));
    }
#else
    if (pid_ > 0) {
        int status = 0;
        ::waitpid(static_cast<pid_t>(pid_), &status, 0);
    }
#endif
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
    if (!::CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr,
                          &si, &pi)) {
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
    ::TerminateProcess(static_cast<HANDLE>(process_handle_), 1);
#else
    ::kill(static_cast<pid_t>(pid_), SIGTERM);
#endif
}

}  // namespace adai
