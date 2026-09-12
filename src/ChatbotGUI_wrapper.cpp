// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-11

/**
 * @file ChatbotGUI_wrapper.cpp
 * @brief Wrapper to launch chatbot_gui with correct environment settings
 *
 * This wrapper fixes library path conflicts (especially with snap) by:
 * - Unsetting problematic snap environment variables
 * - Setting correct library paths to system libraries
 * - Then exec'ing the actual GUI application
 */

#include <limits.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

extern char** environ;

// Resolve the directory this executable actually lives in.
//
// argv[0] is NOT a reliable way to find our own location: when invoked via a
// PATH lookup with a bare command name (e.g. a .desktop launcher's `Exec=
// chatbot_gui`, or the wrapper symlinked into a bin directory), the shell/
// exec passes argv[0] as literally "chatbot_gui" with no '/' in it at all,
// so a naive find_last_of("/") falls back to "." — the *caller's* current
// working directory, not this binary's install directory — and the
// subsequent exec of "./chatbot_gui_binary" fails unless the caller happens
// to be cd'ed into the same directory. TD-103.
//
// /proc/self/exe (Linux) always resolves to this process's real executable
// path regardless of how it was invoked, so prefer that and only fall back
// to argv[0]-based parsing (best-effort) if it's unavailable.
std::string resolve_exe_dir(const char* argv0) {
    char buf[PATH_MAX];
    const ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        const std::string self_path(buf);
        const size_t slash = self_path.find_last_of('/');
        return (slash != std::string::npos) ? self_path.substr(0, slash) : ".";
    }

    // Fallback: argv[0]-based (wrong if invoked via PATH with no '/' in argv[0]).
    const std::string wrapper_path = argv0 ? argv0 : "";
    const size_t last_slash = wrapper_path.find_last_of("/");
    return (last_slash != std::string::npos) ? wrapper_path.substr(0, last_slash) : ".";
}

int main(int argc, char* argv[]) {
    // Get the directory where this wrapper is located
    std::string exe_dir = resolve_exe_dir(argv[0]);

    // Path to the actual GUI executable
    std::string gui_executable = exe_dir + "/chatbot_gui_binary";

    // Environment variables to unset (snap-related)
    const char* unset_vars[] = {"GTK_PATH", "SNAP", "SNAP_COMMON", "SNAP_DATA", nullptr};

    // Unset problematic variables
    for (int i = 0; unset_vars[i] != nullptr; ++i) {
        unsetenv(unset_vars[i]);
    }

    // Set correct library paths (system libraries, not snap)
    // Preserve any existing non-snap paths
    const char* old_ld_path = getenv("LD_LIBRARY_PATH");
    std::string new_ld_path = "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu";

    if (old_ld_path && strstr(old_ld_path, "/snap/") == nullptr) {
        // If there's an existing LD_LIBRARY_PATH that doesn't contain snap paths, append it
        new_ld_path += ":";
        new_ld_path += old_ld_path;
    }

    setenv("LD_LIBRARY_PATH", new_ld_path.c_str(), 1);

    // Set Qt plugin path
    setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/usr/lib/x86_64-linux-gnu/qt5/plugins", 1);

    // Suppress GTK warnings
    setenv("GTK_MODULES", "", 1);

    // Prepare arguments for exec
    std::vector<char*> new_argv;
    new_argv.push_back(const_cast<char*>(gui_executable.c_str()));

    // Copy all arguments except argv[0]
    for (int i = 1; i < argc; ++i) {
        new_argv.push_back(argv[i]);
    }
    new_argv.push_back(nullptr);

    // Execute the actual GUI
    execvp(gui_executable.c_str(), new_argv.data());

    // If we get here, exec failed
    std::cerr << "Error: Failed to execute " << gui_executable << std::endl;
    std::cerr << "Error: " << strerror(errno) << std::endl;
    return 1;
}
