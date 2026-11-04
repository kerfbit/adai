// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-11-04

/**
 * @file ChatbotGUI_wrapper.cpp
 * @brief Wrapper to launch chatbot_gui with correct environment settings
 *
 * This wrapper fixes library path conflicts with snap by:
 * - Unsetting problematic snap environment variables
 * - Removing snap entries from LD_LIBRARY_PATH and QT_PLUGIN_PATH, keeping the user's own in order
 * - Under snap, pointing Qt at the platform plugins of the Qt the GUI was built with
 * - Then exec'ing the actual GUI application
 * See adai::gui_launch::launch_environment() (TD-264).
 */

#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "ChatbotGuiLaunchEnv.hpp"

// TD-264: the platform plugin directory of the Qt chatbot_gui_binary was built with, from CMake.
#ifndef ADAI_QT_PLATFORM_PLUGIN_DIR
#define ADAI_QT_PLATFORM_PLUGIN_DIR ""
#endif

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

    // The platform plugins of the Qt this GUI was built with, if they're still there.
    std::string qt_platform_plugin_dir = ADAI_QT_PLATFORM_PLUGIN_DIR;
    struct stat st {};
    if (qt_platform_plugin_dir.empty() || stat(qt_platform_plugin_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        qt_platform_plugin_dir.clear();
    }

    const adai::gui_launch::EnvChanges changes = adai::gui_launch::launch_environment(
        [](const std::string& name) -> std::optional<std::string> {
            const char* value = getenv(name.c_str());
            return value ? std::optional<std::string>(value) : std::nullopt;
        },
        qt_platform_plugin_dir);
    for (const std::string& name : changes.unset) {
        unsetenv(name.c_str());
    }
    for (const auto& [name, value] : changes.set) {
        setenv(name.c_str(), value.c_str(), 1);
    }

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
