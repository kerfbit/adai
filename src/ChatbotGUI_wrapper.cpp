/**
 * @file ChatbotGUI_wrapper.cpp
 * @brief Wrapper to launch chatbot_gui with correct environment settings
 *
 * This wrapper fixes library path conflicts (especially with snap) by:
 * - Unsetting problematic snap environment variables
 * - Setting correct library paths to system libraries
 * - Then exec'ing the actual GUI application
 */

#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

extern char** environ;

int main(int argc, char* argv[]) {
    // Get the directory where this wrapper is located
    std::string wrapper_path = argv[0];
    size_t last_slash = wrapper_path.find_last_of("/");
    std::string exe_dir =
        (last_slash != std::string::npos) ? wrapper_path.substr(0, last_slash) : ".";

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
