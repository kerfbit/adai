#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-04

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief The environment the `chatbot_gui` wrapper gives `chatbot_gui_binary` (TD-264).
 *
 * A terminal inside a snap (VS Code's, for instance) leaks the snap's library, GTK, GIO and Qt plugin
 * paths into every program started from it, and those make a system-Qt program load the snap's copies
 * and crash. The wrapper removes only what comes from a snap: nothing it sets is specific to an
 * architecture, a distribution or a Qt major version, and the user's own path entries keep their order.
 */
namespace adai::gui_launch {

/// Reads one environment variable; std::nullopt when it isn't set.
using EnvLookup = std::function<std::optional<std::string>(const std::string&)>;

struct EnvChanges {
    std::vector<std::string> unset;
    std::vector<std::pair<std::string, std::string>> set;
};

/// True for a path inside a snap (contains "/snap/").
bool is_snap_path(const std::string& path);

/// A ':'-separated path list without its snap entries; the other entries keep their order.
std::string without_snap_entries(const std::string& path_list);

/// The program runs in a snap's environment: SNAP is set, or a library, GTK, GIO or Qt plugin path points into one.
bool is_snap_environment(const EnvLookup& env);

/**
 * @brief The changes to make before running chatbot_gui_binary.
 *
 * Always unsets SNAP, SNAP_COMMON, SNAP_DATA and GTK_PATH and empties GTK_MODULES (the wrapper's
 * long-standing behaviour). Removes snap entries from LD_LIBRARY_PATH and QT_PLUGIN_PATH (unset when
 * nothing is left) and unsets a snap GIO_MODULE_DIR. Only in a snap environment, and only when the
 * user hasn't set it to a path outside a snap, sets QT_QPA_PLATFORM_PLUGIN_PATH to
 * `qt_platform_plugin_dir`, the platform plugin directory of the Qt the GUI was built with (empty: never).
 */
EnvChanges launch_environment(const EnvLookup& env, const std::string& qt_platform_plugin_dir);

}  // namespace adai::gui_launch
