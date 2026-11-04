// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-04

#include "ChatbotGuiLaunchEnv.hpp"

namespace adai::gui_launch {

bool is_snap_path(const std::string& path) {
    return path.find("/snap/") != std::string::npos;
}

std::string without_snap_entries(const std::string& path_list) {
    std::string kept;
    bool first = true;
    size_t start = 0;
    while (true) {
        const size_t colon = path_list.find(':', start);
        const std::string entry =
            path_list.substr(start, colon == std::string::npos ? std::string::npos : colon - start);
        if (!is_snap_path(entry)) {
            if (!first) {
                kept += ':';
            }
            kept += entry;
            first = false;
        }
        if (colon == std::string::npos) {
            break;
        }
        start = colon + 1;
    }
    return kept;
}

bool is_snap_environment(const EnvLookup& env) {
    if (env("SNAP")) {
        return true;
    }
    for (const char* name : {"LD_LIBRARY_PATH", "GTK_PATH", "GIO_MODULE_DIR", "QT_PLUGIN_PATH"}) {
        const auto value = env(name);
        if (value && is_snap_path(*value)) {
            return true;
        }
    }
    return false;
}

EnvChanges launch_environment(const EnvLookup& env, const std::string& qt_platform_plugin_dir) {
    EnvChanges changes;
    const bool in_snap = is_snap_environment(env);

    for (const char* name : {"GTK_PATH", "SNAP", "SNAP_COMMON", "SNAP_DATA"}) {
        changes.unset.emplace_back(name);
    }
    const auto gio = env("GIO_MODULE_DIR");
    if (gio && is_snap_path(*gio)) {
        changes.unset.emplace_back("GIO_MODULE_DIR");
    }

    // TD-264: only snap entries are removed; the user's own entries keep their order, and no system
    // directory is added (the dynamic linker already searches the system's own).
    for (const char* name : {"LD_LIBRARY_PATH", "QT_PLUGIN_PATH"}) {
        const auto value = env(name);
        if (!value) {
            continue;
        }
        const std::string kept = without_snap_entries(*value);
        if (kept.empty()) {
            changes.unset.emplace_back(name);
        } else if (kept != *value) {
            changes.set.emplace_back(name, kept);
        }
    }

    // TD-264: the build's own Qt (Qt5 or Qt6), and only where a snap could have pointed Qt elsewhere.
    const auto platform_path = env("QT_QPA_PLATFORM_PLUGIN_PATH");
    if (in_snap && !qt_platform_plugin_dir.empty() && (!platform_path || is_snap_path(*platform_path))) {
        changes.set.emplace_back("QT_QPA_PLATFORM_PLUGIN_PATH", qt_platform_plugin_dir);
    } else if (platform_path && is_snap_path(*platform_path)) {
        changes.unset.emplace_back("QT_QPA_PLATFORM_PLUGIN_PATH");
    }

    changes.set.emplace_back("GTK_MODULES", "");  // suppress GTK module warnings
    return changes;
}

}  // namespace adai::gui_launch
