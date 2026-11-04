// Tests for adai::gui_launch (TD-264): the environment the chatbot_gui wrapper gives chatbot_gui_binary.

#include <gtest/gtest.h>
#include <algorithm>
#include <map>
#include "../src/ChatbotGuiLaunchEnv.hpp"

using adai::gui_launch::EnvChanges;
using adai::gui_launch::is_snap_environment;
using adai::gui_launch::launch_environment;
using adai::gui_launch::without_snap_entries;

namespace {

const std::string kPluginDir = "/opt/qt6/plugins/platforms";

adai::gui_launch::EnvLookup lookup(const std::map<std::string, std::string>& vars) {
    return [vars](const std::string& name) -> std::optional<std::string> {
        const auto it = vars.find(name);
        return it == vars.end() ? std::nullopt : std::optional<std::string>(it->second);
    };
}

bool unsets(const EnvChanges& c, const std::string& name) {
    return std::find(c.unset.begin(), c.unset.end(), name) != c.unset.end();
}

std::optional<std::string> sets(const EnvChanges& c, const std::string& name) {
    for (const auto& [key, value] : c.set) {
        if (key == name) {
            return value;
        }
    }
    return std::nullopt;
}

}  // namespace

TEST(ChatbotGuiLaunchEnv, WithoutSnapEntriesKeepsTheOthersInOrder) {
    EXPECT_EQ(without_snap_entries("/opt/b:/snap/core20/current/lib:/opt/a"), "/opt/b:/opt/a");
    EXPECT_EQ(without_snap_entries("/opt/b:/opt/a"), "/opt/b:/opt/a");
    EXPECT_EQ(without_snap_entries("/snap/x:/snap/y"), "");
    EXPECT_EQ(without_snap_entries("/snap/x"), "");
    EXPECT_EQ(without_snap_entries("/home/me/snapshots/lib"), "/home/me/snapshots/lib");
}

TEST(ChatbotGuiLaunchEnv, SnapEnvironmentDetection) {
    EXPECT_FALSE(is_snap_environment(lookup({})));
    EXPECT_FALSE(is_snap_environment(lookup({{"LD_LIBRARY_PATH", "/opt/lib"}})));
    EXPECT_TRUE(is_snap_environment(lookup({{"SNAP", "/snap/code/180"}})));
    EXPECT_TRUE(is_snap_environment(lookup({{"GTK_PATH", "/snap/code/180/usr/lib/gtk-3.0"}})));
    EXPECT_TRUE(is_snap_environment(lookup({{"GIO_MODULE_DIR", "/home/me/snap/code/common/.cache/gio-modules"}})));
    EXPECT_TRUE(is_snap_environment(lookup({{"LD_LIBRARY_PATH", "/opt/lib:/snap/core20/current/lib"}})));
    EXPECT_TRUE(is_snap_environment(lookup({{"QT_PLUGIN_PATH", "/snap/kde/1/plugins"}})));
}

// The old wrapper put /usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu in front of LD_LIBRARY_PATH and
// dropped a list with any snap entry in it entirely.
TEST(ChatbotGuiLaunchEnv, LibraryPathKeepsTheUsersEntriesInOrderAndAddsNoSystemPaths) {
    const EnvChanges c =
        launch_environment(lookup({{"LD_LIBRARY_PATH", "/opt/b:/snap/core20/current/lib:/opt/a"}}), kPluginDir);
    EXPECT_EQ(sets(c, "LD_LIBRARY_PATH"), "/opt/b:/opt/a");

    const EnvChanges plain = launch_environment(lookup({{"LD_LIBRARY_PATH", "/opt/b:/opt/a"}}), kPluginDir);
    EXPECT_FALSE(sets(plain, "LD_LIBRARY_PATH"));
    EXPECT_FALSE(unsets(plain, "LD_LIBRARY_PATH"));

    const EnvChanges none = launch_environment(lookup({}), kPluginDir);
    EXPECT_FALSE(sets(none, "LD_LIBRARY_PATH"));
    EXPECT_FALSE(unsets(none, "LD_LIBRARY_PATH"));

    const EnvChanges only_snap = launch_environment(lookup({{"LD_LIBRARY_PATH", "/snap/x/lib"}}), kPluginDir);
    EXPECT_TRUE(unsets(only_snap, "LD_LIBRARY_PATH"));
}

TEST(ChatbotGuiLaunchEnv, QtPluginPathLosesItsSnapEntries) {
    const EnvChanges c = launch_environment(lookup({{"QT_PLUGIN_PATH", "/snap/kde/1/plugins:/opt/myplugins"}}), kPluginDir);
    EXPECT_EQ(sets(c, "QT_PLUGIN_PATH"), "/opt/myplugins");
}

// The old wrapper always set /usr/lib/x86_64-linux-gnu/qt5/plugins, even for a Qt6 build.
TEST(ChatbotGuiLaunchEnv, PlatformPluginPathOnlyUnderSnapAndFromTheBuild) {
    EXPECT_FALSE(sets(launch_environment(lookup({}), kPluginDir), "QT_QPA_PLATFORM_PLUGIN_PATH"));

    const EnvChanges snap = launch_environment(lookup({{"SNAP", "/snap/code/180"}}), kPluginDir);
    EXPECT_EQ(sets(snap, "QT_QPA_PLATFORM_PLUGIN_PATH"), kPluginDir);

    // No directory known from the build: Qt finds its own.
    EXPECT_FALSE(sets(launch_environment(lookup({{"SNAP", "/snap/code/180"}}), ""), "QT_QPA_PLATFORM_PLUGIN_PATH"));
}

TEST(ChatbotGuiLaunchEnv, UsersOwnPlatformPluginPathIsKept) {
    const EnvChanges c = launch_environment(
        lookup({{"SNAP", "/snap/code/180"}, {"QT_QPA_PLATFORM_PLUGIN_PATH", "/opt/custom/platforms"}}), kPluginDir);
    EXPECT_FALSE(sets(c, "QT_QPA_PLATFORM_PLUGIN_PATH"));
    EXPECT_FALSE(unsets(c, "QT_QPA_PLATFORM_PLUGIN_PATH"));

    const EnvChanges replaced = launch_environment(
        lookup({{"SNAP", "/snap/code/180"}, {"QT_QPA_PLATFORM_PLUGIN_PATH", "/snap/kde/1/plugins/platforms"}}), kPluginDir);
    EXPECT_EQ(sets(replaced, "QT_QPA_PLATFORM_PLUGIN_PATH"), kPluginDir);

    const EnvChanges dropped = launch_environment(
        lookup({{"QT_QPA_PLATFORM_PLUGIN_PATH", "/snap/kde/1/plugins/platforms"}}), "");
    EXPECT_TRUE(unsets(dropped, "QT_QPA_PLATFORM_PLUGIN_PATH"));
}

TEST(ChatbotGuiLaunchEnv, SnapVariablesAreUnset) {
    const EnvChanges c = launch_environment(
        lookup({{"SNAP", "/snap/code/180"}, {"GIO_MODULE_DIR", "/home/me/snap/code/common/.cache/gio-modules"}}), "");
    for (const char* name : {"SNAP", "SNAP_COMMON", "SNAP_DATA", "GTK_PATH", "GIO_MODULE_DIR"}) {
        EXPECT_TRUE(unsets(c, name)) << name;
    }
    EXPECT_EQ(sets(c, "GTK_MODULES"), "");

    const EnvChanges own_gio = launch_environment(lookup({{"GIO_MODULE_DIR", "/opt/gio"}}), "");
    EXPECT_FALSE(unsets(own_gio, "GIO_MODULE_DIR"));
}
