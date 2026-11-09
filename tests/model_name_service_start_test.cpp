/**
 * @file model_name_service_start_test.cpp
 * @brief TD-463: mns_server is the authority on model identity, so it must not start without a
 * database it can use.
 */
#include <gtest/gtest.h>

#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>

#include "../src/ModelNameService.hpp"

namespace fs = std::filesystem;

namespace {

fs::path fresh_dir(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() /
                         ("adai_mns_start_" + name + "_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

// True when start() gives up by itself. A start() that got past a bad database serves until
// stopped, on a port the kernel picks.
bool start_refuses(const fs::path& dir) {
    adai::ModelNameService service(dir.string(), 0);
    auto started = std::async(std::launch::async, [&] { return service.start(); });
    if (started.wait_for(std::chrono::seconds(3)) == std::future_status::ready)
        return !started.get();
    service.stop();
    started.wait();
    return false;
}

}  // namespace

TEST(ModelNameServiceStartTest, RefusesADatabaseThatCannotBeOpened) {
    const fs::path dir = fresh_dir("open");
    fs::create_directories(dir / "models.db");  // a directory can't be opened as a database
    EXPECT_TRUE(start_refuses(dir));
    fs::remove_all(dir);
}

TEST(ModelNameServiceStartTest, RefusesAFileThatIsNotADatabase) {
    const fs::path dir = fresh_dir("garbage");
    std::ofstream(dir / "models.db") << std::string(4096, 'x');
    EXPECT_TRUE(start_refuses(dir));
    fs::remove_all(dir);
}

TEST(ModelNameServiceStartTest, ServesWithAGoodDatabase) {
    const fs::path dir = fresh_dir("good");
    EXPECT_FALSE(start_refuses(dir));
    EXPECT_TRUE(fs::exists(dir / "models.db"));
    fs::remove_all(dir);
}
