/**
 * @file model_name_service_start_test.cpp
 * @brief TD-463: mns_server is the authority on model identity, so it must not start without a
 * database it can use.
 */
#include <gtest/gtest.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <thread>

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

// TD-464: httplib's stop() does nothing until listen() is under way, so a stop() that came first (an
// early SIGTERM, or start_refuses() above giving up on a start slowed by a loaded machine) was lost
// and start() served forever. Every start() below must return soon after stop(), however the two
// interleave: before start() is called, during the database work, or around the bind. A start() that
// never returns is leaked and detached, so the failure shows as a failed test, not a hung binary.
TEST(ModelNameServiceStartTest, StopBeforeOrDuringStartAlwaysEndsStart) {
    const fs::path dir = fresh_dir("stop_race");
    int stuck = 0;
    for (int i = -1; i < 100 && stuck == 0; ++i) {  // i = -1: stop() before start() is even called
        auto* service = new adai::ModelNameService(dir.string(), 0);
        auto done = std::make_shared<std::atomic<bool>>(false);
        auto started = std::make_shared<std::atomic<bool>>(true);
        if (i < 0)
            service->stop();
        std::thread t([service, done, started] {
            *started = service->start();
            *done = true;
        });
        if (i > 0)
            std::this_thread::sleep_for(std::chrono::microseconds(100 * (i % 40)));
        if (i >= 0)
            service->stop();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!*done && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (*done) {
            t.join();
            if (i < 0)
                EXPECT_FALSE(*started) << "a start() after stop() must not serve";
            delete service;
        } else {
            ++stuck;
            t.detach();  // leaked: deleting it under a blocked listen() would be worse
        }
    }
    EXPECT_EQ(stuck, 0) << "start() kept serving after stop()";
    if (stuck == 0)
        fs::remove_all(dir);
}
