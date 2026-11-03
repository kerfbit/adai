// TD-430: Logger's first use from several threads at once, re-initializing while other threads log,
// and re-initializing under a new name. Its own binary: the first test needs a process in which nobody
// has created a logger yet, so nothing here may log before it forks.

// <wingdi.h> defines ERROR as a macro; Logger.hpp must still compile after it.
#define ERROR 0
#include "../src/Logger.hpp"
#undef ERROR

#include <gtest/gtest.h>
#include <atomic>
#include <fcntl.h>
#include <functional>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using adai::Logger;

namespace {

// Runs body in a forked child with stderr discarded and returns its exit status; a crash or a throw
// in the child stays in the child. Only that child is waited for.
int run_in_child(const std::function<int()>& body) {
    const pid_t pid = fork();
    if (pid == 0) {
        const int null_fd = open("/dev/null", O_WRONLY);
        if (null_fd >= 0) dup2(null_fd, STDERR_FILENO);
        int code = 2;
        try {
            code = body();
        } catch (...) {
            code = 3;
        }
        _exit(code);
    }
    if (pid < 0) return -1;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

// 8 threads released together each make their first log call; 1 if any call threw or the threads
// ended up with different loggers.
int concurrent_first_use() {
    constexpr int kThreads = 8;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<int> throws{0};
    std::vector<spdlog::logger*> seen(kThreads, nullptr);
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            ++ready;
            while (!go.load()) {
            }
            try {
                Logger::info("first use from thread {}", i);
                seen[i] = Logger::get().get();
            } catch (...) {
                ++throws;
            }
        });
    }
    while (ready.load() < kThreads) {
    }
    go = true;
    for (auto& t : threads) t.join();
    if (throws.load() != 0) return 1;
    for (auto* l : seen) {
        if (l != seen[0]) return 1;
    }
    return 0;
}

}  // namespace

// Must stay the first test: the children inherit a process in which no logger exists yet.
TEST(LoggerInitRace, ConcurrentFirstUseNeverThrows) {
    for (int trial = 0; trial < 50; ++trial) {
        ASSERT_EQ(run_in_child(concurrent_first_use), 0) << "trial " << trial;
    }
}

TEST(LoggerInitRace, ReinitializingWhileOtherThreadsLogIsSafe) {
    const int code = run_in_child([] {
        std::atomic<bool> stop{false};
        std::atomic<int> throws{0};
        std::vector<std::thread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&] {
                while (!stop.load()) {
                    try {
                        Logger::info("logging during re-init");
                        Logger::set_level(Logger::Level::INFO);
                    } catch (...) {
                        ++throws;
                    }
                }
            });
        }
        for (int i = 0; i < 200; ++i) {
            try {
                Logger::init(Logger::Level::INFO, i % 2 == 0 ? "adai" : "adai-alt");
            } catch (...) {
                ++throws;
            }
        }
        stop = true;
        for (auto& t : threads) t.join();
        return throws.load() == 0 ? 0 : 1;
    });
    EXPECT_EQ(code, 0);
}

TEST(LoggerInitRace, ReinitializingDropsThePreviousName) {
    Logger::init(Logger::Level::INFO, "td430-first");
    ASSERT_NE(spdlog::get("td430-first"), nullptr);
    Logger::init(Logger::Level::INFO, "td430-second");
    EXPECT_EQ(spdlog::get("td430-first"), nullptr);
    EXPECT_NE(spdlog::get("td430-second"), nullptr);
    // Re-using a name that's still registered doesn't throw either.
    EXPECT_NO_THROW(Logger::init(Logger::Level::INFO, "td430-second"));
    Logger::init(Logger::Level::INFO);
}

TEST(LoggerInitRace, ErrorLevelIsErr) {
    EXPECT_EQ(Logger::parse_level("error"), Logger::Level::ERR);
    Logger::set_level(Logger::Level::ERR);
    EXPECT_EQ(Logger::get()->level(), spdlog::level::err);
    Logger::set_level(Logger::Level::INFO);
}
