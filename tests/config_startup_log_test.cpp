// TD-290: config messages logged before the binary sets up its logger reach the log file once it
// does. A binary of its own, so nothing has initialised the logger before the test runs.

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "../src/Config.hpp"
#include "../src/Logger.hpp"

namespace fs = std::filesystem;

TEST(ConfigStartupLogTest, LoadWarningsBeforeLoggerInitReachTheLogFile) {
    const fs::path dir = fs::temp_directory_path() / ("adai_td290_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path conf = dir / "config.conf";
    std::ofstream(conf) << "PORT=8080x\nNOT_A_REAL_KEY=1\nHIPPOCAMPAL_COVERAGE_LOSS_WEIGHT=0.0\n";

    // As every binary does: load the config, then set up the logger from it.
    const adai::ServiceConfig config = adai::ConfigLoader::load(conf.string());
    const fs::path log_path = dir / "service.log";
    adai::Logger::init(adai::Logger::Level::INFO, adai::Logger::FileConfig{log_path.string(), 10, 2});
    adai::Logger::info("after init");
    adai::Logger::get()->flush();

    std::stringstream log;
    log << std::ifstream(log_path).rdbuf();
    const std::string text = log.str();
    EXPECT_NE(text.find("Loading configuration from: " + conf.string()), std::string::npos) << text;
    EXPECT_NE(text.find("Invalid value for PORT: 8080x"), std::string::npos) << text;
    EXPECT_NE(text.find("Unknown configuration key: NOT_A_REAL_KEY"), std::string::npos) << text;
    // A reserved key shipped in the config files is accepted without a warning.
    EXPECT_EQ(text.find("HIPPOCAMPAL_COVERAGE_LOSS_WEIGHT"), std::string::npos) << text;
    // The start-up messages come before the logger's own lines.
    EXPECT_LT(text.find("Loading configuration from"), text.find("after init"));
    fs::remove_all(dir);
}
