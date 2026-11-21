/**
 * @file mns_progress_test.cpp
 * @brief TD-269: MNS records a run's optimizer step (progress.global_step, and each history
 * entry's global_step), keeps it across a restart, resets it for a new run, and adds the columns to
 * a database created before they existed.
 *
 * Runs ModelNameService in-process on a free port with its own data directory, unlike
 * mnsLiveTests, which needs a running mns_server.
 */
#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sqlite3.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "../src/ModelNameClient.hpp"
#include "../src/ModelNameService.hpp"

namespace fs = std::filesystem;

namespace {

int free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    ::close(fd);
    return ntohs(addr.sin_port);
}

fs::path fresh_dir(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() /
                         ("adai_mns_progress_" + name + "_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

// One mns_server instance on a data directory, stopped when it goes out of scope.
class RunningMns {
   public:
    explicit RunningMns(const fs::path& dir)
        : port_(free_port()), service_(std::make_unique<adai::ModelNameService>(dir.string(), port_)) {
        started_ = std::async(std::launch::async, [this] { return service_->start(); });
        httplib::Client probe("127.0.0.1", port_);
        for (int i = 0; i < 100; ++i) {
            if (auto res = probe.Get("/health"); res && res->status == 200) {
                ready_ = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    ~RunningMns() {
        service_->stop();
        started_.wait();
    }
    bool ready() const { return ready_; }
    int port() const { return port_; }
    httplib::Client client() const { return httplib::Client("127.0.0.1", port_); }

   private:
    int port_;
    std::unique_ptr<adai::ModelNameService> service_;
    std::future<bool> started_;
    bool ready_ = false;
};

const char* kArch =
    "\"arch\":{\"d_model\":64,\"num_heads\":2,\"d_ff\":128,\"num_encoder_layers\":1,"
    "\"num_decoder_layers\":1,\"max_seq_length\":16}";

nlohmann::json get_record(httplib::Client& c, const std::string& name) {
    auto res = c.Get("/models/" + name);
    if (!res || res->status != 200)
        return {};
    return nlohmann::json::parse(res->body, nullptr, false);
}

std::string start_training(httplib::Client& c, const std::string& name, bool new_run) {
    auto res = c.Put("/models/" + name + "/state",
                     std::string("{\"state\":\"training\",\"new_run\":") + (new_run ? "true" : "false") +
                         "}",
                     "application/json");
    if (!res || res->status != 200)
        return "";
    return nlohmann::json::parse(res->body, nullptr, false).value("run_id", "");
}

int push(httplib::Client& c, const std::string& name, const std::string& run_id, int epoch,
         const std::string& extra) {
    auto res = c.Put("/models/" + name + "/progress",
                     "{\"run_id\":\"" + run_id + "\",\"session_id\":\"session-01\",\"epoch\":" +
                         std::to_string(epoch) + ",\"loss\":0.5,\"best_loss\":0.4" + extra + "}",
                     "application/json");
    return res ? res->status : 0;
}

}  // namespace

TEST(MnsProgressTest, GlobalStepIsKeptAcrossARestartAndResetForANewRun) {
    const fs::path dir = fresh_dir("restart");
    std::string run_id;
    {
        RunningMns mns(dir);
        ASSERT_TRUE(mns.ready());
        auto c = mns.client();
        ASSERT_EQ(c.Post("/models", std::string("{\"model_name\":\"progress-model\",") + kArch + "}",
                         "application/json")
                      ->status,
                  201);
        run_id = start_training(c, "progress-model", false);
        ASSERT_FALSE(run_id.empty());
        EXPECT_EQ(get_record(c, "progress-model")["progress"]["global_step"], 0);

        ASSERT_EQ(push(c, "progress-model", run_id, 3, ",\"global_step\":5000000000"), 200);  // beyond int
        EXPECT_EQ(get_record(c, "progress-model")["progress"]["global_step"], 5000000000LL);

        // A trainer from before TD-269 sends no step: the last one stays.
        ASSERT_EQ(push(c, "progress-model", run_id, 4, ""), 200);
        const auto record = get_record(c, "progress-model");
        EXPECT_EQ(record["progress"]["epoch"], 4);
        EXPECT_EQ(record["progress"]["global_step"], 5000000000LL);
    }
    {
        RunningMns mns(dir);  // loaded back from models.db
        ASSERT_TRUE(mns.ready());
        auto c = mns.client();
        EXPECT_EQ(get_record(c, "progress-model")["progress"]["global_step"], 5000000000LL);

        ASSERT_FALSE(start_training(c, "progress-model", true).empty());
        EXPECT_EQ(get_record(c, "progress-model")["progress"]["global_step"], 0) << "a new run starts at 0";
    }
    fs::remove_all(dir);
}

// A models.db from before TD-269 has no progress_global_step column; MNS adds it at the end, where
// a new database has it, so its positional INSERT keeps binding the right columns.
TEST(MnsProgressTest, ADatabaseFromBeforeTheColumnGainsIt) {
    const fs::path dir = fresh_dir("migrate");
    {
        sqlite3* db = nullptr;
        ASSERT_EQ(sqlite3_open((dir / "models.db").string().c_str(), &db), SQLITE_OK);
        const char* old_schema = R"(
CREATE TABLE models (
    model_id TEXT PRIMARY KEY, model_name TEXT UNIQUE NOT NULL, role TEXT DEFAULT '',
    state TEXT NOT NULL DEFAULT 'initializing', run_id TEXT DEFAULT '', created_utc TEXT NOT NULL,
    updated_utc TEXT NOT NULL, artifact_host TEXT DEFAULT '', artifact_path TEXT DEFAULT '',
    artifact_checksum TEXT DEFAULT '', artifact_format TEXT DEFAULT 'adai-native',
    d_model INTEGER DEFAULT 0, num_heads INTEGER DEFAULT 0, d_ff INTEGER DEFAULT 0,
    num_encoder_layers INTEGER DEFAULT 0, num_decoder_layers INTEGER DEFAULT 0,
    max_seq_length INTEGER DEFAULT 0, tags_json TEXT DEFAULT '{}',
    current_run_number INTEGER DEFAULT 0, run_started_utc TEXT DEFAULT '',
    progress_session_id TEXT DEFAULT '', progress_epoch INTEGER DEFAULT 0,
    progress_loss REAL DEFAULT 0.0, progress_best_loss REAL DEFAULT 0.0,
    progress_updated_utc TEXT DEFAULT '', run_group TEXT DEFAULT '', kind TEXT DEFAULT 'chatbot',
    connection_json TEXT DEFAULT '{}');
INSERT INTO models (model_id, model_name, role, created_utc, updated_utc, d_model, num_heads, d_ff,
                    num_encoder_layers, num_decoder_layers, max_seq_length, run_group)
VALUES ('id-old', 'old', 'chatbot', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 64, 2, 128, 1, 1,
        16, 'g1');
)";
        char* err = nullptr;
        ASSERT_EQ(sqlite3_exec(db, old_schema, nullptr, nullptr, &err), SQLITE_OK) << (err ? err : "");
        sqlite3_close(db);
    }
    {
        RunningMns mns(dir);
        ASSERT_TRUE(mns.ready());
        auto c = mns.client();
        const auto record = get_record(c, "old");
        ASSERT_FALSE(record.is_null());
        EXPECT_EQ(record["run_group"], "g1");
        EXPECT_EQ(record["progress"]["global_step"], 0);
        const std::string run_id = start_training(c, "old", false);
        ASSERT_FALSE(run_id.empty());
        ASSERT_EQ(push(c, "old", run_id, 1, ",\"global_step\":42"), 200);
    }
    {
        RunningMns mns(dir);
        ASSERT_TRUE(mns.ready());
        auto c = mns.client();
        const auto record = get_record(c, "old");
        EXPECT_EQ(record["progress"]["global_step"], 42);
        EXPECT_EQ(record["run_group"], "g1") << "the columns before the new one still line up";
        EXPECT_EQ(record["kind"], "chatbot");
        EXPECT_EQ(record["state"], "training");
    }
    fs::remove_all(dir);
}

// TD-269: each pass's history entry records the run's step. Sent through the real client, whose
// training_summary values are JSON strings: MNS used to read those as 0 epochs and loss 0.
TEST(MnsProgressTest, HistoryEntriesRecordTheStepEpochsAndLossFromTheClient) {
    const fs::path dir = fresh_dir("history");
    {
        RunningMns mns(dir);
        ASSERT_TRUE(mns.ready());
        auto c = mns.client();
        ASSERT_EQ(c.Post("/models", std::string("{\"model_name\":\"history-model\",") + kArch + "}",
                         "application/json")
                      ->status,
                  201);
        adai::ModelNameClient client("http://127.0.0.1:" + std::to_string(mns.port()), 5000);
        const std::string run_id = client.set_training("history-model", false);
        ASSERT_FALSE(run_id.empty());
        client.push_progress("history-model", run_id, "session-01", 2, 0.8, 0.7, 300);
        adai::ArtifactLocation artifact;
        artifact.path = "/tmp/checkpoint";
        client.set_candidate("history-model", run_id, artifact,
                             {{"epochs", "3"}, {"final_loss", "0.625"}, {"global_step", "450"}});

        // A run that never reaches candidate is archived with its last reported step.
        const std::string run2 = client.set_training("history-model", true);
        client.push_progress("history-model", run2, "session-01", 1, 0.9, 0.9, 75);
        ASSERT_FALSE(client.set_training("history-model", true).empty());
    }
    {
        RunningMns mns(dir);  // and both survive a restart
        ASSERT_TRUE(mns.ready());
        auto c = mns.client();
        const auto history = get_record(c, "history-model")["training_history"];
        ASSERT_EQ(history.size(), 2u) << history.dump();
        EXPECT_EQ(history[0]["epochs"], 3);
        EXPECT_DOUBLE_EQ(history[0]["final_loss"].get<double>(), 0.625);
        EXPECT_EQ(history[0]["global_step"], 450);
        EXPECT_FALSE(history[0]["incomplete"].get<bool>());
        EXPECT_EQ(history[1]["global_step"], 75);
        EXPECT_TRUE(history[1]["incomplete"].get<bool>());
    }
    fs::remove_all(dir);
}
