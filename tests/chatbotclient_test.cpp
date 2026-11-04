// TD-260: ChatbotClient is the HTTP session with chatbot_api_server shared by the `chatbot` CLI and
// chatbot_gui. ChatbotCLI's tests (chatbotcliImprovedTests) cover it through the CLI; these test it
// directly against a stub server, including what only the GUI uses (cancel(), an untrained server).
#include "ChatbotClient.hpp"
#include <gtest/gtest.h>
#include <httplib.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "nlohmann/json.hpp"

namespace {

class StubServer {
   public:
    StubServer() {
        server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            std::lock_guard<std::mutex> lock(mutex_);
            res.status = health_status_;
            res.set_content(nlohmann::json{{"status", health_status_ == 200 ? "ok" : "untrained"},
                                           {"model_loaded", health_status_ == 200}}
                                .dump(),
                            "application/json");
        });
        server_.Post("/chat/session", [this](const httplib::Request& req, httplib::Response& res) {
            record(req);
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (hold_chat_) {
                    chat_waiting_ = true;
                    changed_.notify_all();
                    changed_.wait_for(lock, std::chrono::seconds(10), [this] { return !hold_chat_; });
                }
            }
            const auto body = nlohmann::json::parse(req.body);
            const std::string sid = body.value("session_id", "");
            std::lock_guard<std::mutex> lock(mutex_);
            if (!sid.empty() && sid == lost_session_) {
                res.status = 404;
                res.set_content(R"({"success":false,"error":"Unknown session_id"})", "application/json");
                return;
            }
            const std::string reply_sid = sid.empty() ? "s" + std::to_string(++sessions_) : sid;
            res.set_content(nlohmann::json{{"success", true},
                                           {"response", "echo: " + body.value("message", "")},
                                           {"session_id", reply_sid}}
                                .dump(),
                            "application/json");
        });
        server_.Post("/clear-session", [this](const httplib::Request& req, httplib::Response& res) {
            record(req);
            std::lock_guard<std::mutex> lock(mutex_);
            const std::string sid = nlohmann::json::parse(req.body).value("session_id", "");
            res.status = sid == lost_session_ ? 404 : 200;
            res.set_content(R"({"success":true})", "application/json");
        });
        server_.Post("/chat/session/export", [this](const httplib::Request& req, httplib::Response& res) {
            record(req);
            std::lock_guard<std::mutex> lock(mutex_);
            const std::string sid = nlohmann::json::parse(req.body).value("session_id", "");
            if (sid == lost_session_) {
                res.status = 404;
                return;
            }
            res.set_content(nlohmann::json{{"success", true}, {"data", "conversation of " + sid}}.dump(),
                            "application/json");
        });
        server_.Post("/chat/session/import", [this](const httplib::Request& req, httplib::Response& res) {
            record(req);
            std::lock_guard<std::mutex> lock(mutex_);
            const std::string sid = nlohmann::json::parse(req.body).value("session_id", "");
            if (!sid.empty() && sid == lost_session_) {
                res.status = 404;
                return;
            }
            res.set_content(nlohmann::json{{"success", true},
                                           {"session_id", sid.empty() ? "imported" : sid},
                                           {"message_count", 4}}
                                .dump(),
                            "application/json");
        });
        port_ = server_.bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        server_.wait_until_ready();
    }
    ~StubServer() {
        release_chat();
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }

    void set_health_status(int status) {
        std::lock_guard<std::mutex> lock(mutex_);
        health_status_ = status;
    }
    /// The server answers 404 for this session id, as for one it has expired.
    void lose_session(const std::string& sid) {
        std::lock_guard<std::mutex> lock(mutex_);
        lost_session_ = sid;
    }
    /// /chat/session waits until release_chat() (or 10 s).
    void hold_chat() {
        std::lock_guard<std::mutex> lock(mutex_);
        hold_chat_ = true;
    }
    bool wait_for_held_chat() {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, std::chrono::seconds(5), [this] { return chat_waiting_; });
    }
    void release_chat() {
        std::lock_guard<std::mutex> lock(mutex_);
        hold_chat_ = false;
        changed_.notify_all();
    }
    /// Every request received, as "<path> <body>", oldest first.
    std::vector<std::string> requests() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_;
    }

   private:
    void record(const httplib::Request& req) {
        std::lock_guard<std::mutex> lock(mutex_);
        requests_.push_back(req.path + " " + req.body);
    }

    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::string> requests_;
    int health_status_ = 200;
    std::string lost_session_;
    bool hold_chat_ = false;
    bool chat_waiting_ = false;
    int sessions_ = 0;
    httplib::Server server_;
    std::thread thread_;
    int port_ = 0;
};

}  // namespace

TEST(ChatbotClientTest, ConnectReportsAHealthyServer) {
    StubServer stub;
    ChatbotClient client(stub.url());
    EXPECT_FALSE(client.is_connected());
    const ChatbotClient::Health health = client.connect();
    EXPECT_TRUE(client.is_connected());
    EXPECT_TRUE(health.reachable);
    EXPECT_EQ(health.status, 200);
    EXPECT_TRUE(health.model_loaded);
}

TEST(ChatbotClientTest, ConnectReportsAServerRunningAnUntrainedModel) {
    StubServer stub;
    stub.set_health_status(503);  // ALLOW_UNTRAINED_MODEL=true (TD-246)
    ChatbotClient client(stub.url());
    const ChatbotClient::Health health = client.connect();
    EXPECT_TRUE(health.reachable);
    EXPECT_EQ(health.status, 503);
    EXPECT_FALSE(health.model_loaded);
}

TEST(ChatbotClientTest, ConnectReportsWhyAServerCantBeReached) {
    std::string url;
    {
        StubServer gone;  // a port that was just in use and now refuses connections
        url = gone.url();
    }
    ChatbotClient client(url);
    const ChatbotClient::Health health = client.connect();
    EXPECT_FALSE(health.reachable);
    EXPECT_FALSE(health.error.empty());
}

TEST(ChatbotClientTest, SendBeforeConnectIsAnError) {
    ChatbotClient client("http://127.0.0.1:1");
    const ChatbotClient::Reply reply = client.send_message("hi");
    EXPECT_FALSE(reply.ok);
    EXPECT_EQ(reply.text, "Error: Client not initialized");
}

TEST(ChatbotClientTest, AdoptsTheServersSessionAndSendsOnlyChangedSettings) {
    StubServer stub;
    ChatbotClient client(stub.url());
    ASSERT_TRUE(client.connect().reachable);

    ChatbotClient::Reply reply = client.send_message("one");
    EXPECT_TRUE(reply.ok);
    EXPECT_EQ(reply.text, "echo: one");
    EXPECT_EQ(client.session_id(), "s1");

    client.settings().set_temperature(0.7f);
    client.settings().set_strategy("beam");
    reply = client.send_message("two");
    EXPECT_TRUE(reply.ok);
    const auto requests = stub.requests();
    ASSERT_EQ(requests.size(), 2u);
    EXPECT_EQ(requests[0], R"(/chat/session {"message":"one"})");
    EXPECT_EQ(requests[1],
              R"(/chat/session {"session_id":"s1","message":"two","temperature":0.7,"strategy":"beam"})");
}

TEST(ChatbotClientTest, ALostSessionIsReportedAndTheMessageGoesToANewOne) {
    StubServer stub;
    ChatbotClient client(stub.url());
    ASSERT_TRUE(client.connect().reachable);
    ASSERT_TRUE(client.send_message("one").ok);
    stub.lose_session("s1");

    const ChatbotClient::Reply reply = client.send_message("two");
    EXPECT_TRUE(reply.ok);
    EXPECT_TRUE(reply.session_lost);
    EXPECT_EQ(reply.text, "echo: two");
    EXPECT_EQ(client.session_id(), "s2");
    const auto requests = stub.requests();
    ASSERT_EQ(requests.size(), 3u);
    EXPECT_EQ(requests[2], R"(/chat/session {"message":"two"})");
}

TEST(ChatbotClientTest, ClearReportsEachOutcome) {
    StubServer stub;
    ChatbotClient client(stub.url());
    ASSERT_TRUE(client.connect().reachable);
    EXPECT_EQ(client.clear_session(), ChatbotClient::ClearResult::NoSession);

    ASSERT_TRUE(client.send_message("one").ok);
    EXPECT_EQ(client.clear_session(), ChatbotClient::ClearResult::Cleared);
    EXPECT_EQ(client.session_id(), "s1");

    stub.lose_session("s1");
    EXPECT_EQ(client.clear_session(), ChatbotClient::ClearResult::SessionLost);
    EXPECT_EQ(client.session_id(), "");
}

TEST(ChatbotClientTest, ExportReportsEachOutcome) {
    StubServer stub;
    ChatbotClient client(stub.url());
    ASSERT_TRUE(client.connect().reachable);
    EXPECT_EQ(client.export_session().status, ChatbotClient::Export::Status::NoSession);

    ASSERT_TRUE(client.send_message("one").ok);
    const ChatbotClient::Export exported = client.export_session();
    EXPECT_EQ(exported.status, ChatbotClient::Export::Status::Ok);
    EXPECT_EQ(exported.data, "conversation of s1");

    stub.lose_session("s1");
    EXPECT_EQ(client.export_session().status, ChatbotClient::Export::Status::SessionLost);
    EXPECT_EQ(client.session_id(), "");
}

TEST(ChatbotClientTest, ImportGoesToANewSessionWhenTheServerLostThisOne) {
    StubServer stub;
    ChatbotClient client(stub.url());
    ASSERT_TRUE(client.connect().reachable);
    ASSERT_TRUE(client.send_message("one").ok);
    stub.lose_session("s1");

    const ChatbotClient::Import imported = client.import_session("saved conversation");
    EXPECT_TRUE(imported.ok) << imported.error;
    EXPECT_EQ(imported.message_count, "4");
    EXPECT_EQ(client.session_id(), "imported");
    const auto requests = stub.requests();
    ASSERT_GE(requests.size(), 2u);
    EXPECT_EQ(requests.back(), R"(/chat/session/import {"data":"saved conversation"})");
}

// chatbot_gui cancels a request in flight when its window closes, instead of waiting for the reply
// (up to the 300 s read timeout).
TEST(ChatbotClientTest, CancelEndsARequestInFlight) {
    StubServer stub;
    ChatbotClient client(stub.url());
    ASSERT_TRUE(client.connect().reachable);
    stub.hold_chat();

    ChatbotClient::Reply reply;
    std::atomic<bool> finished{false};
    std::thread sender([&] {
        reply = client.send_message("slow");
        finished = true;
    });
    ASSERT_TRUE(stub.wait_for_held_chat());
    const auto start = std::chrono::steady_clock::now();
    client.cancel();
    sender.join();
    const auto waited = std::chrono::steady_clock::now() - start;
    stub.release_chat();

    EXPECT_TRUE(finished);
    EXPECT_FALSE(reply.ok);
    EXPECT_EQ(reply.text, "Error: Connection failed");
    EXPECT_LT(waited, std::chrono::seconds(3));
}

TEST(ChatbotClientTest, WriteFileAtomicallyReplacesTheFileWhole) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "adai_chatbotclient_write";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string path = (dir / "conversation.txt").string();
    {
        std::ofstream(path) << "an older, longer conversation";
    }
    EXPECT_EQ(chatbot_client::write_file_atomically(path, "new"), "");
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    EXPECT_EQ(text.str(), "new");
    EXPECT_EQ(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}), 1);  // no temp left

    EXPECT_NE(chatbot_client::write_file_atomically((dir / "missing" / "x.txt").string(), "data"), "");
    fs::remove_all(dir);
}
