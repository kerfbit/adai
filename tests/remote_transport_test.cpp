/**
 * @file remote_transport_test.cpp
 * @brief TD-498: RemoteTransport against a stub registry_server: segments in assign/unassign
 * replies, a streamed upload, and no requests at all for an https:// URL or a bad group or kind.
 */
#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <httplib.h>

#include "../src/RegistryTransport.hpp"

namespace fs = std::filesystem;

namespace {

int free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    ::close(fd);
    return ntohs(addr.sin_port);
}

// A stub registry_server on a free port that counts requests; stopped and joined on destruction.
class StubRegistry {
   public:
    StubRegistry() : port_(free_port()) {
        server_.set_pre_routing_handler([this](const httplib::Request&, httplib::Response&) {
            ++requests_;
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }
    ~StubRegistry() {
        server_.stop();
        if (thread_.joinable())
            thread_.join();
    }
    httplib::Server& server() { return server_; }
    void start() {
        thread_ = std::thread([this] { server_.listen("127.0.0.1", port_); });
        server_.wait_until_ready();
    }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
    int port() const { return port_; }
    int requests() const { return requests_.load(); }

   private:
    int port_;
    httplib::Server server_;
    std::thread thread_;
    std::atomic<int> requests_{0};
};

const char* kSegmentsReply =
    R"("segments":[{"path":"/data/big.jsonl","segment_start":0,"segment_count":100},)"
    R"({"path":"/data/big.jsonl","segment_start":100,"segment_count":50}])";

}  // namespace

TEST(RemoteTransportTest, AssignAndUnassignReadTheSegmentsTheyTouched) {
    StubRegistry stub;
    stub.server().Post("/registry/g1/assign", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string(R"({"assigned":3,"paths":["/data/a.jsonl"],)") + kSegmentsReply + "}",
                        "application/json");
    });
    stub.server().Post("/registry/g1/unassign", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string(R"({"unassigned":2,"skipped":0,"paths":[],)") + kSegmentsReply + "}",
                        "application/json");
    });
    stub.start();
    RemoteTransport t(stub.url(), "g1", 5000);

    const AssignResult assigned = t.assign("m1", {}, 0, {});
    EXPECT_EQ(assigned.assigned, 3);
    EXPECT_EQ(assigned.paths, std::vector<std::string>{"/data/a.jsonl"});
    ASSERT_EQ(assigned.segments.size(), 2u);
    EXPECT_EQ(assigned.segments[1].path, "/data/big.jsonl");
    EXPECT_EQ(assigned.segments[1].segment_start, 100);
    EXPECT_EQ(assigned.segments[1].segment_count, 50);

    const UnassignResult unassigned = t.unassign("m1", {}, false, {});
    EXPECT_EQ(unassigned.unassigned, 2);
    ASSERT_EQ(unassigned.segments.size(), 2u);
    EXPECT_EQ(unassigned.segments[0].segment_count, 100);
}

TEST(RemoteTransportTest, UploadSendsTheFileUnchanged) {
    const fs::path dir = fs::temp_directory_path() / ("adai_remote_upload_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const fs::path file = dir / "data.jsonl";
    std::string bytes(3 * 1024 * 1024 + 123, '\0');  // more than one 1 MiB piece
    std::mt19937 rng(7);
    for (auto& c : bytes)
        c = static_cast<char>(rng() & 0xff);
    std::ofstream(file, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    StubRegistry stub;
    std::mutex m;
    std::string received, filename;
    stub.server().Post("/registry/g1/upload", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(m);
        received = req.body;
        filename = req.get_param_value("filename");
        res.set_content(R"({"added":true,"path":"/srv/uploads/data.jsonl"})", "application/json");
    });
    stub.start();
    RemoteTransport t(stub.url(), "g1", 5000);

    const UploadResult result = t.upload_file(file.string(), false);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(result.path, "/srv/uploads/data.jsonl");
    std::lock_guard<std::mutex> lock(m);
    EXPECT_EQ(filename, "data.jsonl");
    EXPECT_EQ(received.size(), bytes.size());
    EXPECT_TRUE(received == bytes) << "the streamed body is the file, byte for byte";
    fs::remove_all(dir);
}

// It used to strip "https://" and send plain HTTP to port 8082; the run group and kind go into the
// URL unencoded. Each case now logs one error and makes no request.
TEST(RemoteTransportTest, HttpsOrABadGroupOrKindMakesNoRequests) {
    StubRegistry stub;
    stub.server().Get(R"(/registry/.*)", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"entries":[]})", "application/json");
    });
    stub.start();

    std::vector<PendingEntry> pending;
    {
        RemoteTransport ok(stub.url(), "g1", 2000);
        EXPECT_TRUE(ok.load_pending(pending));
    }
    const int baseline = stub.requests();
    ASSERT_GE(baseline, 1) << "the stub answers a valid client";

    const std::string https = "https://127.0.0.1:" + std::to_string(stub.port());
    for (const auto& [url, group, kind] : std::vector<std::tuple<std::string, std::string, std::string>>{
             {https, "g1", ""},
             {stub.url(), "../etc", ""},
             {stub.url(), "g1/x", ""},
             {stub.url(), "", ""},
             {stub.url(), "g1", "weights"},
         }) {
        RemoteTransport t(url, group, 2000, kind);
        EXPECT_FALSE(t.load_pending(pending)) << url << " " << group << " " << kind;
    }
    EXPECT_EQ(stub.requests(), baseline) << "no request reached the server";
}

TEST(RegistryNamesTest, GroupAndKindRules) {
    EXPECT_TRUE(valid_registry_group_name("bb01"));
    EXPECT_TRUE(valid_registry_group_name("a.b-c_d"));
    EXPECT_FALSE(valid_registry_group_name(""));
    EXPECT_FALSE(valid_registry_group_name(".hidden"));
    EXPECT_FALSE(valid_registry_group_name(".."));
    EXPECT_FALSE(valid_registry_group_name("a/b"));
    EXPECT_FALSE(valid_registry_group_name("a b"));
    EXPECT_FALSE(valid_registry_group_name(std::string(129, 'a')));
    EXPECT_TRUE(valid_dataset_kind(""));
    EXPECT_TRUE(valid_dataset_kind("world_model"));
    EXPECT_FALSE(valid_dataset_kind("weights"));
}
