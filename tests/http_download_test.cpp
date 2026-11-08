/**
 * TD-312: adai::http_download() against a local httplib::Server (http:// with allow_plain_http, the
 * only scheme a test server here can offer). Covers the body landing in place, the bearer header
 * staying on its own host, refusals before any connection, a stalled server, and no partial file
 * ever replacing the output.
 */
#include <gtest/gtest.h>
#include <httplib.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include "HttpDownload.hpp"

namespace fs = std::filesystem;

#ifdef ADAI_HAVE_LIBCURL
namespace {

class HttpDownloadTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("http_download_test_" + std::to_string(::getpid()) + "_" +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(dir_);
    }
    void TearDown() override {
        server_.stop();
        if (thread_.joinable())
            thread_.join();
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    void start() {
        port_ = server_.bind_to_any_port("127.0.0.1");
        ASSERT_GT(port_, 0);
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!server_.is_running() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        ASSERT_TRUE(server_.is_running());
    }
    std::string url(const std::string& path, const std::string& host = "127.0.0.1") const {
        return "http://" + host + ":" + std::to_string(port_) + path;
    }
    adai::HttpDownloadRequest request(const std::string& path) const {
        adai::HttpDownloadRequest r;
        r.url = url(path);
        r.output_path = (dir_ / "out.bin").string();
        r.allow_plain_http = true;
        r.connect_timeout_seconds = 5;
        r.max_time_seconds = 10;
        return r;
    }
    static std::string read(const fs::path& p) {
        std::ifstream f(p, std::ios::binary);
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    httplib::Server server_;
    std::thread thread_;
    int port_ = 0;
    fs::path dir_;
};

}  // namespace

TEST_F(HttpDownloadTest, WritesTheBodyAndSendsTheBearerToken) {
    std::string seen_auth;
    server_.Get("/file", [&](const httplib::Request& req, httplib::Response& res) {
        seen_auth = req.get_header_value("Authorization");
        res.set_content("hello parquet", "application/octet-stream");
    });
    start();
    auto r = request("/file");
    r.bearer_token = "hf_secret";
    const auto result = adai::http_download(r);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.http_status, 200);
    EXPECT_EQ(read(r.output_path), "hello parquet");
    EXPECT_EQ(seen_auth, "Bearer hf_secret");
    EXPECT_FALSE(fs::exists(r.output_path + ".part"));
}

TEST_F(HttpDownloadTest, AnErrorStatusLeavesTheExistingOutputAndNoPartFile) {
    server_.Get("/missing", [](const httplib::Request&, httplib::Response& res) {
        res.status = 404;
        res.set_content("not found page", "text/html");
    });
    start();
    auto r = request("/missing");
    std::ofstream(r.output_path) << "previous complete copy";
    const auto result = adai::http_download(r);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.http_status, 404);
    EXPECT_EQ(read(r.output_path), "previous complete copy");
    EXPECT_FALSE(fs::exists(r.output_path + ".part"));
}

TEST_F(HttpDownloadTest, RefusesPlainHttpAndOtherHostsBeforeConnecting) {
    std::atomic<int> hits{0};
    server_.Get("/file", [&](const httplib::Request&, httplib::Response& res) {
        ++hits;
        res.set_content("x", "text/plain");
    });
    start();
    auto r = request("/file");
    r.allow_plain_http = false;
    auto result = adai::http_download(r);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("https"), std::string::npos) << result.error;

    r = request("/file");
    r.allowed_hosts = {"huggingface.co", ".huggingface.co"};
    result = adai::http_download(r);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("127.0.0.1"), std::string::npos) << result.error;

    EXPECT_EQ(hits.load(), 0);
    EXPECT_FALSE(fs::exists(r.output_path));
}

TEST_F(HttpDownloadTest, AllowedHostsMatchExactNamesAndSubdomains) {
    server_.Get("/file", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("x", "text/plain");
    });
    start();
    auto r = request("/file");
    r.allowed_hosts = {"127.0.0.1"};
    EXPECT_TRUE(adai::http_download(r).ok);
    r.allowed_hosts = {".0.0.1"};  // a suffix entry matches a longer name
    EXPECT_TRUE(adai::http_download(r).ok);
    r.allowed_hosts = {"27.0.0.1"};  // an exact entry is not a suffix
    EXPECT_FALSE(adai::http_download(r).ok);
}

TEST_F(HttpDownloadTest, TheTokenIsNotSentToTheHostARedirectLeadsTo) {
    std::string auth_at_target = "unset";
    server_.Get("/redirect", [&](const httplib::Request&, httplib::Response& res) {
        res.set_redirect(url("/target", "localhost"));
    });
    server_.Get("/target", [&](const httplib::Request& req, httplib::Response& res) {
        auth_at_target = req.get_header_value("Authorization");
        res.set_content("cdn body", "application/octet-stream");
    });
    start();
    auto r = request("/redirect");
    r.bearer_token = "hf_secret";
    const auto result = adai::http_download(r);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(read(r.output_path), "cdn body");
    EXPECT_EQ(auth_at_target, "");
}

TEST_F(HttpDownloadTest, AStalledServerTimesOut) {
    server_.Get("/stall", [](const httplib::Request&, httplib::Response& res) {
        std::this_thread::sleep_for(std::chrono::seconds(4));
        res.set_content("too late", "text/plain");
    });
    start();
    auto r = request("/stall");
    r.max_time_seconds = 1;
    const auto t0 = std::chrono::steady_clock::now();
    const auto result = adai::http_download(r);
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_FALSE(result.ok);
    EXPECT_LT(elapsed, std::chrono::seconds(3));
    EXPECT_FALSE(fs::exists(r.output_path));
    EXPECT_FALSE(fs::exists(r.output_path + ".part"));
}

TEST_F(HttpDownloadTest, AnEmptyBodyIsAFailure) {
    server_.Get("/empty", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("", "text/plain");
    });
    start();
    auto r = request("/empty");
    EXPECT_FALSE(adai::http_download(r).ok);
    EXPECT_FALSE(fs::exists(r.output_path));
}
#endif

TEST(UrlEncodeComponentTest, KeepsUnreservedAndEncodesTheRest) {
    EXPECT_EQ(adai::url_encode_component("tatsu-lab/alpaca_v1.0~x"), "tatsu-lab%2Falpaca_v1.0~x");
    EXPECT_EQ(adai::url_encode_component("a b&c=$(id)"), "a%20b%26c%3D%24%28id%29");
    EXPECT_EQ(adai::url_encode_component("\xC3\xA9"), "%C3%A9");
}
