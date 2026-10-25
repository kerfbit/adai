// TD-526/TD-550: BrowserWriteGuard — refusing writes a web page could make from a browser.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include <httplib.h>

#include "../src/BrowserWriteGuard.hpp"

using adai::browser_write_refusal;

TEST(BrowserWriteRefusalTest, ReadsAreNeverRefused) {
    for (const char* method : {"GET", "HEAD", "OPTIONS"})
        EXPECT_FALSE(browser_write_refusal(method, "http://evil.example", "text/plain", {}))
            << method;
}

TEST(BrowserWriteRefusalTest, UnlistedOriginIsRefusedWith403) {
    for (const char* method : {"POST", "PUT", "PATCH", "DELETE"}) {
        const auto r = browser_write_refusal(method, "http://evil.example", "application/json", {});
        ASSERT_TRUE(r) << method;
        EXPECT_EQ(r->status, 403);
    }
}

TEST(BrowserWriteRefusalTest, ListedOriginWithJsonIsAllowed) {
    EXPECT_FALSE(browser_write_refusal("POST", "https://dash.example", "application/json",
                                       {"https://dash.example"}));
}

TEST(BrowserWriteRefusalTest, PreflightFreeContentTypesAreRefusedWith415) {
    for (const char* ct : {"text/plain", "Text/Plain; charset=UTF-8", "application/x-www-form-urlencoded",
                           " multipart/form-data; boundary=x"}) {
        const auto r = browser_write_refusal("POST", "", ct, {});
        ASSERT_TRUE(r) << ct;
        EXPECT_EQ(r->status, 415) << ct;
    }
}

// The non-browser clients: JSON, raw uploads, and bodiless POSTs with no Content-Type at all.
TEST(BrowserWriteRefusalTest, ClientsWithoutOriginAreAllowed) {
    for (const char* ct : {"application/json", "application/octet-stream", ""})
        EXPECT_FALSE(browser_write_refusal("POST", "", ct, {})) << ct;
}

TEST(BrowserWriteRefusalTest, JsonBodyEscapesTheMessage) {
    const std::string body =
        adai::browser_write_refusal_json({403, "from 'a\"b\\c'\n"});
    EXPECT_EQ(body, "{\"error\":\"from 'a\\\"b\\\\c'\"}");
}

// Installed on a real httplib::Server: a refused write never reaches its route.
TEST(BrowserWriteGuardServerTest, RefusedWritesNeverReachTheRoute) {
    httplib::Server svr;
    adai::install_browser_write_guard(svr, "test");
    std::atomic<int> writes{0};
    svr.Post("/write", [&](const httplib::Request&, httplib::Response& res) {
        ++writes;
        res.set_content("{}", "application/json");
    });
    svr.Get("/read", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{}", "application/json");
    });
    const int port = svr.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    std::thread t([&] { svr.listen_after_bind(); });
    svr.wait_until_ready();

    httplib::Client cli("127.0.0.1", port);
    auto res = cli.Post("/write", {{"Origin", "http://evil.example"}}, "{}", "text/plain");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 403);
    res = cli.Post("/write", "{}", "text/plain");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 415);
    EXPECT_EQ(writes.load(), 0);

    res = cli.Post("/write", "{}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(writes.load(), 1);
    res = cli.Get("/read", {{"Origin", "http://evil.example"}});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    svr.stop();
    t.join();
}
