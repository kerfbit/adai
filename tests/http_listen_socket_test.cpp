// TD-551: HttpListenSocket — an httplib server's listening port is exclusive to one server.
#include <gtest/gtest.h>

#include <thread>

#include <httplib.h>

#include "../src/HttpListenSocket.hpp"

#ifndef _WIN32

// Why the helper exists: with httplib's default options (SO_REUSEPORT) a second server binds a
// port that is already listening, and the kernel would split connections between the two.
TEST(HttpListenSocketTest, DefaultHttplibOptionsLetASecondServerShareThePort) {
    httplib::Server first, second;
    const int port = first.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    EXPECT_TRUE(second.bind_to_port("127.0.0.1", port));
}

TEST(HttpListenSocketTest, ExclusivePortRefusesASecondServer) {
    httplib::Server first, second;
    adai::use_exclusive_listen_port(first);
    adai::use_exclusive_listen_port(second);
    const int port = first.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    EXPECT_FALSE(second.bind_to_port("127.0.0.1", port));
}

// Even a second server with the default options can't join an exclusive listener.
TEST(HttpListenSocketTest, DefaultServerCantJoinAnExclusiveListener) {
    httplib::Server first, second;
    adai::use_exclusive_listen_port(first);
    const int port = first.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    EXPECT_FALSE(second.bind_to_port("127.0.0.1", port));
}

// SO_REUSEADDR still allows an immediate restart on the same port once the first server is gone.
TEST(HttpListenSocketTest, PortIsReusableAfterTheServerStops) {
    int port = 0;
    {
        httplib::Server first;
        adai::use_exclusive_listen_port(first);
        port = first.bind_to_any_port("127.0.0.1");
        ASSERT_GT(port, 0);
        std::thread t([&] { first.listen_after_bind(); });
        first.wait_until_ready();
        httplib::Client cli("127.0.0.1", port);
        cli.Get("/");  // leaves a TIME_WAIT connection behind
        first.stop();
        t.join();
    }
    httplib::Server again;
    adai::use_exclusive_listen_port(again);
    EXPECT_TRUE(again.bind_to_port("127.0.0.1", port));
}

#endif  // _WIN32
