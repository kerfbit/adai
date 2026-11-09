/**
 * @file portable_socket_test.cpp
 * @brief TD-476: PortableSocket's send_all(), EINTR retry and bounded accept wait, over real socket
 * pairs. Signals go only to a thread of this process, by its own id.
 */
#include <gtest/gtest.h>

#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../src/PortableSocket.hpp"

namespace {

using Clock = std::chrono::steady_clock;

void no_op_handler(int) {}

// Installs a handler without SA_RESTART, so a blocked call sees the signal.
class SignalScope {
   public:
    SignalScope() {
        struct sigaction action {};
        action.sa_handler = no_op_handler;
        sigemptyset(&action.sa_mask);
        sigaction(SIGUSR1, &action, &saved_);
    }
    ~SignalScope() {
        sigaction(SIGUSR1, &saved_, nullptr);
    }

   private:
    struct sigaction saved_ {};
};

class SocketPair {
   public:
    SocketPair() {
        EXPECT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds_), 0);
    }
    ~SocketPair() {
        close_b();
        if (fds_[0] >= 0)
            adai::close_socket(fds_[0]);
    }
    int a() const {
        return fds_[0];
    }
    int b() const {
        return fds_[1];
    }
    void close_b() {
        if (fds_[1] >= 0)
            adai::close_socket(fds_[1]);
        fds_[1] = -1;
    }

   private:
    int fds_[2] = {-1, -1};
};

std::size_t read_all(int fd, std::size_t want) {
    std::vector<char> buf(65536);
    std::size_t got = 0;
    while (got < want) {
        const long n = adai::recv_bytes(fd, buf.data(), buf.size());
        if (n <= 0)
            break;
        got += static_cast<std::size_t>(n);
    }
    return got;
}

}  // namespace

// A signal during a blocking send() returns a short count; send_all() carries on.
TEST(PortableSocketTest, SendAllSurvivesASignalMidSend) {
    SignalScope signals;
    SocketPair pair;
    ASSERT_TRUE(adai::set_recv_timeout(pair.b(), 5));  // a short send fails instead of hanging
    const std::vector<char> data(8 * 1024 * 1024, 'x');
    bool ok = false;
    std::thread sender([&] { ok = adai::send_all(pair.a(), data.data(), data.size()); });

    // Nothing reads yet, so the sender is blocked and still running.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ASSERT_EQ(pthread_kill(sender.native_handle(), SIGUSR1), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_EQ(read_all(pair.b(), data.size()), data.size());
    sender.join();
    EXPECT_TRUE(ok);
}

TEST(PortableSocketTest, SendAllFailsWhenThePeerStopsReading) {
    SocketPair pair;
    ASSERT_TRUE(adai::set_send_timeout(pair.a(), 1));
    const std::vector<char> data(8 * 1024 * 1024, 'x');
    const auto start = Clock::now();
    EXPECT_FALSE(adai::send_all(pair.a(), data.data(), data.size()));
    EXPECT_LT(Clock::now() - start, std::chrono::seconds(10));
}

TEST(PortableSocketTest, SendAllFailsWhenThePeerClosed) {
    SocketPair pair;
    pair.close_b();
    const std::string data(1024, 'x');
    EXPECT_FALSE(adai::send_all(pair.a(), data.data(), data.size()));
}

TEST(PortableSocketTest, RecvRetriesWhenInterrupted) {
    SignalScope signals;
    SocketPair pair;
    char c = 0;
    long n = 0;
    std::thread receiver([&] { n = adai::recv_bytes(pair.a(), &c, 1); });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_EQ(pthread_kill(receiver.native_handle(), SIGUSR1), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_EQ(adai::send_bytes(pair.b(), "k", 1), 1);
    receiver.join();
    EXPECT_EQ(n, 1);
    EXPECT_EQ(c, 'k');
}

TEST(PortableSocketTest, RecvStillTimesOut) {
    SocketPair pair;
    ASSERT_TRUE(adai::set_recv_timeout(pair.a(), 1));
    char c = 0;
    EXPECT_LT(adai::recv_bytes(pair.a(), &c, 1), 0);
}

TEST(PortableSocketTest, WaitReadableIsBoundedAndSeesData) {
    SocketPair pair;
    const auto start = Clock::now();
    EXPECT_FALSE(adai::wait_readable(pair.a(), 100));
    const auto waited = Clock::now() - start;
    EXPECT_GE(waited, std::chrono::milliseconds(90));
    EXPECT_LT(waited, std::chrono::seconds(5));

    ASSERT_EQ(adai::send_bytes(pair.b(), "k", 1), 1);
    EXPECT_TRUE(adai::wait_readable(pair.a(), 1000));
}

TEST(PortableSocketTest, SetupHelpersReportFailure) {
    const adai::socket_t bad = adai::kInvalidSocket;
    EXPECT_FALSE(adai::set_reuse_addr(bad));
    EXPECT_FALSE(adai::set_recv_timeout(bad, 1));
    EXPECT_FALSE(adai::set_send_timeout(bad, 1));
    SocketPair pair;
    EXPECT_TRUE(adai::set_recv_timeout(pair.a(), 1));
}

TEST(PortableSocketTest, WinsockGuardIsOkOffWindows) {
    adai::WinsockGuard guard;
    EXPECT_TRUE(guard.ok());
}
