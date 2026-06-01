#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include "MetricsPushClient.hpp"

#ifdef BUILD_METRICS_API_SERVER
#include <httplib.h>
#endif

// Port range 42200–42299 — reserved for MetricsPushClientTests
// (incremental_trainer_registry_test uses 41800–41899)
static constexpr int kPort_SampleDrop    = 42200;
static constexpr int kPort_EpochEviction = 42201;
static constexpr int kPort_EpochNoDrop   = 42202;
static constexpr int kPort_5xxRetry      = 42203;
static constexpr int kPort_4xxNoRetry    = 42204;
static constexpr int kPort_409NoRetry    = 42205;
static constexpr int kPort_DrainOnEnd    = 42206;
static constexpr int kPort_StartSession  = 42207;

#ifdef BUILD_METRICS_API_SERVER

// ============================================================================
// Helpers
// ============================================================================

/**
 * A gate that blocks httplib server handlers until the test releases it.
 *
 * Usage:
 *   Gate g;
 *   svr.Post("/.*", [&g](auto&, auto& res) { g.wait(); res.status = 201; });
 *   // ... enqueue events, push thread blocks in handler ...
 *   // wait until gate.entered(), then fill queue, then:
 *   g.release();
 */
struct Gate {
    std::mutex mtx;
    std::condition_variable cv;
    bool open{false};
    std::atomic<int> entry_count{0};

    /** Called from server handler thread — blocks until release(). */
    void wait() {
        entry_count.fetch_add(1, std::memory_order_acq_rel);
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this] { return open; });
    }

    /** Called from test thread — unblocks all waiting handlers. */
    void release() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            open = true;
        }
        cv.notify_all();
    }

    /** True once at least one handler has entered wait(). */
    bool entered() const { return entry_count.load(std::memory_order_acquire) > 0; }
};

/** Starts svr.listen() in a background thread; blocks until is_running(). */
static std::thread start_server(httplib::Server& svr, int port) {
    std::thread t([&svr, port]() { svr.listen("127.0.0.1", port); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!svr.is_running()) {
        EXPECT_LT(std::chrono::steady_clock::now(), deadline) << "Server failed to start";
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return t;
}

/** Builds the session base URL used by all tests. */
static std::string session_url(int port) {
    return "http://127.0.0.1:" + std::to_string(port) + "/api/sessions/1-test";
}

/**
 * Spin-waits until gate.entered() is true or the deadline is exceeded.
 * Returns false on timeout (will trigger ASSERT_TRUE from caller).
 */
static bool wait_for_gate(const Gate& gate, std::chrono::milliseconds timeout_ms =
                                                std::chrono::milliseconds(2000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout_ms;
    while (!gate.entered()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

/**
 * Spin-waits until count.load() >= target or the deadline is exceeded.
 * Used to ensure the push thread has processed all expected events BEFORE
 * calling end_session() — otherwise end_session()'s /end event (Session
 * priority) may evict a still-queued sample via the overflow policy.
 */
static bool wait_for_count(const std::atomic<int>& count, int target,
                           std::chrono::milliseconds timeout_ms =
                               std::chrono::milliseconds(2000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout_ms;
    while (count.load(std::memory_order_relaxed) < target) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// ============================================================================
// Queue overflow tests
// ============================================================================

// The push thread blocks in the server handler while the test fills the queue.
// Overflow behaviour is exercised by inspecting the server's POST count after
// the gate is released and end_session() drains everything.

TEST(MetricsPushClientQueuePolicy, SampleEventDroppedWhenQueueFull) {
    // max_queue_depth = 3.  push thread picks up start_epoch and blocks.
    // Three sample enqueues fill the queue; the 4th is silently dropped.
    // Server should see 5 POSTs: epoch/start + 3 samples + /end.
    Gate gate;
    httplib::Server svr;
    std::atomic<int> post_count{0};

    svr.Post("/.*", [&](const httplib::Request&, httplib::Response& res) {
        post_count.fetch_add(1, std::memory_order_relaxed);
        gate.wait();
        res.status = 201;
    });

    auto svr_thread = start_server(svr, kPort_SampleDrop);
    MetricsPushClient client(session_url(kPort_SampleDrop), /*timeout_ms=*/500,
                             /*max_queue_depth=*/3);

    client.start_epoch(1, 100);  // push thread pops this and enters gate.wait()
    ASSERT_TRUE(wait_for_gate(gate)) << "Push thread never entered server handler";

    // Queue is now empty.  Fill to capacity.
    client.update_sample_metrics(1, 0.5f, 1.0f, 0.001f);
    client.update_sample_metrics(2, 0.5f, 1.0f, 0.001f);
    client.update_sample_metrics(3, 0.5f, 1.0f, 0.001f);

    // This 4th sample must be DROPPED (queue.size() >= max_queue_depth).
    client.update_sample_metrics(4, 0.5f, 1.0f, 0.001f);

    gate.release();

    // Wait until epoch/start + 3 samples have been delivered before calling
    // end_session().  Without this, end_session()'s /end event (Session
    // priority) races with the push thread and may evict a still-queued
    // sample via the overflow policy, giving count=4 instead of 5.
    ASSERT_TRUE(wait_for_count(post_count, 4)) << "Timed out waiting for 4 POSTs";
    client.end_session();  // enqueues /end into now-empty queue, joins thread

    svr.stop();
    svr_thread.join();

    EXPECT_EQ(post_count.load(), 5);  // dropped sample not delivered
}

TEST(MetricsPushClientQueuePolicy, EpochEventEvictsSampleWhenQueueFull) {
    // With max_queue_depth = 3 and the queue filled with samples, an Epoch
    // event must evict the oldest Sample instead of being dropped itself.
    Gate gate;
    httplib::Server svr;
    std::atomic<int> post_count{0};
    std::atomic<int> epoch_end_count{0};

    svr.Post("/.*", [&](const httplib::Request& req, httplib::Response& res) {
        post_count.fetch_add(1, std::memory_order_relaxed);
        if (req.path.find("/epoch/end") != std::string::npos) {
            epoch_end_count.fetch_add(1, std::memory_order_relaxed);
        }
        gate.wait();
        res.status = 201;
    });

    auto svr_thread = start_server(svr, kPort_EpochEviction);
    MetricsPushClient client(session_url(kPort_EpochEviction), 500, /*max_queue_depth=*/3);

    client.start_epoch(1, 100);
    ASSERT_TRUE(wait_for_gate(gate)) << "Push thread never entered server handler";

    // Fill queue to capacity with Sample events.
    client.update_sample_metrics(1, 0.5f, 1.0f, 0.001f);
    client.update_sample_metrics(2, 0.5f, 1.0f, 0.001f);
    client.update_sample_metrics(3, 0.5f, 1.0f, 0.001f);

    // Epoch event: evicts the oldest Sample (sample-1), adds itself.
    // Queue after: [sample-2, sample-3, epoch/end]
    client.end_epoch(1, 0.5f, 0.6f, 0.001f);

    gate.release();

    // Wait for epoch/start + sample-2 + sample-3 + epoch/end to be delivered
    // before calling end_session(), to avoid a race where /end evicts one of
    // the remaining samples while they are still in the queue.
    ASSERT_TRUE(wait_for_count(post_count, 4)) << "Timed out waiting for 4 POSTs";
    client.end_session();

    svr.stop();
    svr_thread.join();

    // epoch/start(1) + sample-2(1) + sample-3(1) + epoch/end(1) + /end(1) = 5
    EXPECT_EQ(post_count.load(), 5);
    EXPECT_EQ(epoch_end_count.load(), 1);  // epoch/end was delivered, not dropped
}

TEST(MetricsPushClientQueuePolicy, EpochNotDroppedEvenWhenQueueHasOnlyEpochEvents) {
    // When the queue is full of Epoch events (no Samples to evict), a further
    // Epoch event grows the queue beyond max_queue_depth — it is NEVER dropped.
    Gate gate;
    httplib::Server svr;
    std::atomic<int> epoch_end_count{0};

    svr.Post("/.*", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.path.find("/epoch/end") != std::string::npos) {
            epoch_end_count.fetch_add(1, std::memory_order_relaxed);
        }
        gate.wait();
        res.status = 201;
    });

    auto svr_thread = start_server(svr, kPort_EpochNoDrop);
    // max_queue_depth = 2 so we can fill it with 2 Epoch events quickly.
    MetricsPushClient client(session_url(kPort_EpochNoDrop), 500, /*max_queue_depth=*/2);

    client.start_epoch(1, 100);
    ASSERT_TRUE(wait_for_gate(gate)) << "Push thread never entered server handler";

    // Fill queue to capacity with 2 Epoch events (no Samples present).
    client.end_epoch(1, 0.5f, 0.6f, 0.001f);
    client.end_epoch(2, 0.4f, 0.5f, 0.001f);

    // Third Epoch: no Sample to evict → queue grows to 3 (beyond limit).
    // This event must NOT be dropped.
    client.end_epoch(3, 0.3f, 0.4f, 0.001f);

    gate.release();
    client.end_session();

    svr.stop();
    svr_thread.join();

    EXPECT_EQ(epoch_end_count.load(), 3);  // all 3 end_epoch events delivered
}

// ============================================================================
// Retry and HTTP status code tests
// ============================================================================

TEST(MetricsPushClientRetryPolicy, RetriesOn5xxAndEventuallySucceeds) {
    // First POST returns 500, second returns 201.
    // start_session() must retry and return 201.
    // Only count calls to the /start endpoint — the push thread will also
    // fire a POST for /end which must not inflate the counter.
    httplib::Server svr;
    std::atomic<int> start_call_count{0};

    svr.Post("/.*", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.path.find("/start") != std::string::npos) {
            const int n = start_call_count.fetch_add(1, std::memory_order_relaxed) + 1;
            res.status = (n == 1) ? 500 : 201;
        } else {
            res.status = 201;
        }
    });

    auto svr_thread = start_server(svr, kPort_5xxRetry);
    MetricsPushClient client(session_url(kPort_5xxRetry), /*timeout_ms=*/500);

    const int rc = client.start_session(1, 3, 100);

    client.end_session();
    svr.stop();
    svr_thread.join();

    EXPECT_EQ(rc, 201);
    EXPECT_EQ(start_call_count.load(), 2);  // 1 failed (500) + 1 successful (201)
}

TEST(MetricsPushClientRetryPolicy, DoesNotRetryOn4xx) {
    // A 400 response must be returned immediately without retry.
    // Count only /start requests to avoid counting the /end push.
    httplib::Server svr;
    std::atomic<int> start_call_count{0};

    svr.Post("/.*", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.path.find("/start") != std::string::npos) {
            start_call_count.fetch_add(1, std::memory_order_relaxed);
            res.status = 400;
        } else {
            res.status = 201;
        }
    });

    auto svr_thread = start_server(svr, kPort_4xxNoRetry);
    MetricsPushClient client(session_url(kPort_4xxNoRetry), 500);

    const int rc = client.start_session(1, 3, 100);

    client.end_session();
    svr.stop();
    svr_thread.join();

    EXPECT_EQ(rc, 400);
    EXPECT_EQ(start_call_count.load(), 1);  // no retry on 4xx
}

TEST(MetricsPushClientRetryPolicy, DoesNotRetryOn409) {
    // A 409 Conflict must be returned to the caller without retry so that
    // IncrementalTrainer's suffix-retry loop can handle it at a higher level.
    // Count only /start requests to avoid counting the /end push.
    httplib::Server svr;
    std::atomic<int> start_call_count{0};

    svr.Post("/.*", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.path.find("/start") != std::string::npos) {
            start_call_count.fetch_add(1, std::memory_order_relaxed);
            res.status = 409;
        } else {
            res.status = 201;
        }
    });

    auto svr_thread = start_server(svr, kPort_409NoRetry);
    MetricsPushClient client(session_url(kPort_409NoRetry), 500);

    const int rc = client.start_session(1, 3, 100);

    client.end_session();
    svr.stop();
    svr_thread.join();

    EXPECT_EQ(rc, 409);
    EXPECT_EQ(start_call_count.load(), 1);  // no retry on 409
}

// ============================================================================
// Shutdown drain tests
// ============================================================================

TEST(MetricsPushClientShutdownDrain, EndSessionDrainsAllQueuedEventsBeforeJoin) {
    // end_session() must not return until every queued event has been delivered.
    httplib::Server svr;
    std::atomic<int> post_count{0};

    svr.Post("/.*", [&](const httplib::Request&, httplib::Response& res) {
        post_count.fetch_add(1, std::memory_order_relaxed);
        res.status = 201;
    });

    auto svr_thread = start_server(svr, kPort_DrainOnEnd);
    MetricsPushClient client(session_url(kPort_DrainOnEnd), 500);

    // Enqueue 5 events.
    client.start_epoch(1, 200);
    client.update_sample_metrics(1, 0.8f, 1.2f, 0.001f);
    client.update_sample_metrics(2, 0.7f, 1.1f, 0.001f);
    client.update_sample_metrics(3, 0.6f, 1.0f, 0.001f);
    client.end_epoch(1, 0.6f, 0.7f, 0.001f);

    // end_session() enqueues /end, sets stop_, and blocks until thread exits.
    client.end_session();

    svr.stop();
    svr_thread.join();

    // 5 queued events + the /end event from end_session() = 6 server hits.
    EXPECT_EQ(post_count.load(), 6);
}

// ============================================================================
// start_session body tests
// ============================================================================

TEST(MetricsPushClientStartSession, IncludesLabelAndConfigInPostBody) {
    httplib::Server svr;
    std::string captured_body;
    std::mutex body_mtx;

    // Only capture the body for the /start request.  The push thread also
    // sends a /end POST whose body is "{}" and would overwrite the capture.
    svr.Post("/.*", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.path.find("/start") != std::string::npos) {
            std::lock_guard<std::mutex> lk(body_mtx);
            captured_body = req.body;
        }
        res.status = 201;
    });

    auto svr_thread = start_server(svr, kPort_StartSession);
    MetricsPushClient client(session_url(kPort_StartSession), 500);

    const int rc = client.start_session(5, 10, 500, "my-run-label", "{\"lr\":0.001}");
    client.end_session();

    svr.stop();
    svr_thread.join();

    EXPECT_EQ(rc, 201);
    EXPECT_NE(captured_body.find("\"label\""), std::string::npos);
    EXPECT_NE(captured_body.find("my-run-label"), std::string::npos);
    EXPECT_NE(captured_body.find("\"config\""), std::string::npos);
    EXPECT_NE(captured_body.find("\"lr\""), std::string::npos);
}

#endif  // BUILD_METRICS_API_SERVER

// ============================================================================
// Tests that run without httplib
// ============================================================================

TEST(MetricsPushClientStartSession, EmptySessionUrlReturnsZero) {
    // With an empty session URL, attempt_post() is a no-op and start_session()
    // must return 0 without crashing or blocking.
    MetricsPushClient client("", /*timeout_ms=*/100);
    const int rc = client.start_session(1, 3, 100);
    EXPECT_EQ(rc, 0);
    // Destructor joins the push thread cleanly.
}

TEST(MetricsPushClientShutdownDrain, DestructorDoesNotDeadlockOrCrash) {
    // Construct a client, enqueue events, then let the scope end without
    // calling end_session().  The destructor must set stop_, notify, and join
    // without deadlocking.
    {
        MetricsPushClient client("", /*timeout_ms=*/100);
        client.start_epoch(1, 50);
        client.update_sample_metrics(1, 0.5f, 1.0f, 0.001f);
        client.update_sample_metrics(2, 0.4f, 0.9f, 0.001f);
        client.end_epoch(1, 0.5f, 0.6f, 0.001f);
        // No end_session() — destructor handles cleanup.
    }
    SUCCEED();
}
