#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "MetricsPushClient.hpp"

#ifdef BUILD_METRICS_API_SERVER
#include <httplib.h>
#endif

// Test servers bind OS-assigned ports (start_server()); only the server-down test needs a fixed
// one.
static constexpr int kPort_ServerDown = 42211;  // never listened on

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
    bool entered() const {
        return entry_count.load(std::memory_order_acquire) > 0;
    }
};

/**
 * Binds svr to an OS-assigned loopback port (written to port), runs listen_after_bind() in a
 * background thread and blocks until is_running().
 */
static std::thread start_server(httplib::Server& svr, int& port) {
    // Let the OS pick the port: fixed test ports sat in the ephemeral range, so under parallel
    // ctest another test's outgoing connection could hold one and the server failed to start.
    port = svr.bind_to_any_port("127.0.0.1");
    if (port < 0) {
        ADD_FAILURE() << "Server failed to bind a port";
        return std::thread([] {});
    }
    std::thread t([&svr]() { svr.listen_after_bind(); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!svr.is_running()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            ADD_FAILURE() << "Server failed to start on port " << port;
            break;
        }
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
static bool wait_for_gate(const Gate& gate,
                          std::chrono::milliseconds timeout_ms = std::chrono::milliseconds(2000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout_ms;
    while (!gate.entered()) {
        if (std::chrono::steady_clock::now() > deadline)
            return false;
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
                           std::chrono::milliseconds timeout_ms = std::chrono::milliseconds(2000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout_ms;
    while (count.load(std::memory_order_relaxed) < target) {
        if (std::chrono::steady_clock::now() > deadline)
            return false;
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

    int port = 0;

    auto svr_thread = start_server(svr, port);
    MetricsPushClient client(session_url(port), /*timeout_ms=*/500,
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

    int port = 0;

    auto svr_thread = start_server(svr, port);
    MetricsPushClient client(session_url(port), 500, /*max_queue_depth=*/3);

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

    int port = 0;

    auto svr_thread = start_server(svr, port);
    // max_queue_depth = 2 so we can fill it with 2 Epoch events quickly.
    MetricsPushClient client(session_url(port), 500, /*max_queue_depth=*/2);

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

    int port = 0;

    auto svr_thread = start_server(svr, port);
    MetricsPushClient client(session_url(port), /*timeout_ms=*/500);

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

    int port = 0;

    auto svr_thread = start_server(svr, port);
    MetricsPushClient client(session_url(port), 500);

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

    int port = 0;

    auto svr_thread = start_server(svr, port);
    MetricsPushClient client(session_url(port), 500);

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
    // end_session() must not return until every queued epoch/session event has been delivered.
    // TD-442: step-level Sample events still queued when it stops are dropped (the drain is
    // bounded), so only a sample already sent before the stop can arrive.
    httplib::Server svr;
    std::mutex mtx;
    std::vector<std::string> paths;

    svr.Post("/.*", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mtx);
        paths.push_back(req.path);
        res.status = 201;
    });

    int port = 0;

    auto svr_thread = start_server(svr, port);
    MetricsPushClient client(session_url(port), 500);

    client.start_epoch(1, 200);
    client.update_sample_metrics(1, 0.8f, 1.2f, 0.001f);
    client.update_sample_metrics(2, 0.7f, 1.1f, 0.001f);
    client.update_sample_metrics(3, 0.6f, 1.0f, 0.001f);
    client.end_epoch(1, 0.6f, 0.7f, 0.001f);

    // end_session() enqueues /end, sets stop_, and blocks until thread exits.
    client.end_session();

    svr.stop();
    svr_thread.join();

    std::lock_guard<std::mutex> lk(mtx);
    const auto count = [&](const std::string& suffix) {
        return std::count_if(paths.begin(), paths.end(), [&](const std::string& p) {
            return p.size() >= suffix.size() &&
                   p.compare(p.size() - suffix.size(), suffix.size(), suffix) == 0;
        });
    };
    EXPECT_EQ(count("/1-test/epoch/start"), 1);
    EXPECT_EQ(count("/1-test/epoch/end"), 1);
    EXPECT_EQ(count("/1-test/end"), 1);
    EXPECT_LE(count("/1-test/metrics/sample"), 3);
}

// TD-442: with the metrics server down, end_session() used to send every queued event with retries
// and back-off (~1.2 s each: 13.2 s for 11 events, 20-70 minutes for a full queue), delaying the
// end of the training pass. It now returns within the drain deadline.
TEST(MetricsPushClientShutdownDrain, EndSessionWithTheServerDownReturnsWithinTheDeadline) {
    MetricsPushClient client(session_url(kPort_ServerDown), 500);  // nothing listens there
    client.set_drain_timeout_ms(1000);
    for (int e = 1; e <= 20; ++e) {
        client.start_epoch(e, 100);
        client.end_epoch(e, 0.5f, 0.6f, 0.001f);
    }
    const auto t0 = std::chrono::steady_clock::now();
    client.end_session();
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_LT(elapsed, std::chrono::seconds(4))
        << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms";
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

    int port = 0;

    auto svr_thread = start_server(svr, port);
    MetricsPushClient client(session_url(port), 500);

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

// Regression test: an automatic reconnect (triggered by a 404 mid-session,
// e.g. a transient staleness eviction) must NOT replay reset_best:true even
// when the original start_session() call used it — the reconnect is resuming
// the SAME session, not starting a fresh one, and must not wipe
// best_validation_loss/best_epoch on the server every time it fires.
TEST(MetricsPushClientStartSession, ReconnectAfter404DoesNotReplayResetBest) {
    httplib::Server svr;
    std::vector<std::string> start_bodies;
    std::mutex body_mtx;
    std::atomic<int> sample_call_count{0};

    svr.Post("/.*", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.path.find("/start") != std::string::npos) {
            std::lock_guard<std::mutex> lk(body_mtx);
            start_bodies.push_back(req.body);
            res.status = 200;  // matches TrainingMetricsAPI's real /start response code
        } else if (req.path.find("/metrics/sample") != std::string::npos) {
            // First sample push looks like the session was evicted; subsequent
            // ones (after reconnect) succeed.
            // (TD-555: with the real server's body; only "Unknown session key" means a lost session.)
            if (sample_call_count.fetch_add(1, std::memory_order_relaxed) == 0) {
                res.status = 404;
                res.set_content(R"({"error":"Unknown session key: 1-test"})", "application/json");
            } else {
                res.status = 201;
            }
        } else {
            res.status = 201;
        }
    });

    int port = 0;

    auto svr_thread = start_server(svr, port);
    MetricsPushClient client(session_url(port), 500);

    const int rc = client.start_session(1, 3, 100, "", "", /*reset_best=*/true);
    ASSERT_EQ(rc, 200);

    client.update_sample_metrics(1, 0.8f, 1.2f, 0.001f);
    ASSERT_TRUE(wait_for_count(sample_call_count, 2));  // initial 404 + post-reconnect retry

    client.end_session();
    svr.stop();
    svr_thread.join();

    std::lock_guard<std::mutex> lk(body_mtx);
    ASSERT_EQ(start_bodies.size(), 2u) << "expected the initial /start plus one reconnect /start";
    EXPECT_NE(start_bodies[0].find("\"reset_best\":true"), std::string::npos)
        << "initial start_session() call should carry reset_best:true";
    EXPECT_EQ(start_bodies[1].find("reset_best"), std::string::npos)
        << "reconnect must not replay reset_best:true";
}

TEST(MetricsPushClientLayerGradientNorms, SendsBothArraysAsJson) {
    httplib::Server svr;
    std::string captured_body;
    std::mutex body_mtx;

    svr.Post("/.*", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.path.find("/metrics/layer-gradients") != std::string::npos) {
            std::lock_guard<std::mutex> lk(body_mtx);
            captured_body = req.body;
        }
        res.status = 200;
    });

    int port = 0;

    auto svr_thread = start_server(svr, port);
    MetricsPushClient client(session_url(port), 500);

    client.update_layer_gradient_norms({0.5f, 0.4f, 0.3f}, {0.6f, 0.5f});
    client.end_session();

    svr.stop();
    svr_thread.join();

    std::lock_guard<std::mutex> lk(body_mtx);
    EXPECT_NE(captured_body.find("\"encoder_layer_grad_norms\":[0.5,0.4,0.3]"), std::string::npos)
        << captured_body;
    EXPECT_NE(captured_body.find("\"decoder_layer_grad_norms\":[0.6,0.5]"), std::string::npos)
        << captured_body;
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

// TD-555: a 404 for a route the server doesn't have (POST …/metrics/abnormal used to be GET-only)
// is not a lost session. It used to re-register (/start again) on every such event.
TEST(MetricsPushClientStartSession, MissingRouteIsNotTreatedAsALostSession) {
    httplib::Server svr;
    std::atomic<int> start_calls{0};
    std::atomic<int> abnormal_calls{0};
    svr.Post("/.*", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.path.find("/start") != std::string::npos) {
            ++start_calls;
            res.status = 200;
        } else if (req.path.find("/metrics/abnormal") != std::string::npos) {
            ++abnormal_calls;
            res.status = 404;  // httplib's answer for an unmatched route: no session-key message
        } else {
            res.status = 200;
        }
    });
    int port = 0;
    auto svr_thread = start_server(svr, port);
    MetricsPushClient client(session_url(port), 500);
    ASSERT_EQ(client.start_session(1, 1, 10, "", ""), 200);

    AbnormalSample sample;
    sample.epoch = 1;
    sample.reason = "loss_outlier";
    client.flag_abnormal_sample(sample);
    client.flag_abnormal_sample(sample);
    ASSERT_TRUE(wait_for_count(abnormal_calls, 2));
    client.end_session();
    svr.stop();
    svr_thread.join();

    EXPECT_EQ(abnormal_calls.load(), 2) << "each sample is tried once, not retried after a re-start";
    EXPECT_EQ(start_calls.load(), 1) << "a missing route must not trigger re-registration";
}

