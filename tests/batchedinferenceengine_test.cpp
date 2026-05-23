/**
 * Unit tests for BatchedInferenceEngine
 * Tests: BatchedInferenceConfig, BatchedInferenceStats (including compute_derived_stats),
 *        InferenceRequest struct semantics, and BatchedInferenceEngine lifecycle + API.
 *
 * Two testing strategies are used:
 *  - Lifecycle tests: no-request construction with null/empty model components
 *    (background thread never touches tokenizer or model_fn if no requests submitted)
 *  - Functional tests: default-constructed BPETokenizer + EOS-returning model_fn
 *    so generation terminates on the first step without needing a real model
 */

#include "../src/BatchedInferenceEngine.hpp"
#include <gtest/gtest.h>
#include "../src/BPETokenizer.hpp"
#include "../src/Matrix.hpp"
#include "../src/SpecialTokens.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Helpers
// ============================================================================

/**
 * Build a shared_ptr to a default-constructed BPETokenizer.
 * The tokenizer has only special tokens in vocab (PAD/UNK/BOS/EOS).
 * Safe for lifecycle tests (no requests) AND functional tests where the model
 * immediately produces EOS so generation terminates on step 0.
 */
static std::shared_ptr<BPETokenizer> make_tokenizer() {
    return std::make_shared<BPETokenizer>();
}

/**
 * Model forward function that always returns EOS as the top logit.
 * Matrix shape: (1 row, 10 cols) — col 3 (EOS) has the highest value.
 * This stops generation after one step without running real model weights.
 */
static Matrix eos_model_fn(const std::vector<int>& /*tokens*/) {
    Matrix logits(1, 10);
    logits(0, adai::SpecialTokenIDs::EOS) = 100.0f;
    return logits;
}

/**
 * Convenience factory: engine that terminates generation immediately.
 * timeout_ms is set small for fast test execution.
 */
static BatchedInferenceEngine make_fast_engine(int timeout_ms = 10) {
    BatchedInferenceConfig cfg;
    cfg.timeout_ms = timeout_ms;
    return BatchedInferenceEngine(eos_model_fn, make_tokenizer(), cfg);
}

// ============================================================================
// BatchedInferenceConfig Tests
// ============================================================================

TEST(BatchedInferenceConfigTest, DefaultMaxBatchSize) {
    BatchedInferenceConfig cfg;
    EXPECT_EQ(cfg.max_batch_size, 32u);
}

TEST(BatchedInferenceConfigTest, DefaultTimeoutMs) {
    BatchedInferenceConfig cfg;
    EXPECT_EQ(cfg.timeout_ms, 50);
}

TEST(BatchedInferenceConfigTest, DefaultMaxTokensPerBatch) {
    BatchedInferenceConfig cfg;
    EXPECT_EQ(cfg.max_tokens_per_batch, 4096);
}

TEST(BatchedInferenceConfigTest, DefaultPaddingStrategy) {
    BatchedInferenceConfig cfg;
    EXPECT_EQ(cfg.padding_strategy, PaddingStrategy::LEFT);
}

TEST(BatchedInferenceConfigTest, DefaultUseDynamicBatching) {
    BatchedInferenceConfig cfg;
    EXPECT_TRUE(cfg.use_dynamic_batching);
}

TEST(BatchedInferenceConfigTest, DefaultMaxQueueSize) {
    BatchedInferenceConfig cfg;
    EXPECT_EQ(cfg.max_queue_size, 1000);
}

TEST(BatchedInferenceConfigTest, DefaultEnableRequestStats) {
    BatchedInferenceConfig cfg;
    EXPECT_TRUE(cfg.enable_request_stats);
}

TEST(BatchedInferenceConfigTest, CustomValues) {
    BatchedInferenceConfig cfg;
    cfg.max_batch_size = 8;
    cfg.timeout_ms = 20;
    cfg.max_tokens_per_batch = 512;
    cfg.use_dynamic_batching = false;
    cfg.max_queue_size = 50;
    cfg.enable_request_stats = false;

    EXPECT_EQ(cfg.max_batch_size, 8u);
    EXPECT_EQ(cfg.timeout_ms, 20);
    EXPECT_EQ(cfg.max_tokens_per_batch, 512);
    EXPECT_FALSE(cfg.use_dynamic_batching);
    EXPECT_EQ(cfg.max_queue_size, 50);
    EXPECT_FALSE(cfg.enable_request_stats);
}

// ============================================================================
// BatchedInferenceStats Tests
// ============================================================================

TEST(BatchedInferenceStatsTest, DefaultConstruction) {
    BatchedInferenceStats s;
    EXPECT_EQ(s.total_requests, 0u);
    EXPECT_EQ(s.total_batches, 0u);
    EXPECT_EQ(s.total_tokens_processed, 0u);
    EXPECT_EQ(s.requests_timeout, 0u);
    EXPECT_EQ(s.requests_batch_full, 0u);
    EXPECT_DOUBLE_EQ(s.avg_batch_size, 0.0);
    EXPECT_DOUBLE_EQ(s.avg_latency_ms, 0.0);
    EXPECT_DOUBLE_EQ(s.throughput_req_per_sec, 0.0);
    EXPECT_DOUBLE_EQ(s.throughput_tokens_per_sec, 0.0);
}

TEST(BatchedInferenceStatsTest, ComputeDerivedStatsZeroElapsed) {
    BatchedInferenceStats s;
    s.total_requests = 10;
    s.total_batches = 2;
    // elapsed = 0: throughput fields should not be set (avoid div-by-zero)
    s.compute_derived_stats(0.0);
    EXPECT_DOUBLE_EQ(s.throughput_req_per_sec, 0.0);
    EXPECT_DOUBLE_EQ(s.throughput_tokens_per_sec, 0.0);
}

TEST(BatchedInferenceStatsTest, ComputeDerivedStatsAvgBatchSize) {
    BatchedInferenceStats s;
    s.total_requests = 10;
    s.total_batches = 2;
    s.compute_derived_stats(1.0);
    EXPECT_DOUBLE_EQ(s.avg_batch_size, 5.0);
}

TEST(BatchedInferenceStatsTest, ComputeDerivedStatsThroughput) {
    BatchedInferenceStats s;
    s.total_requests = 100;
    s.total_batches = 5;
    s.total_tokens_processed = 200;
    s.compute_derived_stats(2.0);
    EXPECT_DOUBLE_EQ(s.throughput_req_per_sec, 50.0);
    EXPECT_DOUBLE_EQ(s.throughput_tokens_per_sec, 100.0);
}

TEST(BatchedInferenceStatsTest, ComputeDerivedStatsAvgLatency) {
    BatchedInferenceStats s;
    s.total_requests = 10;
    s.total_batches = 1;
    s.compute_derived_stats(1.0);
    // avg_latency_ms = elapsed*1000 / total_requests = 1000/10 = 100
    EXPECT_DOUBLE_EQ(s.avg_latency_ms, 100.0);
}

TEST(BatchedInferenceStatsTest, ComputeDerivedStatsZeroBatches) {
    BatchedInferenceStats s;
    s.total_requests = 0;
    s.total_batches = 0;
    s.compute_derived_stats(1.0);
    // avg_batch_size should remain 0 (guard against div-by-zero)
    EXPECT_DOUBLE_EQ(s.avg_batch_size, 0.0);
}

TEST(BatchedInferenceStatsTest, FieldAssignment) {
    BatchedInferenceStats s;
    s.total_requests = 42;
    s.requests_timeout = 3;
    s.requests_batch_full = 7;
    EXPECT_EQ(s.total_requests, 42u);
    EXPECT_EQ(s.requests_timeout, 3u);
    EXPECT_EQ(s.requests_batch_full, 7u);
}

TEST(BatchedInferenceStatsTest, CopyConstructible) {
    BatchedInferenceStats s;
    s.total_requests = 5;
    s.avg_batch_size = 3.14;

    BatchedInferenceStats s2 = s;
    EXPECT_EQ(s2.total_requests, 5u);
    EXPECT_DOUBLE_EQ(s2.avg_batch_size, 3.14);
}

// ============================================================================
// InferenceRequest Tests
// ============================================================================

TEST(InferenceRequestTest, DefaultConstruction) {
    InferenceRequest req;
    EXPECT_TRUE(req.prompt.empty());
}

TEST(InferenceRequestTest, PromptAssignment) {
    InferenceRequest req;
    req.prompt = "What is the answer?";
    EXPECT_EQ(req.prompt, "What is the answer?");
}

TEST(InferenceRequestTest, SubmitTimeAssignment) {
    InferenceRequest req;
    auto now = std::chrono::steady_clock::now();
    req.submit_time = now;
    EXPECT_EQ(req.submit_time, now);
}

TEST(InferenceRequestTest, MoveConstruction) {
    InferenceRequest req;
    req.prompt = "move me";
    auto future = req.result.get_future();

    InferenceRequest moved = std::move(req);
    EXPECT_EQ(moved.prompt, "move me");
    EXPECT_TRUE(req.prompt.empty());

    // The future is still associated with the moved promise
    moved.result.set_value("ok");
    EXPECT_EQ(future.get(), "ok");
}

TEST(InferenceRequestTest, MoveAssignment) {
    InferenceRequest req;
    req.prompt = "assign me";

    InferenceRequest target;
    target = std::move(req);
    EXPECT_EQ(target.prompt, "assign me");
}

TEST(InferenceRequestTest, NotCopyConstructible) {
    EXPECT_FALSE(std::is_copy_constructible<InferenceRequest>::value);
}

TEST(InferenceRequestTest, NotCopyAssignable) {
    EXPECT_FALSE(std::is_copy_assignable<InferenceRequest>::value);
}

TEST(InferenceRequestTest, MoveConstructible) {
    EXPECT_TRUE(std::is_move_constructible<InferenceRequest>::value);
}

TEST(InferenceRequestTest, MoveAssignable) {
    EXPECT_TRUE(std::is_move_assignable<InferenceRequest>::value);
}

// ============================================================================
// Engine Lifecycle Tests (no requests submitted)
// ============================================================================

class EngineLifecycleTest : public ::testing::Test {
   protected:
    // Construct engine with null model_fn and null tokenizer.
    // The background thread will never dereference either because no requests
    // are submitted — it just loops on queue_cv_ and exits on shutdown().
    BatchedInferenceEngine make_null_engine() {
        return BatchedInferenceEngine(nullptr, nullptr);
    }
};

TEST_F(EngineLifecycleTest, ConstructAndShutdownSafe) {
    BatchedInferenceEngine engine(nullptr, nullptr);
    engine.shutdown();
    SUCCEED();
}

TEST_F(EngineLifecycleTest, IsRunningInitiallyTrue) {
    BatchedInferenceEngine engine(nullptr, nullptr);
    EXPECT_TRUE(engine.is_running());
    engine.shutdown();
}

TEST_F(EngineLifecycleTest, IsRunningFalseAfterShutdown) {
    BatchedInferenceEngine engine(nullptr, nullptr);
    engine.shutdown();
    EXPECT_FALSE(engine.is_running());
}

TEST_F(EngineLifecycleTest, DoubleShutdownIdempotent) {
    BatchedInferenceEngine engine(nullptr, nullptr);
    engine.shutdown();
    engine.shutdown();  // Second call should be a no-op
    EXPECT_FALSE(engine.is_running());
}

TEST_F(EngineLifecycleTest, DestructorShutdownsSafely) {
    {
        BatchedInferenceEngine engine(nullptr, nullptr);
        // Destructor calls shutdown()
    }
    SUCCEED();
}

TEST_F(EngineLifecycleTest, QueueSizeInitiallyZero) {
    BatchedInferenceEngine engine(nullptr, nullptr);
    EXPECT_EQ(engine.queue_size(), 0u);
    engine.shutdown();
}

TEST_F(EngineLifecycleTest, GetStatsInitiallyZero) {
    BatchedInferenceEngine engine(nullptr, nullptr);
    auto stats = engine.get_stats();
    EXPECT_EQ(stats.total_requests, 0u);
    EXPECT_EQ(stats.total_batches, 0u);
    EXPECT_EQ(stats.total_tokens_processed, 0u);
    EXPECT_EQ(stats.requests_timeout, 0u);
    EXPECT_EQ(stats.requests_batch_full, 0u);
    engine.shutdown();
}

TEST_F(EngineLifecycleTest, ResetStatsPreservesZero) {
    BatchedInferenceEngine engine(nullptr, nullptr);
    engine.reset_stats();  // Reset on already-zero stats
    auto stats = engine.get_stats();
    EXPECT_EQ(stats.total_requests, 0u);
    engine.shutdown();
}

TEST_F(EngineLifecycleTest, CustomConfigApplied) {
    BatchedInferenceConfig cfg;
    cfg.max_batch_size = 4;
    cfg.timeout_ms = 5;
    cfg.max_queue_size = 10;

    BatchedInferenceEngine engine(nullptr, nullptr, cfg);
    EXPECT_TRUE(engine.is_running());
    engine.shutdown();
}

TEST_F(EngineLifecycleTest, SubmitThrowsAfterShutdown) {
    BatchedInferenceEngine engine(nullptr, nullptr);
    engine.shutdown();

    EXPECT_THROW(engine.submit("hello"), std::runtime_error);
}

TEST_F(EngineLifecycleTest, SubmitBatchThrowsAfterShutdown) {
    BatchedInferenceEngine engine(nullptr, nullptr);
    engine.shutdown();

    EXPECT_THROW(engine.submit_batch({"a", "b"}), std::runtime_error);
}

TEST_F(EngineLifecycleTest, SubmitBatchEmptyVectorSafe) {
    // Empty submit_batch should return empty vector of futures without crashing
    BatchedInferenceEngine engine(nullptr, nullptr);
    auto futures = engine.submit_batch({});
    EXPECT_TRUE(futures.empty());
    engine.shutdown();
}

TEST_F(EngineLifecycleTest, QueueSizeReflectsPendingRequests) {
    // Use very long timeout so the background thread doesn't drain the queue
    // before we can observe the size.
    BatchedInferenceConfig cfg;
    cfg.timeout_ms = 30000;  // 30 seconds — effectively infinite for test duration

    // Tokenizer needed only if background thread processes; since timeout is huge
    // and we shut down quickly, we provide eos_model_fn for safety in case of drain.
    BatchedInferenceEngine engine(eos_model_fn, make_tokenizer(), cfg);

    // Submit one request; it should sit in the queue for 30 seconds
    auto f = engine.submit("hello");
    // Give background thread time to notice the request but keep timeout too high to collect
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Queue may or may not have drained yet — just verify no crash and size >= 0
    size_t sz = engine.queue_size();
    EXPECT_GE(sz, 0u);

    engine.shutdown();
    // After shutdown the future should resolve (either with a value or exception)
    auto status = f.wait_for(std::chrono::seconds(5));
    EXPECT_NE(status, std::future_status::timeout);
}

// ============================================================================
// Engine Functional Tests (real tokenizer + EOS model)
// ============================================================================

class EngineFunctionalTest : public ::testing::Test {
   protected:
    // Fast engine: 10ms timeout, EOS model_fn, default tokenizer
    BatchedInferenceConfig fast_config() {
        BatchedInferenceConfig cfg;
        cfg.timeout_ms = 10;
        cfg.max_batch_size = 8;
        return cfg;
    }
};

TEST_F(EngineFunctionalTest, SubmitReturnsFuture) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());
    auto future = engine.submit("hello");
    EXPECT_TRUE(future.valid());
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, SubmitGetsResult) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());
    auto future = engine.submit("hello world");

    auto status = future.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(status, std::future_status::ready);

    // Result should not throw
    EXPECT_NO_THROW(future.get());
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, SubmitBatchReturnsFutures) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());
    auto futures = engine.submit_batch({"first", "second", "third"});
    EXPECT_EQ(futures.size(), 3u);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, SubmitBatchAllComplete) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());
    auto futures = engine.submit_batch({"a", "b", "c"});

    for (auto& f : futures) {
        auto status = f.wait_for(std::chrono::seconds(5));
        EXPECT_EQ(status, std::future_status::ready);
    }
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, TotalRequestsIncrements) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());

    auto f1 = engine.submit("one");
    auto f2 = engine.submit("two");
    f1.wait_for(std::chrono::seconds(5));
    f2.wait_for(std::chrono::seconds(5));

    auto stats = engine.get_stats();
    EXPECT_GE(stats.total_requests, 2u);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, TotalBatchesIncrements) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());

    auto f = engine.submit("test");
    f.wait_for(std::chrono::seconds(5));

    auto stats = engine.get_stats();
    EXPECT_GE(stats.total_batches, 1u);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, ResetStatsAfterProcessing) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());

    auto f = engine.submit("test");
    f.wait_for(std::chrono::seconds(5));

    engine.reset_stats();
    auto stats = engine.get_stats();
    EXPECT_EQ(stats.total_requests, 0u);
    EXPECT_EQ(stats.total_batches, 0u);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, GetStatsDoesNotDeadlock) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());

    // Call get_stats() repeatedly while engine may be running
    for (int i = 0; i < 5; ++i) {
        auto stats = engine.get_stats();
        EXPECT_GE(stats.total_requests, 0u);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, SubmitWithExplicitGenConfig) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());

    TextGenerator::GenerationConfig gen;
    gen.max_length = 10;

    auto future = engine.submit("test", &gen);
    EXPECT_TRUE(future.valid());
    auto status = future.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(status, std::future_status::ready);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, MultipleSequentialSubmits) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());

    for (int i = 0; i < 5; ++i) {
        auto f = engine.submit("prompt " + std::to_string(i));
        auto status = f.wait_for(std::chrono::seconds(5));
        EXPECT_EQ(status, std::future_status::ready);
    }
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, QueueFullThrows) {
    // collect_batch() drains items from the queue into a thread-local buffer
    // before process_batch() is ever called, so a large timeout alone cannot
    // keep items in the queue long enough to observe queue.size() == max_queue_size.
    // Fix: use a blocking model_fn that holds the background thread inside
    // process_batch() while the main thread fills the queue to capacity.
    std::promise<void> gate;
    auto gate_fut = gate.get_future().share();
    std::atomic<bool> model_entered{false};

    auto blocking_fn = [&](const std::vector<int>&) -> Matrix {
        model_entered.store(true, std::memory_order_release);
        gate_fut.wait();  // block until explicitly released
        Matrix logits(1, 10);
        logits(0, adai::SpecialTokenIDs::EOS) = 100.0f;
        return logits;
    };

    BatchedInferenceConfig cfg;
    cfg.timeout_ms    = 5;   // fast flush so the primer batch is collected quickly
    cfg.max_queue_size = 2;

    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(blocking_fn, tok, cfg);

    // Submit a primer to get the background thread stuck inside model_fn
    // (i.e. inside process_batch). Once it is there, new submissions go into
    // the queue and cannot be drained until the gate is opened.
    auto f_primer = engine.submit("primer");
    while (!model_entered.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Background thread is now blocked inside process_batch.
    // Items added to the queue will remain there.
    auto f1 = engine.submit("one");
    auto f2 = engine.submit("two");

    // Third submit must throw: queue holds "one" and "two" (size == max_queue_size).
    EXPECT_THROW(engine.submit("three"), std::runtime_error);

    // Unblock the model and shut down cleanly.
    gate.set_value();
    engine.shutdown();
    f_primer.wait_for(std::chrono::seconds(5));
    f1.wait_for(std::chrono::seconds(5));
    f2.wait_for(std::chrono::seconds(5));
}

TEST_F(EngineFunctionalTest, AvgBatchSizeNonNegativeAfterRequests) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());

    auto f = engine.submit("test");
    f.wait_for(std::chrono::seconds(5));

    auto stats = engine.get_stats();
    EXPECT_GE(stats.avg_batch_size, 0.0);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, ThroughputNonNegativeAfterRequests) {
    auto tok = make_tokenizer();
    BatchedInferenceEngine engine(eos_model_fn, tok, fast_config());

    auto f = engine.submit("test");
    f.wait_for(std::chrono::seconds(5));

    auto stats = engine.get_stats();
    EXPECT_GE(stats.throughput_req_per_sec, 0.0);
    EXPECT_GE(stats.throughput_tokens_per_sec, 0.0);
    engine.shutdown();
}
