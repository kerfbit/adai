/**
 * Unit tests for PipelineInferenceEngine
 * Tests: ThreadSafeQueue, PipelineRequest, EncoderOutput, PipelineConfig,
 *        PipelineStats, and PipelineInferenceEngine lifecycle + functionality
 */

#include "../src/PipelineInferenceEngine.hpp"
#include <gtest/gtest.h>
#include "../src/Matrix.hpp"
#include "../src/SpecialTokens.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Mock types for template instantiation without real model dependencies
// ============================================================================

class MockEncoder {
   public:
    Matrix encode(const std::string& /*text*/) {
        return Matrix(2, 8);  // 2-token sequence, 8-dim hidden
    }
};

class MockDecoder {
   public:
    // Returns hidden states; called with (tokens, encoder_output, mask_ptr)
    Matrix forward_with_cross_attention(const std::vector<int>& tokens,
                                        const Matrix& /*encoder_output*/, const Matrix* /*mask*/) {
        int rows = static_cast<int>(tokens.size());
        return Matrix(rows > 0 ? rows : 1, 8);
    }
};

class MockLMHead {
   public:
    // Returns logits: make EOS token (id=3) always the highest so generation
    // terminates immediately on the first decoder step.
    Matrix forward(const Matrix& /*hidden*/) {
        Matrix logits(1, 10);  // vocab size = 10
        logits(0, adai::SpecialTokenIDs::EOS) = 100.0f;
        return logits;
    }
};

class MockTokenizer {
   public:
    std::string decode(const std::vector<int>& tokens) {
        return "decoded_" + std::to_string(tokens.size());
    }
};

// EOS-selecting head that only emits EOS after N steps (for multi-step tests)
class SlowMockLMHead {
   public:
    explicit SlowMockLMHead(int steps_before_eos = 2) : steps_(0), limit_(steps_before_eos) {}

    Matrix forward(const Matrix& /*hidden*/) {
        Matrix logits(1, 10);
        if (++steps_ >= limit_) {
            logits(0, adai::SpecialTokenIDs::EOS) = 100.0f;
        } else {
            logits(0, 5) = 100.0f;  // token 5 (not BOS/EOS/PAD)
        }
        return logits;
    }

   private:
    int steps_;
    int limit_;
};

using TestPipelineEngine =
    PipelineInferenceEngine<MockEncoder, MockDecoder, MockLMHead, MockTokenizer>;

// ============================================================================
// PipelineConfig Tests
// ============================================================================

TEST(PipelineConfigTest, DefaultValues) {
    PipelineConfig cfg;
    EXPECT_EQ(cfg.max_queue_size, 50u);
    EXPECT_EQ(cfg.encoder_timeout_ms, 50);
    EXPECT_EQ(cfg.decoder_timeout_ms, 50);
    EXPECT_TRUE(cfg.enable_profiling);
}

TEST(PipelineConfigTest, CustomValues) {
    PipelineConfig cfg;
    cfg.max_queue_size = 10;
    cfg.encoder_timeout_ms = 100;
    cfg.decoder_timeout_ms = 200;
    cfg.enable_profiling = false;

    EXPECT_EQ(cfg.max_queue_size, 10u);
    EXPECT_EQ(cfg.encoder_timeout_ms, 100);
    EXPECT_EQ(cfg.decoder_timeout_ms, 200);
    EXPECT_FALSE(cfg.enable_profiling);
}

TEST(PipelineConfigTest, DefaultConstructibleWithNew) {
    PipelineConfig* cfg = new PipelineConfig();
    EXPECT_EQ(cfg->max_queue_size, 50u);
    delete cfg;
}

// ============================================================================
// PipelineStats Tests
// ============================================================================

TEST(PipelineStatsTest, DefaultConstruction) {
    PipelineStats s;
    EXPECT_EQ(s.total_requests, 0u);
    EXPECT_EQ(s.encoder_processed, 0u);
    EXPECT_EQ(s.decoder_processed, 0u);
    EXPECT_DOUBLE_EQ(s.avg_encoder_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(s.avg_decoder_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(s.avg_total_latency_ms, 0.0);
    EXPECT_DOUBLE_EQ(s.avg_throughput_rps, 0.0);
    EXPECT_EQ(s.encoder_queue_size, 0u);
    EXPECT_EQ(s.decoder_queue_size, 0u);
}

TEST(PipelineStatsTest, FieldAssignment) {
    PipelineStats s;
    s.total_requests = 10;
    s.encoder_processed = 8;
    s.decoder_processed = 6;
    s.avg_encoder_time_ms = 5.5;
    s.avg_throughput_rps = 100.0;

    EXPECT_EQ(s.total_requests, 10u);
    EXPECT_EQ(s.encoder_processed, 8u);
    EXPECT_EQ(s.decoder_processed, 6u);
    EXPECT_DOUBLE_EQ(s.avg_encoder_time_ms, 5.5);
    EXPECT_DOUBLE_EQ(s.avg_throughput_rps, 100.0);
}

TEST(PipelineStatsTest, CopyConstructible) {
    PipelineStats s;
    s.total_requests = 5;
    s.avg_encoder_time_ms = 3.14;

    PipelineStats s2 = s;
    EXPECT_EQ(s2.total_requests, 5u);
    EXPECT_DOUBLE_EQ(s2.avg_encoder_time_ms, 3.14);
}

// ============================================================================
// PipelineRequest Tests
// ============================================================================

TEST(PipelineRequestTest, DefaultConstruction) {
    PipelineRequest req;
    EXPECT_EQ(req.request_id, 0u);
    EXPECT_EQ(req.max_length, 100);
    EXPECT_TRUE(req.input_texts.empty());
}

TEST(PipelineRequestTest, FieldAssignment) {
    PipelineRequest req;
    req.request_id = 42;
    req.input_texts = {"hello", "world"};
    req.max_length = 50;

    EXPECT_EQ(req.request_id, 42u);
    EXPECT_EQ(req.input_texts.size(), 2u);
    EXPECT_EQ(req.max_length, 50);
}

TEST(PipelineRequestTest, MoveConstruction) {
    PipelineRequest req;
    req.request_id = 7;
    req.input_texts = {"move me"};
    req.max_length = 20;

    PipelineRequest moved = std::move(req);
    EXPECT_EQ(moved.request_id, 7u);
    EXPECT_EQ(moved.input_texts.size(), 1u);
    EXPECT_EQ(moved.max_length, 20);
}

TEST(PipelineRequestTest, MoveAssignment) {
    PipelineRequest req;
    req.request_id = 99;
    req.input_texts = {"assign me"};

    PipelineRequest target;
    target = std::move(req);
    EXPECT_EQ(target.request_id, 99u);
    EXPECT_EQ(target.input_texts.size(), 1u);
}

TEST(PipelineRequestTest, NotCopyConstructible) {
    EXPECT_FALSE(std::is_copy_constructible<PipelineRequest>::value);
}

TEST(PipelineRequestTest, NotCopyAssignable) {
    EXPECT_FALSE(std::is_copy_assignable<PipelineRequest>::value);
}

TEST(PipelineRequestTest, MoveConstructible) {
    EXPECT_TRUE(std::is_move_constructible<PipelineRequest>::value);
}

TEST(PipelineRequestTest, MoveAssignable) {
    EXPECT_TRUE(std::is_move_assignable<PipelineRequest>::value);
}

// ============================================================================
// EncoderOutput Tests
// ============================================================================

TEST(EncoderOutputTest, DefaultConstruction) {
    EncoderOutput out;
    EXPECT_EQ(out.request_id, 0u);
    EXPECT_EQ(out.max_length, 100);
    EXPECT_TRUE(out.encoder_outputs.empty());
    EXPECT_TRUE(out.input_texts.empty());
}

TEST(EncoderOutputTest, FieldAssignment) {
    EncoderOutput out;
    out.request_id = 5;
    out.max_length = 64;
    out.encoder_outputs.push_back(Matrix(2, 8));
    out.input_texts = {"hi"};

    EXPECT_EQ(out.request_id, 5u);
    EXPECT_EQ(out.max_length, 64);
    EXPECT_EQ(out.encoder_outputs.size(), 1u);
    EXPECT_EQ(out.input_texts.size(), 1u);
}

TEST(EncoderOutputTest, MoveConstruction) {
    EncoderOutput out;
    out.request_id = 3;
    out.encoder_outputs.push_back(Matrix(4, 16));

    EncoderOutput moved = std::move(out);
    EXPECT_EQ(moved.request_id, 3u);
    EXPECT_EQ(moved.encoder_outputs.size(), 1u);
}

TEST(EncoderOutputTest, MoveAssignment) {
    EncoderOutput out;
    out.request_id = 8;
    out.encoder_outputs.push_back(Matrix(1, 4));
    out.input_texts = {"test"};

    EncoderOutput target;
    target = std::move(out);
    EXPECT_EQ(target.request_id, 8u);
    EXPECT_EQ(target.encoder_outputs.size(), 1u);
}

TEST(EncoderOutputTest, NotCopyConstructible) {
    EXPECT_FALSE(std::is_copy_constructible<EncoderOutput>::value);
}

TEST(EncoderOutputTest, MoveConstructible) {
    EXPECT_TRUE(std::is_move_constructible<EncoderOutput>::value);
}

// ============================================================================
// ThreadSafeQueue Tests
// ============================================================================

TEST(ThreadSafeQueueTest, DefaultConstruction) {
    ThreadSafeQueue<int> q;
    EXPECT_EQ(q.size(), 0u);
    EXPECT_TRUE(q.empty());
    EXPECT_FALSE(q.is_shutdown());
}

TEST(ThreadSafeQueueTest, CustomMaxSize) {
    ThreadSafeQueue<int> q(5);
    EXPECT_TRUE(q.empty());
    EXPECT_FALSE(q.is_shutdown());
}

TEST(ThreadSafeQueueTest, PushAndSize) {
    ThreadSafeQueue<int> q(10);
    q.push(1);
    EXPECT_EQ(q.size(), 1u);
    q.push(2);
    EXPECT_EQ(q.size(), 2u);
    EXPECT_FALSE(q.empty());
}

TEST(ThreadSafeQueueTest, TryPopSuccess) {
    ThreadSafeQueue<int> q(10);
    q.push(42);

    int val = 0;
    bool ok = q.try_pop(val, 100);
    EXPECT_TRUE(ok);
    EXPECT_EQ(val, 42);
    EXPECT_TRUE(q.empty());
}

TEST(ThreadSafeQueueTest, TryPopTimeout) {
    ThreadSafeQueue<int> q(10);
    // Queue is empty; should timeout quickly
    int val = 0;
    bool ok = q.try_pop(val, 10);  // 10ms timeout
    EXPECT_FALSE(ok);
}

TEST(ThreadSafeQueueTest, TryPopFIFOOrder) {
    ThreadSafeQueue<int> q(10);
    q.push(1);
    q.push(2);
    q.push(3);

    int a = 0, b = 0, c = 0;
    EXPECT_TRUE(q.try_pop(a, 100));
    EXPECT_TRUE(q.try_pop(b, 100));
    EXPECT_TRUE(q.try_pop(c, 100));
    EXPECT_EQ(a, 1);
    EXPECT_EQ(b, 2);
    EXPECT_EQ(c, 3);
}

TEST(ThreadSafeQueueTest, Shutdown) {
    ThreadSafeQueue<int> q(10);
    EXPECT_FALSE(q.is_shutdown());
    q.shutdown();
    EXPECT_TRUE(q.is_shutdown());
}

TEST(ThreadSafeQueueTest, TryPopReturnsFalseAfterShutdown) {
    ThreadSafeQueue<int> q(10);
    q.shutdown();
    int val = 0;
    bool ok = q.try_pop(val, 100);
    EXPECT_FALSE(ok);
}

TEST(ThreadSafeQueueTest, ShutdownUnblocksWaitingThread) {
    ThreadSafeQueue<int> q(10);
    std::atomic<bool> thread_done{false};

    std::thread t([&]() {
        int val = 0;
        // pop blocks; shutdown should unblock it
        bool ok = q.pop(val);
        (void)ok;
        thread_done = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(thread_done.load());

    q.shutdown();
    t.join();
    EXPECT_TRUE(thread_done.load());
}

TEST(ThreadSafeQueueTest, PopWithDataAvailable) {
    ThreadSafeQueue<int> q(10);
    q.push(77);

    int val = 0;
    bool ok = q.pop(val);
    EXPECT_TRUE(ok);
    EXPECT_EQ(val, 77);
}

TEST(ThreadSafeQueueTest, EmptyAfterConsuming) {
    ThreadSafeQueue<int> q(10);
    q.push(1);
    q.push(2);

    int val = 0;
    q.try_pop(val, 100);
    q.try_pop(val, 100);
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);
}

TEST(ThreadSafeQueueTest, PushStrings) {
    ThreadSafeQueue<std::string> q(5);
    q.push("hello");
    q.push("world");
    EXPECT_EQ(q.size(), 2u);

    std::string s;
    EXPECT_TRUE(q.try_pop(s, 100));
    EXPECT_EQ(s, "hello");
}

TEST(ThreadSafeQueueTest, MoveOnlyTypes) {
    ThreadSafeQueue<std::unique_ptr<int>> q(5);
    q.push(std::make_unique<int>(99));
    EXPECT_EQ(q.size(), 1u);

    std::unique_ptr<int> ptr;
    bool ok = q.try_pop(ptr, 100);
    EXPECT_TRUE(ok);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, 99);
}

TEST(ThreadSafeQueueTest, MultipleShutdownsSafe) {
    ThreadSafeQueue<int> q(10);
    q.shutdown();
    q.shutdown();  // Should not crash
    EXPECT_TRUE(q.is_shutdown());
}

// ============================================================================
// Engine Lifecycle Tests (no requests submitted)
// ============================================================================

class EngineLifecycleTest : public ::testing::Test {
   protected:
    MockEncoder enc_;
    MockDecoder dec_;
    MockLMHead lm_;
    MockTokenizer tok_;
};

TEST_F(EngineLifecycleTest, ConstructAndShutdownWithNullPointers) {
    // Null-pointer construction is safe as long as no requests are submitted
    TestPipelineEngine engine(nullptr, nullptr, nullptr, nullptr);
    engine.shutdown();
    SUCCEED();
}

TEST_F(EngineLifecycleTest, IsRunningInitiallyTrue) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_);
    EXPECT_TRUE(engine.is_running());
    engine.shutdown();
}

TEST_F(EngineLifecycleTest, IsRunningFalseAfterShutdown) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_);
    engine.shutdown();
    EXPECT_FALSE(engine.is_running());
}

TEST_F(EngineLifecycleTest, DoubleShutdownSafe) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_);
    engine.shutdown();
    engine.shutdown();  // Second call should be idempotent
    EXPECT_FALSE(engine.is_running());
}

TEST_F(EngineLifecycleTest, DestructorSafe) {
    {
        TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_);
        // Destructor should shutdown cleanly
    }
    SUCCEED();
}

TEST_F(EngineLifecycleTest, DestructorSafeWithNullPointers) {
    {
        TestPipelineEngine engine(nullptr, nullptr, nullptr, nullptr);
        // No requests - workers see null but should not reach dereference
    }
    SUCCEED();
}

TEST_F(EngineLifecycleTest, InitialStatsAreZero) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_);
    PipelineStats stats = engine.get_stats();
    EXPECT_EQ(stats.total_requests, 0u);
    EXPECT_EQ(stats.encoder_processed, 0u);
    EXPECT_EQ(stats.decoder_processed, 0u);
    EXPECT_DOUBLE_EQ(stats.avg_encoder_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats.avg_decoder_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats.avg_total_latency_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats.avg_throughput_rps, 0.0);
    engine.shutdown();
}

TEST_F(EngineLifecycleTest, InitialQueueSizesZero) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_);
    PipelineStats stats = engine.get_stats();
    EXPECT_EQ(stats.encoder_queue_size, 0u);
    EXPECT_EQ(stats.decoder_queue_size, 0u);
    engine.shutdown();
}

TEST_F(EngineLifecycleTest, CustomConfigPropagated) {
    PipelineConfig cfg;
    cfg.max_queue_size = 5;
    cfg.encoder_timeout_ms = 10;
    cfg.decoder_timeout_ms = 20;

    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, cfg);
    EXPECT_TRUE(engine.is_running());
    engine.shutdown();
}

// ============================================================================
// Engine Functional Tests (with real mock processing)
// ============================================================================

class EngineFunctionalTest : public ::testing::Test {
   protected:
    MockEncoder enc_;
    MockDecoder dec_;
    MockLMHead lm_;
    MockTokenizer tok_;

    PipelineConfig fast_cfg() {
        PipelineConfig cfg;
        cfg.encoder_timeout_ms = 10;
        cfg.decoder_timeout_ms = 10;
        return cfg;
    }
};

TEST_F(EngineFunctionalTest, SubmitBatchReturnsFuture) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit_batch({"hello"}, 10);
    EXPECT_TRUE(future.valid());
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, SubmitBatchGetsResult) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit_batch({"hello world"}, 50);

    // Wait for result (up to 5 seconds)
    auto status = future.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(status, std::future_status::ready);

    auto results = future.get();
    EXPECT_EQ(results.size(), 1u);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, SubmitBatchMultipleInputs) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit_batch({"first", "second", "third"}, 50);

    auto status = future.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(status, std::future_status::ready);

    auto results = future.get();
    EXPECT_EQ(results.size(), 3u);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, SubmitSingleReturnsFuture) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit("hello", 10);
    EXPECT_TRUE(future.valid());
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, SubmitSingleGetsResult) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit("test input", 50);

    // submit() wraps in std::launch::deferred, so the future resolves on .get()
    // (wait_for always returns deferred for such futures, not ready)
    std::string result = future.get();
    EXPECT_FALSE(result.empty());
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, TotalRequestsIncrements) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());

    auto f1 = engine.submit_batch({"one"}, 10);
    auto f2 = engine.submit_batch({"two"}, 10);

    // Wait for completion
    f1.wait_for(std::chrono::seconds(5));
    f2.wait_for(std::chrono::seconds(5));

    PipelineStats stats = engine.get_stats();
    EXPECT_GE(stats.total_requests, 2u);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, EncoderProcessedIncrements) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit_batch({"test"}, 10);

    auto status = future.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(status, std::future_status::ready);
    future.get();  // consume

    PipelineStats stats = engine.get_stats();
    EXPECT_GE(stats.encoder_processed, 1u);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, DecoderProcessedIncrements) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit_batch({"test"}, 10);

    auto status = future.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(status, std::future_status::ready);
    future.get();

    PipelineStats stats = engine.get_stats();
    EXPECT_GE(stats.decoder_processed, 1u);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, AvgEncoderTimeNonNegativeAfterRequest) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit_batch({"test"}, 10);

    future.wait_for(std::chrono::seconds(5));
    future.get();

    PipelineStats stats = engine.get_stats();
    EXPECT_GE(stats.avg_encoder_time_ms, 0.0);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, AvgDecoderTimeNonNegativeAfterRequest) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit_batch({"test"}, 10);

    future.wait_for(std::chrono::seconds(5));
    future.get();

    PipelineStats stats = engine.get_stats();
    EXPECT_GE(stats.avg_decoder_time_ms, 0.0);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, ThroughputPositiveAfterRequest) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit_batch({"test"}, 10);

    future.wait_for(std::chrono::seconds(5));
    future.get();

    PipelineStats stats = engine.get_stats();
    EXPECT_GE(stats.avg_throughput_rps, 0.0);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, SubmitEmptyBatchReturnsEmptyVector) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());
    auto future = engine.submit_batch({}, 10);

    auto status = future.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(status, std::future_status::ready);

    auto results = future.get();
    EXPECT_TRUE(results.empty());
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, MultipleSequentialBatches) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());

    for (int i = 0; i < 3; ++i) {
        auto future = engine.submit_batch({"input " + std::to_string(i)}, 10);
        auto status = future.wait_for(std::chrono::seconds(5));
        EXPECT_EQ(status, std::future_status::ready);
        auto results = future.get();
        EXPECT_EQ(results.size(), 1u);
    }

    PipelineStats stats = engine.get_stats();
    EXPECT_GE(stats.total_requests, 3u);
    engine.shutdown();
}

TEST_F(EngineFunctionalTest, GetStatsDuringOperation) {
    TestPipelineEngine engine(&enc_, &dec_, &lm_, &tok_, fast_cfg());

    // get_stats should be callable at any time without deadlock
    for (int i = 0; i < 5; ++i) {
        PipelineStats s = engine.get_stats();
        EXPECT_GE(s.total_requests, 0u);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    engine.shutdown();
}

// ============================================================================
// Special Token Constants Tests
// ============================================================================

TEST(SpecialTokensTest, BosTokenId) {
    EXPECT_EQ(adai::SpecialTokenIDs::BOS, 2);
}

TEST(SpecialTokensTest, EosTokenId) {
    EXPECT_EQ(adai::SpecialTokenIDs::EOS, 3);
}

TEST(SpecialTokensTest, PadTokenId) {
    EXPECT_EQ(adai::SpecialTokenIDs::PAD, 0);
}

TEST(SpecialTokensTest, BosLtEos) {
    EXPECT_LT(adai::SpecialTokenIDs::BOS, adai::SpecialTokenIDs::EOS);
}
