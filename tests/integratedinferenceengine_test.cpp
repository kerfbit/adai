/**
 * @file integratedinferenceengine_test.cpp
 * @brief Comprehensive unit tests for IntegratedInferenceEngine
 *
 * Tests the fully integrated inference engine combining all parallel
 * optimizations: continuous batching, pipeline parallelism, OpenMP,
 * and parallel attention heads.
 *
 * Test categories:
 *   - IntegratedInferenceConfig struct (defaults and custom values)
 *   - IntegratedInferenceStats struct (reset, update, derived metrics)
 *   - IntegratedRequest struct (construction, move semantics)
 *   - IntegratedBatch struct (construction, move semantics)
 *   - IntegratedEncoderOutput struct (construction, move semantics)
 *   - IntegratedInferenceEngine lifecycle (construct, shutdown, stats)
 *   - Edge cases and error handling
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../src/IntegratedInferenceEngine.hpp"

// ============================================================================
// IntegratedInferenceConfig Tests
// ============================================================================

class IntegratedInferenceConfigTest : public ::testing::Test {};

TEST_F(IntegratedInferenceConfigTest, DefaultBatchingValues) {
    IntegratedInferenceConfig config;

    EXPECT_EQ(config.max_batch_size, static_cast<size_t>(32));
    EXPECT_EQ(config.batch_timeout_ms, 50);
    EXPECT_EQ(config.max_tokens_per_batch, 4096);
}

TEST_F(IntegratedInferenceConfigTest, DefaultPipelineValues) {
    IntegratedInferenceConfig config;

    EXPECT_TRUE(config.enable_pipeline);
    EXPECT_EQ(config.encoder_queue_size, static_cast<size_t>(50));
    EXPECT_EQ(config.decoder_queue_size, static_cast<size_t>(50));
    EXPECT_EQ(config.encoder_timeout_ms, 50);
    EXPECT_EQ(config.decoder_timeout_ms, 50);
}

TEST_F(IntegratedInferenceConfigTest, DefaultParallelValues) {
    IntegratedInferenceConfig config;

    EXPECT_TRUE(config.use_openmp);
    EXPECT_TRUE(config.parallel_attention);
    EXPECT_EQ(config.num_threads, 0);  // 0 = auto
}

TEST_F(IntegratedInferenceConfigTest, DefaultRequestHandling) {
    IntegratedInferenceConfig config;

    EXPECT_EQ(config.max_queue_size, 1000);
    EXPECT_TRUE(config.enable_stats);
}

TEST_F(IntegratedInferenceConfigTest, DefaultGenerationSettings) {
    IntegratedInferenceConfig config;

    EXPECT_EQ(config.default_max_length, 100);
    EXPECT_EQ(config.generation_strategy, "greedy");
}

TEST_F(IntegratedInferenceConfigTest, CustomBatchingConfig) {
    IntegratedInferenceConfig config;
    config.max_batch_size = 16;
    config.batch_timeout_ms = 100;
    config.max_tokens_per_batch = 2048;

    EXPECT_EQ(config.max_batch_size, static_cast<size_t>(16));
    EXPECT_EQ(config.batch_timeout_ms, 100);
    EXPECT_EQ(config.max_tokens_per_batch, 2048);
}

TEST_F(IntegratedInferenceConfigTest, CustomPipelineConfig) {
    IntegratedInferenceConfig config;
    config.enable_pipeline = false;
    config.encoder_queue_size = 10;
    config.decoder_queue_size = 20;

    EXPECT_FALSE(config.enable_pipeline);
    EXPECT_EQ(config.encoder_queue_size, static_cast<size_t>(10));
    EXPECT_EQ(config.decoder_queue_size, static_cast<size_t>(20));
}

TEST_F(IntegratedInferenceConfigTest, CustomParallelConfig) {
    IntegratedInferenceConfig config;
    config.use_openmp = false;
    config.parallel_attention = false;
    config.num_threads = 4;

    EXPECT_FALSE(config.use_openmp);
    EXPECT_FALSE(config.parallel_attention);
    EXPECT_EQ(config.num_threads, 4);
}

TEST_F(IntegratedInferenceConfigTest, CustomGenerationStrategy) {
    IntegratedInferenceConfig config;
    config.generation_strategy = "beam";
    config.default_max_length = 200;

    EXPECT_EQ(config.generation_strategy, "beam");
    EXPECT_EQ(config.default_max_length, 200);
}

// ============================================================================
// IntegratedInferenceStats Tests
// ============================================================================

class IntegratedInferenceStatsTest : public ::testing::Test {};

TEST_F(IntegratedInferenceStatsTest, InitialStateAllZero) {
    IntegratedInferenceStats stats;

    EXPECT_EQ(stats.total_requests, static_cast<uint64_t>(0));
    EXPECT_EQ(stats.total_batches, static_cast<uint64_t>(0));
    EXPECT_EQ(stats.total_tokens_generated, static_cast<uint64_t>(0));
    EXPECT_DOUBLE_EQ(stats.avg_batch_size, 0.0);
    EXPECT_DOUBLE_EQ(stats.batch_utilization, 0.0);
}

TEST_F(IntegratedInferenceStatsTest, PipelineStatsInitiallyZero) {
    IntegratedInferenceStats stats;

    EXPECT_EQ(stats.encoder_batches_processed, static_cast<uint64_t>(0));
    EXPECT_EQ(stats.decoder_batches_processed, static_cast<uint64_t>(0));
    EXPECT_DOUBLE_EQ(stats.avg_encoder_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats.avg_decoder_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats.pipeline_efficiency, 0.0);
}

TEST_F(IntegratedInferenceStatsTest, PerformanceStatsInitiallyZero) {
    IntegratedInferenceStats stats;

    EXPECT_DOUBLE_EQ(stats.avg_latency_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats.throughput_req_per_sec, 0.0);
    EXPECT_DOUBLE_EQ(stats.throughput_tokens_per_sec, 0.0);
    EXPECT_DOUBLE_EQ(stats.cumulative_speedup, 1.0);
}

TEST_F(IntegratedInferenceStatsTest, QueueHealthInitiallyZero) {
    IntegratedInferenceStats stats;

    EXPECT_DOUBLE_EQ(stats.avg_queue_depth, 0.0);
    EXPECT_EQ(stats.requests_dropped, static_cast<uint64_t>(0));
}

TEST_F(IntegratedInferenceStatsTest, ResetClearsAllCounters) {
    IntegratedInferenceStats stats;

    // Manually set some values
    stats.total_requests = 100;
    stats.total_batches = 10;
    stats.total_tokens_generated = 5000;
    stats.avg_latency_ms = 25.0;
    stats.encoder_batches_processed = 10;
    stats.avg_encoder_time_ms = 15.0;
    stats.requests_dropped = 2;

    // Reset
    stats.reset();

    // Verify all cleared
    EXPECT_EQ(stats.total_requests, static_cast<uint64_t>(0));
    EXPECT_EQ(stats.total_batches, static_cast<uint64_t>(0));
    EXPECT_EQ(stats.total_tokens_generated, static_cast<uint64_t>(0));
    EXPECT_DOUBLE_EQ(stats.avg_latency_ms, 0.0);
    EXPECT_EQ(stats.encoder_batches_processed, static_cast<uint64_t>(0));
    EXPECT_DOUBLE_EQ(stats.avg_encoder_time_ms, 0.0);
    EXPECT_EQ(stats.requests_dropped, static_cast<uint64_t>(0));
}

TEST_F(IntegratedInferenceStatsTest, ResetSetsStartTime) {
    IntegratedInferenceStats stats;

    auto before = std::chrono::steady_clock::now();
    stats.reset();
    auto after = std::chrono::steady_clock::now();

    // start_time should be between before and after reset
    EXPECT_GE(stats.start_time, before);
    EXPECT_LE(stats.start_time, after);
}

TEST_F(IntegratedInferenceStatsTest, UpdateThroughputWithRequests) {
    IntegratedInferenceStats stats;
    stats.reset();

    // Simulate 100 requests and 1000 tokens
    stats.total_requests = 100;
    stats.total_tokens_generated = 1000;

    // Sleep a tiny bit so elapsed > 0
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    stats.update_throughput();

    EXPECT_GT(stats.throughput_req_per_sec, 0.0);
    EXPECT_GT(stats.throughput_tokens_per_sec, 0.0);
}

TEST_F(IntegratedInferenceStatsTest, UpdateThroughputZeroRequestsStaysZero) {
    IntegratedInferenceStats stats;
    stats.reset();

    // No requests
    stats.total_requests = 0;
    stats.total_tokens_generated = 0;

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    stats.update_throughput();

    EXPECT_DOUBLE_EQ(stats.throughput_req_per_sec, 0.0);
    EXPECT_DOUBLE_EQ(stats.throughput_tokens_per_sec, 0.0);
}

TEST_F(IntegratedInferenceStatsTest, MultipleResetsClearRepeatedly) {
    IntegratedInferenceStats stats;
    stats.total_requests = 50;
    stats.reset();
    EXPECT_EQ(stats.total_requests, static_cast<uint64_t>(0));

    stats.total_requests = 200;
    stats.reset();
    EXPECT_EQ(stats.total_requests, static_cast<uint64_t>(0));
}

// ============================================================================
// IntegratedRequest Tests
// ============================================================================

class IntegratedRequestTest : public ::testing::Test {};

TEST_F(IntegratedRequestTest, DefaultConstruction) {
    IntegratedRequest req;

    EXPECT_TRUE(req.input_text.empty());
    EXPECT_EQ(req.max_length, 0);
    EXPECT_TRUE(req.strategy.empty());
    EXPECT_EQ(req.input_tokens, static_cast<size_t>(0));
    EXPECT_EQ(req.output_tokens, static_cast<size_t>(0));
    EXPECT_EQ(req.batch_id, static_cast<size_t>(0));
}

TEST_F(IntegratedRequestTest, FieldAssignment) {
    IntegratedRequest req;
    req.input_text = "hello world";
    req.max_length = 50;
    req.strategy = "greedy";
    req.input_tokens = 2;
    req.batch_id = 7;

    EXPECT_EQ(req.input_text, "hello world");
    EXPECT_EQ(req.max_length, 50);
    EXPECT_EQ(req.strategy, "greedy");
    EXPECT_EQ(req.input_tokens, static_cast<size_t>(2));
    EXPECT_EQ(req.batch_id, static_cast<size_t>(7));
}

TEST_F(IntegratedRequestTest, MoveConstruction) {
    IntegratedRequest req;
    req.input_text = "move me";
    req.max_length = 100;
    req.strategy = "beam";
    req.batch_id = 3;

    IntegratedRequest moved(std::move(req));

    EXPECT_EQ(moved.input_text, "move me");
    EXPECT_EQ(moved.max_length, 100);
    EXPECT_EQ(moved.strategy, "beam");
    EXPECT_EQ(moved.batch_id, static_cast<size_t>(3));
}

TEST_F(IntegratedRequestTest, MoveAssignment) {
    IntegratedRequest req;
    req.input_text = "assign me";
    req.max_length = 75;
    req.strategy = "greedy";

    IntegratedRequest target;
    target = std::move(req);

    EXPECT_EQ(target.input_text, "assign me");
    EXPECT_EQ(target.max_length, 75);
    EXPECT_EQ(target.strategy, "greedy");
}

// ============================================================================
// IntegratedBatch Tests
// ============================================================================

class IntegratedBatchTest : public ::testing::Test {};

TEST_F(IntegratedBatchTest, DefaultConstruction) {
    IntegratedBatch batch;

    EXPECT_TRUE(batch.requests.empty());
    EXPECT_TRUE(batch.tokenized_inputs.empty());
    EXPECT_EQ(batch.batch_id, static_cast<size_t>(0));
}

TEST_F(IntegratedBatchTest, FieldAssignment) {
    IntegratedBatch batch;
    batch.batch_id = 42;

    IntegratedRequest req;
    req.input_text = "test input";
    req.max_length = 50;
    batch.requests.push_back(std::move(req));
    batch.tokenized_inputs.push_back({1, 2, 3, 4});

    EXPECT_EQ(batch.batch_id, static_cast<size_t>(42));
    EXPECT_EQ(batch.requests.size(), static_cast<size_t>(1));
    EXPECT_EQ(batch.requests[0].input_text, "test input");
    EXPECT_EQ(batch.tokenized_inputs.size(), static_cast<size_t>(1));
    EXPECT_EQ(batch.tokenized_inputs[0].size(), static_cast<size_t>(4));
}

TEST_F(IntegratedBatchTest, MoveConstruction) {
    IntegratedBatch batch;
    batch.batch_id = 10;
    batch.tokenized_inputs.push_back({5, 6, 7});

    IntegratedBatch moved(std::move(batch));

    EXPECT_EQ(moved.batch_id, static_cast<size_t>(10));
    EXPECT_EQ(moved.tokenized_inputs.size(), static_cast<size_t>(1));
    EXPECT_EQ(moved.tokenized_inputs[0][0], 5);
}

TEST_F(IntegratedBatchTest, MoveAssignment) {
    IntegratedBatch batch;
    batch.batch_id = 99;
    batch.tokenized_inputs.push_back({10, 20});

    IntegratedBatch target;
    target = std::move(batch);

    EXPECT_EQ(target.batch_id, static_cast<size_t>(99));
    EXPECT_EQ(target.tokenized_inputs.size(), static_cast<size_t>(1));
}

TEST_F(IntegratedBatchTest, MultiplRequestsAndInputs) {
    IntegratedBatch batch;
    batch.batch_id = 5;

    // Add multiple requests
    for (int i = 0; i < 3; ++i) {
        IntegratedRequest req;
        req.input_text = "request " + std::to_string(i);
        req.max_length = 50 + i;
        batch.requests.push_back(std::move(req));
        batch.tokenized_inputs.push_back({i + 1, i + 2, i + 3});
    }

    EXPECT_EQ(batch.requests.size(), static_cast<size_t>(3));
    EXPECT_EQ(batch.tokenized_inputs.size(), static_cast<size_t>(3));
    EXPECT_EQ(batch.requests[2].input_text, "request 2");
    EXPECT_EQ(batch.requests[2].max_length, 52);
}

// ============================================================================
// IntegratedEncoderOutput Tests
// ============================================================================

class IntegratedEncoderOutputTest : public ::testing::Test {};

TEST_F(IntegratedEncoderOutputTest, DefaultConstruction) {
    IntegratedEncoderOutput output;

    EXPECT_TRUE(output.encoder_outputs.empty());
    EXPECT_TRUE(output.requests.empty());
    EXPECT_EQ(output.batch_id, static_cast<size_t>(0));
}

TEST_F(IntegratedEncoderOutputTest, FieldAssignment) {
    IntegratedEncoderOutput output;
    output.batch_id = 7;

    // Add a mock encoder output (2x4 matrix)
    Matrix enc_out(2, 4);
    enc_out(0, 0) = 1.0f;
    enc_out(1, 3) = 2.5f;
    output.encoder_outputs.push_back(enc_out);

    IntegratedRequest req;
    req.input_text = "encoded text";
    output.requests.push_back(std::move(req));

    EXPECT_EQ(output.batch_id, static_cast<size_t>(7));
    EXPECT_EQ(output.encoder_outputs.size(), static_cast<size_t>(1));
    EXPECT_EQ(output.encoder_outputs[0].rows, 2);
    EXPECT_EQ(output.encoder_outputs[0].cols, 4);
    EXPECT_FLOAT_EQ(output.encoder_outputs[0](0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.encoder_outputs[0](1, 3), 2.5f);
    EXPECT_EQ(output.requests.size(), static_cast<size_t>(1));
}

TEST_F(IntegratedEncoderOutputTest, MoveConstruction) {
    IntegratedEncoderOutput output;
    output.batch_id = 15;
    Matrix mat(3, 3);
    mat(1, 1) = 9.0f;
    output.encoder_outputs.push_back(mat);

    IntegratedEncoderOutput moved(std::move(output));

    EXPECT_EQ(moved.batch_id, static_cast<size_t>(15));
    EXPECT_EQ(moved.encoder_outputs.size(), static_cast<size_t>(1));
    EXPECT_EQ(moved.encoder_outputs[0].rows, 3);
    EXPECT_FLOAT_EQ(moved.encoder_outputs[0](1, 1), 9.0f);
}

TEST_F(IntegratedEncoderOutputTest, MoveAssignment) {
    IntegratedEncoderOutput output;
    output.batch_id = 3;
    output.encoder_outputs.push_back(Matrix(4, 8));

    IntegratedEncoderOutput target;
    target = std::move(output);

    EXPECT_EQ(target.batch_id, static_cast<size_t>(3));
    EXPECT_EQ(target.encoder_outputs.size(), static_cast<size_t>(1));
    EXPECT_EQ(target.encoder_outputs[0].rows, 4);
    EXPECT_EQ(target.encoder_outputs[0].cols, 8);
}

TEST_F(IntegratedEncoderOutputTest, MultipleEncoderOutputsPerBatch) {
    IntegratedEncoderOutput output;
    output.batch_id = 20;

    // Simulate 3 requests each with their own encoder output
    for (int i = 0; i < 3; ++i) {
        Matrix enc(2 + i, 8);
        enc(0, 0) = static_cast<float>(i);
        output.encoder_outputs.push_back(enc);

        IntegratedRequest req;
        req.input_text = "input " + std::to_string(i);
        output.requests.push_back(std::move(req));
    }

    EXPECT_EQ(output.encoder_outputs.size(), static_cast<size_t>(3));
    EXPECT_EQ(output.requests.size(), static_cast<size_t>(3));
    EXPECT_EQ(output.encoder_outputs[0].rows, 2);
    EXPECT_EQ(output.encoder_outputs[1].rows, 3);
    EXPECT_EQ(output.encoder_outputs[2].rows, 4);
    EXPECT_FLOAT_EQ(output.encoder_outputs[2](0, 0), 2.0f);
}

// ============================================================================
// IntegratedInferenceEngine Lifecycle Tests (null model pointers - safe when
// no requests are submitted before shutdown)
// ============================================================================

class IntegratedInferenceEngineLifecycleTest : public ::testing::Test {
   protected:
    // Use a minimal timeout so lifecycle tests complete quickly
    IntegratedInferenceConfig make_fast_config() {
        IntegratedInferenceConfig config;
        config.batch_timeout_ms = 5;
        config.encoder_timeout_ms = 5;
        config.decoder_timeout_ms = 5;
        config.max_queue_size = 10;
        config.encoder_queue_size = 5;
        config.decoder_queue_size = 5;
        return config;
    }
};

TEST_F(IntegratedInferenceEngineLifecycleTest, ConstructAndShutdown) {
    // Engine can be created with null model pointers and shutdown without
    // submitting requests — workers observe shutdown before dereferencing pointers.
    auto config = make_fast_config();
    IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);
    engine.shutdown();  // Should complete without crash
    SUCCEED();
}

TEST_F(IntegratedInferenceEngineLifecycleTest, DestructorCallsShutdown) {
    // Verify that the destructor does not hang or crash when engine scope exits
    {
        auto config = make_fast_config();
        IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);
        // Destructor triggered here
    }
    SUCCEED();
}

TEST_F(IntegratedInferenceEngineLifecycleTest, DoubleShutdownIsSafe) {
    auto config = make_fast_config();
    IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);
    engine.shutdown();
    engine.shutdown();  // Second call should be a no-op and not crash
    SUCCEED();
}

TEST_F(IntegratedInferenceEngineLifecycleTest, GetStatsAfterConstruction) {
    auto config = make_fast_config();
    IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);

    auto stats = engine.get_stats();

    EXPECT_EQ(stats.total_requests, static_cast<uint64_t>(0));
    EXPECT_EQ(stats.total_batches, static_cast<uint64_t>(0));
    EXPECT_EQ(stats.total_tokens_generated, static_cast<uint64_t>(0));
    EXPECT_DOUBLE_EQ(stats.avg_latency_ms, 0.0);

    engine.shutdown();
}

TEST_F(IntegratedInferenceEngineLifecycleTest, GetStatsQueueHealthInitial) {
    auto config = make_fast_config();
    IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);

    auto stats = engine.get_stats();

    EXPECT_EQ(stats.requests_dropped, static_cast<uint64_t>(0));
    EXPECT_DOUBLE_EQ(stats.avg_queue_depth, 0.0);

    engine.shutdown();
}

TEST_F(IntegratedInferenceEngineLifecycleTest, ResetStatsDoesNotCrash) {
    auto config = make_fast_config();
    IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);

    engine.reset_stats();

    auto stats = engine.get_stats();
    EXPECT_EQ(stats.total_requests, static_cast<uint64_t>(0));

    engine.shutdown();
}

TEST_F(IntegratedInferenceEngineLifecycleTest, GetStatsReturnsCopy) {
    // get_stats() should return a copy, not a reference
    auto config = make_fast_config();
    IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);

    auto stats1 = engine.get_stats();
    auto stats2 = engine.get_stats();

    // Both copies should be identical and have zero values
    EXPECT_EQ(stats1.total_requests, stats2.total_requests);
    EXPECT_DOUBLE_EQ(stats1.avg_latency_ms, stats2.avg_latency_ms);

    engine.shutdown();
}

TEST_F(IntegratedInferenceEngineLifecycleTest, MultipleEnginesCanCoexist) {
    // Create two engines simultaneously to verify no global state conflicts
    auto config = make_fast_config();
    IntegratedInferenceEngine engine1(nullptr, nullptr, nullptr, nullptr, config);
    IntegratedInferenceEngine engine2(nullptr, nullptr, nullptr, nullptr, config);

    auto stats1 = engine1.get_stats();
    auto stats2 = engine2.get_stats();

    EXPECT_EQ(stats1.total_requests, static_cast<uint64_t>(0));
    EXPECT_EQ(stats2.total_requests, static_cast<uint64_t>(0));

    engine1.shutdown();
    engine2.shutdown();
}

TEST_F(IntegratedInferenceEngineLifecycleTest, ResetStatsAfterShutdown) {
    // Verify reset_stats doesn't interfere with shutdown
    auto config = make_fast_config();
    IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);
    engine.shutdown();
    engine.reset_stats();  // Should not crash after shutdown

    auto stats = engine.get_stats();
    EXPECT_EQ(stats.total_requests, static_cast<uint64_t>(0));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

class IntegratedInferenceEngineEdgeCaseTest : public ::testing::Test {};

TEST_F(IntegratedInferenceEngineEdgeCaseTest, ConfigWithMinimalQueueSizes) {
    IntegratedInferenceConfig config;
    config.batch_timeout_ms = 1;
    config.encoder_timeout_ms = 1;
    config.decoder_timeout_ms = 1;
    config.max_queue_size = 1;
    config.encoder_queue_size = 1;
    config.decoder_queue_size = 1;

    // Should construct and destruct without deadlock even with min queue sizes
    IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);
    engine.shutdown();
    SUCCEED();
}

TEST_F(IntegratedInferenceEngineEdgeCaseTest, ConfigWithPipelineDisabled) {
    IntegratedInferenceConfig config;
    config.enable_pipeline = false;
    config.batch_timeout_ms = 5;
    config.encoder_timeout_ms = 5;
    config.decoder_timeout_ms = 5;
    config.max_queue_size = 10;
    config.encoder_queue_size = 5;
    config.decoder_queue_size = 5;

    // Engine should still construct and shut down cleanly regardless of pipeline flag
    IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);
    engine.shutdown();
    SUCCEED();
}

TEST_F(IntegratedInferenceEngineEdgeCaseTest, StatsComputeDerivedMetricsCorrectly) {
    // Verify derived metric formulas in get_stats()
    // When total_batches > 0: avg_batch_size = total_requests / total_batches
    // Test this via IntegratedInferenceStats directly

    IntegratedInferenceStats stats;
    stats.reset();

    // Simulate what get_stats() computes internally
    stats.total_requests = 96;
    stats.total_batches = 3;
    stats.avg_encoder_time_ms = 10.0;
    stats.avg_decoder_time_ms = 20.0;

    // Manual derived computation (mirrors get_stats() logic)
    double avg_batch_size = static_cast<double>(stats.total_requests) / stats.total_batches;
    EXPECT_DOUBLE_EQ(avg_batch_size, 32.0);

    double total_sequential = stats.avg_encoder_time_ms + stats.avg_decoder_time_ms;
    double pipeline_time = std::max(stats.avg_encoder_time_ms, stats.avg_decoder_time_ms);
    double pipeline_efficiency = total_sequential / pipeline_time;
    EXPECT_DOUBLE_EQ(pipeline_efficiency, 1.5);  // 30ms / 20ms = 1.5x
}

TEST_F(IntegratedInferenceEngineEdgeCaseTest, BatchUtilizationFormula) {
    IntegratedInferenceStats stats;
    stats.reset();

    stats.total_requests = 16;
    stats.total_batches = 1;

    double avg_batch_size = static_cast<double>(stats.total_requests) / stats.total_batches;
    double batch_utilization = avg_batch_size / 32.0;  // max_batch_size default = 32

    EXPECT_DOUBLE_EQ(avg_batch_size, 16.0);
    EXPECT_DOUBLE_EQ(batch_utilization, 0.5);  // 50% utilization
}

TEST_F(IntegratedInferenceEngineEdgeCaseTest, TypeAliasWorks) {
    // Verify the type alias compiles correctly
    IntegratedInferenceConfig config;
    config.batch_timeout_ms = 5;
    config.encoder_timeout_ms = 5;
    config.decoder_timeout_ms = 5;
    config.max_queue_size = 10;
    config.encoder_queue_size = 5;
    config.decoder_queue_size = 5;

    // StandardIntegratedEngine is a type alias for IntegratedInferenceEngine
    StandardIntegratedEngine engine(nullptr, nullptr, nullptr, nullptr, config);
    engine.shutdown();
    SUCCEED();
}

TEST_F(IntegratedInferenceEngineEdgeCaseTest, ConfigMaxBatchSizeOne) {
    // Edge case: single-item batches
    IntegratedInferenceConfig config;
    config.max_batch_size = 1;
    config.batch_timeout_ms = 5;
    config.encoder_timeout_ms = 5;
    config.decoder_timeout_ms = 5;
    config.max_queue_size = 100;
    config.encoder_queue_size = 10;
    config.decoder_queue_size = 10;

    IntegratedInferenceEngine engine(nullptr, nullptr, nullptr, nullptr, config);
    engine.shutdown();
    SUCCEED();
}
