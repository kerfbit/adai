/**
 * Unit Tests for Inference Optimizations
 *
 * Tests KV cache correctness, batch processing, and performance profiling utilities.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>
#include "BatchProcessor.hpp"
#include "Decoder.hpp"
#include "KVCache.hpp"
#include "Matrix.hpp"
#include "MultiHeadAttention.hpp"
#include "PerformanceProfiler.hpp"

// ============================================================================
// KVCache Tests
// ============================================================================

class KVCacheTest : public ::testing::Test {
   protected:
    void SetUp() override {}
};

TEST_F(KVCacheTest, InitiallyEmpty) {
    KVCache cache;
    EXPECT_TRUE(cache.is_empty());
    EXPECT_EQ(cache.size(), 0);
    EXPECT_EQ(cache.current_length, 0);
}

TEST_F(KVCacheTest, AppendSingleEntry) {
    KVCache cache;

    Matrix keys(1, 64);
    Matrix values(1, 64);

    // Fill with test data
    for (int i = 0; i < 64; ++i) {
        keys(0, i) = i * 1.0f;
        values(0, i) = i * 2.0f;
    }

    cache.append(keys, values);

    EXPECT_FALSE(cache.is_empty());
    EXPECT_EQ(cache.size(), 1);
    EXPECT_EQ(cache.current_length, 1);

    // Verify data
    const Matrix& cached_keys = cache.get_keys();
    const Matrix& cached_values = cache.get_values();

    EXPECT_EQ(cached_keys.rows, 1);
    EXPECT_EQ(cached_keys.cols, 64);
    EXPECT_FLOAT_EQ(cached_keys(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(cached_keys(0, 63), 63.0f);

    EXPECT_EQ(cached_values.rows, 1);
    EXPECT_FLOAT_EQ(cached_values(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(cached_values(0, 63), 126.0f);
}

TEST_F(KVCacheTest, AppendMultipleEntries) {
    KVCache cache;

    // First append
    Matrix keys1(2, 64);
    Matrix values1(2, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            keys1(i, j) = i * 100.0f + j;
            values1(i, j) = i * 200.0f + j;
        }
    }
    cache.append(keys1, values1);

    EXPECT_EQ(cache.size(), 2);

    // Second append
    Matrix keys2(3, 64);
    Matrix values2(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            keys2(i, j) = (i + 2) * 100.0f + j;
            values2(i, j) = (i + 2) * 200.0f + j;
        }
    }
    cache.append(keys2, values2);

    EXPECT_EQ(cache.size(), 5);

    // Verify all data preserved
    const Matrix& cached_keys = cache.get_keys();
    const Matrix& cached_values = cache.get_values();

    EXPECT_EQ(cached_keys.rows, 5);
    EXPECT_FLOAT_EQ(cached_keys(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(cached_keys(1, 0), 100.0f);
    EXPECT_FLOAT_EQ(cached_keys(2, 0), 200.0f);
    EXPECT_FLOAT_EQ(cached_keys(4, 0), 400.0f);
}

TEST_F(KVCacheTest, ClearCache) {
    KVCache cache;

    Matrix keys(5, 64);
    Matrix values(5, 64);
    cache.append(keys, values);

    EXPECT_FALSE(cache.is_empty());
    EXPECT_EQ(cache.size(), 5);

    cache.clear();

    EXPECT_TRUE(cache.is_empty());
    EXPECT_EQ(cache.size(), 0);
}

TEST_F(KVCacheTest, DecoderKVCacheMultipleLayers) {
    DecoderKVCache cache(4);  // 4 layers

    EXPECT_TRUE(cache.is_empty());
    EXPECT_EQ(cache.current_length(), 0);

    // Add to layer 0
    Matrix keys(2, 64);
    Matrix values(2, 64);
    cache.get_self_attention_cache(0).append(keys, values);

    EXPECT_FALSE(cache.is_empty());
    EXPECT_EQ(cache.current_length(), 2);

    // Layer 1 should still be empty
    EXPECT_TRUE(cache.get_self_attention_cache(1).is_empty());
}

TEST_F(KVCacheTest, DecoderKVCacheClearSelfAttention) {
    DecoderKVCache cache(2);

    Matrix keys(3, 64);
    Matrix values(3, 64);

    cache.get_self_attention_cache(0).append(keys, values);
    cache.get_cross_attention_cache(0).append(keys, values);

    cache.clear_self_attention();

    EXPECT_TRUE(cache.get_self_attention_cache(0).is_empty());
    EXPECT_FALSE(cache.get_cross_attention_cache(0).is_empty());
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

class BatchProcessorTest : public ::testing::Test {
   protected:
    void SetUp() override {}
};

TEST_F(BatchProcessorTest, CreateSimpleBatch) {
    std::vector<std::vector<int>> sequences = {{1, 2, 3}, {4, 5, 6, 7}, {8, 9}};

    TokenBatch batch = create_batch(sequences, 0);

    EXPECT_EQ(batch.batch_size(), 3);
    EXPECT_EQ(batch.max_length, 4);  // Longest sequence
    EXPECT_EQ(batch.pad_token_id, 0);

    // Check padding
    EXPECT_EQ(batch.batch_token_ids[0].size(), 4);
    EXPECT_EQ(batch.batch_token_ids[1].size(), 4);
    EXPECT_EQ(batch.batch_token_ids[2].size(), 4);

    // Check lengths
    EXPECT_EQ(batch.lengths[0], 3);
    EXPECT_EQ(batch.lengths[1], 4);
    EXPECT_EQ(batch.lengths[2], 2);

    // Check padding values
    EXPECT_EQ(batch.batch_token_ids[0][3], 0);  // Padded
    EXPECT_EQ(batch.batch_token_ids[2][2], 0);  // Padded
    EXPECT_EQ(batch.batch_token_ids[2][3], 0);  // Padded
}

TEST_F(BatchProcessorTest, CreatePaddingMask) {
    std::vector<std::vector<int>> sequences = {{1, 2, 3}, {4, 5}};

    TokenBatch batch = create_batch(sequences, 0);
    Matrix mask = create_padding_mask(batch);

    EXPECT_EQ(mask.rows, 2);
    EXPECT_EQ(mask.cols, 3);

    // First sequence: all real tokens
    EXPECT_FLOAT_EQ(mask(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(mask(0, 1), 1.0f);
    EXPECT_FLOAT_EQ(mask(0, 2), 1.0f);

    // Second sequence: 2 real tokens, 1 padding
    EXPECT_FLOAT_EQ(mask(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(mask(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(mask(1, 2), 0.0f);  // Padding
}

TEST_F(BatchProcessorTest, DynamicBatching) {
    std::vector<std::vector<int>> sequences;

    // Create sequences of varying lengths
    for (int i = 0; i < 10; ++i) {
        std::vector<int> seq;
        int length = 5 + (i * 3);  // 5, 8, 11, 14, ...
        for (int j = 0; j < length; ++j) {
            seq.push_back(j);
        }
        sequences.push_back(seq);
    }

    auto batches = create_dynamic_batches(sequences, 4, 5, 0);

    // Should create multiple batches
    EXPECT_GT(batches.size(), 1);

    // Each batch should respect max_batch_size
    for (const auto& batch : batches) {
        EXPECT_LE(batch.batch_size(), 4);
    }

    // Total sequences preserved
    int total_sequences = 0;
    for (const auto& batch : batches) {
        total_sequences += batch.batch_size();
    }
    EXPECT_EQ(total_sequences, 10);
}

TEST_F(BatchProcessorTest, BatchStatistics) {
    std::vector<std::vector<int>> sequences = {
        {1, 2, 3, 4, 5},       // 5 tokens
        {1, 2, 3},             // 3 tokens
        {1, 2, 3, 4, 5, 6, 7}  // 7 tokens
    };

    auto batches = create_dynamic_batches(sequences, 4, 10, 0);
    BatchStats stats = compute_batch_stats(batches);

    EXPECT_EQ(stats.actual_tokens, 15);  // 5 + 3 + 7
    EXPECT_EQ(stats.total_tokens, 21);   // 3 * 7 (all padded to max length 7)
    EXPECT_GT(stats.padding_ratio, 0.0f);
    EXPECT_LT(stats.padding_ratio, 1.0f);
}

TEST_F(BatchProcessorTest, EmptyBatch) {
    std::vector<std::vector<int>> sequences;
    TokenBatch batch = create_batch(sequences, 0);

    EXPECT_TRUE(batch.is_empty());
    EXPECT_EQ(batch.batch_size(), 0);
    EXPECT_EQ(batch.max_length, 0);
}

// ============================================================================
// Performance Profiler Tests
// ============================================================================

class PerformanceProfilerTest : public ::testing::Test {
   protected:
    void SetUp() override {}
};

TEST_F(PerformanceProfilerTest, TimerBasic) {
    Timer timer;

    timer.start();
    // Simulate some work
    volatile int sum = 0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
    double elapsed = timer.stop();

    EXPECT_GT(elapsed, 0.0);
    EXPECT_LT(elapsed, 1000.0);  // Should be < 1 second
}

TEST_F(PerformanceProfilerTest, TimerElapsed) {
    Timer timer;
    timer.start();

    double e1 = timer.elapsed();
    EXPECT_GE(e1, 0.0);

    // Wait a bit
    volatile int sum = 0;
    for (int i = 0; i < 100000; ++i) {
        sum += i;
    }

    double e2 = timer.elapsed();
    EXPECT_GT(e2, e1);
}

TEST_F(PerformanceProfilerTest, ProfileStatsBasic) {
    ProfileStats stats("test");

    stats.add_timing(10.0);
    stats.add_timing(20.0);
    stats.add_timing(15.0);

    EXPECT_EQ(stats.call_count, 3);
    EXPECT_FLOAT_EQ(stats.total_time, 45.0f);
    EXPECT_FLOAT_EQ(stats.mean_time, 15.0f);
    EXPECT_FLOAT_EQ(stats.min_time, 10.0f);
    EXPECT_FLOAT_EQ(stats.max_time, 20.0f);
}

TEST_F(PerformanceProfilerTest, ProfileStatsMedian) {
    ProfileStats stats("test");

    stats.add_timing(10.0);
    stats.add_timing(50.0);
    stats.add_timing(20.0);
    stats.add_timing(30.0);
    stats.add_timing(40.0);

    stats.compute_median();

    EXPECT_FLOAT_EQ(stats.median_time, 30.0f);
}

TEST_F(PerformanceProfilerTest, ProfileStatsPercentile) {
    ProfileStats stats("test");

    for (int i = 1; i <= 100; ++i) {
        stats.add_timing(static_cast<double>(i));
    }

    EXPECT_NEAR(stats.get_percentile(50.0), 50.0, 2.0);  // Median
    EXPECT_NEAR(stats.get_percentile(95.0), 95.0, 2.0);  // P95
    EXPECT_NEAR(stats.get_percentile(99.0), 99.0, 2.0);  // P99
}

TEST_F(PerformanceProfilerTest, ProfilerMultipleSections) {
    Profiler profiler;

    profiler.start("section1");
    volatile int sum = 0;
    for (int i = 0; i < 100000; ++i)
        sum += i;
    profiler.stop("section1");

    profiler.start("section2");
    for (int i = 0; i < 200000; ++i)
        sum += i;
    profiler.stop("section2");

    ProfileStats s1 = profiler.get_stats("section1");
    ProfileStats s2 = profiler.get_stats("section2");

    EXPECT_EQ(s1.call_count, 1);
    EXPECT_EQ(s2.call_count, 1);
    EXPECT_GT(s2.mean_time, 0.0);
}

// TD-097 regression: PROFILE_SCOPE's "guard" used to be bound to the return
// value of an immediately-invoked lambda that called stop() right there on
// the same line — before a single instruction of the profiled block below it
// ran. Put real, non-trivial work *after* the macro invocation, inside the
// same scope: pre-fix, that work was never actually timed (stop() had
// already run), so the recorded duration was ~0ms regardless of the busy
// loop's size. This call shape (a string-literal section name) also failed
// to compile at all against the pre-fix macro's `##name` token-paste.
TEST_F(PerformanceProfilerTest, ProfileScopeMacroTimesTheWholeBlockNotJustItsOwnLine) {
    Profiler profiler;

    {
        PROFILE_SCOPE(profiler, "scoped_section");
        volatile long sum = 0;
        for (int i = 0; i < 20000000; ++i) {
            sum += i;
        }
    }

    ProfileStats stats = profiler.get_stats("scoped_section");
    ASSERT_EQ(stats.call_count, 1);
    EXPECT_GT(stats.total_time, 0.01)
        << "PROFILE_SCOPE recorded near-zero time despite a large busy-loop inside its "
           "scope — stop() likely ran immediately instead of at scope exit";
}

// TD-038 follow-up: ChatbotAPI::generate_response() unconditionally wraps itself in
// PROFILE_SCOPE(profiler_, "generate_response"), and chatbot_api_server serves
// concurrent requests via httplib's real thread pool. Profiler::active_timers used to be
// keyed by section name ALONE, so two threads concurrently timing the *same* name shared
// one Timer instance: a later start() overwrote an earlier one's start_time, and once one
// thread's stop() stopped that shared Timer, the other thread's later stop() found it
// already stopped and silently recorded 0ms instead of its real elapsed time — a wrong
// measurement, not merely a data race. Fixed by keying active_timers on
// (name, thread::id). This test forces a specific, deterministic interleaving (via
// synchronization flags, not scheduling luck) rather than hoping a race manifests:
// thread_a starts "shared_section", thread_b starts the SAME name ~100ms later and
// finishes quickly, then thread_a finally stops. Pre-fix, thread_a's recorded timing
// comes out as ~0ms; post-fix it reflects its own real ~130ms span.
TEST_F(PerformanceProfilerTest, ConcurrentSameSectionNameFromDifferentThreadsRecordsIndependentTimings) {
    Profiler profiler;
    std::atomic<bool> a_started{false};
    std::atomic<bool> b_finished{false};

    std::thread thread_a([&]() {
        profiler.start("shared_section");
        a_started = true;
        while (!b_finished.load()) {
            std::this_thread::yield();
        }
        profiler.stop("shared_section");
    });

    while (!a_started.load()) {
        std::this_thread::yield();
    }
    // Let thread_a's timer "run" for a while before thread_b starts the same named
    // section concurrently — a large, reliable gap (not dependent on scheduler luck).
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::thread thread_b([&]() {
        profiler.start("shared_section");
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        profiler.stop("shared_section");
        b_finished = true;
    });

    thread_a.join();
    thread_b.join();

    ProfileStats stats = profiler.get_stats("shared_section");
    ASSERT_EQ(stats.call_count, 2);
    EXPECT_GT(stats.min_time, 10.0)
        << "one of the two concurrent same-name timings was recorded as ~0ms — "
           "active_timers is not correctly isolated per thread";
    EXPECT_GT(stats.max_time, 90.0)
        << "the longer-running thread's timing was corrupted by the other thread's "
           "concurrent start() call on the same section name";
}

// ============================================================================
// Integration Tests
// ============================================================================

class InferenceOptimizationIntegrationTest : public ::testing::Test {
   protected:
    void SetUp() override {}
};

TEST_F(InferenceOptimizationIntegrationTest, MultiHeadAttentionWithCache) {
    int d_model = 64;
    int num_heads = 4;
    MultiHeadAttention attn(d_model, num_heads);

    // First forward pass - cache empty
    Matrix input1(3, d_model);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < d_model; ++j) {
            input1(i, j) = (i * d_model + j) * 0.01f;
        }
    }

    KVCache cache;
    Matrix output1 = attn.forward_with_cache(input1, nullptr, &cache, true);

    EXPECT_EQ(output1.rows, 3);
    EXPECT_EQ(output1.cols, d_model);
    EXPECT_EQ(cache.size(), 3);

    // Second forward pass - cache has 3 tokens, add 1 more
    Matrix input2(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        input2(0, j) = (3 * d_model + j) * 0.01f;
    }

    Matrix output2 = attn.forward_with_cache(input2, nullptr, &cache, true);

    EXPECT_EQ(output2.rows, 1);
    EXPECT_EQ(output2.cols, d_model);
    EXPECT_EQ(cache.size(), 4);
}

TEST_F(InferenceOptimizationIntegrationTest, DecoderWithCacheBasic) {
    int vocab_size = 100;
    LLMDecoder decoder(vocab_size, 64, 2, 4, 256, 128);

    // Initial tokens
    std::vector<int> initial_tokens = {1, 2, 3};
    DecoderKVCache cache(2);  // 2 layers

    Matrix output1 = decoder.forward_with_cache(initial_tokens, cache, nullptr, true);

    EXPECT_EQ(output1.rows, 3);
    EXPECT_EQ(output1.cols, 64);
    EXPECT_EQ(cache.current_length(), 3);

    // Add one more token
    std::vector<int> new_token = {4};
    Matrix output2 = decoder.forward_with_cache(new_token, cache, nullptr, true);

    EXPECT_EQ(output2.rows, 1);
    EXPECT_EQ(output2.cols, 64);
    EXPECT_EQ(cache.current_length(), 4);
}

// TD-050 (root-caused September 14, 2026): the real diagnostic this item's own tracker entry
// called for — "an actual multi-step incremental-decode-vs-single-shot-full-recompute numerical
// comparison with identical weights (the only way to confirm the bug is still live at all, and if
// so, localize which step first diverges)". The DISABLED_CacheOutputConsistency test this
// replaces did NOT actually test this: it called forward_with_cache() exactly ONCE with all 5
// tokens at once (never exercising forward_with_cache()'s own documented "subsequent calls: only
// computes new token, reuses cache" code path — the entire point of the cache, and the exact call
// pattern EncoderDecoderModel::generate_response_with_strategy() actually uses), and passed
// encoder_output=nullptr (skipping cross-attention entirely, unlike every real production call).
//
// This test reproduces the real incremental call pattern instead: one new token per
// forward_with_cache() call, the same DecoderKVCache instance threaded across every step, and a
// real (non-null) encoder_output so cross-attention is exercised too — then compares each step's
// predicted-next-token hidden state against a from-scratch forward_with_encoder() recompute of
// the identical growing prefix, using the same decoder weights for both.
//
// Result across 40 steps (d_model=128, 4 layers, 8 heads): max abs diff stays flat at
// ~1e-6-2e-6 for every single step, with no growth or accumulation trend as the sequence grows —
// exactly the signature of ordinary float32 summation-order rounding noise, not an algorithmic
// indexing/masking bug (a real bug would show either an immediate large divergence or unbounded
// growth with sequence length). The disabled test's "different order of operations" explanation
// for its own mismatch turns out to have been substantively correct, just never actually verified
// against a test that could tell the difference (it used a single non-incremental call with no
// cross-attention and a 0.5 tolerance loose enough to hide a real bug too) — the same kind of
// unverified-but-plausible-sounding excuse this codebase has repeatedly found masking a genuine
// bug once actually checked (see TD-059, LoRA's merge_with_base()), except this time the
// investigation clears it instead. See EncoderDecoderModel.cpp's now-updated TD-050 comment: the
// greedy-decoding workaround this finding made unnecessary has been removed.
TEST_F(InferenceOptimizationIntegrationTest, IncrementalCacheMatchesFullRecomputePerStep) {
    int vocab_size = 200;
    int d_model = 128;
    int num_layers = 4;
    int num_heads = 8;
    int d_ff = 512;
    int max_seq_length = 256;
    LLMDecoder decoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length);

    // A real, fixed, non-trivial encoder output so cross-attention is genuinely exercised in
    // both paths (matching production, unlike the old test's encoder_output=nullptr).
    int encoder_seq_len = 12;
    Matrix encoder_output(encoder_seq_len, d_model);
    for (int i = 0; i < encoder_seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = std::sin(static_cast<float>(i * d_model + j) * 0.013f);
        }
    }

    std::vector<int> tokens;
    for (int i = 0; i < 40; ++i) {
        tokens.push_back((i * 37 + 5) % vocab_size);
    }

    // Full-recompute reference: at each step t, forward_with_encoder() on tokens[0..t] entirely
    // from scratch (no cache at all) — this is the same primitive
    // generate_response_with_strategy()'s greedy branch already uses as its own workaround, so
    // it's independently exercised/trusted code, not something new being validated here.
    std::vector<std::vector<float>> full_recompute_last_rows;
    for (size_t t = 1; t <= tokens.size(); ++t) {
        std::vector<int> prefix(tokens.begin(), tokens.begin() + static_cast<long>(t));
        Matrix out = decoder.forward_with_encoder(prefix, encoder_output);
        int last = out.rows - 1;
        std::vector<float> row(d_model);
        for (int j = 0; j < d_model; ++j) {
            row[j] = out(last, j);
        }
        full_recompute_last_rows.push_back(std::move(row));
    }

    // Incremental path: production's own exact call pattern — one new token per call, one
    // shared DecoderKVCache growing across every step.
    DecoderKVCache kv_cache(num_layers);
    std::vector<std::vector<float>> incremental_last_rows;
    for (size_t t = 0; t < tokens.size(); ++t) {
        std::vector<int> new_tokens = {tokens[t]};
        Matrix out = decoder.forward_with_cache(new_tokens, kv_cache, &encoder_output, true);
        ASSERT_EQ(out.rows, 1);
        std::vector<float> row(d_model);
        for (int j = 0; j < d_model; ++j) {
            row[j] = out(0, j);
        }
        incremental_last_rows.push_back(std::move(row));
    }

    ASSERT_EQ(full_recompute_last_rows.size(), incremental_last_rows.size());

    // Report the first step where the two paths genuinely diverge (beyond ordinary
    // floating-point accumulation noise), rather than just failing on the first mismatched
    // element — localizing *which* step first goes wrong is exactly what this item's own
    // action item asks for.
    int first_divergent_step = -1;
    float max_diff_at_first_divergence = 0.0f;
    const float divergence_threshold = 1e-3f;
    for (size_t t = 0; t < tokens.size(); ++t) {
        float max_diff = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            max_diff =
                std::max(max_diff, std::abs(full_recompute_last_rows[t][j] - incremental_last_rows[t][j]));
        }
        if (first_divergent_step == -1 && max_diff > divergence_threshold) {
            first_divergent_step = static_cast<int>(t);
            max_diff_at_first_divergence = max_diff;
        }
    }

    if (first_divergent_step != -1) {
        ADD_FAILURE() << "Incremental KV-cache decode diverges from full recompute starting at "
                         "step "
                      << first_divergent_step << " (0-based; token=" << tokens[first_divergent_step]
                      << ", prefix length=" << (first_divergent_step + 1)
                      << "), max abs diff = " << max_diff_at_first_divergence
                      << " (threshold " << divergence_threshold << ")";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
