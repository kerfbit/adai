/**
 * @file datapipeline_test.cpp
 * @brief Comprehensive tests for data pipeline (EfficientBatching, ThreadSafeBatchQueue, and
 * the end-to-end TokenBatchLoader integration test) — TD-052: ParallelDataLoader/
 * DataLoaderConfig/DataLoaderIterator, formerly tested here too, were retired.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include "Dataset.hpp"
#include "EfficientBatching.hpp"
#include "ParallelDataLoader.hpp"

// ============================================================================
// EfficientBatching Tests
// ============================================================================

class EfficientBatchingTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Create test sequences of varying lengths
        test_sequences_ = {
            {1, 2, 3},             // length 3
            {4, 5, 6, 7},          // length 4
            {8, 9},                // length 2
            {10, 11, 12, 13, 14},  // length 5
            {15, 16, 17},          // length 3
            {18, 19, 20, 21}       // length 4
        };
    }

    std::vector<std::vector<int>> test_sequences_;
};

TEST_F(EfficientBatchingTest, PadSequenceRight) {
    std::vector<int> seq = {1, 2, 3};
    auto padded = EfficientBatching::pad_sequence(seq, 5, 0, PaddingStrategy::RIGHT);

    EXPECT_EQ(padded.size(), 5);
    EXPECT_EQ(padded[0], 1);
    EXPECT_EQ(padded[1], 2);
    EXPECT_EQ(padded[2], 3);
    EXPECT_EQ(padded[3], 0);
    EXPECT_EQ(padded[4], 0);
}

TEST_F(EfficientBatchingTest, PadSequenceLeft) {
    std::vector<int> seq = {1, 2, 3};
    auto padded = EfficientBatching::pad_sequence(seq, 5, 0, PaddingStrategy::LEFT);

    EXPECT_EQ(padded.size(), 5);
    EXPECT_EQ(padded[0], 0);
    EXPECT_EQ(padded[1], 0);
    EXPECT_EQ(padded[2], 1);
    EXPECT_EQ(padded[3], 2);
    EXPECT_EQ(padded[4], 3);
}

TEST_F(EfficientBatchingTest, PadSequenceCenter) {
    std::vector<int> seq = {1, 2, 3};
    auto padded = EfficientBatching::pad_sequence(seq, 6, 0, PaddingStrategy::CENTER);

    EXPECT_EQ(padded.size(), 6);
    // Should have 1 padding on left, content, 2 padding on right (or similar)
    EXPECT_EQ(padded[1], 1);
    EXPECT_EQ(padded[2], 2);
    EXPECT_EQ(padded[3], 3);
}

TEST_F(EfficientBatchingTest, AttentionMaskRight) {
    auto mask = EfficientBatching::create_attention_mask(3, 5, PaddingStrategy::RIGHT);

    EXPECT_EQ(mask.size(), 5);
    EXPECT_EQ(mask[0], 1);
    EXPECT_EQ(mask[1], 1);
    EXPECT_EQ(mask[2], 1);
    EXPECT_EQ(mask[3], 0);
    EXPECT_EQ(mask[4], 0);
}

TEST_F(EfficientBatchingTest, AttentionMaskLeft) {
    auto mask = EfficientBatching::create_attention_mask(3, 5, PaddingStrategy::LEFT);

    EXPECT_EQ(mask.size(), 5);
    EXPECT_EQ(mask[0], 0);
    EXPECT_EQ(mask[1], 0);
    EXPECT_EQ(mask[2], 1);
    EXPECT_EQ(mask[3], 1);
    EXPECT_EQ(mask[4], 1);
}

TEST_F(EfficientBatchingTest, CreateDynamicBatchesBasic) {
    auto batches = EfficientBatching::create_dynamic_batches(test_sequences_, 2, 0,
                                                             PaddingStrategy::RIGHT, false);

    EXPECT_EQ(batches.size(), 3);  // 6 sequences / 2 per batch = 3 batches

    for (const auto& batch : batches) {
        EXPECT_LE(batch.sequences.size(), 2);
        EXPECT_EQ(batch.sequences.size(), batch.masks.size());
        EXPECT_EQ(batch.sequences.size(), batch.lengths.size());
    }
}

TEST_F(EfficientBatchingTest, CreateDynamicBatchesSorted) {
    auto batches = EfficientBatching::create_dynamic_batches(
        test_sequences_, 2, 0, PaddingStrategy::RIGHT, true  // sort by length
    );

    EXPECT_EQ(batches.size(), 3);

    // First batch should have shortest sequences (less padding)
    EXPECT_LE(batches[0].max_length, batches[1].max_length);
    EXPECT_LE(batches[1].max_length, batches[2].max_length);
}

TEST_F(EfficientBatchingTest, BatchStatistics) {
    auto batches = EfficientBatching::create_dynamic_batches(test_sequences_, 2, 0,
                                                             PaddingStrategy::RIGHT, false);

    auto stats = EfficientBatching::calculate_statistics(batches);

    EXPECT_EQ(stats.num_batches, 3);
    EXPECT_EQ(stats.total_sequences, 6);
    EXPECT_GT(stats.total_tokens, 0);
    EXPECT_GE(stats.total_padding_tokens, 0);
    EXPECT_GT(stats.avg_batch_size, 0.0);
    EXPECT_GE(stats.padding_ratio, 0.0);
    EXPECT_LE(stats.padding_ratio, 1.0);
    EXPECT_EQ(stats.efficiency_score, 1.0 - stats.padding_ratio);
}

TEST_F(EfficientBatchingTest, BucketedBatches) {
    BucketConfig config;
    config.bucket_boundaries = {3, 5};  // Buckets: <=3, <=5, >5
    config.max_tokens_per_batch = 100;
    config.shuffle_buckets = false;

    auto batches = EfficientBatching::create_bucketed_batches(test_sequences_, config, 0,
                                                              PaddingStrategy::RIGHT);

    EXPECT_GT(batches.size(), 0);

    // Check that batches respect token limit
    for (const auto& batch : batches) {
        EXPECT_LE(batch.total_tokens(), config.max_tokens_per_batch);
    }
}

// TD-074 regression: create_bucketed_batches()'s greedy batch-forming loop
// used to compare each new candidate's length only against the FIRST
// sequence ever added to the batch, not the true running maximum across
// everything already in it. Once a longer sequence joined a batch, later
// candidates were silently compared against the stale, smaller first-element
// length, letting the loop admit sequences well past max_tokens_per_batch.
// A bucket groups by a coarse length range (here, a single bucket covering
// everything up to boundary 200), so a short-then-long-then-short pattern
// within one bucket is exactly the case that exposed the bug.
TEST_F(EfficientBatchingTest, BucketedBatchesRespectTokenLimitWithMixedLengthOrder) {
    std::vector<std::vector<int>> sequences = {
        std::vector<int>(50, 1),   // length 50
        std::vector<int>(100, 1),  // length 100 - joins after the short one
        std::vector<int>(30, 1),   // length 30
        std::vector<int>(30, 1),   // length 30
        std::vector<int>(30, 1),   // length 30
        std::vector<int>(30, 1),   // length 30
    };

    BucketConfig config;
    config.bucket_boundaries = {200};  // single bucket covers all six sequences
    config.max_tokens_per_batch = 300;
    config.shuffle_buckets = false;

    auto batches = EfficientBatching::create_bucketed_batches(sequences, config, 0,
                                                              PaddingStrategy::RIGHT);

    ASSERT_GT(batches.size(), 0);
    for (const auto& batch : batches) {
        // total_tokens() uses the batch's own max_length, i.e. the TRUE
        // padded size — this is what used to come out at 600 (double the
        // 300 limit) before the fix.
        EXPECT_LE(batch.total_tokens(), config.max_tokens_per_batch);
    }
}

TEST_F(EfficientBatchingTest, DataAugmentationTokenDropout) {
    auto sequences = test_sequences_;

    AugmentationConfig config;
    config.enable_token_dropout = true;
    config.token_dropout_prob = 0.5f;
    config.seed = 42;

    EfficientBatching::apply_augmentation(sequences, config);

    // Some tokens should be dropped (sequence lengths should be smaller or equal)
    bool found_shorter = false;
    for (size_t i = 0; i < sequences.size(); ++i) {
        EXPECT_LE(sequences[i].size(), test_sequences_[i].size());
        if (sequences[i].size() < test_sequences_[i].size()) {
            found_shorter = true;
        }
    }
    // With 50% dropout, we should see at least one shorter sequence
    EXPECT_TRUE(found_shorter);
}

TEST_F(EfficientBatchingTest, DataAugmentationTokenMasking) {
    auto sequences = test_sequences_;

    AugmentationConfig config;
    config.enable_token_masking = true;
    config.token_mask_prob = 1.0f;  // Mask all tokens
    config.mask_token_id = 999;
    config.seed = 42;

    EfficientBatching::apply_augmentation(sequences, config);

    // All tokens should be masked
    for (const auto& seq : sequences) {
        for (int token : seq) {
            EXPECT_EQ(token, 999);
        }
    }
}

TEST_F(EfficientBatchingTest, EmptySequencesThrows) {
    std::vector<std::vector<int>> empty_sequences;

    EXPECT_THROW(EfficientBatching::create_dynamic_batches(empty_sequences, 2, 0),
                 std::invalid_argument);
}

TEST_F(EfficientBatchingTest, SingleSequenceBatch) {
    std::vector<std::vector<int>> single = {{1, 2, 3}};

    auto batches = EfficientBatching::create_dynamic_batches(single, 1, 0);

    EXPECT_EQ(batches.size(), 1);
    EXPECT_EQ(batches[0].sequences.size(), 1);
    EXPECT_EQ(batches[0].max_length, 3);
}

TEST_F(EfficientBatchingTest, PaddingRatioCalculation) {
    // Create batch with known padding
    std::vector<std::vector<int>> seqs = {
        {1, 2},       // length 2
        {3, 4, 5, 6}  // length 4
    };

    auto batches = EfficientBatching::create_dynamic_batches(seqs, 2, 0);
    EXPECT_EQ(batches.size(), 1);

    // Max length is 4, so total tokens = 2 * 4 = 8
    // Real tokens = 2 + 4 = 6
    // Padding = 2
    // Padding ratio = 2/8 = 0.25
    EXPECT_EQ(batches[0].total_tokens(), 8);
    EXPECT_EQ(batches[0].padding_tokens(), 2);
    EXPECT_FLOAT_EQ(batches[0].padding_ratio(), 0.25);
}

// ============================================================================
// ThreadSafeBatchQueue Tests
// ============================================================================

TEST(ThreadSafeBatchQueueTest, PushAndPop) {
    ThreadSafeBatchQueue<int> queue(10);

    queue.push(42);
    auto result = queue.pop();

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

TEST(ThreadSafeBatchQueueTest, MultipleElements) {
    ThreadSafeBatchQueue<int> queue(10);

    for (int i = 0; i < 5; ++i) {
        queue.push(i);
    }

    EXPECT_EQ(queue.size(), 5);

    for (int i = 0; i < 5; ++i) {
        auto result = queue.pop();
        EXPECT_TRUE(result.has_value());
        EXPECT_EQ(*result, i);
    }

    EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafeBatchQueueTest, Shutdown) {
    ThreadSafeBatchQueue<int> queue(10);

    queue.push(1);
    queue.push(2);

    queue.shutdown();

    // After shutdown, pop should return empty
    auto result1 = queue.pop();
    EXPECT_TRUE(result1.has_value());  // Still has queued items

    auto result2 = queue.pop();
    EXPECT_TRUE(result2.has_value());

    auto result3 = queue.pop();
    EXPECT_FALSE(result3.has_value());  // Queue empty after shutdown
}

TEST(ThreadSafeBatchQueueTest, Clear) {
    ThreadSafeBatchQueue<int> queue(10);

    queue.push(1);
    queue.push(2);
    queue.push(3);

    EXPECT_EQ(queue.size(), 3);

    queue.clear();

    EXPECT_EQ(queue.size(), 0);
    EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafeBatchQueueTest, ConcurrentAccess) {
    ThreadSafeBatchQueue<int> queue(100);

    // Producer thread
    std::thread producer([&queue]() {
        for (int i = 0; i < 50; ++i) {
            queue.push(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Consumer thread
    std::atomic<int> consumed_count(0);
    std::thread consumer([&queue, &consumed_count]() {
        for (int i = 0; i < 50; ++i) {
            auto result = queue.pop();
            if (result.has_value()) {
                consumed_count++;
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(consumed_count, 50);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(DataPipelineIntegrationTest, EndToEndPipeline) {
    // Create dataset
    Dataset dataset;

    // Add samples with varying lengths
    for (int i = 0; i < 50; ++i) {
        std::string input = "Input " + std::to_string(i);
        std::string target = "Response " + std::to_string(i);

        // Add varying length text
        int extra_words = i % 5;
        for (int j = 0; j < extra_words; ++j) {
            input += " word" + std::to_string(j);
        }

        dataset.add_sample(input, target);
    }

    dataset.split(1.0, 0.0, 0.0);  // All training

    // Create data loader with a real tokenizer function. TD-052: this used to
    // go through ParallelDataLoader/DataLoaderConfig, whose batch generation
    // used raw char codes with no tokenizer parameter at all — retired in
    // favor of TokenBatchLoader, which takes a tokenizer function via
    // constructor injection instead (see TECHNICAL_DEBT.md's resolved
    // archive).
    TokenBatchLoaderConfig loader_config;
    loader_config.batch_size = 8;
    loader_config.num_workers = 2;
    loader_config.shuffle = true;
    loader_config.use_dynamic_batching = true;

    auto tokenizer_fn = [](const std::string& text) {
        std::vector<int> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int>(static_cast<unsigned char>(c)));
        }
        return tokens;
    };

    TokenBatchLoader loader(dataset, loader_config, tokenizer_fn);
    TokenBatchIterator iter(loader);

    // Process one epoch
    size_t total_sequences = 0;

    while (auto batch = iter.next()) {
        EXPECT_GT(batch->batch_size(), 0);
        EXPECT_LE(static_cast<size_t>(batch->batch_size()), loader_config.batch_size);

        // Verify lengths match batch size
        EXPECT_EQ(static_cast<size_t>(batch->batch_size()), batch->lengths.size());

        // Verify every row is padded to the batch's max length
        for (const auto& row : batch->batch_token_ids) {
            EXPECT_EQ(static_cast<int>(row.size()), batch->max_length);
        }

        total_sequences += static_cast<size_t>(batch->batch_size());
    }

    EXPECT_EQ(total_sequences, 50u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
