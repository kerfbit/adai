/**
 * @file datapipeline_test.cpp
 * @brief Comprehensive tests for data pipeline (EfficientBatching and ParallelDataLoader)
 */

#include <gtest/gtest.h>
#include "EfficientBatching.hpp"
#include "ParallelDataLoader.hpp"
#include "Dataset.hpp"
#include <vector>
#include <numeric>
#include <algorithm>

// ============================================================================
// EfficientBatching Tests
// ============================================================================

class EfficientBatchingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test sequences of varying lengths
        test_sequences_ = {
            {1, 2, 3},           // length 3
            {4, 5, 6, 7},        // length 4
            {8, 9},              // length 2
            {10, 11, 12, 13, 14}, // length 5
            {15, 16, 17},        // length 3
            {18, 19, 20, 21}     // length 4
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
    auto batches = EfficientBatching::create_dynamic_batches(
        test_sequences_, 2, 0, PaddingStrategy::RIGHT, false
    );
    
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
    auto batches = EfficientBatching::create_dynamic_batches(
        test_sequences_, 2, 0, PaddingStrategy::RIGHT, false
    );
    
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
    
    auto batches = EfficientBatching::create_bucketed_batches(
        test_sequences_, config, 0, PaddingStrategy::RIGHT
    );
    
    EXPECT_GT(batches.size(), 0);
    
    // Check that batches respect token limit
    for (const auto& batch : batches) {
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
    
    EXPECT_THROW(
        EfficientBatching::create_dynamic_batches(empty_sequences, 2, 0),
        std::invalid_argument
    );
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
        {1, 2},      // length 2
        {3, 4, 5, 6} // length 4
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
// ParallelDataLoader Tests
// ============================================================================

class ParallelDataLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test dataset
        dataset_ = std::make_unique<Dataset>();
        
        // Add test samples
        for (int i = 0; i < 100; ++i) {
            std::string input = "Input " + std::to_string(i);
            std::string target = "Response " + std::to_string(i);
            dataset_->add_sample(input, target);
        }
        
        // Split the data (100% train for simplicity)
        dataset_->split(1.0, 0.0, 0.0);
    }
    
    std::unique_ptr<Dataset> dataset_;
};

TEST_F(ParallelDataLoaderTest, ConstructorAndBasicSetup) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;
    
    ParallelDataLoader loader(*dataset_, config);
    
    EXPECT_FALSE(loader.is_running());
    EXPECT_EQ(loader.current_epoch(), 0);
    EXPECT_EQ(loader.batches_loaded(), 0);
}

TEST_F(ParallelDataLoaderTest, NumBatchesCalculation) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.drop_last = false;
    
    ParallelDataLoader loader(*dataset_, config);
    
    // 100 samples / 10 per batch = 10 batches
    EXPECT_EQ(loader.num_batches(), 10);
}

TEST_F(ParallelDataLoaderTest, NumBatchesDropLast) {
    DataLoaderConfig config;
    config.batch_size = 15;
    config.drop_last = true;
    
    ParallelDataLoader loader(*dataset_, config);
    
    // 100 samples / 15 per batch = 6 complete batches (drop last with 10 samples)
    EXPECT_EQ(loader.num_batches(), 6);
}

TEST_F(ParallelDataLoaderTest, StartAndStop) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;
    
    ParallelDataLoader loader(*dataset_, config);
    
    loader.start();
    EXPECT_TRUE(loader.is_running());
    
    loader.stop();
    EXPECT_FALSE(loader.is_running());
}

TEST_F(ParallelDataLoaderTest, LoadBatches) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;
    config.shuffle = false;
    
    ParallelDataLoader loader(*dataset_, config);
    
    // Use iterator which properly initializes the epoch
    DataLoaderIterator iter(loader);
    
    // Load a few batches
    int batches_received = 0;
    for (int i = 0; i < 5; ++i) {
        auto batch = iter.next();
        if (batch.has_value() && batch->sequences.size() > 0) {
            ++batches_received;
            EXPECT_GT(batch->sequences.size(), 0);
            EXPECT_LE(batch->sequences.size(), config.batch_size);
        }
    }
    
    EXPECT_GT(batches_received, 0);
}

TEST_F(ParallelDataLoaderTest, EpochIteration) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;
    
    ParallelDataLoader loader(*dataset_, config);
    loader.start();
    
    size_t expected_batches = loader.num_batches();
    
    // Iterate through one epoch
    size_t batches_received = 0;
    for (size_t i = 0; i < expected_batches; ++i) {
        auto batch = loader.next_batch();
        if (batch.has_value()) {
            ++batches_received;
        }
    }
    
    EXPECT_EQ(batches_received, expected_batches);
    
    loader.stop();
}

TEST_F(ParallelDataLoaderTest, NewEpoch) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 1;
    
    ParallelDataLoader loader(*dataset_, config);
    
    EXPECT_EQ(loader.current_epoch(), 0);
    
    loader.new_epoch();
    EXPECT_EQ(loader.current_epoch(), 1);
    
    loader.new_epoch();
    EXPECT_EQ(loader.current_epoch(), 2);
}

TEST_F(ParallelDataLoaderTest, DataLoaderIterator) {
    DataLoaderConfig config;
    config.batch_size = 20;
    config.num_workers = 2;
    
    ParallelDataLoader loader(*dataset_, config);
    DataLoaderIterator iter(loader);
    
    size_t batches_received = 0;
    while (auto batch = iter.next()) {
        ++batches_received;
        EXPECT_GT(batch->sequences.size(), 0);
    }
    
    EXPECT_EQ(batches_received, loader.num_batches());
    EXPECT_EQ(iter.batches_returned(), loader.num_batches());
}

TEST_F(ParallelDataLoaderTest, IteratorReset) {
    DataLoaderConfig config;
    config.batch_size = 25;
    config.num_workers = 1;
    
    ParallelDataLoader loader(*dataset_, config);
    DataLoaderIterator iter(loader);
    
    // First iteration
    size_t first_count = 0;
    while (auto batch = iter.next()) {
        ++first_count;
    }
    
    // Reset and iterate again
    iter.reset();
    size_t second_count = 0;
    while (auto batch = iter.next()) {
        ++second_count;
    }
    
    EXPECT_EQ(first_count, second_count);
    EXPECT_GT(first_count, 0);
}

TEST_F(ParallelDataLoaderTest, PrefetchQueueSize) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;
    config.prefetch_factor = 2;
    
    ParallelDataLoader loader(*dataset_, config);
    loader.start();
    
    // Give workers time to prefetch
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Queue should have some batches
    EXPECT_GT(loader.queue_size(), 0);
    
    loader.stop();
}

TEST_F(ParallelDataLoaderTest, DynamicBatchingEnabled) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 1;
    config.use_dynamic_batching = true;
    
    ParallelDataLoader loader(*dataset_, config);
    loader.start();
    
    auto batch = loader.next_batch();
    EXPECT_TRUE(batch.has_value());
    
    // Should have attention masks
    EXPECT_EQ(batch->sequences.size(), batch->masks.size());
    
    loader.stop();
}

TEST_F(ParallelDataLoaderTest, AugmentationEnabled) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 1;
    config.augmentation_config.enable_token_masking = true;
    config.augmentation_config.token_mask_prob = 0.2f;  // Lower probability to avoid all tokens being masked
    
    ParallelDataLoader loader(*dataset_, config);
    
    // Use iterator which properly initializes
    DataLoaderIterator iter(loader);
    
    auto batch = iter.next();
    EXPECT_TRUE(batch.has_value());
    if (batch.has_value()) {
        EXPECT_GT(batch->sequences.size(), 0);
    }
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
    
    // Create data loader with all features
    DataLoaderConfig loader_config;
    loader_config.batch_size = 8;
    loader_config.num_workers = 2;
    loader_config.shuffle = true;
    loader_config.use_dynamic_batching = true;
    loader_config.augmentation_config.enable_token_masking = true;
    loader_config.augmentation_config.token_mask_prob = 0.1f;
    
    ParallelDataLoader loader(dataset, loader_config);
    DataLoaderIterator iter(loader);
    
    // Process one epoch
    size_t total_sequences = 0;
    BatchStatistics cumulative_stats;
    
    while (auto batch = iter.next()) {
        EXPECT_GT(batch->sequences.size(), 0);
        EXPECT_LE(batch->sequences.size(), loader_config.batch_size);
        
        // Verify masks match sequences
        EXPECT_EQ(batch->sequences.size(), batch->masks.size());
        
        // Verify all sequences have same length (padded)
        for (const auto& seq : batch->sequences) {
            EXPECT_EQ(seq.size(), batch->max_length);
        }
        
        total_sequences += batch->sequences.size();
    }
    
    EXPECT_EQ(total_sequences, 50);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
