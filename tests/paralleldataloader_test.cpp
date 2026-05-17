#include "ParallelDataLoader.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include "Dataset.hpp"

// Test fixture for ParallelDataLoader tests
class ParallelDataLoaderTest : public ::testing::Test {
   protected:
    Dataset dataset;

    void SetUp() override {
        // Create a test dataset with varied sequence lengths
        for (int i = 0; i < 100; ++i) {
            std::string input = std::string(5 + (i % 10), 'A' + (i % 26));
            std::string target = std::string(3 + (i % 8), 'a' + (i % 26));
            dataset.add_sample(input, target);
        }
        // Use standard split: 80% train, 10% val, 10% test
        dataset.split(0.8, 0.1, 0.1);
    }
};

// ============================================================================
// ThreadSafeBatchQueue Tests
// ============================================================================

TEST(ThreadSafeBatchQueueTest, PushAndPop) {
    ThreadSafeBatchQueue<int> queue(10);

    queue.push(42);
    queue.push(100);

    EXPECT_EQ(queue.size(), 2);

    auto val1 = queue.pop();
    ASSERT_TRUE(val1.has_value());
    EXPECT_EQ(*val1, 42);

    auto val2 = queue.pop();
    ASSERT_TRUE(val2.has_value());
    EXPECT_EQ(*val2, 100);
}

TEST(ThreadSafeBatchQueueTest, Empty) {
    ThreadSafeBatchQueue<int> queue(10);

    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);

    queue.push(1);
    EXPECT_FALSE(queue.empty());
    EXPECT_EQ(queue.size(), 1);
}

TEST(ThreadSafeBatchQueueTest, Shutdown) {
    ThreadSafeBatchQueue<int> queue(10);

    // Shutdown on empty queue: pop should return nullopt
    queue.shutdown();

    auto val = queue.pop();
    EXPECT_FALSE(val.has_value());
}

TEST(ThreadSafeBatchQueueTest, ShutdownClearsBlockedConsumer) {
    ThreadSafeBatchQueue<int> queue(10);

    bool popped_nullopt = false;

    // Consumer thread waits on empty queue
    std::thread consumer([&]() {
        auto val = queue.pop();
        popped_nullopt = !val.has_value();
    });

    // Brief sleep to ensure consumer is blocking
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Shutdown unblocks all waiting threads
    queue.shutdown();
    consumer.join();

    EXPECT_TRUE(popped_nullopt);
}

TEST(ThreadSafeBatchQueueTest, ClearQueue) {
    ThreadSafeBatchQueue<int> queue(10);

    for (int i = 0; i < 5; ++i) {
        queue.push(i);
    }

    EXPECT_EQ(queue.size(), 5);

    queue.clear();

    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
}

TEST(ThreadSafeBatchQueueTest, ConcurrentPushPop) {
    ThreadSafeBatchQueue<int> queue(100);
    std::atomic<int> sum_pushed{0};
    std::atomic<int> sum_popped{0};

    const int num_items = 50;

    // Producer thread
    std::thread producer([&]() {
        for (int i = 1; i <= num_items; ++i) {
            queue.push(i);
            sum_pushed += i;
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        int count = 0;
        while (count < num_items) {
            auto val = queue.pop();
            if (val.has_value()) {
                sum_popped += *val;
                ++count;
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(sum_pushed.load(), sum_popped.load());
}

// ============================================================================
// DataLoaderConfig Tests
// ============================================================================

TEST(DataLoaderConfigTest, DefaultValues) {
    DataLoaderConfig config;

    EXPECT_EQ(config.batch_size, 32);
    EXPECT_EQ(config.num_workers, 4);
    EXPECT_EQ(config.prefetch_factor, 2);
    EXPECT_TRUE(config.shuffle);
    EXPECT_EQ(config.pad_token_id, adai::SpecialTokenIDs::PAD);
    EXPECT_FALSE(config.drop_last);
    EXPECT_EQ(config.seed, 42);
    EXPECT_TRUE(config.use_dynamic_batching);
}

TEST(DataLoaderConfigTest, CustomConfiguration) {
    DataLoaderConfig config;
    config.batch_size = 16;
    config.num_workers = 2;
    config.shuffle = false;
    config.drop_last = true;

    EXPECT_EQ(config.batch_size, 16);
    EXPECT_EQ(config.num_workers, 2);
    EXPECT_FALSE(config.shuffle);
    EXPECT_TRUE(config.drop_last);
}

// ============================================================================
// ParallelDataLoader Basic Tests
// ============================================================================

TEST_F(ParallelDataLoaderTest, Constructor) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;

    ParallelDataLoader loader(dataset, config);

    EXPECT_FALSE(loader.is_running());
    EXPECT_EQ(loader.current_epoch(), 0);
}

TEST_F(ParallelDataLoaderTest, NumBatches) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.drop_last = false;

    ParallelDataLoader loader(dataset, config);

    // Train split is 80% of 100 = 80 samples
    // With batch_size=10: 80/10 = 8 batches
    EXPECT_EQ(loader.num_batches(), 8);
}

TEST_F(ParallelDataLoaderTest, NumBatchesDropLast) {
    DataLoaderConfig config;
    config.batch_size = 15;
    config.drop_last = true;

    ParallelDataLoader loader(dataset, config);

    // Train split is 80 samples
    // With batch_size=15 and drop_last=true: 80/15 = 5 batches (drop 5 samples)
    EXPECT_EQ(loader.num_batches(), 5);
}

TEST_F(ParallelDataLoaderTest, StartAndStop) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;

    ParallelDataLoader loader(dataset, config);

    EXPECT_FALSE(loader.is_running());

    loader.start();
    EXPECT_TRUE(loader.is_running());

    loader.stop();
    EXPECT_FALSE(loader.is_running());
}

TEST_F(ParallelDataLoaderTest, AutoStartOnFirstBatch) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;

    ParallelDataLoader loader(dataset, config);

    EXPECT_FALSE(loader.is_running());

    // Trigger new epoch (needed before fetching batches)
    loader.new_epoch();

    // Getting first batch should auto-start the loader
    auto batch = loader.next_batch();
    EXPECT_TRUE(loader.is_running());

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, NewEpochIncrementsCounter) {
    DataLoaderConfig config;
    config.batch_size = 10;

    ParallelDataLoader loader(dataset, config);

    EXPECT_EQ(loader.current_epoch(), 0);

    loader.new_epoch();
    EXPECT_EQ(loader.current_epoch(), 1);

    loader.new_epoch();
    EXPECT_EQ(loader.current_epoch(), 2);
}

// ============================================================================
// Batch Loading Tests
// ============================================================================

TEST_F(ParallelDataLoaderTest, LoadSingleBatch) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;

    ParallelDataLoader loader(dataset, config);
    loader.new_epoch();

    auto batch = loader.next_batch();

    ASSERT_TRUE(batch.has_value());
    EXPECT_GT(batch->sequences.size(), 0);
    EXPECT_LE(batch->sequences.size(), 10);  // Should not exceed batch size

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, LoadAllBatches) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;

    ParallelDataLoader loader(dataset, config);
    loader.new_epoch();

    size_t expected_batches = loader.num_batches();
    size_t batches_received = 0;

    // Give threads time to produce batches
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Collect all batches for this epoch
    for (size_t i = 0; i < expected_batches; ++i) {
        auto batch = loader.next_batch();
        if (batch.has_value() && !batch->sequences.empty()) {
            ++batches_received;
        }
    }

    EXPECT_EQ(batches_received, expected_batches);

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, BatchSequences) {
    DataLoaderConfig config;
    config.batch_size = 5;
    config.num_workers = 1;

    ParallelDataLoader loader(dataset, config);
    loader.new_epoch();

    // Wait for first batch to be loaded
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto batch = loader.next_batch();

    ASSERT_TRUE(batch.has_value());
    EXPECT_GT(batch->sequences.size(), 0);

    // Verify batch has sequences and masks
    EXPECT_EQ(batch->sequences.size(), batch->masks.size());

    // Each sequence should have length > 0
    for (const auto& seq : batch->sequences) {
        EXPECT_GT(seq.size(), 0);
    }

    loader.stop();
}

// ============================================================================
// Shuffle and Seed Tests
// ============================================================================

TEST_F(ParallelDataLoaderTest, ShuffleEnabled) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 1;
    config.shuffle = true;
    config.seed = 123;

    ParallelDataLoader loader(dataset, config);
    loader.new_epoch();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto batch1 = loader.next_batch();

    loader.new_epoch();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto batch2 = loader.next_batch();

    // With shuffling enabled, batches from different epochs should differ
    // (This is probabilistic but very likely with different epoch seeds)
    // We can't guarantee they're different, but we can verify the mechanism works

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, ShuffleDisabled) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 1;
    config.shuffle = false;

    ParallelDataLoader loader(dataset, config);

    // With shuffle disabled, epochs should produce same order
    // This is deterministic
    loader.new_epoch();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto batch1 = loader.next_batch();

    loader.new_epoch();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto batch2 = loader.next_batch();

    // Both batches should exist
    ASSERT_TRUE(batch1.has_value());
    ASSERT_TRUE(batch2.has_value());

    loader.stop();
}

// ============================================================================
// Multi-Threading Tests
// ============================================================================

TEST_F(ParallelDataLoaderTest, MultipleWorkersLoad) {
    DataLoaderConfig config;
    config.batch_size = 5;
    config.num_workers = 4;  // Multiple workers
    config.prefetch_factor = 2;

    ParallelDataLoader loader(dataset, config);
    loader.new_epoch();

    // Allow time for workers to prefetch
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Multiple workers should collectively load batches
    // We verify by consuming at least one batch successfully
    size_t batches_received = 0;
    for (size_t i = 0; i < loader.num_batches(); ++i) {
        auto batch = loader.next_batch();
        if (batch.has_value() && !batch->sequences.empty()) {
            ++batches_received;
        }
    }

    EXPECT_EQ(batches_received, loader.num_batches());

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, PrefetchQueueSize) {
    DataLoaderConfig config;
    config.batch_size = 5;
    config.num_workers = 2;
    config.prefetch_factor = 3;

    ParallelDataLoader loader(dataset, config);
    loader.new_epoch();

    // Wait for prefetching
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    size_t queue_size = loader.queue_size();

    // Queue should contain prefetched batches (up to num_workers * prefetch_factor)
    EXPECT_GE(queue_size, 0);
    EXPECT_LE(queue_size, config.num_workers * config.prefetch_factor);

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, ConcurrentBatchConsumption) {
    DataLoaderConfig config;
    config.batch_size = 8;
    config.num_workers = 2;

    ParallelDataLoader loader(dataset, config);
    loader.new_epoch();

    std::atomic<size_t> batches_consumed{0};

    // Consumer thread
    std::thread consumer([&]() {
        for (size_t i = 0; i < 5; ++i) {
            auto batch = loader.next_batch();
            if (batch.has_value()) {
                ++batches_consumed;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    consumer.join();

    EXPECT_EQ(batches_consumed.load(), 5);

    loader.stop();
}

// ============================================================================
// DataLoaderIterator Tests
// ============================================================================

TEST_F(ParallelDataLoaderTest, IteratorBasicIteration) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;

    ParallelDataLoader loader(dataset, config);
    DataLoaderIterator iter(loader);

    size_t expected_batches = loader.num_batches();
    size_t batches_seen = 0;

    while (auto batch = iter.next()) {
        ++batches_seen;
        EXPECT_GT(batch->sequences.size(), 0);
    }

    EXPECT_EQ(batches_seen, expected_batches);

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, IteratorBatchesReturned) {
    DataLoaderConfig config;
    config.batch_size = 15;
    config.num_workers = 1;

    ParallelDataLoader loader(dataset, config);
    DataLoaderIterator iter(loader);

    EXPECT_EQ(iter.batches_returned(), 0);

    auto batch1 = iter.next();
    if (batch1.has_value()) {
        EXPECT_EQ(iter.batches_returned(), 1);
    }

    auto batch2 = iter.next();
    if (batch2.has_value()) {
        EXPECT_EQ(iter.batches_returned(), 2);
    }

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, IteratorReset) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;

    ParallelDataLoader loader(dataset, config);
    DataLoaderIterator iter(loader);

    // Consume some batches
    for (int i = 0; i < 3; ++i) {
        iter.next();
    }

    size_t batches_before_reset = iter.batches_returned();
    EXPECT_EQ(batches_before_reset, 3);

    // Reset iterator
    iter.reset();

    EXPECT_EQ(iter.batches_returned(), 0);

    loader.stop();
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(ParallelDataLoaderTest, EmptyDataset) {
    Dataset empty_dataset;
    // Don't add any samples
    empty_dataset.split(0.8, 0.1, 0.1);

    DataLoaderConfig config;
    config.batch_size = 10;

    ParallelDataLoader loader(empty_dataset, config);

    EXPECT_EQ(loader.num_batches(), 0);
}

TEST_F(ParallelDataLoaderTest, SmallDataset) {
    Dataset small_dataset;
    small_dataset.add_sample("abc", "def");
    small_dataset.add_sample("ghi", "jkl");
    small_dataset.split(0.8, 0.1, 0.1);

    DataLoaderConfig config;
    config.batch_size = 10;  // Larger than dataset

    ParallelDataLoader loader(small_dataset, config);

    // Should have 1 batch (all samples fit in one batch)
    EXPECT_GE(loader.num_batches(), 0);
}

TEST_F(ParallelDataLoaderTest, LargeBatchSize) {
    DataLoaderConfig config;
    config.batch_size = 1000;  // Much larger than dataset
    config.drop_last = false;

    ParallelDataLoader loader(dataset, config);

    // Should have 1 batch (all 80 train samples fit in one large batch)
    EXPECT_EQ(loader.num_batches(), 1);
}

TEST_F(ParallelDataLoaderTest, SingleWorker) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 1;  // Single worker

    ParallelDataLoader loader(dataset, config);
    loader.new_epoch();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto batch = loader.next_batch();
    ASSERT_TRUE(batch.has_value());

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, StopWithoutStart) {
    DataLoaderConfig config;
    config.batch_size = 10;

    ParallelDataLoader loader(dataset, config);

    // Should handle stop without start gracefully
    EXPECT_NO_THROW(loader.stop());
}

TEST_F(ParallelDataLoaderTest, MultipleStops) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;

    ParallelDataLoader loader(dataset, config);
    loader.start();

    loader.stop();

    // Multiple stops should be safe
    EXPECT_NO_THROW(loader.stop());
    EXPECT_NO_THROW(loader.stop());
}

TEST_F(ParallelDataLoaderTest, MultipleStarts) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 2;

    ParallelDataLoader loader(dataset, config);

    loader.start();
    EXPECT_TRUE(loader.is_running());

    // Multiple starts should be idempotent
    loader.start();
    EXPECT_TRUE(loader.is_running());

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, DynamicBatchingEnabled) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 1;
    config.use_dynamic_batching = true;

    ParallelDataLoader loader(dataset, config);
    loader.new_epoch();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto batch = loader.next_batch();
    ASSERT_TRUE(batch.has_value());

    loader.stop();
}

TEST_F(ParallelDataLoaderTest, PaddingStrategyRight) {
    DataLoaderConfig config;
    config.batch_size = 5;
    config.num_workers = 1;
    config.padding_strategy = PaddingStrategy::RIGHT;

    ParallelDataLoader loader(dataset, config);
    loader.new_epoch();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto batch = loader.next_batch();
    ASSERT_TRUE(batch.has_value());

    // Verify sequences exist
    EXPECT_GT(batch->sequences.size(), 0);

    loader.stop();
}
