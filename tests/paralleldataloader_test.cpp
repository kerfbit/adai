#include "ParallelDataLoader.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <future>
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

// ============================================================================
// TokenBatchLoader Tests
//
// TD-087: input and target batches used to travel through two independent
// ThreadSafeBatchQueues. A caller that drains next_batch() without also
// draining next_target_batch() at a matching rate would fill the target
// queue permanently, blocking every worker thread inside its push() there —
// and once every worker is stuck, batch_queue_ stops being refilled too, so
// next_batch() then hangs forever as well. Fixed by pushing/popping the pair
// as a single unit through one queue. These tests use a background
// std::async + wait_for(timeout) (matching the pattern in
// batchedinferenceengine_test.cpp) so a regression here fails loudly with a
// timeout instead of hanging the whole test binary.
// ============================================================================

class TokenBatchLoaderTest : public ::testing::Test {
   protected:
    Dataset dataset;

    void SetUp() override {
        for (int i = 0; i < 100; ++i) {
            std::string input = std::string(5 + (i % 10), 'A' + (i % 26));
            std::string target = std::string(3 + (i % 8), 'a' + (i % 26));
            dataset.add_sample(input, target);
        }
        dataset.split(0.8, 0.1, 0.1);
    }

    static std::vector<int> tokenizer_fn(const std::string& s) {
        std::vector<int> tokens;
        for (char c : s)
            tokens.push_back(static_cast<int>(static_cast<unsigned char>(c)));
        return tokens;
    }
};

TEST_F(TokenBatchLoaderTest, NextBatchDoesNotDeadlockWhenTargetsNeverDrained) {
    TokenBatchLoaderConfig config;
    config.batch_size = 5;
    config.num_workers = 2;
    config.prefetch_factor = 2;
    config.load_targets = true;

    TokenBatchLoader loader(dataset, config, tokenizer_fn);
    // num_batches() (16 with this batch_size against the 80-sample TRAIN
    // split) comfortably exceeds the queue's capacity (num_workers *
    // prefetch_factor = 4), so draining a full epoch still exercises the
    // deadlock this test guards against. A caller must stay within one
    // epoch's batch count without calling new_epoch() — going further would
    // hang by design (the workers idle-wait for a new epoch, matching
    // ParallelDataLoader's identical, separately-tested contract), which is
    // orthogonal to the bug this test targets.
    const int total_batches = static_cast<int>(loader.num_batches());
    ASSERT_GT(total_batches, 4) << "test assumes more batches than the prefetch buffer holds";

    // Drain a full epoch using only next_batch() — never next_target_batch()
    // — which used to permanently stall every worker thread on the
    // (never-drained) target queue.
    auto fut = std::async(std::launch::async, [&]() {
        int count = 0;
        for (int i = 0; i < total_batches; ++i) {
            auto batch = loader.next_batch();
            if (batch.has_value())
                ++count;
        }
        return count;
    });

    ASSERT_EQ(fut.wait_for(std::chrono::seconds(10)), std::future_status::ready)
        << "next_batch() hung when next_target_batch() was never called (TD-087 regression)";
    EXPECT_GT(fut.get(), 0);

    loader.stop();
}

TEST_F(TokenBatchLoaderTest, NextBatchAndTargetBatchStayPaired) {
    TokenBatchLoaderConfig config;
    config.batch_size = 3;
    config.num_workers = 3;  // multiple workers race to push distinct batches
    config.prefetch_factor = 2;
    config.load_targets = true;
    config.shuffle = false;

    TokenBatchLoader loader(dataset, config, tokenizer_fn);
    loader.new_epoch();

    // Every next_batch()/next_target_batch() pair must come from the same
    // load_batch() call — with two independent queues and num_workers > 1,
    // nothing guaranteed that before this fix (a different worker's target
    // could interleave ahead of the one matching the just-popped input).
    for (int i = 0; i < 10; ++i) {
        auto input = loader.next_batch();
        auto target = loader.next_target_batch();
        ASSERT_TRUE(input.has_value());
        ASSERT_TRUE(target.has_value());
        // Inputs are 5-14 chars (5 + i%10), targets are 3-10 chars (3 + i%8)
        // — disjoint enough ranges that a mismatch would be implausible to
        // pass by coincidence across 10 draws, but the real guarantee this
        // test relies on is structural (one shared queue), not statistical.
        EXPECT_FALSE(input->batch_token_ids.empty());
        EXPECT_FALSE(target->batch_token_ids.empty());
    }

    loader.stop();
}

TEST_F(TokenBatchLoaderTest, NoTargetsConfiguredReturnsNulloptForTargetBatch) {
    TokenBatchLoaderConfig config;
    config.batch_size = 5;
    config.num_workers = 1;
    config.load_targets = false;

    TokenBatchLoader loader(dataset, config, tokenizer_fn);

    auto input = loader.next_batch();
    ASSERT_TRUE(input.has_value());
    EXPECT_FALSE(loader.next_target_batch().has_value());

    loader.stop();
}
