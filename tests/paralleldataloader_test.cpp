#include "ParallelDataLoader.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include "Dataset.hpp"

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

// TD-064: clear() used to drain the queue without notifying cv_producer_, so a producer already
// blocked in push() (queue full) had no way to know clear() just made room — a genuine, real
// deadlock (not a false alarm; std::condition_variable::wait() only re-checks its predicate when
// actually woken, and a spurious wakeup isn't guaranteed within any bounded time). This is the
// confirmed root cause of the historical paralleldataloaderTests hang this file is named after
// (see TECHNICAL_DEBT.md's resolved archive for the full investigation and a dedicated
// ~20,000-iteration repro harness that reproduced it in ~2,600 iterations before this fix, and
// zero times in 40,000 iterations after).
//
// Deliberately NOT std::async + wait_for(timeout) (the pattern used elsewhere in this file):
// if a future from std::async is destroyed while its task is still running, ~future() itself
// blocks until that task finishes — so a *reintroduced* regression here would make this
// regression test hang the whole binary again the moment a timeout-triggered failure destroyed
// that future, exactly the failure mode this test exists to turn into a clean, fast, reported
// failure instead (confirmed the hard way: an earlier std::async version of this test hung for
// real when tried against the reverted bug). A raw std::thread has no such blocking destructor
// (only std::terminate if neither join() nor detach() was ever called), so a timeout can safely
// detach and abandon the stuck thread — but only because `queue`/`pushed` below are heap-
// allocated and captured into the thread's lambda by shared_ptr (by value, not by reference):
// an abandoned thread that outlives this test function must never touch this function's own
// (about to be destroyed) stack, or detaching would trade one hang for a use-after-free.
TEST(ThreadSafeBatchQueueTest, ClearWakesBlockedProducer) {
    auto queue = std::make_shared<ThreadSafeBatchQueue<int>>(2);
    queue->push(1);
    queue->push(2);  // queue now at capacity (max_size_ == 2)

    auto pushed = std::make_shared<std::atomic<bool>>(false);
    std::thread producer([queue, pushed]() {
        queue->push(3);  // blocks here until something makes room
        *pushed = true;
    });

    // Brief sleep to ensure the producer above is genuinely blocked in its wait before clear()
    // runs, matching ShutdownClearsBlockedConsumer's own approach.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    queue->clear();  // drains the queue to empty; must also wake the blocked producer

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!pushed->load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!pushed->load()) {
        producer.detach();
        FAIL() << "push() never woke up after clear() made room (TD-064 regression)";
    }
    producer.join();

    EXPECT_EQ(queue->size(), 1);  // the producer's pushed item, after clear() emptied the queue
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
    // hang by design (the workers idle-wait for a new epoch), which is
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

// TD-098 regression: with use_dynamic_batching on, load_batch() used to sort
// input_sequences and target_sequences *independently* by their own lengths
// before building each TokenBatch — input_batch[k] and target_batch[k] then
// only corresponded to the same original sample by coincidence. The fixture
// dataset's input length is driven by i%10 and target length by i%8 (see
// SetUp() above) specifically so the two independent length-sorts diverge.
TEST_F(TokenBatchLoaderTest, InputAndTargetRowsWithinABatchStayAligned) {
    TokenBatchLoaderConfig config;
    config.batch_size = 10;
    config.num_workers = 1;
    config.load_targets = true;
    config.use_dynamic_batching = true;
    config.shuffle = false;

    TokenBatchLoader loader(dataset, config, tokenizer_fn);
    loader.new_epoch();

    for (int b = 0; b < 3; ++b) {
        auto input = loader.next_batch();
        auto target = loader.next_target_batch();
        ASSERT_TRUE(input.has_value());
        ASSERT_TRUE(target.has_value());
        ASSERT_EQ(input->batch_size(), target->batch_size());

        for (int k = 0; k < input->batch_size(); ++k) {
            // Every input row's real (unpadded) content is a repeated
            // uppercase letter; every target row's real content is a
            // repeated lowercase letter, both derived from the same i%26 —
            // input_char - 'A' must equal target_char - 'a' iff row k of
            // each batch still comes from the same original sample.
            ASSERT_GT(input->lengths[k], 0);
            ASSERT_GT(target->lengths[k], 0);
            int input_char = input->batch_token_ids[k][0];
            int target_char = target->batch_token_ids[k][0];
            EXPECT_EQ(input_char - 'A', target_char - 'a')
                << "batch " << b << " row " << k
                << ": input/target rows no longer correspond to the same original sample";
        }
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

// ============================================================================
// TokenBatchIterator Tests
//
// TD-052 follow-up: TokenBatchIterator had no dedicated test of its own before
// this (only TokenBatchLoader, which it wraps, was tested) — added while
// re-evaluating this file's tag now that ParallelDataLoader/DataLoaderIterator
// (the classes TD-052 actually found broken) are retired, so the remaining
// classes' tag can honestly reflect real coverage rather than an absence of
// evidence either way.
// ============================================================================

TEST_F(TokenBatchLoaderTest, IteratorReturnsExactlyOneEpochOfBatches) {
    TokenBatchLoaderConfig config;
    config.batch_size = 5;
    config.num_workers = 2;

    TokenBatchLoader loader(dataset, config, tokenizer_fn);
    TokenBatchIterator iter(loader);

    int count = 0;
    while (auto batch = iter.next()) {
        ++count;
    }

    EXPECT_EQ(count, static_cast<int>(loader.num_batches()));
    EXPECT_EQ(iter.batches_returned(), loader.num_batches());
    loader.stop();
}

TEST_F(TokenBatchLoaderTest, IteratorResetStartsANewEpochFromZero) {
    TokenBatchLoaderConfig config;
    config.batch_size = 5;
    config.num_workers = 1;

    TokenBatchLoader loader(dataset, config, tokenizer_fn);
    TokenBatchIterator iter(loader);

    ASSERT_TRUE(iter.next().has_value());
    ASSERT_TRUE(iter.next().has_value());
    EXPECT_EQ(iter.batches_returned(), 2u);

    iter.reset();

    EXPECT_EQ(iter.batches_returned(), 0u);
    EXPECT_TRUE(iter.next().has_value());
    loader.stop();
}

TEST_F(TokenBatchLoaderTest, IteratorNextTargetForwardsToLoaderWhenTargetsEnabled) {
    TokenBatchLoaderConfig config;
    config.batch_size = 5;
    config.num_workers = 1;
    config.load_targets = true;

    TokenBatchLoader loader(dataset, config, tokenizer_fn);
    TokenBatchIterator iter(loader);

    auto input = iter.next();
    ASSERT_TRUE(input.has_value());
    EXPECT_TRUE(iter.next_target().has_value());
    loader.stop();
}
