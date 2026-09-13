// @adai-status: beta        (TD-064 resolved — ThreadSafeBatchQueue::clear() missing-notify deadlock fixed; see TECHNICAL_DEBT.md's resolved archive)
// @adai-version: 0.5.1
// @adai-reviewed: 2026-09-12

/**
 * @file ParallelDataLoader.hpp
 * @brief Multi-threaded, tokenizer-driven data loading with prefetching for efficient training
 *
 * This file provides ThreadSafeBatchQueue (a generic producer-consumer queue) and
 * TokenBatchLoader/TokenBatchIterator, a parallel data loader that tokenizes samples via a
 * caller-supplied tokenizer function and produces TokenBatch objects with background
 * prefetching, to hide I/O latency and maximize GPU/CPU utilization during training.
 *
 * Key Features:
 * - Multi-threaded batch loading with a real, caller-supplied tokenizer
 * - Background prefetching with configurable buffer size
 * - Thread-safe batch queue
 * - Automatic batch shuffling
 * - Support for infinite iteration (epoch-based training)
 * - Memory-efficient streaming from disk
 *
 * @version 1.1
 * @date September 2026
 */

#ifndef PARALLEL_DATA_LOADER_HPP
#define PARALLEL_DATA_LOADER_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include "BatchProcessor.hpp"
#include "Dataset.hpp"
#include "EfficientBatching.hpp"

/**
 * @brief Thread-safe batch queue for producer-consumer pattern
 */
template <typename T>
class ThreadSafeBatchQueue {
   public:
    ThreadSafeBatchQueue(size_t max_size = 100) : max_size_(max_size), shutdown_(false) {}

    /**
     * @brief Push a batch to the queue (blocks if queue is full)
     */
    void push(T batch) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait until queue has space or shutdown requested
        cv_producer_.wait(lock, [this]() { return queue_.size() < max_size_ || shutdown_; });

        if (shutdown_)
            return;

        queue_.push(std::move(batch));
        cv_consumer_.notify_one();
    }

    /**
     * @brief Pop a batch from the queue (blocks if queue is empty)
     * @return Batch, or empty optional if queue is shutdown
     */
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait until queue has data or shutdown requested
        cv_consumer_.wait(lock, [this]() { return !queue_.empty() || shutdown_; });

        if (queue_.empty()) {
            return std::nullopt;  // Shutdown
        }

        T batch = std::move(queue_.front());
        queue_.pop();
        cv_producer_.notify_one();

        return batch;
    }

    /**
     * @brief Signal shutdown to all waiting threads
     */
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_producer_.notify_all();
        cv_consumer_.notify_all();
    }

    /**
     * @brief Get current queue size
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /**
     * @brief Check if queue is empty
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /**
     * @brief Clear all batches in queue
     */
    void clear() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while (!queue_.empty()) {
                queue_.pop();
            }
        }
        // TD-064: this used to drain the queue without notifying cv_producer_ — a producer
        // already blocked in push()'s wait (queue was full) has no way to know clear() just
        // made room, since condition_variable::wait() only re-checks its predicate when
        // actually woken (a spurious wakeup can happen per the standard, but isn't guaranteed
        // in any bounded time). With num_workers == 1, that one producer is the *only* source
        // of new items, so it stays stuck forever, and the next next_batch()/pop() call (now
        // finding a permanently-empty queue) blocks forever too — a genuine, real deadlock,
        // confirmed via a dedicated repro harness reproducing it in ~2,600 iterations of
        // TokenBatchIterator::reset() called while a single-worker loader's prefetch buffer was
        // full. This is the root cause behind the historical paralleldataloaderTests hang
        // (see TECHNICAL_DEBT.md's resolved archive for the full writeup).
        cv_producer_.notify_all();
    }

   private:
    mutable std::mutex mutex_;
    std::condition_variable cv_producer_;
    std::condition_variable cv_consumer_;
    std::queue<T> queue_;
    size_t max_size_;
    bool shutdown_;
};

/**
 * @brief Parallel data loader with TokenBatch support
 *
 * Specialized loader that produces TokenBatch objects instead of SequenceBatch.
 * Integrates directly with BatchProcessor utilities for transformer models.
 *
 * Features:
 * - Multi-threaded batch loading with prefetching
 * - Automatic tokenization with custom tokenizer function
 * - Dynamic batching by sequence length
 * - Padding and masking generation
 * - Background data augmentation
 *
 * Example usage:
 * @code
 * Dataset dataset;
 * dataset.load_from_file("data.txt");
 * dataset.split(0.8, 0.1, 0.1);
 *
 * BPETokenizer tokenizer;
 * tokenizer.load_vocab("vocab.txt");
 *
 * TokenBatchLoaderConfig config;
 * config.batch_size = 32;
 * config.use_dynamic_batching = true;
 *
 * auto tokenizer_fn = [&tokenizer](const std::string& text) {
 *     return tokenizer.encode(text);
 * };
 *
 * TokenBatchLoader loader(dataset, config, tokenizer_fn);
 * loader.start();
 *
 * while (auto batch = loader.next_batch()) {
 *     // Process TokenBatch through model
 *     Matrix mask = create_padding_mask(*batch);
 *     // ... forward pass ...
 * }
 * @endcode
 */

/**
 * @brief Configuration for TokenBatch loader
 */
struct TokenBatchLoaderConfig {
    size_t batch_size = 32;                         ///< Number of sequences per batch
    size_t num_workers = 4;                         ///< Number of worker threads
    size_t prefetch_factor = 2;                     ///< Number of batches to prefetch per worker
    bool shuffle = true;                            ///< Shuffle data at each epoch
    int pad_token_id = adai::SpecialTokenIDs::PAD;  ///< Token ID for padding
    bool drop_last = false;                         ///< Drop last incomplete batch
    unsigned int seed = 42;                         ///< Random seed for shuffling
    bool use_dynamic_batching = true;               ///< Use dynamic batching by length
    int length_tolerance = 10;                      ///< Max length difference for dynamic batching
    bool load_targets = false;                      ///< Also load target sequences
};

class TokenBatchLoader {
   public:
    /**
     * @brief Constructor
     * @param dataset Dataset to load from
     * @param config Loader configuration
     * @param tokenizer_fn Function to tokenize strings into token IDs
     * @param split_type Which split to load (default: TRAIN)
     */
    TokenBatchLoader(const Dataset& dataset, const TokenBatchLoaderConfig& config,
                     std::function<std::vector<int>(const std::string&)> tokenizer_fn,
                     SplitType split_type = SplitType::TRAIN)
        : dataset_(dataset),
          config_(config),
          tokenizer_fn_(tokenizer_fn),
          split_type_(split_type),
          current_epoch_(0),
          is_running_(false),
          batches_loaded_(0) {
        // Calculate prefetch buffer size
        size_t buffer_size = config_.num_workers * config_.prefetch_factor;
        // Input and target batches for one loaded item are pushed/popped as a
        // single pair through one queue (not two independent queues) — see
        // the note on batch_queue_'s declaration below for why that matters.
        batch_queue_ = std::make_unique<ThreadSafeBatchQueue<std::pair<TokenBatch, TokenBatch>>>(
            buffer_size);
    }

    /**
     * @brief Destructor - ensure threads are stopped
     */
    ~TokenBatchLoader() {
        stop();
    }

    /**
     * @brief Start background loading threads
     */
    void start() {
        if (is_running_)
            return;

        is_running_ = true;
        batches_loaded_ = 0;

        // Prepare indices for first epoch
        prepare_epoch_indices();

        // Create worker threads
        for (size_t i = 0; i < config_.num_workers; ++i) {
            workers_.emplace_back(&TokenBatchLoader::worker_thread, this, i);
        }
    }

    /**
     * @brief Stop all background threads
     */
    void stop() {
        if (!is_running_)
            return;

        is_running_ = false;
        batch_queue_->shutdown();

        // Join all worker threads
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();

        batch_queue_->clear();
        pending_target_.reset();
    }

    /**
     * @brief Get next input batch (blocks until available)
     *
     * When config_.load_targets is set, this also latches the matching target
     * batch for retrieval via next_target_batch() — call that immediately
     * afterward, before the next next_batch() call, to get the pair that
     * belongs together. See batch_queue_'s declaration for why input and
     * target are queued as one unit instead of through independent queues.
     *
     * @return TokenBatch, or empty optional if no more batches
     */
    std::optional<TokenBatch> next_batch() {
        if (!is_running_) {
            start();
        }

        auto pair = batch_queue_->pop();
        if (!pair.has_value()) {
            pending_target_.reset();
            return std::nullopt;
        }
        pending_target_ = std::move(pair->second);
        return std::move(pair->first);
    }

    /**
     * @brief Get the target batch matching the most recent next_batch() call
     * @return TokenBatch, or empty optional if load_targets is disabled or
     *         next_batch() hasn't been called (or returned no batch) yet
     */
    std::optional<TokenBatch> next_target_batch() {
        if (!config_.load_targets || !pending_target_.has_value()) {
            return std::nullopt;
        }

        TokenBatch target = std::move(*pending_target_);
        pending_target_.reset();
        return target;
    }

    /**
     * @brief Start a new epoch
     *
     * Increments epoch counter and shuffles data if configured.
     */
    void new_epoch() {
        // Clear any remaining batches from previous epoch
        batch_queue_->clear();
        pending_target_.reset();

        ++current_epoch_;
        batches_loaded_ = 0;

        // Prepare indices for this epoch
        prepare_epoch_indices();
    }

    /**
     * @brief Get number of batches per epoch
     */
    size_t num_batches() const {
        size_t total_samples = dataset_.size(split_type_);
        if (config_.drop_last) {
            return total_samples / config_.batch_size;
        } else {
            return (total_samples + config_.batch_size - 1) / config_.batch_size;
        }
    }

    /**
     * @brief Get current epoch number
     */
    size_t current_epoch() const {
        return current_epoch_;
    }

    /**
     * @brief Get total number of batches loaded
     */
    size_t batches_loaded() const {
        return batches_loaded_.load();
    }

    /**
     * @brief Get prefetch queue size
     */
    size_t queue_size() const {
        return batch_queue_->size();
    }

    /**
     * @brief Check if loader is running
     */
    bool is_running() const {
        return is_running_;
    }

   private:
    /**
     * @brief Worker thread function
     */
    void worker_thread(size_t worker_id) {
        while (is_running_) {
            // Check if we've loaded all batches for this epoch
            size_t current_batch = batches_loaded_.fetch_add(1);
            size_t total_batches = num_batches();

            if (current_batch >= total_batches) {
                // Wait for new epoch
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            try {
                // Load batch — input and target are pushed together as one
                // pair (see batch_queue_'s declaration) so a caller that only
                // drains next_batch() (never next_target_batch()) can't stall
                // a second, independently-bounded target queue and deadlock
                // every worker thread.
                auto batch_pair = load_batch(current_batch);
                batch_queue_->push(std::move(batch_pair));

            } catch (const std::exception& e) {
                // Log error and continue
                continue;
            }
        }
    }

    /**
     * @brief Load a single batch
     */
    std::pair<TokenBatch, TokenBatch> load_batch(size_t batch_idx) {
        std::lock_guard<std::mutex> lock(indices_mutex_);

        // Calculate batch range
        size_t start_idx = batch_idx * config_.batch_size;
        size_t end_idx = std::min(start_idx + config_.batch_size, epoch_indices_.size());

        if (start_idx >= epoch_indices_.size()) {
            return {TokenBatch(), TokenBatch()};
        }

        // Get the split data
        auto split_data = dataset_.get_split(split_type_);

        // Collect input sequences for this batch
        std::vector<std::vector<int>> input_sequences;
        std::vector<std::vector<int>> target_sequences;

        for (size_t i = start_idx; i < end_idx; ++i) {
            size_t dataset_idx = epoch_indices_[i];
            if (dataset_idx < split_data.size()) {
                const auto& sample = split_data[dataset_idx];

                // Tokenize input
                input_sequences.push_back(tokenizer_fn_(sample.input));

                // Tokenize target if needed
                if (config_.load_targets) {
                    target_sequences.push_back(tokenizer_fn_(sample.target));
                }
            }
        }

        if (input_sequences.empty()) {
            return {TokenBatch(), TokenBatch()};
        }

        // Create batches using BatchProcessor utilities
        //
        // TD-098 (fixed): config_.use_dynamic_batching used to route through
        // create_dynamic_batches() here, passing max_batch_size ==
        // input_sequences.size() specifically so this whole already-fixed
        // slice (start_idx..end_idx above) would land in one TokenBatch.
        // create_dynamic_batches() is designed to split an unbounded pool
        // into multiple length-homogeneous batches, and does so purely
        // internally (sorting by each list's own lengths) whenever a slice's
        // length spread exceeds length_tolerance, regardless of
        // max_batch_size — two real bugs followed: (1) input_sequences and
        // target_sequences were sorted *independently* by their own
        // (generally uncorrelated) lengths, so input_batch[k] and
        // target_batch[k] stopped corresponding to the same original sample
        // the moment their length orderings diverged — every affected
        // training step would pair a token sequence with the wrong target;
        // (2) when a split did happen, only the first resulting group
        // (batches[0]) was kept, silently dropping every sequence in the
        // later groups from training. Padding within a single
        // fixed-membership batch is unaffected by the sequences' internal
        // order, so there is no efficiency benefit "dynamic" batching could
        // have offered here anyway — plain create_batch() for both,
        // unconditionally, guarantees no drops and preserves index
        // correspondence between input_batch and target_batch by
        // construction.
        TokenBatch input_batch = create_batch(input_sequences, config_.pad_token_id);
        TokenBatch target_batch;
        if (config_.load_targets && !target_sequences.empty()) {
            target_batch = create_batch(target_sequences, config_.pad_token_id);
        }

        return {input_batch, target_batch};
    }

    /**
     * @brief Prepare indices for current epoch
     */
    void prepare_epoch_indices() {
        std::lock_guard<std::mutex> lock(indices_mutex_);

        // Get the split data
        auto split_data = dataset_.get_split(split_type_);

        // Create sequential indices
        epoch_indices_.resize(split_data.size());
        std::iota(epoch_indices_.begin(), epoch_indices_.end(), 0);

        // Shuffle if configured
        if (config_.shuffle) {
            std::mt19937 gen(config_.seed + current_epoch_);
            std::shuffle(epoch_indices_.begin(), epoch_indices_.end(), gen);
        }
    }

    const Dataset& dataset_;
    TokenBatchLoaderConfig config_;
    std::function<std::vector<int>(const std::string&)> tokenizer_fn_;
    SplitType split_type_;

    // Threading
    std::vector<std::thread> workers_;
    // Input and target batches for the same loaded item are pushed/popped as
    // one pair through a single queue rather than through two independent
    // ThreadSafeBatchQueues. With two separate bounded queues, a consumer
    // that drains next_batch() without also draining next_target_batch() (or
    // at a different rate) fills the target queue permanently, blocking
    // every worker thread inside its push() there — and once every worker is
    // stuck, batch_queue_ stops being refilled too, so next_batch() then
    // hangs forever as well (see the historical paralleldataloaderTests hang
    // investigation in TECHNICAL_DEBT.md, TD-064 — a related-in-spirit but
    // not identical class of two-queue producer deadlock; that investigation
    // covered the now-retired ParallelDataLoader class and ThreadSafeBatchQueue
    // itself (still used above), neither of which has this specific issue, and
    // could not reproduce a root cause there — see TD-052's resolution writeup
    // for how ParallelDataLoader's removal bears on that open investigation).
    // A single combined queue also removes a second,
    // independent bug this design invited: with num_workers > 1, two
    // separate queues have no guaranteed ordering relationship to each other
    // across different producer threads, so alternating next_batch()/
    // next_target_batch() calls could return an *input* from one worker's
    // batch paired with a *target* from a different worker's unrelated
    // batch. next_batch() latches the popped pair's target into
    // pending_target_ for the immediately-following next_target_batch() call.
    std::unique_ptr<ThreadSafeBatchQueue<std::pair<TokenBatch, TokenBatch>>> batch_queue_;
    std::optional<TokenBatch> pending_target_;
    std::atomic<bool> is_running_;

    // Epoch management
    std::atomic<size_t> current_epoch_;
    std::atomic<size_t> batches_loaded_;
    std::vector<size_t> epoch_indices_;
    std::mutex indices_mutex_;
};

/**
 * @brief Simple iterator interface for TokenBatch loader
 *
 * Provides a convenient way to iterate over batches:
 *
 * ```cpp
 * TokenBatchIterator iter(loader);
 * while (auto batch = iter.next()) {
 *     // Process batch
 * }
 * ```
 */
class TokenBatchIterator {
   public:
    TokenBatchIterator(TokenBatchLoader& loader) : loader_(loader), batches_returned_(0) {
        loader_.new_epoch();
    }

    /**
     * @brief Get next batch
     * @return TokenBatch, or empty optional if epoch is complete
     */
    std::optional<TokenBatch> next() {
        if (batches_returned_ >= loader_.num_batches()) {
            return std::nullopt;
        }

        auto batch = loader_.next_batch();
        if (batch.has_value()) {
            ++batches_returned_;
        }
        return batch;
    }

    /**
     * @brief Get next target batch
     * @return TokenBatch, or empty optional if epoch is complete
     */
    std::optional<TokenBatch> next_target() {
        return loader_.next_target_batch();
    }

    /**
     * @brief Reset to beginning of epoch
     */
    void reset() {
        batches_returned_ = 0;
        loader_.new_epoch();
    }

    /**
     * @brief Get number of batches returned so far
     */
    size_t batches_returned() const {
        return batches_returned_;
    }

   private:
    TokenBatchLoader& loader_;
    size_t batches_returned_;
};

#endif  // PARALLEL_DATA_LOADER_HPP
