/**
 * @file ParallelDataLoader.hpp
 * @brief Multi-threaded data loading with prefetching for efficient training
 * 
 * This file provides utilities for parallel data loading with background prefetching
 * to hide I/O latency and maximize GPU/CPU utilization during training.
 * 
 * Key Features:
 * - Multi-threaded batch loading
 * - Background prefetching with configurable buffer size
 * - Thread-safe batch queue
 * - Automatic batch shuffling
 * - Support for infinite iteration (epoch-based training)
 * - Memory-efficient streaming from disk
 * 
 * @version 1.0
 * @date January 2026
 */

#ifndef PARALLEL_DATA_LOADER_HPP
#define PARALLEL_DATA_LOADER_HPP

#include "EfficientBatching.hpp"
#include "Dataset.hpp"
#include "BatchProcessor.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <memory>
#include <chrono>
#include <optional>

/**
 * @brief Configuration for parallel data loader
 */
struct DataLoaderConfig {
    size_t batch_size = 32;                    ///< Number of sequences per batch
    size_t num_workers = 4;                    ///< Number of worker threads
    size_t prefetch_factor = 2;                ///< Number of batches to prefetch per worker
    bool shuffle = true;                       ///< Shuffle data at each epoch
    int pad_token_id = adai::SpecialTokenIDs::PAD;  ///< Token ID for padding
    PaddingStrategy padding_strategy = PaddingStrategy::RIGHT;  ///< Padding strategy
    bool drop_last = false;                    ///< Drop last incomplete batch
    unsigned int seed = 42;                    ///< Random seed for shuffling
    bool use_dynamic_batching = true;          ///< Use dynamic batching by length
    bool use_bucketing = false;                ///< Use bucketing strategy
    BucketConfig bucket_config;                ///< Configuration for bucketing
    AugmentationConfig augmentation_config;    ///< Configuration for data augmentation
    bool use_token_batch = false;              ///< Use TokenBatch instead of SequenceBatch
    int length_tolerance = 10;                 ///< Max length difference for dynamic batching
};

/**
 * @brief Thread-safe batch queue for producer-consumer pattern
 */
template<typename T>
class ThreadSafeBatchQueue {
public:
    ThreadSafeBatchQueue(size_t max_size = 100) : max_size_(max_size), shutdown_(false) {}
    
    /**
     * @brief Push a batch to the queue (blocks if queue is full)
     */
    void push(T batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait until queue has space or shutdown requested
        cv_producer_.wait(lock, [this]() {
            return queue_.size() < max_size_ || shutdown_;
        });
        
        if (shutdown_) return;
        
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
        cv_consumer_.wait(lock, [this]() {
            return !queue_.empty() || shutdown_;
        });
        
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
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
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
 * @brief Parallel data loader with background prefetching
 * 
 * Loads batches in parallel using multiple worker threads and maintains
 * a prefetch buffer to hide I/O latency.
 */
class ParallelDataLoader {
public:
    /**
     * @brief Constructor
     * @param dataset Dataset to load from
     * @param config Data loader configuration
     */
    ParallelDataLoader(const Dataset& dataset, const DataLoaderConfig& config)
        : dataset_(dataset)
        , config_(config)
        , current_epoch_(0)
        , is_running_(false)
        , batches_loaded_(0)
    {
        // Calculate prefetch buffer size
        size_t buffer_size = config_.num_workers * config_.prefetch_factor;
        batch_queue_ = std::make_unique<ThreadSafeBatchQueue<SequenceBatch>>(buffer_size);
    }
    
    /**
     * @brief Destructor - ensure threads are stopped
     */
    ~ParallelDataLoader() {
        stop();
    }
    
    /**
     * @brief Start background loading threads
     */
    void start() {
        if (is_running_) return;
        
        is_running_ = true;
        batches_loaded_ = 0;
        
        // Create worker threads
        for (size_t i = 0; i < config_.num_workers; ++i) {
            workers_.emplace_back(&ParallelDataLoader::worker_thread, this, i);
        }
    }
    
    /**
     * @brief Stop all background threads
     */
    void stop() {
        if (!is_running_) return;
        
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
    }
    
    /**
     * @brief Get next batch (blocks until available)
     * @return Batch, or empty optional if no more batches
     */
    std::optional<SequenceBatch> next_batch() {
        if (!is_running_) {
            start();
        }
        
        return batch_queue_->pop();
    }
    
    /**
     * @brief Start a new epoch
     * 
     * Increments epoch counter and shuffles data if configured.
     */
    void new_epoch() {
        // Clear any remaining batches from previous epoch
        batch_queue_->clear();
        
        ++current_epoch_;
        batches_loaded_ = 0;
        
        // Prepare indices for this epoch
        prepare_epoch_indices();
    }
    
    /**
     * @brief Get number of batches per epoch
     */
    size_t num_batches() const {
        // Use train split size
        size_t total_samples = dataset_.size(SplitType::TRAIN);
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
                // Load batch
                SequenceBatch batch = load_batch(current_batch);
                
                // Push to queue (blocks if queue is full)
                batch_queue_->push(std::move(batch));
                
            } catch (const std::exception& e) {
                // Log error and continue
                // In production, use proper logging
                continue;
            }
        }
    }
    
    /**
     * @brief Load a single batch
     */
    SequenceBatch load_batch(size_t batch_idx) {
        std::lock_guard<std::mutex> lock(indices_mutex_);
        
        // Calculate batch range
        size_t start_idx = batch_idx * config_.batch_size;
        size_t end_idx = std::min(start_idx + config_.batch_size, epoch_indices_.size());
        
        if (start_idx >= epoch_indices_.size()) {
            return SequenceBatch();  // Empty batch
        }
        
        // Get the split data based on indices
        auto split_data = dataset_.get_split(SplitType::TRAIN);
        
        // Collect sequences for this batch
        // Note: For now we'll create dummy token sequences from the text
        // In a real implementation, this should use a tokenizer
        std::vector<std::vector<int>> batch_sequences;
        for (size_t i = start_idx; i < end_idx; ++i) {
            size_t dataset_idx = epoch_indices_[i];
            if (dataset_idx < split_data.size()) {
                const auto& sample = split_data[dataset_idx];
                
                // Create simple token sequence (char codes for demonstration)
                std::vector<int> tokens;
                for (char c : sample.input) {
                    tokens.push_back(static_cast<int>(static_cast<unsigned char>(c)));
                }
                // Ensure minimum length
                if (tokens.empty()) {
                    tokens.push_back(1);  // Dummy token
                }
                batch_sequences.push_back(tokens);
            }
        }
        
        if (batch_sequences.empty()) {
            return SequenceBatch();
        }
        
        // Apply augmentation if configured
        if (config_.augmentation_config.enable_token_dropout ||
            config_.augmentation_config.enable_token_masking ||
            config_.augmentation_config.enable_sequence_shuffle) {
            EfficientBatching::apply_augmentation(batch_sequences, config_.augmentation_config);
            
            // After augmentation, ensure sequences aren't empty
            batch_sequences.erase(
                std::remove_if(batch_sequences.begin(), batch_sequences.end(),
                    [](const std::vector<int>& seq) { return seq.empty(); }),
                batch_sequences.end()
            );
            
            if (batch_sequences.empty()) {
                return SequenceBatch();
            }
        }
        
        // Create batch with appropriate strategy
        if (config_.use_bucketing) {
            auto batches = EfficientBatching::create_bucketed_batches(
                batch_sequences, config_.bucket_config, 
                config_.pad_token_id, config_.padding_strategy
            );
            return batches.empty() ? SequenceBatch() : batches[0];
        } else {
            auto batches = EfficientBatching::create_dynamic_batches(
                batch_sequences, batch_sequences.size(),
                config_.pad_token_id, config_.padding_strategy,
                config_.use_dynamic_batching
            );
            return batches.empty() ? SequenceBatch() : batches[0];
        }
    }
    
    /**
     * @brief Prepare indices for current epoch
     */
    void prepare_epoch_indices() {
        std::lock_guard<std::mutex> lock(indices_mutex_);
        
        // Get the train split data
        auto train_data = dataset_.get_split(SplitType::TRAIN);
        
        // Create sequential indices based on train data size
        epoch_indices_.resize(train_data.size());
        std::iota(epoch_indices_.begin(), epoch_indices_.end(), 0);
        
        // Shuffle if configured
        if (config_.shuffle) {
            std::mt19937 gen(config_.seed + current_epoch_);
            std::shuffle(epoch_indices_.begin(), epoch_indices_.end(), gen);
        }
    }

    Dataset dataset_;
    DataLoaderConfig config_;
    
    // Threading
    std::vector<std::thread> workers_;
    std::unique_ptr<ThreadSafeBatchQueue<SequenceBatch>> batch_queue_;
    std::atomic<bool> is_running_;
    
    // Epoch management
    std::atomic<size_t> current_epoch_;
    std::atomic<size_t> batches_loaded_;
    std::vector<size_t> epoch_indices_;
    std::mutex indices_mutex_;
};

/**
 * @brief Simple iterator interface for data loader
 * 
 * Provides a convenient way to iterate over batches:
 * 
 * ```cpp
 * DataLoaderIterator iter(loader);
 * while (auto batch = iter.next()) {
 *     // Process batch
 * }
 * ```
 */
class DataLoaderIterator {
public:
    DataLoaderIterator(ParallelDataLoader& loader) 
        : loader_(loader), batches_returned_(0) {
        loader_.new_epoch();
    }
    
    /**
     * @brief Get next batch
     * @return Batch, or empty optional if epoch is complete
     */
    std::optional<SequenceBatch> next() {
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
    ParallelDataLoader& loader_;
    size_t batches_returned_;
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
    size_t batch_size = 32;                    ///< Number of sequences per batch
    size_t num_workers = 4;                    ///< Number of worker threads
    size_t prefetch_factor = 2;                ///< Number of batches to prefetch per worker
    bool shuffle = true;                       ///< Shuffle data at each epoch
    int pad_token_id = adai::SpecialTokenIDs::PAD;  ///< Token ID for padding
    bool drop_last = false;                    ///< Drop last incomplete batch
    unsigned int seed = 42;                    ///< Random seed for shuffling
    bool use_dynamic_batching = true;          ///< Use dynamic batching by length
    int length_tolerance = 10;                 ///< Max length difference for dynamic batching
    bool load_targets = false;                 ///< Also load target sequences
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
    TokenBatchLoader(
        const Dataset& dataset,
        const TokenBatchLoaderConfig& config,
        std::function<std::vector<int>(const std::string&)> tokenizer_fn,
        SplitType split_type = SplitType::TRAIN)
        : dataset_(dataset)
        , config_(config)
        , tokenizer_fn_(tokenizer_fn)
        , split_type_(split_type)
        , current_epoch_(0)
        , is_running_(false)
        , batches_loaded_(0)
    {
        // Calculate prefetch buffer size
        size_t buffer_size = config_.num_workers * config_.prefetch_factor;
        batch_queue_ = std::make_unique<ThreadSafeBatchQueue<TokenBatch>>(buffer_size);
        
        if (config_.load_targets) {
            target_queue_ = std::make_unique<ThreadSafeBatchQueue<TokenBatch>>(buffer_size);
        }
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
        if (is_running_) return;
        
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
        if (!is_running_) return;
        
        is_running_ = false;
        batch_queue_->shutdown();
        if (target_queue_) {
            target_queue_->shutdown();
        }
        
        // Join all worker threads
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
        
        batch_queue_->clear();
        if (target_queue_) {
            target_queue_->clear();
        }
    }
    
    /**
     * @brief Get next input batch (blocks until available)
     * @return TokenBatch, or empty optional if no more batches
     */
    std::optional<TokenBatch> next_batch() {
        if (!is_running_) {
            start();
        }
        
        return batch_queue_->pop();
    }
    
    /**
     * @brief Get next target batch (blocks until available)
     * @return TokenBatch, or empty optional if no more batches
     */
    std::optional<TokenBatch> next_target_batch() {
        if (!config_.load_targets || !target_queue_) {
            return std::nullopt;
        }
        
        return target_queue_->pop();
    }
    
    /**
     * @brief Start a new epoch
     * 
     * Increments epoch counter and shuffles data if configured.
     */
    void new_epoch() {
        // Clear any remaining batches from previous epoch
        batch_queue_->clear();
        if (target_queue_) {
            target_queue_->clear();
        }
        
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
                // Load batch
                auto [input_batch, target_batch] = load_batch(current_batch);
                
                // Push to queue (blocks if queue is full)
                batch_queue_->push(std::move(input_batch));
                
                if (config_.load_targets && target_queue_) {
                    target_queue_->push(std::move(target_batch));
                }
                
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
        TokenBatch input_batch;
        TokenBatch target_batch;
        
        if (config_.use_dynamic_batching) {
            // Use dynamic batching by length
            auto batches = create_dynamic_batches(
                input_sequences, 
                input_sequences.size(),  // All in one batch since we already sized it
                config_.length_tolerance,
                config_.pad_token_id
            );
            input_batch = batches.empty() ? TokenBatch() : batches[0];
            
            if (config_.load_targets && !target_sequences.empty()) {
                auto target_batches = create_dynamic_batches(
                    target_sequences,
                    target_sequences.size(),
                    config_.length_tolerance,
                    config_.pad_token_id
                );
                target_batch = target_batches.empty() ? TokenBatch() : target_batches[0];
            }
        } else {
            // Simple batching with padding
            input_batch = create_batch(input_sequences, config_.pad_token_id);
            
            if (config_.load_targets && !target_sequences.empty()) {
                target_batch = create_batch(target_sequences, config_.pad_token_id);
            }
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
    std::unique_ptr<ThreadSafeBatchQueue<TokenBatch>> batch_queue_;
    std::unique_ptr<ThreadSafeBatchQueue<TokenBatch>> target_queue_;
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
    TokenBatchIterator(TokenBatchLoader& loader) 
        : loader_(loader), batches_returned_(0) {
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

#endif // PARALLEL_DATA_LOADER_HPP
