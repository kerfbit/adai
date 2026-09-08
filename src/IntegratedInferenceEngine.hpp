// @adai-status: beta        (capped by TD-038 — tested but not wired into any shipped binary)
// @adai-version: 0.7.0
// @adai-reviewed: 2026-09-07

/**
 * @file IntegratedInferenceEngine.hpp
 * @brief Fully integrated inference engine combining all parallel optimizations
 *
 * This engine integrates all Priority 1-5 optimizations for maximum performance:
 *
 * Priority 1: OpenMP CPU Parallelization (4.21x on matrix ops)
 * Priority 2: Parallel Data Augmentation (3.82x on preprocessing)
 * Priority 3: Batched Inference (27.80x throughput at batch=32)
 * Priority 4: Attention Head Parallelism (1.3-2.0x on attention)
 * Priority 5: Pipeline Parallelism (1.24x throughput improvement)
 *
 * Combined Architecture:
 *
 *   Multiple Clients
 *        ↓
 *   [Request Queue] ──> Batching Layer (Priority 3) ──> Dynamic Batches
 *        ↓
 *   Pipeline Stage 1: Encoder (Priority 5)
 *     - Uses OpenMP matrix ops (Priority 1)
 *     - Parallel attention heads (Priority 4)
 *        ↓
 *   Pipeline Stage 2: Decoder (Priority 5)
 *     - Uses OpenMP matrix ops (Priority 1)
 *     - Parallel attention heads (Priority 4)
 *        ↓
 *   [Results Distribution] ──> Clients
 *
 * Expected Combined Speedup:
 * - Batching alone: 27.80x (baseline)
 * - + Pipeline: 27.80x × 1.24x = 34.47x
 * - + OpenMP in critical paths: Additional 1.5-2x on compute
 * - + Parallel attention: Additional 1.2-1.5x on attention layers
 *
 * Target: 50-100x improvement over naive sequential processing
 *
 * @version 1.0
 * @date January 2026
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "BPETokenizer.hpp"
#include "BatchedInferenceEngine.hpp"
#include "Decoder.hpp"
#include "LanguageModelHead.hpp"
#include "Matrix.hpp"
#include "PipelineInferenceEngine.hpp"
#include "encoder.hpp"

// ============================================================================
// Integrated Configuration
// ============================================================================

/**
 * @brief Unified configuration combining all optimization strategies
 */
struct IntegratedInferenceConfig {
    // Batching configuration (Priority 3)
    size_t max_batch_size = 32;       ///< Maximum requests per batch
    int batch_timeout_ms = 50;        ///< Max wait time to collect batch
    int max_tokens_per_batch = 4096;  ///< Max total tokens per batch

    // Pipeline configuration (Priority 5)
    bool enable_pipeline = true;     ///< Enable two-stage pipeline
    size_t encoder_queue_size = 50;  ///< Encoder input queue capacity
    size_t decoder_queue_size = 50;  ///< Encoder→decoder queue capacity
    int encoder_timeout_ms = 50;     ///< Encoder stage timeout
    int decoder_timeout_ms = 50;     ///< Decoder stage timeout

    // Parallel processing (Priority 1, 4)
    bool use_openmp = true;          ///< Enable OpenMP parallelization
    bool parallel_attention = true;  ///< Use parallel attention heads
    int num_threads = 0;             ///< OpenMP threads (0 = auto)

    // Request handling
    int max_queue_size = 1000;  ///< Maximum queued requests
    bool enable_stats = true;   ///< Track performance statistics

    // Generation settings
    int default_max_length = 100;                ///< Default generation length
    std::string generation_strategy = "greedy";  ///< Default decoding strategy
};

/**
 * @brief Comprehensive statistics tracking all optimizations
 */
struct IntegratedInferenceStats {
    // Request-level statistics
    uint64_t total_requests = 0;
    uint64_t total_batches = 0;
    uint64_t total_tokens_generated = 0;

    // Batching statistics (Priority 3)
    double avg_batch_size = 0.0;
    double batch_utilization = 0.0;  ///< Actual / max batch size

    // Pipeline statistics (Priority 5)
    uint64_t encoder_batches_processed = 0;
    uint64_t decoder_batches_processed = 0;
    double avg_encoder_time_ms = 0.0;
    double avg_decoder_time_ms = 0.0;
    double pipeline_efficiency = 0.0;  ///< Speedup from pipelining

    // Parallelism statistics (Priority 1, 4)
    double avg_openmp_speedup = 0.0;
    double avg_attention_speedup = 0.0;

    // Overall performance
    double avg_latency_ms = 0.0;
    double throughput_req_per_sec = 0.0;
    double throughput_tokens_per_sec = 0.0;
    double cumulative_speedup = 1.0;  ///< Total speedup vs sequential

    // Queue health
    double avg_queue_depth = 0.0;
    uint64_t requests_dropped = 0;

    std::chrono::steady_clock::time_point start_time;

    void reset() {
        *this = IntegratedInferenceStats();
        start_time = std::chrono::steady_clock::now();
    }

    void update_throughput() {
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
        if (elapsed_sec > 0) {
            throughput_req_per_sec = total_requests / elapsed_sec;
            throughput_tokens_per_sec = total_tokens_generated / elapsed_sec;
        }
    }
};

// ============================================================================
// Integrated Request Structure
// ============================================================================

/**
 * @brief Request with timing and metadata for comprehensive tracking
 */
struct IntegratedRequest {
    std::string input_text;
    int max_length = 0;
    std::string strategy;

    std::promise<std::string> result_promise;
    std::chrono::steady_clock::time_point submit_time;
    std::chrono::steady_clock::time_point batch_time;
    std::chrono::steady_clock::time_point encoder_start;
    std::chrono::steady_clock::time_point decoder_start;

    // Performance tracking
    size_t input_tokens = 0;
    size_t output_tokens = 0;
    size_t batch_id = 0;

    IntegratedRequest() = default;

    IntegratedRequest(IntegratedRequest&& other) noexcept
        : input_text(std::move(other.input_text)),
          max_length(other.max_length),
          strategy(std::move(other.strategy)),
          result_promise(std::move(other.result_promise)),
          submit_time(other.submit_time),
          batch_time(other.batch_time),
          encoder_start(other.encoder_start),
          decoder_start(other.decoder_start),
          input_tokens(other.input_tokens),
          output_tokens(other.output_tokens),
          batch_id(other.batch_id) {}

    IntegratedRequest& operator=(IntegratedRequest&& other) noexcept {
        if (this != &other) {
            input_text = std::move(other.input_text);
            max_length = other.max_length;
            strategy = std::move(other.strategy);
            result_promise = std::move(other.result_promise);
            submit_time = other.submit_time;
            batch_time = other.batch_time;
            encoder_start = other.encoder_start;
            decoder_start = other.decoder_start;
            input_tokens = other.input_tokens;
            output_tokens = other.output_tokens;
            batch_id = other.batch_id;
        }
        return *this;
    }
};

/**
 * @brief Batch of requests ready for pipeline processing
 */
struct IntegratedBatch {
    std::vector<IntegratedRequest> requests;
    std::vector<std::vector<int>> tokenized_inputs;
    size_t batch_id = 0;

    IntegratedBatch() = default;

    IntegratedBatch(IntegratedBatch&& other) noexcept
        : requests(std::move(other.requests)),
          tokenized_inputs(std::move(other.tokenized_inputs)),
          batch_id(other.batch_id) {}

    IntegratedBatch& operator=(IntegratedBatch&& other) noexcept {
        if (this != &other) {
            requests = std::move(other.requests);
            tokenized_inputs = std::move(other.tokenized_inputs);
            batch_id = other.batch_id;
        }
        return *this;
    }
};

/**
 * @brief Encoder output with metadata for decoder stage
 */
struct IntegratedEncoderOutput {
    std::vector<Matrix> encoder_outputs;  ///< Per-request encoder outputs
    std::vector<IntegratedRequest> requests;
    size_t batch_id = 0;

    IntegratedEncoderOutput() = default;

    IntegratedEncoderOutput(IntegratedEncoderOutput&& other) noexcept
        : encoder_outputs(std::move(other.encoder_outputs)),
          requests(std::move(other.requests)),
          batch_id(other.batch_id) {}

    IntegratedEncoderOutput& operator=(IntegratedEncoderOutput&& other) noexcept {
        if (this != &other) {
            encoder_outputs = std::move(other.encoder_outputs);
            requests = std::move(other.requests);
            batch_id = other.batch_id;
        }
        return *this;
    }
};

// ============================================================================
// Integrated Inference Engine
// ============================================================================

/**
 * @brief Fully integrated inference engine with all parallel optimizations
 *
 * This class combines:
 * - Continuous batching for throughput (Priority 3)
 * - Pipeline parallelism for overlap (Priority 5)
 * - OpenMP for compute parallelism (Priority 1)
 * - Parallel attention heads (Priority 4)
 *
 * Thread Architecture:
 * - Main thread: Request submission and result collection
 * - Batcher thread: Groups requests into optimal batches
 * - Encoder thread: Processes batches through encoder (with OpenMP/parallel attention)
 * - Decoder thread: Processes encoder outputs through decoder (with OpenMP/parallel attention)
 */
class IntegratedInferenceEngine {
   private:
    // Model components
    LLMEncoder* encoder_;
    LLMDecoder* decoder_;
    LanguageModelHead* lm_head_;
    BPETokenizer* tokenizer_;

    // Configuration
    IntegratedInferenceConfig config_;

    // Request batching queues
    ThreadSafeQueue<IntegratedRequest> request_queue_;
    ThreadSafeQueue<IntegratedBatch> batch_queue_;
    ThreadSafeQueue<IntegratedEncoderOutput> encoder_to_decoder_queue_;

    // Worker threads
    std::thread batcher_thread_;
    std::thread encoder_thread_;
    std::thread decoder_thread_;

    // Thread control
    std::atomic<bool> shutdown_;

    // Statistics
    IntegratedInferenceStats stats_;
    mutable std::mutex stats_mutex_;
    std::atomic<uint64_t> next_batch_id_;

    // ========================================================================
    // Worker Thread Functions
    // ========================================================================

    /**
     * Batcher thread: Collects requests and forms optimal batches
     * Implements Priority 3 (batched inference)
     */
    void batcher_worker() {
        std::vector<IntegratedRequest> pending_requests;
        auto batch_deadline = std::chrono::steady_clock::now();

        while (!shutdown_.load()) {
            IntegratedRequest request;

            // Try to get a request with timeout
            bool got_request = request_queue_.try_pop(request, config_.batch_timeout_ms);

            if (got_request) {
                request.batch_time = std::chrono::steady_clock::now();
                pending_requests.push_back(std::move(request));
            }

            // Check if we should emit a batch
            bool should_emit = false;

            if (pending_requests.size() >= config_.max_batch_size) {
                should_emit = true;  // Batch full
            } else if (!pending_requests.empty()) {
                auto now = std::chrono::steady_clock::now();
                if (now >= batch_deadline) {
                    should_emit = true;  // Timeout
                }
            }

            if (should_emit && !pending_requests.empty()) {
                // Create batch
                IntegratedBatch batch;
                batch.batch_id = next_batch_id_.fetch_add(1);
                batch.requests = std::move(pending_requests);

                // Tokenize inputs
                for (const auto& req : batch.requests) {
                    batch.tokenized_inputs.push_back(tokenizer_->encode(req.input_text));
                }

                // Submit to encoder queue
                batch_queue_.push(std::move(batch));

                // Reset for next batch
                pending_requests.clear();
                batch_deadline = std::chrono::steady_clock::now() +
                                 std::chrono::milliseconds(config_.batch_timeout_ms);

                // Update stats
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.total_batches++;
                }
            }
        }
    }

    /**
     * Encoder thread: Processes batches through encoder
     * Implements Priority 5 (pipeline stage 1), uses Priority 1 & 4 internally
     */
    void encoder_worker() {
        while (!shutdown_.load()) {
            IntegratedBatch batch;

            if (!batch_queue_.try_pop(batch, config_.encoder_timeout_ms)) {
                continue;
            }

            auto encoder_start = std::chrono::steady_clock::now();

            // Mark timing for all requests
            for (auto& req : batch.requests) {
                req.encoder_start = encoder_start;
            }

            // Process each request through encoder individually
            // Note: LLMEncoder will use OpenMP (Priority 1) and parallel attention (Priority 4)
            std::vector<Matrix> encoder_outputs;
            for (const auto& tokens : batch.tokenized_inputs) {
                // Create square attention mask (all ones = full attention, no padding masking)
                Matrix att_mask(static_cast<int>(tokens.size()), static_cast<int>(tokens.size()));
                att_mask.fill(1.0f);
                encoder_outputs.push_back(encoder_->encode_with_mask(tokens, att_mask));
            }

            auto encoder_end = std::chrono::steady_clock::now();
            double encoder_time_ms =
                std::chrono::duration<double, std::milli>(encoder_end - encoder_start).count();

            // Create output for decoder
            IntegratedEncoderOutput output;
            output.encoder_outputs = std::move(encoder_outputs);
            output.requests = std::move(batch.requests);
            output.batch_id = batch.batch_id;

            // Send to decoder queue
            encoder_to_decoder_queue_.push(std::move(output));

            // Update stats
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.encoder_batches_processed++;

                // Update running average of encoder time
                double alpha = 0.1;  // Exponential moving average weight
                if (stats_.avg_encoder_time_ms == 0.0) {
                    stats_.avg_encoder_time_ms = encoder_time_ms;
                } else {
                    stats_.avg_encoder_time_ms =
                        alpha * encoder_time_ms + (1 - alpha) * stats_.avg_encoder_time_ms;
                }
            }
        }
    }

    /**
     * Decoder thread: Processes encoder outputs through decoder
     * Implements Priority 5 (pipeline stage 2), uses Priority 1 & 4 internally
     */
    void decoder_worker() {
        while (!shutdown_.load()) {
            IntegratedEncoderOutput encoder_output;

            if (!encoder_to_decoder_queue_.try_pop(encoder_output, config_.decoder_timeout_ms)) {
                continue;
            }

            auto decoder_start = std::chrono::steady_clock::now();

            // Mark timing for all requests
            for (auto& req : encoder_output.requests) {
                req.decoder_start = decoder_start;
            }

            // Process through decoder for each request in batch
            // Note: Decoder will use OpenMP (Priority 1) and parallel attention (Priority 4)
            std::vector<std::string> results;

            for (size_t i = 0; i < encoder_output.requests.size(); ++i) {
                auto& req = encoder_output.requests[i];

                // Generate response autoregressively using this request's encoder output
                std::string result = generate_from_encoder_output(encoder_output.encoder_outputs[i],
                                                                  req.max_length, req.strategy);

                results.push_back(result);

                // Track tokens
                req.output_tokens = tokenizer_->encode(result).size();
            }

            auto decoder_end = std::chrono::steady_clock::now();
            double decoder_time_ms =
                std::chrono::duration<double, std::milli>(decoder_end - decoder_start).count();

            // Return results to clients
            for (size_t i = 0; i < encoder_output.requests.size(); ++i) {
                auto& req = encoder_output.requests[i];
                req.result_promise.set_value(results[i]);

                // Update request-level stats
                double latency_ms =
                    std::chrono::duration<double, std::milli>(decoder_end - req.submit_time)
                        .count();

                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.total_requests++;
                    stats_.total_tokens_generated += req.output_tokens;

                    // Update running average latency
                    double alpha = 0.1;
                    if (stats_.avg_latency_ms == 0.0) {
                        stats_.avg_latency_ms = latency_ms;
                    } else {
                        stats_.avg_latency_ms =
                            alpha * latency_ms + (1 - alpha) * stats_.avg_latency_ms;
                    }
                }
            }

            // Update decoder stats
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.decoder_batches_processed++;

                // Update running average of decoder time
                double alpha = 0.1;
                if (stats_.avg_decoder_time_ms == 0.0) {
                    stats_.avg_decoder_time_ms = decoder_time_ms;
                } else {
                    stats_.avg_decoder_time_ms =
                        alpha * decoder_time_ms + (1 - alpha) * stats_.avg_decoder_time_ms;
                }
            }
        }
    }

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /**
     * Prepare batched encoder inputs from tokenized sequences
     */
    Matrix prepare_encoder_inputs(const std::vector<std::vector<int>>& tokenized) {
        // Find max length in batch
        size_t max_len = 0;
        for (const auto& tokens : tokenized) {
            max_len = std::max(max_len, tokens.size());
        }

        // Create padded batch matrix
        size_t batch_size = tokenized.size();
        Matrix batch(batch_size, max_len);
        batch.fill(tokenizer_->get_pad_token_id());

        // Fill with token IDs
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < tokenized[i].size(); ++j) {
                batch(static_cast<int>(i), static_cast<int>(j)) =
                    static_cast<float>(tokenized[i][j]);
            }
        }

        return batch;
    }

    /**
     * Generate text from encoder output (simplified for integration)
     */
    std::string generate_from_encoder_output(const Matrix& encoder_output, int max_length,
                                             const std::string& strategy) {
        // Simplified generation - in production would use TextGenerator with cross-attention
        // For now, decode autoregressively using decoder

        std::vector<int> generated_tokens;
        generated_tokens.push_back(tokenizer_->get_bos_token_id());

        for (int step = 0; step < max_length; ++step) {
            // Forward through decoder with cross-attention to encoder output
            // Uses forward_with_encoder which accepts token IDs and encoder context
            Matrix decoder_output =
                decoder_->forward_with_encoder(generated_tokens, encoder_output);

            // Get logits from LM head for last position
            Matrix last_hidden(1, decoder_output.cols);
            for (int col = 0; col < decoder_output.cols; ++col) {
                last_hidden(0, col) = decoder_output(decoder_output.rows - 1, col);
            }

            Matrix logits = lm_head_->forward(last_hidden);

            // Get next token (greedy for simplicity)
            int next_token = 0;
            float max_score = logits(0, 0);
            for (int i = 1; i < logits.cols; ++i) {
                if (logits(0, i) > max_score) {
                    max_score = logits(0, i);
                    next_token = i;
                }
            }

            if (next_token == tokenizer_->get_eos_token_id()) {
                break;
            }

            generated_tokens.push_back(next_token);
        }

        return tokenizer_->decode(generated_tokens);
    }

   public:
    /**
     * Constructor
     */
    IntegratedInferenceEngine(LLMEncoder* encoder, LLMDecoder* decoder, LanguageModelHead* lm_head,
                              BPETokenizer* tokenizer,
                              const IntegratedInferenceConfig& config = IntegratedInferenceConfig())
        : encoder_(encoder),
          decoder_(decoder),
          lm_head_(lm_head),
          tokenizer_(tokenizer),
          config_(config),
          request_queue_(config.max_queue_size),
          batch_queue_(config.encoder_queue_size),
          encoder_to_decoder_queue_(config.decoder_queue_size),
          shutdown_(false),
          next_batch_id_(0) {
        stats_.reset();

// Set OpenMP thread count if configured
#ifdef _OPENMP
        if (config_.use_openmp && config_.num_threads > 0) {
            omp_set_num_threads(config_.num_threads);
        }
#endif

        // Start worker threads
        batcher_thread_ = std::thread(&IntegratedInferenceEngine::batcher_worker, this);
        encoder_thread_ = std::thread(&IntegratedInferenceEngine::encoder_worker, this);
        decoder_thread_ = std::thread(&IntegratedInferenceEngine::decoder_worker, this);
    }

    /**
     * Destructor
     */
    ~IntegratedInferenceEngine() {
        shutdown();
    }

    /**
     * Submit a request for processing
     *
     * @param input_text Input text to process
     * @param max_length Maximum generation length
     * @param strategy Generation strategy
     * @return Future containing the generated response
     */
    std::future<std::string> submit(const std::string& input_text, int max_length = -1,
                                    const std::string& strategy = "") {
        if (max_length < 0) {
            max_length = config_.default_max_length;
        }

        if (strategy.empty()) {
            // Use config default
        }

        IntegratedRequest request;
        request.input_text = input_text;
        request.max_length = max_length;
        request.strategy = strategy.empty() ? config_.generation_strategy : strategy;
        request.submit_time = std::chrono::steady_clock::now();

        std::future<std::string> result = request.result_promise.get_future();
        request_queue_.push(std::move(request));

        return result;
    }

    /**
     * Shutdown the engine gracefully
     */
    void shutdown() {
        if (!shutdown_.exchange(true)) {
            // Signal shutdown to all queues
            request_queue_.shutdown();
            batch_queue_.shutdown();
            encoder_to_decoder_queue_.shutdown();

            // Wait for workers to finish
            if (batcher_thread_.joinable())
                batcher_thread_.join();
            if (encoder_thread_.joinable())
                encoder_thread_.join();
            if (decoder_thread_.joinable())
                decoder_thread_.join();
        }
    }

    /**
     * Get current statistics
     */
    IntegratedInferenceStats get_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        IntegratedInferenceStats stats_copy = stats_;

        // Calculate derived metrics
        if (stats_copy.total_batches > 0) {
            stats_copy.avg_batch_size =
                static_cast<double>(stats_copy.total_requests) / stats_copy.total_batches;
            stats_copy.batch_utilization = stats_copy.avg_batch_size / config_.max_batch_size;
        }

        if (stats_copy.avg_encoder_time_ms > 0 && stats_copy.avg_decoder_time_ms > 0) {
            double total_sequential =
                stats_copy.avg_encoder_time_ms + stats_copy.avg_decoder_time_ms;
            double pipeline_time =
                std::max(stats_copy.avg_encoder_time_ms, stats_copy.avg_decoder_time_ms);
            stats_copy.pipeline_efficiency = total_sequential / pipeline_time;
        }

        // Update throughput metrics
        const_cast<IntegratedInferenceStats&>(stats_copy).update_throughput();

        return stats_copy;
    }

    /**
     * Reset statistics
     */
    void reset_stats() {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.reset();
    }
};

// Type alias for convenience
using StandardIntegratedEngine = IntegratedInferenceEngine;
