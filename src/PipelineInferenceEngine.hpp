// @adai-status: beta        (tested but not wired into any shipped binary yet)
// @adai-version: 0.7.0
// @adai-reviewed: 2026-09-07

/**
 * Pipeline Parallelism for Encoder-Decoder Models
 * Priority 5: Achieve 2-3x throughput improvement
 *
 * This implementation creates a multi-stage pipeline where:
 * - Stage 1 (Encoder): Processes input batches through encoder
 * - Stage 2 (Decoder): Processes encoder outputs through decoder
 *
 * Multiple batches can be in flight simultaneously, overlapping computation
 * across stages for higher throughput.
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
#include "Decoder.hpp"
#include "LanguageModelHead.hpp"
#include "Matrix.hpp"
#include "SpecialTokens.hpp"
#include "encoder.hpp"

// ============================================================================
// Thread-Safe Queue for Pipeline Stages
// ============================================================================

template <typename T>
class ThreadSafeQueue {
   private:
    std::deque<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool shutdown_;
    size_t max_size_;

   public:
    explicit ThreadSafeQueue(size_t max_size = 100) : shutdown_(false), max_size_(max_size) {}

    // Push item into queue (blocks if full)
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait if queue is full
        cv_.wait(lock, [this] { return queue_.size() < max_size_ || shutdown_; });

        if (shutdown_)
            return;

        queue_.push_back(std::move(item));
        cv_.notify_one();
    }

    // Try to pop item from queue (blocks until available or timeout)
    bool try_pop(T& item, int timeout_ms = 50) {
        std::unique_lock<std::mutex> lock(mutex_);

        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

        if (!cv_.wait_until(lock, deadline, [this] { return !queue_.empty() || shutdown_; })) {
            return false;  // Timeout
        }

        if (shutdown_ && queue_.empty()) {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop_front();
        cv_.notify_one();  // Notify waiting pushers

        return true;
    }

    // Pop item from queue (blocks until available)
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

        if (shutdown_ && queue_.empty()) {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop_front();
        cv_.notify_one();

        return true;
    }

    // Get current queue size
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    // Check if queue is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    // Shutdown queue (unblocks all waiting threads)
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_.notify_all();
    }

    // Check if queue is shutdown
    bool is_shutdown() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return shutdown_;
    }
};

// ============================================================================
// Pipeline Data Structures
// ============================================================================

// Request for pipeline processing
struct PipelineRequest {
    uint64_t request_id;
    std::vector<std::string> input_texts;  // Batch of inputs
    int max_length;
    std::chrono::steady_clock::time_point submit_time;
    std::promise<std::vector<std::string>> result_promise;

    PipelineRequest() : request_id(0), max_length(100) {}

    // Move-only type
    PipelineRequest(PipelineRequest&&) = default;
    PipelineRequest& operator=(PipelineRequest&&) = default;
    PipelineRequest(const PipelineRequest&) = delete;
    PipelineRequest& operator=(const PipelineRequest&) = delete;
};

// Intermediate result after encoder stage
struct EncoderOutput {
    uint64_t request_id;
    std::vector<Matrix> encoder_outputs;  // One per input in batch
    std::vector<std::string> input_texts;
    int max_length;
    std::chrono::steady_clock::time_point submit_time;
    std::promise<std::vector<std::string>> result_promise;

    EncoderOutput() : request_id(0), max_length(100) {}

    // Move-only type
    EncoderOutput(EncoderOutput&&) = default;
    EncoderOutput& operator=(EncoderOutput&&) = default;
    EncoderOutput(const EncoderOutput&) = delete;
    EncoderOutput& operator=(const EncoderOutput&) = delete;
};

// Pipeline configuration
struct PipelineConfig {
    size_t max_queue_size = 50;    // Max requests in each queue
    int encoder_timeout_ms = 50;   // Timeout for encoder stage
    int decoder_timeout_ms = 50;   // Timeout for decoder stage
    bool enable_profiling = true;  // Track stage timings

    PipelineConfig() = default;
};

// Pipeline statistics
struct PipelineStats {
    uint64_t total_requests;
    uint64_t encoder_processed;
    uint64_t decoder_processed;

    double avg_encoder_time_ms;
    double avg_decoder_time_ms;
    double avg_total_latency_ms;
    double avg_throughput_rps;

    uint64_t encoder_queue_size;
    uint64_t decoder_queue_size;

    PipelineStats()
        : total_requests(0),
          encoder_processed(0),
          decoder_processed(0),
          avg_encoder_time_ms(0.0),
          avg_decoder_time_ms(0.0),
          avg_total_latency_ms(0.0),
          avg_throughput_rps(0.0),
          encoder_queue_size(0),
          decoder_queue_size(0) {}
};

// ============================================================================
// Pipeline Inference Engine (Template for flexibility)
// ============================================================================

template <typename EncoderType, typename DecoderType, typename LMHeadType, typename TokenizerType>
class PipelineInferenceEngine {
   private:
    // Model components
    EncoderType* encoder_;
    DecoderType* decoder_;
    LMHeadType* lm_head_;
    TokenizerType* tokenizer_;

    // Pipeline configuration
    PipelineConfig config_;

    // Pipeline queues
    ThreadSafeQueue<PipelineRequest> input_queue_;
    ThreadSafeQueue<EncoderOutput> encoder_to_decoder_queue_;

    // Worker threads
    std::thread encoder_thread_;
    std::thread decoder_thread_;

    // Lifecycle management
    std::atomic<bool> running_;
    std::atomic<uint64_t> next_request_id_;

    // Statistics tracking (protected by mutex)
    mutable std::mutex stats_mutex_;
    PipelineStats stats_;
    std::chrono::steady_clock::time_point start_time_;

    double total_encoder_time_ms_;
    double total_decoder_time_ms_;
    double total_latency_ms_;

    // Special token IDs
    int bos_token_id_;
    int eos_token_id_;
    int pad_token_id_;

    // ========================================================================
    // Pipeline Stage Workers
    // ========================================================================

    /**
     * Encoder stage worker thread
     * Processes input batches through encoder and passes to decoder stage
     */
    void encoder_worker() {
        while (running_) {
            PipelineRequest request;

            // Try to get a request from input queue
            if (!input_queue_.try_pop(request, config_.encoder_timeout_ms)) {
                continue;  // Timeout, check running_ flag
            }

            auto stage_start = std::chrono::steady_clock::now();

            try {
                // Process batch through encoder
                std::vector<Matrix> encoder_outputs;
                encoder_outputs.reserve(request.input_texts.size());

                for (const auto& text : request.input_texts) {
                    Matrix output = encoder_->encode(text);
                    encoder_outputs.push_back(std::move(output));
                }

                auto stage_end = std::chrono::steady_clock::now();
                double stage_time =
                    std::chrono::duration<double, std::milli>(stage_end - stage_start).count();

                // Update statistics
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.encoder_processed++;
                    total_encoder_time_ms_ += stage_time;
                    stats_.avg_encoder_time_ms = total_encoder_time_ms_ / stats_.encoder_processed;
                }

                // Create encoder output and pass to decoder stage
                EncoderOutput enc_output;
                enc_output.request_id = request.request_id;
                enc_output.encoder_outputs = std::move(encoder_outputs);
                enc_output.input_texts = std::move(request.input_texts);
                enc_output.max_length = request.max_length;
                enc_output.submit_time = request.submit_time;
                enc_output.result_promise = std::move(request.result_promise);

                encoder_to_decoder_queue_.push(std::move(enc_output));

            } catch (const std::exception& e) {
                std::cerr << "Encoder stage error: " << e.what() << std::endl;
                request.result_promise.set_exception(std::current_exception());
            }
        }
    }

    /**
     * Decoder stage worker thread
     * Processes encoder outputs through decoder and generates text
     */
    void decoder_worker() {
        while (running_) {
            EncoderOutput enc_output;

            // Try to get encoder output
            if (!encoder_to_decoder_queue_.try_pop(enc_output, config_.decoder_timeout_ms)) {
                continue;  // Timeout, check running_ flag
            }

            auto stage_start = std::chrono::steady_clock::now();

            try {
                // Process each input in the batch through decoder
                std::vector<std::string> results;
                results.reserve(enc_output.input_texts.size());

                for (size_t i = 0; i < enc_output.encoder_outputs.size(); ++i) {
                    const Matrix& encoder_output = enc_output.encoder_outputs[i];

                    // Generate text autoregressively
                    std::vector<int> generated_tokens = {bos_token_id_};

                    for (int step = 0; step < enc_output.max_length; ++step) {
                        // Forward pass through decoder
                        Matrix decoder_output = decoder_->forward_with_cross_attention(
                            generated_tokens, encoder_output, nullptr);

                        // Get logits for last token
                        Matrix last_hidden = Matrix(1, decoder_output.cols);
                        for (int j = 0; j < decoder_output.cols; ++j) {
                            last_hidden(0, j) = decoder_output(decoder_output.rows - 1, j);
                        }

                        Matrix logits = lm_head_->forward(last_hidden);

                        // Greedy decoding: select token with highest probability
                        int next_token = 0;
                        float max_logit = logits(0, 0);
                        for (int j = 1; j < logits.cols; ++j) {
                            if (logits(0, j) > max_logit) {
                                max_logit = logits(0, j);
                                next_token = j;
                            }
                        }

                        // Stop if EOS token generated
                        if (next_token == eos_token_id_) {
                            break;
                        }

                        generated_tokens.push_back(next_token);
                    }

                    // Decode tokens to text
                    std::string result = tokenizer_->decode(generated_tokens);
                    results.push_back(result);
                }

                auto stage_end = std::chrono::steady_clock::now();
                double stage_time =
                    std::chrono::duration<double, std::milli>(stage_end - stage_start).count();

                double total_latency =
                    std::chrono::duration<double, std::milli>(stage_end - enc_output.submit_time)
                        .count();

                // Update statistics
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.decoder_processed++;
                    total_decoder_time_ms_ += stage_time;
                    total_latency_ms_ += total_latency;

                    stats_.avg_decoder_time_ms = total_decoder_time_ms_ / stats_.decoder_processed;
                    stats_.avg_total_latency_ms = total_latency_ms_ / stats_.decoder_processed;

                    // Calculate throughput
                    auto elapsed = std::chrono::steady_clock::now() - start_time_;
                    double elapsed_sec = std::chrono::duration<double>(elapsed).count();
                    if (elapsed_sec > 0) {
                        stats_.avg_throughput_rps = stats_.decoder_processed / elapsed_sec;
                    }
                }

                // Return results to caller
                enc_output.result_promise.set_value(std::move(results));

            } catch (const std::exception& e) {
                std::cerr << "Decoder stage error: " << e.what() << std::endl;
                enc_output.result_promise.set_exception(std::current_exception());
            }
        }
    }

   public:
    /**
     * Constructor
     *
     * @param encoder Encoder model
     * @param decoder Decoder model
     * @param lm_head Language model head
     * @param tokenizer Tokenizer
     * @param config Pipeline configuration
     */
    PipelineInferenceEngine(EncoderType* encoder, DecoderType* decoder, LMHeadType* lm_head,
                            TokenizerType* tokenizer,
                            const PipelineConfig& config = PipelineConfig())
        : encoder_(encoder),
          decoder_(decoder),
          lm_head_(lm_head),
          tokenizer_(tokenizer),
          config_(config),
          input_queue_(config.max_queue_size),
          encoder_to_decoder_queue_(config.max_queue_size),
          running_(true),
          next_request_id_(1),
          total_encoder_time_ms_(0.0),
          total_decoder_time_ms_(0.0),
          total_latency_ms_(0.0),
          bos_token_id_(adai::SpecialTokenIDs::BOS),
          eos_token_id_(adai::SpecialTokenIDs::EOS),
          pad_token_id_(adai::SpecialTokenIDs::PAD) {
        start_time_ = std::chrono::steady_clock::now();

        // Start pipeline worker threads
        encoder_thread_ = std::thread(&PipelineInferenceEngine::encoder_worker, this);
        decoder_thread_ = std::thread(&PipelineInferenceEngine::decoder_worker, this);
    }

    /**
     * Destructor - shuts down pipeline gracefully
     */
    ~PipelineInferenceEngine() {
        shutdown();
    }

    /**
     * Submit batch for processing through pipeline
     *
     * @param input_texts Batch of input texts
     * @param max_length Maximum generation length
     * @return Future with batch of generated outputs
     */
    std::future<std::vector<std::string>> submit_batch(const std::vector<std::string>& input_texts,
                                                       int max_length = 100) {
        PipelineRequest request;
        request.request_id = next_request_id_++;
        request.input_texts = input_texts;
        request.max_length = max_length;
        request.submit_time = std::chrono::steady_clock::now();

        auto future = request.result_promise.get_future();

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_requests++;
        }

        input_queue_.push(std::move(request));

        return future;
    }

    /**
     * Submit single input for processing
     *
     * @param input_text Input text
     * @param max_length Maximum generation length
     * @return Future with generated output
     */
    std::future<std::string> submit(const std::string& input_text, int max_length = 100) {
        auto batch_future = submit_batch({input_text}, max_length);

        // Convert batch future to single-item future
        return std::async(
            std::launch::deferred,
            [](std::future<std::vector<std::string>> f) {
                auto results = f.get();
                return results.empty() ? std::string() : results[0];
            },
            std::move(batch_future));
    }

    /**
     * Get pipeline statistics
     *
     * @return Current pipeline statistics
     */
    PipelineStats get_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        PipelineStats stats = stats_;
        stats.encoder_queue_size = input_queue_.size();
        stats.decoder_queue_size = encoder_to_decoder_queue_.size();

        return stats;
    }

    /**
     * Shutdown pipeline and wait for completion
     */
    void shutdown() {
        running_ = false;

        // Shutdown queues to unblock worker threads
        input_queue_.shutdown();
        encoder_to_decoder_queue_.shutdown();

        // Wait for worker threads to finish
        if (encoder_thread_.joinable()) {
            encoder_thread_.join();
        }
        if (decoder_thread_.joinable()) {
            decoder_thread_.join();
        }
    }

    /**
     * Check if pipeline is running
     */
    bool is_running() const {
        return running_;
    }
};

// Type alias for convenience with standard ADAI components
using StandardPipelineEngine =
    PipelineInferenceEngine<LLMEncoder, LLMDecoder, LanguageModelHead, BPETokenizer>;
