#pragma once

// @adai-status: beta        (capped by TD-050 — see TECHNICAL_DEBT.md)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-07


#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "BPETokenizer.hpp"
#include "Decoder.hpp"
#include "KVCache.hpp"
#include "LanguageModelHead.hpp"
#include "Matrix.hpp"
#include "TextGenerator.hpp"
#include "encoder.hpp"
#ifdef ADAI_ENABLE_GPU
#include "gpu/MatrixGPU.hpp"
#endif

/**
 * EncoderDecoderModel - Complete sequence-to-sequence transformer
 *
 * Combines an encoder and decoder for tasks like:
 * - Machine translation
 * - Text summarization
 * - Chatbot responses with context
 * - Question answering
 * - Dialog systems
 *
 * Architecture:
 *
 * Input Text → Tokenizer → Encoder → Context Vector
 *                                         ↓
 * Output: <bos> → Decoder (with cross-attention) → LM Head → Token Probs
 *                     ↓
 *         Token₁ → Decoder → LM Head → Token Probs
 *                     ↓
 *                   ...
 *                     ↓
 *                  <eos> → Stop
 *
 * Features:
 * - Shared vocabulary between encoder and decoder
 * - Cross-attention from decoder to encoder outputs
 * - Multiple generation strategies (greedy, beam search, sampling)
 * - Training support with cross-entropy loss
 * - Weight sharing between token embedding and output projection (optional)
 * - Save/load complete model
 */
class EncoderDecoderModel {
   private:
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<LLMEncoder> encoder;
    std::unique_ptr<LLMDecoder> decoder;
    std::unique_ptr<LanguageModelHead> lm_head;
    std::unique_ptr<TextGenerator> generator;

    int vocab_size;
    int d_model;
    int encoder_layers;
    int decoder_layers;
    int num_heads;
    int d_ff;
    int max_seq_length;

    // Special token IDs
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;

    // Training state
    bool requires_grad{true};
    float learning_rate{0.001};

    // Cached values for training
    Matrix cached_encoder_output;
    Matrix cached_decoder_output;
    std::vector<int> cached_input_tokens;
    std::vector<int> cached_target_tokens;

#ifdef ADAI_ENABLE_GPU
    bool gpu_initialized_{false};
    // GPU-resident state cached between gpu_forward and gpu_backward
    std::unique_ptr<adai::gpu::GPUMatrix> gpu_encoder_out_;
    std::unique_ptr<adai::gpu::GPUMatrix> gpu_logits_;
    std::unique_ptr<adai::gpu::GPUMemory<int>> gpu_targets_dev_;
    int gpu_target_len_{0};
#endif

    /**
     * Compute cross-entropy loss for language modeling
     *
     * @param logits Output logits from LM head [seq_length, vocab_size]
     * @param target_tokens Target token IDs [seq_length]
     * @return Loss value (scalar)
     */
    float compute_loss(const Matrix& logits, const std::vector<int>& target_tokens);

    /**
     * Compute gradients for cross-entropy loss
     *
     * @param logits Output logits [seq_length, vocab_size]
     * @param target_tokens Target token IDs
     * @return Gradient matrix [seq_length, vocab_size]
     */
    Matrix compute_loss_gradient(const Matrix& logits, const std::vector<int>& target_tokens);

   public:
    /**
     * Constructor
     *
     * @param vocab_size Vocabulary size (shared between encoder/decoder)
     * @param d_model Model dimension
     * @param encoder_layers Number of encoder layers
     * @param decoder_layers Number of decoder layers
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward dimension
     * @param max_seq_length Maximum sequence length
     */
    EncoderDecoderModel(int vocab_size, int d_model = 512, int encoder_layers = 6,
                        int decoder_layers = 6, int num_heads = 8, int d_ff = 2048,
                        int max_seq_length = 512);

    /**
     * Destructor
     */
    ~EncoderDecoderModel();
    EncoderDecoderModel(const EncoderDecoderModel&) = delete;
    EncoderDecoderModel& operator=(const EncoderDecoderModel&) = delete;
    EncoderDecoderModel(EncoderDecoderModel&&) = delete;
    EncoderDecoderModel& operator=(EncoderDecoderModel&&) = delete;

    /**
     * Create a deep copy of this model's weights into a new instance (TD-023).
     *
     * All weight matrices in the encoder, decoder, and LM head are copied.
     * Mutable training state (optimizer moments, gradient accumulators) is
     * NOT included — the clone is inference-only by design.
     *
     * Memory cost: one full copy of all weight tensors (~50–150 MB for typical
     * model sizes).  The temporary serialization files are removed on return.
     *
     * @return Owning pointer to the cloned model.
     * @throws std::runtime_error if serialisation to the temp directory fails.
     */
    std::unique_ptr<EncoderDecoderModel> clone() const;

    /**
     * Generate response for input text (inference mode)
     * Uses configured generation strategy from TextGenerator
     *
     * @param input_text Input text to encode
     * @param max_length Maximum output length
     * @return Generated response text
     */
    std::string generate_response(const std::string& input_text, int max_length = 100);

#ifdef ADAI_ENABLE_GPU
    /**
     * GPU-accelerated equivalent of generate_response(). Same sampling
     * behaviour (uses the shared TextGenerator config — temperature/top-k/
     * top-p as configured), but each decode step runs through the
     * GPU-resident encoder/decoder/lm_head instead of the CPU Matrix path.
     *
     * No GPU KV-cache exists yet, so each step recomputes the full decoded
     * sequence from scratch via gpu_decode() rather than incrementally
     * caching — same algorithmic shape as the CPU "greedy workaround" path,
     * just GPU-accelerated. Only the last position's logits are downloaded
     * per step. (See TD-050 for the KV-cache gap.)
     *
     * See TD-033 in TECHNICAL_DEBT.md - this is the persistent GPU-resident
     * decode path chatbot_api_server's live serving code should be using, but
     * currently isn't: ChatbotAPI::generate_response() calls forward() instead
     * (ChatbotAPI.cpp). Today this function is only reached from
     * ChatbotTrainer's internal generation-quality backfill / BLEU-ROUGE
     * scoring (ChatbotTrainer.cpp), never from a real chat request.
     *
     * @param input_text Input text to encode
     * @param max_length Maximum output length
     * @return Generated response text
     */
    std::string gpu_generate_response(const std::string& input_text, int max_length = 100);
#endif

    /**
     * Generate response with specific strategy
     *
     * @param input_text Input text
     * @param max_length Maximum output length
     * @param strategy "greedy", "beam", "sampling", "topk", "nucleus"
     * @param temperature Temperature for sampling (default 1.0)
     * @param top_k Top-k value for filtering (default 50)
     * @param top_p Top-p value for nucleus sampling (default 0.9)
     * @param num_beams Number of beams for beam search (default 4)
     * @return Generated response text
     */
    std::string generate_response_with_strategy(const std::string& input_text, int max_length = 100,
                                                const std::string& strategy = "greedy",
                                                float temperature = 1.0f, int top_k = 50,
                                                float top_p = 0.9f, int num_beams = 4);

    /**
     * Training step on a single (input, target) pair
     *
     * @param input_text Input text to encode
     * @param target_text Target output text
     * @return Loss value for this step
     */
    float train_step(const std::string& input_text, const std::string& target_text);

    /**
     * Training step on tokenized sequences
     *
     * @param input_tokens Input token IDs
     * @param target_tokens Target token IDs
     * @return Loss value
     */
    float train_step_tokenized(const std::vector<int>& input_tokens,
                               const std::vector<int>& target_tokens);

    /**
     * Evaluate model on validation data (no gradient computation)
     *
     * @param input_text Input text
     * @param target_text Target text
     * @return Loss value
     */
    float evaluate(const std::string& input_text, const std::string& target_text);

    /**
     * Evaluate model on already-tokenized validation data (no gradient computation).
     *
     * Prefer this over the text-based evaluate() whenever pre-tokenized ids are
     * available: encode() on raw text has no length cap, so re-tokenizing raw
     * validation text on every call can silently bypass the max_seq_length
     * truncation applied when the dataset was first tokenized, and pay for
     * redundant BPE encoding on every validation pass.
     *
     * @param input_tokens Already-tokenized (and length-truncated) input ids
     * @param target_tokens Already-tokenized (and length-truncated) target ids
     * @return Loss value
     */
    float evaluate_tokenized(const std::vector<int>& input_tokens,
                             const std::vector<int>& target_tokens);

    /**
     * Compute perplexity on a dataset
     *
     * @param input_texts Vector of input texts
     * @param target_texts Vector of target texts
     * @return Perplexity score
     */
    float compute_perplexity(const std::vector<std::string>& input_texts,
                             const std::vector<std::string>& target_texts);

    /**
     * Set training mode
     *
     * @param mode True for training, false for inference
     */
    void set_training(bool mode);

    /**
     * Set learning rate
     *
     * @param lr Learning rate
     */
    void set_learning_rate(float lr);

    /**
     * Update model weights after backward pass
     */
    void update_weights();

    /**
     * Zero all gradients
     */
    void zero_grad();

    /**
     * Register all model parameters with an external optimizer
     *
     * This method exposes all weight and gradient matrices to an
     * external Optimizer object for centralized gradient management.
     * Call this once after model initialization.
     *
     * @param optimizer Optimizer to register parameters with
     */
    void register_parameters(class Optimizer& optimizer);

    /**
     * Backward pass without updating weights
     *
     * Computes gradients only. Weight updates should be handled
     * by an external optimizer after calling this method.
     *
     * @param grad_output Gradient from loss
     */
    void backward_pass(const Matrix& grad_output);

    /**
     * Compute loss for training (exposed for custom training loops)
     *
     * @param logits Output logits from forward pass
     * @param target_tokens Target token IDs
     * @return Loss value
     */
    float compute_loss_for_training(const Matrix& logits, const std::vector<int>& target_tokens) {
        return compute_loss(logits, target_tokens);
    }

    /**
     * Compute loss gradients for training (exposed for custom training loops)
     *
     * @param logits Output logits from forward pass
     * @param target_tokens Target token IDs
     * @return Gradient matrix
     */
    Matrix compute_loss_gradient_for_training(const Matrix& logits,
                                              const std::vector<int>& target_tokens) {
        return compute_loss_gradient(logits, target_tokens);
    }

    /**
     * Set tokenizer (if using custom tokenizer)
     *
     * @param tokenizer_ptr Pointer to tokenizer
     */
    void set_tokenizer(BPETokenizer* tokenizer_ptr);

    /**
     * Get tokenizer for external use
     *
     * @return Pointer to tokenizer
     */
    BPETokenizer* get_tokenizer() {
        return tokenizer.get();
    }

    /**
     * Configure text generation parameters
     *
     * @param config TextGenerator::GenerationConfig
     */
    void set_generation_config(const TextGenerator::GenerationConfig& config);

    /**
     * Get current generation config
     *
     * @return Current generation config
     */
    TextGenerator::GenerationConfig get_generation_config() const;

    /**
     * Synchronize special token IDs from tokenizer to generator
     * Call this after building/loading vocabulary
     */
    void sync_special_tokens();

    /**
     * Save complete model to file
     *
     * @param filepath Base filepath (will create multiple files)
     */
    void save_model(const std::string& filepath) const;

    /**
     * Load complete model from file
     *
     * @param filepath Base filepath
     */
    void load_model(const std::string& filepath);

    /**
     * Get model dimensions
     */
    int get_vocab_size() const {
        return vocab_size;
    }
    int get_d_model() const {
        return d_model;
    }
    int get_encoder_layers() const {
        return encoder_layers;
    }
    int get_decoder_layers() const {
        return decoder_layers;
    }

    /**
     * Get special token IDs
     */
    int get_bos_token_id() const {
        return bos_token_id;
    }
    int get_eos_token_id() const {
        return eos_token_id;
    }
    int get_pad_token_id() const {
        return pad_token_id;
    }
    int get_num_heads() const {
        return num_heads;
    }
    int get_d_ff() const {
        return d_ff;
    }
    int get_max_seq_length() const {
        return max_seq_length;
    }

    /**
     * Access internal components (for advanced use)
     */
    LLMEncoder* get_encoder() {
        return encoder.get();
    }
    LLMDecoder* get_decoder() {
        return decoder.get();
    }
    LanguageModelHead* get_lm_head() {
        return lm_head.get();
    }
    TextGenerator* get_generator() {
        return generator.get();
    }

    /**
     * Forward pass through complete model (for custom training loops)
     *
     * @param input_tokens Input token IDs
     * @param target_tokens Target token IDs (for teacher forcing)
     * @return Logits matrix [seq_length, vocab_size]
     */
    Matrix forward(const std::vector<int>& input_tokens, const std::vector<int>& target_tokens);

    /**
     * Backward pass (for custom training loops)
     *
     * @param grad_output Gradient from loss [seq_length, vocab_size]
     */
    void backward(const Matrix& grad_output);

#ifdef ADAI_ENABLE_GPU
    /** Upload all weights to GPU (call once before first gpu_forward). */
    void gpu_init_training();

    /** Zero all GPU gradient accumulators (call at start of each accumulation window). */
    void gpu_zero_grads();

    /**
     * Full GPU forward pass.
     * @return scalar loss; caches GPU encoder/decoder outputs for gpu_backward.
     */
    float gpu_forward(const std::vector<int>& input_tokens, const std::vector<int>& target_tokens);

    /**
     * Full GPU backward pass. Accumulates GPU gradients.
     * @param scale Multiply loss gradient by this factor before accumulating
     *              (use 1/gradient_accumulation_steps for proper averaging).
     */
    void gpu_backward(float scale = 1.0f);

    /**
     * GPU-accelerated evaluation (loss only, no gradient computation).
     * Mirrors evaluate() but runs the forward pass through the GPU-resident
     * path instead of the CPU Matrix path, so validation runs at the same
     * speed as training.
     *
     * @param input_text Input text
     * @param target_text Target text
     * @return Loss value
     */
    float gpu_evaluate(const std::string& input_text, const std::string& target_text);

    /**
     * GPU-accelerated evaluation on already-tokenized data (no gradient computation).
     * See evaluate_tokenized() for why this is preferred over the text-based
     * overload whenever pre-tokenized ids are available.
     *
     * @param input_tokens Already-tokenized (and length-truncated) input ids
     * @param target_tokens Already-tokenized (and length-truncated) target ids
     * @return Loss value
     */
    float gpu_evaluate_tokenized(const std::vector<int>& input_tokens,
                                 const std::vector<int>& target_tokens);

    /**
     * Download GPU gradient accumulators to CPU gradient members.
     * Call before optimizer->step().
     */
    void gpu_download_grads();

    /**
     * Re-upload updated CPU weights to GPU mirrors.
     * Call after optimizer->step().
     */
    void gpu_sync_weights();

    /**
     * Block until all previously submitted GPU work (including deferred
     * frees of temporary GPUMatrix/GPUMemory buffers) has completed.
     *
     * Every kernel submission and USM allocation on the training path is
     * asynchronous, but temporaries only actually release their device
     * memory via a queue-ordered deferred free (see GPUMemory::defer_free).
     * Without a periodic drain, the host can queue far more work than the
     * device has retired, so the set of "allocated but not yet freed"
     * buffers grows without bound across samples until the device runs out
     * of memory. Call this once per sample (or at minimum once per
     * optimizer step) to bound that backlog.
     */
    void gpu_synchronize();
#endif
};
