#pragma once

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "BPETokenizer.hpp"
#include "Matrix.hpp"

/**
 * Text Generator for Autoregressive Language Generation
 *
 * Implements various decoding strategies for generating text from language models.
 * Supports greedy decoding, beam search, and sampling with temperature control,
 * top-k filtering, and nucleus (top-p) sampling.
 *
 * Generation Process:
 *   1. Start with prompt tokens (or <bos> token)
 *   2. Model predicts next-token logits
 *   3. Apply generation strategy to select token
 *   4. Append token to sequence
 *   5. Repeat until <eos> or max_length
 *
 * Decoding Strategies:
 *   - Greedy: Select highest probability token (deterministic)
 *   - Beam Search: Maintain k best sequences (semi-deterministic)
 *   - Temperature Sampling: Control randomness (stochastic)
 *   - Top-k Sampling: Sample from k most likely tokens
 *   - Nucleus (Top-p): Sample from smallest set with cumulative prob >= p
 *
 * Features:
 *   - Multiple generation strategies
 *   - Customizable stopping criteria
 *   - Special token handling (<bos>, <eos>, <pad>)
 *   - Repetition penalty
 *   - Length normalization for beam search
 */
class TextGenerator {
   public:
    /**
     * Beam hypothesis for beam search
     * Tracks a candidate sequence with its score
     */
    struct BeamHypothesis {
        std::vector<int> tokens;  // Token sequence
        float score;              // Log probability score
        bool is_finished;         // Whether sequence ended with <eos>

        BeamHypothesis() : score(0.0f), is_finished(false) {}

        BeamHypothesis(const std::vector<int>& toks, float s)
            : tokens(toks), score(s), is_finished(false) {}
    };

    /**
     * Generation configuration parameters
     */
    struct GenerationConfig {
        int max_length = 100;               // Maximum sequence length
        float temperature = 1.0f;           // Sampling temperature (0 = greedy)
        int top_k = 0;                      // Top-k filtering (0 = disabled)
        float top_p = 1.0f;                 // Nucleus sampling threshold
        float repetition_penalty = 1.0f;    // Penalty for repeated tokens (1.0 = none)
        int num_beams = 1;                  // Beam width for beam search
        bool length_penalty = true;         // Apply length normalization in beam search
        float length_penalty_alpha = 0.6f;  // Length penalty exponent
        bool early_stopping = true;         // Stop when all beams finish
        int min_length = 0;                 // Minimum generation length
        int pad_token_id = 0;               // Padding token ID
        int bos_token_id = 2;               // Beginning of sequence token
        int eos_token_id = 3;               // End of sequence token
        int unk_token_id = 1;               // Unknown token ID

        GenerationConfig() = default;
    };

    /**
     * Model forward function signature
     * Takes token IDs, returns logits [seq_len, vocab_size]
     *
     * For decoder-only models: forward(input_ids) -> logits
     * For encoder-decoder: forward(input_ids, encoder_output) -> logits
     */
    using ModelForwardFn = std::function<Matrix(const std::vector<int>&)>;

   private:
    GenerationConfig config;
    std::mt19937 rng;  // Random number generator

    /**
     * Apply temperature scaling to logits
     * Higher temperature = more random, lower = more deterministic
     */
    std::vector<float> apply_temperature(const std::vector<float>& logits, float temperature);

    /**
     * Apply top-k filtering
     * Zero out all logits except top k
     */
    std::vector<float> apply_top_k(const std::vector<float>& logits, int k);

    /**
     * Apply nucleus (top-p) sampling
     * Keep smallest set of tokens with cumulative probability >= p
     */
    std::vector<float> apply_top_p(const std::vector<float>& logits, float p);

    /**
     * Apply repetition penalty
     * Reduce probability of already-generated tokens
     */
    std::vector<float> apply_repetition_penalty(const std::vector<float>& logits,
                                                const std::vector<int>& generated_tokens,
                                                float penalty);

    /**
     * Convert logits to probability distribution (softmax)
     */
    std::vector<float> softmax(const std::vector<float>& logits);

    /**
     * Sample token from probability distribution
     */
    int sample_token(const std::vector<float>& probabilities);

    /**
     * Get argmax (highest probability token)
     */
    int argmax(const std::vector<float>& values);

    /**
     * Compute length-normalized score for beam search
     */
    float compute_length_penalty(int length, float alpha);

    /**
     * Check if token is a stopping token (<eos>, <pad>)
     */
    bool is_stop_token(int token_id);

   public:
    /**
     * Constructor
     *
     * @param config Generation configuration
     * @param seed Random seed for sampling (0 = random)
     */
    TextGenerator(const GenerationConfig& config, unsigned int seed);

    /**
     * Default constructor with default configuration
     */
    TextGenerator();

    /**
     * Greedy Decoding
     *
     * Selects the highest probability token at each step.
     * Deterministic and fast, but may produce repetitive text.
     *
     * @param model_fn Model forward function
     * @param prompt_tokens Initial token sequence (empty for unconditional)
     * @return Generated token sequence
     */
    std::vector<int> generate_greedy(ModelForwardFn model_fn,
                                     const std::vector<int>& prompt_tokens = {});

    /**
     * Beam Search Decoding
     *
     * Maintains num_beams candidate sequences and explores most promising paths.
     * Balances quality and diversity, good for translation/summarization.
     *
     * @param model_fn Model forward function
     * @param prompt_tokens Initial token sequence
     * @param num_beams Number of beams to maintain
     * @return Best generated token sequence
     */
    std::vector<int> generate_beam_search(ModelForwardFn model_fn,
                                          const std::vector<int>& prompt_tokens = {},
                                          int num_beams = -1  // -1 = use config.num_beams
    );

    /**
     * Temperature Sampling
     *
     * Samples from probability distribution with temperature control.
     * temperature < 1.0: More focused, conservative
     * temperature = 1.0: Standard sampling
     * temperature > 1.0: More random, creative
     *
     * @param model_fn Model forward function
     * @param prompt_tokens Initial token sequence
     * @param temperature Sampling temperature
     * @return Generated token sequence
     */
    std::vector<int> generate_sampling(ModelForwardFn model_fn,
                                       const std::vector<int>& prompt_tokens = {},
                                       float temperature = -1.0f  // -1 = use config.temperature
    );

    /**
     * Top-k Sampling
     *
     * Samples from only the k most likely tokens at each step.
     * Prevents sampling from very low probability tokens.
     *
     * @param model_fn Model forward function
     * @param prompt_tokens Initial token sequence
     * @param k Number of top tokens to consider
     * @return Generated token sequence
     */
    std::vector<int> generate_top_k(ModelForwardFn model_fn,
                                    const std::vector<int>& prompt_tokens = {},
                                    int k = -1  // -1 = use config.top_k
    );

    /**
     * Nucleus (Top-p) Sampling
     *
     * Dynamically selects smallest set of tokens with cumulative probability >= p.
     * Adapts to probability distribution shape (conservative when confident,
     * diverse when uncertain).
     *
     * @param model_fn Model forward function
     * @param prompt_tokens Initial token sequence
     * @param p Cumulative probability threshold (0.0-1.0)
     * @return Generated token sequence
     */
    std::vector<int> generate_nucleus(ModelForwardFn model_fn,
                                      const std::vector<int>& prompt_tokens = {},
                                      float p = -1.0f  // -1 = use config.top_p
    );

    /**
     * Combined Sampling (Top-k + Top-p + Temperature)
     *
     * Applies multiple filtering strategies in sequence:
     * 1. Temperature scaling
     * 2. Top-k filtering (if enabled)
     * 3. Top-p filtering (if enabled)
     * 4. Repetition penalty (if enabled)
     * 5. Sample from filtered distribution
     *
     * @param model_fn Model forward function
     * @param prompt_tokens Initial token sequence
     * @return Generated token sequence
     */
    std::vector<int> generate(ModelForwardFn model_fn, const std::vector<int>& prompt_tokens = {});

    /**
     * Generate with string input/output using tokenizer
     *
     * @param model_fn Model forward function
     * @param tokenizer BPE tokenizer for encoding/decoding
     * @param prompt Input text prompt
     * @return Generated text
     */
    std::string generate_text(ModelForwardFn model_fn, BPETokenizer& tokenizer,
                              const std::string& prompt = "");

    /**
     * Batch generation for multiple prompts
     *
     * @param model_fn Model forward function
     * @param tokenizer BPE tokenizer
     * @param prompts Vector of input prompts
     * @return Vector of generated texts
     */
    std::vector<std::string> generate_batch(ModelForwardFn model_fn, BPETokenizer& tokenizer,
                                            const std::vector<std::string>& prompts);

    /**
     * Update generation configuration
     */
    void set_config(const GenerationConfig& new_config);

    /**
     * Get current configuration
     */
    GenerationConfig get_config() const;

    /**
     * Set random seed
     */
    void set_seed(unsigned int seed);
};
