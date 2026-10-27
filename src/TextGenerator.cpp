// @adai-status: stable
// @adai-version: 2.1.1
// @adai-reviewed: 2026-10-26

#include "TextGenerator.hpp"
#include "Activation.hpp"  // TD-216: shared softmax row primitives
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

// Default constructor
TextGenerator::TextGenerator() : rng(std::random_device{}()), tokenizer_(nullptr) {}

// Constructor
TextGenerator::TextGenerator(const GenerationConfig& config, unsigned int seed)
    : config(config), rng(seed == 0 ? std::random_device{}() : seed), tokenizer_(nullptr) {}

// Apply temperature scaling
std::vector<float> TextGenerator::apply_temperature(const std::vector<float>& logits,
                                                    float temperature) {
    // T <= 0 leaves the logits unchanged: every strategy then takes the argmax instead of sampling
    // (TD-515).
    if (temperature <= 0.0f || temperature == 1.0f) {
        return logits;
    }

    std::vector<float> scaled(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled[i] = logits[i] / temperature;
    }
    return scaled;
}

// Apply top-k filtering
std::vector<float> TextGenerator::apply_top_k(const std::vector<float>& logits, int k) {
    if (k <= 0 || k >= static_cast<int>(logits.size())) {
        return logits;
    }

    // Create pairs of (logit, index)
    std::vector<std::pair<float, int>> logit_pairs;
    for (size_t i = 0; i < logits.size(); ++i) {
        logit_pairs.emplace_back(logits[i], static_cast<int>(i));
    }

    // Partial sort to get top k
    std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + k, logit_pairs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Zero out everything except top k
    std::vector<float> filtered(logits.size(), -1e9f);
    for (int i = 0; i < k; ++i) {
        filtered[logit_pairs[i].second] = logit_pairs[i].first;
    }

    return filtered;
}

// Apply nucleus (top-p) sampling
std::vector<float> TextGenerator::apply_top_p(const std::vector<float>& logits, float p) {
    if (p >= 1.0f || logits.empty()) {
        return logits;
    }
    // TD-516: p <= 0 keeps only the most likely token; it used to turn filtering off.
    if (p <= 0.0f) {
        return apply_top_k(logits, 1);
    }

    // Convert to probabilities first
    std::vector<float> probs = softmax(logits);

    // Create pairs and sort by probability (descending)
    std::vector<std::pair<float, int>> prob_pairs;
    for (size_t i = 0; i < probs.size(); ++i) {
        prob_pairs.emplace_back(probs[i], static_cast<int>(i));
    }
    std::sort(prob_pairs.begin(), prob_pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Find cutoff where cumulative probability >= p
    float cumsum = 0.0f;
    std::vector<float> filtered(logits.size(), -1e9f);

    for (const auto& [prob, idx] : prob_pairs) {
        cumsum += prob;
        filtered[idx] = logits[idx];
        if (cumsum >= p) {
            break;
        }
    }

    return filtered;
}

// Apply repetition penalty
//
// TD-066 (fixed): this used to iterate `generated_tokens` and divide/multiply
// a token's logit by `penalty` once per OCCURRENCE, so a token repeated N
// times in the (ever-growing) generated sequence got penalized by a factor of
// penalty^N — compounding multiplicatively and without bound as generation
// continues. The standard algorithm this feature is modeled on (CTRL,
// Keskar et al. 2019; matches HuggingFace transformers' and llama.cpp's
// implementations) penalizes a token once if it has appeared at all,
// regardless of how many times — a simple membership test, not a count.
// Deduplicate first so each previously-generated token is penalized exactly
// once.
std::vector<float> TextGenerator::apply_repetition_penalty(const std::vector<float>& logits,
                                                           const std::vector<int>& generated_tokens,
                                                           float penalty) {
    if (penalty == 1.0f || generated_tokens.empty()) {
        return logits;
    }

    std::vector<float> penalized = logits;
    std::unordered_set<int> seen(generated_tokens.begin(), generated_tokens.end());

    for (int token : seen) {
        if (token >= 0 && token < static_cast<int>(penalized.size())) {
            if (penalized[token] > 0) {
                penalized[token] /= penalty;
            } else {
                penalized[token] *= penalty;
            }
        }
    }

    return penalized;
}

// Softmax (TD-216: the shared row primitive)
std::vector<float> TextGenerator::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    Activation::softmax_row(logits.data(), probs.data(), static_cast<int>(logits.size()));
    return probs;
}

// Sample from probability distribution
int TextGenerator::sample_token(const std::vector<float>& probabilities) {
    std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
    return dist(rng);
}

// Argmax
int TextGenerator::argmax(const std::vector<float>& values) {
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

// Compute length penalty
float TextGenerator::compute_length_penalty(int length, float alpha) {
    // Length penalty: (5 + length)^alpha / (5 + 1)^alpha
    // This is the formula from Google's NMT paper
    return std::pow((5.0f + static_cast<float>(length)) / 6.0f, alpha);
}

// Check if token is stopping token
bool TextGenerator::is_stop_token(int token_id) {
    // Use the utility function from SpecialTokens.hpp
    adai::SpecialTokenConfig token_config(config.pad_token_id, config.unk_token_id,
                                          config.bos_token_id, config.eos_token_id);
    return adai::is_stop_token(token_id, token_config);
}

std::vector<int> TextGenerator::start_tokens(const std::vector<int>& prompt_tokens) const {
    return prompt_tokens.empty() ? std::vector<int>{config.bos_token_id} : prompt_tokens;
}

// TD-516: shared by every strategy. All-NaN logits used to pick token 0 (PAD, a stop token), so a
// broken model (e.g. a truncated checkpoint, TD-423) returned an empty response with no error, and
// min_length was never read.
std::vector<float> TextGenerator::next_logits(const ModelForwardFn& model_fn,
                                              const std::vector<int>& tokens, int new_tokens) {
    Matrix logits = model_fn(tokens);
    if (logits.rows < 1 || logits.cols < 1) {
        throw std::runtime_error("TextGenerator: the model returned no logits");
    }

    // Extract logits for last position [vocab_size]
    const std::vector<float>& last = logits.data[logits.rows - 1];
    bool any_finite = false;
    for (float value : last) {
        if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) {
            throw std::runtime_error("TextGenerator: the model returned non-finite logits");
        }
        any_finite = any_finite || std::isfinite(value);
    }
    if (!any_finite) {
        throw std::runtime_error("TextGenerator: every logit is -inf");
    }

    std::vector<float> next(last.begin(), last.end());
    if (new_tokens < config.min_length) {
        for (int stop : {config.eos_token_id, config.pad_token_id}) {
            if (stop >= 0 && stop < static_cast<int>(next.size())) {
                next[stop] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    return next;
}

// Greedy decoding
std::vector<int> TextGenerator::generate_greedy(const ModelForwardFn& model_fn,
                                                const std::vector<int>& prompt_tokens) {
    std::vector<int> generated = start_tokens(prompt_tokens);

    // TD-516: max_length counts new tokens; it used to count the BOS/prompt too.
    for (int step = 0; step < config.max_length; ++step) {
        std::vector<float> last_logits = next_logits(model_fn, generated, step);

        // Apply repetition penalty if enabled
        if (config.repetition_penalty != 1.0f) {
            last_logits =
                apply_repetition_penalty(last_logits, generated, config.repetition_penalty);
        }

        // Select highest probability token
        int next_token = argmax(last_logits);
        generated.push_back(next_token);

        // Check stopping criteria
        if (is_stop_token(next_token)) {
            break;
        }
    }

    return generated;
}

// Beam search
std::vector<int> TextGenerator::generate_beam_search(const ModelForwardFn& model_fn,
                                                     const std::vector<int>& prompt_tokens,
                                                     int num_beams) {
    if (num_beams == -1) {
        num_beams = config.num_beams;
    }

    if (num_beams == 1) {
        return generate_greedy(model_fn, prompt_tokens);
    }

    // TD-514: a width of 0 used to segfault (beams[0] of an empty vector) and a negative width
    // terminated the process (uncaught length_error). ChatbotAPI passes BEAM_WIDTH straight in.
    if (num_beams < 1) {
        throw std::invalid_argument("TextGenerator: beam width must be at least 1, got " +
                                    std::to_string(num_beams));
    }

    // Initialize beams
    std::vector<BeamHypothesis> beams(num_beams);
    beams[0].tokens = start_tokens(prompt_tokens);
    beams[0].score = 0.0f;
    const int start_length = static_cast<int>(beams[0].tokens.size());
    const int final_length = start_length + std::max(config.max_length, 0);

    // Initialize other beams with very low scores
    for (int i = 1; i < num_beams; ++i) {
        beams[i].tokens = beams[0].tokens;
        beams[i].score = -1e9f;
    }

    std::vector<BeamHypothesis> finished_beams;
    auto normalized = [this](float score, int length) {
        return config.length_penalty
                   ? score / compute_length_penalty(length, config.length_penalty_alpha)
                   : score;
    };

    for (int step = 0; step < config.max_length; ++step) {
        std::vector<std::tuple<float, size_t, int>> candidates;  // (score, beam_idx, token_id)

        // Generate candidates from each beam
        for (size_t beam_idx = 0; beam_idx < beams.size(); ++beam_idx) {
            if (beams[beam_idx].is_finished) {
                continue;
            }

            // Get model predictions for this beam (min_length masks EOS/PAD, TD-516)
            std::vector<float> last_logits = next_logits(model_fn, beams[beam_idx].tokens, step);

            // Convert to log probabilities. Clamped at 0 so a beam's score never rises, which is
            // what makes early stopping exact (the 1e-10 floor could otherwise add a hair).
            std::vector<float> log_probs(last_logits.size());
            std::vector<float> probs = softmax(last_logits);
            for (size_t i = 0; i < probs.size(); ++i) {
                log_probs[i] = std::min(0.0f, std::log(probs[i] + 1e-10f));
            }

            // Get top num_beams tokens
            std::vector<std::pair<float, int>> token_scores;
            for (size_t i = 0; i < log_probs.size(); ++i) {
                token_scores.emplace_back(log_probs[i], static_cast<int>(i));
            }
            std::partial_sort(token_scores.begin(),
                              token_scores.begin() +
                                  std::min(num_beams * 2, static_cast<int>(token_scores.size())),
                              token_scores.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

            // Add candidates
            for (int i = 0; i < std::min(num_beams * 2, static_cast<int>(token_scores.size()));
                 ++i) {
                float token_score = token_scores[i].first;
                int token_id = token_scores[i].second;
                float new_score = beams[beam_idx].score + token_score;

                candidates.emplace_back(new_score, beam_idx, token_id);
            }
        }

        // Sort all candidates by score
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

        // Select top num_beams candidates
        std::vector<BeamHypothesis> new_beams;
        for (size_t i = 0; i < candidates.size() && static_cast<int>(new_beams.size()) < num_beams;
             ++i) {
            auto [score, beam_idx, token_id] = candidates[i];

            BeamHypothesis new_beam;
            new_beam.tokens = beams[beam_idx].tokens;
            new_beam.tokens.push_back(token_id);
            new_beam.score = score;

            // Check if finished (max_length counts new tokens, TD-516)
            if (is_stop_token(token_id) ||
                static_cast<int>(new_beam.tokens.size()) >= final_length) {
                new_beam.is_finished = true;
                new_beam.score = normalized(new_beam.score, static_cast<int>(new_beam.tokens.size()));
                finished_beams.push_back(new_beam);
            } else {
                new_beams.push_back(new_beam);
            }
        }

        beams = new_beams;

        if (beams.empty()) {
            break;
        }

        // TD-516: early stopping. A running beam's raw score only falls (log-probabilities are
        // <= 0), so its best possible final score is its current score divided by the largest
        // length penalty it could still get. When the best finished beam is at least that good
        // for every running beam, nothing can change the result. The flag used to have no effect.
        if (config.early_stopping && !finished_beams.empty()) {
            float best_finished = -std::numeric_limits<float>::infinity();
            for (const auto& beam : finished_beams) {
                best_finished = std::max(best_finished, beam.score);
            }
            bool any_can_win = false;
            for (const auto& beam : beams) {
                const int length = static_cast<int>(beam.tokens.size());
                float bound = beam.score;
                if (config.length_penalty) {
                    const float largest_penalty = std::max(
                        compute_length_penalty(length, config.length_penalty_alpha),
                        compute_length_penalty(final_length, config.length_penalty_alpha));
                    bound = beam.score / largest_penalty;
                }
                if (bound > best_finished) {
                    any_can_win = true;
                    break;
                }
            }
            if (!any_can_win) {
                beams.clear();  // their current-length scores can't win either
                break;
            }
        }
    }

    // Add remaining beams to finished
    for (const auto& beam : beams) {
        BeamHypothesis final_beam = beam;
        final_beam.score = normalized(final_beam.score, static_cast<int>(final_beam.tokens.size()));
        finished_beams.push_back(final_beam);
    }

    // Return best beam
    if (finished_beams.empty()) {
        return start_tokens(prompt_tokens);
    }

    auto best = std::max_element(finished_beams.begin(), finished_beams.end(),
                                 [](const auto& a, const auto& b) { return a.score < b.score; });

    return best->tokens;
}

// Temperature sampling
std::vector<int> TextGenerator::generate_sampling(const ModelForwardFn& model_fn,
                                                  const std::vector<int>& prompt_tokens,
                                                  float temperature) {
    if (temperature == -1.0f) {
        temperature = config.temperature;
    }

    // Temperature 0 or below is greedy (TD-515: a negative temperature used to sample at 1).
    if (temperature <= 0.0f) {
        return generate_greedy(model_fn, prompt_tokens);
    }

    std::vector<int> generated = start_tokens(prompt_tokens);

    for (int step = 0; step < config.max_length; ++step) {
        std::vector<float> last_logits = next_logits(model_fn, generated, step);

        // Apply repetition penalty, then temperature (they commute: the penalty keeps the sign)
        if (config.repetition_penalty != 1.0f) {
            last_logits =
                apply_repetition_penalty(last_logits, generated, config.repetition_penalty);
        }
        last_logits = apply_temperature(last_logits, temperature);

        // Convert to probabilities and sample
        std::vector<float> probs = softmax(last_logits);
        int next_token = sample_token(probs);

        generated.push_back(next_token);

        if (is_stop_token(next_token)) {
            break;
        }
    }

    return generated;
}

// Top-k sampling
std::vector<int> TextGenerator::generate_top_k(const ModelForwardFn& model_fn,
                                               const std::vector<int>& prompt_tokens, int k) {
    if (k == -1) {
        k = config.top_k;
    }

    std::vector<int> generated = start_tokens(prompt_tokens);

    for (int step = 0; step < config.max_length; ++step) {
        std::vector<float> last_logits = next_logits(model_fn, generated, step);

        // TD-516: the repetition penalty runs before filtering, so a penalized token can drop out
        // of the top k (it used to run after, leaving penalized tokens in the candidate set).
        if (config.repetition_penalty != 1.0f) {
            last_logits =
                apply_repetition_penalty(last_logits, generated, config.repetition_penalty);
        }

        // Apply temperature
        last_logits = apply_temperature(last_logits, config.temperature);

        // Apply top-k filtering
        if (k > 0) {
            last_logits = apply_top_k(last_logits, k);
        }

        // TD-515: temperature <= 0 takes the argmax of the filtered logits instead of sampling.
        int next_token = config.temperature <= 0.0f ? argmax(last_logits)
                                                    : sample_token(softmax(last_logits));

        generated.push_back(next_token);

        if (is_stop_token(next_token)) {
            break;
        }
    }

    return generated;
}

// Nucleus (top-p) sampling
std::vector<int> TextGenerator::generate_nucleus(const ModelForwardFn& model_fn,
                                                 const std::vector<int>& prompt_tokens, float p) {
    if (p == -1.0f) {
        p = config.top_p;
    }

    std::vector<int> generated = start_tokens(prompt_tokens);

    for (int step = 0; step < config.max_length; ++step) {
        std::vector<float> last_logits = next_logits(model_fn, generated, step);

        // TD-516: repetition penalty before filtering (see generate_top_k()).
        if (config.repetition_penalty != 1.0f) {
            last_logits =
                apply_repetition_penalty(last_logits, generated, config.repetition_penalty);
        }

        // Apply temperature
        last_logits = apply_temperature(last_logits, config.temperature);

        // Apply top-p filtering
        if (p < 1.0f) {
            last_logits = apply_top_p(last_logits, p);
        }

        // TD-515: temperature <= 0 takes the argmax of the filtered logits instead of sampling.
        int next_token = config.temperature <= 0.0f ? argmax(last_logits)
                                                    : sample_token(softmax(last_logits));

        generated.push_back(next_token);

        if (is_stop_token(next_token)) {
            break;
        }
    }

    return generated;
}

// Combined generation (all filters)
std::vector<int> TextGenerator::generate(const ModelForwardFn& model_fn,
                                         const std::vector<int>& prompt_tokens) {
    // Use beam search if num_beams > 1
    if (config.num_beams > 1) {
        return generate_beam_search(model_fn, prompt_tokens);
    }

    // Otherwise use sampling with all filters
    std::vector<int> generated = start_tokens(prompt_tokens);

    for (int step = 0; step < config.max_length; ++step) {
        std::vector<float> last_logits = next_logits(model_fn, generated, step);

        // TD-516: repetition penalty first, then temperature and the filters.
        if (config.repetition_penalty != 1.0f) {
            last_logits =
                apply_repetition_penalty(last_logits, generated, config.repetition_penalty);
        }

        last_logits = apply_temperature(last_logits, config.temperature);

        if (config.top_k > 0) {
            last_logits = apply_top_k(last_logits, config.top_k);
        }

        if (config.top_p < 1.0f) {
            last_logits = apply_top_p(last_logits, config.top_p);
        }

        // Sample or greedy (TD-515: temperature <= 0 is greedy, as in every strategy)
        int next_token;
        if (config.temperature <= 0.0f) {
            next_token = argmax(last_logits);
        } else {
            std::vector<float> probs = softmax(last_logits);
            next_token = sample_token(probs);
        }

        generated.push_back(next_token);

        if (is_stop_token(next_token)) {
            break;
        }
    }

    return generated;
}

// Generate text from prompt
std::string TextGenerator::generate_text(ModelForwardFn model_fn, BPETokenizer& tokenizer,
                                         const std::string& prompt) {
    // Encode prompt
    std::vector<int> prompt_tokens;
    if (!prompt.empty()) {
        prompt_tokens = tokenizer.encode(prompt, false);  // Don't add special tokens
    }

    // Generate
    std::vector<int> generated = generate(model_fn, prompt_tokens);

    // Decode (skip special tokens)
    return tokenizer.decode(generated, true);
}

// Batch generation
std::vector<std::string> TextGenerator::generate_batch(const ModelForwardFn& model_fn,
                                                       BPETokenizer& tokenizer,
                                                       const std::vector<std::string>& prompts) {
    std::vector<std::string> results;
    results.reserve(prompts.size());

    for (const auto& prompt : prompts) {
        results.push_back(generate_text(model_fn, tokenizer, prompt));
    }

    return results;
}

// Set configuration
void TextGenerator::set_config(const GenerationConfig& new_config) {
    config = new_config;
}

// Get configuration
TextGenerator::GenerationConfig TextGenerator::get_config() const {
    return config;
}

// Set random seed
void TextGenerator::set_seed(unsigned int seed) {
    rng.seed(seed);
}

// Stored model function / tokenizer (used by SpeculativeDecoder)
void TextGenerator::set_model_fn(ModelForwardFn fn) {
    model_fn_ = std::move(fn);
}

void TextGenerator::set_tokenizer(BPETokenizer* tok) {
    tokenizer_ = tok;
}

BPETokenizer* TextGenerator::get_tokenizer() const {
    return tokenizer_;
}

// Get probability distribution for the next token given context tokens.
// Requires that set_model_fn() was called beforehand.
Matrix TextGenerator::get_logits(const std::vector<int>& context) {
    if (!model_fn_) {
        throw std::runtime_error("TextGenerator: no model function set (call set_model_fn first)");
    }
    Matrix logits = model_fn_(context);
    if (logits.rows < 1 || logits.cols < 1) {
        throw std::runtime_error("TextGenerator: the model returned no logits");
    }
    return logits;
}

std::vector<float> TextGenerator::get_next_token_probs(const std::vector<int>& context) {
    Matrix logits_mat = get_logits(context);
    // Extract the last row of logits (next-token prediction)
    int last_row = logits_mat.rows - 1;
    std::vector<float> last_logits(logits_mat.cols);
    for (int j = 0; j < logits_mat.cols; ++j) {
        last_logits[j] = logits_mat(last_row, j);
    }
    return softmax(last_logits);
}
