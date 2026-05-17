#include "TextGenerator.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <queue>

// Default constructor
TextGenerator::TextGenerator() : config(), rng(std::random_device{}()), tokenizer_(nullptr) {}

// Constructor
TextGenerator::TextGenerator(const GenerationConfig& cfg, unsigned int seed)
    : config(cfg), rng(seed == 0 ? std::random_device{}() : seed), tokenizer_(nullptr) {}

// Apply temperature scaling
std::vector<float> TextGenerator::apply_temperature(const std::vector<float>& logits,
                                                    float temperature) {
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
        logit_pairs.push_back({logits[i], static_cast<int>(i)});
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
    if (p >= 1.0f || p <= 0.0f) {
        return logits;
    }

    // Convert to probabilities first
    std::vector<float> probs = softmax(logits);

    // Create pairs and sort by probability (descending)
    std::vector<std::pair<float, int>> prob_pairs;
    for (size_t i = 0; i < probs.size(); ++i) {
        prob_pairs.push_back({probs[i], static_cast<int>(i)});
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
std::vector<float> TextGenerator::apply_repetition_penalty(const std::vector<float>& logits,
                                                           const std::vector<int>& generated_tokens,
                                                           float penalty) {
    if (penalty == 1.0f || generated_tokens.empty()) {
        return logits;
    }

    std::vector<float> penalized = logits;

    for (int token : generated_tokens) {
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

// Softmax
std::vector<float> TextGenerator::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());

    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());

    // Compute exp(x - max)
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }

    // Normalize
    if (sum > 0.0f) {
        for (float& p : probs) {
            p /= sum;
        }
    }

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
    return std::pow((5.0f + length) / 6.0f, alpha);
}

// Check if token is stopping token
bool TextGenerator::is_stop_token(int token_id) {
    // Use the utility function from SpecialTokens.hpp
    adai::SpecialTokenConfig token_config(config.pad_token_id, config.unk_token_id,
                                          config.bos_token_id, config.eos_token_id);
    return adai::is_stop_token(token_id, token_config);
}

// Greedy decoding
std::vector<int> TextGenerator::generate_greedy(ModelForwardFn model_fn,
                                                const std::vector<int>& prompt_tokens) {
    std::vector<int> generated = prompt_tokens;

    // Add <bos> if empty
    if (generated.empty()) {
        generated.push_back(config.bos_token_id);
    }

    for (int step = 0; step < config.max_length; ++step) {
        // Get model predictions
        Matrix logits = model_fn(generated);

        // Extract logits for last position [vocab_size]
        int last_pos = logits.rows - 1;
        std::vector<float> last_logits(logits.cols);
        for (int i = 0; i < logits.cols; ++i) {
            last_logits[i] = logits.data[last_pos][i];
        }

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

        if (static_cast<int>(generated.size()) >= config.max_length) {
            break;
        }
    }

    return generated;
}

// Beam search
std::vector<int> TextGenerator::generate_beam_search(ModelForwardFn model_fn,
                                                     const std::vector<int>& prompt_tokens,
                                                     int num_beams) {
    if (num_beams == -1) {
        num_beams = config.num_beams;
    }

    if (num_beams == 1) {
        return generate_greedy(model_fn, prompt_tokens);
    }

    // Initialize beams
    std::vector<BeamHypothesis> beams(num_beams);
    beams[0].tokens = prompt_tokens;
    if (beams[0].tokens.empty()) {
        beams[0].tokens.push_back(config.bos_token_id);
    }
    beams[0].score = 0.0f;

    // Initialize other beams with very low scores
    for (int i = 1; i < num_beams; ++i) {
        beams[i].tokens = beams[0].tokens;
        beams[i].score = -1e9f;
    }

    std::vector<BeamHypothesis> finished_beams;

    for (int step = 0; step < config.max_length; ++step) {
        std::vector<std::tuple<float, size_t, int>> candidates;  // (score, beam_idx, token_id)

        // Generate candidates from each beam
        for (size_t beam_idx = 0; beam_idx < beams.size(); ++beam_idx) {
            if (beams[beam_idx].is_finished) {
                continue;
            }

            // Get model predictions for this beam
            Matrix logits = model_fn(beams[beam_idx].tokens);

            // Extract last position logits
            int last_pos = logits.rows - 1;
            std::vector<float> last_logits(logits.cols);
            for (int i = 0; i < logits.cols; ++i) {
                last_logits[i] = logits.data[last_pos][i];
            }

            // Convert to log probabilities
            std::vector<float> log_probs(last_logits.size());
            std::vector<float> probs = softmax(last_logits);
            for (size_t i = 0; i < probs.size(); ++i) {
                log_probs[i] = std::log(probs[i] + 1e-10f);
            }

            // Get top num_beams tokens
            std::vector<std::pair<float, int>> token_scores;
            for (size_t i = 0; i < log_probs.size(); ++i) {
                token_scores.push_back({log_probs[i], static_cast<int>(i)});
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

                candidates.push_back({new_score, beam_idx, token_id});
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

            // Check if finished
            if (is_stop_token(token_id) ||
                static_cast<int>(new_beam.tokens.size()) >= config.max_length) {
                new_beam.is_finished = true;

                // Apply length penalty if enabled
                if (config.length_penalty) {
                    new_beam.score /= compute_length_penalty(
                        static_cast<int>(new_beam.tokens.size()), config.length_penalty_alpha);
                }

                finished_beams.push_back(new_beam);
            } else {
                new_beams.push_back(new_beam);
            }
        }

        beams = new_beams;

        // Early stopping if all beams finished
        if (config.early_stopping && beams.empty()) {
            break;
        }

        if (beams.empty()) {
            break;
        }
    }

    // Add remaining beams to finished
    for (const auto& beam : beams) {
        BeamHypothesis final_beam = beam;
        if (config.length_penalty) {
            final_beam.score /= compute_length_penalty(static_cast<int>(final_beam.tokens.size()),
                                                       config.length_penalty_alpha);
        }
        finished_beams.push_back(final_beam);
    }

    // Return best beam
    if (finished_beams.empty()) {
        return prompt_tokens.empty() ? std::vector<int>{config.bos_token_id} : prompt_tokens;
    }

    auto best = std::max_element(finished_beams.begin(), finished_beams.end(),
                                 [](const auto& a, const auto& b) { return a.score < b.score; });

    return best->tokens;
}

// Temperature sampling
std::vector<int> TextGenerator::generate_sampling(ModelForwardFn model_fn,
                                                  const std::vector<int>& prompt_tokens,
                                                  float temperature) {
    if (temperature == -1.0f) {
        temperature = config.temperature;
    }

    // If temperature is 0, use greedy
    if (temperature == 0.0f) {
        return generate_greedy(model_fn, prompt_tokens);
    }

    std::vector<int> generated = prompt_tokens;
    if (generated.empty()) {
        generated.push_back(config.bos_token_id);
    }

    for (int step = 0; step < config.max_length; ++step) {
        Matrix logits = model_fn(generated);

        // Extract last position logits
        int last_pos = logits.rows - 1;
        std::vector<float> last_logits(logits.cols);
        for (int i = 0; i < logits.cols; ++i) {
            last_logits[i] = logits.data[last_pos][i];
        }

        // Apply temperature
        last_logits = apply_temperature(last_logits, temperature);

        // Apply repetition penalty
        if (config.repetition_penalty != 1.0f) {
            last_logits =
                apply_repetition_penalty(last_logits, generated, config.repetition_penalty);
        }

        // Convert to probabilities and sample
        std::vector<float> probs = softmax(last_logits);
        int next_token = sample_token(probs);

        generated.push_back(next_token);

        if (is_stop_token(next_token) || static_cast<int>(generated.size()) >= config.max_length) {
            break;
        }
    }

    return generated;
}

// Top-k sampling
std::vector<int> TextGenerator::generate_top_k(ModelForwardFn model_fn,
                                               const std::vector<int>& prompt_tokens, int k) {
    if (k == -1) {
        k = config.top_k;
    }

    std::vector<int> generated = prompt_tokens;
    if (generated.empty()) {
        generated.push_back(config.bos_token_id);
    }

    for (int step = 0; step < config.max_length; ++step) {
        Matrix logits = model_fn(generated);

        int last_pos = logits.rows - 1;
        std::vector<float> last_logits(logits.cols);
        for (int i = 0; i < logits.cols; ++i) {
            last_logits[i] = logits.data[last_pos][i];
        }

        // Apply temperature
        last_logits = apply_temperature(last_logits, config.temperature);

        // Apply top-k filtering
        if (k > 0) {
            last_logits = apply_top_k(last_logits, k);
        }

        // Apply repetition penalty
        if (config.repetition_penalty != 1.0f) {
            last_logits =
                apply_repetition_penalty(last_logits, generated, config.repetition_penalty);
        }

        std::vector<float> probs = softmax(last_logits);
        int next_token = sample_token(probs);

        generated.push_back(next_token);

        if (is_stop_token(next_token) || static_cast<int>(generated.size()) >= config.max_length) {
            break;
        }
    }

    return generated;
}

// Nucleus (top-p) sampling
std::vector<int> TextGenerator::generate_nucleus(ModelForwardFn model_fn,
                                                 const std::vector<int>& prompt_tokens, float p) {
    if (p == -1.0f) {
        p = config.top_p;
    }

    std::vector<int> generated = prompt_tokens;
    if (generated.empty()) {
        generated.push_back(config.bos_token_id);
    }

    for (int step = 0; step < config.max_length; ++step) {
        Matrix logits = model_fn(generated);

        int last_pos = logits.rows - 1;
        std::vector<float> last_logits(logits.cols);
        for (int i = 0; i < logits.cols; ++i) {
            last_logits[i] = logits.data[last_pos][i];
        }

        // Apply temperature
        last_logits = apply_temperature(last_logits, config.temperature);

        // Apply top-p filtering
        if (p < 1.0f) {
            last_logits = apply_top_p(last_logits, p);
        }

        // Apply repetition penalty
        if (config.repetition_penalty != 1.0f) {
            last_logits =
                apply_repetition_penalty(last_logits, generated, config.repetition_penalty);
        }

        std::vector<float> probs = softmax(last_logits);
        int next_token = sample_token(probs);

        generated.push_back(next_token);

        if (is_stop_token(next_token) || static_cast<int>(generated.size()) >= config.max_length) {
            break;
        }
    }

    return generated;
}

// Combined generation (all filters)
std::vector<int> TextGenerator::generate(ModelForwardFn model_fn,
                                         const std::vector<int>& prompt_tokens) {
    // Use beam search if num_beams > 1
    if (config.num_beams > 1) {
        return generate_beam_search(model_fn, prompt_tokens);
    }

    // Otherwise use sampling with all filters
    std::vector<int> generated = prompt_tokens;
    if (generated.empty()) {
        generated.push_back(config.bos_token_id);
    }

    for (int step = 0; step < config.max_length; ++step) {
        Matrix logits = model_fn(generated);

        int last_pos = logits.rows - 1;
        std::vector<float> last_logits(logits.cols);
        for (int i = 0; i < logits.cols; ++i) {
            last_logits[i] = logits.data[last_pos][i];
        }

        // Apply all filters in sequence
        last_logits = apply_temperature(last_logits, config.temperature);

        if (config.top_k > 0) {
            last_logits = apply_top_k(last_logits, config.top_k);
        }

        if (config.top_p < 1.0f) {
            last_logits = apply_top_p(last_logits, config.top_p);
        }

        if (config.repetition_penalty != 1.0f) {
            last_logits =
                apply_repetition_penalty(last_logits, generated, config.repetition_penalty);
        }

        // Sample or greedy
        int next_token;
        if (config.temperature == 0.0f) {
            next_token = argmax(last_logits);
        } else {
            std::vector<float> probs = softmax(last_logits);
            next_token = sample_token(probs);
        }

        generated.push_back(next_token);

        if (is_stop_token(next_token) || static_cast<int>(generated.size()) >= config.max_length) {
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
std::vector<std::string> TextGenerator::generate_batch(ModelForwardFn model_fn,
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
std::vector<float> TextGenerator::get_next_token_probs(const std::vector<int>& context) {
    if (!model_fn_) {
        throw std::runtime_error("TextGenerator: no model function set (call set_model_fn first)");
    }
    Matrix logits_mat = model_fn_(context);
    // Extract the last row of logits (next-token prediction)
    int last_row = logits_mat.rows - 1;
    std::vector<float> last_logits(logits_mat.cols);
    for (int j = 0; j < logits_mat.cols; ++j) {
        last_logits[j] = logits_mat(last_row, j);
    }
    return softmax(last_logits);
}
