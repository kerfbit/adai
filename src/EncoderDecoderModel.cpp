// @adai-status: beta        (capped by TD-050 — see TECHNICAL_DEBT.md)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-08

#include "EncoderDecoderModel.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <sstream>
#include <stdexcept>
#include "Optimizer.hpp"
#include "SpecialTokens.hpp"

namespace {
// Checkpoint .config format marker. Negative so it can never collide with a
// legacy (pre-marker) file's first field, vocab_size, which is always > 0 —
// this gives an unambiguous rejection instead of a coincidental dimension
// mismatch. Bump kConfigFormatVersion on any future change to the
// EncoderBlock/DecoderBlock forward-pass math that makes existing checkpoint
// weights semantically incompatible even though their shapes are unchanged
// (e.g. this version: switching Post-LN -> Pre-LN).
constexpr int32_t kConfigMagic = -20260721;
constexpr int32_t kConfigFormatVersion = 2;  // 2 = Pre-LN EncoderBlock/DecoderBlock
}  // namespace

// Constructor
EncoderDecoderModel::EncoderDecoderModel(int vocab_size, int d_model, int encoder_layers,
                                         int decoder_layers, int num_heads, int d_ff,
                                         int max_seq_length)
    : vocab_size(vocab_size),
      d_model(d_model),
      encoder_layers(encoder_layers),
      decoder_layers(decoder_layers),
      num_heads(num_heads),
      d_ff(d_ff),
      max_seq_length(max_seq_length) {
    // Initialize tokenizer (will be set externally or use default)
    tokenizer = std::make_unique<BPETokenizer>();

    // Initialize encoder
    encoder = std::make_unique<LLMEncoder>(vocab_size, d_model, encoder_layers, num_heads, d_ff,
                                           max_seq_length);

    // Initialize decoder
    decoder = std::make_unique<LLMDecoder>(vocab_size, d_model, decoder_layers, num_heads, d_ff,
                                           max_seq_length);

    // Initialize language model head
    lm_head = std::make_unique<LanguageModelHead>(d_model, vocab_size);

    // Initialize text generator with default config
    TextGenerator::GenerationConfig gen_config;
    gen_config.max_length = max_seq_length;
    gen_config.bos_token_id = adai::SpecialTokenIDs::BOS;  // <bos> token
    gen_config.eos_token_id = adai::SpecialTokenIDs::EOS;  // <eos> token
    gen_config.pad_token_id = adai::SpecialTokenIDs::PAD;  // <pad> token

    generator = std::make_unique<TextGenerator>(gen_config, 42);

    // Set special token IDs
    bos_token_id = gen_config.bos_token_id;
    eos_token_id = gen_config.eos_token_id;
    pad_token_id = gen_config.pad_token_id;
}

// Destructor
EncoderDecoderModel::~EncoderDecoderModel() {
    // Unique pointers handle cleanup
}

// Compute cross-entropy loss
float EncoderDecoderModel::compute_loss(const Matrix& logits,
                                        const std::vector<int>& target_tokens) {
    int seq_length = logits.rows;
    float total_loss = 0.0f;

    for (int t = 0; t < seq_length && t < static_cast<int>(target_tokens.size()); ++t) {
        // Softmax for current timestep
        std::vector<float> probs(vocab_size);
        float max_logit = logits.data[t][0];
        for (int v = 1; v < vocab_size; ++v) {
            max_logit = std::max(max_logit, logits.data[t][v]);
        }

        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            probs[v] = std::exp(logits.data[t][v] - max_logit);
            sum_exp += probs[v];
        }

        for (int v = 0; v < vocab_size; ++v) {
            probs[v] /= sum_exp;
        }

        // Cross-entropy loss for this timestep
        int target = target_tokens[t];
        if (target >= 0 && target < vocab_size) {
            total_loss -= std::log(probs[target] + 1e-10f);
        }
    }

    return total_loss / static_cast<float>(seq_length);
}

// Compute loss gradient
Matrix EncoderDecoderModel::compute_loss_gradient(const Matrix& logits,
                                                  const std::vector<int>& target_tokens) {
    int seq_length = logits.rows;
    Matrix grad(seq_length, vocab_size);

    for (int t = 0; t < seq_length && t < static_cast<int>(target_tokens.size()); ++t) {
        // Softmax
        std::vector<float> probs(vocab_size);
        float max_logit = logits.data[t][0];
        for (int v = 1; v < vocab_size; ++v) {
            max_logit = std::max(max_logit, logits.data[t][v]);
        }

        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            probs[v] = std::exp(logits.data[t][v] - max_logit);
            sum_exp += probs[v];
        }

        for (int v = 0; v < vocab_size; ++v) {
            probs[v] /= sum_exp;
        }

        // Gradient: softmax - one_hot(target)
        int target = target_tokens[t];
        for (int v = 0; v < vocab_size; ++v) {
            grad.data[t][v] = probs[v];
            if (v == target && target >= 0 && target < vocab_size) {
                grad.data[t][v] -= 1.0f;
            }
        }

        // Scale by sequence length
        for (int v = 0; v < vocab_size; ++v) {
            grad.data[t][v] /= static_cast<float>(seq_length);
        }
    }

    return grad;
}

// Generate response
std::string EncoderDecoderModel::generate_response(const std::string& input_text, int max_length) {
    // Encode input (no special tokens for encoder input)
    std::vector<int> input_tokens = tokenizer->encode(input_text, false);
    int input_len = static_cast<int>(input_tokens.size());
    Matrix encoder_mask(input_len, input_len);
    for (int i = 0; i < input_len; ++i) {
        for (int j = 0; j < input_len; ++j) {
            encoder_mask.data[i][j] = 1.0f;  // No masking for encoder
        }
    }
    cached_encoder_output = encoder->encode_with_mask(input_tokens, encoder_mask);

    // Initialize KV cache for efficient generation
    DecoderKVCache kv_cache(decoder_layers);
    size_t processed_length = 0;

    // Create model forward function with KV caching
    auto model_fn = [this, &kv_cache, &processed_length](const std::vector<int>& tokens) -> Matrix {
        size_t current_length = tokens.size();

        // Determine which tokens are new (not yet processed)
        std::vector<int> new_tokens(tokens.begin() + static_cast<std::ptrdiff_t>(processed_length),
                                    tokens.end());

        // Process only new tokens with cache
        Matrix decoder_out =
            decoder->forward_with_cache(new_tokens, kv_cache, &cached_encoder_output, true);

        // Update processed length
        processed_length = current_length;

        // Project to vocabulary (last position of output)
        Matrix logits = lm_head->forward(decoder_out);

        return logits;
    };

    // Generate using TextGenerator with greedy or configured strategy
    std::vector<int> output_tokens =
        generator->generate(model_fn, {bos_token_id}  // Start with <bos>
        );

    // Decode tokens to text (skip special tokens like <bos>, <eos>, <unk>, <pad>)
    std::string response = tokenizer->decode(output_tokens, true);

    return response;
}

// Generate with specific strategy
std::string EncoderDecoderModel::generate_response_with_strategy(const std::string& input_text,
                                                                 int max_length,
                                                                 const std::string& strategy,
                                                                 float temperature, int top_k,
                                                                 float top_p, int num_beams) {
    // Ensure special token IDs are synced with tokenizer
    sync_special_tokens();

    // Normalize strategy name (handle hyphens)
    std::string normalized_strategy = strategy;
    if (normalized_strategy == "top-k") {
        normalized_strategy = "topk";
    } else if (normalized_strategy == "top-p" || normalized_strategy == "nucleus") {
        normalized_strategy = "nucleus";  // Standardize to "nucleus"
    }

    // Encode input (no special tokens for encoder input)
    std::vector<int> input_tokens = tokenizer->encode(input_text, false);
    int input_len = static_cast<int>(input_tokens.size());
    Matrix encoder_mask(input_len, input_len);
    for (int i = 0; i < input_len; ++i) {
        for (int j = 0; j < input_len; ++j) {
            encoder_mask.data[i][j] = 1.0f;
        }
    }
    cached_encoder_output = encoder->encode_with_mask(input_tokens, encoder_mask);

    // Generate based on strategy
    std::vector<int> output_tokens;

    // IMPORTANT: Beam search requires different handling because each beam has its own sequence
    // and KV caching doesn't work when we need to explore multiple hypotheses simultaneously
    if (normalized_strategy == "beam") {
        // Update config for beam search
        TextGenerator::GenerationConfig config = generator->get_config();
        config.num_beams = num_beams;
        config.max_length = max_length;
        generator->set_config(config);

        // Get actual tokenizer vocab size to mask invalid tokens
        int actual_vocab_size = static_cast<int>(tokenizer->get_vocab_size());

        // Create model function WITHOUT KV caching for beam search
        // Each beam has independent token sequences, so we can't share a cache
        auto beam_model_fn = [this, actual_vocab_size](const std::vector<int>& tokens) -> Matrix {
            // Process all tokens from scratch (no caching)
            Matrix decoder_out = decoder->forward_with_encoder(tokens, cached_encoder_output);

            // Project to vocabulary (last position of output)
            Matrix logits = lm_head->forward(decoder_out);

            // Mask out invalid token IDs beyond actual vocabulary size
            // This prevents generation of tokens that don't exist in the tokenizer
            if (actual_vocab_size < logits.cols) {
                for (int i = 0; i < logits.rows; ++i) {
                    for (int j = actual_vocab_size; j < logits.cols; ++j) {
                        logits.data[i][j] = -1e9f;  // Set to very negative to make prob ~0
                    }
                }
            }

            return logits;
        };

        output_tokens = generator->generate_beam_search(beam_model_fn, {bos_token_id});

        // Decode tokens to text (skip special tokens like <bos>, <eos>, <unk>, <pad>)
        return tokenizer->decode(output_tokens, true);
    }

    // For non-beam strategies, use KV caching for efficiency
    // Initialize KV cache for efficient generation
    DecoderKVCache kv_cache(decoder_layers);
    size_t processed_length = 0;

    // Get actual tokenizer vocab size to mask invalid tokens
    int actual_vocab_size = static_cast<int>(tokenizer->get_vocab_size());

    // Create model forward function with KV caching
    auto model_fn = [this, &kv_cache, &processed_length,
                     actual_vocab_size](const std::vector<int>& tokens) -> Matrix {
        size_t current_length = tokens.size();

        // Determine which tokens are new (not yet processed)
        std::vector<int> new_tokens(tokens.begin() + static_cast<std::ptrdiff_t>(processed_length),
                                    tokens.end());

        // Process only new tokens with cache
        Matrix decoder_out =
            decoder->forward_with_cache(new_tokens, kv_cache, &cached_encoder_output, true);

        // Update processed length
        processed_length = current_length;

        // Project to vocabulary (last position of output)
        Matrix logits = lm_head->forward(decoder_out);

        // Mask out invalid token IDs beyond actual vocabulary size
        // This prevents generation of tokens that don't exist in the tokenizer
        if (actual_vocab_size < logits.cols) {
            for (int i = 0; i < logits.rows; ++i) {
                for (int j = actual_vocab_size; j < logits.cols; ++j) {
                    logits.data[i][j] = -1e9f;  // Set to very negative to make prob ~0
                }
            }
        }

        return logits;
    };

    // Generate based on strategy

    if (normalized_strategy == "greedy") {
        // WORKAROUND: Use non-cached path for greedy due to KV cache bug
        // TODO: See TECHNICAL_DEBT.md TD-050 - Fix KV cache to properly handle autoregressive
        //       generation (self-attention/cross-attention indexing bug), then build the
        //       GPU-resident cache on top of the corrected model.
        int actual_vocab_size = static_cast<int>(tokenizer->get_vocab_size());
        auto greedy_model_fn = [this, actual_vocab_size](const std::vector<int>& tokens) -> Matrix {
            Matrix decoder_out = decoder->forward_with_encoder(tokens, cached_encoder_output);
            Matrix logits = lm_head->forward(decoder_out);
            if (actual_vocab_size < logits.cols) {
                for (int i = 0; i < logits.rows; ++i) {
                    for (int j = actual_vocab_size; j < logits.cols; ++j) {
                        logits.data[i][j] = -1e9f;
                    }
                }
            }
            return logits;
        };
        output_tokens = generator->generate_greedy(greedy_model_fn, {bos_token_id});
    } else if (normalized_strategy == "sampling") {
        output_tokens = generator->generate_sampling(model_fn, {bos_token_id}, temperature);
    } else if (normalized_strategy == "topk") {
        output_tokens = generator->generate_top_k(model_fn, {bos_token_id}, top_k);
    } else if (normalized_strategy == "nucleus") {
        output_tokens = generator->generate_nucleus(model_fn, {bos_token_id}, top_p);
    } else {
        // Default to main generate method (uses config)
        output_tokens = generator->generate(model_fn, {bos_token_id});
    }

    // Decode tokens to text (skip special tokens like <bos>, <eos>, <unk>, <pad>)
    return tokenizer->decode(output_tokens, true);
}

// Training step
float EncoderDecoderModel::train_step(const std::string& input_text,
                                      const std::string& target_text) {
    std::vector<int> input_tokens =
        tokenizer->encode(input_text, false);  // Encoder: no special tokens
    std::vector<int> target_tokens =
        tokenizer->encode(target_text, true);  // Decoder: with special tokens

    return train_step_tokenized(input_tokens, target_tokens);
}

// Training step on tokenized sequences
float EncoderDecoderModel::train_step_tokenized(const std::vector<int>& input_tokens,
                                                const std::vector<int>& target_tokens) {
    if (!requires_grad) {
        throw std::runtime_error("Model is not in training mode");
    }

    // Zero gradients
    zero_grad();

    // Forward pass
    Matrix logits = forward(input_tokens, target_tokens);

    // Compute loss
    float loss = compute_loss(logits, target_tokens);

    // Backward pass
    Matrix grad_loss = compute_loss_gradient(logits, target_tokens);
    backward(grad_loss);

    // Update weights
    update_weights();

    return loss;
}

// Evaluate (no gradients)
float EncoderDecoderModel::evaluate(const std::string& input_text, const std::string& target_text) {
    // encode() has no length cap, unlike the truncate() step ChatbotTrainer
    // applies when first tokenizing the dataset — prefer evaluate_tokenized()
    // with already-truncated ids whenever they're available.
    std::vector<int> input_tokens =
        tokenizer->encode(input_text, false);  // Encoder: no special tokens
    std::vector<int> target_tokens =
        tokenizer->encode(target_text, true);  // Decoder: with special tokens

    return evaluate_tokenized(input_tokens, target_tokens);
}

float EncoderDecoderModel::evaluate_tokenized(const std::vector<int>& input_tokens,
                                              const std::vector<int>& target_tokens) {
    bool prev_mode = requires_grad;
    set_training(false);

    // Forward pass only
    Matrix logits = forward(input_tokens, target_tokens);
    float loss = compute_loss(logits, target_tokens);

    set_training(prev_mode);
    return loss;
}

// Compute perplexity
float EncoderDecoderModel::compute_perplexity(const std::vector<std::string>& input_texts,
                                              const std::vector<std::string>& target_texts) {
    if (input_texts.size() != target_texts.size()) {
        throw std::invalid_argument("Input and target sizes must match");
    }

    float total_loss = 0.0f;
    int num_samples = static_cast<int>(input_texts.size());

    bool prev_mode = requires_grad;
    set_training(false);

    for (size_t i = 0; i < input_texts.size(); ++i) {
        total_loss += evaluate(input_texts[i], target_texts[i]);
    }

    set_training(prev_mode);

    float avg_loss = total_loss / static_cast<float>(num_samples);
    return std::exp(avg_loss);  // Perplexity = exp(cross_entropy)
}

// Set training mode
void EncoderDecoderModel::set_training(bool mode) {
    requires_grad = mode;
    encoder->set_requires_grad(mode);
    decoder->set_training(mode);
}

// Set learning rate
void EncoderDecoderModel::set_learning_rate(float lr) {
    learning_rate = lr;
    // encoder->set_learning_rate(lr);  // LLMEncoder doesn't have this method
    decoder->set_learning_rate(lr);
}

// Update weights
void EncoderDecoderModel::update_weights() {
    // encoder->update_weights(learning_rate);  // LLMEncoder doesn't have this method
    decoder->update_weights(learning_rate);
    lm_head->update_weights();  // LanguageModelHead doesn't take lr parameter
}

// Zero gradients
void EncoderDecoderModel::zero_grad() {
    encoder->zero_grad();
    decoder->zero_grad();
    lm_head->zero_grad();
}

// Register parameters with external optimizer
void EncoderDecoderModel::register_parameters(Optimizer& optimizer) {
    // Register encoder parameters
    encoder->register_parameters_with_optimizer(optimizer);

    // Register decoder parameters
    decoder->register_parameters_with_optimizer(optimizer);

    // Register language model head parameters
    lm_head->set_optimizer(&optimizer);
}

// Backward pass without weight update (for use with external optimizer)
void EncoderDecoderModel::backward_pass(const Matrix& grad_output) {
    // Standard backward pass
    backward(grad_output);
    // Note: No weight update - handled by external optimizer
}

// Set tokenizer
void EncoderDecoderModel::set_tokenizer(BPETokenizer* tokenizer_ptr) {
    tokenizer.reset(tokenizer_ptr);
    sync_special_tokens();
}

// Sync special token IDs from tokenizer
void EncoderDecoderModel::sync_special_tokens() {
    if (!tokenizer) {
        return;
    }

    TextGenerator::GenerationConfig config = generator->get_config();
    config.bos_token_id = tokenizer->get_bos_token_id();
    config.eos_token_id = tokenizer->get_eos_token_id();
    config.pad_token_id = tokenizer->get_pad_token_id();
    set_generation_config(config);
}

// Set generation config
void EncoderDecoderModel::set_generation_config(const TextGenerator::GenerationConfig& config) {
    generator->set_config(config);
    bos_token_id = config.bos_token_id;
    eos_token_id = config.eos_token_id;
    pad_token_id = config.pad_token_id;
}

// Get generation config
TextGenerator::GenerationConfig EncoderDecoderModel::get_generation_config() const {
    return generator->get_config();
}

// Save model
void EncoderDecoderModel::save_model(const std::string& filepath) const {
    // Save architecture config
    std::ofstream config_file(filepath + ".config", std::ios::binary);
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open config file for writing: " + filepath + ".config");
    }
    config_file.write(reinterpret_cast<const char*>(&kConfigMagic), sizeof(int32_t));
    config_file.write(reinterpret_cast<const char*>(&kConfigFormatVersion), sizeof(int32_t));
    config_file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&encoder_layers), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&decoder_layers), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&d_ff), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&max_seq_length), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&bos_token_id), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&eos_token_id), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&pad_token_id), sizeof(int));
    config_file.close();

    // Save tokenizer vocabulary
    tokenizer->save_vocab(filepath + ".vocab");

    // Save encoder weights
    encoder->save_weights(filepath + ".encoder");

    // Save decoder weights
    decoder->save_weights(filepath + ".decoder");

    // Save language model head
    lm_head->save_weights(filepath + ".lm_head");
}

// Load model
void EncoderDecoderModel::load_model(const std::string& filepath) {
    // Load architecture config
    std::ifstream config_file(filepath + ".config", std::ios::binary);
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to load model config");
    }

    int32_t magic = 0, format_version = 0;
    config_file.read(reinterpret_cast<char*>(&magic), sizeof(int32_t));
    config_file.read(reinterpret_cast<char*>(&format_version), sizeof(int32_t));
    if (magic != kConfigMagic) {
        throw std::runtime_error(
            "Checkpoint format marker missing/invalid — this looks like a pre-Pre-LN (Post-LN) "
            "checkpoint saved before the architecture change and is not compatible with the "
            "current Pre-LN EncoderBlock/DecoderBlock math (same weight shapes, different "
            "meaning — loading it would silently produce garbage). Retrain from scratch.");
    }
    if (format_version < kConfigFormatVersion) {
        throw std::runtime_error("Unsupported older checkpoint format_version " +
                                 std::to_string(format_version) + " (current: " +
                                 std::to_string(kConfigFormatVersion) + "). Retrain from scratch.");
    }

    int loaded_vocab_size = 0, loaded_d_model = 0, loaded_encoder_layers = 0,
        loaded_decoder_layers = 0;
    int loaded_num_heads = 0, loaded_d_ff = 0, loaded_max_seq_length = 0;
    int loaded_bos = 0, loaded_eos = 0, loaded_pad = 0;

    config_file.read(reinterpret_cast<char*>(&loaded_vocab_size), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_encoder_layers), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_decoder_layers), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_max_seq_length), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_bos), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_eos), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_pad), sizeof(int));
    config_file.close();

    // Verify architecture match
    if (loaded_vocab_size != vocab_size || loaded_d_model != d_model ||
        loaded_encoder_layers != encoder_layers || loaded_decoder_layers != decoder_layers ||
        loaded_num_heads != num_heads || loaded_d_ff != d_ff ||
        loaded_max_seq_length != max_seq_length) {
        throw std::runtime_error(
            "Model architecture mismatch: saved (vocab=" + std::to_string(loaded_vocab_size) +
            ", d_model=" + std::to_string(loaded_d_model) +
            ", enc_layers=" + std::to_string(loaded_encoder_layers) +
            ", dec_layers=" + std::to_string(loaded_decoder_layers) + ", num_heads=" +
            std::to_string(loaded_num_heads) + ", d_ff=" + std::to_string(loaded_d_ff) +
            ", max_seq=" + std::to_string(loaded_max_seq_length) + ") vs current (vocab=" +
            std::to_string(vocab_size) + ", d_model=" + std::to_string(d_model) + ", enc_layers=" +
            std::to_string(encoder_layers) + ", dec_layers=" + std::to_string(decoder_layers) +
            ", num_heads=" + std::to_string(num_heads) + ", d_ff=" + std::to_string(d_ff) +
            ", max_seq=" + std::to_string(max_seq_length) + ")");
    }

    // Load tokenizer vocabulary
    tokenizer->load_vocab(filepath + ".vocab");

    // Load component weights
    encoder->load_weights(filepath + ".encoder");
    decoder->load_weights(filepath + ".decoder");

    // Load language model head
    lm_head->load_weights(filepath + ".lm_head");

    // Update special tokens
    bos_token_id = loaded_bos;
    eos_token_id = loaded_eos;
    pad_token_id = loaded_pad;
}

// Clone (TD-023): deep-copy weights via save/load; no optimizer state is copied
std::unique_ptr<EncoderDecoderModel> EncoderDecoderModel::clone() const {
    namespace fs = std::filesystem;

    // Build a unique temp path from the object address and a steady-clock timestamp
    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto addr = std::hash<const void*>{}(static_cast<const void*>(this));
    const std::string tmp = fs::temp_directory_path().string() + "/adai_clone_" +
                            std::to_string(addr) + "_" + std::to_string(ts);

    // Persist weights to temp files
    save_model(tmp);

    // RAII: remove all temp files on exit (success or exception)
    auto remove_temps = [&tmp] {
        for (const char* ext : {".config", ".vocab", ".encoder", ".decoder", ".lm_head"}) {
            std::error_code ec;
            std::filesystem::remove(tmp + ext, ec);  // best-effort; ignore errors
        }
    };

    std::unique_ptr<EncoderDecoderModel> copy;
    try {
        copy = std::make_unique<EncoderDecoderModel>(
            vocab_size, d_model, encoder_layers, decoder_layers, num_heads, d_ff, max_seq_length);
        copy->load_model(tmp);  // also sets bos/eos/pad_token_id on copy
    } catch (...) {
        remove_temps();
        throw;
    }
    remove_temps();
    return copy;
}

// Forward pass
Matrix EncoderDecoderModel::forward(const std::vector<int>& input_tokens,
                                    const std::vector<int>& target_tokens) {
    // Cache inputs
    cached_input_tokens = input_tokens;
    cached_target_tokens = target_tokens;

    // Encode input
    int input_len = static_cast<int>(input_tokens.size());
    Matrix encoder_mask(input_len, input_len);
    for (int i = 0; i < input_len; ++i) {
        for (int j = 0; j < input_len; ++j) {
            encoder_mask.data[i][j] = 1.0f;
        }
    }
    cached_encoder_output = encoder->encode_with_mask(input_tokens, encoder_mask);

    // Prepare decoder input (teacher forcing: use target tokens)
    // In training, we use ground truth tokens as decoder input
    // TD-060 (fixed): `i < target_tokens.size() - 1` underflows to SIZE_MAX for
    // an empty target_tokens (size_t is unsigned), turning this into an
    // out-of-bounds read that crashes immediately — confirmed via ASan (SEGV).
    // `forward()` is a public, documented "custom training loops" API, so an
    // empty target_tokens is a plausible caller input, not just theoretical.
    std::vector<int> decoder_input;
    decoder_input.push_back(bos_token_id);
    for (size_t i = 0; i + 1 < target_tokens.size(); ++i) {
        decoder_input.push_back(target_tokens[i]);
    }

    // Decode with cross-attention
    cached_decoder_output = decoder->forward_with_encoder(decoder_input, cached_encoder_output);

    // Project to vocabulary
    Matrix logits = lm_head->forward(cached_decoder_output);

    return logits;
}

// Backward pass
void EncoderDecoderModel::backward(const Matrix& grad_output) {
    if (!requires_grad) {
        return;
    }

    // Backward through LM head
    Matrix grad_decoder = lm_head->backward(grad_output);

    // Backward through decoder, capturing the gradient w.r.t. encoder output
    // (summed across every decoder block's cross-attention)
    Matrix grad_encoder_output;
    decoder->backward(grad_decoder, grad_encoder_output);

    // Backward through encoder
    encoder->backward(grad_encoder_output);
}

#ifdef ADAI_ENABLE_GPU
void EncoderDecoderModel::gpu_init_training() {
    encoder->gpu_upload_weights();
    decoder->gpu_upload_weights();
    lm_head->gpu_upload_weights();
    gpu_initialized_ = true;
}

void EncoderDecoderModel::gpu_zero_grads() {
    encoder->gpu_zero_grads();
    decoder->gpu_zero_grads();
    lm_head->gpu_zero_grads();
}

float EncoderDecoderModel::gpu_forward(const std::vector<int>& input_tokens,
                                       const std::vector<int>& target_tokens) {
    cached_target_tokens = target_tokens;
    const int tgt_len = static_cast<int>(target_tokens.size());

    // Upload integer target IDs to device (resize buffer if needed)
    if (!gpu_targets_dev_ || gpu_target_len_ < tgt_len) {
        gpu_targets_dev_ = std::make_unique<adai::gpu::GPUMemory<int>>(tgt_len);
        gpu_target_len_ = tgt_len;
    }
    gpu_targets_dev_->copy_from_host(target_tokens.data(), static_cast<size_t>(tgt_len));

    // Encoder
    gpu_encoder_out_ = std::make_unique<adai::gpu::GPUMatrix>(encoder->gpu_encode(input_tokens));

    // Decoder (caches its block inputs/outputs for backward internally)
    adai::gpu::GPUMatrix dec_out = decoder->gpu_decode(target_tokens, *gpu_encoder_out_);

    // LM head (caches dec_out internally)
    gpu_logits_ = std::make_unique<adai::gpu::GPUMatrix>(lm_head->gpu_forward(dec_out));

    // Cross-entropy loss entirely on GPU — only a scalar crosses PCIe
    return adai::gpu::matrix_cross_entropy_loss_gpu(gpu_logits_->device_ptr(),
                                                    gpu_targets_dev_->get(), tgt_len, vocab_size);
}

void EncoderDecoderModel::gpu_backward(float scale) {
    const int tgt_len = static_cast<int>(cached_target_tokens.size());

    // Cross-entropy gradient w.r.t. logits
    adai::gpu::GPUMatrix dlogits(tgt_len, vocab_size);
    adai::gpu::matrix_cross_entropy_grad_gpu(gpu_logits_->device_ptr(), gpu_targets_dev_->get(),
                                             dlogits.device_ptr(), tgt_len, vocab_size);

    // Apply gradient accumulation scale (no allocation when scale == 1)
    if (scale != 1.0f)
        dlogits = dlogits.scale(scale);

    // LM head backward — accumulates dW, db; returns d_dec_out
    adai::gpu::GPUMatrix d_dec = lm_head->gpu_backward(dlogits);

    // Decoder backward — accumulates gradients into each block, and returns the
    // gradient w.r.t. encoder_output summed across every decoder block's
    // cross-attention.
    auto [grad_decoder_embed_input, grad_encoder_output] = decoder->gpu_backward(d_dec);
    // decoder->gpu_backward() already applies this gradient to the decoder's
    // token embedding internally (mirrors LLMEncoder::gpu_backward()); the
    // caller has no further use for it.
    (void)grad_decoder_embed_input;

    // Encoder backward
    encoder->gpu_backward(grad_encoder_output);
}

float EncoderDecoderModel::gpu_evaluate(const std::string& input_text,
                                        const std::string& target_text) {
    // encode() has no length cap, unlike the truncate() step ChatbotTrainer
    // applies when first tokenizing the dataset — prefer
    // gpu_evaluate_tokenized() with already-truncated ids whenever available.
    // An untruncated sequence here scales the encoder/decoder's on-device
    // temporaries (attention scores are O(seq^2)) well past what the model
    // ever sees during training, which is what was blowing the GPU memory
    // budget during validation.
    std::vector<int> input_tokens =
        tokenizer->encode(input_text, false);  // Encoder: no special tokens
    std::vector<int> target_tokens =
        tokenizer->encode(target_text, true);  // Decoder: with special tokens

    return gpu_evaluate_tokenized(input_tokens, target_tokens);
}

float EncoderDecoderModel::gpu_evaluate_tokenized(const std::vector<int>& input_tokens,
                                                  const std::vector<int>& target_tokens) {
    bool prev_mode = requires_grad;
    set_training(false);

    // Forward pass only — gpu_forward() already computes cross-entropy loss
    // on-device, so there is nothing further to compute here. Deliberately
    // never call gpu_backward(): validation must not accumulate gradients.
    float loss = gpu_forward(input_tokens, target_tokens);

    set_training(prev_mode);
    return loss;
}

std::string EncoderDecoderModel::gpu_generate_response(const std::string& input_text,
                                                       int max_length) {
    // Matches generate_response(): the caller's max_length is not forwarded to
    // TextGenerator, which generates up to its own, separately configured
    // config.max_length. Kept as a parameter for interface parity.
    (void)max_length;

    std::vector<int> input_tokens = tokenizer->encode(input_text, false);

    // Encode once on GPU; read-only input to every decode step below.
    adai::gpu::GPUMatrix gpu_encoder_out = encoder->gpu_encode(input_tokens);

    // No GPU KV-cache exists yet, so every step recomputes the full decoded
    // sequence from scratch via gpu_decode() (same causal self-attention +
    // cross-attention to gpu_encoder_out as training) — algorithmically the
    // same shape as the CPU "greedy workaround" path in
    // generate_response_with_strategy(), just GPU-accelerated throughout.
    auto model_fn = [this, &gpu_encoder_out](const std::vector<int>& tokens) -> Matrix {
        adai::gpu::GPUMatrix dec_out = decoder->gpu_decode(tokens, gpu_encoder_out);
        adai::gpu::GPUMatrix logits = lm_head->gpu_forward(dec_out);

        // generate() only ever reads the last row of what model_fn returns,
        // so download just that row instead of the whole [tgt, vocab_size]
        // logits matrix.
        const int tgt = static_cast<int>(tokens.size());
        std::vector<float> last_row(vocab_size);
        adai::gpu::matrix_download_gpu(logits.device_ptr() + (tgt - 1) * vocab_size,
                                       last_row.data(), vocab_size);

        Matrix last_logits(1, vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            last_logits.data[0][v] = last_row[v];
        }
        return last_logits;
    };

    std::vector<int> output_tokens = generator->generate(model_fn, {bos_token_id});
    return tokenizer->decode(output_tokens, true);
}

void EncoderDecoderModel::gpu_download_grads() {
    encoder->gpu_download_grads();
    decoder->gpu_download_grads();
    lm_head->gpu_download_grads();
}

void EncoderDecoderModel::gpu_sync_weights() {
    encoder->gpu_upload_weights();
    decoder->gpu_upload_weights();
    lm_head->gpu_upload_weights();
}

void EncoderDecoderModel::gpu_synchronize() {
    adai::gpu::GPUManager::synchronize();
}
#endif
