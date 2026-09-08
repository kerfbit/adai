// @adai-status: beta        (large, actively evolving core trainer)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-07

#include "ChatbotTrainer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include "ConversationContext.hpp"
#include "GenerationQualityMetrics.hpp"
#include "Logger.hpp"
#ifdef ADAI_ENABLE_OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

// ANSI color codes
#define COLOR_RESET "\033[0m"
#define COLOR_INFO "\033[1;36m"
#define COLOR_SUCCESS "\033[1;32m"
#define COLOR_WARNING "\033[1;33m"
#define COLOR_ERROR "\033[1;31m"
#define COLOR_PROGRESS "\033[1;35m"

ChatbotTrainer::ChatbotTrainer(const TrainingConfig& cfg)
    : config(cfg),
      tokenizer(nullptr),
      model(nullptr),
      optimizer(nullptr),
      best_validation_loss(std::numeric_limits<float>::max()),

      current_learning_rate(cfg.learning_rate) {}

ChatbotTrainer::~ChatbotTrainer() {
    join_generation_quality_thread();
}

void ChatbotTrainer::join_generation_quality_thread() {
    if (generation_quality_thread_.has_value() && generation_quality_thread_->joinable()) {
        generation_quality_thread_->join();
        generation_quality_thread_.reset();
    }
}

/**
 * @brief Initialize tokenizer from vocabulary file
 */
bool ChatbotTrainer::load_tokenizer(const std::string& vocab_path) {
    adai::Logger::info("📚 Loading tokenizer from: {}", vocab_path);

    tokenizer = std::make_unique<BPETokenizer>(config.tokenizer_mode);
    try {
        tokenizer->load_vocab(vocab_path);
        adai::Logger::info("✅ Tokenizer loaded (vocab size: {}, mode: {})",
                           tokenizer->get_vocab_size(),
                           tokenizer->is_unicode_mode() ? "unicode" : "ascii");
        return true;
    } catch (const std::exception& e) {
        adai::Logger::error("❌ Failed to load tokenizer: {}", e.what());
        return false;
    }
}

/**
 * @brief Build vocabulary from training texts
 */
bool ChatbotTrainer::build_vocabulary(const std::vector<std::string>& texts, int vocab_size,
                                      const std::string& save_path) {
    adai::Logger::info("🔨 Building vocabulary...");
    adai::Logger::info("  Texts: {}", texts.size());
    adai::Logger::info("  Target vocab size: {}", vocab_size);

    tokenizer = std::make_unique<BPETokenizer>(config.tokenizer_mode);
    try {
        tokenizer->build_vocab(texts, vocab_size, 1);
        tokenizer->save_vocab(save_path);

        adai::Logger::info("✅ Vocabulary built (size: {}, mode: {})", tokenizer->get_vocab_size(),
                           tokenizer->is_unicode_mode() ? "unicode" : "ascii");
        adai::Logger::info("✅ Saved to: {}", save_path);
        return true;
    } catch (const std::exception& e) {
        adai::Logger::error("❌ Failed to build vocabulary: {}", e.what());
        return false;
    }
}

/**
 * @brief Load conversation pairs from file
 *
 * Format: Each pair on two lines:
 * INPUT: <user message>
 * RESPONSE: <bot response>
 * (blank line between pairs)
 */
bool ChatbotTrainer::load_conversation_data(const std::string& filepath) {
    adai::Logger::info("📖 Loading conversation data from: {}", filepath);

    std::ifstream file(filepath);
    if (!file.is_open()) {
        adai::Logger::error("❌ Cannot open file: {}", filepath);
        return false;
    }

    // Detect format from first non-empty line
    std::string first_line;
    while (std::getline(file, first_line)) {
        first_line.erase(0, first_line.find_first_not_of(" \t\r\n"));
        if (!first_line.empty())
            break;
    }
    file.seekg(0);

    int pair_count = 0;

    if (!first_line.empty() && first_line.front() == '{') {
        // JSONL training format
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line.front() != '{')
                continue;
            std::string in, resp;
            SampleMeta meta;
            if (parse_jsonl_sample(line, in, resp, meta)) {
                training_data.emplace_back(std::move(in), std::move(resp), std::move(meta));
                ++pair_count;
            }
        }
    } else {
        // Legacy INPUT:/RESPONSE: format
        std::string line, current_input, current_response;
        while (std::getline(file, line)) {
            line.erase(0, line.find_first_not_of(" \t\n\r"));
            line.erase(line.find_last_not_of(" \t\n\r") + 1);

            if (line.empty()) {
                if (!current_input.empty() && !current_response.empty()) {
                    training_data.emplace_back(current_input, current_response);
                    ++pair_count;
                    current_input.clear();
                    current_response.clear();
                }
                continue;
            }

            if (line.substr(0, 6) == "INPUT:") {
                current_input = line.substr(6);
                current_input.erase(0, current_input.find_first_not_of(" \t"));
            } else if (line.substr(0, 9) == "RESPONSE:") {
                current_response = line.substr(9);
                current_response.erase(0, current_response.find_first_not_of(" \t"));
            }
        }
        if (!current_input.empty() && !current_response.empty()) {
            training_data.emplace_back(current_input, current_response);
            ++pair_count;
        }
    }

    file.close();
    adai::Logger::info("✅ Loaded {} conversation pairs", pair_count);
    return pair_count > 0;
}

/**
 * @brief Split data into training and validation sets with random shuffling
 */
void ChatbotTrainer::split_data() {
    if (config.validation_split <= 0) {
        adai::Logger::warn("⚠️  No validation split, using all data for training");
        return;
    }

    int validation_size = static_cast<int>(training_data.size()) / config.validation_split;
    if (validation_size == 0) {
        adai::Logger::warn("⚠️  Not enough data for validation split");
        return;
    }

    // Randomly shuffle data before splitting
    std::vector<int> indices(training_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Split based on shuffled indices
    for (int i = 0; i < validation_size; i++) {
        validation_data.push_back(training_data[indices[i]]);
    }

    // Keep remaining for training
    std::vector<ConversationPair> temp_training;
    for (int i = validation_size; i < indices.size(); i++) {
        temp_training.push_back(training_data[indices[i]]);
    }
    training_data = std::move(temp_training);

    adai::Logger::info("📊 Data split (randomly shuffled):");
    adai::Logger::info("  Training: {} pairs", training_data.size());
    adai::Logger::info("  Validation: {} pairs", validation_data.size());
}

/**
 * @brief Validate and auto-correct model architecture parameters
 */
void ChatbotTrainer::validate_and_correct_config() {
    bool corrected = false;

    adai::Logger::info("🔍 Validating model configuration...");

    // Validate d_model is divisible by num_heads
    if (config.d_model % config.num_heads != 0) {
        int original_d_model = config.d_model;
        // Round up to nearest multiple of num_heads
        config.d_model =
            ((config.d_model + config.num_heads - 1) / config.num_heads) * config.num_heads;
        adai::Logger::warn("⚠️  d_model ({}) not divisible by num_heads ({})", original_d_model,
                           config.num_heads);
        adai::Logger::warn("   Auto-corrected to: {}", config.d_model);
        corrected = true;
    }

    // Validate d_ff follows recommended ratio (typically 4x d_model)
    int recommended_d_ff = 4 * config.d_model;
    if (config.d_ff != recommended_d_ff) {
        float ratio = static_cast<float>(config.d_ff) / static_cast<float>(config.d_model);
        if (ratio < 2.0f || ratio > 8.0f) {
            int original_d_ff = config.d_ff;
            config.d_ff = recommended_d_ff;
            adai::Logger::warn("⚠️  d_ff ({}) has unusual ratio to d_model (ratio: {})",
                               original_d_ff, ratio);
            adai::Logger::warn("   Auto-corrected to recommended 4x: {}", config.d_ff);
            corrected = true;
        } else {
            adai::Logger::info("   d_ff ratio: {}x d_model (acceptable, recommended: 4x)", ratio);
        }
    }

    // Validate num_heads is a power of 2 (common practice)
    int heads = config.num_heads;
    if ((heads & (heads - 1)) != 0) {
        adai::Logger::warn("⚠️  num_heads ({}) is not a power of 2 (recommended: 2, 4, 8, 16, etc.)",
                           heads);
        adai::Logger::warn("   Keeping current value, but performance may be suboptimal");
    }

    // Validate d_model is reasonable
    if (config.d_model < 64 || config.d_model > 4096) {
        adai::Logger::warn("⚠️  d_model ({}) is outside typical range [64-4096]", config.d_model);
    }

    // Validate learning rate is reasonable
    if (config.learning_rate <= 0.0f || config.learning_rate > 1.0f) {
        adai::Logger::warn("⚠️  learning_rate ({}) is outside typical range (0, 1]",
                           config.learning_rate);
    }

    // Validate min_learning_rate < learning_rate
    if (config.min_learning_rate >= config.learning_rate) {
        int original_min_lr = static_cast<int>(config.min_learning_rate);
        config.min_learning_rate = config.learning_rate * 0.01f;  // 1% of base LR
        adai::Logger::warn("⚠️  min_learning_rate ({}) >= learning_rate ({})", original_min_lr,
                           config.learning_rate);
        adai::Logger::warn("   Auto-corrected to: {}", config.min_learning_rate);
        corrected = true;
    }

    // Validate layer counts
    if (config.num_encoder_layers < 1 || config.num_encoder_layers > 48) {
        adai::Logger::warn("⚠️  num_encoder_layers ({}) is outside typical range [1-48]",
                           config.num_encoder_layers);
    }

    if (config.num_decoder_layers < 1 || config.num_decoder_layers > 48) {
        adai::Logger::warn("⚠️  num_decoder_layers ({}) is outside typical range [1-48]",
                           config.num_decoder_layers);
    }

    // Validate max sequence length
    if (config.max_seq_length < 16 || config.max_seq_length > 8192) {
        adai::Logger::warn("⚠️  max_seq_length ({}) is outside typical range [16-8192]",
                           config.max_seq_length);
    }

    if (corrected) {
        adai::Logger::info("✅ Configuration validated and corrected");
    } else {
        adai::Logger::info("✅ Configuration validated");
    }
}

// ============================================================================
// Tokenized-data cache — binary format, versioned via a leading magic number:
//   [uint32 magic] [uint64 n_train] [uint64 n_val]
//   then n_train + n_val repetitions of one TokenizedPair:
//     [uint32 input_tokens.size()] [int32 * that many]
//     [uint32 target_tokens.size()] [int32 * that many]
//     [uint32 input_text.size()]  [bytes]
//     [uint32 target_text.size()] [bytes]
// Cache staleness (wrong dataset/vocab/config) is handled by the caller via
// tokenized_cache_key (see IncrementalTrainer, which computes it from file +
// vocab checksums + tokenizer_mode + max_seq_length) — this format itself
// only guards against a truncated/corrupt file and a sample-count mismatch.
// ============================================================================
namespace {
constexpr std::uint32_t kTokenizedCacheMagic = 0x544B4331;  // 'TKC1'

bool read_u32(std::ifstream& in, std::uint32_t& out) {
    in.read(reinterpret_cast<char*>(&out), sizeof(out));
    return static_cast<bool>(in);
}

bool read_u64(std::ifstream& in, std::uint64_t& out) {
    in.read(reinterpret_cast<char*>(&out), sizeof(out));
    return static_cast<bool>(in);
}

bool read_tokenized_pair(std::ifstream& in, TokenizedPair& out) {
    std::uint32_t n = 0;
    if (!read_u32(in, n))
        return false;
    out.input_tokens.resize(n);
    if (n > 0 && !in.read(reinterpret_cast<char*>(out.input_tokens.data()),
                          static_cast<std::streamsize>(n * sizeof(int))))
        return false;

    if (!read_u32(in, n))
        return false;
    out.target_tokens.resize(n);
    if (n > 0 && !in.read(reinterpret_cast<char*>(out.target_tokens.data()),
                          static_cast<std::streamsize>(n * sizeof(int))))
        return false;

    if (!read_u32(in, n))
        return false;
    out.input_text.resize(n);
    if (n > 0 && !in.read(out.input_text.data(), static_cast<std::streamsize>(n)))
        return false;

    if (!read_u32(in, n))
        return false;
    out.target_text.resize(n);
    if (n > 0 && !in.read(out.target_text.data(), static_cast<std::streamsize>(n)))
        return false;

    return true;
}

void write_u32(std::ofstream& out, std::uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_tokenized_pair(std::ofstream& out, const TokenizedPair& p) {
    write_u32(out, static_cast<std::uint32_t>(p.input_tokens.size()));
    if (!p.input_tokens.empty())
        out.write(reinterpret_cast<const char*>(p.input_tokens.data()),
                  static_cast<std::streamsize>(p.input_tokens.size() * sizeof(int)));
    write_u32(out, static_cast<std::uint32_t>(p.target_tokens.size()));
    if (!p.target_tokens.empty())
        out.write(reinterpret_cast<const char*>(p.target_tokens.data()),
                  static_cast<std::streamsize>(p.target_tokens.size() * sizeof(int)));
    write_u32(out, static_cast<std::uint32_t>(p.input_text.size()));
    if (!p.input_text.empty())
        out.write(p.input_text.data(), static_cast<std::streamsize>(p.input_text.size()));
    write_u32(out, static_cast<std::uint32_t>(p.target_text.size()));
    if (!p.target_text.empty())
        out.write(p.target_text.data(), static_cast<std::streamsize>(p.target_text.size()));
}
}  // namespace

bool ChatbotTrainer::load_tokenized_cache(const std::string& cache_path) {
    std::ifstream in(cache_path, std::ios::binary);
    if (!in.is_open())
        return false;

    std::uint32_t magic = 0;
    if (!read_u32(in, magic) || magic != kTokenizedCacheMagic)
        return false;

    std::uint64_t n_train = 0, n_val = 0;
    if (!read_u64(in, n_train) || !read_u64(in, n_val))
        return false;

    // Guard against a stale/corrupt cache whose sample counts no longer match
    // the raw data actually loaded this run — the tokenized_cache_key is the
    // primary defense (see IncrementalTrainer), this is a cheap second check.
    if (n_train != training_data.size() || n_val != validation_data.size())
        return false;

    std::vector<TokenizedPair> loaded_train(n_train);
    for (std::uint64_t i = 0; i < n_train; ++i) {
        if (!read_tokenized_pair(in, loaded_train[i]))
            return false;
    }
    std::vector<TokenizedPair> loaded_val(n_val);
    for (std::uint64_t i = 0; i < n_val; ++i) {
        if (!read_tokenized_pair(in, loaded_val[i]))
            return false;
    }

    tokenized_training_data = std::move(loaded_train);
    tokenized_validation_data = std::move(loaded_val);
    return true;
}

void ChatbotTrainer::save_tokenized_cache(const std::string& cache_path) const {
    std::error_code ec;
    fs::create_directories(fs::path(cache_path).parent_path(), ec);
    if (ec) {
        adai::Logger::warn("Tokenized-data cache: failed to create directory for '{}' ({})",
                           cache_path, ec.message());
        return;
    }

    std::ofstream out(cache_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        adai::Logger::warn("Tokenized-data cache: failed to open '{}' for writing", cache_path);
        return;
    }

    write_u32(out, kTokenizedCacheMagic);
    const std::uint64_t n_train = tokenized_training_data.size();
    const std::uint64_t n_val = tokenized_validation_data.size();
    out.write(reinterpret_cast<const char*>(&n_train), sizeof(n_train));
    out.write(reinterpret_cast<const char*>(&n_val), sizeof(n_val));
    for (const auto& p : tokenized_training_data)
        write_tokenized_pair(out, p);
    for (const auto& p : tokenized_validation_data)
        write_tokenized_pair(out, p);

    if (!out.good()) {
        adai::Logger::warn("Tokenized-data cache: write error on '{}'", cache_path);
    }
}

/**
 * @brief Preprocess and tokenize all training and validation data
 */
void ChatbotTrainer::preprocess_data() {
    // Get tokenizer from model (ownership was transferred during initialization)
    BPETokenizer* tokenizer = model ? model->get_tokenizer() : nullptr;

    if (!tokenizer) {
        adai::Logger::error("❌ Tokenizer not initialized!");
        return;
    }

    adai::Logger::info("🔄 Preprocessing and tokenizing data...");

    const std::string cache_path =
        (config.cache_tokenized_data && !config.tokenized_cache_key.empty())
            ? config.tokenized_cache_dir + "/" + config.tokenized_cache_key + ".cache"
            : std::string();

    if (!cache_path.empty() && load_tokenized_cache(cache_path)) {
        // load_tokenized_cache() only populates tokenized_training_data/
        // tokenized_validation_data — restore the two things the encode loops
        // below normally set as a side effect: per-pair token_count (used by
        // outlier detection/quality backfill downstream) and the shuffle index.
        for (std::size_t i = 0; i < training_data.size(); ++i) {
            training_data[i].meta.token_count =
                static_cast<int>(tokenized_training_data[i].input_tokens.size() +
                                 tokenized_training_data[i].target_tokens.size());
        }
        for (std::size_t i = 0; i < validation_data.size(); ++i) {
            validation_data[i].meta.token_count =
                static_cast<int>(tokenized_validation_data[i].input_tokens.size() +
                                 tokenized_validation_data[i].target_tokens.size());
        }
        training_indices.resize(tokenized_training_data.size());
        std::iota(training_indices.begin(), training_indices.end(), 0);

        adai::Logger::info("✅ Tokenized-data cache hit — skipped preprocessing:");
        adai::Logger::info("  Training samples: {}", tokenized_training_data.size());
        adai::Logger::info("  Validation samples: {}", tokenized_validation_data.size());
        return;
    }
    if (!cache_path.empty()) {
        adai::Logger::info("Tokenized-data cache miss ('{}') — preprocessing from scratch",
                           cache_path);
    }

    const int max_len = static_cast<int>(config.max_seq_length);
    // Pre-truncate raw text to ~max_len*5 chars before BPE encoding to keep O(max_len) cost.
    // BPE encoding is O(n * merges), so capping input text length avoids hour-long tokenization
    // on long documents (e.g. minipile entries of 7000+ regex tokens).
    const size_t max_chars = static_cast<size_t>(max_len) * 5;
    // Truncate at a valid UTF-8 character boundary to avoid splitting a
    // multi-byte sequence, which would produce an invalid UTF-8 string.
    auto clip_text = [max_chars](const std::string& s) -> std::string {
        if (s.size() <= max_chars) {
            return s;
        }
        size_t i = max_chars;
        // Back up to the start of the current multi-byte character
        while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) {
            --i;
        }
        return s.substr(0, i);
    };
    auto truncate = [max_len](std::vector<int> ids) -> std::vector<int> {
        if (static_cast<int>(ids.size()) > max_len) {
            ids.resize(max_len);
        }
        return ids;
    };
    // Input-side truncation uses truncate_text_tail()/truncate_tokens_tail() instead
    // of clip_text()/truncate() — see their doc comments in ChatbotTrainer.hpp for why.

    // Tokenize training data — parallel BPE encoding (tokenizer is read-only after load)
    const int n_train = static_cast<int>(training_data.size());
    tokenized_training_data.clear();
    tokenized_training_data.resize(n_train);
    int skipped_train = 0;
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : skipped_train)
#endif
    for (int i = 0; i < n_train; i++) {
        const auto& pair = training_data[i];
        try {
            tokenized_training_data[i] = TokenizedPair(
                truncate_tokens_tail(
                    tokenizer->encode(truncate_text_tail(pair.input, max_chars), false), max_len),
                truncate(tokenizer->encode(clip_text(pair.response), true)), pair.input,
                pair.response);
            training_data[i].meta.token_count =
                static_cast<int>(tokenized_training_data[i].input_tokens.size() +
                                 tokenized_training_data[i].target_tokens.size());
        } catch (const TokenizerEncodingError&) {
            // Leave default-constructed (empty) — filtered out during training
            ++skipped_train;
        }
    }
    if (skipped_train > 0) {
        adai::Logger::warn("Skipped {} training pairs with invalid UTF-8", skipped_train);
    }

    // Tokenize validation data — parallel BPE encoding
    const int n_val = static_cast<int>(validation_data.size());
    tokenized_validation_data.clear();
    tokenized_validation_data.resize(n_val);
    int skipped_val = 0;
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : skipped_val)
#endif
    for (int i = 0; i < n_val; i++) {
        const auto& pair = validation_data[i];
        try {
            tokenized_validation_data[i] = TokenizedPair(
                truncate_tokens_tail(
                    tokenizer->encode(truncate_text_tail(pair.input, max_chars), false), max_len),
                truncate(tokenizer->encode(clip_text(pair.response), true)), pair.input,
                pair.response);
            validation_data[i].meta.token_count =
                static_cast<int>(tokenized_validation_data[i].input_tokens.size() +
                                 tokenized_validation_data[i].target_tokens.size());
        } catch (const TokenizerEncodingError&) {
            ++skipped_val;
        }
    }
    if (skipped_val > 0) {
        adai::Logger::warn("Skipped {} validation pairs with invalid UTF-8", skipped_val);
    }

    // Initialize shuffling indices
    training_indices.resize(tokenized_training_data.size());
    std::iota(training_indices.begin(), training_indices.end(), 0);

    adai::Logger::info("✅ Data preprocessed:");
    adai::Logger::info("  Training samples: {}", tokenized_training_data.size());
    adai::Logger::info("  Validation samples: {}", tokenized_validation_data.size());

    if (!cache_path.empty()) {
        save_tokenized_cache(cache_path);
    }
}

/**
 * @brief Shuffle training data indices for epoch
 */
void ChatbotTrainer::shuffle_training_data() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(training_indices.begin(), training_indices.end(), g);
}

/**
 * @brief Log message based on log level
 */
void ChatbotTrainer::log(LogLevel level, const std::string& message, const std::string& /*color*/) {
    // color parameter accepted for API compatibility but ignored;
    // Logger handles its own coloring via spdlog level-colored sinks.
    if (static_cast<int>(config.log_level) >= static_cast<int>(level)) {
        adai::Logger::info("{}", message);
    }
}

/**
 * @brief Calculate perplexity from loss
 * Perplexity = exp(loss), measures how well the model predicts the next token
 * Lower is better, with 1.0 being perfect prediction
 */
float ChatbotTrainer::calculate_perplexity(float loss) {
    return std::exp(loss);
}

/**
 * @brief Calculate token-level accuracy (stub - requires model output probabilities)
 * This is a placeholder until model exposes prediction probabilities
 * Returns -1.0 to indicate not implemented
 */
float ChatbotTrainer::calculate_accuracy(const std::vector<int>& predictions,
                                         const std::vector<int>& targets) {
    if (predictions.empty() || targets.empty() || predictions.size() != targets.size()) {
        return -1.0f;  // Not implemented yet
    }

    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == targets[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(predictions.size());
}

std::string ChatbotTrainer::truncate_text_tail(const std::string& s, size_t max_chars) {
    if (s.size() <= max_chars) {
        return s;
    }
    size_t start = s.size() - max_chars;
    // Advance to the start of the next full multi-byte character so we don't
    // begin mid-sequence.
    while (start < s.size() && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) {
        ++start;
    }
    return s.substr(start);
}

std::vector<int> ChatbotTrainer::truncate_tokens_tail(std::vector<int> ids, int max_len) {
    if (static_cast<int>(ids.size()) > max_len) {
        ids.erase(ids.begin(), ids.end() - max_len);
    }
    return ids;
}

/**
 * @brief Single entry point for EncoderDecoderModel construction.
 * Reads vocab size from the current tokenizer and all architecture
 * dimensions from config.
 */
void ChatbotTrainer::build_model() {
    model = std::make_unique<EncoderDecoderModel>(
        tokenizer->get_vocab_size(), config.d_model, config.num_encoder_layers,
        config.num_decoder_layers, config.num_heads, config.d_ff, config.max_seq_length);
}

/**
 * @brief Initialize the encoder-decoder model
 */
void ChatbotTrainer::initialize_model() {
    // Validate and correct configuration first
    validate_and_correct_config();

    adai::Logger::info("🧠 Initializing transformer model...");
    adai::Logger::info("  d_model: {}", config.d_model);
    adai::Logger::info("  num_heads: {}", config.num_heads);
    adai::Logger::info("  d_ff: {}", config.d_ff);
    adai::Logger::info("  encoder_layers: {}", config.num_encoder_layers);
    adai::Logger::info("  decoder_layers: {}", config.num_decoder_layers);
    adai::Logger::info("  max_seq_length: {}", config.max_seq_length);
    adai::Logger::info("  learning_rate: {}", config.learning_rate);

    build_model();

    // Transfer tokenizer ownership to the model
    // The model will now own the tokenizer and handle saving/loading
    model->set_tokenizer(tokenizer.release());

    adai::Logger::info("✅ Model initialized");

    // Initialize optimizer
    adai::Logger::info("🎯 Initializing optimizer...");
    {
        std::string opt_type_str;
        switch (config.optimizer_type) {
            case OptimizerType::SGD:
                opt_type_str = "SGD";
                break;
            case OptimizerType::SGD_MOMENTUM:
                opt_type_str = "SGD+Momentum";
                break;
            case OptimizerType::ADAM:
                opt_type_str = "Adam";
                break;
            case OptimizerType::ADAMW:
                opt_type_str = "AdamW";
                break;
            default:
                opt_type_str = "Unknown";
                break;
        }
        adai::Logger::info("  Type: {}", opt_type_str);
    }
    adai::Logger::info("  Learning rate: {}", config.learning_rate);
    adai::Logger::info("  Weight decay: {}", config.weight_decay);
    adai::Logger::info("  Gradient clip norm: {}", config.gradient_clip_norm);
    if (config.adaptive_gradient_clip) {
        adai::Logger::info(
            "  Adaptive clipping: ON  min={} max={} ema_decay={} headroom={} warmup={} spike_k={}",
            config.gradient_clip_min, config.gradient_clip_max, config.gradient_clip_ema_decay,
            config.gradient_clip_headroom, config.gradient_clip_warmup_steps,
            config.gradient_clip_spike_k);
    }
    if (config.optimizer_type == OptimizerType::ADAM ||
        config.optimizer_type == OptimizerType::ADAMW) {
        adai::Logger::info("  Adam beta1: {}", config.adam_beta1);
        adai::Logger::info("  Adam beta2: {}", config.adam_beta2);
    }

    optimizer = std::make_unique<Optimizer>(config.optimizer_type, config.learning_rate);
    optimizer->set_weight_decay(config.weight_decay);
    optimizer->set_max_grad_norm(config.gradient_clip_norm);

    if (config.optimizer_type == OptimizerType::ADAM ||
        config.optimizer_type == OptimizerType::ADAMW) {
        optimizer->set_betas(config.adam_beta1, config.adam_beta2);
    }

    // Register model parameters with optimizer
    model->register_parameters(*optimizer);

    adai::Logger::info("✅ Optimizer initialized");

#ifdef ADAI_ENABLE_GPU
    adai::Logger::info("🖥️  Uploading model weights to GPU...");
    model->gpu_init_training();
    adai::Logger::info("✅ GPU training initialized");
#endif
}

/**
 * @brief Calculate learning rate for current step
 */
float ChatbotTrainer::calculate_learning_rate(int step) {
    float lr = config.learning_rate;
    int warmup = config.warmup_steps;

    // Auto-configure warmup if not set
    if (warmup == 0 && config.lr_schedule != LRSchedule::CONSTANT) {
        warmup = total_training_steps / 10;  // 10% warmup
    }

    switch (config.lr_schedule) {
        case LRSchedule::CONSTANT:
            return lr;

        case LRSchedule::LINEAR_WARMUP:
            if (step < warmup) {
                return lr * (static_cast<float>(step) / static_cast<float>(warmup));
            }
            return lr;

        case LRSchedule::COSINE_DECAY: {
            float progress = static_cast<float>(step) / static_cast<float>(total_training_steps);
            float cosine = 0.5f * (1.0f + std::cos(3.14159265359f * progress));
            return config.min_learning_rate + (lr - config.min_learning_rate) * cosine;
        }

        case LRSchedule::WARMUP_COSINE: {
            // Warmup phase
            if (step < warmup) {
                return lr * (static_cast<float>(step) / static_cast<float>(warmup));
            }
            // Cosine decay phase
            float progress = static_cast<float>(step - warmup) /
                             static_cast<float>(total_training_steps - warmup);
            float cosine = 0.5f * (1.0f + std::cos(3.14159265359f * progress));
            return config.min_learning_rate + (lr - config.min_learning_rate) * cosine;
        }

        case LRSchedule::STEP_DECAY: {
            int decay_steps = config.lr_decay_steps;
            if (decay_steps == 0) {
                decay_steps = total_training_steps / config.num_epochs;
            }
            int num_decays = step / decay_steps;
            return lr * static_cast<float>(std::pow(config.lr_decay_factor, num_decays));
        }

        case LRSchedule::EXPONENTIAL_DECAY: {
            int decay_steps = config.lr_decay_steps;
            if (decay_steps == 0) {
                decay_steps = total_training_steps / config.num_epochs;
            }
            float decay_rate =
                std::pow(config.lr_decay_factor, 1.0f / static_cast<float>(decay_steps));
            return lr * static_cast<float>(std::pow(decay_rate, step));
        }

        default:
            return lr;
    }
}

/**
 * @brief Update learning rate for current step
 */
void ChatbotTrainer::update_learning_rate() {
    current_learning_rate = calculate_learning_rate(global_step);
    optimizer->set_learning_rate(current_learning_rate);
    // Also update model LR for backward compatibility
    model->set_learning_rate(current_learning_rate);
}

/**
 * @brief Get learning rate schedule name
 */
std::string ChatbotTrainer::get_schedule_name() {
    switch (config.lr_schedule) {
        case LRSchedule::CONSTANT:
            return "Constant";
        case LRSchedule::LINEAR_WARMUP:
            return "Linear Warmup";
        case LRSchedule::COSINE_DECAY:
            return "Cosine Decay";
        case LRSchedule::WARMUP_COSINE:
            return "Warmup + Cosine";
        case LRSchedule::STEP_DECAY:
            return "Step Decay";
        case LRSchedule::EXPONENTIAL_DECAY:
            return "Exponential Decay";
        default:
            return "Unknown";
    }
}

/**
 * @brief Train one epoch with gradient accumulation support
 */
float ChatbotTrainer::train_epoch(int epoch) {
    float total_loss = 0.0f;
    float total_grad_norm = 0.0f;
    int num_samples = static_cast<int>(tokenized_training_data.size());
    int effective_batch_size = config.batch_size * config.gradient_accumulation_steps;

    long long epoch_tokens = 0;
    for (const auto& tp : tokenized_training_data) {
        epoch_tokens += static_cast<long long>(tp.input_tokens.size() + tp.target_tokens.size());
    }

    // Notify metrics reporter that epoch is starting (1-based epoch number)
    if (metrics_reporter_) {
        metrics_reporter_->start_epoch(epoch + 1, num_samples);
    }

    log(LogLevel::VERBOSE,
        "\n📈 Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(config.num_epochs));
    if (config.gradient_accumulation_steps > 1) {
        log(LogLevel::VERBOSE,
            "  Using gradient accumulation: " + std::to_string(config.gradient_accumulation_steps) +
                " steps (effective batch size: " + std::to_string(effective_batch_size) + ")");
    }

    // Shuffle data at the start of each epoch
    shuffle_training_data();

    // Reset accumulation state at epoch start
    accumulation_step = 0;
    accumulated_loss = 0.0f;
    int update_count = 0;  // optimizer steps taken this epoch (for running avg)

    // ── TD-013: Advanced diagnostic accumulators ──────────────────────────────
    // Welford online algorithm state for gradient norm variance
    float gn_w_count = 0.0f, gn_w_mean = 0.0f, gn_w_M2 = 0.0f;
    // Welford online algorithm state for per-step loss (outlier z-score)
    float ls_w_count = 0.0f, ls_w_mean = 0.0f, ls_w_M2 = 0.0f;
    // Compute-time vs wall-time accumulators
    double total_compute_ns = 0.0;
    // Weight-update ratio (lr*||g||/||w||) accumulator
    float wu_ratio_sum = 0.0f;
    int wu_count = 0;
    // Outlier detection thresholds come from TrainingConfig (TD-021)
    // Epoch-level wall-clock start for compute_time_ratio denominator
    auto epoch_td013_start = std::chrono::steady_clock::now();
    // ── TD-017: Adaptive gradient clipping state ──────────────────────────────
    // agc_ema is seeded with the fixed gradient_clip_norm value so the first
    // warmup steps start at a sensible scale rather than zero.
    float agc_ema = config.gradient_clip_norm;  // running EMA of raw grad norms
    int agc_step_count = 0;                     // optimizer steps taken (for warmup)
    int agc_spike_count = 0;                    // cumulative spike steps this epoch
    float agc_clip_sum = 0.0f;                  // sum of effective_clip values (for epoch avg)
    int agc_clip_count = 0;
    bool agc_active = config.adaptive_gradient_clip;
    // ─────────────────────────────────────────────────────────────────────────
    // ── TD-013: Activation saturation accumulators ────────────────────────────
    // sat_sum / sat_count gives the epoch-average fraction of near-zero
    // post-GELU units across all FeedForward layers and forward passes.
    float sat_sum = 0.0f;
    int sat_count = 0;
    // ── TD-013: Attention entropy accumulators ─────────────────────────────────
    // ent_sum / ent_count gives the epoch-average per-token Shannon entropy of
    // the softmax attention distribution across all self-attention layers.
    float ent_sum = 0.0f;
    int ent_count = 0;
    // ── Batch padding efficiency accumulators ──────────────────────────────────
    // For each gradient-accumulation window we compute the theoretical padding
    // efficiency: actual_tokens / (max_input_len + max_target_len) / window_size.
    // This quantifies how well sequence lengths are matched within each batch.
    int pad_win_actual = 0;      // sum of all token lengths in the current window
    int pad_win_max_input = 0;   // max input-seq length seen in the current window
    int pad_win_max_target = 0;  // max target-seq length seen in the current window
    int pad_win_count = 0;       // samples accumulated in the current window
    float pad_eff_sum = 0.0f;
    int pad_eff_count = 0;
    // ──────────────────────────────────────────────────────────────────────────
    if (metrics_reporter_) {
        const int enc_layers = model->get_encoder_layers();
        const int dec_layers = model->get_decoder_layers();

#ifdef ADAI_ENABLE_GPU
        // GPU path: FeedForward::gpu_forward()/MultiHeadAttention::gpu_forward() never
        // touch the CPU hooks below (they operate on GPU-resident data), so the
        // GPU-native equivalents receive an already-reduced scalar per call instead.
        // Same sat_sum/sat_count/ent_sum/ent_count bookkeeping either way — the
        // epoch-end averaging/reporting further down is completely backend-agnostic.
        auto gpu_saturation_hook = [&sat_sum, &sat_count](float saturated_fraction) {
            sat_sum += saturated_fraction;
            ++sat_count;
        };
        for (int l = 0; l < enc_layers; ++l) {
            model->get_encoder()
                ->get_encoder_block(l)
                ->get_feed_forward()
                ->set_gpu_activation_stats_hook(gpu_saturation_hook);
        }
        for (int l = 0; l < dec_layers; ++l) {
            model->get_decoder()
                ->get_decoder_block(l)
                ->get_feed_forward()
                ->set_gpu_activation_stats_hook(gpu_saturation_hook);
        }

        auto gpu_entropy_hook = [&ent_sum, &ent_count](float avg_row_entropy) {
            ent_sum += avg_row_entropy;
            ++ent_count;
        };
        for (int l = 0; l < enc_layers; ++l) {
            model->get_encoder()
                ->get_encoder_block(l)
                ->get_self_attention()
                ->set_gpu_attention_stats_hook(gpu_entropy_hook);
        }
        for (int l = 0; l < dec_layers; ++l) {
            model->get_decoder()
                ->get_decoder_block(l)
                ->get_self_attention()
                ->set_gpu_attention_stats_hook(gpu_entropy_hook);
        }
#else
        // Lambda captures sat_sum/sat_count by reference; hooks are cleared
        // before train_epoch() returns so there is no dangling reference risk.
        auto saturation_hook = [&sat_sum, &sat_count](const Matrix& activated) {
            const int total = activated.rows * activated.cols;
            if (total <= 0) {
                return;
            }
            int sat = 0;
            for (int r = 0; r < activated.rows; ++r) {
                for (int c = 0; c < activated.cols; ++c) {
                    if (std::abs(activated(r, c)) < 0.01f) {
                        ++sat;
                    }
                }
            }
            sat_sum += static_cast<float>(sat) / static_cast<float>(total);
            ++sat_count;
        };
        for (int l = 0; l < enc_layers; ++l) {
            model->get_encoder()->get_encoder_block(l)->get_feed_forward()->set_activation_hook(
                saturation_hook);
        }
        for (int l = 0; l < dec_layers; ++l) {
            model->get_decoder()->get_decoder_block(l)->get_feed_forward()->set_activation_hook(
                saturation_hook);
        }

        // Attention entropy hook: H = -sum(a * log(a + eps)) averaged over tokens.
        auto entropy_hook = [&ent_sum, &ent_count](const Matrix& attn_weights) {
            const int seq_len = attn_weights.rows;
            if (seq_len <= 0 || attn_weights.cols <= 0) {
                return;
            }
            float layer_entropy = 0.0f;
            for (int i = 0; i < seq_len; ++i) {
                float row_entropy = 0.0f;
                for (int j = 0; j < attn_weights.cols; ++j) {
                    float a = attn_weights(i, j);
                    if (a > 0.0f) {
                        row_entropy -= a * std::log(a + 1e-10f);
                    }
                }
                layer_entropy += row_entropy;
            }
            ent_sum += layer_entropy / static_cast<float>(seq_len);
            ++ent_count;
        };
        for (int l = 0; l < enc_layers; ++l) {
            model->get_encoder()->get_encoder_block(l)->get_self_attention()->set_attention_hook(
                entropy_hook);
        }
        for (int l = 0; l < dec_layers; ++l) {
            model->get_decoder()->get_decoder_block(l)->get_self_attention()->set_attention_hook(
                entropy_hook);
        }
#endif
    }
    // ─────────────────────────────────────────────────────────────────────────

    for (int i = 0; i < num_samples; i++) {
        // Cooperative abort (serve's /admin/pause) — only honored at an
        // optimizer-step boundary, never mid-accumulation-window, so a pause
        // never leaves a half-applied gradient update.
        if (accumulation_step == 0 && abort_flag_ && abort_flag_->load(std::memory_order_relaxed)) {
            was_aborted_ = true;
            break;
        }

        const auto& pair = tokenized_training_data[training_indices[i]];

        // Update learning rate based on schedule (only at optimizer step)
        if (accumulation_step == 0) {
            update_learning_rate();
        }

        try {
            // Zero gradients at the start of accumulation cycle
            if (accumulation_step == 0) {
                model->zero_grad();
#ifdef ADAI_ENABLE_GPU
                model->gpu_zero_grads();
#endif
                // Reset per-window padding accumulators at start of new window
                pad_win_actual = 0;
                pad_win_max_input = 0;
                pad_win_max_target = 0;
                pad_win_count = 0;
            }

            // Accumulate padding stats for this sample
            {
                int in_len = static_cast<int>(pair.input_tokens.size());
                int tgt_len = static_cast<int>(pair.target_tokens.size());
                pad_win_actual += in_len + tgt_len;
                pad_win_max_input = std::max(pad_win_max_input, in_len);
                pad_win_max_target = std::max(pad_win_max_target, tgt_len);
                ++pad_win_count;
            }

            // Forward + backward pass
            auto compute_t0 = std::chrono::steady_clock::now();
            const float grad_scale = 1.0f / static_cast<float>(config.gradient_accumulation_steps);
            float loss = 0.0f;

#ifdef ADAI_ENABLE_GPU
            // GPU path: forward + cross-entropy loss entirely on device
            loss = model->gpu_forward(pair.input_tokens, pair.target_tokens);

            // Backfill quality from loss
            if (config.enable_loss_quality_backfill) {
                training_data[training_indices[i]].meta.quality = std::exp(-loss);
            }

            accumulated_loss += loss;

            // GPU backward with accumulation scale baked in
            model->gpu_backward(grad_scale);

            // Drain the queue before the next sample: temporaries created above
            // reserve device memory synchronously but free it via a queue-ordered
            // deferred host_task, so without a periodic sync the host can outrun
            // the device and pile up unretired allocations until it OOMs.
            model->gpu_synchronize();
#else
            Matrix logits = model->forward(pair.input_tokens, pair.target_tokens);

            // Compute loss
            loss = model->compute_loss_for_training(logits, pair.target_tokens);

            // Backfill quality from loss (overwritten each epoch; final epoch value is kept)
            if (config.enable_loss_quality_backfill) {
                training_data[training_indices[i]].meta.quality = std::exp(-loss);
            }

            accumulated_loss += loss;  // Track unscaled for logging

            // Backward pass (accumulates gradients)
            Matrix grad_loss =
                model->compute_loss_gradient_for_training(logits, pair.target_tokens);

            // Scale gradients for accumulation
            if (config.gradient_accumulation_steps > 1) {
                for (int r = 0; r < grad_loss.rows; r++) {
                    for (int c = 0; c < grad_loss.cols; c++) {
                        grad_loss.data[r][c] *= grad_scale;
                    }
                }
            }

            model->backward_pass(grad_loss);
#endif

            // Accumulate compute time (forward + backward)
            total_compute_ns +=
                static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - compute_t0)
                                        .count());

            accumulation_step++;

            // Update weights after accumulating enough gradients
            bool should_update = (accumulation_step >= config.gradient_accumulation_steps) ||
                                 (i == num_samples - 1);  // Last sample in epoch

            if (should_update) {
                // Compute padding efficiency for this accumulation window
                if (pad_win_count > 0 && (pad_win_max_input + pad_win_max_target) > 0) {
                    int padded = (pad_win_max_input + pad_win_max_target) * pad_win_count;
                    float win_eff = static_cast<float>(pad_win_actual) / static_cast<float>(padded);
                    pad_eff_sum += win_eff;
                    ++pad_eff_count;
                }

#ifdef ADAI_ENABLE_GPU
                // Pull the GPU-resident gradient accumulators into the CPU grad
                // buffers *before* anything below reads them. gpu_backward() only
                // ever writes to device-resident buffers, so get_gradient_norm()
                // and clip_gradients() would otherwise see the all-zero CPU
                // buffers left by zero_grad() at the top of this accumulation
                // window — silently skipping clipping every step.
                model->gpu_download_grads();
#endif

                // Get gradient norm before clipping
                float grad_norm = optimizer->get_gradient_norm();

                // Save step-level metrics here so sample callback can fire even on NaN
                float step_loss_for_cb =
                    (config.gradient_accumulation_steps > 0)
                        ? accumulated_loss / static_cast<float>(config.gradient_accumulation_steps)
                        : accumulated_loss;

                // Safety check for NaN/Inf gradients
                if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
                    adai::Logger::error(
                        "  ⚠️  WARNING: NaN or Inf gradient detected at sample {}! Skipping update.",
                        (i + 1));
                    // Still fire sample callback so the dashboard shows progress
                    if (sample_callback_) {
                        float running_avg = (update_count > 0)
                                                ? total_loss / static_cast<float>(update_count)
                                                : 0.0f;
                        sample_callback_(i + 1, num_samples, running_avg, step_loss_for_cb, 0.0f,
                                         current_learning_rate);
                    }
                    // Reset accumulation and skip optimizer step
                    accumulation_step = 0;
                    accumulated_loss = 0.0f;
                    model->zero_grad();
#ifdef ADAI_ENABLE_GPU
                    model->gpu_zero_grads();
#endif
                    continue;
                }

                total_grad_norm += grad_norm;

                // ── TD-013: per-step Welford updates, outlier detection, weight-update ratio ──
                {
                    // Welford update: gradient norm variance
                    gn_w_count++;
                    float gn_delta = grad_norm - gn_w_mean;
                    gn_w_mean += gn_delta / gn_w_count;
                    gn_w_M2 += gn_delta * (grad_norm - gn_w_mean);

                    // Welford update: per-step loss (for outlier z-score)
                    ls_w_count++;
                    float ls_delta = step_loss_for_cb - ls_w_mean;
                    ls_w_mean += ls_delta / ls_w_count;
                    ls_w_M2 += ls_delta * (step_loss_for_cb - ls_w_mean);

                    // Weight-update ratio approximation: (lr * ||g||) / ||w||)
                    // Throttled to every 10 steps: weight norm changes slowly and
                    // get_weight_norm() is an O(n_params) traversal.
                    if (optimizer && (update_count % 10 == 0)) {
                        float w_norm = optimizer->get_weight_norm();
                        if (w_norm > 1e-10f) {
                            wu_ratio_sum += current_learning_rate * grad_norm / w_norm;
                            ++wu_count;
                        }
                    }

                    // Outlier detection — flag samples with extreme grad_norm or loss z-score
                    if (metrics_reporter_) {
                        bool grad_outlier = grad_norm > config.grad_norm_outlier_threshold;
                        bool loss_outlier = false;
                        if (ls_w_count >= 10.0f) {
                            float ls_std =
                                (ls_w_M2 > 0.0f) ? std::sqrt(ls_w_M2 / ls_w_count) : 0.0f;
                            float ls_z =
                                (ls_std > 1e-7f) ? (step_loss_for_cb - ls_w_mean) / ls_std : 0.0f;
                            loss_outlier = (ls_z > config.loss_outlier_z_threshold);
                        }
                        if (grad_outlier || loss_outlier) {
                            AbnormalSample ab;
                            ab.epoch = epoch + 1;
                            ab.sample_id = i + 1;
                            ab.loss = step_loss_for_cb;
                            ab.grad_norm = grad_norm;
                            ab.input_text = tokenized_training_data[training_indices[i]].input_text;
                            ab.target_text =
                                tokenized_training_data[training_indices[i]].target_text;
                            ab.timestamp = std::chrono::system_clock::now();
                            if (grad_outlier && loss_outlier) {
                                ab.reason = "grad_norm_and_loss_outlier";
                            } else if (grad_outlier) {
                                ab.reason = "grad_norm_outlier";
                            } else {
                                ab.reason = "loss_outlier";
                            }
                            if (abnormal_sample_count_ < config.max_abnormal_samples) {
                                metrics_reporter_->flag_abnormal_sample(ab);
                                ++abnormal_sample_count_;
                            }
                        }
                    }
                }
                // ─────────────────────────────────────────────────────────────────────

                // ── TD-013: emit running advanced metrics every optimizer step ─────────
                if (metrics_reporter_) {
                    float running_gv = (gn_w_count >= 2.0f) ? (gn_w_M2 / gn_w_count) : 0.0f;
                    auto elapsed_wall_ns = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - epoch_td013_start)
                            .count());
                    float running_ctr = (elapsed_wall_ns > 0.0)
                                            ? static_cast<float>(total_compute_ns / elapsed_wall_ns)
                                            : 0.0f;
                    float running_wur =
                        (wu_count > 0) ? (wu_ratio_sum / static_cast<float>(wu_count)) : 0.0f;
                    metrics_reporter_->update_advanced_epoch_metrics(running_gv, running_ctr,
                                                                     running_wur);
                }
                // ─────────────────────────────────────────────────────────────────────

                // ── TD-017: Adaptive gradient clipping ────────────────────────────────
                float effective_clip = config.gradient_clip_norm;  // legacy fallback

                if (agc_active) {
                    ++agc_step_count;
                    const bool in_warmup = (agc_step_count <= config.gradient_clip_warmup_steps);

                    if (in_warmup) {
                        // Warmup: clip at ceiling to protect; still update EMA
                        agc_ema = config.gradient_clip_ema_decay * grad_norm +
                                  (1.0f - config.gradient_clip_ema_decay) * agc_ema;
                        effective_clip = config.gradient_clip_max;
                    } else {
                        const bool is_spike = (agc_ema > 0.0f) &&
                                              (grad_norm > config.gradient_clip_spike_k * agc_ema);
                        if (!is_spike) {
                            // Normal step — update EMA
                            agc_ema = config.gradient_clip_ema_decay * grad_norm +
                                      (1.0f - config.gradient_clip_ema_decay) * agc_ema;
                        } else {
                            ++agc_spike_count;
                        }
                        float candidate = agc_ema * config.gradient_clip_headroom;
                        effective_clip = std::clamp(candidate, config.gradient_clip_min,
                                                    config.gradient_clip_max);
                    }
                    agc_clip_sum += effective_clip;
                    ++agc_clip_count;
                }

                if (effective_clip > 0.0f) {
                    optimizer->clip_gradients(effective_clip, grad_norm);
                }

                // Push adaptive clip state to metrics service
                if (metrics_reporter_ && agc_active) {
                    metrics_reporter_->update_adaptive_clip_metrics(effective_clip,
                                                                    agc_spike_count);
                }
                // ─────────────────────────────────────────────────────────────────

                // Update weights via optimizer
                optimizer->step();
#ifdef ADAI_ENABLE_GPU
                // Re-upload updated CPU weights to GPU mirrors
                model->gpu_sync_weights();
#endif

                // step_loss_for_cb (not accumulated_loss) — accumulated_loss is the raw SUM
                // of this window's per-sample losses; total_loss is later divided by
                // num_updates (a count of windows, not samples), so summing the unscaled
                // window total here inflated the epoch-end average by roughly the
                // accumulation window size (e.g. ~32x with GRADIENT_ACCUMULATION_STEPS=32),
                // which is exactly what produced the "loss" ≈ 200 / perplexity=inf spike
                // logged once per epoch via end_epoch(). step_loss_for_cb is already
                // properly scaled by config.gradient_accumulation_steps.
                total_loss += step_loss_for_cb;
                global_step++;

                // Log progress
                if ((i + 1) % (config.log_every * config.gradient_accumulation_steps) == 0 ||
                    i == num_samples - 1) {
                    int num_updates =
                        global_step - (epoch * (num_samples / config.gradient_accumulation_steps));
                    float avg_loss = total_loss / static_cast<float>(num_updates);
                    float avg_grad_norm = total_grad_norm / static_cast<float>(num_updates);

                    log(LogLevel::VERBOSE,
                        "  Sample " + std::to_string(i + 1) + "/" + std::to_string(num_samples) +
                            " (Update " + std::to_string(num_updates) + ")" +
                            " - Loss: " + std::to_string(accumulated_loss) +
                            " - Avg: " + std::to_string(avg_loss) +
                            " - LR: " + std::to_string(current_learning_rate) +
                            " - GradNorm: " + std::to_string(avg_grad_norm),
                        COLOR_INFO);
                }

                // Reset accumulation state
                accumulation_step = 0;
                accumulated_loss = 0.0f;
                ++update_count;

                // Per-sample callback: fire after every optimizer step
                if (sample_callback_) {
                    float running_avg =
                        (update_count > 0) ? total_loss / static_cast<float>(update_count) : 0.0f;
                    sample_callback_(i + 1, num_samples, running_avg, step_loss_for_cb, grad_norm,
                                     current_learning_rate);
                }

                // Update metrics service with sample-level metrics
                if (metrics_reporter_) {
                    metrics_reporter_->update_sample_metrics(i + 1, step_loss_for_cb, grad_norm,
                                                             current_learning_rate);
                }
            }
        } catch (const std::exception& e) {
            adai::Logger::error("  ❌ Error training sample {}: {}", (i + 1), e.what());
#ifdef ADAI_ENABLE_GPU
            // The success path's gpu_synchronize() (above) is what drains the SYCL
            // queue's deferred host_task frees back into GPUManager's allocated_bytes_
            // counter. Skipping it here means a failed sample's — and any still-pending
            // prior samples' — frees never retire, so the counter stays inflated and the
            // next sample is more likely to also throw "budget exceeded", which again
            // skips this drain. That spiral is what permanently exhausts the budget.
            try {
                model->gpu_synchronize();
            } catch (const std::exception& sync_err) {
                adai::Logger::error("  ⚠️  gpu_synchronize() failed after sample {}: {}", (i + 1),
                                    sync_err.what());
            }
#endif
            // Still fire sample callback so the dashboard shows the sample was attempted
            if (sample_callback_) {
                float running_avg =
                    (update_count > 0) ? total_loss / static_cast<float>(update_count) : 0.0f;
                sample_callback_(i + 1, num_samples, running_avg, 0.0f, 0.0f,
                                 current_learning_rate);
            }
            // Reset accumulation on error
            accumulation_step = 0;
            accumulated_loss = 0.0f;
        }
    }

    // CRITICAL FIX: Divide by number of actual updates, not num_samples
    // total_loss only accumulates when should_update is true
    int num_updates = global_step - (epoch * (num_samples / config.gradient_accumulation_steps));
    float epoch_loss = (num_updates > 0) ? (total_loss / static_cast<float>(num_updates)) : 0.0f;
    float avg_grad_norm =
        (num_updates > 0) ? (total_grad_norm / static_cast<float>(num_updates)) : 0.0f;
    float epoch_perplexity = calculate_perplexity(epoch_loss);

    training_losses.push_back(epoch_loss);
    training_perplexities.push_back(epoch_perplexity);
    learning_rates.push_back(current_learning_rate);
    gradient_norms.push_back(avg_grad_norm);

    log(LogLevel::NORMAL,
        "✅ Epoch " + std::to_string(epoch + 1) + " complete - Loss: " +
            std::to_string(epoch_loss) + " - Perplexity: " + std::to_string(epoch_perplexity) +
            " - LR: " + std::to_string(current_learning_rate) + " - GradNorm: " +
            std::to_string(avg_grad_norm) + " - Updates: " + std::to_string(num_updates) +
            " - Tokens: " + std::to_string(epoch_tokens),
        COLOR_SUCCESS);

    // ── TD-013: emit advanced epoch diagnostics to metrics reporter ────────────
    if (metrics_reporter_) {
        float gradient_variance = (gn_w_count >= 2.0f) ? (gn_w_M2 / gn_w_count) : 0.0f;
        auto total_wall_ns =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - epoch_td013_start)
                                    .count());
        float compute_time_ratio =
            (total_wall_ns > 0.0) ? static_cast<float>(total_compute_ns / total_wall_ns) : 0.0f;
        float avg_wu_ratio = (wu_count > 0) ? (wu_ratio_sum / static_cast<float>(wu_count)) : 0.0f;
        metrics_reporter_->update_advanced_epoch_metrics(gradient_variance, compute_time_ratio,
                                                         avg_wu_ratio);

        // ── TD-013: Activation saturation – report epoch average and clear hooks ─
        float avg_saturation = (sat_count > 0) ? (sat_sum / static_cast<float>(sat_count)) : -1.0f;
        metrics_reporter_->update_activation_saturation(avg_saturation);
        const int enc_layers = model->get_encoder_layers();
        const int dec_layers = model->get_decoder_layers();
#ifdef ADAI_ENABLE_GPU
        for (int l = 0; l < enc_layers; ++l) {
            model->get_encoder()
                ->get_encoder_block(l)
                ->get_feed_forward()
                ->clear_gpu_activation_stats_hook();
        }
        for (int l = 0; l < dec_layers; ++l) {
            model->get_decoder()
                ->get_decoder_block(l)
                ->get_feed_forward()
                ->clear_gpu_activation_stats_hook();
        }
#else
        for (int l = 0; l < enc_layers; ++l) {
            model->get_encoder()->get_encoder_block(l)->get_feed_forward()->clear_activation_hook();
        }
        for (int l = 0; l < dec_layers; ++l) {
            model->get_decoder()->get_decoder_block(l)->get_feed_forward()->clear_activation_hook();
        }
#endif

        // ── TD-013: Attention entropy – report epoch average and clear hooks ──────
        float avg_entropy = (ent_count > 0) ? (ent_sum / static_cast<float>(ent_count)) : -1.0f;
        metrics_reporter_->update_attention_entropy(avg_entropy);
#ifdef ADAI_ENABLE_GPU
        for (int l = 0; l < enc_layers; ++l) {
            model->get_encoder()
                ->get_encoder_block(l)
                ->get_self_attention()
                ->clear_gpu_attention_stats_hook();
        }
        for (int l = 0; l < dec_layers; ++l) {
            model->get_decoder()
                ->get_decoder_block(l)
                ->get_self_attention()
                ->clear_gpu_attention_stats_hook();
        }
#else
        for (int l = 0; l < enc_layers; ++l) {
            model->get_encoder()
                ->get_encoder_block(l)
                ->get_self_attention()
                ->clear_attention_hook();
        }
        for (int l = 0; l < dec_layers; ++l) {
            model->get_decoder()
                ->get_decoder_block(l)
                ->get_self_attention()
                ->clear_attention_hook();
        }
#endif
        // ── Per-layer gradient norms – vanishing-gradient diagnostic ──────────────
        // Snapshot of each block's gradient norm as of the epoch's last optimizer
        // step (gradients aren't zeroed until the next accumulation window
        // starts). Lets the dashboard show whether gradients are shrinking
        // uniformly across layers (healthy) or specifically in early layers
        // (the classic vanishing-gradient signature), instead of only the
        // whole-model aggregate norm.
        {
            std::vector<float> encoder_layer_norms;
            std::vector<float> decoder_layer_norms;
            encoder_layer_norms.reserve(enc_layers);
            decoder_layer_norms.reserve(dec_layers);
            for (int l = 0; l < enc_layers; ++l) {
                encoder_layer_norms.push_back(model->get_encoder()->get_encoder_block(l)->get_gradient_norm());
            }
            for (int l = 0; l < dec_layers; ++l) {
                decoder_layer_norms.push_back(model->get_decoder()->get_decoder_block(l)->get_gradient_norm());
            }
            metrics_reporter_->update_layer_gradient_norms(encoder_layer_norms, decoder_layer_norms);
        }
        // ─────────────────────────────────────────────────────────────────────────

        // ── Batch padding efficiency – report epoch average ───────────────────────
        float avg_pad_eff =
            (pad_eff_count > 0) ? (pad_eff_sum / static_cast<float>(pad_eff_count)) : -1.0f;
        metrics_reporter_->update_padding_efficiency(avg_pad_eff);
        adai::Logger::info("Batch padding efficiency (epoch {}): {:.4f}", epoch + 1, avg_pad_eff);
        // ── TD-017: Adaptive clip – report epoch average and spike count ────────
        if (agc_active) {
            float avg_clip = (agc_clip_count > 0)
                                 ? (agc_clip_sum / static_cast<float>(agc_clip_count))
                                 : config.gradient_clip_norm;
            metrics_reporter_->update_adaptive_clip_epoch(avg_clip, agc_spike_count);
            adai::Logger::info("Adaptive clip (epoch {}): avg_threshold={:.4f} spikes={}",
                               epoch + 1, avg_clip, agc_spike_count);
        }
        // ─────────────────────────────────────────────────────────────────────
    }
    // ─────────────────────────────────────────────────────────────────────────

    return epoch_loss;
}

/**
 * @brief Validate on validation set (inference-only, no weight updates)
 */
float ChatbotTrainer::validate() {
    if (tokenized_validation_data.empty()) {
        return 0.0f;
    }

    adai::Logger::info("🔍 Validating...");

    // Set model to evaluation mode
    model->set_training(false);

    float total_loss = 0.0f;
    int num_samples = static_cast<int>(tokenized_validation_data.size());

    for (int i = 0; i < num_samples; i++) {
        const auto& pair = tokenized_validation_data[i];

        try {
            // Use the tokenized overloads with the already-truncated ids from
            // tokenized_validation_data — the text overloads re-tokenize
            // pair.input_text/target_text from scratch with no length cap,
            // which let occasional long validation samples blow the GPU
            // memory budget (attention temporaries scale O(seq^2)) far past
            // what training ever allocates for the same dataset.
#ifdef ADAI_ENABLE_GPU
            float loss = model->gpu_evaluate_tokenized(pair.input_tokens, pair.target_tokens);

            // Drain the queue before the next sample — see the identical
            // comment in the training loop above. Validation sets can be far
            // larger than one accumulation window and have no optimizer-step
            // backpressure, so without this the unsynchronized backlog of
            // forward-pass temporaries grows unchecked across the whole
            // validation pass.
            model->gpu_synchronize();
#else
            float loss = model->evaluate_tokenized(pair.input_tokens, pair.target_tokens);
#endif
            total_loss += loss;
            if (config.enable_loss_quality_backfill) {
                validation_data[i].meta.quality = std::exp(-loss);
            }
        } catch (const std::exception& e) {
            adai::Logger::error("  ❌ Error validating sample {}: {}", (i + 1), e.what());
#ifdef ADAI_ENABLE_GPU
            // See the identical comment in the training loop's catch block above —
            // without this, a failed sample's deferred frees never retire and the
            // budget-exceeded failures compound for the rest of the validation pass.
            try {
                model->gpu_synchronize();
            } catch (const std::exception& sync_err) {
                adai::Logger::error("  ⚠️  gpu_synchronize() failed after sample {}: {}", (i + 1),
                                    sync_err.what());
            }
#endif
        }
    }

    // Restore training mode
    model->set_training(true);

    float validation_loss = total_loss / static_cast<float>(num_samples);
    float validation_perplexity = calculate_perplexity(validation_loss);

    validation_losses.push_back(validation_loss);
    validation_perplexities.push_back(validation_perplexity);

    // Update metrics service with validation metrics (TD-015: include accuracy and perplexity)
    if (metrics_reporter_) {
        metrics_reporter_->update_validation_metrics(validation_loss, -1.0f, validation_perplexity);
    }

    // Compute BLEU/ROUGE generation quality on a validation sample subset
    compute_generation_quality_metrics();

    log(LogLevel::NORMAL,
        "  Validation - Loss: " + std::to_string(validation_loss) +
            " - Perplexity: " + std::to_string(validation_perplexity),
        COLOR_INFO);

    // Track best model
    if (validation_loss < best_validation_loss - config.min_delta) {
        best_validation_loss = validation_loss;
        best_epoch = static_cast<int>(training_losses.size());
        epochs_without_improvement = 0;
        adai::Logger::info("  ⭐ New best validation loss!");

        // Update metrics service with best metrics
        if (metrics_reporter_) {
            metrics_reporter_->update_best_metrics(validation_loss, best_epoch);
        }

        // Save best model if early stopping is enabled
        if (config.enable_early_stopping && config.restore_best_weights) {
            best_model_path = "best_model_temp.bin";
            try {
                model->save_model(best_model_path);
                adai::Logger::info("  💾 Best model saved temporarily");
                if (best_model_callback_) {
                    best_model_callback_(best_epoch, best_validation_loss);
                }
            } catch (const std::exception& e) {
                adai::Logger::error("  ❌ Failed to save best model: {}", e.what());
            }
        }
    } else {
        epochs_without_improvement++;
        if (config.enable_early_stopping) {
            adai::Logger::warn("  ⏳ Epochs without improvement: {}/{}", epochs_without_improvement,
                               config.patience);
        }
    }

    return validation_loss;
}

void ChatbotTrainer::compute_generation_quality_metrics() {
    // Only run when explicitly enabled and a metrics service is connected
    if (!config.enable_generation_quality_metrics || !metrics_reporter_) {
        return;
    }
    if (tokenized_validation_data.empty()) {
        return;
    }

    // Determine sample subset: up to generation_quality_sample_size pairs chosen
    // from the front of the validation set for determinism across epochs.
    const int sample_size = std::min(static_cast<int>(tokenized_validation_data.size()),
                                     config.generation_quality_sample_size);

    // Join any outstanding scoring thread before launching a new one (TD-023)
    join_generation_quality_thread();

    if (sample_size >= config.generation_quality_async_threshold) {
        // Async path (TD-023): snapshot model weights and score in a background thread
        // so the training loop can begin the next epoch without waiting.
        //
        // Deliberately still CPU-only (snap->generate_response(), not
        // gpu_generate_response()): clone() rebuilds the model via
        // save_model()+load_model() and never calls gpu_init_training(), so
        // the clone's GPU-resident weight mirrors are never populated.
        // Wiring this up would also mean a background thread submitting
        // kernels to the same shared GPUManager queue concurrently with the
        // main training thread — needs its own validation before enabling.
        std::unique_ptr<EncoderDecoderModel> snapshot = model->clone();

        std::vector<std::string> refs, inputs;
        refs.reserve(sample_size);
        inputs.reserve(sample_size);
        for (int i = 0; i < sample_size; ++i) {
            refs.push_back(tokenized_validation_data[i].target_text);
            inputs.push_back(tokenized_validation_data[i].input_text);
        }

        const int max_tokens = config.generation_quality_max_tokens;
        IMetricsReporter* reporter = metrics_reporter_;

        generation_quality_thread_.emplace([snap = std::move(snapshot), refs = std::move(refs),
                                            inputs = std::move(inputs), max_tokens,
                                            reporter]() mutable {
            snap->set_training(false);
            std::vector<std::string> hypotheses;
            hypotheses.reserve(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                try {
                    hypotheses.push_back(snap->generate_response(inputs[i], max_tokens));
                } catch (const std::exception& e) {
                    adai::Logger::warn(
                        "BLEU/ROUGE async: skipping sample {} — generate_response() threw: {}",
                        static_cast<int>(i), e.what());
                    hypotheses.push_back("");
                }
            }
            if (!hypotheses.empty()) {
                GenerationQualityScore score =
                    GenerationQualityEvaluator::evaluate(refs, hypotheses);
                reporter->update_generation_quality_metrics(score.bleu4, score.rouge1, score.rouge2,
                                                            score.rougeL);
            }
        });
        return;
    }

    // Synchronous path (sample_size < generation_quality_async_threshold)
    std::vector<std::string> references, hypotheses;
    references.reserve(sample_size);
    hypotheses.reserve(sample_size);

    // Eval mode: no dropout / batch-norm updates during generation
    model->set_training(false);
    for (int i = 0; i < sample_size; ++i) {
        const auto& pair = tokenized_validation_data[i];
        references.push_back(pair.target_text);
        try {
#ifdef ADAI_ENABLE_GPU
            hypotheses.push_back(model->gpu_generate_response(
                pair.input_text, config.generation_quality_max_tokens));
#else
            hypotheses.push_back(
                model->generate_response(pair.input_text, config.generation_quality_max_tokens));
#endif
        } catch (const std::exception& e) {
            adai::Logger::warn("BLEU/ROUGE: skipping sample {} — generate_response() threw: {}", i,
                               e.what());
            hypotheses.push_back("");
        }
#ifdef ADAI_ENABLE_GPU
        // gpu_generate_response()'s autoregressive decode loop allocates several
        // temporaries per generated token with no synchronization point of its own.
        // Without draining the SYCL queue here (same rationale as the training/
        // validation loops above), GPUManager's allocated_bytes_ counter races ahead
        // of the deferred host_task frees and never catches up across this loop.
        try {
            model->gpu_synchronize();
        } catch (const std::exception& sync_err) {
            adai::Logger::warn("BLEU/ROUGE: gpu_synchronize() failed after sample {}: {}", i,
                               sync_err.what());
        }
#endif
    }
    model->set_training(true);

    if (references.empty()) {
        return;
    }

    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(references, hypotheses);
    metrics_reporter_->update_generation_quality_metrics(score.bleu4, score.rouge1, score.rouge2,
                                                         score.rougeL);
}

void ChatbotTrainer::backfill_generation_quality() {
    if (!model) {
        return;
    }

    adai::Logger::info("📊 Backfilling generation quality scores (BLEU4)...");
    model->set_training(false);

    auto score_sample = [&](const std::string& input, const std::string& target) -> float {
        float result = 0.0f;
        try {
#ifdef ADAI_ENABLE_GPU
            std::string hyp =
                model->gpu_generate_response(input, config.generation_backfill_max_tokens);
#else
            std::string hyp =
                model->generate_response(input, config.generation_backfill_max_tokens);
#endif
            GenerationQualityScore s = GenerationQualityEvaluator::evaluate({target}, {hyp});
            result = s.bleu4 >= 0.0f ? s.bleu4 : 0.0f;
        } catch (const std::exception& e) {
            adai::Logger::warn("backfill_generation_quality: generate_response() threw: {}",
                               e.what());
            result = 0.0f;
        }
#ifdef ADAI_ENABLE_GPU
        // This lambda runs once per sample across the entire training + validation
        // set (see the two loops below) — without draining the SYCL queue's deferred
        // frees after every call (same gap as the BLEU/ROUGE loop above), the
        // allocated_bytes_ accounting climbs across the whole dataset and eventually
        // exhausts the budget for whatever GPU work runs next.
        try {
            model->gpu_synchronize();
        } catch (const std::exception& sync_err) {
            adai::Logger::warn("backfill_generation_quality: gpu_synchronize() failed: {}",
                               sync_err.what());
        }
#endif
        return result;
    };

    for (size_t i = 0; i < training_data.size(); ++i) {
        training_data[i].meta.quality = score_sample(tokenized_training_data[i].input_text,
                                                     tokenized_training_data[i].target_text);
    }

    for (size_t i = 0; i < validation_data.size(); ++i) {
        validation_data[i].meta.quality = score_sample(tokenized_validation_data[i].input_text,
                                                       tokenized_validation_data[i].target_text);
    }

    model->set_training(true);
    adai::Logger::info(
        "  ✅ Generation quality backfill complete ({} training, {} validation samples)",
        training_data.size(), validation_data.size());
}

// New methods for incremental training support

bool ChatbotTrainer::train(int num_epochs) {
    config.num_epochs = num_epochs;
    was_aborted_ = false;

    try {
        // Initialize model if needed (this also initializes optimizer)
        if (!model) {
            initialize_model();
        }

        // Split data first (on raw text pairs)
        if (validation_data.empty() && config.validation_split > 0) {
            split_data();
        }

        // Preprocess data (tokenize split datasets)
        preprocess_data();

        // Calculate total steps
        int samples_per_update = config.gradient_accumulation_steps;
        int updates_per_epoch = static_cast<int>(
            (tokenized_training_data.size() + static_cast<size_t>(samples_per_update) - 1) /
            static_cast<size_t>(samples_per_update));
        total_training_steps = num_epochs * updates_per_epoch;

        // Train for specified epochs
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // train_epoch() pushes to training_losses / training_perplexities internally.
            float epoch_loss = train_epoch(epoch);

            // was_aborted_ was set inside train_epoch() if abort_flag_ fired at an
            // optimizer-step boundary — stop here rather than validating/starting
            // a further epoch on a training pass the caller has asked to drain.
            if (was_aborted_) {
                break;
            }

            // Validate
            // validate() pushes to validation_losses / validation_perplexities and
            // manages best_validation_loss / epochs_without_improvement internally.
            if (!tokenized_validation_data.empty()) {
                float val_loss = validate();

                // Update end_epoch with actual validation loss now that we have it (1-based epoch)
                if (metrics_reporter_) {
                    metrics_reporter_->end_epoch(epoch + 1, epoch_loss, val_loss,
                                                 current_learning_rate, std::exp(epoch_loss),
                                                 optimizer ? optimizer->get_gradient_norm() : 0.0f);
                }

                // Early stopping check (epochs_without_improvement maintained by validate())
                if (config.enable_early_stopping && epochs_without_improvement >= config.patience) {
                    early_stopped = true;
                    break;
                }
            }

            // Invoke per-epoch callback for real-time monitoring (TD-009)
            if (epoch_callback_) {
                float cb_val = validation_losses.empty() ? 0.0f : validation_losses.back();
                epoch_callback_(epoch, num_epochs, epoch_loss, cb_val, current_learning_rate);
            }
        }

        // validate() saves best_model_path whenever a new best validation loss is
        // found, but never reloads it — without this, training simply ends on
        // whatever the final epoch produced, even if validation regressed after
        // an earlier peak (the flag's name promises a restore that never happened).
        if (config.enable_early_stopping && config.restore_best_weights &&
            !best_model_path.empty()) {
            try {
                model->load_model(best_model_path);
                adai::Logger::info("  Restored best model weights (epoch {}, val_loss {:.4f})",
                                   best_epoch, best_validation_loss);
            } catch (const std::exception& e) {
                adai::Logger::error("  Failed to restore best model weights: {}", e.what());
            }
        }

        if (config.enable_generation_quality_backfill) {
            join_generation_quality_thread();  // finish any async scoring before backfill
            backfill_generation_quality();
        }

        return true;
    } catch (const std::exception& e) {
        adai::Logger::error("❌ Training failed: {}", e.what());
        return false;
    }
}

void ChatbotTrainer::set_tokenizer(std::unique_ptr<BPETokenizer> tok) {
    tokenizer = std::move(tok);
}

void ChatbotTrainer::set_model(std::unique_ptr<EncoderDecoderModel> mdl) {
    model = std::move(mdl);

    // Initialize optimizer for the model if not already done
    if (!optimizer) {
        optimizer = std::make_unique<Optimizer>(config.optimizer_type, config.learning_rate);
        optimizer->set_weight_decay(config.weight_decay);
        optimizer->set_max_grad_norm(config.gradient_clip_norm);

        if (config.optimizer_type == OptimizerType::ADAM ||
            config.optimizer_type == OptimizerType::ADAMW) {
            optimizer->set_betas(config.adam_beta1, config.adam_beta2);
        }

        // Register model parameters with optimizer
        model->register_parameters(*optimizer);
    }
}

std::unique_ptr<EncoderDecoderModel> ChatbotTrainer::release_model() {
    join_generation_quality_thread();  // TD-023: wait for any in-flight scoring thread
    return std::move(model);
}

BPETokenizer* ChatbotTrainer::release_tokenizer() {
    return tokenizer.release();
}

void ChatbotTrainer::add_training_pair(const std::string& input, const std::string& response) {
    training_data.emplace_back(input, response);
}

void ChatbotTrainer::add_validation_pair(const std::string& input, const std::string& response) {
    validation_data.emplace_back(input, response);
}

float ChatbotTrainer::get_final_training_loss() const {
    return training_losses.empty() ? 0.0f : training_losses.back();
}

float ChatbotTrainer::get_final_validation_loss() const {
    return validation_losses.empty() ? 0.0f : validation_losses.back();
}

void ChatbotTrainer::set_epoch_callback(EpochCallback cb) {
    epoch_callback_ = std::move(cb);
}

void ChatbotTrainer::set_sample_callback(SampleCallback cb) {
    sample_callback_ = std::move(cb);
}

void ChatbotTrainer::set_best_model_callback(BestModelCallback cb) {
    best_model_callback_ = std::move(cb);
}

void ChatbotTrainer::save_to(const std::string& path) {
    if (model) {
        model->save_model(path);
    }
}

void ChatbotTrainer::set_metrics_reporter(IMetricsReporter* reporter) {
    metrics_reporter_ = reporter;
}
