#include "ChatbotTrainer.hpp"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include "ConversationContext.hpp"
#include "GenerationQualityMetrics.hpp"
#include "Logger.hpp"
#ifdef ADAI_ENABLE_OPENMP
#include <omp.h>
#include <cmath>
#endif

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

/**
 * @brief Initialize tokenizer from vocabulary file
 */
bool ChatbotTrainer::load_tokenizer(const std::string& vocab_path) {
    adai::Logger::info("📚 Loading tokenizer from: {}", vocab_path);

    tokenizer = std::make_unique<BPETokenizer>();
    try {
        tokenizer->load_vocab(vocab_path);
        adai::Logger::info("✅ Tokenizer loaded (vocab size: {})", tokenizer->get_vocab_size());
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

    tokenizer = std::make_unique<BPETokenizer>();
    try {
        tokenizer->build_vocab(texts, vocab_size, 1);
        tokenizer->save_vocab(save_path);

        adai::Logger::info("✅ Vocabulary built (size: {})", tokenizer->get_vocab_size());
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

    std::string line;
    std::string current_input;
    std::string current_response;
    int pair_count = 0;

    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        if (line.empty()) {
            // End of pair
            if (!current_input.empty() && !current_response.empty()) {
                training_data.emplace_back(current_input, current_response);
                pair_count++;
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

    // Don't forget last pair
    if (!current_input.empty() && !current_response.empty()) {
        training_data.emplace_back(current_input, current_response);
        pair_count++;
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
            tokenized_training_data[i] =
                TokenizedPair(truncate(tokenizer->encode(clip_text(pair.input), false)),
                              truncate(tokenizer->encode(clip_text(pair.response), true)),
                              pair.input, pair.response);
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
            tokenized_validation_data[i] =
                TokenizedPair(truncate(tokenizer->encode(clip_text(pair.input), false)),
                              truncate(tokenizer->encode(clip_text(pair.response), true)),
                              pair.input, pair.response);
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
        const int enc_layers = model->get_encoder_layers();
        const int dec_layers = model->get_decoder_layers();
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
    }
    // ─────────────────────────────────────────────────────────────────────────

    for (int i = 0; i < num_samples; i++) {
        const auto& pair = tokenized_training_data[training_indices[i]];

        // Update learning rate based on schedule (only at optimizer step)
        if (accumulation_step == 0) {
            update_learning_rate();
        }

        try {
            // Zero gradients at the start of accumulation cycle
            if (accumulation_step == 0) {
                model->zero_grad();
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

            // Forward pass using cached tokenized data
            auto compute_t0 = std::chrono::steady_clock::now();
            Matrix logits = model->forward(pair.input_tokens, pair.target_tokens);

            // Compute loss
            float loss = model->compute_loss_for_training(logits, pair.target_tokens);

            // Scale loss by accumulation steps for proper gradient averaging
            float scaled_loss = loss / static_cast<float>(config.gradient_accumulation_steps);
            accumulated_loss += loss;  // Track unscaled for logging

            // Backward pass (accumulates gradients)
            Matrix grad_loss =
                model->compute_loss_gradient_for_training(logits, pair.target_tokens);

            // Scale gradients for accumulation by modifying in-place
            if (config.gradient_accumulation_steps > 1) {
                float scale = 1.0f / static_cast<float>(config.gradient_accumulation_steps);
                for (int r = 0; r < grad_loss.rows; r++) {
                    for (int c = 0; c < grad_loss.cols; c++) {
                        grad_loss.data[r][c] *= scale;
                    }
                }
            }

            model->backward_pass(grad_loss);
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
                    metrics_reporter_->update_adaptive_clip_metrics(effective_clip, agc_spike_count);
                }
                // ─────────────────────────────────────────────────────────────────

                // Update weights via optimizer
                optimizer->step();

                total_loss += accumulated_loss;
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
            std::to_string(avg_grad_norm) + " - Updates: " + std::to_string(num_updates),
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
        for (int l = 0; l < enc_layers; ++l) {
            model->get_encoder()->get_encoder_block(l)->get_feed_forward()->clear_activation_hook();
        }
        for (int l = 0; l < dec_layers; ++l) {
            model->get_decoder()->get_decoder_block(l)->get_feed_forward()->clear_activation_hook();
        }

        // ── TD-013: Attention entropy – report epoch average and clear hooks ──────
        float avg_entropy = (ent_count > 0) ? (ent_sum / static_cast<float>(ent_count)) : -1.0f;
        metrics_reporter_->update_attention_entropy(avg_entropy);
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
            // Use evaluate() which doesn't update weights
            float loss = model->evaluate(pair.input_text, pair.target_text);
            total_loss += loss;
        } catch (const std::exception& e) {
            adai::Logger::error("  ❌ Error validating sample {}: {}", (i + 1), e.what());
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

    std::vector<std::string> references, hypotheses;
    references.reserve(sample_size);
    hypotheses.reserve(sample_size);

    // Eval mode: no dropout / batch-norm updates during generation
    model->set_training(false);
    for (int i = 0; i < sample_size; ++i) {
        const auto& pair = tokenized_validation_data[i];
        references.push_back(pair.target_text);
        try {
            hypotheses.push_back(
                model->generate_response(pair.input_text, config.generation_quality_max_tokens));
        } catch (const std::exception& e) {
            adai::Logger::warn("BLEU/ROUGE: skipping sample {} — generate_response() threw: {}", i,
                               e.what());
            hypotheses.push_back("");
        }
    }
    model->set_training(true);

    if (references.empty()) {
        return;
    }

    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(references, hypotheses);
    metrics_reporter_->update_generation_quality_metrics(score.bleu4, score.rouge1, score.rouge2,
                                                        score.rougeL);
}

// New methods for incremental training support

bool ChatbotTrainer::train(int num_epochs) {
    config.num_epochs = num_epochs;

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
