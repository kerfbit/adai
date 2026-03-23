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
#include "Logger.hpp"
#ifdef ADAI_ENABLE_OPENMP
#include <omp.h>
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
          best_epoch(0),
          global_step(0),
          total_training_steps(0),
          current_learning_rate(cfg.learning_rate),
          accumulation_step(0),
          accumulated_loss(0.0f),
          epochs_without_improvement(0),
          early_stopped(false),
          start_epoch(0),
          metrics_service_(nullptr) {}

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

        int validation_size = training_data.size() / config.validation_split;
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
            adai::Logger::warn("⚠️  d_model ({}) not divisible by num_heads ({})", original_d_model, config.num_heads);
            adai::Logger::warn("   Auto-corrected to: {}", config.d_model);
            corrected = true;
        }

        // Validate d_ff follows recommended ratio (typically 4x d_model)
        int recommended_d_ff = 4 * config.d_model;
        if (config.d_ff != recommended_d_ff) {
            float ratio = static_cast<float>(config.d_ff) / config.d_model;
            if (ratio < 2.0f || ratio > 8.0f) {
                int original_d_ff = config.d_ff;
                config.d_ff = recommended_d_ff;
                adai::Logger::warn("⚠️  d_ff ({}) has unusual ratio to d_model (ratio: {})", original_d_ff, ratio);
                adai::Logger::warn("   Auto-corrected to recommended 4x: {}", config.d_ff);
                corrected = true;
            } else {
                adai::Logger::info("   d_ff ratio: {}x d_model (acceptable, recommended: 4x)", ratio);
            }
        }

        // Validate num_heads is a power of 2 (common practice)
        int heads = config.num_heads;
        if ((heads & (heads - 1)) != 0) {
            adai::Logger::warn("⚠️  num_heads ({}) is not a power of 2 (recommended: 2, 4, 8, 16, etc.)", heads);
            adai::Logger::warn("   Keeping current value, but performance may be suboptimal");
        }

        // Validate d_model is reasonable
        if (config.d_model < 64 || config.d_model > 4096) {
            adai::Logger::warn("⚠️  d_model ({}) is outside typical range [64-4096]", config.d_model);
        }

        // Validate learning rate is reasonable
        if (config.learning_rate <= 0.0f || config.learning_rate > 1.0f) {
            adai::Logger::warn("⚠️  learning_rate ({}) is outside typical range (0, 1]", config.learning_rate);
        }

        // Validate min_learning_rate < learning_rate
        if (config.min_learning_rate >= config.learning_rate) {
            int original_min_lr = config.min_learning_rate;
            config.min_learning_rate = config.learning_rate * 0.01f;  // 1% of base LR
            adai::Logger::warn("⚠️  min_learning_rate ({}) >= learning_rate ({})", original_min_lr, config.learning_rate);
            adai::Logger::warn("   Auto-corrected to: {}", config.min_learning_rate);
            corrected = true;
        }

        // Validate layer counts
        if (config.num_encoder_layers < 1 || config.num_encoder_layers > 48) {
            adai::Logger::warn("⚠️  num_encoder_layers ({}) is outside typical range [1-48]", config.num_encoder_layers);
        }

        if (config.num_decoder_layers < 1 || config.num_decoder_layers > 48) {
            adai::Logger::warn("⚠️  num_decoder_layers ({}) is outside typical range [1-48]", config.num_decoder_layers);
        }

        // Validate max sequence length
        if (config.max_seq_length < 16 || config.max_seq_length > 8192) {
            adai::Logger::warn("⚠️  max_seq_length ({}) is outside typical range [16-8192]", config.max_seq_length);
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

        // Tokenize training data — parallel BPE encoding (tokenizer is read-only after load)
        const int n_train = static_cast<int>(training_data.size());
        tokenized_training_data.clear();
        tokenized_training_data.resize(n_train);
#ifdef ADAI_ENABLE_OPENMP
        #pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int i = 0; i < n_train; i++) {
            const auto& pair = training_data[i];
            tokenized_training_data[i] = TokenizedPair(
                tokenizer->encode(pair.input,    false),   // Encoder: no special tokens
                tokenizer->encode(pair.response, true),    // Decoder: with special tokens
                pair.input, pair.response);
        }

        // Tokenize validation data — parallel BPE encoding
        const int n_val = static_cast<int>(validation_data.size());
        tokenized_validation_data.clear();
        tokenized_validation_data.resize(n_val);
#ifdef ADAI_ENABLE_OPENMP
        #pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int i = 0; i < n_val; i++) {
            const auto& pair = validation_data[i];
            tokenized_validation_data[i] = TokenizedPair(
                tokenizer->encode(pair.input,    false),   // Encoder: no special tokens
                tokenizer->encode(pair.response, true),    // Decoder: with special tokens
                pair.input, pair.response);
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
float ChatbotTrainer::calculate_accuracy(const std::vector<int>& predictions, const std::vector<int>& targets) {
        if (predictions.empty() || targets.empty() || predictions.size() != targets.size()) {
            return -1.0f;  // Not implemented yet
        }
        
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == targets[i]) {
                correct++;
            }
        }
        return static_cast<float>(correct) / predictions.size();
    }

    /**
     * @brief Single entry point for EncoderDecoderModel construction.
     * Reads vocab size from the current tokenizer and all architecture
     * dimensions from config.
     */
void ChatbotTrainer::build_model() {
        model = std::make_unique<EncoderDecoderModel>(
            tokenizer->get_vocab_size(),
            config.d_model,
            config.num_encoder_layers,
            config.num_decoder_layers,
            config.num_heads,
            config.d_ff,
            config.max_seq_length);
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
                case OptimizerType::SGD:           opt_type_str = "SGD"; break;
                case OptimizerType::SGD_MOMENTUM:  opt_type_str = "SGD+Momentum"; break;
                case OptimizerType::ADAM:          opt_type_str = "Adam"; break;
                case OptimizerType::ADAMW:         opt_type_str = "AdamW"; break;
                default:                           opt_type_str = "Unknown"; break;
            }
            adai::Logger::info("  Type: {}", opt_type_str);
        }
        adai::Logger::info("  Learning rate: {}", config.learning_rate);
        adai::Logger::info("  Weight decay: {}", config.weight_decay);
        adai::Logger::info("  Gradient clip norm: {}", config.gradient_clip_norm);
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
                    return lr * (static_cast<float>(step) / warmup);
                }
                return lr;

            case LRSchedule::COSINE_DECAY: {
                float progress = static_cast<float>(step) / total_training_steps;
                float cosine = 0.5f * (1.0f + std::cos(3.14159265359f * progress));
                return config.min_learning_rate + (lr - config.min_learning_rate) * cosine;
            }

            case LRSchedule::WARMUP_COSINE: {
                // Warmup phase
                if (step < warmup) {
                    return lr * (static_cast<float>(step) / warmup);
                }
                // Cosine decay phase
                float progress =
                    static_cast<float>(step - warmup) / (total_training_steps - warmup);
                float cosine = 0.5f * (1.0f + std::cos(3.14159265359f * progress));
                return config.min_learning_rate + (lr - config.min_learning_rate) * cosine;
            }

            case LRSchedule::STEP_DECAY: {
                int decay_steps = config.lr_decay_steps;
                if (decay_steps == 0) {
                    decay_steps = total_training_steps / config.num_epochs;
                }
                int num_decays = step / decay_steps;
                return lr * std::pow(config.lr_decay_factor, num_decays);
            }

            case LRSchedule::EXPONENTIAL_DECAY: {
                int decay_steps = config.lr_decay_steps;
                if (decay_steps == 0) {
                    decay_steps = total_training_steps / config.num_epochs;
                }
                float decay_rate = std::pow(config.lr_decay_factor, 1.0f / decay_steps);
                return lr * std::pow(decay_rate, step);
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
        int num_samples = tokenized_training_data.size();
        int effective_batch_size = config.batch_size * config.gradient_accumulation_steps;

        // Notify metrics service that epoch is starting (1-based epoch number)
        if (metrics_service_) {
            metrics_service_->start_epoch(epoch + 1, num_samples);
        }

        log(LogLevel::VERBOSE, "\n📈 Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(config.num_epochs));
        if (config.gradient_accumulation_steps > 1) {
            log(LogLevel::VERBOSE, "  Using gradient accumulation: " + std::to_string(config.gradient_accumulation_steps) +
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
        float wu_ratio_sum = 0.0f;  int wu_count = 0;
        // Cache outlier thresholds from service config once per epoch
        MetricsServiceConfig adv_cfg;
        if (metrics_service_) { adv_cfg = metrics_service_->get_config(); }
        // Epoch-level wall-clock start for compute_time_ratio denominator
        auto epoch_td013_start = std::chrono::steady_clock::now();
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
                }

                // Forward pass using cached tokenized data
                auto compute_t0 = std::chrono::steady_clock::now();
                Matrix logits = model->forward(pair.input_tokens, pair.target_tokens);

                // Compute loss
                float loss = model->compute_loss_for_training(logits, pair.target_tokens);
                
                // Scale loss by accumulation steps for proper gradient averaging
                float scaled_loss = loss / config.gradient_accumulation_steps;
                accumulated_loss += loss;  // Track unscaled for logging

                // Backward pass (accumulates gradients)
                Matrix grad_loss = model->compute_loss_gradient_for_training(logits, pair.target_tokens);
                
                // Scale gradients for accumulation by modifying in-place
                if (config.gradient_accumulation_steps > 1) {
                    float scale = 1.0f / config.gradient_accumulation_steps;
                    for (int r = 0; r < grad_loss.rows; r++) {
                        for (int c = 0; c < grad_loss.cols; c++) {
                            grad_loss.data[r][c] *= scale;
                        }
                    }
                }
                
                model->backward_pass(grad_loss);
                // Accumulate compute time (forward + backward)
                total_compute_ns += static_cast<double>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - compute_t0).count());

                accumulation_step++;

                // Update weights after accumulating enough gradients
                bool should_update = (accumulation_step >= config.gradient_accumulation_steps) ||
                                    (i == num_samples - 1);  // Last sample in epoch

                if (should_update) {
                    // Get gradient norm before clipping
                    float grad_norm = optimizer->get_gradient_norm();

                    // Save step-level metrics here so sample callback can fire even on NaN
                    float step_loss_for_cb = (config.gradient_accumulation_steps > 0)
                                                ? accumulated_loss / config.gradient_accumulation_steps
                                                : accumulated_loss;

                    // Safety check for NaN/Inf gradients
                    if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
                        adai::Logger::error("  ⚠️  WARNING: NaN or Inf gradient detected at sample {}! Skipping update.", (i + 1));
                        // Still fire sample callback so the dashboard shows progress
                        if (sample_callback_) {
                            float running_avg = (update_count > 0) ? total_loss / update_count : 0.0f;
                            sample_callback_(i + 1, num_samples, running_avg, step_loss_for_cb, 0.0f, current_learning_rate);
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
                        gn_w_M2   += gn_delta * (grad_norm - gn_w_mean);

                        // Welford update: per-step loss (for outlier z-score)
                        ls_w_count++;
                        float ls_delta = step_loss_for_cb - ls_w_mean;
                        ls_w_mean += ls_delta / ls_w_count;
                        ls_w_M2   += ls_delta * (step_loss_for_cb - ls_w_mean);

                        // Weight-update ratio approximation: (lr * ||g||) / ||w||)
                        // Threshold is 1e-10f (not 1e-6f) so near-zero initialized weights
                        // are still counted — get_weight_norm() returns sqrt(sum-of-squares)
                        // which is always >= 0, so 1e-10f guards only true divide-by-zero.
                        if (optimizer) {
                            float w_norm = optimizer->get_weight_norm();
                            if (w_norm > 1e-10f) {
                                wu_ratio_sum += current_learning_rate * grad_norm / w_norm;
                                ++wu_count;
                            }
                        }

                        // Outlier detection — flag samples with extreme grad_norm or loss z-score
                        if (metrics_service_) {
                            bool grad_outlier = grad_norm > adv_cfg.grad_norm_outlier_threshold;
                            bool loss_outlier = false;
                            if (ls_w_count >= 10.0f) {
                                float ls_std = (ls_w_M2 > 0.0f)
                                             ? std::sqrt(ls_w_M2 / ls_w_count) : 0.0f;
                                float ls_z   = (ls_std > 1e-7f)
                                             ? (step_loss_for_cb - ls_w_mean) / ls_std : 0.0f;
                                loss_outlier = (ls_z > adv_cfg.loss_outlier_z_threshold);
                            }
                            if (grad_outlier || loss_outlier) {
                                AbnormalSample ab;
                                ab.epoch      = epoch + 1;
                                ab.sample_id  = i + 1;
                                ab.loss       = step_loss_for_cb;
                                ab.grad_norm  = grad_norm;
                                ab.input_text  = tokenized_training_data[training_indices[i]].input_text;
                                ab.target_text = tokenized_training_data[training_indices[i]].target_text;
                                ab.timestamp  = std::chrono::system_clock::now();
                                if (grad_outlier && loss_outlier)
                                    ab.reason = "grad_norm_and_loss_outlier";
                                else if (grad_outlier)
                                    ab.reason = "grad_norm_outlier";
                                else
                                    ab.reason = "loss_outlier";
                                metrics_service_->flag_abnormal_sample(ab);
                            }
                        }
                    }
                    // ─────────────────────────────────────────────────────────────────────

                    // ── TD-013: emit running advanced metrics every optimizer step ─────────
                    if (metrics_service_) {
                        float running_gv = (gn_w_count >= 2.0f) ? (gn_w_M2 / gn_w_count) : 0.0f;
                        double elapsed_wall_ns = static_cast<double>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - epoch_td013_start).count());
                        float running_ctr = (elapsed_wall_ns > 0.0)
                            ? static_cast<float>(total_compute_ns / elapsed_wall_ns) : 0.0f;
                        float running_wur = (wu_count > 0)
                            ? (wu_ratio_sum / static_cast<float>(wu_count)) : 0.0f;
                        metrics_service_->update_advanced_epoch_metrics(running_gv, running_ctr, running_wur);
                    }
                    // ─────────────────────────────────────────────────────────────────────

                    // Clip gradients
                    if (config.gradient_clip_norm > 0.0f) {
                        optimizer->clip_gradients();
                    }

                    // Update weights via optimizer
                    optimizer->step();

                    total_loss += accumulated_loss;
                    global_step++;

                    // Log progress
                    if ((i + 1) % (config.log_every * config.gradient_accumulation_steps) == 0 || i == num_samples - 1) {
                        int num_updates = global_step - (epoch * (num_samples / config.gradient_accumulation_steps));
                        float avg_loss = total_loss / num_updates;
                        float avg_grad_norm = total_grad_norm / num_updates;
                        
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
                        float running_avg = (update_count > 0) ? total_loss / update_count : 0.0f;
                        sample_callback_(i + 1, num_samples, running_avg, step_loss_for_cb, grad_norm, current_learning_rate);
                    }

                    // Update metrics service with sample-level metrics
                    if (metrics_service_) {
                        metrics_service_->update_sample_metrics(i + 1, step_loss_for_cb, grad_norm, current_learning_rate);
                    }
                }
            } catch (const std::exception& e) {
                adai::Logger::error("  ❌ Error training sample {}: {}", (i + 1), e.what());
                // Still fire sample callback so the dashboard shows the sample was attempted
                if (sample_callback_) {
                    float running_avg = (update_count > 0) ? total_loss / update_count : 0.0f;
                    sample_callback_(i + 1, num_samples, running_avg, 0.0f, 0.0f, current_learning_rate);
                }
                // Reset accumulation on error
                accumulation_step = 0;
                accumulated_loss = 0.0f;
            }
        }

        // CRITICAL FIX: Divide by number of actual updates, not num_samples
        // total_loss only accumulates when should_update is true
        int num_updates = global_step - (epoch * (num_samples / config.gradient_accumulation_steps));
        float epoch_loss = (num_updates > 0) ? (total_loss / num_updates) : 0.0f;
        float avg_grad_norm = (num_updates > 0) ? (total_grad_norm / num_updates) : 0.0f;
        float epoch_perplexity = calculate_perplexity(epoch_loss);
        
        training_losses.push_back(epoch_loss);
        training_perplexities.push_back(epoch_perplexity);
        learning_rates.push_back(current_learning_rate);
        gradient_norms.push_back(avg_grad_norm);

        log(LogLevel::NORMAL, 
            "✅ Epoch " + std::to_string(epoch + 1) + 
            " complete - Loss: " + std::to_string(epoch_loss) + 
            " - Perplexity: " + std::to_string(epoch_perplexity) +
            " - LR: " + std::to_string(current_learning_rate) +
            " - GradNorm: " + std::to_string(avg_grad_norm) +
            " - Updates: " + std::to_string(num_updates),
            COLOR_SUCCESS);

        // ── TD-013: emit advanced epoch diagnostics to metrics service ────────────
        if (metrics_service_) {
            float gradient_variance = (gn_w_count >= 2.0f) ? (gn_w_M2 / gn_w_count) : 0.0f;
            double total_wall_ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - epoch_td013_start).count());
            float compute_time_ratio = (total_wall_ns > 0.0) ? static_cast<float>(total_compute_ns / total_wall_ns) : 0.0f;
            float avg_wu_ratio       = (wu_count > 0) ? (wu_ratio_sum / static_cast<float>(wu_count)) : 0.0f;
            metrics_service_->update_advanced_epoch_metrics(gradient_variance, compute_time_ratio, avg_wu_ratio);
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
        int num_samples = tokenized_validation_data.size();

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

        float validation_loss = total_loss / num_samples;
        float validation_perplexity = calculate_perplexity(validation_loss);
        
        validation_losses.push_back(validation_loss);
        validation_perplexities.push_back(validation_perplexity);

        // Update metrics service with validation metrics (TD-015: include accuracy and perplexity)
        if (metrics_service_) {
            metrics_service_->update_validation_metrics(validation_loss, -1.0f, validation_perplexity);
        }

        log(LogLevel::NORMAL,
            "  Validation - Loss: " + std::to_string(validation_loss) + 
            " - Perplexity: " + std::to_string(validation_perplexity),
            COLOR_INFO);

        // Track best model
        if (validation_loss < best_validation_loss - config.min_delta) {
            best_validation_loss = validation_loss;
            best_epoch = training_losses.size();
            epochs_without_improvement = 0;
            adai::Logger::info("  ⭐ New best validation loss!");

            // Update metrics service with best metrics
            if (metrics_service_) {
                metrics_service_->update_best_metrics(validation_loss, best_epoch);
            }

            // Save best model if early stopping is enabled
            if (config.enable_early_stopping && config.restore_best_weights) {
                best_model_path = "best_model_temp.bin";
                try {
                    model->save_model(best_model_path);
                    adai::Logger::info("  💾 Best model saved temporarily");
                } catch (const std::exception& e) {
                    adai::Logger::error("  ❌ Failed to save best model: {}", e.what());
                }
            }
        } else {
            epochs_without_improvement++;
            if (config.enable_early_stopping) {
                adai::Logger::warn("  ⏳ Epochs without improvement: {}/{}", epochs_without_improvement, config.patience);
            }
        }

        return validation_loss;
    }

    /**
     * @brief Check if early stopping criteria is met
     */
bool ChatbotTrainer::should_early_stop() {
        if (!config.enable_early_stopping || validation_data.empty()) {
            return false;
        }

        return epochs_without_improvement >= config.patience;
    }

    /**
     * @brief Restore best model weights
     */
void ChatbotTrainer::restore_best_model() {
        if (best_model_path.empty()) {
            adai::Logger::warn("⚠️  No best model to restore");
            return;
        }

        adai::Logger::info("🔄 Restoring best model from epoch {}...", best_epoch);

        try {
            // The model is already sized correctly from initialize_model().
            // Just reload the weights in-place — no reconstruction needed.
            model->load_model(best_model_path);
            adai::Logger::info("✅ Best model restored");

            // Clean up temporary file
            std::remove(best_model_path.c_str());
        } catch (const std::exception& e) {
            adai::Logger::error("❌ Failed to restore best model: {}", e.what());
        }
    }

    /**
     * @brief Save model checkpoint with metadata
     */
void ChatbotTrainer::save_checkpoint(const std::string& filepath, int epoch) {
        adai::Logger::info("💾 Saving checkpoint...");

        try {
            model->save_model(filepath);
            
            // Save training state metadata
            std::string metadata_path = filepath + ".metadata";
            std::ofstream meta_file(metadata_path);
            if (meta_file.is_open()) {
                meta_file << "epoch=" << epoch << "\n";
                meta_file << "global_step=" << global_step << "\n";
                meta_file << "learning_rate=" << current_learning_rate << "\n";
                meta_file << "best_validation_loss=" << best_validation_loss << "\n";
                meta_file << "best_epoch=" << best_epoch << "\n";
                meta_file.close();
            }
            
            adai::Logger::info("✅ Checkpoint saved to: {}", filepath);
        } catch (const std::exception& e) {
            adai::Logger::error("❌ Failed to save checkpoint: {}", e.what());
        }
    }

    /**
     * @brief Finalize model by creating standard-named files from best epoch
     */
void ChatbotTrainer::finalize_model(const std::string& output_path) {
        adai::Logger::info("\n🔧 Finalizing model...");

        // Determine best epoch checkpoint path
        std::string best_checkpoint_path = output_path + ".epoch" + std::to_string(best_epoch);
        
        // Extensions that need to be copied/linked
        std::vector<std::string> extensions = {"config", "decoder", "lm_head", "vocab", "encoder"};
        
        adai::Logger::info("📋 Creating standardized model files from epoch {}...", best_epoch);

        // Create empty base file (required for ifstream check in load_model)
        std::ofstream base_file(output_path);
        base_file.close();

        // Create symlinks for all component files
        for (const auto& ext : extensions) {
            std::string src_file = best_checkpoint_path + "." + ext;
            std::string dest_file = output_path + "." + ext;
            
            std::ifstream src(src_file);
            if (src.good()) {
                src.close();
                
                // Remove existing destination if it exists
                std::remove(dest_file.c_str());
                
                // Create symlink using relative path
                std::string src_basename = src_file.substr(src_file.find_last_of("/") + 1);
                std::string link_cmd = "ln -sf " + src_basename + " " + dest_file;
                int result = std::system(link_cmd.c_str());
                
                if (result == 0) {
                    adai::Logger::info("  ✓ Linked {}", ext);
                } else {
                    adai::Logger::warn("  ⚠ Failed to link {}", ext);
                }
            } else {
                adai::Logger::warn("  ⚠ Missing {} file", ext);
            }
        }

        // Cleanup old epoch checkpoints if not keeping all
        if (!config.keep_all_checkpoints) {
            adai::Logger::info("\n🧹 Cleaning up intermediate checkpoints...");
            
            int removed_count = 0;
            for (int epoch = 1; epoch <= config.num_epochs; epoch++) {
                if (epoch == best_epoch) {
                    continue;  // Keep the best epoch
                }
                
                std::string epoch_base = output_path + ".epoch" + std::to_string(epoch);
                
                // Remove all component files for this epoch
                for (const auto& ext : extensions) {
                    std::string file_to_remove = epoch_base + "." + ext;
                    std::remove(file_to_remove.c_str());
                }
                
                // Remove metadata file
                std::remove((epoch_base + ".metadata").c_str());
                removed_count++;
            }
            
            adai::Logger::info("  ✓ Removed {} checkpoint(s) (kept epoch {})", removed_count, best_epoch);
        } else {
            adai::Logger::info("  ℹ Keeping all epoch checkpoints");
        }

        adai::Logger::info("\n✅ Model finalized:");
        adai::Logger::info("  📁 Base model: {}", output_path);
        adai::Logger::info("  🏆 Best epoch: {}", best_epoch);
        adai::Logger::info("  📊 Best validation loss: {}", best_validation_loss);
    }

    /**
     * @brief Load checkpoint and resume training state
     */
bool ChatbotTrainer::load_checkpoint(const std::string& filepath) {
        adai::Logger::info("📂 Loading checkpoint from: {}", filepath);

        try {
            // Load model weights
            model->load_model(filepath);
            
            // Load training state metadata if available
            std::string metadata_path = filepath + ".metadata";
            std::ifstream meta_file(metadata_path);
            if (meta_file.is_open()) {
                std::string line;
                while (std::getline(meta_file, line)) {
                    size_t pos = line.find('=');
                    if (pos != std::string::npos) {
                        std::string key = line.substr(0, pos);
                        std::string value = line.substr(pos + 1);
                        
                        if (key == "epoch") {
                            start_epoch = std::stoi(value) + 1;  // Resume from next epoch
                        } else if (key == "global_step") {
                            global_step = std::stoi(value);
                        } else if (key == "learning_rate") {
                            current_learning_rate = std::stof(value);
                        } else if (key == "best_validation_loss") {
                            best_validation_loss = std::stof(value);
                        } else if (key == "best_epoch") {
                            best_epoch = std::stoi(value);
                        }
                    }
                }
                meta_file.close();
                
                adai::Logger::info("✅ Checkpoint loaded - resuming from epoch {}", start_epoch);
                adai::Logger::info("  Global step: {}", global_step);
                adai::Logger::info("  Best validation loss: {}", best_validation_loss);
            } else {
                adai::Logger::warn("⚠️  No metadata found, starting fresh");
            }
            
            return true;
        } catch (const std::exception& e) {
            adai::Logger::error("❌ Failed to load checkpoint: {}", e.what());
            return false;
        }
    }

    /**
     * @brief Main training loop
     */
void ChatbotTrainer::train(const std::string& output_model_path = "chatbot_model.bin") {
        if (!tokenizer) {
            adai::Logger::error("❌ Tokenizer not initialized!");
            return;
        }

        if (training_data.empty()) {
            adai::Logger::error("❌ No training data loaded!");
            return;
        }

        // Initialize model
        initialize_model();
        
        // Display parallel optimization status (Priority 1-5)
        adai::Logger::info("🚀 Parallel Optimizations Status:");
        #ifdef _OPENMP
        adai::Logger::info("  ✓ Priority 1: OpenMP CPU Parallelization (4.21x on matrix ops)");
        adai::Logger::info("    - Threads: {}", omp_get_max_threads());
        adai::Logger::info("  ✓ Priority 4: Attention Head Parallelism (1.3-2.0x)");
        adai::Logger::info("    - Parallel heads across {} attention heads", config.num_heads);
        #else
        adai::Logger::warn("  ⚠ OpenMP not enabled - compile with -fopenmp for speedup");
        #endif
        adai::Logger::info("  ℹ  Priority 3: Batched inference available at inference time");
        adai::Logger::info("  ℹ  Priority 5: Pipeline parallelism available for serving");
        adai::Logger::info("  ℹ  Combined potential: 5.5x speedup vs sequential baseline");

        // Load checkpoint if resuming
        if (!config.resume_from_checkpoint.empty()) {
            if (!load_checkpoint(config.resume_from_checkpoint)) {
                adai::Logger::error("❌ Failed to load checkpoint, starting fresh");
                start_epoch = 0;
            }
        }

        // Split data
        split_data();

        // Preprocess and tokenize all data
        preprocess_data();

        // Calculate total training steps for LR scheduling
        // Account for gradient accumulation - each optimizer step processes multiple samples
        int samples_per_update = config.gradient_accumulation_steps;
        int updates_per_epoch = (tokenized_training_data.size() + samples_per_update - 1) / samples_per_update;
        total_training_steps = config.num_epochs * updates_per_epoch;

        // Print LR schedule info
        adai::Logger::info("📊 Learning Rate Schedule: {}", get_schedule_name());
        adai::Logger::info("  Base LR: {}", config.learning_rate);
        if (config.lr_schedule != LRSchedule::CONSTANT) {
            int warmup = config.warmup_steps > 0 ? config.warmup_steps : total_training_steps / 10;
            adai::Logger::info("  Warmup steps: {}", warmup);
            adai::Logger::info("  Min LR: {}", config.min_learning_rate);
        }
        adai::Logger::info("  Total steps: {}", total_training_steps);

        // Print early stopping info
        if (config.enable_early_stopping && !validation_data.empty()) {
            adai::Logger::info("⏹️  Early Stopping: Enabled");
            adai::Logger::info("  Patience: {} epochs", config.patience);
            adai::Logger::info("  Min delta: {}", config.min_delta);
            adai::Logger::info("  Restore best weights: {}", (config.restore_best_weights ? "Yes" : "No"));
        } else if (config.enable_early_stopping && validation_data.empty()) {
            adai::Logger::warn("⚠️  Early stopping disabled (no validation data)");
        }

        // Training loop
        adai::Logger::info("\n🚀 Starting training...");
        if (start_epoch > 0) {
            adai::Logger::info("📍 Resuming from epoch {}", start_epoch);
        }
        adai::Logger::info("═══════════════════════════════════════");

        auto start_time = std::time(nullptr);

        for (int epoch = start_epoch; epoch < config.num_epochs; epoch++) {
            // Train epoch
            float train_loss = train_epoch(epoch);

            // Validate
            if (!validation_data.empty()) {
                float val_loss = validate();

                // Check early stopping
                if (should_early_stop()) {
                    adai::Logger::warn("\n⏹️  Early stopping triggered after {} epochs", (epoch + 1));
                    adai::Logger::warn("   No improvement for {} consecutive epochs", config.patience);
                    early_stopped = true;

                    // Restore best model if configured
                    if (config.restore_best_weights) {
                        restore_best_model();
                    }

                    break;
                }
            }

            // Save checkpoint
            if (config.save_checkpoints && (epoch + 1) % config.checkpoint_every == 0) {
                std::string checkpoint_path =
                    output_model_path + ".epoch" + std::to_string(epoch + 1);
                save_checkpoint(checkpoint_path, epoch);
            }

            adai::Logger::info("───────────────────────────────────────");
        }

        auto end_time = std::time(nullptr);
        auto duration = end_time - start_time;

        // Final save
        adai::Logger::info("\n💾 Saving final model...");
        save_checkpoint(output_model_path, config.num_epochs);

        // Finalize model - create standard named files from best epoch
        finalize_model(output_model_path);

        // Print summary
        print_training_summary(duration);
    }

    /**
     * @brief Print training summary
     */
void ChatbotTrainer::print_training_summary(long duration) {
        adai::Logger::info("\n╔═══════════════════════════════════════╗");
        adai::Logger::info("║     🎉 TRAINING COMPLETE! 🎉         ║");
        adai::Logger::info("╚═══════════════════════════════════════╝");

        adai::Logger::info("📊 Training Summary:");
        adai::Logger::info("  Total epochs: {}", config.num_epochs);
        if (early_stopped) {
            adai::Logger::info("  Completed epochs: {} (early stopped)", training_losses.size());
        }
        adai::Logger::info("  Training samples: {}", training_data.size());
        adai::Logger::info("  Validation samples: {}", validation_data.size());
        adai::Logger::info("  Training time: {} seconds", duration);

        if (!training_losses.empty()) {
            adai::Logger::info("  Final training loss: {}", training_losses.back());
            adai::Logger::info("  Initial training loss: {}", training_losses.front());

            if (!learning_rates.empty()) {
                adai::Logger::info("  Final learning rate: {}", learning_rates.back());
                adai::Logger::info("  Initial learning rate: {}", learning_rates.front());
            }

            if (!gradient_norms.empty()) {
                adai::Logger::info("  Final gradient norm: {}", gradient_norms.back());
                adai::Logger::info("  Average gradient norm: {}", (std::accumulate(gradient_norms.begin(), gradient_norms.end(), 0.0f) / gradient_norms.size()));
            }
        }

        if (!validation_losses.empty()) {
            adai::Logger::info("  Final validation loss: {}", validation_losses.back());
            adai::Logger::info("  Best validation loss: {} (epoch {})", best_validation_loss, best_epoch);
        }

    }

    /**
     * @brief Test generation with trained model
     */
void ChatbotTrainer::test_generation(const std::vector<std::string>& test_prompts) {
        if (!model) {
            adai::Logger::error("❌ Model not initialized!");
            return;
        }

        adai::Logger::info("\n🧪 Testing generation...");
        adai::Logger::info("═══════════════════════════════════════");

        for (const auto& prompt : test_prompts) {
            adai::Logger::info("\nPrompt: {}", prompt);

            try {
                std::string response = model->generate_response(prompt, 50);
                adai::Logger::info("Response: {}", response);
            } catch (const std::exception& e) {
                adai::Logger::error("Error: {}", e.what());
            }
        }

        adai::Logger::info("═══════════════════════════════════════");
}

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --data <file>          Training data file (required)" << std::endl;
    std::cout << "  --vocab <file>         Load vocabulary from file" << std::endl;
    std::cout << "  --build-vocab <size>   Build vocabulary from data (default: 5000)" << std::endl;
    std::cout << "  --output <file>        Output model file (default: chatbot_model.bin)"
              << std::endl;
    std::cout << "  --epochs <n>           Number of training epochs (default: 10)" << std::endl;
    std::cout << "  --lr <rate>            Learning rate (default: 0.001)" << std::endl;
    std::cout << "  --d-model <n>          Model dimension (default: 512)" << std::endl;
    std::cout << "  --heads <n>            Number of attention heads (default: 8)" << std::endl;
    std::cout << "  --d-ff <n>             Feed-forward dimension (default: 2048)" << std::endl;
    std::cout << "  --encoder-layers <n>   Number of encoder layers (default: 6)" << std::endl;
    std::cout << "  --decoder-layers <n>   Number of decoder layers (default: 6)" << std::endl;
    std::cout << "  --max-length <n>       Maximum sequence length (default: 512)" << std::endl;
    std::cout << "  --lr-schedule <name>   LR schedule: constant, warmup, cosine, warmup-cosine,"
              << std::endl;
    std::cout << "                         step, exponential (default: warmup-cosine)" << std::endl;
    std::cout << "  --warmup-steps <n>     Warmup steps (default: auto 10%)" << std::endl;
    std::cout << "  --min-lr <rate>        Minimum learning rate (default: 1e-6)" << std::endl;
    std::cout
        << "  --optimizer <name>     Optimizer: sgd, sgd-momentum, adam, adamw (default: adamw)"
        << std::endl;
    std::cout << "  --weight-decay <val>   Weight decay / L2 regularization (default: 0.01)"
              << std::endl;
    std::cout << "  --grad-clip <norm>     Gradient clipping max norm (default: 1.0, 0=disabled)"
              << std::endl;
    std::cout << "  --adam-beta1 <val>     Adam beta1 parameter (default: 0.9)" << std::endl;
    std::cout << "  --adam-beta2 <val>     Adam beta2 parameter (default: 0.999)" << std::endl;
    std::cout << "  --batch-size <n>       Batch size for training (default: 1)" << std::endl;
    std::cout << "  --grad-accum <n>       Gradient accumulation steps (default: 1)" << std::endl;
    std::cout << "  --resume <file>        Resume training from checkpoint (NEW)" << std::endl;
    std::cout << "  --log-level <level>    Logging: silent, normal, verbose, debug (default: verbose, NEW)" << std::endl;
    std::cout << "  --early-stopping       Enable early stopping based on validation loss"
              << std::endl;
    std::cout << "  --patience <n>         Early stopping patience in epochs (default: 5)"
              << std::endl;
    std::cout << "  --min-delta <delta>    Minimum improvement for early stopping (default: 1e-4)"
              << std::endl;
    std::cout << "  --no-restore-best      Don't restore best weights after early stopping"
              << std::endl;
    std::cout << "  --keep-all-checkpoints Keep all epoch checkpoints (default: keep only best)"
              << std::endl;
    std::cout << "  --no-validation        Skip validation split" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  " << program_name
              << " --data conversations.txt --build-vocab 5000 --epochs 20 \\" << std::endl;
    std::cout << "      --lr 0.0001 --optimizer adamw --weight-decay 0.01 --grad-clip 1.0 \\"
              << std::endl;
    std::cout << "      --batch-size 4 --grad-accum 8 --lr-schedule warmup-cosine \\" << std::endl;
    std::cout << "      --early-stopping --patience 3 --output my_model.bin" << std::endl;
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
        int updates_per_epoch = (tokenized_training_data.size() + samples_per_update - 1) / samples_per_update;
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
                if (metrics_service_) {
                    metrics_service_->end_epoch(epoch + 1, epoch_loss, val_loss, current_learning_rate,
                                               std::exp(epoch_loss), optimizer ? optimizer->get_gradient_norm() : 0.0f);
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

void ChatbotTrainer::set_metrics_service(TrainingMetricsService* service) {
    metrics_service_ = service;
}
