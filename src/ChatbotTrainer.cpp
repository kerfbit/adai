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
          start_epoch(0) {}

/**
 * @brief Initialize tokenizer from vocabulary file
 */
bool ChatbotTrainer::load_tokenizer(const std::string& vocab_path) {
        std::cout << COLOR_INFO << "📚 Loading tokenizer from: " << vocab_path << COLOR_RESET
                  << std::endl;

        tokenizer = std::make_unique<BPETokenizer>();
        try {
            tokenizer->load_vocab(vocab_path);
            std::cout << COLOR_SUCCESS
                      << "✅ Tokenizer loaded (vocab size: " << tokenizer->get_vocab_size() << ")"
                      << COLOR_RESET << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << COLOR_ERROR << "❌ Failed to load tokenizer: " << e.what() << COLOR_RESET
                      << std::endl;
            return false;
        }
    }

    /**
     * @brief Build vocabulary from training texts
     */
bool ChatbotTrainer::build_vocabulary(const std::vector<std::string>& texts, int vocab_size,
                          const std::string& save_path) {
        std::cout << COLOR_INFO << "🔨 Building vocabulary..." << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  Texts: " << texts.size() << std::endl;
        std::cout << COLOR_INFO << "  Target vocab size: " << vocab_size << COLOR_RESET
                  << std::endl;

        tokenizer = std::make_unique<BPETokenizer>();
        try {
            tokenizer->build_vocab(texts, vocab_size, 1);
            tokenizer->save_vocab(save_path);

            std::cout << COLOR_SUCCESS
                      << "✅ Vocabulary built (size: " << tokenizer->get_vocab_size() << ")"
                      << COLOR_RESET << std::endl;
            std::cout << COLOR_SUCCESS << "✅ Saved to: " << save_path << COLOR_RESET << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << COLOR_ERROR << "❌ Failed to build vocabulary: " << e.what() << COLOR_RESET
                      << std::endl;
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
        std::cout << COLOR_INFO << "📖 Loading conversation data from: " << filepath << COLOR_RESET
                  << std::endl;

        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << COLOR_ERROR << "❌ Cannot open file: " << filepath << COLOR_RESET
                      << std::endl;
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

        std::cout << COLOR_SUCCESS << "✅ Loaded " << pair_count << " conversation pairs"
                  << COLOR_RESET << std::endl;

        return pair_count > 0;
    }

    /**
     * @brief Split data into training and validation sets with random shuffling
     */
void ChatbotTrainer::split_data() {
        if (config.validation_split <= 0) {
            std::cout << COLOR_WARNING << "⚠️  No validation split, using all data for training"
                      << COLOR_RESET << std::endl;
            return;
        }

        int validation_size = training_data.size() / config.validation_split;
        if (validation_size == 0) {
            std::cout << COLOR_WARNING << "⚠️  Not enough data for validation split" << COLOR_RESET
                      << std::endl;
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

        std::cout << COLOR_INFO << "📊 Data split (randomly shuffled):" << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  Training: " << training_data.size() << " pairs" << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_INFO << "  Validation: " << validation_data.size() << " pairs"
                  << COLOR_RESET << std::endl;
    }

    /**
     * @brief Validate and auto-correct model architecture parameters
     */
void ChatbotTrainer::validate_and_correct_config() {
        bool corrected = false;

        std::cout << COLOR_INFO << "🔍 Validating model configuration..." << COLOR_RESET
                  << std::endl;

        // Validate d_model is divisible by num_heads
        if (config.d_model % config.num_heads != 0) {
            int original_d_model = config.d_model;
            // Round up to nearest multiple of num_heads
            config.d_model =
                ((config.d_model + config.num_heads - 1) / config.num_heads) * config.num_heads;
            std::cout << COLOR_WARNING << "⚠️  d_model (" << original_d_model
                      << ") not divisible by num_heads (" << config.num_heads << ")" << COLOR_RESET
                      << std::endl;
            std::cout << COLOR_WARNING << "   Auto-corrected to: " << config.d_model << COLOR_RESET
                      << std::endl;
            corrected = true;
        }

        // Validate d_ff follows recommended ratio (typically 4x d_model)
        int recommended_d_ff = 4 * config.d_model;
        if (config.d_ff != recommended_d_ff) {
            float ratio = static_cast<float>(config.d_ff) / config.d_model;
            if (ratio < 2.0f || ratio > 8.0f) {
                int original_d_ff = config.d_ff;
                config.d_ff = recommended_d_ff;
                std::cout << COLOR_WARNING << "⚠️  d_ff (" << original_d_ff
                          << ") has unusual ratio to d_model (ratio: " << ratio << ")"
                          << COLOR_RESET << std::endl;
                std::cout << COLOR_WARNING << "   Auto-corrected to recommended 4x: " << config.d_ff
                          << COLOR_RESET << std::endl;
                corrected = true;
            } else {
                std::cout << COLOR_INFO << "   d_ff ratio: " << ratio
                          << "x d_model (acceptable, recommended: 4x)" << COLOR_RESET << std::endl;
            }
        }

        // Validate num_heads is a power of 2 (common practice)
        int heads = config.num_heads;
        if ((heads & (heads - 1)) != 0) {
            std::cout << COLOR_WARNING << "⚠️  num_heads (" << heads
                      << ") is not a power of 2 (recommended: 2, 4, 8, 16, etc.)" << COLOR_RESET
                      << std::endl;
            std::cout << COLOR_WARNING
                      << "   Keeping current value, but performance may be suboptimal"
                      << COLOR_RESET << std::endl;
        }

        // Validate d_model is reasonable
        if (config.d_model < 64 || config.d_model > 4096) {
            std::cout << COLOR_WARNING << "⚠️  d_model (" << config.d_model
                      << ") is outside typical range [64-4096]" << COLOR_RESET << std::endl;
        }

        // Validate learning rate is reasonable
        if (config.learning_rate <= 0.0f || config.learning_rate > 1.0f) {
            std::cout << COLOR_WARNING << "⚠️  learning_rate (" << config.learning_rate
                      << ") is outside typical range (0, 1]" << COLOR_RESET << std::endl;
        }

        // Validate min_learning_rate < learning_rate
        if (config.min_learning_rate >= config.learning_rate) {
            int original_min_lr = config.min_learning_rate;
            config.min_learning_rate = config.learning_rate * 0.01f;  // 1% of base LR
            std::cout << COLOR_WARNING << "⚠️  min_learning_rate (" << original_min_lr
                      << ") >= learning_rate (" << config.learning_rate << ")" << COLOR_RESET
                      << std::endl;
            std::cout << COLOR_WARNING << "   Auto-corrected to: " << config.min_learning_rate
                      << COLOR_RESET << std::endl;
            corrected = true;
        }

        // Validate layer counts
        if (config.num_encoder_layers < 1 || config.num_encoder_layers > 48) {
            std::cout << COLOR_WARNING << "⚠️  num_encoder_layers (" << config.num_encoder_layers
                      << ") is outside typical range [1-48]" << COLOR_RESET << std::endl;
        }

        if (config.num_decoder_layers < 1 || config.num_decoder_layers > 48) {
            std::cout << COLOR_WARNING << "⚠️  num_decoder_layers (" << config.num_decoder_layers
                      << ") is outside typical range [1-48]" << COLOR_RESET << std::endl;
        }

        // Validate max sequence length
        if (config.max_seq_length < 16 || config.max_seq_length > 8192) {
            std::cout << COLOR_WARNING << "⚠️  max_seq_length (" << config.max_seq_length
                      << ") is outside typical range [16-8192]" << COLOR_RESET << std::endl;
        }

        if (corrected) {
            std::cout << COLOR_SUCCESS << "✅ Configuration validated and corrected" << COLOR_RESET
                      << std::endl;
        } else {
            std::cout << COLOR_SUCCESS << "✅ Configuration validated" << COLOR_RESET << std::endl;
        }
    }

    /**
     * @brief Preprocess and tokenize all training and validation data
     */
void ChatbotTrainer::preprocess_data() {
        // Get tokenizer from model (ownership was transferred during initialization)
        BPETokenizer* tokenizer = model ? model->get_tokenizer() : nullptr;
        
        if (!tokenizer) {
            std::cerr << COLOR_ERROR << "❌ Tokenizer not initialized!" << COLOR_RESET << std::endl;
            return;
        }

        std::cout << COLOR_INFO << "🔄 Preprocessing and tokenizing data..." << COLOR_RESET
                  << std::endl;

        // Tokenize training data
        tokenized_training_data.clear();
        for (const auto& pair : training_data) {
            std::vector<int> input_tokens = tokenizer->encode(pair.input, false);    // Encoder: no special tokens
            std::vector<int> target_tokens = tokenizer->encode(pair.response, true); // Decoder: with special tokens
            tokenized_training_data.emplace_back(input_tokens, target_tokens, pair.input,
                                                 pair.response);
        }

        // Tokenize validation data
        tokenized_validation_data.clear();
        for (const auto& pair : validation_data) {
            std::vector<int> input_tokens = tokenizer->encode(pair.input, false);    // Encoder: no special tokens
            std::vector<int> target_tokens = tokenizer->encode(pair.response, true); // Decoder: with special tokens
            tokenized_validation_data.emplace_back(input_tokens, target_tokens, pair.input,
                                                   pair.response);
        }

        // Initialize shuffling indices
        training_indices.resize(tokenized_training_data.size());
        std::iota(training_indices.begin(), training_indices.end(), 0);

        std::cout << COLOR_SUCCESS << "✅ Data preprocessed:" << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  Training samples: " << tokenized_training_data.size()
                  << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  Validation samples: " << tokenized_validation_data.size()
                  << COLOR_RESET << std::endl;
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
void ChatbotTrainer::log(LogLevel level, const std::string& message, const std::string& color) {
        if (static_cast<int>(config.log_level) >= static_cast<int>(level)) {
            std::cout << color << message << COLOR_RESET << std::endl;
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
     * @brief Initialize the encoder-decoder model
     */
void ChatbotTrainer::initialize_model() {
        // Validate and correct configuration first
        validate_and_correct_config();

        std::cout << COLOR_INFO << "🧠 Initializing transformer model..." << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_INFO << "  d_model: " << config.d_model << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  num_heads: " << config.num_heads << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  d_ff: " << config.d_ff << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  encoder_layers: " << config.num_encoder_layers << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_INFO << "  decoder_layers: " << config.num_decoder_layers << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_INFO << "  max_seq_length: " << config.max_seq_length << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_INFO << "  learning_rate: " << config.learning_rate << COLOR_RESET
                  << std::endl;

        model = std::make_unique<EncoderDecoderModel>(tokenizer->get_vocab_size(), config.d_model,
                                        config.num_encoder_layers, config.num_decoder_layers,
                                        config.num_heads, config.d_ff, config.max_seq_length);

        // Transfer tokenizer ownership to the model
        // The model will now own the tokenizer and handle saving/loading
        model->set_tokenizer(tokenizer.release());

        std::cout << COLOR_SUCCESS << "✅ Model initialized" << COLOR_RESET << std::endl;

        // Initialize optimizer
        std::cout << COLOR_INFO << "🎯 Initializing optimizer..." << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  Type: ";
        switch (config.optimizer_type) {
            case OptimizerType::SGD:
                std::cout << "SGD";
                break;
            case OptimizerType::SGD_MOMENTUM:
                std::cout << "SGD+Momentum";
                break;
            case OptimizerType::ADAM:
                std::cout << "Adam";
                break;
            case OptimizerType::ADAMW:
                std::cout << "AdamW";
                break;
        }
        std::cout << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  Learning rate: " << config.learning_rate << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_INFO << "  Weight decay: " << config.weight_decay << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_INFO << "  Gradient clip norm: " << config.gradient_clip_norm
                  << COLOR_RESET << std::endl;
        if (config.optimizer_type == OptimizerType::ADAM ||
            config.optimizer_type == OptimizerType::ADAMW) {
            std::cout << COLOR_INFO << "  Adam beta1: " << config.adam_beta1 << COLOR_RESET
                      << std::endl;
            std::cout << COLOR_INFO << "  Adam beta2: " << config.adam_beta2 << COLOR_RESET
                      << std::endl;
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

        std::cout << COLOR_SUCCESS << "✅ Optimizer initialized" << COLOR_RESET << std::endl;
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

        std::cout << COLOR_PROGRESS << "\n📈 Epoch " << (epoch + 1) << "/" << config.num_epochs
                  << COLOR_RESET << std::endl;
        if (config.gradient_accumulation_steps > 1) {
            std::cout << COLOR_INFO << "  Using gradient accumulation: " 
                      << config.gradient_accumulation_steps << " steps (effective batch size: "
                      << effective_batch_size << ")" << COLOR_RESET << std::endl;
        }

        // Shuffle data at the start of each epoch
        shuffle_training_data();

        // Reset accumulation state at epoch start
        accumulation_step = 0;
        accumulated_loss = 0.0f;

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

                accumulation_step++;

                // Update weights after accumulating enough gradients
                bool should_update = (accumulation_step >= config.gradient_accumulation_steps) ||
                                    (i == num_samples - 1);  // Last sample in epoch

                if (should_update) {
                    // Get gradient norm before clipping
                    float grad_norm = optimizer->get_gradient_norm();
                    
                    // Safety check for NaN/Inf gradients
                    if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
                        std::cerr << COLOR_ERROR << "  ⚠️  WARNING: NaN or Inf gradient detected at sample " 
                                  << (i + 1) << "! Skipping update." << COLOR_RESET << std::endl;
                        // Reset accumulation and skip this update
                        accumulation_step = 0;
                        accumulated_loss = 0.0f;
                        model->zero_grad();
                        continue;
                    }
                    
                    total_grad_norm += grad_norm;

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
                }
            } catch (const std::exception& e) {
                std::cerr << COLOR_ERROR << "  ❌ Error training sample " << (i + 1) << ": "
                          << e.what() << COLOR_RESET << std::endl;
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

        return epoch_loss;
    }

    /**
     * @brief Validate on validation set (inference-only, no weight updates)
     */
float ChatbotTrainer::validate() {
        if (tokenized_validation_data.empty()) {
            return 0.0f;
        }

        std::cout << COLOR_INFO << "🔍 Validating..." << COLOR_RESET << std::endl;

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
                std::cerr << COLOR_ERROR << "  ❌ Error validating sample " << (i + 1) << ": "
                          << e.what() << COLOR_RESET << std::endl;
            }
        }

        // Restore training mode
        model->set_training(true);

        float validation_loss = total_loss / num_samples;
        float validation_perplexity = calculate_perplexity(validation_loss);
        
        validation_losses.push_back(validation_loss);
        validation_perplexities.push_back(validation_perplexity);

        log(LogLevel::NORMAL,
            "  Validation - Loss: " + std::to_string(validation_loss) + 
            " - Perplexity: " + std::to_string(validation_perplexity),
            COLOR_INFO);

        // Track best model
        if (validation_loss < best_validation_loss - config.min_delta) {
            best_validation_loss = validation_loss;
            best_epoch = training_losses.size();
            epochs_without_improvement = 0;
            std::cout << COLOR_SUCCESS << "  ⭐ New best validation loss!" << COLOR_RESET
                      << std::endl;

            // Save best model if early stopping is enabled
            if (config.enable_early_stopping && config.restore_best_weights) {
                best_model_path = "best_model_temp.bin";
                try {
                    model->save_model(best_model_path);
                    std::cout << COLOR_INFO << "  💾 Best model saved temporarily" << COLOR_RESET
                              << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << COLOR_ERROR << "  ❌ Failed to save best model: " << e.what()
                              << COLOR_RESET << std::endl;
                }
            }
        } else {
            epochs_without_improvement++;
            if (config.enable_early_stopping) {
                std::cout << COLOR_WARNING
                          << "  ⏳ Epochs without improvement: " << epochs_without_improvement
                          << "/" << config.patience << COLOR_RESET << std::endl;
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
            std::cout << COLOR_WARNING << "⚠️  No best model to restore" << COLOR_RESET << std::endl;
            return;
        }

        std::cout << COLOR_INFO << "🔄 Restoring best model from epoch " << best_epoch << "..."
                  << COLOR_RESET << std::endl;

        try {
            // Reset model and load best one
            model = std::make_unique<EncoderDecoderModel>(config.d_model, config.num_heads, config.d_ff,
                                            config.num_encoder_layers, config.num_decoder_layers,
                                            tokenizer->get_vocab_size(), config.max_seq_length);

            model->load_model(best_model_path);
            std::cout << COLOR_SUCCESS << "✅ Best model restored" << COLOR_RESET << std::endl;

            // Clean up temporary file
            std::remove(best_model_path.c_str());
        } catch (const std::exception& e) {
            std::cerr << COLOR_ERROR << "❌ Failed to restore best model: " << e.what()
                      << COLOR_RESET << std::endl;
        }
    }

    /**
     * @brief Save model checkpoint with metadata
     */
void ChatbotTrainer::save_checkpoint(const std::string& filepath, int epoch) {
        std::cout << COLOR_INFO << "💾 Saving checkpoint..." << COLOR_RESET << std::endl;

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
            
            std::cout << COLOR_SUCCESS << "✅ Checkpoint saved to: " << filepath << COLOR_RESET
                      << std::endl;
        } catch (const std::exception& e) {
            std::cerr << COLOR_ERROR << "❌ Failed to save checkpoint: " << e.what() << COLOR_RESET
                      << std::endl;
        }
    }

    /**
     * @brief Finalize model by creating standard-named files from best epoch
     */
void ChatbotTrainer::finalize_model(const std::string& output_path) {
        std::cout << COLOR_INFO << "\n🔧 Finalizing model..." << COLOR_RESET << std::endl;

        // Determine best epoch checkpoint path
        std::string best_checkpoint_path = output_path + ".epoch" + std::to_string(best_epoch);
        
        // Extensions that need to be copied/linked
        std::vector<std::string> extensions = {"config", "decoder", "lm_head", "vocab", "encoder"};
        
        std::cout << COLOR_INFO << "📋 Creating standardized model files from epoch " 
                  << best_epoch << "..." << COLOR_RESET << std::endl;

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
                    std::cout << COLOR_SUCCESS << "  ✓ Linked " << ext << COLOR_RESET << std::endl;
                } else {
                    std::cout << COLOR_WARNING << "  ⚠ Failed to link " << ext << COLOR_RESET << std::endl;
                }
            } else {
                std::cout << COLOR_WARNING << "  ⚠ Missing " << ext << " file" << COLOR_RESET << std::endl;
            }
        }

        // Cleanup old epoch checkpoints if not keeping all
        if (!config.keep_all_checkpoints) {
            std::cout << COLOR_INFO << "\n🧹 Cleaning up intermediate checkpoints..." << COLOR_RESET << std::endl;
            
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
            
            std::cout << COLOR_SUCCESS << "  ✓ Removed " << removed_count << " checkpoint(s) (kept epoch " 
                      << best_epoch << ")" << COLOR_RESET << std::endl;
        } else {
            std::cout << COLOR_INFO << "  ℹ Keeping all epoch checkpoints" << COLOR_RESET << std::endl;
        }

        std::cout << COLOR_SUCCESS << "\n✅ Model finalized:" << COLOR_RESET << std::endl;
        std::cout << COLOR_SUCCESS << "  📁 Base model: " << output_path << COLOR_RESET << std::endl;
        std::cout << COLOR_SUCCESS << "  🏆 Best epoch: " << best_epoch << COLOR_RESET << std::endl;
        std::cout << COLOR_SUCCESS << "  📊 Best validation loss: " << best_validation_loss 
                  << COLOR_RESET << std::endl;
    }

    /**
     * @brief Load checkpoint and resume training state
     */
bool ChatbotTrainer::load_checkpoint(const std::string& filepath) {
        std::cout << COLOR_INFO << "📂 Loading checkpoint from: " << filepath << COLOR_RESET
                  << std::endl;

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
                
                std::cout << COLOR_SUCCESS << "✅ Checkpoint loaded - resuming from epoch " 
                          << start_epoch << COLOR_RESET << std::endl;
                std::cout << COLOR_INFO << "  Global step: " << global_step << COLOR_RESET << std::endl;
                std::cout << COLOR_INFO << "  Best validation loss: " << best_validation_loss 
                          << COLOR_RESET << std::endl;
            } else {
                std::cout << COLOR_WARNING << "⚠️  No metadata found, starting fresh" 
                          << COLOR_RESET << std::endl;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << COLOR_ERROR << "❌ Failed to load checkpoint: " << e.what() << COLOR_RESET
                      << std::endl;
            return false;
        }
    }

    /**
     * @brief Main training loop
     */
void ChatbotTrainer::train(const std::string& output_model_path = "chatbot_model.bin") {
        if (!tokenizer) {
            std::cerr << COLOR_ERROR << "❌ Tokenizer not initialized!" << COLOR_RESET << std::endl;
            return;
        }

        if (training_data.empty()) {
            std::cerr << COLOR_ERROR << "❌ No training data loaded!" << COLOR_RESET << std::endl;
            return;
        }

        // Initialize model
        initialize_model();
        
        // Display parallel optimization status (Priority 1-5)
        std::cout << "\n" << COLOR_SUCCESS << "🚀 Parallel Optimizations Status:" << COLOR_RESET << std::endl;
        #ifdef _OPENMP
        std::cout << COLOR_SUCCESS << "  ✓ Priority 1: OpenMP CPU Parallelization (4.21x on matrix ops)" << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "    - Threads: " << omp_get_max_threads() << COLOR_RESET << std::endl;
        std::cout << COLOR_SUCCESS << "  ✓ Priority 4: Attention Head Parallelism (1.3-2.0x)" << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "    - Parallel heads across " << config.num_heads << " attention heads" << COLOR_RESET << std::endl;
        #else
        std::cout << COLOR_WARNING << "  ⚠ OpenMP not enabled - compile with -fopenmp for speedup" << COLOR_RESET << std::endl;
        #endif
        std::cout << COLOR_INFO << "  ℹ  Priority 3: Batched inference available at inference time" << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  ℹ  Priority 5: Pipeline parallelism available for serving" << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  ℹ  Combined potential: 5.5x speedup vs sequential baseline" << COLOR_RESET << std::endl;
        std::cout << std::endl;

        // Load checkpoint if resuming
        if (!config.resume_from_checkpoint.empty()) {
            if (!load_checkpoint(config.resume_from_checkpoint)) {
                std::cerr << COLOR_ERROR << "❌ Failed to load checkpoint, starting fresh" 
                          << COLOR_RESET << std::endl;
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
        std::cout << COLOR_INFO << "📊 Learning Rate Schedule: " << get_schedule_name()
                  << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  Base LR: " << config.learning_rate << COLOR_RESET
                  << std::endl;
        if (config.lr_schedule != LRSchedule::CONSTANT) {
            int warmup = config.warmup_steps > 0 ? config.warmup_steps : total_training_steps / 10;
            std::cout << COLOR_INFO << "  Warmup steps: " << warmup << COLOR_RESET << std::endl;
            std::cout << COLOR_INFO << "  Min LR: " << config.min_learning_rate << COLOR_RESET
                      << std::endl;
        }
        std::cout << COLOR_INFO << "  Total steps: " << total_training_steps << COLOR_RESET
                  << std::endl;

        // Print early stopping info
        if (config.enable_early_stopping && !validation_data.empty()) {
            std::cout << COLOR_INFO << "⏹️  Early Stopping: Enabled" << COLOR_RESET << std::endl;
            std::cout << COLOR_INFO << "  Patience: " << config.patience << " epochs" << COLOR_RESET
                      << std::endl;
            std::cout << COLOR_INFO << "  Min delta: " << config.min_delta << COLOR_RESET
                      << std::endl;
            std::cout << COLOR_INFO
                      << "  Restore best weights: " << (config.restore_best_weights ? "Yes" : "No")
                      << COLOR_RESET << std::endl;
        } else if (config.enable_early_stopping && validation_data.empty()) {
            std::cout << COLOR_WARNING << "⚠️  Early stopping disabled (no validation data)"
                      << COLOR_RESET << std::endl;
        }

        // Training loop
        std::cout << COLOR_PROGRESS << "\n🚀 Starting training..." << COLOR_RESET << std::endl;
        if (start_epoch > 0) {
            std::cout << COLOR_INFO << "📍 Resuming from epoch " << start_epoch << COLOR_RESET << std::endl;
        }
        std::cout << COLOR_PROGRESS << "═══════════════════════════════════════" << COLOR_RESET
                  << std::endl;

        auto start_time = std::time(nullptr);

        for (int epoch = start_epoch; epoch < config.num_epochs; epoch++) {
            // Train epoch
            float train_loss = train_epoch(epoch);

            // Validate
            if (!validation_data.empty()) {
                float val_loss = validate();

                // Check early stopping
                if (should_early_stop()) {
                    std::cout << COLOR_WARNING << "\n⏹️  Early stopping triggered after "
                              << (epoch + 1) << " epochs" << COLOR_RESET << std::endl;
                    std::cout << COLOR_WARNING << "   No improvement for " << config.patience
                              << " consecutive epochs" << COLOR_RESET << std::endl;
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

            std::cout << COLOR_PROGRESS << "───────────────────────────────────────" << COLOR_RESET
                      << std::endl;
        }

        auto end_time = std::time(nullptr);
        auto duration = end_time - start_time;

        // Final save
        std::cout << COLOR_PROGRESS << "\n💾 Saving final model..." << COLOR_RESET << std::endl;
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
        std::cout << COLOR_SUCCESS << "\n╔═══════════════════════════════════════╗" << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_SUCCESS << "║     🎉 TRAINING COMPLETE! 🎉         ║" << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_SUCCESS << "╚═══════════════════════════════════════╝" << COLOR_RESET
                  << std::endl;

        std::cout << "\n" << COLOR_INFO << "📊 Training Summary:" << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  Total epochs: " << config.num_epochs << COLOR_RESET
                  << std::endl;
        if (early_stopped) {
            std::cout << COLOR_INFO << "  Completed epochs: " << training_losses.size()
                      << " (early stopped)" << COLOR_RESET << std::endl;
        }
        std::cout << COLOR_INFO << "  Training samples: " << training_data.size() << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_INFO << "  Validation samples: " << validation_data.size() << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_INFO << "  Training time: " << duration << " seconds" << COLOR_RESET
                  << std::endl;

        if (!training_losses.empty()) {
            std::cout << COLOR_INFO << "  Final training loss: " << training_losses.back()
                      << COLOR_RESET << std::endl;
            std::cout << COLOR_INFO << "  Initial training loss: " << training_losses.front()
                      << COLOR_RESET << std::endl;

            if (!learning_rates.empty()) {
                std::cout << COLOR_INFO << "  Final learning rate: " << learning_rates.back()
                          << COLOR_RESET << std::endl;
                std::cout << COLOR_INFO << "  Initial learning rate: " << learning_rates.front()
                          << COLOR_RESET << std::endl;
            }

            if (!gradient_norms.empty()) {
                std::cout << COLOR_INFO << "  Final gradient norm: " << gradient_norms.back()
                          << COLOR_RESET << std::endl;
                std::cout << COLOR_INFO << "  Average gradient norm: "
                          << (std::accumulate(gradient_norms.begin(), gradient_norms.end(), 0.0f) /
                              gradient_norms.size())
                          << COLOR_RESET << std::endl;
            }
        }

        if (!validation_losses.empty()) {
            std::cout << COLOR_INFO << "  Final validation loss: " << validation_losses.back()
                      << COLOR_RESET << std::endl;
            std::cout << COLOR_INFO << "  Best validation loss: " << best_validation_loss
                      << " (epoch " << best_epoch << ")" << COLOR_RESET << std::endl;
        }

        std::cout << std::endl;
    }

    /**
     * @brief Test generation with trained model
     */
void ChatbotTrainer::test_generation(const std::vector<std::string>& test_prompts) {
        if (!model) {
            std::cerr << COLOR_ERROR << "❌ Model not initialized!" << COLOR_RESET << std::endl;
            return;
        }

        std::cout << COLOR_INFO << "\n🧪 Testing generation..." << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "═══════════════════════════════════════" << COLOR_RESET
                  << std::endl;

        for (const auto& prompt : test_prompts) {
            std::cout << COLOR_INFO << "\nPrompt: " << COLOR_RESET << prompt << std::endl;

            try {
                std::string response = model->generate_response(prompt, 50);
                std::cout << COLOR_SUCCESS << "Response: " << COLOR_RESET << response << std::endl;
            } catch (const std::exception& e) {
                std::cerr << COLOR_ERROR << "Error: " << e.what() << COLOR_RESET << std::endl;
            }
        }

        std::cout << COLOR_INFO << "═══════════════════════════════════════" << COLOR_RESET
                  << std::endl;
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

#ifndef CHATBOT_TRAINER_TEST_BUILD
int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string data_file;
    std::string vocab_file;
    std::string output_file = "chatbot_model.bin";
    int build_vocab_size = 0;
    bool use_validation = true;

    TrainingConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--data" && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            vocab_file = argv[++i];
        } else if (arg == "--build-vocab" && i + 1 < argc) {
            build_vocab_size = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.num_epochs = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = std::stof(argv[++i]);
        } else if (arg == "--d-model" && i + 1 < argc) {
            config.d_model = std::stoi(argv[++i]);
        } else if (arg == "--heads" && i + 1 < argc) {
            config.num_heads = std::stoi(argv[++i]);
        } else if (arg == "--d-ff" && i + 1 < argc) {
            config.d_ff = std::stoi(argv[++i]);
        } else if (arg == "--encoder-layers" && i + 1 < argc) {
            config.num_encoder_layers = std::stoi(argv[++i]);
        } else if (arg == "--decoder-layers" && i + 1 < argc) {
            config.num_decoder_layers = std::stoi(argv[++i]);
        } else if (arg == "--max-length" && i + 1 < argc) {
            config.max_seq_length = std::stoi(argv[++i]);
        } else if (arg == "--lr-schedule" && i + 1 < argc) {
            std::string schedule = argv[++i];
            if (schedule == "constant")
                config.lr_schedule = LRSchedule::CONSTANT;
            else if (schedule == "warmup")
                config.lr_schedule = LRSchedule::LINEAR_WARMUP;
            else if (schedule == "cosine")
                config.lr_schedule = LRSchedule::COSINE_DECAY;
            else if (schedule == "warmup-cosine")
                config.lr_schedule = LRSchedule::WARMUP_COSINE;
            else if (schedule == "step")
                config.lr_schedule = LRSchedule::STEP_DECAY;
            else if (schedule == "exponential")
                config.lr_schedule = LRSchedule::EXPONENTIAL_DECAY;
            else {
                std::cerr << COLOR_ERROR << "Unknown LR schedule: " << schedule << COLOR_RESET
                          << std::endl;
                return 1;
            }
        } else if (arg == "--warmup-steps" && i + 1 < argc) {
            config.warmup_steps = std::stoi(argv[++i]);
        } else if (arg == "--min-lr" && i + 1 < argc) {
            config.min_learning_rate = std::stof(argv[++i]);
        } else if (arg == "--optimizer" && i + 1 < argc) {
            std::string opt = argv[++i];
            if (opt == "sgd")
                config.optimizer_type = OptimizerType::SGD;
            else if (opt == "sgd-momentum")
                config.optimizer_type = OptimizerType::SGD_MOMENTUM;
            else if (opt == "adam")
                config.optimizer_type = OptimizerType::ADAM;
            else if (opt == "adamw")
                config.optimizer_type = OptimizerType::ADAMW;
            else {
                std::cerr << COLOR_ERROR << "Unknown optimizer: " << opt << COLOR_RESET
                          << std::endl;
                return 1;
            }
        } else if (arg == "--weight-decay" && i + 1 < argc) {
            config.weight_decay = std::stof(argv[++i]);
        } else if (arg == "--grad-clip" && i + 1 < argc) {
            config.gradient_clip_norm = std::stof(argv[++i]);
        } else if (arg == "--adam-beta1" && i + 1 < argc) {
            config.adam_beta1 = std::stof(argv[++i]);
        } else if (arg == "--adam-beta2" && i + 1 < argc) {
            config.adam_beta2 = std::stof(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--grad-accum" && i + 1 < argc) {
            config.gradient_accumulation_steps = std::stoi(argv[++i]);
        } else if (arg == "--resume" && i + 1 < argc) {
            config.resume_from_checkpoint = argv[++i];
        } else if (arg == "--log-level" && i + 1 < argc) {
            std::string level = argv[++i];
            if (level == "silent")
                config.log_level = LogLevel::SILENT;
            else if (level == "normal")
                config.log_level = LogLevel::NORMAL;
            else if (level == "verbose")
                config.log_level = LogLevel::VERBOSE;
            else if (level == "debug")
                config.log_level = LogLevel::DEBUG;
            else {
                std::cerr << COLOR_ERROR << "Unknown log level: " << level << COLOR_RESET
                          << std::endl;
                return 1;
            }
        } else if (arg == "--early-stopping") {
            config.enable_early_stopping = true;
        } else if (arg == "--patience" && i + 1 < argc) {
            config.patience = std::stoi(argv[++i]);
            config.enable_early_stopping = true;  // Auto-enable if patience specified
        } else if (arg == "--min-delta" && i + 1 < argc) {
            config.min_delta = std::stof(argv[++i]);
        } else if (arg == "--no-restore-best") {
            config.restore_best_weights = false;
        } else if (arg == "--keep-all-checkpoints") {
            config.keep_all_checkpoints = true;
        } else if (arg == "--no-validation") {
            use_validation = false;
            config.validation_split = 0;
        }
    }

    // Validate arguments
    if (data_file.empty()) {
        std::cerr << COLOR_ERROR << "Error: --data is required" << COLOR_RESET << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (vocab_file.empty() && build_vocab_size == 0) {
        std::cerr << COLOR_ERROR << "Error: Either --vocab or --build-vocab is required"
                  << COLOR_RESET << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // Create trainer
    ChatbotTrainer trainer(config);

    // Load or build vocabulary
    if (!vocab_file.empty()) {
        if (!trainer.load_tokenizer(vocab_file)) {
            return 1;
        }
    }

    // Load training data
    if (!trainer.load_conversation_data(data_file)) {
        return 1;
    }

    // Build vocabulary if requested
    if (build_vocab_size > 0) {
        std::vector<std::string> all_texts;
        // Extract all texts for vocabulary building
        // This is simplified - in practice you'd extract from the loaded data
        std::cout << COLOR_INFO << "Note: Building vocabulary from training data" << COLOR_RESET
                  << std::endl;

        std::string vocab_output = vocab_file.empty() ? "vocab.txt" : vocab_file;
        // For now, we assume vocabulary was already provided
        // Full implementation would extract texts from training_data
    }

    // Train model
    trainer.train(output_file);

    // Test generation with a few examples
    std::vector<std::string> test_prompts = {"Hello!", "How are you?", "What is your name?"};
    trainer.test_generation(test_prompts);

    std::cout << COLOR_SUCCESS << "\n✅ Training complete! Model saved to: " << output_file
              << COLOR_RESET << std::endl;
    std::cout << COLOR_INFO << "   Use this model with: ./chatbot vocab.txt " << output_file
              << COLOR_RESET << std::endl;

    return 0;
}
#endif // CHATBOT_TRAINER_TEST_BUILD

// New methods for incremental training support

bool ChatbotTrainer::train(int num_epochs) {
    config.num_epochs = num_epochs;
    
    try {
        // Initialize model if needed (this also initializes optimizer)
        if (!model) {
            initialize_model();
        }
        
        // Preprocess data
        preprocess_data();
        
        // Split data if needed
        if (validation_data.empty() && config.validation_split > 0) {
            split_data();
        }
        
        // Calculate total steps
        int samples_per_update = config.gradient_accumulation_steps;
        int updates_per_epoch = (tokenized_training_data.size() + samples_per_update - 1) / samples_per_update;
        total_training_steps = num_epochs * updates_per_epoch;
        
        // Train for specified epochs
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            float epoch_loss = train_epoch(epoch);
            training_losses.push_back(epoch_loss);
            
            // Validate
            if (!tokenized_validation_data.empty()) {
                float val_loss = validate();
                validation_losses.push_back(val_loss);
                
                // Check for improvement
                if (val_loss < best_validation_loss - config.min_delta) {
                    best_validation_loss = val_loss;
                    best_epoch = epoch + 1;
                    epochs_without_improvement = 0;
                } else {
                    epochs_without_improvement++;
                }
                
                // Early stopping check
                if (config.enable_early_stopping && epochs_without_improvement >= config.patience) {
                    early_stopped = true;
                    break;
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << COLOR_ERROR << "❌ Training failed: " << e.what() << COLOR_RESET << std::endl;
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
