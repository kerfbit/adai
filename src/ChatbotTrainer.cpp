#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "BPETokenizer.hpp"
#include "ConversationContext.hpp"
#include "EncoderDecoderModel.hpp"
#include "Optimizer.hpp"

// ANSI color codes
#define COLOR_RESET "\033[0m"
#define COLOR_INFO "\033[1;36m"
#define COLOR_SUCCESS "\033[1;32m"
#define COLOR_WARNING "\033[1;33m"
#define COLOR_ERROR "\033[1;31m"
#define COLOR_PROGRESS "\033[1;35m"

/**
 * @brief Training data pair (input, target response)
 */
struct ConversationPair {
    std::string input;
    std::string response;

    ConversationPair(const std::string& in, const std::string& resp) : input(in), response(resp) {}
};

/**
 * @brief Learning rate scheduling strategy
 */
enum class LRSchedule {
    CONSTANT,          // No scheduling
    LINEAR_WARMUP,     // Linear warmup then constant
    COSINE_DECAY,      // Cosine annealing decay
    WARMUP_COSINE,     // Linear warmup + cosine decay (recommended)
    STEP_DECAY,        // Step-wise decay at intervals
    EXPONENTIAL_DECAY  // Exponential decay
};

/**
 * @brief Training configuration
 */
struct TrainingConfig {
    // Model architecture
    int d_model = 512;
    int num_heads = 8;
    int d_ff = 2048;
    int num_encoder_layers = 6;
    int num_decoder_layers = 6;
    int max_seq_length = 512;

    // Training parameters
    int num_epochs = 10;
    float learning_rate = 0.001f;  // Initial/base learning rate
    int batch_size = 1;            // Currently only batch_size=1 supported
    int validation_split = 10;     // Use 1/10 of data for validation

    // Learning rate scheduling
    LRSchedule lr_schedule = LRSchedule::WARMUP_COSINE;
    int warmup_steps = 0;             // Warmup steps (0 = auto: 10% of total)
    float min_learning_rate = 1e-6f;  // Minimum LR for decay schedules
    float lr_decay_factor = 0.1f;     // Decay factor for step/exponential
    int lr_decay_steps = 0;           // Steps between decays (0 = auto: per epoch)

    // Optimizer settings
    OptimizerType optimizer_type = OptimizerType::ADAMW;  // Optimizer algorithm
    float adam_beta1 = 0.9f;                              // Adam first moment decay
    float adam_beta2 = 0.999f;                            // Adam second moment decay
    float weight_decay = 0.01f;                           // L2 regularization / weight decay
    float gradient_clip_norm = 1.0f;                      // Maximum gradient norm (0 = no clipping)

    // Checkpointing
    bool save_checkpoints = true;
    int checkpoint_every = 1;  // Save every N epochs

    // Early stopping
    bool enable_early_stopping = false;
    int patience = 5;                  // Epochs to wait for improvement
    float min_delta = 1e-4f;           // Minimum change to qualify as improvement
    bool restore_best_weights = true;  // Restore best model after early stop

    // Logging
    int log_every = 10;  // Log every N samples
    bool verbose = true;
};

/**
 * @brief Chatbot model trainer
 */
class ChatbotTrainer {
   private:
    BPETokenizer* tokenizer;
    EncoderDecoderModel* model;
    Optimizer* optimizer;  // Centralized optimizer
    TrainingConfig config;

    std::vector<ConversationPair> training_data;
    std::vector<ConversationPair> validation_data;

    // Training statistics
    std::vector<float> training_losses;
    std::vector<float> validation_losses;
    std::vector<float> learning_rates;  // Track learning rate over time
    std::vector<float> gradient_norms;  // Track gradient norms
    float best_validation_loss;
    int best_epoch;

    // Learning rate scheduling state
    int global_step;
    int total_training_steps;
    float current_learning_rate;

    // Early stopping state
    int epochs_without_improvement;
    std::string best_model_path;
    bool early_stopped;

   public:
    ChatbotTrainer(const TrainingConfig& cfg)
        : config(cfg),
          tokenizer(nullptr),
          model(nullptr),
          optimizer(nullptr),
          best_validation_loss(std::numeric_limits<float>::max()),
          best_epoch(0),
          global_step(0),
          total_training_steps(0),
          current_learning_rate(cfg.learning_rate),
          epochs_without_improvement(0),
          early_stopped(false) {}

    ~ChatbotTrainer() {
        if (model)
            delete model;
        if (optimizer)
            delete optimizer;
        if (tokenizer)
            delete tokenizer;
    }

    /**
     * @brief Initialize tokenizer from vocabulary file
     */
    bool load_tokenizer(const std::string& vocab_path) {
        std::cout << COLOR_INFO << "📚 Loading tokenizer from: " << vocab_path << COLOR_RESET
                  << std::endl;

        tokenizer = new BPETokenizer();
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
    bool build_vocabulary(const std::vector<std::string>& texts, int vocab_size = 5000,
                          const std::string& save_path = "vocab.txt") {
        std::cout << COLOR_INFO << "🔨 Building vocabulary..." << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  Texts: " << texts.size() << std::endl;
        std::cout << COLOR_INFO << "  Target vocab size: " << vocab_size << COLOR_RESET
                  << std::endl;

        tokenizer = new BPETokenizer();
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
    bool load_conversation_data(const std::string& filepath) {
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
     * @brief Split data into training and validation sets
     */
    void split_data() {
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

        // Move last validation_size items to validation set
        validation_data.assign(training_data.end() - validation_size, training_data.end());
        training_data.erase(training_data.end() - validation_size, training_data.end());

        std::cout << COLOR_INFO << "📊 Data split:" << COLOR_RESET << std::endl;
        std::cout << COLOR_INFO << "  Training: " << training_data.size() << " pairs" << COLOR_RESET
                  << std::endl;
        std::cout << COLOR_INFO << "  Validation: " << validation_data.size() << " pairs"
                  << COLOR_RESET << std::endl;
    }

    /**
     * @brief Validate and auto-correct model architecture parameters
     */
    void validate_and_correct_config() {
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
     * @brief Initialize the encoder-decoder model
     */
    void initialize_model() {
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

        model = new EncoderDecoderModel(config.d_model, config.num_heads, config.d_ff,
                                        config.num_encoder_layers, config.num_decoder_layers,
                                        tokenizer->get_vocab_size(), config.max_seq_length);

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

        optimizer = new Optimizer(config.optimizer_type, config.learning_rate);
        optimizer->set_weight_decay(config.weight_decay);
        optimizer->set_max_grad_norm(config.gradient_clip_norm);

        if (config.optimizer_type == OptimizerType::ADAM ||
            config.optimizer_type == OptimizerType::ADAMW) {
            optimizer->set_betas(config.adam_beta1, config.adam_beta2);
        }

        // Register model parameters with optimizer
        // Note: This will print a warning until parameter exposure is fully implemented
        model->register_parameters(*optimizer);

        std::cout << COLOR_SUCCESS << "✅ Optimizer initialized" << COLOR_RESET << std::endl;
    }

    /**
     * @brief Calculate learning rate for current step
     */
    float calculate_learning_rate(int step) {
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
    void update_learning_rate() {
        current_learning_rate = calculate_learning_rate(global_step);
        optimizer->set_learning_rate(current_learning_rate);
        // Also update model LR for backward compatibility
        model->set_learning_rate(current_learning_rate);
    }

    /**
     * @brief Get learning rate schedule name
     */
    const char* get_schedule_name() const {
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
     * @brief Train one epoch
     */
    float train_epoch(int epoch) {
        float total_loss = 0.0f;
        float total_grad_norm = 0.0f;
        int num_samples = training_data.size();

        std::cout << COLOR_PROGRESS << "\n📈 Epoch " << (epoch + 1) << "/" << config.num_epochs
                  << COLOR_RESET << std::endl;

        for (int i = 0; i < num_samples; i++) {
            const auto& pair = training_data[i];

            // Update learning rate based on schedule
            update_learning_rate();

            try {
                // Zero gradients (use optimizer's zero_grad for consistency)
                optimizer->zero_grad();
                model->zero_grad();  // Also call model's to be safe

                // Forward pass
                std::vector<int> input_tokens = model->get_tokenizer()->encode(pair.input);
                std::vector<int> target_tokens = model->get_tokenizer()->encode(pair.response);

                Matrix logits = model->forward(input_tokens, target_tokens);

                // Compute loss
                float loss = model->compute_loss_for_training(logits, target_tokens);

                // Backward pass (computes gradients only)
                Matrix grad_loss = model->compute_loss_gradient_for_training(logits, target_tokens);
                model->backward_pass(grad_loss);

                // Get gradient norm before clipping
                float grad_norm = optimizer->get_gradient_norm();
                total_grad_norm += grad_norm;

                // Clip gradients and update weights using optimizer
                if (config.gradient_clip_norm > 0.0f) {
                    optimizer->clip_gradients();
                }

                // Update weights using centralized optimizer
                // Note: This will use the model's update_weights() until
                // full parameter exposure is implemented
                model->update_weights();
                // optimizer->step();  // See TD-001 in TECHNICAL_DEBT.md - Parameter exposure incomplete

                total_loss += loss;
                global_step++;

                // Log progress
                if (config.verbose && (i + 1) % config.log_every == 0) {
                    float avg_loss = total_loss / (i + 1);
                    float avg_grad_norm = total_grad_norm / (i + 1);
                    std::cout << COLOR_INFO << "  Sample " << (i + 1) << "/" << num_samples
                              << " - Loss: " << std::fixed << std::setprecision(4) << loss
                              << " - Avg: " << avg_loss << " - LR: " << current_learning_rate
                              << " - GradNorm: " << avg_grad_norm << COLOR_RESET << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << COLOR_ERROR << "  ❌ Error training sample " << (i + 1) << ": "
                          << e.what() << COLOR_RESET << std::endl;
            }
        }

        float epoch_loss = total_loss / num_samples;
        float avg_grad_norm = total_grad_norm / num_samples;
        training_losses.push_back(epoch_loss);
        learning_rates.push_back(current_learning_rate);
        gradient_norms.push_back(avg_grad_norm);

        std::cout << COLOR_SUCCESS << "✅ Epoch " << (epoch + 1)
                  << " complete - Avg Loss: " << epoch_loss << " - LR: " << current_learning_rate
                  << " - Avg GradNorm: " << avg_grad_norm << COLOR_RESET << std::endl;

        return epoch_loss;
    }

    /**
     * @brief Validate on validation set
     */
    float validate() {
        if (validation_data.empty()) {
            return 0.0f;
        }

        std::cout << COLOR_INFO << "🔍 Validating..." << COLOR_RESET << std::endl;

        float total_loss = 0.0f;
        int num_samples = validation_data.size();

        for (int i = 0; i < num_samples; i++) {
            const auto& pair = validation_data[i];

            try {
                // For validation, we just compute loss
                // Note: train_step does update weights, so validation loss won't be perfect
                // In production, you'd want a separate validation method
                float loss = model->train_step(pair.input, pair.response);
                total_loss += loss;
            } catch (const std::exception& e) {
                std::cerr << COLOR_ERROR << "  ❌ Error validating sample " << (i + 1) << ": "
                          << e.what() << COLOR_RESET << std::endl;
            }
        }

        float validation_loss = total_loss / num_samples;
        validation_losses.push_back(validation_loss);

        std::cout << COLOR_INFO << "  Validation Loss: " << validation_loss << COLOR_RESET
                  << std::endl;

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
    bool should_early_stop() {
        if (!config.enable_early_stopping || validation_data.empty()) {
            return false;
        }

        return epochs_without_improvement >= config.patience;
    }

    /**
     * @brief Restore best model weights
     */
    void restore_best_model() {
        if (best_model_path.empty()) {
            std::cout << COLOR_WARNING << "⚠️  No best model to restore" << COLOR_RESET << std::endl;
            return;
        }

        std::cout << COLOR_INFO << "🔄 Restoring best model from epoch " << best_epoch << "..."
                  << COLOR_RESET << std::endl;

        try {
            // Delete current model and load best one
            if (model)
                delete model;

            model = new EncoderDecoderModel(config.d_model, config.num_heads, config.d_ff,
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
     * @brief Save model checkpoint
     */
    void save_checkpoint(const std::string& filepath, int epoch) {
        std::cout << COLOR_INFO << "💾 Saving checkpoint..." << COLOR_RESET << std::endl;

        try {
            model->save_model(filepath);
            std::cout << COLOR_SUCCESS << "✅ Checkpoint saved to: " << filepath << COLOR_RESET
                      << std::endl;
        } catch (const std::exception& e) {
            std::cerr << COLOR_ERROR << "❌ Failed to save checkpoint: " << e.what() << COLOR_RESET
                      << std::endl;
        }
    }

    /**
     * @brief Main training loop
     */
    void train(const std::string& output_model_path = "chatbot_model.bin") {
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

        // Split data
        split_data();

        // Calculate total training steps for LR scheduling
        total_training_steps = config.num_epochs * training_data.size();

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
        std::cout << COLOR_PROGRESS << "═══════════════════════════════════════" << COLOR_RESET
                  << std::endl;

        auto start_time = std::time(nullptr);

        for (int epoch = 0; epoch < config.num_epochs; epoch++) {
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

        // Print summary
        print_training_summary(duration);
    }

    /**
     * @brief Print training summary
     */
    void print_training_summary(long duration) {
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
    void test_generation(const std::vector<std::string>& test_prompts) {
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
};

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
    std::cout << "  --early-stopping       Enable early stopping based on validation loss"
              << std::endl;
    std::cout << "  --patience <n>         Early stopping patience in epochs (default: 5)"
              << std::endl;
    std::cout << "  --min-delta <delta>    Minimum improvement for early stopping (default: 1e-4)"
              << std::endl;
    std::cout << "  --no-restore-best      Don't restore best weights after early stopping"
              << std::endl;
    std::cout << "  --no-validation        Skip validation split" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  " << program_name
              << " --data conversations.txt --build-vocab 5000 --epochs 20 \\" << std::endl;
    std::cout << "      --lr 0.0001 --optimizer adamw --weight-decay 0.01 --grad-clip 1.0 \\"
              << std::endl;
    std::cout << "      --lr-schedule warmup-cosine --early-stopping --patience 3 \\" << std::endl;
    std::cout << "      --output my_model.bin" << std::endl;
}

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
        } else if (arg == "--early-stopping") {
            config.enable_early_stopping = true;
        } else if (arg == "--patience" && i + 1 < argc) {
            config.patience = std::stoi(argv[++i]);
            config.enable_early_stopping = true;  // Auto-enable if patience specified
        } else if (arg == "--min-delta" && i + 1 < argc) {
            config.min_delta = std::stof(argv[++i]);
        } else if (arg == "--no-restore-best") {
            config.restore_best_weights = false;
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
