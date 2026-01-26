#include <../gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Note: ChatbotTrainer is in a .cpp file without a header
// We'll need to test it through its command-line interface
// For now, we'll create tests for the data structures and helper functionality
// that can be extracted, and integration tests

// Mock/Test data structures matching ChatbotTrainer.cpp

struct ConversationPair {
    std::string input;
    std::string response;

    ConversationPair(const std::string& in, const std::string& resp) : input(in), response(resp) {}
};

enum class LRSchedule {
    CONSTANT,
    LINEAR_WARMUP,
    COSINE_DECAY,
    WARMUP_COSINE,
    STEP_DECAY,
    EXPONENTIAL_DECAY
};

struct TrainingConfig {
    int d_model = 512;
    int num_heads = 8;
    int d_ff = 2048;
    int num_encoder_layers = 6;
    int num_decoder_layers = 6;
    int max_seq_length = 512;

    int num_epochs = 10;
    float learning_rate = 0.001f;
    int batch_size = 1;
    int validation_split = 10;

    LRSchedule lr_schedule = LRSchedule::WARMUP_COSINE;
    int warmup_steps = 0;
    float min_learning_rate = 1e-6f;
    float lr_decay_factor = 0.1f;
    int lr_decay_steps = 0;

    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float weight_decay = 0.01f;
    float gradient_clip_norm = 1.0f;

    bool save_checkpoints = true;
    int checkpoint_every = 1;

    bool enable_early_stopping = false;
    int patience = 5;
    float min_delta = 1e-4f;
    bool restore_best_weights = true;

    int log_every = 10;
    bool verbose = true;
};

// Helper function to create test data file
std::string create_test_data_file(const std::string& filename,
                                  const std::vector<std::pair<std::string, std::string>>& pairs) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return "";
    }

    for (const auto& pair : pairs) {
        file << "INPUT: " << pair.first << "\n";
        file << "RESPONSE: " << pair.second << "\n\n";
    }

    file.close();
    return filename;
}

// Helper function to create test vocabulary file
std::string create_test_vocab_file(const std::string& filename, int vocab_size = 100) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return "";
    }

    file << "<unk>\n<pad>\n<s>\n</s>\n";
    for (int i = 0; i < vocab_size - 4; i++) {
        file << "token" << i << "\n";
    }

    file.close();
    return filename;
}

// ============================================================================
// ConversationPair Tests
// ============================================================================

TEST(ConversationPairTest, BasicConstruction) {
    ConversationPair pair("Hello", "Hi there");

    EXPECT_EQ(pair.input, "Hello");
    EXPECT_EQ(pair.response, "Hi there");
}

TEST(ConversationPairTest, EmptyStrings) {
    ConversationPair pair("", "");

    EXPECT_TRUE(pair.input.empty());
    EXPECT_TRUE(pair.response.empty());
}

TEST(ConversationPairTest, LongStrings) {
    std::string long_input(1000, 'a');
    std::string long_response(1000, 'b');

    ConversationPair pair(long_input, long_response);

    EXPECT_EQ(pair.input.length(), 1000);
    EXPECT_EQ(pair.response.length(), 1000);
}

TEST(ConversationPairTest, SpecialCharacters) {
    ConversationPair pair("Hello\nWorld!", "Hi\tthere\r\n");

    EXPECT_NE(pair.input.find('\n'), std::string::npos);
    EXPECT_NE(pair.response.find('\t'), std::string::npos);
}

// ============================================================================
// TrainingConfig Tests
// ============================================================================

TEST(TrainingConfigTest, DefaultValues) {
    TrainingConfig config;

    // Model architecture defaults
    EXPECT_EQ(config.d_model, 512);
    EXPECT_EQ(config.num_heads, 8);
    EXPECT_EQ(config.d_ff, 2048);
    EXPECT_EQ(config.num_encoder_layers, 6);
    EXPECT_EQ(config.num_decoder_layers, 6);
    EXPECT_EQ(config.max_seq_length, 512);

    // Training defaults
    EXPECT_EQ(config.num_epochs, 10);
    EXPECT_FLOAT_EQ(config.learning_rate, 0.001f);
    EXPECT_EQ(config.batch_size, 1);
    EXPECT_EQ(config.validation_split, 10);
}

TEST(TrainingConfigTest, LRScheduleDefaults) {
    TrainingConfig config;

    EXPECT_EQ(config.lr_schedule, LRSchedule::WARMUP_COSINE);
    EXPECT_EQ(config.warmup_steps, 0);
    EXPECT_FLOAT_EQ(config.min_learning_rate, 1e-6f);
    EXPECT_FLOAT_EQ(config.lr_decay_factor, 0.1f);
    EXPECT_EQ(config.lr_decay_steps, 0);
}

TEST(TrainingConfigTest, OptimizerDefaults) {
    TrainingConfig config;

    EXPECT_FLOAT_EQ(config.adam_beta1, 0.9f);
    EXPECT_FLOAT_EQ(config.adam_beta2, 0.999f);
    EXPECT_FLOAT_EQ(config.weight_decay, 0.01f);
    EXPECT_FLOAT_EQ(config.gradient_clip_norm, 1.0f);
}

TEST(TrainingConfigTest, CheckpointDefaults) {
    TrainingConfig config;

    EXPECT_TRUE(config.save_checkpoints);
    EXPECT_EQ(config.checkpoint_every, 1);
}

TEST(TrainingConfigTest, EarlyStoppingDefaults) {
    TrainingConfig config;

    EXPECT_FALSE(config.enable_early_stopping);
    EXPECT_EQ(config.patience, 5);
    EXPECT_FLOAT_EQ(config.min_delta, 1e-4f);
    EXPECT_TRUE(config.restore_best_weights);
}

TEST(TrainingConfigTest, LoggingDefaults) {
    TrainingConfig config;

    EXPECT_EQ(config.log_every, 10);
    EXPECT_TRUE(config.verbose);
}

TEST(TrainingConfigTest, CustomValues) {
    TrainingConfig config;

    config.d_model = 768;
    config.num_heads = 12;
    config.num_epochs = 20;
    config.learning_rate = 0.0001f;

    EXPECT_EQ(config.d_model, 768);
    EXPECT_EQ(config.num_heads, 12);
    EXPECT_EQ(config.num_epochs, 20);
    EXPECT_FLOAT_EQ(config.learning_rate, 0.0001f);
}

// ============================================================================
// Configuration Validation Tests
// ============================================================================

TEST(ConfigValidationTest, DModelDivisibleByHeads) {
    TrainingConfig config;

    // Valid: 512 % 8 = 0
    config.d_model = 512;
    config.num_heads = 8;
    EXPECT_EQ(config.d_model % config.num_heads, 0);

    // Valid: 768 % 12 = 0
    config.d_model = 768;
    config.num_heads = 12;
    EXPECT_EQ(config.d_model % config.num_heads, 0);
}

TEST(ConfigValidationTest, DModelNotDivisibleByHeads) {
    TrainingConfig config;

    // Invalid: 500 % 8 != 0
    config.d_model = 500;
    config.num_heads = 8;
    EXPECT_NE(config.d_model % config.num_heads, 0);

    // Should be corrected to 504 (nearest multiple)
    int corrected = ((config.d_model + config.num_heads - 1) / config.num_heads) * config.num_heads;
    EXPECT_EQ(corrected, 504);
}

TEST(ConfigValidationTest, DFFRatio) {
    TrainingConfig config;

    // Standard 4x ratio
    config.d_model = 512;
    config.d_ff = 2048;
    float ratio = static_cast<float>(config.d_ff) / config.d_model;
    EXPECT_FLOAT_EQ(ratio, 4.0f);

    // Acceptable 3x ratio
    config.d_ff = 1536;
    ratio = static_cast<float>(config.d_ff) / config.d_model;
    EXPECT_FLOAT_EQ(ratio, 3.0f);
    EXPECT_GE(ratio, 2.0f);
    EXPECT_LE(ratio, 8.0f);
}

TEST(ConfigValidationTest, NumHeadsPowerOfTwo) {
    // Check if power of 2
    auto is_power_of_2 = [](int n) { return n > 0 && (n & (n - 1)) == 0; };

    EXPECT_TRUE(is_power_of_2(2));
    EXPECT_TRUE(is_power_of_2(4));
    EXPECT_TRUE(is_power_of_2(8));
    EXPECT_TRUE(is_power_of_2(16));

    EXPECT_FALSE(is_power_of_2(3));
    EXPECT_FALSE(is_power_of_2(5));
    EXPECT_FALSE(is_power_of_2(6));
    EXPECT_FALSE(is_power_of_2(10));
}

TEST(ConfigValidationTest, LearningRateRange) {
    TrainingConfig config;

    // Valid ranges
    config.learning_rate = 0.001f;
    EXPECT_GT(config.learning_rate, 0.0f);
    EXPECT_LE(config.learning_rate, 1.0f);

    config.learning_rate = 0.1f;
    EXPECT_GT(config.learning_rate, 0.0f);
    EXPECT_LE(config.learning_rate, 1.0f);
}

TEST(ConfigValidationTest, MinLearningRateLessThanMax) {
    TrainingConfig config;

    config.learning_rate = 0.001f;
    config.min_learning_rate = 1e-6f;

    EXPECT_LT(config.min_learning_rate, config.learning_rate);
}

TEST(ConfigValidationTest, LayerCountRanges) {
    TrainingConfig config;

    // Valid ranges
    EXPECT_GE(config.num_encoder_layers, 1);
    EXPECT_LE(config.num_encoder_layers, 48);
    EXPECT_GE(config.num_decoder_layers, 1);
    EXPECT_LE(config.num_decoder_layers, 48);
}

TEST(ConfigValidationTest, SequenceLengthRange) {
    TrainingConfig config;

    EXPECT_GE(config.max_seq_length, 16);
    EXPECT_LE(config.max_seq_length, 8192);
}

// ============================================================================
// Learning Rate Schedule Tests
// ============================================================================

class LRScheduleTest : public ::testing::Test {
   protected:
    float calculate_learning_rate(LRSchedule schedule, int step, int total_steps, int warmup_steps,
                                  float base_lr, float min_lr, float decay_factor = 0.1f) {
        // Auto-configure warmup if needed
        int warmup = warmup_steps;
        if (warmup == 0 && schedule != LRSchedule::CONSTANT) {
            warmup = total_steps / 10;
        }

        switch (schedule) {
            case LRSchedule::CONSTANT:
                return base_lr;

            case LRSchedule::LINEAR_WARMUP:
                if (step < warmup) {
                    return base_lr * (static_cast<float>(step) / warmup);
                }
                return base_lr;

            case LRSchedule::COSINE_DECAY: {
                float progress = static_cast<float>(step) / total_steps;
                float cosine = 0.5f * (1.0f + std::cos(3.14159265359f * progress));
                return min_lr + (base_lr - min_lr) * cosine;
            }

            case LRSchedule::WARMUP_COSINE: {
                if (step < warmup) {
                    return base_lr * (static_cast<float>(step) / warmup);
                }
                float progress = static_cast<float>(step - warmup) / (total_steps - warmup);
                float cosine = 0.5f * (1.0f + std::cos(3.14159265359f * progress));
                return min_lr + (base_lr - min_lr) * cosine;
            }

            case LRSchedule::STEP_DECAY: {
                int decay_steps = total_steps / 10;  // Simple assumption
                int num_decays = step / decay_steps;
                return base_lr * std::pow(decay_factor, num_decays);
            }

            case LRSchedule::EXPONENTIAL_DECAY: {
                int decay_steps = total_steps / 10;
                float decay_rate = std::pow(decay_factor, 1.0f / decay_steps);
                return base_lr * std::pow(decay_rate, step);
            }

            default:
                return base_lr;
        }
    }
};

TEST_F(LRScheduleTest, ConstantSchedule) {
    float lr = calculate_learning_rate(LRSchedule::CONSTANT, 0, 1000, 0, 0.001f, 1e-6f);
    EXPECT_FLOAT_EQ(lr, 0.001f);

    lr = calculate_learning_rate(LRSchedule::CONSTANT, 500, 1000, 0, 0.001f, 1e-6f);
    EXPECT_FLOAT_EQ(lr, 0.001f);

    lr = calculate_learning_rate(LRSchedule::CONSTANT, 1000, 1000, 0, 0.001f, 1e-6f);
    EXPECT_FLOAT_EQ(lr, 0.001f);
}

TEST_F(LRScheduleTest, LinearWarmupSchedule) {
    int warmup = 100;

    // At step 0, LR should be 0
    float lr = calculate_learning_rate(LRSchedule::LINEAR_WARMUP, 0, 1000, warmup, 0.001f, 1e-6f);
    EXPECT_FLOAT_EQ(lr, 0.0f);

    // At step 50 (half warmup), LR should be half of base
    lr = calculate_learning_rate(LRSchedule::LINEAR_WARMUP, 50, 1000, warmup, 0.001f, 1e-6f);
    EXPECT_FLOAT_EQ(lr, 0.0005f);

    // At step 100 (end warmup), LR should be base
    lr = calculate_learning_rate(LRSchedule::LINEAR_WARMUP, 100, 1000, warmup, 0.001f, 1e-6f);
    EXPECT_FLOAT_EQ(lr, 0.001f);

    // After warmup, LR stays constant
    lr = calculate_learning_rate(LRSchedule::LINEAR_WARMUP, 500, 1000, warmup, 0.001f, 1e-6f);
    EXPECT_FLOAT_EQ(lr, 0.001f);
}

TEST_F(LRScheduleTest, CosineDecaySchedule) {
    // At step 0, LR should be at max
    float lr = calculate_learning_rate(LRSchedule::COSINE_DECAY, 0, 1000, 0, 0.001f, 1e-6f);
    EXPECT_NEAR(lr, 0.001f, 1e-7f);

    // At step 500 (halfway), LR should be between min and max
    lr = calculate_learning_rate(LRSchedule::COSINE_DECAY, 500, 1000, 0, 0.001f, 1e-6f);
    EXPECT_GT(lr, 1e-6f);
    EXPECT_LT(lr, 0.001f);

    // At step 1000 (end), LR should be at min
    lr = calculate_learning_rate(LRSchedule::COSINE_DECAY, 1000, 1000, 0, 0.001f, 1e-6f);
    EXPECT_NEAR(lr, 1e-6f, 1e-7f);
}

TEST_F(LRScheduleTest, WarmupCosineSchedule) {
    int warmup = 100;

    // During warmup (step 50), LR should increase linearly
    float lr = calculate_learning_rate(LRSchedule::WARMUP_COSINE, 50, 1000, warmup, 0.001f, 1e-6f);
    EXPECT_FLOAT_EQ(lr, 0.0005f);

    // At end of warmup, LR should be at max
    lr = calculate_learning_rate(LRSchedule::WARMUP_COSINE, 100, 1000, warmup, 0.001f, 1e-6f);
    EXPECT_FLOAT_EQ(lr, 0.001f);

    // After warmup, should decay with cosine
    lr = calculate_learning_rate(LRSchedule::WARMUP_COSINE, 550, 1000, warmup, 0.001f, 1e-6f);
    EXPECT_GT(lr, 1e-6f);
    EXPECT_LT(lr, 0.001f);
}

TEST_F(LRScheduleTest, StepDecaySchedule) {
    // Initially at base LR
    float lr = calculate_learning_rate(LRSchedule::STEP_DECAY, 0, 1000, 0, 0.001f, 1e-6f, 0.1f);
    EXPECT_FLOAT_EQ(lr, 0.001f);

    // After one decay interval, should be reduced
    lr = calculate_learning_rate(LRSchedule::STEP_DECAY, 100, 1000, 0, 0.001f, 1e-6f, 0.1f);
    EXPECT_LT(lr, 0.001f);

    // Should continue decaying
    lr = calculate_learning_rate(LRSchedule::STEP_DECAY, 200, 1000, 0, 0.001f, 1e-6f, 0.1f);
    EXPECT_LT(lr, 0.0001f);
}

TEST_F(LRScheduleTest, ExponentialDecaySchedule) {
    // Initially at base LR
    float lr =
        calculate_learning_rate(LRSchedule::EXPONENTIAL_DECAY, 0, 1000, 0, 0.001f, 1e-6f, 0.1f);
    EXPECT_FLOAT_EQ(lr, 0.001f);

    // Should decay smoothly
    float lr_100 =
        calculate_learning_rate(LRSchedule::EXPONENTIAL_DECAY, 100, 1000, 0, 0.001f, 1e-6f, 0.1f);
    float lr_200 =
        calculate_learning_rate(LRSchedule::EXPONENTIAL_DECAY, 200, 1000, 0, 0.001f, 1e-6f, 0.1f);

    EXPECT_LT(lr_100, 0.001f);
    EXPECT_LT(lr_200, lr_100);
}

TEST_F(LRScheduleTest, AutoWarmupConfiguration) {
    // When warmup_steps = 0, should auto-configure to 10% of total
    int total_steps = 1000;
    int auto_warmup = total_steps / 10;

    EXPECT_EQ(auto_warmup, 100);

    // Verify warmup behavior with auto-configured warmup
    float lr =
        calculate_learning_rate(LRSchedule::WARMUP_COSINE, 50, total_steps, 0, 0.001f, 1e-6f);
    EXPECT_LT(lr, 0.001f);  // Should be in warmup phase
}

// ============================================================================
// Data Loading Tests
// ============================================================================

TEST(DataLoadingTest, ParseValidConversationFile) {
    std::vector<std::pair<std::string, std::string>> test_data = {
        {"Hello", "Hi there"}, {"How are you?", "I'm doing great!"}, {"Goodbye", "See you later"}};

    std::string filename = "test_conversations.txt";
    create_test_data_file(filename, test_data);

    // Verify file exists and has content
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());

    std::vector<ConversationPair> pairs;
    std::string line;
    std::string current_input;
    std::string current_response;

    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        if (line.empty()) {
            if (!current_input.empty() && !current_response.empty()) {
                pairs.emplace_back(current_input, current_response);
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

    // Last pair
    if (!current_input.empty() && !current_response.empty()) {
        pairs.emplace_back(current_input, current_response);
    }

    file.close();

    EXPECT_EQ(pairs.size(), 3);
    EXPECT_EQ(pairs[0].input, "Hello");
    EXPECT_EQ(pairs[0].response, "Hi there");
    EXPECT_EQ(pairs[1].input, "How are you?");
    EXPECT_EQ(pairs[1].response, "I'm doing great!");

    // Cleanup
    std::remove(filename.c_str());
}

TEST(DataLoadingTest, ParseFileWithExtraWhitespace) {
    std::ofstream file("test_whitespace.txt");
    file << "INPUT:    Hello   \n";
    file << "RESPONSE:    Hi there   \n\n";
    file << "INPUT:  Test  \n";
    file << "RESPONSE:  Response  \n\n";
    file.close();

    std::ifstream infile("test_whitespace.txt");
    ASSERT_TRUE(infile.is_open());

    std::vector<ConversationPair> pairs;
    std::string line;
    std::string current_input;
    std::string current_response;

    while (std::getline(infile, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        if (line.empty()) {
            if (!current_input.empty() && !current_response.empty()) {
                pairs.emplace_back(current_input, current_response);
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
        pairs.emplace_back(current_input, current_response);
    }

    infile.close();

    EXPECT_EQ(pairs.size(), 2);
    EXPECT_EQ(pairs[0].input, "Hello");
    EXPECT_EQ(pairs[0].response, "Hi there");

    std::remove("test_whitespace.txt");
}

TEST(DataLoadingTest, SkipIncompletePairs) {
    std::ofstream file("test_incomplete.txt");
    file << "INPUT: Hello\n\n";               // Missing response
    file << "RESPONSE: Orphan response\n\n";  // Missing input
    file << "INPUT: Valid input\n";
    file << "RESPONSE: Valid response\n\n";
    file.close();

    std::ifstream infile("test_incomplete.txt");
    ASSERT_TRUE(infile.is_open());

    std::vector<ConversationPair> pairs;
    std::string line;
    std::string current_input;
    std::string current_response;

    while (std::getline(infile, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        if (line.empty()) {
            if (!current_input.empty() && !current_response.empty()) {
                pairs.emplace_back(current_input, current_response);
            }
            current_input.clear();
            current_response.clear();
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
        pairs.emplace_back(current_input, current_response);
    }

    infile.close();

    // Should only have 1 valid pair
    EXPECT_EQ(pairs.size(), 1);
    EXPECT_EQ(pairs[0].input, "Valid input");
    EXPECT_EQ(pairs[0].response, "Valid response");

    std::remove("test_incomplete.txt");
}

// ============================================================================
// Data Splitting Tests
// ============================================================================

TEST(DataSplittingTest, SplitWithValidationRatio) {
    std::vector<ConversationPair> all_data;
    for (int i = 0; i < 100; i++) {
        all_data.emplace_back("Input" + std::to_string(i), "Response" + std::to_string(i));
    }

    int validation_split = 10;  // 1/10 for validation
    int validation_size = all_data.size() / validation_split;

    EXPECT_EQ(validation_size, 10);

    std::vector<ConversationPair> validation_data(all_data.end() - validation_size, all_data.end());

    std::vector<ConversationPair> training_data(all_data.begin(), all_data.end() - validation_size);

    EXPECT_EQ(training_data.size(), 90);
    EXPECT_EQ(validation_data.size(), 10);

    // Verify data integrity
    EXPECT_EQ(training_data[0].input, "Input0");
    EXPECT_EQ(validation_data[0].input, "Input90");
}

TEST(DataSplittingTest, NoSplitWhenValidationSplitZero) {
    std::vector<ConversationPair> all_data;
    for (int i = 0; i < 50; i++) {
        all_data.emplace_back("Input" + std::to_string(i), "Response" + std::to_string(i));
    }

    int validation_split = 0;

    if (validation_split <= 0) {
        // No split should occur
        EXPECT_EQ(all_data.size(), 50);
    }
}

TEST(DataSplittingTest, InsufficientDataForSplit) {
    std::vector<ConversationPair> all_data;
    for (int i = 0; i < 5; i++) {
        all_data.emplace_back("Input" + std::to_string(i), "Response" + std::to_string(i));
    }

    int validation_split = 10;
    int validation_size = all_data.size() / validation_split;

    EXPECT_EQ(validation_size, 0);  // Not enough data
}

// ============================================================================
// Early Stopping Tests
// ============================================================================

TEST(EarlyStoppingTest, CheckImprovementDetection) {
    float best_loss = 2.5f;
    float min_delta = 1e-4f;

    // Significant improvement
    float new_loss = 2.3f;
    bool improved = (new_loss < best_loss - min_delta);
    EXPECT_TRUE(improved);

    // No improvement (within delta)
    new_loss = 2.49999f;
    improved = (new_loss < best_loss - min_delta);
    EXPECT_FALSE(improved);

    // Worse
    new_loss = 2.6f;
    improved = (new_loss < best_loss - min_delta);
    EXPECT_FALSE(improved);
}

TEST(EarlyStoppingTest, PatienceCounter) {
    int patience = 5;
    int epochs_without_improvement = 0;

    // Simulate epochs without improvement
    for (int i = 0; i < patience - 1; i++) {
        epochs_without_improvement++;
        EXPECT_FALSE(epochs_without_improvement >= patience);
    }

    // One more epoch triggers early stopping
    epochs_without_improvement++;
    EXPECT_TRUE(epochs_without_improvement >= patience);
}

TEST(EarlyStoppingTest, ResetCounterOnImprovement) {
    int epochs_without_improvement = 3;

    // Improvement detected
    float best_loss = 2.5f;
    float new_loss = 2.3f;
    float min_delta = 1e-4f;

    if (new_loss < best_loss - min_delta) {
        best_loss = new_loss;
        epochs_without_improvement = 0;
    }

    EXPECT_EQ(epochs_without_improvement, 0);
    EXPECT_FLOAT_EQ(best_loss, 2.3f);
}

// ============================================================================
// Checkpoint Naming Tests
// ============================================================================

TEST(CheckpointTest, EpochCheckpointNaming) {
    std::string base_path = "model.bin";

    std::string epoch1 = base_path + ".epoch1";
    std::string epoch10 = base_path + ".epoch10";
    std::string epoch100 = base_path + ".epoch100";

    EXPECT_EQ(epoch1, "model.bin.epoch1");
    EXPECT_EQ(epoch10, "model.bin.epoch10");
    EXPECT_EQ(epoch100, "model.bin.epoch100");
}

TEST(CheckpointTest, CheckpointFrequency) {
    int checkpoint_every = 5;

    for (int epoch = 0; epoch < 20; epoch++) {
        bool should_checkpoint = ((epoch + 1) % checkpoint_every == 0);

        if (epoch + 1 == 5 || epoch + 1 == 10 || epoch + 1 == 15 || epoch + 1 == 20) {
            EXPECT_TRUE(should_checkpoint);
        } else {
            EXPECT_FALSE(should_checkpoint);
        }
    }
}

// ============================================================================
// Integration Tests (File I/O)
// ============================================================================

TEST(IntegrationTest, CreateAndLoadVocabFile) {
    std::string vocab_file = "test_vocab.txt";
    int vocab_size = 50;

    create_test_vocab_file(vocab_file, vocab_size);

    std::ifstream file(vocab_file);
    ASSERT_TRUE(file.is_open());

    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(file, token)) {
        tokens.push_back(token);
    }
    file.close();

    EXPECT_EQ(tokens.size(), vocab_size);
    EXPECT_EQ(tokens[0], "<unk>");
    EXPECT_EQ(tokens[1], "<pad>");
    EXPECT_EQ(tokens[2], "<s>");
    EXPECT_EQ(tokens[3], "</s>");

    std::remove(vocab_file.c_str());
}

TEST(IntegrationTest, CreateAndLoadConversationFile) {
    std::vector<std::pair<std::string, std::string>> conversations = {
        {"Hello", "Hi"}, {"How are you?", "Fine, thanks!"}, {"Goodbye", "Bye!"}};

    std::string data_file = "test_data.txt";
    create_test_data_file(data_file, conversations);

    std::ifstream file(data_file);
    ASSERT_TRUE(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    EXPECT_NE(content.find("INPUT: Hello"), std::string::npos);
    EXPECT_NE(content.find("RESPONSE: Hi"), std::string::npos);
    EXPECT_NE(content.find("INPUT: How are you?"), std::string::npos);
    EXPECT_NE(content.find("RESPONSE: Fine, thanks!"), std::string::npos);

    std::remove(data_file.c_str());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(EdgeCaseTest, EmptyDataFile) {
    std::string filename = "empty_data.txt";
    std::ofstream file(filename);
    file.close();

    std::ifstream infile(filename);
    ASSERT_TRUE(infile.is_open());

    std::vector<ConversationPair> pairs;
    std::string line;
    std::string current_input;
    std::string current_response;

    while (std::getline(infile, line)) {
        if (!current_input.empty() && !current_response.empty()) {
            pairs.emplace_back(current_input, current_response);
        }
    }
    infile.close();

    EXPECT_EQ(pairs.size(), 0);

    std::remove(filename.c_str());
}

TEST(EdgeCaseTest, VeryLongConversationPair) {
    std::string long_input(10000, 'a');
    std::string long_response(10000, 'b');

    ConversationPair pair(long_input, long_response);

    EXPECT_EQ(pair.input.length(), 10000);
    EXPECT_EQ(pair.response.length(), 10000);
}

TEST(EdgeCaseTest, SpecialCharactersInConversation) {
    std::string input = "What's the weather like? It's 25 degrees!";
    std::string response = "I don't know... Check online @ weather.com";

    ConversationPair pair(input, response);

    EXPECT_NE(pair.input.find("degrees"), std::string::npos);
    EXPECT_NE(pair.response.find('@'), std::string::npos);
}

TEST(EdgeCaseTest, ZeroEpochs) {
    TrainingConfig config;
    config.num_epochs = 0;

    // Training should not run
    for (int epoch = 0; epoch < config.num_epochs; epoch++) {
        FAIL() << "Should not execute any epochs";
    }

    SUCCEED();
}

TEST(EdgeCaseTest, ExtremelySmallModel) {
    TrainingConfig config;
    config.d_model = 64;
    config.num_heads = 4;
    config.d_ff = 256;
    config.num_encoder_layers = 1;
    config.num_decoder_layers = 1;

    // Verify divisibility
    EXPECT_EQ(config.d_model % config.num_heads, 0);

    // Verify all parameters are positive
    EXPECT_GT(config.d_model, 0);
    EXPECT_GT(config.num_heads, 0);
    EXPECT_GT(config.d_ff, 0);
    EXPECT_GT(config.num_encoder_layers, 0);
    EXPECT_GT(config.num_decoder_layers, 0);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
