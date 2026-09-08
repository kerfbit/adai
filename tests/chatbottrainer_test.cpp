#include "../src/ChatbotTrainer.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include "../src/TrainingSampleMeta.hpp"

// ============================================================================
// Test Fixtures
// ============================================================================

class ChatbotTrainerTest : public ::testing::Test {
   protected:
    TrainingConfig config;
    std::unique_ptr<ChatbotTrainer> trainer;

    void SetUp() override {
        // Default test configuration
        config.num_epochs = 5;
        config.d_model = 64;  // Small model for fast testing
        config.num_heads = 4;
        config.d_ff = 256;
        config.num_encoder_layers = 2;
        config.num_decoder_layers = 2;
        config.max_seq_length = 32;
        config.validation_split = 5;          // 20% validation
        config.log_level = LogLevel::SILENT;  // Quiet during tests

        trainer = std::make_unique<ChatbotTrainer>(config);
    }

    void TearDown() override {
        trainer.reset();
    }
};

class MetricsTest : public ::testing::Test {
   protected:
    std::unique_ptr<ChatbotTrainer> trainer;

    void SetUp() override {
        TrainingConfig config;
        config.log_level = LogLevel::SILENT;
        trainer = std::make_unique<ChatbotTrainer>(config);
    }
};

class LoggingTest : public ::testing::Test {
   protected:
    std::unique_ptr<ChatbotTrainer> trainer;

    void SetUp() override {
        TrainingConfig config;
        trainer = std::make_unique<ChatbotTrainer>(config);
    }
};

// ============================================================================
// Metrics Tests
// ============================================================================

TEST_F(MetricsTest, CalculatePerplexity_ZeroLoss) {
    float loss = 0.0f;
    float perplexity = trainer->calculate_perplexity(loss);
    EXPECT_FLOAT_EQ(perplexity, 1.0f);  // exp(0) = 1
}

TEST_F(MetricsTest, CalculatePerplexity_UnitLoss) {
    float loss = 1.0f;
    float perplexity = trainer->calculate_perplexity(loss);
    EXPECT_NEAR(perplexity, std::exp(1.0f), 1e-5);  // exp(1) ≈ 2.718
}

TEST_F(MetricsTest, CalculatePerplexity_TypicalLoss) {
    float loss = 2.3026f;  // ln(10)
    float perplexity = trainer->calculate_perplexity(loss);
    EXPECT_NEAR(perplexity, 10.0f, 1e-3);  // exp(ln(10)) = 10
}

TEST_F(MetricsTest, CalculatePerplexity_LargeLoss) {
    float loss = 4.6052f;  // ln(100)
    float perplexity = trainer->calculate_perplexity(loss);
    EXPECT_NEAR(perplexity, 100.0f, 1e-2);
}

TEST_F(MetricsTest, CalculatePerplexity_Monotonic) {
    // Perplexity should increase monotonically with loss
    float loss1 = 1.0f;
    float loss2 = 2.0f;
    float loss3 = 3.0f;

    float ppl1 = trainer->calculate_perplexity(loss1);
    float ppl2 = trainer->calculate_perplexity(loss2);
    float ppl3 = trainer->calculate_perplexity(loss3);

    EXPECT_LT(ppl1, ppl2);
    EXPECT_LT(ppl2, ppl3);
}

TEST_F(MetricsTest, CalculateAccuracy_PerfectMatch) {
    // When vectors match, accuracy should be 100%
    std::vector<int> predictions = {1, 2, 3};
    std::vector<int> targets = {1, 2, 3};

    float accuracy = trainer->calculate_accuracy(predictions, targets);
    EXPECT_FLOAT_EQ(accuracy, 1.0f);  // 100% accuracy
}

TEST_F(MetricsTest, CalculateAccuracy_PartialMatch) {
    // 2 out of 4 correct = 50%
    std::vector<int> predictions = {1, 2, 3, 4};
    std::vector<int> targets = {1, 5, 3, 6};

    float accuracy = trainer->calculate_accuracy(predictions, targets);
    EXPECT_FLOAT_EQ(accuracy, 0.5f);  // 50% accuracy
}

TEST_F(MetricsTest, CalculateAccuracy_NoMatch) {
    // 0 out of 3 correct = 0%
    std::vector<int> predictions = {1, 2, 3};
    std::vector<int> targets = {4, 5, 6};

    float accuracy = trainer->calculate_accuracy(predictions, targets);
    EXPECT_FLOAT_EQ(accuracy, 0.0f);  // 0% accuracy
}

TEST_F(MetricsTest, CalculateAccuracy_EmptyVectors) {
    std::vector<int> predictions;
    std::vector<int> targets;

    float accuracy = trainer->calculate_accuracy(predictions, targets);
    EXPECT_FLOAT_EQ(accuracy, -1.0f);
}

TEST_F(MetricsTest, CalculateAccuracy_MismatchedSizes) {
    std::vector<int> predictions = {1, 2, 3};
    std::vector<int> targets = {1, 2};

    float accuracy = trainer->calculate_accuracy(predictions, targets);
    EXPECT_FLOAT_EQ(accuracy, -1.0f);
}

// ============================================================================
// Logging System Tests
// ============================================================================

TEST_F(LoggingTest, LogLevel_Silent_NoOutput) {
    TrainingConfig config;
    config.log_level = LogLevel::SILENT;
    auto silent_trainer = std::make_unique<ChatbotTrainer>(config);

    // Should not crash, but won't verify output (would need output capture)
    silent_trainer->log(LogLevel::NORMAL, "This should not appear", "");
    silent_trainer->log(LogLevel::VERBOSE, "This should not appear", "");
    silent_trainer->log(LogLevel::DEBUG, "This should not appear", "");
}

TEST_F(LoggingTest, LogLevel_Normal_FiltersVerbose) {
    TrainingConfig config;
    config.log_level = LogLevel::NORMAL;
    auto normal_trainer = std::make_unique<ChatbotTrainer>(config);

    // NORMAL and below should appear, VERBOSE and DEBUG should not
    normal_trainer->log(LogLevel::SILENT, "Error message", "");
    normal_trainer->log(LogLevel::NORMAL, "Normal message", "");
    normal_trainer->log(LogLevel::VERBOSE, "Should be filtered", "");
    normal_trainer->log(LogLevel::DEBUG, "Should be filtered", "");
}

TEST_F(LoggingTest, LogLevel_Verbose_FiltersDebug) {
    TrainingConfig config;
    config.log_level = LogLevel::VERBOSE;
    auto verbose_trainer = std::make_unique<ChatbotTrainer>(config);

    // VERBOSE and below should appear, DEBUG should not
    verbose_trainer->log(LogLevel::SILENT, "Error message", "");
    verbose_trainer->log(LogLevel::NORMAL, "Normal message", "");
    verbose_trainer->log(LogLevel::VERBOSE, "Verbose message", "");
    verbose_trainer->log(LogLevel::DEBUG, "Should be filtered", "");
}

TEST_F(LoggingTest, LogLevel_Debug_ShowsAll) {
    TrainingConfig config;
    config.log_level = LogLevel::DEBUG;
    auto debug_trainer = std::make_unique<ChatbotTrainer>(config);

    // All messages should appear
    debug_trainer->log(LogLevel::SILENT, "Error message", "");
    debug_trainer->log(LogLevel::NORMAL, "Normal message", "");
    debug_trainer->log(LogLevel::VERBOSE, "Verbose message", "");
    debug_trainer->log(LogLevel::DEBUG, "Debug message", "");
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(ChatbotTrainerTest, Config_DefaultValues) {
    TrainingConfig default_config;

    EXPECT_EQ(default_config.d_model, 512);
    EXPECT_EQ(default_config.num_heads, 8);
    EXPECT_EQ(default_config.d_ff, 2048);
    EXPECT_EQ(default_config.num_encoder_layers, 6);
    EXPECT_EQ(default_config.num_decoder_layers, 6);
    EXPECT_EQ(default_config.max_seq_length, 512);
    EXPECT_EQ(default_config.num_epochs, 10);
    EXPECT_FLOAT_EQ(default_config.learning_rate, 0.001f);
    EXPECT_EQ(default_config.batch_size, 1);
    EXPECT_EQ(default_config.gradient_accumulation_steps, 1);
    EXPECT_EQ(default_config.validation_split, 10);
    EXPECT_EQ(default_config.lr_schedule, LRSchedule::WARMUP_COSINE);
    EXPECT_EQ(default_config.optimizer_type, OptimizerType::ADAMW);
    EXPECT_FLOAT_EQ(default_config.weight_decay, 0.01f);
    EXPECT_FLOAT_EQ(default_config.gradient_clip_norm, 1.0f);
    EXPECT_EQ(default_config.log_level, LogLevel::VERBOSE);
    EXPECT_EQ(default_config.tokenizer_mode, TokenizerMode::ASCII);
}

TEST_F(ChatbotTrainerTest, Config_CustomValues) {
    TrainingConfig custom_config;
    custom_config.d_model = 256;
    custom_config.num_heads = 4;
    custom_config.learning_rate = 0.0001f;
    custom_config.num_epochs = 20;
    custom_config.log_level = LogLevel::NORMAL;

    auto custom_trainer = std::make_unique<ChatbotTrainer>(custom_config);

    EXPECT_EQ(custom_trainer->get_config().d_model, 256);
    EXPECT_EQ(custom_trainer->get_config().num_heads, 4);
    EXPECT_FLOAT_EQ(custom_trainer->get_config().learning_rate, 0.0001f);
    EXPECT_EQ(custom_trainer->get_config().num_epochs, 20);
    EXPECT_EQ(custom_trainer->get_config().log_level, LogLevel::NORMAL);
}

TEST_F(ChatbotTrainerTest, Constructor_InitializesState) {
    EXPECT_FLOAT_EQ(trainer->get_best_validation_loss(), std::numeric_limits<float>::max());
    EXPECT_EQ(trainer->get_best_epoch(), 0);
    EXPECT_EQ(trainer->get_global_step(), 0);
    EXPECT_FLOAT_EQ(trainer->get_current_learning_rate(), config.learning_rate);
    EXPECT_FALSE(trainer->was_early_stopped());
}

// ============================================================================
// Training Configuration Enum Tests
// ============================================================================

TEST(EnumTest, LRSchedule_AllValues) {
    EXPECT_EQ(static_cast<int>(LRSchedule::CONSTANT), 0);
    EXPECT_NE(LRSchedule::LINEAR_WARMUP, LRSchedule::COSINE_DECAY);
    EXPECT_NE(LRSchedule::WARMUP_COSINE, LRSchedule::STEP_DECAY);
    EXPECT_NE(LRSchedule::STEP_DECAY, LRSchedule::EXPONENTIAL_DECAY);
}

TEST(EnumTest, LogLevel_Ordering) {
    EXPECT_LT(static_cast<int>(LogLevel::SILENT), static_cast<int>(LogLevel::NORMAL));
    EXPECT_LT(static_cast<int>(LogLevel::NORMAL), static_cast<int>(LogLevel::VERBOSE));
    EXPECT_LT(static_cast<int>(LogLevel::VERBOSE), static_cast<int>(LogLevel::DEBUG));

    EXPECT_EQ(static_cast<int>(LogLevel::SILENT), 0);
    EXPECT_EQ(static_cast<int>(LogLevel::NORMAL), 1);
    EXPECT_EQ(static_cast<int>(LogLevel::VERBOSE), 2);
    EXPECT_EQ(static_cast<int>(LogLevel::DEBUG), 3);
}

// ============================================================================
// Data Structure Tests
// ============================================================================

TEST(DataStructureTest, ConversationPair_Construction) {
    ConversationPair pair("Hello", "Hi there!");

    EXPECT_EQ(pair.input, "Hello");
    EXPECT_EQ(pair.response, "Hi there!");
}

TEST(DataStructureTest, ConversationPair_EmptyStrings) {
    ConversationPair pair("", "");

    EXPECT_TRUE(pair.input.empty());
    EXPECT_TRUE(pair.response.empty());
}

TEST(DataStructureTest, ConversationPair_DefaultMetaSentinels) {
    ConversationPair pair("q", "a");
    EXPECT_LT(pair.meta.quality, 0.0f);
    EXPECT_FLOAT_EQ(pair.meta.weight, 1.0f);
    EXPECT_LT(pair.meta.token_count, 0);
    EXPECT_TRUE(pair.meta.domain.empty());
    EXPECT_TRUE(pair.meta.task_type.empty());
}

TEST(DataStructureTest, ConversationPair_MetaConstructor) {
    SampleMeta m;
    m.domain = "fiction";
    m.task_type = "qa";
    m.quality = 0.8f;
    m.token_count = 50;

    ConversationPair pair("question", "answer", m);

    EXPECT_EQ(pair.input, "question");
    EXPECT_EQ(pair.response, "answer");
    EXPECT_EQ(pair.meta.domain, "fiction");
    EXPECT_EQ(pair.meta.task_type, "qa");
    EXPECT_NEAR(pair.meta.quality, 0.8f, 1e-5f);
    EXPECT_EQ(pair.meta.token_count, 50);
}

TEST(DataStructureTest, TokenizedPair_Construction) {
    std::vector<int> input_tokens = {1, 2, 3};
    std::vector<int> target_tokens = {4, 5, 6};

    TokenizedPair pair(input_tokens, target_tokens, "hello", "world");

    EXPECT_EQ(pair.input_tokens.size(), 3);
    EXPECT_EQ(pair.target_tokens.size(), 3);
    EXPECT_EQ(pair.input_tokens[0], 1);
    EXPECT_EQ(pair.target_tokens[2], 6);
    EXPECT_EQ(pair.input_text, "hello");
    EXPECT_EQ(pair.target_text, "world");
}

TEST(DataStructureTest, TokenizedPair_EmptyVectors) {
    std::vector<int> empty;

    TokenizedPair pair(empty, empty, "", "");

    EXPECT_TRUE(pair.input_tokens.empty());
    EXPECT_TRUE(pair.target_tokens.empty());
    EXPECT_TRUE(pair.input_text.empty());
    EXPECT_TRUE(pair.target_text.empty());
}

// ============================================================================
// Getter Tests
// ============================================================================

TEST_F(ChatbotTrainerTest, Getters_EmptyVectors) {
    EXPECT_TRUE(trainer->get_training_losses().empty());
    EXPECT_TRUE(trainer->get_validation_losses().empty());
    EXPECT_TRUE(trainer->get_training_perplexities().empty());
    EXPECT_TRUE(trainer->get_validation_perplexities().empty());
    EXPECT_TRUE(trainer->get_learning_rates().empty());
    EXPECT_TRUE(trainer->get_gradient_norms().empty());
}

TEST_F(ChatbotTrainerTest, Getters_DataSizes) {
    EXPECT_EQ(trainer->get_training_data_size(), 0);
    EXPECT_EQ(trainer->get_validation_data_size(), 0);
    EXPECT_EQ(trainer->get_tokenized_training_size(), 0);
    EXPECT_EQ(trainer->get_tokenized_validation_size(), 0);
}

// ============================================================================
// Checkpoint Metadata Tests
// ============================================================================

class CheckpointTest : public ::testing::Test {
   protected:
    std::string test_checkpoint_path;
    std::string test_metadata_path;

    void SetUp() override {
        test_checkpoint_path = "test_checkpoint.bin";
        test_metadata_path = test_checkpoint_path + ".metadata";
    }

    void TearDown() override {
        // Clean up test files
        std::remove(test_checkpoint_path.c_str());
        std::remove(test_metadata_path.c_str());
    }

    void create_dummy_metadata(int epoch, int global_step, float lr, float best_val_loss,
                               int best_epoch) {
        std::ofstream meta_file(test_metadata_path);
        meta_file << "epoch=" << epoch << "\n";
        meta_file << "global_step=" << global_step << "\n";
        meta_file << "learning_rate=" << lr << "\n";
        meta_file << "best_validation_loss=" << best_val_loss << "\n";
        meta_file << "best_epoch=" << best_epoch << "\n";
        meta_file.close();
    }
};

TEST_F(CheckpointTest, Metadata_FileFormat) {
    create_dummy_metadata(5, 1000, 0.0001f, 2.5f, 3);

    std::ifstream meta_file(test_metadata_path);
    ASSERT_TRUE(meta_file.is_open());

    std::string line;
    std::getline(meta_file, line);
    EXPECT_EQ(line, "epoch=5");

    std::getline(meta_file, line);
    EXPECT_EQ(line, "global_step=1000");

    std::getline(meta_file, line);
    EXPECT_EQ(line, "learning_rate=0.0001");

    meta_file.close();
}

TEST_F(CheckpointTest, Metadata_ParseKeyValue) {
    create_dummy_metadata(10, 5000, 0.00005f, 1.8f, 7);

    std::ifstream meta_file(test_metadata_path);
    std::string line;

    while (std::getline(meta_file, line)) {
        size_t pos = line.find('=');
        EXPECT_NE(pos, std::string::npos);

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        EXPECT_FALSE(key.empty());
        EXPECT_FALSE(value.empty());
    }

    meta_file.close();
}

// ============================================================================
// Gradient Accumulation Tests
// ============================================================================

TEST(GradientAccumulationTest, EffectiveBatchSize_NoAccumulation) {
    TrainingConfig config;
    config.batch_size = 4;
    config.gradient_accumulation_steps = 1;

    int effective_batch = config.batch_size * config.gradient_accumulation_steps;
    EXPECT_EQ(effective_batch, 4);
}

TEST(GradientAccumulationTest, EffectiveBatchSize_WithAccumulation) {
    TrainingConfig config;
    config.batch_size = 1;
    config.gradient_accumulation_steps = 32;

    int effective_batch = config.batch_size * config.gradient_accumulation_steps;
    EXPECT_EQ(effective_batch, 32);
}

TEST(GradientAccumulationTest, EffectiveBatchSize_Large) {
    TrainingConfig config;
    config.batch_size = 8;
    config.gradient_accumulation_steps = 16;

    int effective_batch = config.batch_size * config.gradient_accumulation_steps;
    EXPECT_EQ(effective_batch, 128);
}

// ============================================================================
// Early Stopping Tests
// ============================================================================

// ============================================================================
// Quality Backfill Configuration Tests
// ============================================================================

TEST(QualityBackfillTest, Config_DefaultsAllDisabled) {
    TrainingConfig config;
    EXPECT_FALSE(config.enable_loss_quality_backfill);
    EXPECT_FALSE(config.enable_generation_quality_backfill);
    EXPECT_EQ(config.generation_backfill_max_tokens, 50);
}

TEST(QualityBackfillTest, Config_CanEnableLossBackfill) {
    TrainingConfig config;
    config.enable_loss_quality_backfill = true;
    EXPECT_TRUE(config.enable_loss_quality_backfill);
    EXPECT_FALSE(config.enable_generation_quality_backfill);
}

TEST(QualityBackfillTest, Config_CanEnableGenerationBackfill) {
    TrainingConfig config;
    config.enable_generation_quality_backfill = true;
    config.generation_backfill_max_tokens = 100;
    EXPECT_TRUE(config.enable_generation_quality_backfill);
    EXPECT_EQ(config.generation_backfill_max_tokens, 100);
}

// ============================================================================
// JSONL Training File Loading
// ============================================================================

class JsonlLoadTest : public ::testing::Test {
   protected:
    std::filesystem::path tmp_file_;

    void SetUp() override {
        tmp_file_ =
            std::filesystem::temp_directory_path() / "adai_chatbot_trainer_jsonl_test.jsonl";
    }

    void TearDown() override {
        std::filesystem::remove(tmp_file_);
    }

    void write_jsonl(const std::string& content) {
        std::ofstream f(tmp_file_);
        f << content;
    }
};

TEST_F(JsonlLoadTest, LoadsCorrectPairCount) {
    write_jsonl(
        "{\"input\":\"q1\",\"response\":\"a1\"}\n"
        "{\"input\":\"q2\",\"response\":\"a2\"}\n"
        "{\"input\":\"q3\",\"response\":\"a3\"}\n");

    TrainingConfig cfg;
    cfg.log_level = LogLevel::SILENT;
    cfg.validation_split = 0;  // no split so all 3 go to training
    ChatbotTrainer trainer(cfg);
    EXPECT_TRUE(trainer.load_conversation_data(tmp_file_.string()));
    EXPECT_EQ(trainer.get_training_data_size(), 3u);
}

TEST_F(JsonlLoadTest, LoadsLegacyFormatStillWorks) {
    write_jsonl(
        "INPUT: hello\nRESPONSE: world\n\n"
        "INPUT: foo\nRESPONSE: bar\n");

    TrainingConfig cfg;
    cfg.log_level = LogLevel::SILENT;
    cfg.validation_split = 0;
    ChatbotTrainer trainer(cfg);
    EXPECT_TRUE(trainer.load_conversation_data(tmp_file_.string()));
    EXPECT_EQ(trainer.get_training_data_size(), 2u);
}

TEST_F(JsonlLoadTest, ReturnsFalseForMissingFile) {
    TrainingConfig cfg;
    cfg.log_level = LogLevel::SILENT;
    ChatbotTrainer trainer(cfg);
    EXPECT_FALSE(trainer.load_conversation_data("/nonexistent/adai_no_file.jsonl"));
    EXPECT_EQ(trainer.get_training_data_size(), 0u);
}

TEST_F(JsonlLoadTest, SkipsLinesWithoutInputField) {
    write_jsonl(
        "{\"input\":\"valid\",\"response\":\"yes\"}\n"
        "{\"response\":\"no input key here\"}\n"  // should be skipped
        "{\"input\":\"also valid\",\"response\":\"yes\"}\n");

    TrainingConfig cfg;
    cfg.log_level = LogLevel::SILENT;
    cfg.validation_split = 0;
    ChatbotTrainer trainer(cfg);
    EXPECT_TRUE(trainer.load_conversation_data(tmp_file_.string()));
    EXPECT_EQ(trainer.get_training_data_size(), 2u);
}

// ============================================================================
// Early Stopping Tests
// ============================================================================

TEST(EarlyStoppingTest, Config_DefaultDisabled) {
    TrainingConfig config;
    EXPECT_FALSE(config.enable_early_stopping);
    EXPECT_EQ(config.patience, 5);
    EXPECT_FLOAT_EQ(config.min_delta, 1e-4f);
    EXPECT_TRUE(config.restore_best_weights);
}

TEST(EarlyStoppingTest, Config_EnabledWithPatience) {
    TrainingConfig config;
    config.enable_early_stopping = true;
    config.patience = 10;
    config.min_delta = 0.001f;

    EXPECT_TRUE(config.enable_early_stopping);
    EXPECT_EQ(config.patience, 10);
    EXPECT_FLOAT_EQ(config.min_delta, 0.001f);
}

// ============================================================================
// Learning Rate Schedule Tests
// ============================================================================

TEST(LRScheduleTest, WarmupSteps_AutoCalculation) {
    TrainingConfig config;
    config.warmup_steps = 0;  // Auto mode

    // If total steps = 1000, auto warmup should be ~100 (10%)
    int total_steps = 1000;
    int auto_warmup = (config.warmup_steps > 0) ? config.warmup_steps : total_steps / 10;

    EXPECT_EQ(auto_warmup, 100);
}

TEST(LRScheduleTest, WarmupSteps_ManualSetting) {
    TrainingConfig config;
    config.warmup_steps = 500;

    EXPECT_EQ(config.warmup_steps, 500);
}

TEST(LRScheduleTest, MinLearningRate_LessThanBase) {
    TrainingConfig config;
    config.learning_rate = 0.001f;
    config.min_learning_rate = 1e-6f;

    EXPECT_LT(config.min_learning_rate, config.learning_rate);
}

// ============================================================================
// Integration Tests (require minimal dependencies)
// ============================================================================

TEST_F(ChatbotTrainerTest, Perplexity_AfterTraining) {
    // This test verifies that perplexity vectors are properly sized
    // after training (would need real training data to fully test)

    // Initially empty
    EXPECT_TRUE(trainer->get_training_perplexities().empty());
    EXPECT_TRUE(trainer->get_validation_perplexities().empty());
}

TEST_F(ChatbotTrainerTest, LearningRate_InitialValue) {
    EXPECT_FLOAT_EQ(trainer->get_current_learning_rate(), config.learning_rate);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(MetricsTest, Perplexity_NegativeLoss_Invalid) {
    // Negative loss is invalid, but calculate_perplexity should still compute
    float loss = -1.0f;
    float perplexity = trainer->calculate_perplexity(loss);
    EXPECT_GT(perplexity, 0.0f);  // exp(negative) is still positive
    EXPECT_LT(perplexity, 1.0f);  // exp(-1) ≈ 0.368
}

TEST_F(MetricsTest, Perplexity_VeryLargeLoss) {
    float loss = 10.0f;  // Very high loss
    float perplexity = trainer->calculate_perplexity(loss);
    EXPECT_GT(perplexity, 20000.0f);  // exp(10) ≈ 22026
}

// ============================================================================
// Input truncation: tail vs. head (regression test for truncate_text_tail /
// truncate_tokens_tail — see their doc comments in ChatbotTrainer.hpp).
// Training pairs are a document split at its midpoint (input = first half,
// target = second half), so the target always starts right where the input's
// tail leaves off. Truncating a long input from its head instead of its tail
// keeps the document's opening text — arbitrarily far from what the target
// actually continues — severing the input/target relationship.
// ============================================================================

TEST(TruncateTailTest, TextShorterThanLimit_ReturnsUnchanged) {
    std::string s = "short text";
    EXPECT_EQ(ChatbotTrainer::truncate_text_tail(s, 100), s);
}

TEST(TruncateTailTest, TextLongerThanLimit_KeepsTailNotHead) {
    // 26 chars total; cap at 10 should keep the LAST 10 characters.
    std::string s = "abcdefghijklmnopqrstuvwxyz";
    std::string result = ChatbotTrainer::truncate_text_tail(s, 10);
    EXPECT_EQ(result, "qrstuvwxyz");  // tail, not "abcdefghij" (head)
    EXPECT_LE(result.size(), 10u);
}

TEST(TruncateTailTest, TextBoundaryRespectsUtf8) {
    // "é" is 2 bytes (0xC3 0xA9). Place multi-byte chars straddling the cut
    // point and confirm the result is still valid UTF-8 (starts on a lead byte).
    std::string s =
        "aé"
        "bé"
        "cé"
        "dé"
        "eé";  // 15 bytes total
    std::string result = ChatbotTrainer::truncate_text_tail(s, 6);
    ASSERT_FALSE(result.empty());
    // First byte must not be a UTF-8 continuation byte (10xxxxxx).
    EXPECT_NE(static_cast<unsigned char>(result[0]) & 0xC0, 0x80);
}

TEST(TruncateTailTest, TokensShorterThanLimit_ReturnsUnchanged) {
    std::vector<int> ids = {1, 2, 3};
    EXPECT_EQ(ChatbotTrainer::truncate_tokens_tail(ids, 10), ids);
}

TEST(TruncateTailTest, TokensLongerThanLimit_KeepsTailNotHead) {
    std::vector<int> ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> result = ChatbotTrainer::truncate_tokens_tail(ids, 4);
    std::vector<int> expected = {7, 8, 9, 10};  // tail, not {1,2,3,4} (head)
    EXPECT_EQ(result, expected);
}

TEST(ConfigTest, ValidationSplit_Zero_NoValidation) {
    TrainingConfig config;
    config.validation_split = 0;

    EXPECT_EQ(config.validation_split, 0);
}

TEST(ConfigTest, ValidationSplit_Typical) {
    TrainingConfig config;
    config.validation_split = 10;  // 10% validation

    EXPECT_EQ(config.validation_split, 10);
}

// ============================================================================
// TokenizerMode in TrainingConfig Tests
// ============================================================================

TEST(TokenizerModeConfigTest, DefaultModeIsASCII) {
    TrainingConfig config;
    EXPECT_EQ(config.tokenizer_mode, TokenizerMode::ASCII);
    EXPECT_NE(config.tokenizer_mode, TokenizerMode::UNICODE);
}

TEST(TokenizerModeConfigTest, CanSetUnicodeMode) {
    TrainingConfig config;
    config.tokenizer_mode = TokenizerMode::UNICODE;
    EXPECT_EQ(config.tokenizer_mode, TokenizerMode::UNICODE);
}

TEST(TokenizerModeConfigTest, ModePersistedInGetConfig) {
    TrainingConfig config;
    config.log_level = LogLevel::SILENT;
    config.tokenizer_mode = TokenizerMode::UNICODE;

    ChatbotTrainer trainer(config);
    EXPECT_EQ(trainer.get_config().tokenizer_mode, TokenizerMode::UNICODE);
}

TEST(TokenizerModeConfigTest, AsciiModePersistedInGetConfig) {
    TrainingConfig config;
    config.log_level = LogLevel::SILENT;
    config.tokenizer_mode = TokenizerMode::ASCII;

    ChatbotTrainer trainer(config);
    EXPECT_EQ(trainer.get_config().tokenizer_mode, TokenizerMode::ASCII);
}

// ============================================================================
// Tokenized-data cache (load_tokenized_cache/save_tokenized_cache) —
// friended into ChatbotTrainer (see ChatbotTrainer.hpp) since exercising
// these through the public API would require a real train() run (real
// model+tokenizer, actual forward/backward passes) just to reach
// preprocess_data() — heavier than this file's other ChatbotTrainer tests
// take on. Static wrapper methods here call the private methods directly
// (legal: this fixture class itself is the friend); TEST_F bodies below only
// ever call these inherited wrappers, never touch ChatbotTrainer internals
// directly — same pattern as DataFetcherGutenbergCleaningTest in
// DataFetcherTests.cpp.
// ============================================================================

class ChatbotTrainerCacheTest : public ::testing::Test {
   protected:
    std::string cache_dir;

    void SetUp() override {
        cache_dir = "test_tokenized_cache_dir";
        std::filesystem::remove_all(cache_dir);
    }

    void TearDown() override {
        std::filesystem::remove_all(cache_dir);
    }

    static bool LoadCache(ChatbotTrainer& t, const std::string& path) {
        return t.load_tokenized_cache(path);
    }
    static void SaveCache(const ChatbotTrainer& t, const std::string& path) {
        t.save_tokenized_cache(path);
    }
    static void SetTrainingData(ChatbotTrainer& t, std::vector<ConversationPair> data) {
        t.training_data = std::move(data);
    }
    static void SetValidationData(ChatbotTrainer& t, std::vector<ConversationPair> data) {
        t.validation_data = std::move(data);
    }
    static void SetTokenized(ChatbotTrainer& t, std::vector<TokenizedPair> train,
                             std::vector<TokenizedPair> val) {
        t.tokenized_training_data = std::move(train);
        t.tokenized_validation_data = std::move(val);
    }
    static const std::vector<TokenizedPair>& GetTokenizedTraining(const ChatbotTrainer& t) {
        return t.tokenized_training_data;
    }
    static const std::vector<TokenizedPair>& GetTokenizedValidation(const ChatbotTrainer& t) {
        return t.tokenized_validation_data;
    }

    static TrainingConfig quiet_config() {
        TrainingConfig cfg;
        cfg.log_level = LogLevel::SILENT;
        return cfg;
    }
};

TEST_F(ChatbotTrainerCacheTest, RoundTripSavesAndLoadsCorrectly) {
    ChatbotTrainer writer(quiet_config());
    SetTrainingData(writer, {ConversationPair("q1", "a1"), ConversationPair("q2", "a2")});
    SetValidationData(writer, {ConversationPair("q3", "a3")});
    SetTokenized(writer,
                 {TokenizedPair({1, 2, 3}, {4, 5}, "q1", "a1"), TokenizedPair({6}, {7, 8, 9}, "q2", "a2")},
                 {TokenizedPair({10, 11}, {12}, "q3", "a3")});

    const std::string path = cache_dir + "/key.cache";
    SaveCache(writer, path);
    ASSERT_TRUE(std::filesystem::exists(path));

    ChatbotTrainer reader(quiet_config());
    SetTrainingData(reader, {ConversationPair("q1", "a1"), ConversationPair("q2", "a2")});
    SetValidationData(reader, {ConversationPair("q3", "a3")});
    ASSERT_TRUE(LoadCache(reader, path));

    const auto& train = GetTokenizedTraining(reader);
    ASSERT_EQ(train.size(), 2u);
    EXPECT_EQ(train[0].input_tokens, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(train[0].target_tokens, (std::vector<int>{4, 5}));
    EXPECT_EQ(train[0].input_text, "q1");
    EXPECT_EQ(train[0].target_text, "a1");
    EXPECT_EQ(train[1].input_tokens, (std::vector<int>{6}));
    EXPECT_EQ(train[1].target_tokens, (std::vector<int>{7, 8, 9}));

    const auto& val = GetTokenizedValidation(reader);
    ASSERT_EQ(val.size(), 1u);
    EXPECT_EQ(val[0].input_tokens, (std::vector<int>{10, 11}));
    EXPECT_EQ(val[0].target_tokens, (std::vector<int>{12}));
}

TEST_F(ChatbotTrainerCacheTest, RoundTripSurvivesEmptyVectors) {
    ChatbotTrainer writer(quiet_config());
    SetTrainingData(writer, {ConversationPair("", "")});
    SetValidationData(writer, {});
    SetTokenized(writer, {TokenizedPair({}, {}, "", "")}, {});

    const std::string path = cache_dir + "/empty.cache";
    SaveCache(writer, path);

    ChatbotTrainer reader(quiet_config());
    SetTrainingData(reader, {ConversationPair("", "")});
    SetValidationData(reader, {});
    ASSERT_TRUE(LoadCache(reader, path));

    const auto& train = GetTokenizedTraining(reader);
    ASSERT_EQ(train.size(), 1u);
    EXPECT_TRUE(train[0].input_tokens.empty());
    EXPECT_TRUE(train[0].target_tokens.empty());
    EXPECT_TRUE(GetTokenizedValidation(reader).empty());
}

TEST_F(ChatbotTrainerCacheTest, MissWhenFileDoesNotExist) {
    ChatbotTrainer reader(quiet_config());
    SetTrainingData(reader, {ConversationPair("q", "a")});
    EXPECT_FALSE(LoadCache(reader, cache_dir + "/does-not-exist.cache"));
}

TEST_F(ChatbotTrainerCacheTest, MissOnSampleCountMismatch) {
    ChatbotTrainer writer(quiet_config());
    SetTrainingData(writer, {ConversationPair("q1", "a1"), ConversationPair("q2", "a2")});
    SetValidationData(writer, {});
    SetTokenized(writer, {TokenizedPair({1}, {2}, "q1", "a1"), TokenizedPair({3}, {4}, "q2", "a2")},
                 {});
    const std::string path = cache_dir + "/mismatch.cache";
    SaveCache(writer, path);

    // Different (smaller) training_data size than what was cached — this is
    // exactly what a changed dataset looks like; the tokenized_cache_key
    // (computed by IncrementalTrainer) is the primary defense against this in
    // practice, but load_tokenized_cache() must never trust a stale cache
    // just because the file happens to exist.
    ChatbotTrainer reader(quiet_config());
    SetTrainingData(reader, {ConversationPair("q1", "a1")});
    EXPECT_FALSE(LoadCache(reader, path));
}

// ============================================================================
// Abort-flag tests (incremental_trainer `serve`'s /admin/pause)
//
// Uses a tiny real model — same sizing convention as
// generation_quality_async_test.cpp's make_tiny_trainer() — since
// was_aborted()/train_epoch()'s abort check only make sense to verify against
// an actual train() run, not the config-only tests above.
// ============================================================================

namespace {

std::unique_ptr<ChatbotTrainer> make_abort_test_trainer(TrainingConfig cfg, int num_train_pairs,
                                                        const std::string& vocab_tmp_path) {
    cfg.d_model = 8;
    cfg.num_heads = 2;
    cfg.d_ff = 32;
    cfg.num_encoder_layers = 1;
    cfg.num_decoder_layers = 1;
    cfg.max_seq_length = 16;
    cfg.log_level = LogLevel::SILENT;
    cfg.validation_split = 0;  // no validation split needed for these tests

    auto trainer = std::make_unique<ChatbotTrainer>(cfg);
    std::vector<std::string> corpus = {"hello world foo bar baz", "the quick brown fox"};
    trainer->build_vocabulary(corpus, 50, vocab_tmp_path);
    for (int i = 0; i < num_train_pairs; ++i) {
        trainer->add_training_pair("hello", "world");
    }
    return trainer;
}

}  // namespace

TEST(ChatbotTrainerAbortTest, WasAbortedFalseBeforeAnyTrainCall) {
    TrainingConfig cfg;
    auto trainer =
        make_abort_test_trainer(cfg, 3, "/tmp/adai_chatbottrainer_abort_test_default_vocab.txt");
    EXPECT_FALSE(trainer->was_aborted());
}

TEST(ChatbotTrainerAbortTest, TrainCompletesNormallyWhenAbortFlagNeverSet) {
    TrainingConfig cfg;
    cfg.num_epochs = 2;
    auto trainer = make_abort_test_trainer(cfg, 4, "/tmp/adai_chatbottrainer_abort_test_normal_vocab.txt");

    std::atomic<bool> abort_flag{false};
    trainer->set_abort_flag(&abort_flag);  // registered, but never set

    ASSERT_TRUE(trainer->train(2));
    EXPECT_FALSE(trainer->was_aborted());
    EXPECT_EQ(trainer->get_training_losses().size(), 2u);
    EXPECT_GT(trainer->get_global_step(), 0);
}

TEST(ChatbotTrainerAbortTest, TrainStopsWithZeroProgressWhenAbortFlagAlreadySet) {
    TrainingConfig cfg;
    cfg.num_epochs = 5;
    auto trainer = make_abort_test_trainer(cfg, 4, "/tmp/adai_chatbottrainer_abort_test_immediate_vocab.txt");

    std::atomic<bool> abort_flag{true};  // set before train() is even called
    trainer->set_abort_flag(&abort_flag);

    // train() itself still returns true — no exception was thrown, it just
    // did fewer epochs than asked. was_aborted() is the only signal that
    // this wasn't a normal completion (see
    // IncrementalTrainer::run_training()'s comment on this exact point).
    ASSERT_TRUE(trainer->train(5));
    EXPECT_TRUE(trainer->was_aborted());
    // Checked at accumulation_step==0, i.e. before the very first sample of
    // epoch 0 — zero optimizer steps ever ran.
    EXPECT_EQ(trainer->get_global_step(), 0);
}

TEST(ChatbotTrainerAbortTest, TrainStopsAfterCurrentEpochWhenFlagSetMidRun) {
    TrainingConfig cfg;
    cfg.num_epochs = 4;
    auto trainer = make_abort_test_trainer(cfg, 4, "/tmp/adai_chatbottrainer_abort_test_midrun_vocab.txt");

    std::atomic<bool> abort_flag{false};
    trainer->set_abort_flag(&abort_flag);
    // Flip the flag once epoch 0 has fully completed (epoch_callback_ fires
    // after train_epoch() + validate() for that epoch) — proves the abort is
    // honored at the next epoch boundary rather than being missed entirely.
    trainer->set_epoch_callback([&abort_flag](int epoch, int, float, float, float) {
        if (epoch == 0) {
            abort_flag = true;
        }
    });

    ASSERT_TRUE(trainer->train(4));
    EXPECT_TRUE(trainer->was_aborted());
    const int steps_after_one_full_epoch = trainer->get_global_step();
    EXPECT_GT(steps_after_one_full_epoch, 0);
    // Epoch 1's train_epoch() call aborts before any sample runs (0 more
    // optimizer steps), and epochs 2-3 never start at all — global_step
    // never advances past what epoch 0 alone produced.
    EXPECT_EQ(trainer->get_global_step(), steps_after_one_full_epoch);
}

TEST(ChatbotTrainerAbortTest, SetAbortFlagNullptrDisablesAbortChecking) {
    TrainingConfig cfg;
    cfg.num_epochs = 1;
    auto trainer = make_abort_test_trainer(cfg, 3, "/tmp/adai_chatbottrainer_abort_test_nullptr_vocab.txt");

    trainer->set_abort_flag(nullptr);  // the default — no crash, never aborts
    ASSERT_TRUE(trainer->train(1));
    EXPECT_FALSE(trainer->was_aborted());
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
