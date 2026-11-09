#include "../src/ChatbotTrainer.hpp"
#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include "../src/IMetricsReporter.hpp"
#include "../src/Logger.hpp"
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

// TD-276: token accuracy, as validate() computes it (calculate_accuracy() was removed unused).
TEST(TokenAccuracyTest, CountsPositionsWhoseFirstMaximumIsTheTarget) {
    Matrix logits(4, 5);  // zero-filled
    logits(0, 2) = 3.0f;  // argmax 2
    logits(1, 1) = 2.0f;
    logits(1, 4) = 2.0f;  // tie: the first maximum, 1
    logits(2, 0) = 1.0f;  // argmax 0
    logits(3, 3) = 9.0f;  // row past the targets: not counted
    const TokenAccuracy a = EncoderDecoderModel::token_accuracy(logits, {2, 1, 4});
    EXPECT_EQ(a.total, 3);
    EXPECT_EQ(a.correct, 2);
    const TokenAccuracy tie_second = EncoderDecoderModel::token_accuracy(logits, {2, 4});
    EXPECT_EQ(tie_second.correct, 1) << "a tie goes to the first maximum";
    const TokenAccuracy invalid = EncoderDecoderModel::token_accuracy(logits, {-1, 5, 0});
    EXPECT_EQ(invalid.total, 1) << "targets outside the vocabulary aren't counted";
    EXPECT_EQ(invalid.correct, 1);
    EXPECT_EQ(EncoderDecoderModel::token_accuracy(logits, {}).total, 0);
}

// ============================================================================
// Logging System Tests
// ============================================================================

TEST_F(LoggingTest, LogLevel_Silent_NoOutput) {
    TrainingConfig config;
    config.log_level = LogLevel::SILENT;
    auto silent_trainer = std::make_unique<ChatbotTrainer>(config);

    // Should not crash, but won't verify output (would need output capture)
    silent_trainer->log(LogLevel::NORMAL, "This should not appear");
    silent_trainer->log(LogLevel::VERBOSE, "This should not appear");
    silent_trainer->log(LogLevel::DEBUG, "This should not appear");
}

TEST_F(LoggingTest, LogLevel_Normal_FiltersVerbose) {
    TrainingConfig config;
    config.log_level = LogLevel::NORMAL;
    auto normal_trainer = std::make_unique<ChatbotTrainer>(config);

    // NORMAL and below should appear, VERBOSE and DEBUG should not
    normal_trainer->log(LogLevel::SILENT, "Error message");
    normal_trainer->log(LogLevel::NORMAL, "Normal message");
    normal_trainer->log(LogLevel::VERBOSE, "Should be filtered");
    normal_trainer->log(LogLevel::DEBUG, "Should be filtered");
}

TEST_F(LoggingTest, LogLevel_Verbose_FiltersDebug) {
    TrainingConfig config;
    config.log_level = LogLevel::VERBOSE;
    auto verbose_trainer = std::make_unique<ChatbotTrainer>(config);

    // VERBOSE and below should appear, DEBUG should not
    verbose_trainer->log(LogLevel::SILENT, "Error message");
    verbose_trainer->log(LogLevel::NORMAL, "Normal message");
    verbose_trainer->log(LogLevel::VERBOSE, "Verbose message");
    verbose_trainer->log(LogLevel::DEBUG, "Should be filtered");
}

// TD-276: each message goes out at its own Logger level (every one used to be info), filtered by
// log_level as before.
TEST(ChatbotTrainerLogTest, LevelsMapToLoggerLevelsAndFilter) {
    std::ostringstream out;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(out);
    sink->set_pattern("%l|%v");
    auto logger = adai::Logger::get();
    const auto old_level = logger->level();
    logger->set_level(spdlog::level::debug);
    logger->sinks().push_back(sink);

    TrainingConfig config;
    config.log_level = LogLevel::DEBUG;
    ChatbotTrainer all(config);
    all.log(LogLevel::SILENT, "td276-error");
    all.log(LogLevel::NORMAL, "td276-normal");
    all.log(LogLevel::VERBOSE, "td276-verbose");
    all.log(LogLevel::DEBUG, "td276-debug");
    config.log_level = LogLevel::NORMAL;
    ChatbotTrainer normal(config);
    normal.log(LogLevel::VERBOSE, "td276-filtered");
    normal.log(LogLevel::DEBUG, "td276-filtered");

    logger->flush();
    auto& sinks = logger->sinks();
    sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
    logger->set_level(old_level);

    const std::string log = out.str();
    EXPECT_NE(log.find("error|td276-error"), std::string::npos) << log;
    EXPECT_NE(log.find("info|td276-normal"), std::string::npos) << log;
    EXPECT_NE(log.find("info|td276-verbose"), std::string::npos) << log;
    EXPECT_NE(log.find("debug|td276-debug"), std::string::npos) << log;
    EXPECT_EQ(log.find("td276-filtered"), std::string::npos) << log;
}

TEST_F(LoggingTest, LogLevel_Debug_ShowsAll) {
    TrainingConfig config;
    config.log_level = LogLevel::DEBUG;
    auto debug_trainer = std::make_unique<ChatbotTrainer>(config);

    // All messages should appear
    debug_trainer->log(LogLevel::SILENT, "Error message");
    debug_trainer->log(LogLevel::NORMAL, "Normal message");
    debug_trainer->log(LogLevel::VERBOSE, "Verbose message");
    debug_trainer->log(LogLevel::DEBUG, "Debug message");
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
    EXPECT_EQ(default_config.tokenizer_mode, TokenizerMode::UNICODE);  // TD-564
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

// TD-273: the cheap loss score is on by default; the per-pair generation one stays opt-in.
TEST(QualityBackfillTest, Config_DefaultsLossOnGenerationOff) {
    TrainingConfig config;
    EXPECT_TRUE(config.enable_loss_quality_backfill);
    EXPECT_EQ(config.quality_report_lowest_samples, 5);
    EXPECT_FALSE(config.enable_generation_quality_backfill);
    EXPECT_EQ(config.generation_backfill_max_tokens, 50);
}

TEST(QualityBackfillTest, Config_CanDisableLossBackfill) {
    TrainingConfig config;
    config.enable_loss_quality_backfill = false;
    EXPECT_FALSE(config.enable_loss_quality_backfill);
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
// TD-169: MetricsTracker wiring tests — real end-to-end training runs (this
// codebase's other ChatbotTrainer tests avoid calling train() with real data;
// this is the first to do so, since faithfulness to the real per-epoch data is
// exactly what needs proving here, not just "record_epoch() works in
// isolation" — that's already covered by tests/metricstracker_test.cpp).
// ============================================================================

class MetricsTrackerWiringTest : public ::testing::Test {
   protected:
    std::filesystem::path vocab_path_;
    std::filesystem::path data_path_;

    void SetUp() override {
        // Matches the order every real GPU-enabled binary follows (IncrementalTrainingTool.cpp,
        // ChatbotAPIServer.cpp both call Matrix::gpu_try_initialize() before constructing any
        // trainer/model) -- ChatbotTrainer::setup_training() unconditionally calls
        // model->gpu_init_training() under #ifdef ADAI_ENABLE_GPU with no runtime
        // is_available() check of its own, so it relies on the caller having already
        // initialized (or deliberately left uninitialized) GPUManager first. Without this,
        // every test here failed with "GPU not initialized" the first time this suite actually
        // ran against real GPU hardware (ai-machine) rather than a device-less sandbox, where
        // gpu_try_initialize()'s own probe() short-circuits to false and nothing downstream is
        // reached. gpu_try_initialize() is the safe wrapper (probes, catches, falls back to
        // CPU) -- exactly what a real binary would call, not the throwing gpu_initialize().
        Matrix::gpu_try_initialize();
        vocab_path_ =
            std::filesystem::temp_directory_path() / "adai_metricstracker_wiring_vocab.txt";
        data_path_ = std::filesystem::temp_directory_path() / "adai_metricstracker_wiring_data.jsonl";
    }

    void TearDown() override {
        std::filesystem::remove(vocab_path_);
        std::filesystem::remove(data_path_);
    }

    static TrainingConfig tiny_config() {
        TrainingConfig config;
        config.log_level = LogLevel::SILENT;
        config.d_model = 16;
        config.num_heads = 2;
        config.d_ff = 32;
        config.num_encoder_layers = 1;
        config.num_decoder_layers = 1;
        config.max_seq_length = 16;
        config.validation_split = 0;  // keep these tests focused on the training-loss path
        config.enable_early_stopping = false;
        return config;
    }
};

TEST_F(MetricsTrackerWiringTest, RecordsOneEntryPerEpochMatchingRawVectors) {
    ChatbotTrainer trainer(tiny_config());
    std::vector<std::string> corpus = {"hello world",  "how are you",  "I am fine",   "thank you",
                                       "good morning", "good evening", "see you later", "goodbye"};
    ASSERT_TRUE(trainer.build_vocabulary(corpus, 100, vocab_path_.string()));

    {
        std::ofstream f(data_path_);
        f << "{\"input\":\"hello world\",\"response\":\"how are you\"}\n"
             "{\"input\":\"I am fine\",\"response\":\"thank you\"}\n"
             "{\"input\":\"good morning\",\"response\":\"good evening\"}\n"
             "{\"input\":\"see you later\",\"response\":\"goodbye\"}\n";
    }
    ASSERT_TRUE(trainer.load_conversation_data(data_path_.string()));
    ASSERT_EQ(trainer.get_training_data_size(), 4u);

    const int num_epochs = 3;
    ASSERT_TRUE(trainer.train(num_epochs));

    const MetricsTracker& tracker = trainer.get_metrics_tracker();
    ASSERT_EQ(tracker.size(), static_cast<size_t>(num_epochs));
    ASSERT_EQ(trainer.get_training_losses().size(), static_cast<size_t>(num_epochs));
    ASSERT_EQ(trainer.get_gradient_norms().size(), static_cast<size_t>(num_epochs));
    ASSERT_EQ(trainer.get_learning_rates().size(), static_cast<size_t>(num_epochs));

    // The core correctness claim: metrics_tracker_ isn't just "recording something" -- its
    // history must be the SAME data ChatbotTrainer's own already-tracked vectors hold, epoch
    // for epoch, not an independently-computed or subtly-offset copy.
    const auto& history = tracker.get_history();
    for (int i = 0; i < num_epochs; ++i) {
        EXPECT_EQ(history[i].epoch, i);
        EXPECT_FLOAT_EQ(history[i].train_loss, trainer.get_training_losses()[i]);
        EXPECT_FLOAT_EQ(history[i].gradient_norm, trainer.get_gradient_norms()[i]);
        EXPECT_FLOAT_EQ(history[i].learning_rate, trainer.get_learning_rates()[i]);
        EXPECT_FLOAT_EQ(history[i].validation_loss, 0.0f)
            << "no validation data was loaded, so every recorded validation_loss should be 0";
    }
}

// TD-268: records start_epoch()/end_epoch() calls; every other report is ignored.
class EpochRecordingReporter final : public IMetricsReporter {
   public:
    struct Ended {
        int epoch;
        float loss, validation_loss, learning_rate, perplexity, gradient_norm;
        double seconds;
    };
    std::vector<int> started;
    std::vector<Ended> ended;
    std::vector<SampleQualityReport> quality;  // TD-273

    void start_epoch(int epoch, int /*total_samples*/) override { started.push_back(epoch); }
    void end_epoch(int epoch, float loss, float validation_loss, float learning_rate, float perplexity,
                   float gradient_norm, double epoch_time_seconds) override {
        ended.push_back({epoch, loss, validation_loss, learning_rate, perplexity, gradient_norm,
                         epoch_time_seconds});
    }
    void update_sample_metrics(int, float, float, float) override {}
    void update_validation_metrics(float, float, float) override {}
    void update_best_metrics(float, int) override {}
    void update_advanced_epoch_metrics(float, float, float) override {}
    void flag_abnormal_sample(const AbnormalSample&) override {}
    void update_adaptive_clip_metrics(float, int) override {}
    void update_adaptive_clip_epoch(float, int) override {}
    void update_activation_saturation(float) override {}
    void update_attention_entropy(float) override {}
    void update_layer_gradient_norms(const std::vector<float>&, const std::vector<float>&) override {}
    void update_padding_efficiency(float) override {}
    void update_generation_quality_metrics(float, float, float, float, float, float) override {}
    void update_lejepa_metrics(float, float) override {}
    void update_lejepa_advanced_metrics(float, float, float, float) override {}
    void report_sample_quality(const SampleQualityReport& report) override {
        quality.push_back(report);
    }
};

// TD-344: train() applies DROPOUT while it trains and leaves the model without it, so the saved or
// reused model never drops; a rate outside [0, 1) is refused.
TEST_F(MetricsTrackerWiringTest, TrainAppliesDropoutOnlyWhileTraining) {
    TrainingConfig config = tiny_config();
    config.dropout = 0.5f;
    ChatbotTrainer trainer(config);
    ASSERT_TRUE(trainer.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"},
                                         100, vocab_path_.string()));
    {
        std::ofstream f(data_path_);
        f << "{\"input\":\"hello world\",\"response\":\"how are you\"}\n"
             "{\"input\":\"I am fine\",\"response\":\"thank you\"}\n";
    }
    ASSERT_TRUE(trainer.load_conversation_data(data_path_.string()));
    ASSERT_TRUE(trainer.train(2));
    auto model = trainer.release_model();
    ASSERT_TRUE(model);
    EXPECT_FLOAT_EQ(model->get_dropout(), 0.0f);

    TrainingConfig bad = tiny_config();
    bad.dropout = 1.0f;
    ChatbotTrainer refused(bad);
    ASSERT_TRUE(
        refused.build_vocabulary({"hello world", "how are you"}, 100, vocab_path_.string()));
    ASSERT_TRUE(refused.load_conversation_data(data_path_.string()));
    EXPECT_FALSE(refused.train(1));
}

// TD-268: a run without validation data (validation_split = 0) used to start every epoch on the
// reporter and end none, and the gradient norm sent was the last window's, not the epoch's average.
TEST_F(MetricsTrackerWiringTest, EndsEveryEpochOnTheReporterWithoutValidationData) {
    ChatbotTrainer trainer(tiny_config());
    EpochRecordingReporter reporter;
    trainer.set_metrics_reporter(&reporter);
    ASSERT_TRUE(trainer.build_vocabulary({"hello world", "how are you", "I am fine", "thank you",
                                          "good morning", "good evening"},
                                         100, vocab_path_.string()));
    {
        std::ofstream f(data_path_);
        f << "{\"input\":\"hello world\",\"response\":\"how are you\"}\n"
             "{\"input\":\"I am fine\",\"response\":\"thank you\"}\n"
             "{\"input\":\"good morning\",\"response\":\"good evening\"}\n";
    }
    ASSERT_TRUE(trainer.load_conversation_data(data_path_.string()));

    const int num_epochs = 3;
    ASSERT_TRUE(trainer.train(num_epochs));
    ASSERT_EQ(trainer.get_validation_losses().size(), 0u);

    ASSERT_EQ(reporter.started.size(), static_cast<size_t>(num_epochs));
    ASSERT_EQ(reporter.ended.size(), static_cast<size_t>(num_epochs))
        << "every epoch started on the reporter must be ended";
    for (int i = 0; i < num_epochs; ++i) {
        const auto& e = reporter.ended[i];
        EXPECT_EQ(e.epoch, i + 1);
        EXPECT_EQ(reporter.started[i], i + 1);
        EXPECT_FLOAT_EQ(e.loss, trainer.get_training_losses()[i]);
        EXPECT_FLOAT_EQ(e.validation_loss, 0.0f) << "0 = no validation loss computed";
        EXPECT_FLOAT_EQ(e.gradient_norm, trainer.get_gradient_norms()[i])
            << "the epoch's average gradient norm";
        EXPECT_FLOAT_EQ(e.learning_rate, trainer.get_learning_rates()[i]);
        EXPECT_GT(e.seconds, 0.0);
    }
}

// TD-268: with validation data the epoch still ends once, carrying that epoch's validation loss.
TEST_F(MetricsTrackerWiringTest, EndsEachEpochOnceWithItsValidationLoss) {
    TrainingConfig config = tiny_config();
    ChatbotTrainer trainer(config);
    EpochRecordingReporter reporter;
    trainer.set_metrics_reporter(&reporter);
    ASSERT_TRUE(trainer.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"}, 100,
                                         vocab_path_.string()));
    trainer.add_training_pair("hello world", "how are you");
    trainer.add_training_pair("I am fine", "thank you");
    trainer.add_validation_pair("how are you", "I am fine");

    const int num_epochs = 2;
    ASSERT_TRUE(trainer.train(num_epochs));
    ASSERT_EQ(trainer.get_validation_losses().size(), static_cast<size_t>(num_epochs));
    ASSERT_EQ(reporter.ended.size(), static_cast<size_t>(num_epochs));
    for (int i = 0; i < num_epochs; ++i) {
        EXPECT_FLOAT_EQ(reporter.ended[i].validation_loss, trainer.get_validation_losses()[i]);
        EXPECT_FLOAT_EQ(reporter.ended[i].gradient_norm, trainer.get_gradient_norms()[i]);
    }
}

// TD-273: the quality summary sent to the metrics server.
TEST(SampleQualitySummaryTest, StatisticsAndLowestPairs) {
    std::vector<ConversationPair> pairs;
    const float scores[] = {0.9f, -1.0f, 0.1f, 0.5f, 0.3f, std::nanf(""), 0.7f};
    for (int i = 0; i < 7; ++i) {
        SampleMeta m;
        m.quality = scores[i];
        pairs.emplace_back("in" + std::to_string(i), "out" + std::to_string(i), m);
    }
    const SampleQualityReport r =
        ChatbotTrainer::summarize_sample_quality(pairs, 3, "loss", "train", 2);
    EXPECT_EQ(r.epoch, 3);
    EXPECT_EQ(r.source, "loss");
    EXPECT_EQ(r.split, "train");
    EXPECT_EQ(r.count, 5) << "unscored (negative) and non-finite scores are left out";
    EXPECT_NEAR(r.mean, 0.5f, 1e-6f);
    EXPECT_FLOAT_EQ(r.min, 0.1f);
    EXPECT_FLOAT_EQ(r.max, 0.9f);
    EXPECT_NEAR(r.median, 0.5f, 1e-6f);
    EXPECT_NEAR(r.p10, 0.18f, 1e-6f);  // 0.1 + 0.4 * (0.3 - 0.1)
    EXPECT_NEAR(r.p90, 0.82f, 1e-6f);  // 0.7 + 0.6 * (0.9 - 0.7)
    ASSERT_EQ(r.lowest.size(), 2u);
    EXPECT_EQ(r.lowest[0].input_text, "in2");
    EXPECT_EQ(r.lowest[0].target_text, "out2");
    EXPECT_FLOAT_EQ(r.lowest[0].quality, 0.1f);
    EXPECT_EQ(r.lowest[1].input_text, "in4");

    EXPECT_EQ(ChatbotTrainer::summarize_sample_quality({}, 1, "loss", "train", 5).count, 0);
}

// TD-273: every epoch's loss scores reach the reporter (they used to stay in private copies nobody
// read), per split, with the pairs' dataset metadata kept.
TEST_F(MetricsTrackerWiringTest, ReportsLossQualityEachEpochAndKeepsPairMetadata) {
    TrainingConfig config = tiny_config();
    ASSERT_TRUE(config.enable_loss_quality_backfill) << "on by default";
    config.quality_report_lowest_samples = 2;
    ChatbotTrainer trainer(config);
    EpochRecordingReporter reporter;
    trainer.set_metrics_reporter(&reporter);
    ASSERT_TRUE(trainer.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"},
                                         100, vocab_path_.string()));
    SampleMeta meta;
    meta.domain = "smalltalk";
    trainer.add_training_pair(ConversationPair("hello world", "how are you", meta));
    trainer.add_training_pair(ConversationPair("I am fine", "thank you", meta));
    trainer.add_training_pair(ConversationPair("how are you", "hello world", meta));
    trainer.add_validation_pair(ConversationPair("thank you", "I am fine", meta));

    const int num_epochs = 2;
    ASSERT_TRUE(trainer.train(num_epochs));
    ASSERT_EQ(reporter.quality.size(), static_cast<size_t>(2 * num_epochs))
        << "train + validation per epoch";
    for (int e = 0; e < num_epochs; ++e) {
        const SampleQualityReport& train = reporter.quality[2 * e];
        const SampleQualityReport& val = reporter.quality[2 * e + 1];
        EXPECT_EQ(train.epoch, e + 1);
        EXPECT_EQ(train.source, "loss");
        EXPECT_EQ(train.split, "train");
        EXPECT_EQ(train.count, 3);
        EXPECT_EQ(train.lowest.size(), 2u);
        EXPECT_GT(train.min, 0.0f);
        EXPECT_LE(train.max, 1.0f);
        EXPECT_EQ(val.split, "validation");
        EXPECT_EQ(val.count, 1);
        // exp(-loss) of the pair's own validation loss.
        EXPECT_NEAR(val.mean, std::exp(-trainer.get_validation_losses()[e]), 1e-4f);
    }
    ASSERT_EQ(trainer.get_training_pairs().size(), 3u);
    for (const auto& pair : trainer.get_training_pairs()) {
        EXPECT_EQ(pair.meta.domain, "smalltalk");
        EXPECT_GT(pair.meta.quality, 0.0f);
    }

    ChatbotTrainer off_trainer([&] {
        TrainingConfig off = tiny_config();
        off.enable_loss_quality_backfill = false;
        return off;
    }());
    EpochRecordingReporter off_reporter;
    off_trainer.set_metrics_reporter(&off_reporter);
    ASSERT_TRUE(
        off_trainer.build_vocabulary({"hello world", "how are you"}, 100, vocab_path_.string()));
    off_trainer.add_training_pair("hello world", "how are you");
    ASSERT_TRUE(off_trainer.train(1));
    EXPECT_TRUE(off_reporter.quality.empty());
}

// TD-273: the generation backfill (opt-in) reports once after training, per split.
TEST_F(MetricsTrackerWiringTest, ReportsGenerationQualityAfterTrainingWhenEnabled) {
    TrainingConfig config = tiny_config();
    config.enable_loss_quality_backfill = false;
    config.enable_generation_quality_backfill = true;
    config.generation_backfill_max_tokens = 4;
    ChatbotTrainer trainer(config);
    EpochRecordingReporter reporter;
    trainer.set_metrics_reporter(&reporter);
    ASSERT_TRUE(trainer.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"},
                                         100, vocab_path_.string()));
    trainer.add_training_pair("hello world", "how are you");
    trainer.add_training_pair("I am fine", "thank you");
    trainer.add_validation_pair("how are you", "I am fine");
    ASSERT_TRUE(trainer.train(2));
    ASSERT_EQ(reporter.quality.size(), 2u);
    EXPECT_EQ(reporter.quality[0].source, "generation");
    EXPECT_EQ(reporter.quality[0].split, "train");
    EXPECT_EQ(reporter.quality[0].epoch, 2);
    EXPECT_EQ(reporter.quality[0].count, 2);
    EXPECT_EQ(reporter.quality[1].split, "validation");
    EXPECT_EQ(reporter.quality[1].count, 1);
    for (const auto& r : reporter.quality) {
        EXPECT_GE(r.min, 0.0f);
        EXPECT_LE(r.max, 1.0f);
    }
}

// TD-274: the best-weights copy for restore_best_weights lives in a directory of the trainer's own under
// best_model_dir while training, and nothing is left afterwards; it used to be "best_model_temp.bin" (and
// sidecars) in the working directory, shared by every trainer there and never removed.
TEST_F(MetricsTrackerWiringTest, BestWeightsCopyIsPrivateAndRemovedAfterTraining) {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() / "adai_td274_best_model_dir";
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(base);
    auto best_dirs = [&base] {
        std::vector<std::filesystem::path> dirs;
        for (const auto& e : std::filesystem::directory_iterator(base))
            if (e.is_directory() && e.path().filename().string().rfind("best_model_", 0) == 0)
                dirs.push_back(e.path());
        return dirs;
    };
    auto cwd_leftovers = [] {
        int n = 0;
        for (const auto& e : std::filesystem::directory_iterator(std::filesystem::current_path()))
            if (e.path().filename().string().rfind("best_model_temp", 0) == 0)
                ++n;
        return n;
    };
    const int cwd_before = cwd_leftovers();

    TrainingConfig config = tiny_config();
    config.enable_early_stopping = true;
    config.restore_best_weights = true;
    config.patience = 100;
    config.best_model_dir = base.string();
    std::vector<std::filesystem::path> seen;  // the copy's directory each time a new best was saved
    std::size_t files_in_copy = 0;
    {
        ChatbotTrainer trainer(config);
        ASSERT_TRUE(trainer.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"}, 100,
                                             vocab_path_.string()));
        trainer.add_training_pair("hello world", "how are you");
        trainer.add_training_pair("I am fine", "thank you");
        trainer.add_validation_pair("how are you", "I am fine");
        trainer.set_best_model_callback([&](int, float) {
            const auto dirs = best_dirs();
            ASSERT_EQ(dirs.size(), 1u);
            seen.push_back(dirs[0]);
            files_in_copy = std::distance(std::filesystem::directory_iterator(dirs[0]),
                                          std::filesystem::directory_iterator());
        });
        ASSERT_TRUE(trainer.train(3));
        ASSERT_FALSE(seen.empty()) << "the first epoch is always a new best";
        EXPECT_GT(files_in_copy, 1u) << "the copy and its sidecars sit in that directory";
        for (const auto& d : seen) EXPECT_EQ(d, seen[0]) << "one directory per trainer";
        EXPECT_TRUE(best_dirs().empty()) << "removed once the best weights are restored";

        // A second train() call on the same trainer starts with a fresh copy and removes it again.
        seen.clear();
        ASSERT_TRUE(trainer.train(1));
        EXPECT_TRUE(best_dirs().empty());
    }
    EXPECT_EQ(cwd_leftovers(), cwd_before) << "nothing is written to the working directory";

    // Two trainers in one directory get different copies.
    std::vector<std::filesystem::path> first_dirs;
    ChatbotTrainer a(config), b(config);
    for (ChatbotTrainer* t : {&a, &b}) {
        ASSERT_TRUE(t->build_vocabulary({"hello world", "how are you"}, 100, vocab_path_.string()));
        t->add_training_pair("hello world", "how are you");
        t->add_validation_pair("how are you", "hello world");
        t->set_best_model_callback([&](int, float) {
            for (const auto& d : best_dirs())
                if (std::find(first_dirs.begin(), first_dirs.end(), d) == first_dirs.end())
                    first_dirs.push_back(d);
        });
        ASSERT_TRUE(t->train(1));
    }
    EXPECT_EQ(first_dirs.size(), 2u) << "each trainer has its own directory";
    EXPECT_TRUE(best_dirs().empty());
    std::filesystem::remove_all(base);
}

// TD-276: validate() computes token accuracy and sends it with the validation metrics (it always
// sent -1, so both dashboards hid it).
TEST_F(MetricsTrackerWiringTest, ValidationReportsTokenAccuracy) {
    class AccuracyReporter final : public IMetricsReporter {
       public:
        std::vector<float> accuracies;
        void start_epoch(int, int) override {}
        void end_epoch(int, float, float, float, float, float, double) override {}
        void update_sample_metrics(int, float, float, float) override {}
        void update_validation_metrics(float, float accuracy, float) override {
            accuracies.push_back(accuracy);
        }
        void update_best_metrics(float, int) override {}
        void update_advanced_epoch_metrics(float, float, float) override {}
        void flag_abnormal_sample(const AbnormalSample&) override {}
        void update_adaptive_clip_metrics(float, int) override {}
        void update_adaptive_clip_epoch(float, int) override {}
        void update_activation_saturation(float) override {}
        void update_attention_entropy(float) override {}
        void update_layer_gradient_norms(const std::vector<float>&,
                                         const std::vector<float>&) override {}
        void update_padding_efficiency(float) override {}
        void update_generation_quality_metrics(float, float, float, float, float, float) override {}
        void update_lejepa_metrics(float, float) override {}
        void update_lejepa_advanced_metrics(float, float, float, float) override {}
        void report_sample_quality(const SampleQualityReport&) override {}
    } reporter;

    TrainingConfig config = tiny_config();
    config.learning_rate = 0.01f;
    ChatbotTrainer trainer(config);
    trainer.set_metrics_reporter(&reporter);
    ASSERT_TRUE(
        trainer.build_vocabulary({"hello world", "how are you"}, 100, vocab_path_.string()));
    // Validation is the training pair itself, so accuracy can only rise as the model memorises it.
    trainer.add_training_pair("hello world", "how are you");
    trainer.add_validation_pair("hello world", "how are you");
    const int num_epochs = 40;
    ASSERT_TRUE(trainer.train(num_epochs));

    const auto& acc = trainer.get_validation_accuracies();
    ASSERT_EQ(acc.size(), static_cast<size_t>(num_epochs));
    ASSERT_EQ(reporter.accuracies.size(), static_cast<size_t>(num_epochs));
    for (int i = 0; i < num_epochs; ++i) {
        EXPECT_GE(acc[i], 0.0f);
        EXPECT_LE(acc[i], 1.0f);
        EXPECT_FLOAT_EQ(reporter.accuracies[i], acc[i]);
    }
    EXPECT_FLOAT_EQ(acc.back(), 1.0f) << "a memorised pair predicts every target token";
}

TEST_F(MetricsTrackerWiringTest, ExportsRealCsvAfterTraining) {
    ChatbotTrainer trainer(tiny_config());
    std::vector<std::string> corpus = {"hello world", "how are you", "I am fine", "thank you"};
    ASSERT_TRUE(trainer.build_vocabulary(corpus, 100, vocab_path_.string()));

    {
        std::ofstream f(data_path_);
        f << "{\"input\":\"hello world\",\"response\":\"how are you\"}\n"
             "{\"input\":\"I am fine\",\"response\":\"thank you\"}\n";
    }
    ASSERT_TRUE(trainer.load_conversation_data(data_path_.string()));

    const int num_epochs = 2;
    ASSERT_TRUE(trainer.train(num_epochs));

    std::filesystem::path csv_path =
        std::filesystem::temp_directory_path() / "adai_metricstracker_wiring_export.csv";
    ASSERT_TRUE(trainer.export_metrics_csv(csv_path.string()));
    ASSERT_TRUE(std::filesystem::exists(csv_path));

    std::ifstream f(csv_path);
    std::string header;
    std::getline(f, header);
    EXPECT_EQ(header,
             "epoch,train_loss,validation_loss,train_perplexity,validation_perplexity,"
             "learning_rate,gradient_norm,duration_seconds");

    int data_rows = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) {
            ++data_rows;
        }
    }
    EXPECT_EQ(data_rows, num_epochs);

    std::filesystem::remove(csv_path);
}

// Regression test: DatasetRegistry::load_conversation_pairs() and
// ChatbotTrainer::load_conversation_data() both share parse_jsonl_sample()
// (TrainingSampleMeta.hpp), which only requires a non-empty "input" field — a JSONL line with no
// "response" field at all (e.g. `{"input": "hello world"}`, exactly what a queued lejepa-style
// unpaired-text file might contain) is accepted with response="". Before this fix,
// ChatbotTrainer::preprocess_data() called tokenizer->encode("") on that empty response, which
// unconditionally throws TokenizerInputError (BPETokenizer.cpp's own validate_input()) — and
// since that call happens inside preprocess_data()'s own `#pragma omp parallel for` loop, an
// uncaught exception there terminates the whole process immediately (undefined behavior escaping
// an OpenMP parallel region), not just this one training pass. A real production trainer, not a
// unit test, hit exactly this crash.
TEST_F(MetricsTrackerWiringTest, TrainingSkipsPairsWithEmptyResponseInsteadOfCrashing) {
    ChatbotTrainer trainer(tiny_config());
    std::vector<std::string> corpus = {"hello world", "how are you", "I am fine", "thank you"};
    ASSERT_TRUE(trainer.build_vocabulary(corpus, 100, vocab_path_.string()));

    {
        std::ofstream f(data_path_);
        f << "{\"input\":\"hello world\",\"response\":\"how are you\"}\n"
             "{\"input\":\"I am fine\"}\n"  // no "response" field at all -> response == ""
             "{\"input\":\"I am fine\",\"response\":\"thank you\"}\n";
    }
    ASSERT_TRUE(trainer.load_conversation_data(data_path_.string()));
    // All three raw pairs are loaded as-is -- load_conversation_data() has no reason to reject
    // an empty response either; the skip belongs in preprocess_data(), not here.
    ASSERT_EQ(trainer.get_training_data_size(), 3u);

    // The actual regression check: training must complete normally, not crash, despite one of
    // the three loaded pairs having an empty response.
    bool trained_ok = false;
    EXPECT_NO_THROW({ trained_ok = trainer.train(1); });
    EXPECT_TRUE(trained_ok);
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

// TD-275: targets keep their head and still end on <eos> after truncation.
TEST(TruncateTargetTest, ShortTargetIsUnchanged) {
    const std::vector<int> ids = {2, 10, 11, 3};  // <bos> a b <eos>
    EXPECT_EQ(ChatbotTrainer::truncate_target_keep_eos(ids, 8, 3), ids);
    EXPECT_EQ(ChatbotTrainer::truncate_target_keep_eos(ids, 4, 3), ids);  // exactly max_len
}

TEST(TruncateTargetTest, LongTargetKeepsHeadAndEndsOnEos) {
    const std::vector<int> ids = {10, 11, 12, 13, 14, 3};  // targets carry no <bos> (TD-361)
    EXPECT_EQ(ChatbotTrainer::truncate_target_keep_eos(ids, 4, 3),
              (std::vector<int>{10, 11, 12, 3}));
}

TEST(TruncateTargetTest, NonPositiveLimitLeavesTargetAlone) {
    const std::vector<int> ids = {2, 10, 3};
    EXPECT_EQ(ChatbotTrainer::truncate_target_keep_eos(ids, 0, 3), ids);
}

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

// TD-564: Unicode is the default (it was ASCII).
TEST(TokenizerModeConfigTest, DefaultModeIsUnicode) {
    TrainingConfig config;
    EXPECT_EQ(config.tokenizer_mode, TokenizerMode::UNICODE);
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
    static void Preprocess(ChatbotTrainer& t) {
        t.preprocess_data();  // TD-295
    }
    // TD-267 hooks
    static void DropSkipped(ChatbotTrainer& t) {
        t.drop_skipped_pairs();
    }
    static const std::vector<ConversationPair>& GetTrainingData(const ChatbotTrainer& t) {
        return t.training_data;
    }
    static const std::vector<ConversationPair>& GetValidationData(const ChatbotTrainer& t) {
        return t.validation_data;
    }
    static float Validate(ChatbotTrainer& t) {
        return t.validate();
    }
    static void AppendValidation(ChatbotTrainer& t, ConversationPair raw, TokenizedPair tok) {
        t.validation_data.push_back(std::move(raw));
        t.tokenized_validation_data.push_back(std::move(tok));
    }
    static void SetTokenizedValidationOnly(ChatbotTrainer& t, std::vector<ConversationPair> raw,
                                           std::vector<TokenizedPair> tok) {
        t.validation_data = std::move(raw);
        t.tokenized_validation_data = std::move(tok);
    }
    static int EpochsWithoutImprovement(const ChatbotTrainer& t) {
        return t.epochs_without_improvement;
    }
    // TD-272 hooks
    static float TrainEpoch(ChatbotTrainer& t, int epoch) {
        return t.train_epoch(epoch);
    }
    static void AppendTraining(ChatbotTrainer& t, ConversationPair raw, TokenizedPair tok) {
        t.training_data.push_back(std::move(raw));
        t.tokenized_training_data.push_back(std::move(tok));
        t.training_indices.resize(t.tokenized_training_data.size());
        std::iota(t.training_indices.begin(), t.training_indices.end(), 0);
    }

    static int GlobalStep(const ChatbotTrainer& t) { return t.global_step; }
    static int AgcSteps(const ChatbotTrainer& t) { return t.agc_step_count_; }
    static int RunWarmup(const ChatbotTrainer& t) { return t.run_warmup_steps_; }
    static void InitModel(ChatbotTrainer& t) { t.initialize_model(); }
    static float LearningRateAt(ChatbotTrainer& t, int step, int total, int warmup) {
        t.total_training_steps = total;
        t.run_warmup_steps_ = warmup;
        return t.calculate_learning_rate(step);
    }

    static TrainingConfig tiny_training_config(int accumulation_steps) {
        TrainingConfig cfg = quiet_config();
        cfg.d_model = 16;
        cfg.num_heads = 2;
        cfg.d_ff = 32;
        cfg.num_encoder_layers = 1;
        cfg.num_decoder_layers = 1;
        cfg.max_seq_length = 16;
        cfg.validation_split = 0;
        cfg.enable_early_stopping = false;
        cfg.gradient_accumulation_steps = accumulation_steps;
        cfg.enable_loss_quality_backfill = true;  // exposes each sample's loss as exp(-loss)
        return cfg;
    }

    // TD-561 hook
    static bool BleuLowercases(const ChatbotTrainer& t) {
        return t.bleu_lowercases();
    }
    // TD-271 hook
    static void Split(ChatbotTrainer& t) {
        t.split_data();
    }

    static TrainingConfig quiet_config() {
        TrainingConfig cfg;
        cfg.log_level = LogLevel::SILENT;
        return cfg;
    }
};

// TD-267: pairs that tokenization skips are removed from the raw and tokenized vectors together.
TEST_F(ChatbotTrainerCacheTest, DropSkippedPairsKeepsRawAndTokenizedAligned) {
    ChatbotTrainer t(quiet_config());
    SetTrainingData(t, {ConversationPair("q1", "a1"), ConversationPair("", "skipped"),
                        ConversationPair("q3", "a3"), ConversationPair("bad utf8", "x")});
    SetValidationData(t, {ConversationPair("v1", ""), ConversationPair("v2", "b2")});
    SetTokenized(t,
                 {TokenizedPair({1}, {2}, "q1", "a1"), TokenizedPair(),
                  TokenizedPair({3}, {4}, "q3", "a3"), TokenizedPair()},
                 {TokenizedPair(), TokenizedPair({5}, {6}, "v2", "b2")});
    DropSkipped(t);

    const auto& raw = GetTrainingData(t);
    const auto& tok = GetTokenizedTraining(t);
    ASSERT_EQ(raw.size(), 2u);
    ASSERT_EQ(tok.size(), 2u);
    EXPECT_EQ(raw[0].input, "q1");
    EXPECT_EQ(tok[0].input_text, "q1");
    EXPECT_EQ(raw[1].input, "q3");
    EXPECT_EQ(tok[1].input_text, "q3");
    ASSERT_EQ(GetValidationData(t).size(), 1u);
    ASSERT_EQ(GetTokenizedValidation(t).size(), 1u);
    EXPECT_EQ(GetValidationData(t)[0].input, "v2");
}

// TD-267, end to end: skipped pairs never reach training or validation, and a validation sample
// that throws is excluded from the average instead of counting as loss 0.
TEST_F(ChatbotTrainerCacheTest, SkippedAndFailingPairsDontBiasValidationLoss) {
    Matrix::gpu_try_initialize();
    TrainingConfig cfg = quiet_config();
    cfg.d_model = 16;
    cfg.num_heads = 2;
    cfg.d_ff = 32;
    cfg.num_encoder_layers = 1;
    cfg.num_decoder_layers = 1;
    cfg.max_seq_length = 16;
    cfg.validation_split = 0;
    cfg.enable_early_stopping = false;
    ChatbotTrainer t(cfg);
    std::filesystem::create_directories(cache_dir);
    ASSERT_TRUE(t.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"}, 100,
                                   cache_dir + "/vocab.txt"));
    t.add_training_pair("hello world", "how are you");
    t.add_training_pair("I am fine", "");  // skipped: empty response
    t.add_training_pair("I am fine", "thank you");
    t.add_validation_pair("hello world", "thank you");
    t.add_validation_pair("", "how are you");  // skipped: empty input
    t.add_validation_pair("how are you", "I am fine");

    ASSERT_TRUE(t.train(1));
    EXPECT_EQ(t.get_tokenized_training_size(), 2u);
    EXPECT_EQ(t.get_tokenized_validation_size(), 2u);
    ASSERT_EQ(GetTrainingData(t).size(), 2u);
    for (std::size_t i = 0; i < GetTrainingData(t).size(); ++i)
        EXPECT_EQ(GetTrainingData(t)[i].input, GetTokenizedTraining(t)[i].input_text);

    const float clean = Validate(t);
    ASSERT_GT(clean, 0.0f);

    // A sample whose evaluation throws (token id far outside the vocabulary).
    AppendValidation(t, ConversationPair("broken", "broken"),
                     TokenizedPair({999999}, {999999}, "broken", "broken"));
    const float with_failure = Validate(t);
    EXPECT_NEAR(with_failure, clean, 1e-5f) << "a failed sample must not pull the average down";

    // Every sample failing records nothing and returns the previous loss, not 0.
    const int before = EpochsWithoutImprovement(t);
    const std::size_t recorded = t.get_validation_losses().size();
    SetTokenizedValidationOnly(t, {ConversationPair("broken", "broken")},
                               {TokenizedPair({999999}, {999999}, "broken", "broken")});
    EXPECT_NEAR(Validate(t), with_failure, 1e-6f);
    EXPECT_EQ(t.get_validation_losses().size(), recorded);
    EXPECT_EQ(EpochsWithoutImprovement(t), before + 1);
}

// TD-275, end to end: a response longer than max_seq_length is tokenized to exactly
// max_seq_length ids and still ends with <eos>; targets have no leading <bos> (TD-361).
TEST_F(ChatbotTrainerCacheTest, LongResponseTargetEndsWithEos) {
    Matrix::gpu_try_initialize();
    TrainingConfig cfg = quiet_config();
    cfg.d_model = 16;
    cfg.num_heads = 2;
    cfg.d_ff = 32;
    cfg.num_encoder_layers = 1;
    cfg.num_decoder_layers = 1;
    cfg.max_seq_length = 8;
    cfg.validation_split = 0;
    cfg.enable_early_stopping = false;
    ChatbotTrainer t(cfg);
    std::filesystem::create_directories(cache_dir);
    ASSERT_TRUE(t.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"}, 100,
                                   cache_dir + "/vocab.txt"));
    t.add_training_pair("hello world",
                        "how are you I am fine thank you how are you I am fine thank you");
    t.add_training_pair("hi", "thank you");
    ASSERT_TRUE(t.train(1));

    const auto& tok = GetTokenizedTraining(t);
    ASSERT_EQ(tok.size(), 2u);
    const int eos = 3;  // adai::SpecialTokenIDs::EOS
    EXPECT_EQ(tok[0].target_tokens.size(), 8u);
    EXPECT_NE(tok[0].target_tokens.front(), 2);  // no leading <bos> (TD-361)
    EXPECT_NE(tok[1].target_tokens.front(), 2);
    EXPECT_EQ(tok[0].target_tokens.back(), eos);
    EXPECT_EQ(tok[1].target_tokens.back(), eos);  // short target: unchanged, already ends on <eos>
}

// TD-272: a window holding fewer samples than gradient_accumulation_steps (here the epoch's only,
// 1-sample window with accumulation 4) reports the mean loss of the samples it holds, not a
// quarter of it, and its gradients are rescaled to match.
TEST_F(ChatbotTrainerCacheTest, ShortLastWindowUsesItsRealSampleCount) {
    Matrix::gpu_try_initialize();
    ChatbotTrainer t(tiny_training_config(4));
    std::filesystem::create_directories(cache_dir);
    ASSERT_TRUE(t.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"}, 100,
                                   cache_dir + "/vocab.txt"));
    t.add_training_pair("hello world", "how are you");
    ASSERT_TRUE(t.train(1));

    std::vector<float> step_losses;
    t.set_sample_callback([&](int, int, float, float step_loss, float grad_norm, float) {
        if (grad_norm > 0.0f)
            step_losses.push_back(step_loss);
    });
    TrainEpoch(t, 1);
    ASSERT_EQ(step_losses.size(), 1u);
    const float sample_loss = -std::log(GetTrainingData(t)[0].meta.quality);
    EXPECT_NEAR(step_losses[0], sample_loss, 1e-4f * std::max(1.0f, sample_loss));
}

// TD-272: a sample that fails before its backward pass no longer throws away the gradients the
// window already holds. With 4 good samples, 1 failing one and accumulation 4, every epoch takes
// exactly one step over the 4 good samples, wherever the shuffle puts the failing one. (Before,
// a failure mid-window discarded the earlier samples and the last window stepped on whatever was
// left.)
TEST_F(ChatbotTrainerCacheTest, FailedSampleKeepsTheWindowsEarlierGradients) {
    Matrix::gpu_try_initialize();
    ChatbotTrainer t(tiny_training_config(4));
    std::filesystem::create_directories(cache_dir);
    ASSERT_TRUE(t.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"}, 100,
                                   cache_dir + "/vocab.txt"));
    t.add_training_pair("hello world", "how are you");
    t.add_training_pair("how are you", "I am fine");
    t.add_training_pair("I am fine", "thank you");
    t.add_training_pair("thank you", "hello world");
    ASSERT_TRUE(t.train(1));
    AppendTraining(t, ConversationPair("broken", "broken"),
                   TokenizedPair({999999}, {999999}, "broken", "broken"));

    for (int epoch = 1; epoch <= 4; ++epoch) {
        std::vector<float> step_losses;
        t.set_sample_callback([&](int, int, float, float step_loss, float grad_norm, float) {
            if (grad_norm > 0.0f)
                step_losses.push_back(step_loss);
        });
        TrainEpoch(t, epoch);
        ASSERT_EQ(step_losses.size(), 1u) << "epoch " << epoch;
        float mean = 0.0f;
        for (int k = 0; k < 4; ++k)
            mean += -std::log(GetTrainingData(t)[k].meta.quality);
        mean /= 4.0f;
        EXPECT_NEAR(step_losses[0], mean, 1e-4f * std::max(1.0f, mean)) << "epoch " << epoch;
    }
}

// Every schedule stays finite for tiny passes and for steps carried past this call's total
// (TD-269): WARMUP_COSINE divided by total - warmup, which is 0 when they're equal.
TEST_F(ChatbotTrainerCacheTest, LearningRateIsFiniteForTinyAndContinuedRuns) {
    for (LRSchedule sched :
         {LRSchedule::CONSTANT, LRSchedule::LINEAR_WARMUP, LRSchedule::COSINE_DECAY,
          LRSchedule::WARMUP_COSINE, LRSchedule::STEP_DECAY, LRSchedule::EXPONENTIAL_DECAY}) {
        TrainingConfig cfg = quiet_config();
        cfg.lr_schedule = sched;
        cfg.num_epochs = 3;
        ChatbotTrainer t(cfg);
        for (int total : {0, 1, 2})
            for (int warmup : {0, 1, 2})
                for (int step : {0, 1, 5, 100}) {
                    const float lr = LearningRateAt(t, step, total, warmup);
                    EXPECT_TRUE(std::isfinite(lr)) << "schedule " << static_cast<int>(sched)
                                                   << " total " << total << " warmup " << warmup
                                                   << " step " << step;
                }
    }
}

// TD-270: adaptive clipping keeps its EMA and warmup counter across epochs (it restarted every
// epoch, so with few steps per epoch it never left warmup).
TEST_F(ChatbotTrainerCacheTest, AdaptiveClippingStateCarriesAcrossEpochs) {
    Matrix::gpu_try_initialize();
    TrainingConfig cfg = tiny_training_config(1);
    cfg.adaptive_gradient_clip = true;
    ChatbotTrainer t(cfg);
    std::filesystem::create_directories(cache_dir);
    ASSERT_TRUE(t.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"}, 100,
                                   cache_dir + "/vocab.txt"));
    t.add_training_pair("hello world", "how are you");
    t.add_training_pair("I am fine", "thank you");
    ASSERT_TRUE(t.train(3));
    EXPECT_EQ(AgcSteps(t), GlobalStep(t)) << "one count across all 3 epochs, not per epoch";
    EXPECT_EQ(GlobalStep(t), 6);
}

// TD-269: the training state (optimizer, run-level step, warmup, clip state) round-trips, and
// state from another run is not loaded.
TEST_F(ChatbotTrainerCacheTest, TrainingStateRoundTripsWithinARunOnly) {
    Matrix::gpu_try_initialize();
    std::filesystem::create_directories(cache_dir);
    const std::string vocab = cache_dir + "/vocab.txt";
    const std::string state = cache_dir + "/cp.bin.optimizer";
    ChatbotTrainer first(tiny_training_config(1));
    ASSERT_TRUE(first.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"},
                                       100, vocab));
    first.set_run_id("run-01");
    first.add_training_pair("hello world", "how are you");
    ASSERT_TRUE(first.train(2));
    ASSERT_TRUE(first.save_training_state(state));

    // Same architecture and vocab (the model is fresh; only the state is restored).
    ChatbotTrainer second(tiny_training_config(1));
    ASSERT_TRUE(second.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"},
                                        100, cache_dir + "/vocab2.txt"));
    InitModel(second);
    second.set_run_id("run-01");
    ASSERT_TRUE(second.load_training_state(state));
    EXPECT_EQ(GlobalStep(second), GlobalStep(first));
    EXPECT_EQ(RunWarmup(second), RunWarmup(first));

    ChatbotTrainer other_run(tiny_training_config(1));
    ASSERT_TRUE(other_run.build_vocabulary({"hello world", "how are you", "I am fine", "thank you"},
                                           100, cache_dir + "/vocab3.txt"));
    InitModel(other_run);
    other_run.set_run_id("run-02");
    EXPECT_FALSE(other_run.load_training_state(state)) << "a new run starts fresh";
    EXPECT_EQ(GlobalStep(other_run), 0);
}

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
    // See MetricsTrackerWiringTest::SetUp()'s own comment on why this call is needed before
    // constructing any ChatbotTrainer under a GPU-enabled build.
    Matrix::gpu_try_initialize();
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

// Regression test (TD-062): train_epoch()'s reported epoch_loss/avg_grad_norm
// used to divide by a `num_updates` re-derived from
// `global_step - epoch*(num_samples/accum_steps)`, which drifted by +1 every
// epoch whenever num_samples wasn't an exact multiple of
// gradient_accumulation_steps (confirmed by simulating the exact loop:
// 100 samples / 32 accum steps gave 4 true updates/epoch but the old formula
// reported 4, 5, 6, 7... across successive epochs). 5 pairs with
// accumulation_steps=3 gives 2 updates/epoch (a 3-sample window, then a
// force-flushed 2-sample window) — the same non-divisible shape that
// triggered the drift. This can't assert an exact expected loss value
// (that depends on the model's actual learned weights), but it does verify
// the fixed code path produces the correct update *count* across several
// epochs and never divides by a wrong/zero count (which would show up as a
// non-finite or clearly-wrong-magnitude loss).
TEST(ChatbotTrainerAbortTest, TrainingLossStaysSaneWithNonDivisibleAccumulationWindow) {
    TrainingConfig cfg;
    cfg.num_epochs = 4;
    cfg.gradient_accumulation_steps = 3;
    auto trainer = make_abort_test_trainer(
        cfg, 5, "/tmp/adai_chatbottrainer_abort_test_nondivisible_accum_vocab.txt");

    ASSERT_TRUE(trainer->train(4));
    // 5 samples / 3 accum steps = 2 updates/epoch (window sizes 3 then 2,
    // the second force-flushed at the last sample) — exact, not floor-divided.
    EXPECT_EQ(trainer->get_global_step(), 4 * 2);

    const auto& losses = trainer->get_training_losses();
    ASSERT_EQ(losses.size(), 4u);
    for (float loss : losses) {
        EXPECT_TRUE(std::isfinite(loss));
        EXPECT_GT(loss, 0.0f);
    }
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ============================================================================
// TD-271: the validation split is deterministic, and a tokenized cache only loads for its own split
// ============================================================================

namespace {
std::vector<ConversationPair> td271_pairs(int n) {
    std::vector<ConversationPair> pairs;
    for (int i = 0; i < n; ++i) {
        pairs.emplace_back("question number " + std::to_string(i), "answer number " + std::to_string(i));
    }
    return pairs;
}

// Each pair as "input\nresponse", so pairs sharing an input stay distinct.
std::set<std::string> inputs_of(const std::vector<ConversationPair>& pairs) {
    std::set<std::string> inputs;
    for (const auto& p : pairs) inputs.insert(p.input + "\n" + p.response);
    return inputs;
}
}  // namespace

TEST_F(ChatbotTrainerCacheTest, SplitIsTheSameEveryTimeAndIndependentOfLoadOrder) {
    TrainingConfig cfg = quiet_config();
    cfg.validation_split = 5;
    std::vector<ConversationPair> pairs = td271_pairs(40);

    ChatbotTrainer first(cfg);
    SetTrainingData(first, pairs);
    Split(first);
    ASSERT_EQ(GetValidationData(first).size(), 8u);
    ASSERT_EQ(GetTrainingData(first).size(), 32u);

    ChatbotTrainer again(cfg);
    SetTrainingData(again, pairs);
    Split(again);
    EXPECT_EQ(inputs_of(GetValidationData(again)), inputs_of(GetValidationData(first)));
    for (std::size_t i = 0; i < GetTrainingData(first).size(); ++i) {
        EXPECT_EQ(GetTrainingData(again)[i].input, GetTrainingData(first)[i].input) << "load order kept";
    }

    std::vector<ConversationPair> reversed(pairs.rbegin(), pairs.rend());
    ChatbotTrainer reordered(cfg);
    SetTrainingData(reordered, reversed);
    Split(reordered);
    EXPECT_EQ(inputs_of(GetValidationData(reordered)), inputs_of(GetValidationData(first)));
    EXPECT_EQ(inputs_of(GetTrainingData(reordered)), inputs_of(GetTrainingData(first)));
}

// The old code wrote a cache in one split and loaded it in another as long as the counts matched.
TEST_F(ChatbotTrainerCacheTest, CacheFromADifferentSplitIsNotLoaded) {
    const std::string path = cache_dir + "/other_split.cache";
    std::vector<ConversationPair> train = td271_pairs(3);
    std::vector<ConversationPair> val = {ConversationPair("held out", "kept aside")};
    auto tokenized = [](const std::vector<ConversationPair>& raw) {
        std::vector<TokenizedPair> out;
        for (const auto& p : raw) out.emplace_back(std::vector<int>{4, 5}, std::vector<int>{6, 3}, p.input, p.response);
        return out;
    };

    ChatbotTrainer writer(quiet_config());
    SetTrainingData(writer, train);
    SetValidationData(writer, val);
    SetTokenized(writer, tokenized(train), tokenized(val));
    SaveCache(writer, path);

    ChatbotTrainer same(quiet_config());
    SetTrainingData(same, train);
    SetValidationData(same, val);
    EXPECT_TRUE(LoadCache(same, path));

    // Same counts, but a training pair and the validation pair swapped places.
    std::vector<ConversationPair> other_train = {train[0], train[1], val[0]};
    std::vector<ConversationPair> other_val = {train[2]};
    ChatbotTrainer other(quiet_config());
    SetTrainingData(other, other_train);
    SetValidationData(other, other_val);
    EXPECT_FALSE(LoadCache(other, path));
    EXPECT_TRUE(GetTokenizedTraining(other).empty());

    // A skipped pair's empty placeholder matches whatever raw pair is at its index.
    std::vector<TokenizedPair> with_placeholder = tokenized(train);
    with_placeholder[1] = TokenizedPair();
    SetTokenized(writer, with_placeholder, tokenized(val));
    SaveCache(writer, path);
    EXPECT_TRUE(LoadCache(same, path));
}

// End to end: two passes over the same data with the cache on train on aligned raw/tokenized pairs.
TEST_F(ChatbotTrainerCacheTest, SecondPassWithCacheKeepsRawAndTokenizedPairsAligned) {
    Matrix::gpu_try_initialize();
    TrainingConfig cfg = tiny_training_config(1);
    cfg.validation_split = 4;
    cfg.cache_tokenized_data = true;
    cfg.tokenized_cache_dir = cache_dir;
    cfg.tokenized_cache_key = "td271";
    std::filesystem::create_directories(cache_dir);
    const std::vector<std::string> words = {"hello", "world", "how", "are", "you", "fine", "thanks",
                                            "good", "morning", "evening", "see", "later"};
    std::vector<std::pair<std::string, std::string>> pairs;
    for (std::size_t i = 0; i < words.size(); ++i) {  // 12 distinct pairs
        pairs.emplace_back(words[i % words.size()] + " " + words[(i * 5 + 1) % words.size()],
                           words[(i * 7 + 2) % words.size()] + " " + words[(i + 3) % words.size()]);
    }
    std::vector<std::string> corpus = words;

    std::set<std::string> first_validation;
    for (int pass = 0; pass < 2; ++pass) {
        ChatbotTrainer t(cfg);
        ASSERT_TRUE(t.build_vocabulary(corpus, 100, cache_dir + "/vocab.txt"));
        for (const auto& [in, out] : pairs) t.add_training_pair(in, out);
        ASSERT_TRUE(t.train(1));
        ASSERT_TRUE(std::filesystem::exists(cache_dir + "/td271.cache"));
        ASSERT_EQ(GetTrainingData(t).size(), GetTokenizedTraining(t).size());
        for (std::size_t i = 0; i < GetTrainingData(t).size(); ++i) {
            EXPECT_EQ(GetTokenizedTraining(t)[i].input_text, GetTrainingData(t)[i].input) << "pass " << pass;
        }
        for (std::size_t i = 0; i < GetValidationData(t).size(); ++i) {
            EXPECT_EQ(GetTokenizedValidation(t)[i].input_text, GetValidationData(t)[i].input) << "pass " << pass;
        }
        if (pass == 0) {
            first_validation = inputs_of(GetValidationData(t));
            ASSERT_EQ(first_validation.size(), 3u);
        } else {
            EXPECT_EQ(inputs_of(GetValidationData(t)), first_validation) << "same validation samples";
        }
    }
}

// TD-276: an architecture the model can't be built with is an error instead of being changed (it
// used to round d_model up, and reset an unusual d_ff to 4x, breaking an MNS-registered
// architecture).
TEST_F(ChatbotTrainerCacheTest, ArchitectureIsNeverSilentlyChanged) {
    Matrix::gpu_try_initialize();
    TrainingConfig bad = tiny_training_config(1);
    bad.d_model = 18;  // not a multiple of num_heads = 4
    bad.num_heads = 4;
    std::filesystem::create_directories(cache_dir);
    ChatbotTrainer trainer(bad);
    ASSERT_TRUE(
        trainer.build_vocabulary({"hello world", "how are you"}, 100, cache_dir + "/vocab.txt"));
    EXPECT_THROW(InitModel(trainer), std::invalid_argument);

    TrainingConfig narrow = tiny_training_config(1);
    narrow.d_model = 16;
    narrow.d_ff = 16;  // ratio 1: unusual, but buildable
    ChatbotTrainer kept(narrow);
    ASSERT_TRUE(
        kept.build_vocabulary({"hello world", "how are you"}, 100, cache_dir + "/vocab.txt"));
    ASSERT_NO_THROW(InitModel(kept));
    EXPECT_EQ(kept.get_config().d_ff, 16);
    EXPECT_EQ(kept.get_config().d_model, 16);
}

// TD-561: BLEU compares case-insensitively exactly when the model's tokenizer lowercases.
TEST_F(ChatbotTrainerCacheTest, BleuLowercasesOnlyForALowercaseVocabulary) {
    Matrix::gpu_try_initialize();
    std::filesystem::create_directories(cache_dir);
    ChatbotTrainer cased(tiny_training_config(1));
    ASSERT_TRUE(
        cased.build_vocabulary({"Hello World", "how are you"}, 100, cache_dir + "/cased.txt"));
    InitModel(cased);
    EXPECT_FALSE(BleuLowercases(cased));

    BPETokenizer lower;
    lower.set_normalization(TextNormalization::LOWERCASE);
    lower.build_vocab({"Hello World", "how are you"}, 100);
    lower.save_vocab(cache_dir + "/lower.txt");
    TrainingConfig cfg = tiny_training_config(1);
    ChatbotTrainer lowered(cfg);
    ASSERT_TRUE(lowered.load_tokenizer(cache_dir + "/lower.txt"));
    InitModel(lowered);
    EXPECT_TRUE(BleuLowercases(lowered));
}

// TD-295: training inputs have the format a served prompt has, earlier turns included.
TEST_F(ChatbotTrainerCacheTest, InputsUseTurnTokensAndHistory) {
    TrainingConfig cfg = tiny_training_config(1);
    ChatbotTrainer trainer(cfg);
    std::filesystem::create_directories(cache_dir);
    ASSERT_TRUE(
        trainer.build_vocabulary({"hello world", "how are you"}, 100, cache_dir + "/turns.txt"));
    InitModel(trainer);

    SampleMeta meta;
    meta.history = {{"system", "hello"}, {"user", "how"}, {"assistant", "are"}};
    std::vector<ConversationPair> data;
    data.emplace_back("hello world", "how are you");
    data.emplace_back("you", "world", meta);
    SetTrainingData(trainer, data);
    SetValidationData(trainer, {});
    Preprocess(trainer);

    const auto& tokenized = GetTokenizedTraining(trainer);
    ASSERT_EQ(tokenized.size(), 2u);
    using namespace adai::SpecialTokenIDs;
    const auto& single = tokenized[0].input_tokens;
    EXPECT_EQ(single.front(), USER_TURN);
    EXPECT_EQ(single.back(), ASSISTANT_TURN);
    EXPECT_EQ(std::count(single.begin(), single.end(), USER_TURN), 1);

    const auto& multi = tokenized[1].input_tokens;
    EXPECT_EQ(multi.front(), SYSTEM_TURN);
    EXPECT_EQ(std::count(multi.begin(), multi.end(), USER_TURN), 2);
    EXPECT_EQ(std::count(multi.begin(), multi.end(), ASSISTANT_TURN), 2);
    EXPECT_EQ(multi.back(), ASSISTANT_TURN);
    for (int id : tokenized[1].target_tokens) {
        EXPECT_FALSE(id >= SYSTEM_TURN && id <= ASSISTANT_TURN) << "targets are the reply alone";
    }
}
