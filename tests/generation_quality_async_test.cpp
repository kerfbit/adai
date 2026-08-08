/**
 * @file generation_quality_async_test.cpp
 * @brief TD-023: Parallel Generation Quality Scoring via Model Snapshot — unit tests
 *
 * Covers:
 *   - TrainingConfig::generation_quality_async_threshold default value (50)
 *   - Config threshold can be modified
 *   - ServiceConfig field exists and defaults to 50
 *   - Below-threshold path uses synchronous scoring (no crash, reporter called)
 *   - At-or-above-threshold path uses async scoring (no crash, reporter eventually called)
 *   - Thread join before second epoch (reporter called at least once per epoch over 2 epochs)
 *   - Destructor-join safety when thread may still be running
 *   - NullMetricsReporter no-crash path (async threshold met, null reporter → early return)
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../src/ChatbotTrainer.hpp"
#include "../src/Config.hpp"
#include "../src/IMetricsReporter.hpp"

// ============================================================================
// Helpers
// ============================================================================

/**
 * @brief Minimal IMetricsReporter that records update_generation_quality_metrics() calls.
 */
class RecordingMetricsReporter final : public IMetricsReporter {
   public:
    std::atomic<int> generation_quality_call_count{0};
    mutable std::mutex score_mutex;
    float last_bleu4 = -1.0f;

    void start_epoch(int, int) override {}
    void end_epoch(int, float, float, float, float, float, double) override {}
    void update_sample_metrics(int, float, float, float) override {}
    void update_validation_metrics(float, float, float) override {}
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
    void update_generation_quality_metrics(float bleu4, float, float, float) override {
        std::lock_guard<std::mutex> lk(score_mutex);
        last_bleu4 = bleu4;
        generation_quality_call_count.fetch_add(1, std::memory_order_relaxed);
    }
};

/**
 * @brief Build a minimal, valid ChatbotTrainer that can actually run train().
 *
 * Uses a tiny model (d_model=8, 1 head, 1 layer) to keep tests fast.
 * Vocabulary is built in-memory from the supplied training texts.
 *
 * @param cfg              Base TrainingConfig — caller may adjust fields before passing.
 * @param num_train_pairs  Number of "hello" / "world" training pairs to inject.
 * @param num_val_pairs    Number of validation pairs to inject.
 * @param vocab_tmp_path   Path for the temporary vocab file (test-unique name).
 * @return Initialised ChatbotTrainer with vocabulary and data loaded.
 */
static std::unique_ptr<ChatbotTrainer> make_tiny_trainer(TrainingConfig cfg, int num_train_pairs,
                                                         int num_val_pairs,
                                                         const std::string& vocab_tmp_path) {
    // Ensure tiny-model dims are valid
    cfg.d_model = 8;
    cfg.num_heads = 2;  // d_model (8) divisible by num_heads (2)
    cfg.d_ff = 32;
    cfg.num_encoder_layers = 1;
    cfg.num_decoder_layers = 1;
    cfg.max_seq_length = 16;
    cfg.log_level = LogLevel::SILENT;
    cfg.validation_split = 0;  // we inject validation data explicitly
    cfg.num_epochs = 1;
    cfg.generation_quality_max_tokens = 4;  // very short — keeps generation fast

    auto trainer = std::make_unique<ChatbotTrainer>(cfg);

    std::vector<std::string> corpus = {"hello world foo bar baz", "the quick brown fox",
                                       "test data sample text"};
    trainer->build_vocabulary(corpus, 50, vocab_tmp_path);

    for (int i = 0; i < num_train_pairs; ++i) {
        trainer->add_training_pair("hello", "world");
    }
    for (int i = 0; i < num_val_pairs; ++i) {
        trainer->add_validation_pair("hello", "world");
    }

    return trainer;
}

// ============================================================================
// Config default / mutation tests (no model needed)
// ============================================================================

TEST(GenerationQualityAsync, TrainingConfig_ThresholdDefault) {
    TrainingConfig cfg;
    EXPECT_EQ(cfg.generation_quality_async_threshold, 50);
}

TEST(GenerationQualityAsync, TrainingConfig_ThresholdCanBeModified) {
    TrainingConfig cfg;
    cfg.generation_quality_async_threshold = 25;
    EXPECT_EQ(cfg.generation_quality_async_threshold, 25);
}

TEST(GenerationQualityAsync, TrainingConfig_ThresholdZeroMeansAlwaysAsync) {
    // Setting threshold to 0 means even a single sample triggers the async path.
    TrainingConfig cfg;
    cfg.generation_quality_async_threshold = 0;
    EXPECT_EQ(cfg.generation_quality_async_threshold, 0);
}

TEST(GenerationQualityAsync, ServiceConfig_ThresholdFieldExists) {
    adai::ServiceConfig svc;
    EXPECT_EQ(svc.generation_quality_async_threshold, 50);
}

TEST(GenerationQualityAsync, ServiceConfig_ThresholdCanBeModified) {
    adai::ServiceConfig svc;
    svc.generation_quality_async_threshold = 100;
    EXPECT_EQ(svc.generation_quality_async_threshold, 100);
}

// ============================================================================
// NullMetricsReporter no-crash path
// ============================================================================

TEST(GenerationQualityAsync, NullReporter_AsyncThresholdMet_NoCrash) {
    // metrics_reporter_ == nullptr → compute_generation_quality_metrics() is a no-op.
    // The async branch must not crash or leak a thread.
    TrainingConfig cfg;
    cfg.enable_generation_quality_metrics = true;
    cfg.generation_quality_sample_size = 50;
    cfg.generation_quality_async_threshold = 50;

    auto trainer = make_tiny_trainer(cfg, 6, 3, "/tmp/adai_async_test_null_vocab.txt");
    // No metrics reporter set — default is nullptr.
    EXPECT_TRUE(trainer->train(1));  // must complete without crash or assert
}

// ============================================================================
// Below-threshold: synchronous path
// ============================================================================

TEST(GenerationQualityAsync, BelowThreshold_SyncPath_ReporterCalled) {
    // sample_size=3 < threshold=50 → synchronous path; reporter called before train() returns.
    TrainingConfig cfg;
    cfg.enable_generation_quality_metrics = true;
    cfg.generation_quality_sample_size = 3;
    cfg.generation_quality_async_threshold = 50;

    auto trainer = make_tiny_trainer(cfg, 6, 3, "/tmp/adai_async_test_sync_vocab.txt");

    RecordingMetricsReporter reporter;
    trainer->set_metrics_reporter(&reporter);

    ASSERT_TRUE(trainer->train(1));
    EXPECT_GE(reporter.generation_quality_call_count.load(), 1);
}

// ============================================================================
// At-threshold: async path
// ============================================================================

TEST(GenerationQualityAsync, AtThreshold_AsyncPath_ReporterCalledAfterJoin) {
    // sample_size=3, threshold=3 → async path (3 >= 3).
    // After train() returns the thread may still be running; the destructor
    // (or explicit join) must ensure the reporter is eventually called.
    TrainingConfig cfg;
    cfg.enable_generation_quality_metrics = true;
    cfg.generation_quality_sample_size = 3;
    cfg.generation_quality_async_threshold = 3;  // threshold == sample_size → async

    auto trainer = make_tiny_trainer(cfg, 6, 3, "/tmp/adai_async_test_async_vocab.txt");

    RecordingMetricsReporter reporter;
    trainer->set_metrics_reporter(&reporter);

    ASSERT_TRUE(trainer->train(1));

    // Destroy trainer — destructor must join the background thread before returning.
    trainer.reset();

    // Thread joined by destructor: reporter must have been called.
    EXPECT_GE(reporter.generation_quality_call_count.load(), 1);
}

// ============================================================================
// Thread join before second epoch
// ============================================================================

TEST(GenerationQualityAsync, TwoEpochs_ThreadJoinedBeforeSecondEpoch) {
    // Run two epochs with the async path.  The implementation must join the
    // first epoch's scoring thread before launching the second epoch's thread.
    // If both threads ran concurrently and both tried to use the same reporter,
    // the atomic counter would still be safe.  What we verify here is that
    // train(2) completes without deadlock, crash, or data race.
    TrainingConfig cfg;
    cfg.enable_generation_quality_metrics = true;
    cfg.generation_quality_sample_size = 3;
    cfg.generation_quality_async_threshold = 3;
    cfg.num_epochs = 2;

    auto trainer = make_tiny_trainer(cfg, 6, 3, "/tmp/adai_async_test_2epoch_vocab.txt");

    RecordingMetricsReporter reporter;
    trainer->set_metrics_reporter(&reporter);

    ASSERT_TRUE(trainer->train(2));

    // Destroy trainer to join any outstanding thread.
    trainer.reset();

    // Two validation phases → reporter called once per epoch.
    EXPECT_GE(reporter.generation_quality_call_count.load(), 2);
}

// ============================================================================
// Destructor safety: destroy trainer while thread is still running
// ============================================================================

TEST(GenerationQualityAsync, DestructorJoins_NoUseAfterFree) {
    // Build trainer in a nested scope; destroy it while the scoring thread
    // may be live.  The destructor must join safely.
    TrainingConfig cfg;
    cfg.enable_generation_quality_metrics = true;
    cfg.generation_quality_sample_size = 3;
    cfg.generation_quality_async_threshold = 3;

    RecordingMetricsReporter reporter;

    {
        auto trainer = make_tiny_trainer(cfg, 6, 3, "/tmp/adai_async_test_dtor_vocab.txt");
        trainer->set_metrics_reporter(&reporter);
        ASSERT_TRUE(trainer->train(1));
        // trainer destroyed here — destructor must join before ~RecordingMetricsReporter runs
    }

    // If we reach here without ASAN/TSAN errors, the join was safe.
    SUCCEED();
}

// ============================================================================
// Sync / async score range equivalence
// ============================================================================

TEST(GenerationQualityAsync, SyncAndAsync_ScoresInValidRange) {
    // Verify that both paths produce BLEU-4 values in [0, 1].
    // (Exact equivalence is not guaranteed because the model is randomly
    //  initialised, but the range is always valid.)
    auto run_and_get_bleu = [](int threshold, const std::string& vocab_path) -> float {
        TrainingConfig cfg;
        cfg.enable_generation_quality_metrics = true;
        cfg.generation_quality_sample_size = 3;
        cfg.generation_quality_async_threshold = threshold;

        auto trainer = make_tiny_trainer(cfg, 6, 3, vocab_path);

        RecordingMetricsReporter reporter;
        trainer->set_metrics_reporter(&reporter);

        trainer->train(1);
        trainer.reset();  // join async thread if needed

        return reporter.last_bleu4;
    };

    float sync_bleu = run_and_get_bleu(50, "/tmp/adai_async_test_sync2_vocab.txt");
    float async_bleu = run_and_get_bleu(3, "/tmp/adai_async_test_async2_vocab.txt");

    // Scores must be in valid range
    EXPECT_GE(sync_bleu, 0.0f);
    EXPECT_LE(sync_bleu, 1.0f);
    EXPECT_GE(async_bleu, 0.0f);
    EXPECT_LE(async_bleu, 1.0f);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
