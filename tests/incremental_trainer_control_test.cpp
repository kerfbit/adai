/**
 * @file incremental_trainer_control_test.cpp
 * @brief Proves IncrementalTrainer actually consults an attached
 *        TrainerControlState's live-tunable auto_save_* / max_sessions_to_keep
 *        values (set via PUT /admin/config at runtime) in preference to the
 *        config-file values baked in at construction — and that behavior is
 *        unchanged when no control state is attached (every non-`serve`
 *        command).
 *
 * Uses the same friend-class pattern as ChatbotTrainerCacheTest
 * (chatbottrainer_test.cpp) to reach should_auto_save()/cleanup_old_sessions()
 * and the private members they read, without needing a real training pass.
 * As with that fixture, every private-member access must go through a static
 * helper *defined inside* the friended fixture class — a TEST_F body is a
 * method of a class *derived* from the fixture, and friendship is not
 * inherited, so accessing private members directly from TestBody() (even
 * though it's "inside" the TEST_F for the friended fixture) does not
 * type-check.
 */

#include "IncrementalTrainer.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// Declared as a friend of IncrementalTrainer (IncrementalTrainer.hpp).
class IncrementalTrainerControlTest : public ::testing::Test {
   protected:
    fs::path test_dir;
    fs::path vocab_file;
    fs::path model_file;

    void SetUp() override {
        test_dir = fs::temp_directory_path() /
                   ("incremental_trainer_control_test_" + std::to_string(::getpid()));
        fs::create_directories(test_dir);
        vocab_file = test_dir / "vocab.txt";
        model_file = test_dir / "model.bin";
        write_minimal_vocab(vocab_file.string());
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(test_dir, ec);
    }

    static void write_minimal_vocab(const std::string& path) {
        std::ofstream f(path);
        f << "# BPE Tokenizer Vocabulary v1.0\n";
        f << "VOCAB_SIZE 8\n";
        f << "SPECIAL_TOKENS\n";
        f << "pad_token_id 0\n";
        f << "unk_token_id 1\n";
        f << "bos_token_id 2\n";
        f << "eos_token_id 3\n";
        f << "VOCAB\n";
        f << "<pad>\t0\n<unk>\t1\n<bos>\t2\n<eos>\t3\n";
        const char* chars = "abcd";
        for (int i = 0; i < 4; ++i)
            f << chars[i] << "\t" << (i + 4) << "\n";
    }

    IncrementalConfig make_config(int session_dir_suffix) {
        IncrementalConfig cfg;
        cfg.session_dir = (test_dir / ("sessions_" + std::to_string(session_dir_suffix))).string();
        fs::create_directories(cfg.session_dir);
        cfg.base_config.d_model = 16;
        cfg.base_config.num_heads = 2;
        cfg.base_config.d_ff = 32;
        cfg.base_config.num_encoder_layers = 1;
        cfg.base_config.num_decoder_layers = 1;
        cfg.base_config.max_seq_length = 16;
        return cfg;
    }

    static TrainingSession fake_session(int id, const std::string& checkpoint_path) {
        TrainingSession s;
        s.session_id = id;
        s.samples_trained = 10;
        s.epochs_completed = 1;
        s.final_loss = 1.0f;
        s.final_validation_loss = 1.0f;
        s.checkpoint_path = checkpoint_path;
        return s;
    }

    // ── Friend-only accessors — every one of these must be *defined* inside
    //    this class body (not just declared) for the access check below to
    //    resolve against the friended fixture rather than a derived TEST_F
    //    class. ──────────────────────────────────────────────────────────
    static void SetSamplesSinceLastSave(IncrementalTrainer& t, int v) {
        t.samples_since_last_save = v;
    }
    static bool ShouldAutoSave(IncrementalTrainer& t) {
        return t.should_auto_save();
    }
    static void SetConfigAutoSaveEnabled(IncrementalTrainer& t, bool v) {
        t.config.auto_save_enabled = v;
    }
    static void PushFakeSession(IncrementalTrainer& t, const TrainingSession& s) {
        t.session_history.push_back(s);
    }
    static size_t SessionHistorySize(const IncrementalTrainer& t) {
        return t.session_history.size();
    }
    static int SessionHistoryFrontId(const IncrementalTrainer& t) {
        return t.session_history.front().session_id;
    }
    static int SessionHistoryBackId(const IncrementalTrainer& t) {
        return t.session_history.back().session_id;
    }
    static void CallCleanupOldSessions(IncrementalTrainer& t) {
        t.cleanup_old_sessions();
    }
    static void CallPerformAutoSave(IncrementalTrainer& t, int epoch, int samples) {
        t.perform_auto_save(epoch, samples);
    }
};

TEST_F(IncrementalTrainerControlTest, ShouldAutoSaveFallsBackToConfigWithNoControlAttached) {
    IncrementalConfig cfg = make_config(1);
    cfg.auto_save_enabled = true;
    cfg.auto_save_every_samples = 5;
    cfg.auto_save_every_minutes = 0;  // isolate the sample-count trigger
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), cfg);

    SetSamplesSinceLastSave(trainer, 4);
    EXPECT_FALSE(ShouldAutoSave(trainer));

    SetSamplesSinceLastSave(trainer, 5);
    EXPECT_TRUE(ShouldAutoSave(trainer));
}

TEST_F(IncrementalTrainerControlTest, ShouldAutoSavePrefersControlStateTunablesWhenAttached) {
    IncrementalConfig cfg = make_config(2);
    cfg.auto_save_enabled = true;
    cfg.auto_save_every_samples = 5;
    cfg.auto_save_every_minutes = 0;
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), cfg);

    auto control = std::make_shared<adai::TrainerControlState>();
    control->auto_save_enabled = true;
    control->auto_save_every_samples = 100;  // stricter than config's 5
    control->auto_save_every_minutes = 0;
    trainer.set_control_state(control);

    // Would have fired under the config-file threshold (5) but the attached
    // control state's live value (100) takes precedence.
    SetSamplesSinceLastSave(trainer, 5);
    EXPECT_FALSE(ShouldAutoSave(trainer));

    SetSamplesSinceLastSave(trainer, 100);
    EXPECT_TRUE(ShouldAutoSave(trainer));
}

TEST_F(IncrementalTrainerControlTest, ShouldAutoSaveHonorsControlStateDisabledOverConfigEnabled) {
    IncrementalConfig cfg = make_config(3);
    cfg.auto_save_enabled = true;
    cfg.auto_save_every_samples = 1;
    cfg.auto_save_every_minutes = 0;
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), cfg);

    auto control = std::make_shared<adai::TrainerControlState>();
    control->auto_save_enabled = false;  // PUT /admin/config {"auto_save_enabled":false}
    trainer.set_control_state(control);

    SetSamplesSinceLastSave(trainer, 1000);
    EXPECT_FALSE(ShouldAutoSave(trainer));
}

TEST_F(IncrementalTrainerControlTest, CleanupOldSessionsFallsBackToConfigWithNoControlAttached) {
    IncrementalConfig cfg = make_config(4);
    cfg.max_sessions_to_keep = 2;
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), cfg);

    for (int i = 1; i <= 5; ++i) {
        PushFakeSession(trainer,
                        fake_session(i, (test_dir / ("nonexistent_checkpoint_" +
                                                     std::to_string(i)))
                                            .string()));
    }

    CallCleanupOldSessions(trainer);
    EXPECT_EQ(SessionHistorySize(trainer), 2u);
    // The oldest entries (front of the vector) are the ones removed.
    EXPECT_EQ(SessionHistoryFrontId(trainer), 4);
    EXPECT_EQ(SessionHistoryBackId(trainer), 5);
}

TEST_F(IncrementalTrainerControlTest, CleanupOldSessionsPrefersControlStateMaxSessionsToKeep) {
    IncrementalConfig cfg = make_config(5);
    cfg.max_sessions_to_keep = 50;  // config would keep everything
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), cfg);

    auto control = std::make_shared<adai::TrainerControlState>();
    control->max_sessions_to_keep = 1;  // PUT /admin/config {"max_sessions_to_keep":1}
    trainer.set_control_state(control);

    for (int i = 1; i <= 3; ++i) {
        PushFakeSession(trainer,
                        fake_session(i, (test_dir / ("nonexistent_checkpoint_" +
                                                     std::to_string(i)))
                                            .string()));
    }

    CallCleanupOldSessions(trainer);
    EXPECT_EQ(SessionHistorySize(trainer), 1u);
    EXPECT_EQ(SessionHistoryFrontId(trainer), 3);
}

TEST_F(IncrementalTrainerControlTest, SetControlStateAcceptsNullToDetach) {
    IncrementalConfig cfg = make_config(6);
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), cfg);
    trainer.set_control_state(std::make_shared<adai::TrainerControlState>());
    trainer.set_control_state(nullptr);
    // No crash, and behavior reverts to the config-file fallback path.
    SetConfigAutoSaveEnabled(trainer, false);
    EXPECT_FALSE(ShouldAutoSave(trainer));
}

TEST_F(IncrementalTrainerControlTest, PerformAutoSaveLogsThroughAttachedControlState) {
    IncrementalConfig cfg = make_config(7);
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), cfg);

    auto control = std::make_shared<adai::TrainerControlState>();
    trainer.set_control_state(control);
    EXPECT_TRUE(control->recent_logs().empty());

    CallPerformAutoSave(trainer, /*epoch=*/1, /*samples=*/10);

    const auto entries = control->recent_logs();
    ASSERT_FALSE(entries.empty());
    const auto& last = entries.back();
    EXPECT_EQ(last.level, adai::TrainerLogLevel::Info);
    EXPECT_NE(last.message.find("Checkpoint saved:"), std::string::npos);
    // checkpoints_written/last_checkpoint_time_unix are updated by the same
    // code path — confirms this isn't just a log-only side effect.
    EXPECT_EQ(control->checkpoints_written.load(), 1);
    EXPECT_FALSE(control->get_last_checkpoint_path().empty());
}
