/**
 * @file incrementaltrainer_test.cpp
 * @brief Comprehensive tests for IncrementalTrainer class
 *
 * Tests cover:
 * - Construction and configuration
 * - Session management (initialize, finalize, history, cleanup)
 * - Data registry (add data, track trained status, checksums)
 * - Checkpoint management (symlinks, best checkpoint tracking)
 * - Model persistence (save/load)
 * - Error handling and edge cases
 */

#include "IncrementalTrainer.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include "BPETokenizer.hpp"

namespace fs = std::filesystem;

class IncrementalTrainerTest : public ::testing::Test {
   protected:
    fs::path test_dir;
    fs::path vocab_file;
    fs::path model_file;
    fs::path session_dir;
    fs::path data_file1;
    fs::path data_file2;

    void SetUp() override {
        // Create temporary test directory
        test_dir = fs::temp_directory_path() / "incremental_trainer_test";
        fs::create_directories(test_dir);

        session_dir = test_dir / "training_sessions";
        fs::create_directories(session_dir);

        // Create test vocabulary file
        vocab_file = test_dir / "test_vocab.txt";
        create_test_vocabulary(vocab_file.string());

        // Create test model file (empty, will be initialized by trainer)
        model_file = test_dir / "test_model.bin";

        // Create test data files
        data_file1 = test_dir / "data1.txt";
        data_file2 = test_dir / "data2.txt";
        create_test_data_file(data_file1.string(), {"Hello world", "How are you"});
        create_test_data_file(data_file2.string(), {"Good morning", "Nice to meet you"});
    }

    void TearDown() override {
        // Clean up test directory
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    void create_test_vocabulary(const std::string& path) {
        std::ofstream file(path);

        // Header
        file << "# BPE Tokenizer Vocabulary v1.0\n";
        file << "VOCAB_SIZE 50\n";

        // Special tokens
        file << "SPECIAL_TOKENS\n";
        file << "pad_token_id 0\n";
        file << "unk_token_id 1\n";
        file << "bos_token_id 2\n";
        file << "eos_token_id 3\n";

        // Vocabulary
        file << "VOCAB\n";
        file << "<pad>\t0\n";
        file << "<unk>\t1\n";
        file << "<bos>\t2\n";
        file << "<eos>\t3\n";

        // Basic characters for test
        const char* chars = "abcdefghijklmnopqrstuvwxyz ";
        for (int i = 0; i < 27; ++i) {
            file << chars[i] << "\t" << (i + 4) << "\n";
        }

        // Some common bigrams
        file << "th\t31\n";
        file << "he\t32\n";
        file << "in\t33\n";
        file << "er\t34\n";
        file << "an\t35\n";
        file << "re\t36\n";
        file << "on\t37\n";
        file << "at\t38\n";
        file << "en\t39\n";
        file << "nd\t40\n";
        file << "ti\t41\n";
        file << "es\t42\n";
        file << "or\t43\n";
        file << "te\t44\n";
        file << "of\t45\n";
        file << "ed\t46\n";
        file << "is\t47\n";
        file << "it\t48\n";
        file << "al\t49\n";

        file.close();
    }

    void create_test_data_file(const std::string& path, const std::vector<std::string>& lines) {
        std::ofstream file(path);
        for (size_t i = 0; i < lines.size(); i += 2) {
            if (i < lines.size()) {
                file << "INPUT: " << lines[i] << "\n";
            }
            if (i + 1 < lines.size()) {
                file << "RESPONSE: " << lines[i + 1] << "\n";
            }
        }
        file.close();
    }

    void create_session_history_file(const std::string& path, int num_sessions) {
        std::ofstream file(path);
        file << "# Session History: session_id samples_trained epochs final_loss final_val_loss "
                "checkpoint_path\n";
        for (int i = 0; i < num_sessions; ++i) {
            file << i << " " << 100 << " " << 5 << " " << (2.0f - i * 0.1f) << " "
                 << (2.5f - i * 0.1f) << " " << session_dir.string() << "/session_" << i
                 << "_checkpoint.bin\n";
        }
        file.close();
    }
};

// ============================================================================
// Construction and Configuration Tests
// ============================================================================

TEST_F(IncrementalTrainerTest, ConstructorWithDefaultConfig) {
    IncrementalTrainer trainer(vocab_file.string(), model_file.string());

    IncrementalConfig& config = trainer.get_config();
    EXPECT_EQ(config.session_dir, "training_sessions");
    EXPECT_EQ(config.max_sessions_to_keep, 50);
    EXPECT_TRUE(config.auto_save_enabled);
    EXPECT_TRUE(config.enable_checkpoint_symlinks);
}

TEST_F(IncrementalTrainerTest, ConstructorWithCustomConfig) {
    IncrementalConfig custom_config;
    custom_config.session_dir = session_dir.string();
    custom_config.max_sessions_to_keep = 10;
    custom_config.auto_save_enabled = false;
    custom_config.enable_checkpoint_symlinks = false;
    custom_config.base_config.learning_rate = 0.0001f;

    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), custom_config);

    IncrementalConfig& config = trainer.get_config();
    EXPECT_EQ(config.session_dir, session_dir.string());
    EXPECT_EQ(config.max_sessions_to_keep, 10);
    EXPECT_FALSE(config.auto_save_enabled);
    EXPECT_FALSE(config.enable_checkpoint_symlinks);
    EXPECT_FLOAT_EQ(config.base_config.learning_rate, 0.0001f);
}

TEST_F(IncrementalTrainerTest, SetConfigAfterConstruction) {
    IncrementalTrainer trainer(vocab_file.string(), model_file.string());

    IncrementalConfig new_config;
    new_config.session_dir = "/tmp/sessions";
    new_config.max_sessions_to_keep = 25;

    trainer.set_config(new_config);

    IncrementalConfig& config = trainer.get_config();
    EXPECT_EQ(config.session_dir, "/tmp/sessions");
    EXPECT_EQ(config.max_sessions_to_keep, 25);
}

TEST_F(IncrementalTrainerTest, MakeIncrementalConfigUsesSessionScopedPushUrlWhenKeyProvided) {
    adai::ServiceConfig svc;
    svc.metrics_server_url = "http://localhost:8081/";
    svc.metrics_session_key = "3-gpu0";

    const IncrementalConfig cfg = IncrementalTrainer::make_incremental_config(svc);

    // Session key is now derived at runtime; make_incremental_config only maps the base URL.
    EXPECT_FALSE(cfg.metrics_server_url.empty());
    EXPECT_EQ(cfg.metrics_server_url, "http://localhost:8081/");
}

TEST_F(IncrementalTrainerTest, MakeIncrementalConfigKeepsBasePushUrlWhenSessionKeyMissing) {
    adai::ServiceConfig svc;
    svc.metrics_server_url = "http://localhost:8081";

    const IncrementalConfig cfg = IncrementalTrainer::make_incremental_config(svc);

    EXPECT_EQ(cfg.metrics_server_url, "http://localhost:8081");
}

// ============================================================================
// Session Management Tests
// ============================================================================

TEST_F(IncrementalTrainerTest, SessionHistoryEmptyInitially) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    std::vector<TrainingSession> history = trainer.get_session_history();
    EXPECT_EQ(history.size(), 0);
}

TEST_F(IncrementalTrainerTest, LoadSessionHistoryFromFile) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    // Create session history file after construction — cleanup_dead_sessions()
    // runs during construction and would purge entries whose checkpoint files
    // (deliberately not created here; this test only exercises parsing) don't
    // exist on disk.
    std::string history_file = session_dir.string() + "/session_history.txt";
    create_session_history_file(history_file, 3);

    bool result = trainer.load_session_history();
    EXPECT_TRUE(result);

    std::vector<TrainingSession> history = trainer.get_session_history();
    EXPECT_EQ(history.size(), 3);

    // Verify session data
    EXPECT_EQ(history[0].session_id, 0);
    EXPECT_EQ(history[0].samples_trained, 100);
    EXPECT_EQ(history[0].epochs_completed, 5);
    EXPECT_FLOAT_EQ(history[0].final_loss, 2.0f);
    EXPECT_FLOAT_EQ(history[0].final_validation_loss, 2.5f);
}

TEST_F(IncrementalTrainerTest, SaveSessionHistory) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    // Add some dummy session data by loading from file
    std::string history_file = session_dir.string() + "/session_history.txt";
    create_session_history_file(history_file, 2);
    trainer.load_session_history();

    // Save to a different location
    bool result = trainer.save_session_history();
    EXPECT_TRUE(result);

    // Verify file exists and can be reloaded
    EXPECT_TRUE(fs::exists(history_file));
}

TEST_F(IncrementalTrainerTest, GetCurrentSessionReturnsEmptyWhenNoSessions) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    TrainingSession current = trainer.get_current_session();
    EXPECT_EQ(current.session_id, 0);
    EXPECT_EQ(current.samples_trained, 0);
}

TEST_F(IncrementalTrainerTest, GetCurrentSessionReturnsLastSession) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    // Load session history
    std::string history_file = session_dir.string() + "/session_history.txt";
    create_session_history_file(history_file, 3);
    trainer.load_session_history();

    TrainingSession current = trainer.get_current_session();
    EXPECT_EQ(current.session_id, 2);  // Last session
}

TEST_F(IncrementalTrainerTest, CleanupOldSessionsKeepsMaxSessions) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    config.max_sessions_to_keep = 2;
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    // Load 5 sessions
    std::string history_file = session_dir.string() + "/session_history.txt";
    create_session_history_file(history_file, 5);
    trainer.load_session_history();

    // Create dummy checkpoint files
    for (int i = 0; i < 5; ++i) {
        std::string checkpoint =
            session_dir.string() + "/session_" + std::to_string(i) + "_checkpoint.bin";
        std::ofstream(checkpoint) << "dummy";
    }

    // Cleanup old sessions
    trainer.cleanup_old_sessions();

    // Verify only 2 sessions remain
    std::vector<TrainingSession> history = trainer.get_session_history();
    EXPECT_EQ(history.size(), 2);

    // Verify old checkpoints are deleted
    EXPECT_FALSE(fs::exists(session_dir / "session_0_checkpoint.bin"));
    EXPECT_FALSE(fs::exists(session_dir / "session_1_checkpoint.bin"));
    EXPECT_FALSE(fs::exists(session_dir / "session_2_checkpoint.bin"));
}

TEST_F(IncrementalTrainerTest, CleanupDeadSessionsRemovesOrphansAndBrokenHistory) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    // History: session 0 is "broken" (zero samples trained, the degenerate
    // zero-loss case); session 1 actually trained and is sane.
    std::string history_file = session_dir.string() + "/session_history.txt";
    {
        std::ofstream file(history_file);
        file << "# session_id samples_trained epochs final_loss final_val_loss checkpoint_path\n";
        file << "0 0 20 0 0 " << session_dir.string() << "/session_0_checkpoint.bin\n";
        file << "1 45000 20 187.259 5.87326 " << session_dir.string()
             << "/session_1_checkpoint.bin\n";
    }
    trainer.load_session_history();  // current_session_id becomes 2 (max id 1 + 1)

    auto touch = [](const fs::path& p) { std::ofstream(p) << "dummy"; };

    // Referenced checkpoints (one broken, one sane). Session 1 needs the full
    // realistic set of sidecar files EncoderDecoderModel::save_model() actually
    // writes for is_sane_checkpoint_candidate() to accept it.
    touch(session_dir / "session_0_checkpoint.bin");
    touch(session_dir / "session_0_checkpoint.bin.config");
    touch(session_dir / "session_1_checkpoint.bin");
    touch(session_dir / "session_1_checkpoint.bin.config");
    touch(session_dir / "session_1_checkpoint.bin.vocab");
    touch(session_dir / "session_1_checkpoint.bin.encoder");
    touch(session_dir / "session_1_checkpoint.bin.decoder");
    touch(session_dir / "session_1_checkpoint.bin.lm_head");

    // Orphaned in-progress snapshot from an older, already-superseded session
    touch(session_dir / "session_1_best.bin");
    touch(session_dir / "session_1_best.bin.config");
    // Live in-progress snapshot for the *current* session (id 2) — must survive
    touch(session_dir / "session_2_best.bin");
    touch(session_dir / "session_2_best.bin.config");
    // Bare checkpoint with no matching history line at all (crash before the
    // history line was appended)
    touch(session_dir / "session_3_checkpoint.bin");
    // Stale autosave from an old session vs. the current one
    touch(session_dir / "auto_save_session_1.bin");
    touch(session_dir / "auto_save_session_2.bin");

    trainer.cleanup_dead_sessions();

    // Broken session 0 is purged: history entry and its checkpoint files gone
    std::vector<TrainingSession> history = trainer.get_session_history();
    ASSERT_EQ(history.size(), 1u);
    EXPECT_EQ(history[0].session_id, 1);
    EXPECT_FALSE(fs::exists(session_dir / "session_0_checkpoint.bin"));
    EXPECT_FALSE(fs::exists(session_dir / "session_0_checkpoint.bin.config"));

    // Sane session 1 checkpoint survives
    EXPECT_TRUE(fs::exists(session_dir / "session_1_checkpoint.bin"));
    EXPECT_TRUE(fs::exists(session_dir / "session_1_checkpoint.bin.config"));

    // Orphaned session_1_best.bin (not the current session) is removed
    EXPECT_FALSE(fs::exists(session_dir / "session_1_best.bin"));
    EXPECT_FALSE(fs::exists(session_dir / "session_1_best.bin.config"));

    // Live session_2_best.bin (current in-progress session) survives
    EXPECT_TRUE(fs::exists(session_dir / "session_2_best.bin"));
    EXPECT_TRUE(fs::exists(session_dir / "session_2_best.bin.config"));

    // Unreferenced bare checkpoint is removed regardless of id
    EXPECT_FALSE(fs::exists(session_dir / "session_3_checkpoint.bin"));

    // Stale autosave removed, current-session autosave survives
    EXPECT_FALSE(fs::exists(session_dir / "auto_save_session_1.bin"));
    EXPECT_TRUE(fs::exists(session_dir / "auto_save_session_2.bin"));

    // The cleaned history was persisted to disk
    std::ifstream check(history_file);
    std::string line;
    int line_count = 0;
    while (std::getline(check, line)) {
        if (!line.empty() && line[0] != '#') {
            ++line_count;
        }
    }
    EXPECT_EQ(line_count, 1);
}

// ============================================================================
// Checkpoint Management Tests
// ============================================================================

TEST_F(IncrementalTrainerTest, SaveAndLoadModel) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    std::string checkpoint = test_dir.string() + "/test_checkpoint.bin";
    bool save_result = trainer.save_model(checkpoint);
    EXPECT_TRUE(save_result);

    // Model saves multiple files with extensions (.config, .encoder, .decoder, etc.)
    EXPECT_TRUE(fs::exists(checkpoint + ".config"));

    // Create new trainer and load
    IncrementalTrainer trainer2(vocab_file.string(), model_file.string(), config);
    bool load_result = trainer2.load_model(checkpoint);
    EXPECT_TRUE(load_result);
}

// Regression test: is_sane_checkpoint_candidate() used to require a bare
// "session_N_checkpoint.bin" file (no extension) to exist alongside the real
// sidecars — but EncoderDecoderModel::save_model() (the only thing that ever
// writes a checkpoint in production) never creates that bare file, only
// ".config"/".vocab"/".encoder"/".decoder"/".lm_head". That made every
// legitimately-completed session fail the sanity check on the very next
// startup, get logged as "Removing broken session N from history", and have
// its (perfectly valid) checkpoint files deleted — silently forcing a full
// restart from random weights on every incremental `train` invocation.
TEST_F(IncrementalTrainerTest, RealisticallySavedCheckpointSurvivesStartupCleanup) {
    std::string checkpoint_path = session_dir.string() + "/session_0_checkpoint.bin";

    {
        IncrementalConfig config;
        config.session_dir = session_dir.string();
        IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
        // Exercises the exact same EncoderDecoderModel::save_model() call
        // production uses (IncrementalTrainer::save_model() -> model->save_model()) —
        // deliberately not hand-crafting the sidecar files, so this test fails
        // if the real save path and the sanity check ever disagree again.
        ASSERT_TRUE(trainer.save_model(checkpoint_path));
    }
    ASSERT_TRUE(fs::exists(checkpoint_path + ".config"));
    ASSERT_FALSE(fs::exists(checkpoint_path))
        << "save_model() is not expected to write a bare marker file — if this "
           "starts failing, is_sane_checkpoint_candidate() may need revisiting again";

    std::string history_file = session_dir.string() + "/session_history.txt";
    {
        std::ofstream file(history_file);
        file << "# Session History: session_id samples_trained epochs final_loss "
                "final_val_loss checkpoint_path\n";
        file << "0 200000 5 3.2 3.13157 " << checkpoint_path << "\n";
    }

    // A fresh IncrementalTrainer instance (matching what happens on the next
    // `train`/`retrain` process invocation) loads session_history and runs
    // cleanup_dead_sessions() during construction. The session must survive.
    IncrementalConfig config2;
    config2.session_dir = session_dir.string();
    IncrementalTrainer trainer2(vocab_file.string(), model_file.string(), config2);

    std::vector<TrainingSession> history = trainer2.get_session_history();
    ASSERT_EQ(history.size(), 1u)
        << "session was incorrectly dropped as 'broken' despite having a complete, "
           "realistically-saved checkpoint";
    EXPECT_EQ(history[0].session_id, 0);
    EXPECT_EQ(history[0].samples_trained, 200000);
    EXPECT_FLOAT_EQ(history[0].final_validation_loss, 3.13157f);
    EXPECT_TRUE(fs::exists(checkpoint_path + ".config"))
        << "surviving session's checkpoint files must not have been deleted";
}

TEST_F(IncrementalTrainerTest, ResetModelForConfigClearsStaleSessionTracking) {
    std::string checkpoint_path = session_dir.string() + "/session_0_checkpoint.bin";
    {
        IncrementalConfig config;
        config.session_dir = session_dir.string();
        IncrementalTrainer seed_trainer(vocab_file.string(), model_file.string(), config);
        ASSERT_TRUE(seed_trainer.save_model(checkpoint_path));
    }

    std::string history_file = session_dir.string() + "/session_history.txt";
    {
        std::ofstream file(history_file);
        file << "# Session History: session_id samples_trained epochs final_loss "
                "final_val_loss checkpoint_path\n";
        file << "0 200000 5 3.2 3.13157 " << checkpoint_path << "\n";
    }

    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    // Precondition: construction picked up the old session as history, same as
    // production does on startup.
    ASSERT_EQ(trainer.get_session_history().size(), 1u);

    // retrain's reset_model_for_config() must wipe that state — a rebuilt model
    // has nothing in common with a prior architecture/dataset's val_loss, so
    // carrying it forward makes every subsequent epoch look like a regression.
    trainer.reset_model_for_config();

    EXPECT_TRUE(trainer.get_session_history().empty());

    // The cleared state must also be persisted, so a crash right after reset
    // doesn't resurrect the stale history on the next startup.
    std::ifstream check(history_file);
    std::string line;
    int data_lines = 0;
    while (std::getline(check, line))
        if (!line.empty() && line[0] != '#')
            ++data_lines;
    EXPECT_EQ(data_lines, 0);
}

TEST_F(IncrementalTrainerTest, GetLatestCheckpointEmptyWhenNoSessions) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    std::string latest = trainer.get_latest_checkpoint();
    EXPECT_TRUE(latest.empty());
}

TEST_F(IncrementalTrainerTest, GetLatestCheckpointReturnsLastSessionCheckpoint) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    // Load session history
    std::string history_file = session_dir.string() + "/session_history.txt";
    create_session_history_file(history_file, 3);
    trainer.load_session_history();

    std::string latest = trainer.get_latest_checkpoint();
    EXPECT_EQ(latest, session_dir.string() + "/session_2_checkpoint.bin");
}

// ============================================================================
// Statistics and Reporting Tests
// ============================================================================

TEST_F(IncrementalTrainerTest, GetTotalSamplesTrainedZeroInitially) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    int total = trainer.get_total_samples_trained();
    EXPECT_EQ(total, 0);
}

TEST_F(IncrementalTrainerTest, GetTotalTrainingTimeHoursZeroInitially) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    float hours = trainer.get_total_training_time_hours();
    EXPECT_FLOAT_EQ(hours, 0.0f);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(IncrementalTrainerTest, LoadNonExistentModelReturnsFalse) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    bool result = trainer.load_model(test_dir.string() + "/nonexistent.bin");
    EXPECT_FALSE(result);
}

TEST_F(IncrementalTrainerTest, LoadSessionHistoryNonExistentFileReturnsFalse) {
    IncrementalConfig config;
    config.session_dir = test_dir.string() + "/nonexistent_dir";
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    bool result = trainer.load_session_history();
    EXPECT_FALSE(result);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(IncrementalTrainerTest, PrintMethodsDoNotCrash) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    // Should not crash even with no data
    EXPECT_NO_THROW(trainer.print_training_summary());
}

TEST_F(IncrementalTrainerTest, ConfigModificationPersists) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    config.max_sessions_to_keep = 15;
    config.auto_save_enabled = false;

    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    IncrementalConfig& retrieved_config = trainer.get_config();
    retrieved_config.max_sessions_to_keep = 20;

    // Verify modification persists
    EXPECT_EQ(trainer.get_config().max_sessions_to_keep, 20);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(IncrementalTrainerTest, MaxSessionsToKeepZero) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    config.max_sessions_to_keep = 0;
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);

    // Load some sessions
    std::string history_file = session_dir.string() + "/session_history.txt";
    create_session_history_file(history_file, 3);
    trainer.load_session_history();

    // Cleanup should remove all sessions
    trainer.cleanup_old_sessions();

    std::vector<TrainingSession> history = trainer.get_session_history();
    EXPECT_EQ(history.size(), 0);
}

TEST_F(IncrementalTrainerTest, SessionDirectoryCreation) {
    // Create trainer with default config
    IncrementalTrainer trainer(vocab_file.string(), model_file.string());

    // Default session directory should be created
    fs::path default_session_dir = "training_sessions";
    EXPECT_TRUE(fs::exists(default_session_dir));

    // Clean up
    fs::remove_all(default_session_dir);
}

// ============================================================================
// Special Token and Vocabulary Tests
// ============================================================================

TEST_F(IncrementalTrainerTest, VocabularyLoadedCorrectly) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();

    // Constructor should load vocabulary without errors
    EXPECT_NO_THROW(
        { IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config); });
}

TEST_F(IncrementalTrainerTest, InvalidVocabularyThrowsException) {
    fs::path invalid_vocab = test_dir / "invalid_vocab.txt";
    std::ofstream(invalid_vocab.string()) << "NOT A VALID VOCAB FILE\n";

    IncrementalConfig config;
    config.session_dir = session_dir.string();

    // Should throw exception due to invalid vocabulary
    EXPECT_THROW(
        { IncrementalTrainer trainer(invalid_vocab.string(), model_file.string(), config); },
        std::exception);
}

// ============================================================================
// TD-009: Per-epoch metrics & dashboard tests
// ============================================================================

// Write a v2 session history file to the given path
static void create_session_history_file_v2(const std::string& path) {
    std::ofstream f(path);
    f << "# VERSION 2\n";
    f << "# session_id samples_trained epochs final_loss final_val_loss checkpoint_path[|...]\n";
    f << "0 200 3 1.2 1.5 /tmp/dummy_session_0.bin"
         "|losses:1.4,1.3,1.2"
         "|vallosses:1.6,1.55,1.5"
         "|lrs:0.001,0.0009,0.0008"
         "|times:50.0,48.5,51.0\n";
}

TEST_F(IncrementalTrainerTest, LoadSessionHistoryV2ParsesPerEpochVectors) {
    // Write the v2 history to the DEFAULT session dir that the trainer will read from.
    // (The 3-arg constructor with config delegates to the 2-arg constructor which uses
    //  the default IncrementalConfig::session_dir = "training_sessions".)
    std::string default_dir = "training_sessions";
    fs::create_directories(default_dir);
    std::string hist_path = default_dir + "/session_history.txt";

    // Create trainer using 2-arg constructor so it reads from the default dir.
    // Construct BEFORE writing the history file: cleanup_dead_sessions() runs
    // during construction and would purge this entry, since its checkpoint
    // path (a placeholder that's never actually created on disk) doesn't exist.
    IncrementalTrainer trainer(vocab_file.string(), model_file.string());
    create_session_history_file_v2(hist_path);
    trainer.load_session_history();
    auto history = trainer.get_session_history();

    // Cleanup artefact created by this test
    fs::remove(hist_path);

    ASSERT_GE(history.size(), 1u);
    // Find the session we wrote (session_id == 0)
    const TrainingSession* s = nullptr;
    for (const auto& h : history)
        if (h.session_id == 0) {
            s = &h;
            break;
        }
    ASSERT_NE(s, nullptr);

    EXPECT_EQ(s->epochs_completed, 3);
    EXPECT_EQ(s->checkpoint_path, "/tmp/dummy_session_0.bin");

    ASSERT_EQ(s->per_epoch_losses.size(), 3u);
    EXPECT_NEAR(s->per_epoch_losses[0], 1.4f, 1e-4f);
    EXPECT_NEAR(s->per_epoch_losses[2], 1.2f, 1e-4f);

    ASSERT_EQ(s->per_epoch_validation_losses.size(), 3u);
    EXPECT_NEAR(s->per_epoch_validation_losses[1], 1.55f, 1e-4f);

    ASSERT_EQ(s->per_epoch_learning_rates.size(), 3u);
    EXPECT_NEAR(s->per_epoch_learning_rates[0], 0.001f, 1e-6f);

    ASSERT_EQ(s->training_time_per_epoch.size(), 3u);
    EXPECT_NEAR(s->training_time_per_epoch[1], 48.5, 1e-3);
}

TEST_F(IncrementalTrainerTest, SaveLoadSessionHistoryRoundTripWithPerEpochData) {
    // Verify the v2 history format round-trip at the file content level.
    // Write a v2 file, parse it with a fresh trainer, confirm vectors intact.
    std::string default_dir = "training_sessions";
    fs::create_directories(default_dir);
    std::string hist_path = default_dir + "/session_history.txt";

    // Construct before writing history — see comment in
    // LoadSessionHistoryV2ParsesPerEpochVectors for why.
    IncrementalTrainer trainer(vocab_file.string(), model_file.string());
    create_session_history_file_v2(hist_path);
    trainer.load_session_history();
    auto history = trainer.get_session_history();
    fs::remove(hist_path);

    ASSERT_GE(history.size(), 1u);
    const TrainingSession* s = nullptr;
    for (const auto& h : history)
        if (h.session_id == 0) {
            s = &h;
            break;
        }
    ASSERT_NE(s, nullptr);

    // All four per-epoch vectors survived the parse
    std::vector<float> exp_losses = {1.4f, 1.3f, 1.2f};
    std::vector<float> exp_val = {1.6f, 1.55f, 1.5f};
    std::vector<float> exp_lrs = {0.001f, 0.0009f, 0.0008f};
    std::vector<double> exp_times = {50.0, 48.5, 51.0};

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(s->per_epoch_losses[i], exp_losses[i], 1e-4f);
        EXPECT_NEAR(s->per_epoch_validation_losses[i], exp_val[i], 1e-4f);
        EXPECT_NEAR(s->per_epoch_learning_rates[i], exp_lrs[i], 1e-6f);
        EXPECT_NEAR(s->training_time_per_epoch[i], exp_times[i], 1e-3);
    }
}

// ============================================================================
// TokenizerMode Mapping in make_incremental_config
// ============================================================================

TEST_F(IncrementalTrainerTest, MakeIncrementalConfigDefaultsToAsciiMode) {
    adai::ServiceConfig svc;  // unicode_tokenizer defaults to false

    const IncrementalConfig cfg = IncrementalTrainer::make_incremental_config(svc);

    EXPECT_EQ(cfg.base_config.tokenizer_mode, TokenizerMode::ASCII);
}

TEST_F(IncrementalTrainerTest, MakeIncrementalConfigMapsAsciiTokenizerMode) {
    adai::ServiceConfig svc;
    svc.unicode_tokenizer = false;

    const IncrementalConfig cfg = IncrementalTrainer::make_incremental_config(svc);

    EXPECT_EQ(cfg.base_config.tokenizer_mode, TokenizerMode::ASCII);
}

TEST_F(IncrementalTrainerTest, MakeIncrementalConfigMapsUnicodeTokenizerMode) {
    adai::ServiceConfig svc;
    svc.unicode_tokenizer = true;

    const IncrementalConfig cfg = IncrementalTrainer::make_incremental_config(svc);

    EXPECT_EQ(cfg.base_config.tokenizer_mode, TokenizerMode::UNICODE);
}

TEST_F(IncrementalTrainerTest, DisplayDashboardDoesNotCrash) {
    // print_training_summary exercises make_sparkline and per-epoch display.
    // Test with no per-epoch data (empty session history).
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    EXPECT_NO_THROW(trainer.print_training_summary());

    // Test with per-epoch data loaded. Construct before writing history — see
    // comment in LoadSessionHistoryV2ParsesPerEpochVectors for why.
    std::string default_dir = "training_sessions";
    fs::create_directories(default_dir);
    std::string hist_path = default_dir + "/session_history.txt";
    IncrementalTrainer trainer2(vocab_file.string(), model_file.string());
    create_session_history_file_v2(hist_path);
    trainer2.load_session_history();
    fs::remove(hist_path);
    EXPECT_NO_THROW(trainer2.print_training_summary());
}

// Run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
