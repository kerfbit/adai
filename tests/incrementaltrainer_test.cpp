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

#include <gtest/gtest.h>
#include "IncrementalTrainer.hpp"
#include "BPETokenizer.hpp"
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>

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
        file << "# Session History: session_id samples_trained epochs final_loss final_val_loss checkpoint_path\n";
        for (int i = 0; i < num_sessions; ++i) {
            file << i << " "
                 << 100 << " "
                 << 5 << " "
                 << (2.0f - i * 0.1f) << " "
                 << (2.5f - i * 0.1f) << " "
                 << session_dir.string() << "/session_" << i << "_checkpoint.bin\n";
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

// ============================================================================
// Data Management Tests
// ============================================================================

TEST_F(IncrementalTrainerTest, AddNewDataSingleFile) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    bool result = trainer.add_new_data(data_file1.string());
    EXPECT_TRUE(result);
    
    std::vector<std::string> pending = trainer.get_pending_data_files();
    EXPECT_EQ(pending.size(), 1);
    EXPECT_EQ(pending[0], data_file1.string());
}

TEST_F(IncrementalTrainerTest, AddNewDataMultipleFiles) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    trainer.add_new_data(data_file1.string());
    trainer.add_new_data(data_file2.string());
    
    std::vector<std::string> pending = trainer.get_pending_data_files();
    EXPECT_EQ(pending.size(), 2);
}

TEST_F(IncrementalTrainerTest, AddNewDataBatch) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    std::vector<std::string> files = {data_file1.string(), data_file2.string()};
    bool result = trainer.add_new_data_batch(files);
    EXPECT_TRUE(result);
    
    std::vector<std::string> pending = trainer.get_pending_data_files();
    EXPECT_EQ(pending.size(), 2);
}

TEST_F(IncrementalTrainerTest, AddNonExistentFileReturnsFalse) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    bool result = trainer.add_new_data(test_dir.string() + "/nonexistent.txt");
    EXPECT_FALSE(result);
    
    std::vector<std::string> pending = trainer.get_pending_data_files();
    EXPECT_EQ(pending.size(), 0);
}

TEST_F(IncrementalTrainerTest, ClearPendingData) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    trainer.add_new_data(data_file1.string());
    trainer.add_new_data(data_file2.string());
    
    std::vector<std::string> pending = trainer.get_pending_data_files();
    EXPECT_EQ(pending.size(), 2);
    
    trainer.clear_pending_data();
    
    pending = trainer.get_pending_data_files();
    EXPECT_EQ(pending.size(), 0);
}

TEST_F(IncrementalTrainerTest, GetTrainedDataFilesEmpty) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    std::vector<std::string> trained = trainer.get_trained_data_files();
    EXPECT_EQ(trained.size(), 0);
}

TEST_F(IncrementalTrainerTest, ComputeDataChecksumConsistent) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    std::string checksum1 = trainer.compute_data_checksum(data_file1.string());
    std::string checksum2 = trainer.compute_data_checksum(data_file1.string());
    
    EXPECT_FALSE(checksum1.empty());
    EXPECT_EQ(checksum1, checksum2);
}

TEST_F(IncrementalTrainerTest, ComputeDataChecksumDifferentForDifferentFiles) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    std::string checksum1 = trainer.compute_data_checksum(data_file1.string());
    std::string checksum2 = trainer.compute_data_checksum(data_file2.string());
    
    EXPECT_NE(checksum1, checksum2);
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
    // Create session history file
    std::string history_file = session_dir.string() + "/session_history.txt";
    create_session_history_file(history_file, 3);
    
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
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
        std::string checkpoint = session_dir.string() + "/session_" + std::to_string(i) + "_checkpoint.bin";
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
// Data Registry Tests
// ============================================================================

TEST_F(IncrementalTrainerTest, SaveAndLoadDataRegistry) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    // Add data files
    trainer.add_new_data(data_file1.string());
    
    // Save registry
    bool save_result = trainer.save_data_registry();
    EXPECT_TRUE(save_result);
    
    // Create new trainer and load registry
    IncrementalTrainer trainer2(vocab_file.string(), model_file.string(), config);
    bool load_result = trainer2.load_data_registry();
    EXPECT_TRUE(load_result);
}

TEST_F(IncrementalTrainerTest, IsDataTrainedReturnsFalseForUntrained) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    bool trained = trainer.is_data_trained(data_file1.string());
    EXPECT_FALSE(trained);
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
    
    // These should not crash even with no data
    EXPECT_NO_THROW(trainer.print_training_summary());
    EXPECT_NO_THROW(trainer.print_session_history());
    EXPECT_NO_THROW(trainer.print_data_registry());
}

TEST_F(IncrementalTrainerTest, MultipleDataAddRemoveCycles) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    // Add data
    trainer.add_new_data(data_file1.string());
    EXPECT_EQ(trainer.get_pending_data_files().size(), 1);
    
    // Clear
    trainer.clear_pending_data();
    EXPECT_EQ(trainer.get_pending_data_files().size(), 0);
    
    // Add again
    trainer.add_new_data(data_file1.string());
    trainer.add_new_data(data_file2.string());
    EXPECT_EQ(trainer.get_pending_data_files().size(), 2);
    
    // Clear again
    trainer.clear_pending_data();
    EXPECT_EQ(trainer.get_pending_data_files().size(), 0);
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

TEST_F(IncrementalTrainerTest, EmptyDataFileHandling) {
    // Create empty data file
    fs::path empty_file = test_dir / "empty.txt";
    std::ofstream(empty_file.string()).close();
    
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    // Should still add to pending even if empty
    bool result = trainer.add_new_data(empty_file.string());
    EXPECT_TRUE(result);
}

TEST_F(IncrementalTrainerTest, VeryLongFilePathHandling) {
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    
    // Create a very long but valid path
    std::string long_path = test_dir.string() + "/";
    for (int i = 0; i < 10; ++i) {
        long_path += "very_long_directory_name_here/";
    }
    long_path += "data.txt";
    
    // Should handle gracefully (will return false since file doesn't exist)
    bool result = trainer.add_new_data(long_path);
    EXPECT_FALSE(result);
}

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
    EXPECT_NO_THROW({
        IncrementalTrainer trainer(vocab_file.string(), model_file.string(), config);
    });
}

TEST_F(IncrementalTrainerTest, InvalidVocabularyThrowsException) {
    fs::path invalid_vocab = test_dir / "invalid_vocab.txt";
    std::ofstream(invalid_vocab.string()) << "NOT A VALID VOCAB FILE\n";
    
    IncrementalConfig config;
    config.session_dir = session_dir.string();
    
    // Should throw exception due to invalid vocabulary
    EXPECT_THROW({
        IncrementalTrainer trainer(invalid_vocab.string(), model_file.string(), config);
    }, std::exception);
}

// Run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
