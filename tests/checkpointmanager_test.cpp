#include <gtest/gtest.h>
#include "CheckpointManager.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

class CheckpointManagerTest : public ::testing::Test {
protected:
    std::string test_dir = "test_checkpoints/";
    
    void SetUp() override {
        // Clean up any existing test directory
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }
    
    void TearDown() override {
        // Clean up test directory
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }
    
    void create_dummy_checkpoint(const std::string& filepath) {
        std::ofstream file(filepath);
        file << "dummy checkpoint data";
        file.close();
    }
};

// Test: Constructor
TEST_F(CheckpointManagerTest, Constructor) {
    CheckpointManager manager(test_dir, 5);
    
    EXPECT_TRUE(std::filesystem::exists(test_dir));
    EXPECT_EQ(manager.get_checkpoint_dir(), test_dir);
}

// Test: Save checkpoint
TEST_F(CheckpointManagerTest, SaveCheckpoint) {
    CheckpointManager manager(test_dir, 5);
    
    std::string path = manager.save_checkpoint(0, 3.0f, 3.5f);
    
    EXPECT_FALSE(path.empty());
    EXPECT_EQ(manager.get_checkpoints().size(), 1);
    
    // Verify metadata file exists
    std::string meta_path = path + ".meta";
    EXPECT_TRUE(std::filesystem::exists(meta_path));
}

// Test: Best checkpoint tracking
TEST_F(CheckpointManagerTest, BestCheckpointTracking) {
    CheckpointManager manager(test_dir, 5);
    
    manager.save_checkpoint(0, 5.0f, 5.5f);
    EXPECT_FLOAT_EQ(manager.get_best_validation_loss(), 5.5f);
    
    manager.save_checkpoint(1, 4.0f, 4.5f);  // Better
    EXPECT_FLOAT_EQ(manager.get_best_validation_loss(), 4.5f);
    
    manager.save_checkpoint(2, 3.0f, 5.0f);  // Worse
    EXPECT_FLOAT_EQ(manager.get_best_validation_loss(), 4.5f);  // Should stay
    
    std::string best_path = manager.get_best_checkpoint_path();
    EXPECT_FALSE(best_path.empty());
    EXPECT_NE(best_path.find("epoch_0001"), std::string::npos);
}

// Test: Checkpoint rotation
TEST_F(CheckpointManagerTest, CheckpointRotation) {
    CheckpointManager manager(test_dir, 3);  // Keep only 3
    
    // Save 5 checkpoints
    for (int i = 0; i < 5; ++i) {
        std::string path = manager.save_checkpoint(i, 5.0f - i * 0.5f, 5.5f - i * 0.4f);
        create_dummy_checkpoint(path);  // Create actual file
    }
    
    // Should only keep 3 best
    EXPECT_EQ(manager.get_checkpoints().size(), 3);
}

// Test: Get checkpoint info
TEST_F(CheckpointManagerTest, GetCheckpointInfo) {
    CheckpointManager manager(test_dir, 5);
    
    manager.save_checkpoint(0, 3.0f, 3.5f);
    manager.save_checkpoint(1, 2.5f, 3.0f);
    
    auto info = manager.get_checkpoint_info(1);
    EXPECT_EQ(info.epoch, 1);
    EXPECT_FLOAT_EQ(info.train_loss, 2.5f);
    EXPECT_FLOAT_EQ(info.validation_loss, 3.0f);
    EXPECT_FALSE(info.filepath.empty());
}

// Test: Has checkpoint
TEST_F(CheckpointManagerTest, HasCheckpoint) {
    CheckpointManager manager(test_dir, 5);
    
    manager.save_checkpoint(0, 3.0f, 3.5f);
    
    EXPECT_TRUE(manager.has_checkpoint(0));
    EXPECT_FALSE(manager.has_checkpoint(1));
    EXPECT_FALSE(manager.has_checkpoint(99));
}

// Test: Get all checkpoints
TEST_F(CheckpointManagerTest, GetAllCheckpoints) {
    CheckpointManager manager(test_dir, 5);
    
    manager.save_checkpoint(0, 3.0f, 3.5f);
    manager.save_checkpoint(1, 2.5f, 3.0f);
    manager.save_checkpoint(2, 2.2f, 2.8f);
    
    const auto& checkpoints = manager.get_checkpoints();
    EXPECT_EQ(checkpoints.size(), 3);
}

// Test: Clear all checkpoints
TEST_F(CheckpointManagerTest, ClearAll) {
    CheckpointManager manager(test_dir, 5);
    
    for (int i = 0; i < 3; ++i) {
        std::string path = manager.save_checkpoint(i, 5.0f - i, 5.5f - i);
        create_dummy_checkpoint(path);
    }
    
    EXPECT_EQ(manager.get_checkpoints().size(), 3);
    
    manager.clear_all();
    
    EXPECT_EQ(manager.get_checkpoints().size(), 0);
    EXPECT_TRUE(manager.get_best_checkpoint_path().empty());
}

// Test: Set max checkpoints
TEST_F(CheckpointManagerTest, SetMaxCheckpoints) {
    CheckpointManager manager(test_dir, 5);
    
    for (int i = 0; i < 5; ++i) {
        std::string path = manager.save_checkpoint(i, 5.0f - i * 0.5f, 5.5f - i * 0.4f);
        create_dummy_checkpoint(path);
    }
    
    EXPECT_EQ(manager.get_checkpoints().size(), 5);
    
    manager.set_max_checkpoints(3);
    
    EXPECT_EQ(manager.get_checkpoints().size(), 3);
}

// Test: Load existing checkpoints
TEST_F(CheckpointManagerTest, LoadExistingCheckpoints) {
    // Create first manager and save checkpoints
    {
        CheckpointManager manager1(test_dir, 5);
        for (int i = 0; i < 3; ++i) {
            std::string path = manager1.save_checkpoint(i, 5.0f - i, 5.5f - i);
            create_dummy_checkpoint(path);
        }
    }
    
    // Create new manager - should load existing
    CheckpointManager manager2(test_dir, 5);
    
    EXPECT_EQ(manager2.get_checkpoints().size(), 3);
    EXPECT_FALSE(manager2.get_best_checkpoint_path().empty());
}

// Test: Checkpoint without validation loss
TEST_F(CheckpointManagerTest, CheckpointWithoutValidation) {
    CheckpointManager manager(test_dir, 5);
    
    std::string path = manager.save_checkpoint(0, 3.0f);  // No validation loss
    
    EXPECT_FALSE(path.empty());
    
    auto info = manager.get_checkpoint_info(0);
    EXPECT_FLOAT_EQ(info.train_loss, 3.0f);
    EXPECT_FLOAT_EQ(info.validation_loss, 0.0f);
}

// Test: Best checkpoint marking
TEST_F(CheckpointManagerTest, BestCheckpointMarking) {
    CheckpointManager manager(test_dir, 5);
    
    manager.save_checkpoint(0, 5.0f, 5.5f);
    manager.save_checkpoint(1, 4.0f, 4.5f);  // Best
    manager.save_checkpoint(2, 3.0f, 5.0f);  // Not best
    
    auto info0 = manager.get_checkpoint_info(0);
    auto info1 = manager.get_checkpoint_info(1);
    auto info2 = manager.get_checkpoint_info(2);
    
    EXPECT_FALSE(info0.is_best);
    EXPECT_TRUE(info1.is_best);
    EXPECT_FALSE(info2.is_best);
}

// Test: Timestamp recording
TEST_F(CheckpointManagerTest, TimestampRecording) {
    CheckpointManager manager(test_dir, 5);
    
    auto before = std::time(nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    manager.save_checkpoint(0, 3.0f, 3.5f);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto after = std::time(nullptr);
    
    auto info = manager.get_checkpoint_info(0);
    EXPECT_GE(info.timestamp, before);
    EXPECT_LE(info.timestamp, after);
}

// Test: Print summary (should not crash)
TEST_F(CheckpointManagerTest, PrintSummary) {
    CheckpointManager manager(test_dir, 5);
    
    manager.save_checkpoint(0, 3.0f, 3.5f);
    manager.save_checkpoint(1, 2.5f, 3.0f);
    
    testing::internal::CaptureStdout();
    manager.print_summary();
    std::string output = testing::internal::GetCapturedStdout();
    
    EXPECT_GT(output.length(), 0);
}

// Test: Empty manager operations
TEST_F(CheckpointManagerTest, EmptyManagerOperations) {
    CheckpointManager manager(test_dir, 5);
    
    EXPECT_EQ(manager.get_checkpoints().size(), 0);
    EXPECT_TRUE(manager.get_best_checkpoint_path().empty());
    EXPECT_FALSE(manager.has_checkpoint(0));
    
    auto info = manager.get_checkpoint_info(0);
    EXPECT_EQ(info.epoch, 0);
    EXPECT_TRUE(info.filepath.empty());
}

// Test: Multiple epochs with same validation loss
TEST_F(CheckpointManagerTest, SameValidationLoss) {
    CheckpointManager manager(test_dir, 5);
    
    manager.save_checkpoint(0, 3.0f, 3.5f);
    manager.save_checkpoint(1, 2.5f, 3.5f);  // Same val loss
    manager.save_checkpoint(2, 2.0f, 3.5f);  // Same val loss
    
    // First one with that val loss should be marked best
    EXPECT_FLOAT_EQ(manager.get_best_validation_loss(), 3.5f);
}

// Test: Non-existent checkpoint info
TEST_F(CheckpointManagerTest, NonExistentCheckpointInfo) {
    CheckpointManager manager(test_dir, 5);
    
    auto info = manager.get_checkpoint_info(999);
    EXPECT_EQ(info.epoch, 0);
    EXPECT_TRUE(info.filepath.empty());
    EXPECT_FLOAT_EQ(info.train_loss, 0.0f);
}

// Test: Rotation preserves best
TEST_F(CheckpointManagerTest, RotationPreservesBest) {
    CheckpointManager manager(test_dir, 3);
    
    // Save checkpoints with middle one being best
    std::string path0 = manager.save_checkpoint(0, 5.0f, 5.5f);
    create_dummy_checkpoint(path0);
    
    std::string path1 = manager.save_checkpoint(1, 4.0f, 3.0f);  // Best
    create_dummy_checkpoint(path1);
    
    std::string path2 = manager.save_checkpoint(2, 3.5f, 4.0f);
    create_dummy_checkpoint(path2);
    
    std::string path3 = manager.save_checkpoint(3, 3.0f, 3.8f);
    create_dummy_checkpoint(path3);
    
    // Should rotate but keep best
    EXPECT_TRUE(manager.has_checkpoint(1));  // Best should be kept
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
