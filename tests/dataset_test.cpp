#include <gtest/gtest.h>
#include "Dataset.hpp"
#include <fstream>
#include <filesystem>

class DatasetTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test data files
        create_test_conversation_file();
        create_test_tsv_file();
    }
    
    void TearDown() override {
        // Clean up test files
        std::filesystem::remove("test_conversations.txt");
        std::filesystem::remove("test_data.tsv");
        std::filesystem::remove("test_output.txt");
    }
    
    void create_test_conversation_file() {
        std::ofstream file("test_conversations.txt");
        file << "INPUT: Hello\n";
        file << "RESPONSE: Hi there!\n\n";
        file << "INPUT: How are you?\n";
        file << "RESPONSE: I'm doing great!\n\n";
        file << "INPUT: What's your name?\n";
        file << "RESPONSE: I'm an AI assistant.\n\n";
        file.close();
    }
    
    void create_test_tsv_file() {
        std::ofstream file("test_data.tsv");
        file << "Hello\tHi there!\n";
        file << "How are you?\tI'm doing great!\n";
        file << "What's your name?\tI'm an AI assistant.\n";
        file.close();
    }
};

// Test: Constructor
TEST_F(DatasetTest, Constructor) {
    Dataset dataset;
    EXPECT_TRUE(dataset.empty());
    EXPECT_EQ(dataset.size(), 0);
    EXPECT_FALSE(dataset.is_split());
}

// Test: Load conversation format
TEST_F(DatasetTest, LoadConversationFormat) {
    Dataset dataset;
    EXPECT_TRUE(dataset.load_from_file("test_conversations.txt"));
    EXPECT_EQ(dataset.size(), 3);
    EXPECT_FALSE(dataset.empty());
    
    const auto& data = dataset.get_all();
    EXPECT_EQ(data[0].input, "Hello");
    EXPECT_EQ(data[0].target, "Hi there!");
    EXPECT_EQ(data[1].input, "How are you?");
    EXPECT_EQ(data[1].target, "I'm doing great!");
}

// Test: Load TSV format
TEST_F(DatasetTest, LoadTsvFormat) {
    Dataset dataset;
    EXPECT_TRUE(dataset.load_from_file("test_data.tsv"));
    EXPECT_EQ(dataset.size(), 3);
    
    const auto& data = dataset.get_all();
    EXPECT_EQ(data[0].input, "Hello");
    EXPECT_EQ(data[0].target, "Hi there!");
}

// Test: Load non-existent file
TEST_F(DatasetTest, LoadNonExistentFile) {
    Dataset dataset;
    EXPECT_FALSE(dataset.load_from_file("nonexistent.txt"));
    EXPECT_TRUE(dataset.empty());
}

// Test: Add single sample
TEST_F(DatasetTest, AddSample) {
    Dataset dataset;
    dataset.add_sample("Input 1", "Target 1");
    dataset.add_sample("Input 2", "Target 2");
    
    EXPECT_EQ(dataset.size(), 2);
    const auto& data = dataset.get_all();
    EXPECT_EQ(data[0].input, "Input 1");
    EXPECT_EQ(data[0].target, "Target 1");
}

// Test: Add multiple samples
TEST_F(DatasetTest, AddSamples) {
    Dataset dataset;
    std::vector<DataSample> samples = {
        {"Input 1", "Target 1"},
        {"Input 2", "Target 2"},
        {"Input 3", "Target 3"}
    };
    
    dataset.add_samples(samples);
    EXPECT_EQ(dataset.size(), 3);
}

// Test: Split dataset
TEST_F(DatasetTest, SplitDataset) {
    Dataset dataset;
    dataset.load_from_file("test_conversations.txt");
    
    dataset.split(0.6f, 0.2f, 0.2f);
    
    EXPECT_TRUE(dataset.is_split());
    EXPECT_GT(dataset.size(SplitType::TRAIN), 0);
    
    // Check total equals original
    size_t total = dataset.size(SplitType::TRAIN) + 
                  dataset.size(SplitType::VALIDATION) + 
                  dataset.size(SplitType::TEST);
    EXPECT_EQ(total, dataset.size());
}

// Test: Get split data
TEST_F(DatasetTest, GetSplit) {
    Dataset dataset;
    dataset.load_from_file("test_conversations.txt");
    dataset.split(0.6f, 0.2f, 0.2f);
    
    auto train_data = dataset.get_split(SplitType::TRAIN);
    auto val_data = dataset.get_split(SplitType::VALIDATION);
    auto test_data = dataset.get_split(SplitType::TEST);
    
    EXPECT_GT(train_data.size(), 0);
    EXPECT_EQ(train_data.size() + val_data.size() + test_data.size(), dataset.size());
}

// Test: Shuffle data
TEST_F(DatasetTest, ShuffleData) {
    Dataset dataset(42);  // Fixed seed
    for (int i = 0; i < 10; ++i) {
        dataset.add_sample("Input " + std::to_string(i), "Target " + std::to_string(i));
    }
    
    auto original = dataset.get_all();
    dataset.shuffle();
    auto shuffled = dataset.get_all();
    
    // Data should be different order (with high probability)
    bool different = false;
    for (size_t i = 0; i < original.size(); ++i) {
        if (original[i].input != shuffled[i].input) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
    
    // But same size
    EXPECT_EQ(original.size(), shuffled.size());
}

// Test: Shuffle split
TEST_F(DatasetTest, ShuffleSplit) {
    Dataset dataset(42);
    for (int i = 0; i < 10; ++i) {
        dataset.add_sample("Input " + std::to_string(i), "Target " + std::to_string(i));
    }
    
    dataset.split(0.8f, 0.1f, 0.1f);
    auto original_train = dataset.get_split(SplitType::TRAIN);
    
    dataset.shuffle_split(SplitType::TRAIN);
    auto shuffled_train = dataset.get_split(SplitType::TRAIN);
    
    EXPECT_EQ(original_train.size(), shuffled_train.size());
}

// Test: Dataset statistics
TEST_F(DatasetTest, DatasetStatistics) {
    Dataset dataset;
    dataset.add_sample("Hello", "Hi");
    dataset.add_sample("How are you?", "I'm fine, thanks!");
    dataset.add_sample("Bye", "Goodbye!");
    
    const auto& stats = dataset.get_stats();
    EXPECT_EQ(stats.total_samples, 3);
    EXPECT_GT(stats.avg_input_length, 0);
    EXPECT_GT(stats.avg_target_length, 0);
    EXPECT_GT(stats.max_input_length, 0);
    EXPECT_GT(stats.max_target_length, 0);
}

// Test: Clear dataset
TEST_F(DatasetTest, ClearDataset) {
    Dataset dataset;
    dataset.load_from_file("test_conversations.txt");
    dataset.split(0.8f, 0.1f, 0.1f);
    
    EXPECT_FALSE(dataset.empty());
    EXPECT_TRUE(dataset.is_split());
    
    dataset.clear();
    
    EXPECT_TRUE(dataset.empty());
    EXPECT_FALSE(dataset.is_split());
    EXPECT_EQ(dataset.size(), 0);
}

// Test: Save to file (conversation format)
TEST_F(DatasetTest, SaveConversationFormat) {
    Dataset dataset;
    dataset.add_sample("Hello", "Hi");
    dataset.add_sample("Bye", "Goodbye");
    
    EXPECT_TRUE(dataset.save_to_file("test_output.txt", "conversation"));
    
    // Verify by loading
    Dataset loaded;
    EXPECT_TRUE(loaded.load_from_file("test_output.txt"));
    EXPECT_EQ(loaded.size(), 2);
}

// Test: Save to file (TSV format)
TEST_F(DatasetTest, SaveTsvFormat) {
    Dataset dataset;
    dataset.add_sample("Hello", "Hi");
    dataset.add_sample("Bye", "Goodbye");
    
    EXPECT_TRUE(dataset.save_to_file("test_output.txt", "tsv"));
    
    // Verify by loading
    Dataset loaded;
    EXPECT_TRUE(loaded.load_from_file("test_output.txt"));
    EXPECT_EQ(loaded.size(), 2);
}

// Test: Split with invalid ratios
TEST_F(DatasetTest, SplitInvalidRatios) {
    Dataset dataset;
    dataset.load_from_file("test_conversations.txt");
    
    // Should normalize automatically
    dataset.split(0.5f, 0.5f, 0.5f);  // Sum = 1.5
    
    EXPECT_TRUE(dataset.is_split());
    size_t total = dataset.size(SplitType::TRAIN) + 
                  dataset.size(SplitType::VALIDATION) + 
                  dataset.size(SplitType::TEST);
    EXPECT_EQ(total, dataset.size());
}

// Test: Empty dataset operations
TEST_F(DatasetTest, EmptyDatasetOperations) {
    Dataset dataset;
    
    dataset.shuffle();  // Should not crash
    dataset.split(0.8f, 0.1f, 0.1f);  // Should print warning
    
    EXPECT_TRUE(dataset.empty());
    EXPECT_EQ(dataset.size(), 0);
}

// Test: Print statistics (should not crash)
TEST_F(DatasetTest, PrintStatistics) {
    Dataset dataset;
    dataset.load_from_file("test_conversations.txt");
    dataset.split(0.8f, 0.1f, 0.1f);
    
    // Should not crash
    testing::internal::CaptureStdout();
    dataset.print_stats();
    std::string output = testing::internal::GetCapturedStdout();
    
    EXPECT_GT(output.length(), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
