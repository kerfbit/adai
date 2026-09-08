#include "Dataset.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

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
        std::filesystem::remove("test_output.jsonl");
        std::filesystem::remove("test_data.json");
        std::filesystem::remove("test_data.csv");
        std::filesystem::remove("test_data.jsonl");
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

    void create_test_json_file() {
        std::ofstream file("test_data.json");
        file << "{\"input\": \"Hello\", \"target\": \"Hi there!\"}\n";
        file << "{\"input\": \"How are you?\", \"target\": \"I'm doing great!\"}\n";
        file << "{\"input\": \"What's your name?\", \"target\": \"I'm an AI assistant.\"}\n";
        file.close();
    }

    void create_test_jsonl_training_file() {
        std::ofstream file("test_data.jsonl");
        // Training JSONL format: "response" key (not "target")
        file << "{\"input\":\"Hello\",\"response\":\"Hi there!\"}\n";
        file << "{\"input\":\"How are you?\",\"response\":\"I'm doing great!\","
                "\"domain\":\"dialogue\",\"task_type\":\"chat\",\"quality\":0.8}\n";
        file << "{\"input\":\"What's your name?\",\"response\":\"I'm an AI assistant.\","
                "\"language\":\"en\",\"token_count\":15}\n";
        file.close();
    }

    void create_test_csv_file() {
        std::ofstream file("test_data.csv");
        file << "input,target\n";
        file << "\"Hello\",\"Hi there!\"\n";
        file << "\"How are you?\",\"I'm doing great!\"\n";
        file << "\"What's your name?\",\"I'm an AI assistant.\"\n";
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

// Regression test (TD-056): an existing-but-empty file used to call
// std::string::front() on an empty first_line inside load_from_file()'s
// format-detection chain — undefined behavior (aborts under
// _GLIBCXX_ASSERTIONS). Must return false/empty instead of crashing.
TEST_F(DatasetTest, LoadEmptyFileDoesNotCrash) {
    std::ofstream("test_empty.txt").close();  // create a real, 0-byte file

    Dataset dataset;
    EXPECT_FALSE(dataset.load_from_file("test_empty.txt"));
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
        {"Input 1", "Target 1"}, {"Input 2", "Target 2"}, {"Input 3", "Target 3"}};

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
    size_t total = dataset.size(SplitType::TRAIN) + dataset.size(SplitType::VALIDATION) +
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
    size_t total = dataset.size(SplitType::TRAIN) + dataset.size(SplitType::VALIDATION) +
                   dataset.size(SplitType::TEST);
    EXPECT_EQ(total, dataset.size());
}

// Test: Empty dataset operations
TEST_F(DatasetTest, EmptyDatasetOperations) {
    Dataset dataset;

    dataset.shuffle();                // Should not crash
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

// ============================================================
// v2.0 Enhanced Features Tests
// ============================================================

// Test: Iterator interface
TEST_F(DatasetTest, IteratorInterface) {
    Dataset dataset;
    dataset.add_sample("Input 1", "Target 1");
    dataset.add_sample("Input 2", "Target 2");
    dataset.add_sample("Input 3", "Target 3");

    // Range-based for loop
    int count = 0;
    for (const auto& sample : dataset) {
        EXPECT_FALSE(sample.input.empty());
        EXPECT_FALSE(sample.target.empty());
        count++;
    }
    EXPECT_EQ(count, 3);

    // Manual iteration
    auto it = dataset.begin();
    EXPECT_EQ(it->input, "Input 1");
    ++it;
    EXPECT_EQ(it->input, "Input 2");

    // Const iteration
    const Dataset& const_dataset = dataset;
    auto cit = const_dataset.cbegin();
    EXPECT_EQ(cit->input, "Input 1");
}

// Test: Batch iterator
TEST_F(DatasetTest, BatchIterator) {
    Dataset dataset;
    for (int i = 0; i < 10; ++i) {
        dataset.add_sample("Input " + std::to_string(i), "Target " + std::to_string(i));
    }
    dataset.split(0.8, 0.1, 0.1);

    // Batch iteration
    int batch_count = 0;
    int sample_count = 0;
    for (auto batch : dataset.get_batch_range(SplitType::TRAIN, 3)) {
        batch_count++;
        sample_count += batch.size();
        EXPECT_GT(batch.size(), 0);
        EXPECT_LE(batch.size(), 3);
    }

    EXPECT_EQ(sample_count, dataset.size(SplitType::TRAIN));
    EXPECT_GT(batch_count, 0);
}

// Test: Load JSON format
TEST_F(DatasetTest, LoadJsonFormat) {
    create_test_json_file();

    Dataset dataset;
    EXPECT_TRUE(dataset.load_from_file("test_data.json"));
    EXPECT_EQ(dataset.size(), 3);

    const auto& data = dataset.get_all();
    EXPECT_EQ(data[0].input, "Hello");
    EXPECT_EQ(data[0].target, "Hi there!");
}

// Test: Load CSV format
TEST_F(DatasetTest, LoadCsvFormat) {
    create_test_csv_file();

    Dataset dataset;
    EXPECT_TRUE(dataset.load_from_file("test_data.csv"));
    EXPECT_EQ(dataset.size(), 3);

    const auto& data = dataset.get_all();
    EXPECT_EQ(data[0].input, "Hello");
    EXPECT_EQ(data[0].target, "Hi there!");
}

// Test: Stratified split
TEST_F(DatasetTest, StratifiedSplit) {
    Dataset dataset;
    // Add samples with varying lengths
    dataset.add_sample("Short", "Response");
    dataset.add_sample("Medium length input", "Medium response");
    dataset.add_sample("Very long input text here", "Very long response text here");
    dataset.add_sample("X", "Y");
    dataset.add_sample("Another medium one", "Another response");
    dataset.add_sample("This is quite a long input with many words", "Long response as well");

    dataset.split_stratified(0.5, 0.3, 0.2, 3);

    EXPECT_TRUE(dataset.is_split());
    size_t total = dataset.size(SplitType::TRAIN) + dataset.size(SplitType::VALIDATION) +
                   dataset.size(SplitType::TEST);
    EXPECT_EQ(total, dataset.size());
}

// Test: K-fold cross-validation
TEST_F(DatasetTest, KFoldCrossValidation) {
    Dataset dataset;
    for (int i = 0; i < 15; ++i) {
        dataset.add_sample("Input " + std::to_string(i), "Target " + std::to_string(i));
    }

    dataset.setup_k_fold(5);
    EXPECT_EQ(dataset.get_num_folds(), 5);

    for (int fold = 0; fold < 5; ++fold) {
        std::vector<DataSample> train_data, val_data;
        dataset.get_fold(fold, train_data, val_data);

        EXPECT_GT(train_data.size(), 0);
        EXPECT_GT(val_data.size(), 0);
        EXPECT_EQ(train_data.size() + val_data.size(), dataset.size());
    }
}

// Test: Data augmentation
TEST_F(DatasetTest, DataAugmentation) {
    Dataset dataset;
    dataset.add_sample("Hello", "Hi");
    dataset.add_sample("Bye", "Goodbye");

    size_t original_size = dataset.size();

    // Set augmentation function
    dataset.set_augmentation([](const DataSample& sample) {
        return DataSample("[AUG] " + sample.input, sample.target);
    });

    dataset.augment_data(1);  // 1 augmented per original

    EXPECT_EQ(dataset.size(), original_size * 2);

    // Check some samples are augmented
    bool found_augmented = false;
    for (const auto& sample : dataset) {
        if (sample.input.find("[AUG]") != std::string::npos) {
            found_augmented = true;
            break;
        }
    }
    EXPECT_TRUE(found_augmented);
}

// Test: Filter by length
TEST_F(DatasetTest, FilterByLength) {
    Dataset dataset;
    dataset.add_sample("X", "Y");                                                    // Total: 2
    dataset.add_sample("Hello there", "Hi back");                                    // Total: ~18
    dataset.add_sample("This is a much longer input", "And a longer response too");  // Total: ~50

    size_t original_size = dataset.size();
    dataset.filter_by_length(10, 30);

    EXPECT_LT(dataset.size(), original_size);

    // Check all remaining samples are within range
    for (const auto& sample : dataset) {
        size_t total_len = sample.input.length() + sample.target.length();
        EXPECT_GE(total_len, 10);
        EXPECT_LE(total_len, 30);
    }
}

// Test: Filter by pattern
TEST_F(DatasetTest, FilterByPattern) {
    Dataset dataset;
    dataset.add_sample("Hello world", "Hi");
    dataset.add_sample("Goodbye world", "Bye");
    dataset.add_sample("How are you", "Fine");

    // Keep only samples with "world"
    dataset.filter_by_pattern("world", true);

    EXPECT_EQ(dataset.size(), 2);

    // All remaining should contain "world"
    for (const auto& sample : dataset) {
        bool has_pattern = (sample.input.find("world") != std::string::npos ||
                            sample.target.find("world") != std::string::npos);
        EXPECT_TRUE(has_pattern);
    }
}

// Test: Preprocessing
TEST_F(DatasetTest, Preprocessing) {
    Dataset dataset;
    dataset.add_sample("HELLO", "WORLD");
    dataset.add_sample("GOODBYE", "BYE");

    // Set preprocessing to lowercase
    dataset.set_preprocessing([](const std::string& text) {
        std::string lower = text;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower;
    });

    dataset.apply_preprocessing();

    // Check all samples are lowercase
    for (const auto& sample : dataset) {
        bool is_lowercase_input = (sample.input == "hello" || sample.input == "goodbye");
        bool is_lowercase_target = (sample.target == "world" || sample.target == "bye");
        EXPECT_TRUE(is_lowercase_input);
        EXPECT_TRUE(is_lowercase_target);
    }
}

// Test: Lowercase convenience method
TEST_F(DatasetTest, LowercaseMethod) {
    Dataset dataset;
    dataset.add_sample("HELLO", "WORLD");

    dataset.lowercase();

    auto data = dataset.get_all();
    EXPECT_EQ(data[0].input, "hello");
    EXPECT_EQ(data[0].target, "world");
}

// Test: LazyDataset
TEST_F(DatasetTest, LazyDataset) {
    create_test_conversation_file();

    LazyDataset lazy("test_conversations.txt");

    EXPECT_GT(lazy.size(), 0);

    // Load single sample
    auto sample = lazy.get_sample(0);
    EXPECT_FALSE(sample.input.empty());
    EXPECT_FALSE(sample.target.empty());

    // Load range
    auto samples = lazy.load_range(0, 2);
    EXPECT_EQ(samples.size(), 2);
}

// Regression test (TD-057): a bare "INPUT:"/"RESPONSE:" line (no space or
// content after the colon) used to make LazyDataset::get_sample() call
// substr() with an offset past the end of the line, throwing
// std::out_of_range instead of returning an empty field.
TEST_F(DatasetTest, LazyDatasetHandlesBareLegacyKeys) {
    {
        std::ofstream file("test_bare_keys.txt");
        file << "INPUT:\nRESPONSE:\n\n";
        file.close();
    }

    LazyDataset lazy("test_bare_keys.txt");
    ASSERT_GT(lazy.size(), 0u);

    DataSample sample;
    EXPECT_NO_THROW(sample = lazy.get_sample(0));
    EXPECT_TRUE(sample.input.empty());
    EXPECT_TRUE(sample.target.empty());
}

// Test: Multiple format auto-detection
TEST_F(DatasetTest, FormatAutoDetection) {
    create_test_json_file();
    create_test_csv_file();

    Dataset json_dataset;
    EXPECT_TRUE(json_dataset.load_from_file("test_data.json"));
    EXPECT_GT(json_dataset.size(), 0);

    Dataset csv_dataset;
    EXPECT_TRUE(csv_dataset.load_from_file("test_data.csv"));
    EXPECT_GT(csv_dataset.size(), 0);

    // Both should have loaded same data
    EXPECT_EQ(json_dataset.size(), csv_dataset.size());
}

// Test: Batch iterator edge cases
TEST_F(DatasetTest, BatchIteratorEdgeCases) {
    Dataset dataset;
    dataset.add_sample("A", "1");
    dataset.add_sample("B", "2");
    dataset.add_sample("C", "3");
    dataset.split(1.0, 0.0, 0.0);

    // Batch size larger than dataset
    int count = 0;
    for (auto batch : dataset.get_batch_range(SplitType::TRAIN, 10)) {
        EXPECT_LE(batch.size(), 3);
        count++;
    }
    EXPECT_EQ(count, 1);

    // Batch size of 1
    count = 0;
    for (auto batch : dataset.get_batch_range(SplitType::TRAIN, 1)) {
        EXPECT_EQ(batch.size(), 1);
        count++;
    }
    EXPECT_EQ(count, 3);
}

// Test: K-fold with invalid parameters
TEST_F(DatasetTest, KFoldInvalidParams) {
    Dataset dataset;
    dataset.add_sample("A", "1");

    // K=1 should be invalid
    testing::internal::CaptureStderr();
    dataset.setup_k_fold(1);
    std::string output = testing::internal::GetCapturedStderr();
    EXPECT_GT(output.length(), 0);
}

// Test: Statistics after split
TEST_F(DatasetTest, StatisticsAfterSplit) {
    Dataset dataset;
    dataset.load_from_file("test_conversations.txt");
    dataset.split(0.6, 0.2, 0.2);

    const auto& stats = dataset.get_stats();
    EXPECT_EQ(stats.total_samples, dataset.size());
    EXPECT_EQ(stats.train_samples, dataset.size(SplitType::TRAIN));
    EXPECT_EQ(stats.validation_samples, dataset.size(SplitType::VALIDATION));
    EXPECT_EQ(stats.test_samples, dataset.size(SplitType::TEST));
}

// Test: Filter invalidates split
TEST_F(DatasetTest, FilterInvalidatesSplit) {
    Dataset dataset;
    dataset.load_from_file("test_conversations.txt");
    dataset.split(0.6, 0.2, 0.2);

    EXPECT_TRUE(dataset.is_split());

    dataset.filter_by_length(5, 100);

    // Split should be invalidated
    EXPECT_FALSE(dataset.is_split());
}

// ============================================================================
// JSONL Training Format Tests
// ============================================================================

// Test: Load JSONL training format (input/response keys with optional meta)
TEST_F(DatasetTest, LoadJsonlTrainingFormat) {
    create_test_jsonl_training_file();

    Dataset dataset;
    EXPECT_TRUE(dataset.load_from_file("test_data.jsonl"));
    EXPECT_EQ(dataset.size(), 3);
    EXPECT_FALSE(dataset.empty());

    const auto& data = dataset.get_all();
    EXPECT_EQ(data[0].input, "Hello");
    EXPECT_EQ(data[0].target, "Hi there!");
    EXPECT_EQ(data[1].input, "How are you?");
    EXPECT_EQ(data[1].target, "I'm doing great!");
}

// Test: JSONL metadata is preserved in DataSample.meta
TEST_F(DatasetTest, LoadJsonlPreservesMetadata) {
    create_test_jsonl_training_file();

    Dataset dataset;
    ASSERT_TRUE(dataset.load_from_file("test_data.jsonl"));
    ASSERT_EQ(dataset.size(), 3);

    const auto& data = dataset.get_all();

    // Second sample has domain/task_type/quality
    EXPECT_EQ(data[1].meta.domain, "dialogue");
    EXPECT_EQ(data[1].meta.task_type, "chat");
    EXPECT_NEAR(data[1].meta.quality, 0.8f, 1e-4f);

    // Third sample has language/token_count
    EXPECT_EQ(data[2].meta.language, "en");
    EXPECT_EQ(data[2].meta.token_count, 15);

    // First sample: no optional fields → sentinel defaults
    EXPECT_LT(data[0].meta.quality, 0.0f);
    EXPECT_LT(data[0].meta.token_count, 0);
    EXPECT_TRUE(data[0].meta.domain.empty());
}

// Test: save_to_file with JSONL format (default) round-trips correctly
TEST_F(DatasetTest, SaveJsonlFormatAndReload) {
    Dataset dataset;
    dataset.add_sample("Hello", "Hi");
    dataset.add_sample("Bye", "Goodbye");

    EXPECT_TRUE(dataset.save_to_file("test_output.jsonl", "jsonl"));

    Dataset loaded;
    EXPECT_TRUE(loaded.load_from_file("test_output.jsonl"));
    EXPECT_EQ(loaded.size(), 2);

    const auto& data = loaded.get_all();
    EXPECT_EQ(data[0].input, "Hello");
    EXPECT_EQ(data[0].target, "Hi");
    EXPECT_EQ(data[1].input, "Bye");
    EXPECT_EQ(data[1].target, "Goodbye");
}

// Test: save_to_file default format is JSONL
TEST_F(DatasetTest, SaveDefaultFormatIsJsonl) {
    Dataset dataset;
    dataset.add_sample("q", "a");

    EXPECT_TRUE(dataset.save_to_file("test_output.jsonl"));

    // The saved file should be parseable as JSONL
    std::ifstream f("test_output.jsonl");
    std::string first_line;
    std::getline(f, first_line);
    EXPECT_FALSE(first_line.empty());
    EXPECT_EQ(first_line.front(), '{');
}

// Test: LazyDataset with JSONL training file
TEST_F(DatasetTest, LazyDataset_JsonlFormat) {
    create_test_jsonl_training_file();

    LazyDataset lazy("test_data.jsonl");

    EXPECT_GT(lazy.size(), 0);

    auto sample = lazy.get_sample(0);
    EXPECT_EQ(sample.input, "Hello");
    EXPECT_EQ(sample.target, "Hi there!");

    auto samples = lazy.load_range(0, 2);
    EXPECT_EQ(samples.size(), 2);
    EXPECT_EQ(samples[1].input, "How are you?");
}

// Test: JSONL format auto-detected vs legacy conversation format
TEST_F(DatasetTest, FormatAutoDetection_JsonlVsLegacy) {
    create_test_jsonl_training_file();

    Dataset jsonl_dataset;
    EXPECT_TRUE(jsonl_dataset.load_from_file("test_data.jsonl"));
    EXPECT_EQ(jsonl_dataset.size(), 3);

    Dataset legacy_dataset;
    EXPECT_TRUE(legacy_dataset.load_from_file("test_conversations.txt"));
    EXPECT_EQ(legacy_dataset.size(), 3);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
