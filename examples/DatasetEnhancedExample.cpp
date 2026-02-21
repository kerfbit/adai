/**
 * @file DatasetEnhancedExample.cpp
 * @brief Example demonstrating enhanced Dataset features
 * 
 * This example demonstrates:
 * - Iterator interface for range-based for loops
 * - Batch iteration for mini-batch training
 * - JSON and CSV format support
 * - Stratified splitting
 * - K-fold cross-validation
 * - Data augmentation
 * - Data filtering and preprocessing
 * - Lazy loading for large datasets
 * 
 * @version 2.0
 * @date January 2026
 */

#include <iostream>
#include <iomanip>
#include <cctype>
#include "Dataset.hpp"

// ANSI colors
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define GREEN   "\033[32m"
#define BLUE    "\033[34m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"

void print_section(const std::string& title) {
    std::cout << "\n" << BOLD << CYAN << "========================================" << RESET << "\n";
    std::cout << BOLD << CYAN << title << RESET << "\n";
    std::cout << BOLD << CYAN << "========================================" << RESET << "\n\n";
}

void create_sample_datasets() {
    // Create conversation format sample
    std::ofstream conv("dataset_conversation.txt");
    conv << "INPUT: Hello, how are you?\n";
    conv << "RESPONSE: I'm doing great, thanks for asking!\n\n";
    conv << "INPUT: What's the weather like?\n";
    conv << "RESPONSE: I don't have access to weather information.\n\n";
    conv << "INPUT: Tell me a joke\n";
    conv << "RESPONSE: Why did the programmer quit? Because they didn't get arrays!\n\n";
    conv.close();
    
    // Create JSON format sample
    std::ofstream json("dataset.json");
    json << "{\"input\": \"What is AI?\", \"target\": \"AI is artificial intelligence.\"}\n";
    json << "{\"input\": \"Explain machine learning\", \"target\": \"ML is a subset of AI.\"}\n";
    json << "{\"input\": \"What is deep learning?\", \"target\": \"Deep learning uses neural networks.\"}\n";
    json.close();
    
    // Create CSV format sample
    std::ofstream csv("dataset.csv");
    csv << "input,target\n";
    csv << "\"Hello\",\"Hi there!\"\n";
    csv << "\"Goodbye\",\"See you later!\"\n";
    csv << "\"Thank you\",\"You're welcome!\"\n";
    csv.close();
}

int main() {
    std::cout << BOLD << GREEN << "Dataset Enhanced Features Demo\n" << RESET;
    std::cout << "Demonstrating advanced dataset capabilities\n";
    
    // Create sample data files
    create_sample_datasets();
    
    // ============================================================
    print_section("1. Iterator Interface");
    // ============================================================
    
    Dataset dataset1;
    dataset1.load_from_file("dataset_conversation.txt");
    
    std::cout << "Using range-based for loop with iterators:\n";
    int count = 0;
    for (const auto& sample : dataset1) {
        std::cout << "  [" << ++count << "] " << sample.input 
                  << " -> " << sample.target << "\n";
    }
    
    // ============================================================
    print_section("2. Batch Iteration");
    // ============================================================
    
    Dataset dataset2;
    dataset2.load_from_file("dataset_conversation.txt");
    dataset2.split(0.7, 0.2, 0.1);
    
    std::cout << "Iterating in batches of 2:\n";
    int batch_num = 0;
    for (auto batch : dataset2.get_batch_range(SplitType::TRAIN, 2)) {
        std::cout << "  Batch " << ++batch_num << " (size: " << batch.size() << "):\n";
        for (const auto& sample : batch) {
            std::cout << "    - " << sample.input << "\n";
        }
    }
    
    // ============================================================
    print_section("3. JSON Format Support");
    // ============================================================
    
    Dataset dataset3;
    if (dataset3.load_from_file("dataset.json")) {
        std::cout << "✓ Successfully loaded JSON format\n";
        dataset3.print_stats();
        
        std::cout << "\nSamples:\n";
        for (const auto& sample : dataset3) {
            std::cout << "  Q: " << sample.input << "\n";
            std::cout << "  A: " << sample.target << "\n\n";
        }
    }
    
    // ============================================================
    print_section("4. CSV Format Support");
    // ============================================================
    
    Dataset dataset4;
    if (dataset4.load_from_file("dataset.csv")) {
        std::cout << "✓ Successfully loaded CSV format\n";
        dataset4.print_stats();
    }
    
    // ============================================================
    print_section("5. Stratified Splitting");
    // ============================================================
    
    Dataset dataset5;
    dataset5.load_from_file("dataset_conversation.txt");
    
    std::cout << "Standard split:\n";
    dataset5.split(0.7, 0.2, 0.1);
    std::cout << "  Train: " << dataset5.size(SplitType::TRAIN) << "\n";
    std::cout << "  Val: " << dataset5.size(SplitType::VALIDATION) << "\n";
    std::cout << "  Test: " << dataset5.size(SplitType::TEST) << "\n\n";
    
    std::cout << "Stratified split (by length):\n";
    dataset5.split_stratified(0.7, 0.2, 0.1, 3);
    std::cout << "  Train: " << dataset5.size(SplitType::TRAIN) << "\n";
    std::cout << "  Val: " << dataset5.size(SplitType::VALIDATION) << "\n";
    std::cout << "  Test: " << dataset5.size(SplitType::TEST) << "\n";
    
    // ============================================================
    print_section("6. K-Fold Cross-Validation");
    // ============================================================
    
    Dataset dataset6;
    dataset6.load_from_file("dataset_conversation.txt");
    dataset6.setup_k_fold(3);
    
    std::cout << "3-Fold Cross-Validation:\n";
    for (int fold = 0; fold < dataset6.get_num_folds(); ++fold) {
        std::vector<DataSample> train_data, val_data;
        dataset6.get_fold(fold, train_data, val_data);
        
        std::cout << "  Fold " << fold + 1 << ": "
                  << "Train=" << train_data.size() 
                  << ", Val=" << val_data.size() << "\n";
    }
    
    // ============================================================
    print_section("7. Data Augmentation");
    // ============================================================
    
    Dataset dataset7;
    dataset7.load_from_file("dataset_conversation.txt");
    
    std::cout << "Original size: " << dataset7.size() << "\n";
    
    // Set augmentation function (simple example: add prefix)
    dataset7.set_augmentation([](const DataSample& sample) {
        return DataSample("[Augmented] " + sample.input, sample.target);
    });
    
    dataset7.augment_data(1);  // 1 augmented sample per original
    std::cout << "After augmentation: " << dataset7.size() << "\n";
    
    std::cout << "\nAugmented samples:\n";
    count = 0;
    for (const auto& sample : dataset7) {
        std::cout << "  [" << ++count << "] " << sample.input << "\n";
        if (count >= 6) break;  // Show first 6
    }
    
    // ============================================================
    print_section("8. Data Filtering");
    // ============================================================
    
    Dataset dataset8;
    dataset8.load_from_file("dataset_conversation.txt");
    
    std::cout << "Original size: " << dataset8.size() << "\n";
    
    // Filter by length
    dataset8.filter_by_length(10, 100);
    std::cout << "After length filter (10-100 chars): " << dataset8.size() << "\n";
    
    // Filter by pattern
    Dataset dataset8b;
    dataset8b.load_from_file("dataset_conversation.txt");
    dataset8b.filter_by_pattern("joke", true);
    std::cout << "After pattern filter (contains 'joke'): " << dataset8b.size() << "\n";
    
    // ============================================================
    print_section("9. Preprocessing");
    // ============================================================
    
    Dataset dataset9;
    dataset9.load_from_file("dataset_conversation.txt");
    
    std::cout << "Before preprocessing:\n";
    auto sample = *dataset9.begin();
    std::cout << "  " << sample.input << "\n\n";
    
    // Set preprocessing function (lowercase)
    dataset9.set_preprocessing([](const std::string& text) {
        std::string lower = text;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower;
    });
    
    dataset9.apply_preprocessing();
    
    std::cout << "After preprocessing (lowercase):\n";
    sample = *dataset9.begin();
    std::cout << "  " << sample.input << "\n";
    
    // ============================================================
    print_section("10. Lazy Loading");
    // ============================================================
    
    LazyDataset lazy_dataset("dataset_conversation.txt");
    
    std::cout << "LazyDataset loaded (indexed): " << lazy_dataset.size() << " samples\n";
    std::cout << "Memory-efficient - samples loaded on demand\n\n";
    
    std::cout << "Loading sample 0:\n";
    auto lazy_sample = lazy_dataset.get_sample(0);
    std::cout << "  Input: " << lazy_sample.input << "\n";
    std::cout << "  Target: " << lazy_sample.target << "\n\n";
    
    std::cout << "Loading range (0-2):\n";
    auto range = lazy_dataset.load_range(0, 2);
    for (size_t i = 0; i < range.size(); ++i) {
        std::cout << "  [" << i << "] " << range[i].input << "\n";
    }
    
    // ============================================================
    print_section("Summary");
    // ============================================================
    
    std::cout << BOLD << GREEN << "✓ All enhanced features demonstrated!\n" << RESET;
    std::cout << "\nKey improvements:\n";
    std::cout << "  • Iterator interface for easy iteration\n";
    std::cout << "  • Batch iteration for mini-batch training\n";
    std::cout << "  • JSON and CSV format support\n";
    std::cout << "  • Stratified splitting for balanced datasets\n";
    std::cout << "  • K-fold cross-validation\n";
    std::cout << "  • Data augmentation hooks\n";
    std::cout << "  • Filtering and preprocessing\n";
    std::cout << "  • Lazy loading for large datasets\n";
    
    std::cout << "\n" << BOLD << BLUE << "Dataset abstraction is now production-ready!\n" << RESET;
    
    return 0;
}
