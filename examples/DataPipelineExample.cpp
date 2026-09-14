/**
 * @file DataPipelineExample.cpp
 * @brief Demonstration of efficient batching
 *
 * This example shows how to use EfficientBatching for dynamic/bucketed batching and data
 * augmentation. It used to also demonstrate parallel data loading via TokenBatchLoader
 * (src/ParallelDataLoader.hpp) — retired as part of TD-170 (September 14, 2026) once it became
 * clear its real value-adds (background tokenization prefetch, shuffling, batch grouping for
 * gradient accumulation) were each already duplicated by existing, working ChatbotTrainer
 * machinery, and its padding/batch-dimension output had no model to consume it
 * (EncoderDecoderModel::forward() takes one sequence at a time, no batch dimension anywhere in
 * this codebase's Matrix/model stack) — see TECHNICAL_DEBT.md's resolved archive. The two demo
 * functions that used it (`example_parallel_loading`, `example_training_loop`) were removed along
 * with it rather than rewritten against a replacement, since none exists.
 */

#include "EfficientBatching.hpp"
#include <iostream>
#include <iomanip>

// Example 1: Basic efficient batching
void example_basic_batching() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "EXAMPLE 1: Basic Efficient Batching\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Create test sequences of varying lengths
    std::vector<std::vector<int>> sequences = {
        {1, 2, 3},              // length 3
        {4, 5, 6, 7},           // length 4
        {8, 9},                 // length 2
        {10, 11, 12, 13, 14},   // length 5
        {15, 16, 17},           // length 3
        {18, 19, 20, 21},       // length 4
        {22, 23, 24, 25, 26, 27}, // length 6
        {28, 29, 30}            // length 3
    };
    
    std::cout << "\nCreating batches WITHOUT sorting by length...\n";
    auto batches_unsorted = EfficientBatching::create_dynamic_batches(
        sequences, 3, 0, PaddingStrategy::RIGHT, false
    );
    
    auto stats_unsorted = EfficientBatching::calculate_statistics(batches_unsorted);
    std::cout << "Unsorted batches - Padding ratio: " << std::fixed << std::setprecision(2)
              << (stats_unsorted.padding_ratio * 100) << "%\n";
    
    std::cout << "\nCreating batches WITH sorting by length...\n";
    auto batches_sorted = EfficientBatching::create_dynamic_batches(
        sequences, 3, 0, PaddingStrategy::RIGHT, true
    );
    
    auto stats_sorted = EfficientBatching::calculate_statistics(batches_sorted);
    std::cout << "Sorted batches - Padding ratio: " << std::fixed << std::setprecision(2)
              << (stats_sorted.padding_ratio * 100) << "%\n";
    
    double improvement = ((stats_unsorted.padding_ratio - stats_sorted.padding_ratio) 
                         / stats_unsorted.padding_ratio) * 100;
    std::cout << "\nImprovement from sorting: " << std::fixed << std::setprecision(1) 
              << improvement << "% reduction in padding\n";
}

// Example 2: Bucketing strategy
void example_bucketing() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "EXAMPLE 2: Bucketing Strategy\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Create sequences with wide length variation
    std::vector<std::vector<int>> sequences;
    for (int i = 0; i < 100; ++i) {
        int length = 5 + (i % 50);  // Lengths from 5 to 54
        std::vector<int> seq(length);
        std::iota(seq.begin(), seq.end(), i * 100);
        sequences.push_back(seq);
    }
    
    // Configure buckets
    BucketConfig config;
    config.bucket_boundaries = {10, 20, 30, 40};  // 5 buckets
    config.max_tokens_per_batch = 500;
    config.shuffle_buckets = false;
    
    std::cout << "Creating bucketed batches...\n";
    std::cout << "Bucket boundaries: [";
    for (size_t i = 0; i < config.bucket_boundaries.size(); ++i) {
        std::cout << config.bucket_boundaries[i];
        if (i < config.bucket_boundaries.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "Max tokens per batch: " << config.max_tokens_per_batch << "\n";
    
    auto batches = EfficientBatching::create_bucketed_batches(
        sequences, config, 0, PaddingStrategy::RIGHT
    );
    
    auto stats = EfficientBatching::calculate_statistics(batches);
    std::cout << "\nBucket Statistics:\n";
    std::cout << "Total batches: " << stats.num_batches << "\n";
    std::cout << "Avg batch size: " << std::fixed << std::setprecision(1) 
              << stats.avg_batch_size << " sequences\n";
    std::cout << "Padding ratio: " << std::fixed << std::setprecision(2)
              << (stats.padding_ratio * 100) << "%\n";
    std::cout << "Efficiency score: " << std::fixed << std::setprecision(2)
              << (stats.efficiency_score * 100) << "%\n";
}

// Example 3: Data augmentation
void example_augmentation() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "EXAMPLE 3: Data Augmentation\n";
    std::cout << std::string(80, '=') << "\n";
    
    std::vector<std::vector<int>> original_sequences = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
    };
    
    std::cout << "Original sequences:\n";
    for (size_t i = 0; i < original_sequences.size(); ++i) {
        std::cout << "Seq " << i << ": [";
        for (size_t j = 0; j < original_sequences[i].size(); ++j) {
            std::cout << original_sequences[i][j];
            if (j < original_sequences[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    
    // Token dropout
    std::cout << "\n--- Token Dropout (30% probability) ---\n";
    auto dropout_sequences = original_sequences;
    AugmentationConfig dropout_config;
    dropout_config.enable_token_dropout = true;
    dropout_config.token_dropout_prob = 0.3f;
    dropout_config.seed = 42;
    
    EfficientBatching::apply_augmentation(dropout_sequences, dropout_config);
    
    for (size_t i = 0; i < dropout_sequences.size(); ++i) {
        std::cout << "Seq " << i << ": [";
        for (size_t j = 0; j < dropout_sequences[i].size(); ++j) {
            std::cout << dropout_sequences[i][j];
            if (j < dropout_sequences[i].size() - 1) std::cout << ", ";
        }
        std::cout << "] (length: " << dropout_sequences[i].size() << ")\n";
    }
    
    // Token masking
    std::cout << "\n--- Token Masking (20% probability, mask=999) ---\n";
    auto masking_sequences = original_sequences;
    AugmentationConfig masking_config;
    masking_config.enable_token_masking = true;
    masking_config.token_mask_prob = 0.2f;
    masking_config.mask_token_id = 999;
    masking_config.seed = 123;
    
    EfficientBatching::apply_augmentation(masking_sequences, masking_config);
    
    for (size_t i = 0; i < masking_sequences.size(); ++i) {
        std::cout << "Seq " << i << ": [";
        for (size_t j = 0; j < masking_sequences[i].size(); ++j) {
            std::cout << masking_sequences[i][j];
            if (j < masking_sequences[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                  Data Pipeline Examples                                   ║\n";
    std::cout << "║                                                                            ║\n";
    std::cout << "║  Demonstration of efficient batching and parallel data loading            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";
    
    try {
        example_basic_batching();
        example_bucketing();
        example_augmentation();

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "All examples completed successfully!\n";
        std::cout << std::string(80, '=') << "\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
