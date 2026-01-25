/**
 * @file DataPipelineExample.cpp
 * @brief Demonstration of efficient batching and parallel data loading
 * 
 * This example shows how to use the data pipeline components for efficient
 * training with dynamic batching, parallel loading, and data augmentation.
 */

#include "EfficientBatching.hpp"
#include "ParallelDataLoader.hpp"
#include "Dataset.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

// Helper function to print batch info
void print_batch_info(const SequenceBatch& batch, size_t batch_num) {
    std::cout << "\n=== Batch " << batch_num << " ===\n";
    std::cout << "Number of sequences: " << batch.sequences.size() << "\n";
    std::cout << "Max length: " << batch.max_length << "\n";
    std::cout << "Total tokens: " << batch.total_tokens() << "\n";
    std::cout << "Padding tokens: " << batch.padding_tokens() << "\n";
    std::cout << "Padding ratio: " << std::fixed << std::setprecision(2) 
              << (batch.padding_ratio() * 100) << "%\n";
    
    // Show first sequence and mask
    if (!batch.sequences.empty()) {
        std::cout << "First sequence: [";
        for (size_t i = 0; i < std::min(size_t(10), batch.sequences[0].size()); ++i) {
            std::cout << batch.sequences[0][i];
            if (i < std::min(size_t(10), batch.sequences[0].size()) - 1) std::cout << ", ";
        }
        if (batch.sequences[0].size() > 10) std::cout << ", ...";
        std::cout << "]\n";
        
        std::cout << "First mask: [";
        for (size_t i = 0; i < std::min(size_t(10), batch.masks[0].size()); ++i) {
            std::cout << batch.masks[0][i];
            if (i < std::min(size_t(10), batch.masks[0].size()) - 1) std::cout << ", ";
        }
        if (batch.masks[0].size() > 10) std::cout << ", ...";
        std::cout << "]\n";
    }
}

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

// Example 4: Parallel data loading
void example_parallel_loading() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "EXAMPLE 4: Parallel Data Loading\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Create a dataset
    Dataset dataset;
    
    std::cout << "Creating dataset with 200 samples...\n";
    for (int i = 0; i < 200; ++i) {
        std::string input = "Input " + std::to_string(i);
        std::string target = "Response " + std::to_string(i);
        
        // Varying length text
        int extra_words = i % 10;
        for (int j = 0; j < extra_words; ++j) {
            input += " word" + std::to_string(j);
        }
        
        dataset.add_sample(input, target);
    }
    
    dataset.split(0.8, 0.2, 0.0);  // 80% train, 20% val, 0% test
    
    std::cout << "Dataset split: " << dataset.size(SplitType::TRAIN) << " train, "
              << dataset.size(SplitType::VALIDATION) << " val\n";
    
    // Configure data loader
    DataLoaderConfig loader_config;
    loader_config.batch_size = 16;
    loader_config.num_workers = 4;
    loader_config.prefetch_factor = 2;
    loader_config.shuffle = true;
    loader_config.use_dynamic_batching = true;
    
    std::cout << "\nData Loader Configuration:\n";
    std::cout << "Batch size: " << loader_config.batch_size << "\n";
    std::cout << "Number of workers: " << loader_config.num_workers << "\n";
    std::cout << "Prefetch factor: " << loader_config.prefetch_factor << "\n";
    std::cout << "Dynamic batching: " << (loader_config.use_dynamic_batching ? "Yes" : "No") << "\n";
    
    // Create data loader
    ParallelDataLoader loader(dataset, loader_config);
    
    std::cout << "\nStarting parallel data loading...\n";
    std::cout << "Expected batches per epoch: " << loader.num_batches() << "\n";
    
    // Time the data loading
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DataLoaderIterator iter(loader);
    size_t batches_processed = 0;
    size_t total_sequences = 0;
    
    while (auto batch = iter.next()) {
        ++batches_processed;
        total_sequences += batch->sequences.size();
        
        // Print info for first 3 batches
        if (batches_processed <= 3) {
            print_batch_info(*batch, batches_processed);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n=== Loading Statistics ===\n";
    std::cout << "Batches processed: " << batches_processed << "\n";
    std::cout << "Total sequences: " << total_sequences << "\n";
    std::cout << "Time taken: " << duration.count() << " ms\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(1)
              << (total_sequences * 1000.0 / duration.count()) << " sequences/second\n";
}

// Example 5: Training loop simulation
void example_training_loop() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "EXAMPLE 5: Training Loop Simulation\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Create dataset
    Dataset dataset;
    
    for (int i = 0; i < 100; ++i) {
        std::string input = "Training input " + std::to_string(i);
        std::string target = "Training response " + std::to_string(i);
        
        // Add varying length text
        int extra_words = i % 8;
        for (int j = 0; j < extra_words; ++j) {
            input += " data" + std::to_string(j);
        }
        
        dataset.add_sample(input, target);
    }
    
    dataset.split(1.0, 0.0, 0.0);  // All training data
    
    // Configure loader with augmentation
    DataLoaderConfig config;
    config.batch_size = 8;
    config.num_workers = 2;
    config.shuffle = true;
    config.use_dynamic_batching = true;
    config.augmentation_config.enable_token_masking = true;
    config.augmentation_config.token_mask_prob = 0.15f;
    config.augmentation_config.enable_token_dropout = true;
    config.augmentation_config.token_dropout_prob = 0.1f;
    
    ParallelDataLoader loader(dataset, config);
    
    std::cout << "Simulating 3 training epochs...\n\n";
    
    for (int epoch = 0; epoch < 3; ++epoch) {
        std::cout << "=== Epoch " << (epoch + 1) << " ===\n";
        
        DataLoaderIterator iter(loader);
        size_t step = 0;
        double total_loss = 0.0;
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        while (auto batch = iter.next()) {
            ++step;
            
            // Simulate training step (just measure batch processing time)
            auto step_start = std::chrono::high_resolution_clock::now();
            
            // In real training, you would:
            // 1. Forward pass through model
            // 2. Calculate loss
            // 3. Backward pass
            // 4. Update weights
            
            // Simulate some processing time
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            
            auto step_end = std::chrono::high_resolution_clock::now();
            auto step_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                step_end - step_start
            );
            
            // Fake loss that decreases
            double loss = 2.0 / (epoch + 1) + (0.1 * (loader.num_batches() - step) / loader.num_batches());
            total_loss += loss;
            
            if (step % 5 == 0) {
                std::cout << "Step " << std::setw(3) << step << "/" << loader.num_batches()
                          << " | Loss: " << std::fixed << std::setprecision(4) << loss
                          << " | Batch time: " << step_duration.count() << " μs"
                          << " | Queue size: " << loader.queue_size() << "\n";
            }
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            epoch_end - epoch_start
        );
        
        double avg_loss = total_loss / loader.num_batches();
        
        std::cout << "Epoch " << (epoch + 1) << " complete | "
                  << "Avg Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Time: " << epoch_duration.count() << " ms\n\n";
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
        example_parallel_loading();
        example_training_loop();
        
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "All examples completed successfully!\n";
        std::cout << std::string(80, '=') << "\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
