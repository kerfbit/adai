/**
 * @file DatasetBatchProcessingExample.cpp
 * @brief Examples demonstrating batch processing integration with Dataset system
 *
 * This file shows how to use the batch processing features built into Dataset itself
 * (get_batch_with_padding()/get_dynamic_batches()/get_batch_statistics()).
 *
 * Topics covered:
 * 1. Basic batch creation with padding
 * 2. Dynamic batching by sequence length
 * 3. Batch statistics and efficiency analysis
 *
 * This file used to also cover parallel loading with TokenBatchLoader and a simulated
 * training-pipeline integration built on it (src/ParallelDataLoader.hpp) — retired as part of
 * TD-170 (September 14, 2026) once it became clear its real value-adds (background tokenization
 * prefetch, shuffling, batch grouping for gradient accumulation) were each already duplicated by
 * existing, working ChatbotTrainer machinery, and its padding/batch-dimension output had no model
 * to consume it (EncoderDecoderModel::forward() takes one sequence at a time, no batch dimension
 * anywhere in this codebase's Matrix/model stack) — see TECHNICAL_DEBT.md's resolved archive. The
 * two demo functions that used it were removed along with it rather than rewritten against a
 * replacement, since none exists.
 *
 * @version 1.1
 * @date September 2026
 */

#include "Dataset.hpp"
#include "BatchProcessor.hpp"
#include "BPETokenizer.hpp"
#include "Matrix.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

// ============================================================================
// Example 1: Basic Batch Creation with Padding
// ============================================================================

void example1_basic_batching() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Example 1: Basic Batch Creation                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
    
    // Create a small dataset
    Dataset dataset;
    dataset.add_sample("Hello world", "Hi there");
    dataset.add_sample("How are you today?", "I'm doing well");
    dataset.add_sample("What's your name?", "My name is ADAI");
    dataset.add_sample("Tell me a joke", "Why did the chicken cross the road?");
    dataset.add_sample("Good morning!", "Good morning to you too!");
    
    dataset.split(0.8, 0.1, 0.1);
    
    std::cout << "Dataset loaded with " << dataset.size() << " samples\n";
    dataset.print_stats();
    std::cout << "\n";
    
    // Create simple tokenizer function (char-level for demo)
    auto simple_tokenizer = [](const std::string& text) {
        std::vector<int> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int>(static_cast<unsigned char>(c)));
        }
        return tokens;
    };
    
    // Get a batch with padding
    std::cout << "Creating batch of 3 samples...\n";
    TokenBatch batch = dataset.get_batch_with_padding(
        SplitType::TRAIN,
        0,      // Start index
        3,      // Batch size
        simple_tokenizer,
        0       // Pad token ID
    );
    
    std::cout << "Batch created!\n";
    std::cout << "  Batch size: " << batch.batch_size() << "\n";
    std::cout << "  Max length: " << batch.max_length << "\n";
    std::cout << "  Pad token ID: " << batch.pad_token_id << "\n\n";
    
    // Display each sequence
    for (int i = 0; i < batch.batch_size(); ++i) {
        std::cout << "  Sequence " << i << ":\n";
        std::cout << "    Actual length: " << batch.lengths[i] << "\n";
        std::cout << "    Padded length: " << batch.batch_token_ids[i].size() << "\n";
        std::cout << "    Tokens: [";
        for (size_t j = 0; j < std::min<size_t>(10, batch.batch_token_ids[i].size()); ++j) {
            std::cout << batch.batch_token_ids[i][j];
            if (j < 9 && j < batch.batch_token_ids[i].size() - 1) std::cout << ", ";
        }
        if (batch.batch_token_ids[i].size() > 10) {
            std::cout << ", ...";
        }
        std::cout << "]\n";
    }
    
    // Create padding mask
    std::cout << "\nCreating padding mask...\n";
    Matrix mask = create_padding_mask(batch);
    std::cout << "Mask shape: [" << mask.rows << " x " << mask.cols << "]\n";
    std::cout << "Mask (1=real token, 0=padding):\n";
    for (int i = 0; i < mask.rows; ++i) {
        std::cout << "  [";
        for (int j = 0; j < std::min(20, mask.cols); ++j) {
            std::cout << static_cast<int>(mask(i, j));
            if (j < 19 && j < mask.cols - 1) std::cout << " ";
        }
        if (mask.cols > 20) std::cout << " ...";
        std::cout << "]\n";
    }
    
    std::cout << "\n✓ Example 1 complete!\n";
}

// ============================================================================
// Example 2: Dynamic Batching by Sequence Length
// ============================================================================

void example2_dynamic_batching() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Example 2: Dynamic Batching                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
    
    // Create dataset with varying length samples
    Dataset dataset;
    dataset.add_sample("Hi", "Hello");
    dataset.add_sample("How are you?", "I'm good");
    dataset.add_sample("This is a much longer sentence with many more words", 
                      "Yes, this response is also quite lengthy indeed");
    dataset.add_sample("Short one", "Brief");
    dataset.add_sample("Medium length sentence here", "Another medium response");
    dataset.add_sample("Very very very very very long sentence with lots of words here",
                      "An equally long response with many tokens as well");
    dataset.add_sample("Hey there", "Hi back");
    dataset.add_sample("What's up?", "Not much");
    
    dataset.split(1.0, 0.0, 0.0);  // All in training set
    
    auto simple_tokenizer = [](const std::string& text) {
        std::vector<int> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int>(static_cast<unsigned char>(c)));
        }
        return tokens;
    };
    
    std::cout << "Dataset with varying sequence lengths:\n";
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto samples = dataset.get_all();
        auto tokens = simple_tokenizer(samples[i].input);
        std::cout << "  Sample " << i << ": " << tokens.size() << " tokens\n";
    }
    std::cout << "\n";
    
    // Compare fixed vs dynamic batching
    std::cout << "--- Fixed Batching (batch_size=3) ---\n";
    std::vector<TokenBatch> fixed_batches;
    for (size_t i = 0; i < dataset.size(); i += 3) {
        TokenBatch batch = dataset.get_batch_with_padding(
            SplitType::TRAIN, i, 3, simple_tokenizer, 0);
        if (!batch.is_empty()) {
            fixed_batches.push_back(batch);
        }
    }
    
    float fixed_total_eff = 0.0f;
    size_t fixed_total_tokens = 0;
    for (size_t i = 0; i < fixed_batches.size(); ++i) {
        const auto& batch = fixed_batches[i];
        std::vector<TokenBatch> batch_vec = {batch};
        BatchStats stats = compute_batch_stats(batch_vec);
        float efficiency = 1.0f - stats.padding_ratio;
        fixed_total_eff += efficiency;
        fixed_total_tokens += stats.total_tokens;
        
        std::cout << "Batch " << i << ": size=" << batch.batch_size()
                  << ", max_len=" << batch.max_length
                  << ", eff=" << std::fixed << std::setprecision(1)
                  << (efficiency * 100) << "%\n";
    }
    
    float fixed_avg_eff = fixed_total_eff / fixed_batches.size();
    std::cout << "Total tokens: " << fixed_total_tokens << "\n";
    std::cout << "Average efficiency: " << (fixed_avg_eff * 100) << "%\n\n";
    
    // Dynamic batching
    std::cout << "--- Dynamic Batching (max_batch_size=3, tolerance=5) ---\n";
    auto dynamic_batches = dataset.get_dynamic_batches(
        SplitType::TRAIN,
        simple_tokenizer,
        3,   // max_batch_size
        5,   // length_tolerance
        0    // pad_token_id
    );
    
    float dynamic_total_eff = 0.0f;
    size_t dynamic_total_tokens = 0;
    for (size_t i = 0; i < dynamic_batches.size(); ++i) {
        const auto& batch = dynamic_batches[i];
        std::vector<TokenBatch> batch_vec = {batch};
        BatchStats stats = compute_batch_stats(batch_vec);
        float efficiency = 1.0f - stats.padding_ratio;
        dynamic_total_eff += efficiency;
        dynamic_total_tokens += stats.total_tokens;
        
        std::cout << "Batch " << i << ": size=" << batch.batch_size()
                  << ", max_len=" << batch.max_length
                  << ", eff=" << std::fixed << std::setprecision(1)
                  << (efficiency * 100) << "%\n";
    }
    
    float dynamic_avg_eff = dynamic_total_eff / dynamic_batches.size();
    std::cout << "Total tokens: " << dynamic_total_tokens << "\n";
    std::cout << "Average efficiency: " << (dynamic_avg_eff * 100) << "%\n\n";
    
    // Comparison
    std::cout << "--- Comparison ---\n";
    float token_reduction = (fixed_total_tokens - dynamic_total_tokens) * 100.0f 
                           / fixed_total_tokens;
    float eff_improvement = (dynamic_avg_eff - fixed_avg_eff) * 100.0f;
    
    std::cout << "Token reduction: " << std::fixed << std::setprecision(1)
              << token_reduction << "%\n";
    std::cout << "Efficiency improvement: +" << eff_improvement << "%\n";
    
    std::cout << "\n✓ Example 2 complete!\n";
}

// ============================================================================
// Example 3: Batch Statistics and Efficiency Analysis
// ============================================================================

void example3_batch_statistics() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Example 3: Batch Statistics                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
    
    // Create dataset
    Dataset dataset;
    dataset.add_sample("Short", "Brief");
    dataset.add_sample("A bit longer sentence", "Medium response");
    dataset.add_sample("This is getting even longer now", "Yes indeed it is");
    dataset.add_sample("Very very long sentence with many many words",
                      "Another very long response here");
    dataset.add_sample("Tiny", "Small");
    dataset.add_sample("Medium", "Average");
    
    dataset.split(1.0, 0.0, 0.0);
    
    auto simple_tokenizer = [](const std::string& text) {
        std::vector<int> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int>(static_cast<unsigned char>(c)));
        }
        return tokens;
    };
    
    std::cout << "Analyzing batch statistics for different batch sizes...\n\n";
    std::cout << std::setw(12) << "Batch Size" << " | "
              << std::setw(10) << "Efficiency" << " | "
              << std::setw(12) << "Avg Padding" << " | "
              << std::setw(12) << "Total Tokens" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    std::vector<size_t> batch_sizes = {2, 3, 4, 6};
    for (size_t bs : batch_sizes) {
        auto stats = dataset.get_batch_statistics(
            SplitType::TRAIN, simple_tokenizer, bs);
        
        float avg_padding = (stats.total_tokens - stats.actual_tokens)
                          / static_cast<float>(stats.avg_batch_size);
        float efficiency = 1.0f - stats.padding_ratio;
        
        std::cout << std::setw(12) << bs << " | "
                  << std::setw(9) << std::fixed << std::setprecision(1)
                  << (efficiency * 100) << "% | "
                  << std::setw(11) << std::setprecision(1) << avg_padding << " | "
                  << std::setw(12) << stats.total_tokens << "\n";
    }
    
    std::cout << "\n✓ Example 3 complete!\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║   Dataset Batch Processing Integration Examples           ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    
    try {
        example1_basic_batching();
        example2_dynamic_batching();
        example3_batch_statistics();

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                                                            ║\n";
        std::cout << "║   All examples completed successfully!                    ║\n";
        std::cout << "║                                                            ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
