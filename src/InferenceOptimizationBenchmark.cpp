/**
 * Inference Optimization Benchmark
 *
 * This program benchmarks the inference optimizations:
 * 1. KV Cache for autoregressive generation
 * 2. Batch processing for multiple sequences
 * 3. Combined optimizations
 *
 * Expected Results:
 * - KV Cache: ~2-3x speedup for generation
 * - Batch Processing: ~N/2 speedup for N sequences (with proper batching)
 * - Combined: ~4-6x total improvement
 */

#include <iostream>
#include <vector>
#include <string>
#include "Decoder.hpp"
#include "EncoderDecoderModel.hpp"
#include "BPETokenizer.hpp"
#include "TextGenerator.hpp"
#include "KVCache.hpp"
#include "BatchProcessor.hpp"
#include "PerformanceProfiler.hpp"

// Helper function to create a small test model
EncoderDecoderModel* create_test_model(int vocab_size) {
    return new EncoderDecoderModel(
        128,  // d_model (small for faster testing)
        4,    // num_heads
        512,  // d_ff
        2,    // num_encoder_layers
        2,    // num_decoder_layers
        vocab_size,
        256   // max_seq_length
    );
}

// Benchmark 1: KV Cache vs No Cache
void benchmark_kv_cache() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark 1: KV Cache Optimization" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create model
    int vocab_size = 1000;
    LLMDecoder decoder(vocab_size, 128, 2, 4, 512, 256);

    // Simulate autoregressive generation (10 steps)
    int num_steps = 10;
    std::vector<int> generated_tokens = {1, 2, 3};  // Initial tokens

    Profiler profiler;

    // Without cache
    std::cout << "Testing WITHOUT KV cache..." << std::endl;
    for (int step = 0; step < num_steps; ++step) {
        profiler.start("no_cache");
        
        // Process all tokens from beginning each time (inefficient)
        Matrix output = decoder.forward(generated_tokens);
        
        profiler.stop("no_cache");
        
        // Add a new token
        generated_tokens.push_back(100 + step);
    }

    // With cache
    std::cout << "Testing WITH KV cache..." << std::endl;
    DecoderKVCache kv_cache(2);  // 2 layers
    generated_tokens = {1, 2, 3};  // Reset
    
    for (int step = 0; step < num_steps; ++step) {
        profiler.start("with_cache");
        
        // Process only new token (efficient)
        std::vector<int> new_token = {100 + step};
        Matrix output = decoder.forward_with_cache(new_token, kv_cache, nullptr, true);
        
        profiler.stop("with_cache");
        
        generated_tokens.push_back(100 + step);
    }

    // Compare results
    ProfileStats no_cache_stats = profiler.get_stats("no_cache");
    ProfileStats with_cache_stats = profiler.get_stats("with_cache");

    Profiler::compare(no_cache_stats, with_cache_stats);

    std::cout << "Expected: 2-3x speedup with KV cache" << std::endl;
}

// Benchmark 2: Batch Processing
void benchmark_batch_processing() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark 2: Batch Processing" << std::endl;
    std::cout << "========================================\n" << std::endl;

    int vocab_size = 1000;
    LLMDecoder decoder(vocab_size, 128, 2, 4, 512, 256);

    // Create multiple sequences to process
    std::vector<std::vector<int>> sequences;
    for (int i = 0; i < 8; ++i) {
        std::vector<int> seq;
        int length = 10 + (i * 2);  // Variable lengths: 10, 12, 14, ...
        for (int j = 0; j < length; ++j) {
            seq.push_back((i * 100) + j);
        }
        sequences.push_back(seq);
    }

    Profiler profiler;

    // Sequential processing (one at a time)
    std::cout << "Testing SEQUENTIAL processing..." << std::endl;
    profiler.start("sequential");
    for (const auto& seq : sequences) {
        Matrix output = decoder.forward(seq);
    }
    profiler.stop("sequential");

    // Batch processing (simulated - process all at once)
    // Note: Current implementation processes one at a time, but this demonstrates
    // the concept. Full batching would require batched matrix operations.
    std::cout << "Testing BATCHED processing..." << std::endl;
    
    // Create batches
    auto batches = create_dynamic_batches(sequences, 4, 5, 0);
    BatchStats batch_stats = compute_batch_stats(batches);
    
    std::cout << "\nBatch configuration:" << std::endl;
    batch_stats.print();

    profiler.start("batched");
    for (const auto& batch : batches) {
        // In a full implementation, we'd process the entire batch in one forward pass
        // For now, we process each sequence (but with optimized padding)
        for (const auto& seq : batch.batch_token_ids) {
            Matrix output = decoder.forward(seq);
        }
    }
    profiler.stop("batched");

    // Compare results
    ProfileStats seq_stats = profiler.get_stats("sequential");
    ProfileStats batch_stats_profile = profiler.get_stats("batched");

    Profiler::compare(seq_stats, batch_stats_profile);

    std::cout << "Note: Full batch speedup requires batched matrix operations." << std::endl;
    std::cout << "With true batching: Expected ~" << (sequences.size() / batches.size()) 
              << "x throughput improvement" << std::endl;
}

// Benchmark 3: Combined Optimizations
void benchmark_combined() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark 3: Combined Optimizations" << std::endl;
    std::cout << "========================================\n" << std::endl;

    int vocab_size = 1000;
    LLMDecoder decoder(vocab_size, 128, 2, 4, 512, 256);

    // Generate text with multiple sequences
    std::vector<std::vector<int>> prompts;
    for (int i = 0; i < 4; ++i) {
        prompts.push_back({1, 2, 3 + i});
    }

    int generation_steps = 8;
    Profiler profiler;

    // Baseline: No optimizations
    std::cout << "Testing BASELINE (no optimizations)..." << std::endl;
    profiler.start("baseline");
    for (auto prompt : prompts) {
        for (int step = 0; step < generation_steps; ++step) {
            Matrix output = decoder.forward(prompt);
            prompt.push_back(100 + step);
        }
    }
    profiler.stop("baseline");

    // Optimized: KV cache only
    std::cout << "Testing KV CACHE optimization..." << std::endl;
    profiler.start("kv_cache");
    for (auto prompt : prompts) {
        DecoderKVCache cache(2);
        for (int step = 0; step < generation_steps; ++step) {
            std::vector<int> new_token = {100 + step};
            Matrix output = decoder.forward_with_cache(new_token, cache, nullptr, true);
            prompt.push_back(100 + step);
        }
    }
    profiler.stop("kv_cache");

    // Print comparison
    ProfileStats baseline_stats = profiler.get_stats("baseline");
    ProfileStats kv_stats = profiler.get_stats("kv_cache");

    Profiler::compare(baseline_stats, kv_stats);

    std::cout << "Combined optimizations (KV cache + batching) could achieve:" << std::endl;
    std::cout << "  - 2-3x from KV cache" << std::endl;
    std::cout << "  - 2-4x from batching (depends on batch size)" << std::endl;
    std::cout << "  - Total: 4-12x speedup possible" << std::endl;
}

// Benchmark 4: Latency Analysis
void benchmark_latency() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark 4: Latency Analysis" << std::endl;
    std::cout << "========================================\n" << std::endl;

    int vocab_size = 1000;
    LLMDecoder decoder(vocab_size, 128, 2, 4, 512, 256);

    Profiler profiler;

    // Test different sequence lengths
    std::vector<int> lengths = {10, 20, 50, 100};

    for (int length : lengths) {
        std::vector<int> tokens;
        for (int i = 0; i < length; ++i) {
            tokens.push_back(i);
        }

        std::string name = "length_" + std::to_string(length);
        
        // Measure latency (10 runs)
        for (int run = 0; run < 10; ++run) {
            profiler.start(name);
            Matrix output = decoder.forward(tokens);
            profiler.stop(name);
        }
    }

    std::cout << "\nLatency by sequence length:" << std::endl;
    for (int length : lengths) {
        std::string name = "length_" + std::to_string(length);
        ProfileStats stats = profiler.get_stats(name);
        std::cout << "  Length " << length << ": " 
                  << std::fixed << std::setprecision(2) << stats.mean_time 
                  << " ms (mean)" << std::endl;
    }
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  Inference Optimization Benchmark Suite" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "\nThis benchmark demonstrates the performance" << std::endl;
    std::cout << "improvements from KV caching and batch processing." << std::endl;

    try {
        // Run benchmarks
        benchmark_kv_cache();
        benchmark_batch_processing();
        benchmark_combined();
        benchmark_latency();

        std::cout << "\n==================================================" << std::endl;
        std::cout << "  Benchmark Summary" << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << "\nKey Optimizations Implemented:" << std::endl;
        std::cout << "  ✓ KV Cache for autoregressive generation" << std::endl;
        std::cout << "  ✓ Batch processing utilities" << std::endl;
        std::cout << "  ✓ Performance profiling tools" << std::endl;
        std::cout << "\nExpected Production Improvements:" << std::endl;
        std::cout << "  - 2-3x speedup from KV cache" << std::endl;
        std::cout << "  - 2-4x throughput from batching" << std::endl;
        std::cout << "  - 4-12x combined improvement" << std::endl;
        std::cout << "\n==================================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during benchmark: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
