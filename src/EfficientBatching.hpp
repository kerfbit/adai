/**
 * @file EfficientBatching.hpp
 * @brief Efficient batching utilities for data pipeline optimization
 * 
 * This file provides utilities for efficient batch creation with dynamic batching
 * by sequence length, intelligent padding strategies, and data augmentation.
 * 
 * Key Features:
 * - Dynamic batching by sequence length (group similar lengths together)
 * - Multiple padding strategies (left, right, center)
 * - Batch sorting and bucketing
 * - Data augmentation support (token dropout, masking)
 * - Batch statistics and optimization
 * 
 * @version 1.0
 * @date January 2026
 */

#ifndef EFFICIENT_BATCHING_HPP
#define EFFICIENT_BATCHING_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cmath>

#ifdef ADAI_ENABLE_OPENMP
#include <omp.h>
#endif

/**
 * @brief Padding strategy for batch creation
 */
enum class PaddingStrategy {
    LEFT,    ///< Pad on the left (prefix padding)
    RIGHT,   ///< Pad on the right (suffix padding)
    CENTER   ///< Pad on both sides (center padding)
};

/**
 * @brief Data augmentation configuration
 */
struct AugmentationConfig {
    bool enable_token_dropout = false;      ///< Enable random token dropout
    float token_dropout_prob = 0.1f;        ///< Probability of dropping a token
    bool enable_token_masking = false;      ///< Enable random token masking
    float token_mask_prob = 0.15f;          ///< Probability of masking a token
    int mask_token_id = 3;                  ///< Token ID to use for masking
    bool enable_sequence_shuffle = false;   ///< Enable shuffling within sequences
    float shuffle_prob = 0.05f;             ///< Probability of shuffling adjacent tokens
    unsigned int seed = 42;                 ///< Random seed for reproducibility
};

/**
 * @brief Bucket configuration for dynamic batching
 */
struct BucketConfig {
    std::vector<int> bucket_boundaries;     ///< Length boundaries for buckets (e.g., [32, 64, 128])
    int max_tokens_per_batch = 4096;        ///< Maximum total tokens per batch
    bool shuffle_buckets = true;            ///< Shuffle sequences within buckets
};

/**
 * @brief Batch statistics for monitoring
 */
struct BatchStatistics {
    size_t num_batches = 0;
    size_t total_sequences = 0;
    size_t total_tokens = 0;
    size_t total_padding_tokens = 0;
    double avg_batch_size = 0.0;
    double avg_sequence_length = 0.0;
    double padding_ratio = 0.0;
    double efficiency_score = 0.0;  ///< 1.0 - padding_ratio
};

/**
 * @brief A single batch of sequences with padding
 */
struct SequenceBatch {
    std::vector<std::vector<int>> sequences;  ///< Padded sequences
    std::vector<std::vector<int>> masks;      ///< Attention masks (1=valid, 0=padding)
    std::vector<int> lengths;                 ///< Original sequence lengths
    int max_length = 0;                       ///< Maximum length in batch
    int pad_token_id = 0;                     ///< Padding token ID
    
    /**
     * @brief Get total number of tokens (including padding)
     */
    size_t total_tokens() const {
        return sequences.size() * max_length;
    }
    
    /**
     * @brief Get number of padding tokens
     */
    size_t padding_tokens() const {
        size_t real_tokens = std::accumulate(lengths.begin(), lengths.end(), 0);
        return total_tokens() - real_tokens;
    }
    
    /**
     * @brief Get padding ratio (0.0 to 1.0)
     */
    double padding_ratio() const {
        if (total_tokens() == 0) return 0.0;
        return static_cast<double>(padding_tokens()) / total_tokens();
    }
};

/**
 * @brief Efficient batching utilities
 * 
 * Provides advanced batching capabilities including dynamic batching by sequence length,
 * intelligent padding strategies, and data augmentation.
 */
class EfficientBatching {
public:
    /**
     * @brief Create batches with dynamic batching by sequence length
     * 
     * Groups sequences of similar lengths together to minimize padding.
     * 
     * @param sequences Input sequences (variable length)
     * @param batch_size Maximum number of sequences per batch
     * @param pad_token_id Token ID to use for padding
     * @param strategy Padding strategy (left, right, center)
     * @param sort_by_length Sort sequences by length before batching
     * @return Vector of batches
     */
    static std::vector<SequenceBatch> create_dynamic_batches(
        const std::vector<std::vector<int>>& sequences,
        size_t batch_size,
        int pad_token_id = 0,
        PaddingStrategy strategy = PaddingStrategy::RIGHT,
        bool sort_by_length = true
    ) {
        if (sequences.empty()) {
            throw std::invalid_argument("Cannot create batches from empty sequences");
        }
        
        // Create indices for sorting
        std::vector<size_t> indices(sequences.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Sort by length if requested
        if (sort_by_length) {
            std::sort(indices.begin(), indices.end(), [&sequences](size_t a, size_t b) {
                return sequences[a].size() < sequences[b].size();
            });
        }
        
        // Create batches
        std::vector<SequenceBatch> batches;
        for (size_t i = 0; i < indices.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, indices.size());
            std::vector<size_t> batch_indices(indices.begin() + i, indices.begin() + end);
            
            // Create batch from these indices
            SequenceBatch batch = create_single_batch(sequences, batch_indices, pad_token_id, strategy);
            batches.push_back(batch);
        }
        
        return batches;
    }
    
    /**
     * @brief Create batches using bucketing strategy
     * 
     * Assigns sequences to buckets based on length and creates batches within buckets.
     * This is more efficient than simple sorting for datasets with wide length variation.
     * 
     * @param sequences Input sequences
     * @param config Bucket configuration
     * @param pad_token_id Token ID for padding
     * @param strategy Padding strategy
     * @return Vector of batches
     */
    static std::vector<SequenceBatch> create_bucketed_batches(
        const std::vector<std::vector<int>>& sequences,
        const BucketConfig& config,
        int pad_token_id = 0,
        PaddingStrategy strategy = PaddingStrategy::RIGHT
    ) {
        if (sequences.empty()) {
            throw std::invalid_argument("Cannot create batches from empty sequences");
        }
        
        // Assign sequences to buckets
        std::vector<std::vector<size_t>> buckets(config.bucket_boundaries.size() + 1);
        
        for (size_t i = 0; i < sequences.size(); ++i) {
            int length = sequences[i].size();
            int bucket_idx = 0;
            
            for (size_t b = 0; b < config.bucket_boundaries.size(); ++b) {
                if (length <= config.bucket_boundaries[b]) {
                    bucket_idx = b;
                    break;
                }
                bucket_idx = b + 1;
            }
            
            buckets[bucket_idx].push_back(i);
        }
        
        // Create batches from each bucket
        std::vector<SequenceBatch> batches;
        
        for (auto& bucket : buckets) {
            if (bucket.empty()) continue;
            
            // Shuffle within bucket if requested
            if (config.shuffle_buckets) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::shuffle(bucket.begin(), bucket.end(), gen);
            }
            
            // Calculate batch size based on max tokens
            size_t i = 0;
            while (i < bucket.size()) {
                std::vector<size_t> batch_indices;
                size_t current_tokens = 0;
                
                // Add sequences until we hit max tokens
                while (i < bucket.size()) {
                    size_t seq_idx = bucket[i];
                    size_t seq_len = sequences[seq_idx].size();
                    size_t batch_max_len = batch_indices.empty() ? seq_len :
                        std::max(seq_len, sequences[batch_indices[0]].size());
                    
                    size_t new_tokens = batch_max_len * (batch_indices.size() + 1);
                    
                    if (!batch_indices.empty() && new_tokens > config.max_tokens_per_batch) {
                        break;
                    }
                    
                    batch_indices.push_back(seq_idx);
                    current_tokens = new_tokens;
                    ++i;
                }
                
                if (!batch_indices.empty()) {
                    SequenceBatch batch = create_single_batch(sequences, batch_indices, 
                                                              pad_token_id, strategy);
                    batches.push_back(batch);
                }
            }
        }
        
        return batches;
    }
    
    /**
     * @brief Apply data augmentation to sequences
     * 
     * @param sequences Input sequences (modified in place)
     * @param config Augmentation configuration
     */
    static void apply_augmentation(
        std::vector<std::vector<int>>& sequences,
        const AugmentationConfig& config
    ) {
#ifdef ADAI_ENABLE_OPENMP
        // Parallel version with OpenMP
        // Each thread gets its own RNG seeded differently for thread safety
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::mt19937 gen(config.seed + thread_id);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            #pragma omp for schedule(dynamic, 16)
            for (size_t seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
                auto& seq = sequences[seq_idx];
                
                // Token dropout
                if (config.enable_token_dropout && config.token_dropout_prob > 0.0f) {
                    std::vector<int> new_seq;
                    new_seq.reserve(seq.size());
                    for (int token : seq) {
                        if (dist(gen) >= config.token_dropout_prob) {
                            new_seq.push_back(token);
                        }
                    }
                    if (!new_seq.empty()) {
                        seq = std::move(new_seq);
                    }
                }
                
                // Token masking
                if (config.enable_token_masking && config.token_mask_prob > 0.0f) {
                    for (int& token : seq) {
                        if (dist(gen) < config.token_mask_prob) {
                            token = config.mask_token_id;
                        }
                    }
                }
                
                // Sequence shuffle (adjacent tokens)
                if (config.enable_sequence_shuffle && config.shuffle_prob > 0.0f && seq.size() > 1) {
                    for (size_t i = 0; i < seq.size() - 1; ++i) {
                        if (dist(gen) < config.shuffle_prob) {
                            std::swap(seq[i], seq[i + 1]);
                        }
                    }
                }
            }
        }
#else
        // Sequential fallback
        std::mt19937 gen(config.seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& seq : sequences) {
            // Token dropout
            if (config.enable_token_dropout && config.token_dropout_prob > 0.0f) {
                std::vector<int> new_seq;
                for (int token : seq) {
                    if (dist(gen) >= config.token_dropout_prob) {
                        new_seq.push_back(token);
                    }
                }
                if (!new_seq.empty()) {
                    seq = new_seq;
                }
            }
            
            // Token masking
            if (config.enable_token_masking && config.token_mask_prob > 0.0f) {
                for (int& token : seq) {
                    if (dist(gen) < config.token_mask_prob) {
                        token = config.mask_token_id;
                    }
                }
            }
            
            // Sequence shuffle (adjacent tokens)
            if (config.enable_sequence_shuffle && config.shuffle_prob > 0.0f && seq.size() > 1) {
                for (size_t i = 0; i < seq.size() - 1; ++i) {
                    if (dist(gen) < config.shuffle_prob) {
                        std::swap(seq[i], seq[i + 1]);
                    }
                }
            }
        }
#endif
    }
    
    /**
     * @brief Calculate batch statistics
     * 
     * @param batches Vector of batches to analyze
     * @return Batch statistics
     */
    static BatchStatistics calculate_statistics(const std::vector<SequenceBatch>& batches) {
        BatchStatistics stats;
        stats.num_batches = batches.size();
        
        if (batches.empty()) {
            return stats;
        }
        
        size_t total_sequences = 0;
        size_t total_tokens = 0;
        size_t total_padding = 0;
        
        for (const auto& batch : batches) {
            total_sequences += batch.sequences.size();
            total_tokens += batch.total_tokens();
            total_padding += batch.padding_tokens();
        }
        
        stats.total_sequences = total_sequences;
        stats.total_tokens = total_tokens;
        stats.total_padding_tokens = total_padding;
        stats.avg_batch_size = static_cast<double>(total_sequences) / batches.size();
        stats.avg_sequence_length = total_sequences > 0 ? 
            static_cast<double>(total_tokens - total_padding) / total_sequences : 0.0;
        stats.padding_ratio = total_tokens > 0 ? 
            static_cast<double>(total_padding) / total_tokens : 0.0;
        stats.efficiency_score = 1.0 - stats.padding_ratio;
        
        return stats;
    }
    
    /**
     * @brief Pad a single sequence
     * 
     * @param sequence Input sequence
     * @param target_length Target length after padding
     * @param pad_token_id Token ID for padding
     * @param strategy Padding strategy
     * @return Padded sequence
     */
    static std::vector<int> pad_sequence(
        const std::vector<int>& sequence,
        int target_length,
        int pad_token_id,
        PaddingStrategy strategy
    ) {
        if (sequence.size() >= static_cast<size_t>(target_length)) {
            return sequence;
        }
        
        int padding_needed = target_length - sequence.size();
        std::vector<int> padded;
        padded.reserve(target_length);
        
        switch (strategy) {
            case PaddingStrategy::LEFT:
                // Pad on the left
                padded.insert(padded.end(), padding_needed, pad_token_id);
                padded.insert(padded.end(), sequence.begin(), sequence.end());
                break;
                
            case PaddingStrategy::RIGHT:
                // Pad on the right
                padded.insert(padded.end(), sequence.begin(), sequence.end());
                padded.insert(padded.end(), padding_needed, pad_token_id);
                break;
                
            case PaddingStrategy::CENTER:
                // Pad on both sides
                int left_pad = padding_needed / 2;
                int right_pad = padding_needed - left_pad;
                padded.insert(padded.end(), left_pad, pad_token_id);
                padded.insert(padded.end(), sequence.begin(), sequence.end());
                padded.insert(padded.end(), right_pad, pad_token_id);
                break;
        }
        
        return padded;
    }
    
    /**
     * @brief Create attention mask for a padded sequence
     * 
     * @param original_length Original sequence length (before padding)
     * @param padded_length Length after padding
     * @param strategy Padding strategy used
     * @return Attention mask (1=valid, 0=padding)
     */
    static std::vector<int> create_attention_mask(
        int original_length,
        int padded_length,
        PaddingStrategy strategy
    ) {
        std::vector<int> mask(padded_length, 0);
        
        switch (strategy) {
            case PaddingStrategy::LEFT: {
                int padding = padded_length - original_length;
                for (int i = padding; i < padded_length; ++i) {
                    mask[i] = 1;
                }
                break;
            }
            
            case PaddingStrategy::RIGHT:
                for (int i = 0; i < original_length; ++i) {
                    mask[i] = 1;
                }
                break;
                
            case PaddingStrategy::CENTER: {
                int total_padding = padded_length - original_length;
                int left_pad = total_padding / 2;
                for (int i = left_pad; i < left_pad + original_length; ++i) {
                    mask[i] = 1;
                }
                break;
            }
        }
        
        return mask;
    }

private:
    /**
     * @brief Create a single batch from indices
     */
    static SequenceBatch create_single_batch(
        const std::vector<std::vector<int>>& sequences,
        const std::vector<size_t>& indices,
        int pad_token_id,
        PaddingStrategy strategy
    ) {
        SequenceBatch batch;
        batch.pad_token_id = pad_token_id;
        
        // Find max length in batch
        batch.max_length = 0;
        for (size_t idx : indices) {
            batch.max_length = std::max(batch.max_length, 
                                       static_cast<int>(sequences[idx].size()));
        }
        
        // Pad sequences and create masks
        for (size_t idx : indices) {
            const auto& seq = sequences[idx];
            int original_length = seq.size();
            
            // Pad sequence
            std::vector<int> padded = pad_sequence(seq, batch.max_length, 
                                                  pad_token_id, strategy);
            batch.sequences.push_back(padded);
            
            // Create attention mask
            std::vector<int> mask = create_attention_mask(original_length, 
                                                         batch.max_length, strategy);
            batch.masks.push_back(mask);
            
            // Store original length
            batch.lengths.push_back(original_length);
        }
        
        return batch;
    }
};

#endif // EFFICIENT_BATCHING_HPP
