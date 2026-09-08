#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07


#include <algorithm>
#include <vector>
#include "Matrix.hpp"
#include "SpecialTokens.hpp"

/**
 * Batch Processing Utilities for Transformer Models
 *
 * Provides utilities for batching multiple sequences together for efficient
 * parallel processing. Includes padding, dynamic batching by length, and
 * batch encoding/decoding.
 *
 * Key Benefits:
 * - Process multiple sequences in one forward pass
 * - Better GPU/CPU utilization
 * - Higher throughput for inference
 *
 * Usage:
 * 1. Collect multiple input sequences
 * 2. Batch and pad to same length
 * 3. Process batch through model
 * 4. Unbatch outputs
 */

/**
 * Batch of token sequences with padding information
 */
struct TokenBatch {
    /**
     * Batched token IDs
     * batch_token_ids[i] = token sequence for batch item i
     */
    std::vector<std::vector<int>> batch_token_ids;

    /**
     * Sequence lengths before padding
     * lengths[i] = actual length of sequence i (excluding padding)
     */
    std::vector<int> lengths;

    /**
     * Maximum sequence length in batch (after padding)
     */
    int max_length = 0;

    /**
     * Padding token ID
     */
    int pad_token_id = 0;

    /**
     * Number of sequences in batch
     */
    int batch_size() const {
        return static_cast<int>(batch_token_ids.size());
    }

    /**
     * Check if batch is empty
     */
    bool is_empty() const {
        return batch_token_ids.empty();
    }
};

/**
 * Create a padded batch from multiple sequences
 *
 * Pads all sequences to the length of the longest sequence in the batch.
 *
 * @param sequences Vector of token ID sequences (variable length)
 * @param pad_token_id Token ID to use for padding (default: PAD token)
 * @return TokenBatch with padded sequences
 */
inline TokenBatch create_batch(const std::vector<std::vector<int>>& sequences,
                               int pad_token_id = adai::SpecialTokenIDs::PAD) {
    TokenBatch batch;
    batch.pad_token_id = pad_token_id;

    if (sequences.empty()) {
        batch.max_length = 0;
        return batch;
    }

    // Find maximum length
    batch.max_length = 0;
    for (const auto& seq : sequences) {
        if (seq.size() > static_cast<size_t>(batch.max_length)) {
            batch.max_length = static_cast<int>(seq.size());
        }
    }

    // Pad all sequences to max_length
    batch.batch_token_ids.reserve(sequences.size());
    batch.lengths.reserve(sequences.size());

    for (const auto& seq : sequences) {
        std::vector<int> padded_seq = seq;
        int original_length = static_cast<int>(seq.size());

        // Pad to max_length
        while (padded_seq.size() < static_cast<size_t>(batch.max_length)) {
            padded_seq.push_back(pad_token_id);
        }

        batch.batch_token_ids.push_back(padded_seq);
        batch.lengths.push_back(original_length);
    }

    return batch;
}

/**
 * Create dynamic batches by grouping sequences of similar length
 *
 * Groups sequences into batches where length variation is minimized,
 * reducing wasted computation from padding.
 *
 * @param sequences Vector of token ID sequences
 * @param max_batch_size Maximum number of sequences per batch
 * @param length_tolerance Maximum length difference within a batch
 * @param pad_token_id Token ID to use for padding (default: PAD token)
 * @return Vector of batches
 */
inline std::vector<TokenBatch> create_dynamic_batches(
    const std::vector<std::vector<int>>& sequences, int max_batch_size = 32,
    int length_tolerance = 10, int pad_token_id = adai::SpecialTokenIDs::PAD) {
    std::vector<TokenBatch> batches;

    if (sequences.empty()) {
        return batches;
    }

    // Create pairs of (length, index) for sorting
    std::vector<std::pair<int, int>> length_index_pairs;
    for (size_t i = 0; i < sequences.size(); ++i) {
        length_index_pairs.emplace_back(sequences[i].size(), i);
    }

    // Sort by length
    std::sort(length_index_pairs.begin(), length_index_pairs.end());

    // Group into batches
    std::vector<std::vector<int>> current_batch;
    int current_batch_min_length = length_index_pairs[0].first;

    for (const auto& pair : length_index_pairs) {
        int length = pair.first;
        int index = pair.second;

        bool start_new_batch =
            (current_batch.size() >= static_cast<size_t>(max_batch_size)) ||
            (!current_batch.empty() && (length - current_batch_min_length) > length_tolerance);

        if (start_new_batch && !current_batch.empty()) {
            // Create batch from current_batch
            batches.push_back(create_batch(current_batch, pad_token_id));
            current_batch.clear();
            current_batch_min_length = length;
        }

        // Add sequence to current batch
        current_batch.push_back(sequences[index]);

        if (current_batch.size() == 1) {
            current_batch_min_length = length;
        }
    }

    // Add final batch
    if (!current_batch.empty()) {
        batches.push_back(create_batch(current_batch, pad_token_id));
    }

    return batches;
}

/**
 * Create padding mask for a batch
 *
 * Mask has 1 for real tokens, 0 for padding tokens.
 * Used in attention to prevent attending to padding.
 *
 * @param batch TokenBatch with padding information
 * @return Matrix [batch_size, max_length] with padding mask
 */
inline Matrix create_padding_mask(const TokenBatch& batch) {
    int batch_size = batch.batch_size();
    int max_length = batch.max_length;

    Matrix mask(batch_size, max_length);

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < max_length; ++j) {
            // 1 for real tokens, 0 for padding
            mask(i, j) = (j < batch.lengths[i]) ? 1.0f : 0.0f;
        }
    }

    return mask;
}

/**
 * Unbatch a matrix of outputs back to individual sequences
 *
 * Removes padding from batch outputs, returning variable-length sequences.
 *
 * @param batch_output Matrix [batch_size, max_length, d_model] or [batch_size, max_length]
 * @param batch Original TokenBatch with length information
 * @return Vector of matrices, one per sequence (without padding)
 */
inline std::vector<Matrix> unbatch_outputs(const std::vector<Matrix>& batch_outputs,
                                           const TokenBatch& batch) {
    std::vector<Matrix> individual_outputs;
    individual_outputs.reserve(batch.batch_size());

    for (int i = 0; i < batch.batch_size(); ++i) {
        int actual_length = batch.lengths[i];
        const Matrix& full_output = batch_outputs[i];

        // Extract first actual_length rows (remove padding)
        Matrix trimmed_output(actual_length, full_output.cols);
        for (int row = 0; row < actual_length; ++row) {
            for (int col = 0; col < full_output.cols; ++col) {
                trimmed_output(row, col) = full_output(row, col);
            }
        }

        individual_outputs.push_back(trimmed_output);
    }

    return individual_outputs;
}

/**
 * Statistics about batch efficiency
 */
struct BatchStats {
    int total_tokens = 0;         // Total tokens including padding
    int actual_tokens = 0;        // Actual tokens (excluding padding)
    float padding_ratio = 0.0f;   // Ratio of padding tokens to total
    int num_batches = 0;          // Number of batches created
    float avg_batch_size = 0.0f;  // Average batch size

    void print() const {
        std::cout << "Batch Statistics:" << '\n';
        std::cout << "  Total tokens (with padding): " << total_tokens << '\n';
        std::cout << "  Actual tokens: " << actual_tokens << '\n';
        std::cout << "  Padding ratio: " << (padding_ratio * 100.0f) << "%" << '\n';
        std::cout << "  Number of batches: " << num_batches << '\n';
        std::cout << "  Average batch size: " << avg_batch_size << '\n';
        std::cout << "  Efficiency: " << ((1.0f - padding_ratio) * 100.0f) << "%" << '\n';
    }
};

/**
 * Compute batch statistics
 *
 * @param batches Vector of TokenBatch
 * @return BatchStats with efficiency metrics
 */
inline BatchStats compute_batch_stats(const std::vector<TokenBatch>& batches) {
    BatchStats stats;
    stats.num_batches = static_cast<int>(batches.size());
    stats.total_tokens = 0;
    stats.actual_tokens = 0;
    int total_sequences = 0;

    for (const auto& batch : batches) {
        int batch_total = batch.batch_size() * batch.max_length;
        int batch_actual = 0;
        for (int length : batch.lengths) {
            batch_actual += length;
        }

        stats.total_tokens += batch_total;
        stats.actual_tokens += batch_actual;
        total_sequences += batch.batch_size();
    }

    stats.padding_ratio = (stats.total_tokens > 0)
                              ? static_cast<float>(stats.total_tokens - stats.actual_tokens) /
                                    static_cast<float>(stats.total_tokens)
                              : 0.0f;

    stats.avg_batch_size = (stats.num_batches > 0) ? static_cast<float>(total_sequences) /
                                                         static_cast<float>(stats.num_batches)
                                                   : 0.0f;

    return stats;
}
