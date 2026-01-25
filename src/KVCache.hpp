#pragma once

#include <vector>
#include "Matrix.hpp"

/**
 * Key-Value Cache for Transformer Attention
 *
 * Stores computed key and value tensors during autoregressive generation
 * to avoid redundant computation. In autoregressive decoding, previously
 * generated tokens' key-value pairs remain constant, so we can cache them.
 *
 * Expected Performance Impact: 2-3x speedup in inference
 *
 * Usage Pattern:
 * 1. First forward pass: cache_keys and cache_values are empty
 * 2. Compute keys/values for all positions, store in cache
 * 3. Subsequent passes: only compute K/V for new token
 * 4. Concatenate cached K/V with new K/V for attention
 *
 * Mathematical Insight:
 * For position t in sequence:
 *   Without cache: Compute K, V for all positions [0, t]
 *   With cache: Compute K, V only for position t, reuse [0, t-1]
 *
 * Memory vs Speed Tradeoff:
 * - Memory: O(num_layers * num_heads * seq_len * d_k) per sequence
 * - Speed: ~2-3x faster for long sequences
 */
struct KVCache {
    /**
     * Cached keys: [seq_len, d_model]
     * Accumulated across generation steps
     */
    Matrix keys;

    /**
     * Cached values: [seq_len, d_model]
     * Accumulated across generation steps
     */
    Matrix values;

    /**
     * Current sequence length in cache
     */
    int current_length;

    /**
     * Constructor
     */
    KVCache() : current_length(0) {}

    /**
     * Check if cache is empty
     */
    bool is_empty() const {
        return current_length == 0;
    }

    /**
     * Get number of cached positions
     */
    int size() const {
        return current_length;
    }

    /**
     * Clear the cache
     */
    void clear() {
        keys = Matrix(0, 0);
        values = Matrix(0, 0);
        current_length = 0;
    }

    /**
     * Append new key-value pair to cache
     *
     * @param new_keys Keys for new position(s) [num_new_positions, d_model]
     * @param new_values Values for new position(s) [num_new_positions, d_model]
     */
    void append(const Matrix& new_keys, const Matrix& new_values) {
        if (is_empty()) {
            // Initialize cache with first keys/values
            keys = new_keys;
            values = new_values;
            current_length = new_keys.rows;
        } else {
            // Concatenate new keys/values to existing cache
            // This creates a new matrix with shape [current_length + num_new, d_model]
            Matrix concatenated_keys(current_length + new_keys.rows, keys.cols);
            Matrix concatenated_values(current_length + new_values.rows, values.cols);

            // Copy existing cached data
            for (int i = 0; i < current_length; ++i) {
                for (int j = 0; j < keys.cols; ++j) {
                    concatenated_keys.data[i][j] = keys.data[i][j];
                    concatenated_values.data[i][j] = values.data[i][j];
                }
            }

            // Append new data
            for (int i = 0; i < new_keys.rows; ++i) {
                for (int j = 0; j < new_keys.cols; ++j) {
                    concatenated_keys.data[current_length + i][j] = new_keys.data[i][j];
                    concatenated_values.data[current_length + i][j] = new_values.data[i][j];
                }
            }

            keys = concatenated_keys;
            values = concatenated_values;
            current_length += new_keys.rows;
        }
    }

    /**
     * Get cached keys
     *
     * @return Cached keys matrix [current_length, d_model]
     */
    const Matrix& get_keys() const {
        return keys;
    }

    /**
     * Get cached values
     *
     * @return Cached values matrix [current_length, d_model]
     */
    const Matrix& get_values() const {
        return values;
    }
};

/**
 * Multi-layer KV Cache for Decoder
 *
 * Maintains separate caches for each decoder layer.
 * Each layer has its own key-value cache for self-attention.
 */
struct DecoderKVCache {
    /**
     * Per-layer caches
     * self_attention_caches[layer_idx] = KVCache for that layer's self-attention
     */
    std::vector<KVCache> self_attention_caches;

    /**
     * Cross-attention caches (for encoder-decoder models)
     * cross_attention_caches[layer_idx] = KVCache for that layer's cross-attention
     * Note: Cross-attention K/V are constant (from encoder), computed once
     */
    std::vector<KVCache> cross_attention_caches;

    /**
     * Constructor
     *
     * @param num_layers Number of decoder layers
     */
    explicit DecoderKVCache(int num_layers) {
        self_attention_caches.resize(num_layers);
        cross_attention_caches.resize(num_layers);
    }

    /**
     * Clear all caches
     */
    void clear() {
        for (auto& cache : self_attention_caches) {
            cache.clear();
        }
        for (auto& cache : cross_attention_caches) {
            cache.clear();
        }
    }

    /**
     * Clear self-attention caches only (keep cross-attention)
     */
    void clear_self_attention() {
        for (auto& cache : self_attention_caches) {
            cache.clear();
        }
    }

    /**
     * Get cache for specific layer's self-attention
     */
    KVCache& get_self_attention_cache(int layer_idx) {
        return self_attention_caches[layer_idx];
    }

    /**
     * Get cache for specific layer's cross-attention
     */
    KVCache& get_cross_attention_cache(int layer_idx) {
        return cross_attention_caches[layer_idx];
    }

    /**
     * Check if any cache is populated
     */
    bool is_empty() const {
        for (const auto& cache : self_attention_caches) {
            if (!cache.is_empty()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Get current sequence length from first layer cache
     */
    int current_length() const {
        if (!self_attention_caches.empty()) {
            return self_attention_caches[0].current_length;
        }
        return 0;
    }
};

