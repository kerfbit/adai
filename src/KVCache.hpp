#pragma once

// @adai-status: beta        (TD-050 investigated — root-caused clean, no correctness bug found; see TECHNICAL_DEBT.md)
// @adai-version: 0.7.0
// @adai-reviewed: 2026-10-26


#include <stdexcept>
#include <string>
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
    int current_length{0};

    /**
     * Constructor
     */
    KVCache() = default;

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
     * @throws std::invalid_argument when the shapes don't match each other or the cache (TD-421:
     *         wider rows used to overflow the heap on write, fewer value rows on read)
     */
    void append(const Matrix& new_keys, const Matrix& new_values) {
        if (new_keys.rows != new_values.rows || new_keys.cols != new_values.cols) {
            throw std::invalid_argument(
                "KVCache::append: keys " + std::to_string(new_keys.rows) + "x" +
                std::to_string(new_keys.cols) + " and values " + std::to_string(new_values.rows) +
                "x" + std::to_string(new_values.cols) + " differ in shape");
        }
        if (!is_empty() && new_keys.cols != keys.cols) {
            throw std::invalid_argument("KVCache::append: new rows have width " +
                                        std::to_string(new_keys.cols) + ", the cache " +
                                        std::to_string(keys.cols));
        }
        if (is_empty()) {
            keys = new_keys;
            values = new_values;
            current_length = new_keys.rows;
            return;
        }
        // TD-422: grow in place. Matrix keeps one vector per row, so appending rows moves no
        // existing data (amortized O(new rows * d)). This used to allocate a
        // [current_length + new, d] pair and copy the whole cache on every decode step: O(n^2 * d)
        // per generation (255 ms of copying for one 512-step cache at d_model 512).
        for (int i = 0; i < new_keys.rows; ++i) {
            keys.data.push_back(new_keys.data[i]);
            values.data.push_back(new_values.data[i]);
        }
        keys.rows += new_keys.rows;
        values.rows += new_values.rows;
        current_length += new_keys.rows;
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
    // TD-421: layer indices are checked (an out-of-range index used to read past the vector).
    KVCache& get_self_attention_cache(int layer_idx) {
        return self_attention_caches.at(checked_layer(layer_idx));
    }

    /**
     * Get cache for specific layer's cross-attention
     */
    KVCache& get_cross_attention_cache(int layer_idx) {
        return cross_attention_caches.at(checked_layer(layer_idx));
    }

    size_t checked_layer(int layer_idx) const {
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= self_attention_caches.size()) {
            throw std::out_of_range("DecoderKVCache: layer " + std::to_string(layer_idx) +
                                    " of " + std::to_string(self_attention_caches.size()));
        }
        return static_cast<size_t>(layer_idx);
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
     * Get current sequence length from first layer cache (every layer's self-attention cache grows
     * by the same rows each step)
     */
    int current_length() const {
        if (!self_attention_caches.empty()) {
            return self_attention_caches[0].current_length;
        }
        return 0;
    }
};
