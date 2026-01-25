#pragma once

#include <cmath>
#include "Matrix.hpp"

/**
 * Positional Encoding for Transformer architectures
 *
 * Positional Encoding adds position information to input embeddings, allowing
 * the model to understand the sequential order of tokens. This implementation
 * uses sinusoidal functions as described in "Attention is All You Need".
 *
 * Mathematical Formula:
 * For position pos and dimension i:
 *   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *
 * where:
 *   - pos is the position in the sequence
 *   - i is the dimension index
 *   - d_model is the embedding dimension
 *
 * Benefits:
 * - Deterministic (no learned parameters)
 * - Handles variable sequence lengths
 * - Allows model to attend to relative positions
 * - Generalizes to sequences longer than training data
 */
class PositionalEncoding {
   private:
    Matrix pos_encoding;  // Pre-computed positional encodings [max_len, d_model]
    int max_len;          // Maximum sequence length
    int d_model;          // Embedding dimension

   public:
    /**
     * Constructor for PositionalEncoding
     *
     * Pre-computes sinusoidal positional encodings for all positions
     * up to max_len. These encodings are fixed and not learned during training.
     *
     * @param max_len Maximum sequence length to support
     * @param d_model Embedding dimension (must match input embeddings)
     */
    PositionalEncoding(int max_len, int d_model);

    /**
     * Forward pass: add positional encodings to input embeddings
     *
     * @param input Input embeddings [sequence_length, d_model]
     * @return Input with positional encodings added [sequence_length, d_model]
     *
     * For each position i and dimension j:
     *   output[i][j] = input[i][j] + pos_encoding[i][j]
     *
     * Note: If input sequence length exceeds max_len, only the first
     * max_len positions will receive positional encodings.
     */
    Matrix forward(const Matrix& input);

    /**
     * Get the pre-computed positional encoding matrix
     *
     * @return Reference to positional encoding matrix [max_len, d_model]
     */
    const Matrix& get_encoding() const {
        return pos_encoding;
    }

    /**
     * Get maximum sequence length
     *
     * @return Maximum supported sequence length
     */
    int get_max_len() const {
        return max_len;
    }

    /**
     * Get embedding dimension
     *
     * @return Embedding dimension (d_model)
     */
    int get_d_model() const {
        return d_model;
    }

    /**
     * Print positional encoding configuration
     *
     * @param name Optional name for the encoding
     */
    void print_config(const std::string& name = "PositionalEncoding") const;

    /**
     * Get encoding for a specific position
     *
     * @param pos Position index (0-based)
     * @return Vector of positional encoding values for that position
     * @throws std::out_of_range if pos >= max_len
     */
    std::vector<float> get_position_encoding(int pos) const;

    /**
     * Visualize encoding pattern for analysis
     *
     * Prints a subset of the encoding matrix to show the sinusoidal pattern.
     * Useful for debugging and understanding the encoding structure.
     *
     * @param num_positions Number of positions to display (default: 10)
     * @param num_dims Number of dimensions to display (default: 8)
     */
    void visualize(int num_positions = 10, int num_dims = 8) const;
};
