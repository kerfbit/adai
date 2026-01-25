#include "PositionalEncoding.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>

PositionalEncoding::PositionalEncoding(int max_len, int d_model)
    : pos_encoding(max_len, d_model), max_len(max_len), d_model(d_model) {
    // Pre-compute positional encodings using sinusoidal functions
    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            // Compute the wavelength for this dimension
            float angle = pos / std::pow(10000.0f, (2.0f * (i / 2)) / static_cast<float>(d_model));

            if (i % 2 == 0) {
                // Even indices: use sine
                pos_encoding(pos, i) = std::sin(angle);
            } else {
                // Odd indices: use cosine
                pos_encoding(pos, i) = std::cos(angle);
            }
        }
    }
}

Matrix PositionalEncoding::forward(const Matrix& input) {
    // Create a copy of the input
    Matrix result = input;

    // Add positional encodings to each position
    int seq_len = std::min(input.rows, max_len);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            result(i, j) += pos_encoding(i, j);
        }
    }

    // If sequence is longer than max_len, the remaining positions
    // won't have positional encodings added (though this should be rare)
    if (input.rows > max_len) {
        std::cerr << "Warning: Input sequence length (" << input.rows << ") exceeds max_len ("
                  << max_len << "). "
                  << "Positions beyond max_len will not receive positional encodings." << std::endl;
    }

    return result;
}

std::vector<float> PositionalEncoding::get_position_encoding(int pos) const {
    if (pos < 0 || pos >= max_len) {
        throw std::out_of_range("Position " + std::to_string(pos) + " is out of range [0, " +
                                std::to_string(max_len) + ")");
    }

    std::vector<float> encoding(d_model);
    for (int i = 0; i < d_model; ++i) {
        encoding[i] = pos_encoding(pos, i);
    }
    return encoding;
}

void PositionalEncoding::print_config(const std::string& name) const {
    std::cout << "\n" << name << " Configuration:" << std::endl;
    std::cout << "  Maximum Sequence Length: " << max_len << std::endl;
    std::cout << "  Embedding Dimension (d_model): " << d_model << std::endl;
    std::cout << "  Encoding Type: Sinusoidal (fixed, not learned)" << std::endl;
    std::cout << "  Memory Usage: " << (max_len * d_model * sizeof(float)) / 1024.0f << " KB"
              << std::endl;
}

void PositionalEncoding::visualize(int num_positions, int num_dims) const {
    // Limit display to actual dimensions
    num_positions = std::min(num_positions, max_len);
    num_dims = std::min(num_dims, d_model);

    std::cout << "\nPositional Encoding Visualization:" << std::endl;
    std::cout << "Showing first " << num_positions << " positions and " << num_dims << " dimensions"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Print header
    std::cout << std::setw(6) << "Pos";
    for (int j = 0; j < num_dims; ++j) {
        std::cout << std::setw(10) << ("Dim" + std::to_string(j));
    }
    std::cout << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Print encoding values
    for (int i = 0; i < num_positions; ++i) {
        std::cout << std::setw(6) << i;
        for (int j = 0; j < num_dims; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << pos_encoding(i, j);
        }
        std::cout << std::endl;
    }
    std::cout << std::string(80, '-') << std::endl;

    // Print pattern information
    std::cout << "\nPattern Analysis:" << std::endl;
    std::cout << "  - Even dimensions (0, 2, 4, ...): sine functions" << std::endl;
    std::cout << "  - Odd dimensions (1, 3, 5, ...): cosine functions" << std::endl;
    std::cout << "  - Lower dimensions vary faster (higher frequency)" << std::endl;
    std::cout << "  - Higher dimensions vary slower (lower frequency)" << std::endl;
    std::cout << "  - Range: [-1.0, 1.0] for all positions and dimensions" << std::endl;
}
