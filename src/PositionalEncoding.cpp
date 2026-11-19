// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-11-18

#include "PositionalEncoding.hpp"
#include <algorithm>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "Logger.hpp"

PositionalEncoding::PositionalEncoding(int max_len, int d_model)
    : pos_encoding(max_len, d_model), max_len(max_len), d_model(d_model) {
    // Pre-compute positional encodings using sinusoidal functions
    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            // Compute the wavelength for this dimension
            float angle =
                static_cast<float>(pos) / std::pow(10000.0f, (2.0f * static_cast<float>(i / 2)) /
                                                                 static_cast<float>(d_model));

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
    // TD-478: a wider input read past the table (std::out_of_range from Matrix), a narrower one was
    // encoded in its first columns only.
    if (input.cols != d_model) {
        throw std::invalid_argument("PositionalEncoding::forward: input has " +
                                    std::to_string(input.cols) + " columns, expected d_model " +
                                    std::to_string(d_model));
    }

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
    // won't have positional encodings added (though this should be rare).
    // TD-478: logged once per process; it went to std::cerr on every call, so once per decode step
    // on the uncached generation paths.
    static std::atomic<bool> warned_over_length{false};
    if (input.rows > max_len && !warned_over_length.exchange(true)) {
        adai::Logger::warn(
            "PositionalEncoding: input sequence length ({}) exceeds max_len ({}); positions beyond "
            "max_len get no positional encoding (logged once)",
            input.rows, max_len);
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
    // TD-478: through adai::Logger instead of std::cout, as TD-467 did for MultiHeadAttention.
    const double kilobytes = static_cast<double>(max_len) * d_model * sizeof(float) / 1024.0;
    adai::Logger::info("{}: sinusoidal (fixed), max_len {}, d_model {} ({:.1f} KB)", name, max_len,
                       d_model, kilobytes);
}

void PositionalEncoding::visualize(int num_positions, int num_dims) const {
    // TD-478: through adai::Logger, one line per row, instead of std::cout.
    num_positions = std::max(0, std::min(num_positions, max_len));
    num_dims = std::max(0, std::min(num_dims, d_model));

    adai::Logger::info("Positional encoding: first {} positions, {} dimensions (even dims sine, odd "
                       "cosine; lower dims vary faster; range [-1, 1])",
                       num_positions, num_dims);
    std::ostringstream header;
    header << std::setw(6) << "Pos";
    for (int j = 0; j < num_dims; ++j) {
        header << std::setw(10) << ("Dim" + std::to_string(j));
    }
    adai::Logger::info("{}", header.str());
    for (int i = 0; i < num_positions; ++i) {
        std::ostringstream row;
        row << std::setw(6) << i << std::fixed << std::setprecision(4);
        for (int j = 0; j < num_dims; ++j) {
            row << std::setw(10) << pos_encoding(i, j);
        }
        adai::Logger::info("{}", row.str());
    }
}
