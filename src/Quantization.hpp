#ifndef QUANTIZATION_HPP
#define QUANTIZATION_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "Matrix.hpp"

/**
 * @file Quantization.hpp
 * @brief Model Quantization for Efficient Inference
 *
 * Quantization reduces model size and inference latency by representing weights
 * and activations with lower precision (INT8, INT4) instead of FP32.
 *
 * Techniques Implemented:
 * 1. Post-Training Quantization (PTQ)
 *    - Asymmetric quantization: Q = round(x/scale + zero_point)
 *    - Symmetric quantization: Q = round(x/scale)
 *
 * 2. Calibration
 *    - Min-Max: Use observed min/max values
 *    - Percentile: Clip outliers for better range
 *
 * 3. Per-tensor vs Per-channel quantization
 *    - Per-tensor: Single scale/zero-point for entire tensor
 *    - Per-channel: Different scale/zero-point per output channel
 *
 * Benefits:
 * - 4x memory reduction (FP32 → INT8)
 * - 8x memory reduction (FP32 → INT4)
 * - Faster inference on quantized hardware
 * - Minimal accuracy loss with proper calibration
 *
 * @version 1.0
 * @date January 2026
 */

/**
 * @brief Quantization modes
 */
enum class QuantizationMode {
    SYMMETRIC_INT8,   // Symmetric INT8: [-127, 127]
    ASYMMETRIC_INT8,  // Asymmetric INT8: [0, 255]
    SYMMETRIC_INT4,   // Symmetric INT4: [-7, 7]
    ASYMMETRIC_INT4   // Asymmetric INT4: [0, 15]
};

/**
 * @brief Calibration methods
 */
enum class CalibrationMethod {
    MIN_MAX,     // Use min/max observed values
    PERCENTILE,  // Clip outliers at percentile (e.g., 99.9%)
    MSE          // Minimize mean squared error
};

/**
 * @struct QuantizationParams
 * @brief Parameters for quantization/dequantization
 */
struct QuantizationParams {
    float scale;     // Scaling factor
    int zero_point;  // Zero point for asymmetric quantization
    int qmin;        // Minimum quantized value
    int qmax;        // Maximum quantized value

    QuantizationParams() : scale(1.0f), zero_point(0), qmin(-127), qmax(127) {}

    QuantizationParams(float s, int zp, int qmin_val, int qmax_val)
        : scale(s), zero_point(zp), qmin(qmin_val), qmax(qmax_val) {}
};

/**
 * @class Quantizer
 * @brief Utility for quantizing and dequantizing tensors
 */
class Quantizer {
   private:
    QuantizationMode mode_;
    CalibrationMethod calibration_;
    float percentile_;  // For percentile calibration (e.g., 0.999)

    /**
     * @brief Get quantization range based on mode
     */
    std::pair<int, int> get_quant_range() const {
        switch (mode_) {
            case QuantizationMode::SYMMETRIC_INT8:
                return {-127, 127};
            case QuantizationMode::ASYMMETRIC_INT8:
                return {0, 255};
            case QuantizationMode::SYMMETRIC_INT4:
                return {-7, 7};
            case QuantizationMode::ASYMMETRIC_INT4:
                return {0, 15};
            default:
                return {-127, 127};
        }
    }

    /**
     * @brief Find min/max values in tensor
     */
    std::pair<float, float> find_min_max(const std::vector<float>& data) const {
        if (data.empty()) {
            throw std::invalid_argument("Cannot find min/max of empty data");
        }

        float min_val = data[0];
        float max_val = data[0];

        for (float val : data) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        return {min_val, max_val};
    }

    /**
     * @brief Find percentile-based min/max to clip outliers
     */
    std::pair<float, float> find_percentile_range(std::vector<float> data) const {
        if (data.empty()) {
            throw std::invalid_argument("Cannot find percentile of empty data");
        }

        // Sort data
        std::sort(data.begin(), data.end());

        // Find percentile indices
        int lower_idx = static_cast<int>((1.0f - percentile_) * data.size() / 2);
        int upper_idx = static_cast<int>((percentile_ + (1.0f - percentile_) / 2) * data.size());

        lower_idx = std::max(0, std::min(lower_idx, static_cast<int>(data.size()) - 1));
        upper_idx = std::max(0, std::min(upper_idx, static_cast<int>(data.size()) - 1));

        return {data[lower_idx], data[upper_idx]};
    }

   public:
    /**
     * @brief Construct quantizer
     *
     * @param mode Quantization mode (INT8/INT4, symmetric/asymmetric)
     * @param calibration Calibration method
     * @param percentile Percentile for outlier clipping (default 0.999 = 99.9%)
     */
    Quantizer(QuantizationMode mode = QuantizationMode::SYMMETRIC_INT8,
              CalibrationMethod calibration = CalibrationMethod::MIN_MAX, float percentile = 0.999f)
        : mode_(mode), calibration_(calibration), percentile_(percentile) {}

    /**
     * @brief Calibrate quantization parameters from data
     *
     * @param data Calibration data (e.g., activation values)
     * @return Quantization parameters
     */
    QuantizationParams calibrate(const std::vector<float>& data) {
        auto [qmin, qmax] = get_quant_range();

        // Find data range based on calibration method
        float min_val, max_val;
        if (calibration_ == CalibrationMethod::MIN_MAX) {
            std::tie(min_val, max_val) = find_min_max(data);
        } else if (calibration_ == CalibrationMethod::PERCENTILE) {
            std::tie(min_val, max_val) = find_percentile_range(data);
        } else {
            std::tie(min_val, max_val) = find_min_max(data);
        }

        // Ensure non-zero range
        if (std::abs(max_val - min_val) < 1e-8f) {
            max_val = min_val + 1e-6f;
        }

        QuantizationParams params;
        params.qmin = qmin;
        params.qmax = qmax;

        // Compute scale and zero_point
        bool is_symmetric = (mode_ == QuantizationMode::SYMMETRIC_INT8 ||
                             mode_ == QuantizationMode::SYMMETRIC_INT4);

        if (is_symmetric) {
            // Symmetric quantization: scale based on max abs value
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            params.scale = abs_max / qmax;
            params.zero_point = 0;
        } else {
            // Asymmetric quantization
            params.scale = (max_val - min_val) / (qmax - qmin);
            params.zero_point = qmin - static_cast<int>(std::round(min_val / params.scale));
            params.zero_point = std::max(qmin, std::min(qmax, params.zero_point));
        }

        // Avoid division by zero
        if (params.scale < 1e-8f) {
            params.scale = 1e-6f;
        }

        return params;
    }

    /**
     * @brief Quantize a single value
     *
     * @param value Floating-point value
     * @param params Quantization parameters
     * @return Quantized integer
     */
    int quantize_value(float value, const QuantizationParams& params) const {
        int quantized = static_cast<int>(std::round(value / params.scale) + params.zero_point);
        return std::max(params.qmin, std::min(params.qmax, quantized));
    }

    /**
     * @brief Dequantize a single value
     *
     * @param quantized Quantized integer
     * @param params Quantization parameters
     * @return Floating-point value
     */
    float dequantize_value(int quantized, const QuantizationParams& params) const {
        return params.scale * (quantized - params.zero_point);
    }

    /**
     * @brief Quantize a vector
     *
     * @param data Input floating-point data
     * @param params Quantization parameters
     * @return Quantized integers
     */
    std::vector<int8_t> quantize(const std::vector<float>& data,
                                 const QuantizationParams& params) const {
        std::vector<int8_t> quantized(data.size());

        for (size_t i = 0; i < data.size(); i++) {
            quantized[i] = static_cast<int8_t>(quantize_value(data[i], params));
        }

        return quantized;
    }

    /**
     * @brief Dequantize a vector
     *
     * @param quantized Quantized integers
     * @param params Quantization parameters
     * @return Dequantized floating-point data
     */
    std::vector<float> dequantize(const std::vector<int8_t>& quantized,
                                  const QuantizationParams& params) const {
        std::vector<float> data(quantized.size());

        for (size_t i = 0; i < quantized.size(); i++) {
            data[i] = dequantize_value(quantized[i], params);
        }

        return data;
    }

    /**
     * @brief Quantize a Matrix
     *
     * @param mat Input matrix
     * @param params Quantization parameters
     * @return Quantized matrix data (flattened)
     */
    std::vector<int8_t> quantize_matrix(const Matrix& mat, const QuantizationParams& params) const {
        std::vector<float> data;
        for (int r = 0; r < mat.rows; r++) {
            for (int c = 0; c < mat.cols; c++) {
                data.push_back(mat(r, c));
            }
        }
        return quantize(data, params);
    }

    /**
     * @brief Dequantize to Matrix
     *
     * @param quantized Quantized data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param params Quantization parameters
     * @return Dequantized matrix
     */
    Matrix dequantize_matrix(const std::vector<int8_t>& quantized, int rows, int cols,
                             const QuantizationParams& params) const {
        if ((int)quantized.size() != rows * cols) {
            throw std::invalid_argument("Quantized data size mismatch");
        }

        std::vector<float> data = dequantize(quantized, params);

        Matrix mat(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                mat(r, c) = data[r * cols + c];
            }
        }

        return mat;
    }

    /**
     * @brief Calculate quantization error (MSE)
     *
     * @param original Original floating-point data
     * @param params Quantization parameters
     * @return Mean squared error
     */
    float compute_quantization_error(const std::vector<float>& original,
                                     const QuantizationParams& params) const {
        auto quantized = quantize(original, params);
        auto dequantized = dequantize(quantized, params);

        float mse = 0.0f;
        for (size_t i = 0; i < original.size(); i++) {
            float diff = original[i] - dequantized[i];
            mse += diff * diff;
        }
        mse /= original.size();

        return mse;
    }
};

/**
 * @class QuantizedMatrix
 * @brief Storage for quantized matrix with metadata
 */
class QuantizedMatrix {
   private:
    std::vector<int8_t> data_;
    int rows_;
    int cols_;
    QuantizationParams params_;

   public:
    QuantizedMatrix() : rows_(0), cols_(0) {}

    QuantizedMatrix(const std::vector<int8_t>& data, int rows, int cols,
                    const QuantizationParams& params)
        : data_(data), rows_(rows), cols_(cols), params_(params) {}

    /**
     * @brief Quantize and store matrix
     */
    void quantize_from(const Matrix& mat, Quantizer& quantizer) {
        rows_ = mat.rows;
        cols_ = mat.cols;

        // Calibrate on matrix data
        std::vector<float> calibration_data;
        for (int r = 0; r < rows_; r++) {
            for (int c = 0; c < cols_; c++) {
                calibration_data.push_back(mat(r, c));
            }
        }

        params_ = quantizer.calibrate(calibration_data);
        data_ = quantizer.quantize_matrix(mat, params_);
    }

    /**
     * @brief Dequantize to Matrix
     */
    Matrix dequantize(Quantizer& quantizer) const {
        return quantizer.dequantize_matrix(data_, rows_, cols_, params_);
    }

    /**
     * @brief Get memory savings ratio
     */
    float memory_reduction() const {
        int original_bytes = rows_ * cols_ * sizeof(float);
        int quantized_bytes = data_.size() * sizeof(int8_t) + sizeof(params_);
        return static_cast<float>(original_bytes) / quantized_bytes;
    }

    /**
     * @brief Save to file
     */
    void save(const std::string& filepath) const {
        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        file.write(reinterpret_cast<const char*>(&rows_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&params_.scale), sizeof(float));
        file.write(reinterpret_cast<const char*>(&params_.zero_point), sizeof(int));
        file.write(reinterpret_cast<const char*>(&params_.qmin), sizeof(int));
        file.write(reinterpret_cast<const char*>(&params_.qmax), sizeof(int));

        int size = data_.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        file.write(reinterpret_cast<const char*>(data_.data()), size * sizeof(int8_t));
    }

    /**
     * @brief Load from file
     */
    void load(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        file.read(reinterpret_cast<char*>(&rows_), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols_), sizeof(int));
        file.read(reinterpret_cast<char*>(&params_.scale), sizeof(float));
        file.read(reinterpret_cast<char*>(&params_.zero_point), sizeof(int));
        file.read(reinterpret_cast<char*>(&params_.qmin), sizeof(int));
        file.read(reinterpret_cast<char*>(&params_.qmax), sizeof(int));

        int size;
        file.read(reinterpret_cast<char*>(&size), sizeof(int));
        data_.resize(size);
        file.read(reinterpret_cast<char*>(data_.data()), size * sizeof(int8_t));
    }

    int rows() const {
        return rows_;
    }
    int cols() const {
        return cols_;
    }
    const QuantizationParams& params() const {
        return params_;
    }
};

/**
 * @brief Print quantization statistics
 */
inline void print_quantization_stats(const Matrix& original, const QuantizedMatrix& quantized,
                                     Quantizer& quantizer) {
    Matrix dequantized = quantized.dequantize(quantizer);

    // Calculate error
    float mse = 0.0f;
    for (int r = 0; r < original.rows; r++) {
        for (int c = 0; c < original.cols; c++) {
            float diff = original(r, c) - dequantized(r, c);
            mse += diff * diff;
        }
    }
    mse /= (original.rows * original.cols);

    std::cout << "=== Quantization Statistics ===\n";
    std::cout << "Matrix size: " << original.rows << "x" << original.cols << "\n";
    std::cout << "Memory reduction: " << quantized.memory_reduction() << "x\n";
    std::cout << "MSE: " << mse << "\n";
    std::cout << "RMSE: " << std::sqrt(mse) << "\n";
}

#endif  // QUANTIZATION_HPP
