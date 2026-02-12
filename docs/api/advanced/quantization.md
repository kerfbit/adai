# Model Quantization API Reference

**File:** `src/Quantization.hpp`
**Status:** ✅ Production-ready (Phase 5 - January 2026)
**Purpose:** Compress models with INT8/INT4 quantization (4-8x memory reduction)

---

## Overview

The `Quantizer` and `QuantizedMatrix` classes provide model compression through quantization, reducing memory usage and enabling faster inference with minimal accuracy loss.

### Key Benefits

- **4-8x memory reduction** (FP32 → INT8/INT4)
- **2-4x inference speedup** (with quantized operations)
- **< 1% accuracy loss** (INT8 with proper calibration)
- **Multiple calibration methods** for optimal quality

---

## Quantization Modes

### Symmetric INT8

- **Range:** [-127, 127]
- **Formula:** `Q(x) = round(x / scale)`
- **Use:** General purpose, best balance

### Asymmetric INT8

- **Range:** [0, 255]
- **Formula:** `Q(x) = round(x / scale) + zero_point`
- **Use:** Activations with non-symmetric distributions

### INT4

- **Range:** [-7, 7] (symmetric) or [0, 15] (asymmetric)
- **Formula:** Same as INT8 but 4-bit storage
- **Use:** Maximum compression, slight accuracy trade-off

---

## Quantizer Class

```cpp
class Quantizer {
public:
    Quantizer(QuantizationMode mode = QuantizationMode::SYMMETRIC_INT8,
              CalibrationMethod calibration = CalibrationMethod::MIN_MAX,
              float percentile = 0.999f);

    QuantizationParams calibrate(const std::vector<float>& data);
    int8_t quantize(float value, const QuantizationParams& params) const;
    float dequantize(int8_t qvalue, const QuantizationParams& params) const;

    std::vector<int8_t> quantize_matrix(const Matrix& mat,
                                         const QuantizationParams& params) const;
    Matrix dequantize_matrix(const std::vector<int8_t>& qdata,
                            int rows, int cols,
                            const QuantizationParams& params) const;

    float compute_quantization_error(const std::vector<float>& data,
                                     const QuantizationParams& params) const;
};
```

---

## Calibration Methods

### MIN_MAX
```cpp
// Simple range-based
scale = (max - min) / (qmax - qmin)
zero_point = round(-min / scale)
```

**Pros:** Fast, simple
**Cons:** Sensitive to outliers

### PERCENTILE
```cpp
// Clip outliers at percentile threshold
float p_min = percentile(data, 1 - percentile);
float p_max = percentile(data, percentile);
scale = (p_max - p_min) / (qmax - qmin)
```

**Pros:** Outlier-robust (recommended)
**Cons:** Slightly slower

### MSE
```cpp
// Minimize reconstruction error
argmin_{scale,zp} MSE(Q(D(x)))
```

**Pros:** Optimal quality
**Cons:** Slowest calibration

---

## QuantizedMatrix Class

```cpp
class QuantizedMatrix {
public:
    void quantize_from(const Matrix& mat, Quantizer& quantizer);
    Matrix dequantize(Quantizer& quantizer) const;

    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

    int rows() const;
    int cols() const;
    float memory_reduction() const;
    const QuantizationParams& params() const;
};
```

---

## Complete Example

```cpp
#include "Quantization.hpp"

int main() {
    // 1. Create weight matrix
    Matrix weights(768, 768);
    load_weights(weights, "model.bin");

    std::cout << "Original size: "
              << (768*768*sizeof(float)/1024) << " KB\n";

    // 2. Create quantizer with percentile calibration
    Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8,
                       CalibrationMethod::PERCENTILE,
                       0.999f);

    // 3. Calibrate on weight data
    std::vector<float> calibration_data;
    for (int i = 0; i < weights.rows; i++) {
        for (int j = 0; j < weights.cols; j++) {
            calibration_data.push_back(weights(i, j));
        }
    }

    QuantizationParams params = quantizer.calibrate(calibration_data);

    std::cout << "Quantization params:\n";
    std::cout << "  Scale: " << params.scale << "\n";
    std::cout << "  Zero point: " << params.zero_point << "\n";

    // 4. Quantize matrix
    QuantizedMatrix qmat;
    qmat.quantize_from(weights, quantizer);

    std::cout << "Compressed size: "
              << (768*768*sizeof(int8_t)/1024) << " KB\n";
    std::cout << "Memory reduction: "
              << qmat.memory_reduction() << "x\n";

    // 5. Measure error
    float error = quantizer.compute_quantization_error(
        calibration_data, params);
    std::cout << "MSE: " << error << "\n";
    std::cout << "RMSE: " << std::sqrt(error) << "\n";

    // 6. Save quantized model
    qmat.save("model_int8.bin");

    // 7. Load and use
    QuantizedMatrix loaded;
    loaded.load("model_int8.bin");

    Matrix reconstructed = loaded.dequantize(quantizer);
    // Use reconstructed for inference

    return 0;
}
```

---

## Usage Patterns

### Full Model Quantization

```cpp
// Quantize all model weights
std::vector<std::string> weight_names = {
    "W_Q", "W_K", "W_V", "W_O",
    "W_FF1", "W_FF2"
};

Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8,
                   CalibrationMethod::PERCENTILE);

for (const auto& name : weight_names) {
    Matrix weights = load_weights(name);

    // Calibrate per-layer
    auto data = matrix_to_vector(weights);
    auto params = quantizer.calibrate(data);

    // Quantize and save
    QuantizedMatrix qmat;
    qmat.quantize_from(weights, quantizer);
    qmat.save(name + "_int8.bin");
}
```

### Post-Training Quantization (PTQ)

```cpp
// Step 1: Collect activation statistics
std::vector<float> activation_stats;
for (auto& sample : calibration_set) {
    Matrix activations = model.forward(sample);
    collect_stats(activations, activation_stats);
}

// Step 2: Calibrate
Quantizer act_quantizer(QuantizationMode::ASYMMETRIC_INT8,
                       CalibrationMethod::MSE);
auto params = act_quantizer.calibrate(activation_stats);

// Step 3: Apply at inference
Matrix quant_acts = quantize_tensor(activations, params);
Matrix result = quantized_forward(quant_acts);
```

---

## Performance Characteristics

### Memory Savings

| Precision | Bytes/Value | Model Size (350M params) | Reduction |
| ----------- | ------------- | -------------------------- | ----------- |
| FP32 | 4 | 1.4 GB | 1x |
| FP16 | 2 | 700 MB | 2x |
| INT8 | 1 | 350 MB | 4x |
| INT4 | 0.5 | 175 MB | 8x |

### Accuracy Impact

| Quantization | Typical Accuracy Loss |
| -------------- | ---------------------- |
| INT8 (min-max) | 0.5-2% |
| INT8 (percentile) | 0.1-1% |
| INT8 (MSE) | <0.5% |
| INT4 | 1-5% |

### Inference Speedup

- **CPU:** 2-3x faster (with quantized kernels)
- **GPU:** 2-4x faster (TensorRT, etc.)
- **Edge devices:** 3-5x faster

---

## Best Practices

### 1. Calibration Set Size
```cpp
// Use 100-1000 samples for calibration
// More samples = better statistics
std::vector<float> calibration_data;
for (int i = 0; i < 1000; i++) {
    auto sample = get_representative_sample(i);
    collect_data(sample, calibration_data);
}
```

### 2. Percentile Selection
```cpp
// 99.9% (0.999) works well for most cases
// Higher = less clipping, more precision
// Lower = more clipping, handle outliers better
Quantizer quantizer(mode, CalibrationMethod::PERCENTILE, 0.999f);
```

### 3. Per-Channel vs Per-Tensor
```cpp
// Per-tensor (current implementation):
// - One scale/zero-point for entire tensor
// - Faster, less overhead
// - Good for most cases

// Per-channel (future enhancement):
// - Separate scale/zero-point per output channel
// - Better accuracy for conv/linear layers
// - More overhead
```

### 4. Mixed Precision
```cpp
// Keep sensitive layers in FP32
std::vector<std::string> fp32_layers = {"embedding", "lm_head"};

for (const auto& layer : model.layers) {
    if (is_sensitive(layer.name)) {
        // Keep FP32
        continue;
    } else {
        // Quantize to INT8
        quantize_layer(layer);
    }
}
```

---

## Quantization Modes Reference

```cpp
enum class QuantizationMode {
    SYMMETRIC_INT8,    // [-127, 127], no zero-point
    ASYMMETRIC_INT8,   // [0, 255], with zero-point
    SYMMETRIC_INT4,    // [-7, 7], 4-bit storage
    ASYMMETRIC_INT4    // [0, 15], 4-bit storage
};

enum class CalibrationMethod {
    MIN_MAX,      // Simple range-based
    PERCENTILE,   // Outlier-robust (recommended)
    MSE           // Minimize reconstruction error
};
```

---

## Helper Functions

```cpp
// Print quantization statistics
void print_quantization_stats(const Matrix& original,
                              const QuantizedMatrix& quantized,
                              Quantizer& quantizer);

// Example output:
// Matrix size: 768x768
// Original: 2304 KB (FP32)
// Quantized: 576 KB (INT8)
// Memory reduction: 4.0x
// MSE: 2.06e-09
// RMSE: 4.54e-05
// Max error: 0.0002
```

---

## Combining with LoRA

```cpp
// 1. Train LoRA adapter (full precision)
LoRAAdapter adapter(768, 768, 8);
train(adapter, data);

// 2. Merge adapter with base
Matrix W_merged = adapter.merge_with_base(W_base);

// 3. Quantize merged weights
Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8);
auto params = quantizer.calibrate(extract_data(W_merged));

QuantizedMatrix qmat;
qmat.quantize_from(W_merged, quantizer);
qmat.save("model_lora_int8.bin");

// Result:
// - LoRA: 100x fewer training params
// - INT8: 4x smaller deployment
// - Combined: Extremely efficient!
```

---

## Test Coverage

**File:** `tests/phase5_test.cpp`
**Test Cases:** 15

- Constructor defaults
- Calibration methods (min-max, percentile, MSE)
- Quantize/dequantize round-trip
- Matrix quantization
- Asymmetric quantization
- INT4 mode
- Error analysis
- Save/load
- Memory reduction calculation

**Pass Rate:** 100%

---

## See Also

- [LoRA](lora.md) - Combine for maximum efficiency
- [Phase 5 Guide](../../guides/phase5-advanced-features.md) - Complete quantization tutorial
- [Chatbot Completeness](../../reference/chatbot-completeness.md) - Integration details

---

**Last Updated:** January 25, 2026
**Version:** 1.0
**Status:** Production-ready
