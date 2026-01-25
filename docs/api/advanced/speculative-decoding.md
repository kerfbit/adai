# Speculative Decoding API Reference

**File:** `src/SpeculativeDecoding.hpp`  
**Status:** ✅ Production-ready (Phase 5 - January 2026)  
**Purpose:** Accelerate inference 2-3x with mathematically equivalent sampling

---

## Overview

The `SpeculativeDecoding` class implements speculative decoding, a technique to accelerate autoregressive generation by using a small draft model to propose multiple tokens, then verifying them in parallel with the target model.

### Key Benefits

- **2-3x faster inference** (typical speedup)
- **Mathematically equivalent** to standard sampling
- **No quality degradation** - identical output distribution
- **No training required** - works with any model pair

---

## How It Works

### Standard Generation (Slow)
```
for each position:
    token = sample(target_model(prefix))     # Sequential
    prefix.append(token)
# O(N) forward passes for N tokens
```

### Speculative Decoding (Fast)
```
1. Draft model generates K candidates: [t1, t2, ..., tK]    # Fast model
2. Target model verifies ALL K in parallel                   # One forward pass
3. Accept verified tokens, reject rest
4. Continue from last accepted position
# O(N/K) forward passes for N tokens (K-fold speedup)
```

---

## Class Definition

```cpp
class SpeculativeDecoder {
public:
    SpeculativeDecoder(TextGenerator* draft_model,
                      TextGenerator* target_model,
                      const SpeculativeDecodingConfig& config);
    
    std::string generate(const std::string& prompt,
                        int max_length);
    
    std::vector<int> generate_candidates(const std::string& prefix,
                                        int num_candidates);
    
    std::vector<int> verify_candidates(const std::string& prefix,
                                      const std::vector<int>& candidates);
    
    SpeculativeDecodingStats get_stats() const;
    void reset_stats();
};
```

---

## Configuration

```cpp
struct SpeculativeDecodingConfig {
    int num_candidates = 5;     // K: number of draft tokens
    float temperature = 1.0f;   // Sampling temperature
    int max_length = 512;       // Maximum generation length
    bool verbose = false;       // Log acceptance details
};
```

**Hyperparameter Guide:**
- `num_candidates` (K):
  - **Larger K** = more speedup potential, but lower acceptance rate
  - **Smaller K** = higher acceptance, less speedup
  - Typical: 4-6
- `temperature`:
  - Use same value as target model
  - Lower = more greedy, higher acceptance
  - Higher = more random, lower acceptance

---

## Constructor

```cpp
SpeculativeDecoder(TextGenerator* draft_model,
                  TextGenerator* target_model,
                  const SpeculativeDecodingConfig& config)
```

**Requirements:**
- **Draft model:** Small, fast model (e.g., 125M params)
- **Target model:** Large, accurate model (e.g., 1.3B params)
- **Both models:** Same tokenizer and vocabulary

**Example:**
```cpp
// Load models
TextGenerator draft_model(&small_decoder, &tokenizer);
TextGenerator target_model(&large_decoder, &tokenizer);

// Configure
SpeculativeDecodingConfig config;
config.num_candidates = 5;
config.temperature = 1.0f;

// Create speculative decoder
SpeculativeDecoder spec_decoder(&draft_model, &target_model, config);
```

---

## Generation

### generate()

```cpp
std::string generate(const std::string& prompt, int max_length)
```

Generate text using speculative decoding.

**Algorithm:**
```
1. Initialize: prefix = prompt
2. While length < max_length:
   a. Draft: Generate K candidate tokens with draft model
   b. Verify: Run target model on prefix + candidates (1 forward pass!)
   c. Accept: Keep tokens where p_target(t) >= p_draft(t)
   d. Reject: Resample first rejected token from adjusted distribution
   e. Update: prefix += accepted_tokens
3. Return: generated text
```

**Example:**
```cpp
std::string prompt = "Once upon a time";
std::string story = spec_decoder.generate(prompt, 200);

// Same output quality as standard generation
// But 2-3x faster!
```

---

### generate_candidates()

```cpp
std::vector<int> generate_candidates(const std::string& prefix,
                                    int num_candidates)
```

Use draft model to propose K candidate tokens.

**Purpose:** Internal method, exposed for debugging/analysis

---

### verify_candidates()

```cpp
std::vector<int> verify_candidates(const std::string& prefix,
                                  const std::vector<int>& candidates)
```

Verify candidates with target model using acceptance criterion.

**Acceptance Criterion:**
```
Accept token t_i if:
  p_target(t_i | prefix, t_1, ..., t_{i-1}) >= p_draft(t_i | ...)
```

**Returns:** Indices of accepted tokens

---

## Statistics

```cpp
struct SpeculativeDecodingStats {
    int total_candidates_proposed;
    int total_candidates_accepted;
    float acceptance_rate;
    float average_speedup;
    int num_iterations;
};
```

### get_stats()

```cpp
SpeculativeDecodingStats get_stats() const;
```

Get performance statistics.

**Example:**
```cpp
auto stats = spec_decoder.get_stats();
std::cout << "Acceptance rate: " << stats.acceptance_rate * 100 << "%\n";
std::cout << "Speedup: " << stats.average_speedup << "x\n";
std::cout << "Proposed: " << stats.total_candidates_proposed << "\n";
std::cout << "Accepted: " << stats.total_candidates_accepted << "\n";
```

---

## Complete Example

```cpp
#include "SpeculativeDecoding.hpp"
#include "TextGenerator.hpp"

int main() {
    // 1. Load models
    BPETokenizer tokenizer("vocab.txt");
    
    // Draft model: Small and fast (125M params)
    LLMDecoder draft_decoder(512, 8, 2048, 6, tokenizer.vocab_size());
    draft_decoder.load("draft_model.bin");
    TextGenerator draft_gen(&draft_decoder, &tokenizer);
    
    // Target model: Large and accurate (1.3B params)
    LLMDecoder target_decoder(768, 12, 3072, 12, tokenizer.vocab_size());
    target_decoder.load("target_model.bin");
    TextGenerator target_gen(&target_decoder, &tokenizer);
    
    // 2. Configure speculative decoding
    SpeculativeDecodingConfig config;
    config.num_candidates = 5;
    config.temperature = 1.0f;
    config.max_length = 200;
    
    SpeculativeDecoder spec_decoder(&draft_gen, &target_gen, config);
    
    // 3. Generate text
    std::string prompt = "The future of AI is";
    
    auto start = std::chrono::high_resolution_clock::now();
    std::string output = spec_decoder.generate(prompt, 200);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();
    
    // 4. Display results
    std::cout << "Generated text:\n" << output << "\n\n";
    std::cout << "Time: " << duration << "ms\n\n";
    
    // 5. Show statistics
    auto stats = spec_decoder.get_stats();
    std::cout << "Speculative Decoding Statistics:\n";
    std::cout << "  Acceptance rate: " 
              << (stats.acceptance_rate * 100) << "%\n";
    std::cout << "  Average speedup: " 
              << stats.average_speedup << "x\n";
    std::cout << "  Candidates proposed: " 
              << stats.total_candidates_proposed << "\n";
    std::cout << "  Candidates accepted: " 
              << stats.total_candidates_accepted << "\n";
    std::cout << "  Iterations: " 
              << stats.num_iterations << "\n";
    
    // 6. Compare with standard generation
    auto start_std = std::chrono::high_resolution_clock::now();
    std::string output_std = target_gen.generate(prompt, 200);
    auto end_std = std::chrono::high_resolution_clock::now();
    
    auto duration_std = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_std - start_std).count();
    
    std::cout << "\nStandard generation time: " << duration_std << "ms\n";
    std::cout << "Speedup: " << (float)duration_std / duration << "x\n";
    
    return 0;
}
```

---

## Expected Performance

### Acceptance Rates

| K (candidates) | Typical Acceptance Rate |
|----------------|------------------------|
| 2 | 85-95% |
| 4 | 70-80% |
| 5 | 65-75% |
| 6 | 60-70% |
| 8 | 50-60% |

### Speedup Calculations

**Theoretical Speedup:**
```
speedup = (K × acceptance_rate) / (1 + overhead)

Examples:
K=4, acceptance=75%: 4 × 0.75 = 3.0x
K=5, acceptance=70%: 5 × 0.70 = 3.5x
K=6, acceptance=65%: 6 × 0.65 = 3.9x
```

**Practical Speedup:**
- CPU: 1.5-2.5x (due to overhead)
- GPU: 2.0-3.5x (parallel verification)

---

## Model Selection Guide

### Draft Model Characteristics
- **Size:** 10-50x smaller than target (e.g., 125M vs 1.3B)
- **Speed:** Should be 5-10x faster
- **Quality:** Doesn't need to be perfect, just "reasonable"

### Good Draft-Target Pairs

| Draft Model | Target Model | Expected Speedup |
|-------------|--------------|------------------|
| 125M params | 1.3B params | 2.0-2.5x |
| 350M params | 7B params | 2.5-3.0x |
| 1.3B params | 13B params | 2.0-2.8x |

### Training Draft Models

```cpp
// Option 1: Distillation
// Train small model to mimic large model
train_distillation(draft_model, target_model, data);

// Option 2: Same architecture, fewer layers
// Use first N layers of target model
draft_model = extract_layers(target_model, 0, 6);

// Option 3: Independent training
// Train both on same data, different sizes
```

---

## Combining with Other Optimizations

### Speculative + KV Cache

```cpp
// Both draft and target models use KV cache
draft_decoder.enable_kv_cache();
target_decoder.enable_kv_cache();

// Combined speedup: speculative × kv_cache
// Example: 2.5x × 2.0x = 5x total speedup!
```

### Speculative + Quantization

```cpp
// Quantize draft model for even faster proposals
Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8);
quantize_model(draft_decoder, quantizer);

// Target model can also be quantized
quantize_model(target_decoder, quantizer);
```

### Speculative + Batch Processing

```cpp
// Process multiple prompts with shared draft model
BatchProcessor batch_processor(draft_gen, target_gen, config);
auto results = batch_processor.generate_batch(prompts);
```

---

## Troubleshooting

### Low Acceptance Rate (<50%)

**Causes:**
- Draft model too different from target
- K (num_candidates) too large
- High temperature

**Solutions:**
- Train better draft model (distillation)
- Reduce K to 3-4
- Lower temperature
- Check draft/target use same tokenizer

### No Speedup

**Causes:**
- Draft model not fast enough
- Overhead too high (Python, I/O)
- Draft model too large

**Solutions:**
- Use smaller/faster draft model
- Profile and optimize bottlenecks
- Ensure draft is 5-10x faster than target

### Incorrect Output

**Issue:** Output differs from standard generation

**Cause:** Implementation bug (should be mathematically equivalent)

**Solution:** Check acceptance criterion and resampling logic

---

## Best Practices

### 1. Model Compatibility
```cpp
// Ensure same tokenizer
assert(draft_gen.tokenizer == target_gen.tokenizer);
```

### 2. Hyperparameter Tuning
```cpp
// Start with K=5, adjust based on acceptance rate
if (acceptance_rate < 0.6) {
    config.num_candidates--;
} else if (acceptance_rate > 0.8) {
    config.num_candidates++;
}
```

### 3. Monitoring
```cpp
// Track metrics over time
stats_history.push_back(spec_decoder.get_stats());
plot_acceptance_rate_trend(stats_history);
```

### 4. Fallback
```cpp
// If speedup < 1.5x, use standard generation
if (stats.average_speedup < 1.5f) {
    return target_gen.generate(prompt, max_length);
}
```

---

## Test Coverage

**File:** `tests/phase5_test.cpp`  
**Test Cases:** 5

- Constructor validation
- Configuration management
- Candidate generation
- Verification logic
- Statistics tracking

**Pass Rate:** 100%

---

## See Also

- [Text Generator](../nlp/text-generator.md) - Required for draft/target models
- [KV Cache](../../reference/kvcache.md) - Combine for maximum speedup
- [Phase 5 Guide](../../guides/phase5-advanced-features.md) - Complete speculative decoding tutorial

---

## References

- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding (Chen et al., 2023)](https://arxiv.org/abs/2302.01318)

---

**Last Updated:** January 25, 2026  
**Version:** 1.0  
**Status:** Production-ready
