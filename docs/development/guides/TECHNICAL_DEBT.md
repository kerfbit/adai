# Technical Debt Tracker

This document tracks all known technical debt items, TODOs, and improvement opportunities in the ADAI codebase. Items are prioritized and linked to GitHub issues for tracking.

## Overview

**Last Updated:** February 17, 2026  
**Total Items:** 5  
**High Priority:** 0  
**Medium Priority:** 3  
**Low Priority:** 2

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Active Technical Debt](#active-technical-debt)
  - [TD-003: GPU Memory Management Optimization](#td-003-gpu-memory-management-optimization)
  - [TD-004: Enhanced Metrics Tracking for Training Sessions](#td-004-enhanced-metrics-tracking-for-training-sessions)
  - [TD-005: Checkpoint Management and Symbolic Links](#td-005-checkpoint-management-and-symbolic-links)
  - [TD-006: Fill-in-the-Middle (FIM) Training Data Generation](#td-006-fill-in-the-middle-fim-training-data-generation)
  - [TD-007: Matrix Operations SIMD Acceleration](#td-007-matrix-operations-simd-acceleration)
- [Resolved Items](#resolved-items)
  - [TD-002: Improve Error Handling in BPE Tokenizer](#td-002-improve-error-handling-in-bpe-tokenizer)
  - [TD-001: Complete Optimizer Parameter Exposure](#td-001-complete-optimizer-parameter-exposure)
- [Future Improvements](#future-improvements)
  - [Performance Optimizations](#performance-optimizations)
  - [Code Quality](#code-quality)
  - [Developer Experience](#developer-experience)
- [Process Guidelines](#process-guidelines)
  - [Adding New Technical Debt](#adding-new-technical-debt)
  - [Prioritization Criteria](#prioritization-criteria)
  - [Resolving Technical Debt](#resolving-technical-debt)
- [Statistics](#statistics)
  - [By Priority](#by-priority)
  - [By Component](#by-component)
  - [Effort Distribution](#effort-distribution)
- [References](#references)

## Active Technical Debt

### TD-003: GPU Memory Management Optimization

**Priority:** LOW  
**Status:** Optional Enhancement  
**Component:** GPU / Performance  
**Created:** January 28, 2026

**Description:**
Current GPU implementation transfers data between CPU and GPU for each operation. For better performance with repeated GPU operations, persistent GPU memory buffers could be implemented.

**Current Behavior:**

```cpp
Matrix C = A.multiply_gpu(B);  // Transfers A, B to GPU, then result back
Matrix D = C.add_gpu(A);       // Transfers C, A to GPU again, then result back
```

**Desired Behavior:**

```cpp
// Future enhancement - keep data on GPU between operations
GPUMatrix A_gpu = A.to_gpu();
GPUMatrix B_gpu = B.to_gpu();
GPUMatrix C_gpu = A_gpu.multiply(B_gpu);  // No transfers
GPUMatrix D_gpu = C_gpu.add(A_gpu);       // No transfers
Matrix D = D_gpu.to_cpu();  // Only transfer final result
```

**Impact:**

- **Performance:** Would significantly improve performance for sequences of GPU operations
- **Current Workaround:** Current implementation works correctly, just not optimal for chained operations
- **Users Affected:** Only users leveraging GPU acceleration with multiple sequential operations

**Implementation Notes:**

- Create `GPUMatrix` class that maintains device memory
- Implement conversion operators (`to_gpu()`, `to_cpu()`)
- Add smart memory management (RAII pattern already in place with `GPUMemory`)
- Consider implementing memory pools for frequently allocated sizes

**Files to Modify:**

- `src/gpu/GPUUtils.hpp` - Add `GPUMatrix` class
- `src/Matrix.hpp` - Add conversion methods
- `src/gpu/MatrixGPU.cu` - Update operations to work with persistent GPU memory

**Related:**

- GPU acceleration implemented in commit [current]
- See `docs/guides/building.md` for GPU compilation instructions

---

### TD-004: Enhanced Metrics Tracking for Training Sessions

**Priority:** MEDIUM  
**Status:** Planned  
**Component:** Training / IncrementalTrainer  
**Created:** February 17, 2026  
**Effort Estimate:** 4-6 hours

**Description:**
Training session history currently only tracks high-level metrics (final loss, total samples, epochs). More detailed metrics would enable better analysis, debugging, and visualization of training progress.

**Current Behavior:**

```cpp
// Only tracks: session_id, samples_trained, epochs_completed, final_loss, final_validation_loss
TrainingSession session;
session.final_loss = final_loss;
session.final_validation_loss = final_val_loss;
```

**Desired Behavior:**

```cpp
// Track per-epoch metrics for detailed analysis
TrainingSession session;
session.per_epoch_losses = {/* loss values per epoch */};
session.per_epoch_validation_losses = {/* val loss per epoch */};
session.training_time_per_epoch = {/* time spent per epoch */};
session.learning_rates = {/* LR schedule over time */};
```

**Benefits:**

- Better debugging of training issues (e.g., detect overfitting early)
- Visualization of training progress over time
- Analysis of learning rate schedules effectiveness
- Identify optimal stopping points for future training

**Implementation Tasks:**

- [ ] Extend `TrainingSession` struct to include per-epoch metrics
- [ ] Modify `ChatbotTrainer` to expose per-epoch data
- [ ] Update session serialization/deserialization
- [ ] Add visualization tools/scripts for metrics
- [ ] Update documentation with new metrics

**Files to Modify:**

- `include/IncrementalTrainer.hpp` - Extend `TrainingSession` struct (5 TODOs added)
- `src/IncrementalTrainer.cpp` - Update `finalize_session()` method (line 583) (22 TODOs added)
- `src/IncrementalTrainer.cpp` - Update `save_session_history()` (10 TODOs added)
- `src/IncrementalTrainer.cpp` - Update `load_session_history()` (6 TODOs added)
- `src/IncrementalTrainer.cpp` - Update `train_incremental()` (5 TODOs added)
- `src/IncrementalTrainer.cpp` - Update `print_training_summary()` (6 TODOs added)
- `src/ChatbotTrainer.hpp` - Document existing getter methods (1 TODO added)
- `tests/test_incremental_trainer.cpp` - Add tests for new metrics

**Code Location:**

`src/IncrementalTrainer.cpp:583`

**Granular TODOs Added:** 55 specific implementation tasks across 3 files

**Related:**

- Could integrate with TensorBoard or similar visualization tools
- Foundation for early stopping implementation

---

### TD-005: Checkpoint Management and Symbolic Links

**Priority:** MEDIUM  
**Status:** Planned  
**Component:** Training / IncrementalTrainer  
**Created:** February 17, 2026  
**Effort Estimate:** 2-3 hours

**Description:**
Checkpoint files are currently saved with session-specific names in the training_sessions directory. There's no easy way to access the "latest" or "best" checkpoint without parsing session history. Adding symbolic links would improve usability.

**Current Behavior:**

```bash
training_sessions/
├── session_0_checkpoint.bin
├── session_1_checkpoint.bin
├── session_2_checkpoint.bin
└── session_history.txt  # Must parse to find latest
```

**Desired Behavior:**

```bash
training_sessions/
├── session_0_checkpoint.bin
├── session_1_checkpoint.bin
├── session_2_checkpoint.bin
└── session_history.txt

# In root directory:
latest_checkpoint.bin -> training_sessions/session_2_checkpoint.bin
best_checkpoint.bin -> training_sessions/session_1_checkpoint.bin
```

**Benefits:**

- Easier integration with deployment scripts
- Quick access to latest model without parsing
- Support for "best model" tracking (lowest validation loss)
- Simpler command-line usage

**Implementation Tasks:**

- [ ] Create symbolic link after each checkpoint save
- [ ] Update `latest_checkpoint.bin` symlink to newest session
- [ ] Track best validation loss and update `best_checkpoint.bin` symlink
- [ ] Add configuration option to enable/disable symlinks
- [ ] Handle symlink cleanup during old session removal
- [ ] Add Windows-compatible fallback (copy instead of symlink)

**Files to Modify:**

- `src/IncrementalTrainer.cpp` - Update `finalize_session()` (line 584) (10 TODOs added)
- `src/IncrementalTrainer.cpp` - Modify `save_model()` method (4 TODOs added)
- `src/IncrementalTrainer.cpp` - Update `cleanup_old_sessions()` (3 TODOs added)
- `include/IncrementalTrainer.hpp` - Add symlink management methods (6 TODOs added)

**Code Location:**

`src/IncrementalTrainer.cpp:584`

**Granular TODOs Added:** 23 specific implementation tasks across 2 files

**Platform Considerations:****

- Use `std::filesystem::create_symlink()` on Unix/Linux
- Use file copy or junction points on Windows
- Add error handling for insufficient permissions

---

### TD-006: Fill-in-the-Middle (FIM) Training Data Generation

**Priority:** LOW  
**Status:** Planned  
**Component:** Training / Data Generation  
**Created:** February 17, 2026  
**Effort Estimate:** 6-8 hours

**Description:**
Current training data for Project Gutenberg books uses consecutive sentence pairs (question → answer). Adding Fill-in-the-Middle (FIM) style training where the model predicts a middle sentence given first and last sentences would improve narrative understanding and coherence.

**Current Behavior:**

```cpp
// Simple consecutive pairs
INPUT: "What does this mean: [sentence 1]"
RESPONSE: "[sentence 2]"
```

**Desired Behavior:**

```cpp
// Add ~20% FIM-style pairs
INPUT: "Fill in the middle: <|first|>[sentence 1]<|last|>[sentence 3]"
RESPONSE: "[sentence 2]"
```

**Benefits:**

- Better understanding of narrative flow and context
- Improved coherence in generated responses
- Enhanced ability to reason about story structure
- More robust to partial context scenarios

**Implementation Tasks:**

- [ ] Add FIM data generation to `create_qa_pairs_from_text()`
- [ ] Define special tokens for FIM format (`<|first|>`, `<|middle|>`, `<|last|>`)
- [ ] Add configuration for FIM percentage (default: 20%)
- [ ] Ensure balanced distribution of regular and FIM pairs
- [ ] Add FIM-specific evaluation metrics
- [ ] Update documentation with FIM training approach
- [ ] Test on various text sources

**Files to Modify:**

- `src/IncrementalTrainer.cpp` - Modify `create_qa_pairs_from_text()` (line 920)
- `src/BPETokenizer.cpp` - Add FIM special tokens to vocabulary
- `include/IncrementalTrainer.hpp` - Add FIM configuration options

**Code Location:**

`src/IncrementalTrainer.cpp:920`

**References:**

- FIM approach used successfully in code completion models (Copilot, CodeGen)
- Paper: "Efficient Training of Language Models to Fill in the Middle" (Bavarian et al.)

**Evaluation:**

- Test on narrative coherence tasks
- Measure improvement in multi-turn conversation quality
- Compare perplexity on FIM vs standard test sets

---

### TD-007: Matrix Operations SIMD Acceleration

**Priority:** MEDIUM  
**Status:** Planned  
**Component:** Performance / Matrix Operations  
**Created:** February 17, 2026  
**Effort Estimate:** 12-16 hours

**Description:**
Matrix operations are performance-critical for neural network training and inference. Current implementation uses OpenMP for parallelization but lacks explicit SIMD intrinsics and BLAS library integration. Adding vectorized operations could provide 2-5x speedup for compute-intensive operations.

**Current Behavior:**

```cpp
// Naive scalar operations with OpenMP parallelization
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < other.cols; j++) {
        float sum = 0.0f;
        for (int k = 0; k < cols; k++) {
            sum += data[i][k] * other.data[k][j];  // Scalar multiply-add
        }
        result.data[i][j] = sum;
    }
}
```

**Desired Behavior:**

```cpp
// Use BLAS for large matrices
if (rows > 256 && cols > 256) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, other.cols, cols, 1.0f,
                data_ptr, cols, other.data_ptr, other.cols,
                0.0f, result_ptr, other.cols);
} else {
    // Use SIMD intrinsics for smaller matrices
    __m256 a_vec = _mm256_load_ps(&data[i][k]);
    __m256 b_vec = _mm256_load_ps(&other.data[k][j]);
    sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);  // 8-way FMA
}
```

**Benefits:**

- 2-5x performance improvement for matrix operations
- Reduced training time for large models
- Better CPU utilization with vectorized operations
- Industry-standard BLAS library for proven optimizations
- Platform-specific optimizations (x86 AVX, ARM NEON)

**Implementation Tasks:**

- [ ] Add BLAS library detection and linking (OpenBLAS, Intel MKL, Apple Accelerate)
- [ ] Implement BLAS integration for matrix multiplication (GEMM) on matrices >256x256
- [ ] Add AVX2/AVX-512 intrinsics for x86-64 platforms
  - [ ] Matrix multiplication with `_mm256_fmadd_ps`
  - [ ] Element-wise operations (`_mm256_add_ps`, `_mm256_sub_ps`, `_mm256_mul_ps`)
  - [ ] Horizontal sum reduction with `_mm256_reduce_add_ps`
- [ ] Add ARM NEON intrinsics for ARM64 platforms
  - [ ] Matrix multiplication with `vfmaq_f32`
  - [ ] Element-wise operations (`vaddq_f32`, `vsubq_f32`, `vmulq_f32`)
  - [ ] Horizontal sum with `vaddvq_f32`
- [ ] Implement cache-blocking/tiling for better L1/L2 cache utilization
- [ ] Add runtime CPU feature detection (cpuid/getauxval)
- [ ] Create optimized code paths for common matrix sizes
- [ ] Add SIMD-specific unit tests
- [ ] Benchmark and validate performance improvements
- [ ] Update documentation with build requirements
- [ ] Add CMake options for enabling/disabling SIMD features

**Files to Modify:**

- `src/Matrix.cpp` - Add SIMD implementations for all operations (TODOs already added)
- `src/Matrix.hpp` - Add conditional compilation headers for SIMD
- `cmake/FindBLAS.cmake` - Add BLAS library detection
- `CMakeLists.txt` - Add BLAS linking and SIMD compile flags
- `include/MatrixSIMD.hpp` - New header for SIMD helper functions
- `tests/matrix_simd_test.cpp` - New test suite for SIMD operations
- `docs/guides/building.md` - Document BLAS and SIMD build options

**Code Locations:**

Multiple TODOs added throughout `src/Matrix.cpp`:

- Line ~48: Matrix multiplication (operator*)
- Line ~93: Matrix addition (operator+)
- Line ~125: Matrix subtraction (operator-)
- Line ~204: Hadamard product
- Line ~231: Gradient application
- Line ~281: Sum operation

**Technical Considerations:**

- **Platform Support:** Need separate code paths for x86 (AVX/SSE) and ARM (NEON)
- **Alignment:** SIMD operations require 16/32-byte aligned memory
- **Fallback:** Maintain scalar fallback for unsupported platforms
- **Cache Performance:** Implement matrix tiling for large matrices
- **BLAS Library:** Support multiple BLAS implementations (OpenBLAS default)

**Performance Targets:**

- Matrix multiplication (1024x1024): 4-5x speedup with BLAS
- Element-wise operations: 2-3x speedup with SIMD
- Small matrices (64x64): 1.5-2x speedup with intrinsics
- Training throughput: 15-20% overall improvement

**Dependencies:**

- BLAS library (OpenBLAS recommended, Intel MKL optional)
- C++ compiler with AVX2 support (GCC 4.9+, Clang 3.4+, MSVC 2015+)
- CMake 3.10+ for BLAS detection

**References:**

- Intel Intrinsics Guide: <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/>
- ARM NEON Intrinsics: <https://developer.arm.com/architectures/instruction-sets/simd-isas/neon>
- OpenBLAS: <https://www.openblas.net/>
- BLAS API Reference: <http://www.netlib.org/blas/>

**Related:**

- TD-003: GPU Memory Management (complementary GPU optimization)
- Existing OpenMP parallelization can coexist with SIMD

---

## Resolved Items

### TD-002: Improve Error Handling in BPE Tokenizer

**Resolution Date:** January 28, 2026  
**Component:** NLP / Tokenization  
**Resolved By:** Comprehensive error handling implementation

**Summary:**
Implemented robust error handling and validation for the BPE tokenizer, including custom exception types, UTF-8 validation, input validation, and vocabulary file format validation.

**Changes Made:**

1. ✅ Created custom exception types:
   - `TokenizerInputError` - for empty/invalid input
   - `TokenizerEncodingError` - for UTF-8 encoding issues
   - `VocabularyFileError` - for malformed vocabulary files
   - `TokenIDError` - for out-of-range token IDs

2. ✅ Implemented UTF-8 validation:
   - `is_valid_utf8()` helper method validates character sequences
   - Detects invalid start bytes, incomplete sequences, and malformed continuation bytes
   - Applied to `encode()` and `pre_tokenize()` methods

3. ✅ Added input validation:
   - `validate_input()` helper checks for empty strings and UTF-8 validity
   - `encode()` validates non-empty input with proper UTF-8
   - `decode()` validates non-empty token ID vector and checks for negative IDs

4. ✅ Enhanced vocabulary file validation:
   - Validates filename is not empty
   - Throws descriptive exceptions for file not found
   - Validates special tokens section format and integer parsing
   - Validates vocabulary entries with tab separators and non-negative IDs
   - Validates BPE merges format and non-empty tokens
   - Ensures loaded vocabulary contains required special tokens

5. ✅ Created comprehensive test suite:
   - 27 tests covering all error conditions
   - Tests for exception types, messages, and inheritance
   - Edge case coverage (empty input, invalid UTF-8, malformed files)
   - All tests passing

**Files Modified:**

- `src/BPETokenizer.hpp` - Added custom exception types and validation methods
- `src/BPETokenizer.cpp` - Implemented validation throughout encode/decode/load_vocab
- `tests/tokenizer_error_handling_test.cpp` - New comprehensive test suite (27 tests)
- `tests/CMakeLists.txt` - Added tokenizerErrorHandlingTests target

**Verification:**
All 27 tests pass, validating:

- Empty input detection
- UTF-8 validation for various invalid sequences
- Token ID range checking
- Vocabulary file format validation
- Exception type hierarchy
- Descriptive error messages

---

### TD-001: Complete Optimizer Parameter Exposure

**Resolution Date:** January 28, 2026  
**Component:** Optimizer Integration  
**Resolved By:** Complete parameter registration implementation

**Summary:**
Successfully completed parameter exposure for all model components. The optimizer now fully manages all model parameters through the centralized `optimizer->step()` mechanism.

**Changes Made:**

1. ✅ Verified `LLMEncoder::register_parameters_with_optimizer()` properly exposes all parameters (token embedding, encoder blocks, final norm)
2. ✅ Verified `LLMDecoder::register_parameters_with_optimizer()` properly exposes all parameters (token embedding, decoder blocks, final norm)
3. ✅ Verified `LanguageModelHead::set_optimizer()` properly exposes all parameters (W_output, bias)
4. ✅ Updated `ChatbotTrainer` to use `optimizer->step()` instead of `model->update_weights()`
5. ✅ Removed obsolete TODO comments
6. ✅ Tested training with optimizer integration - works correctly

**Files Modified:**

- `src/ChatbotTrainer.cpp` - Replaced `model->update_weights()` with `optimizer->step()`, removed outdated comments

**Verification:**
Training runs successfully with AdamW optimizer using centralized parameter management. All parameter groups are properly registered and updated through `optimizer->step()`.

---

---

## Future Improvements

These are lower-priority enhancements that don't currently block development:

### Performance Optimizations

1. **Matrix Operations SIMD Acceleration**
   - **Status:** Promoted to Active Technical Debt
   - **See:** [TD-007: Matrix Operations SIMD Acceleration](#td-007-matrix-operations-simd-acceleration)

2. **Memory Pool for Matrix Allocations**
   - Reduce allocation overhead during forward/backward passes
   - Pre-allocate memory for common matrix sizes
   - Reduce memory fragmentation

### Code Quality

1. **Increase Test Coverage**
   - Current coverage: ~85%
   - Target: 95%+
   - Focus on edge cases and error paths

2. **Add Benchmarking Suite**
   - Performance regression testing
   - Track training throughput over time
   - Compare against baseline implementations

### Developer Experience

1. **Add Python Bindings**
   - Enable easier experimentation
   - Broader community adoption
   - Integration with Python ML ecosystem

2. **Improve Build Times**
   - Use precompiled headers
   - Optimize template instantiations
   - Modularize includes

---

## Process Guidelines

### Adding New Technical Debt

When adding a new technical debt item:

1. **Create an entry in this document** with all required fields:
   - Unique ID (TD-XXX)
   - Priority (High/Medium/Low)
   - Component
   - Effort estimate
   - Description and impact
   - Location in code
   - Task checklist
   - Files affected

2. **Create a GitHub issue** using the technical debt template:
   - Link to this document
   - Use label: `technical-debt`
   - Assign priority label
   - Add to project board

3. **Update code comments** to reference the tracking item:

   ```cpp
   // See TD-001 in TECHNICAL_DEBT.md - Parameter exposure incomplete
   ```

4. **Remove untracked TODOs**

- All TODOs must be tracked here or in GitHub issues

### Prioritization Criteria

**High Priority:**

- Blocks new feature development
- Causes bugs or incorrect behavior
- Security or stability issues
- Affects multiple components

**Medium Priority:**

- Improves code maintainability significantly
- Reduces technical complexity
- Enables future features
- Clear path to resolution

**Low Priority:**

- Nice-to-have improvements
- Cosmetic code cleanup
- Performance optimizations (non-critical)
- Developer convenience features

### Resolving Technical Debt

When resolving a debt item:

1. Complete all tasks in the checklist
2. Add tests to prevent regression
3. Update documentation
4. Move item from "Active" to "Resolved" section with resolution date
5. Close related GitHub issue
6. Remove code comments referencing the item

---

## Statistics

### By Priority

| Priority | Count | Percentage |
| ---------- | ------- | ------------ |
| High | 0 | 0% |
| Medium | 3 | 60% |
| Low | 2 | 40% |

### By Component

| Component | Count |
| ---------------------- | ------- |
| Training / IncrementalTrainer | 2 |
| Performance / Matrix Operations | 1 |
| Training / Data Generation | 1 |
| GPU / Performance | 1 |

### Effort Distribution

| Effort Range | Count |
| -------------- | ------- |
| 0-2 hours | 0 |
| 2-4 hours | 2 |
| 4-8 hours | 2 |
| 8+ hours | 1 |

**Total Estimated Effort:** 28-36 hours

---

## References

- [Process Improvement Plan](PROCESS_IMPROVEMENT_PLAN.md) - Section 10: Technical Debt Items
- [Contributing Guide](docs/guides/contributing.md) - Code quality standards
- [GitHub Issues](https://github.com/yourusername/adai/issues?q=is%3Aissue+label%3Atechnical-debt) - Active debt tracking

---

**Maintenance Note:** This document should be reviewed monthly and updated as items are added or resolved.
