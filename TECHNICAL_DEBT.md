# Technical Debt Tracker

This document tracks all known technical debt items, TODOs, and improvement opportunities in the ADAI codebase. Items are prioritized and linked to GitHub issues for tracking.

## Overview

**Last Updated:** January 24, 2026  
**Total Items:** 2  
**High Priority:** 0  
**Medium Priority:** 1  
**Low Priority:** 1

## Active Technical Debt

### 1. Complete Optimizer Parameter Exposure

**ID:** TD-001  
**Priority:** Medium  
**Component:** Optimizer Integration  
**Effort:** 4-6 hours  
**Status:** Open  

**Description:**  
Complete the parameter exposure work to allow external optimizer to fully manage all model parameters. Currently, the optimizer integration is incomplete - models still call `update_weights()` internally instead of delegating to the centralized optimizer via `optimizer->step()`.

**Impact:**

- Code duplication between model-specific weight updates and optimizer
- Cannot easily switch optimizer strategies without modifying model code
- Inconsistent weight update patterns across components

**Location:**

- `src/EncoderDecoderModel.cpp:327-329` - TODO comment for parameter exposure
- `src/ChatbotTrainer.cpp:605` - Commented out `optimizer->step()` call

**Tasks:**

- [ ] Expose parameters in `LLMEncoder` via `register_parameters()` method
- [ ] Expose parameters in `LLMDecoder` via `register_parameters()` method
- [ ] Expose parameters in `LanguageModelHead` via `register_parameters()` method
- [ ] Update `ChatbotTrainer` to use `optimizer->step()` instead of `model->update_weights()`
- [ ] Add integration tests for parameter exposure
- [ ] Update documentation for optimizer integration

**Files Affected:**

- `src/EncoderDecoderModel.cpp`
- `src/ChatbotTrainer.cpp`
- `src/LLMEncoder.hpp` / `src/LLMEncoder.cpp`
- `src/Decoder.hpp` / `src/Decoder.cpp`
- `src/LanguageModelHead.hpp` / `src/LanguageModelHead.cpp`

**Related Issues:** [Create issue in GitHub]

**Notes:**

- Current implementation uses a hybrid approach - optimizer is configured but `update_weights()` is still called
- Full implementation requires exposing weight and gradient pointers from all components
- Should maintain backward compatibility during transition

---

### 2. Improve Error Handling in BPE Tokenizer

**ID:** TD-002  
**Priority:** Low  
**Component:** NLP / Tokenization  
**Effort:** 2-3 hours  
**Status:** Open  

**Description:**  
BPE tokenizer could benefit from more robust error handling and validation, particularly for edge cases like empty strings, invalid UTF-8, and malformed vocabulary files.

**Impact:**

- Potential crashes on invalid input
- Unclear error messages for debugging
- Lack of input validation

**Location:**

- `src/BPETokenizer.cpp` - encode/decode methods

**Tasks:**

- [ ] Add input validation for empty/null strings
- [ ] Add UTF-8 validation
- [ ] Improve vocabulary file format validation
- [ ] Add specific exception types for different error conditions
- [ ] Add error handling tests

**Files Affected:**

- `src/BPETokenizer.hpp`
- `src/BPETokenizer.cpp`
- `tests/test_tokenizer.cpp`

**Related Issues:** [Create issue in GitHub]

**Notes:**

- Current implementation assumes well-formed input
- Should use descriptive exceptions rather than generic errors
- Consider adding debug logging for tokenization process

---

## Resolved Items

None yet.

---

## Future Improvements

These are lower-priority enhancements that don't currently block development:

### Performance Optimizations

1. **Matrix Operations SIMD Acceleration**
   - Add SIMD/vectorization for matrix multiplication
   - Use BLAS library for large matrix operations
   - Estimated 2-5x performance improvement for training

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
|----------|-------|------------|
| High     | 0     | 0%         |
| Medium   | 1     | 50%        |
| Low      | 1     | 50%        |

### By Component

| Component            | Count |
|----------------------|-------|
| Optimizer Integration| 1     |
| NLP / Tokenization   | 1     |

### Effort Distribution

| Effort Range | Count |
|--------------|-------|
| 0-2 hours    | 0     |
| 2-4 hours    | 1     |
| 4-8 hours    | 1     |
| 8+ hours     | 0     |

**Total Estimated Effort:** 6-9 hours

---

## References

- [Process Improvement Plan](PROCESS_IMPROVEMENT_PLAN.md) - Section 10: Technical Debt Items
- [Contributing Guide](docs/guides/contributing.md) - Code quality standards
- [GitHub Issues](https://github.com/yourusername/adai/issues?q=is%3Aissue+label%3Atechnical-debt) - Active debt tracking

---

**Maintenance Note:** This document should be reviewed monthly and updated as items are added or resolved.
