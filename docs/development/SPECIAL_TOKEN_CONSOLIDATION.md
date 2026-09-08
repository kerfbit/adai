# Special Token Consolidation Summary

## Overview

All special token handling has been consolidated into a single header-only source file: **`src/SpecialTokens.hpp`**

## What Was Done

### 1. Analysis

Searched the entire codebase for special token usage across:

- **BPETokenizer** (definitions and encoding/decoding)
- **EncoderDecoderModel** (model initialization and generation)
- **TextGenerator** (generation configuration and stop conditions)
- **BatchProcessor, Dataset, ParallelDataLoader** (padding operations)
- **PipelineInferenceEngine, IntegratedInferenceEngine** (inference pipelines)
- **SpeculativeDecoding** (advanced generation)

### 2. Created Consolidated Header

Created `src/SpecialTokens.hpp` containing:

#### Constants

```cpp
namespace adai::SpecialTokenIDs {
    constexpr int PAD = 0;  // Padding token
    constexpr int UNK = 1;  // Unknown token
    constexpr int BOS = 2;  // Beginning of sequence
    constexpr int EOS = 3;  // End of sequence
}

namespace adai::SpecialTokenStrings {
    constexpr const char* PAD = "<pad>";
    constexpr const char* UNK = "<unk>";
    constexpr const char* BOS = "<bos>";
    constexpr const char* EOS = "<eos>";
}
```

#### Configuration Structure

```cpp
adai::SpecialTokenConfig config;  // Default IDs
config.get_bos_token_id();  // Returns 2
config.validate();  // Ensures IDs are valid and unique
```

#### Utility Functions

- `is_special_token(token_id, config)` - Check if token is special
- `is_special_token_string(token_str)` - Check if string is special token
- `is_stop_token(token_id, config)` - Check if token stops generation (EOS/PAD)
- `get_special_token_string(token_id, config)` - Convert ID to string
- `get_special_token_id(token_str, config)` - Convert string to ID
- `create_special_token_set()` - Create set of special tokens
- `create_special_token_map(config)` - Create string→ID map
- `create_inverse_special_token_map(config)` - Create ID→string map

### 3. Testing

Comprehensive test suite in `tests/specialtokens_test.cpp` (registered as `specialtokensTests` /
`SpecialTokensTests` in `tests/CMakeLists.txt` — runs under `ctest` like every other test in the
suite) covering:

- ✅ Constant definitions
- ✅ Configuration struct
- ✅ Validation logic
- ✅ All utility functions
- ✅ Map creation functions
- ✅ Custom configurations

All tests pass successfully!

## Usage Examples

### Basic Usage

```cpp
#include "SpecialTokens.hpp"
using namespace adai;

// Use constants directly
int bos_id = SpecialTokenIDs::BOS;  // 2

// Create configuration
SpecialTokenConfig config;

// Check special tokens
if (is_special_token(token_id, config)) {
    // Handle special token
}

// Check stop conditions in generation
if (is_stop_token(next_token, config)) {
    break;  // Stop generation
}
```

### Integration with Existing Code

```cpp
// In tokenizer initialization
auto special_map = create_special_token_map();
for (const auto& [token_str, token_id] : special_map) {
    vocab[token_str] = token_id;
}

// In decoder skip logic
if (skip_special_tokens && is_special_token(id, config)) {
    continue;
}

// In generation loop
std::string token_name = get_special_token_string(token_id, config);
```

### Custom Configurations

```cpp
// Create custom token IDs (for compatibility with different models)
SpecialTokenConfig custom(10, 11, 12, 13);
custom.validate();  // Ensure IDs are valid

// Use with utility functions
bool is_special = is_special_token(12, custom);  // true (custom BOS)
```

## Benefits

1. **Single Source of Truth**: All special token definitions in one place
2. **Type Safety**: Compile-time constants prevent runtime errors
3. **Consistency**: Ensures all components use the same token IDs
4. **Documentation**: Comprehensive inline documentation and usage examples
5. **Maintainability**: Easy to update and extend
6. **Header-Only**: No compilation dependencies, easy to integrate
7. **Namespace Protection**: Avoids naming conflicts with `adai::` namespace

## Files Created

1. **`src/SpecialTokens.hpp`** - Main header-only library (380 lines)
2. **`tests/specialtokens_test.cpp`** - Comprehensive GTest suite

## Locations of Special Token Usage

Special tokens are currently used in:

- `src/BPETokenizer.{hpp,cpp}` - Token definitions and encoding/decoding
- `src/EncoderDecoderModel.{hpp,cpp}` - Model configuration
- `src/TextGenerator.{hpp,cpp}` - Generation configuration and termination
- `src/BatchProcessor.hpp` - Padding operations
- `src/Dataset.hpp` - Batch creation
- `src/ParallelDataLoader.hpp` - Data loading and padding
- `src/PipelineInferenceEngine.hpp` - Inference engine
- `src/IntegratedInferenceEngine.hpp` - Integrated inference
- `src/SpeculativeDecoding.hpp` - Advanced generation
- `src/ChatbotAPI.cpp` - API endpoints
- `src/EfficientBatching.hpp` - Batching operations

## Migration Path (Optional)

To migrate existing code to use the new header:

1. Add `#include "SpecialTokens.hpp"` to files
2. Replace hardcoded token IDs with `adai::SpecialTokenIDs::*`
3. Replace special token checks with utility functions
4. Use `SpecialTokenConfig` for configuration passing

**Note**: Migration is optional - the new header can be used alongside existing code.

## Verification

Run the test suite:

```bash
cd build/debug && ctest -R SpecialTokensTests --output-on-failure
```

## Standard Token Definitions

These token IDs are standardized across the ADAI project:

|Token|ID|String|Purpose|
|---------|----|---------|---------------------------------|
|`PAD`|0|`<pad>`|Padding sequences to uniform length|
|`UNK`|1|`<unk>`|Unknown/out-of-vocabulary tokens|
|`BOS`|2|`<bos>`|Beginning of sequence marker|
|`EOS`|3|`<eos>`|End of sequence & stop generation|

**⚠️ Important**: These IDs must not be changed without retraining all models and regenerating all vocabulary files.
