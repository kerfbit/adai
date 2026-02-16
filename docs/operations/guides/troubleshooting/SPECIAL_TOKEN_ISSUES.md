# Special Token Usage Analysis

## Summary

**Status**: ❌ **CRITICAL ISSUES FOUND**

The chatbot has **multiple critical issues** with special token handling that will cause incorrect model behavior:

1. Wrong default special token IDs in model initialization
2. Special token IDs not synchronized between tokenizer and model
3. Inconsistent encoding behavior causing double BOS tokens
4. Mismatch between training and inference token handling

---

## Special Token Definitions

According to `BPETokenizer.hpp` and `BPETokenizer.cpp`:

| Token   | ID | Purpose                     |
| ------- | -- | --------------------------- |
| `<pad>` | 0  | Padding sequences           |
| `<unk>` | 1  | Unknown/out-of-vocab tokens |
| `<bos>` | 2  | Beginning of sequence       |
| `<eos>` | 3  | End of sequence             |

---

## Critical Issues

### 🔴 Issue 1: Wrong Default Special Token IDs

**File**: `src/EncoderDecoderModel.cpp` (Lines 37-38)

```cpp
TextGenerator::GenerationConfig gen_config;
gen_config.max_length = max_seq_length;
gen_config.bos_token_id = 1;  // ❌ WRONG! Should be 2
gen_config.eos_token_id = 2;  // ❌ WRONG! Should be 3
gen_config.pad_token_id = 0;  // ✅ Correct
```

**Problem**: The model initializes with:

- `bos_token_id = 1` (which is actually `<unk>` token!)
- `eos_token_id = 2` (which is actually `<bos>` token!)

**Impact**:

- Generation starts with `<unk>` instead of `<bos>`
- Generation stops at `<bos>` instead of `<eos>`
- Model produces nonsensical output

**Fix**:

```cpp
gen_config.bos_token_id = 2;  // <bos>
gen_config.eos_token_id = 3;  // <eos>
gen_config.pad_token_id = 0;  // <pad>
```

---

### 🔴 Issue 2: set_tokenizer() Doesn't Sync Special Token IDs

**File**: `src/EncoderDecoderModel.cpp` (Line 368-370)

```cpp
void EncoderDecoderModel::set_tokenizer(BPETokenizer* tokenizer_ptr) {
    tokenizer.reset(tokenizer_ptr);
    // ❌ Missing: Update bos_token_id, eos_token_id, pad_token_id from tokenizer!
}
```

**Problem**: When `ChatbotCLI` transfers the tokenizer to the model:

```cpp
// ChatbotCLI.cpp line 52
model->set_tokenizer(tokenizer.release());
```

The model receives a tokenizer with correct IDs (0,1,2,3) but continues using its own wrong IDs (0,1,1,2).

**Impact**:

- Tokenizer has: bos=2, eos=3
- Model uses: bos=1, eos=2
- Mismatch causes generation to fail

**Fix**:

```cpp
void EncoderDecoderModel::set_tokenizer(BPETokenizer* tokenizer_ptr) {
    tokenizer.reset(tokenizer_ptr);
    
    // Sync special token IDs from tokenizer
    TextGenerator::GenerationConfig config = generator->get_config();
    config.bos_token_id = tokenizer->get_bos_token_id();  // Needs getter method!
    config.eos_token_id = tokenizer->get_eos_token_id();
    config.pad_token_id = tokenizer->get_pad_token_id();
    set_generation_config(config);
}
```

**Note**: This requires adding getter methods to `BPETokenizer`:

```cpp
int get_bos_token_id() const { return bos_token_id; }
int get_eos_token_id() const { return eos_token_id; }
int get_pad_token_id() const { return pad_token_id; }
```

---

### 🟡 Issue 3: Double BOS Token in Generation

**File**: `src/EncoderDecoderModel.cpp` (Lines 133, 187, 251)

```cpp
// Line 133 - generate_response()
std::vector<int> input_tokens = tokenizer->encode(input_text);  
// ❌ Defaults to add_special_tokens=true → adds [BOS, ...tokens..., EOS]

// Line 169
std::vector<int> output_tokens = 
    generator->generate(model_fn, {bos_token_id});  
// ❌ Starts decoder with BOS again!
```

**Problem**: The sequence becomes:

1. Encoder input: `[2, ...tokens..., 3]` (BOS, tokens, EOS)
2. Decoder starts with: `[2]` (BOS again)

This creates an inconsistent pattern where the encoder sees different input than what the decoder expects.

**Best Practice** (from documentation):

```cpp
// Training: Don't add BOS/EOS to encoder input
auto train_ids = tokenizer.encode(text, false);

// Generation: Encoder gets plain tokens, decoder starts with BOS
auto input_ids = tokenizer.encode(prompt, false);  // No special tokens for encoder
auto generated = model.generate(input_ids);  // Decoder adds BOS internally
```

**Current Code**:

```cpp
// Adds BOS/EOS to encoder input
auto input_ids = tokenizer.encode(prompt);  // Defaults to true
auto generated = model.generate(input_ids);  // Decoder also adds BOS
```

**Recommended Fix Option 1** (Encoder without special tokens):

```cpp
// EncoderDecoderModel.cpp - All encode() calls
std::vector<int> input_tokens = tokenizer->encode(input_text, false);  // ✅ No BOS/EOS
```

**Recommended Fix Option 2** (Don't add BOS in decoder if already in input):
Keep encoding with special tokens, but check if first token is already BOS before adding it.

---

### 🟡 Issue 4: Training vs Inference Inconsistency

**File**: `src/ChatbotTrainer.cpp` (Lines 310-311, 319-320)

```cpp
std::vector<int> input_tokens = tokenizer->encode(pair.input);
std::vector<int> target_tokens = tokenizer->encode(pair.response);
// ❌ Defaults to add_special_tokens=true
```

**File**: `src/EncoderDecoderModel.cpp` (Line 251)

```cpp
std::vector<int> input_tokens = tokenizer->encode(input_text);
std::vector<int> target_tokens = tokenizer->encode(target_text);
// ❌ Also defaults to true
```

**Problem**: Training data is encoded with BOS/EOS, which may not match the inference pattern.

**Best Practice**:

- **Encoder input**: No special tokens `encode(text, false)`
- **Decoder target**: With special tokens `encode(text, true)` for teacher forcing

**Recommendation**: Make encoding explicit and consistent:

```cpp
// For training
std::vector<int> input_tokens = tokenizer->encode(pair.input, false);   // Encoder
std::vector<int> target_tokens = tokenizer->encode(pair.response, true); // Decoder target
```

---

## Recommendations

### Immediate Fixes (High Priority)

1. **Fix default special token IDs** in `EncoderDecoderModel` constructor
2. **Add getter methods** to `BPETokenizer` for special token IDs
3. **Update set_tokenizer()** to synchronize special token IDs
4. **Make encoding explicit** - always specify `add_special_tokens` parameter

### Code Changes Required

#### 1. `src/BPETokenizer.hpp` - Add Getters

```cpp
// Add to public section
int get_bos_token_id() const { return bos_token_id; }
int get_eos_token_id() const { return eos_token_id; }
int get_pad_token_id() const { return pad_token_id; }
int get_unk_token_id() const { return unk_token_id; }
```

#### 2. `src/EncoderDecoderModel.cpp` - Fix Constructor

```cpp
// Line 37-39
gen_config.bos_token_id = 2;  // Correct <bos>
gen_config.eos_token_id = 3;  // Correct <eos>
gen_config.pad_token_id = 0;  // <pad>
```

#### 3. `src/EncoderDecoderModel.cpp` - Fix set_tokenizer()

```cpp
void EncoderDecoderModel::set_tokenizer(BPETokenizer* tokenizer_ptr) {
    tokenizer.reset(tokenizer_ptr);
    
    // Synchronize special token IDs
    TextGenerator::GenerationConfig config = generator->get_config();
    config.bos_token_id = tokenizer->get_bos_token_id();
    config.eos_token_id = tokenizer->get_eos_token_id();
    config.pad_token_id = tokenizer->get_pad_token_id();
    set_generation_config(config);
}
```

#### 4. `src/EncoderDecoderModel.cpp` - Fix Encoding Calls

```cpp
// All encode() calls for encoder input
std::vector<int> input_tokens = tokenizer->encode(input_text, false);  // No special tokens
```

#### 5. `src/ChatbotTrainer.cpp` - Fix Training Encoding

```cpp
// Line 310-311, 319-320
std::vector<int> input_tokens = tokenizer->encode(pair.input, false);    // Encoder
std::vector<int> target_tokens = tokenizer->encode(pair.response, true); // Decoder
```

---

## Testing After Fixes

1. **Verify special token IDs match**:

   ```cpp
   BPETokenizer tokenizer;
   tokenizer.load_vocab("vocab.txt");
   
   assert(tokenizer.get_bos_token_id() == 2);
   assert(tokenizer.get_eos_token_id() == 3);
   
   model->set_tokenizer(&tokenizer);
   assert(model->get_bos_token_id() == 2);
   assert(model->get_eos_token_id() == 3);
   ```

2. **Verify encoding behavior**:

   ```cpp
   auto tokens_with = tokenizer.encode("hello", true);
   // Should be: [2, ...tokens..., 3]
   
   auto tokens_without = tokenizer.encode("hello", false);
   // Should be: [...tokens...] (no BOS/EOS)
   ```

3. **Verify generation doesn't have double BOS**:

   - Check generated token sequences don't start with `[2, 2, ...]`

4. **Retrain model** after fixes to ensure consistency

---

## References

- `src/BPETokenizer.hpp` - Special token initialization
- `docs/api/nlp/tokenizer.md` - Best practices for special token usage
- `tests/tokenizer_test.cpp` - Expected behavior examples
- Test case `SpecialTokensInitialized` confirms BOS=2, EOS=3
