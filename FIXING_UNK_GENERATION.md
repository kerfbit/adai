# Fixing Unknown Token Generation After Vocabulary Repair

## Problem Description

After fixing vocabulary file and retraining, responses contain a block of `<unk>` (unknown) tokens at the beginning.

## Root Cause Analysis

### Why This Happens

1. **Model Generates UNK Token IDs**: The model is genuinely generating the `unk_token_id` (typically ID 1) multiple times at the start of responses

2. **Possible Causes**:
   - **Vocabulary mismatch**: Using a model trained with old vocabulary but new vocab file
   - **Incomplete training**: Model hasn't learned the new vocabulary mappings yet
   - **Wrong checkpoint**: Loading old checkpoint instead of newly trained model
   - **Poor training coverage**: Training data doesn't use vocabulary tokens effectively

## Solution Steps

### Step 1: Verify Vocabulary is Fixed

```bash
# Run diagnostic tool
python3 diagnose_generation.py vocab.txt

# Should show:
# ✓ All special tokens present
# ✓ No duplicate token IDs
```

### Step 2: Recompile Code

**CRITICAL**: After changing BPETokenizer.cpp, you MUST recompile:

```bash
cd /home/rodney/Repos/adai

# Clean build
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)

# Or use build script if available
./build.sh
```

### Step 3: Remove Old Model Files

```bash
# Delete ALL old checkpoints to avoid accidentally loading them
rm -f chatbot_model.bin*
rm -f *.epoch*
```

### Step 4: Retrain from Scratch

**Do NOT resume from old checkpoints** - train completely fresh:

```bash
./build/bin/chatbot_trainer \
    --data sample_training_data.txt \
    --vocab vocab.txt \
    --output chatbot_model_new.bin \
    --epochs 20 \
    --learning-rate 0.001 \
    --batch-size 32
```

**Important Training Settings**:
- Use at least **20 epochs** (was probably 10 before)
- Ensure training data has good variety
- Monitor validation loss - should decrease steadily

### Step 5: Test the New Model

```bash
# Test with simple inputs first
./build/bin/chatbot vocab.txt chatbot_model_new.bin

# Try simple queries:
Input: Hello
Input: How are you?
Input: What is your name?
```

## Code Fixes Applied

### Fix 1: Ensure Special Tokens Set is Populated

**File**: `src/BPETokenizer.cpp` (line ~577)

```cpp
// After vocabulary validation, ensure special_tokens set is complete
if (vocab.find("<pad>") != vocab.end()) special_tokens.insert("<pad>");
if (vocab.find("<unk>") != vocab.end()) special_tokens.insert("<unk>");
if (vocab.find("<bos>") != vocab.end()) special_tokens.insert("<bos>");
if (vocab.find("<eos>") != vocab.end()) special_tokens.insert("<eos>");
```

**Why**: Ensures special tokens are always in the `special_tokens` set, even if the rebuild logic during vocab loading had edge cases.

### Fix 2: Explicit Skip Special Tokens in Decode

**File**: `src/EncoderDecoderModel.cpp` (lines ~173 and ~242)

```cpp
// OLD:
std::string response = tokenizer->decode(output_tokens);

// NEW:
std::string response = tokenizer->decode(output_tokens, true);  // true = skip special tokens
```

**Why**: Explicitly pass `skip_special_tokens=true` to ensure `<bos>`, `<eos>`, `<unk>`, and `<pad>` tokens are stripped from output.

## Understanding Token Generation

### Normal Generation Flow

```
1. Input: "Hello"
2. Encode: [2, 543, 234, 3]  // <bos>, "hel", "lo", <eos>
3. Encoder processes input
4. Decoder generates: [2, 123, 456, 789, 3]  // <bos>, "Hi", "there", "!", <eos>
5. Decode with skip_special_tokens=true: "Hi there!"
```

### Problem Flow (UNK generation)

```
1. Input: "Hello"
2. Encode: [2, 543, 234, 3]
3. Encoder processes input
4. Decoder generates: [2, 1, 1, 1, 1, 1, ...]  // <bos>, <unk>, <unk>, <unk>, ...
5. Decode with skip_special_tokens=true: "" (empty - all tokens skipped!)
6. If skip_special_tokens=false: "<bos><unk><unk><unk><unk>..."
```

### Why Model Generates UNK

1. **Untrained embeddings**: Model hasn't learned what tokens to generate
2. **Vocabulary shift**: Token IDs changed during vocabulary repair
3. **Insufficient training**: Need more epochs to learn patterns
4. **Poor initialization**: Random weights don't know vocabulary

## Diagnostic Commands

### Check Vocabulary Stats

```bash
# Count total tokens
grep -c -v "^#" vocab.txt

# Verify special tokens
grep "^<.*>" vocab.txt

# Check for duplicates (should be 0)
python3 test_repair_vocab.py vocab.txt --test-only
```

### Check Model/Vocab Compatibility

```bash
# List all model files
ls -lh chatbot_model*.bin* vocab.txt

# Check timestamp - model should be AFTER vocab
stat -c '%y %n' vocab.txt chatbot_model.bin

# Verify model config includes correct vocab size
# (This is automatic during load - will fail if mismatched)
```

### Monitor Training Progress

```bash
# Run training with verbose logging
./build/bin/chatbot_trainer \
    --data sample_training_data.txt \
    --vocab vocab.txt \
    --output model.bin \
    --epochs 20 \
    --log-level verbose

# Watch for:
# - Decreasing training loss
# - Decreasing validation loss  
# - Vocabulary size matches expectations
```

## Verification Checklist

After retraining, verify:

- [ ] Code was recompiled after BPETokenizer.cpp changes
- [ ] Old model checkpoints were deleted
- [ ] Training completed full number of epochs
- [ ] Training loss decreased steadily
- [ ] Validation loss decreased (not increasing)
- [ ] Test generation produces valid text (not `<unk>` tokens)
- [ ] Simple inputs like "Hello" produce reasonable responses

## Common Mistakes

### ❌ Mistake 1: Not Recompiling

```bash
# Fixed BPETokenizer.cpp but didn't recompile
./build/bin/chatbot vocab.txt model.bin  # Uses OLD code!
```

**Fix**: Always recompile after code changes

### ❌ Mistake 2: Using Old Checkpoints

```bash
# Retrained but model loaded old checkpoint
./build/bin/chatbot vocab.txt chatbot_model.bin.epoch5  # OLD!
```

**Fix**: Use freshly trained model, delete old checkpoints

### ❌ Mistake 3: Vocab/Model Mismatch

```bash
# Using vocab from before repair with model from after
./build/bin/chatbot old_vocab.txt new_model.bin  # MISMATCH!
```

**Fix**: Always use matching vocab and model files

### ❌ Mistake 4: Insufficient Training

```bash
# Only trained for 2-3 epochs
--epochs 3  # Not enough!
```

**Fix**: Use at least 20 epochs for small models

## Expected Results

### After Fixes + Retraining

```bash
$ ./build/bin/chatbot vocab.txt chatbot_model_new.bin

🤖 Initializing Chatbot...
📚 Loading tokenizer from: vocab.txt
✅ Tokenizer loaded (vocab size: 9925)
🧠 Initializing transformer model...
✅ Model initialized
📂 Loading pre-trained weights from: chatbot_model_new.bin
✅ Weights loaded successfully

You: Hello!
Bot: Hi! How can I help you today?

You: What is your name?
Bot: I am a chatbot assistant.
```

### If Still Generating UNK

If you still see `<unk>` tokens after all fixes:

1. **Check training data quality**:
   ```bash
   # Training data should have diverse vocabulary
   cat sample_training_data.txt | wc -w  # Word count
   cat sample_training_data.txt | tr ' ' '\n' | sort -u | wc -l  # Unique words
   ```

2. **Increase training epochs significantly**:
   ```bash
   --epochs 50  # More training
   ```

3. **Check learning rate**:
   ```bash
   --learning-rate 0.0005  # Lower LR, more stable training
   ```

4. **Verify model architecture**:
   ```bash
   # Model might be too small
   # Check config in ChatbotTrainer
   ```

## Summary

The issue of `<unk>` token generation after vocabulary repair happens because:

1. **Vocabulary changed**: Token IDs shifted, embeddings misaligned
2. **Model needs retraining**: Must learn new vocabulary from scratch
3. **Code needs recompiling**: C++ changes require rebuild

**Solution**: Recompile → Delete old models → Retrain from scratch → Test

After following all steps, the model should generate proper text without `<unk>` tokens.
