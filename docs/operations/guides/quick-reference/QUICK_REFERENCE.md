# ADAI Build & Vocabulary Quick Reference

---

## QUICK START

Build Everything:

```bash
./scripts/build_and_vocab.sh build
```

Create Vocabulary (5000 tokens):

```bash
./build/bin/vocab_builder --input sample_training_data.txt \
                           --output vocab.txt \
                           --vocab-size 5000 \
                           --format pairs --stats
```

Train Model (10 epochs):

```bash
./build/bin/chatbot_trainer --data sample_training_data.txt \
                             --vocab vocab.txt \
                             --output chatbot_model.bin \
                             --epochs 10
```

Full Workflow:

```bash
./scripts/build_and_vocab.sh full
```

---

## BUILD COMMANDS

Interactive Menu:

```bash
./scripts/build_and_vocab.sh
```

Direct Commands:

```bash
./scripts/build_and_vocab.sh build           # Release mode
./scripts/build_and_vocab.sh build-debug     # Debug mode
./scripts/build_and_vocab.sh clean           # Clean build
```

Manual Build:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

---

## VOCABULARY BUILDER

Basic Usage:

```bash
./build/bin/vocab_builder --input FILE --output FILE
```

Full Options:

- `--input <file>` - Input text file (can use multiple times)
- `--output <file>` - Output vocabulary file
- `--vocab-size <N>` - Target vocabulary size (default: 5000)
- `--threshold <N>` - Min character frequency (default: 1)
- `--format <type>` - plain |pairs| json (default: plain)
- `--stats` - Show statistics
- `--help` - Show help

Input Formats:

- **plain** - One sentence per line
- **pairs** - INPUT: ... / RESPONSE: ...
- **json** - ["text1", "text2", ...]

Examples:

Small vocab for testing:

```bash
./build/bin/vocab_builder --input data.txt --output vocab.txt \
                           --vocab-size 1000
```

From chatbot training data:

```bash
./build/bin/vocab_builder --input training.txt --output vocab.txt \
                           --format pairs --vocab-size 8000 --stats
```

From multiple files:

```bash
./build/bin/vocab_builder --input file1.txt --input file2.txt \
                           --output vocab.txt --vocab-size 15000
```

---

## CHATBOT TRAINER

Basic Usage:

```bash
./build/bin/chatbot_trainer --data FILE --vocab FILE --output FILE
```

Common Options:

- `--data <file>` - Training data (INPUT/RESPONSE format)
- `--vocab <file>` - Vocabulary file
- `--output <file>` - Output model name
- `--epochs <N>` - Number of epochs (default: 10)
- `--learning-rate <F>` - Learning rate (default: 0.001)
- `--batch-size <N>` - Batch size (default: 4)

Examples:

Quick test (5 epochs):

```bash
./build/bin/chatbot_trainer --data sample_training_data.txt \
                             --vocab vocab.txt \
                             --output test_model.bin \
                             --epochs 5
```

Production training:

```bash
./build/bin/chatbot_trainer --data prod_data.txt \
                             --vocab prod_vocab.txt \
                             --output prod_model.bin \
                             --epochs 50 \
                             --learning-rate 0.0005 \
                             --batch-size 8
```

---

## INCREMENTAL TRAINER

Initialize Incremental Training:

```bash
./build/bin/incremental_trainer init vocab.txt chatbot_model.bin
```

Add Training Data:

```bash
./build/bin/incremental_trainer add new_conversations.txt
```

Train on New Data Only (incremental):

```bash
./build/bin/incremental_trainer train 5
```

Full Retrain on All Data:

```bash
./build/bin/incremental_trainer retrain 10
```

Check Status:

```bash
./build/bin/incremental_trainer status
```

View Training History:

```bash
./build/bin/incremental_trainer history
```

Resume from Interruption:

```bash
./build/bin/incremental_trainer resume
```

Common Options:

- `init <vocab> <model>` - Initialize incremental training system
- `add <file>` - Add training data file to queue
- `train <epochs>` - Train on pending data only
- `retrain <epochs>` - Full retrain on all data
- `status` - Show current session status
- `history` - Show all training sessions
- `resume` - Resume interrupted training

Use Cases:

- **Continuous learning**: Add new conversation data weekly/monthly
- **Avoid retraining**: Train only on new data (much faster)
- **Periodic refresh**: Full retrain every 10 sessions for quality
- **Long training protection**: Auto-saves every 30 minutes

---

## CHATBOT CLI

Run Interactive Chatbot:

```bash
./build/bin/chatbot --vocab vocab.txt --model chatbot_model.bin
```

Load Specific Epoch:

```bash
./build/bin/chatbot --vocab vocab.txt \
                    --model chatbot_model.bin.epoch10
```

Interactive Commands:

- `/help` - Show commands
- `/set strategy` - greedy |beam|sampling|top_k| nucleus
- `/set temperature` - 0.1-2.0 (lower = more focused)
- `/set length` - Max response length
- `/stats` - Show conversation statistics
- `/save` - Save conversation
- `/exit` - Exit chatbot

---

## TROUBLESHOOTING

### Model generates gibberish?

- ✓ Train for more epochs (10-50)
- ✓ Check training loss is decreasing
- ✓ Use lower temperature (0.3-0.7)
- ✓ Try greedy strategy first

### Build fails?

- ✓ Install dependencies: `sudo apt-get install build-essential cmake`
- ✓ Clean and rebuild: `./scripts/build_and_vocab.sh clean && ./scripts/build_and_vocab.sh build`

### Vocabulary too small?

- ✓ Increase `--vocab-size` parameter
- ✓ Provide more training data
- ✓ Lower `--threshold` value

### Out of memory?

- ✓ Reduce batch size (`--batch-size 2`)
- ✓ Reduce model size (smaller d-model, fewer layers)
- ✓ Use smaller vocabulary

---

## VOCABULARY SIZE RECOMMENDATIONS

- **Testing** → 1,000 - 2,000 tokens
- **Small Projects** → 2,000 - 5,000 tokens
- **Medium Projects** → 5,000 - 15,000 tokens
- **Large Projects** → 15,000 - 50,000 tokens

---

## TRAINING PARAMETERS

|Parameter|Quick Test|Production|
|---------|----------|----------|
|Epochs|5-10|20-50|
|Learning Rate|0.001|0.0005-0.001|
|Batch Size|4|8-16|
|Vocabulary Size|1000-2000|5000-15000|
|Model Dimension|128-256|256-512|
|Attention Heads|4-8|8-16|

---

## GENERATION STRATEGIES

- **greedy** - Fast, deterministic, consistent
- **beam** - High quality, moderate speed
- **sampling** - Creative, uses temperature control
- **top_k** - Balanced, limits token choices
- **nucleus** - Adaptive, best for chatbots (default)

Temperature Settings:

- **0.1-0.5** - Very focused, factual
- **0.6-0.8** - Balanced (recommended for chatbots)
- **0.9-1.2** - More creative
- **1.3-2.0** - Very creative, random

---

## FILES AND LOCATIONS

### Scripts

- `scripts/build_and_vocab.sh` - Build automation script

### Documentation

- [BUILD_AND_VOCAB_GUIDE.md](../BUILD_AND_VOCAB_GUIDE.md) - Complete guide
- [training-internals.md](../training-internals.md) - Training details

### Executables (after build)

- `build/bin/vocab_builder` - Vocabulary creation
- `build/bin/chatbot_trainer` - Model training (from scratch)
- `build/bin/incremental_trainer` - Incremental/continuous training
- `build/bin/chatbot` - Interactive CLI
- `build/bin/chatbot_gui` - Qt GUI
- `build/bin/chatbot_api_server` - REST API

### Training Data

- `sample_training_data.txt` - Example training data
- `vocab.txt` - Vocabulary file (created)

### Model Files (after training)

- `chatbot_model.bin.*` - Model checkpoints
- `chatbot_model.bin.epoch{N}.*` - Per-epoch saves

---

## COMPLETE WORKFLOW EXAMPLE

### 1. Build the project

```bash
./scripts/build_and_vocab.sh build
```

### 2. Create vocabulary from your training data

```bash
./build/bin/vocab_builder --input my_training_data.txt \
                          --output my_vocab.txt \
                          --vocab-size 5000 \
                          --format pairs \
                          --stats
```

### 3. Train the model

```bash
./build/bin/chatbot_trainer --data my_training_data.txt \
                            --vocab my_vocab.txt \
                            --output my_model.bin \
                            --epochs 20 \
                            --learning-rate 0.001 \
                            --batch-size 8
```

### 4. Test with the best epoch (check training output)

```bash
./build/bin/chatbot --vocab my_vocab.txt \
                    --model my_model.bin.epoch15
```

### 5. Adjust generation settings in chatbot

```text
/set strategy nucleus
/set temperature 0.7
/set length 100
```

---

## INCREMENTAL TRAINING WORKFLOW

### Initial setup and first training

```bash
# 1. Build and create vocabulary (same as before)
./scripts/build_and_vocab.sh build
./build/bin/vocab_builder --input initial_data.txt \
                          --output vocab.txt \
                          --vocab-size 5000 --format pairs

# 2. Initialize incremental training
./build/bin/incremental_trainer init vocab.txt chatbot_model.bin

# 3. Add initial training data
./build/bin/incremental_trainer add initial_data.txt

# 4. Train (this may take hours/days)
./build/bin/incremental_trainer train 20
```

### Adding new data later (much faster!)

```bash
# 1. Add new conversation data
./build/bin/incremental_trainer add week2_conversations.txt

# 2. Train ONLY on new data (5 epochs)
./build/bin/incremental_trainer train 5

# 3. Test the updated model
./build/bin/chatbot --vocab vocab.txt \
                    --model training_sessions/latest_checkpoint.bin
```

### Periodic full retrain (every 10 updates)

```bash
# After 10 incremental updates, do a full retrain
./build/bin/incremental_trainer retrain 15
```

### Check progress anytime

```bash
# View current status
./build/bin/incremental_trainer status

# View full training history
./build/bin/incremental_trainer history
```

Why use incremental training?

- Initial training on 7500 samples: ~6 days
- Add 500 new samples with regular trainer: another 6 days
- Add 500 new samples with incremental trainer: ~12 hours (87% faster!)

---

> **💡 Need help?** See [BUILD_AND_VOCAB_GUIDE.md](../BUILD_AND_VOCAB_GUIDE.md) for basic training or [incremental-training-guide.md](../incremental-training-guide.md) for continuous learning
