# ChatbotTrainer Context Documentation

> **⚠️ DEPRECATED - March 2026**
>
> This document describes the old standalone ChatbotTrainer system which no longer has a command-line entry point.
>
> **ChatbotTrainer is now an internal component** used by `IncrementalTrainer`. All training is performed through the `incremental_trainer` CLI tool.
>
> **See instead:**
> - [IncrementalTrainer Internals](../guides/incremental-trainer-internals.md) - Current training system
> - [Incremental Training Guide](../../operations/guides/incremental-training-guide.md) - User guide
>
> This document is preserved for historical reference only.

---

## Purpose

`ChatbotTrainer` is a comprehensive command-line training harness for transformer-based chatbot models built on the `EncoderDecoderModel` architecture. It provides a complete end-to-end training pipeline with advanced features including:

- **Vocabulary Management:** Build or load BPE tokenizer vocabularies
- **Data Loading:** Parse conversation pair datasets
- **Data Preprocessing:** Pre-tokenize and cache all training data for efficiency
- **Data Augmentation:** Random shuffling per epoch and for validation split
- **Advanced Training:** Multi-epoch training with sophisticated optimizers
- **Gradient Accumulation:** Simulate larger batch sizes with gradient accumulation
- **Learning Rate Scheduling:** 6 different LR scheduling strategies
- **Regularization:** Gradient clipping, weight decay, early stopping
- **Proper Validation:** Inference-only validation without weight updates
- **Model Checkpointing:** Save best models and periodic checkpoints
- **Training Monitoring:** Loss tracking, gradient norms, validation metrics
- **Configuration Validation:** Auto-correct architectural parameters
- **Interactive Testing:** Test generation after training

## File Location

**Implementation:** `src/ChatbotTrainer.cpp`

## Dependencies

### Core Components

- **EncoderDecoderModel:** Main transformer model
- **BPETokenizer:** Byte-pair encoding tokenizer
- **Optimizer:** Gradient descent optimization algorithms (SGD, Adam, AdamW)
- **ConversationContext:** Conversation management (imported but not actively used)

### Standard Libraries

```cpp
#include <iostream>      // Console I/O
#include <fstream>       // File I/O
#include <sstream>       // String streams
#include <vector>        // Dynamic arrays
#include <string>        // String handling
#include <algorithm>     // STL algorithms
#include <numeric>       // accumulate()
#include <ctime>         // Timing
#include <iomanip>       // I/O formatting
```

## Data Structures

### 1. ConversationPair

**Purpose:** Store training examples (input-output pairs)

```cpp
struct ConversationPair {
    std::string input;     // User input text
    std::string response;  // Bot response text

    ConversationPair(const std::string& in, const std::string& resp);
};
```

Usage:

```cpp
ConversationPair pair("Hello!", "Hi there! How can I help you?");
```

### 1b. TokenizedPair (NEW)

**Purpose:** Store pre-tokenized training examples for efficient training

```cpp
struct TokenizedPair {
    std::vector<int> input_tokens;   // Tokenized input
    std::vector<int> target_tokens;  // Tokenized response
    std::string input_text;          // Original input (for debugging)
    std::string target_text;         // Original response (for debugging)

    TokenizedPair(const std::vector<int>& in_tok, const std::vector<int>& tgt_tok,
                  const std::string& in_txt, const std::string& tgt_txt);
};
```

Benefits:

- Eliminates redundant tokenization in training loop (10-100x speedup)
- Pre-processes all data once before training begins
- Keeps original text for debugging and logging

Usage:

```cpp
std::vector<int> input_tokens = tokenizer->encode("Hello!");
std::vector<int> target_tokens = tokenizer->encode("Hi there!");
TokenizedPair pair(input_tokens, target_tokens, "Hello!", "Hi there!");
```

### 2. LRSchedule Enum

**Purpose:** Define learning rate scheduling strategies

```cpp
enum class LRSchedule {
    CONSTANT,           // No scheduling - constant LR
    LINEAR_WARMUP,      // Linear warmup then constant
    COSINE_DECAY,       // Cosine annealing decay
    WARMUP_COSINE,      // Linear warmup + cosine decay (recommended)
    STEP_DECAY,         // Step-wise decay at intervals
    EXPONENTIAL_DECAY   // Exponential decay
};
```

Strategies:

|Schedule|Description|Use Case|
|----------|-------------|----------|
|`CONSTANT`|Fixed learning rate|Simple baselines, debugging|
|`LINEAR_WARMUP`|Gradual increase then plateau|Stabilize early training|
|`COSINE_DECAY`|Smooth cosine curve decay|Long training runs|
|`WARMUP_COSINE`|Warmup + cosine (recommended)|Production transformer training|
|`STEP_DECAY`|Discrete drops at intervals|Traditional CV tasks|
|`EXPONENTIAL_DECAY`|Continuous exponential decay|Fine-tuning, small datasets|

### 3. TrainingConfig

**Purpose:** Comprehensive training configuration

```cpp
struct TrainingConfig {
    // Model Architecture (validated on initialization)
    int d_model = 512;              // Model dimension (must divide by num_heads)
    int num_heads = 8;              // Attention heads (power of 2 recommended)
    int d_ff = 2048;                // FFN dimension (typically 4x d_model)
    int num_encoder_layers = 6;     // Encoder depth
    int num_decoder_layers = 6;     // Decoder depth
    int max_seq_length = 512;       // Maximum sequence length

    // Training Parameters
    int num_epochs = 10;            // Training epochs
    float learning_rate = 0.001f;   // Initial/base learning rate
    int batch_size = 1;             // Batch size (samples per gradient accumulation)
    int gradient_accumulation_steps = 1;  // Accumulate gradients over N steps (NEW)
    int validation_split = 10;      // Validation fraction (1/10 = 10%)

    // Learning Rate Scheduling
    LRSchedule lr_schedule = LRSchedule::WARMUP_COSINE;
    int warmup_steps = 0;           // Warmup steps (0 = auto: 10% of total)
    float min_learning_rate = 1e-6f;  // Minimum LR for decay
    float lr_decay_factor = 0.1f;   // Decay multiplier
    int lr_decay_steps = 0;         // Steps between decays (0 = auto: per epoch)

    // Optimizer Settings
    OptimizerType optimizer_type = OptimizerType::ADAMW;  // Optimizer
    float adam_beta1 = 0.9f;        // Adam first moment decay
    float adam_beta2 = 0.999f;      // Adam second moment decay
    float weight_decay = 0.01f;     // L2 regularization
    float gradient_clip_norm = 1.0f;  // Max gradient norm (0 = disabled)

    // Checkpointing
    bool save_checkpoints = true;
    int checkpoint_every = 1;       // Save every N epochs

    // Early Stopping
    bool enable_early_stopping = false;
    int patience = 5;               // Epochs to wait for improvement
    float min_delta = 1e-4f;        // Minimum improvement threshold
    bool restore_best_weights = true;  // Restore best on early stop

    // Logging (ENHANCED)
    int log_every = 10;                          // Log every N samples (VERBOSE mode)
    LogLevel log_level = LogLevel::VERBOSE;      // NEW: Logging verbosity
    bool verbose = true;                         // Deprecated: use log_level
};
```

Default Configuration:

- **Model:** 6-layer encoder-decoder, 512-dim, 8 heads
- **Training:** 10 epochs, AdamW optimizer, warmup+cosine LR schedule
- **Regularization:** Weight decay 0.01, gradient clipping norm 1.0
- **Monitoring:** Validation on 10% of data, checkpoints every epoch
- **Logging:** VERBOSE level (detailed progress), log every 10 samples

### 3b. LogLevel Enum (NEW)

**Purpose:** Control logging verbosity

```cpp
enum class LogLevel {
    SILENT = 0,   // No output except errors
    NORMAL = 1,   // Basic progress and results
    VERBOSE = 2,  // Detailed progress (default)
    DEBUG = 3     // Debug information
};
```

Levels:

|Level|Output|Use Case|
|-------|--------|----------|
|`SILENT`|Errors only|Production, automation|
|`NORMAL`|Epoch summaries|Standard monitoring|
|`VERBOSE`|Per-sample progress|Development, debugging|
|`DEBUG`|All debug info|Deep debugging|

See [Metrics and Logging Guide](./chatbot-trainer-metrics-logging.md) for complete details.

## ChatbotTrainer Class

### Private Members

```cpp
// Core components
BPETokenizer* tokenizer;           // Vocabulary and tokenization
EncoderDecoderModel* model;        // Transformer model
Optimizer* optimizer;              // Gradient descent optimizer
TrainingConfig config;             // Training configuration

// Data
std::vector<ConversationPair> training_data;    // Training pairs
std::vector<ConversationPair> validation_data;  // Validation pairs

// Pre-tokenized data for efficient training (NEW)
std::vector<TokenizedPair> tokenized_training_data;    // Cached tokenized training data
std::vector<TokenizedPair> tokenized_validation_data;  // Cached tokenized validation data
std::vector<int> training_indices;                     // Shuffling indices for training data

// Training Statistics
std::vector<float> training_losses;    // Per-epoch training loss
std::vector<float> validation_losses;  // Per-epoch validation loss
std::vector<float> learning_rates;     // LR history
std::vector<float> gradient_norms;     // Gradient norm history

// Enhanced Metrics (NEW - January 2026)
std::vector<float> training_perplexities;      // Per-epoch training perplexity
std::vector<float> validation_perplexities;    // Per-epoch validation perplexity
std::vector<float> training_accuracies;        // Token-level accuracy (placeholder)
std::vector<float> validation_accuracies;      // Token-level accuracy (placeholder)

float best_validation_loss;            // Best validation loss seen
int best_epoch;                        // Epoch with best validation

// Learning Rate State
int global_step;                 // Current training step
int total_training_steps;        // Total steps (accounts for gradient accumulation)
float current_learning_rate;     // Current LR value

// Gradient Accumulation State (NEW)
int accumulation_step;           // Current step in accumulation cycle
float accumulated_loss;          // Loss accumulated over accumulation steps

// Early Stopping State
int epochs_without_improvement;  // Consecutive epochs without improvement
std::string best_model_path;     // Temporary path to best model
bool early_stopped;              // Whether early stopping triggered
```

### Public Methods

#### 1. Constructor & Destructor

```cpp
ChatbotTrainer(const TrainingConfig& cfg);
~ChatbotTrainer();  // Cleans up model, optimizer, tokenizer
```

Constructor Initialization:

- Stores configuration
- Initializes all pointers to `nullptr`
- Sets up training state (best_validation_loss to max, step counters to 0)
- Prepares early stopping state

Destructor:

- Safely deletes `model`, `optimizer`, `tokenizer` (checks for nullptr)

#### 2. Vocabulary Management

##### load_tokenizer()

```cpp
bool load_tokenizer(const std::string& vocab_path);
```

**Purpose:** Load pre-built vocabulary file

**Returns:** `true` if successful, `false` on error

Process:

1. Creates new `BPETokenizer` instance
2. Calls `tokenizer->load_vocab(vocab_path)`
3. Prints vocabulary size
4. Catches and reports exceptions

Example:

```cpp
ChatbotTrainer trainer(config);
if (!trainer.load_tokenizer("vocab.txt")) {
    return 1;  // Error
}
```

##### build_vocabulary()

```cpp
bool build_vocabulary(const std::vector<std::string>& texts,
                     int vocab_size = 5000,
                     const std::string& save_path = "vocab.txt");
```

**Purpose:** Build BPE vocabulary from text corpus

Parameters:

- `texts`: Vector of all training texts
- `vocab_size`: Target vocabulary size (default: 5000)
- `save_path`: Output vocabulary file (default: "vocab.txt")

**Returns:** `true` if successful, `false` on error

Process:

1. Creates new `BPETokenizer`
2. Calls `tokenizer->build_vocab(texts, vocab_size, min_frequency=1)`
3. Saves vocabulary to file
4. Reports vocabulary size

Example:

```cpp
std::vector<std::string> corpus = {"Hello world", "How are you", ...};
trainer.build_vocabulary(corpus, 8000, "my_vocab.txt");
```

#### 3. Data Loading

##### load_conversation_data()

```cpp
bool load_conversation_data(const std::string& filepath);
```

**Purpose:** Load conversation pairs from formatted text file

File Format:

```text
INPUT: <user message>
RESPONSE: <bot response>

INPUT: <user message>
RESPONSE: <bot response>

...
```

Parsing Rules:

- Blank lines separate pairs
- Lines starting with "INPUT:" contain user input
- Lines starting with "RESPONSE:" contain bot response
- Leading/trailing whitespace trimmed
- Pairs with missing input or response are skipped

**Returns:** `true` if at least one pair loaded, `false` otherwise

Example File:

```text
INPUT: Hello!
RESPONSE: Hi there! How can I help you today?

INPUT: What's the weather like?
RESPONSE: I don't have access to real-time weather data, but you can check your local forecast online.

INPUT: Tell me a joke
RESPONSE: Why did the programmer quit his job? Because he didn't get arrays!
```

Usage:

```cpp
if (!trainer.load_conversation_data("conversations.txt")) {
    std::cerr << "Failed to load training data" << std::endl;
    return 1;
}
```

##### split_data() (UPDATED)

```cpp
void split_data();
```

**Purpose:** Split loaded data into training and validation sets with random shuffling

Process:

1. Calculates validation size: `validation_size = training_data.size() / validation_split`
2. Creates shuffled indices using `std::shuffle` with random number generator
3. Assigns first `validation_size` shuffled pairs to `validation_data`
4. Assigns remaining shuffled pairs to `training_data`
5. Prints split statistics

Improvements (January 2026):

- **Random Shuffling:** Uses `std::shuffle` instead of simple tail split
- **Better Generalization:** Random splits prevent bias from data ordering
- **Proper RNG:** Uses `std::mt19937` for quality randomization

Example:

- 100 pairs loaded, `validation_split = 10`
- Random shuffle applied
- Results: 90 training pairs, 10 validation pairs (randomly selected)

Notes:

- Called automatically during `train()`
- If `validation_split <= 0`, no split occurs
- If insufficient data, split is skipped with warning

Previous Behavior:

1. Prints split statistics

Example:

- 100 pairs loaded, `validation_split = 10`
- Results: 90 training pairs, 10 validation pairs

Notes:

- Called automatically during `train()`
- If `validation_split <= 0`, no split occurs
- If insufficient data, split is skipped with warning

##### preprocess_data() (NEW)

```cpp
void preprocess_data();
```

**Purpose:** Pre-tokenize and cache all training and validation data

Process:

1. Tokenizes all training data:
   - Encodes input and response text using tokenizer
   - Creates `TokenizedPair` objects
   - Stores in `tokenized_training_data`
2. Tokenizes all validation data:
   - Same process as training data
   - Stores in `tokenized_validation_data`
3. Initializes shuffling indices for training data
4. Reports preprocessing statistics

Performance Impact:

- **Eliminates redundant tokenization:** 10-100x speedup in training loop
- **One-time cost:** Tokenization happens once before training starts
- **Memory efficient:** Stores tokens as compact integer vectors

Example Output:

```text
🔄 Preprocessing and tokenizing data...
✅ Data preprocessed:
  Training samples: 90
  Validation samples: 10
```

Usage:

- Called automatically during `train()` after `split_data()`
- No manual invocation needed

##### shuffle_training_data() (NEW)

```cpp
void shuffle_training_data();
```

**Purpose:** Randomly shuffle training data indices for each epoch

Process:

1. Uses `std::shuffle` with `std::mt19937` random generator
2. Shuffles `training_indices` vector in-place
3. Training loop accesses data through shuffled indices

Benefits:

- **Improved Generalization:** Different sample order each epoch
- **Prevents Overfitting:** Model doesn't memorize sample order
- **Standard Practice:** Required for proper stochastic gradient descent

Usage:

- Called automatically at start of each epoch in `train_epoch()`
- No manual invocation needed

Example:

```cpp
// Before shuffle: indices = [0, 1, 2, 3, 4, ...]
shuffle_training_data();
// After shuffle: indices = [3, 0, 4, 1, 2, ...]
```

##### log() (NEW - January 2026)

```cpp
void log(LogLevel level, const std::string& message,
         const std::string& color = COLOR_RESET);
```

**Purpose:** Log messages based on configured verbosity level

Parameters:

- `level`: Minimum log level required to print message
- `message`: Text to log
- `color`: ANSI color code (optional)

Behavior:

- Only prints if `config.log_level >= level`
- Thread-safe (uses std::cout with automatic mutex)
- Supports ANSI colors for formatted output

Example:

```cpp
log(LogLevel::VERBOSE, "Training sample 100/500", COLOR_INFO);
log(LogLevel::NORMAL, "Epoch complete - Loss: 2.34", COLOR_SUCCESS);
```

##### calculate_perplexity() (NEW - January 2026)

```cpp
float calculate_perplexity(float loss);
```

**Purpose:** Calculate perplexity from cross-entropy loss

**Formula:** `Perplexity = exp(loss)`

**Returns:** Perplexity value (1.0 = perfect, higher = worse)

Interpretation:

- Measures prediction confidence
- More interpretable than raw loss
- Common metric in NLP

Example:

```cpp
float loss = 2.3026;  // Cross-entropy loss
float ppl = calculate_perplexity(loss);  // Returns 10.0
// Interpretation: Model is as confused as if choosing from 10 equally likely tokens
```

##### calculate_accuracy() (NEW - January 2026)

```cpp
float calculate_accuracy(const std::vector<int>& predictions,
                        const std::vector<int>& targets);
```

**Purpose:** Calculate token-level prediction accuracy

**Status:** Placeholder implementation (returns -1.0)

**Future:** Will be implemented when model exposes prediction probabilities

**Formula:** `Accuracy = (correct_tokens / total_tokens) × 100%`

Example (Future):

```cpp
std::vector<int> predictions = model->get_predictions(input);
std::vector<int> targets = {10, 23, 45, 67};
float acc = calculate_accuracy(predictions, targets);  // Returns e.g., 0.75 (75%)
```

#### 4. Configuration Validation

##### validate_and_correct_config()

```cpp
void validate_and_correct_config();
```

**Purpose:** Validate architectural parameters and auto-correct issues

Validations:

1. **d_model divisibility:** Must be divisible by `num_heads`
   - **Auto-fix:** Round up to nearest multiple
   - **Why:** Multi-head attention splits d_model across heads

2. **d_ff ratio:** Typically 4x d_model
   - **Auto-fix:** Set to 4x if ratio < 2x or > 8x
   - **Why:** Standard transformer architecture

3. **num_heads power of 2:** Recommended for efficiency
   - **Warning only:** Keeps user value
   - **Why:** GPU optimization

4. **d_model range:** Check if in [64, 4096]
   - **Warning only**
   - **Why:** Typical transformer sizes

5. **Learning rate range:** Check if in (0, 1]
   - **Warning only**
   - **Why:** Reasonable optimization range

6. **min_learning_rate < learning_rate**
   - **Auto-fix:** Set to 1% of base LR
   - **Why:** Decay schedules require min < max

7. **Layer counts:** Check encoder/decoder in [1, 48]
   - **Warning only**
   - **Why:** Typical transformer depths

8. **Sequence length:** Check if in [16, 8192]
   - **Warning only**
   - **Why:** Practical memory constraints

Example Output:

```text
🔍 Validating model configuration...
⚠️  d_model (500) not divisible by num_heads (8)
   Auto-corrected to: 504
⚠️  d_ff (1000) has unusual ratio to d_model (ratio: 1.98)
   Auto-corrected to recommended 4x: 2016
✅ Configuration validated and corrected
```

#### 5. Model Initialization

##### initialize_model()

```cpp
void initialize_model();
```

**Purpose:** Create and configure model and optimizer

Process:

1. **Validate configuration**
   - Calls `validate_and_correct_config()`

2. **Create EncoderDecoderModel**

   ```cpp
   model = new EncoderDecoderModel(
       config.d_model,
       config.num_heads,
       config.d_ff,
       config.num_encoder_layers,
       config.num_decoder_layers,
       tokenizer->get_vocab_size(),
       config.max_seq_length
   );
   ```

3. **Create Optimizer**

   ```cpp
   optimizer = new Optimizer(config.optimizer_type, config.learning_rate);
   optimizer->set_weight_decay(config.weight_decay);
   optimizer->set_max_grad_norm(config.gradient_clip_norm);
   ```

4. **Configure Adam parameters** (if Adam/AdamW)

   ```cpp
   optimizer->set_betas(config.adam_beta1, config.adam_beta2);
   ```

5. **Register parameters** (placeholder until full implementation)

   ```cpp
   model->register_parameters(*optimizer);
   ```

Output:

```text
🧠 Initializing transformer model...
  d_model: 512
  num_heads: 8
  d_ff: 2048
  encoder_layers: 6
  decoder_layers: 6
  max_seq_length: 512
  learning_rate: 0.001
✅ Model initialized

🎯 Initializing optimizer...
  Type: AdamW
  Learning rate: 0.001
  Weight decay: 0.01
  Gradient clip norm: 1.0
  Adam beta1: 0.9
  Adam beta2: 0.999
✅ Optimizer initialized
```

#### 6. Learning Rate Scheduling

##### calculate_learning_rate()

```cpp
float calculate_learning_rate(int step);
```

**Purpose:** Calculate LR for given training step

Parameters:

- `step`: Current global training step (0-indexed)

**Returns:** Learning rate value for this step

Algorithms:

1. **CONSTANT**

   ```cpp
   return learning_rate;
   ```

2. **LINEAR_WARMUP**

   ```cpp
   if (step < warmup_steps) {
       return learning_rate * (step / warmup_steps);
   }
   return learning_rate;
   ```

3. **COSINE_DECAY**

   ```cpp
   progress = step / total_training_steps;
   cosine = 0.5 * (1 + cos(π * progress));
   return min_lr + (max_lr - min_lr) * cosine;
   ```

4. **WARMUP_COSINE** (Recommended)

   ```cpp
   if (step < warmup_steps) {
       return learning_rate * (step / warmup_steps);  // Warmup phase
   }
   progress = (step - warmup_steps) / (total_steps - warmup_steps);
   cosine = 0.5 * (1 + cos(π * progress));
   return min_lr + (max_lr - min_lr) * cosine;  // Decay phase
   ```

5. **STEP_DECAY**

   ```cpp
   num_decays = step / decay_steps;
   return learning_rate * decay_factor^num_decays;
   ```

6. **EXPONENTIAL_DECAY**

   ```cpp
   decay_rate = decay_factor^(1/decay_steps);
   return learning_rate * decay_rate^step;
   ```

Auto-Configuration:

- `warmup_steps = 0` → Auto-set to 10% of total steps
- `decay_steps = 0` → Auto-set to steps per epoch

##### update_learning_rate()

```cpp
void update_learning_rate();
```

**Purpose:** Update optimizer and model LR for current step

Process:

1. Calculate new LR: `current_learning_rate = calculate_learning_rate(global_step)`
2. Update optimizer: `optimizer->set_learning_rate(current_learning_rate)`
3. Update model: `model->set_learning_rate(current_learning_rate)` (backward compatibility)

**Called:** Every training step in `train_epoch()`

##### get_schedule_name()

```cpp
const char* get_schedule_name() const;
```

**Purpose:** Get human-readable schedule name

Returns:

- `"Constant"`, `"Linear Warmup"`, `"Cosine Decay"`,
- `"Warmup + Cosine"`, `"Step Decay"`, `"Exponential Decay"`

#### 7. Training

##### train_epoch() (UPDATED)

```cpp
float train_epoch(int epoch);
```

**Purpose:** Train for one complete epoch with gradient accumulation support

Parameters:

- `epoch`: Epoch index (0-based)

**Returns:** Average training loss for the epoch

Improvements (January 2026):

- **Gradient Accumulation:** Simulates larger batch sizes
- **Cached Tokenization:** Uses pre-tokenized data (10-100x faster)
- **Data Shuffling:** Randomly shuffles data at epoch start
- **Efficient Logging:** Adjusts logging frequency for accumulation

Process:

1. **Initialize epoch state**

   ```cpp
   float total_loss = 0.0f;
   float total_grad_norm = 0.0f;
   int num_samples = tokenized_training_data.size();  // Uses cached data
   int effective_batch_size = batch_size * gradient_accumulation_steps;
   ```

2. **Shuffle training data**

   ```cpp
   shuffle_training_data();  // Random order each epoch
   ```

3. **Reset accumulation state**

   ```cpp
   accumulation_step = 0;
   accumulated_loss = 0.0f;
   ```

4. **For each training sample:**

   a. **Update learning rate** (only at optimizer step)

      ```cpp
      if (accumulation_step == 0) {
          update_learning_rate();
      }
      ```

   b. **Zero gradients** (only at start of accumulation)

      ```cpp
      if (accumulation_step == 0) {
          model->zero_grad();
      }
      ```

   c. **Forward pass using cached tokens**

      ```cpp
      const auto& pair = tokenized_training_data[training_indices[i]];
      Matrix logits = model->forward(pair.input_tokens, pair.target_tokens);
      ```

   d. **Compute and scale loss**

      ```cpp
      float loss = model->compute_loss_for_training(logits, pair.target_tokens);
      float scaled_loss = loss / gradient_accumulation_steps;
      accumulated_loss += loss;
      ```

   e. **Backward pass with gradient scaling**

      ```cpp
      Matrix grad_loss = model->compute_loss_gradient_for_training(logits, pair.target_tokens);

      // Scale gradients for accumulation
      if (gradient_accumulation_steps > 1) {
          float scale = 1.0f / gradient_accumulation_steps;
          for (int r = 0; r < grad_loss.rows; r++) {
              for (int c = 0; c < grad_loss.cols; c++) {
                  grad_loss.data[r][c] *= scale;
              }
          }
      }

      model->backward_pass(grad_loss);
      accumulation_step++;
      ```

   f. **Update weights** (after accumulating enough gradients)

      ```cpp
      bool should_update = (accumulation_step >= gradient_accumulation_steps) ||
                          (i == num_samples - 1);  // Last sample

      if (should_update) {
          // Get gradient norm before clipping
          float grad_norm = optimizer->get_gradient_norm();
          total_grad_norm += grad_norm;

          // Clip gradients (if enabled)
          if (gradient_clip_norm > 0.0f) {
              optimizer->clip_gradients();
          }

          // Update weights
          model->update_weights();

          total_loss += accumulated_loss;
          global_step++;

          // Reset accumulation
          accumulation_step = 0;
          accumulated_loss = 0.0f;
      }
      ```

   g. **Log progress** (adjusted for accumulation)

```text
      Sample 32/100 (Update 8) - Loss: 2.3456 - Avg: 2.4012 - LR: 0.000123 - GradNorm: 0.8765
      ```

5. **Compute epoch statistics**

   ```cpp
   float epoch_loss = total_loss / num_updates;
   float avg_grad_norm = total_grad_norm / num_updates;
   training_losses.push_back(epoch_loss);
   learning_rates.push_back(current_learning_rate);
   gradient_norms.push_back(avg_grad_norm);
   ```

Performance Impact:

- **Tokenization:** 10-100x faster (eliminated from loop)
- **Gradient Accumulation:** Enables effective batch sizes > 1 without OOM
- **Shuffling:** Better generalization

**Error Handling:** Catches exceptions per sample, resets accumulation state, continues training

##### validate() (UPDATED)

```cpp
float validate();
```

**Purpose:** Evaluate model on validation set (inference-only, no weight updates)

**Returns:** Average validation loss (0.0 if no validation data)

Improvements (January 2026):

- **Proper Validation:** Uses `model->evaluate()` instead of `train_step()`
- **No Weight Updates:** Validation data is NOT used for training
- **Cached Tokenization:** Uses pre-tokenized validation data
- **Training Mode Control:** Sets model to evaluation mode during validation

Process:

1. **Check validation data**

   ```cpp
   if (tokenized_validation_data.empty()) return 0.0f;
   ```

2. **Set evaluation mode**

   ```cpp
   model->set_training(false);  // Disable training mode
   ```

3. **Compute loss on validation set**

   ```cpp
   for (const auto& pair : tokenized_validation_data) {
       // Use evaluate() which doesn't update weights
       float loss = model->evaluate(pair.input_text, pair.target_text);
       total_loss += loss;
   }
   ```

4. **Restore training mode**

   ```cpp
   model->set_training(true);  // Re-enable training
   ```

5. **Track statistics**

   ```cpp
   float validation_loss = total_loss / num_samples;
   validation_losses.push_back(validation_loss);
   ```

6. **Update best model tracking**

   ```cpp
   if (validation_loss < best_validation_loss - min_delta) {
       best_validation_loss = validation_loss;
       best_epoch = training_losses.size();
       epochs_without_improvement = 0;

       // Save best model if early stopping enabled
       if (enable_early_stopping && restore_best_weights) {
           model->save_model("best_model_temp.bin");
       }
   } else {
       epochs_without_improvement++;
   }
   ```

**Critical Fix:** Previous implementation used `train_step()` which contaminated validation data by updating weights. This is now fixed with proper inference-only evaluation.

##### should_early_stop()

```cpp
bool should_early_stop();
```

**Purpose:** Check if early stopping criteria met

**Returns:** `true` if should stop, `false` otherwise

Criteria:

```cpp
return enable_early_stopping &&
       !validation_data.empty() &&
       epochs_without_improvement >= patience;
```

##### restore_best_model()

```cpp
void restore_best_model();
```

**Purpose:** Reload best model weights after early stopping

Process:

1. Check if best model was saved
2. Delete current model
3. Create new model with same architecture
4. Load weights from `best_model_path`
5. Delete temporary file

**Called:** When early stopping triggers and `restore_best_weights = true`

#### 8. Checkpointing

##### save_checkpoint()

```cpp
void save_checkpoint(const std::string& filepath, int epoch);
```

**Purpose:** Save model checkpoint to disk

Parameters:

- `filepath`: Output file path
- `epoch`: Current epoch number (for logging)

Process:

```cpp
model->save_model(filepath);
```

Called:

- Every `checkpoint_every` epochs during training
- At end of training (final model)

Example Checkpoints:

```text
chatbot_model.bin.epoch1
chatbot_model.bin.epoch2
...
chatbot_model.bin  (final)
```

#### 9. Main Training Loop

##### train()

```cpp
void train(const std::string& output_model_path = "chatbot_model.bin");
```

**Purpose:** Execute complete training pipeline

Parameters:

- `output_model_path`: Final model save path (default: "chatbot_model.bin")

Process:

1. **Validate prerequisites**

   ```cpp
   if (!tokenizer) { error("Tokenizer not initialized"); return; }
   if (training_data.empty()) { error("No training data"); return; }
   ```

2. **Initialize model and optimizer**

   ```cpp
   initialize_model();
   ```

3. **Split data**

   ```cpp
   split_data();
   ```

4. **Calculate training steps**

   ```cpp
   total_training_steps = num_epochs * training_data.size();
   ```

5. **Print configuration**
   - LR schedule details
   - Early stopping settings
   - Dataset statistics

6. **Training loop**

   ```cpp
   for (int epoch = 0; epoch < num_epochs; epoch++) {
       // Train epoch
       float train_loss = train_epoch(epoch);

       // Validate
       if (!validation_data.empty()) {
           float val_loss = validate();

           // Check early stopping
           if (should_early_stop()) {
               std::cout << "Early stopping triggered\n";
               early_stopped = true;

               if (restore_best_weights) {
                   restore_best_model();
               }
               break;
           }
       }

       // Save checkpoint
       if (save_checkpoints && (epoch + 1) % checkpoint_every == 0) {
           save_checkpoint(output_model_path + ".epoch" + std::to_string(epoch + 1), epoch);
       }
   }
   ```

7. **Save final model**

   ```cpp
   save_checkpoint(output_model_path, num_epochs);
   ```

8. **Print summary**

   ```cpp
   print_training_summary(duration);
   ```

#### 10. Reporting

##### print_training_summary()

```cpp
void print_training_summary(long duration);
```

**Purpose:** Print comprehensive training summary

Displays:

- Total epochs (and if early stopped)
- Training/validation sample counts
- Training time in seconds
- Final vs initial training loss
- Final vs initial learning rate
- Final and average gradient norms
- Final validation loss
- Best validation loss and epoch

Example Output:

```text
╔═══════════════════════════════════════╗
║     🎉 TRAINING COMPLETE! 🎉         ║
╚═══════════════════════════════════════╝

📊 Training Summary:
  Total epochs: 10
  Training samples: 90
  Validation samples: 10
  Training time: 1234 seconds
  Final training loss: 1.2345
  Initial training loss: 3.4567
  Final learning rate: 0.000001
  Initial learning rate: 0.001000
  Final gradient norm: 0.5432
  Average gradient norm: 0.6789
  Final validation loss: 1.3456
  Best validation loss: 1.2345 (epoch 8)
```

##### test_generation()

```cpp
void test_generation(const std::vector<std::string>& test_prompts);
```

**Purpose:** Test model generation with example prompts

Parameters:

- `test_prompts`: Vector of input prompts to test

Process:

```cpp
for (const auto& prompt : test_prompts) {
    std::cout << "Prompt: " << prompt << std::endl;
    std::string response = model->generate_response(prompt, max_tokens=50);
    std::cout << "Response: " << response << std::endl;
}
```

Example Usage:

```cpp
std::vector<std::string> prompts = {
    "Hello!",
    "How are you?",
    "What is your name?"
};
trainer.test_generation(prompts);
```

## Command-Line Interface

### Main Function

**Purpose:** Parse arguments and execute training

Usage:

```bash
./ChatbotTrainer [options]
```

### Command-Line Arguments

#### Required Arguments

|Argument|Description|Example|
|----------|-------------|---------|
|`--data <file>`|Training data file|`--data conversations.txt`|
|`--vocab <file>` OR `--build-vocab <size>`|Vocabulary source|`--vocab vocab.txt` OR `--build-vocab 5000`|

#### Model Architecture

|Argument|Default|Description|
|----------|---------|-------------|
|`--d-model <n>`|512|Model dimension|
|`--heads <n>`|8|Number of attention heads|
|`--d-ff <n>`|2048|Feed-forward dimension|
|`--encoder-layers <n>`|6|Encoder layer count|
|`--decoder-layers <n>`|6|Decoder layer count|
|`--max-length <n>`|512|Maximum sequence length|

#### Training Parameters

|Argument|Default|Description|
|----------|---------|-------------|
|`--epochs <n>`|10|Number of training epochs|
|`--lr <rate>`|0.001|Initial learning rate|
|`--batch-size <n>`|1|Batch size for training (NEW)|
|`--grad-accum <n>`|1|Gradient accumulation steps (NEW)|
|`--output <file>`|chatbot_model.bin|Output model file|

**Gradient Accumulation:** Simulates larger effective batch sizes by accumulating gradients over multiple samples before updating weights. Effective batch size = `batch_size × grad_accum`. Useful for memory-constrained environments.

#### Learning Rate Scheduling

|Argument|Default|Description|
|----------|---------|-------------|
|`--lr-schedule <name>`|warmup-cosine|Schedule type: constant, warmup, cosine, warmup-cosine, step, exponential|
|`--warmup-steps <n>`|auto (10%)|Warmup steps|
|`--min-lr <rate>`|1e-6|Minimum learning rate|

#### Optimizer Configuration

|Argument|Default|Description|
|----------|---------|-------------|
|`--optimizer <name>`|adamw|Optimizer: sgd, sgd-momentum, adam, adamw|
|`--weight-decay <val>`|0.01|Weight decay / L2 regularization|
|`--grad-clip <norm>`|1.0|Gradient clipping max norm (0=disabled)|
|`--adam-beta1 <val>`|0.9|Adam beta1 parameter|
|`--adam-beta2 <val>`|0.999|Adam beta2 parameter|

#### Early Stopping

|Argument|Default|Description|
|----------|---------|-------------|
|`--early-stopping`|disabled|Enable early stopping|
|`--patience <n>`|5|Epochs to wait for improvement|
|`--min-delta <delta>`|1e-4|Minimum improvement threshold|
|`--no-restore-best`|false|Don't restore best weights|

#### Other Options

|Argument|Default|Description|
|----------|---------|-------------|
|`--no-validation`|false|Skip validation split|
|`--help`|-|Show help message|

### Example Commands

#### Basic Training

```bash
./ChatbotTrainer \
    --data conversations.txt \
    --build-vocab 5000 \
    --epochs 20 \
    --output my_chatbot.bin
```

#### Advanced Training with All Features (UPDATED)

```bash
./ChatbotTrainer \
    --data conversations.txt \
    --vocab my_vocab.txt \
    --epochs 50 \
    --lr 0.0001 \
    --optimizer adamw \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr-schedule warmup-cosine \
    --warmup-steps 1000 \
    --min-lr 1e-6 \
    --early-stopping \
    --patience 5 \
    --d-model 768 \
    --heads 12 \
    --encoder-layers 8 \
    --decoder-layers 8 \
    --output production_model.bin
```

**Note:** With `--batch-size 4` and `--grad-accum 8`, the effective batch size is 32, simulating training on larger batches while fitting in memory.

#### Small Model for Testing

```bash
./ChatbotTrainer \
    --data test_data.txt \
    --build-vocab 1000 \
    --epochs 5 \
    --d-model 256 \
    --heads 4 \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --lr 0.001 \
    --optimizer adam \
    --no-validation \
    --output test_model.bin
```

## Training Pipeline Flow

### Complete Flow Diagram

```text
┌─────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                       │
├─────────────────────────────────────────────────────────┤
│ Parse command-line arguments                            │
│ Create TrainingConfig with user parameters              │
│ Create ChatbotTrainer instance                          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. VOCABULARY SETUP                                     │
├─────────────────────────────────────────────────────────┤
│ Option A: Load existing vocabulary                      │
│   → load_tokenizer(vocab_file)                          │
│                                                          │
│ Option B: Build new vocabulary                          │
│   → Extract texts from training data                    │
│   → build_vocabulary(texts, vocab_size)                 │
│   → Save to file                                        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. DATA LOADING                                         │
├─────────────────────────────────────────────────────────┤
│ load_conversation_data(data_file)                       │
│   → Parse INPUT/RESPONSE pairs                          │
│   → Store in training_data vector                       │
│   → Report pair count                                   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. TRAINING EXECUTION                                   │
├─────────────────────────────────────────────────────────┤
│ train(output_model_path)                                │
│   ├─ initialize_model()                                 │
│   │   ├─ validate_and_correct_config()                  │
│   │   ├─ Create EncoderDecoderModel                     │
│   │   ├─ Create Optimizer                               │
│   │   └─ Register parameters                            │
│   │                                                      │
│   ├─ split_data()                                       │
│   │   └─ Randomly split into training/validation (NEW)  │
│   │                                                      │
│   ├─ preprocess_data() (NEW)                            │
│   │   ├─ Tokenize all training data once                │
│   │   ├─ Tokenize all validation data once              │
│   │   └─ Initialize shuffling indices                   │
│   │                                                      │
│   ├─ Calculate total_training_steps (w/ accumulation)   │
│   │                                                      │
│   └─ FOR each epoch:                                    │
│       ├─ train_epoch(epoch)                             │
│       │   ├─ shuffle_training_data() (NEW)              │
│       │   └─ FOR each training sample:                  │
│       │       ├─ update_learning_rate()                 │
│       │       ├─ zero_grad() (if accum_step == 0)       │
│       │       ├─ forward() [uses cached tokens] (NEW)   │
│       │       ├─ compute_loss()                         │
│       │       ├─ backward_pass() [scaled gradients]     │
│       │       └─ IF accumulation complete:              │
│       │           ├─ clip_gradients()                   │
│       │           └─ update_weights()                   │
│       │                                                  │
│       ├─ validate() (UPDATED - inference only)          │
│       │   ├─ Set evaluation mode (NEW)                  │
│       │   ├─ Compute validation loss [no weight update] │
│       │   ├─ Restore training mode (NEW)                │
│       │   ├─ Track best model                           │
│       │   └─ Update early stopping counters             │
│       │                                                  │
│       ├─ should_early_stop()                            │
│       │   └─ IF true: break and restore_best_model()    │
│       │                                                  │
│       └─ save_checkpoint() (if checkpoint_every)        │
│                                                          │
│   ├─ save_checkpoint(final_model)                       │
│   └─ print_training_summary()                           │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 5. TESTING (Optional)                                   │
├─────────────────────────────────────────────────────────┤
│ test_generation(test_prompts)                           │
│   → Generate responses for sample prompts               │
│   → Display results                                     │
└─────────────────────────────────────────────────────────┘
```

### Per-Epoch Training Flow

```text
EPOCH START
   │
   ├─ Initialize: total_loss=0, total_grad_norm=0
   │
   ├─ FOR EACH TRAINING SAMPLE:
   │  │
   │  ├─ STEP 1: Update Learning Rate
   │  │  └─ calculate_learning_rate(global_step) based on schedule
   │  │
   │  ├─ STEP 2: Zero Gradients
   │  │  ├─ optimizer->zero_grad()
   │  │  └─ model->zero_grad()
   │  │
   │  ├─ STEP 3: Tokenize
   │  │  ├─ input_tokens = tokenizer->encode(input)
   │  │  └─ target_tokens = tokenizer->encode(response)
   │  │
   │  ├─ STEP 4: Forward Pass
   │  │  └─ logits = model->forward(input_tokens, target_tokens)
   │  │
   │  ├─ STEP 5: Compute Loss
   │  │  └─ loss = model->compute_loss_for_training(logits, target_tokens)
   │  │
   │  ├─ STEP 6: Backward Pass
   │  │  ├─ grad = model->compute_loss_gradient_for_training(logits, target_tokens)
   │  │  └─ model->backward_pass(grad)
   │  │
   │  ├─ STEP 7: Gradient Monitoring
   │  │  └─ grad_norm = optimizer->get_gradient_norm()
   │  │
   │  ├─ STEP 8: Gradient Clipping (if enabled)
   │  │  └─ optimizer->clip_gradients()
   │  │
   │  ├─ STEP 9: Update Weights
   │  │  └─ model->update_weights()
   │  │
   │  ├─ STEP 10: Update Statistics
   │  │  ├─ total_loss += loss
   │  │  ├─ total_grad_norm += grad_norm
   │  │  └─ global_step++
   │  │
   │  └─ STEP 11: Log Progress (every log_every samples)
   │
   ├─ COMPUTE EPOCH STATISTICS
   │  ├─ epoch_loss = total_loss / num_samples
   │  ├─ avg_grad_norm = total_grad_norm / num_samples
   │  ├─ training_losses.push_back(epoch_loss)
   │  ├─ learning_rates.push_back(current_learning_rate)
   │  └─ gradient_norms.push_back(avg_grad_norm)
   │
   └─ RETURN epoch_loss
```

## Learning Rate Schedules Visualized

### WARMUP_COSINE (Recommended)

```text
LR
 │
 │     Warmup Phase    │         Cosine Decay Phase
 │                     │
max ─┤        ╱────────┼────╲
     │      ╱          │     ╲
     │    ╱            │      ╲
     │  ╱              │       ╲
     │╱                │        ╲___
min ─┴─────────────────┴─────────────────────────── Steps
     0             warmup         total_steps
```

Characteristics:

- Linear increase to max LR (stabilizes training)
- Smooth cosine decay to min LR (fine-tuning)
- Best for transformer training

### COSINE_DECAY

```text
LR
 │
max ─┤────╲
     │     ╲
     │      ╲
     │       ╲
     │        ╲___
min ─┴─────────────────── Steps
     0            total_steps
```

Characteristics:

- Immediate cosine decay from start
- Smooth continuous decay
- Good for pre-warmed models

### LINEAR_WARMUP

```text
LR
 │
max ─┤      ╱────────────────
     │    ╱
     │  ╱
     │╱
  0 ─┴──────────────────────── Steps
     0        warmup    total_steps
```

Characteristics:

- Linear ramp-up then constant
- Simple and stable
- Good for small models

### STEP_DECAY

```text
LR
 │
max ─┤────────┐
     │        │
     │        └────────┐
     │                 │
     │                 └────────┐
min ─┴──────────────────────────── Steps
     0      decay   2×decay  3×decay
```

Characteristics:

- Discrete drops at intervals
- Traditional CV approach
- Can cause training instability

## Configuration Recommendations

### Small Chatbot (Fast Training)

```cpp
TrainingConfig config;
config.d_model = 256;
config.num_heads = 4;
config.d_ff = 1024;
config.num_encoder_layers = 2;
config.num_decoder_layers = 2;
config.max_seq_length = 256;

config.num_epochs = 10;
config.learning_rate = 0.001f;
config.optimizer_type = OptimizerType::ADAM;
config.lr_schedule = LRSchedule::LINEAR_WARMUP;
config.gradient_clip_norm = 1.0f;
```

**Use Case:** Quick experiments, small datasets (<1000 pairs)

### Medium Chatbot (Balanced)

```cpp
TrainingConfig config;  // Use defaults
config.d_model = 512;
config.num_heads = 8;
config.d_ff = 2048;
config.num_encoder_layers = 6;
config.num_decoder_layers = 6;

config.num_epochs = 20;
config.learning_rate = 0.0001f;
config.optimizer_type = OptimizerType::ADAMW;
config.weight_decay = 0.01f;
config.lr_schedule = LRSchedule::WARMUP_COSINE;
config.enable_early_stopping = true;
config.patience = 5;
```

**Use Case:** Production chatbots, moderate datasets (1K-10K pairs)

### Large Chatbot (High Quality)

```cpp
TrainingConfig config;
config.d_model = 768;
config.num_heads = 12;
config.d_ff = 3072;
config.num_encoder_layers = 12;
config.num_decoder_layers = 12;
config.max_seq_length = 1024;

config.num_epochs = 50;
config.learning_rate = 5e-5f;  // Lower for stability
config.min_learning_rate = 1e-7f;
config.optimizer_type = OptimizerType::ADAMW;
config.weight_decay = 0.01f;
config.adam_beta1 = 0.9f;
config.adam_beta2 = 0.98f;  // Higher for transformers
config.gradient_clip_norm = 1.0f;
config.lr_schedule = LRSchedule::WARMUP_COSINE;
config.warmup_steps = 2000;

config.enable_early_stopping = true;
config.patience = 10;
config.min_delta = 1e-5f;
config.restore_best_weights = true;

config.save_checkpoints = true;
config.checkpoint_every = 5;
```

**Use Case:** Large datasets (>10K pairs), production systems

## Features & Capabilities

### ✅ Implemented Features

1. **Vocabulary Management**
   - Load pre-built BPE vocabularies
   - Build vocabularies from training data
   - Save/load vocabulary files

2. **Data Loading**
   - Parse INPUT/RESPONSE format
   - Automatic train/validation split
   - Flexible split ratios

3. **Model Architecture**
   - Configurable transformer dimensions
   - Variable encoder/decoder depths
   - Attention head configuration
   - Feed-forward network sizing

4. **Advanced Optimization**
   - 4 optimizer types (SGD, SGD+Momentum, Adam, AdamW)
   - Gradient clipping (prevent explosions)
   - Weight decay / L2 regularization
   - Configurable Adam hyperparameters

5. **Learning Rate Scheduling**
   - 6 scheduling strategies
   - Automatic warmup configuration
   - Min/max LR bounds
   - Per-step LR updates

6. **Training Monitoring**
   - Per-epoch loss tracking
   - Gradient norm monitoring
   - Learning rate history
   - Validation metrics
   - Progress logging

7. **Model Checkpointing**
   - Periodic checkpoint saving
   - Final model persistence
   - Best model tracking

8. **Early Stopping**
   - Validation-based stopping
   - Configurable patience
   - Best weight restoration
   - Minimum delta threshold

9. **Configuration Validation**
   - Auto-correct architectural errors
   - Range checking
   - Warning system
   - Best practice enforcement

10. **Command-Line Interface**
    - Comprehensive argument parsing
    - Help documentation
    - Flexible configuration
    - Sensible defaults

11. **Colored Console Output**
    - ANSI color formatting
    - Info/success/warning/error colors
    - Progress indicators
    - Visual separators

### 🚧 Limitations & Future Work

1. **Batch Size = 1 Only**
   - Current: Single sample per step
   - Future: Mini-batch training for efficiency
   - Impact: Slower training, noisier gradients

2. **Validation Uses train_step()**
   - Current: Validation updates weights (incorrect)
   - Future: Separate `evaluate()` method
   - Impact: Validation loss not accurate

3. **Parameter Exposure Incomplete**
   - Current: `model->update_weights()` used
   - Future: `optimizer->step()` with exposed parameters
   - Impact: Optimizer doesn't directly control model weights

4. **No Gradient Accumulation**
   - Current: Update every step
   - Future: Accumulate over multiple samples
   - Impact: Can't simulate large batch sizes

5. **No Mixed Precision**
   - Current: Float32 only
   - Future: FP16/BF16 support
   - Impact: Higher memory usage, slower training

6. **No Distributed Training**
   - Current: Single-threaded
   - Future: Multi-GPU, data parallelism
   - Impact: Limited scalability

7. **Basic Error Recovery**
   - Current: Skip failed samples, continue
   - Future: Better error diagnostics
   - Impact: Silent failures possible

8. **No Curriculum Learning**
   - Current: Random sample order
   - Future: Easy→hard sample ordering
   - Impact: Potentially slower convergence

## Integration with Other Components

### EncoderDecoderModel

Methods Used:

```cpp
// Constructor
EncoderDecoderModel(d_model, num_heads, d_ff, enc_layers, dec_layers, vocab_size, max_len);

// Tokenizer access
BPETokenizer* get_tokenizer();

// Configuration
void set_learning_rate(float lr);
void set_training(bool training);

// Training
Matrix forward(input_tokens, target_tokens);
float compute_loss_for_training(logits, target_tokens);
Matrix compute_loss_gradient_for_training(logits, target_tokens);
void backward_pass(grad_output);
void zero_grad();
void update_weights();
void register_parameters(Optimizer& optimizer);  // Placeholder

// Generation
std::string generate_response(prompt, max_tokens);

// Persistence
void save_model(filepath);
void load_model(filepath);
```

### Optimizer

Methods Used:

```cpp
// Constructor
Optimizer(OptimizerType type, float learning_rate);

// Configuration
void set_learning_rate(float lr);
void set_weight_decay(float decay);
void set_max_grad_norm(float max_norm);
void set_betas(float beta1, float beta2);

// Gradient operations
void zero_grad();
float get_gradient_norm();
void clip_gradients();

// Parameter management
void add_parameter_group(Matrix* matrix);  // Future

// Optimization step
void step();  // Future - currently uses model->update_weights()
```

### BPETokenizer

Methods Used:

```cpp
// Vocabulary management
void build_vocab(texts, vocab_size, min_frequency);
void load_vocab(filepath);
void save_vocab(filepath);
int get_vocab_size();

// Tokenization
std::vector<int> encode(text);
std::string decode(tokens);
```

## Best Practices

### 1. Data Preparation

Format Training Data Correctly:

```text
INPUT: User message here
RESPONSE: Bot response here

INPUT: Another user message
RESPONSE: Another bot response
```

Tips:

- Ensure blank lines between pairs
- Remove extra whitespace
- Use consistent formatting
- Validate data before training

### 2. Vocabulary Building

Choose Appropriate Vocab Size:

- Small datasets (<1K pairs): 1000-2000 tokens
- Medium datasets (1K-10K): 3000-5000 tokens
- Large datasets (>10K): 5000-10000 tokens

Vocabulary Quality:

```bash
# Build from all training texts
./ChatbotTrainer --data train.txt --build-vocab 5000
```

### 3. Learning Rate Selection

Start Conservative:

- Large models: 1e-5 to 1e-4
- Medium models: 1e-4 to 1e-3
- Small models: 1e-3 to 1e-2

Use Warmup:

```bash
--lr 0.0001 --lr-schedule warmup-cosine --warmup-steps 1000
```

### 4. Regularization

Prevent Overfitting:

```bash
--weight-decay 0.01 --grad-clip 1.0 --early-stopping --patience 5
```

Validation Split:

- Small datasets: 1/5 (20%)
- Large datasets: 1/10 (10%)

### 5. Architecture Selection

Follow Divisibility Rules:

- `d_model` must divide by `num_heads`
- `d_ff` typically 4× `d_model`
- `num_heads` should be power of 2

Example Valid Configurations:

```text
d_model=512, num_heads=8, d_ff=2048  ✅
d_model=768, num_heads=12, d_ff=3072 ✅
d_model=500, num_heads=8, d_ff=2000  ❌ (500 ÷ 8 not integer)
```

### 6. Monitoring Training

Watch These Metrics:

- Training loss should decrease
- Validation loss should track training loss
- Gradient norms should be stable (<10)
- Learning rate should follow expected schedule

Warning Signs:

- Loss = NaN → Reduce LR or increase gradient clipping
- Loss not decreasing → Increase LR or check data
- Validation >> Training → Overfitting, add regularization
- Gradient norm >> 10 → Increase gradient clipping

### 7. Checkpointing Strategy

Save Frequently:

```bash
--checkpoint-every 1  # Save every epoch during development
--checkpoint-every 5  # Less frequent for long training
```

Use Early Stopping:

```bash
--early-stopping --patience 5 --restore-best-weights
```

## Output Files

### Model Checkpoints

Naming Convention:

```text
chatbot_model.bin.epoch1
chatbot_model.bin.epoch2
...
chatbot_model.bin  (final)
```

Checkpoint Contents:

- All model weights (encoder, decoder, LM head)
- Embedding matrices
- Architecture configuration

Usage:

```bash
# Load checkpoint for inference
./chatbot vocab.txt chatbot_model.bin.epoch10
```

### Vocabulary File

Format:

```text
<unk>
<pad>
<s>
</s>
token1
token2
...
```

Required for:

- Model inference
- Continued training
- Text generation

### Best Model (Early Stopping)

Temporary File:

```text
best_model_temp.bin  (deleted after restoration)
```

Automatic Management:

- Created during training if early stopping enabled
- Restored to final model if training stops early
- Automatically deleted after restoration

## Error Messages & Troubleshooting

### Common Errors

#### "Tokenizer not initialized"

```text
❌ Tokenizer not initialized!
```

**Cause:** Forgot to load or build vocabulary
**Solution:** Add `--vocab vocab.txt` or `--build-vocab 5000`

#### "No training data loaded"

```text
❌ No training data loaded!
```

**Cause:** Data file not loaded or empty
**Solution:** Check file path and format

#### "Cannot open file"

```text
❌ Cannot open file: conversations.txt
```

**Cause:** File doesn't exist or no read permission
**Solution:** Verify file path and permissions

#### "d_model not divisible by num_heads"

```text
⚠️  d_model (500) not divisible by num_heads (8)
   Auto-corrected to: 504
```

**Cause:** Invalid architecture configuration
**Effect:** Auto-corrected (no action needed)

#### "Unknown optimizer"

```text
Unknown optimizer: adm
```

**Cause:** Typo in optimizer name
**Solution:** Use `sgd`, `sgd-momentum`, `adam`, or `adamw`

#### "Unknown LR schedule"

```text
Unknown LR schedule: cosine-warmup
```

**Cause:** Invalid schedule name
**Solution:** Use `constant`, `warmup`, `cosine`, `warmup-cosine`, `step`, or `exponential`

### Training Issues

#### Loss = NaN

Symptoms:

```text
Sample 10/100 - Loss: nan
```

Causes & Solutions:

1. Learning rate too high → Reduce LR by 10×
2. Gradient explosion → Lower `--grad-clip` threshold
3. Bad data → Check for empty/corrupted samples

#### Loss Not Decreasing

Symptoms:

```text
Epoch 5 - Loss: 3.456  (same as epoch 1)
```

Causes & Solutions:

1. Learning rate too low → Increase LR
2. Model too small → Increase `d_model`, layers
3. Data quality → Review training pairs

#### Validation Loss >> Training Loss

Symptoms:

```text
Training Loss: 1.234
Validation Loss: 5.678
```

Causes & Solutions:

1. Overfitting → Add `--weight-decay`, `--early-stopping`
2. Too few samples → Get more training data
3. Train/val mismatch → Check data distribution

## Performance Considerations

### Memory Usage

Approximate Memory per Model:

|Configuration|Parameters|Memory|
|---------------|-----------|--------|
|Small (256-dim, 2 layers)|~10M|~100 MB|
|Medium (512-dim, 6 layers)|~40M|~400 MB|
|Large (768-dim, 12 layers)|~100M|~1 GB|

Additional Memory:

- Optimizer state (Adam/AdamW): 2× model size
- Gradient storage: 1× model size
- Activations: Depends on sequence length

### Training Speed

Factors:

- Model size (parameters)
- Sequence length
- Number of training samples
- Gradient clipping overhead
- Checkpoint frequency

Typical Speed (CPU):

- Small model: ~10 samples/second
- Medium model: ~2 samples/second
- Large model: ~0.5 samples/second

Optimization Tips:

1. Use smaller max_seq_length if possible
2. Reduce checkpoint frequency for long runs
3. Consider fewer validation samples
4. Use `--no-validation` for fastest training

## Example Workflows

### Workflow 1: Quick Experiment

```bash
# 1. Prepare small dataset (100 pairs)
cat > test_data.txt << EOF
INPUT: Hello
RESPONSE: Hi there!

INPUT: Goodbye
RESPONSE: See you later!
EOF

# 2. Train small model quickly
./ChatbotTrainer \
    --data test_data.txt \
    --build-vocab 500 \
    --epochs 5 \
    --d-model 128 \
    --heads 4 \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --lr 0.001 \
    --no-validation \
    --output quick_test.bin

# 3. Test generation
./chatbot vocab.txt quick_test.bin
```

### Workflow 2: Production Training

```bash
# 1. Prepare production dataset (10K pairs)
# ... (manual data collection/cleaning)

# 2. Build vocabulary from training data
./ChatbotTrainer \
    --data production_data.txt \
    --build-vocab 8000 \
    --epochs 1 \
    --output temp.bin  # We'll discard this

# 3. Train production model with all features
./ChatbotTrainer \
    --data production_data.txt \
    --vocab vocab.txt \
    --epochs 50 \
    --lr 0.00005 \
    --optimizer adamw \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --lr-schedule warmup-cosine \
    --warmup-steps 2000 \
    --min-lr 1e-7 \
    --early-stopping \
    --patience 10 \
    --d-model 768 \
    --heads 12 \
    --encoder-layers 8 \
    --decoder-layers 8 \
    --checkpoint-every 5 \
    --output production_chatbot.bin

# 4. Deploy best checkpoint
cp production_chatbot.bin.epoch35 deployed_model.bin
./chatbot vocab.txt deployed_model.bin
```

### Workflow 3: Hyperparameter Search

```bash
# Test different learning rates
for lr in 0.0001 0.00005 0.00001; do
    ./ChatbotTrainer \
        --data data.txt \
        --vocab vocab.txt \
        --epochs 10 \
        --lr $lr \
        --early-stopping \
        --patience 3 \
        --output model_lr_${lr}.bin
done

# Compare validation losses from summaries

```

## Related Documentation

- **EncoderDecoderModel:** `Context Documentation/ENCODERDECODERMODEL_CONTEXT.md`
- **Optimizer:** `Context Documentation/OPTIMIZER_CONTEXT.md`
- **BPETokenizer:** `Context Documentation/BPE_TOKENIZER_CONTEXT.md`
- **Training Examples:** `TRAINING_EXAMPLE.md`
- **Chatbot Quickstart:** `CHATBOT_TRAINER_QUICKSTART.md`
- **Chatbot README:** `CHATBOT_TRAINER_README.md`

## Summary

`ChatbotTrainer` provides a production-ready training harness for transformer-based chatbots with:

✅ **Complete Pipeline:** Vocabulary → Data → Training → Checkpointing → Testing
✅ **Advanced Optimization:** 4 optimizers, 6 LR schedules, gradient clipping, weight decay
✅ **Gradient Accumulation:** Simulate larger batch sizes without memory overhead (NEW)
✅ **Efficient Training:** Pre-tokenized data caching (10-100x speedup) (NEW)
✅ **Proper Validation:** Inference-only validation without weight contamination (NEW)
✅ **Data Augmentation:** Random shuffling per epoch and for splits (NEW)
✅ **Smart Features:** Early stopping, auto-checkpointing, config validation
✅ **Monitoring:** Loss tracking, gradient norms, validation metrics
✅ **Flexible CLI:** 32+ command-line options for complete control
✅ **Production Ready:** Robust error handling, colored output, comprehensive logging

Ideal For:

- Training conversational AI models
- Experimenting with transformer architectures
- Production chatbot deployment
- Research and education

**Key Strength:** Combines simplicity (single executable) with sophistication (state-of-art optimization techniques).

---

## Recent Improvements (January 2026)

### Performance Enhancements

#### 1. Data Preprocessing & Tokenization Caching

- All data is now pre-tokenized once before training begins
- Eliminates redundant tokenization in the training loop
- **Performance Impact:** 10-100x speedup in training iteration time
- Stores tokenized data in `TokenizedPair` structures

#### 2. Gradient Accumulation

- Simulates larger batch sizes by accumulating gradients over multiple samples
- Enables effective batch sizes larger than memory would allow
- Configurable via `--batch-size` and `--grad-accum` command-line options
- Properly scales gradients and learning rate updates
- **Example:** `--batch-size 4 --grad-accum 8` = effective batch size of 32

#### 3. Data Shuffling

- Random shuffling applied before train/validation split
- Per-epoch shuffling of training data
- Uses modern `std::shuffle` with `std::mt19937` random generator
- **Impact:** Better generalization and reduced overfitting

### Correctness Fixes

#### 4. Proper Validation

- Fixed critical bug where validation used `train_step()` (which updates weights)
- Now uses `model->evaluate()` for inference-only validation
- Sets model to evaluation mode during validation
- **Impact:** Validation metrics are now accurate and reliable

#### 5. Code Quality

- Removed redundant `optimizer->zero_grad()` call
- Fixed deprecated `std::random_shuffle()` usage
- Added `<random>` header for proper RNG
- Improved documentation and comments

### API Changes

New Command-Line Options:

- `--batch-size <n>` - Batch size for training (default: 1)
- `--grad-accum <n>` - Gradient accumulation steps (default: 1)

New Data Structures:

- `TokenizedPair` - Stores pre-tokenized sequences with original text

New Methods:

- `preprocess_data()` - Pre-tokenize all training/validation data
- `shuffle_training_data()` - Randomly shuffle training indices

Updated Methods:

- `split_data()` - Now uses random shuffling
- `train_epoch()` - Supports gradient accumulation and cached tokenization
- `validate()` - Proper inference-only validation

### Migration Notes

Existing training scripts should work without modification. The new features are opt-in via command-line arguments:

```bash
# Old style (still works)
./chatbot_trainer --data train.txt --vocab vocab.txt --epochs 10

# New style with improvements
./chatbot_trainer --data train.txt --vocab vocab.txt --epochs 10 \
    --batch-size 4 --grad-accum 8
```

All improvements are backward compatible and enabled by default (except gradient accumulation which defaults to 1).
