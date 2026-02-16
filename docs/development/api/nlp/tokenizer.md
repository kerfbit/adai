# BPETokenizer Class - Context Document

## Overview

The `BPETokenizer` class implements a Byte Pair Encoding (BPE) tokenizer, a subword tokenization algorithm widely used in natural language processing. BPE iteratively merges the most frequent pairs of characters or character sequences to create a vocabulary of subword units, enabling efficient text encoding while handling out-of-vocabulary words through subword decomposition.

## Purpose

The BPETokenizer class is designed to:

- Build subword vocabularies from text corpora
- Tokenize text into subword units for neural network processing
- Handle unknown words through subword decomposition
- Provide efficient text encoding and decoding
- Support special tokens for sequence modeling (padding, unknown, BOS, EOS)
- Enable vocabulary persistence through save/load functionality
- Integrate with transformer architectures and language models

## Architecture

### Core Components

```text
Text Input
     ↓
Pre-tokenization (regex pattern matching)
     ↓
BPE Algorithm (iterative pair merging)
     ↓
Subword Tokens
     ↓
Vocabulary Mapping
     ↓
Token IDs (integers)
```

### BPE Algorithm Flow

```text
1. Initialize with character vocabulary
2. Count adjacent character pair frequencies
3. Merge most frequent pair
4. Add merged pair to vocabulary
5. Update all occurrences in corpus
6. Repeat steps 2-5 for N iterations
7. Final vocabulary = characters + merged pairs
```

## Class Interface

### Private Members

```cpp
class BPETokenizer {
private:
    // Core data structures
    std::unordered_map<std::string, int> vocab;              // Token → ID
    std::unordered_map<int, std::string> inverse_vocab;      // ID → Token
    std::vector<std::pair<std::string, std::string>> bpe_merges;  // Merge rules
    std::unordered_set<std::string> special_tokens;          // Special token set

    // Special token IDs
    int pad_token_id = 0;    // <pad> for padding sequences
    int unk_token_id = 1;    // <unk> for unknown tokens
    int bos_token_id = 2;    // <bos> for beginning of sequence
    int eos_token_id = 3;    // <eos> for end of sequence

    // Pre-tokenization pattern
    std::regex token_pattern;  // Regex for word boundary detection
};
```

### Public Methods

#### Vocabulary Building

```cpp
/**
 * Build vocabulary from text corpus
 *
 * @param texts Vector of text documents
 * @param vocab_size Target vocabulary size
 * @param frequency_threshold Minimum character frequency
 */
void build_vocab(const std::vector<std::string>& texts,
                 int vocab_size = 10000,
                 int frequency_threshold = 1);
```

**Algorithm:**

1. Count character frequencies across corpus
2. Add frequent characters (≥ threshold) to vocabulary
3. Learn BPE merges to reach target vocabulary size
4. Progress reporting at each stage

#### BPE Merge Learning

```cpp
/**
 * Build BPE merge rules
 *
 * @param texts Vector of text documents
 * @param num_merges Number of merge operations to perform
 */
void build_bpe_merges(const std::vector<std::string>& texts,
                     int num_merges);
```

**Algorithm:**

1. Convert texts to character sequences
2. Find most frequent adjacent pair
3. Merge the pair throughout corpus
4. Add merge rule and update vocabulary
5. Repeat for num_merges iterations

#### Pre-tokenization

```cpp
/**
 * Pre-tokenization step using regex patterns
 *
 * @param text Input text string
 * @return Vector of pre-tokenized words
 */
std::vector<std::string> pre_tokenize(const std::string& text);
```

**Pattern:** `'s| 't | 're | 've | 'm | 'll | 'd | ?[A-Za-z]+ | ?[0-9]+ | ?[^\s\w]+ |\s+`

Matches:

- Contractions: 's, 't, 're, 've, 'm, 'll, 'd
- Words: sequences of letters (with optional leading space)
- Numbers: sequences of digits (with optional leading space)
- Punctuation: sequences of non-word characters
- Whitespace: sequences of spaces

#### BPE Application

```cpp
/**
 * Apply BPE encoding to a single word
 *
 * @param word Input word
 * @return Vector of subword tokens
 */
std::vector<std::string> apply_bpe(const std::string& word);
```

**Algorithm:**

1. Initialize with character-level tokens
2. Apply each merge rule in order
3. Greedily merge matching pairs
4. Return final subword sequence

#### Tokenization

```cpp
/**
 * Tokenize text into subword tokens
 *
 * @param text Input text
 * @return Vector of subword tokens
 */
std::vector<std::string> tokenize(const std::string& text);
```

**Process:**

1. Pre-tokenize into words
2. Apply BPE to each word
3. Concatenate all subword tokens

#### Encoding

```cpp
/**
 * Convert text to token IDs
 *
 * @param text Input text
 * @param add_special_tokens Whether to add BOS/EOS tokens
 * @return Vector of token IDs
 */
std::vector<int> encode(const std::string& text,
                       bool add_special_tokens = true);
```

**Format:** `[BOS, token_ids..., EOS]` (if add_special_tokens=true)

#### Decoding

```cpp
/**
 * Convert token IDs back to text
 *
 * @param ids Vector of token IDs
 * @param skip_special_tokens Whether to exclude special tokens
 * @return Decoded text string
 */
std::string decode(const std::vector<int>& ids,
                  bool skip_special_tokens = true);
```

#### Serialization

```cpp
/**
 * Save vocabulary to file
 *
 * @param filename Output file path
 */
void save_vocab(const std::string& filename) const;

/**
 * Load vocabulary from file
 *
 * @param filename Input file path
 */
void load_vocab(const std::string& filename);
```

**File Format:**

```text
# BPE Tokenizer Vocabulary v1.0
VOCAB_SIZE <size>
SPECIAL_TOKENS
pad_token_id <id>
unk_token_id <id>
bos_token_id <id>
eos_token_id <id>
VOCAB
<escaped_token>\t<id>
...
BPE_MERGES <count>
<escaped_first>\t<escaped_second>
...
```

#### Utility Methods

```cpp
size_t get_vocab_size() const;
void print_vocab_stats() const;
std::vector<std::pair<std::string, int>> get_top_tokens(int k = 10) const;
```

## Special Tokens

### Token Definitions

| Token | ID | Purpose | Usage |
| ------- | ----- | --------- | ------- |
| `<pad>` | 0 | Padding | Fill sequences to uniform length |
| `<unk>` | 1 | Unknown | Replace out-of-vocabulary tokens |
| `<bos>` | 2 | Begin | Mark sequence start |
| `<eos>` | 3 | End | Mark sequence end |

### Special Token Handling

```cpp
// Encoding with special tokens
auto ids = tokenizer.encode("Hello world", true);
// Result: [2, <token_ids>, 3]  // BOS, tokens, EOS

// Decoding without special tokens
auto text = tokenizer.decode(ids, true);
// Result: "hello world"  // No BOS/EOS in output
```

## BPE Algorithm Details

### Pair Frequency Counting

```cpp
std::pair<std::string, std::string> get_most_frequent_pair(
    const std::vector<std::vector<std::string>>& word_tokens);
```

**Algorithm:**

```text
For each word in corpus:
    For each adjacent pair (token_i, token_i+1):
        pair_key = token_i + "|  |  |" + token_i+1
        count[pair_key]++

Return pair with maximum count
```

### Merge Application

```cpp
void merge_tokens(std::vector<std::vector<std::string>>& word_tokens,
                 const std::string& first, const std::string& second);
```

**Algorithm:**

```text
For each word in corpus:
    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == first and tokens[i+1] == second:
            new_tokens.append(first + second)
            i += 2  // Skip next token
        else:
            new_tokens.append(tokens[i])
            i += 1
    tokens = new_tokens
```

## Usage Examples

### Example 1: Building Vocabulary

```cpp
#include "BPETokenizer.hpp"

int main() {
    BPETokenizer tokenizer;

    // Training corpus
    std::vector<std::string> texts = {
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question."
    };

    // Build vocabulary with 1000 tokens
    tokenizer.build_vocab(texts, 1000, 1);

    // Save for later use
    tokenizer.save_vocab("tokenizer.vocab");

    return 0;
}
```

**Output:**

```text
[BPE Tokenizer] Building vocabulary...
[1/3] Counting character frequencies... 3/3 texts processed (150 total characters)
[2/3] Building base vocabulary... Added 45 characters
[3/3] Learning BPE merges (target: 955 merges)...
    Tokenizing text corpus... 42 word tokens
    Merge 955/955 (100.0%) - Latest: 'th' + 'e' → 'the'
[BPE Tokenizer] Vocabulary built successfully! Final size: 1000 tokens
[BPE Tokenizer] Vocabulary saved to tokenizer.vocab
```

### Example 2: Text Encoding

```cpp
#include "BPETokenizer.hpp"

int main() {
    BPETokenizer tokenizer;
    tokenizer.load_vocab("tokenizer.vocab");

    // Encode text
    std::string text = "Hello, world!";
    auto ids = tokenizer.encode(text, true);

    std::cout << "Text: " << text << std::endl;
    std::cout << "Token IDs: [";
    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << ids[i];
        if (i < ids.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Decode back
    auto decoded = tokenizer.decode(ids, true);
    std::cout << "Decoded: " << decoded << std::endl;

    return 0;
}
```

**Output:**

```text
Text: Hello, world!
Token IDs: [2, 15, 42, 87, 23, 19, 45, 3]
Decoded: hello, world!
```

### Example 3: Tokenization Analysis

```cpp
#include "BPETokenizer.hpp"

int main() {
    BPETokenizer tokenizer;
    tokenizer.load_vocab("tokenizer.vocab");

    std::string text = "preprocessing";

    // Get tokens
    auto tokens = tokenizer.tokenize(text);

    std::cout << "Word: " << text << std::endl;
    std::cout << "Subword tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "'" << tokens[i] << "'";
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    return 0;
}
```

**Output:**

```text
Word: preprocessing
Subword tokens: ['pre', 'process', 'ing']
```

### Example 4: Vocabulary Statistics

```cpp
#include "BPETokenizer.hpp"

int main() {
    BPETokenizer tokenizer;
    tokenizer.load_vocab("tokenizer.vocab");

    // Print statistics
    tokenizer.print_vocab_stats();

    // Get most common tokens
    auto top_tokens = tokenizer.get_top_tokens(10);

    std::cout << "\nTop 10 tokens:" << std::endl;
    for (const auto& [token, id] : top_tokens) {
        std::cout << "  ID " << id << ": '" << token << "'" << std::endl;
    }

    return 0;
}
```

**Output:**

```text
Vocabulary size: 1000
Number of BPE merges: 955
Special tokens: 4

Top 10 tokens:
  ID 0: '<pad>'
  ID 1: '<unk>'
  ID 2: '<bos>'
  ID 3: '<eos>'
  ID 4: 'a'
  ID 5: 'b'
  ID 6: 'c'
  ID 7: 'd'
  ID 8: 'e'
  ID 9: 'f'
```

### Example 5: Integration with Neural Network

```cpp
#include "BPETokenizer.hpp"
#include "NeuralNetwork.hpp"

int main() {
    BPETokenizer tokenizer;
    tokenizer.load_vocab("tokenizer.vocab");

    // Create embedding layer
    int vocab_size = tokenizer.get_vocab_size();
    int embedding_dim = 128;

    // Text classification network
    std::vector<int> architecture = {embedding_dim, 64, 3};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };

    NeuralNetwork classifier(architecture, activations,
                             LossType::CATEGORICAL_CROSS_ENTROPY);

    // Encode text
    std::string text = "This is a positive review!";
    auto token_ids = tokenizer.encode(text, false);

    // Create simple bag-of-words embedding
    std::vector<float> features(embedding_dim, 0.0f);
    for (int id : token_ids) {
        if (id < embedding_dim) {
            features[id] += 1.0f;
        }
    }

    // Classify
    auto prediction = classifier.predict(features);

    std::cout << "Text: " << text << std::endl;
    std::cout << "Prediction: [" << prediction[0] << ", "
              << prediction[1] << ", " << prediction[2] << "]" << std::endl;

    return 0;
}
```

## Implementation Details

### Pre-tokenization Pattern

```cpp
std::regex token_pattern(
    R"('s| 't | 're | 've | 'm | 'll | 'd | ?[A-Za-z]+ | ?[0-9]+ | ?[^\s\w]+ |\s+)"
);
```

**Pattern Components:**

- `'s|'t|'re|'ve|'m|'ll|'d` - Common contractions
- `?[A-Za-z]+` - Words (with optional leading space)
- `?[0-9]+` - Numbers (with optional leading space)
- `?[^\s\w]+` - Punctuation (with optional leading space)
- `\s+` - Whitespace sequences

### Character Escaping

Special characters are escaped in save/load:

- Newline: `\n` → `\\n`
- Tab: `\t` → `\\t`
- Carriage return: `\r` → `\\r`
- Backslash: `\\` → `\\\\`
- Space: ` ` → `\\s`

### Progress Reporting

The implementation includes detailed progress reporting:

```cpp
// Character frequency counting
"[1/3] Counting character frequencies... 100/100 texts processed"

// Base vocabulary building
"[2/3] Building base vocabulary... Added 45 characters"

// BPE merge learning
"[3/3] Learning BPE merges (target: 955 merges)..."
"    Merge 100/955 (10.5%) - Latest: 't' + 'h' → 'th'"
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
| ----------- | ------------ | ------- |
| build_vocab | O(N × M + K²) | N=texts, M=avg_length, K=merges |
| build_bpe_merges | O(K × T × L) | K=merges, T=tokens, L=token_length |
| pre_tokenize | O(N) | N=text_length |
| apply_bpe | O(M × L) | M=merges, L=word_length |
| tokenize | O(N × M × L) | N=words, M=merges, L=word_length |
| encode | O(T) | T=number_of_tokens |
| decode | O(T) | T=number_of_tokens |
| save_vocab | O(V) | V=vocab_size |
| load_vocab | O(V) | V=vocab_size |

### Space Complexity

| Component | Space | Notes |
| ----------- | ------- | ------- |
| vocab | O(V) | V=vocab_size |
| inverse_vocab | O(V) | Mirror of vocab |
| bpe_merges | O(M) | M=number_of_merges |
| special_tokens | O(1) | Fixed 4 tokens |
| **Total** | **O(V + M)** | Dominated by vocabulary |

### Memory Usage Estimates

For a typical configuration:

- Vocabulary size: 10,000 tokens
- Average token length: 5 characters
- BPE merges: ~9,996

**Memory:**

- vocab: ~10,000 × (5 + 8) bytes = ~130 KB
- inverse_vocab: ~10,000 × (8 + 5) bytes = ~130 KB
- bpe_merges: ~9,996 × 2 × 5 bytes = ~100 KB
- **Total: ~360 KB**

## Optimization Strategies

### 1. Pre-allocation

```cpp
std::vector<std::string> pre_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    tokens.reserve(text.length() / 5);  // Estimate average word length
    // ... tokenization logic
    return tokens;
}
```

### 2. Efficient Pair Counting

```cpp
// Use string concatenation for pair key
std::string pair_key = tokens[i] + "|  |  |" + tokens[i + 1];
pair_counts[pair_key]++;
```

### 3. In-place Merging

```cpp
// Reuse vectors instead of creating new ones
tokens = new_tokens;
```

### 4. Progress Batching

```cpp
// Update progress every 100 operations instead of every operation
if ((i + 1) % 100 == 0) {
    std::cout << "\r    Processed " << (i + 1) << " merges..." << std::flush;
}
```

## File Format Specification

### Vocabulary File Structure

```text
# BPE Tokenizer Vocabulary v1.0
VOCAB_SIZE 1000
SPECIAL_TOKENS
pad_token_id 0
unk_token_id 1
bos_token_id 2
eos_token_id 3
VOCAB
<pad>    0
<unk>    1
<bos>    2
<eos>    3
a    4
b    5
th    45
the    120
...
BPE_MERGES 955
t    h
th    e
a    n
...
```

### Escape Sequences

| Character | Escaped Form | Example |
| ----------- | -------------- | --------- |
| Space | `\s` | `hello\sworld` |
| Newline | `\n` | `line1\nline2` |
| Tab | `\t` | `col1\tcol2` |
| Return | `\r` | `text\r` |
| Backslash | `\\` | `path\\to\\file` |

## Integration Patterns

### Pattern 1: Text Preprocessing Pipeline

```cpp
class TextPreprocessor {
private:
    BPETokenizer tokenizer;

public:
    void train(const std::vector<std::string>& texts) {
        // Normalize text
        auto normalized = normalize(texts);

        // Build vocabulary
        tokenizer.build_vocab(normalized, 10000);
        tokenizer.save_vocab("vocab.txt");
    }

    std::vector<int> process(const std::string& text) {
        return tokenizer.encode(text, true);
    }

private:
    std::vector<std::string> normalize(
        const std::vector<std::string>& texts) {
        // Lowercase, remove extra spaces, etc.
        return texts;  // Simplified
    }
};
```

### Pattern 2: Batch Encoding

```cpp
std::vector<std::vector<int>> encode_batch(
    const BPETokenizer& tokenizer,
    const std::vector<std::string>& texts,
    int max_length = 128) {

    std::vector<std::vector<int>> batch;

    for (const auto& text : texts) {
        auto ids = tokenizer.encode(text, true);

        // Truncate or pad to max_length
        if (ids.size() > max_length) {
            ids.resize(max_length);
        } else {
            while (ids.size() < max_length) {
                ids.push_back(0);  // pad_token_id
            }
        }

        batch.push_back(ids);
    }

    return batch;
}
```

### Pattern 3: Vocabulary Expansion

```cpp
void expand_vocabulary(BPETokenizer& tokenizer,
                      const std::vector<std::string>& new_tokens) {
    int current_size = tokenizer.get_vocab_size();

    for (const auto& token : new_tokens) {
        // Add new token to vocabulary
        // (requires friend access or public method)
    }

    tokenizer.save_vocab("expanded_vocab.txt");
}
```

## Common Use Cases

### 1. Language Model Tokenization

```cpp
BPETokenizer tokenizer;
tokenizer.load_vocab("lm_vocab.txt");

std::string text = "The transformer architecture revolutionized NLP.";
auto ids = tokenizer.encode(text, true);

// Feed to language model
// auto logits = language_model.forward(ids);
```

### 2. Machine Translation

```cpp
BPETokenizer src_tokenizer, tgt_tokenizer;
src_tokenizer.load_vocab("en_vocab.txt");
tgt_tokenizer.load_vocab("fr_vocab.txt");

std::string english = "Hello, how are you?";
auto src_ids = src_tokenizer.encode(english, true);

// Translation
// auto tgt_ids = translator.translate(src_ids);
// auto french = tgt_tokenizer.decode(tgt_ids, true);
```

### 3. Text Classification

```cpp
BPETokenizer tokenizer;
tokenizer.load_vocab("classifier_vocab.txt");

std::vector<std::string> reviews = {
    "Great product, highly recommend!",
    "Terrible quality, waste of money."
};

for (const auto& review : reviews) {
    auto ids = tokenizer.encode(review, false);
    // auto sentiment = classifier.predict(ids);
}
```

## Best Practices

### 1. Vocabulary Size Selection

```cpp
// Small vocabulary (1K-5K): Limited data, specific domain
tokenizer.build_vocab(texts, 2000);

// Medium vocabulary (10K-30K): General purpose, balanced
tokenizer.build_vocab(texts, 15000);

// Large vocabulary (50K+): Large datasets, complex language
tokenizer.build_vocab(texts, 50000);
```

### 2. Frequency Threshold

```cpp
// Low threshold (1-2): Keep rare characters
tokenizer.build_vocab(texts, 10000, 1);

// Medium threshold (5-10): Filter noise
tokenizer.build_vocab(texts, 10000, 5);

// High threshold (20+): Only common characters
tokenizer.build_vocab(texts, 10000, 20);
```

### 3. Special Token Usage

```cpp
// Training: Don't add BOS/EOS
auto train_ids = tokenizer.encode(text, false);

// Generation: Add BOS, generate until EOS
auto input_ids = tokenizer.encode(prompt, true);
// auto generated = model.generate(input_ids);

// Classification: Add both for consistency
auto class_ids = tokenizer.encode(text, true);
```

### 4. Vocabulary Persistence

```cpp
// Save after training
tokenizer.build_vocab(train_texts, 10000);
tokenizer.save_vocab("model_vocab.txt");

// Load for inference
BPETokenizer inference_tokenizer;
inference_tokenizer.load_vocab("model_vocab.txt");
```

## Testing Strategy

### Unit Tests

```cpp
TEST(BPETokenizerTest, PreTokenization) {
    BPETokenizer tokenizer;
    auto tokens = tokenizer.pre_tokenize("Hello, world!");

    EXPECT_GT(tokens.size(), 0);
    EXPECT_TRUE(std::find(tokens.begin(), tokens.end(), "hello")
                != tokens.end());
}

TEST(BPETokenizerTest, EncodeDecode) {
    BPETokenizer tokenizer;
    tokenizer.load_vocab("test_vocab.txt");

    std::string text = "test input";
    auto ids = tokenizer.encode(text, true);
    auto decoded = tokenizer.decode(ids, true);

    EXPECT_EQ(decoded, "test input");
}

TEST(BPETokenizerTest, SpecialTokens) {
    BPETokenizer tokenizer;

    auto ids = tokenizer.encode("hello", true);

    EXPECT_EQ(ids.front(), 2);  // BOS
    EXPECT_EQ(ids.back(), 3);   // EOS
}
```

## Future Enhancements

1. **Byte-level BPE**: Handle any Unicode character
2. **SentencePiece Integration**: Language-agnostic tokenization
3. **WordPiece Algorithm**: Alternative subword method
4. **Vocabulary Pruning**: Remove low-frequency tokens
5. **Multi-threading**: Parallel vocabulary building
6. **Caching**: Cache tokenization results
7. **Streaming**: Process large files incrementally
8. **Token Statistics**: Frequency analysis tools

## References

- **BPE Paper**: Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016)
- **GPT-2 BPE**: Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
- **BERT WordPiece**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018)

## Related Components

- **Encoder**: Uses BPETokenizer for text encoding
- **NeuralNetwork**: Processes tokenized inputs
- **Transformer**: Complete architecture with tokenization

## Notes

- Text is automatically converted to lowercase during pre-tokenization
- Unknown tokens are mapped to `<unk>` (ID 1)
- Vocabulary building shows progress for large corpora
- Merge order matters: earlier merges take precedence
- File format supports special character escaping
- Thread-safe after vocabulary is built (read-only operations)
