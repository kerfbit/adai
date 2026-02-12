# BPE Tokenizer - Save and Load Vocabulary Example

## Overview

The BPE Tokenizer now supports saving and loading built vocabularies, allowing you to:

- Build a vocabulary once and reuse it across multiple sessions
- Share vocabularies between different programs
- Avoid rebuilding vocabularies which can be time-consuming for large corpora

## Features

The save/load functionality preserves:

- **Vocabulary mappings** (token → ID)
- **Inverse vocabulary** (ID → token)
- **BPE merge rules** (ordered list of merge operations)
- **Special tokens** (pad, unk, bos, eos and their IDs)

## Usage

### Building and Saving a Vocabulary

```cpp
#include "BPETokenizer.hpp"

// Create and build tokenizer
BPETokenizer tokenizer;

// Load training data
std::vector<std::string> training_texts;
// ... load your training data ...

// Build vocabulary
tokenizer.build_vocab(training_texts, 10000);  // Build 10k vocab

// Save vocabulary to file
tokenizer.save_vocab("my_vocab.txt");
```

### Loading a Pre-built Vocabulary

```cpp
#include "BPETokenizer.hpp"

// Create a new tokenizer
BPETokenizer tokenizer;

// Load previously saved vocabulary
tokenizer.load_vocab("my_vocab.txt");

// Now use the tokenizer immediately without rebuilding
auto tokens = tokenizer.encode("Hello world!");
```

### Complete Example

```cpp
#include "BPETokenizer.hpp"
#include <iostream>

int main() {
    // === First run: Build and save ===
    BPETokenizer builder;
    std::vector<std::string> training_data = {
        "The quick brown fox",
        "jumps over the lazy dog",
        // ... more training data ...
    };

    builder.build_vocab(training_data, 5000);
    builder.save_vocab("my_vocab.txt");

    // === Later runs: Just load ===
    BPETokenizer tokenizer;
    tokenizer.load_vocab("my_vocab.txt");

    // Use immediately
    std::string text = "Hello, world!";
    auto ids = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(ids);

    std::cout << "Original: " << text << std::endl;
    std::cout << "Decoded:  " << decoded << std::endl;

    return 0;
}
```

## File Format

The saved vocabulary file uses a structured text format:

``` text
# BPE Tokenizer Vocabulary v1.0
VOCAB_SIZE 10000
SPECIAL_TOKENS
pad_token_id 0
unk_token_id 1
bos_token_id 2
eos_token_id 3
VOCAB
token1 id1
token2 id2
...
BPE_MERGES 9931
first1 second1
first2 second2
...
```

### Special Character Escaping

Special characters in tokens are escaped for proper storage:

- Space: `\s`
- Newline: `\n`
- Tab: `\t`
- Carriage return: `\r`
- Backslash: `\\`

## Benefits

1. **Performance**: Build vocabulary once, use many times
2. **Consistency**: Same vocabulary across different sessions and programs
3. **Portability**: Text-based format is human-readable and version-controllable
4. **Completeness**: All necessary data is preserved (vocab + merges + special tokens)

## Testing

Run the example program to see save/load in action:

```bash
cd build
./src/tokenizer
```

The output will show:

- Building vocabulary from training data
- Saving to `vocab.txt`
- Loading into a new tokenizer
- Verification that both tokenizers produce identical results
