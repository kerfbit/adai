# Project Gutenberg Training Integration

## Overview

The incremental trainer now supports automatic downloading and training from Project Gutenberg's collection of over 70,000 free books. This allows you to easily add diverse, high-quality text data to improve your chatbot's language understanding.

## Quick Start

### Download a Single Book

```bash
# Download Pride and Prejudice and create 500 training pairs
./incremental_trainer gutenberg 1342 500

# Train on it
./incremental_trainer train 5
```

### Download Multiple Books

```bash
# Download and add multiple classic books at once
./incremental_trainer gutenberg-batch 1342,11,84,1661 300

# Each book creates 300 training pairs
./incremental_trainer train 10
```

## Popular Books for Training

### Classic Literature (Formal English)

- **1342** - Pride and Prejudice (Jane Austen) - Formal dialogue, social interactions
- **1661** - Sherlock Holmes Adventures - Logical reasoning, mysteries
- **11** - Alice in Wonderland - Creative, imaginative language
- **84** - Frankenstein - Gothic, philosophical discussions
- **98** - A Tale of Two Cities - Historical narrative
- **1260** - Jane Eyre - First-person narrative, emotional depth

### Adventure & Fantasy

- **2701** - Moby Dick - Descriptive prose, maritime terminology
- **76** - Adventures of Huckleberry Finn - Conversational American English
- **345** - Dracula - Suspense, epistolary format
- **16328** - Beowulf - Epic poetry, heroic language

### Science & Philosophy

- **829** - Gulliver's Travels - Satire, political commentary
- **35** - The Time Machine (H.G. Wells) - Science fiction concepts
- **1952** - The Yellow Wallpaper - Psychological narrative
- **158** - Emma (Jane Austen) - Social dynamics, matchmaking

### Diverse Styles

- **1952** - The Yellow Wallpaper - Stream of consciousness
- **2554** - Crime and Punishment - Psychological thriller
- **4300** - Ulysses - Modernist, experimental
- **1184** - The Count of Monte Cristo - Adventure, revenge

## How It Works

### 1. Download Process

```cpp
trainer.add_gutenberg_book(1342, 500);
```

This:

1. Downloads the book from `https://www.gutenberg.org/files/1342/1342-0.txt`
2. Saves to `gutenberg_data/gutenberg_1342.txt`
3. Automatically falls back to ASCII if UTF-8 unavailable

### 2. Text Processing

The system automatically:

- Removes Project Gutenberg headers and footers
- Extracts valid sentences (20-500 characters)
- Cleans excessive whitespace
- Filters out very short or malformed sentences

### 3. Training Pair Generation

Creates diverse conversation pairs:

**Question Style:**

```text
INPUT: What does this mean: "It is a truth universally acknowledged..."
RESPONSE: "...that a single man in possession of a good fortune must be in want of a wife."
```

**Explanation Style:**

```text
INPUT: Can you explain: "The rain fell in torrents..."
RESPONSE: "...except at occasional intervals, when it was checked by a violent gust of wind."
```

**Summary Style:**

```text
INPUT: Summarize: "Elizabeth felt all the impertinence of her questions..."
RESPONSE: "...but answered them very composedly."
```

## Recommended Book Combinations

### For General Conversation

```bash
./incremental_trainer gutenberg-batch 1342,11,76,98 400
```

- Mix of dialogue-heavy books
- Variety of speaking styles
- ~1600 training pairs total

### For Formal/Professional Language

```bash
./incremental_trainer gutenberg-batch 1661,84,1260,2701 300
```

- Sherlock Holmes (analytical)
- Frankenstein (scientific)
- Jane Eyre (professional correspondence)
- Moby Dick (technical descriptions)

### For Creative/Imaginative Responses

```bash
./incremental_trainer gutenberg-batch 11,345,35,16328 500
```

- Alice in Wonderland (whimsical)
- Dracula (dramatic)
- Time Machine (speculative)
- Beowulf (epic)

## Best Practices

### 1. Start Small

```bash
# First, test with one book
./incremental_trainer gutenberg 1342 200
./incremental_trainer train 3

# If quality is good, add more
./incremental_trainer gutenberg-batch 11,84,1661 300
./incremental_trainer train 5
```

### 2. Mix with Real Conversation Data

```bash
# Combine Gutenberg books with actual conversations
./incremental_trainer add real_conversations.txt
./incremental_trainer gutenberg 1342 300
./incremental_trainer train 5
```

### 3. Control Training Pair Count

More pairs = more training time but more diversity:

- **100-200 pairs**: Quick addition, specific style
- **300-500 pairs**: Balanced (recommended)
- **500-1000 pairs**: Comprehensive coverage

```bash
# Quick style boost
./incremental_trainer gutenberg 1342 100

# Comprehensive training
./incremental_trainer gutenberg 2701 1000
```

### 4. Periodic Full Retrains

After adding several books incrementally, do a full retrain:

```bash
# Add 5 books incrementally
for id in 1342 11 84 1661 2701; do
    ./incremental_trainer gutenberg $id 300
    ./incremental_trainer train 5
done

# Then full retrain to integrate everything
./incremental_trainer retrain 10
```

## Finding More Books

### Browse by Category

Visit: <https://www.gutenberg.org/ebooks/>

Categories:

- Fiction: Popular, Adventure, Mystery, Romance, Science Fiction
- Non-Fiction: History, Philosophy, Science, Biography
- Reference: Dictionaries, Encyclopedias

### Get Book ID from URL

Example: `https://www.gutenberg.org/ebooks/1342`
Book ID: **1342**

### Search

<https://www.gutenberg.org/ebooks/search/?query=shakespeare>

## Advanced Usage

### Custom Pair Generation

Modify `num_pairs` based on book length:

- Short stories: 100-200 pairs
- Novellas: 300-500 pairs  
- Novels: 500-1000 pairs
- Epic works: 1000-2000 pairs

### Batch Processing Script

Create a training plan:

```bash
#!/bin/bash
# classics.sh - Train on classic literature

# Week 1: British classics
./incremental_trainer gutenberg-batch 1342,1260,98,84 400
./incremental_trainer train 5

# Week 2: American classics  
./incremental_trainer gutenberg-batch 76,2701,11 400
./incremental_trainer train 5

# Week 3: Mystery & Adventure
./incremental_trainer gutenberg-batch 1661,345,829 400
./incremental_trainer train 5

# Week 4: Full retrain
./incremental_trainer retrain 10
```

## Troubleshooting

### "Failed to download"

- Book might not exist
- Check book ID at gutenberg.org
- Some books are copyright-restricted
- Try fallback: manually download and use `add` instead

### "No valid sentences found"

- Book might be poetry or non-prose
- Try different book
- Check if book is in English

### Low Quality Output

- Gutenberg text alone isn't enough for conversation
- Mix with real dialogue data
- Use books with lots of dialogue (Jane Austen, Arthur Conan Doyle)

### Too Slow

- Reduce `num_pairs` parameter
- Download books once, reuse training data files
- Files saved in `gutenberg_data/` can be reused

## Performance Impact

### Download Time

- ~5-30 seconds per book (depends on size)
- Network speed dependent
- Downloaded files are cached

### Processing Time

- Sentence extraction: ~1-5 seconds
- Pair generation: ~1-3 seconds
- Total per book: ~10-40 seconds

### Training Impact

| Pairs Added | Training Time (5 epochs) | Model Improvement      |
| ----------- | ------------------------ | ---------------------- |
| 100         | ~30 minutes              | Minor style boost      |
| 300         | ~1.5 hours               | Noticeable improvement |
| 500         | ~2.5 hours               | Significant boost      |
| 1000        | ~5 hours                 | Major enhancement      |

## Examples

### Train on Shakespeare

```bash
# Multiple Shakespeare works
./incremental_trainer gutenberg-batch 1533,1534,1535 500
./incremental_trainer train 10
```

### Science Fiction Focus

```bash
# H.G. Wells collection
./incremental_trainer gutenberg-batch 35,36,159,5230 400
./incremental_trainer train 8
```

### Mystery & Detective

```bash
# Sherlock Holmes & Poe
./incremental_trainer gutenberg-batch 1661,2147,2148,932 350
./incremental_trainer train 7
```

## Integration with Existing Workflow

```bash
# 1. Start with your own conversations
./incremental_trainer add my_conversations.txt
./incremental_trainer train 10

# 2. Enhance with literary style
./incremental_trainer gutenberg 1342 300
./incremental_trainer train 5

# 3. Add more variety
./incremental_trainer gutenberg-batch 11,84,1661 200
./incremental_trainer train 5

# 4. Periodic comprehensive retrain
./incremental_trainer retrain 10
```

## API Usage

```cpp
#include "IncrementalTrainer.hpp"

IncrementalTrainer trainer("vocab.txt", "model.bin");

// Add single book
trainer.add_gutenberg_book(1342, 500);

// Add multiple books
std::vector<int> book_ids = {1342, 11, 84, 1661};
trainer.add_gutenberg_books(book_ids, 300);

// Train
trainer.train_incremental(5);
```

## Tips for Best Results

1. **Start with dialogue-heavy books** (Austen, Conan Doyle)
2. **Mix genres** for versatility
3. **Don't over-train on one style** - variety is key
4. **Combine with real conversation data** for best quality
5. **Use fewer pairs per book** to cover more books
6. **Monitor validation loss** - if it increases, reduce Gutenberg proportion
