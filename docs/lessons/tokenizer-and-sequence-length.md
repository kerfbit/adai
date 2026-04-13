# Tokenizer and Sequence Length

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

The tokenizer is the interface between raw text and the model. It determines what the model sees, how efficiently it sees it, and what it is incapable of representing at all. Sequence length determines how much context the model can attend to at once and is the dominant driver of memory cost in transformer training.

Both decisions are made before training begins and are expensive to change afterward. A model trained with one tokenizer cannot be directly combined with weights trained on another.

---

## What a Tokenizer Does

A tokenizer maps a string of text to a sequence of integer token IDs, and maps those IDs back to text. The mapping is defined by a fixed **vocabulary** — a lookup table of subword units learned from a representative corpus.

The three dominant tokenization algorithms for language models:

| Algorithm | How it works | Used by |
| --- | --- | --- |
| **BPE** (Byte-Pair Encoding) | Iteratively merges the most frequent adjacent byte/character pairs | GPT-2, GPT-4, Llama, RoBERTa |
| **WordPiece** | Merges pairs that maximize likelihood of the training corpus | BERT, DistilBERT |
| **Unigram / SentencePiece** | Probabilistic; starts with a large vocabulary and prunes | T5, ALBERT, multilingual models |

All three produce subword tokenizations — common words become single tokens; rare words are split into multiple subword pieces. This balances vocabulary coverage against sequence length.

---

## Vocabulary Size

The vocabulary size $|V|$ is a hyperparameter with cascading effects:

**Memory**: the embedding matrix has shape $|V| \times d_{model}$ and the output (lm_head) projection has shape $d_{model} \times |V|$. For $|V| = 50 000$ and $d_{model} = 1024$, these two matrices consume ~400 MB at fp32 — a significant fraction of a small model's total parameters.

**Fertility**: the average number of tokens per word. A larger vocabulary encodes common words as single tokens, reducing fertility and therefore sequence length. A smaller vocabulary increases fertility and sequence length, consuming more context window.

**Coverage**: a vocabulary trained on English web text will fragment tokens from code, mathematics, or other languages into many pieces, multiplying sequence length for those domains and reducing model efficiency.

**Typical vocabulary sizes**:

| Model family | Vocabulary size |
| --- | --- |
| BERT | 30 522 |
| GPT-2 | 50 257 |
| Llama 2 | 32 000 |
| Llama 3 | 128 000 |
| GPT-4 (estimated) | ~100 000 |

Larger vocabularies generally benefit multilingual models and models expected to handle code or technical content. Smaller vocabularies are appropriate for narrow-domain models.

---

## Domain Match

A tokenizer trained on a corpus that does not match the training domain will produce poor fertility for out-of-domain content. Signs of tokenizer-domain mismatch:

- Code files tokenize to 3–5× more tokens than prose of equivalent information content
- Mathematical notation is split into single-character tokens
- A language makes up a significant portion of the training data but was underrepresented in tokenizer training

If the target domain differs substantially from the tokenizer's training corpus, consider **extending the vocabulary** with domain-specific tokens rather than training a new tokenizer from scratch. Most frameworks support vocabulary extension; the new embeddings are initialized randomly and learned during training.

---

## Sequence Length

The **maximum sequence length** (context window) is the longest token sequence the model can process in a single forward pass. It is set at training time and cannot be extended without retraining or applying position embedding extrapolation techniques.

Memory cost for standard (full) attention scales **quadratically** with sequence length:

$$\text{Attention memory} \propto L^2$$

where $L$ is the sequence length. Doubling the context length quadruples the memory cost of the attention computation. This is the primary reason context windows are a constrained resource.

**Typical context lengths**:

| Model | Context length |
| --- | --- |
| BERT | 512 |
| GPT-2 | 1 024 |
| Llama 2 | 4 096 |
| Llama 3 | 8 192 |
| GPT-4 | 128 000 |

---

## Choosing Sequence Length

The right sequence length is the shortest length that captures the dependencies your task requires.

- **Classification and short-form tasks**: 512–1 024 tokens is usually sufficient.
- **Document understanding, summarization**: 2 048–8 192 tokens.
- **Long-document reasoning, code generation over large files**: 16 K+ tokens.

Training at long contexts when most examples are short wastes memory. A better approach is **sequence packing**: concatenate multiple short documents into a single training sequence up to the maximum length, separated by a special end-of-document token. This maximizes GPU utilization and avoids padding waste.

---

## Padding and Packing

**Padding** extends short sequences to the maximum length with a special `[PAD]` token. The attention mask prevents the model from attending to padding positions. This approach is simple but wasteful — padding tokens consume memory and compute without contributing to learning.

**Packing** concatenates documents to fill each sequence slot:

``` text
[doc_A tokens][EOS][doc_B tokens][EOS][doc_C tokens...][PAD if needed]
```

Packing dramatically improves throughput when the average document length is much shorter than the maximum sequence length — common in fine-tuning on instruction datasets or short question-answer pairs.

**Caution with packing**: ensure the attention mask prevents cross-document attention, or use a causal mask with document boundary tokens that the model learns to treat as separators. Leaking attention across document boundaries introduces subtle training noise.

---

## Position Embeddings and Length Generalization

The model's ability to generalize to sequences longer than those seen during training depends on the position embedding scheme:

| Scheme | Extrapolation beyond training length |
| --- | --- |
| Learned absolute positions | Poor — positions beyond training length have no learned embedding |
| Sinusoidal | Moderate — mathematically defined beyond training length but degrades |
| RoPE (Rotary) | Good — relative distance encoding transfers better to longer sequences |
| ALiBi | Good — attention bias is a simple function of distance, extrapolates naturally |

If long-context capability is important, prefer **RoPE** or **ALiBi** over learned absolute positions. Training on a mix of short and long sequences also improves length generalization.

---

## Tokenizer Reproducibility

The tokenizer must be saved alongside the model weights. A model evaluated or deployed with a different tokenizer than the one used during training will produce incorrect results — the token ID mapping will be inconsistent even if the vocabulary strings appear similar.

Always save:

- The vocabulary file
- The merges file (BPE) or sentencepiece model
- Special token definitions (`[PAD]`, `[EOS]`, `[BOS]`, `[UNK]`)
- Normalization and pre-tokenization settings (lowercasing, whitespace handling)

---

## Common Mistakes

- **Using a mismatched tokenizer at inference.** Even a tokenizer with the same vocabulary size but different merge order will produce different token IDs and corrupt the model's output.
- **Setting sequence length longer than necessary** to be safe. This multiplies memory cost quadratically and reduces the effective batch size, slowing training.
- **Padding-heavy batches without packing.** If average document length is 25% of max sequence length, 75% of compute is spent on padding.
- **Forgetting to handle special tokens consistently.** If `[EOS]` was appended during training but not at inference, the model never sees a proper stopping signal.
- **Training a tokenizer on the model's training data.** The tokenizer should be trained on a representative sample of the domain, but vocabulary decisions (special tokens, size) must be fixed before model training begins. Re-training the tokenizer invalidates all trained model weights.

---

## Quick Decision Checklist

``` text
1. Choose a tokenizer algorithm (BPE for most cases; SentencePiece for
   multilingual or whitespace-sensitive domains).
2. Set vocabulary size: ~32K for narrow-domain; ~100K for multilingual or code-heavy.
3. Verify fertility on your target domain — sample 1 000 documents and measure
   tokens-per-character. Target: < 0.4 tokens/char for prose.
4. Set max sequence length to the 95th percentile of your document length
   distribution, not the maximum.
5. Use sequence packing when average length < 50% of max length.
6. Choose RoPE or ALiBi if long-context generalization is a requirement.
7. Save the full tokenizer artifact alongside every model checkpoint.
```

---

*See also: [Data Quality and Selection](data-quality-and-selection.md) — tokenizer fertility interacts with deduplication; near-duplicates that tokenize differently may survive hash-based dedup. [Model Size vs. Compute Budget](model-size-vs-compute-budget.md) — sequence length affects the total token count per batch, which feeds directly into the $C \approx 6ND$ estimate.*
