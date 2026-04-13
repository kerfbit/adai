# Data Quality and Selection

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

Model capability is bounded by data quality. A model trained on noisy, duplicated, or poorly distributed data will learn its defects as faithfully as its signal. No amount of architectural cleverness or hyperparameter tuning recovers from a broken dataset.

The goal of data preparation is simple to state: **maximize the information per token that reaches the model**. Every pipeline decision should be evaluated against that standard.

---

## The Four Properties of a Good Training Corpus

**1. Clean** — Noise has been removed: HTML artifacts, encoding errors, boilerplate, truncated documents, and machine-generated spam.

**2. Deduplicated** — Near-duplicate documents inflate the effective weight of repeated content. Models trained on high-duplication corpora memorize rather than generalize, and held-out perplexity becomes misleadingly optimistic.

**3. Balanced** — Domain, language, and format distribution is intentional, not accidental. Web crawls are heavily biased toward English, low-quality content farms, and recent years. Left uncorrected, this becomes the model's worldview.

**4. Representative** — The training distribution should overlap well with the target use distribution. Pretraining on purely general web text and then expecting code generation ability requires fine-tuning; it is not free.

---

## Deduplication

Deduplication is the single step with the clearest documented impact on downstream quality (Lee et al., 2022).

**MinHash / LSH** is the standard method for near-duplicate detection at scale:

1. Tokenize each document into shingles (character or word $n$-grams, typically $n = 5$).
2. Compute a MinHash signature for each document.
3. Use Locality-Sensitive Hashing (LSH) to bucket documents with similar signatures.
4. Within each bucket, compute exact Jaccard similarity and drop duplicates above a threshold (commonly $J \geq 0.8$).

Exact deduplication (SHA-256 on normalized text) is fast and should always be run first. Near-duplicate deduplication is more expensive but catches paraphrased boilerplate and scraped mirrors.

**Practical threshold**: documents appearing $\geq 3$ times in a corpus are strong candidates for removal regardless of quality.

---

## Quality Filtering

Quality filtering removes low-signal documents. Common heuristics applied in sequence:

| Filter | What It Removes |
| --- | --- |
| Length filter | Documents below a minimum token count (e.g., < 100 tokens) |
| Language ID | Documents not in the target language(s) with confidence < 0.9 |
| Perplexity filter | Documents with very high perplexity under a small LM (random text, keyboard spam) |
| Symbol ratio | Documents where punctuation or digits exceed a threshold (e.g., > 30% of characters) |
| Repetition ratio | Documents where the most common line appears in > 30% of all lines |
| URL/HTML ratio | Documents with high density of markup artifacts after extraction |

Do not over-filter. Aggressive quality filters applied without domain awareness can silently remove entire categories of legitimate content (e.g., technical documents with high symbol density, code, multilingual text).

---

## Domain Mixing

A training corpus is a mixture of sources weighted by sampling probability. The weights are a hyperparameter.

**Upsampling** a domain increases its influence beyond its raw token count. This is appropriate for high-quality, rare sources (curated books, academic papers, formal documentation).

**Downsampling** reduces influence. Common web crawl data is typically downsampled; it is large but low quality relative to curated sources.

A simple principled approach:

1. Define source categories (web, books, code, scientific, domain-specific).
2. Assign initial weights proportional to estimated quality, not raw size.
3. Train a small proxy model ($\sim$70 M parameters) on several candidate mixes.
4. Evaluate proxy models on your target task benchmarks.
5. Use the best-performing mix for the full run.

Weights from published models (GPT-3, Llama, Pile) are useful starting points but are not guaranteed to transfer to your domain or tokenizer.

---

## Data Ordering and Curriculum

The **order** in which data is presented affects training stability, especially early in the run.

- **Shuffle globally** before training. Presenting all data from one domain before another creates distribution shift that can destabilize the optimizer.
- **Curriculum learning** (easy-to-hard ordering) can help for specific tasks like code generation or mathematical reasoning, but adds complexity and offers smaller gains than cleaning and deduplication.
- **Epoch repetition**: for small datasets, repeating data (multiple epochs) is necessary but increases memorization risk. Adding noise (dropout, data augmentation) partially mitigates this.

---

## Evaluation of Data Quality

Before full training, validate the corpus:

- **Token count by domain**: confirm proportions match intended mix.
- **Sample inspection**: read 50–100 random documents from each domain. Obvious noise visible by eye indicates systematic filtering gaps.
- **Vocabulary coverage**: measure how many tokens in a held-out target domain are OOV or rare in the training vocabulary. High OOV rates indicate distribution mismatch.
- **Proxy model perplexity**: train a small model and measure perplexity on clean held-out sets. Unexpectedly high perplexity often traces back to a data pipeline bug.

---

## Common Mistakes

- **Deduplicating after tokenization** instead of at the document level. Tokenizer-level duplicates are not the same as document-level duplicates.
- **Using the test set domain as a quality signal** to filter training data. This is a form of data leakage.
- **Assuming more data is always better**. Adding a large noisy source can hurt a model already trained on a smaller clean source.
- **Forgetting metadata**: dates, sources, and licenses. Data you cannot audit cannot be responsibly deployed.

---

## Quick Decision Checklist

``` text
1. Exact dedup (hash-based) → remove identical documents.
2. Near-dedup (MinHash/LSH, J ≥ 0.8) → remove near-duplicates.
3. Language ID filter → keep only target languages.
4. Heuristic quality filters (length, symbol ratio, repetition).
5. Assign domain mix weights; validate on a proxy model.
6. Global shuffle before training.
7. Sample-inspect 50+ documents per domain before committing to a full run.
```

---

*See also: [Selecting a Learning Rate](selecting-a-learning-rate.md) — data pipeline decisions interact with batch size and schedule length.*
