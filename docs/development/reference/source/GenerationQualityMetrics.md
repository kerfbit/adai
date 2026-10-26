# `GenerationQualityMetrics` — Source File Reference

- **Files:** [`src/GenerationQualityMetrics.hpp`](../../../../src/GenerationQualityMetrics.hpp), [`src/GenerationQualityMetrics.cpp`](../../../../src/GenerationQualityMetrics.cpp)
- **Namespace:** global (`GenerationQualityScore`, `GenerationQualityEvaluator`)
- **Built into:** `adai_core`; used by `ChatbotTrainer` (per-epoch scoring and the quality backfill) and the metrics stack (`TrainingMetricsService`, SQLite/Postgres metrics databases)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10` (both files)
- **Tests:** [`tests/generation_quality_test.cpp`](../../../../tests/generation_quality_test.cpp) (23 tests), [`tests/generation_quality_async_test.cpp`](../../../../tests/generation_quality_async_test.cpp) (11 tests, `ChatbotTrainer`'s async scoring)
- **Related:** [Reading BLEU and ROUGE Results](../../../lessons/reading-bleu-rouge.md) (lesson); [ChatbotTrainer.md](ChatbotTrainer.md)
- **Last traced against the code:** 2026-10-12

> Code-traced reference. Behaviour marked **verified** was confirmed by compiling the evaluator into a
> scratch program and comparing it with a hand-written textbook (Papineni et al.) BLEU.

---

## 1. What this file is

A dependency-free **BLEU and ROUGE scorer** for comparing generated responses against reference
responses. All methods are static and pure, so it's thread-safe.

| Piece | Purpose |
|---|---|
| `GenerationQualityScore` | `bleu1`, `bleu2`, `bleu4`, `rouge1`, `rouge2`, `rougeL`, all on a **0–1** scale; each defaults to `-1` ("not computed") |
| `GenerationQualityEvaluator::evaluate(refs, hyps)` | Tokenizes both lists and computes all six scores |
| `GenerationQualityEvaluator::tokenize(text)` | Whitespace split, ASCII lowercase, strip leading/trailing punctuation, drop empty tokens |

### Why it matters

It produces the only **generation-quality** signal in training. Loss measures next-token
prediction; these scores measure whether the model's actual generated answers resemble the targets.
`ChatbotTrainer` scores a sample of validation pairs every epoch (when
`enable_generation_quality_metrics` is on) and reports BLEU-4 and the three ROUGE values to the
metrics service, which stores them per epoch and shows them on the dashboards. They're the numbers
someone would use to judge whether a training run is improving its answers, so their correctness
matters more than their small code size suggests.

---

## 2. API and algorithms

### `tokenize(text)`

`"Don't stop -- it's café's \"quoted\" ..."` → `[don't] [stop] [it's] [café's] [quoted]` (verified).
Internal punctuation stays (`don't`), punctuation-only tokens disappear (`--`, `...`), and only ASCII
is lowercased. That differs from sacreBLEU's `13a` tokenizer, which splits punctuation into separate
tokens, and from `rouge-score`, which replaces non-alphanumerics with spaces. **These scores aren't
comparable to published BLEU/ROUGE numbers**, only to each other.

### `evaluate(refs, hyps)`

Returns the default (all `-1`) when `refs` is empty or the two lists differ in length. Otherwise it
computes:

| Score | Method |
|---|---|
| `bleu1`, `bleu2`, `bleu4` | `compute_corpus_bleu()`: **corpus-level** modified n-gram precision with clipped counts, uniform weights, and a corpus-level brevity penalty (`exp(1 − R/H)` when the total hypothesis length `H` is shorter than the total reference length `R`) |
| `rouge1`, `rouge2` | `compute_corpus_rouge_n()`: per-pair n-gram precision, recall and F1, then the **macro average** over pairs |
| `rougeL` | `compute_corpus_rouge_l()`: per-pair LCS-based F1 (two-row DP in `lcs_length()`), macro-averaged |

ROUGE matches the standard definitions (single reference, no stemming). BLEU's smoothing doesn't.

### BLEU smoothing (verified)

Each order's precision is `(matches + 1) / (hypothesis n-grams + 1)`, **for every order, including
unigrams**. When a hypothesis has no n-grams of some order (shorter than `n` tokens), that order
contributes a fixed `log = -20`. The header calls this *"add-1 smoothing (Lin & Och 2004, method 1)"*,
but Lin & Och's add-one smoothing (Chen & Cherry 2014's *smoothing 2*) applies only to orders n ≥ 2,
and NLTK's *method1* adds a small epsilon to zero numerators only. Applying add-1 to unigrams, and
on very short texts, distorts scores badly:

| Case (single pair) | This code: BLEU-1 / BLEU-4 | Textbook BLEU-4 |
|---|---|---|
| Identical 6-word sentence | 1.000 / 1.000 | 1.000 |
| **Unrelated** 5-word answer (zero words in common) | **0.167 / 0.230** | 0 |
| Unrelated 12-word answer | 0.077 / 0.087 | 0 |
| **Same words, scrambled order** (5 words) | 1.000 / **0.359** | 0 |
| **Exactly correct 3-word answer** (`"yes i do"`) | 1.000 / **0.0067** | undefined; effectively 0 |
| Empty hypothesis | 0 / 0 | 0 |
| 10 unrelated 8-word pairs (the default sample size) | 0.012 / 0.015 | 0 |

So, at the sentence level:
- An answer that shares **no** words with the reference gets a positive BLEU-1, and its BLEU-4 is
  **higher** than its BLEU-1 (because higher orders have fewer n-grams, so `+1` weighs more).
- A perfectly correct short reply scores near zero, far below an unrelated longer one. That inverts
  the ranking for short conversational targets (daily_dialog has many).

With a 10-pair corpus the inflation shrinks (0.015 for fully unrelated output), but it's still not
zero, and short targets are still penalized whenever they make up much of the sample.

---

## 3. Who calls what

| Caller | Uses |
|---|---|
| `ChatbotTrainer::compute_generation_quality_metrics()` | Per epoch: the first `generation_quality_sample_size` (default **10**) validation pairs, generated with `generation_quality_max_tokens` (ignored, TD-359). **Async** (on a `clone()`d model, CPU only) when the sample is at least `generation_quality_async_threshold` (default 50), otherwise synchronous (GPU generation on GPU builds). Reports `bleu4`, `rouge1`, `rouge2`, `rougeL` through `IMetricsReporter::update_generation_quality_metrics()`. A generation that throws is scored as an empty hypothesis |
| `ChatbotTrainer::backfill_generation_quality()` | **Sentence-level** BLEU-4 per sample as `meta.quality`, where the inflation above applies in full. Its results are never read (TD-273) |
| `TrainingMetricsService::update_generation_quality_metrics()` | Stores the four values in its snapshot, logs them, and writes them to the metrics DB. It builds a **fresh** `GenerationQualityScore` from the four values, so **`bleu1`/`bleu2` stay `-1`** |
| `SQLiteMetricsDatabase` / `PostgresMetricsDatabase` | `insert_generation_quality()` writes all six fields. The `bleu1`/`bleu2` columns therefore **always hold `-1`** for real training runs, since they're computed in `evaluate()` and dropped before they reach the DB |

**Two scale mismatches for readers.** The scores are on a 0–1 scale, while the lesson
[Reading BLEU and ROUGE Results](../../../lessons/reading-bleu-rouge.md) gives interpretation ranges
on the conventional 0–100 scale (for example, "< 10 = almost useless"). And the lesson's ranges assume
standard corpus BLEU-4 over a real test set; a smoothed 10-sample BLEU-4 on this tokenization isn't
that number.

---

## 4. Tests

`generation_quality_test.cpp` (23 tests):
- tokenization (lowercase, punctuation, empty);
- BLEU (identical, partial overlap, ordering BLEU-1 ≥ BLEU-2 ≥ BLEU-4 on one example, brevity penalty);
- ROUGE-1/2/L (identical, empty hypothesis, partial, macro averaging, reordering, subsequences);
- edge cases (empty and mismatched input, single words, all scores in range).

`generation_quality_async_test.cpp` (11 tests) covers `ChatbotTrainer`'s async path: threshold
selection, thread joining, and reporting.

**Gaps:**
- `CompletelyDifferentHypothesisScoresNearZero` asserts only `bleu1 < 0.1`. It passes because its
  3-word hypothesis is half the reference's length, so the brevity penalty (0.37) pulls the smoothed
  unigram precision of 0.25 down to 0.09. An unrelated hypothesis of the reference's length would
  fail it, and BLEU-4 isn't checked at all.
- `BLEU1AtLeastBLEU2AtLeastBLEU4` passes on its one example, but the ordering doesn't hold in
  general (the unrelated-answer case above has BLEU-4 > BLEU-1).
- Nothing compares against a reference implementation, and nothing covers short (< 4-token) correct
  answers or scrambled word order.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Add-1 smoothing on every order (including unigrams) and a fixed `-20` for missing orders; the "Lin & Och method 1" label is wrong (§2) | Unrelated answers get positive BLEU (0.23 BLEU-4 for 5 words), scrambled answers score high, correct short answers score ~0; per-epoch BLEU-4 is biased and can rank outputs backwards | **Verified** | [TD-378](../../guides/TECHNICAL_DEBT.md#td-378-bleu-smoothing-inflates-unrelated-answers-and-zeroes-correct-short-ones) |
| `bleu1`/`bleu2` computed but dropped by `TrainingMetricsService`; the DB columns always hold `-1` (§3) | Dead columns that look like data | By inspection | [TD-379](../../guides/TECHNICAL_DEBT.md#td-379-bleu1bleu2-are-computed-but-never-stored) |
| Tokenization differs from sacreBLEU/`rouge-score`; 0–1 scale vs. the lesson's 0–100 ranges; a 10-sample default (§2–3) | Numbers can't be compared with published scores or read against the lesson's ranges; a 10-sample BLEU-4 is very noisy | By inspection | [TD-380](../../guides/TECHNICAL_DEBT.md#td-380-generation-quality-scores-arent-comparable-or-well-sampled) |
| Tests don't catch the smoothing distortion (§4) | The bug passed review as `stable` | By inspection | [TD-378](../../guides/TECHNICAL_DEBT.md#td-378-bleu-smoothing-inflates-unrelated-answers-and-zeroes-correct-short-ones) (tests) |
| `generation_quality_max_tokens` ignored; backfill results unread | Already tracked | — | TD-359, TD-273 |

---

## Addendum (October 25, 2026): textbook BLEU (TD-378)

- `evaluate()`'s BLEU-1/2/4 are unsmoothed corpus BLEU (Papineni et al.). Clipped matches and n-gram counts are summed over every pair,
  and the score is BP · exp(mean log p_n). An order with no matches gives 0. The add-1-on-every-order smoothing and the fixed −20
  for missing orders are gone.
- `sentence_bleu(reference, hypothesis, max_n = 4)` is the explicitly smoothed single-pair score, used for `ChatbotTrainer`'s
  per-sample backfill. It uses Chen & Cherry 2014 method 1 (0.1 matches for an order with none, sacreBLEU `floor`) and effective order
  (orders longer than the hypothesis are skipped), matching sacreBLEU's sentence settings on the same tokens.

| Case | Old BLEU-4 | Corpus BLEU-4 now | `sentence_bleu` |
|---|---|---|---|
| 5 unrelated words vs 6 | 0.188–0.23 | 0 | 0.025 |
| Same 5 words scrambled | 0.359 | 0 | 0.080 |
| Exactly correct 3-word answer | 0.0067 | 0 (no 4-grams) | 1.0 |
| "the cat sat on a mat" vs "the cat sat on the mat" | — | 0.537 | 0.537 |

Tokenization (TD-380) is unchanged, so the scores still aren't comparable with published numbers. Tests:
`GenerationQualityBLEUReference.*`, `GenerationQualitySentenceBLEU.*`.

## Addendum (October 25, 2026): BLEU-1/2 stored (TD-379)

`bleu1`/`bleu2` now reach the metrics DB. `IMetricsReporter::update_generation_quality_metrics()` takes them as two trailing arguments
(default -1), and the push client, API and `TrainingMetricsService` carry them into the `generation_quality` row. Test:
`PushedBleu1AndBleu2AreStored`.
