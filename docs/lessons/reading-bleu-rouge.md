# Reading BLEU and ROUGE Results

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

BLEU and ROUGE are n-gram overlap metrics. They measure how much of the reference text appears in the generated output, and vice versa. Neither metric tells you whether a generation is fluent, factual, or useful — they tell you whether the generated text contains the same sequences of words as the reference. Understanding what those numbers actually represent prevents you from drawing conclusions the metrics do not support.

---

## BLEU

**BLEU** (Bilingual Evaluation Understudy) was designed for machine translation. It measures *precision*: what fraction of n-grams in the hypothesis appear in the reference.

$$\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where $p_n$ is the modified n-gram precision at order $n$, $w_n = 1/N$ is a uniform weight, and **BP** is the brevity penalty:

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}$$

$c$ is the length of the hypothesis, $r$ is the effective reference length. The brevity penalty prevents the model from gaming precision by generating one highly accurate word.

### Reading a BLEU Score

| Score range | Rough interpretation |
| --- | --- |
| < 10 | Output is almost useless; near-random or degenerate |
| 10 – 20 | Understandable but heavily flawed |
| 20 – 30 | Acceptable for some tasks; most MT systems land here early |
| 30 – 40 | Good quality; competitive with human translation on narrow domains |
| 40 – 50 | High quality; approaching human parity on constrained tasks |
| > 50 | Rare outside of constrained, formulaic domains |

These ranges apply to **corpus-level BLEU with four reference n-grams (BLEU-4)**. Sentence-level BLEU is noisy and not comparable — a single short sentence can score 0 or 100 on a technicality.

### What BLEU Does Not Capture

- **Synonyms and paraphrases.** "automobile" and "car" are different n-grams; BLEU penalizes the correct synonym.
- **Word order beyond n-gram windows.** Long-range structural correctness is invisible.
- **Fluency.** A bag of correct unigrams scores reasonably even if the sentence is grammatically broken.
- **Single-reference bias.** BLEU assumes one correct way to express an idea. Multi-reference BLEU is more reliable but rarely reported.

---

## ROUGE

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) was designed for summarization. It measures *recall*: what fraction of n-grams in the reference appear in the hypothesis.

The three variants you will encounter most often:

### ROUGE-N

$$\text{ROUGE-N} = \frac{\sum_{\text{ref}} \sum_{g \in \text{ref}} \text{Count}_\text{match}(g)}{\sum_{\text{ref}} \sum_{g \in \text{ref}} \text{Count}(g)}$$

- **ROUGE-1**: unigram recall — does the summary contain the key words?
- **ROUGE-2**: bigram recall — does the summary contain key phrases?

ROUGE-1 measures content coverage. ROUGE-2 measures phrasal coherence with the reference.

### ROUGE-L

Measures the longest common subsequence (LCS) between hypothesis and reference, normalized by reference length. Unlike n-gram variants, ROUGE-L allows gaps — it rewards structural alignment without requiring exact contiguous matches.

$$\text{ROUGE-L} = \frac{\text{LCS}(H, R)}{|R|}$$

ROUGE-L is the most robust single ROUGE score for summarization; report it alongside ROUGE-1 and ROUGE-2.

### Reading ROUGE Scores

ROUGE scores are reported as fractions (0 – 1) or percentages (0 – 100). The interpretation is domain-dependent, but typical abstractive summarization results:

| Metric | Competitive range |
| --- | --- |
| ROUGE-1 | 0.38 – 0.46 |
| ROUGE-2 | 0.15 – 0.22 |
| ROUGE-L | 0.35 – 0.43 |

Extractive summaries will score higher than abstractive ones by construction — extractive systems copy sentences verbatim. If your model is abstractive and scores comparably to extractive baselines, that is a strong result.

---

## How BLEU and ROUGE Relate

| Property | BLEU | ROUGE |
| --- | --- | --- |
| Primary orientation | Precision | Recall |
| Designed for | Translation | Summarization |
| Brevity penalty | Yes (BP) | No |
| Fluency sensitivity | Low | Low |
| Paraphrase sensitivity | High | High |

For generation tasks that are neither pure translation nor pure summarization, report both. A model with high BLEU but low ROUGE is precise but incomplete. A model with high ROUGE but low BLEU is covering the reference content but generating excess or imprecise text.

---

## Comparing Runs

BLEU and ROUGE differences are only meaningful when:

1. **The tokenization is identical.** Many teams report scores using different tokenizers (Moses, SentencePiece, whitespace split). A switch in tokenizer can move BLEU by 1–3 points with no change in model quality. Always specify the tokenizer used.
2. **The test set is fixed.** Scores across different test splits cannot be compared.
3. **The decoding parameters are fixed.** Beam size, length penalty, and sampling temperature all affect n-gram overlap. Report the decoding configuration alongside the score.
4. **The delta is meaningful.** Differences below ~0.5 BLEU or ~1 ROUGE point are typically within evaluation noise and should not drive architectural decisions in isolation.

---

## Common Mistakes

**Treating a single metric as ground truth.** A model that scores 2 BLEU points higher than a baseline is not necessarily better — validate with human evaluation or a secondary metric before drawing conclusions.

**Comparing sentence-level and corpus-level BLEU.** Sentence-level BLEU is unreliable. Always aggregate over the full test set.

**Ignoring the ceiling.** On a constrained, near-deterministic task (e.g., date formatting), a BLEU of 30 is catastrophic. On open-ended creative generation, it may be the best achievable. Know what a human would score on the same test set.

**Not reporting variance.** For small test sets, resample with different random seeds and report mean ± std, or bootstrap confidence intervals.

---

## Quick Checklist

- [ ] Are you comparing corpus-level scores, not sentence-level?
- [ ] Is the tokenizer the same across all runs being compared?
- [ ] Is the test set identical and held out from all model selection decisions?
- [ ] Are beam size and length penalty fixed across runs?
- [ ] Is the delta large enough (> 0.5 BLEU or > 1 ROUGE point) to act on?
- [ ] Are you reporting ROUGE-1, ROUGE-2, *and* ROUGE-L together?
- [ ] Have you noted whether your baseline is extractive or abstractive?
