# Lessons

Single-point lessons on training, architecture, and model development for the ADAI project. Each document covers one topic in depth: the core concept, the underlying mechanics, practical guidance, common mistakes, and a quick decision checklist.

---

## Training Planning

Decisions made before a run begins. These interact heavily — changes to one typically require revisiting the others.

| Lesson | Summary |
| --- | --- |
| [Model Size vs. Compute Budget](model-size-vs-compute-budget.md) | Chinchilla scaling laws, the $C \approx 6ND$ approximation, and how to size a run from a FLOP budget |
| [Data Quality and Selection](data-quality-and-selection.md) | Deduplication, quality filtering, domain mixing, and corpus validation |
| [Tokenizer and Sequence Length](tokenizer-and-sequence-length.md) | BPE vs. SentencePiece, vocabulary sizing, sequence packing, and position embedding trade-offs |

## Optimization

Hyperparameters and techniques that control how the optimizer moves through parameter space.

| Lesson | Summary |
| --- | --- |
| [Selecting a Learning Rate](selecting-a-learning-rate.md) | The range test, warmup, cosine decay, and the linear scaling rule |
| [Batch Size and Gradient Accumulation](batch-size-and-gradient-accumulation.md) | Effective batch size, the linear scaling rule, and how to implement accumulation correctly |
| [Gradient Clipping](gradient-clipping.md) | Global norm clipping, choosing `max_norm`, fp16 operation order, and gradient norm as a diagnostic |
| [Weight Initialization](weight-initialization.md) | Xavier, He, transformer-specific residual scaling, and fine-tuning initialization |

## Stability and Generalization

Techniques that keep training numerically stable and prevent the model from overfitting.

| Lesson | Summary |
| --- | --- |
| [Precision and Training Stability](precision-and-stability.md) | fp16 vs. bf16, mixed-precision workflows, loss scaling, and attention softmax overflow |
| [Regularization Strategy](regularization-strategy.md) | Weight decay (AdamW), dropout rates by regime, label smoothing, and diagnosing over/underfitting |

## Methodology

Practices that make experiments trustworthy and results recoverable.

| Lesson | Summary |
| --- | --- |
| [Evaluation and Checkpointing Strategy](evaluation-and-checkpointing-strategy.md) | Validation set requirements, evaluation frequency, full checkpoint contents, early stopping, and test set discipline |
| [Reading BLEU and ROUGE Results](reading-bleu-rouge.md) | BLEU precision vs. ROUGE recall, score interpretation ranges, tokenization consistency, and common misreading mistakes |
| [Reproducibility](reproducibility.md) | Seeding, deterministic mode, data pipeline versioning, environment recording, and checkpoint RNG state |

## Architecture Reference

| Document | Summary |
| --- | --- |
| [Architecting Attention](architechting_attention.md) | Structural scaling and training dynamics of modern transformers |
| [Core Model Components](core-model-components.md) | Technical reference for the ADAI C++ transformer implementation |

---

## Reading Order

For someone planning a first training run, the recommended sequence is:

1. [Model Size vs. Compute Budget](model-size-vs-compute-budget.md) — establish what you can afford to train
2. [Data Quality and Selection](data-quality-and-selection.md) — prepare the corpus
3. [Tokenizer and Sequence Length](tokenizer-and-sequence-length.md) — fix the tokenizer and context window
4. [Selecting a Learning Rate](selecting-a-learning-rate.md) — choose the schedule
5. [Batch Size and Gradient Accumulation](batch-size-and-gradient-accumulation.md) — set the effective batch size
6. [Weight Initialization](weight-initialization.md) — verify initialization defaults
7. [Gradient Clipping](gradient-clipping.md) — add the safety net
8. [Precision and Training Stability](precision-and-stability.md) — choose the precision format
9. [Regularization Strategy](regularization-strategy.md) — set weight decay and dropout
10. [Evaluation and Checkpointing Strategy](evaluation-and-checkpointing-strategy.md) — plan how to measure and save progress
11. [Reproducibility](reproducibility.md) — record everything needed to repeat the run
