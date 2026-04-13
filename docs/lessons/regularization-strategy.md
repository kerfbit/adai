# Regularization Strategy

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

Regularization is the set of techniques that prevent a model from fitting the training data so precisely that it fails to generalize to new inputs. The symptoms of under-regularization are a widening gap between training loss and validation loss, and a model that performs well on benchmarks derived from its training distribution but poorly on anything else.

The correct regularization strategy depends on the training regime. **Pretraining at scale** rarely needs explicit regularization — the sheer volume of data provides sufficient diversity. **Fine-tuning on small datasets** is where regularization decisions matter most.

---

## The Three Main Instruments

### 1. Weight Decay

Weight decay adds a penalty to the loss proportional to the magnitude of the weights, discouraging parameters from growing large without corresponding benefit to the loss:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \sum_i \theta_i^2$$

In practice, weight decay is applied directly by the optimizer rather than by modifying the loss. **AdamW** implements this correctly by applying the decay to the weights themselves, not to the adapted gradient — the distinction matters because standard Adam applies the penalty through the gradient, which interacts incorrectly with Adam's per-parameter scaling.

**Typical values**: $\lambda = 0.01$ – $0.1$ for pretraining; $0.01$ – $0.001$ for fine-tuning.

**What not to regularize**: bias terms and layer norm parameters ($\gamma$, $\beta$) should be excluded from weight decay. Decaying these parameters distorts learned normalizations and rarely helps.

### 2. Dropout

Dropout randomly zeroes a fraction $p$ of activations during training, forcing the network to learn redundant representations that do not rely on any single unit:

$$h_i^{train} = h_i \cdot \text{Bernoulli}(1-p) \cdot \frac{1}{1-p}$$

The $\frac{1}{1-p}$ rescaling (inverted dropout) keeps the expected activation magnitude constant, so no adjustment is needed at inference when dropout is disabled.

**Placement in transformers**:

- After the attention output projection
- After each feed-forward sub-layer
- Optionally on the attention weights themselves (attention dropout)
- At the input embedding (embedding dropout)

**Typical rates**:

| Training regime | Dropout rate $p$ |
| --- | --- |
| Large-scale pretraining (>1B tokens) | 0.0 (omit entirely) |
| Medium pretraining | 0.0 – 0.1 |
| Fine-tuning on small datasets | 0.1 – 0.3 |

At large scale, the data diversity provides sufficient regularization; dropout slows training without benefit and is commonly omitted (GPT-3, Llama).

### 3. Label Smoothing

Label smoothing replaces hard one-hot targets with a softened distribution that assigns a small probability $\epsilon$ to all non-target classes:

$$y_{smooth} = (1 - \epsilon) \cdot y_{one\text{-}hot} + \frac{\epsilon}{K}$$

where $K$ is the vocabulary size. This prevents the model from becoming overconfident by penalizing it for assigning zero probability mass outside the target token.

**Typical values**: $\epsilon = 0.0$ – $0.1$. Values above 0.1 degrade output quality; the model becomes uncertain where certainty is warranted.

**When to use**: fine-tuning on tasks with noisy labels, or when the model exhibits extreme overconfidence (logits saturating before convergence). For standard language model pretraining with clean data, label smoothing often provides no benefit.

---

## Regularization by Training Regime

| Regime | Weight Decay | Dropout | Label Smoothing |
| --- | --- | --- | --- |
| Pretraining (large-scale) | 0.1 | 0.0 | 0.0 |
| Pretraining (small/medium) | 0.01 – 0.1 | 0.0 – 0.1 | 0.0 |
| Fine-tuning (large dataset) | 0.01 | 0.0 – 0.1 | 0.0 |
| Fine-tuning (small dataset) | 0.01 – 0.001 | 0.1 – 0.3 | 0.0 – 0.1 |
| Fine-tuning (noisy labels) | 0.01 | 0.1 | 0.05 – 0.1 |

---

## Diagnosing Under- and Over-Regularization

**Under-regularized** (overfitting):

- Validation loss diverges upward while training loss continues to decrease
- Strong performance on training benchmarks, weak on held-out or out-of-distribution inputs
- Overconfident output distributions (predicted probabilities cluster near 1.0)

**Over-regularized** (underfitting):

- Validation loss and training loss are close but both higher than expected
- Model performance does not improve despite more training steps
- Dropout rate so high that the model cannot memorize even simple patterns

The target is a small, stable gap between training and validation loss, with both decreasing together.

---

## Interaction With Other Hyperparameters

**Weight decay and learning rate**: these two must be tuned together. A high learning rate with high weight decay can cause the model to shrink weights faster than the optimizer grows them, stalling training. A safe starting ratio is $\lambda / \eta \approx 0.01$ – $0.1$.

**Dropout and batch size**: small batch sizes already introduce stochastic noise; adding high dropout on top can make gradient estimates unreliable. With batch sizes below 8 samples, reduce or remove dropout.

**Label smoothing and temperature**: if you apply temperature scaling at inference (for sampling diversity), label smoothing during training partially achieves the same effect on the learned distribution. Avoid combining both aggressively.

---

## Techniques Less Commonly Used for Transformers

- **L1 regularization** (weight sparsity): rarely used in practice for transformer weights. L2 (weight decay) dominates.
- **Stochastic depth** (layer dropout): drops entire transformer blocks randomly during training with probability $p_{drop}$ increasing linearly with depth. Used in some vision transformers; less common in language models.
- **Data augmentation**: for text, augmentation is non-trivial. Back-translation, random token masking, and synonym substitution are used in some fine-tuning pipelines but are not standard practice for pretraining.

---

## Common Mistakes

- **Applying weight decay to layer norm and bias parameters.** These should be excluded from the decay parameter group.
- **Using dropout during large-scale pretraining.** It adds training cost and provides no benefit when data is abundant.
- **Tuning dropout before weight decay.** Weight decay is cheaper (no stochastic overhead) and should be tuned first.
- **Interpreting a small train/val gap as evidence of good regularization.** It may simply mean the validation set is not challenging. Always test on out-of-distribution inputs before drawing conclusions.
- **Using label smoothing with a small vocabulary.** At $K = 100$ classes, $\epsilon/K$ is 0.001 per non-target class — negligible. Label smoothing is most meaningful when $K$ is large (e.g., vocabulary sizes of 32 K – 256 K).

---

## Quick Decision Checklist

``` text
1. Use AdamW — not Adam — to apply weight decay correctly.
2. Set weight decay to 0.1 for pretraining; 0.01 for fine-tuning.
3. Exclude bias and layer norm parameters from the decay group.
4. Pretraining at scale: set dropout = 0.0.
5. Fine-tuning on small data: start at dropout = 0.1; increase if overfitting.
6. Only add label smoothing if labels are noisy or the model is overconfident.
7. Monitor the train/val loss gap every checkpoint. A widening gap = more regularization needed.
```

---

*See also: [Selecting a Learning Rate](selecting-a-learning-rate.md) — weight decay and learning rate must be tuned together. [Data Quality and Selection](data-quality-and-selection.md) — at large scale, data quality is the primary regularizer; explicit techniques are secondary.*
