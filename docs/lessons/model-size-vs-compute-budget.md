# Model Size vs. Compute Budget

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

Given a fixed amount of compute, there is an optimal trade-off between **how large to make the model** and **how many tokens to train it on**. Spending the entire budget on a very large model trained briefly, or on a tiny model trained exhaustively, both produce worse results than a balanced allocation.

The key insight from scaling law research is: **most models in practice are undertrained**. The instinct to maximize parameter count for a given memory budget is usually wrong.

---

## Compute, Parameters, and Tokens

Training cost in floating-point operations (FLOPs) is approximated as:

$$C \approx 6 \cdot N \cdot D$$

where:

- $C$ — total training FLOPs
- $N$ — number of model parameters
- $D$ — number of training tokens

The factor of 6 accounts for the forward pass (2 FLOPs per multiply-add), backward pass (roughly $2\times$ the forward), and parameter updates. This is an approximation; it excludes embedding layers, attention's quadratic term at long sequences, and optimizer overhead — but it is accurate enough for planning.

---

## The Chinchilla Result

Hoffmann et al. (2022) — "Training Compute-Optimal Large Language Models" — found that for a given compute budget $C$, the loss-optimal allocation is:

$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

In plain terms: **scale parameters and tokens in equal proportion**. Doubling compute should roughly double both model size and dataset size simultaneously.

The resulting rule of thumb:

$$D_{opt} \approx 20 \cdot N$$

A compute-optimal model should be trained on approximately **20 tokens per parameter**.

| Parameters | Compute-optimal tokens |
| --- | --- |
| 7 B | ~140 B |
| 13 B | ~260 B |
| 70 B | ~1.4 T |
| 1 B | ~20 B |

Prior to Chinchilla, the prevailing practice (following GPT-3) was to train large models on far fewer tokens than optimal — GPT-3 (175 B parameters) was trained on ~300 B tokens, well under the 3.5 T tokens its size called for.

---

## Key Implication: Smaller Models Trained Longer Often Win

A 7 B model trained on 1 T tokens will outperform a 70 B model trained on 30 B tokens at the same compute cost. The smaller model is also cheaper to serve at inference.

This has a practical consequence for deployment-constrained settings: it is often worth **intentionally over-training a small model** beyond the compute-optimal point. The Llama series is an explicit example of this strategy — models trained well past the Chinchilla-optimal token count to maximize quality per inference FLOP.

The compute-optimal frontier maximizes quality per training FLOP. The *inference-optimal* frontier maximizes quality per inference FLOP. They are not the same curve.

---

## The Scaling Laws in Practice

**Kaplan et al. (2020)** established the original power-law relationships between loss, parameters, and data:

$$\mathcal{L}(N) \sim N^{-\alpha_N}, \quad \mathcal{L}(D) \sim D^{-\alpha_D}$$

with roughly $\alpha_N \approx 0.076$ and $\alpha_D \approx 0.095$ for autoregressive language models. These exponents say that returns on data are slightly better than returns on parameters, supporting the Chinchilla finding.

**Important caveat**: scaling laws are measured on held-out loss. Downstream task performance can be non-smooth — there are emergent capability thresholds that scaling laws on perplexity do not predict.

---

## Planning a Training Run

**Step 1: Establish your compute budget**

$$C = \text{(FLOPs per second per GPU)} \times \text{(number of GPUs)} \times \text{(training hours)} \times 3600$$

Account for MFU (Model FLOP Utilization) — real hardware achieves 30–60% of theoretical peak for typical transformer training. Use MFU $\approx 0.4$ as a conservative estimate.

$$C_{effective} = C_{theoretical} \times \text{MFU}$$

**Step 2: Choose $N$ and $D$ from the compute budget**

Using $C \approx 6ND$ and $D \approx 20N$:

$$N_{opt} \approx \sqrt{\frac{C}{120}}, \quad D_{opt} = 20 \cdot N_{opt}$$

**Step 3: Adjust for deployment constraints**

If inference cost matters, shift toward smaller $N$ and larger $D$. If the model will run at high batch sizes on large servers, the compute-optimal allocation is more appropriate.

---

## Diminishing Returns and the Loss Floor

Scaling is not free forever. Observed loss follows:

$$\mathcal{L}(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

where $E$ is an irreducible loss floor determined by the entropy of the data distribution. No amount of scale eliminates $E$. When incremental improvements from adding parameters or tokens become small relative to $E$, you have saturated scaling on that dataset.

Practically: if your validation loss plateaus despite doubling tokens, you are likely hitting the data quality ceiling, not the model capacity ceiling.

---

## Common Mistakes

- **Maximizing $N$ for a fixed memory budget** without considering how many tokens you can afford to train on. A model that fits in memory but is starved of data is wasteful.
- **Treating Chinchilla ratios as hard rules.** They are averages over many model sizes and datasets. Your domain and tokenizer may shift the optimum.
- **Confusing training FLOPs with wall-clock time.** Two runs with identical FLOP counts can have very different runtimes depending on parallelism strategy and hardware.
- **Ignoring inference cost when choosing model size.** A model that is 2× larger costs 2× per token at inference for every query, forever.

---

## Quick Decision Checklist

``` text
1. Estimate C_effective = theoretical FLOPs × GPUs × hours × 3600 × MFU.
2. Compute N_opt = sqrt(C / 120) and D_opt = 20 × N_opt.
3. Check if N_opt fits in available memory. If not, reduce N and increase D.
4. If serving cost matters: bias toward smaller N and larger D (over-train).
5. Validate the plan with a small proxy run — scale the FLOP budget down by
   100–1000× and confirm the loss curve matches predictions before full scale.
6. Track MFU during training. If MFU < 30%, investigate parallelism or I/O.
```

---

*See also: [Selecting a Learning Rate](selecting-a-learning-rate.md) — schedule length must match token count $D$, not wall-clock time. [Batch Size and Gradient Accumulation](batch-size-and-gradient-accumulation.md) — effective batch size affects how $D$ steps map to optimizer steps.*
