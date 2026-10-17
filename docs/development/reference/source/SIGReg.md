# `SIGReg` — Source File Reference

- **Files:** [`src/SIGReg.hpp`](../../../../src/SIGReg.hpp) (103 lines), [`src/SIGReg.cpp`](../../../../src/SIGReg.cpp) (204 lines)
- **Namespace:** global
- **Built into:** `adai_lejepa`; owned by [`LeJEPAEncoder`](LeJEPAEncoder.md) (`SIGReg(d_model, WORLD_MODEL_SIGREG_NUM_SKETCHES = 64)`), used in every `incremental_trainer --objective=lejepa` world-model step
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-09-15` (both files)
- **Tests:** [`tests/sigreg_test.cpp`](../../../../tests/sigreg_test.cpp) (13 tests, `sigregTests`; all pass)
- **Design:** [lejepa_world_model_gated_injection_plan.md](../../../proposals/lejepa_world_model_gated_injection_plan.md), Component 3 (TD-175); the LeJEPA paper (arXiv:2511.08544)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real class.

---

## 1. What this file is

Sketched Isotropic Gaussian Regularization: a parameter-free loss that pushes a set of embeddings toward an isotropic standard Gaussian, which is what
lets the LeJEPA world model train without a stop-gradient or EMA teacher.

```text
P = X · D                                   X: [n, d_model], D: [d_model, K] fixed random unit columns (K = num_sketches)
loss = mean over k ∈ 1..K, t ∈ {0.5, 1, 1.5, 2} of
       (mean_i cos(t·P_ik) − exp(−t²/2))² + (mean_i sin(t·P_ik))²
```

That is, the squared gap between each projection's empirical characteristic function and N(0, 1)'s, at four fixed frequencies (a discretized,
unweighted version of the paper's integral).

| Member | Behaviour |
|---|---|
| `SIGReg(d_model, num_sketches = 64)` | Rejects non-positive sizes before allocating; draws `D` from `std::random_device`-seeded normals and normalizes each column. Fixed for the instance's lifetime |
| `compute_loss(X)` | As above; checks `X.cols == d_model`; 0 for an empty batch |
| `backward(X)` | Analytic gradient through the projections, then `· Dᵀ`. **Verified** against central differences (median relative error 1.5×10⁻⁴, max 3.4×10⁻³) |
| `compute_variance_stats(X)` | Mean and standard deviation, over the K directions, of each projection's population variance (diagnostic only; LeJEPA logs it) |
| `get_num_sketches()`, `get_d_model()` | Accessors |

**What the "batch" is.** `LeJEPAEncoder::train_step()` passes the target view's **token embeddings for one text** (`[tokens, d_model]`), so SIGReg makes
the token positions of each sequence look like samples from N(0, I). The paper applies it across a batch of samples' embeddings; this is a design
deviation, not a bug, but it matters for the batch-size effect below.

---

## 2. Problems

### Fixed directions leave most dimensions unregularized (verified)

The K sketch directions are drawn once per process and never change, so the loss only constrains the K-dimensional subspace they span: 64 of 768
dimensions at the shipped settings. A 256-row Gaussian batch and the same batch projected onto that 64-dimensional span (so **collapsed in 704 of 768
dimensions**, 92%) gave **identical** results: loss 0.00265 and variance mean 1.004 for both. Seen through a fresh set of random directions, the collapsed
batch had variance mean 0.086 (against 0.987) and loss 0.245, nearly the fully-collapsed value. Within a training pass the encoder can therefore collapse
anywhere outside the sketched subspace with no penalty and no sign in the logged SIGReg metrics. Each new process draws new directions, which partly
masks this across `trainer_service` passes, but it also makes runs irreproducible (TD-368's class). Resampling the directions on every call (the point of a
"sketched"/sliced test), or at least every step, closes the gap.

### The loss has a sequence-length-dependent floor (verified)

The empirical characteristic function of n samples has variance ∝ 1/n, so even a true N(0, I) set scores above zero, and the floor shrinks with n:

| n (tokens) | 8 | 20 | 32 | 128 | 512 | 2,048 |
|---|---|---|---|---|---|---|
| Loss of true N(0, I) | 0.0785 | 0.0331 | 0.0225 | 0.0057 | 0.0013 | 0.0003 |

A fully collapsed (all-zero) set scores 0.343, so an ideal 8-token sequence already sits at 23% of the collapsed value. Because the batch is one text's
tokens, short texts get a strong gradient pushing them away from a target they have already reached (the minimizer for small n is not Gaussian), and
the logged `sigreg_loss` mostly tracks text length rather than embedding quality, so it can't be compared across datasets or over time. Debiasing (the
standard unbiased estimator of |φ̂ − φ|², i.e. subtracting the 1/n term) or applying it across samples would fix both.

### Smaller items (by inspection)

- **Unweighted, four-point frequency grid:** the paper's (and the Epps–Pulley test's) Gaussian frequency weighting is replaced by four equal weights;
  sensitivity to high-order shape is limited.
- **No input validation** for non-finite embeddings (a NaN propagates into the loss and gradient).
- **Recomputes** the projection and the characteristic-function sums separately in `compute_loss()`, `compute_variance_stats()` and `backward()` (three
  `X · D` products per step).

---

## 3. Tests

`sigreg_test.cpp` (13): constructor validation, accessors, shape checks, empty batches, loss near zero for an isotropic Gaussian and large for a
degenerate batch, variance statistics (isotropic, collapsed, anisotropic), the gradient against finite differences, and independent directions per
instance agreeing statistically.

**Not covered:** collapse outside the sketched subspace, the loss floor's dependence on n.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Directions fixed per process: only 64 of 768 dimensions constrained (§2) | Collapse in 92% of dimensions gives identical loss and diagnostics; irreproducible directions | **Verified** | [TD-502](../../guides/TECHNICAL_DEBT.md#td-502-sigregs-fixed-sketch-directions-leave-most-dimensions-unregularized) |
| Biased loss with a 1/n floor, applied per sequence (§2) | Short texts pushed off the target; logged loss tracks text length | **Verified** | [TD-503](../../guides/TECHNICAL_DEBT.md#td-503-sigreg-loss-has-a-sequence-length-dependent-floor) |
| Four unweighted frequencies; no NaN check; repeated projections (§2) | Weaker shape test; minor | By inspection | [TD-504](../../guides/TECHNICAL_DEBT.md#td-504-sigreg-frequency-grid-nan-inputs-and-repeated-projections) |
| Unseeded initialization generally | Context | — | TD-368 |
