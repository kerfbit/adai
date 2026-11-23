# `SIGReg` — Source File Reference

- **Files:** [`src/SIGReg.hpp`](../../../../src/SIGReg.hpp) (159 lines), [`src/SIGReg.cpp`](../../../../src/SIGReg.cpp) (277 lines), [`src/SIGRegMath.hpp`](../../../../src/SIGRegMath.hpp) (56 lines, the formulas shared with the GPU kernels, TD-586)
- **Namespace:** global
- **Built into:** `adai_lejepa`; owned by [`LeJEPAEncoder`](LeJEPAEncoder.md) (`SIGReg(d_model, WORLD_MODEL_SIGREG_NUM_SKETCHES = 64)`), used in every `incremental_trainer --objective=lejepa` world-model step
- **Status tag:** `@adai-status: beta` (capped by TD-504); `SIGReg.hpp` and `SIGReg.cpp` `@adai-version: 0.3.0`, `@adai-reviewed: 2026-11-22` (TD-586); `SIGRegMath.hpp` `experimental` 0.1.0
- **Tests:** [`tests/sigreg_test.cpp`](../../../../tests/sigreg_test.cpp) (17 tests, `sigregTests`, plus 4 device tests in GPU builds; all pass, the device ones on the oneAPI CPU device)
- **Design:** [lejepa_world_model_gated_injection_plan.md](../../../proposals/lejepa_world_model_gated_injection_plan.md), Component 3 (TD-175); the LeJEPA paper (arXiv:2511.08544)
- **Last traced against the code:** 2026-10-17; the directions and the device path 2026-11-22 (TD-586)

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
| Directions fixed per process: only 64 of 768 dimensions constrained (§2) | Collapse in 92% of dimensions gives identical loss and diagnostics; irreproducible directions | **Verified** | [TD-502](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-502-sigregs-fixed-sketch-directions-leave-most-dimensions-unregularized) |
| Biased loss with a 1/n floor, applied per sequence (§2) | Short texts pushed off the target; logged loss tracks text length | **Verified** | [TD-503](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-503-sigreg-loss-has-a-sequence-length-dependent-floor) |
| Four unweighted frequencies; no NaN check; repeated projections (§2) | Weaker shape test; minor | By inspection | [TD-504](../../guides/TECHNICAL_DEBT.md#td-504-sigreg-frequency-grid-nan-inputs-and-repeated-projections) |
| Unseeded initialization generally | Context | — | TD-368 |
| Host only: directions drawn with a host RNG before every batch, and the loss, gradient and stats on host matrices | GPU LeJEPA training would have uploaded the directions every batch and done this on the CPU | **Fixed November 22, 2026** (addendum below) | [TD-586](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-586-sigreg-runs-only-on-the-host) |

---

## Addendum (October 20, 2026): per-step directions and the unbiased loss (TD-502, TD-503)

- **Directions (TD-502):** redrawn by `resample_directions()`, which `LeJEPAEncoder::train_step()` calls once per step, from a member
  `std::mt19937`. `set_seed()` reseeds it and redraws, and `get_directions()` returns the current draw.
- **Loss (TD-503):** `compute_loss()` is the unbiased estimator: each squared gap gets `(|φ̂|² − 1)/(n − 1)` added. Its expectation
  is 0 for N(0, I) at any n, so a single value can be slightly negative.
  - `backward()` is its gradient.
  - Fewer than two rows give 0 and a zero gradient.
  - The old mean over N(0, I) was 0.0865/0.0211/0.0054 at n = 8/32/128; the new one is 0.0019/0.0002/0.0001.

Tests: `RedrawnDirectionsPenalizeCollapseOutsideOneDrawsSpan`, `SetSeedMakesTheDirectionsReproducible`,
`UnbiasedLossDoesNotDependOnSequenceLength`, `FewerThanTwoRowsScoreZeroWithZeroGradient`.

---

## Addendum (October 21, 2026): applied across a batch (TD-503)

In training, SIGReg now runs on a `[batch, d_model]` matrix of mean-pooled embeddings, one row per text (`LeJEPAEncoder::train_batch()`),
as the LeJEPA paper does, rather than on one text's tokens. `n` in the estimator is then the batch size
(`WORLD_MODEL_SIGREG_BATCH_SIZE`, default 32).

## Addendum (November 22, 2026): directions from (seed, draw), and the device path (TD-586)

**Directions.** A draw is now a pure function of the seed and a draw number: element `(i, k)` is
`adai::sigreg_direction_raw(seed, draw, i * num_sketches + k)` (`SIGRegMath.hpp`, a splitmix64 counter hash through Box–Muller), and
each column is then normalized. `resample_directions()` only increments the draw number. The host matrix is built by
`ensure_host_directions()` the first time a host method (`compute_loss()`, `backward()`, `compute_variance_stats()`,
`get_directions()`) needs the current draw, and the device copy by a kernel the first time `gpu_compute()` does, so device training
neither computes nor uploads directions on the host. `set_seed(s)` sets the seed and restarts the draw count (its first draw is 1); the
seed still comes from `adai::next_init_seed()` at construction (TD-368). **The random stream changed:** the same `SEED` gives different
directions than before November 22, 2026 (they used to come from a `std::mt19937`), so runs from before don't replay exactly. The host
and the device agree to about 1e-4, not bit for bit: `icpx`'s default fast floating-point model approximates `log`/`cos` on the device
(5e-5 seen on entries of about 0.1). The frequency grid is `adai::sigreg_frequency(j)` = 0.5 × (j + 1), shared with the kernels.

**`gpu_compute(x, out_device, grad)`** (GPU builds): one projection `x · D` on the device, then `sigreg_terms_gpu()` (a work-group per
direction: the projections' population variance, the four frequencies' cos/sin means and TD-503's unbiased gap, and, with a gradient,
`d loss / d projection`, each item owning its rows), `sigreg_finish_gpu()` (one work-group: `out[0]` = the loss, `out[1]`/`out[2]` =
the variance mean and stddev, all left on the device), and the gradient `grad_P · Dᵀ` through `matrix_gemm_strided_gpu()`. One
projection serves all three results (TD-504's repeated-projection item, on the device; the host methods still recompute it). It throws
`std::invalid_argument` for a width other than `d_model` or no rows; one row gives a 0 loss and gradient, as on the host.

Tests (`SIGRegGpuTest`, GPU builds): the direction kernel against the host draw; `gpu_compute()` against the host loss, stats and
gradient at batch 1, 2, 32 and 300; a resample moving the device to the next draw; a bad width throwing.
