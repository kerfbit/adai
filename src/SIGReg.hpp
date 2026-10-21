#pragma once

// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-10-20

// TD-175 (LJ-1a): Sketched Isotropic Gaussian Regularization (SIGReg), from the LeJEPA paper
// (arXiv:2511.08544) — see docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 3
// for the design context this was filed from. Pushes a batch of embeddings toward an isotropic
// standard Gaussian N(0, I) by testing, along a fixed set of random 1D projection directions,
// whether the projected values' empirical characteristic function matches a standard normal's
// analytic characteristic function — this is what lets SIGReg regularize embedding *geometry*
// directly (no representation collapse) without a stop-gradient, an EMA teacher network, or any
// other architecture-specific self-supervision trick, per the paper's own stated goal. Stateless
// with respect to model weights: it has no learnable parameters of its own (`compute_loss()`/
// `backward()` only ever produce a scalar loss and a gradient through the *input* embeddings) —
// the projection directions are the only internal state. TD-502: they are redrawn by
// resample_directions() once per training step (LeJEPAEncoder::train_step() calls it before
// compute_loss()/backward(), which then share that draw), from a seedable RNG. Fixed directions
// constrained only their num_sketches-dimensional span, so collapse elsewhere went unpenalized.
//
// TD-503: compute_loss() is the unbiased estimator of the squared characteristic-function gap.
// The plain |phi_hat - phi|^2 has an expected floor of (1 - |phi_P|^2)/n, which made a true
// Gaussian score higher for shorter sequences (n = one text's tokens) and the logged loss track
// text length. Its expectation is now exactly 0 for an N(0, I) sample of any size, so a single
// value can be slightly negative.
//
// Design note (this implementation's specific characteristic-function test): the LeJEPA paper's
// own SIGReg formulation integrates the squared characteristic-function gap over a continuous
// range of frequencies. This implementation discretizes that integral to a small fixed set of
// representative frequencies (see `SIGReg.cpp`'s `kFrequencies`) for tractability — a standard
// simplification for characteristic-function-based normality tests (e.g. the Epps-Pulley test
// uses the same idea, just with a different frequency-weighting choice). The
// isotropic-Gaussian-near-zero / degenerate-batch-large behavior this regularizer is meant to
// have holds under this simplification too — see `tests/sigreg_test.cpp`.

#include <cstdint>
#include <random>
#include "Matrix.hpp"

/**
 * @brief Sketched Isotropic Gaussian Regularization (SIGReg).
 *
 * Given a batch of embeddings `[batch, d_model]`, projects them onto `num_sketches` fixed random
 * unit directions (the "sketch") and, for each direction, measures how far the projected values'
 * empirical characteristic function is from a standard normal's analytic characteristic function
 * `exp(-t^2/2)` at a small set of frequencies `t`. Averaged over directions and frequencies, this
 * is near zero exactly when every 1D projection of the batch looks like a standard normal —
 * which, for a batch of vectors, is equivalent to the batch being drawn from an isotropic
 * `N(0, I)` (a distribution is isotropic Gaussian iff every 1D linear projection of it is a
 * standard normal — the mathematical property this regularizer actually tests for).
 *
 * No learnable parameters of its own — see the file header for why this is safe to use directly
 * in a loss term without registering anything with an `Optimizer`.
 */
class SIGReg {
   public:
    /**
     * @param d_model      Embedding dimension.
     * @param num_sketches Number of random projection directions (default 64, matching the
     *                     LeJEPA paper's own reported setting).
     */
    explicit SIGReg(int d_model, int num_sketches = 64);

    /** TD-502: draws a new set of unit directions. Call once per training step, before that
     *  step's compute_loss()/backward() (and compute_variance_stats()), which all use the current
     *  draw. */
    void resample_directions();

    /** TD-502: reseeds the direction RNG (seeded from std::random_device at construction) and
     *  redraws, so runs and tests can be reproduced. */
    void set_seed(std::uint32_t seed);

    /**
     * @brief Compute the regularization loss over a batch of embeddings.
     * @param embeddings `[batch, d_model]` — a batch of embedding vectors, one per row.
     * @return The unbiased estimate of the mean squared characteristic-function gap (TD-503):
     *   zero in expectation iff the per-direction projections look like a standard normal, so it
     *   can be slightly negative. 0 when the batch has fewer than 2 rows (nothing to estimate).
     */
    float compute_loss(const Matrix& embeddings) const;

    /**
     * @brief Gradient of `compute_loss()`'s output with respect to the input embeddings.
     * @param embeddings Same input `compute_loss()` would be called with.
     * @return `[batch, d_model]` — same shape as `embeddings`.
     */
    Matrix backward(const Matrix& embeddings) const;

    int get_num_sketches() const {
        return num_sketches_;
    }
    /** The current draw, [d_model, num_sketches], one unit direction per column (tests and
     *  diagnostics). */
    const Matrix& get_directions() const {
        return directions_;
    }
    int get_d_model() const {
        return d_model_;
    }

    /** @brief Per-direction projected variance statistics — a diagnostic distinct from
     *  compute_loss()'s characteristic-function-based test, computed on the same projections
     *  compute_loss() forms internally but never exposes. For a true isotropic N(0, I) batch,
     *  every direction's projected variance should be ~1.0 (mean ~1.0) and uniform across
     *  directions (stddev ~0) — a large stddev means variance is collapsed in specific directions
     *  rather than spread isotropically, a distinction the aggregate compute_loss() scalar alone
     *  cannot make: a batch could have mean variance ~1.0 while being wildly anisotropic. */
    struct VarianceStats {
        float mean;
        float stddev;
    };

    /**
     * @brief Per-direction projected variance across the batch, aggregated to a mean and stddev
     *   over all num_sketches directions. Recomputes the projection rather than caching it,
     *   matching compute_loss()/backward()'s own no-caching style.
     * @param embeddings `[batch, d_model]`, same input compute_loss() would be called with.
     * @return {0.0f, 0.0f} if `embeddings` has zero rows.
     */
    VarianceStats compute_variance_stats(const Matrix& embeddings) const;

   private:
    int d_model_;
    int num_sketches_;
    Matrix directions_;  // [d_model, num_sketches] — each column a unit-norm direction, redrawn by
                         // resample_directions() (TD-502).
    std::mt19937 rng_;
};
