#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-15

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
// the projection directions are the only internal state, fixed once at construction and reused
// for every call.
//
// Design note (this implementation's specific characteristic-function test): the LeJEPA paper's
// own SIGReg formulation integrates the squared characteristic-function gap over a continuous
// range of frequencies. This implementation discretizes that integral to a small fixed set of
// representative frequencies (see `SIGReg.cpp`'s `kFrequencies`) for tractability — a standard
// simplification for characteristic-function-based normality tests (e.g. the Epps-Pulley test
// uses the same idea, just with a different frequency-weighting choice). The
// isotropic-Gaussian-near-zero / degenerate-batch-large behavior this regularizer is meant to
// have holds under this simplification too — see `tests/sigreg_test.cpp`.

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

    /**
     * @brief Compute the regularization loss over a batch of embeddings.
     * @param embeddings `[batch, d_model]` — a batch of embedding vectors, one per row.
     * @return A non-negative scalar, near zero iff the batch's per-direction projections all
     *   look like a standard normal (see the class doc comment).
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
    int get_d_model() const {
        return d_model_;
    }

   private:
    int d_model_;
    int num_sketches_;
    Matrix directions_;  // [d_model, num_sketches] — each column a fixed unit-norm direction,
                         // sampled once at construction and reused for every call.
};
