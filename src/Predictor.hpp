#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-15

#include <memory>
#include "FeedForward.hpp"
#include "Matrix.hpp"
#include "Optimizer.hpp"

/**
 * Predictor (LeJEPA world-model plan, LJ-1b / TD-176)
 *
 * Embedding-space predictor: given a context view's embedding, predicts the target view's
 * embedding. See docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 2.
 *
 * This is deliberately *not* a LanguageModelHead — there is no vocabulary projection and no
 * reconstruction of tokens. Per the LeJEPA paper's own "no architecture-specific tuning"
 * finding, a small feed-forward stack is sufficient, so this class is a thin wrapper around the
 * existing FeedForward layer (d_model -> hidden_dim -> d_model) rather than a new architecture:
 * every actual forward/backward/optimizer detail (Xavier init, GELU, gradient accumulation) is
 * FeedForward's own, unchanged.
 *
 * Used only inside LeJEPAEncoder::train_step() (TD-178) — predicts one masked view's embedding
 * from the other's, compared against the real target embedding to form predictor_loss. Not used
 * at inference time: once LeJEPAEncoder is frozen after pretraining, only its encode() output is
 * consumed by the gated cross-attention injection point (Component 4); the predictor's job ends
 * with pretraining.
 */
class Predictor {
   private:
    std::unique_ptr<FeedForward> net_;  // d_model -> hidden_dim -> d_model
    int d_model_;
    int hidden_dim_;

   public:
    /**
     * @param d_model    Embedding dimension — must match LeJEPAEncoder's own d_model.
     * @param hidden_dim FeedForward's internal hidden width. Unlike a transformer block's own
     *                   d_ff (typically 4x d_model), this can be modest — the predictor's job is
     *                   a simple embedding-space mapping, not the token-level feature expansion
     *                   FeedForward otherwise sits between attention layers to provide.
     */
    Predictor(int d_model, int hidden_dim);

    /**
     * Predict the target view's embedding from the context view's.
     *
     * @param context_embedding [batch, d_model] (or [1, d_model] for a single example)
     * @return Predicted embedding, same shape as the input.
     */
    Matrix forward(const Matrix& context_embedding);

    /**
     * @param grad_output Gradient of the loss w.r.t. this predictor's output, same shape as a
     *                     prior forward() call's return value.
     * @return Gradient w.r.t. the context embedding that was passed to forward() — the caller
     *         (LeJEPAEncoder::train_step) chains this back into the shared encoder.
     */
    Matrix backward(const Matrix& grad_output);

    /** Applies accumulated gradients (optimizer step if one is registered, else plain SGD using
     *  the wrapped FeedForward's own learning_rate) and zeroes them afterward. */
    void update_weights();

    /** Zeroes accumulated gradients without applying them. */
    void zero_grad();

    /** Registers the wrapped FeedForward's parameters with a shared optimizer, mirroring every
     *  other component's own register_parameters_with_optimizer() convention. */
    void register_parameters_with_optimizer(Optimizer& optimizer);

    int get_d_model() const {
        return d_model_;
    }
    int get_hidden_dim() const {
        return hidden_dim_;
    }
};
