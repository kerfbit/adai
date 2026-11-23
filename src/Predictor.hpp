#pragma once

// @adai-status: beta
// @adai-version: 0.3.0
// @adai-reviewed: 2026-11-22

#include <memory>
#include <string>
#include <vector>
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
/*
 * TD-585: in GPU builds the predictor also runs on the device (gpu_forward()/gpu_backward(), with
 * device weight and gradient mirrors), delegating to FeedForward's own GPU path, for LeJEPA training on
 * the device (TD-588).
 */
class Predictor {
   private:
    std::unique_ptr<FeedForward> net_;  // d_model -> hidden_dim -> d_model
    int d_model_;
    int hidden_dim_;
#ifdef ADAI_ENABLE_GPU
    bool gpu_current_ = false;    // the device mirror holds the host weights (cleared by load())
    int gpu_forward_rows_ = -1;  // rows of the last gpu_forward(), -1 before any (gpu_backward() needs one)
#endif

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

    /** TD-427: writes the wrapped FeedForward's weights (FeedForward::save_weights() format). */
    void save(const std::string& filename) const;

    /** TD-427: reads weights written by save(). Returns false, leaving the weights unchanged, when
     *  the file doesn't exist. Throws when it holds a different shape or is the wrong size (a
     *  truncated file would otherwise load partway, TD-369). */
    bool load(const std::string& filename);

    int get_d_model() const {
        return d_model_;
    }
    int get_hidden_dim() const {
        return hidden_dim_;
    }

#ifdef ADAI_ENABLE_GPU
    // TD-585: the device path, delegating to FeedForward's. The device weights are uploaded on first
    // use and again after load(); while training on the device they are the live copy, and
    // gpu_download_weights() brings them back to the host (before save()).

    /** forward() on the device: [rows, d_model] -> [rows, d_model]. Throws std::invalid_argument when
     *  the input's width isn't d_model. */
    adai::gpu::GPUMatrix gpu_forward(const adai::gpu::GPUMatrix& context_embedding);
    /** backward() on the device: adds to the device gradients and returns the gradient w.r.t. the
     *  last gpu_forward()'s input. Throws std::logic_error before any gpu_forward(), and
     *  std::invalid_argument when the shape differs from that call's output. */
    adai::gpu::GPUMatrix gpu_backward(const adai::gpu::GPUMatrix& grad_output);
    /** Copies the host weights to the device. */
    void gpu_upload_weights();
    /** Copies the device weights to the host (no-op before any upload). */
    void gpu_download_weights();
    /** Zeroes the device gradients (uploading first if needed, so they exist). */
    void gpu_zero_grads();
    /** Appends this predictor's device parameters (W1, W2, b1, b2) for a GPUOptimizer, keyed by the
     *  same host matrices register_parameters_with_optimizer() registers; uploads first if needed. */
    void gpu_collect_parameters(std::vector<adai::gpu::GPUParamRef>& out, int layer = -1);
#endif
};
