#pragma once

// @adai-status: beta        (capped by TD-050 — see TECHNICAL_DEBT.md; TD-038 LoRA support added; TD-050 GPU incremental-cache forward added; TD-195 backward() gradient members now accumulate, not overwrite, across calls)
// @adai-version: 0.12.1
// @adai-reviewed: 2026-09-18


#include <functional>
#include <memory>
#include <vector>
#include "KVCache.hpp"
#include "LoRA.hpp"
#include "Matrix.hpp"
#include "Optimizer.hpp"
#ifdef ADAI_ENABLE_GPU
#include "gpu/MatrixGPU.hpp"
#endif

/**
 * Multi-Head Self-Attention Mechanism
 *
 * Implements the multi-head attention mechanism from "Attention is All You Need".
 * This is a core component of transformer architectures, allowing the model to
 * jointly attend to information from different representation subspaces.
 *
 * Key Features:
 * - Parallel attention heads for diverse representation learning
 * - Scaled dot-product attention
 * - Linear projections for queries, keys, values, and output
 * - Gradient computation for backpropagation
 * - Optional attention masking (e.g., for padding or causal attention)
 *
 * Architecture (as designed):
 * 1. Linear projections: Q = XW_q, K = XW_k, V = XW_v
 * 2. Split into num_heads: Each head processes d_k dimensions
 * 3. Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
 * 4. Concatenate heads and apply output projection: Output = Concat(heads)W_o
 *
 * Mathematical Formulation (as designed):
 * For input X ∈ ℝ^(seq_len × d_model):
 * - Q = XW_q, K = XW_k, V = XW_v  (each ∈ ℝ^(seq_len × d_model))
 * - Split into h heads: Q_i, K_i, V_i ∈ ℝ^(seq_len × d_k) where d_k = d_model/h
 * - Attention_i = softmax((Q_iK_i^T)/√d_k)V_i
 * - Output = Concat(Attention_1, ..., Attention_h)W_o
 *
 * TD-059 (fixed September 12, 2026 — see TECHNICAL_DEBT_RESOLVED.md): until this fix,
 * `forward()` and `forward_with_cache()` — the only two entry points anything in this
 * codebase actually called (EncoderBlock/DecoderBlock self-attention, and the KV-cache
 * decode path), plus `gpu_forward()`/`gpu_backward()` — computed `Q * K.transpose()` over
 * the FULL d_model width, once, with no per-head split at all: mathematically single-head
 * attention over d_model dimensions, still scaled by 1/√d_k (wrong for that width — should
 * have been 1/√d_model, off by √num_heads). `forward_parallel()` was the only method that
 * implemented the design correctly (explicit per-head [h*d_k, (h+1)*d_k) column slicing,
 * matching the formulation above exactly) — but nothing called it; `num_heads` had no
 * effect on the actual computation other than gating the constructor's divisibility
 * check. `forward()` now delegates to `forward_parallel()`'s per-head logic (which
 * `forward_with_cache()`, `backward()`, `gpu_forward()`, and `gpu_backward()` also now
 * implement), so genuine multi-head attention is what actually runs. **Every checkpoint
 * trained before this fix was trained under the old (single-head-over-d_model) math and
 * needs retraining from scratch to be meaningful under the new math** — this is not a
 * hot-swappable weight format change. `CrossAttention` (`CrossAttention.cpp`) had the
 * identical gap, independently, and was fixed the same way in the same pass.
 *
 * TD-038 (September 13, 2026): LoRA adapters (see enable_lora() below) are applied only on
 * this CPU path (forward()/forward_parallel()/forward_with_cache()/backward()) — the separate
 * gpu_forward()/gpu_backward() persistent-residency path (TD-033's decode route) does not
 * apply them. Enabling LoRA while a caller uses that GPU path silently runs the unmodified
 * base weights instead; not currently guarded against here.
 */
/// Callback invoked after softmax in every forward() pass, receiving the
/// attention weight matrix [seq_len × seq_len].
using AttentionHookFn = std::function<void(const Matrix&)>;

#ifdef ADAI_ENABLE_GPU
/**
 * Callback invoked immediately after softmax in every gpu_forward() call. GPU-resident
 * data doesn't fit the raw-matrix hook contract above, so this receives the
 * already-reduced average-row-entropy scalar instead (computed on-device via
 * GPUMatrix::row_entropy_avg()).
 */
using GPUAttentionStatsHookFn = std::function<void(float)>;
#endif

class MultiHeadAttention {
   private:
    // Model dimensions
    int d_model;    // Model dimension (embedding size)
    int num_heads;  // Number of attention heads
    int d_k;        // Dimension per head (d_model / num_heads)

    // Learnable weight matrices
    Matrix W_q;  // Query projection [d_model, d_model]
    Matrix W_k;  // Key projection [d_model, d_model]
    Matrix W_v;  // Value projection [d_model, d_model]
    Matrix W_o;  // Output projection [d_model, d_model]

    // Gradient matrices
    Matrix W_q_grad;  // Accumulated gradients for W_q
    Matrix W_k_grad;  // Accumulated gradients for W_k
    Matrix W_v_grad;  // Accumulated gradients for W_v
    Matrix W_o_grad;  // Accumulated gradients for W_o

    // Cached values for backward pass
    Matrix cached_input;              // Input to forward pass
    Matrix cached_Q;                  // Projected queries
    Matrix cached_K;                  // Projected keys
    Matrix cached_V;                  // Projected values
    Matrix cached_attention_weights;  // TD-059: mean-across-heads weights, for
                                      // get_attention_weights()/hooks only — backward() uses
                                      // cached_head_weights_ below, the real per-head values.
    Matrix cached_attention_output;   // Output after applying attention to values

    // TD-059: per-head post-softmax attention weights from the most recent forward pass —
    // cached_head_weights_[h] is [q_rows, kv_rows] (q_rows/kv_rows: seq_len for forward(),
    // num_new_tokens/total_seq_len for forward_with_cache()). backward() differentiates
    // through these directly instead of a single d_model-wide softmax.
    std::vector<Matrix> cached_head_weights_;

    // Optimizer for weight updates
    Optimizer* optimizer{
        nullptr};  // Pointer to optimizer (nullptr means use simple gradient descent)

    // Attention hook (for entropy tracking, TD-013)
    AttentionHookFn attention_hook_;

    // TD-038: optional LoRA adapters on each projection, nullptr = disabled (the default --
    // an instance with no enable_lora() call behaves identically to before this feature
    // existed). See enable_lora()'s own doc comment for the zero-init-is-a-no-op guarantee.
    std::unique_ptr<LoRAAdapter> lora_q_;
    std::unique_ptr<LoRAAdapter> lora_k_;
    std::unique_ptr<LoRAAdapter> lora_v_;
    std::unique_ptr<LoRAAdapter> lora_o_;

#ifdef ADAI_ENABLE_GPU
    // GPU-path equivalent, fired after softmax in gpu_forward() (see GPUAttentionStatsHookFn).
    GPUAttentionStatsHookFn gpu_attention_stats_hook_;
#endif

   public:
    float learning_rate{0.001f};  // Learning rate for weight updates

    /**
     * Constructor
     *
     * Initializes the multi-head attention layer with Xavier/He initialization.
     *
     * @param d_model Dimension of the model (must be divisible by num_heads)
     * @param num_heads Number of attention heads
     *
     * @throws std::invalid_argument if d_model is not divisible by num_heads
     */
    MultiHeadAttention(int d_model, int num_heads);

    /**
     * Forward pass through multi-head attention
     *
     * Computes genuine multi-head self-attention on the input sequence — delegates to
     * forward_parallel()'s per-head implementation (TD-059). Caches intermediate values
     * for backward pass, including per-head attention weights.
     *
     * @param input Input matrix [seq_len, d_model]
     * @param mask Optional attention mask [seq_len, seq_len]
     *             Values of 0 indicate positions to mask out (set to -inf before softmax)
     *             Values of 1 indicate positions to attend to
     * @return Attention output [seq_len, d_model]
     *
     * Process (per head h, over its own [h*d_k, (h+1)*d_k) column slice of Q/K/V):
     * 1. Project input to Q, K, V using W_q, W_k, W_v
     * 2. Compute attention scores: scores_h = Q_h K_h^T / √d_k
     * 3. Apply mask (optional, shared across heads)
     * 4. Apply softmax to get attention weights
     * 5. Apply attention weights to values: output_h = attention_weights_h * V_h
     * 6. Concatenate all heads' output_h and project through W_o
     */
    Matrix forward(const Matrix& input, const Matrix* mask = nullptr);

    /**
     * Forward pass with explicit parallel-vs-sequential head computation control
     *
     * The actual per-head implementation (TD-059) — forward() calls this with
     * use_parallel=true. Splits Q, K, V into num_heads column slices and processes each
     * head independently (optionally via OpenMP), matching "Attention is All You Need"'s
     * formulation exactly. Exposed separately (rather than folded entirely into forward())
     * so AttentionHeadBenchmark.cpp can compare the parallel and sequential branches
     * directly.
     *
     * @param input Input matrix [seq_len, d_model]
     * @param mask Optional attention mask [seq_len, seq_len]
     * @param use_parallel Enable/disable OpenMP parallelization over heads (default: true)
     * @return Attention output [seq_len, d_model]
     *
     * Implementation:
     * 1. Project input to Q, K, V
     * 2. Split Q, K, V into num_heads parts (each of dimension d_k)
     * 3. Compute attention for each head (optionally IN PARALLEL using OpenMP)
     * 4. Concatenate head outputs
     * 5. Apply output projection
     */
    Matrix forward_parallel(const Matrix& input, const Matrix* mask = nullptr,
                            bool use_parallel = true);

    /**
     * Forward pass with KV cache support (for inference optimization)
     *
     * Enables caching of key-value pairs during autoregressive generation. In subsequent
     * calls, only computes K/V for new tokens and reuses cached values. Implements genuine
     * per-head attention (TD-059) over the cached K/V, same as forward_parallel().
     *
     * @param input Input matrix [num_new_tokens, d_model]
     * @param mask Optional attention mask [num_new_tokens, total_seq_len]
     * @param kv_cache Pointer to KVCache structure (nullptr = no caching)
     * @param use_cache If true and kv_cache provided, update cache with new K/V
     * @return Attention output [num_new_tokens, d_model]
     *
     * Cache Behavior:
     * - First call (empty cache): Compute K/V for all tokens, store in cache
     * - Subsequent calls: Compute K/V for new token only, concatenate with cache
     * - Attention computed per-head over all tokens (cached + new)
     *
     * Performance: ~2-3x speedup for long sequences
     */
    Matrix forward_with_cache(const Matrix& input, const Matrix* mask = nullptr,
                              KVCache* kv_cache = nullptr, bool use_cache = true);

    /**
     * Backward pass through multi-head attention
     *
     * Computes gradients with respect to input and weight matrices.
     * Updates weights using accumulated gradients.
     *
     * @param grad_output Gradient from upstream [seq_len, d_model]
     * @return Gradient with respect to input [seq_len, d_model]
     *
     * Note: This function both accumulates gradients AND updates weights.
     * Call zero_grad() before the next forward pass.
     */
    Matrix backward(const Matrix& grad_output);

    /**
     * Set optimizer for weight updates
     *
     * Registers all weight matrices with the optimizer.
     * If optimizer is nullptr, falls back to simple gradient descent.
     *
     * @param opt Pointer to optimizer instance
     */
    void set_optimizer(Optimizer* opt);

    /**
     * Register all parameters with the optimizer
     *
     * Should be called after set_optimizer() or when optimizer changes.
     */
    void register_parameters();

    /**
     * Update weights using accumulated gradients
     *
     * Uses optimizer if available, otherwise falls back to simple gradient descent.
     * Zeros out gradients after update.
     */
    void update_weights();

    /**
     * Zero out all accumulated gradients
     *
     * Resets all gradient matrices to zero.
     * Should be called before each forward/backward pass in training.
     */
    void zero_grad();

    /**
     * Get the model dimension
     *
     * @return Model dimension (d_model)
     */
    int get_d_model() const {
        return d_model;
    }

    /**
     * Get the number of attention heads
     *
     * @return Number of heads
     */
    int get_num_heads() const {
        return num_heads;
    }

    /**
     * Get dimension per head
     *
     * @return Dimension per head (d_k = d_model / num_heads)
     */
    int get_d_k() const {
        return d_k;
    }

    /**
     * Get the last computed attention weights
     *
     * Useful for visualization and interpretation.
     *
     * @return Attention weights matrix [seq_len, seq_len]
     */
    const Matrix& get_attention_weights() const {
        return cached_attention_weights;
    }

    /**
     * Register a callback fired after softmax in every forward() pass.
     * The callback receives the attention weight matrix [seq_len × seq_len].
     * Replaces any previously registered hook.
     */
    void set_attention_hook(AttentionHookFn fn) {
        attention_hook_ = std::move(fn);
    }

    /**
     * Remove any registered attention hook.
     */
    void clear_attention_hook() {
        attention_hook_ = nullptr;
    }

#ifdef ADAI_ENABLE_GPU
    /**
     * Register a callback fired after softmax in every gpu_forward() call.
     * The callback receives the average per-row Shannon entropy of the
     * attention weights. Pass nullptr / call clear_gpu_attention_stats_hook() to disable.
     */
    void set_gpu_attention_stats_hook(GPUAttentionStatsHookFn fn) {
        gpu_attention_stats_hook_ = std::move(fn);
    }

    void clear_gpu_attention_stats_hook() {
        gpu_attention_stats_hook_ = nullptr;
    }
#endif

    /**
     * Print configuration
     *
     * @param name Optional name for the layer
     */
    void print_config(const std::string& name = "MultiHeadAttention") const;

    // ── SafeTensors accessor API ─────────────────────────────────────────────
    const Matrix& get_Wq() const {
        return W_q;
    }
    const Matrix& get_Wk() const {
        return W_k;
    }
    const Matrix& get_Wv() const {
        return W_v;
    }
    const Matrix& get_Wo() const {
        return W_o;
    }
    void set_Wq(const Matrix& m) {
        W_q = m;
    }
    void set_Wk(const Matrix& m) {
        W_k = m;
    }
    void set_Wv(const Matrix& m) {
        W_v = m;
    }
    void set_Wo(const Matrix& m) {
        W_o = m;
    }

    /**
     * Save weights to file
     *
     * @param filename Path to save weights
     */
    void save_weights(const std::string& filename) const;

    /**
     * Load weights from file
     *
     * @param filename Path to load weights from
     * @throws std::runtime_error if file cannot be opened or dimensions mismatch
     */
    void load_weights(const std::string& filename);

    /**
     * Get gradient norm for monitoring
     *
     * @return L2 norm of all gradients
     */
    float get_gradient_norm() const;

    /**
     * Clip gradients by norm
     *
     * @param max_norm Maximum allowed gradient norm
     */
    void clip_gradients(float max_norm);

    // ── TD-038: LoRA (Low-Rank Adaptation) integration ───────────────────────
    /**
     * @brief Attaches LoRA adapters to this layer's Q/K/V/O projections, per config's
     * apply_to_query/key/value/output flags (apply_to_ffn/dropout are not applicable here
     * -- see LoRAConfig's own doc). Safe to call on an already-trained instance: LoRAAdapter's
     * B matrix starts at zero, so forward()'s output is IDENTICAL to the pre-LoRA output
     * until the adapters are actually trained via register_lora_parameters() below. Replaces
     * any adapters from a previous enable_lora() call (each starts fresh, at ΔW=0 again).
     */
    void enable_lora(const LoRAConfig& config);

    /** @brief True if any LoRA adapter is currently attached. */
    bool has_lora() const {
        return lora_q_ || lora_k_ || lora_v_ || lora_o_;
    }

    /**
     * @brief Registers ONLY the active LoRA adapters' own A/B matrices with `optimizer` --
     * NOT W_q/W_k/W_v/W_o. This is what makes LoRA training "freeze the base model": the
     * base weights are never registered anywhere, so Optimizer::step() never touches them,
     * the same register-with-an-external-optimizer pattern EncoderDecoderModel::
     * register_parameters() already uses for full fine-tuning. Independent of
     * set_optimizer()/register_parameters()/update_weights() above, which remain unchanged
     * for the traditional non-LoRA full-fine-tune path (and can still be used simultaneously
     * with LoRA active, e.g. to fine-tune W_o fully while adapting Q/K/V via LoRA, though
     * the usual LoRA setup is to register only the adapters).
     */
    void register_lora_parameters(Optimizer& optimizer);

    /**
     * @brief Folds every active adapter's ΔW = (alpha/r)*A^T*B^T into the corresponding base
     * weight matrix (LoRAAdapter::merge_with_base()) and discards the adapters -- "no
     * additional inference latency" from LoRA.hpp's own file doc, exercised for real. After
     * this call has_lora() is false and forward() runs the plain base-weight path, bit-for-bit
     * equivalent (up to floating-point summation order) to the adapted path just before the
     * merge.
     */
    void merge_lora();

    // LoRA adapter accessors, primarily for tests/inspection. May return nullptr.
    LoRAAdapter* get_lora_q() {
        return lora_q_.get();
    }
    LoRAAdapter* get_lora_k() {
        return lora_k_.get();
    }
    LoRAAdapter* get_lora_v() {
        return lora_v_.get();
    }
    LoRAAdapter* get_lora_o() {
        return lora_o_.get();
    }

#ifdef ADAI_ENABLE_GPU
    struct GPUState {
        // Weight mirrors
        adai::gpu::GPUMatrix Wq, Wk, Wv, Wo;
        // Gradient accumulators
        adai::gpu::GPUMatrix dWq, dWk, dWv, dWo;
        // Cached for backward
        adai::gpu::GPUMatrix cached_input;     // [seq, d_model]
        adai::gpu::GPUMatrix cached_Q;         // [seq, d_model]
        adai::gpu::GPUMatrix cached_K;         // [seq, d_model]
        adai::gpu::GPUMatrix cached_V;         // [seq, d_model]
        // TD-059: one real per-head post-softmax weight matrix, each [seq, seq] — replaces the
        // old single d_model-wide cached_weights now that gpu_forward()/gpu_backward()
        // genuinely split into heads instead of computing one global attention pattern.
        std::vector<adai::gpu::GPUMatrix> cached_head_weights;
        adai::gpu::GPUMatrix cached_attn_out;  // [seq, d_model]

        explicit GPUState(int d)
            : Wq(d, d),
              Wk(d, d),
              Wv(d, d),
              Wo(d, d),
              dWq(d, d),
              dWk(d, d),
              dWv(d, d),
              dWo(d, d),
              cached_input(1, 1),
              cached_Q(1, 1),
              cached_K(1, 1),
              cached_V(1, 1),
              cached_attn_out(1, 1) {}
    };
    std::unique_ptr<GPUState> gpu_;

    void gpu_upload_weights();
    void gpu_download_grads();
    void gpu_zero_grads();
    adai::gpu::GPUMatrix gpu_forward(const adai::gpu::GPUMatrix& input,
                                     const adai::gpu::GPUMatrix* mask = nullptr);
    adai::gpu::GPUMatrix gpu_backward(const adai::gpu::GPUMatrix& dout);

    /**
     * @brief TD-050: GPU incremental self-attention using a GPUKVCache, mirroring
     * forward_with_cache()'s CPU algorithm exactly — new-token Q/K/V computed fresh, K/V
     * appended to the cache, attention computed as [num_new_tokens, total_seq_len] against the
     * full cached K/V. Built entirely from gpu_forward()'s own already-verified per-head
     * primitives (gpu_slice_head_columns()/gpu_scatter_head_columns(), GPUMatrix's own
     * operators) — no new kernels. Inference-only: does not populate GPUState's own
     * backward-oriented caches, so gpu_backward() must not be called after this.
     *
     * @param input New-token(s) input [num_new_tokens, d_model] (typically 1 row during
     *   generation)
     * @param mask Optional mask, shape [num_new_tokens, total_seq_len_after_append]
     * @param kv_cache Cache to append to and read from; nullptr or use_cache=false falls back
     *   to plain gpu_forward()
     * @param use_cache Whether to actually use/update the cache
     */
    adai::gpu::GPUMatrix gpu_forward_with_cache(const adai::gpu::GPUMatrix& input,
                                                const adai::gpu::GPUMatrix* mask,
                                                adai::gpu::GPUKVCache* kv_cache,
                                                bool use_cache = true);
#endif
};
