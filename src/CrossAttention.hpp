#pragma once

// @adai-status: beta        (capped by TD-050 — see TECHNICAL_DEBT.md)
// @adai-version: 0.17.0
// @adai-reviewed: 2026-11-13

#include <cstdint>
#include <memory>
#include <vector>
#include "KVCache.hpp"
#include "LoRA.hpp"
#include "Matrix.hpp"
#include "Optimizer.hpp"
#include "gpu/GPUParamRef.hpp"
#ifdef ADAI_ENABLE_GPU
#include "gpu/MatrixGPU.hpp"
#endif

/**
 * Cross-Attention Mechanism for Transformer Decoder
 *
 * Extends multi-head attention to support cross-attention where queries come
 * from the decoder and keys/values come from the encoder. This is essential
 * for encoder-decoder architectures like machine translation models.
 *
 * Key Difference from Self-Attention:
 * - Self-attention: Q = K = V = input
 * - Cross-attention: Q = decoder_input, K = V = encoder_output
 *
 * This allows the decoder to attend to all positions in the encoder output,
 * enabling the model to focus on relevant parts of the input sequence when
 * generating each output token.
 *
 * Architecture:
 * 1. Query projection: Q = decoder_input * W_q
 * 2. Key projection: K = encoder_output * W_k
 * 3. Value projection: V = encoder_output * W_v
 * 4. Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
 * 5. Output projection: Output = Attention * W_o
 *
 * Mathematical Formulation:
 * For decoder input X_dec ∈ ℝ^(tgt_len × d_model) and
 * encoder output X_enc ∈ ℝ^(src_len × d_model):
 * - Q = X_dec * W_q ∈ ℝ^(tgt_len × d_model)
 * - K = X_enc * W_k ∈ ℝ^(src_len × d_model)
 * - V = X_enc * W_v ∈ ℝ^(src_len × d_model)
 * - Scores = (Q * K^T) / √d_k ∈ ℝ^(tgt_len × src_len)
 * - Attention = softmax(Scores) * V ∈ ℝ^(tgt_len × d_model)
 * - Output = Attention * W_o ∈ ℝ^(tgt_len × d_model)
 *
 * TD-059 (fixed September 12, 2026 — see TECHNICAL_DEBT_RESOLVED.md): despite "extends
 * multi-head attention" above, Q/K/V used to never be split into per-head slices — the
 * formulation was applied once over the full d_model width, making this single-head
 * cross-attention regardless of num_heads (same gap as MultiHeadAttention's self-attention
 * path, found independently in this class, fixed the same way in the same pass).
 * forward()/forward_with_cache()/backward()/gpu_forward()/gpu_backward() now genuinely
 * split into num_heads per-head computations. **Every checkpoint trained before this fix
 * needs retraining from scratch to be meaningful under the new math.**
 *
 * TD-038 (September 13, 2026): LoRA adapters (see enable_lora() below) are applied only on
 * this CPU path (forward()/forward_with_cache()/backward()) — the separate
 * gpu_forward()/gpu_backward() persistent-residency path does not apply them. Since TD-301
 * (November 13, 2026) gpu_forward()/gpu_forward_with_cache() throw std::logic_error while adapters
 * are attached (they used to run the base weights silently); merge_lora() first.
 *
 * TD-301, the device mirror: as MultiHeadAttention's (see its class comment). The setters, load(),
 * merge_lora() and update_weights() update it; an optimizer step taken elsewhere needs
 * EncoderDecoderModel::gpu_sync_weights(). get_last_attention_weights() reflects CPU forwards
 * only; gpu_last_mean_attention_weights() gives the last gpu_forward()'s.
 */
class CrossAttention {
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
    Matrix cached_query_input;        // Query input (from decoder)
    Matrix cached_kv_input;           // Key-Value input (from encoder)
    Matrix cached_Q;                  // Projected queries
    Matrix cached_K;                  // Projected keys
    Matrix cached_V;                  // Projected values
    Matrix cached_attention_weights;  // TD-059: mean-across-heads weights, for callers/
                                      // visualization only — backward() uses
                                      // cached_head_weights_ below, the real per-head values.
    Matrix cached_attention_output;   // Output after applying attention to values

    // TD-059: per-head post-softmax attention weights from the most recent forward pass —
    // cached_head_weights_[h] is [tgt_len, src_len]. backward() differentiates through these
    // directly instead of a single d_model-wide softmax.
    std::vector<Matrix> cached_head_weights_;

    // Optimizer support
    Optimizer* optimizer{nullptr};  // Optional optimizer (nullptr = simple gradient descent)

    // TD-038: optional LoRA adapters on each projection, nullptr = disabled (the default --
    // an instance with no enable_lora() call behaves identically to before this feature
    // existed). Q takes decoder input; K/V take encoder input — see enable_lora()'s doc.
    std::unique_ptr<LoRAAdapter> lora_q_;
    std::unique_ptr<LoRAAdapter> lora_k_;
    std::unique_ptr<LoRAAdapter> lora_v_;
    std::unique_ptr<LoRAAdapter> lora_o_;

    // TD-300: tells a cross-attention KV cache which layer, weights and input its K/V came from.
    // weights_version_ changes with every weight change through the layer (weights_changed(),
    // gpu_download_weights()).
    std::uint64_t instance_id_;
    std::uint64_t weights_version_{0};
    std::uint64_t cache_source_tag(const Matrix* kv_input, int rows, int cols) const;

    // TD-300/TD-301: a new weights version, and W_q/W_k/W_v/W_o uploaded to the device mirror when
    // there is one.
    void weights_changed();

   public:
    float learning_rate{
        0.001f};  // Learning rate for weight updates (used when optimizer is nullptr)

    /**
     * Constructor
     *
     * Initializes the cross-attention layer with Xavier/He initialization.
     *
     * @param d_model Dimension of the model (must be divisible by num_heads)
     * @param num_heads Number of attention heads
     *
     * @throws std::invalid_argument if d_model or num_heads isn't positive, or d_model is not
     *         divisible by num_heads
     */
    CrossAttention(int d_model, int num_heads);

    /**
     * Forward pass through cross-attention
     *
     * Computes cross-attention between decoder queries and encoder key-values.
     * Caches intermediate values for backward pass.
     *
     * @param query_input Query input from decoder [tgt_len, d_model]
     * @param kv_input Key-Value input from encoder [src_len, d_model]
     * @param mask Optional attention mask [tgt_len, src_len]
     *             Values of 0 indicate positions to mask out (set to -inf before softmax)
     *             Values of 1 indicate positions to attend to
     * @return Attention output [tgt_len, d_model]
     *
     * Process:
     * 1. Project query_input to Q using W_q
     * 2. Project kv_input to K, V using W_k, W_v
     * 3. Compute attention scores: scores = QK^T / √d_k
     * 4. Apply mask (optional)
     * 5. Apply softmax to get attention weights
     * 6. Apply attention weights to values: output = attention_weights * V
     * 7. Project through W_o
     */
    Matrix forward(const Matrix& query_input, const Matrix& kv_input, const Matrix* mask = nullptr);

    /**
     * Forward pass with KV cache support (for inference optimization)
     *
     * For cross-attention, K/V from encoder are constant across all generation steps,
     * so we compute and cache them once on the first call.
     *
     * @param query_input Query input from decoder [num_new_tokens, d_model]
     * @param kv_input Key-Value input from encoder [src_len, d_model]. Projected only when the
     *   cache is empty; on later calls it must be the same encoder output (TD-300: the shape and 64
     *   sampled values are checked, std::invalid_argument otherwise, also when this layer's weights
     *   or LoRA adapters changed since the cache was filled). clear() the cache for a new encoder
     *   output.
     * @param mask Optional attention mask [num_new_tokens, src_len]
     * @param kv_cache Cache for encoder K/V pairs (computed once, reused)
     * @param use_cache If true, use/populate cache
     * @return Attention output [num_new_tokens, d_model]
     */
    Matrix forward_with_cache(const Matrix& query_input, const Matrix& kv_input,
                              const Matrix* mask = nullptr, KVCache* kv_cache = nullptr,
                              bool use_cache = true);

    /**
     * Forward pass with a caller-supplied pre-softmax additive score bias (TD-174).
     *
     * Identical to forward() in every other respect (projections, LoRA, per-head split,
     * softmax, output projection, backward() caching) — the only difference is that
     * `score_bias` is added to each head's scaled scores before masking/softmax. This is the
     * entry point the hippocampal-memory repetition penalty (see
     * docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 6) needs: a
     * subtractive, per-key-position bias derived from how much attention each memory slot has
     * already received, applied without needing a boolean mask (which can only fully suppress
     * a position, not gradually discourage it).
     *
     * An all-zero `score_bias` reproduces forward()'s output exactly — adding 0.0f to a score
     * is an exact no-op in IEEE floating point, so this is a strict superset of forward(), not
     * an approximation of it.
     *
     * @param query_input Query input from decoder [tgt_len, d_model]
     * @param kv_input Key-Value input from encoder [src_len, d_model]
     * @param score_bias Additive bias [tgt_len, src_len], added to every head's scaled scores
     *                    before masking/softmax (same value shared across all heads, same
     *                    broadcast convention as `mask`). Typically negative (a penalty); the
     *                    caller decides sign and magnitude.
     * @param mask Optional attention mask [tgt_len, src_len], applied after the bias — a
     *             masked position is forced to -1e9 regardless of its bias value, so masking
     *             always takes precedence over the bias.
     * @return Attention output [tgt_len, d_model]
     */
    Matrix forward_with_scores(const Matrix& query_input, const Matrix& kv_input,
                               const Matrix& score_bias, const Matrix* mask = nullptr);

    /**
     * Backward pass through cross-attention
     *
     * Computes gradients with respect to both inputs and weight matrices.
     * Accumulates gradients for weight updates.
     *
     * @param grad_output Gradient from upstream [tgt_len, d_model]
     * @param grad_query_input Output: Gradient w.r.t. query input [tgt_len, d_model]
     * @param grad_kv_input Output: Gradient w.r.t. key-value input [src_len, d_model]
     *
     * Returns two gradients since cross-attention has two inputs.
     */
    void backward(const Matrix& grad_output, Matrix& grad_query_input, Matrix& grad_kv_input);

    /**
     * Set optimizer for advanced optimization algorithms
     *
     * Enables use of Adam, AdamW, or other optimizers instead of
     * simple gradient descent. When set, automatically registers
     * weight matrices with the optimizer.
     *
     * @param opt Pointer to optimizer (nullptr to use simple gradient descent)
     */
    void set_optimizer(Optimizer* opt);

    /**
     * Register weight matrices with optimizer
     *
     * Called automatically by set_optimizer().
     * Registers W_q, W_k, W_v, W_o and their gradients with the optimizer.
     */
    void register_parameters();

    /**
     * Update weights using accumulated gradients
     *
     * If optimizer is set, uses optimizer->step() for advanced optimization.
     * Otherwise, applies simple gradient descent: W -= learning_rate * grad_W
     * Should be called after backward pass(es) to apply weight updates.
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
     * Compute the L2 norm of accumulated gradients across W_q, W_k, W_v, W_o.
     *
     * @return sqrt(sum of squared gradient values)
     */
    float get_gradient_norm() const;

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
     * @return Number of attention heads
     */
    int get_num_heads() const {
        return num_heads;
    }

    /**
     * @brief TD-180: the mean-across-heads post-softmax attention weights from the most recent
     * forward()/forward_with_cache()/forward_with_scores() call, [tgt_len, src_len]. Was already
     * computed and cached internally for exactly this purpose (see cached_attention_weights'
     * own doc comment, "for callers/visualization only") but had no public accessor until the
     * hippocampal repetition penalty (DecoderBlock's coverage-update mechanism) needed genuine
     * read access to it. Purely additive — exposes existing state, changes no behavior.
     *
     * @return Reference valid until the next forward-family call on this instance.
     */
    const Matrix& get_last_attention_weights() const {
        return cached_attention_weights;
    }

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
    // TD-301: each needs a [d_model, d_model] matrix (std::invalid_argument otherwise) and also
    // updates the device mirror, when there is one.
    void set_Wq(const Matrix& m);
    void set_Wk(const Matrix& m);
    void set_Wv(const Matrix& m);
    void set_Wo(const Matrix& m);

    /**
     * Save model weights to file
     *
     * @param filepath Path to save file
     */
    void save(const std::string& filepath) const;

    /**
     * Load model weights from file
     *
     * @param filepath Path to load file
     */
    void load(const std::string& filepath);

    // ── TD-038: LoRA (Low-Rank Adaptation) integration ───────────────────────
    /**
     * @brief Attaches LoRA adapters to this layer's Q/K/V/O projections, per config's
     * apply_to_query/key/value/output flags. Q adapts the decoder-side query projection; K/V
     * adapt the encoder-side key/value projections — see this class's own doc comment on
     * which input feeds which. Safe to call on an already-trained instance: LoRAAdapter's B
     * matrix starts at zero, so forward()'s output is IDENTICAL to the pre-LoRA output until
     * the adapters are actually trained via register_lora_parameters() below.
     */
    void enable_lora(const LoRAConfig& config);

    /** @brief True if any LoRA adapter is currently attached. */
    bool has_lora() const {
        return lora_q_ || lora_k_ || lora_v_ || lora_o_;
    }

    /**
     * @brief Registers ONLY the active LoRA adapters' own A/B matrices with `optimizer` --
     * NOT W_q/W_k/W_v/W_o. See MultiHeadAttention::register_lora_parameters()'s identical
     * doc comment for the full rationale (same pattern, same class of caller).
     */
    void register_lora_parameters(Optimizer& optimizer);

    /**
     * @brief Folds every active adapter's ΔW into the corresponding base weight matrix
     * (LoRAAdapter::merge_with_base()) and discards the adapters. See
     * MultiHeadAttention::merge_lora()'s identical doc comment.
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
        adai::gpu::GPUMatrix Wq, Wk, Wv, Wo;
        adai::gpu::GPUMatrix dWq, dWk, dWv, dWo;
        // Cached for backward (query and kv come from different sources)
        adai::gpu::GPUMatrix cached_query;     // [tgt, d_model]
        adai::gpu::GPUMatrix cached_kv;        // [src, d_model]
        adai::gpu::GPUMatrix cached_Q;         // [tgt, d_model]
        adai::gpu::GPUMatrix cached_K;         // [src, d_model]
        adai::gpu::GPUMatrix cached_V;         // [src, d_model]
        // TD-059: one real per-head post-softmax weight matrix, each [tgt, src] — replaces the
        // old single d_model-wide cached_weights now that gpu_forward()/gpu_backward()
        // genuinely split into heads.
        std::vector<adai::gpu::GPUMatrix> cached_head_weights;
        adai::gpu::GPUMatrix cached_attn_out;  // [tgt, d_model]

        explicit GPUState(int d)
            : Wq(d, d),
              Wk(d, d),
              Wv(d, d),
              Wo(d, d),
              dWq(d, d),
              dWk(d, d),
              dWv(d, d),
              dWo(d, d),
              cached_query(1, 1),
              cached_kv(1, 1),
              cached_Q(1, 1),
              cached_K(1, 1),
              cached_V(1, 1),
              cached_attn_out(1, 1) {}
    };
    std::unique_ptr<GPUState> gpu_;

    void gpu_upload_weights();
    // TD-468: this layer's device parameter buffers, for the device-resident optimizer
    // (GPUOptimizer), and the device->host weight copy it needs before checkpoints.
    void gpu_collect_parameters(std::vector<adai::gpu::GPUParamRef>& out, int layer = -1);
    void gpu_download_weights();
    void gpu_download_grads();
    void gpu_zero_grads();
    // Returns attention output; query from decoder, kv from encoder. score_bias ([tgt, src], same
    // for every head, added after scaling and before the mask, as forward_with_scores()) is the
    // hippocampal path's coverage/cross-reference bias (TD-342). TD-301: query and kv must be
    // d_model wide, mask and score_bias [tgt, src]; std::logic_error with LoRA attached.
    adai::gpu::GPUMatrix gpu_forward(const adai::gpu::GPUMatrix& query,
                                     const adai::gpu::GPUMatrix& kv,
                                     const adai::gpu::GPUMatrix* mask = nullptr,
                                     const adai::gpu::GPUMatrix* score_bias = nullptr);
    /// TD-342: the mean-across-heads post-softmax weights of the last gpu_forward(), downloaded
    /// ([tgt, src], as get_last_attention_weights()). The hippocampal coverage update needs them
    /// on the host; small (query rows × memory slots).
    Matrix gpu_last_mean_attention_weights() const;
    // Returns {d_query, d_kv}. TD-301: needs a gpu_forward() first (std::logic_error) and dout
    // [tgt, d_model].
    std::pair<adai::gpu::GPUMatrix, adai::gpu::GPUMatrix> gpu_backward(
        const adai::gpu::GPUMatrix& dout);

    /**
     * @brief TD-050: GPU incremental cross-attention using a GPUKVCache. Unlike self-attention,
     * the encoder's K/V never change across decode steps, so `kv_cache` is populated with the
     * encoder's projected K/V exactly once (on the first call, when it's empty) and simply read
     * back on every subsequent call — `kv` (the raw encoder output) is only actually used on
     * that first call; callers may pass the same reference every time regardless. Query is
     * always computed fresh from `query` (the new decoder token's hidden state). Built entirely
     * from gpu_forward()'s own already-verified per-head primitives, no new kernels.
     * Inference-only, same as MultiHeadAttention::gpu_forward_with_cache() — does not populate
     * GPUState's backward-oriented caches.
     *
     * @param query New decoder-token(s) query input [num_new_tokens, d_model]
     * @param kv Raw encoder output [encoder_seq_len, d_model] — read only when kv_cache is empty;
     *   later calls must pass one with the cache's row count and keep this layer's weights
     *   unchanged (TD-300, std::invalid_argument; the content isn't sampled on the device)
     * @param mask Optional mask, shape [num_new_tokens, encoder_seq_len]; any other shape, a
     *   query (or, on the first call, a kv) not d_model wide, or attached LoRA adapters throw
     *   (TD-301)
     * @param kv_cache Cache holding the one-time-computed encoder K/V; nullptr or
     *   use_cache=false falls back to plain gpu_forward()
     * @param use_cache Whether to actually use the cache
     */
    adai::gpu::GPUMatrix gpu_forward_with_cache(const adai::gpu::GPUMatrix& query,
                                                const adai::gpu::GPUMatrix& kv,
                                                const adai::gpu::GPUMatrix* mask,
                                                adai::gpu::GPUKVCache* kv_cache,
                                                bool use_cache = true);
#endif
};
