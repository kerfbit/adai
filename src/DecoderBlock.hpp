#pragma once

// @adai-status: stable        (TD-050 GPU incremental-cache forward added; TD-180 gated world-model/hippocampal cross-attention paths added, CPU forward/backward only — see class doc; TD-187 save()/load() now persist the gated paths, closing TD-180's own documented gap)
// @adai-version: 1.3.0
// @adai-reviewed: 2026-09-17


#include <memory>
#include <string>
#include <utility>
#include "CrossAttention.hpp"
#include "FeedForward.hpp"
#include "HippocampalMemory.hpp"
#include "KVCache.hpp"
#include "LayerNorm.hpp"
#include "Matrix.hpp"
#include "MultiHeadAttention.hpp"
#include "Optimizer.hpp"
#ifdef ADAI_ENABLE_GPU
#include "gpu/MatrixGPU.hpp"
#endif

/**
 * Transformer Decoder Block
 *
 * Implements a single layer of the transformer decoder architecture,
 * combining masked multi-head self-attention, cross-attention to encoder,
 * and position-wise feed-forward networks with residual connections
 * and layer normalization.
 *
 * Architecture (Pre-LN — normalization happens before each sublayer, not
 * after; the residual stream itself is never normalized inside the block,
 * which is why LLMDecoder applies a final LayerNorm once after the last
 * block. This placement is more stable for deep stacks than Post-LN):
 *   Input
 *     ↓
 *   Norm -> Masked Self-Attention (causal) -> Add (residual)
 *     ↓
 *   Norm -> Cross-Attention (to encoder output) -> Add (residual)
 *     ↓
 *   [TD-180: Norm -> World-Model Cross-Attention -> tanh(gate) * (·) -> Add (residual)]
 *     ↓
 *   [TD-180: Norm -> Hippocampal Cross-Attention (repetition-penalized) ->
 *            tanh(gate_h) * (·) -> Add (residual)]
 *     ↓
 *   Norm -> Feed-Forward Network -> Add (residual)
 *     ↓
 *   Output (unnormalized)
 *
 * Mathematical Operations:
 *   self_attn_output = MultiHeadAttention(LayerNorm(input), causal_mask)
 *   residual1 = input + self_attn_output
 *   cross_attn_output = CrossAttention(LayerNorm(residual1), encoder, mask)
 *   residual2 = residual1 + cross_attn_output
 *   [TD-180, if world_model_output given] residual2 += tanh(gate) * WorldModelCrossAttention(...)
 *   [TD-180, if memory given]  residual2 += tanh(gate_h) * HippocampalCrossAttention(...)
 *   ff_output = FeedForward(LayerNorm(residual2))
 *   output = residual2 + ff_output
 *
 * Features:
 *   - Causal self-attention (prevents attending to future)
 *   - Cross-attention to encoder output
 *   - TD-180: two further optional gated cross-attention paths — a frozen, pretrained world
 *     model (docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 4) and a
 *     continuously-written hippocampal episodic memory with an attention-level repetition
 *     penalty (same plan's Component 6). Both are nullable at construction (zero cost when
 *     disabled, the default) and start with a zero-init gate (a strict forward-pass no-op even
 *     when allocated, until training opens the gate) — see forward()'s own doc comment for the
 *     full backward-compatibility guarantee. CPU forward()/backward() only; forward_with_cache()
 *     and the GPU path are unchanged and do not support either gated path yet.
 *   - Position-wise feed-forward transformation
 *   - Three residual connections for gradient flow (five when both gated paths are active)
 *   - Three layer normalizations for training stability (five when both gated paths are active)
 *   - Full backpropagation support
 */
class DecoderBlock {
   private:
    // Core components
    std::unique_ptr<MultiHeadAttention> self_attention;  // Masked self-attention
    std::unique_ptr<CrossAttention> cross_attention;     // Attend to encoder (new!)
    std::unique_ptr<FeedForward> feed_forward;
    std::unique_ptr<LayerNorm> norm1;  // After self-attention
    std::unique_ptr<LayerNorm> norm2;  // After cross-attention
    std::unique_ptr<LayerNorm> norm3;  // After feed-forward

    // TD-180: gated world-model cross-attention path (LeJEPA plan Component 4). Nullable —
    // allocated only when the constructor's enable_world_model flag is true (default false,
    // matching pre-TD-180 behavior exactly: no allocation, no forward-time cost, byte-identical
    // output). gate starts at 0.0f so tanh(gate) == 0 — the path is a strict no-op even once
    // allocated, until training actually opens the gate (see forward()'s own doc comment).
    std::unique_ptr<CrossAttention> world_model_cross_attention;
    std::unique_ptr<LayerNorm> norm_world;
    float gate{0.0f};
    float gate_grad{0.0f};
    bool world_model_path_active_{false};  // set by the most recent forward() call; read by
                                            // backward() to know whether to backprop this path
    Matrix cached_wm_attn;                 // this path's own attention output, for gate's gradient

    // TD-180: gated hippocampal cross-attention path (LeJEPA plan Component 6) — structurally a
    // sibling of the world-model path above, same nullable/zero-init-gate pattern, but the
    // attention scores are additionally penalized by each memory slot's own accumulated
    // coverage before softmax (via CrossAttention::forward_with_scores(), TD-174), which is
    // what makes this a *content*-level repetition penalty rather than the existing
    // token-identity-level one in TextGenerator::apply_repetition_penalty.
    std::unique_ptr<CrossAttention> hippocampal_cross_attention;
    std::unique_ptr<LayerNorm> norm_hippocampal;
    float gate_h{0.0f};
    float gate_h_grad{0.0f};
    bool hippocampal_path_active_{false};
    Matrix cached_hm_attn;

    // Hyperparameters
    int d_model;
    int num_heads;
    int d_ff;
    float dropout_rate;

    // Cached values for backward pass
    Matrix cached_input;
    Matrix cached_self_attn_output;
    Matrix cached_residual1;
    Matrix cached_normed1;
    Matrix cached_cross_attn_output;
    Matrix cached_residual2;
    Matrix cached_normed2;
    Matrix cached_ff_output;
    Matrix cached_residual3;
    Matrix cached_encoder_output;

   public:
    float learning_rate{0.001f};

    /**
     * Get the feed-forward sublayer (for hook registration, diagnostics, etc.)
     *
     * @return Pointer to the internal FeedForward instance
     */
    FeedForward* get_feed_forward() {
        return feed_forward.get();
    }

    /**
     * Get the masked self-attention sublayer (for hook registration, entropy tracking, etc.)
     *
     * @return Pointer to the internal MultiHeadAttention instance
     */
    MultiHeadAttention* get_self_attention() {
        return self_attention.get();
    }

    // ── SafeTensors accessor API ─────────────────────────────────────────────
    CrossAttention* get_cross_attention() {
        return cross_attention.get();
    }
    LayerNorm* get_norm1() {
        return norm1.get();
    }
    LayerNorm* get_norm2() {
        return norm2.get();
    }
    LayerNorm* get_norm3() {
        return norm3.get();
    }

    /** TD-180: current gate value, tanh(gate) applied at forward time — for the
     *  mean(tanh(gate)) metric the Training Standard's Phase 1 fine-tuning pushes per layer. */
    float get_gate() const {
        return gate;
    }

    /** TD-180: hippocampal gate's own current value — same metric role as get_gate(), separate
     *  series. */
    float get_gate_h() const {
        return gate_h;
    }

    /** TD-180: direct gate setters, mainly for tests (gradient checks need a nonzero starting
     *  gate) — TD-187's own load() now sets these directly too, when a saved gate value is
     *  present. Not used by forward() itself; training only ever reaches these values via
     *  update_weights()'s own plain-SGD step. */
    void set_gate(float g) {
        gate = g;
    }
    void set_gate_h(float g) {
        gate_h = g;
    }

    /** TD-180: current accumulated gate gradient — mainly for gradient-check tests (finite
     *  difference vs. this value) and diagnostics. Zeroed by zero_grad()/update_weights(). */
    float get_gate_grad() const {
        return gate_grad;
    }
    float get_gate_h_grad() const {
        return gate_h_grad;
    }

    /** TD-180: nullable sub-component accessors, mirroring get_cross_attention() above.
     *  Return nullptr when the corresponding gated path wasn't enabled at construction. */
    CrossAttention* get_world_model_cross_attention() {
        return world_model_cross_attention.get();
    }
    CrossAttention* get_hippocampal_cross_attention() {
        return hippocampal_cross_attention.get();
    }

    /**
     * Constructor
     *
     * @param d_model Model dimension (embedding size)
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward network hidden dimension
     * @param dropout Dropout rate for regularization (default: 0.1)
     * @param enable_world_model TD-180: allocate the gated world-model cross-attention path
     *   (world_model_cross_attention/norm_world/gate). Default false — matches every
     *   pre-TD-180 caller's behavior exactly (no allocation, forward()'s gated branch is
     *   unreachable regardless of what's passed to it). TD-181's own
     *   world_model_inject_every_n_layers knob is what actually decides this per layer.
     * @param enable_hippocampal TD-180: same, for the gated hippocampal cross-attention path
     *   (hippocampal_cross_attention/norm_hippocampal/gate_h). Default false.
     */
    DecoderBlock(int d_model, int num_heads, int d_ff, float dropout = 0.1f,
                 bool enable_world_model = false, bool enable_hippocampal = false);

    /**
     * Forward pass through decoder block
     *
     * Applies:
     *   1. Masked multi-head self-attention (causal)
     *   2. Residual connection and layer norm
     *   3. Cross-attention to encoder output
     *   4. Residual connection and layer norm
     *   5. TD-180: gated world-model cross-attention (no-op unless both
     *      world_model_output != nullptr AND this instance's own world_model_cross_attention
     *      was allocated at construction)
     *   6. TD-180: gated hippocampal cross-attention, repetition-penalized (no-op unless both
     *      memory != nullptr, memory->size() > 0, AND hippocampal_cross_attention was allocated)
     *   7. Feed-forward network
     *   8. Residual connection and layer norm
     *
     * Backward-compatibility guarantee (per the plan's own Compatibility section): passing
     * nullptr for world_model_output/memory — the default — is bit-identical to pre-TD-180
     * behavior, regardless of whether this instance's gated paths were ever allocated. Even
     * with them allocated and non-null inputs supplied, gate/gate_h start at 0.0f, so
     * tanh(gate) == tanh(gate_h) == 0 and the additive terms are still an exact no-op until
     * training actually opens a gate — see this item's own two dedicated no-op tests.
     *
     * @param input Decoder input [seq_len, d_model]
     * @param encoder_output Encoder output for cross-attention [enc_seq_len, d_model]
     * @param self_attn_mask Causal mask for self-attention [seq_len, seq_len]
     * @param cross_attn_mask Optional padding mask for encoder [seq_len, enc_seq_len]
     * @param world_model_output Frozen world-model encoder output [wm_seq_len, d_model], or
     *   nullptr to skip the gated path entirely.
     * @param world_model_mask Optional padding mask for world_model_output, same convention as
     *   cross_attn_mask.
     * @param memory Hippocampal memory to attend over, or nullptr to skip that path entirely.
     *   read_all()'s key matrix is used as CrossAttention's own kv_input — see this method's
     *   implementation comment for why (both K and V end up derived from the stored keys via
     *   this path's own learned projections; correct in the common case where a slot's value
     *   equals its key, a documented simplification otherwise).
     * @param repetition_alpha Penalty growth rate — score_bias[i][slot] = -repetition_alpha *
     *   coverage[slot], applied uniformly across every query position. 0.0f (default) makes the
     *   penalty a no-op regardless of coverage, independent of the gate itself.
     * @param repetition_decay Per-call coverage decay, 0 < gamma <= 1 (only meaningful when
     *   memory is non-null and non-empty; ignored otherwise). gamma = 1 disables decay.
     * @return Output [seq_len, d_model]
     */
    Matrix forward(const Matrix& input, const Matrix& encoder_output, const Matrix& self_attn_mask,
                   const Matrix* cross_attn_mask = nullptr,
                   const Matrix* world_model_output = nullptr,
                   const Matrix* world_model_mask = nullptr, HippocampalMemory* memory = nullptr,
                   float repetition_alpha = 0.0f, float repetition_decay = 0.95f);

    /**
     * Forward pass with KV cache support (for inference optimization)
     *
     * Same as forward() but with caching for autoregressive generation.
     *
     * @param input Decoder input (new tokens only) [num_new_tokens, d_model]
     * @param encoder_output Encoder output for cross-attention [enc_seq_len, d_model]
     * @param self_attn_mask Causal mask [num_new_tokens, total_seq_len]
     * @param self_attn_cache Cache for self-attention K/V pairs
     * @param cross_attn_cache Cache for cross-attention K/V pairs (computed once from encoder)
     * @param cross_attn_mask Optional padding mask for encoder
     * @param use_cache If true, update caches
     * @return Output [num_new_tokens, d_model]
     */
    Matrix forward_with_cache(const Matrix& input, const Matrix& encoder_output,
                              const Matrix& self_attn_mask, KVCache* self_attn_cache,
                              KVCache* cross_attn_cache, const Matrix* cross_attn_mask = nullptr,
                              bool use_cache = true);

    /**
     * Backward pass through decoder block
     *
     * Computes gradients for all parameters and returns gradient w.r.t. input.
     * Gradients flow through:
     *   1. Third residual connection (split gradient)
     *   2. Feed-forward network (backward)
     *   3. Third layer norm (backward)
     *   4. TD-180: gated hippocampal path (backward, only if it was active in the most recent
     *      forward() call — see world_model_path_active_/hippocampal_path_active_)
     *   5. TD-180: gated world-model path (backward, same condition)
     *   6. Second residual connection (split gradient)
     *   7. Cross-attention (backward)
     *   8. Second layer norm (backward)
     *   9. First residual connection (split gradient)
     *   10. Self-attention (backward)
     *   11. First layer norm (backward)
     *
     * TD-180's own gated paths' K/V-source gradients (w.r.t. world_model_output and the
     * hippocampal memory's keys) are computed (required by CrossAttention::backward()'s own
     * signature) but deliberately discarded, not returned via any new out-param: the world
     * model is frozen during Phase 1 fine-tuning (LeJEPAEncoder::set_requires_grad(false)) and
     * HippocampalMemory's stored keys are never learnable parameters, so nothing currently
     * needs either gradient. This keeps both backward() signatures below completely unchanged.
     *
     * @param grad_output Gradient from next layer [seq_len, d_model]
     * @param grad_encoder_output Out-param: gradient w.r.t. this block's cross-attention
     *   encoder input (K/V source), overwritten (not accumulated) by this call.
     * @return Gradient w.r.t. input [seq_len, d_model]
     */
    Matrix backward(const Matrix& grad_output, Matrix& grad_encoder_output);

    /** Convenience overload: discards the encoder-side gradient. */
    Matrix backward(const Matrix& grad_output);

    /**
     * Update weights using accumulated gradients
     *
     * Updates all sub-components: attention layers, feed-forward, layer norms
     */
    void update_weights();

    /**
     * Zero accumulated gradients
     *
     * Clears gradients in all sub-components, plus gate_grad/gate_h_grad for whichever gated
     * paths were allocated at construction (TD-180).
     */
    void zero_grad();

    /**
     * Compute the L2 norm of accumulated gradients across self-attention,
     * cross-attention, and feed-forward sub-components (combined in
     * quadrature). LayerNorm gradients are omitted, matching
     * EncoderBlock::get_gradient_norm()'s approximation.
     *
     * @return sqrt(sum of squared per-component gradient norms)
     */
    float get_gradient_norm() const;

    /**
     * Set learning rate for all sub-components
     *
     * @param lr New learning rate
     */
    void set_learning_rate(float lr);

    /**
     * Save decoder block parameters to file.
     *
     * TD-187 (closing TD-180's own documented gap): the gated world-model/hippocampal
     * sub-components — `world_model_cross_attention`/`norm_world`/`gate` and
     * `hippocampal_cross_attention`/`norm_hippocampal`/`gate_h` — are now persisted too, but
     * only when this instance actually has them allocated (`enable_world_model`/
     * `enable_hippocampal` at construction). A one-byte presence flag is written to the main
     * file immediately before each optional section (mirroring `LeJEPAEncoder::save()`'s own
     * "header record before variable content" convention, and `HippocampalMemory::save()`'s own
     * `num_slots` count) so `load()` knows whether a gate scalar follows and whether a
     * `.world_model_cross_attn`/`.norm_world` (or `.hippocampal_cross_attn`/
     * `.norm_hippocampal`) file pair was written — a call site never needs to guess. See
     * `load()`'s own doc comment for what happens when the saved flags don't match this
     * instance's own construction.
     *
     * @param filepath Path to save file
     */
    void save(const std::string& filepath);

    /**
     * Load decoder block parameters from file.
     *
     * TD-187: the saved presence flags for the gated world-model/hippocampal paths (see save()'s
     * own doc comment) must match this instance's own construction-time
     * `enable_world_model`/`enable_hippocampal` state exactly — a saved-present-but-
     * constructed-absent (or vice versa) mismatch throws `std::runtime_error`, the same
     * "fail clearly rather than silently drop or fabricate trained state" choice this method
     * already makes for a `d_model`/`num_heads`/`d_ff` mismatch. This is deliberate: silently
     * skipping a saved gated path would discard real trained gate/cross-attention weights with
     * no warning, and silently leaving an allocated-but-absent-from-the-file gated path at its
     * fresh random-init state would silently break a resumed training run in a way nothing
     * downstream could detect. The caller's own fix is always the same as for a dimension
     * mismatch: construct this instance with the same `enable_world_model`/`enable_hippocampal`
     * flags the checkpoint was originally saved with.
     *
     * @param filepath Path to load file
     */
    void load(const std::string& filepath);

    /**
     * Register all decoder block parameters with optimizer, including the gated paths' own
     * CrossAttention/LayerNorm sub-components when allocated (TD-180). gate/gate_h themselves
     * are NOT registered — they're plain floats with no ParameterGroup equivalent in
     * Optimizer's Matrix-based registration API, so they're always updated via plain SGD in
     * update_weights() regardless of whether an optimizer is registered here.
     *
     * @param optimizer Optimizer to register parameters with
     */
    void register_parameters_with_optimizer(class Optimizer& optimizer);

#ifdef ADAI_ENABLE_GPU
    void gpu_upload_weights();
    void gpu_download_grads();
    void gpu_zero_grads();
    // encoder_out: cached GPU encoder output [src_len, d_model]
    adai::gpu::GPUMatrix gpu_forward(const adai::gpu::GPUMatrix& input,
                                     const adai::gpu::GPUMatrix& encoder_out,
                                     const adai::gpu::GPUMatrix* self_mask = nullptr);
    // Returns {grad_input, grad_encoder_output} — gradient w.r.t. decoder input,
    // and gradient w.r.t. this block's cross-attention encoder input.
    std::pair<adai::gpu::GPUMatrix, adai::gpu::GPUMatrix> gpu_backward(
        const adai::gpu::GPUMatrix& dout);

    /**
     * @brief TD-050: GPU incremental decode using GPUKVCache — mirrors gpu_forward()'s own
     * Pre-LN residual structure (norm1 -> self-attn -> residual -> norm2 -> cross-attn ->
     * residual -> norm3 -> feed-forward -> residual) exactly, swapping in
     * MultiHeadAttention::gpu_forward_with_cache()/CrossAttention::gpu_forward_with_cache() for
     * the two attention sub-layers. Feed-forward and every norm are unchanged — nothing about
     * them depends on sequence position, only the attention sub-layers need a cache at all.
     * Inference-only, no backward() counterpart (matches the attention layers' own
     * gpu_forward_with_cache() contract).
     *
     * @param input New-token(s) input [num_new_tokens, d_model]
     * @param encoder_out Raw encoder output — only read on cross_attn_cache's first (cache-
     *   populating) call; safe to pass every time regardless
     * @param self_attn_mask Causal mask [num_new_tokens, total_seq_len_after_append]
     * @param self_attn_cache This block's own self-attention cache (grows every call)
     * @param cross_attn_cache This block's own cross-attention cache (populated once, reused)
     */
    adai::gpu::GPUMatrix gpu_forward_with_cache(const adai::gpu::GPUMatrix& input,
                                                const adai::gpu::GPUMatrix& encoder_out,
                                                const adai::gpu::GPUMatrix& self_attn_mask,
                                                adai::gpu::GPUKVCache* self_attn_cache,
                                                adai::gpu::GPUKVCache* cross_attn_cache,
                                                bool use_cache = true);
#endif
};
