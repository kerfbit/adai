#pragma once

// @adai-status: experimental
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-17

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "BPETokenizer.hpp"
#include "EncoderBlock.hpp"
#include "LayerNorm.hpp"
#include "Matrix.hpp"
#include "Optimizer.hpp"
#include "PositionalEncoding.hpp"
#include "Predictor.hpp"
#include "SIGReg.hpp"
#include "TokenEmbedding.hpp"

/**
 * LeJEPAEncoder (LeJEPA world-model plan, LJ-2a/LJ-2b / TD-177/TD-178)
 *
 * Self-supervised world-model encoder. Structurally a transformer encoder stack — same
 * composition as LLMEncoder (BPETokenizer, TokenEmbedding, PositionalEncoding, a stack of
 * EncoderBlock, a final LayerNorm) — the difference from LLMEncoder is entirely in what it is
 * trained on and with what objective (train_step(), below), not in construction.
 * See docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 1.
 *
 * TD-177 scope (construction): constructor, encode(), save()/load(), print_config(), and the
 * plumbing methods (set_requires_grad/set_learning_rate/register_parameters_with_optimizer)
 * needed for a training loop to attach to — mirroring how little "logic" the equivalent methods
 * on LLMEncoder itself contain.
 *
 * TD-178 scope (this training loop): train_step(text) constructs two augmented "views" of the
 * same token sequence via span masking — a context view (a contiguous span replaced by the
 * tokenizer's own unk token) and the target view (the original, unmasked sequence) — encodes
 * both, predicts the context view's embedding forward to the target view's via `predictor`
 * (scored only at the masked span's own positions, the only positions where prediction is
 * actually informative — an unmasked position's context-view token is literally identical to
 * its target-view counterpart), and applies `sigreg` to the target view's embeddings to keep
 * this encoder's output isotropic-Gaussian (see SIGReg's own class doc for why). Both loss
 * terms' gradients flow into the *same* shared encoder weights from both the context and
 * target forward passes — LeJEPA's own point (per the plan's Background section) is removing
 * the need for a stop-gradient/EMA-teacher asymmetry a classic Siamese-network JEPA setup would
 * otherwise need. `train_step()` is a fully self-contained training step (forward + backward +
 * weight update in one call, returning `{predictor_loss, sigreg_loss}` for the caller to log) —
 * matching the plan's own public interface, which does not expose backward()/zero_grad()/
 * update_weights() at all, only train_step() itself as the training entry point.
 *
 * encode()'s output is a drop-in match for LLMEncoder::encode()'s own shape contract
 * ([seq_len, d_model]) — required by TD-179's `HippocampalMemory`, which stores/reads keys
 * produced by whichever encoder a caller passes in.
 *
 * TD-189: both loss terms' gradients flow into the same shared encoder weights, but as two
 * *sequential* weight updates within one train_step() call, not one joint combined update —
 * the target view's gradient (SIGReg + the predictor's target-side term) is applied first, then
 * the context view's (the predictor's context-side term). This is a deliberate consequence of
 * every shared component's own backward() overwriting (not accumulating) its gradient members
 * across separate calls; see train_step()'s own implementation comment for the full rationale.
 * Still no stop-gradient/EMA-teacher asymmetry either way — LeJEPA's own point (per the plan's
 * Background section).
 */
class LeJEPAEncoder {
   private:
    std::unique_ptr<BPETokenizer> tokenizer;  // own instance; shared vocab with LLMEncoder means
                                               // loading the same vocab file, not a shared pointer
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> positional_encoding;
    std::vector<std::unique_ptr<EncoderBlock>> encoder_blocks;
    std::unique_ptr<LayerNorm> final_norm;
    std::unique_ptr<Predictor> predictor;  // embedding-space predictor, used by train_step()
    std::unique_ptr<SIGReg> sigreg;        // isotropic-Gaussian regularizer, used by train_step()

    int vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length;
    bool requires_grad{true};
    float learning_rate{0.001f};
    float sigreg_lambda{1.0f};  // λ weighting SIGReg's gradient contribution in train_step()

    // Set by register_parameters_with_optimizer(); nullptr = plain-SGD fallback. Tracked here
    // (not just delegated to each sub-component's own optimizer pointer) so update_weights()
    // can call optimizer_->step() exactly ONCE per update_weights() call when a shared optimizer
    // is in use — see update_weights()'s own doc comment for why calling every sub-component's
    // own update_weights() unconditionally would be a real bug here (each would independently
    // call the SAME shared Optimizer's step(), over-applying it once per sub-component instead
    // of once per update_weights() call). TD-189: train_step() itself now calls update_weights()
    // (hence optimizer_->step()) TWICE — once per view's gradient, see train_step()'s own
    // implementation comment — so "once" here is per update_weights() call, not per train_step().
    Optimizer* optimizer_{nullptr};

    // Cached value for backward()'s own token_embedding->backward() call, same
    // guard-by-requires_grad convention as LLMEncoder's own cache. Reused across both the
    // context and target forward passes within one train_step() call — see train_step()'s own
    // implementation comment for the ordering this requires.
    std::vector<int> cached_token_ids;

    // Seeded once at construction (one std::random_device query) rather than per train_step()
    // call — train_step()'s own span-start draw is the only user.
    std::mt19937 span_rng_{std::random_device{}()};

    // Diagnostic state set inside train_step(), valid only immediately after the most recent
    // call (encode() does not update these) — see the getters below for what each one means.
    float last_gradient_norm_{0.0f};
    float last_masking_ratio_{0.0f};
    float last_predictor_target_cosine_sim_{0.0f};
    float last_sigreg_variance_mean_{0.0f};
    float last_sigreg_variance_stddev_{0.0f};

    /** Core forward pipeline shared by encode() and train_step() — embedding, positional
     *  encoding, encoder blocks, final norm. Populates the cache above when requires_grad. */
    Matrix encode_tokens(const std::vector<int>& token_ids);

    /** Backward pass through this encoder alone (final_norm -> encoder_blocks reverse ->
     *  token_embedding), mirroring LLMEncoder::backward() exactly. No-op if !requires_grad.
     *  Private: only train_step() calls this, matching the plan's own public interface, which
     *  exposes no backward() at all. */
    void backward(const Matrix& grad_output);

    /** Zeroes every sub-component's accumulated gradients, predictor included (SIGReg has none
     *  of its own — see its class doc). */
    void zero_grad();

    /**
     * Applies accumulated gradients. With a registered optimizer (optimizer_ != nullptr): calls
     * optimizer_->step() exactly once, then zero_grad() — mirroring the apply-then-clear
     * sequence every individual component's own update_weights() already follows internally,
     * but done once for the whole encoder rather than once per sub-component (each
     * sub-component's own update_weights() would otherwise each independently re-invoke the
     * SAME shared Optimizer::step(), which iterates every one of its registered parameter
     * groups on every call — calling it N times per training step would apply N full optimizer
     * steps to the entire model instead of one). Without a registered optimizer: delegates to
     * each sub-component's own update_weights() (each independently falls back to its own
     * plain-SGD path and auto-zeros its own gradients — no shared mutable state to conflict
     * over in that path, so no equivalent hazard).
     */
    void update_weights();

   public:
    /**
     * @param vocab_size Size of the vocabulary
     * @param d_model Dimension of the model embeddings
     * @param num_layers Number of encoder layers
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward dimension — also used as the Predictor's hidden_dim (see
     *             constructor doc in LeJEPAEncoder.cpp for why no separate parameter exists)
     * @param max_seq_length Maximum sequence length
     * @param sigreg_num_sketches Number of random sketch projections `sigreg` uses internally
     *        (TD-183's WORLD_MODEL_SIGREG_NUM_SKETCHES config key). Default 64 matches SIGReg's
     *        own constructor default byte-for-byte, so every pre-existing caller that doesn't
     *        pass this is unaffected — added as a trailing parameter (TD-177/TD-178 both predate
     *        this) rather than, say, a setter, because SIGReg's sketch directions are fixed at
     *        its own construction time (see SIGReg's class doc) with no way to resize them
     *        afterward.
     */
    LeJEPAEncoder(int vocab_size, int d_model = 512, int num_layers = 6, int num_heads = 8,
                  int d_ff = 2048, int max_seq_length = 512, int sigreg_num_sketches = 64);

    /**
     * Encode one view of input text to contextualized embeddings — same shape contract as
     * LLMEncoder::encode(): [seq_len, d_model], so this is a drop-in for anything that only
     * needs embeddings (e.g. HippocampalMemory's key/value source, TD-179).
     */
    Matrix encode(const std::string& text);

    /**
     * Self-supervised training step on a single example (TD-178) — see the class doc above for
     * the full view-construction/loss/gradient-flow design. Fully self-contained: forward,
     * backward, and weight updates all happen inside this one call (TD-189: two sequential
     * updates, one per view's gradient — see the class doc and this method's own implementation
     * comment for why).
     *
     * @param text Raw text — no paired target required, unlike EncoderDecoderModel::train_step.
     * @return {predictor_loss, sigreg_loss}, logged as two separate series (see
     *         TrainingMetricsService::update_lejepa_metrics()) rather than summed into one
     *         number, so each term's own trend is independently visible.
     * @throws std::invalid_argument if `text` tokenizes to fewer than 2 tokens — span masking
     *         needs at least one context token and one target token to be meaningful.
     */
    std::pair<float, float> train_step(const std::string& text);

    /** λ weighting SIGReg's gradient contribution against the predictor's own, inside
     *  train_step(). Corresponds to the plan's WORLD_MODEL_SIGREG_LAMBDA config key
     *  (TD-183's job to actually read that key and call this setter). */
    void set_sigreg_lambda(float lambda) {
        sigreg_lambda = lambda;
    }

    float get_sigreg_lambda() const {
        return sigreg_lambda;
    }

    /** Number of random sketch projections `sigreg` was constructed with (TD-183's
     *  WORLD_MODEL_SIGREG_NUM_SKETCHES) — fixed at construction, see the constructor's own doc
     *  comment for why there's no matching setter. */
    int get_sigreg_num_sketches() const {
        return sigreg->get_num_sketches();
    }

    /** Load tokenizer vocabulary from file — same convention as LLMEncoder. */
    void load_tokenizer_vocab(const std::string& vocab_file);

    /** Build tokenizer from a text corpus — same convention as LLMEncoder. */
    void build_tokenizer(const std::vector<std::string>& corpus, int vocab_size = 10000);

    /** Frozen (false) after the pretraining phase completes — see the plan's Training Standard. */
    void set_requires_grad(bool requires_grad);

    void set_learning_rate(float lr);

    /** Registers token_embedding/encoder_blocks/final_norm/predictor with a shared optimizer.
     *  SIGReg has no learnable parameters of its own (see its own class doc), so nothing to
     *  register for it. */
    void register_parameters_with_optimizer(Optimizer& optimizer);

    /**
     * Persist the encoder itself — token_embedding, encoder_blocks, final_norm — to `directory`
     * (created if absent), mirroring ModelSerializer's own "directory, created if absent"
     * convention rather than LLMEncoder's filename-with-suffix one. `predictor`/`sigreg` are
     * deliberately NOT persisted: they are pretraining scaffolding whose job ends once this
     * encoder is frozen (see the class doc above and the plan's Component 2/3 docs) — the
     * reusable downstream artifact is the encoder's own weights, exactly what LLMEncoder's own
     * save_weights() persists for itself.
     */
    void save(const std::string& directory) const;

    /** @throws std::runtime_error if `directory` is missing its config or the saved architecture
     *  does not match this instance's own (vocab_size/d_model/num_layers/num_heads/d_ff). */
    void load(const std::string& directory);

    void print_config() const;

    int get_embedding_dim() const {
        return d_model;
    }

    int get_num_layers() const {
        return num_layers;
    }

    /**
     * Get encoder block at a specific layer (for hook registration, diagnostics, etc.) —
     * mirrors LLMEncoder::get_encoder_block().
     *
     * @throws std::out_of_range if layer is outside [0, num_layers)
     */
    EncoderBlock* get_encoder_block(int layer) {
        if (layer < 0 || layer >= num_layers) {
            throw std::out_of_range("Layer index out of range");
        }
        return encoder_blocks[layer].get();
    }

    // ── Advanced diagnostics (post-TD-208 metrics work) ─────────────────────────────────────
    // All five valid only immediately after the most recent train_step() call.

    /** Combined gradient norm (sqrt of sum of squares, matching EncoderBlock::get_gradient_norm()'s
     *  own combining convention) across train_step()'s two internal sequential updates (TD-189),
     *  captured right before each update_weights() call zeroes it — calling optimizer_->
     *  get_gradient_norm() from OUTSIDE train_step() after it returns always reads 0.0f, since
     *  update_weights() already zeroed every gradient buffer by then. Returns 0.0f when no
     *  optimizer is registered (the plain-SGD fallback path has no single combined-norm primitive
     *  across every sub-component, and production always registers an optimizer). */
    float get_last_gradient_norm() const {
        return last_gradient_norm_;
    }

    /** Actual span-masking ratio applied this step (span_len / seq_len) — can deviate from the
     *  nominal 25% for short sequences due to train_step()'s own [1, seq_len-1] clamp. */
    float get_last_masking_ratio() const {
        return last_masking_ratio_;
    }

    /** Cosine similarity between the predictor's output and the target-view embedding, averaged
     *  over the masked span — isolates directional prediction quality from the MSE predictor_loss's
     *  magnitude-conflated signal. Range [-1, 1]. */
    float get_last_predictor_target_cosine_sim() const {
        return last_predictor_target_cosine_sim_;
    }

    /** Mean per-direction projected variance of the target-view embeddings across sigreg's own
     *  sketch directions (see SIGReg::compute_variance_stats()) — target ~1.0 for an appropriately
     *  spread (isotropic) batch. Complementary to sigreg_loss: this says "how spread out overall,"
     *  not "how close to the analytic Gaussian shape." */
    float get_last_sigreg_variance_mean() const {
        return last_sigreg_variance_mean_;
    }

    /** Standard deviation of that same per-direction variance across directions — near 0 means
     *  spread is uniform across directions; large means collapsed in specific directions even if
     *  the mean looks healthy, a distinction sigreg_loss's aggregate scalar can't make. */
    float get_last_sigreg_variance_stddev() const {
        return last_sigreg_variance_stddev_;
    }
};
