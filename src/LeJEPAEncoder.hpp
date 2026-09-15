#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-15

#include <memory>
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
 * LeJEPAEncoder (LeJEPA world-model plan, LJ-2a / TD-177)
 *
 * Self-supervised world-model encoder. Structurally a transformer encoder stack — same
 * composition as LLMEncoder (BPETokenizer, TokenEmbedding, PositionalEncoding, a stack of
 * EncoderBlock, a final LayerNorm) — the difference from LLMEncoder is entirely in what it is
 * trained on and with what objective (LeJEPAEncoder::train_step, TD-178), not in construction.
 * See docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 1.
 *
 * TD-177 scope: construction, encode(), save()/load(), print_config(), and the plumbing methods
 * (set_requires_grad/set_learning_rate/register_parameters_with_optimizer) needed for a later
 * training loop to attach to — mirroring exactly how little "logic" the equivalent methods on
 * LLMEncoder itself contain. `train_step()` (the actual self-supervised training loop) is TD-178's
 * job, deliberately not implemented here; this class holds `predictor_`/`sigreg_` as constructed
 * members (per the plan's own class spec) purely so TD-178 can add `train_step()` without also
 * needing to touch this constructor.
 *
 * encode()'s output is a drop-in match for LLMEncoder::encode()'s own shape contract
 * ([seq_len, d_model]) — required by TD-179's `HippocampalMemory`, which stores/reads keys
 * produced by whichever encoder a caller passes in.
 */
class LeJEPAEncoder {
   private:
    std::unique_ptr<BPETokenizer> tokenizer;  // own instance; shared vocab with LLMEncoder means
                                               // loading the same vocab file, not a shared pointer
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> positional_encoding;
    std::vector<std::unique_ptr<EncoderBlock>> encoder_blocks;
    std::unique_ptr<LayerNorm> final_norm;
    std::unique_ptr<Predictor> predictor;  // embedding-space predictor — used by train_step()
                                            // (TD-178), constructed here only
    std::unique_ptr<SIGReg> sigreg;        // isotropic-Gaussian regularizer — same as above

    int vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length;
    bool requires_grad{true};
    float learning_rate{0.001f};
    float sigreg_lambda{1.0f};  // λ weighting SIGReg term against predictor loss in train_step()

    // Cached values for a future backward pass (TD-178) — same fields, same guard-by-
    // requires_grad convention as LLMEncoder's own cache, populated by encode() below so
    // train_step() can add a private backward() later without changing encode() at all.
    std::vector<int> cached_token_ids;
    std::vector<Matrix> cached_encoder_outputs;

   public:
    /**
     * @param vocab_size Size of the vocabulary
     * @param d_model Dimension of the model embeddings
     * @param num_layers Number of encoder layers
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward dimension — also used as the Predictor's hidden_dim (see
     *             constructor doc in LeJEPAEncoder.cpp for why no separate parameter exists)
     * @param max_seq_length Maximum sequence length
     */
    LeJEPAEncoder(int vocab_size, int d_model = 512, int num_layers = 6, int num_heads = 8,
                  int d_ff = 2048, int max_seq_length = 512);

    /**
     * Encode one view of input text to contextualized embeddings — same shape contract as
     * LLMEncoder::encode(): [seq_len, d_model], so this is a drop-in for anything that only
     * needs embeddings (e.g. HippocampalMemory's key/value source, TD-179).
     */
    Matrix encode(const std::string& text);

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
};
