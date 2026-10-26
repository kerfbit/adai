// @adai-status: beta        (capped by TD-050 — see TECHNICAL_DEBT.md; TD-038 LoRA support added; TD-050 GPU incremental-cache generation wired in; TD-050 CPU beam-vs-cache guard added to generate_response()/generate_response_with_strategy(); TD-182 set_world_model()/get_world_model() added; TD-185 set_hippocampal_memory()/get_hippocampal_memory() + write-policy call site added; TD-186 wired world-model/hippocampal read paths into forward()/generate_response()'s beam branches)
// @adai-version: 0.24.1
// @adai-reviewed: 2026-10-26

#include "EncoderDecoderModel.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <sstream>
#include <stdexcept>
#include "HippocampalMemory.hpp"  // TD-185: forward-declared in the header, complete type needed here
#include "LeJEPAEncoder.hpp"      // TD-182: forward-declared in the header, complete type needed here
#include "Logger.hpp"
#include "Optimizer.hpp"
#include "SpecialTokens.hpp"
using adai::Logger;

namespace {

// TD-424: the last row of a decoder output, which is all a generation step needs projected.
Matrix last_row(const Matrix& m) {
    Matrix row(1, m.cols);
    if (m.rows > 0)
        row.data[0] = m.data[m.rows - 1];
    return row;
}

// TD-347: generation stops at max_seq_length. The generated sequence (BOS included) is the
// decoder's input, and positions past max_seq_length have no positional encoding: the model never
// trained on them. requested <= 0 means "the stored length", still capped.
int capped_generation_length(int requested, int stored, int max_seq_length) {
    const int length = requested > 0 ? requested : stored;
    return max_seq_length > 0 ? std::min(length, max_seq_length) : length;
}

// TD-359: applies one call's max_length to the generator, whose stored config is otherwise what
// TextGenerator reads, and puts the stored value back when the call returns. max_length <= 0 keeps
// the stored value. Either way the length is capped at max_seq_length (TD-347).
class ScopedMaxLength {
   public:
    ScopedMaxLength(TextGenerator& generator, int max_length, int max_seq_length)
        : generator_(generator), previous_(generator.get_config().max_length) {
        auto config = generator_.get_config();
        config.max_length = capped_generation_length(max_length, previous_, max_seq_length);
        generator_.set_config(config);
    }
    ~ScopedMaxLength() {
        auto config = generator_.get_config();
        config.max_length = previous_;
        generator_.set_config(config);
    }
    ScopedMaxLength(const ScopedMaxLength&) = delete;
    ScopedMaxLength& operator=(const ScopedMaxLength&) = delete;

   private:
    TextGenerator& generator_;
    int previous_;
};

// TD-343: commits the hippocampal memory's coverage/association update once per decode step.
// The forwards only read the memory now; a generation path observes each decoder call's step
// weights (the newest token's attention, averaged over injected layers) and the token length it
// was for. Calls at the same length (beam hypotheses for one step) are averaged into a single
// update, committed when the length advances or at flush().
class HippocampalStepCommitter {
   public:
    HippocampalStepCommitter(HippocampalMemory* memory, float repetition_decay,
                             float association_decay)
        : memory_(memory != nullptr && memory->size() > 0 ? memory : nullptr),
          repetition_decay_(repetition_decay),
          association_decay_(association_decay) {}

    bool active() const {
        return memory_ != nullptr;
    }

    void observe(size_t step_length, const Matrix& weights) {
        if (memory_ == nullptr || weights.rows == 0) {
            return;
        }
        if (count_ > 0 && step_length != step_length_) {
            flush();
        }
        if (count_ == 0) {
            sum_ = Matrix(1, weights.cols);
            step_length_ = step_length;
        } else if (weights.cols != sum_.cols) {
            return;
        }
        for (int j = 0; j < weights.cols; ++j) {
            sum_(0, j) += weights(weights.rows - 1, j);
        }
        ++count_;
    }

    void flush() {
        if (memory_ == nullptr || count_ == 0) {
            return;
        }
        for (int j = 0; j < sum_.cols; ++j) {
            sum_(0, j) /= static_cast<float>(count_);
        }
        memory_->commit_attention(sum_, repetition_decay_, association_decay_);
        count_ = 0;
    }

   private:
    HippocampalMemory* memory_;
    float repetition_decay_;
    float association_decay_;
    Matrix sum_;
    size_t step_length_ = 0;
    int count_ = 0;
};
// Checkpoint .config format marker. Negative so it can never collide with a
// legacy (pre-marker) file's first field, vocab_size, which is always > 0 —
// this gives an unambiguous rejection instead of a coincidental dimension
// mismatch. Bump kConfigFormatVersion on any future change to the
// EncoderBlock/DecoderBlock forward-pass math that makes existing checkpoint
// weights semantically incompatible even though their shapes are unchanged
// (e.g. this version: switching Post-LN -> Pre-LN).
constexpr int32_t kConfigMagic = -20260721;
constexpr int32_t kConfigFormatVersion = 2;  // 2 = Pre-LN EncoderBlock/DecoderBlock
}  // namespace

// Constructor
EncoderDecoderModel::EncoderDecoderModel(int vocab_size, int d_model, int encoder_layers,
                                         int decoder_layers, int num_heads, int d_ff,
                                         int max_seq_length,
                                         int world_model_inject_every_n_layers)
    : vocab_size(vocab_size),
      d_model(d_model),
      encoder_layers(encoder_layers),
      decoder_layers(decoder_layers),
      num_heads(num_heads),
      d_ff(d_ff),
      max_seq_length(max_seq_length) {
    // Initialize tokenizer (will be set externally or use default)
    tokenizer = std::make_unique<BPETokenizer>();

    // Initialize encoder
    encoder = std::make_unique<LLMEncoder>(vocab_size, d_model, encoder_layers, num_heads, d_ff,
                                           max_seq_length);

    // Initialize decoder
    decoder = std::make_unique<LLMDecoder>(vocab_size, d_model, decoder_layers, num_heads, d_ff,
                                           max_seq_length, world_model_inject_every_n_layers);

    // Initialize language model head
    lm_head = std::make_unique<LanguageModelHead>(d_model, vocab_size);

    // Initialize text generator with default config
    TextGenerator::GenerationConfig gen_config;
    gen_config.max_length = max_seq_length;
    gen_config.bos_token_id = adai::SpecialTokenIDs::BOS;  // <bos> token
    gen_config.eos_token_id = adai::SpecialTokenIDs::EOS;  // <eos> token
    gen_config.pad_token_id = adai::SpecialTokenIDs::PAD;  // <pad> token

    generator = std::make_unique<TextGenerator>(gen_config, 42);

    // Set special token IDs
    bos_token_id = gen_config.bos_token_id;
    eos_token_id = gen_config.eos_token_id;
    pad_token_id = gen_config.pad_token_id;
}

// Destructor
EncoderDecoderModel::~EncoderDecoderModel() {
    // Unique pointers handle cleanup
}

void EncoderDecoderModel::set_world_model(std::unique_ptr<LeJEPAEncoder> wm) {
    world_model = std::move(wm);
}

void EncoderDecoderModel::set_hippocampal_memory(std::unique_ptr<HippocampalMemory> hm) {
    hippocampal_memory = std::move(hm);
}

void EncoderDecoderModel::maybe_write_hippocampal_memory(const std::string& response_text) {
    if (!world_model || !hippocampal_memory || response_text.empty() ||
        !hippocampal_updates_enabled_) {  // TD-343
        return;
    }

    try {
        // [seq_len, d_model]. TD-342: on the device when the GPU is active; only the encoding
        // comes back for the host-side memory.
#ifdef ADAI_ENABLE_GPU
        Matrix embeddings = adai::gpu::GPUManager::is_available()
                                ? Matrix::from_gpu(world_model->gpu_encode(response_text))
                                : world_model->encode(response_text);
#else
        Matrix embeddings = world_model->encode(response_text);
#endif
        if (embeddings.rows == 0) {
            return;
        }

        // Mean-pool across the sequence dimension — see this method's own header doc comment.
        Matrix pooled(1, embeddings.cols);
        for (int j = 0; j < embeddings.cols; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < embeddings.rows; ++i) {
                sum += embeddings(i, j);
            }
            pooled(0, j) = sum / static_cast<float>(embeddings.rows);
        }

        hippocampal_memory->write(pooled, pooled);
    } catch (const std::exception& e) {
        // Best-effort: an episodic-memory write failure must not take down a response that
        // already generated successfully — see this method's own header doc comment.
        Logger::warn("EncoderDecoderModel: hippocampal memory write skipped ({})", e.what());
    }
}

// Compute cross-entropy loss
float EncoderDecoderModel::compute_loss(const Matrix& logits,
                                        const std::vector<int>& target_tokens) {
    int seq_length = logits.rows;
    float total_loss = 0.0f;

    for (int t = 0; t < seq_length && t < static_cast<int>(target_tokens.size()); ++t) {
        // Softmax for current timestep
        // TODO: See TD-216 in TECHNICAL_DEBT.md - inline copy of Activation::softmax.
        std::vector<float> probs(vocab_size);
        float max_logit = logits.data[t][0];
        for (int v = 1; v < vocab_size; ++v) {
            max_logit = std::max(max_logit, logits.data[t][v]);
        }

        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            probs[v] = std::exp(logits.data[t][v] - max_logit);
            sum_exp += probs[v];
        }

        for (int v = 0; v < vocab_size; ++v) {
            probs[v] /= sum_exp;
        }

        // Cross-entropy loss for this timestep
        int target = target_tokens[t];
        if (target >= 0 && target < vocab_size) {
            total_loss -= std::log(probs[target] + 1e-10f);
        }
    }

    return total_loss / static_cast<float>(seq_length);
}

// Compute loss gradient
Matrix EncoderDecoderModel::compute_loss_gradient(const Matrix& logits,
                                                  const std::vector<int>& target_tokens) {
    int seq_length = logits.rows;
    Matrix grad(seq_length, vocab_size);

    for (int t = 0; t < seq_length && t < static_cast<int>(target_tokens.size()); ++t) {
        // Softmax
        // TODO: See TD-216 in TECHNICAL_DEBT.md - inline copy of Activation::softmax.
        std::vector<float> probs(vocab_size);
        float max_logit = logits.data[t][0];
        for (int v = 1; v < vocab_size; ++v) {
            max_logit = std::max(max_logit, logits.data[t][v]);
        }

        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            probs[v] = std::exp(logits.data[t][v] - max_logit);
            sum_exp += probs[v];
        }

        for (int v = 0; v < vocab_size; ++v) {
            probs[v] /= sum_exp;
        }

        // Gradient: softmax - one_hot(target)
        int target = target_tokens[t];
        for (int v = 0; v < vocab_size; ++v) {
            grad.data[t][v] = probs[v];
            if (v == target && target >= 0 && target < vocab_size) {
                grad.data[t][v] -= 1.0f;
            }
        }

        // Scale by sequence length
        for (int v = 0; v < vocab_size; ++v) {
            grad.data[t][v] /= static_cast<float>(seq_length);
        }
    }

    return grad;
}

// Generate response
std::string EncoderDecoderModel::generate_response(const std::string& input_text, int max_length) {
    // TD-156: serializes this whole (potentially multi-token) generation against any other
    // thread's forward()/backward()/generate_response()/generate_response_with_strategy() call
    // on this same instance — see model_mutex_'s doc comment in EncoderDecoderModel.hpp.
    std::lock_guard<std::mutex> lock(model_mutex_);
    // TD-359: this call's max_length (it used to be ignored: generate_response("hello world", 5)
    // produced 9 tokens with the generator at 64).
    const ScopedMaxLength scoped_max_length(*generator, max_length, max_seq_length);

    // Encode input (no special tokens for encoder input)
    // TODO: See TD-365 in TECHNICAL_DEBT.md - never truncated to max_seq_length
    std::vector<int> input_tokens = tokenizer->encode(input_text, false);
    int input_len = static_cast<int>(input_tokens.size());
    Matrix encoder_mask(input_len, input_len);
    for (int i = 0; i < input_len; ++i) {
        for (int j = 0; j < input_len; ++j) {
            encoder_mask.data[i][j] = 1.0f;  // No masking for encoder
        }
    }
    cached_encoder_output = encoder->encode_with_mask(input_tokens, encoder_mask);

    // TD-186: gated world-model cross-attention wiring — see forward()'s own doc comment
    // (EncoderDecoderModel.hpp) for the full rationale; the reasoning is identical here, minus
    // the decode-back trick since input_text is already this call's own parameter. Computed
    // once per call (not per beam step) since the input doesn't change across steps — captured
    // by value (a raw pointer into world_model_encoding, which outlives beam_model_fn's own
    // usage within this call) into beam_model_fn below, and (TD-342) into the KV-cached model_fn
    // further down, whose forward_with_cache() now runs the gated paths too.
    Matrix world_model_encoding;
    const Matrix* world_model_output_ptr = nullptr;
    if (world_model && !input_text.empty()) {
        try {
            world_model_encoding = world_model->encode(input_text);
            if (world_model_encoding.rows > 0) {
                world_model_output_ptr = &world_model_encoding;
            }
        } catch (const std::exception& e) {
            Logger::warn("EncoderDecoderModel::generate_response(): world-model encode skipped "
                        "({})",
                        e.what());
        }
    }

    // TD-050 follow-up (September 14, 2026): beam search explores multiple diverging token-
    // sequence hypotheses by calling model_fn once per beam per step with that beam's own
    // sequence; a single shared DecoderKVCache has no way to correctly serve more than one
    // hypothesis at once (this is exactly why generate_response_with_strategy()'s own "beam"
    // branch below never uses DecoderKVCache, using a separate non-cached beam_model_fn
    // instead). This plain entry point had no equivalent guard, even though it's just as
    // reachable: generator's GenerationConfig is a persistent member of this class, and
    // generate_response_with_strategy()'s config-sync step mutates it — so a prior, unrelated
    // call with strategy="beam" (num_beams > 1) leaves that value set on this instance, and
    // TextGenerator::generate() silently routes to generate_beam_search() using whatever
    // model_fn is handed to it, corrupting every beam but whichever one happens to match the
    // cache's assumed prefix. Checking generator's own already-stored config here — the same
    // check gpu_generate_response() already uses for the identical constraint — closes this
    // gap without needing this method to take its own strategy parameter.
    // TD-343: one hippocampal coverage/association update per decode step (a step is one token
    // length; beam hypotheses at that step are averaged), not one per forward and layer.
    HippocampalStepCommitter committer(hippocampal_updates_enabled_ ? hippocampal_memory.get()
                                                                    : nullptr,
                                       hippocampal_repetition_decay_,
                                       hippocampal_association_decay_);
    if (generator->get_config().num_beams > 1) {
        // Get actual tokenizer vocab size to mask invalid tokens — mirrors
        // generate_response_with_strategy()'s own beam_model_fn exactly, masking included, so
        // this path and that one compute identically given the same inputs.
        int actual_vocab_size = static_cast<int>(tokenizer->get_vocab_size());
        auto beam_model_fn = [this, &committer, actual_vocab_size,
                              world_model_output_ptr](const std::vector<int>& tokens) -> Matrix {
            // Process all tokens from scratch (no caching) — mirrors
            // generate_response_with_strategy()'s own beam_model_fn.
            Matrix decoder_out = decoder->forward_with_encoder(
                tokens, cached_encoder_output, world_model_output_ptr, hippocampal_memory.get(),
                hippocampal_repetition_alpha_, hippocampal_repetition_decay_,
                hippocampal_cross_reference_alpha_, hippocampal_association_decay_);
            if (committer.active()) {
                committer.observe(tokens.size(), decoder->last_hippocampal_step_weights());
            }
            // TD-424: project only the last position; TextGenerator reads only the last row. This
            // used to project every position (~30x the LM-head work per step and per hypothesis).
            Matrix logits = lm_head->forward(last_row(decoder_out));

            // Mask out invalid token IDs beyond actual vocabulary size, same as
            // generate_response_with_strategy()'s beam_model_fn.
            if (actual_vocab_size < logits.cols) {
                for (int i = 0; i < logits.rows; ++i) {
                    for (int j = actual_vocab_size; j < logits.cols; ++j) {
                        logits.data[i][j] = -1e9f;
                    }
                }
            }

            return logits;
        };
        std::vector<int> output_tokens =
            generator->generate_beam_search(beam_model_fn, {bos_token_id});
        std::string beam_response = tokenizer->decode(output_tokens, true);
        committer.flush();
        maybe_write_hippocampal_memory(beam_response);  // TD-185
        return beam_response;
    }

    // Initialize KV cache for efficient generation
    DecoderKVCache kv_cache(decoder_layers);
    size_t processed_length = 0;

    // Create model forward function with KV caching
    // TODO: See TD-362 in TECHNICAL_DEBT.md - unlike the strategy path, this cached model_fn doesn't mask IDs beyond the tokenizer vocab
    auto model_fn = [this, &committer, &kv_cache, &processed_length,
                     world_model_output_ptr](const std::vector<int>& tokens) -> Matrix {
        size_t current_length = tokens.size();

        // Determine which tokens are new (not yet processed)
        std::vector<int> new_tokens(tokens.begin() + static_cast<std::ptrdiff_t>(processed_length),
                                    tokens.end());

        // Process only new tokens with cache
        Matrix decoder_out =
            decoder->forward_with_cache(new_tokens, kv_cache, &cached_encoder_output, true,
                                        world_model_output_ptr, hippocampal_memory.get(),
                                        hippocampal_repetition_alpha_, hippocampal_repetition_decay_,
                                        hippocampal_cross_reference_alpha_,
                                        hippocampal_association_decay_);  // TD-342
            if (committer.active()) {
                committer.observe(tokens.size(), decoder->last_hippocampal_step_weights());
            }

        // Update processed length
        processed_length = current_length;

        // Project to vocabulary (last position of output)
        Matrix logits = lm_head->forward(decoder_out);

        return logits;
    };

    // Generate using TextGenerator with greedy or configured strategy
    std::vector<int> output_tokens =
        generator->generate(model_fn, {bos_token_id}  // Start with <bos>
        );

    // Decode tokens to text (skip special tokens like <bos>, <eos>, <unk>, <pad>)
    std::string response = tokenizer->decode(output_tokens, true);
    committer.flush();
    maybe_write_hippocampal_memory(response);  // TD-185

    return response;
}

// Generate with specific strategy
std::string EncoderDecoderModel::generate_response_with_strategy(const std::string& input_text,
                                                                 int max_length,
                                                                 const std::string& strategy,
                                                                 float temperature, int top_k,
                                                                 float top_p, int num_beams) {
    // TD-156: see generate_response()'s identical lock above / model_mutex_'s doc comment in
    // EncoderDecoderModel.hpp. This is the entry point RAGInference calls directly (bypassing
    // forward()), so it needs its own lock rather than relying on forward()'s.
    std::lock_guard<std::mutex> lock(model_mutex_);

    // Ensure special token IDs are synced with tokenizer
    sync_special_tokens();

    // TD-100 (fixed): generate_greedy()/generate_sampling()/generate_top_k()/
    // generate_nucleus() each take at most ONE of their own filter values as
    // an explicit parameter (e.g. generate_top_k()'s own `k`) — everything
    // else, including max_length itself, is read straight from `generator`'s
    // *stored* config. Only the "beam" branch below ever pushed this call's
    // arguments into that config; every other strategy silently generated
    // using whichever max_length/temperature/top_k/top_p `generator` already
    // happened to hold (its constructor default, or whatever an unrelated
    // earlier call last left behind) — this caller's own arguments of the
    // same name were accepted, validated by nothing, and then ignored.
    // Concretely: "topk" and "nucleus" read config.temperature for their own
    // internal apply_temperature() step (they only take k/p as an explicit
    // parameter), so a caller's `temperature` was dropped for those two
    // strategies specifically; max_length was dropped for every strategy
    // except "beam". Syncing once, up front, for every strategy closes all
    // of these gaps without changing any strategy's own filtering logic.
    // Normalize strategy name (handle hyphens)
    std::string normalized_strategy = strategy;
    if (normalized_strategy == "top-k") {
        normalized_strategy = "topk";
    } else if (normalized_strategy == "top-p" || normalized_strategy == "nucleus") {
        normalized_strategy = "nucleus";  // Standardize to "nucleus"
    }

    // TD-261: beam search runs for "beam", or for an unrecognized strategy with num_beams > 1 (it
    // falls to generator->generate(), which itself routes to beam search when num_beams > 1 and must
    // not get the KV-cached model_fn). An explicit greedy/sampling/topk/nucleus ignores num_beams:
    // `num_beams > 1` used to send every strategy to beam search (chatbot_gui passes its Beam Width,
    // default 5; RLHFTrainer got the old header default, 4).
    const bool single_hypothesis_strategy =
        normalized_strategy == "greedy" || normalized_strategy == "sampling" ||
        normalized_strategy == "topk" || normalized_strategy == "nucleus";
    const bool use_beam =
        normalized_strategy == "beam" || (!single_hypothesis_strategy && num_beams > 1);

    {
        TextGenerator::GenerationConfig synced_config = generator->get_config();
        synced_config.max_length =
            capped_generation_length(max_length, synced_config.max_length, max_seq_length);  // TD-347
        synced_config.temperature = temperature;
        synced_config.top_k = top_k;
        synced_config.top_p = top_p;
        // The generator keeps this config between calls, and generate_response() switches to beam
        // search when the stored num_beams > 1: store a beam width only when beam search runs.
        synced_config.num_beams = use_beam ? num_beams : 1;
        generator->set_config(synced_config);
    }

    // Encode input (no special tokens for encoder input)
    // TODO: See TD-365 in TECHNICAL_DEBT.md - never truncated to max_seq_length
    std::vector<int> input_tokens = tokenizer->encode(input_text, false);
    int input_len = static_cast<int>(input_tokens.size());
    Matrix encoder_mask(input_len, input_len);
    for (int i = 0; i < input_len; ++i) {
        for (int j = 0; j < input_len; ++j) {
            encoder_mask.data[i][j] = 1.0f;
        }
    }
    cached_encoder_output = encoder->encode_with_mask(input_tokens, encoder_mask);

    // TD-186: gated world-model cross-attention wiring — see generate_response()'s identical
    // comment just above (same rationale; beam and, since TD-342, cached decoding) and forward()'s own
    // doc comment (EncoderDecoderModel.hpp) for the full picture.
    Matrix world_model_encoding;
    const Matrix* world_model_output_ptr = nullptr;
    if (world_model && !input_text.empty()) {
        try {
            world_model_encoding = world_model->encode(input_text);
            if (world_model_encoding.rows > 0) {
                world_model_output_ptr = &world_model_encoding;
            }
        } catch (const std::exception& e) {
            Logger::warn(
                "EncoderDecoderModel::generate_response_with_strategy(): world-model encode "
                "skipped ({})",
                e.what());
        }
    }

    // Generate based on strategy
    std::vector<int> output_tokens;

    // IMPORTANT: Beam search requires different handling because each beam has its own sequence
    // and KV caching doesn't work when we need to explore multiple hypotheses simultaneously, so it
    // uses the uncached beam_model_fn below (see use_beam above for when it runs).
    // TD-343: one hippocampal coverage/association update per decode step (a step is one token
    // length; beam hypotheses at that step are averaged), not one per forward and layer.
    HippocampalStepCommitter committer(hippocampal_updates_enabled_ ? hippocampal_memory.get()
                                                                    : nullptr,
                                       hippocampal_repetition_decay_,
                                       hippocampal_association_decay_);
    if (use_beam) {  // the config sync above already set num_beams and max_length

        // Get actual tokenizer vocab size to mask invalid tokens
        int actual_vocab_size = static_cast<int>(tokenizer->get_vocab_size());

        // Create model function WITHOUT KV caching for beam search
        // Each beam has independent token sequences, so we can't share a cache
        auto beam_model_fn = [this, &committer, actual_vocab_size,
                              world_model_output_ptr](const std::vector<int>& tokens) -> Matrix {
            // Process all tokens from scratch (no caching)
            Matrix decoder_out = decoder->forward_with_encoder(
                tokens, cached_encoder_output, world_model_output_ptr, hippocampal_memory.get(),
                hippocampal_repetition_alpha_, hippocampal_repetition_decay_,
                hippocampal_cross_reference_alpha_, hippocampal_association_decay_);
            if (committer.active()) {
                committer.observe(tokens.size(), decoder->last_hippocampal_step_weights());
            }

            // TD-424: project only the last position; TextGenerator reads only the last row. This
            // used to project every position (~30x the LM-head work per step and per hypothesis).
            Matrix logits = lm_head->forward(last_row(decoder_out));

            // Mask out invalid token IDs beyond actual vocabulary size
            // This prevents generation of tokens that don't exist in the tokenizer
            if (actual_vocab_size < logits.cols) {
                for (int i = 0; i < logits.rows; ++i) {
                    for (int j = actual_vocab_size; j < logits.cols; ++j) {
                        logits.data[i][j] = -1e9f;  // Set to very negative to make prob ~0
                    }
                }
            }

            return logits;
        };

        output_tokens = generator->generate_beam_search(beam_model_fn, {bos_token_id});

        // Decode tokens to text (skip special tokens like <bos>, <eos>, <unk>, <pad>)
        std::string beam_response = tokenizer->decode(output_tokens, true);
        committer.flush();
        maybe_write_hippocampal_memory(beam_response);  // TD-185
        return beam_response;
    }

    // For non-beam strategies, use KV caching for efficiency
    // Initialize KV cache for efficient generation
    DecoderKVCache kv_cache(decoder_layers);
    size_t processed_length = 0;

    // Get actual tokenizer vocab size to mask invalid tokens
    int actual_vocab_size = static_cast<int>(tokenizer->get_vocab_size());

    // Create model forward function with KV caching
    auto model_fn = [this, &committer, &kv_cache, &processed_length, actual_vocab_size,
                     world_model_output_ptr](const std::vector<int>& tokens) -> Matrix {
        size_t current_length = tokens.size();

        // Determine which tokens are new (not yet processed)
        std::vector<int> new_tokens(tokens.begin() + static_cast<std::ptrdiff_t>(processed_length),
                                    tokens.end());

        // Process only new tokens with cache
        Matrix decoder_out =
            decoder->forward_with_cache(new_tokens, kv_cache, &cached_encoder_output, true,
                                        world_model_output_ptr, hippocampal_memory.get(),
                                        hippocampal_repetition_alpha_, hippocampal_repetition_decay_,
                                        hippocampal_cross_reference_alpha_,
                                        hippocampal_association_decay_);  // TD-342
            if (committer.active()) {
                committer.observe(tokens.size(), decoder->last_hippocampal_step_weights());
            }

        // Update processed length
        processed_length = current_length;

        // Project to vocabulary (last position of output)
        Matrix logits = lm_head->forward(decoder_out);

        // Mask out invalid token IDs beyond actual vocabulary size
        // This prevents generation of tokens that don't exist in the tokenizer
        if (actual_vocab_size < logits.cols) {
            for (int i = 0; i < logits.rows; ++i) {
                for (int j = actual_vocab_size; j < logits.cols; ++j) {
                    logits.data[i][j] = -1e9f;  // Set to very negative to make prob ~0
                }
            }
        }

        return logits;
    };

    // Generate based on strategy

    if (normalized_strategy == "greedy") {
        // TD-050 (root-caused September 14, 2026): this used to bypass the KV cache here via a
        // dedicated non-cached greedy_model_fn ("WORKAROUND: Use non-cached path for greedy due
        // to KV cache bug"), on the assumption that DecoderKVCache had a self-attention/
        // cross-attention indexing bug producing incorrect results specifically for greedy
        // decoding. A real multi-step incremental-decode-vs-full-recompute numerical comparison
        // (tests/inference_optimization_test.cpp,
        // InferenceOptimizationIntegrationTest.IncrementalCacheMatchesFullRecomputePerStep) found
        // no such bug: across 40 decode steps the two paths agree to ~1e-6, with no growth or
        // accumulation trend as the sequence lengthens — ordinary float32 rounding noise, not an
        // algorithmic divergence. Greedy now uses the same cached `model_fn` every other
        // non-beam strategy already did, closing the O(n) vs O(n^2)-per-generation gap this
        // workaround left on the table for greedy specifically.
        output_tokens = generator->generate_greedy(model_fn, {bos_token_id});
    } else if (normalized_strategy == "sampling") {
        output_tokens = generator->generate_sampling(model_fn, {bos_token_id}, temperature);
    } else if (normalized_strategy == "topk") {
        output_tokens = generator->generate_top_k(model_fn, {bos_token_id}, top_k);
    } else if (normalized_strategy == "nucleus") {
        output_tokens = generator->generate_nucleus(model_fn, {bos_token_id}, top_p);
    } else {
        // Default to main generate method (uses config)
        output_tokens = generator->generate(model_fn, {bos_token_id});
    }

    // Decode tokens to text (skip special tokens like <bos>, <eos>, <unk>, <pad>)
    std::string response = tokenizer->decode(output_tokens, true);
    committer.flush();
    maybe_write_hippocampal_memory(response);  // TD-185
    return response;
}

// Training step
float EncoderDecoderModel::train_step(const std::string& input_text,
                                      const std::string& target_text) {
    std::vector<int> input_tokens =
        tokenizer->encode(input_text, false);  // Encoder: no special tokens
    // Decoder targets: [w1 ... wn, EOS], without the tokenizer's leading BOS (TD-361)
    std::vector<int> target_tokens =
        strip_leading_bos(tokenizer->encode(target_text, true), bos_token_id);

    return train_step_tokenized(input_tokens, target_tokens);
}

// Training step on tokenized sequences
float EncoderDecoderModel::train_step_tokenized(const std::vector<int>& input_tokens,
                                                const std::vector<int>& target_tokens) {
    if (!requires_grad) {
        throw std::runtime_error("Model is not in training mode");
    }

    // Zero gradients
    zero_grad();

    // Forward pass
    Matrix logits = forward(input_tokens, target_tokens);

    // Compute loss
    float loss = compute_loss(logits, target_tokens);

    // Backward pass
    Matrix grad_loss = compute_loss_gradient(logits, target_tokens);
    backward(grad_loss);

    // Update weights
    update_weights();

    return loss;
}

// Evaluate (no gradients)
float EncoderDecoderModel::evaluate(const std::string& input_text, const std::string& target_text) {
    // encode() has no length cap, unlike the truncate() step ChatbotTrainer
    // applies when first tokenizing the dataset — prefer evaluate_tokenized()
    // with already-truncated ids whenever they're available.
    std::vector<int> input_tokens =
        tokenizer->encode(input_text, false);  // Encoder: no special tokens
    // Decoder targets: [w1 ... wn, EOS], without the tokenizer's leading BOS (TD-361)
    std::vector<int> target_tokens =
        strip_leading_bos(tokenizer->encode(target_text, true), bos_token_id);

    return evaluate_tokenized(input_tokens, target_tokens);
}

float EncoderDecoderModel::evaluate_tokenized(const std::vector<int>& input_tokens,
                                              const std::vector<int>& target_tokens) {
    bool prev_mode = requires_grad;
    set_training(false);

    // Forward pass only
    Matrix logits = forward(input_tokens, target_tokens);
    float loss = compute_loss(logits, target_tokens);

    set_training(prev_mode);
    return loss;
}

// Compute perplexity
float EncoderDecoderModel::compute_perplexity(const std::vector<std::string>& input_texts,
                                              const std::vector<std::string>& target_texts) {
    if (input_texts.size() != target_texts.size()) {
        throw std::invalid_argument("Input and target sizes must match");
    }

    float total_loss = 0.0f;
    int num_samples = static_cast<int>(input_texts.size());

    bool prev_mode = requires_grad;
    set_training(false);

    for (size_t i = 0; i < input_texts.size(); ++i) {
        total_loss += evaluate(input_texts[i], target_texts[i]);
    }

    set_training(prev_mode);

    float avg_loss = total_loss / static_cast<float>(num_samples);
    return std::exp(avg_loss);  // Perplexity = exp(cross_entropy)
}

// Set training mode
void EncoderDecoderModel::set_training(bool mode) {
    requires_grad = mode;
    encoder->set_requires_grad(mode);
    decoder->set_training(mode);
}

// Set learning rate
// TODO: See TD-360 in TECHNICAL_DEBT.md - never reaches the encoder (or the LM head)
void EncoderDecoderModel::set_learning_rate(float lr) {
    learning_rate = lr;
    // encoder->set_learning_rate(lr);  // stale: LLMEncoder::set_learning_rate() exists (TD-093 fix); only
    // update_weights() is missing — see TD-360
    decoder->set_learning_rate(lr);
}

// Update weights
// TODO: See TD-360 in TECHNICAL_DEBT.md - never updates the encoder; TD-346 skips the final decoder norm
void EncoderDecoderModel::update_weights() {
    // encoder->update_weights(learning_rate);  // LLMEncoder doesn't have this method
    decoder->update_weights(learning_rate);
    lm_head->update_weights();  // LanguageModelHead doesn't take lr parameter
}

// Zero gradients
void EncoderDecoderModel::zero_grad() {
    encoder->zero_grad();
    decoder->zero_grad();
    lm_head->zero_grad();
}

// Register parameters with external optimizer
void EncoderDecoderModel::register_parameters(Optimizer& optimizer) {
    // Register encoder parameters
    encoder->register_parameters_with_optimizer(optimizer);

    // Register decoder parameters
    decoder->register_parameters_with_optimizer(optimizer);

    // Register language model head parameters
    lm_head->set_optimizer(&optimizer);
}

// TD-038: LoRA -- attach adapters to every attention layer's Q/K/V/O projections, across
// both the encoder (self-attention only) and decoder (self- and cross-attention).
void EncoderDecoderModel::enable_lora(const LoRAConfig& config) {
    for (int i = 0; i < encoder->get_num_layers(); ++i) {
        encoder->get_encoder_block(i)->get_self_attention()->enable_lora(config);
    }
    for (int i = 0; i < decoder->get_num_layers(); ++i) {
        DecoderBlock* block = decoder->get_decoder_block(i);
        block->get_self_attention()->enable_lora(config);
        block->get_cross_attention()->enable_lora(config);
    }
}

bool EncoderDecoderModel::has_lora() {
    for (int i = 0; i < encoder->get_num_layers(); ++i) {
        if (encoder->get_encoder_block(i)->get_self_attention()->has_lora()) {
            return true;
        }
    }
    for (int i = 0; i < decoder->get_num_layers(); ++i) {
        DecoderBlock* block = decoder->get_decoder_block(i);
        if (block->get_self_attention()->has_lora() || block->get_cross_attention()->has_lora()) {
            return true;
        }
    }
    return false;
}

void EncoderDecoderModel::register_lora_parameters(Optimizer& optimizer) {
    for (int i = 0; i < encoder->get_num_layers(); ++i) {
        encoder->get_encoder_block(i)->get_self_attention()->register_lora_parameters(optimizer);
    }
    for (int i = 0; i < decoder->get_num_layers(); ++i) {
        DecoderBlock* block = decoder->get_decoder_block(i);
        block->get_self_attention()->register_lora_parameters(optimizer);
        block->get_cross_attention()->register_lora_parameters(optimizer);
    }
}

void EncoderDecoderModel::merge_lora() {
    for (int i = 0; i < encoder->get_num_layers(); ++i) {
        encoder->get_encoder_block(i)->get_self_attention()->merge_lora();
    }
    for (int i = 0; i < decoder->get_num_layers(); ++i) {
        DecoderBlock* block = decoder->get_decoder_block(i);
        block->get_self_attention()->merge_lora();
        block->get_cross_attention()->merge_lora();
    }
}

// Backward pass without weight update (for use with external optimizer)
void EncoderDecoderModel::backward_pass(const Matrix& grad_output) {
    // Standard backward pass
    backward(grad_output);
    // Note: No weight update - handled by external optimizer
}

// Set tokenizer
void EncoderDecoderModel::set_tokenizer(BPETokenizer* tokenizer_ptr) {
    tokenizer.reset(tokenizer_ptr);
    sync_special_tokens();
}

// Sync special token IDs from tokenizer
void EncoderDecoderModel::sync_special_tokens() {
    if (!tokenizer) {
        return;
    }

    TextGenerator::GenerationConfig config = generator->get_config();
    config.bos_token_id = tokenizer->get_bos_token_id();
    config.eos_token_id = tokenizer->get_eos_token_id();
    config.pad_token_id = tokenizer->get_pad_token_id();
    set_generation_config(config);
}

// Set generation config
void EncoderDecoderModel::set_generation_config(const TextGenerator::GenerationConfig& config) {
    generator->set_config(config);
    bos_token_id = config.bos_token_id;
    eos_token_id = config.eos_token_id;
    pad_token_id = config.pad_token_id;
}

// Get generation config
TextGenerator::GenerationConfig EncoderDecoderModel::get_generation_config() const {
    return generator->get_config();
}

// Save model
void EncoderDecoderModel::save_model(const std::string& filepath) const {
    // Save architecture config
    std::ofstream config_file(filepath + ".config", std::ios::binary);
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open config file for writing: " + filepath + ".config");
    }
    config_file.write(reinterpret_cast<const char*>(&kConfigMagic), sizeof(int32_t));
    config_file.write(reinterpret_cast<const char*>(&kConfigFormatVersion), sizeof(int32_t));
    config_file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&encoder_layers), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&decoder_layers), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&d_ff), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&max_seq_length), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&bos_token_id), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&eos_token_id), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&pad_token_id), sizeof(int));
    config_file.close();

    // Save tokenizer vocabulary
    // Throws VocabularyFileError if the vocab can't be written (TD-221), so a checkpoint is never
    // reported as saved without its .vocab.
    tokenizer->save_vocab(filepath + ".vocab");

    // Save encoder weights
    encoder->save_weights(filepath + ".encoder");

    // Save decoder weights
    decoder->save_weights(filepath + ".decoder");

    // Save language model head
    lm_head->save_weights(filepath + ".lm_head");
}

// Load model
// TODO: See TD-363 in TECHNICAL_DEBT.md - config reads unchecked; special tokens not synced into the generator config
void EncoderDecoderModel::load_model(const std::string& filepath) {
    // Load architecture config
    std::ifstream config_file(filepath + ".config", std::ios::binary);
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to load model config");
    }

    int32_t magic = 0, format_version = 0;
    config_file.read(reinterpret_cast<char*>(&magic), sizeof(int32_t));
    config_file.read(reinterpret_cast<char*>(&format_version), sizeof(int32_t));
    if (magic != kConfigMagic) {
        throw std::runtime_error(
            "Checkpoint format marker missing/invalid — this looks like a pre-Pre-LN (Post-LN) "
            "checkpoint saved before the architecture change and is not compatible with the "
            "current Pre-LN EncoderBlock/DecoderBlock math (same weight shapes, different "
            "meaning — loading it would silently produce garbage). Retrain from scratch.");
    }
    if (format_version < kConfigFormatVersion) {
        throw std::runtime_error("Unsupported older checkpoint format_version " +
                                 std::to_string(format_version) + " (current: " +
                                 std::to_string(kConfigFormatVersion) + "). Retrain from scratch.");
    }

    int loaded_vocab_size = 0, loaded_d_model = 0, loaded_encoder_layers = 0,
        loaded_decoder_layers = 0;
    int loaded_num_heads = 0, loaded_d_ff = 0, loaded_max_seq_length = 0;
    int loaded_bos = 0, loaded_eos = 0, loaded_pad = 0;

    config_file.read(reinterpret_cast<char*>(&loaded_vocab_size), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_encoder_layers), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_decoder_layers), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_max_seq_length), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_bos), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_eos), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_pad), sizeof(int));
    config_file.close();

    // Verify architecture match
    if (loaded_vocab_size != vocab_size || loaded_d_model != d_model ||
        loaded_encoder_layers != encoder_layers || loaded_decoder_layers != decoder_layers ||
        loaded_num_heads != num_heads || loaded_d_ff != d_ff ||
        loaded_max_seq_length != max_seq_length) {
        throw std::runtime_error(
            "Model architecture mismatch: saved (vocab=" + std::to_string(loaded_vocab_size) +
            ", d_model=" + std::to_string(loaded_d_model) +
            ", enc_layers=" + std::to_string(loaded_encoder_layers) +
            ", dec_layers=" + std::to_string(loaded_decoder_layers) + ", num_heads=" +
            std::to_string(loaded_num_heads) + ", d_ff=" + std::to_string(loaded_d_ff) +
            ", max_seq=" + std::to_string(loaded_max_seq_length) + ") vs current (vocab=" +
            std::to_string(vocab_size) + ", d_model=" + std::to_string(d_model) + ", enc_layers=" +
            std::to_string(encoder_layers) + ", dec_layers=" + std::to_string(decoder_layers) +
            ", num_heads=" + std::to_string(num_heads) + ", d_ff=" + std::to_string(d_ff) +
            ", max_seq=" + std::to_string(max_seq_length) + ")");
    }

    // Load tokenizer vocabulary
    tokenizer->load_vocab(filepath + ".vocab");

    // Load component weights
    encoder->load_weights(filepath + ".encoder");
    decoder->load_weights(filepath + ".decoder");

    // Load language model head
    lm_head->load_weights(filepath + ".lm_head");

    // Update special tokens
    bos_token_id = loaded_bos;
    eos_token_id = loaded_eos;
    pad_token_id = loaded_pad;
}

// Clone (TD-023): deep-copy weights via save/load; no optimizer state is copied
// TODO: See TD-363 in TECHNICAL_DEBT.md - the copy lacks world model, hippocampal memory, generation config and LoRA adapters
std::unique_ptr<EncoderDecoderModel> EncoderDecoderModel::clone() const {
    namespace fs = std::filesystem;

    // Build a unique temp path from the object address and a steady-clock timestamp
    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto addr = std::hash<const void*>{}(static_cast<const void*>(this));
    const std::string tmp = fs::temp_directory_path().string() + "/adai_clone_" +
                            std::to_string(addr) + "_" + std::to_string(ts);

    // Persist weights to temp files
    save_model(tmp);

    // Remove every temp file on exit (success or exception). TD-544: save_model() writes the five
    // top-level files plus a component file per embedding, block and norm, all named
    // <tmp>.<ext> or <tmp>_<name>; only the five used to be removed, so each clone (one per
    // epoch for the async quality metrics) left a copy of the weights in the temp directory.
    auto remove_temps = [&tmp] {
        const fs::path dir = fs::path(tmp).parent_path();
        const std::string stem = fs::path(tmp).filename().string();
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.size() > stem.size() && name.compare(0, stem.size(), stem) == 0 &&
                (name[stem.size()] == '.' || name[stem.size()] == '_')) {
                std::error_code rm_ec;
                fs::remove(it->path(), rm_ec);  // best-effort; ignore errors
            }
        }
    };

    std::unique_ptr<EncoderDecoderModel> copy;
    try {
        copy = std::make_unique<EncoderDecoderModel>(
            vocab_size, d_model, encoder_layers, decoder_layers, num_heads, d_ff, max_seq_length);
        copy->load_model(tmp);  // also sets bos/eos/pad_token_id on copy
    } catch (...) {
        remove_temps();
        throw;
    }
    remove_temps();
    return copy;
}

std::vector<int> EncoderDecoderModel::strip_leading_bos(std::vector<int> ids, int bos_id) {
    if (!ids.empty() && ids.front() == bos_id) {
        ids.erase(ids.begin());
    }
    return ids;
}

std::vector<int> EncoderDecoderModel::make_decoder_input(
    const std::vector<int>& target_tokens) const {
    // TD-060: `i + 1 < size()` rather than `i < size() - 1`, which underflows for an empty target.
    // Targets carry no leading BOS (TD-361, strip_leading_bos()), so this BOS is the only one.
    std::vector<int> decoder_input;
    decoder_input.reserve(target_tokens.size());
    decoder_input.push_back(bos_token_id);
    for (size_t i = 0; i + 1 < target_tokens.size(); ++i) {
        decoder_input.push_back(target_tokens[i]);
    }
    return decoder_input;
}

// Forward pass
Matrix EncoderDecoderModel::forward(const std::vector<int>& input_tokens,
                                    const std::vector<int>& target_tokens) {
    // TD-156: see generate_response()'s identical lock above / model_mutex_'s doc comment in
    // EncoderDecoderModel.hpp. This is the entry point ChatbotAPI's plain and
    // speculative-decoding paths call directly, on the HTTP handler thread.
    std::lock_guard<std::mutex> lock(model_mutex_);

    // Cache inputs
    cached_input_tokens = input_tokens;
    cached_target_tokens = target_tokens;

    // Encode input
    int input_len = static_cast<int>(input_tokens.size());
    Matrix encoder_mask(input_len, input_len);
    for (int i = 0; i < input_len; ++i) {
        for (int j = 0; j < input_len; ++j) {
            encoder_mask.data[i][j] = 1.0f;
        }
    }
    cached_encoder_output = encoder->encode_with_mask(input_tokens, encoder_mask);

    // Prepare decoder input (teacher forcing: use target tokens)
    // In training, we use ground truth tokens as decoder input
    // TD-060 (fixed): `i < target_tokens.size() - 1` underflows to SIZE_MAX for
    // an empty target_tokens (size_t is unsigned), turning this into an
    // out-of-bounds read that crashes immediately — confirmed via ASan (SEGV).
    // `forward()` is a public, documented "custom training loops" API, so an
    // empty target_tokens is a plausible caller input, not just theoretical.
    std::vector<int> decoder_input = make_decoder_input(target_tokens);

    // TD-186: gated world-model / hippocampal cross-attention wiring — see this method's own
    // doc comment (EncoderDecoderModel.hpp) for the full rationale. Both remain no-ops when not
    // attached (world_model_output_ptr stays nullptr, hippocampal_memory.get() is nullptr),
    // reproducing this method's exact pre-TD-186 behavior byte-for-byte.
    Matrix world_model_encoding;
    const Matrix* world_model_output_ptr = nullptr;
    if (world_model) {
        // TODO: See TD-363 in TECHNICAL_DEBT.md - decoding ids back to text is lossy (TD-224); the world model sees different text
        std::string decoded_input = tokenizer->decode(input_tokens, /*skip_special_tokens=*/true);
        if (!decoded_input.empty()) {
            try {
                world_model_encoding = world_model->encode(decoded_input);
                if (world_model_encoding.rows > 0) {
                    world_model_output_ptr = &world_model_encoding;
                }
            } catch (const std::exception& e) {
                Logger::warn("EncoderDecoderModel::forward(): world-model encode skipped ({})",
                            e.what());
            }
        }
    }

    // Decode with cross-attention
    cached_decoder_output = decoder->forward_with_encoder(
        decoder_input, cached_encoder_output, world_model_output_ptr, hippocampal_memory.get(),
        hippocampal_repetition_alpha_, hippocampal_repetition_decay_,
                hippocampal_cross_reference_alpha_, hippocampal_association_decay_);

    // Project to vocabulary
    // The full projection is what training needs. TD-424: ChatbotAPI's CPU generation model_fn and
    // /chat/batch also call forward() per step and use only the last row; that goes with TD-506,
    // which replaces those callbacks (their last row is also misaligned).
    Matrix logits = lm_head->forward(cached_decoder_output);

    return logits;
}

// Backward pass
void EncoderDecoderModel::backward(const Matrix& grad_output) {
    // TD-156: see generate_response()'s identical lock above / model_mutex_'s doc comment in
    // EncoderDecoderModel.hpp. backward() itself reads no cached_* member directly, but
    // lm_head/decoder/encoder's own internal activation caches were written by this same
    // instance's most recent forward() — this lock keeps that pairing atomic against a
    // concurrent forward() call on another thread, not just the four members forward() writes.
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!requires_grad) {
        return;
    }

    // Backward through LM head
    Matrix grad_decoder = lm_head->backward(grad_output);

    // Backward through decoder, capturing the gradient w.r.t. encoder output
    // (summed across every decoder block's cross-attention)
    Matrix grad_encoder_output;
    decoder->backward(grad_decoder, grad_encoder_output);

    // Backward through encoder
    encoder->backward(grad_encoder_output);
}

#ifdef ADAI_ENABLE_GPU
void EncoderDecoderModel::gpu_init_training() {
    encoder->gpu_upload_weights();
    decoder->gpu_upload_weights();
    lm_head->gpu_upload_weights();
    gpu_initialized_ = true;
}

void EncoderDecoderModel::gpu_zero_grads() {
    encoder->gpu_zero_grads();
    decoder->gpu_zero_grads();
    lm_head->gpu_zero_grads();
}

// TD-342: the frozen world model's encoding of `text`, computed on the device; nullptr when no
// world model is attached, the text is empty, or encoding fails (as forward()'s CPU encode).
std::unique_ptr<adai::gpu::GPUMatrix> EncoderDecoderModel::gpu_world_model_encoding(
    const std::string& text) {
    if (!world_model || text.empty()) {
        return nullptr;
    }
    try {
        auto encoding = std::make_unique<adai::gpu::GPUMatrix>(world_model->gpu_encode(text));
        if (encoding->rows > 0) {
            return encoding;
        }
    } catch (const std::exception& e) {
        Logger::warn("EncoderDecoderModel: GPU world-model encode skipped ({})", e.what());
    }
    return nullptr;
}

float EncoderDecoderModel::gpu_forward(const std::vector<int>& input_tokens,
                                       const std::vector<int>& target_tokens) {
    // The decoder input has one row per target (TD-358), so an empty target has nothing to score
    // and the logits/label row counts the GPU loss and gradient rely on would disagree.
    if (target_tokens.empty()) {
        throw std::invalid_argument("EncoderDecoderModel::gpu_forward: empty target sequence");
    }
    cached_target_tokens = target_tokens;
    const int tgt_len = static_cast<int>(target_tokens.size());

    // Upload integer target IDs to device (resize buffer if needed)
    if (!gpu_targets_dev_ || gpu_target_len_ < tgt_len) {
        gpu_targets_dev_ = std::make_unique<adai::gpu::GPUMemory<int>>(tgt_len);
        gpu_target_len_ = tgt_len;
    }
    gpu_targets_dev_->copy_from_host(target_tokens.data(), static_cast<size_t>(tgt_len));

    // Encoder
    gpu_encoder_out_ = std::make_unique<adai::gpu::GPUMatrix>(encoder->gpu_encode(input_tokens));

    // Decoder (caches its block inputs/outputs for backward internally). TD-358: teacher forcing
    // feeds `[BOS] + target[:-1]` (exactly what forward() feeds) while `target_tokens` stay the
    // labels below, so logits row t predicts target[t] from the targets before it. Feeding the
    // targets themselves let position t see the token it was scored against, so GPU training
    // learned to copy its input instead of predicting the next token.
    const std::vector<int> decoder_input = make_decoder_input(target_tokens);
    // TD-342: the gated paths, as forward(): the world model sees the decoded input text.
    std::unique_ptr<adai::gpu::GPUMatrix> gpu_wm;
    if (world_model) {
        gpu_wm = gpu_world_model_encoding(tokenizer->decode(input_tokens, /*skip_special_tokens=*/true));
    }
    adai::gpu::GPUMatrix dec_out = decoder->gpu_decode(
        decoder_input, *gpu_encoder_out_, gpu_wm.get(), hippocampal_memory.get(),
        hippocampal_repetition_alpha_, hippocampal_repetition_decay_,
        hippocampal_cross_reference_alpha_, hippocampal_association_decay_);

    // LM head (caches dec_out internally)
    gpu_logits_ = std::make_unique<adai::gpu::GPUMatrix>(lm_head->gpu_forward(dec_out));

    // Cross-entropy loss entirely on GPU — only a scalar crosses PCIe
    return adai::gpu::matrix_cross_entropy_loss_gpu(gpu_logits_->device_ptr(),
                                                    gpu_targets_dev_->get(), tgt_len, vocab_size);
}

// TD-358: labels are `target_tokens` (cached by gpu_forward()), aligned with the shifted decoder
// input.
void EncoderDecoderModel::gpu_backward(float scale) {
    const int tgt_len = static_cast<int>(cached_target_tokens.size());

    // Cross-entropy gradient w.r.t. logits
    adai::gpu::GPUMatrix dlogits(tgt_len, vocab_size);
    adai::gpu::matrix_cross_entropy_grad_gpu(gpu_logits_->device_ptr(), gpu_targets_dev_->get(),
                                             dlogits.device_ptr(), tgt_len, vocab_size);

    // Apply gradient accumulation scale (no allocation when scale == 1)
    if (scale != 1.0f)
        dlogits = dlogits.scale(scale);

    // LM head backward — accumulates dW, db; returns d_dec_out
    adai::gpu::GPUMatrix d_dec = lm_head->gpu_backward(dlogits);

    // Decoder backward — accumulates gradients into each block, and returns the
    // gradient w.r.t. encoder_output summed across every decoder block's
    // cross-attention.
    auto [grad_decoder_embed_input, grad_encoder_output] = decoder->gpu_backward(d_dec);
    // decoder->gpu_backward() already applies this gradient to the decoder's
    // token embedding internally (mirrors LLMEncoder::gpu_backward()); the
    // caller has no further use for it.
    (void)grad_decoder_embed_input;

    // Encoder backward
    encoder->gpu_backward(grad_encoder_output);
}

float EncoderDecoderModel::gpu_evaluate(const std::string& input_text,
                                        const std::string& target_text) {
    // encode() has no length cap, unlike the truncate() step ChatbotTrainer
    // applies when first tokenizing the dataset — prefer
    // gpu_evaluate_tokenized() with already-truncated ids whenever available.
    // An untruncated sequence here scales the encoder/decoder's on-device
    // temporaries (attention scores are O(seq^2)) well past what the model
    // ever sees during training, which is what was blowing the GPU memory
    // budget during validation.
    std::vector<int> input_tokens =
        tokenizer->encode(input_text, false);  // Encoder: no special tokens
    // Decoder targets: [w1 ... wn, EOS], without the tokenizer's leading BOS (TD-361)
    std::vector<int> target_tokens =
        strip_leading_bos(tokenizer->encode(target_text, true), bos_token_id);

    return gpu_evaluate_tokenized(input_tokens, target_tokens);
}

float EncoderDecoderModel::gpu_evaluate_tokenized(const std::vector<int>& input_tokens,
                                                  const std::vector<int>& target_tokens) {
    bool prev_mode = requires_grad;
    set_training(false);

    // Forward pass only — gpu_forward() already computes cross-entropy loss
    // on-device, so there is nothing further to compute here. Deliberately
    // never call gpu_backward(): validation must not accumulate gradients.
    float loss = gpu_forward(input_tokens, target_tokens);

    set_training(prev_mode);
    return loss;
}

std::string EncoderDecoderModel::gpu_generate_response(const std::string& input_text,
                                                       int max_length) {
    // TD-033: see model_mutex_'s doc comment in EncoderDecoderModel.hpp — encoder/decoder/
    // lm_head's own gpu_forward() calls below write through to their persistent per-instance
    // GPUState, which two concurrent callers on the same model instance would race on.
    std::lock_guard<std::mutex> lock(model_mutex_);

    // TD-359: this call's max_length, as generate_response() (it used to be ignored).
    const ScopedMaxLength scoped_max_length(*generator, max_length, max_seq_length);

    // TODO: See TD-365 in TECHNICAL_DEBT.md - never truncated to max_seq_length
    std::vector<int> input_tokens = tokenizer->encode(input_text, false);

    // Encode once on GPU; read-only input to every decode step below.
    adai::gpu::GPUMatrix gpu_encoder_out = encoder->gpu_encode(input_tokens);
    // TD-342: the frozen world model's encoding, on the device, for the gated paths.
    std::unique_ptr<adai::gpu::GPUMatrix> gpu_wm = gpu_world_model_encoding(input_text);
    // TD-343: one hippocampal coverage/association update per decode step (a step is one token
    // length; beam hypotheses at that step are averaged), not one per forward and layer.
    HippocampalStepCommitter committer(hippocampal_updates_enabled_ ? hippocampal_memory.get()
                                                                    : nullptr,
                                       hippocampal_repetition_decay_,
                                       hippocampal_association_decay_);

    // TD-050 (September 14, 2026): beam search explores multiple diverging hypotheses through
    // one model_fn — a single shared KV cache cannot correctly serve that (each beam's calls
    // would corrupt the cache the others depend on), the same constraint the CPU path already
    // respects (generate_response_with_strategy()'s own "beam" branch never uses DecoderKVCache
    // either). This entry point has no explicit strategy name — generate() itself decides
    // beam-vs-not from generator's own already-stored config — so check that directly and use
    // the same full-recompute-every-step gpu_decode() this method used unconditionally before
    // the cache existed, for beam search specifically.
    if (generator->get_config().num_beams > 1) {
        auto beam_model_fn = [this, &committer, &gpu_encoder_out, &gpu_wm](const std::vector<int>& tokens) -> Matrix {
            adai::gpu::GPUMatrix dec_out = decoder->gpu_decode(tokens, gpu_encoder_out, gpu_wm.get(),
                                                                hippocampal_memory.get(),
                                                                hippocampal_repetition_alpha_,
                                                                hippocampal_repetition_decay_,
                                                                hippocampal_cross_reference_alpha_,
                                                                hippocampal_association_decay_);
            if (committer.active()) {
                committer.observe(tokens.size(), decoder->last_hippocampal_step_weights());
            }
            adai::gpu::GPUMatrix logits = lm_head->gpu_forward(dec_out);
            const int tgt = static_cast<int>(tokens.size());
            std::vector<float> last_row(vocab_size);
            adai::gpu::matrix_download_gpu(logits.device_ptr() + (tgt - 1) * vocab_size,
                                           last_row.data(), vocab_size);
            Matrix last_logits(1, vocab_size);
            for (int v = 0; v < vocab_size; ++v) {
                last_logits.data[0][v] = last_row[v];
            }
            return last_logits;
        };
        std::vector<int> output_tokens = generator->generate(beam_model_fn, {bos_token_id});
        std::string response = tokenizer->decode(output_tokens, true);
        committer.flush();
        maybe_write_hippocampal_memory(response);  // TD-342, as the CPU paths (TD-185)
        return response;
    }

    // Real GPU-resident incremental cache (TD-050) — replaces the previous full-recompute-
    // every-step gpu_decode() call. Mirrors generate_response()'s own CPU cached model_fn:
    // tracks processed_length, slices only the new tokens each call, threads one
    // GPUDecoderKVCache across the whole generation.
    // TODO: See TD-387 in TECHNICAL_DEBT.md - input longer than max_seq_length throws from the cross-attention cache
    adai::gpu::GPUDecoderKVCache gpu_kv_cache(decoder_layers, max_seq_length, d_model);
    size_t processed_length = 0;
    auto model_fn = [this, &committer, &gpu_encoder_out, &gpu_kv_cache, &processed_length,
                     &gpu_wm](const std::vector<int>& tokens) -> Matrix {
        std::vector<int> new_tokens(tokens.begin() + static_cast<std::ptrdiff_t>(processed_length),
                                    tokens.end());
        adai::gpu::GPUMatrix dec_out =
            decoder->gpu_decode_step(new_tokens, gpu_kv_cache, gpu_encoder_out, gpu_wm.get(),
                                     hippocampal_memory.get(),
                                     hippocampal_repetition_alpha_,
                                     hippocampal_repetition_decay_,
                                     hippocampal_cross_reference_alpha_,
                                     hippocampal_association_decay_);
            if (committer.active()) {
                committer.observe(tokens.size(), decoder->last_hippocampal_step_weights());
            }
        processed_length = tokens.size();
        adai::gpu::GPUMatrix logits = lm_head->gpu_forward(dec_out);

        // generate() only ever reads the last row of what model_fn returns, so download just
        // that row — dec_out/logits now has only new_tokens.size() rows (not tokens.size()),
        // unlike the old full-recompute version, since this is the incremental output.
        const int tgt = static_cast<int>(new_tokens.size());
        std::vector<float> last_row(vocab_size);
        adai::gpu::matrix_download_gpu(logits.device_ptr() + (tgt - 1) * vocab_size,
                                       last_row.data(), vocab_size);

        Matrix last_logits(1, vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            last_logits.data[0][v] = last_row[v];
        }
        return last_logits;
    };

    std::vector<int> output_tokens = generator->generate(model_fn, {bos_token_id});
    std::string response = tokenizer->decode(output_tokens, true);
    committer.flush();
    maybe_write_hippocampal_memory(response);  // TD-342, as the CPU paths (TD-185)
    return response;
}

// TODO: See TD-362 in TECHNICAL_DEBT.md - no vocab masking, no world model/memory write, different unknown-strategy default and num_beams handling than the CPU path
std::string EncoderDecoderModel::gpu_generate_response_with_strategy(
    const std::string& input_text, int max_length, const std::string& strategy,
    float temperature, int top_k, float top_p, int num_beams) {
    // TD-033: GPU-resident analog of generate_response_with_strategy() — see this class's
    // header for why no separate KV-cache-vs-no-cache branch per strategy is needed here, unlike
    // that CPU version. Locked for the same reason generate_response_with_strategy() locks
    // (model_mutex_'s doc comment): encoder/decoder/lm_head's own gpu_forward() calls below
    // write through to their persistent per-instance GPUState, which two concurrent callers
    // (this is ChatbotAPI's real serving path once GPU-resident decode is wired in) would race
    // on.
    std::lock_guard<std::mutex> lock(model_mutex_);
    sync_special_tokens();

    // Sync generator's config from this call's own arguments, the same way
    // generate_response_with_strategy() does (see that function's TD-100 comment) — every
    // strategy's filters must see the caller's actual values, not whatever generator's config
    // last happened to hold.
    {
        TextGenerator::GenerationConfig synced_config = generator->get_config();
        synced_config.max_length =
            capped_generation_length(max_length, synced_config.max_length, max_seq_length);  // TD-347
        synced_config.temperature = temperature;
        synced_config.top_k = top_k;
        synced_config.top_p = top_p;
        synced_config.num_beams = num_beams;
        generator->set_config(synced_config);
    }

    // Normalize strategy name — accepts both ChatbotAPI's spellings ("temperature", "top_k")
    // and generate_response_with_strategy()'s own ("sampling", "topk", hyphenated forms).
    std::string normalized_strategy = strategy;
    if (normalized_strategy == "temperature") {
        normalized_strategy = "sampling";
    } else if (normalized_strategy == "top_k" || normalized_strategy == "top-k") {
        normalized_strategy = "topk";
    } else if (normalized_strategy == "top_p" || normalized_strategy == "top-p") {
        normalized_strategy = "nucleus";
    }

    // TODO: See TD-365 in TECHNICAL_DEBT.md - never truncated to max_seq_length
    std::vector<int> input_tokens = tokenizer->encode(input_text, false);

    // Encode once on GPU; read-only input to every decode step below.
    adai::gpu::GPUMatrix gpu_encoder_out = encoder->gpu_encode(input_tokens);
    // TD-342: the frozen world model's encoding, on the device, for the gated paths.
    std::unique_ptr<adai::gpu::GPUMatrix> gpu_wm = gpu_world_model_encoding(input_text);
    // TD-343: one hippocampal coverage/association update per decode step (a step is one token
    // length; beam hypotheses at that step are averaged), not one per forward and layer.
    HippocampalStepCommitter committer(hippocampal_updates_enabled_ ? hippocampal_memory.get()
                                                                    : nullptr,
                                       hippocampal_repetition_decay_,
                                       hippocampal_association_decay_);

    // TD-050 (September 14, 2026): beam search gets its own, deliberately non-cached model_fn —
    // same reasoning as gpu_generate_response()'s identical guard: a single shared KV cache
    // cannot correctly serve beam search's multiple diverging hypotheses (each beam's calls
    // would corrupt the cache the others depend on). This mirrors generate_response_with_
    // strategy()'s own CPU "beam" branch, which never uses DecoderKVCache either. Every other
    // strategy below shares one real GPU-resident incremental cache instead of gpu_decode()'s
    // previous full-recompute-every-step call.
    auto beam_model_fn = [this, &committer, &gpu_encoder_out, &gpu_wm](const std::vector<int>& tokens) -> Matrix {
        adai::gpu::GPUMatrix dec_out = decoder->gpu_decode(tokens, gpu_encoder_out, gpu_wm.get(),
                                                                hippocampal_memory.get(),
                                                                hippocampal_repetition_alpha_,
                                                                hippocampal_repetition_decay_,
                                                                hippocampal_cross_reference_alpha_,
                                                                hippocampal_association_decay_);
            if (committer.active()) {
                committer.observe(tokens.size(), decoder->last_hippocampal_step_weights());
            }
        adai::gpu::GPUMatrix logits = lm_head->gpu_forward(dec_out);
        const int tgt = static_cast<int>(tokens.size());
        std::vector<float> last_row(vocab_size);
        adai::gpu::matrix_download_gpu(logits.device_ptr() + (tgt - 1) * vocab_size,
                                       last_row.data(), vocab_size);
        Matrix last_logits(1, vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            last_logits.data[0][v] = last_row[v];
        }
        return last_logits;
    };

    // TODO: See TD-387 in TECHNICAL_DEBT.md - input longer than max_seq_length throws from the cross-attention cache
    adai::gpu::GPUDecoderKVCache gpu_kv_cache(decoder_layers, max_seq_length, d_model);
    size_t processed_length = 0;
    auto model_fn = [this, &committer, &gpu_encoder_out, &gpu_kv_cache, &processed_length,
                     &gpu_wm](const std::vector<int>& tokens) -> Matrix {
        std::vector<int> new_tokens(tokens.begin() + static_cast<std::ptrdiff_t>(processed_length),
                                    tokens.end());
        adai::gpu::GPUMatrix dec_out =
            decoder->gpu_decode_step(new_tokens, gpu_kv_cache, gpu_encoder_out, gpu_wm.get(),
                                     hippocampal_memory.get(),
                                     hippocampal_repetition_alpha_,
                                     hippocampal_repetition_decay_,
                                     hippocampal_cross_reference_alpha_,
                                     hippocampal_association_decay_);
            if (committer.active()) {
                committer.observe(tokens.size(), decoder->last_hippocampal_step_weights());
            }
        processed_length = tokens.size();
        adai::gpu::GPUMatrix logits = lm_head->gpu_forward(dec_out);

        const int tgt = static_cast<int>(new_tokens.size());
        std::vector<float> last_row(vocab_size);
        adai::gpu::matrix_download_gpu(logits.device_ptr() + (tgt - 1) * vocab_size,
                                       last_row.data(), vocab_size);

        Matrix last_logits(1, vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            last_logits.data[0][v] = last_row[v];
        }
        return last_logits;
    };

    std::vector<int> output_tokens;
    if (normalized_strategy == "greedy") {
        output_tokens = generator->generate_greedy(model_fn, {bos_token_id});
    } else if (normalized_strategy == "beam") {
        output_tokens = generator->generate_beam_search(beam_model_fn, {bos_token_id});
    } else if (normalized_strategy == "sampling") {
        output_tokens = generator->generate_sampling(model_fn, {bos_token_id}, temperature);
    } else if (normalized_strategy == "topk") {
        output_tokens = generator->generate_top_k(model_fn, {bos_token_id}, top_k);
    } else {
        // Default to nucleus sampling (including explicit "nucleus" strategy) — matches
        // ChatbotAPI::generate_response()'s own default-branch semantics exactly, since that's
        // this method's actual caller.
        output_tokens = generator->generate_nucleus(model_fn, {bos_token_id}, top_p);
    }

    std::string response = tokenizer->decode(output_tokens, true);
    committer.flush();
    maybe_write_hippocampal_memory(response);  // TD-342, as the CPU paths (TD-185)
    return response;
}

std::vector<adai::gpu::GPUParamRef> EncoderDecoderModel::gpu_collect_parameters() {
    std::vector<adai::gpu::GPUParamRef> out;
    encoder->gpu_collect_parameters(out);
    decoder->gpu_collect_parameters(out);
    lm_head->gpu_collect_parameters(out);
    return out;
}

void EncoderDecoderModel::gpu_download_weights() {
    encoder->gpu_download_weights();
    decoder->gpu_download_weights();
    lm_head->gpu_download_weights();
}

void EncoderDecoderModel::gpu_download_grads() {
    encoder->gpu_download_grads();
    decoder->gpu_download_grads();
    lm_head->gpu_download_grads();
}

void EncoderDecoderModel::gpu_sync_weights() {
    encoder->gpu_upload_weights();
    decoder->gpu_upload_weights();
    lm_head->gpu_upload_weights();
}

void EncoderDecoderModel::gpu_synchronize() {
    adai::gpu::GPUManager::synchronize();
}
#endif
