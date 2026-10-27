// @adai-status: experimental
// @adai-version: 0.10.0
// @adai-reviewed: 2026-10-26

#include "LeJEPAEncoder.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
// LeJEPAEncoder::save()/load() use a fixed set of filenames inside the caller-supplied
// directory, mirroring ModelSerializer's own "<dir>/<fixed-name>" convention rather than
// LLMEncoder's base-filename-plus-suffix one (see the class doc's rationale).
constexpr const char* kConfigFile = "config.bin";
constexpr const char* kTokenEmbeddingFile = "token_embedding.bin";
constexpr const char* kFinalNormFile = "final_norm.bin";
constexpr const char* kPredictorFile = "predictor.bin";  // TD-427

std::string encoder_block_file(size_t index) {
    return "encoder_block_" + std::to_string(index) + ".bin";
}

// TD-178's own span-masking view construction: the fraction of the token sequence replaced by
// the tokenizer's own unk token to form the "context" view (the complementary "target" view is
// simply the original, unmasked sequence). 25% sits between BERT-style single-token masking
// ratios (~15%) and I-JEPA's larger contiguous target-block ratios in the vision domain — a
// reasonable, documented default rather than a value copied from either without justification;
// TD-183's own config wiring is free to make this tunable later if the pilot's own results ask
// for it.
constexpr float kMaskRatio = 0.25f;
}  // namespace

LeJEPAEncoder::LeJEPAEncoder(int vocab_size, int d_model, int num_layers, int num_heads, int d_ff,
                             int max_seq_length, int sigreg_num_sketches)
    : vocab_size(vocab_size),
      d_model(d_model),
      num_layers(num_layers),
      num_heads(num_heads),
      d_ff(d_ff),
      max_seq_length(max_seq_length) {
    tokenizer = std::make_unique<BPETokenizer>();
    token_embedding = std::make_unique<TokenEmbedding>(vocab_size, d_model);
    positional_encoding = std::make_unique<PositionalEncoding>(max_seq_length, d_model);
    final_norm = std::make_unique<LayerNorm>(d_model);

    for (int i = 0; i < num_layers; i++) {
        encoder_blocks.push_back(std::make_unique<EncoderBlock>(d_model, num_heads, d_ff));
    }

    // Predictor's hidden_dim has no dedicated constructor parameter — the plan's own
    // constructor signature is deliberately identical to LLMEncoder's (see the class doc's
    // "same shape contract" note), so reusing d_ff (already sized for this exact "expand, then
    // project back to d_model" shape inside every EncoderBlock's own FeedForward) is a natural,
    // parameter-free default rather than inventing a new argument for a component whose actual
    // use starts in TD-178.
    predictor = std::make_unique<Predictor>(d_model, d_ff);
    sigreg = std::make_unique<SIGReg>(d_model, sigreg_num_sketches);

    std::cout << "LeJEPA Encoder initialized with:" << '\n';
    std::cout << "  Vocab size: " << vocab_size << '\n';
    std::cout << "  Model dimension: " << d_model << '\n';
    std::cout << "  Number of layers: " << num_layers << '\n';
    std::cout << "  Number of heads: " << num_heads << '\n';
    std::cout << "  Feed-forward dimension: " << d_ff << '\n';
    std::cout << "  Max sequence length: " << max_seq_length << '\n';
    std::cout << "  SIGReg sketches: " << sigreg_num_sketches << '\n';
}

Matrix LeJEPAEncoder::encode(const std::string& text) {
    std::vector<int> token_ids = tokenizer->encode(text, true);

    if (static_cast<int>(token_ids.size()) > max_seq_length) {
        token_ids.resize(max_seq_length);
    }

    return encode_tokens(token_ids);
}

Matrix LeJEPAEncoder::encode_tokens(const std::vector<int>& token_ids) {
    if (requires_grad) {
        cached_token_ids = token_ids;
    }

    Matrix embeddings = token_embedding->forward(token_ids);
    Matrix encoded = positional_encoding->forward(embeddings);

    for (int i = 0; i < num_layers; i++) {
        encoded = encoder_blocks[i]->forward(encoded);
    }

    encoded = final_norm->forward(encoded);

    return encoded;
}

void LeJEPAEncoder::backward(const Matrix& grad_output) {
    if (!requires_grad) {
        return;
    }

    Matrix grad = final_norm->backward(grad_output);

    for (int i = num_layers - 1; i >= 0; i--) {
        grad = encoder_blocks[i]->backward(grad);
    }

    token_embedding->backward(cached_token_ids, grad);
}

void LeJEPAEncoder::zero_grad() {
    token_embedding->zero_grad();
    for (auto& block : encoder_blocks) {
        block->zero_grad();
    }
    final_norm->zero_grad();
    predictor->zero_grad();
}

void LeJEPAEncoder::update_weights() {
    if (optimizer_) {
        // Exactly one step across every parameter this encoder (and predictor) registered with
        // it — see this method's own doc comment in LeJEPAEncoder.hpp for why calling each
        // sub-component's own update_weights() here instead would over-apply the shared
        // optimizer once per sub-component rather than once per training step.
        optimizer_->step();
        zero_grad();
    } else {
        token_embedding->update_weights();
        for (auto& block : encoder_blocks) {
            block->update_weights();
        }
        final_norm->update_weights();
        predictor->update_weights();
    }
}

std::pair<float, float> LeJEPAEncoder::train_step(const std::string& text) {
    std::vector<int> token_ids = tokenizer->encode(text, true);
    if (static_cast<int>(token_ids.size()) > max_seq_length) {
        token_ids.resize(max_seq_length);
    }
    if (token_ids.size() < 2) {
        throw std::invalid_argument(
            "LeJEPAEncoder::train_step: text must tokenize to at least 2 tokens (need both a "
            "context region and a masked target span)");
    }
    float sigreg_loss = 0.0f;
    const float predictor_loss =
        train_tokens(token_ids, /*per_sequence_sigreg=*/true, nullptr, sigreg_loss);
    return {predictor_loss, sigreg_loss};
}

LeJEPAEncoder::BatchStepResult LeJEPAEncoder::train_batch(const std::vector<std::string>& texts) {
    BatchStepResult result;
    std::vector<std::vector<int>> batch_ids;
    for (size_t t = 0; t < texts.size(); ++t) {
        std::vector<int> ids;
        try {
            ids = tokenizer->encode(texts[t], true);
        } catch (const std::exception&) {
            continue;  // e.g. an empty text: one bad sample mustn't fail the whole batch (TD-429)
        }
        if (static_cast<int>(ids.size()) > max_seq_length) {
            ids.resize(max_seq_length);
        }
        if (ids.size() < 2) {
            continue;  // too short to mask a span (train_step() throws for this)
        }
        result.trained.push_back(t);
        batch_ids.push_back(std::move(ids));
    }
    const int batch = static_cast<int>(batch_ids.size());
    if (batch == 0) {
        return result;
    }

    // Phase 1: each text's target view, mean-pooled to one row per text.
    Matrix pooled(batch, d_model);
    for (int b = 0; b < batch; ++b) {
        const Matrix tgt = encode_tokens(batch_ids[static_cast<size_t>(b)]);
        const float inv_rows = 1.0f / static_cast<float>(tgt.rows);
        for (int i = 0; i < tgt.rows; ++i) {
            for (int j = 0; j < d_model; ++j) {
                pooled(b, j) += tgt(i, j) * inv_rows;
            }
        }
    }
    sigreg->resample_directions();  // TD-502: one draw for this batch's loss, stats and gradient
    result.sigreg_loss = sigreg->compute_loss(pooled);  // 0 when batch < 2
    const auto variance_stats = sigreg->compute_variance_stats(pooled);
    last_sigreg_variance_mean_ = variance_stats.mean;
    last_sigreg_variance_stddev_ = variance_stats.stddev;
    const bool use_sigreg = requires_grad && batch >= 2;
    Matrix pooled_grad;
    if (use_sigreg) {
        pooled_grad = sigreg->backward(pooled);  // [batch, d_model]
        // The batch objective is mean_b(predictor_b) + lambda * SIGReg(pooled): per sample,
        // (1/batch) * d predictor_b + lambda * d SIGReg / d pooled_b, in one step. Each text here
        // takes its own step with its full predictor gradient, so the SIGReg part is scaled by
        // `batch` to keep that balance (unscaled, it was `batch` times weaker).
        const float scale = static_cast<float>(batch);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < d_model; ++j) {
                pooled_grad(b, j) *= scale;
            }
        }
    }

    // Phase 2: train each text, adding its pooled row's SIGReg gradient.
    double masking_sum = 0.0, cosine_sum = 0.0, grad_norm_sq_sum = 0.0;
    std::vector<float> row_grad(static_cast<size_t>(d_model));
    for (int b = 0; b < batch; ++b) {
        if (use_sigreg) {
            for (int j = 0; j < d_model; ++j) {
                row_grad[static_cast<size_t>(j)] = pooled_grad(b, j);
            }
        }
        float unused_sigreg = 0.0f;
        result.predictor_losses.push_back(train_tokens(batch_ids[static_cast<size_t>(b)],
                                                       /*per_sequence_sigreg=*/false,
                                                       use_sigreg ? &row_grad : nullptr,
                                                       unused_sigreg));
        masking_sum += last_masking_ratio_;
        cosine_sum += last_predictor_target_cosine_sim_;
        grad_norm_sq_sum += static_cast<double>(last_gradient_norm_) * last_gradient_norm_;
    }
    last_masking_ratio_ = static_cast<float>(masking_sum / batch);
    last_predictor_target_cosine_sim_ = static_cast<float>(cosine_sum / batch);
    last_gradient_norm_ = static_cast<float>(std::sqrt(grad_norm_sq_sum / batch));
    // Restore the batch-level variance stats (train_tokens() leaves them alone without
    // per_sequence_sigreg, but keep the contract explicit).
    last_sigreg_variance_mean_ = variance_stats.mean;
    last_sigreg_variance_stddev_ = variance_stats.stddev;
    return result;
}

float LeJEPAEncoder::train_tokens(const std::vector<int>& token_ids, bool per_sequence_sigreg,
                                  const std::vector<float>* pooled_sigreg_grad,
                                  float& sigreg_loss_out) {
    const int seq_len = static_cast<int>(token_ids.size());

    // View construction: a contiguous span becomes the "target" region, masked out (replaced
    // with the tokenizer's own unk token) in the context view; the target view is simply the
    // original, unmasked sequence. Span placement is randomized per call — see kMaskRatio's own
    // doc comment for the ratio's rationale.
    int span_len = std::max(1, static_cast<int>(std::round(seq_len * kMaskRatio)));
    span_len = std::min(span_len, seq_len - 1);  // leave at least one context token unmasked
    last_masking_ratio_ = static_cast<float>(span_len) / static_cast<float>(seq_len);

    std::uniform_int_distribution<int> start_dist(0, seq_len - span_len);
    const int span_start = start_dist(span_rng_);

    std::vector<int> context_tokens = token_ids;
    const int unk_id = tokenizer->get_unk_token_id();
    for (int i = span_start; i < span_start + span_len; ++i) {
        context_tokens[i] = unk_id;
    }

    // Forward passes. Order matters for backward() below: whichever of these two runs LAST is
    // the one whose activations are still the "live" cache — see the class doc's own note on
    // this. ctx_embeddings is captured by value (Matrix has value semantics), so it stays valid
    // even after the target forward pass below overwrites the shared cache.
    Matrix ctx_embeddings = encode_tokens(context_tokens);
    Matrix tgt_embeddings = encode_tokens(token_ids);

    Matrix predicted = predictor->forward(ctx_embeddings);  // [seq_len, d_model]

    // predictor_loss: mean-squared error between predicted and the real target embedding,
    // scored ONLY at the masked span's own rows — an unmasked row's context-view token is
    // literally identical to its target-view counterpart, so scoring it would teach nothing.
    const int n_masked_elems = span_len * d_model;
    float predictor_loss = 0.0f;
    for (int i = span_start; i < span_start + span_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            const float diff = predicted(i, j) - tgt_embeddings(i, j);
            predictor_loss += diff * diff;
        }
    }
    predictor_loss /= static_cast<float>(n_masked_elems);

    // Cosine similarity between the prediction and the real target embedding at the same masked
    // rows the MSE loss above scores — isolates directional accuracy from predictor_loss's
    // magnitude-conflated signal. Computed unconditionally (not gated on requires_grad) since it's
    // a pure read of already-computed forward-pass outputs, no gradient involved.
    float cos_sum = 0.0f;
    for (int i = span_start; i < span_start + span_len; ++i) {
        float dot = 0.0f, norm_p = 0.0f, norm_t = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            const float p = predicted(i, j);
            const float t = tgt_embeddings(i, j);
            dot += p * t;
            norm_p += p * p;
            norm_t += t * t;
        }
        const float denom = std::sqrt(norm_p) * std::sqrt(norm_t);
        cos_sum += (denom > 1e-8f) ? (dot / denom) : 0.0f;
    }
    last_predictor_target_cosine_sim_ = cos_sum / static_cast<float>(span_len);

    float sigreg_loss = 0.0f;
    if (per_sequence_sigreg) {
        // TD-502: new directions every step, shared by this step's loss, variance stats and
        // backward.
        sigreg->resample_directions();
        sigreg_loss = sigreg->compute_loss(tgt_embeddings);
        const auto variance_stats = sigreg->compute_variance_stats(tgt_embeddings);
        last_sigreg_variance_mean_ = variance_stats.mean;
        last_sigreg_variance_stddev_ = variance_stats.stddev;
    }
    sigreg_loss_out = sigreg_loss;

    if (requires_grad) {
        // d(predictor_loss)/d(predicted) and its mirror-image w.r.t. target_embeddings — zero
        // everywhere outside the masked span, matching the loss itself being restricted there.
        Matrix grad_predicted(seq_len, d_model);
        Matrix grad_target_from_predictor(seq_len, d_model);
        for (int i = span_start; i < span_start + span_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                const float diff = predicted(i, j) - tgt_embeddings(i, j);
                const float g = 2.0f * diff / static_cast<float>(n_masked_elems);
                grad_predicted(i, j) = g;
                grad_target_from_predictor(i, j) = -g;
            }
        }

        Matrix grad_target_total = grad_target_from_predictor;
        if (per_sequence_sigreg) {
            Matrix grad_target_from_sigreg = sigreg->backward(tgt_embeddings);
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < d_model; ++j) {
                    grad_target_total(i, j) += sigreg_lambda * grad_target_from_sigreg(i, j);
                }
            }
        } else if (pooled_sigreg_grad != nullptr) {
            // The pooled row is the mean of this text's target rows, so each row gets 1/seq_len
            // of the pooled gradient.
            const float share = sigreg_lambda / static_cast<float>(seq_len);
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < d_model; ++j) {
                    grad_target_total(i, j) += share * (*pooled_sigreg_grad)[static_cast<size_t>(j)];
                }
            }
        }

        // TODO: See TD-428 in TECHNICAL_DEBT.md - stale: TD-195 made these backward()s accumulate, so the two
        // sequential update_weights() below (two optimizer steps per sample) are no longer needed
        // TD-189: LayerNorm::backward()/FeedForward::backward()/MultiHeadAttention::backward()
        // (used inside encoder_blocks/final_norm) OVERWRITE their own gradient members on every
        // call rather than accumulating — only TokenEmbedding::backward() actually accumulates
        // (+=). Calling backward() twice back-to-back for the two views, as this method used to,
        // silently discarded whichever view's gradient the FIRST call computed for every weight
        // except token_embedding: in practice only the context view's (predictor's own) gradient
        // ever reached encoder_blocks/final_norm, and SIGReg's contribution — meant to keep this
        // encoder's *target*-view output isotropic-Gaussian — never influenced anything but the
        // embedding table. The fix: consume each view's gradient with its own update_weights()
        // call immediately after its own backward(), so nothing is overwritten before it's
        // applied. This trades one atomic combined update for two sequential ones — both views'
        // gradients still reach every shared weight, just not simultaneously, the same tradeoff
        // any sequential mini-batch update makes. Widening every shared component's backward()
        // to accumulate across calls instead would fix this more fundamentally, but is a much
        // larger, riskier change affecting every other caller of those components (see TD-189's
        // own entry in TECHNICAL_DEBT.md for why that's tracked separately).

        // Step 1: target view's gradient (predictor's target-side term + SIGReg) — cache is
        // currently the target view's (the most recent encode_tokens() call above). Applied on
        // its own so it can't be clobbered by step 2's backward() below.
        backward(grad_target_total);
        // Captured BEFORE update_weights() zeroes it — see get_last_gradient_norm()'s own doc
        // comment for why reading this from outside train_step() always sees 0.0f otherwise.
        const float grad_norm_1 = optimizer_ ? optimizer_->get_gradient_norm() : 0.0f;
        update_weights();

        // Step 2: predictor's own backward (uses its own cache from predictor->forward() above,
        // independent of the encoder's weights, so it's safe to compute after step 1's update)
        // followed by the context view's gradient into the encoder. Re-running the context
        // view's forward pass repopulates the encoder's shared cache with its own activations
        // (needed since step 1's backward() re-pointed it at the target view) — deterministic
        // given fixed weights (no dropout/randomness in EncoderBlock's own forward), modulo the
        // small change step 1's update just made to those weights, the same bounded staleness any
        // sequential mini-batch update incurs.
        Matrix grad_ctx_embeddings = predictor->backward(grad_predicted);
        encode_tokens(context_tokens);
        backward(grad_ctx_embeddings);
        const float grad_norm_2 = optimizer_ ? optimizer_->get_gradient_norm() : 0.0f;
        update_weights();

        // Combined norm across both sequential updates, matching EncoderBlock::get_gradient_norm()'s
        // own sqrt-of-sum-of-squares combining convention. 0.0f (the member's default) when no
        // optimizer is registered — the plain-SGD fallback path has no single combined-norm
        // primitive across every sub-component (Predictor/LayerNorm don't expose one), and
        // production always registers an optimizer, so this scope is intentionally narrow.
        last_gradient_norm_ = std::sqrt(grad_norm_1 * grad_norm_1 + grad_norm_2 * grad_norm_2);
    } else {
        last_gradient_norm_ = 0.0f;
    }

    return predictor_loss;
}

void LeJEPAEncoder::load_tokenizer_vocab(const std::string& vocab_file) {
    tokenizer->load_vocab(vocab_file);
    std::cout << "Loaded LeJEPA encoder tokenizer vocabulary from: " << vocab_file << '\n';
}

void LeJEPAEncoder::build_tokenizer(const std::vector<std::string>& corpus, int vocab_size) {
    tokenizer->build_vocab(corpus, vocab_size);
    std::cout << "Built LeJEPA encoder tokenizer with vocabulary size: "
              << tokenizer->get_vocab_size() << '\n';
}

void LeJEPAEncoder::set_requires_grad(bool requires_grad) {
    this->requires_grad = requires_grad;
}

void LeJEPAEncoder::set_learning_rate(float lr) {
    learning_rate = lr;
    token_embedding->learning_rate = lr;
    for (auto& block : encoder_blocks) {
        block->learning_rate = lr;
    }
    final_norm->learning_rate = lr;
    // Predictor exposes no settable learning_rate of its own (see Predictor.hpp) — it is meant
    // to always be driven through a registered Optimizer (register_parameters_with_optimizer()
    // below), not this class's plain-SGD fallback path.
}

void LeJEPAEncoder::register_parameters_with_optimizer(Optimizer& optimizer) {
    optimizer_ = &optimizer;

    token_embedding->set_optimizer(&optimizer);

    for (auto& block : encoder_blocks) {
        block->register_parameters_with_optimizer(optimizer);
    }

    final_norm->set_optimizer(&optimizer);

    predictor->register_parameters_with_optimizer(optimizer);
    // SIGReg has no learnable parameters of its own (stateless w.r.t. weights — see its class
    // doc) — nothing to register for it.
}

std::string LeJEPAEncoder::default_checkpoint_dir(const std::string& session_dir) {
    return session_dir + "/world_model_checkpoint";
}

bool LeJEPAEncoder::has_checkpoint(const std::string& directory) {
    std::error_code ec;
    return !directory.empty() &&
           std::filesystem::is_regular_file(directory + "/" + kConfigFile, ec);
}

std::string LeJEPAEncoder::resolve_checkpoint_dir(const std::string& session_dir,
                                                  const std::string& artifact_path) {
    if (!artifact_path.empty()) {
        return artifact_path;
    }
    const std::string current = default_checkpoint_dir(session_dir);
    const std::string legacy = session_dir + "/world_model";
    if (!has_checkpoint(current) && has_checkpoint(legacy)) {
        return legacy;  // saved before TD-548; the next save moves it to `current`
    }
    return current;
}

// TD-427: the Predictor is saved too. It used to be skipped, so every resumed pass started with a random
// predictor (predictor loss 0.0017 trained -> 2.05 after save/load) whose gradients disturbed the encoder.
void LeJEPAEncoder::save(const std::string& directory) const {
    std::filesystem::create_directories(directory);

    std::string config_path = directory + "/" + kConfigFile;
    std::ofstream config_file(config_path, std::ios::binary);
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + config_path);
    }
    config_file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&d_ff), sizeof(int));
    config_file.write(reinterpret_cast<const char*>(&max_seq_length), sizeof(int));
    config_file.close();

    token_embedding->save_weights(directory + "/" + kTokenEmbeddingFile);

    for (size_t i = 0; i < encoder_blocks.size(); ++i) {
        encoder_blocks[i]->save_weights(directory + "/" + encoder_block_file(i));
    }

    final_norm->save_weights(directory + "/" + kFinalNormFile);
    predictor->save(directory + "/" + kPredictorFile);

    std::cout << "Saved LeJEPA encoder weights to " << directory << '\n';
}

std::unique_ptr<LeJEPAEncoder> LeJEPAEncoder::clone() const {
    namespace fs = std::filesystem;
    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path dir = fs::temp_directory_path() /
                         ("adai_lejepa_clone_" +
                          std::to_string(std::hash<const void*>{}(static_cast<const void*>(this))) +
                          "_" + std::to_string(ts));
    struct RemoveDir {
        fs::path path;
        ~RemoveDir() {
            std::error_code ec;
            fs::remove_all(path, ec);  // best-effort
        }
    } cleanup{dir};

    save(dir.string());
    auto copy = std::make_unique<LeJEPAEncoder>(vocab_size, d_model, num_layers, num_heads, d_ff,
                                                max_seq_length, get_sigreg_num_sketches());
    copy->load(dir.string());
    if (tokenizer->get_vocab_size() > 0) {
        const std::string vocab = (dir / "tokenizer.vocab").string();
        tokenizer->save_vocab(vocab);
        copy->tokenizer->load_vocab(vocab);
    }
    copy->set_sigreg_lambda(sigreg_lambda);
    copy->set_learning_rate(learning_rate);
    copy->set_requires_grad(requires_grad);
    return copy;
}

void LeJEPAEncoder::load(const std::string& directory) {
#ifdef ADAI_ENABLE_GPU
    gpu_weights_current_ = false;  // TD-342: re-upload before the next gpu_encode()
#endif
    std::string config_path = directory + "/" + kConfigFile;
    std::ifstream config_file(config_path, std::ios::binary);
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + config_path);
    }

    // TODO: See TD-429 in TECHNICAL_DEBT.md - reads unchecked, max_seq_length not compared; std::cout here and in
    // the constructor/save/tokenizer methods
    int loaded_vocab_size = 0, loaded_d_model = 0, loaded_num_layers = 0;
    int loaded_num_heads = 0, loaded_d_ff = 0, loaded_max_seq_length = 0;

    config_file.read(reinterpret_cast<char*>(&loaded_vocab_size), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_num_layers), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(int));
    config_file.read(reinterpret_cast<char*>(&loaded_max_seq_length), sizeof(int));
    config_file.close();

    if (loaded_vocab_size != vocab_size || loaded_d_model != d_model ||
        loaded_num_layers != num_layers || loaded_num_heads != num_heads || loaded_d_ff != d_ff) {
        throw std::runtime_error(
            "LeJEPAEncoder architecture mismatch: saved (vocab=" +
            std::to_string(loaded_vocab_size) + ", d_model=" + std::to_string(loaded_d_model) +
            ", layers=" + std::to_string(loaded_num_layers) +
            ", heads=" + std::to_string(loaded_num_heads) +
            ", d_ff=" + std::to_string(loaded_d_ff) +
            ") vs current (vocab=" + std::to_string(vocab_size) +
            ", d_model=" + std::to_string(d_model) + ", layers=" + std::to_string(num_layers) +
            ", heads=" + std::to_string(num_heads) + ", d_ff=" + std::to_string(d_ff) + ")");
    }

    token_embedding->load_weights(directory + "/" + kTokenEmbeddingFile);

    for (size_t i = 0; i < encoder_blocks.size(); ++i) {
        encoder_blocks[i]->load_weights(directory + "/" + encoder_block_file(i));
    }

    final_norm->load_weights(directory + "/" + kFinalNormFile);

    // TD-427: a checkpoint from before the predictor was saved has none, and a damaged predictor file
    // shouldn't cost the encoder, so either case keeps a fresh predictor (re-learned this pass).
    const std::string predictor_path = directory + "/" + kPredictorFile;
    try {
        if (!predictor->load(predictor_path)) {
            adai::Logger::warn("LeJEPAEncoder::load(): no {} in {} (checkpoint predates TD-427); "
                               "the predictor starts fresh",
                               kPredictorFile, directory);
        }
    } catch (const std::exception& e) {
        adai::Logger::warn("LeJEPAEncoder::load(): predictor not restored ({}); it starts fresh",
                           e.what());
    }

    std::cout << "Loaded LeJEPA encoder weights from " << directory << '\n';
}

void LeJEPAEncoder::print_config() const {
    std::cout << "\n=== LeJEPA Encoder Configuration ===" << '\n';
    std::cout << "Vocabulary size: " << vocab_size << '\n';
    std::cout << "Model dimension (d_model): " << d_model << '\n';
    std::cout << "Number of encoder layers: " << num_layers << '\n';
    std::cout << "Number of attention heads: " << num_heads << '\n';
    std::cout << "Feed-forward dimension: " << d_ff << '\n';
    std::cout << "Max sequence length: " << max_seq_length << '\n';
    std::cout << "SIGReg lambda: " << sigreg_lambda << '\n';
    std::cout << "=====================================\n" << '\n';
}

#ifdef ADAI_ENABLE_GPU
adai::gpu::GPUMatrix LeJEPAEncoder::gpu_encode(const std::string& text) {
    std::vector<int> token_ids = tokenizer->encode(text, true);
    if (static_cast<int>(token_ids.size()) > max_seq_length) {
        token_ids.resize(max_seq_length);
    }
    if (!gpu_weights_current_) {
        token_embedding->gpu_upload_weights();
        for (auto& block : encoder_blocks) {
            block->gpu_upload_weights();
        }
        final_norm->gpu_upload_weights();
        gpu_pe_ = std::make_unique<adai::gpu::GPUMatrix>(positional_encoding->get_encoding().to_gpu());
        gpu_weights_current_ = true;
    }
    adai::gpu::GPUMatrix x = token_embedding->gpu_forward(token_ids, gpu_pe_.get(), gpu_pe_->rows);
    for (auto& block : encoder_blocks) {
        x = block->gpu_forward(x);
    }
    return final_norm->gpu_forward(x);
}
#endif

