// @adai-status: beta        (capped by TD-504; TD-566's device-loss stop doesn't cover LeJEPA passes — see TECHNICAL_DEBT.md)
// @adai-version: 0.13.0
// @adai-reviewed: 2026-11-22

#include "LeJEPAEncoder.hpp"
#include "BinaryIO.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <type_traits>
#ifdef ADAI_ENABLE_GPU
#include "GPUOptimizer.hpp"
#endif

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

LeJEPAEncoder::~LeJEPAEncoder() = default;

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
        encoder_blocks.back()->scale_residual_init(num_layers);  // TD-371
    }

    // Predictor's hidden_dim has no dedicated constructor parameter — the plan's own
    // constructor signature is deliberately identical to LLMEncoder's (see the class doc's
    // "same shape contract" note), so reusing d_ff (already sized for this exact "expand, then
    // project back to d_model" shape inside every EncoderBlock's own FeedForward) is a natural,
    // parameter-free default rather than inventing a new argument for a component whose actual
    // use starts in TD-178.
    predictor = std::make_unique<Predictor>(d_model, d_ff);
    sigreg = std::make_unique<SIGReg>(d_model, sigreg_num_sketches);

    // TD-429: one debug line (an 8-line std::cout banner on every construction before).
    adai::Logger::debug("LeJEPAEncoder: vocab {}, d_model {}, {} layers, {} heads, d_ff {}, "
                        "max_seq_length {}, {} SIGReg sketches",
                        vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length,
                        sigreg_num_sketches);
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

bool LeJEPAEncoder::tokenize_one(const std::string& text, std::vector<int>& ids) {
    ++texts_tokenized_;
    try {
        ids = tokenizer->encode(text, true);
    } catch (const std::exception&) {
        return false;  // e.g. an empty text: one bad sample mustn't fail the whole batch (TD-429)
    }
    if (static_cast<int>(ids.size()) > max_seq_length) {
        ids.resize(max_seq_length);
    }
    return ids.size() >= 2;  // too short to mask a span (train_step() throws for this)
}

std::vector<std::vector<int>> LeJEPAEncoder::tokenize_batch(const std::vector<std::string>& texts,
                                                             std::vector<size_t>& trained) {
    std::vector<std::vector<int>> batch_ids;
    for (size_t t = 0; t < texts.size(); ++t) {
        std::vector<int> ids;
        if (tokenize_one(texts[t], ids)) {
            trained.push_back(t);
            batch_ids.push_back(std::move(ids));
        }
    }
    return batch_ids;
}

void LeJEPAEncoder::prepare_texts(const std::vector<std::string>& texts) {
    prepared_ids_.assign(texts.size(), {});
    prepared_ok_.assign(texts.size(), false);
    for (size_t t = 0; t < texts.size(); ++t) {
        prepared_ok_[t] = tokenize_one(texts[t], prepared_ids_[t]);
    }
#ifdef ADAI_ENABLE_GPU
    gpu_prepared_.reset();  // uploaded when training on the device first needs it
#endif
}

LeJEPAEncoder::BatchStepResult LeJEPAEncoder::train_prepared(size_t first, size_t last) {
    last = std::min(last, prepared_ids_.size());
    std::vector<size_t> trained;
    for (size_t t = first; t < last; ++t) {
        if (prepared_ok_[t]) {
            trained.push_back(t);
        }
    }
    BatchStepResult result;
#ifdef ADAI_ENABLE_GPU
    if (gpu_optimizer_) {
        if (!gpu_prepared_) {  // every prepared text's IDs, end to end, in one upload for the whole pass
            std::vector<int> all;
            prepared_offset_.assign(prepared_ids_.size(), 0);
            for (size_t t = 0; t < prepared_ids_.size(); ++t) {
                prepared_offset_[t] = static_cast<int>(all.size());
                if (prepared_ok_[t]) {
                    all.insert(all.end(), prepared_ids_[t].begin(), prepared_ids_[t].end());
                }
            }
            gpu_prepared_ = std::make_unique<adai::gpu::GPUMemory<int>>(std::max<size_t>(all.size(), 1));
            if (!all.empty()) {
                gpu_prepared_->copy_from_host(all.data(), all.size());
            }
        }
        std::vector<int> lengths;
        std::vector<const int*> device_ids;
        for (size_t t : trained) {
            lengths.push_back(static_cast<int>(prepared_ids_[t].size()));
            device_ids.push_back(gpu_prepared_->get() + prepared_offset_[t]);
        }
        result = gpu_train_core(lengths, device_ids);
    } else
#endif
    {
        std::vector<std::vector<int>> ids;
        for (size_t t : trained) {
            ids.push_back(prepared_ids_[t]);
        }
        result = train_ids_host(ids);
    }
    result.trained.clear();
    for (size_t t : trained) {
        result.trained.push_back(t - first);
    }
    return result;
}

void LeJEPAEncoder::choose_span(int seq_len, int& span_start, int& span_len) {
    // A contiguous span becomes the "target" region, masked out (replaced with the tokenizer's own unk
    // token) in the context view. Placement is randomized per call — see kMaskRatio's own doc comment for
    // the ratio's rationale.
    span_len = std::max(1, static_cast<int>(std::round(seq_len * kMaskRatio)));
    span_len = std::min(span_len, seq_len - 1);  // leave at least one context token unmasked
    std::uniform_int_distribution<int> start_dist(0, seq_len - span_len);
    span_start = start_dist(span_rng_);
}

LeJEPAEncoder::BatchStepResult LeJEPAEncoder::train_batch(const std::vector<std::string>& texts) {
    std::vector<size_t> trained;
    const std::vector<std::vector<int>> batch_ids = tokenize_batch(texts, trained);
#ifdef ADAI_ENABLE_GPU
    if (gpu_optimizer_) {  // TD-588: training on the device
        BatchStepResult device = gpu_train_batch(batch_ids);
        device.trained = std::move(trained);
        return device;
    }
#endif
    BatchStepResult result = train_ids_host(batch_ids);
    result.trained = std::move(trained);
    return result;
}

// train_batch()'s host step for tokenized texts (each at least 2 IDs). `trained` is left to the caller.
LeJEPAEncoder::BatchStepResult LeJEPAEncoder::train_ids_host(const std::vector<std::vector<int>>& batch_ids) {
    BatchStepResult result;
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

// The host step (builds without a GPU, or before gpu_begin_training()); gpu_train_batch() is the device one (TD-588).
float LeJEPAEncoder::train_tokens(const std::vector<int>& token_ids, bool per_sequence_sigreg,
                                  const std::vector<float>* pooled_sigreg_grad,
                                  float& sigreg_loss_out) {
    const int seq_len = static_cast<int>(token_ids.size());

    // View construction: the context view masks a span (choose_span()); the target view is simply the
    // original, unmasked sequence.
    int span_start = 0, span_len = 0;
    choose_span(seq_len, span_start, span_len);
    last_masking_ratio_ = static_cast<float>(span_len) / static_cast<float>(seq_len);

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

        // TD-428: one update from both views' gradients. backward() accumulates (TD-195), so the
        // target view's and the context view's gradients add up in every shared weight.
        // The encoder's activation cache holds the target view (the last encode_tokens() above).
        backward(grad_target_total);

        // The predictor uses its own cache. Re-running the context view puts its activations back
        // in the encoder's cache; the weights haven't changed, so they are the same ones.
        Matrix grad_ctx_embeddings = predictor->backward(grad_predicted);
        encode_tokens(context_tokens);
        backward(grad_ctx_embeddings);

        // Read before update_weights() zeroes the gradients: see get_last_gradient_norm(). Stays
        // 0 without an optimizer, which has no combined-norm primitive across sub-components.
        last_gradient_norm_ = optimizer_ ? optimizer_->get_gradient_norm() : 0.0f;
        update_weights();
    } else {
        last_gradient_norm_ = 0.0f;
    }

    return predictor_loss;
}

void LeJEPAEncoder::set_tokenizer_mode(TokenizerMode mode) {
    tokenizer = std::make_unique<BPETokenizer>(mode);
}

void LeJEPAEncoder::load_tokenizer_vocab(const std::string& vocab_file) {
    tokenizer->load_vocab(vocab_file);
    adai::Logger::info("Loaded LeJEPA encoder tokenizer vocabulary from {}", vocab_file);
}

void LeJEPAEncoder::build_tokenizer(const std::vector<std::string>& corpus, int vocab_size) {
    tokenizer->build_vocab(corpus, vocab_size);
    adai::Logger::info("Built LeJEPA encoder tokenizer with vocabulary size {}",
                       tokenizer->get_vocab_size());
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

    // TD-429: checked writes (a failed write or flush throws instead of leaving a short config).
    adai::binio::Writer config_file(directory + "/" + kConfigFile);
    for (int field : {vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length}) {
        config_file.value(field, "LeJEPA config");
    }
    config_file.close();

    token_embedding->save_weights(directory + "/" + kTokenEmbeddingFile);

    for (size_t i = 0; i < encoder_blocks.size(); ++i) {
        encoder_blocks[i]->save_weights(directory + "/" + encoder_block_file(i));
    }

    final_norm->save_weights(directory + "/" + kFinalNormFile);
    predictor->save(directory + "/" + kPredictorFile);

    adai::Logger::info("Saved LeJEPA encoder weights to {}", directory);
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
    // TD-429: checked reads. A short config used to leave its fields 0 and fail as a misleading
    // "architecture mismatch"; a short or over-long one now throws, before anything changes.
    adai::binio::Reader config_file(directory + "/" + kConfigFile);
    const int loaded_vocab_size = config_file.value<int>("vocab_size");
    const int loaded_d_model = config_file.value<int>("d_model");
    const int loaded_num_layers = config_file.value<int>("num_layers");
    const int loaded_num_heads = config_file.value<int>("num_heads");
    const int loaded_d_ff = config_file.value<int>("d_ff");
    const int loaded_max_seq_length = config_file.value<int>("max_seq_length");
    config_file.expect_end();

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
    // TD-429: no weight depends on max_seq_length (the positional encoding is the fixed sinusoidal
    // table, built for this instance's length), so a different one loads, with a warning.
    if (loaded_max_seq_length != max_seq_length) {
        adai::Logger::warn("LeJEPA weights in {} were saved with max_seq_length {}; loading them "
                           "with max_seq_length {} (positional encoding is fixed, no weights depend "
                           "on it)",
                           directory, loaded_max_seq_length, max_seq_length);
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

    adai::Logger::info("Loaded LeJEPA encoder weights from {}", directory);
}

void LeJEPAEncoder::print_config() const {
    adai::Logger::info("LeJEPAEncoder configuration: vocab {}, d_model {}, {} encoder layers, {} "
                       "heads, d_ff {}, max_seq_length {}, SIGReg lambda {}",
                       vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length,
                       sigreg_lambda);
}

#ifdef ADAI_ENABLE_GPU
adai::gpu::GPUMatrix LeJEPAEncoder::gpu_encode(const std::string& text) {
    std::vector<int> token_ids = tokenizer->encode(text, true);
    if (static_cast<int>(token_ids.size()) > max_seq_length) {
        token_ids.resize(max_seq_length);
    }
    gpu_ensure_weights();
    adai::gpu::GPUMatrix x = token_embedding->gpu_forward(token_ids, gpu_pe_.get(), gpu_pe_->rows);
    for (auto& block : encoder_blocks) {
        x = block->gpu_forward(x);
    }
    return final_norm->gpu_forward(x);
}

// Uploads the host weights when the device copy is stale (first use, or after load()). During device
// training the device copy is the live one, so this does nothing.
void LeJEPAEncoder::gpu_ensure_weights() {
    if (gpu_weights_current_) {
        return;
    }
    token_embedding->gpu_upload_weights();
    for (auto& block : encoder_blocks) {
        block->gpu_upload_weights();
    }
    final_norm->gpu_upload_weights();
    predictor->gpu_upload_weights();
    gpu_pe_ = std::make_unique<adai::gpu::GPUMatrix>(positional_encoding->get_encoding().to_gpu());
    gpu_weights_current_ = true;
}

// encode_tokens() on the device, from IDs already there. The layers cache their activations for
// gpu_backward(), as the host layers do.
adai::gpu::GPUMatrix LeJEPAEncoder::gpu_encode_ids(const int* device_ids, int n) {
    adai::gpu::GPUMatrix x = token_embedding->gpu_forward(device_ids, n, gpu_pe_.get(), gpu_pe_->rows);
    for (auto& block : encoder_blocks) {
        x = block->gpu_forward(x);
    }
    return final_norm->gpu_forward(x);
}

void LeJEPAEncoder::gpu_backward(const adai::gpu::GPUMatrix& grad_output) {
    adai::gpu::GPUMatrix grad = final_norm->gpu_backward(grad_output);
    for (int i = num_layers - 1; i >= 0; --i) {
        grad = encoder_blocks[static_cast<size_t>(i)]->gpu_backward(grad);
    }
    token_embedding->gpu_backward(grad);
}

void LeJEPAEncoder::gpu_zero_grads() {
    token_embedding->gpu_zero_grads();
    for (auto& block : encoder_blocks) {
        block->gpu_zero_grads();
    }
    final_norm->gpu_zero_grads();
    predictor->gpu_zero_grads();
}

void LeJEPAEncoder::set_gpu_diagnostics(float* acc) {
    for (auto& block : encoder_blocks) {
        block->get_feed_forward()->set_gpu_saturation_accumulator(acc);
        block->get_self_attention()->set_gpu_entropy_accumulator(acc ? acc + 2 : nullptr);
    }
}

std::size_t LeJEPAEncoder::gpu_begin_training() {
    if (!optimizer_) {
        throw std::logic_error("LeJEPAEncoder::gpu_begin_training: register_parameters_with_optimizer() first");
    }
    gpu_ensure_weights();
    gpu_zero_grads();
    std::vector<adai::gpu::GPUParamRef> params;
    token_embedding->gpu_collect_parameters(params);
    for (int l = 0; l < num_layers; ++l) {
        encoder_blocks[static_cast<size_t>(l)]->gpu_collect_parameters(params, l);
    }
    final_norm->gpu_collect_parameters(params);
    predictor->gpu_collect_parameters(params);
    gpu_optimizer_ = std::make_unique<GPUOptimizer>(*optimizer_, params);
    const std::size_t host_groups = gpu_optimizer_->host_group_count();
    if (host_groups > 0) {
        adai::Logger::warn("LeJEPAEncoder: {} optimizer groups have no device copy and step on the host", host_groups);
    }
    adai::Logger::info("LeJEPAEncoder: training on the device ({} parameter groups)",
                       gpu_optimizer_->device_group_count());
    return host_groups;
}

void LeJEPAEncoder::sync_host_from_device() {
    if (!gpu_optimizer_) {
        return;
    }
    token_embedding->gpu_download_weights();
    for (auto& block : encoder_blocks) {
        block->gpu_download_weights();
    }
    final_norm->gpu_download_weights();
    predictor->gpu_download_weights();
    gpu_optimizer_->download_state();
}

LeJEPAEncoder::BatchStepResult LeJEPAEncoder::gpu_train_batch(const std::vector<std::vector<int>>& ids) {
    if (!gpu_optimizer_) {
        throw std::logic_error("LeJEPAEncoder::gpu_train_batch: gpu_begin_training() first");
    }
    // One upload of every text's IDs, end to end.
    std::vector<int> all_ids;
    std::vector<int> lengths;
    std::vector<size_t> offset;
    for (const auto& v : ids) {
        if (v.size() < 2 || static_cast<int>(v.size()) > max_seq_length) {
            throw std::invalid_argument("LeJEPAEncoder::gpu_train_batch: each text needs 2 to max_seq_length IDs");
        }
        offset.push_back(all_ids.size());
        lengths.push_back(static_cast<int>(v.size()));
        all_ids.insert(all_ids.end(), v.begin(), v.end());
    }
    if (all_ids.size() > gpu_ids_capacity_) {
        gpu_ids_ = std::make_unique<adai::gpu::GPUMemory<int>>(all_ids.size());
        gpu_ids_capacity_ = all_ids.size();
    }
    if (!all_ids.empty()) {
        gpu_ids_->copy_from_host(all_ids.data(), all_ids.size());
    }
    std::vector<const int*> device_ids;
    for (size_t o : offset) {
        device_ids.push_back(gpu_ids_->get() + o);
    }
    BatchStepResult result = gpu_train_core(lengths, device_ids);
    for (size_t b = 0; b < ids.size(); ++b) {
        result.trained.push_back(b);
    }
    return result;
}

// The device step for texts whose IDs are on the device (TD-588; TD-589's prepared texts or one batch's
// upload). `trained` is left to the caller.
LeJEPAEncoder::BatchStepResult LeJEPAEncoder::gpu_train_core(const std::vector<int>& lengths,
                                                             const std::vector<const int*>& device_ids) {
    BatchStepResult result;
    const int batch = static_cast<int>(lengths.size());
    if (batch == 0) {
        return result;
    }
    if (static_cast<std::size_t>(max_seq_length) > gpu_ctx_capacity_) {
        gpu_ctx_ids_ = std::make_unique<adai::gpu::GPUMemory<int>>(static_cast<std::size_t>(max_seq_length));
        gpu_ctx_capacity_ = static_cast<std::size_t>(max_seq_length);
    }
    const std::size_t stats_size = 3 + 2 * static_cast<std::size_t>(batch);
    if (stats_size > gpu_stats_capacity_) {
        gpu_stats_ = std::make_unique<adai::gpu::GPUMemory<float>>(stats_size);
        gpu_stats_capacity_ = stats_size;
    }
    float* stats = gpu_stats_->get();  // [sigreg loss, var mean, var stddev, losses..., cosines...]

    // Phase 1: every text's target view, mean-pooled into one row each.
    adai::gpu::GPUMatrix pooled(batch, d_model);
    for (int b = 0; b < batch; ++b) {
        const int n = lengths[static_cast<size_t>(b)];
        const adai::gpu::GPUMatrix tgt = gpu_encode_ids(device_ids[static_cast<size_t>(b)], n);
        adai::gpu::lejepa_mean_pool_gpu(tgt.device_ptr(), n, d_model, pooled.device_ptr() + static_cast<size_t>(b) * d_model);
    }
    sigreg->resample_directions();  // TD-502: one draw for this batch's loss, stats and gradient
    const bool use_sigreg = requires_grad && batch >= 2;
    adai::gpu::GPUMatrix pooled_grad(1, 1);
    sigreg->gpu_compute(pooled, stats, use_sigreg ? &pooled_grad : nullptr);

    // Phase 2: train each text, adding its pooled row's SIGReg gradient (scaled by the batch size, as
    // train_batch() explains, and spread over the text's rows).
    const int unk_id = tokenizer->get_unk_token_id();
    double masking_sum = 0.0, grad_norm_sq_sum = 0.0;
    for (int b = 0; b < batch; ++b) {
        const int n = lengths[static_cast<size_t>(b)];
        const int* tgt_ids = device_ids[static_cast<size_t>(b)];
        int span_start = 0, span_len = 0;
        choose_span(n, span_start, span_len);
        masking_sum += static_cast<double>(span_len) / n;
        adai::gpu::lejepa_context_ids_gpu(tgt_ids, gpu_ctx_ids_->get(), n, span_start, span_len, unk_id);

        // Context view first, then the target view, so the encoder's caches hold the target view for
        // the first backward (as in train_tokens()).
        const adai::gpu::GPUMatrix ctx = gpu_encode_ids(gpu_ctx_ids_->get(), n);
        const adai::gpu::GPUMatrix tgt = gpu_encode_ids(tgt_ids, n);
        const adai::gpu::GPUMatrix predicted = predictor->gpu_forward(ctx);
        adai::gpu::GPUMatrix grad_predicted(n, d_model);
        adai::gpu::GPUMatrix grad_target(n, d_model);
        adai::gpu::lejepa_masked_mse_gpu(predicted.device_ptr(), tgt.device_ptr(), n, d_model, span_start, span_len,
                                         stats + 3 + b, stats + 3 + batch + b, grad_predicted.device_ptr(),
                                         grad_target.device_ptr());
        if (!requires_grad) {
            continue;
        }
        if (use_sigreg) {
            adai::gpu::lejepa_add_row_broadcast_gpu(grad_target.device_ptr(), n, d_model,
                                                    pooled_grad.device_ptr() + static_cast<size_t>(b) * d_model,
                                                    sigreg_lambda * static_cast<float>(batch) / static_cast<float>(n));
        }
        gpu_backward(grad_target);
        // The predictor keeps its own cache. Re-encoding the context view puts its activations back in
        // the encoder's caches (the layers hold one set each; a second set per layer would save this
        // forward pass but touch every layer's device state). The weights haven't changed, so they are
        // the same activations.
        const adai::gpu::GPUMatrix grad_ctx = predictor->gpu_backward(grad_predicted);
        gpu_encode_ids(gpu_ctx_ids_->get(), n);
        gpu_backward(grad_ctx);

        double sq = 0.0;
        for (float g : gpu_optimizer_->gradient_sq_norms()) {  // one float per group
            sq += g;
        }
        grad_norm_sq_sum += sq;
        gpu_optimizer_->step(1.0f);
        gpu_zero_grads();
    }

    // The batch's results, in one download.
    std::vector<float> host_stats(stats_size);
    gpu_stats_->copy_to_host(host_stats.data(), stats_size);
    result.sigreg_loss = host_stats[0];
    last_sigreg_variance_mean_ = host_stats[1];
    last_sigreg_variance_stddev_ = host_stats[2];
    double cosine_sum = 0.0;
    for (int b = 0; b < batch; ++b) {
        result.predictor_losses.push_back(host_stats[3 + static_cast<size_t>(b)]);
        cosine_sum += host_stats[3 + static_cast<size_t>(batch) + b];
    }
    last_masking_ratio_ = static_cast<float>(masking_sum / batch);
    last_predictor_target_cosine_sim_ = static_cast<float>(cosine_sum / batch);
    last_gradient_norm_ = requires_grad ? static_cast<float>(std::sqrt(grad_norm_sq_sum / batch)) : 0.0f;
    return result;
}
#endif

