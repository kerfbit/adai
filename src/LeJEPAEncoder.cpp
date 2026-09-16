// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-15

#include "LeJEPAEncoder.hpp"

#include <algorithm>
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
                             int max_seq_length)
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
    sigreg = std::make_unique<SIGReg>(d_model);

    std::cout << "LeJEPA Encoder initialized with:" << '\n';
    std::cout << "  Vocab size: " << vocab_size << '\n';
    std::cout << "  Model dimension: " << d_model << '\n';
    std::cout << "  Number of layers: " << num_layers << '\n';
    std::cout << "  Number of heads: " << num_heads << '\n';
    std::cout << "  Feed-forward dimension: " << d_ff << '\n';
    std::cout << "  Max sequence length: " << max_seq_length << '\n';
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

    if (requires_grad) {
        cached_encoder_outputs.clear();
        cached_encoder_outputs.push_back(encoded);
    }

    for (int i = 0; i < num_layers; i++) {
        encoded = encoder_blocks[i]->forward(encoded);
        if (requires_grad) {
            cached_encoder_outputs.push_back(encoded);
        }
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

    const int seq_len = static_cast<int>(token_ids.size());
    if (seq_len < 2) {
        throw std::invalid_argument(
            "LeJEPAEncoder::train_step: text must tokenize to at least 2 tokens (need both a "
            "context region and a masked target span)");
    }

    // View construction: a contiguous span becomes the "target" region, masked out (replaced
    // with the tokenizer's own unk token) in the context view; the target view is simply the
    // original, unmasked sequence. Span placement is randomized per call — see kMaskRatio's own
    // doc comment for the ratio's rationale.
    int span_len = std::max(1, static_cast<int>(std::round(seq_len * kMaskRatio)));
    span_len = std::min(span_len, seq_len - 1);  // leave at least one context token unmasked

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> start_dist(0, seq_len - span_len);
    const int span_start = start_dist(gen);

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

    const float sigreg_loss = sigreg->compute_loss(tgt_embeddings);

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

        Matrix grad_target_from_sigreg = sigreg->backward(tgt_embeddings);
        Matrix grad_target_total(seq_len, d_model);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                grad_target_total(i, j) =
                    grad_target_from_predictor(i, j) + sigreg_lambda * grad_target_from_sigreg(i, j);
            }
        }

        // Backward into the target view's own encode pass first — its activations are still
        // the live cache (the most recent encode_tokens() call above). No stop-gradient/EMA
        // teacher here per LeJEPA's own design (see the class doc) — both views' gradients
        // flow into the same shared encoder weights.
        backward(grad_target_total);

        // Predictor's own backward uses ITS OWN cache (from predictor->forward() above),
        // independent of the encoder's cache state, so this is safe regardless of order.
        Matrix grad_ctx_embeddings = predictor->backward(grad_predicted);

        // Re-run the context view's forward pass once more to repopulate the encoder's shared
        // cache with its own activations (overwritten by the target forward pass above) before
        // backward()ing into it. Deterministic given fixed weights (no dropout/randomness in
        // EncoderBlock's own forward), so this reproduces the exact same activations already
        // used for `predicted` above — a small redundant-compute cost, not a correctness gap.
        encode_tokens(context_tokens);
        backward(grad_ctx_embeddings);

        update_weights();
    }

    return {predictor_loss, sigreg_loss};
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

    std::cout << "Saved LeJEPA encoder weights to " << directory << '\n';
}

void LeJEPAEncoder::load(const std::string& directory) {
    std::string config_path = directory + "/" + kConfigFile;
    std::ifstream config_file(config_path, std::ios::binary);
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + config_path);
    }

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
