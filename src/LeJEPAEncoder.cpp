// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-15

#include "LeJEPAEncoder.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
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
