// @adai-status: stable
// @adai-version: 1.4.1
// @adai-reviewed: 2026-10-26

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "EncoderBlock.hpp"
#include "gpu/GPUParamSync.hpp"
#include "Logger.hpp"
#include "FeedForward.hpp"
#include "MultiHeadAttention.hpp"
#include "TokenEmbedding.hpp"
#include "encoder.hpp"
#include "BinaryIO.hpp"
#include <stdexcept>

// LLMEncoder Implementation
LLMEncoder::LLMEncoder(int vocab_size, int d_model, int num_layers, int num_heads, int d_ff,
                       int max_seq_length)
    : vocab_size(vocab_size),
      d_model(d_model),
      num_layers(num_layers),
      num_heads(num_heads),
      d_ff(d_ff),
      max_seq_length(max_seq_length) {
    // Initialize components
    tokenizer = std::make_unique<BPETokenizer>();
    token_embedding = std::make_unique<TokenEmbedding>(vocab_size, d_model);
    positional_encoding = std::make_unique<PositionalEncoding>(max_seq_length, d_model);
    final_norm = std::make_unique<LayerNorm>(d_model);

    // Initialize encoder blocks
    for (int i = 0; i < num_layers; i++) {
        encoder_blocks.push_back(std::make_unique<EncoderBlock>(d_model, num_heads, d_ff));
    }

    std::cout << "LLM Encoder initialized with:" << '\n';
    std::cout << "  Vocab size: " << vocab_size << '\n';
    std::cout << "  Model dimension: " << d_model << '\n';
    std::cout << "  Number of layers: " << num_layers << '\n';
    std::cout << "  Number of heads: " << num_heads << '\n';
    std::cout << "  Feed-forward dimension: " << d_ff << '\n';
    std::cout << "  Max sequence length: " << max_seq_length << '\n';
}

std::vector<int> LLMEncoder::keep_last_positions(std::vector<int> token_ids) const {
    const auto limit = static_cast<std::size_t>(std::max(0, max_seq_length));
    if (token_ids.size() > limit)
        token_ids.erase(token_ids.begin(),
                        token_ids.end() - static_cast<std::ptrdiff_t>(limit));
    return token_ids;
}

Matrix LLMEncoder::encode(const std::string& text, bool add_special_tokens) {
    // TD-364: without special tokens by default, as EncoderDecoderModel tokenizes encoder input.
    // TD-365: over-long input keeps its last max_seq_length tokens, the training rule; with special
    // tokens, BOS and EOS stay at the ends (resize() used to cut off the EOS it had just added).
    std::vector<int> token_ids;
    if (add_special_tokens && max_seq_length >= 2) {
        std::vector<int> body = tokenizer->encode(text, false);
        const auto room = static_cast<std::size_t>(max_seq_length - 2);
        if (body.size() > room)
            body.erase(body.begin(), body.end() - static_cast<std::ptrdiff_t>(room));
        token_ids.reserve(body.size() + 2);
        token_ids.push_back(tokenizer->get_bos_token_id());
        token_ids.insert(token_ids.end(), body.begin(), body.end());
        token_ids.push_back(tokenizer->get_eos_token_id());
    } else {
        token_ids = keep_last_positions(tokenizer->encode(text, false));
    }

    // TODO: See TD-367 in TECHNICAL_DEBT.md - std::cout on every call (and in the constructor, save/load)
    std::cout << "Encoding " << token_ids.size() << " tokens..." << '\n';

    if (requires_grad) {
        cached_token_ids = token_ids;
    }

    // Get token embeddings
    Matrix embeddings = token_embedding->forward(token_ids);

    // Add positional encoding
    Matrix encoded = positional_encoding->forward(embeddings);

    if (requires_grad) {
        cached_encoder_outputs.clear();
        cached_encoder_outputs.push_back(encoded);
    }

    // Pass through encoder blocks
    for (int i = 0; i < num_layers; i++) {
        encoded = encoder_blocks[i]->forward(encoded);
        if (requires_grad) {
            cached_encoder_outputs.push_back(encoded);
        }
    }

    // Apply final layer normalization
    encoded = final_norm->forward(encoded);

    return encoded;
}

namespace {
// TD-365: positions past max_seq_length have no positional encoding, so refuse them (they used to
// pass with a warning: 44 of 60 positions had no order information on a max_seq_length 16 model).
void check_encoder_length(std::size_t n, int max_seq_length, const char* where) {
    if (max_seq_length > 0 && n > static_cast<std::size_t>(max_seq_length)) {
        throw std::invalid_argument(std::string(where) + ": " + std::to_string(n) +
                                    " tokens exceed max_seq_length " +
                                    std::to_string(max_seq_length) +
                                    " (truncate with keep_last_positions())");
    }
}
}  // namespace

Matrix LLMEncoder::encode_with_mask(const std::vector<int>& token_ids,
                                    const Matrix& attention_mask) {
    check_encoder_length(token_ids.size(), max_seq_length, "LLMEncoder::encode_with_mask");
    if (requires_grad) {
        cached_token_ids = token_ids;
    }

    // Get token embeddings
    Matrix embeddings = token_embedding->forward(token_ids);

    // Add positional encoding
    Matrix encoded = positional_encoding->forward(embeddings);

    if (requires_grad) {
        cached_encoder_outputs.clear();
        cached_encoder_outputs.push_back(encoded);
    }

    // Pass through encoder blocks with mask
    for (int i = 0; i < num_layers; i++) {
        encoded = encoder_blocks[i]->forward(encoded, &attention_mask);
        if (requires_grad) {
            cached_encoder_outputs.push_back(encoded);
        }
    }

    // Apply final layer normalization
    encoded = final_norm->forward(encoded);

    return encoded;
}

std::vector<float> LLMEncoder::get_sentence_embedding(const std::string& text) {
    Matrix encoded = encode(text);
    std::vector<float> pooled(d_model, 0.0f);

    // Average over sequence length
    for (int j = 0; j < d_model; j++) {
        for (int i = 0; i < encoded.rows; i++) {
            pooled[j] += encoded(i, j);
        }
        pooled[j] /= static_cast<float>(encoded.rows);
    }

    return pooled;
}

// TODO: See TD-366 in TECHNICAL_DEBT.md - duplicate of get_sentence_embedding(); neither has a production caller
std::vector<float> LLMEncoder::get_sentence_embedding_trainable(const std::string& text) {
    Matrix encoded = encode(text);
    std::vector<float> pooled(d_model);

    // Average over sequence length
    for (int j = 0; j < d_model; j++) {
        float sum = 0.0f;
        for (int i = 0; i < encoded.rows; i++) {
            sum += encoded(i, j);
        }
        pooled[j] = sum / static_cast<float>(encoded.rows);
    }

    return pooled;
}

void LLMEncoder::backward_sentence_embedding(const std::vector<float>& grad_output) {
    if (!requires_grad) {
        return;
    }

    // Gradient from mean pooling: distribute gradient equally to all tokens
    Matrix grad_encoded(static_cast<int>(cached_token_ids.size()), d_model);
    for (int i = 0; i < static_cast<int>(cached_token_ids.size()); i++) {
        for (int j = 0; j < d_model; j++) {
            grad_encoded(i, j) = grad_output[j] / static_cast<float>(cached_token_ids.size());
        }
    }

    backward(grad_encoded);
}

void LLMEncoder::backward(const Matrix& grad_output) {
    if (!requires_grad) {
        return;
    }

    // Gradient through final layer norm
    Matrix grad = final_norm->backward(grad_output);

    // Backward through encoder blocks (in reverse order)
    for (int i = num_layers - 1; i >= 0; i--) {
        grad = encoder_blocks[i]->backward(grad);
    }

    // Gradient through positional encoding (just passes through)
    // positional encoding adds fixed values, no learnable params

    // Gradient through token embeddings
    token_embedding->backward(cached_token_ids, grad);
}

void LLMEncoder::zero_grad() {
    token_embedding->zero_grad();
    for (auto& block : encoder_blocks) {
        block->zero_grad();
    }
    final_norm->zero_grad();
}

void LLMEncoder::set_requires_grad(bool requires_grad) {
    this->requires_grad = requires_grad;
}

// TD-093 (fixed): the "propagation happens through their components" comment
// was aspirational, not actual — nothing here ever touched encoder_blocks, and
// EncoderBlock has no set_learning_rate() of its own to delegate to (only a
// public learning_rate member that its own update_weights() re-syncs to
// sub-components right before applying gradients). Every encoder block was
// permanently stuck at its constructor default (0.001f) no matter what this
// was called with. See Decoder.cpp's LLMDecoder::set_learning_rate() for the
// identical defect on the decoder side and why it's currently masked in the
// shipped trainer (Optimizer path bypasses these members once registered).
void LLMEncoder::set_learning_rate(float lr) {
    learning_rate = lr;
    token_embedding->learning_rate = lr;
    for (auto& block : encoder_blocks) {
        block->learning_rate = lr;
    }
    final_norm->learning_rate = lr;
}

void LLMEncoder::load_tokenizer_vocab(const std::string& vocab_file) {
    tokenizer->load_vocab(vocab_file);
    std::cout << "Loaded tokenizer vocabulary from: " << vocab_file << '\n';
}

void LLMEncoder::build_tokenizer(const std::vector<std::string>& corpus, int vocab_size) {
    tokenizer->build_vocab(corpus, vocab_size);
    std::cout << "Built tokenizer with vocabulary size: " << tokenizer->get_vocab_size() << '\n';
}

void LLMEncoder::print_config() const {
    std::cout << "\n=== LLM Encoder Configuration ===" << '\n';
    std::cout << "Vocabulary size: " << vocab_size << '\n';
    std::cout << "Model dimension (d_model): " << d_model << '\n';
    std::cout << "Number of encoder layers: " << num_layers << '\n';
    std::cout << "Number of attention heads: " << num_heads << '\n';
    std::cout << "Feed-forward dimension: " << d_ff << '\n';
    std::cout << "Max sequence length: " << max_seq_length << '\n';
    std::cout << "=================================\n" << '\n';
}

int LLMEncoder::get_embedding_dim() const {
    return d_model;
}

void LLMEncoder::save_weights(const std::string& filename) {
    // Write header with dimensions for validation (TD-423: checked writes)
    adai::binio::Writer file(filename);
    for (int field : {vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length}) {
        file.value(field, "encoder header");
    }
    file.close();

    // Save component weights to separate files.
    // TD-538: the token embedding and final norm use encoder-specific names. EncoderDecoderModel
    // saves the encoder as <path>.encoder and the decoder as <path>.decoder, so both strip to
    // the same base, and the old shared "_token_emb.bin"/"_final_norm.bin" names were then
    // overwritten by the decoder's: every checkpoint lost the encoder's embedding and final norm.
    std::string base = filename.substr(0, filename.find_last_of('.'));
    token_embedding->save_weights(base + "_encoder_token_emb.bin");

    for (size_t i = 0; i < encoder_blocks.size(); ++i) {
        encoder_blocks[i]->save_weights(base + "_encoder_block_" + std::to_string(i) + ".bin");
    }

    final_norm->save_weights(base + "_encoder_final_norm.bin");

    std::cout << "Saved Encoder weights to " << filename << '\n';
}

void LLMEncoder::load_weights(const std::string& filename) {
    // Read and verify header (TD-423/TD-367: checked reads)
    // TODO: See TD-367 in TECHNICAL_DEBT.md - max_seq_length read but not compared
    adai::binio::Reader file(filename);
    const int loaded_vocab_size = file.value<int>("vocab_size");
    const int loaded_d_model = file.value<int>("d_model");
    const int loaded_num_layers = file.value<int>("num_layers");
    const int loaded_num_heads = file.value<int>("num_heads");
    const int loaded_d_ff = file.value<int>("d_ff");
    file.value<int>("max_seq_length");
    file.expect_end();

    if (loaded_vocab_size != vocab_size || loaded_d_model != d_model ||
        loaded_num_layers != num_layers || loaded_num_heads != num_heads || loaded_d_ff != d_ff) {
        throw std::runtime_error(
            "Encoder architecture mismatch: saved (vocab=" + std::to_string(loaded_vocab_size) +
            ", d_model=" + std::to_string(loaded_d_model) + ", layers=" +
            std::to_string(loaded_num_layers) + ", heads=" + std::to_string(loaded_num_heads) +
            ", d_ff=" + std::to_string(loaded_d_ff) +
            ") vs current (vocab=" + std::to_string(vocab_size) +
            ", d_model=" + std::to_string(d_model) + ", layers=" + std::to_string(num_layers) +
            ", heads=" + std::to_string(num_heads) + ", d_ff=" + std::to_string(d_ff) + ")");
    }


    // Load component weights from separate files. TD-538: files saved before the
    // encoder-specific names have only the shared "_token_emb.bin"/"_final_norm.bin", which in an
    // EncoderDecoderModel checkpoint hold the decoder's weights (the encoder's were overwritten).
    // Fall back to them, as the old code always loaded, so those checkpoints load unchanged.
    std::string base = filename.substr(0, filename.find_last_of('.'));
    auto component_path = [&base](const std::string& name) {
        const std::string current = base + "_encoder_" + name;
        if (std::ifstream(current).good()) {
            return current;
        }
        adai::Logger::warn("Encoder checkpoint has no {}; loading the legacy shared file {} "
                           "(pre-TD-538 checkpoints hold the decoder's copy there)",
                           current, base + "_" + name);
        return base + "_" + name;
    };
    token_embedding->load_weights(component_path("token_emb.bin"));

    for (size_t i = 0; i < encoder_blocks.size(); ++i) {
        encoder_blocks[i]->load_weights(base + "_encoder_block_" + std::to_string(i) + ".bin");
    }

    final_norm->load_weights(component_path("final_norm.bin"));

    std::cout << "Loaded Encoder weights from " << filename << '\n';
}

void LLMEncoder::register_parameters_with_optimizer(Optimizer& optimizer) {
    // Register token embedding parameters
    token_embedding->set_optimizer(&optimizer);

    // Register all encoder block parameters
    for (auto& block : encoder_blocks) {
        block->register_parameters_with_optimizer(optimizer);
    }

    // Register final layer norm parameters
    final_norm->set_optimizer(&optimizer);
}

#ifdef ADAI_ENABLE_GPU
const adai::gpu::GPUMatrix* LLMEncoder::gpu_positional_table() {
    // The canonical sinusoidal table (PositionalEncoding), uploaded once; it never changes.
    if (!gpu_pe_) {
        const Matrix& pe = positional_encoding->get_encoding();
        gpu_pe_ = std::make_unique<adai::gpu::GPUMatrix>(pe.rows, pe.cols);
        std::vector<float> flat;
        flat.reserve(static_cast<size_t>(pe.rows) * static_cast<size_t>(pe.cols));
        for (const auto& row : pe.data)
            flat.insert(flat.end(), row.begin(), row.end());
        gpu_pe_->upload(flat.data(), pe.rows * pe.cols);
    }
    return gpu_pe_.get();
}

void LLMEncoder::gpu_upload_weights() {
    token_embedding->gpu_upload_weights();  // TD-547
    for (auto& block : encoder_blocks)
        block->gpu_upload_weights();
    final_norm->gpu_upload_weights();
}

void LLMEncoder::gpu_collect_parameters(std::vector<adai::gpu::GPUParamRef>& out, int /*layer*/) {
    token_embedding->gpu_collect_parameters(out, -1);  // TD-547
    for (size_t l = 0; l < encoder_blocks.size(); ++l)
        encoder_blocks[l]->gpu_collect_parameters(out, static_cast<int>(l));
    final_norm->gpu_collect_parameters(out, -1);
}

void LLMEncoder::gpu_download_weights() {
    token_embedding->gpu_download_weights();
    for (auto& block : encoder_blocks)
        block->gpu_download_weights();
    final_norm->gpu_download_weights();
}

void LLMEncoder::gpu_download_grads() {
    token_embedding->gpu_download_grads();
    for (auto& block : encoder_blocks)
        block->gpu_download_grads();
    final_norm->gpu_download_grads();
}

void LLMEncoder::gpu_zero_grads() {
    token_embedding->gpu_zero_grads();
    for (auto& block : encoder_blocks)
        block->gpu_zero_grads();
    final_norm->gpu_zero_grads();
}

adai::gpu::GPUMatrix LLMEncoder::gpu_encode(const std::vector<int>& token_ids) {
    check_encoder_length(token_ids.size(), max_seq_length, "LLMEncoder::gpu_encode");
    const int seq = static_cast<int>(token_ids.size());

    if (requires_grad) {
        cached_token_ids = token_ids;
    }

    // TD-547: embedding + positional encoding on the device; only the token IDs cross.
    const adai::gpu::GPUMatrix* pe = gpu_positional_table();
    (void)seq;

    // All-ones encoder mask: no masking needed (skip masked_fill)
    adai::gpu::GPUMatrix x = token_embedding->gpu_forward(token_ids, pe, pe->rows);
    for (auto& block : encoder_blocks)
        x = block->gpu_forward(x);  // no self-mask for encoder

    return final_norm->gpu_forward(x);
}

void LLMEncoder::gpu_backward(const adai::gpu::GPUMatrix& dout) {
    if (!requires_grad) {
        return;
    }

    adai::gpu::GPUMatrix d = final_norm->gpu_backward(dout);
    for (int i = num_layers - 1; i >= 0; --i) {
        d = encoder_blocks[i]->gpu_backward(d);
    }

    // TD-547: into the device embedding gradient (the gradient used to be downloaded and
    // accumulated into the host table every sample).
    token_embedding->gpu_backward(d);
}
#endif
