#include "encoder.hpp"
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
#include "FeedForward.hpp"
#include "MultiHeadAttention.hpp"
#include "TokenEmbedding.hpp"

// LLMEncoder Implementation
LLMEncoder::LLMEncoder(int vocab_size, int d_model, int num_layers, int num_heads, int d_ff,
                       int max_seq_length)
    : vocab_size(vocab_size),
      d_model(d_model),
      num_layers(num_layers),
      num_heads(num_heads),
      d_ff(d_ff),
      max_seq_length(max_seq_length),
      requires_grad(false),
      learning_rate(0.001f) {
    // Initialize components
    tokenizer = std::make_unique<BPETokenizer>();
    token_embedding = std::make_unique<TokenEmbedding>(vocab_size, d_model);
    positional_encoding = std::make_unique<PositionalEncoding>(max_seq_length, d_model);
    final_norm = std::make_unique<LayerNorm>(d_model);

    // Initialize encoder blocks
    for (int i = 0; i < num_layers; i++) {
        encoder_blocks.push_back(std::make_unique<EncoderBlock>(d_model, num_heads, d_ff));
    }

    std::cout << "LLM Encoder initialized with:" << std::endl;
    std::cout << "  Vocab size: " << vocab_size << std::endl;
    std::cout << "  Model dimension: " << d_model << std::endl;
    std::cout << "  Number of layers: " << num_layers << std::endl;
    std::cout << "  Number of heads: " << num_heads << std::endl;
    std::cout << "  Feed-forward dimension: " << d_ff << std::endl;
    std::cout << "  Max sequence length: " << max_seq_length << std::endl;
}

Matrix LLMEncoder::encode(const std::string& text) {
    // Tokenize input text
    std::vector<int> token_ids = tokenizer->encode(text, true);

    // Truncate if necessary
    if (token_ids.size() > max_seq_length) {
        token_ids.resize(max_seq_length);
    }

    std::cout << "Encoding " << token_ids.size() << " tokens..." << std::endl;

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

Matrix LLMEncoder::encode_with_mask(const std::vector<int>& token_ids,
                                    const Matrix& attention_mask) {
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
        pooled[j] /= encoded.rows;
    }

    return pooled;
}

std::vector<float> LLMEncoder::get_sentence_embedding_trainable(const std::string& text) {
    Matrix encoded = encode(text);
    std::vector<float> pooled(d_model);

    // Average over sequence length
    for (int j = 0; j < d_model; j++) {
        float sum = 0.0f;
        for (int i = 0; i < encoded.rows; i++) {
            sum += encoded(i, j);
        }
        pooled[j] = sum / encoded.rows;
    }

    return pooled;
}

void LLMEncoder::backward_sentence_embedding(const std::vector<float>& grad_output) {
    if (!requires_grad)
        return;

    // Gradient from mean pooling: distribute gradient equally to all tokens
    Matrix grad_encoded(cached_token_ids.size(), d_model);
    for (int i = 0; i < cached_token_ids.size(); i++) {
        for (int j = 0; j < d_model; j++) {
            grad_encoded(i, j) = grad_output[j] / cached_token_ids.size();
        }
    }

    backward(grad_encoded);
}

void LLMEncoder::backward(const Matrix& grad_output) {
    if (!requires_grad)
        return;

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
void LLMEncoder::set_learning_rate(float lr) {
    learning_rate = lr;
    token_embedding->learning_rate = lr;
    final_norm->learning_rate = lr;
    // Learning rate propagation to encoder blocks happens through their components
}
void LLMEncoder::load_tokenizer_vocab(const std::string& vocab_file) {
    tokenizer->load_vocab(vocab_file);
    std::cout << "Loaded tokenizer vocabulary from: " << vocab_file << std::endl;
}
void LLMEncoder::build_tokenizer(const std::vector<std::string>& corpus, int vocab_size) {
    tokenizer->build_vocab(corpus, vocab_size);
    std::cout << "Built tokenizer with vocabulary size: " << tokenizer->get_vocab_size()
              << std::endl;
}
void LLMEncoder::print_config() const {
    std::cout << "\n=== LLM Encoder Configuration ===" << std::endl;
    std::cout << "Vocabulary size: " << vocab_size << std::endl;
    std::cout << "Model dimension (d_model): " << d_model << std::endl;
    std::cout << "Number of encoder layers: " << num_layers << std::endl;
    std::cout << "Number of attention heads: " << num_heads << std::endl;
    std::cout << "Feed-forward dimension: " << d_ff << std::endl;
    std::cout << "Max sequence length: " << max_seq_length << std::endl;
    std::cout << "=================================\n" << std::endl;
}
int LLMEncoder::get_embedding_dim() const {
    return d_model;
}
void LLMEncoder::save_weights(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write header with dimensions for validation
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_ff), sizeof(int));
    file.write(reinterpret_cast<const char*>(&max_seq_length), sizeof(int));

    file.close();

    // Save component weights to separate files
    std::string base = filename.substr(0, filename.find_last_of('.'));
    token_embedding->save_weights(base + "_token_emb.bin");

    for (size_t i = 0; i < encoder_blocks.size(); ++i) {
        encoder_blocks[i]->save_weights(base + "_encoder_block_" + std::to_string(i) + ".bin");
    }

    final_norm->save_weights(base + "_final_norm.bin");

    std::cout << "Saved Encoder weights to " << filename << std::endl;
}
void LLMEncoder::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Read and verify header
    int loaded_vocab_size, loaded_d_model, loaded_num_layers;
    int loaded_num_heads, loaded_d_ff, loaded_max_seq_length;

    file.read(reinterpret_cast<char*>(&loaded_vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_num_layers), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_max_seq_length), sizeof(int));

    if (loaded_vocab_size != vocab_size || loaded_d_model != d_model ||
        loaded_num_layers != num_layers) {
        throw std::runtime_error("Model architecture mismatch in encoder");
    }

    file.close();

    // Load component weights from separate files
    std::string base = filename.substr(0, filename.find_last_of('.'));
    token_embedding->load_weights(base + "_token_emb.bin");

    for (size_t i = 0; i < encoder_blocks.size(); ++i) {
        encoder_blocks[i]->load_weights(base + "_encoder_block_" + std::to_string(i) + ".bin");
    }

    final_norm->load_weights(base + "_final_norm.bin");

    std::cout << "Loaded Encoder weights from " << filename << std::endl;
}  // Example usage and testing
int main() {
    std::cout << "=== LLM Chatbot Encoder Demo ===" << std::endl;

    // Initialize encoder with smaller dimensions for demo
    int vocab_size = 5000;
    int d_model = 256;
    int num_layers = 4;
    int num_heads = 8;
    int d_ff = 1024;
    int max_seq_length = 128;

    LLMEncoder encoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length);

    // Print configuration
    encoder.print_config();

    // Build tokenizer from trimmed text file
    std::vector<std::string> training_texts;
    std::ifstream file("../texts/pg10_trimmed.txt");

    if (!file.is_open()) {
        std::cerr << "Error: Could not open ../texts/pg10_trimmed.txt\n";
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            training_texts.push_back(line);
        }
    }
    file.close();

    if (training_texts.empty()) {
        std::cerr << "Error: No training data found in file\n";
        return 1;
    }

    std::cout << "Building tokenizer from " << training_texts.size()
              << " lines from pg10_trimmed.txt..." << std::endl;
    encoder.build_tokenizer(training_texts, vocab_size);

    // Encode sample text
    std::string test_text = "Hello, how can I help you today?";
    std::cout << "\nEncoding text: \"" << test_text << "\"" << std::endl;

    Matrix encoded_output = encoder.encode(test_text);

    std::cout << "Encoded output shape: [" << encoded_output.rows << ", " << encoded_output.cols
              << "]" << std::endl;

    // Get sentence embedding
    std::vector<float> sentence_emb = encoder.get_sentence_embedding(test_text);
    std::cout << "Sentence embedding dimension: " << sentence_emb.size() << std::endl;

    std::cout << "\nFirst 10 values of sentence embedding:" << std::endl;
    for (int i = 0; i < std::min(10, (int)sentence_emb.size()); i++) {
        std::cout << "  [" << i << "]: " << sentence_emb[i] << std::endl;
    }

    std::cout << "\n=== Encoder Demo Complete ===" << std::endl;

    return 0;
}
