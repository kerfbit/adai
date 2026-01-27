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
#include "encoder.hpp"

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
    std::cout << "Saving model weights to: " << filename << std::endl;
    // In production, implement full serialization
}

void LLMEncoder::load_weights(const std::string& filename) {
    std::cout << "Loading model weights from: " << filename << std::endl;
    // In production, implement full deserialization
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
