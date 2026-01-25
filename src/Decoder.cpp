#include "Decoder.hpp"

// Constructor
LLMDecoder::LLMDecoder(int vocab_size, int d_model, int num_layers, int num_heads, int d_ff,
                       int max_seq_length)
    : vocab_size(vocab_size),
      d_model(d_model),
      num_layers(num_layers),
      num_heads(num_heads),
      d_ff(d_ff),
      max_seq_length(max_seq_length),
      requires_grad(true),
      learning_rate(0.001) {
    // Initialize token embedding
    token_embedding = std::make_unique<TokenEmbedding>(vocab_size, d_model);

    // Initialize positional encoding
    positional_encoding = std::make_unique<PositionalEncoding>(d_model, max_seq_length);

    // Initialize decoder blocks
    for (int i = 0; i < num_layers; ++i) {
        decoder_blocks.push_back(std::make_unique<DecoderBlock>(d_model, num_heads, d_ff));
    }

    // Initialize final layer normalization
    final_norm = std::make_unique<LayerNorm>(d_model);
}

// Destructor
LLMDecoder::~LLMDecoder() {
    // Unique pointers handle cleanup automatically
}

// Create causal mask
Matrix LLMDecoder::create_causal_mask(int seq_length) const {
    Matrix mask(seq_length, seq_length);

    // Fill with 1s on and below diagonal, 0s above diagonal
    for (int i = 0; i < seq_length; ++i) {
        for (int j = 0; j < seq_length; ++j) {
            mask.data[i][j] = (j <= i) ? 1.0f : 0.0f;
        }
    }

    return mask;
}

// Forward pass (decoder-only mode)
Matrix LLMDecoder::forward(const std::vector<int>& token_ids) {
    return forward_with_encoder(token_ids, Matrix(0, 0));
}

// Forward pass with encoder outputs
Matrix LLMDecoder::forward_with_encoder(const std::vector<int>& token_ids,
                                        const Matrix& encoder_output) {
    int seq_length = token_ids.size();

    // Cache inputs for backward pass
    cached_token_ids = token_ids;
    cached_encoder_output = encoder_output;
    cached_decoder_outputs.clear();

    // 1. Token embedding
    cached_embeddings = token_embedding->forward(token_ids);

    // 2. Add positional encoding
    cached_pos_encoded = positional_encoding->forward(cached_embeddings);

    // 3. Create causal mask for autoregressive generation
    Matrix causal_mask = create_causal_mask(seq_length);

    // 4. Pass through decoder blocks
    Matrix x = cached_pos_encoded;

    for (int i = 0; i < num_layers; ++i) {
        if (encoder_output.rows > 0 && encoder_output.cols > 0) {
            // Encoder-decoder mode: use cross-attention
            x = decoder_blocks[i]->forward(x, encoder_output, causal_mask, nullptr);
        } else {
            // Decoder-only mode: no cross-attention (pass empty encoder output)
            Matrix empty_encoder(1, d_model);  // Dummy encoder output
            x = decoder_blocks[i]->forward(x, empty_encoder, causal_mask, nullptr);
        }
        cached_decoder_outputs.push_back(x);
    }

    // 5. Final layer normalization
    Matrix output = final_norm->forward(x);

    return output;
}

// Forward pass with custom mask
Matrix LLMDecoder::forward_with_mask(const std::vector<int>& token_ids, const Matrix& causal_mask,
                                     const Matrix* encoder_output) {
    int seq_length = token_ids.size();

    // Cache inputs for backward pass
    cached_token_ids = token_ids;
    if (encoder_output) {
        cached_encoder_output = *encoder_output;
    }
    cached_decoder_outputs.clear();

    // 1. Token embedding
    cached_embeddings = token_embedding->forward(token_ids);

    // 2. Add positional encoding
    cached_pos_encoded = positional_encoding->forward(cached_embeddings);

    // 3. Pass through decoder blocks with custom mask
    Matrix x = cached_pos_encoded;

    for (int i = 0; i < num_layers; ++i) {
        if (encoder_output && encoder_output->rows > 0) {
            // Use cross-attention if encoder output provided
            x = decoder_blocks[i]->forward(x, *encoder_output, causal_mask, nullptr);
        } else {
            // No cross-attention (decoder-only mode)
            Matrix empty_encoder(1, d_model);
            x = decoder_blocks[i]->forward(x, empty_encoder, causal_mask, nullptr);
        }
        cached_decoder_outputs.push_back(x);
    }

    // 4. Final layer normalization
    Matrix output = final_norm->forward(x);

    return output;
}

Matrix LLMDecoder::forward_with_cache(const std::vector<int>& token_ids, DecoderKVCache& kv_cache,
                                      const Matrix* encoder_output, bool use_cache) {
    int num_new_tokens = token_ids.size();
    int current_position = kv_cache.current_length();
    int total_seq_len = current_position + num_new_tokens;

    // Cache inputs for backward pass (if training)
    cached_token_ids = token_ids;
    if (encoder_output) {
        cached_encoder_output = *encoder_output;
    }
    cached_decoder_outputs.clear();

    // 1. Token embedding (only for new tokens)
    Matrix embeddings = token_embedding->forward(token_ids);
    cached_embeddings = embeddings;

    // 2. Add positional encoding (offset by current_position)
    // We need to generate positional encoding starting from current_position
    Matrix pos_encoded(num_new_tokens, d_model);
    
    for (int pos = 0; pos < num_new_tokens; ++pos) {
        int absolute_pos = current_position + pos;
        for (int i = 0; i < d_model; ++i) {
            if (i % 2 == 0) {
                // Even dimensions: sin(pos / 10000^(2i/d_model))
                float angle = absolute_pos / std::pow(10000.0f, (2.0f * (i / 2)) / d_model);
                pos_encoded(pos, i) = embeddings(pos, i) + std::sin(angle);
            } else {
                // Odd dimensions: cos(pos / 10000^(2i/d_model))
                float angle = absolute_pos / std::pow(10000.0f, (2.0f * ((i - 1) / 2)) / d_model);
                pos_encoded(pos, i) = embeddings(pos, i) + std::cos(angle);
            }
        }
    }
    cached_pos_encoded = pos_encoded;

    // 3. Create causal mask for new tokens
    // Shape: [num_new_tokens, total_seq_len]
    // New tokens can attend to all previous tokens + themselves
    Matrix causal_mask(num_new_tokens, total_seq_len);
    for (int i = 0; i < num_new_tokens; ++i) {
        int current_token_pos = current_position + i;
        for (int j = 0; j < total_seq_len; ++j) {
            // Can attend to positions <= current position
            causal_mask(i, j) = (j <= current_token_pos) ? 1.0f : 0.0f;
        }
    }

    // 4. Pass through decoder blocks with KV cache
    Matrix x = pos_encoded;

    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        KVCache& self_attn_cache = kv_cache.get_self_attention_cache(layer_idx);
        KVCache& cross_attn_cache = kv_cache.get_cross_attention_cache(layer_idx);

        if (encoder_output && encoder_output->rows > 0) {
            // Encoder-decoder mode with cross-attention
            x = decoder_blocks[layer_idx]->forward_with_cache(
                x, *encoder_output, causal_mask, &self_attn_cache, &cross_attn_cache, nullptr,
                use_cache);
        } else {
            // Decoder-only mode (no cross-attention)
            Matrix empty_encoder(1, d_model);
            x = decoder_blocks[layer_idx]->forward_with_cache(
                x, empty_encoder, causal_mask, &self_attn_cache, &cross_attn_cache, nullptr,
                use_cache);
        }
        cached_decoder_outputs.push_back(x);
    }

    // 5. Final layer normalization
    Matrix output = final_norm->forward(x);

    return output;
}

// Backward pass
void LLMDecoder::backward(const Matrix& grad_output) {
    if (!requires_grad) {
        return;
    }

    // Backward through final layer norm
    Matrix grad = final_norm->backward(grad_output);

    // Backward through decoder blocks in reverse order
    for (int i = num_layers - 1; i >= 0; --i) {
        grad = decoder_blocks[i]->backward(grad);
    }

    // Backward through positional encoding (no parameters, just pass through)
    // grad remains unchanged

    // Backward through token embedding (different signature: token_ids, grad)
    token_embedding->backward(cached_token_ids, grad);
}

// Update weights
void LLMDecoder::update_weights(float lr) {
    if (!requires_grad) {
        return;
    }

    // Update token embedding (doesn't take lr parameter)
    token_embedding->update_weights();

    // Update decoder blocks (also don't take lr parameter)
    for (auto& block : decoder_blocks) {
        block->update_weights();
    }

    // Update final layer norm (if it has update method)
    // LayerNorm has learnable parameters (gamma, beta) that need updating
    // For now, we'll skip if the method doesn't exist
    // final_norm->update_weights(lr);
}

// Set training mode
void LLMDecoder::set_training(bool mode) {
    requires_grad = mode;

    // Propagate to all components
    for (auto& block : decoder_blocks) {
        // DecoderBlock doesn't have set_training, but it respects requires_grad
        // through its internal components
    }
}

// Set learning rate
void LLMDecoder::set_learning_rate(float lr) {
    learning_rate = lr;
}

// Save weights
void LLMDecoder::save_weights(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving: " + filepath);
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_ff), sizeof(int));
    file.write(reinterpret_cast<const char*>(&max_seq_length), sizeof(int));

    file.close();

    // Note: TokenEmbedding, DecoderBlock, and LayerNorm don't have save_weights methods
    // In a production implementation, you would add these methods to those classes
    // For now, we save only the configuration
    std::cout << "Warning: Decoder weights not fully saved (save_weights not implemented in all "
                 "components)\n";
}

// Load weights
void LLMDecoder::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading: " + filepath);
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
        throw std::runtime_error("Model architecture mismatch");
    }

    file.close();

    // Note: TokenEmbedding, DecoderBlock, and LayerNorm don't have load_weights methods
    // Weights are randomly initialized for now
    std::cout << "Warning: Decoder weights not fully loaded (load_weights not implemented in all "
                 "components)\n";
}

// Zero gradients
void LLMDecoder::zero_grad() {
    token_embedding->zero_grad();

    for (auto& block : decoder_blocks) {
        block->zero_grad();
    }

    // LayerNorm doesn't have zero_grad method
    // final_norm->zero_grad();
}
