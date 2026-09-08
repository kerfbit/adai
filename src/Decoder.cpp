// @adai-status: beta        (capped by TD-050 — see TECHNICAL_DEBT.md)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-08

#include "Decoder.hpp"
#include "Logger.hpp"
using adai::Logger;

// Constructor
LLMDecoder::LLMDecoder(int vocab_size, int d_model, int num_layers, int num_heads, int d_ff,
                       int max_seq_length)
    : vocab_size(vocab_size),
      d_model(d_model),
      num_layers(num_layers),
      num_heads(num_heads),
      d_ff(d_ff),
      max_seq_length(max_seq_length) {
    // Initialize token embedding
    token_embedding = std::make_unique<TokenEmbedding>(vocab_size, d_model);

    // Initialize positional encoding
    positional_encoding = std::make_unique<PositionalEncoding>(max_seq_length, d_model);

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
Matrix LLMDecoder::create_causal_mask(int seq_length) {
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
    int seq_length = static_cast<int>(token_ids.size());

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
    int seq_length = static_cast<int>(token_ids.size());

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
    int num_new_tokens = static_cast<int>(token_ids.size());
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
                float angle =
                    static_cast<float>(absolute_pos) /
                    std::pow(10000.0f, (2.0f * static_cast<float>(static_cast<float>(i) / 2)) /
                                           static_cast<float>(d_model));
                pos_encoded(pos, i) = embeddings(pos, i) + std::sin(angle);
            } else {
                // Odd dimensions: cos(pos / 10000^(2i/d_model))
                float angle =
                    static_cast<float>(absolute_pos) /
                    std::pow(10000.0f, static_cast<float>(i - 1) / static_cast<float>(d_model));
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
            x = decoder_blocks[layer_idx]->forward_with_cache(x, *encoder_output, causal_mask,
                                                              &self_attn_cache, &cross_attn_cache,
                                                              nullptr, use_cache);
        } else {
            // Decoder-only mode (no cross-attention)
            Matrix empty_encoder(1, d_model);
            x = decoder_blocks[layer_idx]->forward_with_cache(x, empty_encoder, causal_mask,
                                                              &self_attn_cache, &cross_attn_cache,
                                                              nullptr, use_cache);
        }
        cached_decoder_outputs.push_back(x);
    }

    // 5. Final layer normalization
    Matrix output = final_norm->forward(x);

    return output;
}

// Backward pass
void LLMDecoder::backward(const Matrix& grad_output) {
    Matrix unused_grad_encoder_output;
    backward(grad_output, unused_grad_encoder_output);
}

void LLMDecoder::backward(const Matrix& grad_output, Matrix& grad_encoder_output) {
    if (!requires_grad) {
        return;
    }

    // Backward through final layer norm
    Matrix grad = final_norm->backward(grad_output);

    // Backward through decoder blocks in reverse order, summing each block's
    // encoder-side gradient contribution — the same encoder_output feeds every
    // block's cross-attention.
    bool have_encoder_grad = false;
    for (int i = num_layers - 1; i >= 0; --i) {
        Matrix grad_enc_i;
        grad = decoder_blocks[i]->backward(grad, grad_enc_i);
        if (!have_encoder_grad) {
            grad_encoder_output = grad_enc_i;
            have_encoder_grad = true;
        } else {
            for (int r = 0; r < grad_encoder_output.rows; ++r) {
                for (int c = 0; c < grad_encoder_output.cols; ++c) {
                    grad_encoder_output(r, c) += grad_enc_i(r, c);
                }
            }
        }
    }

    // Backward through positional encoding (no parameters, just pass through)
    // grad remains unchanged

    // Backward through token embedding (different signature: token_ids, grad)
    token_embedding->backward(cached_token_ids, grad);
}

// Update weights
void LLMDecoder::update_weights(float learning_rate) {
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

    // Save component weights to separate files
    std::string base = filepath.substr(0, filepath.find_last_of('.'));
    token_embedding->save_weights(base + "_token_emb.bin");

    for (size_t i = 0; i < decoder_blocks.size(); ++i) {
        decoder_blocks[i]->save(base + "_decoder_block_" + std::to_string(i) + ".bin");
    }

    final_norm->save_weights(base + "_final_norm.bin");

    Logger::info("Saved Decoder weights to {}", filepath);
}

// Load weights
void LLMDecoder::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading: " + filepath);
    }

    // Read and verify header
    int loaded_vocab_size = 0, loaded_d_model = 0, loaded_num_layers = 0;
    int loaded_num_heads = 0, loaded_d_ff = 0, loaded_max_seq_length = 0;

    file.read(reinterpret_cast<char*>(&loaded_vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_num_layers), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_max_seq_length), sizeof(int));

    if (loaded_vocab_size != vocab_size || loaded_d_model != d_model ||
        loaded_num_layers != num_layers || loaded_num_heads != num_heads || loaded_d_ff != d_ff) {
        throw std::runtime_error(
            "Decoder architecture mismatch: saved (vocab=" + std::to_string(loaded_vocab_size) +
            ", d_model=" + std::to_string(loaded_d_model) + ", layers=" +
            std::to_string(loaded_num_layers) + ", heads=" + std::to_string(loaded_num_heads) +
            ", d_ff=" + std::to_string(loaded_d_ff) +
            ") vs current (vocab=" + std::to_string(vocab_size) +
            ", d_model=" + std::to_string(d_model) + ", layers=" + std::to_string(num_layers) +
            ", heads=" + std::to_string(num_heads) + ", d_ff=" + std::to_string(d_ff) + ")");
    }

    file.close();

    // Load component weights from separate files
    std::string base = filepath.substr(0, filepath.find_last_of('.'));
    token_embedding->load_weights(base + "_token_emb.bin");

    for (size_t i = 0; i < decoder_blocks.size(); ++i) {
        decoder_blocks[i]->load(base + "_decoder_block_" + std::to_string(i) + ".bin");
    }

    final_norm->load_weights(base + "_final_norm.bin");

    Logger::info("Loaded Decoder weights from {}", filepath);
}

// Zero gradients
void LLMDecoder::zero_grad() {
    token_embedding->zero_grad();

    for (auto& block : decoder_blocks) {
        block->zero_grad();
    }

    final_norm->zero_grad();
}

void LLMDecoder::register_parameters_with_optimizer(Optimizer& optimizer) {
    // Register token embedding parameters
    token_embedding->set_optimizer(&optimizer);

    // Register all decoder block parameters
    for (auto& block : decoder_blocks) {
        block->register_parameters_with_optimizer(optimizer);
    }

    // Register final layer norm parameters
    final_norm->set_optimizer(&optimizer);
}

#ifdef ADAI_ENABLE_GPU
void LLMDecoder::gpu_upload_weights() {
    for (auto& block : decoder_blocks)
        block->gpu_upload_weights();
    final_norm->gpu_upload_weights();
}

void LLMDecoder::gpu_download_grads() {
    for (auto& block : decoder_blocks)
        block->gpu_download_grads();
    final_norm->gpu_download_grads();
}

void LLMDecoder::gpu_zero_grads() {
    for (auto& block : decoder_blocks)
        block->gpu_zero_grads();
    final_norm->gpu_zero_grads();
}

adai::gpu::GPUMatrix LLMDecoder::gpu_decode(const std::vector<int>& token_ids,
                                            const adai::gpu::GPUMatrix& encoder_out) {
    const int tgt = static_cast<int>(token_ids.size());

    if (requires_grad) {
        cached_token_ids = token_ids;
    }

    // Embedding + positional encoding on CPU (fast)
    Matrix embeddings = token_embedding->forward(token_ids);
    Matrix pos_encoded = positional_encoding->forward(embeddings);

    adai::gpu::GPUMatrix x(tgt, d_model);
    {
        std::vector<float> flat;
        flat.reserve(tgt * d_model);
        for (const auto& row : pos_encoded.data)
            for (float v : row)
                flat.push_back(v);
        x.upload(flat.data(), tgt * d_model);
    }

    // Build causal mask on GPU: mask[i][j] = 0 where j > i (mask out future)
    // We store values as 0.0f for masked positions so masked_fill_inplace works.
    adai::gpu::GPUMatrix causal_mask(tgt, tgt);
    {
        std::vector<float> cm(tgt * tgt);
        for (int i = 0; i < tgt; ++i)
            for (int j = 0; j < tgt; ++j)
                cm[i * tgt + j] = (j <= i) ? 1.0f : 0.0f;
        causal_mask.upload(cm.data(), tgt * tgt);
    }

    for (auto& block : decoder_blocks)
        x = block->gpu_forward(x, encoder_out, &causal_mask);

    return final_norm->gpu_forward(x);
}

std::pair<adai::gpu::GPUMatrix, adai::gpu::GPUMatrix> LLMDecoder::gpu_backward(
    const adai::gpu::GPUMatrix& dout) {
    adai::gpu::GPUMatrix d = final_norm->gpu_backward(dout);

    // Backward through decoder blocks in reverse, summing each block's
    // encoder-side gradient contribution (GPUMatrix is move-only with no
    // default-constructible zero, so accumulate via std::optional).
    std::optional<adai::gpu::GPUMatrix> grad_encoder_sum;
    for (int i = static_cast<int>(decoder_blocks.size()) - 1; i >= 0; --i) {
        auto [d_input, d_enc] = decoder_blocks[i]->gpu_backward(d);
        d = std::move(d_input);
        if (!grad_encoder_sum) {
            grad_encoder_sum.emplace(std::move(d_enc));
        } else {
            *grad_encoder_sum = *grad_encoder_sum + d_enc;
        }
    }

    // TokenEmbedding has no GPU backward — download and finish on host, reusing
    // the existing, already-correct CPU path (mirrors LLMEncoder::gpu_backward()).
    if (requires_grad) {
        Matrix grad_host = Matrix::from_gpu(d);
        token_embedding->backward(cached_token_ids, grad_host);
    }

    return {std::move(d), std::move(*grad_encoder_sum)};
}
#endif
