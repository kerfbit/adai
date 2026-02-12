#include "EncoderDecoderModel.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "Optimizer.hpp"

// Constructor
EncoderDecoderModel::EncoderDecoderModel(int vocab_size, int d_model, int encoder_layers,
                                         int decoder_layers, int num_heads, int d_ff,
                                         int max_seq_length)
    : vocab_size(vocab_size),
      d_model(d_model),
      encoder_layers(encoder_layers),
      decoder_layers(decoder_layers),
      num_heads(num_heads),
      d_ff(d_ff),
      max_seq_length(max_seq_length),
      requires_grad(true),
      learning_rate(0.001) {
    // Initialize tokenizer (will be set externally or use default)
    tokenizer = std::make_unique<BPETokenizer>();

    // Initialize encoder
    encoder = std::make_unique<LLMEncoder>(vocab_size, d_model, encoder_layers, num_heads, d_ff,
                                           max_seq_length);

    // Initialize decoder
    decoder = std::make_unique<LLMDecoder>(vocab_size, d_model, decoder_layers, num_heads, d_ff,
                                           max_seq_length);

    // Initialize language model head
    lm_head = std::make_unique<LanguageModelHead>(d_model, vocab_size);

    // Initialize text generator with default config
    TextGenerator::GenerationConfig gen_config;
    gen_config.max_length = max_seq_length;
    gen_config.bos_token_id = 1;  // Default <bos>
    gen_config.eos_token_id = 2;  // Default <eos>
    gen_config.pad_token_id = 0;  // Default <pad>

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

// Compute cross-entropy loss
float EncoderDecoderModel::compute_loss(const Matrix& logits,
                                        const std::vector<int>& target_tokens) {
    int seq_length = logits.rows;
    float total_loss = 0.0f;

    for (int t = 0; t < seq_length && t < static_cast<int>(target_tokens.size()); ++t) {
        // Softmax for current timestep
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

    return total_loss / seq_length;
}

// Compute loss gradient
Matrix EncoderDecoderModel::compute_loss_gradient(const Matrix& logits,
                                                  const std::vector<int>& target_tokens) {
    int seq_length = logits.rows;
    Matrix grad(seq_length, vocab_size);

    for (int t = 0; t < seq_length && t < static_cast<int>(target_tokens.size()); ++t) {
        // Softmax
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
            grad.data[t][v] /= seq_length;
        }
    }

    return grad;
}

// Generate response
std::string EncoderDecoderModel::generate_response(const std::string& input_text, int max_length) {
    // Encode input
    std::vector<int> input_tokens = tokenizer->encode(input_text);
    int input_len = input_tokens.size();
    Matrix encoder_mask(input_len, input_len);
    for (int i = 0; i < input_len; ++i) {
        for (int j = 0; j < input_len; ++j) {
            encoder_mask.data[i][j] = 1.0f;  // No masking for encoder
        }
    }
    cached_encoder_output = encoder->encode_with_mask(input_tokens, encoder_mask);

    // Initialize KV cache for efficient generation
    DecoderKVCache kv_cache(decoder_layers);
    size_t processed_length = 0;

    // Create model forward function with KV caching
    auto model_fn = [this, &kv_cache, &processed_length](const std::vector<int>& tokens) -> Matrix {
        size_t current_length = tokens.size();
        
        // Determine which tokens are new (not yet processed)
        std::vector<int> new_tokens(tokens.begin() + processed_length, tokens.end());
        
        // Process only new tokens with cache
        Matrix decoder_out = decoder->forward_with_cache(new_tokens, kv_cache, 
                                                         &cached_encoder_output, true);
        
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

    return response;
}

// Generate with specific strategy
std::string EncoderDecoderModel::generate_response_with_strategy(const std::string& input_text,
                                                                 int max_length,
                                                                 const std::string& strategy,
                                                                 float temperature, int top_k,
                                                                 float top_p, int num_beams) {
    // Encode input
    std::vector<int> input_tokens = tokenizer->encode(input_text);
    int input_len = input_tokens.size();
    Matrix encoder_mask(input_len, input_len);
    for (int i = 0; i < input_len; ++i) {
        for (int j = 0; j < input_len; ++j) {
            encoder_mask.data[i][j] = 1.0f;
        }
    }
    cached_encoder_output = encoder->encode_with_mask(input_tokens, encoder_mask);

    // Initialize KV cache for efficient generation
    DecoderKVCache kv_cache(decoder_layers);
    size_t processed_length = 0;

    // Create model forward function with KV caching
    auto model_fn = [this, &kv_cache, &processed_length](const std::vector<int>& tokens) -> Matrix {
        size_t current_length = tokens.size();
        
        // Determine which tokens are new (not yet processed)
        std::vector<int> new_tokens(tokens.begin() + processed_length, tokens.end());
        
        // Process only new tokens with cache
        Matrix decoder_out = decoder->forward_with_cache(new_tokens, kv_cache, 
                                                         &cached_encoder_output, true);
        
        // Update processed length
        processed_length = current_length;

        // Project to vocabulary (last position of output)
        Matrix logits = lm_head->forward(decoder_out);

        return logits;
    };

    // Generate based on strategy
    std::vector<int> output_tokens;

    if (strategy == "greedy") {
        output_tokens = generator->generate_greedy(model_fn, {bos_token_id});
    } else if (strategy == "beam") {
        // Update config for beam search
        TextGenerator::GenerationConfig config = generator->get_config();
        config.num_beams = num_beams;
        config.max_length = max_length;
        generator->set_config(config);
        output_tokens = generator->generate_beam_search(model_fn, {bos_token_id});
    } else if (strategy == "sampling") {
        output_tokens = generator->generate_sampling(model_fn, {bos_token_id}, temperature);
    } else if (strategy == "topk") {
        output_tokens = generator->generate_top_k(model_fn, {bos_token_id}, top_k);
    } else if (strategy == "nucleus") {
        output_tokens = generator->generate_nucleus(model_fn, {bos_token_id}, top_p);
    } else {
        // Default to main generate method (uses config)
        output_tokens = generator->generate(model_fn, {bos_token_id});
    }

    // Decode tokens to text (skip special tokens like <bos>, <eos>, <unk>, <pad>)
    return tokenizer->decode(output_tokens, true);
}

// Training step
float EncoderDecoderModel::train_step(const std::string& input_text,
                                      const std::string& target_text) {
    std::vector<int> input_tokens = tokenizer->encode(input_text);
    std::vector<int> target_tokens = tokenizer->encode(target_text);

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
    bool prev_mode = requires_grad;
    set_training(false);

    std::vector<int> input_tokens = tokenizer->encode(input_text);
    std::vector<int> target_tokens = tokenizer->encode(target_text);

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
    int num_samples = input_texts.size();

    bool prev_mode = requires_grad;
    set_training(false);

    for (size_t i = 0; i < input_texts.size(); ++i) {
        total_loss += evaluate(input_texts[i], target_texts[i]);
    }

    set_training(prev_mode);

    float avg_loss = total_loss / num_samples;
    return std::exp(avg_loss);  // Perplexity = exp(cross_entropy)
}

// Set training mode
void EncoderDecoderModel::set_training(bool mode) {
    requires_grad = mode;
    // encoder->set_training(mode);  // LLMEncoder doesn't have this method
    decoder->set_training(mode);
}

// Set learning rate
void EncoderDecoderModel::set_learning_rate(float lr) {
    learning_rate = lr;
    // encoder->set_learning_rate(lr);  // LLMEncoder doesn't have this method
    decoder->set_learning_rate(lr);
}

// Update weights
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

// Backward pass without weight update (for use with external optimizer)
void EncoderDecoderModel::backward_pass(const Matrix& grad_output) {
    // Standard backward pass
    backward(grad_output);
    // Note: No weight update - handled by external optimizer
}

// Set tokenizer
void EncoderDecoderModel::set_tokenizer(BPETokenizer* tokenizer_ptr) {
    tokenizer.reset(tokenizer_ptr);
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
    tokenizer->save_vocab(filepath + ".vocab");

    // Save encoder weights
    encoder->save_weights(filepath + ".encoder");

    // Save decoder weights
    decoder->save_weights(filepath + ".decoder");

    // Save language model head
    lm_head->save_weights(filepath + ".lm_head");
}

// Load model
void EncoderDecoderModel::load_model(const std::string& filepath) {
    // Load architecture config
    std::ifstream config_file(filepath + ".config", std::ios::binary);
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to load model config");
    }

    int loaded_vocab_size, loaded_d_model, loaded_encoder_layers, loaded_decoder_layers;
    int loaded_num_heads, loaded_d_ff, loaded_max_seq_length;
    int loaded_bos, loaded_eos, loaded_pad;

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
        loaded_encoder_layers != encoder_layers || loaded_decoder_layers != decoder_layers) {
        throw std::runtime_error("Model architecture mismatch");
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

// Forward pass
Matrix EncoderDecoderModel::forward(const std::vector<int>& input_tokens,
                                    const std::vector<int>& target_tokens) {
    // Cache inputs
    cached_input_tokens = input_tokens;
    cached_target_tokens = target_tokens;

    // Encode input
    int input_len = input_tokens.size();
    Matrix encoder_mask(input_len, input_len);
    for (int i = 0; i < input_len; ++i) {
        for (int j = 0; j < input_len; ++j) {
            encoder_mask.data[i][j] = 1.0f;
        }
    }
    cached_encoder_output = encoder->encode_with_mask(input_tokens, encoder_mask);

    // Prepare decoder input (teacher forcing: use target tokens)
    // In training, we use ground truth tokens as decoder input
    std::vector<int> decoder_input;
    decoder_input.push_back(bos_token_id);
    for (size_t i = 0; i < target_tokens.size() - 1; ++i) {
        decoder_input.push_back(target_tokens[i]);
    }

    // Decode with cross-attention
    cached_decoder_output = decoder->forward_with_encoder(decoder_input, cached_encoder_output);

    // Project to vocabulary
    Matrix logits = lm_head->forward(cached_decoder_output);

    return logits;
}

// Backward pass
void EncoderDecoderModel::backward(const Matrix& grad_output) {
    if (!requires_grad) {
        return;
    }

    // Backward through LM head
    Matrix grad_decoder = lm_head->backward(grad_output);

    // Backward through decoder
    decoder->backward(grad_decoder);

    // Backward through encoder (via cross-attention gradients in decoder)
    // The decoder's cross-attention already computed gradients for encoder output
    // We need to propagate those back through the encoder
    // This is handled internally by the decoder's backward pass

    // For now, we'll implement a simple version
    // In a full implementation, we'd need to collect cross-attention gradients
    // and propagate them back through the encoder

    // Simplified: just update encoder based on its own gradients
    // (This would be enhanced in a production implementation)
}
