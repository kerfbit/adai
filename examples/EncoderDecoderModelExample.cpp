#include <iostream>
#include <string>
#include <vector>
#include "EncoderDecoderModel.hpp"

/**
 * Example: EncoderDecoderModel for Sequence-to-Sequence Tasks
 *
 * Demonstrates:
 * 1. Model initialization
 * 2. Text generation with different strategies
 * 3. Training on (input, output) pairs
 * 4. Save/load model
 * 5. Multi-turn conversation
 */

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

int main() {
    std::cout << "EncoderDecoderModel Example\n";
    std::cout << "===========================\n\n";

    // ========================================================================
    // Example 1: Initialize Model
    // ========================================================================
    print_section("Example 1: Initialize EncoderDecoderModel");

    int vocab_size = 1000;
    int d_model = 128;  // Small for demonstration
    int encoder_layers = 2;
    int decoder_layers = 2;
    int num_heads = 4;
    int d_ff = 512;
    int max_seq_length = 64;

    EncoderDecoderModel model(vocab_size, d_model, encoder_layers, decoder_layers, num_heads, d_ff,
                              max_seq_length);

    std::cout << "Model initialized:\n";
    std::cout << "  Vocabulary size: " << model.get_vocab_size() << "\n";
    std::cout << "  Model dimension: " << model.get_d_model() << "\n";
    std::cout << "  Encoder layers: " << model.get_encoder_layers() << "\n";
    std::cout << "  Decoder layers: " << model.get_decoder_layers() << "\n";
    std::cout << "  BOS token: " << model.get_bos_token_id() << "\n";
    std::cout << "  EOS token: " << model.get_eos_token_id() << "\n";
    std::cout << "  PAD token: " << model.get_pad_token_id() << "\n";

    // ========================================================================
    // Example 2: Setup Tokenizer
    // ========================================================================
    print_section("Example 2: Configure Tokenizer");

    // Build simple vocabulary
    BPETokenizer* tokenizer = model.get_tokenizer();
    std::vector<std::string> corpus = {
        "hello world",       "how are you",          "i am fine",       "thank you", "goodbye",
        "what is your name", "my name is assistant", "nice to meet you"};

    std::cout << "Building vocabulary from corpus...\n";
    tokenizer->build_vocab(corpus, 100);
    std::cout << "Vocabulary size: " << tokenizer->get_vocab_size() << "\n";

    // Test encoding/decoding
    std::string test_text = "hello";
    std::vector<int> tokens = tokenizer->encode(test_text);
    std::string decoded = tokenizer->decode(tokens);
    std::cout << "\nTokenization test:\n";
    std::cout << "  Input: \"" << test_text << "\"\n";
    std::cout << "  Tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "  Decoded: \"" << decoded << "\"\n";

    // ========================================================================
    // Example 3: Generate Response (Greedy Decoding)
    // ========================================================================
    print_section("Example 3: Generate Response - Greedy Decoding");

    std::cout << "Note: Model is randomly initialized, so output will be random.\n";
    std::cout << "In practice, you would train the model first.\n\n";

    std::string input1 = "hello";
    std::cout << "Input: \"" << input1 << "\"\n";

    try {
        std::string response1 = model.generate_response_with_strategy(input1, 20, "greedy");
        std::cout << "Response (greedy): \"" << response1 << "\"\n";
    } catch (const std::exception& e) {
        std::cout << "Generation skipped (model needs training)\n";
        std::cout << "Error: " << e.what() << "\n";
    }

    // ========================================================================
    // Example 4: Different Generation Strategies
    // ========================================================================
    print_section("Example 4: Multiple Generation Strategies");

    std::string input2 = "how are you";
    std::cout << "Input: \"" << input2 << "\"\n\n";

    // Temperature sampling
    std::cout << "Strategy: Temperature Sampling (temp=0.8)\n";
    try {
        std::string response2 = model.generate_response_with_strategy(input2, 20, "sampling", 0.8f);
        std::cout << "Response: \"" << response2 << "\"\n\n";
    } catch (const std::exception& e) {
        std::cout << "Skipped: " << e.what() << "\n\n";
    }

    // Top-k sampling
    std::cout << "Strategy: Top-k Sampling (k=10, temp=1.0)\n";
    try {
        std::string response3 = model.generate_response_with_strategy(input2, 20, "topk", 1.0f, 10);
        std::cout << "Response: \"" << response3 << "\"\n\n";
    } catch (const std::exception& e) {
        std::cout << "Skipped: " << e.what() << "\n\n";
    }

    // Nucleus sampling
    std::cout << "Strategy: Nucleus Sampling (p=0.9, temp=1.0)\n";
    try {
        std::string response4 =
            model.generate_response_with_strategy(input2, 20, "nucleus", 1.0f, 50, 0.9f);
        std::cout << "Response: \"" << response4 << "\"\n\n";
    } catch (const std::exception& e) {
        std::cout << "Skipped: " << e.what() << "\n\n";
    }

    // Beam search
    std::cout << "Strategy: Beam Search (beams=4)\n";
    try {
        std::string response5 =
            model.generate_response_with_strategy(input2, 20, "beam", 1.0f, 50, 0.9f, 4);
        std::cout << "Response: \"" << response5 << "\"\n";
    } catch (const std::exception& e) {
        std::cout << "Skipped: " << e.what() << "\n";
    }

    // ========================================================================
    // Example 5: Training on (Input, Output) Pairs
    // ========================================================================
    print_section("Example 5: Training on Sequence Pairs");

    std::cout << "Training model on conversation pairs...\n\n";

    std::vector<std::pair<std::string, std::string>> training_data = {
        {"hello", "hi there"},
        {"how are you", "i am fine"},
        {"what is your name", "i am assistant"},
        {"goodbye", "see you later"}};

    model.set_training(true);
    model.set_learning_rate(0.001f);

    std::cout << "Training for 3 epochs:\n";
    for (int epoch = 1; epoch <= 3; ++epoch) {
        float epoch_loss = 0.0f;
        int num_pairs = 0;

        for (const auto& pair : training_data) {
            try {
                float loss = model.train_step(pair.first, pair.second);
                epoch_loss += loss;
                num_pairs++;

                if (num_pairs == 1 && epoch == 1) {
                    std::cout << "  Example pair: \"" << pair.first << "\" -> \"" << pair.second
                              << "\"\n";
                    std::cout << "  Initial loss: " << loss << "\n";
                }
            } catch (const std::exception& e) {
                std::cout << "  Warning: " << e.what() << "\n";
            }
        }

        if (num_pairs > 0) {
            float avg_loss = epoch_loss / num_pairs;
            std::cout << "Epoch " << epoch << " - Avg Loss: " << avg_loss << "\n";
        }
    }

    model.set_training(false);
    std::cout << "\nTraining completed!\n";

    // ========================================================================
    // Example 6: Evaluation
    // ========================================================================
    print_section("Example 6: Model Evaluation");

    std::cout << "Evaluating on validation data...\n\n";

    std::vector<std::pair<std::string, std::string>> val_data = {{"hello", "hi there"},
                                                                 {"goodbye", "see you later"}};

    for (const auto& pair : val_data) {
        try {
            float val_loss = model.evaluate(pair.first, pair.second);
            std::cout << "Input: \"" << pair.first << "\"\n";
            std::cout << "Target: \"" << pair.second << "\"\n";
            std::cout << "Validation Loss: " << val_loss << "\n\n";
        } catch (const std::exception& e) {
            std::cout << "Evaluation error: " << e.what() << "\n\n";
        }
    }

    // Compute perplexity
    try {
        std::vector<std::string> inputs, targets;
        for (const auto& pair : val_data) {
            inputs.push_back(pair.first);
            targets.push_back(pair.second);
        }

        float perplexity = model.compute_perplexity(inputs, targets);
        std::cout << "Overall Perplexity: " << perplexity << "\n";
    } catch (const std::exception& e) {
        std::cout << "Perplexity calculation skipped: " << e.what() << "\n";
    }

    // ========================================================================
    // Example 7: Configure Generation Parameters
    // ========================================================================
    print_section("Example 7: Configure Generation Parameters");

    TextGenerator::GenerationConfig gen_config;
    gen_config.max_length = 30;
    gen_config.temperature = 0.7f;
    gen_config.top_k = 40;
    gen_config.top_p = 0.9f;
    gen_config.repetition_penalty = 1.2f;
    gen_config.num_beams = 1;
    gen_config.length_penalty = 1.0f;
    gen_config.early_stopping = true;

    model.set_generation_config(gen_config);

    std::cout << "Generation configuration updated:\n";
    std::cout << "  Max length: " << gen_config.max_length << "\n";
    std::cout << "  Temperature: " << gen_config.temperature << "\n";
    std::cout << "  Top-k: " << gen_config.top_k << "\n";
    std::cout << "  Top-p: " << gen_config.top_p << "\n";
    std::cout << "  Repetition penalty: " << gen_config.repetition_penalty << "\n";

    // ========================================================================
    // Example 8: Save and Load Model
    // ========================================================================
    print_section("Example 8: Save and Load Model");

    std::string model_path = "encoder_decoder_model";

    try {
        std::cout << "Saving model to: " << model_path << ".*\n";
        model.save_model(model_path);
        std::cout << "Model saved successfully!\n\n";

        // Create a new model and load weights
        std::cout << "Creating new model instance...\n";
        EncoderDecoderModel loaded_model(vocab_size, d_model, encoder_layers, decoder_layers,
                                         num_heads, d_ff, max_seq_length);

        std::cout << "Loading weights from: " << model_path << ".*\n";
        loaded_model.load_model(model_path);
        std::cout << "Model loaded successfully!\n";

        std::cout << "\nLoaded model configuration:\n";
        std::cout << "  Vocab size: " << loaded_model.get_vocab_size() << "\n";
        std::cout << "  Model dimension: " << loaded_model.get_d_model() << "\n";
        std::cout << "  Encoder layers: " << loaded_model.get_encoder_layers() << "\n";
        std::cout << "  Decoder layers: " << loaded_model.get_decoder_layers() << "\n";

    } catch (const std::exception& e) {
        std::cout << "Save/load failed: " << e.what() << "\n";
    }

    // ========================================================================
    // Example 9: Multi-Turn Conversation Simulation
    // ========================================================================
    print_section("Example 9: Multi-Turn Conversation");

    std::cout << "Simulating a multi-turn conversation:\n";
    std::cout << "(Note: With random initialization, responses will be random)\n\n";

    std::vector<std::string> conversation = {"hello", "how are you", "what is your name",
                                             "goodbye"};

    for (const auto& user_input : conversation) {
        std::cout << "User: " << user_input << "\n";

        try {
            std::string bot_response = model.generate_response(user_input, 20);
            std::cout << "Bot:  " << bot_response << "\n\n";
        } catch (const std::exception& e) {
            std::cout << "Bot:  [Error: " << e.what() << "]\n\n";
        }
    }

    // ========================================================================
    // Example 10: Access Internal Components
    // ========================================================================
    print_section("Example 10: Access Internal Components");

    std::cout << "Accessing internal components for custom operations:\n\n";

    LLMEncoder* encoder = model.get_encoder();
    LLMDecoder* decoder = model.get_decoder();
    LanguageModelHead* lm_head = model.get_lm_head();
    TextGenerator* generator = model.get_generator();

    std::cout << "Encoder:\n";
    // Note: LLMEncoder doesn't expose getters for num_layers and d_model
    std::cout << "  (Internal encoder details not accessible)\n\n";

    std::cout << "Decoder:\n";
    std::cout << "  Layers: " << decoder->get_num_layers() << "\n";
    std::cout << "  d_model: " << decoder->get_d_model() << "\n";
    std::cout << "  Vocab size: " << decoder->get_vocab_size() << "\n\n";

    std::cout << "Language Model Head:\n";
    std::cout << "  (Internal LM head details not accessible)\n\n";

    std::cout << "Text Generator:\n";
    TextGenerator::GenerationConfig current_config = generator->get_config();
    std::cout << "  Max length: " << current_config.max_length << "\n";
    std::cout << "  Temperature: " << current_config.temperature << "\n";

    // ========================================================================
    // Summary
    // ========================================================================
    print_section("Summary");

    std::cout << "This example demonstrated:\n";
    std::cout << "  ✓ EncoderDecoderModel initialization\n";
    std::cout << "  ✓ Tokenizer configuration\n";
    std::cout << "  ✓ Text generation with multiple strategies\n";
    std::cout << "  ✓ Training on (input, output) pairs\n";
    std::cout << "  ✓ Model evaluation and perplexity\n";
    std::cout << "  ✓ Generation parameter configuration\n";
    std::cout << "  ✓ Model persistence (save/load)\n";
    std::cout << "  ✓ Multi-turn conversation\n";
    std::cout << "  ✓ Access to internal components\n\n";

    std::cout << "Next steps for production use:\n";
    std::cout << "  1. Train on large-scale conversation dataset\n";
    std::cout << "  2. Implement proper data preprocessing pipeline\n";
    std::cout << "  3. Add validation and early stopping\n";
    std::cout << "  4. Fine-tune hyperparameters\n";
    std::cout << "  5. Deploy with API endpoint\n";
    std::cout << "  6. Add conversation history management\n";
    std::cout << "  7. Implement safety filters and moderation\n\n";

    std::cout << "Example completed successfully!\n";

    return 0;
}
