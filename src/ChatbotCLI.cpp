#include "ChatbotCLI.hpp"
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "BPETokenizer.hpp"
#include "ConversationContext.hpp"
#include "EncoderDecoderModel.hpp"

ChatbotCLI::ChatbotCLI(const std::string& vocab_file, const std::string& model_file,
                       const std::string& conv_save_file)
    : vocab_path(vocab_file),
      model_path(model_file),
      conversation_save_path(conv_save_file),
      max_response_length(100),
      temperature(1.0f),
      top_p(0.9f),
      top_k(50),
      beam_width(5),
      generation_strategy("nucleus") {
    // Smart pointers automatically initialize to nullptr
}

ChatbotCLI::~ChatbotCLI() = default;

bool ChatbotCLI::initialize() {
        std::cout << COLOR_SYSTEM << "🤖 Initializing Chatbot..." << COLOR_RESET << std::endl;

        // Load tokenizer
        std::cout << COLOR_SYSTEM << "📚 Loading tokenizer from: " << vocab_path << COLOR_RESET
                  << std::endl;
        tokenizer = std::make_unique<BPETokenizer>();
        tokenizer->load_vocab(vocab_path);
        std::cout << COLOR_SYSTEM
                  << "✅ Tokenizer loaded (vocab size: " << tokenizer->get_vocab_size() << ")"
                  << COLOR_RESET << std::endl;

        // Initialize model
        std::cout << COLOR_SYSTEM << "🧠 Initializing transformer model..." << COLOR_RESET
                  << std::endl;
        model = std::make_unique<EncoderDecoderModel>(
            tokenizer->get_vocab_size(),  // vocab_size
            512,                          // d_model
            6,                            // encoder_layers
            6,                            // decoder_layers
            8,                            // num_heads
            2048,                         // d_ff
            1024                          // max_seq_length
        );

        // Transfer tokenizer ownership to the model
        model->set_tokenizer(tokenizer.release());

        // Load pre-trained weights if available
        std::ifstream model_file(model_path);
        if (model_file.good()) {
            std::cout << COLOR_SYSTEM << "💾 Loading model weights from: " << model_path
                      << COLOR_RESET << std::endl;
            try {
                model->load_model(model_path);
                std::cout << COLOR_SYSTEM << "✅ Model weights loaded successfully!" << COLOR_RESET
                          << std::endl;
            } catch (...) {
                std::cout << COLOR_SYSTEM
                          << "⚠️  Failed to load model weights. Using random initialization."
                          << COLOR_RESET << std::endl;
            }
        } else {
            std::cout << COLOR_SYSTEM
                      << "ℹ️  No pre-trained model found. Using random initialization."
                      << COLOR_RESET << std::endl;
            std::cout << COLOR_SYSTEM << "   (Train the model first for better results)"
                      << COLOR_RESET << std::endl;
        }

        // Create conversation context manager
        context = std::make_unique<ConversationContext>(20,   // max 20 messages
                                                        2048  // max 2048 tokens
        );
        std::cout << COLOR_SYSTEM << "✅ Conversation manager initialized" << COLOR_RESET
                  << std::endl;

        std::cout << COLOR_SYSTEM << "🎉 Chatbot ready!" << COLOR_RESET << std::endl << std::endl;
        return true;
    }

void ChatbotCLI::print_welcome() {
        std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║          🤖 ADAI Transformer Chatbot CLI v1.0            ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << std::endl;
        std::cout << COLOR_SYSTEM << "Commands:" << COLOR_RESET << std::endl;
        std::cout << "  /help         - Show this help message" << std::endl;
        std::cout << "  /clear        - Clear conversation history" << std::endl;
        std::cout << "  /save         - Save conversation to file" << std::endl;
        std::cout << "  /load         - Load conversation from file" << std::endl;
        std::cout << "  /stats        - Show conversation statistics" << std::endl;
        std::cout << "  /settings     - Show current settings" << std::endl;
        std::cout << "  /set <param>  - Change generation parameter" << std::endl;
        std::cout << "  /system <msg> - Set system message" << std::endl;
        std::cout << "  /exit, /quit  - Exit the chatbot" << std::endl;
        std::cout << std::endl;
        std::cout << COLOR_SYSTEM << "Generation strategies:" << COLOR_RESET << std::endl;
        std::cout << "  greedy, beam, sampling, top-k, nucleus" << std::endl;
        std::cout << std::endl;
    }

void ChatbotCLI::print_stats() {
        std::cout << std::endl;
        std::cout << COLOR_SYSTEM << "📊 Conversation Statistics:" << COLOR_RESET << std::endl;
        std::cout << "  Total messages: " << context->get_message_count() << std::endl;
        std::cout << "  Estimated tokens: " << context->get_total_tokens() << std::endl;
        std::cout << std::endl;
    }

void ChatbotCLI::print_settings() {
        std::cout << std::endl;
        std::cout << COLOR_SYSTEM << "⚙️  Current Settings:" << COLOR_RESET << std::endl;
        std::cout << "  Strategy: " << generation_strategy << std::endl;
        std::cout << "  Max length: " << max_response_length << std::endl;
        std::cout << "  Temperature: " << temperature << std::endl;
        std::cout << "  Top-p (nucleus): " << top_p << std::endl;
        std::cout << "  Top-k: " << top_k << std::endl;
        std::cout << "  Beam width: " << beam_width << std::endl;
        std::cout << std::endl;
    }

void ChatbotCLI::handle_command(const std::string& command) {
        std::string_view cmd_view(command);
        
        if (command == "/help") {
            print_welcome();
        } else if (command == "/clear") {
            context->clear();
            std::cout << COLOR_SYSTEM << "✅ Conversation history cleared" << COLOR_RESET
                      << std::endl;
        } else if (command == "/save") {
            try {
                context->save_to_file(conversation_save_path);
                std::cout << COLOR_SYSTEM << "✅ Conversation saved to: " << conversation_save_path
                          << COLOR_RESET << std::endl;
            } catch (...) {
                std::cout << COLOR_ERROR << "❌ Failed to save conversation" << COLOR_RESET
                          << std::endl;
            }
        } else if (command == "/load") {
            try {
                context->load_from_file(conversation_save_path);
                std::cout << COLOR_SYSTEM
                          << "✅ Conversation loaded from: " << conversation_save_path
                          << COLOR_RESET << std::endl;
            } catch (...) {
                std::cout << COLOR_ERROR << "❌ Failed to load conversation" << COLOR_RESET
                          << std::endl;
            }
        } else if (command == "/stats") {
            print_stats();
        } else if (command == "/settings") {
            print_settings();
        } else if (cmd_view.size() > 5 && cmd_view.substr(0, 5) == "/set ") {
            handle_setting(cmd_view.substr(5));
        } else if (cmd_view.size() > 8 && cmd_view.substr(0, 8) == "/system ") {
            std::string system_msg(cmd_view.substr(8));
            context->set_system_message(system_msg);
            std::cout << COLOR_SYSTEM << "✅ System message set" << COLOR_RESET << std::endl;
        } else {
            std::cout << COLOR_ERROR << "❓ Unknown command. Type /help for available commands."
                      << COLOR_RESET << std::endl;
        }
    }

void ChatbotCLI::handle_setting(std::string_view setting) {
        size_t space_pos = setting.find(' ');
        if (space_pos == std::string_view::npos) {
            std::cout << COLOR_ERROR << "❌ Usage: /set <parameter> <value>" << COLOR_RESET
                      << std::endl;
            return;
        }

        std::string_view param = setting.substr(0, space_pos);
        std::string_view value = setting.substr(space_pos + 1);

        if (param == "strategy") {
            if (value == "greedy" || value == "beam" || value == "sampling" || value == "top-k" ||
                value == "nucleus") {
                generation_strategy = value;
                std::cout << COLOR_SYSTEM << "✅ Generation strategy set to: " << value
                          << COLOR_RESET << std::endl;
            } else {
                std::cout << COLOR_ERROR
                          << "❌ Invalid strategy. Use: greedy, beam, sampling, top-k, or nucleus"
                          << COLOR_RESET << std::endl;
            }
        } else if (param == "length" || param == "max_length") {
            max_response_length = std::stoi(std::string(value));
            std::cout << COLOR_SYSTEM << "✅ Max response length set to: " << max_response_length
                      << COLOR_RESET << std::endl;
        } else if (param == "temperature" || param == "temp") {
            temperature = std::stof(std::string(value));
            std::cout << COLOR_SYSTEM << "✅ Temperature set to: " << temperature << COLOR_RESET
                      << std::endl;
        } else if (param == "top_p" || param == "top-p") {
            top_p = std::stof(std::string(value));
            std::cout << COLOR_SYSTEM << "✅ Top-p set to: " << top_p << COLOR_RESET << std::endl;
        } else if (param == "top_k" || param == "top-k") {
            top_k = std::stoi(std::string(value));
            std::cout << COLOR_SYSTEM << "✅ Top-k set to: " << top_k << COLOR_RESET << std::endl;
        } else if (param == "beam_width" || param == "beam-width") {
            beam_width = std::stoi(std::string(value));
            std::cout << COLOR_SYSTEM << "✅ Beam width set to: " << beam_width << COLOR_RESET
                      << std::endl;
        } else {
            std::cout << COLOR_ERROR << "❌ Unknown parameter: " << param << COLOR_RESET
                      << std::endl;
            std::cout << COLOR_SYSTEM
                      << "Available: strategy, length, temperature, top_p, top_k, beam_width"
                      << COLOR_RESET << std::endl;
        }
    }

std::string ChatbotCLI::generate_response(const std::string& user_input) {
        // Add user message to context
        context->add_user_message(user_input);

        // Format context for model
        std::string formatted_context = context->format_with_special_tokens();

        // Generate response using EncoderDecoderModel
        std::string response;

        try {
            response = model->generate_response_with_strategy(
                formatted_context, max_response_length, generation_strategy, temperature, top_p,
                top_k, beam_width);
        } catch (const std::exception& e) {
            response = "[Error generating response: " + std::string(e.what()) + "]";
        }

        // Add assistant response to context
        context->add_assistant_message(response);

        return response;
    }

void ChatbotCLI::run() {
        if (!initialize()) {
            std::cerr << COLOR_ERROR << "Failed to initialize chatbot!" << COLOR_RESET << std::endl;
            return;
        }

        print_welcome();

        std::string user_input;
        bool running = true;

        while (running) {
            // Display prompt
            std::cout << COLOR_USER << "You: " << COLOR_RESET;

            // Get user input
            std::getline(std::cin, user_input);

            // Trim whitespace
            user_input.erase(0, user_input.find_first_not_of(" \t\n\r"));
            user_input.erase(user_input.find_last_not_of(" \t\n\r") + 1);

            // Skip empty input
            if (user_input.empty()) {
                continue;
            }

            // Check for exit commands
            if (user_input == "/exit" || user_input == "/quit") {
                running = false;
                continue;
            }

            // Handle commands
            if (user_input[0] == '/') {
                handle_command(user_input);
                continue;
            }

            // Generate and display response
            std::cout << COLOR_BOT << "Bot: " << COLOR_RESET;
            std::string response = generate_response(user_input);
            std::cout << response << std::endl << std::endl;
        }

        // Save conversation on exit
        std::cout << std::endl
                  << COLOR_SYSTEM << "💾 Saving conversation..." << COLOR_RESET << std::endl;
        try {
            context->save_to_file(conversation_save_path);
            std::cout << COLOR_SYSTEM << "✅ Conversation saved to: " << conversation_save_path
                      << COLOR_RESET << std::endl;
        } catch (...) {
            std::cout << COLOR_ERROR << "❌ Failed to save conversation" << COLOR_RESET
                      << std::endl;
        }

        std::cout << COLOR_SYSTEM << "👋 Goodbye!" << COLOR_RESET << std::endl;
    }
