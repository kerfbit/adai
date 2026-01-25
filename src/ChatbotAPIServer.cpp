#include "ChatbotAPI.hpp"
#include "EncoderDecoderModel.hpp"
#include "TextGenerator.hpp"
#include "BPETokenizer.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <csignal>
#include <memory>

// Global pointer for signal handler
ChatbotAPI* g_api_server = nullptr;

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\nReceived shutdown signal, stopping server..." << std::endl;
        if (g_api_server) {
            g_api_server->stop();
        }
        exit(0);
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "Options:\n"
              << "  --model <path>       Path to model file (required)\n"
              << "  --vocab <path>       Path to vocabulary file (required)\n"
              << "  --port <number>      Port number (default: 8080)\n"
              << "  --timeout <minutes>  Session timeout in minutes (default: 30)\n"
              << "  --d-model <number>   Model dimension (default: 512)\n"
              << "  --num-heads <number> Number of attention heads (default: 8)\n"
              << "  --d-ff <number>      Feed-forward dimension (default: 2048)\n"
              << "  --enc-layers <n>     Number of encoder layers (default: 6)\n"
              << "  --dec-layers <n>     Number of decoder layers (default: 6)\n"
              << "  --max-seq-len <n>    Maximum sequence length (default: 1024)\n"
              << "  --max-gen-len <n>    Maximum generation length (default: 100)\n"
              << "  --temperature <f>    Generation temperature (default: 1.0)\n"
              << "  --top-p <f>          Nucleus sampling threshold (default: 0.9)\n"
              << "  --strategy <str>     Generation strategy: greedy, beam, temperature, top_k, nucleus (default: nucleus)\n"
              << "  --help               Show this help message\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Default configuration
    std::string model_path;
    std::string vocab_path;
    int port = 8080;
    int session_timeout = 30;
    
    // Model architecture parameters
    size_t d_model = 512;
    size_t num_heads = 8;
    size_t d_ff = 2048;
    size_t num_encoder_layers = 6;
    size_t num_decoder_layers = 6;
    size_t max_seq_length = 1024;
    
    // Generation parameters
    size_t max_gen_length = 100;
    float temperature = 1.0f;
    float top_p = 0.9f;
    std::string strategy = "nucleus";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--timeout" && i + 1 < argc) {
            session_timeout = std::atoi(argv[++i]);
        } else if (arg == "--d-model" && i + 1 < argc) {
            d_model = std::atoi(argv[++i]);
        } else if (arg == "--num-heads" && i + 1 < argc) {
            num_heads = std::atoi(argv[++i]);
        } else if (arg == "--d-ff" && i + 1 < argc) {
            d_ff = std::atoi(argv[++i]);
        } else if (arg == "--enc-layers" && i + 1 < argc) {
            num_encoder_layers = std::atoi(argv[++i]);
        } else if (arg == "--dec-layers" && i + 1 < argc) {
            num_decoder_layers = std::atoi(argv[++i]);
        } else if (arg == "--max-seq-len" && i + 1 < argc) {
            max_seq_length = std::atoi(argv[++i]);
        } else if (arg == "--max-gen-len" && i + 1 < argc) {
            max_gen_length = std::atoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::atof(argv[++i]);
        } else if (arg == "--strategy" && i + 1 < argc) {
            strategy = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (vocab_path.empty()) {
        std::cerr << "Error: --vocab is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "         ADAI Chatbot API Server" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Model path: " << (model_path.empty() ? "<new model>" : model_path) << std::endl;
    std::cout << "  Vocabulary: " << vocab_path << std::endl;
    std::cout << "  Port: " << port << std::endl;
    std::cout << "  Session timeout: " << session_timeout << " minutes" << std::endl;
    std::cout << "  Model architecture:" << std::endl;
    std::cout << "    - d_model: " << d_model << std::endl;
    std::cout << "    - num_heads: " << num_heads << std::endl;
    std::cout << "    - d_ff: " << d_ff << std::endl;
    std::cout << "    - encoder_layers: " << num_encoder_layers << std::endl;
    std::cout << "    - decoder_layers: " << num_decoder_layers << std::endl;
    std::cout << "    - max_seq_length: " << max_seq_length << std::endl;
    std::cout << "  Generation parameters:" << std::endl;
    std::cout << "    - max_length: " << max_gen_length << std::endl;
    std::cout << "    - temperature: " << temperature << std::endl;
    std::cout << "    - top_p: " << top_p << std::endl;
    std::cout << "    - strategy: " << strategy << std::endl;
    std::cout << "==================================================" << std::endl;

    try {
        // Initialize tokenizer
        std::cout << "\n[1/4] Loading tokenizer..." << std::endl;
        auto tokenizer = std::make_unique<BPETokenizer>();
        tokenizer->load_vocab(vocab_path);
        std::cout << "  Vocabulary size: " << tokenizer->get_vocab_size() << std::endl;

        // Initialize model
        std::cout << "\n[2/4] Initializing encoder-decoder model..." << std::endl;
        auto model = std::make_unique<EncoderDecoderModel>(
            d_model,
            num_heads,
            d_ff,
            num_encoder_layers,
            num_decoder_layers,
            tokenizer->get_vocab_size(),
            max_seq_length
        );

        // Load model weights if path provided
        if (!model_path.empty()) {
            std::cout << "  Loading model weights from: " << model_path << std::endl;
            try {
                model->load_model(model_path);
                std::cout << "  Model weights loaded successfully" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "  Warning: Failed to load model weights: " << e.what() << std::endl;
                std::cerr << "  Using random initialization" << std::endl;
            }
        } else {
            std::cout << "  Using randomly initialized model (training required)" << std::endl;
        }

        // Initialize API server
        std::cout << "\n[3/4] Initializing API server..." << std::endl;
        auto api = std::make_unique<ChatbotAPI>(
            model.get(),
            tokenizer.get(),
            port,
            session_timeout
        );

        // Set generation configuration
        ChatbotAPI::GenerationConfig config;
        config.max_length = max_gen_length;
        config.temperature = temperature;
        config.top_p = top_p;
        config.strategy = strategy;
        api->set_generation_config(config);

        // Set up signal handlers
        g_api_server = api.get();
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // Start server
        std::cout << "\n==================================================" << std::endl;
        std::cout << "Server starting on http://0.0.0.0:" << port << std::endl;
        std::cout << "Available endpoints:" << std::endl;
        std::cout << "  POST   /chat           - Single-turn conversation" << std::endl;
        std::cout << "  POST   /chat/session   - Multi-turn conversation" << std::endl;
        std::cout << "  POST   /clear-session  - Clear session history" << std::endl;
        std::cout << "  GET    /health         - Health check" << std::endl;
        std::cout << "\nPress Ctrl+C to stop the server" << std::endl;
        std::cout << "==================================================" << std::endl;

        if (!api->start()) {
            std::cerr << "Failed to start server" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Server stopped" << std::endl;
    return 0;
}
