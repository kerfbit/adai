#include "ChatbotAPI.hpp"
#include "EncoderDecoderModel.hpp"
#include "TextGenerator.hpp"
#include "BPETokenizer.hpp"
#include "Config.hpp"
#include "Logger.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <csignal>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>

// TODO: See TECHNICAL_DEBT.md Future Enhancement #6 - Model State Persistence on Shutdown
// TODO: See TECHNICAL_DEBT.md Future Enhancement #7 - Graceful Reload (Zero-Downtime Restart)
// TODO: See TECHNICAL_DEBT.md Future Enhancement #13 - Metrics Endpoint for Prometheus
// TODO: See TECHNICAL_DEBT.md Future Enhancement #14 - systemd Socket Activation
// TODO: See TECHNICAL_DEBT.md Future Enhancement #16 - Health Check Enhancements

// ============================================================================
// Signal Handling for Graceful Shutdown
// ============================================================================

// Atomic flag for signal handling (async-signal-safe)
static std::atomic<bool> shutdown_requested{false};

// Global pointer for signal handler (only used to stop the server)
static ChatbotAPI* g_api_server = nullptr;

/**
 * @brief Signal handler for SIGINT and SIGTERM
 * 
 * This handler is async-signal-safe and only sets an atomic flag.
 * The actual cleanup is performed in the main thread.
 * 
 * @param signal Signal number
 */
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        // Set shutdown flag (atomic operation is async-signal-safe)
        shutdown_requested.store(true);
        
        // Stop the server (this is safe to call from signal handler)
        if (g_api_server) {
            g_api_server->stop();
        }
    }
    // TODO: See TECHNICAL_DEBT.md Future Enhancement #3 - Add SIGHUP handler for config reload
    // SIGHUP should trigger configuration reload without service restart
    // TODO: See TECHNICAL_DEBT.md Future Enhancement #7 - Add SIGUSR1 handler for graceful model reload
    // SIGUSR1 should trigger background model loading and atomic swap
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "\nConfiguration Sources (in order of priority):\n"
              << "  1. Command-line arguments (highest priority)\n"
              << "  2. Environment variables\n"
              << "  3. Configuration file\n"
              << "  4. Default values (lowest priority)\n"
              << "\nOptions:\n"
              << "  --config <path>      Path to configuration file (default: /etc/adai/config.conf)\n"
              << "  --model <path>       Path to model file\n"
              << "  --vocab <path>       Path to vocabulary file\n"
              << "  --port <number>      Port number (default: 8080)\n"
              << "  --timeout <minutes>  Session timeout in minutes (default: 30)\n"
              << "  --log-level <level>  Logging level: DEBUG, INFO, WARN, ERROR (default: INFO)\n"
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
              << "\nEnvironment Variables:\n"
              << "  All configuration can be set via environment variables.\n"
              << "  Examples: MODEL_PATH, VOCAB_PATH, PORT, LOG_LEVEL, etc.\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Load configuration from file and environment variables
    std::string config_file_path;
    bool use_custom_config = false;
    
    // First pass: check for --config argument
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file_path = argv[++i];
            use_custom_config = true;
            break;
        }
    }
    
    // Load base configuration
    adai::ServiceConfig config = use_custom_config ? 
        adai::ConfigLoader::load(config_file_path) : 
        adai::ConfigLoader::load();

    // Parse command line arguments (these override config file and env vars)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--config") {
            // Already handled in first pass
            ++i;  // Skip the value
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            config.vocab_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::atoi(argv[++i]);
        } else if (arg == "--timeout" && i + 1 < argc) {
            config.session_timeout = std::atoi(argv[++i]);
        } else if (arg == "--log-level" && i + 1 < argc) {
            config.log_level = argv[++i];
        } else if (arg == "--d-model" && i + 1 < argc) {
            config.d_model = std::atoi(argv[++i]);
        } else if (arg == "--num-heads" && i + 1 < argc) {
            config.num_heads = std::atoi(argv[++i]);
        } else if (arg == "--d-ff" && i + 1 < argc) {
            config.d_ff = std::atoi(argv[++i]);
        } else if (arg == "--enc-layers" && i + 1 < argc) {
            config.num_encoder_layers = std::atoi(argv[++i]);
        } else if (arg == "--dec-layers" && i + 1 < argc) {
            config.num_decoder_layers = std::atoi(argv[++i]);
        } else if (arg == "--max-seq-len" && i + 1 < argc) {
            config.max_seq_length = std::atoi(argv[++i]);
        } else if (arg == "--max-gen-len" && i + 1 < argc) {
            config.max_gen_length = std::atoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::atof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            config.top_p = std::atof(argv[++i]);
        } else if (arg == "--strategy" && i + 1 < argc) {
            config.strategy = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required configuration
    if (config.vocab_path.empty()) {
        std::cerr << "Error: Vocabulary path is required (use --vocab, VOCAB_PATH env var, or config file)" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // Initialize logger with configured level
    adai::Logger::init();
    adai::Logger::set_level(config.log_level);

    // Print loaded configuration
    adai::ConfigLoader::print(config);

    try {
        // Initialize tokenizer
        adai::Logger::info("");
        adai::Logger::info("[1/4] Loading tokenizer...");
        auto tokenizer = std::make_unique<BPETokenizer>();
        tokenizer->load_vocab(config.vocab_path);
        adai::Logger::info("  Vocabulary size: {}", tokenizer->get_vocab_size());

        // Initialize model
        adai::Logger::info("");
        adai::Logger::info("[2/4] Initializing encoder-decoder model...");
        auto model = std::make_unique<EncoderDecoderModel>(
            tokenizer->get_vocab_size(),  // vocab_size (first parameter)
            config.d_model,                // d_model
            config.num_encoder_layers,     // encoder_layers
            config.num_decoder_layers,     // decoder_layers
            config.num_heads,              // num_heads
            config.d_ff,                   // d_ff
            config.max_seq_length          // max_seq_length
        );

        // Load model weights if path provided
        if (!config.model_path.empty()) {
            adai::Logger::info("  Loading model weights from: {}", config.model_path);
            try {
                model->load_model(config.model_path);
                adai::Logger::info("  Model weights loaded successfully");
            } catch (const std::exception& e) {
                adai::Logger::warn("  Failed to load model weights: {}", e.what());
                adai::Logger::warn("  Using random initialization");
            }
        } else {
            adai::Logger::info("  Using randomly initialized model (training required)");
        }

        // Initialize API server
        adai::Logger::info("");
        adai::Logger::info("[3/4] Initializing API server...");
        auto api = std::make_unique<ChatbotAPI>(
            model.get(),
            tokenizer.get(),
            config.port,
            config.session_timeout
        );

        // Set generation configuration
        ChatbotAPI::GenerationConfig gen_config;
        gen_config.max_length = config.max_gen_length;
        gen_config.temperature = config.temperature;
        gen_config.top_p = config.top_p;
        gen_config.strategy = config.strategy;
        api->set_generation_config(gen_config);

        // Set up signal handlers
        g_api_server = api.get();
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // Start server
        adai::Logger::info("");
        adai::Logger::info("[4/4] Starting API server...");
        adai::Logger::info("==================================================");
        adai::Logger::info("Server starting on http://0.0.0.0:{}", config.port);
        adai::Logger::info("Available endpoints:");
        adai::Logger::info("  POST   /chat           - Single-turn conversation");
        adai::Logger::info("  POST   /chat/session   - Multi-turn conversation");
        adai::Logger::info("  POST   /clear-session  - Clear session history");
        adai::Logger::info("  GET    /health         - Health check");
        // TODO: See TECHNICAL_DEBT.md Future Enhancement #13 - Add /metrics endpoint
        // Expose Prometheus metrics: request_count, request_duration, active_sessions, etc.
        // TODO: See TECHNICAL_DEBT.md Future Enhancement #16 - Enhanced /health endpoint
        // Return detailed component status, memory usage, readiness/liveness checks
        adai::Logger::info("");
        adai::Logger::info("Press Ctrl+C to stop the server");
        adai::Logger::info("==================================================");

        if (!api->start()) {
            adai::Logger::error("Failed to start server");
            return 1;
        }

        // ================================================================
        // Graceful Shutdown Sequence
        // ================================================================
        
        if (shutdown_requested.load()) {
            adai::Logger::info("");
            adai::Logger::info("==================================================");
            adai::Logger::info("         Initiating Graceful Shutdown");
            adai::Logger::info("==================================================");
            
            // Step 1: Server has already been stopped by signal handler
            adai::Logger::info("[1/3] API server stopped");
            
            // Step 2: Save model state if needed
            // Note: Currently the API server doesn't modify the model,
            // but this is where we would save it if we had online learning
            // TODO: See TECHNICAL_DEBT.md Future Enhancement #6 - Model State Persistence
            // Automatically save model weights during graceful shutdown if MODEL_PATH is configured
            // Add checkpoint metadata (timestamp, loss, training state)
            if (!config.model_path.empty()) {
                adai::Logger::info("[2/3] Model state: {} (read-only)", config.model_path);
                // Future: Call model->save_weights(config.model_path) here
            } else {
                adai::Logger::info("[2/3] Model state: not persisted (no model path configured)");
            }
            
            // Step 3: Cleanup resources (RAII will handle this)
            adai::Logger::info("[3/3] Cleaning up resources...");
            
            adai::Logger::info("");
            adai::Logger::info("Graceful shutdown complete");
            adai::Logger::info("==================================================");
        }

    } catch (const std::exception& e) {
        adai::Logger::error("Error: {}", e.what());
        return 1;
    }

    adai::Logger::info("");
    adai::Logger::info("Server shutdown complete");
    return 0;
}
