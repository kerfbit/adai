// @adai-status: beta        (capped by TD-035 — shipped as chatbot_api_server, no dedicated test)
// @adai-version: 0.8.0
// @adai-reviewed: 2026-09-07

#include <unistd.h>  // getpid() — POSIX (Linux + macOS)
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include "BPETokenizer.hpp"
#include "ChatbotAPI.hpp"
#include "Config.hpp"
#include "DocumentStore.hpp"
#include "EncoderDecoderModel.hpp"
#include "Logger.hpp"
#include "Matrix.hpp"
#include "RAGInference.hpp"
#include "TextGenerator.hpp"
#include "encoder.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef BUILD_MNS_SERVER
#include "ModelNameClient.hpp"
#endif

// TODO: See TECHNICAL_DEBT.md Future Enhancement #6 - Model State Persistence on Shutdown
// TODO: See TECHNICAL_DEBT.md Future Enhancement #7 - Graceful Reload (Zero-Downtime Restart)
// TODO: See TECHNICAL_DEBT.md Future Enhancement #13 - Metrics Endpoint for Prometheus
// TODO: See TECHNICAL_DEBT.md Future Enhancement #14 - systemd Socket Activation
// TODO: See TECHNICAL_DEBT.md Future Enhancement #16 - Health Check Enhancements

// ============================================================================
// Signal Handling for Graceful Shutdown and Config Reload
// ============================================================================

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
// Atomic flag for signal handling (async-signal-safe)
static std::atomic<bool> shutdown_requested{false};
static std::atomic<bool> reload_config_requested{false};

// Global pointer for signal handler (only used to stop the server)
static ChatbotAPI* g_api_server = nullptr;

// Global configuration state (protected by mutex)
static adai::ServiceConfig* g_config = nullptr;
static std::mutex* g_config_mutex = nullptr;
static std::string* g_config_file_path = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/**
 * @brief Signal handler for SIGINT, SIGTERM, and SIGHUP
 *
 * This handler is async-signal-safe and only sets atomic flags.
 * The actual cleanup/reload is performed in the main thread.
 *
 * @param signal Signal number
 */
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        // Set shutdown flag (atomic operation is async-signal-safe)
        shutdown_requested.store(true);

        // Note: Do NOT stop the server here. It's unsafe to call complex functions
        // from a signal handler. The main loop will detect the flag and stop the server.
    } else if (signal == SIGHUP) {
        // Configuration hot-reload implemented
        // Set reload flag to trigger config reload in main thread
        reload_config_requested.store(true);
        adai::Logger::info("SIGHUP received - configuration reload requested");
    }
    // TODO: See TECHNICAL_DEBT.md Future Enhancement #7 - Add SIGUSR1 handler for graceful model
    // reload SIGUSR1 should trigger background model loading and atomic swap
}

void print_usage(const char* program_name) {
    std::cout
        << "Usage: " << program_name << " [OPTIONS]\n"
        << "\nConfiguration Sources (in order of priority):\n"
        << "  1. Command-line arguments (highest priority)\n"
        << "  2. Environment variables\n"
        << "  3. Configuration file\n"
        << "  4. Default values (lowest priority)\n"
        << "\nOptions:\n"
        << "  --config <path>      Path to configuration file (default: ./config.chatbot.conf or "
           "/etc/adai/config.chatbot.conf)\n"
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
        << "  --strategy <str>     Generation strategy: greedy, beam, temperature, top_k, nucleus "
           "(default: nucleus)\n"
        << "  --help               Show this help message\n"
        << "\nEnvironment Variables:\n"
        << "  All configuration can be set via environment variables.\n"
        << "  Examples: MODEL_PATH, VOCAB_PATH, PORT, LOG_LEVEL, etc.\n"
        << '\n';
}

int main(int argc, char* argv[]) {
#ifdef _OPENMP
    // If OMP_NUM_THREADS is not set, default to maximum available cores
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        int num_procs = omp_get_num_procs();
        omp_set_num_threads(num_procs);
        std::cout << "[OpenMP] Auto-configured to use " << num_procs << " threads" << '\n';
    } else {
        std::cout << "[OpenMP] Using " << omp_get_max_threads() << " threads (from OMP_NUM_THREADS)"
                  << '\n';
    }
    // omp_set_nested(1); // Enable nested parallelism if needed for advanced tasks
#else
    std::cout << "[OpenMP] Not compiled with OpenMP support - running single-threaded" << std::endl;
#endif

    // Load configuration from file and environment variables
    std::string config_file_path;

    // First pass: check for --config argument
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file_path = argv[++i];
            break;
        }
    }

    // Discovery: --config > ./config.chatbot.conf > /etc/adai/config.chatbot.conf
    // > ./config.conf (legacy) > /etc/adai/config.conf (legacy)
    const std::string resolved_config_path =
        adai::ConfigLoader::discover_config_path(config_file_path, "config.chatbot.conf");

    // Load base configuration
    adai::ServiceConfig config = adai::ConfigLoader::load(resolved_config_path);

    // Parse command line arguments (these override config file and env vars)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--config") {
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
            config.temperature = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--top-p" && i + 1 < argc) {
            config.top_p = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--strategy" && i + 1 < argc) {
            config.strategy = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required configuration
    if (config.vocab_path.empty()) {
        std::cerr << "Error: Vocabulary path is required (use --vocab, VOCAB_PATH env var, or "
                     "config file)"
                  << '\n';
        print_usage(argv[0]);
        return 1;
    }

    // Initialize logger with configured level and file rotation
    if (!config.log_file_path.empty()) {
        // Initialize with file rotation
        adai::Logger::FileConfig file_config;
        file_config.path = config.log_file_path;
        file_config.max_size_mb = config.log_max_size_mb;
        file_config.max_files = config.log_max_files;
        file_config.compress = config.log_compress;

        adai::Logger::init(adai::Logger::Level::INFO, file_config);
    } else {
        // Console-only logging
        adai::Logger::init();
    }
    adai::Logger::set_level(config.log_level);

    // Print loaded configuration
    adai::ConfigLoader::print(config);

    try {
#ifdef BUILD_MNS_SERVER
        // Resolve model path from Model Name Service if configured
        if (!config.name_service_url.empty() &&
            (!config.model_role.empty() || !config.model_name.empty())) {
            adai::Logger::info("[MNS] Resolving model via {}", config.name_service_url);
            try {
                adai::ModelNameClient mns_client(config.name_service_url,
                                                 config.name_service_timeout_ms);
                adai::ResolvedModel resolved;
                if (!config.model_role.empty()) {
                    resolved = mns_client.resolve_role(config.model_role);
                    adai::Logger::info("[MNS] Role '{}' -> model '{}' (state={})",
                                       config.model_role, resolved.model_name, resolved.state);
                } else {
                    resolved = mns_client.resolve_model(config.model_name);
                    adai::Logger::info("[MNS] Model '{}' (state={})", resolved.model_name,
                                       resolved.state);
                }
                if (!resolved.artifact.path.empty()) {
                    config.model_path = resolved.artifact.path;
                    adai::Logger::info("[MNS] Using model path: {}", config.model_path);
                }

                // MNS is the authoritative source for architecture — see CLAUDE.md
                // "Configuration". Falls back to config.chatbot.conf's D_MODEL etc.
                // when the lookup fails.
                if (auto arch = mns_client.get_architecture(resolved.model_name)) {
                    config.d_model = arch->d_model;
                    config.num_heads = arch->num_heads;
                    config.d_ff = arch->d_ff;
                    config.num_encoder_layers = arch->num_encoder_layers;
                    config.num_decoder_layers = arch->num_decoder_layers;
                    config.max_seq_length = arch->max_seq_length;
                    adai::Logger::info("[MNS] Architecture resolved from MNS for model '{}'",
                                       resolved.model_name);
                } else {
                    adai::Logger::warn(
                        "[MNS] No architecture on record for '{}'; using local config "
                        "architecture",
                        resolved.model_name);
                }
            } catch (const std::exception& e) {
                adai::Logger::warn(
                    "[MNS] Resolution failed: {} — using configured model_path/architecture",
                    e.what());
            }
        }
#endif

        // Initialize tokenizer
        adai::Logger::info("");
        adai::Logger::info("[1/4] Loading tokenizer...");
        auto tokenizer = std::make_unique<BPETokenizer>();
        tokenizer->load_vocab(config.vocab_path);
        adai::Logger::info("  Vocabulary size: {}", tokenizer->get_vocab_size());

        // GPU initialisation (optional)
        if (config.gpu_enabled) {
            adai::Logger::info("");
            adai::Logger::info(
                "[GPU] Attempting GPU initialisation (device {}, {:.0f}% memory budget)...",
                config.gpu_device_id, config.gpu_memory_fraction * 100.0f);
#ifdef ADAI_ENABLE_GPU
            if (Matrix::gpu_try_initialize(config.gpu_device_id, config.gpu_memory_fraction)) {
                adai::Logger::info("[GPU] GPU ready. {}", Matrix::gpu_info());
            } else {
#if defined(ADAI_GPU_BACKEND_SYCL)
                adai::Logger::warn(
                    "[GPU] No Intel GPU device found or SYCL initialisation failed"
                    " — running on CPU");
#else
                adai::Logger::warn(
                    "[GPU] No CUDA device found or initialisation failed"
                    " — running on CPU");
#endif
            }
#else
            adai::Logger::warn(
                "[GPU] GPU_ENABLED is set but this binary was built without GPU support"
                " (rebuild with -DENABLE_GPU=ON for CUDA or -DENABLE_SYCL=ON for Intel Arc)");
#endif
        } else {
            adai::Logger::info("[GPU] GPU acceleration disabled (set GPU_ENABLED=true to enable)");
        }

        // Initialize model
        adai::Logger::info("");
        adai::Logger::info("[2/4] Initializing encoder-decoder model...");
        auto model = std::make_shared<EncoderDecoderModel>(
            tokenizer->get_vocab_size(),  // vocab_size (first parameter)
            config.d_model,               // d_model
            config.num_encoder_layers,    // encoder_layers
            config.num_decoder_layers,    // decoder_layers
            config.num_heads,             // num_heads
            config.d_ff,                  // d_ff
            config.max_seq_length         // max_seq_length
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
        auto api = std::make_unique<ChatbotAPI>(model.get(), tokenizer.get(), config.port,
                                                config.session_timeout);

        // Set generation configuration
        ChatbotAPI::GenerationConfig gen_config;
        gen_config.max_length = config.max_gen_length;
        gen_config.temperature = config.temperature;
        gen_config.top_p = config.top_p;
        gen_config.strategy = config.strategy;
        api->set_generation_config(gen_config);

        // Optionally initialize RAG engine
        std::shared_ptr<RAGInference> rag_engine;
        if (config.rag_enabled) {
            adai::Logger::info("");
            adai::Logger::info("[+] Initializing RAG engine...");
            try {
                auto rag_encoder = std::make_shared<LLMEncoder>(
                    static_cast<int>(tokenizer->get_vocab_size()), static_cast<int>(config.d_model),
                    static_cast<int>(config.num_encoder_layers), static_cast<int>(config.num_heads),
                    static_cast<int>(config.d_ff), static_cast<int>(config.max_seq_length));
                rag_encoder->load_tokenizer_vocab(config.vocab_path);

                auto doc_store = std::make_shared<DocumentStore>(rag_encoder);

                int doc_count = 0;
                if (!config.rag_docs_path.empty()) {
                    namespace fs = std::filesystem;
                    fs::path docs_dir(config.rag_docs_path);
                    if (fs::exists(docs_dir) && fs::is_directory(docs_dir)) {
                        for (const auto& entry : fs::directory_iterator(docs_dir)) {
                            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                                std::ifstream file(entry.path());
                                if (file.is_open()) {
                                    std::string text((std::istreambuf_iterator<char>(file)),
                                                     std::istreambuf_iterator<char>());
                                    if (!text.empty()) {
                                        doc_store->addDocument(entry.path().stem().string(), text);
                                        doc_count++;
                                    }
                                }
                            }
                        }
                        adai::Logger::info("  Indexed {} documents from: {}", doc_count,
                                           config.rag_docs_path);
                    } else {
                        adai::Logger::warn("  RAG docs path not found or not a directory: {}",
                                           config.rag_docs_path);
                    }
                } else {
                    adai::Logger::warn("  RAG_DOCS_PATH not set - no documents indexed");
                }

                RAGInference::RAGConfig rag_config;
                rag_config.num_retrieved_docs = config.rag_num_docs;
                rag_config.retrieval_threshold = config.rag_threshold;
                rag_config.max_context_length = config.rag_max_context_length;
                rag_config.gen_config.max_length = static_cast<int>(config.max_gen_length);
                rag_config.gen_config.temperature = config.temperature;
                rag_config.gen_config.top_p = config.top_p;
                rag_config.gen_config.top_k = static_cast<int>(config.top_k);

                rag_engine = std::make_shared<RAGInference>(model, doc_store, rag_config);
                api->enableRAG(rag_engine);
                adai::Logger::info("  RAG enabled: retrieving {} docs per query (threshold: {})",
                                   config.rag_num_docs, config.rag_threshold);
            } catch (const std::exception& e) {
                adai::Logger::warn("  RAG initialization failed: {} - continuing without RAG",
                                   e.what());
            }
        }

        // Initialize global config state for reload
        std::mutex config_mutex;
        g_config = &config;
        g_config_mutex = &config_mutex;

        // Store config file path for reload (same path resolved at load time above)
        std::string stored_config_path = resolved_config_path;
        g_config_file_path = &stored_config_path;

        // Set up signal handlers
        g_api_server = api.get();
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        std::signal(SIGHUP, signal_handler);  // Configuration hot-reload

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
        if (config.rag_enabled && rag_engine) {
            adai::Logger::info("  RAG:   enabled ({} docs indexed, retrieving top-{})",
                               rag_engine->getNumDocuments(), config.rag_num_docs);
        }
        // TODO: See TECHNICAL_DEBT.md Future Enhancement #13 - Add /metrics endpoint
        // Expose Prometheus metrics: request_count, request_duration, active_sessions, etc.
        // TODO: See TECHNICAL_DEBT.md Future Enhancement #16 - Enhanced /health endpoint
        // Return detailed component status, memory usage, readiness/liveness checks
        adai::Logger::info("");
        adai::Logger::info("Press Ctrl+C to stop the server");
        adai::Logger::info("Send SIGHUP (kill -HUP {}) to reload configuration", getpid());
        adai::Logger::info("==================================================");

        // Start server in a background thread to allow main thread to handle signals
        std::atomic<bool> server_error{false};
        std::thread server_thread([&]() {
            if (!api->start()) {
                adai::Logger::error("Failed to start server");
                server_error = true;
                shutdown_requested.store(true);
            }
        });

        // ================================================================
        // Main Service Loop - Check for config reload requests
        // ================================================================

        // Store original port for comparison
        int original_port = config.port;

        while (!shutdown_requested.load() && !server_error) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Check if config reload was requested
            if (reload_config_requested.load()) {
                reload_config_requested.store(false);

                // Reload configuration
                if (adai::ConfigLoader::reload(config, stored_config_path, config_mutex)) {
                    // Update logger level if changed
                    {
                        std::lock_guard<std::mutex> lock(config_mutex);
                        adai::Logger::set_level(config.log_level);
                    }

                    // Update generation configuration
                    ChatbotAPI::GenerationConfig new_gen_config;
                    {
                        std::lock_guard<std::mutex> lock(config_mutex);
                        new_gen_config.max_length = config.max_gen_length;
                        new_gen_config.temperature = config.temperature;
                        new_gen_config.top_p = config.top_p;
                        new_gen_config.strategy = config.strategy;
                    }
                    api->set_generation_config(new_gen_config);

                    adai::Logger::info("Generation configuration updated");

                    // Note: Some changes like port, model architecture cannot be applied
                    // without service restart. These are validated but logged as warnings.
                    {
                        std::lock_guard<std::mutex> lock(config_mutex);
                        if (config.port != original_port) {
                            adai::Logger::warn(
                                "Note: Port change ({} -> {}) requires service restart to take "
                                "effect",
                                original_port, config.port);
                        }
                    }
                } else {
                    adai::Logger::error(
                        "Configuration reload failed - continuing with current configuration");
                }
            }
        }

        // ================================================================
        // Graceful Shutdown Sequence
        // ================================================================

        if (shutdown_requested.load()) {
            adai::Logger::info("");
            adai::Logger::info("==================================================");
            adai::Logger::info("         Initiating Graceful Shutdown");
            adai::Logger::info("==================================================");

            // Step 1: Stop server and join thread
            adai::Logger::info("[1/3] Stopping API server...");
            if (g_api_server) {
                g_api_server->stop();
            }
            if (server_thread.joinable()) {
                server_thread.join();
            }
            adai::Logger::info("      API server stopped");

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

        if (server_error) {
            return 1;
        }

    } catch (const std::exception& e) {
        adai::Logger::error("Error: {}", e.what());
        return 1;
    }

    adai::Logger::info("");
    adai::Logger::info("Server shutdown complete");
    return 0;
}
