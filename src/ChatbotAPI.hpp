#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "BPETokenizer.hpp"
#include "BatchProcessor.hpp"
#include "ConversationContext.hpp"
#include "EncoderDecoderModel.hpp"
#include "RAGInference.hpp"
#include "TextGenerator.hpp"

/**
 * @brief Session information for multi-turn conversations
 */
struct Session {
    std::unique_ptr<ConversationContext> context;
    std::chrono::steady_clock::time_point last_access;

    Session(size_t max_messages = 10, size_t max_tokens = 2048)
        : context(std::make_unique<ConversationContext>(max_messages, max_tokens)),
          last_access(std::chrono::steady_clock::now()) {}
};

/**
 * @brief ChatbotAPI - REST API layer for chatbot service
 *
 * Provides HTTP endpoints for single-turn and multi-turn conversations,
 * session management, and health checks.
 */
class ChatbotAPI {
   public:
    // Allow test class to access private members
    friend class ChatbotAPITest;

    /**
     * @brief Generation parameters for chat responses
     */
    struct GenerationConfig {
        size_t max_length = 100;
        float temperature = 1.0f;
        float top_p = 0.9f;
        size_t top_k = 50;
        std::string strategy = "nucleus";  // "greedy", "beam", "temperature", "top_k", "nucleus"
        size_t beam_width = 4;
    };

    /**
     * @brief Batch request for processing multiple messages
     */
    struct BatchRequest {
        std::vector<std::string> messages;
        std::vector<std::string> session_ids;  // Optional: for batch session processing
        GenerationConfig config;
    };

    /**
     * @brief Batch response containing multiple generated responses
     */
    struct BatchResponse {
        std::vector<std::string> responses;
        std::vector<std::string> session_ids;  // Returned session IDs
        bool success = true;
        std::string error;
        BatchStats stats;  // Efficiency statistics
    };

    /**
     * @brief Constructor
     * @param model Encoder-decoder model for text generation
     * @param tokenizer BPE tokenizer for text processing
     * @param port Port number for HTTP server (default: 8080)
     * @param session_timeout_minutes Session timeout in minutes (default: 30)
     */
    ChatbotAPI(EncoderDecoderModel* model, BPETokenizer* tokenizer, int port = 8080,
               int session_timeout_minutes = 30);

    /**
     * @brief Destructor
     */
    ~ChatbotAPI();
    ChatbotAPI(const ChatbotAPI&) = delete;
    ChatbotAPI& operator=(const ChatbotAPI&) = delete;
    ChatbotAPI(ChatbotAPI&&) = delete;
    ChatbotAPI& operator=(ChatbotAPI&&) = delete;

    /**
     * @brief Start the HTTP server (blocking)
     * @return true if server started successfully
     */
    bool start();

    /**
     * @brief Stop the HTTP server
     */
    void stop();

    /**
     * @brief Check if server is running
     * @return true if server is running
     */
    bool is_running() const {
        return running_;
    }

    /**
     * @brief Set default generation configuration
     * @param config Generation parameters
     */
    void set_generation_config(const GenerationConfig& config) {
        std::lock_guard<std::mutex> lock(config_mutex_);
        default_config_ = config;
    }

    /**
     * @brief Enable RAG for all inference paths
     * @param rag_engine Initialized RAGInference engine with documents already loaded
     */
    void enableRAG(std::shared_ptr<RAGInference> rag_engine);

    /**
     * @brief Generate batch responses (stateless)
     * @param inputs Vector of input messages
     * @param config Generation configuration
     * @return BatchResponse with generated responses and statistics
     */
    BatchResponse generate_batch_responses(const std::vector<std::string>& inputs,
                                           const GenerationConfig& config);

    /**
     * @brief Generate batch responses with sessions (stateful)
     * @param inputs Vector of input messages
     * @param session_ids Vector of session IDs (empty for new sessions)
     * @param config Generation configuration
     * @return BatchResponse with generated responses, session IDs, and statistics
     */
    BatchResponse generate_batch_session_responses(const std::vector<std::string>& inputs,
                                                   const std::vector<std::string>& session_ids,
                                                   const GenerationConfig& config);

    // JSON utilities (public for testing)
    static std::string parse_json_string(const std::string& json, const std::string& key);
    static std::vector<std::string> parse_json_array(const std::string& json,
                                                     const std::string& key);
    static std::string create_json_response(const std::string& response, bool success = true,
                                            const std::string& error = "");
    static std::string create_batch_json_response(const BatchResponse& batch_response);
    static std::string create_error_response(const std::string& error);

   private:
    // HTTP endpoint handlers
    std::string handle_chat(const std::string& request_body);
    std::string handle_chat_session(const std::string& request_body);
    std::string handle_clear_session(const std::string& request_body);
    std::string handle_health();

    // Batch endpoint handlers
    std::string handle_batch_chat(const std::string& request_body);
    std::string handle_batch_chat_session(const std::string& request_body);

    // Session management
    static std::string create_session_id();
    Session* get_or_create_session(const std::string& session_id);
    void cleanup_expired_sessions();
    bool is_session_expired(const Session& session);

    // Text generation
    std::string generate_response(const std::string& input, const GenerationConfig& config);

    // Model components
    EncoderDecoderModel* model_;
    BPETokenizer* tokenizer_;

    // RAG engine (optional; when set, all generate_response calls route through it)
    std::shared_ptr<RAGInference> rag_engine_;

    // Server configuration
    int port_;
    std::chrono::minutes session_timeout_;
    bool running_{false};

    // Session storage (thread-safe)
    std::unordered_map<std::string, std::unique_ptr<Session>> sessions_;
    mutable std::mutex sessions_mutex_;

    // Configuration
    GenerationConfig default_config_;
    mutable std::mutex config_mutex_;

    // HTTP server implementation pointer (forward declaration to avoid including httplib.h here)
    class ServerImpl;
    std::unique_ptr<ServerImpl> server_impl_;
};
