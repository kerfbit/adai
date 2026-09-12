#pragma once

// @adai-status: beta        (capped by TD-033 — generate_response() never uses GPU-resident decode, see TECHNICAL_DEBT.md)
// @adai-version: 0.9.5
// @adai-reviewed: 2026-09-12


#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "BPETokenizer.hpp"
#include "BatchProcessor.hpp"
#include "ConversationContext.hpp"
#include "EncoderDecoderModel.hpp"
#include "BatchedInferenceEngine.hpp"
#include "PerformanceProfiler.hpp"
#include "PipelineInferenceEngine.hpp"
#include "RAGInference.hpp"
#include "SpeculativeDecoding.hpp"
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
     * @param draft_model Optional second model (same architecture/tokenizer, different weights)
     *        used as the fast "draft" model for speculative decoding (TD-038). nullptr
     *        (default) disables it — generate_response() then always uses the normal
     *        strategy-based path, unchanged from before this parameter existed.
     * @param speculative_num_candidates Draft-model candidates per verification round; ignored
     *        when draft_model is nullptr.
     */
    ChatbotAPI(EncoderDecoderModel* model, BPETokenizer* tokenizer, int port = 8080,
               int session_timeout_minutes = 30, EncoderDecoderModel* draft_model = nullptr,
               int speculative_num_candidates = 4);

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
     * @brief Enable/disable the GET /admin/profile endpoint (TD-038).
     *
     * generate_response() is always internally timed via PerformanceProfiler.hpp's Profiler
     * (start()/stop() are cheap map lookups next to actual model inference — negligible
     * overhead regardless of this setting). What this flag actually gates is exposure: with
     * profiling disabled (the default), GET /admin/profile reports itself disabled rather than
     * returning timing data, the same opt-in stance this codebase's daemons already take for
     * other introspection/control surfaces (e.g. METRICS_API_ALLOW_CONTROL).
     */
    void enable_profiling(bool enabled = true) {
        profiling_enabled_ = enabled;
    }

    /**
     * @brief Enable batched/queued inference mode (TD-038).
     *
     * generate_response() normally runs generation inline on the HTTP handler's own thread.
     * With this enabled, it instead submits the request to a background BatchedInferenceEngine
     * worker thread and blocks on the returned future — real queueing/batching behavior, not a
     * no-op wrapper. Note this mode always uses BatchedInferenceEngine's combined
     * top-k/top-p/temperature sampling (TextGenerator::generate_text()'s only decoding path);
     * GenerationConfig::strategy ("greedy"/"beam"/etc.) is not consulted while this is enabled,
     * the same way rag_engine_/draft_model_ above each override strategy for their own reasons.
     */
    void enable_batched_inference(const BatchedInferenceConfig& config = BatchedInferenceConfig());

    /**
     * @brief Enable two-stage pipeline inference mode (TD-038).
     *
     * generate_response() normally runs generation inline on the HTTP handler's own thread via
     * TextGenerator (which supports temperature/top_p/top_k/beam/repetition-penalty). This mode
     * instead routes through a real StandardPipelineEngine (PipelineInferenceEngine.hpp) with
     * dedicated encoder and decoder worker threads overlapping across requests. That engine
     * reimplements its own generation loop directly against the model's raw encoder/decoder/
     * lm_head components rather than delegating to TextGenerator, and that loop is
     * **always greedy** — GenerationConfig::temperature/top_p/top_k/strategy are not consulted
     * while this is enabled, the same way rag_engine_/draft_model_/batched_engine_ above each
     * override generation for their own reasons.
     *
     * vocab_path must name the exact same vocabulary file the rest of this ChatbotAPI instance
     * was built with: PipelineInferenceEngine's encoder stage calls the model's own LLMEncoder::
     * encode(text), which tokenizes internally via LLMEncoder's own private BPETokenizer member
     * — a second tokenizer instance, entirely separate from ChatbotAPI's tokenizer_ (used
     * everywhere else, including this same pipeline's own decode() step). Loading a different or
     * mismatched vocab here would silently produce token IDs incompatible with the model's
     * trained embeddings. load_tokenizer_vocab() throws VocabularyFileError on a bad path; this
     * method does not catch it — callers decide whether a load failure should disable the mode
     * (see ChatbotAPIServer.cpp's tolerant handling of --draft-model for the established
     * precedent) or abort startup.
     */
    void enable_pipeline_inference(const std::string& vocab_path,
                                   const PipelineConfig& config = PipelineConfig());

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
    /**
     * @brief Escape a string for embedding inside a JSON string literal
     *        (quotes, backslashes, and the standard control-character escapes).
     *
     * TD-063 (fixed): every JSON-building function in this file used to escape
     * some string fields (e.g. the "response"/"responses" fields) char-by-char
     * inline but not others (session_id, the "error" field in both
     * create_json_response()'s failure branch and create_batch_json_response()'s
     * failure branch, and session_ids[] in the batch response) — all of which
     * can carry client-controlled or exception-message text (e.g. every
     * top-level handler's catch block passes e.what() straight to
     * create_error_response()). A crafted session_id or a message that
     * triggers an exception whose text embeds a '"' could inject additional
     * JSON fields into the response or break its structure entirely. This
     * helper is now the single escaping implementation used everywhere in
     * this file.
     */
    static std::string escape_json_string(const std::string& s);

   private:
    // HTTP endpoint handlers
    std::string handle_chat(const std::string& request_body);
    std::string handle_chat_session(const std::string& request_body);
    std::string handle_clear_session(const std::string& request_body);
    std::string handle_health();
    std::string handle_profile();

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

    // Speculative decoding (TD-038): non-owning, non-null only when a --draft-model was
    // configured. When set, generate_response() always uses speculative decoding regardless of
    // GenerationConfig::strategy — a server-level decoding mechanism, the same way rag_engine_
    // above takes over every generate_response call once set rather than being a per-request
    // opt-in/out.
    EncoderDecoderModel* draft_model_{nullptr};
    int speculative_num_candidates_{4};

    // Batched/queued inference (TD-038): owned (unlike model_/tokenizer_/draft_model_, which
    // outlive this object and are never owned by it) since the engine's own background worker
    // thread's lifetime needs to be tied to this object's, matching BatchedInferenceEngine's own
    // RAII shutdown-on-destruct design. nullptr (default) means generate_response() runs inline
    // on the caller's own thread, unchanged from before this member existed.
    std::unique_ptr<BatchedInferenceEngine> batched_engine_;

    // Pipeline inference (TD-038): owned, same lifetime reasoning as batched_engine_ above (its
    // own encoder/decoder worker threads must not outlive this object). nullptr (default) means
    // generate_response() is unaffected by this mode's existence.
    std::unique_ptr<StandardPipelineEngine> pipeline_engine_;

    // RAG engine (optional; when set, all generate_response calls route through it)
    std::shared_ptr<RAGInference> rag_engine_;

    // Performance profiling (TD-038) — always timed internally; enable_profiling() only gates
    // whether GET /admin/profile reports the results.
    Profiler profiler_;
    bool profiling_enabled_{false};

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
