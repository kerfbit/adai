#include "ChatbotAPI.hpp"
#include <httplib.h>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <iostream>

// ServerImpl using cpp-httplib
class ChatbotAPI::ServerImpl {
public:
    httplib::Server server;
};

ChatbotAPI::ChatbotAPI(EncoderDecoderModel* model,
                       BPETokenizer* tokenizer,
                       int port,
                       int session_timeout_minutes)
    : model_(model),
      tokenizer_(tokenizer),
      port_(port),
      session_timeout_(session_timeout_minutes),
      running_(false),
      server_impl_(std::make_unique<ServerImpl>()) {
    
    // Set up HTTP endpoints
    
    // POST /chat - Single-turn conversation
    server_impl_->server.Post("/chat", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_chat(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });

    // POST /chat/session - Multi-turn conversation with session
    server_impl_->server.Post("/chat/session", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_chat_session(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });

    // POST /clear-session - Clear conversation history for a session
    server_impl_->server.Post("/clear-session", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_clear_session(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });

    // GET /health - Health check
    server_impl_->server.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        std::string response = handle_health();
        res.set_content(response, "application/json");
        res.status = 200;
    });

    std::cout << "ChatbotAPI initialized on port " << port_ << std::endl;
}

ChatbotAPI::~ChatbotAPI() {
    stop();
}

bool ChatbotAPI::start() {
    if (running_) {
        std::cerr << "Server is already running" << std::endl;
        return false;
    }

    std::cout << "Starting chatbot API server on port " << port_ << "..." << std::endl;
    running_ = true;
    
    // Start periodic session cleanup (every 5 minutes)
    // Note: In production, this should be a separate background thread
    
    bool result = server_impl_->server.listen("0.0.0.0", port_);
    running_ = false;
    return result;
}

void ChatbotAPI::stop() {
    if (running_) {
        std::cout << "Stopping chatbot API server..." << std::endl;
        server_impl_->server.stop();
        running_ = false;
    }
}

// ============================================================================
// HTTP Endpoint Handlers
// ============================================================================

std::string ChatbotAPI::handle_chat(const std::string& request_body) {
    // Parse request
    std::string message = parse_json_string(request_body, "message");
    if (message.empty()) {
        return create_error_response("Missing 'message' field in request");
    }

    // Get generation config from request or use defaults
    GenerationConfig config;
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        config = default_config_;
    }

    // Generate response
    std::string response = generate_response(message, config);
    
    return create_json_response(response);
}

std::string ChatbotAPI::handle_chat_session(const std::string& request_body) {
    // Parse request
    std::string session_id = parse_json_string(request_body, "session_id");
    std::string message = parse_json_string(request_body, "message");
    
    if (message.empty()) {
        return create_error_response("Missing 'message' field in request");
    }

    // Get or create session
    Session* session = get_or_create_session(session_id);
    
    // Add user message to conversation context
    session->context->add_user_message(message);
    session->last_access = std::chrono::steady_clock::now();

    // Format context for model
    std::string formatted_context = session->context->format_for_model();

    // Get generation config
    GenerationConfig config;
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        config = default_config_;
    }

    // Generate response
    std::string response = generate_response(formatted_context, config);

    // Add assistant response to conversation context
    session->context->add_assistant_message(response);

    // Create JSON response with session_id
    std::ostringstream oss;
    oss << "{\"success\":true,\"response\":\"";
    // Escape quotes in response
    for (char c : response) {
        if (c == '"') oss << "\\\"";
        else if (c == '\\') oss << "\\\\";
        else if (c == '\n') oss << "\\n";
        else if (c == '\r') oss << "\\r";
        else if (c == '\t') oss << "\\t";
        else oss << c;
    }
    oss << "\",\"session_id\":\"" << session_id << "\"}";
    
    return oss.str();
}

std::string ChatbotAPI::handle_clear_session(const std::string& request_body) {
    std::string session_id = parse_json_string(request_body, "session_id");
    
    if (session_id.empty()) {
        return create_error_response("Missing 'session_id' field in request");
    }

    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        it->second->context->clear();
        return "{\"success\":true,\"message\":\"Session cleared\"}";
    } else {
        return create_error_response("Session not found");
    }
}

std::string ChatbotAPI::handle_health() {
    cleanup_expired_sessions();
    
    size_t active_sessions;
    {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        active_sessions = sessions_.size();
    }
    
    std::ostringstream oss;
    oss << "{\"status\":\"ok\",\"active_sessions\":" << active_sessions << "}";
    return oss.str();
}

// ============================================================================
// Session Management
// ============================================================================

std::string ChatbotAPI::create_session_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; ++i) {
        oss << std::setw(1) << dis(gen);
    }
    return oss.str();
}

Session* ChatbotAPI::get_or_create_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    
    // If session_id is empty or not found, create new session
    if (session_id.empty() || sessions_.find(session_id) == sessions_.end()) {
        std::string new_id = session_id.empty() ? create_session_id() : session_id;
        sessions_[new_id] = std::make_unique<Session>();
        return sessions_[new_id].get();
    }
    
    // Update last access time and return existing session
    Session* session = sessions_[session_id].get();
    session->last_access = std::chrono::steady_clock::now();
    return session;
}

void ChatbotAPI::cleanup_expired_sessions() {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    
    auto it = sessions_.begin();
    while (it != sessions_.end()) {
        if (is_session_expired(*it->second)) {
            std::cout << "Removing expired session: " << it->first << std::endl;
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
}

bool ChatbotAPI::is_session_expired(const Session& session) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - session.last_access);
    return elapsed >= session_timeout_;
}

// ============================================================================
// JSON Utilities (Simple parser without external dependencies)
// ============================================================================

std::string ChatbotAPI::parse_json_string(const std::string& json, const std::string& key) {
    // Simple JSON string parser (handles basic cases)
    // Format: {"key":"value"}
    std::string search_key = "\"" + key + "\"";
    size_t key_pos = json.find(search_key);
    if (key_pos == std::string::npos) {
        return "";
    }

    size_t colon_pos = json.find(':', key_pos);
    if (colon_pos == std::string::npos) {
        return "";
    }

    size_t value_start = json.find('"', colon_pos);
    if (value_start == std::string::npos) {
        return "";
    }
    value_start++; // Skip opening quote

    size_t value_end = value_start;
    while (value_end < json.length()) {
        if (json[value_end] == '"' && (value_end == 0 || json[value_end - 1] != '\\')) {
            break;
        }
        value_end++;
    }

    if (value_end >= json.length()) {
        return "";
    }

    std::string value = json.substr(value_start, value_end - value_start);
    
    // Unescape common escape sequences
    std::string unescaped;
    for (size_t i = 0; i < value.length(); ++i) {
        if (value[i] == '\\' && i + 1 < value.length()) {
            char next = value[i + 1];
            if (next == 'n') { unescaped += '\n'; i++; }
            else if (next == 'r') { unescaped += '\r'; i++; }
            else if (next == 't') { unescaped += '\t'; i++; }
            else if (next == '"') { unescaped += '"'; i++; }
            else if (next == '\\') { unescaped += '\\'; i++; }
            else unescaped += value[i];
        } else {
            unescaped += value[i];
        }
    }
    
    return unescaped;
}

std::string ChatbotAPI::create_json_response(const std::string& response, bool success, const std::string& error) {
    std::ostringstream oss;
    oss << "{\"success\":" << (success ? "true" : "false");
    
    if (success) {
        oss << ",\"response\":\"";
        // Escape quotes and special characters
        for (char c : response) {
            if (c == '"') oss << "\\\"";
            else if (c == '\\') oss << "\\\\";
            else if (c == '\n') oss << "\\n";
            else if (c == '\r') oss << "\\r";
            else if (c == '\t') oss << "\\t";
            else oss << c;
        }
        oss << "\"";
    } else {
        oss << ",\"error\":\"" << error << "\"";
    }
    
    oss << "}";
    return oss.str();
}

std::string ChatbotAPI::create_error_response(const std::string& error) {
    return create_json_response("", false, error);
}

// ============================================================================
// Text Generation
// ============================================================================

std::string ChatbotAPI::generate_response(const std::string& input, const GenerationConfig& config) {
    try {
        // Tokenize input
        std::vector<int> input_tokens = tokenizer_->encode(input);
        
        // Create a TextGenerator with appropriate configuration
        TextGenerator::GenerationConfig gen_config;
        gen_config.max_length = config.max_length;
        gen_config.temperature = config.temperature;
        gen_config.top_p = config.top_p;
        gen_config.top_k = config.top_k;
        gen_config.num_beams = config.beam_width;
        
        TextGenerator generator(gen_config, 0);  // seed=0 for random
        
        // Create model forward function (uses encoder-decoder model)
        auto model_fn = [this, &input_tokens](const std::vector<int>& decoder_tokens) -> Matrix {
            // For encoder-decoder model: encode input once, then use decoder
            // This is a simplified version - in practice, you might cache encoder output
            return model_->forward(input_tokens, decoder_tokens);
        };
        
        // Generate tokens based on strategy
        std::vector<int> generated_tokens;
        
        if (config.strategy == "greedy") {
            generated_tokens = generator.generate_greedy(model_fn, {});
        } else if (config.strategy == "beam") {
            generated_tokens = generator.generate_beam_search(model_fn, {}, config.beam_width);
        } else if (config.strategy == "temperature") {
            generated_tokens = generator.generate_sampling(model_fn, {}, config.temperature);
        } else if (config.strategy == "top_k") {
            generated_tokens = generator.generate_top_k(model_fn, {}, config.top_k);
        } else if (config.strategy == "nucleus") {
            generated_tokens = generator.generate_nucleus(model_fn, {}, config.top_p);
        } else {
            // Default to nucleus sampling
            generated_tokens = generator.generate_nucleus(model_fn, {}, config.top_p);
        }
        
        // Detokenize generated tokens
        std::string response = tokenizer_->decode(generated_tokens);
        return response;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Generation failed: ") + e.what());
    }
}
