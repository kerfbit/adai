#include "ChatbotCLI.hpp"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>

// Helper to escape JSON string
std::string escape_json_string(const std::string& input) {
    std::ostringstream ss;
    for (char c : input) {
        if (c == '"') ss << "\\\"";
        else if (c == '\\') ss << "\\\\";
        else if (c == '\b') ss << "\\b";
        else if (c == '\f') ss << "\\f";
        else if (c == '\n') ss << "\\n";
        else if (c == '\r') ss << "\\r";
        else if (c == '\t') ss << "\\t";
        else if (static_cast<unsigned char>(c) < 0x20) 
            ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
        else ss << c;
    }
    return ss.str();
}

std::string unescape_json_string(const std::string& s) {
    std::string res;
    res.reserve(s.length());
    for (size_t i = 0; i < s.length(); ++i) {
        if (s[i] == '\\' && i + 1 < s.length()) {
            switch (s[i + 1]) {
                case '"': res += '"'; break;
                case '\\': res += '\\'; break;
                case '/': res += '/'; break;
                case 'b': res += '\b'; break;
                case 'f': res += '\f'; break;
                case 'n': res += '\n'; break;
                case 'r': res += '\r'; break;
                case 't': res += '\t'; break;
                default: res += s[i]; i--; break; 
            }
            i++;
        } else {
            res += s[i];
        }
    }
    return res;
}

std::string parse_json_value(const std::string& json, const std::string& key) {
    size_t key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return "";
    
    size_t colon_pos = json.find(":", key_pos);
    if (colon_pos == std::string::npos) return "";
    
    size_t value_start = json.find_first_not_of(" \t\n\r", colon_pos + 1);
    if (value_start == std::string::npos) return "";
    
    if (json[value_start] == '"') {
        // String value
        size_t value_end = value_start + 1;
        bool escaped = false;
        while (value_end < json.length()) {
            if (json[value_end] == '"' && !escaped) break;
            if (json[value_end] == '\\') escaped = !escaped;
            else escaped = false;
            value_end++;
        }
        return unescape_json_string(json.substr(value_start + 1, value_end - value_start - 1));
    } else {
        // Number/boolean/null
        size_t value_end = json.find_first_of(" ,}", value_start);
        if (value_end == std::string::npos) value_end = json.length();
        return json.substr(value_start, value_end - value_start);
    }
}

ChatbotCLI::ChatbotCLI(const std::string& server_url,
                       const std::string& conv_save_file)
    : server_url(server_url),
      conversation_save_path(conv_save_file),
      max_response_length(100),
      temperature(1.0f),
      top_p(0.9f),
      top_k(50),
      beam_width(5),
      generation_strategy("nucleus") {
}

ChatbotCLI::~ChatbotCLI() = default;

bool ChatbotCLI::initialize() {
    std::cout << COLOR_SYSTEM << "🤖 Connecting to Chatbot API at " << server_url << "..." << COLOR_RESET << std::endl;
    
    try {
        client = std::make_unique<httplib::Client>(server_url);
        // Set timeouts
        client->set_connection_timeout(10, 0); // 10s connection timeout
        client->set_read_timeout(300, 0);      // 300s read timeout (generation can be VERY slow)
        client->set_write_timeout(30, 0);     // 30s write timeout
        
        auto res = client->Get("/health");
        if (res && res->status == 200) {
            std::cout << COLOR_SYSTEM << "✅ Connected successfully!" << COLOR_RESET << std::endl;
            return true;
        } else {
            std::cout << COLOR_ERROR << "❌ Failed to connect to server." << COLOR_RESET << std::endl;
            if (res) std::cout << "Status: " << res->status << std::endl;
            else std::cout << "Connection Error: " << (client ? "Unknown" : "Client null") << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << COLOR_ERROR << "❌ Exception: " << e.what() << COLOR_RESET << std::endl;
        return false;
    }
}

void ChatbotCLI::print_welcome() {
        std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║          🤖 ADAI Chatbot API Client v1.0                 ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << std::endl;
        std::cout << COLOR_SYSTEM << "Commands:" << COLOR_RESET << std::endl;
        std::cout << "  /help         - Show this help message" << std::endl;
        std::cout << "  /clear        - Clear conversation history" << std::endl;
        std::cout << "  /settings     - Show current settings" << std::endl;
        std::cout << "  /set <param>  - Change generation parameter" << std::endl;
        std::cout << "  /exit, /quit  - Exit the chatbot" << std::endl;
        std::cout << std::endl;
}

void ChatbotCLI::print_stats() {
    std::cout << COLOR_SYSTEM << "Stats not available in API mode." << COLOR_RESET << std::endl;
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
            if (!session_id.empty()) {
                std::string body = "{\"session_id\":\"" + session_id + "\"}";
                auto res = client->Post("/clear-session", body, "application/json");
                if (res && res->status == 200) {
                     std::cout << COLOR_SYSTEM << "✅ Conversation history cleared (Session " << session_id << ")" << COLOR_RESET << std::endl;
                } else {
                     std::cout << COLOR_ERROR << "❌ Failed to clear session" << COLOR_RESET << std::endl;
                }
            } else {
                std::cout << COLOR_SYSTEM << "✅ No active session to clear" << COLOR_RESET << std::endl;
            }
        } else if (command == "/stats") {
            print_stats();
        } else if (command == "/settings") {
            print_settings();
        } else if (cmd_view.size() > 5 && cmd_view.substr(0, 5) == "/set ") {
            handle_setting(cmd_view.substr(5));
        } else if (command == "/save" || command == "/load") {
            std::cout << COLOR_ERROR << "❌ Save/Load not supported in API client mode yet." << COLOR_RESET << std::endl;
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
             generation_strategy = std::string(value);
             std::cout << COLOR_SYSTEM << "✅ Generation strategy set to: " << value << COLOR_RESET << std::endl;
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
        }
}

std::string ChatbotCLI::generate_response(const std::string& user_input) {
    if (!client) return "Error: Client not initialized";

    std::stringstream ss;
    ss << "{";
    if (!session_id.empty()) {
        ss << "\"session_id\":\"" << session_id << "\",";
    }
    ss << "\"message\":\"" << escape_json_string(user_input) << "\",";
    ss << "\"max_length\":" << max_response_length << ",";
    ss << "\"temperature\":" << temperature << ",";
    ss << "\"top_p\":" << top_p << ",";
    ss << "\"top_k\":" << top_k << ",";
    ss << "\"beam_width\":" << beam_width << ",";
    ss << "\"strategy\":\"" << generation_strategy << "\"";
    ss << "}";

    std::string body = ss.str();
    
    // Choose endpoint based on session requirement
    std::string endpoint = "/chat/session"; 
    
    auto res = client->Post(endpoint.c_str(), body, "application/json");
    
    if (res && res->status == 200) {
        std::string response = parse_json_value(res->body, "response");
        
        // Update session ID if provided
        std::string new_sid = parse_json_value(res->body, "session_id");
        if (!new_sid.empty() && new_sid != session_id) {
            session_id = new_sid;
        }
        
        return response;
    } else {
        std::stringstream err;
        err << "Error: " << (res ? std::to_string(res->status) : "Connection failed");
        if (res && res->body.size() > 0) {
             err << " - " << res->body;
        }
        return err.str();
    }
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
        std::cout << COLOR_USER << "You: " << COLOR_RESET;
        std::getline(std::cin, user_input);

        user_input.erase(0, user_input.find_first_not_of(" \t\n\r"));
        user_input.erase(user_input.find_last_not_of(" \t\n\r") + 1);

        if (user_input.empty()) continue;

        if (user_input == "/exit" || user_input == "/quit") {
            running = false;
            continue;
        }

        if (user_input[0] == '/') {
            handle_command(user_input);
            continue;
        }

        std::cout << COLOR_BOT << "Bot: " << COLOR_RESET;
        std::cout.flush(); // Ensure "Bot:" prints before response
        
        // If message is very short, response might be fast, but if long, we should wait
        std::string response = generate_response(user_input);
        std::cout << response << std::endl << std::endl;
    }

    std::cout << COLOR_SYSTEM << "👋 Goodbye!" << COLOR_RESET << std::endl;
}
