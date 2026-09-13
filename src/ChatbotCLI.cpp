// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-09-13

#include "ChatbotCLI.hpp"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <utility>

// Helper to escape JSON string
std::string escape_json_string(const std::string& input) {
    std::ostringstream ss;
    for (char c : input) {
        if (c == '"') {
            ss << "\\\"";
        } else if (c == '\\') {
            ss << "\\\\";
        } else if (c == '\b') {
            ss << "\\b";
        } else if (c == '\f') {
            ss << "\\f";
        } else if (c == '\n') {
            ss << "\\n";
        } else if (c == '\r') {
            ss << "\\r";
        } else if (c == '\t') {
            ss << "\\t";
        } else if (static_cast<unsigned char>(c) < 0x20) {
            ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
        } else {
            ss << c;
        }
    }
    return ss.str();
}

std::string unescape_json_string(const std::string& s) {
    std::string res;
    res.reserve(s.length());
    for (size_t i = 0; i < s.length(); ++i) {
        if (s[i] == '\\' && i + 1 < s.length()) {
            switch (s[i + 1]) {
                case '"':
                    res += '"';
                    break;
                case '\\':
                    res += '\\';
                    break;
                case '/':
                    res += '/';
                    break;
                case 'b':
                    res += '\b';
                    break;
                case 'f':
                    res += '\f';
                    break;
                case 'n':
                    res += '\n';
                    break;
                case 'r':
                    res += '\r';
                    break;
                case 't':
                    res += '\t';
                    break;
                default:
                    res += s[i];
                    i--;
                    break;
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
    if (key_pos == std::string::npos) {
        return "";
    }

    size_t colon_pos = json.find(':', key_pos);
    if (colon_pos == std::string::npos) {
        return "";
    }

    size_t value_start = json.find_first_not_of(" \t\n\r", colon_pos + 1);
    if (value_start == std::string::npos) {
        return "";
    }

    if (json[value_start] == '"') {
        // String value
        size_t value_end = value_start + 1;
        bool escaped = false;
        while (value_end < json.length()) {
            if (json[value_end] == '"' && !escaped) {
                break;
            }
            if (json[value_end] == '\\') {
                escaped = !escaped;
            } else {
                escaped = false;
            }
            value_end++;
        }
        return unescape_json_string(json.substr(value_start + 1, value_end - value_start - 1));
    }  // Number/boolean/null
    size_t value_end = json.find_first_of(" ,}", value_start);
    if (value_end == std::string::npos) {
        value_end = json.length();
    }
    return json.substr(value_start, value_end - value_start);
}

ChatbotCLI::ChatbotCLI(std::string server_url, std::string conv_save_file)
    : server_url(std::move(server_url)),
      conversation_save_path(std::move(conv_save_file)),

      generation_strategy("nucleus") {}

ChatbotCLI::~ChatbotCLI() = default;

bool ChatbotCLI::initialize() {
    std::cout << COLOR_SYSTEM << "🤖 Connecting to Chatbot API at " << server_url << "..."
              << COLOR_RESET << '\n';

    try {
        client = std::make_unique<httplib::Client>(server_url);
        // Set timeouts
        client->set_connection_timeout(10, 0);  // 10s connection timeout
        client->set_read_timeout(300, 0);       // 300s read timeout (generation can be VERY slow)
        client->set_write_timeout(30, 0);       // 30s write timeout

        auto res = client->Get("/health");
        if (res && res->status == 200) {
            std::cout << COLOR_SYSTEM << "✅ Connected successfully!" << COLOR_RESET << '\n';
            return true;
        }
        std::cout << COLOR_ERROR << "❌ Failed to connect to server." << COLOR_RESET << '\n';
        if (res) {
            std::cout << "Status: " << res->status << '\n';
        } else {
            std::cout << "Connection Error: " << (client ? "Unknown" : "Client null") << '\n';
        }
        return false;

    } catch (const std::exception& e) {
        std::cout << COLOR_ERROR << "❌ Exception: " << e.what() << COLOR_RESET << '\n';
        return false;
    }
}

void ChatbotCLI::print_welcome() {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << '\n';
    std::cout << "║          🤖 ADAI Chatbot API Client v1.0                 ║" << '\n';
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << '\n';
    std::cout << '\n';
    std::cout << COLOR_SYSTEM << "Commands:" << COLOR_RESET << '\n';
    std::cout << "  /help         - Show this help message" << '\n';
    std::cout << "  /clear        - Clear conversation history" << '\n';
    std::cout << "  /settings     - Show current settings" << '\n';
    std::cout << "  /set <param>  - Change generation parameter" << '\n';
    std::cout << "  /exit, /quit  - Exit the chatbot" << '\n';
    std::cout << '\n';
}

void ChatbotCLI::print_stats() {
    std::cout << COLOR_SYSTEM << "Stats not available in API mode." << COLOR_RESET << '\n';
}

void ChatbotCLI::print_settings() {
    std::cout << '\n';
    std::cout << COLOR_SYSTEM << "⚙️  Current Settings:" << COLOR_RESET << '\n';
    std::cout << "  Strategy: " << generation_strategy << '\n';
    std::cout << "  Max length: " << max_response_length << '\n';
    std::cout << "  Temperature: " << temperature << '\n';
    std::cout << "  Top-p (nucleus): " << top_p << '\n';
    std::cout << "  Top-k: " << top_k << '\n';
    std::cout << "  Beam width: " << beam_width << '\n';
    std::cout << '\n';
}

void ChatbotCLI::handle_command(const std::string& command) {
    std::string_view cmd_view(command);

    if (command == "/help") {
        print_welcome();
    } else if (command == "/clear") {
        if (!session_id.empty()) {
            std::string body = R"({"session_id":")" + session_id + "\"}";
            auto res = client->Post("/clear-session", body, "application/json");
            if (res && res->status == 200) {
                std::cout << COLOR_SYSTEM << "✅ Conversation history cleared (Session "
                          << session_id << ")" << COLOR_RESET << '\n';
            } else {
                std::cout << COLOR_ERROR << "❌ Failed to clear session" << COLOR_RESET << '\n';
            }
        } else {
            std::cout << COLOR_SYSTEM << "✅ No active session to clear" << COLOR_RESET << '\n';
        }
    } else if (command == "/stats") {
        print_stats();
    } else if (command == "/settings") {
        print_settings();
    } else if (cmd_view.size() > 5 && cmd_view.substr(0, 5) == "/set ") {
        handle_setting(cmd_view.substr(5));
    } else if (command == "/save") {
        save_conversation();
    } else if (command == "/load") {
        load_conversation();
    } else {
        std::cout << COLOR_ERROR << "❓ Unknown command. Type /help for available commands."
                  << COLOR_RESET << '\n';
    }
}

void ChatbotCLI::handle_setting(std::string_view setting) {
    size_t space_pos = setting.find(' ');
    if (space_pos == std::string_view::npos) {
        std::cout << COLOR_ERROR << "❌ Usage: /set <parameter> <value>" << COLOR_RESET << '\n';
        return;
    }

    std::string_view param = setting.substr(0, space_pos);
    std::string_view value = setting.substr(space_pos + 1);

    if (param == "strategy") {
        static const std::vector<std::string> VALID_STRATEGIES = {"greedy", "beam", "sampling",
                                                                  "top-k", "nucleus"};
        std::string strat_val = std::string(value);
        bool valid = false;
        for (const auto& s : VALID_STRATEGIES) {
            if (s == strat_val) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            std::cout << COLOR_ERROR << "❌ Invalid strategy '" << value
                      << "'. Valid options: greedy, beam, sampling, top-k, nucleus" << COLOR_RESET
                      << '\n';
            return;
        }
        generation_strategy = strat_val;
        std::cout << COLOR_SYSTEM << "✅ Generation strategy set to: " << value << COLOR_RESET
                  << '\n';
    } else if (param == "length" || param == "max_length" || param == "temperature" ||
              param == "temp" || param == "top_p" || param == "top-p" || param == "top_k" ||
              param == "top-k" || param == "beam_width" || param == "beam-width") {
        // TD-072 (fixed): std::stoi/std::stof on a malformed value (e.g. a
        // typo in "/set length abc") used to throw uncaught out of this
        // function, propagate through run()'s main loop, and hit main()'s
        // top-level catch — a clean exit, but one that silently ends the
        // entire interactive session (losing session_id/conversation state)
        // over a single mistyped command, unlike a one-shot CLI tool where
        // the same unguarded stoi pattern just means "re-run the command."
        try {
            if (param == "length" || param == "max_length") {
                max_response_length = std::stoi(std::string(value));
                std::cout << COLOR_SYSTEM << "✅ Max response length set to: "
                          << max_response_length << COLOR_RESET << '\n';
            } else if (param == "temperature" || param == "temp") {
                temperature = std::stof(std::string(value));
                std::cout << COLOR_SYSTEM << "✅ Temperature set to: " << temperature
                          << COLOR_RESET << '\n';
            } else if (param == "top_p" || param == "top-p") {
                top_p = std::stof(std::string(value));
                std::cout << COLOR_SYSTEM << "✅ Top-p set to: " << top_p << COLOR_RESET << '\n';
            } else if (param == "top_k" || param == "top-k") {
                top_k = std::stoi(std::string(value));
                std::cout << COLOR_SYSTEM << "✅ Top-k set to: " << top_k << COLOR_RESET << '\n';
            } else {
                beam_width = std::stoi(std::string(value));
                std::cout << COLOR_SYSTEM << "✅ Beam width set to: " << beam_width << COLOR_RESET
                          << '\n';
            }
        } catch (const std::exception&) {
            std::cout << COLOR_ERROR << "❌ Invalid value '" << value << "' for parameter '"
                      << param << "'" << COLOR_RESET << '\n';
        }
    } else {
        std::cout << COLOR_ERROR << "❌ Unknown parameter: " << param << COLOR_RESET << '\n';
    }
}

std::string ChatbotCLI::generate_response(const std::string& user_input) {
    if (!client) {
        return "Error: Client not initialized";
    }

    std::stringstream ss;
    ss << "{";
    if (!session_id.empty()) {
        ss << R"("session_id":")" << session_id << "\",";
    }
    ss << R"("message":")" << escape_json_string(user_input) << "\",";
    ss << "\"max_length\":" << max_response_length << ",";
    ss << "\"temperature\":" << temperature << ",";
    ss << "\"top_p\":" << top_p << ",";
    ss << "\"top_k\":" << top_k << ",";
    ss << "\"beam_width\":" << beam_width << ",";
    ss << R"("strategy":")" << generation_strategy << "\"";
    ss << "}";

    std::string body = ss.str();

    // Choose endpoint based on session requirement
    std::string endpoint = "/chat/session";

    auto res = client->Post(endpoint, body, "application/json");

    if (res && res->status == 200) {
        std::string response = parse_json_value(res->body, "response");

        // Update session ID if provided
        std::string new_sid = parse_json_value(res->body, "session_id");
        if (!new_sid.empty() && new_sid != session_id) {
            session_id = new_sid;
        }

        return response;
    }
    std::stringstream err;
    err << "Error: " << (res ? std::to_string(res->status) : "Connection failed");
    if (res && !res->body.empty()) {
        err << " - " << res->body;
    }
    return err.str();
}

void ChatbotCLI::save_conversation() {
    // TD-053: session state lives server-side, not in this object -- "saving" means asking the
    // server to export it, then writing the result to our local file.
    if (session_id.empty()) {
        std::cout << COLOR_ERROR << "❌ Nothing to save yet — send a message first."
                  << COLOR_RESET << '\n';
        return;
    }
    if (!client) {
        std::cout << COLOR_ERROR << "❌ Client not initialized" << COLOR_RESET << '\n';
        return;
    }

    std::string body = R"({"session_id":")" + escape_json_string(session_id) + "\"}";
    auto res = client->Post("/chat/session/export", body, "application/json");
    if (!res) {
        std::cout << COLOR_ERROR << "❌ Failed to reach server to export conversation"
                  << COLOR_RESET << '\n';
        return;
    }
    if (parse_json_value(res->body, "success") != "true") {
        std::string error = parse_json_value(res->body, "error");
        std::cout << COLOR_ERROR << "❌ " << (error.empty() ? "Export failed" : error)
                  << COLOR_RESET << '\n';
        return;
    }

    std::ofstream out(conversation_save_path);
    if (!out.is_open()) {
        std::cout << COLOR_ERROR << "❌ Failed to open '" << conversation_save_path
                  << "' for writing" << COLOR_RESET << '\n';
        return;
    }
    out << parse_json_value(res->body, "data");
    out.close();

    std::cout << COLOR_SYSTEM << "✅ Conversation saved to " << conversation_save_path
              << COLOR_RESET << '\n';
}

void ChatbotCLI::load_conversation() {
    // TD-053: reads our local file, then asks the server to import it into a session (a fresh
    // one if we don't have one yet) -- there is no local ConversationContext of our own to
    // populate.
    std::ifstream in(conversation_save_path);
    if (!in.is_open()) {
        std::cout << COLOR_ERROR << "❌ No saved conversation found at '"
                  << conversation_save_path << "'" << COLOR_RESET << '\n';
        return;
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    in.close();
    std::string data = buf.str();
    if (data.empty()) {
        std::cout << COLOR_ERROR << "❌ Saved conversation file '" << conversation_save_path
                  << "' is empty" << COLOR_RESET << '\n';
        return;
    }
    if (!client) {
        std::cout << COLOR_ERROR << "❌ Client not initialized" << COLOR_RESET << '\n';
        return;
    }

    std::stringstream ss;
    ss << "{";
    if (!session_id.empty()) {
        ss << R"("session_id":")" << escape_json_string(session_id) << "\",";
    }
    ss << R"("data":")" << escape_json_string(data) << "\"";
    ss << "}";

    auto res = client->Post("/chat/session/import", ss.str(), "application/json");
    if (!res) {
        std::cout << COLOR_ERROR << "❌ Failed to reach server to import conversation"
                  << COLOR_RESET << '\n';
        return;
    }
    if (parse_json_value(res->body, "success") != "true") {
        std::string error = parse_json_value(res->body, "error");
        std::cout << COLOR_ERROR << "❌ " << (error.empty() ? "Import failed" : error)
                  << COLOR_RESET << '\n';
        return;
    }

    std::string new_sid = parse_json_value(res->body, "session_id");
    if (!new_sid.empty()) {
        session_id = new_sid;
    }

    std::cout << COLOR_SYSTEM << "✅ Conversation loaded from " << conversation_save_path;
    std::string message_count = parse_json_value(res->body, "message_count");
    if (!message_count.empty()) {
        std::cout << " (" << message_count << " messages)";
    }
    std::cout << COLOR_RESET << '\n';
}

void ChatbotCLI::run() {
    if (!initialize()) {
        std::cerr << COLOR_ERROR << "Failed to initialize chatbot!" << COLOR_RESET << '\n';
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

        if (user_input.empty()) {
            continue;
        }

        if (user_input == "/exit" || user_input == "/quit") {
            // TD-053: automatic save-on-exit, matching chatbot-guide.md's documented behavior.
            // Silently skipped (not an error) when no conversation ever started — save_conversation()
            // itself would print a "nothing to save yet" message otherwise, which is the right
            // message for an explicit /save but noise on every exit of a session that never sent
            // a message.
            if (!session_id.empty()) {
                save_conversation();
            }
            running = false;
            continue;
        }

        if (user_input[0] == '/') {
            handle_command(user_input);
            continue;
        }

        std::cout << COLOR_BOT << "Bot: " << COLOR_RESET;
        std::cout.flush();  // Ensure "Bot:" prints before response

        // If message is very short, response might be fast, but if long, we should wait
        std::string response = generate_response(user_input);
        std::cout << response << '\n' << '\n';
    }

    std::cout << COLOR_SYSTEM << "👋 Goodbye!" << COLOR_RESET << '\n';
}
