// @adai-status: stable
// @adai-version: 1.4.0
// @adai-reviewed: 2026-11-03

#include "ChatbotCLI.hpp"
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <utility>
#include "GenerationSettings.hpp"

// TODO: See TD-259 in TECHNICAL_DEBT.md - these three helpers have external linkage at global scope; put
// them in an anonymous namespace.
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
    // TODO: See TD-256 in TECHNICAL_DEBT.md - omits /save, /load and /stats, which handle_command() accepts.
    std::cout << COLOR_SYSTEM << "Commands:" << COLOR_RESET << '\n';
    std::cout << "  /help         - Show this help message" << '\n';
    std::cout << "  /clear        - Clear conversation history" << '\n';
    std::cout << "  /settings     - Show current settings" << '\n';
    std::cout << "  /set <param> <value>   - Change a generation setting for your messages:" << '\n';
    std::cout << "                 strategy, length, temperature, top_p, top_k, beam_width" << '\n';
    std::cout << "  /set <param> default  - Go back to the server's default" << '\n';
    std::cout << "  /exit, /quit  - Exit the chatbot" << '\n';
    std::cout << '\n';
}

void ChatbotCLI::print_stats() {
    std::cout << COLOR_SYSTEM << "Stats not available in API mode." << COLOR_RESET << '\n';
}

void ChatbotCLI::print_settings() {
    std::cout << '\n';
    std::cout << COLOR_SYSTEM << "⚙️  Current Settings:" << COLOR_RESET << '\n';
    // TD-254: a setting not changed with /set is the server's default, which the client doesn't know.
    const auto show = [this](const char* label, const char* key, const auto& value) {
        std::cout << "  " << label << ": ";
        if (is_overridden(key)) {
            std::cout << value << '\n';
        } else {
            std::cout << "server default" << '\n';
        }
    };
    show("Strategy", "strategy", generation_strategy);
    show("Max length", "max_length", max_response_length);
    show("Temperature", "temperature", temperature);
    show("Top-p (nucleus)", "top_p", top_p);
    show("Top-k", "top_k", top_k);
    show("Beam width", "beam_width", beam_width);
    std::cout << '\n';
}

void ChatbotCLI::handle_command(const std::string& command) {
    std::string_view cmd_view(command);

    if (command == "/help") {
        print_welcome();
    } else if (command == "/clear") {
        if (!session_id.empty()) {
            // TODO: See TD-257 in TECHNICAL_DEBT.md - session_id is not escaped here (save/load escape it).
            std::string body = R"({"session_id":")" + session_id + "\"}";
            auto res = client->Post("/clear-session", body, "application/json");
            if (res && res->status == 404) {  // TD-232: the server no longer holds it
                session_id.clear();
                std::cout << COLOR_SYSTEM << "✅ No active session on the server" << COLOR_RESET
                          << '\n';
            } else if (res && res->status == 200) {
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

namespace {

// TD-259: the whole value must be the number; std::stoi/std::stof read "10abc" as 10.
bool parse_whole(std::string_view text, int& out) {
    const std::string s(text);
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(s.c_str(), &end, 10);
    if (*end != '\0' || errno == ERANGE || value < INT_MIN || value > INT_MAX) return false;
    out = static_cast<int>(value);
    return true;
}

bool parse_real(std::string_view text, float& out) {
    const std::string s(text);
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const float value = std::strtof(s.c_str(), &end);
    if (*end != '\0' || errno == ERANGE || !std::isfinite(value)) return false;
    out = value;
    return true;
}

}  // namespace

// TD-254: a setting changed here is sent with every message, and the server applies it to that
// message (checked against the same GenerationLimits); a setting never changed, or set back to
// "default", is left to the server. These used to be sent always and ignored by the server.
// TD-259: values must be whole numbers in range ("10abc" and a negative length used to be accepted).
void ChatbotCLI::handle_setting(std::string_view setting) {
    using Limits = adai::GenerationLimits;
    size_t space_pos = setting.find(' ');
    if (space_pos == std::string_view::npos) {
        std::cout << COLOR_ERROR << "❌ Usage: /set <parameter> <value>  (or /set <parameter> default)"
                  << COLOR_RESET << '\n';
        return;
    }

    std::string_view param = setting.substr(0, space_pos);
    std::string_view value = setting.substr(space_pos + 1);

    std::string key;
    if (param == "strategy") {
        key = "strategy";
    } else if (param == "length" || param == "max_length") {
        key = "max_length";
    } else if (param == "temperature" || param == "temp") {
        key = "temperature";
    } else if (param == "top_p" || param == "top-p") {
        key = "top_p";
    } else if (param == "top_k" || param == "top-k") {
        key = "top_k";
    } else if (param == "beam_width" || param == "beam-width") {
        key = "beam_width";
    } else {
        std::cout << COLOR_ERROR << "❌ Unknown parameter: " << param << COLOR_RESET << '\n';
        return;
    }

    if (value == "default") {
        overridden.erase(key);
        std::cout << COLOR_SYSTEM << "✅ " << key << " set back to the server's default" << COLOR_RESET
                  << '\n';
        return;
    }

    const auto invalid = [&](const std::string& rule) {
        std::cout << COLOR_ERROR << "❌ Invalid value '" << value << "' for parameter '" << param
                  << "': " << rule << COLOR_RESET << '\n';
    };
    int whole = 0;
    float real = 0.0f;
    if (key == "strategy") {
        const std::string strategy = adai::canonical_generation_strategy(value);
        if (strategy.empty()) {
            std::cout << COLOR_ERROR << "❌ Invalid strategy '" << value
                      << "'. Valid options: greedy, beam, temperature (or sampling), top_k, nucleus"
                      << COLOR_RESET << '\n';
            return;
        }
        set_generation_strategy(strategy);
        std::cout << COLOR_SYSTEM << "✅ Generation strategy set to: " << strategy << COLOR_RESET << '\n';
    } else if (key == "max_length") {
        if (!parse_whole(value, whole) || whole < Limits::kMinMaxLength || whole > Limits::kMaxMaxLength) {
            return invalid("a whole number from 1 to 4096");
        }
        set_max_response_length(whole);
        std::cout << COLOR_SYSTEM << "✅ Max response length set to: " << max_response_length << COLOR_RESET
                  << '\n';
    } else if (key == "temperature") {
        if (!parse_real(value, real) || real < Limits::kMinTemperature || real > Limits::kMaxTemperature) {
            return invalid("a number from 0 to 2");
        }
        set_temperature(real);
        std::cout << COLOR_SYSTEM << "✅ Temperature set to: " << temperature << COLOR_RESET << '\n';
    } else if (key == "top_p") {
        if (!parse_real(value, real) || real < Limits::kMinTopP || real > Limits::kMaxTopP) {
            return invalid("a number from 0 to 1");
        }
        set_top_p(real);
        std::cout << COLOR_SYSTEM << "✅ Top-p set to: " << top_p << COLOR_RESET << '\n';
    } else if (key == "top_k") {
        if (!parse_whole(value, whole) || whole < Limits::kMinTopK || whole > Limits::kMaxTopK) {
            return invalid("a whole number from 1 to 1000");
        }
        set_top_k(whole);
        std::cout << COLOR_SYSTEM << "✅ Top-k set to: " << top_k << COLOR_RESET << '\n';
    } else {
        if (!parse_whole(value, whole) || whole < Limits::kMinBeamWidth || whole > Limits::kMaxBeamWidth) {
            return invalid("a whole number from 1 to 16");
        }
        set_beam_width(whole);
        std::cout << COLOR_SYSTEM << "✅ Beam width set to: " << beam_width << COLOR_RESET << '\n';
    }
}

std::string ChatbotCLI::generate_response(const std::string& user_input) {
    if (!client) {
        return "Error: Client not initialized";
    }

    // TD-232: session ids are server-issued. Once the server no longer holds ours (expired,
    // evicted, or the server restarted) it answers 404; start a new session and send again.
    httplib::Result res = post_chat_turn(user_input);
    if (res && res->status == 404 && !session_id.empty()) {
        std::cout << COLOR_SYSTEM << "ℹ️  The server no longer has this session; starting a new one"
                  << COLOR_RESET << '\n';
        session_id.clear();
        res = post_chat_turn(user_input);
    }

    // TD-255: an error reply is {"success":false,"error":...} -- with HTTP 400/500 since TD-236, with 200
    // from older servers. Show its error field, never an empty reply or the raw JSON.
    last_turn_failed = true;
    if (!res) {
        return "Error: Connection failed";
    }
    if (res->status == 200 && parse_json_value(res->body, "success") != "false") {
        last_turn_failed = false;
        std::string response = parse_json_value(res->body, "response");

        // Update session ID if provided
        std::string new_sid = parse_json_value(res->body, "session_id");
        if (!new_sid.empty() && new_sid != session_id) {
            session_id = new_sid;
        }

        return response;
    }
    std::stringstream err;
    err << "Error: ";
    if (res->status != 200) {
        err << res->status;
    }
    const std::string error = parse_json_value(res->body, "error");
    if (!error.empty()) {
        err << (res->status != 200 ? " - " : "") << error;
    } else if (res->status == 200) {
        err << "the server reported a failure without a reason";
    }
    const std::string request_id = parse_json_value(res->body, "request_id");
    if (!request_id.empty()) {
        err << " (request id " << request_id << ")";
    }
    return err.str();
}

httplib::Result ChatbotCLI::post_chat_turn(const std::string& user_input) {
    return client->Post("/chat/session", chat_turn_body(user_input), "application/json");
}

std::string ChatbotCLI::chat_turn_body(const std::string& user_input) const {
    std::stringstream ss;
    ss << "{";
    if (!session_id.empty()) {
        // TODO: See TD-257 in TECHNICAL_DEBT.md - session_id is not escaped here.
        ss << R"("session_id":")" << session_id << "\",";
    }
    ss << R"("message":")" << escape_json_string(user_input) << "\"";
    // TD-254: only the settings changed with /set; the server's defaults apply to the rest.
    if (is_overridden("max_length")) ss << ",\"max_length\":" << max_response_length;
    if (is_overridden("temperature")) ss << ",\"temperature\":" << temperature;
    if (is_overridden("top_p")) ss << ",\"top_p\":" << top_p;
    if (is_overridden("top_k")) ss << ",\"top_k\":" << top_k;
    if (is_overridden("beam_width")) ss << ",\"beam_width\":" << beam_width;
    if (is_overridden("strategy")) ss << R"(,"strategy":")" << escape_json_string(generation_strategy) << "\"";
    ss << "}";
    return ss.str();
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

    // TODO: See TD-258 in TECHNICAL_DEBT.md - overwrites any previously saved conversation (including on the
    // automatic /exit save) and isn't atomic; write to a temp file and rename.
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

    auto post_import = [&] {
        std::stringstream ss;
        ss << "{";
        if (!session_id.empty()) {
            ss << R"("session_id":")" << escape_json_string(session_id) << "\",";
        }
        ss << R"("data":")" << escape_json_string(data) << "\"";
        ss << "}";
        return client->Post("/chat/session/import", ss.str(), "application/json");
    };
    auto res = post_import();
    if (res && res->status == 404 && !session_id.empty()) {  // TD-232: import into a new session
        session_id.clear();
        res = post_import();
    }
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

    // TODO: See TD-258 in TECHNICAL_DEBT.md - no SIGINT handling: Ctrl+C exits without the /exit auto-save.
    while (running) {
        std::cout << COLOR_USER << "You: " << COLOR_RESET;
        // TD-253: end of input (Ctrl+D, or piped/redirected stdin running out) ends the session like
        // /exit. getline's result used to go unchecked, so at EOF the empty line was skipped and the
        // loop printed prompts forever (18.6 million in 3 s) without ever saving.
        const bool at_end = !std::getline(std::cin, user_input);
        if (at_end) {
            std::cout << '\n';  // finish the "You: " line
            user_input = "/exit";
        }

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
        if (last_turn_failed) {
            std::cout << COLOR_ERROR << response << COLOR_RESET << '\n' << '\n';  // TD-255
        } else {
            std::cout << response << '\n' << '\n';
        }
    }

    std::cout << COLOR_SYSTEM << "👋 Goodbye!" << COLOR_RESET << '\n';
}
