// @adai-status: stable
// @adai-version: 1.7.0
// @adai-reviewed: 2026-11-04

#include "ChatbotCLI.hpp"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include "GenerationSettings.hpp"

namespace {

// ANSI colours for the terminal (TD-259: constants here rather than macros in the header).
constexpr const char* COLOR_RESET = "\033[0m";
constexpr const char* COLOR_USER = "\033[1;36m";    // Cyan
constexpr const char* COLOR_BOT = "\033[1;32m";     // Green
constexpr const char* COLOR_SYSTEM = "\033[1;33m";  // Yellow
constexpr const char* COLOR_ERROR = "\033[1;31m";   // Red

// TD-258: set by Ctrl+C (SIGINT) while run() is active; run() then saves and exits.
volatile std::sig_atomic_t g_interrupted = 0;

void on_interrupt(int /*signal*/) {
    g_interrupted = 1;
}

// TD-258: installs on_interrupt() for SIGINT for run()'s duration, and restores the previous handler.
class InterruptHandler {
   public:
    InterruptHandler() {
        g_interrupted = 0;
#ifdef _WIN32
        previous_ = std::signal(SIGINT, on_interrupt);  // the CRT resets it to SIG_DFL once delivered
#else
        struct sigaction action {};
        action.sa_handler = on_interrupt;
        sigemptyset(&action.sa_mask);
        // No SA_RESTART, so a read waiting at the prompt returns and getline() ends. SA_RESETHAND: a
        // second Ctrl+C gets the default action and ends the process at once.
        action.sa_flags = SA_RESETHAND;
        installed_ = sigaction(SIGINT, &action, &previous_) == 0;
#endif
    }
    ~InterruptHandler() {
#ifdef _WIN32
        std::signal(SIGINT, previous_ == SIG_ERR ? SIG_DFL : previous_);
#else
        if (installed_) {
            sigaction(SIGINT, &previous_, nullptr);
        }
#endif
    }
    InterruptHandler(const InterruptHandler&) = delete;
    InterruptHandler& operator=(const InterruptHandler&) = delete;

   private:
#ifdef _WIN32
    void (*previous_)(int) = SIG_DFL;
#else
    struct sigaction previous_ {};
    bool installed_ = false;
#endif
};

// TD-258: `path` with the local date and time added before its extension
// (conversation_history.txt -> conversation_history-20261103-142501.txt), and -2, -3, ... after that
// when a file of that name already exists.
std::string dated_save_path(const std::string& path) {
    namespace fs = std::filesystem;
    const fs::path original(path);
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &local);
    const std::string base = (original.parent_path() / original.stem()).string() + "-" + stamp;
    const std::string extension = original.extension().string();
    std::string candidate = base + extension;
    std::error_code ec;
    for (int n = 2; fs::exists(candidate, ec); ++n) {
        candidate = base + "-" + std::to_string(n) + extension;
    }
    return candidate;
}
}  // namespace

ChatbotCLI::ChatbotCLI(std::string server_url, std::string conv_save_file)
    : client(std::move(server_url)),
      conversation_save_path(std::move(conv_save_file)),
      active_save_path(conversation_save_path) {
    client.settings().strategy = "nucleus";
}

ChatbotCLI::~ChatbotCLI() = default;
ChatbotCLI::ChatbotCLI(ChatbotCLI&&) noexcept = default;
ChatbotCLI& ChatbotCLI::operator=(ChatbotCLI&&) noexcept = default;

bool ChatbotCLI::initialize() {
    std::cout << COLOR_SYSTEM << "🤖 Connecting to Chatbot API at " << client.server_url() << "..."
              << COLOR_RESET << '\n';

    const ChatbotClient::Health health = client.connect();
    if (health.reachable && health.status == 200) {
        std::cout << COLOR_SYSTEM << "✅ Connected successfully!" << COLOR_RESET << '\n';
        return true;
    }
    std::cout << COLOR_ERROR << "❌ Failed to connect to server." << COLOR_RESET << '\n';
    if (health.reachable) {
        std::cout << "Status: " << health.status << '\n';
    } else {
        std::cout << "Connection Error: " << (health.error.empty() ? "Unknown" : health.error) << '\n';
    }
    return false;
}

namespace {

// TD-256: the /set parameters, their other names and their ranges (the server's GenerationLimits).
void print_set_parameters() {
    using Limits = adai::GenerationLimits;
    std::cout << "  " << COLOR_SYSTEM << "/set parameters:" << COLOR_RESET << '\n';
    std::cout << "    strategy     greedy, beam, temperature (or sampling), top_k (or top-k), nucleus (or top_p)"
              << '\n';
    std::cout << "    length       " << Limits::kMinMaxLength << " to " << Limits::kMaxMaxLength
              << " new tokens (or max_length)" << '\n';
    std::cout << "    temperature  " << Limits::kMinTemperature << " to " << Limits::kMaxTemperature
              << " (or temp)" << '\n';
    std::cout << "    top_p        " << Limits::kMinTopP << " to " << Limits::kMaxTopP << " (or top-p)" << '\n';
    std::cout << "    top_k        " << Limits::kMinTopK << " to " << Limits::kMaxTopK << " (or top-k)" << '\n';
    std::cout << "    beam_width   " << Limits::kMinBeamWidth << " to " << Limits::kMaxBeamWidth
              << " (or beam-width)" << '\n';
}

}  // namespace

void ChatbotCLI::print_welcome() {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << '\n';
    std::cout << "║          🤖 ADAI Chatbot API Client v1.0                 ║" << '\n';
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << '\n';
    std::cout << '\n';
    // TD-256: every command handle_command() accepts.
    std::cout << COLOR_SYSTEM << "Commands:" << COLOR_RESET << '\n';
    std::cout << "  /help                 - Show this help message" << '\n';
    std::cout << "  /settings             - Show the generation settings your messages use" << '\n';
    std::cout << "  /set <param> <value>  - Change a generation setting for your messages" << '\n';
    std::cout << "  /set <param> default  - Go back to the server's default" << '\n';
    std::cout << "  /clear                - Clear the conversation on the server" << '\n';
    std::cout << "  /save                 - Save the conversation to the save file" << '\n';
    std::cout << "  /load                 - Load the conversation from the save file" << '\n';
    std::cout << "  /stats                - Conversation statistics (not available from the API server)" << '\n';
    std::cout << "  /exit, /quit          - Save the conversation and exit (end of input, Ctrl+D, Ctrl+C too)" << '\n';
    std::cout << '\n';
    print_set_parameters();
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
    show("Strategy", "strategy", get_generation_strategy());
    show("Max length", "max_length", get_max_response_length());
    show("Temperature", "temperature", get_temperature());
    show("Top-p (nucleus)", "top_p", get_top_p());
    show("Top-k", "top_k", get_top_k());
    show("Beam width", "beam_width", get_beam_width());
    std::cout << '\n';
}

void ChatbotCLI::handle_command(const std::string& command) {
    std::string_view cmd_view(command);

    if (command == "/help") {
        print_welcome();
    } else if (command == "/clear") {
        const std::string cleared = client.session_id();
        switch (client.clear_session()) {
            case ChatbotClient::ClearResult::Cleared:
                std::cout << COLOR_SYSTEM << "✅ Conversation history cleared (Session " << cleared << ")"
                          << COLOR_RESET << '\n';
                break;
            case ChatbotClient::ClearResult::NoSession:
                std::cout << COLOR_SYSTEM << "✅ No active session to clear" << COLOR_RESET << '\n';
                break;
            case ChatbotClient::ClearResult::SessionLost:  // TD-232: the server no longer holds it
                report_lost_session();
                break;
            case ChatbotClient::ClearResult::Failed:
                std::cout << COLOR_ERROR << "❌ Failed to clear session" << COLOR_RESET << '\n';
                break;
        }
    } else if (command == "/stats") {
        print_stats();
    } else if (command == "/settings") {
        print_settings();
    } else if (command == "/set" || cmd_view.substr(0, 5) == "/set ") {
        handle_setting(cmd_view.substr(std::min<size_t>(5, cmd_view.size())));  // TD-256: bare /set shows usage
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
        print_set_parameters();  // TD-256
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
        client.settings().reset(key);
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
        std::cout << COLOR_SYSTEM << "✅ Max response length set to: " << get_max_response_length() << COLOR_RESET
                  << '\n';
    } else if (key == "temperature") {
        if (!parse_real(value, real) || real < Limits::kMinTemperature || real > Limits::kMaxTemperature) {
            return invalid("a number from 0 to 2");
        }
        set_temperature(real);
        std::cout << COLOR_SYSTEM << "✅ Temperature set to: " << get_temperature() << COLOR_RESET << '\n';
    } else if (key == "top_p") {
        if (!parse_real(value, real) || real < Limits::kMinTopP || real > Limits::kMaxTopP) {
            return invalid("a number from 0 to 1");
        }
        set_top_p(real);
        std::cout << COLOR_SYSTEM << "✅ Top-p set to: " << get_top_p() << COLOR_RESET << '\n';
    } else if (key == "top_k") {
        if (!parse_whole(value, whole) || whole < Limits::kMinTopK || whole > Limits::kMaxTopK) {
            return invalid("a whole number from 1 to 1000");
        }
        set_top_k(whole);
        std::cout << COLOR_SYSTEM << "✅ Top-k set to: " << get_top_k() << COLOR_RESET << '\n';
    } else {
        if (!parse_whole(value, whole) || whole < Limits::kMinBeamWidth || whole > Limits::kMaxBeamWidth) {
            return invalid("a whole number from 1 to 16");
        }
        set_beam_width(whole);
        std::cout << COLOR_SYSTEM << "✅ Beam width set to: " << get_beam_width() << COLOR_RESET << '\n';
    }
}

std::string ChatbotCLI::generate_response(const std::string& user_input) {
    const ChatbotClient::Reply reply = client.send_message(user_input);
    if (reply.session_lost) {
        // TD-232: the server no longer held the session, so the client sent this in a new one.
        report_lost_session();
        std::cout << COLOR_SYSTEM << "ℹ️  Sending this message in a new session" << COLOR_RESET << '\n';
    }
    last_turn_failed = !reply.ok;  // TD-255
    return reply.text;
}

void ChatbotCLI::report_lost_session() {
    // TD-258: the server forgot the session (expired, evicted or restarted; TD-232) and with it every
    // earlier message. Say so, and stop treating the save file as this conversation's: the next save
    // would otherwise replace the saved conversation with only what follows.
    std::cout << COLOR_SYSTEM
              << "ℹ️  The server no longer has this session (it expired or the server restarted) and has "
                 "forgotten its earlier messages";
    if (owns_save_file) {
        std::cout << "; they were last saved to " << active_save_path;
        if (active_save_path == conversation_save_path) {
            std::cout << " (/load restores them)";
        }
    }
    std::cout << COLOR_RESET << '\n';
    active_save_path = conversation_save_path;
    owns_save_file = false;
}

void ChatbotCLI::save_conversation() {
    // TD-053: session state lives server-side, not in this object -- "saving" means asking the
    // server to export it, then writing the result to our local file.
    using Status = ChatbotClient::Export::Status;
    const ChatbotClient::Export exported = client.export_session();
    switch (exported.status) {
        case Status::Ok:
            break;
        case Status::NoSession:
            std::cout << COLOR_ERROR << "❌ Nothing to save yet — send a message first." << COLOR_RESET << '\n';
            return;
        case Status::NotConnected:
            std::cout << COLOR_ERROR << "❌ Client not initialized" << COLOR_RESET << '\n';
            return;
        case Status::Unreachable:
            std::cout << COLOR_ERROR << "❌ Failed to reach server to export conversation" << COLOR_RESET << '\n';
            return;
        case Status::SessionLost:  // TD-232/TD-258: nothing left on the server to save
            report_lost_session();
            std::cout << COLOR_ERROR << "❌ Nothing to save; " << active_save_path << " is unchanged"
                      << COLOR_RESET << '\n';
            return;
        case Status::Failed:
            std::cout << COLOR_ERROR << "❌ " << (exported.error.empty() ? "Export failed" : exported.error)
                      << COLOR_RESET << '\n';
            return;
    }

    // TD-258: replace an existing file only when it holds this session's conversation; otherwise save
    // to a new dated file. Either way write a temporary file and rename it, so a failure keeps the old one.
    std::string target = active_save_path;
    std::error_code exists_error;
    const bool keep_existing = !owns_save_file && std::filesystem::exists(target, exists_error);
    if (keep_existing) {
        target = dated_save_path(target);
    }
    const std::string error = chatbot_client::write_file_atomically(target, exported.data);
    if (!error.empty()) {
        std::cout << COLOR_ERROR << "❌ " << error << COLOR_RESET << '\n';
        return;
    }
    if (keep_existing) {
        std::cout << COLOR_SYSTEM << "ℹ️  " << active_save_path
                  << " holds a conversation this session didn't load, so it was kept" << COLOR_RESET
                  << '\n';
    }
    active_save_path = target;
    owns_save_file = true;

    std::cout << COLOR_SYSTEM << "✅ Conversation saved to " << active_save_path << COLOR_RESET << '\n';
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
    const ChatbotClient::Import imported = client.import_session(data);
    if (!imported.ok) {
        std::cout << COLOR_ERROR << "❌ " << imported.error << COLOR_RESET << '\n';
        return;
    }
    active_save_path = conversation_save_path;  // TD-258: saves now continue this file
    owns_save_file = true;

    std::cout << COLOR_SYSTEM << "✅ Conversation loaded from " << conversation_save_path;
    if (!imported.message_count.empty()) {
        std::cout << " (" << imported.message_count << " messages)";
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

    const InterruptHandler interrupt_handler;  // TD-258
    while (running) {
        // TD-253: end of input (Ctrl+D, or piped/redirected stdin running out) ends the session like
        // /exit. getline's result used to go unchecked, so at EOF the empty line was skipped and the
        // loop printed prompts forever (18.6 million in 3 s) without ever saving.
        bool at_end = false;
        if (g_interrupted == 0) {
            std::cout << COLOR_USER << "You: " << COLOR_RESET;
            at_end = !std::getline(std::cin, user_input);
        }
        if (g_interrupted != 0) {
            // TD-258: Ctrl+C at the prompt (the read returns) or during a reply ends the session like
            // /exit. It used to kill the process without the auto-save.
            std::cout << '\n' << COLOR_SYSTEM << "Interrupted" << COLOR_RESET << '\n';
            user_input = "/exit";
        } else if (at_end) {
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
            if (!client.session_id().empty()) {
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
