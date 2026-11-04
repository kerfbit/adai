#ifndef CHATBOT_CLI_HPP
#define CHATBOT_CLI_HPP

// @adai-status: stable
// @adai-version: 1.7.0
// @adai-reviewed: 2026-11-04

#include <string>
#include <string_view>
#include "ChatbotClient.hpp"  // TD-260: the HTTP session, shared with chatbot_gui

/**
 * @brief Interactive command-line interface for the ADAI transformer-based chatbot
 *
 * Provides a user-friendly terminal application with conversation management,
 * multiple generation strategies, configurable parameters, and persistent conversation history.
 */
class ChatbotCLI {
   public:
    /**
     * @brief Construct a new ChatbotCLI object
     *
     * @param server_url URL of the Chatbot API (e.g., "http://localhost:8080")
     * @param conv_save_file Path for saving conversation history (default:
     * "conversation_history.txt")
     */
    ChatbotCLI(std::string server_url, std::string conv_save_file = "conversation_history.txt");

    /**
     * @brief Destructor - automatically cleans up resources via smart pointers
     */
    ~ChatbotCLI();

    // Prevent copying (contains unique resources)
    ChatbotCLI(const ChatbotCLI&) = delete;
    ChatbotCLI& operator=(const ChatbotCLI&) = delete;

    // Allow moving (defined in the .cpp, where httplib::Client is complete)
    ChatbotCLI(ChatbotCLI&&) noexcept;
    ChatbotCLI& operator=(ChatbotCLI&&) noexcept;

    /**
     * @brief Initialize the chatbot (connect to server)
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Run the main chatbot loop (interactive REPL). Returns on /exit, /quit, end of input
     * (Ctrl+D, or piped stdin running out; TD-253) or Ctrl+C (SIGINT, TD-258), saving the
     * conversation in each case. Ctrl+C at the prompt ends the session at once; during a reply it
     * ends it once the reply is in. A second Ctrl+C exits immediately, without saving.
     */
    void run();

    /**
     * @brief Print welcome message and command help
     */
    static void print_welcome();

    /**
     * @brief Print conversation statistics
     */
    static void print_stats();

    /**
     * @brief Print current generation settings
     */
    void print_settings();

    /**
     * @brief Handle a command entered by the user
     * @param command The command string (e.g., "/help", "/set temp 0.8")
     */
    void handle_command(const std::string& command);

    /**
     * @brief Handle a /set command to change generation parameters
     * @param setting The parameter and value (e.g., "temp 0.8")
     */
    void handle_setting(std::string_view setting);

    /**
     * @brief Generate a response to user input
     * @param user_input The user's message
     * @return The generated response, or "Error: ..." with the server's error message when the turn
     * failed (TD-255; did_last_turn_fail() then returns true)
     */
    std::string generate_response(const std::string& user_input);
    /// Whether the last generate_response() call failed (TD-255).
    bool did_last_turn_fail() const {
        return last_turn_failed;
    }

    /**
     * @brief Save the current conversation to conversation_save_path (TD-053).
     *
     * ChatbotCLI holds no local conversation state of its own — the actual history lives
     * server-side in ChatbotAPI's Session/ConversationContext, keyed by session_id — so this
     * exports it over HTTP (POST /chat/session/export) and writes the returned serialized text
     * to the local file, rather than serializing anything held in this object directly. Prints
     * a clear message either way (success, "nothing to save yet" if no session exists, or a
     * network/file error) — called both by the explicit /save command and automatically on
     * /exit, /quit, end of input (TD-253) and Ctrl+C (TD-258).
     *
     * TD-258: the file is written to a temporary file beside it and renamed over it, so a failed
     * or interrupted save leaves the previous file whole. A file that already exists is replaced
     * only when it holds this session's conversation (this session loaded or saved it); otherwise
     * the conversation goes to a new file named after it with the date and time added
     * (conversation_history-20261103-142501.txt), which later saves keep using.
     */
    void save_conversation();

    /**
     * @brief Load a conversation from conversation_save_path and restore it server-side (TD-053).
     *
     * Reads the local file and POSTs its content to /chat/session/import, adopting whatever
     * session_id the server returns (a fresh session if this instance had none yet, matching
     * generate_response()'s own "adopt the server's session id" pattern). Prints a clear message
     * either way (success with the restored message count, or a file/network/parse error).
     */
    void load_conversation();

    // Accessors for testing
    const std::string& get_generation_strategy() const {
        return client.settings().strategy;
    }
    int get_max_response_length() const {
        return client.settings().max_length;
    }
    float get_temperature() const {
        return client.settings().temperature;
    }
    float get_top_p() const {
        return client.settings().top_p;
    }
    int get_top_k() const {
        return client.settings().top_k;
    }
    int get_beam_width() const {
        return client.settings().beam_width;
    }
    const std::string& get_server_url() const {
        return client.server_url();
    }
    const std::string& get_conversation_save_path() const {
        return conversation_save_path;
    }
    /// TD-258: the file the next save writes (conversation_save_path, or the dated file a save
    /// went to instead of replacing a conversation this session didn't load).
    const std::string& get_active_save_path() const {
        return active_save_path;
    }

    /// TD-254: whether this setting (its JSON name: max_length, temperature, top_p, top_k, beam_width,
    /// strategy) was changed by /set or a setter, and so is sent with each message. The others use
    /// the server's defaults.
    bool is_overridden(const std::string& setting) const {
        return client.settings().is_overridden(setting);
    }

    // Setters for testing; each one also sends that setting with each message (TD-254)
    void set_generation_strategy(const std::string& strategy) {
        client.settings().set_strategy(strategy);
    }
    void set_max_response_length(int length) {
        client.settings().set_max_length(length);
    }
    void set_temperature(float temp) {
        client.settings().set_temperature(temp);
    }
    void set_top_p(float p) {
        client.settings().set_top_p(p);
    }
    void set_top_k(int k) {
        client.settings().set_top_k(k);
    }
    void set_beam_width(int width) {
        client.settings().set_beam_width(width);
    }

    /// The JSON body POST /chat/session sends for user_input (TD-254: only overridden settings).
    std::string chat_turn_body(const std::string& user_input) const {
        return client.chat_turn_body(user_input);
    }

   private:
    ChatbotClient client;  // TD-260: server session, settings and requests
    std::string conversation_save_path;
    bool last_turn_failed{false};  // TD-255

    // TD-258: where this session's conversation is saved, and whether that file holds it (this
    // session loaded or saved it there), so a save may replace it.
    std::string active_save_path;
    bool owns_save_file{false};

    // TD-258: the server no longer holds this session; tells the user and stops treating the save
    // file as this conversation's. The client has already forgotten the session id.
    void report_lost_session();
};

#endif  // CHATBOT_CLI_HPP
