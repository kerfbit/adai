#ifndef CHATBOT_CLI_HPP
#define CHATBOT_CLI_HPP

#include <memory>
#include <string>
#include <string_view>
#include <../external/cpp-httplib/httplib.h>

// Forward declarations removed: BPETokenizer, EncoderDecoderModel, ConversationContext

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
     * @param conv_save_file Path for saving conversation history (default: "conversation_history.txt")
     */
    ChatbotCLI(const std::string& server_url,
               const std::string& conv_save_file = "conversation_history.txt");

    /**
     * @brief Destructor - automatically cleans up resources via smart pointers
     */
    ~ChatbotCLI();

    // Prevent copying (contains unique resources)
    ChatbotCLI(const ChatbotCLI&) = delete;
    ChatbotCLI& operator=(const ChatbotCLI&) = delete;

    // Allow moving
    ChatbotCLI(ChatbotCLI&&) = default;
    ChatbotCLI& operator=(ChatbotCLI&&) = default;

    /**
     * @brief Initialize the chatbot (connect to server)
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Run the main chatbot loop (interactive REPL)
     */
    void run();

    /**
     * @brief Print welcome message and command help
     */
    void print_welcome();

    /**
     * @brief Print conversation statistics
     */
    void print_stats();

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
     * @return The generated response
     */
    std::string generate_response(const std::string& user_input);

    // Accessors for testing
    const std::string& get_generation_strategy() const { return generation_strategy; }
    int get_max_response_length() const { return max_response_length; }
    float get_temperature() const { return temperature; }
    float get_top_p() const { return top_p; }
    int get_top_k() const { return top_k; }
    int get_beam_width() const { return beam_width; }
    const std::string& get_server_url() const { return server_url; }
    const std::string& get_conversation_save_path() const { return conversation_save_path; }

    // Setters for testing
    void set_generation_strategy(const std::string& strategy) { generation_strategy = strategy; }
    void set_max_response_length(int length) { max_response_length = length; }
    void set_temperature(float temp) { temperature = temp; }
    void set_top_p(float p) { top_p = p; }
    void set_top_k(int k) { top_k = k; }
    void set_beam_width(int width) { beam_width = width; }

   private:
    std::string server_url;
    std::string conversation_save_path;
    std::string session_id;
    
    std::unique_ptr<httplib::Client> client;

    // Generation parameters
    int max_response_length;
    float temperature;
    float top_p;
    int top_k;
    int beam_width;
    std::string generation_strategy;
};

// ANSI color codes for better CLI experience
#define COLOR_RESET "\033[0m"
#define COLOR_USER "\033[1;36m"    // Cyan
#define COLOR_BOT "\033[1;32m"     // Green
#define COLOR_SYSTEM "\033[1;33m"  // Yellow
#define COLOR_ERROR "\033[1;31m"   // Red

#endif  // CHATBOT_CLI_HPP
