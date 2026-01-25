#ifndef CONVERSATIONCONTEXT_HPP
#define CONVERSATIONCONTEXT_HPP

#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @class ConversationContext
 * @brief Manages conversation history for multi-turn chatbot interactions
 *
 * Maintains message history with role-based tracking (user/assistant/system),
 * automatic context length management, and formatting for model input.
 *
 * Features:
 * - Role-based message tracking (user, assistant, system)
 * - Automatic truncation to max token/message limits
 * - Sliding window context management
 * - Special token formatting for models
 * - Conversation state persistence (save/load)
 * - Context summarization support
 */
class ConversationContext {
   public:
    /**
     * @struct Message
     * @brief Represents a single message in the conversation
     */
    struct Message {
        std::string role;     // "user", "assistant", or "system"
        std::string content;  // The actual message text
        int token_count;      // Estimated token count for this message

        Message(const std::string& r, const std::string& c, int tokens = 0)
            : role(r), content(c), token_count(tokens) {}
    };

    /**
     * @brief Constructor with configurable limits
     * @param max_messages Maximum number of messages to retain (0 = unlimited)
     * @param max_tokens Maximum total tokens in context (0 = unlimited)
     * @param keep_system_message Whether to always keep system message
     */
    ConversationContext(int max_messages = 20, int max_tokens = 2048,
                        bool keep_system_message = true);

    /**
     * @brief Destructor - cleans up system message
     */
    ~ConversationContext();

    /**
     * @brief Add a user message to the conversation
     * @param content The user's message text
     * @param token_count Estimated token count (0 = auto-estimate)
     */
    void add_user_message(const std::string& content, int token_count = 0);

    /**
     * @brief Add an assistant message to the conversation
     * @param content The assistant's response text
     * @param token_count Estimated token count (0 = auto-estimate)
     */
    void add_assistant_message(const std::string& content, int token_count = 0);

    /**
     * @brief Set or update the system message (context/instructions)
     * @param content System prompt/instructions
     * @param token_count Estimated token count (0 = auto-estimate)
     */
    void set_system_message(const std::string& content, int token_count = 0);

    /**
     * @brief Add a message with custom role
     * @param role Message role (user/assistant/system/custom)
     * @param content Message text
     * @param token_count Estimated token count (0 = auto-estimate)
     */
    void add_message(const std::string& role, const std::string& content, int token_count = 0);

    /**
     * @brief Format conversation history for model input
     * @param include_system Whether to include system message
     * @param separator Token separator (default: newline)
     * @return Formatted conversation string
     */
    std::string format_for_model(bool include_system = true,
                                 const std::string& separator = "\n") const;

    /**
     * @brief Format with special tokens for encoder-decoder models
     * @param bos_token Begin-of-sequence token
     * @param eos_token End-of-sequence token
     * @param sep_token Separator between messages
     * @return Formatted conversation with special tokens
     */
    std::string format_with_special_tokens(const std::string& bos_token = "<bos>",
                                           const std::string& eos_token = "<eos>",
                                           const std::string& sep_token = "<sep>") const;

    /**
     * @brief Get the last user message
     * @return Last user message content
     * @throws std::runtime_error if no user messages exist
     */
    std::string get_last_user_message() const;

    /**
     * @brief Get the last assistant message
     * @return Last assistant message content
     * @throws std::runtime_error if no assistant messages exist
     */
    std::string get_last_assistant_message() const;

    /**
     * @brief Get all messages in the conversation
     * @return Vector of all messages
     */
    std::vector<Message> get_messages() const;

    /**
     * @brief Get system message if set
     * @return System message content (empty if not set)
     */
    std::string get_system_message() const;

    /**
     * @brief Get current total token count
     * @return Sum of tokens across all messages
     */
    int get_total_tokens() const;

    /**
     * @brief Get current message count
     * @return Number of messages (excluding system message)
     */
    int get_message_count() const;

    /**
     * @brief Check if conversation is empty
     * @return True if no messages (system message doesn't count)
     */
    bool is_empty() const;

    /**
     * @brief Clear all messages except system message
     */
    void clear();

    /**
     * @brief Clear all messages including system message
     */
    void clear_all();

    /**
     * @brief Truncate to fit within token/message limits
     *
     * Removes oldest messages (keeping system message) until within limits.
     * Uses sliding window strategy.
     */
    void truncate_to_limits();

    /**
     * @brief Manually set max message limit
     * @param max_messages New limit (0 = unlimited)
     */
    void set_max_messages(int max_messages);

    /**
     * @brief Manually set max token limit
     * @param max_tokens New limit (0 = unlimited)
     */
    void set_max_tokens(int max_tokens);

    /**
     * @brief Save conversation to file
     * @param filepath Path to save file
     * @throws std::runtime_error on write failure
     */
    void save_to_file(const std::string& filepath) const;

    /**
     * @brief Load conversation from file
     * @param filepath Path to load file
     * @throws std::runtime_error on read failure
     */
    void load_from_file(const std::string& filepath);

    /**
     * @brief Get conversation summary statistics
     * @return String with stats (message count, token count, etc.)
     */
    std::string get_statistics() const;

    /**
     * @brief Create a summarized version of the conversation
     * @param keep_recent Number of recent messages to keep in full
     * @param summary_text Summary of older messages
     * @return New ConversationContext with summarized history
     */
    ConversationContext create_summarized(int keep_recent = 5,
                                          const std::string& summary_text = "") const;

   private:
    std::deque<Message> messages;  // Conversation history (deque for efficient removal)
    Message* system_message;       // Optional system message
    int max_messages;              // Max number of messages to keep
    int max_tokens;                // Max total tokens
    bool keep_system_message;      // Whether to preserve system message
    int total_tokens;              // Cached total token count

    /**
     * @brief Estimate token count for a message
     * @param content Message text
     * @return Estimated token count (rough approximation)
     */
    int estimate_tokens(const std::string& content) const;

    /**
     * @brief Update total token count cache
     */
    void update_token_count();

    /**
     * @brief Remove oldest message (excluding system)
     */
    void remove_oldest_message();
};

#endif  // CONVERSATIONCONTEXT_HPP
