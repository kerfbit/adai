#include "ConversationContext.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

ConversationContext::ConversationContext(int max_messages, int max_tokens, bool keep_system_message)
    : max_messages(max_messages),
      max_tokens(max_tokens),
      keep_system_message(keep_system_message),
      total_tokens(0),
      system_message(nullptr) {}

ConversationContext::~ConversationContext() {
    if (system_message != nullptr) {
        delete system_message;
    }
}

void ConversationContext::add_user_message(const std::string& content, int token_count) {
    add_message("user", content, token_count);
}

void ConversationContext::add_assistant_message(const std::string& content, int token_count) {
    add_message("assistant", content, token_count);
}

void ConversationContext::set_system_message(const std::string& content, int token_count) {
    // Delete old system message if exists
    if (system_message != nullptr) {
        total_tokens -= system_message->token_count;
        delete system_message;
    }

    // Create new system message
    if (token_count == 0) {
        token_count = estimate_tokens(content);
    }

    system_message = new Message("system", content, token_count);
    total_tokens += token_count;
}

void ConversationContext::add_message(const std::string& role, const std::string& content,
                                      int token_count) {
    // Estimate tokens if not provided
    if (token_count == 0) {
        token_count = estimate_tokens(content);
    }

    // Add message
    messages.emplace_back(role, content, token_count);
    total_tokens += token_count;

    // Truncate if needed
    truncate_to_limits();
}

std::string ConversationContext::format_for_model(bool include_system,
                                                  const std::string& separator) const {
    std::ostringstream oss;

    // Add system message if requested and exists
    if (include_system && system_message != nullptr) {
        oss << "System: " << system_message->content << separator;
    }

    // Add all conversation messages
    for (const auto& msg : messages) {
        // Capitalize first letter of role
        std::string role_cap = msg.role;
        if (!role_cap.empty()) {
            role_cap[0] = std::toupper(role_cap[0]);
        }

        oss << role_cap << ": " << msg.content << separator;
    }

    return oss.str();
}

std::string ConversationContext::format_with_special_tokens(const std::string& bos_token,
                                                            const std::string& eos_token,
                                                            const std::string& sep_token) const {
    std::ostringstream oss;

    oss << bos_token;

    // Add system message if exists
    if (system_message != nullptr) {
        oss << " [SYSTEM] " << system_message->content << sep_token;
    }

    // Add all messages with role tags
    for (const auto& msg : messages) {
        std::string role_upper = msg.role;
        std::transform(role_upper.begin(), role_upper.end(), role_upper.begin(), ::toupper);

        oss << " [" << role_upper << "] " << msg.content << sep_token;
    }

    oss << " " << eos_token;

    return oss.str();
}

std::string ConversationContext::get_last_user_message() const {
    // Search backwards for last user message
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->role == "user") {
            return it->content;
        }
    }

    throw std::runtime_error("No user messages in conversation");
}

std::string ConversationContext::get_last_assistant_message() const {
    // Search backwards for last assistant message
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->role == "assistant") {
            return it->content;
        }
    }

    throw std::runtime_error("No assistant messages in conversation");
}

std::vector<ConversationContext::Message> ConversationContext::get_messages() const {
    return std::vector<Message>(messages.begin(), messages.end());
}

std::string ConversationContext::get_system_message() const {
    if (system_message != nullptr) {
        return system_message->content;
    }
    return "";
}

int ConversationContext::get_total_tokens() const {
    return total_tokens;
}

int ConversationContext::get_message_count() const {
    return static_cast<int>(messages.size());
}

bool ConversationContext::is_empty() const {
    return messages.empty();
}

void ConversationContext::clear() {
    // Clear all messages but keep system message
    total_tokens = (system_message != nullptr) ? system_message->token_count : 0;
    messages.clear();
}

void ConversationContext::clear_all() {
    // Clear everything including system message
    messages.clear();

    if (system_message != nullptr) {
        delete system_message;
        system_message = nullptr;
    }

    total_tokens = 0;
}

void ConversationContext::truncate_to_limits() {
    // Truncate by message count
    if (max_messages > 0) {
        while (static_cast<int>(messages.size()) > max_messages) {
            remove_oldest_message();
        }
    }

    // Truncate by token count
    if (max_tokens > 0) {
        int system_tokens = (system_message != nullptr) ? system_message->token_count : 0;

        while (total_tokens > max_tokens && !messages.empty()) {
            // Keep at least one message if possible
            if (messages.size() == 1 && total_tokens <= max_tokens * 1.2) {
                break;
            }
            remove_oldest_message();
        }
    }
}

void ConversationContext::set_max_messages(int max_msgs) {
    max_messages = max_msgs;
    truncate_to_limits();
}

void ConversationContext::set_max_tokens(int max_toks) {
    max_tokens = max_toks;
    truncate_to_limits();
}

void ConversationContext::save_to_file(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    // Write metadata
    file << "MAX_MESSAGES:" << max_messages << "\n";
    file << "MAX_TOKENS:" << max_tokens << "\n";
    file << "KEEP_SYSTEM:" << (keep_system_message ? "1" : "0") << "\n";
    file << "---\n";

    // Write system message if exists
    if (system_message != nullptr) {
        file << "SYSTEM|" << system_message->token_count << "|" << system_message->content << "\n";
    }

    // Write all messages
    for (const auto& msg : messages) {
        file << msg.role << "|" << msg.token_count << "|" << msg.content << "\n";
    }

    file.close();
}

void ConversationContext::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    // Clear current state
    clear_all();

    std::string line;
    bool metadata_section = true;

    while (std::getline(file, line)) {
        if (line == "---") {
            metadata_section = false;
            continue;
        }

        if (metadata_section) {
            // Parse metadata
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = line.substr(0, colon_pos);
                std::string value = line.substr(colon_pos + 1);

                if (key == "MAX_MESSAGES") {
                    max_messages = std::stoi(value);
                } else if (key == "MAX_TOKENS") {
                    max_tokens = std::stoi(value);
                } else if (key == "KEEP_SYSTEM") {
                    keep_system_message = (value == "1");
                }
            }
        } else {
            // Parse message: role|token_count|content
            size_t first_pipe = line.find('|');
            if (first_pipe == std::string::npos)
                continue;

            size_t second_pipe = line.find('|', first_pipe + 1);
            if (second_pipe == std::string::npos)
                continue;

            std::string role = line.substr(0, first_pipe);
            int token_count = std::stoi(line.substr(first_pipe + 1, second_pipe - first_pipe - 1));
            std::string content = line.substr(second_pipe + 1);

            if (role == "SYSTEM") {
                set_system_message(content, token_count);
            } else {
                add_message(role, content, token_count);
            }
        }
    }

    file.close();
}

std::string ConversationContext::get_statistics() const {
    std::ostringstream oss;

    oss << "Conversation Statistics:\n";
    oss << "  Messages: " << messages.size() << "\n";
    oss << "  Total Tokens: " << total_tokens << "\n";
    oss << "  System Message: " << (system_message != nullptr ? "Yes" : "No") << "\n";

    if (max_messages > 0) {
        oss << "  Max Messages: " << max_messages << "\n";
    }
    if (max_tokens > 0) {
        oss << "  Max Tokens: " << max_tokens << "\n";
    }

    // Count by role
    int user_count = 0;
    int assistant_count = 0;
    int other_count = 0;

    for (const auto& msg : messages) {
        if (msg.role == "user") {
            user_count++;
        } else if (msg.role == "assistant") {
            assistant_count++;
        } else {
            other_count++;
        }
    }

    oss << "  User Messages: " << user_count << "\n";
    oss << "  Assistant Messages: " << assistant_count << "\n";
    if (other_count > 0) {
        oss << "  Other Messages: " << other_count << "\n";
    }

    return oss.str();
}

ConversationContext ConversationContext::create_summarized(int keep_recent,
                                                           const std::string& summary_text) const {
    ConversationContext summarized(max_messages, max_tokens, keep_system_message);

    // Copy system message
    if (system_message != nullptr) {
        summarized.set_system_message(system_message->content, system_message->token_count);
    }

    // If we have a summary, add it as a system-like message
    if (!summary_text.empty() && static_cast<int>(messages.size()) > keep_recent) {
        summarized.add_message("system", "[Summary of earlier conversation: " + summary_text + "]",
                               estimate_tokens(summary_text) + 10);
    }

    // Add recent messages
    int start_idx = std::max(0, static_cast<int>(messages.size()) - keep_recent);
    for (int i = start_idx; i < static_cast<int>(messages.size()); ++i) {
        const auto& msg = messages[i];
        summarized.add_message(msg.role, msg.content, msg.token_count);
    }

    return summarized;
}

int ConversationContext::estimate_tokens(const std::string& content) const {
    // Simple estimation: ~4 characters per token (rough BPE approximation)
    // Add some overhead for special tokens and formatting
    int char_count = static_cast<int>(content.length());
    int estimated = std::max(1, static_cast<int>(std::ceil(char_count / 4.0)));

    // Add overhead for spaces and punctuation
    int space_count = static_cast<int>(std::count(content.begin(), content.end(), ' '));
    estimated += space_count / 2;

    return estimated;
}

void ConversationContext::update_token_count() {
    total_tokens = 0;

    if (system_message != nullptr) {
        total_tokens += system_message->token_count;
    }

    for (const auto& msg : messages) {
        total_tokens += msg.token_count;
    }
}

void ConversationContext::remove_oldest_message() {
    if (!messages.empty()) {
        total_tokens -= messages.front().token_count;
        messages.pop_front();
    }
}
