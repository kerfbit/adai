// @adai-status: stable
// @adai-version: 1.4.0
// @adai-reviewed: 2026-11-04

#include "ConversationContext.hpp"
#include "ConversationFormat.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

// TD-266: the line format (escaping and parsing) lives in ConversationFormat, shared with chatbot_gui.
using adai::conversation_format::escape_line;

ConversationContext::ConversationContext(int max_messages, int max_tokens, bool keep_system_message)
    : max_messages(max_messages),
      max_tokens(max_tokens),
      keep_system_message(keep_system_message) {}

void ConversationContext::add_user_message(const std::string& content, int token_count) {
    add_message("user", content, token_count);
}

void ConversationContext::add_assistant_message(const std::string& content, int token_count) {
    add_message("assistant", content, token_count);
}

void ConversationContext::set_system_message(const std::string& content, int token_count) {
    // Replace old system message if exists
    if (system_message.has_value()) {
        total_tokens -= system_message->token_count;
    }

    // Create new system message
    if (token_count == 0) {
        token_count = count_message_tokens("system", content);  // TD-294
    }

    system_message.emplace("system", content, token_count);
    total_tokens += token_count;
}

void ConversationContext::add_message(const std::string& role, const std::string& content,
                                      int token_count) {
    // Count tokens if not provided (TD-294: with the token counter when there is one)
    if (token_count == 0) {
        token_count = count_message_tokens(role, content);
    }

    // Add message
    messages.emplace_back(role, content, token_count);
    total_tokens += token_count;

    // Truncate if needed
    truncate_to_limits();
}

// TODO: See TD-295 in TECHNICAL_DEBT.md - plain-text role labels let message content impersonate turns
std::string ConversationContext::format_for_model(bool include_system,
                                                  const std::string& separator) const {
    std::ostringstream oss;

    // Add system message if requested and exists
    if (include_system && system_message.has_value()) {
        oss << "System: " << system_message->content << separator;
    }

    // Add all conversation messages
    for (const auto& msg : messages) {
        // Capitalize first letter of role
        std::string role_cap = msg.role;
        if (!role_cap.empty()) {
            role_cap[0] = static_cast<char>(std::toupper(role_cap[0]));
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
    if (system_message.has_value()) {
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
    if (system_message.has_value()) {
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
    // TD-125: keep_system_message now actually gates this, per its own doc
    // comment ("Whether to always keep system message"). Regular messages are
    // always dropped; the system message survives only when the flag is set.
    messages.clear();
    if (keep_system_message) {
        total_tokens = system_message.has_value() ? system_message->token_count : 0;
    } else {
        system_message.reset();
        total_tokens = 0;
    }
}

void ConversationContext::clear_all() {
    // Deliberately unconditional, unlike clear() above: clear_all()'s whole
    // contract is "including system message" regardless of the persistent
    // keep_system_message setting — it's the explicit nuclear option, not a
    // configurable one. (See TD-125.)
    messages.clear();
    system_message.reset();
    total_tokens = 0;
}

namespace {
// TD-292: the longest tail of `content` whose estimated token count fits `max_tokens`, starting on a
// UTF-8 character boundary.
std::string tail_within_tokens(const std::string& content, int max_tokens,
                               const std::function<int(const std::string&)>& count) {
    size_t lo = 0, hi = content.size();  // find the smallest start whose tail fits
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (count(content.substr(mid)) <= max_tokens) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    size_t start = lo;
    while (start < content.size() && (static_cast<unsigned char>(content[start]) & 0xC0) == 0x80) {
        ++start;
    }
    return content.substr(start);
}
}  // namespace

void ConversationContext::truncate_to_limits() {
    // Truncate by message count
    if (max_messages > 0) {
        while (static_cast<int>(messages.size()) > max_messages) {
            remove_oldest_message();
        }
    }

    // Truncate by token count
    if (max_tokens > 0) {
        // TD-292: evict the oldest messages, never the newest. A message that alone exceeded 1.2x
        // the budget used to evict everything, itself included, leaving an empty conversation (and
        // an empty prompt that failed the request).
        while (total_tokens > max_tokens && messages.size() > 1) {
            remove_oldest_message();
        }
        // A lone newest message keeps the existing 1.2x slack. Past that it keeps its tail, within
        // the space the system message leaves; only with no space left at all is it evicted.
        if (messages.size() == 1 && total_tokens > max_tokens * 1.2) {
            const int system_tokens = system_message.has_value() ? system_message->token_count : 0;
            const int available = max_tokens - system_tokens;
            Message& last = messages.back();
            if (available > 0) {
                total_tokens -= last.token_count;
                auto count = [this, &last](const std::string& s) {
                    return count_message_tokens(last.role, s);
                };
                last.content = tail_within_tokens(last.content, available, count);
                last.token_count = count(last.content);
                total_tokens += last.token_count;
            } else {
                remove_oldest_message();
            }
        }

        // TD-125: keep_system_message previously had no effect here — the
        // system message sat outside `messages` entirely, so once regular
        // messages were exhausted the loop above (guarded by
        // `!messages.empty()`) simply stopped, leaving the context over
        // budget forever if the system message alone exceeded max_tokens.
        // When the flag is false the system message is no longer exempt: it
        // is evicted as a last resort so the token budget is actually
        // honored. When true (the default), this is a no-op and the system
        // message is preserved exactly as before.
        if (!keep_system_message && total_tokens > max_tokens && system_message.has_value()) {
            total_tokens -= system_message->token_count;
            system_message.reset();
        }
    }
}

void ConversationContext::set_max_messages(int max_messages) {
    this->max_messages = max_messages;
    truncate_to_limits();
}

void ConversationContext::set_max_tokens(int max_tokens) {
    this->max_tokens = max_tokens;
    truncate_to_limits();
}

std::string ConversationContext::serialize() const {
    std::ostringstream oss;

    // Write metadata
    oss << "MAX_MESSAGES:" << max_messages << "\n";
    oss << "MAX_TOKENS:" << max_tokens << "\n";
    oss << "KEEP_SYSTEM:" << (keep_system_message ? "1" : "0") << "\n";
    oss << "---\n";

    // Write system message if exists
    if (system_message.has_value()) {
        oss << "SYSTEM|" << system_message->token_count << "|"
            << escape_line(system_message->content) << "\n";
    }

    // Write all messages
    for (const auto& msg : messages) {
        // TODO: See TD-295 in TECHNICAL_DEBT.md - roles are written unescaped; '|' or a newline in a role corrupts the line
        oss << msg.role << "|" << msg.token_count << "|" << escape_line(msg.content) << "\n";
    }

    return oss.str();
}

// TODO: See TD-296 in TECHNICAL_DEBT.md - stream not checked after writing; not atomic (use temp file + rename)
void ConversationContext::save_to_file(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    file << serialize();
    file.close();
}

void ConversationContext::deserialize(const std::string& data) {
    // TD-293: parse into a temporary with this context's own limits (TD-291) and swap it in at the
    // end; clearing first used to leave the session empty when a line failed to parse.
    ConversationContext parsed(max_messages, max_tokens, keep_system_message);
    parsed.token_counter = token_counter;  // TD-294: imported messages count like the rest

    const adai::conversation_format::Parsed lines = adai::conversation_format::parse(data);
    // Data without the metadata/messages separator isn't a serialized conversation: refuse it
    // rather than replace the history with nothing.
    if (!lines.has_separator) {
        throw std::invalid_argument("not a serialized conversation (no '---' line)");
    }
    // TD-291: token counts are recomputed by add_message(), not taken from the file.
    if (lines.system_message) {
        parsed.set_system_message(*lines.system_message);
    }
    for (const auto& [role, content] : lines.messages) {
        parsed.add_message(role, content);
    }

    messages = std::move(parsed.messages);
    system_message = std::move(parsed.system_message);
    total_tokens = parsed.total_tokens;
}

void ConversationContext::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    std::ostringstream buf;
    buf << file.rdbuf();
    file.close();
    deserialize(buf.str());
}

std::string ConversationContext::get_statistics() const {
    std::ostringstream oss;

    oss << "Conversation Statistics:\n";
    oss << "  Messages: " << messages.size() << "\n";
    oss << "  Total Tokens: " << total_tokens << "\n";
    oss << "  System Message: " << (system_message.has_value() ? "Yes" : "No") << "\n";

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
    summarized.token_counter = token_counter;  // TD-294

    // Copy system message
    if (system_message.has_value()) {
        summarized.set_system_message(system_message->content, system_message->token_count);
    }

    // If we have a summary, add it as a system-like message
    if (!summary_text.empty() && static_cast<int>(messages.size()) > keep_recent) {
        // TD-294: counted like any message when there is a counter (0 = count it).
        summarized.add_message("system", "[Summary of earlier conversation: " + summary_text + "]",
                               token_counter ? 0 : estimate_tokens(summary_text) + 10);
    }

    // Add recent messages
    int start_idx = std::max(0, static_cast<int>(messages.size()) - keep_recent);
    for (int i = start_idx; i < static_cast<int>(messages.size()); ++i) {
        const auto& msg = messages[i];
        summarized.add_message(msg.role, msg.content, msg.token_count);
    }

    return summarized;
}

// TD-294: a byte-based guess, used only when no token counter is set (set_token_counter()).
int ConversationContext::estimate_tokens(const std::string& content) {
    // Simple estimation: ~4 characters per token (rough BPE approximation)
    // Add some overhead for special tokens and formatting
    int char_count = static_cast<int>(content.length());
    int estimated = std::max(1, static_cast<int>(std::ceil(char_count / 4.0)));

    // Add overhead for spaces and punctuation
    int space_count = static_cast<int>(std::count(content.begin(), content.end(), ' '));
    estimated += space_count / 2;

    return estimated;
}

void ConversationContext::set_token_counter(TokenCounter counter) {
    token_counter = std::move(counter);
    if (system_message.has_value()) {
        system_message->token_count = count_message_tokens("system", system_message->content);
    }
    for (auto& msg : messages) {
        msg.token_count = count_message_tokens(msg.role, msg.content);
    }
    update_token_count();
    truncate_to_limits();
}

int ConversationContext::count_message_tokens(const std::string& role,
                                              const std::string& content) const {
    if (!token_counter) {
        return estimate_tokens(content);
    }
    // The line format_for_model() writes for this message, with its default separator.
    std::string line = role;
    if (!line.empty()) {
        line[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(line[0])));
    }
    line += ": " + content + "\n";
    return std::max(1, token_counter(line));
}

// TD-294: used by set_token_counter() (TD-296 noted it was never called).
void ConversationContext::update_token_count() {
    total_tokens = 0;

    if (system_message.has_value()) {
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
