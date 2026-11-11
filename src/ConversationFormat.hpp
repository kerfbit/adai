#pragma once

// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10

#include <optional>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief The serialized conversation format: what ConversationContext::serialize() writes, the `chatbot`
 * CLI and `chatbot_gui` save, and `/chat/session/export`/`import` carry.
 *
 * Metadata lines, a "---" line, then one line per message, "role|token_count|content", the system
 * message as role "SYSTEM". Content escapes '\\', '\n' and '\r' so each message is one line (TD-071).
 * Shared by ConversationContext (adai_nlp) and chatbot_gui, which links no model library (TD-266).
 */
namespace adai::conversation_format {

/// Escapes '\\', '\n' and '\r' so `content` fits on one line.
std::string escape_line(const std::string& content);
/// Reverses escape_line().
std::string unescape_line(const std::string& line);

struct Parsed {
    bool has_separator = false;                                   // false: not a serialized conversation
    std::optional<std::string> system_message;                    // the "SYSTEM" line, if any
    std::vector<std::pair<std::string, std::string>> messages;    // (role, content), in order
};

/// Parses serialized data. Metadata and stored token counts are ignored (TD-291); a line without two
/// '|' separators is skipped.
Parsed parse(const std::string& data);

}  // namespace adai::conversation_format
