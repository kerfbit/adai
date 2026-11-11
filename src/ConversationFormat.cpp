// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10

#include "ConversationFormat.hpp"
#include <sstream>

namespace adai::conversation_format {

// TD-071 (fixed): save_to_file()/load_from_file() use one line per message
// ("role|token_count|content"), read back with std::getline() — so a
// message whose content contains an embedded newline (extremely common for
// a chatbot: a multi-line user message, a multi-paragraph assistant reply)
// used to split across multiple lines on disk. On load, only the first
// line was parsed as the message (truncating its content); every
// continuation line had no '|' delimiter and was silently dropped —
// permanent, silent data loss on every save+load round-trip for any
// multi-line message. Escape '\\', '\n', and '\r' so each message is
// always exactly one line on disk, and unescape on the way back in.
std::string escape_line(const std::string& content) {
    std::string out;
    out.reserve(content.size());
    for (char c : content) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            default:
                out += c;
        }
    }
    return out;
}

std::string unescape_line(const std::string& line) {
    std::string out;
    out.reserve(line.size());
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\\' && i + 1 < line.size()) {
            const char next = line[i + 1];
            if (next == '\\') {
                out += '\\';
                ++i;
                continue;
            }
            if (next == 'n') {
                out += '\n';
                ++i;
                continue;
            }
            if (next == 'r') {
                out += '\r';
                ++i;
                continue;
            }
        }
        out += line[i];
    }
    return out;
}

Parsed parse(const std::string& data) {
    Parsed parsed;
    std::istringstream iss(data);
    std::string line;
    while (std::getline(iss, line)) {
        if (line == "---") {
            parsed.has_separator = true;
            continue;
        }
        // TD-291: the metadata section (MAX_MESSAGES, MAX_TOKENS, KEEP_SYSTEM) is ignored: imported
        // limits used to replace a context's own, and 0 made it unlimited.
        if (!parsed.has_separator) {
            continue;
        }

        // Parse message: role|token_count|content
        const size_t first_pipe = line.find('|');
        if (first_pipe == std::string::npos) {
            continue;
        }
        const size_t second_pipe = line.find('|', first_pipe + 1);
        if (second_pipe == std::string::npos) {
            continue;
        }

        // TD-291: the stored token count is ignored (counts of 1 or negative defeated the budget,
        // and a non-numeric one threw mid-parse); ConversationContext recomputes it.
        std::string role = line.substr(0, first_pipe);
        std::string content = unescape_line(line.substr(second_pipe + 1));
        if (role == "SYSTEM") {
            parsed.system_message = std::move(content);
        } else {
            parsed.messages.emplace_back(std::move(role), std::move(content));
        }
    }
    return parsed;
}

}  // namespace adai::conversation_format
