// @adai-status: stable
// @adai-version: 1.4.0
// @adai-reviewed: 2026-11-09

/**
 * @file VocabBuilderHelpers.hpp
 * @brief File-loading helper functions for the VocabBuilder CLI utility.
 *
 * Extracted as a header so the helpers can be unit-tested independently
 * of the main() entry point in VocabBuilder.cpp.
 */

#pragma once

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "DatasetRegistry.hpp"
#include "nlohmann/json.hpp"

/**
 * TD-534: pick a loader for `filename` from its first non-blank line: `{` (JSONL training data)
 * or `INPUT:`/`RESPONSE:` (legacy training data) -> "pairs", `[` -> "json", anything else (or an
 * unreadable/empty file) -> "plain". Training data is JSONL, and reading it as "plain" puts JSON
 * syntax and key names into the vocabulary.
 */
inline std::string detect_input_format(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (line.compare(0, 3, "\xEF\xBB\xBF") == 0)
            line.erase(0, 3);  // UTF-8 BOM
        const size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
            continue;
        if (line[start] == '{' || line.compare(start, 6, "INPUT:") == 0 ||
            line.compare(start, 9, "RESPONSE:") == 0)
            return "pairs";
        if (line[start] == '[')
            return "json";
        return "plain";
    }
    return "plain";
}

/**
 * TD-535: parse a whole-string decimal integer in [min_value, max_value]. Returns false (leaving
 * `out` unchanged) for empty input, trailing characters ("8000x"), overflow or out-of-range values.
 */
inline bool parse_int_in_range(const std::string& text, long min_value, long max_value, int& out) {
    // strtol() would skip leading whitespace and accept a '+'; require a digit or '-' first.
    if (text.empty() || !(text[0] == '-' || (text[0] >= '0' && text[0] <= '9')))
        return false;
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || end != text.c_str() + text.size() || value < min_value ||
        value > max_value || value < INT_MIN || value > INT_MAX)
        return false;
    out = static_cast<int>(value);
    return true;
}

/**
 * Load text from plain text file (one non-empty line per sample).
 */
inline std::vector<std::string> load_plain_text(const std::string& filename) {
    std::vector<std::string> texts;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << "\n";
        return texts;
    }

    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (first && line.compare(0, 3, "\xEF\xBB\xBF") == 0)
            line.erase(0, 3);
        first = false;
        if (!line.empty() && line.back() == '\r')  // TD-537: a CRLF file
            line.pop_back();
        if (!line.empty()) {
            texts.push_back(line);
        }
    }

    file.close();
    return texts;
}

/**
 * Load text from a training data file (JSONL or legacy INPUT:/RESPONSE: format).
 * Returns the concatenated input and response texts for vocabulary building.
 */
// TD-537: DatasetRegistry's parser, the one training reads with, so the vocabulary is learned from
// the same text.
inline std::vector<std::string> load_pairs_format(const std::string& filename) {
    std::vector<std::string> texts;
    std::vector<ConversationPair> pairs;
    DatasetRegistry::load_conversation_pairs(filename, pairs);
    for (const auto& pair : pairs) {
        if (!pair.input.empty())
            texts.push_back(pair.input);
        if (!pair.response.empty())
            texts.push_back(pair.response);
        for (const auto& turn : pair.meta.history)  // TD-295
            texts.push_back(turn.second);
    }
    return texts;
}

/**
 * Load text from a JSON array of strings (["text1", "text2", ...]). TD-536: parsed with
 * nlohmann/json, so escapes are decoded; it used to take every quoted string, object keys
 * included, and an escaped quote ended a string early. Anything but an array of strings is
 * refused (an error, and no texts), as is invalid JSON. Empty strings are skipped.
 */
inline std::vector<std::string> load_json_format(const std::string& filename) {
    std::vector<std::string> texts;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << "\n";
        return texts;
    }

    const nlohmann::json doc = nlohmann::json::parse(file, nullptr, /*allow_exceptions=*/false);
    if (doc.is_discarded()) {
        std::cerr << "Error: " << filename << " is not valid JSON\n";
        return texts;
    }
    if (!doc.is_array()) {
        std::cerr << "Error: " << filename << " must be a JSON array of strings\n";
        return texts;
    }
    for (std::size_t i = 0; i < doc.size(); ++i) {
        if (!doc[i].is_string()) {
            std::cerr << "Error: element " << i << " of " << filename
                      << " is not a string; the json format is an array of strings\n";
            return {};
        }
        std::string text = doc[i].get<std::string>();
        if (!text.empty()) {
            texts.push_back(std::move(text));
        }
    }
    return texts;
}
