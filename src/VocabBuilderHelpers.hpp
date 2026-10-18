// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-10-18

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
#include "TrainingSampleMeta.hpp"

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
// TODO: See TD-537 in TECHNICAL_DEBT.md - keeps the \r of CRLF lines.
inline std::vector<std::string> load_plain_text(const std::string& filename) {
    std::vector<std::string> texts;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << "\n";
        return texts;
    }

    std::string line;
    while (std::getline(file, line)) {
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
// TODO: See TD-537 in TECHNICAL_DEBT.md - legacy branch keeps leading spaces and \r, skips indented lines and keeps
// unpaired texts, unlike ChatbotTrainer's parser (verified).
inline std::vector<std::string> load_pairs_format(const std::string& filename) {
    std::vector<std::string> texts;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << "\n";
        return texts;
    }

    // Detect format from first non-empty line
    std::string first_line;
    while (std::getline(file, first_line)) {
        first_line.erase(0, first_line.find_first_not_of(" \t\r\n"));
        if (!first_line.empty())
            break;
    }
    file.seekg(0);

    std::string line;
    if (!first_line.empty() && first_line.front() == '{') {
        // JSONL training format
        while (std::getline(file, line)) {
            if (line.empty() || line.front() != '{')
                continue;
            std::string in, resp;
            SampleMeta meta;
            if (parse_jsonl_sample(line, in, resp, meta)) {
                if (!in.empty())
                    texts.push_back(in);
                if (!resp.empty())
                    texts.push_back(resp);
            }
        }
    } else {
        // Legacy INPUT:/RESPONSE: format
        while (std::getline(file, line)) {
            if (line.find("INPUT:") == 0) {
                texts.push_back(line.substr(6));
            } else if (line.find("RESPONSE:") == 0) {
                texts.push_back(line.substr(9));
            }
        }
    }

    file.close();
    return texts;
}

/**
 * Load text from JSON array format (expects ["text1", "text2", ...]).
 * Uses a simple quote-scanning parser — not a full JSON library.
 */
// TODO: See TD-536 in TECHNICAL_DEBT.md - takes every quoted string (keys too) and doesn't unescape.
inline std::vector<std::string> load_json_format(const std::string& filename) {
    std::vector<std::string> texts;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << "\n";
        return texts;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    size_t pos = 0;
    while ((pos = content.find('"', pos)) != std::string::npos) {
        pos++;  // Skip opening quote
        size_t end = content.find('"', pos);
        if (end == std::string::npos) {
            break;
        }

        std::string text = content.substr(pos, end - pos);
        if (!text.empty()) {
            texts.push_back(text);
        }
        pos = end + 1;
    }

    return texts;
}
