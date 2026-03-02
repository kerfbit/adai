/**
 * @file VocabBuilderHelpers.hpp
 * @brief File-loading helper functions for the VocabBuilder CLI utility.
 *
 * Extracted as a header so the helpers can be unit-tested independently
 * of the main() entry point in VocabBuilder.cpp.
 */

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
    while (std::getline(file, line)) {
        if (!line.empty()) {
            texts.push_back(line);
        }
    }

    file.close();
    return texts;
}

/**
 * Load text from INPUT/RESPONSE paired format (chatbot training data).
 */
inline std::vector<std::string> load_pairs_format(const std::string& filename) {
    std::vector<std::string> texts;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << "\n";
        return texts;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("INPUT:") == 0) {
            texts.push_back(line.substr(6));   // Skip "INPUT:"
        } else if (line.find("RESPONSE:") == 0) {
            texts.push_back(line.substr(9));   // Skip "RESPONSE:"
        }
    }

    file.close();
    return texts;
}

/**
 * Load text from JSON array format (expects ["text1", "text2", ...]).
 * Uses a simple quote-scanning parser — not a full JSON library.
 */
inline std::vector<std::string> load_json_format(const std::string& filename) {
    std::vector<std::string> texts;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << "\n";
        return texts;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    size_t pos = 0;
    while ((pos = content.find('"', pos)) != std::string::npos) {
        pos++;  // Skip opening quote
        size_t end = content.find('"', pos);
        if (end == std::string::npos) break;

        std::string text = content.substr(pos, end - pos);
        if (!text.empty()) {
            texts.push_back(text);
        }
        pos = end + 1;
    }

    return texts;
}
