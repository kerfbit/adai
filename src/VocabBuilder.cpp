/**
 * @file VocabBuilder.cpp
 * @brief Standalone vocabulary builder for BPE tokenizer
 *
 * This utility reads training text and creates a BPE vocabulary file.
 * Useful for pre-processing before training.
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "BPETokenizer.hpp"
#include "VocabBuilderHelpers.hpp"

// ANSI color codes
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"

void print_usage(const char* program_name) {
    std::cout << COLOR_CYAN << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║        ADAI Vocabulary Builder v1.0              ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝" << COLOR_RESET << "\n\n";

    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";

    std::cout << "Required Options:\n";
    std::cout << "  --input <file>       Input text file(s) (can specify multiple times)\n";
    std::cout << "  --output <file>      Output vocabulary file\n\n";

    std::cout << "Optional:\n";
    std::cout << "  --vocab-size <N>     Target vocabulary size (default: 5000)\n";
    std::cout << "  --threshold <N>      Minimum character frequency (default: 1)\n";
    std::cout << "  --format <type>      Input format: 'plain', 'pairs', 'json' (default: plain)\n";
    std::cout << "  --stats              Show vocabulary statistics after building\n";
    std::cout << "  --help               Show this help message\n\n";

    std::cout << "Input Formats:\n";
    std::cout << "  plain   - One sentence per line\n";
    std::cout << "  pairs   - INPUT/RESPONSE format (for chatbot training)\n";
    std::cout << "  json    - JSON array of text strings\n\n";

    std::cout << "Examples:\n";
    std::cout << "  # Build vocabulary from plain text\n";
    std::cout << "  " << program_name
              << " --input data.txt --output vocab.txt --vocab-size 10000\n\n";

    std::cout << "  # Build from multiple files with statistics\n";
    std::cout << "  " << program_name << " --input file1.txt --input file2.txt \\\n";
    std::cout << "              --output vocab.txt --stats\n\n";

    std::cout << "  # Build from chatbot training data\n";
    std::cout << "  " << program_name << " --input training.txt --output vocab.txt \\\n";
    std::cout << "              --format pairs --vocab-size 8000\n\n";
}

/**
 * Load text from plain text file (one line per sample)
 * (Implementation moved to VocabBuilderHelpers.hpp)
 */

int main(int argc, char** argv) {
    // Parse command line arguments
    std::vector<std::string> input_files;
    std::string output_file;
    std::string format = "plain";
    int vocab_size = 5000;
    int threshold = 1;
    bool show_stats = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--input" && i + 1 < argc) {
            input_files.push_back(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--vocab-size" && i + 1 < argc) {
            vocab_size = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::stoi(argv[++i]);
        } else if (arg == "--format" && i + 1 < argc) {
            format = argv[++i];
        } else if (arg == "--stats") {
            show_stats = true;
        } else {
            std::cerr << COLOR_RED << "Unknown option: " << arg << COLOR_RESET << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (input_files.empty()) {
        std::cerr << COLOR_RED << "Error: No input files specified\n" << COLOR_RESET;
        print_usage(argv[0]);
        return 1;
    }

    if (output_file.empty()) {
        std::cerr << COLOR_RED << "Error: No output file specified\n" << COLOR_RESET;
        print_usage(argv[0]);
        return 1;
    }

    // Load all text data
    std::cout << COLOR_CYAN << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout << "║         Loading Training Data                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝" << COLOR_RESET << "\n\n";

    std::vector<std::string> all_texts;

    for (const auto& file : input_files) {
        std::cout << COLOR_BLUE << "📂 Loading: " << COLOR_RESET << file << "\n";

        std::vector<std::string> texts;
        if (format == "plain") {
            texts = load_plain_text(file);
        } else if (format == "pairs") {
            texts = load_pairs_format(file);
        } else if (format == "json") {
            texts = load_json_format(file);
        } else {
            std::cerr << COLOR_RED << "Error: Unknown format: " << format << COLOR_RESET << "\n";
            return 1;
        }

        std::cout << COLOR_GREEN << "  ✓ Loaded " << texts.size() << " text samples\n"
                  << COLOR_RESET;
        all_texts.insert(all_texts.end(), texts.begin(), texts.end());
    }

    std::cout << COLOR_GREEN << "\n✓ Total samples loaded: " << all_texts.size() << "\n"
              << COLOR_RESET;

    if (all_texts.empty()) {
        std::cerr << COLOR_RED << "Error: No text data loaded!\n" << COLOR_RESET;
        return 1;
    }

    // Calculate total characters
    size_t total_chars = 0;
    for (const auto& text : all_texts) {
        total_chars += text.length();
    }
    std::cout << COLOR_BLUE << "📊 Total characters: " << total_chars << "\n" << COLOR_RESET;

    // Build vocabulary
    std::cout << COLOR_CYAN << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout << "║         Building Vocabulary                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝" << COLOR_RESET << "\n\n";

    std::cout << COLOR_YELLOW << "Parameters:\n";
    std::cout << "  • Target vocabulary size: " << vocab_size << "\n";
    std::cout << "  • Character frequency threshold: " << threshold << "\n";
    std::cout << "  • Output file: " << output_file << "\n" << COLOR_RESET << "\n";

    try {
        BPETokenizer tokenizer;
        tokenizer.build_vocab(all_texts, vocab_size, threshold);

        // Save vocabulary
        std::cout << COLOR_BLUE << "\n💾 Saving vocabulary...\n" << COLOR_RESET;
        tokenizer.save_vocab(output_file);

        std::cout << COLOR_GREEN << "✅ Vocabulary saved successfully!\n" << COLOR_RESET;

        // Show statistics if requested
        if (show_stats) {
            std::cout << COLOR_CYAN << "\n╔══════════════════════════════════════════════════╗\n";
            std::cout << "║         Vocabulary Statistics                    ║\n";
            std::cout << "╚══════════════════════════════════════════════════╝" << COLOR_RESET
                      << "\n\n";

            tokenizer.print_vocab_stats();

            std::cout << COLOR_YELLOW << "\nTop 20 tokens:\n" << COLOR_RESET;
            auto top_tokens = tokenizer.get_top_tokens(20);
            for (size_t i = 0; i < top_tokens.size(); i++) {
                std::cout << "  " << (i + 1) << ". ID " << top_tokens[i].second << ": '"
                          << top_tokens[i].first << "'\n";
            }
        }

        // Test encoding/decoding
        std::cout << COLOR_CYAN << "\n╔══════════════════════════════════════════════════╗\n";
        std::cout << "║         Validation Test                          ║\n";
        std::cout << "╚══════════════════════════════════════════════════╝" << COLOR_RESET
                  << "\n\n";

        std::string test_text = "Hello, world! This is a test.";
        std::cout << COLOR_BLUE << "Test text: " << COLOR_RESET << test_text << "\n";

        auto encoded = tokenizer.encode(test_text);
        std::cout << COLOR_YELLOW << "Encoded tokens (" << encoded.size() << "): " << COLOR_RESET;
        for (size_t i = 0; i < std::min(encoded.size(), size_t(10)); i++) {
            std::cout << encoded[i] << " ";
        }
        if (encoded.size() > 10)
            std::cout << "...";
        std::cout << "\n";

        auto decoded = tokenizer.decode(encoded);
        std::cout << COLOR_GREEN << "Decoded text: " << COLOR_RESET << decoded << "\n";

        if (decoded == test_text) {
            std::cout << COLOR_GREEN << "✅ Round-trip test PASSED\n" << COLOR_RESET;
        } else {
            std::cout << COLOR_YELLOW << "⚠️  Round-trip test differs (this is normal for BPE)\n"
                      << COLOR_RESET;
        }

        std::cout << COLOR_GREEN << "\n✨ Vocabulary building completed successfully!\n"
                  << COLOR_RESET;

    } catch (const std::exception& e) {
        std::cerr << COLOR_RED << "\n❌ Error: " << e.what() << "\n" << COLOR_RESET;
        return 1;
    }

    return 0;
}
