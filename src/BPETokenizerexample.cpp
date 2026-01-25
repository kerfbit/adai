
#include <fstream>
#include <sstream>
#include "BPETokenizer.hpp"

// Example usage and testing
int main() {
    // Create tokenizer
    BPETokenizer tokenizer;

    // Read training data from trimmed text file
    std::vector<std::string> training_texts;
    std::ifstream file("texts/pg10_trimmed.txt");

    if (!file.is_open()) {
        std::cerr << "Error: Could not open texts/pg10_trimmed.txt\n";
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            training_texts.push_back(line);
        }
    }
    file.close();

    if (training_texts.empty()) {
        std::cerr << "Error: No training data found in file\n";
        return 1;
    }

    std::cout << "Loaded " << training_texts.size() << " lines from training file\n";
    std::cout << "Building vocabulary from training data...\n";
    tokenizer.build_vocab(training_texts);

    tokenizer.print_vocab_stats();

    // Test tokenization
    std::string test_text = "Hello, this is a fascinating test of tokenization!";
    std::cout << "\nOriginal text: " << test_text << std::endl;

    auto tokens = tokenizer.tokenize(test_text);
    std::cout << "Tokens: ";
    for (const auto& token : tokens) {
        std::cout << "'" << token << "' ";
    }
    std::cout << std::endl;

    // Test encoding/decoding
    auto ids = tokenizer.encode(test_text);
    std::cout << "Token IDs: ";
    for (int id : ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    std::string decoded = tokenizer.decode(ids);
    std::cout << "Decoded text: " << decoded << std::endl;

    // Show top tokens
    std::cout << "\nTop 250 tokens by ID:\n";
    auto top_tokens = tokenizer.get_top_tokens(250);
    for (const auto& pair : top_tokens) {
        std::cout << "ID " << pair.second << ": '" << pair.first << "'\n";
    }

    // Test with unknown words
    std::cout << "\nTesting with unknown words:\n";
    std::string unknown_text = "Supercalifragilisticexpialidocious";
    auto unknown_tokens = tokenizer.tokenize(unknown_text);
    std::cout << "Unknown word tokens: ";
    for (const auto& token : unknown_tokens) {
        std::cout << "'" << token << "' ";
    }
    std::cout << std::endl;

    // Save vocabulary
    std::cout << "\n=== Saving Vocabulary ===\n";
    tokenizer.save_vocab("vocab.txt");

    // Load vocabulary into a new tokenizer
    std::cout << "\n=== Loading Vocabulary into New Tokenizer ===\n";
    BPETokenizer loaded_tokenizer;
    loaded_tokenizer.load_vocab("vocab.txt");

    // Test that loaded tokenizer works the same
    std::cout << "\n=== Testing Loaded Tokenizer ===\n";
    std::string test_text2 = "Hello, this is a fascinating test of tokenization!";
    auto loaded_tokens = loaded_tokenizer.tokenize(test_text2);
    auto loaded_ids = loaded_tokenizer.encode(test_text2);
    std::string loaded_decoded = loaded_tokenizer.decode(loaded_ids);

    std::cout << "Original text: " << test_text2 << std::endl;
    std::cout << "Decoded text:  " << loaded_decoded << std::endl;

    // Verify they produce the same results
    bool same_tokens = (tokens == loaded_tokens);
    bool same_ids = (ids == loaded_ids);
    bool same_decoded = (decoded == loaded_decoded);

    std::cout << "\n=== Verification ===\n";
    std::cout << "Tokens match: " << (same_tokens ? "YES" : "NO") << std::endl;
    std::cout << "IDs match: " << (same_ids ? "YES" : "NO") << std::endl;
    std::cout << "Decoded text matches: " << (same_decoded ? "YES" : "NO") << std::endl;

    return 0;
}
