#include "src/BPETokenizer.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "=== Testing BPE Tokenizer Save/Load ===" << std::endl;
    
    // Create training data with special characters
    std::vector<std::string> training_data = {
        "The quick brown fox",
        "jumps over the lazy dog",
        "Hello, world!",
        "Test with spaces	and tabs",
        "Multi-line\ntext\nhere",
        "Special chars: !@#$%"
    };
    
    // Build vocabulary
    std::cout << "\n1. Building vocabulary..." << std::endl;
    BPETokenizer original;
    original.build_vocab(training_data, 500);
    
    // Test encoding with original
    std::string test_text = "Hello, world! Test text.";
    auto original_tokens = original.tokenize(test_text);
    auto original_ids = original.encode(test_text);
    auto original_decoded = original.decode(original_ids);
    
    std::cout << "\n2. Original tokenizer results:" << std::endl;
    std::cout << "   Text: " << test_text << std::endl;
    std::cout << "   Tokens: ";
    for (const auto& tok : original_tokens) {
        std::cout << "'" << tok << "' ";
    }
    std::cout << std::endl;
    std::cout << "   Decoded: " << original_decoded << std::endl;
    
    // Save vocabulary
    std::cout << "\n3. Saving vocabulary..." << std::endl;
    original.save_vocab("test_vocab.txt");
    
    // Load into new tokenizer
    std::cout << "\n4. Loading vocabulary into new tokenizer..." << std::endl;
    BPETokenizer loaded;
    loaded.load_vocab("test_vocab.txt");
    
    // Test with loaded tokenizer
    auto loaded_tokens = loaded.tokenize(test_text);
    auto loaded_ids = loaded.encode(test_text);
    auto loaded_decoded = loaded.decode(loaded_ids);
    
    std::cout << "\n5. Loaded tokenizer results:" << std::endl;
    std::cout << "   Text: " << test_text << std::endl;
    std::cout << "   Tokens: ";
    for (const auto& tok : loaded_tokens) {
        std::cout << "'" << tok << "' ";
    }
    std::cout << std::endl;
    std::cout << "   Decoded: " << loaded_decoded << std::endl;
    
    // Verify they match
    std::cout << "\n6. Verification:" << std::endl;
    bool tokens_match = (original_tokens == loaded_tokens);
    bool ids_match = (original_ids == loaded_ids);
    bool decoded_match = (original_decoded == loaded_decoded);
    
    std::cout << "   Tokens match: " << (tokens_match ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "   IDs match: " << (ids_match ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "   Decoded match: " << (decoded_match ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    if (tokens_match && ids_match && decoded_match) {
        std::cout << "\n✓ ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
