// Test to verify special token fixes
#include <iostream>
#include <cassert>
#include "BPETokenizer.hpp"
#include "EncoderDecoderModel.hpp"

void test_tokenizer_getters() {
    std::cout << "=== Test 1: BPETokenizer Special Token Getters ===" << std::endl;
    
    BPETokenizer tokenizer;
    
    // Check default special token IDs
    std::cout << "pad_token_id: " << tokenizer.get_pad_token_id() << " (expected 0)" << std::endl;
    std::cout << "unk_token_id: " << tokenizer.get_unk_token_id() << " (expected 1)" << std::endl;
    std::cout << "bos_token_id: " << tokenizer.get_bos_token_id() << " (expected 2)" << std::endl;
    std::cout << "eos_token_id: " << tokenizer.get_eos_token_id() << " (expected 3)" << std::endl;
    
    assert(tokenizer.get_pad_token_id() == 0);
    assert(tokenizer.get_unk_token_id() == 1);
    assert(tokenizer.get_bos_token_id() == 2);
    assert(tokenizer.get_eos_token_id() == 3);
    
    std::cout << "✓ All tokenizer getters work correctly!\n" << std::endl;
}

void test_model_default_ids() {
    std::cout << "=== Test 2: Model Default Special Token IDs ===" << std::endl;
    
    EncoderDecoderModel model(100, 64);
    
    std::cout << "bos_token_id: " << model.get_bos_token_id() << " (expected 2)" << std::endl;
    std::cout << "eos_token_id: " << model.get_eos_token_id() << " (expected 3)" << std::endl;
    std::cout << "pad_token_id: " << model.get_pad_token_id() << " (expected 0)" << std::endl;
    
    assert(model.get_bos_token_id() == 2);
    assert(model.get_eos_token_id() == 3);
    assert(model.get_pad_token_id() == 0);
    
    std::cout << "✓ Model initialized with correct default IDs!\n" << std::endl;
}

void test_tokenizer_sync() {
    std::cout << "=== Test 3: Tokenizer ID Synchronization ===" << std::endl;
    
    EncoderDecoderModel model(100, 64, 2, 2, 8, 2048, 512);
    
    // Create a tokenizer with vocab
    BPETokenizer* tokenizer = new BPETokenizer();
    std::vector<std::string> texts = {"hello world", "this is a test"};
    tokenizer->build_vocab(texts, 50, 1);
    
    std::cout << "Before set_tokenizer:" << std::endl;
    std::cout << "  Model bos_token_id: " << model.get_bos_token_id() << std::endl;
    
    // Transfer tokenizer to model
    model.set_tokenizer(tokenizer);
    
    std::cout << "After set_tokenizer:" << std::endl;
    std::cout << "  Model bos_token_id: " << model.get_bos_token_id() << " (expected 2)" << std::endl;
    std::cout << "  Model eos_token_id: " << model.get_eos_token_id() << " (expected 3)" << std::endl;
    std::cout << "  Model pad_token_id: " << model.get_pad_token_id() << " (expected 0)" << std::endl;
    
    assert(model.get_bos_token_id() == 2);
    assert(model.get_eos_token_id() == 3);
    assert(model.get_pad_token_id() == 0);
    
    std::cout << "✓ Special token IDs synchronized correctly!\n" << std::endl;
}

void test_encoding_behavior() {
    std::cout << "=== Test 4: Encoding Behavior ===" << std::endl;
    
    BPETokenizer tokenizer;
    std::vector<std::string> texts = {"hello world"};
    tokenizer.build_vocab(texts, 30, 1);
    
    // Test encoding with special tokens
    auto tokens_with = tokenizer.encode("hello", true);
    std::cout << "encode('hello', true): [";
    for (size_t i = 0; i < tokens_with.size(); ++i) {
        std::cout << tokens_with[i];
        if (i < tokens_with.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    assert(tokens_with.size() >= 2);
    assert(tokens_with.front() == 2);  // BOS
    assert(tokens_with.back() == 3);   // EOS
    
    // Test encoding without special tokens
    auto tokens_without = tokenizer.encode("hello", false);
    std::cout << "encode('hello', false): [";
    for (size_t i = 0; i < tokens_without.size(); ++i) {
        std::cout << tokens_without[i];
        if (i < tokens_without.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    assert(tokens_without.size() < tokens_with.size());
    assert(tokens_without.front() != 2);  // No BOS
    assert(tokens_without.back() != 3);   // No EOS
    
    std::cout << "✓ Encoding with/without special tokens works correctly!\n" << std::endl;
}

void test_decoding_behavior() {
    std::cout << "=== Test 5: Decoding Behavior ===" << std::endl;
    
    BPETokenizer tokenizer;
    std::vector<std::string> texts = {"hello world"};
    tokenizer.build_vocab(texts, 30, 1);
    
    // Create a sequence with special tokens
    auto encoded = tokenizer.encode("hello", true);  // [BOS, tokens..., EOS]
    
    // Decode with skip_special_tokens=true
    std::string decoded_skip = tokenizer.decode(encoded, true);
    std::cout << "Decoded (skip special): '" << decoded_skip << "'" << std::endl;
    
    // Should not contain special token strings
    assert(decoded_skip.find("<bos>") == std::string::npos);
    assert(decoded_skip.find("<eos>") == std::string::npos);
    
    // Decode with skip_special_tokens=false
    std::string decoded_keep = tokenizer.decode(encoded, false);
    std::cout << "Decoded (keep special): '" << decoded_keep << "'" << std::endl;
    
    std::cout << "✓ Decoding with skip_special_tokens works correctly!\n" << std::endl;
}

int main() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing Special Token Fixes" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    try {
        test_tokenizer_getters();
        test_model_default_ids();
        test_tokenizer_sync();
        test_encoding_behavior();
        test_decoding_behavior();
        
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "✅ ALL TESTS PASSED!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ TEST FAILED: Unknown error" << std::endl;
        return 1;
    }
}
