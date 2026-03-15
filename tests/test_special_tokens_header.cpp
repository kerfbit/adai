#include <iostream>
#include <cassert>
#include "SpecialTokens.hpp"

using namespace adai;

void test_constants() {
    std::cout << "=== Test 1: Special Token Constants ===" << std::endl;
    
    assert(SpecialTokenIDs::PAD == 0);
    assert(SpecialTokenIDs::UNK == 1);
    assert(SpecialTokenIDs::BOS == 2);
    assert(SpecialTokenIDs::EOS == 3);
    
    std::cout << "PAD token ID: " << SpecialTokenIDs::PAD << std::endl;
    std::cout << "UNK token ID: " << SpecialTokenIDs::UNK << std::endl;
    std::cout << "BOS token ID: " << SpecialTokenIDs::BOS << std::endl;
    std::cout << "EOS token ID: " << SpecialTokenIDs::EOS << std::endl;
    
    std::cout << "✓ All constants correct!\n" << std::endl;
}

void test_config_struct() {
    std::cout << "=== Test 2: SpecialTokenConfig Struct ===" << std::endl;
    
    // Test default constructor
    SpecialTokenConfig config;
    assert(config.get_pad_token_id() == 0);
    assert(config.get_unk_token_id() == 1);
    assert(config.get_bos_token_id() == 2);
    assert(config.get_eos_token_id() == 3);
    
    std::cout << "Default config:" << std::endl;
    std::cout << "  PAD: " << config.get_pad_token_id() << std::endl;
    std::cout << "  UNK: " << config.get_unk_token_id() << std::endl;
    std::cout << "  BOS: " << config.get_bos_token_id() << std::endl;
    std::cout << "  EOS: " << config.get_eos_token_id() << std::endl;
    
    // Test custom constructor
    SpecialTokenConfig custom(10, 11, 12, 13);
    assert(custom.get_pad_token_id() == 10);
    assert(custom.get_bos_token_id() == 12);
    
    std::cout << "✓ Config struct works correctly!\n" << std::endl;
}

void test_validation() {
    std::cout << "=== Test 3: Config Validation ===" << std::endl;
    
    // Valid config
    SpecialTokenConfig valid;
    try {
        valid.validate();
        std::cout << "✓ Valid config passes validation" << std::endl;
    } catch (...) {
        std::cerr << "✗ Valid config failed validation!" << std::endl;
        assert(false);
    }
    
    // Invalid config - negative ID
    SpecialTokenConfig invalid1(-1, 1, 2, 3);
    try {
        invalid1.validate();
        std::cerr << "✗ Negative ID should fail validation!" << std::endl;
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Negative ID correctly rejected: " << e.what() << std::endl;
    }
    
    // Invalid config - duplicate IDs
    SpecialTokenConfig invalid2(0, 0, 2, 3);
    try {
        invalid2.validate();
        std::cerr << "✗ Duplicate IDs should fail validation!" << std::endl;
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Duplicate IDs correctly rejected: " << e.what() << std::endl;
    }
    
    std::cout << "✓ Validation works correctly!\n" << std::endl;
}

void test_utility_functions() {
    std::cout << "=== Test 4: Utility Functions ===" << std::endl;
    
    SpecialTokenConfig config;
    
    // Test is_special_token
    assert(is_special_token(0, config) == true);   // PAD
    assert(is_special_token(1, config) == true);   // UNK
    assert(is_special_token(2, config) == true);   // BOS
    assert(is_special_token(3, config) == true);   // EOS
    assert(is_special_token(4, config) == false);  // Not special
    std::cout << "✓ is_special_token works correctly" << std::endl;
    
    // Test is_special_token_string
    assert(is_special_token_string("<pad>") == true);
    assert(is_special_token_string("<unk>") == true);
    assert(is_special_token_string("<bos>") == true);
    assert(is_special_token_string("<eos>") == true);
    assert(is_special_token_string("hello") == false);
    std::cout << "✓ is_special_token_string works correctly" << std::endl;
    
    // Test is_stop_token
    assert(is_stop_token(3, config) == true);   // EOS
    assert(is_stop_token(0, config) == true);   // PAD
    assert(is_stop_token(1, config) == false);  // UNK is not a stop token
    assert(is_stop_token(2, config) == false);  // BOS is not a stop token
    std::cout << "✓ is_stop_token works correctly" << std::endl;
    
    // Test get_special_token_string
    assert(get_special_token_string(0, config) == "<pad>");
    assert(get_special_token_string(1, config) == "<unk>");
    assert(get_special_token_string(2, config) == "<bos>");
    assert(get_special_token_string(3, config) == "<eos>");
    std::cout << "✓ get_special_token_string works correctly" << std::endl;
    
    // Test get_special_token_id
    assert(get_special_token_id("<pad>", config) == 0);
    assert(get_special_token_id("<unk>", config) == 1);
    assert(get_special_token_id("<bos>", config) == 2);
    assert(get_special_token_id("<eos>", config) == 3);
    std::cout << "✓ get_special_token_id works correctly" << std::endl;
    
    // Test error handling
    try {
        get_special_token_string(999, config);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ get_special_token_string correctly throws for invalid ID" << std::endl;
    }
    
    try {
        get_special_token_id("invalid", config);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ get_special_token_id correctly throws for invalid string" << std::endl;
    }
    
    std::cout << "✓ All utility functions work correctly!\n" << std::endl;
}

void test_map_creation() {
    std::cout << "=== Test 5: Map Creation Functions ===" << std::endl;
    
    SpecialTokenConfig config;
    
    // Test create_special_token_set
    auto token_set = create_special_token_set();
    assert(token_set.size() == 4);
    assert(token_set.count("<pad>") > 0);
    assert(token_set.count("<unk>") > 0);
    assert(token_set.count("<bos>") > 0);
    assert(token_set.count("<eos>") > 0);
    std::cout << "✓ create_special_token_set works correctly" << std::endl;
    
    // Test create_special_token_map
    auto token_map = create_special_token_map(config);
    assert(token_map.size() == 4);
    assert(token_map["<pad>"] == 0);
    assert(token_map["<unk>"] == 1);
    assert(token_map["<bos>"] == 2);
    assert(token_map["<eos>"] == 3);
    std::cout << "✓ create_special_token_map works correctly" << std::endl;
    
    // Test create_inverse_special_token_map
    auto inverse_map = create_inverse_special_token_map(config);
    assert(inverse_map.size() == 4);
    assert(inverse_map[0] == "<pad>");
    assert(inverse_map[1] == "<unk>");
    assert(inverse_map[2] == "<bos>");
    assert(inverse_map[3] == "<eos>");
    std::cout << "✓ create_inverse_special_token_map works correctly" << std::endl;
    
    std::cout << "✓ All map creation functions work correctly!\n" << std::endl;
}

void test_custom_config() {
    std::cout << "=== Test 6: Custom Configuration ===" << std::endl;
    
    // Create custom config with different IDs
    SpecialTokenConfig custom(100, 101, 102, 103);
    
    // Test that utility functions work with custom config
    assert(is_special_token(100, custom) == true);
    assert(is_special_token(102, custom) == true);
    assert(is_special_token(0, custom) == false);  // Default PAD ID not special in custom
    
    assert(is_stop_token(103, custom) == true);  // Custom EOS
    assert(is_stop_token(100, custom) == true);  // Custom PAD
    assert(is_stop_token(3, custom) == false);   // Default EOS not a stop token in custom
    
    assert(get_special_token_string(102, custom) == "<bos>");
    assert(get_special_token_id("<eos>", custom) == 103);
    
    std::cout << "✓ Custom configuration works correctly!\n" << std::endl;
}

int main() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing SpecialTokens.hpp Header-Only Library" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    try {
        test_constants();
        test_config_struct();
        test_validation();
        test_utility_functions();
        test_map_creation();
        test_custom_config();
        
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
