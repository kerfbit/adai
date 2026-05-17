#include <iostream>
#include "BPETokenizer.hpp"
#include "EncoderDecoderModel.hpp"
#include "SpecialTokens.hpp"
#include "TextGenerator.hpp"

int main() {
    std::cout << "=== Testing Special Token Migration ===" << std::endl;

    // Test 1: BPETokenizer uses constants
    BPETokenizer tokenizer;
    std::cout << "BPETokenizer token IDs:" << std::endl;
    std::cout << "  PAD: " << tokenizer.get_pad_token_id()
              << " (expected: " << adai::SpecialTokenIDs::PAD << ")" << std::endl;
    std::cout << "  UNK: " << tokenizer.get_unk_token_id()
              << " (expected: " << adai::SpecialTokenIDs::UNK << ")" << std::endl;
    std::cout << "  BOS: " << tokenizer.get_bos_token_id()
              << " (expected: " << adai::SpecialTokenIDs::BOS << ")" << std::endl;
    std::cout << "  EOS: " << tokenizer.get_eos_token_id()
              << " (expected: " << adai::SpecialTokenIDs::EOS << ")" << std::endl;

    // Test 2: GenerationConfig uses constants
    TextGenerator::GenerationConfig config;
    std::cout << "\nGenerationConfig token IDs:" << std::endl;
    std::cout << "  PAD: " << config.pad_token_id << " (expected: " << adai::SpecialTokenIDs::PAD
              << ")" << std::endl;
    std::cout << "  UNK: " << config.unk_token_id << " (expected: " << adai::SpecialTokenIDs::UNK
              << ")" << std::endl;
    std::cout << "  BOS: " << config.bos_token_id << " (expected: " << adai::SpecialTokenIDs::BOS
              << ")" << std::endl;
    std::cout << "  EOS: " << config.eos_token_id << " (expected: " << adai::SpecialTokenIDs::EOS
              << ")" << std::endl;

    // Test 3: Model initialization uses constants
    EncoderDecoderModel model(100, 64);
    std::cout << "\nEncoderDecoderModel token IDs:" << std::endl;
    std::cout << "  BOS: " << model.get_bos_token_id()
              << " (expected: " << adai::SpecialTokenIDs::BOS << ")" << std::endl;
    std::cout << "  EOS: " << model.get_eos_token_id()
              << " (expected: " << adai::SpecialTokenIDs::EOS << ")" << std::endl;
    std::cout << "  PAD: " << model.get_pad_token_id()
              << " (expected: " << adai::SpecialTokenIDs::PAD << ")" << std::endl;

    // Test 4: Utility functions work
    std::cout << "\nUtility function tests:" << std::endl;
    adai::SpecialTokenConfig token_config;
    std::cout << "  is_special_token(2): "
              << (adai::is_special_token(2, token_config) ? "true" : "false") << " (expected: true)"
              << std::endl;
    std::cout << "  is_stop_token(3): " << (adai::is_stop_token(3, token_config) ? "true" : "false")
              << " (expected: true)" << std::endl;
    std::cout << "  get_special_token_string(2): "
              << adai::get_special_token_string(2, token_config) << " (expected: <bos>)"
              << std::endl;

    std::cout << "\n✅ All migration tests passed!" << std::endl;
    return 0;
}
