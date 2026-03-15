#include <iostream>
#include <iomanip>
#include "src/EncoderDecoderModel.hpp"
#include "src/BPETokenizer.hpp"

int main() {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    BPETokenizer* tokenizer = model.get_tokenizer();
    std::vector<std::string> corpus = {"hello world", "how are you", "test message"};
    tokenizer->build_vocab(corpus, vocab_size);
    
    std::cout << "Actual vocab size: " << tokenizer->get_vocab_size() << std::endl;
    std::cout << "Model vocab size: " << vocab_size << std::endl;
    std::cout << "BOS token ID: " << tokenizer->get_bos_token_id() << std::endl;
    std::cout << "EOS token ID: " << tokenizer->get_eos_token_id() << std::endl;
    
    model.set_training(false);
    
    // Test greedy with detailed output
    std::cout << "\n=== Testing Greedy ===" << std::endl;
    std::string greedy_response = model.generate_response_with_strategy(
        "hello", 10, "greedy", 1.0f, 50, 0.9f, 3);
    std::cout << "Response length: " << greedy_response.length() << std::endl;
    std::cout << "Response (as hex): ";
    for (unsigned char c : greedy_response) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)c << " ";
    }
    std::cout << std::dec << std::endl;
    std::cout << "Response (raw): [" << greedy_response << "]" << std::endl;
    
    // Test beam
    std::cout << "\n=== Testing Beam ===" << std::endl;
    std::string beam_response = model.generate_response_with_strategy(
        "hello", 10, "beam", 1.0f, 50, 0.9f, 3);
    std::cout << "Response length: " << beam_response.length() << std::endl;
    std::cout << "Response (as hex): ";
    for (unsigned char c : beam_response) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)c << " ";
    }
    std::cout << std::dec << std::endl;
    std::cout << "Response (raw): [" << beam_response << "]" << std::endl;
    
    return 0;
}
