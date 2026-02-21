#include "ChatbotCLI.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Default paths
    std::string vocab_path = "vocab.txt";
    std::string model_path = "latest_checkpoint.bin";
    std::string conv_save_path = "conversation_history.txt";

    // Parse command line arguments
    if (argc > 1) {
        vocab_path = argv[1];
    }
    if (argc > 2) {
        model_path = argv[2];
    }
    if (argc > 3) {
        conv_save_path = argv[3];
    }

    // Show usage if help requested
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: " << argv[0] << " [vocab_file] [model_file] [conversation_save_file]"
                  << std::endl;
        std::cout << std::endl;
        std::cout << "Default values:" << std::endl;
        std::cout << "  vocab_file: vocab.txt" << std::endl;
        std::cout << "  model_file: latest_checkpoint.bin" << std::endl;
        std::cout << "  conversation_save_file: conversation_history.txt" << std::endl;
        return 0;
    }

    // Create and run chatbot
    ChatbotCLI chatbot(vocab_path, model_path, conv_save_path);
    chatbot.run();

    return 0;
}
