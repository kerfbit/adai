// @adai-status: beta        (thin main() wrapper, no dedicated test)
// @adai-version: 0.8.0
// @adai-reviewed: 2026-09-07

#include <iostream>
#include <string>
#include "ChatbotCLI.hpp"

int main(int argc, char* argv[]) {
    // Default values
    std::string server_url = "http://localhost:8080";
    std::string conv_save_path = "conversation_history.txt";

    // Show usage if help requested
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: " << argv[0] << " [server_url] [conversation_save_file]" << '\n';
        std::cout << '\n';
        std::cout << "Default values:" << '\n';
        std::cout << "  server_url: http://localhost:8080" << '\n';
        std::cout << "  conversation_save_file: conversation_history.txt" << '\n';
        std::cout << '\n';
        std::cout << "Example: " << argv[0] << " http://localhost:8080" << '\n';
        return 0;
    }

    // Parse command line arguments
    if (argc > 1) {
        server_url = argv[1];
    }
    if (argc > 2) {
        conv_save_path = argv[2];
    }

    // Create and run chatbot
    try {
        ChatbotCLI chatbot(server_url, conv_save_path);
        chatbot.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
