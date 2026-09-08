// @adai-status: beta        (capped by TD-036 — thin main() wrapper, no smoke test)
// @adai-version: 0.7.0
// @adai-reviewed: 2026-09-07

#include <QApplication>
#include <iostream>
#include <string>
#include "ChatbotGUI.hpp"

int main(int argc, char* argv[]) {
    // Create Qt application
    QApplication app(argc, argv);

    // Set application metadata
    QApplication::setApplicationName("ADAI Chatbot");
    QApplication::setApplicationVersion("1.0");
    QApplication::setOrganizationName("ADAI");

    // Default paths
    std::string vocab_path = "vocab.txt";
    std::string model_path = "chatbot_model.bin";

    // Parse command line arguments
    if (argc > 1) {
        vocab_path = argv[1];
    }
    if (argc > 2) {
        model_path = argv[2];
    }

    // Show usage if help requested
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: " << argv[0] << " [vocab_file] [model_file]" << std::endl;
        std::cout << std::endl;
        std::cout << "Default values:" << std::endl;
        std::cout << "  vocab_file: vocab.txt" << std::endl;
        std::cout << "  model_file: chatbot_model.bin" << std::endl;
        std::cout << std::endl;
        std::cout << "Example:" << std::endl;
        std::cout << "  " << argv[0] << " my_vocab.txt my_model.bin" << std::endl;
        return 0;
    }

    // Create and show the main window
    ChatbotGUI window(vocab_path, model_path);
    window.show();

    // Run the application event loop
    return app.exec();
}
