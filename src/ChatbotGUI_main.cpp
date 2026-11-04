// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-11-04

#include <QApplication>
#include <iostream>
#include <string>
#include "ChatbotGUI.hpp"

namespace {

void print_usage(const char* program, std::ostream& out) {
    out << "Usage: " << program << " [server_url]" << '\n';
    out << '\n';
    out << "A desktop client for chatbot_api_server, like the chatbot CLI: messages go to the model" << '\n';
    out << "the server serves, and the server keeps the conversation." << '\n';
    out << '\n';
    out << "Default values:" << '\n';
    out << "  server_url: http://localhost:8080" << '\n';
    out << '\n';
    out << "Example: " << program << " http://ai-machine:8080" << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    // TD-266: --help is handled before QApplication exists, so it works without a display.
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0], std::cout);
        return 0;
    }
    // TD-260: the GUI used to take [vocab_file] [model_file] and run its own copy of the model.
    if (argc > 2) {
        std::cerr << "chatbot_gui takes one argument, the server's URL; it no longer loads a vocabulary "
                     "or model itself (chatbot_api_server does)."
                  << '\n'
                  << '\n';
        print_usage(argv[0], std::cerr);
        return 1;
    }
    const std::string server_url = argc > 1 ? argv[1] : "http://localhost:8080";

    // Create Qt application
    QApplication app(argc, argv);

    // Set application metadata
    QApplication::setApplicationName("ADAI Chatbot");
    QApplication::setApplicationVersion("1.0");
    QApplication::setOrganizationName("ADAI");

    // Create and show the main window
    ChatbotGUI window(server_url);
    window.show();

    // Run the application event loop
    return app.exec();
}
