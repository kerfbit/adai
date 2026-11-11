// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-11-11

#include <QApplication>
#include <iostream>
#include <string>
#include "Config.hpp"
#include "MnsManagerGUI.hpp"

namespace {

void print_usage(const char* program, std::ostream& out) {
    out << "ADAI Model Name Service Manager (GUI)\n\n"
        << "Usage: " << program << " [--url URL] [--config PATH]\n\n"
        << "Options:\n"
        << "  --url URL       MNS server URL (default: NAME_SERVICE_URL from the config, else "
           "http://localhost:8083)\n"
        << "  --config PATH   Config file to read NAME_SERVICE_URL from (default: config.mns.conf, "
           "found as every ADAI binary finds its config)\n"
        << "  --help, -h      Show this help\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    // TD-568: the arguments are read before QApplication exists, so --help works without a
    // display, and a mistake exits 1 instead of opening the window on the default server.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0], std::cout);
            return 0;
        }
    }
    std::string server_url;
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--url" || arg == "--config") {
            if (i + 1 >= argc || std::string(argv[i + 1]).empty()) {
                std::cerr << "Missing value for " << arg << "\n\n";
                print_usage(argv[0], std::cerr);
                return 1;
            }
            (arg == "--url" ? server_url : config_path) = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n\n";
            print_usage(argv[0], std::cerr);
            return 1;
        }
    }

    QApplication app(argc, argv);
    QApplication::setApplicationName("ADAI MNS Manager");
    QApplication::setApplicationVersion("1.0");
    QApplication::setOrganizationName("ADAI");

    if (server_url.empty()) {
        // Discovery: --config > ./config.mns.conf > /etc/adai/config.mns.conf
        // > ./config.conf (legacy) > /etc/adai/config.conf (legacy).
        config_path = adai::ConfigLoader::discover_config_path(config_path, "config.mns.conf");
        adai::ServiceConfig cfg = adai::ConfigLoader::load(config_path);
        if (!cfg.name_service_url.empty())
            server_url = cfg.name_service_url;
    }
    if (server_url.empty())
        server_url = "http://localhost:8083";

    MnsManagerGUI window(server_url);
    window.show();

    return app.exec();
}
