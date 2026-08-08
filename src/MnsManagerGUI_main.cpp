#include <QApplication>
#include <fstream>
#include <iostream>
#include <string>
#include "Config.hpp"
#include "MnsManagerGUI.hpp"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("ADAI MNS Manager");
    QApplication::setApplicationVersion("1.0");
    QApplication::setOrganizationName("ADAI");

    std::string server_url;
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--url") && i + 1 < argc) {
            server_url = argv[++i];
        } else if ((arg == "--config") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "ADAI Model Name Service Manager (GUI)\n\n"
                      << "Usage: " << argv[0] << " [--url URL] [--config PATH]\n\n"
                      << "Options:\n"
                      << "  --url URL       MNS server URL (default: from config or "
                         "http://localhost:8083)\n"
                      << "  --config PATH   Path to config.mns.conf\n"
                      << "  --help          Show this help\n";
            return 0;
        }
    }

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
