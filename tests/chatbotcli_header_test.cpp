// TD-259: this file includes only ChatbotCLI.hpp, to check what the header brings into its includers,
// and defines the global JSON helper names ChatbotCLI.cpp used to define with external linkage.
#include <string>
#include "../src/ChatbotCLI.hpp"

bool chatbotcli_header_defines_colour_macros() {
#if defined(COLOR_RESET) || defined(COLOR_USER) || defined(COLOR_BOT) || defined(COLOR_SYSTEM) || \
    defined(COLOR_ERROR)
    return true;
#else
    return false;
#endif
}

bool chatbotcli_header_includes_httplib() {
#ifdef CPPHTTPLIB_HTTPLIB_H
    return true;
#else
    return false;
#endif
}

std::string escape_json_string(const std::string& /*input*/) {
    return "escape";
}

std::string unescape_json_string(const std::string& /*s*/) {
    return "unescape";
}

std::string parse_json_value(const std::string& /*json*/, const std::string& /*key*/) {
    return "parse";
}

std::string td259_own_global_json_helper() {
    return escape_json_string("") + "|" + unescape_json_string("") + "|" + parse_json_value("", "");
}
