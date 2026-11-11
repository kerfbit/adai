#pragma once

// @adai-status: stable        (tests: chatbotguilogicTests, and through the widget in chatbotGuiWidgetTests)
// @adai-version: 1.0.0
// @adai-reviewed: 2026-11-10

#include <string>
#include <vector>
#include "ChatbotClient.hpp"
#include "ConversationFormat.hpp"

namespace chatbot_gui {

// TD-037: ChatbotGUI's generation-strategy QComboBox (see ChatbotGUI.cpp's createSettingsPanel())
// lists strategies in this exact order, and onStrategyChanged(int index) maps the selected index to
// a strategy name through this function, extracted so it's testable without a QApplication/QComboBox.
// TD-260: the GUI sends the name to chatbot_api_server, so these are the server's names, and item 0,
// "Server default", is "" (the strategy isn't sent). An out-of-range index (Qt only ever calls back
// with a real item index) is the server default too.
inline std::string generation_strategy_for_index(int index) {
    switch (index) {
        case 1:
            return "nucleus";
        case 2:
            return "top_k";
        case 3:
            return "greedy";
        case 4:
            return "beam";
        case 5:
            return "temperature";
        default:
            return "";
    }
}

// TD-260: each numeric setting's spin box starts one step below the server's allowed range, where it
// shows "server default" (QSpinBox::setSpecialValueText). These record a control's value in the
// overrides the client sends: at or below `server_default` the setting isn't sent (TD-254), so the
// server's default applies; anything else is sent.
inline void apply_int_setting(GenerationOverrides& settings, const std::string& key, int value,
                              int server_default) {
    if (value <= server_default) {
        settings.reset(key);
    } else if (key == "max_length") {
        settings.set_max_length(value);
    } else if (key == "top_k") {
        settings.set_top_k(value);
    } else if (key == "beam_width") {
        settings.set_beam_width(value);
    }
}

inline void apply_real_setting(GenerationOverrides& settings, const std::string& key, double value,
                               double server_default) {
    // The slack absorbs floating-point error in the spin box's value.
    if (value <= server_default + 1e-6) {
        settings.reset(key);
    } else if (key == "temperature") {
        settings.set_temperature(static_cast<float>(value));
    } else if (key == "top_p") {
        settings.set_top_p(static_cast<float>(value));
    }
}

inline void apply_strategy(GenerationOverrides& settings, int index) {
    const std::string strategy = generation_strategy_for_index(index);
    if (strategy.empty()) {
        settings.reset("strategy");
    } else {
        settings.set_strategy(strategy);
    }
}

// TD-260: what the chat pane says after connecting. `ok` is false when the user should act.
struct ConnectionNotice {
    bool ok = false;
    std::string text;
};

inline ConnectionNotice connection_notice(const std::string& server_url,
                                          const ChatbotClient::Health& health) {
    if (!health.reachable) {
        return {false, "Could not reach chatbot_api_server at " + server_url +
                           (health.error.empty() ? "" : " (" + health.error + ")") +
                           ". Start it, or give its URL: chatbot_gui http://<host>:8080. "
                           "Messages are sent there once it is up."};
    }
    if (health.status == 200) {
        return {true, "Connected to chatbot_api_server at " + server_url + "."};
    }
    if (health.status == 503 && !health.model_loaded) {
        // TD-246: a server started with ALLOW_UNTRAINED_MODEL=true.
        return {false, "Connected to " + server_url +
                           ", but the server is running an untrained model: its replies are random."};
    }
    return {false, server_url + " answered its health check with HTTP " + std::to_string(health.status) +
                       "; is it chatbot_api_server?"};
}

// TD-260: the note shown when the server had forgotten the session a message was sent in.
inline std::string session_lost_notice() {
    return "The server no longer had this conversation (it expired or the server restarted), so "
           "earlier messages are forgotten. This message started a new conversation.";
}

// TD-266: one message of a loaded conversation, as the chat pane shows it.
struct DisplayedMessage {
    std::string sender;
    std::string text;
    bool is_user = false;
};

// TD-266: the server's copy of a conversation (/chat/session/export data) as chat-pane messages, in
// order: the system message first, then each message. "user" is shown as "You", "assistant" as "Bot",
// a system message as "System", and any other role by its own name.
inline std::vector<DisplayedMessage> conversation_messages(const std::string& data) {
    const adai::conversation_format::Parsed parsed = adai::conversation_format::parse(data);
    std::vector<DisplayedMessage> shown;
    if (parsed.system_message) {
        shown.push_back({"System", *parsed.system_message, false});
    }
    for (const auto& [role, content] : parsed.messages) {
        if (role == "user") {
            shown.push_back({"You", content, true});
        } else if (role == "assistant") {
            shown.push_back({"Bot", content, false});
        } else if (role == "system") {
            shown.push_back({"System", content, false});
        } else {
            shown.push_back({role, content, false});
        }
    }
    return shown;
}

}  // namespace chatbot_gui
