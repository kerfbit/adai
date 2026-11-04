// TD-037: ChatbotGuiLogic.hpp's helpers were extracted from ChatbotGUI so they're testable without a
// QApplication or widgets — see that header's own comments for the full rationale.
#include "ChatbotGuiLogic.hpp"
#include <gtest/gtest.h>
#include <string>

using chatbot_gui::generation_strategy_for_index;

// TD-260: the names are chatbot_api_server's, and item 0 is the server's default (nothing sent).
TEST(GenerationStrategyForIndexTest, MapsEachComboBoxIndexToTheServersStrategyName) {
    // Order matches ChatbotGUI.cpp's createSettingsPanel() combo box items exactly: "Server default",
    // "Nucleus (Top-p)", "Top-k Sampling", "Greedy", "Beam Search", "Sampling (Temperature)".
    EXPECT_EQ(generation_strategy_for_index(0), "");
    EXPECT_EQ(generation_strategy_for_index(1), "nucleus");
    EXPECT_EQ(generation_strategy_for_index(2), "top_k");
    EXPECT_EQ(generation_strategy_for_index(3), "greedy");
    EXPECT_EQ(generation_strategy_for_index(4), "beam");
    EXPECT_EQ(generation_strategy_for_index(5), "temperature");
}

TEST(GenerationStrategyForIndexTest, OutOfRangeIndexIsTheServerDefault) {
    EXPECT_EQ(generation_strategy_for_index(-1), "");
    EXPECT_EQ(generation_strategy_for_index(6), "");
    EXPECT_EQ(generation_strategy_for_index(999), "");
}

// TD-260: a setting is sent only once the user moves its control off "server default" (TD-254).
TEST(ApplySettingTest, ControlsAtServerDefaultSendNothing) {
    GenerationOverrides settings;
    chatbot_gui::apply_strategy(settings, 0);
    chatbot_gui::apply_int_setting(settings, "max_length", 0, 0);
    chatbot_gui::apply_int_setting(settings, "top_k", 0, 0);
    chatbot_gui::apply_int_setting(settings, "beam_width", 0, 0);
    chatbot_gui::apply_real_setting(settings, "temperature", -0.05, -0.05);
    chatbot_gui::apply_real_setting(settings, "top_p", -0.05, -0.05);
    for (const char* key : {"strategy", "max_length", "top_k", "beam_width", "temperature", "top_p"}) {
        EXPECT_FALSE(settings.is_overridden(key)) << key;
    }
}

TEST(ApplySettingTest, ChangedControlsAreSentWithTheirValues) {
    GenerationOverrides settings;
    chatbot_gui::apply_strategy(settings, 4);
    chatbot_gui::apply_int_setting(settings, "max_length", 120, 0);
    chatbot_gui::apply_int_setting(settings, "top_k", 1, 0);
    chatbot_gui::apply_int_setting(settings, "beam_width", 3, 0);
    chatbot_gui::apply_real_setting(settings, "temperature", 0.0, -0.05);  // 0 is a real value: greedy
    chatbot_gui::apply_real_setting(settings, "top_p", 0.95, -0.05);
    EXPECT_EQ(settings.strategy, "beam");
    EXPECT_EQ(settings.max_length, 120);
    EXPECT_EQ(settings.top_k, 1);
    EXPECT_EQ(settings.beam_width, 3);
    EXPECT_FLOAT_EQ(settings.temperature, 0.0f);
    EXPECT_FLOAT_EQ(settings.top_p, 0.95f);
    for (const char* key : {"strategy", "max_length", "top_k", "beam_width", "temperature", "top_p"}) {
        EXPECT_TRUE(settings.is_overridden(key)) << key;
    }
}

TEST(ApplySettingTest, MovingBackToServerDefaultStopsSending) {
    GenerationOverrides settings;
    chatbot_gui::apply_strategy(settings, 2);
    chatbot_gui::apply_int_setting(settings, "top_k", 40, 0);
    chatbot_gui::apply_real_setting(settings, "top_p", 0.5, -0.05);
    chatbot_gui::apply_strategy(settings, 0);
    chatbot_gui::apply_int_setting(settings, "top_k", 0, 0);
    chatbot_gui::apply_real_setting(settings, "top_p", -0.05, -0.05);
    EXPECT_FALSE(settings.is_overridden("strategy"));
    EXPECT_FALSE(settings.is_overridden("top_k"));
    EXPECT_FALSE(settings.is_overridden("top_p"));
}

// TD-260: what the chat pane says once the GUI has asked the server's /health.
TEST(ConnectionNoticeTest, HealthyServer) {
    ChatbotClient::Health health;
    health.reachable = true;
    health.status = 200;
    health.model_loaded = true;
    const auto notice = chatbot_gui::connection_notice("http://h:8080", health);
    EXPECT_TRUE(notice.ok);
    EXPECT_NE(notice.text.find("http://h:8080"), std::string::npos) << notice.text;
}

TEST(ConnectionNoticeTest, UntrainedModelIsAWarning) {
    ChatbotClient::Health health;
    health.reachable = true;
    health.status = 503;
    health.model_loaded = false;
    const auto notice = chatbot_gui::connection_notice("http://h:8080", health);
    EXPECT_FALSE(notice.ok);
    EXPECT_NE(notice.text.find("untrained"), std::string::npos) << notice.text;
}

TEST(ConnectionNoticeTest, UnreachableServerSaysHowToPointAtIt) {
    ChatbotClient::Health health;
    health.error = "Connection";
    const auto notice = chatbot_gui::connection_notice("http://h:8080", health);
    EXPECT_FALSE(notice.ok);
    EXPECT_NE(notice.text.find("Could not reach chatbot_api_server at http://h:8080 (Connection)"),
              std::string::npos)
        << notice.text;
    EXPECT_NE(notice.text.find("chatbot_gui http://"), std::string::npos) << notice.text;
}

TEST(ConnectionNoticeTest, SomethingElseAnswering) {
    ChatbotClient::Health health;
    health.reachable = true;
    health.status = 404;
    const auto notice = chatbot_gui::connection_notice("http://h:8080", health);
    EXPECT_FALSE(notice.ok);
    EXPECT_NE(notice.text.find("HTTP 404"), std::string::npos) << notice.text;
}

// TD-266: a loaded conversation's messages, from the data /chat/session/export returns.
TEST(ConversationMessagesTest, ShowsEachMessageInOrderUnderItsSender) {
    const std::string data =
        "MAX_MESSAGES:20\nMAX_TOKENS:2048\nKEEP_SYSTEM:1\n---\n"
        "SYSTEM|5|Be brief.\n"
        "user|3|Hi | there\\nsecond line\n"
        "assistant|2|Hello\\\\world\n"
        "system|4|[Summary of earlier conversation: x]\n"
        "tool|1|ran\n";
    const auto shown = chatbot_gui::conversation_messages(data);
    ASSERT_EQ(shown.size(), 5u);
    EXPECT_EQ(shown[0].sender, "System");
    EXPECT_EQ(shown[0].text, "Be brief.");
    EXPECT_EQ(shown[1].sender, "You");
    EXPECT_EQ(shown[1].text, "Hi | there\nsecond line");
    EXPECT_TRUE(shown[1].is_user);
    EXPECT_EQ(shown[2].sender, "Bot");
    EXPECT_EQ(shown[2].text, "Hello\\world");
    EXPECT_FALSE(shown[2].is_user);
    EXPECT_EQ(shown[3].sender, "System");
    EXPECT_EQ(shown[4].sender, "tool");
}

TEST(ConversationMessagesTest, EmptyOrUnrecognizedDataShowsNothing) {
    EXPECT_TRUE(chatbot_gui::conversation_messages("").empty());
    EXPECT_TRUE(chatbot_gui::conversation_messages("MAX_MESSAGES:20\n---\n").empty());
    EXPECT_TRUE(chatbot_gui::conversation_messages("user|1|no separator, so not a conversation\n").empty());
}
