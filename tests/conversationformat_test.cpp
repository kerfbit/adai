// Tests for adai::conversation_format (TD-266): the serialized conversation line format shared by
// ConversationContext and chatbot_gui.

#include <gtest/gtest.h>
#include "../src/ConversationContext.hpp"
#include "../src/ConversationFormat.hpp"

namespace cf = adai::conversation_format;

TEST(ConversationFormat, EscapeRoundTrips) {
    const std::string content = "a\\b\nc\rd|e\\n";
    const std::string line = cf::escape_line(content);
    EXPECT_EQ(line.find('\n'), std::string::npos);
    EXPECT_EQ(line.find('\r'), std::string::npos);
    EXPECT_EQ(cf::unescape_line(line), content);
    EXPECT_EQ(cf::unescape_line("trailing\\"), "trailing\\");
    EXPECT_EQ(cf::unescape_line("\\t"), "\\t");
}

TEST(ConversationFormat, ParseSkipsMetadataAndMalformedLines) {
    const cf::Parsed parsed = cf::parse(
        "MAX_MESSAGES:20\nuser|1|metadata section, ignored\n---\n"
        "SYSTEM|0|sys\nno pipes here\nuser|only one pipe\nuser|9|hello|world\nassistant|x|hi\n");
    EXPECT_TRUE(parsed.has_separator);
    ASSERT_TRUE(parsed.system_message);
    EXPECT_EQ(*parsed.system_message, "sys");
    ASSERT_EQ(parsed.messages.size(), 2u);
    EXPECT_EQ(parsed.messages[0], std::make_pair(std::string("user"), std::string("hello|world")));
    EXPECT_EQ(parsed.messages[1], std::make_pair(std::string("assistant"), std::string("hi")));
}

TEST(ConversationFormat, NoSeparatorIsNotAConversation) {
    const cf::Parsed parsed = cf::parse("user|1|hello\n");
    EXPECT_FALSE(parsed.has_separator);
    EXPECT_TRUE(parsed.messages.empty());
}

// What ConversationContext writes, the shared parser reads back unchanged.
TEST(ConversationFormat, ParsesWhatConversationContextSerializes) {
    ConversationContext context(20, 4096);
    context.set_system_message("Be brief.\nAlways.");
    context.add_user_message("first\\line\r\nsecond");
    context.add_assistant_message("reply | with pipe");
    const cf::Parsed parsed = cf::parse(context.serialize());
    ASSERT_TRUE(parsed.system_message);
    EXPECT_EQ(*parsed.system_message, "Be brief.\nAlways.");
    ASSERT_EQ(parsed.messages.size(), 2u);
    EXPECT_EQ(parsed.messages[0].first, "user");
    EXPECT_EQ(parsed.messages[0].second, "first\\line\r\nsecond");
    EXPECT_EQ(parsed.messages[1].second, "reply | with pipe");
}
