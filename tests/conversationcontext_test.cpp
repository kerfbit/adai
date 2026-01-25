#include "../src/ConversationContext.hpp"
#include <../gtest/gtest.h>
#include <cstdio>
#include <fstream>

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(ConversationContextTest, ConstructorDefault) {
    EXPECT_NO_THROW({ ConversationContext context; });
}

TEST(ConversationContextTest, ConstructorWithParameters) {
    EXPECT_NO_THROW({ ConversationContext context(10, 500, true); });
}

TEST(ConversationContextTest, ConstructorSetsLimits) {
    ConversationContext context(15, 1000, false);

    // Limits should be set (we can verify by checking behavior)
    for (int i = 0; i < 20; ++i) {
        context.add_user_message("Message " + std::to_string(i));
    }

    // Should be truncated to 15
    EXPECT_LE(context.get_message_count(), 15);
}

TEST(ConversationContextTest, ConstructorUnlimitedMessages) {
    ConversationContext context(0, 1000);  // Unlimited messages

    for (int i = 0; i < 30; ++i) {
        context.add_user_message("Message");
    }

    EXPECT_GT(context.get_message_count(), 20);  // Should have more than default limit
}

TEST(ConversationContextTest, ConstructorUnlimitedTokens) {
    ConversationContext context(10, 0);  // Unlimited tokens

    // Add messages with many tokens
    for (int i = 0; i < 5; ++i) {
        context.add_user_message(std::string(500, 'a'));  // Very long messages
    }

    EXPECT_GT(context.get_total_tokens(), 500);  // Should have substantial tokens
}

// ============================================================================
// Message Addition Tests
// ============================================================================

TEST(ConversationContextTest, AddUserMessage) {
    ConversationContext context;

    EXPECT_NO_THROW({ context.add_user_message("Hello"); });

    EXPECT_EQ(context.get_message_count(), 1);
}

TEST(ConversationContextTest, AddAssistantMessage) {
    ConversationContext context;

    EXPECT_NO_THROW({ context.add_assistant_message("Hi there"); });

    EXPECT_EQ(context.get_message_count(), 1);
}

TEST(ConversationContextTest, AddMultipleMessages) {
    ConversationContext context;

    context.add_user_message("Message 1");
    context.add_assistant_message("Response 1");
    context.add_user_message("Message 2");
    context.add_assistant_message("Response 2");

    EXPECT_EQ(context.get_message_count(), 4);
}

TEST(ConversationContextTest, AddMessageWithRole) {
    ConversationContext context;

    context.add_message("user", "User message");
    context.add_message("assistant", "Assistant message");
    context.add_message("system", "System message");

    EXPECT_EQ(context.get_message_count(), 3);
}

TEST(ConversationContextTest, AddMessageWithTokenCount) {
    ConversationContext context;

    context.add_user_message("Hello", 5);

    EXPECT_EQ(context.get_total_tokens(), 5);
}

TEST(ConversationContextTest, AddMessageAutoEstimateTokens) {
    ConversationContext context;

    context.add_user_message("Hello world");  // Auto-estimate

    EXPECT_GT(context.get_total_tokens(), 0);
}

TEST(ConversationContextTest, SetSystemMessage) {
    ConversationContext context;

    EXPECT_NO_THROW({ context.set_system_message("You are a helpful assistant."); });

    EXPECT_FALSE(context.get_system_message().empty());
}

TEST(ConversationContextTest, UpdateSystemMessage) {
    ConversationContext context;

    context.set_system_message("First system message");
    int first_tokens = context.get_total_tokens();

    context.set_system_message("Second system message");
    int second_tokens = context.get_total_tokens();

    // System message should be replaced, not added
    EXPECT_EQ(context.get_system_message(), "Second system message");
}

// ============================================================================
// Message Retrieval Tests
// ============================================================================

TEST(ConversationContextTest, GetLastUserMessage) {
    ConversationContext context;

    context.add_user_message("First user message");
    context.add_assistant_message("Assistant response");
    context.add_user_message("Second user message");

    EXPECT_EQ(context.get_last_user_message(), "Second user message");
}

TEST(ConversationContextTest, GetLastAssistantMessage) {
    ConversationContext context;

    context.add_assistant_message("First assistant message");
    context.add_user_message("User message");
    context.add_assistant_message("Second assistant message");

    EXPECT_EQ(context.get_last_assistant_message(), "Second assistant message");
}

TEST(ConversationContextTest, GetLastUserMessageThrowsWhenEmpty) {
    ConversationContext context;

    EXPECT_THROW({ context.get_last_user_message(); }, std::runtime_error);
}

TEST(ConversationContextTest, GetLastAssistantMessageThrowsWhenEmpty) {
    ConversationContext context;

    EXPECT_THROW({ context.get_last_assistant_message(); }, std::runtime_error);
}

TEST(ConversationContextTest, GetMessages) {
    ConversationContext context;

    context.add_user_message("Message 1");
    context.add_assistant_message("Response 1");
    context.add_user_message("Message 2");

    std::vector<ConversationContext::Message> messages = context.get_messages();

    EXPECT_EQ(messages.size(), 3);
    EXPECT_EQ(messages[0].role, "user");
    EXPECT_EQ(messages[0].content, "Message 1");
    EXPECT_EQ(messages[1].role, "assistant");
    EXPECT_EQ(messages[2].role, "user");
}

TEST(ConversationContextTest, GetSystemMessage) {
    ConversationContext context;

    EXPECT_TRUE(context.get_system_message().empty());

    context.set_system_message("System prompt");

    EXPECT_EQ(context.get_system_message(), "System prompt");
}

// ============================================================================
// Formatting Tests
// ============================================================================

TEST(ConversationContextTest, FormatForModelBasic) {
    ConversationContext context;

    context.add_user_message("Hello");
    context.add_assistant_message("Hi there");

    std::string formatted = context.format_for_model(false);

    EXPECT_NE(formatted.find("User:"), std::string::npos);
    EXPECT_NE(formatted.find("Assistant:"), std::string::npos);
    EXPECT_NE(formatted.find("Hello"), std::string::npos);
    EXPECT_NE(formatted.find("Hi there"), std::string::npos);
}

TEST(ConversationContextTest, FormatForModelWithSystem) {
    ConversationContext context;

    context.set_system_message("You are helpful");
    context.add_user_message("Hello");

    std::string formatted = context.format_for_model(true);

    EXPECT_NE(formatted.find("System:"), std::string::npos);
    EXPECT_NE(formatted.find("You are helpful"), std::string::npos);
}

TEST(ConversationContextTest, FormatForModelWithoutSystem) {
    ConversationContext context;

    context.set_system_message("You are helpful");
    context.add_user_message("Hello");

    std::string formatted = context.format_for_model(false);

    EXPECT_EQ(formatted.find("System:"), std::string::npos);
}

TEST(ConversationContextTest, FormatWithSpecialTokens) {
    ConversationContext context;

    context.set_system_message("System prompt");
    context.add_user_message("Hello");
    context.add_assistant_message("Hi");

    std::string formatted = context.format_with_special_tokens("<bos>", "<eos>", "<sep>");

    EXPECT_NE(formatted.find("<bos>"), std::string::npos);
    EXPECT_NE(formatted.find("<eos>"), std::string::npos);
    EXPECT_NE(formatted.find("<sep>"), std::string::npos);
    EXPECT_NE(formatted.find("[SYSTEM]"), std::string::npos);
    EXPECT_NE(formatted.find("[USER]"), std::string::npos);
    EXPECT_NE(formatted.find("[ASSISTANT]"), std::string::npos);
}

TEST(ConversationContextTest, FormatEmptyConversation) {
    ConversationContext context;

    std::string formatted = context.format_for_model(false);

    EXPECT_TRUE(formatted.empty() || formatted == "\n" || formatted == "");
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(ConversationContextTest, GetTotalTokens) {
    ConversationContext context;

    EXPECT_EQ(context.get_total_tokens(), 0);

    context.add_user_message("Hello", 5);
    EXPECT_EQ(context.get_total_tokens(), 5);

    context.add_assistant_message("Hi", 3);
    EXPECT_EQ(context.get_total_tokens(), 8);
}

TEST(ConversationContextTest, GetMessageCount) {
    ConversationContext context;

    EXPECT_EQ(context.get_message_count(), 0);

    context.add_user_message("Message 1");
    EXPECT_EQ(context.get_message_count(), 1);

    context.add_assistant_message("Response 1");
    EXPECT_EQ(context.get_message_count(), 2);
}

TEST(ConversationContextTest, IsEmpty) {
    ConversationContext context;

    EXPECT_TRUE(context.is_empty());

    context.add_user_message("Message");
    EXPECT_FALSE(context.is_empty());

    context.clear();
    EXPECT_TRUE(context.is_empty());
}

TEST(ConversationContextTest, GetStatistics) {
    ConversationContext context(20, 2048);

    context.set_system_message("System");
    context.add_user_message("User 1");
    context.add_user_message("User 2");
    context.add_assistant_message("Assistant 1");

    std::string stats = context.get_statistics();

    EXPECT_NE(stats.find("Messages:"), std::string::npos);
    EXPECT_NE(stats.find("Total Tokens:"), std::string::npos);
    EXPECT_NE(stats.find("User Messages:"), std::string::npos);
    EXPECT_NE(stats.find("Assistant Messages:"), std::string::npos);
}

// ============================================================================
// Truncation Tests
// ============================================================================

TEST(ConversationContextTest, AutomaticMessageTruncation) {
    ConversationContext context(3, 0);  // Max 3 messages

    for (int i = 0; i < 5; ++i) {
        context.add_user_message("Message " + std::to_string(i));
    }

    EXPECT_EQ(context.get_message_count(), 3);

    // Should keep most recent messages
    auto messages = context.get_messages();
    EXPECT_EQ(messages[0].content, "Message 2");
    EXPECT_EQ(messages[1].content, "Message 3");
    EXPECT_EQ(messages[2].content, "Message 4");
}

TEST(ConversationContextTest, AutomaticTokenTruncation) {
    ConversationContext context(0, 50);  // Max 50 tokens

    context.add_user_message("Message 1", 20);
    context.add_assistant_message("Response 1", 20);
    context.add_user_message("Message 2", 20);  // Should trigger truncation

    EXPECT_LE(context.get_total_tokens(), 50);
    EXPECT_LT(context.get_message_count(), 3);  // Some messages removed
}

TEST(ConversationContextTest, SystemMessageNotTruncated) {
    ConversationContext context(3, 0, true);  // Keep system message

    context.set_system_message("System prompt");

    for (int i = 0; i < 5; ++i) {
        context.add_user_message("Message " + std::to_string(i));
    }

    EXPECT_FALSE(context.get_system_message().empty());
    EXPECT_EQ(context.get_system_message(), "System prompt");
}

TEST(ConversationContextTest, ManualTruncation) {
    ConversationContext context(20, 0);

    for (int i = 0; i < 15; ++i) {
        context.add_user_message("Message");
    }

    EXPECT_EQ(context.get_message_count(), 15);

    context.set_max_messages(5);

    EXPECT_EQ(context.get_message_count(), 5);
}

TEST(ConversationContextTest, TokenLimitUpdate) {
    ConversationContext context(0, 1000);

    for (int i = 0; i < 10; ++i) {
        context.add_user_message("Message", 50);
    }

    int initial_count = context.get_message_count();

    context.set_max_tokens(100);

    EXPECT_LT(context.get_message_count(), initial_count);
    EXPECT_LE(context.get_total_tokens(), 100);
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST(ConversationContextTest, ClearMessages) {
    ConversationContext context;

    context.set_system_message("System");
    context.add_user_message("User");
    context.add_assistant_message("Assistant");

    context.clear();

    EXPECT_TRUE(context.is_empty());
    EXPECT_EQ(context.get_message_count(), 0);
    EXPECT_FALSE(context.get_system_message().empty());  // System kept
}

TEST(ConversationContextTest, ClearAll) {
    ConversationContext context;

    context.set_system_message("System");
    context.add_user_message("User");

    context.clear_all();

    EXPECT_TRUE(context.is_empty());
    EXPECT_EQ(context.get_message_count(), 0);
    EXPECT_TRUE(context.get_system_message().empty());  // System removed
}

TEST(ConversationContextTest, ClearEmptyContext) {
    ConversationContext context;

    EXPECT_NO_THROW({
        context.clear();
        context.clear_all();
    });
}

// ============================================================================
// Persistence Tests
// ============================================================================

TEST(ConversationContextTest, SaveToFile) {
    ConversationContext context(20, 2048);

    context.set_system_message("System prompt");
    context.add_user_message("Hello");
    context.add_assistant_message("Hi");

    std::string filepath = "test_conversation.txt";

    EXPECT_NO_THROW({ context.save_to_file(filepath); });

    // Verify file exists
    std::ifstream file(filepath);
    EXPECT_TRUE(file.good());
    file.close();

    // Cleanup
    std::remove(filepath.c_str());
}

TEST(ConversationContextTest, LoadFromFile) {
    ConversationContext context1(20, 2048);

    context1.set_system_message("System");
    context1.add_user_message("User message");
    context1.add_assistant_message("Assistant message");

    std::string filepath = "test_load.txt";
    context1.save_to_file(filepath);

    ConversationContext context2;

    EXPECT_NO_THROW({ context2.load_from_file(filepath); });

    EXPECT_EQ(context2.get_message_count(), context1.get_message_count());
    EXPECT_EQ(context2.get_system_message(), context1.get_system_message());

    // Cleanup
    std::remove(filepath.c_str());
}

TEST(ConversationContextTest, SaveLoadRoundTrip) {
    ConversationContext context1(15, 1500, true);

    context1.set_system_message("You are helpful");
    context1.add_user_message("Question 1");
    context1.add_assistant_message("Answer 1");
    context1.add_user_message("Question 2");
    context1.add_assistant_message("Answer 2");

    std::string filepath = "test_roundtrip.txt";
    context1.save_to_file(filepath);

    ConversationContext context2;
    context2.load_from_file(filepath);

    EXPECT_EQ(context2.get_message_count(), 4);
    EXPECT_EQ(context2.get_system_message(), "You are helpful");
    EXPECT_EQ(context2.get_last_user_message(), "Question 2");
    EXPECT_EQ(context2.get_last_assistant_message(), "Answer 2");

    auto messages = context2.get_messages();
    EXPECT_EQ(messages[0].content, "Question 1");
    EXPECT_EQ(messages[1].content, "Answer 1");

    // Cleanup
    std::remove(filepath.c_str());
}

TEST(ConversationContextTest, LoadFromNonexistentFile) {
    ConversationContext context;

    EXPECT_THROW({ context.load_from_file("nonexistent_file.txt"); }, std::runtime_error);
}

// ============================================================================
// Summarization Tests
// ============================================================================

TEST(ConversationContextTest, CreateSummarized) {
    ConversationContext context;

    for (int i = 0; i < 10; ++i) {
        context.add_user_message("User " + std::to_string(i));
        context.add_assistant_message("Assistant " + std::to_string(i));
    }

    ConversationContext summarized = context.create_summarized(4, "Earlier messages");

    EXPECT_LE(summarized.get_message_count(), 5);  // 4 recent + summary
    EXPECT_LT(summarized.get_message_count(), context.get_message_count());
}

TEST(ConversationContextTest, SummarizedKeepsRecentMessages) {
    ConversationContext context;

    context.add_user_message("Old message 1");
    context.add_assistant_message("Old response 1");
    context.add_user_message("Recent message");
    context.add_assistant_message("Recent response");

    ConversationContext summarized = context.create_summarized(2, "Summary");

    auto messages = summarized.get_messages();

    // Should have summary + 2 recent messages
    bool has_recent = false;
    for (const auto& msg : messages) {
        if (msg.content == "Recent message") {
            has_recent = true;
        }
    }
    EXPECT_TRUE(has_recent);
}

TEST(ConversationContextTest, SummarizedWithSystemMessage) {
    ConversationContext context;

    context.set_system_message("System prompt");

    for (int i = 0; i < 10; ++i) {
        context.add_user_message("Message " + std::to_string(i));
    }

    ConversationContext summarized = context.create_summarized(3);

    EXPECT_EQ(summarized.get_system_message(), "System prompt");
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(ConversationContextTest, EmptyMessage) {
    ConversationContext context;

    EXPECT_NO_THROW({ context.add_user_message(""); });

    EXPECT_EQ(context.get_message_count(), 1);
}

TEST(ConversationContextTest, VeryLongMessage) {
    ConversationContext context(0, 0);  // Unlimited to prevent truncation

    std::string long_message(10000, 'a');

    EXPECT_NO_THROW({ context.add_user_message(long_message); });

    EXPECT_EQ(context.get_message_count(), 1);
    EXPECT_GT(context.get_total_tokens(), 0);
}

TEST(ConversationContextTest, SpecialCharacters) {
    ConversationContext context;

    context.add_user_message("Hello\nWorld\tTest!");
    context.add_assistant_message("Response with símbolos and 中文");

    std::string formatted = context.format_for_model(false);

    EXPECT_NE(formatted.find("Hello"), std::string::npos);
    EXPECT_NE(formatted.find("símbolos"), std::string::npos);
}

TEST(ConversationContextTest, MultipleConsecutiveUserMessages) {
    ConversationContext context;

    context.add_user_message("User 1");
    context.add_user_message("User 2");
    context.add_user_message("User 3");

    EXPECT_EQ(context.get_message_count(), 3);
    EXPECT_EQ(context.get_last_user_message(), "User 3");
}

TEST(ConversationContextTest, MultipleConsecutiveAssistantMessages) {
    ConversationContext context;

    context.add_assistant_message("Assistant 1");
    context.add_assistant_message("Assistant 2");

    EXPECT_EQ(context.get_message_count(), 2);
    EXPECT_EQ(context.get_last_assistant_message(), "Assistant 2");
}

TEST(ConversationContextTest, CustomRoles) {
    ConversationContext context;

    context.add_message("function", "execute_search()");
    context.add_message("function_result", "Found 10 results");
    context.add_message("custom_role", "Custom content");

    EXPECT_EQ(context.get_message_count(), 3);

    auto messages = context.get_messages();
    EXPECT_EQ(messages[0].role, "function");
    EXPECT_EQ(messages[2].role, "custom_role");
}

// ============================================================================
// Token Estimation Tests
// ============================================================================

TEST(ConversationContextTest, TokenEstimationNonEmpty) {
    ConversationContext context;

    context.add_user_message("Hello world");

    EXPECT_GT(context.get_total_tokens(), 0);
}

TEST(ConversationContextTest, TokenEstimationIncreases) {
    ConversationContext context;

    context.add_user_message("Short");
    int tokens_short = context.get_total_tokens();

    context.add_user_message("This is a much longer message with many more words");
    int tokens_long = context.get_total_tokens();

    EXPECT_GT(tokens_long, tokens_short);
}

TEST(ConversationContextTest, ManualTokenCountOverridesEstimation) {
    ConversationContext context;

    context.add_user_message("Hello world", 100);

    EXPECT_EQ(context.get_total_tokens(), 100);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(ConversationContextTest, MultiTurnConversation) {
    ConversationContext context(10, 500);

    context.set_system_message("You are a helpful chatbot.");

    // Simulate conversation
    context.add_user_message("Hi, my name is Alice.");
    context.add_assistant_message("Hello Alice! Nice to meet you.");
    context.add_user_message("What's my name?");
    context.add_assistant_message("Your name is Alice.");
    context.add_user_message("That's correct!");
    context.add_assistant_message("I'm glad I remembered!");

    EXPECT_EQ(context.get_message_count(), 6);

    std::string formatted = context.format_for_model(true);
    EXPECT_NE(formatted.find("Alice"), std::string::npos);
}

TEST(ConversationContextTest, ConversationWithTruncationAndSave) {
    ConversationContext context(5, 0);

    // Add many messages
    for (int i = 0; i < 10; ++i) {
        context.add_user_message("Message " + std::to_string(i));
    }

    EXPECT_EQ(context.get_message_count(), 5);

    // Save truncated conversation
    std::string filepath = "test_truncated.txt";
    context.save_to_file(filepath);

    // Load and verify
    ConversationContext loaded;
    loaded.load_from_file(filepath);

    EXPECT_EQ(loaded.get_message_count(), 5);

    // Cleanup
    std::remove(filepath.c_str());
}

TEST(ConversationContextTest, DynamicLimitAdjustment) {
    ConversationContext context(20, 1000);

    // Add messages
    for (int i = 0; i < 15; ++i) {
        context.add_user_message("Message", 30);
    }

    int count_before = context.get_message_count();

    // Reduce both limits
    context.set_max_messages(5);
    context.set_max_tokens(100);

    EXPECT_LT(context.get_message_count(), count_before);
    EXPECT_LE(context.get_message_count(), 5);
    EXPECT_LE(context.get_total_tokens(), 100);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(ConversationContextTest, LargeConversationPerformance) {
    ConversationContext context(0, 0);  // Unlimited

    // Add many messages
    for (int i = 0; i < 100; ++i) {
        context.add_user_message("User message " + std::to_string(i));
        context.add_assistant_message("Assistant response " + std::to_string(i));
    }

    EXPECT_EQ(context.get_message_count(), 200);

    // Format should complete in reasonable time
    std::string formatted = context.format_for_model(true);
    EXPECT_GT(formatted.length(), 1000);
}

TEST(ConversationContextTest, RepeatedTruncation) {
    ConversationContext context(10, 200);

    // Add messages that repeatedly trigger truncation
    for (int i = 0; i < 50; ++i) {
        context.add_user_message("Message", 10);
    }

    EXPECT_LE(context.get_message_count(), 10);
    EXPECT_LE(context.get_total_tokens(), 200);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
