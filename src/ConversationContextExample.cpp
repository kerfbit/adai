#include <iostream>
#include <string>
#include "ConversationContext.hpp"

void print_separator(const std::string& title) {
    std::cout << "\n========== " << title << " ==========\n";
}

void example1_basic_conversation() {
    print_separator("Example 1: Basic Conversation");

    ConversationContext context(10, 500);  // Max 10 messages, 500 tokens

    // Set system message
    context.set_system_message("You are a helpful AI assistant.");

    // Add conversation turns
    context.add_user_message("Hello! What's the weather like?");
    context.add_assistant_message("I'm an AI and don't have access to real-time weather data.");

    context.add_user_message("Can you help me with math?");
    context.add_assistant_message("Of course! I'd be happy to help with math problems.");

    // Display formatted conversation
    std::cout << context.format_for_model(true) << std::endl;

    // Show statistics
    std::cout << context.get_statistics() << std::endl;
}

void example2_message_retrieval() {
    print_separator("Example 2: Message Retrieval");

    ConversationContext context;

    context.add_user_message("What is 2 + 2?");
    context.add_assistant_message("2 + 2 equals 4.");
    context.add_user_message("What about 5 + 3?");
    context.add_assistant_message("5 + 3 equals 8.");

    // Retrieve specific messages
    std::cout << "Last user message: " << context.get_last_user_message() << std::endl;
    std::cout << "Last assistant message: " << context.get_last_assistant_message() << std::endl;

    // Get all messages
    std::cout << "\nAll messages:\n";
    for (const auto& msg : context.get_messages()) {
        std::cout << "  [" << msg.role << "] " << msg.content << " (" << msg.token_count
                  << " tokens)\n";
    }
}

void example3_special_token_formatting() {
    print_separator("Example 3: Special Token Formatting");

    ConversationContext context;

    context.set_system_message("You are a creative writing assistant.");
    context.add_user_message("Write a short poem about coding.");
    context.add_assistant_message(
        "Lines of code dance on the screen,\nAlgorithms flow, precise and clean.");

    // Format with special tokens for encoder-decoder models
    std::string formatted = context.format_with_special_tokens("<bos>", "<eos>", "<sep>");

    std::cout << "Formatted with special tokens:\n" << formatted << std::endl;
}

void example4_automatic_truncation() {
    print_separator("Example 4: Automatic Truncation");

    // Create context with strict limits
    ConversationContext context(3, 100);  // Max 3 messages, 100 tokens

    std::cout << "Adding 5 messages to context with max_messages=3...\n\n";

    for (int i = 1; i <= 5; ++i) {
        context.add_user_message("User message " + std::to_string(i));
        context.add_assistant_message("Assistant response " + std::to_string(i));

        std::cout << "After adding message pair " << i << ":\n";
        std::cout << "  Message count: " << context.get_message_count() << "\n";
        std::cout << "  Total tokens: " << context.get_total_tokens() << "\n";
    }

    std::cout << "\nFinal conversation (oldest messages removed):\n";
    std::cout << context.format_for_model(false) << std::endl;
}

void example5_manual_limits() {
    print_separator("Example 5: Manual Limit Adjustment");

    ConversationContext context;

    // Add several messages
    for (int i = 1; i <= 10; ++i) {
        context.add_user_message("Message " + std::to_string(i));
    }

    std::cout << "Initial message count: " << context.get_message_count() << "\n";

    // Reduce limit
    std::cout << "Setting max_messages to 5...\n";
    context.set_max_messages(5);

    std::cout << "New message count: " << context.get_message_count() << "\n";
    std::cout << "\nRemaining messages:\n";
    for (const auto& msg : context.get_messages()) {
        std::cout << "  " << msg.content << "\n";
    }
}

void example6_persistence() {
    print_separator("Example 6: Save and Load Conversation");

    // Create and populate conversation
    ConversationContext context1(20, 1000);
    context1.set_system_message("You are a knowledgeable tutor.");
    context1.add_user_message("Explain recursion.");
    context1.add_assistant_message("Recursion is when a function calls itself.");
    context1.add_user_message("Can you give an example?");
    context1.add_assistant_message("Sure! Factorial is a classic example: n! = n * (n-1)!");

    std::cout << "Original conversation:\n";
    std::cout << context1.format_for_model(true) << std::endl;

    // Save to file
    std::string filepath = "conversation_save.txt";
    context1.save_to_file(filepath);
    std::cout << "Conversation saved to " << filepath << "\n\n";

    // Load into new context
    ConversationContext context2;
    context2.load_from_file(filepath);

    std::cout << "Loaded conversation:\n";
    std::cout << context2.format_for_model(true) << std::endl;

    std::cout << "Statistics match: "
              << (context1.get_message_count() == context2.get_message_count() ? "Yes" : "No")
              << std::endl;
}

void example7_conversation_summarization() {
    print_separator("Example 7: Conversation Summarization");

    ConversationContext context;

    // Simulate a long conversation
    context.add_user_message("What's machine learning?");
    context.add_assistant_message("Machine learning is a subset of AI...");
    context.add_user_message("What's supervised learning?");
    context.add_assistant_message("Supervised learning uses labeled data...");
    context.add_user_message("What about unsupervised?");
    context.add_assistant_message("Unsupervised learning finds patterns without labels...");
    context.add_user_message("Can you explain neural networks?");
    context.add_assistant_message("Neural networks are inspired by biological neurons...");
    context.add_user_message("What's backpropagation?");
    context.add_assistant_message(
        "Backpropagation is an algorithm for training neural networks...");

    std::cout << "Original conversation has " << context.get_message_count() << " messages\n\n";

    // Create summarized version keeping only last 2 message pairs
    ConversationContext summarized = context.create_summarized(
        4,  // Keep last 4 messages (2 pairs)
        "Earlier discussion covered ML basics, supervised/unsupervised learning");

    std::cout << "Summarized conversation:\n";
    std::cout << summarized.format_for_model(true) << std::endl;

    std::cout << "\nSummarized version has " << summarized.get_message_count() << " messages\n";
}

void example8_clear_operations() {
    print_separator("Example 8: Clear Operations");

    ConversationContext context;

    context.set_system_message("You are a helpful assistant.");
    context.add_user_message("Hello");
    context.add_assistant_message("Hi there!");
    context.add_user_message("How are you?");
    context.add_assistant_message("I'm doing well, thank you!");

    std::cout << "Initial state:\n";
    std::cout << "  Messages: " << context.get_message_count() << "\n";
    std::cout << "  Has system message: " << (!context.get_system_message().empty() ? "Yes" : "No")
              << "\n";

    // Clear messages but keep system
    context.clear();
    std::cout << "\nAfter clear():\n";
    std::cout << "  Messages: " << context.get_message_count() << "\n";
    std::cout << "  Has system message: " << (!context.get_system_message().empty() ? "Yes" : "No")
              << "\n";

    // Add more messages
    context.add_user_message("New conversation");
    context.add_assistant_message("Starting fresh!");

    std::cout << "\nAfter adding new messages:\n";
    std::cout << "  Messages: " << context.get_message_count() << "\n";
    std::cout << context.format_for_model(true) << std::endl;

    // Clear everything
    context.clear_all();
    std::cout << "After clear_all():\n";
    std::cout << "  Messages: " << context.get_message_count() << "\n";
    std::cout << "  Has system message: " << (!context.get_system_message().empty() ? "Yes" : "No")
              << "\n";
    std::cout << "  Is empty: " << (context.is_empty() ? "Yes" : "No") << "\n";
}

void example9_token_management() {
    print_separator("Example 9: Token Management");

    // Create context with token limit
    ConversationContext context(0, 200);  // Unlimited messages, 200 token limit

    std::cout << "Adding messages with token limit of 200...\n\n";

    // Manually specify token counts
    context.add_user_message("Short message", 5);
    context.add_assistant_message("Short response", 5);
    std::cout << "After short messages: " << context.get_total_tokens() << " tokens\n";

    // Add longer message (auto-estimated)
    context.add_user_message(
        "This is a much longer message that will consume more tokens in the context window.");
    std::cout << "After long user message: " << context.get_total_tokens() << " tokens\n";

    context.add_assistant_message(
        "Here's an even longer response that explains something in great detail with many words.");
    std::cout << "After long assistant message: " << context.get_total_tokens() << " tokens\n";

    // Add more to trigger truncation
    context.add_user_message("Another message that might cause truncation of earlier messages.");
    std::cout << "After triggering truncation: " << context.get_total_tokens() << " tokens\n";
    std::cout << "Messages remaining: " << context.get_message_count() << "\n";

    std::cout << "\nFinal conversation:\n";
    std::cout << context.format_for_model(false) << std::endl;
}

void example10_multi_turn_chatbot_simulation() {
    print_separator("Example 10: Multi-Turn Chatbot Simulation");

    ConversationContext context(10, 500, true);
    context.set_system_message("You are a friendly chatbot that remembers conversation context.");

    // Simulate a realistic conversation
    std::vector<std::pair<std::string, std::string>> conversation = {
        {"Hi, my name is Alice.", "Hello Alice! Nice to meet you."},
        {"What's your favorite color?", "I don't have preferences, but I'd love to know yours!"},
        {"I like blue. What can you remember about me?",
         "Your name is Alice, and you mentioned you like the color blue."},
        {"That's correct! You have a good memory.",
         "Thank you! I try to keep track of our conversation."},
        {"Can you help me learn about transformers?",
         "Of course, Alice! Transformers are a type of neural network architecture..."}};

    for (const auto& [user_msg, bot_msg] : conversation) {
        context.add_user_message(user_msg);
        context.add_assistant_message(bot_msg);

        std::cout << "User: " << user_msg << "\n";
        std::cout << "Bot:  " << bot_msg << "\n\n";
    }

    std::cout << context.get_statistics() << std::endl;

    std::cout << "\nFormatted for model input:\n";
    std::cout << context.format_for_model(true) << std::endl;
}

int main() {
    std::cout << "ConversationContext Examples\n";
    std::cout << "============================\n";

    example1_basic_conversation();
    example2_message_retrieval();
    example3_special_token_formatting();
    example4_automatic_truncation();
    example5_manual_limits();
    example6_persistence();
    example7_conversation_summarization();
    example8_clear_operations();
    example9_token_management();
    example10_multi_turn_chatbot_simulation();

    std::cout << "\n========== All Examples Complete ==========\n";

    return 0;
}
