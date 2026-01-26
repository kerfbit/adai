#include "../src/ChatbotCLI.hpp"
#include <../gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Test Fixtures
// ============================================================================

class ChatbotCLITest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Create minimal test vocab file
        vocab_file = "test_vocab.txt";
        model_file = "test_model.bin";
        conv_file = "test_conversation.txt";

        std::ofstream vocab(vocab_file);
        vocab << "hello 100\n";
        vocab << "world 50\n";
        vocab << "test 25\n";
        vocab.close();

        // Note: Model file doesn't need to exist for testing settings/accessors
    }

    void TearDown() override {
        std::remove(vocab_file.c_str());
        std::remove(model_file.c_str());
        std::remove(conv_file.c_str());
    }

    std::string vocab_file;
    std::string model_file;
    std::string conv_file;
};

// ============================================================================
// Constructor and Initialization Tests
// ============================================================================

TEST_F(ChatbotCLITest, ConstructorSetsDefaultParameters) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    EXPECT_EQ(cli.get_max_response_length(), 100);
    EXPECT_FLOAT_EQ(cli.get_temperature(), 1.0f);
    EXPECT_FLOAT_EQ(cli.get_top_p(), 0.9f);
    EXPECT_EQ(cli.get_top_k(), 50);
    EXPECT_EQ(cli.get_beam_width(), 5);
    EXPECT_EQ(cli.get_generation_strategy(), "nucleus");
}

TEST_F(ChatbotCLITest, ConstructorStoresFilePaths) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    EXPECT_EQ(cli.get_vocab_path(), vocab_file);
    EXPECT_EQ(cli.get_model_path(), model_file);
    EXPECT_EQ(cli.get_conversation_save_path(), conv_file);
}

TEST_F(ChatbotCLITest, DefaultConversationSavePath) {
    ChatbotCLI cli(vocab_file, model_file);

    EXPECT_EQ(cli.get_conversation_save_path(), "conversation_history.txt");
}

// ============================================================================
// Parameter Setter/Getter Tests
// ============================================================================

TEST_F(ChatbotCLITest, SetAndGetGenerationStrategy) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    cli.set_generation_strategy("greedy");
    EXPECT_EQ(cli.get_generation_strategy(), "greedy");

    cli.set_generation_strategy("beam");
    EXPECT_EQ(cli.get_generation_strategy(), "beam");

    cli.set_generation_strategy("sampling");
    EXPECT_EQ(cli.get_generation_strategy(), "sampling");
}

TEST_F(ChatbotCLITest, SetAndGetMaxResponseLength) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    cli.set_max_response_length(50);
    EXPECT_EQ(cli.get_max_response_length(), 50);

    cli.set_max_response_length(200);
    EXPECT_EQ(cli.get_max_response_length(), 200);
}

TEST_F(ChatbotCLITest, SetAndGetTemperature) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    cli.set_temperature(0.5f);
    EXPECT_FLOAT_EQ(cli.get_temperature(), 0.5f);

    cli.set_temperature(1.5f);
    EXPECT_FLOAT_EQ(cli.get_temperature(), 1.5f);
}

TEST_F(ChatbotCLITest, SetAndGetTopP) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    cli.set_top_p(0.8f);
    EXPECT_FLOAT_EQ(cli.get_top_p(), 0.8f);

    cli.set_top_p(0.95f);
    EXPECT_FLOAT_EQ(cli.get_top_p(), 0.95f);
}

TEST_F(ChatbotCLITest, SetAndGetTopK) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    cli.set_top_k(10);
    EXPECT_EQ(cli.get_top_k(), 10);

    cli.set_top_k(100);
    EXPECT_EQ(cli.get_top_k(), 100);
}

TEST_F(ChatbotCLITest, SetAndGetBeamWidth) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    cli.set_beam_width(3);
    EXPECT_EQ(cli.get_beam_width(), 3);

    cli.set_beam_width(10);
    EXPECT_EQ(cli.get_beam_width(), 10);
}

// ============================================================================
// Command Handling Tests (via handle_setting)
// ============================================================================

TEST_F(ChatbotCLITest, HandleSettingStrategy) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    // Redirect cout to suppress output
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("strategy greedy");
    EXPECT_EQ(cli.get_generation_strategy(), "greedy");

    cli.handle_setting("strategy beam");
    EXPECT_EQ(cli.get_generation_strategy(), "beam");

    cli.handle_setting("strategy nucleus");
    EXPECT_EQ(cli.get_generation_strategy(), "nucleus");

    std::cout.rdbuf(old);  // Restore cout
}

TEST_F(ChatbotCLITest, HandleSettingMaxLength) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("length 150");
    EXPECT_EQ(cli.get_max_response_length(), 150);

    cli.handle_setting("max_length 75");
    EXPECT_EQ(cli.get_max_response_length(), 75);

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingTemperature) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("temperature 0.7");
    EXPECT_FLOAT_EQ(cli.get_temperature(), 0.7f);

    cli.handle_setting("temp 1.2");
    EXPECT_FLOAT_EQ(cli.get_temperature(), 1.2f);

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingTopP) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("top_p 0.85");
    EXPECT_FLOAT_EQ(cli.get_top_p(), 0.85f);

    cli.handle_setting("top-p 0.92");
    EXPECT_FLOAT_EQ(cli.get_top_p(), 0.92f);

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingTopK) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("top_k 20");
    EXPECT_EQ(cli.get_top_k(), 20);

    cli.handle_setting("top-k 40");
    EXPECT_EQ(cli.get_top_k(), 40);

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingBeamWidth) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("beam_width 7");
    EXPECT_EQ(cli.get_beam_width(), 7);

    cli.handle_setting("beam-width 12");
    EXPECT_EQ(cli.get_beam_width(), 12);

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingInvalidStrategy) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    std::string original = cli.get_generation_strategy();
    cli.handle_setting("strategy invalid");
    EXPECT_EQ(cli.get_generation_strategy(), original);  // Should not change

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingMissingValue) {
    ChatbotCLI cli(vocab_file, model_file, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    int original_length = cli.get_max_response_length();
    cli.handle_setting("length");  // Missing value
    EXPECT_EQ(cli.get_max_response_length(), original_length);  // Should not change

    std::cout.rdbuf(old);
}

// ============================================================================
// Helper Function Tests (from original test suite)
// ============================================================================

bool is_valid_command(const std::string& input) {
    if (input.empty() || input[0] != '/')
        return false;

    std::vector<std::string> valid_commands = {"/help",     "/clear", "/save",   "/load", "/stats",
                                               "/settings", "/set",   "/system", "/exit", "/quit"};

    for (const auto& cmd : valid_commands) {
        if (input == cmd || input.substr(0, cmd.length()) == cmd) {
            return true;
        }
    }
    return false;
}

bool is_valid_strategy(const std::string& strategy) {
    const std::vector<std::string> VALID_STRATEGIES = {"greedy", "beam", "sampling", "top-k",
                                                        "nucleus"};
    for (const auto& valid : VALID_STRATEGIES) {
        if (strategy == valid)
            return true;
    }
    return false;
}

TEST(CommandValidationTest, RecognizeValidCommands) {
    EXPECT_TRUE(is_valid_command("/help"));
    EXPECT_TRUE(is_valid_command("/clear"));
    EXPECT_TRUE(is_valid_command("/save"));
    EXPECT_TRUE(is_valid_command("/load"));
    EXPECT_TRUE(is_valid_command("/stats"));
    EXPECT_TRUE(is_valid_command("/settings"));
    EXPECT_TRUE(is_valid_command("/exit"));
    EXPECT_TRUE(is_valid_command("/quit"));
}

TEST(CommandValidationTest, RecognizeCommandsWithParameters) {
    EXPECT_TRUE(is_valid_command("/set temp 0.8"));
    EXPECT_TRUE(is_valid_command("/system You are a helpful assistant"));
}

TEST(CommandValidationTest, RejectInvalidCommands) {
    EXPECT_FALSE(is_valid_command("/invalid"));
    EXPECT_FALSE(is_valid_command("help"));     // Missing slash
    EXPECT_FALSE(is_valid_command(""));         // Empty
    EXPECT_FALSE(is_valid_command("random"));   // Not a command
}

TEST(StrategyValidationTest, RecognizeValidStrategies) {
    EXPECT_TRUE(is_valid_strategy("greedy"));
    EXPECT_TRUE(is_valid_strategy("beam"));
    EXPECT_TRUE(is_valid_strategy("sampling"));
    EXPECT_TRUE(is_valid_strategy("top-k"));
    EXPECT_TRUE(is_valid_strategy("nucleus"));
}

TEST(StrategyValidationTest, RejectInvalidStrategies) {
    EXPECT_FALSE(is_valid_strategy("invalid"));
    EXPECT_FALSE(is_valid_strategy("random"));
    EXPECT_FALSE(is_valid_strategy(""));
    EXPECT_FALSE(is_valid_strategy("GREEDY"));  // Case sensitive
}

// ============================================================================
// Color Code Tests
// ============================================================================

TEST(ColorCodeTest, ColorCodesAreDefined) {
    EXPECT_STREQ(COLOR_RESET, "\033[0m");
    EXPECT_STREQ(COLOR_USER, "\033[1;36m");
    EXPECT_STREQ(COLOR_BOT, "\033[1;32m");
    EXPECT_STREQ(COLOR_SYSTEM, "\033[1;33m");
    EXPECT_STREQ(COLOR_ERROR, "\033[1;31m");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
