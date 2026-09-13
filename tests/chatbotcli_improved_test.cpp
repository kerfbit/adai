#include <../gtest/gtest.h>
#include <httplib.h>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "../src/BPETokenizer.hpp"
#include "../src/ChatbotAPI.hpp"
#include "../src/ChatbotCLI.hpp"
#include "../src/ConversationContext.hpp"
#include "../src/EncoderDecoderModel.hpp"

// ============================================================================
// Test Fixtures
// ============================================================================

class ChatbotCLITest : public ::testing::Test {
   protected:
    void SetUp() override {
        server_url = "http://localhost:8080";
        conv_file = "test_conversation.txt";
    }

    void TearDown() override {
        std::remove(conv_file.c_str());
    }

    std::string server_url;
    std::string conv_file;
};

// ============================================================================
// Constructor and Initialization Tests
// ============================================================================

TEST_F(ChatbotCLITest, ConstructorSetsDefaultParameters) {
    ChatbotCLI cli(server_url, conv_file);

    EXPECT_EQ(cli.get_max_response_length(), 100);
    EXPECT_FLOAT_EQ(cli.get_temperature(), 1.0f);
    EXPECT_FLOAT_EQ(cli.get_top_p(), 0.9f);
    EXPECT_EQ(cli.get_top_k(), 50);
    EXPECT_EQ(cli.get_beam_width(), 5);
    EXPECT_EQ(cli.get_generation_strategy(), "nucleus");
}

TEST_F(ChatbotCLITest, ConstructorStoresServerUrlAndSavePath) {
    ChatbotCLI cli(server_url, conv_file);

    EXPECT_EQ(cli.get_server_url(), server_url);
    EXPECT_EQ(cli.get_conversation_save_path(), conv_file);
}

TEST_F(ChatbotCLITest, DefaultConversationSavePath) {
    ChatbotCLI cli(server_url);

    EXPECT_EQ(cli.get_conversation_save_path(), "conversation_history.txt");
}

// ============================================================================
// Parameter Setter/Getter Tests
// ============================================================================

TEST_F(ChatbotCLITest, SetAndGetGenerationStrategy) {
    ChatbotCLI cli(server_url, conv_file);

    cli.set_generation_strategy("greedy");
    EXPECT_EQ(cli.get_generation_strategy(), "greedy");

    cli.set_generation_strategy("beam");
    EXPECT_EQ(cli.get_generation_strategy(), "beam");

    cli.set_generation_strategy("sampling");
    EXPECT_EQ(cli.get_generation_strategy(), "sampling");
}

TEST_F(ChatbotCLITest, SetAndGetMaxResponseLength) {
    ChatbotCLI cli(server_url, conv_file);

    cli.set_max_response_length(50);
    EXPECT_EQ(cli.get_max_response_length(), 50);

    cli.set_max_response_length(200);
    EXPECT_EQ(cli.get_max_response_length(), 200);
}

TEST_F(ChatbotCLITest, SetAndGetTemperature) {
    ChatbotCLI cli(server_url, conv_file);

    cli.set_temperature(0.5f);
    EXPECT_FLOAT_EQ(cli.get_temperature(), 0.5f);

    cli.set_temperature(1.5f);
    EXPECT_FLOAT_EQ(cli.get_temperature(), 1.5f);
}

TEST_F(ChatbotCLITest, SetAndGetTopP) {
    ChatbotCLI cli(server_url, conv_file);

    cli.set_top_p(0.8f);
    EXPECT_FLOAT_EQ(cli.get_top_p(), 0.8f);

    cli.set_top_p(0.95f);
    EXPECT_FLOAT_EQ(cli.get_top_p(), 0.95f);
}

TEST_F(ChatbotCLITest, SetAndGetTopK) {
    ChatbotCLI cli(server_url, conv_file);

    cli.set_top_k(10);
    EXPECT_EQ(cli.get_top_k(), 10);

    cli.set_top_k(100);
    EXPECT_EQ(cli.get_top_k(), 100);
}

TEST_F(ChatbotCLITest, SetAndGetBeamWidth) {
    ChatbotCLI cli(server_url, conv_file);

    cli.set_beam_width(3);
    EXPECT_EQ(cli.get_beam_width(), 3);

    cli.set_beam_width(10);
    EXPECT_EQ(cli.get_beam_width(), 10);
}

// ============================================================================
// Command Handling Tests (via handle_setting)
// ============================================================================

TEST_F(ChatbotCLITest, HandleSettingStrategy) {
    ChatbotCLI cli(server_url, conv_file);

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
    ChatbotCLI cli(server_url, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("length 150");
    EXPECT_EQ(cli.get_max_response_length(), 150);

    cli.handle_setting("max_length 75");
    EXPECT_EQ(cli.get_max_response_length(), 75);

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingTemperature) {
    ChatbotCLI cli(server_url, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("temperature 0.7");
    EXPECT_FLOAT_EQ(cli.get_temperature(), 0.7f);

    cli.handle_setting("temp 1.2");
    EXPECT_FLOAT_EQ(cli.get_temperature(), 1.2f);

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingTopP) {
    ChatbotCLI cli(server_url, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("top_p 0.85");
    EXPECT_FLOAT_EQ(cli.get_top_p(), 0.85f);

    cli.handle_setting("top-p 0.92");
    EXPECT_FLOAT_EQ(cli.get_top_p(), 0.92f);

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingTopK) {
    ChatbotCLI cli(server_url, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("top_k 20");
    EXPECT_EQ(cli.get_top_k(), 20);

    cli.handle_setting("top-k 40");
    EXPECT_EQ(cli.get_top_k(), 40);

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingBeamWidth) {
    ChatbotCLI cli(server_url, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cli.handle_setting("beam_width 7");
    EXPECT_EQ(cli.get_beam_width(), 7);

    cli.handle_setting("beam-width 12");
    EXPECT_EQ(cli.get_beam_width(), 12);

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingInvalidStrategy) {
    ChatbotCLI cli(server_url, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    std::string original = cli.get_generation_strategy();
    cli.handle_setting("strategy invalid");
    EXPECT_EQ(cli.get_generation_strategy(), original);  // Should not change

    std::cout.rdbuf(old);
}

TEST_F(ChatbotCLITest, HandleSettingMissingValue) {
    ChatbotCLI cli(server_url, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    int original_length = cli.get_max_response_length();
    cli.handle_setting("length");                               // Missing value
    EXPECT_EQ(cli.get_max_response_length(), original_length);  // Should not change

    std::cout.rdbuf(old);
}

// TD-072 regression: a non-numeric value for a numeric parameter used to
// throw std::invalid_argument straight out of handle_setting() uncaught —
// in the real interactive run() loop this propagated to main()'s top-level
// catch and ended the whole session (losing session_id/conversation state)
// over a single mistyped command. handle_setting() should report the bad
// value and leave the parameter untouched instead of throwing.
TEST_F(ChatbotCLITest, HandleSettingNonNumericValueDoesNotThrow) {
    ChatbotCLI cli(server_url, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    int original_length = cli.get_max_response_length();
    float original_temp = cli.get_temperature();
    EXPECT_NO_THROW(cli.handle_setting("length abc"));
    EXPECT_NO_THROW(cli.handle_setting("temp not-a-number"));
    EXPECT_EQ(cli.get_max_response_length(), original_length);
    EXPECT_FLOAT_EQ(cli.get_temperature(), original_temp);

    std::cout.rdbuf(old);
}

// ============================================================================
// TD-053: /save and /load
//
// ChatbotCLI holds no local conversation state -- history lives server-side in ChatbotAPI's
// Session/ConversationContext -- so /save and /load are real network round-trips
// (POST /chat/session/export, POST /chat/session/import), not local file operations on
// anything this object owns directly. The three tests below exercise paths that need no server
// at all (no active session yet, no saved file, an empty saved file); the real round-trip
// against a live ChatbotAPI is further down, alongside the other real-server tests.
// ============================================================================

TEST_F(ChatbotCLITest, SaveWithNoActiveSessionPrintsErrorAndWritesNoFile) {
    ChatbotCLI cli(server_url, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    cli.handle_command("/save");
    std::cout.rdbuf(old);

    EXPECT_NE(buffer.str().find("Nothing to save"), std::string::npos) << buffer.str();
    std::ifstream file(conv_file);
    EXPECT_FALSE(file.good());
}

TEST_F(ChatbotCLITest, LoadWithNoSavedFilePrintsError) {
    ChatbotCLI cli(server_url, conv_file);

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    cli.handle_command("/load");
    std::cout.rdbuf(old);

    EXPECT_NE(buffer.str().find("No saved conversation found"), std::string::npos) << buffer.str();
}

TEST_F(ChatbotCLITest, LoadWithEmptySavedFilePrintsError) {
    ChatbotCLI cli(server_url, conv_file);
    { std::ofstream(conv_file).close(); }  // create an empty file

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    cli.handle_command("/load");
    std::cout.rdbuf(old);

    EXPECT_NE(buffer.str().find("is empty"), std::string::npos) << buffer.str();
}

// ============================================================================
// TD-053: /save and /load, real round-trip against a live ChatbotAPI
//
// Real ChatbotAPI bound to 127.0.0.1 on a background thread, driven by a real ChatbotCLI over
// a real httplib::Client -- same style as trainer_admin_api_test.cpp's tests, not a mocked HTTP
// layer, since this is exactly the network round-trip TD-053 added.
// ============================================================================

namespace {

constexpr int kChatbotApiTestBasePort = 43900;
constexpr int kChatbotApiTestPortSpan = 100;

int pick_chatbot_api_test_port() {
    return kChatbotApiTestBasePort +
           ((static_cast<int>(::getpid()) + static_cast<int>(reinterpret_cast<uintptr_t>(
                                                 &kChatbotApiTestBasePort))) %
            kChatbotApiTestPortSpan);
}

/// Starts `api` on a background thread and polls GET /health until it responds or 3s elapse.
bool start_chatbot_api(ChatbotAPI& api, int port, std::thread& thread) {
    thread = std::thread([&api] { api.start(); });

    httplib::Client probe("127.0.0.1", port);
    probe.set_connection_timeout(std::chrono::milliseconds(200));
    probe.set_read_timeout(std::chrono::milliseconds(200));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto res = probe.Get("/health"); res && res->status == 200) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}

}  // namespace

class ChatbotCLISaveLoadRoundTripTest : public ::testing::Test {
   protected:
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<EncoderDecoderModel> model;
    std::unique_ptr<ChatbotAPI> api;
    std::thread server_thread;
    int port = 0;
    std::string conv_file;

    void SetUp() override {
        port = pick_chatbot_api_test_port();
        conv_file = "test_cli_save_load_roundtrip_" + std::to_string(port) + ".txt";

        std::vector<std::string> test_texts = {"hello world test", "hi there how are you"};
        tokenizer = std::make_unique<BPETokenizer>();
        tokenizer->build_vocab(test_texts, 50);

        model = std::make_unique<EncoderDecoderModel>(tokenizer->get_vocab_size(), 32, 1, 1, 2, 64,
                                                      128);
        api = std::make_unique<ChatbotAPI>(model.get(), tokenizer.get(), port, 30);

        ASSERT_TRUE(start_chatbot_api(*api, port, server_thread))
            << "ChatbotAPI never became reachable on port " << port;
    }

    void TearDown() override {
        api->stop();
        if (server_thread.joinable()) {
            server_thread.join();
        }
        std::remove(conv_file.c_str());
    }
};

TEST_F(ChatbotCLISaveLoadRoundTripTest, SaveThenLoadRestoresSessionOnAFreshClient) {
    std::string server_url = "http://127.0.0.1:" + std::to_string(port);
    // Deliberately contains a literal '"' -- ConversationContext::serialize()'s own escaping
    // (escape_for_line()) only handles '\\'/'\n'/'\r', not '"' (no need to, for its own
    // pipe-delimited-lines format), so this quote survives into the exported "data" field
    // completely unescaped from ConversationContext's point of view. It's on
    // handle_export_session()/parse_json_value() to escape/unescape it correctly at the JSON
    // layer instead -- exactly the step a naive "just embed the string" implementation would
    // skip, so this is the case that actually exercises that step (confirmed via
    // revert-confirm-fail: a plain "hello world" message does not, since it contains nothing
    // that needs escaping at either layer).
    const std::string user_message = R"(say "hi" please)";

    // First client: establish a session by sending one message, then /save it.
    {
        ChatbotCLI cli(server_url, conv_file);
        ASSERT_TRUE(cli.initialize());

        std::stringstream buffer;
        std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
        std::string response = cli.generate_response(user_message);
        cli.handle_command("/save");
        std::cout.rdbuf(old);

        // Not asserting on response content -- this tiny randomly-initialized model's actual
        // generated text isn't the point here, only that a real session now exists to save.
        EXPECT_NE(buffer.str().find("Conversation saved to"), std::string::npos) << buffer.str();
    }

    // The saved file must be a real, complete, correctly round-tripped export -- parse it with
    // ConversationContext's own (separately tested) load_from_file() and confirm the exact
    // original message, quote included, survived the full server-export -> HTTP JSON ->
    // local-file trip.
    ConversationContext loaded_directly;
    ASSERT_NO_THROW(loaded_directly.load_from_file(conv_file));
    ASSERT_EQ(loaded_directly.get_message_count(), 2);
    EXPECT_EQ(loaded_directly.get_messages()[0].content, user_message);

    // Second client: a fresh instance (simulating a new process, no session_id yet) loads the
    // same file and must adopt a real session with the restored message count.
    {
        ChatbotCLI cli(server_url, conv_file);
        ASSERT_TRUE(cli.initialize());

        std::stringstream buffer;
        std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
        cli.handle_command("/load");
        std::cout.rdbuf(old);

        EXPECT_NE(buffer.str().find("Conversation loaded from"), std::string::npos) << buffer.str();
        // Two messages: the user message plus the assistant's reply, both recorded by
        // handle_chat_session() server-side during the first client's generate_response() call.
        EXPECT_NE(buffer.str().find("(2 messages)"), std::string::npos) << buffer.str();
    }
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
    EXPECT_FALSE(is_valid_command("help"));    // Missing slash
    EXPECT_FALSE(is_valid_command(""));        // Empty
    EXPECT_FALSE(is_valid_command("random"));  // Not a command
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
