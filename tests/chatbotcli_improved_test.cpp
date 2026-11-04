#include <../gtest/gtest.h>
#include <httplib.h>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
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
// TD-254: /set settings are sent, and only those
// ============================================================================

namespace {
// Runs handle_setting() with std::cout captured; returns what it printed.
std::string set_quietly(ChatbotCLI& cli, const std::string& setting) {
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    cli.handle_setting(setting);
    std::cout.rdbuf(old);
    return buffer.str();
}
}  // namespace

// Nothing changed with /set means nothing but the message is sent, so the server's own defaults
// apply. The CLI used to send all six settings with its own built-in values on every message.
TEST_F(ChatbotCLITest, UnchangedSettingsAreNotSent) {
    ChatbotCLI cli(server_url, conv_file);
    EXPECT_EQ(cli.chat_turn_body("hi"), R"({"message":"hi"})");
}

TEST_F(ChatbotCLITest, ChangedSettingsAreSent) {
    ChatbotCLI cli(server_url, conv_file);
    set_quietly(cli, "temp 0.25");
    set_quietly(cli, "length 12");
    const std::string body = cli.chat_turn_body("hi");
    EXPECT_NE(body.find(R"("temperature":0.25)"), std::string::npos) << body;
    EXPECT_NE(body.find(R"("max_length":12)"), std::string::npos) << body;
    EXPECT_EQ(body.find("top_p"), std::string::npos) << body;
    EXPECT_EQ(body.find("strategy"), std::string::npos) << body;

    // The CLI's old strategy names are sent as the server's names.
    set_quietly(cli, "strategy sampling");
    EXPECT_EQ(cli.get_generation_strategy(), "temperature");
    set_quietly(cli, "strategy top-k");
    EXPECT_NE(cli.chat_turn_body("hi").find(R"("strategy":"top_k")"), std::string::npos);
}

TEST_F(ChatbotCLITest, SetDefaultStopsSendingASetting) {
    ChatbotCLI cli(server_url, conv_file);
    set_quietly(cli, "top_k 20");
    ASSERT_TRUE(cli.is_overridden("top_k"));
    const std::string out = set_quietly(cli, "top-k default");
    EXPECT_FALSE(cli.is_overridden("top_k"));
    EXPECT_NE(out.find("server's default"), std::string::npos) << out;
    EXPECT_EQ(cli.chat_turn_body("hi"), R"({"message":"hi"})");
}

// TD-259: a /set value must be the whole number and within the server's range; it used to take
// "10abc" as 10 and accept a negative length or top_p 5 (which the server would now refuse).
TEST_F(ChatbotCLITest, SetRejectsPartialAndOutOfRangeValues) {
    ChatbotCLI cli(server_url, conv_file);
    for (const char* bad : {"length 10abc", "length 0", "length -5", "length 5000", "temp 2.5", "temp -1",
                            "temp nan", "top_p 5", "top_p 0.5x", "top_k 0", "top_k 1001", "beam_width 17",
                            "beam_width 3.5", "strategy fast"}) {
        const std::string out = set_quietly(cli, bad);
        EXPECT_NE(out.find("Invalid"), std::string::npos) << bad << ": " << out;
    }
    for (const char* key : {"max_length", "temperature", "top_p", "top_k", "beam_width", "strategy"}) {
        EXPECT_FALSE(cli.is_overridden(key)) << key;
    }
    EXPECT_EQ(cli.chat_turn_body("hi"), R"({"message":"hi"})");

    // The limits themselves are accepted.
    for (const char* ok : {"length 4096", "temp 0", "temp 2", "top_p 1", "top_k 1000", "beam_width 16"}) {
        const std::string out = set_quietly(cli, ok);
        EXPECT_EQ(out.find("Invalid"), std::string::npos) << ok << ": " << out;
    }
}

TEST_F(ChatbotCLITest, SettingsShowServerDefaultForUnchangedValues) {
    ChatbotCLI cli(server_url, conv_file);
    set_quietly(cli, "temp 0.5");
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    cli.handle_command("/settings");
    std::cout.rdbuf(old);
    EXPECT_NE(buffer.str().find("Temperature: 0.5"), std::string::npos) << buffer.str();
    EXPECT_NE(buffer.str().find("Top-k: server default"), std::string::npos) << buffer.str();
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

// TD-254: a /set reaches the server's generation. With the server's default of 40 tokens, /set length 1
// caps the reply at one token; /set temp 9 is refused by the CLI itself.
TEST_F(ChatbotCLISaveLoadRoundTripTest, SetLengthReachesTheServer) {
    ChatbotAPI::GenerationConfig config;
    config.max_length = 40;
    config.strategy = "greedy";
    api->set_generation_config(config);
    ChatbotCLI cli("http://127.0.0.1:" + std::to_string(port), conv_file);
    ASSERT_TRUE(cli.initialize());

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    cli.handle_setting("length 1");
    const std::string reply = cli.generate_response("hello world");
    std::cout.rdbuf(old);

    EXPECT_EQ(reply.rfind("Error:", 0), std::string::npos) << reply;
    const std::size_t reply_tokens = reply.empty() ? 0 : tokenizer->encode(reply, false).size();
    EXPECT_LE(reply_tokens, 1u) << reply;
}

// TD-232: session ids are server-issued, so once the server drops ours (expiry, eviction, restart)
// it answers 404. The CLI starts a new session and sends the message again instead of failing.
TEST_F(ChatbotCLISaveLoadRoundTripTest, LostSessionStartsANewOneAndTheTurnSucceeds) {
    ChatbotCLI cli("http://127.0.0.1:" + std::to_string(port), conv_file);
    ASSERT_TRUE(cli.initialize());

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    const std::string first = cli.generate_response("hello world");

    // The server forgets every session.
    api->set_session_expiry(std::chrono::steady_clock::duration::zero(), std::chrono::milliseconds(1));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    api->set_session_expiry(std::chrono::minutes(30), std::chrono::minutes(1));

    const std::string second = cli.generate_response("hello again");
    std::cout.rdbuf(old);

    EXPECT_EQ(first.rfind("Error:", 0), std::string::npos) << first;
    EXPECT_EQ(second.rfind("Error:", 0), std::string::npos) << second;
    EXPECT_NE(buffer.str().find("starting a new one"), std::string::npos) << buffer.str();
}

// TD-253: run() ends at end of input like /exit. The old loop ignored getline's failure: at EOF it
// printed "You: " forever, or, when the last line had no newline, sent that line to the server again
// on every pass (a failed getline leaves the string as it was). These tests feed std::cin from a
// string; a guard on std::cout stands in for the user typing /exit after 50 prompts, so the old code
// fails instead of hanging the test.
namespace {

class EndOfInputGuard : public std::stringbuf {
   public:
    explicit EndOfInputGuard(std::istringstream& in) : in_(in) {}
    bool tripped() const { return tripped_; }
    int prompts() const { return prompts_; }

   protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (std::string_view(s, static_cast<size_t>(n)) == "You: " && ++prompts_ == kMaxPrompts) {
            tripped_ = true;
            in_.clear();
            in_.str("/exit\n");
            std::cin.clear();
        }
        return prompts_ > kMaxPrompts ? n : std::stringbuf::xsputn(s, n);  // keep the output short
    }

   private:
    static constexpr int kMaxPrompts = 50;
    std::istringstream& in_;
    int prompts_ = 0;
    bool tripped_ = false;
};

struct RunResult {
    std::string output;
    bool guard_tripped = false;
    int prompts = 0;
};

RunResult run_with_input(ChatbotCLI& cli, const std::string& input) {
    std::istringstream in(input);
    EndOfInputGuard out(in);
    std::streambuf* old_in = std::cin.rdbuf(in.rdbuf());
    std::streambuf* old_out = std::cout.rdbuf(&out);
    cli.run();
    std::cout.rdbuf(old_out);
    std::cin.rdbuf(old_in);
    std::cin.clear();
    return {out.str(), out.tripped(), out.prompts()};
}

}  // namespace

TEST_F(ChatbotCLISaveLoadRoundTripTest, EmptyInputEndsTheSession) {
    ChatbotCLI cli("http://127.0.0.1:" + std::to_string(port), conv_file);
    const RunResult r = run_with_input(cli, "");
    EXPECT_FALSE(r.guard_tripped) << "run() kept prompting after end of input";
    EXPECT_EQ(r.prompts, 1);
    EXPECT_NE(r.output.find("Goodbye"), std::string::npos) << r.output.substr(0, 2000);
    EXPECT_EQ(r.output.find("Conversation saved to"), std::string::npos);  // no session, nothing to save
}

TEST_F(ChatbotCLISaveLoadRoundTripTest, EndOfInputSavesTheConversationLikeExit) {
    ChatbotCLI cli("http://127.0.0.1:" + std::to_string(port), conv_file);
    const RunResult r = run_with_input(cli, "hello world\n\n");  // a message, a blank line, then EOF
    EXPECT_FALSE(r.guard_tripped) << "run() kept prompting after end of input";
    EXPECT_NE(r.output.find("Conversation saved to"), std::string::npos) << r.output.substr(0, 2000);
    EXPECT_NE(r.output.find("Goodbye"), std::string::npos);
    ConversationContext saved;
    ASSERT_NO_THROW(saved.load_from_file(conv_file));
    EXPECT_EQ(saved.get_message_count(), 2);
}

// A last line without a newline is still sent before the session ends.
TEST_F(ChatbotCLISaveLoadRoundTripTest, LastLineWithoutNewlineIsStillSent) {
    ChatbotCLI cli("http://127.0.0.1:" + std::to_string(port), conv_file);
    const RunResult r = run_with_input(cli, "hello world");
    EXPECT_FALSE(r.guard_tripped) << "run() kept sending the last line";
    EXPECT_EQ(r.prompts, 2);
    EXPECT_NE(r.output.find("Bot:"), std::string::npos) << r.output.substr(0, 2000);
    EXPECT_NE(r.output.find("Conversation saved to"), std::string::npos);
}

// TD-255: a failed turn shows the server's error message, never an empty reply or the raw JSON. With
// HTTP 400/500 (TD-236) and with HTTP 200 + "success":false (older servers).
TEST_F(ChatbotCLISaveLoadRoundTripTest, RefusedSettingShowsTheServersError) {
    ChatbotCLI cli("http://127.0.0.1:" + std::to_string(port), conv_file);
    ASSERT_TRUE(cli.initialize());
    cli.set_temperature(9.0f);  // the setter skips /set's checks; the server refuses it

    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    const std::string reply = cli.generate_response("hello world");
    std::cout.rdbuf(old);

    EXPECT_TRUE(cli.did_last_turn_fail());
    EXPECT_EQ(reply.rfind("Error: 400 - ", 0), 0u) << reply;
    EXPECT_NE(reply.find("temperature"), std::string::npos) << reply;
    EXPECT_EQ(reply.find('{'), std::string::npos) << reply;
}

namespace {

/// A /chat/session that always answers with `status` and `body`, on a kernel-assigned port.
class StubChatServer {
   public:
    StubChatServer(int status, std::string body) {
        server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"status":"ok"})", "application/json");
        });
        server_.Post("/chat/session", [status, body](const httplib::Request&, httplib::Response& res) {
            res.status = status;
            res.set_content(body, "application/json");
        });
        port_ = server_.bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        server_.wait_until_ready();
    }
    ~StubChatServer() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }

   private:
    httplib::Server server_;
    std::thread thread_;
    int port_ = 0;
};

std::string reply_from(const StubChatServer& stub, bool& failed) {
    ChatbotCLI cli(stub.url(), "test_cli_stub_unused.txt");
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    const bool connected = cli.initialize();
    const std::string reply = connected ? cli.generate_response("hello") : "not connected";
    std::cout.rdbuf(old);
    failed = cli.did_last_turn_fail();
    return reply;
}

}  // namespace

TEST(ChatbotCLIErrorReplyTest, SuccessFalseWithHttp200ShowsTheError) {
    StubChatServer stub(200, R"({"success":false,"error":"model is busy","session_id":"abc"})");
    bool failed = false;
    const std::string reply = reply_from(stub, failed);
    EXPECT_TRUE(failed);
    EXPECT_EQ(reply, "Error: model is busy");
}

TEST(ChatbotCLIErrorReplyTest, SuccessFalseWithoutAReasonIsStillAnError) {
    StubChatServer stub(200, R"({"success":false})");
    bool failed = false;
    const std::string reply = reply_from(stub, failed);
    EXPECT_TRUE(failed);
    EXPECT_EQ(reply.rfind("Error:", 0), 0u) << reply;
    EXPECT_GT(reply.size(), std::string("Error: ").size());
}

TEST(ChatbotCLIErrorReplyTest, ErrorStatusShowsOnlyTheErrorField) {
    StubChatServer stub(400, R"({"success":false,"error":"\"temperature\" must be between 0 and 2"})");
    bool failed = false;
    const std::string reply = reply_from(stub, failed);
    EXPECT_TRUE(failed);
    EXPECT_EQ(reply, R"(Error: 400 - "temperature" must be between 0 and 2)");
}

TEST(ChatbotCLIErrorReplyTest, ServerErrorShowsItsRequestId) {
    StubChatServer stub(500, R"({"success":false,"error":"Internal server error","request_id":"r-42"})");
    bool failed = false;
    const std::string reply = reply_from(stub, failed);
    EXPECT_TRUE(failed);
    EXPECT_EQ(reply, "Error: 500 - Internal server error (request id r-42)");
}

TEST(ChatbotCLIErrorReplyTest, ErrorBodyThatIsNotJsonShowsTheStatusOnly) {
    StubChatServer stub(502, "<html>Bad Gateway</html>");
    bool failed = false;
    const std::string reply = reply_from(stub, failed);
    EXPECT_TRUE(failed);
    EXPECT_EQ(reply, "Error: 502");
}

TEST(ChatbotCLIErrorReplyTest, SuccessfulReplyIsNotAnError) {
    StubChatServer stub(200, R"({"success":true,"response":"hi there","session_id":"abc"})");
    bool failed = true;
    const std::string reply = reply_from(stub, failed);
    EXPECT_FALSE(failed);
    EXPECT_EQ(reply, "hi there");
}

TEST(ChatbotCLIErrorReplyTest, RunPrintsTheErrorInErrorColor) {
    StubChatServer stub(200, R"({"success":false,"error":"model is busy"})");
    ChatbotCLI cli(stub.url(), "test_cli_stub_unused.txt");
    const RunResult r = run_with_input(cli, "hello\n/exit\n");
    EXPECT_NE(r.output.find(std::string(COLOR_ERROR) + "Error: model is busy" + COLOR_RESET), std::string::npos)
        << r.output.substr(0, 2000);
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
