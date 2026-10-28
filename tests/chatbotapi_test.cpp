/**
 * ChatbotAPI Test Suite
 *
 * Tests for the ChatbotAPI REST endpoints including:
 * - Health check endpoint
 * - Single chat endpoint
 * - Session-based chat endpoint
 * - Batch chat endpoint
 * - Batch session chat endpoint
 * - JSON parsing utilities
 * - Session management
 * - Batch processing efficiency
 */

#include "../src/ChatbotAPI.hpp"
#include "nlohmann/json.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include "../src/BPETokenizer.hpp"
#include "../src/ConversationContext.hpp"
#include "../src/EncoderDecoderModel.hpp"
#include "../src/Optimizer.hpp"
#include "../src/RAGInference.hpp"
#include <condition_variable>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <spdlog/sinks/ostream_sink.h>
#include <sstream>
#include "cpp-httplib/httplib.h"
#include "../src/Logger.hpp"
#include <future>

class ChatbotAPITest : public ::testing::Test {
   protected:
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<EncoderDecoderModel> model;
    std::unique_ptr<ChatbotAPI> api;

    void SetUp() override {
        // Create a small test vocabulary
        std::vector<std::string> test_texts = {"hello world test", "hi there how are you",
                                               "what is the answer to this question",
                                               "test message for training"};

        // Initialize tokenizer
        tokenizer = std::make_unique<BPETokenizer>();
        tokenizer->build_vocab(test_texts, 50);  // Small vocab for testing

        // Create a small test model (minimal size for testing)
        int vocab_size = tokenizer->get_vocab_size();
        int d_model = 32;        // Small embedding dimension
        int encoder_layers = 1;  // Single layer for speed
        int decoder_layers = 1;
        int num_heads = 2;  // Minimal heads
        int d_ff = 64;      // Small feedforward
        int max_seq_length = 128;

        model = std::make_unique<EncoderDecoderModel>(
            vocab_size, d_model, encoder_layers, decoder_layers, num_heads, d_ff, max_seq_length);
        // TD-362: generation tokenizes with the model's own tokenizer (as chatbot_api_server's model
        // gets its checkpoint's .vocab); give it the same vocabulary as the API's.
        model->get_tokenizer()->build_vocab(test_texts, 50);

        // Create API (don't start server for unit tests)
        api = std::make_unique<ChatbotAPI>(model.get(), tokenizer.get(),
                                           8080,  // port
                                           30     // session timeout minutes
        );
    }

    void TearDown() override {
        api.reset();
        model.reset();
        tokenizer.reset();
    }

    // generate_response() is private; ChatbotAPI friends ChatbotAPITest
    // specifically, and that friendship does not extend to the subclasses
    // gtest's TEST_F macro generates, so TEST_F bodies call this wrapper
    // (an ordinary inherited member) instead of api->generate_response()
    // directly.
    std::string call_generate_response(const std::string& input,
                                       const ChatbotAPI::GenerationConfig& config) {
        return api->generate_response(input, config);
    }

    // handle_chat_session() is private; same friendship-doesn't-propagate
    // reason as call_generate_response() above.
    std::string call_handle_chat_session(const std::string& request_body) {
        return api->handle_chat_session(request_body);
    }
    std::string call_handle_batch_chat(const std::string& request_body) {
        return api->handle_batch_chat(request_body);
    }
    std::string call_handle_batch_chat_session(const std::string& request_body) {
        return api->handle_batch_chat_session(request_body);
    }

    // handle_export_session()/handle_import_session() are private (TD-053); same
    // friendship-doesn't-propagate reason as call_generate_response() above.
    std::string call_handle_export_session(const std::string& request_body) {
        return api->handle_export_session(request_body);
    }
    std::string call_handle_import_session(const std::string& request_body) {
        return api->handle_import_session(request_body);
    }

    // handle_profile() is private; same friendship-doesn't-propagate reason as
    // call_generate_response() above (TD-038).
    std::string call_handle_profile() {
        return api->handle_profile();
    }
    std::string call_handle_health() {
        return api->handle_health();
    }
    // TD-232: ids are server-issued; a first turn with no session_id creates one.
    std::string start_session(const std::string& message = "hello world") {
        const std::string reply = call_handle_chat_session(R"({"message":")" + message + "\"}");
        return api->parse_json_string(reply, "session_id");
    }
    std::size_t call_session_count() {
        std::lock_guard<std::mutex> lock(api->sessions_mutex_);
        return api->sessions_.size();
    }

    // batched_engine_ is private; same friendship-doesn't-propagate reason as
    // call_generate_response() above (TD-038). Used to prove generate_response() actually routed
    // a request through the engine's real queue/worker thread, not merely that it didn't throw —
    // the plain (non-batched) path can also succeed and throw nothing, so total_requests is the
    // discriminator that only the batched path can move.
    BatchedInferenceStats call_get_batched_stats() {
        return api->batched_engine_ ? api->batched_engine_->get_stats() : BatchedInferenceStats();
    }

};

// ============================================================================
// JSON Parsing Tests
// ============================================================================

TEST_F(ChatbotAPITest, ParseJsonString_BasicKey) {
    std::string json = R"({"message":"Hello world"})";
    std::string value = api->parse_json_string(json, "message");
    EXPECT_EQ(value, "Hello world");
}

TEST_F(ChatbotAPITest, ParseJsonString_EscapedQuotes) {
    std::string json = R"({"message":"He said \"hello\""})";
    std::string value = api->parse_json_string(json, "message");
    EXPECT_EQ(value, "He said \"hello\"");
}

TEST_F(ChatbotAPITest, ParseJsonString_Newlines) {
    std::string json = R"({"message":"Line 1\nLine 2"})";
    std::string value = api->parse_json_string(json, "message");
    EXPECT_TRUE(value.find('\n') != std::string::npos);
}

TEST_F(ChatbotAPITest, ParseJsonString_MissingKey) {
    std::string json = R"({"other":"value"})";
    std::string value = api->parse_json_string(json, "message");
    EXPECT_EQ(value, "");
}

TEST_F(ChatbotAPITest, ParseJsonString_EmptyValue) {
    std::string json = R"({"message":""})";
    std::string value = api->parse_json_string(json, "message");
    EXPECT_EQ(value, "");
}

TEST_F(ChatbotAPITest, ParseJsonArray_BasicArray) {
    std::string json = R"({"messages":["msg1","msg2","msg3"]})";
    std::vector<std::string> values = api->parse_json_array(json, "messages");
    ASSERT_EQ(values.size(), 3);
    EXPECT_EQ(values[0], "msg1");
    EXPECT_EQ(values[1], "msg2");
    EXPECT_EQ(values[2], "msg3");
}

TEST_F(ChatbotAPITest, ParseJsonArray_EmptyArray) {
    std::string json = R"({"messages":[]})";
    std::vector<std::string> values = api->parse_json_array(json, "messages");
    EXPECT_EQ(values.size(), 0);
}

TEST_F(ChatbotAPITest, ParseJsonArray_SingleElement) {
    std::string json = R"({"messages":["only_one"]})";
    std::vector<std::string> values = api->parse_json_array(json, "messages");
    ASSERT_EQ(values.size(), 1);
    EXPECT_EQ(values[0], "only_one");
}

TEST_F(ChatbotAPITest, ParseJsonArray_WithSpaces) {
    std::string json = R"({"messages": [ "msg1" , "msg2" , "msg3" ]})";
    std::vector<std::string> values = api->parse_json_array(json, "messages");
    ASSERT_EQ(values.size(), 3);
    EXPECT_EQ(values[0], "msg1");
}

TEST_F(ChatbotAPITest, ParseJsonArray_EscapedStrings) {
    std::string json = R"({"messages":["Line 1\nLine 2","Quote: \"test\""]})";
    std::vector<std::string> values = api->parse_json_array(json, "messages");
    ASSERT_EQ(values.size(), 2);
    EXPECT_TRUE(values[0].find('\n') != std::string::npos);
    EXPECT_TRUE(values[1].find('"') != std::string::npos);
}

// TD-226: the array parser took the first ']' anywhere for the end of the array, so a message
// containing ']' made the whole batch empty.
TEST_F(ChatbotAPITest, ParseJsonArray_BracketsCommasAndQuotesInsideStrings) {
    const std::string json = R"({"messages":["see [1] here","a, b] c","say \"hi\" [x]"]})";
    std::vector<std::string> values = api->parse_json_array(json, "messages");
    ASSERT_EQ(values.size(), 3u);
    EXPECT_EQ(values[0], "see [1] here");
    EXPECT_EQ(values[1], "a, b] c");
    EXPECT_EQ(values[2], "say \"hi\" [x]");
}

// TD-228: the string parser returned the next key's name for a non-string value, ran past a value
// ending in an escaped backslash, and left \uXXXX escapes literal.
TEST_F(ChatbotAPITest, ParseJsonString_TD228Inputs) {
    EXPECT_EQ(api->parse_json_string(R"({"session_id":123,"message":"hi"})", "session_id"), "");
    EXPECT_EQ(api->parse_json_string(R"({"message":"path C:\\","session_id":"abc"})", "message"),
              "path C:\\");
    EXPECT_EQ(api->parse_json_string(R"({"message":"caf\u00e9"})", "message"), "caf\xc3\xa9");
    EXPECT_EQ(api->parse_json_string(R"({"message":"hi"} trailing)", "message"), "")
        << "a body that isn't one JSON object";
}

TEST_F(ChatbotAPITest, HandlersRejectMalformedBodiesAndMistypedFields) {
    // The routes answer these with HTTP 400 (std::invalid_argument is the handlers' BadRequest).
    EXPECT_THROW(call_handle_batch_chat(R"({"messages":["a", )"), std::invalid_argument);
    EXPECT_THROW(call_handle_batch_chat(R"({"messages":"not an array"})"), std::invalid_argument);
    EXPECT_THROW(call_handle_batch_chat(R"({"messages":["ok", 7]})"), std::invalid_argument);
    EXPECT_THROW(call_handle_chat_session(R"({"session_id":123,"message":"hi"})"),
                 std::invalid_argument);
    EXPECT_THROW(call_handle_chat_session("not json"), std::invalid_argument);
}

TEST_F(ChatbotAPITest, BatchChatAcceptsMessagesWithBrackets) {
    ChatbotAPI::GenerationConfig config;
    config.max_length = 3;
    config.strategy = "greedy";
    api->set_generation_config(config);
    const std::string response = call_handle_batch_chat(R"({"messages":["see [1] here","second"]})");
    EXPECT_NE(response.find("\"success\":true"), std::string::npos) << response;
    EXPECT_EQ(response.find("Missing or empty"), std::string::npos) << response;
}

TEST_F(ChatbotAPITest, ParseJsonArray_MissingKey) {
    std::string json = R"({"other":["val1","val2"]})";
    std::vector<std::string> values = api->parse_json_array(json, "messages");
    EXPECT_EQ(values.size(), 0);
}

// ============================================================================
// JSON Response Creation Tests
// ============================================================================

TEST_F(ChatbotAPITest, CreateJsonResponse_Success) {
    std::string response = api->create_json_response("Test response", true);
    EXPECT_TRUE(response.find("\"success\":true") != std::string::npos);
    EXPECT_TRUE(response.find("Test response") != std::string::npos);
}

TEST_F(ChatbotAPITest, CreateJsonResponse_Error) {
    std::string response = api->create_json_response("", false, "Test error");
    EXPECT_TRUE(response.find("\"success\":false") != std::string::npos);
    EXPECT_TRUE(response.find("Test error") != std::string::npos);
}

TEST_F(ChatbotAPITest, CreateErrorResponse) {
    std::string response = api->create_error_response("Error message");
    EXPECT_TRUE(response.find("\"success\":false") != std::string::npos);
    EXPECT_TRUE(response.find("Error message") != std::string::npos);
}

TEST_F(ChatbotAPITest, CreateBatchJsonResponse_Success) {
    ChatbotAPI::BatchResponse batch_resp;
    batch_resp.success = true;
    batch_resp.responses = {"Response 1", "Response 2", "Response 3"};
    batch_resp.stats.total_tokens = 100;
    batch_resp.stats.actual_tokens = 80;
    batch_resp.stats.padding_ratio = 0.2f;
    batch_resp.stats.num_batches = 1;
    batch_resp.stats.avg_batch_size = 3.0f;

    std::string response = api->create_batch_json_response(batch_resp);

    EXPECT_TRUE(response.find("\"success\":true") != std::string::npos);
    EXPECT_TRUE(response.find("Response 1") != std::string::npos);
    EXPECT_TRUE(response.find("Response 2") != std::string::npos);
    EXPECT_TRUE(response.find("Response 3") != std::string::npos);
    EXPECT_TRUE(response.find("\"total_tokens\":100") != std::string::npos);
    EXPECT_TRUE(response.find("\"actual_tokens\":80") != std::string::npos);
}

TEST_F(ChatbotAPITest, CreateBatchJsonResponse_WithSessionIds) {
    ChatbotAPI::BatchResponse batch_resp;
    batch_resp.success = true;
    batch_resp.responses = {"Resp 1", "Resp 2"};
    batch_resp.session_ids = {"session_1", "session_2"};
    batch_resp.stats.total_tokens = 50;
    batch_resp.stats.actual_tokens = 45;
    batch_resp.stats.padding_ratio = 0.1f;
    batch_resp.stats.num_batches = 1;
    batch_resp.stats.avg_batch_size = 2.0f;

    std::string response = api->create_batch_json_response(batch_resp);

    EXPECT_TRUE(response.find("\"session_ids\"") != std::string::npos);
    EXPECT_TRUE(response.find("session_1") != std::string::npos);
    EXPECT_TRUE(response.find("session_2") != std::string::npos);
}

TEST_F(ChatbotAPITest, CreateBatchJsonResponse_Error) {
    ChatbotAPI::BatchResponse batch_resp;
    batch_resp.success = false;
    batch_resp.error = "Batch processing failed";

    std::string response = api->create_batch_json_response(batch_resp);

    EXPECT_TRUE(response.find("\"success\":false") != std::string::npos);
    EXPECT_TRUE(response.find("Batch processing failed") != std::string::npos);
}

// ============================================================================
// JSON Escaping Regression Tests (TD-063)
//
// create_json_response()'s error branch, create_batch_json_response()'s error
// and session_ids branches, and handle_chat_session()'s inline session_id
// field all used to write their string arguments straight into the response
// with no escaping — while an adjacent field in the very same function (the
// success "response"/"responses" text) was carefully escaped. Every top-level
// endpoint handler's catch block passes e.what() straight to
// create_error_response(), and session_id can be entirely client-supplied
// (get_or_create_session() uses a non-empty client-supplied session_id
// verbatim as the session's key), so a quote character reaching either path
// used to inject additional JSON fields or break the response's structure.
// handle_chat_session() is private and its TEST_F-generated subclass doesn't
// inherit ChatbotAPI's friendship with the ChatbotAPITest base fixture (C++
// friendship isn't transitive to subclasses), so it isn't called directly
// here — it's covered indirectly, since it now calls the same
// escape_json_string() exercised below.
// ============================================================================

TEST_F(ChatbotAPITest, EscapeJsonString_EscapesQuotesBackslashesAndControlChars) {
    EXPECT_EQ(ChatbotAPI::escape_json_string(R"(say "hi")"), R"(say \"hi\")");
    EXPECT_EQ(ChatbotAPI::escape_json_string("a\\b"), "a\\\\b");
    EXPECT_EQ(ChatbotAPI::escape_json_string("line1\nline2"), "line1\\nline2");
    EXPECT_EQ(ChatbotAPI::escape_json_string("plain text"), "plain text");
}

// TD-229: control characters other than \n, \r and \t passed through raw, and strict JSON parsers
// reject them; invalid UTF-8 (model output can split a multi-byte character) is rejected too.
TEST_F(ChatbotAPITest, EscapeJsonString_EveryControlCharacterAndInvalidUtf8ParseStrictly) {
    std::string input;
    for (int c = 0; c < 0x20; ++c)
        input += static_cast<char>(c);
    input += "\"\\ caf\xc3\xa9 \xe6\x97\xa5 \xf0\x9f\x98\x80";  // valid 2-, 3- and 4-byte characters
    const std::string valid_part = input;
    input += "\xff|\xe6\x97|\xc0\xaf|\xed\xa0\x80";  // stray byte, truncated, overlong, surrogate

    const std::string escaped = ChatbotAPI::escape_json_string(input);
    nlohmann::json parsed;
    ASSERT_NO_THROW(parsed = nlohmann::json::parse("\"" + escaped + "\"")) << escaped;
    const std::string back = parsed.get<std::string>();
    EXPECT_EQ(back.compare(0, valid_part.size(), valid_part), 0) << "valid text round-trips";
    const std::string replacement = "\xef\xbf\xbd";  // U+FFFD
    EXPECT_EQ(back.substr(valid_part.size()),
              replacement + "|" + replacement + replacement + "|" + replacement + replacement + "|" +
                  replacement + replacement + replacement);
}

TEST_F(ChatbotAPITest, ResponsesAreStrictJsonWhateverTheyCarry) {
    const std::string nasty = std::string("a\x01" "b\x08" "c\x1f\"\\\n") + "\xff\xfe";
    for (const std::string& body :
         {api->create_json_response(nasty), api->create_error_response(nasty)}) {
        EXPECT_NO_THROW((void)nlohmann::json::parse(body)) << body;
    }
    ChatbotAPI::BatchResponse batch;
    batch.success = true;
    batch.responses = {nasty, "plain"};
    batch.session_ids = {nasty, "id"};
    const std::string body = api->create_batch_json_response(batch);
    nlohmann::json parsed;
    ASSERT_NO_THROW(parsed = nlohmann::json::parse(body)) << body;
}

TEST_F(ChatbotAPITest, CreateErrorResponse_EscapesEmbeddedQuote) {
    // A crafted error message containing a quote used to break out of the
    // "error" string and inject a sibling field into the response.
    std::string response = api->create_error_response(R"(bad input: "oops","injected":true)");

    EXPECT_EQ(response.find(R"("injected":true)"), std::string::npos)
        << "unescaped quote let attacker-controlled JSON escape the error string: " << response;
    EXPECT_NE(response.find(R"(\")"), std::string::npos);
}

TEST_F(ChatbotAPITest, CreateBatchJsonResponse_EscapesEmbeddedQuoteInSessionId) {
    ChatbotAPI::BatchResponse batch_resp;
    batch_resp.success = true;
    batch_resp.responses = {"hi"};
    batch_resp.session_ids = {R"(evil","admin":true)"};
    batch_resp.stats.total_tokens = 1;
    batch_resp.stats.actual_tokens = 1;
    batch_resp.stats.padding_ratio = 0.0f;
    batch_resp.stats.num_batches = 1;
    batch_resp.stats.avg_batch_size = 1.0f;

    std::string response = api->create_batch_json_response(batch_resp);

    EXPECT_EQ(response.find(R"("admin":true)"), std::string::npos)
        << "unescaped quote in session_ids[] injected a sibling field: " << response;
}

TEST_F(ChatbotAPITest, CreateBatchJsonResponse_EscapesEmbeddedQuoteInError) {
    ChatbotAPI::BatchResponse batch_resp;
    batch_resp.success = false;
    batch_resp.error = R"(failed","pwned":true)";

    std::string response = api->create_batch_json_response(batch_resp);

    EXPECT_EQ(response.find(R"("pwned":true)"), std::string::npos)
        << "unescaped quote in batch error injected a sibling field: " << response;
}

// ============================================================================
// Session Continuity Regression Tests (TD-095)
//
// handle_chat_session() used to only echo back whatever session_id arrived in
// the request. On the very first message of a new conversation the request
// carries no session_id at all (see ChatbotCLI::generate_response(), which
// only includes the field once it has a real one to send) — get_or_create_session()
// allocates a fresh internal id in that case, but handle_chat_session() never
// recovered it, so the response's "session_id" field stayed "" and the caller
// could never send it back on the next message. Every message silently
// created and abandoned a brand-new session — multi-turn history never
// accumulated. generate_batch_session_responses() already solved this exact
// case via a reverse pointer-lookup into sessions_; the same pattern is now
// applied to handle_chat_session().
// ============================================================================

TEST_F(ChatbotAPITest, HandleChatSession_NewConversationReturnsNonEmptySessionId) {
    std::string response = call_handle_chat_session(R"({"message":"hello"})");
    std::string session_id = api->parse_json_string(response, "session_id");

    EXPECT_FALSE(session_id.empty())
        << "server allocated a new session but never reported its id back to the caller: "
        << response;
}

TEST_F(ChatbotAPITest, HandleChatSession_ReturnedSessionIdContinuesTheSameConversation) {
    std::string first_response = call_handle_chat_session(R"({"message":"hello"})");
    std::string session_id = api->parse_json_string(first_response, "session_id");
    ASSERT_FALSE(session_id.empty());

    std::string second_request =
        R"({"session_id":")" + session_id + R"(","message":"how are you"})";
    std::string second_response = call_handle_chat_session(second_request);
    std::string second_session_id = api->parse_json_string(second_response, "session_id");

    EXPECT_EQ(second_session_id, session_id)
        << "the id the caller sent back should identify the same, still-live session";
}

// ============================================================================
// Session Export/Import Tests (TD-053)
//
// A session's conversation history lives entirely server-side (Session::context) — a remote
// ChatbotCLI has no local state of its own to persist, so /save and /load need real network
// round-trips instead of a local file operation. These two endpoints transport
// ConversationContext::serialize()/deserialize()'s string format over HTTP.
// ============================================================================

// TD-236: a missing field is a BadRequest (HTTP 400 from the route); it used to be returned with 200.
TEST_F(ChatbotAPITest, HandleExportSession_MissingSessionIdIsABadRequest) {
    try {
        call_handle_export_session(R"({})");
        FAIL() << "expected a bad request";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("Missing 'session_id'"), std::string::npos) << e.what();
    }
}

// TD-232: an id the server never issued is "Session not found" (HTTP 404 from the route).
TEST_F(ChatbotAPITest, HandleExportSession_UnknownSessionIdIsNotFound) {
    try {
        call_handle_export_session(R"({"session_id":"does-not-exist"})");
        FAIL() << "expected Session not found";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "Session not found");
    }
}

TEST_F(ChatbotAPITest, HandleExportSession_ReturnsSerializedConversationData) {
    std::string chat_response = call_handle_chat_session(R"({"message":"hello"})");
    std::string session_id = api->parse_json_string(chat_response, "session_id");
    ASSERT_FALSE(session_id.empty());

    std::string response =
        call_handle_export_session(R"({"session_id":")" + session_id + R"("})");
    EXPECT_NE(response.find(R"("success":true)"), std::string::npos) << response;
    std::string data = api->parse_json_string(response, "data");
    // ConversationContext::serialize()'s own format -- exercised directly in
    // conversationcontext_test.cpp, only spot-checked here to confirm the real conversation
    // (not a placeholder) crossed the HTTP boundary.
    EXPECT_NE(data.find("MAX_MESSAGES:"), std::string::npos) << data;
    EXPECT_NE(data.find("user|"), std::string::npos) << data;
}

// TD-227: turns on one session run one at a time (each holds the session's mutex from the user
// message to the reply). The ConversationContext has no locking of its own, and concurrent turns
// used to race on it; run under the TSan build to see the race on the old code.
TEST_F(ChatbotAPITest, ConcurrentTurnsOnOneSessionAreSerialized) {
    ChatbotAPI::GenerationConfig config;
    config.max_length = 4;
    config.strategy = "greedy";
    api->set_generation_config(config);

    constexpr int kTurns = 4;
    const std::string sid = start_session();
    ASSERT_FALSE(sid.empty());
    std::vector<std::thread> threads;
    std::vector<std::string> responses(kTurns);
    for (int i = 0; i < kTurns; ++i) {
        threads.emplace_back([&, i] {
            responses[i] = call_handle_chat_session(R"({"session_id":")" + sid +
                                                    R"(","message":"hello world"})");
        });
    }
    for (auto& t : threads) t.join();
    for (const auto& r : responses) {
        EXPECT_NE(r.find("\"success\":true"), std::string::npos) << r;
    }
    // Every turn (and the first) added its user message and its reply, in pairs.
    const std::string exported = call_handle_export_session(R"({"session_id":")" + sid + "\"}");
    ConversationContext restored(20, 100000);
    restored.deserialize(api->parse_json_string(exported, "data"));
    EXPECT_EQ(restored.get_message_count(), static_cast<size_t>(2 * (kTurns + 1))) << exported;
}

// TD-227: cleanup (run by /health) erasing a session while a request is using it. The request held a
// raw pointer into the erased entry: a use-after-free. Sessions are shared_ptrs now, so it completes.
TEST_F(ChatbotAPITest, CleanupDuringAnInFlightRequestKeepsTheSessionAlive) {
    api = std::make_unique<ChatbotAPI>(model.get(), tokenizer.get(), 8080, /*session_timeout=*/0);
    ChatbotAPI::GenerationConfig config;
    config.max_length = 40;
    config.strategy = "greedy";
    api->set_generation_config(config);

    std::atomic<bool> done{false};
    std::string response;
    std::thread request([&] {
        response = call_handle_chat_session(R"({"message":"hello world"})");  // a new session
        done = true;
    });
    int cleanups = 0;
    while (!done) {
        call_handle_health();  // timeout 0: erases every session, including the request's
        ++cleanups;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    request.join();
    EXPECT_GT(cleanups, 0);
    EXPECT_NE(response.find("\"success\":true"), std::string::npos) << response;
}

// TD-292: an over-long message used to evict the whole history (and itself), then fail with HTTP 500
// on the empty prompt. It's refused now (HTTP 400 from the route) and the history is kept.
TEST_F(ChatbotAPITest, OversizedChatMessageIsRejectedAndTheHistoryKept) {
    ChatbotAPI::GenerationConfig config;
    config.max_length = 3;
    config.strategy = "greedy";
    api->set_generation_config(config);
    const std::string sid = start_session();
    ASSERT_FALSE(sid.empty());

    std::string huge;
    for (int i = 0; i < 3000; ++i)
        huge += "hello ";
    EXPECT_THROW(call_handle_chat_session(R"({"session_id":")" + sid + R"(","message":")" + huge + "\"}"),
                 std::invalid_argument);

    const std::string exported = call_handle_export_session(R"({"session_id":")" + sid + "\"}");
    ConversationContext restored(20, 100000);
    restored.deserialize(api->parse_json_string(exported, "data"));
    EXPECT_EQ(restored.get_message_count(), 2u) << "the earlier turn is still there";
}

// TD-291: imports are capped in size (the session's own limits bound what it keeps).
TEST_F(ChatbotAPITest, OversizedImportIsRejected) {
    const std::string data(2 * 1024 * 1024, 'x');
    EXPECT_THROW(call_handle_import_session(R"({"data":")" + data + "\"}"), std::invalid_argument);
}

TEST_F(ChatbotAPITest, HandleImportSession_MissingDataIsABadRequest) {
    try {
        call_handle_import_session(R"({"session_id":"whatever"})");
        FAIL() << "expected a bad request";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("Missing 'data'"), std::string::npos) << e.what();
    }
}

TEST_F(ChatbotAPITest, HandleImportSession_CreatesNewSessionWhenNoneGiven) {
    ConversationContext source;
    source.add_user_message("Imported question");
    source.add_assistant_message("Imported answer");

    std::string request = R"({"data":")" + api->escape_json_string(source.serialize()) + R"("})";
    std::string response = call_handle_import_session(request);

    EXPECT_NE(response.find(R"("success":true)"), std::string::npos) << response;
    std::string session_id = api->parse_json_string(response, "session_id");
    EXPECT_FALSE(session_id.empty())
        << "server should allocate and report a fresh session id, the same TD-095 fix "
           "handle_chat_session() already applies";
    EXPECT_NE(response.find(R"("message_count":2)"), std::string::npos) << response;
}

TEST_F(ChatbotAPITest, ExportThenImportRoundTripPreservesMessageContent) {
    // Real end-to-end round trip through both new endpoints, including a message content that
    // needs real JSON escaping (an embedded quote) to survive the trip -- the direct analog of
    // ChatbotCLISaveLoadRoundTripTest.SaveThenLoadRestoresSessionOnAFreshClient in
    // chatbotcli_improved_test.cpp, but exercising ChatbotAPI's own handlers directly rather
    // than through a real ChatbotCLI + real sockets.
    std::string chat_response =
        call_handle_chat_session(R"({"message":")" +
                                 api->escape_json_string(R"(say "hi" please)") + R"("})");
    std::string session_id = api->parse_json_string(chat_response, "session_id");
    ASSERT_FALSE(session_id.empty());

    std::string export_response =
        call_handle_export_session(R"({"session_id":")" + session_id + R"("})");
    ASSERT_NE(export_response.find(R"("success":true)"), std::string::npos) << export_response;
    std::string data = api->parse_json_string(export_response, "data");

    std::string import_response =
        call_handle_import_session(R"({"data":")" + api->escape_json_string(data) + R"("})");
    ASSERT_NE(import_response.find(R"("success":true)"), std::string::npos) << import_response;
    EXPECT_NE(import_response.find(R"("message_count":2)"), std::string::npos) << import_response;

    ConversationContext restored;
    ASSERT_NO_THROW(restored.deserialize(data));
    ASSERT_EQ(restored.get_message_count(), 2);
    EXPECT_EQ(restored.get_messages()[0].content, R"(say "hi" please)");
}

// ============================================================================
// Generation Configuration Tests
// ============================================================================

TEST_F(ChatbotAPITest, GenerationConfig_Defaults) {
    ChatbotAPI::GenerationConfig config;
    EXPECT_EQ(config.max_length, 100);
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_FLOAT_EQ(config.top_p, 0.9f);
    EXPECT_EQ(config.top_k, 50);
    EXPECT_EQ(config.strategy, "nucleus");
    EXPECT_EQ(config.beam_width, 4);
}

TEST_F(ChatbotAPITest, SetGenerationConfig) {
    ChatbotAPI::GenerationConfig config;
    config.max_length = 50;
    config.temperature = 0.8f;
    config.strategy = "greedy";

    api->set_generation_config(config);

    // Config is set internally, we can't easily verify without server running
    // But we can check that the method doesn't crash
    SUCCEED();
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

TEST_F(ChatbotAPITest, GenerateBatchResponses_BasicBatch) {
    std::vector<std::string> inputs = {"hello", "hi world", "test message"};

    ChatbotAPI::GenerationConfig config;
    config.max_length = 10;  // Short for speed
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 3);
    EXPECT_GT(response.stats.total_tokens, 0);
    EXPECT_GT(response.stats.actual_tokens, 0);
    EXPECT_GE(response.stats.padding_ratio, 0.0f);
    EXPECT_LE(response.stats.padding_ratio, 1.0f);
}

TEST_F(ChatbotAPITest, GenerateBatchResponses_EmptyInput) {
    std::vector<std::string> inputs;
    ChatbotAPI::GenerationConfig config;

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    // Should handle empty input gracefully
    EXPECT_EQ(response.responses.size(), 0);
}

TEST_F(ChatbotAPITest, GenerateBatchResponses_SingleInput) {
    std::vector<std::string> inputs = {"single message"};
    ChatbotAPI::GenerationConfig config;
    config.max_length = 10;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 1);
}

TEST_F(ChatbotAPITest, GenerateBatchResponses_VariableLengths) {
    std::vector<std::string> inputs = {
        "hi",                                                                      // Very short
        "hello world",                                                             // Short
        "this is a medium length message",                                         // Medium
        "this is a much longer message that should test variable length handling"  // Long
    };

    ChatbotAPI::GenerationConfig config;
    config.max_length = 10;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 4);

    // With variable lengths, we should see some batching efficiency
    // Multiple batches should be created
    EXPECT_GE(response.stats.num_batches, 1);
}

// TD-089: generate_batch_responses() used to iterate create_dynamic_batches()'
// output to build the response list. That function sorts sequences by length
// internally and its TokenBatches carry no memory of each sequence's original
// position, so with inputs of differing lengths the returned responses came
// back in length-sorted order instead of matching request order — response[i]
// was not necessarily the answer to inputs[i]. Verified here by comparing
// against generate_response()'s single-input path (deterministic under
// "greedy"), called once per input in the caller's original order, which is
// unaffected by any internal batching/reordering.
TEST_F(ChatbotAPITest, GenerateBatchResponses_PreservesInputOrder) {
    // Deliberately NOT in ascending (or descending) length order — this is
    // what actually exercises create_dynamic_batches()'s internal
    // length-sort: an already length-sorted input list would "reorder" into
    // the same order it started in and wouldn't catch a regression here.
    std::vector<std::string> inputs = {
        "this is a much longer message that should test variable length handling",  // Long
        "hi",                                                                       // Very short
        "this is a medium length message",                                         // Medium
        "hello world",                                                              // Short
    };

    ChatbotAPI::GenerationConfig config;
    config.max_length = 8;
    config.strategy = "greedy";

    std::vector<std::string> expected;
    for (const auto& input : inputs) {
        expected.push_back(call_generate_response(input, config));
    }

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    ASSERT_TRUE(response.success);
    ASSERT_EQ(response.responses.size(), inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_EQ(response.responses[i], expected[i])
            << "Batch response at index " << i << " (input: '" << inputs[i] << "') does not "
            << "match the single-input reference for the same input — batch responses are "
            << "misordered relative to the request.";
    }
}

TEST_F(ChatbotAPITest, BatchResponse_Statistics) {
    std::vector<std::string> inputs = {"test1", "test2", "test3"};
    ChatbotAPI::GenerationConfig config;
    config.max_length = 10;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);

    // Verify statistics are populated
    EXPECT_GT(response.stats.total_tokens, 0);
    EXPECT_GT(response.stats.actual_tokens, 0);
    EXPECT_LE(response.stats.actual_tokens, response.stats.total_tokens);
    EXPECT_GE(response.stats.padding_ratio, 0.0f);
    EXPECT_LE(response.stats.padding_ratio, 1.0f);
    EXPECT_GE(response.stats.num_batches, 1);
    EXPECT_GT(response.stats.avg_batch_size, 0.0f);

    // Efficiency should be inverse of padding ratio
    float expected_efficiency = (1.0f - response.stats.padding_ratio) * 100.0f;
    EXPECT_NEAR(expected_efficiency, expected_efficiency, 0.1f);
}

// ============================================================================
// Session Management Tests
// ============================================================================

TEST_F(ChatbotAPITest, SessionCreation) {
    // Sessions are created internally by get_or_create_session
    // We test this indirectly through the API
    SUCCEED();
}

TEST_F(ChatbotAPITest, GenerateBatchSessionResponses_NewSessions) {
    std::vector<std::string> inputs = {"Hello", "Hi there"};
    std::vector<std::string> session_ids = {"", ""};  // Empty = create new

    ChatbotAPI::GenerationConfig config;
    config.max_length = 10;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response =
        api->generate_batch_session_responses(inputs, session_ids, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 2);
    EXPECT_EQ(response.session_ids.size(), 2);

    // Session IDs should be created (non-empty)
    EXPECT_FALSE(response.session_ids[0].empty());
    EXPECT_FALSE(response.session_ids[1].empty());

    // Session IDs should be different
    EXPECT_NE(response.session_ids[0], response.session_ids[1]);
}

TEST_F(ChatbotAPITest, GenerateBatchSessionResponses_ExistingSessions) {
    // First batch: create sessions
    std::vector<std::string> inputs1 = {"First message"};
    std::vector<std::string> session_ids1 = {""};

    ChatbotAPI::GenerationConfig config;
    config.max_length = 10;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response1 =
        api->generate_batch_session_responses(inputs1, session_ids1, config);

    ASSERT_TRUE(response1.success);
    ASSERT_EQ(response1.session_ids.size(), 1);

    // Second batch: use existing session
    std::vector<std::string> inputs2 = {"Follow-up message"};
    std::vector<std::string> session_ids2 = {response1.session_ids[0]};

    ChatbotAPI::BatchResponse response2 =
        api->generate_batch_session_responses(inputs2, session_ids2, config);

    EXPECT_TRUE(response2.success);
    EXPECT_EQ(response2.session_ids.size(), 1);
    EXPECT_EQ(response2.session_ids[0], response1.session_ids[0]);
}

TEST_F(ChatbotAPITest, GenerateBatchSessionResponses_MixedSessions) {
    const std::string existing = start_session();
    ASSERT_FALSE(existing.empty());
    std::vector<std::string> inputs = {"Msg1", "Msg2", "Msg3"};
    std::vector<std::string> session_ids = {existing, "", existing};

    ChatbotAPI::GenerationConfig config;
    config.max_length = 10;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response =
        api->generate_batch_session_responses(inputs, session_ids, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 3);
    ASSERT_EQ(response.session_ids.size(), 3);
    EXPECT_EQ(response.session_ids[0], existing);
    EXPECT_EQ(response.session_ids[1].size(), 32u);  // a new server-issued id
}

// TD-232: a batch naming an id the server didn't issue fails whole (HTTP 404) and creates nothing.
TEST_F(ChatbotAPITest, GenerateBatchSessionResponses_UnknownIdIsNotFound) {
    ChatbotAPI::GenerationConfig config;
    config.max_length = 4;
    EXPECT_THROW(api->generate_batch_session_responses({"a", "b"}, {"", "client-made"}, config),
                 std::runtime_error);    EXPECT_EQ(call_session_count(), 0u);
}

// ============================================================================
// Strategy Tests
// ============================================================================

TEST_F(ChatbotAPITest, GenerationStrategy_Greedy) {
    std::vector<std::string> inputs = {"test"};
    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);
    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 1);
}

TEST_F(ChatbotAPITest, GenerationStrategy_Beam) {
    std::vector<std::string> inputs = {"test"};
    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "beam";
    config.beam_width = 2;

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);
    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 1);
}

TEST_F(ChatbotAPITest, GenerationStrategy_Temperature) {
    std::vector<std::string> inputs = {"test"};
    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "temperature";
    config.temperature = 0.8f;

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);
    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 1);
}

TEST_F(ChatbotAPITest, GenerationStrategy_TopK) {
    std::vector<std::string> inputs = {"test"};
    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "top_k";
    config.top_k = 10;

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);
    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 1);
}

TEST_F(ChatbotAPITest, GenerationStrategy_Nucleus) {
    std::vector<std::string> inputs = {"test"};
    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "nucleus";
    config.top_p = 0.9f;

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);
    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 1);
}

TEST_F(ChatbotAPITest, GenerationStrategy_Default) {
    std::vector<std::string> inputs = {"test"};
    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "invalid_strategy";  // Should default to nucleus

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);
    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 1);
}

// ============================================================================
// Efficiency Tests
// ============================================================================

TEST_F(ChatbotAPITest, BatchEfficiency_UniformLength) {
    // All messages same length should have high efficiency
    std::vector<std::string> inputs = {"test1", "test2", "test3", "test4"};

    ChatbotAPI::GenerationConfig config;
    config.max_length = 10;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);

    // Uniform length should result in lower padding ratio
    EXPECT_LT(response.stats.padding_ratio, 0.3f);  // Less than 30% padding
}

TEST_F(ChatbotAPITest, BatchEfficiency_VaryingLength) {
    // Variable length messages
    std::vector<std::string> inputs = {"a", "hello world test", "ab",
                                       "this is a much longer message for testing"};

    ChatbotAPI::GenerationConfig config;
    config.max_length = 10;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);

    // Variable lengths may result in higher padding, but dynamic batching helps
    // Should still be reasonable
    EXPECT_LT(response.stats.padding_ratio, 0.7f);  // Less than 70% padding
}

// ============================================================================
// Large Batch Tests
// ============================================================================

TEST_F(ChatbotAPITest, LargeBatch_10Messages) {
    std::vector<std::string> inputs;
    for (int i = 0; i < 10; ++i) {
        inputs.push_back("message " + std::to_string(i));
    }

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 10);
}

TEST_F(ChatbotAPITest, LargeBatch_50Messages) {
    std::vector<std::string> inputs;
    for (int i = 0; i < 50; ++i) {
        inputs.push_back("test message number " + std::to_string(i));
    }

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 50);

    // Should create multiple batches (max batch size is 32)
    EXPECT_GE(response.stats.num_batches, 2);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ChatbotAPITest, EdgeCase_VeryLongMessage) {
    std::string long_msg(500, 'x');  // 500 character message
    std::vector<std::string> inputs = {long_msg};

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 1);
}

TEST_F(ChatbotAPITest, EdgeCase_SpecialCharacters) {
    std::vector<std::string> inputs = {"Hello\nWorld", "Tab\there", "Quote\"test",
                                       "Backslash\\test"};

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 4);
}

TEST_F(ChatbotAPITest, EdgeCase_UnicodeCharacters) {
    std::vector<std::string> inputs = {"Hello 世界", "Test éàü", "Emoji 🚀"};

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response = api->generate_batch_responses(inputs, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 3);
}

// ============================================================================
// Profiling (TD-038): GET /admin/profile wiring
// ============================================================================
//
// These are integration tests in the sense the TD's own action items ask for — proving
// enable_profiling() and generate_response()'s PROFILE_SCOPE actually connect end to end through
// a real ChatbotAPI instance and real (small) model inference, not just that Profiler/ProfileStats
// work in isolation (PerformanceProfilerTest already covers that).

TEST_F(ChatbotAPITest, Profile_DisabledByDefaultReportsDisabled) {
    std::string response = call_handle_profile();
    EXPECT_NE(response.find("\"enabled\":false"), std::string::npos);
    EXPECT_EQ(response.find("\"enabled\":true"), std::string::npos);
}

TEST_F(ChatbotAPITest, Profile_DisabledEvenAfterGeneratingResponses) {
    // generate_response() always times internally (PROFILE_SCOPE runs regardless of
    // enable_profiling()) — confirms the flag gates exposure only, not collection: stats exist
    // internally the moment enable_profiling() is called later, with no warm-up gap.
    ChatbotAPI::GenerationConfig config;
    config.max_length = 3;
    config.strategy = "greedy";
    call_generate_response("hello", config);

    EXPECT_NE(call_handle_profile().find("\"enabled\":false"), std::string::npos);
}

TEST_F(ChatbotAPITest, Profile_EnabledReportsRealTimingAfterGeneration) {
    api->enable_profiling(true);

    ChatbotAPI::GenerationConfig config;
    config.max_length = 3;
    config.strategy = "greedy";
    call_generate_response("hello", config);
    call_generate_response("world", config);

    std::string response = call_handle_profile();
    EXPECT_NE(response.find("\"enabled\":true"), std::string::npos);
    EXPECT_NE(response.find("\"section\":\"generate_response\""), std::string::npos);
    EXPECT_NE(response.find("\"call_count\":2"), std::string::npos)
        << "both generate_response() calls above must be reflected: " << response;
    // mean_ms must be present and non-negative — real inference always takes >= 0ms; the
    // point of this assertion is that a real number appears here at all, not a specific value.
    EXPECT_NE(response.find("\"mean_ms\":"), std::string::npos);
}

TEST_F(ChatbotAPITest, Profile_EnabledBeforeAnyGenerationReportsZeroCalls) {
    api->enable_profiling(true);
    std::string response = call_handle_profile();
    EXPECT_NE(response.find("\"enabled\":true"), std::string::npos);
    EXPECT_NE(response.find("\"call_count\":0"), std::string::npos) << response;
}

// ============================================================================
// Speculative Decoding (TD-038): draft-model wiring
// ============================================================================
//
// Integration tests in the same sense as the Profiling section above — proving the wiring
// through a real second EncoderDecoderModel and real (small) inference, not just
// SpeculativeDecoder's own existing unit tests in isolation.

TEST_F(ChatbotAPITest, SpeculativeDecoding_ZeroCandidatesReachesTheDraftPathAndThrows) {
    // A deliberately observable, falsifiable signal that generate_response() genuinely reaches
    // SpeculativeDecoder rather than silently falling through to the plain strategy-based path:
    // SpeculativeDecoder rejects num_candidates=0 (TD-507; it used to propose nothing and end with
    // zero tokens), and generate_response() rethrows that as a generation failure. Plain
    // generation never throws on this model, so the exception can only come from the speculative
    // path.
    auto draft_model = std::make_unique<EncoderDecoderModel>(tokenizer->get_vocab_size(),
                                                             /*d_model=*/16, /*enc_layers=*/1,
                                                             /*dec_layers=*/1, /*num_heads=*/2,
                                                             /*d_ff=*/32, /*max_seq_length=*/128);
    api = std::make_unique<ChatbotAPI>(model.get(), tokenizer.get(), 8080, 30, draft_model.get(),
                                       /*speculative_num_candidates=*/0);

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    EXPECT_THROW(call_generate_response("hello", config), std::runtime_error);
}

TEST_F(ChatbotAPITest, SpeculativeDecoding_RealCandidateCountGeneratesWithoutThrowing) {
    auto draft_model = std::make_unique<EncoderDecoderModel>(tokenizer->get_vocab_size(), 16, 1, 1,
                                                             2, 32, 128);
    api = std::make_unique<ChatbotAPI>(model.get(), tokenizer.get(), 8080, 30, draft_model.get(),
                                       /*speculative_num_candidates=*/4);

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    EXPECT_NO_THROW(call_generate_response("hello world", config));
}

TEST_F(ChatbotAPITest, SpeculativeDecoding_NoDraftModelUsesNormalPathUnaffected) {
    // No draft model configured (the fixture's default `api`) — confirms adding the
    // draft_model_/speculative_num_candidates_ constructor parameters didn't disturb the
    // existing, already-covered normal generation path.
    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";
    EXPECT_NO_THROW(call_generate_response("hello", config));
}

// ============================================================================
// Batched Inference (TD-038): BatchedInferenceEngine wiring
// ============================================================================
//
// Integration tests in the same sense as the two sections above — proving generate_response()
// genuinely routes through a real BatchedInferenceEngine worker thread (not a no-op wrapper)
// via real (small) inference through a real ChatbotAPI instance.

TEST_F(ChatbotAPITest, BatchedInference_DisabledByDefaultUsesNormalPathUnaffected) {
    // No enable_batched_inference() call — confirms adding batched_engine_ didn't disturb the
    // existing, already-covered normal generation path.
    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";
    EXPECT_NO_THROW(call_generate_response("hello", config));
}

TEST_F(ChatbotAPITest, BatchedInference_EnabledGeneratesThroughTheQueueWithoutThrowing) {
    // A short timeout keeps this test fast rather than waiting out the default 50ms per request.
    BatchedInferenceConfig batch_config;
    batch_config.timeout_ms = 10;
    api->enable_batched_inference(batch_config);

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";
    EXPECT_NO_THROW(call_generate_response("hello world", config));

    // Falsifiable proof the request actually went through the engine's real queue/worker thread
    // rather than silently falling through to the plain inline path (which would also succeed
    // without throwing, so EXPECT_NO_THROW alone doesn't prove this).
    EXPECT_EQ(call_get_batched_stats().total_requests, 1u);
}

TEST_F(ChatbotAPITest, BatchedInference_ConcurrentRequestsAllCompleteCorrectly) {
    // Real proof of wiring: if generate_response() silently fell back to the inline path (or the
    // engine's worker thread were broken), this would either hang (nothing ever resolves the
    // futures) or throw. Firing several requests concurrently from real threads exercises the
    // engine's actual queue/worker-thread machinery, not just a single call that could coincidentally
    // succeed even if submit()/process_batch() were subtly wrong. This is also the test that
    // surfaced (and, once fixed, now safely exercises) the Profiler thread-safety bug: every one of
    // these concurrent generate_response() calls internally profiles itself via the always-on
    // PROFILE_SCOPE.
    BatchedInferenceConfig batch_config;
    batch_config.timeout_ms = 10;
    api->enable_batched_inference(batch_config);
    api->enable_profiling(true);

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    constexpr int kNumThreads = 6;
    std::vector<std::thread> threads;
    std::vector<std::string> results(kNumThreads);
    std::vector<bool> threw(kNumThreads, false);
    std::vector<std::string> errors(kNumThreads);

    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&, i]() {
            try {
                results[i] = call_generate_response("hello world", config);
            } catch (const std::exception& e) {
                threw[i] = true;
                errors[i] = e.what();
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    for (int i = 0; i < kNumThreads; ++i) {
        // Not asserting non-empty: generate_text() (used by every request here) always samples
        // via combined top-k/top-p/temperature regardless of GenerationConfig::strategy, so on
        // this tiny randomly-initialized test model it can legitimately sample EOS as the very
        // first token — decode(..., skip_special_tokens=true) then correctly returns "". That's
        // valid output, not a wiring failure; only an actual exception indicates one here.
        EXPECT_FALSE(threw[i]) << "concurrent request " << i << " threw unexpectedly: "
                               << errors[i];
    }

    // Every concurrent call's PROFILE_SCOPE must have recorded independently and correctly —
    // exactly the scenario the Profiler thread-safety fix (PerformanceProfiler.hpp) targets.
    std::string profile = call_handle_profile();
    EXPECT_NE(profile.find("\"call_count\":" + std::to_string(kNumThreads)), std::string::npos)
        << "expected all " << kNumThreads << " concurrent calls reflected: " << profile;

    // Same discriminator as BatchedInference_EnabledGeneratesThroughTheQueueWithoutThrowing above:
    // proves all kNumThreads requests genuinely passed through the engine, not just that none of
    // them threw.
    EXPECT_EQ(call_get_batched_stats().total_requests, static_cast<uint64_t>(kNumThreads));
}

// ============================================================================
// TD-156: Plain (non-batched, non-pipelined) path concurrency
// ============================================================================
//
// Unlike every concurrency test above, this one deliberately doesn't enable TD-038's
// batched inference mode — it exercises the plain default path every
// chatbot_api_server deployment uses unless that flag is passed. That path calls
// model_->forward() directly on each request's own handler thread via a per-request model_fn
// closure (see ChatbotAPI::generate_response()), with no serialization of its own; the batched
// engine above is incidentally safe only because it routes every
// call through its own single worker thread, which is exactly what makes it a poor stand-in
// for testing this path.
//
// Before the TD-156 fix (EncoderDecoderModel::forward()/generate_response()/
// generate_response_with_strategy()/backward() all writing/reading cached_encoder_output et al.
// with no lock), concurrent calls into the same model instance raced on those members'
// internal heap buffers — confirmed via a manual repro crashing 13/15 runs with genuine glibc
// heap-corruption signatures ("double free or corruption (!prev)", "malloc(): unaligned tcache
// chunk detected", "free(): corrupted unsorted chunks"). That corruption aborts the whole
// process (SIGABRT from glibc's own allocator, not a catchable C++ exception), so the
// regression signal here is "this test binary completes at all" as much as any individual
// EXPECT — running many rounds raises the odds a reverted fix actually gets caught in one CI
// run rather than passing by luck the way a single round of 6 threads sometimes did in manual
// testing.
TEST_F(ChatbotAPITest, PlainPathConcurrentRequestsDoNotCorruptTheHeap) {
    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    constexpr int kNumRounds = 20;
    constexpr int kNumThreads = 6;

    for (int round = 0; round < kNumRounds; ++round) {
        std::vector<std::thread> threads;
        std::vector<bool> threw(kNumThreads, false);
        std::vector<std::string> errors(kNumThreads);

        for (int i = 0; i < kNumThreads; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    call_generate_response("hello world", config);
                } catch (const std::exception& e) {
                    threw[i] = true;
                    errors[i] = e.what();
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }

        for (int i = 0; i < kNumThreads; ++i) {
            // Not asserting non-empty output, same reason as the batched
            // version of this test above: this tiny randomly-initialized model can legitimately
            // sample EOS as its very first token. Only an actual exception (or the process not
            // surviving to report it) indicates the bug this test targets.
            EXPECT_FALSE(threw[i]) << "round " << round << ", concurrent request " << i
                                   << " threw unexpectedly: " << errors[i];
        }
    }
}

// ============================================================================
// TD-506: CPU generation callbacks are aligned with training
// ============================================================================
//
// A model trained to echo a few phrases must generate them exactly through every ChatbotAPI CPU
// path. The callbacks used to call the training forward(), whose decoder input is
// [BOS] + targets[:-1]: given the generated sequence, its last row predicted the last token
// already generated, so a correctly trained model answered wrongly (0/12 vs 12/12 when verified).

class ChatbotAPIEchoTest : public ChatbotAPITest {
   protected:
    const std::vector<std::string> phrases = {"hello world", "hi there", "how are you",
                                              "test message"};

    void SetUp() override {
        ChatbotAPITest::SetUp();
        // Initialization isn't seeded (TD-368). With 300 epochs about 1 attempt in 4 stopped near
        // loss 0.03, short of the target, and 3 failed attempts in a row failed the test about once
        // per 100 fixtures; with 1000 (training stops early once it reaches the target) 40 of 40
        // attempts converged, in 4 s on average. A fresh model is still tried before giving up.
        bool trained = train_echo();
        for (int attempt = 1; attempt < 3 && !trained; ++attempt) {
            model = std::make_unique<EncoderDecoderModel>(tokenizer->get_vocab_size(), 32, 1, 1, 2,
                                                          64, 128);
            model->get_tokenizer()->build_vocab(
                {"hello world test", "hi there how are you", "what is the answer to this question",
                 "test message for training"},
                50);
            api = std::make_unique<ChatbotAPI>(model.get(), tokenizer.get(), 8080, 30);
            trained = train_echo();
        }
        // This used to check whichever model the loop built last, which after three failed attempts
        // was a fresh, untrained fourth one (hence the reported loss of about ln 50).
        ASSERT_TRUE(trained) << "the echo model didn't train (loss " << echo_loss() << ")";
    }

    static constexpr float kTrainedLoss = 0.01f;

    // Teacher-forced CPU training, as ChatbotTrainer does it: targets are the input's tokens + EOS.
    // Returns whether the mean loss got below kTrainedLoss.
    bool train_echo() {
        Optimizer optimizer(OptimizerType::ADAM, 0.01f);
        model->register_parameters(optimizer);
        for (int epoch = 0; epoch < 1000; ++epoch) {
            if (epoch % 50 == 0 && epoch >= 100) {
                model->set_training(false);
                if (echo_loss() < kTrainedLoss)
                    return true;
            }
            model->set_training(true);
            for (const auto& phrase : phrases) {
                const std::vector<int> input = tokenizer->encode(phrase, false);
                std::vector<int> target = input;
                target.push_back(tokenizer->get_eos_token_id());
                model->zero_grad();
                optimizer.zero_grad();
                Matrix logits = model->forward(input, target);
                model->backward_pass(model->compute_loss_gradient_for_training(logits, target));
                optimizer.step();
            }
        }
        model->set_training(false);
        return echo_loss() < kTrainedLoss;
    }

    float echo_loss() {
        float total = 0.0f;
        for (const auto& phrase : phrases) {
            const std::vector<int> input = tokenizer->encode(phrase, false);
            std::vector<int> target = input;
            target.push_back(tokenizer->get_eos_token_id());
            total += model->compute_loss_for_training(model->forward(input, target), target);
        }
        return total / static_cast<float>(phrases.size());
    }

    std::string expected(const std::string& phrase) {
        return tokenizer->decode(tokenizer->encode(phrase, false));
    }

    static ChatbotAPI::GenerationConfig greedy_config() {
        ChatbotAPI::GenerationConfig config;
        config.max_length = 20;
        config.strategy = "greedy";
        return config;
    }
};

TEST_F(ChatbotAPIEchoTest, PlainCpuGenerationEchoesExactly) {
    for (const auto& phrase : phrases) {
        EXPECT_EQ(call_generate_response(phrase, greedy_config()), expected(phrase)) << phrase;
    }
}

TEST_F(ChatbotAPIEchoTest, PlainCpuBeamSearchEchoesExactly) {
    auto config = greedy_config();
    config.strategy = "beam";
    config.beam_width = 3;
    for (const auto& phrase : phrases) {
        EXPECT_EQ(call_generate_response(phrase, config), expected(phrase)) << phrase;
    }
}

TEST_F(ChatbotAPIEchoTest, BatchEndpointEchoesExactly) {
    ChatbotAPI::BatchResponse response = api->generate_batch_responses(phrases, greedy_config());
    ASSERT_TRUE(response.success) << response.error;
    ASSERT_EQ(response.responses.size(), phrases.size());
    for (size_t i = 0; i < phrases.size(); ++i) {
        EXPECT_EQ(response.responses[i], expected(phrases[i])) << phrases[i];
    }
}

TEST_F(ChatbotAPIEchoTest, SpeculativeDecodingEchoesExactly) {
    // The trained model is its own draft: every candidate is the target's own choice and is
    // accepted, so the output is the greedy answer (TD-507's sampling issues don't arise).
    api = std::make_unique<ChatbotAPI>(model.get(), tokenizer.get(), 8080, 30, model.get(),
                                       /*speculative_num_candidates=*/3);
    for (const auto& phrase : phrases) {
        EXPECT_EQ(call_generate_response(phrase, greedy_config()), expected(phrase)) << phrase;
    }
}

TEST_F(ChatbotAPIEchoTest, BatchedInferenceEchoesExactly) {
    BatchedInferenceConfig batch_config;
    batch_config.timeout_ms = 10;
    api->enable_batched_inference(batch_config);
    // TD-214: batched mode samples whatever the strategy; temperature 0 makes it greedy.
    auto config = greedy_config();
    config.temperature = 0.0f;
    for (const auto& phrase : phrases) {
        EXPECT_EQ(call_generate_response(phrase, config), expected(phrase)) << phrase;
    }
}

TEST_F(ChatbotAPIEchoTest, NextTokenLogitsIsTheLastRowOfTheAlignedForward) {
    // TD-424: next_token_logits() projects only the last decoder row. Its row equals the last row
    // of the training forward whose decoder input is the same sequence ([BOS] + generated).
    const std::vector<int> input = tokenizer->encode("hello world", false);
    std::vector<int> generated = {tokenizer->get_bos_token_id()};
    for (int id : input)
        generated.push_back(id);

    std::vector<int> shifted_targets(generated.begin() + 1, generated.end());
    shifted_targets.push_back(tokenizer->get_eos_token_id());
    const Matrix full = model->forward(input, shifted_targets);
    const Matrix next = model->next_token_logits(input, generated);

    ASSERT_EQ(next.rows, 1);
    ASSERT_EQ(next.cols, full.cols);
    for (int j = 0; j < full.cols; ++j) {
        EXPECT_NEAR(next.data[0][j], full.data[full.rows - 1][j], 1e-4f) << j;
    }
    // Without the leading BOS the same sequence is accepted unchanged.
    const Matrix again = model->next_token_logits(input, {generated.begin() + 1, generated.end()});
    for (int j = 0; j < full.cols; ++j) {
        EXPECT_NEAR(again.data[0][j], next.data[0][j], 1e-6f) << j;
    }
}

TEST_F(ChatbotAPIEchoTest, NextTokenLogitsLastRowsMatchTheAlignedForward) {
    // TD-507: speculative decoding's target asks for its last K + 1 rows in one pass.
    const std::vector<int> input = tokenizer->encode("how are you", false);
    std::vector<int> generated = {tokenizer->get_bos_token_id()};
    for (int id : input)
        generated.push_back(id);
    std::vector<int> shifted_targets(generated.begin() + 1, generated.end());
    shifted_targets.push_back(tokenizer->get_eos_token_id());
    const Matrix full = model->forward(input, shifted_targets);

    const Matrix tail = model->next_token_logits(input, generated, 3);
    ASSERT_EQ(tail.rows, 3);
    for (int r = 0; r < 3; ++r)
        for (int j = 0; j < full.cols; ++j)
            EXPECT_NEAR(tail.data[r][j], full.data[full.rows - 3 + r][j], 1e-4f) << r << " " << j;

    // More rows than positions gives every position.
    EXPECT_EQ(model->next_token_logits(input, generated, 1000).rows, full.rows);
}

#ifdef ADAI_ENABLE_GPU
// TD-033: confirms the GPU-resident decode path (ChatbotAPI::generate_response() ->
// EncoderDecoderModel::gpu_generate_response_with_strategy()) actually gets exercised and
// produces valid, non-crashing output for every strategy ChatbotAPI::GenerationConfig::strategy
// can hold, on a real device. Skips gracefully when no physical GPU is present -- confirmed via
// tests/gpuutils_test.cpp (TD-041) that this codebase's own gpu/sycl presets cannot be assumed
// to have real hardware attached; asserting real generation here unconditionally would make this
// test fail everywhere except on a machine that happens to have a GPU, rather than verify
// anything.
TEST_F(ChatbotAPITest, GpuResidentPathProducesValidOutputForEveryStrategy) {
    if (!adai::gpu::GPUManager::is_available() && !adai::gpu::GPUManager::initialize()) {
        GTEST_SKIP() << "No GPU device present.";
    }

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;

    for (const std::string& strategy : {"greedy", "beam", "temperature", "top_k", "nucleus"}) {
        config.strategy = strategy;
        std::string response;
        EXPECT_NO_THROW(response = call_generate_response("hello world", config))
            << "strategy=" << strategy;
    }
}

// TD-033: GPU-resident analog of PlainPathConcurrentRequestsDoNotCorruptTheHeap above --
// confirms EncoderDecoderModel::gpu_generate_response_with_strategy()'s model_mutex_ lock
// (added alongside this same fix; see that method's own doc comment in
// EncoderDecoderModel.hpp for why it's needed: encoder/decoder/lm_head's own gpu_forward()
// calls write through to persistent per-instance GPUState that two concurrent callers would
// otherwise race on, the same class of bug TD-156 fixed for the CPU cached_* members). Not
// verified via revert-confirm-fail against a real crash the way TD-156's own regression test
// was: with no physical GPU present in any environment this codebase has been developed in so
// far, GPUManager::is_available() never becomes true, so the GPU-resident branch in
// ChatbotAPI::generate_response() is never actually taken regardless of whether the lock is
// present -- there is nothing to race on without real hardware. Skips gracefully for the same
// reason as the test above; left in place as permanent coverage for whenever a GPU is available.
TEST_F(ChatbotAPITest, GpuResidentPathConcurrentRequestsDoNotCrash) {
    if (!adai::gpu::GPUManager::is_available() && !adai::gpu::GPUManager::initialize()) {
        GTEST_SKIP() << "No GPU device present.";
    }

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";

    constexpr int kNumRounds = 20;
    constexpr int kNumThreads = 6;

    for (int round = 0; round < kNumRounds; ++round) {
        std::vector<std::thread> threads;
        std::vector<bool> threw(kNumThreads, false);
        std::vector<std::string> errors(kNumThreads);

        for (int i = 0; i < kNumThreads; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    call_generate_response("hello world", config);
                } catch (const std::exception& e) {
                    threw[i] = true;
                    errors[i] = e.what();
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }

        for (int i = 0; i < kNumThreads; ++i) {
            EXPECT_FALSE(threw[i]) << "round " << round << ", concurrent request " << i
                                   << " threw unexpectedly: " << errors[i];
        }
    }
}
#endif  // ADAI_ENABLE_GPU

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ============================================================================
// TD-230: RAG generation runs outside config_mutex_
// ============================================================================

namespace {
// A RAG engine whose buildPrompt() blocks until released, standing in for a slow retrieval.
class BlockingRAG : public RAGInference {
   public:
    using RAGInference::RAGInference;
    std::string buildPrompt(const std::string&, const std::string&) override {
        std::unique_lock<std::mutex> lock(m);
        entered = true;
        cv.notify_all();
        cv.wait(lock, [this] { return released; });
        return "hello world";
    }
    void wait_entered() {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return entered; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(m);
        released = true;
        cv.notify_all();
    }

   private:
    std::mutex m;
    std::condition_variable cv;
    bool entered = false;
    bool released = false;
};
}  // namespace

TEST_F(ChatbotAPITest, SlowRagGenerationDoesNotBlockConfigChanges) {
    std::shared_ptr<EncoderDecoderModel> shared_model(model.get(), [](EncoderDecoderModel*) {});
    auto store = std::make_shared<DocumentStore>(model->copy_encoder());
    auto rag = std::make_shared<BlockingRAG>(shared_model, store);
    api->enableRAG(rag);

    ChatbotAPI::GenerationConfig config;
    auto answer = std::async(std::launch::async,
                             [&] { return call_generate_response("hello", config); });
    rag->wait_entered();

    // Used to wait for the whole RAG generation: generate_response() held config_mutex_.
    ChatbotAPI::GenerationConfig changed;
    changed.temperature = 0.5f;
    auto set = std::async(std::launch::async, [&] { api->set_generation_config(changed); });
    const auto status = set.wait_for(std::chrono::seconds(5));
    rag->release();
    EXPECT_EQ(status, std::future_status::ready)
        << "set_generation_config() waited for the RAG generation";
    EXPECT_NO_THROW(answer.get());
    set.get();
}

// ============================================================================
// TD-231: sessions expire without GET /health
// ============================================================================

TEST_F(ChatbotAPITest, IdleSessionsExpireWithoutAHealthCheck) {
    ASSERT_FALSE(start_session("hello").empty());
    ASSERT_EQ(call_session_count(), 1u);

    // Sessions used to be removed only by handle_health(); nothing here calls it.
    api->set_session_expiry(std::chrono::milliseconds(50), std::chrono::milliseconds(10));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (call_session_count() != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_EQ(call_session_count(), 0u);
}

TEST_F(ChatbotAPITest, ActiveSessionsSurviveTheSweeper) {
    api->set_session_expiry(std::chrono::seconds(30), std::chrono::milliseconds(5));
    ASSERT_FALSE(start_session("hello").empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(call_session_count(), 1u);
}

TEST_F(ChatbotAPITest, DestructorStopsTheSweeperPromptly) {
    // Default sweep interval is a minute; destruction must not wait for it.
    const auto start = std::chrono::steady_clock::now();
    api.reset();
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
}

// ============================================================================
// TD-232: server-issued session ids and the session cap
// ============================================================================

TEST_F(ChatbotAPITest, UnknownSessionIdIsNotFoundAndCreatesNothing) {
    for (const char* body : {R"({"session_id":"client-made","message":"hello"})",
                             R"({"session_id":"client-made","data":"x"})"}) {
        const bool import = std::string(body).find("data") != std::string::npos;
        EXPECT_THROW(import ? call_handle_import_session(body) : call_handle_chat_session(body),
                     std::runtime_error)
            << body;
    }
    // A client-chosen id used to create a session under that name.
    EXPECT_EQ(call_session_count(), 0u);
}

TEST_F(ChatbotAPITest, IssuedSessionIdContinuesTheConversation) {
    const std::string sid = start_session();
    ASSERT_EQ(sid.size(), 32u);
    const std::string reply =
        call_handle_chat_session(R"({"session_id":")" + sid + R"(","message":"again"})");
    EXPECT_EQ(api->parse_json_string(reply, "session_id"), sid);
    EXPECT_EQ(call_session_count(), 1u);
}

TEST_F(ChatbotAPITest, SessionCapEvictsTheLeastRecentlyUsed) {
    ChatbotAPI::GenerationConfig config;
    config.max_length = 2;
    config.strategy = "greedy";
    api->set_generation_config(config);
    api->set_max_sessions(3);

    const std::string a = start_session();
    const std::string b = start_session();
    const std::string c = start_session();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    call_handle_chat_session(R"({"session_id":")" + a + R"(","message":"keep"})");  // a is now newest
    const std::string d = start_session();  // at the cap: evicts b

    EXPECT_EQ(call_session_count(), 3u);
    EXPECT_THROW(call_handle_chat_session(R"({"session_id":")" + b + R"(","message":"hi"})"),
                 std::runtime_error);
    for (const auto& sid : {a, c, d}) {
        EXPECT_NO_THROW(call_handle_export_session(R"({"session_id":")" + sid + "\"}")) << sid;
    }
}

// ============================================================================
// TD-234: batch endpoints use generate_response()
// ============================================================================

namespace {
// Records every query and history, so a test sees which inputs went through RAG and in what order;
// the prompt it returns is the query (TD-486: ChatbotAPI generates from it).
class EchoRAG : public RAGInference {
   public:
    using RAGInference::RAGInference;
    std::string buildPrompt(const std::string& query, const std::string& history) override {
        ++calls;
        std::lock_guard<std::mutex> lock(m);
        queries.push_back(query);
        histories.push_back(history);
        return query;
    }
    std::vector<std::string> queries;
    std::atomic<int> calls{0};
    std::mutex m;
    std::vector<std::string> histories;
};
}  // namespace

TEST_F(ChatbotAPITest, BatchEndpointsGoThroughTheConfiguredRagEngine) {
    std::shared_ptr<EncoderDecoderModel> shared_model(model.get(), [](EncoderDecoderModel*) {});
    auto rag = std::make_shared<EchoRAG>(shared_model,
                                         std::make_shared<DocumentStore>(model->copy_encoder()));
    api->enableRAG(rag);

    const auto batch = nlohmann::json::parse(
        call_handle_batch_chat(R"({"messages":["first","second question","3"]})"));
    ASSERT_TRUE(batch.value("success", false)) << batch.dump();
    EXPECT_EQ(batch["responses"].size(), 3u);
    EXPECT_EQ(rag->queries, (std::vector<std::string>{"first", "second question", "3"}));
    EXPECT_EQ(rag->calls.load(), 3);

    // Batch-session generates from each session's formatted conversation.
    ChatbotAPI::GenerationConfig config;
    const auto with_sessions = api->generate_batch_session_responses({"hello", "there"}, {"", ""}, config);
    ASSERT_TRUE(with_sessions.success) << with_sessions.error;
    ASSERT_EQ(with_sessions.responses.size(), 2u);
    EXPECT_EQ(rag->calls.load(), 5);
    EXPECT_EQ(rag->queries[3], "hello");
    EXPECT_EQ(rag->queries[4], "there");
}

// ============================================================================
// TD-235: batch size limit
// ============================================================================

TEST_F(ChatbotAPITest, OversizedBatchIsRejectedBeforeAnyGeneration) {
    std::shared_ptr<EncoderDecoderModel> shared_model(model.get(), [](EncoderDecoderModel*) {});
    auto rag = std::make_shared<EchoRAG>(shared_model,
                                         std::make_shared<DocumentStore>(model->copy_encoder()));
    api->enableRAG(rag);
    api->set_max_batch_size(3);

    nlohmann::json four = {{"messages", {"a", "b", "c", "d"}}};
    for (bool sessions : {false, true}) {
        try {
            sessions ? call_handle_batch_chat_session(four.dump()) : call_handle_batch_chat(four.dump());
            FAIL() << "a batch of 4 over a limit of 3 was accepted";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("the limit is 3"), std::string::npos) << e.what();
        }
    }
    EXPECT_EQ(rag->calls.load(), 0);
    EXPECT_EQ(call_session_count(), 0u);

    const auto three = nlohmann::json::parse(call_handle_batch_chat(R"({"messages":["a","b","c"]})"));
    EXPECT_TRUE(three.value("success", false)) << three.dump();
}

// ============================================================================
// TD-236: errors carry their HTTP status
// ============================================================================

TEST_F(ChatbotAPITest, MissingFieldsAndBadImportsAreBadRequests) {
    for (const char* body : {"{}", R"({"message":""})"}) {
        EXPECT_THROW(call_handle_chat_session(body), std::invalid_argument) << body;
    }
    EXPECT_THROW(call_handle_batch_chat(R"({"messages":[]})"), std::invalid_argument);
    EXPECT_THROW(call_handle_batch_chat_session("{}"), std::invalid_argument);
    // Data that doesn't parse as a conversation (no "---" line).
    EXPECT_THROW(call_handle_import_session(R"({"data":"not a conversation"})"), std::invalid_argument);
}

namespace {
class ThrowingRAG : public RAGInference {
   public:
    using RAGInference::RAGInference;
    std::string buildPrompt(const std::string&, const std::string&) override { throw std::runtime_error("generation broke"); }
};
}  // namespace

TEST_F(ChatbotAPITest, FailedBatchGenerationIsAServerError) {
    std::shared_ptr<EncoderDecoderModel> shared_model(model.get(), [](EncoderDecoderModel*) {});
    api->enableRAG(std::make_shared<ThrowingRAG>(
        shared_model, std::make_shared<DocumentStore>(model->copy_encoder())));
    // A runtime_error (HTTP 500), not a BadRequest; it used to be "success":false with HTTP 200.
    for (bool sessions : {false, true}) {
        try {
            sessions ? call_handle_batch_chat_session(R"({"messages":["a"]})")
                     : call_handle_batch_chat(R"({"messages":["a"]})");
            FAIL() << "a failed generation was returned as a normal reply";
        } catch (const std::invalid_argument& e) {
            FAIL() << "not a client error: " << e.what();
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("generation broke"), std::string::npos) << e.what();
        }
    }
}

// ============================================================================
// TD-237: start() and stop() on different threads
// ============================================================================

// chatbot_api_server runs start() on a server thread and stop() from the main thread. running_ was a
// plain bool, a data race TSan reports; it's atomic now.
TEST_F(ChatbotAPITest, StopFromAnotherThreadEndsStart) {
    api = std::make_unique<ChatbotAPI>(model.get(), tokenizer.get(), /*port=*/0, 30);
    std::atomic<bool> returned{false};
    std::thread server([&] {
        api->start();
        returned = true;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!api->is_running() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_TRUE(api->is_running());
    EXPECT_FALSE(api->start()) << "a second start() while running must refuse";
    std::this_thread::sleep_for(std::chrono::milliseconds(50));  // let listen() get going
    api->stop();
    server.join();
    EXPECT_TRUE(returned.load());
    EXPECT_FALSE(api->is_running());
}

// ============================================================================
// TD-238: internal errors don't reach the client
// ============================================================================

namespace {
int free_local_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    ::close(fd);
    return ntohs(addr.sin_port);
}

class LeakyRAG : public RAGInference {
   public:
    using RAGInference::RAGInference;
    std::string buildPrompt(const std::string&, const std::string&) override {
        throw std::runtime_error("cannot open /srv/adai/secret_model.bin");
    }
};
}  // namespace

TEST_F(ChatbotAPITest, InternalErrorsAreLoggedNotSentToTheClient) {
    const int port = free_local_port();
    api = std::make_unique<ChatbotAPI>(model.get(), tokenizer.get(), port, 30);
    std::shared_ptr<EncoderDecoderModel> shared_model(model.get(), [](EncoderDecoderModel*) {});
    api->enableRAG(std::make_shared<LeakyRAG>(shared_model,
                                              std::make_shared<DocumentStore>(model->copy_encoder())));

    std::ostringstream log;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(log);
    adai::Logger::get()->sinks().push_back(sink);

    std::thread server([&] { api->start(); });
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(std::chrono::milliseconds(200));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!client.Get("/health") && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::vector<httplib::Result> replies;
    replies.push_back(client.Post("/chat", R"({"message":"hi"})", "application/json"));
    replies.push_back(client.Post("/chat/batch", R"({"messages":["hi"]})", "application/json"));
    replies.push_back(client.Post("/chat/session", R"({"message":"hi"})", "application/json"));
    api->stop();
    server.join();
    auto& sinks = adai::Logger::get()->sinks();
    sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());

    for (const auto& r : replies) {
        ASSERT_TRUE(r);
        EXPECT_EQ(r->status, 500) << r->body;
        EXPECT_EQ(r->body.find("secret"), std::string::npos) << "internal detail leaked: " << r->body;
        const auto body = nlohmann::json::parse(r->body);
        EXPECT_EQ(body.value("error", ""), "Internal server error");
        const std::string id = body.value("request_id", "");
        EXPECT_EQ(id.size(), 16u);
        // The log has the detail under the same id, for the operator.
        EXPECT_NE(log.str().find("request " + id + "): "), std::string::npos) << log.str();
    }
    EXPECT_NE(log.str().find("secret_model.bin"), std::string::npos);
}

// ============================================================================
// TD-485: RAG gets the latest message, not the whole conversation
// ============================================================================

TEST_F(ChatbotAPITest, SessionTurnsGiveRagTheLatestMessageAndTheHistory) {
    std::shared_ptr<EncoderDecoderModel> shared_model(model.get(), [](EncoderDecoderModel*) {});
    auto rag = std::make_shared<EchoRAG>(shared_model,
                                         std::make_shared<DocumentStore>(model->copy_encoder()));
    api->enableRAG(rag);

    const std::string sid = start_session("first question");
    const auto reply = nlohmann::json::parse(
        call_handle_chat_session(R"({"session_id":")" + sid + R"(","message":"second question"})"));
    // The query used to be the whole formatted conversation ("User: first question\n...").
    ASSERT_TRUE(reply.value("success", false)) << reply.dump();
    ASSERT_EQ(rag->queries.size(), 2u);
    EXPECT_EQ(rag->queries[1], "second question");
    EXPECT_EQ(rag->histories[0], "") << "a first turn has no earlier conversation";
    EXPECT_NE(rag->histories[1].find("User: first question"), std::string::npos) << rag->histories[1];
    EXPECT_NE(rag->histories[1].find("Assistant: "), std::string::npos);
    EXPECT_EQ(rag->histories[1].find("second question"), std::string::npos)
        << "the latest message is the question, not part of the history";
}

// ============================================================================
// TD-486: RAG requests generate through ChatbotAPI's own path
// ============================================================================

namespace {
class FixedPromptRAG : public RAGInference {
   public:
    using RAGInference::RAGInference;
    std::string buildPrompt(const std::string&, const std::string&) override { return "hello world"; }
};
}  // namespace

TEST_F(ChatbotAPITest, RagHonoursTheRequestStrategyAndConfig) {
    std::shared_ptr<EncoderDecoderModel> shared_model(model.get(), [](EncoderDecoderModel*) {});
    auto rag = std::make_shared<FixedPromptRAG>(shared_model,
                                                std::make_shared<DocumentStore>(model->copy_encoder()));
    ChatbotAPI::GenerationConfig greedy;
    greedy.strategy = "greedy";
    greedy.max_length = 6;

    // Without RAG, greedy generation from the prompt RAG would build.
    const std::string expected = call_generate_response("hello world", greedy);

    // RAG used to generate itself with its own startup settings and the default sampling, so its
    // greedy answers varied run to run and ignored max_length.
    api->enableRAG(rag);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(call_generate_response("anything", greedy), expected);

    // A changed config (as a SIGHUP reload makes) reaches RAG requests.
    ChatbotAPI::GenerationConfig shorter = greedy;
    shorter.max_length = 2;
    api->enableRAG(nullptr);
    const std::string expected_short = call_generate_response("hello world", shorter);
    api->enableRAG(rag);
    EXPECT_EQ(call_generate_response("anything", shorter), expected_short);
}

#ifdef ADAI_ENABLE_GPU
// TD-486: on a GPU server a RAG request takes the GPU decode like any other (it used to generate on
// the CPU inside RAGInference). Greedy through RAG equals greedy from the same prompt without it.
TEST_F(ChatbotAPITest, RagRequestsTakeTheGpuDecode) {
    if (!adai::gpu::GPUManager::is_available() && !adai::gpu::GPUManager::initialize()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    std::shared_ptr<EncoderDecoderModel> shared_model(model.get(), [](EncoderDecoderModel*) {});
    auto rag = std::make_shared<FixedPromptRAG>(shared_model,
                                                std::make_shared<DocumentStore>(model->copy_encoder()));
    ChatbotAPI::GenerationConfig greedy;
    greedy.strategy = "greedy";
    greedy.max_length = 6;
    const std::string expected = call_generate_response("hello world", greedy);
    api->enableRAG(rag);
    EXPECT_EQ(call_generate_response("anything", greedy), expected);
}
#endif  // ADAI_ENABLE_GPU
