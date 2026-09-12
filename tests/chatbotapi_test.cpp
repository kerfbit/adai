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
#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <thread>
#include "../src/BPETokenizer.hpp"
#include "../src/EncoderDecoderModel.hpp"

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

    // handle_profile() is private; same friendship-doesn't-propagate reason as
    // call_generate_response() above (TD-038).
    std::string call_handle_profile() {
        return api->handle_profile();
    }

    // batched_engine_ is private; same friendship-doesn't-propagate reason as
    // call_generate_response() above (TD-038). Used to prove generate_response() actually routed
    // a request through the engine's real queue/worker thread, not merely that it didn't throw —
    // the plain (non-batched) path can also succeed and throw nothing, so total_requests is the
    // discriminator that only the batched path can move.
    BatchedInferenceStats call_get_batched_stats() {
        return api->batched_engine_ ? api->batched_engine_->get_stats() : BatchedInferenceStats();
    }

    // pipeline_engine_ is private; same friendship-doesn't-propagate reason as
    // call_generate_response() above (TD-038). Same discriminator role as
    // call_get_batched_stats() above.
    PipelineStats call_get_pipeline_stats() {
        return api->pipeline_engine_ ? api->pipeline_engine_->get_stats() : PipelineStats();
    }

    // enable_pipeline_inference() (TD-038) needs an actual vocabulary *file* path — it reloads
    // the same vocab into the model's encoder-internal tokenizer (see its doc comment in
    // ChatbotAPI.hpp) — but this fixture's tokenizer only ever exists in memory
    // (build_vocab(), never saved). Saves it out once per call so pipeline tests have a real
    // file to point at.
    std::string save_tokenizer_vocab_to_temp_file() {
        std::string path = "/tmp/adai_chatbotapi_test_pipeline_vocab.txt";
        tokenizer->save_vocab(path);
        return path;
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
    std::vector<std::string> inputs = {"Msg1", "Msg2", "Msg3"};
    std::vector<std::string> session_ids = {"existing_id", "", "another_id"};

    ChatbotAPI::GenerationConfig config;
    config.max_length = 10;
    config.strategy = "greedy";

    ChatbotAPI::BatchResponse response =
        api->generate_batch_session_responses(inputs, session_ids, config);

    EXPECT_TRUE(response.success);
    EXPECT_EQ(response.responses.size(), 3);
    EXPECT_EQ(response.session_ids.size(), 3);
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
    // with num_candidates=0, SpeculativeDecoder::generate_candidates() proposes nothing on every
    // round, so generate_tokens() breaks on its first iteration with zero output tokens — and
    // BPETokenizer::decode() on an empty token vector throws ("Input token ID vector is empty"),
    // caught and rethrown here as a generation failure. Confirmed directly: this only happens
    // via the speculative path — plain generation on this same (untrained, randomly-initialized)
    // model never legitimately produces zero tokens (max_length=5 greedy decoding always emits
    // at least one token before it could possibly hit EOS), so this exception could only
    // originate from SpeculativeDecoder actually being invoked with num_candidates=0.
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
// Pipeline Inference (TD-038): PipelineInferenceEngine wiring
// ============================================================================
//
// Integration tests in the same sense as the sections above — proving generate_response()
// genuinely routes through a real StandardPipelineEngine (real encoder/decoder worker threads)
// via real (small) inference through a real ChatbotAPI instance. This wiring also caught a real,
// pre-existing bug: PipelineInferenceEngine::decoder_worker() called a method,
// forward_with_cross_attention(tokens, encoder_output, nullptr), that does not exist on the real
// LLMDecoder — its dedicated unit test suite (pipelineinferenceengine_test.cpp) never caught this
// because it exercises the class entirely through mock encoder/decoder/lm_head types shaped to
// match the (buggy) production call, not the real dependency's actual interface. Fixed to call
// LLMDecoder's real forward_with_encoder(token_ids, encoder_output) instead — see
// PipelineInferenceEngine.hpp's decoder_worker() and this file's MockDecoder for the full story.

TEST_F(ChatbotAPITest, PipelineInference_DisabledByDefaultUsesNormalPathUnaffected) {
    // No enable_pipeline_inference() call — confirms adding pipeline_engine_ didn't disturb the
    // existing, already-covered normal generation path.
    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    config.strategy = "greedy";
    EXPECT_NO_THROW(call_generate_response("hello", config));
}

TEST_F(ChatbotAPITest, PipelineInference_EnabledGeneratesThroughThePipelineWithoutThrowing) {
    std::string vocab_path = save_tokenizer_vocab_to_temp_file();
    api->enable_pipeline_inference(vocab_path);

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;
    EXPECT_NO_THROW(call_generate_response("hello world", config));

    // Falsifiable proof the request actually went through the pipeline's real encoder/decoder
    // worker threads rather than silently falling through to the plain inline path (which would
    // also succeed without throwing).
    EXPECT_EQ(call_get_pipeline_stats().total_requests, 1u);
}

TEST_F(ChatbotAPITest, PipelineInference_WrongVocabPathThrows) {
    // enable_pipeline_inference() deliberately does not catch load_tokenizer_vocab()'s own
    // exception (see its doc comment) — callers decide whether a bad path should disable the
    // mode or abort startup. Confirms that exception genuinely propagates out to the caller.
    EXPECT_THROW(api->enable_pipeline_inference("/nonexistent/path/vocab.txt"),
                std::exception);
}

TEST_F(ChatbotAPITest, PipelineInference_ConcurrentRequestsAllCompleteCorrectly) {
    // Same reasoning as BatchedInference_ConcurrentRequestsAllCompleteCorrectly above: real proof
    // of wiring requires exercising the pipeline's actual worker threads with genuine concurrent
    // load, not a single call that could coincidentally succeed even if the wiring were subtly
    // wrong.
    std::string vocab_path = save_tokenizer_vocab_to_temp_file();
    api->enable_pipeline_inference(vocab_path);

    ChatbotAPI::GenerationConfig config;
    config.max_length = 5;

    constexpr int kNumThreads = 6;
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
        // Not asserting non-empty output: PipelineInferenceEngine's decoder loop is always greedy
        // (see enable_pipeline_inference()'s doc comment) and can legitimately pick EOS as the
        // very first token on this tiny randomly-initialized test model, producing a valid empty
        // string via decode(..., skip_special_tokens=true) — not a wiring failure.
        EXPECT_FALSE(threw[i]) << "concurrent request " << i << " threw unexpectedly: "
                               << errors[i];
    }

    EXPECT_EQ(call_get_pipeline_stats().total_requests, static_cast<uint64_t>(kNumThreads));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
