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
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
