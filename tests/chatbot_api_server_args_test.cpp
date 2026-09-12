// Tests for src/ChatbotApiServerArgs.{hpp,cpp} — chatbot_api_server's argv parsing and
// required-config validation (TD-035), extracted from ChatbotAPIServer.cpp so it's testable
// without loading a real tokenizer/model or starting a real HTTP server. ChatbotAPI's own
// request-handling behavior is already covered by chatbotapiTests.

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "ChatbotApiServerArgs.hpp"

using adai::apply_chatbot_api_server_args;
using adai::extract_config_path_arg;
using adai::ServiceConfig;
using adai::validate_chatbot_api_server_config;

namespace {
std::vector<char*> make_argv(std::vector<std::string>& storage) {
    std::vector<char*> argv;
    for (auto& s : storage)
        argv.push_back(s.data());
    return argv;
}
}  // namespace

TEST(ExtractConfigPathArg, ReturnsNulloptWhenAbsent) {
    std::vector<std::string> raw = {"chatbot_api_server", "--port", "8080"};
    auto argv = make_argv(raw);
    EXPECT_FALSE(extract_config_path_arg(static_cast<int>(argv.size()), argv.data()).has_value());
}

TEST(ExtractConfigPathArg, ReturnsValueWhenPresent) {
    std::vector<std::string> raw = {"chatbot_api_server", "--config", "/tmp/c.conf"};
    auto argv = make_argv(raw);
    auto v = extract_config_path_arg(static_cast<int>(argv.size()), argv.data());
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, "/tmp/c.conf");
}

TEST(ExtractConfigPathArg, MissingValueIsIgnoredNotCrashed) {
    std::vector<std::string> raw = {"chatbot_api_server", "--config"};
    auto argv = make_argv(raw);
    EXPECT_FALSE(extract_config_path_arg(static_cast<int>(argv.size()), argv.data()).has_value());
}

class ApplyChatbotApiServerArgsTest : public ::testing::Test {
   protected:
    ServiceConfig config;
};

TEST_F(ApplyChatbotApiServerArgsTest, NoArgsLeavesConfigUntouchedAndNoError) {
    std::vector<std::string> raw = {"chatbot_api_server"};
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    EXPECT_FALSE(r.help);
    EXPECT_FALSE(r.error);
}

TEST_F(ApplyChatbotApiServerArgsTest, HelpFlagShortCircuitsBeforeAnyOtherProcessing) {
    std::vector<std::string> raw = {"chatbot_api_server", "--port", "9999", "--help"};
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    EXPECT_TRUE(r.help);
    EXPECT_FALSE(r.error);
}

TEST_F(ApplyChatbotApiServerArgsTest, ShortDashHAlsoTriggersHelp) {
    std::vector<std::string> raw = {"chatbot_api_server", "-h"};
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    EXPECT_TRUE(r.help);
}

TEST_F(ApplyChatbotApiServerArgsTest, UnknownArgumentIsAnError) {
    std::vector<std::string> raw = {"chatbot_api_server", "--bogus"};
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    EXPECT_TRUE(r.error);
    EXPECT_NE(r.error_message.find("--bogus"), std::string::npos);
}

TEST_F(ApplyChatbotApiServerArgsTest, ConfigFlagIsSkippedNotTreatedAsUnknown) {
    // --config's value was already consumed by extract_config_path_arg(); this pass must skip
    // over it (both the flag and its value) rather than erroring on either.
    std::vector<std::string> raw = {"chatbot_api_server", "--config", "/tmp/c.conf", "--port",
                                    "9090"};
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    EXPECT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(config.port, 9090);
}

TEST_F(ApplyChatbotApiServerArgsTest, EveryStringAndIntFlagIsApplied) {
    std::vector<std::string> raw = {
        "chatbot_api_server", "--model",      "/m.bin",  "--vocab",       "/v.txt",
        "--port",             "9090",         "--timeout", "15",          "--log-level",
        "DEBUG",              "--d-model",    "128",     "--num-heads",   "4",
        "--d-ff",             "512",          "--enc-layers", "3",        "--dec-layers",
        "3",                  "--max-seq-len", "256",    "--max-gen-len", "50",
        "--strategy",         "greedy",
    };
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(config.model_path, "/m.bin");
    EXPECT_EQ(config.vocab_path, "/v.txt");
    EXPECT_EQ(config.port, 9090);
    EXPECT_EQ(config.session_timeout, 15);
    EXPECT_EQ(config.log_level, "DEBUG");
    EXPECT_EQ(config.d_model, 128u);
    EXPECT_EQ(config.num_heads, 4u);
    EXPECT_EQ(config.d_ff, 512u);
    EXPECT_EQ(config.num_encoder_layers, 3u);
    EXPECT_EQ(config.num_decoder_layers, 3u);
    EXPECT_EQ(config.max_seq_length, 256u);
    EXPECT_EQ(config.max_gen_length, 50u);
    EXPECT_EQ(config.strategy, "greedy");
}

TEST_F(ApplyChatbotApiServerArgsTest, DraftModelFlagsAreApplied) {
    // TD-038: --draft-model / --speculative-candidates enable speculative decoding.
    std::vector<std::string> raw = {"chatbot_api_server", "--draft-model", "/draft.bin",
                                    "--speculative-candidates", "6"};
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(config.draft_model_path, "/draft.bin");
    EXPECT_EQ(config.speculative_num_candidates, 6);
}

TEST_F(ApplyChatbotApiServerArgsTest, DraftModelPathDefaultsEmpty) {
    std::vector<std::string> raw = {"chatbot_api_server"};
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error);
    EXPECT_TRUE(config.draft_model_path.empty());
}

TEST_F(ApplyChatbotApiServerArgsTest, FloatFlagsAreApplied) {
    std::vector<std::string> raw = {"chatbot_api_server", "--temperature", "0.7", "--top-p",
                                    "0.85"};
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error);
    EXPECT_FLOAT_EQ(config.temperature, 0.7f);
    EXPECT_FLOAT_EQ(config.top_p, 0.85f);
}

TEST_F(ApplyChatbotApiServerArgsTest, ProfileFlagDefaultsFalseAndCanBeEnabled) {
    // TD-038: --profile enables ChatbotAPI::enable_profiling() / GET /admin/profile — a runtime
    // toggle on the result, not part of ServiceConfig (see ChatbotApiServerArgs.hpp).
    std::vector<std::string> raw_off = {"chatbot_api_server"};
    auto argv_off = make_argv(raw_off);
    auto r_off =
        apply_chatbot_api_server_args(static_cast<int>(argv_off.size()), argv_off.data(), config);
    EXPECT_FALSE(r_off.profile);

    std::vector<std::string> raw_on = {"chatbot_api_server", "--profile"};
    auto argv_on = make_argv(raw_on);
    auto r_on =
        apply_chatbot_api_server_args(static_cast<int>(argv_on.size()), argv_on.data(), config);
    ASSERT_FALSE(r_on.error);
    EXPECT_TRUE(r_on.profile);
}

TEST_F(ApplyChatbotApiServerArgsTest, BatchedInferenceFlagDefaultsFalseAndCanBeEnabled) {
    // TD-038: --batched-inference enables ChatbotAPI::enable_batched_inference() — a runtime
    // toggle on the result, not part of ServiceConfig, same reasoning as --profile above.
    std::vector<std::string> raw_off = {"chatbot_api_server"};
    auto argv_off = make_argv(raw_off);
    auto r_off =
        apply_chatbot_api_server_args(static_cast<int>(argv_off.size()), argv_off.data(), config);
    EXPECT_FALSE(r_off.batched_inference);
    EXPECT_EQ(r_off.batch_timeout_ms, 50);

    std::vector<std::string> raw_on = {"chatbot_api_server", "--batched-inference"};
    auto argv_on = make_argv(raw_on);
    auto r_on =
        apply_chatbot_api_server_args(static_cast<int>(argv_on.size()), argv_on.data(), config);
    ASSERT_FALSE(r_on.error);
    EXPECT_TRUE(r_on.batched_inference);
}

TEST_F(ApplyChatbotApiServerArgsTest, BatchTimeoutMsFlagIsApplied) {
    std::vector<std::string> raw = {"chatbot_api_server", "--batched-inference",
                                    "--batch-timeout-ms", "5"};
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_TRUE(r.batched_inference);
    EXPECT_EQ(r.batch_timeout_ms, 5);
}

TEST_F(ApplyChatbotApiServerArgsTest, PipelineInferenceFlagDefaultsFalseAndCanBeEnabled) {
    // TD-038: --pipeline-inference enables ChatbotAPI::enable_pipeline_inference() — a runtime
    // toggle on the result, not part of ServiceConfig, same reasoning as --profile above.
    std::vector<std::string> raw_off = {"chatbot_api_server"};
    auto argv_off = make_argv(raw_off);
    auto r_off =
        apply_chatbot_api_server_args(static_cast<int>(argv_off.size()), argv_off.data(), config);
    EXPECT_FALSE(r_off.pipeline_inference);

    std::vector<std::string> raw_on = {"chatbot_api_server", "--pipeline-inference"};
    auto argv_on = make_argv(raw_on);
    auto r_on =
        apply_chatbot_api_server_args(static_cast<int>(argv_on.size()), argv_on.data(), config);
    ASSERT_FALSE(r_on.error);
    EXPECT_TRUE(r_on.pipeline_inference);
}

TEST_F(ApplyChatbotApiServerArgsTest, IntegratedInferenceFlagDefaultsFalseAndCanBeEnabled) {
    // TD-038: --integrated-inference enables ChatbotAPI::enable_integrated_inference() — a
    // runtime toggle on the result, not part of ServiceConfig, same reasoning as --profile above.
    std::vector<std::string> raw_off = {"chatbot_api_server"};
    auto argv_off = make_argv(raw_off);
    auto r_off =
        apply_chatbot_api_server_args(static_cast<int>(argv_off.size()), argv_off.data(), config);
    EXPECT_FALSE(r_off.integrated_inference);

    std::vector<std::string> raw_on = {"chatbot_api_server", "--integrated-inference"};
    auto argv_on = make_argv(raw_on);
    auto r_on =
        apply_chatbot_api_server_args(static_cast<int>(argv_on.size()), argv_on.data(), config);
    ASSERT_FALSE(r_on.error);
    EXPECT_TRUE(r_on.integrated_inference);
}

TEST_F(ApplyChatbotApiServerArgsTest, CliValueOverridesWhateverConfigAlreadyHad) {
    config.port = 8080;
    std::vector<std::string> raw = {"chatbot_api_server", "--port", "1234"};
    auto argv = make_argv(raw);
    auto r = apply_chatbot_api_server_args(static_cast<int>(argv.size()), argv.data(), config);
    ASSERT_FALSE(r.error);
    EXPECT_EQ(config.port, 1234);
}

TEST(ValidateChatbotApiServerConfig, EmptyVocabPathIsAnError) {
    ServiceConfig config;
    config.vocab_path = "";
    auto err = validate_chatbot_api_server_config(config);
    ASSERT_TRUE(err.has_value());
    EXPECT_NE(err->find("Vocabulary path"), std::string::npos);
}

TEST(ValidateChatbotApiServerConfig, NonEmptyVocabPathPasses) {
    ServiceConfig config;
    config.vocab_path = "/vocab.txt";
    EXPECT_FALSE(validate_chatbot_api_server_config(config).has_value());
}
