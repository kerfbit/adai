// Tests for src/MnsCliCommands.{hpp,cpp} — mns_cli's per-command argument parsing and JSON-body
// construction (TD-035), extracted from MnsCliTool.cpp so it's testable without a real mns_server
// to talk to. Each build_*_request() is pure: given a command's argv slice, it returns the
// method/path/body mns_cli would send (or an error), with no network I/O at all — actually
// sending the request stays in MnsCliTool.cpp, unchanged and untested here (that's exercised
// live, end to end, by real mns_server usage and MNSLiveTests on the server side).

#include <gtest/gtest.h>
#include "MnsCliCommands.hpp"

using adai::build_delete_request;
using adai::build_get_request;
using adai::build_health_request;
using adai::build_link_world_model_request;
using adai::build_list_request;
using adai::build_promote_request;
using adai::build_register_request;
using adai::build_resolve_request;
using adai::build_resolve_role_request;
using adai::build_roles_request;
using adai::build_set_candidate_request;
using adai::build_set_training_request;
using adai::build_update_run_group_request;
using adai::HttpMethod;
using adai::json_escape;
using adai::parse_url;
using adai::ServiceConfig;

TEST(ParseUrl, DefaultsWhenGivenBareHost) {
    auto u = parse_url("localhost");
    EXPECT_EQ(u.host, "localhost");
    EXPECT_EQ(u.port, 8083) << "default port should be untouched when none is specified";
}

TEST(ParseUrl, StripsHttpAndParsesPort) {
    auto u = parse_url("http://example.com:9090");
    EXPECT_EQ(u.host, "example.com");
    EXPECT_EQ(u.port, 9090);
}

TEST(ParseUrl, StripsHttpsAndTrailingPath) {
    auto u = parse_url("https://example.com:1234/some/path");
    EXPECT_EQ(u.host, "example.com");
    EXPECT_EQ(u.port, 1234);
}

TEST(ParseUrl, BareHostNoPortNoScheme) {
    auto u = parse_url("myhost");
    EXPECT_EQ(u.host, "myhost");
    EXPECT_EQ(u.port, 8083);
}

TEST(JsonEscape, EscapesQuotesBackslashesAndControlChars) {
    EXPECT_EQ(json_escape("a\"b"), "a\\\"b");
    EXPECT_EQ(json_escape("a\\b"), "a\\\\b");
    EXPECT_EQ(json_escape("a\nb"), "a\\nb");
    EXPECT_EQ(json_escape("a\tb"), "a\\tb");
    EXPECT_EQ(json_escape("plain"), "plain");
}

TEST(BuildListRequest, NoFiltersIsBarePath) {
    auto r = build_list_request({});
    EXPECT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Get);
    EXPECT_EQ(r.path, "/models");
}

TEST(BuildListRequest, AppliesAllThreeFiltersWithCorrectSeparators) {
    auto r = build_list_request({"--state", "training", "--role", "chatbot", "--limit", "5"});
    EXPECT_EQ(r.path, "/models?state=training&role=chatbot&limit=5");
}

TEST(BuildListRequest, LimitZeroOrUnsetIsOmitted) {
    auto r = build_list_request({"--limit", "0"});
    EXPECT_EQ(r.path, "/models");
}

// TD-196
TEST(BuildListRequest, AppliesKindFilter) {
    auto r = build_list_request({"--kind", "world_model"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.path, "/models?kind=world_model");
}

// TD-196
TEST(BuildListRequest, AllFourFiltersCombineWithCorrectSeparators) {
    auto r = build_list_request(
        {"--state", "training", "--role", "chatbot", "--kind", "encoder", "--limit", "5"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.path, "/models?state=training&role=chatbot&kind=encoder&limit=5");
}

TEST(BuildGetRequest, MissingNameIsAnError) {
    auto r = build_get_request({});
    EXPECT_TRUE(r.error);
}

TEST(BuildGetRequest, BuildsCorrectPath) {
    auto r = build_get_request({"my-model"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Get);
    EXPECT_EQ(r.path, "/models/my-model");
}

TEST(BuildRegisterRequest, RequiresNameAndRole) {
    ServiceConfig cfg;
    EXPECT_TRUE(build_register_request({"only-name"}, cfg).error);
    EXPECT_TRUE(build_register_request({}, cfg).error);
}

TEST(BuildRegisterRequest, UsesConfigDefaultsWhenNoOverridesGiven) {
    ServiceConfig cfg;
    cfg.d_model = 128;
    cfg.num_heads = 4;
    cfg.d_ff = 512;
    cfg.num_encoder_layers = 2;
    cfg.num_decoder_layers = 2;
    cfg.max_seq_length = 64;
    cfg.run_group = "default-group";

    auto r = build_register_request({"my-model", "chatbot"}, cfg);
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Post);
    EXPECT_EQ(r.path, "/models");
    EXPECT_NE(r.body.find("\"model_name\":\"my-model\""), std::string::npos);
    EXPECT_NE(r.body.find("\"role\":\"chatbot\""), std::string::npos);
    EXPECT_NE(r.body.find("\"run_group\":\"default-group\""), std::string::npos);
    EXPECT_NE(r.body.find("\"d_model\":128"), std::string::npos);
    EXPECT_NE(r.body.find("\"num_heads\":4"), std::string::npos);
}

TEST(BuildRegisterRequest, CliArchFlagsOverrideConfigDefaults) {
    ServiceConfig cfg;
    cfg.d_model = 128;
    auto r = build_register_request({"m", "role", "--d-model", "256"}, cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"d_model\":256"), std::string::npos);
    EXPECT_EQ(r.body.find("\"d_model\":128"), std::string::npos);
}

TEST(BuildRegisterRequest, RunGroupFlagOverridesConfig) {
    ServiceConfig cfg;
    cfg.run_group = "from-config";
    auto r = build_register_request({"m", "role", "--run-group", "from-cli"}, cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"run_group\":\"from-cli\""), std::string::npos);
}

TEST(BuildRegisterRequest, TagsAreIncludedWhenPresentAndOmittedWhenNot) {
    ServiceConfig cfg;
    auto with_tags = build_register_request({"m", "role", "--tag", "env=prod"}, cfg);
    ASSERT_FALSE(with_tags.error);
    EXPECT_NE(with_tags.body.find("\"tags\":{\"env\":\"prod\"}"), std::string::npos);

    auto without_tags = build_register_request({"m", "role"}, cfg);
    ASSERT_FALSE(without_tags.error);
    EXPECT_EQ(without_tags.body.find("\"tags\""), std::string::npos);
}

TEST(BuildRegisterRequest, NameAndRoleAreJsonEscaped) {
    ServiceConfig cfg;
    auto r = build_register_request({"weird\"name", "role"}, cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("weird\\\"name"), std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────
// TD-196: --kind / --num-layers / --encoder / --decoder / --sigreg-* flags
// ─────────────────────────────────────────────────────────────────────────

TEST(BuildRegisterRequest, KindDefaultsToChatbotWhenNotGiven) {
    ServiceConfig cfg;
    auto r = build_register_request({"m", "role"}, cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"kind\":\"chatbot\""), std::string::npos);
}

TEST(BuildRegisterRequest, InvalidKindIsAnError) {
    ServiceConfig cfg;
    EXPECT_TRUE(build_register_request({"m", "role", "--kind", "bogus"}, cfg).error);
}

TEST(BuildRegisterRequest, EncoderKindNumLayersSetsOwnFieldAndZeroesDecoderLayers) {
    ServiceConfig cfg;
    cfg.num_encoder_layers = 6;
    cfg.num_decoder_layers = 6;  // config defaults, would collide with encoder's own-field rule
                                 // if --num-layers didn't zero this out.
    auto r = build_register_request({"m", "role", "--kind", "encoder", "--num-layers", "3"}, cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"kind\":\"encoder\""), std::string::npos);
    EXPECT_NE(r.body.find("\"num_encoder_layers\":3"), std::string::npos);
    EXPECT_NE(r.body.find("\"num_decoder_layers\":0"), std::string::npos);
}

TEST(BuildRegisterRequest, DecoderKindNumLayersSetsOwnFieldAndZeroesEncoderLayers) {
    ServiceConfig cfg;
    cfg.num_encoder_layers = 6;
    cfg.num_decoder_layers = 6;
    auto r = build_register_request({"m", "role", "--kind", "decoder", "--num-layers", "4"}, cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"num_decoder_layers\":4"), std::string::npos);
    EXPECT_NE(r.body.find("\"num_encoder_layers\":0"), std::string::npos);
}

TEST(BuildRegisterRequest, WorldModelKindNumLayersUsesEncoderLayersField) {
    ServiceConfig cfg;
    auto r =
        build_register_request({"m", "role", "--kind", "world_model", "--num-layers", "5"}, cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"num_encoder_layers\":5"), std::string::npos);
    EXPECT_NE(r.body.find("\"num_decoder_layers\":0"), std::string::npos);
}

TEST(BuildRegisterRequest, WorldModelKindIncludesSigregParamsInConnection) {
    ServiceConfig cfg;
    auto r = build_register_request({"m", "role", "--kind", "world_model", "--num-layers", "5",
                                     "--sigreg-lambda", "2.5", "--sigreg-num-sketches", "32"},
                                    cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"connection\":{"), std::string::npos);
    EXPECT_NE(r.body.find("\"sigreg_lambda\":2.5"), std::string::npos);
    EXPECT_NE(r.body.find("\"sigreg_num_sketches\":32"), std::string::npos);
}

TEST(BuildRegisterRequest, EncoderOrDecoderFlagsRejectedForNonChatbotKind) {
    ServiceConfig cfg;
    EXPECT_TRUE(build_register_request(
                    {"m", "role", "--kind", "encoder", "--encoder", "e", "--decoder", "d"}, cfg)
                    .error);
}

TEST(BuildRegisterRequest, EncoderDecoderFlagsProduceConnectionObjectAndOmitInlineArch) {
    ServiceConfig cfg;
    auto r = build_register_request({"m", "role", "--encoder", "my-enc", "--decoder", "my-dec"},
                                    cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"kind\":\"chatbot\""), std::string::npos);
    EXPECT_NE(r.body.find("\"connection\":{\"encoder_name\":\"my-enc\""), std::string::npos);
    EXPECT_NE(r.body.find("\"decoder_name\":\"my-dec\""), std::string::npos);
    EXPECT_EQ(r.body.find("\"arch\""), std::string::npos)
        << "a new-style linked chatbot must not send inline architecture at all";
}

TEST(BuildRegisterRequest, EncoderDecoderFlagsRejectCombinationWithLegacyArchFlags) {
    ServiceConfig cfg;
    auto r = build_register_request(
        {"m", "role", "--encoder", "my-enc", "--decoder", "my-dec", "--d-model", "999"}, cfg);
    EXPECT_TRUE(r.error);
}

TEST(BuildRegisterRequest, OnlyEncoderFlagWithoutDecoderStillBuildsRequest) {
    // build_register_request itself doesn't enforce the both-or-neither rule (that's
    // handle_register()'s own job, server-side) — it should still build a request the server can
    // reject; it must not silently drop the one flag that was given.
    ServiceConfig cfg;
    auto r = build_register_request({"m", "role", "--encoder", "my-enc"}, cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"encoder_name\":\"my-enc\""), std::string::npos);
    EXPECT_NE(r.body.find("\"decoder_name\":\"\""), std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────
// TD-196: link-world-model
// ─────────────────────────────────────────────────────────────────────────

TEST(BuildLinkWorldModelRequest, RequiresChatbotName) {
    EXPECT_TRUE(build_link_world_model_request({}).error);
}

TEST(BuildLinkWorldModelRequest, RequiresWorldModelFlag) {
    EXPECT_TRUE(build_link_world_model_request({"my-chatbot"}).error);
    EXPECT_TRUE(build_link_world_model_request({"my-chatbot", "--inject-every-n-layers", "2"})
                    .error);
}

TEST(BuildLinkWorldModelRequest, BuildsCorrectPostRequestWithDefaults) {
    auto r = build_link_world_model_request({"my-chatbot", "--world-model", "my-wm"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Post);
    EXPECT_EQ(r.path, "/models/my-chatbot/link-world-model");
    EXPECT_NE(r.body.find("\"world_model_name\":\"my-wm\""), std::string::npos);
    EXPECT_NE(r.body.find("\"world_model_inject_every_n_layers\":0"), std::string::npos);
    EXPECT_NE(r.body.find("\"hippocampal_memory_enabled\":false"), std::string::npos);
    EXPECT_NE(r.body.find("\"hippocampal_memory_capacity\":512"), std::string::npos);
}

TEST(BuildLinkWorldModelRequest, EmptyWorldModelNameBuildsDetachRequest) {
    auto r = build_link_world_model_request({"my-chatbot", "--world-model", ""});
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"world_model_name\":\"\""), std::string::npos);
}

TEST(BuildLinkWorldModelRequest, AllOptionalFlagsRoundTrip) {
    auto r = build_link_world_model_request(
        {"my-chatbot", "--world-model", "my-wm", "--inject-every-n-layers", "3",
         "--hippocampal-enabled", "--hippocampal-capacity", "256",
         "--hippocampal-repetition-alpha", "0.1", "--hippocampal-repetition-decay", "0.9",
         "--hippocampal-cross-reference-alpha", "0.2", "--hippocampal-association-decay", "0.8"});
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"world_model_inject_every_n_layers\":3"), std::string::npos);
    EXPECT_NE(r.body.find("\"hippocampal_memory_enabled\":true"), std::string::npos);
    EXPECT_NE(r.body.find("\"hippocampal_memory_capacity\":256"), std::string::npos);
    EXPECT_NE(r.body.find("\"hippocampal_repetition_alpha\":0.1"), std::string::npos);
    EXPECT_NE(r.body.find("\"hippocampal_repetition_decay\":0.9"), std::string::npos);
    EXPECT_NE(r.body.find("\"hippocampal_cross_reference_alpha\":0.2"), std::string::npos);
    EXPECT_NE(r.body.find("\"hippocampal_association_decay\":0.8"), std::string::npos);
}

TEST(BuildUpdateRunGroupRequest, RequiresRunGroupFlagLiterally) {
    EXPECT_TRUE(build_update_run_group_request({"name"}).error);
    EXPECT_TRUE(build_update_run_group_request({"name", "--wrong-flag", "v"}).error);
    EXPECT_TRUE(build_update_run_group_request({"name", "--run-group"}).error)
        << "missing the value itself must also be an error";
}

TEST(BuildUpdateRunGroupRequest, BuildsCorrectPutRequest) {
    auto r = build_update_run_group_request({"my-model", "--run-group", "nightly"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Put);
    EXPECT_EQ(r.path, "/models/my-model/run_group");
    EXPECT_EQ(r.body, "{\"run_group\":\"nightly\"}");
}

TEST(BuildResolveRequest, RequiresName) {
    EXPECT_TRUE(build_resolve_request({}).error);
}

TEST(BuildResolveRequest, BuildsCorrectPath) {
    auto r = build_resolve_request({"my-model"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.path, "/models/my-model/resolve");
}

TEST(BuildSetTrainingRequest, RequiresName) {
    EXPECT_TRUE(build_set_training_request({}).error);
}

TEST(BuildSetTrainingRequest, DefaultsNewRunFalseAndOmitsSessionKeyWhenNotGiven) {
    auto r = build_set_training_request({"my-model"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Put);
    EXPECT_EQ(r.path, "/models/my-model/state");
    EXPECT_NE(r.body.find("\"new_run\":false"), std::string::npos);
    EXPECT_EQ(r.body.find("metrics_session_key"), std::string::npos);
}

TEST(BuildSetTrainingRequest, NewRunFlagAndSessionKeyAreIncluded) {
    auto r = build_set_training_request({"my-model", "--new-run", "session-42"});
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"new_run\":true"), std::string::npos);
    EXPECT_NE(r.body.find("\"metrics_session_key\":\"session-42\""), std::string::npos);
}

TEST(BuildSetCandidateRequest, RequiresNameAndRunId) {
    EXPECT_TRUE(build_set_candidate_request({"only-name"}).error);
}

TEST(BuildSetCandidateRequest, DefaultsArtifactFormatToAdaiNative) {
    auto r = build_set_candidate_request({"my-model", "run-1"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Put);
    EXPECT_EQ(r.path, "/models/my-model/state");
    EXPECT_NE(r.body.find("\"format\":\"adai-native\""), std::string::npos);
}

TEST(BuildSetCandidateRequest, ArtifactAndSummaryFieldsRoundTrip) {
    auto r = build_set_candidate_request(
        {"my-model", "run-1", "--artifact-path", "/opt/m.bin", "--artifact-host", "host1",
         "--artifact-checksum", "abc123", "--artifact-format", "onnx", "--summary",
         "epochs=10"});
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"path\":\"/opt/m.bin\""), std::string::npos);
    EXPECT_NE(r.body.find("\"host\":\"host1\""), std::string::npos);
    EXPECT_NE(r.body.find("\"checksum\":\"abc123\""), std::string::npos);
    EXPECT_NE(r.body.find("\"format\":\"onnx\""), std::string::npos);
    EXPECT_NE(r.body.find("\"training_summary\":{\"epochs\":\"10\"}"), std::string::npos);
}

TEST(BuildDeleteRequest, RequiresName) {
    EXPECT_TRUE(build_delete_request({}).error);
}

TEST(BuildDeleteRequest, BuildsCorrectDeleteRequest) {
    auto r = build_delete_request({"my-model"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Delete);
    EXPECT_EQ(r.path, "/models/my-model");
}

TEST(BuildRolesRequest, BuildsCorrectPath) {
    auto r = build_roles_request();
    EXPECT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Get);
    EXPECT_EQ(r.path, "/roles");
}

TEST(BuildResolveRoleRequest, RequiresRole) {
    EXPECT_TRUE(build_resolve_role_request({}).error);
}

TEST(BuildResolveRoleRequest, BuildsCorrectPath) {
    auto r = build_resolve_role_request({"chatbot"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.path, "/roles/chatbot/production");
}

TEST(BuildPromoteRequest, RequiresRoleAndModelName) {
    EXPECT_TRUE(build_promote_request({"only-role"}).error);
}

TEST(BuildPromoteRequest, BuildsCorrectPutRequest) {
    auto r = build_promote_request({"chatbot", "my-model-v3"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Put);
    EXPECT_EQ(r.path, "/roles/chatbot/production");
    EXPECT_EQ(r.body, "{\"model_name\":\"my-model-v3\"}");
}

TEST(BuildHealthRequest, BuildsCorrectPath) {
    auto r = build_health_request();
    EXPECT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Get);
    EXPECT_EQ(r.path, "/health");
}
