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
