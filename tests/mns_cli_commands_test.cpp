// Tests for src/MnsCliCommands.{hpp,cpp} — mns_cli's per-command argument parsing and JSON-body
// construction (TD-035), extracted from MnsCliTool.cpp so it's testable without a real mns_server
// to talk to. Each build_*_request() is pure: given a command's argv slice, it returns the
// method/path/body mns_cli would send (or an error), with no network I/O at all — actually
// sending the request stays in MnsCliTool.cpp, unchanged and untested here (that's exercised
// live, end to end, by real mns_server usage and MNSLiveTests on the server side).

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
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
using adai::url_encode;
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

TEST(ParseUrl, StripsHttpsAndKeepsThePathAsAPrefix) {
    auto u = parse_url("https://example.com:1234/some/path/");
    EXPECT_EQ(u.host, "example.com");
    EXPECT_EQ(u.port, 1234);
    EXPECT_TRUE(u.https);
    EXPECT_EQ(u.base_path, "/some/path");
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
    // TD-449: other control characters used to pass through raw (invalid JSON).
    EXPECT_EQ(json_escape(std::string("a\x01" "b\x1f")), "a\\u0001b\\u001f");
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

namespace {
// TD-450: register needs the whole architecture; these supply it.
const std::vector<std::string> kChatArch = {"--d-model", "128", "--num-heads", "4", "--d-ff", "512",
                                            "--encoder-layers", "2", "--decoder-layers", "2", "--max-seq-length", "64"};
const std::vector<std::string> kStandaloneArch = {"--d-model", "128", "--num-heads", "4", "--d-ff", "512",
                                                  "--max-seq-length", "64"};
std::vector<std::string> with(std::vector<std::string> args, const std::vector<std::string>& extra) {
    args.insert(args.end(), extra.begin(), extra.end());
    return args;
}
}  // namespace

TEST(BuildRegisterRequest, TakesTheArchitectureFromTheFlags) {
    ServiceConfig cfg;
    cfg.run_group = "default-group";

    auto r = build_register_request(with({"my-model", "chatbot"}, kChatArch), cfg);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(r.method, HttpMethod::Post);
    EXPECT_EQ(r.path, "/models");
    EXPECT_NE(r.body.find("\"model_name\":\"my-model\""), std::string::npos);
    EXPECT_NE(r.body.find("\"role\":\"chatbot\""), std::string::npos);
    EXPECT_NE(r.body.find("\"run_group\":\"default-group\""), std::string::npos);
    EXPECT_NE(r.body.find("\"d_model\":128"), std::string::npos);
    EXPECT_NE(r.body.find("\"num_heads\":4"), std::string::npos);
    EXPECT_NE(r.body.find("\"num_decoder_layers\":2"), std::string::npos);
    EXPECT_NE(r.body.find("\"max_seq_length\":64"), std::string::npos);
}

TEST(BuildRegisterRequest, RunGroupFlagOverridesConfig) {
    ServiceConfig cfg;
    cfg.run_group = "from-config";
    auto r = build_register_request(with({"m", "role", "--run-group", "from-cli"}, kChatArch), cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"run_group\":\"from-cli\""), std::string::npos);
}

TEST(BuildRegisterRequest, TagsAreIncludedWhenPresentAndOmittedWhenNot) {
    ServiceConfig cfg;
    auto with_tags = build_register_request(with({"m", "role", "--tag", "env=prod"}, kChatArch), cfg);
    ASSERT_FALSE(with_tags.error);
    EXPECT_NE(with_tags.body.find("\"tags\":{\"env\":\"prod\"}"), std::string::npos);

    auto without_tags = build_register_request(with({"m", "role"}, kChatArch), cfg);
    ASSERT_FALSE(without_tags.error);
    EXPECT_EQ(without_tags.body.find("\"tags\""), std::string::npos);
}

TEST(BuildRegisterRequest, NameAndRoleAreJsonEscaped) {
    ServiceConfig cfg;
    auto r = build_register_request(with({"weird\"name", "role"}, kChatArch), cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("weird\\\"name"), std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────
// TD-196: --kind / --num-layers / --encoder / --decoder / --sigreg-* flags
// ─────────────────────────────────────────────────────────────────────────

TEST(BuildRegisterRequest, KindDefaultsToChatbotWhenNotGiven) {
    ServiceConfig cfg;
    auto r = build_register_request(with({"m", "role"}, kChatArch), cfg);
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
    auto r = build_register_request(with({"m", "role", "--kind", "encoder", "--num-layers", "3"}, kStandaloneArch), cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"kind\":\"encoder\""), std::string::npos);
    EXPECT_NE(r.body.find("\"num_encoder_layers\":3"), std::string::npos);
    EXPECT_NE(r.body.find("\"num_decoder_layers\":0"), std::string::npos);
}

TEST(BuildRegisterRequest, DecoderKindNumLayersSetsOwnFieldAndZeroesEncoderLayers) {
    ServiceConfig cfg;
    cfg.num_encoder_layers = 6;
    cfg.num_decoder_layers = 6;
    auto r = build_register_request(with({"m", "role", "--kind", "decoder", "--num-layers", "4"}, kStandaloneArch), cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"num_decoder_layers\":4"), std::string::npos);
    EXPECT_NE(r.body.find("\"num_encoder_layers\":0"), std::string::npos);
}

TEST(BuildRegisterRequest, WorldModelKindNumLayersUsesEncoderLayersField) {
    ServiceConfig cfg;
    auto r =
        build_register_request(with({"m", "role", "--kind", "world_model", "--num-layers", "5"}, kStandaloneArch), cfg);
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"num_encoder_layers\":5"), std::string::npos);
    EXPECT_NE(r.body.find("\"num_decoder_layers\":0"), std::string::npos);
}

TEST(BuildRegisterRequest, WorldModelKindIncludesSigregParamsInConnection) {
    ServiceConfig cfg;
    auto r = build_register_request(with({"m", "role", "--kind", "world_model", "--num-layers", "5",
                                          "--sigreg-lambda", "2.5", "--sigreg-num-sketches", "32"},
                                         kStandaloneArch),
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

// TD-449: only the settings given are sent, with "merge":true, so mns_server keeps the others; all eight
// used to be sent with defaults, which turned hippocampal memory off on every relink.
TEST(BuildLinkWorldModelRequest, SendsOnlyTheGivenSettingsAndAsksTheServerToMerge) {
    auto r = build_link_world_model_request({"my-chatbot", "--world-model", "my-wm"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.method, HttpMethod::Post);
    EXPECT_EQ(r.path, "/models/my-chatbot/link-world-model");
    EXPECT_EQ(r.body, "{\"world_model_name\":\"my-wm\",\"merge\":true}");

    r = build_link_world_model_request({"my-chatbot", "--world-model", "my-wm", "--inject-every-n-layers", "2"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.body, "{\"world_model_name\":\"my-wm\",\"merge\":true,\"world_model_inject_every_n_layers\":2}");

    r = build_link_world_model_request({"my-chatbot", "--world-model", "my-wm", "--hippocampal-disabled"});
    ASSERT_FALSE(r.error);
    EXPECT_NE(r.body.find("\"hippocampal_memory_enabled\":false"), std::string::npos);
    EXPECT_TRUE(build_link_world_model_request(
                    {"c", "--world-model", "w", "--hippocampal-enabled", "--hippocampal-disabled"})
                    .error);
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

// TD-460: only an explicit --revive brings a retired model back.
TEST(BuildSetCandidateRequest, ReviveIsSentOnlyWhenAskedFor) {
    auto plain = build_set_candidate_request({"my-model", "run-1"});
    ASSERT_FALSE(plain.error);
    EXPECT_EQ(plain.body.find("revive"), std::string::npos);
    auto revive = build_set_candidate_request({"my-model", "run-1", "--revive"});
    ASSERT_FALSE(revive.error);
    EXPECT_NE(revive.body.find("\"revive\":true"), std::string::npos);
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

// ---------------------------------------------------------------------------------------------
// TD-448: unknown options, missing values, stray arguments and bad numbers are usage errors.
// ---------------------------------------------------------------------------------------------

namespace {
bool error_contains(const adai::MnsCliRequest& r, const std::string& text) {
    return r.error && r.error_message.find(text) != std::string::npos;
}
}  // namespace

TEST(Td448UnknownOptions, AMisspelledFlagIsAnErrorInEveryBuilder) {
    ServiceConfig cfg;
    // --dmodel used to be skipped, registering the config's d_model permanently.
    EXPECT_TRUE(error_contains(build_register_request({"m", "chat", "--dmodel", "256"}, cfg), "Unknown option: --dmodel"));
    EXPECT_TRUE(error_contains(build_list_request({"--stat", "training"}), "Unknown option: --stat"));
    EXPECT_TRUE(error_contains(build_link_world_model_request({"c", "--world-model", "w", "--inject-every", "2"}),
                               "Unknown option: --inject-every"));
    EXPECT_TRUE(error_contains(build_set_candidate_request({"m", "run-01", "--artifact-pth", "/x"}),
                               "Unknown option: --artifact-pth"));
}

TEST(Td448UnknownOptions, StrayPositionalArgumentsAreErrors) {
    ServiceConfig cfg;
    EXPECT_TRUE(error_contains(build_register_request({"m", "chat", "256"}, cfg), "Unexpected argument: 256"));
    EXPECT_TRUE(error_contains(build_list_request({"training"}), "Unexpected argument: training"));
}

TEST(Td448MissingValues, AValueFlagWithNoValueSaysSo) {
    ServiceConfig cfg;
    EXPECT_TRUE(error_contains(build_register_request({"m", "chat", "--d-model"}, cfg), "Missing value for --d-model"));
    EXPECT_TRUE(error_contains(build_register_request({"m", "chat", "--d-model", "--num-heads", "4"}, cfg),
                               "Missing value for --d-model"));
    EXPECT_TRUE(error_contains(build_list_request({"--limit"}), "Missing value for --limit"));
    EXPECT_TRUE(error_contains(build_set_candidate_request({"m", "run-01", "--artifact-path"}),
                               "Missing value for --artifact-path"));
    // An empty value is still a value: --world-model "" detaches.
    EXPECT_FALSE(build_link_world_model_request({"c", "--world-model", ""}).error);
}

TEST(Td448SetTraining, StrayArgumentsNoLongerBecomeTheSessionKey) {
    // --new_run used to become the metrics session key, continuing the current run.
    EXPECT_TRUE(error_contains(build_set_training_request({"m", "--new_run"}), "Unknown option: --new_run"));
    EXPECT_TRUE(error_contains(build_set_training_request({"m", "key-a", "key-b"}), "Unexpected argument: key-b"));
    auto r = build_set_training_request({"m", "key-a", "--new-run"});
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_NE(r.body.find("\"new_run\":true"), std::string::npos);
    EXPECT_NE(r.body.find("\"metrics_session_key\":\"key-a\""), std::string::npos);
}

TEST(Td448Numbers, CountsMustBeWholeAndInRange) {
    ServiceConfig cfg;
    // abc aborted mns_cli, -1 wrapped to 18446744073709551615, 512x became 512.
    for (const std::string bad : {"abc", "-1", "512x", "0", "", "1e3", "99999999999"}) {
        EXPECT_TRUE(error_contains(build_register_request({"m", "chat", "--d-model", bad}, cfg),
                                   "Invalid value for --d-model"))
            << "'" << bad << "'";
    }
    EXPECT_TRUE(error_contains(build_register_request({"m", "enc", "--kind", "encoder", "--num-layers", "0"}, cfg),
                               "Invalid value for --num-layers"));
    EXPECT_TRUE(error_contains(build_list_request({"--limit", "-5"}), "Invalid value for --limit"));
    EXPECT_TRUE(error_contains(build_link_world_model_request({"c", "--world-model", "w", "--hippocampal-capacity", "x"}),
                               "Invalid value for --hippocampal-capacity"));

    auto r = build_register_request({"m", "chat", "--d-model", "256", "--num-heads", "4", "--d-ff", "1024",
                                     "--encoder-layers", "0", "--decoder-layers", "2", "--max-seq-length", "64"},
                                    cfg);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_NE(r.body.find("\"d_model\":256"), std::string::npos);
    EXPECT_NE(r.body.find("\"num_encoder_layers\":0"), std::string::npos);
}

TEST(Td448Numbers, FloatsMustBeFiniteAndInRange) {
    ServiceConfig cfg;
    EXPECT_TRUE(error_contains(build_register_request({"m", "wm", "--kind", "world_model", "--sigreg-lambda", "nan"}, cfg),
                               "Invalid value for --sigreg-lambda"));
    EXPECT_TRUE(error_contains(build_register_request({"m", "wm", "--kind", "world_model", "--sigreg-lambda", "-1"}, cfg),
                               "Invalid value for --sigreg-lambda"));
    EXPECT_TRUE(error_contains(build_link_world_model_request({"c", "--world-model", "w", "--hippocampal-repetition-decay", "1.5"}),
                               "Invalid value for --hippocampal-repetition-decay"));
    EXPECT_TRUE(error_contains(build_link_world_model_request({"c", "--world-model", "w", "--hippocampal-repetition-alpha", "0.1x"}),
                               "Invalid value for --hippocampal-repetition-alpha"));
}

TEST(Td448KeyValues, TagsAndSummariesNeedAnEqualsSign) {
    ServiceConfig cfg;
    EXPECT_TRUE(error_contains(build_register_request({"m", "chat", "--tag", "owner"}, cfg), "Invalid value for --tag"));
    EXPECT_TRUE(error_contains(build_set_candidate_request({"m", "run-01", "--summary", "=1"}), "Invalid value for --summary"));
}

// ---------------------------------------------------------------------------------------------
// TD-450: register's architecture comes from the flags or --from-config, never from mns_cli's config.
// ---------------------------------------------------------------------------------------------

namespace {
class Td450FromConfig : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = (std::filesystem::temp_directory_path() /
                 ("td450_" + std::to_string(::getpid()) + "_" +
                  ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".conf"))
                    .string();
        std::ofstream(path_) << "D_MODEL=768\nNUM_HEADS=12\nD_FF=3072\nNUM_ENCODER_LAYERS=6\n"
                                "NUM_DECODER_LAYERS=4\nMAX_SEQ_LENGTH=512\n";
    }
    void TearDown() override { std::filesystem::remove(path_); }
    std::string path_;
};

ServiceConfig mns_conf_like() {
    // config.mns.conf's old architecture block; none of it may reach a registration.
    ServiceConfig cfg;
    cfg.d_model = 512;
    cfg.num_heads = 8;
    cfg.d_ff = 2048;
    cfg.num_encoder_layers = 6;
    cfg.num_decoder_layers = 6;
    cfg.max_seq_length = 1024;
    return cfg;
}
}  // namespace

TEST(Td450Register, AFlaglessRegisterIsAnErrorNotTheDaemonConfigsArchitecture) {
    // It used to register mns_cli's config.mns.conf architecture (512/8), permanently.
    auto r = build_register_request({"m", "chatbot"}, mns_conf_like());
    ASSERT_TRUE(r.error);
    EXPECT_NE(r.error_message.find("Missing: --d-model, --num-heads, --d-ff, --encoder-layers, --decoder-layers, "
                                   "--max-seq-length"),
              std::string::npos)
        << r.error_message;
    EXPECT_NE(r.error_message.find("--from-config"), std::string::npos);
}

TEST(Td450Register, OnlyTheMissingFieldsAreListed) {
    auto r = build_register_request({"m", "chatbot", "--d-model", "256", "--num-heads", "8"}, mns_conf_like());
    ASSERT_TRUE(r.error);
    EXPECT_NE(r.error_message.find("Missing: --d-ff, --encoder-layers, --decoder-layers, --max-seq-length."),
              std::string::npos)
        << r.error_message;

    auto standalone = build_register_request(with({"e", "enc", "--kind", "encoder"}, kStandaloneArch), mns_conf_like());
    ASSERT_TRUE(standalone.error);
    EXPECT_NE(standalone.error_message.find("Missing: --num-layers."), std::string::npos) << standalone.error_message;
}

TEST(Td450Register, ThePrintedNoticeShowsTheArchitectureAndItsSource) {
    auto r = build_register_request(with({"m", "chatbot"}, kChatArch), mns_conf_like());
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(r.notice,
              "Registering m (chatbot) with d_model=128 num_heads=4 d_ff=512 num_encoder_layers=2 "
              "num_decoder_layers=2 max_seq_length=64 (from the command line)");
    // A linked chatbot sends no architecture, so there's nothing to print.
    EXPECT_TRUE(build_register_request({"m", "chatbot", "--encoder", "e", "--decoder", "d"}, mns_conf_like())
                    .notice.empty());
}

TEST_F(Td450FromConfig, TakesTheTrainersArchitectureWithFlagsOverriding) {
    auto r = build_register_request({"m", "chatbot", "--from-config", path_, "--max-seq-length", "256"},
                                    mns_conf_like());
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_NE(r.body.find("\"arch\":{\"d_model\":768,\"num_heads\":12,\"d_ff\":3072,\"num_encoder_layers\":6,"
                          "\"num_decoder_layers\":4,\"max_seq_length\":256}"),
              std::string::npos)
        << r.body;
    EXPECT_NE(r.notice.find("(from " + path_ + " and the command line)"), std::string::npos) << r.notice;
}

TEST_F(Td450FromConfig, AStandaloneKindTakesItsOwnLayerCountAndZeroesTheOther) {
    auto dec = build_register_request({"d", "dec", "--kind", "decoder", "--from-config", path_}, mns_conf_like());
    ASSERT_FALSE(dec.error) << dec.error_message;
    EXPECT_NE(dec.body.find("\"num_encoder_layers\":0,\"num_decoder_layers\":4"), std::string::npos) << dec.body;

    auto wm = build_register_request({"w", "wm", "--kind", "world_model", "--from-config", path_, "--num-layers", "3"},
                                     mns_conf_like());
    ASSERT_FALSE(wm.error) << wm.error_message;
    EXPECT_NE(wm.body.find("\"num_encoder_layers\":3,\"num_decoder_layers\":0"), std::string::npos) << wm.body;
}

TEST_F(Td450FromConfig, BadCombinationsAreErrors) {
    EXPECT_TRUE(error_contains(build_register_request({"m", "chatbot", "--from-config", path_ + ".missing"}, {}),
                               "--from-config: can't read"));
    EXPECT_TRUE(error_contains(
        build_register_request({"m", "chatbot", "--encoder", "e", "--decoder", "d", "--from-config", path_}, {}),
        "cannot be combined"));
    // --num-layers used to be ignored on a chatbot.
    EXPECT_TRUE(error_contains(build_register_request({"m", "chatbot", "--from-config", path_, "--num-layers", "2"}, {}),
                               "--num-layers is for --kind encoder/decoder/world_model"));
}

// ---------------------------------------------------------------------------------------------
// TD-451: https://, strict URLs, exact arguments, header lists, global options and --help.
// ---------------------------------------------------------------------------------------------

TEST(Td451ParseUrl, HttpsUsesPort443AndTheRightScheme) {
    // https://host used to keep port 8083 and connect with plain HTTP.
    auto u = parse_url("https://mns.example.com");
    EXPECT_TRUE(u.error.empty()) << u.error;
    EXPECT_TRUE(u.https);
    EXPECT_EQ(u.port, 443);
    EXPECT_EQ(u.url_for("/health"), "https://mns.example.com:443/health");

    auto plain = parse_url("http://10.0.0.5");
    EXPECT_FALSE(plain.https);
    EXPECT_EQ(plain.url_for("/models"), "http://10.0.0.5:8083/models");

    EXPECT_EQ(parse_url("https://proxy.example.com/mns").url_for("/health"), "https://proxy.example.com:443/mns/health");
    auto v6 = parse_url("http://[::1]:9000");
    EXPECT_TRUE(v6.error.empty()) << v6.error;
    EXPECT_EQ(v6.host, "[::1]");
    EXPECT_EQ(v6.port, 9000);
}

TEST(Td451ParseUrl, BadSchemesHostsAndPortsAreErrors) {
    // A bad port used to be ignored (8083 kept); ftp:// was read as a host name.
    for (const std::string bad : {"http://host:abc", "http://host:0", "http://host:99999", "http://host:",
                                  "http://host:80x", "ftp://host", "http://", "http://:8083", "http://[::1"}) {
        EXPECT_FALSE(parse_url(bad).error.empty()) << bad;
    }
}

TEST(Td451ExactArguments, ExtraArgumentsAreErrorsNotIgnored) {
    // `delete m --url X` would have deleted m on the default server once global options stop at the
    // command word; extra arguments used to be ignored.
    EXPECT_TRUE(error_contains(build_delete_request({"m", "--url", "http://other"}), "Unknown option: --url"));
    EXPECT_TRUE(error_contains(build_get_request({"m", "n"}), "Unexpected argument: n"));
    EXPECT_TRUE(error_contains(build_resolve_request({}), "Missing argument"));
    EXPECT_TRUE(error_contains(build_resolve_role_request({"chatbot", "x"}), "Unexpected argument: x"));
    EXPECT_TRUE(error_contains(build_promote_request({"chatbot", "m", "x"}), "Unexpected argument: x"));
    EXPECT_TRUE(error_contains(build_roles_request({"x"}), "Unexpected argument: x"));
    EXPECT_TRUE(error_contains(build_health_request({"--verbose"}), "Unknown option: --verbose"));
    EXPECT_FALSE(build_health_request({}).error);
}

TEST(Td451ExactArguments, UpdateTakesItsOptionInAnyOrder) {
    auto r = build_update_run_group_request({"--run-group", "nightly", "my-model"});
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(r.path, "/models/my-model/run_group");
    EXPECT_EQ(r.body, "{\"run_group\":\"nightly\"}");
    EXPECT_TRUE(error_contains(build_update_run_group_request({"m", "--run-group", "a", "extra"}),
                               "Unexpected argument: extra"));
}

TEST(Td451Headers, TheSharedHeaderListMatchesTheMetricsFormat) {
    EXPECT_EQ(adai::parse_http_header_list(" CF-Access-Client-Id: abc ;CF-Access-Client-Secret: xyz;bad;: x; "),
              (std::vector<std::string>{"CF-Access-Client-Id: abc", "CF-Access-Client-Secret: xyz"}));
}

#ifdef MNS_CLI_BINARY_PATH
namespace {
struct CliRun {
    int exit_code = -1;
    std::string output;
};
CliRun run_mns_cli(const std::string& args) {
    const auto out = std::filesystem::temp_directory_path() / ("td451_out_" + std::to_string(::getpid()));
    const std::string cmd = std::string(MNS_CLI_BINARY_PATH) + " " + args + " > " + out.string() + " 2>&1";
    const int status = std::system(cmd.c_str());
    std::stringstream text;
    text << std::ifstream(out).rdbuf();
    std::filesystem::remove(out);
    return {WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status), text.str()};
}
}  // namespace

TEST(Td451MnsCliBinary, CommandHelpPrintsThatCommandAndSendsNothing) {
    // `list --help` used to send a list request.
    const auto r = run_mns_cli("--config /dev/null --url http://127.0.0.1:1 list --help");
    EXPECT_EQ(r.exit_code, 0) << r.output;
    EXPECT_NE(r.output.find("list [--state STATE]"), std::string::npos) << r.output;
    EXPECT_EQ(r.output.find("register <name>"), std::string::npos) << "only list's entry: " << r.output;
    EXPECT_EQ(r.output.find("failed"), std::string::npos) << r.output;
}

TEST(Td451MnsCliBinary, GlobalOptionsStopAtTheCommandWord) {
    const auto after = run_mns_cli("--config /dev/null --url http://127.0.0.1:1 get m --url http://127.0.0.1:2");
    EXPECT_EQ(after.exit_code, 1);
    EXPECT_NE(after.output.find("Unknown option: --url"), std::string::npos) << after.output;
    const auto unknown = run_mns_cli("--config /dev/null --verbose health");
    EXPECT_EQ(unknown.exit_code, 1);
    EXPECT_NE(unknown.output.find("Unknown global option: --verbose"), std::string::npos) << unknown.output;
    const auto bad_url = run_mns_cli("--config /dev/null --url http://127.0.0.1:abc health");
    EXPECT_EQ(bad_url.exit_code, 1);
    EXPECT_NE(bad_url.output.find("Invalid port"), std::string::npos) << bad_url.output;
}

TEST(Td451MnsCliBinary, HttpsIsSpokenNotDowngraded) {
    // The old binary connected to 127.0.0.1:8083 with plain HTTP for https://127.0.0.1:1.
    const auto r = run_mns_cli("--config /dev/null --url https://127.0.0.1:1 health");
    EXPECT_EQ(r.exit_code, 1);
#ifdef ADAI_METRICS_PUSH_CURL
    EXPECT_NE(r.output.find("request to https://127.0.0.1:1/health failed"), std::string::npos) << r.output;
#else
    EXPECT_NE(r.output.find("can't use https://"), std::string::npos) << r.output;
#endif
}
#endif

#ifdef MNS_CLI_BINARY_PATH
// The real binary: a malformed number used to abort with a core dump (exit 134) before any request.
TEST(Td448MnsCliBinary, AMalformedNumberIsAUsageErrorNotACrash) {
    for (const std::string args : {"register foo chatbot --d-model abc", "register foo chatbot --dmodel 256",
                                   "set-training foo --new_run", "list --limit", "--url"}) {
        const std::string cmd = std::string(MNS_CLI_BINARY_PATH) + " --config /dev/null " + args + " >/dev/null 2>&1";
        const int status = std::system(cmd.c_str());
        ASSERT_NE(status, -1);
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 1)
            << args << (WIFSIGNALED(status) ? ": killed by signal " + std::to_string(WTERMSIG(status))
                                            : ": exit " + std::to_string(WEXITSTATUS(status)));
    }
}
// TD-450: the real binary refuses a flagless register before connecting, even when its own config
// file has an architecture block (the old binary sent it and failed only on the connection).
TEST(Td450MnsCliBinary, AFlaglessRegisterNeverReachesTheServer) {
    const auto dir = std::filesystem::temp_directory_path() / ("td450_bin_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    const auto conf = dir / "config.mns.conf";
    const auto out = dir / "out.txt";
    std::ofstream(conf) << "D_MODEL=512\nNUM_HEADS=8\nD_FF=2048\nNUM_ENCODER_LAYERS=6\nNUM_DECODER_LAYERS=6\n"
                           "MAX_SEQ_LENGTH=1024\n";
    const std::string cmd = std::string(MNS_CLI_BINARY_PATH) + " --config " + conf.string() +
                            " --url http://127.0.0.1:1 register foo chatbot > " + out.string() + " 2>&1";
    const int status = std::system(cmd.c_str());
    std::stringstream text;
    text << std::ifstream(out).rdbuf();
    std::filesystem::remove_all(dir);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    EXPECT_NE(text.str().find("Missing: --d-model"), std::string::npos) << text.str();
    EXPECT_EQ(text.str().find("connection to"), std::string::npos) << text.str();
}
#endif


// TD-449: names and query values weren't URL-encoded: `get a/state` requested /models/a/state, and
// `list --role "a b&c"` built /models?role=a b&c.
TEST(UrlEncoding, NamesAndQueryValuesAreEncoded) {
    EXPECT_EQ(url_encode("my-model_1.v2~x"), "my-model_1.v2~x");
    EXPECT_EQ(url_encode("a/b c&d?"), "a%2Fb%20c%26d%3F");
    EXPECT_EQ(build_get_request({"a/state"}).path, "/models/a%2Fstate");
    EXPECT_EQ(build_list_request({"--role", "a b&c", "--state", "x=y"}).path, "/models?state=x%3Dy&role=a%20b%26c");
    EXPECT_EQ(build_link_world_model_request({"a/b", "--world-model", "w"}).path, "/models/a%2Fb/link-world-model");
}

// TD-449: --sigreg-* were silently dropped for kinds other than world_model.
TEST(BuildRegisterRequest, SigregFlagsAreOnlyForWorldModels) {
    adai::ServiceConfig cfg;
    for (const std::string kind : {"chatbot", "encoder", "decoder"}) {
        auto r = build_register_request({"m", "r", "--kind", kind, "--sigreg-lambda", "0.5"}, cfg);
        EXPECT_TRUE(r.error) << kind;
        EXPECT_NE(r.error_message.find("only valid with --kind world_model"), std::string::npos) << r.error_message;
    }
    auto r = build_register_request({"m", "r", "--kind", "world_model", "--d-model", "64", "--num-heads", "2",
                                     "--d-ff", "128", "--num-layers", "1", "--max-seq-length", "32",
                                     "--sigreg-num-sketches", "16"},
                                    cfg);
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_NE(r.body.find("\"sigreg_num_sketches\":16"), std::string::npos);
}
