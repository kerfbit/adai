// TD-037: MnsJsonHelpers.hpp had zero test coverage despite being tagged `stable` and having a
// real bug-fix history (TD-068, TD-102, both referenced in its own comments) — exactly the
// "non-widget logic GTest can already exercise" TD-037's own Description calls out as the
// tractable near-term step, ahead of any QTest/widget-testing framework decision. This is pure,
// Qt-free C++ (no MnsManagerGUI instance needed at all).
#include "MnsJsonHelpers.hpp"
#include <gtest/gtest.h>

using namespace mns_gui;

// ---------------------------------------------------------------------------
// find_string_end
// ---------------------------------------------------------------------------

TEST(FindStringEndTest, FindsTheClosingQuote) {
    std::string body = R"("hello")";
    EXPECT_EQ(find_string_end(body, 1), 6u);
}

TEST(FindStringEndTest, SkipsEscapedQuotes) {
    // he said \"hi\" -- the escaped quotes must not be mistaken for the closing one.
    std::string body = R"("he said \"hi\"")";
    auto end = find_string_end(body, 1);
    ASSERT_NE(end, std::string::npos);
    EXPECT_EQ(body[end], '"');
    EXPECT_EQ(body.substr(1, end - 1), R"(he said \"hi\")");
}

TEST(FindStringEndTest, ReturnsNposForAnUnterminatedString) {
    std::string body = R"("never closed)";
    EXPECT_EQ(find_string_end(body, 1), std::string::npos);
}

// ---------------------------------------------------------------------------
// json_value
// ---------------------------------------------------------------------------

TEST(JsonValueTest, ExtractsAStringValue) {
    EXPECT_EQ(json_value(R"({"model_name":"chatbot-main"})", "model_name"), "chatbot-main");
}

TEST(JsonValueTest, ExtractsANumericValueVerbatim) {
    EXPECT_EQ(json_value(R"({"d_model":512,"num_heads":8})", "d_model"), "512");
    EXPECT_EQ(json_value(R"({"d_model":512,"num_heads":8})", "num_heads"), "8");
}

TEST(JsonValueTest, ExtractsABooleanValueVerbatim) {
    EXPECT_EQ(json_value(R"({"admin_enabled":true})", "admin_enabled"), "true");
}

TEST(JsonValueTest, ReturnsEmptyForAMissingKey) {
    EXPECT_EQ(json_value(R"({"a":"b"})", "missing"), "");
}

TEST(JsonValueTest, HandlesTrailingValueAtEndOfBody) {
    // No trailing comma/brace/space after the value at all.
    EXPECT_EQ(json_value(R"("port":8083)", "port"), "8083");
}

TEST(JsonValueTest, StringValueMayContainEscapedQuotes) {
    EXPECT_EQ(json_value(R"({"error":"bad \"input\" given"})", "error"), R"(bad \"input\" given)");
}

// ---------------------------------------------------------------------------
// json_array_objects
// ---------------------------------------------------------------------------

TEST(JsonArrayObjectsTest, SplitsMultipleObjectsUnderTheGivenKey) {
    std::string body = R"({"models":[{"model_name":"a"},{"model_name":"b"}]})";
    auto objs = json_array_objects(body, "models");
    ASSERT_EQ(objs.size(), 2u);
    EXPECT_EQ(json_value(objs[0], "model_name"), "a");
    EXPECT_EQ(json_value(objs[1], "model_name"), "b");
}

TEST(JsonArrayObjectsTest, FallsBackToTheFirstArrayWhenTheKeyIsAbsent) {
    std::string body = R"({"entries":[{"x":1},{"x":2}]})";
    auto objs = json_array_objects(body, "does_not_exist");
    ASSERT_EQ(objs.size(), 2u);
}

TEST(JsonArrayObjectsTest, ReturnsEmptyWhenNoArrayExistsAtAll) {
    EXPECT_TRUE(json_array_objects(R"({"foo":"bar"})", "models").empty());
}

// TD-068 regression: a string value containing a literal '{' or '}' must not desynchronize the
// brace-depth counter and corrupt this or any later object in the array.
TEST(JsonArrayObjectsTest, StringValuesContainingBracesDoNotDesyncDepthCounting) {
    std::string body = R"({"models":[{"tags":{"note":"uses { and } in a label"}},{"model_name":"b"}]})";
    auto objs = json_array_objects(body, "models");
    ASSERT_EQ(objs.size(), 2u);
    EXPECT_EQ(json_value(objs[1], "model_name"), "b");
}

TEST(JsonArrayObjectsTest, StopsAtAnUnterminatedString) {
    std::string body = R"({"models":[{"model_name":"a)";
    // Must not hang or throw -- just stops parsing at the malformed input.
    auto objs = json_array_objects(body, "models");
    EXPECT_TRUE(objs.empty());
}

// ---------------------------------------------------------------------------
// json_escape
// ---------------------------------------------------------------------------

TEST(JsonEscapeTest, EscapesQuotesBackslashesAndNewlines) {
    EXPECT_EQ(json_escape("say \"hi\""), R"(say \"hi\")");
    EXPECT_EQ(json_escape("C:\\models\\"), R"(C:\\models\\)");
    EXPECT_EQ(json_escape("line1\nline2"), "line1\\nline2");
}

TEST(JsonEscapeTest, LeavesOrdinaryTextUnchanged) {
    EXPECT_EQ(json_escape("chatbot-main"), "chatbot-main");
}

// ---------------------------------------------------------------------------
// json_pretty
// ---------------------------------------------------------------------------

TEST(JsonPrettyTest, IndentsNestedObjectsAndArrays) {
    std::string out = json_pretty(R"({"a":1,"b":[2,3]})");
    EXPECT_NE(out.find("{\n  \"a\": 1"), std::string::npos);
    EXPECT_NE(out.find("[\n    2"), std::string::npos);
}

// TD-102 regression: a string ending in an escaped backslash right before its closing quote
// (e.g. a Windows path) must not be misread as an escaped closing quote.
TEST(JsonPrettyTest, HandlesAStringEndingInAnEscapedBackslash) {
    std::string body = R"({"path":"C:\\models\\","next":"ok"})";
    std::string out = json_pretty(body);
    EXPECT_NE(out.find(R"("path": "C:\\models\\")"), std::string::npos);
    EXPECT_NE(out.find(R"("next": "ok")"), std::string::npos);
}

// ---------------------------------------------------------------------------
// parse_tags
// ---------------------------------------------------------------------------

TEST(ParseTagsTest, ParsesCommaSeparatedKeyValuePairs) {
    auto tags = parse_tags("env=prod,region=us-east");
    ASSERT_EQ(tags.size(), 2u);
    EXPECT_EQ(tags[0].first, "env");
    EXPECT_EQ(tags[0].second, "prod");
    EXPECT_EQ(tags[1].first, "region");
    EXPECT_EQ(tags[1].second, "us-east");
}

TEST(ParseTagsTest, TrimsSurroundingWhitespaceAroundKeysAndValues) {
    auto tags = parse_tags(" env = prod , region = us-east ");
    ASSERT_EQ(tags.size(), 2u);
    EXPECT_EQ(tags[0].first, "env");
    EXPECT_EQ(tags[0].second, "prod");
    EXPECT_EQ(tags[1].first, "region");
    EXPECT_EQ(tags[1].second, "us-east");
}

TEST(ParseTagsTest, SkipsEntriesWithNoEqualsSign) {
    auto tags = parse_tags("env=prod,malformed,region=us-east");
    ASSERT_EQ(tags.size(), 2u);
    EXPECT_EQ(tags[0].first, "env");
    EXPECT_EQ(tags[1].first, "region");
}

TEST(ParseTagsTest, ReturnsEmptyForAnEmptyString) {
    EXPECT_TRUE(parse_tags("").empty());
}

// ---------------------------------------------------------------------------
// build_register_model_body
// ---------------------------------------------------------------------------

TEST(BuildRegisterModelBodyTest, IncludesNameRoleAndArchitecture) {
    RegisterModelArch arch{512, 8, 2048, 6, 6, 128};
    std::string body = build_register_model_body("chatbot-main", "chat", arch, "");

    EXPECT_EQ(json_value(body, "model_name"), "chatbot-main");
    EXPECT_EQ(json_value(body, "role"), "chat");
    // The arch sub-object's fields are checked directly via json_value on the whole body
    // (json_value finds the first matching key regardless of nesting, which is unambiguous here).
    EXPECT_EQ(json_value(body, "d_model"), "512");
    EXPECT_EQ(json_value(body, "num_heads"), "8");
    EXPECT_EQ(json_value(body, "d_ff"), "2048");
    EXPECT_EQ(json_value(body, "num_encoder_layers"), "6");
    EXPECT_EQ(json_value(body, "num_decoder_layers"), "6");
    EXPECT_EQ(json_value(body, "max_seq_length"), "128");
}

TEST(BuildRegisterModelBodyTest, OmitsTagsEntirelyWhenTagsStringIsEmpty) {
    RegisterModelArch arch{};
    std::string body = build_register_model_body("m", "r", arch, "");
    EXPECT_EQ(body.find("\"tags\""), std::string::npos);
}

TEST(BuildRegisterModelBodyTest, IncludesParsedTagsWhenPresent) {
    RegisterModelArch arch{};
    std::string body = build_register_model_body("m", "r", arch, "env=prod, region=us-east");
    EXPECT_NE(body.find(R"("tags":{"env":"prod","region":"us-east"})"), std::string::npos);
}

TEST(BuildRegisterModelBodyTest, EscapesNameAndRole) {
    RegisterModelArch arch{};
    std::string body = build_register_model_body("model \"x\"", "role", arch, "");
    EXPECT_NE(body.find(R"("model_name":"model \"x\"")"), std::string::npos);
}

TEST(BuildRegisterModelBodyTest, ProducesValidJsonParsableByJsonValue) {
    // End-to-end sanity: the body this builds must itself be parsable by json_value/
    // json_array_objects, the same functions the rest of the app uses to read server responses.
    RegisterModelArch arch{256, 4, 1024, 3, 3, 64};
    std::string body = build_register_model_body("m", "r", arch, "k=v");
    EXPECT_EQ(json_value(body, "model_name"), "m");
    EXPECT_EQ(json_value(body, "k"), "v");
}

// ---------------------------------------------------------------------------
// build_set_candidate_body
// ---------------------------------------------------------------------------

TEST(BuildSetCandidateBodyTest, AlwaysIncludesTheCandidateState) {
    EXPECT_EQ(build_set_candidate_body("", ""), R"({"state":"candidate"})");
}

TEST(BuildSetCandidateBodyTest, IncludesRunIdWhenProvided) {
    std::string body = build_set_candidate_body("run-42", "");
    EXPECT_EQ(json_value(body, "state"), "candidate");
    EXPECT_EQ(json_value(body, "run_id"), "run-42");
    EXPECT_EQ(body.find("\"artifact\""), std::string::npos);
}

TEST(BuildSetCandidateBodyTest, IncludesArtifactPathWhenProvided) {
    std::string body = build_set_candidate_body("", "/data/model.bin");
    EXPECT_EQ(body.find("\"run_id\""), std::string::npos);
    EXPECT_NE(body.find(R"("artifact":{"path":"/data/model.bin")"), std::string::npos);
}

TEST(BuildSetCandidateBodyTest, IncludesBothWhenBothProvided) {
    std::string body = build_set_candidate_body("run-42", "/data/model.bin");
    EXPECT_EQ(json_value(body, "run_id"), "run-42");
    EXPECT_NE(body.find("\"artifact\""), std::string::npos);
}

// ---------------------------------------------------------------------------
// ParsedUrl::from
// ---------------------------------------------------------------------------

TEST(ParsedUrlTest, ParsesHostAndPort) {
    auto p = ParsedUrl::from("http://localhost:8083");
    EXPECT_EQ(p.host, "localhost");
    EXPECT_EQ(p.port, 8083);
}

TEST(ParsedUrlTest, ParsesHttpsScheme) {
    auto p = ParsedUrl::from("https://mns.example.com:9000");
    EXPECT_EQ(p.host, "mns.example.com");
    EXPECT_EQ(p.port, 9000);
}

TEST(ParsedUrlTest, DefaultsPortWhenAbsent) {
    // ParsedUrl's own default port (8083) must survive when the URL has none.
    auto p = ParsedUrl::from("http://localhost");
    EXPECT_EQ(p.host, "localhost");
    EXPECT_EQ(p.port, 8083);
}

TEST(ParsedUrlTest, StripsATrailingPath) {
    auto p = ParsedUrl::from("http://localhost:8083/models");
    EXPECT_EQ(p.host, "localhost");
    EXPECT_EQ(p.port, 8083);
}

TEST(ParsedUrlTest, HandlesNoSchemeAtAll) {
    auto p = ParsedUrl::from("localhost:8083");
    EXPECT_EQ(p.host, "localhost");
    EXPECT_EQ(p.port, 8083);
}

TEST(ParsedUrlTest, IgnoresAnUnparseablePortAndKeepsTheDefault) {
    auto p = ParsedUrl::from("http://localhost:notaport");
    EXPECT_EQ(p.host, "localhost");
    EXPECT_EQ(p.port, 8083);  // default, since std::stoi threw and was swallowed
}
