// TD-037: MnsJsonHelpers.hpp had zero test coverage despite being tagged `stable` and having a
// real bug-fix history (TD-068, TD-102, both referenced in its own comments) — exactly the
// "non-widget logic GTest can already exercise" TD-037's own Description calls out as the
// tractable near-term step, ahead of any QTest/widget-testing framework decision. This is pure,
// Qt-free C++ (no MnsManagerGUI instance needed at all).
#include "MnsJsonHelpers.hpp"
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"

using namespace mns_gui;

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

// TD-452: the helpers parse JSON, so text that isn't JSON gives nothing rather than a guess.
TEST(JsonValueTest, ReturnsEmptyForTextThatIsntJson) {
    EXPECT_EQ(json_value(R"("port":8083)", "port"), "");
    EXPECT_EQ(json_value("not json", "port"), "");
}

// TD-452: string values are unescaped.
TEST(JsonValueTest, UnescapesStringValues) {
    EXPECT_EQ(json_value(R"({"error":"bad \"input\" given"})", "error"), R"(bad "input" given)");
    EXPECT_EQ(json_value(R"({"path":"C:\\models\\v3.bin"})", "path"), R"(C:\models\v3.bin)");
    EXPECT_EQ(json_value(R"({"name":"caf\u00e9\tbar"})", "name"), "caf\xc3\xa9\tbar");
}

TEST(JsonValueTest, AcceptsSpacesAroundTheColon) {
    EXPECT_EQ(json_value(R"({ "state" : "training" })", "state"), "training");
}

TEST(JsonValueTest, ReturnsAnObjectValueWhole) {
    EXPECT_EQ(json_value(R"({"artifact":{"host":"h","path":"/m.bin"},"state":"x"})", "artifact"),
              R"({"host":"h","path":"/m.bin"})");
}

TEST(JsonValueTest, PrefersTheKeyNearestTheTop) {
    // A record's own run_id, not the one in its training history listed first.
    std::string body = R"({"training_history":[{"run_id":"run-01"}],"run_id":"run-02"})";
    EXPECT_EQ(json_value(body, "run_id"), "run-02");
    EXPECT_EQ(json_value(R"({"artifact":{"path":"/m.bin"}})", "path"), "/m.bin");
}

TEST(JsonValueTest, ReturnsNullAsNull) {
    EXPECT_EQ(json_value(R"({"production_model":null})", "production_model"), "null");
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

// TD-452: a missing array is empty; it used to fall back to whichever array came first (an error
// body's, say).
TEST(JsonArrayObjectsTest, ReturnsEmptyWhenTheKeyIsAbsent) {
    std::string body = R"({"entries":[{"x":1},{"x":2}]})";
    EXPECT_TRUE(json_array_objects(body, "does_not_exist").empty());
}

TEST(JsonArrayObjectsTest, AnEmptyKeyReadsATopLevelArray) {
    EXPECT_EQ(json_array_objects(R"([{"a":1},{"b":2}])", "").size(), 2u);
    EXPECT_TRUE(json_array_objects(R"([{"a":1}])", "models").empty());
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

// TD-452: every control character is escaped, so the body stays valid JSON.
TEST(JsonEscapeTest, EscapesEveryControlCharacter) {
    EXPECT_EQ(json_escape("a\tb\rc\x01"
                          "d"),
              "a\\tb\\rc\\u0001d");
    const std::string body = "{\"k\":\"" + json_escape("tab\there\x1f") + "\"}";
    EXPECT_FALSE(nlohmann::json::parse(body, nullptr, false).is_discarded());
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

// TD-452: empty containers stay on one line, the server's field order is kept, and text that
// isn't JSON comes back unchanged.
TEST(JsonPrettyTest, KeepsEmptyContainersAndFieldOrder) {
    EXPECT_EQ(json_pretty(R"({"z":{},"a":[]})"), "{\n  \"z\": {},\n  \"a\": []\n}");
    EXPECT_EQ(json_pretty("not json"), "not json");
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
// build_set_training_body (TD-454)
// ---------------------------------------------------------------------------

TEST(BuildSetTrainingBodyTest, ContinuesTheCurrentRunWithoutARunId) {
    std::string body = build_set_training_body(false);
    EXPECT_EQ(body, R"({"state":"training"})");
    EXPECT_EQ(body.find("\"run_id\""), std::string::npos);  // MNS allocates it
}

TEST(BuildSetTrainingBodyTest, AsksForANewRun) {
    std::string body = build_set_training_body(true);
    EXPECT_EQ(json_value(body, "state"), "training");
    EXPECT_EQ(json_value(body, "new_run"), "true");
    EXPECT_EQ(body.find("\"run_id\""), std::string::npos);
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
