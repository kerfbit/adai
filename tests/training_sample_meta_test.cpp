/**
 * Tests for TrainingSampleMeta.hpp — SampleMeta struct, sample_to_jsonl(),
 * and parse_jsonl_sample().
 */
#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <random>
#include "../external/nlohmann/json.hpp"
#include "../src/TrainingSampleMeta.hpp"

// ============================================================================
// SampleMeta defaults
// ============================================================================

TEST(SampleMetaTest, DefaultFieldValues) {
    SampleMeta m;
    EXPECT_TRUE(m.domain.empty());
    EXPECT_TRUE(m.task_type.empty());
    EXPECT_TRUE(m.language.empty());
    EXPECT_TRUE(m.split.empty());
    EXPECT_LT(m.quality, 0.0f);       // sentinel: not set
    EXPECT_FLOAT_EQ(m.weight, 1.0f);  // default: unweighted
    EXPECT_LT(m.token_count, 0);      // sentinel: not computed
}

TEST(SampleMetaTest, CustomFieldValues) {
    SampleMeta m;
    m.domain = "fiction";
    m.task_type = "qa";
    m.language = "en";
    m.split = "train";
    m.quality = 0.85f;
    m.weight = 2.0f;
    m.token_count = 128;

    EXPECT_EQ(m.domain, "fiction");
    EXPECT_EQ(m.task_type, "qa");
    EXPECT_EQ(m.language, "en");
    EXPECT_EQ(m.split, "train");
    EXPECT_FLOAT_EQ(m.quality, 0.85f);
    EXPECT_FLOAT_EQ(m.weight, 2.0f);
    EXPECT_EQ(m.token_count, 128);
}

// ============================================================================
// sample_to_jsonl
// ============================================================================

TEST(SampleToJsonlTest, MinimalOutputContainsInputAndResponse) {
    const std::string line = sample_to_jsonl("hello", "world");
    EXPECT_NE(line.find("\"input\""), std::string::npos);
    EXPECT_NE(line.find("\"response\""), std::string::npos);
    EXPECT_NE(line.find("hello"), std::string::npos);
    EXPECT_NE(line.find("world"), std::string::npos);
    EXPECT_EQ(line.front(), '{');
    EXPECT_EQ(line.back(), '}');
}

TEST(SampleToJsonlTest, OmitsUnsetOptionalFields) {
    const std::string line = sample_to_jsonl("q", "a");
    // Default SampleMeta: quality<0, weight=1.0, token_count<0, empty strings
    EXPECT_EQ(line.find("\"quality\""), std::string::npos);
    EXPECT_EQ(line.find("\"weight\""), std::string::npos);
    EXPECT_EQ(line.find("\"token_count\""), std::string::npos);
    EXPECT_EQ(line.find("\"domain\""), std::string::npos);
    EXPECT_EQ(line.find("\"task_type\""), std::string::npos);
    EXPECT_EQ(line.find("\"language\""), std::string::npos);
    EXPECT_EQ(line.find("\"split\""), std::string::npos);
}

TEST(SampleToJsonlTest, IncludesAllSetMetadataFields) {
    SampleMeta m;
    m.domain = "code";
    m.task_type = "instruction";
    m.language = "fr";
    m.split = "val";
    m.quality = 0.9f;
    m.weight = 1.5f;
    m.token_count = 64;

    const std::string line = sample_to_jsonl("input text", "output text", m);

    EXPECT_NE(line.find("\"domain\""), std::string::npos);
    EXPECT_NE(line.find("\"task_type\""), std::string::npos);
    EXPECT_NE(line.find("\"language\""), std::string::npos);
    EXPECT_NE(line.find("\"split\""), std::string::npos);
    EXPECT_NE(line.find("\"quality\""), std::string::npos);
    EXPECT_NE(line.find("\"weight\""), std::string::npos);
    EXPECT_NE(line.find("\"token_count\""), std::string::npos);
}

TEST(SampleToJsonlTest, EscapesDoubleQuotes) {
    const std::string line = sample_to_jsonl("say \"hello\"", "ok");
    EXPECT_NE(line.find("\\\"hello\\\""), std::string::npos);
}

TEST(SampleToJsonlTest, EscapesNewline) {
    const std::string line = sample_to_jsonl("line1\nline2", "ok");
    EXPECT_NE(line.find("\\n"), std::string::npos);
    EXPECT_EQ(line.find('\n'), std::string::npos);  // no literal newline
}

TEST(SampleToJsonlTest, EscapesBackslash) {
    const std::string line = sample_to_jsonl("path\\file", "ok");
    EXPECT_NE(line.find("\\\\"), std::string::npos);
}

TEST(SampleToJsonlTest, OmitsQualityWhenZeroExact) {
    // quality=0.0 is a valid set value (not negative sentinel)
    SampleMeta m;
    m.quality = 0.0f;
    const std::string line = sample_to_jsonl("q", "a", m);
    EXPECT_NE(line.find("\"quality\""), std::string::npos);
}

// ============================================================================
// parse_jsonl_sample
// ============================================================================

TEST(ParseJsonlSampleTest, ParsesMinimalLine) {
    const std::string line = R"({"input":"hi","response":"hello"})";
    std::string in, resp;
    SampleMeta meta;
    EXPECT_TRUE(parse_jsonl_sample(line, in, resp, meta));
    EXPECT_EQ(in, "hi");
    EXPECT_EQ(resp, "hello");
}

TEST(ParseJsonlSampleTest, MissingInputReturnsFalse) {
    const std::string line = R"({"response":"hello"})";
    std::string in, resp;
    SampleMeta meta;
    EXPECT_FALSE(parse_jsonl_sample(line, in, resp, meta));
}

TEST(ParseJsonlSampleTest, EmptyLineReturnsFalse) {
    std::string in, resp;
    SampleMeta meta;
    EXPECT_FALSE(parse_jsonl_sample("", in, resp, meta));
}

TEST(ParseJsonlSampleTest, ParsesAllMetadataFields) {
    const std::string line =
        R"({"input":"q","response":"a","domain":"science","task_type":"chat",)"
        R"("language":"de","split":"test","quality":0.75,"weight":2.0,"token_count":42})";
    std::string in, resp;
    SampleMeta meta;
    ASSERT_TRUE(parse_jsonl_sample(line, in, resp, meta));

    EXPECT_EQ(meta.domain, "science");
    EXPECT_EQ(meta.task_type, "chat");
    EXPECT_EQ(meta.language, "de");
    EXPECT_EQ(meta.split, "test");
    EXPECT_NEAR(meta.quality, 0.75f, 1e-5f);
    EXPECT_NEAR(meta.weight, 2.0f, 1e-5f);
    EXPECT_EQ(meta.token_count, 42);
}

TEST(ParseJsonlSampleTest, MissingOptionalFieldsUseSentinels) {
    const std::string line = R"({"input":"only input","response":"only response"})";
    std::string in, resp;
    SampleMeta meta;
    ASSERT_TRUE(parse_jsonl_sample(line, in, resp, meta));

    EXPECT_TRUE(meta.domain.empty());
    EXPECT_TRUE(meta.task_type.empty());
    EXPECT_TRUE(meta.language.empty());
    EXPECT_TRUE(meta.split.empty());
    EXPECT_LT(meta.quality, 0.0f);       // sentinel
    EXPECT_FLOAT_EQ(meta.weight, 1.0f);  // default
    EXPECT_LT(meta.token_count, 0);      // sentinel
}

// ============================================================================
// TD-531: escape decoding; TD-532: top-level keys only
// ============================================================================

namespace {
std::string parse_input(const std::string& line) {
    std::string in, resp;
    SampleMeta meta;
    parse_jsonl_sample(line, in, resp, meta);
    return in;
}
}  // namespace

// The \u escapes below are in ordinary string literals with a doubled backslash on purpose: GCC
// turns \uXXXX inside a raw string literal into the character itself at compile time, so the
// decoder would never see an escape.
TEST(ParseJsonlSampleTest, DecodesUnicodeEscapes) {
    // Python json.dumps / pandas to_json escape non-ASCII by default.
    EXPECT_EQ(parse_input("{\"input\":\"Caf\\u00e9 au lait?\",\"response\":\"x\"}"),
              "Caf\xC3\xA9 au lait?");
    EXPECT_EQ(parse_input("{\"input\":\"don\\u2019t\",\"response\":\"x\"}"),
              "don\xE2\x80\x99t");
    EXPECT_EQ(parse_input("{\"input\":\"\\u00E9\\u00e9\",\"response\":\"x\"}"),
              "\xC3\xA9\xC3\xA9");
}

TEST(ParseJsonlSampleTest, DecodesSurrogatePairs) {
    EXPECT_EQ(parse_input("{\"input\":\"smile \\ud83d\\ude00!\",\"response\":\"x\"}"),
              "smile \xF0\x9F\x98\x80!");
}

TEST(ParseJsonlSampleTest, UnpairedSurrogatesBecomeReplacementCharacter) {
    EXPECT_EQ(parse_input("{\"input\":\"a\\ud83db\",\"response\":\"x\"}"),
              "a\xEF\xBF\xBD" "b");
    EXPECT_EQ(parse_input("{\"input\":\"a\\ude00b\",\"response\":\"x\"}"),
              "a\xEF\xBF\xBD" "b");
}

TEST(ParseJsonlSampleTest, DecodesEveryStandardEscape) {
    EXPECT_EQ(parse_input(R"({"input":"q\" b\\ s\/ \b\f\n\r\t","response":"x"})"),
              "q\" b\\ s/ \b\f\n\r\t");
}

TEST(ParseJsonlSampleTest, InvalidUnicodeEscapeIsKeptLiterally) {
    EXPECT_EQ(parse_input("{\"input\":\"bad \\u12G4 end\",\"response\":\"x\"}"),
              "bad \\u12G4 end");
}

TEST(ParseJsonlSampleTest, ReadsTopLevelKeysNotNestedOnes) {
    const std::string line =
        R"({"meta":{"input":"WRONG","response":"WRONG","quality":0.1},"input":"right",)"
        R"("tags":["input","response"],"response":"also right","quality":0.9})";
    std::string in, resp;
    SampleMeta meta;
    ASSERT_TRUE(parse_jsonl_sample(line, in, resp, meta));
    EXPECT_EQ(in, "right");
    EXPECT_EQ(resp, "also right");
    EXPECT_NEAR(meta.quality, 0.9f, 1e-6f);
}

TEST(ParseJsonlSampleTest, KeyTextInsideAValueIsNotAKey) {
    const std::string line = R"({"note":"\"input\":\"WRONG\"","input":"right","response":"r"})";
    EXPECT_EQ(parse_input(line), "right");
}

TEST(ParseJsonlSampleTest, ToleratesWhitespaceAndRawControlCharacters) {
    // sample_to_jsonl() writes other control characters raw (TD-533); keep reading them.
    const std::string line = "  { \"input\" :\t\"a\x01" "b\" , \"response\" : \"c\" }  ";
    std::string in, resp;
    SampleMeta meta;
    ASSERT_TRUE(parse_jsonl_sample(line, in, resp, meta));
    EXPECT_EQ(in, "a\x01" "b");
    EXPECT_EQ(resp, "c");
}

TEST(ParseJsonlSampleTest, RepeatedKeyKeepsLastValue) {
    EXPECT_EQ(parse_input(R"({"input":"first","input":"second","response":"r"})"), "second");
}

TEST(ParseJsonlSampleTest, NonStringInputIsRejected) {
    EXPECT_EQ(parse_input(R"({"input":42,"response":"r"})"), "");
    EXPECT_EQ(parse_input(R"({"input":{"text":"x"},"response":"r"})"), "");
}

TEST(ParseJsonlSampleTest, MalformedLineKeepsMembersBeforeTheError) {
    std::string in, resp;
    SampleMeta meta;
    EXPECT_TRUE(parse_jsonl_sample(R"({"input":"kept","response":"r", oops})", in, resp, meta));
    EXPECT_EQ(in, "kept");
    EXPECT_EQ(resp, "r");
    EXPECT_FALSE(parse_jsonl_sample(R"({"input":"unterminated)", in, resp, meta));
    EXPECT_FALSE(parse_jsonl_sample("[\"input\",\"x\"]", in, resp, meta));
}

// Differential check against the vendored nlohmann/json on standard (ensure_ascii-style) JSON:
// random strings over ASCII, accented letters, CJK and emoji, serialized with every non-ASCII
// character escaped, must decode to the same text.
TEST(ParseJsonlSampleTest, MatchesNlohmannOnEscapedUnicode) {
    const std::vector<std::string> pieces = {
        "a", "Z", " ", "\"", "\\", "/", "\n", "\t", "\xC3\xA9", "\xC3\xB1", "\xE2\x80\x94",
        "\xE2\x80\x99", "\xE4\xB8\xAD", "\xF0\x9F\x98\x80", "\xF0\x9F\x91\x8D", "{", "}", ":", ","};
    std::mt19937 rng(531);
    std::uniform_int_distribution<size_t> pick(0, pieces.size() - 1), len(0, 40);
    for (int trial = 0; trial < 500; ++trial) {
        std::string input, response;
        for (size_t n = len(rng); n > 0; --n) input += pieces[pick(rng)];
        for (size_t n = len(rng); n > 0; --n) response += pieces[pick(rng)];
        if (input.empty()) input = "x";
        nlohmann::json obj;
        obj["meta"] = {{"input", "nested"}};
        obj["input"] = input;
        obj["response"] = response;
        const std::string line = obj.dump(-1, ' ', /*ensure_ascii=*/true);

        std::string in, resp;
        SampleMeta meta;
        ASSERT_TRUE(parse_jsonl_sample(line, in, resp, meta)) << line;
        const auto ref = nlohmann::json::parse(line);
        ASSERT_EQ(in, ref["input"].get<std::string>()) << line;
        ASSERT_EQ(resp, ref["response"].get<std::string>()) << line;
    }
}

// ============================================================================
// Roundtrip: serialize → parse
// ============================================================================

TEST(RoundtripTest, MinimalRoundtrip) {
    const std::string serialized = sample_to_jsonl("What is AI?", "Artificial intelligence.");
    std::string in, resp;
    SampleMeta meta;
    ASSERT_TRUE(parse_jsonl_sample(serialized, in, resp, meta));
    EXPECT_EQ(in, "What is AI?");
    EXPECT_EQ(resp, "Artificial intelligence.");
}

TEST(RoundtripTest, FullMetadataRoundtrip) {
    SampleMeta orig;
    orig.domain = "dialogue";
    orig.task_type = "instruction";
    orig.language = "en";
    orig.split = "train";
    orig.quality = 0.95f;
    orig.weight = 1.2f;
    orig.token_count = 200;

    const std::string serialized = sample_to_jsonl("instruct me", "done", orig);

    std::string in, resp;
    SampleMeta parsed;
    ASSERT_TRUE(parse_jsonl_sample(serialized, in, resp, parsed));

    EXPECT_EQ(in, "instruct me");
    EXPECT_EQ(resp, "done");
    EXPECT_EQ(parsed.domain, orig.domain);
    EXPECT_EQ(parsed.task_type, orig.task_type);
    EXPECT_EQ(parsed.language, orig.language);
    EXPECT_EQ(parsed.split, orig.split);
    EXPECT_NEAR(parsed.quality, orig.quality, 1e-4f);
    EXPECT_NEAR(parsed.weight, orig.weight, 1e-4f);
    EXPECT_EQ(parsed.token_count, orig.token_count);
}

TEST(RoundtripTest, EscapedCharactersRoundtrip) {
    const std::string input = "say \"hello\" and\nnewline\\path";
    const std::string response = "ok\t tab";
    const std::string serialized = sample_to_jsonl(input, response);

    std::string in, resp;
    SampleMeta meta;
    ASSERT_TRUE(parse_jsonl_sample(serialized, in, resp, meta));
    EXPECT_EQ(in, input);
    EXPECT_EQ(resp, response);
}

// ============================================================================
// Quality score conversion (exp(-loss) formula)
// ============================================================================

TEST(QualityConversionTest, ZeroLossGivesQualityOne) {
    float loss = 0.0f;
    float quality = std::exp(-loss);
    EXPECT_FLOAT_EQ(quality, 1.0f);
}

TEST(QualityConversionTest, HighLossGivesLowQuality) {
    float quality_low = std::exp(-5.0f);
    float quality_high = std::exp(-0.5f);
    EXPECT_LT(quality_low, quality_high);
    EXPECT_GT(quality_low, 0.0f);
    EXPECT_LT(quality_high, 1.0f);
}

TEST(QualityConversionTest, QualityIsInRange) {
    for (float loss : {0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f}) {
        float q = std::exp(-loss);
        EXPECT_GT(q, 0.0f);
        EXPECT_LE(q, 1.0f);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
