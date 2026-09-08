/**
 * Unit tests for SpecialTokens.hpp.
 *
 * Ported from the orphaned tests/test_special_tokens_header.cpp (a plain-assert,
 * standalone-main file that was never registered in CMakeLists.txt and so never
 * ran as part of the suite) into proper GTest cases, matching this repo's
 * convention. Same scenarios, same coverage — now actually wired into ctest.
 */
#include <gtest/gtest.h>
#include "../src/SpecialTokens.hpp"

using namespace adai;

TEST(SpecialTokensTest, StandardTokenIdConstants) {
    EXPECT_EQ(SpecialTokenIDs::PAD, 0);
    EXPECT_EQ(SpecialTokenIDs::UNK, 1);
    EXPECT_EQ(SpecialTokenIDs::BOS, 2);
    EXPECT_EQ(SpecialTokenIDs::EOS, 3);
}

TEST(SpecialTokensTest, StandardTokenStringConstants) {
    EXPECT_STREQ(SpecialTokenStrings::PAD, "<pad>");
    EXPECT_STREQ(SpecialTokenStrings::UNK, "<unk>");
    EXPECT_STREQ(SpecialTokenStrings::BOS, "<bos>");
    EXPECT_STREQ(SpecialTokenStrings::EOS, "<eos>");
}

TEST(SpecialTokenConfigTest, DefaultConstructorUsesStandardIds) {
    SpecialTokenConfig config;
    EXPECT_EQ(config.get_pad_token_id(), 0);
    EXPECT_EQ(config.get_unk_token_id(), 1);
    EXPECT_EQ(config.get_bos_token_id(), 2);
    EXPECT_EQ(config.get_eos_token_id(), 3);
}

TEST(SpecialTokenConfigTest, CustomConstructorStoresGivenIds) {
    SpecialTokenConfig custom(10, 11, 12, 13);
    EXPECT_EQ(custom.get_pad_token_id(), 10);
    EXPECT_EQ(custom.get_unk_token_id(), 11);
    EXPECT_EQ(custom.get_bos_token_id(), 12);
    EXPECT_EQ(custom.get_eos_token_id(), 13);
}

TEST(SpecialTokenConfigTest, ValidateAcceptsDefaultConfig) {
    SpecialTokenConfig valid;
    EXPECT_NO_THROW(valid.validate());
}

TEST(SpecialTokenConfigTest, ValidateRejectsNegativeId) {
    SpecialTokenConfig invalid(-1, 1, 2, 3);
    EXPECT_THROW(invalid.validate(), std::invalid_argument);
}

TEST(SpecialTokenConfigTest, ValidateRejectsDuplicateIds) {
    SpecialTokenConfig invalid(0, 0, 2, 3);
    EXPECT_THROW(invalid.validate(), std::invalid_argument);
}

TEST(SpecialTokensTest, IsSpecialTokenRecognizesAllFour) {
    SpecialTokenConfig config;
    EXPECT_TRUE(is_special_token(0, config));   // PAD
    EXPECT_TRUE(is_special_token(1, config));   // UNK
    EXPECT_TRUE(is_special_token(2, config));   // BOS
    EXPECT_TRUE(is_special_token(3, config));   // EOS
    EXPECT_FALSE(is_special_token(4, config));  // ordinary vocab id
}

TEST(SpecialTokensTest, IsSpecialTokenStringRecognizesAllFour) {
    EXPECT_TRUE(is_special_token_string("<pad>"));
    EXPECT_TRUE(is_special_token_string("<unk>"));
    EXPECT_TRUE(is_special_token_string("<bos>"));
    EXPECT_TRUE(is_special_token_string("<eos>"));
    EXPECT_FALSE(is_special_token_string("hello"));
}

TEST(SpecialTokensTest, IsStopTokenOnlyEosAndPad) {
    SpecialTokenConfig config;
    EXPECT_TRUE(is_stop_token(3, config));   // EOS
    EXPECT_TRUE(is_stop_token(0, config));   // PAD
    EXPECT_FALSE(is_stop_token(1, config));  // UNK is not a stop token
    EXPECT_FALSE(is_stop_token(2, config));  // BOS is not a stop token
}

TEST(SpecialTokensTest, GetSpecialTokenStringRoundTrips) {
    SpecialTokenConfig config;
    EXPECT_EQ(get_special_token_string(0, config), "<pad>");
    EXPECT_EQ(get_special_token_string(1, config), "<unk>");
    EXPECT_EQ(get_special_token_string(2, config), "<bos>");
    EXPECT_EQ(get_special_token_string(3, config), "<eos>");
}

TEST(SpecialTokensTest, GetSpecialTokenIdRoundTrips) {
    SpecialTokenConfig config;
    EXPECT_EQ(get_special_token_id("<pad>", config), 0);
    EXPECT_EQ(get_special_token_id("<unk>", config), 1);
    EXPECT_EQ(get_special_token_id("<bos>", config), 2);
    EXPECT_EQ(get_special_token_id("<eos>", config), 3);
}

TEST(SpecialTokensTest, GetSpecialTokenStringThrowsForUnknownId) {
    SpecialTokenConfig config;
    EXPECT_THROW(get_special_token_string(999, config), std::invalid_argument);
}

TEST(SpecialTokensTest, GetSpecialTokenIdThrowsForUnknownString) {
    SpecialTokenConfig config;
    EXPECT_THROW(get_special_token_id("invalid", config), std::invalid_argument);
}

TEST(SpecialTokensTest, CreateSpecialTokenSetContainsAllFourStrings) {
    auto token_set = create_special_token_set();
    EXPECT_EQ(token_set.size(), 4u);
    EXPECT_EQ(token_set.count("<pad>"), 1u);
    EXPECT_EQ(token_set.count("<unk>"), 1u);
    EXPECT_EQ(token_set.count("<bos>"), 1u);
    EXPECT_EQ(token_set.count("<eos>"), 1u);
}

TEST(SpecialTokensTest, CreateSpecialTokenMapMapsStringsToIds) {
    SpecialTokenConfig config;
    auto token_map = create_special_token_map(config);
    EXPECT_EQ(token_map.size(), 4u);
    EXPECT_EQ(token_map["<pad>"], 0);
    EXPECT_EQ(token_map["<unk>"], 1);
    EXPECT_EQ(token_map["<bos>"], 2);
    EXPECT_EQ(token_map["<eos>"], 3);
}

TEST(SpecialTokensTest, CreateInverseSpecialTokenMapMapsIdsToStrings) {
    SpecialTokenConfig config;
    auto inverse_map = create_inverse_special_token_map(config);
    EXPECT_EQ(inverse_map.size(), 4u);
    EXPECT_EQ(inverse_map[0], "<pad>");
    EXPECT_EQ(inverse_map[1], "<unk>");
    EXPECT_EQ(inverse_map[2], "<bos>");
    EXPECT_EQ(inverse_map[3], "<eos>");
}

TEST(SpecialTokensTest, UtilityFunctionsRespectCustomConfig) {
    SpecialTokenConfig custom(100, 101, 102, 103);

    EXPECT_TRUE(is_special_token(100, custom));
    EXPECT_TRUE(is_special_token(102, custom));
    EXPECT_FALSE(is_special_token(0, custom));  // default PAD id isn't special under custom config

    EXPECT_TRUE(is_stop_token(103, custom));   // custom EOS
    EXPECT_TRUE(is_stop_token(100, custom));   // custom PAD
    EXPECT_FALSE(is_stop_token(3, custom));    // default EOS id isn't a stop token under custom config

    EXPECT_EQ(get_special_token_string(102, custom), "<bos>");
    EXPECT_EQ(get_special_token_id("<eos>", custom), 103);
}
