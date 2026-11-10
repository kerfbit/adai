/**
 * Unit tests for SpecialTokens.hpp.
 *
 * Ported from the orphaned tests/test_special_tokens_header.cpp (a plain-assert,
 * standalone-main file that was never registered in CMakeLists.txt and so never
 * ran as part of the suite) into proper GTest cases, matching this repo's
 * convention. Same scenarios, same coverage — now actually wired into ctest.
 *
 * TD-505: the tests of the removed lookup, set/map and validate() helpers went with them.
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

TEST(SpecialTokensTest, IsStopTokenOnlyEosAndPad) {
    SpecialTokenConfig config;
    EXPECT_TRUE(is_stop_token(3, config));   // EOS
    EXPECT_TRUE(is_stop_token(0, config));   // PAD
    EXPECT_FALSE(is_stop_token(1, config));  // UNK is not a stop token
    EXPECT_FALSE(is_stop_token(2, config));  // BOS is not a stop token
}

TEST(SpecialTokensTest, IsStopTokenRespectsCustomConfig) {
    SpecialTokenConfig custom(100, 101, 102, 103);
    EXPECT_TRUE(is_stop_token(103, custom));   // custom EOS
    EXPECT_TRUE(is_stop_token(100, custom));   // custom PAD
    EXPECT_FALSE(is_stop_token(3, custom));    // default EOS id isn't a stop token under custom config
}

// TD-565: the reserved block and its placeholder names.
TEST(SpecialTokensTest, ReservedSlotsFollowTheTurnTokens) {
    EXPECT_EQ(SpecialTokenIDs::FIRST_UNASSIGNED, SpecialTokenIDs::ASSISTANT_TURN + 1);
    EXPECT_EQ(SpecialTokenIDs::RESERVED_SLOTS, 20);
    EXPECT_EQ(reserved_slot_token(7), "<reserved_7>");
    EXPECT_EQ(reserved_slot_token(19), "<reserved_19>");
}
