#include <gtest/gtest.h>
#include <stdexcept>
#include "KVCache.hpp"

// TD-421/TD-422: direct tests of KVCache and DecoderKVCache.

namespace {
Matrix filled(int rows, int cols, float base) {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = base + static_cast<float>(i * cols + j);
    return m;
}
}  // namespace

TEST(KVCache, AppendKeepsEveryRowInOrder) {
    KVCache cache;
    cache.append(filled(2, 4, 0.0f), filled(2, 4, 100.0f));
    for (int step = 0; step < 5; ++step)
        cache.append(filled(1, 4, 10.0f * (step + 1)), filled(1, 4, 1000.0f + step));
    ASSERT_EQ(cache.size(), 7);
    ASSERT_EQ(cache.get_keys().rows, 7);
    ASSERT_EQ(cache.get_values().rows, 7);
    ASSERT_EQ(static_cast<int>(cache.get_keys().data.size()), 7);
    EXPECT_FLOAT_EQ(cache.get_keys()(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(cache.get_keys()(1, 3), 7.0f);
    EXPECT_FLOAT_EQ(cache.get_keys()(2, 0), 10.0f);
    EXPECT_FLOAT_EQ(cache.get_keys()(6, 3), 53.0f);
    EXPECT_FLOAT_EQ(cache.get_values()(6, 0), 1004.0f);

    cache.clear();
    EXPECT_TRUE(cache.is_empty());
    cache.append(filled(1, 8, 0.0f), filled(1, 8, 0.0f));  // a cleared cache takes a new width
    EXPECT_EQ(cache.get_keys().cols, 8);
}

// TD-421: shapes are checked. These used to overflow the heap (ASan-verified) or be accepted.
TEST(KVCache, RejectsMismatchedShapes) {
    KVCache cache;
    EXPECT_THROW(cache.append(filled(3, 4, 0), filled(1, 8, 0)), std::invalid_argument)
        << "inconsistent first append";
    EXPECT_TRUE(cache.is_empty());
    cache.append(filled(1, 4, 0), filled(1, 4, 0));
    EXPECT_THROW(cache.append(filled(1, 8, 0), filled(1, 8, 0)), std::invalid_argument)
        << "wider rows (heap write overflow)";
    EXPECT_THROW(cache.append(filled(3, 4, 0), filled(1, 4, 0)), std::invalid_argument)
        << "fewer value rows than key rows (heap read overflow)";
    EXPECT_EQ(cache.size(), 1) << "a rejected append changes nothing";
}

TEST(DecoderKVCache, LayerIndicesAreChecked) {
    DecoderKVCache cache(2);
    EXPECT_NO_THROW(cache.get_self_attention_cache(1));
    EXPECT_NO_THROW(cache.get_cross_attention_cache(0));
    EXPECT_THROW(cache.get_self_attention_cache(2), std::out_of_range);
    EXPECT_THROW(cache.get_self_attention_cache(-1), std::out_of_range);
    EXPECT_THROW(cache.get_cross_attention_cache(5), std::out_of_range);
}

// TD-422: appending grows in place; rows already cached keep their storage (nothing is copied).
TEST(KVCache, AppendDoesNotCopyCachedRows) {
    KVCache cache;
    cache.append(filled(1, 64, 0), filled(1, 64, 0));
    const float* first_row = cache.get_keys().data[0].data();
    for (int step = 0; step < 300; ++step)
        cache.append(filled(1, 64, 1), filled(1, 64, 1));
    EXPECT_EQ(cache.get_keys().data[0].data(), first_row) << "the first row's buffer was reallocated";
    EXPECT_EQ(cache.size(), 301);
}
