// TD-233: SecureRandom draws from the OS CSPRNG.

#include "../src/SecureRandom.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <set>
#include <string>
#include <vector>

TEST(SecureRandomTest, HexHasTwoDigitsPerByte) {
    for (std::size_t bytes : {0u, 1u, 16u, 33u}) {
        const std::string s = adai::secure_random_hex(bytes);
        EXPECT_EQ(s.size(), 2 * bytes);
        EXPECT_EQ(s.find_first_not_of("0123456789abcdef"), std::string::npos) << s;
    }
}

TEST(SecureRandomTest, DrawsDoNotRepeat) {
    std::set<std::string> seen;
    for (int i = 0; i < 10000; ++i) seen.insert(adai::secure_random_hex(16));
    EXPECT_EQ(seen.size(), 10000u);
}

TEST(SecureRandomTest, LargeBufferIsFilledAndRoughlyUniform) {
    std::vector<unsigned char> buf(1 << 20, 0);
    adai::secure_random_bytes(buf.data(), buf.size());
    std::array<double, 256> counts{};
    for (unsigned char b : buf) counts[b] += 1;
    // Chi-square with 255 degrees of freedom: mean 255, sd ~22.6. 400 is over 6 sd out.
    const double expected = static_cast<double>(buf.size()) / 256.0;
    double chi2 = 0;
    for (double c : counts) chi2 += (c - expected) * (c - expected) / expected;
    EXPECT_LT(chi2, 400.0);
    EXPECT_GT(chi2, 130.0);  // and not suspiciously flat
}
