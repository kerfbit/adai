/**
 * @file init_rng_test.cpp
 * @brief TD-368: a seed replays weight initialization and the trainer's random choices.
 */
#include <gtest/gtest.h>

#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../src/Dropout.hpp"
#include "../src/EncoderDecoderModel.hpp"
#include "../src/FeedForward.hpp"
#include "../src/InitRng.hpp"
#include "../src/Matrix.hpp"
#include "../src/SIGReg.hpp"

namespace {

Matrix randomized(int seed) {
    adai::set_init_seed(seed);
    Matrix m(16, 16);
    m.randomize(1.0f);
    return m;
}

Matrix model_output(int seed) {
    adai::set_init_seed(seed);
    EncoderDecoderModel model(40, 16, 1, 1, 2, 32, 16);
    model.set_training(false);
    return model.forward({4, 5, 6, 7}, {8, 9, 10});
}

Matrix feed_forward_output(int seed) {
    adai::set_init_seed(seed);
    FeedForward layer(8, 16);
    Matrix input(3, 8);
    for (auto& row : input.data) {
        for (auto& x : row) {
            x = 0.5f;
        }
    }
    return layer.forward(input);
}

Matrix dropped(int seed) {
    adai::set_init_seed(seed);
    Dropout d;
    d.set_rate(0.5f);
    d.set_active(true);
    Matrix x(16, 16);
    for (auto& row : x.data) {
        for (auto& v : row) {
            v = 1.0f;
        }
    }
    d.forward(x);
    return x;
}

}  // namespace

TEST(InitRngTest, SameSeedSameSequence) {
    adai::set_init_seed(5);
    const auto a = adai::next_init_seed(), b = adai::next_init_seed();
    EXPECT_NE(a, b);
    adai::set_init_seed(5);
    EXPECT_EQ(adai::next_init_seed(), a);
    EXPECT_EQ(adai::next_init_seed(), b);
    adai::set_init_seed(6);
    EXPECT_NE(adai::next_init_seed(), a);
}

TEST(InitRngTest, RejectsSeedsBelowOne) {
    EXPECT_THROW(adai::set_init_seed(0), std::invalid_argument);
    EXPECT_THROW(adai::set_init_seed(-3), std::invalid_argument);
}

TEST(InitRngTest, MatrixRandomizeFollowsTheSeed) {
    EXPECT_EQ(randomized(11).data, randomized(11).data);
    EXPECT_NE(randomized(11).data, randomized(12).data);
}

TEST(InitRngTest, ConsecutiveMatricesDiffer) {
    adai::set_init_seed(3);
    Matrix a(8, 8), b(8, 8);
    a.randomize(1.0f);
    b.randomize(1.0f);
    EXPECT_NE(a.data, b.data);
}

TEST(InitRngTest, FeedForwardFollowsTheSeed) {
    EXPECT_EQ(feed_forward_output(21).data, feed_forward_output(21).data);
    EXPECT_NE(feed_forward_output(21).data, feed_forward_output(22).data);
}

TEST(InitRngTest, WholeModelFollowsTheSeed) {
    EXPECT_EQ(model_output(31).data, model_output(31).data);
    EXPECT_NE(model_output(31).data, model_output(32).data);
}

TEST(InitRngTest, DropoutMasksFollowTheSeed) {
    EXPECT_EQ(dropped(41).data, dropped(41).data);
    EXPECT_NE(dropped(41).data, dropped(42).data);
}

TEST(InitRngTest, SigregDirectionsFollowTheSeed) {
    const auto directions = [](int seed) {
        adai::set_init_seed(seed);
        return SIGReg(16, 8).get_directions().data;
    };
    EXPECT_EQ(directions(51), directions(51));
    EXPECT_NE(directions(51), directions(52));
}

TEST(InitRngTest, ThreadsNeverGetTheSameStateTwice) {
    adai::set_init_seed(7);
    std::vector<std::uint32_t> values(8000);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&values, t] {
            for (int i = 0; i < 1000; ++i) {
                values[t * 1000 + i] = adai::next_init_seed();
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_GT(std::set<std::uint32_t>(values.begin(), values.end()).size(), 7990u);
}
