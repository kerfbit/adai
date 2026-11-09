/**
 * @file device_selection_test.cpp
 * @brief TD-391: one card is listed once per SYCL runtime, so GPU_DEVICE_ID must not depend on
 * which runtime enumerates first.
 */
#include <gtest/gtest.h>

#include <vector>

#include "../src/gpu/DeviceSelection.hpp"

using adai::gpu::DeviceCandidate;
using adai::gpu::select_devices;

namespace {
const DeviceCandidate kArcZe{"level_zero", "Intel", "Arc B60"};
const DeviceCandidate kArcCl{"opencl", "Intel", "Arc B60"};
const DeviceCandidate kCpuCl{"opencl", "Intel", "Core i5"};
using Indices = std::vector<std::size_t>;
}  // namespace

TEST(DeviceSelectionTest, OneCardOnTwoRuntimesIsOneLevelZeroDevice) {
    EXPECT_EQ(select_devices({kArcZe, kArcCl}), Indices({0}));
    EXPECT_EQ(select_devices({kArcCl, kArcZe}), Indices({1})) << "whichever enumerates first";
}

TEST(DeviceSelectionTest, OpenCLAloneIsKept) {
    EXPECT_EQ(select_devices({kArcCl}), Indices({0}));
}

TEST(DeviceSelectionTest, TwoIdenticalCardsStayTwo) {
    EXPECT_EQ(select_devices({kArcCl, kArcCl, kArcZe, kArcZe}), Indices({2, 3}));
    EXPECT_EQ(select_devices({kArcZe, kArcZe}), Indices({0, 1})) << "same backend: two cards";
}

TEST(DeviceSelectionTest, ACardOnlyOpenCLListsSurvives) {
    EXPECT_EQ(select_devices({kArcCl, kArcCl, kArcZe}), Indices({2, 1}));
}

TEST(DeviceSelectionTest, DifferentDevicesAreNotDuplicates) {
    EXPECT_EQ(select_devices({kCpuCl, kArcZe, kArcCl}), Indices({1, 0}))
        << "the CPU device (tests only) follows the GPU";
    const DeviceCandidate renamed{"opencl", "Intel", "Arc B60 (OpenCL)"};
    EXPECT_EQ(select_devices({renamed, kArcZe}), Indices({1, 0})) << "id 0 is still Level Zero";
}

TEST(DeviceSelectionTest, OtherBackendsRankBetween) {
    const DeviceCandidate other{"other", "NV", "X"};
    EXPECT_EQ(select_devices({kCpuCl, other, kArcZe}), Indices({2, 1, 0}));
    EXPECT_TRUE(select_devices({}).empty());
}
