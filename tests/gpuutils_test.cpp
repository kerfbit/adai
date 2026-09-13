// GPUManager/GPUMemory tests (TD-041). Compiled only when ENABLE_GPU or ENABLE_SYCL is
// configured (see tests/CMakeLists.txt) -- exercises whichever backend GPUUtils.hpp resolves
// to (CUDA or SYCL), which share an identical GPUManager/GPUMemory public interface, so no
// backend-specific code is needed here (same convention as matrixgpu_td003_test.cpp for
// GPUMatrix).
//
// Adaptive to whether real GPU hardware is actually present at runtime, rather than assuming it
// the way matrixgpu_td003_test.cpp does: this repo's ENABLE_GPU/ENABLE_SYCL presets are
// typically only built on a machine known to have a device, but nothing here should spuriously
// fail on a CI runner or dev sandbox that only has the compiler toolchain installed (confirmed
// via this exact scenario while filing this test -- matrixgpu_td003_test.cpp's own tests fail
// outright in such an environment, e.g. "cudaMalloc failed: no CUDA-capable device is
// detected"; flagged separately as a follow-up, out of TD-041's own stated scope). When
// GPUManager::probe() reports no device, tests assert the documented soft-fail contract instead
// of exercising real device I/O -- both are real, valuable things to verify, just not always at
// the same time on the same machine.
//
// get_device_info() called before any successful initialize() is deliberately NOT tested here:
// confirmed to behave differently per backend in that state (CUDA's cudaGetDeviceProperties()
// call throws std::runtime_error when passed the still-default current_device_ of -1; SYCL's
// own bounds check catches the same -1 first and returns an "Invalid device ID" string instead
// of throwing) -- a real, minor API-consistency gap, but reconciling it is out of scope for a
// test-coverage item, so it's left untested here rather than baking a backend-specific branch
// into an otherwise backend-agnostic file. See TECHNICAL_DEBT.md if this needs a proper item.

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>
#include "../src/gpu/GPUUtils.hpp"

using adai::gpu::GPUManager;
using adai::gpu::GPUMemory;

namespace {
// GPUManager keeps all of its state in `inline static` members shared process-wide (no way to
// fully reconstruct a fresh instance), so every test must leave it the way it found it.
class GPUManagerTest : public ::testing::Test {
   protected:
    void TearDown() override {
        GPUManager::cleanup();
    }
};
}  // namespace

TEST_F(GPUManagerTest, ProbeDoesNotThrow) {
    EXPECT_NO_THROW({ GPUManager::probe(); });
}

TEST_F(GPUManagerTest, ProbeDiagnosticIsNonEmpty) {
    EXPECT_FALSE(GPUManager::probe_diagnostic().empty());
}

TEST_F(GPUManagerTest, IsAvailableFalseBeforeInitialize) {
    EXPECT_FALSE(GPUManager::is_available());
}

TEST_F(GPUManagerTest, InitializeMatchesProbe) {
    // The documented soft-fail contract: initialize() returns exactly what probe() predicted,
    // and never throws merely for "no device present" (only for invalid arguments once a device
    // has actually been found).
    const bool device_present = GPUManager::probe();
    bool init_result = false;
    EXPECT_NO_THROW({ init_result = GPUManager::initialize(); });
    EXPECT_EQ(init_result, device_present);
    EXPECT_EQ(GPUManager::is_available(), device_present);
}

TEST_F(GPUManagerTest, DeviceCountMatchesProbe) {
    const bool device_present = GPUManager::probe();
    GPUManager::initialize();
    if (device_present) {
        EXPECT_GT(GPUManager::device_count(), 0);
    } else {
        EXPECT_EQ(GPUManager::device_count(), 0);
    }
}

TEST_F(GPUManagerTest, SetDeviceThrowsWhenIndexOutOfRange) {
    // device_count_ is never reset by cleanup() (only allocated_bytes_/initialized_ are), so an
    // earlier test in this binary may have already populated it via a real initialize() call --
    // test against an index guaranteed out of range for whatever the *current* count is, rather
    // than assuming zero devices.
    const int out_of_range = GPUManager::device_count() + 100;
    EXPECT_THROW(GPUManager::set_device(out_of_range), std::out_of_range);
    EXPECT_THROW(GPUManager::set_device(-1), std::out_of_range);
}

TEST_F(GPUManagerTest, MemoryGettersAreZeroWhenNeverInitialized) {
    if (GPUManager::probe()) {
        GTEST_SKIP() << "Real GPU present; an earlier test in this binary may already have "
                        "initialized GPUManager and set a nonzero memory budget that "
                        "cleanup() deliberately does not reset (only allocated_bytes_ is) -- "
                        "this assertion is only reliable in a virgin, never-initialized process.";
    }
    EXPECT_EQ(GPUManager::get_memory_limit_bytes(), 0u);
    EXPECT_EQ(GPUManager::get_used_memory_bytes(), 0u);
    EXPECT_EQ(GPUManager::get_available_memory_bytes(), 0u);
}

TEST_F(GPUManagerTest, SynchronizeDoesNotThrowBeforeInitialize) {
    EXPECT_NO_THROW(GPUManager::synchronize());
}

TEST_F(GPUManagerTest, InvalidDeviceIdThrowsOnceADeviceExists) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present; initialize()'s device_id validation is only "
                        "reached once device_count() > 0.";
    }
    EXPECT_THROW(GPUManager::initialize(GPUManager::device_count() + 10), std::out_of_range);
}

TEST_F(GPUManagerTest, InvalidMemoryFractionThrowsOnceADeviceExists) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present; initialize()'s memory_fraction validation is "
                        "only reached once device_count() > 0.";
    }
    EXPECT_THROW(GPUManager::initialize(0, 0.0f), std::invalid_argument);
    EXPECT_THROW(GPUManager::initialize(0, 1.5f), std::invalid_argument);
}

// ============================================================================
// Real device init / allocation / copy -- only meaningful with actual hardware.
// ============================================================================

TEST_F(GPUManagerTest, RealDeviceInitializeAndQuery) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f));
    EXPECT_TRUE(GPUManager::is_available());
    EXPECT_EQ(GPUManager::current_device(), 0);
    EXPECT_GT(GPUManager::get_memory_limit_bytes(), 0u);
    EXPECT_FALSE(GPUManager::get_device_info(0).empty());
    EXPECT_NO_THROW(GPUManager::synchronize());
}

TEST_F(GPUManagerTest, ReserveAndReleaseMemoryTracksBudget) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f));
    const size_t before = GPUManager::get_used_memory_bytes();
    GPUManager::reserve_memory(1024);
    EXPECT_EQ(GPUManager::get_used_memory_bytes(), before + 1024);
    GPUManager::release_memory(1024);
    EXPECT_EQ(GPUManager::get_used_memory_bytes(), before);
}

TEST_F(GPUManagerTest, ReserveMemoryThrowsWhenBudgetExceeded) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f));
    EXPECT_THROW(GPUManager::reserve_memory(GPUManager::get_memory_limit_bytes() + 1),
                std::runtime_error);
}

TEST_F(GPUManagerTest, GPUMemoryRoundTripsHostData) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f));

    const std::vector<float> host_in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    GPUMemory<float> mem(host_in.size());
    EXPECT_EQ(mem.size(), host_in.size());

    mem.copy_from_host(host_in.data(), host_in.size());
    GPUManager::synchronize();

    std::vector<float> host_out(host_in.size(), 0.0f);
    mem.copy_to_host(host_out.data(), host_out.size());

    EXPECT_EQ(host_out, host_in);
}

TEST_F(GPUManagerTest, GPUMemoryTracksAllocationInBudget) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f));

    const size_t before = GPUManager::get_used_memory_bytes();
    {
        GPUMemory<float> mem(256);
        EXPECT_EQ(GPUManager::get_used_memory_bytes(), before + 256 * sizeof(float));
    }
    // Destructor released it back.
    EXPECT_EQ(GPUManager::get_used_memory_bytes(), before);
}

TEST_F(GPUManagerTest, GPUMemoryMoveTransfersOwnership) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f));

    GPUMemory<float> a(16);
    GPUMemory<float> b(std::move(a));
    EXPECT_EQ(a.size(), 0u);
    EXPECT_EQ(b.size(), 16u);
    EXPECT_EQ(a.get(), nullptr);
    EXPECT_NE(b.get(), nullptr);
}

TEST_F(GPUManagerTest, GPUMemoryOutOfRangeCopyThrows) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f));

    GPUMemory<float> mem(4);
    std::vector<float> host(8, 0.0f);
    EXPECT_THROW(mem.copy_from_host(host.data(), host.size()), std::out_of_range);
    EXPECT_THROW(mem.copy_to_host(host.data(), host.size()), std::out_of_range);
}
