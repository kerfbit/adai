// GPUManager stub-path tests (TD-041). Exercises the CPU-only stub implementation compiled
// into src/gpu/GPUUtils.hpp when ADAI_ENABLE_GPU is not defined -- the default build, and what
// every CPU-only deployment of adai actually runs. Always built (no ENABLE_GPU/ENABLE_SYCL
// gate needed), unlike gpuutils_test.cpp: the stub class only exists in this configuration, and
// GPUUtils.hpp is header-only/self-contained here (no adai_gpu library to link).

#include <gtest/gtest.h>
#include <stdexcept>
#include "../src/gpu/GPUUtils.hpp"

using adai::gpu::GPUManager;

TEST(GPUManagerStubTest, ProbeIsFalse) {
    EXPECT_FALSE(GPUManager::probe());
}

TEST(GPUManagerStubTest, ProbeDiagnosticMentionsNotCompiled) {
    EXPECT_EQ(GPUManager::probe_diagnostic(), "GPU support not compiled");
}

TEST(GPUManagerStubTest, InitializeReturnsFalseRegardlessOfArgs) {
    EXPECT_FALSE(GPUManager::initialize());
    EXPECT_FALSE(GPUManager::initialize(3, 0.75f, false));
}

TEST(GPUManagerStubTest, CleanupDoesNotThrow) {
    EXPECT_NO_THROW(GPUManager::cleanup());
}

TEST(GPUManagerStubTest, IsAvailableIsFalse) {
    GPUManager::initialize();
    EXPECT_FALSE(GPUManager::is_available());
}

TEST(GPUManagerStubTest, DeviceCountIsZero) {
    EXPECT_EQ(GPUManager::device_count(), 0);
}

TEST(GPUManagerStubTest, CurrentDeviceIsNegativeOne) {
    EXPECT_EQ(GPUManager::current_device(), -1);
}

TEST(GPUManagerStubTest, SetDeviceThrows) {
    EXPECT_THROW(GPUManager::set_device(0), std::runtime_error);
}

TEST(GPUManagerStubTest, GetDeviceInfoMentionsNotCompiled) {
    EXPECT_EQ(GPUManager::get_device_info(), "GPU support not compiled");
    EXPECT_EQ(GPUManager::get_device_info(0), "GPU support not compiled");
}

TEST(GPUManagerStubTest, SynchronizeDoesNotThrow) {
    EXPECT_NO_THROW(GPUManager::synchronize());
}

TEST(GPUManagerStubTest, MemoryGettersAreZero) {
    EXPECT_EQ(GPUManager::get_memory_limit_bytes(), 0u);
    EXPECT_EQ(GPUManager::get_used_memory_bytes(), 0u);
    EXPECT_EQ(GPUManager::get_available_memory_bytes(), 0u);
}
