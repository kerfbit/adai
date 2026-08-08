// GPU-native reduction/kernel tests for the training-diagnostics fix (TD-013 GPU
// gap: activation_saturation_ratio/attention_entropy always -1.0 under GPU training)
// and the CUDA TD-003 port. Only compiled when ENABLE_GPU or ENABLE_SYCL is
// configured (see tests/CMakeLists.txt) — exercises whichever backend
// MatrixGPU.hpp resolves to (CUDA or SYCL), which share an identical GPUMatrix
// public interface, so no backend-specific code is needed in this file.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../src/gpu/MatrixGPU.hpp"

using adai::gpu::GPUMatrix;

namespace {

GPUMatrix upload(const std::vector<float>& host, int rows, int cols) {
    GPUMatrix m(rows, cols);
    m.upload(host.data(), static_cast<int>(host.size()));
    return m;
}

std::vector<float> download(const GPUMatrix& m) {
    std::vector<float> host(static_cast<size_t>(m.size()));
    m.download(host.data(), m.size());
    return host;
}

}  // namespace

// ============================================================================
// count_below_threshold — the new primitive behind activation_saturation_ratio.
// ============================================================================

TEST(MatrixGPUTD003Test, CountBelowThresholdAllBelow) {
    GPUMatrix m = upload({0.0f, 0.0f, 0.0f, 0.0f}, 2, 2);
    EXPECT_FLOAT_EQ(m.count_below_threshold(0.01f), 1.0f);
}

TEST(MatrixGPUTD003Test, CountBelowThresholdNoneBelow) {
    GPUMatrix m = upload({5.0f, -5.0f, 3.0f, -3.0f}, 2, 2);
    EXPECT_FLOAT_EQ(m.count_below_threshold(0.01f), 0.0f);
}

TEST(MatrixGPUTD003Test, CountBelowThresholdMixed) {
    // 2 of 4 elements have |x| < 0.01
    GPUMatrix m = upload({0.0f, 5.0f, 0.001f, -5.0f}, 2, 2);
    EXPECT_FLOAT_EQ(m.count_below_threshold(0.01f), 0.5f);
}

TEST(MatrixGPUTD003Test, CountBelowThresholdLargeMatrixMultiBlock) {
    // 256*4 = 1024 elements, exercises the multi-block recursive reduction
    // path (matrix_sum_gpu/matrix_count_below_threshold_gpu recurse when
    // there's more than one 256-thread block's worth of data).
    const int n = 1024;
    std::vector<float> host(n, 0.0f);
    for (int i = 0; i < n; i += 2)
        host[i] = 5.0f;  // half saturated, half not
    GPUMatrix m = upload(host, 1, n);
    EXPECT_NEAR(m.count_below_threshold(0.01f), 0.5f, 1e-5f);
}

// ============================================================================
// row_entropy_avg — the new primitive behind attention_entropy.
// ============================================================================

TEST(MatrixGPUTD003Test, RowEntropyOneHotIsZero) {
    // A one-hot row has entropy 0 (log(1) = 0 for the single p=1 term; all
    // other terms are exactly 0 and skipped by the p>0 guard).
    GPUMatrix m = upload({1.0f, 0.0f, 0.0f, 0.0f}, 1, 4);
    EXPECT_NEAR(m.row_entropy_avg(), 0.0f, 1e-4f);
}

TEST(MatrixGPUTD003Test, RowEntropyUniformMatchesLogCols) {
    // A uniform distribution over `cols` outcomes has entropy ln(cols).
    const int cols = 8;
    std::vector<float> host(cols, 1.0f / static_cast<float>(cols));
    GPUMatrix m = upload(host, 1, cols);
    EXPECT_NEAR(m.row_entropy_avg(), std::log(static_cast<float>(cols)), 1e-3f);
}

TEST(MatrixGPUTD003Test, RowEntropyIsNonNegativeForMultipleRows) {
    GPUMatrix m =
        upload({0.25f, 0.25f, 0.25f, 0.25f, 0.7f, 0.1f, 0.1f, 0.1f, 0.1f, 0.7f, 0.1f, 0.1f}, 3, 4);
    EXPECT_GE(m.row_entropy_avg(), 0.0f);
}

// ============================================================================
// Backend-agnostic transfer helpers (Part D).
// ============================================================================

TEST(MatrixGPUTD003Test, CopyDeviceToDeviceRoundTrip) {
    GPUMatrix src = upload({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
    GPUMatrix dst(2, 2);
    dst.zero();
    adai::gpu::matrix_copy_device_to_device_gpu(src.device_ptr(), dst.device_ptr(), src.size());

    std::vector<float> host = download(dst);
    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f);
    EXPECT_FLOAT_EQ(host[3], 4.0f);
}

TEST(MatrixGPUTD003Test, DownloadAtOffsetReadsCorrectRow) {
    // Mirrors EncoderDecoderModel.cpp's gpu_generate_response use: download one
    // row from an arbitrary offset into a larger buffer.
    GPUMatrix m = upload({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 3, 2);
    std::vector<float> last_row(2);
    adai::gpu::matrix_download_gpu(m.device_ptr() + 2 * 2, last_row.data(), 2);
    EXPECT_FLOAT_EQ(last_row[0], 5.0f);
    EXPECT_FLOAT_EQ(last_row[1], 6.0f);
}

// ============================================================================
// Spot-checks of ported TD-003 kernels (CUDA port target — these already
// worked on SYCL; on CUDA this is the first time they've ever executed).
// ============================================================================

TEST(MatrixGPUTD003Test, AddInplaceAccumulates) {
    GPUMatrix dst = upload({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
    GPUMatrix src = upload({10.0f, 20.0f, 30.0f, 40.0f}, 2, 2);
    dst.add_inplace(src);

    std::vector<float> host = download(dst);
    EXPECT_FLOAT_EQ(host[0], 11.0f);
    EXPECT_FLOAT_EQ(host[1], 22.0f);
    EXPECT_FLOAT_EQ(host[2], 33.0f);
    EXPECT_FLOAT_EQ(host[3], 44.0f);
}

TEST(MatrixGPUTD003Test, SoftmaxRowsSumToOneAndMatchKnownValues) {
    // Row [0, 0] -> uniform [0.5, 0.5]; row [0, log(3)] -> [1/4, 3/4].
    GPUMatrix m = upload({0.0f, 0.0f, 0.0f, std::log(3.0f)}, 2, 2);
    m.softmax_rows_inplace();

    std::vector<float> host = download(m);
    EXPECT_NEAR(host[0], 0.5f, 1e-4f);
    EXPECT_NEAR(host[1], 0.5f, 1e-4f);
    EXPECT_NEAR(host[2], 0.25f, 1e-3f);
    EXPECT_NEAR(host[3], 0.75f, 1e-3f);
}

TEST(MatrixGPUTD003Test, MaskedFillReplacesMaskedElements) {
    GPUMatrix m = upload({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
    GPUMatrix mask = upload({1.0f, 0.0f, 0.0f, 1.0f}, 2, 2);  // 0 = fill
    m.masked_fill_inplace(mask, -1e9f);

    std::vector<float> host = download(m);
    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], -1e9f);
    EXPECT_FLOAT_EQ(host[2], -1e9f);
    EXPECT_FLOAT_EQ(host[3], 4.0f);
}

TEST(MatrixGPUTD003Test, SumRowsReducesColumnwise) {
    // [[1,2,3],[4,5,6]] -> column sums [5,7,9]
    GPUMatrix m = upload({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 2, 3);
    GPUMatrix result = m.sum_rows();

    std::vector<float> host = download(result);
    ASSERT_EQ(host.size(), 3u);
    EXPECT_FLOAT_EQ(host[0], 5.0f);
    EXPECT_FLOAT_EQ(host[1], 7.0f);
    EXPECT_FLOAT_EQ(host[2], 9.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
