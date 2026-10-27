// GPU-native reduction/kernel tests for the training-diagnostics fix (TD-013 GPU
// gap: activation_saturation_ratio/attention_entropy always -1.0 under GPU training)
// and the CUDA TD-003 port. Only compiled when ENABLE_GPU or ENABLE_SYCL is
// configured (see tests/CMakeLists.txt) — exercises whichever backend
// MatrixGPU.hpp resolves to (CUDA or SYCL), which share an identical GPUMatrix
// public interface, so no backend-specific code is needed in this file.
//
// Every test guards on GPUManager::probe() and GTEST_SKIP()s when no physical device is
// present, matching the pattern established in tests/gpuutils_test.cpp for TD-041. Without
// this, every test here failed outright (not skipped) in a sandbox/CI runner that has the
// CUDA/SYCL compiler toolchain installed but no physical GPU device -- upload()'s GPUMatrix
// construction calls straight into GPUMemory's constructor, which allocates real device memory
// unconditionally (cudaMalloc/sycl equivalent), with no GPUManager-mediated soft-fail check in
// between. Confirmed directly: `C++ exception with description "cudaMalloc failed: no
// CUDA-capable device is detected" thrown in the test body.` on every test in this file, in a
// sandbox with /dev/nvidia* absent and no GPU PCI device.
//
// probe() alone is NOT sufficient once a device is actually present, though — it only reports
// that a device exists, not that GPUManager::initialize() has been called. Every test also
// explicitly calls initialize() right after the probe() guard (idempotent — an already-
// initialized process just returns true immediately), matching gpuutils_test.cpp's own
// pattern. Missing this was a real bug caught the first time this suite actually ran against
// real GPU hardware (ai-machine, SYCL/Intel ARC) rather than a device-less sandbox: every test
// failed with `C++ exception with description "GPU not initialized. Call GPUManager::
// initialize() first." thrown in the test body.` since probe() reported a device present (so
// nothing skipped) but nothing in this file had ever called initialize().

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "../src/Activation.hpp"
#include "../src/gpu/MatrixGPU.hpp"

using adai::gpu::GPUManager;
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
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
    GPUMatrix m = upload({0.0f, 0.0f, 0.0f, 0.0f}, 2, 2);
    EXPECT_FLOAT_EQ(m.count_below_threshold(0.01f), 1.0f);
}

TEST(MatrixGPUTD003Test, CountBelowThresholdNoneBelow) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
    GPUMatrix m = upload({5.0f, -5.0f, 3.0f, -3.0f}, 2, 2);
    EXPECT_FLOAT_EQ(m.count_below_threshold(0.01f), 0.0f);
}

TEST(MatrixGPUTD003Test, CountBelowThresholdMixed) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
    // 2 of 4 elements have |x| < 0.01
    GPUMatrix m = upload({0.0f, 5.0f, 0.001f, -5.0f}, 2, 2);
    EXPECT_FLOAT_EQ(m.count_below_threshold(0.01f), 0.5f);
}

TEST(MatrixGPUTD003Test, CountBelowThresholdLargeMatrixMultiBlock) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
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
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
    // A one-hot row has entropy 0 (log(1) = 0 for the single p=1 term; all
    // other terms are exactly 0 and skipped by the p>0 guard).
    GPUMatrix m = upload({1.0f, 0.0f, 0.0f, 0.0f}, 1, 4);
    EXPECT_NEAR(m.row_entropy_avg(), 0.0f, 1e-4f);
}

TEST(MatrixGPUTD003Test, RowEntropyUniformMatchesLogCols) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
    // A uniform distribution over `cols` outcomes has entropy ln(cols).
    const int cols = 8;
    std::vector<float> host(cols, 1.0f / static_cast<float>(cols));
    GPUMatrix m = upload(host, 1, cols);
    EXPECT_NEAR(m.row_entropy_avg(), std::log(static_cast<float>(cols)), 1e-3f);
}

TEST(MatrixGPUTD003Test, RowEntropyIsNonNegativeForMultipleRows) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
    GPUMatrix m =
        upload({0.25f, 0.25f, 0.25f, 0.25f, 0.7f, 0.1f, 0.1f, 0.1f, 0.1f, 0.7f, 0.1f, 0.1f}, 3, 4);
    EXPECT_GE(m.row_entropy_avg(), 0.0f);
}

// ============================================================================
// Backend-agnostic transfer helpers (Part D).
// ============================================================================

TEST(MatrixGPUTD003Test, CopyDeviceToDeviceRoundTrip) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
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
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
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
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
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
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
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
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
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
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
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

// ============================================================================
// TD-393 / TD-384 regressions. The work-group reductions must be exact for any
// sub-group size (TD-393: at size 8, 256 ones summed to 64), and the two-pass
// row kernels (max/mean, then sum) must not race on their shared scratch buffer
// (TD-384). Rows wider than one work-group and many rows make the race likely.
// ============================================================================

TEST(MatrixGPUTD003Test, SumIsExactAcrossWorkGroupBoundaries) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
    for (int n : {1, 7, 64, 65, 255, 256, 257, 1000, 4096, 100000}) {
        GPUMatrix m = upload(std::vector<float>(static_cast<size_t>(n), 1.0f), 1, n);
        EXPECT_FLOAT_EQ(m.sum(), static_cast<float>(n)) << "n = " << n;
    }
}

TEST(MatrixGPUTD003Test, CountBelowThresholdIsExactForFullWorkGroup) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
    GPUMatrix m = upload(std::vector<float>(256, 0.0f), 1, 256);
    EXPECT_FLOAT_EQ(m.count_below_threshold(0.01f), 1.0f);
}

TEST(MatrixGPUTD003Test, SoftmaxWideRowsMatchHost) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
    const int rows = 64, cols = 300;
    std::vector<float> host(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            host[static_cast<size_t>(r) * cols + c] =
                std::sin(0.37f * static_cast<float>(c) + static_cast<float>(r)) * 4.0f +
                static_cast<float>(r % 5);
    for (int trial = 0; trial < 5; ++trial) {
        GPUMatrix m = upload(host, rows, cols);
        m.softmax_rows_inplace();
        std::vector<float> out = download(m);
        for (int r = 0; r < rows; ++r) {
            const float* x = &host[static_cast<size_t>(r) * cols];
            float mx = x[0];
            for (int c = 1; c < cols; ++c) mx = std::max(mx, x[c]);
            double denom = 0.0;
            for (int c = 0; c < cols; ++c) denom += std::exp(static_cast<double>(x[c] - mx));
            for (int c = 0; c < cols; ++c) {
                const double want = std::exp(static_cast<double>(x[c] - mx)) / denom;
                ASSERT_NEAR(out[static_cast<size_t>(r) * cols + c], want, 1e-5)
                    << "trial " << trial << " row " << r << " col " << c;
            }
        }
    }
}

TEST(MatrixGPUTD003Test, LayerNormWideRowsMatchHost) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f))
        << "GPUManager::initialize() failed despite probe() reporting a device present.";
    const int rows = 32, cols = 768;
    const float eps = 1e-5f;
    std::vector<float> host(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            host[static_cast<size_t>(r) * cols + c] =
                std::cos(0.11f * static_cast<float>(c) * static_cast<float>(r + 1)) * 3.0f +
                static_cast<float>(r);
    GPUMatrix gamma = upload(std::vector<float>(cols, 1.0f), 1, cols);
    GPUMatrix beta = upload(std::vector<float>(cols, 0.0f), 1, cols);
    for (int trial = 0; trial < 5; ++trial) {
        GPUMatrix x = upload(host, rows, cols);
        GPUMatrix normed(rows, cols), mean(rows, 1), rstd(rows, 1);
        GPUMatrix y = x.layer_norm(gamma, beta, eps, normed, mean, rstd);
        std::vector<float> out = download(y);
        std::vector<float> means = download(mean);
        for (int r = 0; r < rows; ++r) {
            const float* v = &host[static_cast<size_t>(r) * cols];
            double mu = 0.0;
            for (int c = 0; c < cols; ++c) mu += v[c];
            mu /= cols;
            double var = 0.0;
            for (int c = 0; c < cols; ++c) var += (v[c] - mu) * (v[c] - mu);
            var /= cols;
            const double rs = 1.0 / std::sqrt(var + eps);
            ASSERT_NEAR(means[r], mu, 1e-3) << "trial " << trial << " row " << r;
            for (int c = 0; c < cols; ++c)
                ASSERT_NEAR(out[static_cast<size_t>(r) * cols + c], (v[c] - mu) * rs, 1e-3)
                    << "trial " << trial << " row " << r << " col " << c;
        }
    }
}

// ============================================================================
// TD-388: matrix_gemm_strided_gpu() — row-major GEMM on views (pointer offsets and leading
// dimensions), which the cached attention paths use to read one head's columns of a KV cache in
// place. Checked against a host reference for every transpose combination.
// ============================================================================

TEST(MatrixGPUTD388Test, StridedGemmMatchesHostReference) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f));

    const int m = 3, n = 5, k = 4;
    const int a_off = 2, b_off = 1, c_off = 3;  // views start inside wider rows
    const float alpha = 0.75f;
    for (bool trans_a : {false, true}) {
        for (bool trans_b : {false, true}) {
            // Stored shapes: A is m x k (k x m when transposed), B is k x n (n x k when transposed).
            const int a_rows = trans_a ? k : m, a_cols = trans_a ? m : k;
            const int b_rows = trans_b ? n : k, b_cols = trans_b ? k : n;
            const int lda = a_cols + 4, ldb = b_cols + 3, ldc = n + 6;
            std::vector<float> a(static_cast<size_t>(a_rows) * lda), b(static_cast<size_t>(b_rows) * ldb);
            for (size_t i = 0; i < a.size(); ++i) a[i] = std::sin(0.37f * static_cast<float>(i));
            for (size_t i = 0; i < b.size(); ++i) b[i] = std::cos(0.23f * static_cast<float>(i));
            std::vector<float> c(static_cast<size_t>(m) * ldc, 99.0f);

            GPUMatrix da = upload(a, a_rows, lda), db = upload(b, b_rows, ldb), dc = upload(c, m, ldc);
            adai::gpu::matrix_gemm_strided_gpu(trans_a, trans_b, m, n, k, alpha,
                                               da.device_ptr() + a_off, lda,
                                               db.device_ptr() + b_off, ldb,
                                               dc.device_ptr() + c_off, ldc);
            const std::vector<float> out = download(dc);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    double ref = 0.0;
                    for (int p = 0; p < k; ++p) {
                        const float av = trans_a ? a[static_cast<size_t>(p) * lda + a_off + i]
                                                 : a[static_cast<size_t>(i) * lda + a_off + p];
                        const float bv = trans_b ? b[static_cast<size_t>(j) * ldb + b_off + p]
                                                 : b[static_cast<size_t>(p) * ldb + b_off + j];
                        ref += static_cast<double>(av) * bv;
                    }
                    ASSERT_NEAR(out[static_cast<size_t>(i) * ldc + c_off + j], alpha * ref, 1e-4)
                        << trans_a << trans_b << " (" << i << "," << j << ")";
                }
                // Columns outside the view are untouched.
                for (int j = 0; j < c_off; ++j)
                    ASSERT_EQ(out[static_cast<size_t>(i) * ldc + j], 99.0f);
            }
        }
    }
}

// ============================================================================
// TD-216: the GPU GELU forward and backward kernels agree with Activation (they share
// ActivationConstants.hpp; the literals used to be typed separately in each backend).
// ============================================================================

TEST(MatrixGPUTD216Test, GeluForwardAndBackwardMatchTheCpu) {
    if (!GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(GPUManager::initialize(0, 0.5f));

    const int n = 64;
    std::vector<float> x(n), dout(n);
    Matrix cpu_x(1, n);
    for (int i = 0; i < n; ++i) {
        x[i] = -4.0f + 8.0f * static_cast<float>(i) / (n - 1);
        dout[i] = 0.5f + 0.01f * static_cast<float>(i);
        cpu_x(0, i) = x[i];
    }
    const Matrix cpu_y = Activation::gelu(cpu_x);
    const Matrix cpu_dy = Activation::gelu_derivative(cpu_x);

    GPUMatrix y = upload(x, 1, n);
    adai::gpu::matrix_apply_activation_gpu(y.device_ptr(), n, adai::gpu::ActivationType::GELU);
    GPUMatrix px = upload(x, 1, n), d = upload(dout, 1, n), din(1, n);
    adai::gpu::matrix_gelu_backward_gpu(px.device_ptr(), d.device_ptr(), din.device_ptr(), n);
    const std::vector<float> gy = download(y), gd = download(din);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(gy[i], cpu_y(0, i), 1e-5f) << "x = " << x[i];
        EXPECT_NEAR(gd[i], dout[i] * cpu_dy(0, i), 1e-5f) << "x = " << x[i];
    }
}
