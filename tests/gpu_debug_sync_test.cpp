/**
 * @file gpu_debug_sync_test.cpp
 * @brief TD-574: the GPU reset diagnostics. Settings parsing, the last-work record and the trace
 *        file run in every build. In a SYCL build with a device (or ADAI_SYCL_ALLOW_CPU_DEVICE=1
 *        for the oneAPI CPU device), each kernel family's host version must give the device
 *        kernel's results, and ADAI_GPU_SYNC=kernel must name the kernel that ran.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "../src/gpu/GPUDebugSync.hpp"
#if defined(ADAI_GPU_BACKEND_SYCL)
#include "../src/DropoutMask.hpp"
#include "../src/gpu/MatrixGPU.hpp"
#endif

namespace {

using adai::gpu::DebugSyncMode;
using adai::gpu::GPUDebugSettings;
using adai::gpu::parse_gpu_debug_settings;

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

struct ResetSettings {
    ~ResetSettings() {
        adai::gpu::set_gpu_debug_settings_for_testing(GPUDebugSettings{});
    }
};

}  // namespace

TEST(GPUDebugSyncTest, UnsetMeansEverythingOff) {
    const GPUDebugSettings s = parse_gpu_debug_settings(nullptr, nullptr, nullptr);
    EXPECT_EQ(s.sync, DebugSyncMode::Off);
    EXPECT_EQ(s.host_kernels, 0u);
    EXPECT_TRUE(s.trace_file.empty());
    EXPECT_TRUE(s.problems.empty());
    EXPECT_FALSE(s.any());
}

TEST(GPUDebugSyncTest, ReadsModesAndFamiliesInAnyCase) {
    EXPECT_EQ(parse_gpu_debug_settings("step", nullptr, nullptr).sync, DebugSyncMode::Step);
    EXPECT_EQ(parse_gpu_debug_settings(" Kernel ", nullptr, nullptr).sync, DebugSyncMode::Kernel);
    EXPECT_EQ(parse_gpu_debug_settings("off", nullptr, nullptr).sync, DebugSyncMode::Off);
    const GPUDebugSettings s =
        parse_gpu_debug_settings(nullptr, "Embedding, optimizer,,dropout", "/tmp/t");
    EXPECT_EQ(s.host_kernels,
              adai::gpu::kHostEmbedding | adai::gpu::kHostOptimizer | adai::gpu::kHostDropout);
    EXPECT_EQ(s.trace_file, "/tmp/t");
    EXPECT_TRUE(s.problems.empty());
    EXPECT_TRUE(s.any());
}

TEST(GPUDebugSyncTest, UnknownValuesAreIgnoredAndReported) {
    const GPUDebugSettings s = parse_gpu_debug_settings("always", "embedding,softmax", nullptr);
    EXPECT_EQ(s.sync, DebugSyncMode::Off);
    EXPECT_EQ(s.host_kernels, adai::gpu::kHostEmbedding);
    ASSERT_EQ(s.problems.size(), 2u);
    EXPECT_NE(s.problems[0].find("always"), std::string::npos);
    EXPECT_NE(s.problems[1].find("softmax"), std::string::npos);
}

TEST(GPUDebugSyncTest, ReadsKernelEveryN) {
    const GPUDebugSettings s = parse_gpu_debug_settings("kernel:64", nullptr, nullptr);
    EXPECT_EQ(s.sync, DebugSyncMode::Kernel);
    EXPECT_EQ(s.sync_every, 64u);
    EXPECT_TRUE(s.problems.empty());
    EXPECT_EQ(parse_gpu_debug_settings("kernel", nullptr, nullptr).sync_every, 1u);
    for (const char* bad :
         {"kernel:0", "kernel:", "kernel:abc", "kernel:8x", "kernel:-4", "kernel:2000000"}) {
        const GPUDebugSettings b = parse_gpu_debug_settings(bad, nullptr, nullptr);
        EXPECT_EQ(b.sync, DebugSyncMode::Off) << bad;
        EXPECT_EQ(b.problems.size(), 1u) << bad;
    }
}

TEST(GPUDebugSyncTest, KernelEveryNWaitsBeforeEveryNthOperation) {
    ResetSettings reset;
    GPUDebugSettings s;
    s.sync = DebugSyncMode::Kernel;
    s.sync_every = 3;
    adai::gpu::set_gpu_debug_settings_for_testing(s);
    std::string waits;
    for (int i = 0; i < 7; ++i) {
        waits += adai::gpu::wait_before_gpu_operation() ? 'W' : '.';
        adai::gpu::note_gpu_kernel("k_gpu");
    }
    EXPECT_EQ(waits, "W..W..W");
    EXPECT_NE(adai::gpu::gpu_debug_summary().find("ADAI_GPU_SYNC=kernel:3"), std::string::npos);
    EXPECT_NE(adai::gpu::describe_last_gpu_work().find("kernel:3"), std::string::npos);

    s.sync_every = 1;
    adai::gpu::set_gpu_debug_settings_for_testing(s);
    EXPECT_TRUE(adai::gpu::wait_before_gpu_operation());
    adai::gpu::note_gpu_kernel("k_gpu");
    EXPECT_TRUE(adai::gpu::wait_before_gpu_operation());

    s.sync = DebugSyncMode::Step;
    adai::gpu::set_gpu_debug_settings_for_testing(s);
    EXPECT_FALSE(adai::gpu::wait_before_gpu_operation());
}

TEST(GPUDebugSyncTest, RecordsLastWorkAndKeepsOneTraceRecord) {
    ResetSettings reset;
    const std::string path =
        (std::filesystem::temp_directory_path() / "gpu_debug_sync_test_trace.txt").string();
    GPUDebugSettings s;
    s.trace_file = path;
    adai::gpu::set_gpu_debug_settings_for_testing(s);

    adai::gpu::note_gpu_phase("forward");
    adai::gpu::note_gpu_kernel("a_rather_long_kernel_name_gpu");
    adai::gpu::note_gpu_kernel("short_gpu");

    const std::string d = adai::gpu::describe_last_gpu_work();
    EXPECT_NE(d.find("kernel short_gpu in phase forward (GPU work #2)"), std::string::npos) << d;
    EXPECT_NE(d.find("may not have reached"), std::string::npos) << d;

    const std::string trace = read_file(path);
    EXPECT_EQ(std::count(trace.begin(), trace.end(), '\n'), 1);
    EXPECT_EQ(trace.rfind("2 forward short_gpu ", 0), 0u) << trace;
    EXPECT_EQ(trace.find("long"), std::string::npos) << "an older record shows through: " << trace;

    EXPECT_NE(adai::gpu::gpu_debug_summary().find("ADAI_GPU_TRACE_FILE=" + path),
              std::string::npos);
    std::filesystem::remove(path);
}

TEST(GPUDebugSyncTest, SummaryIsEmptyWhenNothingIsSet) {
    ResetSettings reset;
    adai::gpu::set_gpu_debug_settings_for_testing(GPUDebugSettings{});
    EXPECT_TRUE(adai::gpu::gpu_debug_summary().empty());
}

#if defined(ADAI_GPU_BACKEND_SYCL)

namespace {

using adai::gpu::GPUMemory;

bool gpu_ready() {
    return adai::gpu::GPUManager::probe() && adai::gpu::GPUManager::initialize(0, 0.5f);
}

template <typename T>
GPUMemory<T> upload(const std::vector<T>& h) {
    GPUMemory<T> d(h.size());
    d.copy_from_host(h.data(), h.size());
    return d;
}

template <typename T>
std::vector<T> download(const GPUMemory<T>& d) {
    std::vector<T> h(d.size());
    d.copy_to_host(h.data(), h.size());
    return h;
}

std::vector<float> random_vector(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v)
        x = dist(rng);
    return v;
}

void expect_close(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a[i], b[i], tol * std::max(1.0f, std::abs(b[i]))) << "at " << i;
}

GPUDebugSettings host_families(unsigned families) {
    GPUDebugSettings s;
    s.host_kernels = families;
    return s;
}

}  // namespace

class GPUHostKernelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        if (!gpu_ready())
            GTEST_SKIP() << "no GPU device";
    }
    ResetSettings reset_;
};

TEST_F(GPUHostKernelTest, EmbeddingGatherAndScatterMatchTheDevice) {
    const int vocab = 10, d = 8, n = 6, pe_rows = 5, pos_offset = 2;
    const std::vector<float> table = random_vector(vocab * d, 1);
    const std::vector<float> pe = random_vector(pe_rows * d, 2);
    const std::vector<int> ids = {3, 7, 3, 0, 9, 3};  // repeats, and positions past pe_rows
    const std::vector<float> dout = random_vector(n * d, 3);
    const std::vector<float> grad0 = random_vector(vocab * d, 4);

    auto run = [&](unsigned families, std::vector<float>& out, std::vector<float>& grad) {
        adai::gpu::set_gpu_debug_settings_for_testing(host_families(families));
        auto d_table = upload(table);
        auto d_pe = upload(pe);
        auto d_ids = upload(ids);
        auto d_dout = upload(dout);
        auto d_grad = upload(grad0);
        GPUMemory<float> d_out(static_cast<size_t>(n) * d);
        adai::gpu::embedding_gather_add_pe_gpu(d_table.get(), d_ids.get(), d_pe.get(), pe_rows,
                                               pos_offset, d_out.get(), n, d);
        adai::gpu::embedding_scatter_add_gpu(d_dout.get(), d_ids.get(), d_grad.get(), n, d);
        out = download(d_out);
        grad = download(d_grad);
    };
    std::vector<float> dev_out, dev_grad, host_out, host_grad;
    run(0, dev_out, dev_grad);
    EXPECT_EQ(adai::gpu::host_kernel_runs(), 0u);
    run(adai::gpu::kHostEmbedding, host_out, host_grad);
    EXPECT_EQ(adai::gpu::host_kernel_runs(), 2u) << "the host versions didn't run";
    expect_close(host_out, dev_out, 0.0f);
    expect_close(host_grad, dev_grad, 1e-6f);  // the device adds repeats in any order
}

TEST_F(GPUHostKernelTest, DropoutMatchesTheDevice) {
    const std::vector<float> data = random_vector(1000, 5);
    auto run = [&](unsigned families, std::vector<float>& out, std::vector<float>& mask) {
        adai::gpu::set_gpu_debug_settings_for_testing(host_families(families));
        auto d_data = upload(data);
        GPUMemory<float> d_mask(data.size());
        adai::gpu::dropout_forward_gpu(d_data.get(), d_mask.get(), static_cast<int>(data.size()),
                                       0.9f, 1234ULL);
        out = download(d_data);
        mask = download(d_mask);
    };
    std::vector<float> dev_out, dev_mask, host_out, host_mask;
    run(0, dev_out, dev_mask);
    run(adai::gpu::kHostDropout, host_out, host_mask);
    EXPECT_EQ(adai::gpu::host_kernel_runs(), 1u) << "the host version didn't run";
    expect_close(host_mask, dev_mask, 0.0f);
    expect_close(host_out, dev_out, 0.0f);
}

TEST_F(GPUHostKernelTest, OptimizerUpdateAndNormsMatchTheDevice) {
    const int size = 777;
    const std::vector<float> w0 = random_vector(size, 6);
    const std::vector<float> g = random_vector(size, 7);
    std::vector<float> m0 = random_vector(size, 8);
    std::vector<float> v0 = random_vector(size, 9);
    for (auto& x : v0)
        x = std::abs(x);
    for (auto rule :
         {adai::gpu::GPUOptimizerRule::ADAMW, adai::gpu::GPUOptimizerRule::ADAM,
          adai::gpu::GPUOptimizerRule::SGD_MOMENTUM, adai::gpu::GPUOptimizerRule::SGD}) {
        adai::gpu::GPUOptimizerUpdate u;
        u.rule = rule;
        u.learning_rate = 0.01f;
        u.weight_decay = 0.1f;
        u.bias_correction1 = 0.19f;
        u.bias_correction2 = 0.0199f;
        u.grad_scale = 0.5f;
        const bool use_m = rule != adai::gpu::GPUOptimizerRule::SGD;
        const bool use_v =
            rule == adai::gpu::GPUOptimizerRule::ADAM || rule == adai::gpu::GPUOptimizerRule::ADAMW;
        auto run = [&](unsigned families, std::vector<float>& w, std::vector<float>& m,
                       std::vector<float>& v, float& norm) {
            adai::gpu::set_gpu_debug_settings_for_testing(host_families(families));
            auto d_w = upload(w0);
            auto d_g = upload(g);
            auto d_m = upload(m0);
            auto d_v = upload(v0);
            auto d_norm = upload(std::vector<float>{1.5f});
            adai::gpu::matrix_sum_squares_accumulate_gpu(d_g.get(), size, d_norm.get());
            adai::gpu::optimizer_update_gpu(d_w.get(), d_g.get(), use_m ? d_m.get() : nullptr,
                                            use_v ? d_v.get() : nullptr, size, u);
            w = download(d_w);
            m = download(d_m);
            v = download(d_v);
            norm = download(d_norm)[0];
        };
        std::vector<float> dw, dm, dv, hw, hm, hv;
        float dn = 0.0f, hn = 0.0f;
        run(0, dw, dm, dv, dn);
        run(adai::gpu::kHostOptimizer, hw, hm, hv, hn);
        EXPECT_EQ(adai::gpu::host_kernel_runs(), 2u) << "the host versions didn't run";
        SCOPED_TRACE(static_cast<int>(rule));
        expect_close(hw, dw, 1e-6f);
        expect_close(hm, dm, 1e-6f);
        expect_close(hv, dv, 1e-6f);
        EXPECT_NEAR(hn, dn, 1e-4f * dn);
    }
}

TEST_F(GPUHostKernelTest, KernelModeNamesTheKernelAndPhase) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "gpu_debug_sync_test_kernel.txt").string();
    GPUDebugSettings s;
    s.sync = DebugSyncMode::Kernel;
    s.trace_file = path;
    adai::gpu::set_gpu_debug_settings_for_testing(s);

    auto a = upload(random_vector(64, 10));
    auto b = upload(random_vector(64, 11));
    GPUMemory<float> c(64);
    adai::gpu::GPUManager::debug_sync_point("forward");
    adai::gpu::matrix_add_gpu(a.get(), b.get(), c.get(), 64);

    const std::string d = adai::gpu::describe_last_gpu_work();
    EXPECT_NE(d.find("kernel matrix_add_gpu in phase forward"), std::string::npos) << d;
    EXPECT_NE(d.find("this kernel failed"), std::string::npos) << d;
    EXPECT_NE(read_file(path).find(" forward matrix_add_gpu "), std::string::npos);

    // An upload is traced too, and the kernel before it has finished by then.
    auto e = upload(random_vector(4, 12));
    EXPECT_NE(adai::gpu::describe_last_gpu_work().find("kernel upload"), std::string::npos);
    std::filesystem::remove(path);
}

#endif  // ADAI_GPU_BACKEND_SYCL
