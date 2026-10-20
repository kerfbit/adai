/**
 * @file gpu_optimizer_test.cpp
 * @brief TD-468: the device-resident optimizer (GPUOptimizer and its kernels) must apply exactly
 *        the CPU Optimizer's updates. Skips without a GPU; on a machine without one, run with
 *        ADAI_SYCL_ALLOW_CPU_DEVICE=1 in a SYCL build to use the oneAPI CPU device (TD-394).
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "../src/EncoderDecoderModel.hpp"
#include "../src/GPUOptimizer.hpp"
#include "../src/Matrix.hpp"
#include "../src/Optimizer.hpp"
#include "../src/PositionalEncoding.hpp"
#include "../src/TokenEmbedding.hpp"
#include "../src/gpu/MatrixGPU.hpp"

namespace {

using adai::gpu::GPUMatrix;
using adai::gpu::GPUParamRef;

Matrix random_matrix(int rows, int cols, std::mt19937& rng, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(rows, cols);
    for (auto& row : m.data)
        for (auto& x : row)
            x = d(rng);
    return m;
}

std::vector<float> flat(const Matrix& m) {
    std::vector<float> v;
    for (const auto& row : m.data)
        v.insert(v.end(), row.begin(), row.end());
    return v;
}

float max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::abs(a[i] - b[i]) / std::max(1.0f, std::abs(b[i])));
    return worst;
}

bool gpu_ready() {
    return adai::gpu::GPUManager::probe() && adai::gpu::GPUManager::initialize(0, 0.5f);
}

}  // namespace

class GPUOptimizerRuleTest : public ::testing::TestWithParam<OptimizerType> {};

// Three steps with different gradients and a gradient scale (the TD-272 rescale times the clip
// factor) give the same weights as the CPU Optimizer stepping scaled gradients.
TEST_P(GPUOptimizerRuleTest, DeviceStepMatchesCpuOptimizer) {
    if (!gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const OptimizerType type = GetParam();
    std::mt19937 rng(7);
    Matrix w_cpu = random_matrix(37, 53, rng, 0.5f);
    Matrix g_cpu(37, 53);
    Matrix w_dev_host = w_cpu;  // the host copy GPUOptimizer is keyed by
    Matrix g_dev_host(37, 53);

    Optimizer cpu(type, 0.01f);
    Optimizer dev(type, 0.01f);
    for (Optimizer* o : {&cpu, &dev}) {
        o->set_weight_decay(0.05f);
        o->set_betas(0.9f, 0.99f);
    }
    cpu.add_parameter_group(&w_cpu, &g_cpu);
    dev.add_parameter_group(&w_dev_host, &g_dev_host);

    GPUMatrix w_dev(37, 53), g_dev(37, 53);
    const auto w0 = flat(w_cpu);
    w_dev.upload(w0.data(), w_dev.size());
    GPUOptimizer gpu(dev, {GPUParamRef{&w_dev_host, &w_dev, &g_dev, -1}});
    ASSERT_EQ(gpu.device_group_count(), 1u);
    ASSERT_EQ(gpu.host_group_count(), 0u);

    const float scale = 0.5f;
    for (int step = 0; step < 3; ++step) {
        const Matrix g = random_matrix(37, 53, rng, 1.0f);
        const auto gf = flat(g);
        g_dev.upload(gf.data(), g_dev.size());
        for (int i = 0; i < g.rows; ++i)
            for (int j = 0; j < g.cols; ++j)
                g_cpu.data[i][j] = g.data[i][j] * scale;

        double expect_sq = 0.0;
        for (float x : gf)
            expect_sq += static_cast<double>(x) * x;
        const float got_sq = gpu.gradient_sq_norms()[0];
        EXPECT_NEAR(got_sq, expect_sq, 1e-3 * expect_sq);

        cpu.step();
        gpu.step(scale);
    }
    std::vector<float> got(static_cast<size_t>(w_dev.size()));
    w_dev.download(got.data(), w_dev.size());
    EXPECT_LT(max_rel_diff(got, flat(w_cpu)), 1e-5f);
}

INSTANTIATE_TEST_SUITE_P(AllRules, GPUOptimizerRuleTest,
                         ::testing::Values(OptimizerType::SGD, OptimizerType::SGD_MOMENTUM,
                                           OptimizerType::ADAM, OptimizerType::ADAMW));

// The moments round-trip through the CPU optimizer (TD-269's training state): a second
// GPUOptimizer built after download_state() continues exactly where the first stopped.
TEST(GPUOptimizerTest, StateRoundTripsThroughTheCpuOptimizer) {
    if (!gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    std::mt19937 rng(3);
    Matrix w_host = random_matrix(16, 16, rng, 0.5f);
    Matrix g_host(16, 16);
    Optimizer opt(OptimizerType::ADAMW, 0.01f);
    opt.add_parameter_group(&w_host, &g_host);
    GPUMatrix w_dev(16, 16), g_dev(16, 16);
    const auto w0 = flat(w_host);
    w_dev.upload(w0.data(), w_dev.size());
    const Matrix g = random_matrix(16, 16, rng, 1.0f);
    const auto gf = flat(g);
    g_dev.upload(gf.data(), g_dev.size());

    // Reference: one optimizer living through three steps.
    Matrix w_ref = w_host, g_ref(16, 16);
    Optimizer ref(OptimizerType::ADAMW, 0.01f);
    ref.add_parameter_group(&w_ref, &g_ref);
    g_ref = g;
    for (int s = 0; s < 3; ++s)
        ref.step();

    {
        GPUOptimizer first(opt, {GPUParamRef{&w_host, &w_dev, &g_dev, -1}});
        first.gradient_sq_norms();
        first.step(1.0f);
        first.step(1.0f);
        first.download_state();
    }
    GPUOptimizer second(opt, {GPUParamRef{&w_host, &w_dev, &g_dev, -1}});
    second.gradient_sq_norms();
    second.step(1.0f);
    std::vector<float> got(static_cast<size_t>(w_dev.size()));
    w_dev.download(got.data(), w_dev.size());
    EXPECT_LT(max_rel_diff(got, flat(w_ref)), 1e-5f);
}

// A real model: one device-resident step gives the same model as the old path (download every
// gradient, CPU Optimizer step), compared through the model's logits.
TEST(GPUOptimizerTest, ModelStepMatchesTheDownloadAndCpuStepPath) {
    if (!gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    EncoderDecoderModel a(64, 32, 1, 1, 2, 64, 32);
    a.get_tokenizer()->build_vocab({"the cat sat on the mat", "a dog ran far away home"}, 64);
    a.set_training(true);
    std::unique_ptr<EncoderDecoderModel> b = a.clone();
    b->set_training(true);

    Optimizer opt_a(OptimizerType::ADAMW, 0.01f), opt_b(OptimizerType::ADAMW, 0.01f);
    opt_a.set_weight_decay(0.01f);
    opt_b.set_weight_decay(0.01f);
    a.register_parameters(opt_a);
    b->register_parameters(opt_b);
    a.gpu_init_training();
    b->gpu_init_training();

    BPETokenizer* tok = a.get_tokenizer();
    const std::vector<int> in = tok->encode("the cat sat", false);
    const std::vector<int> tgt =
        EncoderDecoderModel::strip_leading_bos(tok->encode("on the mat", true), 2);

    GPUOptimizer gpu(opt_a, a.gpu_collect_parameters());
    // TD-547: the token embeddings have device tables too, so nothing is stepped on the host.
    EXPECT_EQ(gpu.host_group_count(), 0u);

    for (EncoderDecoderModel* m : {&a, b.get()}) {
        m->zero_grad();
        m->gpu_zero_grads();
        m->gpu_forward(in, tgt);
        m->gpu_backward(1.0f);
    }
    gpu.gradient_sq_norms();
    gpu.step(1.0f);
    a.gpu_download_weights();

    b->gpu_download_grads();
    opt_b.step();

    const Matrix la = a.forward(in, tgt);
    const Matrix lb = b->forward(in, tgt);
    EXPECT_LT(max_rel_diff(flat(la), flat(lb)), 1e-4f);
}

// TD-547: the device lookup equals TokenEmbedding::forward() + PositionalEncoding::forward(),
// including the positional offset the cached decode step uses and no positional row past the
// table, and the device backward equals backward() with repeated tokens.
TEST(GPUOptimizerTest, DeviceEmbeddingMatchesTheHostLookupAndBackward) {
    if (!gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    TokenEmbedding emb(50, 16);
    PositionalEncoding pe(8, 16);  // max_len 8, d_model 16
    GPUMatrix pe_dev(8, 16);
    const auto pe_flat = flat(pe.get_encoding());
    pe_dev.upload(pe_flat.data(), pe_dev.size());

    const std::vector<int> ids = {3, 7, 3, 49, 0};
    const Matrix expect = pe.forward(emb.forward(ids));
    GPUMatrix got = emb.gpu_forward(ids, &pe_dev, 8);
    std::vector<float> got_flat(static_cast<size_t>(got.size()));
    got.download(got_flat.data(), got.size());
    EXPECT_LT(max_rel_diff(got_flat, flat(expect)), 1e-6f);

    // Offset 6: rows 6 and 7 get positional rows, rows 8+ none (past the 8-row table).
    GPUMatrix off = emb.gpu_forward({3, 7, 9}, &pe_dev, 8, 6);
    std::vector<float> off_flat(static_cast<size_t>(off.size()));
    off.download(off_flat.data(), off.size());
    for (int j = 0; j < 16; ++j) {
        EXPECT_NEAR(off_flat[j], emb.get_embeddings()(3, j) + pe.get_encoding()(6, j), 1e-6f);
        EXPECT_NEAR(off_flat[32 + j], emb.get_embeddings()(9, j), 1e-6f);
    }

    std::mt19937 rng(11);
    const Matrix dout = random_matrix(5, 16, rng, 1.0f);
    emb.gpu_forward(ids, &pe_dev, 8);
    emb.gpu_zero_grads();
    GPUMatrix dout_dev(5, 16);
    const auto dflat = flat(dout);
    dout_dev.upload(dflat.data(), dout_dev.size());
    emb.gpu_backward(dout_dev);
    emb.zero_grad();
    emb.gpu_download_grads();
    TokenEmbedding ref(50, 16);
    ref.backward(ids, dout);
    // Same gradient (token 3 repeats, so its row is the sum of two rows).
    EXPECT_NEAR(emb.get_gradient_norm(), ref.get_gradient_norm(), 1e-4f);
    EXPECT_THROW(emb.gpu_forward({50}, &pe_dev, 8), std::out_of_range);
}
