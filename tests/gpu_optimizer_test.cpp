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

#include "../src/CrossAttention.hpp"
#include "../src/DecoderBlock.hpp"
#include "../src/EncoderDecoderModel.hpp"
#include "../src/HippocampalMemory.hpp"
#include "../src/LeJEPAEncoder.hpp"
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

// ============================================================================
// TD-342: the gated world-model/hippocampal paths on the device
// ============================================================================

namespace {

std::unique_ptr<HippocampalMemory> filled_memory(int d_model) {
    auto memory = std::make_unique<HippocampalMemory>(d_model, 4);
    for (int slot = 0; slot < 3; ++slot) {
        Matrix key(1, d_model), value(1, d_model);
        for (int j = 0; j < d_model; ++j) {
            key(0, j) = 0.1f * std::sin(static_cast<float>(slot * 7 + j));
            value(0, j) = key(0, j);
        }
        memory->write(key, value);
    }
    return memory;
}

Matrix causal(int n) {
    Matrix m(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m(i, j) = (j <= i) ? 1.0f : 0.0f;
    return m;
}

}  // namespace

TEST(GPUGatedPathTest, GateKernelsMatchTheHostFormula) {
    if (!gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    std::mt19937 rng(11);
    const int n = 1000;
    Matrix residual = random_matrix(1, n, rng, 1.0f), attn = random_matrix(1, n, rng, 1.0f);
    Matrix d_res = random_matrix(1, n, rng, 1.0f);
    const float g = 0.37f, t = std::tanh(g);

    GPUMatrix res_dev = residual.to_gpu(), attn_dev = attn.to_gpu(), d_res_dev = d_res.to_gpu();
    GPUMatrix gate_dev(1, 1), gate_grad_dev(1, 1), d_attn_dev(1, n);
    gate_dev.upload(&g, 1);
    const float start = 0.25f;
    gate_grad_dev.upload(&start, 1);

    adai::gpu::gated_residual_add_gpu(res_dev.device_ptr(), attn_dev.device_ptr(),
                                      gate_dev.device_ptr(), n);
    adai::gpu::gated_residual_backward_gpu(d_res_dev.device_ptr(), attn_dev.device_ptr(),
                                           gate_dev.device_ptr(), d_attn_dev.device_ptr(),
                                           gate_grad_dev.device_ptr(), n);
    Matrix res_out = Matrix::from_gpu(res_dev), d_attn = Matrix::from_gpu(d_attn_dev);
    float gate_grad = 0.0f;
    gate_grad_dev.download(&gate_grad, 1);

    double dot = 0.0;
    for (int j = 0; j < n; ++j) {
        EXPECT_NEAR(res_out(0, j), residual(0, j) + t * attn(0, j), 1e-5f);
        EXPECT_NEAR(d_attn(0, j), t * d_res(0, j), 1e-5f);
        dot += static_cast<double>(d_res(0, j)) * attn(0, j);
    }
    EXPECT_NEAR(gate_grad, start + (1.0f - t * t) * static_cast<float>(dot), 1e-3f);
}

// The score bias the hippocampal path adds before softmax, on the device, as
// CrossAttention::forward_with_scores() on the host.
TEST(GPUGatedPathTest, ScoreBiasMatchesForwardWithScores) {
    if (!gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    std::mt19937 rng(5);
    CrossAttention attn(16, 2);
    Matrix query = random_matrix(3, 16, rng, 0.5f), kv = random_matrix(4, 16, rng, 0.5f);
    Matrix bias = random_matrix(3, 4, rng, 1.0f);
    Matrix cpu = attn.forward_with_scores(query, kv, bias);
    attn.gpu_upload_weights();
    GPUMatrix q_dev = query.to_gpu(), kv_dev = kv.to_gpu(), bias_dev = bias.to_gpu();
    Matrix gpu = Matrix::from_gpu(attn.gpu_forward(q_dev, kv_dev, nullptr, &bias_dev));
    EXPECT_LT(max_rel_diff(flat(gpu), flat(cpu)), 1e-4f);
    EXPECT_LT(max_rel_diff(flat(attn.gpu_last_mean_attention_weights()),
                           flat(attn.get_last_attention_weights())),
              1e-4f);
}

// Both gated paths open, with a world-model output and a memory: the GPU forward and backward
// match the CPU ones (output, input gradient, gate gradients), and the memory's coverage ends up
// the same.
TEST(GPUGatedPathTest, DecoderBlockGatedPathsMatchTheCpu) {
    if (!gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const int d = 16, tgt = 3, src = 4, wm_len = 3;
    std::mt19937 rng(3);
    DecoderBlock block(d, 2, 32, 0.1f, /*enable_world_model=*/true, /*enable_hippocampal=*/true);
    block.set_gate(0.5f);
    block.set_gate_h(-0.4f);
    Matrix input = random_matrix(tgt, d, rng, 0.5f), enc = random_matrix(src, d, rng, 0.5f);
    Matrix wm = random_matrix(wm_len, d, rng, 0.5f), dout = random_matrix(tgt, d, rng, 0.5f);
    Matrix mask = causal(tgt);
    auto mem_cpu = filled_memory(d);
    auto mem_gpu = filled_memory(d);

    block.zero_grad();
    Matrix out_cpu = block.forward(input, enc, mask, nullptr, &wm, nullptr, mem_cpu.get(), 0.3f,
                                   0.9f, 0.2f, 0.9f);
    // TD-343: the forward only reads the memory; commit the step as generation does.
    mem_cpu->commit_attention(block.last_hippocampal_attention(), 0.9f, 0.9f);
    Matrix d_input_cpu = block.backward(dout);
    const float gate_grad_cpu = block.get_gate_grad(), gate_h_grad_cpu = block.get_gate_h_grad();
    ASSERT_NE(gate_grad_cpu, 0.0f);
    ASSERT_NE(gate_h_grad_cpu, 0.0f);

    block.gpu_upload_weights();
    block.gpu_zero_grads();
    GPUMatrix in_dev = input.to_gpu(), enc_dev = enc.to_gpu(), wm_dev = wm.to_gpu();
    GPUMatrix mask_dev = mask.to_gpu(), dout_dev = dout.to_gpu();
    Matrix out_gpu = Matrix::from_gpu(block.gpu_forward(in_dev, enc_dev, &mask_dev, &wm_dev,
                                                        mem_gpu.get(), 0.3f, 0.9f, 0.2f, 0.9f));
    mem_gpu->commit_attention(block.last_hippocampal_attention(), 0.9f, 0.9f);  // GPU download
    auto [d_input_dev, d_enc_dev] = block.gpu_backward(dout_dev);
    Matrix d_input_gpu = Matrix::from_gpu(d_input_dev);
    block.zero_grad();
    block.gpu_download_grads();  // adds the device gate gradients into the (zeroed) host ones

    EXPECT_LT(max_rel_diff(flat(out_gpu), flat(out_cpu)), 1e-4f);
    EXPECT_LT(max_rel_diff(flat(d_input_gpu), flat(d_input_cpu)), 1e-4f);
    EXPECT_NEAR(block.get_gate_grad(), gate_grad_cpu, 1e-4f * std::max(1.0f, std::abs(gate_grad_cpu)));
    EXPECT_NEAR(block.get_gate_h_grad(), gate_h_grad_cpu,
                1e-4f * std::max(1.0f, std::abs(gate_h_grad_cpu)));
    ASSERT_EQ(mem_cpu->committed_steps(), 1u);
    ASSERT_EQ(mem_gpu->committed_steps(), 1u);
    for (int slot = 0; slot < mem_cpu->size(); ++slot) {
        EXPECT_GT(mem_cpu->coverage_vector()[slot], 0.0f);
        EXPECT_NEAR(mem_gpu->coverage_vector()[slot], mem_cpu->coverage_vector()[slot], 1e-5f);
    }
}

TEST(GPUGatedPathTest, WorldModelGpuEncodeMatchesEncode) {
    if (!gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    LeJEPAEncoder world_model(64, 32, 2, 4, 64, 32);
    world_model.build_tokenizer({"the cat sat on the mat", "a dog ran far away home"}, 64);
    const Matrix cpu = world_model.encode("the cat sat on the mat");
    const Matrix gpu = Matrix::from_gpu(world_model.gpu_encode("the cat sat on the mat"));
    ASSERT_EQ(gpu.rows, cpu.rows);
    EXPECT_LT(max_rel_diff(flat(gpu), flat(cpu)), 1e-4f);
}

// A gated model trained on the device the way ChatbotTrainer does: every parameter, gates
// included, is a device group, and the gates open.
TEST(GPUGatedPathTest, GatedModelTrainsItsGatesOnTheDevice) {
    if (!gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const std::vector<std::string> corpus = {"the cat sat on the mat", "a dog ran far away home"};
    EncoderDecoderModel model(64, 32, 1, 2, 2, 64, 32, /*world_model_inject_every_n_layers=*/1);
    model.get_tokenizer()->build_vocab(corpus, 64);
    auto world_model = std::make_unique<LeJEPAEncoder>(64, 32, 1, 2, 64, 32);
    world_model->build_tokenizer(corpus, 64);
    world_model->set_requires_grad(false);
    model.set_world_model(std::move(world_model));
    model.set_hippocampal_memory(filled_memory(32));
    model.set_training(true);

    Optimizer opt(OptimizerType::ADAMW, 0.01f);
    model.register_parameters(opt);
    model.gpu_init_training();
    GPUOptimizer gpu(opt, model.gpu_collect_parameters());
    EXPECT_EQ(gpu.host_group_count(), 0u) << "the gated sub-layers and gates must be device groups";

    BPETokenizer* tok = model.get_tokenizer();
    const std::vector<int> in = tok->encode("the cat sat", false);
    const std::vector<int> tgt =
        EncoderDecoderModel::strip_leading_bos(tok->encode("on the mat", true), 2);
    for (int step = 0; step < 3; ++step) {
        model.gpu_zero_grads();
        model.gpu_forward(in, tgt);
        model.gpu_backward(1.0f);
        gpu.gradient_sq_norms();
        gpu.step(1.0f);
    }
    model.gpu_download_weights();
    for (int l = 0; l < model.get_decoder()->get_num_layers(); ++l) {
        DecoderBlock* block = model.get_decoder()->get_decoder_block(l);
        EXPECT_NE(block->get_gate(), 0.0f) << "block " << l << " world-model gate";
        EXPECT_NE(block->get_gate_h(), 0.0f) << "block " << l << " memory gate";
    }
}

