/**
 * @file predictor_test.cpp
 * @brief Tests for Predictor (TD-176 / LJ-1b) — the LeJEPA world-model plan's embedding-space
 *        predictor, from docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 2.
 *
 * TD-176's own Action Items ask for three things specifically: forward()/backward()/
 * update_weights()/zero_grad() implemented per the proposal's interface, a gradient check on a
 * toy embedding pair, and register_parameters_with_optimizer() wired per the codebase's standard
 * pattern (FeedForward::set_optimizer(), which auto-registers — see EncoderBlock's own use of the
 * identical convention).
 */

#include "../src/Predictor.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <algorithm>
#include "../src/Matrix.hpp"
#include "../src/Optimizer.hpp"

TEST(PredictorTest, ConstructorRejectsNonPositiveDimensions) {
    EXPECT_THROW(Predictor(0, 32), std::invalid_argument);
    EXPECT_THROW(Predictor(-4, 32), std::invalid_argument);
    EXPECT_THROW(Predictor(16, 0), std::invalid_argument);
    EXPECT_THROW(Predictor(16, -1), std::invalid_argument);
}

TEST(PredictorTest, AccessorsReturnConstructorArguments) {
    Predictor predictor(32, 48);
    EXPECT_EQ(predictor.get_d_model(), 32);
    EXPECT_EQ(predictor.get_hidden_dim(), 48);
}

TEST(PredictorTest, ForwardRejectsMismatchedDModel) {
    Predictor predictor(16, 32);
    Matrix wrong_shape(4, 8);
    EXPECT_THROW(predictor.forward(wrong_shape), std::invalid_argument);
}

TEST(PredictorTest, BackwardRejectsMismatchedDModel) {
    Predictor predictor(16, 32);
    Matrix context(4, 16);
    predictor.forward(context);

    Matrix wrong_shape(4, 8);
    EXPECT_THROW(predictor.backward(wrong_shape), std::invalid_argument);
}

TEST(PredictorTest, ForwardPreservesShape) {
    const int d_model = 12;
    const int batch = 5;
    Predictor predictor(d_model, 24);

    Matrix context(batch, d_model);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < d_model; ++j) {
            context.data[i][j] = 0.1f * static_cast<float>(i + j);
        }
    }

    Matrix predicted = predictor.forward(context);

    EXPECT_EQ(predicted.rows, batch);
    EXPECT_EQ(predicted.cols, d_model);
}

TEST(PredictorTest, ForwardHandlesSingleEmbedding) {
    // The typical single-example case: predicting one context view's embedding, not a batch.
    const int d_model = 8;
    Predictor predictor(d_model, 16);

    Matrix context(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        context.data[0][j] = 0.5f;
    }

    Matrix predicted = predictor.forward(context);

    EXPECT_EQ(predicted.rows, 1);
    EXPECT_EQ(predicted.cols, d_model);
}

TEST(PredictorTest, ForwardIsDeterministicForFixedWeights) {
    Predictor predictor(10, 20);

    Matrix context(3, 10);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 10; ++j) {
            context.data[i][j] = 0.05f * static_cast<float>(i * 10 + j);
        }
    }

    Matrix out1 = predictor.forward(context);
    Matrix out2 = predictor.forward(context);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 10; ++j) {
            EXPECT_FLOAT_EQ(out1.data[i][j], out2.data[i][j]);
        }
    }
}

TEST(PredictorTest, BackwardPreservesShape) {
    const int d_model = 10;
    const int batch = 4;
    Predictor predictor(d_model, 20);

    Matrix context(batch, d_model);
    predictor.forward(context);

    Matrix grad_output(batch, d_model);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output.data[i][j] = 0.01f;
        }
    }

    Matrix grad_input = predictor.backward(grad_output);

    EXPECT_EQ(grad_input.rows, batch);
    EXPECT_EQ(grad_input.cols, d_model);
}

// The gradient check TD-176's own Action Items specifically ask for: a toy embedding pair
// (a small batch of context embeddings, standing in for a context-view/target-view pair) run
// through forward()/backward(), verified against a central-difference numerical estimate of
// d(sum(output .* grad_output))/d(embeddings) — the standard technique for validating an
// arbitrary upstream gradient without needing a real loss function.
TEST(PredictorTest, GradientMatchesFiniteDifference) {
    const int d_model = 5;
    const int hidden_dim = 8;
    const int batch = 4;
    Predictor predictor(d_model, hidden_dim);

    Matrix embeddings(batch, d_model);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < d_model; ++j) {
            embeddings.data[i][j] = dist(gen);
        }
    }

    // A fixed, non-uniform upstream gradient — standing in for dL/d(predicted_embedding) from
    // whatever loss (e.g. LeJEPAEncoder::train_step's predictor_loss) sits on top of this.
    Matrix grad_output(batch, d_model);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output.data[i][j] = 0.1f * static_cast<float>(i - j);
        }
    }

    predictor.forward(embeddings);
    Matrix analytic_grad = predictor.backward(grad_output);
    ASSERT_EQ(analytic_grad.rows, batch);
    ASSERT_EQ(analytic_grad.cols, d_model);

    auto weighted_output_sum = [&](const Matrix& e) {
        Matrix out = predictor.forward(e);
        float total = 0.0f;
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < d_model; ++j) {
                total += out.data[i][j] * grad_output.data[i][j];
            }
        }
        return total;
    };

    const float epsilon = 1e-3f;
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < d_model; ++j) {
            const float orig = embeddings.data[i][j];

            embeddings.data[i][j] = orig + epsilon;
            const float loss_plus = weighted_output_sum(embeddings);

            embeddings.data[i][j] = orig - epsilon;
            const float loss_minus = weighted_output_sum(embeddings);

            embeddings.data[i][j] = orig;

            const float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
            EXPECT_NEAR(analytic_grad.data[i][j], numerical_grad, 5e-3f)
                << "mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(PredictorTest, UpdateWeightsChangesSubsequentOutput) {
    Predictor predictor(16, 32);

    Matrix context(3, 16);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 16; ++j) {
            context.data[i][j] = 0.2f;
        }
    }

    Matrix out_before = predictor.forward(context);

    Matrix grad_output(3, 16);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 16; ++j) {
            grad_output.data[i][j] = 1.0f;
        }
    }
    predictor.backward(grad_output);
    predictor.update_weights();

    Matrix out_after = predictor.forward(context);

    bool changed = false;
    for (int i = 0; i < 3 && !changed; ++i) {
        for (int j = 0; j < 16 && !changed; ++j) {
            if (std::abs(out_after.data[i][j] - out_before.data[i][j]) > 1e-6f) {
                changed = true;
            }
        }
    }
    EXPECT_TRUE(changed);
}

TEST(PredictorTest, ZeroGradDoesNotThrowAndUpdateStaysWellBehaved) {
    Predictor predictor(8, 16);
    predictor.zero_grad();  // safe even with no prior forward/backward

    Matrix context(2, 8);
    predictor.forward(context);
    Matrix grad_output(2, 8);
    predictor.backward(grad_output);
    predictor.zero_grad();

    // Grads were just zeroed, so this update should be a no-op (no NaN/throw either way).
    EXPECT_NO_THROW(predictor.update_weights());
}

TEST(PredictorTest, RegisterParametersWithOptimizerEnablesTraining) {
    Predictor predictor(16, 32);
    Optimizer optimizer(OptimizerType::ADAM, 0.001f);
    predictor.register_parameters_with_optimizer(optimizer);

    Matrix context(3, 16);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 16; ++j) {
            context.data[i][j] = 0.3f;
        }
    }

    Matrix out_before = predictor.forward(context);

    Matrix grad_output(3, 16);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 16; ++j) {
            grad_output.data[i][j] = 0.5f;
        }
    }
    predictor.backward(grad_output);
    EXPECT_NO_THROW(predictor.update_weights());

    Matrix out_after = predictor.forward(context);

    bool changed = false;
    for (int i = 0; i < 3 && !changed; ++i) {
        for (int j = 0; j < 16 && !changed; ++j) {
            if (std::abs(out_after.data[i][j] - out_before.data[i][j]) > 1e-6f) {
                changed = true;
            }
        }
    }
    EXPECT_TRUE(changed);
}

TEST(PredictorTest, DifferentInstancesProduceDifferentOutputs) {
    // No shared seed in the interface — two independently-constructed instances get independent
    // random initialization (Xavier/He, via FeedForward), so they shouldn't agree by coincidence.
    Predictor predictor_a(16, 32);
    Predictor predictor_b(16, 32);

    Matrix context(2, 16);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 16; ++j) {
            context.data[i][j] = 0.4f;
        }
    }

    Matrix out_a = predictor_a.forward(context);
    Matrix out_b = predictor_b.forward(context);

    bool differ = false;
    for (int i = 0; i < 2 && !differ; ++i) {
        for (int j = 0; j < 16 && !differ; ++j) {
            if (std::abs(out_a.data[i][j] - out_b.data[i][j]) > 1e-5f) {
                differ = true;
            }
        }
    }
    EXPECT_TRUE(differ);
}

// ============================================================================
// TD-427: save()/load()
// ============================================================================

namespace {

Matrix predictor_probe(int d_model) {
    Matrix x(2, d_model);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < d_model; ++j)
            x(i, j) = 0.1f * std::sin(static_cast<float>(i * d_model + j));
    return x;
}

bool same(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols)
        return false;
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            if (a(i, j) != b(i, j))
                return false;
    return true;
}

std::string temp_file(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

}  // namespace

TEST(PredictorTest, SaveLoadRoundTripsTheOutput) {
    const std::string path = temp_file("adai_predictor_roundtrip.bin");
    Predictor trained(16, 24);
    Predictor fresh(16, 24);
    const Matrix x = predictor_probe(16);
    ASSERT_FALSE(same(trained.forward(x), fresh.forward(x)));  // independent random inits

    trained.save(path);
    ASSERT_TRUE(fresh.load(path));
    EXPECT_TRUE(same(trained.forward(x), fresh.forward(x)));
    std::filesystem::remove(path);
}

TEST(PredictorTest, LoadOfAMissingFileReturnsFalse) {
    Predictor predictor(16, 24);
    EXPECT_FALSE(predictor.load(temp_file("adai_predictor_does_not_exist.bin")));
}

TEST(PredictorTest, LoadRejectsTruncatedAndMismatchedFilesWithoutChangingWeights) {
    const std::string path = temp_file("adai_predictor_bad.bin");
    Predictor source(16, 24), target(16, 24);
    const Matrix x = predictor_probe(16);
    const Matrix before = target.forward(x);
    source.save(path);

    std::filesystem::resize_file(path, std::filesystem::file_size(path) / 2);  // truncated
    EXPECT_THROW(target.load(path), std::runtime_error);
    EXPECT_TRUE(same(target.forward(x), before));

    Predictor other_shape(16, 32);
    other_shape.save(path);
    EXPECT_THROW(target.load(path), std::runtime_error);
    EXPECT_TRUE(same(target.forward(x), before));
    std::filesystem::remove(path);
}


#ifdef ADAI_ENABLE_GPU
// TD-585: the predictor's device path, against its host path, on the oneAPI CPU device in a SYCL build
// (ADAI_SYCL_ALLOW_CPU_DEVICE=1); skipped without a device.
#include "../src/GPUOptimizer.hpp"
#include "../src/gpu/GPUUtils.hpp"

namespace {

bool predictor_gpu_ready() {
    return adai::gpu::GPUManager::probe() && adai::gpu::GPUManager::initialize(0, 0.5f);
}

Matrix random_input(int rows, int cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    Matrix m(rows, cols);
    for (auto& row : m.data)
        for (auto& v : row)
            v = d(rng);
    return m;
}

float max_abs_diff(const Matrix& a, const Matrix& b) {
    float worst = 0.0f;
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            worst = std::max(worst, std::abs(a(i, j) - b(i, j)));
    return worst;
}

// A second predictor with the first one's weights (through save()/load()).
Predictor clone_of(const Predictor& p, int d_model, int hidden, const std::string& tag) {
    const std::string path =
        (std::filesystem::temp_directory_path() / ("adai_predictor_gpu_" + tag + "_" + std::to_string(::getpid()) + ".bin"))
            .string();
    p.save(path);
    Predictor copy(d_model, hidden);
    EXPECT_TRUE(copy.load(path));
    std::filesystem::remove(path);
    return copy;
}

}  // namespace

TEST(PredictorGpuTest, ForwardAndBackwardMatchTheHost) {
    if (!predictor_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const int d_model = 16, hidden = 24, rows = 5;
    Predictor p(d_model, hidden);
    const Matrix x = random_input(rows, d_model, 1);
    const Matrix dout = random_input(rows, d_model, 2);

    const Matrix host_out = p.forward(x);
    const Matrix host_dx = p.backward(dout);
    const Matrix dev_out = Matrix::from_gpu(p.gpu_forward(x.to_gpu()));
    const Matrix dev_dx = Matrix::from_gpu(p.gpu_backward(dout.to_gpu()));
    EXPECT_LT(max_abs_diff(dev_out, host_out), 1e-5f);
    EXPECT_LT(max_abs_diff(dev_dx, host_dx), 1e-5f);
}

// One optimizer step from the device gradients equals one from the host gradients: the gradients land
// in the device mirrors, and every parameter is a device group.
TEST(PredictorGpuTest, ADeviceStepMatchesAHostStep) {
    if (!predictor_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const int d_model = 16, hidden = 24, rows = 7;
    Predictor host_p(d_model, hidden);
    Predictor dev_p = clone_of(host_p, d_model, hidden, "step");
    const Matrix x = random_input(rows, d_model, 3);
    const Matrix dout = random_input(rows, d_model, 4);

    Optimizer host_opt(OptimizerType::ADAM, 0.01f);
    host_p.register_parameters_with_optimizer(host_opt);
    host_p.forward(x);
    host_p.backward(dout);
    host_p.update_weights();

    Optimizer dev_opt(OptimizerType::ADAM, 0.01f);
    dev_p.register_parameters_with_optimizer(dev_opt);
    dev_p.gpu_zero_grads();
    std::vector<adai::gpu::GPUParamRef> params;
    dev_p.gpu_collect_parameters(params);
    ASSERT_EQ(params.size(), 4u) << "W1, W2, b1, b2";
    GPUOptimizer gpu_opt(dev_opt, params);
    EXPECT_EQ(gpu_opt.host_group_count(), 0u);
    dev_p.gpu_forward(x.to_gpu());
    dev_p.gpu_backward(dout.to_gpu());
    gpu_opt.gradient_sq_norms();
    gpu_opt.step(1.0f);
    dev_p.gpu_download_weights();

    const Matrix probe = random_input(3, d_model, 5);
    EXPECT_LT(max_abs_diff(dev_p.forward(probe), host_p.forward(probe)), 1e-5f);
}

TEST(PredictorGpuTest, LoadedWeightsReachTheDevice) {
    if (!predictor_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const int d_model = 16, hidden = 24;
    Predictor source(d_model, hidden);
    Predictor p(d_model, hidden);
    const Matrix x = random_input(4, d_model, 6);
    p.gpu_forward(x.to_gpu());  // uploads p's own weights first
    const std::string path =
        (std::filesystem::temp_directory_path() / ("adai_predictor_gpu_load_" + std::to_string(::getpid()) + ".bin"))
            .string();
    source.save(path);
    ASSERT_TRUE(p.load(path));
    std::filesystem::remove(path);
    EXPECT_LT(max_abs_diff(Matrix::from_gpu(p.gpu_forward(x.to_gpu())), source.forward(x)), 1e-5f);
}

TEST(PredictorGpuTest, BadShapesAndBackwardBeforeForwardThrow) {
    if (!predictor_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    Predictor p(16, 24);
    EXPECT_THROW(p.gpu_backward(random_input(2, 16, 7).to_gpu()), std::logic_error);
    EXPECT_THROW(p.gpu_forward(random_input(2, 15, 8).to_gpu()), std::invalid_argument);
    p.gpu_forward(random_input(2, 16, 9).to_gpu());
    EXPECT_THROW(p.gpu_backward(random_input(3, 16, 10).to_gpu()), std::invalid_argument);
}
#endif  // ADAI_ENABLE_GPU
