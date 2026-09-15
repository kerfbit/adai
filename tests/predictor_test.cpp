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
#include <random>
#include <stdexcept>
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
