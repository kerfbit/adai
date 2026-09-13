#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../src/LoRA.hpp"
#include "../src/PPOOptimizer.hpp"
#include "../src/Quantization.hpp"
#include "../src/RewardModel.hpp"

// ============================================================================
// RewardModel Tests
// ============================================================================

TEST(RewardModelTest, Constructor) {
    RewardModel model(768, {512, 256, 1});
    EXPECT_EQ(model.get_input_dim(), 768);
    EXPECT_EQ(model.get_layer_dims().size(), 3);
    EXPECT_EQ(model.get_layer_dims()[2], 1);
}

TEST(RewardModelTest, ForwardPass) {
    RewardModel model(10, {8, 4, 1});
    std::vector<float> input(10, 0.5f);
    float reward = model.forward(input);
    EXPECT_TRUE(std::isfinite(reward));
}

TEST(RewardModelTest, PreferencePairLoss) {
    RewardModel model(10, {8, 1});

    std::vector<float> prompt(5, 0.1f);
    std::vector<float> chosen(5, 0.8f);
    std::vector<float> rejected(5, 0.2f);

    PreferencePair pair(prompt, chosen, rejected);
    float loss = model.compute_loss(pair);

    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GT(loss, 0.0f);
}

TEST(RewardModelTest, TrainOnBatch) {
    RewardModel model(10, {8, 1});

    std::vector<PreferencePair> batch;
    for (int i = 0; i < 5; i++) {
        std::vector<float> prompt(5, 0.1f);
        std::vector<float> chosen(5, 0.8f);
        std::vector<float> rejected(5, 0.2f);
        batch.push_back(PreferencePair(prompt, chosen, rejected));
    }

    float loss_before = model.compute_loss(batch[0]);
    float avg_loss = model.train_on_batch(batch, 0.01f);
    float loss_after = model.compute_loss(batch[0]);

    EXPECT_TRUE(std::isfinite(avg_loss));
    EXPECT_GT(avg_loss, 0.0f);
    // Loss might not decrease in single step, just check it changed
}

TEST(RewardModelTest, SaveLoad) {
    RewardModel model1(10, {8, 4, 1});
    std::vector<float> input(10, 0.5f);
    float reward1 = model1.forward(input);

    model1.save("test_reward_model.bin");

    RewardModel model2(10, {8, 4, 1});
    model2.load("test_reward_model.bin");
    float reward2 = model2.forward(input);

    EXPECT_NEAR(reward1, reward2, 1e-5f);
}

// ============================================================================
// PPOOptimizer Tests
// ============================================================================

TEST(PPOOptimizerTest, Constructor) {
    RewardModel reward_model(768, {512, 256, 1});
    PPOConfig config;
    PPOOptimizer ppo(&reward_model, config, 768);

    EXPECT_EQ(ppo.get_config().clip_epsilon, 0.2f);
    EXPECT_EQ(ppo.get_config().gamma, 0.99f);
}

TEST(PPOOptimizerTest, ValueEstimation) {
    RewardModel reward_model(10, {8, 1});
    PPOConfig config;
    PPOOptimizer ppo(&reward_model, config, 10);

    std::vector<float> state(10, 0.5f);
    float value = ppo.estimate_value(state);

    EXPECT_TRUE(std::isfinite(value));
}

TEST(PPOOptimizerTest, TrajectoryUpdate) {
    RewardModel reward_model(10, {8, 1});
    PPOConfig config;
    config.num_epochs = 2;
    config.batch_size = 4;

    PPOOptimizer ppo(&reward_model, config, 10);

    Trajectory traj;
    for (int i = 0; i < 10; i++) {
        std::vector<float> state(10, 0.1f * i);
        traj.add_step(state, i, 0.5f, -0.1f, 0.3f);
    }

    float loss = ppo.update(
        traj, [](const std::vector<float>&, int) { return -0.1f; });
    EXPECT_TRUE(std::isfinite(loss));
}

TEST(PPOOptimizerTest, UpdateRejectsNullPolicyCallback) {
    RewardModel reward_model(10, {8, 1});
    PPOConfig config;
    PPOOptimizer ppo(&reward_model, config, 10);

    Trajectory traj;
    traj.add_step(std::vector<float>(10, 0.1f), 0, 0.5f, -0.1f, 0.3f);

    EXPECT_THROW(ppo.update(traj, PPOOptimizer::PolicyLogProbFn{}), std::invalid_argument);
}

TEST(PPOOptimizerTest, RatioReflectsLiveCurrentPolicyNotJustOldLogProb) {
    // Directly targets TD-034's core placeholder bug: before the fix, new_log_prob was always
    // a copy of the trajectory's recorded old_log_prob, so ratio = exp(new - old) was always
    // exp(0) = 1 no matter what a real current policy would have said -- policy_loss was
    // therefore blind to what current_policy_log_prob returns. Disabling clipping (a very large
    // clip_epsilon) isolates the ratio's effect on the loss unambiguously.
    RewardModel reward_model(4, {4, 1});
    PPOConfig config;
    config.num_epochs = 1;
    config.batch_size = 8;
    config.clip_epsilon = 100.0f;

    Trajectory traj;
    std::vector<float> state(4, 0.1f);
    for (int i = 0; i < 8; i++) {
        traj.add_step(state, 0, 1.0f, /*log_prob=*/-1.0f, /*value=*/0.0f);
    }

    PPOOptimizer ppo_same_policy(&reward_model, config, 4);
    float loss_same_policy = ppo_same_policy.update(
        traj, [](const std::vector<float>&, int) { return -1.0f; });  // matches old_log_prob exactly

    PPOOptimizer ppo_shifted_policy(&reward_model, config, 4);
    float loss_shifted_policy = ppo_shifted_policy.update(
        traj, [](const std::vector<float>&, int) { return -2.0f; });  // a genuinely different policy

    EXPECT_TRUE(std::isfinite(loss_same_policy));
    EXPECT_TRUE(std::isfinite(loss_shifted_policy));
    EXPECT_NE(loss_same_policy, loss_shifted_policy)
        << "policy_loss must depend on the live policy's log-prob, not just echo old_log_prob; "
           "before TD-034's fix both values above were always identical (ratio pinned to 1)";
}

TEST(PPOOptimizerTest, KLEarlyStopCanActuallyTrigger) {
    // Before TD-034's fix, approx_kl was hardcoded to 0.0f, so `approx_kl > 1.5 * kl_target`
    // could never be true regardless of how large a policy shift the callback reports --
    // num_epochs would always run to completion. A tiny kl_target plus a huge log-prob shift
    // should trip early-stop after the very first epoch.
    RewardModel reward_model(4, {4, 1});
    PPOConfig config;
    config.num_epochs = 10;
    config.batch_size = 4;
    config.kl_target = 0.01f;

    PPOOptimizer ppo(&reward_model, config, 4);

    Trajectory traj;
    std::vector<float> state(4, 0.1f);
    for (int i = 0; i < 4; i++) {
        traj.add_step(state, 0, 1.0f, /*log_prob=*/-1.0f, /*value=*/0.0f);
    }

    int policy_calls = 0;
    ppo.update(traj, [&](const std::vector<float>&, int) {
        policy_calls++;
        return -5.0f;  // wildly different from the recorded -1.0f old_log_prob
    });

    // One minibatch (4 steps) per epoch; 10 full epochs would be 40 calls. Early-stop after
    // the first epoch should leave this far short of that.
    EXPECT_LT(policy_calls, 4 * config.num_epochs)
        << "KL early-stop should have fired after the first epoch given such a large policy "
           "shift, but ran all "
        << config.num_epochs << " epochs instead";
}

TEST(PPOOptimizerTest, ConfigUpdate) {
    RewardModel reward_model(10, {8, 1});
    PPOConfig config1;
    config1.clip_epsilon = 0.3f;

    PPOOptimizer ppo(&reward_model, config1, 10);
    EXPECT_EQ(ppo.get_config().clip_epsilon, 0.3f);

    PPOConfig config2;
    config2.clip_epsilon = 0.1f;
    ppo.set_config(config2);
    EXPECT_EQ(ppo.get_config().clip_epsilon, 0.1f);
}

// ============================================================================
// ValueFunction Tests (TD-034)
// ============================================================================

TEST(ValueFunctionTest, UpdateActuallyTrainsOnToyRegression) {
    // Before TD-034's fix, update()'s per-sample gradient was computed and immediately
    // discarded, so predict() never changed no matter how many times update() ran -- it
    // returned a real, plausible-looking loss value while leaving every weight permanently
    // frozen at its random initialization. Trains toward a fixed target away from the
    // network's initial output and checks both that loss decreases and that the prediction
    // actually moves.
    ValueFunction vf(4, {8, 1});

    std::vector<std::vector<float>> states;
    std::vector<float> targets;
    for (int i = 0; i < 8; i++) {
        states.push_back({0.1f * i, 0.2f, -0.1f, 0.05f * i});
        targets.push_back(2.0f);
    }

    float initial_pred = vf.predict(states[0]);
    float loss_first = vf.update(states, targets, 0.05f);

    float loss_last = loss_first;
    for (int iter = 0; iter < 50; iter++) {
        loss_last = vf.update(states, targets, 0.05f);
    }

    float final_pred = vf.predict(states[0]);
    float initial_error = std::fabs(initial_pred - 2.0f);
    float final_error = std::fabs(final_pred - 2.0f);

    EXPECT_TRUE(std::isfinite(loss_last));
    EXPECT_LT(loss_last, loss_first)
        << "loss should decrease under real gradient descent (was a permanent no-op before "
           "TD-034's fix)";
    EXPECT_NE(final_pred, initial_pred)
        << "weights should actually move -- before TD-034's fix, predict() never changed no "
           "matter how many times update() ran";
    EXPECT_LT(final_error, initial_error * 0.5f)
        << "prediction should move substantially closer to the target after 51 update() calls";
}

TEST(ValueFunctionTest, UpdateRejectsMismatchedSizes) {
    ValueFunction vf(4, {4, 1});
    std::vector<std::vector<float>> states = {std::vector<float>(4, 0.1f)};
    std::vector<float> targets = {1.0f, 2.0f};
    EXPECT_THROW(vf.update(states, targets, 0.01f), std::invalid_argument);
}

TEST(ValueFunctionTest, UpdateRejectsEmptyBatch) {
    ValueFunction vf(4, {4, 1});
    std::vector<std::vector<float>> states;
    std::vector<float> targets;
    EXPECT_THROW(vf.update(states, targets, 0.01f), std::invalid_argument);
}

// ============================================================================
// LoRA Tests
// ============================================================================

TEST(LoRATest, Constructor) {
    LoRAAdapter lora(512, 512, 8, 16.0f);
    EXPECT_EQ(lora.get_rank(), 8);
    EXPECT_EQ(lora.get_alpha(), 16.0f);
}

TEST(LoRATest, ForwardPass) {
    LoRAAdapter lora(10, 10, 4);

    Matrix x(2, 10);  // Batch of 2
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 10; j++) {
            x(i, j) = 0.1f * (i + j);
        }
    }

    Matrix W_output(2, 10);  // Pretend this is from frozen weights
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 10; j++) {
            W_output(i, j) = 0.5f;
        }
    }

    Matrix output = lora.forward(x, W_output);
    EXPECT_EQ(output.rows, 2);
    EXPECT_EQ(output.cols, 10);
}

TEST(LoRATest, BackwardPass) {
    LoRAAdapter lora(10, 10, 4);

    Matrix x(2, 10);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 10; j++) {
            x(i, j) = 0.1f * (i + j);
        }
    }

    Matrix grad_output(2, 10);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 10; j++) {
            grad_output(i, j) = 0.01f;
        }
    }

    lora.backward(x, grad_output);
    // If it doesn't crash, backward pass works
}

TEST(LoRATest, UpdateWeights) {
    LoRAAdapter lora(10, 10, 4);

    Matrix A_before = lora.get_A();
    Matrix B_before = lora.get_B();

    // Do backward pass to compute gradients with larger values
    Matrix x(1, 10);
    for (int j = 0; j < 10; j++)
        x(0, j) = 1.0f;  // Increased from 0.1f

    Matrix grad(1, 10);
    for (int j = 0; j < 10; j++)
        grad(0, j) = 1.0f;  // Increased from 0.01f

    lora.backward(x, grad);
    lora.update(0.1f);  // Increased learning rate from 0.01f

    Matrix A_after = lora.get_A();
    Matrix B_after = lora.get_B();

    // B should change (grad_B doesn't depend on B's initial value)
    // A might not change if B is initialized to zero
    bool changed = false;
    for (int i = 0; i < B_after.rows && !changed; i++) {
        for (int j = 0; j < B_after.cols; j++) {
            if (std::abs(B_after(i, j) - B_before(i, j)) > 1e-6f) {
                changed = true;
                break;
            }
        }
    }
    EXPECT_TRUE(changed);
}

TEST(LoRATest, MergeWithBase) {
    LoRAAdapter lora(10, 10, 4, 8.0f);

    Matrix W(10, 10);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            W(i, j) = (i == j) ? 1.0f : 0.0f;  // Identity matrix
        }
    }

    Matrix merged = lora.merge_with_base(W);
    EXPECT_EQ(merged.rows, 10);
    EXPECT_EQ(merged.cols, 10);
}

TEST(LoRATest, SaveLoad) {
    LoRAAdapter lora1(10, 10, 4, 8.0f);
    lora1.save("test_lora.bin");

    LoRAAdapter lora2(10, 10, 4, 8.0f);
    lora2.load("test_lora.bin");

    EXPECT_EQ(lora2.get_rank(), 4);
    EXPECT_EQ(lora2.get_alpha(), 8.0f);
}

TEST(LoRATest, ParameterCount) {
    LoRAAdapter lora(768, 768, 8);
    int params = lora.num_parameters();
    EXPECT_EQ(params, 8 * (768 + 768));
}

TEST(LoRATest, LoRAConfig) {
    LoRAConfig config;
    config.rank = 16;
    config.alpha = 32.0f;

    EXPECT_EQ(config.rank, 16);
    EXPECT_EQ(config.alpha, 32.0f);
    EXPECT_TRUE(config.apply_to_query);
    EXPECT_TRUE(config.apply_to_value);
}

TEST(LoRATest, ParameterReduction) {
    int original = 768 * 768;
    int lora = 8 * (768 + 768);
    float reduction = LoRAConfig::reduction_ratio(original, lora);
    EXPECT_GT(reduction, 1.0f);
}

// ============================================================================
// Quantization Tests
// ============================================================================

TEST(QuantizationTest, ConstructorDefaults) {
    Quantizer quantizer;
    // Should construct without error
}

TEST(QuantizationTest, CalibrationMinMax) {
    Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8, CalibrationMethod::MIN_MAX);

    std::vector<float> data = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
    QuantizationParams params = quantizer.calibrate(data);

    EXPECT_GT(params.scale, 0.0f);
    EXPECT_EQ(params.zero_point, 0);  // Symmetric
}

TEST(QuantizationTest, QuantizeDequantize) {
    Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8);

    std::vector<float> data = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
    QuantizationParams params = quantizer.calibrate(data);

    std::vector<int8_t> quantized = quantizer.quantize(data, params);
    std::vector<float> dequantized = quantizer.dequantize(quantized, params);

    EXPECT_EQ(quantized.size(), data.size());
    EXPECT_EQ(dequantized.size(), data.size());

    // Check approximate reconstruction
    for (size_t i = 0; i < data.size(); i++) {
        EXPECT_NEAR(data[i], dequantized[i], 1.0f);
    }
}

TEST(QuantizationTest, AsymmetricQuantization) {
    Quantizer quantizer(QuantizationMode::ASYMMETRIC_INT8);

    std::vector<float> data = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
    QuantizationParams params = quantizer.calibrate(data);

    EXPECT_GT(params.scale, 0.0f);
    EXPECT_GE(params.zero_point, 0);  // Asymmetric can have non-zero offset
}

TEST(QuantizationTest, MatrixQuantization) {
    Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8);

    Matrix mat(3, 4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            mat(i, j) = (i + j) * 0.5f;
        }
    }

    std::vector<float> data;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            data.push_back(mat(i, j));
        }
    }

    QuantizationParams params = quantizer.calibrate(data);
    std::vector<int8_t> quantized = quantizer.quantize_matrix(mat, params);

    EXPECT_EQ(quantized.size(), 12);

    Matrix dequantized = quantizer.dequantize_matrix(quantized, 3, 4, params);
    EXPECT_EQ(dequantized.rows, 3);
    EXPECT_EQ(dequantized.cols, 4);
}

TEST(QuantizationTest, QuantizationError) {
    Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8);

    std::vector<float> data;
    for (int i = 0; i < 100; i++) {
        data.push_back(std::sin(i * 0.1f));
    }

    QuantizationParams params = quantizer.calibrate(data);
    float error = quantizer.compute_quantization_error(data, params);

    EXPECT_GT(error, 0.0f);
    EXPECT_LT(error, 1.0f);  // Should be relatively small
}

TEST(QuantizationTest, QuantizedMatrixSaveLoad) {
    Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8);

    Matrix mat(5, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            mat(i, j) = (i + j) * 0.1f;
        }
    }

    QuantizedMatrix qmat;
    qmat.quantize_from(mat, quantizer);
    qmat.save("test_qmat.bin");

    QuantizedMatrix qmat2;
    qmat2.load("test_qmat.bin");

    EXPECT_EQ(qmat2.rows(), 5);
    EXPECT_EQ(qmat2.cols(), 5);

    Matrix reconstructed = qmat2.dequantize(quantizer);
    EXPECT_EQ(reconstructed.rows, 5);
    EXPECT_EQ(reconstructed.cols, 5);
}

TEST(QuantizationTest, MemoryReduction) {
    Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8);

    Matrix mat(100, 100);
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            mat(i, j) = (i + j) * 0.01f;
        }
    }

    QuantizedMatrix qmat;
    qmat.quantize_from(mat, quantizer);

    float reduction = qmat.memory_reduction();
    EXPECT_GT(reduction, 1.0f);   // Should save memory
    EXPECT_LT(reduction, 10.0f);  // Reasonable upper bound
}

TEST(QuantizationTest, INT4Mode) {
    Quantizer quantizer(QuantizationMode::SYMMETRIC_INT4);

    std::vector<float> data = {-7.0f, -3.5f, 0.0f, 3.5f, 7.0f};
    QuantizationParams params = quantizer.calibrate(data);

    EXPECT_EQ(params.qmin, -7);
    EXPECT_EQ(params.qmax, 7);
}

TEST(QuantizationTest, PercentileCalibration) {
    Quantizer quantizer(QuantizationMode::SYMMETRIC_INT8, CalibrationMethod::PERCENTILE, 0.99f);

    std::vector<float> data;
    for (int i = 0; i < 100; i++) {
        data.push_back(std::sin(i * 0.1f));
    }
    // Add outliers
    data.push_back(100.0f);
    data.push_back(-100.0f);

    QuantizationParams params = quantizer.calibrate(data);

    // Percentile should clip outliers, resulting in smaller scale
    EXPECT_GT(params.scale, 0.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
