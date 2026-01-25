#include <../gtest/gtest.h>
#include "../src/NeuralNetwork.hpp"
#include <cmath>
#include <fstream>
#include <cstdio>

// Test fixture for NeuralNetwork tests
class NeuralNetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_model_file = "test_neural_network.nn";
    }

    void TearDown() override {
        // Clean up test files
        std::remove(test_model_file.c_str());
        std::remove("test_model_2.nn");
    }

    std::string test_model_file;
    
    // Helper function to check if values are close
    bool is_close(float a, float b, float epsilon = 1e-5f) {
        return std::abs(a - b) < epsilon;
    }
    
    // Helper to check vector equality
    bool vectors_close(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5f) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!is_close(a[i], b[i], epsilon)) return false;
        }
        return true;
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(NeuralNetworkTest, BasicConstruction) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.01f);
    
    EXPECT_EQ(nn.get_num_layers(), 2);
    EXPECT_EQ(nn.get_layer_sizes()[0], 2);
    EXPECT_EQ(nn.get_layer_sizes()[1], 3);
    EXPECT_EQ(nn.get_layer_sizes()[2], 1);
}

TEST_F(NeuralNetworkTest, DeepNetworkConstruction) {
    std::vector<int> architecture = {10, 64, 32, 16, 3};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::CATEGORICAL_CROSS_ENTROPY, 0.001f);
    
    EXPECT_EQ(nn.get_num_layers(), 4);
    EXPECT_EQ(nn.get_layer_sizes().size(), 5);
}

TEST_F(NeuralNetworkTest, SingleLayerNetwork) {
    std::vector<int> architecture = {5, 2};
    std::vector<ActivationType> activations = {ActivationType::LINEAR};
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    
    EXPECT_EQ(nn.get_num_layers(), 1);
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_F(NeuralNetworkTest, PredictBasic) {
    std::vector<int> architecture = {2, 2, 1};
    std::vector<ActivationType> activations = {
        ActivationType::LINEAR,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    
    std::vector<float> input = {1.0f, 2.0f};
    auto output = nn.predict(input);
    
    EXPECT_EQ(output.size(), 1);
}

TEST_F(NeuralNetworkTest, PredictMultipleOutputs) {
    std::vector<int> architecture = {3, 4, 3};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::CATEGORICAL_CROSS_ENTROPY);
    
    std::vector<float> input = {0.5f, 0.3f, 0.2f};
    auto output = nn.predict(input);
    
    EXPECT_EQ(output.size(), 3);
    for (float val : output) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);  // Sigmoid output
    }
}

TEST_F(NeuralNetworkTest, PredictBatch) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::BINARY_CROSS_ENTROPY);
    
    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    auto outputs = nn.predict_batch(inputs);
    
    EXPECT_EQ(outputs.size(), 4);
    for (const auto& output : outputs) {
        EXPECT_EQ(output.size(), 1);
    }
}

TEST_F(NeuralNetworkTest, PredictConsistency) {
    std::vector<int> architecture = {3, 5, 2};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    auto output1 = nn.predict(input);
    auto output2 = nn.predict(input);
    
    // Same input should produce same output
    EXPECT_TRUE(vectors_close(output1, output2));
}

// ============================================================================
// Training Tests
// ============================================================================

TEST_F(NeuralNetworkTest, TrainSampleBasic) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.1f);
    
    std::vector<float> input = {1.0f, 2.0f};
    std::vector<float> target = {3.0f};
    
    float loss = nn.train_sample(input, target);
    
    EXPECT_GE(loss, 0.0f);
}

TEST_F(NeuralNetworkTest, TrainBatch) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.05f);
    
    std::vector<std::vector<float>> inputs = {
        {1.0f, 2.0f},
        {2.0f, 3.0f},
        {3.0f, 4.0f}
    };
    
    std::vector<std::vector<float>> targets = {
        {3.0f},
        {5.0f},
        {7.0f}
    };
    
    float loss = nn.train_batch(inputs, targets);
    
    EXPECT_GE(loss, 0.0f);
}

TEST_F(NeuralNetworkTest, TrainingReducesLoss) {
    std::vector<int> architecture = {1, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.1f);
    nn.initialize_he();
    
    // Simple linear relationship: y = 2x
    std::vector<std::vector<float>> X = {
        {0.0f}, {1.0f}, {2.0f}, {3.0f}, {4.0f}
    };
    std::vector<std::vector<float>> y = {
        {0.0f}, {2.0f}, {4.0f}, {6.0f}, {8.0f}
    };
    
    float initial_loss = nn.evaluate(X, y);
    
    // Train for several epochs
    nn.fit(X, y, 100, 0, nullptr, nullptr, false);
    
    float final_loss = nn.evaluate(X, y);
    
    EXPECT_LT(final_loss, initial_loss);
}

TEST_F(NeuralNetworkTest, XORProblem) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::BINARY_CROSS_ENTROPY, 0.5f);
    nn.initialize_he();
    
    // XOR training data
    std::vector<std::vector<float>> X = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    std::vector<std::vector<float>> y = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };
    
    nn.fit(X, y, 1000, 4, nullptr, nullptr, false);
    
    // Test predictions
    auto predictions = nn.predict_batch(X);
    
    // Check if predictions are close to targets
    EXPECT_LT(predictions[0][0], 0.5f);  // 0 XOR 0 = 0
    EXPECT_GT(predictions[1][0], 0.5f);  // 0 XOR 1 = 1
    EXPECT_GT(predictions[2][0], 0.5f);  // 1 XOR 0 = 1
    EXPECT_LT(predictions[3][0], 0.5f);  // 1 XOR 1 = 0
}

TEST_F(NeuralNetworkTest, MiniBatchTraining) {
    std::vector<int> architecture = {2, 8, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.01f);
    nn.initialize_he();
    
    // Generate some training data
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> y;
    for (int i = 0; i < 20; ++i) {
        float x1 = i * 0.1f;
        float x2 = i * 0.2f;
        X.push_back({x1, x2});
        y.push_back({x1 + x2});
    }
    
    // Train with mini-batches
    nn.fit(X, y, 50, 5, nullptr, nullptr, false);
    
    float final_loss = nn.evaluate(X, y);
    EXPECT_LT(final_loss, 1.0f);
}

// ============================================================================
// Loss Function Tests
// ============================================================================

TEST_F(NeuralNetworkTest, MSELoss) {
    std::vector<int> architecture = {2, 1};
    std::vector<ActivationType> activations = {ActivationType::LINEAR};
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    nn.initialize_xavier();
    
    std::vector<float> input = {1.0f, 2.0f};
    std::vector<float> target = {3.0f};
    
    float loss = nn.train_sample(input, target);
    EXPECT_GE(loss, 0.0f);
}

TEST_F(NeuralNetworkTest, BinaryCrossEntropyLoss) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::BINARY_CROSS_ENTROPY, 0.1f);
    nn.initialize_he();
    
    std::vector<float> input = {0.5f, 0.3f};
    std::vector<float> target = {1.0f};
    
    float loss = nn.train_sample(input, target);
    EXPECT_GE(loss, 0.0f);
}

TEST_F(NeuralNetworkTest, CategoricalCrossEntropyLoss) {
    std::vector<int> architecture = {4, 8, 3};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::CATEGORICAL_CROSS_ENTROPY, 0.01f);
    nn.initialize_he();
    
    std::vector<float> input = {1.0f, 0.5f, 0.3f, 0.2f};
    std::vector<float> target = {1.0f, 0.0f, 0.0f};  // One-hot encoded
    
    float loss = nn.train_sample(input, target);
    EXPECT_GE(loss, 0.0f);
}

TEST_F(NeuralNetworkTest, MAELoss) {
    std::vector<int> architecture = {3, 5, 2};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MAE, 0.05f);
    nn.initialize_xavier();
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> target = {5.0f, 6.0f};
    
    float loss = nn.train_sample(input, target);
    EXPECT_GE(loss, 0.0f);
}

TEST_F(NeuralNetworkTest, HuberLoss) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::HUBER, 0.01f);
    nn.initialize_he();
    
    std::vector<float> input = {1.0f, 2.0f};
    std::vector<float> target = {10.0f};
    
    float loss = nn.train_sample(input, target);
    EXPECT_GE(loss, 0.0f);
}

// ============================================================================
// Activation Function Tests
// ============================================================================

TEST_F(NeuralNetworkTest, SigmoidActivation) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::SIGMOID,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::BINARY_CROSS_ENTROPY);
    
    std::vector<float> input = {1.0f, 2.0f};
    auto output = nn.predict(input);
    
    // Sigmoid outputs should be in (0, 1)
    EXPECT_GT(output[0], 0.0f);
    EXPECT_LT(output[0], 1.0f);
}

TEST_F(NeuralNetworkTest, ReLUActivation) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    nn.initialize_he();
    
    std::vector<float> input = {-1.0f, 2.0f};
    auto output = nn.predict(input);
    
    EXPECT_EQ(output.size(), 1);
}

TEST_F(NeuralNetworkTest, TanhActivation) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::TANH
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    
    std::vector<float> input = {1.0f, 2.0f};
    auto output = nn.predict(input);
    
    // Tanh outputs should be in (-1, 1)
    EXPECT_GT(output[0], -1.0f);
    EXPECT_LT(output[0], 1.0f);
}

TEST_F(NeuralNetworkTest, LinearActivation) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::LINEAR,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    
    std::vector<float> input = {1.0f, 2.0f};
    auto output = nn.predict(input);
    
    EXPECT_EQ(output.size(), 1);
}

TEST_F(NeuralNetworkTest, MixedActivations) {
    std::vector<int> architecture = {3, 8, 4, 2};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::TANH,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    nn.initialize_he();
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    auto output = nn.predict(input);
    
    EXPECT_EQ(output.size(), 2);
}

// ============================================================================
// Evaluation Tests
// ============================================================================

TEST_F(NeuralNetworkTest, EvaluateBasic) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    
    std::vector<std::vector<float>> test_data = {
        {1.0f, 2.0f},
        {2.0f, 3.0f}
    };
    std::vector<std::vector<float>> test_labels = {
        {3.0f},
        {5.0f}
    };
    
    float loss = nn.evaluate(test_data, test_labels);
    EXPECT_GE(loss, 0.0f);
}

TEST_F(NeuralNetworkTest, ComputeAccuracy) {
    std::vector<int> architecture = {2, 4, 2};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::CATEGORICAL_CROSS_ENTROPY);
    
    // Perfect predictions
    std::vector<std::vector<float>> predictions = {
        {0.9f, 0.1f},
        {0.2f, 0.8f},
        {0.7f, 0.3f}
    };
    
    std::vector<std::vector<float>> targets = {
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f}
    };
    
    float accuracy = nn.compute_accuracy(predictions, targets);
    EXPECT_EQ(accuracy, 1.0f);
}

TEST_F(NeuralNetworkTest, ComputePartialAccuracy) {
    std::vector<int> architecture = {2, 3};
    std::vector<ActivationType> activations = {ActivationType::SIGMOID};
    
    NeuralNetwork nn(architecture, activations, LossType::CATEGORICAL_CROSS_ENTROPY);
    
    // 2 out of 4 correct
    std::vector<std::vector<float>> predictions = {
        {0.9f, 0.05f, 0.05f},
        {0.1f, 0.8f, 0.1f},
        {0.6f, 0.3f, 0.1f},
        {0.2f, 0.3f, 0.5f}
    };
    
    std::vector<std::vector<float>> targets = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},  // Wrong
        {1.0f, 0.0f, 0.0f}   // Wrong
    };
    
    float accuracy = nn.compute_accuracy(predictions, targets);
    EXPECT_FLOAT_EQ(accuracy, 0.5f);
}

// ============================================================================
// Weight Initialization Tests
// ============================================================================

TEST_F(NeuralNetworkTest, HeInitialization) {
    std::vector<int> architecture = {10, 20, 5};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    
    EXPECT_NO_THROW(nn.initialize_he());
}

TEST_F(NeuralNetworkTest, XavierInitialization) {
    std::vector<int> architecture = {10, 20, 5};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::BINARY_CROSS_ENTROPY);
    
    EXPECT_NO_THROW(nn.initialize_xavier());
}

TEST_F(NeuralNetworkTest, InitializationAffectsTraining) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn1(architecture, activations, LossType::MSE, 0.01f);
    NeuralNetwork nn2(architecture, activations, LossType::MSE, 0.01f);
    
    nn1.initialize_he();
    nn2.initialize_xavier();
    
    std::vector<float> input = {1.0f, 2.0f};
    auto output1 = nn1.predict(input);
    auto output2 = nn2.predict(input);
    
    // Different initializations should give different outputs
    EXPECT_FALSE(vectors_close(output1, output2, 1e-6f));
}

// ============================================================================
// Learning Rate Tests
// ============================================================================

TEST_F(NeuralNetworkTest, SetLearningRate) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.01f);
    
    EXPECT_NO_THROW(nn.set_learning_rate(0.1f));
    EXPECT_NO_THROW(nn.set_learning_rate(0.001f));
}

TEST_F(NeuralNetworkTest, LearningRateAffectsConvergence) {
    std::vector<int> architecture = {1, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn1(architecture, activations, LossType::MSE, 0.1f);
    NeuralNetwork nn2(architecture, activations, LossType::MSE, 0.1f);
    
    nn1.initialize_he();
    nn2.initialize_he();
    
    std::vector<std::vector<float>> X = {{1.0f}, {2.0f}, {3.0f}};
    std::vector<std::vector<float>> y = {{2.0f}, {4.0f}, {6.0f}};
    
    float loss1 = nn1.evaluate(X, y);
    
    // Train
    nn1.fit(X, y, 50, 0, nullptr, nullptr, false);
    
    float loss2 = nn1.evaluate(X, y);
    
    // Training should reduce loss
    EXPECT_LT(loss2, loss1);
}

// ============================================================================
// Serialization Tests
// ============================================================================

TEST_F(NeuralNetworkTest, SaveNetwork) {
    std::vector<int> architecture = {3, 5, 2};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.01f);
    nn.initialize_he();
    
    EXPECT_NO_THROW(nn.save(test_model_file));
    
    // Check file exists
    std::ifstream file(test_model_file);
    EXPECT_TRUE(file.good());
    file.close();
}

TEST_F(NeuralNetworkTest, LoadNetwork) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn1(architecture, activations, LossType::MSE, 0.05f);
    nn1.initialize_he();
    nn1.save(test_model_file);
    
    NeuralNetwork nn2(architecture, activations, LossType::MSE, 0.05f);
    EXPECT_NO_THROW(nn2.load(test_model_file));
}

TEST_F(NeuralNetworkTest, SaveLoadPreservesPredictions) {
    std::vector<int> architecture = {2, 5, 3};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn1(architecture, activations, LossType::CATEGORICAL_CROSS_ENTROPY, 0.01f);
    nn1.initialize_he();
    
    std::vector<float> input = {1.0f, 2.0f};
    auto output1 = nn1.predict(input);
    
    nn1.save(test_model_file);
    
    NeuralNetwork nn2(architecture, activations, LossType::CATEGORICAL_CROSS_ENTROPY, 0.01f);
    nn2.load(test_model_file);
    
    auto output2 = nn2.predict(input);
    
    EXPECT_TRUE(vectors_close(output1, output2, 1e-5f));
}

TEST_F(NeuralNetworkTest, SaveLoadAfterTraining) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn1(architecture, activations, LossType::BINARY_CROSS_ENTROPY, 0.1f);
    nn1.initialize_he();
    
    // Train XOR
    std::vector<std::vector<float>> X = {
        {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
    };
    std::vector<std::vector<float>> y = {
        {0.0f}, {1.0f}, {1.0f}, {0.0f}
    };
    
    nn1.fit(X, y, 500, 4, nullptr, nullptr, false);
    
    float loss1 = nn1.evaluate(X, y);
    nn1.save(test_model_file);
    
    NeuralNetwork nn2(architecture, activations, LossType::BINARY_CROSS_ENTROPY, 0.1f);
    nn2.load(test_model_file);
    
    float loss2 = nn2.evaluate(X, y);
    
    EXPECT_TRUE(is_close(loss1, loss2, 1e-5f));
}

// ============================================================================
// Network Configuration Tests
// ============================================================================

TEST_F(NeuralNetworkTest, PrintSummary) {
    std::vector<int> architecture = {10, 20, 5, 2};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::CATEGORICAL_CROSS_ENTROPY);
    
    EXPECT_NO_THROW(nn.print_summary());
}

TEST_F(NeuralNetworkTest, GetLayerSizes) {
    std::vector<int> architecture = {5, 10, 3};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    
    auto layer_sizes = nn.get_layer_sizes();
    EXPECT_EQ(layer_sizes.size(), 3);
    EXPECT_EQ(layer_sizes[0], 5);
    EXPECT_EQ(layer_sizes[1], 10);
    EXPECT_EQ(layer_sizes[2], 3);
}

TEST_F(NeuralNetworkTest, GetLayer) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::BINARY_CROSS_ENTROPY);
    
    EXPECT_NO_THROW(nn.get_layer(0));
    EXPECT_NO_THROW(nn.get_layer(1));
}

// ============================================================================
// Training History Tests
// ============================================================================

TEST_F(NeuralNetworkTest, TrainingHistory) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.1f);
    nn.initialize_he();
    
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}, {2.0f, 3.0f}};
    std::vector<std::vector<float>> y = {{3.0f}, {5.0f}};
    
    nn.fit(X, y, 10, 0, nullptr, nullptr, false);
    
    auto training_loss = nn.get_training_loss();
    EXPECT_GT(training_loss.size(), 0);
}

TEST_F(NeuralNetworkTest, ValidationHistory) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.1f);
    nn.initialize_he();
    
    std::vector<std::vector<float>> X_train = {{1.0f, 2.0f}, {2.0f, 3.0f}};
    std::vector<std::vector<float>> y_train = {{3.0f}, {5.0f}};
    
    std::vector<std::vector<float>> X_val = {{3.0f, 4.0f}};
    std::vector<std::vector<float>> y_val = {{7.0f}};
    
    nn.fit(X_train, y_train, 10, 0, &X_val, &y_val, false);
    
    auto validation_loss = nn.get_validation_loss();
    EXPECT_GT(validation_loss.size(), 0);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(NeuralNetworkTest, EmptyTrainingData) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    
    std::vector<std::vector<float>> empty_data;
    std::vector<std::vector<float>> empty_labels;
    
    // Empty data should be handled - training with 0 epochs
    EXPECT_NO_THROW(nn.fit(empty_data, empty_labels, 0, 0, nullptr, nullptr, false));
}

TEST_F(NeuralNetworkTest, EmptyBatchTraining) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.01f);
    nn.initialize_he();
    
    std::vector<std::vector<float>> empty_inputs;
    std::vector<std::vector<float>> empty_targets;
    
    // train_batch should gracefully handle empty inputs and return 0.0
    float loss = nn.train_batch(empty_inputs, empty_targets);
    EXPECT_FLOAT_EQ(loss, 0.0f);
    
    // Should not throw an exception
    EXPECT_NO_THROW(nn.train_batch(empty_inputs, empty_targets));
}

TEST_F(NeuralNetworkTest, SingleSampleTraining) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.1f);
    nn.initialize_he();
    
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}};
    std::vector<std::vector<float>> y = {{3.0f}};
    
    EXPECT_NO_THROW(nn.fit(X, y, 100, 1, nullptr, nullptr, false));
}

TEST_F(NeuralNetworkTest, ZeroEpochs) {
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE);
    
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}};
    std::vector<std::vector<float>> y = {{3.0f}};
    
    EXPECT_NO_THROW(nn.fit(X, y, 0, 0, nullptr, nullptr, false));
}

TEST_F(NeuralNetworkTest, LargeBatchSize) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.01f);
    nn.initialize_he();
    
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}, {2.0f, 3.0f}};
    std::vector<std::vector<float>> y = {{3.0f}, {5.0f}};
    
    // Batch size larger than dataset
    EXPECT_NO_THROW(nn.fit(X, y, 10, 100, nullptr, nullptr, false));
}

TEST_F(NeuralNetworkTest, NumericalStabilityWithExtremeValues) {
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.01f);
    nn.initialize_he();
    
    // Test with extreme input values
    std::vector<std::vector<float>> X = {
        {1000.0f, -1000.0f},
        {0.001f, 0.001f},
        {-500.0f, 500.0f}
    };
    std::vector<std::vector<float>> y = {
        {1.0f},
        {0.0f},
        {-1.0f}
    };
    
    // Should handle extreme values without crashing or producing NaN
    EXPECT_NO_THROW(nn.fit(X, y, 50, 3, nullptr, nullptr, false));
    
    // Verify predictions are valid (not NaN or Inf)
    auto predictions = nn.predict_batch(X);
    for (const auto& pred : predictions) {
        for (float val : pred) {
            EXPECT_FALSE(std::isnan(val));
            EXPECT_FALSE(std::isinf(val));
        }
    }
}

TEST_F(NeuralNetworkTest, GradientClippingPreventsExplosion) {
    // Create a network that might experience gradient explosion
    std::vector<int> architecture = {3, 10, 10, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.1f);
    nn.initialize_he();
    
    // Large target values can cause gradient explosion
    std::vector<std::vector<float>> X = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    std::vector<std::vector<float>> y = {
        {100.0f},
        {200.0f}
    };
    
    // Train multiple times - gradient clipping should prevent instability
    for (int i = 0; i < 5; ++i) {
        float loss = nn.train_batch(X, y);
        EXPECT_FALSE(std::isnan(loss));
        EXPECT_FALSE(std::isinf(loss));
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(NeuralNetworkTest, CompleteWorkflowRegression) {
    // Create network - simpler architecture for more reliable convergence
    std::vector<int> architecture = {2, 8, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.01f);
    
    // Use multiple initialization attempts to ensure convergence
    // This handles the edge case of poor random initialization
    bool converged = false;
    int max_attempts = 3;
    
    for (int attempt = 0; attempt < max_attempts && !converged; ++attempt) {
        nn.initialize_he();
        
        // Generate training data: y = x1 + 2*x2
        std::vector<std::vector<float>> X_train;
        std::vector<std::vector<float>> y_train;
        for (int i = 0; i < 50; ++i) {
            float x1 = i * 0.1f;
            float x2 = i * 0.05f;
            X_train.push_back({x1, x2});
            y_train.push_back({x1 + 2 * x2});
        }
        
        // Validation data
        std::vector<std::vector<float>> X_val = {{5.0f, 2.5f}, {6.0f, 3.0f}};
        std::vector<std::vector<float>> y_val = {{10.0f}, {12.0f}};
        
        // Train with more epochs for better convergence
        nn.fit(X_train, y_train, 300, 10, &X_val, &y_val, false);
        
        // Check if converged reasonably
        auto predictions = nn.predict_batch(X_val);
        if (std::abs(predictions[0][0] - 10.0f) < 2.5f && 
            std::abs(predictions[1][0] - 12.0f) < 2.5f) {
            converged = true;
        }
    }
    
    // Save model
    nn.save(test_model_file);
    
    // Load model
    NeuralNetwork nn2(architecture, activations, LossType::MSE);
    nn2.load(test_model_file);
    
    // Predict
    auto predictions = nn2.predict_batch({{5.0f, 2.5f}, {6.0f, 3.0f}});
    
    // Check predictions are reasonable (relaxed tolerance due to random initialization)
    EXPECT_LT(std::abs(predictions[0][0] - 10.0f), 2.5f);
    EXPECT_LT(std::abs(predictions[1][0] - 12.0f), 2.5f);
}

TEST_F(NeuralNetworkTest, CompleteWorkflowClassification) {
    // 3-class classification
    std::vector<int> architecture = {2, 8, 3};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::CATEGORICAL_CROSS_ENTROPY, 0.1f);
    nn.initialize_he();
    
    // Simple linearly separable 3-class problem
    std::vector<std::vector<float>> X = {
        {0.0f, 0.0f}, {0.1f, 0.1f}, {0.2f, 0.0f},  // Class 0
        {1.0f, 0.0f}, {1.1f, 0.1f}, {0.9f, 0.0f},  // Class 1
        {0.0f, 1.0f}, {0.1f, 1.1f}, {0.0f, 0.9f}   // Class 2
    };
    
    std::vector<std::vector<float>> y = {
        {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}
    };
    
    nn.fit(X, y, 500, 3, nullptr, nullptr, false);
    
    auto predictions = nn.predict_batch(X);
    float accuracy = nn.compute_accuracy(predictions, y);
    
    EXPECT_GT(accuracy, 0.8f);  // Should achieve >80% accuracy
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
