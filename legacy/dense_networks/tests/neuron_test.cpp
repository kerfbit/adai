#include <../gtest/gtest.h>
#include "../src/Neuron.hpp"
#include <cmath>
#include <vector>

// Test fixture for Neuron tests
class NeuronTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set a fixed seed for reproducible tests
        Neuron::set_random_seed(42);
    }
};

// Test fixture for NeuronLayer tests
class NeuronLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        Neuron::set_random_seed(42);
    }
};

// ============================================================================
// Neuron Construction Tests
// ============================================================================

TEST_F(NeuronTest, ConstructorWithInputSize) {
    Neuron neuron(5, ActivationType::RELU, 0.01f);
    
    EXPECT_EQ(neuron.get_weights().size(), 5);
    EXPECT_FLOAT_EQ(neuron.get_bias(), 0.0f);
    EXPECT_FLOAT_EQ(neuron.get_learning_rate(), 0.01f);
    EXPECT_EQ(neuron.get_activation_type(), ActivationType::RELU);
}

TEST_F(NeuronTest, ConstructorWithWeights) {
    std::vector<float> weights = {1.0f, 2.0f, 3.0f};
    float bias = 0.5f;
    
    Neuron neuron(weights, bias, ActivationType::SIGMOID, 0.02f);
    
    EXPECT_EQ(neuron.get_weights(), weights);
    EXPECT_FLOAT_EQ(neuron.get_bias(), bias);
    EXPECT_FLOAT_EQ(neuron.get_learning_rate(), 0.02f);
    EXPECT_EQ(neuron.get_activation_type(), ActivationType::SIGMOID);
}

// ============================================================================
// Activation Function Tests
// ============================================================================

TEST_F(NeuronTest, LinearActivation) {
    std::vector<float> weights = {1.0f};
    Neuron neuron(weights, 0.0f, ActivationType::LINEAR);
    
    std::vector<float> input = {5.0f};
    float output = neuron.forward(input);
    
    EXPECT_FLOAT_EQ(output, 5.0f);
}

TEST_F(NeuronTest, SigmoidActivation) {
    std::vector<float> weights = {1.0f};
    Neuron neuron(weights, 0.0f, ActivationType::SIGMOID);
    
    std::vector<float> input = {0.0f};
    float output = neuron.forward(input);
    
    EXPECT_NEAR(output, 0.5f, 1e-5f);  // sigmoid(0) = 0.5
}

TEST_F(NeuronTest, TanhActivation) {
    std::vector<float> weights = {1.0f};
    Neuron neuron(weights, 0.0f, ActivationType::TANH);
    
    std::vector<float> input = {0.0f};
    float output = neuron.forward(input);
    
    EXPECT_NEAR(output, 0.0f, 1e-5f);  // tanh(0) = 0
}

TEST_F(NeuronTest, ReLUActivation) {
    std::vector<float> weights = {1.0f};
    Neuron neuron(weights, 0.0f, ActivationType::RELU);
    
    std::vector<float> input_positive = {5.0f};
    std::vector<float> input_negative = {-5.0f};
    
    EXPECT_FLOAT_EQ(neuron.forward(input_positive), 5.0f);
    EXPECT_FLOAT_EQ(neuron.forward(input_negative), 0.0f);
}

TEST_F(NeuronTest, LeakyReLUActivation) {
    std::vector<float> weights = {1.0f};
    Neuron neuron(weights, 0.0f, ActivationType::LEAKY_RELU);
    
    std::vector<float> input_positive = {10.0f};
    std::vector<float> input_negative = {-10.0f};
    
    EXPECT_FLOAT_EQ(neuron.forward(input_positive), 10.0f);
    EXPECT_FLOAT_EQ(neuron.forward(input_negative), -0.1f);  // 0.01 * -10
}

TEST_F(NeuronTest, SoftplusActivation) {
    std::vector<float> weights = {1.0f};
    Neuron neuron(weights, 0.0f, ActivationType::SOFTPLUS);
    
    std::vector<float> input = {0.0f};
    float output = neuron.forward(input);
    
    EXPECT_NEAR(output, std::log(2.0f), 1e-5f);  // softplus(0) = ln(2)
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_F(NeuronTest, ForwardPassBasic) {
    std::vector<float> weights = {1.0f, 2.0f, 3.0f};
    float bias = 1.0f;
    
    Neuron neuron(weights, bias, ActivationType::LINEAR);
    
    std::vector<float> input = {1.0f, 1.0f, 1.0f};
    float output = neuron.forward(input);
    
    // Expected: 1*1 + 2*1 + 3*1 + 1 = 7
    EXPECT_FLOAT_EQ(output, 7.0f);
}

TEST_F(NeuronTest, ForwardPassWithDifferentInputs) {
    std::vector<float> weights = {0.5f, -0.5f};
    float bias = 0.0f;
    
    Neuron neuron(weights, bias, ActivationType::LINEAR);
    
    std::vector<float> input = {2.0f, 4.0f};
    float output = neuron.forward(input);
    
    // Expected: 0.5*2 + (-0.5)*4 = 1 - 2 = -1
    EXPECT_FLOAT_EQ(output, -1.0f);
}

TEST_F(NeuronTest, ForwardPassMultipleCalls) {
    std::vector<float> weights = {1.0f};
    Neuron neuron(weights, 0.0f, ActivationType::LINEAR);
    
    std::vector<float> input1 = {3.0f};
    std::vector<float> input2 = {5.0f};
    
    float output1 = neuron.forward(input1);
    float output2 = neuron.forward(input2);
    
    EXPECT_FLOAT_EQ(output1, 3.0f);
    EXPECT_FLOAT_EQ(output2, 5.0f);
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST_F(NeuronTest, BackwardPassLinear) {
    std::vector<float> weights = {1.0f, 1.0f};
    float bias = 0.0f;
    
    Neuron neuron(weights, bias, ActivationType::LINEAR, 0.1f);
    
    std::vector<float> input = {1.0f, 1.0f};
    float output = neuron.forward(input);  // output = 2
    
    // Gradient from loss (e.g., MSE with target=3: gradient = output - target)
    float gradient = -1.0f;  // (2 - 3) = -1
    
    std::vector<float> input_grads = neuron.backward(gradient);
    
    // Input gradients should be delta * weights = -1 * [1, 1] = [-1, -1]
    EXPECT_FLOAT_EQ(input_grads[0], -1.0f);
    EXPECT_FLOAT_EQ(input_grads[1], -1.0f);
    
    // Weights should be updated: w = w - lr * delta * input
    // delta = gradient * f'(z) = -1 * 1 = -1 (for linear)
    // w_new = 1.0 - 0.1 * (-1) * 1.0 = 1.0 + 0.1 = 1.1
    auto new_weights = neuron.get_weights();
    EXPECT_NEAR(new_weights[0], 1.1f, 1e-5f);
    EXPECT_NEAR(new_weights[1], 1.1f, 1e-5f);
}

TEST_F(NeuronTest, BackwardPassReLU) {
    std::vector<float> weights = {2.0f};
    Neuron neuron(weights, 0.0f, ActivationType::RELU, 0.1f);
    
    // Test with positive input (ReLU active)
    std::vector<float> input_pos = {1.0f};
    neuron.forward(input_pos);
    auto grads_pos = neuron.backward(1.0f);
    
    // ReLU derivative = 1 for positive inputs
    EXPECT_FLOAT_EQ(grads_pos[0], 2.0f);  // 1.0 * 2.0
    
    // Test with negative input (ReLU inactive)
    std::vector<float> input_neg = {-1.0f};
    neuron.forward(input_neg);
    auto grads_neg = neuron.backward(1.0f);
    
    // ReLU derivative = 0 for negative inputs
    EXPECT_FLOAT_EQ(grads_neg[0], 0.0f);
}

TEST_F(NeuronTest, BackwardPassSigmoid) {
    std::vector<float> weights = {1.0f};
    Neuron neuron(weights, 0.0f, ActivationType::SIGMOID, 0.1f);
    
    std::vector<float> input = {0.0f};
    float output = neuron.forward(input);  // sigmoid(0) = 0.5
    
    auto input_grads = neuron.backward(1.0f);
    
    // Sigmoid derivative at 0: f'(0) = 0.5 * (1 - 0.5) = 0.25
    // Input gradient = delta * weight = (1.0 * 0.25) * 1.0 = 0.25
    EXPECT_NEAR(input_grads[0], 0.25f, 1e-5f);
}

// ============================================================================
// Weight Initialization Tests
// ============================================================================

TEST_F(NeuronTest, RandomizeWeights) {
    Neuron neuron(10, ActivationType::RELU);
    neuron.randomize(0.1f);
    
    auto weights = neuron.get_weights();
    
    // Check all weights are within [-0.1, 0.1]
    for (float w : weights) {
        EXPECT_GE(w, -0.1f);
        EXPECT_LE(w, 0.1f);
    }
}

TEST_F(NeuronTest, XavierInitialization) {
    Neuron neuron(100, ActivationType::TANH);
    neuron.xavier_init(100, 50);
    
    auto weights = neuron.get_weights();
    
    // Xavier limit = sqrt(6 / (100 + 50)) = sqrt(0.04) ≈ 0.2
    float expected_limit = std::sqrt(6.0f / 150.0f);
    
    for (float w : weights) {
        EXPECT_GE(w, -expected_limit);
        EXPECT_LE(w, expected_limit);
    }
    
    EXPECT_FLOAT_EQ(neuron.get_bias(), 0.0f);
}

TEST_F(NeuronTest, HeInitialization) {
    Neuron neuron(64, ActivationType::RELU);
    neuron.he_init(64);
    
    auto weights = neuron.get_weights();
    
    // He initialization uses normal distribution with stddev = sqrt(2/fan_in)
    // Just check weights are in a reasonable range
    float expected_stddev = std::sqrt(2.0f / 64.0f);
    
    // Most weights should be within 3 standard deviations
    for (float w : weights) {
        EXPECT_GE(w, -3.0f * expected_stddev);
        EXPECT_LE(w, 3.0f * expected_stddev);
    }
    
    EXPECT_FLOAT_EQ(neuron.get_bias(), 0.0f);
}

// ============================================================================
// Getters and Setters Tests
// ============================================================================

TEST_F(NeuronTest, SetWeights) {
    Neuron neuron(3, ActivationType::RELU);
    
    std::vector<float> new_weights = {1.5f, 2.5f, 3.5f};
    neuron.set_weights(new_weights);
    
    EXPECT_EQ(neuron.get_weights(), new_weights);
}

TEST_F(NeuronTest, SetBias) {
    Neuron neuron(2, ActivationType::RELU);
    
    neuron.set_bias(2.5f);
    
    EXPECT_FLOAT_EQ(neuron.get_bias(), 2.5f);
}

TEST_F(NeuronTest, SetLearningRate) {
    Neuron neuron(2, ActivationType::RELU);
    
    neuron.set_learning_rate(0.001f);
    
    EXPECT_FLOAT_EQ(neuron.get_learning_rate(), 0.001f);
}

// ============================================================================
// Save/Load Tests
// ============================================================================

TEST_F(NeuronTest, SaveAndLoad) {
    std::vector<float> weights = {1.5f, 2.5f, 3.5f};
    float bias = 0.75f;
    
    Neuron original(weights, bias, ActivationType::GELU, 0.02f);
    
    // Save to file
    std::ofstream out_file("test_neuron.dat");
    original.save(out_file);
    out_file.close();
    
    // Load into new neuron
    Neuron loaded(3);
    std::ifstream in_file("test_neuron.dat");
    loaded.load(in_file);
    in_file.close();
    
    // Verify loaded neuron matches original
    EXPECT_EQ(loaded.get_weights(), original.get_weights());
    EXPECT_FLOAT_EQ(loaded.get_bias(), original.get_bias());
    EXPECT_FLOAT_EQ(loaded.get_learning_rate(), original.get_learning_rate());
    EXPECT_EQ(loaded.get_activation_type(), original.get_activation_type());
    
    // Clean up
    std::remove("test_neuron.dat");
}

// ============================================================================
// NeuronLayer Tests
// ============================================================================

TEST_F(NeuronLayerTest, LayerConstruction) {
    NeuronLayer layer(5, 3, ActivationType::RELU, 0.01f);
    
    EXPECT_EQ(layer.size(), 3);
    EXPECT_EQ(layer.get_input_size(), 5);
}

TEST_F(NeuronLayerTest, LayerForwardPass) {
    NeuronLayer layer(2, 3, ActivationType::LINEAR, 0.01f);
    
    // Initialize with known values for testing
    layer.xavier_init(2);
    
    std::vector<float> input = {1.0f, 2.0f};
    std::vector<float> output = layer.forward(input);
    
    EXPECT_EQ(output.size(), 3);
}

TEST_F(NeuronLayerTest, LayerBackwardPass) {
    NeuronLayer layer(2, 2, ActivationType::LINEAR, 0.1f);
    
    std::vector<float> input = {1.0f, 1.0f};
    auto output = layer.forward(input);
    
    std::vector<float> gradients = {1.0f, 1.0f};
    auto input_grads = layer.backward(gradients);
    
    EXPECT_EQ(input_grads.size(), 2);
}

TEST_F(NeuronLayerTest, LayerHeInitialization) {
    NeuronLayer layer(10, 5, ActivationType::RELU);
    
    layer.he_init();
    
    // Just verify it doesn't crash and produces reasonable values
    std::vector<float> input(10, 1.0f);
    auto output = layer.forward(input);
    
    EXPECT_EQ(output.size(), 5);
}

TEST_F(NeuronLayerTest, LayerXavierInitialization) {
    NeuronLayer layer(8, 4, ActivationType::TANH);
    
    layer.xavier_init(4);
    
    std::vector<float> input(8, 0.5f);
    auto output = layer.forward(input);
    
    EXPECT_EQ(output.size(), 4);
}

TEST_F(NeuronLayerTest, LayerSetLearningRate) {
    NeuronLayer layer(3, 2, ActivationType::RELU, 0.01f);
    
    layer.set_learning_rate(0.001f);
    
    // Verify by training and checking weight updates
    std::vector<float> input = {1.0f, 1.0f, 1.0f};
    layer.forward(input);
    
    std::vector<float> gradients = {0.5f, 0.5f};
    layer.backward(gradients);
    
    // With smaller learning rate, updates should be smaller
}

TEST_F(NeuronLayerTest, LayerSaveAndLoad) {
    NeuronLayer original(3, 2, ActivationType::RELU, 0.02f);
    original.xavier_init(2);
    
    // Save to file
    std::ofstream out_file("test_layer.dat");
    original.save(out_file);
    out_file.close();
    
    // Load into new layer
    NeuronLayer loaded(3, 2, ActivationType::RELU);
    std::ifstream in_file("test_layer.dat");
    loaded.load(in_file);
    in_file.close();
    
    // Test that forward pass produces same results
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    auto output_original = original.forward(input);
    auto output_loaded = loaded.forward(input);
    
    EXPECT_EQ(output_original.size(), output_loaded.size());
    for (size_t i = 0; i < output_original.size(); ++i) {
        EXPECT_NEAR(output_original[i], output_loaded[i], 1e-5f);
    }
    
    // Clean up
    std::remove("test_layer.dat");
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(NeuronLayerTest, SimpleXORNetwork) {
    // Create a simple 2-layer network for XOR
    NeuronLayer hidden(2, 4, ActivationType::TANH, 0.5f);
    NeuronLayer output(4, 1, ActivationType::SIGMOID, 0.5f);
    
    hidden.xavier_init(4);
    output.xavier_init(1);
    
    // XOR training data
    std::vector<std::vector<float>> X = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    std::vector<float> y = {0.0f, 1.0f, 1.0f, 0.0f};
    
    // Train for a few iterations
    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            // Forward pass
            auto h = hidden.forward(X[i]);
            auto pred = output.forward(h);
            
            // Backward pass
            std::vector<float> output_grad = {pred[0] - y[i]};
            auto hidden_grad = output.backward(output_grad);
            hidden.backward(hidden_grad);
        }
    }
    
    // Test predictions
    for (size_t i = 0; i < X.size(); ++i) {
        auto h = hidden.forward(X[i]);
        auto pred = output.forward(h);
        
        // Check if prediction is close to target
        float error = std::abs(pred[0] - y[i]);
        EXPECT_LT(error, 0.2f);  // Allow 20% error after 1000 epochs
    }
}

TEST_F(NeuronTest, LinearRegressionLearning) {
    // Test that a single neuron can learn a simple linear function: y = 2x + 1
    Neuron neuron(1, ActivationType::LINEAR, 0.01f);
    neuron.randomize(0.1f);
    
    // Training data
    std::vector<float> X = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};  // y = 2x + 1
    
    // Train for many iterations
    for (int epoch = 0; epoch < 10000; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<float> input = {X[i]};
            float pred = neuron.forward(input);
            
            // MSE gradient
            float gradient = pred - y[i];
            neuron.backward(gradient);
        }
    }
    
    // Test predictions
    std::vector<float> test_input = {5.0f};
    float prediction = neuron.forward(test_input);
    
    // Expected: 2*5 + 1 = 11
    EXPECT_NEAR(prediction, 11.0f, 0.5f);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(NeuronTest, ZeroInputSize) {
    Neuron neuron(0, ActivationType::RELU);
    
    EXPECT_EQ(neuron.get_weights().size(), 0);
}

TEST_F(NeuronTest, SingleNeuronSingleInput) {
    Neuron neuron(1, ActivationType::LINEAR);
    neuron.set_weights({2.0f});
    neuron.set_bias(1.0f);
    
    std::vector<float> input = {3.0f};
    float output = neuron.forward(input);
    
    EXPECT_FLOAT_EQ(output, 7.0f);  // 2*3 + 1 = 7
}

TEST_F(NeuronTest, VeryLargeLearningRate) {
    Neuron neuron(2, ActivationType::LINEAR, 10.0f);
    neuron.set_weights({1.0f, 1.0f});
    
    std::vector<float> input = {1.0f, 1.0f};
    neuron.forward(input);
    neuron.backward(1.0f);
    
    // Weights should change significantly
    auto weights = neuron.get_weights();
    EXPECT_NE(weights[0], 1.0f);
}

TEST_F(NeuronTest, VerySmallLearningRate) {
    Neuron neuron(2, ActivationType::LINEAR, 1e-10f);
    neuron.set_weights({1.0f, 1.0f});
    
    std::vector<float> input = {1.0f, 1.0f};
    neuron.forward(input);
    neuron.backward(1.0f);
    
    // Weights should barely change
    auto weights = neuron.get_weights();
    EXPECT_NEAR(weights[0], 1.0f, 1e-5f);
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
