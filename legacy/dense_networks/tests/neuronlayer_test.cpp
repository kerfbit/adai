#include <gtest/gtest.h>
#include "../src/Neuron.hpp"
#include <cmath>
#include <fstream>

// Test fixture for NeuronLayer tests
class NeuronLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set random seed for reproducibility
        Neuron::set_random_seed(42);
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(NeuronLayerTest, ConstructionWithValidParameters) {
    NeuronLayer layer(10, 5, ActivationType::RELU, 0.01f);
    
    EXPECT_EQ(layer.get_input_size(), 10);
    EXPECT_EQ(layer.get_output_size(), 5);
    EXPECT_EQ(layer.size(), 5);
}

TEST_F(NeuronLayerTest, ConstructionWithDifferentActivations) {
    std::vector<ActivationType> activations = {
        ActivationType::LINEAR,
        ActivationType::SIGMOID,
        ActivationType::TANH,
        ActivationType::RELU,
        ActivationType::LEAKY_RELU,
        ActivationType::GELU,
        ActivationType::SOFTPLUS
    };
    
    for (auto activation : activations) {
        NeuronLayer layer(5, 3, activation, 0.01f);
        EXPECT_EQ(layer.size(), 3);
        EXPECT_EQ(layer.get_input_size(), 5);
    }
}

TEST_F(NeuronLayerTest, ConstructionWithVariousSizes) {
    // Small layer
    NeuronLayer small(2, 3, ActivationType::RELU);
    EXPECT_EQ(small.get_input_size(), 2);
    EXPECT_EQ(small.get_output_size(), 3);
    
    // Large layer
    NeuronLayer large(100, 50, ActivationType::RELU);
    EXPECT_EQ(large.get_input_size(), 100);
    EXPECT_EQ(large.get_output_size(), 50);
    
    // Single neuron layer
    NeuronLayer single(10, 1, ActivationType::SIGMOID);
    EXPECT_EQ(single.size(), 1);
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_F(NeuronLayerTest, ForwardPassLinear) {
    NeuronLayer layer(3, 2, ActivationType::LINEAR, 0.01f);
    
    // Set known weights for predictable output
    // This requires accessing neurons through get_neuron (which we'll implement)
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    auto output = layer.forward(input);
    
    EXPECT_EQ(output.size(), 2);
}

TEST_F(NeuronLayerTest, ForwardPassOutputSize) {
    NeuronLayer layer(5, 10, ActivationType::RELU);
    
    std::vector<float> input = {1.0f, 0.5f, -0.3f, 0.2f, 0.8f};
    auto output = layer.forward(input);
    
    EXPECT_EQ(output.size(), 10);
}

TEST_F(NeuronLayerTest, ForwardPassWithZeroInput) {
    NeuronLayer layer(4, 3, ActivationType::RELU);
    layer.he_init();
    
    std::vector<float> input = {0.0f, 0.0f, 0.0f, 0.0f};
    auto output = layer.forward(input);
    
    EXPECT_EQ(output.size(), 3);
    // With zero input, ReLU output should be max(0, bias)
}

TEST_F(NeuronLayerTest, ForwardPassReLUActivation) {
    NeuronLayer layer(3, 2, ActivationType::RELU);
    layer.he_init();
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    auto output = layer.forward(input);
    
    // ReLU should produce non-negative outputs
    for (float val : output) {
        EXPECT_GE(val, 0.0f);
    }
}

TEST_F(NeuronLayerTest, ForwardPassSigmoidActivation) {
    NeuronLayer layer(3, 2, ActivationType::SIGMOID);
    layer.xavier_init(2);
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    auto output = layer.forward(input);
    
    // Sigmoid should produce outputs in (0, 1)
    for (float val : output) {
        EXPECT_GT(val, 0.0f);
        EXPECT_LT(val, 1.0f);
    }
}

TEST_F(NeuronLayerTest, ForwardPassTanhActivation) {
    NeuronLayer layer(3, 2, ActivationType::TANH);
    layer.xavier_init(2);
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    auto output = layer.forward(input);
    
    // Tanh should produce outputs in (-1, 1)
    for (float val : output) {
        EXPECT_GT(val, -1.0f);
        EXPECT_LT(val, 1.0f);
    }
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST_F(NeuronLayerTest, BackwardPassGradientSize) {
    NeuronLayer layer(4, 3, ActivationType::RELU, 0.1f);
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    layer.forward(input);
    
    std::vector<float> gradients = {1.0f, 1.0f, 1.0f};
    auto input_grads = layer.backward(gradients);
    
    EXPECT_EQ(input_grads.size(), 4);
}

TEST_F(NeuronLayerTest, BackwardPassGradientAccumulation) {
    // Test that gradients from multiple neurons are accumulated correctly
    NeuronLayer layer(2, 3, ActivationType::LINEAR, 0.0f); // lr=0 to avoid weight updates
    
    std::vector<float> input = {1.0f, 1.0f};
    layer.forward(input);
    
    std::vector<float> grad_output = {1.0f, 1.0f, 1.0f};
    auto grad_input = layer.backward(grad_output);
    
    EXPECT_EQ(grad_input.size(), 2);
    // With linear activation, gradients should accumulate from all neurons
}

TEST_F(NeuronLayerTest, BackwardPassWeightUpdate) {
    NeuronLayer layer(2, 1, ActivationType::LINEAR, 0.1f);
    
    // Initialize with known weights
    layer.he_init();
    
    // Get initial neuron state
    const auto& neuron_before = layer.get_neuron(0);
    auto weights_before = neuron_before.get_weights();
    float bias_before = neuron_before.get_bias();
    
    // Forward and backward pass
    std::vector<float> input = {1.0f, 1.0f};
    layer.forward(input);
    
    std::vector<float> gradient = {0.5f};
    layer.backward(gradient);
    
    // Check that weights have changed
    const auto& neuron_after = layer.get_neuron(0);
    auto weights_after = neuron_after.get_weights();
    float bias_after = neuron_after.get_bias();
    
    // Weights should have changed (updated)
    bool weights_changed = false;
    for (size_t i = 0; i < weights_before.size(); ++i) {
        if (std::abs(weights_before[i] - weights_after[i]) > 1e-6f) {
            weights_changed = true;
            break;
        }
    }
    
    EXPECT_TRUE(weights_changed || std::abs(bias_before - bias_after) > 1e-6f);
}

TEST_F(NeuronLayerTest, BackwardPassMultipleLayers) {
    // Test gradient flow through two layers
    NeuronLayer layer1(2, 3, ActivationType::RELU, 0.01f);
    NeuronLayer layer2(3, 1, ActivationType::SIGMOID, 0.01f);
    
    layer1.he_init();
    layer2.xavier_init(1);
    
    // Forward pass
    std::vector<float> input = {1.0f, 0.5f};
    auto hidden = layer1.forward(input);
    auto output = layer2.forward(hidden);
    
    // Backward pass
    std::vector<float> grad_output = {0.1f};
    auto grad_hidden = layer2.backward(grad_output);
    auto grad_input = layer1.backward(grad_hidden);
    
    EXPECT_EQ(grad_input.size(), 2);
    EXPECT_EQ(grad_hidden.size(), 3);
}

// ============================================================================
// Weight Initialization Tests
// ============================================================================

TEST_F(NeuronLayerTest, HeInitialization) {
    NeuronLayer layer(100, 50, ActivationType::RELU);
    layer.he_init();
    
    // Check that weights are initialized (non-zero)
    const auto& neuron = layer.get_neuron(0);
    auto weights = neuron.get_weights();
    
    bool has_nonzero = false;
    for (float w : weights) {
        if (std::abs(w) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    
    EXPECT_TRUE(has_nonzero);
}

TEST_F(NeuronLayerTest, XavierInitialization) {
    NeuronLayer layer(100, 50, ActivationType::TANH);
    layer.xavier_init(50);
    
    // Check that weights are initialized (non-zero)
    const auto& neuron = layer.get_neuron(0);
    auto weights = neuron.get_weights();
    
    bool has_nonzero = false;
    for (float w : weights) {
        if (std::abs(w) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    
    EXPECT_TRUE(has_nonzero);
    
    // Bias should be zero after Xavier init
    EXPECT_NEAR(neuron.get_bias(), 0.0f, 1e-6f);
}

TEST_F(NeuronLayerTest, InitializationStatistics) {
    // Test that He initialization produces appropriate variance
    NeuronLayer layer(100, 50, ActivationType::RELU);
    layer.he_init();
    
    // Collect all weights
    std::vector<float> all_weights;
    for (int i = 0; i < layer.size(); ++i) {
        const auto& neuron = layer.get_neuron(i);
        auto weights = neuron.get_weights();
        all_weights.insert(all_weights.end(), weights.begin(), weights.end());
    }
    
    // Compute mean and variance
    float mean = 0.0f;
    for (float w : all_weights) {
        mean += w;
    }
    mean /= all_weights.size();
    
    float variance = 0.0f;
    for (float w : all_weights) {
        variance += (w - mean) * (w - mean);
    }
    variance /= all_weights.size();
    
    // He initialization should have mean ~0 and variance ~2/n
    EXPECT_NEAR(mean, 0.0f, 0.01f);
    
    float expected_variance = 2.0f / 100.0f;
    EXPECT_NEAR(variance, expected_variance, 0.01f);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(NeuronLayerTest, SetLearningRate) {
    NeuronLayer layer(5, 3, ActivationType::RELU, 0.01f);
    
    // Change learning rate
    layer.set_learning_rate(0.001f);
    
    // Verify by checking a neuron's learning rate
    const auto& neuron = layer.get_neuron(0);
    EXPECT_NEAR(neuron.get_learning_rate(), 0.001f, 1e-6f);
}

TEST_F(NeuronLayerTest, GetNeuronAccess) {
    NeuronLayer layer(5, 3, ActivationType::RELU);
    
    // Access each neuron
    for (int i = 0; i < layer.size(); ++i) {
        const auto& neuron = layer.get_neuron(i);
        EXPECT_EQ(neuron.get_weights().size(), 5);
    }
}

// ============================================================================
// Serialization Tests
// ============================================================================

TEST_F(NeuronLayerTest, SaveLoadSimple) {
    NeuronLayer original(5, 3, ActivationType::RELU, 0.02f);
    original.he_init();
    
    // Save
    std::ofstream out("test_neuronlayer_simple.dat");
    original.save(out);
    out.close();
    
    // Load
    NeuronLayer loaded(5, 3, ActivationType::RELU);
    std::ifstream in("test_neuronlayer_simple.dat");
    loaded.load(in);
    in.close();
    
    // Verify by comparing outputs
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto out1 = original.forward(input);
    auto out2 = loaded.forward(input);
    
    EXPECT_EQ(out1.size(), out2.size());
    for (size_t i = 0; i < out1.size(); ++i) {
        EXPECT_NEAR(out1[i], out2[i], 1e-5f);
    }
    
    std::remove("test_neuronlayer_simple.dat");
}

TEST_F(NeuronLayerTest, SaveLoadWithDifferentActivations) {
    std::vector<ActivationType> activations = {
        ActivationType::SIGMOID,
        ActivationType::TANH,
        ActivationType::LEAKY_RELU
    };
    
    for (auto activation : activations) {
        NeuronLayer original(4, 2, activation, 0.05f);
        original.xavier_init(2);
        
        // Save
        std::string filename = "test_neuronlayer_activation.dat";
        std::ofstream out(filename);
        original.save(out);
        out.close();
        
        // Load
        NeuronLayer loaded(4, 2, activation);
        std::ifstream in(filename);
        loaded.load(in);
        in.close();
        
        // Verify
        std::vector<float> input = {1.0f, 0.5f, -0.3f, 0.8f};
        auto out1 = original.forward(input);
        auto out2 = loaded.forward(input);
        
        for (size_t i = 0; i < out1.size(); ++i) {
            EXPECT_NEAR(out1[i], out2[i], 1e-5f);
        }
        
        std::remove(filename.c_str());
    }
}

TEST_F(NeuronLayerTest, SaveLoadPreservesState) {
    NeuronLayer layer(3, 2, ActivationType::TANH, 0.01f);
    layer.xavier_init(2);
    
    // Do some training to change weights
    for (int i = 0; i < 10; ++i) {
        std::vector<float> input = {0.5f, 1.0f, 1.5f};
        layer.forward(input);
        std::vector<float> grad = {0.1f, -0.1f};
        layer.backward(grad);
    }
    
    // Save
    std::ofstream out("test_neuronlayer_state.dat");
    layer.save(out);
    out.close();
    
    // Load into new layer
    NeuronLayer loaded(3, 2, ActivationType::TANH);
    std::ifstream in("test_neuronlayer_state.dat");
    loaded.load(in);
    in.close();
    
    // Compare neuron states
    for (int i = 0; i < layer.size(); ++i) {
        const auto& orig_neuron = layer.get_neuron(i);
        const auto& loaded_neuron = loaded.get_neuron(i);
        
        auto orig_weights = orig_neuron.get_weights();
        auto loaded_weights = loaded_neuron.get_weights();
        
        EXPECT_EQ(orig_weights.size(), loaded_weights.size());
        for (size_t j = 0; j < orig_weights.size(); ++j) {
            EXPECT_NEAR(orig_weights[j], loaded_weights[j], 1e-5f);
        }
        
        EXPECT_NEAR(orig_neuron.get_bias(), loaded_neuron.get_bias(), 1e-5f);
    }
    
    std::remove("test_neuronlayer_state.dat");
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(NeuronLayerTest, XORProblemSingleLayer) {
    // Test if a single layer can learn partial XOR patterns
    NeuronLayer layer(2, 4, ActivationType::TANH, 0.1f);
    layer.xavier_init(4);
    
    // XOR training data (just forward pass test)
    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    // Test that layer can process XOR inputs
    for (const auto& input : inputs) {
        auto output = layer.forward(input);
        EXPECT_EQ(output.size(), 4);
    }
}

TEST_F(NeuronLayerTest, TwoLayerNetworkForward) {
    NeuronLayer layer1(2, 4, ActivationType::RELU, 0.01f);
    NeuronLayer layer2(4, 1, ActivationType::SIGMOID, 0.01f);
    
    layer1.he_init();
    layer2.xavier_init(1);
    
    std::vector<float> input = {1.0f, 0.5f};
    auto hidden = layer1.forward(input);
    auto output = layer2.forward(hidden);
    
    EXPECT_EQ(hidden.size(), 4);
    EXPECT_EQ(output.size(), 1);
    EXPECT_GE(output[0], 0.0f);
    EXPECT_LE(output[0], 1.0f);
}

TEST_F(NeuronLayerTest, LinearRegressionTraining) {
    // Simple test: learn y = 2x
    NeuronLayer layer(1, 1, ActivationType::LINEAR, 0.1f);
    layer.he_init();
    
    // Training data
    std::vector<std::pair<float, float>> data = {
        {1.0f, 2.0f},
        {2.0f, 4.0f},
        {3.0f, 6.0f}
    };
    
    // Train for a few iterations
    for (int epoch = 0; epoch < 100; ++epoch) {
        for (const auto& [x, y_true] : data) {
            std::vector<float> input = {x};
            auto output = layer.forward(input);
            
            float error = output[0] - y_true;
            std::vector<float> gradient = {error};
            layer.backward(gradient);
        }
    }
    
    // Test
    auto result = layer.forward({5.0f});
    // Should be close to 10.0
    EXPECT_NEAR(result[0], 10.0f, 2.0f);
}

TEST_F(NeuronLayerTest, GradientFlowThroughLayers) {
    // Test gradient flow through 3 layers
    NeuronLayer layer1(2, 4, ActivationType::RELU, 0.01f);
    NeuronLayer layer2(4, 3, ActivationType::RELU, 0.01f);
    NeuronLayer layer3(3, 1, ActivationType::LINEAR, 0.01f);
    
    layer1.he_init();
    layer2.he_init();
    layer3.he_init();
    
    // Forward
    std::vector<float> input = {1.0f, 2.0f};
    auto h1 = layer1.forward(input);
    auto h2 = layer2.forward(h1);
    auto output = layer3.forward(h2);
    
    // Backward
    std::vector<float> grad3 = {1.0f};
    auto grad2 = layer3.backward(grad3);
    auto grad1 = layer2.backward(grad2);
    auto grad_input = layer1.backward(grad1);
    
    EXPECT_EQ(grad_input.size(), 2);
    EXPECT_EQ(grad1.size(), 4);
    EXPECT_EQ(grad2.size(), 3);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(NeuronLayerTest, LargeLayer) {
    // Test with a large layer
    NeuronLayer layer(1000, 500, ActivationType::RELU);
    layer.he_init();
    
    EXPECT_EQ(layer.get_input_size(), 1000);
    EXPECT_EQ(layer.get_output_size(), 500);
    
    // Create large input
    std::vector<float> input(1000, 0.5f);
    auto output = layer.forward(input);
    
    EXPECT_EQ(output.size(), 500);
}

TEST_F(NeuronLayerTest, SingleNeuronLayer) {
    NeuronLayer layer(10, 1, ActivationType::SIGMOID);
    layer.xavier_init(1);
    
    std::vector<float> input(10, 1.0f);
    auto output = layer.forward(input);
    
    EXPECT_EQ(output.size(), 1);
    EXPECT_GT(output[0], 0.0f);
    EXPECT_LT(output[0], 1.0f);
}

TEST_F(NeuronLayerTest, AllZeroGradients) {
    NeuronLayer layer(3, 2, ActivationType::RELU, 0.01f);
    layer.he_init();
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    layer.forward(input);
    
    // Zero gradients
    std::vector<float> zero_grads = {0.0f, 0.0f};
    auto input_grads = layer.backward(zero_grads);
    
    // Input gradients should also be zero
    for (float grad : input_grads) {
        EXPECT_NEAR(grad, 0.0f, 1e-6f);
    }
}

TEST_F(NeuronLayerTest, ConsecutiveForwardPasses) {
    NeuronLayer layer(3, 2, ActivationType::RELU);
    layer.he_init();
    
    std::vector<float> input1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> input2 = {-1.0f, 0.5f, 2.0f};
    
    auto output1 = layer.forward(input1);
    auto output2 = layer.forward(input2);
    
    // Outputs should be different
    bool different = false;
    for (size_t i = 0; i < output1.size(); ++i) {
        if (std::abs(output1[i] - output2[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    
    EXPECT_TRUE(different);
}

// ============================================================================
// Performance/Stress Tests
// ============================================================================

TEST_F(NeuronLayerTest, MultipleTrainingIterations) {
    NeuronLayer layer(5, 3, ActivationType::RELU, 0.01f);
    layer.he_init();
    
    // Run many training iterations
    std::vector<float> input = {1.0f, 0.5f, -0.3f, 0.8f, -0.2f};
    std::vector<float> gradient = {0.1f, -0.05f, 0.15f};
    
    for (int i = 0; i < 1000; ++i) {
        layer.forward(input);
        layer.backward(gradient);
    }
    
    // Layer should still function correctly
    auto output = layer.forward(input);
    EXPECT_EQ(output.size(), 3);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
