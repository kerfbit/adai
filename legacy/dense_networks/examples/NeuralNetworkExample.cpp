#include "NeuralNetwork.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

void test_xor_problem() {
    std::cout << "=== XOR Problem ===" << std::endl;
    
    // Network: 2 inputs → 4 hidden → 1 output
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,      // Hidden layer
        ActivationType::SIGMOID    // Output layer
    };
    
    NeuralNetwork nn(architecture, activations, 
                     LossType::BINARY_CROSS_ENTROPY, 0.1f);
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
    
    // Print network summary
    nn.print_summary();
    
    // Train the network
    std::cout << "\nTraining..." << std::endl;
    nn.fit(X, y, 5000, 4, nullptr, nullptr, false);
    
    // Test predictions
    std::cout << "\nTest Results:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < X.size(); ++i) {
        auto pred = nn.predict(X[i]);
        std::cout << "  [" << X[i][0] << ", " << X[i][1] << "] -> " 
                  << pred[0] << " (expected: " << y[i][0] << ")" << std::endl;
    }
    
    // Print final loss
    auto& loss_history = nn.get_training_loss();
    if (!loss_history.empty()) {
        std::cout << "\nFinal training loss: " << loss_history.back() << std::endl;
    }
}

void test_linear_regression() {
    std::cout << "\n=== Linear Regression (y = 2x + 1) ===" << std::endl;
    
    // Network: 1 input → 4 hidden → 1 output
    std::vector<int> architecture = {1, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.01f);
    nn.initialize_he();
    
    // Generate training data: y = 2x + 1
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> y;
    
    for (float x = -5.0f; x <= 5.0f; x += 0.5f) {
        X.push_back({x});
        y.push_back({2.0f * x + 1.0f});
    }
    
    // Train
    std::cout << "Training..." << std::endl;
    nn.fit(X, y, 1000, 0, nullptr, nullptr, false);
    
    // Test
    std::cout << "\nTest Results:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (float x = -2.0f; x <= 2.0f; x += 1.0f) {
        auto pred = nn.predict({x});
        float expected = 2.0f * x + 1.0f;
        std::cout << "  f(" << x << ") = " << pred[0] 
                  << " (expected: " << expected << ")" << std::endl;
    }
    
    auto& loss_history = nn.get_training_loss();
    if (!loss_history.empty()) {
        std::cout << "\nFinal training loss: " << loss_history.back() << std::endl;
    }
}

void test_three_class_classification() {
    std::cout << "\n=== Three-Class Classification ===" << std::endl;
    
    // Network: 2 inputs → 8 hidden → 3 outputs
    std::vector<int> architecture = {2, 8, 3};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, 
                     LossType::CATEGORICAL_CROSS_ENTROPY, 0.05f);
    nn.initialize_he();
    
    // Synthetic 3-class data (clusters)
    std::vector<std::vector<float>> X = {
        // Class 0 (bottom-left)
        {0.1f, 0.1f}, {0.2f, 0.1f}, {0.1f, 0.2f},
        // Class 1 (top-left)
        {0.1f, 0.9f}, {0.2f, 0.9f}, {0.1f, 0.8f},
        // Class 2 (right)
        {0.9f, 0.5f}, {0.8f, 0.5f}, {0.9f, 0.6f}
    };
    
    std::vector<std::vector<float>> y = {
        // Class 0
        {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
        // Class 1
        {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        // Class 2
        {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}
    };
    
    nn.print_summary();
    
    // Train
    std::cout << "\nTraining..." << std::endl;
    nn.fit(X, y, 2000, 9, nullptr, nullptr, false);
    
    // Test
    std::cout << "\nTest Results:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    auto predictions = nn.predict_batch(X);
    float accuracy = nn.compute_accuracy(predictions, y);
    
    std::cout << "Training accuracy: " << accuracy * 100.0f << "%" << std::endl;
    
    // Show some predictions
    std::cout << "\nSample predictions:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), X.size()); ++i) {
        auto pred = nn.predict(X[i]);
        int pred_class = std::max_element(pred.begin(), pred.end()) - pred.begin();
        int true_class = std::max_element(y[i].begin(), y[i].end()) - y[i].begin();
        
        std::cout << "  Input: [" << X[i][0] << ", " << X[i][1] << "] -> "
                  << "Class " << pred_class << " (expected: " << true_class << ")"
                  << " [" << pred[0] << ", " << pred[1] << ", " << pred[2] << "]"
                  << std::endl;
    }
}

void test_save_load() {
    std::cout << "\n=== Save/Load Test ===" << std::endl;
    
    // Create and train a simple network
    std::vector<int> architecture = {2, 3, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn1(architecture, activations, LossType::MSE, 0.1f);
    nn1.initialize_xavier();
    
    // Train on simple data
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}, {2.0f, 3.0f}};
    std::vector<std::vector<float>> y = {{0.5f}, {0.7f}};
    nn1.fit(X, y, 100, 0, nullptr, nullptr, false);
    
    // Save
    nn1.save("test_network.dat");
    std::cout << "Network saved to test_network.dat" << std::endl;
    
    // Load into new network
    NeuralNetwork nn2(architecture, activations, LossType::MSE, 0.1f);
    nn2.load("test_network.dat");
    std::cout << "Network loaded from test_network.dat" << std::endl;
    
    // Compare predictions
    std::cout << "\nComparing predictions:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (const auto& input : X) {
        auto pred1 = nn1.predict(input);
        auto pred2 = nn2.predict(input);
        std::cout << "  Input: [" << input[0] << ", " << input[1] << "]" << std::endl;
        std::cout << "    Original: " << pred1[0] << std::endl;
        std::cout << "    Loaded:   " << pred2[0] << std::endl;
        std::cout << "    Difference: " << std::abs(pred1[0] - pred2[0]) << std::endl;
    }
}

void test_deep_network() {
    std::cout << "\n=== Deep Network Test ===" << std::endl;
    
    // Deep network: 4 → 8 → 6 → 4 → 2
    std::vector<int> architecture = {4, 8, 6, 4, 2};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.01f);
    nn.initialize_he();
    
    nn.print_summary();
    
    // Test forward pass
    std::vector<float> test_input = {1.0f, 0.5f, -0.3f, 0.8f};
    auto output = nn.predict(test_input);
    
    std::cout << "\nForward pass test:" << std::endl;
    std::cout << "  Input size: " << test_input.size() << std::endl;
    std::cout << "  Output size: " << output.size() << std::endl;
    std::cout << "  Output: [" << output[0] << ", " << output[1] << "]" << std::endl;
}

int main() {
    std::cout << "=== Neural Network Examples ===" << std::endl;
    std::cout << std::endl;
    
    try {
        test_xor_problem();
        test_linear_regression();
        test_three_class_classification();
        test_save_load();
        test_deep_network();
        
        std::cout << "\n=== All Tests Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
