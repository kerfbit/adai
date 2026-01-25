#include "Neuron.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== NeuronLayer Example ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Example 1: Single layer forward pass
    std::cout << "\n1. Single Layer Forward Pass" << std::endl;
    NeuronLayer layer1(4, 8, ActivationType::RELU, 0.01f);
    layer1.he_init();
    
    std::vector<float> input1 = {1.0f, 0.5f, -0.3f, 0.2f};
    std::vector<float> output1 = layer1.forward(input1);
    
    std::cout << "   Input size: " << layer1.get_input_size() << std::endl;
    std::cout << "   Output size: " << layer1.get_output_size() << std::endl;
    std::cout << "   Number of neurons: " << layer1.size() << std::endl;
    std::cout << "   Outputs: [";
    for (size_t i = 0; i < output1.size(); ++i) {
        std::cout << output1[i];
        if (i < output1.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Example 2: Two-layer network
    std::cout << "\n2. Two-Layer Network" << std::endl;
    NeuronLayer hidden(3, 5, ActivationType::TANH, 0.05f);
    NeuronLayer output(5, 2, ActivationType::SIGMOID, 0.05f);
    
    hidden.xavier_init(5);
    output.xavier_init(2);
    
    std::vector<float> input2 = {1.0f, 2.0f, 3.0f};
    auto hidden_output = hidden.forward(input2);
    auto final_output = output.forward(hidden_output);
    
    std::cout << "   Input: [" << input2[0] << ", " << input2[1] << ", " << input2[2] << "]" << std::endl;
    std::cout << "   Hidden layer output: [";
    for (size_t i = 0; i < hidden_output.size(); ++i) {
        std::cout << hidden_output[i];
        if (i < hidden_output.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "   Final output: [" << final_output[0] << ", " << final_output[1] << "]" << std::endl;
    
    // Example 3: Training demonstration (XOR problem)
    std::cout << "\n3. Training XOR Problem" << std::endl;
    NeuronLayer xor_hidden(2, 4, ActivationType::TANH, 0.1f);
    NeuronLayer xor_output(4, 1, ActivationType::SIGMOID, 0.1f);
    
    xor_hidden.xavier_init(4);
    xor_output.xavier_init(1);
    
    // XOR training data
    std::vector<std::pair<std::vector<float>, float>> xor_data = {
        {{0.0f, 0.0f}, 0.0f},
        {{0.0f, 1.0f}, 1.0f},
        {{1.0f, 0.0f}, 1.0f},
        {{1.0f, 1.0f}, 0.0f}
    };
    
    // Train for a few epochs
    std::cout << "   Training..." << std::endl;
    for (int epoch = 0; epoch < 500; ++epoch) {
        float total_loss = 0.0f;
        
        for (const auto& [x, y_true] : xor_data) {
            // Forward pass
            auto h = xor_hidden.forward(x);
            auto y_pred = xor_output.forward(h);
            
            // Compute loss (MSE)
            float error = y_pred[0] - y_true;
            total_loss += error * error;
            
            // Backward pass
            std::vector<float> grad_output = {2.0f * error};
            auto grad_hidden = xor_output.backward(grad_output);
            xor_hidden.backward(grad_hidden);
        }
        
        if (epoch % 100 == 0) {
            std::cout << "   Epoch " << epoch << ", Loss: " << total_loss / xor_data.size() << std::endl;
        }
    }
    
    // Test the trained network
    std::cout << "\n   Testing XOR:" << std::endl;
    for (const auto& [x, y_true] : xor_data) {
        auto h = xor_hidden.forward(x);
        auto y_pred = xor_output.forward(h);
        std::cout << "   [" << x[0] << ", " << x[1] << "] -> " << y_pred[0] 
                  << " (expected: " << y_true << ")" << std::endl;
    }
    
    // Example 4: Inspect individual neurons
    std::cout << "\n4. Neuron Inspection" << std::endl;
    NeuronLayer inspect_layer(5, 3, ActivationType::RELU, 0.01f);
    inspect_layer.he_init();
    
    std::cout << "   Layer configuration:" << std::endl;
    std::cout << "   Total parameters: " 
              << (inspect_layer.get_input_size() + 1) * inspect_layer.size() << std::endl;
    
    for (int i = 0; i < inspect_layer.size(); ++i) {
        const auto& neuron = inspect_layer.get_neuron(i);
        std::cout << "   Neuron " << i << " - bias: " << neuron.get_bias() 
                  << ", weights: " << neuron.get_weights().size() << std::endl;
    }
    
    std::cout << "\n=== Example Complete ===" << std::endl;
    
    return 0;
}
