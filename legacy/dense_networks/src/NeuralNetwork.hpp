#pragma once

#include "Neuron.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>

/**
 * Loss function types supported by the NeuralNetwork class
 */
enum class LossType {
    MSE,                      // Mean Squared Error
    MAE,                      // Mean Absolute Error
    BINARY_CROSS_ENTROPY,     // Binary Classification
    CATEGORICAL_CROSS_ENTROPY,// Multi-class Classification
    HUBER                     // Robust regression loss
};

/**
 * NeuralNetwork class - provides a complete feed-forward neural network
 * 
 * Supports multi-layer architectures, various activation functions, flexible
 * training configurations, and comprehensive model persistence.
 */
class NeuralNetwork {
private:
    std::vector<NeuronLayer> layers;
    std::vector<int> layer_sizes;
    LossType loss_function;
    
    // Training history
    std::vector<float> training_loss_history;
    std::vector<float> validation_loss_history;
    std::vector<float> training_accuracy_history;
    std::vector<float> validation_accuracy_history;
    
    // Helper functions
    float compute_loss(const std::vector<float>& predictions,
                      const std::vector<float>& targets);
    std::vector<float> compute_loss_gradient(const std::vector<float>& predictions,
                                             const std::vector<float>& targets);
    
    // Gradient clipping and numerical stability
    void clip_gradients(std::vector<float>& gradients, float max_norm = 5.0f);
    bool is_valid_output(const std::vector<float>& output);
    bool check_network_health();
    
public:
    // Constructors
    /**
     * Create a neural network with specified architecture
     * 
     * @param layer_sizes Architecture specification [input, hidden1, ..., output]
     * @param activations Activation function for each layer (size = num_layers)
     * @param loss Loss function type
     * @param learning_rate Learning rate for all layers
     */
    NeuralNetwork(const std::vector<int>& layer_sizes,
                  const std::vector<ActivationType>& activations,
                  LossType loss,
                  float learning_rate = 0.01f);
    
    // Forward pass
    /**
     * Predict output for given input
     * 
     * @param input Input vector
     * @return Prediction vector
     */
    std::vector<float> predict(const std::vector<float>& input);
    
    /**
     * Predict outputs for batch of inputs
     * 
     * @param inputs Batch of input vectors
     * @return Batch of prediction vectors
     */
    std::vector<std::vector<float>> predict_batch(
        const std::vector<std::vector<float>>& inputs);
    
    // Training
    /**
     * Train the network on a single sample
     * 
     * @param input Input vector
     * @param target Target output vector
     * @return Loss value
     */
    float train_sample(const std::vector<float>& input,
                      const std::vector<float>& target);
    
    /**
     * Train the network on a batch of samples
     * 
     * @param inputs Batch of input vectors
     * @param targets Batch of target vectors
     * @return Average loss
     */
    float train_batch(const std::vector<std::vector<float>>& inputs,
                     const std::vector<std::vector<float>>& targets);
    
    /**
     * Train the network for multiple epochs
     * 
     * @param train_data Training input data
     * @param train_labels Training target labels
     * @param epochs Number of training epochs
     * @param batch_size Mini-batch size (0 = full batch)
     * @param val_data Validation data (nullptr = no validation)
     * @param val_labels Validation labels (nullptr = no validation)
     * @param verbose Print training progress
     */
    void fit(const std::vector<std::vector<float>>& train_data,
            const std::vector<std::vector<float>>& train_labels,
            int epochs,
            int batch_size = 0,
            const std::vector<std::vector<float>>* val_data = nullptr,
            const std::vector<std::vector<float>>* val_labels = nullptr,
            bool verbose = true);
    
    // Evaluation
    /**
     * Evaluate network on test data
     * 
     * @param test_data Test input data
     * @param test_labels Test target labels
     * @return Average loss
     */
    float evaluate(const std::vector<std::vector<float>>& test_data,
                  const std::vector<std::vector<float>>& test_labels);
    
    /**
     * Compute classification accuracy
     * 
     * @param predictions Predicted outputs
     * @param targets Target outputs
     * @return Accuracy (0.0 to 1.0)
     */
    float compute_accuracy(const std::vector<std::vector<float>>& predictions,
                          const std::vector<std::vector<float>>& targets);
    
    // Network configuration
    /**
     * Set learning rate for all layers
     * 
     * @param lr New learning rate
     */
    void set_learning_rate(float lr);
    
    /**
     * Get network architecture summary
     */
    void print_summary() const;
    
    /**
     * Get training history
     */
    const std::vector<float>& get_training_loss() const { 
        return training_loss_history; 
    }
    const std::vector<float>& get_validation_loss() const { 
        return validation_loss_history; 
    }
    const std::vector<float>& get_training_accuracy() const {
        return training_accuracy_history;
    }
    const std::vector<float>& get_validation_accuracy() const {
        return validation_accuracy_history;
    }
    
    // Weight initialization
    /**
     * Initialize all weights using He initialization
     */
    void initialize_he();
    
    /**
     * Initialize all weights using Xavier initialization
     */
    void initialize_xavier();
    
    // Serialization
    /**
     * Save network to file
     * 
     * @param filename Output file path
     */
    void save(const std::string& filename) const;
    
    /**
     * Load network from file
     * 
     * @param filename Input file path
     */
    void load(const std::string& filename);
    
    // Layer access
    /**
     * Get number of layers in the network
     */
    int get_num_layers() const { return layers.size(); }
    
    /**
     * Get reference to specific layer
     * 
     * @param index Layer index
     * @return Const reference to layer
     */
    const NeuronLayer& get_layer(int index) const { 
        return layers[index]; 
    }
    
    /**
     * Get layer sizes
     */
    const std::vector<int>& get_layer_sizes() const {
        return layer_sizes;
    }
};
