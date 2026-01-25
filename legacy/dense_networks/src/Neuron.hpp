#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>

/**
 * Activation function types supported by the Neuron class
 */
enum class ActivationType {
    LINEAR,     // f(x) = x
    SIGMOID,    // f(x) = 1 / (1 + e^(-x))
    TANH,       // f(x) = tanh(x)
    RELU,       // f(x) = max(0, x)
    LEAKY_RELU, // f(x) = x if x > 0, else αx (α = 0.01)
    GELU,       // f(x) = x × Φ(x), Gaussian Error Linear Unit
    SOFTPLUS    // f(x) = ln(1 + e^x)
};

/**
 * Neuron class - represents a single computational unit in a neural network
 * 
 * Encapsulates the fundamental operations of a neuron:
 * - Weighted sum of inputs
 * - Bias addition
 * - Activation function application
 * - Backpropagation for training
 */
class Neuron {
private:
    std::vector<float> weights;       // Weight vector [w₁, w₂, ..., wₙ]
    float bias;                       // Bias term b
    float learning_rate;              // Learning rate for weight updates
    
    // Cached values for backpropagation
    std::vector<float> last_input;
    float last_pre_activation;        // z value
    float last_activation;            // a value
    
    // Activation function type
    ActivationType activation_type;
    
    // Random number generator for initialization
    static std::mt19937 rng;
    
    // Helper functions for activation and derivatives
    static float apply_activation(float x, ActivationType type);
    static float activation_derivative(float x, ActivationType type);
    static float gelu_approximation(float x);
    static float gelu_derivative(float x);
    
public:
    // Constructors
    /**
     * Create a neuron with specified input size and activation function
     * 
     * @param input_size Number of inputs to the neuron
     * @param activation Activation function type (default: RELU)
     * @param lr Learning rate for weight updates (default: 0.01)
     */
    Neuron(int input_size, ActivationType activation = ActivationType::RELU, 
           float lr = 0.01f);
    
    /**
     * Create a neuron with pre-initialized weights and bias
     * 
     * @param init_weights Initial weight values
     * @param init_bias Initial bias value
     * @param activation Activation function type (default: RELU)
     * @param lr Learning rate for weight updates (default: 0.01)
     */
    Neuron(const std::vector<float>& init_weights, float init_bias,
           ActivationType activation = ActivationType::RELU, float lr = 0.01f);
    
    // Forward pass
    /**
     * Compute neuron output for given inputs
     * 
     * Performs: z = Σ(wᵢ × xᵢ) + b, then a = f(z)
     * 
     * @param inputs Input values
     * @return Neuron activation (output)
     */
    float forward(const std::vector<float>& inputs);
    
    // Backward pass
    /**
     * Compute gradients and update weights using backpropagation
     * 
     * @param gradient Error gradient from next layer (∂L/∂a)
     * @return Gradients with respect to inputs (∂L/∂x)
     */
    std::vector<float> backward(float gradient);
    
    // Getters and setters
    const std::vector<float>& get_weights() const { return weights; }
    float get_bias() const { return bias; }
    float get_learning_rate() const { return learning_rate; }
    ActivationType get_activation_type() const { return activation_type; }
    
    void set_weights(const std::vector<float>& new_weights);
    void set_bias(float new_bias) { bias = new_bias; }
    void set_learning_rate(float lr) { learning_rate = lr; }
    
    // Utility functions
    /**
     * Randomize weights with uniform distribution
     * 
     * @param scale Scale factor for random values (default: 0.1)
     */
    void randomize(float scale = 0.1f);
    
    /**
     * Initialize weights using Xavier/Glorot initialization
     * Best for: Sigmoid, Tanh activations
     * 
     * @param fan_in Number of input units
     * @param fan_out Number of output units
     */
    void xavier_init(int fan_in, int fan_out);
    
    /**
     * Initialize weights using He initialization
     * Best for: ReLU, Leaky ReLU activations
     * 
     * @param fan_in Number of input units
     */
    void he_init(int fan_in);
    
    // Serialization
    /**
     * Save neuron weights and parameters to file
     * 
     * @param file Output file stream
     */
    void save(std::ofstream& file) const;
    
    /**
     * Load neuron weights and parameters from file
     * 
     * @param file Input file stream
     */
    void load(std::ifstream& file);
    
    // Static helper to initialize RNG seed
    static void set_random_seed(unsigned int seed);
};

/**
 * NeuronLayer class - represents a layer of neurons
 * 
 * Provides a convenient way to manage multiple neurons as a single layer
 */
class NeuronLayer {
private:
    std::vector<Neuron> neurons;
    int input_size;
    int output_size;
    
public:
    /**
     * Create a layer with specified input/output sizes
     * 
     * @param in_size Number of inputs to each neuron
     * @param out_size Number of neurons in the layer
     * @param activation Activation function type for all neurons
     * @param lr Learning rate for all neurons
     */
    NeuronLayer(int in_size, int out_size, ActivationType activation,
                float lr = 0.01f);
    
    /**
     * Forward pass through the layer
     * 
     * @param inputs Input vector
     * @return Output vector (one value per neuron)
     */
    std::vector<float> forward(const std::vector<float>& inputs);
    
    /**
     * Backward pass through the layer
     * 
     * @param gradients Gradient vector (one per output)
     * @return Input gradients
     */
    std::vector<float> backward(const std::vector<float>& gradients);
    
    /**
     * Initialize all neurons with He initialization
     */
    void he_init();
    
    /**
     * Initialize all neurons with Xavier initialization
     * 
     * @param fan_out Number of outputs from this layer
     */
    void xavier_init(int fan_out);
    
    /**
     * Set learning rate for all neurons
     * 
     * @param lr New learning rate
     */
    void set_learning_rate(float lr);
    
    /**
     * Get number of neurons in the layer
     */
    int size() const { return output_size; }
    
    /**
     * Get input size
     */
    int get_input_size() const { return input_size; }
    
    /**
     * Get output size (same as number of neurons)
     */
    int get_output_size() const { return output_size; }
    
    /**
     * Get reference to specific neuron
     * 
     * @param index Neuron index (0 to output_size-1)
     * @return Const reference to neuron
     */
    const Neuron& get_neuron(int index) const;
    
    /**
     * Save layer to file
     */
    void save(std::ofstream& file) const;
    
    /**
     * Load layer from file
     */
    void load(std::ifstream& file);
};
