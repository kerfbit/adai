#include "Neuron.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

// Initialize static random number generator
std::mt19937 Neuron::rng(std::random_device{}());

// Neuron implementation

Neuron::Neuron(int input_size, ActivationType activation, float lr)
    : bias(0.0f), learning_rate(lr), activation_type(activation),
      last_pre_activation(0.0f), last_activation(0.0f) {
    weights.resize(input_size, 0.0f);
    last_input.resize(input_size, 0.0f);
}

Neuron::Neuron(const std::vector<float>& init_weights, float init_bias,
               ActivationType activation, float lr)
    : weights(init_weights), bias(init_bias), learning_rate(lr),
      activation_type(activation), last_pre_activation(0.0f), last_activation(0.0f) {
    last_input.resize(weights.size(), 0.0f);
}

float Neuron::forward(const std::vector<float>& inputs) {
    assert(inputs.size() == weights.size() && "Input size mismatch");
    
    // Cache inputs for backpropagation
    last_input = inputs;
    
    // Compute weighted sum: z = Σ(wᵢ × xᵢ) + b
    last_pre_activation = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        last_pre_activation += weights[i] * inputs[i];
    }
    
    // Apply activation function: a = f(z)
    last_activation = apply_activation(last_pre_activation, activation_type);
    
    return last_activation;
}

std::vector<float> Neuron::backward(float gradient) {
    // Compute activation gradient: δ = gradient × f'(z)
    float delta = gradient * activation_derivative(last_pre_activation, activation_type);
    
    // Compute gradients for inputs
    std::vector<float> input_gradients(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        input_gradients[i] = delta * weights[i];
    }
    
    // Update weights: w = w - lr × δ × x
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate * delta * last_input[i];
    }
    
    // Update bias: b = b - lr × δ
    bias -= learning_rate * delta;
    
    return input_gradients;
}

void Neuron::set_weights(const std::vector<float>& new_weights) {
    assert(new_weights.size() == weights.size() && "Weight size mismatch");
    weights = new_weights;
}

void Neuron::randomize(float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& w : weights) {
        w = dist(rng);
    }
    bias = dist(rng);
}

void Neuron::xavier_init(int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& w : weights) {
        w = dist(rng);
    }
    bias = 0.0f;
}

void Neuron::he_init(int fan_in) {
    float stddev = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (auto& w : weights) {
        w = dist(rng);
    }
    bias = 0.0f;
}

void Neuron::save(std::ofstream& file) const {
    file << "# Neuron Weights v1.0\n";
    file << "INPUT_SIZE " << weights.size() << "\n";
    file << "ACTIVATION " << static_cast<int>(activation_type) << "\n";
    file << "LEARNING_RATE " << learning_rate << "\n";
    file << "WEIGHTS\n";
    for (const auto& w : weights) {
        file << w << "\n";
    }
    file << "BIAS\n" << bias << "\n";
}

void Neuron::load(std::ifstream& file) {
    std::string line;
    int input_size = 0;
    int activation_int = 0;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        
        if (key == "INPUT_SIZE") {
            iss >> input_size;
            weights.resize(input_size);
            last_input.resize(input_size);
        } else if (key == "ACTIVATION") {
            iss >> activation_int;
            activation_type = static_cast<ActivationType>(activation_int);
        } else if (key == "LEARNING_RATE") {
            iss >> learning_rate;
        } else if (key == "WEIGHTS") {
            for (int i = 0; i < input_size; ++i) {
                std::getline(file, line);
                weights[i] = std::stof(line);
            }
        } else if (key == "BIAS") {
            std::getline(file, line);
            bias = std::stof(line);
            break;  // End of neuron data
        }
    }
}

void Neuron::set_random_seed(unsigned int seed) {
    rng.seed(seed);
}

// Activation function implementations

float Neuron::apply_activation(float x, ActivationType type) {
    switch (type) {
        case ActivationType::LINEAR:
            return x;
            
        case ActivationType::SIGMOID:
            return 1.0f / (1.0f + std::exp(-x));
            
        case ActivationType::TANH:
            return std::tanh(x);
            
        case ActivationType::RELU:
            return std::max(0.0f, x);
            
        case ActivationType::LEAKY_RELU:
            return x > 0.0f ? x : 0.01f * x;
            
        case ActivationType::GELU:
            return gelu_approximation(x);
            
        case ActivationType::SOFTPLUS:
            return std::log(1.0f + std::exp(x));
            
        default:
            return x;
    }
}

float Neuron::activation_derivative(float x, ActivationType type) {
    switch (type) {
        case ActivationType::LINEAR:
            return 1.0f;
            
        case ActivationType::SIGMOID: {
            float sig = apply_activation(x, ActivationType::SIGMOID);
            return sig * (1.0f - sig);
        }
            
        case ActivationType::TANH: {
            float t = std::tanh(x);
            return 1.0f - t * t;
        }
            
        case ActivationType::RELU:
            return x > 0.0f ? 1.0f : 0.0f;
            
        case ActivationType::LEAKY_RELU:
            return x > 0.0f ? 1.0f : 0.01f;
            
        case ActivationType::GELU:
            return gelu_derivative(x);
            
        case ActivationType::SOFTPLUS:
            return 1.0f / (1.0f + std::exp(-x));  // sigmoid(x)
            
        default:
            return 1.0f;
    }
}

float Neuron::gelu_approximation(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

float Neuron::gelu_derivative(float x) {
    // Approximate derivative of GELU
    const float sqrt_2_over_pi = 0.7978845608f;
    float x_squared = x * x;
    float x_cubed = x_squared * x;
    
    float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    float tanh_inner = std::tanh(inner);
    float sech_squared = 1.0f - tanh_inner * tanh_inner;
    
    float term1 = 0.5f * (1.0f + tanh_inner);
    float term2 = 0.5f * x * sech_squared * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x_squared);
    
    return term1 + term2;
}

// NeuronLayer implementation

NeuronLayer::NeuronLayer(int in_size, int out_size, ActivationType activation, float lr)
    : input_size(in_size), output_size(out_size) {
    
    neurons.reserve(out_size);
    for (int i = 0; i < out_size; ++i) {
        neurons.emplace_back(in_size, activation, lr);
    }
}

std::vector<float> NeuronLayer::forward(const std::vector<float>& inputs) {
    assert(inputs.size() == static_cast<size_t>(input_size) && "Input size mismatch");
    
    std::vector<float> outputs;
    outputs.reserve(neurons.size());
    
    for (auto& neuron : neurons) {
        outputs.push_back(neuron.forward(inputs));
    }
    
    return outputs;
}

std::vector<float> NeuronLayer::backward(const std::vector<float>& gradients) {
    assert(gradients.size() == neurons.size() && "Gradient size mismatch");
    
    std::vector<float> input_gradients(input_size, 0.0f);
    
    for (size_t i = 0; i < neurons.size(); ++i) {
        auto neuron_grads = neurons[i].backward(gradients[i]);
        for (size_t j = 0; j < input_gradients.size(); ++j) {
            input_gradients[j] += neuron_grads[j];
        }
    }
    
    return input_gradients;
}

void NeuronLayer::he_init() {
    for (auto& neuron : neurons) {
        neuron.he_init(input_size);
    }
}

void NeuronLayer::xavier_init(int fan_out) {
    for (auto& neuron : neurons) {
        neuron.xavier_init(input_size, fan_out);
    }
}

void NeuronLayer::set_learning_rate(float lr) {
    for (auto& neuron : neurons) {
        neuron.set_learning_rate(lr);
    }
}

const Neuron& NeuronLayer::get_neuron(int index) const {
    assert(index >= 0 && index < output_size && "Neuron index out of range");
    return neurons[index];
}

void NeuronLayer::save(std::ofstream& file) const {
    file << "# NeuronLayer v1.0\n";
    file << "INPUT_SIZE " << input_size << "\n";
    file << "OUTPUT_SIZE " << output_size << "\n";
    file << "NEURONS\n";
    
    for (const auto& neuron : neurons) {
        neuron.save(file);
    }
}

void NeuronLayer::load(std::ifstream& file) {
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        
        if (key == "INPUT_SIZE") {
            iss >> input_size;
        } else if (key == "OUTPUT_SIZE") {
            iss >> output_size;
            neurons.clear();
            neurons.reserve(output_size);
        } else if (key == "NEURONS") {
            for (int i = 0; i < output_size; ++i) {
                Neuron neuron(input_size);
                neuron.load(file);
                neurons.push_back(neuron);
            }
            break;
        }
    }
}
