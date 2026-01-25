# Neural Network - Context Document

## Overview

The `NeuralNetwork` class provides a complete feed-forward neural network implementation built on top of the `Neuron` and `NeuronLayer` classes. It supports multi-layer architectures, various activation functions, flexible training configurations, and comprehensive model persistence. This implementation is designed for both educational purposes and practical machine learning applications.

## Purpose

The NeuralNetwork class is designed to:
- Provide a complete end-to-end neural network framework
- Support arbitrary network architectures (depth and width)
- Enable supervised learning with backpropagation
- Offer flexible loss functions and optimization strategies
- Integrate seamlessly with the existing Neuron class infrastructure
- Support batch training and mini-batch gradient descent
- Provide model evaluation and prediction capabilities

## Architecture

### Network Structure

```
Input Layer (features)
         ↓
    Hidden Layer 1 (n₁ neurons)
         ↓
    Hidden Layer 2 (n₂ neurons)
         ↓
         ...
         ↓
    Hidden Layer L (nₗ neurons)
         ↓
    Output Layer (classes/outputs)
```

### Information Flow

**Forward Propagation:**
```
x⁽⁰⁾ = input
For each layer l = 1 to L:
    z⁽ˡ⁾ = W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
    a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾)
output = a⁽ᴸ⁾
```

**Backward Propagation:**
```
δ⁽ᴸ⁾ = ∇ₐL ⊙ f'(z⁽ᴸ⁾)
For each layer l = L-1 to 1:
    δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ · δ⁽ˡ⁺¹⁾ ⊙ f'(z⁽ˡ⁾)
    ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
    ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

## Class Interface

### Main NeuralNetwork Class

```cpp
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
    
public:
    // Constructors
    /**
     * Create a neural network with specified architecture
     * 
     * @param layer_sizes Vector of layer sizes [input, hidden1, ..., hiddenN, output]
     * @param activations Activation function for each layer
     * @param loss Loss function type
     * @param learning_rate Initial learning rate
     */
    NeuralNetwork(const std::vector<int>& layer_sizes,
                  const std::vector<ActivationType>& activations,
                  LossType loss = LossType::MSE,
                  float learning_rate = 0.01f);
    
    // Forward pass
    /**
     * Predict output for given input
     * 
     * @param input Input feature vector
     * @return Predicted output vector
     */
    std::vector<float> predict(const std::vector<float>& input);
    
    /**
     * Predict outputs for batch of inputs
     * 
     * @param inputs Batch of input vectors
     * @return Batch of predicted outputs
     */
    std::vector<std::vector<float>> predict_batch(
        const std::vector<std::vector<float>>& inputs);
    
    // Training
    /**
     * Train the network on a single sample
     * 
     * @param input Input features
     * @param target Target output
     * @return Loss value
     */
    float train_sample(const std::vector<float>& input,
                      const std::vector<float>& target);
    
    /**
     * Train the network on a batch of samples
     * 
     * @param inputs Batch of input vectors
     * @param targets Batch of target vectors
     * @return Average loss over batch
     */
    float train_batch(const std::vector<std::vector<float>>& inputs,
                     const std::vector<std::vector<float>>& targets);
    
    /**
     * Train the network for multiple epochs
     * 
     * @param train_data Training dataset
     * @param train_labels Training labels
     * @param epochs Number of training epochs
     * @param batch_size Mini-batch size (0 for full batch)
     * @param val_data Validation dataset (optional)
     * @param val_labels Validation labels (optional)
     * @param verbose Print training progress
     */
    void fit(const std::vector<std::vector<float>>& train_data,
            const std::vector<std::vector<float>>& train_labels,
            int epochs,
            int batch_size = 32,
            const std::vector<std::vector<float>>* val_data = nullptr,
            const std::vector<std::vector<float>>* val_labels = nullptr,
            bool verbose = true);
    
    // Evaluation
    /**
     * Evaluate network on test data
     * 
     * @param test_data Test input vectors
     * @param test_labels Test target vectors
     * @return Average loss on test set
     */
    float evaluate(const std::vector<std::vector<float>>& test_data,
                  const std::vector<std::vector<float>>& test_labels);
    
    /**
     * Compute classification accuracy
     * 
     * @param predictions Network predictions
     * @param targets True labels
     * @return Accuracy (0.0 to 1.0)
     */
    float compute_accuracy(const std::vector<std::vector<float>>& predictions,
                          const std::vector<std::vector<float>>& targets);
    
    // Network configuration
    /**
     * Set learning rate for all layers
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
     */
    void save(const std::string& filename) const;
    
    /**
     * Load network from file
     */
    void load(const std::string& filename);
    
    // Layer access
    int get_num_layers() const { return layers.size(); }
    const NeuronLayer& get_layer(int index) const { return layers[index]; }
};
```

## Loss Functions

### Supported Loss Types

```cpp
enum class LossType {
    MSE,              // Mean Squared Error
    MAE,              // Mean Absolute Error
    BINARY_CROSS_ENTROPY,   // Binary Classification
    CATEGORICAL_CROSS_ENTROPY,  // Multi-class Classification
    HUBER             // Robust regression loss
};
```

### Loss Function Formulations

| Loss Type | Formula | Use Case | Gradient |
|-----------|---------|----------|----------|
| MSE | L = ½Σ(yᵢ - ŷᵢ)² | Regression | ∇L = ŷ - y |
| MAE | L = Σ\|yᵢ - ŷᵢ\| | Robust regression | ∇L = sign(ŷ - y) |
| Binary CE | L = -Σ[y log(ŷ) + (1-y)log(1-ŷ)] | Binary classification | ∇L = (ŷ - y)/(ŷ(1-ŷ)) |
| Categorical CE | L = -Σ yᵢ log(ŷᵢ) | Multi-class | ∇L = ŷ - y (with softmax) |
| Huber | L = ½x² if \|x\|≤δ else δ(\|x\|-½δ) | Outlier-robust | Piecewise gradient |

## Usage Examples

### Example 1: Binary Classification (XOR Problem)

```cpp
#include "Neuron.hpp"
#include "NeuralNetwork.hpp"

int main() {
    // Define network architecture: 2 inputs, 4 hidden, 1 output
    std::vector<int> architecture = {2, 4, 1};
    std::vector<ActivationType> activations = {
        ActivationType::TANH,      // Hidden layer
        ActivationType::SIGMOID    // Output layer
    };
    
    // Create network
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
    
    // Train the network
    nn.fit(X, y, 10000, 4, nullptr, nullptr, true);
    
    // Test predictions
    for (size_t i = 0; i < X.size(); ++i) {
        auto pred = nn.predict(X[i]);
        std::cout << "Input: [" << X[i][0] << ", " << X[i][1] << "] "
                  << "Predicted: " << pred[0] 
                  << " Target: " << y[i][0] << std::endl;
    }
    
    // Save the model
    nn.save("xor_model.nn");
    
    return 0;
}
```

### Example 2: Multi-class Classification (Iris Dataset)

```cpp
#include "Neuron.hpp"
#include "NeuralNetwork.hpp"
#include <iostream>

int main() {
    // Network: 4 features → 8 hidden → 3 classes
    std::vector<int> architecture = {4, 8, 3};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,      // Hidden layer
        ActivationType::LINEAR     // Output (will apply softmax in loss)
    };
    
    NeuralNetwork nn(architecture, activations,
                     LossType::CATEGORICAL_CROSS_ENTROPY, 0.01f);
    nn.initialize_he();
    
    // Load Iris dataset (pseudo-code)
    auto [X_train, y_train] = load_iris_train();
    auto [X_test, y_test] = load_iris_test();
    
    // Train with validation
    nn.fit(X_train, y_train, 200, 16, &X_test, &y_test, true);
    
    // Evaluate
    float test_loss = nn.evaluate(X_test, y_test);
    float accuracy = nn.compute_accuracy(nn.predict_batch(X_test), y_test);
    
    std::cout << "Test Loss: " << test_loss << std::endl;
    std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
    
    // Print network summary
    nn.print_summary();
    
    return 0;
}
```

### Example 3: Regression (House Price Prediction)

```cpp
#include "Neuron.hpp"
#include "NeuralNetwork.hpp"

int main() {
    // Network: 10 features → 64 → 32 → 1 output
    std::vector<int> architecture = {10, 64, 32, 1};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::LINEAR    // Linear output for regression
    };
    
    NeuralNetwork nn(architecture, activations, LossType::MSE, 0.001f);
    nn.initialize_he();
    
    // Load housing dataset
    auto [X_train, y_train] = load_housing_data();
    
    // Normalize features (important for regression)
    normalize_features(X_train);
    
    // Train the model
    nn.fit(X_train, y_train, 500, 32, nullptr, nullptr, true);
    
    // Make predictions
    std::vector<float> new_house = {/* features */};
    normalize_features(new_house);
    auto price = nn.predict(new_house);
    
    std::cout << "Predicted price: $" << price[0] << std::endl;
    
    return 0;
}
```

### Example 4: Deep Network with Custom Architecture

```cpp
#include "Neuron.hpp"
#include "NeuralNetwork.hpp"

int main() {
    // Deep network: 784 → 256 → 128 → 64 → 10
    std::vector<int> architecture = {784, 256, 128, 64, 10};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::RELU,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(architecture, activations,
                     LossType::CATEGORICAL_CROSS_ENTROPY, 0.001f);
    nn.initialize_he();
    
    // Print network architecture
    nn.print_summary();
    
    // Load MNIST-like dataset
    auto [X_train, y_train] = load_mnist_train();
    auto [X_val, y_val] = load_mnist_val();
    
    // Train with mini-batch gradient descent
    nn.fit(X_train, y_train, 50, 128, &X_val, &y_val, true);
    
    // Plot training history (pseudo-code)
    plot_training_curves(nn.get_training_loss(), 
                        nn.get_validation_loss());
    
    return 0;
}
```

## Implementation Details

### Network Construction

```cpp
NeuralNetwork::NeuralNetwork(const std::vector<int>& sizes,
                            const std::vector<ActivationType>& activations,
                            LossType loss, float lr)
    : layer_sizes(sizes), loss_function(loss) {
    
    // Create layers
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        layers.emplace_back(sizes[i], sizes[i + 1], activations[i], lr);
    }
}
```

### Forward Propagation

```cpp
std::vector<float> NeuralNetwork::predict(const std::vector<float>& input) {
    std::vector<float> activation = input;
    
    // Pass through each layer
    for (auto& layer : layers) {
        activation = layer.forward(activation);
    }
    
    return activation;
}
```

### Backward Propagation

```cpp
float NeuralNetwork::train_sample(const std::vector<float>& input,
                                 const std::vector<float>& target) {
    // Forward pass
    std::vector<float> output = predict(input);
    
    // Compute loss
    float loss = compute_loss(output, target);
    
    // Compute output gradient
    std::vector<float> gradient = compute_loss_gradient(output, target);
    
    // Backward pass through layers
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i].backward(gradient);
    }
    
    return loss;
}
```

### Mini-Batch Training

```cpp
void NeuralNetwork::fit(const std::vector<std::vector<float>>& train_data,
                       const std::vector<std::vector<float>>& train_labels,
                       int epochs, int batch_size,
                       const std::vector<std::vector<float>>* val_data,
                       const std::vector<std::vector<float>>* val_labels,
                       bool verbose) {
    
    int n_samples = train_data.size();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        // Shuffle training data
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{});
        
        // Mini-batch training
        for (int i = 0; i < n_samples; i += batch_size) {
            int batch_end = std::min(i + batch_size, n_samples);
            
            std::vector<std::vector<float>> batch_inputs;
            std::vector<std::vector<float>> batch_targets;
            
            for (int j = i; j < batch_end; ++j) {
                batch_inputs.push_back(train_data[indices[j]]);
                batch_targets.push_back(train_labels[indices[j]]);
            }
            
            epoch_loss += train_batch(batch_inputs, batch_targets);
        }
        
        epoch_loss /= (n_samples / batch_size);
        training_loss_history.push_back(epoch_loss);
        
        // Validation
        if (val_data && val_labels) {
            float val_loss = evaluate(*val_data, *val_labels);
            validation_loss_history.push_back(val_loss);
            
            if (verbose && epoch % 10 == 0) {
                std::cout << "Epoch " << epoch 
                         << " - Loss: " << epoch_loss
                         << " - Val Loss: " << val_loss << std::endl;
            }
        } else if (verbose && epoch % 10 == 0) {
            std::cout << "Epoch " << epoch 
                     << " - Loss: " << epoch_loss << std::endl;
        }
    }
}
```

## Advanced Features

### Learning Rate Scheduling

```cpp
class LearningRateScheduler {
public:
    virtual float get_lr(int epoch, float initial_lr) = 0;
};

class StepDecay : public LearningRateScheduler {
private:
    int step_size;
    float gamma;
    
public:
    StepDecay(int step = 100, float decay = 0.5f)
        : step_size(step), gamma(decay) {}
    
    float get_lr(int epoch, float initial_lr) override {
        return initial_lr * std::pow(gamma, epoch / step_size);
    }
};

class ExponentialDecay : public LearningRateScheduler {
private:
    float decay_rate;
    
public:
    ExponentialDecay(float rate = 0.95f) : decay_rate(rate) {}
    
    float get_lr(int epoch, float initial_lr) override {
        return initial_lr * std::pow(decay_rate, epoch);
    }
};
```

### Early Stopping

```cpp
class EarlyStopping {
private:
    int patience;
    float min_delta;
    float best_loss;
    int counter;
    
public:
    EarlyStopping(int patience = 10, float min_delta = 1e-4f)
        : patience(patience), min_delta(min_delta),
          best_loss(std::numeric_limits<float>::max()), counter(0) {}
    
    bool should_stop(float current_loss) {
        if (current_loss < best_loss - min_delta) {
            best_loss = current_loss;
            counter = 0;
            return false;
        }
        
        counter++;
        return counter >= patience;
    }
};
```

### Regularization

```cpp
class L2Regularization {
public:
    static float compute_penalty(const NeuralNetwork& network, float lambda) {
        float penalty = 0.0f;
        
        for (int i = 0; i < network.get_num_layers(); ++i) {
            const auto& layer = network.get_layer(i);
            // Sum of squared weights
            // Implementation requires access to layer weights
        }
        
        return 0.5f * lambda * penalty;
    }
};
```

### Dropout (Future Enhancement)

```cpp
class Dropout {
private:
    float dropout_rate;
    bool training_mode;
    
public:
    Dropout(float rate = 0.5f) : dropout_rate(rate), training_mode(true) {}
    
    std::vector<float> forward(const std::vector<float>& input) {
        if (!training_mode) return input;
        
        std::vector<float> output = input;
        std::bernoulli_distribution dist(1.0f - dropout_rate);
        
        for (auto& val : output) {
            if (!dist(rng)) {
                val = 0.0f;
            } else {
                val /= (1.0f - dropout_rate);  // Inverted dropout
            }
        }
        
        return output;
    }
    
    void set_training(bool training) { training_mode = training; }
};
```

## Model Evaluation Metrics

### Classification Metrics

```cpp
class ClassificationMetrics {
public:
    static float accuracy(const std::vector<int>& predictions,
                         const std::vector<int>& targets) {
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == targets[i]) correct++;
        }
        return static_cast<float>(correct) / predictions.size();
    }
    
    static float precision(const std::vector<int>& predictions,
                          const std::vector<int>& targets, int positive_class) {
        int true_positives = 0, false_positives = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == positive_class) {
                if (targets[i] == positive_class) {
                    true_positives++;
                } else {
                    false_positives++;
                }
            }
        }
        
        return static_cast<float>(true_positives) / 
               (true_positives + false_positives);
    }
    
    static float recall(const std::vector<int>& predictions,
                       const std::vector<int>& targets, int positive_class) {
        int true_positives = 0, false_negatives = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (targets[i] == positive_class) {
                if (predictions[i] == positive_class) {
                    true_positives++;
                } else {
                    false_negatives++;
                }
            }
        }
        
        return static_cast<float>(true_positives) / 
               (true_positives + false_negatives);
    }
    
    static float f1_score(float precision, float recall) {
        return 2.0f * (precision * recall) / (precision + recall);
    }
};
```

### Regression Metrics

```cpp
class RegressionMetrics {
public:
    static float r_squared(const std::vector<float>& predictions,
                          const std::vector<float>& targets) {
        float mean = 0.0f;
        for (auto val : targets) mean += val;
        mean /= targets.size();
        
        float ss_tot = 0.0f, ss_res = 0.0f;
        for (size_t i = 0; i < targets.size(); ++i) {
            ss_tot += (targets[i] - mean) * (targets[i] - mean);
            ss_res += (targets[i] - predictions[i]) * 
                     (targets[i] - predictions[i]);
        }
        
        return 1.0f - (ss_res / ss_tot);
    }
    
    static float rmse(const std::vector<float>& predictions,
                     const std::vector<float>& targets) {
        float sum = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float diff = predictions[i] - targets[i];
            sum += diff * diff;
        }
        return std::sqrt(sum / predictions.size());
    }
};
```

## File Format for Model Persistence

### Network File Structure

```
# Neural Network v1.0
ARCHITECTURE
<num_layers>
<layer_sizes[0]> <layer_sizes[1]> ... <layer_sizes[n]>
LOSS_FUNCTION
<loss_type>
LAYERS
<layer_0_data>
<layer_1_data>
...
<layer_n_data>
```

### Save/Load Implementation

```cpp
void NeuralNetwork::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    file << "# Neural Network v1.0\n";
    file << "ARCHITECTURE\n";
    file << layer_sizes.size() << "\n";
    for (auto size : layer_sizes) {
        file << size << " ";
    }
    file << "\n";
    
    file << "LOSS_FUNCTION\n";
    file << static_cast<int>(loss_function) << "\n";
    
    file << "LAYERS\n";
    for (const auto& layer : layers) {
        layer.save(file);
    }
    
    file.close();
}

void NeuralNetwork::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    std::string line;
    
    // Parse architecture
    while (std::getline(file, line)) {
        if (line == "ARCHITECTURE") {
            int num_layers;
            file >> num_layers;
            layer_sizes.resize(num_layers);
            for (int i = 0; i < num_layers; ++i) {
                file >> layer_sizes[i];
            }
            std::getline(file, line);  // Consume newline
            break;
        }
    }
    
    // Parse loss function
    while (std::getline(file, line)) {
        if (line == "LOSS_FUNCTION") {
            int loss_int;
            file >> loss_int;
            loss_function = static_cast<LossType>(loss_int);
            std::getline(file, line);  // Consume newline
            break;
        }
    }
    
    // Load layers
    while (std::getline(file, line)) {
        if (line == "LAYERS") {
            layers.clear();
            for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                NeuronLayer layer(layer_sizes[i], layer_sizes[i + 1],
                                ActivationType::RELU);
                layer.load(file);
                layers.push_back(layer);
            }
            break;
        }
    }
    
    file.close();
}
```

## Performance Optimization

### Parallelization Strategies

```cpp
// Parallel batch processing
std::vector<std::vector<float>> NeuralNetwork::predict_batch_parallel(
    const std::vector<std::vector<float>>& inputs) {
    
    std::vector<std::vector<float>> outputs(inputs.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < inputs.size(); ++i) {
        outputs[i] = predict(inputs[i]);
    }
    
    return outputs;
}
```

### Memory Management

```cpp
class MemoryPool {
private:
    std::vector<std::vector<float>> buffer_pool;
    size_t next_buffer;
    
public:
    MemoryPool(size_t num_buffers, size_t buffer_size) {
        buffer_pool.resize(num_buffers);
        for (auto& buf : buffer_pool) {
            buf.resize(buffer_size);
        }
        next_buffer = 0;
    }
    
    std::vector<float>& get_buffer() {
        return buffer_pool[next_buffer++ % buffer_pool.size()];
    }
};
```

## Integration with Existing Components

### With BPE Tokenizer

```cpp
#include "BPETokenizer.hpp"
#include "NeuralNetwork.hpp"

class TextClassifier {
private:
    BPETokenizer tokenizer;
    NeuralNetwork network;
    int max_seq_length;
    
public:
    TextClassifier(int vocab_size, int embedding_dim, int num_classes)
        : tokenizer(vocab_size),
          network({max_seq_length * embedding_dim, 128, 64, num_classes},
                 {ActivationType::RELU, ActivationType::RELU, 
                  ActivationType::LINEAR},
                 LossType::CATEGORICAL_CROSS_ENTROPY),
          max_seq_length(100) {}
    
    std::vector<float> classify(const std::string& text) {
        // Tokenize
        auto tokens = tokenizer.encode(text);
        
        // Convert to features (simplified)
        std::vector<float> features = tokens_to_features(tokens);
        
        // Predict
        return network.predict(features);
    }
};
```

### With Transformer Components

```cpp
#include "encoder.hpp"
#include "NeuralNetwork.hpp"

class HybridModel {
private:
    TokenEmbedding embedder;
    PositionalEncoding pos_encoder;
    NeuralNetwork classifier;
    
public:
    HybridModel(int vocab_size, int d_model, int num_classes)
        : embedder(vocab_size, d_model),
          pos_encoder(512, d_model),
          classifier({d_model, 128, num_classes},
                    {ActivationType::GELU, ActivationType::LINEAR},
                    LossType::CATEGORICAL_CROSS_ENTROPY) {}
    
    std::vector<float> forward(const std::vector<int>& token_ids) {
        // Embed tokens
        auto embeddings = embedder.forward(token_ids);
        
        // Add positional encoding
        auto encoded = pos_encoder.forward(embeddings);
        
        // Convert matrix to vector (mean pooling)
        std::vector<float> pooled = mean_pool(encoded);
        
        // Classify
        return classifier.predict(pooled);
    }
};
```

## Testing Examples

### Unit Tests

```cpp
#include "NeuralNetwork.hpp"
#include <cassert>
#include <cmath>

void test_network_construction() {
    std::vector<int> arch = {2, 3, 1};
    std::vector<ActivationType> acts = {
        ActivationType::RELU,
        ActivationType::SIGMOID
    };
    
    NeuralNetwork nn(arch, acts);
    assert(nn.get_num_layers() == 2);
}

void test_forward_pass() {
    std::vector<int> arch = {2, 2, 1};
    std::vector<ActivationType> acts = {
        ActivationType::LINEAR,
        ActivationType::LINEAR
    };
    
    NeuralNetwork nn(arch, acts);
    
    std::vector<float> input = {1.0f, 2.0f};
    auto output = nn.predict(input);
    
    assert(output.size() == 1);
}

void test_training() {
    // Simple linear regression: y = 2x
    std::vector<int> arch = {1, 1};
    std::vector<ActivationType> acts = {ActivationType::LINEAR};
    
    NeuralNetwork nn(arch, acts, LossType::MSE, 0.01f);
    
    std::vector<std::vector<float>> X = {{1.0f}, {2.0f}, {3.0f}, {4.0f}};
    std::vector<std::vector<float>> y = {{2.0f}, {4.0f}, {6.0f}, {8.0f}};
    
    nn.fit(X, y, 1000, 4);
    
    auto pred = nn.predict({5.0f});
    assert(std::abs(pred[0] - 10.0f) < 0.5f);  // Should be close to 10
}

void run_all_tests() {
    test_network_construction();
    test_forward_pass();
    test_training();
    std::cout << "All tests passed!" << std::endl;
}
```

## Best Practices

### 1. Data Preprocessing

```cpp
// Normalize features to [0, 1] or [-1, 1]
void normalize_features(std::vector<std::vector<float>>& data) {
    int n_features = data[0].size();
    
    for (int j = 0; j < n_features; ++j) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        
        for (auto& sample : data) {
            min_val = std::min(min_val, sample[j]);
            max_val = std::max(max_val, sample[j]);
        }
        
        float range = max_val - min_val;
        for (auto& sample : data) {
            sample[j] = (sample[j] - min_val) / range;
        }
    }
}

// Standardize features (zero mean, unit variance)
void standardize_features(std::vector<std::vector<float>>& data) {
    int n_features = data[0].size();
    
    for (int j = 0; j < n_features; ++j) {
        float mean = 0.0f, std_dev = 0.0f;
        
        for (auto& sample : data) {
            mean += sample[j];
        }
        mean /= data.size();
        
        for (auto& sample : data) {
            std_dev += (sample[j] - mean) * (sample[j] - mean);
        }
        std_dev = std::sqrt(std_dev / data.size());
        
        for (auto& sample : data) {
            sample[j] = (sample[j] - mean) / std_dev;
        }
    }
}
```

### 2. Hyperparameter Tuning

```cpp
struct HyperParameters {
    std::vector<int> architecture;
    std::vector<ActivationType> activations;
    float learning_rate;
    int batch_size;
    LossType loss;
};

HyperParameters grid_search(
    const std::vector<std::vector<float>>& train_data,
    const std::vector<std::vector<float>>& train_labels,
    const std::vector<std::vector<float>>& val_data,
    const std::vector<std::vector<float>>& val_labels) {
    
    std::vector<float> learning_rates = {0.001f, 0.01f, 0.1f};
    std::vector<int> batch_sizes = {16, 32, 64};
    
    HyperParameters best_params;
    float best_val_loss = std::numeric_limits<float>::max();
    
    for (auto lr : learning_rates) {
        for (auto bs : batch_sizes) {
            NeuralNetwork nn({4, 8, 3}, 
                           {ActivationType::RELU, ActivationType::LINEAR},
                           LossType::CATEGORICAL_CROSS_ENTROPY, lr);
            
            nn.fit(train_data, train_labels, 100, bs, 
                  &val_data, &val_labels, false);
            
            float val_loss = nn.evaluate(val_data, val_labels);
            
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                best_params.learning_rate = lr;
                best_params.batch_size = bs;
            }
        }
    }
    
    return best_params;
}
```

### 3. Model Checkpointing

```cpp
class ModelCheckpoint {
private:
    std::string filepath;
    float best_loss;
    
public:
    ModelCheckpoint(const std::string& path)
        : filepath(path), best_loss(std::numeric_limits<float>::max()) {}
    
    void check(const NeuralNetwork& network, float current_loss) {
        if (current_loss < best_loss) {
            best_loss = current_loss;
            network.save(filepath);
            std::cout << "Model saved with loss: " << current_loss << std::endl;
        }
    }
};
```

## Future Enhancements

1. **Batch Normalization**: Add normalization between layers
2. **Residual Connections**: Support for ResNet-style skip connections
3. **Attention Mechanisms**: Self-attention layers
4. **Recurrent Layers**: LSTM and GRU support
5. **Convolutional Layers**: For image processing
6. **GPU Acceleration**: CUDA backend for faster training
7. **Automatic Differentiation**: More flexible gradient computation
8. **Model Visualization**: Export to visualization formats
9. **Transfer Learning**: Pre-trained model support
10. **Distributed Training**: Multi-GPU and multi-node training

## References

- **Deep Learning**: Goodfellow, Bengio, and Courville (2016)
- **Neural Networks**: Rojas, R. (1996)
- **Backpropagation**: Rumelhart et al. (1986)
- **Adam Optimizer**: Kingma & Ba (2014)
- **Dropout**: Srivastava et al. (2014)
- **Batch Normalization**: Ioffe & Szegedy (2015)

## Related Components

- `Neuron`: Single neuron implementation
- `NeuronLayer`: Layer of neurons
- `BPETokenizer`: Text tokenization
- `Encoder`: Transformer encoder
- `Matrix`: Linear algebra operations

## Notes

- The NeuralNetwork class provides a flexible framework for experimentation
- For production use, consider optimizing with matrix operations instead of neuron-by-neuron computation
- Always normalize/standardize input features for better convergence
- Use appropriate activation functions for each layer type
- Monitor validation loss to prevent overfitting
- Consider using learning rate schedules for better convergence
