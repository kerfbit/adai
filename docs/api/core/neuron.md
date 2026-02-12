# Neuron Class - Context Document

## Overview

The `Neuron` class represents a single computational unit in a neural network layer. It encapsulates the fundamental operations of a neuron: weighted sum of inputs, bias addition, and activation function application. This class serves as a building block for constructing feed-forward neural networks and can be used within the transformer architecture.

## Purpose

The Neuron class is designed to:

- Provide a low-level abstraction for neural network computations
- Enable flexible construction of custom layer architectures
- Support both forward pass (inference) and gradient computation (training)
- Integrate seamlessly with existing components (Matrix, Activation, etc.)

## Architecture

### Core Components

```text
Inputs (x₁, x₂, ..., xₙ)
         ↓
    Weighted Sum: z = Σ(wᵢ × xᵢ) + b
         ↓
Activation Function: a = f(z)
         ↓
      Output (a)
```

### Mathematical Formulation

**Forward Pass:**

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
a = f(z)
```

Where:

- `w₁, w₂, ..., wₙ` are learnable weights
- `x₁, x₂, ..., xₙ` are input values
- `b` is the learnable bias term
- `f(·)` is the activation function
- `z` is the pre-activation value
- `a` is the activation (output)

**Backward Pass (Gradient Computation):**

```text
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w = δ × x
∂L/∂b = ∂L/∂a × ∂a/∂z = δ
∂L/∂x = ∂L/∂a × ∂a/∂z × ∂z/∂x = δ × w
```

Where:

- `L` is the loss function
- `δ = ∂L/∂a × f'(z)` is the error gradient

## Class Interface

### Public Members

```cpp
class Neuron {
private:
    std::vector<float> weights;  // Weight vector [w₁, w₂, ..., wₙ]
    float bias;                  // Bias term b
    float learning_rate;         // Learning rate for weight updates

    // Cached values for backpropagation
    std::vector<float> last_input;
    float last_pre_activation;   // z value
    float last_activation;        // a value

    // Activation function pointer
    ActivationType activation_type;

public:
    // Constructors
    Neuron(int input_size, ActivationType activation = ActivationType::RELU,
           float lr = 0.01f);
    Neuron(const std::vector<float>& init_weights, float init_bias,
           ActivationType activation = ActivationType::RELU, float lr = 0.01f);

    // Forward pass
    float forward(const std::vector<float>& inputs);

    // Backward pass
    std::vector<float> backward(float gradient);

    // Weight updates
    void update_weights();

    // Getters and setters
    const std::vector<float>& get_weights() const;
    float get_bias() const;
    void set_weights(const std::vector<float>& new_weights);
    void set_bias(float new_bias);
    void set_learning_rate(float lr);

    // Utility functions
    void randomize(float scale = 0.1f);
    void xavier_init(int fan_in, int fan_out);
    void he_init(int fan_in);

    // Serialization
    void save(std::ofstream& file) const;
    void load(std::ifstream& file);
};
```

## Activation Functions

### Supported Types

```cpp
enum class ActivationType {
    LINEAR,    // f(x) = x
    SIGMOID,   // f(x) = 1 / (1 + e^(-x))
    TANH,      // f(x) = tanh(x)
    RELU,      // f(x) = max(0, x)
    LEAKY_RELU,// f(x) = x if x > 0, else αx (α = 0.01)
    GELU,      // f(x) = x × Φ(x), Gaussian Error Linear Unit
    SOFTPLUS   // f(x) = ln(1 + e^x)
};
```

### Activation Function Properties

| Function | Range | Derivative | Use Case |
| ---------- | ------- | ------------ | ---------- |
| Linear | (-∞, ∞) | f'(x) = 1 | Regression output |
| Sigmoid | (0, 1) | f'(x) = f(x)(1-f(x)) | Binary classification |
| Tanh | (-1, 1) | f'(x) = 1 - f(x)² | Hidden layers |
| ReLU | [0, ∞) | f'(x) = 1 if x>0 else 0 | Deep networks |
| Leaky ReLU | (-∞, ∞) | f'(x) = 1 if x>0 else α | Avoid dead neurons |
| GELU | (-∞, ∞) | f'(x) = Φ(x) + x·φ(x) | Transformers |
| Softplus | (0, ∞) | f'(x) = sigmoid(x) | Smooth ReLU |

## Weight Initialization Strategies

### Xavier Initialization (Glorot)

Best for: Sigmoid, Tanh activations

```cpp
void Neuron::xavier_init(int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& w : weights) {
        w = dist(rng);
    }
    bias = 0.0f;
}
```

### He Initialization

Best for: ReLU, Leaky ReLU activations

```cpp
void Neuron::he_init(int fan_in) {
    float stddev = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (auto& w : weights) {
        w = dist(rng);
    }
    bias = 0.0f;
}
```

## Usage Examples

### Creating a Single Neuron

```cpp
// Create a neuron with 5 inputs, ReLU activation
Neuron neuron(5, ActivationType::RELU, 0.01f);
neuron.xavier_init(5, 1);

// Forward pass
std::vector<float> inputs = {1.0f, 0.5f, -0.3f, 0.8f, -0.2f};
float output = neuron.forward(inputs);

// Backward pass with gradient from next layer
float gradient = 0.5f;  // Error gradient from loss
std::vector<float> input_gradients = neuron.backward(gradient);

// Update weights
neuron.update_weights();
```

### Building a Layer with Multiple Neurons

```cpp
class NeuronLayer {
private:
    std::vector<Neuron> neurons;
    int input_size, output_size;

public:
    NeuronLayer(int in_size, int out_size, ActivationType activation)
        : input_size(in_size), output_size(out_size) {

        for (int i = 0; i < out_size; ++i) {
            neurons.emplace_back(in_size, activation);
            neurons[i].he_init(in_size);
        }
    }

    std::vector<float> forward(const std::vector<float>& inputs) {
        std::vector<float> outputs;
        for (auto& neuron : neurons) {
            outputs.push_back(neuron.forward(inputs));
        }
        return outputs;
    }

    std::vector<float> backward(const std::vector<float>& gradients) {
        std::vector<float> input_gradients(input_size, 0.0f);

        for (size_t i = 0; i < neurons.size(); ++i) {
            auto neuron_grads = neurons[i].backward(gradients[i]);
            for (size_t j = 0; j < input_gradients.size(); ++j) {
                input_gradients[j] += neuron_grads[j];
            }
        }
        return input_gradients;
    }

    void update_weights() {
        for (auto& neuron : neurons) {
            neuron.update_weights();
        }
    }
};
```

### Integration with Transformer Architecture

```cpp
// Using neurons in feed-forward network
class CustomFeedForward {
private:
    NeuronLayer layer1;
    NeuronLayer layer2;

public:
    CustomFeedForward(int d_model, int d_ff)
        : layer1(d_model, d_ff, ActivationType::GELU),
          layer2(d_ff, d_model, ActivationType::LINEAR) {}

    std::vector<float> forward(const std::vector<float>& input) {
        auto hidden = layer1.forward(input);
        return layer2.forward(hidden);
    }
};
```

## Implementation Details

### Forward Pass Implementation

```cpp
float Neuron::forward(const std::vector<float>& inputs) {
    assert(inputs.size() == weights.size());

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
```

### Backward Pass Implementation

```cpp
std::vector<float> Neuron::backward(float gradient) {
    // Compute activation gradient: δ = gradient × f'(z)
    float delta = gradient * activation_derivative(last_pre_activation,
                                                    activation_type);

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
```

## Performance Considerations

### Memory Efficiency

- **Weight Storage**: O(n) where n is input size
- **Cache Storage**: O(n) for gradient computation
- **Recommendation**: For large networks, consider batch processing

### Computational Complexity

- **Forward Pass**: O(n) - linear in input size
- **Backward Pass**: O(n) - linear in input size
- **Update**: O(n) - linear in number of weights

### Optimization Tips

1. **Vectorization**: Use SIMD instructions for dot product computation
2. **Cache Locality**: Store neuron weights contiguously in memory
3. **Batch Processing**: Process multiple samples simultaneously
4. **Sparse Weights**: Consider pruning small weights to zero

## Integration with Existing Components

### With Matrix Class

```cpp
// Convert neuron layer to matrix operations for efficiency
Matrix layer_to_matrix(const NeuronLayer& layer) {
    // Convert individual neurons to weight matrix
    // Each row represents one neuron's weights
}
```

### With Activation Class

```cpp
// Use existing Activation class implementations
float apply_activation(float x, ActivationType type) {
    Matrix input(1, 1);
    input(0, 0) = x;

    switch(type) {
        case ActivationType::GELU:
            return Activation::gelu(input)(0, 0);
        // ... other types
    }
}
```

## Testing Strategy

### Unit Tests

```cpp
// Test forward pass
void test_forward_pass() {
    Neuron n(3, ActivationType::LINEAR);
    n.set_weights({1.0f, 2.0f, 3.0f});
    n.set_bias(1.0f);

    std::vector<float> input = {1.0f, 1.0f, 1.0f};
    float output = n.forward(input);

    assert(std::abs(output - 7.0f) < 1e-5f);  // 1+2+3+1 = 7
}

// Test gradient computation
void test_backward_pass() {
    Neuron n(2, ActivationType::LINEAR, 0.1f);
    n.set_weights({1.0f, 1.0f});
    n.set_bias(0.0f);

    std::vector<float> input = {1.0f, 1.0f};
    float output = n.forward(input);  // output = 2

    // Compute gradients (loss = 0.5 * (output - target)²)
    float target = 3.0f;
    float gradient = output - target;  // gradient = -1

    auto input_grads = n.backward(gradient);

    // Check weight updates
    auto new_weights = n.get_weights();
    assert(std::abs(new_weights[0] - 1.1f) < 1e-5f);  // 1.0 - 0.1*(-1)*1
}
```

### Integration Tests

```cpp
// Test with small network
void test_xor_network() {
    // XOR function learning test
    NeuronLayer hidden(2, 2, ActivationType::TANH);
    NeuronLayer output(2, 1, ActivationType::SIGMOID);

    // Training data for XOR
    std::vector<std::pair<std::vector<float>, float>> data = {
        {{0.0f, 0.0f}, 0.0f},
        {{0.0f, 1.0f}, 1.0f},
        {{1.0f, 0.0f}, 1.0f},
        {{1.0f, 1.0f}, 0.0f}
    };

    // Train for several epochs
    // Verify convergence
}
```

## Save/Load Functionality

### File Format

```text
# Neuron Weights v1.0
INPUT_SIZE <n>
ACTIVATION <type>
LEARNING_RATE <lr>
WEIGHTS
<w₁>
<w₂>
...
<wₙ>
BIAS
<b>
```

### Implementation

```cpp
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
    // Parse header and restore state
    // Implementation similar to BPETokenizer::load_vocab()
}
```

## Future Enhancements

1. **Dropout Support**: Add regularization during training
2. **Weight Decay**: L2 regularization for preventing overfitting
3. **Momentum**: Add momentum to weight updates
4. **Adaptive Learning Rates**: Adam, RMSprop optimizers
5. **Quantization**: Support for reduced precision (int8, fp16)
6. **GPU Acceleration**: CUDA/OpenCL support for parallel computation

## References

- **Backpropagation**: Rumelhart et al., "Learning representations by back-propagating errors" (1986)
- **Xavier Initialization**: Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010)
- **He Initialization**: He et al., "Delving Deep into Rectifiers" (2015)
- **GELU Activation**: Hendrycks & Gimpel, "Gaussian Error Linear Units" (2016)

## Related Components

- `Matrix`: Core tensor operations
- `Activation`: Activation function implementations
- `FeedForward`: High-level feed-forward network
- `LayerNorm`: Normalization for stability
- `Encoder`: Complete transformer encoder

## Notes

- The Neuron class provides a pedagogical foundation but may be superseded by matrix-based implementations for performance
- Consider using this class for experimentation and prototyping before optimizing with vectorized operations
- Integrates naturally with existing BPETokenizer for end-to-end NLP pipelines
