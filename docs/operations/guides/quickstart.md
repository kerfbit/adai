# Neural Network Quick Start Guide

## Creating a Network

```cpp
#include "NeuralNetwork.hpp"

// Define architecture: [input_size, hidden1, hidden2, ..., output_size]
std::vector<int> architecture = {4, 8, 3};

// Define activations (one per layer, excludes input)
std::vector<ActivationType> activations = {
    ActivationType::RELU,      // First hidden layer
    ActivationType::SIGMOID    // Output layer
};

// Create network
NeuralNetwork nn(architecture, activations,
                 LossType::CATEGORICAL_CROSS_ENTROPY, 0.01f);

// Initialize weights
nn.initialize_he();  // For ReLU
// or
nn.initialize_xavier();  // For Sigmoid/Tanh
```

## Training

```cpp
// Prepare data
std::vector<std::vector<float>> X_train = {{...}, {...}, ...};
std::vector<std::vector<float>> y_train = {{...}, {...}, ...};

// Train
nn.fit(X_train, y_train,
       100,    // epochs
       32,     // batch_size (0 = full batch)
       nullptr, nullptr, // validation data (optional)
       true);  // verbose

// With validation
std::vector<std::vector<float>> X_val = {{...}, {...}, ...};
std::vector<std::vector<float>> y_val = {{...}, {...}, ...};

nn.fit(X_train, y_train, 100, 32, &X_val, &y_val, true);
```

## Prediction

```cpp
// Single prediction
std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
auto output = nn.predict(input);

// Batch prediction
std::vector<std::vector<float>> inputs = {{...}, {...}, ...};
auto outputs = nn.predict_batch(inputs);
```

## Evaluation

```cpp
// Compute loss
float test_loss = nn.evaluate(X_test, y_test);

// Compute accuracy (classification only)
auto predictions = nn.predict_batch(X_test);
float accuracy = nn.compute_accuracy(predictions, y_test);
std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
```

## Save/Load

```cpp
// Save trained model
nn.save("my_model.dat");

// Load model
NeuralNetwork loaded_nn(architecture, activations, loss_type);
loaded_nn.load("my_model.dat");
```

## Common Architectures

### Binary Classification

```cpp
std::vector<int> arch = {features, 16, 8, 1};
std::vector<ActivationType> acts = {
    ActivationType::RELU,
    ActivationType::RELU,
    ActivationType::SIGMOID
};
NeuralNetwork nn(arch, acts, LossType::BINARY_CROSS_ENTROPY, 0.01f);
```

### Multi-class Classification

```cpp
std::vector<int> arch = {features, 32, 16, num_classes};
std::vector<ActivationType> acts = {
    ActivationType::RELU,
    ActivationType::RELU,
    ActivationType::SIGMOID
};
NeuralNetwork nn(arch, acts, LossType::CATEGORICAL_CROSS_ENTROPY, 0.01f);
```

### Regression

```cpp
std::vector<int> arch = {features, 64, 32, 1};
std::vector<ActivationType> acts = {
    ActivationType::RELU,
    ActivationType::RELU,
    ActivationType::LINEAR
};
NeuralNetwork nn(arch, acts, LossType::MSE, 0.01f);
```

## Available Activation Functions

- `ActivationType::LINEAR` - No activation (regression output)
- `ActivationType::SIGMOID` - Binary classification output
- `ActivationType::TANH` - Hidden layers
- `ActivationType::RELU` - Most common for hidden layers
- `ActivationType::LEAKY_RELU` - Prevent dead neurons
- `ActivationType::GELU` - Transformer networks
- `ActivationType::SOFTPLUS` - Smooth ReLU alternative

## Available Loss Functions

- `LossType::MSE` - Regression
- `LossType::MAE` - Robust regression
- `LossType::BINARY_CROSS_ENTROPY` - Binary classification
- `LossType::CATEGORICAL_CROSS_ENTROPY` - Multi-class
- `LossType::HUBER` - Outlier-robust regression

## Utility Functions

```cpp
// Print network information
nn.print_summary();

// Change learning rate
nn.set_learning_rate(0.001f);

// Access training history
auto& train_loss = nn.get_training_loss();
auto& val_loss = nn.get_validation_loss();
auto& train_acc = nn.get_training_accuracy();
auto& val_acc = nn.get_validation_accuracy();

// Access layers
int num_layers = nn.get_num_layers();
const auto& layer = nn.get_layer(0);
```

## Best Practices

1. **Normalize inputs**: Scale to [0, 1] or standardize
2. **Choose activation wisely**:
   - ReLU for hidden layers
   - Sigmoid for binary output
   - Linear for regression output
3. **Initialize properly**:
   - He init for ReLU
   - Xavier init for Sigmoid/Tanh
4. **Use mini-batches**: Typically 16-128
5. **Monitor validation**: Prevent overfitting
6. **Start with simple**: 1-2 hidden layers
7. **Tune learning rate**: Start with 0.01, adjust as needed

## Troubleshooting

### Network not learning

- Try different learning rate (0.001 - 0.1)
- Check data normalization
- Increase network capacity (more neurons/layers)
- Check loss function matches task

### Overfitting

- Reduce network size
- Use validation set
- Stop training earlier
- Add regularization (future feature)

### Poor accuracy

- Increase training epochs
- Adjust architecture
- Normalize inputs
- Try different activation functions

### Exploding gradients

- Reduce learning rate
- Check for proper initialization
- Normalize inputs
