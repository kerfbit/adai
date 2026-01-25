#include "NeuralNetwork.hpp"
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>

// Constructor
NeuralNetwork::NeuralNetwork(const std::vector<int>& sizes,
                            const std::vector<ActivationType>& activations,
                            LossType loss,
                            float learning_rate)
    : layer_sizes(sizes), loss_function(loss) {
    
    assert(sizes.size() >= 2 && "Network must have at least input and output layers");
    assert(activations.size() == sizes.size() - 1 && "Activation count must match layer count");
    
    // Create layers
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        layers.emplace_back(sizes[i], sizes[i + 1], activations[i], learning_rate);
    }
}

// Forward pass
std::vector<float> NeuralNetwork::predict(const std::vector<float>& input) {
    assert(input.size() == static_cast<size_t>(layer_sizes[0]) && "Input size mismatch");
    
    std::vector<float> activation = input;
    for (auto& layer : layers) {
        activation = layer.forward(activation);
    }
    return activation;
}

std::vector<std::vector<float>> NeuralNetwork::predict_batch(
    const std::vector<std::vector<float>>& inputs) {
    
    std::vector<std::vector<float>> predictions;
    predictions.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        predictions.push_back(predict(input));
    }
    
    return predictions;
}

// Loss computation
float NeuralNetwork::compute_loss(const std::vector<float>& predictions,
                                 const std::vector<float>& targets) {
    assert(predictions.size() == targets.size() && "Prediction/target size mismatch");
    
    float loss = 0.0f;
    
    switch (loss_function) {
        case LossType::MSE: {
            // Mean Squared Error: L = 0.5 * Σ(y - ŷ)²
            for (size_t i = 0; i < predictions.size(); ++i) {
                float diff = predictions[i] - targets[i];
                loss += diff * diff;
            }
            loss *= 0.5f;
            break;
        }
        
        case LossType::MAE: {
            // Mean Absolute Error: L = Σ|y - ŷ|
            for (size_t i = 0; i < predictions.size(); ++i) {
                loss += std::abs(predictions[i] - targets[i]);
            }
            break;
        }
        
        case LossType::BINARY_CROSS_ENTROPY: {
            // Binary Cross Entropy: L = -Σ[y log(ŷ) + (1-y)log(1-ŷ)]
            const float epsilon = 1e-7f;
            for (size_t i = 0; i < predictions.size(); ++i) {
                float p = std::max(epsilon, std::min(1.0f - epsilon, predictions[i]));
                loss -= targets[i] * std::log(p) + (1.0f - targets[i]) * std::log(1.0f - p);
            }
            break;
        }
        
        case LossType::CATEGORICAL_CROSS_ENTROPY: {
            // Categorical Cross Entropy: L = -Σ y_i log(ŷ_i)
            const float epsilon = 1e-7f;
            for (size_t i = 0; i < predictions.size(); ++i) {
                float p = std::max(epsilon, predictions[i]);
                loss -= targets[i] * std::log(p);
            }
            break;
        }
        
        case LossType::HUBER: {
            // Huber Loss: L = 0.5*x² if |x|<=δ else δ(|x|-0.5δ)
            const float delta = 1.0f;
            for (size_t i = 0; i < predictions.size(); ++i) {
                float diff = std::abs(predictions[i] - targets[i]);
                if (diff <= delta) {
                    loss += 0.5f * diff * diff;
                } else {
                    loss += delta * (diff - 0.5f * delta);
                }
            }
            break;
        }
    }
    
    return loss;
}

std::vector<float> NeuralNetwork::compute_loss_gradient(
    const std::vector<float>& predictions,
    const std::vector<float>& targets) {
    
    assert(predictions.size() == targets.size() && "Prediction/target size mismatch");
    
    std::vector<float> gradients(predictions.size());
    
    switch (loss_function) {
        case LossType::MSE: {
            // ∂L/∂ŷ = ŷ - y
            for (size_t i = 0; i < predictions.size(); ++i) {
                gradients[i] = predictions[i] - targets[i];
            }
            break;
        }
        
        case LossType::MAE: {
            // ∂L/∂ŷ = sign(ŷ - y)
            for (size_t i = 0; i < predictions.size(); ++i) {
                float diff = predictions[i] - targets[i];
                gradients[i] = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
            }
            break;
        }
        
        case LossType::BINARY_CROSS_ENTROPY: {
            // ∂L/∂ŷ = (ŷ - y) / (ŷ(1-ŷ))
            const float epsilon = 1e-7f;
            for (size_t i = 0; i < predictions.size(); ++i) {
                float p = std::max(epsilon, std::min(1.0f - epsilon, predictions[i]));
                gradients[i] = (p - targets[i]) / (p * (1.0f - p));
            }
            break;
        }
        
        case LossType::CATEGORICAL_CROSS_ENTROPY: {
            // ∂L/∂ŷ = -y/ŷ (simplified with softmax: ŷ - y)
            // For simplicity, using direct gradient
            for (size_t i = 0; i < predictions.size(); ++i) {
                gradients[i] = predictions[i] - targets[i];
            }
            break;
        }
        
        case LossType::HUBER: {
            // ∂L/∂ŷ = x if |x|<=δ else δ*sign(x)
            const float delta = 1.0f;
            for (size_t i = 0; i < predictions.size(); ++i) {
                float diff = predictions[i] - targets[i];
                if (std::abs(diff) <= delta) {
                    gradients[i] = diff;
                } else {
                    gradients[i] = delta * ((diff > 0.0f) ? 1.0f : -1.0f);
                }
            }
            break;
        }
    }
    
    return gradients;
}

// Gradient clipping
void NeuralNetwork::clip_gradients(std::vector<float>& gradients, float max_norm) {
    float norm = 0.0f;
    for (float g : gradients) {
        norm += g * g;
    }
    norm = std::sqrt(norm);
    
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (float& g : gradients) {
            g *= scale;
        }
    }
}

bool NeuralNetwork::is_valid_output(const std::vector<float>& output) {
    for (float val : output) {
        if (std::isnan(val) || std::isinf(val)) {
            return false;
        }
    }
    return true;
}

bool NeuralNetwork::check_network_health() {
    // Check all layers for NaN/Inf in weights
    for (const auto& layer : layers) {
        for (int i = 0; i < layer.size(); ++i) {
            const auto& neuron = layer.get_neuron(i);
            const auto& weights = neuron.get_weights();
            float bias = neuron.get_bias();
            
            if (std::isnan(bias) || std::isinf(bias)) {
                return false;
            }
            
            for (float w : weights) {
                if (std::isnan(w) || std::isinf(w)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Training
float NeuralNetwork::train_sample(const std::vector<float>& input,
                                  const std::vector<float>& target) {
    // Forward pass
    auto prediction = predict(input);
    
    // Check for numerical issues
    if (!is_valid_output(prediction)) {
        // Reinitialize network if we hit numerical instability
        initialize_he();
        prediction = predict(input);
    }
    
    // Compute loss
    float loss = compute_loss(prediction, target);
    
    // Check if loss is valid
    if (std::isnan(loss) || std::isinf(loss)) {
        return 0.0f; // Return zero loss for invalid state
    }
    
    // Compute output gradient
    auto gradient = compute_loss_gradient(prediction, target);
    
    // Clip gradient to prevent explosion
    clip_gradients(gradient, 5.0f);
    
    // Backward pass through all layers
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i].backward(gradient);
        // Clip intermediate gradients
        clip_gradients(gradient, 5.0f);
    }
    
    return loss;
}

float NeuralNetwork::train_batch(const std::vector<std::vector<float>>& inputs,
                                const std::vector<std::vector<float>>& targets) {
    assert(inputs.size() == targets.size() && "Input/target batch size mismatch");
    
    if (inputs.empty()) {
        return 0.0f;
    }
    
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        total_loss += train_sample(inputs[i], targets[i]);
    }
    
    return total_loss / inputs.size();
}

void NeuralNetwork::fit(const std::vector<std::vector<float>>& train_data,
                       const std::vector<std::vector<float>>& train_labels,
                       int epochs,
                       int batch_size,
                       const std::vector<std::vector<float>>* val_data,
                       const std::vector<std::vector<float>>* val_labels,
                       bool verbose) {
    
    assert(train_data.size() == train_labels.size() && "Training data/label size mismatch");
    
    // Use full batch if batch_size is 0
    if (batch_size == 0) {
        batch_size = train_data.size();
    }
    
    // Create shuffled indices
    std::vector<size_t> indices(train_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 rng(rd());
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data
        std::shuffle(indices.begin(), indices.end(), rng);
        
        float epoch_loss = 0.0f;
        int num_batches = (train_data.size() + batch_size - 1) / batch_size;
        
        // Mini-batch training
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, static_cast<int>(train_data.size()));
            
            // Create mini-batch
            std::vector<std::vector<float>> batch_inputs;
            std::vector<std::vector<float>> batch_labels;
            
            for (int i = start_idx; i < end_idx; ++i) {
                batch_inputs.push_back(train_data[indices[i]]);
                batch_labels.push_back(train_labels[indices[i]]);
            }
            
            // Train on batch
            float batch_loss = train_batch(batch_inputs, batch_labels);
            epoch_loss += batch_loss * (end_idx - start_idx);
        }
        
        epoch_loss /= train_data.size();
        
        // Check for numerical stability
        if (std::isnan(epoch_loss) || std::isinf(epoch_loss)) {
            // Critical failure - reinitialize and restart this epoch
            initialize_he();
            continue;
        }
        
        training_loss_history.push_back(epoch_loss);
        
        // Validation
        if (val_data != nullptr && val_labels != nullptr) {
            float val_loss = evaluate(*val_data, *val_labels);
            validation_loss_history.push_back(val_loss);
            
            // Compute accuracies if classification
            if (loss_function == LossType::BINARY_CROSS_ENTROPY || 
                loss_function == LossType::CATEGORICAL_CROSS_ENTROPY) {
                
                auto train_preds = predict_batch(train_data);
                float train_acc = compute_accuracy(train_preds, train_labels);
                training_accuracy_history.push_back(train_acc);
                
                auto val_preds = predict_batch(*val_data);
                float val_acc = compute_accuracy(val_preds, *val_labels);
                validation_accuracy_history.push_back(val_acc);
                
                if (verbose) {
                    std::cout << "Epoch " << epoch + 1 << "/" << epochs
                             << " - loss: " << epoch_loss
                             << " - acc: " << train_acc
                             << " - val_loss: " << val_loss
                             << " - val_acc: " << val_acc << std::endl;
                }
            } else {
                if (verbose) {
                    std::cout << "Epoch " << epoch + 1 << "/" << epochs
                             << " - loss: " << epoch_loss
                             << " - val_loss: " << val_loss << std::endl;
                }
            }
        } else {
            if (verbose) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs
                         << " - loss: " << epoch_loss << std::endl;
            }
        }
    }
}

// Evaluation
float NeuralNetwork::evaluate(const std::vector<std::vector<float>>& test_data,
                             const std::vector<std::vector<float>>& test_labels) {
    assert(test_data.size() == test_labels.size() && "Test data/label size mismatch");
    
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < test_data.size(); ++i) {
        auto prediction = predict(test_data[i]);
        total_loss += compute_loss(prediction, test_labels[i]);
    }
    
    return total_loss / test_data.size();
}

float NeuralNetwork::compute_accuracy(const std::vector<std::vector<float>>& predictions,
                                     const std::vector<std::vector<float>>& targets) {
    assert(predictions.size() == targets.size() && "Prediction/target size mismatch");
    
    int correct = 0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        // For binary classification
        if (predictions[i].size() == 1) {
            int pred_class = predictions[i][0] >= 0.5f ? 1 : 0;
            int true_class = targets[i][0] >= 0.5f ? 1 : 0;
            if (pred_class == true_class) correct++;
        } 
        // For multi-class classification
        else {
            int pred_class = std::max_element(predictions[i].begin(), 
                                             predictions[i].end()) - predictions[i].begin();
            int true_class = std::max_element(targets[i].begin(), 
                                             targets[i].end()) - targets[i].begin();
            if (pred_class == true_class) correct++;
        }
    }
    
    return static_cast<float>(correct) / predictions.size();
}

// Configuration
void NeuralNetwork::set_learning_rate(float lr) {
    for (auto& layer : layers) {
        layer.set_learning_rate(lr);
    }
}

void NeuralNetwork::print_summary() const {
    std::cout << "========================================" << std::endl;
    std::cout << "Neural Network Architecture" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Number of layers: " << layers.size() << std::endl;
    
    int total_params = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        int layer_params = (layer_sizes[i] + 1) * layer_sizes[i + 1];
        total_params += layer_params;
        
        std::cout << "Layer " << i << ": " 
                  << layer_sizes[i] << " -> " << layer_sizes[i + 1]
                  << " (" << layer_params << " parameters)" << std::endl;
    }
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total parameters: " << total_params << std::endl;
    std::cout << "Loss function: ";
    
    switch (loss_function) {
        case LossType::MSE:
            std::cout << "Mean Squared Error" << std::endl;
            break;
        case LossType::MAE:
            std::cout << "Mean Absolute Error" << std::endl;
            break;
        case LossType::BINARY_CROSS_ENTROPY:
            std::cout << "Binary Cross Entropy" << std::endl;
            break;
        case LossType::CATEGORICAL_CROSS_ENTROPY:
            std::cout << "Categorical Cross Entropy" << std::endl;
            break;
        case LossType::HUBER:
            std::cout << "Huber Loss" << std::endl;
            break;
    }
    
    std::cout << "========================================" << std::endl;
}

// Weight initialization
void NeuralNetwork::initialize_he() {
    for (auto& layer : layers) {
        layer.he_init();
    }
}

void NeuralNetwork::initialize_xavier() {
    for (size_t i = 0; i < layers.size(); ++i) {
        int fan_out = (i + 1 < layers.size()) ? layer_sizes[i + 2] : layer_sizes[i + 1];
        layers[i].xavier_init(fan_out);
    }
}

// Serialization
void NeuralNetwork::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving: " + filename);
    }
    
    file << "# Neural Network v1.0\n";
    file << "ARCHITECTURE\n";
    file << layer_sizes.size() << "\n";
    for (int size : layer_sizes) {
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
        throw std::runtime_error("Failed to open file for loading: " + filename);
    }
    
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        
        if (key == "ARCHITECTURE") {
            int num_layers;
            std::getline(file, line);
            num_layers = std::stoi(line);
            
            layer_sizes.clear();
            std::getline(file, line);
            std::istringstream size_stream(line);
            int size;
            while (size_stream >> size) {
                layer_sizes.push_back(size);
            }
        } else if (key == "LOSS_FUNCTION") {
            int loss_int;
            std::getline(file, line);
            loss_int = std::stoi(line);
            loss_function = static_cast<LossType>(loss_int);
        } else if (key == "LAYERS") {
            layers.clear();
            for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                NeuronLayer layer(layer_sizes[i], layer_sizes[i + 1], 
                                 ActivationType::LINEAR); // Temporary
                layer.load(file);
                layers.push_back(layer);
            }
            break;
        }
    }
    
    file.close();
}
