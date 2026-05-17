#ifndef REWARD_MODEL_HPP
#define REWARD_MODEL_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "Matrix.hpp"

/**
 * @file RewardModel.hpp
 * @brief Reward Model for Reinforcement Learning from Human Feedback (RLHF)
 *
 * The RewardModel learns to predict human preferences between response pairs.
 * It's trained on preference data where humans indicate which of two responses
 * they prefer for a given prompt. The model outputs a scalar reward score.
 *
 * Architecture:
 * - Input: Encoded sequence representations from base model
 * - Multiple fully-connected layers with activations
 * - Output: Single scalar reward value
 *
 * Training:
 * - Uses Bradley-Terry preference model
 * - Loss: -log(sigmoid(r_chosen - r_rejected))
 * - Optimized to rank preferred responses higher
 *
 * @version 1.0
 * @date January 2026
 */

/**
 * @brief Preference pair for RLHF training
 *
 * Contains a prompt and two responses (chosen vs rejected) as rated by humans.
 */
struct PreferencePair {
    std::vector<float> prompt_encoding;    // Encoded prompt representation
    std::vector<float> chosen_encoding;    // Encoding of preferred response
    std::vector<float> rejected_encoding;  // Encoding of rejected response

    PreferencePair(const std::vector<float>& prompt, const std::vector<float>& chosen,
                   const std::vector<float>& rejected)
        : prompt_encoding(prompt), chosen_encoding(chosen), rejected_encoding(rejected) {}
};

/**
 * @class RewardModel
 * @brief Neural network that learns to predict human preference rewards
 *
 * The reward model is a key component of RLHF pipelines. It learns to score
 * responses based on human preference data, then guides the policy model
 * during PPO training.
 *
 * Example Usage:
 * @code
 * // Create reward model
 * RewardModel reward_model(768, {512, 256, 128, 1});
 *
 * // Train on preference pairs
 * std::vector<PreferencePair> pairs = load_preferences();
 * for (int epoch = 0; epoch < 10; epoch++) {
 *     float loss = reward_model.train_on_batch(pairs, 0.001);
 *     std::cout << "Loss: " << loss << std::endl;
 * }
 *
 * // Score a response
 * std::vector<float> encoding = encode_response(prompt, response);
 * float reward = reward_model.predict_reward(encoding);
 * @endcode
 */
class RewardModel {
   private:
    int input_dim_;                           // Input feature dimension
    std::vector<int> layer_dims_;             // Dimensions of hidden layers
    std::vector<Matrix> weights_;             // Weight matrices for each layer
    std::vector<std::vector<float>> biases_;  // Bias vectors for each layer
    std::vector<Matrix> activations_;         // Cached activations (forward pass)
    std::vector<Matrix> pre_activations_;     // Cached pre-activation values

    /**
     * @brief Apply ReLU activation function
     */
    float relu(float x) const {
        return std::max(0.0f, x);
    }

    /**
     * @brief ReLU derivative
     */
    float relu_derivative(float x) const {
        return x > 0.0f ? 1.0f : 0.0f;
    }

    /**
     * @brief Apply sigmoid activation
     */
    float sigmoid(float x) const {
        return 1.0f / (1.0f + std::exp(-x));
    }

    /**
     * @brief Sigmoid derivative
     */
    float sigmoid_derivative(float x) const {
        float s = sigmoid(x);
        return s * (1.0f - s);
    }

    /**
     * @brief Initialize weights with He initialization
     */
    void initialize_weights() {
        weights_.clear();
        biases_.clear();

        int prev_dim = input_dim_;
        for (size_t i = 0; i < layer_dims_.size(); i++) {
            int curr_dim = layer_dims_[i];

            // He initialization: sqrt(2 / fan_in)
            float std_dev = std::sqrt(2.0f / prev_dim);
            Matrix w(prev_dim, curr_dim);
            for (int r = 0; r < prev_dim; r++) {
                for (int c = 0; c < curr_dim; c++) {
                    w(r, c) = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_dev;
                }
            }
            weights_.push_back(w);

            // Initialize biases to zero
            std::vector<float> b(curr_dim, 0.0f);
            biases_.push_back(b);

            prev_dim = curr_dim;
        }
    }

   public:
    /**
     * @brief Construct a reward model
     *
     * @param input_dim Dimension of input encodings
     * @param layer_dims Dimensions of hidden layers (last should be 1 for scalar output)
     */
    RewardModel(int input_dim, const std::vector<int>& layer_dims)
        : input_dim_(input_dim), layer_dims_(layer_dims) {
        if (layer_dims.empty()) {
            throw std::invalid_argument("Must specify at least one layer");
        }
        if (layer_dims.back() != 1) {
            throw std::invalid_argument("Final layer must have dimension 1 for scalar reward");
        }

        initialize_weights();
    }

    /**
     * @brief Forward pass to compute reward score
     *
     * @param input Input encoding vector
     * @return Scalar reward value
     */
    float forward(const std::vector<float>& input) {
        if ((int)input.size() != input_dim_) {
            throw std::invalid_argument("Input dimension mismatch");
        }

        // Clear cached values
        activations_.clear();
        pre_activations_.clear();

        // Convert input to Matrix
        Matrix activation(1, input_dim_);
        for (int i = 0; i < input_dim_; i++) {
            activation(0, i) = input[i];
        }
        activations_.push_back(activation);

        // Forward through layers
        for (size_t i = 0; i < weights_.size(); i++) {
            // Linear: z = xW + b
            Matrix z = activation * weights_[i];
            for (int j = 0; j < z.cols; j++) {
                z(0, j) += biases_[i][j];
            }
            pre_activations_.push_back(z);

            // Activation (ReLU for hidden, linear for output)
            Matrix act(1, z.cols);
            if (i < weights_.size() - 1) {
                // Hidden layers: ReLU
                for (int j = 0; j < z.cols; j++) {
                    act(0, j) = relu(z(0, j));
                }
            } else {
                // Output layer: linear (no activation)
                act = z;
            }

            activations_.push_back(act);
            activation = act;
        }

        // Return scalar reward
        return activations_.back()(0, 0);
    }

    /**
     * @brief Predict reward for a single encoding
     *
     * @param encoding Input encoding vector
     * @return Predicted reward score
     */
    float predict_reward(const std::vector<float>& encoding) {
        return forward(encoding);
    }

    /**
     * @brief Compute Bradley-Terry loss for preference pair
     *
     * Loss = -log(sigmoid(r_chosen - r_rejected))
     *
     * @param pair Preference pair with chosen and rejected responses
     * @return Loss value
     */
    float compute_loss(const PreferencePair& pair) {
        // Concatenate prompt + response for full context
        std::vector<float> chosen_input = pair.prompt_encoding;
        chosen_input.insert(chosen_input.end(), pair.chosen_encoding.begin(),
                            pair.chosen_encoding.end());

        std::vector<float> rejected_input = pair.prompt_encoding;
        rejected_input.insert(rejected_input.end(), pair.rejected_encoding.begin(),
                              pair.rejected_encoding.end());

        // Get rewards
        float r_chosen = forward(chosen_input);
        float r_rejected = forward(rejected_input);

        // Bradley-Terry loss
        float logit = r_chosen - r_rejected;
        float loss = -std::log(sigmoid(logit) + 1e-8f);  // Add epsilon for stability

        return loss;
    }

    /**
     * @brief Train on a batch of preference pairs
     *
     * @param pairs Vector of preference pairs
     * @param learning_rate Learning rate for gradient descent
     * @return Average loss across batch
     */
    float train_on_batch(const std::vector<PreferencePair>& pairs, float learning_rate) {
        if (pairs.empty()) {
            throw std::invalid_argument("Cannot train on empty batch");
        }

        float total_loss = 0.0f;

        // Accumulate gradients
        std::vector<Matrix> weight_grads;
        std::vector<std::vector<float>> bias_grads;

        for (size_t i = 0; i < weights_.size(); i++) {
            weight_grads.push_back(Matrix(weights_[i].rows, weights_[i].cols));
            bias_grads.push_back(std::vector<float>(biases_[i].size(), 0.0f));
        }

        // Process each pair
        for (const auto& pair : pairs) {
            // Concatenate inputs
            std::vector<float> chosen_input = pair.prompt_encoding;
            chosen_input.insert(chosen_input.end(), pair.chosen_encoding.begin(),
                                pair.chosen_encoding.end());

            std::vector<float> rejected_input = pair.prompt_encoding;
            rejected_input.insert(rejected_input.end(), pair.rejected_encoding.begin(),
                                  pair.rejected_encoding.end());

            // Forward passes
            float r_chosen = forward(chosen_input);
            float r_rejected = forward(rejected_input);

            // Compute loss
            float logit = r_chosen - r_rejected;
            float prob = sigmoid(logit);
            total_loss += -std::log(prob + 1e-8f);

            // Backward pass gradient: d_loss/d_logit = -(1 - sigmoid(logit))
            float grad_logit = -(1.0f - prob);

            // Backprop through chosen (gradient flows with +1 coefficient)
            backward(chosen_input, grad_logit, weight_grads, bias_grads);

            // Backprop through rejected (gradient flows with -1 coefficient)
            backward(rejected_input, -grad_logit, weight_grads, bias_grads);
        }

        // Average gradients and update weights
        float batch_size = static_cast<float>(pairs.size());
        for (size_t i = 0; i < weights_.size(); i++) {
            for (int r = 0; r < weights_[i].rows; r++) {
                for (int c = 0; c < weights_[i].cols; c++) {
                    weights_[i](r, c) -= learning_rate * weight_grads[i](r, c) / batch_size;
                }
            }
            for (size_t j = 0; j < biases_[i].size(); j++) {
                biases_[i][j] -= learning_rate * bias_grads[i][j] / batch_size;
            }
        }

        return total_loss / batch_size;
    }

    /**
     * @brief Backward pass to accumulate gradients
     *
     * @param input Input that was used in forward pass
     * @param output_grad Gradient flowing back from loss
     * @param weight_grads Accumulated weight gradients
     * @param bias_grads Accumulated bias gradients
     */
    void backward(const std::vector<float>& input, float output_grad,
                  std::vector<Matrix>& weight_grads, std::vector<std::vector<float>>& bias_grads) {
        // Recompute forward pass to get activations
        forward(input);

        // Start with gradient at output
        Matrix grad(1, 1);
        grad(0, 0) = output_grad;

        // Backprop through layers in reverse
        for (int i = weights_.size() - 1; i >= 0; i--) {
            // Gradient w.r.t pre-activation
            Matrix grad_pre_act(1, pre_activations_[i].cols);
            if (i < (int)weights_.size() - 1) {
                // Hidden layer: apply ReLU derivative
                for (int j = 0; j < grad.cols; j++) {
                    grad_pre_act(0, j) = grad(0, j) * relu_derivative(pre_activations_[i](0, j));
                }
            } else {
                // Output layer: linear (derivative = 1)
                grad_pre_act = grad;
            }

            // Gradient w.r.t weights: dL/dW = activation^T * grad
            Matrix activation_T = activations_[i].transpose();
            Matrix dW = activation_T * grad_pre_act;

            // Accumulate weight gradients
            for (int r = 0; r < dW.rows; r++) {
                for (int c = 0; c < dW.cols; c++) {
                    weight_grads[i](r, c) += dW(r, c);
                }
            }

            // Accumulate bias gradients
            for (int j = 0; j < grad_pre_act.cols; j++) {
                bias_grads[i][j] += grad_pre_act(0, j);
            }

            // Gradient w.r.t input: grad * W^T
            Matrix weights_T = weights_[i].transpose();
            grad = grad_pre_act * weights_T;
        }
    }

    /**
     * @brief Save reward model to file
     *
     * @param filepath Path to save file
     */
    void save(const std::string& filepath) const {
        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + filepath);
        }

        // Write dimensions
        file.write(reinterpret_cast<const char*>(&input_dim_), sizeof(int));
        int num_layers = layer_dims_.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
        file.write(reinterpret_cast<const char*>(layer_dims_.data()), num_layers * sizeof(int));

        // Write weights and biases
        for (size_t i = 0; i < weights_.size(); i++) {
            int rows = weights_[i].rows;
            int cols = weights_[i].cols;
            file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
            file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    float val = weights_[i](r, c);
                    file.write(reinterpret_cast<const char*>(&val), sizeof(float));
                }
            }

            int bias_size = biases_[i].size();
            file.write(reinterpret_cast<const char*>(&bias_size), sizeof(int));
            file.write(reinterpret_cast<const char*>(biases_[i].data()), bias_size * sizeof(float));
        }
    }

    /**
     * @brief Load reward model from file
     *
     * @param filepath Path to load file
     */
    void load(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + filepath);
        }

        // Read dimensions
        file.read(reinterpret_cast<char*>(&input_dim_), sizeof(int));
        int num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
        layer_dims_.resize(num_layers);
        file.read(reinterpret_cast<char*>(layer_dims_.data()), num_layers * sizeof(int));

        // Read weights and biases
        weights_.clear();
        biases_.clear();

        for (int i = 0; i < num_layers; i++) {
            int rows, cols;
            file.read(reinterpret_cast<char*>(&rows), sizeof(int));
            file.read(reinterpret_cast<char*>(&cols), sizeof(int));

            Matrix w(rows, cols);
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    float val;
                    file.read(reinterpret_cast<char*>(&val), sizeof(float));
                    w(r, c) = val;
                }
            }
            weights_.push_back(w);

            int bias_size;
            file.read(reinterpret_cast<char*>(&bias_size), sizeof(int));
            std::vector<float> b(bias_size);
            file.read(reinterpret_cast<char*>(b.data()), bias_size * sizeof(float));
            biases_.push_back(b);
        }
    }

    /**
     * @brief Get input dimension
     */
    int get_input_dim() const {
        return input_dim_;
    }

    /**
     * @brief Get layer dimensions
     */
    const std::vector<int>& get_layer_dims() const {
        return layer_dims_;
    }
};

#endif  // REWARD_MODEL_HPP
