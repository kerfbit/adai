#include <iostream>
#include <vector>
#include <chrono>
#include "RewardModel.hpp"
#include "PPOOptimizer.hpp"
#include "LoRA.hpp"
#include "Quantization.hpp"

/**
 * @file Phase5Examples.cpp
 * @brief Comprehensive examples demonstrating Phase 5 advanced features
 * 
 * This file contains complete working examples of:
 * 1. RLHF Training Pipeline
 * 2. LoRA Parameter-Efficient Fine-Tuning
 * 3. Model Quantization
 * 
 * Compile: g++ -std=c++17 -O2 Phase5Examples.cpp -o phase5_examples
 * Run: ./phase5_examples
 */

// Helper function to create dummy encoding data
std::vector<float> create_dummy_encoding(int dim, float base_value) {
    std::vector<float> encoding(dim);
    for (int i = 0; i < dim; i++) {
        encoding[i] = base_value + 0.01f * i;
    }
    return encoding;
}

// =============================================================================
// Example 1: RLHF Training Pipeline
// =============================================================================

void example_rlhf_training() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "Example 1: RLHF Training Pipeline\n";
    std::cout << "============================================================\n\n";
    
    // Step 1: Create Reward Model
    std::cout << "Step 1: Creating reward model...\n";
    int encoding_dim = 768;  // Typical transformer dimension
    // RewardModel expects prompt+response concatenation, so 2x encoding_dim
    RewardModel reward_model(encoding_dim * 2, {512, 256, 128, 1});
    
    // Step 2: Prepare preference data
    std::cout << "Step 2: Preparing preference data...\n";
    std::vector<PreferencePair> preferences;
    
    // Simulate 100 preference pairs
    for (int i = 0; i < 100; i++) {
        auto prompt_enc = create_dummy_encoding(encoding_dim, 0.1f);
        auto chosen_enc = create_dummy_encoding(encoding_dim, 0.8f);   // Good response
        auto rejected_enc = create_dummy_encoding(encoding_dim, 0.2f); // Bad response
        
        preferences.push_back(PreferencePair(prompt_enc, chosen_enc, rejected_enc));
    }
    
    std::cout << "Created " << preferences.size() << " preference pairs\n";
    
    // Step 3: Train reward model
    std::cout << "\nStep 3: Training reward model...\n";
    float learning_rate = 0.001f;
    
    for (int epoch = 0; epoch < 10; epoch++) {
        auto start = std::chrono::high_resolution_clock::now();
        float loss = reward_model.train_on_batch(preferences, learning_rate);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Epoch " << epoch + 1 << "/10 - Loss: " << loss 
                  << " - Time: " << duration.count() << "ms\n";
    }
    
    // Step 4: Save reward model
    std::cout << "\nStep 4: Saving reward model...\n";
    reward_model.save("reward_model_example.bin");
    std::cout << "Saved to: reward_model_example.bin\n";
    
    // Step 5: Test reward prediction
    std::cout << "\nStep 5: Testing reward prediction...\n";
    auto test_prompt = create_dummy_encoding(encoding_dim, 0.5f);
    auto test_good_resp = create_dummy_encoding(encoding_dim, 0.9f);
    auto test_bad_resp = create_dummy_encoding(encoding_dim, 0.1f);
    
    // Concatenate prompt + response
    std::vector<float> test_good = test_prompt;
    test_good.insert(test_good.end(), test_good_resp.begin(), test_good_resp.end());
    
    std::vector<float> test_bad = test_prompt;
    test_bad.insert(test_bad.end(), test_bad_resp.begin(), test_bad_resp.end());
    
    float reward_good = reward_model.predict_reward(test_good);
    float reward_bad = reward_model.predict_reward(test_bad);
    
    std::cout << "Good response reward: " << reward_good << "\n";
    std::cout << "Bad response reward: " << reward_bad << "\n";
    std::cout << "Difference: " << (reward_good - reward_bad) << "\n";
    
    // Step 6: PPO Optimizer Setup
    std::cout << "\nStep 6: Setting up PPO optimizer...\n";
    PPOConfig ppo_config;
    ppo_config.clip_epsilon = 0.2f;
    ppo_config.gamma = 0.99f;
    ppo_config.num_epochs = 4;
    ppo_config.batch_size = 32;
    ppo_config.learning_rate = 1e-5f;
    
    PPOOptimizer ppo(&reward_model, ppo_config, encoding_dim);
    
    std::cout << "PPO Configuration:\n";
    std::cout << "  Clip epsilon: " << ppo_config.clip_epsilon << "\n";
    std::cout << "  Gamma: " << ppo_config.gamma << "\n";
    std::cout << "  Learning rate: " << ppo_config.learning_rate << "\n";
    
    // Step 7: Simulate PPO training iteration
    std::cout << "\nStep 7: Running PPO training iteration...\n";
    Trajectory trajectory;
    
    for (int step = 0; step < 50; step++) {
        auto state = create_dummy_encoding(encoding_dim, 0.5f);
        int action = step % 100;  // Dummy action
        float reward = 0.5f + 0.1f * (step % 5);
        float log_prob = -0.1f * step;
        float value = ppo.estimate_value(state);
        
        trajectory.add_step(state, action, reward, log_prob, value);
    }
    
    float policy_loss = ppo.update(trajectory);
    std::cout << "Policy loss: " << policy_loss << "\n";
    std::cout << "Trajectory length: " << trajectory.length() << " steps\n";
    
    std::cout << "\n✓ RLHF training pipeline complete!\n";
}

// =============================================================================
// Example 2: LoRA Parameter-Efficient Fine-Tuning
// =============================================================================

void example_lora_finetuning() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "Example 2: LoRA Parameter-Efficient Fine-Tuning\n";
    std::cout << "============================================================\n\n";
    
    // Step 1: Setup
    std::cout << "Step 1: Setting up LoRA configuration...\n";
    int d_model = 768;
    int rank = 8;
    float alpha = 16.0f;
    
    LoRAConfig config;
    config.rank = rank;
    config.alpha = alpha;
    config.apply_to_query = true;
    config.apply_to_key = true;
    config.apply_to_value = true;
    config.apply_to_output = true;
    
    std::cout << "LoRA Config:\n";
    std::cout << "  Rank: " << config.rank << "\n";
    std::cout << "  Alpha: " << config.alpha << "\n";
    std::cout << "  Applied to: Q, K, V, O projections\n";
    
    // Step 2: Create LoRA adapters
    std::cout << "\nStep 2: Creating LoRA adapters...\n";
    std::vector<LoRAAdapter> adapters;
    
    adapters.push_back(LoRAAdapter(d_model, d_model, rank, alpha)); // Query
    adapters.push_back(LoRAAdapter(d_model, d_model, rank, alpha)); // Key
    adapters.push_back(LoRAAdapter(d_model, d_model, rank, alpha)); // Value
    adapters.push_back(LoRAAdapter(d_model, d_model, rank, alpha)); // Output
    
    int total_params = 0;
    for (const auto& adapter : adapters) {
        total_params += adapter.num_parameters();
    }
    
    int original_params = 4 * d_model * d_model;
    float reduction = static_cast<float>(original_params) / total_params;
    
    std::cout << "Original parameters: " << original_params << "\n";
    std::cout << "LoRA parameters: " << total_params << "\n";
    std::cout << "Reduction: " << reduction << "x\n";
    
    // Step 3: Simulate forward pass
    std::cout << "\nStep 3: Simulating forward pass with LoRA...\n";
    Matrix input(2, d_model);  // Batch of 2 sequences
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < d_model; j++) {
            input(i, j) = 0.01f * (i + j);
        }
    }
    
    // Simulate frozen weight output
    Matrix frozen_output(2, d_model);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < d_model; j++) {
            frozen_output(i, j) = 0.5f;
        }
    }
    
    // Apply LoRA
    Matrix adapted_output = adapters[0].forward(input, frozen_output);
    std::cout << "Output shape: " << adapted_output.rows << "x" << adapted_output.cols << "\n";
    
    // Step 4: Simulate backward pass
    std::cout << "\nStep 4: Simulating backward pass...\n";
    Matrix grad_output(2, d_model);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < d_model; j++) {
            grad_output(i, j) = 0.001f;
        }
    }
    
    for (auto& adapter : adapters) {
        adapter.backward(input, grad_output);
        adapter.update(0.001f);  // Learning rate
    }
    
    std::cout << "Backward pass complete - gradients computed and applied\n";
    
    // Step 5: Save LoRA adapters
    std::cout << "\nStep 5: Saving LoRA adapters...\n";
    for (size_t i = 0; i < adapters.size(); i++) {
        std::string filename = "lora_adapter_" + std::to_string(i) + ".bin";
        adapters[i].save(filename);
        std::cout << "  Saved: " << filename << "\n";
    }
    
    // Step 6: Demonstrate merging
    std::cout << "\nStep 6: Demonstrating LoRA merging...\n";
    Matrix base_weights(d_model, d_model);
    for (int i = 0; i < d_model; i++) {
        for (int j = 0; j < d_model; j++) {
            base_weights(i, j) = (i == j) ? 1.0f : 0.0f;  // Identity
        }
    }
    
    Matrix merged = adapters[0].merge_with_base(base_weights);
    std::cout << "Merged weights shape: " << merged.rows << "x" << merged.cols << "\n";
    std::cout << "Can now use merged weights for inference (no LoRA overhead)\n";
    
    // Step 7: Statistics
    std::cout << "\nStep 7: LoRA Statistics Summary\n";
    print_lora_statistics(d_model, 12, config);  // 12 layers
    
    std::cout << "\n✓ LoRA fine-tuning example complete!\n";
}

// =============================================================================
// Example 3: Model Quantization
// =============================================================================

void example_quantization() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "Example 3: Model Quantization\n";
    std::cout << "============================================================\n\n";
    
    // Step 1: Create test weight matrix
    std::cout << "Step 1: Creating test weight matrix...\n";
    int rows = 768;
    int cols = 768;
    Matrix weights(rows, cols);
    
    // Initialize with realistic weight distribution
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weights(i, j) = 0.02f * std::sin(i * 0.1f) * std::cos(j * 0.1f);
        }
    }
    
    std::cout << "Weight matrix: " << rows << "x" << cols << "\n";
    std::cout << "FP32 size: " << (rows * cols * sizeof(float)) / 1024.0f << " KB\n";
    
    // Step 2: Test different quantization modes
    std::cout << "\nStep 2: Testing quantization modes...\n\n";
    
    // Prepare calibration data
    std::vector<float> calibration_data;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            calibration_data.push_back(weights(i, j));
        }
    }
    
    std::vector<QuantizationMode> modes = {
        QuantizationMode::SYMMETRIC_INT8,
        QuantizationMode::ASYMMETRIC_INT8,
        QuantizationMode::SYMMETRIC_INT4
    };
    
    std::vector<std::string> mode_names = {
        "Symmetric INT8",
        "Asymmetric INT8",
        "Symmetric INT4"
    };
    
    for (size_t m = 0; m < modes.size(); m++) {
        std::cout << "--- " << mode_names[m] << " ---\n";
        
        Quantizer quantizer(modes[m], CalibrationMethod::MIN_MAX);
        
        QuantizationParams params = quantizer.calibrate(calibration_data);
        
        std::cout << "Quantization params:\n";
        std::cout << "  Scale: " << params.scale << "\n";
        std::cout << "  Zero point: " << params.zero_point << "\n";
        std::cout << "  Range: [" << params.qmin << ", " << params.qmax << "]\n";
        
        // Quantize
        QuantizedMatrix qmat;
        qmat.quantize_from(weights, quantizer);
        
        std::cout << "  Memory reduction: " << qmat.memory_reduction() << "x\n";
        
        // Measure error
        float error = quantizer.compute_quantization_error(calibration_data, params);
        std::cout << "  MSE: " << error << "\n";
        std::cout << "  RMSE: " << std::sqrt(error) << "\n\n";
    }
    
    // Step 3: Detailed analysis with percentile calibration
    std::cout << "Step 3: Percentile calibration analysis...\n";
    
    std::vector<float> test_data = calibration_data;
    // Add outliers
    test_data.push_back(100.0f);
    test_data.push_back(-100.0f);
    
    Quantizer minmax_quantizer(QuantizationMode::SYMMETRIC_INT8,
                               CalibrationMethod::MIN_MAX);
    Quantizer percentile_quantizer(QuantizationMode::SYMMETRIC_INT8,
                                   CalibrationMethod::PERCENTILE,
                                   0.999f);
    
    auto params_minmax = minmax_quantizer.calibrate(test_data);
    auto params_percentile = percentile_quantizer.calibrate(test_data);
    
    std::cout << "\nMin-Max vs Percentile:\n";
    std::cout << "Min-Max scale: " << params_minmax.scale << "\n";
    std::cout << "Percentile scale: " << params_percentile.scale << "\n";
    std::cout << "Improvement: " << (params_minmax.scale / params_percentile.scale) << "x better range utilization\n";
    
    // Step 4: Save and load quantized model
    std::cout << "\nStep 4: Saving quantized model...\n";
    
    Quantizer final_quantizer(QuantizationMode::SYMMETRIC_INT8,
                             CalibrationMethod::PERCENTILE,
                             0.999f);
    
    QuantizedMatrix final_qmat;
    final_qmat.quantize_from(weights, final_quantizer);
    final_qmat.save("quantized_weights_int8.bin");
    
    std::cout << "Saved to: quantized_weights_int8.bin\n";
    
    // Step 5: Load and verify
    std::cout << "\nStep 5: Loading and verifying...\n";
    
    QuantizedMatrix loaded_qmat;
    loaded_qmat.load("quantized_weights_int8.bin");
    
    Matrix reconstructed = loaded_qmat.dequantize(final_quantizer);
    
    std::cout << "Reconstructed shape: " << reconstructed.rows << "x" << reconstructed.cols << "\n";
    
    // Calculate reconstruction error
    float reconstruction_error = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float diff = weights(i, j) - reconstructed(i, j);
            reconstruction_error += diff * diff;
        }
    }
    reconstruction_error = std::sqrt(reconstruction_error / (rows * cols));
    
    std::cout << "Reconstruction RMSE: " << reconstruction_error << "\n";
    
    // Step 6: Performance summary
    std::cout << "\nStep 6: Quantization Summary\n";
    print_quantization_stats(weights, final_qmat, final_quantizer);
    
    std::cout << "\n✓ Model quantization example complete!\n";
}

// =============================================================================
// Main Function
// =============================================================================

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Phase 5 Advanced Features Examples  \n";
    std::cout << "========================================\n";
    
    try {
        // Run all examples
        example_rlhf_training();
        example_lora_finetuning();
        example_quantization();
        
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  All Examples Completed Successfully  \n";
        std::cout << "========================================\n\n";
        
        std::cout << "Generated files:\n";
        std::cout << "  - reward_model_example.bin\n";
        std::cout << "  - lora_adapter_0.bin\n";
        std::cout << "  - lora_adapter_1.bin\n";
        std::cout << "  - lora_adapter_2.bin\n";
        std::cout << "  - lora_adapter_3.bin\n";
        std::cout << "  - quantized_weights_int8.bin\n";
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
