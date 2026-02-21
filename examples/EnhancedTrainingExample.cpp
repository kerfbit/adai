/**
 * @file EnhancedTrainingExample.cpp
 * @brief Example demonstrating the enhanced training pipeline
 * 
 * This example shows how to use Dataset, MetricsTracker, and CheckpointManager
 * for production-ready model training.
 * 
 * Features demonstrated:
 * - Dataset loading and splitting
 * - Metrics tracking with perplexity calculation
 * - Checkpoint management with rotation
 * - Early stopping
 * - CSV export for visualization
 * 
 * @version 1.0
 * @date January 2026
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include "Dataset.hpp"
#include "MetricsTracker.hpp"
#include "CheckpointManager.hpp"

// Mock training functions for demonstration
struct MockModel {
    float loss_decay = 5.0f;
    
    float train_epoch(const std::vector<DataSample>& data, int epoch) {
        // Simulate decreasing training loss
        float base_loss = loss_decay * std::exp(-0.15f * epoch);
        float noise = (std::rand() % 100) / 1000.0f;  // Small random noise
        return base_loss + noise;
    }
    
    float validate(const std::vector<DataSample>& data, int epoch) {
        // Simulate validation loss (slightly higher than training)
        float base_loss = (loss_decay + 0.5f) * std::exp(-0.12f * epoch);
        float noise = (std::rand() % 100) / 1000.0f;
        return base_loss + noise;
    }
    
    void save(const std::string& path) {
        std::ofstream file(path);
        file << "model_weights=dummy_data\n";
        file.close();
    }
    
    void load(const std::string& path) {
        std::cout << "Loading model from: " << path << std::endl;
    }
};

void create_sample_data() {
    std::ofstream file("sample_training_data.txt");
    
    // Create sample conversation data
    file << "INPUT: Hello\n";
    file << "RESPONSE: Hi there! How can I help you?\n\n";
    
    file << "INPUT: How are you?\n";
    file << "RESPONSE: I'm doing great, thanks for asking!\n\n";
    
    file << "INPUT: What's your name?\n";
    file << "RESPONSE: I'm an AI assistant designed to help you.\n\n";
    
    file << "INPUT: Tell me a joke\n";
    file << "RESPONSE: Why did the scarecrow win an award? He was outstanding in his field!\n\n";
    
    file << "INPUT: What's the weather like?\n";
    file << "RESPONSE: I don't have access to real-time weather data, but you can check a weather service.\n\n";
    
    file << "INPUT: How do I learn programming?\n";
    file << "RESPONSE: Start with the basics, practice regularly, and build small projects.\n\n";
    
    file << "INPUT: Goodbye\n";
    file << "RESPONSE: Goodbye! Have a great day!\n\n";
    
    file << "INPUT: What can you do?\n";
    file << "RESPONSE: I can answer questions, help with tasks, and have conversations.\n\n";
    
    file << "INPUT: Tell me about AI\n";
    file << "RESPONSE: AI is the simulation of human intelligence by machines.\n\n";
    
    file << "INPUT: What's 2+2?\n";
    file << "RESPONSE: 2+2 equals 4.\n\n";
    
    file.close();
}

int main() {
    std::srand(42);  // Fixed seed for reproducibility
    
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║   Enhanced Training Pipeline Example               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    // ============================================================
    // 1. Create Sample Data
    // ============================================================
    std::cout << "Step 1: Creating sample training data...\n";
    create_sample_data();
    std::cout << "✓ Sample data created\n\n";
    
    // ============================================================
    // 2. Load and Prepare Dataset
    // ============================================================
    std::cout << "Step 2: Loading and preparing dataset...\n";
    
    Dataset dataset(42);  // Fixed seed for reproducibility
    
    if (!dataset.load_from_file("sample_training_data.txt")) {
        std::cerr << "Error: Could not load dataset\n";
        return 1;
    }
    
    // Split into train/val/test (70%/20%/10%)
    dataset.split(0.7f, 0.2f, 0.1f);
    
    std::cout << "✓ Dataset loaded and split\n";
    dataset.print_stats();
    std::cout << "\n";
    
    // Get splits
    auto train_data = dataset.get_split(SplitType::TRAIN);
    auto val_data = dataset.get_split(SplitType::VALIDATION);
    auto test_data = dataset.get_split(SplitType::TEST);
    
    std::cout << "Training samples: " << train_data.size() << "\n";
    std::cout << "Validation samples: " << val_data.size() << "\n";
    std::cout << "Test samples: " << test_data.size() << "\n\n";
    
    // ============================================================
    // 3. Initialize Training Infrastructure
    // ============================================================
    std::cout << "Step 3: Initializing training infrastructure...\n";
    
    MockModel model;
    MetricsTracker metrics(3);  // Smoothing window of 3
    CheckpointManager checkpoints("checkpoints/", 3);  // Keep 3 best checkpoints
    
    std::cout << "✓ Model initialized\n";
    std::cout << "✓ Metrics tracker initialized (smoothing window: 3)\n";
    std::cout << "✓ Checkpoint manager initialized (max checkpoints: 3)\n\n";
    
    // ============================================================
    // 4. Training Configuration
    // ============================================================
    int num_epochs = 15;
    int patience = 3;
    float min_delta = 0.001f;
    
    std::cout << "Training Configuration:\n";
    std::cout << "  Epochs: " << num_epochs << "\n";
    std::cout << "  Early stopping patience: " << patience << "\n";
    std::cout << "  Min delta: " << min_delta << "\n\n";
    
    // ============================================================
    // 5. Training Loop
    // ============================================================
    std::cout << "Step 4: Starting training...\n";
    std::cout << "════════════════════════════════════════════════════════\n\n";
    
    float best_val_loss = std::numeric_limits<float>::max();
    int epochs_without_improvement = 0;
    bool early_stopped = false;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        auto epoch_start = std::time(nullptr);
        
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << "\n";
        std::cout << "────────────────────────────────────────────────────────\n";
        
        // Shuffle training data
        dataset.shuffle_split(SplitType::TRAIN);
        train_data = dataset.get_split(SplitType::TRAIN);
        
        // Training phase
        std::cout << "Training...\n";
        float train_loss = model.train_epoch(train_data, epoch);
        std::cout << "  Train loss: " << std::fixed << std::setprecision(4) 
                 << train_loss << "\n";
        
        // Validation phase
        std::cout << "Validating...\n";
        float val_loss = model.validate(val_data, epoch);
        std::cout << "  Val loss: " << val_loss << "\n";
        
        // Calculate epoch duration
        auto epoch_duration = std::time(nullptr) - epoch_start;
        
        // Record metrics
        float mock_lr = 0.001f * std::exp(-0.1f * epoch);  // Decaying LR
        float mock_grad_norm = 1.0f + (std::rand() % 50) / 100.0f;
        
        metrics.record_epoch(epoch, train_loss, val_loss, 
                           mock_lr, mock_grad_norm, epoch_duration);
        
        // Save checkpoint
        std::string checkpoint_path = checkpoints.save_checkpoint(
            epoch, train_loss, val_loss
        );
        model.save(checkpoint_path);
        std::cout << "  Checkpoint: " << checkpoint_path << "\n";
        
        // Check for improvement
        if (val_loss < best_val_loss - min_delta) {
            best_val_loss = val_loss;
            epochs_without_improvement = 0;
            std::cout << "  ⭐ New best validation loss!\n";
        } else {
            epochs_without_improvement++;
            std::cout << "  No improvement (" << epochs_without_improvement 
                     << "/" << patience << ")\n";
        }
        
        // Check convergence
        if (metrics.is_converging()) {
            std::cout << "  ✓ Training is converging\n";
        }
        
        // Check overfitting
        if (metrics.is_overfitting(0.5f)) {
            std::cout << "  ⚠ Possible overfitting detected!\n";
        }
        
        std::cout << "\n";
        
        // Early stopping check
        if (epochs_without_improvement >= patience) {
            std::cout << "🛑 Early stopping triggered!\n";
            std::cout << "   No improvement for " << patience << " consecutive epochs\n\n";
            early_stopped = true;
            break;
        }
    }
    
    // ============================================================
    // 6. Training Summary
    // ============================================================
    std::cout << "════════════════════════════════════════════════════════\n";
    std::cout << "Training Complete!\n";
    std::cout << "════════════════════════════════════════════════════════\n\n";
    
    if (early_stopped) {
        std::cout << "Status: Early stopped after " << metrics.size() << " epochs\n\n";
    } else {
        std::cout << "Status: Completed all " << num_epochs << " epochs\n\n";
    }
    
    // Print metrics summary
    std::cout << "Step 5: Metrics Summary\n";
    metrics.print_summary();
    
    // Print full history
    std::cout << "\nStep 6: Full Training History\n";
    metrics.print_history();
    
    // Export metrics to CSV
    std::cout << "\nStep 7: Exporting metrics...\n";
    if (metrics.export_csv("training_metrics.csv")) {
        std::cout << "✓ Metrics exported to: training_metrics.csv\n";
        std::cout << "  Use this file to visualize training curves\n";
    }
    
    // Checkpoint summary
    std::cout << "\nStep 8: Checkpoint Summary\n";
    checkpoints.print_summary();
    
    // ============================================================
    // 7. Load Best Model
    // ============================================================
    std::cout << "\nStep 9: Loading best model...\n";
    
    std::string best_checkpoint = checkpoints.get_best_checkpoint_path();
    if (!best_checkpoint.empty()) {
        model.load(best_checkpoint);
        std::cout << "✓ Best model loaded from: " << best_checkpoint << "\n";
        std::cout << "  Best validation loss: " 
                 << checkpoints.get_best_validation_loss() << "\n";
    } else {
        std::cout << "⚠ No best checkpoint found\n";
    }
    
    // ============================================================
    // 8. Analysis
    // ============================================================
    std::cout << "\n════════════════════════════════════════════════════════\n";
    std::cout << "Analysis\n";
    std::cout << "════════════════════════════════════════════════════════\n";
    
    float improvement = metrics.calculate_improvement_rate();
    std::cout << "Overall improvement: " << std::fixed << std::setprecision(2) 
             << improvement << "%\n";
    
    std::cout << "Best train loss: " << std::setprecision(4) 
             << metrics.get_best_train_loss() 
             << " (epoch " << metrics.get_best_train_epoch() << ")\n";
    
    std::cout << "Best validation loss: " 
             << metrics.get_best_validation_loss() 
             << " (epoch " << metrics.get_best_validation_epoch() << ")\n";
    
    // ============================================================
    // 9. Cleanup
    // ============================================================
    std::cout << "\n════════════════════════════════════════════════════════\n";
    std::cout << "Example complete!\n";
    std::cout << "════════════════════════════════════════════════════════\n";
    
    std::cout << "\nGenerated files:\n";
    std::cout << "  - sample_training_data.txt (training data)\n";
    std::cout << "  - training_metrics.csv (metrics export)\n";
    std::cout << "  - checkpoints/ (model checkpoints)\n";
    
    std::cout << "\nNext steps:\n";
    std::cout << "  1. Visualize training_metrics.csv with your favorite tool\n";
    std::cout << "  2. Examine checkpoints in checkpoints/ directory\n";
    std::cout << "  3. Integrate with your actual model training code\n\n";
    
    return 0;
}
