/**
 * @file EnhancedTrainingExample.cpp
 * @brief Example demonstrating the enhanced training pipeline
 *
 * This example shows how to use MetricsTracker for model training, plus a simple best-checkpoint-tracking
 * pattern (TD-168: this used to demonstrate a separate CheckpointManager
 * class, retired once it turned out to be a fully-built, tested duplicate
 * of checkpoint bookkeeping IncrementalTrainer already implements inline —
 * see TECHNICAL_DEBT.md's resolved archive. The inline pattern below is
 * intentionally simple, mirroring the essence of that real implementation
 * rather than reintroducing a second reusable class nothing else calls).
 *
 * TD-315 (October 11, 2026): this used to load and split its sample data with the
 * `Dataset` class (src/Dataset.hpp), retired because no production code used it —
 * see TECHNICAL_DEBT_RESOLVED.md. The data is now a small in-memory list, shuffled
 * and split inline. The real training pipeline loads JSONL through ChatbotTrainer
 * and DatasetRegistry.
 *
 * Features demonstrated:
 * - Simple in-memory train/validation/test split
 * - Metrics tracking with perplexity calculation
 * - Best-checkpoint tracking (simple inline pattern, no rotation)
 * - Early stopping
 * - CSV export for visualization
 *
 * @version 1.2
 * @date October 2026
 */

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "MetricsTracker.hpp"

// (input, response) pair
using Sample = std::pair<std::string, std::string>;

// Mock training functions for demonstration
struct MockModel {
    float loss_decay = 5.0f;
    
    float train_epoch(const std::vector<Sample>& data, int epoch) {
        // Simulate decreasing training loss
        float base_loss = loss_decay * std::exp(-0.15f * epoch);
        float noise = (std::rand() % 100) / 1000.0f;  // Small random noise
        return base_loss + noise;
    }
    
    float validate(const std::vector<Sample>& data, int epoch) {
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

std::vector<Sample> create_sample_data() {
    return {
        {"Hello", "Hi there! How can I help you?"},
        {"How are you?", "I'm doing great, thanks for asking!"},
        {"What's your name?", "I'm an AI assistant designed to help you."},
        {"Tell me a joke", "Why did the scarecrow win an award? He was outstanding in his field!"},
        {"What's the weather like?",
         "I don't have access to real-time weather data, but you can check a weather service."},
        {"How do I learn programming?",
         "Start with the basics, practice regularly, and build small projects."},
        {"Goodbye", "Goodbye! Have a great day!"},
        {"What can you do?", "I can answer questions, help with tasks, and have conversations."},
        {"Tell me about AI", "AI is the simulation of human intelligence by machines."},
        {"What's 2+2?", "2+2 equals 4."},
    };
}

int main() {
    std::srand(42);  // Fixed seed for reproducibility
    
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║   Enhanced Training Pipeline Example               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    // ============================================================
    // 1-2. Create and Split Sample Data
    // ============================================================
    std::cout << "Step 1-2: Creating and splitting sample training data...\n";

    std::vector<Sample> all_data = create_sample_data();
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::shuffle(all_data.begin(), all_data.end(), rng);

    // Split into train/val/test (70%/20%/10%)
    const size_t train_end = all_data.size() * 7 / 10;
    const size_t val_end = train_end + all_data.size() * 2 / 10;
    std::vector<Sample> train_data(all_data.begin(), all_data.begin() + train_end);
    std::vector<Sample> val_data(all_data.begin() + train_end, all_data.begin() + val_end);
    std::vector<Sample> test_data(all_data.begin() + val_end, all_data.end());
    std::cout << "✓ Sample data created and split\n\n";

    std::cout << "Training samples: " << train_data.size() << "\n";
    std::cout << "Validation samples: " << val_data.size() << "\n";
    std::cout << "Test samples: " << test_data.size() << "\n\n";
    
    // ============================================================
    // 3. Initialize Training Infrastructure
    // ============================================================
    std::cout << "Step 3: Initializing training infrastructure...\n";
    
    MockModel model;
    MetricsTracker metrics(3);  // Smoothing window of 3

    // Simple inline best-checkpoint tracking (see this file's own header doc
    // for why this replaced a separate CheckpointManager class).
    const std::string checkpoint_dir = "checkpoints/";
    std::filesystem::create_directories(checkpoint_dir);
    std::string best_checkpoint_path;
    float best_checkpoint_val_loss = std::numeric_limits<float>::max();

    std::cout << "✓ Model initialized\n";
    std::cout << "✓ Metrics tracker initialized (smoothing window: 3)\n";
    std::cout << "✓ Checkpoint directory ready: " << checkpoint_dir << "\n\n";
    
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
        std::shuffle(train_data.begin(), train_data.end(), rng);
        
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
        
        // Save checkpoint and track the best one by validation loss (no
        // rotation here — see this file's own header doc)
        std::ostringstream checkpoint_name;
        checkpoint_name << checkpoint_dir << "checkpoint_epoch_" << std::setw(4)
                       << std::setfill('0') << epoch << ".bin";
        std::string checkpoint_path = checkpoint_name.str();
        model.save(checkpoint_path);
        std::cout << "  Checkpoint: " << checkpoint_path << "\n";
        if (val_loss < best_checkpoint_val_loss) {
            best_checkpoint_val_loss = val_loss;
            best_checkpoint_path = checkpoint_path;
        }

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
    std::cout << "  Best checkpoint: " << best_checkpoint_path << "\n";
    std::cout << "  Best val loss: " << best_checkpoint_val_loss << "\n";

    // ============================================================
    // 7. Load Best Model
    // ============================================================
    std::cout << "\nStep 9: Loading best model...\n";

    if (!best_checkpoint_path.empty()) {
        model.load(best_checkpoint_path);
        std::cout << "✓ Best model loaded from: " << best_checkpoint_path << "\n";
        std::cout << "  Best validation loss: " << best_checkpoint_val_loss << "\n";
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
    std::cout << "  - training_metrics.csv (metrics export)\n";
    std::cout << "  - checkpoints/ (model checkpoints)\n";
    
    std::cout << "\nNext steps:\n";
    std::cout << "  1. Visualize training_metrics.csv with your favorite tool\n";
    std::cout << "  2. Examine checkpoints in checkpoints/ directory\n";
    std::cout << "  3. Integrate with your actual model training code\n\n";
    
    return 0;
}
