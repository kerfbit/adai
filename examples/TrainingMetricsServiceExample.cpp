/**
 * @file TrainingMetricsServiceExample.cpp
 * @brief Example demonstrating the TrainingMetricsService for real-time metrics tracking
 * 
 * This example shows:
 * 1. Setting up the metrics service
 * 2. Integrating with training loops
 * 3. Polling metrics from another thread
 * 
 * Note: Printing and saving metrics are handled by the metrics-api-server daemon.
 * Query the daemon's REST API (default: http://localhost:8081) to access metrics.
 */

#include "TrainingMetricsService.hpp"
#include "IncrementalTrainer.hpp"
#include "ChatbotTrainer.hpp"
#include "Logger.hpp"
#include <thread>
#include <chrono>
#include <signal.h>

// Global flag for graceful shutdown
std::atomic<bool> running(true);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        adai::Logger::info("Received shutdown signal");
        running = false;
    }
}

/**
 * @brief Example 1: Basic usage with manual metric updates
 */
void example_basic_usage() {
    adai::Logger::info("\n=== Example 1: Basic Usage ===\n");
    
    // Configure metrics service
    MetricsServiceConfig config;
    config.enable_persistence = true;
    config.metrics_file = "training_sessions/example_metrics.jsonl";
    config.summary_file = "training_sessions/example_summary.json";
    config.persist_every_samples = 50;
    config.persist_every_seconds = 10;
    
    TrainingMetricsService service(config);
    
    // Start a training session
    service.start_session(1, 10, 100);  // session_id=1, 10 epochs, 100 samples per epoch
    
    // Simulate training loop
    for (int epoch = 0; epoch < 10; epoch++) {
        service.start_epoch(epoch, 100);
        
        float total_loss = 0.0f;
        
        for (int sample = 1; sample <= 100; sample++) {
            // Simulate training step
            float loss = 2.0f - epoch * 0.15f + (std::rand() % 100) / 1000.0f;
            float grad_norm = 1.0f - epoch * 0.08f + (std::rand() % 50) / 500.0f;
            float lr = 0.001f * (1.0f - epoch / 10.0f);
            
            // Update metrics after each sample
            service.update_sample_metrics(sample, loss, grad_norm, lr);
            
            total_loss += loss;
            
            // Simulate processing time
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Calculate epoch metrics
        float epoch_loss = total_loss / 100.0f;
        float val_loss = epoch_loss * 1.1f;  // Simulate validation loss
        float lr = 0.001f * (1.0f - epoch / 10.0f);
        float perplexity = std::exp(epoch_loss);
        
        service.update_validation_metrics(val_loss);
        service.update_best_metrics(val_loss, epoch);
        service.end_epoch(epoch, epoch_loss, val_loss, lr, perplexity, 0.5f);
        
        adai::Logger::info("Epoch {} completed - Loss: {:.4f}, Val Loss: {:.4f}",
                          epoch, epoch_loss, val_loss);
    }
    
    // End session — the metrics-api-server daemon handles persistence and export
    service.end_session();
}

/**
 * @brief Example 2: Integration with IncrementalTrainer
 */
void example_integration_with_trainer() {
    adai::Logger::info("\n=== Example 2: Integration with IncrementalTrainer ===\n");
    
    // Initialize global metrics service with push enabled so the
    // metrics-api-server daemon handles printing and saving.
    MetricsServiceConfig config;
    config.enable_push = true;
    config.push_url = "http://localhost:8081";
    GlobalMetricsService::initialize(config);
    
    auto& service = GlobalMetricsService::instance();
    
    try {
        // Create trainer
        IncrementalTrainer trainer("vocab.txt");
        
        // Start metrics session
        int session_id = 2;
        service.start_session(session_id, 5, 500);
        
        // Configure trainer to use metrics service via callbacks
        auto& base_config = trainer.get_config().base_config;
        
        // Note: You would modify IncrementalTrainer to accept callbacks
        // and wire them to the metrics service. This is pseudocode:
        /*
        trainer.set_epoch_callback([&](int epoch, int total, float loss, float val_loss, float lr) {
            service.end_epoch(epoch, loss, val_loss, lr);
        });
        
        trainer.set_sample_callback([&](int sample, int total, float loss, float grad_norm, float lr) {
            service.update_sample_metrics(sample, loss, grad_norm, lr);
        });
        */
        
        // Simulate training
        adai::Logger::info("Starting training with metrics service...");
        std::this_thread::sleep_for(std::chrono::seconds(30));
        
        // End session — the metrics-api-server daemon handles saving and output
        service.end_session();
        
    } catch (const std::exception& e) {
        adai::Logger::error("Training error: {}", e.what());
    }
    
    GlobalMetricsService::shutdown();
}

/**
 * @brief Example 3: Querying historical metrics
 */
void example_query_history() {
    adai::Logger::info("\n=== Example 3: Query Historical Metrics ===\n");
    
    TrainingMetricsService service;
    
    // Create multiple sessions
    for (int session = 1; session <= 3; session++) {
        service.start_session(session, 3, 50);
        
        for (int epoch = 0; epoch < 3; epoch++) {
            service.start_epoch(epoch, 50);
            
            for (int sample = 1; sample <= 50; sample++) {
                float loss = 2.0f - (session * 0.2f + epoch * 0.1f);
                service.update_sample_metrics(sample, loss, 0.5f, 0.001f);
            }
            
            service.end_epoch(epoch, 1.5f - epoch * 0.1f, 1.6f - epoch * 0.1f, 0.001f);
        }
        
        service.end_session();
    }
    
    // Query all history
    auto all_history = service.get_history();
    adai::Logger::info("Total historical records: {}", all_history.size());
    
    // Query specific session
    auto session_2_history = service.get_session_history(2);
    adai::Logger::info("Session 2 records: {}", session_2_history.size());
    
    // Get epoch losses
    auto losses = service.get_epoch_losses();
    adai::Logger::info("Epoch losses: {}", losses.size());
    for (size_t i = 0; i < losses.size(); i++) {
        adai::Logger::info("  Epoch {}: {:.4f}", i, losses[i]);
    }
    
    // Get validation losses
    auto val_losses = service.get_epoch_validation_losses();
    adai::Logger::info("Validation losses: {}", val_losses.size());
    for (size_t i = 0; i < val_losses.size(); i++) {
        adai::Logger::info("  Epoch {}: {:.4f}", i, val_losses[i]);
    }
}

int main(int argc, char* argv[]) {
    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    adai::Logger::info("TrainingMetricsService Examples");
    adai::Logger::info("================================\n");
    
    try {
        // Run examples
        example_basic_usage();
        example_query_history();
        
        // Note: example_integration_with_trainer() requires actual training data
        // Uncomment if you have vocab.txt and want to test full integration:
        // example_integration_with_trainer();
        
    } catch (const std::exception& e) {
        adai::Logger::error("Example failed: {}", e.what());
        return 1;
    }
    
    adai::Logger::info("\nAll examples completed successfully!");
    return 0;
}
