// Example: Using TrainingMetricsService with REST API
// This example demonstrates how to:
// 1. Start the metrics API server
// 2. Update metrics from training loop
// 3. Poll metrics from client applications

#include "TrainingMetricsService.hpp"
#include "TrainingMetricsAPI.hpp"
#include <thread>
#include <chrono>
#include <iostream>

// ============================================================================
// Example 1: Training code with metrics service
// ============================================================================

void training_thread_example(std::shared_ptr<TrainingMetricsService> metrics) {
    // Simulate training session
    const int session_id = 1;
    const int num_epochs = 5;
    const int samples_per_epoch = 1000;
    
    metrics->start_session(session_id, num_epochs, samples_per_epoch);
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        metrics->start_epoch(epoch, samples_per_epoch);
        
        float epoch_loss_sum = 0.0f;
        
        // Simulate training samples
        for (int sample = 0; sample < samples_per_epoch; ++sample) {
            // Simulate training
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // Simulate loss decreasing over time
            float loss = 5.0f / (1.0f + epoch + sample * 0.001f);
            float grad_norm = 2.0f + (rand() % 100) / 100.0f;
            float learning_rate = 0.001f * (1.0f - epoch * 0.1f);
            
            // Update sample metrics
            metrics->update_sample_metrics(sample, loss, grad_norm, learning_rate);
            epoch_loss_sum += loss;
        }
        
        // Calculate epoch metrics
        float epoch_loss = epoch_loss_sum / samples_per_epoch;
        float validation_loss = epoch_loss * 1.1f;  // Slightly higher
        float perplexity = std::exp(epoch_loss);
        float learning_rate = 0.001f * (1.0f - epoch * 0.1f);
        float gradient_norm = 2.0f;
        
        // End epoch with metrics
        metrics->end_epoch(epoch, epoch_loss, validation_loss, learning_rate, 
                          perplexity, gradient_norm);
        
        // Track best metrics
        if (epoch == 0 || validation_loss < metrics->get_current_snapshot().best_validation_loss) {
            metrics->update_best_metrics(validation_loss, epoch);
        }
    }
    
    metrics->end_session();
    std::cout << "Training completed!" << std::endl;
}

// ============================================================================
// Example 2: Starting the REST API server
// ============================================================================

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  Training Metrics API Example" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Configure metrics service
    MetricsServiceConfig config;
    config.enable_persistence = true;
    config.metrics_file = "example_metrics.jsonl";
    config.summary_file = "example_summary.json";
    config.persist_every_samples = 100;
    config.persist_every_seconds = 10;
    
    // Create metrics service
    auto metrics_service = std::make_shared<TrainingMetricsService>(config);
    
    // Create REST API
    int api_port = 8081;
    auto metrics_api = std::make_unique<TrainingMetricsAPI>(metrics_service, api_port);
    
    // Start training in background
    std::thread training_thread(training_thread_example, metrics_service);
    
    std::cout << "\nStarting metrics API server on port " << api_port << std::endl;
    std::cout << "Access metrics at http://localhost:" << api_port << "/api/metrics/current" << std::endl;
    std::cout << "\nAvailable endpoints:" << std::endl;
    std::cout << "  GET /api/metrics/current       - Current snapshot" << std::endl;
    std::cout << "  GET /api/metrics/summary       - Metrics summary" << std::endl;
    std::cout << "  GET /api/metrics/history       - Historical data" << std::endl;
    std::cout << "  GET /api/session/status        - Training status" << std::endl;
    std::cout << "  GET /api/session/epochs        - Epoch metrics" << std::endl;
    std::cout << "\nPress Ctrl+C to stop\n" << std::endl;
    
    // Start API server (blocking)
    metrics_api->start();
    
    // Wait for training to complete
    training_thread.join();
    
    return 0;
}

// ============================================================================
// Example 3: Client code to poll metrics (Python)
// ============================================================================

/*
#!/usr/bin/env python3
import requests
import time
import json

def poll_training_metrics():
    """Poll training metrics and display progress"""
    api_url = "http://localhost:8081"
    
    while True:
        try:
            # Get session status
            response = requests.get(f"{api_url}/api/session/status")
            status = response.json()
            
            if not status['is_training']:
                print("Training not active")
                time.sleep(5)
                continue
            
            # Display progress
            progress = status['progress_percent']
            epoch = status['current_epoch']
            total_epochs = status['total_epochs']
            samples_per_sec = status['samples_per_second']
            time_remaining = status['estimated_time_remaining_seconds']
            
            print(f"\rEpoch {epoch + 1}/{total_epochs} | "
                  f"Progress: {progress:.1f}% | "
                  f"Speed: {samples_per_sec:.1f} samples/s | "
                  f"ETA: {time_remaining:.0f}s", end='')
            
            # Get current metrics
            response = requests.get(f"{api_url}/api/metrics/current")
            metrics = response.json()
            
            if metrics['current_sample'] % 100 == 0:
                print(f"\n  Loss: {metrics['current_loss']:.4f} | "
                      f"Val Loss: {metrics['current_validation_loss']:.4f} | "
                      f"LR: {metrics['current_learning_rate']:.6f}")
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n\nStopped polling")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    poll_training_metrics()
*/

// ============================================================================
// Example 4: Client code to poll metrics (Bash/curl)
// ============================================================================

/*
#!/bin/bash
# Simple polling script using curl

API_URL="http://localhost:8081"

while true; do
    # Get current status
    STATUS=$(curl -s "$API_URL/api/session/status")
    
    # Parse JSON (requires jq)
    IS_TRAINING=$(echo "$STATUS" | jq -r '.is_training')
    
    if [ "$IS_TRAINING" = "false" ]; then
        echo "Training not active"
        sleep 5
        continue
    fi
    
    # Get and display metrics
    EPOCH=$(echo "$STATUS" | jq -r '.current_epoch')
    TOTAL_EPOCHS=$(echo "$STATUS" | jq -r '.total_epochs')
    PROGRESS=$(echo "$STATUS" | jq -r '.progress_percent')
    
    METRICS=$(curl -s "$API_URL/api/metrics/current")
    LOSS=$(echo "$METRICS" | jq -r '.current_loss')
    VAL_LOSS=$(echo "$METRICS" | jq -r '.current_validation_loss')
    
    echo "Epoch $((EPOCH + 1))/$TOTAL_EPOCHS | Progress: $PROGRESS% | Loss: $LOSS | Val Loss: $VAL_LOSS"
    
    sleep 2
done
*/

// ============================================================================
// Example 5: Integrating with existing training code
// ============================================================================

/*
// In your existing training loop:

#include "TrainingMetricsService.hpp"

void train_model() {
    // Create metrics service
    MetricsServiceConfig config;
    auto metrics = std::make_shared<TrainingMetricsService>(config);
    
    // Start session
    metrics->start_session(1, num_epochs, samples_per_epoch);
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        metrics->start_epoch(epoch, samples_per_epoch);
        
        for (int sample = 0; sample < samples_per_epoch; ++sample) {
            // Your existing training code
            float loss = train_single_sample();
            float grad_norm = calculate_gradient_norm();
            
            // Add metrics update
            metrics->update_sample_metrics(sample, loss, grad_norm, learning_rate);
        }
        
        // Calculate validation loss
        float val_loss = validate();
        
        // End epoch
        metrics->end_epoch(epoch, train_loss, val_loss, learning_rate);
    }
    
    metrics->end_session();
}

// In a separate thread or process, start the API server:
void start_metrics_api() {
    auto api = std::make_unique<TrainingMetricsAPI>(
        GlobalMetricsService::instance(),  // Use singleton
        8081
    );
    api->start();  // Blocking
}
*/
