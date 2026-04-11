# Training Metrics Service

A pollable daemon service for real-time tracking and recording of all training metrics.

## Overview

The `TrainingMetricsService` provides a thread-safe, non-blocking interface for monitoring training progress in real-time. It's designed to be polled by monitoring threads, external services, or dashboards without interfering with the training process.

## Features

### Core Capabilities

- **Thread-Safe Polling**: Non-blocking access to current training metrics from any thread
- **Persistent Storage**: Automatic persistence to disk in multiple formats (JSON Lines, JSON summary, Prometheus)
- **Historical Tracking**: Query past training sessions and metrics
- **Multiple Export Formats**: JSON, Prometheus, CSV for integration with various tools
- **Throughput Monitoring**: Automatic calculation of samples/second and ETA
- **Best Checkpoint Tracking**: Automatically tracks best validation loss

### Metrics Tracked

#### Per-Sample Metrics

- Current loss
- Gradient norm
- Learning rate
- Perplexity

#### Per-Epoch Metrics

- Training loss
- Validation loss
- Learning rate schedule
- Epoch duration
- Perplexity
- Gradient norms
- Gradient norm variance (TD-013)
- Compute time ratio (TD-013)
- Weight update ratio (TD-013)
- Activation saturation ratio (TD-013)
- Validation perplexity (TD-015)
- Validation accuracy (TD-015)
- BLEU-1, BLEU-2, BLEU-4 generation quality scores (TD-016)
- ROUGE-1, ROUGE-2, ROUGE-L generation quality scores (TD-016)

#### Session Metrics

- Total samples trained
- Total training time
- Best validation loss and epoch
- Throughput (samples/second)
- Estimated time remaining

## Quick Start

### Basic Usage

```cpp
#include "TrainingMetricsService.hpp"

// Configure the service
MetricsServiceConfig config;
config.enable_persistence = true;
config.metrics_file = "training_sessions/metrics.jsonl";
config.persist_every_samples = 100;

// Create service
TrainingMetricsService service(config);

// Start a training session
service.start_session(session_id, num_epochs, samples_per_epoch);

// In your training loop:
for (int epoch = 0; epoch < num_epochs; epoch++) {
    service.start_epoch(epoch, total_samples);

    for (int sample = 0; sample < total_samples; sample++) {
        // ... training step ...

        // Update metrics
        service.update_sample_metrics(sample, loss, grad_norm, learning_rate);
    }

    // End of epoch
    service.end_epoch(epoch, train_loss, val_loss, learning_rate, perplexity);
}

// End session
service.end_session();
```

### Polling from Another Thread

```cpp
// Background monitoring thread
void monitoring_daemon(TrainingMetricsService& service) {
    while (running) {
        // Sleep between polls
        std::this_thread::sleep_for(std::chrono::seconds(5));

        // Get current snapshot (thread-safe, non-blocking)
        auto snapshot = service.get_current_snapshot();

        if (snapshot.is_training) {
            std::cout << "Epoch: " << snapshot.current_epoch << "/" << snapshot.total_epochs << "\n";
            std::cout << "Loss: " << snapshot.current_loss << "\n";
            std::cout << "Throughput: " << snapshot.samples_per_second << " samples/sec\n";
            std::cout << "ETA: " << snapshot.estimated_time_remaining_seconds << " seconds\n";
        }
    }
}

// Start monitoring in background
std::thread monitor(monitoring_daemon, std::ref(service));
```

### Global Singleton Access

```cpp
// Initialize once
MetricsServiceConfig config;
GlobalMetricsService::initialize(config);

// Access from anywhere
auto& service = GlobalMetricsService::instance();
service.start_session(1, 10, 1000);

// Shutdown when done
GlobalMetricsService::shutdown();
```

## Export Formats

### JSON

Real-time snapshot of current training state:

```json
{
  "session_id": 1,
  "is_training": true,
  "timestamp": "2026-03-06 14:30:45.123",
  "current_epoch": 5,
  "total_epochs": 10,
  "current_sample": 450,
  "total_samples": 1000,
  "current_loss": 1.234567,
  "running_loss": 1.245678,
  "current_validation_loss": 1.345678,
  "current_learning_rate": 0.000950,
  "current_gradient_norm": 0.456789,
  "current_perplexity": 3.456789,
  "best_validation_loss": 1.234567,
  "best_epoch": 4,
  "total_samples_trained": 4450,
  "total_training_time_seconds": 123.456789,
  "samples_per_second": 36.045678,
  "estimated_time_remaining_seconds": 153.789012,
  "gradient_variance": 0.012345,
  "compute_time_ratio": 0.823456,
  "weight_update_ratio": 0.000456,
  "activation_saturation_ratio": 0.1234,
  "current_validation_perplexity": 3.678901,
  "current_validation_accuracy": -1.0,
  "current_bleu4": -1.0,
  "current_rouge1": -1.0,
  "current_rouge2": -1.0,
  "current_rougeL": -1.0
}
```

### JSON Summary

Complete training history with per-epoch arrays:

```json
{
  "session_id": 1,
  "timestamp": "2026-03-06 14:30:45.123",
  "total_epochs_completed": 5,
  "total_samples_trained": 5000,
  "total_training_time_seconds": 150.5,
  "best_validation_loss": 1.234,
  "best_epoch": 4,
  "epoch_losses": [2.5, 2.1, 1.8, 1.5, 1.3],
  "epoch_validation_losses": [2.6, 2.2, 1.9, 1.6, 1.4],
  "epoch_learning_rates": [0.001, 0.00095, 0.0009, 0.00085, 0.0008],
  "epoch_perplexities": [12.18, 8.17, 6.05, 4.48, 3.67],
  "epoch_durations": [30.1, 30.3, 29.9, 30.2, 30.0]
}
```

### Prometheus Format

Compatible with Prometheus monitoring:

```prometheus
# HELP training_loss Current training loss
# TYPE training_loss gauge
training_loss 1.234567 1709738445123

# HELP training_validation_loss Current validation loss
# TYPE training_validation_loss gauge
training_validation_loss 1.345678 1709738445123

# HELP training_samples_per_second Training throughput
# TYPE training_samples_per_second gauge
training_samples_per_second 36.045678 1709738445123
```

### CSV

Simple tabular format for spreadsheet analysis:

```csv
timestamp,session_id,epoch,sample,loss,validation_loss,learning_rate,gradient_norm,perplexity
2026-03-06 14:30:45.123,1,5,450,1.234567,1.345678,0.000950,0.456789,3.456789
```

### JSON Lines (Persistent Storage)

Each line is a complete JSON object for efficient appending and streaming:

```jsonl
{"timestamp":"2026-03-06 14:30:00.000","session_id":1,"epoch":0,"sample":100,"loss":2.5,...}
{"timestamp":"2026-03-06 14:30:05.000","session_id":1,"epoch":0,"sample":200,"loss":2.4,...}
{"timestamp":"2026-03-06 14:30:10.000","session_id":1,"epoch":0,"sample":300,"loss":2.3,...}
```

## Configuration Options

```cpp
struct MetricsServiceConfig {
    // Persistence
    bool enable_persistence = true;
    std::string metrics_file = "training_sessions/metrics.jsonl";
    std::string summary_file = "training_sessions/metrics_summary.json";
    int persist_every_samples = 100;        // Write after N samples
    int persist_every_seconds = 30;         // Or after N seconds

    // Data retention
    int max_records_in_memory = 10000;      // Trim older records
    int max_records_on_disk = 100000;       // Maximum historical records
    bool compress_old_records = false;      // Future: compress old data

    // Monitoring
    bool enable_prometheus_format = false;  // Export Prometheus metrics
    std::string prometheus_file = "training_sessions/metrics.prom";

    // Generation quality (TD-016)
    bool enable_generation_quality = false; // Opt-in: requires generate_response() per sample
    int generation_quality_sample_size = 10; // Validation pairs to sample per epoch
};
```

## Integration with IncrementalTrainer

To integrate the metrics service with `IncrementalTrainer`, wire the callbacks:

```cpp
// In IncrementalTrainer.cpp, add metrics service member:
// std::unique_ptr<TrainingMetricsService> metrics_service_;

// In constructor:
MetricsServiceConfig metrics_config;
metrics_service_ = std::make_unique<TrainingMetricsService>(metrics_config);

// In train_incremental():
metrics_service_->start_session(current_session_id, num_epochs, total_samples);

// Wire ChatbotTrainer callbacks:
trainer.set_epoch_callback([this](int epoch, int total, float loss, float val_loss, float lr) {
    metrics_service_->end_epoch(epoch, loss, val_loss, lr);
});

trainer.set_sample_callback([this](int sample, int total, float loss, float grad_norm, float lr) {
    metrics_service_->update_sample_metrics(sample, loss, grad_norm, lr);
});

// After training:
metrics_service_->end_session();
```

## Monitoring Dashboard Example

Create a simple monitoring script:

```bash
#!/bin/bash
# monitor_training.sh - Poll metrics in real-time

while true; do
    clear
    echo "=== Training Metrics ==="
    echo ""
    cat training_sessions/metrics_summary.json | jq '.current_epoch, .current_loss, .samples_per_second'
    echo ""
    sleep 5
done
```

Or use Python:

```python
import json
import time

while True:
    with open('training_sessions/metrics_summary.json') as f:
        data = json.load(f)

    print(f"Epoch: {data['current_epoch']}/{data['total_epochs']}")
    print(f"Loss: {data['current_loss']:.4f}")
    print(f"Val Loss: {data['current_validation_loss']:.4f}")
    print(f"Throughput: {data['samples_per_second']:.2f} samples/sec")
    print(f"ETA: {data['estimated_time_remaining_seconds']:.0f} seconds")
    print("-" * 40)

    time.sleep(5)
```

## Performance Considerations

- **Thread Safety**: All public methods are protected by mutex for safe concurrent access
- **Non-Blocking**: Polling operations never block the training thread
- **Efficient Persistence**: Writes are batched and only occur at configurable intervals
- **Memory Management**: Automatic trimming of in-memory history to prevent memory growth
- **Minimal Overhead**: Metrics updates are fast O(1) operations with minimal allocation

## Use Cases

1. **Real-Time Monitoring**: Poll metrics from a web dashboard or CLI tool
2. **Distributed Training**: Multiple workers can poll centralized metrics
3. **Early Stopping**: External service can monitor and signal training to stop
4. **Alerting**: Monitor for anomalies and send alerts (high loss, low throughput, etc.)
5. **Visualization**: Feed metrics to TensorBoard, Grafana, or custom dashboards
6. **Experiment Tracking**: Integrate with MLflow, Weights & Biases, etc.
7. **Resource Management**: Adjust batch size or learning rate based on throughput

## Building the Example

The example is included in the `examples/` directory:

```bash
# Build the project
cd build
cmake .. -DBUILD_EXAMPLES=ON
make

# Run the example
./examples/training_metrics_service_example
```

## API Reference

See [TrainingMetricsService.hpp](../src/TrainingMetricsService.hpp) for complete API documentation.

### Full C++ API Catalog

Session Lifecycle

- `start_session(int session_id, int total_epochs = 0, int total_samples = 0)` - Begin tracking a new session
- `end_session()` - Finalize session and persist metrics
- `is_session_active() const` - Check if a session is currently running

Epoch Lifecycle

- `start_epoch(int epoch, int total_samples = 0)` - Begin tracking an epoch
- `end_epoch(int epoch, float loss, float validation_loss, float learning_rate, float perplexity = 0.0f, float gradient_norm = 0.0f)` - Record epoch completion

Real-time Updates

- `update_sample_metrics(int sample, float loss, float gradient_norm, float learning_rate)` - Update per-sample metrics
- `update_validation_metrics(float validation_loss, float validation_accuracy = -1.0f, float validation_perplexity = -1.0f)` - Update validation metrics independently
- `update_best_metrics(float validation_loss, int epoch)` - Record a new best validation loss
- `update_advanced_epoch_metrics(float gradient_variance, float compute_time_ratio, float weight_update_ratio)` - Record per-epoch diagnostic ratios (TD-013)
- `update_activation_saturation(float ratio)` - Record average fraction of near-zero post-GELU units for the epoch; pass `-1.0` to indicate not computed (TD-013)
- `update_generation_quality_metrics(float bleu4, float rouge1, float rouge2, float rougeL)` - Record BLEU/ROUGE generation quality scores for the current epoch; pass `-1.0` for any score not computed (TD-016)

Data Export & Polling (Thread-safe)

- `get_current_snapshot() const` - Thread-safe poll of current state
- `to_json() const` - Export current state as JSON
- `to_json_summary() const` - Export full history as JSON
- `to_prometheus() const` - Export in Prometheus format
- `to_csv_header() const` - Get CSV header string
- `to_csv_row() const` - Get current metrics as CSV row

Historical Queries

- `get_history(int max_records = 1000) const` - Query historical records
- `get_session_history(int session_id) const` - Get history for a specific session
- `get_epoch_losses() const` - Retrieve all epoch training losses
- `get_epoch_validation_losses() const` - Retrieve all epoch validation losses

Control & Configuration

- `flush_to_disk()` - Force immediate persistence of metrics to disk
- `clear_history()` - Clear historical records from memory
- `set_config(const MetricsServiceConfig& config)` - Update active configuration
- `get_config() const` - Retrieve current configuration

### REST API Integration

If building with the optional API server, the service also exposes a full REST catalog. See [TRAINING_METRICS_API.md](TRAINING_METRICS_API.md) for details on endpoints like `/api/metrics/current`, `/api/session/epochs`, and `/api/control/flush`.

## License

Same as the main ADAI project.
