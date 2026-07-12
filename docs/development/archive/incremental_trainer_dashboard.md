# Proposal: Incremental Trainer Dashboard and Enhanced Logging

**Status:** Proposed
**Date:** March 1, 2026
**Author:** GitHub Copilot
**Related Issues:** TD-004

## 1. Summary

This document proposes a significant upgrade to the `IncrementalTrainer`'s user interface and observability. The current command-line output is a static report at the end of training. This proposal outlines a plan to:

1. **Integrate Structured Logging:** Replace all `std::cout` statements with the new `Logger` utility for structured, configurable, and persistent logging of training events.
2. **Implement a Dynamic CLI Dashboard:** Transform the static CLI report into a real-time, dynamic dashboard that updates after each epoch, providing rich insights into the training process.
3. **Expand Tracked Metrics:** Enhance the `TrainingSession` data structure to track more granular per-epoch metrics, including epoch duration and estimated time to completion (ETA).

These changes will dramatically improve the developer experience, making it easier to monitor, debug, and analyze training runs.

## 2. Motivation

The current `IncrementalTrainer` provides limited visibility during a training run. A developer has to wait until the entire session is complete to see the results, and the logs are unstructured `std::cout` messages. This makes it difficult to:

* Detect overfitting or other training problems as they happen.
* Estimate how long a training run will take.
* Analyze the performance of different learning rates or other hyperparameters over time.
* Correlate training events with other system logs.
* Programmatically parse training output.

By implementing a dynamic dashboard and structured logging, we can provide the visibility and data needed to train models more effectively. This work builds upon the foundation laid in **TD-004: Enhanced Metrics Tracking for Training Sessions**.

## 3. Proposed Changes

### 3.1. Enhanced Metrics Tracking

To power the new dashboard, we will first expand the metrics tracked in `IncrementalTrainer.hpp`.

The `TrainingSession` struct will be updated to include:

```cpp
// In include/IncrementalTrainer.hpp

struct TrainingSession {
    // ... existing fields ...
    std::vector<double> per_epoch_losses;
    std::vector<double> per_epoch_validation_losses;
    std::vector<double> learning_rates;
    std::vector<double> training_time_per_epoch; // in seconds
};
```

The `IncrementalTrainer` class will have new member variables to track timing:

```cpp
// In include/IncrementalTrainer.hpp

class IncrementalTrainer {
    // ...
private:
    // ... existing members ...
    std::chrono::steady_clock::time_point session_start_time;
    std::chrono::steady_clock::time_point epoch_start_time;
};
```

### 3.2. Structured Logging Integration

All `std::cout` and `std::cerr` statements within `src/IncrementalTrainer.cpp` will be replaced with calls to the `Logger` singleton.

Examples:

* Session start/end events will be logged at the `info` level.

    ```cpp
    Logger::info("Starting incremental training session for {} epochs.", config.epochs);
    ```

* Epoch start/end will be logged at the `info` or `debug` level.

    ```cpp
    Logger::debug("Starting epoch {}/{}", epoch + 1, config.epochs);
    ```

* Data loading and preprocessing steps.

    ```cpp
    Logger::info("Creating QA pairs from {} text sources.", text_files.size());
    ```

* Errors and warnings will use `Logger::error` and `Logger::warn`.

    ```cpp
    Logger::error("Failed to save session history to {}", history_path.string());
    ```

This change provides consistent, timestamped, and level-filtered logs that can be written to both the console and a file, as configured in the new logging system.

### 3.3. Dynamic CLI Dashboard

The core of this proposal is to replace the `print_training_summary()` method with a dynamic dashboard that updates in-place after each epoch.

The dashboard will be rendered to the console and will look something like this:

```text
[ADAI Incremental Training] - Session In Progress...

╭──────────────────────────────────────────────────────────────────────────────╮
│ Session ID: 20260301-143015   Epoch: 5/10 [50%]   ETA: 4m 32s                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Epoch Stats:                                                                │
│    - Duration:         54.3s                                                 │
│    - Loss:             1.2345 (↓ -0.1)                                       │
│    - Validation Loss:  1.5678 (↑ +0.05)                                      │
│    - Learning Rate:    0.0001                                                │
│                                                                              │
│  Overall Stats:                                                              │
│    - Elapsed Time:     4m 28s                                                │
│    - Avg Epoch Time:   53.6s                                                 │
│    - Best Val Loss:    1.5178 (Epoch 3)                                      │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

Epoch 5/10 [█████████████████████████████▋....................]  75% - Loss: 1.23
```

Implementation Details:

1. **`display_dashboard()` Method:** A new private method `display_dashboard(const TrainingSession& session, int current_epoch)` will be created.
2. **In-place Updates:** This method will use ANSI escape codes (`\033[F` to move up lines, `\033[K` to clear lines) to clear the previous dashboard output and redraw it with new data after each epoch. This creates a smooth, non-scrolling display.
3. **ETA Calculation:** The Estimated Time to Completion (ETA) will be calculated based on the average time per epoch and the number of remaining epochs.
4. **Progress Bar:** A simple progress bar will show the progress within the current epoch's batch processing.
5. **Final Summary:** After training completes, the dashboard will remain on screen with a "Session Complete" status.

## 4. Implementation Plan

1. **[TD-004] Extend `TrainingSession`:** Modify `IncrementalTrainer.hpp` to add the new per-epoch metric vectors as described in section 3.1 and in TD-004.
2. **Add Timing Members:** Add `session_start_time` and `epoch_start_time` members to the `IncrementalTrainer` class.
3. **Integrate Logging:** Replace all `std::cout` calls in `src/IncrementalTrainer.cpp` with appropriate `Logger` calls.
4. **Update `train_incremental()`:**
    * At the start of the method, record `session_start_time`.
    * Inside the epoch loop, record `epoch_start_time`.
    * At the end of each epoch, calculate the duration and push all per-epoch metrics to the `TrainingSession` vectors.
    * Call the new `display_dashboard()` method.
5. **Create `display_dashboard()`:** Implement the dashboard rendering logic using ANSI escape codes for a dynamic, in-place updating view.
6. **Update Tests:** Extend `tests/test_incremental_trainer.cpp` to verify that the new per-epoch metrics are correctly saved and loaded.

## 5. Benefits

* **Real-time Insight:** Immediately see how training is progressing without waiting for completion.
* **Improved Debugging:** Quickly spot issues like exploding gradients, slow convergence, or overfitting.
* **Better Planning:** The ETA provides a reliable estimate of how long training will take.
* **Professional Polish:** A dynamic dashboard provides a much more professional and user-friendly experience.
* **Structured Observability:** Integration with the new logging framework enables robust, machine-parseable logs for production monitoring and analysis.
