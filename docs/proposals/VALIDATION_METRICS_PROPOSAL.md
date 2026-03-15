# Proposal: Adding Metrics to Epoch Validation Segment

## Objective
To track, record, and expose key performance indicators (KPIs) during the validation segment of training epochs. This will provide better insights into model generalization, help detect over-fitting early, and enable more comprehensive monitoring on the dashboard.

## Background
Currently, the training pipeline effectively tracks metrics during the training passes (forward/backward passes, loss calculation). However, the validation phase—which evaluates the model against a held-out dataset at the end of an epoch or at specified step intervals—needs more granular and systematized metric tracking to be properly integrated into the existing `TrainingMetricsService` and dashboard.

## Proposed Changes

1. **Extend Metrics Data Structures**:
   - Introduce validation-specific metric fields (e.g., `val_loss`, `val_accuracy`, `val_perplexity`) in the core metric tracking objects.
   - Differentiate between iteration-level training metrics and epoch-level (or evaluation-interval-level) validation metrics.

2. **Integration with `TrainingMetricsService`**:
   - Add new endpoints or update existing ones (e.g., in `metrics-api-server`) to accept validation metric payloads.
   - Ensure the database or in-memory storage used by the metrics service correctly stores the validation metrics tied to their corresponding epoch and global step.

3. **Validation Loop Hooks**:
   - Update the model's training loop to accumulate loss and accuracy during the validation pass.
   - Emit the aggregated metrics to the `TrainingMetricsService` at the end of the validation segment.

4. **Dashboard Updates**:
   - Update `dashboard.html` / `serve_dashboard.py` to fetch validation metrics.
   - Plot validation loss and training loss on the same charts for easy visual comparison of model generalization.

## Implementation Details

* **Core C++ Changes**: 
  Modify the `Optimizer` or `TrainingMetrics` classes to handle validation states. Add a `record_validation_metrics(epoch, step, metrics_dict)` function.
* **API Changes**:
  Update JSON schemas in the metric API to support `"phase": "training" | "validation"`.
* **UI**:
  Use Chart.js (or existing dashboard charting library) to plot validation curves with distinct colors and markers.

## Benefits
* **Early Stopping**: Paves the way for implementing automated early stopping based on `val_loss`.
* **Model Health Visibility**: Clear visual indicators of over-fitting (e.g., training loss decreasing while validation loss increases).
* **Experiment Tracking**: Better historical comparisons between different hyperparameter tuning runs.

## Alternatives Considered
* *Logging to stdout/files only*: Simpler, but lacks the real-time visualization and historical tracking benefits of integrating with the existing `TrainingMetricsService` API.
* *Separate Validation Service*: Overkill. The existing metrics service is well-equipped to handle validation metrics simply by adding a phase/context tag.