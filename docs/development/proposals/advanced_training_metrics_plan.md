# Planning Document: Advanced Training Metrics

## 1. Overview

The goal of this initiative is to enhance the existing `TrainingMetricsService` and related systems by introducing advanced training diagnostics. These metrics provide deeper visibility into throughput efficiency, optimization health, generative quality, and automated outlier detection for the training dataset.

## 2. Selected Metrics

Based on recent evaluations, the following metrics will be added to the training pipeline:

### 2.1. Throughput & Efficiency

- **Batch Padding Efficiency:** Track the ratio of actual tokens to padding tokens in training batches to monitor batch packing effectiveness.
- **Compute vs. Data Time:** Differentiate the time spent on reading/processing data vs. performing forward/backward passes.

### 2.2. Outlier Detection (Dataset Quality)

- **Abnormal Sample Tagging:** Identify and tag data samples that yield abnormally large loss or gradient norm values. This enables manual review and potential deletion/repair of poor-quality items from the dataset.

### 2.3. Optimization Health

- **Weight Update Magnitude Ratio:** Track the ratio of parameter updates to current weight magnitudes, which assists in dynamically tuning and verifying the learning rate.
- **Gradient Variance:** Track the variance of gradients across batches/steps to ensure stable updates.
- **Activation Saturation (Percentage):** Monitor the percentage of inactive/dead neurons or saturation points (especially in FeedForward layers) during the forward pass.

### 2.4. Generation Quality

- **BLEU Score:** Calculate the BLEU score evaluated against target responses during validation to measure n-gram overlap.
- **ROUGE Score:** Compute ROUGE metrics on a validation subset to track sequence generation accuracy.
- **Attention Entropy:** Measure how focused (sharp) or unfocused (uniform) the attention weights are across epochs.

---

## 3. Implementation Steps

### Phase 1: Core Definitions and Extensibility

1. **Extend Metrics Data Structures:**
   - Update `TrainingMetricsSnapshot` and `PersistentMetricsRecord` in `src/TrainingMetricsService.hpp` with new float/double fields (`batch_padding_efficiency`, `compute_time_ratio`, `weight_update_ratio`, `gradient_variance`, `activation_saturation_pct`, `attention_entropy`, `bleu_score`, `rouge_score`).
   - Create a new structure `AbnormalSample` containing `epoch`, `sample_id`, `input_text`, `target_text`, `loss`, `grad_norm`, and `reason`.

### Phase 2: Throughput and Outliers

1. **Batch Efficiency Tracking:**
   - Extract padding efficiency from `BatchStats` (via `compute_batch_stats`) and feed it into the sample callbacks.
2. **Compute vs Data Tracking:**
   - Use `std::chrono` inside `ChatbotTrainer::train_epoch` to measure data fetching vs. `model->forward`/`backward` execution times.
3. **Outlier Catching:**
   - Implement `check_abnormal_metrics(loss, grad_norm)` inside the training loop. If a threshold is exceeded, route the sample to the `TrainingMetricsService` and output to `training_sessions/abnormal_samples.json`.

### Phase 3: Optimizations and Activations Hooks

1. **Optimizer Metrics:**
   - Modify the `Optimizer` interface (e.g., `AdamW`) to calculate and optionally return the weight update ratio (update L2 norm / weight L2 norm) and running gradient variance.
2. **Activation Saturation:**
   - Augment `FeedForward` layer and `EncoderDecoderModel` to track and report percentage of zeroed/dead activations.

### Phase 4: Generative Metrics hooks

1. **Generation Scores (BLEU/ROUGE):**
   - Implement text generation (on a sub-sample of validation data) within the `validate()` phase.
   - Introduce utility functions `compute_bleu_score` and `compute_rouge_score` comparing generated tokens to target text.
2. **Attention Entropy:**
   - Expose the attention weight distributions computed in `MultiHeadAttention` to compute their entropy, aggregating it inside `EncoderDecoderModel` for logging.

### Phase 5: API & UI Exposure

1. **REST API Expansion:**
   - Expose the newly added snapshots and histories in `TrainingMetricsAPI`.
   - Add endpoint `GET /api/metrics/abnormal` to retrieve tagged outlier samples.
2. **Dashboard Integration (Tizen / Web):**
   - Bind the REST outputs to the frontend visualizations.

## 4. Risks & Considerations

- **Performance Impact:** Running BLEU/ROUGE on validation data can be exceptionally slow. This should be run on a limited randomly sampled subset of the validation set (e.g., 5-10%).
- **Memory Footprint:** Tracking attention entropy and activation saturation across large models requires careful memory accumulation so as not to stall training or OOM.
