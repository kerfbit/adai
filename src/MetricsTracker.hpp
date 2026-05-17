#ifndef METRICS_TRACKER_HPP
#define METRICS_TRACKER_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

/**
 * @file MetricsTracker.hpp
 * @brief Training metrics tracking and analysis
 * 
 * Provides comprehensive metrics tracking for training:
 * - Loss tracking (train/validation)
 * - Perplexity calculation
 * - Learning rate history
 * - Gradient norms
 * - Metric curves and trends
 * - CSV export for visualization
 * 
 * @version 1.0
 * @date January 2026
 */

/**
 * @brief Epoch metrics snapshot
 */
struct EpochMetrics {
    int epoch;
    float train_loss;
    float validation_loss;
    float train_perplexity;
    float validation_perplexity;
    float learning_rate;
    float gradient_norm;
    long duration_seconds;
    
    EpochMetrics()
        : epoch(0), train_loss(0.0f), validation_loss(0.0f),
          train_perplexity(0.0f), validation_perplexity(0.0f),
          learning_rate(0.0f), gradient_norm(0.0f), duration_seconds(0) {}
};

/**
 * @brief Training metrics tracker
 * 
 * Tracks and analyzes training metrics over time:
 * - Per-epoch metrics (loss, perplexity, LR, gradient norm)
 * - Best metrics tracking
 * - Trend analysis
 * - Export to CSV for visualization
 * 
 * Example usage:
 * @code
 * MetricsTracker tracker;
 * 
 * for (int epoch = 0; epoch < num_epochs; epoch++) {
 *     float train_loss = train_epoch();
 *     float val_loss = validate();
 *     
 *     tracker.record_epoch(epoch, train_loss, val_loss, 
 *                         learning_rate, gradient_norm, duration);
 *     tracker.print_summary();
 * }
 * 
 * tracker.export_csv("training_metrics.csv");
 * @endcode
 */
class MetricsTracker {
private:
    std::vector<EpochMetrics> history_;
    
    // Best metrics
    float best_train_loss_;
    int best_train_epoch_;
    float best_validation_loss_;
    int best_validation_epoch_;
    float best_train_perplexity_;
    float best_validation_perplexity_;
    
    // Moving averages for smoothing
    std::vector<float> train_loss_smoothed_;
    std::vector<float> validation_loss_smoothed_;
    int smoothing_window_;
    
    /**
     * @brief Calculate perplexity from loss
     * Perplexity = exp(loss)
     */
    float calculate_perplexity(float loss) const {
        if (loss < 0.0f) return 0.0f;
        return std::exp(loss);
    }
    
    /**
     * @brief Calculate moving average
     */
    float calculate_moving_average(const std::vector<float>& values, size_t window) const {
        if (values.empty()) return 0.0f;
        
        size_t start = values.size() > window ? values.size() - window : 0;
        float sum = std::accumulate(values.begin() + start, values.end(), 0.0f);
        return sum / (values.size() - start);
    }
    
    /**
     * @brief Update smoothed metrics
     */
    void update_smoothed_metrics() {
        std::vector<float> train_losses;
        std::vector<float> val_losses;
        
        for (const auto& metrics : history_) {
            train_losses.push_back(metrics.train_loss);
            if (metrics.validation_loss > 0.0f) {
                val_losses.push_back(metrics.validation_loss);
            }
        }
        
        train_loss_smoothed_.clear();
        validation_loss_smoothed_.clear();
        
        for (size_t i = 0; i < train_losses.size(); ++i) {
            size_t start = i >= smoothing_window_ ? i - smoothing_window_ + 1 : 0;
            float sum = 0.0f;
            for (size_t j = start; j <= i; ++j) {
                sum += train_losses[j];
            }
            train_loss_smoothed_.push_back(sum / (i - start + 1));
        }
        
        for (size_t i = 0; i < val_losses.size(); ++i) {
            size_t start = i >= smoothing_window_ ? i - smoothing_window_ + 1 : 0;
            float sum = 0.0f;
            for (size_t j = start; j <= i; ++j) {
                sum += val_losses[j];
            }
            validation_loss_smoothed_.push_back(sum / (i - start + 1));
        }
    }
    
public:
    /**
     * @brief Constructor
     * @param smoothing_window Window size for moving average (default: 3)
     */
    MetricsTracker(int smoothing_window = 3)
        : best_train_loss_(std::numeric_limits<float>::max()),
          best_train_epoch_(0),
          best_validation_loss_(std::numeric_limits<float>::max()),
          best_validation_epoch_(0),
          best_train_perplexity_(std::numeric_limits<float>::max()),
          best_validation_perplexity_(std::numeric_limits<float>::max()),
          smoothing_window_(smoothing_window) {}
    
    /**
     * @brief Record metrics for an epoch
     * 
     * @param epoch Epoch number
     * @param train_loss Training loss
     * @param validation_loss Validation loss (0.0 if not computed)
     * @param learning_rate Current learning rate
     * @param gradient_norm Gradient norm
     * @param duration_seconds Epoch duration in seconds
     */
    void record_epoch(int epoch, float train_loss, float validation_loss = 0.0f,
                     float learning_rate = 0.0f, float gradient_norm = 0.0f,
                     long duration_seconds = 0) {
        EpochMetrics metrics;
        metrics.epoch = epoch;
        metrics.train_loss = train_loss;
        metrics.validation_loss = validation_loss;
        metrics.train_perplexity = calculate_perplexity(train_loss);
        metrics.validation_perplexity = validation_loss > 0.0f ? 
            calculate_perplexity(validation_loss) : 0.0f;
        metrics.learning_rate = learning_rate;
        metrics.gradient_norm = gradient_norm;
        metrics.duration_seconds = duration_seconds;
        
        history_.push_back(metrics);
        
        // Update best metrics
        if (train_loss < best_train_loss_) {
            best_train_loss_ = train_loss;
            best_train_epoch_ = epoch;
            best_train_perplexity_ = metrics.train_perplexity;
        }
        
        if (validation_loss > 0.0f && validation_loss < best_validation_loss_) {
            best_validation_loss_ = validation_loss;
            best_validation_epoch_ = epoch;
            best_validation_perplexity_ = metrics.validation_perplexity;
        }
        
        // Update smoothed metrics
        update_smoothed_metrics();
    }
    
    /**
     * @brief Get metrics for a specific epoch
     * @param epoch Epoch number
     * @return Epoch metrics (or empty metrics if not found)
     */
    EpochMetrics get_epoch_metrics(int epoch) const {
        for (const auto& metrics : history_) {
            if (metrics.epoch == epoch) {
                return metrics;
            }
        }
        return EpochMetrics();
    }
    
    /**
     * @brief Get all metrics history
     * @return Vector of epoch metrics
     */
    const std::vector<EpochMetrics>& get_history() const {
        return history_;
    }
    
    /**
     * @brief Get best training loss
     * @return Best training loss value
     */
    float get_best_train_loss() const {
        return best_train_loss_;
    }
    
    /**
     * @brief Get best validation loss
     * @return Best validation loss value
     */
    float get_best_validation_loss() const {
        return best_validation_loss_;
    }
    
    /**
     * @brief Get epoch with best training loss
     * @return Epoch number
     */
    int get_best_train_epoch() const {
        return best_train_epoch_;
    }
    
    /**
     * @brief Get epoch with best validation loss
     * @return Epoch number
     */
    int get_best_validation_epoch() const {
        return best_validation_epoch_;
    }
    
    /**
     * @brief Get smoothed training loss curve
     * @return Vector of smoothed training losses
     */
    const std::vector<float>& get_smoothed_train_loss() const {
        return train_loss_smoothed_;
    }
    
    /**
     * @brief Get smoothed validation loss curve
     * @return Vector of smoothed validation losses
     */
    const std::vector<float>& get_smoothed_validation_loss() const {
        return validation_loss_smoothed_;
    }
    
    /**
     * @brief Calculate improvement rate
     * @return Percentage improvement from first to last epoch
     */
    float calculate_improvement_rate() const {
        if (history_.size() < 2) return 0.0f;
        
        float first_loss = history_.front().train_loss;
        float last_loss = history_.back().train_loss;
        
        if (first_loss == 0.0f) return 0.0f;
        
        return ((first_loss - last_loss) / first_loss) * 100.0f;
    }
    
    /**
     * @brief Check if training is converging
     * @param window Number of recent epochs to check
     * @param threshold Maximum change threshold
     * @return True if loss change is below threshold
     */
    bool is_converging(size_t window = 5, float threshold = 0.001f) const {
        if (history_.size() < window + 1) return false;
        
        std::vector<float> recent_losses;
        for (size_t i = history_.size() - window; i < history_.size(); ++i) {
            recent_losses.push_back(history_[i].train_loss);
        }
        
        float mean = std::accumulate(recent_losses.begin(), recent_losses.end(), 0.0f) / window;
        
        float variance = 0.0f;
        for (float loss : recent_losses) {
            variance += (loss - mean) * (loss - mean);
        }
        variance /= window;
        
        float std_dev = std::sqrt(variance);
        
        return std_dev < threshold;
    }
    
    /**
     * @brief Detect overfitting
     * @param gap_threshold Threshold for train/val loss gap
     * @return True if validation loss is significantly higher than training loss
     */
    bool is_overfitting(float gap_threshold = 0.5f) const {
        if (history_.empty()) return false;
        
        const auto& latest = history_.back();
        if (latest.validation_loss == 0.0f) return false;
        
        float gap = latest.validation_loss - latest.train_loss;
        return gap > gap_threshold;
    }
    
    /**
     * @brief Export metrics to CSV file
     * @param filepath Output CSV file path
     * @return True if successful
     */
    bool export_csv(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << filepath << std::endl;
            return false;
        }
        
        // Write header
        file << "epoch,train_loss,validation_loss,train_perplexity,validation_perplexity,"
             << "learning_rate,gradient_norm,duration_seconds\n";
        
        // Write data
        for (const auto& metrics : history_) {
            file << metrics.epoch << ","
                 << metrics.train_loss << ","
                 << metrics.validation_loss << ","
                 << metrics.train_perplexity << ","
                 << metrics.validation_perplexity << ","
                 << metrics.learning_rate << ","
                 << metrics.gradient_norm << ","
                 << metrics.duration_seconds << "\n";
        }
        
        file.close();
        return true;
    }
    
    /**
     * @brief Print current metrics summary
     */
    void print_summary() const {
        if (history_.empty()) {
            std::cout << "No metrics recorded yet.\n";
            return;
        }
        
        const auto& latest = history_.back();
        
        std::cout << "\n╔══════════════════════════════════════════════╗\n";
        std::cout << "║         Training Metrics Summary            ║\n";
        std::cout << "╚══════════════════════════════════════════════╝\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Current Epoch: " << latest.epoch << "\n";
        std::cout << "  Train Loss: " << latest.train_loss 
                 << " (Perplexity: " << latest.train_perplexity << ")\n";
        
        if (latest.validation_loss > 0.0f) {
            std::cout << "  Val Loss: " << latest.validation_loss 
                     << " (Perplexity: " << latest.validation_perplexity << ")\n";
        }
        
        std::cout << "  Learning Rate: " << latest.learning_rate << "\n";
        std::cout << "  Gradient Norm: " << latest.gradient_norm << "\n";
        
        std::cout << "\n  Best Metrics:\n";
        std::cout << "    Best Train Loss: " << best_train_loss_ 
                 << " (Epoch " << best_train_epoch_ << ")\n";
        
        if (best_validation_loss_ < std::numeric_limits<float>::max()) {
            std::cout << "    Best Val Loss: " << best_validation_loss_ 
                     << " (Epoch " << best_validation_epoch_ << ")\n";
        }
        
        if (history_.size() > 1) {
            float improvement = calculate_improvement_rate();
            std::cout << "\n  Improvement: " << improvement << "%\n";
            
            if (is_converging()) {
                std::cout << "  Status: Converging ✓\n";
            }
            
            if (is_overfitting()) {
                std::cout << "  Warning: Possible overfitting detected!\n";
            }
        }
        
        std::cout << "══════════════════════════════════════════════\n\n";
    }
    
    /**
     * @brief Print detailed training history
     */
    void print_history() const {
        std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║              Training History                           ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(6) << "Epoch" 
                 << std::setw(12) << "Train Loss"
                 << std::setw(12) << "Val Loss"
                 << std::setw(12) << "Train PPL"
                 << std::setw(12) << "Val PPL"
                 << std::setw(10) << "LR"
                 << std::setw(10) << "Grad\n";
        std::cout << std::string(74, '-') << "\n";
        
        for (const auto& metrics : history_) {
            std::cout << std::setw(6) << metrics.epoch
                     << std::setw(12) << metrics.train_loss
                     << std::setw(12) << (metrics.validation_loss > 0.0f ? 
                                          std::to_string(metrics.validation_loss) : "N/A")
                     << std::setw(12) << metrics.train_perplexity
                     << std::setw(12) << (metrics.validation_perplexity > 0.0f ?
                                          std::to_string(metrics.validation_perplexity) : "N/A")
                     << std::setw(10) << metrics.learning_rate
                     << std::setw(10) << metrics.gradient_norm << "\n";
        }
        
        std::cout << "══════════════════════════════════════════════════════════\n\n";
    }
    
    /**
     * @brief Clear all metrics
     */
    void clear() {
        history_.clear();
        train_loss_smoothed_.clear();
        validation_loss_smoothed_.clear();
        best_train_loss_ = std::numeric_limits<float>::max();
        best_train_epoch_ = 0;
        best_validation_loss_ = std::numeric_limits<float>::max();
        best_validation_epoch_ = 0;
        best_train_perplexity_ = std::numeric_limits<float>::max();
        best_validation_perplexity_ = std::numeric_limits<float>::max();
    }
    
    /**
     * @brief Get number of recorded epochs
     * @return Number of epochs
     */
    size_t size() const {
        return history_.size();
    }
};

#endif  // METRICS_TRACKER_HPP
