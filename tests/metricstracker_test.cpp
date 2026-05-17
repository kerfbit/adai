#include "MetricsTracker.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>

class MetricsTrackerTest : public ::testing::Test {
   protected:
    void TearDown() override {
        // Clean up test files
        std::filesystem::remove("test_metrics.csv");
    }
};

// Test: Constructor
TEST_F(MetricsTrackerTest, Constructor) {
    MetricsTracker tracker;
    EXPECT_EQ(tracker.size(), 0);
    EXPECT_EQ(tracker.get_best_train_epoch(), 0);
    EXPECT_EQ(tracker.get_best_validation_epoch(), 0);
}

// Test: Record single epoch
TEST_F(MetricsTrackerTest, RecordEpoch) {
    MetricsTracker tracker;
    tracker.record_epoch(0, 2.5f, 2.8f, 0.001f, 1.2f, 100);

    EXPECT_EQ(tracker.size(), 1);

    auto metrics = tracker.get_epoch_metrics(0);
    EXPECT_EQ(metrics.epoch, 0);
    EXPECT_FLOAT_EQ(metrics.train_loss, 2.5f);
    EXPECT_FLOAT_EQ(metrics.validation_loss, 2.8f);
    EXPECT_FLOAT_EQ(metrics.learning_rate, 0.001f);
    EXPECT_FLOAT_EQ(metrics.gradient_norm, 1.2f);
    EXPECT_EQ(metrics.duration_seconds, 100);
}

// Test: Perplexity calculation
TEST_F(MetricsTrackerTest, PerplexityCalculation) {
    MetricsTracker tracker;
    tracker.record_epoch(0, 1.0f, 1.5f);

    auto metrics = tracker.get_epoch_metrics(0);
    EXPECT_NEAR(metrics.train_perplexity, std::exp(1.0f), 0.01f);
    EXPECT_NEAR(metrics.validation_perplexity, std::exp(1.5f), 0.01f);
}

// Test: Best train loss tracking
TEST_F(MetricsTrackerTest, BestTrainLossTracking) {
    MetricsTracker tracker;

    tracker.record_epoch(0, 3.0f, 3.5f);
    EXPECT_FLOAT_EQ(tracker.get_best_train_loss(), 3.0f);
    EXPECT_EQ(tracker.get_best_train_epoch(), 0);

    tracker.record_epoch(1, 2.5f, 3.0f);
    EXPECT_FLOAT_EQ(tracker.get_best_train_loss(), 2.5f);
    EXPECT_EQ(tracker.get_best_train_epoch(), 1);

    tracker.record_epoch(2, 2.8f, 2.9f);                   // Higher than best
    EXPECT_FLOAT_EQ(tracker.get_best_train_loss(), 2.5f);  // Should stay at 2.5
    EXPECT_EQ(tracker.get_best_train_epoch(), 1);
}

// Test: Best validation loss tracking
TEST_F(MetricsTrackerTest, BestValidationLossTracking) {
    MetricsTracker tracker;

    tracker.record_epoch(0, 3.0f, 3.5f);
    EXPECT_FLOAT_EQ(tracker.get_best_validation_loss(), 3.5f);
    EXPECT_EQ(tracker.get_best_validation_epoch(), 0);

    tracker.record_epoch(1, 2.5f, 3.0f);
    EXPECT_FLOAT_EQ(tracker.get_best_validation_loss(), 3.0f);
    EXPECT_EQ(tracker.get_best_validation_epoch(), 1);

    tracker.record_epoch(2, 2.2f, 3.2f);                        // Higher val loss
    EXPECT_FLOAT_EQ(tracker.get_best_validation_loss(), 3.0f);  // Should stay
    EXPECT_EQ(tracker.get_best_validation_epoch(), 1);
}

// Test: Multiple epochs
TEST_F(MetricsTrackerTest, MultipleEpochs) {
    MetricsTracker tracker;

    for (int i = 0; i < 10; ++i) {
        float train_loss = 5.0f - i * 0.3f;  // Decreasing
        float val_loss = 5.5f - i * 0.25f;
        tracker.record_epoch(i, train_loss, val_loss, 0.001f, 1.0f, 100);
    }

    EXPECT_EQ(tracker.size(), 10);

    const auto& history = tracker.get_history();
    EXPECT_EQ(history.size(), 10);
    EXPECT_EQ(history[0].epoch, 0);
    EXPECT_EQ(history[9].epoch, 9);
}

// Test: Improvement rate calculation
TEST_F(MetricsTrackerTest, ImprovementRate) {
    MetricsTracker tracker;

    tracker.record_epoch(0, 10.0f);
    tracker.record_epoch(1, 8.0f);
    tracker.record_epoch(2, 6.0f);

    float improvement = tracker.calculate_improvement_rate();
    EXPECT_NEAR(improvement, 40.0f, 0.1f);  // (10 - 6) / 10 * 100 = 40%
}

// Test: Convergence detection
TEST_F(MetricsTrackerTest, ConvergenceDetection) {
    MetricsTracker tracker;

    // Add stable losses (converging)
    for (int i = 0; i < 10; ++i) {
        tracker.record_epoch(i, 2.0f + i * 0.0001f);  // Very small changes
    }

    EXPECT_TRUE(tracker.is_converging(5, 0.001f));
}

// Test: Not converging
TEST_F(MetricsTrackerTest, NotConverging) {
    MetricsTracker tracker;

    // Add unstable losses
    for (int i = 0; i < 10; ++i) {
        float loss = 2.0f + (i % 2 == 0 ? 0.5f : -0.5f);  // Oscillating
        tracker.record_epoch(i, loss);
    }

    EXPECT_FALSE(tracker.is_converging(5, 0.001f));
}

// Test: Overfitting detection
TEST_F(MetricsTrackerTest, OverfittingDetection) {
    MetricsTracker tracker;

    // Train loss much lower than validation loss
    tracker.record_epoch(0, 1.0f, 3.0f);

    EXPECT_TRUE(tracker.is_overfitting(0.5f));
}

// Test: Not overfitting
TEST_F(MetricsTrackerTest, NotOverfitting) {
    MetricsTracker tracker;

    // Similar train and validation loss
    tracker.record_epoch(0, 2.0f, 2.2f);

    EXPECT_FALSE(tracker.is_overfitting(0.5f));
}

// Test: Export to CSV
TEST_F(MetricsTrackerTest, ExportCSV) {
    MetricsTracker tracker;

    tracker.record_epoch(0, 3.0f, 3.5f, 0.001f, 1.2f, 100);
    tracker.record_epoch(1, 2.5f, 3.0f, 0.0009f, 1.1f, 95);
    tracker.record_epoch(2, 2.2f, 2.8f, 0.0008f, 1.0f, 90);

    EXPECT_TRUE(tracker.export_csv("test_metrics.csv"));

    // Verify file exists and has content
    std::ifstream file("test_metrics.csv");
    EXPECT_TRUE(file.is_open());

    std::string line;
    std::getline(file, line);  // Header
    EXPECT_GT(line.length(), 0);

    int line_count = 0;
    while (std::getline(file, line)) {
        line_count++;
    }
    EXPECT_EQ(line_count, 3);  // 3 epochs

    file.close();
}

// Test: Smoothed metrics
TEST_F(MetricsTrackerTest, SmoothedMetrics) {
    MetricsTracker tracker(3);  // Window of 3

    tracker.record_epoch(0, 5.0f, 5.5f);
    tracker.record_epoch(1, 4.0f, 4.5f);
    tracker.record_epoch(2, 3.0f, 3.5f);
    tracker.record_epoch(3, 2.0f, 2.5f);

    const auto& smoothed_train = tracker.get_smoothed_train_loss();
    const auto& smoothed_val = tracker.get_smoothed_validation_loss();

    EXPECT_EQ(smoothed_train.size(), 4);
    EXPECT_EQ(smoothed_val.size(), 4);

    // Last smoothed value should be average of last 3
    float expected = (3.0f + 2.0f + 4.0f) / 3.0f;
    EXPECT_NEAR(smoothed_train[3], expected, 0.01f);
}

// Test: Clear metrics
TEST_F(MetricsTrackerTest, ClearMetrics) {
    MetricsTracker tracker;

    tracker.record_epoch(0, 3.0f, 3.5f);
    tracker.record_epoch(1, 2.5f, 3.0f);

    EXPECT_EQ(tracker.size(), 2);

    tracker.clear();

    EXPECT_EQ(tracker.size(), 0);
    EXPECT_EQ(tracker.get_best_train_loss(), std::numeric_limits<float>::max());
    EXPECT_EQ(tracker.get_best_validation_loss(), std::numeric_limits<float>::max());
}

// Test: Get non-existent epoch
TEST_F(MetricsTrackerTest, GetNonExistentEpoch) {
    MetricsTracker tracker;
    tracker.record_epoch(0, 3.0f);

    auto metrics = tracker.get_epoch_metrics(99);  // Doesn't exist
    EXPECT_EQ(metrics.epoch, 0);                   // Should return empty metrics
    EXPECT_FLOAT_EQ(metrics.train_loss, 0.0f);
}

// Test: Print summary (should not crash)
TEST_F(MetricsTrackerTest, PrintSummary) {
    MetricsTracker tracker;
    tracker.record_epoch(0, 3.0f, 3.5f, 0.001f, 1.2f);

    testing::internal::CaptureStdout();
    tracker.print_summary();
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_GT(output.length(), 0);
}

// Test: Print history (should not crash)
TEST_F(MetricsTrackerTest, PrintHistory) {
    MetricsTracker tracker;
    tracker.record_epoch(0, 3.0f, 3.5f);
    tracker.record_epoch(1, 2.5f, 3.0f);

    testing::internal::CaptureStdout();
    tracker.print_history();
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_GT(output.length(), 0);
}

// Test: Without validation loss
TEST_F(MetricsTrackerTest, WithoutValidationLoss) {
    MetricsTracker tracker;

    tracker.record_epoch(0, 3.0f);  // No validation loss
    tracker.record_epoch(1, 2.5f);

    EXPECT_FLOAT_EQ(tracker.get_best_train_loss(), 2.5f);
    // Best validation loss should remain at max
    EXPECT_EQ(tracker.get_best_validation_loss(), std::numeric_limits<float>::max());
}

// Test: Zero loss handling
TEST_F(MetricsTrackerTest, ZeroLossHandling) {
    MetricsTracker tracker;

    tracker.record_epoch(0, 0.0f, 0.0f);

    auto metrics = tracker.get_epoch_metrics(0);
    EXPECT_FLOAT_EQ(metrics.train_perplexity, 1.0f);       // exp(0) = 1
    EXPECT_FLOAT_EQ(metrics.validation_perplexity, 0.0f);  // Since val_loss is 0, it's not computed
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
