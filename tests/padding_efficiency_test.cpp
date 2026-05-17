/**
 * @file padding_efficiency_test.cpp
 * @brief Unit tests for batch padding efficiency tracking in TrainingMetricsService.
 *
 * Tests cover:
 *  - update_padding_efficiency() stores the value in the snapshot
 *  - end_epoch() pushes the current value into epoch_padding_efficiencies
 *  - to_json() emits "current_padding_efficiency"
 *  - Efficiency computation logic (window-level and epoch average)
 *  - Edge cases: single sample, window_size=1, all identical lengths
 */

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include "TrainingMetricsService.hpp"

// ============================================================================
// Helper: create a service with persistence disabled
// ============================================================================
static MetricsServiceConfig no_persist_config() {
    MetricsServiceConfig cfg;
    cfg.enable_persistence = false;
    cfg.enable_push = false;
    cfg.enable_prometheus_format = false;
    return cfg;
}

// ============================================================================
// update_padding_efficiency
// ============================================================================

TEST(PaddingEfficiency, UpdateStoresCurrentValue) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 2, 100);
    svc.start_epoch(1, 100);

    svc.update_padding_efficiency(0.75f);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_padding_efficiency, 0.75f);
}

TEST(PaddingEfficiency, DefaultIsNegativeOne) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_padding_efficiency, -1.0f);
}

TEST(PaddingEfficiency, UpdateToOnePointZero) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);

    svc.update_padding_efficiency(1.0f);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_padding_efficiency, 1.0f);
}

TEST(PaddingEfficiency, UpdateNotComputedMarker) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);

    // -1 is the "not computed" sentinel
    svc.update_padding_efficiency(-1.0f);

    auto snap = svc.get_current_snapshot();
    EXPECT_FLOAT_EQ(snap.current_padding_efficiency, -1.0f);
}

// ============================================================================
// end_epoch pushes value into per-epoch history
// ============================================================================

TEST(PaddingEfficiency, EndEpochPushesHistory) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 3, 30);

    svc.start_epoch(1, 10);
    svc.update_padding_efficiency(0.80f);
    svc.end_epoch(1, 1.0f, 1.1f, 0.001f);

    auto snap = svc.get_current_snapshot();
    ASSERT_EQ(snap.epoch_padding_efficiencies.size(), 1u);
    EXPECT_FLOAT_EQ(snap.epoch_padding_efficiencies[0], 0.80f);
}

TEST(PaddingEfficiency, EndEpochAccumulatesAcrossEpochs) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 3, 30);

    svc.start_epoch(1, 10);
    svc.update_padding_efficiency(0.80f);
    svc.end_epoch(1, 1.0f, 1.1f, 0.001f);

    svc.start_epoch(2, 10);
    svc.update_padding_efficiency(0.90f);
    svc.end_epoch(2, 0.9f, 1.0f, 0.0009f);

    svc.start_epoch(3, 10);
    svc.update_padding_efficiency(0.95f);
    svc.end_epoch(3, 0.8f, 0.9f, 0.0008f);

    auto snap = svc.get_current_snapshot();
    ASSERT_EQ(snap.epoch_padding_efficiencies.size(), 3u);
    EXPECT_FLOAT_EQ(snap.epoch_padding_efficiencies[0], 0.80f);
    EXPECT_FLOAT_EQ(snap.epoch_padding_efficiencies[1], 0.90f);
    EXPECT_FLOAT_EQ(snap.epoch_padding_efficiencies[2], 0.95f);
}

TEST(PaddingEfficiency, NotComputedSentinelPreservedInHistory) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);
    // Never call update_padding_efficiency — stays at -1
    svc.end_epoch(1, 1.0f, 1.1f, 0.001f);

    auto snap = svc.get_current_snapshot();
    ASSERT_EQ(snap.epoch_padding_efficiencies.size(), 1u);
    EXPECT_FLOAT_EQ(snap.epoch_padding_efficiencies[0], -1.0f);
}

// ============================================================================
// to_json emits current_padding_efficiency
// ============================================================================

TEST(PaddingEfficiency, ToJsonContainsField) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);
    svc.update_padding_efficiency(0.88f);

    std::string json = svc.to_json();
    EXPECT_NE(json.find("\"current_padding_efficiency\""), std::string::npos);
}

TEST(PaddingEfficiency, ToJsonValueIsCorrect) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);
    svc.start_epoch(1, 10);
    svc.update_padding_efficiency(0.50f);

    std::string json = svc.to_json();
    // Field should appear with value approximately 0.5
    EXPECT_NE(json.find("\"current_padding_efficiency\": 0.500000"), std::string::npos);
}

TEST(PaddingEfficiency, ToJsonDefaultSentinel) {
    TrainingMetricsService svc(no_persist_config());
    svc.start_session(1, 1, 10);

    std::string json = svc.to_json();
    // Default -1.0 sentinel present
    EXPECT_NE(json.find("\"current_padding_efficiency\": -1.000000"), std::string::npos);
}

// ============================================================================
// Window efficiency computation logic (pure arithmetic, no service required)
// ============================================================================

// Helper that mirrors the arithmetic in ChatbotTrainer::train_epoch()
static float compute_window_efficiency(const std::vector<int>& input_lengths,
                                       const std::vector<int>& target_lengths) {
    if (input_lengths.empty())
        return -1.0f;
    int actual = 0;
    int max_in = 0;
    int max_tgt = 0;
    for (size_t i = 0; i < input_lengths.size(); ++i) {
        actual += input_lengths[i] + target_lengths[i];
        if (input_lengths[i] > max_in)
            max_in = input_lengths[i];
        if (target_lengths[i] > max_tgt)
            max_tgt = target_lengths[i];
    }
    int padded = (max_in + max_tgt) * static_cast<int>(input_lengths.size());
    if (padded == 0)
        return 1.0f;
    return static_cast<float>(actual) / static_cast<float>(padded);
}

TEST(PaddingEfficiencyComputation, IdenticalLengthsIsOne) {
    // All sequences the same length → no wasted padding → efficiency = 1.0
    std::vector<int> in_lens = {10, 10, 10, 10};
    std::vector<int> tgt_lens = {8, 8, 8, 8};
    float eff = compute_window_efficiency(in_lens, tgt_lens);
    EXPECT_FLOAT_EQ(eff, 1.0f);
}

TEST(PaddingEfficiencyComputation, OneTokenVsMany) {
    // One long sequence and one short sequence in a window of 2
    // input_lengths = {1, 10},  target_lengths = {1, 10}
    // actual = (1+1) + (10+10) = 22
    // padded = (10 + 10) * 2 = 40
    // efficiency = 22/40 = 0.55
    std::vector<int> in_lens = {1, 10};
    std::vector<int> tgt_lens = {1, 10};
    float eff = compute_window_efficiency(in_lens, tgt_lens);
    EXPECT_NEAR(eff, 0.55f, 1e-5f);
}

TEST(PaddingEfficiencyComputation, SingleSampleWindowIsOne) {
    // A window with only one sample is trivially 100% efficient
    std::vector<int> in_lens = {7};
    std::vector<int> tgt_lens = {5};
    float eff = compute_window_efficiency(in_lens, tgt_lens);
    EXPECT_FLOAT_EQ(eff, 1.0f);
}

TEST(PaddingEfficiencyComputation, WorstCaseSingleLong) {
    // Window of 4 samples: one has length 100, rest have length 1
    // actual = (100+100) + 3*(1+1) = 200 + 6 = 206
    // padded = (100 + 100) * 4 = 800
    // efficiency = 206/800 = 0.2575
    std::vector<int> in_lens = {100, 1, 1, 1};
    std::vector<int> tgt_lens = {100, 1, 1, 1};
    float eff = compute_window_efficiency(in_lens, tgt_lens);
    EXPECT_NEAR(eff, 206.0f / 800.0f, 1e-5f);
}

TEST(PaddingEfficiencyComputation, EmptyWindowReturnsSentinel) {
    std::vector<int> in_lens = {};
    std::vector<int> tgt_lens = {};
    float eff = compute_window_efficiency(in_lens, tgt_lens);
    EXPECT_FLOAT_EQ(eff, -1.0f);
}

TEST(PaddingEfficiencyComputation, EpochAverageOfTwoWindows) {
    // Simulate averaging two window efficiencies
    float eff1 = compute_window_efficiency({10, 10}, {8, 8});  // 1.0
    float eff2 = compute_window_efficiency({1, 10}, {1, 10});  // 0.55
    float avg = (eff1 + eff2) / 2.0f;
    EXPECT_NEAR(avg, 0.775f, 1e-5f);
}
