/**
 * BatchProcessor Unit Tests
 *
 * Comprehensive test suite for BatchProcessor utilities - critical components
 * for efficient transformer batch processing.
 *
 * Tests cover:
 * - TokenBatch structure and methods
 * - Batch creation with padding
 * - Dynamic batching by sequence length
 * - Padding mask generation
 * - Unbatching outputs
 * - Batch statistics computation
 * - Edge cases and error handling
 */

#include "BatchProcessor.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

class BatchProcessorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Create test sequences of varying lengths
        test_sequences_ = {
            {1, 2, 3},        // length 3
            {4, 5, 6, 7, 8},  // length 5
            {9, 10},          // length 2
            {11, 12, 13, 14}  // length 4
        };
    }

    std::vector<std::vector<int>> test_sequences_;
};

// ============================================================================
// TokenBatch Structure Tests
// ============================================================================

TEST_F(BatchProcessorTest, TokenBatchBatchSize) {
    TokenBatch batch;
    batch.batch_token_ids = {{1, 2}, {3, 4}, {5, 6}};

    EXPECT_EQ(batch.batch_size(), 3);
}

TEST_F(BatchProcessorTest, TokenBatchIsEmpty) {
    TokenBatch empty_batch;
    EXPECT_TRUE(empty_batch.is_empty());

    TokenBatch non_empty_batch;
    non_empty_batch.batch_token_ids = {{1, 2}};
    EXPECT_FALSE(non_empty_batch.is_empty());
}

// ============================================================================
// Basic Batch Creation Tests
// ============================================================================

TEST_F(BatchProcessorTest, CreateBatchBasic) {
    TokenBatch batch = create_batch(test_sequences_, 0);

    // Check batch properties
    EXPECT_EQ(batch.batch_size(), 4);
    EXPECT_EQ(batch.max_length, 5);  // Longest sequence is length 5
    EXPECT_EQ(batch.pad_token_id, 0);

    // Check lengths are preserved
    EXPECT_EQ(batch.lengths[0], 3);
    EXPECT_EQ(batch.lengths[1], 5);
    EXPECT_EQ(batch.lengths[2], 2);
    EXPECT_EQ(batch.lengths[3], 4);
}

TEST_F(BatchProcessorTest, CreateBatchPaddingCorrect) {
    TokenBatch batch = create_batch(test_sequences_, 0);

    // First sequence: [1, 2, 3] padded to length 5
    EXPECT_EQ(batch.batch_token_ids[0].size(), 5);
    EXPECT_EQ(batch.batch_token_ids[0][0], 1);
    EXPECT_EQ(batch.batch_token_ids[0][1], 2);
    EXPECT_EQ(batch.batch_token_ids[0][2], 3);
    EXPECT_EQ(batch.batch_token_ids[0][3], 0);  // Padding
    EXPECT_EQ(batch.batch_token_ids[0][4], 0);  // Padding

    // Third sequence: [9, 10] padded to length 5
    EXPECT_EQ(batch.batch_token_ids[2].size(), 5);
    EXPECT_EQ(batch.batch_token_ids[2][0], 9);
    EXPECT_EQ(batch.batch_token_ids[2][1], 10);
    EXPECT_EQ(batch.batch_token_ids[2][2], 0);  // Padding
    EXPECT_EQ(batch.batch_token_ids[2][3], 0);  // Padding
    EXPECT_EQ(batch.batch_token_ids[2][4], 0);  // Padding
}

TEST_F(BatchProcessorTest, CreateBatchEmptySequences) {
    std::vector<std::vector<int>> empty_sequences;
    TokenBatch batch = create_batch(empty_sequences, 0);

    EXPECT_TRUE(batch.is_empty());
    EXPECT_EQ(batch.batch_size(), 0);
    EXPECT_EQ(batch.max_length, 0);
}

TEST_F(BatchProcessorTest, CreateBatchCustomPadToken) {
    TokenBatch batch = create_batch(test_sequences_, 99);

    EXPECT_EQ(batch.pad_token_id, 99);
    // Check padding uses custom token
    EXPECT_EQ(batch.batch_token_ids[0][3], 99);
    EXPECT_EQ(batch.batch_token_ids[0][4], 99);
}

TEST_F(BatchProcessorTest, CreateBatchSingleSequence) {
    std::vector<std::vector<int>> single_seq = {{1, 2, 3}};
    TokenBatch batch = create_batch(single_seq, 0);

    EXPECT_EQ(batch.batch_size(), 1);
    EXPECT_EQ(batch.max_length, 3);
    EXPECT_EQ(batch.lengths[0], 3);
    // No padding needed - all sequences same length
    EXPECT_EQ(batch.batch_token_ids[0].size(), 3);
}

// ============================================================================
// Dynamic Batching Tests
// ============================================================================

TEST_F(BatchProcessorTest, DynamicBatchesBasic) {
    // Sequences with varying lengths
    std::vector<std::vector<int>> sequences = {
        {1, 2, 3},            // 3
        {4, 5},               // 2
        {6, 7, 8, 9},         // 4
        {10, 11, 12},         // 3
        {13, 14, 15, 16, 17}  // 5
    };

    auto batches = create_dynamic_batches(sequences, 2, 1, 0);  // max_batch_size=2, tolerance=1

    // Should create multiple batches grouping similar lengths
    EXPECT_GT(batches.size(), 0);

    // Total sequences should be preserved
    int total_seqs = 0;
    for (const auto& batch : batches) {
        total_seqs += batch.batch_size();
    }
    EXPECT_EQ(total_seqs, 5);
}

TEST_F(BatchProcessorTest, DynamicBatchesLengthTolerance) {
    std::vector<std::vector<int>> sequences = {
        {1, 2},        // length 2
        {3, 4, 5},     // length 3
        {6, 7, 8, 9},  // length 4
        {10, 11}       // length 2
    };

    // Tolerance=1: sequences within 1 token can be batched together
    auto batches = create_dynamic_batches(sequences, 10, 1, 0);

    // Should group [2, 3] and [4] separately, or [2] and [3, 4] separately
    // Depends on sorting - should have limited padding due to tolerance
    for (const auto& batch : batches) {
        // Check that max_length - min_length <= tolerance in each batch
        int min_len = *std::min_element(batch.lengths.begin(), batch.lengths.end());
        int max_len = batch.max_length;
        EXPECT_LE(max_len - min_len, 1);
    }
}

TEST_F(BatchProcessorTest, DynamicBatchesBatchSizeLimit) {
    std::vector<std::vector<int>> sequences(10, {1, 2, 3});  // 10 identical sequences

    auto batches = create_dynamic_batches(sequences, 3, 100, 0);  // max_batch_size=3

    // Should create ceil(10/3) = 4 batches
    EXPECT_EQ(batches.size(), 4);
    EXPECT_EQ(batches[0].batch_size(), 3);
    EXPECT_EQ(batches[1].batch_size(), 3);
    EXPECT_EQ(batches[2].batch_size(), 3);
    EXPECT_EQ(batches[3].batch_size(), 1);
}

TEST_F(BatchProcessorTest, DynamicBatchesEmptyInput) {
    std::vector<std::vector<int>> empty_sequences;
    auto batches = create_dynamic_batches(empty_sequences, 32, 10, 0);

    EXPECT_TRUE(batches.empty());
}

TEST_F(BatchProcessorTest, DynamicBatchesSingleBatch) {
    std::vector<std::vector<int>> sequences = {{1, 2, 3}, {4, 5, 6, 7}};

    // Large tolerance and batch size - everything in one batch
    auto batches = create_dynamic_batches(sequences, 100, 100, 0);

    EXPECT_EQ(batches.size(), 1);
    EXPECT_EQ(batches[0].batch_size(), 2);
}

// ============================================================================
// Padding Mask Tests
// ============================================================================

TEST_F(BatchProcessorTest, PaddingMaskBasic) {
    TokenBatch batch = create_batch(test_sequences_, 0);
    Matrix mask = create_padding_mask(batch);

    // Mask dimensions: [batch_size, max_length]
    EXPECT_EQ(mask.rows, 4);
    EXPECT_EQ(mask.cols, 5);

    // First sequence (length 3): [1, 1, 1, 0, 0]
    EXPECT_FLOAT_EQ(mask(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(mask(0, 1), 1.0f);
    EXPECT_FLOAT_EQ(mask(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(mask(0, 3), 0.0f);
    EXPECT_FLOAT_EQ(mask(0, 4), 0.0f);

    // Second sequence (length 5): [1, 1, 1, 1, 1]
    EXPECT_FLOAT_EQ(mask(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(mask(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(mask(1, 2), 1.0f);
    EXPECT_FLOAT_EQ(mask(1, 3), 1.0f);
    EXPECT_FLOAT_EQ(mask(1, 4), 1.0f);

    // Third sequence (length 2): [1, 1, 0, 0, 0]
    EXPECT_FLOAT_EQ(mask(2, 0), 1.0f);
    EXPECT_FLOAT_EQ(mask(2, 1), 1.0f);
    EXPECT_FLOAT_EQ(mask(2, 2), 0.0f);
    EXPECT_FLOAT_EQ(mask(2, 3), 0.0f);
    EXPECT_FLOAT_EQ(mask(2, 4), 0.0f);
}

TEST_F(BatchProcessorTest, PaddingMaskNoPadding) {
    std::vector<std::vector<int>> same_length = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    TokenBatch batch = create_batch(same_length, 0);
    Matrix mask = create_padding_mask(batch);

    // All 1s - no padding needed
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            EXPECT_FLOAT_EQ(mask(i, j), 1.0f);
        }
    }
}

TEST_F(BatchProcessorTest, PaddingMaskEmptyBatch) {
    TokenBatch empty_batch;
    empty_batch.max_length = 0;

    Matrix mask = create_padding_mask(empty_batch);

    EXPECT_EQ(mask.rows, 0);
    EXPECT_EQ(mask.cols, 0);
}

TEST_F(BatchProcessorTest, PaddingMaskSingleToken) {
    std::vector<std::vector<int>> sequences = {
        {1},       // length 1
        {2, 3, 4}  // length 3
    };

    TokenBatch batch = create_batch(sequences, 0);
    Matrix mask = create_padding_mask(batch);

    // First sequence: [1, 0, 0]
    EXPECT_FLOAT_EQ(mask(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(mask(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(mask(0, 2), 0.0f);

    // Second sequence: [1, 1, 1]
    EXPECT_FLOAT_EQ(mask(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(mask(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(mask(1, 2), 1.0f);
}

// ============================================================================
// Unbatching Tests
// ============================================================================

TEST_F(BatchProcessorTest, UnbatchOutputsBasic) {
    TokenBatch batch = create_batch(test_sequences_, 0);

    // Create fake outputs (one matrix per batch item)
    // Each matrix is [max_length, d_model] where d_model=4
    std::vector<Matrix> batch_outputs;
    for (int i = 0; i < batch.batch_size(); ++i) {
        Matrix output(batch.max_length, 4);
        // Fill with unique values for testing
        for (int row = 0; row < batch.max_length; ++row) {
            for (int col = 0; col < 4; ++col) {
                output(row, col) = i * 100.0f + row * 10.0f + col;
            }
        }
        batch_outputs.push_back(output);
    }

    auto unbatched = unbatch_outputs(batch_outputs, batch);

    // Should have one output per sequence
    EXPECT_EQ(unbatched.size(), 4);

    // First sequence (length 3): should be [3, 4]
    EXPECT_EQ(unbatched[0].rows, 3);
    EXPECT_EQ(unbatched[0].cols, 4);

    // Second sequence (length 5): should be [5, 4]
    EXPECT_EQ(unbatched[1].rows, 5);
    EXPECT_EQ(unbatched[1].cols, 4);

    // Third sequence (length 2): should be [2, 4]
    EXPECT_EQ(unbatched[2].rows, 2);
    EXPECT_EQ(unbatched[2].cols, 4);

    // Fourth sequence (length 4): should be [4, 4]
    EXPECT_EQ(unbatched[3].rows, 4);
    EXPECT_EQ(unbatched[3].cols, 4);
}

TEST_F(BatchProcessorTest, UnbatchOutputsValuePreservation) {
    std::vector<std::vector<int>> sequences = {{1, 2}, {3, 4, 5, 6}};
    TokenBatch batch = create_batch(sequences, 0);

    // Create outputs with specific values
    std::vector<Matrix> batch_outputs;
    Matrix output1(4, 2);  // max_length=4, d_model=2
    output1(0, 0) = 1.1f;
    output1(0, 1) = 1.2f;
    output1(1, 0) = 2.1f;
    output1(1, 1) = 2.2f;
    output1(2, 0) = 9.9f;
    output1(2, 1) = 9.9f;  // Padding (should be removed)
    output1(3, 0) = 9.9f;
    output1(3, 1) = 9.9f;  // Padding (should be removed)
    batch_outputs.push_back(output1);

    Matrix output2(4, 2);
    output2(0, 0) = 10.1f;
    output2(0, 1) = 10.2f;
    output2(1, 0) = 11.1f;
    output2(1, 1) = 11.2f;
    output2(2, 0) = 12.1f;
    output2(2, 1) = 12.2f;
    output2(3, 0) = 13.1f;
    output2(3, 1) = 13.2f;
    batch_outputs.push_back(output2);

    auto unbatched = unbatch_outputs(batch_outputs, batch);

    // First sequence: only first 2 rows
    EXPECT_EQ(unbatched[0].rows, 2);
    EXPECT_FLOAT_EQ(unbatched[0](0, 0), 1.1f);
    EXPECT_FLOAT_EQ(unbatched[0](0, 1), 1.2f);
    EXPECT_FLOAT_EQ(unbatched[0](1, 0), 2.1f);
    EXPECT_FLOAT_EQ(unbatched[0](1, 1), 2.2f);

    // Second sequence: all 4 rows
    EXPECT_EQ(unbatched[1].rows, 4);
    EXPECT_FLOAT_EQ(unbatched[1](0, 0), 10.1f);
    EXPECT_FLOAT_EQ(unbatched[1](3, 1), 13.2f);
}

TEST_F(BatchProcessorTest, UnbatchOutputsNoPadding) {
    std::vector<std::vector<int>> same_length = {{1, 2, 3}, {4, 5, 6}};
    TokenBatch batch = create_batch(same_length, 0);

    std::vector<Matrix> batch_outputs;
    for (int i = 0; i < 2; ++i) {
        Matrix output(3, 2);
        batch_outputs.push_back(output);
    }

    auto unbatched = unbatch_outputs(batch_outputs, batch);

    // No padding - output dimensions should match input
    EXPECT_EQ(unbatched[0].rows, 3);
    EXPECT_EQ(unbatched[1].rows, 3);
}

TEST_F(BatchProcessorTest, UnbatchOutputsSingleSequence) {
    std::vector<std::vector<int>> single_seq = {{1, 2, 3, 4}};
    TokenBatch batch = create_batch(single_seq, 0);

    std::vector<Matrix> batch_outputs;
    Matrix output(4, 3);
    batch_outputs.push_back(output);

    auto unbatched = unbatch_outputs(batch_outputs, batch);

    EXPECT_EQ(unbatched.size(), 1);
    EXPECT_EQ(unbatched[0].rows, 4);
    EXPECT_EQ(unbatched[0].cols, 3);
}

// ============================================================================
// Batch Statistics Tests
// ============================================================================

TEST_F(BatchProcessorTest, BatchStatsBasic) {
    auto batches = create_dynamic_batches(test_sequences_, 10, 10, 0);
    BatchStats stats = compute_batch_stats(batches);

    // Total actual tokens: 3 + 5 + 2 + 4 = 14
    EXPECT_EQ(stats.actual_tokens, 14);

    // Max length in batch is 5, so total tokens = 4 * 5 = 20
    EXPECT_EQ(stats.total_tokens, 20);

    // Padding ratio: (20 - 14) / 20 = 6 / 20 = 0.3
    EXPECT_FLOAT_EQ(stats.padding_ratio, 0.3f);

    EXPECT_GT(stats.num_batches, 0);
}

TEST_F(BatchProcessorTest, BatchStatsMultipleBatches) {
    std::vector<std::vector<int>> sequences = {
        {1, 2, 3},        // 3
        {4, 5},           // 2
        {6, 7, 8, 9, 10}  // 5
    };

    auto batches = create_dynamic_batches(sequences, 1, 0, 0);  // Force one sequence per batch
    BatchStats stats = compute_batch_stats(batches);

    EXPECT_EQ(stats.num_batches, 3);
    EXPECT_EQ(stats.actual_tokens, 10);  // 3 + 2 + 5
    EXPECT_FLOAT_EQ(stats.avg_batch_size, 1.0f);
}

TEST_F(BatchProcessorTest, BatchStatsNoPadding) {
    std::vector<std::vector<int>> same_length(5, {1, 2, 3, 4});
    auto batches = create_dynamic_batches(same_length, 10, 10, 0);
    BatchStats stats = compute_batch_stats(batches);

    // No padding - efficiency should be 100%
    EXPECT_FLOAT_EQ(stats.padding_ratio, 0.0f);
    EXPECT_EQ(stats.total_tokens, stats.actual_tokens);
}

TEST_F(BatchProcessorTest, BatchStatsEmptyBatches) {
    std::vector<TokenBatch> empty_batches;
    BatchStats stats = compute_batch_stats(empty_batches);

    EXPECT_EQ(stats.num_batches, 0);
    EXPECT_EQ(stats.total_tokens, 0);
    EXPECT_EQ(stats.actual_tokens, 0);
    EXPECT_FLOAT_EQ(stats.padding_ratio, 0.0f);
    EXPECT_FLOAT_EQ(stats.avg_batch_size, 0.0f);
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

TEST_F(BatchProcessorTest, EdgeCaseVeryLongSequence) {
    std::vector<int> long_seq(1000, 1);
    std::vector<int> short_seq = {2, 3};

    TokenBatch batch = create_batch({long_seq, short_seq}, 0);

    EXPECT_EQ(batch.max_length, 1000);
    EXPECT_EQ(batch.lengths[0], 1000);
    EXPECT_EQ(batch.lengths[1], 2);

    // Padding should work correctly
    EXPECT_EQ(batch.batch_token_ids[1].size(), 1000);
    EXPECT_EQ(batch.batch_token_ids[1][0], 2);
    EXPECT_EQ(batch.batch_token_ids[1][1], 3);
    EXPECT_EQ(batch.batch_token_ids[1][2], 0);  // Padding
}

TEST_F(BatchProcessorTest, EdgeCaseAllEmptySequencesInvalid) {
    // Note: Empty sequences are invalid input - batch creation assumes non-empty sequences
    // This test documents expected behavior with single empty sequence
    std::vector<std::vector<int>> has_empty = {{}, {1, 2}};
    TokenBatch batch = create_batch(has_empty, 0);

    // Should still create batch (empty sequence has length 0)
    EXPECT_EQ(batch.batch_size(), 2);
}

TEST_F(BatchProcessorTest, EdgeCaseSingleTokenSequences) {
    std::vector<std::vector<int>> single_tokens = {{1}, {2}, {3}, {4}};
    TokenBatch batch = create_batch(single_tokens, 0);

    EXPECT_EQ(batch.max_length, 1);
    EXPECT_EQ(batch.batch_size(), 4);

    // No padding needed
    for (const auto& seq : batch.batch_token_ids) {
        EXPECT_EQ(seq.size(), 1);
    }
}

TEST_F(BatchProcessorTest, EdgeCaseWideLengthVariation) {
    std::vector<std::vector<int>> varied = {
        {1},                                  // 1
        {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}  // 11
    };

    TokenBatch batch = create_batch(varied, 0);

    EXPECT_EQ(batch.max_length, 11);

    // First sequence should be heavily padded
    EXPECT_EQ(batch.batch_token_ids[0].size(), 11);
    EXPECT_EQ(batch.batch_token_ids[0][0], 1);
    for (int i = 1; i < 11; ++i) {
        EXPECT_EQ(batch.batch_token_ids[0][i], 0);  // All padding
    }
}

TEST_F(BatchProcessorTest, EdgeCaseDynamicBatchingStrictTolerance) {
    std::vector<std::vector<int>> sequences = {
        {1, 2},               // 2
        {3, 4, 5},            // 3
        {6, 7, 8, 9},         // 4
        {10, 11, 12, 13, 14}  // 5
    };

    // Zero tolerance - each sequence in its own batch
    auto batches = create_dynamic_batches(sequences, 10, 0, 0);

    // Should create separate batches for each length
    EXPECT_EQ(batches.size(), 4);
    for (const auto& batch : batches) {
        EXPECT_EQ(batch.batch_size(), 1);
    }
}
