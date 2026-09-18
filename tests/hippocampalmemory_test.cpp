/**
 * @file hippocampalmemory_test.cpp
 * @brief Tests for HippocampalMemory (TD-179 / HM-1) — bounded episodic buffer from
 *        docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 5.
 *
 * TD-179's own Action Items ask for three things specifically: write()/read_all()/
 * coverage_vector()/decay_coverage()/clear() per the proposal's interface, a dedicated unit
 * test that FIFO eviction drops the oldest slot (not a random one), and save()/load() for
 * session persistence. Salience-gated writing is explicitly out of scope.
 */

#include "../src/HippocampalMemory.hpp"
#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include "../src/Matrix.hpp"

namespace {

// A [1, d_model] row filled with a single distinguishing value — makes it easy to tell slots
// apart by inspection (e.g. "this row is the 5th one written") without needing real embeddings.
Matrix make_row(int d_model, float value) {
    Matrix m(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        m(0, j) = value;
    }
    return m;
}

bool row_equals(const Matrix& m, int row, float expected, int d_model) {
    for (int j = 0; j < d_model; ++j) {
        if (m(row, j) != expected) {
            return false;
        }
    }
    return true;
}

}  // namespace

TEST(HippocampalMemoryTest, ConstructorRejectsNonPositiveDimensions) {
    EXPECT_THROW(HippocampalMemory(0, 8), std::invalid_argument);
    EXPECT_THROW(HippocampalMemory(-4, 8), std::invalid_argument);
    EXPECT_THROW(HippocampalMemory(16, 0), std::invalid_argument);
    EXPECT_THROW(HippocampalMemory(16, -1), std::invalid_argument);
}

TEST(HippocampalMemoryTest, AccessorsReturnConstructorArguments) {
    HippocampalMemory memory(16, 32);
    EXPECT_EQ(memory.get_d_model(), 16);
    EXPECT_EQ(memory.get_capacity(), 32);
}

TEST(HippocampalMemoryTest, SizeIsZeroInitially) {
    HippocampalMemory memory(8, 4);
    EXPECT_EQ(memory.size(), 0);
}

TEST(HippocampalMemoryTest, WriteRejectsMismatchedKeyShape) {
    HippocampalMemory memory(8, 4);
    Matrix wrong_key(2, 8);   // wrong row count
    Matrix wrong_key2(1, 4);  // wrong column count
    Matrix value = make_row(8, 1.0f);

    EXPECT_THROW(memory.write(wrong_key, value), std::invalid_argument);
    EXPECT_THROW(memory.write(wrong_key2, value), std::invalid_argument);
}

TEST(HippocampalMemoryTest, WriteRejectsMismatchedValueShape) {
    HippocampalMemory memory(8, 4);
    Matrix key = make_row(8, 1.0f);
    Matrix wrong_value(1, 5);

    EXPECT_THROW(memory.write(key, wrong_value), std::invalid_argument);
}

TEST(HippocampalMemoryTest, WriteIncreasesSize) {
    HippocampalMemory memory(8, 4);
    memory.write(make_row(8, 1.0f), make_row(8, 1.0f));
    EXPECT_EQ(memory.size(), 1);
    memory.write(make_row(8, 2.0f), make_row(8, 2.0f));
    EXPECT_EQ(memory.size(), 2);
}

TEST(HippocampalMemoryTest, WriteBeyondCapacityDoesNotExceedIt) {
    const int capacity = 4;
    HippocampalMemory memory(8, capacity);
    for (int i = 0; i < capacity + 10; ++i) {
        memory.write(make_row(8, static_cast<float>(i)), make_row(8, static_cast<float>(i)));
    }
    EXPECT_EQ(memory.size(), capacity);
}

// TD-179's own Action Item: FIFO eviction must drop the oldest slot specifically, not a random
// one. Writes distinguishable rows (value == write index) one past capacity, then confirms the
// surviving slots are exactly the most-recently-written `capacity` of them, oldest-first.
TEST(HippocampalMemoryTest, FIFOEvictionEvictsOldestSlotNotRandom) {
    const int d_model = 4;
    const int capacity = 3;
    HippocampalMemory memory(d_model, capacity);

    for (int i = 0; i < capacity + 1; ++i) {
        memory.write(make_row(d_model, static_cast<float>(i)), make_row(d_model, static_cast<float>(i)));
    }

    ASSERT_EQ(memory.size(), capacity);
    auto [keys, values] = memory.read_all();

    // Slot 0 (value 0.0f) must be gone; the surviving slots must be 1, 2, 3 in that order.
    for (int i = 0; i < capacity; ++i) {
        float expected = static_cast<float>(i + 1);
        EXPECT_TRUE(row_equals(keys, i, expected, d_model)) << "row " << i;
        EXPECT_TRUE(row_equals(values, i, expected, d_model)) << "row " << i;
    }
}

TEST(HippocampalMemoryTest, ReadAllReturnsEmptyMatricesWhenEmpty) {
    HippocampalMemory memory(8, 4);
    auto [keys, values] = memory.read_all();
    EXPECT_EQ(keys.rows, 0);
    EXPECT_EQ(keys.cols, 8);
    EXPECT_EQ(values.rows, 0);
    EXPECT_EQ(values.cols, 8);
}

TEST(HippocampalMemoryTest, ReadAllMatchesWrittenKeysAndValues) {
    const int d_model = 4;
    HippocampalMemory memory(d_model, 8);

    memory.write(make_row(d_model, 1.0f), make_row(d_model, 10.0f));
    memory.write(make_row(d_model, 2.0f), make_row(d_model, 20.0f));

    auto [keys, values] = memory.read_all();
    ASSERT_EQ(keys.rows, 2);
    ASSERT_EQ(values.rows, 2);
    EXPECT_TRUE(row_equals(keys, 0, 1.0f, d_model));
    EXPECT_TRUE(row_equals(keys, 1, 2.0f, d_model));
    EXPECT_TRUE(row_equals(values, 0, 10.0f, d_model));
    EXPECT_TRUE(row_equals(values, 1, 20.0f, d_model));
}

TEST(HippocampalMemoryTest, CoverageVectorStartsAtZeroForNewSlots) {
    HippocampalMemory memory(8, 4);
    memory.write(make_row(8, 1.0f), make_row(8, 1.0f));
    memory.write(make_row(8, 2.0f), make_row(8, 2.0f));

    auto& coverage = memory.coverage_vector();
    ASSERT_EQ(coverage.size(), 2u);
    EXPECT_FLOAT_EQ(coverage[0], 0.0f);
    EXPECT_FLOAT_EQ(coverage[1], 0.0f);
}

TEST(HippocampalMemoryTest, CoverageVectorIsMutableAndPersists) {
    HippocampalMemory memory(8, 4);
    memory.write(make_row(8, 1.0f), make_row(8, 1.0f));

    memory.coverage_vector()[0] += 0.5f;
    memory.coverage_vector()[0] += 0.25f;

    EXPECT_FLOAT_EQ(memory.coverage_vector()[0], 0.75f);
}

TEST(HippocampalMemoryTest, NewSlotsStartWithZeroCoverageEvenAfterOthersAccumulate) {
    HippocampalMemory memory(8, 4);
    memory.write(make_row(8, 1.0f), make_row(8, 1.0f));
    memory.coverage_vector()[0] = 0.9f;

    memory.write(make_row(8, 2.0f), make_row(8, 2.0f));

    EXPECT_FLOAT_EQ(memory.coverage_vector()[0], 0.9f);
    EXPECT_FLOAT_EQ(memory.coverage_vector()[1], 0.0f);
}

TEST(HippocampalMemoryTest, DecayCoverageMultipliesAllBySameGamma) {
    HippocampalMemory memory(8, 4);
    memory.write(make_row(8, 1.0f), make_row(8, 1.0f));
    memory.write(make_row(8, 2.0f), make_row(8, 2.0f));
    memory.coverage_vector()[0] = 1.0f;
    memory.coverage_vector()[1] = 2.0f;

    memory.decay_coverage(0.5f);

    EXPECT_FLOAT_EQ(memory.coverage_vector()[0], 0.5f);
    EXPECT_FLOAT_EQ(memory.coverage_vector()[1], 1.0f);
}

TEST(HippocampalMemoryTest, CoverageStaysAlignedWithSlotsAcrossEviction) {
    const int d_model = 4;
    const int capacity = 2;
    HippocampalMemory memory(d_model, capacity);

    memory.write(make_row(d_model, 1.0f), make_row(d_model, 1.0f));
    memory.coverage_vector()[0] = 5.0f;
    memory.write(make_row(d_model, 2.0f), make_row(d_model, 2.0f));
    memory.coverage_vector()[1] = 7.0f;

    // This write evicts slot 0 (coverage 5.0f) — slot with coverage 7.0f must now be index 0.
    memory.write(make_row(d_model, 3.0f), make_row(d_model, 3.0f));

    ASSERT_EQ(memory.size(), capacity);
    EXPECT_FLOAT_EQ(memory.coverage_vector()[0], 7.0f);
    EXPECT_FLOAT_EQ(memory.coverage_vector()[1], 0.0f);  // the newly-written slot
}

TEST(HippocampalMemoryTest, ClearRemovesAllSlotsAndCoverage) {
    HippocampalMemory memory(8, 4);
    memory.write(make_row(8, 1.0f), make_row(8, 1.0f));
    memory.write(make_row(8, 2.0f), make_row(8, 2.0f));

    memory.clear();

    EXPECT_EQ(memory.size(), 0);
    EXPECT_TRUE(memory.coverage_vector().empty());
}

TEST(HippocampalMemoryTest, SaveAndLoadRoundTripsSlotsAndCoverage) {
    const int d_model = 4;
    const std::string filepath = "test_hippocampal_memory.bin";

    HippocampalMemory memory1(d_model, 8);
    memory1.write(make_row(d_model, 1.0f), make_row(d_model, 10.0f));
    memory1.write(make_row(d_model, 2.0f), make_row(d_model, 20.0f));
    memory1.coverage_vector()[0] = 0.3f;
    memory1.coverage_vector()[1] = 0.6f;
    memory1.save(filepath);

    HippocampalMemory memory2(d_model, 8);
    memory2.load(filepath);

    ASSERT_EQ(memory2.size(), 2);
    auto [keys, values] = memory2.read_all();
    EXPECT_TRUE(row_equals(keys, 0, 1.0f, d_model));
    EXPECT_TRUE(row_equals(keys, 1, 2.0f, d_model));
    EXPECT_TRUE(row_equals(values, 0, 10.0f, d_model));
    EXPECT_TRUE(row_equals(values, 1, 20.0f, d_model));
    EXPECT_FLOAT_EQ(memory2.coverage_vector()[0], 0.3f);
    EXPECT_FLOAT_EQ(memory2.coverage_vector()[1], 0.6f);

    std::remove(filepath.c_str());
}

TEST(HippocampalMemoryTest, LoadRejectsMismatchedDModel) {
    const std::string filepath = "test_hippocampal_memory_mismatch.bin";

    HippocampalMemory memory1(4, 8);
    memory1.write(make_row(4, 1.0f), make_row(4, 1.0f));
    memory1.save(filepath);

    HippocampalMemory memory2(8, 8);  // different d_model
    EXPECT_THROW(memory2.load(filepath), std::runtime_error);

    std::remove(filepath.c_str());
}

TEST(HippocampalMemoryTest, LoadNonExistentFileThrows) {
    HippocampalMemory memory(8, 4);
    EXPECT_THROW(memory.load("nonexistent_hippocampal_memory_file.bin"), std::runtime_error);
}

// Regression tests: load() must validate the header's own slot count against the file's actual
// remaining size before trusting it for loaded_coverage.reserve() — previously a corrupted or
// wrong-format file could hand reserve() a huge or negative value and throw an unrelated
// std::length_error/std::bad_alloc instead of this class's own consistent std::runtime_error.
TEST(HippocampalMemoryTest, LoadRejectsNegativeSlotCount) {
    const int d_model = 4;
    const std::string filepath = "test_hippocampal_memory_negative_slots.bin";

    std::ofstream file(filepath, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    const int capacity = 8;
    const int negative_slots = -1;
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<const char*>(&capacity), sizeof(int));
    file.write(reinterpret_cast<const char*>(&negative_slots), sizeof(int));
    file.close();

    HippocampalMemory memory(d_model, capacity);
    EXPECT_THROW(memory.load(filepath), std::runtime_error);

    std::remove(filepath.c_str());
}

TEST(HippocampalMemoryTest, LoadRejectsSlotCountLargerThanFileActuallyContains) {
    const int d_model = 4;
    const std::string filepath = "test_hippocampal_memory_truncated.bin";

    // A real, valid one-slot file...
    HippocampalMemory memory1(d_model, 8);
    memory1.write(make_row(d_model, 1.0f), make_row(d_model, 1.0f));
    memory1.save(filepath);

    // ...whose header is then corrupted to claim far more slots than the file actually holds.
    std::fstream file(filepath, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file.is_open());
    file.seekp(2 * sizeof(int));  // the num_slots field, after d_model and capacity
    const int bogus_slots = 1000000;
    file.write(reinterpret_cast<const char*>(&bogus_slots), sizeof(int));
    file.close();

    HippocampalMemory memory2(d_model, 8);
    EXPECT_THROW(memory2.load(filepath), std::runtime_error);

    std::remove(filepath.c_str());
}

TEST(HippocampalMemoryTest, LoadAcceptsDifferentCapacityAndEvictsExcess) {
    const int d_model = 4;
    const std::string filepath = "test_hippocampal_memory_capacity.bin";

    HippocampalMemory memory1(d_model, 8);
    for (int i = 0; i < 5; ++i) {
        memory1.write(make_row(d_model, static_cast<float>(i)), make_row(d_model, static_cast<float>(i)));
    }
    memory1.save(filepath);

    // Loading instance has a smaller capacity than the saved session's slot count.
    HippocampalMemory memory2(d_model, 3);
    memory2.load(filepath);

    ASSERT_EQ(memory2.size(), 3);
    auto [keys, values] = memory2.read_all();
    // The 3 most-recently-written slots (2, 3, 4) must survive, oldest-first.
    EXPECT_TRUE(row_equals(keys, 0, 2.0f, d_model));
    EXPECT_TRUE(row_equals(keys, 1, 3.0f, d_model));
    EXPECT_TRUE(row_equals(keys, 2, 4.0f, d_model));

    std::remove(filepath.c_str());
}
