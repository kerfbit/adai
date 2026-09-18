/**
 * @file hippocampalmemory_test.cpp
 * @brief Tests for HippocampalMemory (TD-179 / HM-1) — bounded episodic buffer from
 *        docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 5. Also covers
 *        TD-193's least-used eviction policy and reloadable swap file.
 *
 * TD-179's own Action Items ask for three things specifically: write()/read_all()/
 * coverage_vector()/decay_coverage()/clear() per the proposal's interface, a dedicated unit
 * test that eviction drops the correct slot (not a random one), and save()/load() for session
 * persistence. TD-193 replaces the original FIFO eviction policy with a least-used one (lowest
 * coverage_ value, ties broken by oldest — FIFO is what this degenerates to when nothing's
 * coverage has ever been touched) and adds an optional, reloadable swap file for evicted slots.
 */

#include "../src/HippocampalMemory.hpp"
#include <gtest/gtest.h>
#include <cstdio>
#include <filesystem>
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

// TD-193: least-used eviction with every slot tied at coverage 0.0f (never touched) degenerates
// to plain FIFO — ties broken by oldest/lowest index. Writes distinguishable rows (value == write
// index) one past capacity, then confirms the surviving slots are exactly the most-recently-
// written `capacity` of them, oldest-first.
TEST(HippocampalMemoryTest, EvictionTiesBreakByOldestSlotNotRandom) {
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

// TD-193's own discriminating case: a genuinely least-used (not merely oldest) slot must be the
// one evicted. Slot 0 is the OLDEST but has high coverage (heavily used); slot 1 is NEWER but has
// the lowest coverage of all three — slot 1 must be the one evicted, not slot 0.
TEST(HippocampalMemoryTest, LeastUsedEvictionEvictsLowestCoverageRegardlessOfAge) {
    const int d_model = 4;
    const int capacity = 3;
    HippocampalMemory memory(d_model, capacity);

    memory.write(make_row(d_model, 0.0f), make_row(d_model, 0.0f));
    memory.coverage_vector()[0] = 9.0f;  // oldest, but heavily used
    memory.write(make_row(d_model, 1.0f), make_row(d_model, 1.0f));
    memory.coverage_vector()[1] = 0.1f;  // newer, but barely used — least used overall
    memory.write(make_row(d_model, 2.0f), make_row(d_model, 2.0f));
    memory.coverage_vector()[2] = 5.0f;

    memory.write(make_row(d_model, 3.0f), make_row(d_model, 3.0f));

    ASSERT_EQ(memory.size(), capacity);
    auto [keys, values] = memory.read_all();
    bool slot1_survived = false;
    for (int i = 0; i < capacity; ++i) {
        if (row_equals(keys, i, 1.0f, d_model)) {
            slot1_survived = true;
        }
    }
    EXPECT_FALSE(slot1_survived) << "the least-used slot (value 1.0, coverage 0.1) should have "
                                    "been evicted despite not being the oldest";
    EXPECT_TRUE(row_equals(keys, 0, 0.0f, d_model)) << "the oldest-but-heavily-used slot survives";
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

    // This write evicts slot 0 — the *least used* (coverage 5.0f < 7.0f), not merely the oldest —
    // so the slot with coverage 7.0f must now be index 0.
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

    // Loading instance has a smaller capacity than the saved session's slot count. All 5 saved
    // slots have equal (untouched, 0.0f) coverage, so the least-used trim degenerates to FIFO.
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

// ============================================================================
// TD-193: swap file (least-used eviction destination + recall_from_swap())
// ============================================================================

TEST(HippocampalMemoryTest, SwapDisabledByDefaultDiscardsEvictedSlots) {
    const int d_model = 4;
    HippocampalMemory memory(d_model, 2);  // no swap_filepath supplied

    EXPECT_FALSE(memory.swap_enabled());

    memory.write(make_row(d_model, 1.0f), make_row(d_model, 1.0f));
    memory.write(make_row(d_model, 2.0f), make_row(d_model, 2.0f));
    memory.write(make_row(d_model, 3.0f), make_row(d_model, 3.0f));  // evicts slot 0

    // Nothing to recall — matches this class's original (pre-TD-193) behavior exactly.
    EXPECT_FALSE(memory.recall_from_swap());
}

TEST(HippocampalMemoryTest, EvictedSlotIsAppendedToSwapFileWhenEnabled) {
    const int d_model = 4;
    const std::string swap_path = "test_hippocampal_memory_swap.bin";
    std::remove(swap_path.c_str());

    HippocampalMemory memory(d_model, 2, swap_path);
    EXPECT_TRUE(memory.swap_enabled());

    memory.write(make_row(d_model, 1.0f), make_row(d_model, 1.0f));
    memory.write(make_row(d_model, 2.0f), make_row(d_model, 2.0f));
    memory.write(make_row(d_model, 3.0f), make_row(d_model, 3.0f));  // evicts slot 0 -> swap

    EXPECT_TRUE(std::filesystem::exists(swap_path));
    EXPECT_GT(std::filesystem::file_size(swap_path), 0u);

    std::remove(swap_path.c_str());
}

TEST(HippocampalMemoryTest, RecallFromSwapReturnsFalseWhenSwapDisabled) {
    HippocampalMemory memory(4, 2);
    EXPECT_FALSE(memory.recall_from_swap());
}

TEST(HippocampalMemoryTest, RecallFromSwapReturnsFalseWhenSwapFileDoesNotExist) {
    const std::string swap_path = "test_hippocampal_memory_swap_missing.bin";
    std::remove(swap_path.c_str());

    HippocampalMemory memory(4, 2, swap_path);
    EXPECT_FALSE(memory.recall_from_swap());
}

TEST(HippocampalMemoryTest, RecallFromSwapBringsBackEvictedSlotWithResetCoverage) {
    const int d_model = 4;
    const std::string swap_path = "test_hippocampal_memory_swap_recall.bin";
    std::remove(swap_path.c_str());

    HippocampalMemory memory(d_model, 2, swap_path);
    memory.write(make_row(d_model, 1.0f), make_row(d_model, 10.0f));  // coverage 0 — least used
    memory.write(make_row(d_model, 2.0f), make_row(d_model, 20.0f));
    memory.coverage_vector()[1] = 5.0f;  // value 2.0's slot is now the heavily-used one

    // At capacity (2): evicts the slot holding (1.0, 10.0) — coverage 0, the lowest — into swap.
    memory.write(make_row(d_model, 3.0f), make_row(d_model, 30.0f));
    ASSERT_EQ(memory.size(), 2);

    // Recalling makes room the same way write() would (capacity is still 2, already full with
    // value 2.0's slot [coverage 5] and value 3.0's slot [coverage 0, just written]) — so the
    // recall itself evicts value 3.0's slot back out to swap while bringing value 1.0's back in.
    ASSERT_TRUE(memory.recall_from_swap());
    ASSERT_EQ(memory.size(), 2);

    auto [keys, values] = memory.read_all();
    bool found_recalled = false;
    for (int i = 0; i < memory.size(); ++i) {
        if (row_equals(keys, i, 1.0f, d_model) && row_equals(values, i, 10.0f, d_model)) {
            found_recalled = true;
            EXPECT_FLOAT_EQ(memory.coverage_vector()[i], 0.0f)
                << "a recalled slot's coverage should reset to 0.0f, treated as freshly relevant";
        }
    }
    EXPECT_TRUE(found_recalled) << "the slot evicted earlier (value 1.0/10.0) should be back";

    std::remove(swap_path.c_str());
}

TEST(HippocampalMemoryTest, RecallFromSwapIsLIFOMostRecentlyEvictedFirst) {
    const int d_model = 4;
    const std::string swap_path = "test_hippocampal_memory_swap_lifo.bin";
    std::remove(swap_path.c_str());

    HippocampalMemory memory(d_model, 1, swap_path);  // capacity 1: every write but the first evicts
    memory.write(make_row(d_model, 1.0f), make_row(d_model, 1.0f));  // evicted first (oldest)
    memory.write(make_row(d_model, 2.0f), make_row(d_model, 2.0f));  // evicted second
    memory.write(make_row(d_model, 3.0f), make_row(d_model, 3.0f));  // stays live

    ASSERT_EQ(memory.size(), 1);

    // First recall should bring back the MOST recently evicted (value 2.0), not the oldest.
    ASSERT_TRUE(memory.recall_from_swap());
    auto [keys1, values1] = memory.read_all();
    (void)values1;
    bool has_two = false;
    for (int i = 0; i < memory.size(); ++i) {
        if (row_equals(keys1, i, 2.0f, d_model)) {
            has_two = true;
        }
    }
    EXPECT_TRUE(has_two) << "the most recently evicted slot (value 2.0) should come back first";

    std::remove(swap_path.c_str());
}

TEST(HippocampalMemoryTest, RecallFromSwapDrainsToFalseOnceEmpty) {
    const int d_model = 4;
    const std::string swap_path = "test_hippocampal_memory_swap_drain.bin";
    std::remove(swap_path.c_str());

    // Populate the swap file with a tight-capacity writer (2 evictions -> 2 swap records).
    {
        HippocampalMemory writer(d_model, 1, swap_path);
        writer.write(make_row(d_model, 1.0f), make_row(d_model, 1.0f));  // evicted on next write
        writer.write(make_row(d_model, 2.0f), make_row(d_model, 2.0f));  // evicted on next write
        writer.write(make_row(d_model, 3.0f), make_row(d_model, 3.0f));  // stays live
    }

    // A fresh reader with plenty of spare capacity, so recalling never needs to evict anything
    // back — otherwise (e.g. at capacity) each recall would perpetually swap one thing back out
    // as it brings another in, and the file would never actually drain.
    HippocampalMemory reader(d_model, 10, swap_path);
    EXPECT_TRUE(reader.recall_from_swap());
    EXPECT_TRUE(reader.recall_from_swap());
    EXPECT_FALSE(reader.recall_from_swap()) << "swap should now be empty";
    EXPECT_EQ(reader.size(), 2);

    std::remove(swap_path.c_str());
}

TEST(HippocampalMemoryTest, RecallFromSwapRejectsMismatchedDModel) {
    const std::string swap_path = "test_hippocampal_memory_swap_mismatch.bin";
    std::remove(swap_path.c_str());

    HippocampalMemory writer(4, 1, swap_path);
    writer.write(make_row(4, 1.0f), make_row(4, 1.0f));
    writer.write(make_row(4, 2.0f), make_row(4, 2.0f));  // evicts one slot into swap_path

    HippocampalMemory reader(8, 1, swap_path);  // different d_model, same swap file
    EXPECT_THROW(reader.recall_from_swap(), std::runtime_error);

    std::remove(swap_path.c_str());
}

TEST(HippocampalMemoryTest, LoadExcessTrimAppendsLeastUsedSlotsToSwap) {
    const int d_model = 4;
    const std::string filepath = "test_hippocampal_memory_load_swap_main.bin";
    const std::string swap_path = "test_hippocampal_memory_load_swap_swap.bin";
    std::remove(filepath.c_str());
    std::remove(swap_path.c_str());

    HippocampalMemory memory1(d_model, 8);
    for (int i = 0; i < 3; ++i) {
        memory1.write(make_row(d_model, static_cast<float>(i)), make_row(d_model, static_cast<float>(i)));
    }
    memory1.save(filepath);

    // Loading instance has both a smaller capacity AND a swap file configured — the 2 excess
    // slots trimmed at load() time should land in swap rather than being discarded.
    HippocampalMemory memory2(d_model, 1, swap_path);
    memory2.load(filepath);

    ASSERT_EQ(memory2.size(), 1);
    EXPECT_TRUE(std::filesystem::exists(swap_path));

    // A fresh reader with spare capacity can recall both trimmed slots without re-evicting (see
    // RecallFromSwapDrainsToFalseOnceEmpty for why an at-capacity reader wouldn't drain cleanly).
    HippocampalMemory reader(d_model, 10, swap_path);
    EXPECT_TRUE(reader.recall_from_swap());
    EXPECT_TRUE(reader.recall_from_swap());
    EXPECT_FALSE(reader.recall_from_swap()) << "exactly 2 slots should have been trimmed to swap";

    std::remove(filepath.c_str());
    std::remove(swap_path.c_str());
}
