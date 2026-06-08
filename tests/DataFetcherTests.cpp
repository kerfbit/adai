/**
 * DataFetcherTests — unit tests for DataFetcher (TD-028 Phase 10)
 *
 * Only covers offline behaviour: config defaults, construction, and batch calls
 * with empty input that return immediately without touching the network.
 * Real Gutenberg / HuggingFace integration tests are left to manual / CI runs
 * that have network access.
 */
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "../src/DataFetcher.hpp"

// ============================================================================
// FetcherConfig defaults
// ============================================================================

TEST(FetcherConfigTest, DefaultOutputDirs) {
    FetcherConfig cfg;
    EXPECT_EQ(cfg.gutenberg_output_dir,   "gutenberg_data");
    EXPECT_EQ(cfg.huggingface_output_dir, "huggingface_data");
}

// ============================================================================
// DataFetcher construction
// ============================================================================

TEST(DataFetcherTest, DefaultConstructionDoesNotCrash) {
    DataFetcher fetcher;
    (void)fetcher;
}

TEST(DataFetcherTest, CustomConfigConstructionDoesNotCrash) {
    FetcherConfig cfg;
    cfg.gutenberg_output_dir   = "/tmp/adai_guten_test";
    cfg.huggingface_output_dir = "/tmp/adai_hf_test";
    DataFetcher fetcher(cfg);
    (void)fetcher;
}

// ============================================================================
// Batch with empty input — no network calls
// ============================================================================

TEST(DataFetcherTest, FetchGutenbergBatchEmptyInputReturnsEmptyVector) {
    DataFetcher fetcher;
    const auto result = fetcher.fetch_gutenberg_batch({});
    EXPECT_TRUE(result.empty());
}
