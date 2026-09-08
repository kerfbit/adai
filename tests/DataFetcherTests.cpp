/**
 * DataFetcherTests — unit tests for DataFetcher (TD-028 Phase 10)
 *
 * Only covers offline behaviour: config defaults, construction, and batch calls
 * with empty input that return immediately without touching the network.
 * Real Gutenberg / HuggingFace integration tests are left to manual / CI runs
 * that have network access.
 */
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../src/DataFetcher.hpp"

namespace fs = std::filesystem;

// ============================================================================
// FetcherConfig defaults
// ============================================================================

TEST(FetcherConfigTest, DefaultOutputDirs) {
    FetcherConfig cfg;
    EXPECT_EQ(cfg.gutenberg_output_dir, "gutenberg_data");
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
    cfg.gutenberg_output_dir = "/tmp/adai_guten_test";
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

// ============================================================================
// convert_huggingface_slice — offset/wraparound (Phase 12: rotating slices)
//
// Fully offline: feeds a hand-written fixture JSONL directly, bypassing
// ensure_huggingface_cached() (and therefore the network) entirely.
// ============================================================================

namespace {

class HuggingfaceSliceTest : public ::testing::Test {
   protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "adai_hf_slice_test";
        fs::remove_all(tmp_dir_);
        fs::create_directories(tmp_dir_);
        cached_jsonl_ = (tmp_dir_ / "full_dataset.jsonl").string();
        output_file_ = (tmp_dir_ / "slice_out.jsonl").string();
    }

    void TearDown() override {
        fs::remove_all(tmp_dir_);
    }

    // Writes n_rows key-value rows: {"instruction":"qN","output":"aN"}
    void write_fixture(int n_rows) {
        std::ofstream f(cached_jsonl_);
        for (int i = 0; i < n_rows; ++i) {
            f << "{\"instruction\":\"q" << i << "\",\"output\":\"a" << i << "\"}\n";
        }
    }

    // Reads output_file_ and returns the "input" values in order (via a
    // minimal ad-hoc scan — sample_to_jsonl's exact field name isn't the
    // point here, just that content from the right rows made it through).
    std::vector<std::string> read_output_inputs() const {
        std::vector<std::string> out;
        std::ifstream f(output_file_);
        std::string line;
        while (std::getline(f, line)) {
            const auto pos = line.find("\"input\":\"");
            if (pos == std::string::npos)
                continue;
            const auto start = pos + std::string("\"input\":\"").size();
            const auto end = line.find('"', start);
            out.push_back(line.substr(start, end - start));
        }
        return out;
    }

    fs::path tmp_dir_;
    std::string cached_jsonl_;
    std::string output_file_;
};

TEST_F(HuggingfaceSliceTest, FirstCallStartsAtRowZero) {
    write_fixture(10);
    int next_offset = -1;
    const int pairs = DataFetcher::convert_huggingface_slice(cached_jsonl_, output_file_, "", "",
                                                              /*offset_rows=*/0, /*num_pairs=*/3,
                                                              next_offset);
    EXPECT_EQ(pairs, 3);
    EXPECT_EQ(next_offset, 3);
    EXPECT_EQ(read_output_inputs(), (std::vector<std::string>{"q0", "q1", "q2"}));
}

TEST_F(HuggingfaceSliceTest, SecondCallContinuesFromCursor) {
    write_fixture(10);
    int next_offset = -1;
    const int pairs = DataFetcher::convert_huggingface_slice(cached_jsonl_, output_file_, "", "",
                                                              /*offset_rows=*/3, /*num_pairs=*/3,
                                                              next_offset);
    EXPECT_EQ(pairs, 3);
    EXPECT_EQ(next_offset, 6);
    EXPECT_EQ(read_output_inputs(), (std::vector<std::string>{"q3", "q4", "q5"}));
}

TEST_F(HuggingfaceSliceTest, WrapsAroundWhenDatasetExhaustedMidRequest) {
    write_fixture(10);
    int next_offset = -1;
    // Only q8,q9 remain from offset 8; requesting 5 should wrap to q0,q1,q2.
    const int pairs = DataFetcher::convert_huggingface_slice(cached_jsonl_, output_file_, "", "",
                                                              /*offset_rows=*/8, /*num_pairs=*/5,
                                                              next_offset);
    EXPECT_EQ(pairs, 5);
    EXPECT_EQ(next_offset, 3);  // wrap pass consumed rows 0,1,2
    EXPECT_EQ(read_output_inputs(),
             (std::vector<std::string>{"q8", "q9", "q0", "q1", "q2"}));
}

TEST_F(HuggingfaceSliceTest, OffsetAtEndOfDatasetWrapsImmediately) {
    write_fixture(10);
    int next_offset = -1;
    // offset == total row count: nothing left before EOF, must wrap fully.
    const int pairs = DataFetcher::convert_huggingface_slice(cached_jsonl_, output_file_, "", "",
                                                              /*offset_rows=*/10, /*num_pairs=*/3,
                                                              next_offset);
    EXPECT_EQ(pairs, 3);
    EXPECT_EQ(next_offset, 3);
    EXPECT_EQ(read_output_inputs(), (std::vector<std::string>{"q0", "q1", "q2"}));
}

TEST_F(HuggingfaceSliceTest, SingleWraparoundIsBoundedWhenDatasetSmallerThanRequest) {
    write_fixture(2);
    int next_offset = -1;
    // Dataset has only 2 rows; requesting 5 can produce at most 2 (first
    // pass) + 2 (one wraparound pass) = 4 — never an unbounded/infinite loop.
    const int pairs = DataFetcher::convert_huggingface_slice(cached_jsonl_, output_file_, "", "",
                                                              /*offset_rows=*/0, /*num_pairs=*/5,
                                                              next_offset);
    EXPECT_EQ(pairs, 4);
    EXPECT_EQ(next_offset, 2);
    EXPECT_EQ(read_output_inputs(), (std::vector<std::string>{"q0", "q1", "q0", "q1"}));
}

TEST_F(HuggingfaceSliceTest, ExactlySatisfiedFromOffsetDoesNotWrap) {
    write_fixture(10);
    int next_offset = -1;
    const int pairs = DataFetcher::convert_huggingface_slice(cached_jsonl_, output_file_, "", "",
                                                              /*offset_rows=*/7, /*num_pairs=*/3,
                                                              next_offset);
    EXPECT_EQ(pairs, 3);
    EXPECT_EQ(next_offset, 10);
    EXPECT_EQ(read_output_inputs(), (std::vector<std::string>{"q7", "q8", "q9"}));
}

TEST_F(HuggingfaceSliceTest, MissingCachedFileReturnsZeroPairs) {
    int next_offset = -1;
    const int pairs = DataFetcher::convert_huggingface_slice(
        (tmp_dir_ / "does_not_exist.jsonl").string(), output_file_, "", "",
        /*offset_rows=*/0, /*num_pairs=*/5, next_offset);
    EXPECT_EQ(pairs, 0);
}

// Regression test for a real production incident: SetFit/sst2 (and any
// dataset shaped like it — free text plus a *numeric* classification field
// that happens to share a name with an auto-detect candidate, here
// {"text","label"}) silently produced zero pairs for the entire dataset.
// hf_detect_fields() used to accept a candidate purely because both key
// names were present, so it locked onto "text"/"label" on row 0 and then
// never recovered: label's value is a bare JSON number (`"label":0`), which
// hf_extract_string() correctly refuses to treat as a string, so every
// subsequent row's extraction silently failed too — det_in/det_out are only
// (re)detected once, on the first row, so there was no path back to the
// working single-text-field fallback that TinyStories-shaped ({"text":...}
// only) datasets already use successfully.
TEST_F(HuggingfaceSliceTest, NumericFieldMatchingCandidateNameFallsBackToSingleText) {
    {
        std::ofstream f(cached_jsonl_);
        for (int i = 0; i < 5; ++i) {
            f << "{\"text\":\"This is a reasonably long piece of review text number " << i
              << ". It goes on for a while so the mid-sentence splitter has enough to work with.\","
                 "\"label\":"
              << (i % 2) << "}\n";
        }
    }
    int next_offset = -1;
    const int pairs = DataFetcher::convert_huggingface_slice(cached_jsonl_, output_file_, "", "",
                                                              /*offset_rows=*/0, /*num_pairs=*/5,
                                                              next_offset);
    EXPECT_GT(pairs, 0) << "should fall back to the single-text-field splitter instead of "
                           "silently detecting the numeric 'label' field as free-text output";
}

}  // namespace

// ============================================================================
// Gutenberg cleaning helpers (Phase 13) — strip_toc, strip_chapter_markers,
// strip_illustration_markers, is_standalone_chapter_marker_line,
// clean_gutenberg_text.
//
// These are private static members; DataFetcherGutenbergCleaningTest is
// declared a friend in DataFetcher.hpp specifically so this fixture can call
// them directly with hand-written fixture strings, fully offline.
// ============================================================================

// Note: friendship is not inherited by subclasses in C++, and gtest's
// TEST_F macro generates a subclass of the fixture per test — so the private
// DataFetcher calls must happen inside member functions of this exact class
// (which *is* the declared friend), not directly inside TEST_F bodies. These
// protected wrappers are that indirection.
class DataFetcherGutenbergCleaningTest : public ::testing::Test {
   protected:
    static bool IsStandaloneMarker(const std::string& s) {
        return DataFetcher::is_standalone_chapter_marker_line(s);
    }
    static std::string StripIllustrationMarkers(const std::string& s) {
        return DataFetcher::strip_illustration_markers(s);
    }
    static std::string StripChapterMarkers(const std::string& s) {
        return DataFetcher::strip_chapter_markers(s);
    }
    static std::string StripToc(const std::string& s) {
        return DataFetcher::strip_toc(s);
    }
    static std::string CleanGutenbergText(const std::string& s) {
        return DataFetcher::clean_gutenberg_text(s);
    }
    static std::vector<std::string> ExtractSentences(const std::string& s) {
        return DataFetcher::extract_sentences(s);
    }
};

TEST_F(DataFetcherGutenbergCleaningTest, IsStandaloneMarkerAcceptsBareMarkerLine) {
    EXPECT_TRUE(IsStandaloneMarker("CHAPTER I."));
    EXPECT_TRUE(IsStandaloneMarker("Chapter 12"));
    EXPECT_TRUE(IsStandaloneMarker("BOOK II"));
    EXPECT_TRUE(IsStandaloneMarker("part 3."));
}

TEST_F(DataFetcherGutenbergCleaningTest, IsStandaloneMarkerRejectsTocEntryWithTrailingTitle) {
    // TOC entries have the title trailing on the same line — must NOT match,
    // since this predicate is what distinguishes a TOC entry from the real
    // in-body marker (see strip_toc's use of it as the stop condition).
    EXPECT_FALSE(IsStandaloneMarker("CHAPTER I.     Down the Rabbit-Hole"));
}

TEST_F(DataFetcherGutenbergCleaningTest, IsStandaloneMarkerRejectsOrdinaryProse) {
    EXPECT_FALSE(IsStandaloneMarker("Alice was beginning to get very tired."));
    EXPECT_FALSE(IsStandaloneMarker(""));
}

TEST_F(DataFetcherGutenbergCleaningTest, StripIllustrationMarkersRemovesBracketedMarkers) {
    const std::string text =
        "Some text [Illustration] more text [Illustration: A caption here] end.\n"
        "[Figure 3] [Frontispiece] [Footnote: see below] done.";
    const std::string cleaned = StripIllustrationMarkers(text);
    EXPECT_EQ(cleaned.find("Illustration"), std::string::npos);
    EXPECT_EQ(cleaned.find("Figure 3"), std::string::npos);
    EXPECT_EQ(cleaned.find("Frontispiece"), std::string::npos);
    EXPECT_EQ(cleaned.find("Footnote"), std::string::npos);
    EXPECT_NE(cleaned.find("Some text"), std::string::npos);
    EXPECT_NE(cleaned.find("more text"), std::string::npos);
    EXPECT_NE(cleaned.find("done."), std::string::npos);
}

TEST_F(DataFetcherGutenbergCleaningTest, StripChapterMarkersRemovesMarkerAndTitleLine) {
    // Shaped like the real gutenberg.org/files/11/11-0.txt body marker.
    const std::string text =
        "CHAPTER I.\n"
        "Down the Rabbit-Hole\n"
        "\n"
        "\n"
        "Alice was beginning to get very tired of sitting by her sister on the\n"
        "bank, and of having nothing to do.\n";
    const std::string cleaned = StripChapterMarkers(text);
    EXPECT_EQ(cleaned.find("CHAPTER I."), std::string::npos);
    EXPECT_EQ(cleaned.find("Down the Rabbit-Hole"), std::string::npos);
    EXPECT_NE(cleaned.find("Alice was beginning"), std::string::npos);
}

TEST_F(DataFetcherGutenbergCleaningTest, StripChapterMarkersPreservesOrdinaryProse) {
    const std::string text = "This is just ordinary prose that mentions a chapter in passing.\n";
    EXPECT_EQ(StripChapterMarkers(text), text);
}

TEST_F(DataFetcherGutenbergCleaningTest, StripTocRemovesHeaderAndEntriesUpToRealMarker) {
    // Shaped like the real gutenberg.org/files/11/11-0.txt TOC block.
    const std::string text =
        "Contents\n"
        "\n"
        " CHAPTER I.     Down the Rabbit-Hole\n"
        " CHAPTER II.    The Pool of Tears\n"
        "\n"
        "\n"
        "CHAPTER I.\n"
        "Down the Rabbit-Hole\n"
        "\n"
        "Alice was beginning to get very tired.\n";
    const std::string cleaned = StripToc(text);
    EXPECT_EQ(cleaned.find("Contents"), std::string::npos);
    EXPECT_EQ(cleaned.find("The Pool of Tears"), std::string::npos);
    // strip_toc only removes the TOC block itself — the real standalone
    // marker line and everything after it (including its own title line,
    // which strip_chapter_markers is responsible for) must survive.
    EXPECT_NE(cleaned.find("CHAPTER I."), std::string::npos);
    EXPECT_NE(cleaned.find("Alice was beginning"), std::string::npos);
}

TEST_F(DataFetcherGutenbergCleaningTest, StripTocLeavesTextUntouchedWhenNoMarkerFound) {
    // No standalone chapter marker anywhere — strip_toc must not guess/over-strip.
    const std::string text =
        "Contents\n"
        "\n"
        "Some front matter that never resolves into a real chapter heading.\n";
    EXPECT_EQ(StripToc(text), text);
}

TEST_F(DataFetcherGutenbergCleaningTest, CleanGutenbergTextStripsEverythingTogether) {
    // End-to-end fixture combining license markers + illustration + TOC +
    // chapter marker/title + real prose, mirroring the real book excerpt
    // fetched from gutenberg.org/files/11/11-0.txt during design.
    const std::string raw =
        "Some Project Gutenberg License preamble text.\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK 11 ***\n"
        "\n"
        "[Illustration]\n"
        "\n"
        "Alice's Adventures in Wonderland\n"
        "\n"
        "Contents\n"
        "\n"
        " CHAPTER I.     Down the Rabbit-Hole\n"
        " CHAPTER II.    The Pool of Tears\n"
        "\n"
        "\n"
        "CHAPTER I.\n"
        "Down the Rabbit-Hole\n"
        "\n"
        "\n"
        "Alice was beginning to get very tired of sitting by her sister on the "
        "bank, and of having nothing to do: once or twice she had peeped into "
        "the book her sister was reading, but it had no pictures or "
        "conversations in it, and what is the use of a book, thought Alice, "
        "without pictures or conversations?\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK 11 ***\n"
        "Some license postamble text.\n";

    const std::string cleaned = CleanGutenbergText(raw);

    EXPECT_EQ(cleaned.find("License preamble"), std::string::npos);
    EXPECT_EQ(cleaned.find("license postamble"), std::string::npos);
    EXPECT_EQ(cleaned.find("Illustration"), std::string::npos);
    EXPECT_EQ(cleaned.find("Contents"), std::string::npos);
    EXPECT_EQ(cleaned.find("The Pool of Tears"), std::string::npos);
    EXPECT_EQ(cleaned.find("CHAPTER I."), std::string::npos);
    EXPECT_EQ(cleaned.find("Down the Rabbit-Hole"), std::string::npos);
    EXPECT_NE(cleaned.find("Alice was beginning"), std::string::npos);
    EXPECT_NE(cleaned.find("without pictures or conversations?"), std::string::npos);
}

TEST_F(DataFetcherGutenbergCleaningTest, ExtractSentencesDoesNotSplitOnDecimalNumbers) {
    // Regression test for a real bug found while verifying this change against
    // gutenberg.org/files/11/11-0.txt: "THE MILLENNIUM FULCRUM EDITION 3.0"
    // was splitting into "...EDITION 3." with a stray leading "0 " glued onto
    // the start of the next sentence.
    const std::string text =
        "This edition is THE MILLENNIUM FULCRUM EDITION 3.0 of the classic "
        "story. Alice was beginning to get very tired of sitting by her "
        "sister on the bank and having nothing at all to do today.";
    const auto sentences = ExtractSentences(text);
    ASSERT_FALSE(sentences.empty());
    for (const auto& s : sentences) {
        EXPECT_TRUE(s.empty() || s.front() != '0')
            << "sentence incorrectly starts with a stray digit from a decimal "
               "number: '"
            << s << "'";
    }
    // The decimal point itself must survive intact within its sentence.
    bool found_edition_sentence = false;
    for (const auto& s : sentences) {
        if (s.find("EDITION 3.0") != std::string::npos)
            found_edition_sentence = true;
    }
    EXPECT_TRUE(found_edition_sentence);
}

// ============================================================================
// convert_gutenberg_slice — offset/wraparound (Phase 13: rotating slices)
//
// Fully offline: feeds a hand-written fixture sentence file directly,
// bypassing ensure_gutenberg_cached() (and therefore the network) entirely.
//
// create_qa_pairs_from_text() interleaves two pair "shapes" per two-sentence
// step — a templated question (randomized wording via rand(), so its exact
// text isn't asserted here) paired with the very next sentence as the
// answer, and (when a third sentence is available) a "Summarize: ..." pair
// whose answer is the sentence after that. Both answer/response values are
// deterministic (verbatim sentence text), so tests assert on those.
// ============================================================================

namespace {

class GutenbergSliceTest : public ::testing::Test {
   protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "adai_gutenberg_slice_test";
        fs::remove_all(tmp_dir_);
        fs::create_directories(tmp_dir_);
        cached_sentences_ = (tmp_dir_ / "sentences.txt").string();
        output_file_ = (tmp_dir_ / "slice_out.jsonl").string();
    }

    void TearDown() override {
        fs::remove_all(tmp_dir_);
    }

    // Writes n sentences: "Sentence 0.", "Sentence 1.", ...
    void write_fixture(int n) {
        std::ofstream f(cached_sentences_);
        for (int i = 0; i < n; ++i) {
            f << "Sentence " << i << ".\n";
        }
    }

    // Reads output_file_ and returns the "response" values in order — these
    // are verbatim sentence text regardless of pair shape (question vs.
    // "Summarize:"), so they're a deterministic proxy for which sentences
    // were consumed.
    std::vector<std::string> read_output_responses() const {
        std::vector<std::string> out;
        std::ifstream f(output_file_);
        std::string line;
        while (std::getline(f, line)) {
            const auto pos = line.find("\"response\":\"");
            if (pos == std::string::npos)
                continue;
            const auto start = pos + std::string("\"response\":\"").size();
            const auto end = line.find('"', start);
            out.push_back(line.substr(start, end - start));
        }
        return out;
    }

    fs::path tmp_dir_;
    std::string cached_sentences_;
    std::string output_file_;
};

TEST_F(GutenbergSliceTest, FirstCallStartsAtSentenceZero) {
    write_fixture(10);
    int next_offset = -1;
    // Trace: i=0 -> A(s0->s1), B(summarize s0,s1->s2) [2 pairs]; i=2 -> A(s2->s3)
    // [3rd pair, cap reached]; next i=4.
    const int pairs =
        DataFetcher::convert_gutenberg_slice(cached_sentences_, output_file_,
                                             /*offset_index=*/0, /*num_pairs=*/3, next_offset);
    EXPECT_EQ(pairs, 3);
    EXPECT_EQ(next_offset, 4);
    EXPECT_EQ(read_output_responses(),
             (std::vector<std::string>{"Sentence 1.", "Sentence 2.", "Sentence 3."}));
}

TEST_F(GutenbergSliceTest, SecondCallContinuesFromCursor) {
    write_fixture(10);
    int next_offset = -1;
    // Trace from offset 4: i=4 -> A(s4->s5), B(summarize s4,s5->s6) [2 pairs];
    // i=6 -> A(s6->s7) [3rd pair, cap reached]; next i=8.
    const int pairs =
        DataFetcher::convert_gutenberg_slice(cached_sentences_, output_file_,
                                             /*offset_index=*/4, /*num_pairs=*/3, next_offset);
    EXPECT_EQ(pairs, 3);
    EXPECT_EQ(next_offset, 8);
    EXPECT_EQ(read_output_responses(),
             (std::vector<std::string>{"Sentence 5.", "Sentence 6.", "Sentence 7."}));
}

TEST_F(GutenbergSliceTest, WrapsAroundWhenBookExhaustedMidRequest) {
    write_fixture(10);
    int next_offset = -1;
    // Trace from offset 8 (sentences 8,9 only remain): i=8 -> A(s8->s9) [1
    // pair; i+2=10 is not < size()=10, so no "Summarize" pair]; loop exits
    // (i becomes 10). 1 < 3 requested, so wrap from 0: i=0 -> A(s0->s1),
    // B(summarize s0,s1->s2) [2 more pairs, satisfies remaining 2]; next i=2.
    const int pairs =
        DataFetcher::convert_gutenberg_slice(cached_sentences_, output_file_,
                                             /*offset_index=*/8, /*num_pairs=*/3, next_offset);
    EXPECT_EQ(pairs, 3);
    EXPECT_EQ(next_offset, 2);
    EXPECT_EQ(read_output_responses(),
             (std::vector<std::string>{"Sentence 9.", "Sentence 1.", "Sentence 2."}));
}

TEST_F(GutenbergSliceTest, OffsetAtEndOfBookWrapsImmediately) {
    write_fixture(10);
    int next_offset = -1;
    // offset == total sentence count: first pass makes zero progress (loop
    // condition false immediately), so this is a full wraparound: i=0 ->
    // A(s0->s1), B(summarize s0,s1->s2) [2 pairs, satisfies num_pairs=2];
    // next i=2.
    const int pairs =
        DataFetcher::convert_gutenberg_slice(cached_sentences_, output_file_,
                                             /*offset_index=*/10, /*num_pairs=*/2, next_offset);
    EXPECT_EQ(pairs, 2);
    EXPECT_EQ(next_offset, 2);
    EXPECT_EQ(read_output_responses(),
             (std::vector<std::string>{"Sentence 1.", "Sentence 2."}));
}

TEST_F(GutenbergSliceTest, MissingCachedFileReturnsZeroPairs) {
    int next_offset = -1;
    const int pairs = DataFetcher::convert_gutenberg_slice(
        (tmp_dir_ / "does_not_exist.txt").string(), output_file_,
        /*offset_index=*/0, /*num_pairs=*/5, next_offset);
    EXPECT_EQ(pairs, 0);
}

}  // namespace
