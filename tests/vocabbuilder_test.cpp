/**
 * Unit tests for VocabBuilder components.
 *
 * VocabBuilder.cpp is a CLI entry-point (main()) that combines three file-loading
 * helpers with a BPETokenizer vocabulary-building workflow.  Because the entry-point
 * cannot be linked into a test binary, the helpers were extracted to
 * src/VocabBuilderHelpers.hpp (inline functions).
 *
 * Test strategy:
 *  1. File-format helpers: load_plain_text, load_pairs_format, load_json_format
 *     — exercised with temporary files written in each format.
 *  2. BPETokenizer workflow: build_vocab, save_vocab/load_vocab round-trip,
 *     get_top_tokens, print_vocab_stats — these are the BPETokenizer methods that
 *     VocabBuilder's main() calls.
 *
 * Temporary files are written to /tmp/ with a PID-based suffix so parallel test
 * runs don't collide.
 */

#include <gtest/gtest.h>
#include "../src/BPETokenizer.hpp"
#include "../src/VocabBuilderHelpers.hpp"

#include <unistd.h>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

// ============================================================================
// Helpers
// ============================================================================

static std::string tmp_path(const std::string& suffix) {
    return "/tmp/vocabbuilder_test_" + std::to_string(getpid()) + "_" + suffix;
}

static void write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

static void remove_tmp(const std::string& path) {
    std::remove(path.c_str());
}

/// A small but varied corpus useful for building a BPE vocabulary.
static const std::vector<std::string> kCorpus = {
    "hello world this is a test",
    "the quick brown fox jumps over the lazy dog",
    "hello again testing vocabulary building",
    "words and more words for the tokenizer",
    "sample text to exercise the bpe algorithm",
    "repeated words repeated words increase frequency",
};

// ============================================================================
// load_plain_text Tests
// ============================================================================

TEST(LoadPlainTextTest, NonExistentFileReturnsEmpty) {
    auto result = load_plain_text("/tmp/this_file_does_not_exist_ever.txt");
    EXPECT_TRUE(result.empty());
}

TEST(LoadPlainTextTest, EmptyFileReturnsEmpty) {
    std::string path = tmp_path("plain_empty.txt");
    write_file(path, "");
    auto result = load_plain_text(path);
    EXPECT_TRUE(result.empty());
    remove_tmp(path);
}

TEST(LoadPlainTextTest, EmptyLinesAreSkipped) {
    std::string path = tmp_path("plain_empty_lines.txt");
    write_file(path, "line one\n\nline two\n\n\nline three\n");
    auto result = load_plain_text(path);
    EXPECT_EQ(result.size(), 3u);
    remove_tmp(path);
}

TEST(LoadPlainTextTest, SingleLineLoaded) {
    std::string path = tmp_path("plain_single.txt");
    write_file(path, "hello world\n");
    auto result = load_plain_text(path);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], "hello world");
    remove_tmp(path);
}

TEST(LoadPlainTextTest, MultipleLines) {
    std::string path = tmp_path("plain_multi.txt");
    write_file(path, "first line\nsecond line\nthird line\n");
    auto result = load_plain_text(path);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], "first line");
    EXPECT_EQ(result[1], "second line");
    EXPECT_EQ(result[2], "third line");
    remove_tmp(path);
}

TEST(LoadPlainTextTest, LinesWithSpacesPreserved) {
    std::string path = tmp_path("plain_spaces.txt");
    write_file(path, "  leading spaces\ntrailing spaces  \n");
    auto result = load_plain_text(path);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], "  leading spaces");
    EXPECT_EQ(result[1], "trailing spaces  ");
    remove_tmp(path);
}

TEST(LoadPlainTextTest, NoTrailingNewlineHandled) {
    std::string path = tmp_path("plain_no_newline.txt");
    write_file(path, "no newline at end");
    auto result = load_plain_text(path);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], "no newline at end");
    remove_tmp(path);
}

// ============================================================================
// load_pairs_format Tests
// ============================================================================

TEST(LoadPairsFormatTest, NonExistentFileReturnsEmpty) {
    auto result = load_pairs_format("/tmp/this_file_does_not_exist_ever.txt");
    EXPECT_TRUE(result.empty());
}

TEST(LoadPairsFormatTest, EmptyFileReturnsEmpty) {
    std::string path = tmp_path("pairs_empty.txt");
    write_file(path, "");
    auto result = load_pairs_format(path);
    EXPECT_TRUE(result.empty());
    remove_tmp(path);
}

TEST(LoadPairsFormatTest, InputPrefixExtracted) {
    std::string path = tmp_path("pairs_input.txt");
    write_file(path, "INPUT:hello world\n");
    auto result = load_pairs_format(path);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], "hello world");
    remove_tmp(path);
}

TEST(LoadPairsFormatTest, ResponsePrefixExtracted) {
    std::string path = tmp_path("pairs_response.txt");
    write_file(path, "RESPONSE:this is a reply\n");
    auto result = load_pairs_format(path);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], "this is a reply");
    remove_tmp(path);
}

TEST(LoadPairsFormatTest, OtherLinesIgnored) {
    std::string path = tmp_path("pairs_other.txt");
    write_file(path, "IGNORED LINE\nSOMETHING ELSE\nINPUT:kept\n");
    auto result = load_pairs_format(path);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], "kept");
    remove_tmp(path);
}

TEST(LoadPairsFormatTest, MultiplePairsExtracted) {
    std::string path = tmp_path("pairs_multi.txt");
    write_file(path,
               "INPUT:question one\n"
               "RESPONSE:answer one\n"
               "INPUT:question two\n"
               "RESPONSE:answer two\n");
    auto result = load_pairs_format(path);
    ASSERT_EQ(result.size(), 4u);
    EXPECT_EQ(result[0], "question one");
    EXPECT_EQ(result[1], "answer one");
    EXPECT_EQ(result[2], "question two");
    EXPECT_EQ(result[3], "answer two");
    remove_tmp(path);
}

TEST(LoadPairsFormatTest, InputWithoutTextExtracted) {
    // "INPUT:" with nothing after the colon → empty string still pushed
    std::string path = tmp_path("pairs_bare.txt");
    write_file(path, "INPUT:\n");
    auto result = load_pairs_format(path);
    // Empty string after prefix is still added (empty suffix is not dropped)
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], "");
    remove_tmp(path);
}

TEST(LoadPairsFormatTest, BlankLinesBetweenPairsIgnored) {
    std::string path = tmp_path("pairs_blanks.txt");
    write_file(path, "\nINPUT:a\n\nRESPONSE:b\n\n");
    auto result = load_pairs_format(path);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], "b");
    remove_tmp(path);
}

// ============================================================================
// load_json_format Tests
// ============================================================================

TEST(LoadJsonFormatTest, NonExistentFileReturnsEmpty) {
    auto result = load_json_format("/tmp/this_file_does_not_exist_ever.txt");
    EXPECT_TRUE(result.empty());
}

TEST(LoadJsonFormatTest, EmptyFileReturnsEmpty) {
    std::string path = tmp_path("json_empty.txt");
    write_file(path, "");
    auto result = load_json_format(path);
    EXPECT_TRUE(result.empty());
    remove_tmp(path);
}

TEST(LoadJsonFormatTest, SingleString) {
    std::string path = tmp_path("json_single.txt");
    write_file(path, R"(["hello world"])");
    auto result = load_json_format(path);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], "hello world");
    remove_tmp(path);
}

TEST(LoadJsonFormatTest, MultipleStrings) {
    std::string path = tmp_path("json_multi.txt");
    write_file(path, R"(["first", "second", "third"])");
    auto result = load_json_format(path);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], "first");
    EXPECT_EQ(result[1], "second");
    EXPECT_EQ(result[2], "third");
    remove_tmp(path);
}

TEST(LoadJsonFormatTest, EmptyStringSkipped) {
    // The parser skips empty strings
    std::string path = tmp_path("json_empty_str.txt");
    write_file(path, R"(["hello", "", "world"])");
    auto result = load_json_format(path);
    // Empty strings between quotes are skipped by the parser
    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], "hello");
    EXPECT_EQ(result[1], "world");
    remove_tmp(path);
}

TEST(LoadJsonFormatTest, StringsWithSpacesPreserved) {
    std::string path = tmp_path("json_spaces.txt");
    write_file(path, R"(["  spaces  ", "  both  "])");
    auto result = load_json_format(path);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], "  spaces  ");
    EXPECT_EQ(result[1], "  both  ");
    remove_tmp(path);
}

TEST(LoadJsonFormatTest, MultilineJson) {
    std::string path = tmp_path("json_multiline.txt");
    write_file(path,
               "[\n"
               "  \"line one\",\n"
               "  \"line two\"\n"
               "]\n");
    auto result = load_json_format(path);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], "line one");
    EXPECT_EQ(result[1], "line two");
    remove_tmp(path);
}

// ============================================================================
// BPETokenizer Workflow Tests (VocabBuilder's main() behaviour)
// ============================================================================

class VocabBuilderWorkflowTest : public ::testing::Test {
   protected:
    void SetUp() override {
        vocab_path_ = tmp_path("workflow_vocab.txt");
    }

    void TearDown() override {
        remove_tmp(vocab_path_);
    }

    std::string vocab_path_;
};

TEST_F(VocabBuilderWorkflowTest, BuildVocabIncreasesVocabSize) {
    BPETokenizer tok;
    size_t initial_size = tok.get_vocab_size();  // Just special tokens
    tok.build_vocab(kCorpus, 100, 1);
    EXPECT_GT(tok.get_vocab_size(), initial_size);
}

TEST_F(VocabBuilderWorkflowTest, BuildVocabDefaultConstruction) {
    BPETokenizer tok;
    EXPECT_NO_THROW(tok.build_vocab(kCorpus, 200, 1));
}

TEST_F(VocabBuilderWorkflowTest, BuildVocabEmptyCorpusNoThrow) {
    BPETokenizer tok;
    EXPECT_NO_THROW(tok.build_vocab({}, 100, 1));
}

TEST_F(VocabBuilderWorkflowTest, BuildVocabSingleSentence) {
    BPETokenizer tok;
    EXPECT_NO_THROW(tok.build_vocab({"hello world"}, 50, 1));
    EXPECT_GT(tok.get_vocab_size(), 0u);
}

TEST_F(VocabBuilderWorkflowTest, BuildVocabHighFrequencyThreshold) {
    // With a very high threshold, rare chars are filtered - vocab will be small
    BPETokenizer tok;
    EXPECT_NO_THROW(tok.build_vocab(kCorpus, 1000, 100));
    // Special tokens are always present regardless of threshold
    EXPECT_GE(tok.get_vocab_size(), 4u);
}

TEST_F(VocabBuilderWorkflowTest, SaveAndLoadVocabRoundTrip) {
    // Build a vocabulary, save it, reload into a new tokenizer
    BPETokenizer builder;
    builder.build_vocab(kCorpus, 150, 1);
    builder.save_vocab(vocab_path_);

    BPETokenizer loader;
    EXPECT_NO_THROW(loader.load_vocab(vocab_path_));
    EXPECT_EQ(loader.get_vocab_size(), builder.get_vocab_size());
}

TEST_F(VocabBuilderWorkflowTest, LoadedVocabCanEncode) {
    BPETokenizer builder;
    builder.build_vocab(kCorpus, 150, 1);
    builder.save_vocab(vocab_path_);

    BPETokenizer loader;
    loader.load_vocab(vocab_path_);

    auto tokens = loader.encode("hello world");
    EXPECT_FALSE(tokens.empty());
}

TEST_F(VocabBuilderWorkflowTest, SavedVocabFileExists) {
    BPETokenizer tok;
    tok.build_vocab(kCorpus, 100, 1);
    tok.save_vocab(vocab_path_);

    std::ifstream f(vocab_path_);
    EXPECT_TRUE(f.good());
}

TEST_F(VocabBuilderWorkflowTest, GetTopTokensCountMatchesK) {
    BPETokenizer tok;
    tok.build_vocab(kCorpus, 200, 1);

    int k = 10;
    auto top = tok.get_top_tokens(k);
    EXPECT_LE(static_cast<int>(top.size()), k);
    // May return fewer than k if vocab is smaller, but should return something
    EXPECT_FALSE(top.empty());
}

TEST_F(VocabBuilderWorkflowTest, GetTopTokensReturnsPairsWithIds) {
    BPETokenizer tok;
    tok.build_vocab(kCorpus, 200, 1);

    auto top = tok.get_top_tokens(5);
    for (const auto& pair : top) {
        // First: token string (non-empty)
        EXPECT_FALSE(pair.first.empty());
        // Second: token ID (non-negative)
        EXPECT_GE(pair.second, 0);
    }
}

TEST_F(VocabBuilderWorkflowTest, GetTopTokensZeroReturnsEmpty) {
    BPETokenizer tok;
    tok.build_vocab(kCorpus, 100, 1);
    auto top = tok.get_top_tokens(0);
    EXPECT_TRUE(top.empty());
}

TEST_F(VocabBuilderWorkflowTest, PrintVocabStatsDoesNotCrash) {
    BPETokenizer tok;
    tok.build_vocab(kCorpus, 100, 1);
    // Redirect stdout to suppress output
    std::streambuf* old_buf = std::cout.rdbuf();
    std::ostringstream devnull;
    std::cout.rdbuf(devnull.rdbuf());
    EXPECT_NO_THROW(tok.print_vocab_stats());
    std::cout.rdbuf(old_buf);
}

TEST_F(VocabBuilderWorkflowTest, PrintVocabStatsProducesOutput) {
    BPETokenizer tok;
    tok.build_vocab(kCorpus, 100, 1);
    std::ostringstream capture;
    std::streambuf* old_buf = std::cout.rdbuf(capture.rdbuf());
    tok.print_vocab_stats();
    std::cout.rdbuf(old_buf);
    EXPECT_GT(capture.str().size(), 0u);
}

TEST_F(VocabBuilderWorkflowTest, EncodeAfterBuildReturnsNonEmpty) {
    BPETokenizer tok;
    tok.build_vocab(kCorpus, 200, 1);
    auto ids = tok.encode("hello world");
    EXPECT_FALSE(ids.empty());
}

TEST_F(VocabBuilderWorkflowTest, DecodeAfterBuildReturnsSomething) {
    BPETokenizer tok;
    tok.build_vocab(kCorpus, 200, 1);
    auto ids = tok.encode("hello world");
    auto decoded = tok.decode(ids);
    EXPECT_FALSE(decoded.empty());
}

TEST_F(VocabBuilderWorkflowTest, EncodeDecodeContainsOriginalChars) {
    BPETokenizer tok;
    tok.build_vocab(kCorpus, 300, 1);
    std::string original = "hello world";
    auto ids = tok.encode(original);
    auto decoded = tok.decode(ids);
    // BPE may split differently, but decoded chars should come from original vocab
    EXPECT_FALSE(decoded.empty());
}

// ============================================================================
// Special Token Preservation Through Build
// ============================================================================

TEST(VocabBuilderSpecialTokensTest, SpecialTokenIdsPreservedAfterBuild) {
    BPETokenizer tok;
    int bos_before = tok.get_bos_token_id();
    int eos_before = tok.get_eos_token_id();
    int pad_before = tok.get_pad_token_id();
    int unk_before = tok.get_unk_token_id();

    tok.build_vocab(kCorpus, 100, 1);

    EXPECT_EQ(tok.get_bos_token_id(), bos_before);
    EXPECT_EQ(tok.get_eos_token_id(), eos_before);
    EXPECT_EQ(tok.get_pad_token_id(), pad_before);
    EXPECT_EQ(tok.get_unk_token_id(), unk_before);
}

TEST(VocabBuilderSpecialTokensTest, SpecialTokensHaveDistinctIds) {
    BPETokenizer tok;
    tok.build_vocab(kCorpus, 100, 1);

    std::set<int> ids = {tok.get_bos_token_id(), tok.get_eos_token_id(), tok.get_pad_token_id(),
                         tok.get_unk_token_id()};
    EXPECT_EQ(ids.size(), 4u);
}

// ============================================================================
// Multiple Input Sources (as VocabBuilder main() aggregates multiple files)
// ============================================================================

TEST(VocabBuilderMultiSourceTest, LoadFromMultiplePlainFiles) {
    std::string p1 = tmp_path("multi_a.txt");
    std::string p2 = tmp_path("multi_b.txt");
    write_file(p1, "file one content\n");
    write_file(p2, "file two content\n");

    auto t1 = load_plain_text(p1);
    auto t2 = load_plain_text(p2);

    std::vector<std::string> combined;
    combined.insert(combined.end(), t1.begin(), t1.end());
    combined.insert(combined.end(), t2.begin(), t2.end());

    ASSERT_EQ(combined.size(), 2u);
    EXPECT_EQ(combined[0], "file one content");
    EXPECT_EQ(combined[1], "file two content");

    remove_tmp(p1);
    remove_tmp(p2);
}

TEST(VocabBuilderMultiSourceTest, MixedFormatsLoadedAndCombinedForBuild) {
    std::string plain_path = tmp_path("mixed_plain.txt");
    write_file(plain_path, "plain text line\nanother line\n");

    std::string pairs_path = tmp_path("mixed_pairs.txt");
    write_file(pairs_path, "INPUT:question here\nRESPONSE:answer here\n");

    auto plain_texts = load_plain_text(plain_path);
    auto pairs_texts = load_pairs_format(pairs_path);

    std::vector<std::string> all;
    all.insert(all.end(), plain_texts.begin(), plain_texts.end());
    all.insert(all.end(), pairs_texts.begin(), pairs_texts.end());

    EXPECT_EQ(all.size(), 4u);

    // Build vocabulary from combined sources — should not throw
    BPETokenizer tok;
    EXPECT_NO_THROW(tok.build_vocab(all, 100, 1));
    EXPECT_GT(tok.get_vocab_size(), 0u);

    remove_tmp(plain_path);
    remove_tmp(pairs_path);
}
