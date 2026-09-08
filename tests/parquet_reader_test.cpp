/**
 * ParquetReaderTests — unit tests for the native Parquet -> JSONL reader
 * (see src/ParquetReader.hpp for why this exists: it replaces a Python
 * pandas/pyarrow subprocess that crashed with SIGILL on any CPU lacking
 * AVX2/AVX512).
 *
 * Fixtures are small, real .parquet files (not hand-typed fake bytes —
 * testing a byte-format parser against my own possibly-wrong assumptions
 * about the format would prove nothing) checked into
 * tests/fixtures/parquet/, generated once via the dev-only
 * generate_fixtures.py script in that directory. Assertions use the same
 * lightweight substring-scan style as DataFetcherTests.cpp's
 * HuggingfaceSliceTest, rather than pulling in a JSON parsing library.
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../src/ParquetReader.hpp"

namespace fs = std::filesystem;

namespace {

fs::path fixtures_dir() {
    return fs::path(__FILE__).parent_path() / "fixtures" / "parquet";
}

class ParquetReaderTest : public ::testing::Test {
   protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "adai_parquet_reader_test";
        fs::remove_all(tmp_dir_);
        fs::create_directories(tmp_dir_);
        out_path_ = (tmp_dir_ / "out.jsonl").string();
    }

    void TearDown() override { fs::remove_all(tmp_dir_); }

    std::vector<std::string> read_lines() const {
        std::vector<std::string> lines;
        std::ifstream f(out_path_);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty())
                lines.push_back(line);
        }
        return lines;
    }

    // Extracts the string value of "key":"..." from a JSON line, or "" if
    // the key isn't present as a quoted string (matches how
    // DataFetcher.cpp's own hf_extract_string treats a missing/non-string
    // field — including a bare `null`).
    static std::string field(const std::string& line, const std::string& key) {
        const std::string needle = "\"" + key + "\":\"";
        const auto pos = line.find(needle);
        if (pos == std::string::npos)
            return "";
        const auto start = pos + needle.size();
        const auto end = line.find('"', start);
        if (end == std::string::npos)
            return "";
        return line.substr(start, end - start);
    }

    static bool field_is_null(const std::string& line, const std::string& key) {
        return line.find("\"" + key + "\":null") != std::string::npos;
    }

    fs::path tmp_dir_;
    std::string out_path_;
};

}  // namespace

TEST_F(ParquetReaderTest, PlainUncompressed) {
    const std::string in = (fixtures_dir() / "plain_uncompressed.parquet").string();
    const long long rows = ParquetReader::convert_to_jsonl(in, out_path_, /*append=*/false);
    ASSERT_EQ(rows, 3);

    const auto lines = read_lines();
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(field(lines[0], "text"), "hello world");
    EXPECT_EQ(field(lines[1], "text"), "second row");
    EXPECT_EQ(field(lines[2], "text"), "third row here");
}

TEST_F(ParquetReaderTest, PlainSnappy) {
    const std::string in = (fixtures_dir() / "plain_snappy.parquet").string();
    const long long rows = ParquetReader::convert_to_jsonl(in, out_path_, /*append=*/false);
    ASSERT_EQ(rows, 3);

    const auto lines = read_lines();
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(field(lines[0], "text"), "hello world");
    EXPECT_EQ(field(lines[1], "text"), "second row");
    EXPECT_EQ(field(lines[2], "text"), "third row here");
}

// Matches real HuggingFace export shape: SNAPPY compression + RLE_DICTIONARY
// encoding — the highest-priority fixture.
TEST_F(ParquetReaderTest, DictionarySnappy) {
    const std::string in = (fixtures_dir() / "dict_snappy.parquet").string();
    const long long rows = ParquetReader::convert_to_jsonl(in, out_path_, /*append=*/false);
    ASSERT_EQ(rows, 120);

    const auto lines = read_lines();
    ASSERT_EQ(lines.size(), 120u);
    // Fixture pattern: {"apple","banana","apple","cherry","banana","apple"} repeated 20x.
    static const char* kExpected[] = {"apple", "banana", "apple", "cherry", "banana", "apple"};
    for (size_t i = 0; i < lines.size(); ++i) {
        EXPECT_EQ(field(lines[i], "text"), kExpected[i % 6]) << "row " << i;
    }
}

TEST_F(ParquetReaderTest, MultiColumnPreservesSchemaOrder) {
    const std::string in = (fixtures_dir() / "multi_column.parquet").string();
    const long long rows = ParquetReader::convert_to_jsonl(in, out_path_, /*append=*/false);
    ASSERT_EQ(rows, 30);

    const auto lines = read_lines();
    ASSERT_EQ(lines.size(), 30u);
    for (int i = 0; i < 30; ++i) {
        EXPECT_EQ(field(lines[i], "instruction"), "do thing " + std::to_string(i));
        EXPECT_EQ(field(lines[i], "output"), "result " + std::to_string(i));
        // instruction must appear before output on the line (schema order).
        EXPECT_LT(lines[i].find("\"instruction\""), lines[i].find("\"output\""));
    }
}

TEST_F(ParquetReaderTest, NullsEmitUnquotedJsonNull) {
    const std::string in = (fixtures_dir() / "with_nulls.parquet").string();
    const long long rows = ParquetReader::convert_to_jsonl(in, out_path_, /*append=*/false);
    ASSERT_EQ(rows, 40);

    const auto lines = read_lines();
    ASSERT_EQ(lines.size(), 40u);
    // Fixture pattern: {"a", null, "b", null, null, "c", "a", "b"} repeated 5x.
    static const char* kExpected[] = {"a", nullptr, "b", nullptr, nullptr, "c", "a", "b"};
    for (size_t i = 0; i < lines.size(); ++i) {
        if (kExpected[i % 8] == nullptr) {
            EXPECT_TRUE(field_is_null(lines[i], "text")) << "row " << i << ": " << lines[i];
        } else {
            EXPECT_EQ(field(lines[i], "text"), kExpected[i % 8]) << "row " << i;
        }
    }
}

// Regression guard for the "only reads row_groups[0]" bug class: this
// fixture has 4 row groups (50 rows each) in one small file.
TEST_F(ParquetReaderTest, MultipleRowGroupsAllRead) {
    const std::string in = (fixtures_dir() / "multi_row_group.parquet").string();
    const long long rows = ParquetReader::convert_to_jsonl(in, out_path_, /*append=*/false);
    ASSERT_EQ(rows, 200);

    const auto lines = read_lines();
    ASSERT_EQ(lines.size(), 200u);
    EXPECT_EQ(field(lines[0], "text"), "row 0");
    EXPECT_EQ(field(lines[49], "text"), "row 49");    // last row of group 0
    EXPECT_EQ(field(lines[50], "text"), "row 50");    // first row of group 1
    EXPECT_EQ(field(lines[149], "text"), "row 149");  // last row of group 2
    EXPECT_EQ(field(lines[150], "text"), "row 150");  // first row of group 3
    EXPECT_EQ(field(lines[199], "text"), "row 199");  // last row overall
}

TEST_F(ParquetReaderTest, AppendFalseTruncatesExistingFile) {
    {
        std::ofstream f(out_path_);
        f << "{\"stale\":\"data\"}\n";
    }
    const std::string in = (fixtures_dir() / "plain_uncompressed.parquet").string();
    const long long rows = ParquetReader::convert_to_jsonl(in, out_path_, /*append=*/false);
    ASSERT_EQ(rows, 3);

    const auto lines = read_lines();
    ASSERT_EQ(lines.size(), 3u);
    for (const auto& l : lines) {
        EXPECT_EQ(l.find("stale"), std::string::npos);
    }
}

TEST_F(ParquetReaderTest, AppendTrueAddsToExistingFile) {
    const std::string in = (fixtures_dir() / "plain_uncompressed.parquet").string();
    ASSERT_EQ(ParquetReader::convert_to_jsonl(in, out_path_, /*append=*/false), 3);
    ASSERT_EQ(ParquetReader::convert_to_jsonl(in, out_path_, /*append=*/true), 3);

    const auto lines = read_lines();
    ASSERT_EQ(lines.size(), 6u);
    EXPECT_EQ(field(lines[0], "text"), "hello world");
    EXPECT_EQ(field(lines[3], "text"), "hello world");
}

TEST_F(ParquetReaderTest, MissingFileReturnsNegativeOneAndTouchesNothing) {
    const long long rows =
        ParquetReader::convert_to_jsonl((fixtures_dir() / "does_not_exist.parquet").string(),
                                        out_path_, /*append=*/false);
    EXPECT_EQ(rows, -1);
    EXPECT_FALSE(fs::exists(out_path_));
}

TEST_F(ParquetReaderTest, CorruptFileReturnsNegativeOne) {
    const std::string bogus = (tmp_dir_ / "bogus.parquet").string();
    {
        std::ofstream f(bogus, std::ios::binary);
        f << "PAR1not a real parquet footer at allPAR1";
    }
    const long long rows = ParquetReader::convert_to_jsonl(bogus, out_path_, /*append=*/false);
    EXPECT_EQ(rows, -1);
}

// ============================================================================
// Real-data parity check — DISABLED by default (never runs in normal test
// runs or CI): converts the real roneneldan/TinyStories parquet parts and
// compares row-for-row against the reference full_dataset.jsonl produced by
// the old Python (pandas/pyarrow) pipeline this component replaces. Requires
// huggingface_data/roneneldan_TinyStories_train/ to exist locally (gitignored,
// populated by a real `dataset_manager huggingface roneneldan/TinyStories ...
// train` fetch) — self-skips otherwise. Run explicitly with:
//   ./parquetReaderTests --gtest_also_run_disabled_tests \
//       --gtest_filter=ParquetReaderRealDataTest.*
// ============================================================================

TEST(ParquetReaderRealDataTest, DISABLED_TinyStoriesPartsMatchPythonReference) {
    const fs::path base = fs::path(__FILE__).parent_path().parent_path() / "huggingface_data" /
                          "roneneldan_TinyStories_train";
    const fs::path reference = base / "full_dataset.jsonl";
    if (!fs::exists(reference)) {
        GTEST_SKIP() << "No local " << reference << " — skipping real-data parity check.";
    }

    const fs::path tmp_out = fs::temp_directory_path() / "adai_parquet_realdata_test.jsonl";
    std::error_code ec;
    fs::remove(tmp_out, ec);

    long long total_rows = 0;
    for (int i = 0; i < 5; ++i) {
        const fs::path part = base / "parquet" / ("part_" + std::to_string(i) + ".parquet");
        if (!fs::exists(part)) {
            GTEST_SKIP() << "Missing " << part << " — skipping real-data parity check.";
        }
        const long long rows =
            ParquetReader::convert_to_jsonl(part.string(), tmp_out.string(), /*append=*/i > 0);
        ASSERT_GE(rows, 0) << "part_" << i << " failed to convert";
        total_rows += rows;
    }

    auto extract_text = [](const std::string& line) -> std::string {
        const std::string needle = "\"text\":";
        auto pos = line.find(needle);
        if (pos == std::string::npos)
            return "";
        pos += needle.size();
        // Tolerate optional whitespace after the colon (pandas.to_json emits
        // "key": "value"; ParquetReader emits compact "key":"value" — both
        // valid JSON, and the real consumer hf_extract_string() in
        // DataFetcher.cpp already tolerates both).
        while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos])))
            ++pos;
        if (pos >= line.size() || line[pos] != '"')
            return "";
        ++pos;
        std::string out;
        for (; pos < line.size(); ++pos) {
            if (line[pos] == '"' && line[pos - 1] != '\\')
                break;
            out += line[pos];
        }
        return out;
    };

    std::ifstream ref_f(reference);
    std::ifstream out_f(tmp_out);
    std::string ref_line, out_line;
    long long line_no = 0;
    long long mismatches = 0;
    while (std::getline(ref_f, ref_line) && std::getline(out_f, out_line)) {
        ++line_no;
        if (extract_text(ref_line) != extract_text(out_line)) {
            ++mismatches;
            if (mismatches <= 5) {
                ADD_FAILURE() << "line " << line_no << " mismatch:\n  ref: "
                              << ref_line.substr(0, 200) << "\n  got: " << out_line.substr(0, 200);
            }
        }
    }
    const bool ref_exhausted = !std::getline(ref_f, ref_line);
    const bool out_exhausted = !std::getline(out_f, out_line);
    EXPECT_TRUE(ref_exhausted) << "reference has more lines than our output";
    EXPECT_TRUE(out_exhausted) << "our output has more lines than reference";
    EXPECT_EQ(mismatches, 0) << mismatches << " line(s) differed out of " << line_no;
    std::cout << "Compared " << line_no << " lines, " << mismatches << " mismatches, " << total_rows
              << " total rows converted.\n";
}
