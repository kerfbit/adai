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
#include <cstdint>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include "../src/ParquetReader.hpp"

namespace fs = std::filesystem;

namespace {

fs::path fixtures_dir() {
    // __FILE__ bakes in this machine's compile-time absolute source path -- correct for a
    // normal ctest run (the source tree is always right there next to the build), but breaks
    // the instant this test binary is packaged and run somewhere else without that exact path
    // existing (confirmed on ai-machine's GPU test package, which ships a flat bin/ with no
    // source tree at all). ADAI_TEST_FIXTURES_DIR lets a relocated package point this at
    // wherever it actually placed the fixtures, without changing behavior for the normal
    // in-tree ctest case.
    if (const char* override_dir = std::getenv("ADAI_TEST_FIXTURES_DIR");
        override_dir != nullptr && *override_dir != '\0') {
        return fs::path(override_dir);
    }
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
// TD-470: crafted footers fail cleanly instead of hanging or crashing
// ============================================================================

namespace {

// "PAR1" + footer + little-endian footer length + "PAR1".
std::string write_parquet_with_footer(const fs::path& dir, const std::string& name,
                                      const std::string& footer) {
    const std::string path = (dir / name).string();
    std::ofstream f(path, std::ios::binary);
    f << "PAR1" << footer;
    const auto len = static_cast<std::uint32_t>(footer.size());
    for (int b = 0; b < 4; ++b)
        f.put(static_cast<char>((len >> (8 * b)) & 0xFF));
    f << "PAR1";
    return path;
}

}  // namespace

// An unknown FileMetaData field (delta 15, LIST) holding a bool list whose varint size is 2^62.
// Skipping a bool list element consumed no bytes, so this tiny file spun at 100% CPU forever.
TEST_F(ParquetReaderTest, HugeBoolListInFooterFailsInsteadOfLooping) {
    std::string footer;
    footer += static_cast<char>(0xF9);  // field delta 15, type 9 (LIST)
    footer += static_cast<char>(0xF1);  // list: size escape (15), element type 1 (BOOL)
    for (int i = 0; i < 8; ++i)
        footer += static_cast<char>(0x80);  // varint 2^62 ...
    footer += static_cast<char>(0x40);      // ... last byte
    footer += static_cast<char>(0x00);      // STOP
    const std::string path = write_parquet_with_footer(tmp_dir_, "bool_list.parquet", footer);
    EXPECT_LT(fs::file_size(path), 32u);
    EXPECT_EQ(ParquetReader::convert_to_jsonl(path, out_path_, /*append=*/false), -1);
}

// An unknown FileMetaData field (delta 15, STRUCT) containing 400,000 nested structs. Skipping
// recursed once per level and overflowed the stack (SIGSEGV), killing registry_server.
TEST_F(ParquetReaderTest, DeeplyNestedFooterFailsInsteadOfCrashing) {
    const size_t depth = 400000;
    std::string footer;
    footer += static_cast<char>(0xFC);              // field delta 15, type 12 (STRUCT)
    footer.append(depth, static_cast<char>(0x1C));  // each opens a nested struct (delta 1)
    footer.append(depth + 2, '\0');                 // close them all, then FileMetaData's STOP
    const std::string path = write_parquet_with_footer(tmp_dir_, "nested.parquet", footer);
    EXPECT_EQ(ParquetReader::convert_to_jsonl(path, out_path_, /*append=*/false), -1);
}

// ============================================================================
// TD-471: declared sizes are checked before anything is allocated
// ============================================================================

namespace {

// Just enough of a Thrift compact-protocol writer to build a one-column Parquet file.
struct ThriftWriter {
    std::string out;
    std::vector<int> last_ids{0};

    void varint(std::uint64_t v) {
        while (v >= 0x80) {
            out += static_cast<char>((v & 0x7F) | 0x80);
            v >>= 7;
        }
        out += static_cast<char>(v);
    }
    void zigzag(std::int64_t v) {
        varint((static_cast<std::uint64_t>(v) << 1) ^ static_cast<std::uint64_t>(v >> 63));
    }
    void field(int id, int type) {
        out += static_cast<char>(((id - last_ids.back()) << 4) | type);
        last_ids.back() = id;
    }
    void i32(int id, std::int64_t v) { field(id, 5), zigzag(v); }
    void i64(int id, std::int64_t v) { field(id, 6), zigzag(v); }
    void binary(int id, const std::string& v) { field(id, 8), varint(v.size()), out += v; }
    void begin_struct(int id) { field(id, 12), last_ids.push_back(0); }
    void begin_list_element() { last_ids.push_back(0); }
    void end_struct() { out += '\0', last_ids.pop_back(); }
    void list(int id, int elem_type, int size) {
        field(id, 9);
        out += static_cast<char>((size << 4) | elem_type);
    }
};

struct SyntheticColumn {
    std::int64_t row_group_rows = 3;  // RowGroup.num_rows
    std::int64_t column_values = 3;   // ColumnMetaData.num_values
    std::int32_t page_values = 3;     // DataPageHeader.num_values
    std::int32_t compressed_page_size = 12;
    std::int32_t uncompressed_page_size = 12;
};

// One REQUIRED INT32 column "x" holding 1, 2, 3: PLAIN, uncompressed, one data page.
std::string write_synthetic_parquet(const fs::path& dir, const std::string& name,
                                    const SyntheticColumn& col) {
    ThriftWriter page;
    page.i32(1, 0);  // type DATA_PAGE
    page.i32(2, col.uncompressed_page_size);
    page.i32(3, col.compressed_page_size);
    page.begin_struct(5);  // data_page_header
    page.i32(1, col.page_values);
    page.i32(2, 0);  // PLAIN
    page.i32(3, 3);  // definition levels: RLE
    page.i32(4, 3);  // repetition levels: RLE
    page.end_struct();
    page.out += '\0';
    std::string body;
    for (std::uint32_t v : {1u, 2u, 3u})
        for (int b = 0; b < 4; ++b)
            body += static_cast<char>((v >> (8 * b)) & 0xFF);

    ThriftWriter meta;
    meta.i32(1, 1);       // version
    meta.list(2, 12, 2);  // schema
    meta.begin_list_element();
    meta.binary(4, "schema");
    meta.i32(5, 1);  // num_children
    meta.end_struct();
    meta.begin_list_element();
    meta.i32(1, 1);  // INT32
    meta.i32(3, 0);  // REQUIRED
    meta.binary(4, "x");
    meta.end_struct();
    meta.i64(3, col.row_group_rows);
    meta.list(4, 12, 1);  // row_groups
    meta.begin_list_element();
    meta.list(1, 12, 1);  // columns
    meta.begin_list_element();
    meta.i64(2, 4);         // file_offset
    meta.begin_struct(3);   // meta_data
    meta.i32(1, 1);         // INT32
    meta.list(2, 5, 1);     // encodings
    meta.zigzag(0);         // PLAIN
    meta.list(3, 8, 1);     // path_in_schema
    meta.varint(1);
    meta.out += 'x';
    meta.i32(4, 0);  // UNCOMPRESSED
    meta.i64(5, col.column_values);
    meta.i64(6, static_cast<std::int64_t>(page.out.size() + body.size()));
    meta.i64(7, static_cast<std::int64_t>(page.out.size() + body.size()));
    meta.i64(9, 4);  // data_page_offset, right after "PAR1"
    meta.end_struct();
    meta.end_struct();  // ColumnChunk
    meta.i64(2, static_cast<std::int64_t>(body.size()));
    meta.i64(3, col.row_group_rows);
    meta.end_struct();  // RowGroup
    meta.out += '\0';

    const std::string path = (dir / name).string();
    std::ofstream f(path, std::ios::binary);
    f << "PAR1" << page.out << body << meta.out;
    const auto len = static_cast<std::uint32_t>(meta.out.size());
    for (int b = 0; b < 4; ++b)
        f.put(static_cast<char>((len >> (8 * b)) & 0xFF));
    f << "PAR1";
    return path;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace

// The control for the tests below: the synthetic file itself is valid.
TEST_F(ParquetReaderTest, SyntheticColumnFileConverts) {
    const std::string path = write_synthetic_parquet(tmp_dir_, "ok.parquet", {});
    ASSERT_EQ(ParquetReader::convert_to_jsonl(path, out_path_, /*append=*/false), 3);
    EXPECT_EQ(read_lines(), (std::vector<std::string>{"{\"x\":1}", "{\"x\":2}", "{\"x\":3}"}));
}

// A row group claiming 2^31 rows made decode_column_chunk() reserve 2^31 cells (about 150 GB),
// and the std::length_error escaped convert_to_jsonl().
TEST_F(ParquetReaderTest, RowGroupDeclaringBillionsOfRowsIsRejected) {
    SyntheticColumn col;
    col.row_group_rows = col.column_values = std::int64_t{1} << 31;
    const std::string path = write_synthetic_parquet(tmp_dir_, "rows.parquet", col);
    EXPECT_EQ(ParquetReader::convert_to_jsonl(path, out_path_, /*append=*/false), -1);
}

// A REQUIRED column whose 12-byte page claims 2^31 - 1 values: the old reader filled 8 GB of
// definition levels before finding the values missing.
TEST_F(ParquetReaderTest, PageClaimingMoreValuesThanItsColumnIsRejected) {
    SyntheticColumn col;
    col.page_values = 0x7FFFFFFF;
    const std::string path = write_synthetic_parquet(tmp_dir_, "page.parquet", col);
    EXPECT_EQ(ParquetReader::convert_to_jsonl(path, out_path_, /*append=*/false), -1);
}

// A negative compressed page size became a huge size_t in string::resize() (std::length_error).
TEST_F(ParquetReaderTest, NegativeOrOversizedPageSizeIsRejected) {
    for (std::int32_t size : {-1, 1 << 30}) {
        SyntheticColumn col;
        col.compressed_page_size = size;
        const std::string path = write_synthetic_parquet(tmp_dir_, "size.parquet", col);
        EXPECT_EQ(ParquetReader::convert_to_jsonl(path, out_path_, /*append=*/false), -1) << size;
    }
}

// Random byte mutations of every fixture: convert_to_jsonl() never throws, and a failure leaves
// the output exactly as it was, both when appending and when replacing. Before TD-471, 62 of
// 25,000 such mutations escaped as std::length_error, and a failure in a later row group kept
// the earlier row groups' rows.
TEST_F(ParquetReaderTest, MutatedFixturesNeverThrowAndFailuresLeaveOutputUnchanged) {
    const std::vector<std::string> fixtures = {
        "plain_uncompressed.parquet", "plain_snappy.parquet", "dict_snappy.parquet",
        "multi_column.parquet",       "with_nulls.parquet",   "multi_row_group.parquet"};
    const std::string original_output = "{\"keep\":true}\n";
    const std::string mutated_path = (tmp_dir_ / "mutated.parquet").string();
    std::uint64_t rng = 0x9E3779B97F4A7C15ull;
    auto next = [&rng]() {
        rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
        return rng;
    };
    int failures = 0, mutations = 0;
    for (const std::string& name : fixtures) {
        const std::string original = read_file((fixtures_dir() / name).string());
        ASSERT_FALSE(original.empty()) << name;
        for (int trial = 0; trial < 1500; ++trial) {
            std::string bytes = original;
            const int edits = 1 + static_cast<int>(next() % 4);
            for (int e = 0; e < edits; ++e) {
                const std::size_t at = 4 + next() % (bytes.size() - 12);  // keep magic + length
                const std::uint64_t pick = next();
                bytes[at] = static_cast<char>(pick % 3 == 0 ? 0xFF : pick % 3 == 1 ? 0x80 : pick >> 8);
            }
            {
                std::ofstream f(mutated_path, std::ios::binary | std::ios::trunc);
                f << bytes;
            }
            for (bool append : {true, false}) {
                {
                    std::ofstream f(out_path_, std::ios::binary | std::ios::trunc);
                    f << original_output;
                }
                long long rows = 0;
                ASSERT_NO_THROW(rows = ParquetReader::convert_to_jsonl(mutated_path, out_path_, append))
                    << name << " trial " << trial;
                ++mutations;
                if (rows < 0) {
                    ++failures;
                    ASSERT_EQ(read_file(out_path_), original_output)
                        << name << " trial " << trial << (append ? " (append)" : " (replace)");
                }
            }
        }
    }
    EXPECT_GT(failures, mutations / 10);  // the mutations really do hit the parser
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
