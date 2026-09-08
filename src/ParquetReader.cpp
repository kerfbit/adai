// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-08

#include "ParquetReader.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <vector>
#include "Logger.hpp"

using adai::Logger;

// ============================================================================
// This file implements just enough of the Parquet file format — Thrift
// compact-protocol footer/page-header parsing, raw-block Snappy
// decompression, and the RLE/bit-packed hybrid encoding — to read what real
// HuggingFace `datasets-server` parquet exports actually use. See
// ParquetReader.hpp for the exact supported/unsupported scope. Every helper
// below is file-local (anonymous namespace), matching DataFetcher.cpp's own
// convention of not exposing internal helpers as class members.
// ============================================================================

namespace {

// ============================================================================
// Parquet format enums (parquet.thrift / parquet.format) — exact integer
// values, verified against a real decoded file footer this session.
// ============================================================================

enum class PhysicalType : std::int32_t {
    BOOLEAN = 0,
    INT32 = 1,
    INT64 = 2,
    INT96 = 3,
    FLOAT = 4,
    DOUBLE = 5,
    BYTE_ARRAY = 6,
    FIXED_LEN_BYTE_ARRAY = 7,
};

enum class FieldRepetitionType : std::int32_t {
    REQUIRED = 0,
    OPTIONAL = 1,
    REPEATED = 2,
};

enum class Encoding : std::int32_t {
    PLAIN = 0,
    PLAIN_DICTIONARY = 2,
    RLE = 3,
    BIT_PACKED = 4,
    DELTA_BINARY_PACKED = 5,
    DELTA_LENGTH_BYTE_ARRAY = 6,
    DELTA_BYTE_ARRAY = 7,
    RLE_DICTIONARY = 8,
    BYTE_STREAM_SPLIT = 9,
};

enum class CompressionCodec : std::int32_t {
    UNCOMPRESSED = 0,
    SNAPPY = 1,
    GZIP = 2,
    LZO = 3,
    BROTLI = 4,
    LZ4 = 5,
    ZSTD = 6,
    LZ4_RAW = 7,
};

enum class PageType : std::int32_t {
    DATA_PAGE = 0,
    INDEX_PAGE = 1,
    DICTIONARY_PAGE = 2,
    DATA_PAGE_V2 = 3,
};

// ============================================================================
// Thrift compact-protocol primitives — not a general Thrift library, just
// enough to walk FileMetaData/SchemaElement/RowGroup/ColumnChunk/
// ColumnMetaData/PageHeader/DataPageHeader/DictionaryPageHeader.
// ============================================================================

struct Cursor {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t pos = 0;
};

bool read_byte(Cursor& c, std::uint8_t& out) {
    if (c.pos >= c.size)
        return false;
    out = c.data[c.pos++];
    return true;
}

// Unsigned LEB128 varint (7 bits/byte, MSB = continuation), max 10 bytes for 64 bits.
bool read_varint(Cursor& c, std::uint64_t& out) {
    out = 0;
    for (int i = 0; i < 10; ++i) {
        std::uint8_t b;
        if (!read_byte(c, b))
            return false;
        out |= static_cast<std::uint64_t>(b & 0x7F) << (7 * i);
        if ((b & 0x80) == 0)
            return true;
    }
    return false;  // malformed: too long
}

bool read_zigzag_varint(Cursor& c, std::int64_t& out) {
    std::uint64_t v;
    if (!read_varint(c, v))
        return false;
    out = static_cast<std::int64_t>(v >> 1) ^ -static_cast<std::int64_t>(v & 1);
    return true;
}

bool read_binary(Cursor& c, std::string& out) {
    std::uint64_t len;
    if (!read_varint(c, len))
        return false;
    if (len > c.size - c.pos)  // c.pos <= c.size always holds, so this can't underflow
        return false;
    out.assign(reinterpret_cast<const char*>(c.data + c.pos), static_cast<std::size_t>(len));
    c.pos += static_cast<std::size_t>(len);
    return true;
}

bool read_double(Cursor& c, double& out) {
    if (c.pos + 8 > c.size)
        return false;
    std::uint64_t bits = 0;
    for (int i = 0; i < 8; ++i)
        bits |= static_cast<std::uint64_t>(c.data[c.pos + i]) << (8 * i);
    c.pos += 8;
    std::memcpy(&out, &bits, sizeof(out));
    return true;
}

// Reads a list/set header: one byte (or, for size>=15, one byte + a size varint).
bool read_list_header(Cursor& c, int& elem_type, std::uint64_t& size) {
    std::uint8_t h;
    if (!read_byte(c, h))
        return false;
    elem_type = h & 0x0F;
    size = (h >> 4) & 0x0F;
    if (size == 15) {
        std::uint64_t extra;
        if (!read_varint(c, extra))
            return false;
        size = extra;
    }
    return true;
}

// Skips one value of the given compact-protocol type ID, recursing into
// nested structs/lists. Makes every struct field we don't explicitly parse
// safely ignorable instead of corrupting the rest of the parse.
bool skip_value(Cursor& c, int compact_type) {
    switch (compact_type) {
        case 1:
        case 2:  // BOOLEAN_TRUE / BOOLEAN_FALSE — value is in the type nibble, 0 bytes
            return true;
        case 3: {  // BYTE / I8
            std::uint8_t b;
            return read_byte(c, b);
        }
        case 4:
        case 5:
        case 6: {  // I16 / I32 / I64
            std::int64_t v;
            return read_zigzag_varint(c, v);
        }
        case 7: {  // DOUBLE
            double d;
            return read_double(c, d);
        }
        case 8: {  // BINARY / STRING
            std::string s;
            return read_binary(c, s);
        }
        case 9:
        case 10: {  // LIST / SET
            int elem_type;
            std::uint64_t size;
            if (!read_list_header(c, elem_type, size))
                return false;
            for (std::uint64_t i = 0; i < size; ++i) {
                if (!skip_value(c, elem_type))
                    return false;
            }
            return true;
        }
        case 12: {  // STRUCT
            int last_field_id = 0;
            while (true) {
                std::uint8_t b;
                if (!read_byte(c, b))
                    return false;
                if (b == 0x00)
                    break;  // STOP
                int type = b & 0x0F;
                int delta = (b >> 4) & 0x0F;
                if (delta != 0) {
                    last_field_id += delta;
                } else {
                    std::int64_t zz;
                    if (!read_zigzag_varint(c, zz))
                        return false;
                    last_field_id = static_cast<int>(zz);
                }
                if (!skip_value(c, type))
                    return false;
            }
            return true;
        }
        default:
            // MAP(11) and anything else: Parquet's own structs never emit these.
            return false;
    }
}

struct FieldHeader {
    int id = 0;
    int type = 0;
    bool is_stop = false;
};

bool next_field(Cursor& c, int& last_field_id, FieldHeader& out) {
    std::uint8_t b;
    if (!read_byte(c, b))
        return false;
    if (b == 0x00) {
        out.is_stop = true;
        return true;
    }
    out.is_stop = false;
    out.type = b & 0x0F;
    int delta = (b >> 4) & 0x0F;
    if (delta != 0) {
        out.id = last_field_id + delta;
    } else {
        std::int64_t zz;
        if (!read_zigzag_varint(c, zz))
            return false;
        out.id = static_cast<int>(zz);
    }
    last_field_id = out.id;
    return true;
}

// ============================================================================
// Parquet metadata structs — only the fields this reader actually needs.
// ============================================================================

struct SchemaElement {
    bool has_type = false;
    PhysicalType type = PhysicalType::BYTE_ARRAY;
    bool has_repetition = false;
    FieldRepetitionType repetition = FieldRepetitionType::REQUIRED;
    std::string name;
};

struct ColumnMetaData {
    PhysicalType type = PhysicalType::BYTE_ARRAY;
    std::vector<std::int32_t> encodings;
    std::vector<std::string> path_in_schema;
    CompressionCodec codec = CompressionCodec::UNCOMPRESSED;
    std::int64_t num_values = 0;
    std::int64_t data_page_offset = -1;
    bool has_dictionary_page_offset = false;
    std::int64_t dictionary_page_offset = -1;
};

struct ColumnChunk {
    bool has_file_path = false;        // field 1 present => external chunk reference, unsupported
    bool has_crypto_metadata = false;  // fields 8/9 present => encrypted column, unsupported
    bool has_meta_data = false;
    ColumnMetaData meta_data;
};

struct RowGroup {
    std::vector<ColumnChunk> columns;
    std::int64_t num_rows = 0;
};

struct FileMetaData {
    std::vector<SchemaElement> schema;
    std::vector<RowGroup> row_groups;
};

struct PageHeader {
    PageType type = PageType::DATA_PAGE;
    std::int32_t uncompressed_page_size = 0;
    std::int32_t compressed_page_size = 0;
    bool has_data_page_header = false;
    std::int32_t dp_num_values = 0;
    std::int32_t dp_encoding = 0;
    bool has_dictionary_page_header = false;
    std::int32_t dict_num_values = 0;
    bool has_data_page_header_v2 = false;  // presence alone => unsupported
};

// ============================================================================
// Thrift struct parsers
// ============================================================================

bool parse_schema_element(Cursor& c, SchemaElement& out) {
    int last_field_id = 0;
    while (true) {
        FieldHeader fh;
        if (!next_field(c, last_field_id, fh))
            return false;
        if (fh.is_stop)
            break;
        switch (fh.id) {
            case 1: {  // type
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.type = static_cast<PhysicalType>(v);
                out.has_type = true;
                break;
            }
            case 3: {  // repetition_type
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.repetition = static_cast<FieldRepetitionType>(v);
                out.has_repetition = true;
                break;
            }
            case 4: {  // name
                if (!read_binary(c, out.name))
                    return false;
                break;
            }
            default:
                if (!skip_value(c, fh.type))
                    return false;
        }
    }
    return true;
}

bool parse_column_metadata(Cursor& c, ColumnMetaData& out) {
    int last_field_id = 0;
    while (true) {
        FieldHeader fh;
        if (!next_field(c, last_field_id, fh))
            return false;
        if (fh.is_stop)
            break;
        switch (fh.id) {
            case 1: {  // type
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.type = static_cast<PhysicalType>(v);
                break;
            }
            case 2: {  // encodings: list<i32>
                int elem_type;
                std::uint64_t size;
                if (!read_list_header(c, elem_type, size))
                    return false;
                out.encodings.clear();
                for (std::uint64_t i = 0; i < size; ++i) {
                    if (elem_type == 5) {  // I32
                        std::int64_t v;
                        if (!read_zigzag_varint(c, v))
                            return false;
                        out.encodings.push_back(static_cast<std::int32_t>(v));
                    } else if (!skip_value(c, elem_type)) {
                        return false;
                    }
                }
                break;
            }
            case 3: {  // path_in_schema: list<string>
                int elem_type;
                std::uint64_t size;
                if (!read_list_header(c, elem_type, size))
                    return false;
                out.path_in_schema.clear();
                for (std::uint64_t i = 0; i < size; ++i) {
                    if (elem_type == 8) {  // BINARY/STRING
                        std::string s;
                        if (!read_binary(c, s))
                            return false;
                        out.path_in_schema.push_back(std::move(s));
                    } else if (!skip_value(c, elem_type)) {
                        return false;
                    }
                }
                break;
            }
            case 4: {  // codec
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.codec = static_cast<CompressionCodec>(v);
                break;
            }
            case 5: {  // num_values
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.num_values = v;
                break;
            }
            case 9: {  // data_page_offset
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.data_page_offset = v;
                break;
            }
            case 11: {  // dictionary_page_offset
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.dictionary_page_offset = v;
                out.has_dictionary_page_offset = true;
                break;
            }
            default:
                if (!skip_value(c, fh.type))
                    return false;
        }
    }
    return true;
}

bool parse_column_chunk(Cursor& c, ColumnChunk& out) {
    int last_field_id = 0;
    while (true) {
        FieldHeader fh;
        if (!next_field(c, last_field_id, fh))
            return false;
        if (fh.is_stop)
            break;
        switch (fh.id) {
            case 1: {  // file_path — presence => external chunk reference, unsupported
                std::string s;
                if (!read_binary(c, s))
                    return false;
                out.has_file_path = true;
                break;
            }
            case 3: {  // meta_data
                if (!parse_column_metadata(c, out.meta_data))
                    return false;
                out.has_meta_data = true;
                break;
            }
            case 8: {  // crypto_metadata — presence indicates column encryption
                if (!skip_value(c, fh.type))
                    return false;
                out.has_crypto_metadata = true;
                break;
            }
            case 9: {  // encrypted_column_metadata
                std::string s;
                if (!read_binary(c, s))
                    return false;
                out.has_crypto_metadata = true;
                break;
            }
            default:
                if (!skip_value(c, fh.type))
                    return false;
        }
    }
    return true;
}

bool parse_row_group(Cursor& c, RowGroup& out) {
    int last_field_id = 0;
    while (true) {
        FieldHeader fh;
        if (!next_field(c, last_field_id, fh))
            return false;
        if (fh.is_stop)
            break;
        switch (fh.id) {
            case 1: {  // columns: list<ColumnChunk>
                int elem_type;
                std::uint64_t size;
                if (!read_list_header(c, elem_type, size))
                    return false;
                out.columns.clear();
                out.columns.reserve(size);
                for (std::uint64_t i = 0; i < size; ++i) {
                    if (elem_type == 12) {  // STRUCT
                        ColumnChunk cc;
                        if (!parse_column_chunk(c, cc))
                            return false;
                        out.columns.push_back(std::move(cc));
                    } else if (!skip_value(c, elem_type)) {
                        return false;
                    }
                }
                break;
            }
            case 3: {  // num_rows
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.num_rows = v;
                break;
            }
            default:
                if (!skip_value(c, fh.type))
                    return false;
        }
    }
    return true;
}

bool parse_file_metadata(Cursor& c, FileMetaData& out) {
    int last_field_id = 0;
    while (true) {
        FieldHeader fh;
        if (!next_field(c, last_field_id, fh))
            return false;
        if (fh.is_stop)
            break;
        switch (fh.id) {
            case 2: {  // schema: list<SchemaElement>
                int elem_type;
                std::uint64_t size;
                if (!read_list_header(c, elem_type, size))
                    return false;
                out.schema.clear();
                out.schema.reserve(size);
                for (std::uint64_t i = 0; i < size; ++i) {
                    if (elem_type == 12) {
                        SchemaElement se;
                        if (!parse_schema_element(c, se))
                            return false;
                        out.schema.push_back(std::move(se));
                    } else if (!skip_value(c, elem_type)) {
                        return false;
                    }
                }
                break;
            }
            case 4: {  // row_groups: list<RowGroup>
                int elem_type;
                std::uint64_t size;
                if (!read_list_header(c, elem_type, size))
                    return false;
                out.row_groups.clear();
                out.row_groups.reserve(size);
                for (std::uint64_t i = 0; i < size; ++i) {
                    if (elem_type == 12) {
                        RowGroup rg;
                        if (!parse_row_group(c, rg))
                            return false;
                        out.row_groups.push_back(std::move(rg));
                    } else if (!skip_value(c, elem_type)) {
                        return false;
                    }
                }
                break;
            }
            case 8:  // encryption_algorithm — presence => encrypted file, unsupported
                return false;
            default:
                if (!skip_value(c, fh.type))
                    return false;
        }
    }
    return true;
}

bool parse_page_header(Cursor& c, PageHeader& out) {
    int last_field_id = 0;
    while (true) {
        FieldHeader fh;
        if (!next_field(c, last_field_id, fh))
            return false;
        if (fh.is_stop)
            break;
        switch (fh.id) {
            case 1: {  // type
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.type = static_cast<PageType>(v);
                break;
            }
            case 2: {  // uncompressed_page_size
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.uncompressed_page_size = static_cast<std::int32_t>(v);
                break;
            }
            case 3: {  // compressed_page_size
                std::int64_t v;
                if (!read_zigzag_varint(c, v))
                    return false;
                out.compressed_page_size = static_cast<std::int32_t>(v);
                break;
            }
            case 5: {  // data_page_header
                int inner_last = 0;
                while (true) {
                    FieldHeader ifh;
                    if (!next_field(c, inner_last, ifh))
                        return false;
                    if (ifh.is_stop)
                        break;
                    if (ifh.id == 1) {  // num_values
                        std::int64_t v;
                        if (!read_zigzag_varint(c, v))
                            return false;
                        out.dp_num_values = static_cast<std::int32_t>(v);
                    } else if (ifh.id == 2) {  // encoding
                        std::int64_t v;
                        if (!read_zigzag_varint(c, v))
                            return false;
                        out.dp_encoding = static_cast<std::int32_t>(v);
                    } else if (!skip_value(c, ifh.type)) {
                        return false;
                    }
                }
                out.has_data_page_header = true;
                break;
            }
            case 7: {  // dictionary_page_header
                int inner_last = 0;
                while (true) {
                    FieldHeader ifh;
                    if (!next_field(c, inner_last, ifh))
                        return false;
                    if (ifh.is_stop)
                        break;
                    if (ifh.id == 1) {  // num_values
                        std::int64_t v;
                        if (!read_zigzag_varint(c, v))
                            return false;
                        out.dict_num_values = static_cast<std::int32_t>(v);
                    } else if (!skip_value(c, ifh.type)) {
                        return false;  // includes encoding (2, unused) and is_sorted (3)
                    }
                }
                out.has_dictionary_page_header = true;
                break;
            }
            case 8:  // data_page_header_v2 — presence alone => unsupported
                if (!skip_value(c, fh.type))
                    return false;
                out.has_data_page_header_v2 = true;
                break;
            default:
                if (!skip_value(c, fh.type))
                    return false;
        }
    }
    return true;
}

// ============================================================================
// Snappy raw-block decompression (snappy::RawUncompress's format — Parquet
// never uses the framed/streaming format). Decompression only.
// ============================================================================

bool snappy_raw_uncompress(const std::uint8_t* src, std::size_t src_len, std::string& out) {
    Cursor c{src, src_len, 0};
    std::uint64_t uncompressed_len;
    if (!read_varint(c, uncompressed_len))
        return false;
    out.clear();
    out.reserve(static_cast<std::size_t>(uncompressed_len));

    while (out.size() < uncompressed_len) {
        std::uint8_t tag;
        if (!read_byte(c, tag))
            return false;
        int type = tag & 0x03;

        if (type == 0) {  // literal
            std::size_t len_field = tag >> 2;
            std::size_t lit_len;
            if (len_field < 60) {
                lit_len = len_field + 1;
            } else {
                int extra_bytes = static_cast<int>(len_field) - 59;  // 1..4
                std::uint32_t v = 0;
                for (int i = 0; i < extra_bytes; ++i) {
                    std::uint8_t b;
                    if (!read_byte(c, b))
                        return false;
                    v |= static_cast<std::uint32_t>(b) << (8 * i);
                }
                lit_len = static_cast<std::size_t>(v) + 1;
            }
            if (c.pos + lit_len > c.size)
                return false;
            if (out.size() + lit_len > uncompressed_len)
                return false;
            out.append(reinterpret_cast<const char*>(c.data + c.pos), lit_len);
            c.pos += lit_len;
            continue;
        }

        std::size_t length;
        std::size_t offset;
        if (type == 1) {  // 1-byte offset copy
            length = ((tag >> 2) & 0x07) + 4;
            std::uint8_t b0;
            if (!read_byte(c, b0))
                return false;
            offset = (static_cast<std::size_t>((tag >> 5) & 0x07) << 8) | b0;
        } else if (type == 2) {  // 2-byte offset copy
            length = (tag >> 2) + 1;
            if (c.pos + 2 > c.size)
                return false;
            offset = static_cast<std::size_t>(c.data[c.pos]) |
                     (static_cast<std::size_t>(c.data[c.pos + 1]) << 8);
            c.pos += 2;
        } else {  // type == 3, 4-byte offset copy
            length = (tag >> 2) + 1;
            if (c.pos + 4 > c.size)
                return false;
            offset = static_cast<std::size_t>(c.data[c.pos]) |
                     (static_cast<std::size_t>(c.data[c.pos + 1]) << 8) |
                     (static_cast<std::size_t>(c.data[c.pos + 2]) << 16) |
                     (static_cast<std::size_t>(c.data[c.pos + 3]) << 24);
            c.pos += 4;
        }
        if (offset == 0 || offset > out.size())
            return false;
        if (out.size() + length > uncompressed_len)
            return false;
        // Self-overlapping copies (offset < length) are valid and produce
        // repeating patterns — must copy byte-by-byte, not memcpy/append in bulk.
        std::size_t start = out.size() - offset;
        out.reserve(out.size() + length);
        for (std::size_t i = 0; i < length; ++i)
            out.push_back(out[start + i]);
    }
    return out.size() == uncompressed_len;
}

// ============================================================================
// RLE / bit-packed hybrid decoder — used for definition levels and for
// dictionary-index streams.
// ============================================================================

// Bits needed to represent values 0..max_level inclusive (Parquet's own
// definition-level bit-width rule). 0 for max_level<=0.
int bit_width_for_max_level(int max_level) {
    if (max_level <= 0)
        return 0;
    int width = 0;
    while ((1 << width) <= max_level)
        ++width;
    return width;
}

bool decode_rle_bitpacked_hybrid(const std::uint8_t* data, std::size_t len, int bit_width,
                                 std::size_t num_values, std::vector<std::uint32_t>& out) {
    out.clear();
    out.reserve(num_values);
    if (bit_width == 0) {
        out.assign(num_values, 0);
        return true;
    }
    if (bit_width < 0 || bit_width > 32)
        return false;

    Cursor c{data, len, 0};
    const int byte_width = (bit_width + 7) / 8;

    while (out.size() < num_values) {
        std::uint64_t run_header;
        if (!read_varint(c, run_header))
            return false;

        if ((run_header & 1) == 0) {
            // RLE run: run_header>>1 repetitions of one byte_width-byte little-endian value.
            std::uint64_t run_length = run_header >> 1;
            if (c.pos + static_cast<std::size_t>(byte_width) > c.size)
                return false;
            std::uint32_t value = 0;
            for (int i = 0; i < byte_width; ++i)
                value |= static_cast<std::uint32_t>(c.data[c.pos + i]) << (8 * i);
            c.pos += static_cast<std::size_t>(byte_width);
            std::uint64_t to_emit = std::min<std::uint64_t>(run_length, num_values - out.size());
            for (std::uint64_t i = 0; i < to_emit; ++i)
                out.push_back(value);
        } else {
            // Bit-packed run: run_header>>1 groups of 8 values, LSB-first packing.
            std::uint64_t num_groups = run_header >> 1;
            std::uint64_t total_bytes = num_groups * static_cast<std::uint64_t>(bit_width);
            if (c.pos + total_bytes > c.size)
                return false;
            const std::size_t run_end = c.pos + static_cast<std::size_t>(total_bytes);
            const std::uint64_t values_in_run = num_groups * 8;
            const std::uint32_t mask =
                (bit_width == 32) ? 0xFFFFFFFFu : ((1u << bit_width) - 1u);
            std::uint64_t bit_buffer = 0;
            int bits_in_buffer = 0;
            for (std::uint64_t i = 0; i < values_in_run; ++i) {
                while (bits_in_buffer < bit_width) {
                    bit_buffer |= static_cast<std::uint64_t>(c.data[c.pos]) << bits_in_buffer;
                    ++c.pos;
                    bits_in_buffer += 8;
                }
                std::uint32_t value = static_cast<std::uint32_t>(bit_buffer & mask);
                bit_buffer >>= bit_width;
                bits_in_buffer -= bit_width;
                if (out.size() < num_values)
                    out.push_back(value);
                // else: trailing pad in the final group, discard.
            }
            c.pos = run_end;  // defensive re-sync; should already land here exactly
        }
    }
    return out.size() == num_values;
}

// ============================================================================
// Value decoding
// ============================================================================

struct ColumnValue {
    enum class Kind { STRING, INT64, DOUBLE, BOOLEAN } kind = Kind::STRING;
    std::string str_val;
    std::int64_t int_val = 0;
    double dbl_val = 0.0;
    bool bool_val = false;
};

using Cell = std::optional<ColumnValue>;

// Decodes exactly `count` PLAIN-encoded values of `type`, appending to `out`.
bool decode_plain_values(const std::uint8_t* data, std::size_t len, PhysicalType type,
                         std::size_t count, std::vector<ColumnValue>& out) {
    Cursor c{data, len, 0};
    out.clear();
    out.reserve(count);

    switch (type) {
        case PhysicalType::BOOLEAN: {
            std::size_t needed = (count + 7) / 8;
            if (needed > c.size)
                return false;
            for (std::size_t i = 0; i < count; ++i) {
                std::size_t byte_off = i / 8;
                int bit_off = static_cast<int>(i % 8);
                bool bit = ((c.data[byte_off] >> bit_off) & 1) != 0;
                ColumnValue v;
                v.kind = ColumnValue::Kind::BOOLEAN;
                v.bool_val = bit;
                out.push_back(std::move(v));
            }
            return true;
        }
        case PhysicalType::INT32: {
            for (std::size_t i = 0; i < count; ++i) {
                if (c.pos + 4 > c.size)
                    return false;
                std::uint32_t bits = 0;
                for (int b = 0; b < 4; ++b)
                    bits |= static_cast<std::uint32_t>(c.data[c.pos + b]) << (8 * b);
                c.pos += 4;
                std::int32_t iv;
                std::memcpy(&iv, &bits, sizeof(iv));
                ColumnValue v;
                v.kind = ColumnValue::Kind::INT64;
                v.int_val = iv;
                out.push_back(std::move(v));
            }
            return true;
        }
        case PhysicalType::INT64: {
            for (std::size_t i = 0; i < count; ++i) {
                if (c.pos + 8 > c.size)
                    return false;
                std::uint64_t bits = 0;
                for (int b = 0; b < 8; ++b)
                    bits |= static_cast<std::uint64_t>(c.data[c.pos + b]) << (8 * b);
                c.pos += 8;
                std::int64_t iv;
                std::memcpy(&iv, &bits, sizeof(iv));
                ColumnValue v;
                v.kind = ColumnValue::Kind::INT64;
                v.int_val = iv;
                out.push_back(std::move(v));
            }
            return true;
        }
        case PhysicalType::FLOAT: {
            for (std::size_t i = 0; i < count; ++i) {
                if (c.pos + 4 > c.size)
                    return false;
                std::uint32_t bits = 0;
                for (int b = 0; b < 4; ++b)
                    bits |= static_cast<std::uint32_t>(c.data[c.pos + b]) << (8 * b);
                c.pos += 4;
                float fv;
                std::memcpy(&fv, &bits, sizeof(fv));
                ColumnValue v;
                v.kind = ColumnValue::Kind::DOUBLE;
                v.dbl_val = static_cast<double>(fv);
                out.push_back(std::move(v));
            }
            return true;
        }
        case PhysicalType::DOUBLE: {
            for (std::size_t i = 0; i < count; ++i) {
                if (c.pos + 8 > c.size)
                    return false;
                std::uint64_t bits = 0;
                for (int b = 0; b < 8; ++b)
                    bits |= static_cast<std::uint64_t>(c.data[c.pos + b]) << (8 * b);
                c.pos += 8;
                double dv;
                std::memcpy(&dv, &bits, sizeof(dv));
                ColumnValue v;
                v.kind = ColumnValue::Kind::DOUBLE;
                v.dbl_val = dv;
                out.push_back(std::move(v));
            }
            return true;
        }
        case PhysicalType::BYTE_ARRAY: {
            for (std::size_t i = 0; i < count; ++i) {
                if (c.pos + 4 > c.size)
                    return false;
                std::uint32_t slen = 0;
                for (int b = 0; b < 4; ++b)
                    slen |= static_cast<std::uint32_t>(c.data[c.pos + b]) << (8 * b);
                c.pos += 4;
                if (c.pos + slen > c.size)
                    return false;
                ColumnValue v;
                v.kind = ColumnValue::Kind::STRING;
                v.str_val.assign(reinterpret_cast<const char*>(c.data + c.pos), slen);
                c.pos += slen;
                out.push_back(std::move(v));
            }
            return true;
        }
        case PhysicalType::INT96:
        case PhysicalType::FIXED_LEN_BYTE_ARRAY:
        default:
            Logger::error("ParquetReader: unsupported PLAIN physical type {}",
                         static_cast<int>(type));
            return false;
    }
}

// ============================================================================
// Column-chunk decode orchestration
// ============================================================================

// Parses a Thrift PageHeader starting at `offset` by reading a bounded probe
// buffer (page headers are always tiny — a few dozen bytes — so 8KB is
// generous headroom) and reports how many bytes it actually consumed, so the
// caller knows exactly where the page body begins.
bool read_page_header_at(std::ifstream& file, std::int64_t offset, PageHeader& header,
                         std::int64_t& body_offset) {
    constexpr std::size_t kProbeSize = 8192;
    file.clear();
    file.seekg(offset, std::ios::beg);
    if (!file.good())
        return false;
    std::vector<std::uint8_t> probe(kProbeSize);
    file.read(reinterpret_cast<char*>(probe.data()), static_cast<std::streamsize>(probe.size()));
    std::streamsize got = file.gcount();
    if (got <= 0)
        return false;
    probe.resize(static_cast<std::size_t>(got));

    Cursor c{probe.data(), probe.size(), 0};
    if (!parse_page_header(c, header))
        return false;
    body_offset = offset + static_cast<std::int64_t>(c.pos);
    return true;
}

bool read_page_body(std::ifstream& file, std::int64_t body_offset, std::int32_t compressed_size,
                    CompressionCodec codec, const std::string& path_for_errors,
                    const char* page_kind, std::string& uncompressed) {
    std::string compressed;
    file.clear();
    file.seekg(body_offset, std::ios::beg);
    compressed.resize(static_cast<std::size_t>(compressed_size));
    file.read(&compressed[0], compressed_size);
    if (file.gcount() != compressed_size) {
        Logger::error("ParquetReader: short read on {} page in '{}'", page_kind, path_for_errors);
        return false;
    }
    if (codec == CompressionCodec::UNCOMPRESSED) {
        uncompressed = std::move(compressed);
        return true;
    }
    if (codec == CompressionCodec::SNAPPY) {
        if (!snappy_raw_uncompress(reinterpret_cast<const std::uint8_t*>(compressed.data()),
                                   compressed.size(), uncompressed)) {
            Logger::error("ParquetReader: Snappy decompression failed ({} page) in '{}'", page_kind,
                         path_for_errors);
            return false;
        }
        return true;
    }
    Logger::error("ParquetReader: unsupported compression codec {} in '{}'",
                 static_cast<int>(codec), path_for_errors);
    return false;
}

bool decode_column_chunk(std::ifstream& file, const std::string& path_for_errors,
                         const ColumnChunk& chunk, bool is_optional,
                         std::int64_t expected_num_rows, std::vector<Cell>& out) {
    out.clear();
    out.reserve(static_cast<std::size_t>(std::max<std::int64_t>(expected_num_rows, 0)));

    const ColumnMetaData& md = chunk.meta_data;
    const int max_def_level = is_optional ? 1 : 0;
    const int def_bit_width = bit_width_for_max_level(max_def_level);

    std::vector<ColumnValue> dictionary;
    bool have_dictionary = false;
    std::int64_t cursor_offset =
        md.has_dictionary_page_offset ? md.dictionary_page_offset : md.data_page_offset;

    if (md.has_dictionary_page_offset) {
        PageHeader dph;
        std::int64_t body_offset;
        if (!read_page_header_at(file, cursor_offset, dph, body_offset)) {
            Logger::error("ParquetReader: failed to read dictionary page header in '{}'",
                         path_for_errors);
            return false;
        }
        if (dph.type != PageType::DICTIONARY_PAGE || !dph.has_dictionary_page_header) {
            Logger::error("ParquetReader: expected DICTIONARY_PAGE at offset {} in '{}'",
                         cursor_offset, path_for_errors);
            return false;
        }
        std::string uncompressed;
        if (!read_page_body(file, body_offset, dph.compressed_page_size, md.codec, path_for_errors,
                            "dictionary", uncompressed))
            return false;
        if (!decode_plain_values(reinterpret_cast<const std::uint8_t*>(uncompressed.data()),
                                 uncompressed.size(), md.type,
                                 static_cast<std::size_t>(dph.dict_num_values), dictionary)) {
            Logger::error("ParquetReader: failed to decode dictionary page values in '{}'",
                         path_for_errors);
            return false;
        }
        have_dictionary = true;
        cursor_offset = body_offset + dph.compressed_page_size;
    }

    std::int64_t values_read = 0;
    while (values_read < md.num_values) {
        PageHeader ph;
        std::int64_t body_offset;
        if (!read_page_header_at(file, cursor_offset, ph, body_offset)) {
            Logger::error("ParquetReader: failed to read data page header in '{}'", path_for_errors);
            return false;
        }
        if (ph.has_data_page_header_v2) {
            Logger::error("ParquetReader: DataPageV2 is not supported (column chunk in '{}')",
                         path_for_errors);
            return false;
        }
        if (ph.type != PageType::DATA_PAGE || !ph.has_data_page_header) {
            Logger::error("ParquetReader: expected DATA_PAGE at offset {} in '{}'", cursor_offset,
                         path_for_errors);
            return false;
        }

        std::string page_str;
        if (!read_page_body(file, body_offset, ph.compressed_page_size, md.codec, path_for_errors,
                            "data", page_str))
            return false;

        const std::uint8_t* page = reinterpret_cast<const std::uint8_t*>(page_str.data());
        const std::size_t page_len = page_str.size();
        std::size_t page_pos = 0;
        const std::size_t page_num_values = static_cast<std::size_t>(ph.dp_num_values);

        std::vector<std::uint32_t> def_levels;
        if (max_def_level > 0) {
            if (page_pos + 4 > page_len) {
                Logger::error("ParquetReader: truncated definition-level section in '{}'",
                             path_for_errors);
                return false;
            }
            std::uint32_t level_bytes = 0;
            for (int b = 0; b < 4; ++b)
                level_bytes |= static_cast<std::uint32_t>(page[page_pos + b]) << (8 * b);
            page_pos += 4;
            if (page_pos + level_bytes > page_len) {
                Logger::error("ParquetReader: truncated definition-level bytes in '{}'",
                             path_for_errors);
                return false;
            }
            if (!decode_rle_bitpacked_hybrid(page + page_pos, level_bytes, def_bit_width,
                                             page_num_values, def_levels)) {
                Logger::error("ParquetReader: failed to decode definition levels in '{}'",
                             path_for_errors);
                return false;
            }
            page_pos += level_bytes;
        } else {
            def_levels.assign(page_num_values, 0);  // REQUIRED: every row is present
        }

        std::size_t non_null_count = 0;
        for (std::uint32_t lvl : def_levels) {
            if (static_cast<int>(lvl) == max_def_level)
                ++non_null_count;
        }

        std::vector<ColumnValue> values;
        const std::uint8_t* value_data = page + page_pos;
        const std::size_t value_len = page_len - page_pos;

        if (ph.dp_encoding == static_cast<std::int32_t>(Encoding::PLAIN)) {
            if (!decode_plain_values(value_data, value_len, md.type, non_null_count, values)) {
                Logger::error("ParquetReader: failed to decode PLAIN values in '{}'",
                             path_for_errors);
                return false;
            }
        } else if (ph.dp_encoding == static_cast<std::int32_t>(Encoding::RLE_DICTIONARY) ||
                  ph.dp_encoding == static_cast<std::int32_t>(Encoding::PLAIN_DICTIONARY)) {
            if (!have_dictionary) {
                Logger::error(
                    "ParquetReader: dictionary-encoded data page with no dictionary page in '{}'",
                    path_for_errors);
                return false;
            }
            if (value_len < 1) {
                Logger::error("ParquetReader: truncated dictionary-index bit-width byte in '{}'",
                             path_for_errors);
                return false;
            }
            int idx_bit_width = value_data[0];
            std::vector<std::uint32_t> indices;
            if (!decode_rle_bitpacked_hybrid(value_data + 1, value_len - 1, idx_bit_width,
                                             non_null_count, indices)) {
                Logger::error("ParquetReader: failed to decode dictionary indices in '{}'",
                             path_for_errors);
                return false;
            }
            values.reserve(non_null_count);
            for (std::uint32_t idx : indices) {
                if (idx >= dictionary.size()) {
                    Logger::error(
                        "ParquetReader: dictionary index {} out of range ({} entries) in '{}'", idx,
                        dictionary.size(), path_for_errors);
                    return false;
                }
                values.push_back(dictionary[idx]);
            }
        } else {
            Logger::error("ParquetReader: unsupported page encoding {} in '{}'", ph.dp_encoding,
                         path_for_errors);
            return false;
        }

        if (values.size() != non_null_count) {
            Logger::error("ParquetReader: value count mismatch in '{}' (expected {}, got {})",
                         path_for_errors, non_null_count, values.size());
            return false;
        }

        std::size_t value_idx = 0;
        for (std::uint32_t lvl : def_levels) {
            if (static_cast<int>(lvl) == max_def_level) {
                out.push_back(Cell(values[value_idx++]));
            } else {
                out.push_back(std::nullopt);
            }
        }

        values_read += ph.dp_num_values;
        cursor_offset = body_offset + ph.compressed_page_size;
    }

    if (values_read != md.num_values) {
        Logger::error("ParquetReader: column value count mismatch in '{}' (expected {}, decoded {})",
                     path_for_errors, md.num_values, values_read);
        return false;
    }
    if (static_cast<std::int64_t>(out.size()) != expected_num_rows) {
        Logger::error("ParquetReader: row count mismatch in '{}' (row group declares {}, decoded {})",
                     path_for_errors, expected_num_rows, out.size());
        return false;
    }
    return true;
}

// ============================================================================
// JSON emission
// ============================================================================

void json_escape_append(const std::string& s, std::string& out) {
    for (unsigned char ch : s) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (ch < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out += buf;
                } else {
                    out += static_cast<char>(ch);
                }
        }
    }
}

void append_cell_json(const Cell& cell, std::string& out) {
    if (!cell.has_value()) {
        out += "null";
        return;
    }
    const ColumnValue& v = *cell;
    switch (v.kind) {
        case ColumnValue::Kind::STRING:
            out += '"';
            json_escape_append(v.str_val, out);
            out += '"';
            break;
        case ColumnValue::Kind::INT64:
            out += std::to_string(v.int_val);
            break;
        case ColumnValue::Kind::DOUBLE: {
            std::ostringstream oss;
            oss << v.dbl_val;
            out += oss.str();
            break;
        }
        case ColumnValue::Kind::BOOLEAN:
            out += v.bool_val ? "true" : "false";
            break;
    }
}

}  // namespace

// ============================================================================
// ParquetReader::convert_to_jsonl
// ============================================================================

long long ParquetReader::convert_to_jsonl(const std::string& parquet_path,
                                          const std::string& out_jsonl_path, bool append) {
    std::ifstream file(parquet_path, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("ParquetReader: cannot open '{}'", parquet_path);
        return -1;
    }

    file.seekg(0, std::ios::end);
    std::streamoff file_size = file.tellg();
    if (file_size < 12) {  // magic(4) + footer_len(4) + magic(4), minimum for an empty file
        Logger::error("ParquetReader: '{}' is too small to be a valid Parquet file", parquet_path);
        return -1;
    }

    char head[4];
    file.seekg(0, std::ios::beg);
    file.read(head, 4);
    if (file.gcount() != 4 || std::memcmp(head, "PAR1", 4) != 0) {
        Logger::error("ParquetReader: '{}' is missing the leading PAR1 magic", parquet_path);
        return -1;
    }

    char tail[8];
    file.seekg(file_size - 8, std::ios::beg);
    file.read(tail, 8);
    if (file.gcount() != 8 || std::memcmp(tail + 4, "PAR1", 4) != 0) {
        Logger::error("ParquetReader: '{}' is missing the trailing PAR1 magic", parquet_path);
        return -1;
    }
    std::uint32_t footer_len = 0;
    for (int b = 0; b < 4; ++b)
        footer_len |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(tail[b])) << (8 * b);
    if (static_cast<std::streamoff>(footer_len) + 8 > file_size) {
        Logger::error("ParquetReader: '{}' declares an impossible footer length ({})", parquet_path,
                     footer_len);
        return -1;
    }

    std::vector<std::uint8_t> footer(footer_len);
    file.seekg(file_size - 8 - static_cast<std::streamoff>(footer_len), std::ios::beg);
    file.read(reinterpret_cast<char*>(footer.data()), footer_len);
    if (static_cast<std::uint32_t>(file.gcount()) != footer_len) {
        Logger::error("ParquetReader: short read on footer of '{}'", parquet_path);
        return -1;
    }

    Cursor fc{footer.data(), footer.size(), 0};
    FileMetaData meta;
    if (!parse_file_metadata(fc, meta)) {
        Logger::error("ParquetReader: failed to parse footer metadata of '{}' (unsupported or "
                     "corrupt — e.g. encrypted file, or a malformed footer)",
                     parquet_path);
        return -1;
    }
    if (meta.schema.size() < 2) {
        Logger::error("ParquetReader: '{}' has no leaf columns in its schema", parquet_path);
        return -1;
    }

    // schema[0] is the implicit root group; schema[1..] are the flat leaf
    // columns this reader supports. Reject anything nested/repeated up front.
    std::vector<const SchemaElement*> leaf_columns;
    for (std::size_t i = 1; i < meta.schema.size(); ++i) {
        const SchemaElement& se = meta.schema[i];
        if (!se.has_type) {
            Logger::error(
                "ParquetReader: '{}' has a nested/group schema element ('{}') — only flat schemas "
                "are supported",
                parquet_path, se.name);
            return -1;
        }
        if (se.has_repetition && se.repetition == FieldRepetitionType::REPEATED) {
            Logger::error("ParquetReader: '{}' has a REPEATED column ('{}') — not supported",
                         parquet_path, se.name);
            return -1;
        }
        leaf_columns.push_back(&se);
    }

    // Preflight: reject unsupported codecs/encodings/features across every
    // column of every row group, entirely from footer data, before writing
    // any output.
    for (const RowGroup& rg : meta.row_groups) {
        for (const ColumnChunk& cc : rg.columns) {
            if (cc.has_file_path) {
                Logger::error(
                    "ParquetReader: '{}' references an external column-chunk file — not supported",
                    parquet_path);
                return -1;
            }
            if (cc.has_crypto_metadata) {
                Logger::error("ParquetReader: '{}' has an encrypted column — not supported",
                             parquet_path);
                return -1;
            }
            if (!cc.has_meta_data) {
                Logger::error("ParquetReader: '{}' has a column chunk with no metadata",
                             parquet_path);
                return -1;
            }
            const ColumnMetaData& md = cc.meta_data;
            if (md.codec != CompressionCodec::UNCOMPRESSED && md.codec != CompressionCodec::SNAPPY) {
                Logger::error("ParquetReader: '{}' uses unsupported compression codec {}",
                             parquet_path, static_cast<int>(md.codec));
                return -1;
            }
            if (md.type == PhysicalType::INT96 || md.type == PhysicalType::FIXED_LEN_BYTE_ARRAY) {
                Logger::error("ParquetReader: '{}' uses unsupported physical type {}", parquet_path,
                             static_cast<int>(md.type));
                return -1;
            }
            for (std::int32_t enc : md.encodings) {
                if (enc != static_cast<std::int32_t>(Encoding::PLAIN) &&
                    enc != static_cast<std::int32_t>(Encoding::RLE) &&
                    enc != static_cast<std::int32_t>(Encoding::RLE_DICTIONARY) &&
                    enc != static_cast<std::int32_t>(Encoding::PLAIN_DICTIONARY)) {
                    Logger::error("ParquetReader: '{}' uses unsupported encoding {}", parquet_path,
                                 enc);
                    return -1;
                }
            }
        }
    }

    std::ofstream out(out_jsonl_path, append ? (std::ios::out | std::ios::app) : std::ios::out);
    if (!out.is_open()) {
        Logger::error("ParquetReader: cannot open '{}' for writing", out_jsonl_path);
        return -1;
    }

    long long total_rows = 0;
    std::string line;
    for (const RowGroup& rg : meta.row_groups) {
        if (rg.columns.size() != leaf_columns.size()) {
            Logger::error(
                "ParquetReader: '{}' row group has {} column chunk(s), expected {} (schema mismatch)",
                parquet_path, rg.columns.size(), leaf_columns.size());
            return -1;
        }

        std::vector<std::vector<Cell>> columns(leaf_columns.size());
        for (std::size_t ci = 0; ci < leaf_columns.size(); ++ci) {
            const SchemaElement& se = *leaf_columns[ci];
            const ColumnChunk& cc = rg.columns[ci];
            if (cc.meta_data.path_in_schema.empty() ||
                cc.meta_data.path_in_schema.back() != se.name) {
                Logger::error(
                    "ParquetReader: '{}' column chunk {} does not match schema column '{}' — "
                    "unexpected column ordering",
                    parquet_path, ci, se.name);
                return -1;
            }
            const bool is_optional =
                se.has_repetition && se.repetition == FieldRepetitionType::OPTIONAL;
            if (!decode_column_chunk(file, parquet_path, cc, is_optional, rg.num_rows, columns[ci]))
                return -1;  // decode_column_chunk already logged the specific reason
        }

        for (std::int64_t row = 0; row < rg.num_rows; ++row) {
            line.clear();
            line += '{';
            for (std::size_t ci = 0; ci < leaf_columns.size(); ++ci) {
                if (ci > 0)
                    line += ',';
                line += '"';
                json_escape_append(leaf_columns[ci]->name, line);
                line += "\":";
                append_cell_json(columns[ci][static_cast<std::size_t>(row)], line);
            }
            line += "}\n";
            out.write(line.data(), static_cast<std::streamsize>(line.size()));
            ++total_rows;
        }
    }

    if (!out.good()) {
        Logger::error("ParquetReader: write error on '{}'", out_jsonl_path);
        return -1;
    }

    return total_rows;
}
