#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07


#include <string>

/**
 * @brief Native, dependency-free Parquet → JSONL reader.
 *
 * Replaces the Python (pandas/pyarrow) subprocess `DataFetcher` used to shell
 * out to for converting HuggingFace-hosted `.parquet` files to JSONL — that
 * approach depends on a `python3` interpreter plus pip-installed pandas or
 * pyarrow being present on whatever host runs the fetch, and pip's prebuilt
 * wheels for those packages turned out to require AVX2/AVX512 instructions
 * that aren't guaranteed to exist on an arbitrary deployment machine (crashes
 * with `SIGILL` on import when absent — see registry_server's `metricsserver`
 * deployment). This reader has zero external dependencies: no libsnappy, no
 * Arrow/Parquet C++ library. Thrift compact-protocol footer/page-header
 * parsing and Snappy decompression are purpose-built here for exactly how
 * Parquet uses them, not general-purpose implementations. All internal
 * helpers (Thrift decoding, Snappy decompression, RLE/bit-packed hybrid
 * decoding, value decoding, row-group orchestration) are file-local static
 * functions in ParquetReader.cpp, mirroring DataFetcher.cpp's own convention
 * of keeping helper functions un-exposed rather than class members.
 *
 * Scope is deliberately narrow — exactly what real HuggingFace
 * `datasets-server` parquet exports use: flat (non-nested, non-repeated)
 * schemas, `SNAPPY` or `UNCOMPRESSED` codec, `PLAIN` and
 * `RLE_DICTIONARY`/`PLAIN_DICTIONARY` encodings, DataPage **V1** only.
 * `BYTE_ARRAY` (string) columns are the primary target; `INT32`/`INT64`/
 * `FLOAT`/`DOUBLE`/`BOOLEAN` are also decoded (PLAIN only) so a stray numeric
 * column doesn't fail the whole file. Anything outside this — DataPageV2,
 * other compression codecs, nested/repeated schema elements, unsupported
 * encodings, `INT96`/`FIXED_LEN_BYTE_ARRAY`, encrypted columns/files,
 * external column-chunk file references — fails loudly via `Logger::error`
 * and returns -1. It never silently mis-parses: wrong output here would
 * silently corrupt training data, a strictly worse failure than a loud crash.
 */
class ParquetReader {
   public:
    /**
     * @brief Convert a single Parquet file's rows to JSONL.
     *
     * Appends (or truncates+writes, if @p append is false) one JSON object
     * per row to @p out_jsonl_path — keys are the column names verbatim from
     * the Parquet schema (in schema order), values are the row's data (JSON
     * `null` for a missing/OPTIONAL-absent field). Iterates every row group
     * in the file, not just the first — real HuggingFace exports commonly
     * have hundreds of row groups per part.
     *
     * @param parquet_path  Path to the local `.parquet` file to read.
     * @param out_jsonl_path Path to write/append JSONL rows to.
     * @param append        false = truncate and write; true = append.
     * @return Number of rows written, or -1 on any unsupported/corrupt input
     *         (nothing is ever partially written to @p out_jsonl_path on
     *         failure — the file is left exactly as it was before the call).
     */
    static long long convert_to_jsonl(const std::string& parquet_path,
                                      const std::string& out_jsonl_path, bool append);
};
