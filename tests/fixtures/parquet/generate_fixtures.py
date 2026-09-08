#!/usr/bin/env python3
"""
Dev-only fixture generator for tests/parquet_reader_test.cpp.

NOT part of the build or CI — never invoked by CMake, never a runtime or
build dependency of the C++ project. This script exists purely so a human
can regenerate the small binary .parquet fixtures checked into this
directory, using this dev machine's local pyarrow install (the same
pyarrow install used throughout this session as a correctness oracle for
ParquetReader, since it's independent of the native C++ implementation
being tested).

Run manually, from the repo root:
    python3 tests/fixtures/parquet/generate_fixtures.py

Every fixture pins data_page_version="1.0" explicitly (pyarrow's current
default, but pinned here so a future pyarrow upgrade can't silently start
producing DataPageV2 fixtures — ParquetReader deliberately does not support
V2, see src/ParquetReader.hpp).
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))


def write(name, table, **kwargs):
    path = os.path.join(HERE, name)
    kwargs.setdefault("data_page_version", "1.0")
    pq.write_table(table, path, **kwargs)
    print(f"wrote {path}")


# 1. Simplest baseline: one string column, no compression, no dictionary.
write(
    "plain_uncompressed.parquet",
    pa.table({"text": ["hello world", "second row", "third row here"]}),
    compression="NONE",
    use_dictionary=False,
)

# 2. Same shape, SNAPPY-compressed, still no dictionary (forces PLAIN
#    encoding through a real Snappy-compressed page).
write(
    "plain_snappy.parquet",
    pa.table({"text": ["hello world", "second row", "third row here"]}),
    compression="SNAPPY",
    use_dictionary=False,
)

# 3. SNAPPY + dictionary encoding — matches real HuggingFace export shape
#    (repeated values so pyarrow actually dictionary-encodes the column;
#    an all-unique column would fall back to PLAIN regardless of the flag).
repeated_values = ["apple", "banana", "apple", "cherry", "banana", "apple"] * 20
write(
    "dict_snappy.parquet",
    pa.table({"text": repeated_values}),
    compression="SNAPPY",
    use_dictionary=True,
)

# 4. Multi-column, instruction/output-style — validates row-zip/column-order.
write(
    "multi_column.parquet",
    pa.table(
        {
            "instruction": [f"do thing {i}" for i in range(30)],
            "output": [f"result {i}" for i in range(30)],
        }
    ),
    compression="SNAPPY",
    use_dictionary=True,
)

# 5. Real nulls, to exercise definition-level bookkeeping specifically.
values_with_nulls = ["a", None, "b", None, None, "c", "a", "b"] * 5
write(
    "with_nulls.parquet",
    pa.table({"text": pa.array(values_with_nulls, type=pa.string())}),
    compression="SNAPPY",
    use_dictionary=True,
)

# 6. Multiple row groups in one small file — regression guard for the
#    "only reads row_groups[0]" bug class, without needing a huge file.
write(
    "multi_row_group.parquet",
    pa.table({"text": [f"row {i}" for i in range(200)]}),
    compression="SNAPPY",
    use_dictionary=False,
    row_group_size=50,  # 200 rows / 50 per group = 4 row groups
)

print("done")
