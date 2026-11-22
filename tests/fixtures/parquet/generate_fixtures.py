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

It also writes pyarrow's reading of each fixture as <name>.expected.jsonl (TD-472), the oracle
parquet_reader_test.cpp compares ParquetReader's output with.

Every fixture pins data_page_version="1.0" explicitly (pyarrow's current
default, but pinned here so a future pyarrow upgrade can't silently start
producing DataPageV2 fixtures — ParquetReader deliberately does not support
V2, see src/ParquetReader.hpp).
"""

import json
import math
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

# 7. TD-472: every physical type the reader decodes, with nulls in each, awkward strings (quotes,
#    backslashes, control characters, non-ASCII) and doubles that need all 17 significant digits.
write(
    "all_types.parquet",
    pa.table(
        {
            "i32": pa.array([0, -1, 2147483647, -2147483648, None, 42], type=pa.int32()),
            "i64": pa.array([0, -1, 9223372036854775807, -9223372036854775808, 7, None],
                            type=pa.int64()),
            "f32": pa.array([0.1, -2.5, None, 3.4028234663852886e38, 1e-38, float("inf")],
                            type=pa.float32()),
            "f64": pa.array([3.141592653589793, -1e-300, 2.5e10, None, 0.1 + 0.2, float("nan")],
                            type=pa.float64()),
            "flag": pa.array([True, False, None, True, False, True], type=pa.bool_()),
            "text": pa.array(['quote " and \\ backslash', "line\nbreak\ttab\x01ctl", "",
                              None, "caf\u00e9 \u2014 \U0001F600", "plain"], type=pa.string()),
        }
    ),
    compression="SNAPPY",
    use_dictionary=["text"],
)


# TD-472: pyarrow's own reading of every fixture, as JSON Lines, so parquet_reader_test.cpp can
# compare ParquetReader's output with it without Python or a network at test time. A non-finite
# float is written as null: JSON has no NaN or infinity, and that is what ParquetReader writes.
def expected_jsonl(name):
    table = pq.read_table(os.path.join(HERE, name))
    path = os.path.join(HERE, name.replace(".parquet", ".expected.jsonl"))
    with open(path, "w", encoding="utf-8") as out:
        for row in table.to_pylist():
            row = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                   for k, v in row.items()}
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"wrote {path}")


for name in sorted(os.listdir(HERE)):
    if name.endswith(".parquet"):
        expected_jsonl(name)

print("done")
