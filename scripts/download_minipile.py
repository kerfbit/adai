#!/usr/bin/env python3
"""
Download minipile parquet shards and convert to incremental trainer format.

Minipile has 5 train shards at:
  https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-0000N-of-00005.parquet

Output: huggingface_data/JeanKaddour_minipile_train_training_large.txt
Format: INPUT\tOUTPUT  (tab-separated, one pair per line)

Strategy: split each document at a mid-sentence boundary to create a
prompt (first half) / completion (second half) pair — same as the existing
single-text-field logic in IncrementalTrainer.cpp.
"""

import os
import re
import sys
import urllib.request
import pyarrow.parquet as pq
import io

TARGET_PAIRS = 50000
OUTPUT_FILE  = "huggingface_data/JeanKaddour_minipile_train_training_large.txt"
SHARDS = [
    "https://huggingface.co/datasets/JeanKaddour/minipile/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "https://huggingface.co/datasets/JeanKaddour/minipile/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet",
    "https://huggingface.co/datasets/JeanKaddour/minipile/resolve/refs%2Fconvert%2Fparquet/default/train/0002.parquet",
]

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_CHARS = 1024 * 5  # match clip_text in ChatbotTrainer.cpp (max_seq_length * 5)

SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def split_text(text: str):
    """Split text at a sentence boundary near the midpoint."""
    text = text.strip()
    if not text:
        return None, None

    mid = len(text) // 2
    # Find the first sentence boundary at or after the midpoint
    m = SENTENCE_END.search(text, mid)
    if m:
        prompt     = text[:m.start() + 1].strip()
        completion = text[m.end():].strip()
    else:
        # No sentence boundary found — split at whitespace near middle
        space = text.rfind(' ', 0, mid)
        if space == -1:
            return None, None
        prompt     = text[:space].strip()
        completion = text[space:].strip()

    if len(prompt) < 20 or len(completion) < 20:
        return None, None

    # Apply the same char-level clip used by ChatbotTrainer before we hand text off
    prompt     = prompt[:MAX_CHARS]
    completion = completion[:MAX_CHARS]
    return prompt, completion


def download_shard(url: str, idx: int) -> bytes:
    cache_path = f"huggingface_data/minipile_shard_{idx:04d}.parquet"
    if os.path.exists(cache_path):
        print(f"  Using cached shard {idx+1}/{len(SHARDS)}: {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            return f.read()

    print(f"  Downloading shard {idx+1}/{len(SHARDS)}: {url.split('/')[-1]} ...", flush=True)
    req = urllib.request.Request(url)
    if HF_TOKEN:
        req.add_header("Authorization", f"Bearer {HF_TOKEN}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    print(f"    {len(data) // 1024 // 1024} MB downloaded", flush=True)

    with open(cache_path, "wb") as f:
        f.write(data)
    return data


os.makedirs("huggingface_data", exist_ok=True)

pairs_written = 0
skipped = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for shard_idx, url in enumerate(SHARDS):
        if pairs_written >= TARGET_PAIRS:
            break

        try:
            raw = download_shard(url, shard_idx)
        except Exception as e:
            print(f"  ⚠ Failed to download shard {shard_idx+1}: {e}", file=sys.stderr)
            continue

        table = pq.read_table(io.BytesIO(raw))
        col_names = table.schema.names
        print(f"  Columns: {col_names}  Rows: {len(table)}", flush=True)

        if "text" not in col_names:
            print(f"  ⚠ No 'text' column found in shard {shard_idx+1}", file=sys.stderr)
            continue

        texts = table.column("text").to_pylist()
        for text in texts:
            if pairs_written >= TARGET_PAIRS:
                break
            if not isinstance(text, str):
                skipped += 1
                continue

            prompt, completion = split_text(text)
            if prompt is None:
                skipped += 1
                continue

            # Escape tabs and newlines within fields
            prompt     = prompt.replace('\t', ' ').replace('\n', ' ')
            completion = completion.replace('\t', ' ').replace('\n', ' ')

            out_f.write(f"INPUT: {prompt}\n\nRESPONSE: {completion}\n\n")
            pairs_written += 1

            if pairs_written % 5000 == 0:
                print(f"  ✓ {pairs_written} pairs written ...", flush=True)

        print(f"  Shard {shard_idx+1} done — total pairs so far: {pairs_written}", flush=True)

print(f"\n✅ Done. {pairs_written} pairs written to {OUTPUT_FILE}  (skipped {skipped} rows)")
print(f"   File size: {os.path.getsize(OUTPUT_FILE) // 1024} KB")
