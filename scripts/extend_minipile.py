#!/usr/bin/env python3
"""
Select 10,000 additional minipile training pairs that:
  1. Do not duplicate any pair already in the existing training file
  2. Contain only printable ASCII characters (no non-ASCII / mojibake)

Strategy:
  - Build a dedup fingerprint set from INPUT texts in the existing file
  - Scan parquet shards in order; for each document:
      a. Reject if any byte is outside ASCII 0x09-0x7E range
      b. Split at mid-sentence boundary (same logic as download_minipile.py)
      c. Skip if prompt fingerprint is already in the dedup set
  - Write 10,000 new pairs in INPUT:/RESPONSE: format to a new file
  - Then `add` that file via incremental_trainer

Output: huggingface_data/JeanKaddour_minipile_train_extra_10k.txt
"""

import hashlib
import io
import os
import re
import sys
import urllib.request

import pyarrow.parquet as pq

TARGET_PAIRS  = 10000
EXISTING_FILE = "huggingface_data/JeanKaddour_minipile_train_training_large.txt"
OUTPUT_FILE   = "huggingface_data/JeanKaddour_minipile_train_extra_10k.txt"

# Full shard list — we'll walk them in order until TARGET_PAIRS reached
SHARDS = [
    ("0000", "https://huggingface.co/datasets/JeanKaddour/minipile/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"),
    ("0001", "https://huggingface.co/datasets/JeanKaddour/minipile/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet"),
    ("0002", "https://huggingface.co/datasets/JeanKaddour/minipile/resolve/refs%2Fconvert%2Fparquet/default/train/0002.parquet"),
]

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
MAX_CHARS = 1024 * 5   # match clip_text in ChatbotTrainer.cpp

SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# ---------------------------------------------------------------------------
# Only accept bytes in the printable-ASCII + common whitespace range.
# Rejects: multi-byte UTF-8, control chars (except tab/newline/CR).
# ---------------------------------------------------------------------------
_ASCII_OK = re.compile(r'^[\x09\x0A\x0D\x20-\x7E]*$')

def is_ascii_clean(text: str) -> bool:
    return bool(_ASCII_OK.match(text))


def fingerprint(text: str) -> str:
    """SHA-256 of the first 512 chars (sufficient for dedup)."""
    return hashlib.sha256(text[:512].encode('utf-8', errors='replace')).hexdigest()


def split_text(text: str):
    """Split at a sentence boundary near the midpoint. Returns (prompt, completion) or (None, None)."""
    text = text.strip()
    if not text:
        return None, None
    mid = len(text) // 2
    m = SENTENCE_END.search(text, mid)
    if m:
        prompt     = text[:m.start() + 1].strip()
        completion = text[m.end():].strip()
    else:
        space = text.rfind(' ', 0, mid)
        if space == -1:
            return None, None
        prompt     = text[:space].strip()
        completion = text[space:].strip()

    if len(prompt) < 20 or len(completion) < 20:
        return None, None

    return prompt[:MAX_CHARS], completion[:MAX_CHARS]


def load_existing_fingerprints(path: str) -> set:
    """Parse INPUT: lines from the existing training file and build a fingerprint set."""
    fps = set()
    if not os.path.exists(path):
        print(f"  (No existing file found at {path} — starting fresh)", flush=True)
        return fps

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith("INPUT: "):
                text = line[7:].rstrip('\n')
                fps.add(fingerprint(text))

    print(f"  Loaded {len(fps):,} fingerprints from existing training file.", flush=True)
    return fps


def load_shard(shard_id: str, url: str) -> bytes:
    cache = f"huggingface_data/minipile_shard_{shard_id}.parquet"
    if os.path.exists(cache):
        print(f"  Using cached shard {shard_id}: {cache}", flush=True)
        with open(cache, 'rb') as f:
            return f.read()

    print(f"  Downloading shard {shard_id} from HuggingFace ...", flush=True)
    req = urllib.request.Request(url)
    if HF_TOKEN:
        req.add_header("Authorization", f"Bearer {HF_TOKEN}")
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    print(f"    {len(data) // 1024 // 1024} MB downloaded", flush=True)
    with open(cache, 'wb') as f:
        f.write(data)
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
os.makedirs("huggingface_data", exist_ok=True)

print("Building dedup fingerprint set from existing training data ...", flush=True)
existing_fps = load_existing_fingerprints(EXISTING_FILE)

pairs_written = 0
stats = {"dup": 0, "non_ascii": 0, "short": 0, "ok": 0}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
    for shard_id, url in SHARDS:
        if pairs_written >= TARGET_PAIRS:
            break

        try:
            raw = load_shard(shard_id, url)
        except Exception as e:
            print(f"  ⚠ Failed to load shard {shard_id}: {e}", file=sys.stderr)
            continue

        table  = pq.read_table(io.BytesIO(raw))
        texts  = table.column("text").to_pylist()
        print(f"  Shard {shard_id}: {len(texts):,} rows", flush=True)

        for row_idx, text in enumerate(texts):
            if pairs_written >= TARGET_PAIRS:
                break
            if not isinstance(text, str):
                stats["short"] += 1
                continue

            # --- ASCII filter (applied to raw document before splitting) ---
            if not is_ascii_clean(text):
                stats["non_ascii"] += 1
                continue

            prompt, completion = split_text(text)
            if prompt is None:
                stats["short"] += 1
                continue

            # --- Dedup check ---
            fp = fingerprint(prompt)
            if fp in existing_fps:
                stats["dup"] += 1
                continue

            # --- Accept ---
            existing_fps.add(fp)   # prevent within-batch duplicates too
            prompt     = prompt.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            completion = completion.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            out_f.write(f"INPUT: {prompt}\n\nRESPONSE: {completion}\n\n")
            pairs_written += 1
            stats["ok"] += 1

            if pairs_written % 1000 == 0:
                print(f"  ✓ {pairs_written:,} pairs written  "
                      f"(row {row_idx:,}, dup={stats['dup']:,}, "
                      f"non_ascii={stats['non_ascii']:,}, short={stats['short']:,})",
                      flush=True)

        print(f"  Shard {shard_id} done — pairs so far: {pairs_written:,}", flush=True)

print(f"\n✅ Done.")
print(f"   Pairs written : {pairs_written:,}")
print(f"   Skipped (dup) : {stats['dup']:,}")
print(f"   Skipped (non-ASCII) : {stats['non_ascii']:,}")
print(f"   Skipped (too short) : {stats['short']:,}")
print(f"   Output file   : {OUTPUT_FILE}")
print(f"   File size     : {os.path.getsize(OUTPUT_FILE) // 1024:,} KB")
