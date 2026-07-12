# Dataset Manager Guide

`dataset_manager` is the standalone tool for managing the ADAI training data queue. It adds local files, downloads from Project Gutenberg and HuggingFace, and inspects or clears the queue — all without touching the model or requiring a tokenizer. The training binary (`incremental_trainer train`) consumes whatever this tool queues.

---

## Table of Contents

1. [Overview](#overview)
2. [Usage](#usage)
3. [Commands](#commands)
   - [add](#add)
   - [gutenberg](#gutenberg)
   - [gutenberg-batch](#gutenberg-batch)
   - [huggingface](#huggingface)
   - [status](#status)
   - [list-pending / list-trained](#list-pending--list-trained)
   - [clear-pending](#clear-pending)
4. [File Layout](#file-layout)
5. [Configuration](#configuration)
6. [Distributed Mode](#distributed-mode)
7. [Workflows](#workflows)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The dataset manager maintains two lists in the session directory:

| File | Purpose |
| --- | --- |
| `{SESSION_DIR}/pending_files.txt` | Files queued for the next training run |
| `{SESSION_DIR}/data_registry.txt` | Files that have already been trained, with metadata |

`incremental_trainer train` reads the pending list, claims those files, trains on them, and moves them to the registry on success. `dataset_manager` only writes to the pending list and reads from the registry — it never touches model weights or checkpoints.

**Key behaviors:**

- `add` silently skips a file if it is already in the trained registry or already pending.
- Downloaded Gutenberg and HuggingFace files are written to local cache directories (`gutenberg_data/`, `huggingface_data/`). Re-downloading the same book ID overwrites the cache.
- `clear-pending` removes files from the pending queue only; it does not erase them from the trained registry.

---

## Usage

```
./build/bin/dataset_manager [--config <path>] <command> [args...]
```

**Global option:**

| Option | Description |
| --- | --- |
| `--config <path>` | Path to `config.conf`. Auto-discovered if omitted: `./config.conf` → `/etc/adai/config.conf`. |

Run without arguments to print the built-in help.

---

## Commands

### `add`

Add a local training file to the pending queue.

```bash
./build/bin/dataset_manager add <data_file>
```

**Arguments:**

| Argument | Description |
| --- | --- |
| `<data_file>` | Path to a local file in `INPUT:` / `RESPONSE:` format |

The file must exist on disk at the time of the call. Relative paths are resolved from the current working directory and stored as-is, so run from the project root or use absolute paths.

**Examples:**

```bash
./build/bin/dataset_manager add sample_training_data.txt
./build/bin/dataset_manager add /opt/adai/data/conversations_2026_w24.txt
```

**Skip conditions:**

- File does not exist on disk → error
- File is already in the trained registry → skipped with a warning
- File is already in the pending queue → skipped with a warning

**Training data format** — every file must use alternating `INPUT:` / `RESPONSE:` pairs:

```text
INPUT: Hello
RESPONSE: Hi! How can I help you?
INPUT: What is the weather like?
RESPONSE: I don't have access to real-time data, but I can help with other questions.
```

---

### `gutenberg`

Download a single Project Gutenberg book, convert it to training pairs, and queue it.

```bash
./build/bin/dataset_manager gutenberg <book_id> [num_pairs]
```

**Arguments:**

| Argument | Default | Description |
| --- | --- | --- |
| `<book_id>` | required | Numeric Gutenberg book ID |
| `[num_pairs]` | `500` | Maximum INPUT/RESPONSE pairs to generate |

**What happens:**

1. Downloads `https://www.gutenberg.org/files/<id>/<id>-0.txt` (falls back to `<id>.txt` if unavailable).
2. Strips the Gutenberg header and footer.
3. Extracts sentences (20–500 characters each).
4. Generates up to `num_pairs` QA pairs using three styles:
   - *Continuation* — adjacent sentence pairs
   - *Question* — `"What does this mean: <sentence>"` → next sentence
   - *Summary* — `"Summarize: <context>"` → following sentence
5. Writes the result to `gutenberg_data/gutenberg_<id>_training.txt`.
6. Adds the training file to the pending queue.

**Examples:**

```bash
# Default 500 pairs
./build/bin/dataset_manager gutenberg 1342

# Larger extract from a long novel
./build/bin/dataset_manager gutenberg 2701 1000

# Quick style test with a short extract
./build/bin/dataset_manager gutenberg 11 150
```

**Popular book IDs:**

| ID | Title | Best for |
| --- | --- | --- |
| 1342 | Pride and Prejudice (Austen) | Formal dialogue, social interactions |
| 11 | Alice in Wonderland (Carroll) | Creative, whimsical language |
| 84 | Frankenstein (Shelley) | Philosophical, gothic prose |
| 1661 | Sherlock Holmes (Doyle) | Analytical reasoning, mysteries |
| 2701 | Moby Dick (Melville) | Descriptive, technical prose |
| 16328 | Beowulf | Epic, heroic language |
| 1260 | Jane Eyre (Brontë) | First-person narrative, emotion |
| 98 | A Tale of Two Cities (Dickens) | Historical narrative |
| 345 | Dracula (Stoker) | Suspense, epistolary |
| 35 | The Time Machine (Wells) | Science fiction concepts |
| 76 | Huckleberry Finn (Twain) | Conversational American English |
| 829 | Gulliver's Travels (Swift) | Satire, political commentary |

Find any book ID at: <https://www.gutenberg.org/ebooks/>

**Pair count guidelines:**

| Book type | Recommended pairs |
| --- | --- |
| Short story / novella | 100–300 |
| Novel | 300–700 |
| Long novel / epic | 700–1500 |

---

### `gutenberg-batch`

Download multiple Gutenberg books in a single command, each generating the same number of pairs.

```bash
./build/bin/dataset_manager gutenberg-batch <id1,id2,...> [num_pairs_each]
```

**Arguments:**

| Argument | Default | Description |
| --- | --- | --- |
| `<id1,id2,...>` | required | Comma-separated book IDs, no spaces |
| `[num_pairs_each]` | `500` | Pairs to generate per book |

Books are downloaded sequentially. If a single book fails, the others still proceed. The command reports how many were successfully added.

**Examples:**

```bash
# General conversation mix — 4 books, 400 pairs each
./build/bin/dataset_manager gutenberg-batch 1342,11,76,98 400

# Formal / analytical tone
./build/bin/dataset_manager gutenberg-batch 1661,84,1260,2701 300

# Creative and imaginative
./build/bin/dataset_manager gutenberg-batch 11,345,35,16328 500

# H.G. Wells collection
./build/bin/dataset_manager gutenberg-batch 35,36,159,5230 400

# Shakespeare (plays in prose conversion)
./build/bin/dataset_manager gutenberg-batch 1533,1534,1535 500
```

**Recommended combinations by goal:**

| Goal | Command |
| --- | --- |
| General conversation | `gutenberg-batch 1342,11,76,98 400` |
| Formal / professional | `gutenberg-batch 1661,84,1260,2701 300` |
| Creative / imaginative | `gutenberg-batch 11,345,35,16328 500` |
| Mystery & detective | `gutenberg-batch 1661,2147,2148,932 350` |
| Science fiction | `gutenberg-batch 35,36,159,5230 400` |

---

### `huggingface`

Download a HuggingFace dataset, convert it to training pairs, and queue it.

```bash
./build/bin/dataset_manager huggingface <dataset_id> [num_pairs] [split] [input_field] [output_field]
```

**Arguments:**

| Argument | Default | Description |
| --- | --- | --- |
| `<dataset_id>` | required | HuggingFace dataset identifier |
| `[num_pairs]` | `500` | Maximum pairs to extract |
| `[split]` | `train` | Dataset split (`train`, `validation`, `test`) |
| `[input_field]` | auto | JSON field name for the input/question text |
| `[output_field]` | auto | JSON field name for the output/answer text |

**No Python or `huggingface_hub` library required.** The fetcher uses the HuggingFace datasets-server REST API directly, downloading rows as JSON in chunks of 100.

**Auto-detection:** When `input_field` and `output_field` are omitted, the fetcher inspects the first row and matches common patterns: `instruction`/`output`, `question`/`answer`, `prompt`/`completion`, and `dialog` arrays. Supply explicit field names if auto-detection fails.

**Gated datasets** require a HuggingFace access token:

```bash
export HF_TOKEN=hf_your_token_here
./build/bin/dataset_manager huggingface owner/gated-dataset 300
```

**Output file:** `huggingface_data/<safe_id>_<split>_training.txt` (slashes in `dataset_id` become underscores in the filename).

**Examples:**

```bash
# Daily conversation pairs — auto-detected dialog array format
./build/bin/dataset_manager huggingface daily_dialog 500

# Instruction-following with explicit field mapping
./build/bin/dataset_manager huggingface tatsu-lab/alpaca 300 train instruction output

# Dolly instruction dataset
./build/bin/dataset_manager huggingface databricks/databricks-dolly-15k 500

# Chain-of-thought Q&A
./build/bin/dataset_manager huggingface Open-Orca/OpenOrca 500 train question response

# Validation split instead of train
./build/bin/dataset_manager huggingface daily_dialog 100 validation

# Custom field names for a less common dataset structure
./build/bin/dataset_manager huggingface owner/dataset 400 train prompt completion
```

**Popular datasets:**

| Dataset ID | Format | Notes |
| --- | --- | --- |
| `daily_dialog` | dialog array | Everyday conversation pairs; auto-detected |
| `tatsu-lab/alpaca` | instruction/output | Instruction-following; explicit fields recommended |
| `databricks/databricks-dolly-15k` | instruction/response | Instruction dataset; auto-detected |
| `Open-Orca/OpenOrca` | question/response | Chain-of-thought Q&A; explicit fields recommended |

Find datasets at: <https://huggingface.co/datasets>

---

### `status`

Show a summary of the pending queue and trained registry.

```bash
./build/bin/dataset_manager status
```

**Output includes:**

- Count of pending files
- Count of trained files
- Total samples trained across all registry entries
- Full registry table (file path, date trained, sample count)
- List of currently pending files

---

### `list-pending` / `list-trained`

Print file paths one per line — suitable for scripting.

```bash
./build/bin/dataset_manager list-pending
./build/bin/dataset_manager list-trained
```

**Examples:**

```bash
# Count pending files
./build/bin/dataset_manager list-pending | wc -l

# Check if a specific file is pending
./build/bin/dataset_manager list-pending | grep "conversations_week24"

# Archive all trained file paths
./build/bin/dataset_manager list-trained > trained_manifest.txt
```

---

### `clear-pending`

Remove all files from the pending queue without affecting the trained registry.

```bash
./build/bin/dataset_manager clear-pending
```

Use this to abandon queued data before starting a fresh dataset composition. Files that were previously trained remain in the registry; only the pending queue is cleared.

---

## File Layout

```text
{SESSION_DIR}/                         # default: training_sessions/
├── pending_files.txt                  # current pending queue (one path per line)
└── data_registry.txt                  # trained-file registry with metadata

gutenberg_data/                        # Gutenberg cache (relative to working dir)
├── gutenberg_1342.txt                 # raw downloaded text
├── gutenberg_1342_training.txt        # converted INPUT/RESPONSE pairs ← queued
├── gutenberg_11.txt
└── gutenberg_11_training.txt

huggingface_data/                      # HuggingFace cache
├── daily_dialog_train/                # raw JSON row chunks (directory)
├── daily_dialog_train_training.txt    # converted pairs ← queued
├── tatsu-lab_alpaca_train/
└── tatsu-lab_alpaca_train_training.txt
```

`SESSION_DIR` is controlled by the `SESSION_DIR` key in `config.conf` (default: `training_sessions`).

---

## Configuration

`dataset_manager` reads the same `config.conf` as `incremental_trainer`. The keys that affect it:

| Key | Default | Effect |
| --- | --- | --- |
| `SESSION_DIR` | `training_sessions` | Where `pending_files.txt` and `data_registry.txt` live |
| `REGISTRY_SERVER_URL` | — | If set, uses distributed remote registry instead of local flat files |
| `RUN_GROUP` | — | Logical namespace for the remote registry |
| `RUN_ID` | auto | Per-node identifier for claim/release operations |
| `REGISTRY_TIMEOUT_MS` | `5000` | HTTP timeout for remote registry calls |

---

## Distributed Mode

When `REGISTRY_SERVER_URL` is set in `config.conf`, `dataset_manager` routes all queue operations through the `registry_server` HTTP daemon instead of local flat files. This allows multiple trainer nodes to share a single pending queue without racing.

```ini
# config.conf
REGISTRY_SERVER_URL=http://coordinator-host:8082
RUN_GROUP=my-training-pool
```

Start the coordinator:

```bash
./build/bin/registry_server --port 8082 --data-dir registry_sessions
```

In distributed mode, `dataset_manager add / gutenberg / huggingface` still writes data files to local disk, but the queue entry is recorded on the remote server. All nodes see the same pending list; `incremental_trainer train` atomically claims files before training them so no two nodes train the same file.

For single-node use, leave `REGISTRY_SERVER_URL` unset (or commented out).

---

## Workflows

### Adding local data incrementally

```bash
# Queue a new conversation file
./build/bin/dataset_manager add conversations_week24.txt

# Verify it is pending
./build/bin/dataset_manager status

# Train on it (only the new file)
./build/bin/incremental_trainer train 15
```

### Building a diverse corpus from scratch

```bash
# Mix of real conversations and literary data
./build/bin/dataset_manager add real_conversations.txt
./build/bin/dataset_manager gutenberg-batch 1342,11,1661,84 400
./build/bin/dataset_manager huggingface daily_dialog 500

# Confirm what is queued before training
./build/bin/dataset_manager list-pending

# Train on everything
./build/bin/incremental_trainer train 25
```

### Replacing the pending queue

```bash
# Discard previously queued (but not yet trained) files
./build/bin/dataset_manager clear-pending

# Queue a different dataset instead
./build/bin/dataset_manager huggingface tatsu-lab/alpaca 500
./build/bin/incremental_trainer train 20
```

### Re-training files that have already been trained

Files in the trained registry are skipped by `add`. To include them again:

1. Use `incremental_trainer reset --keep-data --yes` — this marks all registry files as pending again.
2. Or use `incremental_trainer retrain` — this trains on all registry files without touching the pending list.

`dataset_manager` has no direct command to "un-train" a file; that is the trainer's responsibility.

### Scripted bulk ingestion

```bash
#!/usr/bin/env bash
# Ingest a directory of conversation files
for f in /opt/adai/data/conversations/*.txt; do
    ./build/bin/dataset_manager add "$f"
done

./build/bin/dataset_manager status
./build/bin/incremental_trainer train 20
```

---

## Troubleshooting

### "Data file not found"

The path passed to `add` does not exist on disk at the time of the call.

```bash
ls -lh my_file.txt    # verify the file exists
./build/bin/dataset_manager add "$(pwd)/my_file.txt"   # use absolute path
```

### "Data file already trained, skipping"

The file is already in the trained registry. If you genuinely want to retrain on it, use `incremental_trainer retrain` (which trains on all registry files) or `incremental_trainer reset --keep-data --yes` (which moves everything back to pending).

### "Data file already in pending queue"

The file was queued in a previous call but not yet trained. Run `incremental_trainer train` to consume it, or `clear-pending` if you no longer want it.

### Gutenberg download fails

- Verify the book ID at <https://www.gutenberg.org/ebooks/>
- Some books use non-standard URL patterns; try the fallback manually:
  ```bash
  curl -L "https://www.gutenberg.org/files/1342/1342.txt" -o my_book.txt
  ./build/bin/dataset_manager add my_book.txt
  ```
- Books in languages other than English or in non-prose formats (poetry, dictionaries) may produce few or no valid pairs.

### "No valid sentences found" from Gutenberg

The book may be poetry, a reference work, or written in a non-standard prose style. Choose a prose novel instead, or manually curate and use `add`.

### HuggingFace download fails

- Check the dataset ID at <https://huggingface.co/datasets>
- Verify the dataset-server API accepts it:
  ```bash
  curl "https://datasets-server.huggingface.co/is-valid?dataset=daily_dialog"
  ```
- For gated datasets, set `HF_TOKEN`:
  ```bash
  export HF_TOKEN=hf_your_token_here
  ./build/bin/dataset_manager huggingface owner/dataset 300
  ```

### "Could not extract training pairs" from HuggingFace

Auto-detection failed. Inspect the dataset schema on HuggingFace and provide explicit field names:

```bash
# Find field names from the dataset card, then:
./build/bin/dataset_manager huggingface owner/dataset 300 train <input_field> <output_field>
```

### Pending queue is not visible to the trainer

Both tools must use the same `SESSION_DIR`. Check that `config.conf` is being found and has `SESSION_DIR` set consistently, or pass `--config config.conf` explicitly to both:

```bash
./build/bin/dataset_manager --config config.conf add data.txt
./build/bin/incremental_trainer --config config.conf train 10
```
