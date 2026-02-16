# Vocabulary Test and Repair Script

## Overview

`test_repair_vocab.py` is a comprehensive tool for validating and repairing BPE (Byte Pair Encoding) tokenizer vocabulary files. It checks for structural integrity, detects common issues, and can automatically repair many problems.

## Features

### Validation Checks

The script performs the following validation checks:

1. **Header Validation**: Ensures the file starts with the correct header comment
2. **VOCAB_SIZE Declaration**: Verifies VOCAB_SIZE is properly declared
3. **Special Tokens**: Confirms all required special tokens are present:
   - `pad_token_id`
   - `unk_token_id`
   - `bos_token_id`
   - `eos_token_id`
4. **Format Validation**: Checks that vocab entries are tab-separated
5. **Duplicate Detection**: Identifies duplicate token IDs and duplicate tokens
6. **Size Consistency**: Verifies declared VOCAB_SIZE matches actual entry count
7. **ID Range Validation**: Ensures token IDs are within valid range
8. **Encoding Issues**: Handles files with non-UTF-8 encoding

### Repair Capabilities

The script can automatically repair:

- **VOCAB_SIZE mismatch**: Updates declared size to match actual entries
- **Duplicate token IDs**: Removes duplicate entries (keeps first occurrence)
- **Duplicate tokens**: Removes duplicate token entries
- **Encoding issues**: Reads files with multiple encoding fallbacks

## Usage

### Basic Validation (Test Only)

```bash
# Test the default vocab.txt file
python3 test_repair_vocab.py

# Test a specific file
python3 test_repair_vocab.py /path/to/vocab.txt

# Test only (no repairs even if --repair flag might be default)
python3 test_repair_vocab.py vocab.txt --test-only
```

### Repair Mode

```bash
# Repair and overwrite original file
python3 test_repair_vocab.py vocab.txt --repair

# Repair and save to a new file
python3 test_repair_vocab.py vocab.txt --repair --output vocab_repaired.txt

# Short form
python3 test_repair_vocab.py vocab.txt --repair -o vocab_fixed.txt
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `file` | Path to vocab.txt file (default: `vocab.txt`) |
| `--repair` | Attempt to repair detected issues |
| `--output FILE` | Output path for repaired file (default: overwrite original) |
| `-o FILE` | Short form of --output |
| `--test-only` | Only run tests, do not repair |

## Output

The script provides a detailed validation report including:

- File information (path, declared size, actual entries)
- Warnings (non-critical issues)
- Errors (critical issues)
- Repairs made (when run with --repair)

### Exit Codes

- `0`: File is valid or has only warnings
- `1`: File has errors

## Examples

### Example 1: Check file validity

```bash
$ python3 test_repair_vocab.py vocab.txt

BPE Vocabulary Validator and Repair Tool
======================================================================

Running validation checks...

======================================================================
VALIDATION REPORT
======================================================================

File: vocab.txt
Declared VOCAB_SIZE: 10000
Actual vocab entries: 9854
Special tokens: 4

⚠ WARNINGS (1):
  - VOCAB_SIZE mismatch: declared=10000, actual=9854

✓ No errors found!

======================================================================

⚠ Vocabulary file has warnings but no critical errors.
```

### Example 2: Repair file with issues

```bash
$ python3 test_repair_vocab.py vocab.txt --repair -o vocab_clean.txt

BPE Vocabulary Validator and Repair Tool
======================================================================

Running validation checks...

Attempting repairs...

✓ Repaired file written to: vocab_clean.txt

======================================================================
VALIDATION REPORT
======================================================================

File: vocab.txt
Declared VOCAB_SIZE: 9380
Actual vocab entries: 9380
Special tokens: 4

🔧 REPAIRS MADE (2):
  - Updated VOCAB_SIZE to 9380
  - Total duplicates removed: 474

======================================================================

✓ Vocabulary file is valid!
```

## Expected File Format

The vocab.txt file should follow this structure:

```
# BPE Tokenizer Vocabulary v1.0
VOCAB_SIZE 10000
SPECIAL_TOKENS
pad_token_id 0
unk_token_id 1
bos_token_id 2
eos_token_id 3
VOCAB
token1	9999
token2	9998
token3	9997
...
```

### Format Rules

1. First line must be: `# BPE Tokenizer Vocabulary v1.0`
2. Second line: `VOCAB_SIZE` followed by an integer
3. `SPECIAL_TOKENS` section with required token definitions
4. `VOCAB` section with tab-separated token-ID pairs
5. Tokens and IDs must be unique
6. IDs must be non-negative integers

## Common Issues

### Issue: Invalid vocab IDs (strings instead of numbers)

This occurs when the second column contains text instead of numeric IDs. The script will report these as errors but cannot automatically determine the correct IDs. Manual correction required.

### Issue: Duplicate token IDs

Multiple tokens assigned the same ID. The repair mode removes duplicates, keeping only the first occurrence.

### Issue: Encoding errors

File contains non-UTF-8 characters. The script automatically tries multiple encodings (latin-1, cp1252, iso-8859-1) with character replacement.

### Issue: VOCAB_SIZE mismatch

Declared size doesn't match actual entries. Repair mode updates the declared size.

## Troubleshooting

**Q: Script reports "File not found"**
- Ensure the file path is correct
- Use absolute path if relative path doesn't work

**Q: Many "Invalid vocab ID" errors**
- The vocab file may be corrupted or in wrong format
- Check that entries are tab-separated (not spaces)
- Verify the second column contains only integers

**Q: Encoding warnings appear**
- Your file uses non-UTF-8 encoding
- Consider re-saving the file with UTF-8 encoding
- The script handles this automatically but may replace some characters

## Integration

This script can be integrated into build pipelines or pre-training validation:

```bash
#!/bin/bash
# Validate vocab before training
python3 test_repair_vocab.py vocab.txt --test-only
if [ $? -ne 0 ]; then
    echo "Vocabulary validation failed!"
    exit 1
fi
echo "Vocabulary is valid, proceeding with training..."
```

## License

This script is provided as-is for vocabulary file validation and repair.
