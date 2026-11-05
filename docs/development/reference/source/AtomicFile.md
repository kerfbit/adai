# AtomicFile

- **Files:** [`src/AtomicFile.hpp`](../../../../src/AtomicFile.hpp), [`src/AtomicFile.cpp`](../../../../src/AtomicFile.cpp)
- **Built into:** `adai_core`
- **Status tags:** `experimental` 0.1.0 (new November 4, 2026, TD-495)
- **Tests:** `tests/atomicfile_test.cpp` (`atomicFileTests`)

## What it is

`adai::write_file_atomically(path, data)` replaces a file so a reader sees either the old contents or all of the new ones:

1. creates the parent directory if needed;
2. writes `data` to `<path>.tmp-<12 random hex digits>` in the same directory;
3. flushes and fsyncs it (`_commit()` on Windows);
4. renames it over `path` (`std::filesystem::rename`, which replaces an existing file);
5. fsyncs the directory, so the rename survives a crash (POSIX; best effort).

It returns an empty string on success, otherwise what failed; the old file is then untouched and the temporary removed. A
write that fails part-way (disk full, a file-size limit, a quota) therefore never leaves a cut-off file, which is what TD-495 found
`LocalTransport` doing with the dataset queues.

The target directory must be writable (the temporary is created there), unlike rewriting a file in place.

## Users

- `LocalTransport::save_pending()`/`save_registry()` (`src/RegistryTransport.cpp`), the queues and trained registries of
  `registry_server` and standalone trainers (TD-495).

Older code with its own temp-and-rename saves (vocabularies, training state, hippocampal memory, `chatbot`'s conversation saves
through `chatbot_client::write_file_atomically()`) hasn't been moved to it; they don't fsync.

## Tests

`WritesAndReplaces`; `FailedWriteLeavesTheOldFileWhole` (in a child with a 4 KiB file-size limit and `SIGXFSZ` ignored, so the write
fails with `EFBIG` part-way: the error is reported, the old 20,000-byte file is unchanged and no temporary remains);
`UnwritableDirectoryIsReportedAndTheFileKept`.
