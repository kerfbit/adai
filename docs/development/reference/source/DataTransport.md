# `DataTransport` — Source File Reference

- **File:** [`src/DataTransport.hpp`](../../../../src/DataTransport.hpp) (544 lines, header-only)
- **Built into:** `incremental_trainer`; included by `DatasetRegistry.hpp` and `TrainingPassFiles.cpp` too; the real implementation only with `BUILD_FTP_TRANSPORT` (libcurl), else both methods throw
- **Status tag:** `@adai-status: stable`, `@adai-version: 2.0.0` (MAJOR, TD-340: the global `using adai::Logger;` and `fs` alias, which includers could rely on, are gone), `@adai-reviewed: 2026-11-21`
- **Tests:** [`tests/DataTransportFtpTests.cpp`](../../../../tests/DataTransportFtpTests.cpp) (34), `dataTransportFtpsTests` (8, a real `FtpDataServer` and libcurl on loopback), [`tests/DataTransportTests.cpp`](../../../../tests/DataTransportTests.cpp) (13 run in a libcurl build), [`tests/StartupSweepTests.cpp`](../../../../tests/StartupSweepTests.cpp) (10, the trainer's download-directory sweep)
- **Last traced against the code:** 2026-11-21 (TD-340)

> Code-traced reference. Behaviour marked **verified** was confirmed end-to-end on loopback: a local `registry_server` with its FTP server,
> a large JSONL file queued in TD-205 segments, acquired with `RemoteTransport` and downloaded with the real `DataTransport`.

---

## 1. What this file is

The **trainer-side FTP download client** for distributed training. When `registry_server` runs with FTP enabled, `acquire` returns an
`AcquireResponse` with the FTP host, port, `ftps_enabled`, and one `FileToken` per claimed file or segment: a single-use username and password,
the server-relative `ftp_path`, the `registry_path`, `size_bytes` and a `checksum`. `DataTransport` downloads each file into a local directory so
the trainer reads it like a local file. In distributed mode this is how training data reaches the GPU host: wrong bytes here are trained on, and
a failure releases the claim and fails the pass.

---

## 2. API

### `DataTransport(FtpsVerification{ca_file, pinned_public_key})` (TD-339)

How FTPS authenticates the registry. A `pinned_public_key` (`sha256//…`, which `registry_server` logs at start-up, or a key file) is checked on
its own and works with the default self-signed certificate. A `ca_file` turns on peer and hostname verification. The trainer fills both from
`FTPS_PINNED_PUBLIC_KEY`/`FTPS_CA_FILE` (`config.trainer.conf` or the environment).

### `fetch(token, host, port, download_dir, large_file_warn_bytes = 0, ftps_enabled = false)`

1. **Create** `download_dir`. The local path is `local_path_for(token, dir)`: `dir / download_file_name(registry_path)` (`ftp_path` when the
   registry path is empty), a 16-hex-digit FNV-1a hash of the full path, `_`, and the basename (TD-337).
2. **Checksum (TD-338):** if the token has a checksum and a local file exists, its sidecar `<local>.checksum` must hold that checksum;
   otherwise the file and sidecar are deleted.
3. **Condition B:** if `size_bytes > 0` and a local file of exactly that size remains, return it without connecting (`[CLEANUP-B]`). Tokens with
   no checksum keep this size-only check.
4. A large-transfer warning when `size_bytes ≥ large_file_warn_bytes`.
5. **Verification checks (TD-339):** with verification configured, a registry offering plain FTP is refused before connecting (the
   credentials would cross unencrypted), and a missing CA file is an error. FTPS with no verification logs a warning once per process.
6. **URL:** `ftp://<user>:<password>@<host>:<port>/<path>`, each path segment percent-encoded (`url_encode_ftp_path()`; dataset names contain
   spaces). FTPS is explicit TLS (`AUTH TLS`, `CURLUSESSL_ALL`) on the same URL.
7. **Sidecar:** write the token's checksum before any bytes land, so a partial file is only resumed against the same version.
8. **Up to 4 attempts** (`kMaxRetries = 3`). Each attempt:
   - works out the resume offset (an empty file starts over, an oversized one is deleted, a smaller one resumes with
     `CURLOPT_RESUME_FROM_LARGE`) and opens the file to append or truncate;
   - sets `CURLFTPMETHOD_NOCWD` (the server refuses `CWD`, and paths contain `/`), PASV, a 30 s connect timeout, and aborts below 1 B/s for 60 s;
   - with FTPS, verification as configured;
   - retries only `is_transient()` errors (refused connection, timeout, recv/send error, got nothing, partial file, accept timeout) after
     1, 2 and 4 s (`set_retry_base_backoff_ms()` changes the 1 s base; the tests use 1 ms). Login and URL errors fail at once. A refused
     connection is retried since TD-340: a `registry_server` restarting mid-pass refuses connections for a few seconds.
9. On final failure, keep the file and its sidecar when it has any bytes, so the next fetch (the next pass) resumes it against the same
   checksum; delete an empty one; throw (TD-340: it used to delete the file, so a failed pass started over). Otherwise check the size (when
   `size_bytes > 0`); on a mismatch, delete both and throw.

The write callback calls `adai::note_pass_progress()` for every chunk (TD-569), so a long download counts as progress for `trainer_service`'s
stall check; the 60 s low-speed limit ends a transfer that receives nothing long before that limit matters.

`ensure_curl_initialized()` calls `curl_global_init()` once per process (`std::call_once`, TD-340): `incremental_trainer`'s `main()` calls it
before any thread exists, and `fetch()`/`fetch_all()` call it before starting, since libcurl's implicit initialization from several download
threads at once is safe only on a thread-safe-init build. The header names `adai::Logger` and `std::filesystem` in full; it no longer puts
`using`/alias declarations at global scope.

`local_path_for()`, `checksum_path_for()` and `remove_download()` (file + sidecar) are public statics, so the trainer's clean-up, the start-up
sweep and the tests use the same names.

### `fetch_all(resp, download_dir, max_parallel = 4, warn_bytes = 0)`

Groups the tokens by local path (TD-337), so a file's TD-205 segment tokens trigger **one** download, then runs `min(files, max_parallel)`
threads that take indexes from an atomic counter and `fetch()` each distinct file. Results come back in `resp.files` order, each token getting
its file's path. After all threads join, the **first** error in index order is rethrown; files that did download stay on disk.

The trainer calls it through `resolve_pass_files()` (`TrainingPassFiles.cpp`, [LeJEPATrainingPass.md](LeJEPATrainingPass.md) §1) for every pass, with `DOWNLOAD_DIR` (or
`<SESSION_DIR>/downloads`) and `MAX_PARALLEL_DOWNLOADS`. On an exception the pass releases its exact units (TD-333). After the pass the
downloads and sidecars are removed. The start-up sweep (`StartupSweep.hpp`) matches files by the same names: it deletes zero-byte, trained and
orphaned downloads with their sidecars, and keeps non-empty pending ones for Condition B or a resume.

Without `BUILD_FTP_TRANSPORT`, both methods throw "FTP transport not compiled"; the trainer only calls them when `ftp_server_host` is set.

---

## 3. Notes

- **Credentials sit in the URL.** They aren't logged, and they're single-use and short-lived (30 min TTL by default).
- **Verification is opt-in.** Without `FTPS_PINNED_PUBLIC_KEY`/`FTPS_CA_FILE`, FTPS encrypts but doesn't authenticate the registry, and a
  warning says so. The shipped `config.trainer.conf` documents both.
- **Robustness items (TD-340, fixed November 21, 2026):** `curl_global_init()` once, partial files kept for the next pass, a refused
  connection retried, and no global `using`/alias in the header (§2).
- **Upgrade note (TD-337):** downloads from before October 19, 2026 have the old basename names; the first sweep after the upgrade deletes them
  as orphans and they are downloaded again.

---

## 4. Tests

- **`DataTransportFtpTests.cpp` (32):** Condition B (skip, size mismatch, unknown size, a different or missing checksum); `fetch_all()` order and
  error propagation; segment tokens of one file downloading once; full-path naming with same-basename files kept apart; URL encoding; every
  `is_transient()` code; the retry constants.
- **`dataTransportFtpsTests` (8):** a real FTPS `FtpDataServer` and libcurl on loopback: the pin, the CA file, refusing plain FTP when
  verification is configured, a wrong pin or CA.
- **`DataTransportTests.cpp` (13 with libcurl):** stubs and struct defaults.
- **`StartupSweepTests.cpp` (10):** the download-directory sweep.

**Not covered:** a real transfer that resumes or retries.

---

## 5. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| No `curl_global_init()`; partial file deleted on failure; `COULDNT_CONNECT` not retried; global `using`/alias in a header | A possible init race on an old libcurl; lost resume across passes; a registry restart fails a pass; namespace pollution | **Fixed November 21, 2026** | [TD-340](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-340-datatransport-robustness-and-header-hygiene) (LOW) |
| Basename names: segments downloaded N times to one path, same-named files colliding | — | **Fixed October 19, 2026** | [TD-337](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-337-datatransport-names-downloads-by-basename-so-segments-and-same-named-files-collide) |
| Condition B reusing a same-size stale file | — | **Fixed October 19, 2026** | [TD-338](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-338-datatransport-reuses-same-size-stale-files-without-checking-the-checksum) |
| FTPS without server verification | — | **Fixed October 24, 2026** when a pin or CA file is set | [TD-339](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-339-ftps-downloads-dont-verify-the-registrys-certificate) |

`stable` stands; TD-340's robustness and hygiene items are fixed (November 21, 2026). The tests cover a kept partial file, the refused
connection's retries and an empty leftover being removed (`AFailedTransferKeepsAPartialFileToResume`, `ARefusedConnectionIsRetried`,
`SameSizeFileWithDifferentChecksumIsDownloadedAgain`).

## 6. History

- **October 19, 2026:** full-path names, one download per file, checksum sidecars (TD-337, TD-338).
- **October 24, 2026:** FTPS verification (TD-339).
- **November 11, 2026:** progress per downloaded chunk (TD-569). This document was re-traced against the code, folding in its addenda.
