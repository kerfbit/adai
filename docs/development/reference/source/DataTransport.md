# `DataTransport` — Source File Reference

- **File:** [`src/DataTransport.hpp`](../../../../src/DataTransport.hpp), header-only (the implementation is inline in the header)
- **Namespace:** global (`dt_detail` for helpers)
- **Built into:** `incremental_trainer` (the only includer, `IncrementalTrainingTool.cpp`); real implementation only when `BUILD_FTP_TRANSPORT` (libcurl found), otherwise stubs that throw
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/DataTransportTests.cpp`](../../../../tests/DataTransportTests.cpp) (14 defined; 13 run in a libcurl build, since some stub tests compile only without `BUILD_FTP_TRANSPORT`), [`tests/DataTransportFtpTests.cpp`](../../../../tests/DataTransportFtpTests.cpp) (30), [`tests/StartupSweepTests.cpp`](../../../../tests/StartupSweepTests.cpp) (7, the download-dir sweep in the trainer)
- **Last traced against the code:** 2026-10-11

> Code-traced reference. Behaviour marked **verified** was confirmed end-to-end on loopback:
> a local `registry_server` with its embedded FTP server enabled, a 15.8 MB, 200,000-pair JSONL
> file queued and split into three TD-205 segments, then acquired with `RemoteTransport` and
> downloaded with the real `DataTransport`.

---

## 1. What this file is

The **trainer-side FTP download client** for distributed training (Phase 10). When
`registry_server` runs with `--ftp-enabled`, `acquire` returns an `AcquireResponse` with the FTP
host/port and one `FileToken` per claimed file or segment. Each token carries a single-use username
and password, the server-relative `ftp_path`, `size_bytes` and a `checksum`. `DataTransport`
downloads each file into a local `download_dir` so the trainer can read it like a local file.

| Phase | Feature |
|---|---|
| 1 | Single-file fetch via libcurl; size verification; large-file warnings |
| 2 | Resume partial files (FTP `REST`); retry transient errors with 1 s / 2 s / 4 s backoff; parallel `fetch_all()`; skip complete files ("Condition B") |
| 3 | Explicit FTPS (`AUTH TLS`) when the server enables it |

### Why it matters

In distributed mode this is how training data physically reaches the GPU host. If it fetches the
wrong bytes, the trainer trains on them. If it fails, `IncrementalTrainingTool` releases the claim
and the pass is skipped.

---

## 2. API

### `fetch(token, host, port, download_dir, large_file_warn_bytes = 0, ftps = false)`

1. **Create** `download_dir`. The local path is **`download_dir / filename(token.ftp_path)`**,
   the basename only.
2. **Condition B:** if `size_bytes > 0` and a local file of exactly that size exists, return it
   immediately ("`[CLEANUP-B]` … Skipping re-download"). **The size is the only check.**
3. If `size_bytes ≥ large_file_warn_bytes`, log a large-transfer warning.
4. **URL:** `ftp://<user>:<password>@<host>:<port>/<path>`. Each path segment is percent-encoded by
   `url_encode_ftp_path()`. The credentials aren't encoded; they're hex tokens from the server.
5. **Up to 4 attempts** (`kMaxRetries = 3`). Each attempt:
   - Works out the resume offset from the local file: an empty file starts over, an oversized one is
     deleted, a smaller one resumes with `CURLOPT_RESUME_FROM_LARGE`.
   - Opens the file to append (resume) or truncate.
   - Sets `CURLFTPMETHOD_NOCWD` (the server forbids `CWD`, and paths contain `/`), PASV not EPSV, a
     30 s connect timeout, and aborts below 1 B/s for 60 s.
   - With FTPS, sets `CURLUSESSL_ALL` with **peer and host verification off** (self-signed server
     certificate).
   - Retries only `is_transient()` errors: timeout, recv/send error, got nothing, partial file,
     accept timeout. `COULDNT_CONNECT`, login and URL errors fail at once.
6. On final failure, **delete the local file** and throw. Otherwise verify the final size
   (`size_bytes > 0` only); on a mismatch, delete and throw.

### `fetch_all(resp, download_dir, max_parallel = 4, warn_bytes = 0)`

Starts `min(n, max_parallel)` threads that take indexes from an atomic counter and call `fetch()`.
Results come back in `resp.files` order. After all threads join, it rethrows the **first** error in
index order. Files that did download stay on disk.

The trainer (`IncrementalTrainingTool.cpp`, train/resume/retrain and LeJEPA paths) calls it with
the `DatasetConfig`'s (`dcfg`) `download_dir`/`max_parallel_downloads`. On an exception it logs "Download
failed" and releases the claim (with bare paths, so TD-333 applies to segments). The trainer's
startup sweep (`StartupSweep.hpp`'s `startup_sweep()`, Conditions A/D/G, tested by `StartupSweepTests`) deletes zero-byte, trained and
orphaned files from `download_dir`. It keeps any non-empty file, which is what Condition B later
reuses.

Without `BUILD_FTP_TRANSPORT`, both methods throw "FTP transport not compiled". The trainer only
reaches them when `ftp_server_host` is non-empty.

---

## 3. Local file identity (verified)

The local name is the **basename** of `ftp_path`. A test (`UsesBasenameOfFtpPath`) pins this.
Two consequences:

> **Segments of one file download to the same local path, in parallel, each in full.** All three
> segment tokens had `ftp_path = g/uploads/big.jsonl` and full-file `size_bytes = 15769547`.
> `fetch_all()` mapped all three to `dl/big.jsonl` and ran them **concurrently**, so the whole
> 15.8 MB file crossed the wire three times. In 20 repeated runs the final file was intact (the
> threads write identical bytes at identical offsets), but the threads race:
> - each opens with `trunc`;
> - each checks the final size and **deletes the file on a mismatch**.
>
> So one thread's verification can see another's in-progress file, delete it, and fail a pass that
> would have succeeded. That wasn't reproduced on loopback, but nothing prevents it.

> **Condition B trusts a same-name, same-size file (verified).** A local `big.jsonl` of the right
> size but different content (all `Z` bytes) was returned **without downloading**, even though the
> token carried a `checksum` (`"15769547_<mtime>"`, `DatasetRegistry::compute_checksum()` format)
> that would have caught it. Realistic triggers:
> - a dataset re-uploaded under the same name with an edit that keeps its size;
> - two different datasets with the same basename in different directories (`uploads/` vs
>   `datasets/`, or two kinds' sub-pools) sharing one `download_dir`.
>
> The startup sweep keeps non-empty files, so a stale copy survives across passes.

Using the full `ftp_path` (or a hash of it) as the local name, deduplicating tokens per path in
`fetch_all()`, and checking the token checksum before Condition B would close all three.

---

## 4. Other notes (by inspection)

- **FTPS doesn't authenticate the server** (`SSL_VERIFYPEER`/`VERIFYHOST` off). The control
  channel is encrypted, but anyone who can intercept the connection can impersonate the registry
  and serve arbitrary training data, or collect the one-time credentials. There's no option to
  pin or trust a CA/certificate. Documented in the header as an internal-network trade-off.
- **Credentials sit in the URL.** They aren't logged here, and they're single-use and short-lived
  (default 30 min TTL), so the exposure is small.
- **No `curl_global_init()`.** Implicit global init on the first `curl_easy_init()` from several
  `fetch_all()` threads at once is only safe on a thread-safe-init libcurl. 8.5.0 on this dev
  machine is (`CURL_VERSION_THREADSAFE`); the deployed host wasn't checked.
- **A final failure deletes the partial file**, so resume only helps within the 4 attempts or after
  a crash, not across ordinary failed passes.
- **`COULDNT_CONNECT` isn't retried**, so a registry restart during a pass fails it immediately.
  It's a documented choice, but a restart is exactly the transient case.
- **Header hygiene:** `using adai::Logger;` and `namespace fs = std::filesystem;` at global scope
  leak into every includer. The non-`inline` `static` `curl_write` is fine only while there's a
  single includer.

**Seen while testing, outside this file:** `registry_server` took about 40 s to exit after SIGTERM
once FTP had been used. `FtpDataServer::stop()` joins a sweep thread that sleeps a full minute
between checks of `running_`. That belongs to `FtpDataServer.hpp`'s reference.

---

## 5. Tests

**`DataTransportFtpTests.cpp` (30 tests):**
- Condition B: skip, size mismatch, unknown size;
- `fetch_all` ordering and error propagation;
- basename mapping;
- URL encoding;
- every `is_transient()` code;
- the retry constants.

**`DataTransportTests.cpp` (14 defined, 13 run with libcurl):** stubs and struct defaults.

**`StartupSweepTests.cpp` (7):** the trainer's download-dir sweep.

**Not covered:**
- duplicate `ftp_path` tokens (segments);
- basename collisions;
- stale same-size files;
- FTPS;
- a real transfer with resume or retry.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Local name is the basename: segment tokens of one file download it N times, concurrently, to one path; different files with the same basename collide (§3) | Redundant transfers; a verification race can delete another thread's file; cross-file overwrites | **Verified** (shared path, redundant parallel downloads) / by inspection (race, collisions) | [TD-337](../../guides/TECHNICAL_DEBT.md#td-337-datatransport-names-downloads-by-basename-so-segments-and-same-named-files-collide) |
| Condition B reuses any same-name, same-size file; the token checksum is ignored (§3) | Stale or wrong data trained silently | **Verified** | [TD-338](../../guides/TECHNICAL_DEBT.md#td-338-datatransport-reuses-same-size-stale-files-without-checking-the-checksum) |
| FTPS without server verification (§4) | Registry impersonation or MITM can inject training data | By inspection | [TD-339](../../guides/TECHNICAL_DEBT.md#td-339-ftps-downloads-dont-verify-the-registrys-certificate) |
| No `curl_global_init()`; partial file deleted on failure; `COULDNT_CONNECT` not retried; global `using`/namespace alias in a header (§4) | Possible init race on older libcurl; lost resume; registry restarts fail passes; namespace pollution | By inspection | [TD-340](../../guides/TECHNICAL_DEBT.md#td-340-datatransport-robustness-and-header-hygiene) |
| Segment release on download failure uses bare paths | Already tracked | — | TD-333 |
