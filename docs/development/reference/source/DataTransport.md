# `DataTransport` — Source File Reference

- **File:** [`src/DataTransport.hpp`](../../../../src/DataTransport.hpp), header-only (the implementation is inline in the header)
- **Namespace:** global (`dt_detail` for helpers)
- **Built into:** `incremental_trainer` (the only includer, `IncrementalTrainingTool.cpp`); real implementation only when `BUILD_FTP_TRANSPORT` (libcurl found), otherwise stubs that throw
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.1.0`, `@adai-reviewed: 2026-10-19`
- **Tests:** [`tests/DataTransportTests.cpp`](../../../../tests/DataTransportTests.cpp) (14 defined; 13 run in a libcurl build, since some stub tests compile only without `BUILD_FTP_TRANSPORT`), [`tests/DataTransportFtpTests.cpp`](../../../../tests/DataTransportFtpTests.cpp) (32), [`tests/StartupSweepTests.cpp`](../../../../tests/StartupSweepTests.cpp) (10, the download-dir sweep in the trainer)
- **Last traced against the code:** 2026-10-19

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

1. **Create** `download_dir`. The local path is `local_path_for(token, dir)` =
   `dir / download_file_name(token.registry_path)` (`ftp_path` if the registry path is empty): a
   16-hex-digit hash of the full registry path, `_`, the basename (TD-337).
2. **Checksum check (TD-338):** if the token has a checksum and a local file exists, its sidecar
   `<local>.checksum` must hold that checksum; otherwise the file (and sidecar) is deleted and
   downloaded again.
3. **Condition B:** if `size_bytes > 0` and a local file of exactly that size exists (and passed step
   2), return it immediately ("`[CLEANUP-B]` … Skipping re-download"). Tokens without a checksum keep
   the size-only check.
4. **Sidecar:** write the token's checksum to `<local>.checksum` before transferring, so a partial
   file is only resumed against the same version.
5. If `size_bytes ≥ large_file_warn_bytes`, log a large-transfer warning.
6. **URL:** `ftp://<user>:<password>@<host>:<port>/<path>`. Each path segment is percent-encoded by
   `url_encode_ftp_path()`. The credentials aren't encoded; they're hex tokens from the server.
7. **Up to 4 attempts** (`kMaxRetries = 3`). Each attempt:
   - Works out the resume offset from the local file: an empty file starts over, an oversized one is
     deleted, a smaller one resumes with `CURLOPT_RESUME_FROM_LARGE`.
   - Opens the file to append (resume) or truncate.
   - Sets `CURLFTPMETHOD_NOCWD` (the server forbids `CWD`, and paths contain `/`), PASV not EPSV, a
     30 s connect timeout, and aborts below 1 B/s for 60 s.
   - With FTPS, sets `CURLUSESSL_ALL` with **peer and host verification off** (self-signed server
     certificate).
   - Retries only `is_transient()` errors: timeout, recv/send error, got nothing, partial file,
     accept timeout. `COULDNT_CONNECT`, login and URL errors fail at once.
8. On final failure, **delete the local file and its sidecar** and throw. Otherwise verify the final
   size (`size_bytes > 0` only); on a mismatch, delete both and throw.

`local_path_for()`, `checksum_path_for()` and `remove_download()` (file + sidecar) are public statics,
so the trainer's cleanup and tests use the same names.

### `fetch_all(resp, download_dir, max_parallel = 4, warn_bytes = 0)`

Groups the tokens by local path first (TD-337), so the TD-205 segment tokens of one file trigger
**one** download, then starts `min(files, max_parallel)` threads that take indexes from an atomic
counter and call `fetch()` once per distinct file. Results come back in `resp.files` order, each
token getting its file's path (segments of one file share it). After all threads join, it rethrows the **first** error in
index order. Files that did download stay on disk.

The trainer (`IncrementalTrainingTool.cpp`, train/resume/retrain and LeJEPA paths) calls it with
the `DatasetConfig`'s (`dcfg`) `download_dir` (`<SESSION_DIR>/downloads` when unset, TD-411)
and `max_parallel_downloads`, all through `resolve_pass_files()`. On an exception it logs "Download
failed" and releases the batch's exact units (TD-333). After the pass `cleanup_downloads()` removes
each download and its sidecar. The trainer's startup sweep (`StartupSweep.hpp`'s `startup_sweep()`,
Conditions A/D/G, tested by `StartupSweepTests`) matches files by the same names. It deletes
zero-byte, trained and orphaned files (with their sidecars), and keeps non-empty pending ones, which
Condition B or a resume later reuses once the checksum matches.

Without `BUILD_FTP_TRANSPORT`, both methods throw "FTP transport not compiled". The trainer only
reaches them when `ftp_server_host` is non-empty.

---

## 3. Local file identity

Both problems below were fixed October 19, 2026 (TD-337, TD-338).

> **Fixed: segments of one file downloaded to the same local path, in parallel, each in full.** The
> local name used to be the basename of `ftp_path` (pinned by the removed `UsesBasenameOfFtpPath`
> test). All three segment tokens of `g/uploads/big.jsonl` mapped to `dl/big.jsonl`, and
> `fetch_all()` ran them concurrently, so the 15.8 MB file crossed the wire three times. Each thread
> also opened the file with `trunc` and deleted it on a size mismatch, so one thread's check could
> delete another's in-progress file. Different files with the same basename (`uploads/` vs
> `datasets/`, or two kinds' sub-pools) collided too. Now names come from the full registry path, and
> `fetch_all()` downloads each file once. Against a real FTP `registry_server` with a 13 MB file in
> three segments, the old code made 3 `RETR`s and the new one 1.

> **Fixed: Condition B trusted a same-name, same-size file.** A local `big.jsonl` of the right size
> but different content was returned without downloading, though the token carried a `checksum`
> (`"<size>_<mtime>"`) that would have caught it. The startup sweep keeps non-empty files, so a
> stale copy survived across passes. Now the `.checksum` sidecar must match. With the real server,
> a rerun reused the download without connecting, and after a same-size edit on the server the next
> fetch downloaded the new content; the old code had kept a locally edited copy.

**Upgrade note:** downloads already on disk under their old basename names don't match the new
names. The first startup sweep after the upgrade deletes them as orphans (condition G), and they are
downloaded once more.

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

**`DataTransportFtpTests.cpp` (32 tests):**
- Condition B: skip, size mismatch, unknown size, and (TD-338) a different or missing checksum;
- `fetch_all` ordering and error propagation, and (TD-337) segment tokens of one file downloading once;
- full-path naming, with same-basename files not colliding;
- URL encoding;
- every `is_transient()` code;
- the retry constants.

**`DataTransportTests.cpp` (14 defined, 13 run with libcurl):** stubs and struct defaults.

**`StartupSweepTests.cpp` (10):** the trainer's download-dir sweep.

**Not covered:**
- FTPS;
- a real transfer with resume or retry.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Local name is the basename: segment tokens of one file download it N times, concurrently, to one path; different files with the same basename collide (§3) | Redundant transfers; a verification race could delete another thread's file; cross-file overwrites | **Fixed October 19, 2026** (was verified) | [TD-337](../../guides/TECHNICAL_DEBT.md#td-337-datatransport-names-downloads-by-basename-so-segments-and-same-named-files-collide) |
| Condition B reuses any same-name, same-size file; the token checksum is ignored (§3) | Stale or wrong data trained silently | **Fixed October 19, 2026** (was verified) | [TD-338](../../guides/TECHNICAL_DEBT.md#td-338-datatransport-reuses-same-size-stale-files-without-checking-the-checksum) |
| FTPS without server verification (§4) | Registry impersonation or MITM can inject training data | **Fixed October 24, 2026** when `FTPS_PINNED_PUBLIC_KEY`/`FTPS_CA_FILE` is set (was by inspection) | [TD-339](../../guides/TECHNICAL_DEBT.md#td-339-ftps-downloads-dont-verify-the-registrys-certificate) |
| No `curl_global_init()`; partial file deleted on failure; `COULDNT_CONNECT` not retried; global `using`/namespace alias in a header (§4) | Possible init race on older libcurl; lost resume; registry restarts fail passes; namespace pollution | By inspection | [TD-340](../../guides/TECHNICAL_DEBT.md#td-340-datatransport-robustness-and-header-hygiene) |
| Segment release on download failure uses bare paths | Already tracked | — | TD-333 |

## Addendum (October 19, 2026): full-path names, one download per file, checksum sidecars (TD-337, TD-338)

A download is now named by `download_file_name(registry_path)` (`RegistryTransport.hpp`): a 16-hex-digit FNV-1a hash of the full registry path, then `_` and the basename. `DataTransport::local_path_for()` gives the path. `fetch_all()` downloads each distinct file once and hands that path to every token of it, so TD-205 segments of one file share one transfer. `fetch()` records the token checksum in `<download>.checksum` before transferring. When the token carries a checksum, an existing file is reused (Condition B) or resumed only if that sidecar matches; otherwise it is deleted and downloaded again. Checked against a real FTP-enabled `registry_server` with a 13 MB file in three segments. The old code made 3 `RETR`s and kept a stale same-size copy. The new code made 1 `RETR`, and it downloaded the file again after a same-size edit on the server.

---

## Addendum (October 24, 2026): FTPS server verification (TD-339)

`DataTransport(FtpsVerification{ca_file, pinned_public_key})` authenticates the registry's FTPS server:
- **Pin:** a `pinned_public_key` (`CURLOPT_PINNEDPUBLICKEY`) is checked on its own and works with the self-signed default certificate.
- **CA file:** a `ca_file` turns on peer and hostname verification.

When either is set, plain FTP is refused, and a missing CA file fails before connecting. When neither is set, FTPS is unverified as
before and a warning is logged once. The trainer reads `FTPS_PINNED_PUBLIC_KEY`/`FTPS_CA_FILE` from `config.trainer.conf` or the
environment (`DatasetConfig`, `resolve_pass_files()`). The §4 "doesn't authenticate the server" gap is closed when one of them is
configured. Tests: `DataTransportFtpsTests` (real `FtpDataServer` + libcurl on loopback).
