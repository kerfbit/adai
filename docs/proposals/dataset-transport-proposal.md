# Proposal: Dataset Transport System for Remote Trainers

**Status:** Draft  
**Date:** 2026-06-20  
**Author:** rjv717

---

## Problem Statement

The current registry system coordinates which trainer works on which dataset file using atomic path-based assignment (`acquire` / `release` / `trained`). This works well when all processes share a filesystem, but breaks down when a trainer runs on a remote machine: it receives an absolute path (e.g., `/data/training.txt`) that is only valid on the data-holding host. There is no mechanism to transfer the actual file contents across the network.

---

## Current State

| Component | Responsibility |
|-----------|---------------|
| `RegistryServer` | HTTP coordination daemon — paths and assignment state only |
| `RemoteTransport` | HTTP client for acquire/release/trained — paths only |
| `LocalTransport` | File-lock-based coordination — paths only |
| `DataVersion` | Struct holding `data_file` path, checksum, sample count, trained flag |

**Gap:** The registry knows _which_ files are available and _who_ owns them, but has no way to deliver file contents to a trainer on a different host. Trainers on remote machines silently receive valid-looking paths that resolve to `ENOENT`.

---

## Proposed Solution: FTP Dataset Transport

Add a **DataTransport** layer alongside the existing `RegistryTransport`. The registry continues to use HTTP for coordination (acquire / release / trained). File delivery uses FTP, with the `RegistryServer` issuing one short-lived per-file token per acquired file in the `/acquire` response.

Transfer is **strictly one-way: registry machine → trainer machine.** There is no mechanism for a trainer to push files back.

### Ownership Model

| Machine | Owner | Owns |
|---------|-------|------|
| Registry machine | `RegistryServer` process | `data_dir/` and all source dataset files within it |
| Trainer machine | Trainer process | `download_dir/` and all downloaded files within it |

The registry server's FTP root is `data_dir/`. The trainer creates and manages `download_dir/` locally. Neither side reads from or writes to the other's directory directly.

### Design Principles

1. **Per-file tokens:** each file in an acquire batch receives its own independent token. Tokens are scoped to exactly one file and one `run_id`.
2. **FTP delivery:** the FTP server on the registry machine is the only mechanism for file transfer. HTTP is not used for file content.
3. **Checksum-verified:** the existing `DataVersion::checksum` (size + mtime) is returned alongside each token so the trainer can verify the download before training begins.
4. **Backward-compatible:** localhost trainers continue to read files directly by path. The FTP path is only used when `ftp_server_host` is present in the acquire response.
5. **Unidirectional:** FTP accounts are provisioned read-only (RETR only). No STOR, APPE, DELE, or MKD commands are permitted.

---

## Architecture

```
Trainer Machine                             Registry Machine
┌──────────────────────────────────┐       ┌──────────────────────────────────────┐
│  IncrementalTrainer              │       │  RegistryServer  :8082               │
│    │                             │       │    POST /registry/<group>/acquire     │
│    ▼                             │  HTTP │    POST /registry/<group>/release     │
│  DatasetRegistry                 │◄─────►│    POST /registry/<group>/trained     │
│    RemoteTransport               │       │                                      │
│      acquire() ────────────────► │       │  Acquire response includes           │
│      ◄── [per-file FTP tokens] ──│       │  per-file FTP credentials + checksum │
│                                  │       │                                      │
│  DataTransport (new)             │       │  FtpDataServer  :2121                │
│    fetch(file_token) ──────────► │  FTP  │    RETR <ftp_path> (read-only)       │
│    verify_checksum() ────────────│◄──────│    Virtual users, one per token      │
│                                  │       │    FTP root = data_dir/              │
│  download_dir/  ◄── trainer owns │       │    data_dir/  ◄── registry owns      │
└──────────────────────────────────┘       └──────────────────────────────────────┘
```

---

## FTP Server Design

The `FtpDataServer` is implemented directly in C++ and runs on the registry machine as a sub-service of `RegistryServer` (same binary, separate thread, configurable port defaulting to `2121`). No external FTP daemon (vsftpd, proftpd, pure-ftpd, etc.) is used or required.

### Virtual User Model

Each per-file token maps to a **virtual FTP user** that exists only for the duration of the token's TTL:

| FTP field | Value |
|-----------|-------|
| Username | `adai_<token_id>` (e.g., `adai_a3f9c12d`) |
| Password | `<token_secret>` (random 32-byte hex string) |
| Chroot / accessible path | Exactly one file: `<ftp_relative_path>` |
| Permitted commands | RETR only — all write commands refused |
| Passive mode (PASV) | Enabled by default for NAT traversal |

The FTP server validates credentials on login and restricts each virtual user to RETR of its single assigned file. After a successful RETR or token expiry, the virtual user is removed and the credentials cannot be reused.

### FTP Root

The FTP server's root directory is the registry's `data_dir/`. All `ftp_path` values in acquire responses are paths relative to this root. The trainer never sees absolute paths from the registry machine.

---

## Changes to Existing Endpoints

### `POST /registry/<group>/acquire` — response extension

The response is extended from a flat list of paths to a structured object. Each acquired file has its own token entry.

**Before:**
```json
["path/to/file1.txt", "path/to/file2.txt"]
```

**After:**
```json
{
  "run_id": "trainer-host-4201",
  "ftp_server_host": "192.168.1.19",
  "ftp_server_port": 2121,
  "files": [
    {
      "registry_path": "/data/batch1.txt",
      "ftp_path": "batch1.txt",
      "ftp_username": "adai_a3f9c12d",
      "ftp_password": "e3b0c44298fc1c149af...",
      "checksum": "1048576_1718890000",
      "size_bytes": 1048576,
      "token_expires_utc": "2026-06-20T14:30:00Z"
    },
    {
      "registry_path": "/data/batch2.txt",
      "ftp_path": "batch2.txt",
      "ftp_username": "adai_7d82b301",
      "ftp_password": "f4c2d5e819ab2f...",
      "checksum": "2097152_1718890100",
      "size_bytes": 2097152,
      "token_expires_utc": "2026-06-20T14:30:00Z"
    }
  ]
}
```

Clients that do not use FTP transport ignore the `ftp_*` fields and continue to work as before. When `ftp_server_host` is absent or empty, the trainer falls back to direct file path access (localhost behavior unchanged).

No other registry endpoints change.

---

## Client-Side Changes

### New class: `DataTransport`

```cpp
struct FileToken {
    std::string registry_path;     // original path from registry, used for mark_trained
    std::string ftp_path;          // relative path on FTP server
    std::string ftp_username;
    std::string ftp_password;
    std::string checksum;
    size_t      size_bytes;
    std::string token_expires_utc;
};

struct AcquireResponse {
    std::string              run_id;
    std::string              ftp_server_host;
    int                      ftp_server_port = 2121;
    std::vector<FileToken>   files;
};

class DataTransport {
public:
    // Download one file via FTP to download_dir using its per-file token.
    // Verifies checksum after download. Returns local path.
    // Throws on checksum mismatch, expired token, or FTP error.
    std::filesystem::path fetch(
        const FileToken& token,
        const std::string& ftp_host,
        int ftp_port,
        const std::filesystem::path& download_dir
    );

    // Download all files in the acquire response, up to max_parallel connections.
    // Returns local paths in the same order as resp.files.
    std::vector<std::filesystem::path> fetch_all(
        const AcquireResponse& resp,
        const std::filesystem::path& download_dir,
        int max_parallel = 4
    );
};
```

### `IncrementalTrainer` integration

```cpp
AcquireResponse resp = reg.acquire_pending(run_id);

std::vector<std::filesystem::path> local_paths;

if (resp.ftp_server_host.empty()) {
    // Localhost — read files directly by registry path (existing behavior).
    for (const auto& f : resp.files)
        local_paths.push_back(f.registry_path);
} else {
    // Remote — fetch each file via FTP to trainer's download_dir.
    DataTransport dt;
    local_paths = dt.fetch_all(resp, config.download_dir, config.max_parallel_downloads);
}

bool ok = train_on_files(local_paths, config.base_config.num_epochs);

// Collect registry paths (not local paths) for mark_trained / release.
std::vector<std::string> registry_paths;
for (const auto& f : resp.files)
    registry_paths.push_back(f.registry_path);

if (ok)
    reg.mark_trained(run_id, registry_paths, counts);
else
    reg.release_pending(run_id, registry_paths);

// Trainer cleans up its own download_dir files after commit or release.
for (const auto& p : local_paths)
    std::filesystem::remove(p);
```

---

## Directory Ownership Summary

### Registry machine — `RegistryServer` owns `data_dir/`

- Source dataset files live here, placed by the data pipeline (out of scope for this proposal).
- `RegistryServer` reads files to serve via FTP; it does not modify them.
- `FtpDataServer` uses `data_dir/` as its FTP root.
- No trainer process has write access to this directory.

### Trainer machine — Trainer process owns `download_dir/`

- Created at trainer startup if it does not exist.
- FTP downloads land here as `<download_dir>/<original_filename>`.
- Trainer reads from here during training.
- Trainer deletes files after `mark_trained` or `release_pending` completes.
- No registry process has any access to this directory.

---

## Security Considerations

| Concern | Mitigation |
|---------|-----------|
| Unauthorized file access | Per-file tokens are scoped to one `run_id` and one file path; another `run_id` cannot use them |
| Token replay after use | Token is invalidated server-side after a successful RETR completes |
| Token expiry | Each token carries a UTC expiry timestamp; FTP server rejects login after expiry |
| Path traversal | FTP server chroots each virtual user to their single assigned file; directory listing is disabled |
| Write prevention | FTP server refuses STOR, APPE, DELE, MKD, and RMD for all virtual users |
| Credential interception | Phase 1: acceptable on trusted internal network; Phase 2: FTPS (FTP over TLS) |
| Token storage | Tokens are held only in the server's in-memory virtual-user table; not written to disk; lost on restart (tokens expire before restart is a concern at normal TTLs) |

Transfer is read-only and unidirectional by design. There is no upload path, so there is no mechanism for a compromised trainer to inject files into the registry's `data_dir/`.

---

## Configuration

New keys added to `Config` (alongside the existing `registry_server_url`):

```toml
[registry]
registry_server_url    = "http://192.168.1.19:8082"   # existing — HTTP coordination
ftp_server_port        = 2121                         # new — FTP control port (registry side)
ftp_pasv_port_min      = 50000                        # new — PASV data port range lower bound
ftp_pasv_port_max      = 50099                        # new — PASV data port range upper bound
ftp_token_ttl_minutes  = 30                           # new — per-file token lifetime
data_server_secret     = "change-me-in-production"   # new — HMAC key for token signing

[trainer]
download_dir                = "/var/adai/datasets"    # new — trainer-owned download directory
max_parallel_downloads      = 4                       # new — concurrent FTP connections
large_file_warn_threshold_mb = 500                    # new — log explanation when file exceeds this size (no transfer cap)
```

`ftp_server_host` and `ftp_server_port` are returned in the `/acquire` response and do not need to be configured on the trainer side; the trainer uses the values the registry provides.

---

## Implementation Phases

### Phase 1 — FTP server + per-file token issuance (MVP)

- Implement `FtpDataServer` in C++ within the `RegistryServer` binary:
  - TCP listener thread accepts control connections on the configured port.
  - Per-connection state machine handles the FTP command sequence: USER → PASS → PASV → RETR → QUIT.
  - Only USER, PASS, PASV, PORT, TYPE, RETR, QUIT, and NOOP commands are implemented; all others return `502 Command not implemented`.
  - PASV allocates a port from a configurable range (`pasv_port_min`–`pasv_port_max`); the server reports its own host IP and the selected port in the PASV response. Ports are tracked in a small in-process pool and released when the data connection closes.
- Implement in-memory virtual user table: `unordered_map<string, VirtualUser>` keyed on `ftp_username`, holding `{password, ftp_path, expiry, used}`.
  - `RegistryServer` inserts entries at acquire time and removes them on expiry or after successful RETR.
  - Login validates username + password; post-login RETR validates the requested path matches the user's single assigned `ftp_path`.
- Extend `/acquire` HTTP response to return per-file token structs (see response schema above).
- Add `DataTransport::fetch()` on the trainer side using libcurl in FTP mode.
- Update `IncrementalTrainer` to call `fetch_all()` when `ftp_server_host` is non-empty.
- Verify checksum post-download using the `checksum` field from the token.
- Log a clearly worded message before and after transferring any file whose `size_bytes` exceeds `large_file_warn_threshold_mb`. The pre-transfer log states the filename, size in MB, and that a large transfer is beginning. The post-transfer log states how long the transfer took and the average throughput in MB/s. No size cap is enforced — all `.txt` dataset files are valid regardless of size.
- Trainer cleans up `download_dir` files after commit or release.

### Phase 2 — Reliability

- Resume partial downloads: track bytes received; reconnect using FTP REST command.
- `DataTransport::fetch_all()` with configurable parallelism.
- Automatic retry with exponential backoff on transient FTP errors.
- Startup sweep: trainer deletes stale files in `download_dir` from a previous crashed run before acquiring new work.

### Phase 3 — Security hardening

- FTPS (FTP over TLS) — encrypt credentials and data in transit.
- Token signing: HMAC-SHA256 over `run_id + file_path + expiry`, verified on FTP login.
- Rate limiting: cap concurrent FTP sessions per `run_id`.
- Audit log: every token issuance, FTP login, RETR start/complete, and token expiry logged with `run_id`, `ftp_path`, bytes transferred.

---

## Alternatives Considered

| Alternative | Reason not chosen |
|-------------|-------------------|
| HTTP chunked streaming | Less suitable for large binary files than FTP; FTP's native RETR + REST (resume) is a better fit for dataset-sized transfers |
| Shared NFS / network mount | Works well in a dedicated cluster but requires infrastructure outside the ADAI process; not portable to ad-hoc multi-machine setups |
| Per-batch tokens (one token for all files) | Chosen against: a single compromised or expired token would block the entire batch; per-file tokens allow individual file retries and finer revocation |
| Object storage (S3/Minio) with presigned URLs | Strong long-term option but adds a mandatory external dependency; better as a follow-on transport backend |
| rsync / scp | Requires SSH key management out-of-band; no integration with the acquire/release lifecycle |

---

## Stale Download Cleanup

Files can be left in `download_dir` by several distinct failure conditions. Each condition has a different registry state, local file state, and appropriate recovery action. Cleanup is condition-driven rather than time-driven.

### Condition Matrix

| ID | Condition | Local file state | Registry state | Token state |
|----|-----------|-----------------|----------------|-------------|
| A | Crash or kill **during** FTP transfer | Partial file (`actual_size < size_bytes`) | File assigned to `run_id` | Valid or expired |
| B | Crash or kill **after** download, **before** training begins | Complete file, checksum passes | File assigned to `run_id` | Consumed |
| C | Crash or kill **during** training | Complete file, checksum passes | File assigned to `run_id` | Consumed |
| D | Crash or kill **after** `mark_trained`, before local deletion | Complete file | File marked `trained`, removed from pending queue | Consumed |
| E | Checksum mismatch after download completes | Complete file, checksum **fails** | File assigned to `run_id` | Consumed |
| F | Token expired before or mid-transfer | Absent or partial file | File assigned to `run_id` | Expired, cannot reuse |
| G | Orphaned file from a prior session or different `run_id` | Complete or partial file | No active assignment for current `run_id` | N/A |
| H | Disk write error during download | Partial file | File assigned to `run_id` | Valid or consumed |

---

### Cleanup Procedures by Condition

**Condition A — Partial file, token still valid**

1. Detect: file exists in `download_dir` and `actual_size < size_bytes` from the acquire response, token has not yet expired.
2. Delete the partial file.
3. Re-download the file immediately using the existing token.
4. Log: `"[CLEANUP-A] Partial download detected for <filename> (<actual>/<expected> bytes). Token still valid — retrying transfer."`

**Condition A — Partial file, token expired**

1. Detect: same size check, but token `expires_utc` has passed.
2. Delete the partial file.
3. Call `release_pending(run_id, {registry_path})` to return the file to the unassigned pool.
4. Log: `"[CLEANUP-A] Partial download detected for <filename>. Token expired — releasing to registry queue for re-acquisition."`

**Condition B — Complete valid file, training not yet attempted**

1. Detect: file exists, `actual_size == size_bytes`, checksum passes, registry still shows file assigned to `run_id`.
2. No deletion. Skip re-download.
3. Proceed directly to training using the already-downloaded local file.
4. Log: `"[CLEANUP-B] Found previously downloaded file <filename> ready for training. Skipping re-download."`

**Condition C — Complete valid file, training interrupted**

1. Detect: same as B — file is complete and assigned, but training did not finish.
2. Release the file back to the registry: `release_pending(run_id, {registry_path})`.
3. Delete the local file.
4. Log: `"[CLEANUP-C] Training was interrupted for <filename>. Releasing assignment and deleting local copy for clean re-acquisition."`
5. Rationale: training state is unknown; releasing ensures the file is retrained from a known-good starting point rather than risk a partial training epoch being silently counted.

**Condition D — File trained, local copy not deleted**

1. Detect: file exists in `download_dir` but the registry `/queue` endpoint shows no pending assignment for it (already trained and removed from the queue).
2. Delete the local file immediately.
3. Log: `"[CLEANUP-D] Found locally retained file <filename> already marked trained in registry. Deleting local copy."`

**Condition E — Checksum mismatch**

1. Detect: raised in-process immediately when `DataTransport::fetch()` computes the post-download checksum. Does not require a startup sweep.
2. Delete the corrupt local file.
3. Call `release_pending(run_id, {registry_path})`.
4. Log: `"[CLEANUP-E] Checksum mismatch for <filename> (expected <expected>, got <actual>). File is corrupt — releasing to registry queue. Do not use this file for training."`

**Condition F — Token expired**

1. Detect: raised in-process when the FTP server returns `530 Login incorrect` due to expiry, or when `token_expires_utc` is checked before the connection attempt.
2. Delete any partial local file.
3. Call `release_pending(run_id, {registry_path})`.
4. Log: `"[CLEANUP-F] Transfer token for <filename> has expired. Releasing assignment — file will be re-acquirable by any trainer."`

**Condition G — Orphaned file**

1. Detect: during startup sweep, enumerate all files in `download_dir` and cross-reference against the registry `/runs` endpoint for the current `run_id`. Any file present locally but absent from the current assignment list is orphaned.
2. Delete the orphaned file.
3. Log: `"[CLEANUP-G] Found orphaned file <filename> in download_dir with no active registry assignment for run_id <run_id>. Deleting."`
4. Note: do not call `release_pending` for orphaned files — they belong to a different session or `run_id` and the registry already handles that assignment independently.

**Condition H — Disk write error**

1. Detect: raised in-process when the OS returns an I/O error during FTP data write. Does not require a startup sweep.
2. Delete the partial local file.
3. Call `release_pending(run_id, {registry_path})`.
4. Log available disk space on `download_dir`'s filesystem.
5. Log: `"[CLEANUP-H] Disk write error while downloading <filename>. Available space: <N> MB. Releasing assignment. Free disk space before retrying."`
6. Do not retry automatically. Halt the current acquire cycle and surface the error to the operator.

---

### Cleanup Trigger Points

| Trigger | Conditions handled |
|---------|--------------------|
| Startup sweep (before first acquire) | A (expired token), B, C, D, G |
| In-process, immediately on error | E, F, H |
| Normal teardown (after `mark_trained` or `release_pending`) | Routine deletion — not a failure path |

The startup sweep queries the registry `/runs` endpoint once to get the full assignment state for the current `run_id`, then classifies each file in `download_dir` against the conditions above. In-process handlers fire synchronously at the point of failure without waiting for the next startup.
