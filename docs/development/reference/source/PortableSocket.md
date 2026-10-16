# `PortableSocket` — Source File Reference

- **File:** [`src/PortableSocket.hpp`](../../../../src/PortableSocket.hpp) (header-only, 161 lines)
- **Namespace:** `adai`
- **Used by:** [`src/FtpDataServer.hpp`](../../../../src/FtpDataServer.hpp) only (`registry_server`'s embedded FTP/FTPS server; see [FtpDataServer.md](FtpDataServer.md))
- **Status tag:** `@adai-status: beta` ("TD-161 — new, no test coverage yet beyond the callers it replaces"), `@adai-version: 0.1.0`, `@adai-reviewed: 2026-09-13`
- **Tests:** none of its own
- **Last traced against the code:** 2026-10-16

> Code-traced reference. Behaviour marked **verified** was confirmed with scratch programs on Linux. The Windows branches were reviewed
> only; there's no Windows build or CI here.

---

## 1. What this file is

The thin POSIX/Winsock shim that let `FtpDataServer` build for the MinGW-w64 target (`cmake/toolchains/mingw-w64.cmake`, TD-161), the
socket counterpart of `PortableTime.hpp` (TD-160). It wraps only what actually differs:

| Member | POSIX | Windows |
|---|---|---|
| `socket_t`, `kInvalidSocket`, `kShutdownBoth` | `int`, `-1`, `SHUT_RDWR` | `SOCKET`, `INVALID_SOCKET`, `SD_BOTH` |
| `WinsockGuard` | no-op | `WSAStartup(2.2)` / `WSACleanup()` (refcounted by the OS) |
| `close_socket(s)` | `close()` | `closesocket()` |
| `set_reuse_addr(s)` | `setsockopt(SO_REUSEADDR)` | same, `const char*` option pointer |
| `set_recv_timeout(s, seconds)` | `SO_RCVTIMEO` with `struct timeval` | `SO_RCVTIMEO` with a millisecond `DWORD` |
| `send_bytes()` / `recv_bytes()` | `send(…, MSG_NOSIGNAL)` / `recv()` | `int` length cast |
| `last_socket_error()` | `strerror(errno)` | `"WSA error N"` |

Verified on Linux: `set_recv_timeout()` on a listening socket makes `accept()` return `EAGAIN` after the timeout (1.02 s for 1 s; `FtpDataServer`
relies on this to bound the PASV wait), and `send_bytes()` to a closed peer returns −1 with `EPIPE` instead of raising `SIGPIPE`.

---

## 2. Gaps

### `send_bytes()` can write part of a buffer, and its caller assumes all of it (verified)

`send()` on a blocking socket can return fewer bytes than requested when a signal interrupts it after some data was queued, even with
`SA_RESTART`. A probe sending 8 MiB over a socket pair with a slow reader and one signal mid-send got **219,264 bytes** back. `send_bytes()` passes
the count through, and `FtpDataServer`'s `RETR` loop checks only `sent < 0` and then adds the **full** chunk size to `bytes_sent`, so a short
write silently truncates the transfer while the server reports success. (The trainer's download would then fail on size, or worse, be reused if
sizes happened to match; TD-338.) Control-channel replies ignore the result entirely. A `send_all()` that loops until the buffer is written
(retrying `EINTR`) would close this. In practice it needs a signal during a transfer, which is rare in `registry_server` today.

### Smaller items (by inspection)

- **Ignored results.** `WinsockGuard` ignores a failed `WSAStartup()` (and still calls `WSACleanup()`); `set_reuse_addr()` and `set_recv_timeout()`
  ignore `setsockopt()` failures.
- **Windows `accept()` timeout.** Winsock doesn't apply `SO_RCVTIMEO` to `accept()`, so on Windows the PASV wait it's used to bound would block
  indefinitely (relevant only to the untested MinGW build).
- **`errno` timing.** `last_socket_error()` reads `errno` whenever it's called; callers that log or allocate first can report a later call's error.
- **No `EINTR` handling** in `recv_bytes()`: an interrupted read surfaces as an error and ends the connection.
- **No tests**, and the Windows branches have never been built or run in this repository's environments.

---

## 3. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Short writes passed through; `FtpDataServer` counts the whole chunk as sent (§2) | A signal mid-transfer silently truncates a served dataset file | **Verified** (219 KB of 8 MiB) | [TD-476](../../guides/TECHNICAL_DEBT.md#td-476-portablesocket-short-writes-truncate-ftp-transfers-unchecked-socket-setup) |
| Ignored `WSAStartup()`/`setsockopt()` results; no Windows `accept()` timeout; late `errno`; no `EINTR` retry; no tests or Windows build (§2) | Silent misconfiguration; untested portability | By inspection | [TD-476](../../guides/TECHNICAL_DEBT.md#td-476-portablesocket-short-writes-truncate-ftp-transfers-unchecked-socket-setup) |
| FTP server's own connection/timeout gaps | Already tracked | — | TD-372–TD-377 |
