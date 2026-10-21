# `FtpDataServer` — Source File Reference

- **File:** [`src/FtpDataServer.hpp`](../../../../src/FtpDataServer.hpp) (header-only, 1,064 lines)
- **Namespace:** global (`TokenStore`, `VirtualUser`, `FtpDataServer`), plus `ftp_detail` helpers
- **Built into:** `registry_server` (started with `--ftp-enabled`); `DataTransport.hpp` and `RegistryTransport.hpp` include it for shared types
- **Build flags:** `BUILD_FTPS` (OpenSSL: FTPS, HMAC passwords, `RAND_bytes`, constant-time compare)
- **Status tag:** `@adai-status: beta`, `@adai-version: 0.9.0`, `@adai-reviewed: 2026-09-13`
- **Tests:** [`tests/FtpDataServerTests.cpp`](../../../../tests/FtpDataServerTests.cpp) → `ftpDataServerTests` (62 tests); [`tests/registry_ftp_confinement_test.cpp`](../../../../tests/registry_ftp_confinement_test.cpp) (4 end-to-end tests)
- **Client side:** [DataTransport.md](DataTransport.md) (libcurl downloader in the trainer)
- **Design history:** [dataset-transport-proposal.md](../../archive/dataset-transport-proposal.md) (archived proposal)
- **Last traced against the code:** 2026-10-12

> Code-traced reference. Behaviour marked **verified** was confirmed against a live server built
> from this header with `BUILD_FTPS` (plain FTP and FTPS) and driven by Python's `ftplib`/`ssl`
> clients in scratch tests.

---

## 1. What this file is

A minimal, **read-only FTP/FTPS server embedded in `registry_server`** for delivering dataset files to
remote trainers. When a trainer acquires pending files over HTTP, `registry_server` mints one
**per-file virtual user** (`issue_token()`) and returns its credentials with the file entry. The
trainer then logs in over FTP and `RETR`s exactly that one file.

```text
trainer ──HTTP acquire──→ registry_server ── issue_token(run_id, ftp_path, ttl) ──→ TokenStore
   │                         (returns username/password/expiry + ftp host/port/ftps flag)
   └──FTP: [AUTH TLS] USER/PASS → [PBSZ/PROT P] → TYPE I → PASV → RETR <ftp_path> ──→ FtpDataServer
                                                                     (token marked consumed on success)
```

The intended security properties: a token grants one file, expires after its TTL (default 30
minutes), can't log in again once that file has been fully sent, and is rate-limited per `run_id`.
With `BUILD_FTPS` and `--ftps`, both channels can be encrypted. **Several of these properties don't
hold in practice (§5).**

### Why it matters

It's the only way a remote trainer receives training data when the registry runs in FTP mode. It's a
network-facing service in `registry_server`, so a robustness bug here can take the registry down, and
a confidentiality bug exposes the datasets.

---

## 2. `TokenStore` and `VirtualUser`

| Member | Behaviour |
|---|---|
| `VirtualUser` | `password`, `ftp_path` (relative to `data_dir`), `run_id`, `expiry`, `consumed` |
| `insert(username, user)` | `users_[username] = user`, so **an existing user is overwritten**. Usernames carry only 32 random bits (`adai_` + 8 hex), so a collision (likely after roughly 77,000 live tokens) would silently invalidate another trainer's token. That's far beyond normal load |
| `authenticate(user, pass, &path, &run_id)` | False if unknown, consumed or expired. With `BUILD_FTPS` the comparison is constant-time (`CRYPTO_memcmp`); without it, plain `!=`. **Expiry is checked only at login**: an authenticated session keeps working past its token's expiry |
| `mark_consumed(user)` | Set after a successful `RETR`. It blocks new logins with that token but **not further `RETR`s on the already-authenticated connection**, so "prevent replay" applies across connections only |
| `sweep_expired()` | Erases consumed and expired tokens, audit-logging expiries. Runs once a minute |
| `remove(user)` | Erase. No production caller |

---

## 3. Helpers (`ftp_detail`)

| Function | Behaviour |
|---|---|
| `random_hex(bytes)` | `RAND_bytes()` with `BUILD_FTPS` (TD-040 hardening), otherwise a thread-local `mt19937` seeded from `random_device` (not a CSPRNG) |
| `utc_string(tp)` | ISO-8601 `YYYY-MM-DDTHH:MM:SSZ` via `adai::gmtime_utc` |
| `local_ip()` | Returns `"0.0.0.0"`. No caller |
| `hmac_sha256(key, msg)` | Lowercase hex HMAC-SHA256 |
| `generate_self_signed(ctx)` | RSA-2048, a fixed serial of `1`, CN `ADAI-FTP-Server`, 1-year validity. A new key is generated on every start, so a client can't pin it (and `DataTransport` doesn't verify it anyway, TD-339) |

**About the HMAC passwords.** With `--ftp-secret`, the password is
`HMAC(secret, username:ftp_path:expiry)` instead of 32 random bytes. The header says this makes
"forged tokens rejected even if the issuance algorithm is known", but random passwords are equally
unforgeable. The server stores the password either way and compares it directly, so HMAC adds no
stateless-verification benefit. It's harmless, but the claim is overstated. The secret can only be
passed on the command line, so it's visible in `ps` output (TD-284 covers the config keys being
ignored).

---

## 4. `FtpDataServer`

### Construction, tokens, lifecycle

| Member | Behaviour |
|---|---|
| Constructor | `(data_dir, control_port, pasv_min, pasv_max, advertise_ip = "", server_secret = "", ftps = false, cert = "", key = "", max_sessions_per_run = 4)`. Fills the PASV port pool. No validation (`pasv_min > pasv_max` gives an empty pool, so every `PASV` fails with 425) |
| `issue_token(run_id, ftp_path, ttl_min)` | Mints the user (above) and audit-logs the issuance (never the password). `registry_server` calls it from `handle_acquire()` with `ftp_token_ttl_min` (live-tunable via `PUT /admin/config`). Path containment is enforced by that caller (TD-040), not here |
| `start()` | With FTPS: builds an `SSL_CTX` (TLS 1.2 minimum) from the cert/key files or a self-signed cert. Binds `INADDR_ANY:control_port` (backlog 16) and starts the listener and sweep threads. Failures are logged, and `start()` returns with `running_` false; **there's no return value**, so a caller can't tell whether the server came up |
| `stop()` | Closes the listen socket, then joins the listener and sweep threads. **The sweep thread sleeps for a full minute between checks, so `stop()` takes up to 60 s (verified: 56.9 s and 58.1 s)**, which delays `registry_server` shutdown. **Connection threads are detached and never stopped**, so connections open at shutdown keep running against a destroyed object (use-after-free) |
| Accessors | `tokens()`, `advertise_ip()`, `control_port()`, `ftps_enabled()` |

### The connection state machine

`listener_loop()` accepts connections and spawns **one detached thread per connection, with no limit**.
`handle_connection()` sends `220` and then reads commands with `read_line()`:

- `read_line()` reads **one byte per `recv`/`SSL_read` call** and has **no length limit and no timeout**.
  A 2 MB "line" with no newline was accepted and the connection stayed open (verified). Any client can
  therefore tie up a thread and unbounded memory before logging in.
- An **empty line ends the session**: a stray `\r\n` disconnected the client (verified).

| Command | Behaviour |
|---|---|
| `QUIT`, `NOOP`, `SYST`, `PWD` (always `/`), `TYPE` | `TYPE` is recorded but ignored; transfers are always binary |
| `USER` | Stores the pending user and resets login state. **It doesn't release a rate-limit slot already held by this connection** (below) |
| `PASS` | `authenticate()`, then `acquire_session(run_id)` (421 when the run is at `max_sessions_per_run`, default 4), then audit-logs the login |
| `AUTH TLS`/`SSL`, `PBSZ`, `PROT C`/`P` | With `BUILD_FTPS` and FTPS enabled: TLS handshake on the control socket; `PROT P` sets data-channel TLS |
| `FEAT` | Lists `PASV` (plus `AUTH TLS`, `PBSZ`, `PROT` when FTPS is on). **It always writes with the raw socket, even after `AUTH TLS`**, so `FEAT` on an encrypted control channel corrupts it (verified: the client got `WRONG_VERSION_NUMBER`). libcurl doesn't send `FEAT` after `AUTH`, so `DataTransport` isn't affected today |
| Authenticated only: `PASV` | Takes the **lowest free port** in the pool, binds `INADDR_ANY:port` (backlog 1), and replies with `advertise_ip`, or **`127.0.0.1` when `--ftp-ip` isn't set**, so remote trainers can't connect unless the operator sets it |
| `RETR <path>` | Exact match against the token's path (leading `/` stripped); the file must exist; `PASV` must have been sent. Sends `150`, then `accept()`s on the PASV socket with a 30 s timeout, **from any address**, and streams the file in 64 KiB chunks (TLS-wrapped under `PROT P`). On success it marks the token consumed and sends `226` |
| `SIZE <path>` | Same path check; returns the size |
| `PORT` (502), `CWD`/`CDUP`/listings (550), writes (502), anything else (502) | Refused |

### Verified security gaps

> **The data connection can be stolen.** `RETR` accepts the first connection to the PASV port from
> any host; it doesn't check that the peer matches the control connection's client. A third socket
> that connected to the advertised data port before the real client received the whole file:
> 300,000 of 300,000 bytes, and the server logged `226 Transfer complete` and consumed the token.
> **The same worked under FTPS with `PROT P`**: the server accepts a TLS handshake from any client on
> the data port, so a TLS-capable thief received all 300,000 bytes. PASV ports are allocated
> lowest-first from a small, fixed range (default 50000–50099), so they're easy to watch. Besides the
> leaked data, the real trainer's download fails, and since the token is now consumed, it can't
> retry with that token.

> **FTPS isn't enforced.** With `--ftps` enabled, a client that never sends `AUTH TLS` can log in and
> download in plaintext. Verified: a plaintext `USER`/`PASS`/`RETR` on the FTPS server returned the
> full file. Nothing requires `PROT P` either. `DataTransport` does request TLS (`CURLUSESSL_ALL`), so
> the risk is a downgrade or a misconfigured client, not today's trainer.

> **Rate-limit slots leak on re-login.** `USER` resets the connection's login without releasing the
> slot its earlier `PASS` claimed, and the end-of-session cleanup releases only the latest one. After
> one connection logged in twice and quit, a run with `max_sessions_per_run = 2` and **one** open
> session was refused a second (`421 Too many concurrent sessions`). Each such connection permanently
> leaks a slot until `registry_server` restarts; after `max_sessions_per_run` leaks, that `run_id` can't
> download anything. libcurl doesn't re-login on one connection, so this needs a misbehaving client.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| PASV data connection accepted from any host, with or without `PROT P` (§4) | Anyone who can reach the PASV range can steal dataset files and make the trainer's download fail with a consumed token | **Verified** | [TD-372](../../guides/TECHNICAL_DEBT.md#td-372-ftp-data-connections-are-accepted-from-any-host-so-files-can-be-stolen) |
| `--ftps` doesn't require `AUTH TLS` or `PROT P` (§4) | Credentials and data can go in plaintext on an "FTPS" server | **Verified** | [TD-373](../../guides/TECHNICAL_DEBT.md#td-373---ftps-doesnt-require-encryption) |
| `USER` after `PASS` leaks a rate-limit slot (§4) | A misbehaving client can lock a `run_id` out of FTP until restart | **Verified** | [TD-374](../../guides/TECHNICAL_DEBT.md#td-374-ftp-re-login-leaks-rate-limit-slots) |
| One detached thread per connection, unlimited; no idle or read timeout; unbounded `read_line()`; connection threads outlive `stop()` (§4) | Pre-auth resource exhaustion of `registry_server`; use-after-free at shutdown with open connections | **Verified** (unbounded line, no timeout); by inspection (thread limit, use-after-free) | [TD-375](../../guides/TECHNICAL_DEBT.md#td-375-ftpdataserver-has-no-connection-limits-or-timeouts) |
| `stop()` takes up to 60 s (the sweep thread's sleep) (§4) | Slow `registry_server` shutdown and restarts | **Verified** | [TD-376](../../guides/TECHNICAL_DEBT.md#td-376-ftpdataserver-slow-shutdown-unencrypted-feat-and-silent-start-failures) |
| `FEAT` written unencrypted after `AUTH TLS` (§4) | Breaks the TLS control channel for clients that send it | **Verified** | [TD-376](../../guides/TECHNICAL_DEBT.md#td-376-ftpdataserver-slow-shutdown-unencrypted-feat-and-silent-start-failures) |
| Token expiry checked only at login; consumed tokens can still `RETR` on their open connection; 32-bit usernames silently overwrite on collision (§2) | A token's limits are looser than documented | By inspection | [TD-377](../../guides/TECHNICAL_DEBT.md#td-377-ftpdataserver-token-semantics-and-setup-gaps) |
| `PASV` advertises `127.0.0.1` without `--ftp-ip`; `start()` reports failure only in logs; no constructor validation; empty line disconnects; HMAC benefit overstated; `--ftp-secret` visible in `ps`; `local_ip()`/`remove()` unused (§3–4) | Confusing setup failures and dead code | By inspection | [TD-377](../../guides/TECHNICAL_DEBT.md#td-377-ftpdataserver-token-semantics-and-setup-gaps) |
| FTPS certificate not verified by the client | Already tracked | — | TD-339 |
| FTP/FTPS security keys ignored in `config.registry.conf` (CLI-only) | Already tracked | — | TD-284 |

---

## 6. Tests

`FtpDataServerTests.cpp` has 62 tests, all at the unit level:
- `TokenStore` (authenticate, wrong/unknown/expired/consumed, sweep, overwrite, concurrency);
- `random_hex`, `utc_string`, HMAC helpers;
- `issue_token` (formats, uniqueness, path, zero TTL, `run_id`, HMAC vs. random);
- constructor flags and accessors;
- the shared acquire-response types used by `DataTransport`/`RegistryTransport`.

`registry_ftp_confinement_test.cpp` has 4 tests of the TD-040 fix against `registry_server`'s handlers:
an out-of-tree pending entry is left out of the acquire response, stays unclaimed, and can still be
added for direct filesystem use.

**Not covered:** any live FTP session behaviour in this file's own tests:
- the command state machine;
- PASV/`RETR` and the data-connection peer check;
- FTPS enforcement;
- rate limiting across re-logins;
- timeouts, line length and `stop()` latency.

---

## Addendum (October 21, 2026): data-connection binding and required TLS (TD-372, TD-373, TD-376)

- **Data connection (TD-372):** `cmd_retr()` accepts it through `accept_from_client()`, which takes only a connection from the
  control connection's client IP. Others are logged and closed while it keeps waiting, within 30 s. The PASV backlog is 4. TLS
  session reuse on the data channel isn't required (TD-549).
- **FTPS (TD-373):** with FTPS on, `USER`/`PASS` get 530 until `AUTH TLS`, and `RETR` gets 521 without `PROT P`.
- **`stop()` (TD-376):** wakes the token-sweep thread through a condition variable, so it returns at once instead of after up to 60 s.

Tests: `ForeignDataConnectionGetsNothingAndTheClientStillDownloads`, `PlaintextLoginIsRefusedWhenFtpsIsEnabled`. Also verified end to
end with `registry_server --ftps` and `curl --ssl-reqd --ftp-method nocwd`.

