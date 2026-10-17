# `PortableTime` — Source File Reference

- **File:** [`src/PortableTime.hpp`](../../../../src/PortableTime.hpp) (header-only, 84 lines)
- **Namespace:** `adai`
- **Used by:** `DaemonConfigStore`, `DatasetRegistry`, `ModelNameService`, `RegistryServer`, `FtpDataServer` (formatting), and `SQLiteMetricsDatabase`, `PostgresMetricsDatabase`, `TrainingMetricsAPI` (parsing)
- **Status tag:** `@adai-status: beta` ("TD-160 — new, no test coverage yet beyond the callers it replaces"), `@adai-version: 0.1.0`, `@adai-reviewed: 2026-09-12`
- **Tests:** none of its own
- **Last traced against the code:** 2026-10-16

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the header and the callers' exact parsing logic.

---

## 1. What this file is

Portable replacements for the POSIX-only UTC time functions the daemons use, so they build for MinGW (TD-160):

| Function | Behaviour |
|---|---|
| `gmtime_utc(time, result)` | `gmtime_r()` on POSIX; `gmtime_s()` (reversed arguments) on Windows; `nullptr` on failure |
| `timegm_utc(tm)` | `timegm()` / `_mkgmtime()`: broken-down UTC → `time_t` |
| `strptime_utc(s, separator, tm)` | A from-scratch parser for `YYYY-MM-DD<sep>HH:MM:SS` (`sep` = `'T'` or `' '`) via `sscanf("%d-%d-%d<sep>%d:%d:%d%n")`; returns a pointer past the seconds, or `nullptr`. Used on every platform so there's one parser |

Formatting (`gmtime_utc()` + `strftime`/`put_time`) is straightforward and correct. The parsing side is where the problems are, partly in
`strptime_utc()` itself and partly in how all three parsing callers use it: each copies the same pattern.

```cpp
std::tm tm{}; int ms = 0;
if (auto* end = strptime_utc(s, 'T', &tm)) { if (*end == '.') ms = std::atoi(end + 1); }
return from_time_t(timegm_utc(&tm)) + milliseconds(ms);      // used even when parsing failed
```

---

## 2. Problems (verified)

Run through that exact pattern:

| Input | Parsed as | Problem |
|---|---|---|
| `2026-10-15T10:00:00.120Z` | `…10:00:00.120` | Correct |
| `2026-10-15 10:00:00.12+00` (PostgreSQL's text form of 120 ms) | `…10:00:00.012` | `atoi()` reads the fraction as an integer, so `.12` is 12 ms and `.1` is 1 ms |
| `2026-10-15T10:00:00.123456Z` (microseconds) | `…10:02:03.456` | 123,456 "ms" adds **two minutes** |
| `2026-10-15T12:00:00+02:00` | `…12:00:00` | The UTC offset is ignored: two hours wrong |
| `2026-13-45T25:61:99` | `2027-02-15T02:02:39` | No range checks; `timegm()` silently normalizes |
| `not-a-date`, `2026-10-15` (date only) | `1899-12-31T00:00:00` | A parse failure isn't reported: the zeroed `tm` becomes a timestamp in 1899 |

Where it bites:
- **`GET /api/sessions/{key}/metrics/db-history?from=…&to=…`** (`TrainingMetricsAPI::parse_iso8601()`): a date-only or otherwise unparsable `from`
  silently means "since 1899" (everything) and such a `to` silently means "until 1899" (nothing); an offset like `+02:00` shifts the window by two
  hours. The client gets a 200 with the wrong rows rather than a 400.
- **The PostgreSQL backend** reads back its own `TIMESTAMPTZ` columns in text form, which drops trailing zeros (`.12`, `.1`), so `recorded_at`,
  `created_at` and `last_update_at` come back up to ~0.9 s early. The SQLite backend stores its own fixed `.mmm` format and is unaffected unless a
  row was written elsewhere.

`strptime_utc()` also accepts leading whitespace and signs in every field (`%d`), and `sscanf`'s `%d` has undefined behaviour on integer overflow.

---

## 3. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Parse failures become 1899 instead of an error; offsets ignored; out-of-range fields normalized; whitespace/signs accepted (§2) | `db-history` silently returns everything or nothing for a date-only or malformed bound, and shifts the window for offsets | **Verified** | [TD-477](../../guides/TECHNICAL_DEBT.md#td-477-timestamp-parsing-turns-bad-input-into-1899-ignores-offsets-and-misreads-fractions) |
| Fractions parsed with `atoi()` by all three callers: `.12` → 12 ms, microseconds → minutes (§2) | PostgreSQL timestamps read back up to ~0.9 s early | **Verified** | [TD-477](../../guides/TECHNICAL_DEBT.md#td-477-timestamp-parsing-turns-bad-input-into-1899-ignores-offsets-and-misreads-fractions) |
| No tests; parsing logic duplicated in three callers (§1) | The defects above went unnoticed | By inspection | [TD-477](../../guides/TECHNICAL_DEBT.md#td-477-timestamp-parsing-turns-bad-input-into-1899-ignores-offsets-and-misreads-fractions) |
