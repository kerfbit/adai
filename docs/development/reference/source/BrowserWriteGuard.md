# `BrowserWriteGuard` — Source File Reference

- **Files:** [`src/BrowserWriteGuard.hpp`](../../../../src/BrowserWriteGuard.hpp) (67 lines), [`src/BrowserWriteGuard.cpp`](../../../../src/BrowserWriteGuard.cpp) (62 lines)
- **Namespace:** `adai` (`BrowserWriteRefusal`, `browser_write_refusal()`, `browser_write_refusal_json()`, `is_write_method()`,
  `is_cors_simple_content_type()`, `install_browser_write_guard<Server>()`)
- **Built into:** `adai_core` (the header doesn't include httplib; the installer is a template over the server type)
- **Used by:** [`TrainingMetricsAPI`](TrainingMetricsAPI.md) (with its CORS allow-list), [`RegistryServer`](RegistryServer.md),
  [`ModelNameService`](ModelNameService.md), [`ChatbotAPI`](ChatbotAPI.md), [`TrainerServiceProxy`](TrainerServiceProxy.md) and
  [`TrainerAdminAPI`](TrainerAdminAPI.md) (no allow-list)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-10-24` (both files)
- **Tests:** [`tests/browser_write_guard_test.cpp`](../../../../tests/browser_write_guard_test.cpp) (7 tests, `browserWriteGuardTests`),
  plus the TD-526 route tests in `training_metrics_api_routes_test.cpp`
- **Last traced against the code:** 2026-10-24

> Code-traced reference, written with the file (TD-550, October 24, 2026).

---

## 1. What this file is

One refusal rule for writes a web page could make from an operator's browser, shared by every HTTP daemon. None of them authenticate
requests, and their routes parse bodies whatever the `Content-Type`. A page on any origin can make the browser send a CORS "simple" request
(a POST with `text/plain`, form-encoded or multipart, which needs no preflight). The server executes it even though the page can't read
the reply. That was verified against `registry_server` (`pending/add`) and `metrics_api_server` (`control/clear`, TD-526).

## 2. The rule

`browser_write_refusal(method, origin, content_type, allowed_origins)` refuses only writes (POST/PUT/PATCH/DELETE):

| Request | Result |
|---|---|
| `Origin` present and not in `allowed_origins` | 403 `cross-origin writes are not allowed from '<origin>'` |
| `Content-Type` is `text/plain`, `application/x-www-form-urlencoded` or `multipart/form-data` (parameters and case ignored) | 415 |
| anything else (JSON, `application/octet-stream`, no `Content-Type`) | allowed |

Browsers always send `Origin` on cross-origin requests and on POSTs. The trainer, the CLIs (`dataset_manager`, `mns_cli`), the Android
ops dashboard and `curl` send none, so they're unaffected as long as they don't use a form content type. `curl -d` without `-H` sends
`application/x-www-form-urlencoded` and now gets 415; every documented `curl` example already sets `application/json`. Reads are never
refused, and responses get no CORS headers here, so the browser still can't read them.

`install_browser_write_guard(server, name, allowed_origins = {})` installs this as the server's pre-routing handler. It logs each refusal
at WARN and answers with `{"error":"…"}`. httplib keeps one pre-routing handler per server, so a server that needs its own must call
`browser_write_refusal()` from it.

## 3. Callers

`TrainingMetricsAPI` passes its `METRICS_API_CORS_ALLOWED_ORIGINS` list (explicit origins only; `*` never permits writes) and adds its
own CORS response headers. The other five daemons pass no list, since nothing browser-based writes to them; a future web client would
add its origin there.
