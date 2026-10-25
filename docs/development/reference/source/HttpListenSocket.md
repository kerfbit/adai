# `HttpListenSocket` — Source File Reference

- **Files:** [`src/HttpListenSocket.hpp`](../../../../src/HttpListenSocket.hpp) (41 lines, header-only template)
- **Namespace:** `adai` (`use_exclusive_listen_port<Server>()`)
- **Used by:** [`RegistryServer`](RegistryServer.md), [`ModelNameService`](ModelNameService.md), [`TrainingMetricsAPI`](TrainingMetricsAPI.md),
  [`ChatbotAPI`](ChatbotAPI.md), [`TrainerServiceProxy`](TrainerServiceProxy.md), [`TrainerAdminAPI`](TrainerAdminAPI.md)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-10-24`
- **Tests:** [`tests/http_listen_socket_test.cpp`](../../../../tests/http_listen_socket_test.cpp) (4 tests, `httpListenSocketTests`)
- **Last traced against the code:** 2026-10-24

> Code-traced reference, written with the file (TD-551, October 24, 2026).

---

## What it does

cpp-httplib's `default_socket_options` sets only `SO_REUSEPORT` on POSIX. A second process started on a port that's already listening
therefore binds without error, and the kernel spreads new connections across both, each with its own locks and in-memory state over the
same data. With two `registry_server` instances, serialized fetches came back as duplicate slices (TD-551).

`use_exclusive_listen_port(server)` replaces the options with `SO_REUSEADDR` alone. An immediate restart over old TIME_WAIT connections
still works, but a second listener's bind fails, so `listen()` returns false and the daemon reports it. On Windows it does nothing;
httplib already sets `SO_EXCLUSIVEADDRUSE`. Call it before `listen()`/`bind_to_port()`. It's a template, so the header doesn't include
httplib.

## What each daemon does when the port is taken

| Daemon | Result |
|---|---|
| `registry_server` | logs "could not listen on port N (already in use?)", exits 1 (TD-494) |
| `mns_server`, `metrics_api_server`, `chatbot_api_server` | report the failed start, exit 1 |
| `trainer_service` | logs "admin proxy failed to bind" and keeps supervising without an admin port (by design; the admin API is optional) |
