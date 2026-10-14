# `MnsCliTool` — Source File Reference

- **File:** [`src/MnsCliTool.cpp`](../../../../src/MnsCliTool.cpp) (331 lines; `mns_cli`'s `main()`)
- **Built into:** `mns_cli`, together with [MnsCliCommands](MnsCliCommands.md) (argument parsing and request bodies)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-11`
- **Tests:** none for this file directly; its request building is tested through `mns_cli_commands_test.cpp`
- **User guide:** [mns-cli-guide.md](../../../operations/guides/mns-cli-guide.md)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real `mns_cli`.

---

## 1. What this file is

The thin shell around [MnsCliCommands](MnsCliCommands.md): global options, config loading, dispatch to the
command builders, one HTTP call, and printing.

```text
mns_cli [--url URL] [--config PATH] <command> [args...]
  1. pull --url/--config out of argv (anywhere on the line); everything else is the command and its args
  2. config = ConfigLoader::load(discover_config_path(--config, "config.mns.conf"))
  3. URL = --url, else config/env NAME_SERVICE_URL, else http://localhost:8083
  4. no command, or first word --help/-h → usage, exit 0
  5. request = build_<command>_request(args[, config]);  error → message on stderr, exit 1
  6. send with httplib (connect 5 s, read 10 s); print the response body on stdout
  7. exit 0 for 2xx, 1 for anything else (including connection failure, reported on stderr)
```

Commands: `list`, `get`, `register`, `update`, `link-world-model`, `resolve`, `set-training`, `set-candidate`,
`delete`, `roles`, `resolve-role`, `promote`, `health`. Their arguments and bodies are in
[MnsCliCommands.md](MnsCliCommands.md) §2.

---

## 2. Gaps

### `register`'s architecture defaults come from the MNS daemon's config

`mns_cli` loads **`config.mns.conf`**, the `mns_server` daemon's file. That's where `register` takes any
architecture field not given as a flag. The repository's `config.mns.conf` has `D_MODEL=512`, `NUM_HEADS=8`,
`MAX_SEQ_LENGTH=1024`, while `config.trainer.conf`'s fallback architecture is `D_MODEL=768`, `NUM_HEADS=12`. So
`mns_cli register my-model chatbot` with no architecture flags registers 512/8 regardless of what the trainer's
config says. Because MNS is authoritative and the architecture is immutable, the trainer then silently
adopts 512/8, contrary to an operator who reads `config.trainer.conf` and expects 768. The usage text doesn't
say where the defaults come from, and [mns-cli-guide.md](../../../operations/guides/mns-cli-guide.md) is out of
date: it says the defaults come from `config.conf` (the pre-split file, now only a fallback after
`config.mns.conf`) "so registration stays consistent with what the trainer will use", which the split
configuration no longer guarantees. (By inspection of the config files and the guide.)

### `https://` URLs connect with plain HTTP to port 8083 (verified)

`parse_url()` strips `https://` but keeps the default port 8083, and `make_http()` always builds a plain
`httplib::Client`. `mns_cli --url https://127.0.0.1 health` reported `Error: connection to 127.0.0.1:8083
failed`. So `mns_cli` can't reach MNS through a TLS endpoint such as the Cloudflare tunnel the Android ops
dashboard uses, and it has no way to send Cloudflare Access headers either. The same gap exists in
`MetricsPushClient` (TD-444).

### Smaller items

- **`--help` only works first (verified).** `mns_cli list --help` sent a `list` request instead of printing
  usage; per-command help doesn't exist (and unknown flags are ignored, TD-448).
- **Global options are taken from anywhere on the line**, so a command argument can't be `--url` or `--config`
  (e.g. a tag value). Unlikely in practice.
- **Uncaught exceptions** from the builders abort the process (TD-448); `main()` has no `try`/`catch`.
- **Hygiene.** `ModelNameClient.hpp` is included but unused; the file comment still mentions `config.conf`; the
  four `http_*()` helpers are copies of each other.

---

## 3. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `register` takes unspecified architecture from `config.mns.conf` (512/8), not the trainer's config (768/12) (§2) | A flagless `register` permanently sets an architecture the operator didn't intend | By inspection (config files) | [TD-450](../../guides/TECHNICAL_DEBT.md#td-450-mns_cli-register-takes-architecture-defaults-from-configmnsconf-not-the-trainers-config) |
| `https://` sends plain HTTP to port 8083; no custom headers (§2) | `mns_cli` can't use a TLS/tunnel endpoint | **Verified** | [TD-451](../../guides/TECHNICAL_DEBT.md#td-451-mns_cli-has-no-https-and-minor-option-handling-gaps) |
| `--help` only as the first word; global options taken anywhere; unused include; stale comment; duplicated helpers (§2) | Minor usability and hygiene | **Verified** (`--help`); by inspection | [TD-451](../../guides/TECHNICAL_DEBT.md#td-451-mns_cli-has-no-https-and-minor-option-handling-gaps) |
| Unknown flags ignored; uncaught number-parsing exceptions | Already tracked | — | TD-448 |
