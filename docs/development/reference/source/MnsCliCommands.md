# `MnsCliCommands` — Source File Reference

- **Files:** [`src/MnsCliCommands.hpp`](../../../../src/MnsCliCommands.hpp), [`src/MnsCliCommands.cpp`](../../../../src/MnsCliCommands.cpp) (519 lines together)
- **Namespace:** `adai`
- **Built into:** `mns_cli` (with [`src/MnsCliTool.cpp`](../../../../src/MnsCliTool.cpp), which sends the requests and prints the replies)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-09-11` (both files)
- **Tests:** [`tests/mns_cli_commands_test.cpp`](../../../../tests/mns_cli_commands_test.cpp) (51 tests)
- **User guide:** [mns-cli-guide.md](../../../operations/guides/mns-cli-guide.md)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program linked against
> the real functions, and with the real `mns_cli` binary.

---

## 1. What this file is

`mns_cli`'s per-command argument parsing and request building, pulled out of `MnsCliTool.cpp` for testing
(TD-035). Each `build_*_request(args)` is pure: it takes the command's own arguments (and, for `register`, the
loaded `ServiceConfig` for architecture defaults) and returns an `MnsCliRequest`: method, path, JSON body, or
an error message. `MnsCliTool.cpp`'s `cmd_*()` wrappers send it.

This matters more than for most CLIs because **a model's architecture is immutable once registered** (CLAUDE.md:
"set once at `mns_cli register` and is immutable thereafter"), so a mistake in `register` can't be corrected
without deleting the record.

---

## 2. Commands

| Builder | Request | Arguments |
|---|---|---|
| `build_list_request` | `GET /models[?state=&role=&kind=&limit=]` | `--state`, `--role`, `--kind`, `--limit` |
| `build_get_request`, `build_resolve_request`, `build_delete_request` | `GET /models/{n}`, `GET /models/{n}/resolve`, `DELETE /models/{n}` | `<name>` |
| `build_register_request(args, cfg)` | `POST /models` | `<name> <role>`; `--kind` (`encoder`/`decoder`/`world_model`/`chatbot`, default `chatbot`); `--d-model`, `--num-heads`, `--d-ff`, `--max-seq-length`; `--num-layers` (standalone kinds: sets their own layer field and zeroes the other) or `--encoder-layers`/`--decoder-layers` (legacy chatbot); `--encoder`/`--decoder` (new-style linked chatbot: rejected with other kinds or with explicit architecture flags); `--sigreg-lambda`/`--sigreg-num-sketches` (`world_model` only); `--run-group`; repeated `--tag k=v`. Unset architecture comes from `cfg` |
| `build_link_world_model_request` | `POST /models/{n}/link-world-model` | `<chatbot> --world-model <name>` (empty detaches); `--inject-every-n-layers`, `--hippocampal-enabled`, `--hippocampal-capacity`, four `--hippocampal-*` floats |
| `build_update_run_group_request` | `PUT /models/{n}/run_group` | Exactly `<name> --run-group <value>` |
| `build_set_training_request` | `PUT /models/{n}/state` `{"state":"training","new_run":…}` | `<name> [--new-run] [session-key]` |
| `build_set_candidate_request` | `PUT /models/{n}/state` `{"state":"candidate",…}` | `<name> <run-id>`; `--artifact-path/-host/-checksum/-format` (default `adai-native`), repeated `--summary k=v` |
| `build_roles_request`, `build_resolve_role_request`, `build_promote_request`, `build_health_request` | `GET /roles`, `GET /roles/{r}/production`, `PUT /roles/{r}/production`, `GET /health` | — |
| `parse_url(raw)` | Strips `http://`/`https://` and any path; splits `host:port` (default `localhost:8083`; an unparsable port keeps 8083) | — |
| `json_escape(s)` | Escapes `"`, `\`, `\n`, `\r`, `\t` | — |

---

## 3. Problems (verified)

### Typos in flags are silently ignored, and `register` can't be undone

Every builder skips arguments it doesn't recognize, and a value-taking flag at the end of the line is dropped.
`register m chat --dmodel 256` (a missing hyphen) built a request with the config file's `d_model` of 512 and no
error. Since architecture is immutable, the model is now permanently 512-wide; fixing it means deleting and
re-registering. `list`, `link-world-model` and `set-candidate` behave the same way.

### `set-training` turns any stray argument into the session key

Anything that isn't `--new-run` becomes the metrics session key, the last one winning. `set-training m --new_run`
(an underscore) built `{"state":"training","new_run":false,"metrics_session_key":"--new_run"}`: it **continues the
current run** instead of starting one, and links the model to a metrics session named `--new_run`.

### A malformed number aborts `mns_cli`

The numeric flags use `std::stoul`/`std::stoi`/`std::stof`, and nothing catches their exceptions (neither the
builders nor `MnsCliTool.cpp`'s `main()`). `mns_cli register foo chatbot --d-model abc` printed `terminate called after
throwing an instance of 'std::invalid_argument'` and **aborted with a core dump (exit 134)**. Negative values don't
throw: `--d-model -1` became **18,446,744,073,709,551,615** in the request body (`stoul` wraps), and trailing junk
is truncated (`512x` → 512).

### `--kind encoder` without `--num-layers` sends both layer counts

For `encoder`/`decoder`/`world_model`, the own-kind layer field is set and the other zeroed only when
`--num-layers` is given. Without it, the request carries the config file's `num_encoder_layers` **and**
`num_decoder_layers` (6 and 6 in the probe), which `mns_server`'s per-kind validation rejects (the unused field must be
0). The command fails, but with a server error about layer counts rather than a usage message (the rejection is by
inspection of the server).

---

## 4. Smaller items (by inspection, except where noted)

- **Relinking resets the hippocampal settings.** `link-world-model` always sends all eight connection fields, using
  defaults for any flag not given, and `mns_server` replaces the whole connection. Re-running it to change
  `--inject-every-n-layers` silently turns hippocampal memory off and resets its tuning.
- **Paths and query strings aren't encoded (verified).** `get a/state` requested `/models/a/state`, a different
  endpoint; `list --role "a b&c"` built `/models?role=a b&c`.
- **`json_escape()` misses other control characters (verified).** A `\x01` in a tag value went into the body raw,
  which isn't valid JSON. (Its test, `EscapesQuotesBackslashesAndControlChars`, covers only `\n`, `\r`, `\t`.)
- **Ignored or misplaced flags.** `--sigreg-*` is ignored for kinds other than `world_model`; `update` requires the
  exact order `<name> --run-group <value>`.
- **`parse_url()` accepts `https://`** but `mns_cli` connects with plain HTTP (see `MnsCliTool.cpp`).

---

## 5. Tests

`mns_cli_commands_test.cpp` (51 tests) covers `parse_url()`, `json_escape()`, and every builder: filters and
separators, required arguments, config defaults vs. flags, kinds and `--num-layers`, the
`--encoder`/`--decoder` rules, `world_model`'s SIGReg connection fields, tags, link/detach, `update`'s literal
`--run-group`, `set-training`'s `--new-run`/session key, `set-candidate`'s artifact and summary.

**Not covered:** unknown or misspelled flags, malformed or negative numbers, a standalone kind without
`--num-layers`, characters needing URL encoding.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Unknown flags silently ignored; `register` then uses config defaults for an immutable architecture (§3) | A one-character typo permanently registers the wrong architecture | **Verified** | [TD-448](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-448-mns_cli-ignores-unknown-flags-misreads-set-training-arguments-and-aborts-on-bad-numbers) |
| `set-training` treats any stray argument as the session key (§3) | `--new_run` continues the run and links a bogus metrics key | **Verified** | [TD-448](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-448-mns_cli-ignores-unknown-flags-misreads-set-training-arguments-and-aborts-on-bad-numbers) |
| Unchecked `stoul`/`stoi`/`stof`: aborts with a core dump on bad input, wraps negatives, truncates junk (§3) | Crash instead of a usage error; absurd values sent | **Verified** (real binary) | [TD-448](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-448-mns_cli-ignores-unknown-flags-misreads-set-training-arguments-and-aborts-on-bad-numbers) |
| Standalone kind without `--num-layers` sends both layer counts (§3) | Confusing server rejection | **Verified** (request); by inspection (server) | [TD-449](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-449-mns_cli-request-building-gaps-layer-counts-relink-resets-encoding) |
| Relink resets hippocampal settings; unencoded paths/queries; incomplete `json_escape()`; ignored `--sigreg-*`; `https://` accepted (§4) | Silent config resets; wrong endpoints or invalid JSON for unusual names | Partly verified | [TD-449](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-449-mns_cli-request-building-gaps-layer-counts-relink-resets-encoding) |

---

## Addendum (November 1, 2026): strict options (TD-448)

Every `build_*_request()` with options (`list`, `register`, `link-world-model`, `set-training`,
`set-candidate`) now reads them through one `parse_options()` helper and returns a usage error for:
- **an unknown option** (`Unknown option: --dmodel`). `register m chat --dmodel 256` used to register the
  config's `d_model`, which can't be changed afterwards;
- **a value flag with no value**, at the end of the line or followed by another `--` flag
  (`Missing value for --d-model`). An empty value is still a value, so `--world-model ""` detaches;
- **a stray positional** (`Unexpected argument: 256`);
- **a number that isn't whole and in range.** Counts go through `parse_strict_size()`: architecture sizes,
  `--num-layers`, `--sigreg-num-sketches` and `--hippocampal-capacity` are at least 1, layer counts,
  `--inject-every-n-layers` and `--limit` at least 0, all at most `INT_MAX`. Floats go through
  `parse_strict_float()`: finite, `--sigreg-lambda` and the alphas at least 0, the decays 0–1. `abc` used
  to abort `mns_cli`, `-1` wrapped to 18,446,744,073,709,551,615, and `512x` became 512;
- **a `--tag`/`--summary` without `key=value`**, which used to be dropped.

`set-training` takes `--new-run` and at most one session key; `--new_run` used to become the session key.
In `MnsCliTool.cpp`, a trailing `--url`/`--config` is `Missing value`, and `main()` catches anything still
thrown and exits 2. `register --help` is now an unknown option rather than a request; per-command help
remains TD-451.

Tests: eight new tests in `mns_cli_commands_test.cpp` (59 in all), one of which runs the real binary (the
old one exited 134 on `register foo chatbot --d-model abc`).

## Addendum (November 1, 2026): `register` needs the whole architecture (TD-450)

`build_register_request()` no longer fills architecture fields from its `ServiceConfig` (now used only for the
`run_group` default). Each field comes from a flag or, for those not given, from `--from-config <file>`, loaded with
`ConfigLoader::load()` like the trainer loads its config. A missing field returns an error naming every missing flag
(`Missing: --d-ff, --encoder-layers, ...`) and suggesting `--from-config`. For `--kind encoder`/`decoder`/`world_model`
the own layer count is `--num-layers`, else the file's `NUM_DECODER_LAYERS` (decoder) or `NUM_ENCODER_LAYERS`, and the
other count is 0, which closes TD-449's layer-count item. `--num-layers` with a chatbot and `--from-config` with
`--encoder`/`--decoder` are errors. `MnsCliRequest::notice` carries the line `mns_cli` prints on stderr before
sending, for example `Registering m (chatbot) with d_model=768 num_heads=12 ... (from config.trainer.conf)`.

## Addendum (November 1, 2026): `parse_url()` and exact arguments (TD-451)

`ParsedUrl` gained `https`, `base_path`, `error` and `url_for(path)`. `https://` defaults to port 443 (`http://` and a
bare host keep 8083), a URL path becomes a prefix for every request, `[IPv6]:port` works, and a bad port, a missing host
or another scheme sets `error` (a bad port used to be ignored). `get`, `resolve`, `delete`, `roles`, `resolve-role`,
`promote`, `health` and `update` now reject extra arguments and unknown options (`build_roles_request()` and
`build_health_request()` take the argument list, defaulting to empty); `update` takes `--run-group` in any position.
This matters because `mns_cli` now stops reading global options at the command word, so `delete m --url X` reaches
`build_delete_request()` and is refused instead of deleting `m` on the default server.

## Addendum (November 1, 2026): relink merge, URL encoding, escaping, `--sigreg-*` (TD-449)

`build_link_world_model_request()` sends only the settings given, plus `"merge":true`, so `mns_server` keeps the rest;
it used to send all eight with defaults, so a relink turned hippocampal memory off. `--hippocampal-disabled` (exclusive
with `--hippocampal-enabled`) turns it off explicitly. A new `url_encode()` encodes every name in a path and every
`list` filter value. `json_escape()` writes any other control character below 0x20 as `\u00XX`.
`build_register_request()` refuses `--sigreg-lambda`/`--sigreg-num-sketches` for kinds other than `world_model`.
