# `ChatbotAPIServer.cpp` — Source File Reference

- **File:** [`src/ChatbotAPIServer.cpp`](../../../../src/ChatbotAPIServer.cpp) (the `main()` of `chatbot_api_server`; no header)
- **Built into:** `chatbot_api_server` (with `Config.cpp` and `ChatbotApiServerArgs.cpp`; links `adai_api` and the model libraries); only when `BUILD_API_SERVER` is on
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.7.1`, `@adai-reviewed: 2026-09-19`
- **Tests:** none directly. The argv layer is tested by `chatbotApiServerArgsTests` and the HTTP layer by `chatbotapiTests`; `main()` itself is untested
- **Related references:** [ChatbotApiServerArgs.md](ChatbotApiServerArgs.md) (its CLI parsing), [ChatbotAPI.md](ChatbotAPI.md) (the HTTP class it runs), [BPETokenizer.md](BPETokenizer.md)
- **Operator docs:** [chatbot-guide.md](../../../operations/guides/chatbot-guide.md), [SYSTEMD_DEPLOYMENT.md](../../../operations/deployment/SYSTEMD_DEPLOYMENT.md), [configuration_guide.md](../../configuration_guide.md)
- **Last traced against the code:** 2026-10-08

> Code-traced reference. The shutdown-race finding was confirmed against the vendored cpp-httplib
> source (`external/cpp-httplib/httplib.h`); the rest is from reading `main()` and the code it calls.

---

## 1. What this file is

The process entry point for the chat inference service. It wires configuration, the Model Name
Service, the tokenizer, the model (plus the optional draft model, world model, hippocampal memory
and RAG engine) and `ChatbotAPI` together, runs the HTTP server on a background thread, and keeps
the main thread for signal-driven reload and shutdown.

Besides `main()` it defines two functions: `signal_handler()` and `print_usage()`.

### Why it matters

It decides **what model actually answers users**: which weights, which architecture, which
optional components. It's also where operational behaviour lives: what happens when a load fails,
what `SIGHUP` reloads, what's saved on shutdown. Several of its fallbacks are deliberately
tolerant ("warn and continue"). That keeps the service up, but some of them keep it up serving
something other than what the operator asked for (§9).

---

## 2. Startup sequence

```text
 0. OpenMP: use all cores unless OMP_NUM_THREADS is set
 1. Config:  extract --config ─► discover_config_path ─► ConfigLoader::load (file + env)
             ─► apply CLI flags ─► help/error exit ─► validate (vocab path only)
 2. Logger:  file rotation if LOG_FILE_PATH set, else console; set level; print config
 3. MNS (only if built with BUILD_MNS_SERVER and NAME_SERVICE_URL + MODEL_ROLE/MODEL_NAME set):
             resolve role or name ─► model_path from artifact
             ─► architecture (6 fields) overrides local config
             ─► linked world model: artifact path, architecture, connection + hippocampal params
             (any failure: warn, keep local config)
 4. [1/4] Tokenizer: BPETokenizer().load_vocab(VOCAB_PATH)   ← throws to main's catch → exit 1
 5. GPU (if GPU_ENABLED): gpu_try_initialize(device, memory fraction); else CPU
 6. [2/4] Model: EncoderDecoderModel(vocab size, architecture, inject_every_n if world model on)
             ─► load_model(MODEL_PATH)   (failure → warn, RANDOM weights; no path → random)
 7. Draft model (if DRAFT_MODEL_PATH): same architecture, load weights (failure → disabled)
 8. World model (if WORLD_MODEL_ENABLED and its checkpoint dir exists): LeJEPAEncoder, frozen
             ─► hippocampal memory (if enabled): restore <MODEL_PATH>.hippocampal(.assoc)
 9. [3/4] ChatbotAPI(model, tokenizer, port, timeout, draft, K)
             ─► --profile / --batched-inference / --pipeline-inference / --integrated-inference
             ─► set_generation_config(max_gen_length, temperature, top_p, strategy)
10. RAG (if RAG_ENABLED): new LLMEncoder + DocumentStore over RAG_DOCS_PATH/*.txt ─► enableRAG()
11. Signals: SIGINT/SIGTERM → shutdown flag; SIGHUP → reload flag
12. [4/4] server thread: api->start()   (bind failure → server_error, shutdown)
13. main loop: every 100 ms, handle reload; exit on shutdown/server error
14. Shutdown: api->stop(), join, save hippocampal memory, exit 0 (1 if server_error)
```

---

## 3. Functions

### `signal_handler(int signal)`

`SIGINT`/`SIGTERM` set `shutdown_requested`; `SIGHUP` sets `reload_config_requested` and **calls
`adai::Logger::info()`**. Its doc comment says it is "async-signal-safe and only sets atomic flags".
The `SIGTERM` branch is, but the `SIGHUP` branch isn't: the logger (spdlog) takes locks and
allocates. If the signal lands on a thread that's in the middle of logging, it can deadlock or
corrupt the logger. The reload branch in the main loop already logs, so the call can simply move
there.

### `print_usage(const char* program_name)`

Prints configuration precedence and every flag. Its flag list matches
`apply_chatbot_api_server_args()` exactly. Two texts are wrong:

- `--batched-inference` says "(real batching, not per-request)"; it generates one request at a time
  (TD-211).
- The architecture defaults it lists (`--d-model` 512, …) are only fallbacks; MNS overrides them
  when the model resolves.

### `main(int argc, char* argv[])`

The sequence in §2. The following sections cover the parts with non-obvious behaviour.

---

## 4. MNS resolution (step 3)

Compiled only with `BUILD_MNS_SERVER`. With `NAME_SERVICE_URL` and either `MODEL_ROLE` (preferred)
or `MODEL_NAME` set, it creates a `ModelNameClient` and:

1. resolves the role or name, and uses `artifact.path` as `config.model_path` when non-empty;
2. copies the model's architecture from MNS into the six config fields. This is what keeps the
   server checkpoint-compatible with the trainer (CLAUDE.md, "Model architecture is
   MNS-authoritative");
3. **TD-196:** if the chatbot record links a world model, resolves that model's artifact and
   architecture, enables the world model, and copies the injection and hippocampal settings
   (and the structural `sigreg_num_sketches`) from MNS. `sigreg_lambda` is deliberately skipped
   because it only matters for training.

Every failure logs a warning and keeps local config, by design: a server that can't reach MNS
still starts. The consequence is that an MNS outage silently swaps in local `MODEL_PATH` and
architecture values, which then feed step 6's "random init on failure" fallback (§9).

---

## 5. Model, draft model, world model, hippocampal memory (steps 6–8)

- **Main model:** constructed from the (possibly MNS) architecture, then
  `load_model(MODEL_PATH)`. If the load fails (missing file, architecture mismatch, corrupt or
  vocab-less checkpoint; see TD-221), the server **logs a warning and serves the randomly
  initialized model**. With no `MODEL_PATH` it also serves random weights, logging only "training
  required". In both cases `/health` reports `ok`.
- **Draft model** (`--draft-model`, TD-038): same architecture and tokenizer, different weights.
  It's built without world-model injection. A load failure disables speculative decoding for the
  run.
- **World model** (TD-193/196, experimental): only if `WORLD_MODEL_ENABLED` and the checkpoint
  directory (MNS artifact path, else `<SESSION_DIR>/world_model`) exists. It's a frozen
  `LeJEPAEncoder` with its own copy of the vocab. Injection must be decided at model construction,
  which is why step 6 passes `inject_every_n_layers`.
- **Hippocampal memory** (TD-193/194, only inside the world-model branch): capacity `d_model` ×
  `HIPPOCAMPAL_MEMORY_CAPACITY`. It restores `<MODEL_PATH>.hippocampal` and then `.assoc` (in that
  order, which `load_associations()` requires), and swaps to `<MODEL_PATH>.hippocampal.swap`.

> **Hippocampal paths come from `MODEL_PATH` with a fixed suffix.** With an empty `MODEL_PATH`
> (allowed: random-init mode) they become `.hippocampal`, `.hippocampal.assoc` and
> `.hippocampal.swap` in the **current working directory**. Under systemd that's usually `/`
> or the service's working directory. The memory also changes with every response but is saved
> **only on graceful shutdown**, so a crash, `SIGKILL` or the hang in §8 loses everything since the
> last clean stop.

---

## 6. API and RAG setup (steps 9–10)

`ChatbotAPI` gets raw pointers to the model, tokenizer and draft model. `main()` owns all three,
and they outlive the API. Mode flags are applied with independent `if`s, so several can be enabled
at once (TD-243). `--pipeline-inference` failures are caught and the mode is disabled (see
TD-223 for the tokenizer state that can leave behind). The generation config is built from
`max_gen_length`, `temperature`, `top_p` and `strategy`; `top_k` and `beam_width` are never copied
(TD-244).

**RAG** (`RAG_ENABLED`): builds a **new `LLMEncoder`** from the model's architecture and loads the
vocab into it. It indexes every non-empty `*.txt` directly in `RAG_DOCS_PATH` (not recursively,
with the file stem as document ID), then creates `RAGInference` over the **trained** model, with
generation settings from config (including `top_k`, unlike the main path).

> **The RAG retrieval encoder is never given trained weights.** `LLMEncoder::load_weights()`
> exists, but step 10 never calls it. Its random initialization is used to embed every document
> and query (`DocumentStore::addDocument()`/`retrieve()` → `encoder->encode()`), so retrieval ranks
> documents by **untrained** embeddings: at best crude token overlap through random projections,
> not learned similarity. Generation then uses the trained model, so answers look plausible while
> the retrieved context is poorly chosen. The model's own trained encoder
> (`model->get_encoder()`) is available and has the same architecture.

---

## 7. `SIGHUP` reload (main loop)

On the reload flag, `ConfigLoader::reload(config, path, config_mutex)` re-reads the **file and
env** into a fresh config and validates it with `ConfigLoader::validate()`, the only place that
validator runs (TD-241). On success, `main()`:

- applies the new log level;
- rebuilds `GenerationConfig` from `max_gen_length`, `temperature`, `top_p` and `strategy` and
  pushes it to the API. This **discards CLI overrides** (TD-242) and still omits
  `top_k`/`beam_width` (TD-244);
- warns if the port changed (that needs a restart).

Everything else in the reloaded config is silently ignored: session timeout, RAG settings, mode
flags, model and architecture, GPU settings. The "Configuration reload" log suggests otherwise.
Reload also doesn't redo MNS resolution, so the in-memory `config` reverts to local
file values for the model path and architecture. Those aren't used after startup, but they're
wrong if anything ever reads them. `config_mutex` is local to `main()`, and nothing else takes it.

---

## 8. Shutdown

The main loop exits on `shutdown_requested` (signal, or the server thread failing to bind). Then:

1. `g_api_server->stop()`, join the server thread;
2. log that model weights are read-only (not saved);
3. save hippocampal memory and associations if attached (failures logged);
4. exit 0, or 1 if the server failed to start. Any exception during startup → exit 1.

> **Startup shutdown race (confirmed against httplib).** `ChatbotAPI::start()` sets its
> `running_` flag and then calls `httplib::Server::listen()`. httplib's `stop()` does nothing
> unless the server is already inside `listen_internal()` (`if (is_running_) {...}`). A
> `SIGTERM` arriving after the server thread starts but before httplib is listening, e.g. a
> quick `systemctl restart`, gets its `stop()` silently ignored. `listen()` then blocks forever,
> `join()` hangs, and further `SIGTERM`s change nothing because the flag is already set. Only
> systemd's stop timeout or `SIGKILL` ends the process, which also skips the hippocampal save.

---

## 9. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Model weight load failure (or no `MODEL_PATH`) serves a **randomly initialized model** while `/health` says `ok` (§5) | Users get gibberish; monitoring sees a healthy service | By inspection | [TD-246](../../guides/TECHNICAL_DEBT.md#td-246-chatbot_api_server-serves-a-randomly-initialized-model-when-loading-fails) |
| RAG retrieval encoder never loads trained weights (§6) | RAG ranks documents with untrained embeddings | By inspection (`load_weights()` never called) | [TD-247](../../guides/TECHNICAL_DEBT.md#td-247-rag-retrieval-in-chatbot_api_server-uses-an-untrained-encoder) |
| `SIGHUP` handler calls the logger, not async-signal-safe (§3) | Possible deadlock or logger corruption on reload | By inspection | [TD-249](../../guides/TECHNICAL_DEBT.md#td-249-chatbot_api_servers-sighup-handler-isnt-async-signal-safe) |
| Shutdown request between server-thread start and httplib `listen` is lost; join hangs (§8) | Needs `SIGKILL`; skips the hippocampal save | Confirmed against httplib source | [TD-248](../../guides/TECHNICAL_DEBT.md#td-248-chatbot_api_server-can-hang-if-shutdown-arrives-during-server-startup) |
| Hippocampal memory saved only on graceful shutdown; paths derive from `MODEL_PATH`, landing in the working directory when it's empty (§5) | Memory lost on crash; stray files | By inspection | [TD-250](../../guides/TECHNICAL_DEBT.md#td-250-hippocampal-memory-in-chatbot_api_server-is-lost-on-crash-and-can-land-in-the-working-directory) |
| Reload ignores everything except log level and four generation fields, without saying so (§7) | Operators think settings applied when they weren't | By inspection | [TD-251](../../guides/TECHNICAL_DEBT.md#td-251-chatbot_api_server-reload-silently-ignores-most-settings) |
| Startup log lists only 4 of 9 endpoints (missing the batch, export and import routes) | Misleading startup output | By inspection | [TD-252](../../guides/TECHNICAL_DEBT.md#td-252-chatbot_api_server-startup-log-and-usage-text-are-inaccurate) |
| `--batched-inference` help text claims real batching (§3) | Already TD-211 | — | TD-211 |
| CLI and validation issues (atoi, no startup validation, reload drops CLI overrides, top_k/beam_width, multiple modes) | Already TD-240–245 | — | TD-240–245 |
