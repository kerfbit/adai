# Technical Debt Tracker

This document tracks all known technical debt items, TODOs, and improvement opportunities in the ADAI codebase. Items are prioritized and linked to GitHub issues for tracking.

## Overview

**Last Updated:** September 21, 2026
**Total Items:** 13
**High Priority:** 1
**Medium Priority:** 7
**Low Priority:** 5
**Future Enhancements:** 19
**Resolved Items:** 197
**Deferred Decisions:** 3

**September 25, 2026:** Filed and resolved
[TD-210](../archive/TECHNICAL_DEBT_RESOLVED.md#td-210-add-lejepa-advanced-training-diagnostics-and-fix-a-zero-gradient-norm-bug)
— added richer diagnostics to the LeJEPA training track (masking ratio, predictor/target cosine
similarity, SIGReg per-direction projected variance, activation saturation, attention entropy,
compute-time ratio), mirroring the chatbot path's own "Advanced Epoch Diagnostics" where it
transfers. Research for this surfaced a real, already-shipping bug: `LeJEPAEncoder::train_step()`
calls its internal `update_weights()` (which zeroes gradients) twice before returning, so the
`gradient_norm` field the dashboard has shown since TD-208 has always read ~0.0 — fixed by
capturing the norm from each internal update before its own zeroing. Also found and fixed a second,
independent ordering bug while running a real local end-to-end pass: `run_lejepa_training_pass()`
called every buffered-field setter (`update_lejepa_metrics()`, and now the new advanced-metrics
setters) *after* `metrics_client->end_epoch()`, but `MetricsPushClient::end_epoch()` reads those
buffers synchronously while building its JSON body — so `predictor_loss`/`sigreg_loss` have never
actually reached the dashboard from a real trainer process since TD-208 shipped (only a synthetic
curl round-trip, which posts fields directly in one request, could have missed this). See its own
archive entry.

**September 24, 2026:** Filed and resolved
[TD-209](../archive/TECHNICAL_DEBT_RESOLVED.md#td-209-lejepa-training-pass-had-no-mid-pass-checkpoint)
— `run_lejepa_training_pass()` only saved its checkpoint once, at the very end of the whole pass;
for a now-multi-day run over the full pending pool, any crash lost 100% of progress. Added
periodic saves reusing the existing `AUTO_SAVE_*` config keys, verified against a real kill-and-
resume cycle (not just code review). Also surfaced, but left unfixed as out of scope (doesn't
affect the real ai-machine deployment): in local/standalone mode, the checkpoint directory and the
`world_model`-kind dataset sub-pool collide at the same filesystem path. See its own archive entry.

**September 23, 2026:** Filed and resolved
[TD-208](../archive/TECHNICAL_DEBT_RESOLVED.md#td-208-finished-td-178s-deferred-lejepa-metrics-dashboard-wiring)
— TD-178 deliberately deferred how `predictor_loss`/`sigreg_loss` reach a dashboard to whichever
training loop drives it; that loop (`run_lejepa_training_pass()`, `--objective=lejepa`) did nothing
at all until now, discovered while smoke-testing LeJEPA on real hardware right after TD-207. Wired
the full round-trip — client push, HTTP route, history vectors, DB persistence on both backends,
and Android dashboard display (closing a dead-round-trip gap padding-efficiency's own precedent
left open). Also fixed two bugs found in passing: `handle_post_epoch_end()` applied several
optional per-epoch fields (including the new ones) *after* `end_epoch()` already read/persisted
them, making their history/DB values one epoch stale for as long as those fields have existed; and
a SQLite migration-gate mistake that would have silently skipped adding the new columns on any
database already migrated for TD-013. See its own archive entry.

**September 21, 2026:** Filed and resolved
[TD-207](../archive/TECHNICAL_DEBT_RESOLVED.md#td-207-mid-epoch-auto-save-crashed-on-a-models-first-ever-training-pass-null-model-deref)
— the first-ever real training pass at the new 768-dim/24-decoder-layer architecture crashed with a
SIGSEGV inside checkpoint serialization; root-caused via a local AddressSanitizer repro (no GPU
needed) to `perform_auto_save()` dereferencing `IncrementalTrainer::model`, which isn't populated
until *after* `trainer.train()` returns, whenever it fires mid-epoch. Fixed by saving through the
live `ChatbotTrainer`'s own model instead, mirroring the pre-existing best-model-snapshot callback's
already-correct pattern. See its own archive entry, including a separate unrelated gap found (but
left unfixed) in standalone-mode's pending-file pre-check ignoring TD-202 kind sub-pools.

**September 21, 2026:** Filed and resolved
[TD-206](../archive/TECHNICAL_DEBT_RESOLVED.md#td-206-first-real-gpu-hardware-validation-of-the-sycl-backend--3-latent-bugs-found-and-fixed)
— the first time this codebase's SYCL backend has ever run against physical GPU hardware (an
Intel Arc Pro B60, "ai-machine"), closing the "needs real hardware validation" gap TD-033/TD-050/
TD-059 all flagged. Found and fixed three real bugs no amount of device-less compilation could
have caught: `-fsycl` was scoped to a single CMake target instead of applied globally, so
`BUILD_TESTING=ON` had never once been combined with `ENABLE_SYCL=ON` before now; a GPU-resident
incremental decode cache-resize check compared only `.rows`, coincidentally colliding with a
`(1,1)` sentinel on the very first single-token decode step this codebase has ever run (a genuine
crash on real hardware, previously unreachable without a physical device); and six separate test
files checked `GPUManager::probe()` but never called `initialize()`, harmless in a device-less
sandbox but crashing the instant real hardware made `probe()` succeed. Full test suite now passes
on real GPU hardware (93/95 — the other 2 are packaging artifacts of the ad hoc validation script,
not code bugs) and the regular non-SYCL build stays a clean 136/136. See its own archive entry.

**September 20, 2026 (same day):** Filed and resolved
[TD-205](../archive/TECHNICAL_DEBT_RESOLVED.md#td-205-dataset-registry-row-range-segments--ops-dashboard-full-dataset-management-segment)
— the user's own explicit follow-up request, "expand the dataset portion of the opsdashboard into
a full planning and management segment." Added addressable row-range segments within one physical
file at the registry-transport/server/training layers (their own choice over the simpler
client-side-file-splitting alternative), so a large file can be queued/assigned/trained in
independently-manageable slices instead of only as a whole unit — the primitive behind both "use
only part of a dataset" and staged, incremental release via the existing assign/unassign
primitives. Expanded the ops dashboard's Registry tab into a full planning-and-management segment
on top of it: a pool-health overview across all kinds, kind filter chips, full CLI parity
(unassign/delete/manual-add/upload/migrate-to-kind/create-segments), and uniform
`ConfirmActionDialog` gating across every mutating action — including Assign and both Fetch
dialogs, which a self-caught bug during implementation found were still calling the ViewModel
directly despite the screen's own doc comment claiming otherwise. See its own archive entry.

**September 20, 2026 (same day):** Filed and resolved
[TD-204](../archive/TECHNICAL_DEBT_RESOLVED.md#td-204-td-203s-own-dataset_kind-precedence-fix-was-itself-a-regression--reverted)
— a further full-text review pass, this time over TD-203's own diff, found that TD-203's fix for
"incremental_trainer discards a config-file DATASET_KIND" was itself wrong: letting an explicit
config value win over the automatic objective→kind default meant a single static DATASET_KIND
setting would silently apply to every objective run against that config file, reintroducing the
exact chatbot/world_model pending-data cross-contamination TD-202 exists to prevent. Reverted to
the unconditional objective-based default (only `--dataset-kind` may override it), and extracted
the mapping into a new pure, directly-testable function (`adai::resolve_dataset_kind()`) whose
signature has no way to consult a pre-loaded config value at all — closing the actual gap (this
logic previously lived untestable inline in `main()`) that let both the original oversight and
TD-203's bad fix go undetected. See its own archive entry.

**September 20, 2026 (same day):** Filed and resolved
[TD-203](../archive/TECHNICAL_DEBT_RESOLVED.md#td-203-five-flaws-found-in-td-202s-dataset-kind-sub-pool-rollout-by-a-full-text-review-pass)
— a requested full-text review pass over TD-202's own diff found 5 issues, all fixed together:
`incremental_trainer` silently discarded a config-file/env `DATASET_KIND` (the objective-based
default unconditionally overwrote it instead of only applying when unset); `dataset_manager
migrate` silently ignored a global `--kind` flag despite its own `--help` text claiming that flag
scopes "every command"; `--dataset-kind` was undocumented in `incremental_trainer --help`;
`registry_server`'s own doc comment/`--help` were stale about the new kind-scoped routes; and
`DatasetRegistry::add_pending_path_unchecked()` silently also skipped the `is_trained()` guard
`add_file()` performs (and `migrate` never loaded the destination's trained registry, so the
check would have been a no-op even if present). See its own archive entry.

**September 19, 2026 (same day):** Filed and resolved
[TD-202](../archive/TECHNICAL_DEBT_RESOLVED.md#td-202-dataset-registry-gained-per-trainable-piece-sub-pools-dataset_kind)
— user request: "With the variety of models that need training within a single named set we need
to upgrade the dataset registry to account for and organize different datasets being used to for
each trainable piece of the overall model." A `run_group` previously had exactly one shared pending
pool with no way to separate `--objective=lejepa`'s world-model pretraining data from plain chatbot
fine-tuning pairs — confirmed they drew from the exact same pool with zero isolation. Gave each
kind (reusing MNS's own `encoder`/`decoder`/`world_model`/`chatbot` vocabulary from TD-196) its own
physical sub-pool, server-routed via an optional kind segment in each route's regex (the closed
alternation doubles as the validation — an unrecognized kind simply 404s); `incremental_trainer`
now selects its pool automatically from the training objective; `dataset_manager` gained a global
`--kind` flag and a `migrate` command for moving already-queued legacy data into a kind's own pool.
See its own archive entry.

**September 19, 2026 (same day):** Filed and resolved
[TD-201](../archive/TECHNICAL_DEBT_RESOLVED.md#td-201-linkworldmodeldialogs-confirm-preview-showed-a-literal-name-placeholder-instead-of-the-real-chatbot-name)
— found during a second full-text review pass over the TD-199/TD-200 diffs. `LinkWorldModelDialog`
built its `ConfirmActionDialog` preview as the literal string `"POST /models/{name}/link-world-
model"` — an unsubstituted template placeholder, not the real chatbot name — because the dialog's
own signature never received it at all. This broke the explicit "always shows the literal HTTP
call" guarantee both this dialog's and `ConfirmActionDialog`'s own doc comments state, right before
an operator authenticates and commits to the action. The sibling "Detach world model" action for
the identical endpoint, built inline in `ModelDetailScreen.kt`, already interpolated
`model.model_name` correctly — confirming this was an isolated oversight, not a systemic gap.
Fixed by adding a `chatbotName` parameter and interpolating it into the preview; the actual network
call was never affected (it already used the real `modelName` internally). See its own archive
entry.

**September 19, 2026 (same day):** Filed and resolved
[TD-200](../archive/TECHNICAL_DEBT_RESOLVED.md#td-200-registerchatbotdialogs-encoderdecoder-picker-was-starved-by-the-list-screens-own-kind-filter)
— found during a full-text review pass over TD-199's own diff (requested immediately after that
commit landed). `RegisterChatbotDialog`'s Encoder/Decoder pickers (Linked mode) were built by
filtering `ModelListUiState.models` — but that list is always restricted to whichever kind-filter
chip is currently selected on the Models screen (`ModelListViewModel.refresh()` always calls
`listModels(kind = selectedKind)`). Filtering to "Encoder" (or "Chatbot"/"World model") before
tapping "+" → "Register Chatbot" → "Linked" silently emptied the Decoder picker (or both), with no
indication anything was filtered — defeating the exact "compose a chatbot from an encoder+decoder
pair" flow TD-199 was built for. Fixed by giving the register-chatbot flow its own dedicated,
unfiltered fetch (`ModelListViewModel.loadChatbotLinkCandidates()`), mirroring
`ModelDetailViewModel.loadWorldModelCandidates()`'s own already-correct pattern from the same
commit. See its own archive entry.

**September 19, 2026 (same day):** Filed and resolved
[TD-199](../archive/TECHNICAL_DEBT_RESOLVED.md#td-199-android-ops-dashboard-models-section-had-zero-td-196-awareness)
— user request: "the mns section of the opsdashboard... needs to contain a full model design
system." The Android ops dashboard's Models tab predated TD-196 entirely: `ModelRecordDto` had no
`kind`/`connection` field, `MnsApiService` had no register or link-world-model call at all, and
the detail screen only ever rendered the old flat 6-field architecture regardless of what a model
actually was. Brought to full parity with the backend: `ModelRecordDto`/`ArchDto` gained `kind`
and a new `ConnectionDto`; `MnsApiService`/`ModelRepository` gained a `kind` list filter plus
`registerModel()`/`linkWorldModel()`; the list screen gained kind-filter chips and a "+" entry
point into 4 separate per-kind registration dialogs (encoder/decoder/world_model/chatbot, the
chatbot one offering a Linked-via-picker or Legacy-inline-architecture toggle); the detail screen
became kind-aware (a linked chatbot shows clickable Encoder/Decoder rows instead of dead inline
zeros, a "World Model" section shows the link + hippocampal tuning with Link/Detach actions). Per
this app's own established convention, register and link-world-model both route through
`ConfirmActionDialog`'s biometric/PIN gate exactly like every other admin action, previewing the
literal HTTP call first. See its own archive entry.

**September 19, 2026 (same day):** Filed and resolved
[TD-198](../archive/TECHNICAL_DEBT_RESOLVED.md#td-198-mns-first-world-model-resolution-fetched-sigreg_lambda-that-was-never-consumed-on-the-frozen-attach-path)
— found during the same full-text review pass as TD-197. TD-196's MNS-first world-model
resolution blocks in `ChatbotAPIServer.cpp` and `IncrementalTrainingTool.cpp` fetched a linked
world model's own `sigreg_lambda` (via an extra `ModelNameClient::get_connection()` call) and
assigned it to `config`/`svc_config.world_model_sigreg_lambda` — but that field is never read
anywhere on the actual frozen-attach path (`IncrementalTrainer::maybe_attach_world_model()`
freezes the world model with `set_requires_grad(false)` before any training step could use it,
`IncrementalConfig` has no `world_model_sigreg_lambda` field to carry the value there even if it
did matter, and `sigreg_lambda` only affects `LeJEPAEncoder::train_step()`'s own gradient
weighting — never construction or `load()`). `sigreg_num_sketches`, fetched by the same call, is
genuinely structural (shapes the SIGReg module's own sketch matrices, so it must match whatever
the checkpoint was saved with) and stays. Not a wrong-output bug — a dead, silently-discarded
assignment that read as if sigreg strength were configurable per-attachment when it structurally
isn't for a frozen model. Fixed by removing just the dead assignment, keeping the
`get_connection()` call (still needed for `sigreg_num_sketches`). See its own archive entry.

**September 19, 2026 (same day):** Filed and resolved
[TD-197](../archive/TECHNICAL_DEBT_RESOLVED.md#td-197-mns-handle_register-never-validated-a-chatbots-connectionworld_model_name-bypassing-handle_link_world_models-own-checks)
— found during a full-text review pass over TD-196's own diff (requested immediately after that
commit landed). `handle_register()` parsed a chatbot's `connection.world_model_name`/
`world_model_inject_every_n_layers` straight off the request body with no validation at all,
while `handle_link_world_model()` enforced both "target must be `kind==world_model`" and the
`d_model`-match constraint. Confirmed live (not just by reading the diff): registering a chatbot
with `connection.world_model_name` pointing at a dimensionally-incompatible or wrong-kind record
succeeded with `201`, persisting exactly the inconsistent pairing the link endpoint exists to
reject — a real path to a downstream `CrossAttention` dimension mismatch once such a chatbot is
served or trained, not just an API-consistency nit. Fixed by factoring the shared checks into a
new `validate_world_model_link()`, called from both `handle_register()` and
`handle_link_world_model()` so the two can no longer drift apart. See its own archive entry.

**September 19, 2026:** Filed and resolved
[TD-196](../archive/TECHNICAL_DEBT_RESOLVED.md#td-196-mns-model-kind-schema--encoderdecoderworld-model-connection-standard)
— user request: since the world model, encoder, and decoder are separate information-flow
systems, MNS needed to account for the design of each piece as well as the connection standard
between them. Previously MNS treated every registered model as shape-identical (one flat 6-field
`ModelArchitecture`), with a world model registered under `role: "world_model"` using
`--decoder-layers 0` as an undocumented "encoder-only" convention (TD-184) and zero
cross-referencing between a chatbot's MNS record and its world model's — the entire
encoder/decoder/world-model "connection standard" (injection cadence, the `d_model` compatibility
constraint, every `HIPPOCAMPAL_*` setting) lived only in local config files kept in sync purely by
operator discipline across two binaries. Added a `kind` column (`encoder`/`decoder`/`world_model`/
`chatbot`, default `chatbot` — every pre-existing row keeps working unchanged) and a
`connection_json` column holding a new `ModelConnection` struct (`encoder_name`/`decoder_name`,
world-model link + hippocampal tuning, sigreg params). Encoder and decoder are now independently
registrable entities; a `chatbot` is either legacy-bundled (unchanged) or a named pairing of a
linked encoder + decoder, validated dimensionally compatible by the server at register time (409 on
mismatch). New `POST /models/{name}/link-world-model` attaches/replaces/detaches a chatbot's world
model, server-validating `d_model` compatibility (409 on mismatch) whenever injection is enabled.
`ModelNameClient::get_architecture()` needed no signature or call-site change — it transparently
composes the 6-field `ModelArchitecture` from the linked encoder+decoder for a new-style chatbot,
or falls back to inline fields for a legacy one, so `ChatbotAPIServer.cpp`/
`IncrementalTrainingTool.cpp`'s existing call sites are untouched. Also closed the TD-184 gap:
both binaries now resolve a linked world model's own artifact path/architecture/connection
settings from MNS first, falling back to local config only when unlinked or MNS is unreachable.
Verified live end-to-end against a real `mns_server` (register encoder/decoder/world_model,
register/link a chatbot with deliberate dimension mismatches confirming 409s, composed
`get_architecture()` verified via a standalone client probe) plus 18 new `MNSLiveTests` handler
tests and the full `ctest` suite. See its own archive entry.

**September 18, 2026 (same day):** Filed and resolved
[TD-195](../archive/TECHNICAL_DEBT_RESOLVED.md#td-195-layernormfeedforwardmultiheadattentionlora-backward-overwrote-gradients-instead-of-accumulating-them-silently-breaking-gradient_accumulation_steps)
— reported by the user, following up on a background-task suggestion filed earlier this session
during a LeJEPA-batch code review. `LayerNorm`/`FeedForward`/`MultiHeadAttention`/`LoRA`'s own
`backward()` methods overwrote their gradient members (`=`) instead of accumulating (`+=`) across
calls — confirmed via a precise standalone repro using the real `EncoderDecoderModel` production
API: an "accumulated" run (`zero_grad()` once, 3 `forward()`+`backward_pass()` calls, one
`update_weights()`) produced an effective gradient bit-identical to the *last* sample alone, not
the sum of all three. Since `ChatbotTrainer.cpp`'s own `GRADIENT_ACCUMULATION_STEPS` (live default:
32) relies on exactly this accumulation pattern, this meant 31 of every 32 samples' gradient
contributions to every attention/feed-forward/layer-norm weight in the entire model (encoder and
decoder) were silently discarded on every optimizer step — a severe, load-bearing correctness bug
in the main, non-experimental training pipeline, not just the LeJEPA batch. Fixed by changing the
overwriting assignments to accumulate, confirmed via the same repro (now matches the sum) and the
full `ctest` suite. See its own archive entry.

**September 18, 2026 (same day):** Filed and resolved
[TD-194](../archive/TECHNICAL_DEBT_RESOLVED.md#td-194-hippocampalmemory-cross-reference-association-layer)
— user request: a memory cross-reference layer — memories used close together should strengthen
their cross-reference, and a pre-existing activation routine should gate pulling in
cross-referenced memories. Added a `size() x size()` association matrix to `HippocampalMemory`,
kept in lockstep with `slots_`/`coverage_`; `DecoderBlock`'s existing gated hippocampal
cross-attention path both writes it (Hebbian-strengthens any two slots attended together in the
same step, "used close together") and reads it (a `tanh`-gated boost — the same activation
`gate_h`'s own blend already uses — added into the existing `score_bias` mechanism, pulling a
slot toward attention when it's linked to something recently used). New opt-in
`cross_reference_alpha`/`association_decay` parameters default to off, threaded through every
existing call site with no behavior change when unused. Persisted separately from `save()`/
`load()`'s own state file via `save_associations()`/`load_associations()`, wired into
`chatbot_api_server`'s TD-193 lifecycle alongside the main hippocampal state. See its own archive
entry.

**September 18, 2026 (same day):** Filed and resolved
[TD-193](../archive/TECHNICAL_DEBT_RESOLVED.md#td-193-hippocampalmemory-gains-least-used-eviction-a-reloadable-swap-file-and-persistence-in-chatbot_api_server)
— user request: hippocampal memory should persist from conversation to conversation, and evicted
memories should go to a recoverable "swap" file rather than being discarded. `HippocampalMemory`'s
eviction policy changed from FIFO to least-used (lowest attention `coverage_`, ties broken by
oldest), evicted slots now append to an optional, reloadable swap file (`recall_from_swap()`,
LIFO), and `chatbot_api_server` — which never attached a world model or hippocampal memory at all
before this — now does both (opt-in, `WORLD_MODEL_ENABLED`/`HIPPOCAMPAL_MEMORY_ENABLED`), loading
persisted state on startup and saving it on graceful shutdown. See its own archive entry for the
full design, the live end-to-end verification, and an important caveat found along the way
(hippocampal memory's write-path only fires on beam-search generation, and this session's own
TD-186 pilot already found beam search collapsing to empty output at toy scale).

**September 17, 2026 (same day):** Filed and resolved
[TD-192](../archive/TECHNICAL_DEBT_RESOLVED.md#td-192-hippocampalmemory-coverage-eviction-was-on-instead-of-o1)
— found during the same follow-up review pass as TD-190/TD-191. `HippocampalMemory::slots_` is a
`std::deque` specifically so FIFO eviction (`pop_front()`) is O(1), but the parallel `coverage_`
was a `std::vector<float>`, so its own eviction (`erase(begin())`) was an O(n) shift every time
`write()` evicted at capacity. Changed `coverage_` to `std::deque<float>` to match — same
`operator[]`-only access pattern throughout, so no other code needed to change beyond the one
external caller that named the type explicitly. See its own archive entry.

**September 17, 2026 (same day):** Filed and resolved
[TD-191](../archive/TECHNICAL_DEBT_RESOLVED.md#td-191-hippocampalmemoryload-trusted-an-untrusted-slot-count-before-allocating)
— found during the same follow-up review pass as TD-190. `HippocampalMemory::load()` read its
file's own `num_slots` header field and used it directly in `loaded_coverage.reserve(num_slots)`
with no validation — a corrupted or wrong-format file could hand `reserve()` a negative value
(converted to an enormous `size_t`) or a huge-but-positive one, throwing an unrelated
`std::length_error`/`std::bad_alloc` (or, worse, silently reading past EOF into zero-initialized
garbage for a very long time) instead of the class's own consistent `std::runtime_error`. Now
validated against the file's own actual remaining size before being trusted for anything. See its
own archive entry.

**September 17, 2026 (same day):** Filed and resolved
[TD-190](../archive/TECHNICAL_DEBT_RESOLVED.md#td-190-optimizerstep-gave-a-stale-gradient-parameter-group-a-phantom-momentum-decay-update)
— found during a follow-up review pass over TD-189's own fix. Because `LeJEPAEncoder::train_step()`
now calls `update_weights()` (hence `optimizer_->step()`) twice per call but `predictor` only ever
has a real gradient on one of those two calls, `Optimizer::step_adam()`/`step_sgd_momentum()`
still decayed `predictor`'s stale momentum/velocity into a nonzero weight nudge on the call where
its gradient was exactly zero — a real, if small, artifact of sharing one `Optimizer` across
parameter groups that don't all have a gradient on every call. Fixed generically in `Optimizer`
rather than by decoupling `predictor` from the shared optimizer (which would have left it training
at an unconfigured, default learning rate instead). See its own archive entry.

**September 17, 2026 (same day):** Filed and resolved
[TD-189](../archive/TECHNICAL_DEBT_RESOLVED.md#td-189-lejepaencodertrain_step-silently-dropped-the-target-viewsigreg-gradient-for-every-weight-but-token_embedding)
— found during a full-text review of the LeJEPA batch's new files. `train_step()` called
`backward()` twice (once per augmented view) before a single `update_weights()`; since
`LayerNorm`/`FeedForward`/`MultiHeadAttention::backward()` all overwrite (not accumulate) their
own gradient members, the second call silently discarded whichever gradient the first call
computed for every `encoder_blocks`/`final_norm` weight — only `TokenEmbedding::backward()`
actually accumulates. In practice `sigreg_lambda` had zero effect on any weight but the token
embedding table, regardless of its value. See its own archive entry for the fix and the
regression test that reproduces it against the pre-fix code.

**September 17, 2026:** Filed and resolved
[TD-187](../archive/TECHNICAL_DEBT_RESOLVED.md#td-187-decoderblocksaveload-did-not-persist-the-gated-world-modelhippocampal-paths)
the same day — closes TD-180's own documented "gated world-model/hippocampal paths not persisted
by `DecoderBlock::save()`/`load()`" gap. Never spent time as an active entry in this document; see
its own archive entry for the full writeup.

**September 17, 2026 (same day):** Filed and resolved
[TD-188](../archive/TECHNICAL_DEBT_RESOLVED.md#td-188-chatbottrainerpreprocess_data-crashes-uncaught-on-a-pair-with-an-empty-input-or-response)
— `ChatbotTrainer::preprocess_data()` crashed the whole process (uncaught `TokenizerInputError`,
undefined behavior escaping its own OpenMP parallel region) on any pair with an empty input or
response, e.g. a queued JSONL line with only an `"input"` field. Also never spent time as an
active entry; see its own archive entry.

**September 15, 2026:** Filed TD-174 through TD-186 (13 items) — the construction pieces of
[lejepa_world_model_gated_injection_plan.md](../../proposals/lejepa_world_model_gated_injection_plan.md),
a proposed (not yet implemented) design for a self-supervised world-model encoder and a
hippocampal-style episodic memory, both injected into `DecoderBlock` via gated cross-attention.
Unlike most entries in this tracker, these were not found by investigating existing code — they
don't exist yet — so each entry's Description points at the proposal doc as the design source
rather than reporting an investigation. Sequenced into
[Tier 10](#recommended-execution-order) of the Recommended Execution Order below (added
September 15, 2026, same day) — re-derived directly from each entry's own "Depends on" statement
rather than assumed from the proposal doc's own bullet order.

## Recommended Execution Order

Analyzed September 12, 2026 to sequence the active items by dependency and risk rather than
just priority label — several depend on each other or on a single owner decision, and the
"Medium/Low" labels alone don't capture that. Re-derive this ordering rather than trusting it
blindly once several of these items have moved. (For the current exact count and breakdown, see
the Overview and Statistics sections — deliberately not restated here as a number, since this
prose goes stale every time an item resolves or a new one is filed while the tiering below does
not.)

**Tier 1 — Code fix landed, retrain still outstanding.** [TD-059](#td-059-multi-head-and-cross-attention-never-actually-split-into-heads)'s
owner decision landed September 12, 2026 (fix the attention math for real and retrain everything,
not document the current single-head behavior as-is), and the code side of that landed September
13, 2026 — `MultiHeadAttention`/`CrossAttention` now genuinely split into heads on both CPU and GPU
paths, verified via an independent-reference test and finite-difference gradient checks, full
ctest green. What's still outstanding is the actual retrain: this dev environment has no real
trained checkpoint or dataset to retrain against, so that step needs the user's own training
infrastructure. [TD-050](#td-050-gpu-resident-kv-cache-for-autoregressive-generation)'s incremental
attention kernels are now unblocked to build against final math (the code fix is in);
[TD-033](#td-033-chatbot_api_server-inference-never-uses-persistent-gpu-resident-decode) was never
blocked on this in the first place (its routing work doesn't touch attention math).

**Tier 2 — Contained, high-confidence wins** (proven patterns or small isolated scope, no design
ambiguity, can start immediately regardless of Tier 1's outcome):
- [TD-033](#td-033-chatbot_api_server-inference-never-uses-persistent-gpu-resident-decode) —
  wired in and verified September 13, 2026 (code + concurrency fix + tests); only the before/after
  latency benchmark remains, blocked on real GPU hardware not available in this dev environment.
- TD-053 resolved September 13, 2026 (`/save`/`/load`/auto-save-on-exit implemented for real,
  transporting `ConversationContext::serialize()`/`deserialize()` over two new `ChatbotAPI`
  endpoints) — see [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-053-chatbotclis-save-and-load-commands-are-non-functional-everywhere).
  Its remaining doc-verification item continues as
  [TD-164](#td-164-chatbot-guidemd-needs-a-live-pair-verification-pass).

**Tier 3 — Resolved:** TD-034 resolved September 13, 2026 (real policy-ratio/KL via a
caller-supplied log-prob callback, plus a real `ValueFunction` backward pass) — see
[archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-034-ppooptimizers-core-update-loop-is-a-placeholder-not-real-ppo).
That unblocked
[TD-038](#td-038-advanced-features-tested-in-isolation-never-wired-into-a-shipped-binary)'s
`RewardModel` item, also done September 13, 2026 — user chose the full RLHF fine-tuning loop; new
`src/RLHFTrainer.{hpp,cpp}` drives `RewardModel`/`PPOOptimizer` against a live policy with a real,
tested policy-gradient update (see TD-038's own Update for the mechanism and its disclosed scope
limits). `LoRA`, deferred in TD-038's first pass alongside `Quantization`, was revisited the same
day once TD-059's attention-math fix made touching `MultiHeadAttention`/`CrossAttention`'s forward
pass a well-understood change rather than an open risk — now wired in too (see TD-038's second
Update), catching two more real bugs in `LoRAAdapter` itself along the way. Only `Quantization`
remains deliberately deferred — TD-038 itself needs no separate pick.

**Tier 4 — Sustained, low-risk test-coverage investment** (systematic, already-validated pattern,
no open design questions): [TD-048](#td-048-android-uidientry-point-classes-are-untested-and-unreleased)
(Compose UI testing now adopted in both modules — **all 12/12 screens done as of September 13,
2026, including the admin-action confirm-dialog flows on all 5 screens that had one** (see the
"admin-dialog infrastructure, built at last" update), **plus both apps' DI containers and
`Application`/`Activity` entry points** (see the "DI containers and entry points" update) —
`app`-module coverage verified live on a real device (`Medium_Phone_API_35`), `opsdashboard`
compile-verified only, since this sandbox can't install/run that module's debug APK on a device at
all (`wear-sdk` shared-library requirement, no Wear-capable device available here) — what's left
is the `wearcomplications` services (untouched, a distinct area) and real-device verification of
every `opsdashboard` piece once a Wear-capable device is available)
and [TD-037](#td-037-no-qt-test-infrastructure-for-gui-classes) — the logic-extraction half is
done (39 new tests across `MnsJsonHelpers.hpp`/`ChatbotGuiLogic.hpp`); only full widget-level
testing (would need revisiting the QTest decision) remains, not currently planned.

**Tier 5 — Resolved except hardware validation:** [TD-050](#td-050-gpu-resident-kv-cache-for-autoregressive-generation)
(4-6h remaining). The CPU-side root-cause step is done (a real incremental-vs-full-recompute
comparison found no `DecoderKVCache` bug, and the greedy-decoding workaround built around that
assumption was removed), and the GPU-resident cache itself is designed and implemented the same
day — `GPUKVCache`/`GPUDecoderKVCache` plus incremental `gpu_forward_with_cache()`/
`gpu_decode_step()` methods throughout the attention/decoder stack, built entirely from
already-verified primitives (no new kernels), compile-verified under both the `gpu` and `sycl`
presets. What remains is purely the on-device numeric validation and latency benchmark — both
blocked on real GPU hardware not available in this environment, same as TD-033's own remaining
item.

**Tier 6 — Process, not code:** [TD-047](#td-047-android-datarepositoryapi-layer-has-no-ci-or-release-history)'s
remaining items are cutting the first real Android release (small, whenever desired) and
resolving the `origin` repo's GitHub billing issue (on the account holder, not something to fix in
a coding session). [TD-039](#td-039-core-trainingmetrics-classes-too-large-and-fast-moving-to-certify-stable)
has no fixed action — it resolves once the trainer/metrics API surface stops changing, not before.

**Tier 7 — Standalone features, lowest urgency:** [TD-006](#td-006-fill-in-the-middle-fim-training-data-generation)
(6-8h, FIM training data) and [TD-014](#td-014-llm-operations-and-training-tooling-suite) (no
estimate, likely the largest remaining item). Note TD-014's planned `adai-weights-tool`
(quantization) overlaps with TD-038's deferred Quantization-wiring decision — scope those two
together when either is picked up, rather than designing quantization twice. TD-163 (the
`AttentionHeadBenchmark` "hang") is resolved — see
[Deferred Decisions](#attentionheadbenchmarks-apparent-hang--confirmed-host-cpu-contention-not-a-bug).

**Tier 8 — Newly filed, same "tested but never wired" shape as TD-038's original list:** found via
a follow-up sweep (checking what every non-test, non-example `.cpp` actually includes/calls, not
just what compiles) that turned up three more fully-built, independently-tested classes with zero
production callers. TD-169 (`MetricsTracker`) was the safest of the three — purely additive, no
competing on-disk state — and was wired into `ChatbotTrainer`/`IncrementalTrainer` the same day it
was filed; see its [resolved entry](../archive/TECHNICAL_DEBT_RESOLVED.md#td-169-metricstracker-was-a-fully-built-tested-duplicate-of-chatbottrainers-own-inline-metrics-tracking-never-wired-in).
TD-168 (`CheckpointManager`) turned out to be the opposite case — the owner decision landed on
retiring it, the same way TD-052 retired `ParallelDataLoader`'s broken sibling, once it was clear
`IncrementalTrainer`'s real session-based retention model wasn't a drop-in match for
`CheckpointManager`'s own epoch-keyed one; see its [resolved entry](../archive/TECHNICAL_DEBT_RESOLVED.md#td-168-checkpointmanager-was-a-fully-built-tested-duplicate-of-incrementaltrainers-own-inline-checkpoint-logic-never-wired-in).
TD-170 (`TokenBatchLoader`/`ThreadSafeBatchQueue`) also ended up retired, once reading
`ChatbotTrainer::train_epoch()`'s actual ~700-line implementation showed its real value-adds
(background tokenization prefetch, shuffling, batch grouping for gradient accumulation) were each
already duplicated by existing, working machinery there, and its padding/batch-dimension output
had no model to consume it — see its
[resolved entry](../archive/TECHNICAL_DEBT_RESOLVED.md#td-170-paralleldataloaders-tokenbatchloaderthreadsafebatchqueue-retired--no-production-caller-and-nothing-to-attach-to).
That investigation split off
[TD-171](#td-171-no-batch-dimension-anywhere-in-the-model-stack--real-parallel-batched-training-not-supported):
the underlying reason `TokenBatchLoader` had nowhere to attach is that no layer in this codebase's
model stack has a batch dimension at all — flagged as its own large, multi-session architecture
question, not attempted.

**Tier 9 — Resolved except a live host's own deployment cutover:**
[TD-172](#td-172-incremental_trainers-serve-command-embeds-the-always-on-service-in-the-same-binary-as-its-cli-commands)
(14-20h, matched). `incremental_trainer serve` — the always-on systemd-managed training service —
used to be a branch of the same `main()` that also handles every one-shot CLI command; both design
decisions (**process supervisor** over in-process reuse; **loopback HTTP, supervisor proxies to
the child's own `TrainerAdminAPI`** over a file-based channel) and the full implementation landed
the same day, September 14, 2026 — new `trainer_service` binary, `ChildProcess`/
`TrainerServiceProxy` classes (19 new unit tests), `incremental_trainer resume --admin-port`,
`serve` removed entirely, `scripts/adai-trainer.service`/`install_incremental_trainer.sh`/
CLAUDE.md updated, verified end-to-end against a real training pass. What remains is purely
operational: an actual live deployed host still needs its own systemd unit and binaries updated to
cut over, which is outside a coding session's reach — same category as TD-047's release cut or
TD-033/TD-050's hardware-blocked validation.

**Tier 10 — RESOLVED IN FULL (September 15-16, 2026): large exploratory research batch (LeJEPA
world model + hippocampal memory).** TD-174 through
[TD-186](../archive/TECHNICAL_DEBT_RESOLVED.md#td-186-lejepa--hippocampal-memory-pilot-run) (13
items, filed September 15, 2026 from
[lejepa_world_model_gated_injection_plan.md](../../proposals/lejepa_world_model_gated_injection_plan.md)),
built and pilot-tested across two days — the plan's own Status line called it "research/pilot
stage, not yet scoped for full implementation," and its Risks section treated "the gate never
opens" as a legitimate, useful outcome rather than a failure; TD-186's own pilot run landed
exactly there (no-go at toy scale, mechanism proven sound — see that item's own archive entry).
The sequencing below is kept as a historical record of how 13 interdependent items were actually
staged, re-derived directly from each entry's own "Depends on" statement rather than copied from
the plan doc's own bullet order (which interleaves two logically-parallel tracks — LeJEPA/
"cortical" and hippocampal/"episodic" — into one linear reading sequence):

- **Level 0 — fully standalone, startable immediately, in any order:** all three items originally
  at this level are now resolved, same day (September 15, 2026): **TD-174
  (`forward_with_scores`)** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-174-crossattentionforward_with_scores-score-bias-entry-point)),
  **TD-175 (`SIGReg`)** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-175-sigreg-sketched-isotropic-gaussian-regularization)),
  and **TD-176 (`Predictor`)** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-176-predictor-embedding-space-predictor)) —
  three unrelated, independently-testable pieces with no reason not to have parallelized across
  sessions if more than one had been available.
- **Level 1:** **TD-177 (`LeJEPAEncoder` construction) — resolved September 15, 2026, same day**
  (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-177-lejepaencoder-construction)) — needed
  TD-175/TD-176 to exist as class members even though its own Description said "depends on
  nothing else in this list" (true for the *logic*, not for *compiling*).
- **Level 2 — two independent branches, both resolved same day (September 15, 2026):**
  **TD-178 (`train_step`)** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-178-lejepaencodertrain_step-self-supervised-training-loop)),
  the LeJEPA/cortical branch, and **TD-179 (`HippocampalMemory` buffer)** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-179-hippocampalmemory-buffer)), the
  hippocampal/episodic branch — both only needed TD-177; TD-179 never needed TD-178 either
  (confirmed in its own entry), so the two were independently parallelizable.
- **Level 3 — the synchronization point:** **TD-180 (gated `DecoderBlock` extension) — resolved
  September 15, 2026, same day** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-180-gated-decoderblock-extension-world-model--hippocampal-memory-repetition-penalized))
  — the one item in this batch touching existing production code, needed TD-174/TD-177/TD-179
  (all Levels 0-2, all done) — both branches had landed by the time this was picked up. It never
  needed TD-178: the gated-attention path and the world model's training loop are independent.
  **TD-183 (`--objective=lejepa` mode) — resolved September 16, 2026** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-183-incremental_trainer---objectivelejepa-mode--world-model-config-keys))
  — only needed TD-178 (already done), so it was also Level 3 and ran independently of TD-180 — it
  continued the LeJEPA-training branch, not the decoder-injection one.
- **Level 4 — fans back out now that TD-180 has landed:** **TD-181 (injection-frequency knob)**
  (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-181-sparse-world-model-injection-knob)),
  **TD-182 (`set_world_model()`)** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-182-encoderdecodermodelset_world_model)),
  and **TD-185 (`HippocampalMemory` wiring)** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-185-hippocampalmemory-wiring--config--write-policy-call-site))
  — all resolved (TD-181/182 September 15, 2026; TD-185 September 16, 2026) — all only needed
  TD-180 (TD-185's other dependency, TD-179, was already satisfied by Level 2).
  **TD-184 (World-Model MNS Registration + Checkpointing) — resolved September 16, 2026** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-184-world-model-mns-registration--checkpointing))
  — needed TD-177 and TD-183, both already resolved by the time this was picked up.
- **Level 5 — the pilot, and this batch's actual go/no-go gate: TD-186 — resolved September 16,
  2026** (see
  [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-186-lejepa--hippocampal-memory-pilot-run)).
  Result: **no-go on this pilot's own toy-scale numbers, but not because the machinery is
  broken** — the pilot investigation itself found and fixed two real, previously-invisible gaps
  (no `LLMDecoder`/`EncoderDecoderModel` forward-pass ever actually threaded `world_model_output`
  into a gated `DecoderBlock`; no constructor ever allocated the hippocampal gated path at all)
  before either gate could be exercised for the first time. With both fixed, `mean(tanh(gate))`/
  `mean(tanh(gate_h))` both move under gradient (confirmed both by unit test and by this pilot's
  own live run) but only by ~0.01–0.05 after 40 toy-corpus epochs — noise-level for a 32-dim,
  2-layer model, not a clear "opens meaningfully" signal — and the beam-generation diversity
  comparison the plan's own Evaluation standard calls for came back inconclusive because
  generation collapsed to empty output across *every* configuration tested, baseline included,
  a known risk of teacher-forcing a tiny model on a handful of repetitive synthetic examples with
  no dropout/regularization, not a symptom of the gated paths themselves. Same category of
  environment limit as TD-059's blocked retrain: the code path is now sound end-to-end, but a
  decisive answer needs real production data/checkpoint scale, not available in this sandbox.

This closes out Tier 10 in full — **all 13 items (TD-174 through TD-186) are now resolved**, first
landing September 15, 2026 and closing September 16, 2026. See each item's own archive entry
(linked from its own line above) for the individual verification writeups; TD-186's own entry
carries the pilot's full numbers and the two forward-pass-wiring fixes it found along the way. If
[reasoning_process_plan.md](../../proposals/reasoning_process_plan.md)'s own chunks are ever
picked up, see that plan's interaction notes referenced from TD-180 (`RP-2a`, phase-conditioned
gates) and TD-186 (`RP-3`/`RP-6`, joint pilot) — that plan itself remains unstarted, independent
of this tier closing.

## Table of Contents

- [Overview](#overview)
- [Recommended Execution Order](#recommended-execution-order)
- [Table of Contents](#table-of-contents)
- [Active Technical Debt](#active-technical-debt)
  - [TD-059: Multi-Head and Cross-Attention Never Actually Split Into Heads](#td-059-multi-head-and-cross-attention-never-actually-split-into-heads)
  - [TD-050: GPU-Resident KV-Cache for Autoregressive Generation](#td-050-gpu-resident-kv-cache-for-autoregressive-generation)
  - [TD-033: chatbot_api_server Inference Never Uses Persistent GPU-Resident Decode](#td-033-chatbot_api_server-inference-never-uses-persistent-gpu-resident-decode)
  - [TD-014: LLM Operations and Training Tooling Suite](#td-014-llm-operations-and-training-tooling-suite)
  - [TD-006: Fill-in-the-Middle (FIM) Training Data Generation](#td-006-fill-in-the-middle-fim-training-data-generation)
  - [TD-037: No Qt Test Infrastructure for GUI Classes](#td-037-no-qt-test-infrastructure-for-gui-classes)
  - [TD-038: Advanced Features Tested in Isolation, Never Wired Into a Shipped Binary](#td-038-advanced-features-tested-in-isolation-never-wired-into-a-shipped-binary)
  - [TD-039: Core Training/Metrics Classes Too Large and Fast-Moving to Certify Stable](#td-039-core-trainingmetrics-classes-too-large-and-fast-moving-to-certify-stable)
  - [TD-047: Android Data/Repository/API Layer Has No CI or Release History](#td-047-android-datarepositoryapi-layer-has-no-ci-or-release-history)
  - [TD-048: Android UI/DI/Entry-Point Classes Are Untested and Unreleased](#td-048-android-uidientry-point-classes-are-untested-and-unreleased)
  - [TD-164: chatbot-guide.md Needs a Live-Pair Verification Pass](#td-164-chatbot-guidemd-needs-a-live-pair-verification-pass)
  - [TD-171: No Batch Dimension Anywhere in the Model Stack — Real Parallel Batched Training Not Supported](#td-171-no-batch-dimension-anywhere-in-the-model-stack--real-parallel-batched-training-not-supported)
  - [TD-172: incremental_trainer's `serve` Command Embeds the Always-On Service in the Same Binary as Its CLI Commands](#td-172-incremental_trainers-serve-command-embeds-the-always-on-service-in-the-same-binary-as-its-cli-commands)
- [Resolved Items](#resolved-items) (197 items — see [archive](../archive/TECHNICAL_DEBT_RESOLVED.md); re-derive from the Overview's own Resolved Items count above rather than trusting this number blindly — it has drifted stale before)
- [Future Improvements](#future-improvements)
  - [Performance Optimizations](#performance-optimizations)
  - [Code Quality](#code-quality)
  - [Developer Experience](#developer-experience)
  - [Configuration and Service Management](#configuration-and-service-management)
  - [Logging and Observability](#logging-and-observability)
  - [Container and Deployment](#container-and-deployment)
- [Deferred Decisions](#deferred-decisions)
  - [AMD Radeon / ROCm-HIP GPU Backend — Not Pursued](#amd-radeon--rocm-hip-gpu-backend--not-pursued)
  - [ThreadSanitizer "Data Races" in Matrix.cpp's OpenMP-Parallelized Code — Confirmed Tool Limitation, Not a Bug](#threadsanitizer-data-races-in-matrixcpps-openmp-parallelized-code--confirmed-tool-limitation-not-a-bug)
  - [AttentionHeadBenchmark's Apparent Hang — Confirmed Host CPU Contention, Not a Bug](#attentionheadbenchmarks-apparent-hang--confirmed-host-cpu-contention-not-a-bug)
- [Process Guidelines](#process-guidelines)
  - [Adding New Technical Debt](#adding-new-technical-debt)
  - [Prioritization Criteria](#prioritization-criteria)
  - [Resolving Technical Debt](#resolving-technical-debt)
- [Statistics](#statistics)
  - [By Priority](#by-priority)
  - [By Component](#by-component)
  - [Effort Distribution](#effort-distribution)
  - [Future Enhancements Summary](#future-enhancements-summary)
  - [Deferred Decisions Summary](#deferred-decisions-summary)
- [References](#references)

## Active Technical Debt

### TD-059: Multi-Head and Cross-Attention Never Actually Split Into Heads

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open — code fix landed and verified (September 13, 2026); retraining not performed | Core Model Architecture | September 8, 2026 | 30-50 hours (implementation + full retrain/validation cycle) |

Description:
Found while reading `src/MultiHeadAttention.cpp`/`.hpp` end to end, then confirmed independently
in `src/CrossAttention.cpp`/`.hpp`. **The production attention computation used by every encoder
and decoder layer in this codebase has never actually implemented multi-head attention** — it
computes a single global attention pattern over the full `d_model` width, with `num_heads` having
no effect on the math at all beyond gating the constructor's divisibility check.

Specifically:

- `MultiHeadAttention::forward()` — called by both `EncoderBlock::forward()` (self-attention) and
  `DecoderBlock::forward()` (self-attention) — computes `cached_Q * cached_K.transpose()` where
  `cached_Q`/`cached_K` are the full `[seq_len, d_model]` projections. There is no per-head slicing
  anywhere in this function: no `d_k`-wide column ranges, no per-head loop, nothing. `Matrix::operator*`
  is a plain, general-purpose GEMM (confirmed by reading `Matrix.cpp`) with no head-aware behavior of
  its own, so this really is one attention computation across all `d_model` dimensions at once —
  mathematically identical to single-head attention with a head dimension of `d_model`.
- `MultiHeadAttention::forward_with_cache()` (the KV-cache-based autoregressive decode path) has the
  identical gap: `Q_new * K_full.transpose()` over the full width, no per-head split.
- Both of the above still scale by `1.0f / std::sqrt(static_cast<float>(d_k))` — correct *only* for a
  dot product contracted over `d_k` dimensions. Since the actual contraction here is over `d_model =
  num_heads * d_k` dimensions, the correct scale for what's actually computed would be
  `1/sqrt(d_model)`; using `1/sqrt(d_k)` instead makes the pre-softmax logits too large by a factor of
  `sqrt(num_heads)`, pushing softmax toward more peaked (lower-entropy) distributions than intended —
  a secondary numerical bug riding on top of the missing head split.
- `CrossAttention::scaled_dot_product_attention()` and `CrossAttention::forward_with_cache()` — the
  cross-attention path used by every `DecoderBlock::forward()` call — have the exact same two problems,
  implemented independently (this is a separate class, not inheriting `MultiHeadAttention`).
- `MultiHeadAttention::forward_parallel()` is the **only** method in either file that implements the
  design correctly: it explicitly slices `[h*d_k, (h+1)*d_k)` column ranges per head (`start_dim = h *
  d_k`), runs scaled dot-product attention per head with the correct `1/sqrt(d_k)` scale (correct
  *for that width*), and concatenates — matching the "Attention is All You Need" formulation exactly.
  **Grep confirms zero callers of `forward_parallel()` anywhere outside `MultiHeadAttention.cpp`/`.hpp`
  itself** — it is unreachable dead code, apparently written later (its own doc comment: "Properly
  splits input into num_heads... This provides 2-4x speedup") specifically to add the correct
  behavior, but never wired into `EncoderBlock`/`DecoderBlock` in place of `forward()`.

Why this was never caught: `tests/multiheadattention_test.cpp` has 45+ tests, but every one checks
generic properties — output shape, attention weights summing to 1 per row, non-negativity, gradient
existence, save/load round-trip — that hold identically whether or not heads are actually split.
Nothing compares `forward()`'s output against a hand-computed or `forward_parallel()`-derived
per-head reference, and nothing asserts that changing `num_heads` (with `d_model` fixed) changes the
attention pattern in a way consistent with genuine multi-head behavior. The same is true of
`CrossAttention`'s test suite.

**Decision made (September 12, 2026): fix the math and retrain.** Every model ever trained with
this codebase — every existing checkpoint — was trained under the current
(single-head-over-d_model, mis-scaled) math, and switching `forward()`/`forward_with_cache()` over
to `forward_parallel()`'s approach changes what every layer's `W_q`/`W_k`/`W_v`/`W_o` weights
actually mean at inference time, so every existing checkpoint needs retraining from scratch to be
meaningful under the new math. The owner of model quality chose this over documenting the current
single-head behavior as-is.

**Code fix landed September 13, 2026.** `MultiHeadAttention::forward()` now delegates to
`forward_parallel()`'s per-head logic (kept as a separate public method, not inlined, since
`benchmarks/AttentionHeadBenchmark.cpp` calls it directly to compare the parallel/sequential
branches); `forward_with_cache()`, `backward()`, `gpu_forward()`, and `gpu_backward()` were
rewritten the same way. `CrossAttention` got the identical treatment across all five entry points
(it had no `forward_parallel()` to reuse, so this is new code written directly). Both classes now
cache real per-head post-softmax weights (`cached_head_weights_`) for `backward()` to differentiate
through per-head, instead of one d_model-wide softmax; `get_attention_weights()` returns the
mean-across-heads matrix (still a valid row-stochastic distribution) for callers/hooks that just
want a single summary. The GPU paths extract/scatter each head's column slice using the existing
`matrix_copy_device_to_device_gpu()` primitive rather than new CUDA/SYCL kernels — deliberately,
since this sandbox has the CUDA and SYCL toolchains to *compile* against (confirmed: both the `gpu`
and `sycl` presets build `adai_attention` clean) but no physical GPU device to *run* either backend
on, so new low-level kernel code would have shipped with zero ability to catch a bug in it; reusing
an already-exercised primitive keeps that residual risk bounded. The two unused dead-code
`scaled_dot_product_attention()` private methods (one per class, both TD-059-carriers not called
by anything) were deleted along with the `cached_scores` members they were the only readers of.

Verification: a first regression-test draft (comparing `num_heads=1` vs `num_heads=4` outputs under
shared weights) turned out to be a weak check — `d_k = d_model/num_heads` changes with `num_heads`
regardless of whether heads are genuinely split, and the pre-fix scale factor already varied with
`d_k` on its own, so that draft passed against *both* the pre-fix and post-fix code. Replaced with
`MatchesIndependentPerHeadReference` in both `tests/multiheadattention_test.cpp` and
`tests/crossattention_test.cpp` — an independent, from-scratch reimplementation of the documented
per-head formula, computed from the class's own weights via public accessors, confirmed via the
standard revert-confirm-fail cycle to fail against the pre-fix code and pass against the fix.
Finite-difference gradient checks (`BackwardPassMatchesNumericalGradient`, both classes, `num_heads
> 1`) confirm the rewritten `backward()` is the correct gradient of the rewritten `forward()`.
Existing `EncoderBlockTest`/`DecoderBlockTest` numeric-gradient-check tests continue to pass
(composed correctness through `EncoderBlock`/`DecoderBlock`). Full targeted suites
(`multiheadattentionTests` 60/60, `crossattentionTests` 41/41, `encoderblockTests` 34/34,
`decoderblockTests` 28/28) and the full `ctest -j8` (127/127) all pass clean. `adai_attention`
compiles clean under both the `gpu` (CUDA) and `sycl` (Intel oneAPI) presets — see the residual
GPU-runtime-verification-gap note above.

**Retraining not performed.** This dev environment has no real trained checkpoint or training
dataset to retrain — `training_sessions/` here holds only test-fixture output from `ctest` runs
(`alpha-key`, `both-fields`, etc.), not a real model. Retraining every existing checkpoint under
the new math has to happen on whatever infrastructure holds the real training data and currently-
deployed checkpoints, which is outside this environment/session's reach. This item stays open until
that's done and the before/after quality benchmark below is run.

While investigating: `gpu_forward()`/`gpu_backward()` (not called out in the original filing) had
the identical bug — training already runs on GPU when available (`ChatbotTrainer::train_epoch()`
via `EncoderDecoderModel::gpu_forward()`/`gpu_backward()`, see TD-033's writeup for that call
chain), so this was the actual production training path, not just a documentation gap. Fixed
identically to the CPU path.

Action Items:

- [x] Wire `MultiHeadAttention::forward()`/`forward_with_cache()`/`backward()` through genuine
  per-head logic, and do the same for `CrossAttention`'s three CPU entry points.
- [x] Fix both scale factors — resolved as a side effect of the per-head rewrite; `1/sqrt(d_k)` was
  always the right constant, it was just being applied to the wrong (`d_model`-wide) contraction.
- [x] Fix `gpu_forward()`/`gpu_backward()` identically for both classes (found during this pass —
  not in the original filing's scope, but the same bug, and the actual training-time path).
- [x] Add tests that would have caught the original bug (see Verification above for why the first
  draft wasn't strong enough, and what replaced it).
- [x] Update `MultiHeadAttention.hpp`/`CrossAttention.hpp`'s class-level architecture comments; the
  `forward_parallel()`/`scaled_dot_product_attention()` dead-code question resolved as: keep
  `forward_parallel()` (real external caller in the benchmark), delete `scaled_dot_product_attention()`
  (genuinely unused by anything).
- [ ] Retrain from scratch and validate against a from-scratch baseline before declaring any
  existing deployment upgraded — needs the real training data/infrastructure this session doesn't
  have access to.
- [ ] Benchmark generation quality/perplexity before vs. after on the same held-out data to confirm
  genuine multi-head attention is actually an improvement, not just "different," before retiring
  the old checkpoints.

Files to Modify:

- `src/MultiHeadAttention.cpp` / `src/MultiHeadAttention.hpp` — done
- `src/CrossAttention.cpp` / `src/CrossAttention.hpp` — done
- `tests/multiheadattention_test.cpp` / `tests/crossattention_test.cpp` — done (new
  architecture-sensitive coverage)
- Every existing trained checkpoint (out of source-tree scope, and the real remaining cost driver
  of this item) — not done, needs the user's own training infrastructure

---

### TD-050: GPU-Resident KV-Cache for Autoregressive Generation

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — CPU cache root-caused clean, GPU-resident cache designed/implemented, and the flagged CPU beam-vs-cache gap fixed (all September 14, 2026); correctness validation and benchmark blocked on real GPU hardware | GPU / Inference / Training | July 4, 2026 | 4-6 hours (revised down again — only the hardware-blocked validation/benchmark remain) |

Description:
The GPU decode path has no working incremental KV-cache at all: `EncoderDecoderModel::gpu_generate_response()` (added to GPU-accelerate BLEU/ROUGE scoring during validation) recomputes the full sequence from scratch every decode step via `LLMDecoder::gpu_decode()` — O(n) work per step, O(n^2) total over a generation, instead of O(1) per step / O(n) total with a real cache. This is functionally correct but leaves an easy performance win on the table now that generation runs on GPU.

**Update (September 14, 2026):** the CPU side is resolved. `generate_response_with_strategy()`'s
greedy branch used to bypass `DecoderKVCache`/`forward_with_cache()` entirely via a dedicated
non-cached `greedy_model_fn`, on the assumption (never actually verified) that the cache had a
self-attention/cross-attention indexing bug producing incorrect results specifically for greedy
decoding. The real diagnostic this item's own tracker text had been calling for since September
8 — "an actual multi-step incremental-decode-vs-single-shot-full-recompute numerical comparison
with identical weights" — was finally run: `tests/inference_optimization_test.cpp`'s new
`IncrementalCacheMatchesFullRecomputePerStep` reproduces the real production call pattern (one new
token per `forward_with_cache()` call, one `DecoderKVCache` instance threaded across every step, a
real non-null `encoder_output` so cross-attention is exercised) and compares each step's hidden
state against a from-scratch `forward_with_encoder()` recompute of the identical growing prefix.
Across 40 decode steps (`d_model=128`, 4 layers, 8 heads): max abs diff stays flat at ~1e-6-2e-6
for every single step, with no growth or accumulation trend as the sequence lengthens — the
signature of ordinary float32 summation-order rounding noise, not an algorithmic bug (a real
indexing/masking bug would show either an immediate large divergence or unbounded growth). **No
CPU cache correctness bug exists.** The workaround's premise was wrong; it has been removed —
`generate_response_with_strategy()`'s greedy branch now uses the same cached `model_fn` every
other non-beam strategy already used, closing the O(n) vs O(n^2) gap the workaround left open for
greedy specifically. The previous, much older (pre-existing, disabled) `DISABLED_
CacheOutputConsistency` test this replaces had actually tested something different and much
weaker — a single non-incremental `forward_with_cache()` call with `encoder_output=nullptr` (no
cross-attention at all) against a 0.5-tolerance threshold loose enough to hide a real bug either
way — and never actually confirmed or refuted anything about the real incremental, cross-attention
call pattern production uses.

**Update (September 14, 2026, same day):** the GPU-resident cache itself is designed and
implemented, compile-verified under both the `gpu` (CUDA, `nvcc`) and `sycl` (Intel oneAPI,
`icpx`) presets (this sandbox has both toolchains but no physical GPU device — see TD-059's own
writeup for the identical residual-verification gap this inherits). Followed TD-059's own
precedent throughout: every new method is built from already-existing, already-verified
primitives (`GPUMatrix`'s own operators, `matrix_copy_device_to_device_gpu()`, the per-head
`gpu_slice_head_columns()`/`gpu_scatter_head_columns()` helpers TD-059 itself introduced) — no new
low-level CUDA/SYCL kernels were written, so correctness risk is bounded by what those primitives
already proved, not by new, unverifiable kernel code.

- `adai::gpu::GPUKVCache`/`GPUDecoderKVCache` (`src/gpu/MatrixGPU.hpp`, placed *outside* the
  CUDA-vs-SYCL `#if`/`#else` split so one definition compiles under either backend unchanged):
  unlike the CPU `KVCache` (which reallocates a bigger `Matrix` on every `append()`), this
  pre-allocates a `[max_seq_length, d_model]` device buffer once and grows a length counter in
  place — no per-decode-step device malloc/free churn, per this item's own original ask.
- `MultiHeadAttention::gpu_forward_with_cache()` / `CrossAttention::gpu_forward_with_cache()`:
  mirror the CPU `forward_with_cache()` algorithms exactly (new-token Q/K/V, append to cache,
  attend against the full cached K/V) using `gpu_forward()`'s own already-verified per-head loop
  unchanged. Cross-attention's cache is populated from the encoder output exactly once (first
  call) and simply read back on every later call, matching the CPU cache's identical design.
- `DecoderBlock::gpu_forward_with_cache()` / `LLMDecoder::gpu_decode_step()`: same Pre-LN residual
  structure as the existing `gpu_forward()`/`gpu_decode()`, swapping in the cache-based attention
  calls. `gpu_decode_step()`'s positional-encoding-with-offset math is copied verbatim from
  `Decoder.cpp`'s own CPU `forward_with_cache()` for exact parity (`current_position =
  kv_cache.current_length()`, absolute position = that plus each new token's index).
- `EncoderDecoderModel::gpu_generate_response()`/`gpu_generate_response_with_strategy()`: now use
  `gpu_decode_step()` + one `GPUDecoderKVCache` threaded across a whole generation, replacing the
  previous full-recompute-every-step `gpu_decode()` call, for every strategy **except beam
  search**. Found and closed a real correctness trap while wiring this in: `generate_beam_search()`
  calls `model_fn` once per beam per step with each beam's own *diverging* token sequence — a
  single shared KV cache has no way to correctly serve more than one hypothesis at once, so using
  the cached `model_fn` there would silently corrupt every beam but whichever one happened to
  match the cache's assumed prefix. The CPU `generate_response_with_strategy()` already avoids
  this (its own "beam" branch never uses `DecoderKVCache`); the GPU path previously didn't need
  to care (no cache existed at all, so beam search always used full recompute) but *would* have
  silently broken the moment a shared cache was introduced without this guard. Both GPU methods
  now check for `num_beams > 1` and route to a separate, deliberately non-cached
  `beam_model_fn` (the original full-recompute `gpu_decode()` call) in that case.
- A related, pre-existing gap was found and flagged as a separate follow-up task rather than fixed
  in the moment (out of this item's own GPU-cache scope): the CPU `EncoderDecoderModel::
  generate_response()` (the plain, non-strategy method) had no equivalent beam-vs-cache guard at
  all — it always built one cached `model_fn` regardless, and `TextGenerator::generate()` would
  silently route to beam search using it if `generator`'s persistent config happened to have
  `num_beams > 1` left over from an earlier `generate_response_with_strategy(..., "beam", ...)`
  call.

  **Fixed September 14, 2026** (same day, later): confirmed genuinely reachable, not just
  theoretical — `RAGInference::generateResponse()` calls `generate_response_with_strategy(...,
  /*strategy=*/"", ..., config.gen_config.num_beams)` (see its own TD-101 comment), so any RAG
  caller that ever sets `num_beams > 1` hits exactly this path. Reproduced directly: a
  `generate_response_with_strategy(..., "beam", ..., num_beams=3)` call followed by a plain
  `generate_response()` call on the same instance **segfaults** — worse than silently wrong output.
  Root cause: at beam-search step 0 every beam shares the same starting token sequence and length;
  the first beam's call advances the shared `DecoderKVCache`'s `processed_length` to that length,
  so the *second* beam's call computes an empty "new tokens" slice, producing a 0-row `Matrix`
  whose `rows - 1` the caller then indexes as `-1` — a genuine out-of-bounds access, not merely a
  numerical divergence. Fixed the same way `gpu_generate_response()` already was: check
  `generator->get_config().num_beams > 1` and route to a separate, non-cached `beam_model_fn`
  (mirroring `generate_response_with_strategy()`'s own, vocab-masking included) via
  `generate_beam_search()` directly instead of the generic `generate()`. Also found and fixed the
  identical class of gap one level deeper: `generate_response_with_strategy()`'s own `else`
  (unrecognized-strategy) fallback branch called the generic `generate()` with its *cached*
  `model_fn` too, so a single call with an unrecognized/empty strategy string **and** `num_beams >
  1` — exactly `RAGInference`'s own calling pattern — hit the same corruption in one call, with no
  leftover state from a previous call required. Fixed by widening that branch's guard condition
  from `normalized_strategy == "beam"` to `normalized_strategy == "beam" || num_beams > 1`.
  Checked every other `TextGenerator::generate()`/`generate_text()` call site in the codebase
  (`gpu_generate_response_with_strategy()`, `ChatbotAPI::generate_response()`/
  `generate_batch_responses()`'s CPU fallbacks, `BatchedInferenceEngine::process_batch()`) for the
  same class of risk: all dispatch explicitly per strategy without a generic-`generate()` fallback,
  or use an inherently stateless (no persistent cache) `model_fn`, except
  `BatchedInferenceEngine`, which is generic enough that a *future* caller could hit this if it
  ever passes both a cached `model_fn` and `num_beams > 1` — not reachable today (no caller sets
  `num_beams > 1` through it), so left as a documented contract on `InferenceRequest::model_fn`
  rather than a code change. Added `EncoderDecoderModelTest.
  GenerateResponseNotCorruptedByLeftoverBeamConfig` (`tests/encoderdecoder_test.cpp`): confirmed
  against the pre-fix code that it reproduces the exact segfault (via a scoped `git stash` of just
  the fix, rebuild, run, restore — not merely inferred), and against the post-fix code that
  `generate_response()`'s output with leftover `num_beams > 1` now exactly matches an explicit
  `generate_response_with_strategy(..., "beam", ...)` call (beam search has no sampling RNG, so
  the two safe, non-cached paths are deterministic and must agree). Full `ctest` suite: 129/129
  passing.

Action Items:

- [x] Root-cause the existing CPU `DecoderKVCache` correctness bug (self-attention and/or
  cross-attention cache indexing) in `src/KVCache.hpp` / `src/Decoder.cpp` `forward_with_cache()`
  before building the GPU equivalent on top of the same flawed model. **Resolved: no bug found**
  — see the Update above. `src/KVCache.hpp`/`src/Decoder.cpp` needed no changes; the fix was
  removing the now-unnecessary greedy-decoding workaround in `src/EncoderDecoderModel.cpp`.
- [x] Design a GPU-resident cache type (e.g. `GPUKVCache`) holding persistent per-layer `GPUMatrix` key/value buffers, sized for `max_seq_length` and appended to in-place as new tokens are generated (no per-step malloc_device/free churn). Done — see the Update above.
- [x] Add incremental self-attention kernels that compute Q/K/V for only the newest token(s) and attend against the full cached K/V (mirrors the CPU cache's intent), plus a one-time cross-attention K/V cache populated from the encoder output and reused unchanged across all decode steps. Done — no new kernels needed, built from existing primitives (see Update above).
- [x] Add `LLMDecoder::gpu_decode_step()` (single-token incremental decode using the cache) alongside the existing full-sequence `gpu_decode()` (retained for training's teacher-forced forward pass, which doesn't need a cache). Done.
- [x] Wire `EncoderDecoderModel::gpu_generate_response()` to use the new incremental path instead of recomputing the full sequence every step. Done — for every strategy except beam search, which keeps the full-recompute path deliberately (see Update above).
- [ ] Validate correctness against the existing full-recompute GPU path (identical token-for-token output for greedy decoding) — the CPU path is already confirmed correct and every new GPU method reuses already-verified primitives, but genuine on-device numeric validation still needs real hardware. **Blocked on real GPU hardware** — not available in this environment.
- [ ] Benchmark generation latency before/after for representative `max_length` values (e.g. 50, 100 tokens) to confirm the expected O(n) vs O(n^2) improvement. **Blocked on real GPU hardware**, same as TD-033's own remaining benchmark item — not available in this environment.
- [x] Fix the flagged CPU `generate_response()` beam-vs-cache gap (and the identical gap in
  `generate_response_with_strategy()`'s unrecognized-strategy fallback). Done — see the
  September 14, 2026 (same day, later) update above.

Files to Modify:

- `src/KVCache.hpp` / `src/Decoder.{hpp,cpp}` (CPU) — done, no fix needed (root-cause investigation found no bug)
- `src/EncoderDecoderModel.{hpp,cpp}` — done: greedy-decoding cache-bypass workaround removed; `gpu_generate_response()`/`gpu_generate_response_with_strategy()` wired to the incremental GPU path with a beam-search guard; `generate_response()`'s own beam-vs-cache guard added, `generate_response_with_strategy()`'s unrecognized-strategy fallback guard widened (see the September 14, 2026, same day, later update above)
- `tests/inference_optimization_test.cpp` — done: real incremental-vs-full-recompute regression test added (CPU)
- `tests/encoderdecoder_test.cpp` — done: `GenerateResponseNotCorruptedByLeftoverBeamConfig` regression test, confirmed to reproduce the pre-fix segfault
- `src/BatchedInferenceEngine.hpp` — done: documented the same beam-vs-cache contract on `InferenceRequest::model_fn` (not currently reachable, no code change needed)
- `src/gpu/MatrixGPU.hpp` — done: new `GPUKVCache`/`GPUDecoderKVCache` types (backend-agnostic, one definition for both CUDA and SYCL)
- `src/MultiHeadAttention.{hpp,cpp}` / `src/CrossAttention.{hpp,cpp}` — done: `gpu_forward_with_cache()` on both
- `src/DecoderBlock.{hpp,cpp}` / `src/Decoder.{hpp,cpp}` (GPU) — done: `gpu_forward_with_cache()` / `gpu_decode_step()`
- `tests/` — not done: no test can exercise the new GPU code paths without real hardware (compile-verified only, under both `gpu` and `sycl` presets)

Context: added alongside `gpu_evaluate()` and `gpu_generate_response()` (July 2026), which GPU-accelerated validation loss and BLEU/ROUGE scoring — see git history around that change for the full-recompute implementation this replaces.

---

### TD-033: chatbot_api_server Inference Never Uses Persistent GPU-Resident Decode

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — wired in and verified September 13, 2026; before/after benchmark not performed (no GPU hardware in this environment) | GPU / Inference / Performance | September 7, 2026 | 6-8 hours |

Description:
Originally filed as "Matrix GPU Dispatch Doesn't Use Persistent GPUMatrix Residency," on the claim that nothing in the training/inference hot path calls `to_gpu()`/`from_gpu()` and that TD-003's persistent-residency win was never realized outside code that explicitly opts in. That claim is only half true, and the half that's false changes where the fix belongs — this entry replaces it with the verified scope.

**Training already has full persistent GPU residency.** `ChatbotTrainer::train_epoch()` calls `EncoderDecoderModel::gpu_forward()`/`gpu_backward()` (`src/ChatbotTrainer.cpp`) under `#ifdef ADAI_ENABLE_GPU`, which cascades through `EncoderBlock`/`DecoderBlock::gpu_forward()`/`gpu_backward()` (`src/EncoderBlock.cpp`, `src/DecoderBlock.cpp`) into `MultiHeadAttention`/`FeedForward`/`LayerNorm::gpu_forward()`/`gpu_backward()`, all chaining `GPUMatrix` end-to-end with cached device-resident buffers (`cached_Q`, `cached_K`, `cached_hidden`, etc.). This is exactly TD-003's intended design (resolved May 3, 2026), fully wired up, and identical on both backends — `GPUMatrix` exposes the same API in `src/gpu/MatrixGPU.hpp` (CUDA) and `src/gpu/sycl/MatrixGPU_SYCL.hpp` (SYCL).

**The real, live gap is inference serving.** `ChatbotAPI::generate_response()` (`src/ChatbotAPI.cpp`) builds its `model_fn` around `EncoderDecoderModel::forward()` — the plain CPU `Matrix` path — never the GPU-resident one. `Matrix::operator*()` (`src/Matrix.cpp`) auto-dispatches per call to `Matrix::multiply_gpu()` whenever `GPUManager::is_available()` and both inner dimensions are ≥32; `multiply_gpu()` (and `add_gpu`/`transpose_gpu`/`scale_gpu`/`hadamard_gpu`) allocate device memory, upload, compute, and download on every single call, with zero reuse of `GPUMatrix`. So on a GPU build, every live chat request re-pays a full alloc+upload+compute+download for every matmul, in every layer, for every generated token — worse than the training case, since it's per-token request latency rather than amortized training throughput. Confirmed backend-symmetric: `GPUMemory` does a fresh `cudaMalloc` (`src/gpu/GPUUtils.hpp`) or `sycl::malloc_device` (`src/gpu/sycl/GPUUtils_SYCL.hpp`) per call either way — this is a `Matrix.cpp`/`ChatbotAPI.cpp` dispatch problem, not a CUDA-vs-SYCL difference.

`EncoderDecoderModel::gpu_generate_response()` (`src/EncoderDecoderModel.cpp`) already exists as a persistent-residency decode path, but it's wired only into `ChatbotTrainer`'s internal generation-quality backfill / BLEU-ROUGE scoring (`src/ChatbotTrainer.cpp`), never into `ChatbotAPI` — so the one caller that would most benefit (real chat latency) has never been connected to it.

Related: TD-050 (GPU-Resident KV-Cache) covers a different problem inside `gpu_generate_response()` itself — once called, it recomputes the full sequence from scratch every step (no KV-cache), which is O(n²) instead of O(n). TD-050 is about making `gpu_generate_response()` fast once used; this item is about it not being used by `chatbot_api_server` at all.

Discovered during the per-file production-readiness rollout (September 7, 2026) — see [file-status-standard.md](file-status-standard.md); corrected same day after tracing both GPU backends end-to-end.

**Wired in September 13, 2026.** `ChatbotAPI::generate_response()`'s plain path now checks
`GPUManager::is_available()` and, when true, calls a new
`EncoderDecoderModel::gpu_generate_response_with_strategy()` instead of building the CPU
`model_fn`/`TextGenerator` path — added alongside the existing `gpu_generate_response()` (kept
unchanged for its original BLEU/ROUGE-scoring caller in `ChatbotTrainer.cpp`) rather than
extending it, since `ChatbotAPI` needs per-request `strategy`/`temperature`/`top_k`/`top_p`/
`num_beams`, which the original method never took as parameters (it relied on whatever config
`generator` already held).

Confirmed by reading `TextGenerator.cpp` directly (not merely assumed) that a single model_fn
returning only the last position's logits — the same shape `gpu_generate_response()`'s model_fn
already used — is the correct, sufficient contract for *every* strategy: `generate_greedy()`,
`generate_sampling()`, `generate_top_k()`, `generate_nucleus()`, and `generate_beam_search()`
(called once per beam per step, with that beam's own token sequence) all extract only
`logits.rows - 1`'s row from whatever model_fn returns. This means, unlike the CPU
`generate_response_with_strategy()`, the GPU-resident version needs no separate KV-cache-vs-
no-cache branch per strategy — `gpu_decode()` already recomputes the full sequence from scratch
every call regardless (no GPU KV-cache exists yet — TD-050), so one model_fn serves every
strategy uniformly, including beam search.

**Found and fixed a real concurrency gap while doing this wiring — not merely a hypothetical
one.** `model_mutex_` (added by TD-156 to serialize `forward()`/`backward()`/`generate_response()`/
`generate_response_with_strategy()` against `chatbot_api_server`'s real concurrent request
threads) was deliberately *not* held across `gpu_forward()`/`gpu_backward()`/
`gpu_generate_response()`, justified at the time by "not reachable from chatbot_api_server's live
serving path" — a justification this exact fix removes. Confirmed by reading
`MultiHeadAttention::gpu_forward()` (and the analogous `EncoderBlock`/`LayerNorm`/`FeedForward`
methods it calls into): `encoder->gpu_encode()`/`decoder->gpu_decode()`/`lm_head->gpu_forward()`
all write through to those sub-objects' own persistent per-instance `GPUState` (`cached_Q`,
`cached_K`, `cached_head_weights`, `cached_attn_out`) — exactly as shared and exactly as
unsynchronized as the four CPU-side `cached_*` members TD-156 already found racing. Two
concurrent chat requests hitting the newly-wired GPU-resident path would have reintroduced that
same class of heap-corruption bug. Fixed by holding `model_mutex_` for the full duration of both
`gpu_generate_response()` and the new `gpu_generate_response_with_strategy()` — the same fix
shape TD-156 used, applied before this bug could ever manifest for real rather than after.

Action Items:

- [x] Give `ChatbotAPI::generate_response()`'s `model_fn` a GPU-resident branch that calls a
  GPU-resident decode path when `GPUManager::is_available()`, instead of unconditionally calling
  `model_->forward()`.
- [x] Confirm `TextGenerator`'s beam/top-k/nucleus/temperature strategies all work against the
  GPU-resident decode path — confirmed by reading `TextGenerator.cpp` directly; see above.
- [ ] Benchmark chat response latency before/after on a representative prompt/`max_length` on
  both CUDA and SYCL builds to confirm the per-token round-trip elimination — **not performed**:
  this dev environment has the CUDA (`nvcc`) and Intel oneAPI (`icpx`) toolchains but no physical
  GPU device in either configuration (confirmed directly — see TD-041's writeup for the same
  finding), so `GPUManager::is_available()` never becomes true here and the new branch, while
  compiled and reachable, has never actually executed end-to-end on real hardware. Needs the
  user's own GPU-equipped machine.
- [x] Leave `Matrix::multiply_gpu()` and siblings as-is for callers that only have CPU `Matrix`
  data and no persistent-residency alternative — untouched; this item routed the identified hot
  path around them rather than removing them.

Files Modified:

- `src/ChatbotAPI.cpp` — `generate_response()`'s plain path, GPU-resident branch
- `src/EncoderDecoderModel.hpp` / `src/EncoderDecoderModel.cpp` — new
  `gpu_generate_response_with_strategy()`; `model_mutex_` now held across both GPU generation
  methods (concurrency fix, see above)
- `tests/chatbotapi_test.cpp` — two new tests (functional correctness across every strategy;
  concurrent-request safety mirroring TD-156's own regression test), both `GTEST_SKIP()`-guarded
  on `GPUManager::is_available()`/`initialize()` for the reason given in the benchmark item above
  — real, permanent coverage for whenever a GPU is available, not exercised by this session's own
  verification run

---

### TD-014: LLM Operations and Training Tooling Suite

| Priority | Status | Component | Created |
|----------|--------|-----------|---------|
| MEDIUM | Planned | Tooling / Toolchain | March 8, 2026 |

Description:
As the core machine learning models mature, standalone tools are missing for dataset lifecycle, evaluation, and deployment optimization. We lack isolated binaries to handle quantization, standardized inference evaluation, dataset PII scrubbing/dedup, and concurrent load testing.

Action Items:

- Implement `adai-weights-tool` for FP16/INT8 quantization and converting weight formats.
- Develop `adai-eval` for deterministic pipeline benchmarking on standardized Q&A lists.
- Isolate data-filtering scripts into an `adai-data-prep` tool for reproducible data hygiene.
- Add a dedicated target in `CMakeLists.txt` for these auxiliary tools to avoid bloating main executables.

### TD-006: Fill-in-the-Middle (FIM) Training Data Generation

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Planned | Training / Data Generation | February 17, 2026 | 6-8 hours |

Description:
Current training data for Project Gutenberg books uses consecutive sentence pairs (question → answer). Adding Fill-in-the-Middle (FIM) style training where the model predicts a middle sentence given first and last sentences would improve narrative understanding and coherence.

Current Behavior:

```cpp
// Simple consecutive pairs
INPUT: "What does this mean: [sentence 1]"
RESPONSE: "[sentence 2]"
```

Desired Behavior:

```cpp
// Add ~20% FIM-style pairs
INPUT: "Fill in the middle: <|first|>[sentence 1]<|last|>[sentence 3]"
RESPONSE: "[sentence 2]"
```

Benefits:

- Better understanding of narrative flow and context
- Improved coherence in generated responses
- Enhanced ability to reason about story structure
- More robust to partial context scenarios

Implementation Tasks:

- [ ] Add FIM data generation to `create_qa_pairs_from_text()`
- [ ] Define special tokens for FIM format (`<|first|>`, `<|middle|>`, `<|last|>`)
- [ ] Add configuration for FIM percentage (default: 20%)
- [ ] Ensure balanced distribution of regular and FIM pairs
- [ ] Add FIM-specific evaluation metrics
- [ ] Update documentation with FIM training approach
- [ ] Test on various text sources

Files to Modify:

- `src/DataFetcher.cpp` - Modify `create_qa_pairs_from_text()` (moved here from
  `IncrementalTrainer.cpp` in the TD-028 refactor; corrected September 7, 2026)
- `src/BPETokenizer.cpp` - Add FIM special tokens to vocabulary
- `src/DataFetcher.hpp` - Add FIM configuration options

Code Location:

`src/DataFetcher.cpp` — `create_qa_pairs_from_text()` (tagged with a `// TODO: TD-006` comment)

References:

- FIM approach used successfully in code completion models (Copilot, CodeGen)
- Paper: "Efficient Training of Language Models to Fill in the Middle" (Bavarian et al.)

Evaluation:

- Test on narrative coherence tasks
- Measure improvement in multi-turn conversation quality
- Compare perplexity on FIM vs standard test sets

---

### TD-037: No Qt Test Infrastructure for GUI Classes

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open (logic-extraction path adopted; widget-level testing still not attempted) | GUI / Testing | September 7, 2026 | 8-12 hours |

Description:
`ChatbotGUI.{cpp,hpp}` and `MnsManagerGUI.{cpp,hpp}` have no automated coverage, and this repo
has no QTest (or any Qt-aware test) infrastructure at all — every other test in `tests/` is a
plain GTest with no Qt event loop. Real widget behavior is hard to test without one; the more
tractable near-term step is separating non-widget logic (state transitions, signal/slot wiring
decisions, data formatting) out of the widget classes into plain C++ that GTest can already
exercise, deferring full widget testing until QTest is actually adopted.

**Update (September 13, 2026):** picked the logic-extraction path over adopting QTest — this
entry's own Description already called that out as the more tractable near-term step, so this
wasn't treated as an open question needing a separate decision. Also found and closed a real gap
along the way: `MnsJsonHelpers.hpp` (used by `MnsManagerGUI` for JSON parsing/formatting and URL
parsing) was *already* plain, Qt-free, GTest-exercisable code — including two real bug fixes in
its own history (TD-068, TD-102) — yet had zero tests of its own despite being tagged `stable`.

- Added `tests/mnsjsonhelpers_test.cpp` (37 tests, `MnsJsonHelpersTests` in `ctest`): full coverage
  of `find_string_end`, `json_value`, `json_array_objects` (including the TD-068 brace-desync
  regression), `json_escape`, `json_pretty` (including the TD-102 escaped-backslash regression),
  and `ParsedUrl::from`.
- Extracted two more pieces of real logic that were previously inline in `MnsManagerGUI`'s slots,
  into new functions in `MnsJsonHelpers.hpp`, each with its own tests: `parse_tags()` (the
  "Register" tab's comma-separated `key=value` tags field, extracted verbatim from
  `onRegisterModel()`) and `build_register_model_body()`/`build_set_candidate_body()` (the
  `POST /models` and `PUT /models/{name}/state` request bodies `onRegisterModel()`/
  `onSetCandidate()` used to build inline). Verified via revert-confirm-fail: temporarily broke
  the tags-joining comma logic and confirmed the new test caught it before restoring.
- Extracted `ChatbotGUI::onStrategyChanged()`'s combo-box-index-to-strategy-name switch statement
  into a new `src/ChatbotGuiLogic.hpp` (`generation_strategy_for_index()`), with its own
  `tests/chatbotguilogic_test.cpp` (2 tests, `ChatbotGuiLogicTests` in `ctest`) covering every
  combo box index plus the out-of-range fallback.
- Confirmed both `chatbot_gui`/`mns_manager_gui` binaries still build and link cleanly against the
  refactored call sites.
- `MnsJsonHelpers.hpp` bumped 1.0.1 → 1.1.0 (stays `stable`); `ChatbotGuiLogic.hpp` is a new file,
  tagged `beta` from the start (100% branch coverage). `ChatbotGUI.cpp`/`MnsManagerGUI.cpp` stay
  `beta` (still capped — the widgets themselves remain untested) but their status comments now
  note what's covered.

Action Items:

- [x] Decide whether to adopt QTest (`Qt::Test` component, `QTEST_MAIN`) as a second test
  framework alongside GTest, or to keep pushing logic out of the widget classes instead — the
  latter, per this entry's own Description.
- [x] Extract testable non-widget logic from `ChatbotGUI`/`MnsManagerGUI` into plain classes — done
  for the highest-value pieces (see the Update above): URL/JSON parsing, the tags mini-parser, two
  request-body builders, and the strategy-index mapping. `onSetTraining`/`onPromote`/`onRetire`/
  `onDelete`'s own request bodies are one-or-zero-field string literals with no real logic worth
  extracting on their own.
- [x] Add tests for the extracted logic — done (39 new tests total across two new test files).
- [ ] Full widget-level testing (button clicks, layout, signal/slot wiring under a real Qt event
  loop) remains untested and would need the QTest framework decision revisited — deliberately not
  pursued this pass, since the logic-extraction path already captured the highest-value, lowest-risk
  coverage available without it.

Files to Modify:

- `src/ChatbotGUI.cpp` / `src/ChatbotGUI.hpp` — done (see Update above); the widgets themselves
  (button clicks, layout) remain untested.
- `src/MnsManagerGUI.cpp` / `src/MnsManagerGUI.hpp` — done (see Update above); same widget-level
  caveat.
- `src/MnsJsonHelpers.hpp` — done (tested, gained `parse_tags`/`build_register_model_body`/
  `build_set_candidate_body`).
- `src/ChatbotGuiLogic.hpp` — new file, done.
- `tests/CMakeLists.txt` — done (`mnsjsonhelpersTests`, `chatbotguilogicTests` registered); no
  QTest integration needed, per the Decision above.

---

### TD-038: Advanced Features Tested in Isolation, Never Wired Into a Shipped Binary

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open (7/7 non-blocked items done — Quantization deliberately deferred) | Advanced Features / Integration | September 7, 2026 | 16-24 hours |

Description:
`BatchedInferenceEngine`, `IntegratedInferenceEngine`, `PipelineInferenceEngine`,
`SpeculativeDecoding`, `LoRA`, `Quantization`, `RewardModel`, and `PerformanceProfiler` each had
real, passing dedicated tests — the gap wasn't test coverage, it was that nothing in `chatbot`,
`chatbot_api_server`, or `incremental_trainer` actually called any of them. Each needed its own
integration decision (a CLI flag, a config option, a training-mode switch) rather than one shared
fix — grouped here because they shared the identical structural gap, not because one change
resolves all eight.

**Update (September 12, 2026):** all five non-blocked, non-deferred items are now wired into
`chatbot_api_server` — see the commits below. Two of the five (`PipelineInferenceEngine`,
`IntegratedInferenceEngine`) turned out to be genuinely broken, not just untested: each had a real
bug that had never been exercised against real model components, because their own dedicated unit
test suites used mock/null types shaped to match the (buggy) production code rather than the real
dependency — exactly the failure mode this TD's title describes, confirmed in the most direct way
possible. `LoRA`/`Quantization` were explicitly scoped out after a user decision at the time (both
would require touching `MultiHeadAttention`'s forward pass for any real integration — the same
foundational-class risk class as TD-059, which had not yet landed); `RewardModel` was genuinely
blocked on TD-034, resolved September 13, 2026. `LoRA` was revisited and wired in later the same
day, once TD-059's fix had landed and made touching the attention forward pass a well-understood,
already-tested change rather than an open risk — see the Update below. `Quantization` remains
deliberately deferred (see its own Action Item).

**Update (September 13, 2026):** `RewardModel`/`PPOOptimizer` wiring done — user chose the "full
RLHF fine-tuning loop" option (rollout generation, encoding bridge, PPOOptimizer, and a real
mechanism applying the PPO policy gradient to `EncoderDecoderModel`'s actual weights), not just
RewardModel-only scoring or deferring the item further. New `src/RLHFTrainer.{hpp,cpp}` is the
first real integration driving both classes against a live policy, closing two gaps neither had
anywhere else in the tree:
- **Encoding bridge**: `RewardModel` takes fixed-width vectors, not token sequences.
  `RLHFTrainer::encode_to_vector()` mean-pools `LLMEncoder::encode_with_mask()`'s per-token output
  (the same real encoder representations the policy already learns) rather than using
  `LLMEncoder::encode(std::string)`, which turned out to be unusable — its own `tokenizer` member
  is a fresh, never-vocab-loaded `BPETokenizer`, confirmed by reading the constructor directly.
- **Applying the policy gradient**: `PPOOptimizer::update()` computes a real clipped-ratio loss and
  trains its own internal `ValueFunction`, but — per its own doc comment, confirmed by reading its
  full body — it has no policy reference and never touches any model's weights.
  `RLHFTrainer::apply_policy_gradient()` builds the actual advantage-weighted gradient at the
  logits level (`advantage * (softmax(logits) - one_hot(action))` — the same shape
  `EncoderDecoderModel::compute_loss_gradient()` already uses for ordinary cross-entropy training,
  generalized by a per-position advantage weight; `advantage ≡ 1` reduces to that exact proven
  formula, used as the sign-convention check) and applies it via `backward_pass()` + a real
  `Optimizer::step()` — the same `zero_grad → forward → backward_pass → step` shape
  `ChatbotTrainer` already uses in production. Deliberately NOT
  `EncoderDecoderModel::update_weights()`, which a full read of its body showed never actually
  updates the encoder's weights at all (`// encoder->update_weights(learning_rate); // LLMEncoder
  doesn't have this method`) — only the external-optimizer path updates every component including
  the encoder.

Real, previously-undiscovered subtleties found and worked around along the way:
`EncoderBlock::forward()` caches its own activations unconditionally, with no `requires_grad` guard
at that level — any encoder call for reward-scoring or value-estimation between a policy
`forward()` and its matching `backward_pass()` would silently corrupt the gradient. Fixed by strict
ordering (all side encoder calls complete before the one authoritative policy `forward()`,
documented prominently in `RLHFTrainer.hpp`), not by a code change to `EncoderBlock` itself
(out of scope here). Also: `PPOOptimizer::compute_gae()`'s advantage normalization is per-trajectory
and, combined with an end-loaded reward and an uncalibrated, randomly-initialized `ValueFunction`,
made the sign of a rollout's own GAE advantage arbitrary on a cold start — not a bug, but it meant
the original correctness test (asserting a preferred response's log-prob rises after a full
`run_iteration()`) could fail for reasons unrelated to the gradient mechanism itself. Resolved by
exposing `apply_policy_gradient()` as its own public, directly-testable method taking
caller-supplied advantages, and testing the sign convention against it in isolation
(`PositiveAdvantageIncreasesLogProbNegativeAdvantageDecreasesIt`), independent of
`RewardModel`/`ValueFunction`/GAE's cold-start dynamics.

Scope limitation, disclosed in `RLHFTrainer.hpp`'s own doc comment: `run_iteration()` applies
exactly one gradient step per rollout, using the rollout-time policy's own log-probs as
`old_log_probs` — this makes PPO's clipped-ratio term inert on the very rollout it was computed
from (ratio ≡ 1), mathematically equivalent to REINFORCE with a learned (GAE) baseline. Real
intra-rollout PPO clipping (multiple gradient epochs per rollout, re-deriving log-probs between
them) is not implemented. Also out of scope: wiring `RLHFTrainer` into an actual CLI subcommand of
`incremental_trainer` or `chatbot_api_server` — the class is built, tested (4/4 tests passing,
including a full training-loop weight-change proof and the isolated gradient-sign proof), and
ready to be driven, but nothing yet calls it from a shipped binary's command-line surface. Flagged
here rather than assumed, since the user's "full RLHF fine-tuning loop" choice didn't explicitly
promise that CLI wiring.

**Update (September 13, 2026):** `LoRA` wiring done — revisited after being deferred alongside
`Quantization` in the first pass above, now that TD-059's fix made touching
`MultiHeadAttention`/`CrossAttention`'s forward pass a well-understood, already-tested change
rather than an open risk. `LoRAAdapter` (`src/LoRA.hpp`) is now genuinely attached to every
self-/cross-attention layer's Q/K/V/O projections, in both the encoder and decoder:
- **Wiring shape**: `MultiHeadAttention`/`CrossAttention` each gained `enable_lora(LoRAConfig)`,
  `has_lora()`, `register_lora_parameters(Optimizer&)`, and `merge_lora()`, following
  `LoRAAdapter::forward(x, W_output)`'s own existing "add my delta to an already-computed base
  output" contract — `forward_parallel()`/`forward_with_cache()` apply each active adapter right
  after its corresponding base projection, so the per-head attention math itself needed no changes
  at all, only the four projection call sites and the matching four spots in `backward()`.
  `EncoderDecoderModel` cascades the same three calls across every attention layer via the
  existing `LLMEncoder`/`LLMDecoder` per-block accessors (`get_encoder_block(i)`/
  `get_decoder_block(i)`), so a caller enables/trains/merges LoRA for the whole model in one call
  each, the same shape as `register_parameters(Optimizer&)`'s existing full-fine-tune cascade.
  `register_lora_parameters()` registers ONLY the adapters' own A/B matrices — never the base
  W_q/W_k/W_v/W_o — so a base model stays genuinely frozen whenever only that method (not also
  `register_parameters()`) is called against a training optimizer.
- **A real, previously-undiscovered bug found and fixed along the way**: `LoRAAdapter::backward()`
  used to return `void` and never exposed the LoRA branch's own gradient w.r.t. its input `x` at
  all — harmless for a standalone adapter with no caller before it, but a real, silent
  gradient-flow gap for exactly how this class is now used (`MultiHeadAttention`/`CrossAttention`
  have layers before them across a multi-block encoder/decoder stack; without this, LoRA on any
  layer but the very first would have silently truncated backpropagation to everything upstream of
  it, undetectable by that adapter's own existing unit tests since none had a caller feeding it a
  further input). Fixed by deriving and returning the correct contribution
  (`scale * (grad_output * B) * A`) and wiring it into both attention classes' `backward()`, then
  verifying with finite-difference gradient checks against the model's actual forward pass with
  LoRA active (`MultiHeadAttentionLoRATest.BackwardPassMatchesNumericalGradientWithLoraActive`,
  and `CrossAttention`'s equivalent) — not just checked in isolation.
- **A second real, previously-undiscovered bug**: `LoRAAdapter::merge_with_base()` computed
  `ΔW = B*A` and validated its `W` argument against an (output_dim, input_dim) shape — the
  column-vector "y = W*x" convention. `forward()` uses the opposite row-vector "y = x*W"
  convention throughout (`ΔW_effective = A^T*B^T`, matching how every real caller in this codebase,
  including this class's own new integration, stores its weight matrices). A square adapter (the
  only shape this class's own pre-existing tests ever exercised) couldn't catch this — both
  conventions pass the same shape check, and `(B*A)` happens to have the same shape as
  `(A^T*B^T)` when square, just different (generically wrong) numbers. Caught only once
  `merge_with_base()`'s actual output was compared against `forward()`'s own output for the same
  weights on a deliberately non-square adapter (`LoRATest.MergeMatchesForwardOutput`), rather than
  just its shape as the original test did.
- **Scope**: FeedForward layers are NOT touched — `LoRAConfig::apply_to_ffn` (already `false` by
  default) is not implemented by any class in this pass; only the Q/K/V/O attention projections
  `apply_to_query`/`key`/`value`/`output` gate. LoRA is applied only on the CPU forward/backward
  path (`forward()`/`forward_parallel()`/`forward_with_cache()`/`backward()`) — the separate
  persistent-GPU-residency decode path (`gpu_forward()`/`gpu_backward()`, TD-033's route) does not
  apply adapters at all; enabling LoRA while a caller uses that GPU path silently runs the
  unmodified base weights, not currently guarded against. Also out of scope, same as RewardModel
  above: wiring this into an actual CLI subcommand of `incremental_trainer`/`chatbot_api_server`
  (e.g. a `--lora-rank`/`--lora-checkpoint` flag) — the mechanism is built, tested (13 new tests:
  2 unit-level in `phase5_test.cpp` covering the two bugs above, 4 each in
  `multiheadattention_test.cpp`/`crossattention_test.cpp` covering the real wiring (no-op at
  init, finite-difference gradient check with LoRA active, base-weight freezing, merge
  equivalence), 3 model-wide in `encoderdecoder_test.cpp` covering the same properties cascaded
  across every encoder/decoder attention layer), and ready to be driven, but nothing yet calls it
  from a shipped binary's command-line surface.

Action Items:

- [x] `BatchedInferenceEngine`: wired into `ChatbotAPI::enable_batched_inference()` /
  `chatbot_api_server --batched-inference`. Found and fixed a real bug along the way:
  `process_batch()`'s per-response token-count stat called `tokenizer_->encode()` unguarded on a
  generated response that can legitimately be empty, and one request's empty response would
  spuriously fail every other request batched alongside it.
- [x] `PipelineInferenceEngine`: wired into `ChatbotAPI::enable_pipeline_inference()` /
  `chatbot_api_server --pipeline-inference`. Found and fixed a real bug: `decoder_worker()` called
  a method, `forward_with_cross_attention(tokens, encoder_output, nullptr)`, that does not exist
  on the real `LLMDecoder` — its own test suite and `PipelineBenchmark.cpp` both instantiated the
  class exclusively against mocks shaped to match that wrong call, silently masking the mismatch.
- [x] `IntegratedInferenceEngine`: wired into `ChatbotAPI::enable_integrated_inference()` /
  `chatbot_api_server --integrated-inference`. Found and fixed two real bugs, worse than
  `BatchedInferenceEngine`'s: `decoder_worker()` and `batcher_worker()` each called
  `tokenizer_->encode()` unguarded with no try/catch anywhere in either function, so an empty
  generated response or empty input text would escape the worker thread and call
  `std::terminate()` — crashing the entire `chatbot_api_server` process, not just failing one
  request. Confirmed via genuine reproduction (real, non-mock model components — this class isn't
  templated) before and after the fix.
- [x] `SpeculativeDecoding`: `ChatbotAPI::draft_model_` + `--draft-model`/
  `--speculative-candidates` CLI flags / `DRAFT_MODEL_PATH`/`SPECULATIVE_NUM_CANDIDATES` config
  keys.
- [x] `PerformanceProfiler`: `ChatbotAPI::enable_profiling()` / `chatbot_api_server --profile` →
  `GET /admin/profile`. Found and fixed a real, session-introduced thread-safety bug along the
  way: `Profiler`'s internal maps had no locking, and `active_timers` was keyed by section name
  alone, so concurrent same-name `start()`/`stop()` pairs (exactly `generate_response()`'s case
  under `chatbot_api_server`'s real thread pool) corrupted each other's recorded timings.
- [ ] `Quantization`: still explicitly deferred — needs a scoping decision: a standalone
  checkpoint-manipulation CLI tool (lower risk) vs. actually modifying `MultiHeadAttention`'s
  forward pass for real inference/training integration (the real thing, higher risk). Unlike
  `LoRA` below, no user decision to revisit this has been made yet.
- [x] `LoRA`: `MultiHeadAttention`/`CrossAttention` gained `enable_lora()`/`has_lora()`/
  `register_lora_parameters()`/`merge_lora()`, applied to every Q/K/V/O projection per
  `LoRAConfig`'s `apply_to_*` flags; `EncoderDecoderModel` cascades all four across every encoder/
  decoder attention layer. Found and fixed two real, previously-undiscovered bugs in
  `LoRAAdapter` itself along the way (`backward()` never returning its branch's own input
  gradient; `merge_with_base()` using the wrong matrix-orientation convention) — see the Update
  above for both. Not yet wired into any CLI subcommand of a shipped binary — see the
  scope-limitation note above.
- [x] `RewardModel`/`PPOOptimizer`: new `src/RLHFTrainer.{hpp,cpp}` drives both against a live
  `EncoderDecoderModel` policy — real rollout generation, the encoding bridge, and a real policy
  gradient applied to the model's actual weights (see the Update above for the full mechanism and
  the two gaps closed). Not yet wired into any CLI subcommand of a shipped binary — see the
  scope-limitation note above.
- [x] Add an integration test per wired feature proving the wiring works end-to-end — done for all
  five items above (ChatbotAPI-level integration tests plus live end-to-end verification against a
  real running `chatbot_api_server` process for each), for `RewardModel`/`PPOOptimizer`
  (`tests/rlhftrainer_test.cpp`, 4/4 passing), and for `LoRA` (13 new tests across four files, all
  passing — see the Update above).

Files to Modify:

- `src/Quantization.hpp` — remaining, unattempted work.
- `src/LoRA.hpp` (fixed `backward()`/`merge_with_base()`, added `register_parameters()`),
  `src/MultiHeadAttention.{hpp,cpp}`, `src/CrossAttention.{hpp,cpp}`,
  `src/EncoderDecoderModel.{hpp,cpp}` (LoRA cascade), `tests/phase5_test.cpp`,
  `tests/multiheadattention_test.cpp`, `tests/crossattention_test.cpp`,
  `tests/encoderdecoder_test.cpp` — done.
- `src/RLHFTrainer.{hpp,cpp}` (new), `tests/rlhftrainer_test.cpp` (new), `src/PPOOptimizer.hpp`
  (new public `compute_advantages()`), `src/RewardModel.hpp` (status tag only) — done.
- Already done (prior items): `src/BatchedInferenceEngine.hpp`, `src/PipelineInferenceEngine.hpp`,
  `src/IntegratedInferenceEngine.hpp`, `src/SpeculativeDecoding.hpp`, `src/PerformanceProfiler.hpp`,
  `src/ChatbotAPI.{hpp,cpp}`, `src/ChatbotApiServerArgs.{hpp,cpp}`, `src/ChatbotAPIServer.cpp`,
  `src/Config.{hpp,cpp}`.

---

### TD-039: Core Training/Metrics Classes Too Large and Fast-Moving to Certify Stable

| Priority | Status | Component | Created |
|----------|--------|-----------|---------|
| MEDIUM | Open | Training / Metrics / Core | September 7, 2026 |

Description:
`ChatbotTrainer`, `IncrementalTrainer`, `TrainingMetricsService`, and `TrainingMetricsAPI` are the
largest files in the tree (2000-2700 lines each) and already have substantial test coverage
(14/19/7/5 test-file cross-references respectively) — this isn't a test-coverage gap like the
other items here. It's churn: all four gained real features during the same session this tracker
was built in (the trainer admin API, session registry work), and a file that's still actively
changing shape isn't a good candidate for a `stable` claim regardless of how well-tested its
current snapshot is. No effort estimate — this isn't a fixed-scope task, it resolves when the API
surface stops changing.

Action Items:

- [ ] Let the current round of feature work (trainer admin API, session-scoped metrics) land and
  settle — no more structural changes planned.
- [ ] Do one deliberate final correctness/API-freeze review pass per file.
- [ ] Only then bump each to `MAJOR >= 1` and `@adai-status: stable`.

Files to Modify:

- `src/ChatbotTrainer.cpp` / `src/ChatbotTrainer.hpp`
- `src/IncrementalTrainer.cpp` / `src/IncrementalTrainer.hpp`
- `src/TrainingMetricsService.cpp` / `src/TrainingMetricsService.hpp`
- `src/TrainingMetricsAPI.cpp` / `src/TrainingMetricsAPI.hpp`

---

### TD-047: Android Data/Repository/API Layer Has No CI or Release History

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open (CI added; release process defined, first real release still pending) | Android / CI | September 7, 2026 | 10-14 hours |

Description:
41 files across `android/app` (15) and `android/opsdashboard`/`android/wearsync` (26) — the
repository/network/DTO/poller/contract layer — each have real dedicated unit tests and pass, but
the entire Android surface landed in a single commit, both apps still declare
`versionName = "0.1.0"`, and no CI workflow builds or tests either app. Being tested in isolation
isn't the same as having a release process; that's the actual gap here, distinct from TD-048
(files with no test at all).

**Update (September 12, 2026):** `.github/workflows/android-ci.yml` added — builds `:app`/
`:opsdashboard` (`assembleDebug`) and runs `testDebugUnitTest` across all modules on every
`android/**` push/PR. Verified genuinely green on a real GitHub Actions run (kerfbit mirror), not
just locally — the first version failed on its actual first run despite passing every local
check, because `wearface`'s `resignWffApk` task needs `~/.android/debug.keystore` to already
exist and a dev machine has one from prior Android Studio use, but a fresh CI runner doesn't;
fixed by generating the same well-known debug keystore as an explicit CI step. **The primary
`origin` (rjv717/adai) repo's own runs have not actually executed yet** — GitHub reports "recent
account payments have failed or your spending limit needs to be increased" on that account,
unrelated to this workflow; needs that billing issue resolved before this CI job is actually
gating anything on the repo's primary remote (it is confirmed working on the kerfbit mirror in
the meantime).

Setting up this CI job also surfaced a real, already-shipped regression it now guards against:
`opsdashboard`'s test suite didn't even compile on `main` (`ModelRepositoryTest.kt` was calling
`clearStaleTrainingLock` with the pre-TD-147 one-argument signature) — fixed alongside the CI
work; see that commit for the full story.

**Update (September 12, 2026, later same day):** [`android-release.md`](android-release.md) (the
versioning/release convention doc) and [`android-release.yml`](../../../.github/workflows/android-release.yml)
(the tag-triggered workflow enforcing it) added together — `app-vX.Y.Z`/`opsdashboard-vX.Y.Z` tags
now validate `versionName`/`versionCode` against the tagged commit, re-run the full test suite, and
build+publish both a debug-signed and an unsigned release APK to a GitHub Release, failing loudly
before publishing anything if any check doesn't hold. Neither app has actually cut a real release
with it yet — both are still at `0.1.0`/`versionCode 1` — but every path through the mechanism was
verified against real GitHub Actions runs (on the kerfbit mirror, to avoid touching the primary
`origin` repo's release history with test artifacts): a malformed tag (`app-v1.0.0-beta`) rejected
before any other step ran, a version-mismatched tag (`app-v9.9.9` against the real `0.1.0` code)
rejected with the expected error, a stale-`versionCode` tag (a throwaway commit bumping only
`versionName`) rejected once a prior release existed to compare against, and a correct
`app-v0.1.0` tag matching the code exactly running the full happy path — real `assembleRelease`/
`assembleDebug` builds and a real GitHub Release with both APKs attached. All test tags, the
throwaway commit/branch, and the test release were deleted afterward; nothing from this
verification was left in place. Real production signing (a keystore + GitHub secrets) is
explicitly out of scope — see the doc's own "Signing status" section for why that's a deliberate,
one-way decision left to a human, not something to generate as a side effect of automating this.

Action Items:

- [x] Add a GitHub Actions workflow building both `:app` and `:opsdashboard` and running their
  unit tests (`./gradlew testDebugUnitTest`).
- [ ] Resolve the `origin` repo's GitHub billing/spending-limit issue so `android-ci.yml` actually
  runs there (confirmed working on the kerfbit mirror already).
- [x] Establish a real release/versioning process before either app leaves `0.1.0` — defined in
  `android-release.md` and mechanically enforced by `android-release.yml`.
- [ ] Actually cut the first real release of each app under this process (both are still at
  `0.1.0`/`versionCode 1`).
- [x] Decide on real release signing — deliberately deferred (September 12, 2026): both apps are
  currently side-loaded on the maintainer's own devices, not distributed further, so the
  debug-signed APK already produced is sufficient. `android-release.md`'s "Signing status"
  section records the concrete plan for if/when that changes (self-managed keystore vs. Play App
  Signing) so this doesn't need to be re-decided later.

Files to Modify:

- 41 files under `android/app/src/main` and `android/opsdashboard/src/main` /
  `android/wearsync/src/main` tagged `beta` — see [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md)
  for the exact list.
- `.github/workflows/android-ci.yml` — done.
- `docs/development/guides/android-release.md`, `.github/workflows/android-release.yml` — done.

---

### TD-048: Android UI/DI/Entry-Point Classes Are Untested and Unreleased

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open (12/12 ViewModels done; Compose UI testing adopted, 12/12 screens covered; admin-dialog confirm-click flows and DI containers/entry points now covered too; `wearcomplications` services and full real-device verification remain) | Android / Testing | September 7, 2026 | 24-32 hours |

Description:
62 files — Compose screens, ViewModels without tests, DI containers, `Activity`/`Application`
entry points, and the `wearcomplications` services — genuinely have zero automated coverage, on
top of sharing TD-047's no-CI/no-release problem. `BiometricAdminAuthGate.kt` is a partial
exception: hard to unit-test (`BiometricPrompt` needs a real `FragmentActivity`), but it was read
manually during the rollout and appears complete — the gap there is coverage, not a known defect.

**Update (September 12, 2026):** all 8 ViewModels now have real test coverage —
`ChatViewModel`, `GroupListViewModel`, `SessionListViewModel`, `ModelListViewModel` (the four
originally tagged `experimental` here), plus `ModelDetailViewModel`, `AdminViewModel`,
`SessionDetailViewModel`, and `TrainerViewModel`. **The latter four turned out to be mistagged**
— each was already `beta`/"capped by TD-047" despite having zero dedicated tests (TD-047's own
scope is explicitly the repository/DTO layer, never ViewModels, and `beta` requires real test
coverage per the file-status standard). `AdminUiState.kt` was promoted alongside `AdminViewModel`
since its own documented per-daemon error-independence invariant is what got tested. Every test
was verified via the standard revert-confirm-fail cycle (a real behavior removed from the
ViewModel, the corresponding test confirmed to fail, then restored) — several of these are
regression tests for real historical bugs that had never had ViewModel-level coverage before
(e.g. `ModelDetailViewModelTest`'s `clearStaleTrainingLock` test is the first regression test for
TD-147's own fix). `AdminAuthGate.kt` was checked and confirmed to be correctly tagged as-is — a
pure interface/sealed-type declaration with no behavior of its own to test, unlike the four
mistagged ViewModels.

**Update (September 13, 2026):** Compose UI testing adopted — `ConversationListScreen` (`app`
module) is the first screen with real coverage, establishing the pattern for the remaining 11.
This repo had no `androidx.compose.ui.test` infrastructure at all, and the existing plain-JVM
`test`-source-set fakes (`FakeConversationDao`, `FakeApiClientProvider`, `FakeSettingsRepository`
— used by `ChatViewModelTest`/`ChatRepositoryTest`) live in a source set instrumented
(`androidTest`) tests can't see; rather than duplicate them, `android/app/src/sharedTest/java` was
added and wired into both the `test` and `androidTest` source sets (`build.gradle.kts`), and the
three fake files moved there — they depend only on plain Kotlin/coroutines and this module's own
domain interfaces, nothing Android-framework-specific, so nothing about moving them changes what
either source set can compile or run. `androidx.compose.ui:ui-test-junit4` (`androidTestImplementation`)
and `ui-test-manifest` (`debugImplementation`) added, both version-pinned via the existing
`compose-bom` platform already in use, no new version to track. `ConversationRow`'s delete
`IconButton` gained a `Modifier.testTag("delete_conversation_$id")` — every row otherwise shares
one `contentDescription` ("Delete conversation"), which a UI test can't disambiguate between rows
without depending on list order. 5 tests added
(`ConversationListScreenTest`): empty state, populated state (titles shown, placeholder hidden),
row click → `onOpenConversation(id)`, settings icon click → `onOpenSettings()`, delete click →
conversation removed from the rendered list, FAB click → new conversation created and opened.

**Update (September 13, 2026, continued):** `ChatScreen` (`app` module) done as the second screen
— along with it, `ChatInputBar`, `MessageBubble`, and `ErrorBanner` (three separate files it
composes) each got their own specific behavior genuinely exercised and were promoted
`experimental` → `beta` alongside it: `ChatInputBar`'s text-entry/send wiring, `MessageBubble`'s
FAILED-state rendering and retry click, and `ErrorBanner`'s message rendering and dismiss-action
callback. `TypingIndicator` (also composed by `ChatScreen`) was deliberately left `experimental`
— none of the 4 new tests ever observe `isSending == true` for long enough to assert on it,
since the fake chat service resolves synchronously with no artificial delay, so nothing here
actually exercises its own rendering. Along the way, found and fixed a real inaccuracy in
`ConversationListScreenTest`'s own doc comment from earlier the same day: it claimed
`FakeSettingsRepository()`'s default made `ConversationRepository.deleteConversation()` skip
`ApiClientProvider` entirely, but that fake's actual default constructor value is a *configured*
`ServerSettings(host = "localhost", ...)` (needed elsewhere for `ChatViewModelTest`'s real
fake-network path) — meaning the delete test had silently been attempting a real (harmlessly
failing, try/caught) network call to `localhost:8080` the whole time. Fixed by passing a blank
`ServerSettings()` explicitly, matching what the comment always claimed was happening.
4 tests added (`ChatScreenTest`): send a message → user bubble then assistant reply bubble
appear; back button → `onBack()`; a failed send → `ErrorBanner` shows the server's error message,
Dismiss hides it; retrying a pre-seeded `FAILED` message → resends and shows a new assistant
reply, `"Failed — tap to retry"` no longer rendered.

**Update (September 13, 2026, continued further):** `SettingsScreen` (`app` module) done as the
third screen, promoted `experimental` → `beta` alongside it. 5 tests added (`SettingsScreenTest`):
back button → `onBack()` without persisting anything; entering a host and clicking Save →
`SettingsViewModel.save()` is reached with the typed value and `onBack()` eventually fires;
toggling the HTTPS switch swaps the Port field for the two Cloudflare Access Client fields; a
blank-host "Test Connection" click shows the validation error without ever reaching the fake
server; a successful "Test Connection" shows the real "Connected — N active session(s)" message
from the fake `health()` response. Two real findings along the way:
- A genuine test-writing mistake, caught by its own revert-confirm-fail rather than shipped
  silently: the HTTPS-toggle test first tried clicking the `Switch`'s adjacent label `Text`, which
  has no click handler of its own (`Switch` and its label are separate composables in
  `SettingsScreen.kt`, not wrapped in one shared `clickable`) — the assertion failed for the right
  reason (the switch never toggled) but the wrong cause (a test bug, not a production one). Fixed
  by targeting `isToggleable()` instead of the label text.
- The Save-button test was originally written to assert TD-130's happens-before ordering (`save()`
  landing before `onBack()` fires) the same way `ChatScreenTest`'s tests assert ordering-sensitive
  outcomes — but revert-confirm-fail disproved that this actually works here: temporarily restoring
  TD-130's original fire-and-forget shape in `SettingsScreen.kt` did *not* make the assertion fail,
  because `performClick()` waits for the real device's Main dispatcher to reach idle before
  returning, and by then every scheduled coroutine on it — sequenced or not — has typically already
  run. A real-device instrumented test genuinely cannot distinguish "sequenced" from
  "fire-and-forget-but-fast" the way `SettingsViewModelTest`'s own `StandardTestDispatcher`
  -controlled plain-JVM test already does; only that finer-grained test actually guards TD-130's
  ordering invariant. Reworded the UI test's assertion and doc comment to claim only what it can
  actually verify (the wiring reaches `save()` with the right value, and `onBack()` eventually
  fires) — confirmed to still fail for a real reason via a second revert-confirm-fail pass
  (removing the `save()` call entirely).

**Update (September 13, 2026, continued still further):** `GroupListScreen` (`opsdashboard`
module) done as the fourth screen — the first in `opsdashboard`, which needed its own
`androidx.compose.ui.test`/`ui-test-manifest` dependencies and `src/sharedTest` source set wired
up first (identical setup to `:app`'s own, done earlier the same day); its 6 fake test-utility
files moved there and newly tagged (`beta`, not previously in-scope for the file-status standard
under `src/test`). 5 tests added (`GroupListScreenTest`): no configured groups shows the
placeholder message; multiple configured groups each show their own row with the right pending
count; one group's fetch failing shows its own error without hiding or corrupting the others'
results (mirrors `GroupListViewModelTest`'s own TD-128-style isolation test at the UI level);
clicking a row invokes `onOpenGroup(name)`; clicking the settings icon invokes `onOpenSettings()`.

**Known limitation, disclosed rather than worked around:** `opsdashboard`'s debug APK could not
be installed on this sandbox's phone emulator to run these tests live —
`INSTALL_FAILED_MISSING_SHARED_LIBRARY`, because its merged manifest declares
`<uses-library android:name="wear-sdk" android:required="true"/>` (auto-injected by its Wear
watch-face-push dependencies), and no Wear-capable emulator/device was available here even with
the `google_apis_playstore` system image. Per user decision, proceeded compile-verified only:
`GroupListScreenTest` compiles clean and was cross-checked carefully against the real production
code path by hand (`SafeCall.kt`/`ApiResult.kt`'s exact error-message strings, confirmed to
produce `"Error: connection refused"` verbatim for an `IOException("connection refused")`;
`ConversationListScreenTest`'s already-device-verified `ListItem` + `Modifier.clickable` and
`IconButton` + `contentDescription` interaction patterns, structurally identical here) rather than
run end-to-end. `GroupListScreen.kt` promoted `experimental` → `beta` with an explicit
real-device-still-unverified caveat in its own status comment, matching `GPUUtils.hpp`'s TD-041
precedent for hardware-gated code in this same tracker. Whoever next has access to a Wear-capable
emulator or real device should run `opsdashboard`'s `connectedDebugAndroidTest` for real at least
once to close this gap — nothing here is expected to fail, but it hasn't actually been observed
passing on a device the way the three `app`-module screens have.

**Update (September 13, 2026, yet further still):** `ModelListScreen` (`opsdashboard` module) done
as the fifth screen, same compile-verified-only caveat as `GroupListScreen` above (this sandbox
still can't install/run `opsdashboard` on a device). 5 tests added (`ModelListScreenTest`): no
models registered shows the placeholder message; multiple models each show their own row with
role (including the blank-role → `"(none)"` fallback) and state; a fetch failing with no prior
models shows `FullScreenError`; clicking a row invokes `onOpenModel(name)`; clicking the settings
icon invokes `onOpenSettings()`. Deliberately did not attempt `ModelListViewModelTest`'s own
"a failed *second* poll keeps the first poll's models" scenario at the UI level — that test needs
virtual time control (`advanceTimeBy` past the 8s poll interval) a real-device instrumented test
can't do without an actual ~9s real-time wait, and the existing ViewModel-level test already
covers it with precise control; duplicating it here for a screen-level test would cost real
wall-clock time in the suite for coverage that already exists. `ModelListScreen.kt` promoted
`experimental` → `beta` with the same real-device-still-unverified caveat.

**Update (September 13, 2026, and further still):** `SessionListScreen` (`opsdashboard` module)
done as the sixth screen, same compile-verified-only caveat. 7 tests added
(`SessionListScreenTest`): no sessions shows the placeholder message; idle-only sessions (hidden
by default) show the correct hidden-count message; clicking "Show idle" reveals the idle rows and
flips the button label to "Hide idle"; a mix of training/idle sessions shows only the training row
by default, with the exact `"Epoch N/M · loss X.XXXX"` formatting and `StatusBadge` label; a fetch
failing with no prior sessions shows `FullScreenError`; clicking a row invokes
`onOpenSession(key)`; clicking the settings icon invokes `onOpenSettings()`. The idle-session
hide/show filtering (`visibleSessions`/`hiddenIdleCount`) lives entirely in the composable, not
the ViewModel — only the `showIdle` flag flip itself is covered by `SessionListViewModelTest` —
so this is genuinely new coverage, not a duplicate of existing tests. `SessionListScreen.kt`
promoted `experimental` → `beta` with the same real-device-still-unverified caveat.

**Update (September 13, 2026, continuing on):** `ModelDetailScreen` (`opsdashboard` module) done
as the seventh screen. 9 tests added (`ModelDetailScreenTest`): all detail fields render correctly
(architecture, artifact, identity fields); blank optional fields (role, run ID, artifact
host/path) fall back to their documented placeholder text (`"(none)"`/`"(local)"`); each training
history entry renders with the right formatting; the "Training History" section is hidden
entirely when there's none; the three admin-action buttons' enabled/disabled state correctly
follows `model.state` (training → only "Clear lock" enabled; candidate → "Retire"/"Promote"
enabled; production → all three disabled); a fetch failing with no prior model shows
`FullScreenError`; the back button invokes `onBack()`.

**Deliberately out of scope for this pass:** the admin-action confirm-dialog flow itself (clicking
"Clear stale training lock"/"Retire candidate"/"Promote to production" and confirming) is not
tested. `ConfirmActionDialog` unconditionally does `LocalContext.current as FragmentActivity` and
reads `LocalAdminAuthGate.current` (a `staticCompositionLocalOf` that throws if never provided)
the moment it composes — true the instant the dialog opens, before its own confirm button is ever
clicked. A plain `createComposeRule()`'s default test host activity is a bare `ComponentActivity`,
not a `FragmentActivity` (only `MainActivity` itself is), so that cast would crash immediately;
there is also no fake `AdminAuthGate` implementation anywhere in this codebase yet. Testing this
flow for real needs a small, genuinely new piece of test infrastructure — a minimal
`androidTest`-only `FragmentActivity` host used via `createAndroidComposeRule<...>()`, plus a
`FakeAdminAuthGate` — that wasn't built speculatively here since this sandbox can't run any of it
live anyway to confirm it's wired correctly (see the `wear-sdk` limitation below). Flagged as a
concrete follow-up rather than left implicit: whoever has real device access and picks this up
should build that infrastructure once (it will be needed again for `AdminScreen` and
`GroupDetailScreen`'s own admin actions) rather than per-screen. `ModelDetailScreen.kt` promoted
`experimental` → `beta` anyway — the read-only majority of the screen has real coverage, and
`beta`'s own definition ("known gaps exist") fits a disclosed, scoped gap like this one — with
both this gap and the real-device-still-unverified caveat noted in its own status comment.

**Update (September 13, 2026, continuing on still further):** `SessionDetailScreen` (`opsdashboard`
module) done as the eighth screen. 8 tests added (`SessionDetailScreenTest`): a live session renders
its poller-status label, progress fields, and training metrics; a stale/not-effectively-training
session shows the "no — stale" reason; generation-quality metrics show their disabled message when
`current_bleu4 < 0` and show BLEU/ROUGE values otherwise; a fetch failing with no prior status shows
`FullScreenError`; a 404 on the very first poll correctly invokes `onEvicted()`; the "End session"
admin button renders enabled by default; the back button invokes `onBack()`. Notable finding:
`SessionDetailViewModel` uses `AdaptivePoller` (not the simpler `FixedIntervalPoller` the list
screens use), but `AdaptivePoller.run()` also calls `poll()` immediately before any delay, and a 404
response drives it straight to `PollerPhase.EVICTED`, which short-circuits the loop before it ever
reaches its `delay()` call — so the eviction scenario is deterministically observable via
`waitUntil()` on a real device just like the list screens' first-poll scenarios, with no virtual-time
control needed. Deliberately out of scope for this pass, for the same reason and with the same
infrastructure gap as `ModelDetailScreen` above: the "End session" confirm-dialog flow itself
(`ConfirmActionDialog`'s `FragmentActivity`/`LocalAdminAuthGate` requirement) — only the button's
default-enabled rendering is checked, not the click. `SessionDetailScreen.kt` promoted
`experimental` → `beta`, same rationale as `ModelDetailScreen`.

**Update (September 13, 2026, still continuing on further):** `SettingsScreen` (`opsdashboard`
module) done as the ninth screen. Unlike every screen so far, there was no pre-existing
`SettingsViewModelTest` to reuse a Fixture shape from (opsdashboard's `SettingsViewModel` had zero
coverage at any level) — `SettingsScreenTest`'s own Fixture was built directly against
`SettingsViewModel`'s real constructor (`FakeSettingsRepository` + a real `WatchFacePushRepository`).
9 tests added: the default shared-host layout (per-service Host fields hidden, Port fields shown);
toggling "one host for all services" swaps that; toggling the HTTPS-relay switch shows the
Cloudflare/Trainer Access Client fields and hides all Port fields; adding and removing a registry
group chip; selecting a poll-interval chip and Save persisting it and firing `onBack()`; toggling
watch-sync hides/shows the session-key-override field; Back without Save persists nothing; the
watch-face section's "Install / update" button renders enabled by default. Three `Switch`
composables share this screen (unlike `SettingsScreen` in the `app` module, which has only one), so
each gained its own `testTag` (`switch_use_shared_host`/`switch_use_https_relay`/
`switch_watch_sync_enabled`) — a no-op production change, same precedent as `ConversationListScreen`'s
delete-button `testTag`. `WatchFacePushRepository` is a concrete class wrapping
`androidx.wear.watchfacepush` (a real Wear system service), not an interface — no fake exists for it,
and building one would mean reshaping production code for a screen-test task alone, which this pass
doesn't do; it's constructed here with the instrumentation's real target `Context` (safe — the
constructor only stores the `Context`, touching no Wear API until `pushWatchFace()`/`isSupported()`
are actually called). Deliberately out of scope for this pass: clicking "Install / update" or
"Activate" — both call into `WatchFacePushManagerFactory`, a real AndroidX Wear library this sandbox
has no paired-watch/Wear-capable emulator to exercise (the same `wear-sdk` limitation already
blocking a live device run of this whole module). Only the button's default rendering is checked.
`SettingsScreen.kt` promoted `experimental` → `beta` (0.3.0) — the gap is disclosed in its own status
comment and here.

**Correction (September 13, 2026):** this entry's "8/8 ViewModels done" claim (Action Items below)
was stale/wrong — enumerating every `*ViewModel.kt` under both modules' `src/main` against every
`*ViewModelTest.kt` under the matching `src/test` found two real gaps it had missed: opsdashboard's
`SettingsViewModel` (fixed by this update, see below) and `app`'s `ConversationListViewModel`
(still open — flagged separately). There are 12 ViewModels total (3 `app`, 9 `opsdashboard`); the
true count as of this correction is 11/12.

`SettingsViewModel` (`opsdashboard` module) now has `SettingsViewModelTest` (15 tests): initial
state loads every field from the repository correctly (including the `watchSyncSessionKeyOverride
== null` → `""` mapping); every field-change handler updates its own state field independently;
`addGroup` trims the input, ignores blank/whitespace-only input, and ignores a name already in the
list; `removeGroup` removes only the matching entry; `save()` trims string fields, falls back to
each service's default port on an unparseable port string, and persists a blank/whitespace-only
session-key override as `null` rather than as a literal blank string (verified via
revert-confirm-fail — temporarily un-trimming that field before the `takeIf` made the test fail for
the right reason); and `pushWatchFace()`/`activateWatchFace()`'s success/validation-failure/
failure/no-slot-id branches, including a real fake instead of a mock.

That last group needed `WatchFacePushRepository` split into an interface (production change) plus
`WearWatchFacePushRepository` (the real `androidx.wear.watchfacepush`-backed implementation, wired
into `AppContainer` exactly where the old concrete class was) and a new
`FakeWatchFacePushRepository` (shared `src/sharedTest` fake) — the same interface-plus-fake shape
already used for `AdminAuthGate`/`BiometricAdminAuthGate` in this same module. This also let
`SettingsScreenTest` (from the prior update) drop its awkward real-`Context`-backed
`WatchFacePushRepository` construction in favor of the new fake, and gained three tests actually
exercising the "Install / update" button's success/validation-failure/failure paths — a
previously-disclosed gap this closes at the ViewModel-orchestration level. Still out of scope:
clicking "Activate" itself, which first checks/requests a runtime permission via
`rememberLauncherForActivityResult` — real system-permission-dialog interaction needs a real
device, not a fake repository. `SettingsScreen.kt`'s status comment updated to reflect this.

**Update (September 13, 2026, on and on):** `GroupDetailScreen` (`opsdashboard` module) done as the
tenth screen. Reused `GroupDetailViewModelTest`'s own real-repository Fixture shape
(`RegistryRepository`/`ModelRepository` backed by the shared `src/sharedTest` fakes).
`FakeRegistryApiService` gained a configurable `runsResponse` (it was previously hardcoded to
always return an empty `RunsResponseDto`, so the "Claimed by run" section could never be exercised
by any test). 10 tests added (`GroupDetailScreenTest`): the empty-state placeholders for all three
sections; a populated queue/runs/registry all rendering correctly together, including the exact
formatted row text (entry count, human file size, source, formatted timestamp); TD-128's
"Last refresh failed" banner on a `registry()` failure; the "Release" button's default-enabled
rendering; and — unlike `ModelDetailScreen`/`SessionDetailScreen` — the Gutenberg-fetch,
HuggingFace-fetch, and Assign-model dialogs' full click-through flows, each verified end-to-end
against the fake's captured call arguments (including their own input-validation-gated confirm
buttons, e.g. the Gutenberg dialog's Fetch button staying disabled until a positive book ID is
entered). Those three dialogs are plain `AlertDialog`s with no `ConfirmActionDialog` gate, unlike
this same screen's "Force release" action, which does use `ConfirmActionDialog` and hits the same
`FragmentActivity`/`LocalAdminAuthGate` infrastructure gap already flagged for
`ModelDetailScreen`/`SessionDetailScreen` — only that button's default-enabled rendering is checked,
not the click. `GroupDetailScreen.kt` promoted `experimental` → `beta` (0.3.0).

**Update (September 13, 2026, closing the loop):** the last ViewModel-coverage gap from the
September 13 Correction above is closed — `app` module's `ConversationListViewModel` now has
`ConversationListViewModelTest` (7 tests): `conversations` starts empty with nothing in the
repository and reflects it sorted by most recently updated once populated (`conversations` is a
`stateIn(SharingStarted.WhileSubscribed(5_000), emptyList())`, so these tests start a
`backgroundScope` collector before reading `.value`, the same shape this module's poller-based
ViewModel tests already use for their own StateFlows); `createConversation()` inserts a new
conversation and invokes its callback with the real generated id; `deleteConversation()` removes it
from the list, calls the server's clear-session endpoint when settings are configured (and doesn't
when they aren't), and — verified via revert-confirm-fail (temporarily dropping
`ConversationRepository.deleteConversation()`'s `try`/`catch` around the clear-session call made the
test fail for the right reason) — still deletes locally even when that call throws, matching the
repository's own "local deletion is authoritative" doc comment. `RecordingFakeChatApiService`
(shared `src/sharedTest` fake) gained a configurable `clearSessionResponse` and a
`clearedSessionIds` recorder to support this. True ViewModel coverage is now **12/12**.

**Update (September 13, 2026, moving right along):** `AdminScreen` (`opsdashboard` module) done as
the eleventh screen — the most infrastructure-gap-constrained one yet. Per this screen's own doc
comment, EVERY field edit (not just one admin action, unlike every prior screen) flows through
`ConfirmActionDialog`, gated the instant it composes by the same `FragmentActivity`/
`LocalAdminAuthGate` requirement flagged for `ModelDetailScreen`/`SessionDetailScreen`/
`GroupDetailScreen` — clicking a field's "Next" button is what makes `pendingValue` non-null and
composes it. 9 tests added (`AdminScreenTest`), scoped to what doesn't touch that: each of the
three daemon sections' three render states (populated, admin-disabled, error-with-no-config) —
including the genuinely distinct behavior that `ApiResult.NotFound`'s raw "Not found" message is
shown verbatim for `mns_server`/`registry_server` but mapped to a friendlier "admin routes not
enabled" message only for `metrics_api_server` (see `MetricsSection`); the boolean field's `Switch`
rendering; and the Int/String edit dialogs' own input validation (opening with the right title, the
Int dialog's "Next" button gated on parseable input via `hasSetTextAction()` to target the dialog's
field unambiguously) — stopping before ever clicking "Next". `AdminScreen.kt` promoted
`experimental` → `beta` (0.2.0).

**Update (September 13, 2026, the twelfth and last one):** `TrainerScreen` (`opsdashboard` module)
done as the twelfth and final screen — **all 12 screens now have Compose UI test coverage.** Reused
`TrainerViewModelTest`'s own real-repository Fixture shape (`TrainerRepository` backed by the
shared `src/sharedTest` fakes). Same infrastructure gap as `AdminScreen` (its closest sibling):
every config-field edit AND all three control actions (Checkpoint/Pause/Resume) flow through
`ConfirmActionDialog`, so — as with `AdminScreen` — only rendering, button enabled/disabled state,
and the edit dialogs' own input validation are exercised, stopping before "Next"/before any control
action is clicked. 12 tests added (`TrainerScreenTest`): the status section's not-loaded/populated/
paused-badge states and its per-field `DetailRow`s; the Checkpoint/Pause/Resume buttons' correct
enabled state derived from `status.phase`/`status.paused`; the config section's not-loaded/
populated states and its Int/Bool field rendering; the Int edit dialog's input-validation gating;
the activity log's empty state and per-entry message rendering (deliberately not asserting the
exact formatted timestamp text, since `SimpleDateFormat(..., Locale.getDefault())` against the
device's default time zone isn't deterministic across environments); and the Settings icon.
Deliberately not attempted, for the same reason `ModelListScreenTest` already skipped its own
analogous scenario: a "status refresh succeeds once, then a later poll fails, revealing the
stale-data-with-error banner" test, since `FixedIntervalPoller`'s real 5s interval would need
virtual-time control unavailable at this level. `TrainerScreen.kt` promoted `experimental` → `beta`
(0.2.0); `TrainerUiState.kt` promoted the same way (mirrors `AdminUiState.kt`'s earlier promotion).

**Update (September 13, 2026, the admin-dialog infrastructure, built at last):** built the
admin-action confirm-dialog testing infrastructure this entry has been flagging since
`ModelDetailScreen`'s own update, and used it to close that exact gap on all five screens that had
one. Two new pieces, both under `opsdashboard`:
`com.adai.ops.testutil.ConfirmDialogTestActivity` (`src/androidTest`-only — a bare
`FragmentActivity` subclass, declared in a new `src/androidTest/AndroidManifest.xml`, that does
nothing but give `createAndroidComposeRule<ConfirmDialogTestActivity>()` a real `FragmentActivity`
to launch, since `ConfirmActionDialog`'s `LocalContext.current as FragmentActivity` cast crashes
under the plain `createComposeRule()`/`ComponentActivity` host every other screen test uses) and
`FakeAdminAuthGate` (shared `src/sharedTest` fake implementing `AdminAuthGate`, configurable to
return any `AdminAuthResult`, recording every `reason` it was called with) — the same
interface-plus-fake shape as `WatchFacePushRepository`/`FakeWatchFacePushRepository`.

Five new test files, one per screen with a confirm-dialog-gated flow, each wrapping the screen
under test in `CompositionLocalProvider(LocalAdminAuthGate provides fakeGate) { ... }`:
`ModelDetailScreenConfirmActionTest` (4 tests — the representative one, covering every
`AdminAuthResult` branch: `Success` invokes the action and closes the dialog; `Cancelled` leaves it
open with no error, per its own "not an error" doc comment; `Failed` shows the error message
inside the still-open dialog; a plain "Cancel" tap never calls `authenticate()` at all — verified
via revert-confirm-fail, temporarily making the `Cancelled` branch call `onConfirm()` and
confirming the *"leaves it open"* test would then fail), `SessionDetailScreenConfirmActionTest`,
`GroupDetailScreenConfirmActionTest`, `AdminScreenConfirmActionTest`, and
`TrainerScreenConfirmActionTest` (each just the `Success` happy path for one representative
flow — `AdminAuthResult`'s branch handling is screen-agnostic, so re-testing every branch on
every screen would be pure duplication). Every one of these asserts the real API call the action
makes lands with the right arguments (e.g. `GroupDetailScreenConfirmActionTest` confirms
`forceRelease`'s empty `run_id` and file list; `AdminScreenConfirmActionTest` confirms the PUT body
carries the edited field) and that the dialog actually closes afterward.

Verified end-to-end beyond plain `compileDebugAndroidTestKotlin`: the full androidTest APK now
assembles (`./gradlew :opsdashboard:assembleDebugAndroidTest`), which exercises real manifest
merging and resource linking for the new activity/manifest — not just Kotlin compilation — the
strongest verification available without a Wear-capable device in this sandbox. All 5 screens'
own status comments updated to say their confirm-dialog flow is now covered.

**Update (September 13, 2026, DI containers and entry points):** added coverage for both apps'
manual-DI graphs (`AppContainer.kt`/`AppViewModelProvider.kt`) and launcher entry points
(`Application`/`MainActivity`) — the last item on this entry's original Action Items list.

`app` module — booted `Medium_Phone_API_35` and ran these for real, the first `app`-module test
files verified live end-to-end since `SettingsScreenTest`'s own device run early in this rollout:
`AppContainerTest` (2 tests) reaches the real, already-running `ChatbotApp.container` singleton
(via `ApplicationProvider`) rather than constructing a second `AppContainer` against the same
on-disk Room file — a real collision risk, since every instrumented test in this module already
runs inside that live process — and proves `conversationRepository`/`chatRepository` share one
actual database, cleaning up the one row it creates afterward instead of touching the database
file itself. `AppViewModelProviderTest` (3 tests) exercises `factory()`/`chatFactory()` the same
way `AdaiNavHost` does, including that two `ConversationListViewModel`s built from the same
factory share the same repository. `MainActivityTest` (1 test) launches the real activity and
navigates conversation-list → settings → back. Two genuine, real-device-only bugs surfaced and
fixed along the way (neither reachable from a JVM unit test): `viewModelFactory { initializer {} }`'s
legacy `Factory.create(Class<T>)` throws `UnsupportedOperationException` — the
`create(Class<T>, CreationExtras)` overload with `CreationExtras.Empty` is required instead; and a
brand-new `stateIn(WhileSubscribed(5_000), emptyList())` instance's very first plain `.first()`
can return that initial empty default synchronously, before its upstream has actually run its
first real query — fixed by using `first { predicate }` (keeps collecting until it matches) where
that race mattered. All 23 pre-existing `app`-module instrumented tests plus these 6 new ones pass
together on the real device.

`opsdashboard` module — same shape (`AppContainerTest`, 1 test; `AppViewModelProviderTest`, 6
tests, one per factory method; `MainActivityTest`, 3 tests covering the start destination, a
bottom-nav tab switch, and settings navigation), applying both real-device-discovered fixes above
proactively since they're general `viewModelFactory`/`stateIn` gotchas, not `app`-module-specific
ones. Compile-verified only, plus the full androidTest APK assembling cleanly (same verification
level as every other `opsdashboard` addition, per the standing `wear-sdk` sandbox limitation).
`MainActivityTest` deliberately never clicks an admin action (would open a real `BiometricPrompt`
this sandbox can't drive).

`ChatbotApp.kt`/`OpsApp.kt` promoted `experimental` → `beta` too, on the strength of being
exercised indirectly by these same test files (both `Application.onCreate()` must have already
run correctly by the time `ApplicationProvider.getApplicationContext<...>()` returns a working
`.container`). Not attempted in this pass: the `wearcomplications` services (a distinct, unrelated
untested area — background services, not UI/DI) and `BiometricAdminAuthGate.kt`'s own
`BiometricPrompt` integration test, both already flagged separately below.

Action Items:

- [x] Add ViewModel unit tests first (cheapest — no Compose/Activity needed) — 12 of 12 done (see
  the Correction and "closing the loop" updates above). See the per-ViewModel test files under
  `android/app/src/test`/`android/opsdashboard/src/test`.
- [x] Adopt Compose UI testing (`androidx.compose.ui.test`) for screens once ViewModels are
  covered — infrastructure adopted for both modules now; **all 12 screens done** (`ConversationListScreen`,
  `ChatScreen`, `SettingsScreen` (`app` module), `GroupListScreen`, `ModelListScreen`,
  `SessionListScreen`, `ModelDetailScreen`, `SessionDetailScreen`, `SettingsScreen`,
  `GroupDetailScreen`, `AdminScreen`, and `TrainerScreen` (`opsdashboard` module) — see updates
  above). Note: this sandbox cannot install/run `opsdashboard`'s debug APK on a device (see
  `GroupListScreen`'s update above, `wear-sdk` shared-library requirement), so every
  `opsdashboard` screen's coverage is compile-verified only here, pending real Wear-capable
  device access to actually run any of them.
- [x] Build the admin-action confirm-dialog testing infrastructure (a minimal `androidTest`-only
  `FragmentActivity` host + `createAndroidComposeRule<...>()` + a `FakeAdminAuthGate`) — done, see
  the "admin-dialog infrastructure, built at last" update above. Used to add a dedicated
  `*ScreenConfirmActionTest` file per gated screen (`ModelDetailScreen`, `SessionDetailScreen`,
  `GroupDetailScreen`'s force-release, `AdminScreen`, `TrainerScreen`), closing this gap
  everywhere it was flagged.
- [ ] `BiometricAdminAuthGate.kt` specifically: consider an instrumented test using
  `BiometricPrompt`'s test/fake authenticator support instead of leaving it permanently untested.
- [x] DI containers (`AppContainer.kt`/`AppViewModelProvider.kt` in both apps) and `Activity`/
  `Application` entry points — done, see the "DI containers and entry points" update above
  (`app` module verified live on a real device; `opsdashboard` compile-verified only, per the
  standing `wear-sdk` limitation). The `wearcomplications` services remain entirely untested —
  not attempted in this pass, a distinct area (background services, not UI/DI).

Files to Modify:

- ~33 remaining files under `android/app/src/main`, `android/opsdashboard/src/main`, and
  `android/wearcomplications/src/main` still tagged `experimental` — see
  [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md) for the exact, current list (all of it
  now `wearcomplications`, the one area this pass didn't touch).
- Done: the 8 ViewModel files, `AdminUiState.kt`, `ConversationListScreen.kt`, `ChatScreen.kt`,
  `ChatInputBar.kt`, `MessageBubble.kt`, `ErrorBanner.kt`, `SettingsScreen.kt`, `ChatbotApp.kt`,
  `MainActivity.kt`, `AppContainer.kt`, `AppViewModelProvider.kt` (`app` module), and
  `GroupListScreen.kt`/`ModelListScreen.kt`/`SessionListScreen.kt`/`ModelDetailScreen.kt`/
  `SessionDetailScreen.kt`/`SettingsScreen.kt`/`GroupDetailScreen.kt`/`AdminScreen.kt`/
  `TrainerScreen.kt`/`TrainerUiState.kt`/`OpsApp.kt`/`MainActivity.kt`/`AppContainer.kt`/
  `AppViewModelProvider.kt` (`opsdashboard` module — their admin-action confirm-dialog
  flows are now covered too, via the new `*ScreenConfirmActionTest` files, see the "admin-dialog
  infrastructure, built at last" update above; `SettingsScreen.kt`'s watch-face-push Activate-click
  is the one remaining disclosed gap, still uncovered — see the Correction above). **All 12 TD-048
  screens are now done, including their admin-dialog confirm-click flows.**

---

### TD-164: chatbot-guide.md Needs a Live-Pair Verification Pass

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Documentation | September 13, 2026 | 2-4 hours |

Description:
Split off from [TD-053](../archive/TECHNICAL_DEBT_RESOLVED.md#td-053-chatbotclis-save-and-load-commands-are-non-functional-everywhere)
when that item was resolved. TD-053 fixed `chatbot-guide.md` sections verifiable by reading source
directly: the Quick Start banner, File Requirements, Default File Paths, "Starting the Chatbot",
and Command-Line Help sections (all corrected September 8, 2026, when the doc was found to
describe an entire earlier CLI architecture — a standalone binary taking `[vocab_file]
[model_file] [conversation_save_file]` — that predates `ChatbotCLI` becoming a thin HTTP client
for `chatbot_api_server`), plus the Conversation History / `/save` / `/load` sections and the
`vocab.txt`/`model.bin` example invocations and tokenizer-loading troubleshooting entry (all
corrected September 13, 2026, once `/save`/`/load`/auto-save-on-exit became real). What's left is
a genuinely interactive check — Commands Reference details, Generation Strategies, and
Configuration Parameters — none of which can be confidently verified by reading source alone the
way the sections above were.

Action Items:

- [ ] Start a real `chatbot_api_server` + `chatbot` pair and walk every example in the Commands
  Reference, Generation Strategies, and Configuration Parameters sections, fixing any output that
  no longer matches.
- [ ] Remove the "targeted correction, not a full re-verification" caveat from the top-of-file
  banner once done.

Files to Modify:

- `docs/operations/guides/chatbot-guide.md`

---

### TD-171: No Batch Dimension Anywhere in the Model Stack — Real Parallel Batched Training Not Supported

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open — flagged, not started | Core Model Architecture | September 14, 2026 | Not estimated (large, multi-session architecture project) |

Description:
Split off while investigating TD-170 (`TokenBatchLoader`, resolved by retirement — see the
resolved archive): `EncoderDecoderModel::forward()` and every layer beneath it —
`MultiHeadAttention`, `CrossAttention`, `EncoderBlock`, `DecoderBlock`, `FeedForward`, `LayerNorm`,
`TokenEmbedding`, `PositionalEncoding`, `LanguageModelHead`, and `Matrix` itself — processes exactly
one sequence at a time. There is no batch dimension anywhere in this codebase's model/Matrix stack.
This is the underlying reason `TokenBatchLoader`'s padded, multi-sequence `TokenBatch` output had
nowhere to attach: it assumes a model that can consume a batch dimension, and this one cannot.

Real batched training (multiple sequences forwarded/backwarded together in one GEMM call per
layer, with padding and attention masking to keep sequences independent) would require adding a
batch dimension throughout `Matrix` and every layer built on it. This is not a bug fix or a wiring
task — it is a foundational architecture change, larger in scope than TD-059's per-head attention
fix (which touched exactly two files), and would need its own careful masking-strategy design
before any code is written, given how many layers are involved. Not attempted here; flagged so the
gap is on record rather than rediscovered from scratch next time someone reaches for real batched
training.

Action Items:

- [ ] Owner decision: is this worth pursuing at all? The current per-sample training loop already
  works correctly (gradient accumulation already gives a real "effective batch size" for training
  dynamics) — the only thing true batch-dimension support would add is CPU/GPU parallelism
  efficiency, not new training capability.
- [ ] If pursued: design the batch-dimension convention (e.g. leading batch axis vs. some other
  layout) and the padding/masking strategy for every attention layer, before touching any
  production code — this is a design document on its own, not a first coding step.
- [ ] Scope as its own multi-session project if picked up — not something to fold into any other
  active item's effort estimate.

Files to Modify (if pursued): effectively the entire model stack — `src/Matrix.{hpp,cpp}`,
`src/MultiHeadAttention.{hpp,cpp}`, `src/CrossAttention.{hpp,cpp}`, `src/EncoderBlock.{hpp,cpp}`,
`src/DecoderBlock.{hpp,cpp}`, `src/FeedForward.{hpp,cpp}`, `src/LayerNorm.{hpp,cpp}`,
`src/TokenEmbedding.{hpp,cpp}`, `src/PositionalEncoding.{hpp,cpp}`,
`src/LanguageModelHead.{hpp,cpp}`, `src/EncoderDecoderModel.{hpp,cpp}`, and every GPU backend
equivalent under `src/gpu/`.

---

### TD-172: incremental_trainer's `serve` Command Embeds the Always-On Service in the Same Binary as Its CLI Commands

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — implemented, unit-tested, and verified end-to-end locally (September 14, 2026); only a live deployed host's own systemd cutover remains | Training / Deployment / Tooling | September 14, 2026 | 14-20 hours (matched the revised estimate — see the Implementation update below) |

**Follow-up:** a same-day review of this item's own control pattern ("is this a *complete* control
system, not just a happy path") found pause/resume had no real service-level effect and a wedged
child could hang shutdown forever — fixed and fully resolved as
[TD-173](../archive/TECHNICAL_DEBT_RESOLVED.md#td-173-trainer_services-pauseresume-had-no-real-service-level-effect-and-a-wedged-child-could-hang-shutdown-forever)
rather than reopening this entry.

Description:
`incremental_trainer` is one binary, built from `IncrementalTrainingTool.cpp`'s single ~894-line
`main()`, dispatching on `args[0]` to `init`/`train`/`retrain`/`reset`/`resume`/`serve`/`status`/
`history`. Every one of those except `serve` is a one-shot command: do one thing (optionally
fork+daemonize via `launch_background()`) and exit. `serve` is categorically different — it's the
always-on supervisory service `scripts/adai-trainer.service` installs under systemd
(`ExecStart=... incremental_trainer --config ... serve`, `Type=simple`, `Restart=always`): it
never exits, owns a `TrainerControlState`, starts `TrainerAdminAPI`'s HTTP admin server on a
background thread (see CLAUDE.md "Incremental trainer admin API"), and loops
`resume_last_session()` calls forever with a 45s idle-poll interval
(`IncrementalTrainingTool.cpp` lines ~775-867).

This means the interactive CLI tool and the production service are the exact same binary, built
from the exact same translation unit, dispatched through the exact same `if/else if` chain as
`init`/`status`/`history`. Concretely:
- The systemd unit's whole lifecycle (start/stop/restart, `Restart=always`/`RestartSec=45`
  crash-loop policy, hardening flags) is pinned to whatever `incremental_trainer` happens to be at
  deploy time — there is no way to update, restart, or reason about "the service" independently of
  "the CLI tool," even though they have entirely different operational profiles (one exits in
  seconds, the other runs for weeks).
- `TrainerControlState`/`TrainerAdminAPI` (`src/TrainerControlState.hpp`, `src/TrainerAdminAPI.{hpp,cpp}`)
  are already cleanly factored, reusable, well-scoped classes — the debt isn't in them, it's that
  the ~90 lines of *orchestration* that wires them up (construct control state from `svc_config`,
  start the admin API thread, run the poll loop, handle `SIGTERM`/`SIGINT`) lives inline in the
  same `main()` as one-shot argument parsing for seven unrelated commands, rather than in its own
  small, focused entry point.
- Every future change to the one-shot CLI commands (new flags, new commands, argument-parsing
  fixes) risks touching the same file the production service is built from, and vice versa —
  there's no compiler-enforced boundary between "code that must be safe to run for weeks under
  systemd" and "code that runs for a few seconds from a terminal."

This is the same shape of debt TD-028 already fixed once for this exact tool: `dataset_manager`
was split out of `IncrementalTrainingTool.cpp` into its own binary linking only the library
components it actually needs (`DatasetRegistry`/`DataFetcher`), rather than growing another branch
of the same `main()`. This item applies that identical pattern to `serve`.

**Decision (September 14, 2026):** owner chose the process-supervisor design (option 2 of the two
originally flagged) — the new service binary treats `incremental_trainer` as an external
subprocess it launches and monitors per training pass, rather than linking `IncrementalTrainer`
in-process itself. This is a real architecture change, not just a file move: `TrainerControlState`
today holds fine-grained *live* fields a training pass writes continuously mid-pass
(`current_epoch`, `samples_trained_this_pass`, `last_loss`, `best_loss`, checkpoint counters) and
two control flags (`paused`, `checkpoint_requested`) it reads back the same way — all safe today
only because both sides are threads in one process. With the training pass in a separate process,
none of that is directly shareable any more; it needs a real IPC boundary. Concretely, this
decision means:
- `incremental_trainer` needs a mode where one invocation performs exactly one training pass and
  exits with a meaningful status (already close to what `resume`/`train`/`retrain` do today via
  `resume_last_session()`/`train_on_files()`/`retrain_on_files()` — the supervisor becomes the
  thing that calls this repeatedly instead of a human or a systemd restart cycle).
- Live progress (phase/epoch/loss/checkpoint counters) has to reach the supervisor from the child
  some way other than a shared `TrainerControlState*` — some form of periodic status reporting
  across the process boundary.
- Pause/checkpoint requests have to reach a *running* child some way other than an in-memory
  atomic flag it already sees. `ChatbotTrainer::set_abort_flag()` (what `paused` already drives
  today) still works fine as the *child's own* internal cooperative-abort mechanism; the gap is
  purely how the supervisor tells a specific running child process to trip it (or to force a
  checkpoint) from outside.
- Crash/exit handling gets genuinely simpler than today's in-process story: the supervisor sees
  the child's exit code/signal directly (`waitpid` et al.) instead of relying on the process-wide
  `Restart=always`/`RestartSec=45` systemd policy to recover from *any* crash including one in the
  admin API thread itself — a training-pass crash now can't take the admin API down with it, since
  they're different processes. This is the concrete benefit the original entry called out as the
  reason to consider option 2 at all.

**IPC decision (September 14, 2026, same day):** loopback HTTP, supervisor-proxies-to-child —
explicitly chosen over the file-based channel first proposed here, on the grounds that it's already
an established pattern in this codebase (this *is* exactly what `TrainerAdminAPI` already is: an
httplib server bound to `127.0.0.1`) and is more portable (no new file-format/staleness-detection
protocol to invent, and cpp-httplib already builds on every platform `incremental_trainer` ships
on, including Windows — confirmed `incremental_trainer.exe` is a real packaged target, unlike a
Unix-domain-socket approach which would need new platform-specific code `PortableSocket.hpp`
doesn't currently cover). Concretely:
- `incremental_trainer` gains a mode where a single-pass invocation also starts its own
  `TrainerAdminAPI`, bound to a port the supervisor assigns at launch (env var or `--admin-port`
  flag — only one child ever runs at a time under this design, so no port-discovery problem).
  **`TrainerControlState`/`TrainerAdminAPI` need zero changes** — they keep working exactly as they
  do today, just owned by the child process instead of `serve`'s process. This is the direct payoff
  of choosing this option: the best-tested, most subtle part of the current system (the atomic
  field semantics, the log ring buffer, the `checkpoint`'s `wait_ms` busy-poll loop, the
  pause/resume `wake()` condvar dance) is reused completely unchanged.
- The supervisor hosts its own always-on `TrainerAdminAPI`-shaped HTTP listener (same host/port
  config keys operators already know) that acts as a thin reverse proxy: while a child is alive,
  forward every `/admin/*` request to the child's port and relay its response; while idle (no child
  running), answer directly (e.g. `phase: idle`) without proxying anywhere.
- Handle the short window between spawning a child and its admin port coming up (the proxy should
  say "starting," not error or hang) and the moment a child exits (proxy should fall back to
  "idle" instead of erroring on a now-dead connection).

**Implemented (September 14, 2026, same day):** every action item below is done. Summary of what
landed:
- `incremental_trainer resume` gained `--admin-port <N>` (`IncrementalTrainerArgs.hpp/.cpp`): when
  set (always alongside `--foreground`, only ever passed by `trainer_service`), it constructs a
  `TrainerControlState`/`TrainerAdminAPI` exactly as the old `serve` command used to, bound to
  `127.0.0.1:<N>`, for the duration of that one pass — waiting (bounded, 2s) for the listener to
  actually bind before starting the pass, and stopping/joining it cleanly afterward. Absent for
  every interactive/manual invocation, so nothing changes for `train`/`retrain`/plain `resume`.
- New `src/ChildProcess.{hpp,cpp}`: cross-platform launch/monitor helper (POSIX fork/execvp/waitpid;
  Windows CreateProcess/GetExitCodeProcess/TerminateProcess), unit-tested against real `/bin/sh`
  children (8 tests: exit-code capture, still-running polling, `request_stop()` termination,
  double-start rejection, destructor reaping a still-running child without hanging, sequential
  reuse).
- New `src/TrainerServiceProxy.{hpp,cpp}`: the supervisor's own admin HTTP listener. Proxies every
  `/admin/*` request to the current child's `TrainerAdminAPI` port via a plain `httplib::Client`
  call when one is set; falls back to an idle-shaped default (matching `TrainerAdminAPI::handle_
  status()`'s own shape) when none is, and to a distinct 503 "starting or exiting" response when a
  child is expected but its connection fails (the two edge windows called out in the IPC decision
  above). `GET`/`PUT /admin/config` while idle read/write the shared `daemon_config.db` directly
  (reusing `DaemonConfigStore`) so a config change made while idle still applies to the next child.
  11 tests: idle defaults for every endpoint, a real `TrainerAdminAPI` instance used as the "live
  child" to verify real proxying (not a hand-rolled fake), the unreachable-port 503 case, and the
  live-to-idle transition.
- New `src/TrainerServiceMain.cpp`: the `trainer_service` binary's `main()` — reuses
  `parse_incremental_trainer_global_args()` for its own `--config`/`--model`/`--gpu-strategy`
  passthrough (no new parser needed; `--foreground`/`--admin-port` are what *it* passes to each
  child, not something an operator passes to it), resolves the sibling `incremental_trainer`
  binary path from `argv[0]`, and runs the launch/poll loop (immediate relaunch after a pass that
  did work, 45s poll interval otherwise — the same shape `serve`'s loop used) with
  `SIGTERM`/`SIGINT` forwarded to whatever child is currently running.
- New `TRAINER_CHILD_ADMIN_PORT` config key (`Config.hpp/.cpp`, default 8085) — the child's private
  port, distinct from `TRAINER_ADMIN_PORT` (the supervisor's own public listener).
- `src/CMakeLists.txt`: new `trainer_service` target, gated on `HTTPLIB_INCLUDE_DIR` like
  `incremental_trainer`'s own admin API — but with no reduced-functionality fallback build, since
  hosting the admin proxy is this binary's entire purpose. Deliberately minimal dependencies: no
  `adai_models`/`adai_nlp`/GPU objects at all, just `adai_core` + httplib + pthread.
  `IncrementalTrainingTool.cpp`'s `serve` branch removed entirely (owner chose no deprecated
  alias — a clean break, consistent with this tracker's usual preference).
- `scripts/adai-trainer.service` rewritten for `trainer_service` (documents the real operational
  improvement this design gives for free: a GPU-driver crash during a pass now only takes down the
  *child*, not the whole always-on process/admin-API — previously the identical crash took `serve`
  itself down, relying on systemd's own restart). `scripts/install_incremental_trainer.sh` copies
  `trainer_service` alongside `incremental_trainer` when built (optional, like `registry_server`),
  both locally and over the `--remote` SSH+rsync path; existing shell-test suite (57 tests) still
  passes unchanged. `CLAUDE.md`'s Executable Targets table and "Incremental trainer admin API"
  section updated for the new binary and process boundary.
- Note on the originally-planned "factor the shared bootstrap preamble" action item: turned out
  unnecessary once the actual scope became clear — `trainer_service` never needs the ~140-line
  MNS-resolution/architecture-sync preamble at all (that logic stays entirely inside each spawned
  `incremental_trainer` child, unchanged); it only needs a plain `ConfigLoader::discover_config_
  path()`/`load()` call for its own settings, which isn't duplication worth abstracting — every
  other binary in this codebase already calls those same two static methods independently.

Verified end-to-end, not just unit-by-unit: built a minimal real session (`vocab_builder` +
`incremental_trainer init` + one pending file) and ran `trainer_service` against it directly —
observed a real training pass complete, the child's own `TrainerAdminAPI` come up
(`Admin API listening on 127.0.0.1:18501` in the log), `GET /admin/status` through the
supervisor's proxy port correctly relay that child's live phase, an immediate second pass launch
after the first did work, and a clean SIGTERM shutdown with `pgrep` confirming zero leftover
`incremental_trainer`/`trainer_service` processes afterward. Full `ctest` suite: 131/131 passing
(two single-test flakes — `TrainerAdminAPITests`, `ScriptsTests_monitor_training` — seen once each
under full `-j$(nproc)` parallel load and confirmed as pre-existing resource-contention flakiness,
not real failures: both passed standalone and on a second full-suite run, and a *different*,
completely unrelated test flaked instead on that second run).

Action Items:

- [x] **Owner decision** on the new service binary's shape: **process supervisor** (external
  subprocess model), not in-process library reuse. See the Decision update above.
- [x] **IPC mechanism decision:** loopback HTTP with the supervisor proxying to whichever child is
  currently alive. See the IPC decision update above. `TrainerControlState`/`TrainerAdminAPI`
  require no redesign under this choice.
- [x] Give `incremental_trainer` a clean, supervisor-friendly single-pass invocation. Done —
  `resume --admin-port <N>`, see the Implementation update above.
- [x] Build the new supervisor binary. Done — `trainer_service`/`TrainerServiceMain.cpp`.
- [x] Implement the supervisor's own admin HTTP listener as a thin reverse proxy. Done —
  `TrainerServiceProxy`.
- [x] Factor the shared bootstrap preamble. Turned out unnecessary — see the Implementation
  update above for why.
- [x] Remove the `serve` branch. Done — owner chose a clean removal, no deprecated alias.
- [x] Update `scripts/adai-trainer.service`. Done.
- [x] Update `scripts/install_incremental_trainer.sh` / its test suite. Done — 57/57 still passing.
- [x] Add the new binary to `src/CMakeLists.txt` / CLAUDE.md. Done.
- [x] Review `tests/incremental_trainer_control_test.cpp` / `..._background_test.cpp`. Reviewed —
  neither exercised the `serve` branch through the CLI binary specifically (both already test
  `TrainerControlState`/`IncrementalTrainer` directly), so no change was needed.
- [x] New tests for the supervisor↔child boundary. Done — `ChildProcessTests` (8),
  `TrainerServiceProxyTests` (11).
- [ ] **Not done, not this session's to do:** cut over an actual live deployed host's systemd
  unit and binaries. Everything in this repo is ready for that cutover (updated
  `adai-trainer.service`, `trainer_service` binary, install-script support) but actually running
  it against a real production host is an operational step outside a coding session's reach —
  same category as TD-047's "cut the first Android release" or TD-033/TD-050's hardware-blocked
  validation items.

Files Modified:

- `src/IncrementalTrainingTool.cpp` — `serve` branch removed; `--admin-port` single-pass invocation
  mode added to `resume`; usage text updated
- `src/IncrementalTrainerArgs.{hpp,cpp}` — `--admin-port` flag parsing; `"serve"` removed from
  `incremental_trainer_command_defers_gpu_init()`
- `src/ChildProcess.{hpp,cpp}` (new) — cross-platform child-process launch/monitor helper
- `src/TrainerServiceProxy.{hpp,cpp}` (new) — the supervisor's reverse-proxy admin HTTP listener
- `src/TrainerServiceMain.cpp` (new) — the `trainer_service` binary's `main()`
- `src/Config.{hpp,cpp}` / `config.trainer.conf` — new `TRAINER_CHILD_ADMIN_PORT` key
- `src/CMakeLists.txt` — new `trainer_service` target
- `tests/incremental_trainer_args_test.cpp` — `--admin-port` parsing tests; `"serve"` removed from
  the defers-GPU-init test
- `tests/child_process_test.cpp` (new), `tests/trainer_service_proxy_test.cpp` (new)
- `tests/CMakeLists.txt` — new `childProcessTests`/`trainerServiceProxyTests` targets
- `scripts/adai-trainer.service` — rewritten for `trainer_service`
- `scripts/install_incremental_trainer.sh` — installs `trainer_service` alongside
  `incremental_trainer` (local and `--remote`), optional like `registry_server`
- `CLAUDE.md` — Executable Targets table, "Incremental trainer admin API" section, `IncrementalConfig`
  paragraph, config key table, Active Technical Debt Tags table

Context: originally scoped at 8-12 hours under the (not chosen) in-process-reuse option, where
`TrainerControlState`/`TrainerAdminAPI` needed no change at all — this item was purely about where
~90 lines of orchestration live. The process-supervisor decision trades that simplicity for real
process-crash isolation between the admin API and a training pass; a first pass at the IPC choice
(file-based status/control) would have needed re-deriving `TrainerControlState`'s serialization and
a new staleness-detection protocol, pushing the estimate up to 20-28 hours. Settling on loopback
HTTP with the supervisor proxying to the child — chosen for being an already-established pattern in
this codebase (`TrainerAdminAPI` already is exactly this: an httplib server on `127.0.0.1`) and for
portability (no new per-platform IPC code, unlike a Unix-domain-socket approach) — brought
`TrainerControlState`/`TrainerAdminAPI` back to needing zero changes, settling the estimate at
14-20 hours: real work, but proxy-layer-and-process-lifecycle work, not a redesign of the
already-tested control/status machinery. Does not touch `IncrementalTrainer.{hpp,cpp}`'s own API
surface (the child process still calls it exactly as today), so it remains compatible with
[TD-039](#td-039-core-trainingmetrics-classes-too-large-and-fast-moving-to-certify-stable)'s
freeze-in-place plan for that class. Direct precedent for the binary split itself: TD-028 (June 7,
2026) split `dataset_manager` out of this exact same `IncrementalTrainingTool.cpp` for the
identical reason (a command that needed only a subset of the tool's dependencies was growing
another branch of one large `main()` instead of becoming its own focused binary) — see its
[resolved entry](../archive/TECHNICAL_DEBT_RESOLVED.md#td-028-separate-dataset-management-from-incrementaltrainer).

---

## Resolved Items

196 items resolved. See [archive/TECHNICAL_DEBT_RESOLVED.md](../archive/TECHNICAL_DEBT_RESOLVED.md) for full details.

---
## Future Improvements

These are lower-priority enhancements that don't currently block development:

### Performance Optimizations

1. **Memory Pool for Matrix Allocations**
   - Reduce allocation overhead during forward/backward passes
   - Pre-allocate memory for common matrix sizes
   - Reduce memory fragmentation

2. **Monitor Augmentation Performance** (from Priority 2 completion)
   - **Priority:** Low
   - **Effort:** Ongoing
   - **Description:** Track real-world speedup from the parallel data augmentation implementation (Priority 2) across different hardware configurations and dataset sizes.
   - **Implementation:**
     - Collect augmentation throughput metrics during training runs
     - Compare 1-thread vs multi-thread performance in production workloads
     - Record results in benchmark log for regression detection
   - **Files:** `src/EfficientBatching.hpp`, `benchmarks/AugmentationBenchmark.cpp`
   - **Reference:** `docs/development/guides/AUGMENTATION_CHECKLIST.md`

3. **Batched Inference Engine (Priority 3)**
   - **Priority:** Medium
   - **Effort:** Medium (estimated 2-4 days)
   - **Description:** Implement a batched inference engine to process multiple inference requests simultaneously, achieving 10-20x throughput improvement for serving/production workloads.
   - **Expected Impact:** 10-20x throughput improvement
   - **Implementation:**
     - Group incoming inference requests into dynamic batches
     - Process batches through the model in a single forward pass
     - Return individual results to each caller
     - Tune batch size and timeout for latency vs throughput trade-off
   - **Benefits:**
     - Dramatically higher request throughput
     - Better GPU/CPU utilization
     - Lower per-request cost at scale
   - **Files:** `src/BatchedInferenceEngine.hpp`, `src/ChatbotAPIServer.cpp`, `benchmarks/BatchedInferenceBenchmark.cpp`
   - **Reference:** `docs/development/guides/AUGMENTATION_CHECKLIST.md`, `docs/development/archive/BATCHED_INFERENCE_SUMMARY.md`

4. **Attention Head Parallelism (Priority 4)**
   - **Priority:** Medium
   - **Effort:** Medium (estimated 2-3 days)
   - **Description:** Parallelize multi-head attention computation so each attention head is processed concurrently, targeting a 2-4x speedup in the attention layer.
   - **Expected Impact:** 2-4x speedup in attention computations
   - **Implementation:**
     - Distribute attention heads across OpenMP threads
     - Ensure thread-safe accumulation of outputs
     - Validate correctness against single-threaded reference
   - **Benefits:**
     - Faster inference and training for transformer-based models
     - Better utilization of multi-core CPUs
   - **Files:** `src/MultiHeadAttention.hpp`, `src/MultiHeadAttention.cpp`, `benchmarks/AttentionHeadBenchmark.cpp`
   - **Reference:** `docs/development/guides/AUGMENTATION_CHECKLIST.md`, `docs/development/archive/ATTENTION_HEAD_PARALLELISM_SUMMARY.md`

### Code Quality

1. **Add Benchmarking Suite**
   - Performance regression testing
   - Track training throughput over time
   - Compare against baseline implementations

### Developer Experience

1. **Add Python Bindings**
   - Enable easier experimentation
   - Broader community adoption
   - Integration with Python ML ecosystem

2. **Improve Build Times**
   - Use precompiled headers
   - Optimize template instantiations
   - Modularize includes

### Configuration and Service Management

Related to Steps 1-5: Daemon Service Implementation

1. **JSON Configuration Format Support** (Step 1 Enhancement)
   - **Priority:** Low
   - **Effort:** 2-3 hours
   - **Description:** Support JSON in addition to key=value format
   - **Implementation:**
     - Add JSON parsing library (nlohmann/json or similar)
     - Implement `load_from_json()` method in ConfigLoader
     - Support both formats with auto-detection
     - Add schema validation for JSON config
   - **Benefits:**
     - More expressive configuration (nested objects, arrays)
     - Better tooling support (editors, validators)
     - Easier integration with deployment tools
   - **Files:** `src/Config.cpp`, `src/Config.hpp`, `CMakeLists.txt`

2. **Configuration Profiles** (Step 1 Enhancement)
   - **Priority:** Low
   - **Effort:** 3-4 hours
   - **Description:** Support named configuration profiles (dev, staging, prod)
   - **Implementation:**
     - Add `--profile` command-line argument
     - Load base config + profile-specific overrides
     - Support profile inheritance
   - **Files:** `src/Config.cpp`, `config.dev.conf`, `config.prod.conf`

3. **Model State Persistence on Shutdown** (Step 2 Enhancement)
   - **Priority:** Medium
   - **Effort:** 4-6 hours
   - **Description:** Automatically save model weights during graceful shutdown
   - **Implementation:**
     - Check if MODEL_PATH is configured during shutdown
     - If set, call model->save_weights() in shutdown sequence
     - Add checkpoint metadata (timestamp, loss, etc.)
     - Log save progress with structured logging
   - **Benefits:**
     - No manual model saving needed
     - Automatic checkpointing on restart
     - Reduced risk of losing trained state
   - **Files:** `src/ChatbotAPIServer.cpp`

4. **Graceful Reload (Zero-Downtime Restart)** (Step 2 Enhancement)
   - **Priority:** Low
   - **Effort:** 8-12 hours
   - **Description:** Reload model without dropping connections
   - **Implementation:**
     - Implement SIGHUP handler for reload signal
     - Load new model in background thread
     - Atomic swap to new model after loading
     - Keep serving requests during reload
   - **Benefits:**
     - True zero-downtime deployments
     - Seamless model updates
   - **Files:** `src/ChatbotAPIServer.cpp`, `src/ChatbotAPI.cpp`

### Logging and Observability

Related to Step 3: Structured Logging

1. **JSON Log Output Format** (Step 3 Enhancement)
   - **Priority:** Medium
   - **Effort:** 2-3 hours
   - **Description:** Support JSON-formatted logs for machine parsing
   - **Implementation:**
     - Add `LOG_FORMAT` config option (text/json)
     - Implement JSON formatter in Logger class
     - Include structured fields: timestamp, level, message, component, context
     - Support ECS (Elastic Common Schema) format
   - **Benefits:**
     - Better integration with log aggregation (ELK, Splunk)
     - Easier parsing and analysis
     - Structured error reporting
   - **Example Output:**

     ```json
     {"timestamp":"2026-03-01T16:15:17.862Z","level":"info","message":"Server started","port":8080,"pid":1234}
     ```

   - **Files:** `src/Logger.cpp`, `src/Logger.hpp`, `src/Config.hpp`

2. **Per-Module Log Levels** (Step 3 Enhancement)
    - **Priority:** Low
    - **Effort:** 4-5 hours
    - **Description:** Different log levels for different components
    - **Implementation:**
      - Create named loggers per component (api, model, tokenizer)
      - Support per-module configuration: `LOG_LEVEL_API=DEBUG`
      - Add logger registry for runtime level changes
    - **Benefits:**
      - Fine-grained debugging control
      - Reduce log noise in production
      - Debug specific components without full verbosity
    - **Example:**

      ```ini
      LOG_LEVEL_API=DEBUG
      LOG_LEVEL_MODEL=INFO
      LOG_LEVEL_TOKENIZER=WARN
      ```

    - **Files:** `src/Logger.cpp`, `src/Logger.hpp`, `src/Config.cpp`

3. **Custom Log Sinks** (Step 3 Enhancement)
    - **Priority:** Low
    - **Effort:** 6-8 hours
    - **Description:** Support multiple log destinations
    - **Implementation:**
      - Add syslog sink for system logging
      - Add network sink (TCP/UDP) for centralized logging
      - Add custom sink interface for extensibility
      - Configure via `LOG_SINKS` config option
    - **Files:** `src/Logger.cpp`, `src/sinks/SyslogSink.cpp`, `src/sinks/NetworkSink.cpp`

### Container and Deployment

Related to Steps 4-5: Docker and systemd

1. **Kubernetes Deployment Manifests** (Step 4 Enhancement)
    - **Priority:** Low
    - **Effort:** 4-6 hours
    - **Description:** Create production-ready Kubernetes manifests
    - **Implementation:**
      - Create Deployment manifest with resource limits
      - Add Service manifest for load balancing
      - Add ConfigMap for configuration
      - Add HorizontalPodAutoscaler for auto-scaling
      - Add probes (liveness, readiness, startup)
    - **Files:** `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/configmap.yaml`, `docs/operations/KUBERNETES_DEPLOYMENT.md`

2. **Metrics Endpoint for Prometheus** (Step 5 Enhancement)
    - **Priority:** Medium
    - **Effort:** 6-8 hours
    - **Description:** Expose /metrics endpoint with application metrics
    - **Implementation:**
      - Add Prometheus C++ client library
      - Expose metrics: request_count, request_duration, active_sessions, model_inference_time
      - Add custom metrics: generation_tokens_total, vocabulary_size, etc.
      - Integrate with systemd service monitoring
    - **Benefits:**
      - Production monitoring and alerting
      - Performance tracking over time
      - Integration with Grafana dashboards
    - **Files:** `src/MetricsExporter.cpp`, `src/ChatbotAPIServer.cpp`, `CMakeLists.txt`

3. **systemd Socket Activation** (Step 5 Enhancement)
    - **Priority:** Low
    - **Effort:** 5-7 hours
    - **Description:** Support on-demand service activation
    - **Implementation:**
      - Add socket activation support with sd_listen_fds()
      - Create adai.socket unit file
      - Modify server to accept pre-bound socket from systemd
      - Support both direct and socket-activated startup
    - **Benefits:**
      - Reduced resource usage for idle services
      - Faster first-request handling (systemd pre-binds socket)
      - Better system integration
    - **Files:** `src/ChatbotAPIServer.cpp`, `scripts/adai.socket`, `docs/operations/SYSTEMD_DEPLOYMENT.md`

4. **Multi-Instance Service Templates** (Step 5 Enhancement)
    - **Priority:** Low
    - **Effort:** 2-3 hours
    - **Description:** Support running multiple chatbot instances
    - **Implementation:**
      - Convert `adai.service` to template unit (`adai@.service`)
      - Use %i instance identifier for port assignment
      - Support per-instance configuration files
    - **Example Usage:**

      ```bash
      systemctl start adai@8080
      systemctl start adai@8081
      systemctl start adai@8082
      ```

    - **Files:** `scripts/adai@.service`, `docs/operations/SYSTEMD_DEPLOYMENT.md`

5. **Health Check Enhancements** (Steps 4-5 Enhancement)
    - **Priority:** Low
    - **Effort:** 3-4 hours
    - **Description:** More comprehensive health checking
    - **Implementation:**
      - Add detailed health endpoint returning component status
      - Check vocabulary loaded, model initialized, memory usage
      - Support readiness vs liveness checks (Kubernetes)
      - Add configurable health check timeout
    - **Response Example:**

      ```json
      {
        "status": "healthy",
        "components": {
          "tokenizer": {"status": "ok", "vocab_size": 9925},
          "model": {"status": "ok", "initialized": true},
          "memory": {"status": "ok", "usage_mb": 512}
        },
        "uptime_seconds": 3600
      }
      ```

    - **Files:** `src/ChatbotAPIServer.cpp`, `src/ChatbotAPI.cpp`

---

## Deferred Decisions

Architecture decisions that were considered and explicitly not pursued, with the reasoning and the
condition that would reopen them. Unlike Future Improvements above, these are not queued work —
they're a record of "not now, and here's why," so the reasoning isn't re-litigated from scratch later.

### AMD Radeon / ROCm-HIP GPU Backend — Not Pursued

**Date:** September 7, 2026
**Component:** GPU / Architecture

**Decision:** Do not add a third GPU backend (ROCm/HIP) alongside the existing CUDA
(`src/gpu/MatrixGPU.cu`) and SYCL (`src/gpu/sycl/`) paths.

**Reasoning:**

- No AMD Radeon hardware is available or targeted for this project — a HIP backend would be
  maintained by inspection only, the same way `MatrixGPU.cu` already is on the primary dev
  machine (whose only GPU is an integrated Intel UHD 620 — neither CUDA nor a discrete Intel ARC
  device is actually present there either).
- The two existing backends already demonstrate the sync cost: CUDA and SYCL were edited in the
  same commit, on the same day, and still drifted (see the September 2026 fix that added float4
  vectorization, sub-group/warp-shuffle reduction, and `GPU_STRATEGY` queue-priority parity to the
  SYCL backend to catch it up with CUDA). A third backend multiplies that sync surface rather than
  adding to it linearly.
- Unlike SYCL (a ground-up rewrite in a different kernel-lambda paradigm), HIP is close to
  source-compatible with CUDA — AMD's hipify tools can mechanically translate most of
  `MatrixGPU.cu` (`cuda*` → `hip*`, `cublas` → `hipblas`/`rocblas`) — so adding it later is
  expected to cost meaningfully less than SYCL did.

**Revisit when:** Either (a) AMD Radeon/ROCm hardware becomes available to build and test
against, or (b) a concrete deployment target requires it. The CMake mutual-exclusion pattern
(`ENABLE_GPU`/`ENABLE_SYCL`, see `CMakeLists.txt`) already extends cleanly to a third
`ENABLE_HIP` option if/when that happens.

---

### ThreadSanitizer "Data Races" in Matrix.cpp's OpenMP-Parallelized Code — Confirmed Tool Limitation, Not a Bug

**Date:** September 12, 2026
**Component:** Testing / Sanitizers / Core Matrix Ops

**Decision:** Do not modify `src/Matrix.cpp`'s `#pragma omp parallel for` regions (`operator*()`,
`transpose()`, `scale()`, and the other OpenMP-parallelized methods) in response to the ThreadSanitizer
"data race" warnings they produce under the `tsan` CMake preset — running `chatbotapiTests` (or any
other suite exercising real matrix multiplication) reports roughly 20-90+ of them depending on how
much multiplication the run does. Originally surfaced as a side observation while verifying TD-156's
concurrency fix, flagged for follow-up, and investigated here to a definitive conclusion rather than
left open.

**Reasoning:**

- Confirmed via `git stash` (independently, across two separate investigations) that these warnings
  are pre-existing on pristine `main`, unrelated to any specific change — identical counts with or
  without, e.g., TD-156's mutex.
- Read the full race-report bodies (not just the one-line summaries `TSAN_OPTIONS` prints by
  default) with `TSAN_OPTIONS=history_size=7` for deeper history. Every one follows the same shape:
  a worker thread spawned by libgomp (visible in the stack traces via `pthread_create` inside
  `libgomp.so.1`) reads/writes a `Matrix`'s backing `std::vector<std::vector<float>>` inside a
  `#pragma omp parallel for` region, and the "previous access" it's reported racing against is the
  *calling* thread's own move-assignment/`delete`/further construction of that same `Matrix`
  immediately after the parallel region returns — exactly the access pattern OpenMP's **mandatory
  implicit barrier** at the end of a `parallel for` (no `nowait` is used anywhere in this file)
  guarantees is safe.
- Manual review of every flagged call site confirms genuinely disjoint memory per loop iteration
  (distinct output rows/elements per index; each `Matrix`'s backing buffers are already fully
  allocated, single-threaded, before its parallel region starts — see `Matrix::Matrix(int, int)`'s
  `data.resize(...)`) — there is no aliasing bug to fix in this file's own logic.
- This is a known, documented ThreadSanitizer + GCC-`libgomp` limitation, not specific to this
  codebase: a ThreadSanitizer co-author has stated on the GCC mailing list that TSan reliably
  supports "pthread-based synchronization and atomic compiler builtins" only ([Konstantin
  Serebryany, gcc.gnu.org, 2013](https://gcc.gnu.org/legacy-ml/gcc/2013-02/msg00295.html)). GCC's
  `libgomp` implements its own barrier/thread-pool synchronization without routing through
  primitives TSan instruments, so TSan cannot see the happens-before edge the barrier actually
  provides at runtime. The documented fix for accurate OpenMP race detection is **Archer**, an
  LLVM/OMPT-based race predictor built on Clang's `libomp` rather than GCC's `libgomp` ([HPC Wiki,
  ThreadSanitizer](https://hpc-wiki.info/hpc/ThreadSanitizer)) — a real but nontrivial toolchain
  change (Clang + `libomp` + Archer instead of GCC + `libgomp` for `tsan` builds specifically), not
  a code fix in this repository.
- Tested the commonly-suggested `TSAN_OPTIONS=ignore_noninstrumented_modules=1` workaround directly
  against `chatbotapiTests`: it does **not** suppress these warnings (18 races still reported in that
  run) — that flag only helps when the racing access itself happens inside literal non-instrumented
  code, but here both accesses are in our own instrumented `Matrix.cpp`; only the *synchronization
  connecting them* (libgomp's barrier) is invisible to TSan, which this flag doesn't address.
- A durable comment documenting this (with a pointer back to this entry) was added directly in
  `src/Matrix.cpp`, since that's where anyone confused by a `tsan` run will actually be looking.

**Revisit when:** The project adopts a Clang + `libomp` (+ Archer) toolchain for `tsan` builds
instead of GCC + `libgomp` — until then, expect roughly 20-90+ TSan warnings inside `Matrix.cpp`'s
OpenMP regions on any `tsan` run that exercises real matrix multiplication, all attributable to this
limitation. A genuinely different-shaped warning — e.g., one whose two accesses implicate actually
overlapping memory rather than a pre-barrier/post-barrier access pair — would still be worth
investigating on its own merits rather than dismissed by pattern-matching against this entry.

---

### AttentionHeadBenchmark's Apparent Hang — Confirmed Host CPU Contention, Not a Bug

**Date:** September 13, 2026
**Component:** Benchmarks / Tooling

**Decision:** Do not change `MultiHeadAttention.cpp`/`Matrix.cpp`/`Timer` in response to
`benchmarks/AttentionHeadBenchmark.cpp` appearing to hang — confirmed the binary was never actually
stuck, just running 200-300x slower than expected under real host CPU contention in this dev
sandbox. Originally logged as TD-163 ("hangs indefinitely, root cause unknown") after being found
incidentally while verifying TD-059's attention-head-splitting fix; investigated here to a
definitive conclusion rather than left open.

**Reasoning:**

- Reproduced directly: `timeout 15 ./attention_head_benchmark` prints only the
  "Benchmark: Scaling with Number of Attention Heads" table header before the timeout kills it,
  matching TD-163's original report exactly.
- Built a disposable, incrementally-instrumented repro harness calling
  `MultiHeadAttention::forward_parallel()` directly (not through the full benchmark) to localize
  the stall. `OMP_NUM_THREADS=1` still stalled — ruling out nested-OpenMP/thread-pool-exhaustion
  theories, since a team of one thread doesn't exhibit those. Instrumenting matrix dimensions at
  every call showed no growth over time — ruling out a leak/accumulation bug. The *position* of
  the apparent stall moved between runs depending only on how much `std::cerr` instrumentation was
  added (i.e. how much wall-clock time each call itself took) — the signature of external timing
  variance, not a fixed code path.
- Re-ran with a much longer timeout (90s): it **completed**, just slowly — 50 sequential-path
  iterations of a trivial 128×512 `forward_parallel()` call took ~18.7 **seconds** (should be low
  tens of milliseconds). Not a deadlock; a severe slowdown.
- `/usr/bin/time -v` on a 20s run: **5885 involuntary vs. 380 voluntary context switches**, and
  "Percent of CPU this job got: 194%" on an 8-core box (should approach 800% for a
  fully-parallel, uncontended `#pragma omp parallel for` region). `cat /sys/fs/cgroup/cpu.stat`
  showed **zero** cgroup throttling (`nr_throttled 0`) — this is not a CPU quota/cgroup limit, it's
  the kernel scheduler genuinely preempting this process's threads in favor of other real,
  concurrently-running processes on the same host (`top`/`uptime` at the time: load average
  5.8-6.7 on 8 cores, from several other Claude Code sessions, an editor, the desktop compositor,
  and — found and stopped mid-investigation — a leftover Android emulator from earlier unrelated
  work in this same session).
- Every `forward_parallel()` call enters several small `#pragma omp parallel for` regions
  regardless of its own `use_parallel` argument — `Matrix::operator*`'s own internal
  `if (rows > 64)`-gated parallelism fires for the Q/K/V projections either way; only the per-head
  loop is actually gated by `use_parallel`. Each such region ends in OpenMP's mandatory implicit
  barrier, so *every* participating thread must be scheduled promptly for the region to complete —
  one thread preempted by unrelated host load stalls the whole team. A benchmark built around many
  small, frequent parallel regions is disproportionately sensitive to exactly this kind of
  contention compared to a typical single-threaded workload, which is why it manifests here and
  not elsewhere in the codebase.
- Tested `OMP_WAIT_POLICY=PASSIVE` (idle threads block immediately instead of spin-waiting) as a
  possible mitigation: no meaningful improvement (still ~18.6s for the same config) — confirms the
  bottleneck is the kernel scheduler not promptly running this process's threads at all, not
  wasted CPU cycles spent spinning while waiting.
- A durable comment documenting this (with a pointer back to this entry) was added directly in
  `benchmarks/AttentionHeadBenchmark.cpp`, since that's where anyone confused by an apparent hang
  will actually be looking.

**Revisit when:** Running this benchmark on a quiet, dedicated machine (or CI runner) still shows
it stalling for more than a few seconds — that would indicate a real regression rather than this
environment's characteristic contention. Until then, expect this benchmark to run slowly and
unpredictably whenever the host is under heavy concurrent load, with no code change needed.

---

## Process Guidelines

### Adding New Technical Debt

When adding a new technical debt item:

1. **Create an entry in this document** with all required fields:
   - Unique ID (TD-XXX)
   - Priority (High/Medium/Low)
   - Component
   - Effort estimate
   - Description and impact
   - Location in code
   - Task checklist
   - Files affected

2. **Create a GitHub issue** using the technical debt template:
   - Link to this document
   - Use label: `technical-debt`
   - Assign priority label
   - Add to project board

3. **Update code comments** to reference the tracking item:

   ```cpp
   // See TD-001 in TECHNICAL_DEBT.md - Parameter exposure incomplete
   ```

4. **Remove untracked TODOs**

- All TODOs must be tracked here or in GitHub issues

### Prioritization Criteria

High Priority:

- Blocks new feature development
- Causes bugs or incorrect behavior
- Security or stability issues
- Affects multiple components

Medium Priority:

- Improves code maintainability significantly
- Reduces technical complexity
- Enables future features
- Clear path to resolution

Low Priority:

- Nice-to-have improvements
- Cosmetic code cleanup
- Performance optimizations (non-critical)
- Developer convenience features

### Resolving Technical Debt

When resolving a debt item:

1. Complete all tasks in the checklist
2. Add tests to prevent regression
3. Update documentation
4. Move item from "Active" to "Resolved" section with resolution date
5. Close related GitHub issue
6. Remove code comments referencing the item

---

## Statistics

### By Priority

Recomputed directly from the 13 `### TD-NNN` entries under [Active Technical Debt](#active-technical-debt) — re-derive this from that list rather than trusting it blindly once an item resolves or a new one is filed.

|Priority|Count|Percentage|
|----------|-------|------------|
|High|1|8%|
|Medium|7|54%|
|Low|5|38%|

**Total Active Items:** 13

### By Component

|Component|Count|
|----------------------|-------|
|Core Model Architecture|2|
|GPU / Inference / Training|1|
|GPU / Inference / Performance|1|
|Tooling / Toolchain|1|
|Training / Data Generation|1|
|GUI / Testing|1|
|Advanced Features / Integration|1|
|Training / Metrics / Core|1|
|Android / CI|1|
|Android / Testing|1|
|Documentation|1|
|Training / Deployment / Tooling|1|

### Effort Distribution

|Effort Range|Count|
|--------------|-------|
|0-2 hours|0|
|2-4 hours|1|
|4-8 hours|3|
|8+ hours|6|
|Not estimated|3|

**Total Estimated Effort (Active Items):** 120-178 hours (excludes TD-014, TD-039, and TD-171, which have no effort estimate. The entire TD-174 through TD-186 LeJEPA world-model batch is now resolved — see Tier 10 in the Recommended Execution Order above — so it no longer contributes to this total at all.)

### Future Enhancements Summary

**Total Future Enhancements:** 19
**Estimated Total Effort:** 100+ hours

By Category:

- Configuration and Service Management: 4 items (17-25 hours)
- Logging and Observability: 3 items (12-16 hours)
- Container and Deployment: 5 items (26-36 hours)
- Performance Optimizations: 4 items (38-66 hours)
- Code Quality: 1 item (4-6 hours)
- Developer Experience: 2 items (8-12 hours)

By Priority:

- High: 0 items
- Medium: 4 items (Batched Inference, Attention Head, Metrics Endpoint, Model State Persistence)
- Low: 15 items

Recently Completed:

- TD-020: Persistent Metrics Storage via SQL Database - September 7, 2026 (tracker correction; implementation predates this entry)
- TD-027: Install Script for incremental_trainer Sub-System - June 7, 2026
- TD-028: Separate Dataset Management from IncrementalTrainer - June 7, 2026
- TD-023: Parallel Generation Quality Scoring via Model Snapshot - June 4, 2026
- TD-022: Remove Direct Terminal Output from IncrementalTrainer and Dependencies - June 2, 2026
- TD-021: IncrementalTrainer × Metrics Service Decoupling - May 31, 2026
- TD-003: GPU Memory Management Optimization (GPUMatrix) - May 3, 2026
- TD-016: BLEU/ROUGE Generation Quality Scoring - April 11, 2026
- TD-013b: Batch Padding Efficiency Tracking - April 11, 2026
- TD-013 (partial): FeedForward activation saturation hooks - April 11, 2026
- TD-007: Matrix Operations SIMD Acceleration - April 11, 2026
- TD-015: Validation Metrics Integration - March 14, 2026
- TD-013: Advanced Training Metrics and Outlier Detection - March 14, 2026
- TD-009: Incremental Trainer Dashboard and Structured Logging - March 2, 2026
- TD-004: Enhanced Metrics Tracking (absorbed by TD-009) - March 2, 2026
- TD-010: Configuration Hot-Reloading - March 1, 2026
- TD-011: File Rotation and Management - March 1, 2026
- TD-012: Increase Test Coverage - March 1, 2026
- TD-008: Daemon Service Implementation - March 1, 2026
- TD-005: Checkpoint Management - February 18, 2026
- TD-002: BPE Tokenizer Error Handling - February 18, 2026
- TD-001: Optimizer Parameter Exposure - January 28, 2026

### Deferred Decisions Summary

**Total Deferred Decisions:** 3

- AMD Radeon / ROCm-HIP GPU Backend — Not Pursued (September 7, 2026)
- ThreadSanitizer "Data Races" in Matrix.cpp's OpenMP-Parallelized Code — Confirmed Tool Limitation, Not a Bug (September 12, 2026)
- AttentionHeadBenchmark's Apparent Hang — Confirmed Host CPU Contention, Not a Bug (September 13, 2026)

---

## References

- [Process Improvement Plan](PROCESS_IMPROVEMENT_PLAN.md) - Section 10: Technical Debt Items
- [Contributing Guide](docs/guides/contributing.md) - Code quality standards
- [GitHub Issues](https://github.com/yourusername/adai/issues?q=is%3Aissue+label%3Atechnical-debt) - Active debt tracking

---

**Maintenance Note:** This document should be reviewed monthly and updated as items are added or resolved.
