# Technical Debt Tracker

This document tracks the open technical debt items, TODOs, and improvement opportunities in the ADAI codebase, by priority. Resolved items
are in [TECHNICAL_DEBT_RESOLVED.md](../archive/TECHNICAL_DEBT_RESOLVED.md).

## Overview

**Last Updated:** November 22, 2026
**Active Items:** 81 (7 high, 17 medium, 57 low)
**Future Enhancements:** 16 open (19 listed: 1 withdrawn, 1 done, 1 moot)
**Resolved Items:** 509 ([archive](../archive/TECHNICAL_DEBT_RESOLVED.md))
**Deferred Decisions:** 3

**How items move (November 14, 2026).** An item stays here while any of its action items is open, a fix that has landed in code
included ("Open — code fix landed", with a run on ai-machine, a retraining or a measurement still to do). Once every action item is
done, the item moves to [the archive](../archive/TECHNICAL_DEBT_RESOLVED.md), newest first, with its resolution date. On November 14,
2026, 253 items whose fixes had landed with nothing left to do moved there, together with the dated notes in this overview about
completed work and the old execution order, which had become a log of finished runs. Both are kept at the end of the archive.

**Where things stand.** Of the 86 open items:
- **24 wait on something outside a coding session:** 11 on a run on ai-machine (TD-575, the deployed trainer's startup abort, comes first;
  most of the rest are measurements of fixes that landed) or, for TD-384, a CUDA host; 8 on the
  retraining with a rebuilt, cased vocabulary (TD-358, TD-059, TD-538, TD-563 and the steps before and after it); and 5 on Android
  releases and devices, a deployment cutover and a walk-through against a live server.
- **62 are code work** that can be done now, most of them LOW cleanups found by the code-traced reference docs. 8 of those are
  deferred with RLHF or LoRA, which have no production entry point, and a few wait on an owner decision.

The order to take them in is in [Recommended Execution Order](#recommended-execution-order). The dated notes below are about items
that are still open; the earlier ones are in the archive's
[archived overview notes](../archive/TECHNICAL_DEBT_RESOLVED.md#archived-overview-notes).

**November 26, 2026:** Filed [TD-596](#td-596-the-gpu-memory-pool-grows-until-it-fills-the-budget) (MEDIUM), with its code fix: the SYCL memory pool kept
freed blocks of every size until it filled 90% of the card, so ai-machine's world-model trainer held 16 GB for a few GB of real use.
It now reuses blocks up to a quarter larger and caps its cache (`GPU_POOL_CACHE_MB`, 1 GB), and the trainers log their GPU memory
every epoch. Filed [TD-597](#td-597-a-gpu-build-cant-train-a-chatbot-without-a-gpu) (MEDIUM) while testing it: a GPU build's chatbot training fails
outright when no GPU is initialized.

**November 26, 2026:** Filed [TD-595](#td-595-gpu-attention-training-copies-every-heads-rows-one-at-a-time) (MEDIUM), with its code fix: the world-model pass ran
at only 0.56 samples per second on the GPU because the GPU attention's per-head split copied every row of every head separately, 88%
of all GPU operations of a training step. Each head now multiplies its columns in place, 7.5 times fewer operations per text in a
16-layer trace; ai-machine's measurement remains.

**November 26, 2026:** Filed [TD-591](#td-591-a-download-the-registry-cant-resume-fails-every-training-pass) (MEDIUM): on ai-machine the LeJEPA trainer
failed 50 passes in a row on a half-downloaded 2 GB TinyStories file, because the trainer kept resuming it with FTP's `REST`, which
`FtpDataServer` doesn't implement. The trainer now downloads the whole file again when the server refuses to resume. Split out the
same day: [TD-592](#td-592-ftpdataserver-doesnt-implement-rest-so-downloads-cant-resume) (LOW), `REST` in the server, and [TD-593](#td-593-one-failed-download-fails-the-whole-training-pass) (LOW), training the
files that arrived when one download fails. Filed [TD-594](#td-594-a-download-that-outlasts-its-ftp-token-cant-be-retried) (MEDIUM) the same day: what started it was a 2 GB
download broken 33 minutes in by an automatic `registry_server` restart, and its retry being refused because the file's 30-minute FTP
token had expired and the restarted server no longer knew it.

**November 23, 2026:** Closed [TD-590](../archive/TECHNICAL_DEBT_RESOLVED.md#td-590-nothing-checks-that-gpu-training-stays-on-the-device)
(MEDIUM): host↔device copies are counted and held to budget by tests, and on ai-machine's Arc, on Level Zero, device-resident LeJEPA
passes moved exactly that budget (the token IDs once, then one small download per text and per batch, no host round trip) with no
engine reset in 739 passes. The first run had gone through OpenCL (`libumf.so.1` moved by an apt upgrade); the GPU test package now
checks the Level Zero adapters and fails a run that isn't on Level Zero.

**November 23, 2026:** Closed [TD-395](../archive/TECHNICAL_DEBT_RESOLVED.md#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) (LOW): `GPUMatrix` is one
definition for both backends, after the backend split, and its SYCL `copy()` no longer drains the queue.

**November 22, 2026:** Closed [TD-589](../archive/TECHNICAL_DEBT_RESOLVED.md#td-589-the-lejepa-pass-re-tokenizes-every-epoch-and-reads-diagnostics-on-the-host)
(MEDIUM): a LeJEPA pass tokenizes each text once (its IDs uploaded once on the device), and on the device its activation-saturation and
attention-entropy diagnostics accumulate there, read once per epoch, instead of reading -1.

**November 22, 2026:** Closed [TD-588](../archive/TECHNICAL_DEBT_RESOLVED.md#td-588-lejepa-has-no-device-resident-training-step)
(MEDIUM): with the GPU in use, `--objective=lejepa` now trains on the device (`LeJEPAEncoder::gpu_begin_training()`/`gpu_train_batch()`,
a `GPUOptimizer` with every group on the device, one stats download per batch, a sync before each checkpoint), checked against host
training on the oneAPI CPU device. Its activation-saturation and attention-entropy diagnostics read -1 there until TD-589.

**November 22, 2026:** Closed [TD-587](../archive/TECHNICAL_DEBT_RESOLVED.md#td-587-lejepas-view-loss-and-pooling-arithmetic-is-host-loops)
(MEDIUM): kernels for LeJEPA's context view, masked loss and cosine (into device floats), mean pooling and the pooled gradient's
broadcast, in both backends, checked against CPU references.

**November 22, 2026:** Closed [TD-586](../archive/TECHNICAL_DEBT_RESOLVED.md#td-586-sigreg-runs-only-on-the-host)
(MEDIUM): SIGReg's directions are a pure function of (seed, draw), built where they're used, and `SIGReg::gpu_compute()` computes the
loss, variance stats and gradient on the device from one projection, leaving them there. This changes SIGReg's random stream, so earlier
`SEED` runs don't replay exactly.

**November 22, 2026:** Closed [TD-585](../archive/TECHNICAL_DEBT_RESOLVED.md#td-585-lejepas-predictor-has-no-gpu-path)
(MEDIUM): `Predictor` has a device path (forward, backward, device weights and gradients for a `GPUOptimizer`), checked against its host
path on the oneAPI CPU device. LeJEPA uses it from TD-588.

**November 22, 2026:** Filed [TD-585](../archive/TECHNICAL_DEBT_RESOLVED.md#td-585-lejepas-predictor-has-no-gpu-path) to [TD-590](../archive/TECHNICAL_DEBT_RESOLVED.md#td-590-nothing-checks-that-gpu-training-stays-on-the-device) (MEDIUM), at the owner's request, to move LeJEPA
(`--objective=lejepa`) training fully onto the GPU. It trains on the host today: only `Matrix::multiply()`'s implicit dispatch reaches the
device, one product at a time with an upload and a download, because device-resident training was only ever built for the chatbot
trainer and TD-342 put only the frozen world model's inference on the device. The six items take the predictor, SIGReg (directions
drawn on the device), the loss and pooling arithmetic, the training step, the pass's tokenization and diagnostics onto the device, and
add a transfer counter so a test can hold GPU training to the owner's budget.

**November 22, 2026:** Closed [TD-394](../archive/TECHNICAL_DEBT_RESOLVED.md#td-394-gpu-kernel-tests-cant-run-without-a-gpu)
(MEDIUM) and [TD-393](../archive/TECHNICAL_DEBT_RESOLVED.md#td-393-sycl-sub-group-reduction-is-wrong-when-the-sub-group-size-is-8)
(MEDIUM): `gpuKernelReferenceTests` checks every `matrix_*_gpu` kernel against a CPU reference, TD-393's reduction sizes included, and
passes on the oneAPI CPU device; no CI job (owner decision). It found `matrix_batch_multiply_gpu` computes nothing in the SYCL backend
(added to [TD-386](#td-386-gpu-kernel-input-checks-and-dead-batch-functions), which owns that dead function).

Closed [TD-581](../archive/TECHNICAL_DEBT_RESOLVED.md#td-581-mnss-legacy-jsonl-import-drops-every-model)
(LOW): `mns_server`'s one-time import of a pre-SQLite `models.jsonl` saves each model through `persist_model()`, which now names its
columns; roles are imported only for imported models, and every skipped record is logged. With it, [TD-464](#td-464-mns_server-startup-crash-early-sigterm-hang-and-fixed-log-level)'s
early-SIGTERM hang is fixed (the same lost `stop()` as TD-582, which hung two MNS test suites under load); its crash and `LOG_LEVEL`
items remain.

**November 21, 2026:** Closed [TD-583](../archive/TECHNICAL_DEBT_RESOLVED.md#td-583-the-registry-lock-race-is-tested-only-across-threads)
(LOW): the registry lock race is also tested across a forked process, and fails with the lock scope TD-498 replaced. Closed [TD-582](../archive/TECHNICAL_DEBT_RESOLVED.md#td-582-sighuptests-fails-under-a-loaded-parallel-ctest)
(LOW): the failure was a hang. A pass's admin API lost a `stop()` that came before httplib was listening, so a pass that failed at once
never exited; `TrainerAdminAPI` and `TrainerServiceProxy` now use TD-528's start/stop handshake. Closed [TD-556](../archive/TECHNICAL_DEBT_RESOLVED.md#td-556-admin-api-test-binaries-abort-under-a-parallel-ctest)
(LOW): `registry_ftp_confinement_test` takes kernel-assigned ports and starts `registry_server` again with new ones when another process
took one first. Closed [TD-332](../archive/TECHNICAL_DEBT_RESOLVED.md#td-332-operator-guide-covers-only-8-of-15-dataset_manager-commands)
(LOW): the `dataset_manager` guide covers every command and how pools, `--kind` and the run group are chosen. Closed [TD-331](../archive/TECHNICAL_DEBT_RESOLVED.md#td-331-dataset_manager-has-no-tests-for-its-commands)
(LOW): `datasetManagerCliTests` runs the real `dataset_manager` locally and against a `registry_server` it starts (`segment`,
`assign --count`, `delete`, `remove`, `migrate` with segments). Closed [TD-472](../archive/TECHNICAL_DEBT_RESOLVED.md#td-472-parquetreader-contract-memory-use-and-test-coverage-gaps)
(LOW): every Parquet fixture is compared with pyarrow's reading on every build, which found that doubles were written to 6 significant
digits and NaN as invalid JSON; fixed. Closed [TD-340](../archive/TECHNICAL_DEBT_RESOLVED.md#td-340-datatransport-robustness-and-header-hygiene)
(LOW): FTP downloads initialize libcurl once at startup, keep a failed transfer's partial file for the next pass, retry a refused
connection, and `DataTransport.hpp` no longer leaks `using` declarations. Closed [TD-335](../archive/TECHNICAL_DEBT_RESOLVED.md#td-335-datasetregistry-dead-code-and-unused-integrity-data)
(LOW): `DatasetRegistry`'s dead helpers are gone, `MAX_FILES_PER_RUN` limits what a pass claims, acquired claims show in memory, and a
trained file that changed is reported. Closed [TD-513](../archive/TECHNICAL_DEBT_RESOLVED.md#td-513-startup_sweep-basename-matching-and-unchecked-filesystem-calls)
(LOW): `startup_sweep()` no longer throws on filesystem errors, releases every segment this run claimed on an empty download, and leaves
another run's. Filed [TD-582](../archive/TECHNICAL_DEBT_RESOLVED.md#td-582-sighuptests-fails-under-a-loaded-parallel-ctest) (LOW, `SighupTests` failing under a loaded parallel run),
[TD-583](../archive/TECHNICAL_DEBT_RESOLVED.md#td-583-the-registry-lock-race-is-tested-only-across-threads) (LOW, the registry lock race tested only across threads) and [TD-584](../archive/TECHNICAL_DEBT_RESOLVED.md#td-584-commit-1d4a87afs-message-misstates-the-td-498-race-tests-results) (LOW, `1d4a87af`'s
commit message misstating that test's results; closed the same day: the owner left the message as is, with the TD as the correction).
Closed [TD-498](../archive/TECHNICAL_DEBT_RESOLVED.md#td-498-localtransportremotetransport-divergence-dead-model_id-and-client-hygiene)
(LOW): `LocalTransport` commits as `registry_server` does and keeps the registry under its lock (a race test lost commits before),
remote assign/unassign read segments, `https://` and bad group names are refused, and uploads stream; `model_id` moved to TD-462. Closed [TD-540](../archive/TECHNICAL_DEBT_RESOLVED.md#td-540-incremental_trainer-reset-re-queues-trained-files-locally-not-on-the-registry-server) (MEDIUM): the ops dashboard's
Registry tab can re-queue trained files (one, or all, optionally assigned to a model); the deployment re-queue isn't needed, since
`bb01` has no trained files. Filed [TD-581](../archive/TECHNICAL_DEBT_RESOLVED.md#td-581-mnss-legacy-jsonl-import-drops-every-model) (LOW): `mns_server`'s import of a pre-SQLite data directory
drops every model, since TD-196 added columns its positional insert doesn't supply, and still imports the roles; verified with a real
server. Closed [TD-269](../archive/TECHNICAL_DEBT_RESOLVED.md#td-269-optimizer-state-and-lr-schedule-restart-every-incremental-training-pass)
(MEDIUM): MNS now shows the run's optimizer step (`progress.global_step`, and each pass's history entry). Found on the way: every
trainer-written history entry had stored 0 epochs and loss 0, because MNS didn't read the summary's numbers sent as strings; fixed.
Filed [TD-580](../archive/TECHNICAL_DEBT_RESOLVED.md#td-580-the-ops-dashboards-registry-tab-lists-only-the-groups-typed-into-settings) (LOW): the ops
dashboard's Registry tab listed only the group names typed into Settings, so `bb01` didn't show on the tablet. Fixed the same day: it
also lists the `run_group` of every non-retired model in MNS. The same day the Wear OS modules were removed and `opsdashboard`'s
instrumented tests ran on an emulator for the first time (TD-048, TD-573).

**November 19, 2026:** Filed [TD-579](../archive/TECHNICAL_DEBT_RESOLVED.md#td-579-gutenberg-downloads-dont-fall-back-to-a-mirror-when-gutenbergorg-is-unavailable) (MEDIUM): `www.gutenberg.org` answered 503 for days, so
`bodacious-badger`'s world-model books (run group `bb01`) couldn't be queued, while the official mirrors served the same files.
Fixed the same day: Gutenberg downloads fall back to the mirrors in `GUTENBERG_MIRRORS`.

**November 18, 2026:** Filed [TD-575](#td-575-deployed-incremental_trainer-aborts-at-startup-in-bpetokenizerload_vocab) (MEDIUM, the deployed trainer's startup abort in `load_vocab`) and
[TD-576](../archive/TECHNICAL_DEBT_RESOLVED.md#td-576-feedforwardgradienttestclipgradients-uses-an-absolute-tolerance) (LOW, a test tolerance, fixed the same day), both split from TD-574 when it closed.
Filed and fixed [TD-577](../archive/TECHNICAL_DEBT_RESOLVED.md#td-577-bpe-merge-learning-recounts-the-whole-corpus-every-round)
(MEDIUM): BPE merge learning recounted the whole corpus every round, so `bodacious-badger`'s vocabulary build would have taken days; it now
takes seconds and learns the same merges. Filed and fixed [TD-578](../archive/TECHNICAL_DEBT_RESOLVED.md#td-578-mainline_kernelsh-rebuilds-change-the-packages-root-mtime-failing-its-test-under-load)
(LOW): `package_sycl_gpu_tests_test` failed once in the full parallel `ctest` run that checked TD-577, because `mainline_kernel.sh`'s package
rebuild restamped the package's root directory.

**November 12, 2026:** Filed [TD-573](#td-573-what-the-android-apps-need-to-reach-stable) (LOW): no Android file is `stable`, and
TD-047/TD-048 cap them without saying what `stable` would take. It sets that bar for `app` and `opsdashboard` (a first release, the
`opsdashboard` instrumented tests actually run, each file covered, a decision on documentation) and lists what each `experimental`
file needs. The wear modules are set aside as optional future work.

**October 24, 2026:** Filed [TD-554](#td-554-no-aggregate-version-per-cmake-build) at the owner's request (LOW): derive an aggregate version
for each CMake build (preset × executable target) from the per-file versions of the sources it compiles. Builds on TD-553.

**October 24, 2026:** Filed [TD-553](#td-553-per-file-versions-have-no-build-number) at the owner's request (LOW): add a build number to the
per-file `@adai-version` scheme, following SemVer 2.0.0 for what resets on minor and major updates.

**October 19, 2026:** Filed [TD-547](#td-547-gpu-training-does-the-token-embeddings-on-the-host) (MEDIUM), the part of the owner's
"GPU does the training math" rule that TD-468's device-resident optimizer leaves open: the token embeddings still run on the host.
**Now:** fixed in code October 19, 2026 (a device table with gather and scatter-add kernels); a measurement on ai-machine remains.

**October 18, 2026:** Filed [TD-540](../archive/TECHNICAL_DEBT_RESOLVED.md#td-540-incremental_trainer-reset-re-queues-trained-files-locally-not-on-the-registry-server) (MEDIUM) and fixed it in the same change, found while applying the owner's rule that the trainer
always takes its data and configuration from the remote services first. `incremental_trainer reset --keep-data` re-queued trained files by
editing a local `data_registry.txt`, so against `registry_server` it re-queued nothing. A new server-side re-queue
(`POST /registry/<group>[/<kind>]/requeue`, `dataset_manager requeue`, `RegistryTransport::requeue_trained()`) replaces it.
**Now:** the deployment's re-queue before the retraining and a re-queue in the Android ops dashboard remain.

**October 17, 2026:** Filed [TD-478](../archive/TECHNICAL_DEBT_RESOLVED.md#td-478-positionalencoding-width-check-warning-spam-and-duplicated-formula) (LOW; fixed November 18, 2026) from the code-traced [PositionalEncoding.md](../reference/source/PositionalEncoding.md),
which verified the encoding (within 3.1×10⁻⁵ of a double reference) and the decoder's two inline copies (bit-identical): `forward()` doesn't check
input width (770-wide threw `out_of_range`, 100-wide was silently partial, verified), the over-length warning goes to `std::cerr` on every call, and
the formula exists in three places.

**October 17, 2026:** Filed [TD-481](#td-481-ppo-advantages-are-normalized-within-each-response) and [TD-482](#td-482-ppooptimizer-never-updates-the-policy-and-its-ratio-is-always-1) (both LOW; RLHF has no production entry point, and the owner
will plan its use later) from the code-traced [PPOOptimizer.md](../reference/source/PPOOptimizer.md). Verified: advantages are normalized within each
single response, so a reward of +100 gives the same update as +1, the early tokens of a well-rewarded response are pushed down, and one-token responses
learn nothing (TD-481); `PPOOptimizer::update()` never updates the policy, and `RLHFTrainer` feeds it the rollout's own log-probs so the ratio is always
1 (no clipping, KL ≡ 0, reported loss 0.000000); the real update is an unclipped policy gradient (TD-482).

**October 17, 2026:** Filed [TD-499](#td-499-rewardmodel-unchecked-loading-invalid-widths-and-non-he-initialization) (LOW; deferred with RLHF, which has no production entry point) from the code-traced
[RewardModel.md](../reference/source/RewardModel.md). The backward pass was verified correct against finite differences and the model learns a toy
preference task. Verified gaps: a half-truncated file loads and the next prediction throws a dimension error; `load()` silently adopts the file's
architecture; `{0, 1}` layer widths give a model whose reward is always 0; initialization has a third of He's variance and uses unseeded `rand()`.

**October 17, 2026:** Filed [TD-477](#td-477-timestamp-parsing-turns-bad-input-into-1899-ignores-offsets-and-misreads-fractions) (LOW) from the code-traced [PortableTime.md](../reference/source/PortableTime.md),
verified with the callers' exact parsing logic: a date-only or malformed timestamp becomes `1899-12-31` instead of an error (so `db-history`'s
`from`/`to` silently return everything or nothing), UTC offsets are ignored, out-of-range fields roll over, and all three callers parse fractions
with `atoi()` (`.12` → 12 ms, microseconds → +2 minutes), so PostgreSQL timestamps read back up to ~0.9 s early.

**October 15, 2026:** Filed [TD-473](#td-473-profiler-keeps-every-timing-forever-sorts-under-its-lock-and-uses-the-wall-clock) (LOW) from the code-traced
[PerformanceProfiler.md](../reference/source/PerformanceProfiler.md): `ChatbotAPI` times every response whether or not profiling is enabled,
and `ProfileStats` keeps every sample forever (16 MB after 2M requests, verified); `get_stats()` sorts a full copy under the profiler's
mutex (72 ms at 2M, plus 107 ms for the percentiles), stalling in-flight requests' timers; and `Timer` uses `high_resolution_clock`, which is
`system_clock` here, so clock steps produce wrong or negative durations.

**October 14, 2026:** Filed [TD-432](#td-432-loraadapterload-silently-adopts-the-files-shape) and [TD-433](#td-433-lora-initialization-update-and-config-gaps) (both LOW) from the code-traced
[LoRA.md](../reference/source/LoRA.md), which verified the adapter's gradients (1.7×10⁻⁴ vs. finite
differences). `LoRAAdapter::load()` silently adopts the file's dimensions, rank and α (TD-432, verified);
initialization uses the unseeded global `rand()`, `update()` doesn't zero gradients, and `dropout`/
`apply_to_ffn` are unused (TD-433). `src/LoRA.hpp` was re-tagged `@adai-status: experimental`: nothing
outside tests calls `enable_lora()`, and adapters aren't persisted with checkpoints (TD-302).

**October 14, 2026:** Filed [TD-436](#td-436-sycl-icpx-builds-get-no--march-or-simd-flags) and [TD-437](#td-437-simd-self-check-test-never-compiles-and-no-startup-cpu-check) from the code-traced
[MatrixSIMD.md](../reference/source/MatrixSIMD.md). One is MEDIUM and verified: both CMake checks that add
`-march=native` and `-mavx2 -mfma` match only `"GNU|Clang"`, and oneAPI's `icpx` reports `IntelLLVM`, so the
`sycl` preset (ai-machine's trainer) compiles `Matrix.cpp` with no SIMD at all, a 6–8× slower CPU multiply
(TD-436). One is LOW: the only compile-time vs. runtime SIMD test never compiles (verified), nothing checks
the CPU at startup, and `has_avx2()`/`has_fma()` skip the OS-support check (TD-437).
**Now:** TD-436 was fixed in code October 19, 2026 (`icpx` builds get `-march=x86-64-v3 -mavx2 -mfma`); a before/after
measurement on ai-machine remains. TD-437 is open.

**October 14, 2026:** Filed [TD-447](#td-447-metricstracker-printer-state-leak-convergence-off-by-one-and-smoothing-gaps) (LOW) from the code-traced
[MetricsTracker.md](../reference/source/MetricsTracker.md). Production only uses its CSV export and final-epoch
overfitting warning; the verified gaps are in the rest: `print_summary()`/`print_history()` leave `std::cout` in
fixed 4-decimal mode, `is_converging()` needs `window + 1` epochs, a zero smoothing window gives `NaN`, and the
smoothed validation curve skips epochs without validation.

**October 13, 2026:** Filed [TD-399](#td-399-imetricsreporter-buffering-and-ordering-contract-is-undocumented) and [TD-400](#td-400-nullmetricsreporter-keeps-reporting-only-work-switched-on) (both LOW) from the code-traced
[IMetricsReporter.md](../reference/source/IMetricsReporter.md): the interface doesn't say which methods
`MetricsPushClient` buffers until `end_epoch()` or that they must be called between `start_epoch()`
and `end_epoch()`, and `AbnormalSample` texts travel untruncated (TD-399); and `ChatbotTrainer`
treats a non-null reporter as "metrics enabled" while `incremental_trainer` always passes one
(`NullMetricsReporter` when there's no server), so reporting-only work still runs (TD-400; cost
measured as negligible on a tiny model).

**October 12, 2026:** Filed [TD-384](#td-384-shared-memory-race-in-gpu-softmax-layernorm-and-cross-entropy-kernels-cuda-and-sycl) through [TD-386](#td-386-gpu-kernel-input-checks-and-dead-batch-functions) from the code-traced
[MatrixGPU_cu.md](../reference/source/MatrixGPU_cu.md) (no CUDA device available; all by inspection,
with the training kernels' math checked by hand). One is **HIGH**: softmax, LayerNorm forward and
cross-entropy loss/grad read `scratch[0]` and then reuse the same shared buffer for the next reduction
without a barrier, a data race that can silently corrupt a row (possibly `NaN`) in attention,
LayerNorm and the loss. The SYCL backend that ai-machine trains on has the identical pattern (TD-384).
Two are LOW: `matrix_sum_gpu` allocates and synchronizes per call, every training step (TD-385); and
unchecked kernel inputs plus uncalled batch functions (TD-386).
**Now:** TD-384 was fixed in code October 17, 2026 and verified on the oneAPI CPU device and on ai-machine's Arc; only a CUDA
`racecheck` remains. TD-385's per-call allocation was fixed November 9, 2026; its per-sample loss download stays by design. TD-386 is open.

**October 12, 2026:** Filed [TD-393](../archive/TECHNICAL_DEBT_RESOLVED.md#td-393-sycl-sub-group-reduction-is-wrong-when-the-sub-group-size-is-8) and [TD-394](../archive/TECHNICAL_DEBT_RESOLVED.md#td-394-gpu-kernel-tests-cant-run-without-a-gpu) from the code-traced
[MatrixGPU_SYCL_cpp.md](../reference/source/MatrixGPU_SYCL_cpp.md), the first time any GPU kernel in
this repository has actually been run (a scratch copy of the SYCL backend on the oneAPI CPU device).
Both are MEDIUM: the sub-group sum reduction is correct only when the compiler picks a sub-group size
≥ 16; at 8 (the CPU device, Arc A-series) it drops ¾ of the partials per level, so 256 ones sum to 64
(TD-393, verified; probably latent on ai-machine's Xe2, which has no size 8); and no test can run the
GPU kernels without a GPU, which is how this and TD-384 went unnoticed (TD-394). **TD-384 is now
verified**: the race reproduces deterministically on the CPU device (LayerNorm forward 0.32 relative
error) and adding the barrier fixes all four kernels. TD-385 and TD-386 were amended for SYCL.
**Now:** TD-393 was fixed in code October 17, 2026, and TD-394's opt-in CPU device (`ADAI_SYCL_ALLOW_CPU_DEVICE=1`) landed
October 19, 2026; both closed November 22, 2026, with a kernel-versus-reference suite and no CI job (owner decision).

**October 12, 2026:** Filed [TD-395](../archive/TECHNICAL_DEBT_RESOLVED.md#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) (LOW) from the code-traced
[MatrixGPU_SYCL.md](../reference/source/MatrixGPU_SYCL.md): `GPUMatrix` is maintained twice, once per
backend, although only `copy()`/`zero()` differ, and the SYCL `copy()` drains the queue. Running the
SYCL `GPUMatrix` and the shared `GPUKVCache` on the oneAPI CPU device (their first run anywhere)
verified TD-387 at the cache level and TD-389's SYCL findings; both were amended.
**Now:** TD-387 is fixed in code (a CPU-match test on the Arc remains); TD-395 closed November 23, 2026; TD-389 is open.

## Recommended Execution Order

Rewritten November 14, 2026 for the open items (86 since TD-591 to TD-597 were filed and TD-395, TD-590, TD-589, TD-588, TD-587, TD-586, TD-585, TD-394, TD-393, TD-581, TD-583, TD-582, TD-556, TD-269, TD-540, TD-498, TD-584, TD-513, TD-335, TD-340, TD-472, TD-331, TD-332, TD-411, TD-341, TD-301, TD-050, TD-033, TD-574, TD-392, TD-478, TD-367, TD-425, TD-429, TD-314, TD-579, TD-580 and TD-576 to TD-578 moved and TD-575 to TD-590 were filed), by what each is waiting for and then by area. Every open item is listed; a few
appear twice, where a run and code work are split. The earlier order, a log of the training-fix, data-delivery, trainer-speed, model-features, metrics and generation runs and of the
per-file tiers, is in the archive's [archived execution-order notes](../archive/TECHNICAL_DEBT_RESOLVED.md#archived-execution-order-notes).
Re-derive this order when items move rather than trusting it blindly.

### 1. Runs on ai-machine (the user runs them; packages are built here)

The November 14, 2026 runs (packages `f19c0aee`, `e7b5f0c1` and `e7b5f0c1-dirty`) closed TD-301 and gave TD-574's second to fifth
rounds; the decode benchmark lost the device or hung before any result every time, until the November 17 full run at
`GPU_STRATEGY=normal` closed TD-050 and TD-033. A full `./run_gpu_tests.sh` runs everything here.

- [TD-575](#td-575-deployed-incremental_trainer-aborts-at-startup-in-bpetokenizerload_vocab) (MEDIUM): first. The deployed `incremental_trainer` aborts at startup in `BPETokenizer::load_vocab`;
  read the dump and journal on ai-machine, then fix the cause and make the failure a clean failed pass. TD-574 closed November 18:
  Ubuntu's 7.0 kernel resets nothing, and it is ai-machine's default kernel.
- [TD-596](#td-596-the-gpu-memory-pool-grows-until-it-fills-the-budget) (MEDIUM): check the world-model trainer's GPU memory with the bounded pool (its per-epoch log line and
  qmassa; 16 GB before).
- [TD-595](#td-595-gpu-attention-training-copies-every-heads-rows-one-at-a-time) (MEDIUM): measure the world-model pass's samples per second with the per-head copies gone (0.56 before),
  then `GPU_QUEUE_DEPTH` again.
- [TD-591](#td-591-a-download-the-registry-cant-resume-fails-every-training-pass) (MEDIUM): deploy the trainer-side fix (a refused resume downloads the whole file); the world-model pool
  trains nothing until then, or until the partial TinyStories file is deleted by hand.
- [TD-387](#td-387-gpu-generation-fails-on-input-longer-than-max_seq_length) (MEDIUM): the over-length prompt already runs on the Arc without throwing; a test comparing its output with the CPU path's
  is still to be written.
- Measurements of fixes that landed in code: [TD-436](#td-436-sycl-icpx-builds-get-no--march-or-simd-flags) (MEDIUM, a training pass's wall time with the `icpx` SIMD flags),
  [TD-390](#td-390-sycl-deferred-frees-and-blocking-copies-stall-the-in-order-queue) (MEDIUM, a step's time and `host_task` count; its asynchronous uploads are code work), [TD-547](#td-547-gpu-training-does-the-token-embeddings-on-the-host) (MEDIUM, embeddings on the
  device) and [TD-468](#td-468-gpu-training-round-trips-every-gradient-and-weight-through-a-3-slow-cpu-optimizer) (MEDIUM, the per-step optimizer time; its vectorized loops are code work).
- [TD-220](#td-220-bpe-merge-learning-is-corrupted-by-the--pair-key-separator) (HIGH): run `scripts/audit_vocab_merges.py` on ai-machine's vocabularies before the retraining reuses any of them.

A CUDA host, which this project doesn't have: [TD-384](#td-384-shared-memory-race-in-gpu-softmax-layernorm-and-cross-entropy-kernels-cuda-and-sycl) (HIGH)'s `compute-sanitizer --tool racecheck` over a training step (the fix
itself is verified on the oneAPI CPU device and on the Arc).

### 2. The retraining (operations, in this order)

Every GPU-trained checkpoint predates TD-358, none has the encoder's own embedding (TD-538), and the per-head attention (TD-059) and
cased vocabulary (TD-560) need new weights. So the next real training run starts from scratch, after these steps:

1. Deploy current binaries everywhere first ([TD-563](#td-563-rebuild-vocabularies-and-retrain-with-case-and-deploy-in-order) (MEDIUM): an old `load_vocab()` would misread a new vocabulary).
2. Scan the queued datasets for `\u` escapes and drop tokenized caches built from affected files ([TD-531](#td-531-jsonl-training-reader-doesnt-decode-uxxxx-corrupting-non-ascii-text) (HIGH)).
3. Re-measure the token cost on the registry pool at the chosen vocabulary size ([TD-562](#td-562-vocabulary-sizing-and-tools-assume-lowercase-text) (LOW)).
4. Rebuild the cased vocabulary with the turn tokens and reserved ids ([TD-563](#td-563-rebuild-vocabularies-and-retrain-with-case-and-deploy-in-order)).
5. Have every file pending again: `incremental_trainer retrain` re-queues the trained files itself (TD-411); to restart with `train` or
   `trainer_service` instead, run `dataset_manager requeue --all --model <name>` first ([TD-540](../archive/TECHNICAL_DEBT_RESOLVED.md#td-540-incremental_trainer-reset-re-queues-trained-files-locally-not-on-the-registry-server) (MEDIUM)).
6. Retrain from scratch: [TD-358](#td-358-gpu-training-feeds-unshifted-targets-teaching-the-model-to-copy-instead-of-predict) (HIGH), [TD-538](#td-538-encoder-token-embedding-and-final-norm-are-overwritten-in-every-checkpoint) (HIGH), [TD-059](#td-059-multi-head-and-cross-attention-never-actually-split-into-heads) (HIGH).
7. Afterwards: compare against the lowercase model (TD-563), benchmark generation quality before and after (TD-059), and test RAG
   retrieval with the trained checkpoint ([TD-247](#td-247-rag-retrieval-in-chatbot_api_server-uses-an-untrained-encoder) (HIGH)).

### 3. Other work outside a coding session

- [TD-047](#td-047-android-datarepositoryapi-layer-has-no-ci-or-release-history) (MEDIUM): resolve the `origin` repository's GitHub billing so the Android CI runs, and cut the first release of each app.
- [TD-048](#td-048-android-uidientry-point-classes-are-untested-and-unreleased) (MEDIUM): an instrumented test of the real `BiometricPrompt` (`BiometricAdminAuthGate.kt`), which needs a device.
- [TD-573](#td-573-what-the-android-apps-need-to-reach-stable) (LOW): the Android `stable` bar; its three setup steps are code work, the rest needs the release and a device.
- [TD-172](#td-172-incremental_trainers-serve-command-embeds-the-always-on-service-in-the-same-binary-as-its-cli-commands) (MEDIUM): cut a live host over to `trainer_service` (unit file and binaries).
- [TD-164](#td-164-chatbot-guidemd-needs-a-live-pair-verification-pass) (LOW): walk `chatbot-guide.md` against a live `chatbot_api_server` and `chatbot` pair.

### 4. Code work: training and data

- [TD-597](#td-597-a-gpu-build-cant-train-a-chatbot-without-a-gpu) (MEDIUM): a GPU build's chatbot training fails without a GPU (`GPU_ENABLED=false` or none found).
- [TD-594](#td-594-a-download-that-outlasts-its-ftp-token-cant-be-retried) (MEDIUM): a retry needs a fresh FTP token once the 30-minute one has expired; until then, raise
  `ftp_token_ttl_minutes` on `registry_server`.
- [TD-593](#td-593-one-failed-download-fails-the-whole-training-pass) (LOW): train the files that arrived when one download fails, instead of failing the pass.
- [TD-592](#td-592-ftpdataserver-doesnt-implement-rest-so-downloads-cant-resume) (LOW): `REST` in `FtpDataServer`, so an interrupted download resumes (needs `registry_server` redeployed).

### 5. Code work: GPU

- LeJEPA training on the device (owner request, November 22, 2026), in this order ([TD-585](../archive/TECHNICAL_DEBT_RESOLVED.md#td-585-lejepas-predictor-has-no-gpu-path), the predictor's
  device path, and [TD-586](../archive/TECHNICAL_DEBT_RESOLVED.md#td-586-sigreg-runs-only-on-the-host), SIGReg on the device, and
  [TD-587](../archive/TECHNICAL_DEBT_RESOLVED.md#td-587-lejepas-view-loss-and-pooling-arithmetic-is-host-loops), the view, loss and
  pooling kernels, and [TD-588](../archive/TECHNICAL_DEBT_RESOLVED.md#td-588-lejepa-has-no-device-resident-training-step), the
  device-resident training step, and [TD-589](../archive/TECHNICAL_DEBT_RESOLVED.md#td-589-the-lejepa-pass-re-tokenizes-every-epoch-and-reads-diagnostics-on-the-host),
  tokenizing once and device diagnostics, closed November 22, and [TD-590](../archive/TECHNICAL_DEBT_RESOLVED.md#td-590-nothing-checks-that-gpu-training-stays-on-the-device),
  the transfer budget, closed November 23 after its ai-machine run): done.
- [TD-389](#td-389-gpumatrix-shape-checks-uninitialized-memory-and-untested-gpu-kv-caches) (LOW): `GPUMatrix`'s shape checks and the GPU KV-cache tests, now in one place since [TD-395](../archive/TECHNICAL_DEBT_RESOLVED.md#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) (closed November 23) made `GPUMatrix` one definition.
- [TD-390](#td-390-sycl-deferred-frees-and-blocking-copies-stall-the-in-order-queue) (MEDIUM)'s asynchronous uploads and [TD-468](#td-468-gpu-training-round-trips-every-gradient-and-weight-through-a-3-slow-cpu-optimizer) (MEDIUM)'s vectorized update loops, measured on ai-machine (section 1).
- [TD-368](#td-368-weight-initialization-cant-be-seeded-or-reproduced) (MEDIUM): GPU replays match losses but not the last bits of the weights (float atomics); make the device reductions
  deterministic if exact replay is wanted.
- Smaller: [TD-386](#td-386-gpu-kernel-input-checks-and-dead-batch-functions) (LOW, kernel input checks and dead batch functions), [TD-383](#td-383-gpumanager-lifecycle-gaps-set_device-cleanup-initialize) (LOW,
  `GPUManager` lifecycle), [TD-385](#td-385-matrix_sum_gpu-allocates-and-synchronizes-on-every-call) (LOW, the per-sample loss download, kept by design while the trainer reads each loss).

### 6. Code work: model layers and memory

The remaining hygiene items of the layers, mostly found by the code-traced reference docs. Each is small; take them a file at a time,
as TD-426 and TD-467 were:
- [TD-370](#td-370-feedforward-argument-checks-gpu-null-dereference-and-console-output) (LOW, `FeedForward`), [TD-518](#td-518-tokenembedding-unchecked-loading-full-matrix-updates-and-validation-gaps) (LOW,
  `TokenEmbedding`), [TD-356](#td-356-encoderblock-gradient-norm-and-clipping-ignore-layernorm-header-describes-post-ln) (LOW) and [TD-357](#td-357-encoderblock-learning-rate-handling-unchecked-loads-and-console-output) (LOW, `EncoderBlock`), [TD-348](#td-348-llmdecoder-dead-code-and-robustness-gaps) (LOW, `LLMDecoder`), [TD-345](#td-345-decoderblock-persistence-gaps-for-gated-state) (LOW, `DecoderBlock`
  persistence), [TD-303](#td-303-decoder-only-mode-still-runs-cross-attention-over-a-dummy-input) (LOW, decoder-only mode), [TD-504](#td-504-sigreg-frequency-grid-nan-inputs-and-repeated-projections) (LOW,
  `SIGReg`).
- [TD-435](#td-435-matrix-accessor-cost-sum-accuracy-randomize-race-and-api-hygiene) (LOW, `Matrix`) and [TD-437](#td-437-simd-self-check-test-never-compiles-and-no-startup-cpu-check) (LOW, the SIMD self-check and a startup CPU check).
- [TD-397](#td-397-hippocampal-swap-file-grows-without-bound) (LOW) and [TD-398](#td-398-hippocampalmemory-per-step-copies-and-unchecked-saves) (LOW): the hippocampal swap file's cap and the per-step copies.

### 7. Code work: serving and RAG

- Batched inference v1.x (the honest version before real batching, [real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md)):
  [TD-211](#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model) (MEDIUM), then [TD-213](#td-213-batchedinferenceengine-stats-are-misleading-and-unused-in-production) (LOW), [TD-214](#td-214-batched-inference-mode-generates-differently-from-the-inline-path) (LOW), [TD-217](#td-217-chatbatch-reports-hypothetical-batching-stats-as-if-they-were-real) (LOW) and [TD-218](#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs) (LOW).
- [TD-233](#td-233-chatbotapi-session-ids-come-from-a-non-cryptographic-generator) (MEDIUM): an auth token for session export/import if the server is reachable beyond localhost.
- [TD-263](#td-263-chatbot_gui-freezes-while-generating) (MEDIUM): a Stop control in `chatbot_gui` needs `chatbot_api_server` to cancel a generation first.
- [TD-296](#td-296-conversationcontext-hygiene) (LOW, `ConversationContext` hygiene), [TD-351](#td-351-documentstore-embedding-and-housekeeping-gaps) (LOW, `DocumentStore`), [TD-380](#td-380-generation-quality-scores-arent-comparable-or-well-sampled) (LOW, generation-quality sample size),
  [TD-473](#td-473-profiler-keeps-every-timing-forever-sorts-under-its-lock-and-uses-the-wall-clock) (LOW, `PerformanceProfiler`).

### 8. Code work: metrics, MNS and platform

- [TD-453](#td-453-hand-written-json-duplicated-across-mns-and-metrics-code) (LOW) first: hand-written JSON in the MNS and metrics code, which the next two then use: [TD-527](#td-527-trainingmetricsapi-per-poll-logging-parsing-and-unbounded-history-queries) (LOW, the metrics API's
  route boilerplate) and [TD-459](#td-459-modelnameclient-validation-unescaping-and-slow-failure-gaps) (LOW, `ModelNameClient`).
- [TD-399](#td-399-imetricsreporter-buffering-and-ordering-contract-is-undocumented) (LOW) and [TD-400](#td-400-nullmetricsreporter-keeps-reporting-only-work-switched-on) (LOW): the metrics reporter's contract and a real "disabled" state.
- [TD-447](#td-447-metricstracker-printer-state-leak-convergence-off-by-one-and-smoothing-gaps) (LOW, `MetricsTracker`), [TD-480](#td-480-postgresmetricsdatabase-one-time-schema-bootstrap-reconnects-under-lock-and-unsafe-retries) (LOW, `PostgresMetricsDatabase`), [TD-477](#td-477-timestamp-parsing-turns-bad-input-into-1899-ignores-offsets-and-misreads-fractions) (LOW, timestamp parsing).
- [TD-462](#td-462-modelnameservice-dead-datasets-proxy-admin-settings-race-and-swallowed-persistence-errors) (LOW) and [TD-464](#td-464-mns_server-startup-crash-early-sigterm-hang-and-fixed-log-level) (LOW): `mns_server`'s datasets proxy, settings race and startup gaps.
- FTP server: [TD-376](#td-376-ftpdataserver-slow-shutdown-unencrypted-feat-and-silent-start-failures) (LOW) and [TD-377](#td-377-ftpdataserver-token-semantics-and-setup-gaps) (LOW).

### 9. Code work: clients

- [TD-558](#td-558-android-ops-dashboard-doesnt-show-per-sample-quality-reports) (LOW): per-sample quality reports in the Android ops dashboard.
- [TD-559](#td-559-tizen-dashboard-doesnt-show-per-sample-quality-reports) (LOW): the same on the Tizen TV dashboard.

### 10. Owner decisions, deferred features and process

- [TD-553](#td-553-per-file-versions-have-no-build-number) (LOW) then [TD-554](#td-554-no-aggregate-version-per-cmake-build) (LOW): a build number per file, then a version per CMake build (owner decisions first).
- [TD-039](#td-039-core-trainingmetrics-classes-too-large-and-fast-moving-to-certify-stable) (MEDIUM): freeze and certify the core training and metrics classes once their API stops changing.
- [TD-171](#td-171-no-batch-dimension-anywhere-in-the-model-stack--real-parallel-batched-training-not-supported) (LOW): whether to pursue a batch dimension for training at all (the inference half is the v2 plan).
- [TD-014](#td-014-llm-operations-and-training-tooling-suite) (MEDIUM): the operations tooling suite. Scope its quantization tool together with [TD-038](#td-038-advanced-features-tested-in-isolation-never-wired-into-a-shipped-binary) (LOW)'s deferred
  `Quantization` wiring and [TD-483](#td-483-quantization-asymmetric-range-mode-less-files-and-unpacked-int4) (LOW).
- [TD-466](#td-466-modelserializer-claims-t5-compatibility-it-doesnt-have-and-skips-input-validation) (LOW): the T5-faithful architecture variant (owner decision made, large).
- [TD-006](#td-006-fill-in-the-middle-fim-training-data-generation) (LOW): Fill-in-the-Middle training data.
- Deferred with RLHF, which has no production entry point: [TD-481](#td-481-ppo-advantages-are-normalized-within-each-response) (LOW), [TD-482](#td-482-ppooptimizer-never-updates-the-policy-and-its-ratio-is-always-1) (LOW), [TD-499](#td-499-rewardmodel-unchecked-loading-invalid-widths-and-non-he-initialization) (LOW), [TD-500](#td-500-rlhf-rollouts-never-explore-and-updates-train-on-a-different-sequence) (LOW),
  [TD-501](#td-501-rlhftrainer-lacks-a-kl-penalty-batch-updates-and-efficient-value-states) (LOW). Deferred with LoRA, likewise: [TD-302](#td-302-lora-adapters-arent-persisted-or-counted-in-gradient-norms) (LOW), [TD-432](#td-432-loraadapterload-silently-adopts-the-files-shape) (LOW), [TD-433](#td-433-lora-initialization-update-and-config-gaps) (LOW).

## Table of Contents

- [Overview](#overview)
- [Recommended Execution Order](#recommended-execution-order)
- [Table of Contents](#table-of-contents)
- [Active Technical Debt](#active-technical-debt)
  - [TD-059: Multi-Head and Cross-Attention Never Actually Split Into Heads](#td-059-multi-head-and-cross-attention-never-actually-split-into-heads)
  - [TD-014: LLM Operations and Training Tooling Suite](#td-014-llm-operations-and-training-tooling-suite)
  - [TD-006: Fill-in-the-Middle (FIM) Training Data Generation](#td-006-fill-in-the-middle-fim-training-data-generation)
  - [TD-038: Advanced Features Tested in Isolation, Never Wired Into a Shipped Binary](#td-038-advanced-features-tested-in-isolation-never-wired-into-a-shipped-binary)
  - [TD-039: Core Training/Metrics Classes Too Large and Fast-Moving to Certify Stable](#td-039-core-trainingmetrics-classes-too-large-and-fast-moving-to-certify-stable)
  - [TD-047: Android Data/Repository/API Layer Has No CI or Release History](#td-047-android-datarepositoryapi-layer-has-no-ci-or-release-history)
  - [TD-048: Android UI/DI/Entry-Point Classes Are Untested and Unreleased](#td-048-android-uidientry-point-classes-are-untested-and-unreleased)
  - [TD-164: chatbot-guide.md Needs a Live-Pair Verification Pass](#td-164-chatbot-guidemd-needs-a-live-pair-verification-pass)
  - [TD-171: No Batch Dimension Anywhere in the Model Stack — Real Parallel Batched Training Not Supported](#td-171-no-batch-dimension-anywhere-in-the-model-stack--real-parallel-batched-training-not-supported)
  - [TD-172: incremental_trainer's `serve` Command Embeds the Always-On Service in the Same Binary as Its CLI Commands](#td-172-incremental_trainers-serve-command-embeds-the-always-on-service-in-the-same-binary-as-its-cli-commands)
  - [TD-211: BatchedInferenceEngine Queues and Serializes Requests but Never Batches the Model](#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model)
  - [TD-213: BatchedInferenceEngine Stats Are Misleading and Unused in Production](#td-213-batchedinferenceengine-stats-are-misleading-and-unused-in-production)
  - [TD-214: Batched-Inference Mode Generates Differently From the Inline Path](#td-214-batched-inference-mode-generates-differently-from-the-inline-path)
  - [TD-217: `/chat/batch` Reports Hypothetical Batching Stats as if They Were Real](#td-217-chatbatch-reports-hypothetical-batching-stats-as-if-they-were-real)
  - [TD-218: BatchProcessor Helpers Have Dead Surface, Lost Ordering, and Unchecked Inputs](#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs)
  - [TD-220: BPE Merge Learning Is Corrupted by the `"|||"` Pair-Key Separator](#td-220-bpe-merge-learning-is-corrupted-by-the--pair-key-separator)
  - [TD-233: ChatbotAPI Session IDs Come From a Non-Cryptographic Generator](#td-233-chatbotapi-session-ids-come-from-a-non-cryptographic-generator)
  - [TD-247: RAG Retrieval in `chatbot_api_server` Uses an Untrained Encoder](#td-247-rag-retrieval-in-chatbot_api_server-uses-an-untrained-encoder)
  - [TD-263: `chatbot_gui` Freezes While Generating](#td-263-chatbot_gui-freezes-while-generating)
  - [TD-296: ConversationContext Hygiene](#td-296-conversationcontext-hygiene)
  - [TD-302: LoRA Adapters Aren't Persisted or Counted in Gradient Norms](#td-302-lora-adapters-arent-persisted-or-counted-in-gradient-norms)
  - [TD-303: Decoder-Only Mode Still Runs Cross-Attention over a Dummy Input](#td-303-decoder-only-mode-still-runs-cross-attention-over-a-dummy-input)
  - [TD-345: `DecoderBlock` Persistence Gaps for Gated State](#td-345-decoderblock-persistence-gaps-for-gated-state)
  - [TD-348: `LLMDecoder` Dead Code and Robustness Gaps](#td-348-llmdecoder-dead-code-and-robustness-gaps)
  - [TD-351: `DocumentStore` Embedding and Housekeeping Gaps](#td-351-documentstore-embedding-and-housekeeping-gaps)
  - [TD-356: `EncoderBlock` Gradient Norm and Clipping Ignore `LayerNorm`; Header Describes Post-LN](#td-356-encoderblock-gradient-norm-and-clipping-ignore-layernorm-header-describes-post-ln)
  - [TD-357: `EncoderBlock` Learning-Rate Handling, Unchecked Loads and Console Output](#td-357-encoderblock-learning-rate-handling-unchecked-loads-and-console-output)
  - [TD-358: GPU Training Feeds Unshifted Targets, Teaching the Model to Copy Instead of Predict](#td-358-gpu-training-feeds-unshifted-targets-teaching-the-model-to-copy-instead-of-predict)
  - [TD-368: Weight Initialization Can't Be Seeded or Reproduced](#td-368-weight-initialization-cant-be-seeded-or-reproduced)
  - [TD-370: `FeedForward` Argument Checks, GPU Null Dereference and Console Output](#td-370-feedforward-argument-checks-gpu-null-dereference-and-console-output)
  - [TD-376: `FtpDataServer` Slow Shutdown, Unencrypted `FEAT` and Silent Start Failures](#td-376-ftpdataserver-slow-shutdown-unencrypted-feat-and-silent-start-failures)
  - [TD-377: `FtpDataServer` Token Semantics and Setup Gaps](#td-377-ftpdataserver-token-semantics-and-setup-gaps)
  - [TD-380: Generation-Quality Scores Aren't Comparable or Well-Sampled](#td-380-generation-quality-scores-arent-comparable-or-well-sampled)
  - [TD-383: `GPUManager` Lifecycle Gaps (`set_device`, `cleanup`, `initialize`)](#td-383-gpumanager-lifecycle-gaps-set_device-cleanup-initialize)
  - [TD-384: Shared-Memory Race in GPU Softmax, LayerNorm and Cross-Entropy Kernels (CUDA and SYCL)](#td-384-shared-memory-race-in-gpu-softmax-layernorm-and-cross-entropy-kernels-cuda-and-sycl)
  - [TD-385: `matrix_sum_gpu` Allocates and Synchronizes on Every Call](#td-385-matrix_sum_gpu-allocates-and-synchronizes-on-every-call)
  - [TD-386: GPU Kernel Input Checks and Dead Batch Functions](#td-386-gpu-kernel-input-checks-and-dead-batch-functions)
  - [TD-387: GPU Generation Fails on Input Longer Than `max_seq_length`](#td-387-gpu-generation-fails-on-input-longer-than-max_seq_length)
  - [TD-389: `GPUMatrix` Shape Checks, Uninitialized Memory and Untested GPU KV Caches](#td-389-gpumatrix-shape-checks-uninitialized-memory-and-untested-gpu-kv-caches)
  - [TD-390: SYCL Deferred Frees and Blocking Copies Stall the In-Order Queue](#td-390-sycl-deferred-frees-and-blocking-copies-stall-the-in-order-queue)
  - [TD-397: Hippocampal Swap File Grows Without Bound](#td-397-hippocampal-swap-file-grows-without-bound)
  - [TD-398: `HippocampalMemory` Per-Step Copies and Unchecked Saves](#td-398-hippocampalmemory-per-step-copies-and-unchecked-saves)
  - [TD-399: `IMetricsReporter` Buffering and Ordering Contract Is Undocumented](#td-399-imetricsreporter-buffering-and-ordering-contract-is-undocumented)
  - [TD-400: `NullMetricsReporter` Keeps Reporting-Only Work Switched On](#td-400-nullmetricsreporter-keeps-reporting-only-work-switched-on)
  - [TD-432: `LoRAAdapter::load()` Silently Adopts the File's Shape](#td-432-loraadapterload-silently-adopts-the-files-shape)
  - [TD-433: `LoRA` Initialization, Update and Config Gaps](#td-433-lora-initialization-update-and-config-gaps)
  - [TD-435: `Matrix` Accessor Cost, `sum()` Accuracy, `randomize()` Race and API Hygiene](#td-435-matrix-accessor-cost-sum-accuracy-randomize-race-and-api-hygiene)
  - [TD-436: SYCL (`icpx`) Builds Get No `-march` or SIMD Flags](#td-436-sycl-icpx-builds-get-no--march-or-simd-flags)
  - [TD-437: SIMD Self-Check Test Never Compiles and No Startup CPU Check](#td-437-simd-self-check-test-never-compiles-and-no-startup-cpu-check)
  - [TD-447: `MetricsTracker` Printer State Leak, Convergence Off-by-One and Smoothing Gaps](#td-447-metricstracker-printer-state-leak-convergence-off-by-one-and-smoothing-gaps)
  - [TD-453: Hand-Written JSON Duplicated Across MNS and Metrics Code](#td-453-hand-written-json-duplicated-across-mns-and-metrics-code)
  - [TD-459: `ModelNameClient` Validation, Unescaping and Slow-Failure Gaps](#td-459-modelnameclient-validation-unescaping-and-slow-failure-gaps)
  - [TD-462: `ModelNameService` Dead Datasets Proxy, Admin-Settings Race and Swallowed Persistence Errors](#td-462-modelnameservice-dead-datasets-proxy-admin-settings-race-and-swallowed-persistence-errors)
  - [TD-464: `mns_server` Startup Crash, Early-SIGTERM Hang and Fixed Log Level](#td-464-mns_server-startup-crash-early-sigterm-hang-and-fixed-log-level)
  - [TD-466: `ModelSerializer` Claims T5 Compatibility It Doesn't Have and Skips Input Validation](#td-466-modelserializer-claims-t5-compatibility-it-doesnt-have-and-skips-input-validation)
  - [TD-468: GPU Training Round-Trips Every Gradient and Weight Through a 3× Slow CPU Optimizer](#td-468-gpu-training-round-trips-every-gradient-and-weight-through-a-3-slow-cpu-optimizer)
  - [TD-473: `Profiler` Keeps Every Timing Forever, Sorts Under Its Lock and Uses the Wall Clock](#td-473-profiler-keeps-every-timing-forever-sorts-under-its-lock-and-uses-the-wall-clock)
  - [TD-477: Timestamp Parsing Turns Bad Input Into 1899, Ignores Offsets and Misreads Fractions](#td-477-timestamp-parsing-turns-bad-input-into-1899-ignores-offsets-and-misreads-fractions)
  - [TD-480: `PostgresMetricsDatabase` One-Time Schema Bootstrap, Reconnects Under Lock and Unsafe Retries](#td-480-postgresmetricsdatabase-one-time-schema-bootstrap-reconnects-under-lock-and-unsafe-retries)
  - [TD-481: PPO Advantages Are Normalized Within Each Response](#td-481-ppo-advantages-are-normalized-within-each-response)
  - [TD-482: `PPOOptimizer` Never Updates the Policy and Its Ratio Is Always 1](#td-482-ppooptimizer-never-updates-the-policy-and-its-ratio-is-always-1)
  - [TD-483: `Quantization` Asymmetric Range, Mode-Less Files and Unpacked INT4](#td-483-quantization-asymmetric-range-mode-less-files-and-unpacked-int4)
  - [TD-499: `RewardModel` Unchecked Loading, Invalid Widths and Non-He Initialization](#td-499-rewardmodel-unchecked-loading-invalid-widths-and-non-he-initialization)
  - [TD-500: RLHF Rollouts Never Explore and Updates Train on a Different Sequence](#td-500-rlhf-rollouts-never-explore-and-updates-train-on-a-different-sequence)
  - [TD-501: `RLHFTrainer` Lacks a KL Penalty, Batch Updates and Efficient Value States](#td-501-rlhftrainer-lacks-a-kl-penalty-batch-updates-and-efficient-value-states)
  - [TD-504: `SIGReg` Frequency Grid, NaN Inputs and Repeated Projections](#td-504-sigreg-frequency-grid-nan-inputs-and-repeated-projections)
  - [TD-518: `TokenEmbedding` Unchecked Loading, Full-Matrix Updates and Validation Gaps](#td-518-tokenembedding-unchecked-loading-full-matrix-updates-and-validation-gaps)
  - [TD-527: `TrainingMetricsAPI` Per-Poll Logging, Parsing and Unbounded History Queries](#td-527-trainingmetricsapi-per-poll-logging-parsing-and-unbounded-history-queries)
  - [TD-531: JSONL Training Reader Doesn't Decode `\uXXXX`, Corrupting Non-ASCII Text](#td-531-jsonl-training-reader-doesnt-decode-uxxxx-corrupting-non-ascii-text)
  - [TD-538: Encoder Token Embedding and Final Norm Are Overwritten in Every Checkpoint](#td-538-encoder-token-embedding-and-final-norm-are-overwritten-in-every-checkpoint)
  - [TD-547: GPU Training Does the Token Embeddings on the Host](#td-547-gpu-training-does-the-token-embeddings-on-the-host)
  - [TD-553: Per-File Versions Have No Build Number](#td-553-per-file-versions-have-no-build-number)
  - [TD-554: No Aggregate Version per CMake Build](#td-554-no-aggregate-version-per-cmake-build)
  - [TD-558: Android Ops Dashboard Doesn't Show Per-Sample Quality Reports](#td-558-android-ops-dashboard-doesnt-show-per-sample-quality-reports)
  - [TD-559: Tizen Dashboard Doesn't Show Per-Sample Quality Reports](#td-559-tizen-dashboard-doesnt-show-per-sample-quality-reports)
  - [TD-562: Vocabulary Sizing and Tools Assume Lowercase Text](#td-562-vocabulary-sizing-and-tools-assume-lowercase-text)
  - [TD-563: Rebuild Vocabularies and Retrain With Case, and Deploy in Order](#td-563-rebuild-vocabularies-and-retrain-with-case-and-deploy-in-order)
  - [TD-573: What the Android Apps Need to Reach `stable`](#td-573-what-the-android-apps-need-to-reach-stable)
  - [TD-575: Deployed `incremental_trainer` Aborts at Startup in `BPETokenizer::load_vocab`](#td-575-deployed-incremental_trainer-aborts-at-startup-in-bpetokenizerload_vocab)
  - [TD-591: A Download the Registry Can't Resume Fails Every Training Pass](#td-591-a-download-the-registry-cant-resume-fails-every-training-pass)
  - [TD-592: `FtpDataServer` Doesn't Implement `REST`, So Downloads Can't Resume](#td-592-ftpdataserver-doesnt-implement-rest-so-downloads-cant-resume)
  - [TD-593: One Failed Download Fails the Whole Training Pass](#td-593-one-failed-download-fails-the-whole-training-pass)
  - [TD-594: A Download That Outlasts Its FTP Token Can't Be Retried](#td-594-a-download-that-outlasts-its-ftp-token-cant-be-retried)
  - [TD-595: GPU Attention Training Copies Every Head's Rows One at a Time](#td-595-gpu-attention-training-copies-every-heads-rows-one-at-a-time)
  - [TD-596: The GPU Memory Pool Grows Until It Fills the Budget](#td-596-the-gpu-memory-pool-grows-until-it-fills-the-budget)
  - [TD-597: A GPU Build Can't Train a Chatbot Without a GPU](#td-597-a-gpu-build-cant-train-a-chatbot-without-a-gpu)
- [Resolved Items](#resolved-items) (218 items — see [archive](../archive/TECHNICAL_DEBT_RESOLVED.md); re-derive from the Overview's own Resolved Items count above rather than trusting this number blindly — it has drifted stale before)
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

**Update (October 11, 2026): capabilities removed with `src/Dataset.hpp` (TD-315).** The header was
retired because nothing in production used it (the real pipeline is `ChatbotTrainer` +
`DatasetRegistry` + `TrainingSampleMeta.hpp`'s JSONL helpers, all kept). These pieces have no other
implementation in the codebase and are candidates to recreate in `adai-data-prep`/`adai-eval`, on
top of the registry's JSONL format rather than as a standalone container. The full API and the bugs
to avoid are recorded in [Dataset.md](../reference/source/Dataset.md) and the TD-315–320 archive
entries; the last version is in git history (`git show 6a5438c:src/Dataset.hpp`).

- **Splitting:** seeded random train/validation/test split; length-stratified split (binning by
  total input+target length); k-fold cross-validation (`setup_k_fold()`/`get_fold()`).
- **Transforms:** augmentation hook (`set_augmentation()`/`augment_data()`, k copies per sample),
  preprocessing hook (`set_preprocessing()`/`apply_preprocessing()`), length and substring filters,
  lowercasing.
- **Import/export:** `INPUT:`/`RESPONSE:`, JSONL, JSON, CSV and TSV loading with format detection;
  JSONL/conversation/TSV saving. Recreate with a real JSON/CSV parser (TD-316's bugs).
- **Statistics:** sample counts per split; average and max input/target lengths.
- **Lazy access:** `LazyDataset` (index line offsets, read samples on demand) for files too large to
  load whole.
- **Batch wrappers:** tokenize a split and pad (`get_batch_with_padding()`/`get_target_batch_with_padding()`),
  length-grouped batches, padding statistics. These would only matter once the model has a batch
  dimension (TD-171 / the real-batching v2 plan); `BatchProcessor.hpp` keeps the underlying helpers.

**Update (October 11, 2026): data augmentation removed with `src/EfficientBatching.hpp` (TD-355).**
`EfficientBatching::apply_augmentation()` was the codebase's only training-data augmentation:
per-token dropout (drop with probability p, keeping the original if everything would drop), token
masking (replace with a mask token), and adjacent-token swaps, with an OpenMP-parallel variant. Nothing
in training used it. If recreated (e.g. in `adai-data-prep`), avoid its two bugs: derive each
sequence's RNG from `(seed, sequence index)` so output doesn't depend on thread count (TD-352), and
mask with a real `<mask>` token, not `EOS` (TD-353). Last version: `git show e8f3580:src/EfficientBatching.hpp`.
Its token-budget bucketing is preserved separately in
[real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md) §4.1.

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
  `android/wearsync/src/main` (removed November 21, 2026, with the other wear modules and the CI
  workflows' debug-keystore step, which only `wearface` needed) tagged `beta` — see [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md)
  for the exact list.
- `.github/workflows/android-ci.yml` — done.
- `docs/development/guides/android-release.md`, `.github/workflows/android-release.yml` — done.

---

### TD-048: Android UI/DI/Entry-Point Classes Are Untested and Unreleased

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open (12/12 ViewModels done; Compose UI testing adopted, 12/12 screens covered; admin-dialog confirm-click flows and DI containers/entry points now covered too; the wear modules were removed November 21, 2026, and `opsdashboard`'s 109 instrumented tests now run and pass on an emulator; `BiometricAdminAuthGate` remains) | Android / Testing | September 7, 2026 | 24-32 hours |

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

**Update (November 21, 2026): wear modules removed, `opsdashboard`'s instrumented tests run at last.** At the owner's request
`wearsync`, `wearface`, `wearcomplications` and `opsdashboard`'s `data/wearsync/` package were deleted, with the watch-sync and
watch-face-push settings, the bundled `wearface.apk` and the wear dependencies. Without them the merged manifest no longer requires
`wear-sdk`, so `opsdashboard`'s debug APK installs on the `Medium_Phone_API_35` AVD, and `connectedDebugAndroidTest` ran on a device
for the first time: 109 tests, 67 pass, 42 fail. The failures are in the tests, not the change (none touch removed code): 19
`*ConfirmActionTest`s can't start `ConfirmDialogTestActivity` ("resolved to different process com.adai.ops.test"); others click or
look for nodes below the screen's fold without scrolling (`SettingsScreenTest`'s Save), inside merged nodes (a group chip's Remove
button, the bottom bar's "Models" tab) or assert enabled states that differ from the screen (`ModelDetailScreenTest`,
`TrainerScreenTest`). `app`'s 23 instrumented tests and both modules' unit tests (`app` 27, `opsdashboard` 127) pass. The
`wearcomplications` services this item listed no longer exist.

**Update (November 21, 2026, later): all 109 pass.** Fixed the same day; one real bug among them:
- **App bug:** in `GroupDetailScreen`'s pending-queue row the text column took the whole row, so the Assign and Delete icon buttons
  were measured at zero width and couldn't be tapped on a phone. The text column now has `weight(1f)` (the trained-files row too).
- **Test host:** `ConfirmDialogTestActivity` moved from the androidTest manifest to the app's `debug` source set (fixed the 19
  "resolved to different process" failures; the same way `ui-test-manifest` supplies `ComponentActivity`).
- **Scrolling:** a new androidTest helper (`testutil/ComposeScrolling.kt`) scrolls a screen's LazyColumn to a node before finding it,
  since items below the fold aren't composed; `SettingsScreenTest` scrolls its column before typing and clicking (a click on an
  off-screen node lands outside the window).
- **Stale expectations:** the group list now sums the legacy pool and every kind (TD-205), so its fakes return entries for the legacy
  path only; BLEU prints on the 0-100 scale with two decimals (TD-380); the segments dialog pre-fills N; "Chatbot" and "Checkpoint" match
  several nodes now (dialog-scoped and exact matchers; `AdminActionButton` prefixes its label with a space); the bottom bar's tab is
  found in the unmerged tree.

Run with `./gradlew :opsdashboard:connectedDebugAndroidTest` on the `Medium_Phone_API_35` AVD: 109 of 109 pass, with `app`'s 23
instrumented tests and both modules' unit tests (`app` 27, `opsdashboard` 127).

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
  **Note (November 12, 2026):** `androidx.biometric` 1.1.0 has no test authenticator; TD-573's setup step 3 lists the real options.
- [x] DI containers (`AppContainer.kt`/`AppViewModelProvider.kt` in both apps) and `Activity`/
  `Application` entry points — done, see the "DI containers and entry points" update above
  (`app` module verified live on a real device; `opsdashboard` compile-verified only, per the
  standing `wear-sdk` limitation). The `wearcomplications` services remain entirely untested —
  not attempted in this pass, a distinct area (background services, not UI/DI). **November 21, 2026:** removed with the other
  wear modules; `opsdashboard` now runs on an emulator (see the update above).

Files to Modify:

- ~33 remaining files under `android/app/src/main`, `android/opsdashboard/src/main`, and
  `android/wearcomplications/src/main` still tagged `experimental` — see
  [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md) for the exact, current list.
  **Correction (November 12, 2026):** that list is not all `wearcomplications`. Of 36 `experimental` Android files, 4 are in
  `wearcomplications`, 2 in `opsdashboard`'s `data/wearsync/`, 25 elsewhere in `opsdashboard` and 5 in `app`; TD-573 lists what
  each needs.
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
| LOW | Open — inference half planned as v2 (October 8, 2026), B0 CPU results November 9, 2026, Arc results November 10, 2026; training half not started | Core Model Architecture | September 14, 2026 | Not estimated (large, multi-session architecture project) |

**Update (November 10, 2026):** B0 ran on ai-machine's Arc Pro B60 ([results](../../proposals/real_batching_v2_plan.md#43-phase-b0-results-on-the-arc-pro-b60-november-10-2026)).
Stacking single rows is up to 9.4× faster for the dense multiplies at B = 32, but a whole decode step gains only 1.11-1.16×, because
the per-sequence attention products are 82-88% of a GPU decode step (2-27% on the CPU), and one decode step at B = 1 takes 1.1 ms
on the card against 0.5 ms on the CPU. Stacking alone doesn't pay on the GPU; B1 should wait until attention cost is understood.

**Update (November 9, 2026):** owner chose to benchmark first. Phase B0 ran on the CPU ([results](../../proposals/real_batching_v2_plan.md#42-phase-b0-results-november-9-2026)): a decode step is 3-4.7× faster stacked on 8 threads at B ≥ 4 but only 6-20% on one thread, and sequences of 32 or more rows gain nothing. The GPU decision needs `stacked_gemm_benchmark` run on ai-machine; no model code has changed.

**Update (October 8, 2026):** the inference half now has a design proposal, [real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md). It
recommends **stacking** sequences' rows into one matrix rather than adding a batch axis: `FeedForward`,
`LayerNorm`, `LanguageModelHead` and `TokenEmbedding` are already row-independent and need no
change, so only `PositionalEncoding` (per-sequence offsets) and the two attention classes
(within-sequence blocks) do. That's much smaller than the "entire model stack" scope below, but it
still has to pass a feasibility benchmark (phase B0) first. Batched training remains out of scope.

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

### TD-211: BatchedInferenceEngine Queues and Serializes Requests but Never Batches the Model

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — decided October 8, 2026: real batching planned as v2; v1.x honesty pass outstanding | Inference / Serving | October 8, 2026 | 2-4 hours for the v1.x honesty pass; real batching tracked in [real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md) |

Description:
Found while writing the code-traced reference
[BatchedInferenceEngine.md](../reference/source/BatchedInferenceEngine.md). `src/BatchedInferenceEngine.hpp`
is tagged `stable`. Its file and class comments advertise "continuous batching", "process entire
batch in single model forward pass", "pad & batch", "dynamic batch sizing", and a "10-20x
throughput improvement". None of the model-level batching exists. `process_batch()` loops over the
collected requests and calls `TextGenerator::generate_text()` once per request, sequentially, on a
single worker thread. There is no padding, no combined forward pass, and no length grouping. It
can't be added locally either: the model stack has no batch dimension
([TD-171](#td-171-no-batch-dimension-anywhere-in-the-model-stack--real-parallel-batched-training-not-supported)).

What the engine really is: an asynchronous bounded queue with one worker thread that serializes
generation. In `chatbot_api_server --batched-inference` that means concurrent requests wait their
turn, which raises per-request latency rather than throughput.

Related dead or placeholder surface:

- `BatchedInferenceConfig::use_dynamic_batching` and `enable_request_stats` are never read by the
  engine (only default-value tests reference them). `padding_strategy` was the same until it was
  removed October 11, 2026 with `EfficientBatching.hpp` (TD-355).
- `max_tokens_per_batch` is checked against a fixed 100-tokens-per-request estimate in
  `should_flush_batch()`. With defaults (4096 / 100 = 41) it can never fire before
  `max_batch_size` (32) does.
- `collect_batch()` computes one deadline per call, so an idle worker wakes every `timeout_ms` (20
  times a second by default) just to return an empty group. Negligible CPU, but it's a polling
  loop where a plain wait would do.
- `batched_inference_benchmark` uses a sleep-based synthetic model. Any speedup it reports can't
  come from batching, so the "10-20x" (this file) and "27.80x at batch=32"
  (`IntegratedInferenceEngine.hpp`) claims are unsupported.
- The **Batched Inference Engine (Priority 3)** entry under
  [Future Improvements](#performance-optimizations) still describes the feature as unbuilt
  "single forward pass" work.

Impact: misleading documentation on a `stable` file, and an operator flag
(`--batched-inference`) whose name promises a throughput gain it can't deliver.

Action Items:

- [x] Owner decision (October 8, 2026): keep the class and its name; deliver real batching as
  **v2** (phases B3–B4 of [real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md)), not a re-scope to a plain queue.
- [ ] v1.x honesty pass: rewrite the file/class doc comments and `--batched-inference`'s `--help`
  text to describe today's queue-and-serialize behaviour and point at the v2 plan; document the
  three unused config fields and `max_tokens_per_batch`'s estimate as reserved for v2 (they become
  real inputs in B4) rather than removing them.
- [ ] v1.x: replace the per-call deadline polling with a wait for the first request, then a
  timeout measured from that request's arrival.
- [ ] v1.x: remove or qualify the "10-20x" / "27.80x" throughput claims; v2's acceptance criteria
  replace them with measured numbers.
- [ ] v2 (B4): `process_batch()` drives a real batched decode loop with continuous batching; bump
  the file to `@adai-version 2.0.0` and re-review its `stable` tag.

Location in code: `src/BatchedInferenceEngine.hpp` (file header, `BatchedInferenceConfig`,
`collect_batch()`, `should_flush_batch()`, `process_batch()`); tagged `TODO: See TD-211`.

Files to Modify:

- `src/BatchedInferenceEngine.hpp`
- `src/ChatbotAPIServer.cpp` (`--help` text)
- `benchmarks/BatchedInferenceBenchmark.cpp`
- `tests/batchedinferenceengine_test.cpp`
- `docs/development/reference/source/BatchedInferenceEngine.md`

---

### TD-213: BatchedInferenceEngine Stats Are Misleading and Unused in Production

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Serving / Observability | October 8, 2026 | 2-4 hours |

Description:
Found alongside TD-211:

- `BatchedInferenceStats::avg_latency_ms` is `elapsed × 1000 / total_requests`, which is the
  inverse of throughput including idle time, not a latency. It's also unguarded against
  `total_requests == 0`, so `get_stats()` on a fresh engine reports `+∞`.
- `InferenceRequest::submit_time` is recorded on every request but never read, so real
  queue-wait or end-to-end latency isn't measured anywhere.
- `requests_timeout` / `requests_batch_full` count **groups** flushed, not requests, despite their
  names. Groups drained at shutdown count toward neither.
- `total_tokens_processed` is approximated by re-encoding the decoded output, and
  `throughput_*` include idle time.
- No production code reads `get_stats()`: there's no HTTP endpoint or metrics export, so even
  correct stats would be invisible to an operator.

Action Items:

- [ ] Compute real per-request latency from `submit_time` (queue wait plus generation, ideally
  both), and guard every derived field against a zero denominator.
- [ ] Rename the flush counters (or count requests) so names match semantics.
- [ ] Decide whether to expose the stats (e.g. alongside `GET /admin/profile`) or drop the unused
  surface. Do this after TD-211's owner decision.
- [ ] Update `BatchedInferenceStatsTest.ComputeDerivedStatsAvgLatency`, which currently encodes the
  wall-time ÷ requests formula.

Location in code: `src/BatchedInferenceEngine.hpp` (`BatchedInferenceStats`,
`InferenceRequest::submit_time`, `get_stats()`); tagged `TODO: See TD-213`.

Files to Modify:

- `src/BatchedInferenceEngine.hpp`
- `tests/batchedinferenceengine_test.cpp`
- optionally `src/ChatbotAPI.{hpp,cpp}` / `src/ChatbotAPIServer.cpp` if exposed

---

### TD-214: Batched-Inference Mode Generates Differently From the Inline Path

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Serving | October 8, 2026 | 2-4 hours |

Description:
Found alongside TD-211. With `--batched-inference`, the same request can produce different output
than the default inline path, for two reasons:

- **`strategy` and `beam_width` are ignored.** `ChatbotAPI::generate_response()` maps only
  `max_length`, `temperature`, `top_p`, and `top_k` into the engine's `GenerationConfig`. Decoding
  is always `TextGenerator::generate()`'s combined temperature/top-k/top-p sampling (greedy only at
  temperature 0). This is documented on `enable_batched_inference()`, but it's still a silent
  per-request behaviour change. Mapping `beam_width` to `num_beams` would also need the
  `InferenceRequest::model_fn` beam contract checked; it's safe today because `model_->forward()`
  is full-recompute.
- **One shared, fixed-seed RNG.** The constructor builds a single `TextGenerator` seeded with `0`
  and reuses it for every request. A request's sampled output therefore depends on every request
  generated before it. It's reproducible only for an identical request sequence, and it isn't
  independent per request the way the inline path's fresh generator is.

Action Items:

- [ ] Map `strategy`/`beam_width` (or reject unsupported strategies explicitly) in the batched
  branch of `ChatbotAPI::generate_response()`.
- [ ] Give each request its own RNG seed (e.g. derived per request, or from an optional
  per-request seed) instead of sharing one stream.
- [ ] Add a test that the batched and inline paths agree for greedy decoding on the same input.

Location in code: `src/BatchedInferenceEngine.hpp` (constructor), `src/ChatbotAPI.cpp` (batched
branch of `generate_response()`), `src/ChatbotAPI.hpp` (`enable_batched_inference()` doc); tagged
`TODO: See TD-214`.

Files to Modify:

- `src/BatchedInferenceEngine.hpp`
- `src/ChatbotAPI.{hpp,cpp}`
- `tests/chatbotapi_test.cpp`

---

### TD-217: `/chat/batch` Reports Hypothetical Batching Stats as if They Were Real

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open — decided October 8, 2026: relabel as an estimate | Inference / Serving / API | October 8, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [BatchProcessor.md](../reference/source/BatchProcessor.md).
`ChatbotAPI::generate_batch_responses()` (behind `POST /chat/batch` and, via
`generate_batch_session_responses()`, `POST /chat/batch-session`) calls
`create_dynamic_batches(inputs, 32, 10, PAD)` **only** to compute `compute_batch_stats()`, then
generates every input on its own, unpadded, in its original order. The response's `stats` object
(`total_tokens`, `actual_tokens`, `padding_ratio`, `num_batches`, `avg_batch_size`, `efficiency`)
therefore describes grouping that never happens.

The REST docs present this as real: [rest-api.md](../api/rest-api.md) says `/chat/batch` processes
messages "for higher throughput" with "20-60% reduction in padding overhead", and
[batch-processing.md](../api/batch-processing.md) makes similar claims. No batched forward pass
exists ([TD-171](#td-171-no-batch-dimension-anywhere-in-the-model-stack--real-parallel-batched-training-not-supported)),
so there's no throughput gain or padding to reduce.

Action Items:

- [x] Owner decision (October 8, 2026): keep the figures but **relabel them as an estimate**
  rather than dropping them.
- [ ] Implement the relabel: emit the block as `"padding_estimate"` (same fields), and document in
  `rest-api.md` that it estimates padding for a hypothetical grouping
  (`max_batch_size` 32, `length_tolerance` 10) and is not a measurement of work done. Choose
  whether to keep `"stats"` as a deprecated alias for one release; either way, note the response
  change in the API changelog.
- [ ] Once real batching lands (phase B5 of [real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md)), add measured figures alongside or in
  place of the estimate.
- [ ] Correct the throughput and padding-reduction claims in `rest-api.md` and
  `batch-processing.md`.
- [ ] Update `chatbotapi_test.cpp` for whichever response shape is chosen.

Location in code: `src/ChatbotAPI.hpp` (`BatchResponse::stats`), `src/ChatbotAPI.cpp`
(`generate_batch_responses()`, `create_batch_json_response()`); tagged `TODO: See TD-217`.

Files to Modify:

- `src/ChatbotAPI.{hpp,cpp}`
- `tests/chatbotapi_test.cpp`
- `docs/development/api/rest-api.md`, `docs/development/api/batch-processing.md`

---

### TD-218: BatchProcessor Helpers Have Dead Surface, Lost Ordering, and Unchecked Inputs

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open — decided October 8, 2026: keep as groundwork for v2 | Core / Batching | October 8, 2026 | 2-4 hours (v1.x); v2 tracked in [real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md) |

Description:
Found alongside TD-217. `src/BatchProcessor.hpp` is tagged `stable`, but:

- **Its headline benefit doesn't exist.** The file comment's "Process multiple sequences in one
  forward pass" can't happen here (TD-171). `create_padding_mask()` and `unbatch_outputs()` have
  no production callers. The training-side consumer of this same padding model,
  `TokenBatchLoader`, was retired for exactly this reason (TD-170).
- **Order and identity are lost.** `create_dynamic_batches()` returns length-sorted batches, and
  `TokenBatch` has no original-index field. This already caused a real bug: `/chat/batch` returned
  responses in the wrong order (fixed in `ChatbotAPI` by not using the batches for generation).
  Any future caller is set up to repeat it.
- **Mask shape doesn't match the attention layers.** `create_padding_mask()` returns
  `[batch_size, max_length]` key-validity rows; `MultiHeadAttention`/`CrossAttention` take a
  per-sequence `[q, k]` mask.
- **Unchecked inputs.** `unbatch_outputs()` assumes `batch_outputs.size() >= batch_size()` and
  `rows >= lengths[i]`, and its doc comment describes a 3-D input `Matrix` can't represent.
  `TokenBatch`'s invariants (`lengths.size() == batch_size()`, rows `max_length` long) aren't
  enforced, so a hand-built batch makes the other helpers read out of bounds.
- **`BatchStats::print()` writes to `std::cout`,** against the project's "never `std::cout` in
  library code" rule. Nothing in `src/` calls it.

Action Items:

- [x] Owner decision (October 8, 2026): **keep** the helpers as groundwork for real batching,
  delivered as v2 (phase B1 of [real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md)), rather than retiring them as TD-170 did.
- [ ] v1.x: rewrite the file comment to describe what the helpers do today and point at the v2
  plan; add bounds checks to `unbatch_outputs()` and fix its doc comment.
- [ ] v2 (B1): `TokenBatch` gains `source_indices` and a packed (stacked, `offsets`) form;
  per-sequence `[q, k]` mask builder replaces/augments `create_padding_mask()`; `print()` replaced
  by an `adai::Logger`-based helper; file goes to `@adai-version 2.0.0`, `stable` tag re-reviewed.

Location in code: `src/BatchProcessor.hpp` (file comment, `TokenBatch`,
`create_dynamic_batches()`, `create_padding_mask()`, `unbatch_outputs()`, `BatchStats::print()`);
tagged `TODO: See TD-218`.

Files to Modify:

- `src/BatchProcessor.hpp`
- `tests/batchprocessor_test.cpp`, `tests/inference_optimization_test.cpp`
- `docs/development/reference/source/BatchProcessor.md`

---

### TD-220: BPE Merge Learning Is Corrupted by the `"|||"` Pair-Key Separator

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open — code fix landed October 17, 2026 (vocab audit remains) | NLP / Tokenizer | October 8, 2026 | 1-2 hours (fix) + audit of existing vocab files |

Description:
Found while writing the code-traced reference [BPETokenizer.md](../reference/source/BPETokenizer.md),
then confirmed by running the real code. `BPETokenizer::get_most_frequent_pair()` counts pairs in a
map keyed by `first + "|||" + second`, then recovers the pair by splitting at the **first**
`"|||"`. When a token contains or ends with `|`, the split lands in the wrong place:

- If it yields an empty `first` (e.g. pair `("|", "|")`, key `"|||||"`), `build_bpe_merges()`
  treats that as "no more pairs" and **stops learning merges entirely**. A corpus of
  `"|| || || ab ab"` lines asked for 52 merges and learned 0.
- Otherwise it yields a wrong, non-empty pair. `merge_tokens()` then changes nothing, so the same
  wrong pair wins every following round and is recorded as a merge rule **repeatedly**.

Fuzzing 200 small corpora over `{|, !}`: all 200 stopped short of their target, and 9 saved files
contained repeated merge rules. Markdown tables, `||` in code, and shell pipelines in training data
are realistic triggers.

Impact: a vocabulary is baked into every checkpoint trained on it, so a truncated or degenerate
merge list silently degrades every model built on that vocab until it's rebuilt and the model
retrained.

**Fix landed October 17, 2026:** pair counts are keyed by `std::pair<std::string, std::string>`
(`StringPairHash`), so nothing is split back. Ties go to the lexicographically smallest pair, which
makes vocab builds reproducible; previously the order depended on hash-table iteration.

New tests in `tokenizer_test.cpp` (6 of the 7 new tokenizer tests fail on the old code; 60 pass in
total):
- `GetMostFrequentPairHandlesPipeTokens` and `GetMostFrequentPairBreaksTiesDeterministically`;
- `PipeHeavyCorpusStillLearnsMerges` (the 0-of-52 corpus);
- `RandomPipeCorporaNeverRepeatAMergeRule` (40 random `{|, !}` corpora).

New script `scripts/audit_vocab_merges.py` checks saved vocab files for duplicate rules and, with
`--expected-merges N`, short merge lists; it exits 1 if any file looks damaged. There are no vocab
files on the dev machine, so the audit has to run on ai-machine.

Action Items:

- [x] Key the pair counts by `std::pair<std::string, std::string>` (with a hash) instead of a
  string separator.
- [x] Regression tests: pipe-only corpus learns merges; no repeated merge rules; merge count
  reaches the target when pairs remain.
- [ ] Audit existing vocab files (run `scripts/audit_vocab_merges.py` on ai-machine) (`vocab.txt`, every `<checkpoint>.vocab` on ai-machine): compare
  the `BPE_MERGES` count against the intended target and look for repeated rules. Rebuild and
  retrain any affected vocab.

Location in code: `src/BPETokenizer.cpp` (`get_most_frequent_pair()`); the `TODO: See TD-220` tag is now a fix comment.

Files to Modify:

- `src/BPETokenizer.cpp`
- `tests/tokenizer_test.cpp`

---

### TD-233: ChatbotAPI Session IDs Come From a Non-Cryptographic Generator

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — code fix landed October 27, 2026 | Inference / Serving / API | October 8, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). `create_session_id()` draws 32 hex digits from a `static std::mt19937` seeded once from
`std::random_device`. `mt19937` is predictable once enough output is observed, yet a session ID is
the only credential for a conversation: `POST /chat/session/export` returns the full history to
anyone who presents it. Found by reading the code.

Action Items:

- [x] Generate IDs from a CSPRNG (e.g. `getrandom()` or `/dev/urandom`).
- [ ] Consider requiring an auth token for export/import if the server is reachable beyond localhost.

**Fix (October 27, 2026).** New `src/SecureRandom.{hpp,cpp}` (`adai::secure_random_bytes()`, `secure_random_hex()`) reads the
OS CSPRNG directly: `getrandom()` on Linux, `arc4random_buf()` on macOS/BSD, `BCryptGenRandom()` on Windows (`adai_core` links
`bcrypt` there; cross-compiled with MinGW and run under Wine). It throws rather than fall back to a weaker generator.
`create_session_id()` returns `secure_random_hex(16)`: the same 32 hex digits, now 128 bits from the OS. The same weakness existed in
`FtpDataServer`'s `ftp_detail::random_hex()`, which makes the per-session FTP user and, by default, the password: it used `RAND_bytes()`
only in FTPS builds and otherwise (or on failure) a `std::mt19937`; it now calls `secure_random_hex()` in every build. Tests:
`SecureRandomTests` (length and digits, 10,000 draws without a repeat, byte-frequency chi-square on 1 MiB). Predictability itself
isn't something a unit test can show; the fix is that no `mt19937` remains on either path.

The second item stays open: export/import still need only the session id. Deployments reachable beyond localhost go through
Cloudflare Access, so an API token is an owner decision for direct exposure.

Location in code: `src/ChatbotAPI.cpp` (`create_session_id()`); tagged `TD-233`.

Files to Modify:

- `src/ChatbotAPI.cpp`
- `src/SecureRandom.{hpp,cpp}` (new), `src/FtpDataServer.hpp`

---

### TD-247: RAG Retrieval in `chatbot_api_server` Uses an Untrained Encoder

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open — code fix landed October 26, 2026 | Inference / Serving / Server | October 8, 2026 | 2-4 hours |

Description:
Found alongside TD-246. When `RAG_ENABLED`, `main()` builds a **new** `LLMEncoder` from the model's
architecture and loads only the vocab into it (`load_tokenizer_vocab()`). `LLMEncoder::load_weights()`
is never called, so `DocumentStore::addDocument()`/`retrieve()` embed every document and query with
**randomly initialized weights**. Retrieval ranking is therefore not learned similarity (at best
crude token overlap through random projections), while generation uses the trained model, so
answers look plausible over poorly chosen context. The trained encoder is already available via
`model->get_encoder()` with the same architecture. Found by reading the code.

Action Items:

- [x] Embed with the trained encoder: share `model->get_encoder()` (taking care that RAG encoding and generation don't use it concurrently outside `EncoderDecoderModel`'s locking), or load the checkpoint's encoder weights into the RAG encoder.
- [x] If no trained model is loaded (TD-246), refuse to enable RAG.
- [ ] Test: retrieval ranks a document containing the query's content above unrelated ones with a trained checkpoint. (Not done: needs
  a trained checkpoint. The test below proves the RAG encoder has the trained weights.)

**Fix (October 26, 2026).** `EncoderDecoderModel::copy_encoder()` returns an independent copy of the trained encoder: weights through a
temporary `save_weights()`/`load_weights()`, the model's own tokenizer vocabulary, and `requires_grad` off. RAG uses it. A copy, not
`get_encoder()`: retrieval would otherwise race with generation (the layers cache activations; TD-350).

The old encoder was wrong in a second way too. `LLMEncoder` has its own tokenizer, which the model never fills. The old RAG encoder
loaded `VOCAB_PATH` into it rather than the checkpoint's `.vocab`, so documents could even tokenize differently from the model. With no
trained model loaded (`ALLOW_UNTRAINED_MODEL`, TD-246), RAG is not enabled, and the server logs an error.

Tests:
- `EncoderDecoderModelTest.CopyEncoderEncodesLikeTheModelsEncoder`: the copy encodes exactly like the model's own encoding path (its
  tokenizer, then `encode_with_mask()`), and differently from a fresh encoder.
- `ChatbotApiServerStartupTest.RagIsNotEnabledWithoutATrainedModel`: on the old server RAG initialized over random weights.

Location in code: `src/ChatbotAPIServer.cpp` (RAG setup), `src/EncoderDecoderModel.{hpp,cpp}` (`copy_encoder()`).

Files to Modify:

- `src/ChatbotAPIServer.cpp`
- `src/DocumentStore.{hpp,cpp}` (if ownership changes)
- `tests/rag_*`

---

### TD-263: `chatbot_gui` Freezes While Generating

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — code fix landed November 4, 2026 (with TD-260) | Client / GUI | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-260. `onSendMessage()` calls `generateResponse()` synchronously on the GUI
thread. One `processEvents()` beforehand lets "Generating..." paint, but the event loop is then
blocked for the entire generation (seconds to minutes on CPU): the window can't repaint, scroll or
close, and desktops may flag it as "not responding".

Action Items:

- [x] Run generation on a worker thread (`QtConcurrent::run` + `QFutureWatcher`, or a `QThread`), posting the result back to the GUI thread; disable settings while it runs. Each request runs on a `std::thread`, its result posted back with `QMetaObject::invokeMethod`; input, buttons and settings are disabled until it arrives. Closing the window cancels the request (`ChatbotClient::cancel()`) instead of waiting up to the 300 s read timeout.
- [ ] Add a cancel/stop control if the generator can support it. Still open: `chatbot_api_server` can't stop a generation, so a Stop button would only stop waiting while the server still adds the turn to the session.

Location in code: `src/ChatbotGUI.cpp` (`runRequest()`, `onSendMessage()`).

Files to Modify:

- `src/ChatbotGUI.{hpp,cpp}`
- `src/CMakeLists.txt` (if `Qt::Concurrent` is added)

---

### TD-296: ConversationContext Hygiene

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | NLP / ConversationContext | October 9, 2026 | 1 hour |

Description:
Found alongside TD-291: `save_to_file()` truncates then writes without checking the stream
afterwards (a failed write after opening goes unreported) and isn't atomic; `update_token_count()`
is never called; `ChatbotTrainer.cpp` includes `ConversationContext.hpp` without using it.

Action Items:

- [ ] Check the stream and write via temp file + rename.
- [ ] Remove `update_token_count()` (or use it after imports, alongside TD-291) and the unused include.

Location in code: `src/ConversationContext.{hpp,cpp}`, `src/ChatbotTrainer.cpp`; tagged `TODO: See TD-296`.

Files to Modify:

- `src/ConversationContext.{hpp,cpp}`
- `src/ChatbotTrainer.cpp`
---

### TD-302: LoRA Adapters Aren't Persisted or Counted in Gradient Norms

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Attention / CrossAttention | October 10, 2026 | 2-4 hours |

Description:
Found alongside TD-297, by inspection. Neither `CrossAttention::save()` nor SafeTensors
(`ModelSerializer`) stores LoRA adapters, so unmerged LoRA training is lost on save unless
`merge_lora()` is called first. `get_gradient_norm()` covers only the base weights' gradients, so
gradient clipping and norm reporting ignore adapter gradients, which matters most when only the
adapters are trained. `MultiHeadAttention` has the same shape. LoRA has no production caller yet
(`EncoderDecoderModel::enable_lora()` is never called outside tests), so this is latent.

Action Items:

- [ ] Persist adapters (e.g. extra SafeTensors entries) or document merge-before-save and enforce it.
- [ ] Include active adapters' gradients in `get_gradient_norm()`.

Location in code: `src/CrossAttention.cpp` (`get_gradient_norm()`, `save()`); tagged `TODO: See TD-302`.

Files to Modify:

- `src/CrossAttention.cpp`
- `src/MultiHeadAttention.cpp`
- `src/ModelSerializer.cpp`

---

### TD-303: Decoder-Only Mode Still Runs Cross-Attention over a Dummy Input

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Attention / CrossAttention | October 10, 2026 | 1 hour |

Description:
Found alongside TD-297, by inspection. With no encoder output, `LLMDecoder::forward()`/
`forward_with_cache()` pass a zero `Matrix(1, d_model)` to every block, under a comment saying
"no cross-attention". The cross-attention layer still runs. With `V = 0` its output and gradients are
exactly zero, so it's wasted compute, not a correctness bug.

Action Items:

- [ ] Skip the cross-attention sub-layer in `DecoderBlock` when no encoder output is given (e.g. a null pointer), and fix the comment.

Location in code: `src/Decoder.cpp` (decoder-only branches); tagged `TODO: See TD-303`.

Files to Modify:

- `src/Decoder.cpp`
- `src/DecoderBlock.{hpp,cpp}`
---

### TD-345: `DecoderBlock` Persistence Gaps for Gated State

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Transformer / DecoderBlock | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-341, by inspection. `load()` never checks the stream, so a truncated legacy
block file loads garbage `LayerNorm` values or an unset gate rather than throwing (same class as
TD-298), and it sets `learning_rate` from the file before validating. The main checkpoint format,
SafeTensors via `ModelSerializer`, has no accessors for the gates or the gated cross-attention/norm
sub-layers, so once TD-341 lets them train, their state is lost on save.

Action Items:

- [ ] Check reads in `load()` and validate before assigning; add SafeTensors entries (and accessors) for the gates and gated sub-layers.

**Load/save checks (October 26, 2026, with TD-423).** `DecoderBlock::save()`/`load()` (flags and gates included; `dropout_rate`/`learning_rate` assigned only after validation) reads and writes through `adai::binio` (`src/BinaryIO.{hpp,cpp}`): a short or over-long file throws `std::runtime_error`, the file is read into temporaries before anything changes, and a failed write or flush throws. Covered by `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`. This TD's other items stay open.

Location in code: `src/DecoderBlock.cpp` (`load()`); tagged `TODO: See TD-345`.

Files to Modify:

- `src/DecoderBlock.{hpp,cpp}`
- `src/ModelSerializer.cpp`

---

### TD-348: `LLMDecoder` Dead Code and Robustness Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Transformer / Decoder | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-346, by inspection:
- `set_training()`'s "propagate to all components" loop has an empty body.
- Every forward copies `cached_encoder_output`, which nothing reads.
- `forward_with_mask()` has no production caller.
- `load_weights()` reads the header without checking the stream.
- The constructor validates only the injection knob (not vocab, `d_model`, layers, heads).
- `gpu_backward()` dereferences `*grad_encoder_sum` unconditionally (undefined behaviour with zero
  layers).

Action Items:

- [ ] Remove the empty loop, the unused copy and (if still unused) `forward_with_mask()`; check header reads; validate constructor arguments; guard the zero-layer case.

**Load/save checks (October 26, 2026, with TD-423).** `LLMDecoder::save_weights()`/`load_weights()`'s header reads and writes through `adai::binio` (`src/BinaryIO.{hpp,cpp}`): a short or over-long file throws `std::runtime_error`, the file is read into temporaries before anything changes, and a failed write or flush throws. Covered by `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`. This TD's other items stay open.

Location in code: `src/Decoder.cpp` (`set_training()`, `forward_with_encoder()`, `load_weights()`, `gpu_backward()`); tagged `TODO: See TD-348`.

Files to Modify:

- `src/Decoder.{hpp,cpp}`

---

### TD-351: `DocumentStore` Embedding and Housekeeping Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | RAG / DocumentStore | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-349, by inspection:
- Mean pooling includes the special tokens `encode(text, true)` adds, so every embedding shares
  those rows and cosine scores bunch together (0.89 between unrelated texts with untrained weights).
- `metadata` is stored but never read; there's no filtering API.
- No persistence: every document is re-embedded at each server start.
- A NaN similarity (only reachable with NaN weights, e.g. a TD-298-style corrupt load) violates
  `partial_sort`'s strict weak ordering.

Action Items:

- [ ] Pool over content tokens only; add metadata filtering or drop the field; optionally cache embeddings on disk keyed by file checksum; treat NaN scores as lowest.

**Update (November 12, 2026, while fixing TD-349):** the first item no longer applies. Since TD-364 `LLMEncoder::encode()` adds no
special tokens by default and `DocumentStore` never asks for them, so nothing pools BOS/EOS rows. The metadata, persistence and NaN items
remain (the metadata is now copied to every chunk, still unread).

Location in code: `src/DocumentStore.cpp` (`getSentenceEmbedding()`, `retrieve()`); tagged `TODO: See TD-351`.

Files to Modify:

- `src/DocumentStore.{hpp,cpp}`

---

### TD-356: `EncoderBlock` Gradient Norm and Clipping Ignore `LayerNorm`; Header Describes Post-LN

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Transformer / EncoderBlock | October 11, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [EncoderBlock.md](../reference/source/EncoderBlock.md), by inspection. `get_gradient_norm()`
combines only the attention and feed-forward norms, and `clip_gradients()` scales only those two, so
`LayerNorm` gradients are left out of the reported norm and unscaled when clipping (which changes the
gradient direction). `clip_gradients()` has no caller. The header's `forward()`/`backward()` doc step
lists describe Post-LN order ("residual connection and layer norm") while the class is Pre-LN, and
the feature list advertises "optional dropout" and "gradient accumulation and clipping".
(`backward()` itself was verified correct against finite differences.)

Action Items:

- [ ] Include `LayerNorm` gradients in `get_gradient_norm()`/`clip_gradients()` (add a norm accessor to `LayerNorm`), or remove the unused `clip_gradients()`.
- [ ] Rewrite the stale header comments to the Pre-LN steps and drop the unimplemented features.

Location in code: `src/EncoderBlock.cpp` (`get_gradient_norm()`, `clip_gradients()`), `src/EncoderBlock.hpp` (class/method docs); tagged `TODO: See TD-356`.

Files to Modify:

- `src/EncoderBlock.{hpp,cpp}`
- `src/LayerNorm.{hpp,cpp}`

---

### TD-357: `EncoderBlock` Learning-Rate Handling, Unchecked Loads and Console Output

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Transformer / EncoderBlock | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-356, by inspection. `update_weights()` copies the block's public `learning_rate`
into all four sub-layers on every call, silently overriding anything set on them directly, and there's
no `set_learning_rate()` (`LLMEncoder` has none either, so `EncoderDecoderModel::set_learning_rate()`
never reaches the encoder). `load_weights()` never checks its reads, so a truncated file leaves
garbage `LayerNorm` parameters (same class as TD-298/TD-345), and the saved dropout value is read and
discarded. `save_weights()`, `load_weights()` and `print_config()` write to `std::cout` in library
code.

Action Items:

- [ ] Add `set_learning_rate()` (propagating to sub-layers, like `DecoderBlock`'s) and stop overwriting in `update_weights()`; check reads in `load_weights()`; switch output to `Logger`.

**Load/save checks (October 26, 2026, with TD-423).** `EncoderBlock::save_weights()`/`load_weights()` reads and writes through `adai::binio` (`src/BinaryIO.{hpp,cpp}`): a short or over-long file throws `std::runtime_error`, the file is read into temporaries before anything changes, and a failed write or flush throws. Covered by `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`. This TD's other items stay open.

Location in code: `src/EncoderBlock.cpp` (`update_weights()`, `load_weights()`, `save_weights()`, `print_config()`); tagged `TODO: See TD-357`.

Files to Modify:

- `src/EncoderBlock.{hpp,cpp}`
- `src/LLMEncoder.cpp`

---

### TD-358: GPU Training Feeds Unshifted Targets, Teaching the Model to Copy Instead of Predict

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open — code fix landed October 17, 2026 | Models / EncoderDecoderModel | October 11, 2026 | 1-2 hours (plus retraining every GPU-trained checkpoint) |

Description:
Found while writing the code-traced reference [EncoderDecoderModel.md](../reference/source/EncoderDecoderModel.md). `EncoderDecoderModel::gpu_forward()`
calls `decoder->gpu_decode(target_tokens, …)` with the **unshifted** targets as decoder input, then
`matrix_cross_entropy_loss_gpu()` scores row *t* against `targets[t]`. Under the causal mask,
position *t* sees `targets[t]` itself, so the loss is minimized by copying the current input token.
The CPU `forward()` correctly feeds `[BOS] + target[:-1]`. `gpu_backward()` and `gpu_evaluate()` use
the same misalignment, so training and validation loss look excellent.

`ChatbotTrainer` uses this path whenever `ADAI_ENABLE_GPU` is set, i.e. every `incremental_trainer`
pass on ai-machine (SYCL). **Reproduced on the CPU** with the same model classes and the GPU path's
exact decoder input, on a tiny echo task (150 epochs each):

| Decoder input during training | Final loss | Greedy generation exactly right |
|---|---|---|
| `[BOS] + target[:-1]` (CPU `forward()`) | 0.0008 | 12 / 12 |
| `target` unshifted (GPU `gpu_forward()`) | 0.0006 | **0 / 12** (`2 2 2 17 17 17 …`) |

The GPU-style model reaches equally low loss but can't generate: it learned "output the token I was
given". Every checkpoint trained on the GPU path needs retraining after the fix.

**Fix landed October 17, 2026:** `EncoderDecoderModel::make_decoder_input()` builds `[BOS] + target[:-1]` and both `forward()` and
`gpu_forward()` use it (`target_tokens` stay the labels, so `gpu_backward()` needed no change); `gpu_forward()` now throws
`invalid_argument` for an empty target. Verified on the oneAPI CPU SYCL device (scratch build with CPU devices enabled): pre-fix
`gpu_forward()` loss matched the unshifted CPU reference exactly (4.879566, against the shifted 4.751482); post-fix it matches the CPU
`forward()` loss exactly on three pairs. New tests `EncoderDecoderGPUTest.GpuForwardLossMatchesCpuForwardTeacherForcing` (failed
before the fix, passes after) and `GpuForwardRejectsEmptyTarget` run on any GPU build with a device; the first also fails while the
TD-384 kernel race is present, so the two land together. Remaining: run the tests on ai-machine, then retrain.

Action Items:

- [x] In `gpu_forward()`, build the decoder input exactly as `forward()` does (`[BOS] + target[:-1]`, via one shared helper) while keeping `targets` as the labels.
- [x] Add a CPU-vs-GPU equivalence test (same weights → same logits and loss for `forward()` and `gpu_forward()`).
- [x] Run it on ai-machine (with TD-384's fix); until GPU CI exists, ship a run package for the remote host. Done November 10, 2026: `EncoderDecoderGPUTest` passed on ai-machine's Arc Pro B60 through the GPU test package, and again November 12.
- [ ] Retrain affected models; consider bumping the checkpoint format version so GPU-trained pre-fix checkpoints are flagged.

Location in code: `src/EncoderDecoderModel.cpp` (`gpu_forward()`); tagged `TODO: See TD-358`.

Files to Modify:

- `src/EncoderDecoderModel.{hpp,cpp}`
- `tests/encoderdecoder_test.cpp`

On-device result (November 10, 2026): on ai-machine's Arc Pro B60 (Level Zero), package `adai-sycl-gpu-tests-bf3501cb-20261109`, run November 10, 2026, `EncoderDecoderGPUTest` passed in full, including
`GpuForwardLossMatchesCpuForwardTeacherForcing` and `GpuAndCpuGenerationAgree`, so the GPU path now trains on the shifted targets
on the real card. Retraining every GPU-trained checkpoint remains.

---

### TD-368: Weight Initialization Can't Be Seeded or Reproduced

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — code fix landed November 9, 2026 | Transformer / FeedForward | October 12, 2026 | 2-4 hours |

Description:
Found while tracing `FeedForward` and verified. Every `FeedForward` constructor seeds a fresh
`std::mt19937` from `std::random_device`, and `Matrix::randomize()` (used by the other layers'
initialization) does the same with a function-static generator. Two `FeedForward(4, 8)` instances got
different weights, and there's no seed setting anywhere in `src/` (no config key, no global seed).
So a training run can't be replayed from initialization, a reported divergence can't be reproduced
bit-for-bit, and a CPU-vs-GPU equivalence check (as TD-358's fix needs) must copy weights explicitly
rather than construct two identical models.

Action Items:

- [x] Add one process-wide seedable source: `adai::set_init_seed()`/`next_init_seed()` (`src/InitRng.{hpp,cpp}`), the `SEED` config key and `incremental_trainer --seed`; a random seed when unset, and it's logged.
- [x] Route `FeedForward`, `Matrix::randomize()` and the layers built on it through it, along with epoch shuffles, `Dropout`, `SIGReg` directions and LeJEPA spans.
- [x] Test: two models built with the same seed have identical weights.
- [ ] GPU replays are close, not exact (see below).

Fix (November 9, 2026): each user takes one value from `next_init_seed()` to seed its own generator, so nothing is shared between threads (this also closes TD-435's `randomize()` static-RNG race). With `SEED` unset (0), the first use picks a seed in the range `SEED` accepts and logs `Random seed N (set SEED=N to replay this run)`. Generation, `SpeculativeDecoder`, run ids and the deferred RLHF/LoRA `rand()` initializers are left out: serving isn't part of a training replay, and ids must stay unique.

Verified with real `incremental_trainer` passes (dropout on, 3 epochs): `SEED=7` twice, and `--seed 7` over `SEED=0`, gave byte-identical weight files on the CPU; `SEED=8` and an unset seed differed; and setting a logged random seed reproduced that run exactly. On the oneAPI CPU device two `SEED=7` passes used the same initialization and order but weren't byte-identical (epoch 1 loss equal to 6 decimals, epoch 2 gradient norm 8.183522 vs 8.183523), a last-bit difference that fits the parallel float reductions; the cause wasn't confirmed. Real GPUs are unmeasured.

Tests: `init_rng_test.cpp` (9 tests; 6 fail when the source ignores the seed) and `incremental_trainer_args_test.cpp`'s `SeedFlag`.

Location in code: `src/InitRng.{hpp,cpp}`, `src/Matrix.cpp`, `src/FeedForward.cpp`, `src/Dropout.cpp`, `src/SIGReg.cpp`, `src/LeJEPAEncoder.hpp`, `src/ChatbotTrainer.cpp`, `src/Config.{hpp,cpp}`, `src/IncrementalTrainerArgs.{hpp,cpp}`, `src/IncrementalTrainingTool.cpp`, `config.trainer.conf`.

On-device result (November 10, 2026): on ai-machine's Arc Pro B60 (Level Zero), package `adai-sycl-gpu-tests-bf3501cb-20261109`, run November 10, 2026, two `SEED=7` passes (3 epochs) printed identical
losses and gradient norms to 6 decimals in every epoch, and `SEED=8` differed from the first epoch, but the two `SEED=7` runs'
weight files weren't byte-identical. As on the oneAPI CPU device, that fits the device kernels' float atomics (the embedding
scatter-add, the gate-gradient sums and the gradient-norm sums add in a different order each run); the archive didn't include the weights, so the size of
the difference wasn't measured. Seeding replays initialization and data order on the GPU; bit-exact replay there would need
order-fixed reductions in place of those atomics.

---

### TD-370: `FeedForward` Argument Checks, GPU Null Dereference and Console Output

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Transformer / FeedForward | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `FeedForward`; the CPU items were verified, the GPU ones are by inspection:
- The constructor doesn't validate dimensions: `FeedForward(0, 4)` constructs with an infinite init
  scale. (`Predictor` checks its own arguments; the blocks don't.)
- `set_W1()`…`set_b2()` don't check shape (`set_W1(3×3)` was accepted, and the next `forward()` threw
  a multiply error), and they don't mark the GPU mirror stale.
- `backward()` before any `forward()` fails with a misleading *"Matrix dimensions must match for
  addition"*.
- `gpu_backward()` dereferences `gpu_` without a check, so calling it before `gpu_forward()` crashes.
  `gpu_forward()` doesn't check `input.cols`.
- The GPU saturation hook's `count_below_threshold()` returns to the host, synchronizing the queue
  once per FFN per forward while installed.
- `save_weights()`, `load_weights()` and `print_config()` print to `std::cout`, and `print_config()`
  uses `int` arithmetic that overflows for very large dimensions.

Action Items:

- [ ] Validate constructor arguments; check shapes in the setters and invalidate (or re-upload) the GPU mirror.
- [ ] Give `backward()` and `gpu_backward()` a clear "no forward pass" error; check `input.cols` in `gpu_forward()`.
- [ ] Batch the GPU hook's reduction per step rather than per layer, or document the cost.
- [ ] Switch to `adai::Logger`; use `size_t` in `print_config()`.

Location in code: `src/FeedForward.cpp` (constructor, `backward()`, `gpu_forward()`, `gpu_backward()`, `save_weights()`), `src/FeedForward.hpp` (setters); tagged `TODO: See TD-370`.

Files to Modify:

- `src/FeedForward.{hpp,cpp}`

---

### TD-376: `FtpDataServer` Slow Shutdown, Unencrypted `FEAT` and Silent Start Failures

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / FtpDataServer | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `FtpDataServer` and verified:
- `sweep_loop()` sleeps a full minute between checks and `stop()` joins it, so `stop()` took 56.9 s and
  58.1 s in two runs, delaying every `registry_server` shutdown and restart.
- `FEAT` is always written to the raw socket, even after `AUTH TLS`, which corrupted the encrypted
  control channel (`WRONG_VERSION_NUMBER`). libcurl doesn't send `FEAT` after `AUTH`, so trainers
  aren't affected today.
- `start()` reports bind, listen and TLS-setup failures only in the log; it has no return value, so
  `registry_server` can't tell whether FTP came up.

Action Items:

- [ ] Replace the sleep with a condition-variable wait that `stop()` notifies.
- [ ] Send `FEAT` through `send_reply()`-style TLS-aware output.
- [ ] Return `bool` from `start()` and have `registry_server` exit (or report unhealthy) when FTP was requested but failed.

Location in code: `src/FtpDataServer.hpp` (`sweep_loop()`, the `FEAT` branch, `start()`); tagged `TODO: See TD-376`.

Files to Modify:

- `src/FtpDataServer.hpp`
- `src/RegistryServer.cpp`

**Partial fix (October 21, 2026, with TD-372).** The token sweep now waits on a condition variable that `stop()` signals, so `stop()`
returns at once instead of after up to 60 s. Unencrypted `FEAT` and the silent `start()` failure remain.

---

### TD-377: `FtpDataServer` Token Semantics and Setup Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / FtpDataServer | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `FtpDataServer`, by inspection:
- Token limits are looser than documented: expiry is checked only at login, so a session outlives its
  token; a consumed token can still `RETR` again on its open connection; usernames carry 32 random
  bits and `TokenStore::insert()` silently overwrites on a collision.
- `PASV` advertises `127.0.0.1` when `--ftp-ip` isn't set, which no remote trainer can use.
- The constructor doesn't validate the PASV range (`min > max` gives an empty pool).
- An empty command line ends the session.
- The header claims HMAC passwords make forged tokens rejectable, but random passwords are equally
  unforgeable and the server stores the password either way; `--ftp-secret` is passed in argv, so
  it's visible in `ps` (the config key is ignored, TD-284).
- `ftp_detail::local_ip()` and `TokenStore::remove()` are unused.

Action Items:

- [ ] Re-check expiry on `RETR`; end the session after a successful `RETR` (or refuse further ones); use 64-bit usernames or refuse to overwrite.
- [ ] Require `--ftp-ip` when FTP is enabled (or derive it), and validate constructor arguments.
- [ ] Ignore empty lines; correct the HMAC comment; read the secret from config or a file (with TD-284); remove the unused helpers.

Location in code: `src/FtpDataServer.hpp` (`TokenStore`, `cmd_pasv()`); tagged `TODO: See TD-377`.

Files to Modify:

- `src/FtpDataServer.hpp`
- `src/RegistryServer.cpp`

---

### TD-380: Generation-Quality Scores Aren't Comparable or Well-Sampled

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open — code fix landed October 25, 2026 | Metrics / GenerationQuality | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `GenerationQualityMetrics`, by inspection:
- `tokenize()` drops punctuation and keeps internal apostrophes, unlike sacreBLEU's `13a` tokenizer
  (punctuation as separate tokens) and `rouge-score` (non-alphanumerics as spaces), so these numbers
  can't be compared with published BLEU/ROUGE.
- Scores are on a 0–1 scale, while the lesson [Reading BLEU and ROUGE Results](../../lessons/reading-bleu-rouge.md)
  gives interpretation ranges on the 0–100 scale for standard corpus BLEU-4.
- `generation_quality_sample_size` defaults to 10 pairs, which makes BLEU-4 very noisy from epoch to
  epoch.

Action Items:

- [x] Document the tokenization and scale in the header, metrics API and dashboards (or report 0–100).
- [x] Add a note to the lesson that the training metric isn't standard corpus BLEU.
- [ ] Raise the default sample size (or report a confidence interval) once TD-359 makes generation length bounded. (TD-359's fix
  landed October 26, 2026; this can now be done.)

**Owner decision (October 25, 2026).** Standard tokenizers and the 0–100 scale.

**Fix (October 25, 2026).**
- **BLEU.** `GenerationQualityEvaluator` follows sacreBLEU's defaults: the 13a tokenizer (`tokenize_13a()`: punctuation as tokens,
  `.`/`,` kept between digits, `-` split after a digit, entities unescaped, case kept), and sacreBLEU's `compute_bleu()`. That means
  `exp` smoothing, 0 when no unigram matches, and an order with no hypothesis n-grams giving about 0 at corpus level.
  `sentence_bleu()` is sacreBLEU's `sentence_bleu()` (adds effective order). This replaces TD-378's unsmoothed corpus and
  floor-smoothed sentence scores.
- **ROUGE.** rouge-score's defaults (`tokenize_rouge()`: lowercase, `[a-z0-9]` runs, no stemming); the F1 math was already the same.
- **Scale and consumers.** Every score is on 0–100. The per-sample backfill stores `sentence_bleu() / 100` in `SampleMeta::quality`,
  which stays [0, 1]. The Tizen dashboard colors at 30/10 (was 0.3/0.1) and shows 2 decimals, the Android ops dashboard shows 2
  decimals, and the lesson notes the change.

Tests (`generation_quality_test.cpp`, rewritten): tokenizer outputs and every score are checked against sacreBLEU 2.6.0 and
rouge-score, installed in a scratch directory to produce the values. All 16 tests match within 0.001, including:
- punctuation, case and number handling;
- `exp`-smoothed zero-match orders;
- the corpus/sentence effective-order difference.

Stored generation-quality values before this date are on 0–1 with the old tokenization.

Location in code: `src/GenerationQualityMetrics.hpp` (`tokenize()`), `src/ChatbotTrainer.hpp` (`generation_quality_sample_size`); tagged `TODO: See TD-380`.

Files to Modify:

- `src/GenerationQualityMetrics.hpp`
- `src/ChatbotTrainer.hpp`
- `docs/lessons/reading-bleu-rouge.md`

---

### TD-383: `GPUManager` Lifecycle Gaps (`set_device`, `cleanup`, `initialize`)

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | GPU / GPUManager | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `GPUUtils.hpp`, by inspection:
- `set_device()` says it "also resets the memory budget", but it only calls `cudaSetDevice()`; the
  stream and cuBLAS handle stay bound to the old device. No caller exists, so it's a latent trap.
- `cleanup()` sets `allocated_bytes_` to 0 while layers' persistent `GPUState` buffers still exist,
  so after a re-`initialize()` the budget undercounts.
- `initialize()` returns true without applying new arguments once initialized, and leaks the stream
  if the cuBLAS step throws.
- `probe_diagnostic()` discards the CUDA error, so a driver failure reads as "0 device(s)".
- `GPUMemory`'s `count × sizeof(T)` isn't overflow-checked.

Action Items:

- [ ] Make `set_device()` rebuild the stream, handle and budget (or remove it); have `cleanup()` refuse or warn while allocations are live; apply or reject differing arguments on repeat `initialize()`; free partial state on failure; include the CUDA error text in `probe_diagnostic()`.

**SYCL amendment (October 12, 2026, [GPUUtils_SYCL.md](../reference/source/GPUUtils_SYCL.md)):** the
SYCL `set_device()` does recreate its queue, but allocations made on the old queue are later freed
with `sycl::free(ptr, new_queue)` (undefined across contexts) and the budget isn't recomputed;
`cleanup()` zeroes the counter and live buffers are then never freed (by design, TD-096); repeat
`initialize()` also ignores new arguments. Tagged in `src/gpu/sycl/GPUUtils_SYCL.hpp` too.

Location in code: `src/gpu/GPUUtils.hpp` (`initialize()`, `cleanup()`, `set_device()`, `probe_diagnostic()`); tagged `TODO: See TD-383`.

Files to Modify:

- `src/gpu/GPUUtils.hpp`

---

### TD-384: Shared-Memory Race in GPU Softmax, LayerNorm and Cross-Entropy Kernels (CUDA and SYCL)

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open — code fix landed October 17, 2026 | GPU / Kernels | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `MatrixGPU.cu`, by inspection (a definite race under the CUDA and SYCL memory
models). `softmax_rows_kernel`, `layer_norm_fwd_kernel`, `cross_entropy_loss_kernel` and
`cross_entropy_grad_kernel` each run two tree reductions through the same shared `scratch` buffer.
After the first reduction's final barrier, every thread reads `scratch[0]` (the row max or the mean),
computes its next partial, and writes `scratch[lid]`, with **no barrier between the read and
thread 0's write to `scratch[0]`**. Warps (CUDA) or sub-groups (SYCL) of a 256-thread block aren't
scheduled in lockstep, so a lagging warp can read thread 0's partial sum instead of the max or mean.
The result for that row is silently wrong: softmax `exp(x − wrong_max)` can overflow to `inf`/`NaN`,
LayerNorm centres on a variance partial, and the loss and gradient for that position are wrong.

It's likely rare, since thread 0 must finish a full loop pass before the lagging warp's single load,
but it's intermittent, data-dependent and affects attention, every LayerNorm and the loss in GPU
training. **`MatrixGPU_SYCL.cpp` has the identical pattern in the same four kernels**, and ai-machine
trains on SYCL (Intel Arc), so its runs are exposed too.

**Verified October 12, 2026** ([MatrixGPU_SYCL_cpp.md](../reference/source/MatrixGPU_SYCL_cpp.md)):
contrary to the original note here, it **does** reproduce on the oneAPI CPU device, deterministically
(the CPU runtime runs a work-group's sub-groups in turn, so sub-group 0 always overwrites
`scratch[0]` first). Wrong results: softmax rows (2.5×10⁻² relative error), LayerNorm forward
(0.32), cross-entropy loss (4×10⁻³) and grad (2×10⁻³), identical over five runs. Adding one
`item.barrier(local_space)` after each of the four reads made all of them pass (≤ 2×10⁻⁷; LayerNorm
backward 6.6×10⁻⁵ vs. finite differences). On a GPU it's timing-dependent rather than certain.

**Fix landed October 17, 2026:** all eight sites now wait after reading `scratch[0]`
(`__syncthreads()` in CUDA, `item.barrier(local_space)` in SYCL). These are the only `scratch[0]`
reads followed by another reduction; the rest are each kernel's last use of the buffer. Verified on
the oneAPI CPU device (scratch build with device selection widened to `device_type::all`):
- new tests `MatrixGPUTD003Test.SoftmaxWideRowsMatchHost` (64 × 300) and `LayerNormWideRowsMatchHost`
  (32 × 768, 5 trials each) pass, and so does `EncoderDecoderGPUTest.GpuForwardLossMatchesCpuForwardTeacherForcing`;
- all three fail on every run without the barriers, even with TD-393 fixed.

The CUDA change compiles (`gpu` preset) but hasn't been run, since no CUDA device is available.
Remaining: a racecheck on real hardware, and the on-device run on ai-machine together with TD-358.

Action Items:

- [x] Add a barrier (`__syncthreads()` / `item.barrier(local_space)`) after each `scratch[0]` read that's followed by another reduction, in all four kernels in both backends (or reduce into a second buffer).
- [ ] On a GPU host, run `compute-sanitizer --tool racecheck` (CUDA) over a training step to confirm no other shared-memory races remain.
- [x] Do this with TD-358, before the retraining runs.

Location in code: `src/gpu/MatrixGPU.cu` and `src/gpu/sycl/MatrixGPU_SYCL.cpp` (softmax rows, LayerNorm forward, cross-entropy loss and grad); previously tagged `TODO: See TD-384` at all eight sites; now marked with `TD-384:` fix comments.

Files to Modify:

- `src/gpu/MatrixGPU.cu`
- `src/gpu/sycl/MatrixGPU_SYCL.cpp`

On-device result (November 10, 2026): the SYCL fix verified on ai-machine's Arc Pro B60 (Level Zero), package `adai-sycl-gpu-tests-bf3501cb-20261109`, run November 10, 2026: every softmax, LayerNorm and
loss test passed (`matrixgpuTd003Tests`, `multiheadattentionTests`, `encoderdecoderTests`, `chatbottrainerTests`), as did three
real training passes until the driver reset in one of them (TD-566). CUDA remains compile-checked only.

---

### TD-385: `matrix_sum_gpu` Allocates and Synchronizes on Every Call

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open — code fix landed November 9, 2026 | GPU / Kernels | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `MatrixGPU.cu`, by inspection. `matrix_sum_gpu()` reduces recursively and
allocates a `GPUMemory` buffer at every level (`cudaMalloc`/`cudaFree`, both synchronizing), ending
with a blocking one-float download. `matrix_cross_entropy_loss_gpu()` calls it every training step,
so each step stalls the CPU on the GPU and churns the allocator. `matrix_count_below_threshold_gpu()`
(the saturation hook) has the same shape. **SYCL (October 12, 2026):** both SYCL versions have the same
shape plus a `.wait()` on the submit; the SYCL `count_below_threshold` also lacks a `size <= 0` guard.

Action Items:

- [x] Reduce without a buffer or a wait per level.
- [ ] Keep the loss on the device until it's actually needed on the host. Not done: the trainer reads each sample's loss (`CLAUDE.md`'s one scalar per sample), so that download stays. Needs a device-side loss accumulator in `ChatbotTrainer` if it's ever worth doing.

Fix (November 9, 2026): `matrix_sum_gpu()` and `matrix_count_below_threshold_gpu()` allocate one buffer for every level (`reduction_buffer_size()`), and `finish_reduction()` launches each further level in place and downloads only the final value, in both backends.
The SYCL `.wait()`s on each submit are gone (in-order queue), which leaves one wait per reduction. `count_below_threshold` returns 0 for `size <= 0` in both backends (SYCL threw, TD-392's allocation, fixed November 18, 2026). TD-390 had already removed the allocator's own synchronization.
A 262,144-element sum on the oneAPI CPU device went from about 210-230 to 130-138 us per call. At 512 elements, the loss's size, there was no clear change (950-1000 vs 915-940 us), so this is mostly tidying for the training path; on-device timing is still unmeasured.

Tests: `matrixgpu_td003_test.cpp`'s `SumAndCountAreExactAcrossReductionLevels` (each size where a level appears, up to three levels; passed before the change too, guarding the rewrite) and `EmptyReductionsAreZero` (threw on old SYCL). SYCL ran on the oneAPI CPU device; CUDA is compile-checked only.

Location in code: `src/gpu/MatrixGPU.cu`, `src/gpu/sycl/MatrixGPU_SYCL.cpp` (`reduction_buffer_size()`, `finish_reduction()`).

Files Modified:

- `src/gpu/MatrixGPU.cu`
- `src/gpu/sycl/MatrixGPU_SYCL.cpp`
- `tests/matrixgpu_td003_test.cpp`

---

### TD-386: GPU Kernel Input Checks and Dead Batch Functions

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | GPU / Kernels | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `MatrixGPU.cu`, by inspection:
- `matrix_cross_entropy_loss_gpu`/`_grad_gpu` don't check that target IDs are in `[0, vocab_size)`,
  so a bad ID reads out of bounds on the device.
- The `float4` element-wise kernels require 16-byte-aligned pointers. Current callers pass
  `cudaMalloc` base pointers, so they're fine, but nothing checks or documents it for future sliced
  pointers.
- No wrapper checks dimensions, and only launch errors are caught (`cudaGetLastError()`), so faults
  surface at a later, unrelated synchronizing call.
- `matrix_batch_add_gpu`, `matrix_batch_multiply_gpu` and `matrix_add_scalar_gpu` have no callers in
  either backend.
- **SYCL (October 12, 2026):** the SYCL batch helpers `q.memcpy` from the caller's **host** pointer
  arrays without waiting; unlike CUDA's pageable staging, SYCL doesn't guarantee the host buffer has
  been read when the call returns, so a temporary array could be read after it's freed.
- **Verified November 22, 2026 (oneAPI CPU device, TD-394's `gpuKernelReferenceTests`):** `matrix_batch_multiply_gpu` returns
  without computing anything; its outputs keep whatever the memory held. Waiting for `gemm_batch()` before returning makes it correct,
  so oneMKL reads host memory after the call returns: the stack-local group parameters (`m64`, `k64`, `n64`, `lda`, `ldb`, `ldc`,
  `group_size`, `trans_n`, `alpha`, `beta`) or the caller's pointer arrays. Its test is `DISABLED_BatchMultiply` until this item
  decides; `matrix_batch_add_gpu` passes.

Action Items:

- [ ] Validate targets on the host before upload (or clamp/flag on the device); document or assert alignment; add a debug-build `cudaStreamSynchronize` + check after launches.
- [ ] Remove the uncalled batch functions, or keep them explicitly as TD-171 groundwork with a test.

Location in code: `src/gpu/MatrixGPU.cu` (element-wise wrappers, cross-entropy, batch functions); tagged `TODO: See TD-386`.

Files to Modify:

- `src/gpu/MatrixGPU.{hpp,cu}`
- `src/gpu/sycl/MatrixGPU_SYCL.{hpp,cpp}`

---

### TD-387: GPU Generation Fails on Input Longer Than `max_seq_length`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — code fix landed October 26, 2026 (with TD-365) | GPU / GPUMatrix | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `MatrixGPU.hpp`, by inspection (definite). `gpu_generate_response()` and
`gpu_generate_response_with_strategy()` build a `GPUDecoderKVCache(decoder_layers, max_seq_length,
d_model)`, whose cross-attention caches hold at most `max_seq_length` rows. On the first decode step,
`CrossAttention::gpu_forward_with_cache()` appends the **entire encoder output** to that cache, but
encoder input is never truncated to `max_seq_length` (TD-365). So any chat input longer than
`max_seq_length` tokens makes `GPUKVCache::append()` throw `out_of_range`, and the request fails on a
GPU server (the TD-033 serving path), while the CPU path answers the same request (with the TD-365
positional-encoding gap). The self-attention cache has the same cap but isn't reachable today:
`TextGenerator` stops at `max_length = max_seq_length` generated tokens, which fills it exactly.

Action Items:

- [x] Truncate encoder input once where the model tokenizes (TD-365), which removes the overflow; or size the cross-attention caches from the actual encoder length.
- [ ] Test (on a GPU host): an over-length prompt generates without throwing and matches the CPU path. (Run on the oneAPI CPU
  device, and on ai-machine's Arc through the GPU test package on November 10 and 12, 2026: no throw. The CPU-match part is not
  checked.)

**Fixed by TD-365 (October 26, 2026).** Both GPU generation paths now keep the encoder input's last `max_seq_length` tokens, so the
encoder output fits the cross-attention cache. Test `EncoderDecoderGPUTest.GpuGenerationAcceptsOverLongInput`: a 480-character input
on a `max_seq_length = 16` model goes through `gpu_generate_response()` and `gpu_generate_response_with_strategy()` without throwing,
on the oneAPI CPU device (`build/sycl-cpudev`, `ADAI_SYCL_ALLOW_CPU_DEVICE=1`).

**Verified October 12, 2026** ([MatrixGPU_SYCL.md](../reference/source/MatrixGPU_SYCL.md)): on the
oneAPI CPU device, appending a 12-row encoder output to a `GPUDecoderKVCache(2, max_seq_length = 8,
4)` cross-attention cache throws `"GPUKVCache::append: exceeds max_seq_length capacity"`.

Location in code: `src/gpu/MatrixGPU.hpp` (`GPUDecoderKVCache` constructor), `src/EncoderDecoderModel.cpp` (both GPU generation paths); tagged `TODO: See TD-387`.

Files to Modify:

- `src/EncoderDecoderModel.cpp`
- `src/gpu/MatrixGPU.hpp`

---

### TD-389: `GPUMatrix` Shape Checks, Uninitialized Memory and Untested GPU KV Caches

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | GPU / GPUMatrix | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `MatrixGPU.hpp`, by inspection:
- `GPUMatrix::add_inplace`, `softmax_backward`, `gelu_backward`, `add_row_bias`,
  `masked_fill_inplace`, `layer_norm` and `layer_norm_backward` don't check shapes (the arithmetic
  operators do), so a caller's shape bug becomes silent device memory corruption or a fault.
- `GPUMatrix(r, c)` leaves device memory uninitialized (unlike CPU `Matrix`), and a short `upload()`
  leaves the rest as garbage; `r × c` is unchecked.
- `count_below_threshold()` on an empty matrix returns `NaN`.
- `GPUKVCache` validates its sizes only after allocating, and `GPUDecoderKVCache` doesn't
  bounds-check layer indices.
- **`GPUKVCache`/`GPUDecoderKVCache` have no tests at all**, not even GPU-gated ones.

Action Items:

- [ ] Add shape checks to the unchecked operations; zero (or document) new allocations; validate before allocating; check layer indices.
- [ ] Add GPU-gated tests for the KV caches (append, capacity, materialize, clear) mirroring `KVCache`'s CPU tests.

**Verified on SYCL, October 12, 2026** ([MatrixGPU_SYCL.md](../reference/source/MatrixGPU_SYCL.md)):
on the oneAPI CPU device, `big(4×4).add_inplace(small(2×2))` was accepted with no error (reading past
`small`'s buffer), and a fresh `GPUMatrix` held garbage (`6.6×10⁻³⁷`). The `GPUKVCache` itself ran
correctly (offsets, materialize, clear), its first run anywhere; the tests above should still be added
(TD-394's CPU-device mode makes that possible without a GPU).

Location in code: `src/gpu/MatrixGPU.hpp` (`GPUMatrix` constructor and unchecked operations, `GPUKVCache` constructor); tagged `TODO: See TD-389`.

Files to Modify:

- `src/gpu/MatrixGPU.hpp` (the one `GPUMatrix` since TD-395)
- `tests/matrixgpu_td003_test.cpp`

---

### TD-390: SYCL Deferred Frees and Blocking Copies Stall the In-Order Queue

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — code fix landed October 20, 2026 | GPU / GPUManager | October 12, 2026 | 4-8 hours |

Description:
Found while tracing `GPUUtils_SYCL.hpp`; the pattern was verified on a CPU queue, and the GPU
magnitude is unmeasured. `GPUMemory`'s destructor and move assignment call `defer_free()`, which
submits a `host_task` on the in-order queue to run `sycl::free` and release the budget. That's correct
for ordering, but on an in-order queue a `host_task` waits for every earlier command to finish, and
nothing after it starts until the host callback returns: a device→host→device round trip with the
device idle, **for every `GPUMatrix` temporary destroyed**. `copy_from_host`/`copy_to_host` also
`.wait()` on the whole queue, so every upload drains it.

A standalone `icpx -fsycl` program reproducing the pattern on the oneAPI CPU device (allocate, small
kernel, `host_task` free) took **1.35 ms per iteration**, against **0.051 ms** for a blocking `wait()` +
`free` and **0.014 ms** for the kernel alone with a reused buffer. `MultiHeadAttention.cpp` and
`CrossAttention.cpp` alone have about 50 `GPUMatrix` uses each, so ai-machine's SYCL training and
generation likely spend a large share of their time in these bubbles. The `allocated_bytes_` lag that
`ChatbotTrainer` works around with periodic `gpu_synchronize()` calls is the same mechanism.

Action Items:

- [ ] Measure first on ai-machine: time one training step, and count `host_task`s per step (or use `onetrace`/`unitrace`).
- [x] Replace per-free `host_task`s with a caching allocator: keep freed blocks in a size-bucketed pool tied to the in-order queue (safe to reuse immediately, since later kernels on the same queue run after earlier ones), and return memory to USM only at `cleanup()` or under budget pressure.
- [ ] Make uploads asynchronous where the host buffer outlives the copy (or stage through a pinned buffer), keeping `.wait()` only on downloads.

Location in code: `src/gpu/sycl/GPUUtils_SYCL.hpp` (`GPUMemory::defer_free()`, `copy_from_host()`/`copy_to_host()`); tagged `TODO: See TD-390`.

Files to Modify:

- `src/gpu/sycl/GPUUtils_SYCL.hpp`
- `src/ChatbotTrainer.cpp` (remove the accounting workaround syncs afterwards)

**Fix (October 20, 2026).**
- **Caching allocator:** `GPUMemory` takes its blocks from `GPUManager::allocate_block()` and returns them with
  `release_block()`, which puts the block in a size-bucketed pool: no `host_task` and no wait. Sizes are 512-byte
  steps up to 1 MiB, then an eighth of the next power of two, at most 12.5% over (`pool_size()`). Reuse is safe at
  once on the in-order queue: commands still using a released block were submitted earlier, so they run before
  anything its next owner submits.
- **Budget:** live and cached bytes both count against `GPU_MEMORY_FRACTION`. An allocation that doesn't fit
  waits for the queue and returns the pool to USM (`trim_pool_locked()`) before refusing. `cleanup()` and
  `set_device()` trim too, and a block from before them is never pooled (generation counter).
- **No accounting lag:** the budget counter now drops when a `GPUMemory` is destroyed, rather than when a
  `host_task` runs. `get_used_memory_bytes()` reports live blocks at their pool size, and
  `get_cached_memory_bytes()` reports the pool.
- **Trainer:** `ChatbotTrainer`'s six workaround `gpu_synchronize()` calls are gone (after each training and
  validation sample, both catch blocks, and the two generation-quality loops). Each sample's own uploads and loss
  download still order the work. One side effect: the per-sample compute-time metric now counts the backward's
  submission, not its device time, which moves into the next sample's upload wait. Per-epoch throughput is
  unchanged.
- **Uploads:** deliberately still `.wait()`. Callers pass host temporaries that may not outlive an asynchronous
  copy, and every training sample downloads its loss anyway, so asynchronous uploads would need staging buffers
  for little gain.

Measured on the oneAPI CPU device (`ADAI_SYCL_ALLOW_CPU_DEVICE=1`):
- **Per temporary:** allocate + small kernel + free went from 1.25 ms to 0.0145 ms (86×). This used the real
  `GPUMemory`, built once with the old header and once with the new.
- **Real pass:** the same SYCL `incremental_trainer` pass (d_model 128, 2+2 layers, 120 samples, 1 epoch) went
  from 69.4 s to 38.4 s.

New tests (`gpuutils_test.cpp`, SYCL only):
- a released block is reused and counted at once, without `synchronize()` (the old header still counted it);
- reuse keeps queue order (a copy queued before the release still reads the old data);
- budget pressure returns the pool to the device;
- a block from before `cleanup()` isn't pooled;
- `pool_size()` rounding.

Still open: measuring on ai-machine.

**CUDA follow-up (October 20, 2026).** CUDA's `GPUMemory` called `cudaMalloc()`/`cudaFree()`, and `cudaFree()` synchronizes
the device: the same stall, once per temporary. It now goes through `GPUManager::allocate_block()`/`release_block()`,
which use `cudaMallocAsync()`/`cudaFreeAsync()` on the ADAI stream, the stream-ordered allocator `MatrixGPU.cu`'s batched
paths already used. Every ADAI kernel, copy, memset and cuBLAS call runs on that one stream (checked), so a freed block's
earlier uses complete before it is reused.
- **Pool:** the device's default memory pool keeps freed memory up to the ADAI budget (its release threshold) instead of
  returning it to the driver at each synchronization.
- **Allocation failure:** a failed allocation retries once after `cudaMemPoolTrimTo()`.
- **Teardown:** `cleanup()` trims the pool. A block destroyed after `cleanup()` falls back to `cudaFree()`, which also frees
  pool memory.
- **Fallback:** devices without memory-pool support (`cudaDevAttrMemoryPoolsSupported`) keep `cudaMalloc()`/`cudaFree()`.
- **Budget:** the counter is unchanged; it already dropped at release.

Compile-checked only (no CUDA device here). New CUDA-only tests, which skip without a device:
- `CudaUsesTheStreamOrderedPoolWhenSupported`;
- `CudaFreeKeepsStreamOrder`: a copy queued before its source is freed still reads the old data.

---


### TD-397: Hippocampal Swap File Grows Without Bound

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Memory / HippocampalMemory | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `HippocampalMemory`, by inspection. With a swap path configured (as
`chatbot_api_server` always does: `<model>.hippocampal.swap`), every eviction appends a
`(2·d_model + 1)`-float record (about 4 KB at `d_model = 512`). Once the buffer is full, that's one
record per generated response. Only `recall_from_swap()` ever shrinks the file, and nothing in
production calls it, so a long-running server's swap file grows without bound next to the model
checkpoint.

Action Items:

- [ ] Cap the swap file (a ring of N records, or drop the oldest records when over a size limit), make the cap configurable, and log its size at startup.

Location in code: `src/HippocampalMemory.cpp` (`append_to_swap()`); tagged `TODO: See TD-397`.

Files to Modify:

- `src/HippocampalMemory.{hpp,cpp}`
- `src/Config.{hpp,cpp}`

---

### TD-398: `HippocampalMemory` Per-Step Copies and Unchecked Saves

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Memory / HippocampalMemory | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `HippocampalMemory`, by inspection:
- `read_all()` copies every slot into fresh `[n, d_model]` key and value matrices, and
  `DecoderBlock::forward()` calls it on every forward pass of every injected layer.
- `save()` and `save_associations()` don't check their writes, so a full disk produces a silently
  truncated file (which `load()` then rejects as corrupt).
- The main state and the association matrix are separate files saved by separate calls; if one save
  fails or is stale, `load_associations()` rejects the mismatch (or, with a coincidentally equal
  count, loads associations for different slots).

Action Items:

- [ ] Cache the packed key/value matrices and update them incrementally on write/evict; check stream state after saves; save both files atomically together (or into one file).
- [x] Checked writes and atomic replacement of each file (done with TD-250, November 1, 2026: written to `<path>.tmp` through `adai::binio::Writer`, then renamed). Still open: the `read_all()` copies, and the two files are replaced one after the other (`chatbot_api_server` saves both under the model lock, so they match unless the process dies between the two renames).

Location in code: `src/HippocampalMemory.cpp` (`read_all()`, `save()`, `save_associations()`); tagged `TODO: See TD-398`.

Files to Modify:

- `src/HippocampalMemory.{hpp,cpp}`

---

### TD-399: `IMetricsReporter` Buffering and Ordering Contract Is Undocumented

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Metrics / Reporter | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IMetricsReporter`, by inspection. In `MetricsPushClient`, eight of the
interface's methods (`update_advanced_epoch_metrics`, both adaptive-clip methods,
`update_activation_saturation`, `update_attention_entropy`, `update_padding_efficiency` and both
LeJEPA methods) only store a value; it's sent inside the next `end_epoch()` payload, and
`start_epoch()` resets those buffers. So they must be called between `start_epoch()` and `end_epoch()`
of the same epoch. Both callers do this today, but the rule appears only in a call-site comment in
`IncrementalTrainingTool.cpp`; the interface's doc comments don't say which methods are buffered or
that ordering matters, so a new implementation or a reordered caller would lose or misattribute
values silently. Separately, `AbnormalSample` carries full, untruncated `input_text`/`target_text` into
the push queue and the metrics DB (capped only by `max_abnormal_samples` per epoch).

Action Items:

- [ ] Document per method whether it's pushed immediately or included in the next `end_epoch()`, and the ordering rule, in the interface.
- [ ] Optionally truncate `AbnormalSample` texts (for example to 1 KB) before queuing.

Location in code: `src/IMetricsReporter.hpp` (class doc, `AbnormalSample`); tagged `TODO: See TD-399`.

Files to Modify:

- `src/IMetricsReporter.hpp`
- `src/MetricsPushClient.cpp`

---

### TD-400: `NullMetricsReporter` Keeps Reporting-Only Work Switched On

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Metrics / Reporter | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IMetricsReporter`; cost measured. `ChatbotTrainer` uses
`if (metrics_reporter_)` (ten places) as an "are metrics enabled?" switch, gating work that exists
only to be reported: installing the saturation/entropy hooks on every layer (on GPU, the saturation
hook synchronizes the queue per layer, TD-370), per-layer gradient norms, the per-epoch diagnostics,
and BLEU/ROUGE generation when enabled. `IncrementalTrainer` always passes a reporter, a
`NullMetricsReporter` when there's no metrics server or the session failed to start, so under
`incremental_trainer` all of that runs with nowhere to report. On a tiny model (2 layers, d = 64, two
epochs) the cost was within noise (5.10 s vs. 5.12 s; 5.21 s with BLEU), so it's mostly a semantic trap,
but it scales with model size, BLEU sample size and GPU synchronization.

Action Items:

- [ ] Add `virtual bool enabled() const` to `IMetricsReporter` (false for `NullMetricsReporter`), and have `ChatbotTrainer` check `metrics_reporter_ && metrics_reporter_->enabled()`.

Location in code: `src/IMetricsReporter.hpp` (`NullMetricsReporter`), `src/ChatbotTrainer.cpp` (the hook-installation guard and its siblings); tagged `TODO: See TD-400`.

Files to Modify:

- `src/IMetricsReporter.hpp`
- `src/ChatbotTrainer.cpp`

---

### TD-432: `LoRAAdapter::load()` Silently Adopts the File's Shape

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Attention / LoRA | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `LoRA` and verified. `LoRAAdapter::load()` reads `input_dim`, `output_dim`, `rank`
and `alpha` from the file and **replaces** the adapter's own values with them, reallocating `A`/`B`,
without checking them against the adapter (or the attention layer) it's loading into. Loading a 64×64
rank-8 file into a 6×5 rank-2 adapter succeeded silently (now rank 8, `A` 8×64), and the next
`forward()` failed with "Matrix dimensions incompatible for multiplication". A corrupt file's zero or
negative rank isn't rejected, and neither `save()` nor `load()` checks the stream. (LoRA is test-only
today; adapters aren't saved with checkpoints at all, TD-302.)

Action Items:

- [ ] Validate the file's shape against the adapter (throw on mismatch, leaving it unchanged), reject non-positive rank/dims, and check reads and writes.

Location in code: `src/LoRA.hpp` (`LoRAAdapter::load()`, `save()`); tagged `TODO: See TD-432`.

Files to Modify:

- `src/LoRA.hpp`

---

### TD-433: `LoRA` Initialization, Update and Config Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Attention / LoRA | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `LoRA`, by inspection:
- `A` is initialized with the C `rand()`: process-global, unseeded (identical values every run),
  not thread-safe, and separate from the project's other RNGs (TD-368).
- `update(lr)` (plain SGD) doesn't zero the gradients afterwards, unlike every layer's `update_weights()`.
- `LoRAConfig::dropout` is never applied and `apply_to_ffn` has no implementation (the attention
  classes call it "not applicable").
- `LoRAConfig::reduction_ratio()` divides by the adapter parameter count (infinite when every
  `apply_to_*` is false).
- `forward(x, W_output)`'s second parameter is the base layer's **output**, not a weight.

Action Items:

- [ ] Use the project RNG (TD-368) for initialization; zero gradients in `update()` (or document); implement or remove `dropout`/`apply_to_ffn`; guard the ratio; rename the parameter.

Location in code: `src/LoRA.hpp` (`initialize()`, `update()`, `LoRAConfig`); tagged `TODO: See TD-433`.

Files to Modify:

- `src/LoRA.hpp`

---

### TD-435: `Matrix` Accessor Cost, `sum()` Accuracy, `randomize()` Race and API Hygiene

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Core / Matrix | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `Matrix`:
- **Accessor cost (verified).** `operator()(i, j)` is bounds-checked and defined out-of-line in
  `Matrix.cpp`, so without LTO it can't be inlined: 5.2M reads took 16.7 ms vs 6.9 ms through `data[i][j]`
  (2.4×). About 80 sites in `MultiHeadAttention.cpp` use it in loops.
- **`sum()` accuracy (verified).** It accumulates in `float` (8 AVX2 lanes plus a scalar tail): 16.8M
  copies of 1e-4 summed to 1714.48 vs the exact 1677.72 (2.2%). Few callers today.
- **`randomize()` race (by inspection).** Its function-static `mt19937` is shared without locking, so two
  threads initializing layers at once race on it (seeding itself is TD-368).
- **Hygiene (by inspection).** `data`/`rows`/`cols` are public and mutable, so nothing keeps them
  consistent; negative sizes throw `length_error`/`bad_alloc` instead of a dimension error; `print()` writes
  to `std::cout`; the `*_gpu()` wrappers (other than `multiply_gpu()` via `operator*`) have no production
  caller and fall back to the CPU silently.

Action Items:

- [ ] Inline `operator()` in the header (keep the bounds check under a debug macro, or offer an unchecked accessor) and measure the attention loops.
- [ ] Accumulate `sum()` in `double` (or pairwise/Kahan).
- [x] Make `randomize()` take a generator (with TD-368) or guard the static one. Done with TD-368, November 9, 2026: each call seeds its own generator from `adai::next_init_seed()`.
- [ ] Reject negative sizes in the constructor; decide whether to keep the unused `*_gpu()` wrappers.

Location in code: `src/Matrix.cpp` (`operator()`, `sum()`, `randomize()`, `Matrix(int, int)`); `src/Matrix.hpp` (public members); tagged `TODO: See TD-435`.

Files to Modify:

- `src/Matrix.hpp`
- `src/Matrix.cpp`

---

### TD-436: SYCL (`icpx`) Builds Get No `-march` or SIMD Flags

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — code fix landed October 19, 2026 | Build / SIMD | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MatrixSIMD` and verified. The top-level release flags (`-march=native -mtune=native`,
or `-march=x86-64` for `PORTABLE_BUILD`) and `src/CMakeLists.txt`'s `ENABLE_SIMD` block (`-mavx2 -mfma`)
both test `CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"`. oneAPI's `icpx` reports `IntelLLVM`, so the `sycl`
preset gets neither: `build/sycl`'s `compile_commands.json` shows `Matrix.cpp` compiled with only
`-O3 -DNDEBUG -std=gnu++17 -fsycl -fiopenmp`, and `icpx`'s default target defines `__SSE2__` but not
`__AVX2__`. `ADAI_SIMD_AVX2` is therefore undefined and ai-machine's trainer runs every CPU-side `Matrix`
operation on the non-SIMD paths. Timing the real `Matrix.cpp` built with `icpx` (dev machine, 8 threads):
64×512×2048 took 136.9 ms as configured vs 16.7 ms with `-march=native`; 8×512×8000 89.8 vs 12.9 ms (6–8×).
Most products on ai-machine go to the GPU (`cols ≥ 32 && other.cols ≥ 32`), so the impact is on the
CPU-side remainder (small products, element-wise ops, GPU-unavailable fallback); how much of a pass that is
hasn't been measured on ai-machine. The release-mode `-ffast-math -fno-finite-math-only -funroll-loops` set
is skipped for `icpx` too.

Action Items:

- [x] Add `IntelLLVM` to both compiler-ID checks (or use `-march=native` for it), keeping the `-fno-finite-math-only` guard; confirm with the `SIMD:` configure message and `compile_commands.json`.
- [ ] Measure a training pass's wall time on ai-machine before/after.
- [x] Mind that `-march=native` ties the binary to the build host's CPU (TD-437's startup check); build on ai-machine or pick an explicit `-march`.

**Fix (October 19, 2026).** Both compiler-ID checks now include `IntelLLVM`.
- **`-march`:** the `sycl` build gets `-march=x86-64-v3` (AVX2, FMA, BMI: Haswell / Zen and newer), not `-march=native`. It's compiled on the
  dev machine and run on ai-machine, so the build host's CPU says nothing about the target. Owner decision. `PORTABLE_BUILD` still means
  `-march=x86-64` and no intrinsics.
- **SIMD:** the `ENABLE_SIMD` block adds `-mavx2 -mfma`.
- **No `-ffast-math` for `icpx`:** `add_compile_options` also reaches the SYCL kernel sources, where it would allow approximate `exp`/`log` in
  the softmax and cross-entropy kernels. `icpx` 2026's default already contracts and reassociates, and its `-###` shows no no-NaNs flag, so NaN
  checks still work.

Verified:
- **Configure and flags:** `cmake --preset sycl` reports `-O3 -march=x86-64-v3` and "SIMD: AVX2 + FMA intrinsics enabled". `Matrix.cpp` compiles
  with `-O3 -funroll-loops -march=x86-64-v3 -mavx2 -mfma`, and `MatrixGPU_SYCL.cpp` without fast-math.
- **Speed:** `Matrix.cpp` built with `icpx` (8 threads, this machine) took 64×512×2048 from 135.1 to 16.1 ms and 8×512×8000 from 82.0 to
  11.4 ms. The `sycl` `incremental_trainer` builds and contains AVX2 FMA code.
- **Caveat:** an x86-64-v3 binary won't start on a pre-2013 CPU, and with no startup CPU check (TD-437) it fails with SIGILL. Check ai-machine
  with `grep -c avx2 /proc/cpuinfo`.

Context: the owner's rule is that GPU training does all the training math on the GPU, so the CPU `Matrix` paths should matter little to a
GPU pass once TD-468's device-resident optimizer lands. This fix helps the CPU-side remainder in the meantime, and CPU builds.

Location in code: `CMakeLists.txt` (release flags), `src/CMakeLists.txt` (`ENABLE_SIMD` block); tagged `TODO: See TD-436`.

Files to Modify:

- `CMakeLists.txt`
- `src/CMakeLists.txt`

---

### TD-437: SIMD Self-Check Test Never Compiles and No Startup CPU Check

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Build / SIMD | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MatrixSIMD`:
- **Dead test (verified).** `-mavx2 -mfma` are `PRIVATE` to `adai_core`, so `ADAI_SIMD_AVX2` is defined in
  `Matrix.cpp` but not in other files including `Matrix.hpp`. `MatrixSIMDCPUDetect.CompileTimeAVX2MatchesRuntime`
  sits behind `#if defined(ADAI_SIMD_AVX2)` in the test file, so it never compiles in the debug build
  (`--gtest_list_tests` lists only the two always-passing "ReturnsBool" tests).
- **No startup check (by inspection).** `has_avx2()`/`has_fma()` have no production caller, so a
  `-march=native`/`-mavx2` binary run on an older CPU dies with `SIGILL` and no explanation.
- **Incomplete detection (by inspection).** They check only CPUID feature bits, not `OSXSAVE`/`XGETBV`
  (OS support for YMM state); `__builtin_cpu_supports("avx2")` handles this. Matters once TD-434's runtime
  dispatch relies on them.
- **Hygiene.** `ADAI_SIMD_SSE4` is defined but never used; `<immintrin.h>` is pulled into every file
  including `Matrix.hpp`.

Action Items:

- [ ] Expose what `Matrix.cpp` was compiled with (e.g. `adai::simd::compiled_with_avx2()` defined in `Matrix.cpp`) and test that against `has_avx2()`.
- [ ] Check the CPU once at startup (or in `Matrix.cpp`'s static init) and fail with a clear message instead of `SIGILL`.
- [ ] Add the `OSXSAVE`/`XGETBV` check (or use `__builtin_cpu_supports`); drop `ADAI_SIMD_SSE4` or use it.

Location in code: `src/MatrixSIMD.hpp` (`has_avx2()`, `has_fma()`, macros); `tests/matrix_simd_test.cpp` (`CompileTimeAVX2MatchesRuntime`); tagged `TODO: See TD-437`.

Files to Modify:

- `src/MatrixSIMD.hpp`
- `src/Matrix.cpp`
- `tests/matrix_simd_test.cpp`

**Follow-up (October 20, 2026, TD-434).** The multiply's runtime dispatch uses the new
`adai::simd::cpu_supports_avx2_fma()`, which checks OSXSAVE/XGETBV as well as CPUID, so it can't select AVX2
where the OS doesn't save YMM state. `has_avx2()` itself is unchanged, and the self-check test still never
compiles.

---

### TD-447: `MetricsTracker` Printer State Leak, Convergence Off-by-One and Smoothing Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / MetricsTracker | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MetricsTracker`. Production (`ChatbotTrainer`/`IncrementalTrainer`) uses only
`record_epoch()`, `export_csv()` and `is_overfitting()`; nothing here affects training.
- **Printer state leak (verified).** `print_summary()`/`print_history()` set `std::fixed` and `setprecision(4)` on
  `std::cout` and never restore them: `0.123456789` printed `0.123457` before and `0.1235` after.
- **Convergence off by one (verified).** `is_converging(window)` returns `false` unless `history_.size() >= window + 1`;
  five identical epochs with `window = 5` reported not converging.
- **Smoothing gaps (verified).** `MetricsTracker(0)` gives a `-nan` smoothed loss (0/0); a negative window wraps through
  the `int`/`size_t` comparison; epochs with `val = 0` are skipped, so `get_smoothed_validation_loss()` is shorter than
  the history (2 vs 3) and its indexes don't match epochs.
- **By inspection.** `is_overfitting()` checks only the final epoch against a fixed 0.5-nat (≈1.65× perplexity) gap
  and inherits TD-267's low validation bias; `calculate_moving_average()` is dead; the header uses an `#ifndef` guard
  and `std::cout`/`std::cerr` (CLAUDE.md conventions); CSV epochs are 0-based while the metrics server's are 1-based.

Action Items:

- [ ] Save and restore stream flags in the printers (or route them through `Logger`).
- [ ] Fix the `is_converging()` size check; validate the smoothing window; keep validation smoothing aligned with epochs (NaN placeholder or paired epoch numbers).
- [ ] Remove `calculate_moving_average()`; switch to `#pragma once`; document the 0-based CSV epochs.

Location in code: `src/MetricsTracker.hpp`; tagged `TODO: See TD-447`.

Files to Modify:

- `src/MetricsTracker.hpp`
- `tests/metricstracker_test.cpp`

---

### TD-453: Hand-Written JSON Duplicated Across MNS and Metrics Code

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Core / JSON | October 14, 2026 | 4-8 hours |

Description:
Found while tracing `MnsJsonHelpers`, by inspection. Hand-written JSON escaping and string-search parsing is implemented
separately in at least `MnsJsonHelpers.hpp`, `MnsCliCommands.cpp`, `ModelNameClient.cpp` (`json_escape_client()`,
`find_string_end_client()`, explicitly "duplicated here"), `ModelNameService.cpp`, `MetricsPushClient.cpp`
(`escape_json_string()`) and `TrainingMetricsAPI.cpp` (`escape_json()`), each with different escaping coverage and
parsing rules. That's the root of several filed defects: non-finite numbers emitted and accepted (TD-443), incomplete
escaping (TD-449, TD-452), and `ChatbotAPI`'s parser rejecting `]` (TD-226). The repository already vendors
`external/nlohmann` and uses it in `ModelSerializer`, `ChatbotAPI` and `DataFetcher`.

**Progress (October 25, 2026).** `TrainingMetricsAPI`'s push handlers now parse request bodies with nlohmann/json (TD-527), and
`MetricsPushClient` writes numbers through one shared helper (TD-443). The MNS files and the API's response writers are still
hand-written.

**Progress (November 10, 2026).** `MnsJsonHelpers.hpp` (the MNS Manager GUI's) parses and escapes with nlohmann/json
(TD-452), and the GUI builds its register body with `MnsCliCommands`' builder rather than its own. Still hand-written:
`MnsCliCommands.cpp`, `ModelNameClient.cpp`, `ModelNameService.cpp`, `MetricsPushClient.cpp` and the metrics API's
response writers.

Action Items:

- [ ] Move the MNS (server, client, CLI, GUI) and metrics (server, push client) request/response code to `nlohmann/json`, one file at a time behind the existing tests.
- [ ] Delete the per-file escape/scan helpers once their callers are migrated.

Location in code: the helpers listed above; tagged `TODO: See TD-453`.

Files to Modify:

- `src/MnsJsonHelpers.hpp`, `src/MnsCliCommands.cpp`, `src/ModelNameClient.cpp`, `src/ModelNameService.cpp`, `src/MetricsPushClient.cpp`, `src/TrainingMetricsAPI.cpp`

---

### TD-459: `ModelNameClient` Validation, Unescaping and Slow-Failure Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | MNS / Client | October 14, 2026 | 2-4 hours |

Description:
Found while tracing `ModelNameClient`, by inspection:
- **Unvalidated architecture.** `ChatbotAPIServer` and `IncrementalTrainingTool` copy `get_architecture()`'s six fields into their
  config without checks. A missing field parses as 0 (`json_int_client()`), and a `MODEL_NAME` pointing at a standalone
  `encoder`/`world_model` record yields `num_decoder_layers = 0`, building a decoderless model instead of failing clearly.
- **No unescaping.** `json_string_client()` returns JSON escapes intact; `resolve_model().artifact.path` (used to load weights)
  would keep `\\` for a backslash.
- **Slow failure.** Each call can take 3 × `timeout_ms` + 1.2 s (~16 s at the default 5 s) when MNS doesn't answer; startup
  makes several in sequence, so a routable-but-down MNS delays `chatbot_api_server`/trainer startup by about a minute before
  the local-config fallback.
- **Minor.** No URL encoding of names/roles/query values; `https://` sets port 443 but uses a plain `httplib::Client` (TD-444/TD-451);
  a new client per attempt.

Action Items:

- [ ] Validate architecture (all fields > 0, kind compatible with the caller) and fail clearly otherwise.
- [ ] Unescape JSON strings (or adopt TD-453's library).
- [ ] Fail fast after the first unanswered call in a startup sequence (or use one shorter startup timeout); URL-encode path/query parts.

Location in code: `src/ModelNameClient.cpp` (`json_string_client()`, `get_architecture()`, `list_models()`); `src/ChatbotAPIServer.cpp` (architecture copy); tagged `TODO: See TD-459`.

Files to Modify:

- `src/ModelNameClient.cpp`
- `src/ChatbotAPIServer.cpp`
- `src/IncrementalTrainingTool.cpp`

---

### TD-462: `ModelNameService` Dead Datasets Proxy, Admin-Settings Race and Swallowed Persistence Errors

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | MNS / Server | October 14, 2026 | 2-4 hours |

Description:
Found while tracing `ModelNameService`:
- **Compact JSON only (verified).** The request parsers search for `"key":"`/`"key":` without whitespace; a standard spaced body
  (`{"model_name": "spaced", "role": "chatbot"}`, as Python's `json.dumps()` emits) returned 400 "model_name required". In-repo
  clients send compact JSON. `json_string()` also turns `\u00e9` into `u00e9`. (Umbrella fix: TD-453.)
- **Dead datasets proxy (by inspection).** `handle_datasets()` filters `registry_server`'s `/history` by `model_id`, but
  `RemoteTransport::mark_trained()` never sends one, so the result is always empty; it also uses MNS's global `registry_group_`
  rather than the model's own `run_group`, and has no TD-202 kind segment. No client calls it.
- **Data race (by inspection).** `handle_admin_put_config()` writes `registry_url_`/`registry_group_` without a lock while
  `handle_datasets()` reads them on another httplib thread (undefined behaviour).
- **Swallowed persistence errors (by inspection).** `persist_model()`, `persist_roles()` and `handle_delete()` log or ignore SQLite
  failures and still return success, so memory and disk can diverge until restart.
- **Stale role (by inspection).** Promoting a model to a new role rewrites its `role` but leaves the old role's mapping to it.

Action Items:

- [ ] Fix or remove the datasets proxy (record `model_id` on mark-trained, use the model's `run_group` and kind). Recording `model_id`
  was also TD-498's item, moved here November 21, 2026: nothing sets `DataVersion::model_id` and `RemoteTransport::commit_trained()`
  doesn't send it (its `TODO: See TD-462`), so this proxy is its only consumer.
- [ ] Guard the registry settings with a mutex.
- [ ] Return 500 (and roll back the in-memory change) when persistence fails.
- [ ] Remove the old role mapping when a model is promoted to a different role.

Location in code: `src/ModelNameService.cpp` (`json_string()`, `handle_datasets()`, `handle_admin_put_config()`, `persist_model()`, `handle_promote()`); tagged `TODO: See TD-462`.

Files to Modify:

- `src/ModelNameService.{hpp,cpp}`
- `src/RegistryTransport.cpp`

---

### TD-464: `mns_server` Startup Crash, Early-SIGTERM Hang and Fixed Log Level

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open — early-SIGTERM hang fixed November 22, 2026 | MNS / Server | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `ModelNameServiceServer`:
- **Crash (verified).** `main()` catches its own `create_directories()` failure and only warns, but `ModelNameService::start()`
  repeats it on the server thread with no handler: `--data-dir /proc/mnsx` aborted with `terminate called after throwing an
  instance of 'std::filesystem::…::filesystem_error'` (exit 134, core dumped).
- **Early-SIGTERM hang (verified).** If SIGTERM arrives before the server thread reaches `listen()`, both `stop()` calls (the
  handler's and `main()`'s) happen before there's anything to stop; the thread then listens and `main()` blocks in `join()`.
  In 30 trials sending SIGTERM 0–90 ms after launch, 1 was still running 2 s later and needed SIGKILL. Same class as TD-248. The
  handler also calls `httplib::Server::stop()` from signal context (not async-signal-safe; TD-249's class).
- **By inspection.** `LOG_LEVEL` is ignored (`Logger::init(INFO)`, as TD-431); the file comment and `--help` omit
  `PUT …/progress`, `PUT …/run_group` and `POST …/link-world-model`.

Action Items:

- [ ] Fail startup cleanly when the data directory can't be created (catch in `start()` or check in `main()` and exit 1).
- [x] Only set a flag in the signal handler; after `stop()`, re-check the flag once `listen()` has started (or use `wait_until_ready()` before stopping).
      **Done November 22, 2026:** the handler only sets `shutdown_requested`, and `ModelNameService::start()`/`stop()` use the
      `starting_`/`stop_requested_` handshake of `TrainingMetricsAPI` (TD-528) and `TrainerAdminAPI` (TD-582), so `main()`'s `stop()` ends
      a start that hasn't reached `listen()`; a stop during startup exits 0. Found again as `ModelNameServiceStartTests` and
      `MnsProgressTests` hanging to ctest's timeout under a loaded parallel run. `mns_server` sent `SIGTERM` 0–120 ms after launch: 0 of
      70 hung (1 of 30 before). Test: `ModelNameServiceStartTest.StopBeforeOrDuringStartAlwaysEndsStart`, which fails on the old code.
- [ ] Honour `LOG_LEVEL`; update the endpoint lists.

Location in code: `src/ModelNameServiceServer.cpp` (`signal_handler()`, `main()`); `src/ModelNameService.cpp` (`start()`); tagged `TODO: See TD-464`.

Files to Modify:

- `src/ModelNameServiceServer.cpp`
- `src/ModelNameService.cpp`

---

### TD-466: `ModelSerializer` Claims T5 Compatibility It Doesn't Have and Skips Input Validation

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open (decision: make it T5-faithful) | Models / ModelSerializer | October 14, 2026 | 16-24 hours |

Description:
Found while tracing `ModelSerializer`, by inspection:
- **Not T5.** Tensor names and `config.json` (`architectures: T5ForConditionalGeneration`, `model_type: t5`) claim HuggingFace T5, but
  ADAI uses sinusoidal positions (T5: learned, bucketed relative attention bias, which `transformers` would initialize randomly),
  LayerNorm with bias and mean-centering (T5: weight-only RMSNorm), a GELU FFN with biases (config says `feed_forward_proj: "relu"`;
  T5 linears have no biases), 1/√d_k attention scaling (T5: none), separate encoder/decoder embeddings (T5: one shared embedding) and
  an untied, biased LM head. Exports don't load meaningfully into `transformers`; importing a real T5 checkpoint always fails because
  `require()` demands bias tensors T5 doesn't have. The header's "unknown keys ignored (e.g. T5 position biases)" case is unreachable.
  (Embeddings aren't scaled in the forward pass, which already matches T5.)
- **Validation.** `read_safetensors()` ignores `dtype` (F16/BF16 fail later as "size mismatch in from_flat"); a huge/corrupt header
  length or `data_offsets` with `end < start` become giant allocations; offsets aren't checked against the file size. Import sets
  weights as it goes, so a missing tensor leaves the model half-overwritten.
- **Hygiene.** `json_escape()` is a no-op and unused; the files are tagged `stable` with no production caller.

**Decision (October 14, 2026):** owner chose to make the format **T5-faithful** (an export loads in `transformers`'
`T5ForConditionalGeneration` and produces the same outputs; real T5 checkpoints import), rather than relabelling it as an
ADAI-specific format. Only part of the gap can be closed in the serializer; the rest is a model-architecture change:

- *Mappable at export/import, no model change:* fold the 1/√d_k attention scale into the exported `q.weight` (and unfold on
  import); `tie_word_embeddings: false` for the untied LM head; `feed_forward_proj: "gelu"` (non-gated GELU) instead of `"relu"`;
  correct special-token IDs and `d_kv`.
- *Not representable in T5, so they require a T5-style architecture variant of `EncoderDecoderModel`:* RMSNorm without bias
  (`T5LayerNorm`) instead of `LayerNorm`; bias-free Q/K/V/O, FFN and LM head; bucketed relative position bias (per head, computed in
  layer 0 and shared) instead of sinusoidal positional encoding; one embedding shared by encoder and decoder; and, if the LM head is
  tied, T5's `d_model^-0.5` rescale before it. Existing checkpoints can't be converted exactly, so this is a new variant (selected
  per model, e.g. an MNS architecture flag, since architecture is immutable) trained from scratch or imported from a real T5
  checkpoint. Optionally support T5 v1.1's gated GELU (`wi_0`/`wi_1`).
- For the current (non-T5) architecture, TD-465's fix still applies: export the decoder embedding separately (done November 12,
  2026). Until the variant
  exists, the current export should stop claiming `T5ForConditionalGeneration`.

Action Items:

- [ ] Implement the export-side mappings above; stop labelling non-T5 models as `T5ForConditionalGeneration`.
- [ ] Add the T5-style architecture variant (RMSNorm, bias-free linears, relative position bias, shared embedding) and its MNS/config selection.
- [ ] Map T5 tensor names both ways for the variant, including `relative_attention_bias`; import a real `t5-small` checkpoint.
- [ ] Verify against `transformers` (encoder output and logits within tolerance for the same input); add that comparison as a test or script (needs `transformers`/`torch` in the dev environment).
- [ ] Validate dtype (accept F16/BF16 with conversion), header size and offsets; load into temporaries and apply only after all tensors validate.
- [ ] Remove `json_escape()`; reconsider the `stable` tag.

Location in code: `src/ModelSerializer.{hpp,cpp}` (class comment, `config.json` writer, `read_safetensors()`, `import_safetensors()`); the model classes for the variant; tagged `TODO: See TD-466`.

Files to Modify:

- `src/ModelSerializer.{hpp,cpp}`
- `src/EncoderDecoderModel.{hpp,cpp}`, `src/LayerNorm.*` (or a new RMSNorm), `src/MultiHeadAttention.*`, `src/CrossAttention.*`, `src/FeedForward.*`, `src/LanguageModelHead.*`, positional encoding
- `src/ModelNameService.cpp`, `src/ModelNameClient.cpp` (architecture variant flag)
- `tests/model_serializer_test.cpp`

---

### TD-468: GPU Training Round-Trips Every Gradient and Weight Through a 3× Slow CPU Optimizer

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — device-resident optimizer landed October 19, 2026 | Core / Optimizer | October 14, 2026 | 8-16 hours |

Description:
Found while tracing `Optimizer`:
- **Slow update loops (verified).** `step_sgd/…/step_adamw()`, the norm and clipping loops all index through `Matrix::operator()`
  (out-of-line, bounds-checked, TD-435); an AdamW step makes about six accessor calls per parameter. On 5.24M parameters (20 × 512²,
  `-O3 -march=native`, 8 threads) `Optimizer::step()` took **45.0 ms vs 14.2 ms** for the same AdamW over raw row pointers (3.2×);
  `get_gradient_norm()` adds 6.4 ms.
- **GPU training uses it on the host (by inspection, with an estimate).** The optimizer is CPU-only, so each optimizer step in
  `ChatbotTrainer`'s GPU path does `gpu_download_grads()` (every gradient to the host), `Optimizer::step()` on the CPU, then
  `gpu_sync_weights()` (every weight back to the device). For `config.trainer.conf`'s fallback architecture (d_model 768, d_ff 3072,
  8 + 24 layers ≈ 283M parameters before embeddings; MNS's record may differ) that's an estimated ~2.4 s of CPU AdamW at the measured
  rate (~0.8 s with raw pointers) plus ~2.3 GB of host↔device copies per optimizer step (every 32 samples with
  `GRADIENT_ACCUMULATION_STEPS=32`), on a SYCL build compiled without SIMD flags (TD-436). Not yet measured on ai-machine.

Action Items:

- [ ] Rewrite the update, norm and clip loops over raw row pointers (vectorizable); re-measure.
- [ ] Measure the per-step download/step/upload time on ai-machine.
- [x] Add a GPU-resident optimizer (AdamW kernels for CUDA and SYCL) keeping weights, gradients and moments on the device, so GPU training skips the round trip; keep the CPU optimizer for CPU training.

**Fix (October 19, 2026): the device-resident optimizer.** This follows the owner's rule that GPU training does all the training math on the GPU
and moves no data without a critical reason.
- **New kernels:** `src/GPUOptimizer.{hpp,cpp}` plus two kernels in both backends (`MatrixGPU.cu`, `sycl/MatrixGPU_SYCL.cpp`):
  - `matrix_sum_squares_accumulate_gpu()`: a single pass that adds into a device slot, with no temporary buffer and no host wait;
  - `optimizer_update_gpu()`: the CPU `Optimizer`'s SGD, momentum, Adam and AdamW formulas, in the same order, with a gradient scale.
- **Binding:** every layer now reports its device weight and gradient buffers (`gpu_collect_parameters()`), and `GPUOptimizer` binds the CPU
  optimizer's groups to them by host-weight identity. The moments live on the device.
- **Per step:** `ChatbotTrainer`'s GPU path reduces every group's gradient on the device and downloads **one float per group**. Those are the
  squared norms that the NaN check, the clip, the zero-gradient skip and the per-layer norm metrics need. The TD-272 window rescale and the
  clip factor are folded into the update's gradient scale. The per-step `gpu_download_grads()` and `gpu_sync_weights()` are gone.
- **While training:** the device weights are the live copy. `sync_host_from_device()` copies weights and moments back to the host only
  before a checkpoint (`save_to()`, the best snapshot, `save_training_state()`), before the per-epoch quality-metrics `clone()`, and at the end
  of `train()`. `train()` uploads the host weights when it binds, since the host copy is authoritative between passes.
- **Host groups:** groups with no device mirror are stepped on the host with no transfer. These are the two token embeddings, and the gated
  world-model/hippocampal sub-layers when attached (TD-342). The embeddings are TD-547.

Verified on the oneAPI CPU SYCL device (new opt-in `ADAI_SYCL_ALLOW_CPU_DEVICE=1`, TD-394), with the real kernels and classes:
- **`gpuOptimizerTests` (6):**
  - each rule matches the CPU `Optimizer` over three scaled steps (≤ 1e-5 relative);
  - the moments round-trip through the CPU optimizer;
  - a real model's device step matches the old download-and-CPU-step path through its logits (36 device groups, 2 host).
- **Existing suites on the device path:** `chatbottrainerTests` (83), `encoderdecoderTests` (90) and `incrementaltrainerTests` (58) all pass
  with every training test on the device path. The `IncrementalTrainer` fixture now initializes the GPU, as `incremental_trainer` does.
- **Real binary:** a SYCL `incremental_trainer` trained a pass on the CPU device (validation loss 4.09 → 3.94). A second pass resumed its
  checkpoint and restored its optimizer state at global step 18.

CUDA is compile-checked only. Still to do: the first action item (the CPU loops, which now matter only for CPU training), and measuring a
step on ai-machine.

Location in code: `src/Optimizer.cpp` (update/norm/clip loops); `src/ChatbotTrainer.cpp` (`gpu_download_grads()`/`gpu_sync_weights()` around `optimizer->step()`); tagged `TODO: See TD-468`.

Files to Modify:

- `src/Optimizer.{hpp,cpp}`
- `src/ChatbotTrainer.cpp`
- `src/gpu/MatrixGPU.{hpp,cu}`, `src/gpu/sycl/MatrixGPU_SYCL.{hpp,cpp}`

**Follow-up (October 19, 2026, TD-547).** The token embeddings are now device groups too, so a model without the gated paths has no
host groups left, and GPU training no longer zeroes host gradients every window.

---

### TD-473: `Profiler` Keeps Every Timing Forever, Sorts Under Its Lock and Uses the Wall Clock

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Profiler | October 15, 2026 | 1-2 hours |

Description:
Found while tracing `PerformanceProfiler`:
- **Unbounded samples (verified).** `ChatbotAPI::generate_response()` is always wrapped in `PROFILE_SCOPE` (`--profile` only gates
  `GET /admin/profile`'s output), and `ProfileStats::timings` keeps every sample: after 2,000,000 simulated requests it held 2M entries
  (16 MB), 8 bytes per request for the life of `chatbot_api_server`.
- **Sorting under the lock (verified).** `Profiler::get_stats()` copies and sorts the vector (`compute_median()`) while holding the profiler's
  mutex, which every in-flight `generate_response()` needs for `start()`/`stop()`: 72 ms at 2M samples. `handle_profile()` then sorts two more
  copies for p95/p99 (107 ms).
- **Wall clock (verified).** `Timer` uses `std::chrono::high_resolution_clock`, which in libstdc++ is `system_clock` (`is_steady == false`); an
  NTP step or manual change during a request records a wrong or negative duration.
- **By inspection.** `ScopedTimer`, `print()`, `print_all()`, `compare()` and `Benchmark::compare()` write to `std::cout` (test-only); percentiles
  use the floor index; `compare()` divides by `optimized.mean_time` unchecked.

Action Items:

- [ ] Use `std::chrono::steady_clock`.
- [ ] Replace the unbounded vector with a fixed-size reservoir or log-bucket histogram (constant memory, no sorting), and copy only summary values under the lock.
- [ ] Guard `compare()`'s division; route output through `Logger`.

Location in code: `src/PerformanceProfiler.hpp` (`Timer`, `ProfileStats`, `Profiler::get_stats()`); tagged `TODO: See TD-473`.

Files to Modify:

- `src/PerformanceProfiler.hpp`
- `src/ChatbotAPI.cpp` (`handle_profile()`)
- `tests/inference_optimization_test.cpp`

---


### TD-477: Timestamp Parsing Turns Bad Input Into 1899, Ignores Offsets and Misreads Fractions

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Platform / PortableTime | October 17, 2026 | 1-2 hours |

Description:
Found while tracing `PortableTime` and verified by running `strptime_utc()` through the callers' exact logic (`TrainingMetricsAPI::parse_iso8601()`,
`SQLiteMetricsDatabase::parse_timestamp()`, `PostgresMetricsDatabase::parse_timestamp()`), which all do
`tm{}; if (strptime_utc(...)) { if (*end == '.') ms = atoi(end + 1); } return timegm_utc(&tm) + ms;`:
- **Failure isn't reported.** `not-a-date` and a date-only `2026-10-15` both became `1899-12-31T00:00:00` (the zeroed `tm`). For
  `GET …/metrics/db-history`, a bad `from` silently means "since 1899" (everything) and a bad `to` "until 1899" (nothing), with HTTP 200.
- **Offsets ignored.** `2026-10-15T12:00:00+02:00` parsed as 12:00 UTC (two hours wrong).
- **No range checks.** `2026-13-45T25:61:99` became `2027-02-15T02:02:39` (`timegm()` normalizes); `%d` also accepts whitespace and signs, and has
  undefined behaviour on overflow.
- **Fractions.** `atoi()` reads the digits as an integer: `.12` → 12 ms, `.1` → 1 ms, `.123456` → 123,456 ms (+2 min). PostgreSQL prints its
  `TIMESTAMPTZ` columns without trailing zeros (`.12`), so the Postgres backend's `recorded_at`/`created_at`/`last_update_at` read back up to ~0.9 s early.
  (SQLite stores a fixed `.mmm` and is unaffected.)

Action Items:

- [ ] Add `parse_iso8601_utc()` returning `std::optional<time_point>`: strict digits and ranges, fractional seconds by digit position, `Z`/`±HH:MM` offsets, optional date-only form.
- [ ] Replace the three callers' copies; make `db-history` return 400 for an unparsable bound.
- [ ] Unit tests for the table above.

Location in code: `src/PortableTime.hpp` (`strptime_utc()`); `src/TrainingMetricsAPI.cpp`, `src/SQLiteMetricsDatabase.cpp`, `src/PostgresMetricsDatabase.cpp` (`parse_*`); tagged `TODO: See TD-477`.

Files to Modify:

- `src/PortableTime.hpp`
- `src/TrainingMetricsAPI.cpp`, `src/SQLiteMetricsDatabase.cpp`, `src/PostgresMetricsDatabase.cpp`

---

### TD-480: `PostgresMetricsDatabase` One-Time Schema Bootstrap, Reconnects Under Lock and Unsafe Retries

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Metrics / Database | October 17, 2026 | 2-4 hours |

Description:
Found while tracing `PostgresMetricsDatabase`, by inspection:
- **One-time bootstrap.** `bootstrap_schema()` runs only in the constructor and only if `pool_[0]` connected; if PostgreSQL is down when
  `metrics_api_server` starts, later reconnects succeed but a fresh database never gets its tables, so every statement fails until restart.
- **Blocking under the lock.** `acquire_connection()` calls `ensure_connection()` (a blocking `PQconnectdb()`) while holding `pool_mutex_`, and
  `execute_with_retry()` reconnects under it too, so one slow reconnect stalls every caller; no `connect_timeout`/`statement_timeout` unless the URL sets one.
- **Unsafe retries.** An insert whose result was lost with its connection (committed but unacknowledged) is re-executed, duplicating the row; non-retryable
  errors (constraint, syntax) are retried with full back-off.
- **Archive collision.** `archive_session()` inserts under `archived_key` with plain `INSERT`; an existing row with that key aborts the transaction (unlikely:
  keys carry a millisecond timestamp and counter).

Action Items:

- [ ] Bootstrap the schema on the first successful connection (any time), not just at construction.
- [ ] Reconnect outside `pool_mutex_`; set `connect_timeout` and `statement_timeout` defaults.
- [ ] Classify errors (retry only connection failures); make inserts idempotent where retried (e.g. client-side IDs with `ON CONFLICT DO NOTHING`).

Location in code: `src/PostgresMetricsDatabase.cpp` (constructor, `acquire_connection()`, `execute_with_retry()`, `archive_session()`); tagged `TODO: See TD-480`.

Files to Modify:

- `src/PostgresMetricsDatabase.{hpp,cpp}`

---

### TD-481: PPO Advantages Are Normalized Within Each Response

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open (deferred with RLHF) | RLHF / PPOOptimizer | October 17, 2026 | 1-2 hours |

Description:
Found while tracing `PPOOptimizer` and verified. `compute_gae()` normalizes each trajectory's (one response's) advantages to mean 0 and unit standard
deviation, and `RLHFTrainer::apply_policy_gradient()` weights each token's log-probability gradient by those advantages. With the reward on the last
token (as `RLHFTrainer` assigns it) and value estimates of 0:
- reward +1 → advantages −1.370, −0.727, −0.043, +0.684, +1.457; reward **+100 → identical**;
- reward −1 → the same with signs flipped;
- a one-token response with reward +5 → advantage **0.000**.

So the update ignores the reward's magnitude, pushes **down** the early tokens of a well-rewarded response (and up those of a badly rewarded one), and
learns nothing from one-token responses. RLHF has no production entry point (owner, October 17, 2026: its use will be planned later).

Action Items:

- [ ] Normalize advantages across a batch of rollouts (or not at all, relying on the value baseline), not within one response.
- [ ] Test: a positive reward gives non-negative advantages for every token under a zero baseline; magnitude scales the update.

Location in code: `src/PPOOptimizer.hpp` (`compute_gae()`); `src/RLHFTrainer.cpp` (use of `compute_advantages()`); tagged `TODO: See TD-481`.

Files to Modify:

- `src/PPOOptimizer.hpp`
- `src/RLHFTrainer.cpp`
- `tests/phase5_test.cpp`, `tests/rlhftrainer_test.cpp`

---

### TD-482: `PPOOptimizer` Never Updates the Policy and Its Ratio Is Always 1

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open (deferred with RLHF) | RLHF / PPOOptimizer | October 17, 2026 | 2-4 hours |

Description:
Found while tracing `PPOOptimizer`:
- **No policy update (by inspection).** `update()` computes the clipped surrogate `−min(r·A, clip(r)·A)` and an approximate KL, trains only its
  `ValueFunction`, and returns the loss; nothing differentiates it into the policy. The real update is `RLHFTrainer::apply_policy_gradient()`, an
  unclipped policy gradient.
- **Ratio always 1 (verified).** `RLHFTrainer` passes a callback returning the rollout's own old log-probs, so `r = 1` on every call: no clipping, KL ≡ 0
  (the `1.5 × kl_target` early stop never fires) and a reported loss of 0.000000 (`−mean(A)` of normalized advantages).
- **Unused configuration (by inspection).** `entropy_coef`, `value_coef`, `max_grad_norm` and the private `clip_gradients()` are never used;
  `reward_model_` is stored but never read; minibatches aren't shuffled; `ValueFunction` uses the unseeded C `rand()`.
- The header and the replaced API doc promise "stable policy updates via the clipped surrogate objective" and KL monitoring.

Action Items:

- [ ] Apply the clipped objective to the policy (compute new log-probs with the live model each PPO epoch and backpropagate the clipped loss, with KL early stopping), or rename this to what it is (value function + GAE) and make `RLHFTrainer`'s update the documented REINFORCE-with-baseline.
- [ ] Wire `entropy_coef`/`max_grad_norm`/`value_coef` or remove them; drop the unused `reward_model_`; seed/shuffle properly.

Location in code: `src/PPOOptimizer.hpp` (`update()`, `PPOConfig`); `src/RLHFTrainer.cpp` (the `log_prob_fn` callback, `apply_policy_gradient()`); tagged `TODO: See TD-482`.

Files to Modify:

- `src/PPOOptimizer.hpp`
- `src/RLHFTrainer.{hpp,cpp}`

---

### TD-483: `Quantization` Asymmetric Range, Mode-Less Files and Unpacked INT4

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Advanced / Quantization | October 17, 2026 | 2-4 hours |

Description:
Found while tracing `Quantization.hpp` (no production use: tests and examples only; a quantization tool is deferred under TD-014):
- **Asymmetric range (verified).** `calibrate()` clamps `zero_point` to `[qmin, qmax]` instead of widening the calibration range to include 0. For data
  in [5, 10], `zero_point = −255` was clamped to 0, every value quantized to 255, and the round trip returned 5.000 for all of 5, 6, 7.5, 9, 10. Any all-positive
  (e.g. post-ReLU) or all-negative tensor collapses to one value.
- **Mode not saved (verified).** `QuantizedMatrix::save()` stores scale/zero point/qmin/qmax but not the mode, and `dequantize()` decides signed vs unsigned
  decoding from the caller's `Quantizer`. An `ASYMMETRIC_INT8` matrix of 0/5/15/20, reloaded and dequantized with a default `Quantizer`, gave 0, 5.02, −5.10,
  −0.08 (TD-075's corruption).
- **INT4 unpacked (verified).** One value per byte: `memory_reduction()` reported 3.98× for `SYMMETRIC_INT4`; the header and the replaced API doc promise 8×.
- **By inspection.** Per-channel quantization is described but not implemented; `CalibrationMethod::MSE` silently falls back to min/max; `load()` checks no
  reads and doesn't validate `size` against `rows × cols`; `memory_reduction()` uses `int` byte counts; `print_quantization_stats()` writes to `std::cout`.

Action Items:

- [ ] Asymmetric calibration: `min = min(min, 0)`, `max = max(max, 0)` (standard), then compute `zero_point` without clamping away the range.
- [ ] Decode from the saved `qmin`/`qmax` (or save the mode); validate `load()`.
- [ ] Pack INT4 two per byte (or correct the claims); implement or remove per-channel and MSE calibration.

Location in code: `src/Quantization.hpp` (`calibrate()`, `decode_stored_value()`, `QuantizedMatrix::save()/load()/memory_reduction()`, header comment); tagged `TODO: See TD-483`.

Files to Modify:

- `src/Quantization.hpp`
- `tests/quantization_test.cpp`

---

### TD-499: `RewardModel` Unchecked Loading, Invalid Widths and Non-He Initialization

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | RLHF / RewardModel | October 17, 2026 | 1-2 hours |

Description:
Found while tracing `RewardModel.hpp` (see [RewardModel.md](../reference/source/RewardModel.md); used only by `RLHFTrainer`, which has no production
entry point). The backward pass is correct (finite differences: median relative error 1e-4) and the model learns a toy preference task (200/200).
- **Unchecked `load()` (verified):** a file truncated to half loaded without error, and the next `predict_reward()` threw `Matrix dimensions incompatible
  for multiplication: [1x5] * [6x5]`; a `(6, {5, 4, 1})` model loading a `(10, {3, 1})` file silently became the file's architecture (as TD-432); a negative
  layer count throws `std::length_error`. No check that each layer's rows/cols match `layer_dims`.
- **Invalid widths (verified):** `RewardModel(4, {0, 1})` constructs and always returns 0; `input_dim` isn't validated.
- **Initialization (verified):** uniform on ±√(2/fan_in) has a third of He's variance (std 0.026 vs 0.045 at fan-in 1,000), despite the "He" comment;
  unseeded global `rand()` (TD-368's class).
- **By inspection:** `backward()` recomputes the forward pass `train_on_batch()` just ran (four forwards per pair; 7.0 s per 100-pair batch at
  `(1536, {512, 256, 128, 1})`); no regularization or clipping; `forward()` writes caches, so it isn't thread-safe.

Action Items:

- [ ] Check every read in `load()`, validate counts and per-layer shapes, and reject a file whose architecture differs from the constructed one.
- [ ] Validate `input_dim` and every width > 0.
- [ ] Use a seedable normal He initialization; reuse the forward caches in `backward()`.

Location in code: `src/RewardModel.hpp` (constructor, `initialize_weights()`, `backward()`, `load()`); tagged `TODO: See TD-499`.

Files to Modify:

- `src/RewardModel.hpp`
- `tests/phase5_test.cpp`

---

### TD-500: RLHF Rollouts Never Explore and Updates Train on a Different Sequence

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | RLHF / RLHFTrainer | October 17, 2026 | 1-2 hours |

Description:
Found while tracing `RLHFTrainer.cpp` (see [RLHFTrainer.md](../reference/source/RLHFTrainer.md); no production entry point). Verified with the tests'
tiny model:
- **No exploration:** `run_iteration()` calls `generate_response_with_strategy(prompt, max_len, strategy, temperature)` without `num_beams`, so the
  header default (4) forces beam search (TD-261). With `"sampling"` at temperature 1.0, six rollouts of one prompt gave 1 distinct response; with
  `num_beams = 1`, 6. The policy gradient only ever sees the current mode. The tests use `"greedy"`, so they don't notice.
- **Different sequence:** generation encodes the prompt without special tokens, but the old-log-prob and update `forward()` calls use
  `encode(prompt, true)` (3 encoder tokens instead of 1 for `"hello"`; response log-probability −69.2 → −68.2; the TD-364 mismatch). The response is decoded
  and re-encoded with `encode(response, true)`, so `response_tokens` begin with BOS (id 2) and end with EOS (id 3): actions generation never takes (EOS
  is also added to responses cut off at `max_response_length`) receive advantage-weighted gradient, and with TD-361 the first row trains "BOS after BOS".
  Re-tokenizing decoded text can also differ from the sampled IDs (TD-224).

Action Items:

- [x] Pass `num_beams = 1` (with TD-261's fix) and test that sampling rollouts differ. (Done with TD-261, October 25, 2026;
  `ExplicitStrategiesIgnoreNumBeams` checks that sampling with `num_beams = 5` gives distinct answers.)
- [ ] Return the generated token IDs from generation and train on exactly those, with the same encoder input generation used (no special tokens).

Location in code: `src/RLHFTrainer.cpp` (`run_iteration()`: generation call, prompt/response encoding); tagged `TODO: See TD-500`.

Files to Modify:

- `src/RLHFTrainer.cpp`
- `src/EncoderDecoderModel.{hpp,cpp}` (a token-returning generation entry point)
- `tests/rlhftrainer_test.cpp`

---

### TD-501: `RLHFTrainer` Lacks a KL Penalty, Batch Updates and Efficient Value States

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | RLHF / RLHFTrainer | October 17, 2026 | 4-8 hours |

Description:
Found while tracing `RLHFTrainer.cpp` (by inspection):
- **No KL penalty** against a frozen reference (SFT) policy, the standard RLHF guard against reward hacking and drift; with TD-482's ratio ≡ 1 there is
  no step-size control at all.
- **One optimizer step per prompt** (`apply_policy_gradient()` steps inside the per-prompt loop), so updates are single-sample REINFORCE; rewards
  aren't normalized across the batch (TD-481 normalizes within a response instead).
- **Value states cost one encoder pass per response token** over a growing context (64 passes for a 64-token response), on top of beam generation,
  scoring and two forwards.
- `set_training(true)` is left on after the update; TD-341's gates aren't optimizer parameters, so they don't learn here either.

Action Items:

- [ ] Add a KL penalty (per-token reward shaping against a frozen reference model's log-probs).
- [ ] Accumulate gradients over the batch and step once; normalize rewards/advantages across the batch (with TD-481).
- [ ] Derive value states from one encoder/decoder pass; restore the previous training mode.

Location in code: `src/RLHFTrainer.cpp` (`run_iteration()` value-state loop, `apply_policy_gradient()`); tagged `TODO: See TD-501`.

Files to Modify:

- `src/RLHFTrainer.{hpp,cpp}`
- `src/PPOOptimizer.hpp`
- `tests/rlhftrainer_test.cpp`

---

### TD-504: `SIGReg` Frequency Grid, NaN Inputs and Repeated Projections

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | World Model / SIGReg | October 17, 2026 | 1-2 hours |

Description:
Found while tracing `SIGReg.{hpp,cpp}` (by inspection): four equally weighted frequencies (0.5, 1, 1.5, 2) replace the paper's/Epps–Pulley's
Gaussian-weighted integral, limiting sensitivity to distribution shape; non-finite embeddings aren't detected (a NaN propagates into the loss and
gradient); `compute_loss()`, `compute_variance_stats()` and `backward()` each recompute `X · D` and the characteristic-function sums (three projections
per step).

Action Items:

- [ ] Use a weighted quadrature over t; check for non-finite inputs; compute loss, gradient and diagnostics in one pass.

Location in code: `src/SIGReg.cpp` (`kFrequencies`, `compute_loss()`, `compute_variance_stats()`, `backward()`); tagged `TODO: See TD-504`.

Files to Modify:

- `src/SIGReg.{hpp,cpp}`
- `src/LeJEPAEncoder.cpp`

---

### TD-518: `TokenEmbedding` Unchecked Loading, Full-Matrix Updates and Validation Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Layers / TokenEmbedding | October 17, 2026 | 1-2 hours |

Description:
Found while tracing `TokenEmbedding.cpp`:
- **Verified:** `load_pretrained()`/`load_weights()` don't check reads: a file cut to half its size loaded without error and the rows past the cut kept
  their random initialization (the TD-467/TD-369 class).
- **By inspection:** `zero_grad()` and the no-optimizer update loop over all `vocab × d_model` entries every step (24.6 M floats at 32k × 768) though
  only a few rows have gradients; `get_gradient_norm()` sums in `float` (TD-435); `vocab_size`/`d_model` aren't validated (0 gives an infinite init
  scale) nor is `set_embeddings()`'s shape; `cached_token_ids` is never read; load/save/`print_config()` write to `std::cout`.

Action Items:

- [ ] Check every read and the file size in `load_pretrained()`; validate sizes and `set_embeddings()`'s shape.
- [ ] Zero/update only touched rows (track them in `backward()`); use `Logger`; drop the unused cache.

**Load/save checks (October 26, 2026, with TD-423).** `TokenEmbedding::save_embeddings()`/`load_pretrained()` reads and writes through `adai::binio` (`src/BinaryIO.{hpp,cpp}`): a short or over-long file throws `std::runtime_error`, the file is read into temporaries before anything changes, and a failed write or flush throws. Covered by `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`. This TD's other items stay open.

Location in code: `src/TokenEmbedding.cpp` (`load_pretrained()`, `zero_grad()`, `update_weights()`, constructor, `set_embeddings()`); tagged `TODO: See TD-518`.

Files to Modify:

- `src/TokenEmbedding.{hpp,cpp}`
- `tests/tokenembedding_test.cpp`

**GPU follow-up (October 19, 2026, TD-547).** GPU training no longer zeroes the host embedding gradient at all. The device table's gradient
is zeroed with one device memset per window, and the host table is never touched while training. The CPU path is unchanged.

---

### TD-527: `TrainingMetricsAPI` Per-Poll Logging, Parsing and Unbounded History Queries

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open — code fix landed October 25, 2026 | Metrics / API | October 17, 2026 | 1-2 hours |

Description:
Found while tracing `TrainingMetricsAPI.cpp`:
- **Verified:** `GET /api/sessions/{key}/metrics/current` logs `[metrics/current] matched key='…' len=…` at INFO on every request (five polls, five
  lines); dashboards poll it continuously, so it crowds the log and speeds rotation (TD-430/TD-431).
- **By inspection:** pushes find keys by substring and parse with `stoi`/`stof` (malformed numbers → 400 with the bare exception text, `stof`);
  `GET …/metrics/history?max_records=N` is uncapped; history timestamps are whole seconds; `handle_session_status()` switches the stream to fixed
  precision part-way, rounding `samples_per_second` to 2 decimals; every route repeats the same try/catch boilerplate (2,760 lines, TD-039).

Action Items:

- [x] Drop the per-request log line (or make it DEBUG); cap `max_records`; emit millisecond timestamps.
- [ ] Parse bodies with `nlohmann/json` (TD-453) and return field-specific errors; factor the per-route boilerplate. (Parsing and
  errors done; the per-route try/catch boilerplate is unchanged.)

**Fix (October 25, 2026).**
- **Logging.** `metrics/current` logs at debug.
- **History.** `max_records` is capped at 10,000, the in-memory history size. Records gain `timestamp_ms`; `timestamp` keeps its
  seconds string, which the dashboards read.
- **Status.** In `handle_session_status()` only `progress_percent` is rounded to 2 decimals.
- **Push bodies.** Every push handler (session start, epoch start/end, sample, validation, best, abnormal, generation quality,
  advanced, layer gradients) reads its body through one `PushBody` helper on nlohmann/json:
  - absent fields take defaults, and `null` in a number field is NaN (TD-443);
  - a wrong type is a 400 naming the field, and invalid JSON is a 400;
  - a key inside a nested object or a string can no longer be read as a top-level field;
  - string escapes are decoded, and `config` is re-serialized, with `model_id` folded in as a JSON field.

  The two substring parsers are gone (cpp net −387 lines).

The per-route try/catch boilerplate (TD-039) is unchanged.

Tests (`training_metrics_api_routes_test.cpp`):
- `PushBodiesParseAsJsonWithFieldErrors`: `config` with its own `session_id` sent first; an escaped label; a string where a number
  belongs; truncated JSON; null and 1e-7 values.
- `HistoryHasMillisecondTimestampsAndCappedMaxRecords`.

Against the old code both fail: `session_id` came out 99 from inside `config`, the label was cut to `say \`, errors had no field
name, and there was no `timestamp_ms`.

Location in code: `src/TrainingMetricsAPI.cpp` (`metrics/current` route, `handle_metrics_history()`, push handlers); tagged `TODO: See TD-527`.

Files to Modify:

- `src/TrainingMetricsAPI.cpp`

---

### TD-531: JSONL Training Reader Doesn't Decode `\uXXXX`, Corrupting Non-ASCII Text

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| HIGH | Open — code fix landed October 17, 2026 (dataset scan remains) | Data / JSONL Format | October 17, 2026 | 2-4 hours |

Description:
Found while tracing `TrainingSampleMeta.hpp` (see [TrainingSampleMeta.md](../reference/source/TrainingSampleMeta.md)); verified. `tsm_detail::json_str()`,
used by `parse_jsonl_sample()` for every training pair (`ChatbotTrainer::load_conversation_data()`, `DatasetRegistry::count_pairs()`/segments,
`VocabBuilderHelpers::load_pairs_format()`), decodes only `\n`, `\t`, `\r`; every other escape becomes the character after the backslash. JSON writers
escape non-ASCII by default (Python `json.dumps` `ensure_ascii=True`, pandas `to_json` `force_ascii=True`): `"Caf\u00e9 au lait?"` parsed as
`Cafu00e9 au lait?` and `"Oui \u2014 tr\u00e8s bon \ud83d\ude00"` as `Oui u2014 tru00e8s bon ud83dude00`. Any dataset prepared with standard tooling
and queued through `dataset_manager add`/upload therefore trains (and builds vocabularies) on garbage wherever it has an accented letter, curly
apostrophe (`don’t` → `donu2019t`), dash or emoji, silently. `\b`, `\f`, `\/` are mis-decoded too. (DataFetcher's own output is raw UTF-8; its
HuggingFace input scanner had the same gap, fixed by TD-314 on November 19, 2026.)

**Fix landed October 17, 2026:** `parse_jsonl_sample()` now uses `tsm_detail::parse_object_members()`, a small top-level JSON object reader. It
decodes every escape, including `\uXXXX` and surrogate pairs (unpaired surrogates become U+FFFD), and also fixes TD-532.

It is hand-written rather than `nlohmann/json`. `nlohmann` would reject lines `sample_to_jsonl()` itself writes (raw control characters, TD-533),
silently dropping those pairs, and the header reaches many files through `ChatbotTrainer.hpp`. Correctness is checked against `nlohmann` instead.

Tests (`training_sample_meta_test.cpp`, 20 → 32):
- `\u` escapes, surrogate pairs, every standard escape, nested keys, raw control characters, repeated keys and malformed lines;
- `MatchesNlohmannOnEscapedUnicode`, 500 random lines written by `nlohmann` with `ensure_ascii`, which must decode identically;
- 9 of the new tests fail against the old reader.

The local TinyStories training file (2,119,719 lines, 0 with `\u` escapes) parses identically with the old and new readers.

Action Items:

- [x] Parse JSONL lines with full escape and surrogate-pair handling. Done with a hand-written top-level reader, cross-checked against `nlohmann/json`, for the reasons above; TD-453 remains open (TD-314 closed November 19, 2026).
- [x] Add tests with `\u` escapes, surrogate pairs and every standard escape.
- [ ] Scan queued datasets for `\u` before retraining, and drop tokenized caches built from affected files so they re-tokenize. A pending file needs nothing else; re-reading it decodes correctly. Check with: `grep -lE '\\u[0-9a-fA-F]{4}' <registry data dir>/**/*.jsonl`. The local TinyStories copy has none.

Location in code: `src/TrainingSampleMeta.hpp` (`tsm_detail::parse_object_members()`, `parse_jsonl_sample()`); the `TODO: See TD-531` tag is now a fix comment.

Files to Modify:

- `src/TrainingSampleMeta.hpp`
- `tests/training_sample_meta_test.cpp`

---

### TD-538: Encoder Token Embedding and Final Norm Are Overwritten in Every Checkpoint

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open — code fix landed October 18, 2026 | Transformer / LLMEncoder | October 18, 2026 | 1-2 hours (plus retraining) |

Description:
Found while fixing TD-405; verified. `EncoderDecoderModel::save_model(path)` saves the encoder as `<path>.encoder` and the decoder as
`<path>.decoder`. `LLMEncoder` and `LLMDecoder` both derive their component file names by stripping the last extension, so both use the base
`<path>`. Both then wrote `<path>_token_emb.bin` and `<path>_final_norm.bin`. The decoder saves second, so the encoder's embedding and final
LayerNorm were overwritten in every checkpoint, and every load gave the encoder the decoder's copies.

Verified: a save/load round trip changed the logits by up to 1.54, and the reloaded encoder embedding differed by up to 5.08 while the decoder's
embedding and the LM head matched exactly. Every resumed training pass, every `clone()` (TD-363) and every `chatbot_api_server` that loads
`MODEL_PATH` ran its encoder on the decoder's embedding and norm.

**Fix landed October 18, 2026:** `LLMEncoder` now writes `<path>_encoder_token_emb.bin` and `<path>_encoder_final_norm.bin`; the decoder keeps
the old names. On load, if the encoder-specific files are missing (any checkpoint saved before the fix), the encoder falls back to the old shared
files with a warning. That is exactly what old checkpoints always loaded, so they load unchanged.

New tests: `EncoderDecoderModelTest.SaveLoadRoundTripReproducesLogits` (failed with a 1.54 logit difference before the fix) and
`LegacyCheckpointWithoutEncoderSpecificFilesStillLoads`.

Action Items:

- [x] Give the encoder's embedding and final norm their own file names, with a fallback for old checkpoints.
- [x] Test that a round trip reproduces the logits exactly.
- [ ] Retrain rather than continuing from an existing checkpoint: none has the encoder's own embedding or norm (with TD-358).
- [x] `IncrementalTrainer::remove_model_files()` (TD-407) should delete the new file names too. (Done: TD-407's prefix-based removal covers them.)

Location in code: `src/LLMEncoder.cpp` (`save_weights()`, `load_weights()`).

Files to Modify:

- `src/LLMEncoder.cpp`
- `tests/encoderdecoder_test.cpp`

---

### TD-547: GPU Training Does the Token Embeddings on the Host

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — code fix landed October 19, 2026 | GPU / Training | October 19, 2026 | 4-8 hours |

Description:
Found while building TD-468's device-resident optimizer. `TokenEmbedding` has no device mirror, so GPU training still does all of this on the host:
- **Each sample:**
  - the encoder and decoder embedding lookup plus positional encoding (`LLMEncoder::gpu_encode()`, `LLMDecoder::gpu_decode()`), with the result
    uploaded;
  - the embedding backward: the gradient downloaded (`Matrix::from_gpu`) and accumulated into the host table.
- **Each accumulation window:** `model->zero_grad()` zeroes both `vocab × d_model` host gradient tables (TD-518's full-matrix zeroing).
- **Each optimizer step:** `GPUOptimizer` steps the two embedding groups on the host, a full AdamW over `vocab × d_model` parameters twice:
  - ~49M parameters at 32k × 768;
  - likely the largest remaining CPU cost of a GPU pass.

That breaks the owner's rule that GPU training does the training math on the GPU and moves no data without a critical reason.

Action Items:

- [x] Give `TokenEmbedding` a device table and gradient, a gather kernel (lookup + positional-encoding add, with a device positional table),
  and a scatter-add backward (atomic adds for repeated tokens), in both backends.
- [x] Report them from `gpu_collect_parameters()` so `GPUOptimizer` steps them on the device; leave only the token IDs and scalars crossing.
- [ ] Verify on the oneAPI CPU device against the host path, as for TD-468; measure on ai-machine. (CPU device done; ai-machine open.)

**Fix (October 19, 2026).**
- **Device table:** `TokenEmbedding` has a device table and gradient (`gpu_upload_weights()`, `gpu_download_weights()`,
  `gpu_zero_grads()`, `gpu_collect_parameters()`).
  - `gpu_forward(ids, pe, pe_rows, pos_offset)` uploads only the token IDs and runs `embedding_gather_add_pe_gpu()`: the lookup plus
    positional row `pos_offset + i`, with none past the table, as `PositionalEncoding::forward()`.
  - `gpu_backward(dout)` runs `embedding_scatter_add_gpu()`, with atomic adds, since a token can repeat.
  - Both kernels exist in CUDA and SYCL.
- **Encoder and decoder:** `LLMEncoder`/`LLMDecoder` keep a device copy of their positional table and use the device embedding in
  `gpu_encode()`, `gpu_decode()`, their `gpu_backward()`s, and the KV-cached `gpu_decode_step()`. Generation therefore reads the live device
  table during training too.
- **Optimizer:** the embeddings are now device groups, so `GPUOptimizer` steps nothing on the host for a model without the gated
  world-model/hippocampal paths: 38 device groups, 0 host groups.
- **Zeroing:** in GPU mode `ChatbotTrainer` no longer calls `model->zero_grad()` at each window start, NaN skip or failed backward. That
  zeroed every host gradient, including both `vocab × d_model` tables. It now calls `GPUOptimizer::zero_host_grads()` (host groups only)
  plus `gpu_zero_grads()`.
- **Behavior change:** `gpu_decode_step()` used to compute positional rows inline, including past `max_seq_length`, where the full forward
  adds none (TD-347). It now uses the same table rows as `gpu_decode()`.

Verified on the oneAPI CPU device (`ADAI_SYCL_ALLOW_CPU_DEVICE=1`):
- **New test `DeviceEmbeddingMatchesTheHostLookupAndBackward`:**
  - the lookup equals `forward()` + `PositionalEncoding::forward()`;
  - the offset path adds rows 6–7 and nothing past the 8-row table;
  - the backward equals `backward()` with a repeated token;
  - an out-of-range ID throws.
- **`ModelStepMatchesTheDownloadAndCpuStepPath`:** now 38 device groups and 0 host, still equal to the old path.
- **Suites:** `chatbottrainerTests` (83), `encoderdecoderTests` (90) and `incrementaltrainerTests` (58) pass.
- **Real binary:** two SYCL `incremental_trainer` passes trained and resumed (global step 18 restored).

CUDA is compile-checked only.

Still crossing per sample: the token IDs and target IDs (inputs), the scalar loss, and the decoder's causal mask (`tgt²` floats built on the
host, small). The per-sample `gpu_synchronize()` (TD-390) also remains.

Location in code: `src/TokenEmbedding.{hpp,cpp}`, `src/LLMEncoder.cpp` (`gpu_encode()`, `gpu_backward()`), `src/Decoder.cpp` (`gpu_decode()`,
`gpu_backward()`), `src/gpu/MatrixGPU.cu`, `src/gpu/sycl/MatrixGPU_SYCL.cpp`.

---

### TD-553: Per-File Versions Have No Build Number

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Tooling / File status standard | October 24, 2026 | 2-4 hours |

Description:
Filed at the owner's request. The per-file `@adai-version` tag ([file-status-standard.md](file-status-standard.md)) is plain
`MAJOR.MINOR.PATCH`, and `scripts/check_file_status.py` accepts only that (`SEMVER_RE = ^(\d+)\.(\d+)\.(\d+)$`). This causes two
gaps:
- **No build number:** nothing counts the changes to a file between version bumps. A file edited in five commits at `1.2.0` looks the
  same as one edited once.
- **Reset rules unwritten:** the standard says what PATCH, MINOR and MAJOR mean, but not what happens to the lower parts on a bump.
  Practice has followed SemVer, for example `1.1.3 → 1.2.0`, but it isn't written down or checked.

Proposed scheme, following [SemVer 2.0.0](https://semver.org/):
- **Format:** `MAJOR.MINOR.PATCH+BUILD`, for example `1.2.0+17`. BUILD uses SemVer's build-metadata syntax (§10), so standard SemVer
  tools still parse the version and ignore BUILD when comparing precedence.
- **Resets** (SemVer §6–§8):
  - a PATCH bump keeps MAJOR.MINOR;
  - a MINOR bump resets PATCH to 0;
  - a MAJOR bump resets MINOR and PATCH to 0;
  - a pre-1.0 file follows the same rules (`0.9.3 → 0.10.0`).
- **BUILD:** a per-file counter that increases by one on every committed change to the file, including the commits that bump MAJOR or
  MINOR. It never resets, so it stays a monotonic change count across the file's whole history, like a build number that keeps
  climbing across releases.
- **Owner decision to confirm:** keep BUILD monotonic (recommended above), or reset it to 0 on every MAJOR/MINOR bump. Avoid a fourth dotted
  component (`1.2.0.17`, the .NET style), which isn't valid SemVer and would break the `stable ⇒ MAJOR ≥ 1` parsing.

Action Items:

- [ ] Get the owner's decision on BUILD's reset behavior (monotonic recommended).
- [ ] Update file-status-standard.md: the `+BUILD` format, the SemVer reset rules for MINOR and MAJOR bumps, when BUILD increments, and
  that BUILD never affects precedence or the `stable ⇒ MAJOR ≥ 1` rule.
- [ ] `check_file_status.py`: accept `MAJOR.MINOR.PATCH+BUILD` (BUILD a non-negative integer, with or without a `build.` prefix, as
  decided). With `--changed`, compare against the file's version at the base commit:
  - BUILD increased by at least 1;
  - the version didn't go backwards;
  - a MINOR bump reset PATCH, and a MAJOR bump reset MINOR and PATCH.
- [ ] `gen_status_report.py`: show BUILD in the status dashboard.
- [ ] Migrate every in-scope file (`src/`, Android `src/main`, `tizen-metrics-app/js/`, operational `scripts/`). Seed BUILD from the
  file's git history (`git log --follow --oneline -- <file> | wc -l`) or start at `+1`, as decided, in one mechanical commit.
- [ ] Update CLAUDE.md's "File status tag" convention and CONTRIBUTING.md, including that new files start at `0.1.0+1`.

Location in code: `docs/development/guides/file-status-standard.md`, `scripts/check_file_status.py`, `scripts/gen_status_report.py`,
and the header tag of every in-scope file.

---

### TD-554: No Aggregate Version per CMake Build

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Build / Versioning | October 24, 2026 | 1-2 days |

Description:
Filed at the owner's request. Every in-scope source file carries its own `@adai-version` (and, after TD-553, a build number), but nothing
combines them into a version for what is actually built and deployed:
- The top-level `CMakeLists.txt` declares a fixed `project(adai VERSION 1.0.0)` that nothing updates or reads.
- No binary reports a version (`--version`, `/health`, `/admin/status`).

So two `registry_server` builds, or a `sycl` and a `portable` `incremental_trainer`, can't be told apart, and there's no way to see
which files changed between what ai-machine runs and what's in the repository.

"Build pattern" here means a CMake preset × executable target: `debug`/`release`/`portable`/`sycl`/`gpu` × `registry_server`,
`mns_server`, `metrics_api_server`, `chatbot_api_server`, `incremental_trainer`, `trainer_service`, `dataset_manager`, `mns_cli`,
`vocab_builder` and the rest.

Proposed scheme (details for the owner to confirm):
- **Constituents:** the target's sources, the in-repo headers they include, and the sources of every `adai_*` library it links, resolved
  by CMake at configure time. Each constituent's `@adai-version` tag is read from its header.
- **Manifest:** a generated `<target>.versions` (file → version, status) per preset, kept with the build and committed for released
  builds as the baseline.
- **Aggregate version:** SemVer, bumped from the previous released manifest with the same rules as TD-553:
  - any constituent MAJOR increase (or a file added or removed from the target) → aggregate MAJOR + 1, MINOR and PATCH reset;
  - otherwise any MINOR increase → MINOR + 1, PATCH reset;
  - otherwise any PATCH or BUILD change → PATCH + 1.
- **Build metadata:** the preset and the git revision, for example `registry_server 1.4.2+sycl.g2928ede` (dirty-tree marked). A build
  counter from TD-553 (the sum of constituent builds, or the commit count) is an alternative.
- **Aggregate status:** the lowest `@adai-status` among the constituents, so a "stable" binary can't hide an experimental file.

Action Items:

- [ ] Owner decision: the aggregation rules above (bump-from-baseline vs. another derivation), the build-metadata contents, and where the
  released baselines live.
- [ ] CMake: a function (e.g. `adai_target_version(<target>)`) that collects a target's constituent sources, reads their tags (a small
  script run at configure/build time, sharing parsing with `check_file_status.py`), writes the manifest, and generates a
  per-target `<target>_version.hpp` with the aggregate version, status, preset and git revision.
- [ ] Binaries: `--version` on every executable, and the version in each daemon's `/health` (and `trainer_service`'s `/admin/status`),
  so operators can see what's deployed on ai-machine and the coordinator.
- [ ] Release flow: a script to diff a build's manifest against the last released one (which files changed and how), and to record the
  new baseline. Add a per-target table to `gen_status_report.py`.
- [ ] Use the project's own `project(VERSION)` for the overall release, or retire it, so there's one meaning for "the version".
- [ ] Docs: file-status-standard.md (the aggregate rules), the server bundle and deployment guides, and CLAUDE.md.

Depends on TD-553 for the build number and the written reset rules.

Location in code: `CMakeLists.txt`, `src/CMakeLists.txt`, `CMakePresets.json`, `scripts/check_file_status.py` /
`scripts/gen_status_report.py` (shared tag parsing), and each executable's `main()` / `/health` handler.

---

### TD-558: Android Ops Dashboard Doesn't Show Per-Sample Quality Reports

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Android / Ops dashboard | November 4, 2026 | 3-5 hours |

Description:
Filed at the owner's request with TD-273. Since TD-273 the trainer sends `metrics_api_server` a per-sample quality report after every
epoch: for each split, the distribution of each pair's score (`exp(-loss)`, or BLEU-4 / 100 from the opt-in generation backfill) as
`count`/`mean`/`min`/`p10`/`median`/`p90`/`max`, plus the lowest-scoring pairs with their input and target text. The server serves them at
`GET /api/sessions/{key}/metrics/sample-quality` (the newest 500, oldest first, in memory) and `…/metrics/db-sample-quality[?limit=N]`
(the database), documented in [TRAINING_METRICS_API.md](../TRAINING_METRICS_API.md). The Android ops dashboard's session detail screen
(`SessionDetailScreen.kt`) shows current metrics, advanced diagnostics, generation quality and per-epoch history charts, but nothing
reads the new route, so the lowest-scoring pairs (the data the model fits worst: noisy, mislabelled or truncated examples) are only
visible with `curl`.

Action Items:

- [ ] `MetricsApiService.sampleQuality(key)` for `GET api/sessions/{key}/metrics/sample-quality`, with `SampleQualityResponseDto`/
      `SampleQualityReportDto`/`LowQualitySampleDto` (nullable statistics: the server sends `null` for a non-finite value; keep
      `ignoreUnknownKeys`), a `MetricsRepository.sampleQuality()` and a `sampleQuality` field in `SessionDetailUiState`, refreshed with the
      other per-session reads in `SessionDetailViewModel`.
- [ ] A "Sample quality" section on the session detail screen for chatbot sessions (not LeJEPA ones, which send no reports): for the
      latest epoch, a row per split and source with count, mean, p10 and median; a per-epoch chart of the training and validation `p10`
      and `mean` (`MetricHistoryChart`), so a falling worst decile is visible; and the latest report's lowest-scoring pairs as an expandable
      list showing score, input and target (the server caps texts at 4 KiB; clip further in the list, full text on expand).
- [ ] Say which score a report is (loss vs generation) and that higher is better; show "No sample-quality reports" for a session that has
      none (an older trainer, `LOSS_QUALITY_BACKFILL=false`, or a LeJEPA pass), and tolerate a 404 from an older `metrics_api_server`.
- [ ] Tests: DTO parsing including `null` statistics, the ViewModel state, and a Compose UI test of the section (empty, one split, both
      splits, generation reports), matching the existing session-detail tests.
- [ ] Update the file status tags, and note the section in [TRAINING_METRICS_API.md](../TRAINING_METRICS_API.md)'s sample-quality entry (the ops dashboard has no operator guide of its own).

Location in code: `android/opsdashboard/src/main/java/com/adai/ops/network/MetricsApiService.kt`, `network/dto/`,
`data/metrics/MetricsRepository.kt`, `ui/metrics/SessionDetailViewModel.kt`, `ui/metrics/SessionDetailScreen.kt`.

Files to Modify:

- `android/opsdashboard/src/main/java/com/adai/ops/network/MetricsApiService.kt` and a DTO file under `network/dto/`
- `android/opsdashboard/src/main/java/com/adai/ops/data/metrics/MetricsRepository.kt`
- `android/opsdashboard/src/main/java/com/adai/ops/ui/metrics/SessionDetailViewModel.kt`, `SessionDetailScreen.kt`
- the matching tests under `android/opsdashboard/src/test/` (and `androidTest/` for the Compose UI test)

---

### TD-559: Tizen Dashboard Doesn't Show Per-Sample Quality Reports

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Tizen / Metrics dashboard | November 4, 2026 | 2-4 hours |

Description:
Filed at the owner's request with TD-273, as TD-558 for the Android app. The Tizen TV dashboard (`tizen-metrics-app/`) polls
`…/metrics/current` and `…/metrics/history` (`poll()`, `applyMetrics()` in `js/app.js`) and shows primary, loss and auxiliary metric columns,
but doesn't read `GET /api/sessions/{key}/metrics/sample-quality`, so the per-epoch quality distribution and lowest-scoring pairs the
trainer now reports aren't on the TV.

Action Items:

- [ ] Fetch `…/metrics/sample-quality` with `fetchJSON()` for the selected session, less often than the live metrics (reports arrive once
      per epoch, so once per epoch change or every 30 s is enough), and tolerate a 404 from an older `metrics_api_server`.
- [ ] Show the latest epoch's training and validation `mean`/`p10`/`median` in the auxiliary column, colored like the generation-quality
      tiles (`applyGenQualColor()`), with the score type (loss or generation) labelled.
- [ ] A remote-navigable panel (`js/navigation.js` focus handling) listing the latest report's lowest-scoring pairs: score, input and
      target, escaped with `escapeHtml()` and clipped to a few lines for a 10-foot UI. `null` statistics show as "—".
- [ ] Optional: plot the per-epoch training/validation `p10` with `js/chart.js` alongside the loss chart.
- [ ] Check it on the TV (or the Tizen emulator) against a real `metrics_api_server`. The app reads with `fetch(…, {mode: 'cors'})`, so
      the new route needs the same `METRICS_API_CORS_ALLOWED_ORIGINS` entry as the existing ones (TD-526); no server change is expected.
- [ ] Update `tizen-metrics-app/README.md` and the file status tags.

Location in code: `tizen-metrics-app/js/app.js` (`poll()`, `applyMetrics()`), `js/navigation.js`, `js/chart.js`, `index.html`, `css/`.

Files to Modify:

- `tizen-metrics-app/js/app.js`, `js/navigation.js`, `js/chart.js`
- `tizen-metrics-app/index.html`, `tizen-metrics-app/css/`
- `tizen-metrics-app/README.md`

---

### TD-562: Vocabulary Sizing and Tools Assume Lowercase Text

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open — code fix landed November 4, 2026 (real-data measurement pending) | NLP / Tokenizer | November 4, 2026 | 2-3 hours |

Description:
Filed with TD-560. With case kept, the same corpus has about 23% more distinct words (measured on local prose), so a vocabulary sized
for lowercase text spends more of its tokens on case variants and splits more words. `BPETokenizer::recommend_vocab_size()` (used when
`VOCAB_BUILD_SIZE=0`, the shipped default) counts words without regard to case, and the vocabulary tools and checks assume lowercase
tokens.

Action Items:

- [x] Make `recommend_vocab_size()` account for the normalization (count distinct pre-tokens as the tokenizer will see them, or scale for
      case), so an auto-sized cased vocabulary isn't undersized; log the normalization with the chosen size.
- [x] `vocab_builder`: report the normalization it built with; `scripts/audit_vocab_merges.py` (TD-220) and any other vocabulary tooling
      read the v2 header.
- [ ] Re-measure the token cost on the real training data (the registry pool) at the chosen vocabulary size before the rebuild, and
      record it next to the TD-560 numbers.

Fix (November 4, 2026): `recommend_vocab_size()` takes a `TextNormalization` (default `CASED`) and, for a cased vocabulary, multiplies
by 1 + the share of words with an ASCII capital (capped at +30%) before the architecture cap; `bootstrap_vocab()` logs the letter case.
`vocab_builder` prints `Letter case: kept|lowercased`; `scripts/audit_vocab_merges.py` reports each file's format version and letter case
(a v1 file as lowercase). Test: `RecommendedSizeAccountsForLetterCase`. The real-data measurement (third item) needs the registry pool
on ai-machine and stays open.

Location in code: `src/BPETokenizer.cpp` (`recommend_vocab_size()`), `src/VocabBuilder.cpp`, `scripts/audit_vocab_merges.py`.

Files to Modify:

- `src/BPETokenizer.cpp`, `src/VocabBuilder.cpp`, `scripts/audit_vocab_merges.py`
- `tests/tokenizer_test.cpp`

---

### TD-563: Rebuild Vocabularies and Retrain With Case, and Deploy in Order

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — format refusal landed November 4, 2026; rebuild, retrain and rollout pending (TD-358) | Training / Operations | November 4, 2026 | Not estimated (retraining) |

Description:
Filed with TD-560. A cased vocabulary needs a newly trained model (its token ids differ), and the binaries that read it must understand
the v2 header: `load_vocab()` today skips header lines it doesn't recognize, so an older `chatbot_api_server`, `incremental_trainer`
or `vocab_builder` would load a v2 vocabulary and silently lowercase every input, which a cased model has never seen.

Action Items:

- [ ] Deploy binaries with TD-560 everywhere (`incremental_trainer`/`trainer_service` hosts, `chatbot_api_server`, `vocab_builder`, any
      tool reading vocabularies) before the first cased vocabulary exists, and make new binaries refuse a vocabulary
      format they don't know (so the next format change can't repeat this).
- [ ] Rebuild the chatbot vocabulary (and the world model's, if it has its own) with case, in the same rebuild as the conversation turn
      tokens ([proposal](../../proposals/conversation_turn_tokens_plan.md)) and, if decided, TD-224's whitespace change, then retrain
      within the TD-358 retraining so checkpoints are retrained once.
- [ ] Check the training data keeps its case through the pipeline (registry uploads, `DataFetcher`'s Gutenberg/HuggingFace cleaning,
      the JSONL reader); none of it is known to lowercase, but verify on queued files.
- [ ] After retraining, compare against the lowercase model: loss and perplexity per token aren't comparable across vocabularies, so use
      generation quality (BLEU with TD-561's case handling, ROUGE) and a manual check of capitalization in replies.
- [ ] Operator docs: which binaries must be updated first, how to build a cased vocabulary, and that RAG retrieval now distinguishes
      case (`GPU` vs `gpu`).

Progress (November 4, 2026): the code half of the first item is done with TD-560: `load_vocab()` refuses a format newer than
`kVocabFormatMajor` (2) and any unrecognized header line, so binaries from this release on can't misread a future format. Checked with
the real `chatbot_api_server`: given a vocabulary marked v3.0 with an unknown header line, the old server started and served with it,
the new one exited 1 ("vocabulary format v3.0, newer than this build reads (v2); update the binary"). A v2 `LOWERCASE` file, which the new
trainer writes when it continues an old session, is read correctly even by old binaries (they lowercase anyway); only a `CASED`
vocabulary needs the new binaries. Deploying them, rebuilding, retraining and the remaining items are operations work.

Location in code: none of its own; operations and training. Depends on TD-560 and TD-562; pairs with TD-358 and the turn-token proposal.

Files to Modify:

- `docs/operations/OPERATIONS_MANUAL.md`, `docs/operations/guides/training-guide.md`, deployment notes for ai-machine

---

### TD-573: What the Android Apps Need to Reach `stable`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Android / `app`, `opsdashboard` | November 12, 2026 | 2-4 days, plus device access and a release |

Description:
No Android file is `stable` (0 of 118 on November 12, 2026). TD-047 (no release history) and TD-048 (UI never run on a device) cap
them, but neither says what `stable` would take. This item sets that bar for the two phone apps and lists what stands between each
file and it. **Scope:** `android/app` (34 files: 29 `beta`, 5 `experimental`) and `android/opsdashboard` (74: 49 `beta`, 25
`experimental`), 108 files in all. The wear modules (`wearcomplications`, `wearsync`, `wearface`) and `opsdashboard`'s
`data/wearsync/` package were set aside on November 12, 2026, and removed on November 21, 2026 (owner decisions).

**The bar.** `file-status-standard.md`'s `stable` means tested, documented, no known correctness gap that would surprise a caller, and
in use. For these apps that means:
1. **Released** (TD-047): a first real release of each app under [`android-release.md`](android-release.md), so the files are in
   use at a known version rather than `0.1.0`/`versionCode 1`; and the Android CI running on `origin` (rjv717/adai), whose GitHub
   billing stops it today (it runs on the kerfbit mirror).
2. **Tests run, not just compiled** (TD-048): every `opsdashboard` instrumented test (12 screens, the confirm-action dialogs, the DI
   container and `MainActivity`) has only been compiled. It needs at least one full `connectedDebugAndroidTest` run on a device or
   emulator, recorded in TD-048. The `app` module's instrumented tests have run on the `Medium_Phone_API_35` AVD. `opsdashboard`'s
   first ran there on November 21, 2026 (setup step 1 below), and after the same day's fixes all 109 pass (TD-048's updates).
3. **Each file covered** by a test that exercises it (named, or drawn inside a tested screen), with no open item against it.
4. **Documented**: the Android files have no code-traced reference documents (`docs/development/reference/source/` covers C++ and
   scripts). Decide what "documented" means for them, for example one reference document per screen or package rather than per file,
   and write it.

**`experimental` files and what each needs** (none of these is named by a test):

| Files | Needs |
|---|---|
| `opsdashboard` `ui/common/BiometricAdminAuthGate.kt` | A real-prompt test (setup step 3); today `FakeAdminAuthGate` replaces it in every screen test |
| `opsdashboard` `ui/registry/RegistryActionDialogs.kt`, `ui/models/LinkWorldModelDialog.kt`, `ui/models/RegisterModelDialogs.kt` | Compose tests of each dialog's validation and submit (`CreateSegmentsDialog`'s range parsing, registration fields; the upload dialog's file picker and overwrite checkbox need setup step 2) |
| `opsdashboard` `network/SafeCall.kt`, `network/dto/CommonDtos.kt`, both apps' `network/CloudflareAccessInterceptor.kt`, `app` `data/db/Converters.kt` | Unit tests: the HTTP-status mapping (404/409/other, error-body parsing), header injection, the Room converters |
| `opsdashboard` `polling/FixedIntervalPoller.kt`, `ui/navigation/OpsNavHost.kt`, `ui/scaffold/OpsAppScaffold.kt`, `ui/common/ConfirmActionDialog.kt` | Already exercised by tests that name them (poller unit tests; navigation and confirm-dialog instrumented tests): `beta` once those tests have run (bar 2) |
| `opsdashboard` `ui/common/{AdminActionButton,LoadingErrorState,MetricHistoryChart,ModelPickerDropdown,SettingsAction,StaleBadge}.kt`, `ui/navigation/OpsDestinations.kt`, the `*Route.kt` files (`AdminRoute`, `SessionsRoute`, `ModelsRoute`, `RegistryRoute`, `TrainerRoute`), `app` `ui/chat/TypingIndicator.kt` | Drawn inside screens the Compose tests cover, but not checked themselves: either assertions about them in those tests, or a short test each |
| Both apps' `ui/theme/Theme.kt` and `Type.kt` | Styling with no behaviour to test; decide whether theme files are in scope for `stable` or exempt, and record it in the standard |

**Test setup.** Each app already has what most of the above needs: JUnit 4 and `kotlinx-coroutines-test` for unit tests; fakes in
`src/sharedTest` shared by unit and instrumented tests (`opsdashboard`: fake metrics, MNS, registry and trainer API services, API
client provider, settings repository, `FakeAdminAuthGate`; `app`: fake DAOs, API client provider, settings
repository), used instead of a mocking library; and `androidx.test.ext.junit`, `compose-ui-test-junit4` and `AndroidJUnitRunner` for
instrumented and Compose tests, with `ConfirmDialogTestActivity` as the host for admin confirm dialogs. The network helpers and
converters are plain JVM code (a fake OkHttp `Interceptor.Chain` covers the interceptors). Three things were missing:
1. **Done (November 21, 2026), by removing the wear modules.** **`opsdashboard` requires the `wear-sdk` library to install.** Its merged manifest declares
   `<uses-library android:name="wear-sdk" android:required="true"/>`, brought in by a wear dependency (the app's own manifest doesn't
   name it), so the APK won't install on a phone or emulator without that library. That is why its instrumented tests have never run.
   Override it to `required="false"` in `opsdashboard`'s manifest (`tools:node`/`tools:replace`), after checking that the watch-face
   push code (`WatchFacePushRepository`, `WearWatchFacePushRepository`) copes when the library is absent. **Do this first:** it blocks
   bar 2 for every `opsdashboard` file.
2. **No way to drive the system file picker.** The upload dialog launches `ActivityResultContracts.GetContent()`, which a Compose test
   can't operate. Add `androidx.test.espresso:espresso-intents` (androidTest) and stub the picker's result with `intending(...)`.
3. **No test path for a real `BiometricPrompt`.** `androidx.biometric` 1.1.0 has no test authenticator (TD-048's item assumed one).
   Options: an emulator with an enrolled fingerprint, driven with `adb emu finger touch` from the test (needs the emulator, not a
   physical device), or keeping the real prompt as a documented manual check while `FakeAdminAuthGate` covers every screen. Decide which.

**`beta` files:** 35 are capped by TD-047 (15 `app`, 20 `opsdashboard`) and 42 by TD-048 (14 `app`, 28 `opsdashboard`);
`GroupDetailScreen.kt`'s note cites TD-205 and TD-048. They need bars 1, 2 and 4, and a check that nothing else is open against them.

Action Items:

- [x] Setup 1: make `wear-sdk` optional for `opsdashboard` (checking the watch-face push code without it). Done November 21, 2026
  by removing the wear modules and their dependencies instead; the APK no longer requires `wear-sdk`.
- [ ] Setup 2: add `espresso-intents` for the file picker.
- [ ] Setup 3: decide how `BiometricAdminAuthGate` is checked (emulator fingerprint, or a documented manual check).
- [ ] Cut the first release of each app (TD-047), and get the Android CI running on `origin`.
- [x] Run `opsdashboard`'s instrumented tests on an emulator, and fix what fails. Done November 21, 2026 on `Medium_Phone_API_35`:
  67 of 109 passed at first; after fixing the tests and one layout bug (TD-048's updates), all 109 pass.
- [ ] Add the tests in the `experimental` table above.
- [ ] Decide what "documented" means for the Android files, and whether theme files are exempt; write the documents.
- [ ] Then review the files for promotion, a module at a time.

**The wear modules (removed).** `wearcomplications`, `wearsync`, `wearface` and `opsdashboard`'s `data/wearsync/` were set aside on
November 12, 2026 and deleted on November 21, 2026, at the owner's decision, with `opsdashboard`'s watch-sync and watch-face-push
settings, the bundled `wearface.apk` and the wear libraries (`play-services-wearable`, `watchfacepush`, the watch-face validator,
WorkManager).

Location in code: `android/app/src/main`, `android/opsdashboard/src/main` (see [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md)
for the current list).

Related: TD-047, TD-048.

---

### TD-575: Deployed `incremental_trainer` Aborts at Startup in `BPETokenizer::load_vocab`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — the code half was already done by TD-414 (a current binary exits 1, a failed pass, instead of aborting); what's left is on ai-machine: the vocabulary's error message, and deploying current binaries | Training / Deployment | November 18, 2026 | 2-4 hours, plus a look at ai-machine |

Description:
Split from TD-574 (November 18, 2026). During TD-574's seventh round (November 15, 2026) the `incremental_trainer` deployed on
ai-machine (run by `trainer_service`, not the test package) aborted five times. Its journal entry for one of them (November 15,
23:54 UTC) is a `SIGABRT` from an exception nobody caught: `BPETokenizer::load_vocab`, called from `IncrementalTrainer::build_model()`
at startup. Every pass `trainer_service` launches would end the same way, so the deployed service trains nothing. Likely cause, not
yet checked: the vocabulary format refusal (TD-560/TD-563), an older or newer vocabulary file than the deployed binary reads, or a
missing `VOCAB_PATH`. Either way, a vocabulary that can't be loaded should end the pass with a clear error and a failed-pass exit
code, not an abort with a core dump.

Also on the deployed host (from TD-574): an explicit `GPU_STRATEGY=background` in `/etc/adai/config.*.conf` would override the new
default `normal`. Less urgent now that ai-machine runs the 7.0 kernel, which resets no priority (TD-574), but still worth matching.

**What ai-machine showed (November 18, 2026).** The service is in a restart loop and has trained nothing: every 47 s (systemd's
`RestartSec=45`) a pass loads its configuration, resolves model `ambitious-aardvark` from MNS (d_model 512, 6+6 layers), starts
the GPU (`GPU_STRATEGY` `full`) and aborts with `VocabularyFileError: Could not open vocabulary file:
/home/rodney/Repos/adai/vocab.txt`.
- The binary is `/opt/adai/bin/incremental_trainer` of **September 23, 2026**, before TD-414 (the handler), TD-358 (GPU targets),
  TD-538, TD-560 and the rest of the training fixes; there is no `trainer_service` (TD-172's cutover never happened).
- It reads `/opt/adai/config/config.conf`, the legacy single-file config, whose `VOCAB_PATH` is a path on the development
  machine. `/etc/adai/config.conf` names `/opt/adai/vocab/vocab.txt`, but this binary doesn't read it. Even the right path in the
  wrong place would fail: the unit's `ProtectHome=yes` hides `/home`.
- Its log file is the relative `chatbot_server.log`, which `ProtectSystem=strict` makes read-only, so it logs to the journal only.

Fixing the path alone would let a September binary train, with TD-358's unshifted GPU targets: it would damage the checkpoint
further. So the service stays stopped until the deployment is current (TD-172's cutover to `trainer_service`, TD-563's order).

Action items:
- [x] On ai-machine: `coredumpctl info 2455` (or the newest `incremental_trainer` dump) and the journal around it; the exception's
  message names the reason. (November 18, 2026: the journal; above.)
- [x] Stop and disable the looping `adai-trainer` unit until the deployment is current. (November 18, 2026. The installed unit
  is the old one: `incremental_trainer --config /opt/adai/config/config.conf --foreground resume`, `Restart=on-failure`. Also
  deployed: `chatbot_api_server` of July 4, the vocabulary of July 4 (lowercase v1, before TD-220's merge fix), and only the legacy
  `/etc/adai/config.conf` of July 4.)
- [x] Upgrade the server host first (192.168.1.16: `mns_server`, `registry_server`, `metrics_api_server`, `mns_cli`,
  `dataset_manager`), November 18, 2026: the portable build of `98d65486`, packed by `package_server_bundle.sh`, put in place
  by the new `redeploy_server_bundle.sh install`, which kept the host's units (`adai-mns`, `adai-registry` with plain FTP,
  `adai-metrics-api` with `--config /opt/adai/etc/config.conf`; the old, stopped `adai-metrics` left alone), configs and data.
  All three answer `/health`; MNS still lists its three models and the metrics server its sessions. Backup and rollback:
  `/var/backups/adai-server-bundle/20261118T064630Z`. The binaries it replaced were from September 20 and October 1.
- [x] Register the new model in MNS (owner decision November 18: a new model rather than retraining `ambitious-aardvark`), and
  settle `ambitious-aardvark`'s stale `training` lock (`run-01`, since August) and `chatbot-v1`. (November 18, 2026:
  `ambitious-aardvark` and `chatbot-v1` retired; `bodacious-badger` registered, kind `chatbot`, d_model 768, 12 heads, d_ff
  3072, 8 encoder + 24 decoder layers, max sequence 1024, run group `bb01`, display name "Bodacious Badger"; the never-trained
  `world-model-v1` (768, 16 layers, max sequence 512) moved to `bb01` and linked with injection every 8 decoder layers. The
  world model reads only a prompt's first 512 tokens; the decoder's world-model cross-attention takes its output at any length.)
- [ ] Queue `bb01`'s data (owner choice November 18: Alpaca, Dolly and Daily Dialog for the chatbot; the 14 Gutenberg books
  and TinyStories for the world model; fetched again with today's code). Done from the development machine with
  `dataset_manager` pointed at 192.168.1.16, after two `DataFetcher` fixes deployed to the registry server the same day:
  Dolly's `context` (an instruction's material was only read from `input`) and one-utterance-per-row dialogs (Daily Dialog
  produced no pairs), plus `all` as a fetch's pair count (a larger count wraps to the start and repeats it silently: the
  first Alpaca fetch, 52,002 requested, repeated 28 rows, since 28 rows have no answer; deleted and fetched again with
  `all`). Chatbot pool: Alpaca 51,974 pairs, Dolly 15,011, Daily Dialog 49,188. World-model pool: TinyStories 2,141,468 pairs (HuggingFace's copy has 2,141,709 rows, not the 2,119,719 September's
  fetch counted; 241 rows too short to split; no duplicates);
  the books wait on Project Gutenberg, down for maintenance on November 18 until after 21:00 UTC (every download
  answered 503); then `dataset_manager --kind world_model gutenberg-batch <the 14 ids> all bodacious-badger` against the
  registry server. On the server host before: the instruction sets `chatbot-v1` fetched in September (Alpaca, Dolly 15k, and
  Daily Dialog, fetched but never queued), 14 Gutenberg books for the world model, and TinyStories twice (2 GB of JSONL, 2.1
  million pairs, queued in `my-training-group/decoder` for `ambitious-aardvark` and never trained); all fetched by
  September's `DataFetcher`.
- [ ] Deploy on ai-machine: `package-sycl.sh --system-oneapi` (November 18, 2026) packs the SYCL `incremental_trainer`,
  `trainer_service`, `chatbot_api_server`, `vocab_builder` and `dataset_manager` (oneAPI 2026.2.0) with
  `scripts/ai_machine/deploy.sh`, the `adai-trainer@.service` template (owner decision: the world model first, so
  `adai-trainer@lejepa` before `@chatbot`), a GPU-capable `adai.service`, and configs for `bodacious-badger` with every path
  under `/opt/adai`. The oneAPI runtime is Intel's apt package (owner decision, as in August), so `deploy.sh install` takes
  the bundled 2026.0 `/opt/adai/lib` off the library path. Then the vocabulary (from `bb01`'s data), then
  `systemctl enable --now adai-trainer@lejepa`.
- [ ] Fix the cause: deploy current binaries and `trainer_service` (TD-172's cutover) with per-service configs under
  `/etc/adai/` whose `VOCAB_PATH` names the deployed vocabulary, together with the vocabulary (TD-563's order).
- [x] Catch the exception in `incremental_trainer`'s startup (log it, exit with a failed-pass code), and test that a bad vocabulary
  file fails the pass without an abort. (November 18, 2026: already done in code by TD-414, November 2. The crash's stack has
  `main` → `IncrementalTrainer::IncrementalTrainer` → `build_model` → `load_vocab` → `__cxa_throw` → `std::terminate`, with no
  handler, so the deployed binary predates TD-414; every current command runs inside `main()`'s `try`. Added the deployed path to
  `SighupTest.AFirstTrainBuildsTheVocabularyAndExceptionsExitOne`: `--admin-port N resume` with an unreadable vocabulary exits
  1 (a failed pass), and exits 134 (`SIGABRT`) with `main()`'s handler removed, as on ai-machine.)
- [x] `grep -H GPU_STRATEGY /etc/adai/config.*.conf` on ai-machine; remove an explicit `background`. (November 18, 2026: the
  running pass logs `full`; nothing to change.)

Location in code: `src/IncrementalTrainer.cpp` (`build_model()`), `src/BPETokenizer.cpp` (`load_vocab()`),
`src/IncrementalTrainingTool.cpp`.

Related: TD-574 (archived), TD-563, TD-560.

---

### TD-591: A Download the Registry Can't Resume Fails Every Training Pass

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — the trainer-side fix landed November 26, 2026; deploying it to ai-machine remains. `REST` in `FtpDataServer` is [TD-592](#td-592-ftpdataserver-doesnt-implement-rest-so-downloads-cant-resume), training the files that did arrive is [TD-593](#td-593-one-failed-download-fails-the-whole-training-pass) | Data / DataTransport | November 26, 2026 | A deployment |

Description:
Found on ai-machine, November 26, 2026. The LeJEPA trainer (`adai-trainer@lejepa`, pool `bb01/world_model`) failed 50 passes in a
row, each a few seconds after start, with `FTP transfer failed for 'bb01/world_model/datasets/roneneldan_TinyStories_train_…jsonl':
FTP: command REST failed`. An earlier pass had stopped downloading that file at 1,198,997,576 of 2,058,492,441 bytes (58%).
TD-340 keeps a partial download so the next pass resumes it with `CURLOPT_RESUME_FROM_LARGE`, which sends FTP's `REST`. But
`FtpDataServer` doesn't implement `REST`: it falls through to `502 Command not implemented`, curl reports
`CURLE_FTP_COULDNT_USE_REST`, which `is_transient()` doesn't retry, and the failure keeps the partial file again. So every pass failed
the same way, `trainer_service` backed off to 900 s, and the pool trained nothing although its other 14 files were downloaded. One
file that can't be downloaded also fails the whole pass. The resume path had never run against the real server: TD-340's tests only
check that the partial file is kept.

Action Items:

- [x] Trainer side: when a resumed attempt fails with `CURLE_FTP_COULDNT_USE_REST` or `CURLE_BAD_DOWNLOAD_RESUME`, delete the
      partial file and download the whole file in the same fetch, without using up a retry (once per fetch). Test against a real
      `FtpDataServer` on loopback. (November 26, 2026: `DataTransport::fetch()`;
      `FtpsVerificationTest.ARefusedResumeDownloadsTheWholeFile` failed with the production error before the fix.)
- [ ] Deploy the fixed `incremental_trainer` to ai-machine. Until then, delete the partial file (and its `.checksum` sidecar) under
      the trainer's download directory and restart `adai-trainer@lejepa`.
- [ ] Find what interrupted the first download (the journal before the first `REST failed`).
- Split out November 26, 2026: `REST` in `FtpDataServer`, so a download really resumes, is [TD-592](#td-592-ftpdataserver-doesnt-implement-rest-so-downloads-cant-resume); training the
  files that did arrive when one download fails is [TD-593](#td-593-one-failed-download-fails-the-whole-training-pass).

Location in code: `src/DataTransport.hpp` (`fetch()`, the restart after a refused resume).

Related: TD-340 (archived), TD-569, TD-575, TD-592, TD-593.

---

### TD-592: `FtpDataServer` Doesn't Implement `REST`, So Downloads Can't Resume

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / FtpDataServer | November 26, 2026 | 2-4 hours |

Description:
Split from TD-591 (November 26, 2026). `FtpDataServer`'s command loop has no `REST`: it falls through to `502 Command not implemented`.
The trainer resumes a partial download with `CURLOPT_RESUME_FROM_LARGE`, which sends `REST <offset>` before `RETR`. Since TD-591's
trainer-side fix, a refused resume downloads the whole file again, so nothing is stuck any more, but every interrupted download starts
from byte 0: on ai-machine that is up to 2 GB (TinyStories) again after a pass stops at 58%. TD-340's resume across passes only pays off
with a server that resumes.

Action Items:

- [ ] Implement `REST <offset>`: parse a non-negative 64-bit offset (`501` on a bad one), keep it on the session, reply
      `350 Restarting at <offset>.`, apply it to the next `RETR` (seek the file; `554` or `551` if the offset is past its end) and reset
      it after that `RETR` (and on any other transfer command), per RFC 3659's stream-mode restart. Advertise `REST STREAM` in `FEAT`.
- [ ] Check the token semantics still hold: a resumed `RETR` uses the same single-use token as the first try only if TD-377's
      "consumed token" rules allow it; otherwise the trainer's next pass gets a fresh token anyway (each pass acquires again).
- [ ] Test: a unit test of `REST` + `RETR` in `FtpDataServerTests.cpp`, and an end-to-end resume in `DataTransportFtpsTests.cpp`
      (a partial file with the right first bytes is completed, and only the missing bytes are sent).
- [ ] Redeploy `registry_server` on the server host (192.168.1.16).

Location in code: `src/FtpDataServer.hpp` (the command loop's unimplemented-command reply); tagged `TODO: See TD-592`.

Files to Modify:

- `src/FtpDataServer.hpp`
- `tests/FtpDataServerTests.cpp`, `tests/DataTransportFtpsTests.cpp`

Related: TD-591, TD-340 (archived), TD-377.

---

### TD-593: One Failed Download Fails the Whole Training Pass

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / TrainingPassFiles | November 26, 2026 | 2-4 hours |

Description:
Split from TD-591 (November 26, 2026). `resolve_pass_files()` (`src/TrainingPassFiles.cpp`) calls `DataTransport::fetch_all()`, which
rethrows the first download error after every thread finishes; `resolve_pass_files()` then returns false and the pass fails, releasing
every claimed file. On ai-machine the 14 Gutenberg files of `bb01/world_model` were downloaded and ready, but the one TinyStories file
that couldn't be downloaded failed 50 passes in a row, so none of them trained. Whatever makes one file undownloadable (a server bug, a
file deleted on the registry, a corrupt checksum) blocks the whole pool.

Action Items:

- [ ] Have `fetch_all()` (or a new variant) report each file's result instead of only the first error, and have `resolve_pass_files()`
      train the files that arrived, releasing the failed ones back to pending (or recording the failure on the registry), with a
      warning naming each.
- [ ] Decide what a pass in which every download failed returns (still a failed pass, so `trainer_service` backs off), and what
      happens to a file that fails pass after pass (a failure count, then leave it for an operator).
- [ ] Keep the TD-205 segment semantics: segment tokens of one file share a download, so they succeed or fail together.
- [ ] Test with a `fetch_all()` where one token fails: the others train, the failed one is released; all four callers
      (`IncrementalTrainingTool.cpp` ×3, `LeJEPATrainingPass.cpp`).

Location in code: `src/TrainingPassFiles.cpp` (`resolve_pass_files()`), `src/DataTransport.hpp` (`fetch_all()`).

Files to Modify:

- `src/TrainingPassFiles.{hpp,cpp}`
- `src/DataTransport.hpp`
- `src/IncrementalTrainingTool.cpp`, `src/LeJEPATrainingPass.cpp`
- `tests/training_pass_files_test.cpp`

Related: TD-591, TD-333 (archived: a failed pass releases its exact units).

---

### TD-594: A Download That Outlasts Its FTP Token Can't Be Retried

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / DataTransport / registry_server | November 26, 2026 | 3-5 hours |

Description:
Found November 26, 2026, tracing what started TD-591 on ai-machine. `registry_server` issues each acquired file's FTP token with a
fixed lifetime (`FTP_TOKEN_TTL_MINUTES`, default 30) when the trainer acquires, and `FtpDataServer` checks the expiry at login only
(TD-377). A transfer that is already running outlives it, which is why a 45-minute TinyStories download (2 GB at 0.72 MB/s) finished on
November 24. But `DataTransport::fetch()`'s retries (TD-340) log in again with the same token. On November 26, 32.7 minutes into the
same download, `registry_server` on the server host was restarted (07:03:56, together with PackageKit, ModemManager, polkit, fwupd and
ssh: needrestart after unattended-upgrades updated `libexpat1` and the Kerberos libraries), which aborted the transfer at 1,198,981,120 bytes (`Transferred a partial
file`, a transient error). The retry one second later got `Login denied`, which isn't retried, for two reasons, either enough: the
token had expired, and `FtpDataServer` keeps tokens only in memory, so the restarted server didn't know it. The pass failed, kept the
partial file, and TD-591's `REST` loop began. So any download that breaks after its token has expired, or across a `registry_server`
restart, fails its pass, however transient the cause; and at this link speed every file above about 1.3 GB takes longer than 30 minutes. The token's lifetime also starts at acquire, not when the file's download starts, so with
`MAX_PARALLEL_DOWNLOADS` smaller than the file count a queued file can expire before its download begins.

Action Items:

- [ ] Let a retry get a fresh token (which also covers a restarted `registry_server`, whose in-memory tokens are gone): on
      `CURLE_LOGIN_DENIED` after a transfer had started (or before every retry), ask the registry for
      new credentials for that file (a small `POST /registry/{group}[/{kind}]/token` for a file the caller's `run_id` holds, or a
      re-acquire), then retry; `fetch()` needs a callback to get them, since it only sees the `FileToken`.
- [ ] Size the lifetime: issue a token valid for `max(FTP_TOKEN_TTL_MINUTES, size / a minimum rate)`, or start its clock at first
      login, so a queued file's token doesn't expire before its download starts.
- [ ] Until then (operations): raise `ftp_token_ttl_minutes` on `registry_server` (`PUT /admin/config`, live, persisted) above the
      longest download, e.g. 120.
- [x] Find why the server closed the data connection on November 26. (`registry_server`'s journal: restarted at 07:03:56 by the
      package system's service restarts; `RETR aborted … bytes_sent=1198981120`.)
- [x] Operations: stop the package system restarting the ADAI services on both hosts. (November 26, 2026: confirmed on the server
      host, unattended-upgrades ran 06:58-07:04:30 and upgraded `libexpat1` and the Kerberos libraries libcurl loads, then needrestart
      restarted the services using them; `/etc/needrestart/conf.d/adai.conf` with `$nrconf{override_rc}->{qr(^adai-)} = 0;` added on
      both hosts. ADAI services now need a manual restart after such an upgrade.)
- [x] Check whether ai-machine's restart at 06:30 the same day, which ended a two-day training pass, was its own unattended upgrade.
      (Yes: the same libraries, upgraded 06:26-06:31:17; `adai-trainer@lejepa` was restarted at 06:30:54.)
- [ ] Find why the link runs at 0.72 MB/s (plain FTP, so not TLS).
- [ ] Note: `registry_server` on the server host reads the legacy `/etc/adai/config.conf`, not `config.registry.conf` (TD-575's
      per-service config cutover).
- [ ] Test: a token that expires during a transfer that then breaks; the retry downloads with fresh credentials.

Location in code: `src/DataTransport.hpp` (`fetch()`'s retry loop), `src/RegistryServer.cpp` (`issue_token()` at acquire),
`src/FtpDataServer.hpp` (`TokenStore::authenticate()`).

Files to Modify:

- `src/DataTransport.hpp`, `src/RegistryTransport.hpp` (a fresh-token request)
- `src/RegistryServer.cpp`
- `tests/DataTransportFtpsTests.cpp`, `tests/registry_server_args_test.cpp`

Related: TD-591, TD-592, TD-593, TD-340 (archived), TD-377, TD-494 (archived: claim leases).

---

### TD-595: GPU Attention Training Copies Every Head's Rows One at a Time

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — code fix landed November 26, 2026; the measurement on ai-machine remains | Attention / GPU | November 26, 2026 | A run on ai-machine |

Description:
Found November 26, 2026, investigating why moving LeJEPA training onto the GPU (TD-585 to TD-590) only took ai-machine's world-model
pass from 0.26 to 0.56 samples per second (768 wide, 16 layers, 12 heads). TD-059's GPU per-head attention split Q, K and V into heads
with `gpu_slice_head_columns()`/`gpu_scatter_head_columns()` (and CrossAttention's `ca_` copies), which issue one device-to-device
copy per row per head: forward slices Q, K and V and gathers each head's output, backward slices the output gradient, V, Q and K and
gathers dQ, dK and dV. So the copies grow with rows × heads × layers. A trace on the oneAPI CPU device (`ADAI_GPU_TRACE_FILE`, a
16-layer, 4-head LeJEPA encoder) counted 32,225 GPU operations for one text, and over six texts 231,564 of 263,258 operations (88%)
were `matrix_copy_device_to_device_gpu`, against 10,791 matrix multiplies. Each is a separate submission, and with `GPU_QUEUE_DEPTH=8`
(TD-574) the host waits for the queue to drain before every eighth, so the GPU mostly waited on the host. At ai-machine's size
(12 heads, texts of a few hundred tokens) that is on the order of half a million copies per text (an estimate from the code).
TD-388 had already given the cached decode path `matrix_gemm_strided_gpu()`, which multiplies a head's column view in place.

Action Items:

- [x] Compute every head of `MultiHeadAttention::gpu_forward()`/`gpu_backward()` and `CrossAttention::gpu_forward()`/`gpu_backward()`
      with `matrix_gemm_strided_gpu()` on the heads' column views, writing each head's output and dQ/dK/dV columns in place; fold the
      softmax scale into the products' alpha; delete the slice/scatter helpers. (November 26, 2026. The same 16-layer trace: 4,321
      operations for one text instead of 32,225, 17,274 for a batch of four instead of 194,170; the test ran in 3.9 s instead of 15.4.)
- [x] Test outputs, input gradients and weight gradients against the CPU path with several heads, a causal mask and (cross-attention)
      a score bias and unequal lengths: `MultiHeadAttentionGPUContractTest.PerHeadTrainingMatchesTheCpu`,
      `CrossAttentionGPUContractTest.PerHeadTrainingMatchesTheCpu` (the weight gradients exactly, through their norms; computing
      cross-attention's dK from K instead of Q fails three checks). Nothing compared the GPU backward with the CPU's before.
- [ ] Measure samples per second of the world-model pass on ai-machine with the new package, against 0.56.
- [ ] Then re-measure `GPU_QUEUE_DEPTH` (8 since TD-574): with about 15 times fewer operations a deeper queue may cost no more resets.
- [ ] If the per-head products still dominate, batch the heads (one strided-batched GEMM per product instead of one per head).

Location in code: `src/MultiHeadAttention.cpp`, `src/CrossAttention.cpp` (`gpu_forward()`, `gpu_backward()`).

Related: TD-059, TD-388, TD-574, TD-585 to TD-590 (archived).

---

### TD-596: The GPU Memory Pool Grows Until It Fills the Budget

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — code fix landed November 26, 2026; checking the footprint on ai-machine remains | GPU / GPUManager | November 26, 2026 | A run on ai-machine |

Description:
Found November 26, 2026: qmassa showed ai-machine's world-model trainer (768 wide, 16 layers) holding 16 GB of the Arc, against
roughly 3-4 GB of weights, gradients, Adam moments and activations (an estimate). TD-390's SYCL pool keeps every block a `GPUMemory`
releases and reuses one only for the same `pool_size()`, whose 512-byte steps ran up to 1 MiB. A text's temporaries (attention
scores, activations, gradients) are sized by its token count, so texts of varying length keep asking for new sizes, and the pool was
trimmed only when live plus cached memory reached the budget, `GPU_MEMORY_FRACTION` (0.9) of the card: 22 GB on ai-machine. Measured
on the oneAPI CPU device (768 wide, 12 heads, 4 layers, 40 texts, a 1.6 GB budget): about 600 MB live throughout, while the cache
grew by about 30 MB per text to the budget (972 MB cached at text 31), was freed all at once (a full queue wait) and grew again. So
the trainer's footprint swings up to the budget, which can starve any other process on the card, and nothing reported live and
cached memory. CUDA's stream-ordered pool had the same shape: its release threshold was the whole budget.

Action Items:

- [x] Coarser size steps (512 bytes only up to 64 KiB, then an eighth of the next power of two), and a request takes the smallest
      cached block up to a quarter larger, returned at its own size (`live_block_size_`).
- [x] Cap the cache: `GPU_POOL_CACHE_MB` (default 1024, 0 = none; `incremental_trainer` and `chatbot_api_server` apply it). Past it,
      the least recently released blocks are evicted to three quarters of it and freed, without waiting, once a barrier queued
      behind them completes. CUDA: the pool's release threshold.
- [x] Report it: `GPUManager::memory_summary()` ("X MB in use (peak Y MB), Z MB cached (limit L MB)"), logged once per epoch by
      the LeJEPA pass and `ChatbotTrainer`. (November 26, 2026; the same 40 texts: cached memory levels off at 224-253 MB, live and
      its peak unchanged at about 600 and 700 MB. Tests: `GPUManagerTest.TheCacheStaysUnderItsLimit`,
      `ARequestTakesASlightlyLargerCachedBlock`, `PeakAndSummaryReportWhatAnEpochHeld`, `GPUManagerPoolSize.FineStepsEndAt64KiB`,
      `ConfigTest.GpuPoolCacheDefaultFileEnvAndRange`.)
- [ ] On ai-machine: the trainer's "[lejepa] GPU memory" lines and qmassa with the new package; record what training really holds.

Location in code: `src/gpu/sycl/GPUUtils_SYCL.hpp` (`allocate_block()`, `release_block()`, `evict_cached_locked()`),
`src/gpu/GPUUtils.hpp` (the release threshold).

Related: TD-390 (archived), TD-595, TD-383.

---

### TD-597: A GPU Build Can't Train a Chatbot Without a GPU

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / ChatbotTrainer | November 26, 2026 | 1-2 hours |

Description:
Found November 26, 2026, running `SighupTests` in the SYCL build on the oneAPI CPU device (it normally runs in the CPU-only debug
build). Since TD-468 (October 19, 2026) `ChatbotTrainer::train()` calls `bind_gpu_optimizer()` unconditionally in any
`ADAI_ENABLE_GPU` build, which uploads the model (`EncoderDecoderModel::gpu_init_training()`) without checking that a GPU was
initialized. With `GPU_ENABLED=false`, or when initialization found no device (which `incremental_trainer` logs as a warning and
carries on), every chatbot pass fails with `GPU not initialized. Call GPUManager::initialize() first.` instead of training on the
CPU. Three `SighupTests` fail in a SYCL build for this reason (`AFirstTrainBuildsTheVocabularyAndExceptionsExitOne`,
`AChatbotPassReclaimsACrashedPassesFilesWithoutMns`, and the one at `sighup_test.cpp:881`); they pass in the debug build.

Action Items:

- [ ] Bind the device optimizer only when `GPUManager::is_available()`, and check that every GPU branch of `train()`/`train_epoch()`
      (and validation) falls back to the host path when `gpu_optimizer_` is null; or refuse clearly at start-up if GPU builds are
      meant to require a GPU.
- [ ] Run `SighupTests` (and the trainer suites) in the SYCL build with `GPU_ENABLED=false`, and add them to the GPU test package.

Location in code: `src/ChatbotTrainer.cpp` (`train()`, `bind_gpu_optimizer()`).

Related: TD-468 (archived), TD-547.

---

## Resolved Items

481 items resolved (256 of them moved there on November 14, 2026). See [archive/TECHNICAL_DEBT_RESOLVED.md](../archive/TECHNICAL_DEBT_RESOLVED.md)
for full details, newest first, followed by the archived overview notes and execution-order notes.

---
## Future Improvements

These are lower-priority enhancements that don't currently block development:

### Performance Optimizations

1. **Memory Pool for Matrix Allocations**
   - Reduce allocation overhead during forward/backward passes
   - Pre-allocate memory for common matrix sizes
   - Reduce memory fragmentation

2. **Monitor Augmentation Performance** (from Priority 2 completion) — **withdrawn October 11, 2026:** the
   augmentation code and its benchmark were retired with `src/EfficientBatching.hpp` (TD-355).
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
   - **Status (October 8, 2026):** The class and `--batched-inference` flag exist, but the engine only queues and serializes requests; it never runs a batched forward pass. The model-level batching below is still unbuilt and blocked on TD-171. See [TD-211](#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model) before picking this up. **Planned as v2 (October 8, 2026):** [real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md).
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

4. **Attention Head Parallelism (Priority 4)** — **done:** `MultiHeadAttention::forward_parallel()` computes each head on its own
   OpenMP thread (`#pragma omp parallel for` over the heads, TD-059), with results bit-for-bit equal to the serial loop (TD-467).
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

3. **Model State Persistence on Shutdown** (Step 2 Enhancement) — **moot (November 14, 2026):** `chatbot_api_server` never changes
   the model's weights, so there's nothing to save at shutdown; the one state it does change while serving, the hippocampal memory, has
   been saved periodically and at shutdown since TD-250. Training checkpoints are the trainer's (`AUTO_SAVE_*`, TD-416's drain on SIGTERM).
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

2. **Metrics Endpoint for Prometheus** (Step 5 Enhancement) — partly there: `metrics_api_server` can write training metrics as a
    Prometheus text file (`metrics_enable_prometheus`, off by default); no daemon serves a `/metrics` endpoint.
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
4. Mark each action item done (`- [x]`) as it lands. A fix that has landed in code with something still to do (a run on ai-machine,
   a retraining, a measurement) keeps the item here as "Open — code fix landed <date>", with the remaining step unchecked.
5. Once every action item is done, move the entry to the top of
   [TECHNICAL_DEBT_RESOLVED.md](../archive/TECHNICAL_DEBT_RESOLVED.md) with a `Resolution Date | Component | Resolved By` table,
   remove it from the [Table of Contents](#table-of-contents), point links to it at the archive, and update the
   [Overview](#overview) counts, the [Recommended Execution Order](#recommended-execution-order) and the [Statistics](#statistics).
   Dated overview notes about it move to the archive's archived overview notes once every item they mention is resolved.
6. Close the related GitHub issue, if one was opened.
7. Remove or update code comments referencing the item (`TODO: See TD-NNN` tags; a `// TD-NNN` comment that explains a design
   choice can stay).

---

## Statistics

### By Priority

Recomputed November 18, 2026 from the 99 `### TD-NNN` entries under [Active Technical Debt](#active-technical-debt). Re-derive it from
that list when an item moves or a new one is filed.

|Priority|Count|Percentage|
|----------|-------|------------|
|High|7|7%|
|Medium|20|20%|
|Low|72|73%|

**Total Active Items:** 99

### By Component

|Component|Count|
|----------------------|-------|
|GPU / Kernels|4|
|Attention / CrossAttention|2|
|GPU / GPUManager|3|
|GPU / GPUMatrix|3|
|Attention / LoRA|2|
|Build / SIMD|2|
|Core Model Architecture|2|
|Data / FtpDataServer|2|
|Data / dataset_manager CLI|2|
|GPU / Testing|1|
|Inference / Serving|2|
|Inference / Serving / API|2|
|MNS / Server|2|
|Memory / HippocampalMemory|2|
|Metrics / Reporter|2|
|NLP / Tokenizer|2|
|RLHF / PPOOptimizer|2|
|RLHF / RLHFTrainer|2|
|Transformer / DecoderBlock|1|
|Transformer / EncoderBlock|2|
|Transformer / FeedForward|3|
|Transformer / LLMEncoder|2|
|Advanced / Quantization|1|
|Advanced Features / Integration|1|
|Android / CI|1|
|Android / Ops dashboard|1|
|Android / Testing|1|
|Android / `app`, `opsdashboard`|1|
|Build / Versioning|1|
|Client / GUI|1|
|Core / Batching|1|
|Core / JSON|1|
|Core / Matrix|1|
|Core / Optimizer|1|
|Data / DataFetcher|1|
|Data / DataTransport|1|
|Data / DatasetRegistry|1|
|Data / JSONL Format|1|
|Data / ParquetReader|1|
|Data / RegistryTransport|1|
|Documentation|1|
|GPU / Training|1|
|Inference / Profiler|1|
|Inference / Serving / Observability|1|
|Inference / Serving / Server|1|
|Layers / PositionalEncoding|1|
|Layers / TokenEmbedding|1|
|MNS / Client|1|
|Metrics / API|1|
|Metrics / Database|1|
|Metrics / GenerationQuality|1|
|Models / EncoderDecoderModel|1|
|Models / LanguageModelHead|1|
|Models / ModelSerializer|1|
|NLP / ConversationContext|1|
|Platform / PortableTime|1|
|RAG / DocumentStore|1|
|RLHF / RewardModel|1|
|Tests / Trainer admin API|1|
|Tizen / Metrics dashboard|1|
|Tooling / File status standard|1|
|Tooling / Toolchain|1|
|Training / ChatbotTrainer|1|
|Training / Data Generation|1|
|Training / Deployment|1|
|Training / Deployment / Tooling|1|
|Training / IncrementalTrainer|1|
|Training / Metrics / Core|1|
|Training / MetricsTracker|1|
|Training / Operations|1|
|Training / StartupSweep|1|
|Transformer / Decoder|1|
|World Model / LeJEPAEncoder|1|
|World Model / SIGReg|1|

### Effort Distribution

Each entry's own estimate, made when it was filed. For the 28 entries whose fix has landed in code, most of it is spent; what remains
there is a run, a measurement or a retraining.

|Effort Range|Count|
|--------------|-------|
|0-2 hours|50|
|2-4 hours|29|
|4-8 hours|7|
|8+ hours|7|
|Not estimated|6|

**Total Estimated Effort (Active Items):** 254-446 hours as originally estimated. Not counted, having no estimate in hours: TD-014 and
TD-039 (none given), TD-171 and TD-563 (large: an architecture project and the retraining), TD-554 (1-2 days) and TD-573 (2-4 days,
plus devices and a release).

### Future Enhancements Summary

**Open Future Enhancements:** 16 of the 19 listed (Monitor Augmentation Performance was withdrawn with `EfficientBatching.hpp`,
Attention Head Parallelism is done, and Model State Persistence on Shutdown is moot).

By Category:

- Configuration and Service Management: 3 open (JSON configuration, configuration profiles, graceful reload)
- Logging and Observability: 3 open
- Container and Deployment: 5 open (the Prometheus metrics endpoint partly there as a file export)
- Performance Optimizations: 2 open (memory pool, batched inference)
- Code Quality: 1 open
- Developer Experience: 2 open

By Priority:

- High: 0 items
- Medium: 3 open (Batched Inference, JSON Log Output, Metrics Endpoint)
- Low: 13 open

Recently Completed: resolved items are listed newest first at the top of
[TECHNICAL_DEBT_RESOLVED.md](../archive/TECHNICAL_DEBT_RESOLVED.md), each with its resolution date. The most recent, on November 13,
2026, are TD-300, TD-299, TD-426 and TD-467.

### Deferred Decisions Summary

**Total Deferred Decisions:** 3

- AMD Radeon / ROCm-HIP GPU Backend — Not Pursued (September 7, 2026)
- ThreadSanitizer "Data Races" in Matrix.cpp's OpenMP-Parallelized Code — Confirmed Tool Limitation, Not a Bug (September 12, 2026)
- AttentionHeadBenchmark's Apparent Hang — Confirmed Host CPU Contention, Not a Bug (September 13, 2026)

---

## References

- [Resolved items and archived notes](../archive/TECHNICAL_DEBT_RESOLVED.md)
- [Process Improvement Plan](../archive/PROCESS_IMPROVEMENT_PLAN.md) (archived) - Section 10: Technical Debt Items
- [Contributing Guide](workflow/contributing.md) - Code quality standards
- [File status standard](file-status-standard.md) - the `@adai-status` tags the items cap
- [GitHub Issues](https://github.com/rjv717/adai/issues?q=is%3Aissue+label%3Atechnical-debt) - debt tracked as issues, where opened

---

**Maintenance Note:** This document should be reviewed monthly and updated as items are added or resolved.
