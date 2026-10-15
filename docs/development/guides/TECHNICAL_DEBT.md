# Technical Debt Tracker

This document tracks all known technical debt items, TODOs, and improvement opportunities in the ADAI codebase. Items are prioritized and linked to GitHub issues for tracking.

## Overview

**Last Updated:** October 14, 2026
**Total Items:** 256
**High Priority:** 28
**Medium Priority:** 111
**Low Priority:** 117
**Future Enhancements:** 19
**Resolved Items:** 208
**Deferred Decisions:** 3

**October 11, 2026:** Filed [TD-304](#td-304-session-counters-can-be-duplicated-and-daemonconfigstore-has-no-busy-timeout) through [TD-308](#td-308-session-counters-reuse-the-config-stores-table-and-scan-it-per-allocation) from the code-traced
[DaemonConfigStore.md](../reference/source/DaemonConfigStore.md). Two are MEDIUM: the store has no
SQLite busy timeout and standalone session numbers are allocated with a non-atomic read-modify-write,
so concurrent writers fail instantly and two processes can be handed the same session number
(TD-304, verified: 284 of 400 concurrent increments threw and 3 were lost); and a rejected or failed
`PUT /admin/config` still partly applies, in memory or in the DB (TD-305). Three are LOW
(TD-306–308): missing range checks on two daemons' PUTs, hidden `load_all()` read errors, and the
counter store reusing the config table.

Also October 11, 2026: filed [TD-309](#td-309-gutenberg-sentence-splitting-can-crash-registry_server) through [TD-314](#td-314-datafetcher-source-coverage-and-hygiene-gaps) from the code-traced
[DataFetcher.md](../reference/source/DataFetcher.md). Two are **HIGH** and reachable through
`registry_server`'s fetch endpoints: the Gutenberg sentence regex segfaults on long unpunctuated text,
so one `fetch/gutenberg` request can crash the registry daemon (TD-309, verified at 100,000
characters); and the HuggingFace `split` can contain `..`, so `fetch/huggingface` writes outside the
data directory (TD-310, verified). Three are MEDIUM: Gutenberg pairs are next-sentence non-sequiturs
(TD-311, verified), downloads go through shell `curl` commands with the HF token in argv and no
timeouts (TD-312), and partial or concurrent HuggingFace downloads are cached as complete (TD-313).
TD-314 is LOW.

Also October 11, 2026: filed [TD-315](../archive/TECHNICAL_DEBT_RESOLVED.md#td-315-datasethpp-is-tagged-stable-but-has-no-production-caller) through [TD-320](../archive/TECHNICAL_DEBT_RESOLVED.md#td-320-dataset-hygiene-tolower-ub-unchecked-saves-console-output) from the code-traced
[Dataset.md](../reference/source/Dataset.md). `src/Dataset.hpp` is tagged `stable` but has no
production caller (only tests and examples), so the owner decision in TD-315 (retire it, or keep it
and fix the rest) comes first. The one MEDIUM item is TD-317 (verified): a zero `batch_size` loops
forever, `num_bins = 0` segfaults, and `clear()` leaves stale k-fold indices that read freed memory.
The rest are LOW: parsers that silently lose or mangle data (TD-316, verified), O(n²) `add_sample()`
and split bookkeeping (TD-318), `LazyDataset` phantom samples (TD-319), and hygiene (TD-320).

Later October 11, 2026, per the owner's decision on TD-315: **retired `src/Dataset.hpp`** along with
`tests/dataset_test.cpp` and the `dataset_enhanced_example`/`dataset_batch_processing_example`
examples (`EnhancedTrainingExample` was kept, reworked to split its sample data inline). This
closes [TD-315](../archive/TECHNICAL_DEBT_RESOLVED.md#td-315-datasethpp-is-tagged-stable-but-has-no-production-caller) through [TD-320](../archive/TECHNICAL_DEBT_RESOLVED.md#td-320-dataset-hygiene-tolower-ub-unchecked-saves-console-output) and [TD-219](../archive/TECHNICAL_DEBT_RESOLVED.md#td-219-datasets-batch-wrappers-have-wrong-doc-examples-and-a-partial-stats-helper); all seven are in the resolved archive. The removed
capabilities (k-fold, stratified splitting, augmentation and preprocessing hooks, multi-format
import, lazy indexing) are recorded under [TD-014](#td-014-llm-operations-and-training-tooling-suite)
for possible recreation in `adai-data-prep`/`adai-eval`.

Also October 11, 2026: filed [TD-321](#td-321-dataset_manager-treats---count--0-as-all) through [TD-325](#td-325-dataset_manager-huggingface-positional-arguments-shift-silently) from the code-traced
[DatasetManagerArgs.md](../reference/source/DatasetManagerArgs.md). The one MEDIUM item is TD-321
(verified): `dataset_manager assign`/`migrate` treat `--count` ≤ 0 as "all", so a typo like
`--count -2` assigns or migrates the entire pending pool. The rest are LOW: unknown flags silently
become file targets (TD-322, verified), partial and non-positive numbers are accepted (TD-323,
verified), `segment --ranges` can overflow or overlap (TD-324, verified), and `huggingface`'s
positional arguments shift silently (TD-325).

Also October 11, 2026: filed [TD-326](#td-326-dataset_manager-migrate-duplicates-segmented-data-across-pools) through [TD-332](#td-332-operator-guide-covers-only-8-of-15-dataset_manager-commands) from the code-traced
[DatasetManagerTool.md](../reference/source/DatasetManagerTool.md). One is **HIGH** (verified):
`dataset_manager migrate` drops segment ranges and ignores failed source deletes, so migrating a
segmented file puts the whole file in the new pool while every segment stays in the legacy pool,
which trains it twice (TD-326). Two are MEDIUM: `remove` reports success and exits 0 against
`registry_server` while nothing changes (TD-327, verified), and only `assign` uses the MNS
`run_group` (TD-329). The rest are LOW (TD-328, TD-330–332): relative local paths (verified),
output hygiene, no command-level tests, and an operator guide covering 8 of 15 commands.

Also October 11, 2026: filed [TD-333](#td-333-segment-claims-arent-released-on-failure-and-trained-records-ignore-segments) through [TD-336](#td-336-legacy-training-file-parser-truncates-responses-bom-breaks-jsonl-detection) from the code-traced
[DatasetRegistry.md](../reference/source/DatasetRegistry.md). One is **HIGH** (verified): the
training lifecycle records success and failure by bare path, so a failed pass leaves its TD-205
segment claimed (stuck after a `retrain` or any standalone pass), and a trained segment marks the
whole file trained while later segments' samples go unrecorded (TD-333). Three are LOW: trained
sample counts are always 0 (TD-334), dead code and an unused checksum (TD-335), and legacy-parser
truncation plus BOM detection (TD-336).

Also October 11, 2026: filed [TD-337](#td-337-datatransport-names-downloads-by-basename-so-segments-and-same-named-files-collide) through [TD-340](#td-340-datatransport-robustness-and-header-hygiene) from the code-traced
[DataTransport.md](../reference/source/DataTransport.md), tested end-to-end against a local
`registry_server` with FTP enabled. Three are MEDIUM: downloads are named by basename, so TD-205
segments of one file download it N times concurrently into one path, and same-named files collide
(TD-337, verified); a same-name, same-size stale file is reused without checking the token checksum
(TD-338, verified); and FTPS doesn't verify the registry's certificate (TD-339). TD-340 is LOW.

Also October 11, 2026: filed [TD-341](#td-341-world-model-and-hippocampal-gates-never-train-in-production) through [TD-345](#td-345-decoderblock-persistence-gaps-for-gated-state) from the code-traced
[DecoderBlock.md](../reference/source/DecoderBlock.md). One is **HIGH** (verified): the TD-180
world-model/hippocampal gates aren't optimizer parameters and `ChatbotTrainer` only calls
`optimizer->step()`, so the gates stay at 0 and the gated cross-attentions never learn. World-model
injection and hippocampal memory are no-ops in every trained model (TD-341). Two are MEDIUM:
KV-cached generation and the GPU path skip the gated paths (TD-342), and hippocampal memory is
updated on every forward pass rather than once per decode step (TD-343). Two are LOW: unused dropout
(TD-344) and gated-state persistence gaps (TD-345).

Also October 11, 2026: filed [TD-346](#td-346-llmdecoderupdate_weights-never-updates-the-final-norm) through [TD-348](#td-348-llmdecoder-dead-code-and-robustness-gaps) from the code-traced
[Decoder.md](../reference/source/Decoder.md). One is MEDIUM (verified): the KV-cached and full decoder
forwards diverge once a sequence passes `max_seq_length`, so over-length generation depends on the
strategy (TD-347). Two are LOW: `update_weights()` never trains the final norm (TD-346, verified) and
dead code/robustness gaps (TD-348). TD-343 was amended: memory is updated once per injected layer,
not just once per forward.

Also October 11, 2026: filed [TD-349](#td-349-rag-indexes-whole-files-but-embeds-only-their-first-max_seq_length-tokens) through [TD-351](#td-351-documentstore-embedding-and-housekeeping-gaps) from the code-traced
[DocumentStore.md](../reference/source/DocumentStore.md). Two are MEDIUM: RAG indexes whole files but
embeds only their first `max_seq_length` tokens, so later content can't be retrieved (TD-349,
verified), and `retrieve()` isn't safe concurrently, which is a prerequisite for fixing TD-230
(TD-350). TD-351 is LOW.

Also October 11, 2026: filed [TD-352](../archive/TECHNICAL_DEBT_RESOLVED.md#td-352-parallel-data-augmentation-isnt-reproducible) through [TD-355](../archive/TECHNICAL_DEBT_RESOLVED.md#td-355-efficientbatching-is-an-unused-duplicate-of-batchprocessor) from the code-traced
[EfficientBatching.md](../reference/source/EfficientBatching.md), all LOW because nothing in production
calls the header: OpenMP augmentation isn't reproducible across thread counts (TD-352, verified),
token masking inserts `EOS` (TD-353, verified), bad arguments hang or overflow the heap (TD-354,
verified), and the header duplicates `BatchProcessor` unused (TD-355, owner decision on merge or
retire).

Later October 11, 2026, per the owner's decision on TD-355: **retired `src/EfficientBatching.hpp`**
along with `tests/datapipeline_test.cpp`, `examples/DataPipelineExample.cpp` and
`benchmarks/AugmentationBenchmark.cpp`, and removed `BatchedInferenceConfig`'s dead `padding_strategy`
field (its only tie to the header; `BatchedInferenceEngine.hpp` is now `@adai-version 2.0.0`). This
closes [TD-352](../archive/TECHNICAL_DEBT_RESOLVED.md#td-352-parallel-data-augmentation-isnt-reproducible) through [TD-355](../archive/TECHNICAL_DEBT_RESOLVED.md#td-355-efficientbatching-is-an-unused-duplicate-of-batchprocessor). The token-budget bucketing algorithm is preserved as a design note
in [real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md) §4.1, and the augmentation
operations are recorded under [TD-014](#td-014-llm-operations-and-training-tooling-suite).

Also October 11, 2026: filed [TD-356](#td-356-encoderblock-gradient-norm-and-clipping-ignore-layernorm-header-describes-post-ln) and [TD-357](#td-357-encoderblock-learning-rate-handling-unchecked-loads-and-console-output) from the code-traced
[EncoderBlock.md](../reference/source/EncoderBlock.md), both LOW (`backward()` itself was verified
correct): gradient norm/clipping ignore `LayerNorm` and the header still describes Post-LN (TD-356),
and learning-rate handling, unchecked loads and console output (TD-357). TD-344 was amended: the
encoder's `dropout` is never applied either.

Also October 11, 2026: filed [TD-358](#td-358-gpu-training-feeds-unshifted-targets-teaching-the-model-to-copy-instead-of-predict) through [TD-363](#td-363-encoderdecodermodel-clone-load-and-world-model-input-gaps) from the code-traced
[EncoderDecoderModel.md](../reference/source/EncoderDecoderModel.md). One is **HIGH** and serious
(verified by CPU reproduction): `gpu_forward()` feeds the unshifted targets to the decoder, so **GPU
training teaches the model to copy its input token instead of predicting the next one**. Loss looks
excellent but generation fails, and every ai-machine (GPU) checkpoint needs retraining after the fix
(TD-358). Two are MEDIUM: `generate_response()` ignores `max_length` (TD-359, verified), and CPU and
GPU generation behave differently (TD-362). Three are LOW: the no-optimizer path never updates the
encoder (TD-360, verified), a double `BOS` in training targets (TD-361, verified), and
clone/load/world-model-input gaps (TD-363). TD-343 was amended: training-time generation also writes
hippocampal memory.

Also October 11, 2026: filed [TD-364](#td-364-llmencoderencode-input-doesnt-match-the-models-encoder-input) through [TD-367](#td-367-llmencoder-console-output-and-unchecked-weight-io) from the code-traced
[LLMEncoder.md](../reference/source/LLMEncoder.md). Two are MEDIUM and verified: `LLMEncoder::encode()`
adds `BOS`/`EOS` while `EncoderDecoderModel` trains its encoder without special tokens, so pipeline
inference feeds the trained encoder shifted, out-of-distribution input, and truncation drops the `EOS`
it just added (TD-364); and chat input is never truncated to `max_seq_length`, so a 60-token input on a
16-position model ran with 44 positions that had no positional encoding (TD-365). Two are LOW: serving
encoders cache activations nobody reads (TD-366), and console output plus unchecked weight I/O (TD-367).
TD-360 was corrected: `LLMEncoder::set_learning_rate()` exists; only `update_weights()` is missing.

**October 12, 2026:** Filed [TD-368](#td-368-weight-initialization-cant-be-seeded-or-reproduced) through [TD-371](#td-371-residual-branch-output-isnt-scaled-for-depth) from the code-traced
[FeedForward.md](../reference/source/FeedForward.md) (`backward()` itself was verified correct against
finite differences). Two are MEDIUM and verified: weight initialization is seeded from
`std::random_device` with no way to set a seed, here and in `Matrix::randomize()`, so training runs
and bug reproductions can't be replayed (TD-368); and `load_weights()` loads a truncated file without
an error, keeping the old weights for everything past the cut (TD-369). Two are LOW: missing argument
and shape checks, a GPU null dereference and console output (TD-370); and an owner decision on scaling
residual-branch output for depth, since a fresh FFN's output std is about 1.35× its input (TD-371).

Also October 12, 2026: filed [TD-372](#td-372-ftp-data-connections-are-accepted-from-any-host-so-files-can-be-stolen) through [TD-377](#td-377-ftpdataserver-token-semantics-and-setup-gaps) from the code-traced
[FtpDataServer.md](../reference/source/FtpDataServer.md), verified against a live server built from the
header and driven by Python FTP/TLS clients. One is **HIGH**: the PASV data connection is accepted
from any host, so a third party that connects first receives the whole file (300,000 of 300,000
bytes), even under FTPS with `PROT P`; the token is consumed, so the real trainer's download fails
too (TD-372). Three are MEDIUM: `--ftps` doesn't require `AUTH TLS` or `PROT P` (TD-373, verified);
logging in twice on one connection leaks a rate-limit slot until restart (TD-374, verified); and
there's no connection cap, idle timeout or line-length limit, and connection threads outlive `stop()`
(TD-375). Two are LOW: `stop()` takes up to 60 s, `FEAT` is written unencrypted after `AUTH TLS`,
and `start()` failures are only logged (TD-376, verified); token semantics and setup gaps (TD-377).

Also October 12, 2026: filed [TD-378](#td-378-bleu-smoothing-inflates-unrelated-answers-and-zeroes-correct-short-ones) through [TD-380](#td-380-generation-quality-scores-arent-comparable-or-well-sampled) from the code-traced
[GenerationQualityMetrics.md](../reference/source/GenerationQualityMetrics.md). One is MEDIUM and
verified: BLEU applies add-1 smoothing to every order including unigrams (mislabelled "Lin & Och
method 1"), so an unrelated 5-word answer scores BLEU-4 0.23, a scrambled one 0.36, and an exactly
correct 3-word answer 0.0067; the per-epoch BLEU-4 can rank outputs backwards (TD-378). Two are LOW:
`bleu1`/`bleu2` are dropped before the metrics DB, whose columns always hold `-1` (TD-379); and the
scores aren't comparable with published numbers or the BLEU/ROUGE lesson's 0–100 ranges, with a
noisy 10-sample default (TD-380).

Also October 12, 2026: filed [TD-381](#td-381-chatbot_api_server-ignores-gpu_strategy) through [TD-383](#td-383-gpumanager-lifecycle-gaps-set_device-cleanup-initialize) from the code-traced
[GPUUtils.md](../reference/source/GPUUtils.md) (CUDA `GPUManager`/`GPUMemory`; no CUDA device was
available, so all three are by inspection). Two are MEDIUM: `chatbot_api_server` never passes
`GPU_STRATEGY` to GPU initialization, so it always uses the low-priority stream although
`config.chatbot.conf` ships `full` (TD-381); and `GPUManager` has no locking while
`Matrix::operator*` dispatches to the shared GPU stream from any thread, including `ChatbotTrainer`'s
supposedly CPU-only async quality thread during training (TD-382). One is LOW: `set_device()`,
`cleanup()`, `initialize()` and `probe_diagnostic()` lifecycle gaps (TD-383).

Also October 12, 2026: filed [TD-384](#td-384-shared-memory-race-in-gpu-softmax-layernorm-and-cross-entropy-kernels-cuda-and-sycl) through [TD-386](#td-386-gpu-kernel-input-checks-and-dead-batch-functions) from the code-traced
[MatrixGPU_cu.md](../reference/source/MatrixGPU_cu.md) (no CUDA device available; all by inspection,
with the training kernels' math checked by hand). One is **HIGH**: softmax, LayerNorm forward and
cross-entropy loss/grad read `scratch[0]` and then reuse the same shared buffer for the next reduction
without a barrier, a data race that can silently corrupt a row (possibly `NaN`) in attention,
LayerNorm and the loss. The SYCL backend that ai-machine trains on has the identical pattern (TD-384).
Two are LOW: `matrix_sum_gpu` allocates and synchronizes per call, every training step (TD-385); and
unchecked kernel inputs plus uncalled batch functions (TD-386).

Also October 12, 2026: filed [TD-387](#td-387-gpu-generation-fails-on-input-longer-than-max_seq_length) through [TD-389](#td-389-gpumatrix-shape-checks-uninitialized-memory-and-untested-gpu-kv-caches) from the code-traced
[MatrixGPU.md](../reference/source/MatrixGPU.md) (by inspection; no GPU available). One is MEDIUM:
GPU generation sizes its cross-attention KV cache at `max_seq_length`, but encoder input isn't
truncated (TD-365), so a chat input longer than that throws `out_of_range` on a GPU server while the
CPU path answers (TD-387). Two are LOW: `GPUKVCache` reallocates and copies the whole populated cache
every decode step (TD-388); and unchecked `GPUMatrix` shapes, uninitialized device memory and no tests
for the GPU KV caches (TD-389).

Also October 12, 2026: filed [TD-390](#td-390-sycl-deferred-frees-and-blocking-copies-stall-the-in-order-queue) through [TD-392](#td-392-sycl-gpumemory0-throws-where-cuda-succeeds) from the code-traced
[GPUUtils_SYCL.md](../reference/source/GPUUtils_SYCL.md), the SYCL manager that ai-machine trains on.
Two are MEDIUM: every `GPUMemory` destruction queues a `host_task` free, which on an in-order queue is
a device→host→device round trip, and every upload/download drains the queue (TD-390, verified on a CPU
queue: 1.35 ms per alloc/kernel/free iteration vs 0.051 ms blocking and 0.014 ms kernel-only); and
GPUs are collected from every SYCL platform, so one Arc card appears once per backend and
`GPU_DEVICE_ID` silently picks by platform order (TD-391). One is LOW: `GPUMemory(0)` throws on SYCL
but not CUDA (TD-392, verified). TD-382 and TD-383 were amended for the SYCL manager.

Also October 12, 2026: filed [TD-393](#td-393-sycl-sub-group-reduction-is-wrong-when-the-sub-group-size-is-8) and [TD-394](#td-394-gpu-kernel-tests-cant-run-without-a-gpu) from the code-traced
[MatrixGPU_SYCL_cpp.md](../reference/source/MatrixGPU_SYCL_cpp.md), the first time any GPU kernel in
this repository has actually been run (a scratch copy of the SYCL backend on the oneAPI CPU device).
Both are MEDIUM: the sub-group sum reduction is correct only when the compiler picks a sub-group size
≥ 16; at 8 (the CPU device, Arc A-series) it drops ¾ of the partials per level, so 256 ones sum to 64
(TD-393, verified; probably latent on ai-machine's Xe2, which has no size 8); and no test can run the
GPU kernels without a GPU, which is how this and TD-384 went unnoticed (TD-394). **TD-384 is now
verified**: the race reproduces deterministically on the CPU device (LayerNorm forward 0.32 relative
error) and adding the barrier fixes all four kernels. TD-385 and TD-386 were amended for SYCL.

Also October 12, 2026: filed [TD-395](#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) (LOW) from the code-traced
[MatrixGPU_SYCL.md](../reference/source/MatrixGPU_SYCL.md): `GPUMatrix` is maintained twice, once per
backend, although only `copy()`/`zero()` differ, and the SYCL `copy()` drains the queue. Running the
SYCL `GPUMatrix` and the shared `GPUKVCache` on the oneAPI CPU device (their first run anywhere)
verified TD-387 at the cache level and TD-389's SYCL findings; both were amended.

**October 13, 2026:** Filed [TD-396](#td-396-hippocampal-memory-eviction-starves-new-memories) through [TD-398](#td-398-hippocampalmemory-per-step-copies-and-unchecked-saves) from the code-traced
[HippocampalMemory.md](../reference/source/HippocampalMemory.md). One is MEDIUM and verified by
simulation with the real class: least-used eviction starts new memories at coverage 0, so with
near-uniform attention (likely while TD-341 keeps the hippocampal attention untrained) the newest
memory is always evicted next and the buffer freezes on its first `capacity − 1` entries (TD-396).
Two are LOW: the swap file grows without bound on a long-running server (TD-397); and per-step
copies plus unchecked saves (TD-398).

Also October 13, 2026: filed [TD-399](#td-399-imetricsreporter-buffering-and-ordering-contract-is-undocumented) and [TD-400](#td-400-nullmetricsreporter-keeps-reporting-only-work-switched-on) (both LOW) from the code-traced
[IMetricsReporter.md](../reference/source/IMetricsReporter.md): the interface doesn't say which methods
`MetricsPushClient` buffers until `end_epoch()` or that they must be called between `start_epoch()`
and `end_epoch()`, and `AbnormalSample` texts travel untruncated (TD-399); and `ChatbotTrainer`
treats a non-null reporter as "metrics enabled" while `incremental_trainer` always passes one
(`NullMetricsReporter` when there's no server), so reporting-only work still runs (TD-400; cost
measured as negligible on a tiny model).

Also October 13, 2026: filed [TD-401](#td-401-incremental_trainer-flag-parsing-crashes-or-silently-misreads-bad-values) through [TD-404](#td-404-derive_run_id-can-collide-across-hosts) from the code-traced
[IncrementalTrainerArgs.md](../reference/source/IncrementalTrainerArgs.md), checked against the built
binary. One is MEDIUM: `--admin-port abc` aborts `incremental_trainer` (uncaught `std::stoi`),
`8084x` is accepted, a mistyped `--objective=` silently trains the chatbot, and a flag missing its
value becomes a positional argument (TD-401, verified). Three are LOW: `--dataset-kind` is
unvalidated (a `LocalTransport` path can leave `session_dir`) (TD-402); `trainer_service` drops
`--objective`/`--dataset-kind` (TD-403); `derive_run_id()` can collide across hosts and is always used
by the LeJEPA path (TD-404).

Also October 13, 2026: filed [TD-405](#td-405-each-training-pass-restarts-from-the-best-ever-checkpoint-not-the-latest) through [TD-410](#td-410-incrementaltrainer-session-bookkeeping-and-hard-coded-training-policy) from the code-traced
[IncrementalTrainer.md](../reference/source/IncrementalTrainer.md), verified by constructing the real
class against scratch session directories with real checkpoints. Three are **HIGH**: every pass
restarts from the lowest validation loss across all history although each pass validates on
different data, so later passes are discarded whenever new data validates worse (TD-405, verified);
a paused or crashed pass's auto-save is deleted and its session dropped on the next start, and its
in-progress best snapshot is never found (TD-406, verified); and checkpoint deletion removes only the
top-level sidecars, 0.1 of 29.6 MB, so retention never bounds `SESSION_DIR` (TD-407, verified). Two
are MEDIUM: the `latest`/`best` symlinks always dangle (TD-408) and the tokenized cache key has no
preprocessing version (TD-409). One is LOW: session bookkeeping and hard-coded policy (TD-410).

Also October 13, 2026: filed [TD-411](#td-411-resume-every-trainer_service-pass-never-downloads-over-ftp) through [TD-416](#td-416-incremental_trainer-command-hygiene) from the code-traced
[IncrementalTrainingTool.md](../reference/source/IncrementalTrainingTool.md), checked against the built
`incremental_trainer`/`dataset_manager`. Two are **HIGH**: `resume`, the command `trainer_service` runs
every pass, never downloads over FTP and uses registry-host paths (TD-411); and a pass that loads zero
samples returns success, so its files are marked trained (TD-412, verified). Together, an always-on
service on a remote FTP registry would mark every claimed file trained without training. Three are
MEDIUM: the standalone pre-check reads the root pending list instead of the `<kind>` sub-pool, so
`train` reports "No pending data" (TD-413, verified); the first-run vocab bootstrap is unreachable, so
`train`/`status`/`history` abort without a vocab (TD-414, verified); the LeJEPA pass ignores segment
ranges and strands claims after a crash (TD-415). One is LOW: command hygiene (TD-416).

Also October 13, 2026: filed [TD-417](#td-417---integrated-inference-answers-differ-from-normal-chat) through [TD-420](#td-420-integratedinferenceengine-claims-dead-config-and-fictional-stats) from the code-traced
[IntegratedInferenceEngine.md](../reference/source/IntegratedInferenceEngine.md), run against a small
real model. Three are MEDIUM: `--integrated-inference` ignores the request strategy (always greedy),
adds special tokens to encoder input and skips the generator's masking/repetition handling, so answers
differ from normal `/chat` (TD-417, verified: 3/3 prompts differ, `top_k` returns greedy output); an
exception in the encoder/decoder worker threads terminates `chatbot_api_server`, and shutdown drops
queued requests with broken promises (TD-418, shutdown verified); and the engine drives the model
without `model_mutex_` while `/chat/batch` uses it under the mutex (TD-419). One is LOW: unbatched
despite "50–100×"/"Production Ready" claims, dead config and fictional stats (TD-420).

Also October 13, 2026: filed [TD-421](#td-421-kvcacheappend-has-no-shape-checks) and [TD-422](#td-422-kvcache-copies-the-whole-cache-on-every-append) from the code-traced
[KVCache.md](../reference/source/KVCache.md), which merges and replaces the old `kvcache.md`. One is
MEDIUM: every `append()` reallocates and copies the whole cache, so CPU cached generation spends
O(n²·d) on copying (255 ms per cache for 512 steps at d_model 512; about 6 s for a 24-layer decoder)
(TD-422, verified). One is LOW: `append()` has no shape checks, so mismatched input overflows the heap
(ASan-verified write and read), and layer indices are unchecked (TD-421).

**October 14, 2026:** Filed [TD-423](#td-423-a-truncated-lm_head-file-loads-as-all-nan-logits) through [TD-425](#td-425-languagemodelhead-hygiene) from the code-traced
[LanguageModelHead.md](../reference/source/LanguageModelHead.md). Two are MEDIUM and verified: a
truncated `.lm_head` checkpoint sidecar loads without error with `NaN` weights, so every logit is `NaN`
(TD-423, the TD-298 defect in a file every checkpoint has); and uncached generation paths (beam,
`ChatbotAPI`'s CPU `model_fn`, `/chat/batch`) project every position through the LM head each step
although only the last row is used, about 30× the work at 100 tokens (TD-424). One is LOW: hygiene
(TD-425).

Also October 14, 2026: filed [TD-426](#td-426-layernorm-silent-setters-missing-shape-checks-and-unchecked-loads) (LOW) from the code-traced
[LayerNorm.md](../reference/source/LayerNorm.md), which verified the CPU backward pass against finite
differences (4.6×10⁻⁵): `set_gamma()`/`set_beta()` silently ignore shape mismatches (verified), shapes
aren't checked in `forward()`/`backward()`, and `load_weights()` doesn't check its reads.

Also October 14, 2026: filed [TD-427](#td-427-lejepa-predictor-is-never-saved-so-each-pass-restarts-it) through [TD-429](#td-429-lejepaencoder-exception-escape-console-output-and-unchecked-loads) from the code-traced
[LeJEPAEncoder.md](../reference/source/LeJEPAEncoder.md). One is MEDIUM and verified: `save()`/`load()`
skip the `Predictor`, so every resumed `--objective=lejepa` pass starts with a random predictor (loss
0.0017 trained → 2.05 after reload) whose gradients disturb the resumed encoder (TD-427). Two are LOW:
`train_step()`'s two sequential updates rest on a TD-189 rationale that TD-195 made obsolete (TD-428);
an escaping exception type, console output and unchecked loads (TD-429).

Also October 14, 2026: filed [TD-430](#td-430-logger-lazy-initialization-races-and-can-throw-from-a-log-call) and [TD-431](#td-431-trainer-processes-share-one-rotating-log-file-and-ignore-log_level) from the code-traced
[Logger.md](../reference/source/Logger.md). One is MEDIUM and verified: with `LOG_FILE_PATH` empty,
`trainer_service` and every `incremental_trainer` pass it launches fall back to the same
`chatbot_server.log` and rotate it from two processes, retaining about 27% less history than configured,
and the trainer binaries ignore `LOG_LEVEL` (TD-431). One is LOW and verified: unsynchronized lazy
initialization makes concurrent first-use log calls throw `spdlog_ex` (TD-430).

Also October 14, 2026: filed [TD-432](#td-432-loraadapterload-silently-adopts-the-files-shape) and [TD-433](#td-433-lora-initialization-update-and-config-gaps) (both LOW) from the code-traced
[LoRA.md](../reference/source/LoRA.md), which verified the adapter's gradients (1.7×10⁻⁴ vs. finite
differences). `LoRAAdapter::load()` silently adopts the file's dimensions, rank and α (TD-432, verified);
initialization uses the unseeded global `rand()`, `update()` doesn't zero gradients, and `dropout`/
`apply_to_ffn` are unused (TD-433). `src/LoRA.hpp` was re-tagged `@adai-status: experimental`: nothing
outside tests calls `enable_lora()`, and adapters aren't persisted with checkpoints (TD-302).

Also October 14, 2026: filed [TD-434](#td-434-matrixoperator-runs-short-products-single-threaded-and-portable-builds-have-no-simd) and [TD-435](#td-435-matrix-accessor-cost-sum-accuracy-randomize-race-and-api-hygiene) from the code-traced
[Matrix.md](../reference/source/Matrix.md), which verified every CPU `operator*` path correct (≤ 1.1×10⁻⁵
relative error). One is MEDIUM and verified by timing: `operator*` only parallelizes when `rows > 64`, so
the ≤ 64-row products that training samples and decode steps produce run on one thread (2–6× slower than a
work-based gate on this 8-core machine), and `portable` builds have no SIMD multiply at all because AVX2 is
selected only at compile time (7–10× slower) (TD-434). One is LOW: the out-of-line, bounds-checked
`operator()` costs 2.4× in hot loops (verified), `sum()` accumulates in `float` (2.2% error at 16.8M
elements, verified), `randomize()`'s static generator isn't thread-safe, and API hygiene (TD-435).

Also October 14, 2026: filed [TD-436](#td-436-sycl-icpx-builds-get-no--march-or-simd-flags) and [TD-437](#td-437-simd-self-check-test-never-compiles-and-no-startup-cpu-check) from the code-traced
[MatrixSIMD.md](../reference/source/MatrixSIMD.md). One is MEDIUM and verified: both CMake checks that add
`-march=native` and `-mavx2 -mfma` match only `"GNU|Clang"`, and oneAPI's `icpx` reports `IntelLLVM`, so the
`sycl` preset (ai-machine's trainer) compiles `Matrix.cpp` with no SIMD at all, a 6–8× slower CPU multiply
(TD-436). One is LOW: the only compile-time vs. runtime SIMD test never compiles (verified), nothing checks
the CPU at startup, and `has_avx2()`/`has_fma()` skip the OS-support check (TD-437).

Also October 14, 2026: filed [TD-438](#td-438-metrics_api_server-accepts-invalid-numbers-and-exits-0-on-bad-arguments) and [TD-439](#td-439-metrics_api_server-backend-name-and-admin-store-location-gaps) from the code-traced
[MetricsApiServerArgs.md](../reference/source/MetricsApiServerArgs.md). One is MEDIUM and verified with the
real server: numeric flags go through `atoi`/`strtoul` unchecked, so `--port abc` listens on a random
ephemeral port while printing `:0`, `--port 99999` listens on 34463, `--max-live-sessions -1` removes the cap
and `--sweep-interval-seconds 0` disables eviction; saved admin overrides have the same gaps; and a bad
argument exits 0, so systemd doesn't treat it as a failure (TD-438). One is LOW: an unknown
`--storage-backend` is accepted (file-only fallback), and the admin store is located from the file's
`db_path` rather than `--db-path` (TD-439).

Also October 14, 2026: filed [TD-440](#td-440-metrics-storage-has-no-retention-and-max_records_on_disk-is-never-applied) and [TD-441](#td-441-imetricsdatabase-contract-gaps-step-counter-limit-order-unbounded-reads) from the code-traced
[MetricsDatabase.md](../reference/source/MetricsDatabase.md). One is MEDIUM and verified by search: nothing
deletes metrics rows or rotates the JSONL file, and `max_records_on_disk` (a CLI flag, config key and
admin-mutable setting) is read by no persistence code, so `metrics_api_server`'s storage grows without
bound; the per-step `gradient_variance_history` and the `generation_quality` table are also written but
never read (TD-440). One is LOW: the gradient-variance step counter restarts per service instance,
`query_history()`'s `limit` keeps the oldest rows, compare/export read whole histories, and the interface
contract is undocumented (TD-441).

Also October 14, 2026: filed [TD-442](#td-442-metricspushclientend_session-drain-can-delay-a-passs-final-save-by-an-hour) through [TD-444](#td-444-metricspushclient-transport-gaps-no-https-a-connection-per-event-silent-failures) from the code-traced
[MetricsPushClient.md](../reference/source/MetricsPushClient.md). Two are MEDIUM and verified: `end_session()`
drains every queued event at about 1.2 s per failed event (13.2 s for 11 events with the server down, so
roughly 20–70 minutes for a full 1,024-event queue) and `IncrementalTrainer` calls it before the final
checkpoint and `mark_trained()`, so a metrics outage delays a pass's results and a stop in that window loses
them (TD-442); and push payloads send NaN/inf as `nan`/`inf`, which the server stores and re-emits as invalid
JSON, `end_epoch()`'s `std::fixed` turns values below 5e-7 (a 1e-7 learning rate) into 0, and adaptive-clipping
metrics are buffered but never sent (TD-443). One is LOW: `https://` URLs send plain HTTP, a new connection per
event, silent non-404 failures, and a small race (TD-444).

Also October 14, 2026: filed [TD-445](#td-445-finished-metrics-sessions-hold-max_live_sessions-slots-for-an-hour) and [TD-446](#td-446-metricssessionregistry-sweep-interval-changes-miss-logging-and-lock-held-io) from the code-traced
[MetricsSessionRegistry.md](../reference/source/MetricsSessionRegistry.md). One is MEDIUM and verified:
`max_live_sessions` counts finished and stale sessions until their TTL (3,600 s as shipped) and nothing evicts
one to make room, so with the cap of 16, more than 16 trainer passes an hour (each registers a new key) get
`503 metrics_server_full` and train without metrics (TD-445). One is LOW: a runtime sweep interval ≤ 0 spins the
sweep thread (0.51 CPU-s per 2 s, verified) and 0 → positive never starts it (verified), unknown-key lookups log
one line per live session, and archiving I/O runs under the registry lock (TD-446).

Also October 14, 2026: filed [TD-447](#td-447-metricstracker-printer-state-leak-convergence-off-by-one-and-smoothing-gaps) (LOW) from the code-traced
[MetricsTracker.md](../reference/source/MetricsTracker.md). Production only uses its CSV export and final-epoch
overfitting warning; the verified gaps are in the rest: `print_summary()`/`print_history()` leave `std::cout` in
fixed 4-decimal mode, `is_converging()` needs `window + 1` epochs, a zero smoothing window gives `NaN`, and the
smoothed validation curve skips epochs without validation.

Also October 14, 2026: filed [TD-448](#td-448-mns_cli-ignores-unknown-flags-misreads-set-training-arguments-and-aborts-on-bad-numbers) and [TD-449](#td-449-mns_cli-request-building-gaps-layer-counts-relink-resets-encoding) from the code-traced
[MnsCliCommands.md](../reference/source/MnsCliCommands.md). One is MEDIUM and verified: every `mns_cli` command
silently ignores unknown flags, so `register m chat --dmodel 256` permanently registers the config default
architecture (immutable after registration); `set-training m --new_run` continues the current run with metrics key
`--new_run`; and a malformed number aborts the process with a core dump while `-1` wraps to 2⁶⁴−1 (TD-448). One is
LOW: a standalone kind without `--num-layers` sends both layer counts, relinking a world model resets the hippocampal
settings, and names/queries aren't URL-encoded (TD-449).

Also October 14, 2026: filed [TD-450](#td-450-mns_cli-register-takes-architecture-defaults-from-configmnsconf-not-the-trainers-config) and [TD-451](#td-451-mns_cli-has-no-https-and-minor-option-handling-gaps) from the code-traced
[MnsCliTool.md](../reference/source/MnsCliTool.md). One is MEDIUM: `mns_cli` loads `config.mns.conf` (the daemon's
file, `D_MODEL=512`/`NUM_HEADS=8`), so a flagless `register` permanently sets that architecture rather than
`config.trainer.conf`'s 768/12, and mns-cli-guide.md still says the defaults come from `config.conf` and match the
trainer (TD-450). One is LOW: `https://` URLs connect with plain HTTP to port 8083 (verified), `--help` works only
as the first word (verified), and minor hygiene (TD-451).

Also October 14, 2026: filed [TD-452](#td-452-mnsjsonhelpers-parsing-gaps-and-a-gui-register-body-without-td-196-fields) and [TD-453](#td-453-hand-written-json-duplicated-across-mns-and-metrics-code) (both LOW) from the code-traced
[MnsJsonHelpers.md](../reference/source/MnsJsonHelpers.md). The MNS Manager GUI's helpers don't unescape string values,
match only compact JSON, fall back to any array, and escape only `"`, `\` and `\n` (verified); its register body has no
`kind`/`run_group`/`connection`, so the GUI can't manage TD-196 records (TD-452). The same hand-written JSON escaping and
scanning is duplicated across six MNS/metrics files with different coverage while `nlohmann/json` is vendored (TD-453).

Also October 14, 2026: filed [TD-454](#td-454-mns-manager-guis-training--candidate-workflow-fails-against-server-allocated-run-ids) and [TD-455](#td-455-mns-manager-gui-error-detection-ui-thread-http-and-register-defaults) from the code-traced
[MnsManagerGUI.md](../reference/source/MnsManagerGUI.md). One is MEDIUM and verified against a real `mns_server` with the
GUI's exact request bodies: Set Training sends a typed `run_id` the server ignores (it allocates `run-01`) and never sends
`new_run`, so Set Candidate with the typed ID gets `409 … superseded` and the GUI can't start a new run; the GUI guide
documents this failing flow (TD-454). One is LOW: success is judged by the substring `"error"` rather than the HTTP status,
all HTTP runs on the UI thread, and the Register tab's defaults (128/4/512, layer minimum 1) match neither config (TD-455).

Also October 14, 2026: filed [TD-456](#td-456-mns_server-treats-a-mistyped---admin-enabled-as-true-and-doesnt-range-check---port) (MEDIUM, verified with the real `mns_server`) from the code-traced
[MnsServerArgs.md](../reference/source/MnsServerArgs.md): an unrecognized `--admin-enabled` value falls back to `true`, so
`--admin-enabled flase` left `PUT /admin/config` open (an unauthenticated request repointed `registry_group`); and `--port`
isn't range-checked (`99999` listened on 34463, `8083x` → 8083, `0` → random).

Also October 14, 2026: filed [TD-457](#td-457-modelnameclient-retries-non-idempotent-requests-after-a-timeout) through [TD-459](#td-459-modelnameclient-validation-unescaping-and-slow-failure-gaps) from the code-traced
[ModelNameClient.md](../reference/source/ModelNameClient.md). Two are MEDIUM and verified: timeouts and 5xx are retried for
non-idempotent requests, so one `set_training(new_run = true)` reached a slow mock server twice (on `mns_server`, a `retrain`
then skips a run number) and `register_model()` would throw after succeeding (TD-457); and a NaN epoch loss pushed by
`push_progress()` is stored and re-emitted as `nan` by `mns_server`, making both the model's record and the whole
`GET /models` list invalid JSON, and later archiving it into `training_history` (TD-458). One is LOW: unvalidated
architecture fields, no unescaping, ~16 s per call when MNS doesn't answer, no URL encoding (TD-459).

Also October 14, 2026: filed [TD-460](#td-460-mns-retirement-leaves-role-mappings-and-is-undone-by-running-trainers) through [TD-462](#td-462-modelnameservice-dead-datasets-proxy-admin-settings-race-and-swallowed-persistence-errors) from the code-traced
[ModelNameService.md](../reference/source/ModelNameService.md) (its 62 live tests all pass against a local `mns_server`). Two
are MEDIUM and verified against a real server: retiring a role's production model leaves `roles_` pointing at it (so
`MODEL_ROLE` keeps resolving a retired model), and a model retired mid-pass is revived by the trainer's normal `set_candidate()`
because `retired → candidate` skips the run ownership check (TD-460); a legacy chatbot registers with no architecture (all
zeros) and no kind checks `d_model % num_heads`, permanently registering unbuildable models (TD-461). One is LOW: a dead
datasets proxy (no `model_id` is ever recorded), an unlocked admin-settings data race, swallowed persistence errors, compact-only
JSON parsing, a stale role after promoting to a new role (TD-462).

Also October 14, 2026: filed [TD-463](#td-463-mns_server-runs-in-memory-and-reports-healthy-when-modelsdb-cant-be-opened) and [TD-464](#td-464-mns_server-startup-crash-early-sigterm-hang-and-fixed-log-level) from the code-traced
[ModelNameServiceServer.md](../reference/source/ModelNameServiceServer.md). One is MEDIUM and verified: when `models.db` can't be
opened, `mns_server` logs one error and runs in memory, reporting `/health` ok and accepting registrations that vanish on restart
(TD-463). One is LOW and verified: an unusable data directory aborts with a core dump (uncaught exception on the server thread),
SIGTERM before `listen()` hung 1 of 30 early-stop trials, and `LOG_LEVEL` is ignored (TD-464).

**October 10, 2026:** Filed [TD-297](#td-297-layer-update_weights-steps-a-shared-optimizer-once-per-layer) through [TD-303](#td-303-decoder-only-mode-still-runs-cross-attention-over-a-dummy-input) from the code-traced
[CrossAttention.md](../reference/source/CrossAttention.md). Two are MEDIUM: layer `update_weights()`
calls a shared optimizer's `step()`, which steps every registered layer, so the public
`EncoderDecoderModel::train_step()` path over-steps every parameter 6–10 times per update (TD-297,
verified; the shipped trainers call `step()` once and are unaffected); and `CrossAttention::load()`
loads truncated files as NaN weights without error (TD-298, verified). Five are LOW (TD-299–303):
SIGFPE on `num_heads == 0`, stale cached encoder K/V, GPU/CPU divergence, LoRA persistence, and
decoder-only mode running cross-attention over a dummy input.

**October 9, 2026:** Filed [TD-260](#td-260-chatbot_gui-never-loads-a-trained-checkpoint) through [TD-266](#td-266-chatbot_gui-minor-gaps) from the code-traced
[ChatbotGUI.md](../reference/source/ChatbotGUI.md) (`chatbot_gui`). Two are **HIGH**: the GUI
never loads a trained checkpoint, because it checks for a file at the bare model path, which
`save_model()` never writes, and so silently runs on random weights (TD-260); and
`EncoderDecoderModel::generate_response_with_strategy()` forces beam search whenever
`num_beams > 1`, overriding the chosen strategy in the GUI (default beam width 5) and in
`RLHFTrainer` (header default 4) (TD-261). Three are MEDIUM (architecture from local config only
and an unusable state on init failure, UI freeze during generation, launcher hardcoding Qt5/x86_64
paths) and two LOW (prompt-format mismatch with the server, minor gaps).

Also October 9, 2026: filed [TD-267](#td-267-chatbottrainer-keeps-skipped-pairs-biasing-validation-loss-low) through [TD-276](#td-276-chatbottrainer-hygiene-dead-fields-logging-stale-test-file) from the code-traced
[ChatbotTrainer.md](../reference/source/ChatbotTrainer.md). One is **HIGH**: pairs that
`preprocess_data()` skips (empty input/response, invalid UTF-8) stay as empty entries, so in
training each throws and discards its accumulation window, and in validation each counts as loss 0,
biasing validation loss low. A real run dropped it from about 3.4 to 2.5 with one empty pair
(TD-267). Five are MEDIUM: `end_epoch()` only reported with validation data (TD-268); optimizer state
and LR schedule restart every incremental pass (TD-269); adaptive clipping resets every epoch and can
never activate (TD-270); a cache hit can pair with a different random split (TD-271); mid-window
errors drop gradients and the last window is under-weighted (TD-272). Four are LOW (TD-273–276).

Also October 9, 2026: filed [TD-277](#td-277-childprocess-reports-every-signal-death-as-exit-code-128) through [TD-282](#td-282-childprocess-windows-path-has-known-gaps) from the code-traced
[ChildProcess.md](../reference/source/ChildProcess.md) (`trainer_service`'s child launcher),
with exit-code behaviour confirmed by running the real class. Three are MEDIUM: every signal death
is reported as 128, losing the signal number, so a GPU-driver SIGSEGV can't be told apart from an OOM
SIGKILL (TD-277); a failed `exec` (missing binary) looks like a successful launch and leads to a
silent retry loop (TD-278); and `request_stop()` could send `kill(-1, SIGTERM)` under the concurrent use
its header permits (latent; TD-279). Three are LOW (orphaned passes, fork/fd hygiene, Windows gaps).

Also October 9, 2026: filed [TD-283](#td-283-sighup-terminates-every-binary-except-chatbot_api_server) through [TD-290](#td-290-configloader-output-hygiene) from the code-traced
[Config.md](../reference/source/Config.md) (`ServiceConfig`/`ConfigLoader`), with parsing and reload
behaviour confirmed by running the real loader. Two are **HIGH**: `SIGHUP` terminates every binary
except `chatbot_api_server`, although CLAUDE.md said `incremental_trainer` hot-reloads on it (TD-283);
and `registry_server` ignores the FTP/FTPS security keys its own config file ships with, so
`FTPS_ENABLED=true` or a real secret in the file silently yields plaintext FTP and random passwords
(TD-284). Four are MEDIUM: `reload()` reports success without applying untracked changes (TD-285);
two boolean parsers (TD-286); negative/partial numbers accepted (TD-287); `validate()` gaps (TD-288).
Two are LOW (TD-289–290).

Also October 9, 2026: filed [TD-291](#td-291-conversation-imports-can-lift-session-limits-and-budgets) through [TD-296](#td-296-conversationcontext-hygiene) from the code-traced
[ConversationContext.md](../reference/source/ConversationContext.md), each confirmed by running the
real class. Two are **HIGH**: conversation imports replace the session's limits and trust stored token
counts, so `POST /chat/session/import` can make a session unbounded (TD-291); and a single message
over 1.2× the token budget evicts the entire conversation including itself, producing an HTTP 500
and lost history (TD-292). Two are MEDIUM: a malformed import wipes the live session (TD-293), and
the token budget is unrelated to the model's sequence length (TD-294). Two are LOW (TD-295–296).

**October 8, 2026:** Filed
[TD-211](#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model)
through
[TD-214](#td-214-batched-inference-mode-generates-differently-from-the-inline-path), all found while
writing the code-traced reference
[BatchedInferenceEngine.md](../reference/source/BatchedInferenceEngine.md). The headline (TD-211):
the `stable`-tagged `BatchedInferenceEngine` is a single-worker request queue that generates each
request sequentially. Despite its comments and the `--batched-inference` flag's name, it never
batches the model, and can't until TD-171. TD-212 covers accepted requests that can still fail or be
abandoned (one throwing request fails its whole group; shutdown drains only one group; `submit()`
races shutdown). TD-213 covers misleading, unexposed stats. TD-214 covers batched mode ignoring
`strategy`/`beam_width` and sharing one fixed-seed RNG across requests. Each item is tagged in code
with `TODO: See TD-NNN`.

Same day, filed
[TD-215](#td-215-activation-has-misleading-header-comments-and-unchecked-inputs) and
[TD-216](#td-216-softmax-and-gelu-math-is-duplicated-across-cpu-and-gpu-code) from the code-traced
[Activation.md](../reference/source/Activation.md): two wrong header comments plus two unchecked
inputs in the `stable` `Activation` class (TD-215), and softmax/GELU math re-typed across CPU hot
paths and both GPU backends, risking silent CPU/GPU drift (TD-216).

Also the same day, filed
[TD-217](#td-217-chatbatch-reports-hypothetical-batching-stats-as-if-they-were-real) through
[TD-219](../archive/TECHNICAL_DEBT_RESOLVED.md#td-219-datasets-batch-wrappers-have-wrong-doc-examples-and-a-partial-stats-helper) from
the code-traced [BatchProcessor.md](../reference/source/BatchProcessor.md): `/chat/batch`'s public
`stats` describe grouping that never happens (TD-217); `BatchProcessor.hpp`'s helpers have
unused surface, lose input order, and skip input checks (TD-218); and `Dataset`'s batch wrappers
carry wrong doc examples and a stats helper that only samples one batch (TD-219). Decisions the same day: TD-217 relabels the stats as an estimate, and TD-211/TD-218 keep
their classes, with real batching planned as a v2 upgrade in
[real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md).

Also the same day, filed
[TD-220](#td-220-bpe-merge-learning-is-corrupted-by-the--pair-key-separator) through
[TD-225](#td-225-bpetokenizer-logging-and-api-hygiene) from the code-traced
[BPETokenizer.md](../reference/source/BPETokenizer.md), each bug confirmed by running the real
tokenizer. Two are **HIGH**: BPE merge learning is corrupted by its `"|||"` pair-key separator
(pipe-heavy corpora stop learning merges or repeat a wrong rule; TD-220), and `save_vocab()` fails
silently, so checkpoints can be written without their vocab (TD-221). Two are MEDIUM: a
never-invalidated per-thread BPE cache (TD-222), and mutators that can leave the tokenizer gutted
or with duplicate IDs (TD-223). Two are LOW: lossy lowercase/whitespace normalization, pending an
owner decision (TD-224), and logging/API hygiene (TD-225).

Also the same day, filed [TD-226](#td-226-chatbatch-rejects-any-request-whose-messages-contain-) through [TD-239](#td-239-chatbotapi-logging-and-dead-code-cleanup) from the code-traced
[ChatbotAPI.md](../reference/source/ChatbotAPI.md), one item per bug. The JSON bugs were confirmed
by running the real helpers, and are all in `ChatbotAPI`'s own hand-written parser, not the
vendored nlohmann/json, which handles the same inputs correctly. Two are **HIGH**: `/chat/batch`
rejects any request whose messages contain `]` (TD-226), and session pointers are used outside the
session lock, allowing same-session races and a possible use-after-free (TD-227). Eight are MEDIUM
(JSON string parsing, invalid JSON output, RAG serializing all requests, session expiry only on
`/health`, unlimited client-chosen sessions, non-cryptographic session IDs, batch endpoints
bypassing the dispatcher, no batch size limit) and four are LOW.

Also the same day, filed [TD-240](#td-240-chatbot_api_server-silently-accepts-non-numeric-and-partial-numbers) through [TD-245](#td-245-chatbot_api_server-argument-errors-are-misleading-or-silent) from the code-traced
[ChatbotApiServerArgs.md](../reference/source/ChatbotApiServerArgs.md) (`chatbot_api_server`'s
command-line layer), with the parser's behaviour confirmed by running it. Four are MEDIUM:
non-numeric and partial numbers silently become 0 or truncated values (`--port abc` binds port 0;
TD-240); the full config validator never runs at startup, only on `SIGHUP` reload (TD-241);
reload discards CLI generation overrides (TD-242); and `TOP_K`/`BEAM_WIDTH` are parsed but never
applied (TD-244). Two are LOW: every engine-mode flag passed gets started (TD-243), and misleading
or silent argument errors (TD-245).

Also the same day, filed [TD-246](#td-246-chatbot_api_server-serves-a-randomly-initialized-model-when-loading-fails) through [TD-252](#td-252-chatbot_api_server-startup-log-and-usage-text-are-inaccurate) from the code-traced
[ChatbotAPIServer.md](../reference/source/ChatbotAPIServer.md) (`chatbot_api_server`'s `main()`).
Two are **HIGH**: a failed or missing model load silently serves randomly initialized weights while
`/health` reports `ok` (TD-246), and RAG retrieval embeds documents and queries with an untrained
encoder because its weights are never loaded (TD-247). Three are MEDIUM: a shutdown arriving during
server startup is lost and the process hangs (confirmed against httplib; TD-248), the `SIGHUP`
handler isn't async-signal-safe (TD-249), and hippocampal memory is saved only on graceful shutdown,
with paths that can land in the working directory (TD-250). Two are LOW: reload silently ignores
most settings (TD-251), and the startup log/usage text are inaccurate (TD-252).

Also the same day, filed [TD-253](#td-253-chatbot-spins-forever-at-end-of-input) through [TD-259](#td-259-chatbot-code-hygiene-unchecked-set-values-global-helpers-heavy-header) from the code-traced
[ChatbotCLI.md](../reference/source/ChatbotCLI.md) (the `chatbot` terminal client), confirmed by
running the real binary against a stub server. One is **HIGH**: at end of input (Ctrl+D or piped
stdin) the REPL loops forever, emitting 284 MB of prompts in 3 seconds and never auto-saving
(TD-253). Two are MEDIUM: `/set` settings are sent with each request but ignored by the server
(TD-254), and `success:false` replies are shown as empty responses (TD-255). Four are LOW: help
banner gaps (TD-256), unescaped `session_id` (TD-257), conversation save loss/overwrite
(TD-258), and code hygiene (TD-259).

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

**Tier 11 — Newly filed (October 8, 2026): `BatchedInferenceEngine` honesty and robustness.**
[TD-212](#td-212-batchedinferenceengine-can-fail-or-abandon-requests-it-already-accepted) (4-6h)
is the only one that's a plain bug fix with no decision attached, so it can go first.
[TD-211](#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model)
had its owner decision on October 8, 2026: keep the class and deliver real batching as a v2
([real_batching_v2_plan.md](../../proposals/real_batching_v2_plan.md), built on
[TD-171](#td-171-no-batch-dimension-anywhere-in-the-model-stack--real-parallel-batched-training-not-supported)'s
inference half); only its v1.x honesty pass is open here.
[TD-213](#td-213-batchedinferenceengine-stats-are-misleading-and-unused-in-production)'s latency
fix can land in v1.x; whether to expose the stats is best decided alongside v2's phase B4, which
replaces how they're produced.
[TD-214](#td-214-batched-inference-mode-generates-differently-from-the-inline-path) is independent
and small.

**Tier 12 — Newly filed (October 8, 2026): `Activation` cleanup.**
[TD-215](#td-215-activation-has-misleading-header-comments-and-unchecked-inputs) (1-2h) is a
self-contained fix to one file and its tests.
[TD-216](#td-216-softmax-and-gelu-math-is-duplicated-across-cpu-and-gpu-code) (4-6h) touches attention,
generation, training and both GPU backends; the GPU half can only be compiled and checked on a host
with the CUDA or SYCL toolchain, so it's best done alongside other GPU work.

**Tier 13 — Newly filed (October 8, 2026): batch-padding helpers.**
[TD-217](#td-217-chatbatch-reports-hypothetical-batching-stats-as-if-they-were-real) (1-2h) is the
only one visible to API clients, so do it first. The decision is made (relabel as an estimate), but
it still changes the public `/chat/batch` response, so it needs a changelog note.
[TD-218](#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs) was
decided the same day: keep the helpers as groundwork for the same v2 plan, so only its v1.x fixes
are open here.
[TD-219](../archive/TECHNICAL_DEBT_RESOLVED.md#td-219-datasets-batch-wrappers-have-wrong-doc-examples-and-a-partial-stats-helper) was closed
October 11, 2026, when `Dataset.hpp` was retired (TD-315).

**Tier 14 — Newly filed (October 8, 2026): tokenizer bugs.**
[TD-220](#td-220-bpe-merge-learning-is-corrupted-by-the--pair-key-separator) and
[TD-221](#td-221-bpetokenizersave_vocab-fails-silently-so-checkpoints-can-lack-their-vocab) are
HIGH, small (1-2h each) and independent: do them first. TD-220 also needs an audit of existing
vocab files, since any vocab it truncated is baked into checkpoints trained on it.
[TD-222](#td-222-apply_bpes-thread-local-cache-is-never-invalidated) and
[TD-223](#td-223-tokenizer-mutators-can-leave-it-gutted-or-with-duplicate-ids) touch the same
mutators (`load_vocab()`/`build_vocab()`), so do them together.
[TD-225](#td-225-bpetokenizer-logging-and-api-hygiene) is independent cleanup.
[TD-224](#td-224-tokenizer-normalization-is-lossy-no-capitals-newlines-or-tabs) needs an owner
decision and, if pursued, a retrain, so it's the largest by far despite its LOW label.

**Tier 15 — Newly filed (October 8, 2026): `ChatbotAPI` (`chatbot_api_server`'s HTTP layer).**
Do [TD-226](#td-226-chatbatch-rejects-any-request-whose-messages-contain-) first. It's HIGH, user-visible, and replacing the hand-written parser with the
already-vendored nlohmann/json also resolves [TD-228](#td-228-chatbotapiparse_json_string-mis-parses-common-json) (and [TD-229](#td-229-chatbotapi-responses-can-be-invalid-json-unescaped-control-characters) if responses are built with it
too). [TD-227](#td-227-chatbotapi-session-pointers-are-used-outside-the-session-lock) (HIGH) is the other priority, and fits naturally with the other session items
[TD-231](#td-231-chatbotapi-sessions-only-expire-when-get-health-is-called), [TD-232](#td-232-chatbotapi-accepts-unlimited-client-chosen-session-ids) and [TD-233](#td-233-chatbotapi-session-ids-come-from-a-non-cryptographic-generator). [TD-230](#td-230-rag-mode-serializes-every-chatbotapi-request-behind-config_mutex_) and [TD-237](#td-237-chatbotapirunning_-is-a-non-atomic-flag-written-from-two-threads) are one-line-scale fixes.
[TD-234](#td-234-chatbotapi-batch-endpoints-bypass-the-inference-dispatcher) and [TD-235](#td-235-chatbotapi-batch-endpoints-have-no-size-limit) both touch the batch endpoints, so do them together, alongside TD-217's
relabel. [TD-236](#td-236-chatbotapi-returns-http-200-for-validation-errors), [TD-238](#td-238-chatbotapi-returns-internal-exception-text-to-clients) and [TD-239](#td-239-chatbotapi-logging-and-dead-code-cleanup) are cleanup; TD-236 changes status codes clients may
depend on, so check the CLI and Android clients first.

**Tier 16 — Newly filed (October 8, 2026): `chatbot_api_server` startup and config.**
[TD-240](#td-240-chatbot_api_server-silently-accepts-non-numeric-and-partial-numbers) and [TD-241](#td-241-chatbot_api_server-never-validates-its-configuration-at-startup) are small and together close the "invalid settings accepted silently" gap:
strict number parsing, then the full validator at startup (extended with the generation fields).
[TD-244](#td-244-chatbot_api_server-ignores-the-top_k-and-beam_width-settings) is a two-line wiring fix plus two flags. [TD-242](#td-242-sighup-reload-discards-chatbot_api_servers-cli-generation-overrides) needs a small design choice (re-apply
argv after reload vs. tracking CLI-owned keys), so do it after the others. [TD-243](#td-243-chatbot_api_server-starts-every-engine-mode-flag-passed-not-just-one) and
[TD-245](#td-245-chatbot_api_server-argument-errors-are-misleading-or-silent) are parser cleanup that fits in the same change as TD-240.

**Tier 17 — Newly filed (October 8, 2026): `chatbot_api_server`'s `main()`.**
[TD-246](#td-246-chatbot_api_server-serves-a-randomly-initialized-model-when-loading-fails) and [TD-247](#td-247-rag-retrieval-in-chatbot_api_server-uses-an-untrained-encoder) are HIGH and independent: each makes the service silently return worse
answers than the operator configured, so do them first. TD-247 should refuse to enable RAG without
a loaded model, which depends on TD-246's "model loaded" state. [TD-248](#td-248-chatbot_api_server-can-hang-if-shutdown-arrives-during-server-startup) and [TD-249](#td-249-chatbot_api_servers-sighup-handler-isnt-async-signal-safe) are small
signal/shutdown fixes to do together. [TD-250](#td-250-hippocampal-memory-in-chatbot_api_server-is-lost-on-crash-and-can-land-in-the-working-directory) matters only once the experimental hippocampal
memory is used in production. [TD-251](#td-251-chatbot_api_server-reload-silently-ignores-most-settings) and [TD-252](#td-252-chatbot_api_server-startup-log-and-usage-text-are-inaccurate) are operator-facing cleanup and pair
naturally with Tier 16's reload work (TD-242).

**Tier 18 — Newly filed (October 8, 2026): `chatbot` CLI client.** [TD-253](#td-253-chatbot-spins-forever-at-end-of-input) is HIGH and a
one-line fix; do it first. [TD-254](#td-254-chatbots-set-generation-settings-have-no-effect) needs an owner decision (per-request settings in
`ChatbotAPI`, or remove them from the CLI). Supporting them is easiest after TD-226's
nlohmann/json switch, and it should be decided together with TD-239's misleading
"config from request" comment. [TD-255](#td-255-chatbot-shows-server-error-replies-as-empty-responses) and [TD-257](#td-257-chatbot-sends-session_id-unescaped-in-two-requests) touch the same request/response code and can
go together. [TD-256](#td-256-chatbots-help-banner-omits-save-load-and-stats), [TD-258](#td-258-chatbot-can-lose-or-overwrite-saved-conversations) and [TD-259](#td-259-chatbot-code-hygiene-unchecked-set-values-global-helpers-heavy-header) are independent cleanup.

**Tier 19 — Newly filed (October 9, 2026): `chatbot_gui` and strategy dispatch.** [TD-260](#td-260-chatbot_gui-never-loads-a-trained-checkpoint) is HIGH
and about an hour: until it's fixed the GUI can't be used with any trained model. [TD-261](#td-261-generate_response_with_strategy-forces-beam-search-whenever-num_beams--1) is HIGH
and lives in `EncoderDecoderModel`, not the GUI; it also changes RLHF rollouts, so re-check any
RLHF results produced before the fix. [TD-262](#td-262-chatbot_gui-ignores-mns-architecture-and-becomes-unusable-on-init-failure) (architecture/init) pairs with TD-260. [TD-263](#td-263-chatbot_gui-freezes-while-generating)
and [TD-264](#td-264-chatbot_gui-launcher-hardcodes-qt5-and-x86_64-paths) are independent. [TD-265](#td-265-chatbot_gui-and-chatbot_api_server-format-conversation-context-differently) needs a format decision shared with the server.
[TD-266](#td-266-chatbot_gui-minor-gaps) is cleanup.

**Tier 20 — Newly filed (October 9, 2026): `ChatbotTrainer`.** [TD-267](#td-267-chatbottrainer-keeps-skipped-pairs-biasing-validation-loss-low) is HIGH and small: fix
it first, and treat validation losses (and best-epoch choices) from runs on data containing
empty/invalid pairs as optimistic until then. [TD-268](#td-268-chatbottrainer-only-ends-metrics-epochs-when-there-is-validation-data), [TD-270](#td-270-adaptive-gradient-clipping-resets-every-epoch) and [TD-275](#td-275-chatbottrainer-drops-eos-from-long-training-responses) are small,
independent fixes. [TD-269](#td-269-optimizer-state-and-lr-schedule-restart-every-incremental-training-pass) needs an owner decision (run-wide vs. per-pass schedule) and checkpoint
format work, so scope it alongside the `IncrementalTrainer` reference. [TD-271](#td-271-chatbottrainers-tokenized-cache-can-pair-with-a-different-random-split) and [TD-272](#td-272-chatbottrainer-silently-drops-gradients-on-mid-window-errors-and-under-weights-the-last-window)
touch the data/accumulation paths and should get tests first. [TD-273](#td-273-chatbottrainer-quality-backfill-writes-results-nobody-can-read), [TD-274](#td-274-chatbottrainer-saves-the-best-model-to-a-fixed-file-in-the-working-directory) and [TD-276](#td-276-chatbottrainer-hygiene-dead-fields-logging-stale-test-file) are
cleanup.

**Tier 21 — Newly filed (October 9, 2026): `ChildProcess` (`trainer_service`).** [TD-277](#td-277-childprocess-reports-every-signal-death-as-exit-code-128) and
[TD-279](#td-279-childprocessrequest_stop-can-signal-every-process-if-called-concurrently) are about an hour each; do them together. [TD-278](#td-278-childprocess-reports-a-failed-exec-as-a-successful-launch) and [TD-281](#td-281-childprocess-forks-a-multithreaded-process-and-leaks-file-descriptors) share a natural fix
(`posix_spawn()`), so do them as one change. [TD-280](#td-280-trainer_service-passes-can-be-orphaned-if-the-supervisor-is-hard-killed) matters only outside systemd. [TD-282](#td-282-childprocess-windows-path-has-known-gaps) only
matters if a Windows `trainer_service` is ever shipped.

**Tier 22 — Newly filed (October 9, 2026): `Config`.** [TD-283](#td-283-sighup-terminates-every-binary-except-chatbot_api_server) and [TD-284](#td-284-registry_server-ignores-the-ftpftps-security-keys-its-config-file-ships-with) are HIGH and
operator-facing: one turns a documented reload command into a kill, the other silently ignores
security settings. Do them first, and fix the docs (both are annotated in CLAUDE.md /
`config.registry.conf` meanwhile). [TD-286](#td-286-configloader-parses-booleans-two-different-ways), [TD-287](#td-287-configloader-accepts-negative-and-partial-numbers) and [TD-289](#td-289-configloader-keeps-inline-comments-and-silently-accepts-enum-typos) are parser fixes that belong in one
change with shared helpers. [TD-285](#td-285-configloaderreload-reports-success-without-applying-untracked-changes) and [TD-288](#td-288-configloadervalidate-misses-fields-that-crash-or-misbehave) pair with Tier 16's TD-241/TD-242 (validation and
reload in `chatbot_api_server`). [TD-290](#td-290-configloader-output-hygiene) is cleanup.

**Tier 23 — Newly filed (October 9, 2026): `ConversationContext`.** [TD-291](#td-291-conversation-imports-can-lift-session-limits-and-budgets) and [TD-292](#td-292-one-long-message-wipes-the-whole-conversation) are HIGH
and both reachable from any `chatbot_api_server` client; do them first, together with [TD-293](#td-293-a-malformed-conversation-import-wipes-the-live-session) (all
three touch `deserialize()`/`truncate_to_limits()`). Pair them with Tier 15's session items (TD-227,
TD-231, TD-232). [TD-294](#td-294-conversation-token-budget-is-unrelated-to-the-models-sequence-length) needs a small `ChatbotAPI` setting and pairs with TD-266. [TD-295](#td-295-conversation-role-labels-are-spoofable-and-roles-arent-escaped) and
[TD-296](#td-296-conversationcontext-hygiene) are cleanup.

**Tier 24 — Newly filed (October 10, 2026): `CrossAttention`.** [TD-297](#td-297-layer-update_weights-steps-a-shared-optimizer-once-per-layer) is the most important:
fix it at the model level the way TD-178 fixed `LeJEPAEncoder`, since the same pattern spans every
layer type. [TD-298](#td-298-crossattentionload-loads-truncated-files-as-nan-weights) and [TD-299](#td-299-crossattention-crashes-with-sigfpe-when-num_heads-is-zero) are small input-hardening fixes. [TD-300](#td-300-crossattentionforward_with_cache-silently-reuses-stale-encoder-kv), [TD-301](#td-301-crossattention-gpu-path-diverges-from-the-cpu-path) and
[TD-302](#td-302-lora-adapters-arent-persisted-or-counted-in-gradient-norms) matter mainly before LoRA or the GPU cached path get wider use; [TD-303](#td-303-decoder-only-mode-still-runs-cross-attention-over-a-dummy-input) is cleanup.

**Tier 25 — Newly filed (October 11, 2026): `DaemonConfigStore` and admin config.** [TD-304](#td-304-session-counters-can-be-duplicated-and-daemonconfigstore-has-no-busy-timeout)
first: a busy timeout is a one-line change, and an atomic `increment()` fixes duplicate session
numbers in both transports. [TD-305](#td-305-a-rejected-or-failed-put-adminconfig-still-partly-applies) pairs with it (add `set_many()` in the same change) and with
[TD-306](#td-306-registry_server-and-metrics_api_server-admin-puts-dont-range-check-values), since all three touch the admin PUT handlers. [TD-307](#td-307-daemonconfigstoreload_all-hides-read-errors) and [TD-308](#td-308-session-counters-reuse-the-config-stores-table-and-scan-it-per-allocation) are cleanup.

**Tier 26 — Newly filed (October 11, 2026): `DataFetcher`.** [TD-309](#td-309-gutenberg-sentence-splitting-can-crash-registry_server) and [TD-310](#td-310-huggingface-split-can-write-outside-the-registry-data-directory) are HIGH,
small and reachable from any client of `registry_server`'s fetch endpoints; do them first.
[TD-312](#td-312-datafetcher-downloads-through-shell-curl-commands) and [TD-313](#td-313-huggingface-cache-accepts-partial-downloads-and-races-concurrent-fetches) are best done together (both rework the download path; libcurl gives
timeouts and per-part error handling). [TD-311](#td-311-gutenberg-training-pairs-are-next-sentence-non-sequiturs) needs the user's decision on what Gutenberg data
should teach before it's worth implementing. [TD-314](#td-314-datafetcher-source-coverage-and-hygiene-gaps) is cleanup.

**Tier 27 — Resolved (October 11, 2026): `Dataset` retired.** [TD-315](../archive/TECHNICAL_DEBT_RESOLVED.md#td-315-datasethpp-is-tagged-stable-but-has-no-production-caller) through [TD-320](../archive/TECHNICAL_DEBT_RESOLVED.md#td-320-dataset-hygiene-tolower-ub-unchecked-saves-console-output) and
[TD-219](../archive/TECHNICAL_DEBT_RESOLVED.md#td-219-datasets-batch-wrappers-have-wrong-doc-examples-and-a-partial-stats-helper) closed with `src/Dataset.hpp`'s retirement; see the resolved archive. Anything worth
rebuilding is listed under [TD-014](#td-014-llm-operations-and-training-tooling-suite).

**Tier 28 — Newly filed (October 11, 2026): `dataset_manager` argument parsing.** [TD-321](#td-321-dataset_manager-treats---count--0-as-all) first:
it's a few lines and prevents a one-typo reassignment of the whole pending pool. [TD-322](#td-322-dataset_manager-treats-unknown-flags-as-file-targets),
[TD-323](#td-323-dataset_manager-accepts-partial-and-out-of-range-numbers), [TD-324](#td-324-dataset_manager-segment-range-validation-gaps) and [TD-325](#td-325-dataset_manager-huggingface-positional-arguments-shift-silently) are small parser hardening, best done in the same change (a
shared strict-integer helper also serves TD-240 and TD-287).

**Tier 29 — Newly filed (October 11, 2026): `dataset_manager` commands.** [TD-326](#td-326-dataset_manager-migrate-duplicates-segmented-data-across-pools) first, before
anyone runs `migrate` over a segmented legacy pool; add [TD-331](#td-331-dataset_manager-has-no-tests-for-its-commands)'s CLI-level test alongside it.
[TD-327](#td-327-dataset_manager-remove-reports-success-in-remote-mode-without-removing-anything) and [TD-329](#td-329-dataset_manager-uses-the-mns-run_group-only-in-assign) are small and change what operators see. [TD-328](#td-328-dataset_manager-stores-relative-paths-in-the-local-pool), [TD-330](#td-330-dataset_manager-output-exit-code-and-flag-hygiene) and
[TD-332](#td-332-operator-guide-covers-only-8-of-15-dataset_manager-commands) are cleanup (do TD-332 after TD-321 so the guide documents the fixed semantics).

**Tier 30 — Newly filed (October 11, 2026): `DatasetRegistry`.** [TD-333](#td-333-segment-claims-arent-released-on-failure-and-trained-records-ignore-segments) is HIGH and spans the
registry, both transports, `registry_server` and the trainer's call sites; do it before segments
(TD-205) see real use, and pass real counts ([TD-334](#td-334-trained-sample-counts-are-always-zero)) in the same change since it touches the same
`mark_trained()` calls. [TD-335](#td-335-datasetregistry-dead-code-and-unused-integrity-data) and [TD-336](#td-336-legacy-training-file-parser-truncates-responses-bom-breaks-jsonl-detection) are cleanup.

**Tier 31 — Newly filed (October 11, 2026): `DataTransport`.** [TD-337](#td-337-datatransport-names-downloads-by-basename-so-segments-and-same-named-files-collide) and [TD-338](#td-338-datatransport-reuses-same-size-stale-files-without-checking-the-checksum) change the
same local-file handling and belong together (and with TD-333, since all three are about segments in
distributed training). [TD-339](#td-339-ftps-downloads-dont-verify-the-registrys-certificate) pairs with TD-284's FTPS config fix. [TD-340](#td-340-datatransport-robustness-and-header-hygiene) is cleanup.

**Tier 32 — Newly filed (October 11, 2026): `DecoderBlock`.** [TD-341](#td-341-world-model-and-hippocampal-gates-never-train-in-production) is HIGH and small: until it's
fixed, every world-model/hippocampal feature is inert in trained models. Do [TD-345](#td-345-decoderblock-persistence-gaps-for-gated-state)'s SafeTensors
part with it, so trained gates survive checkpoints, and [TD-343](#td-343-hippocampal-memory-is-updated-on-every-forward-pass) before relying on hippocampal
memory. [TD-342](#td-342-kv-cached-generation-and-the-gpu-path-skip-the-gated-paths) is the larger follow-up that makes the paths apply consistently across generation
strategies and devices. [TD-344](#td-344-decoderblock-dropout-is-stored-but-never-applied) needs a decision (implement or remove dropout).

**Tier 33 — Newly filed (October 11, 2026): `LLMDecoder`.** [TD-347](#td-347-cached-and-full-decoder-forwards-diverge-past-max_seq_length) first: capping generation at
`max_seq_length` is small and removes a strategy-dependent divergence. Fold [TD-346](#td-346-llmdecoderupdate_weights-never-updates-the-final-norm) into TD-297's
top-level update fix. [TD-348](#td-348-llmdecoder-dead-code-and-robustness-gaps) is cleanup.

**Tier 34 — Newly filed (October 11, 2026): `DocumentStore`.** Do these after TD-247 (RAG embeds with
an untrained encoder, so retrieval quality is moot until it's fixed). [TD-350](#td-350-documentstoreretrieve-isnt-safe-to-call-concurrently) must land before or
with TD-230. [TD-349](#td-349-rag-indexes-whole-files-but-embeds-only-their-first-max_seq_length-tokens) (chunking) is the main quality fix; [TD-351](#td-351-documentstore-embedding-and-housekeeping-gaps) is cleanup.

**Tier 35 — Resolved (October 11, 2026): `EfficientBatching` retired.** [TD-352](../archive/TECHNICAL_DEBT_RESOLVED.md#td-352-parallel-data-augmentation-isnt-reproducible) through [TD-355](../archive/TECHNICAL_DEBT_RESOLVED.md#td-355-efficientbatching-is-an-unused-duplicate-of-batchprocessor)
closed with the header's retirement. Its token-budget bucketing is a design note in the v2 plan (§4.1);
augmentation is listed under [TD-014](#td-014-llm-operations-and-training-tooling-suite).

**Tier 36 — Newly filed (October 11, 2026): `EncoderBlock`.** All cleanup: [TD-356](#td-356-encoderblock-gradient-norm-and-clipping-ignore-layernorm-header-describes-post-ln) and [TD-357](#td-357-encoderblock-learning-rate-handling-unchecked-loads-and-console-output).
Do the dropout decision (TD-344) for both blocks at once.

**Tier 37 — Newly filed (October 11, 2026): `EncoderDecoderModel`. Do [TD-358](#td-358-gpu-training-feeds-unshifted-targets-teaching-the-model-to-copy-instead-of-predict) first of everything
in this document.** It's a small code fix, but every GPU-trained checkpoint learned the wrong task, so
fix it, verify on ai-machine with a CPU-vs-GPU equivalence test, and plan the retraining. Fold
[TD-361](#td-361-training-targets-get-a-double-bos) (double `BOS`) into the same change, since both alter training alignment and need the same
retraining. Then [TD-359](#td-359-generate_response-and-gpu_generate_response-ignore-max_length) (small) and [TD-362](#td-362-cpu-and-gpu-generation-paths-behave-differently). [TD-360](#td-360-encoderdecodermodel-no-optimizer-path-never-updates-the-encoder) and [TD-363](#td-363-encoderdecodermodel-clone-load-and-world-model-input-gaps) are cleanup.

**Tier 38 — Newly filed (October 11, 2026): `LLMEncoder`.** Do [TD-365](#td-365-encoder-input-is-never-truncated-to-max_seq_length-when-serving) together with TD-347 (the
decoder's over-length divergence) and TD-294 (token budget vs. `max_seq_length`): truncate or reject
once, where the model tokenizes. Fix [TD-364](#td-364-llmencoderencode-input-doesnt-match-the-models-encoder-input) before anyone relies on pipeline inference.
[TD-366](#td-366-serving-encoders-cache-activations-for-backward-dead-state-and-duplicate-api) and [TD-367](#td-367-llmencoder-console-output-and-unchecked-weight-io) are cleanup.

**Tier 39 — Newly filed (October 12, 2026): `FeedForward`.** Do [TD-368](#td-368-weight-initialization-cant-be-seeded-or-reproduced) early: a seedable
initialization is what makes CPU-vs-GPU equivalence tests (TD-358) and training regressions
reproducible. Fix [TD-369](#td-369-feedforwardload_weights-loads-truncated-files-as-a-silent-mix-of-old-and-new-weights) with TD-298, TD-345, TD-357 and TD-367 as one pass over every legacy
per-layer loader. Decide [TD-371](#td-371-residual-branch-output-isnt-scaled-for-depth) alongside the open Pre-LN residual-growth investigation; it changes
initialization only, but needs a retraining comparison. [TD-370](#td-370-feedforward-argument-checks-gpu-null-dereference-and-console-output) is cleanup.

**Tier 40 — Newly filed (October 12, 2026): `FtpDataServer`.** Fix [TD-372](#td-372-ftp-data-connections-are-accepted-from-any-host-so-files-can-be-stolen) before FTP delivery is
used on any network you don't fully trust: check the data connection's peer address against the
control connection's. Do [TD-373](#td-373---ftps-doesnt-require-encryption) with TD-339 (client certificate verification) and TD-284 (FTPS
config keys) as one FTPS-hardening pass. Then [TD-375](#td-375-ftpdataserver-has-no-connection-limits-or-timeouts) and [TD-374](#td-374-ftp-re-login-leaks-rate-limit-slots), both small. [TD-376](#td-376-ftpdataserver-slow-shutdown-unencrypted-feat-and-silent-start-failures) and
[TD-377](#td-377-ftpdataserver-token-semantics-and-setup-gaps) are cleanup; TD-376's shutdown delay is a one-line condition-variable change.

**Tier 41 — Newly filed (October 12, 2026): `GenerationQualityMetrics`.** Fix [TD-378](#td-378-bleu-smoothing-inflates-unrelated-answers-and-zeroes-correct-short-ones) before
reading anything into BLEU trends; it's small and self-contained. Do [TD-379](#td-379-bleu1bleu2-are-computed-but-never-stored) alongside it (same
reporting path). [TD-380](#td-380-generation-quality-scores-arent-comparable-or-well-sampled) is documentation plus a sample-size default.

**Tier 42 — Newly filed (October 12, 2026): `GPUUtils` (CUDA).** [TD-381](#td-381-chatbot_api_server-ignores-gpu_strategy) is a few lines; do it
whenever `chatbot_api_server`'s GPU startup is touched. Do [TD-382](#td-382-gpu-work-from-any-thread-shares-unsynchronized-gpumanager-state) before relying on the async
BLEU/ROUGE path on a GPU host (or with TD-378 and the TD-358 retraining runs, since both use that
thread). [TD-383](#td-383-gpumanager-lifecycle-gaps-set_device-cleanup-initialize) is cleanup. Revisit all three when the SYCL `GPUUtils_SYCL.hpp` is traced, since
ai-machine runs that backend.

**Tier 43 — Newly filed (October 12, 2026): `MatrixGPU.cu`.** Fix [TD-384](#td-384-shared-memory-race-in-gpu-softmax-layernorm-and-cross-entropy-kernels-cuda-and-sycl) together with TD-358,
in both backends, before the TD-358 retraining runs: it's four one-line barriers per backend, and it
removes an intermittent source of wrong rows from every GPU training step. [TD-385](#td-385-matrix_sum_gpu-allocates-and-synchronizes-on-every-call) and
[TD-386](#td-386-gpu-kernel-input-checks-and-dead-batch-functions) are cleanup.

**Tier 44 — Newly filed (October 12, 2026): `MatrixGPU.hpp`.** Fix [TD-387](#td-387-gpu-generation-fails-on-input-longer-than-max_seq_length) together with TD-365:
truncating encoder input once, where the model tokenizes, fixes both the CPU positional-encoding gap
and the GPU crash. [TD-388](#td-388-gpukvcache-copies-the-whole-cache-every-decode-step) matters once GPU serving latency is measured (TD-033/TD-050's open
benchmark). [TD-389](#td-389-gpumatrix-shape-checks-uninitialized-memory-and-untested-gpu-kv-caches) is cleanup, but its KV-cache tests are worth adding before that benchmark.

**Tier 45 — Newly filed (October 12, 2026): `GPUUtils_SYCL`.** [TD-391](#td-391-sycl-lists-the-same-gpu-once-per-platform-and-doesnt-log-the-backend) first, as a cheap check: have
ai-machine log the selected platform (or run `sycl-ls`) to confirm it's on Level Zero. Then measure
[TD-390](#td-390-sycl-deferred-frees-and-blocking-copies-stall-the-in-order-queue) on ai-machine (time a training step with and without a buffer pool) before rewriting the
allocator; it's likely the largest SYCL performance item. [TD-392](#td-392-sycl-gpumemory0-throws-where-cuda-succeeds) is small.

**Tier 46 — Newly filed (October 12, 2026): `MatrixGPU_SYCL.cpp`.** Do [TD-394](#td-394-gpu-kernel-tests-cant-run-without-a-gpu) first or alongside
TD-384: a CPU-device test mode turns the harness that found both bugs into a permanent regression
test, so TD-384's fix (and TD-358's) can be verified without ai-machine. Fix [TD-393](#td-393-sycl-sub-group-reduction-is-wrong-when-the-sub-group-size-is-8) in the same
change (pin the sub-group size or loop over the partials).

**Tier 47 — Newly filed (October 12, 2026): `MatrixGPU_SYCL.hpp`.** [TD-395](#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) is a refactor; do it
before the next `GPUMatrix` change (for example TD-389's shape checks), so that change is made once.

**Tier 48 — Newly filed (October 13, 2026): `HippocampalMemory`.** Fix [TD-396](#td-396-hippocampal-memory-eviction-starves-new-memories) together with
TD-341: while the gate is stuck at 0 the frozen memory changes nothing, but the moment TD-341 makes
the hippocampal path live, a memory that never records anything new would be what it reads.
[TD-397](#td-397-hippocampal-swap-file-grows-without-bound) matters for any long-running `chatbot_api_server` with hippocampal memory enabled; [TD-398](#td-398-hippocampalmemory-per-step-copies-and-unchecked-saves)
is cleanup.

**Tier 49 — Newly filed (October 13, 2026): `IMetricsReporter`.** Both are cleanup: [TD-399](#td-399-imetricsreporter-buffering-and-ordering-contract-is-undocumented) is
documentation plus an optional text cap; do [TD-400](#td-400-nullmetricsreporter-keeps-reporting-only-work-switched-on) together with TD-370 if GPU hook synchronization
turns out to matter on ai-machine.

**Tier 50 — Newly filed (October 13, 2026): `IncrementalTrainerArgs`.** [TD-401](#td-401-incremental_trainer-flag-parsing-crashes-or-silently-misreads-bad-values) and [TD-402](#td-402---dataset-kind-is-not-validated) are one
small validation pass over the parser (mirroring TD-321–TD-325's `dataset_manager` fixes). [TD-403](#td-403-trainer_service-drops---objective-and---dataset-kind)
needs a decision on whether `trainer_service` should support LeJEPA; [TD-404](#td-404-derive_run_id-can-collide-across-hosts) matters only with a
shared remote registry.

**Tier 51 — Newly filed (October 13, 2026): `IncrementalTrainer`. Do these before the TD-358
retraining**, or the retrained model inherits the same problems: [TD-405](#td-405-each-training-pass-restarts-from-the-best-ever-checkpoint-not-the-latest) (resume from the latest
completed checkpoint by default; keep "best" as an explicit choice), [TD-406](#td-406-a-paused-or-crashed-passs-checkpoint-is-deleted-on-the-next-start) (find and keep the
interrupted session's snapshots instead of deleting them), and [TD-407](#td-407-checkpoint-deletion-leaks-the-per-layer-weight-files) (delete every file
`save_model()` writes, ideally via a shared helper that lists them). Ask the user to check
`SESSION_DIR`'s disk use on ai-machine. [TD-408](#td-408-latestbest-checkpoint-symlinks-always-dangle) and [TD-409](#td-409-tokenized-data-cache-key-has-no-preprocessing-version) are small; clear the tokenized cache as
part of any tokenization fix. [TD-410](#td-410-incrementaltrainer-session-bookkeeping-and-hard-coded-training-policy) is cleanup.

**Tier 52 — Newly filed (October 13, 2026): `IncrementalTrainingTool`.** Fix [TD-412](#td-412-a-training-pass-that-loads-zero-samples-marks-its-files-trained) immediately (a
one-line "zero samples = failure" guard), then [TD-411](#td-411-resume-every-trainer_service-pass-never-downloads-over-ftp) (share `train`'s download step with
`resume_last_session()`). Ask the user whether ai-machine's `trainer_service` uses a remote FTP
registry; if so, files marked trained since it went live may need re-queuing. [TD-414](#td-414-first-run-vocabulary-bootstrap-is-unreachable-from-the-cli) and
[TD-413](#td-413-standalone-trainretrain-pre-check-reads-the-wrong-pending-pool) fix standalone first-run use; [TD-415](#td-415-lejepa-pass-ignores-segment-ranges-and-strands-claims-after-a-crash) with TD-404; [TD-416](#td-416-incremental_trainer-command-hygiene) is cleanup.

**Tier 53 — Newly filed (October 13, 2026): `IntegratedInferenceEngine`.** Decide first whether to
keep `--integrated-inference` at all: it batches nothing (TD-211's gap), so v1.x could simply route
`/chat` through the normal path and retire the engine with the v2 real-batching work. If kept, do
[TD-418](#td-418-integratedinferenceengine-worker-exceptions-terminate-the-server-shutdown-drops-requests) and [TD-419](#td-419-integratedinferenceengine-uses-the-model-without-its-mutex) (small, prevent crashes and races), then [TD-417](#td-417---integrated-inference-answers-differ-from-normal-chat); [TD-420](#td-420-integratedinferenceengine-claims-dead-config-and-fictional-stats) is documentation.

**Tier 54 — Newly filed (October 13, 2026): `KVCache`.** [TD-422](#td-422-kvcache-copies-the-whole-cache-on-every-append): pre-allocate to `max_seq_length`
like `GPUKVCache` and expose views, ideally together with TD-388 (the GPU cache's per-step copy) and
TD-398 (`HippocampalMemory::read_all()`'s copy), the same pattern three times. [TD-421](#td-421-kvcacheappend-has-no-shape-checks) is a few
checks to add in the same change.

**Tier 55 — Newly filed (October 14, 2026): `LanguageModelHead`, `LayerNorm`.** Fix [TD-423](#td-423-a-truncated-lm_head-file-loads-as-all-nan-logits) in the same pass as
TD-298, TD-345, TD-357, TD-367, TD-369 and [TD-426](#td-426-layernorm-silent-setters-missing-shape-checks-and-unchecked-loads) (every legacy per-layer loader). [TD-424](#td-424-uncached-generation-projects-every-position-through-the-lm-head) is a small
"project only the last row" change in the generation callbacks; [TD-425](#td-425-languagemodelhead-hygiene) is cleanup.

**Tier 56 — Newly filed (October 14, 2026): `LeJEPAEncoder`.** Fix [TD-427](#td-427-lejepa-predictor-is-never-saved-so-each-pass-restarts-it) before the next
world-model pretraining run (save/load the predictor with the encoder). Do [TD-428](#td-428-lejepa-two-update-workaround-is-obsolete-after-td-195) with a short
before/after comparison since it changes the optimization; [TD-429](#td-429-lejepaencoder-exception-escape-console-output-and-unchecked-loads) is cleanup.

**Tier 57 — Newly filed (October 14, 2026): `Logger`.** [TD-431](#td-431-trainer-processes-share-one-rotating-log-file-and-ignore-log_level) is a quick win for debugging
ai-machine: give `trainer_service` and its children distinct log files (or console-only when
`LOG_FILE_PATH` is empty, per LOG_FILE_STANDARDS.md) and honour `LOG_LEVEL`. [TD-430](#td-430-logger-lazy-initialization-races-and-can-throw-from-a-log-call) is a small
`std::call_once`/mutex change.

**Tier 58 — Newly filed (October 14, 2026): `LoRA`.** Both are cleanup for an experimental, test-only
feature: do [TD-432](#td-432-loraadapterload-silently-adopts-the-files-shape) and [TD-433](#td-433-lora-initialization-update-and-config-gaps) together with TD-302 if LoRA gets a real entry point (for example an
`incremental_trainer` fine-tuning mode); otherwise leave them parked.

**Tier 59 — Newly filed (October 14, 2026): `Matrix`.** [TD-434](#td-434-matrixoperator-runs-short-products-single-threaded-and-portable-builds-have-no-simd) is a cheap, high-leverage CPU
speedup (a one-line OpenMP gate change, verified 2–6× on the common shapes) plus runtime AVX2 dispatch for
`portable` binaries; re-run `matrixTests`/`matrix_simd_test` and a CPU training smoke test after it.
[TD-435](#td-435-matrix-accessor-cost-sum-accuracy-randomize-race-and-api-hygiene) is cleanup; fold the `randomize()` part into TD-368's seeding fix.

**Tier 60 — Newly filed (October 14, 2026): `MatrixSIMD`.** [TD-436](#td-436-sycl-icpx-builds-get-no--march-or-simd-flags) is a one-line CMake fix
for ai-machine's build; ship it with the next ai-machine deployment and compare a pass's wall time
before/after. [TD-437](#td-437-simd-self-check-test-never-compiles-and-no-startup-cpu-check) is cleanup; do its detection fix before TD-434's runtime dispatch relies on it.

**Tier 61 — Newly filed (October 14, 2026): `MetricsApiServerArgs`.** Do [TD-438](#td-438-metrics_api_server-accepts-invalid-numbers-and-exits-0-on-bad-arguments) together with
TD-240 (the same `atoi` pattern in `chatbot_api_server`) using one shared checked-integer helper, and
[TD-439](#td-439-metrics_api_server-backend-name-and-admin-store-location-gaps) alongside TD-306.

**Tier 62 — Newly filed (October 14, 2026): `MetricsDatabase`.** [TD-440](#td-440-metrics-storage-has-no-retention-and-max_records_on_disk-is-never-applied) needs a short owner
decision (how long to keep history, and whether the write-only tables should get readers or be dropped)
before implementing retention; check the metrics host's disk usage first. [TD-441](#td-441-imetricsdatabase-contract-gaps-step-counter-limit-order-unbounded-reads) is cleanup.

**Tier 63 — Newly filed (October 14, 2026): `MetricsPushClient`.** Do [TD-442](#td-442-metricspushclientend_session-drain-can-delay-a-passs-final-save-by-an-hour) before relying on the
always-on trainer with a metrics server that may restart (a bounded drain plus saving before `end_session()` is a
small change). [TD-443](#td-443-metrics-push-payloads-send-naninf-round-small-values-to-0-and-omit-adaptive-clipping) should fix both ends (client serialization and the server's `stof` parsing/output)
so diverging runs stay visible. [TD-444](#td-444-metricspushclient-transport-gaps-no-https-a-connection-per-event-silent-failures) matters only if metrics move behind TLS or the Cloudflare tunnel.

**Tier 64 — Newly filed (October 14, 2026): `MetricsSessionRegistry`.** [TD-445](#td-445-finished-metrics-sessions-hold-max_live_sessions-slots-for-an-hour) is a small fix with a
direct effect on the always-on trainer's visibility (evict the oldest finished session before refusing); do it
with TD-442. [TD-446](#td-446-metricssessionregistry-sweep-interval-changes-miss-logging-and-lock-held-io) is cleanup; its sweep-interval part goes with TD-306/TD-438's range checks.

**Tier 65 — Newly filed (October 14, 2026): `MetricsTracker`.** [TD-447](#td-447-metricstracker-printer-state-leak-convergence-off-by-one-and-smoothing-gaps) is cleanup with no training
impact; do it whenever the header is next touched.

**Tier 66 — Newly filed (October 14, 2026): `MnsCliCommands`.** Do [TD-448](#td-448-mns_cli-ignores-unknown-flags-misreads-set-training-arguments-and-aborts-on-bad-numbers) before the next model
registration (strict flag parsing is a small change with a large payoff, since registration is irreversible);
[TD-449](#td-449-mns_cli-request-building-gaps-layer-counts-relink-resets-encoding) alongside it.

**Tier 67 — Newly filed (October 14, 2026): `MnsCliTool`.** Do [TD-450](#td-450-mns_cli-register-takes-architecture-defaults-from-configmnsconf-not-the-trainers-config) with TD-448 before the next
registration (make `register` require explicit architecture or read the trainer's config, and fix the guide).
[TD-451](#td-451-mns_cli-has-no-https-and-minor-option-handling-gaps) only matters if `mns_cli` must go through the tunnel; pair its HTTPS work with TD-444.

**Tier 68 — Newly filed (October 14, 2026): `MnsJsonHelpers`.** [TD-453](#td-453-hand-written-json-duplicated-across-mns-and-metrics-code) is the umbrella: moving the MNS and
metrics code onto the vendored `nlohmann/json` fixes the escaping parts of TD-443, TD-449 and [TD-452](#td-452-mnsjsonhelpers-parsing-gaps-and-a-gui-register-body-without-td-196-fields) in one
place; do it with TD-226. TD-452's GUI register fields are separate and only matter if the GUI is used.

**Tier 69 — Newly filed (October 14, 2026): `MnsManagerGUI`.** [TD-454](#td-454-mns-manager-guis-training--candidate-workflow-fails-against-server-allocated-run-ids) is a small fix (drop the run-ID
requirement, add a "new run" checkbox, use the run ID from the record for Set Candidate) plus a guide correction; do it
with TD-452 if the GUI is in use. [TD-455](#td-455-mns-manager-gui-error-detection-ui-thread-http-and-register-defaults) is cleanup.

**Tier 70 — Newly filed (October 14, 2026): `MnsServerArgs`.** Fix [TD-456](#td-456-mns_server-treats-a-mistyped---admin-enabled-as-true-and-doesnt-range-check---port) together with TD-438 and TD-240 using one
shared checked port/boolean parser for all the daemons.

**Tier 71 — Newly filed (October 14, 2026): `ModelNameClient`.** Fix [TD-458](#td-458-a-nan-training-loss-makes-mnss-json-unparseable) with TD-443 (shared non-finite
handling on both ends) before relying on MNS's JSON from any strict client; [TD-457](#td-457-modelnameclient-retries-non-idempotent-requests-after-a-timeout) is a small change (retry only
idempotent requests); [TD-459](#td-459-modelnameclient-validation-unescaping-and-slow-failure-gaps) is cleanup, its architecture validation the most useful part.

**Tier 72 — Newly filed (October 14, 2026): `ModelNameService`.** [TD-460](#td-460-mns-retirement-leaves-role-mappings-and-is-undone-by-running-trainers) and [TD-461](#td-461-mns-accepts-unbuildable-architectures-at-registration) are small
server-side checks with direct operational effect (role resolution and irreversible registration); do them with TD-448/TD-450
before the next registration or promotion. [TD-462](#td-462-modelnameservice-dead-datasets-proxy-admin-settings-race-and-swallowed-persistence-errors) is cleanup; its JSON part goes with TD-453.

**Tier 73 — Newly filed (October 14, 2026): `ModelNameServiceServer`.** [TD-463](#td-463-mns_server-runs-in-memory-and-reports-healthy-when-modelsdb-cant-be-opened) is a few lines (fail startup, or
report persistence in `/health`) protecting the system of record; [TD-464](#td-464-mns_server-startup-crash-early-sigterm-hang-and-fixed-log-level) goes with TD-248/TD-249's shutdown fixes.

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
  - [TD-211: BatchedInferenceEngine Queues and Serializes Requests but Never Batches the Model](#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model)
  - [TD-212: BatchedInferenceEngine Can Fail or Abandon Requests It Already Accepted](#td-212-batchedinferenceengine-can-fail-or-abandon-requests-it-already-accepted)
  - [TD-213: BatchedInferenceEngine Stats Are Misleading and Unused in Production](#td-213-batchedinferenceengine-stats-are-misleading-and-unused-in-production)
  - [TD-214: Batched-Inference Mode Generates Differently From the Inline Path](#td-214-batched-inference-mode-generates-differently-from-the-inline-path)
  - [TD-215: Activation Has Misleading Header Comments and Unchecked Inputs](#td-215-activation-has-misleading-header-comments-and-unchecked-inputs)
  - [TD-216: Softmax and GELU Math Is Duplicated Across CPU and GPU Code](#td-216-softmax-and-gelu-math-is-duplicated-across-cpu-and-gpu-code)
  - [TD-217: `/chat/batch` Reports Hypothetical Batching Stats as if They Were Real](#td-217-chatbatch-reports-hypothetical-batching-stats-as-if-they-were-real)
  - [TD-218: BatchProcessor Helpers Have Dead Surface, Lost Ordering, and Unchecked Inputs](#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs)
  - [TD-220: BPE Merge Learning Is Corrupted by the `"|||"` Pair-Key Separator](#td-220-bpe-merge-learning-is-corrupted-by-the--pair-key-separator)
  - [TD-221: `BPETokenizer::save_vocab()` Fails Silently, So Checkpoints Can Lack Their Vocab](#td-221-bpetokenizersave_vocab-fails-silently-so-checkpoints-can-lack-their-vocab)
  - [TD-222: `apply_bpe()`'s Thread-Local Cache Is Never Invalidated](#td-222-apply_bpes-thread-local-cache-is-never-invalidated)
  - [TD-223: Tokenizer Mutators Can Leave It Gutted or With Duplicate IDs](#td-223-tokenizer-mutators-can-leave-it-gutted-or-with-duplicate-ids)
  - [TD-224: Tokenizer Normalization Is Lossy: No Capitals, Newlines, or Tabs](#td-224-tokenizer-normalization-is-lossy-no-capitals-newlines-or-tabs)
  - [TD-225: BPETokenizer Logging and API Hygiene](#td-225-bpetokenizer-logging-and-api-hygiene)
  - [TD-226: `/chat/batch` Rejects Any Request Whose Messages Contain `]`](#td-226-chatbatch-rejects-any-request-whose-messages-contain-)
  - [TD-227: ChatbotAPI Session Pointers Are Used Outside the Session Lock](#td-227-chatbotapi-session-pointers-are-used-outside-the-session-lock)
  - [TD-228: `ChatbotAPI::parse_json_string()` Mis-Parses Common JSON](#td-228-chatbotapiparse_json_string-mis-parses-common-json)
  - [TD-229: ChatbotAPI Responses Can Be Invalid JSON (Unescaped Control Characters)](#td-229-chatbotapi-responses-can-be-invalid-json-unescaped-control-characters)
  - [TD-230: RAG Mode Serializes Every ChatbotAPI Request Behind `config_mutex_`](#td-230-rag-mode-serializes-every-chatbotapi-request-behind-config_mutex_)
  - [TD-231: ChatbotAPI Sessions Only Expire When `GET /health` Is Called](#td-231-chatbotapi-sessions-only-expire-when-get-health-is-called)
  - [TD-232: ChatbotAPI Accepts Unlimited Client-Chosen Session IDs](#td-232-chatbotapi-accepts-unlimited-client-chosen-session-ids)
  - [TD-233: ChatbotAPI Session IDs Come From a Non-Cryptographic Generator](#td-233-chatbotapi-session-ids-come-from-a-non-cryptographic-generator)
  - [TD-234: ChatbotAPI Batch Endpoints Bypass the Inference Dispatcher](#td-234-chatbotapi-batch-endpoints-bypass-the-inference-dispatcher)
  - [TD-235: ChatbotAPI Batch Endpoints Have No Size Limit](#td-235-chatbotapi-batch-endpoints-have-no-size-limit)
  - [TD-236: ChatbotAPI Returns HTTP 200 for Validation Errors](#td-236-chatbotapi-returns-http-200-for-validation-errors)
  - [TD-237: `ChatbotAPI::running_` Is a Non-Atomic Flag Written From Two Threads](#td-237-chatbotapirunning_-is-a-non-atomic-flag-written-from-two-threads)
  - [TD-238: ChatbotAPI Returns Internal Exception Text to Clients](#td-238-chatbotapi-returns-internal-exception-text-to-clients)
  - [TD-239: ChatbotAPI Logging and Dead Code Cleanup](#td-239-chatbotapi-logging-and-dead-code-cleanup)
  - [TD-240: `chatbot_api_server` Silently Accepts Non-Numeric and Partial Numbers](#td-240-chatbot_api_server-silently-accepts-non-numeric-and-partial-numbers)
  - [TD-241: `chatbot_api_server` Never Validates Its Configuration at Startup](#td-241-chatbot_api_server-never-validates-its-configuration-at-startup)
  - [TD-242: `SIGHUP` Reload Discards `chatbot_api_server`'s CLI Generation Overrides](#td-242-sighup-reload-discards-chatbot_api_servers-cli-generation-overrides)
  - [TD-243: `chatbot_api_server` Starts Every Engine Mode Flag Passed, Not Just One](#td-243-chatbot_api_server-starts-every-engine-mode-flag-passed-not-just-one)
  - [TD-244: `chatbot_api_server` Ignores the `TOP_K` and `BEAM_WIDTH` Settings](#td-244-chatbot_api_server-ignores-the-top_k-and-beam_width-settings)
  - [TD-245: `chatbot_api_server` Argument Errors Are Misleading or Silent](#td-245-chatbot_api_server-argument-errors-are-misleading-or-silent)
  - [TD-246: `chatbot_api_server` Serves a Randomly Initialized Model When Loading Fails](#td-246-chatbot_api_server-serves-a-randomly-initialized-model-when-loading-fails)
  - [TD-247: RAG Retrieval in `chatbot_api_server` Uses an Untrained Encoder](#td-247-rag-retrieval-in-chatbot_api_server-uses-an-untrained-encoder)
  - [TD-248: `chatbot_api_server` Can Hang If Shutdown Arrives During Server Startup](#td-248-chatbot_api_server-can-hang-if-shutdown-arrives-during-server-startup)
  - [TD-249: `chatbot_api_server`'s `SIGHUP` Handler Isn't Async-Signal-Safe](#td-249-chatbot_api_servers-sighup-handler-isnt-async-signal-safe)
  - [TD-250: Hippocampal Memory in `chatbot_api_server` Is Lost on Crash and Can Land in the Working Directory](#td-250-hippocampal-memory-in-chatbot_api_server-is-lost-on-crash-and-can-land-in-the-working-directory)
  - [TD-251: `chatbot_api_server` Reload Silently Ignores Most Settings](#td-251-chatbot_api_server-reload-silently-ignores-most-settings)
  - [TD-252: `chatbot_api_server` Startup Log and Usage Text Are Inaccurate](#td-252-chatbot_api_server-startup-log-and-usage-text-are-inaccurate)
  - [TD-253: `chatbot` Spins Forever at End of Input](#td-253-chatbot-spins-forever-at-end-of-input)
  - [TD-254: `chatbot`'s `/set` Generation Settings Have No Effect](#td-254-chatbots-set-generation-settings-have-no-effect)
  - [TD-255: `chatbot` Shows Server Error Replies as Empty Responses](#td-255-chatbot-shows-server-error-replies-as-empty-responses)
  - [TD-256: `chatbot`'s Help Banner Omits `/save`, `/load` and `/stats`](#td-256-chatbots-help-banner-omits-save-load-and-stats)
  - [TD-257: `chatbot` Sends `session_id` Unescaped in Two Requests](#td-257-chatbot-sends-session_id-unescaped-in-two-requests)
  - [TD-258: `chatbot` Can Lose or Overwrite Saved Conversations](#td-258-chatbot-can-lose-or-overwrite-saved-conversations)
  - [TD-259: `chatbot` Code Hygiene: Unchecked `/set` Values, Global Helpers, Heavy Header](#td-259-chatbot-code-hygiene-unchecked-set-values-global-helpers-heavy-header)
  - [TD-260: `chatbot_gui` Never Loads a Trained Checkpoint](#td-260-chatbot_gui-never-loads-a-trained-checkpoint)
  - [TD-261: `generate_response_with_strategy()` Forces Beam Search Whenever `num_beams > 1`](#td-261-generate_response_with_strategy-forces-beam-search-whenever-num_beams--1)
  - [TD-262: `chatbot_gui` Ignores MNS Architecture and Becomes Unusable on Init Failure](#td-262-chatbot_gui-ignores-mns-architecture-and-becomes-unusable-on-init-failure)
  - [TD-263: `chatbot_gui` Freezes While Generating](#td-263-chatbot_gui-freezes-while-generating)
  - [TD-264: `chatbot_gui` Launcher Hardcodes Qt5 and x86_64 Paths](#td-264-chatbot_gui-launcher-hardcodes-qt5-and-x86_64-paths)
  - [TD-265: `chatbot_gui` and `chatbot_api_server` Format Conversation Context Differently](#td-265-chatbot_gui-and-chatbot_api_server-format-conversation-context-differently)
  - [TD-266: `chatbot_gui` Minor Gaps](#td-266-chatbot_gui-minor-gaps)
  - [TD-267: ChatbotTrainer Keeps Skipped Pairs, Biasing Validation Loss Low](#td-267-chatbottrainer-keeps-skipped-pairs-biasing-validation-loss-low)
  - [TD-268: ChatbotTrainer Only Ends Metrics Epochs When There Is Validation Data](#td-268-chatbottrainer-only-ends-metrics-epochs-when-there-is-validation-data)
  - [TD-269: Optimizer State and LR Schedule Restart Every Incremental Training Pass](#td-269-optimizer-state-and-lr-schedule-restart-every-incremental-training-pass)
  - [TD-270: Adaptive Gradient Clipping Resets Every Epoch](#td-270-adaptive-gradient-clipping-resets-every-epoch)
  - [TD-271: ChatbotTrainer's Tokenized Cache Can Pair With a Different Random Split](#td-271-chatbottrainers-tokenized-cache-can-pair-with-a-different-random-split)
  - [TD-272: ChatbotTrainer Silently Drops Gradients on Mid-Window Errors and Under-Weights the Last Window](#td-272-chatbottrainer-silently-drops-gradients-on-mid-window-errors-and-under-weights-the-last-window)
  - [TD-273: ChatbotTrainer Quality Backfill Writes Results Nobody Can Read](#td-273-chatbottrainer-quality-backfill-writes-results-nobody-can-read)
  - [TD-274: ChatbotTrainer Saves the Best Model to a Fixed File in the Working Directory](#td-274-chatbottrainer-saves-the-best-model-to-a-fixed-file-in-the-working-directory)
  - [TD-275: ChatbotTrainer Drops `<eos>` From Long Training Responses](#td-275-chatbottrainer-drops-eos-from-long-training-responses)
  - [TD-276: ChatbotTrainer Hygiene: Dead Fields, Logging, Stale Test File](#td-276-chatbottrainer-hygiene-dead-fields-logging-stale-test-file)
  - [TD-277: `ChildProcess` Reports Every Signal Death as Exit Code 128](#td-277-childprocess-reports-every-signal-death-as-exit-code-128)
  - [TD-278: `ChildProcess` Reports a Failed `exec` as a Successful Launch](#td-278-childprocess-reports-a-failed-exec-as-a-successful-launch)
  - [TD-279: `ChildProcess::request_stop()` Can Signal Every Process if Called Concurrently](#td-279-childprocessrequest_stop-can-signal-every-process-if-called-concurrently)
  - [TD-280: `trainer_service` Passes Can Be Orphaned if the Supervisor Is Hard-Killed](#td-280-trainer_service-passes-can-be-orphaned-if-the-supervisor-is-hard-killed)
  - [TD-281: `ChildProcess` Forks a Multithreaded Process and Leaks File Descriptors](#td-281-childprocess-forks-a-multithreaded-process-and-leaks-file-descriptors)
  - [TD-282: `ChildProcess` Windows Path Has Known Gaps](#td-282-childprocess-windows-path-has-known-gaps)
  - [TD-283: `SIGHUP` Terminates Every Binary Except `chatbot_api_server`](#td-283-sighup-terminates-every-binary-except-chatbot_api_server)
  - [TD-284: `registry_server` Ignores the FTP/FTPS Security Keys Its Config File Ships With](#td-284-registry_server-ignores-the-ftpftps-security-keys-its-config-file-ships-with)
  - [TD-285: `ConfigLoader::reload()` Reports Success Without Applying Untracked Changes](#td-285-configloaderreload-reports-success-without-applying-untracked-changes)
  - [TD-286: `ConfigLoader` Parses Booleans Two Different Ways](#td-286-configloader-parses-booleans-two-different-ways)
  - [TD-287: `ConfigLoader` Accepts Negative and Partial Numbers](#td-287-configloader-accepts-negative-and-partial-numbers)
  - [TD-288: `ConfigLoader::validate()` Misses Fields That Crash or Misbehave](#td-288-configloadervalidate-misses-fields-that-crash-or-misbehave)
  - [TD-289: `ConfigLoader` Keeps Inline Comments and Silently Accepts Enum Typos](#td-289-configloader-keeps-inline-comments-and-silently-accepts-enum-typos)
  - [TD-290: `ConfigLoader` Output Hygiene](#td-290-configloader-output-hygiene)
  - [TD-291: Conversation Imports Can Lift Session Limits and Budgets](#td-291-conversation-imports-can-lift-session-limits-and-budgets)
  - [TD-292: One Long Message Wipes the Whole Conversation](#td-292-one-long-message-wipes-the-whole-conversation)
  - [TD-293: A Malformed Conversation Import Wipes the Live Session](#td-293-a-malformed-conversation-import-wipes-the-live-session)
  - [TD-294: Conversation Token Budget Is Unrelated to the Model's Sequence Length](#td-294-conversation-token-budget-is-unrelated-to-the-models-sequence-length)
  - [TD-295: Conversation Role Labels Are Spoofable and Roles Aren't Escaped](#td-295-conversation-role-labels-are-spoofable-and-roles-arent-escaped)
  - [TD-296: ConversationContext Hygiene](#td-296-conversationcontext-hygiene)
  - [TD-297: Layer `update_weights()` Steps a Shared Optimizer Once per Layer](#td-297-layer-update_weights-steps-a-shared-optimizer-once-per-layer)
  - [TD-298: `CrossAttention::load()` Loads Truncated Files as NaN Weights](#td-298-crossattentionload-loads-truncated-files-as-nan-weights)
  - [TD-299: `CrossAttention` Crashes with SIGFPE When `num_heads` Is Zero](#td-299-crossattention-crashes-with-sigfpe-when-num_heads-is-zero)
  - [TD-300: `CrossAttention::forward_with_cache()` Silently Reuses Stale Encoder K/V](#td-300-crossattentionforward_with_cache-silently-reuses-stale-encoder-kv)
  - [TD-301: `CrossAttention` GPU Path Diverges from the CPU Path](#td-301-crossattention-gpu-path-diverges-from-the-cpu-path)
  - [TD-302: LoRA Adapters Aren't Persisted or Counted in Gradient Norms](#td-302-lora-adapters-arent-persisted-or-counted-in-gradient-norms)
  - [TD-303: Decoder-Only Mode Still Runs Cross-Attention over a Dummy Input](#td-303-decoder-only-mode-still-runs-cross-attention-over-a-dummy-input)
  - [TD-304: Session Counters Can Be Duplicated and `DaemonConfigStore` Has No Busy Timeout](#td-304-session-counters-can-be-duplicated-and-daemonconfigstore-has-no-busy-timeout)
  - [TD-305: A Rejected or Failed `PUT /admin/config` Still Partly Applies](#td-305-a-rejected-or-failed-put-adminconfig-still-partly-applies)
  - [TD-306: `registry_server` and `metrics_api_server` Admin PUTs Don't Range-Check Values](#td-306-registry_server-and-metrics_api_server-admin-puts-dont-range-check-values)
  - [TD-307: `DaemonConfigStore::load_all()` Hides Read Errors](#td-307-daemonconfigstoreload_all-hides-read-errors)
  - [TD-308: Session Counters Reuse the Config Store's Table and Scan It per Allocation](#td-308-session-counters-reuse-the-config-stores-table-and-scan-it-per-allocation)
  - [TD-309: Gutenberg Sentence Splitting Can Crash `registry_server`](#td-309-gutenberg-sentence-splitting-can-crash-registry_server)
  - [TD-310: HuggingFace `split` Can Write Outside the Registry Data Directory](#td-310-huggingface-split-can-write-outside-the-registry-data-directory)
  - [TD-311: Gutenberg Training Pairs Are Next-Sentence Non-Sequiturs](#td-311-gutenberg-training-pairs-are-next-sentence-non-sequiturs)
  - [TD-312: `DataFetcher` Downloads Through Shell `curl` Commands](#td-312-datafetcher-downloads-through-shell-curl-commands)
  - [TD-313: HuggingFace Cache Accepts Partial Downloads and Races Concurrent Fetches](#td-313-huggingface-cache-accepts-partial-downloads-and-races-concurrent-fetches)
  - [TD-314: `DataFetcher` Source Coverage and Hygiene Gaps](#td-314-datafetcher-source-coverage-and-hygiene-gaps)
  - [TD-321: `dataset_manager` Treats `--count` ≤ 0 as "All"](#td-321-dataset_manager-treats---count--0-as-all)
  - [TD-322: `dataset_manager` Treats Unknown Flags as File Targets](#td-322-dataset_manager-treats-unknown-flags-as-file-targets)
  - [TD-323: `dataset_manager` Accepts Partial and Out-of-Range Numbers](#td-323-dataset_manager-accepts-partial-and-out-of-range-numbers)
  - [TD-324: `dataset_manager segment` Range Validation Gaps](#td-324-dataset_manager-segment-range-validation-gaps)
  - [TD-325: `dataset_manager huggingface` Positional Arguments Shift Silently](#td-325-dataset_manager-huggingface-positional-arguments-shift-silently)
  - [TD-326: `dataset_manager migrate` Duplicates Segmented Data Across Pools](#td-326-dataset_manager-migrate-duplicates-segmented-data-across-pools)
  - [TD-327: `dataset_manager remove` Reports Success in Remote Mode Without Removing Anything](#td-327-dataset_manager-remove-reports-success-in-remote-mode-without-removing-anything)
  - [TD-328: `dataset_manager` Stores Relative Paths in the Local Pool](#td-328-dataset_manager-stores-relative-paths-in-the-local-pool)
  - [TD-329: `dataset_manager` Uses the MNS `run_group` Only in `assign`](#td-329-dataset_manager-uses-the-mns-run_group-only-in-assign)
  - [TD-330: `dataset_manager` Output, Exit Code and Flag Hygiene](#td-330-dataset_manager-output-exit-code-and-flag-hygiene)
  - [TD-331: `dataset_manager` Has No Tests for Its Commands](#td-331-dataset_manager-has-no-tests-for-its-commands)
  - [TD-332: Operator Guide Covers Only 8 of 15 `dataset_manager` Commands](#td-332-operator-guide-covers-only-8-of-15-dataset_manager-commands)
  - [TD-333: Segment Claims Aren't Released on Failure and Trained Records Ignore Segments](#td-333-segment-claims-arent-released-on-failure-and-trained-records-ignore-segments)
  - [TD-334: Trained Sample Counts Are Always Zero](#td-334-trained-sample-counts-are-always-zero)
  - [TD-335: `DatasetRegistry` Dead Code and Unused Integrity Data](#td-335-datasetregistry-dead-code-and-unused-integrity-data)
  - [TD-336: Legacy Training-File Parser Truncates Responses; BOM Breaks JSONL Detection](#td-336-legacy-training-file-parser-truncates-responses-bom-breaks-jsonl-detection)
  - [TD-337: `DataTransport` Names Downloads by Basename, So Segments and Same-Named Files Collide](#td-337-datatransport-names-downloads-by-basename-so-segments-and-same-named-files-collide)
  - [TD-338: `DataTransport` Reuses Same-Size Stale Files Without Checking the Checksum](#td-338-datatransport-reuses-same-size-stale-files-without-checking-the-checksum)
  - [TD-339: FTPS Downloads Don't Verify the Registry's Certificate](#td-339-ftps-downloads-dont-verify-the-registrys-certificate)
  - [TD-340: `DataTransport` Robustness and Header Hygiene](#td-340-datatransport-robustness-and-header-hygiene)
  - [TD-341: World-Model and Hippocampal Gates Never Train in Production](#td-341-world-model-and-hippocampal-gates-never-train-in-production)
  - [TD-342: KV-Cached Generation and the GPU Path Skip the Gated Paths](#td-342-kv-cached-generation-and-the-gpu-path-skip-the-gated-paths)
  - [TD-343: Hippocampal Memory Is Updated on Every Forward Pass](#td-343-hippocampal-memory-is-updated-on-every-forward-pass)
  - [TD-344: `DecoderBlock` Dropout Is Stored but Never Applied](#td-344-decoderblock-dropout-is-stored-but-never-applied)
  - [TD-345: `DecoderBlock` Persistence Gaps for Gated State](#td-345-decoderblock-persistence-gaps-for-gated-state)
  - [TD-346: `LLMDecoder::update_weights()` Never Updates the Final Norm](#td-346-llmdecoderupdate_weights-never-updates-the-final-norm)
  - [TD-347: Cached and Full Decoder Forwards Diverge Past `max_seq_length`](#td-347-cached-and-full-decoder-forwards-diverge-past-max_seq_length)
  - [TD-348: `LLMDecoder` Dead Code and Robustness Gaps](#td-348-llmdecoder-dead-code-and-robustness-gaps)
  - [TD-349: RAG Indexes Whole Files but Embeds Only Their First `max_seq_length` Tokens](#td-349-rag-indexes-whole-files-but-embeds-only-their-first-max_seq_length-tokens)
  - [TD-350: `DocumentStore::retrieve()` Isn't Safe to Call Concurrently](#td-350-documentstoreretrieve-isnt-safe-to-call-concurrently)
  - [TD-351: `DocumentStore` Embedding and Housekeeping Gaps](#td-351-documentstore-embedding-and-housekeeping-gaps)
  - [TD-356: `EncoderBlock` Gradient Norm and Clipping Ignore `LayerNorm`; Header Describes Post-LN](#td-356-encoderblock-gradient-norm-and-clipping-ignore-layernorm-header-describes-post-ln)
  - [TD-357: `EncoderBlock` Learning-Rate Handling, Unchecked Loads and Console Output](#td-357-encoderblock-learning-rate-handling-unchecked-loads-and-console-output)
  - [TD-358: GPU Training Feeds Unshifted Targets, Teaching the Model to Copy Instead of Predict](#td-358-gpu-training-feeds-unshifted-targets-teaching-the-model-to-copy-instead-of-predict)
  - [TD-359: `generate_response()` and `gpu_generate_response()` Ignore `max_length`](#td-359-generate_response-and-gpu_generate_response-ignore-max_length)
  - [TD-360: `EncoderDecoderModel` No-Optimizer Path Never Updates the Encoder](#td-360-encoderdecodermodel-no-optimizer-path-never-updates-the-encoder)
  - [TD-361: Training Targets Get a Double `BOS`](#td-361-training-targets-get-a-double-bos)
  - [TD-362: CPU and GPU Generation Paths Behave Differently](#td-362-cpu-and-gpu-generation-paths-behave-differently)
  - [TD-363: `EncoderDecoderModel` Clone, Load and World-Model Input Gaps](#td-363-encoderdecodermodel-clone-load-and-world-model-input-gaps)
  - [TD-364: `LLMEncoder::encode()` Input Doesn't Match the Model's Encoder Input](#td-364-llmencoderencode-input-doesnt-match-the-models-encoder-input)
  - [TD-365: Encoder Input Is Never Truncated to `max_seq_length` When Serving](#td-365-encoder-input-is-never-truncated-to-max_seq_length-when-serving)
  - [TD-366: Serving Encoders Cache Activations for Backward; Dead State and Duplicate API](#td-366-serving-encoders-cache-activations-for-backward-dead-state-and-duplicate-api)
  - [TD-367: `LLMEncoder` Console Output and Unchecked Weight I/O](#td-367-llmencoder-console-output-and-unchecked-weight-io)
  - [TD-368: Weight Initialization Can't Be Seeded or Reproduced](#td-368-weight-initialization-cant-be-seeded-or-reproduced)
  - [TD-369: `FeedForward::load_weights()` Loads Truncated Files as a Silent Mix of Old and New Weights](#td-369-feedforwardload_weights-loads-truncated-files-as-a-silent-mix-of-old-and-new-weights)
  - [TD-370: `FeedForward` Argument Checks, GPU Null Dereference and Console Output](#td-370-feedforward-argument-checks-gpu-null-dereference-and-console-output)
  - [TD-371: Residual-Branch Output Isn't Scaled for Depth](#td-371-residual-branch-output-isnt-scaled-for-depth)
  - [TD-372: FTP Data Connections Are Accepted From Any Host, So Files Can Be Stolen](#td-372-ftp-data-connections-are-accepted-from-any-host-so-files-can-be-stolen)
  - [TD-373: `--ftps` Doesn't Require Encryption](#td-373---ftps-doesnt-require-encryption)
  - [TD-374: FTP Re-Login Leaks Rate-Limit Slots](#td-374-ftp-re-login-leaks-rate-limit-slots)
  - [TD-375: `FtpDataServer` Has No Connection Limits or Timeouts](#td-375-ftpdataserver-has-no-connection-limits-or-timeouts)
  - [TD-376: `FtpDataServer` Slow Shutdown, Unencrypted `FEAT` and Silent Start Failures](#td-376-ftpdataserver-slow-shutdown-unencrypted-feat-and-silent-start-failures)
  - [TD-377: `FtpDataServer` Token Semantics and Setup Gaps](#td-377-ftpdataserver-token-semantics-and-setup-gaps)
  - [TD-378: BLEU Smoothing Inflates Unrelated Answers and Zeroes Correct Short Ones](#td-378-bleu-smoothing-inflates-unrelated-answers-and-zeroes-correct-short-ones)
  - [TD-379: `bleu1`/`bleu2` Are Computed but Never Stored](#td-379-bleu1bleu2-are-computed-but-never-stored)
  - [TD-380: Generation-Quality Scores Aren't Comparable or Well-Sampled](#td-380-generation-quality-scores-arent-comparable-or-well-sampled)
  - [TD-381: `chatbot_api_server` Ignores `GPU_STRATEGY`](#td-381-chatbot_api_server-ignores-gpu_strategy)
  - [TD-382: GPU Work From Any Thread Shares Unsynchronized `GPUManager` State](#td-382-gpu-work-from-any-thread-shares-unsynchronized-gpumanager-state)
  - [TD-383: `GPUManager` Lifecycle Gaps (`set_device`, `cleanup`, `initialize`)](#td-383-gpumanager-lifecycle-gaps-set_device-cleanup-initialize)
  - [TD-384: Shared-Memory Race in GPU Softmax, LayerNorm and Cross-Entropy Kernels (CUDA and SYCL)](#td-384-shared-memory-race-in-gpu-softmax-layernorm-and-cross-entropy-kernels-cuda-and-sycl)
  - [TD-385: `matrix_sum_gpu` Allocates and Synchronizes on Every Call](#td-385-matrix_sum_gpu-allocates-and-synchronizes-on-every-call)
  - [TD-386: GPU Kernel Input Checks and Dead Batch Functions](#td-386-gpu-kernel-input-checks-and-dead-batch-functions)
  - [TD-387: GPU Generation Fails on Input Longer Than `max_seq_length`](#td-387-gpu-generation-fails-on-input-longer-than-max_seq_length)
  - [TD-388: `GPUKVCache` Copies the Whole Cache Every Decode Step](#td-388-gpukvcache-copies-the-whole-cache-every-decode-step)
  - [TD-389: `GPUMatrix` Shape Checks, Uninitialized Memory and Untested GPU KV Caches](#td-389-gpumatrix-shape-checks-uninitialized-memory-and-untested-gpu-kv-caches)
  - [TD-390: SYCL Deferred Frees and Blocking Copies Stall the In-Order Queue](#td-390-sycl-deferred-frees-and-blocking-copies-stall-the-in-order-queue)
  - [TD-391: SYCL Lists the Same GPU Once per Platform and Doesn't Log the Backend](#td-391-sycl-lists-the-same-gpu-once-per-platform-and-doesnt-log-the-backend)
  - [TD-392: SYCL `GPUMemory(0)` Throws Where CUDA Succeeds](#td-392-sycl-gpumemory0-throws-where-cuda-succeeds)
  - [TD-393: SYCL Sub-Group Reduction Is Wrong When the Sub-Group Size Is 8](#td-393-sycl-sub-group-reduction-is-wrong-when-the-sub-group-size-is-8)
  - [TD-394: GPU Kernel Tests Can't Run Without a GPU](#td-394-gpu-kernel-tests-cant-run-without-a-gpu)
  - [TD-395: `GPUMatrix` Is Duplicated Between the CUDA and SYCL Headers](#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers)
  - [TD-396: Hippocampal Memory Eviction Starves New Memories](#td-396-hippocampal-memory-eviction-starves-new-memories)
  - [TD-397: Hippocampal Swap File Grows Without Bound](#td-397-hippocampal-swap-file-grows-without-bound)
  - [TD-398: `HippocampalMemory` Per-Step Copies and Unchecked Saves](#td-398-hippocampalmemory-per-step-copies-and-unchecked-saves)
  - [TD-399: `IMetricsReporter` Buffering and Ordering Contract Is Undocumented](#td-399-imetricsreporter-buffering-and-ordering-contract-is-undocumented)
  - [TD-400: `NullMetricsReporter` Keeps Reporting-Only Work Switched On](#td-400-nullmetricsreporter-keeps-reporting-only-work-switched-on)
  - [TD-401: `incremental_trainer` Flag Parsing Crashes or Silently Misreads Bad Values](#td-401-incremental_trainer-flag-parsing-crashes-or-silently-misreads-bad-values)
  - [TD-402: `--dataset-kind` Is Not Validated](#td-402---dataset-kind-is-not-validated)
  - [TD-403: `trainer_service` Drops `--objective` and `--dataset-kind`](#td-403-trainer_service-drops---objective-and---dataset-kind)
  - [TD-404: `derive_run_id()` Can Collide Across Hosts](#td-404-derive_run_id-can-collide-across-hosts)
  - [TD-405: Each Training Pass Restarts From the Best-Ever Checkpoint, Not the Latest](#td-405-each-training-pass-restarts-from-the-best-ever-checkpoint-not-the-latest)
  - [TD-406: A Paused or Crashed Pass's Checkpoint Is Deleted on the Next Start](#td-406-a-paused-or-crashed-passs-checkpoint-is-deleted-on-the-next-start)
  - [TD-407: Checkpoint Deletion Leaks the Per-Layer Weight Files](#td-407-checkpoint-deletion-leaks-the-per-layer-weight-files)
  - [TD-408: `latest`/`best` Checkpoint Symlinks Always Dangle](#td-408-latestbest-checkpoint-symlinks-always-dangle)
  - [TD-409: Tokenized-Data Cache Key Has No Preprocessing Version](#td-409-tokenized-data-cache-key-has-no-preprocessing-version)
  - [TD-410: `IncrementalTrainer` Session Bookkeeping and Hard-Coded Training Policy](#td-410-incrementaltrainer-session-bookkeeping-and-hard-coded-training-policy)
  - [TD-411: `resume` (Every `trainer_service` Pass) Never Downloads Over FTP](#td-411-resume-every-trainer_service-pass-never-downloads-over-ftp)
  - [TD-412: A Training Pass That Loads Zero Samples Marks Its Files Trained](#td-412-a-training-pass-that-loads-zero-samples-marks-its-files-trained)
  - [TD-413: Standalone `train`/`retrain` Pre-Check Reads the Wrong Pending Pool](#td-413-standalone-trainretrain-pre-check-reads-the-wrong-pending-pool)
  - [TD-414: First-Run Vocabulary Bootstrap Is Unreachable From the CLI](#td-414-first-run-vocabulary-bootstrap-is-unreachable-from-the-cli)
  - [TD-415: LeJEPA Pass Ignores Segment Ranges and Strands Claims After a Crash](#td-415-lejepa-pass-ignores-segment-ranges-and-strands-claims-after-a-crash)
  - [TD-416: `incremental_trainer` Command Hygiene](#td-416-incremental_trainer-command-hygiene)
  - [TD-417: `--integrated-inference` Answers Differ From Normal `/chat`](#td-417---integrated-inference-answers-differ-from-normal-chat)
  - [TD-418: `IntegratedInferenceEngine` Worker Exceptions Terminate the Server; Shutdown Drops Requests](#td-418-integratedinferenceengine-worker-exceptions-terminate-the-server-shutdown-drops-requests)
  - [TD-419: `IntegratedInferenceEngine` Uses the Model Without Its Mutex](#td-419-integratedinferenceengine-uses-the-model-without-its-mutex)
  - [TD-420: `IntegratedInferenceEngine` Claims, Dead Config and Fictional Stats](#td-420-integratedinferenceengine-claims-dead-config-and-fictional-stats)
  - [TD-421: `KVCache::append()` Has No Shape Checks](#td-421-kvcacheappend-has-no-shape-checks)
  - [TD-422: `KVCache` Copies the Whole Cache on Every Append](#td-422-kvcache-copies-the-whole-cache-on-every-append)
  - [TD-423: A Truncated `.lm_head` File Loads as All-`NaN` Logits](#td-423-a-truncated-lm_head-file-loads-as-all-nan-logits)
  - [TD-424: Uncached Generation Projects Every Position Through the LM Head](#td-424-uncached-generation-projects-every-position-through-the-lm-head)
  - [TD-425: `LanguageModelHead` Hygiene](#td-425-languagemodelhead-hygiene)
  - [TD-426: `LayerNorm` Silent Setters, Missing Shape Checks and Unchecked Loads](#td-426-layernorm-silent-setters-missing-shape-checks-and-unchecked-loads)
  - [TD-427: LeJEPA `Predictor` Is Never Saved, So Each Pass Restarts It](#td-427-lejepa-predictor-is-never-saved-so-each-pass-restarts-it)
  - [TD-428: LeJEPA Two-Update Workaround Is Obsolete After TD-195](#td-428-lejepa-two-update-workaround-is-obsolete-after-td-195)
  - [TD-429: `LeJEPAEncoder` Exception Escape, Console Output and Unchecked Loads](#td-429-lejepaencoder-exception-escape-console-output-and-unchecked-loads)
  - [TD-430: `Logger` Lazy Initialization Races and Can Throw From a Log Call](#td-430-logger-lazy-initialization-races-and-can-throw-from-a-log-call)
  - [TD-431: Trainer Processes Share One Rotating Log File and Ignore `LOG_LEVEL`](#td-431-trainer-processes-share-one-rotating-log-file-and-ignore-log_level)
  - [TD-432: `LoRAAdapter::load()` Silently Adopts the File's Shape](#td-432-loraadapterload-silently-adopts-the-files-shape)
  - [TD-433: `LoRA` Initialization, Update and Config Gaps](#td-433-lora-initialization-update-and-config-gaps)
  - [TD-434: `Matrix::operator*` Runs Short Products Single-Threaded and Portable Builds Have No SIMD](#td-434-matrixoperator-runs-short-products-single-threaded-and-portable-builds-have-no-simd)
  - [TD-435: `Matrix` Accessor Cost, `sum()` Accuracy, `randomize()` Race and API Hygiene](#td-435-matrix-accessor-cost-sum-accuracy-randomize-race-and-api-hygiene)
  - [TD-436: SYCL (`icpx`) Builds Get No `-march` or SIMD Flags](#td-436-sycl-icpx-builds-get-no--march-or-simd-flags)
  - [TD-437: SIMD Self-Check Test Never Compiles and No Startup CPU Check](#td-437-simd-self-check-test-never-compiles-and-no-startup-cpu-check)
  - [TD-438: `metrics_api_server` Accepts Invalid Numbers and Exits 0 on Bad Arguments](#td-438-metrics_api_server-accepts-invalid-numbers-and-exits-0-on-bad-arguments)
  - [TD-439: `metrics_api_server` Backend Name and Admin-Store Location Gaps](#td-439-metrics_api_server-backend-name-and-admin-store-location-gaps)
  - [TD-440: Metrics Storage Has No Retention and `max_records_on_disk` Is Never Applied](#td-440-metrics-storage-has-no-retention-and-max_records_on_disk-is-never-applied)
  - [TD-441: `IMetricsDatabase` Contract Gaps: Step Counter, `limit` Order, Unbounded Reads](#td-441-imetricsdatabase-contract-gaps-step-counter-limit-order-unbounded-reads)
  - [TD-442: `MetricsPushClient::end_session()` Drain Can Delay a Pass's Final Save by an Hour](#td-442-metricspushclientend_session-drain-can-delay-a-passs-final-save-by-an-hour)
  - [TD-443: Metrics Push Payloads Send `nan`/`inf`, Round Small Values to 0 and Omit Adaptive Clipping](#td-443-metrics-push-payloads-send-naninf-round-small-values-to-0-and-omit-adaptive-clipping)
  - [TD-444: `MetricsPushClient` Transport Gaps: No HTTPS, a Connection per Event, Silent Failures](#td-444-metricspushclient-transport-gaps-no-https-a-connection-per-event-silent-failures)
  - [TD-445: Finished Metrics Sessions Hold `max_live_sessions` Slots for an Hour](#td-445-finished-metrics-sessions-hold-max_live_sessions-slots-for-an-hour)
  - [TD-446: `MetricsSessionRegistry` Sweep-Interval Changes, Miss Logging and Lock-Held I/O](#td-446-metricssessionregistry-sweep-interval-changes-miss-logging-and-lock-held-io)
  - [TD-447: `MetricsTracker` Printer State Leak, Convergence Off-by-One and Smoothing Gaps](#td-447-metricstracker-printer-state-leak-convergence-off-by-one-and-smoothing-gaps)
  - [TD-448: `mns_cli` Ignores Unknown Flags, Misreads `set-training` Arguments and Aborts on Bad Numbers](#td-448-mns_cli-ignores-unknown-flags-misreads-set-training-arguments-and-aborts-on-bad-numbers)
  - [TD-449: `mns_cli` Request-Building Gaps: Layer Counts, Relink Resets, Encoding](#td-449-mns_cli-request-building-gaps-layer-counts-relink-resets-encoding)
  - [TD-450: `mns_cli register` Takes Architecture Defaults from `config.mns.conf`, Not the Trainer's Config](#td-450-mns_cli-register-takes-architecture-defaults-from-configmnsconf-not-the-trainers-config)
  - [TD-451: `mns_cli` Has No HTTPS and Minor Option Handling Gaps](#td-451-mns_cli-has-no-https-and-minor-option-handling-gaps)
  - [TD-452: `MnsJsonHelpers` Parsing Gaps and a GUI Register Body Without TD-196 Fields](#td-452-mnsjsonhelpers-parsing-gaps-and-a-gui-register-body-without-td-196-fields)
  - [TD-453: Hand-Written JSON Duplicated Across MNS and Metrics Code](#td-453-hand-written-json-duplicated-across-mns-and-metrics-code)
  - [TD-454: MNS Manager GUI's Training → Candidate Workflow Fails Against Server-Allocated Run IDs](#td-454-mns-manager-guis-training--candidate-workflow-fails-against-server-allocated-run-ids)
  - [TD-455: MNS Manager GUI Error Detection, UI-Thread HTTP and Register Defaults](#td-455-mns-manager-gui-error-detection-ui-thread-http-and-register-defaults)
  - [TD-456: `mns_server` Treats a Mistyped `--admin-enabled` as `true` and Doesn't Range-Check `--port`](#td-456-mns_server-treats-a-mistyped---admin-enabled-as-true-and-doesnt-range-check---port)
  - [TD-457: `ModelNameClient` Retries Non-Idempotent Requests After a Timeout](#td-457-modelnameclient-retries-non-idempotent-requests-after-a-timeout)
  - [TD-458: A NaN Training Loss Makes MNS's JSON Unparseable](#td-458-a-nan-training-loss-makes-mnss-json-unparseable)
  - [TD-459: `ModelNameClient` Validation, Unescaping and Slow-Failure Gaps](#td-459-modelnameclient-validation-unescaping-and-slow-failure-gaps)
  - [TD-460: MNS Retirement Leaves Role Mappings and Is Undone by Running Trainers](#td-460-mns-retirement-leaves-role-mappings-and-is-undone-by-running-trainers)
  - [TD-461: MNS Accepts Unbuildable Architectures at Registration](#td-461-mns-accepts-unbuildable-architectures-at-registration)
  - [TD-462: `ModelNameService` Dead Datasets Proxy, Admin-Settings Race and Swallowed Persistence Errors](#td-462-modelnameservice-dead-datasets-proxy-admin-settings-race-and-swallowed-persistence-errors)
  - [TD-463: `mns_server` Runs In-Memory and Reports Healthy When `models.db` Can't Be Opened](#td-463-mns_server-runs-in-memory-and-reports-healthy-when-modelsdb-cant-be-opened)
  - [TD-464: `mns_server` Startup Crash, Early-SIGTERM Hang and Fixed Log Level](#td-464-mns_server-startup-crash-early-sigterm-hang-and-fixed-log-level)
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
| LOW | Open — inference half planned as v2 (October 8, 2026); training half not started | Core Model Architecture | September 14, 2026 | Not estimated (large, multi-session architecture project) |

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

### TD-212: BatchedInferenceEngine Can Fail or Abandon Requests It Already Accepted

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving | October 8, 2026 | 4-6 hours |

Description:
Found alongside TD-211. Three paths let an accepted request end badly through no fault of its own:

1. **One throwing request fails its whole group.** `process_batch()` runs every request's
   generation inside one `try`. If request *k*'s `model_fn` or tokenizer throws, the outer `catch`
   sets that exception on **every** promise in the group, including requests before *k* whose text
   was already generated (and is discarded) and requests after *k* that never ran. This is the same
   blast-radius shape as the empty-response bug fixed in the same function (see its comment block),
   but for real generation errors instead of the stats update.
2. **Shutdown drains only one final group.** `batch_processing_loop()` calls
   `collect_batch_no_wait()` once, which takes at most `max_batch_size` requests. Anything else
   still queued is destroyed with the engine, and its future throws `std::future_error`
   (`broken_promise`). `shutdown()`'s doc comment says it "waits for pending requests to complete".
3. **`submit()` races `shutdown()`.** `submit()` checks `running_` *before* taking
   `queue_mutex_`. A request that passes the check and is enqueued after the worker's final drain is
   never processed and also ends in `broken_promise`.

Not hit by `chatbot_api_server` today (its engine lives for the whole process, and its
`model_fn` rarely throws), but (1) can turn one bad request into N failed HTTP responses under
concurrent load.

Action Items:

- [ ] Wrap each request's `generate_text()` call in its own `try/catch` and fail only that
  request's promise.
- [ ] Make shutdown drain the queue completely (loop `collect_batch_no_wait()` until empty), or
  explicitly fail leftovers with a clear "engine shut down" exception instead of `broken_promise`.
- [ ] Re-check `running_` under `queue_mutex_` in `submit()` so no request can be enqueued after
  the final drain.
- [ ] Add regression tests for all three (none exist today).

Location in code: `src/BatchedInferenceEngine.hpp` (`submit()`, `shutdown()`,
`batch_processing_loop()`, `collect_batch_no_wait()`, `process_batch()`); tagged
`TODO: See TD-212`.

Files to Modify:

- `src/BatchedInferenceEngine.hpp`
- `tests/batchedinferenceengine_test.cpp`

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

### TD-215: Activation Has Misleading Header Comments and Unchecked Inputs

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Core Math / Activation | October 8, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [Activation.md](../reference/source/Activation.md).
`src/Activation.{hpp,cpp}` is tagged `stable`, but:

- **Wrong comment, `softmax_derivative`.** The header says the formula is "For cross-entropy
  loss". It's actually the general softmax vector–Jacobian product, correct for any upstream
  gradient. For softmax + cross-entropy the right gradient is `y − one_hot(target)` directly, and
  chaining `softmax_derivative` after it differentiates through softmax twice. The old API doc's
  examples made exactly that mistake (corrected in the October 7 merge), so the comment has
  already misled once.
- **Wrong comment, `leaky_relu_derivative`.** The header says "alpha if x < 0, else 1", which gives
  1 at `x == 0`. The code uses `x > 0 ? 1 : alpha`, so it returns `alpha` at 0, and the tests
  encode the code's behaviour.
- **Undefined behaviour, `softmax` on an empty row.** The row-max step reads `input(i, 0)`
  unconditionally, which is out of bounds when `cols == 0`. No production caller passes empty rows
  today.
- **No shape check, `softmax_derivative`.** `output` and `grad_output` are assumed to have the same
  shape; a smaller `grad_output` reads out of bounds.

Deliberately **not** filed: the derivatives' mixed input conventions (some take the
pre-activation input, `sigmoid_derivative`/`tanh_derivative` take the output). That's by design,
each being the cheapest form, and it's documented per function in Activation.md.

Action Items:

- [ ] Fix both header comments to match the code.
- [ ] Guard `softmax` against `cols == 0` (return the empty result) and validate shapes in
  `softmax_derivative` (throw `std::invalid_argument`, matching `FeedForward`'s style).
- [ ] Add tests for the two new guards.

Location in code: `src/Activation.hpp` (`softmax_derivative`, `leaky_relu_derivative` doc
comments), `src/Activation.cpp` (`softmax`, `softmax_derivative`); tagged `TODO: See TD-215`.

Files to Modify:

- `src/Activation.{hpp,cpp}`
- `tests/activation_test.cpp`
- `docs/development/reference/source/Activation.md` (§11)

---

### TD-216: Softmax and GELU Math Is Duplicated Across CPU and GPU Code

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Core Math / Activation | October 8, 2026 | 4-6 hours |

Description:
Found alongside TD-215. `Activation` is meant to be the reference implementation, but the same
math is re-typed in many places, so a change in one place (a different masking convention,
temperature, exact-`erf` GELU) silently leaves the others behind. For GELU specifically, that
means CPU and GPU training would quietly diverge, and checkpoints trained on one path would behave
differently on the other.

Independent copies today:

- **Softmax forward (CPU), max-subtracted, inline:**
  `MultiHeadAttention::forward_parallel()` (`src/MultiHeadAttention.cpp`, the main self-attention
  path), `TextGenerator::softmax()` (`src/TextGenerator.cpp`), two loss/gradient loops in
  `src/EncoderDecoderModel.cpp`, and two in `src/RLHFTrainer.cpp`.
- **Softmax backward (CPU), inline copies of `softmax_derivative`'s formula:**
  `MultiHeadAttention::backward()` and `CrossAttention::backward()`.
- **GELU constants (`0.044715`, `√(2/π) = 0.7978845608`), hardcoded literals:** the forward
  activation and `gelu_backward` kernels in both `src/gpu/MatrixGPU.cu` (CUDA) and
  `src/gpu/sycl/MatrixGPU_SYCL.cpp` (SYCL), besides `Activation`'s own private constants.

The CPU copies exist for a reason: every `Activation` call allocates a new `Matrix`, and the hot
paths avoid that by working in place or on `std::vector<float>`. So the fix is a shared,
allocation-free primitive, not routing everything through the current API.

Action Items:

- [ ] Add an allocation-free row primitive (e.g. `Activation::softmax_row_inplace(float*, int)` and
  a matching backward helper) and switch the CPU copies above to it. Verify with the existing
  attention gradient checks and `chatbotapiTests`/`textgeneratorTests`.
- [ ] Move the GELU constants into a small header usable from host C++, CUDA, and SYCL device code
  (plain `constexpr float`s) and include it in all three places.
- [ ] Add a CPU-vs-GPU GELU agreement test to the existing GPU test targets (runs only where a GPU
  backend is built).

Location in code: every site listed above, tagged `TODO: See TD-216`.

Files to Modify:

- `src/Activation.{hpp,cpp}`
- `src/MultiHeadAttention.cpp`, `src/CrossAttention.cpp`
- `src/TextGenerator.cpp`, `src/EncoderDecoderModel.cpp`, `src/RLHFTrainer.cpp`
- `src/gpu/MatrixGPU.cu`, `src/gpu/sycl/MatrixGPU_SYCL.cpp`
- new shared constants header under `src/` or `src/gpu/`

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
| **HIGH** | Open | NLP / Tokenizer | October 8, 2026 | 1-2 hours (fix) + audit of existing vocab files |

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

Action Items:

- [ ] Key the pair counts by `std::pair<std::string, std::string>` (with a hash) instead of a
  string separator.
- [ ] Regression tests: pipe-only corpus learns merges; no repeated merge rules; merge count
  reaches the target when pairs remain.
- [ ] Audit existing vocab files (`vocab.txt`, every `<checkpoint>.vocab` on ai-machine): compare
  the `BPE_MERGES` count against the intended target and look for repeated rules. Rebuild and
  retrain any affected vocab.

Location in code: `src/BPETokenizer.cpp` (`get_most_frequent_pair()`); tagged `TODO: See TD-220`.

Files to Modify:

- `src/BPETokenizer.cpp`
- `tests/tokenizer_test.cpp`

---

### TD-221: `BPETokenizer::save_vocab()` Fails Silently, So Checkpoints Can Lack Their Vocab

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | NLP / Tokenizer | October 8, 2026 | 1-2 hours |

Description:
Found alongside TD-220 and confirmed by running the real code. If `save_vocab()` can't open its
file it prints to `std::cerr` and **returns normally**; write errors aren't checked at all. Saving
to an unwritable path raised no exception.

`EncoderDecoderModel::save()` calls it for every checkpoint (`<checkpoint>.vocab`), so a bad path
or full disk produces a checkpoint with weights but no vocab, and nobody finds out until
`load()` throws `VocabularyFileError`, possibly long after the run that wrote it.
`IncrementalTrainer`'s vocab bootstrap and `vocab_builder` have the same exposure (a
"successful" build that wrote nothing).

Action Items:

- [ ] Throw `VocabularyFileError` when the file can't be opened, and check the stream state after
  writing (and after `close()`).
- [ ] Write to a temporary file and rename it into place, so a failed save never leaves a
  truncated vocab where a good one used to be.
- [ ] Make sure `EncoderDecoderModel::save()`'s callers (checkpointing in `IncrementalTrainer`,
  `ChatbotTrainer`) surface the error rather than logging success.
- [ ] Tests: unwritable path throws; failed save leaves an existing file intact.

Location in code: `src/BPETokenizer.cpp` (`save_vocab()`), `src/EncoderDecoderModel.cpp`
(`save()`); tagged `TODO: See TD-221`.

Files to Modify:

- `src/BPETokenizer.cpp`
- `src/EncoderDecoderModel.cpp`
- `tests/tokenizer_test.cpp`, `tests/tokenizer_error_handling_test.cpp`

---

### TD-222: `apply_bpe()`'s Thread-Local Cache Is Never Invalidated

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | NLP / Tokenizer | October 8, 2026 | 2-4 hours |

Description:
Found alongside TD-220 and confirmed by running the real code. `apply_bpe()` memoizes results in a
`thread_local` map keyed by `(tokenizer address, word)`. The cache was added for preprocessing
speed, and its comment assumes merges never change while it's in use. But it's never invalidated:

- **After `load_vocab()`** on an instance that has already encoded text, words encode with the
  **old** merges. Verified: after reloading to a tiny vocab, `"abab"` still encoded to the old
  single token instead of the correct four.
- **Address reuse:** a tokenizer constructed at the same address as a destroyed one, on the same
  thread, inherits its cached results. Verified with placement-new.
- **Never freed:** entries for destroyed tokenizers stay for the thread's lifetime.
  `measure_fertility()` creates a temporary tokenizer on every call, adding another entry each
  time.

Production mostly dodges this (each `trainer_service` pass is a fresh process; servers load once),
but `--pipeline-inference`'s reload of the encoder's tokenizer and any in-process
reload-and-retrain path are exposed. The cache comment also refers to a nonexistent
`train_bpe()` (the real mutators are `build_vocab()`/`build_bpe_merges()`/`load_vocab()`).

Action Items:

- [ ] Give each tokenizer a unique, never-reused instance ID plus a generation counter bumped by
  every mutator, and key the cache on `(instance_id, generation)`, dropping stale entries on
  mismatch. Alternatively, move the cache into the instance behind a lock-free or sharded
  structure.
- [ ] Bound the cache size, and free a destroyed tokenizer's entries.
- [ ] Fix the comment (`train_bpe()` → the real mutators).
- [ ] Tests: reload-then-encode matches a fresh tokenizer; same-address reconstruction doesn't
  inherit results.

Location in code: `src/BPETokenizer.cpp` (`apply_bpe()`, `load_vocab()`, `measure_fertility()`);
tagged `TODO: See TD-222`.

Files to Modify:

- `src/BPETokenizer.{hpp,cpp}`
- `tests/tokenizer_test.cpp`

---

### TD-223: Tokenizer Mutators Can Leave It Gutted or With Duplicate IDs

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | NLP / Tokenizer | October 8, 2026 | 2-4 hours |

Description:
Found alongside TD-220 and confirmed by running the real code:

- **`load_vocab()` isn't exception-safe.** It clears every table before parsing, so a malformed
  file leaves the tokenizer gutted (a trained 12-token tokenizer had 1 token after a failed load).
  `chatbot_api_server --pipeline-inference` catches a failed `enable_pipeline_inference()` (which
  reloads the encoder's private tokenizer), logs a warning and keeps serving, potentially with that
  tokenizer broken.
- **A second `build_vocab()` corrupts IDs.** It restarts base IDs at 4 and appends to the existing
  merges. Two calls produced 3 IDs shared by two different tokens. `tokenizer_test`'s
  `RepeatedBuildVocab` says "Second build should replace vocabulary" but only checks that the size
  changed, so it passes.

No production caller reuses an instance for a second build today; every one constructs a fresh
tokenizer.

Action Items:

- [ ] `load_vocab()`: parse into local tables, validate, then swap them in, so a failure leaves the
  previous state untouched (strong exception guarantee).
- [ ] `build_vocab()`: reset to the four specials first (true replacement, as the test intends), or
  throw if the tokenizer is already trained.
- [ ] Fix `RepeatedBuildVocab` to check ID uniqueness and that the first build's tokens are gone;
  add a failed-load-preserves-state test.

Location in code: `src/BPETokenizer.cpp` (`load_vocab()`, `build_vocab()`),
`src/ChatbotAPIServer.cpp` (pipeline-inference catch), `tests/tokenizer_test.cpp`
(`RepeatedBuildVocab`); tagged `TODO: See TD-223`.

Files to Modify:

- `src/BPETokenizer.cpp`
- `tests/tokenizer_test.cpp`, `tests/tokenizer_error_handling_test.cpp`

---

### TD-224: Tokenizer Normalization Is Lossy: No Capitals, Newlines, or Tabs

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open — owner decision needed | NLP / Tokenizer | October 8, 2026 | Not estimated (requires a new vocab format version and retraining) |

Description:
Found alongside TD-220 and confirmed by running the real code. `pre_tokenize()` lowercases every
input (all bytes in ASCII mode, ASCII bytes in Unicode mode) and replaces every whitespace run with
a single space. `"Hello World\nSecond\tline"` round-trips as `"hello world second line"`.
Consequences:

- No model trained here can read case, line structure or indentation, or produce them: no
  capitalized sentences, paragraphs, lists or code layout in chatbot output.
- The vocab file's `\n`/`\t`/`\r` escaping can never fire for real tokens.

This is a design property, not an accident in one line, and changing it invalidates every existing
vocab and checkpoint.

Action Items:

- [ ] Owner decision: keep it (and document it as a product limitation), or move to
  case-preserving, whitespace-preserving tokenization (GPT-2-style, typically with byte-level
  fallback).
- [ ] If changed: bump the vocab file format (`# BPE Tokenizer Vocabulary v2.0` plus a
  normalization field), keep loading v1 files with the old normalization so existing checkpoints
  stay usable, and plan a retrain.

Location in code: `src/BPETokenizer.cpp` (`pre_tokenize()`); tagged `TODO: See TD-224`.

Files to Modify (if changed): `src/BPETokenizer.{hpp,cpp}`, `tests/tokenizer_test.cpp`,
`docs/development/reference/source/BPETokenizer.md`, plus a retraining plan.

---

### TD-225: BPETokenizer Logging and API Hygiene

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | NLP / Tokenizer | October 8, 2026 | 4-6 hours |

Description:
Found alongside TD-220. Smaller issues in the `stable`-tagged tokenizer:

- **`std::cout`/`std::cerr` in library code** throughout (`build_vocab()`, `build_bpe_merges()`,
  `pre_tokenize()`, `decode()`, `save_vocab()`, `load_vocab()`, `print_vocab_stats()`), against
  the project's logging rule. `pre_tokenize()` prints progress for any input producing ≥ 1,000
  matches, including long chat inputs on the server and interleaved lines from parallel
  preprocessing threads.
- **`decode()` skips unknown IDs** with only a `cerr` warning, so out-of-range model output is
  silently truncated.
- **`encode()`/`decode()` aren't `const`** though they mutate nothing. That forces
  `measure_fertility()` to copy the whole tokenizer on every call, and blocks encoding through a
  `const BPETokenizer&`.
- **`get_top_tokens()` is misnamed:** it returns the lowest IDs, not the most frequent tokens (the
  tokenizer stores no frequencies). `vocab_builder` prints its output as if it were frequency.
- **Unportable rebuilds:** base-unit IDs and pair-count ties follow `unordered_map` iteration
  order, so the same corpus can produce different vocabs on different standard libraries.
- **Unchecked special IDs:** `load_vocab()` accepts any special-token IDs a file declares, while
  `BatchProcessor`, `TextGenerator` and `Dataset` use the `adai::SpecialTokenIDs` constants
  directly. Nothing checks that they agree.
- **Lenient UTF-8:** `is_valid_utf8()` accepts overlong encodings, surrogates and lead bytes up to
  `0xF7`.
- **Stale pointer:** `DataFetcher.cpp`'s TD-006 TODO says FIM special tokens are "defined in
  BPETokenizer.cpp"; they aren't (the tokenizer has only four specials).

Action Items:

- [ ] Replace console output with `adai::Logger` (progress at `debug`, or a callback).
- [ ] Count unknown IDs in `decode()` and log once per call at `warn`, or offer a strict mode that
  throws.
- [ ] Mark `encode()`, `tokenize()`, `pre_tokenize()`, `apply_bpe()`, `decode()` `const` (the
  cache is `thread_local`, so no `mutable` is needed), and drop `measure_fertility()`'s copy.
- [ ] Rename `get_top_tokens()` (e.g. `get_tokens_by_id()`) or record merge frequencies during
  training to make it true; update `vocab_builder`.
- [ ] Make base-ID assignment and tie-breaking deterministic (sort by frequency, then by token).
- [ ] Warn (or throw) in `load_vocab()` when the special IDs differ from `SpecialTokenIDs`.
- [ ] Tighten `is_valid_utf8()` to RFC 3629.
- [ ] Fix `DataFetcher.cpp`'s pointer to say the FIM tokens still need adding (TD-006).

Location in code: `src/BPETokenizer.{hpp,cpp}`, `src/DataFetcher.cpp`; tagged `TODO: See TD-225`.

Files to Modify:

- `src/BPETokenizer.{hpp,cpp}`, `src/VocabBuilder.cpp`, `src/DataFetcher.cpp`
- `tests/tokenizer_test.cpp`, `tests/vocabbuilder_test.cpp`

---

### TD-226: `/chat/batch` Rejects Any Request Whose Messages Contain `]`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Inference / Serving / API | October 8, 2026 | 2-4 hours (switch request parsing to nlohmann/json; also resolves TD-228) |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). `ChatbotAPI::parse_json_array()` finds the array's `[` and then the **first `]` anywhere after
it**, even inside a string value. Verified by running the real function:
`{"messages":["see [1] here","second"]}` returns **0 items**, so `POST /chat/batch` and
`POST /chat/batch-session` answer "Missing or empty 'messages' array" for a valid request.
Brackets in chat text (citations, code, markdown lists) are common, so this rejects ordinary batch
traffic. The vendored nlohmann/json (`external/nlohmann/json.hpp`, already used by
`ModelSerializer.cpp`) parses the same body correctly (2 items).

Action Items:

- [ ] Replace `parse_json_array()` (and `parse_json_string()`, TD-228) with nlohmann/json parsing in every handler; reject malformed bodies with HTTP 400.
- [ ] Regression test: messages containing `[`, `]`, `,` and escaped quotes round-trip through `/chat/batch`.

Location in code: `src/ChatbotAPI.cpp` (`parse_json_array()`); tagged `TODO: See TD-226`.

Files to Modify:

- `src/ChatbotAPI.{hpp,cpp}`
- `tests/chatbotapi_test.cpp`

---

### TD-227: ChatbotAPI Session Pointers Are Used Outside the Session Lock

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Inference / Serving / API | October 8, 2026 | 4-6 hours |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). `get_or_create_session()` returns a raw `Session*` after releasing `sessions_mutex_`.
`handle_chat_session()`, `handle_import_session()` and `generate_batch_session_responses()` then
read and mutate `session->context` (a `ConversationContext`, which has no internal locking) with no
lock held, across a generation that can take seconds. Consequences:

- two concurrent requests on the same `session_id` race on the same `ConversationContext`
  (undefined behaviour, corrupted history);
- `POST /clear-session` can clear the history mid-request;
- `cleanup_expired_sessions()` (run by `GET /health`) can **erase** the session mid-request, a
  use-after-free. That's unlikely with the default 30-minute timeout, since `last_access` is
  refreshed when a request starts, but nothing prevents it.

Found by reading the code paths.

Action Items:

- [ ] Hold sessions as `std::shared_ptr<Session>` so an in-flight request keeps its session alive after erasure.
- [ ] Add a per-session mutex held for the read-modify-generate-append sequence (serializing turns within one conversation, which is also the correct conversational semantics).
- [ ] Tests: concurrent same-session requests under TSan; cleanup during an in-flight request.

Location in code: `src/ChatbotAPI.{hpp,cpp}` (`Session`, `get_or_create_session()`, `handle_chat_session()`, `handle_import_session()`, `generate_batch_session_responses()`); tagged `TODO: See TD-227`.

Files to Modify:

- `src/ChatbotAPI.{hpp,cpp}`
- `tests/chatbotapi_test.cpp`

---

### TD-228: `ChatbotAPI::parse_json_string()` Mis-Parses Common JSON

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / API | October 8, 2026 | 1-2 hours (no extra work if done with TD-226) |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). The hand-written key lookup in `parse_json_string()` returns wrong values for valid JSON. Verified
by running the real function:

| Input | Returned | Correct |
|---|---|---|
| `{"session_id":123,"message":"hi"}` → `session_id` | `"message"` (the next key's name) | reject / `""` |
| `{"message":"path C:\\","session_id":"abc"}` → `message` | `path C:\",` (runs past the closing quote) | `path C:\` |
| `{"message":"caf\u00e9"}` | `caf\u00e9` (escape left literal) | `café` |

It also can't distinguish a missing key from an empty string. Wrong session IDs and garbled
messages reach the model and the session map. nlohmann/json handles all of these correctly.

Action Items:

- [ ] Covered by TD-226's switch to nlohmann/json; add these three inputs as regression tests.

Location in code: `src/ChatbotAPI.cpp` (`parse_json_string()`); tagged `TODO: See TD-228`.

Files to Modify:

- `src/ChatbotAPI.cpp`
- `tests/chatbotapi_test.cpp`

---

### TD-229: ChatbotAPI Responses Can Be Invalid JSON (Unescaped Control Characters)

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / API | October 8, 2026 | 1 hour |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). `escape_json_string()` (the single escaping function since TD-063) escapes only `"`, `\`, `\n`,
`\r` and `\t`. Other characters in U+0000–U+001F pass through raw. Verified: `"a\x01b\bc"` is
emitted with bytes `01` and `08`, which strict JSON parsers reject. Model output, exception messages
and imported session data can all carry such bytes. nlohmann/json's `dump()` escapes them
correctly (`"a\u0001b\bc"`).

Action Items:

- [ ] Escape every control character (`\b`, `\f`, and `\u00XX` for the rest), or build responses with nlohmann/json.
- [ ] Test that every response parses with a strict JSON parser.

Location in code: `src/ChatbotAPI.cpp` (`escape_json_string()`); tagged `TODO: See TD-229`.

Files to Modify:

- `src/ChatbotAPI.cpp`
- `tests/chatbotapi_test.cpp`

---

### TD-230: RAG Mode Serializes Every ChatbotAPI Request Behind `config_mutex_`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / API | October 8, 2026 | 1 hour |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). In `generate_response()`, the RAG path is
`{ lock_guard(config_mutex_); if (rag_engine_) return rag_engine_->generate(input); }`, so
`config_mutex_` is held for the **whole RAG generation**. Every handler takes that same mutex to copy
`default_config_`, and `set_generation_config()` (including on `SIGHUP` reload) needs it too. So with
RAG enabled, every chat, session and batch request waits at its config copy behind whichever
generation is running. Found by reading the code.

**Note (October 11, 2026):** releasing the lock alone would expose `DocumentStore::retrieve()` to
concurrent calls, which race inside the RAG encoder; see [TD-350](#td-350-documentstoreretrieve-isnt-safe-to-call-concurrently), which must land first or together.

Action Items:

- [ ] Copy the `rag_engine_` `shared_ptr` under the lock, release it, then generate.
- [ ] Test: a slow fake RAG engine doesn't block a concurrent `set_generation_config()`.

Location in code: `src/ChatbotAPI.cpp` (`generate_response()`); tagged `TODO: See TD-230`.

Files to Modify:

- `src/ChatbotAPI.cpp`
- `tests/chatbotapi_test.cpp`

---

### TD-231: ChatbotAPI Sessions Only Expire When `GET /health` Is Called

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / API | October 8, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). `cleanup_expired_sessions()` is called only from `handle_health()`. `start()` carries a comment
saying periodic cleanup should be a background thread, but none exists. On a server nobody
health-checks, idle sessions are never removed, so memory grows for the process lifetime. Found by
reading the code.

Action Items:

- [ ] Run expiry on a background thread (or opportunistically in `get_or_create_session()`), stopped cleanly by `stop()`/the destructor.
- [ ] Test: sessions expire without any `/health` call.

Location in code: `src/ChatbotAPI.cpp` (`start()`, `cleanup_expired_sessions()`); tagged `TODO: See TD-231`.

Files to Modify:

- `src/ChatbotAPI.{hpp,cpp}`
- `tests/chatbotapi_test.cpp`

---

### TD-232: ChatbotAPI Accepts Unlimited Client-Chosen Session IDs

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / API | October 8, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). `get_or_create_session()` creates a session under any unknown non-empty `session_id` a client
sends, with no limit on ID length, format, or the number of sessions. One request per new ID adds a
session (with its `ConversationContext`), so any client can grow server memory without bound; with
TD-231, nothing reclaims it unless `/health` is polled. The ChatbotCLI relies on choosing its own
IDs, so the fix must keep that working. Found by reading the code.

Action Items:

- [ ] Cap total sessions (evict least-recently-used) and validate client IDs (length and character set).
- [ ] Decide whether client-chosen IDs stay allowed or move to server-issued IDs only (needs a ChatbotCLI change).
- [ ] Tests: cap enforcement and ID validation.

Location in code: `src/ChatbotAPI.cpp` (`get_or_create_session()`); tagged `TODO: See TD-232`.

Files to Modify:

- `src/ChatbotAPI.{hpp,cpp}`
- `src/ChatbotCLI.*` (if IDs become server-issued)
- `tests/chatbotapi_test.cpp`

---

### TD-233: ChatbotAPI Session IDs Come From a Non-Cryptographic Generator

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / API | October 8, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). `create_session_id()` draws 32 hex digits from a `static std::mt19937` seeded once from
`std::random_device`. `mt19937` is predictable once enough output is observed, yet a session ID is
the only credential for a conversation: `POST /chat/session/export` returns the full history to
anyone who presents it. Found by reading the code.

Action Items:

- [ ] Generate IDs from a CSPRNG (e.g. `getrandom()` or `/dev/urandom`).
- [ ] Consider requiring an auth token for export/import if the server is reachable beyond localhost.

Location in code: `src/ChatbotAPI.cpp` (`create_session_id()`); tagged `TODO: See TD-233`.

Files to Modify:

- `src/ChatbotAPI.cpp`

---

### TD-234: ChatbotAPI Batch Endpoints Bypass the Inference Dispatcher

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / API | October 8, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). `generate_batch_responses()` (behind `/chat/batch` and `/chat/batch-session`) runs its own
`TextGenerator` loop over `model_->forward()` instead of calling `generate_response()`. So batch
requests never use the GPU-resident decode path (TD-033), RAG, speculative decoding or the TD-038
engines: on a GPU host they run on the slow per-matmul CPU fallback, and a server configured for
RAG answers batch requests without retrieval. Found by reading the code.

Action Items:

- [ ] Generate each batch input through `generate_response()` (keeping input order), so every mode applies consistently.
- [ ] Test: with a mode enabled (e.g. a fake RAG engine), `/chat/batch` goes through it.

Location in code: `src/ChatbotAPI.cpp` (`generate_batch_responses()`); tagged `TODO: See TD-234`.

Files to Modify:

- `src/ChatbotAPI.cpp`
- `tests/chatbotapi_test.cpp`

---

### TD-235: ChatbotAPI Batch Endpoints Have No Size Limit

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / API | October 8, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). `/chat/batch` and `/chat/batch-session` accept any number of messages and generate them one after
another on the request's HTTP thread. A single large request occupies that thread (and the model)
for as long as all its generations take. Found by reading the code.

Action Items:

- [ ] Add a configurable maximum batch size (reject larger requests with HTTP 413/400).
- [ ] Document the limit in `rest-api.md`.

Location in code: `src/ChatbotAPI.cpp` (`handle_batch_chat()`, `handle_batch_chat_session()`); tagged `TODO: See TD-235`.

Files to Modify:

- `src/ChatbotAPI.cpp`
- `src/Config.{hpp,cpp}`
- `docs/development/api/rest-api.md`

---

### TD-236: ChatbotAPI Returns HTTP 200 for Validation Errors

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Serving / API | October 8, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). Handlers that *return* an error, such as "Missing 'message' field", "Session not found" or a failed
import, send `{"success":false,...}` with status **200**. Only thrown exceptions map to 400/500.
Clients and monitoring that key off the status code see failures as successes. Found by reading the
code.

Action Items:

- [ ] Return 400 for malformed/missing input and 404 for unknown sessions; keep the JSON body shape.
- [ ] Update `rest-api.md` and check the CLI and Android clients handle the new codes.

Location in code: `src/ChatbotAPI.cpp` (route lambdas and handlers); tagged `TODO: See TD-236`.

Files to Modify:

- `src/ChatbotAPI.cpp`
- `tests/chatbotapi_test.cpp`
- `docs/development/api/rest-api.md`

---

### TD-237: `ChatbotAPI::running_` Is a Non-Atomic Flag Written From Two Threads

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Serving / API | October 8, 2026 | 1 hour |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). `chatbot_api_server` calls `start()` on a server thread (which sets and clears `running_`) and
`stop()` from the main thread on `SIGINT`/`SIGTERM` (which reads and clears it). `running_` is a
plain `bool`, so that's a data race (undefined behaviour, though benign on x86 in practice). Found by
reading the code.

Action Items:

- [ ] Make `running_` `std::atomic<bool>`.

Location in code: `src/ChatbotAPI.hpp` (`running_`), `src/ChatbotAPI.cpp` (`start()`, `stop()`); tagged `TODO: See TD-237`.

Files to Modify:

- `src/ChatbotAPI.{hpp,cpp}`

---

### TD-238: ChatbotAPI Returns Internal Exception Text to Clients

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Serving / API | October 8, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). Every route lambda passes `e.what()` straight into the response, and `generate_response()` wraps
failures as "Generation failed: <internal message>". Internal details (file paths, tokenizer and
model errors) reach any client. Found by reading the code.

Action Items:

- [ ] Log the full exception server-side via `adai::Logger`; return a generic message plus a request/correlation ID.

Location in code: `src/ChatbotAPI.cpp` (route lambdas, `generate_response()`); tagged `TODO: See TD-238`.

Files to Modify:

- `src/ChatbotAPI.cpp`
- `tests/chatbotapi_test.cpp`

---

### TD-239: ChatbotAPI Logging and Dead Code Cleanup

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Serving / API | October 8, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [ChatbotAPI.md](../reference/source/ChatbotAPI.md). Small hygiene issues found while tracing the class:

- `std::cout`/`std::cerr` in library code (constructor, `start()`, `stop()`,
  `cleanup_expired_sessions()`), against the logging rule.
- `ChatbotAPI::BatchRequest` is declared but never used.
- `handle_chat()`'s comment says it gets generation config "from request or use defaults", but
  requests can't set it (matching `rest-api.md`, which documents server flags only).
- `start()`'s comment about periodic cleanup describes something that doesn't exist (TD-231).

Action Items:

- [ ] Switch to `adai::Logger`; remove `BatchRequest`; fix both comments.

Location in code: `src/ChatbotAPI.{hpp,cpp}`; tagged `TODO: See TD-239`.

Files to Modify:

- `src/ChatbotAPI.{hpp,cpp}`
---

### TD-240: `chatbot_api_server` Silently Accepts Non-Numeric and Partial Numbers

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / CLI | October 8, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [ChatbotApiServerArgs.md](../reference/source/ChatbotApiServerArgs.md), and confirmed by running the real
parser. `apply_chatbot_api_server_args()` parses every numeric flag with `std::atoi`/`std::atof`,
which return 0 for non-numbers and stop at the first non-digit, and nothing reports either case:

| Input | Applied |
|---|---|
| `--port abc` | `port = 0`, which the OS treats as "any free port", so the server starts somewhere unannounced |
| `--port 8080x` | `port = 8080` |
| `--temperature hot` | `temperature = 0`, i.e. greedy decoding |
| `--timeout abc` / `--max-gen-len abc` / `--batch-timeout-ms abc` | 0 |

The same applies to every architecture flag and `--speculative-candidates`.

Action Items:

- [ ] Parse with full-consumption checks (`std::from_chars` or `std::stoi`/`std::stof` plus a check that the whole string was consumed) and return `error = true` with a message naming the flag and value.
- [ ] Tests: non-numeric and trailing-garbage values for each numeric flag are errors.

Location in code: `src/ChatbotApiServerArgs.cpp` (`apply_chatbot_api_server_args()`); tagged `TODO: See TD-240`.

Files to Modify:

- `src/ChatbotApiServerArgs.cpp`
- `tests/chatbot_api_server_args_test.cpp`

---

### TD-241: `chatbot_api_server` Never Validates Its Configuration at Startup

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / CLI | October 8, 2026 | 1-2 hours |

Description:
Found alongside TD-240. `ConfigLoader::validate()` (`src/Config.cpp`, port 1–65535,
`session_timeout >= 1`, log level and size limits, `d_model` range and more) is only called from
`ConfigLoader::reload()`, the `SIGHUP` path. At startup `ConfigLoader::load()` doesn't call it, and
`validate_chatbot_api_server_config()` checks only that `vocab_path` is set. So values from the
CLI, env or file that `validate()` would reject are accepted at startup, and then the same file is
**rejected on the next `SIGHUP`** ("keeping current configuration"). Verified with the real
parser: `--timeout -5` is accepted, after which every session counts as expired at the next
`/health`.

`validate()` also checks none of the generation settings: `strategy` (an unknown value such as
`--strategy bogus` is accepted and silently becomes nucleus sampling in `ChatbotAPI`),
`temperature`, `top_p`, `max_gen_length`.

Action Items:

- [ ] Call `ConfigLoader::validate()` from `validate_chatbot_api_server_config()` (after CLI flags are applied) and fail startup on errors.
- [ ] Extend `validate()` with generation settings: `strategy` in {greedy, beam, temperature, top_k, nucleus}, `temperature >= 0`, `0 < top_p <= 1`, `max_gen_length >= 1`.
- [ ] Tests: invalid port/timeout/strategy fail validation; the default config passes.

Location in code: `src/ChatbotApiServerArgs.cpp` (`validate_chatbot_api_server_config()`), `src/Config.cpp` (`ConfigLoader::validate()`); tagged `TODO: See TD-241`.

Files to Modify:

- `src/ChatbotApiServerArgs.cpp`
- `src/Config.cpp`
- `tests/chatbot_api_server_args_test.cpp`
- `tests/config_test.cpp`

---

### TD-242: `SIGHUP` Reload Discards `chatbot_api_server`'s CLI Generation Overrides

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / CLI | October 8, 2026 | 2-4 hours |

Description:
Found alongside TD-240. `ConfigLoader::reload()` rebuilds the config from the file and env only,
and `ChatbotAPIServer.cpp`'s main loop then pushes `max_gen_length`, `temperature`, `top_p` and
`strategy` from it into `ChatbotAPI::set_generation_config()`. CLI flags are never re-applied, so a
server started with `--temperature 0.7` silently reverts to the file's temperature on the first
`SIGHUP`, contradicting the documented precedence (CLI > env > file). Found by reading the code.

Action Items:

- [ ] Re-apply the original argv overrides after `reload()` (keep `argc`/`argv` or the parsed overrides), or record which keys came from the CLI and keep them fixed across reloads.
- [ ] Log which settings changed on reload, and which were held by CLI overrides.
- [ ] Test: CLI-set temperature survives a reload.

Location in code: `src/ChatbotAPIServer.cpp` (main loop reload branch); tagged `TODO: See TD-242`.

Files to Modify:

- `src/ChatbotAPIServer.cpp`
- `src/ChatbotApiServerArgs.{hpp,cpp}`

---

### TD-243: `chatbot_api_server` Starts Every Engine Mode Flag Passed, Not Just One

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Serving / CLI | October 8, 2026 | 1 hour |

Description:
Found alongside TD-240 and confirmed with the real parser: `--batched-inference
--pipeline-inference --integrated-inference` sets all three with no error.
`ChatbotApiServerArgs.hpp` says at most one should be set and that `ChatbotAPIServer.cpp` "only ever
enables one", but `main()` checks each with an independent `if`. So all three engines are created
(worker threads, plus the pipeline mode's encoder vocab reload) while
`ChatbotAPI::generate_response()` silently uses only the first in its precedence order.

Action Items:

- [ ] Reject more than one mode flag with an error in `apply_chatbot_api_server_args()`.
- [ ] Fix the header comment.
- [ ] Also reject `--batch-timeout-ms` without `--batched-inference` (currently silently ignored).
- [ ] Tests for both.

Location in code: `src/ChatbotApiServerArgs.{hpp,cpp}`, `src/ChatbotAPIServer.cpp` (mode enabling); tagged `TODO: See TD-243`.

Files to Modify:

- `src/ChatbotApiServerArgs.{hpp,cpp}`
- `tests/chatbot_api_server_args_test.cpp`

---

### TD-244: `chatbot_api_server` Ignores the `TOP_K` and `BEAM_WIDTH` Settings

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / CLI | October 8, 2026 | 1-2 hours |

Description:
Found alongside TD-240. `ServiceConfig` parses `TOP_K` and `BEAM_WIDTH` from the config file and
environment (`src/Config.cpp`), but `ChatbotAPIServer.cpp` never copies them into
`ChatbotAPI::GenerationConfig`, either at startup or on `SIGHUP` reload, so `ChatbotAPI` always
uses its built-in defaults (`top_k` 50, `beam_width` 4). There are no `--top-k`/`--beam-width` CLI
flags either. Found by reading the code.

Action Items:

- [ ] Copy `config.top_k` and `config.beam_width` into the `GenerationConfig` at startup and on reload.
- [ ] Add `--top-k` and `--beam-width` flags (and to `print_usage()`).
- [ ] Tests: flags applied; config values reach `ChatbotAPI`.

Location in code: `src/ChatbotAPIServer.cpp` (startup and reload `GenerationConfig` construction), `src/ChatbotApiServerArgs.cpp`; tagged `TODO: See TD-244`.

Files to Modify:

- `src/ChatbotAPIServer.cpp`
- `src/ChatbotApiServerArgs.cpp`
- `tests/chatbot_api_server_args_test.cpp`

---

### TD-245: `chatbot_api_server` Argument Errors Are Misleading or Silent

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Serving / CLI | October 8, 2026 | 1 hour |

Description:
Found alongside TD-240 and confirmed with the real parser:

- A known flag missing its value (e.g. a trailing `--port`) is reported as
  `Unknown argument: --port`.
- A trailing `--config` with no value is silently ignored, and normal config discovery runs.
- The header comment names the test file `ChatbotApiServerArgs_test.cpp`; the real file is
  `chatbot_api_server_args_test.cpp`.

Action Items:

- [ ] Report "Missing value for --port" for known flags at the end of argv, including `--config`.
- [ ] Fix the header comment's test file name.
- [ ] Tests for the missing-value messages.

Location in code: `src/ChatbotApiServerArgs.{hpp,cpp}`; tagged `TODO: See TD-245`.

Files to Modify:

- `src/ChatbotApiServerArgs.{hpp,cpp}`
- `tests/chatbot_api_server_args_test.cpp`
---

### TD-246: `chatbot_api_server` Serves a Randomly Initialized Model When Loading Fails

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Inference / Serving / Server | October 8, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [ChatbotAPIServer.md](../reference/source/ChatbotAPIServer.md). In `main()`, if
`model->load_model(MODEL_PATH)` throws (missing file, architecture mismatch after an MNS outage
falls back to local config, a corrupt checkpoint, or one missing its `.vocab` per TD-221), the server
logs a warning, "Using random initialization", and **keeps serving the untrained model**. With no
`MODEL_PATH` at all it does the same, logging only "training required". In both cases
`GET /health` reports `{"status":"ok"}`, so clients get gibberish and monitoring sees a healthy
service. Found by reading the code.

Action Items:

- [ ] Fail startup (exit 1) when the model can't be loaded, unless an explicit development opt-in (e.g. `ALLOW_UNTRAINED_MODEL=true`) is set.
- [ ] Report model state in `/health` (`model_loaded`, model path/name), and return a non-`ok` status when serving untrained weights.
- [ ] Test: a bad `MODEL_PATH` exits non-zero by default.

Location in code: `src/ChatbotAPIServer.cpp` (model load), `src/ChatbotAPI.cpp` (`handle_health()`); tagged `TODO: See TD-246`.

Files to Modify:

- `src/ChatbotAPIServer.cpp`
- `src/ChatbotAPI.{hpp,cpp}`
- `src/Config.{hpp,cpp}` (opt-in flag)
- `docs/operations/guides/chatbot-guide.md`

---

### TD-247: RAG Retrieval in `chatbot_api_server` Uses an Untrained Encoder

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Inference / Serving / Server | October 8, 2026 | 2-4 hours |

Description:
Found alongside TD-246. When `RAG_ENABLED`, `main()` builds a **new** `LLMEncoder` from the model's
architecture and loads only the vocab into it (`load_tokenizer_vocab()`). `LLMEncoder::load_weights()`
is never called, so `DocumentStore::addDocument()`/`retrieve()` embed every document and query with
**randomly initialized weights**. Retrieval ranking is therefore not learned similarity (at best
crude token overlap through random projections), while generation uses the trained model, so
answers look plausible over poorly chosen context. The trained encoder is already available via
`model->get_encoder()` with the same architecture. Found by reading the code.

Action Items:

- [ ] Embed with the trained encoder: share `model->get_encoder()` (taking care that RAG encoding and generation don't use it concurrently outside `EncoderDecoderModel`'s locking), or load the checkpoint's encoder weights into the RAG encoder.
- [ ] If no trained model is loaded (TD-246), refuse to enable RAG.
- [ ] Test: retrieval ranks a document containing the query's content above unrelated ones with a trained checkpoint.

Location in code: `src/ChatbotAPIServer.cpp` (RAG setup); tagged `TODO: See TD-247`.

Files to Modify:

- `src/ChatbotAPIServer.cpp`
- `src/DocumentStore.{hpp,cpp}` (if ownership changes)
- `tests/rag_*`

---

### TD-248: `chatbot_api_server` Can Hang If Shutdown Arrives During Server Startup

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / Server | October 8, 2026 | 1-2 hours |

Description:
Found alongside TD-246 and confirmed against the vendored cpp-httplib source.
`ChatbotAPI::start()` sets its `running_` flag, then calls `httplib::Server::listen()`. httplib's
`Server::stop()` only acts once the server is inside `listen_internal()` (`if (is_running_) {...}`).
A `SIGTERM`/`SIGINT` that arrives after the server thread starts but before httplib is listening
(e.g. a quick `systemctl restart`) has its `stop()` silently ignored. `listen()` then blocks
forever, `server_thread.join()` hangs, and further signals do nothing because the shutdown flag is
already set. Only systemd's stop timeout or `SIGKILL` ends the process, which also skips the
hippocampal memory save (TD-250).

Action Items:

- [ ] After launching the server thread, wait for `httplib::Server::wait_until_ready()` before entering the main loop; or, on shutdown, call `stop()` repeatedly until the server thread exits.
- [ ] Test: requesting shutdown immediately after start terminates promptly.

Location in code: `src/ChatbotAPIServer.cpp` (server thread start, shutdown), `src/ChatbotAPI.cpp` (`start()`, `stop()`); tagged `TODO: See TD-248`.

Files to Modify:

- `src/ChatbotAPIServer.cpp`
- `src/ChatbotAPI.{hpp,cpp}`

---

### TD-249: `chatbot_api_server`'s `SIGHUP` Handler Isn't Async-Signal-Safe

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / Server | October 8, 2026 | 1 hour |

Description:
Found alongside TD-246. `signal_handler()`'s doc comment says it "is async-signal-safe and only
sets atomic flags", but the `SIGHUP` branch also calls `adai::Logger::info()`. spdlog takes locks
and allocates, so a `SIGHUP` delivered to a thread that's mid-log can deadlock or corrupt logger
state. The main loop's reload branch already logs the reload, so the call isn't needed.

Action Items:

- [ ] Remove the logger call from the handler; log "SIGHUP received" in the main loop when it sees the flag.
- [ ] Check the other daemons' signal handlers for the same pattern.

Location in code: `src/ChatbotAPIServer.cpp` (`signal_handler()`); tagged `TODO: See TD-249`.

Files to Modify:

- `src/ChatbotAPIServer.cpp`

---

### TD-250: Hippocampal Memory in `chatbot_api_server` Is Lost on Crash and Can Land in the Working Directory

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / Serving / Server | October 8, 2026 | 2-4 hours |

Description:
Found alongside TD-246. When the world model and hippocampal memory are attached
(experimental, TD-193/194), the memory changes with every generated response, but `main()` saves it
**only on graceful shutdown**. A crash, `SIGKILL`, OOM kill or the hang in TD-248 loses everything
since the last clean stop. Its paths are `MODEL_PATH + ".hippocampal"`, `".hippocampal.assoc"`
and `".hippocampal.swap"`, so with an empty `MODEL_PATH` (random-init mode) they become hidden
files in the process's working directory.

Action Items:

- [ ] Save periodically (time- or write-count-based) using write-to-temp-then-rename, in addition to shutdown.
- [ ] Add an explicit `HIPPOCAMPAL_STATE_PATH` (defaulting to the current derivation), and refuse to attach with neither it nor `MODEL_PATH` set.
- [ ] Test: state survives a simulated crash after the periodic save.

Location in code: `src/ChatbotAPIServer.cpp` (hippocampal attach, shutdown save); tagged `TODO: See TD-250`.

Files to Modify:

- `src/ChatbotAPIServer.cpp`
- `src/Config.{hpp,cpp}`
- `src/HippocampalMemory.{hpp,cpp}` (if atomic save lives there)

---

### TD-251: `chatbot_api_server` Reload Silently Ignores Most Settings

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Serving / Server | October 8, 2026 | 1-2 hours |

Description:
Found alongside TD-246. On `SIGHUP`, `main()` applies only the log level and four generation
fields (`max_gen_length`, `temperature`, `top_p`, `strategy`), and warns only for a port change.
Changes to session timeout, RAG settings, model/architecture, GPU settings and the rest are
dropped without any message, while the log reports a successful configuration reload. Reload also
doesn't redo MNS resolution, so the in-memory `config` reverts to local file values for model path
and architecture (unused after startup, but wrong). Found by reading the code.

Action Items:

- [ ] After reload, diff old vs. new config and log every changed key that wasn't applied ("requires restart").
- [ ] Apply what can be applied live (e.g. session timeout, via a `ChatbotAPI` setter).
- [ ] Keep MNS-resolved fields when replacing `config`.

Location in code: `src/ChatbotAPIServer.cpp` (main loop reload branch); tagged `TODO: See TD-251`.

Files to Modify:

- `src/ChatbotAPIServer.cpp`
- `src/ChatbotAPI.hpp` (if adding setters)

---

### TD-252: `chatbot_api_server` Startup Log and Usage Text Are Inaccurate

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / Serving / Server | October 8, 2026 | 1 hour |

Description:
Found alongside TD-246:

- The startup banner lists only 4 of the 9 endpoints. It omits `/chat/batch`,
  `/chat/batch-session`, `/chat/session/export` and `/chat/session/import`, and lists
  `/admin/profile` only with `--profile`.
- `print_usage()` presents `--d-model 512` etc. as defaults without saying MNS overrides them when
  the model resolves.

(`--batched-inference`'s "real batching" wording is already TD-211.)

Action Items:

- [ ] List every registered route (ideally generated from the same table the constructor registers).
- [ ] Note in `print_usage()` that architecture flags are fallbacks overridden by MNS.

Location in code: `src/ChatbotAPIServer.cpp` (startup banner, `print_usage()`); tagged `TODO: See TD-252`.

Files to Modify:

- `src/ChatbotAPIServer.cpp`
---

### TD-253: `chatbot` Spins Forever at End of Input

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Client / CLI | October 8, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [ChatbotCLI.md](../reference/source/ChatbotCLI.md), and confirmed by running the real
`chatbot` binary against a stub server. `ChatbotCLI::run()` never checks `std::getline()`'s result.
At EOF (Ctrl+D in a terminal, or stdin from a file or pipe running out) `getline` fails and leaves
the line empty, the empty-line check `continue`s, and the loop prints `You: ` forever. With stdin
at EOF it printed **18.6 million prompts (284 MB) in 3 seconds** before being killed: in a
terminal it pins a CPU core, and with output redirected it fills the disk at about 95 MB/s. It
also never reaches the `/exit` auto-save. The old internals doc listed Ctrl+D as an exit path.

Action Items:

- [ ] Break out of the loop when `std::getline()` fails (`if (!std::getline(std::cin, user_input)) break;`), running the same auto-save as `/exit`.
- [ ] Test: `chatbot < /dev/null` against a stub server exits promptly with status 0.

Location in code: `src/ChatbotCLI.cpp` (`run()`); tagged `TODO: See TD-253`.

Files to Modify:

- `src/ChatbotCLI.cpp`
- `tests/chatbotcli_improved_test.cpp`

---

### TD-254: `chatbot`'s `/set` Generation Settings Have No Effect

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Client / CLI | October 8, 2026 | 2-4 hours |

Description:
Found alongside TD-253 and confirmed by running the real binary: after `/set temp 0.2` the next
request body carried `"temperature":0.2`. `ChatbotCLI::generate_response()` sends `max_length`,
`temperature`, `top_p`, `top_k`, `beam_width` and `strategy` with every message, but
`ChatbotAPI`'s handlers ignore them and always use the server's defaults (see
[ChatbotAPI.md](../reference/source/ChatbotAPI.md) §3). The CLI confirms each `/set` and
`/settings` shows the new values, yet generation never changes. The strategy names also don't
match: the CLI accepts `sampling` and `top-k`, the server uses `temperature` and `top_k`, so even a
server that honoured them would fall back to `nucleus`. The user guide
([chatbot-guide.md](../../operations/guides/chatbot-guide.md)) documents `/set` as working (its
Generation Strategies section is already flagged under TD-164).

Action Items:

- [ ] Owner decision: support per-request generation settings in `ChatbotAPI` (validated and bounded; easiest after TD-226's nlohmann/json switch), or remove the per-request settings from the CLI.
- [ ] If supported: align strategy names on the server's (`temperature`, `top_k`) and accept the CLI's old names as aliases.
- [ ] If removed: make `/set` print that settings are server-side (`--strategy` etc.) and drop the fields from the request.
- [ ] Update `chatbot-guide.md` either way.

Location in code: `src/ChatbotCLI.cpp` (`handle_setting()`, `generate_response()`), `src/ChatbotAPI.cpp` (`handle_chat_session()`); tagged `TODO: See TD-254`.

Files to Modify:

- `src/ChatbotCLI.cpp`
- `src/ChatbotAPI.cpp`
- `tests/chatbotcli_test.cpp`
- `tests/chatbotapi_test.cpp`
- `docs/operations/guides/chatbot-guide.md`

---

### TD-255: `chatbot` Shows Server Error Replies as Empty Responses

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Client / CLI | October 8, 2026 | 1 hour |

Description:
Found alongside TD-253 and confirmed against a stub server. `generate_response()` treats any
HTTP 200 reply as success and prints its `response` field. `ChatbotAPI` returns HTTP 200 with
`{"success":false,"error":...}` for errors it returns rather than throws (TD-236). The CLI doesn't
check `success`, so the user sees an empty `Bot:` line and the error message is lost. Thrown server
errors (HTTP 500) are printed as the raw JSON body.

Action Items:

- [ ] Check `success`; on `false`, print the `error` field in `COLOR_ERROR`.
- [ ] Print only the `error` field (not the raw JSON) for non-200 replies.
- [ ] Test with a stub returning `success:false`.

Location in code: `src/ChatbotCLI.cpp` (`generate_response()`); tagged `TODO: See TD-255`.

Files to Modify:

- `src/ChatbotCLI.cpp`
- `tests/chatbotcli_improved_test.cpp`

---

### TD-256: `chatbot`'s Help Banner Omits `/save`, `/load` and `/stats`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Client / CLI | October 8, 2026 | 1 hour |

Description:
Found alongside TD-253. `print_welcome()` (also shown by `/help`) lists `/help`, `/clear`,
`/settings`, `/set` and `/exit`/`/quit`, but not `/save` and `/load` (TD-053) or `/stats`, all of
which `handle_command()` accepts. `/set` alone (no parameter) is reported as "Unknown command"
rather than with its usage.

Action Items:

- [ ] List every command, including `/set` parameters and valid values.
- [ ] Show `/set` usage for a bare `/set`.

Location in code: `src/ChatbotCLI.cpp` (`print_welcome()`, `handle_command()`); tagged `TODO: See TD-256`.

Files to Modify:

- `src/ChatbotCLI.cpp`

---

### TD-257: `chatbot` Sends `session_id` Unescaped in Two Requests

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Client / CLI | October 8, 2026 | 1 hour |

Description:
Found alongside TD-253. `/clear` and `generate_response()` put `session_id` into the request
JSON without `escape_json_string()`, while `/save` and `/load` do escape it. Server-generated IDs
are hex, so this is safe today, but an ID containing `"` or `\` (e.g. one restored via a hand-edited
save file, or a server accepting client-chosen IDs per TD-232) would produce malformed or
injected request JSON.

Action Items:

- [ ] Escape `session_id` in every request body (or build request bodies with nlohmann/json).

Location in code: `src/ChatbotCLI.cpp` (`handle_command()` `/clear`, `generate_response()`); tagged `TODO: See TD-257`.

Files to Modify:

- `src/ChatbotCLI.cpp`

---

### TD-258: `chatbot` Can Lose or Overwrite Saved Conversations

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Client / CLI | October 8, 2026 | 2-4 hours |

Description:
Found alongside TD-253:

- No `SIGINT` handling: Ctrl+C kills the process without the auto-save that `/exit` performs.
- The `/exit` auto-save always writes the same file, so starting a new conversation and exiting
  **silently overwrites** a previously saved one.
- Saves truncate and then write (not atomic), so a crash mid-write leaves a partial file.
- After a server restart, the CLI keeps sending its old `session_id` and the server silently
  creates a fresh, empty session under it (TD-232), so the history disappears with no message.

Action Items:

- [ ] Handle `SIGINT` by setting a flag the REPL checks, then auto-save and exit.
- [ ] Warn (or prompt) before an auto-save would overwrite a file the current session didn't load; consider timestamped files.
- [ ] Write saves to a temporary file and rename.
- [ ] Detect a reset session (e.g. `message_count` from the server) and tell the user.

Location in code: `src/ChatbotCLI.cpp` (`run()`, `save_conversation()`); tagged `TODO: See TD-258`.

Files to Modify:

- `src/ChatbotCLI.{hpp,cpp}`
- `tests/chatbotcli_improved_test.cpp`

---

### TD-259: `chatbot` Code Hygiene: Unchecked `/set` Values, Global Helpers, Heavy Header

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Client / CLI | October 8, 2026 | 1-2 hours |

Description:
Found alongside TD-253:

- `/set` numbers use `std::stoi`/`std::stof`, which accept partial input (`10abc` → 10), and
  nothing range-checks them (negative length, `top_p` 5). Moot while TD-254 stands.
- `escape_json_string()`, `unescape_json_string()` and `parse_json_value()` in `ChatbotCLI.cpp`
  have external linkage at global scope, so another translation unit defining the same global names
  would collide (`MetricsPushClient.cpp` is safe only because its copy is in an anonymous namespace).
- `ChatbotCLI.hpp` includes all of `httplib.h` (via an unusual `<../external/...>` path) and
  `#define`s the `COLOR_*` macros for every includer.

Action Items:

- [ ] Use full-consumption parsing and range checks for `/set` values.
- [ ] Move the JSON helpers into an anonymous namespace (or replace them with nlohmann/json).
- [ ] Forward-declare `httplib::Client` in the header (pimpl or `unique_ptr` to an incomplete type) and move the colour macros to `constexpr` values in the `.cpp`.

Location in code: `src/ChatbotCLI.{hpp,cpp}`; tagged `TODO: See TD-259`.

Files to Modify:

- `src/ChatbotCLI.{hpp,cpp}`
---

### TD-260: `chatbot_gui` Never Loads a Trained Checkpoint

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Client / GUI | October 9, 2026 | 1 hour |

Description:
Found while writing the code-traced reference [ChatbotGUI.md](../reference/source/ChatbotGUI.md). `ChatbotGUI::initializeChatbot()`
calls `model->load_model(model_path)` only if `std::ifstream(model_path).good()`. But
`EncoderDecoderModel::save_model(path)` writes only sidecar files (`path.config`, `.vocab`,
`.encoder`, `.decoder`, `.lm_head`) and **never a file at `path` itself**, so for any real
checkpoint the check fails, `load_model()` is never called, and the GUI **silently** chats with a
randomly initialized model. No message is shown. `IncrementalTrainer.cpp` documents this exact
trap ("checking for that bare path here would reject every legitimately-saved checkpoint
unconditionally") and checks the five sidecars instead. Passing an existing sidecar such as
`model.config` doesn't help: `load_model()` appends the suffixes again and throws.

Action Items:

- [ ] Check for the sidecars (or just call `load_model()` and handle its exception), the same way `IncrementalTrainer` does; consider a shared `EncoderDecoderModel::checkpoint_exists(path)` helper.
- [ ] When no checkpoint is found, show a clear warning in the chat pane that the model is untrained (matching TD-246's server-side stance).
- [ ] Update the default/usage text (`chatbot_model.bin`) to describe a checkpoint base path.

Location in code: `src/ChatbotGUI.cpp` (`initializeChatbot()`); tagged `TODO: See TD-260`.

Files to Modify:

- `src/ChatbotGUI.cpp`
- `src/ChatbotGUI_main.cpp`
- `src/EncoderDecoderModel.{hpp,cpp}` (optional helper)

---

### TD-261: `generate_response_with_strategy()` Forces Beam Search Whenever `num_beams > 1`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Core Model / Generation | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-260. In `EncoderDecoderModel::generate_response_with_strategy()`, the beam
branch is taken when `normalized_strategy == "beam" || num_beams > 1`, and it returns before the
`greedy`/`sampling`/`topk`/`nucleus` branches. The comment just above says those explicit
strategies "are unaffected", but the code order contradicts it. So any caller passing
`num_beams > 1` gets beam search **whatever strategy it asked for**:

- `chatbot_gui` always passes its Beam Width spin box (default **5**), so choosing Nucleus, Top-k,
  Greedy or Sampling still runs beam search unless the user sets Beam Width to 1.
- `RLHFTrainer` (≈ line 177) omits `num_beams`, so it gets the header default of **4**: its
  configured `generation_strategy` (e.g. sampling, which policy-gradient training relies on for
  diverse rollouts) is silently replaced by beam search.
- RAG passes `num_beams` from its config (default 1), so it's unaffected.

The `num_beams > 1` guard exists for a real reason: an unrecognized strategy falls through to
`TextGenerator::generate()`, which itself routes to beam search when `num_beams > 1`, and must not
be handed the KV-cached `model_fn`.

Action Items:

- [ ] Take the beam branch only for `strategy == "beam"`, or for an *unrecognized* strategy with `num_beams > 1` (preserving the cache-safety guard); explicit greedy/sampling/topk/nucleus must ignore `num_beams`.
- [ ] Change the header default `num_beams = 4` to 1 (in both overloads) so omitted arguments don't imply beam search.
- [ ] `chatbot_gui`: pass `beam_width` only when the strategy is beam (otherwise 1). `RLHFTrainer`: pass `num_beams` explicitly.
- [ ] Fix the misleading comment; add tests that each explicit strategy with `num_beams = 5` does not run beam search.

Location in code: `src/EncoderDecoderModel.{hpp,cpp}` (`generate_response_with_strategy()`), `src/ChatbotGUI.cpp` (`generateResponse()`), `src/RLHFTrainer.cpp`; tagged `TODO: See TD-261`.

Files to Modify:

- `src/EncoderDecoderModel.{hpp,cpp}`
- `src/ChatbotGUI.cpp`
- `src/RLHFTrainer.cpp`
- `tests/encoderdecoder_test.cpp`

---

### TD-262: `chatbot_gui` Ignores MNS Architecture and Becomes Unusable on Init Failure

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Client / GUI | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-260. `initializeChatbot()` takes the model architecture only from
`config.chatbot.conf` (discovered from the working directory), not from MNS (which
`chatbot_api_server` and the trainer treat as authoritative) or the checkpoint's own `.config`.
A checkpoint trained under a different architecture fails to load. Any exception during init then:

- shows a warning saying "Using default/random initialization", which is wrong;
- returns `false`, so the constructor shows a **second**, critical dialog;
- leaves `context` null, so every message answers "[Error: Chatbot not initialized]".

Action Items:

- [ ] Read the architecture from the checkpoint's `.config` (or resolve via MNS when `NAME_SERVICE_URL`/`MODEL_NAME` are configured), falling back to the config file.
- [ ] Show one accurate error dialog, and keep the GUI usable (or disable Send) with a clear status.
- [ ] Add a `--config` option instead of relying on the working directory.

Location in code: `src/ChatbotGUI.cpp` (`initializeChatbot()`, constructor); tagged `TODO: See TD-262`.

Files to Modify:

- `src/ChatbotGUI.{hpp,cpp}`
- `src/ChatbotGUI_main.cpp`

---

### TD-263: `chatbot_gui` Freezes While Generating

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Client / GUI | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-260. `onSendMessage()` calls `generateResponse()` synchronously on the GUI
thread. One `processEvents()` beforehand lets "Generating..." paint, but the event loop is then
blocked for the entire generation (seconds to minutes on CPU): the window can't repaint, scroll or
close, and desktops may flag it as "not responding".

Action Items:

- [ ] Run generation on a worker thread (`QtConcurrent::run` + `QFutureWatcher`, or a `QThread`), posting the result back to the GUI thread; disable settings while it runs.
- [ ] Add a cancel/stop control if the generator can support it.

Location in code: `src/ChatbotGUI.cpp` (`onSendMessage()`); tagged `TODO: See TD-263`.

Files to Modify:

- `src/ChatbotGUI.{hpp,cpp}`
- `src/CMakeLists.txt` (if `Qt::Concurrent` is added)

---

### TD-264: `chatbot_gui` Launcher Hardcodes Qt5 and x86_64 Paths

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Client / GUI | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-260. `ChatbotGUI_wrapper.cpp` always sets
`QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins`, but CMake builds
`chatbot_gui_binary` against **Qt6 when available** (Qt5 is only the fallback), so a Qt6 build is
pointed at Qt5 platform plugins. Library paths are hardcoded for x86_64 Debian/Ubuntu. It also
puts system paths **before** the user's `LD_LIBRARY_PATH` and drops it entirely if it contains
`/snap/`.

Action Items:

- [ ] Pass the Qt major version and plugin directory from CMake (`configure_file` or a compile definition), or only set `QT_QPA_PLATFORM_PLUGIN_PATH` when running under snap.
- [ ] Derive library paths from the build (or don't override them unless snap is detected).
- [ ] Preserve the user's `LD_LIBRARY_PATH` order, removing only `/snap/` entries.

Location in code: `src/ChatbotGUI_wrapper.cpp`; tagged `TODO: See TD-264`.

Files to Modify:

- `src/ChatbotGUI_wrapper.cpp`
- `src/CMakeLists.txt`

---

### TD-265: `chatbot_gui` and `chatbot_api_server` Format Conversation Context Differently

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Client / GUI | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-260. The GUI builds each prompt with
`ConversationContext::format_with_special_tokens()` (`<bos> [USER] … <sep>` tags, which
`BPETokenizer` treats as ordinary characters, not special tokens). `ChatbotAPI` sessions use
`format_for_model()` (`User: …` / `Assistant: …`). The same model sees differently shaped
prompts depending on the client, so behaviour isn't comparable between them.

Action Items:

- [ ] Owner decision: pick one multi-turn format (ideally whatever the training data uses) and use it in both clients.

Location in code: `src/ChatbotGUI.cpp` (`generateResponse()`); tagged `TODO: See TD-265`.

Files to Modify:

- `src/ChatbotGUI.cpp` and/or `src/ChatbotAPI.cpp`

---

### TD-266: `chatbot_gui` Minor Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Client / GUI | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-260:

- The context limit is hardcoded to 480 tokens with a comment about a 512-token model,
  regardless of `MAX_SEQ_LENGTH`.
- Loading a conversation doesn't redisplay its messages (a code comment notes this).
- `ChatbotGUI_main.cpp` creates the `QApplication` before checking `--help`, so `--help` needs a
  display.
- GPU acceleration is never initialized (no `GPU_ENABLED` handling); the GUI always runs on CPU.

Action Items:

- [ ] Derive the context limit from the model's `max_seq_length` minus the response budget.
- [ ] Redisplay loaded messages (needs a message accessor on `ConversationContext`).
- [ ] Handle `--help` before constructing `QApplication`.
- [ ] Honour `GPU_ENABLED` the way `chatbot_api_server` does.

Location in code: `src/ChatbotGUI.cpp`, `src/ChatbotGUI_main.cpp`; tagged `TODO: See TD-266`.

Files to Modify:

- `src/ChatbotGUI.{hpp,cpp}`
- `src/ChatbotGUI_main.cpp`
- `src/ConversationContext.hpp`
---

### TD-267: ChatbotTrainer Keeps Skipped Pairs, Biasing Validation Loss Low

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Training / ChatbotTrainer | October 9, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [ChatbotTrainer.md](../reference/source/ChatbotTrainer.md), and confirmed by a real training pass.
`preprocess_data()` skips pairs with an empty input/response (TD-188) or invalid UTF-8, logging
"Skipped N…", but leaves each one's slot as an **empty `TokenizedPair`**. The comment says they're
"filtered out during training"; nothing filters them:

- **Training:** each empty pair reaches `model->forward()`, which throws ("Matrix index out of
  bounds" in the test run). The catch block resets accumulation, which also **discards gradients
  already accumulated** from earlier samples in that window.
- **Validation:** each empty pair evaluates to loss 0 with no error but still counts in
  `validate()`'s average (sum ÷ all pairs). In the test run, adding one empty pair to two good ones
  dropped validation loss from about 3.4 to 2.5. Validation loss drives best-model tracking, early
  stopping, the metrics dashboard and MNS progress.

`validate()` also counts samples that throw as 0, the same bias.

Action Items:

- [ ] After tokenizing, compact `tokenized_*_data` (and the matching `training_data`/`validation_data`, keeping indices aligned) to drop skipped pairs, or skip empty pairs in `train_epoch()` and `validate()`.
- [ ] In `validate()`, divide by the number of successfully evaluated samples, and report how many failed.
- [ ] Regression tests: an empty pair changes neither validation loss nor training error count; tokenized sizes exclude skipped pairs.

Location in code: `src/ChatbotTrainer.cpp` (`preprocess_data()`, `train_epoch()`, `validate()`); tagged `TODO: See TD-267`.

Files to Modify:

- `src/ChatbotTrainer.cpp`
- `tests/chatbottrainer_test.cpp`

---

### TD-268: ChatbotTrainer Only Ends Metrics Epochs When There Is Validation Data

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / ChatbotTrainer | October 9, 2026 | 1 hour |

Description:
Found alongside TD-267. `train()` calls `metrics_reporter_->end_epoch()` only inside
`if (!tokenized_validation_data.empty())`. A run with `validation_split = 0`, or too little data to
split, starts epochs on the reporter (`train_epoch()` calls `start_epoch()`) but never ends them, so
the dashboard never records epoch completion, loss or perplexity for those runs. The gradient norm
passed to `end_epoch()` is also the last window's (`optimizer->get_gradient_norm()`), not the
epoch average already computed in `gradient_norms.back()`.

Action Items:

- [ ] Call `end_epoch()` every epoch, passing a "no validation" value (as the epoch callback does with 0.0) when there's no validation data.
- [ ] Pass `gradient_norms.back()` as the epoch gradient norm.
- [ ] Test with a reporter mock and `validation_split = 0`.

Location in code: `src/ChatbotTrainer.cpp` (`train()`); tagged `TODO: See TD-268`.

Files to Modify:

- `src/ChatbotTrainer.cpp`
- `tests/chatbottrainer_test.cpp`

---

### TD-269: Optimizer State and LR Schedule Restart Every Incremental Training Pass

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / ChatbotTrainer | October 9, 2026 | 4-8 hours |

Description:
Found alongside TD-267. `IncrementalTrainer` constructs a **fresh `ChatbotTrainer` per pass**,
and `ChatbotTrainer::set_model()` creates a **new `Optimizer`**. So every pass:

- restarts AdamW with zero first and second moments (and bias correction from step 1);
- restarts `global_step` at 0 and computes `total_training_steps` for just this pass, so the default
  `WARMUP_COSINE` schedule warms up from LR 0 and decays to `min_learning_rate` **within each pass**.

A long run made of many short passes never follows one coherent schedule; it repeats warmup and
decay, and loses optimizer state each time. Optimizer state isn't saved in checkpoints, so a
process restart loses it as well.

Action Items:

- [ ] Owner decision: make the schedule run-wide (pass a global step offset and the run's planned total steps from `IncrementalTrainer`), or document per-pass schedules as intended.
- [ ] Persist optimizer state (moments and step count) with the checkpoint and restore it in `set_model()` (or a new `set_optimizer_state()`).
- [ ] Track run-level `global_step` in the MNS/session record.

Location in code: `src/ChatbotTrainer.cpp` (`set_model()`, `calculate_learning_rate()`), `src/IncrementalTrainer.cpp` (per-pass trainer construction); tagged `TODO: See TD-269`.

Files to Modify:

- `src/ChatbotTrainer.{hpp,cpp}`
- `src/IncrementalTrainer.{hpp,cpp}`
- `src/Optimizer.{hpp,cpp}` (state save/load)
- `src/EncoderDecoderModel.cpp` (checkpoint sidecar)

---

### TD-270: Adaptive Gradient Clipping Resets Every Epoch

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / ChatbotTrainer | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-267. The TD-017 adaptive-clipping state (`agc_ema`, `agc_step_count`,
`agc_spike_count`) is declared locally in `train_epoch()`, so each epoch restarts it: a new
`gradient_clip_warmup_steps` (default 100) warmup clipped at `gradient_clip_max` (5.0), with the EMA
re-seeded from `gradient_clip_norm`. With ≤ 100 optimizer steps per epoch (common for small passes
or large accumulation windows) **adaptive clipping never leaves warmup**, so enabling it just clips at
the 5.0 ceiling.

Action Items:

- [ ] Move the adaptive-clip state to `ChatbotTrainer` members (initialized once per `train()` call, or once per run alongside TD-269).
- [ ] Test: the EMA and warmup counter carry across epochs.

Location in code: `src/ChatbotTrainer.cpp` (`train_epoch()`); tagged `TODO: See TD-270`.

Files to Modify:

- `src/ChatbotTrainer.{hpp,cpp}`
- `tests/adaptive_clipping_test.cpp`

---

### TD-271: ChatbotTrainer's Tokenized Cache Can Pair With a Different Random Split

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / ChatbotTrainer | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-267. `train()` runs `split_data()` (a fresh `std::random_device` shuffle)
**before** `preprocess_data()` loads the tokenized cache. The cache holds the tokenized split from
the run that wrote it. Counts match, so the load succeeds, but `tokenized_training_data[i]` no longer
corresponds to `training_data[i]`, and per-index metadata writes (`meta.token_count`, quality
backfill) land on the wrong pairs. Training still uses a valid split (the cached one). Separately,
the split is different on every pass, so validation loss isn't measured on the same samples from
one pass to the next. Only the misalignment depends on `cache_tokenized_data` (default off).

Action Items:

- [ ] Make the split deterministic (seed from the cache key or dataset checksum), or cache the split itself (indices) alongside the tokenized data.
- [ ] Include the split seed or validation membership in the cache key.
- [ ] Test: two `train()` calls with caching produce aligned raw/tokenized pairs.

Location in code: `src/ChatbotTrainer.cpp` (`split_data()`, `preprocess_data()`); tagged `TODO: See TD-271`.

Files to Modify:

- `src/ChatbotTrainer.cpp`
- `src/IncrementalTrainer.cpp` (cache key)
- `tests/chatbottrainer_test.cpp`

---

### TD-272: ChatbotTrainer Silently Drops Gradients on Mid-Window Errors and Under-Weights the Last Window

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / ChatbotTrainer | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-267:

- When a sample throws, or the window's gradient norm is NaN/Inf, `train_epoch()` resets
  `accumulation_step` without stepping, so gradients accumulated from the **earlier** samples in
  that window are zeroed at the next window start and contribute nothing.
- The final window of an epoch is force-flushed even if it holds fewer than
  `gradient_accumulation_steps` samples, but gradients are still scaled by
  `1/gradient_accumulation_steps` and its reported loss divided by the same, so that update is
  proportionally too small and its loss under-reported.

Action Items:

- [ ] On a per-sample error, keep the window's valid gradients: track the samples actually accumulated and step on them (rescaling), or document the drop and count dropped samples in metrics.
- [ ] Scale the last window by its real sample count (rescale gradients before the step; divide the loss by the actual count).
- [ ] Tests for both.

Location in code: `src/ChatbotTrainer.cpp` (`train_epoch()`); tagged `TODO: See TD-272`.

Files to Modify:

- `src/ChatbotTrainer.cpp`
- `tests/chatbottrainer_test.cpp`

---

### TD-273: ChatbotTrainer Quality Backfill Writes Results Nobody Can Read

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / ChatbotTrainer | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-267. `enable_loss_quality_backfill` and `enable_generation_quality_backfill`
write `meta.quality` into `ChatbotTrainer`'s private `training_data`/`validation_data`, which have
no accessor. `IncrementalTrainer` passes pairs via `add_training_pair()` (no metadata) and never
reads them back, and neither option has a config key. The generation backfill would run a full
generation per sample for nothing if enabled in code.

Action Items:

- [ ] Owner decision: remove both backfills, or wire them up (pass `SampleMeta` in, expose the updated metadata, add config keys, persist quality back to the dataset).

Location in code: `src/ChatbotTrainer.cpp` (`backfill_generation_quality()`, loss backfill in `train_epoch()`/`validate()`); tagged `TODO: See TD-273`.

Files to Modify:

- `src/ChatbotTrainer.{hpp,cpp}`
- `src/IncrementalTrainer.cpp`
- `src/Config.{hpp,cpp}`

---

### TD-274: ChatbotTrainer Saves the Best Model to a Fixed File in the Working Directory

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / ChatbotTrainer | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-267. With early stopping and `restore_best_weights` on, `validate()` saves each
new best to the relative path `"best_model_temp.bin"` (five sidecar files) in the process's working
directory. It's never cleaned up, it's shared by every trainer running in that directory, and
`train()` restores from whatever is there at the end.

Action Items:

- [ ] Save under the session directory (or a unique temp directory) and remove it after restoring.

Location in code: `src/ChatbotTrainer.cpp` (`validate()`); tagged `TODO: See TD-274`.

Files to Modify:

- `src/ChatbotTrainer.cpp`

---

### TD-275: ChatbotTrainer Drops `<eos>` From Long Training Responses

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / ChatbotTrainer | October 9, 2026 | 1 hour |

Description:
Found alongside TD-267. `preprocess_data()` encodes responses **with** `<bos>`/`<eos>` and then
keeps the **first** `max_seq_length` tokens. For responses longer than that, the `<eos>` is cut, so
the model never sees those targets end, which teaches it to keep generating on long-form data.

Action Items:

- [ ] Truncate the content before adding specials (keep `<bos>` + first `max_seq_length − 2` tokens + `<eos>`), or decide explicitly that long targets shouldn't end.

Location in code: `src/ChatbotTrainer.cpp` (`preprocess_data()`); tagged `TODO: See TD-275`.

Files to Modify:

- `src/ChatbotTrainer.cpp`
- `tests/chatbottrainer_test.cpp`

---

### TD-276: ChatbotTrainer Hygiene: Dead Fields, Logging, Stale Test File

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / ChatbotTrainer | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-267:

- `calculate_accuracy()` is tested but unused by training; `training_accuracies`/`validation_accuracies` are never filled; validation always reports accuracy −1.
- `TrainingConfig::verbose` is deprecated and unused; `log()` sends every level to `Logger::info` (DEBUG not mapped) and ignores its colour argument; `COLOR_*` macros are redefined locally.
- `validate_and_correct_config()` (standalone `initialize_model()` path only) silently changes `d_model`/`d_ff`, which would break compatibility with an MNS-registered architecture.
- `tests/chatbottrainer_test.cpp.old` is an unbuilt January-2026 leftover.

Action Items:

- [ ] Compute token accuracy in `validate()` (or remove the dead vectors and getters).
- [ ] Remove `verbose`; map `log()` levels to `Logger` levels.
- [ ] Make `validate_and_correct_config()` fail instead of silently changing architecture.
- [ ] Delete `tests/chatbottrainer_test.cpp.old`.

Location in code: `src/ChatbotTrainer.{hpp,cpp}`; tagged `TODO: See TD-276`.

Files to Modify:

- `src/ChatbotTrainer.{hpp,cpp}`
- `tests/chatbottrainer_test.cpp.old` (delete)
---

### TD-277: `ChildProcess` Reports Every Signal Death as Exit Code 128

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / Service Supervisor | October 9, 2026 | 1 hour |

Description:
Found while writing the code-traced reference [ChildProcess.md](../reference/source/ChildProcess.md), and confirmed by running the real
class: a child killed by SIGSEGV and one killed by SIGKILL both come back from `poll_exit()` as
**128**. The header calls 128 "POSIX convention", but the shell convention is **128 + signal number**
(139 for SIGSEGV, 137 for SIGKILL). On ai-machine the known pass-killing failure is a SIGSEGV inside
the Intel graphics compiler ([ONEAPI_SYCL_DRIVER_SEGFAULT.md](../../operations/guides/troubleshooting/ONEAPI_SYCL_DRIVER_SEGFAULT.md)),
and an OOM kill would be SIGKILL. `trainer_service`'s `/admin/status` `last_exit_code` can't tell them
apart, which removes exactly the diagnosis TD-172's process isolation was meant to make possible.

Action Items:

- [ ] Report `128 + WTERMSIG(status)` for signal deaths (and note core dumps via `WCOREDUMP` in a log line).
- [ ] Update the header and `TrainerServiceControlState::last_exit_code`'s comment; keep `total_passes_crashed` counting unchanged.
- [ ] Test: SIGSEGV → 139, SIGKILL → 137.

Location in code: `src/ChildProcess.{hpp,cpp}` (`poll_exit()`), `src/TrainerServiceControlState.hpp`; tagged `TODO: See TD-277`.

Files to Modify:

- `src/ChildProcess.{hpp,cpp}`
- `src/TrainerServiceControlState.hpp`
- `tests/child_process_test.cpp`

---

### TD-278: `ChildProcess` Reports a Failed `exec` as a Successful Launch

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / Service Supervisor | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-277 and confirmed by running the real class: with a missing binary,
`start()` returns **true** (after `fork()`), and the failure only appears later as exit code **127**
from `poll_exit()`. `trainer_service` counts that as a crashed pass and retries after every poll
interval, indefinitely, without ever reporting that `incremental_trainer` couldn't be executed
(e.g. after a bad install or a wrong path).

Action Items:

- [ ] Detect exec failure in `start()`: a close-on-exec pipe the child writes `errno` to before `_exit()`, or `posix_spawn()` (which reports exec errors directly on glibc).
- [ ] Log the exec error with the path; have `trainer_service` back off (or stop) on repeated launch failures rather than retrying silently.
- [ ] Test: `start({"/nonexistent"})` returns false.

Location in code: `src/ChildProcess.cpp` (`start()`); tagged `TODO: See TD-278`.

Files to Modify:

- `src/ChildProcess.cpp`
- `src/TrainerServiceMain.cpp`
- `tests/child_process_test.cpp`

---

### TD-279: `ChildProcess::request_stop()` Can Signal Every Process if Called Concurrently

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / Service Supervisor | October 9, 2026 | 1 hour |

Description:
Found alongside TD-277. The header says `request_stop()` is safe to call from a signal handler's
thread. On POSIX it checks the plain `bool running_` and then calls `kill(pid_, SIGTERM)`, but nothing
is atomic, and `stop_and_wait()`'s give-up path sets `pid_ = -1` **before** clearing `running_`
(compilers may also reorder the other paths' stores). A concurrent call could therefore read
`running_ == true` and `pid_ == -1`, and `kill(-1, SIGTERM)` sends SIGTERM to **every process the user
can signal**. It isn't reachable today (`trainer_service`'s handler only sets a flag), but the
documented contract invites it.

Action Items:

- [ ] Guard `kill()` with `pid_ > 0` (and the Windows path with a non-null handle).
- [ ] Make `running_`/`pid_` atomic, or drop the signal-handler safety claim from the header.

Location in code: `src/ChildProcess.{hpp,cpp}` (`request_stop()`, `stop_and_wait()`); tagged `TODO: See TD-279`.

Files to Modify:

- `src/ChildProcess.{hpp,cpp}`

---

### TD-280: `trainer_service` Passes Can Be Orphaned if the Supervisor Is Hard-Killed

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / Service Supervisor | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-277. `ChildProcess` signals only the direct child, sets no process group, and
uses no `PR_SET_PDEATHSIG`. Under systemd the default `KillMode=control-group` kills the whole cgroup,
but if `trainer_service` is hard-killed outside systemd (manual `kill -9`, a crash, another
supervisor), a running pass keeps going, still holding dataset claims and the GPU, while a
restarted `trainer_service` launches a second pass.

Action Items:

- [ ] In the child before `exec`: `prctl(PR_SET_PDEATHSIG, SIGTERM)` (Linux), re-checking `getppid()` for the fork race; optionally a process group so stop requests reach grandchildren.
- [ ] Document the systemd `KillMode` requirement in the service file.

Location in code: `src/ChildProcess.cpp` (`start()`); tagged `TODO: See TD-280`.

Files to Modify:

- `src/ChildProcess.cpp`
- `scripts/adai-trainer.service`

---

### TD-281: `ChildProcess` Forks a Multithreaded Process and Leaks File Descriptors

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / Service Supervisor | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-277. `trainer_service` runs its httplib admin proxy on other threads, so
`start()`'s `fork()` happens in a multithreaded process. The child calls only `execvp()`, but that
isn't on POSIX's async-signal-safe list (its PATH search may allocate), a theoretical post-fork
deadlock. The child also inherits every descriptor not marked close-on-exec: httplib marks its
listening socket, but accepted connections use plain `accept()`, so an admin connection in flight at
launch time stays open in the pass until it exits.

Action Items:

- [ ] Use `posix_spawn()`/`posix_spawnp()` (also addresses TD-278), with an absolute path resolved before spawning.
- [ ] Close inherited descriptors in the child (`close_range(3, ~0U, CLOSE_RANGE_CLOEXEC)` where available, or spawn file actions).

Location in code: `src/ChildProcess.cpp` (`start()`); tagged `TODO: See TD-281`.

Files to Modify:

- `src/ChildProcess.cpp`

---

### TD-282: `ChildProcess` Windows Path Has Known Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / Service Supervisor | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-277; affects only Windows builds (none shipped today):

- command-line quoting escapes `"` but not backslashes preceding a quote, so arguments ending in `\` break;
- a child that genuinely exits with code 259 (`STILL_ACTIVE`) is treated as still running forever;
- `CTRL_BREAK_EVENT` only reaches a child sharing the supervisor's console, otherwise the fallback is an immediate `TerminateProcess()`.

Action Items:

- [ ] Use the standard `CommandLineToArgvW`-compatible quoting algorithm.
- [ ] Detect exit via `WaitForSingleObject(handle, 0)` instead of comparing to `STILL_ACTIVE`.
- [ ] Document (or replace) the console-event stop mechanism.

Location in code: `src/ChildProcess.cpp` (`#ifdef _WIN32` paths); tagged `TODO: See TD-282`.

Files to Modify:

- `src/ChildProcess.cpp`
---

### TD-283: `SIGHUP` Terminates Every Binary Except `chatbot_api_server`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Config | October 9, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [Config.md](../reference/source/Config.md). CLAUDE.md says "Client binaries
(`chatbot_api_server`, `incremental_trainer`) additionally hot-reload their file via `SIGHUP`." Only
`chatbot_api_server` registers a `SIGHUP` handler. `incremental_trainer`, `trainer_service`,
`metrics_api_server`, `mns_server` and `registry_server` register only `SIGTERM`/`SIGINT`, so
`SIGHUP` takes the default action and **terminates the process**. An operator following the docs
(`kill -HUP <trainer pid>`) kills a training pass mid-epoch; sending it to `trainer_service` kills the
supervisor and its admin API. Found by inspecting every `std::signal()` registration.

Action Items:

- [ ] Install a `SIGHUP` handler in every long-running binary: either a real reload (where `ConfigLoader::reload()` makes sense) or `SIG_IGN` with a log line saying reload isn't supported.
- [ ] Correct CLAUDE.md's "Configuration" paragraph (annotated with this TD for now) and the operations docs.
- [ ] Test: `SIGHUP` doesn't terminate each daemon.

Location in code: `src/IncrementalTrainingTool.cpp`, `src/TrainerServiceMain.cpp`, `src/TrainingMetricsAPIServer.cpp`, `src/ModelNameServiceServer.cpp`, `src/RegistryServer.cpp` (signal registration); CLAUDE.md; tagged `TODO: See TD-283`.

Files to Modify:

- the five binaries above
- `CLAUDE.md`
- `docs/operations/...` (any reload instructions)

---

### TD-284: `registry_server` Ignores the FTP/FTPS Security Keys Its Config File Ships With

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Config | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-283. `registry_server` reads only `FTP_TOKEN_TTL_MINUTES` and
`FTP_MAX_SESSIONS_PER_RUN` from its config file (`RegistryServerArgs.cpp`). The FTP listener settings
are **CLI-only** (`--ftp-port`, `--ftp-pasv-min/max`, `--ftp-secret`, `--ftps`, `--ftp-cert`,
`--ftp-key`), per `RegistryServer.cpp`'s own "CLI-only listener settings" comment. Yet
`config.registry.conf` ships `FTP_SERVER_PORT`, `FTP_DATA_SERVER_SECRET=change-me-in-production` and
`FTPS_ENABLED`, and `ConfigLoader` parses them without a word. An operator who sets
`FTPS_ENABLED=true` or a real HMAC secret in the file silently gets **plaintext FTP and random token
passwords**. Nothing in `src/` reads `ServiceConfig::ftp_data_server_secret`, and the PASV-range
fields are also unread.

Action Items:

- [ ] Make `registry_server` honour the file keys (CLI still overriding), or remove them from `ServiceConfig` and `config.registry.conf`.
- [ ] Warn at startup when file and CLI disagree, and when FTPS is off on a non-loopback advertise IP.
- [ ] Until fixed, the shipped config file carries a note (annotated with this TD).

Location in code: `src/Config.{hpp,cpp}` (FTP keys), `src/RegistryServer.cpp`, `src/RegistryServerArgs.cpp`, `config.registry.conf`; tagged `TODO: See TD-284`.

Files to Modify:

- `src/RegistryServer.cpp`
- `src/RegistryServerArgs.{hpp,cpp}`
- `src/Config.{hpp,cpp}`
- `config.registry.conf`
- `tests/registry_server_args_test.cpp`

---

### TD-285: `ConfigLoader::reload()` Reports Success Without Applying Untracked Changes

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Config | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-283 and confirmed by running the real loader. `reload()` returns `true`
**without applying the new config** whenever `detect_changes()` finds nothing, but `detect_changes()`
compares only about 25 of roughly 110 fields (paths, port, session timeout, logging, architecture, six
generation fields). Changing only other keys (`METRICS_SERVER_URL`, `AUTO_SAVE_*`, RAG, GPU, MNS,
hippocampal…) logs "No configuration changes detected" and keeps the old values. Verified with
`METRICS_SERVER_URL` and `AUTO_SAVE_EVERY_MINUTES`. Conversely, if any tracked field changes, the
whole new config (untracked fields included) is applied, but only tracked changes are logged.

Action Items:

- [ ] Always apply the validated new config (or compare every field, ideally via a generated field list) and log every change.
- [ ] Separately report which changed keys the calling binary can't apply live (pairs with TD-251).
- [ ] Test: an untracked-only change is applied and logged.

Location in code: `src/Config.cpp` (`reload()`, `detect_changes()`); tagged `TODO: See TD-285`.

Files to Modify:

- `src/Config.{hpp,cpp}`
- `tests/config_test.cpp`

---

### TD-286: `ConfigLoader` Parses Booleans Two Different Ways

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Config | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-283 and confirmed by running the real loader. Eight file keys accept
`true/1/yes/on` case-insensitively; seven others (`GRADIENT_CLIP_ADAPTIVE`, `ENABLE_METRICS_SERVICE`,
`METRICS_ENABLE_PERSISTENCE`, `METRICS_ENABLE_PROMETHEUS`, `METRICS_API_ALLOW_CONTROL`,
`ENABLE_GENERATION_QUALITY_METRICS`, `FTPS_ENABLED`) accept only lowercase `true/1/yes`. Any other
spelling is silently **false**, with no warning: `ENABLE_METRICS_SERVICE=On` and
`METRICS_API_ALLOW_CONTROL=True` both turn the feature off. The env parser is stricter (warns on
unknown values), except `FTPS_ENABLED`, which bypasses it.

Action Items:

- [ ] Use one shared boolean parser (the env one: case-insensitive, true/false sets, warn on anything else) for every file and env key.
- [ ] Test each boolean key with `True`/`ON`/`Yes`.

Location in code: `src/Config.cpp` (`load_from_file()`, `load_from_env()` `FTPS_ENABLED`); tagged `TODO: See TD-286`.

Files to Modify:

- `src/Config.cpp`
- `tests/config_test.cpp`

---

### TD-287: `ConfigLoader` Accepts Negative and Partial Numbers

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Config | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-283 and confirmed by running the real loader. File values are parsed with
unchecked `std::stoi`/`stof`/`stoull`/`stoul`: `PORT=8080x` gives 8080 (trailing garbage dropped),
and `size_t` fields wrap negatives, so `D_MODEL=-1` gives **18446744073709551615**. Nothing
range-checks at load time (TD-241), so these reach model construction and port binding directly.

Action Items:

- [ ] Use full-consumption parsing that rejects negatives for `size_t` fields; warn and keep the default on any rejection.
- [ ] Tests for trailing garbage and negative sizes.

Location in code: `src/Config.cpp` (`load_from_file()`, `get_env_*()` helpers); tagged `TODO: See TD-287`.

Files to Modify:

- `src/Config.cpp`
- `tests/config_test.cpp`

---

### TD-288: `ConfigLoader::validate()` Misses Fields That Crash or Misbehave

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Config | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-283 and confirmed by running the real `validate()`: it accepts
`gradient_accumulation_steps = 0` (`ChatbotTrainer` divides by it and takes a modulo by it),
`learning_rate = -1` and `gpu_memory_fraction = 7.5`. It also never checks `batch_size`,
`num_epochs`, `weight_decay`, any adaptive-clipping field, `hippocampal_association_decay`/
`cross_reference_alpha`, any metrics/registry/FTP/MNS/trainer-admin port, interval or size,
`FTP_PASV_PORT_MIN ≤ MAX`, or auto-save values. And `validate()` itself only runs on reload (TD-241).

Action Items:

- [ ] Add range checks for every numeric field used for division, allocation, ports or probabilities.
- [ ] Run `validate()` at startup in every binary (TD-241 covers `chatbot_api_server`).
- [ ] Tests for each new rule.

Location in code: `src/Config.cpp` (`validate()`); tagged `TODO: See TD-288`.

Files to Modify:

- `src/Config.cpp`
- `tests/config_test.cpp`

---

### TD-289: `ConfigLoader` Keeps Inline Comments and Silently Accepts Enum Typos

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Config | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-283 and confirmed by running the real loader. `#` is a comment only at the start
of a line, so `MODEL_PATH=/models/chat # production model` stores the comment as part of the path.
`TOKENIZER_MODE` treats anything but `unicode` (e.g. the typo `utf8`) as ASCII without a warning;
`STRATEGY`, `LOG_LEVEL`, `METRICS_STORAGE_BACKEND` and `DATASET_KIND` aren't checked at load either.

Action Items:

- [ ] Strip an unquoted ` #…` suffix (or document that inline comments aren't supported, and warn when a value contains ` #`).
- [ ] Warn on unknown enum values for `TOKENIZER_MODE`, `STRATEGY`, `LOG_LEVEL`, `METRICS_STORAGE_BACKEND`, `DATASET_KIND`.

Location in code: `src/Config.cpp` (`load_from_file()`); tagged `TODO: See TD-289`.

Files to Modify:

- `src/Config.cpp`
- `tests/config_test.cpp`

---

### TD-290: `ConfigLoader` Output Hygiene

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Config | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-283:

- Load-time messages go to `std::cerr` (partly unavoidable before the logger starts), and the header's
  inline `gpu_strategy_from_string()` writes to `std::cerr` too.
- `print()` writes to `std::cout` and covers only server, architecture, generation and RAG settings
  (no metrics, registry, MNS, GPU, training or world-model sections).
- `HIPPOCAMPAL_COVERAGE_LOSS_WEIGHT` ships in `config.chatbot.conf`/`config.trainer.conf`, is
  deliberately unread, and produces an "Unknown configuration key" warning on every load.

Action Items:

- [ ] Buffer pre-logger messages and replay them through `adai::Logger` once initialised.
- [ ] Extend `print()` (or a `to_string()` used by every binary) to all sections, still redacting secrets.
- [ ] Accept reserved keys silently (or remove the key from the shipped files).

Location in code: `src/Config.{hpp,cpp}`; tagged `TODO: See TD-290`.

Files to Modify:

- `src/Config.{hpp,cpp}`
- `config.chatbot.conf`
- `config.trainer.conf`
---

### TD-291: Conversation Imports Can Lift Session Limits and Budgets

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | NLP / ConversationContext | October 9, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [ConversationContext.md](../reference/source/ConversationContext.md), and confirmed by running the real
class. `ConversationContext::deserialize()` **replaces** the instance's `max_messages`/`max_tokens`
with the imported `MAX_MESSAGES`/`MAX_TOKENS` lines, and trusts each message's stored `token_count`:

- `MAX_MESSAGES:0`/`MAX_TOKENS:0` make the session **unlimited**. 500 imported messages were all
  kept (a 2.5 MB prompt), and the session stayed unlimited for later messages.
- A token count of `1` on huge messages, or a **negative** count (`-100000` made `total_tokens`
  negative), defeats the token budget.

`chatbot_api_server` exposes this as `POST /chat/session/import` (TD-053), so any client can give its
session unbounded history, memory and prompt size, overriding `ChatbotAPI`'s `Session(10, 2048)`
limits, and combined with TD-232 it can do so across unlimited sessions.

Action Items:

- [ ] In `deserialize()` (or an import-specific overload), keep the caller's limits instead of adopting imported ones; ignore or clamp imported metadata.
- [ ] Recompute token counts on import (ignore stored counts), rejecting negative values; enforce a maximum import size in `ChatbotAPI`.
- [ ] Tests: imported limits and token counts can't exceed the receiving session's limits.

Location in code: `src/ConversationContext.cpp` (`deserialize()`), `src/ChatbotAPI.cpp` (`handle_import_session()`); tagged `TODO: See TD-291`.

Files to Modify:

- `src/ConversationContext.{hpp,cpp}`
- `src/ChatbotAPI.cpp`
- `tests/conversationcontext_test.cpp`
- `tests/chatbotapi_test.cpp`

---

### TD-292: One Long Message Wipes the Whole Conversation

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | NLP / ConversationContext | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-291 and confirmed by running the real class. `truncate_to_limits()` evicts
the oldest messages until under `max_tokens`, keeping a lone remaining message only if it's
within 1.2× the budget. The **newest** message isn't protected, so a message that alone exceeds
1.2 × `max_tokens` evicts **everything, including itself**. With `ChatbotAPI`'s 2048-token session
budget, a 12,000-character user message left 0 messages. `handle_chat_session()` then formats an
empty prompt, `generate_response("")` fails (`BPETokenizer::encode("")` throws), the client gets
HTTP 500, and the conversation history is gone.

Action Items:

- [ ] Never evict the message just added: truncate it (e.g. keep its tail) to fit, or reject it with a clear error before touching history.
- [ ] In `ChatbotAPI`, reject over-long messages with a 4xx before adding them.
- [ ] Test: an oversized message leaves earlier history intact and produces a clear error or truncated content.

Location in code: `src/ConversationContext.cpp` (`truncate_to_limits()`); tagged `TODO: See TD-292`.

Files to Modify:

- `src/ConversationContext.cpp`
- `src/ChatbotAPI.cpp`
- `tests/conversationcontext_test.cpp`

---

### TD-293: A Malformed Conversation Import Wipes the Live Session

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | NLP / ConversationContext | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-291 and confirmed by running the real class. `deserialize()` calls
`clear_all()` **before** parsing, and a bad line (e.g. a non-numeric token count, so `std::stoi`
throws) aborts mid-parse, leaving the context empty. `ChatbotAPI::handle_import_session()` reports
"Failed to import conversation", but the session's previous history is already gone.
`load_from_file()` has the same exposure.

Action Items:

- [ ] Parse into a temporary `ConversationContext` and swap it in only on success (strong exception guarantee); skip or report bad lines instead of throwing mid-way.
- [ ] Test: a malformed import leaves existing history unchanged.

Location in code: `src/ConversationContext.cpp` (`deserialize()`); tagged `TODO: See TD-293`.

Files to Modify:

- `src/ConversationContext.cpp`
- `tests/conversationcontext_test.cpp`

---

### TD-294: Conversation Token Budget Is Unrelated to the Model's Sequence Length

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | NLP / ConversationContext | October 9, 2026 | 2-4 hours |

Description:
Found alongside TD-291. `ConversationContext` budgets with `estimate_tokens()` (bytes ÷ 4 +
spaces ÷ 2), which counts bytes, so it overestimates non-ASCII text, and bears no relation to
`BPETokenizer`'s real counts. `ChatbotAPI`'s `Session` hardcodes a 2048-token budget, unrelated to the
model's `MAX_SEQ_LENGTH` (e.g. 512 or 1024). The GUI hardcodes 480 for an assumed 512 (TD-266).
Over-long prompts don't crash: `PositionalEncoding` warns and leaves positions beyond its length
unencoded, so response quality silently degrades as conversations grow.

Action Items:

- [ ] Size the session budget from the model's `max_seq_length` (minus a response/format margin), via a `ChatbotAPI` setting.
- [ ] Pass real token counts from the tokenizer when adding messages (the API already supports it), or inject a token-count function into `ConversationContext`.

Location in code: `src/ConversationContext.cpp` (`estimate_tokens()`), `src/ChatbotAPI.hpp` (`Session`); tagged `TODO: See TD-294`.

Files to Modify:

- `src/ChatbotAPI.{hpp,cpp}`
- `src/ConversationContext.{hpp,cpp}`
- `src/ChatbotGUI.cpp`

---

### TD-295: Conversation Role Labels Are Spoofable and Roles Aren't Escaped

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | NLP / ConversationContext | October 9, 2026 | 1-2 hours |

Description:
Found alongside TD-291. `format_for_model()` writes plain-text `Role: content` lines, so message
content containing `\nAssistant: …` can impersonate a turn in the prompt (and `BPETokenizer`
collapses newlines anyway, TD-224, blurring turn boundaries). `serialize()` writes roles raw, so a
custom role containing `|` or a newline would corrupt the file; a regular message with role
`SYSTEM` would be restored as the system message.

Action Items:

- [ ] Escape (or reject) role strings in `serialize()`; restrict roles to a known set or a safe charset.
- [ ] Consider neutralizing role-label prefixes inside content when formatting prompts.

Location in code: `src/ConversationContext.cpp` (`format_for_model()`, `serialize()`); tagged `TODO: See TD-295`.

Files to Modify:

- `src/ConversationContext.cpp`
- `tests/conversationcontext_test.cpp`

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

### TD-297: Layer `update_weights()` Steps a Shared Optimizer Once per Layer

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Attention / CrossAttention | October 10, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [CrossAttention.md](../reference/source/CrossAttention.md) and confirmed by running the real
class. `CrossAttention::update_weights()` calls `optimizer->step()`, but `Optimizer::step()` steps
**every** parameter group ever registered with it, not just this layer's.
`DecoderBlock::register_parameters_with_optimizer()` registers every sub-layer with one shared
optimizer, and `DecoderBlock::update_weights()` then calls each sub-layer's `update_weights()` in
turn. With two `CrossAttention`s sharing an SGD optimizer, the second layer's weights moved a full
step on the first layer's call and again on its own: two steps. A full decoder block makes 6–10
`step()` calls per update (each also advancing Adam's step counter). `MultiHeadAttention`,
`FeedForward`, `LayerNorm`, `TokenEmbedding` and `LanguageModelHead` follow the same pattern.
Calling `set_optimizer()` twice with the same optimizer also registers the groups twice (verified:
8 groups instead of 4), doubling every update.

The shipped trainers (`ChatbotTrainer`, `RLHFTrainer`) call `optimizer->step()` once directly and
are unaffected, but the public `EncoderDecoderModel::train_step()`/`update_weights()` path
over-steps whenever an optimizer is registered. `LeJEPAEncoder` hit and fixed exactly this
(TD-178).

Action Items:

- [ ] Apply TD-178's fix at the model level: when an optimizer is registered, `EncoderDecoderModel`/`LLMDecoder`/`DecoderBlock` `update_weights()` call `step()` once at the top and only zero grads below; or have layers skip `step()` when the optimizer is shared.
- [ ] Make `set_optimizer()`/`register_parameters()` idempotent for the same optimizer (or dedupe in `Optimizer::add_parameter_group()`).
- [ ] Test: two layers on one optimizer each move by exactly one step per `EncoderDecoderModel::update_weights()`.

Location in code: `src/CrossAttention.cpp` (`update_weights()`, `register_parameters()`), `src/DecoderBlock.cpp` (`update_weights()`); tagged `TODO: See TD-297`.

Files to Modify:

- `src/CrossAttention.cpp`
- `src/DecoderBlock.cpp`
- `src/Decoder.cpp`
- `src/EncoderDecoderModel.cpp`
- `src/Optimizer.cpp`
- `tests/encoderdecoder_test.cpp`

---

### TD-298: `CrossAttention::load()` Loads Truncated Files as NaN Weights

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Attention / CrossAttention | October 10, 2026 | 1-2 hours |

Description:
Found alongside TD-297 and confirmed by running the real class. `load()` pre-sets each value to
`NAN` and never checks the stream, so a file cut off after `W_k` loaded **without error** with all of
`W_v`/`W_o` NaN. `learning_rate` is read into the member before the dimension check, so a rejected
(mismatched) file still changes the layer's learning rate. `save()` doesn't check its writes. This
is the legacy per-block `<block>.cross_attn` format written by `DecoderBlock::save()`; the main
checkpoint path is SafeTensors (`ModelSerializer`).

Action Items:

- [ ] Check the stream after every read (and the file size up front); throw on short reads.
- [ ] Read the header into locals and assign `learning_rate` only after validation; check `save()`'s stream before returning.
- [ ] Test: truncated and empty files throw and leave the layer unchanged.

Location in code: `src/CrossAttention.cpp` (`load()`, `save()`); tagged `TODO: See TD-298`.

Files to Modify:

- `src/CrossAttention.cpp`
- `tests/crossattention_test.cpp`

---

### TD-299: `CrossAttention` Crashes with SIGFPE When `num_heads` Is Zero

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Attention / CrossAttention | October 10, 2026 | 1 hour |

Description:
Found alongside TD-297 and confirmed by running the real class. The constructor validates
`d_model % num_heads` in its body, but the initializer list has already computed
`d_k(d_model / num_heads)`, so `CrossAttention(8, 0)` kills the process with SIGFPE instead of
throwing. A negative `num_heads` that divides `d_model` is accepted with a negative `d_k`.
`mns_server` rejects 0 at register time and `ConfigLoader::validate()` rejects values outside 1–64,
but `chatbot_api_server` doesn't run `validate()` at startup (TD-241), so a local `NUM_HEADS=0` with
no MNS crashes the process. `MultiHeadAttention` should be checked for the same pattern.

Action Items:

- [ ] Validate `d_model > 0` and `num_heads > 0` before computing `d_k` (e.g. via a static helper in the initializer list).
- [ ] Test: `num_heads` of 0 and negative values throw `std::invalid_argument`.

Location in code: `src/CrossAttention.cpp` (constructor); tagged `TODO: See TD-299`.

Files to Modify:

- `src/CrossAttention.cpp`
- `src/MultiHeadAttention.cpp`
- `tests/crossattention_test.cpp`

---

### TD-300: `CrossAttention::forward_with_cache()` Silently Reuses Stale Encoder K/V

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Attention / CrossAttention | October 10, 2026 | 1-2 hours |

Description:
Found alongside TD-297 and confirmed by running the real class. Once the `KVCache` is filled,
`forward_with_cache()` ignores `kv_input` entirely. Reusing a cache with a different encoder output
silently returned attention over the **old** encoder's K/V (identical to the first call), and a mask
sized for the new encoder was rejected with a confusing `src_len` error. Enabling or changing LoRA
after the cache is filled also leaves stale K/V. Production is safe today because
`EncoderDecoderModel` builds a fresh `DecoderKVCache` per generation, but nothing documents or checks
the contract.

Action Items:

- [ ] Check `kv_input.rows` against the cached K rows (when non-empty) and throw on mismatch; document that callers must `clear()` the cache for a new encoder output.
- [ ] Test: a cache reused with a different-length encoder output throws.

Location in code: `src/CrossAttention.cpp` (`forward_with_cache()`); tagged `TODO: See TD-300`.

Files to Modify:

- `src/CrossAttention.{hpp,cpp}`
- `tests/crossattention_test.cpp`

---

### TD-301: `CrossAttention` GPU Path Diverges from the CPU Path

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Attention / CrossAttention | October 10, 2026 | 2-4 hours |

Description:
Found alongside TD-297, by inspection (no GPU in the environment it was traced in). The
`ADAI_ENABLE_GPU` path:
- ignores LoRA adapters entirely, with no guard or warning (the header notes it);
- does no shape validation on `query`/`kv` widths or mask dimensions (`masked_fill_inplace` uses only
  the element count);
- holds weight snapshots that go stale after any CPU-side change (`update_weights()`, `load()`,
  `set_W*()`, `merge_lora()`) until the owner calls `EncoderDecoderModel::gpu_sync_weights()`;
- doesn't refresh `get_last_attention_weights()`.

`DecoderBlock`'s GPU path also runs only the main cross-attention; the world-model and hippocampal
instances are CPU-only.

Action Items:

- [ ] Throw or warn in `gpu_forward()`/`gpu_forward_with_cache()` when `has_lora()`; add the same shape checks as the CPU path.
- [ ] Document (or track) the snapshot contract; consider a dirty flag set by CPU-side weight changes.

Location in code: `src/CrossAttention.cpp` (`gpu_forward()`, `gpu_forward_with_cache()`); tagged `TODO: See TD-301`.

Files to Modify:

- `src/CrossAttention.{hpp,cpp}`
- `src/MultiHeadAttention.cpp`

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

### TD-304: Session Counters Can Be Duplicated and `DaemonConfigStore` Has No Busy Timeout

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Core / DaemonConfigStore | October 11, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [DaemonConfigStore.md](../reference/source/DaemonConfigStore.md) and confirmed by running the real
class. `DaemonConfigStore` never sets `sqlite3_busy_timeout()`, so any write while another
connection holds the write lock fails immediately: with a second connection in a write
transaction, `set()` threw `database is locked` after 0.1 ms.

`LocalTransport::next_session()` (standalone trainers, no `REGISTRY_SERVER_URL`) allocates session
numbers as `load_all()` → `+1` → `set()` with no transaction and no lock. Two processes each doing
200 such increments against one `session_counters.db`:
- 284 of 400 calls **threw** (`database is locked`);
- the counter ended at **113** although 116 calls reported success, so 3 increments were lost (two
  callers handed the same session number).

`IncrementalTrainer::begin_run()` catches the throw and continues with an **empty** session ID.
`registry_server`'s `handle_session_next` is safe within one process (it holds
`g_session_store_mtx`) but would race the same way across processes. `trainer_service` and its child
also share `TRAINER_ADMIN_DIR/daemon_config.db`, so a write overlapping a child's start or exit
fails instead of waiting.

Action Items:

- [ ] Set `sqlite3_busy_timeout()` (e.g. 5 s) in the constructor.
- [ ] Add an atomic `increment(key)` (`BEGIN IMMEDIATE` read-modify-write, or `UPDATE … RETURNING`) and use it in both `next_session()` implementations.
- [ ] Tests: two connections writing concurrently both succeed; concurrent increments are never lost or duplicated.

Location in code: `src/DaemonConfigStore.cpp` (constructor, `set()`), `src/RegistryTransport.cpp` (`LocalTransport::next_session()`), `src/RegistryServer.cpp` (`handle_session_next`); tagged `TODO: See TD-304`.

Files to Modify:

- `src/DaemonConfigStore.{hpp,cpp}`
- `src/RegistryTransport.cpp`
- `src/RegistryServer.cpp`
- `tests/daemonconfigstore_test.cpp`

---

### TD-305: A Rejected or Failed `PUT /admin/config` Still Partly Applies

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Admin config API | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-304, by inspection. Two problems:

- **Trainer PUTs apply keys before validating later ones.** Both `TrainerAdminAPI::handle_put_config()`
  and `TrainerServiceProxy::handle_put_config_idle()` apply the body key by key and return 400 at the
  first invalid value. `{"auto_save_enabled":false,"auto_save_every_samples":-5}` is rejected with 400,
  but auto-save is already off:
  - in a live child, in memory only, so live state and `daemon_config.db` disagree;
  - in the idle proxy, already written to the DB, so the next pass starts with auto-save off.
- **Multi-key persistence isn't atomic in any daemon.** `TrainerAdminAPI`, `ModelNameService`,
  `TrainingMetricsAPI` and `registry_server` call `DaemonConfigStore::set()` once per key with no
  transaction, after the in-memory state is already updated. If one `set()` throws (e.g. locked,
  TD-304), earlier keys are persisted and later ones aren't, and the exception escapes the handler,
  so the client sees an error for a change that partly applied and will partly revert on restart.

Action Items:

- [ ] Validate every key in the body before applying or persisting any of them.
- [ ] Add `DaemonConfigStore::set_many()` (one transaction) and persist before (or atomically with) updating in-memory state; return a clear 500 when persistence fails.
- [ ] Tests: a body with one invalid key changes nothing; a persistence failure leaves memory and DB consistent.

Location in code: `src/TrainerAdminAPI.cpp` (`handle_put_config()`), `src/TrainerServiceProxy.cpp` (`handle_put_config_idle()`), `src/DaemonConfigStore.hpp`; tagged `TODO: See TD-305`.

Files to Modify:

- `src/DaemonConfigStore.{hpp,cpp}`
- `src/TrainerAdminAPI.cpp`
- `src/TrainerServiceProxy.cpp`
- `src/ModelNameService.cpp`
- `src/TrainingMetricsAPI.cpp`
- `src/RegistryServer.cpp`

---

### TD-306: `registry_server` and `metrics_api_server` Admin PUTs Don't Range-Check Values

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Admin config API | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-304, by inspection. `registry_server`'s `PUT /admin/config` accepts any integer
for `ftp_token_ttl_minutes`/`ftp_max_sessions_per_run`, and `metrics_api_server`'s accepts any
integer for its nine tunables. Negative values are applied and persisted; a negative
`max_live_sessions` is cast to `size_t` and becomes enormous. The trainer PUTs already validate their
ranges.

Action Items:

- [ ] Add range checks matching each setting's meaning and return 400 for out-of-range values (validating all keys first, per TD-305).

Location in code: `src/RegistryServer.cpp` (admin PUT handler), `src/TrainingMetricsAPI.cpp` (admin PUT handler); tagged `TODO: See TD-306`.

Files to Modify:

- `src/RegistryServer.cpp`
- `src/TrainingMetricsAPI.cpp`

---

### TD-307: `DaemonConfigStore::load_all()` Hides Read Errors

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Core / DaemonConfigStore | October 11, 2026 | 1 hour |

Description:
Found alongside TD-304, by inspection. `load_all()` stops at the first non-`SQLITE_ROW` step result
without checking whether it was an error, so a mid-scan failure returns a partial map with no
warning. The `PRAGMA journal_mode=WAL` result is ignored: on filesystems without shared memory,
SQLite falls back to rollback journalling where readers can be blocked, so a daemon's startup
overlay can silently drop persisted overrides.

Action Items:

- [ ] Check the final step result and warn (or throw) on errors; log when WAL couldn't be enabled.

Location in code: `src/DaemonConfigStore.cpp` (constructor, `load_all()`); tagged `TODO: See TD-307`.

Files to Modify:

- `src/DaemonConfigStore.cpp`

---

### TD-308: Session Counters Reuse the Config Store's Table and Scan It per Allocation

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Core / DaemonConfigStore | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-304, by inspection. `registry_server` and `LocalTransport` keep session counters
in `session_counters.db` through `DaemonConfigStore`, so the file contains a table named
`daemon_config` holding counters. Each allocation reads **every** counter ever stored via
`load_all()` to look up one key, and the table gains a row per run forever. It's cheap at today's
scale but confusing and O(runs) per allocation.

Action Items:

- [ ] Add `get(key)` (or fold this into TD-304's `increment(key)`), and consider a dedicated counter class or table name.

Location in code: `src/RegistryTransport.cpp` (`LocalTransport::next_session()`), `src/RegistryServer.cpp` (`handle_session_next`); tagged `TODO: See TD-308`.

Files to Modify:

- `src/DaemonConfigStore.{hpp,cpp}`
- `src/RegistryTransport.cpp`
- `src/RegistryServer.cpp`
---

### TD-309: Gutenberg Sentence Splitting Can Crash `registry_server`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Data / DataFetcher | October 11, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [DataFetcher.md](../reference/source/DataFetcher.md) and confirmed by running the real
code. `DataFetcher::extract_sentences()` matches `std::regex("[^.!?]+[.!?]+")` over the whole cleaned
book. libstdc++'s regex engine recurses per character, so a long run with no `.`, `!` or `?` overflows
the stack: 20,000 characters was fine, but **100,000 and 400,000 characters died with SIGSEGV**.
Realistic triggers include a book in a script without ASCII sentence punctuation (e.g. Chinese, which
uses `。`), a long poem, or a large index or table.

`registry_server`'s `POST /registry/<group>/fetch/gutenberg` runs `ensure_gutenberg_cached()` →
`extract_sentences()` in its request thread, so **one fetch request kills the registry daemon**. No
sentence cache is written on the way, so every retry of the same book crashes it again.
`dataset_manager`'s local `gutenberg` command crashes the same way.

Action Items:

- [ ] Replace the regex with a linear hand-written scanner (also the natural place for TD-311's abbreviation handling).
- [ ] Cap the length of a candidate sentence span, and treat a book with no usable sentences as a clean failure.
- [ ] Test: a 1 MB unpunctuated input returns without crashing.

Location in code: `src/DataFetcher.cpp` (`extract_sentences()`); tagged `TODO: See TD-309`.

Files to Modify:

- `src/DataFetcher.cpp`
- `tests/DataFetcherTests.cpp`

---

### TD-310: HuggingFace `split` Can Write Outside the Registry Data Directory

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Data / DataFetcher | October 11, 2026 | 1 hour |

Description:
Found alongside TD-309 and confirmed by running the real code. `registry_server`'s
`fetch/huggingface` allow-lists `dataset_id` and `split` to `[A-Za-z0-9._-/]+`, which permits `..`
and `/`. `DataFetcher` flattens `/` in `dataset_id` but concatenates `split` into the cache path
as-is (`<dir>/<safe_id>_<split>/`), and the registry builds the training-file path from `split` the
same way. `split = "x/../../../../ESCAPED"` made `ensure_huggingface_cached()` create `ESCAPED/`
outside the group's data directory. With a real download, `parquet_info.json`, `parquet/` and
`full_dataset.jsonl` would be written there, and the produced training file and pending entry would
point there too. Any client that can reach the registry's fetch endpoint can do this.

Action Items:

- [ ] Reject `..` path components (or any `/`) in `split` in the registry allow-list, and flatten `split` in `DataFetcher` like `dataset_id`.
- [ ] Test: the endpoint rejects a `split` containing `..`, and `DataFetcher` never creates paths outside its output dir.

Location in code: `src/DataFetcher.cpp` (`ensure_huggingface_cached()`, `fetch_huggingface()`), `src/RegistryServer.cpp` (`kSafeHfIdentifier`); tagged `TODO: See TD-310`.

Files to Modify:

- `src/DataFetcher.cpp`
- `src/RegistryServer.cpp`
- `tests/dataset_registry_live_test.cpp`

---

### TD-311: Gutenberg Training Pairs Are Next-Sentence Non-Sequiturs

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / DataFetcher | October 11, 2026 | 4-8 hours |

Description:
Found alongside TD-309 and confirmed by running the real code. `create_qa_pairs_from_text()` pairs
`"<template>" + s[i]` with `s[i+1]` and `"Summarize: " + s[i] + s[i+1]` with `s[i+2]`. The
"answer" is just whatever the author wrote next, and the "summary" is the following sentence, so the
model is trained to answer "Can you explain: X" with an unrelated continuation. `extract_sentences()`
also splits on abbreviations: `Mr.`, `Mrs.` and `Dr.` end a "sentence", so `"Mr."` is dropped and
the rest is split mid-clause (verified on Pride-and-Prejudice-style text). The template comes from
unseeded `rand()` (same sequence every run), and every pair is tagged `language="en"` regardless of
the book. Every Gutenberg fetch produces this data.

Action Items:

- [ ] Decide the intended task (the user's call). Options: plain continuation/completion pairs labelled as such (`task_type="completion"`), real QA from a better source, or dropping the QA/summary templates.
- [ ] Handle common abbreviations in sentence splitting (fold into TD-309's scanner).
- [ ] Use a seeded RNG (or no templates), and set `language` from the book's metadata or leave it empty.

Location in code: `src/DataFetcher.cpp` (`create_qa_pairs_from_text()`, `generate_question_from_sentence()`, `extract_sentences()`); tagged `TODO: See TD-311`.

Files to Modify:

- `src/DataFetcher.cpp`
- `tests/DataFetcherTests.cpp`

---

### TD-312: `DataFetcher` Downloads Through Shell `curl` Commands

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / DataFetcher | October 11, 2026 | 4-8 hours |

Description:
Found alongside TD-309, by inspection. Every download is `std::system("curl -L -f -s … \"<path>\"
\"<url>\"")`. Inside double quotes, `$(…)` and backticks still expand.
- `registry_server` allow-lists `dataset_id`/`split`, so its endpoint isn't injectable (but see
  TD-310 for `..`).
- `dataset_manager`'s local mode passes them unvalidated (the operator's own shell, but a pasted ID
  containing `$(…)` runs it).
- Parquet URLs from HuggingFace's API response are interpolated unvalidated.
- `HF_TOKEN` is put on the curl command line, visible to other local users via `ps`.
- `curl` runs with no `--connect-timeout`/`--max-time`, synchronously inside `registry_server`'s
  request handler, so a stalled server holds a worker thread indefinitely.

Action Items:

- [ ] Download with libcurl (already an optional dependency for the FTP transport) or `posix_spawn` with an argv array, never a shell.
- [ ] Pass the token via a header option, not argv; set connect and total timeouts; validate URLs are `https://` on expected hosts.

Location in code: `src/DataFetcher.cpp` (`download_file()`, `download_hf_full_dataset()`); tagged `TODO: See TD-312`.

Files to Modify:

- `src/DataFetcher.cpp`
- `src/CMakeLists.txt`

---

### TD-313: HuggingFace Cache Accepts Partial Downloads and Races Concurrent Fetches

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / DataFetcher | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-309, by inspection. `download_hf_full_dataset()` logs and skips any Parquet part
that fails to download or convert, and succeeds if one part converted. `ensure_huggingface_cached()`
then treats any non-empty `full_dataset.jsonl` as complete, so the missing parts are never retried.
An interrupted download (a crash mid-append) likewise leaves a truncated cache that's served
indefinitely. Two concurrent `fetch/huggingface` requests for the same dataset (different models, or a
retry) download into the same `parquet/part_<i>.parquet` and `full_dataset.jsonl` at once, and the
second request may see the first's half-written cache as complete.

Action Items:

- [ ] Download into a temp directory and rename into place only when every part succeeds; write a completion marker and check it instead of file size.
- [ ] Serialize fetches per dataset in `registry_server` (a per-dataset mutex or lock file).

Location in code: `src/DataFetcher.cpp` (`download_hf_full_dataset()`, `ensure_huggingface_cached()`); tagged `TODO: See TD-313`.

Files to Modify:

- `src/DataFetcher.cpp`
- `src/RegistryServer.cpp`

---

### TD-314: `DataFetcher` Source Coverage and Hygiene Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / DataFetcher | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-309, by inspection:
- Only `config=default` is ever requested, so multi-config HuggingFace datasets fail with "No parquet
  files found" and there's no way to choose a config.
- Only Gutenberg's `files/<id>/<id>-0.txt` and `<id>.txt` URLs are tried, not
  `cache/epub/<id>/pg<id>.txt`; not checked against the live site.
- `fetch_gutenberg()` re-downloads the book on every call.
- `fetch_huggingface()`/`download_hf_full_dataset()` print with `std::cout`/`std::cerr` in library
  code.
- The JSON field scanners don't decode `\uXXXX` escapes and match the first `"key"` anywhere in a row,
  including nested objects.

Action Items:

- [ ] Add an optional HF `config` parameter (CLI, endpoint, dashboard).
- [ ] Add the `cache/epub` URL pattern; cache in `fetch_gutenberg()`; switch output to `Logger`.
- [ ] Parse rows with the vendored nlohmann/json instead of the string scanners.

Location in code: `src/DataFetcher.cpp` (`download_hf_full_dataset()`, `get_gutenberg_url()`, `fetch_gutenberg()`, `hf_extract_string()`); tagged `TODO: See TD-314`.

Files to Modify:

- `src/DataFetcher.{hpp,cpp}`
- `src/RegistryServer.cpp`
- `src/DatasetManagerTool.cpp`

---

### TD-321: `dataset_manager` Treats `--count` ≤ 0 as "All"

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / dataset_manager CLI | October 11, 2026 | 1-2 hours |

Description:
Found while writing the code-traced reference [DatasetManagerArgs.md](../reference/source/DatasetManagerArgs.md) and confirmed by running the real
parsers. `parse_assign_args()` and `parse_migrate_args()` accept any integer for `--count`, and a
value ≤ 0 with no file targets selects **everything**:
- `LocalTransport::assign()` and `registry_server`'s assign handler both treat "no paths and
  `count <= 0`" as `assign_all`.
- `dataset_manager migrate` copies `count` entries only if `count > 0`, otherwise all of them.

So `dataset_manager assign my-model --count -2` (or `--count 0`) assigns **every pending file** to
`my-model`, and `migrate chatbot --count 0` migrates every legacy entry. Assigning everything is
only documented as "omit `--count`", so a typo silently turns "first N" into "the whole pool".

Action Items:

- [ ] Reject `--count` values < 1 in both parsers (keep "omit `--count`" as the only way to say all).
- [ ] Optionally have the transports/server treat `count <= 0` with an explicit count parameter as an error rather than `assign_all`.
- [ ] Tests: `--count 0` and `--count -2` are usage errors.

Location in code: `src/DatasetManagerArgs.cpp` (`parse_assign_args()`, `parse_migrate_args()`); tagged `TODO: See TD-321`.

Files to Modify:

- `src/DatasetManagerArgs.cpp`
- `src/DatasetManagerTool.cpp`
- `tests/dataset_manager_args_test.cpp`

---

### TD-322: `dataset_manager` Treats Unknown Flags as File Targets

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / dataset_manager CLI | October 11, 2026 | 1 hour |

Description:
Found alongside TD-321 and confirmed by running the parsers. `assign`, `unassign`, `delete` and
`migrate` push every argument they don't recognize into `targets`:
- `delete f.jsonl --delete-file` (typo) deletes the entry but **keeps the file**, with no warning.
- `unassign m --froce` silently does nothing useful.
- A value-less `assign m --count` (or `--count=3`) targets a file named `--count`/`--count=3` and
  reports "no matching pending files" instead of a usage error.

Action Items:

- [ ] Reject any argument starting with `--` that isn't a known flag, and require a value after `--count`.
- [ ] Tests for typo'd flags and value-less `--count`.

Location in code: `src/DatasetManagerArgs.cpp` (`parse_assign_args()`, `parse_unassign_args()`, `parse_delete_args()`, `parse_migrate_args()`); tagged `TODO: See TD-322`.

Files to Modify:

- `src/DatasetManagerArgs.cpp`
- `tests/dataset_manager_args_test.cpp`

---

### TD-323: `dataset_manager` Accepts Partial and Out-of-Range Numbers

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / dataset_manager CLI | October 11, 2026 | 1 hour |

Description:
Found alongside TD-321 and confirmed by running the parsers. Every numeric argument uses bare
`std::stoi`, so trailing garbage is dropped (`--count 3abc` → 3, `gutenberg 1342abc 500x` → 1342,
500), and negative or zero values pass (`gutenberg -5 -10`, `huggingface x -1`,
`gutenberg-batch 1342,0`). The registry rejects a non-positive `book_id` later; in local mode
`DataFetcher` just writes zero pairs. Same class of bug as TD-240 and TD-287.

Action Items:

- [ ] Add a strict integer helper (full-string match, range check) and require `book_id > 0` and `num_pairs > 0`.
- [ ] Tests for partial and non-positive values.

Location in code: `src/DatasetManagerArgs.cpp` (all `std::stoi` call sites); tagged `TODO: See TD-323`.

Files to Modify:

- `src/DatasetManagerArgs.cpp`
- `tests/dataset_manager_args_test.cpp`

---

### TD-324: `dataset_manager segment` Range Validation Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / dataset_manager CLI | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-321. Confirmed by running the parser:
- `--ranges 0-2147483647` overflows `b - a + 1` (UBSan: signed integer overflow), producing a
  segment of -2147483648 pairs.
- Overlapping ranges (`0-9,5-14`) and a repeated `--ranges` are accepted, so the same pairs are
  queued (and trained) twice.
- `--count -5 --ranges 0-1` is accepted, `--count` silently ignored (the exclusivity check uses
  `split_count > 0`).

By inspection: `DatasetManagerTool.cpp` checks the file's pair count but never checks explicit
ranges against it, so a range past the end is queued.

Action Items:

- [ ] Compute the count in 64-bit (or reject `B` near `INT_MAX`); reject overlapping/duplicate ranges and repeated `--ranges`; reject `--count` < 1.
- [ ] In `DatasetManagerTool.cpp`, reject ranges that extend past `count_pairs()`.

Location in code: `src/DatasetManagerArgs.cpp` (`parse_segment_args()`), `src/DatasetManagerTool.cpp` (`segment`); tagged `TODO: See TD-324`.

Files to Modify:

- `src/DatasetManagerArgs.cpp`
- `src/DatasetManagerTool.cpp`
- `tests/dataset_manager_args_test.cpp`

---

### TD-325: `dataset_manager huggingface` Positional Arguments Shift Silently

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / dataset_manager CLI | October 11, 2026 | 1 hour |

Description:
Found alongside TD-321, by inspection. `huggingface <dataset_id> [num_pairs] [split]
[input_field] [output_field] [model]` is purely positional, so giving `model` requires every
argument before it, and `huggingface ds 500 my-model` silently sets `split = "my-model"`. `add`,
`remove` and `gutenberg` also ignore extra arguments rather than rejecting them.

Action Items:

- [ ] Add named flags (`--split`, `--input-field`, `--output-field`, `--model`), keeping the positional form for compatibility; reject unexpected extra arguments.

Location in code: `src/DatasetManagerArgs.cpp` (`parse_huggingface_args()`, `parse_add_args()`, `parse_remove_args()`, `parse_gutenberg_args()`); tagged `TODO: See TD-325`.

Files to Modify:

- `src/DatasetManagerArgs.cpp`
- `src/DatasetManagerTool.cpp`
- `tests/dataset_manager_args_test.cpp`

---

### TD-326: `dataset_manager migrate` Duplicates Segmented Data Across Pools

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Data / dataset_manager CLI | October 11, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [DatasetManagerTool.md](../reference/source/DatasetManagerTool.md) and confirmed by running the real
binary against a scratch local pool. `migrate` copies each legacy entry with
`dest.add_pending_path_unchecked(entry.path)`, which **drops `segment_start`/`segment_count`**
(TD-205), then calls `source.delete_entries({entry.path}, …)` and `dest.assign_model(…)` **without
checking either result**.

Migrating a 30-pair file split into 3 segments:
- the `chatbot` pool got the **whole file** (all 30 pairs);
- the other two segments failed ("already pending");
- the source delete matched nothing (segment-aware identity), so **all 3 segments stayed in the
  legacy pool**;
- it printed "Migrated 1/3".

Training both pools trains every pair twice. Any failed source delete (e.g. a remote error) also
leaves an entry in both pools, and a failed `assign_model()` silently loses the model assignment.
`migrate` is TD-202's one-time upgrade path, so operators run it over their whole legacy pool.

Action Items:

- [ ] Carry `segment_start`/`segment_count` into `add_pending_path_unchecked()` and match/delete source entries by `(path, segment_start, segment_count)` (TD-205's `SegmentTarget`).
- [ ] Check `delete_entries()`/`assign_model()` results; on failure, roll back the destination add (or report clearly) and don't count the entry as migrated.
- [ ] Test: migrating a segmented file moves exactly its segments and leaves the source empty.

Location in code: `src/DatasetManagerTool.cpp` (`migrate`); tagged `TODO: See TD-326`.

Files to Modify:

- `src/DatasetManagerTool.cpp`
- `src/DatasetRegistry.{hpp,cpp}`
- tests (new CLI-level test)

---

### TD-327: `dataset_manager remove` Reports Success in Remote Mode Without Removing Anything

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / dataset_manager CLI | October 11, 2026 | 1 hour |

Description:
Found alongside TD-326 and confirmed against a live local `registry_server`.
`DatasetRegistry::remove_pending()` erases the entry from its in-memory list, calls
`save_pending_list()` and **ignores its result**. `RemoteTransport::save_pending()` is a no-op that
returns `false`. `dataset_manager remove <path>` printed "✅ Removed from pending queue… Pending
files: 0" and **exited 0**, yet `list-pending` still showed the file, which will still be trained.
The usage text says `remove` is "local mode only", but nothing enforces it. `clear-pending` does
fail in remote mode, but with the misleading "Failed to save pending list".

Action Items:

- [ ] Refuse `remove`/`clear-pending` in remote mode with a message pointing at `delete` (or implement them via `delete_entries()`).
- [ ] Make `remove_pending()` return the save result.

Location in code: `src/DatasetManagerTool.cpp` (`remove`, `clear-pending`), `src/DatasetRegistry.cpp` (`remove_pending()`); tagged `TODO: See TD-327`.

Files to Modify:

- `src/DatasetManagerTool.cpp`
- `src/DatasetRegistry.cpp`

---

### TD-328: `dataset_manager` Stores Relative Paths in the Local Pool

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / dataset_manager CLI | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-326 and confirmed by running the binary. In local mode, `add data.jsonl` stored
`data.jsonl` verbatim (`DatasetRegistry::add_file()` doesn't canonicalize), and `DataFetcher`'s local
output dirs are relative (`gutenberg_data/`, `huggingface_data/`). A trainer whose working
directory differs (e.g. a systemd unit with its own `WorkingDirectory`) resolves these against the
wrong directory and can't find the files.

Action Items:

- [ ] Store `fs::absolute()`/`weakly_canonical()` paths in local mode (in `add_file()` or the CLI).

Location in code: `src/DatasetManagerTool.cpp` (`add`, local `gutenberg`/`huggingface`); tagged `TODO: See TD-328`.

Files to Modify:

- `src/DatasetManagerTool.cpp`
- `src/DatasetRegistry.cpp`

---

### TD-329: `dataset_manager` Uses the MNS `run_group` Only in `assign`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / dataset_manager CLI | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-326, by inspection. Only `assign` resolves the model in MNS and adopts its
`run_group` (the MNS-authoritative standard in CLAUDE.md). `status`, `list-*`, `unassign`, `delete`,
`migrate`, `segment` and the fetch commands all use the config file's `RUN_GROUP`. When MNS has
given a model a different group, `assign` writes into that group while every other command reads
or changes another, so operators see and edit the wrong pool.

Action Items:

- [ ] Resolve the effective `run_group` once at startup (from MNS when `NAME_SERVICE_URL` + a model are known, else config) and use it for every command; print which group is in use.

Location in code: `src/DatasetManagerTool.cpp` (`assign`'s MNS block); tagged `TODO: See TD-329`.

Files to Modify:

- `src/DatasetManagerTool.cpp`

---

### TD-330: `dataset_manager` Output, Exit Code and Flag Hygiene

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / dataset_manager CLI | October 11, 2026 | 1 hour |

Description:
Found alongside TD-326, by inspection:
- Success lines report the **requested** pair count ("Book added (500 pairs)"), not what was
  written.
- `models` silently falls back to `http://localhost:8083` when `NAME_SERVICE_URL` is unset, while
  `assign` skips verification in that case.
- There's no `--help` (it prints "Unknown command").
- `--config`/`--kind` are stripped from anywhere in argv, and a trailing one with no value becomes
  an ordinary argument.

Action Items:

- [ ] Report actual pair counts; make `models` require `NAME_SERVICE_URL`; add `--help`; only parse global flags before the command and require their values.

Location in code: `src/DatasetManagerTool.cpp` (global flag loop, `models`); tagged `TODO: See TD-330`.

Files to Modify:

- `src/DatasetManagerTool.cpp`

---

### TD-331: `dataset_manager` Has No Tests for Its Commands

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / dataset_manager CLI | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-326. `main()` composes `DatasetRegistry` calls per command, but only the
parsers (`datasetManagerArgsTests`) and `DatasetRegistry` itself are tested. TD-326 and TD-327 live
exactly in that composition and went unnoticed. Both reproduced in seconds by running the binary
against a scratch `SESSION_DIR` and a local `registry_server`.

Action Items:

- [ ] Add a CLI-level test target that runs `dataset_manager` against a scratch `SESSION_DIR` (and a local `registry_server` for remote mode), covering `migrate` with segments, `remove`, `assign --count`, `delete` and `segment`.

Location in code: `src/DatasetManagerTool.cpp` (`main()`); tagged `TODO: See TD-331`.

Files to Modify:

- `tests/CMakeLists.txt`
- new test file

---

### TD-332: Operator Guide Covers Only 8 of 15 `dataset_manager` Commands

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / dataset_manager CLI | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-326, by inspection. `docs/operations/guides/dataset-manager-guide.md` documents
`add`, `gutenberg`, `gutenberg-batch`, `huggingface`, `status`, `list-pending`/`list-trained` and
`clear-pending`. It doesn't cover `remove`, `assign`, `unassign`, `delete`, `migrate`, `segment`,
`models` or the `--kind` flag, which are the commands that change pools and assignments. The usage
text in `main()` is the only complete description.

Action Items:

- [ ] Add sections for the missing commands and `--kind`, including the "omit = all" semantics (TD-321) and remote/local differences.

Location in code: `src/DatasetManagerTool.cpp` (usage text); tagged `TODO: See TD-332`.

Files to Modify:

- `docs/operations/guides/dataset-manager-guide.md`

---

### TD-333: Segment Claims Aren't Released on Failure and Trained Records Ignore Segments

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Data / DatasetRegistry | October 11, 2026 | 4-8 hours |

Description:
Found while writing the code-traced reference [DatasetRegistry.md](../reference/source/DatasetRegistry.md) and confirmed by running the real class
against a scratch `LocalTransport` pool (one 30-pair file split into three TD-205 segments).
`acquire_pending()` hands out segments correctly, but every production caller
(`IncrementalTrainingTool.cpp`'s train/resume/retrain/LeJEPA paths and
`IncrementalTrainer::resume_last_session()`) then calls `mark_trained()`/`release_pending()` with
**bare paths** (`resp.registry_paths()`):

- **Failure doesn't release segments.** Since TD-205, a bare path releases only whole-file entries.
  `release_pending("run-A", {path})` left the segment **claimed by `run-A`**, and a pass with a
  different `run_id` couldn't acquire it. The same `run_id` can reclaim its own claims, so an MNS
  `resume` recovers. But after a `retrain` (new MNS run), or in standalone mode (run ID from
  hostname + PID, new every pass), the segment stays stuck until released by hand.
- **Success is recorded per path.** Training two segments of one file produced **one** trained
  file with 10 samples (`build_new_versions()` skips a path already in `trained_set_`), and the whole
  path now counts as trained, so re-queuing any segment of it is **rejected**. Segments trained in
  separate passes can't all be recorded.

`registry_server`'s release/trained handlers follow the same path-keyed rules.

Action Items:

- [ ] Pass `FileToken` segment ranges through as `SegmentTarget`s to `release_pending()`/`mark_trained()` from every caller.
- [ ] Key trained records by `(path, segment_start, segment_count)` (local registry format, `DataVersion`, `registry_server`), and make `is_trained()` segment-aware.
- [ ] Tests: acquire → fail → release frees the segment for another run; training two segments records both, with their sample counts.

Location in code: `src/DatasetRegistry.cpp` (`mark_trained()`, `release_pending()`, `build_new_versions()`, `is_trained()`), callers in `src/IncrementalTrainingTool.cpp` and `src/IncrementalTrainer.cpp`; tagged `TODO: See TD-333`.

Files to Modify:

- `src/DatasetRegistry.{hpp,cpp}`
- `src/RegistryTransport.{hpp,cpp}`
- `src/RegistryServer.cpp`
- `src/IncrementalTrainingTool.cpp`
- `src/IncrementalTrainer.cpp`
- `tests/DatasetRegistryTests.cpp`

---

### TD-334: Trained Sample Counts Are Always Zero

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / DatasetRegistry | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-333, by inspection of every production call site. Each `mark_trained()` call
passes `std::vector<int> counts(n, 0)` (trainer, retrain, LeJEPA, `resume_last_session()`), so every
`DataVersion::num_samples` is 0 and `total_samples_trained()` (`dataset_manager status`'s
"Samples: N total trained") always reports 0. The trainer has the real per-file pair counts (it loads
them in `train_on_files()`); they're just not passed back.

Action Items:

- [ ] Return per-file (per-segment) pair counts from `train_on_files()`/`retrain_on_files()` and pass them to `mark_trained()`.

Location in code: `src/IncrementalTrainingTool.cpp` / `src/IncrementalTrainer.cpp` (`mark_trained()` calls); tagged `TODO: See TD-334`.

Files to Modify:

- `src/IncrementalTrainingTool.cpp`
- `src/IncrementalTrainer.{hpp,cpp}`

---

### TD-335: `DatasetRegistry` Dead Code and Unused Integrity Data

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / DatasetRegistry | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-333, by inspection:
- `compute_checksum()` (`size_mtime`) is stored in every `DataVersion` but never compared anywhere.
- `count_pairs()` materializes every pair (all text) just to count them.
- The private `registry_file_path()`/`pending_file_path()` are unused and ignore the TD-202 kind
  subdirectory; `add_files()` and `print_run_assignments()` have no callers.
- `make_config()` never copies `max_files_per_run` (always 0 = claim all) or `data_registry_file`.
- `acquire_pending()` doesn't update the in-memory `run_id` of entries already in `pending_`.

Action Items:

- [ ] Use the checksum (warn when a queued/trained file changed) or drop it; count pairs without storing them; delete the unused helpers; map the missing config fields; refresh in-memory claims after acquire.

Location in code: `src/DatasetRegistry.cpp` (`compute_checksum()`, `count_pairs()`, `make_config()`, `acquire_pending()`), `src/DatasetRegistry.hpp` (private helpers); tagged `TODO: See TD-335`.

Files to Modify:

- `src/DatasetRegistry.{hpp,cpp}`
- `src/Config.{hpp,cpp}`

---

### TD-336: Legacy Training-File Parser Truncates Responses; BOM Breaks JSONL Detection

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / DatasetRegistry | October 11, 2026 | 1 hour |

Description:
Found alongside TD-333, by inspection. `load_conversation_pairs()`:
- In the legacy `INPUT:`/`RESPONSE:` format, keeps only the **first line** of a response; continuation
  lines are dropped.
- Detects JSONL by the first non-empty line starting with `{`. A UTF-8 byte-order mark before it
  makes a JSONL file parse as legacy and load 0 pairs.

Action Items:

- [ ] Append continuation lines to the current response; strip a leading BOM before detection; test both.

Location in code: `src/DatasetRegistry.cpp` (`load_conversation_pairs()`); tagged `TODO: See TD-336`.

Files to Modify:

- `src/DatasetRegistry.cpp`
- `tests/DatasetRegistryTests.cpp`

---

### TD-337: `DataTransport` Names Downloads by Basename, So Segments and Same-Named Files Collide

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / DataTransport | October 11, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [DataTransport.md](../reference/source/DataTransport.md) and confirmed end-to-end on loopback
(local `registry_server` with FTP enabled, a 15.8 MB file split into three TD-205 segments).
`fetch()` saves to `download_dir / filename(token.ftp_path)`:

- **Segments download the whole file N times, concurrently, to one path.** All three segment tokens
  had the same `ftp_path` and full-file `size_bytes`; `fetch_all()` sent all three to `dl/big.jsonl`
  in parallel, transferring the full 15.8 MB three times. The file was intact in 20 runs (identical
  bytes), but each thread opens with `trunc` and **deletes the file on a size mismatch**, so one
  thread's final check can delete another's in-progress download and fail the pass.
- **Different files with the same basename collide**, e.g. `<group>/uploads/x.jsonl` vs
  `<group>/datasets/x.jsonl`, or two kinds' sub-pools sharing one `download_dir`.

The `UsesBasenameOfFtpPath` test pins the basename behaviour.

Action Items:

- [ ] Name local files from the full `ftp_path` (or a hash of it).
- [ ] Deduplicate tokens by `ftp_path` in `fetch_all()` (download once, return the path for each segment token).
- [ ] Tests: three segment tokens of one file trigger one download; same-basename files in different directories don't collide.

Location in code: `src/DataTransport.hpp` (`fetch()` local path, `fetch_all()`); tagged `TODO: See TD-337`.

Files to Modify:

- `src/DataTransport.hpp`
- `tests/DataTransportFtpTests.cpp`

---

### TD-338: `DataTransport` Reuses Same-Size Stale Files Without Checking the Checksum

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / DataTransport | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-337 and confirmed end-to-end. "Condition B" returns an existing local file
whenever its size equals `token.size_bytes`, without downloading. A planted same-name, same-size file
of different content (all `Z` bytes) was returned and would have been trained, although the token
carried `checksum = "15769547_<mtime>"`, which would have caught it. The trainer's startup sweep
keeps non-empty files, so a stale copy survives across passes. Triggers include a dataset re-uploaded
under the same name with a size-preserving edit, and TD-337's basename collisions.

Action Items:

- [ ] Store the token checksum alongside each download (e.g. a sidecar file) and skip only when it matches; otherwise re-download.
- [ ] Test: same-size different-checksum file is re-downloaded.

Location in code: `src/DataTransport.hpp` (`fetch()` Condition B); tagged `TODO: See TD-338`.

Files to Modify:

- `src/DataTransport.hpp`
- `tests/DataTransportFtpTests.cpp`

---

### TD-339: FTPS Downloads Don't Verify the Registry's Certificate

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / DataTransport | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-337, by inspection. With FTPS enabled, `fetch()` sets `CURLOPT_SSL_VERIFYPEER`
and `CURLOPT_SSL_VERIFYHOST` to 0 to accept the registry's self-signed certificate. The control
channel is encrypted, but anyone able to intercept the connection can impersonate `registry_server`,
serve arbitrary training data, and collect the one-time credentials. There's no option to pin the
server certificate or trust a private CA. This pairs with TD-284 (`registry_server` ignores the
config file's FTPS keys).

Action Items:

- [ ] Add a trainer config option for a CA bundle or pinned certificate (`CURLOPT_CAINFO` / `CURLOPT_PINNEDPUBLICKEY`), enable verification when set, and warn loudly when FTPS runs unverified.

Location in code: `src/DataTransport.hpp` (`fetch()` FTPS options); tagged `TODO: See TD-339`.

Files to Modify:

- `src/DataTransport.hpp`
- `src/Config.{hpp,cpp}`
- `src/IncrementalTrainingTool.cpp`

---

### TD-340: `DataTransport` Robustness and Header Hygiene

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Data / DataTransport | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-337, by inspection:
- `curl_global_init()` is never called; implicit init from several `fetch_all()` threads at once is
  safe only on a thread-safe-init libcurl (8.5.0 on the dev machine is; the deployed host wasn't
  checked).
- A final failure deletes the partial file, so resume can't help across failed passes.
- `CURLE_COULDNT_CONNECT` isn't retried, so a registry restart mid-pass fails the pass immediately.
- `using adai::Logger;` and `namespace fs = std::filesystem;` at global scope in a header leak into
  every includer.

Action Items:

- [ ] Call `curl_global_init()` once at trainer startup; keep partial files on failure; treat `COULDNT_CONNECT` as retryable with backoff; move the `using`/alias into a namespace or the .cpp.

Location in code: `src/DataTransport.hpp` (top-level `using`, retry loop, failure cleanup); tagged `TODO: See TD-340`.

Files to Modify:

- `src/DataTransport.hpp`
- `src/IncrementalTrainingTool.cpp`

---

### TD-341: World-Model and Hippocampal Gates Never Train in Production

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Transformer / DecoderBlock | October 11, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [DecoderBlock.md](../reference/source/DecoderBlock.md) and confirmed by running the real class.
`DecoderBlock`'s TD-180 gates (`gate`, `gate_h`) are plain floats updated only by plain SGD inside
`DecoderBlock::update_weights()`; `register_parameters_with_optimizer()` doesn't register them
(`Optimizer` takes only `Matrix` groups). `ChatbotTrainer` never calls `update_weights()`: it
registers the model with one optimizer and calls `optimizer->step()` directly (which is also how it
avoids TD-297). So **nothing ever updates the gates**.

A block with the world-model path enabled, trained 50 steps the way `ChatbotTrainer` does, ended at
`gate = 0.000000` (gradients up to 34 were computed, then zeroed). With `tanh(gate) = 0` the gated
cross-attention's own gradient is also zero, so its weights didn't change at all. The same 50 steps
via `update_weights()` moved the gate to −1.04. World-model injection (TD-180/181/186) and
hippocampal memory (TD-185/194) therefore remain exact no-ops in every trained model, whatever MNS
links. Nothing reads `get_gate()` outside tests, so no metric reveals it.

Action Items:

- [ ] Make each gate a 1×1 `Matrix` (value + gradient) and register it in `register_parameters_with_optimizer()`; drop the plain-SGD gate step from `update_weights()` (or keep it only for the no-optimizer case).
- [ ] Push per-layer `tanh(gate)`/`tanh(gate_h)` as a training metric so a stuck gate is visible.
- [ ] Test: training via the trainer's optimizer-only path moves a gate off zero.

Location in code: `src/DecoderBlock.cpp` (`update_weights()`, `register_parameters_with_optimizer()`), `src/DecoderBlock.hpp` (gate members); tagged `TODO: See TD-341`.

Files to Modify:

- `src/DecoderBlock.{hpp,cpp}`
- `src/ChatbotTrainer.cpp` (metrics)
- `tests/decoderblock_test.cpp`

---

### TD-342: KV-Cached Generation and the GPU Path Skip the Gated Paths

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Transformer / DecoderBlock | October 11, 2026 | 4-8 hours |

Description:
Found alongside TD-341, by inspection (documented in the class header). `forward_with_cache()` has no
world-model or memory parameters, and its no-cache fallback calls `forward()` without them, so every
KV-cached generation strategy (greedy, sampling, …) ignores an attached world model and hippocampal
memory; only the non-cached beam branch passes them (TD-186). `gpu_forward()`,
`gpu_forward_with_cache()` and `gpu_backward()` implement only the core sub-layers, and the GPU
upload/download/zero helpers skip the gated sub-layers, so GPU training never trains them either.
Once TD-341 is fixed, inference behaviour will differ by generation strategy and device.

Action Items:

- [ ] Thread `world_model_output`/`memory` (and the hippocampal knobs) through `forward_with_cache()` and `LLMDecoder::forward_with_cache()`, caching the world-model K/V like the encoder's.
- [ ] Add the gated paths to the GPU forward/backward and the upload/download/zero helpers, or refuse to enable them on GPU with a clear error.

Location in code: `src/DecoderBlock.cpp` (`forward_with_cache()`, GPU functions); tagged `TODO: See TD-342`.

Files to Modify:

- `src/DecoderBlock.{hpp,cpp}`
- `src/Decoder.cpp`
- `src/EncoderDecoderModel.cpp`

---

### TD-343: Hippocampal Memory Is Updated on Every Forward Pass

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Transformer / DecoderBlock | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-341, by inspection. When `memory` is passed, `forward()` decays and accumulates
coverage and Hebbian associations on **every** call: training forwards, validation forwards, and one
call per hypothesis in beam search (the only generation branch that passes memory). The design
("called once per decode step") assumes one update per generated token, so coverage and associations
build up several times faster than intended, strengthening the repetition penalty and cross-reference
pull, and training passes pollute the persistent memory state that's saved on shutdown.

**Update (October 11, 2026, from [Decoder.md](../reference/source/Decoder.md)):** `LLMDecoder` passes the
same `memory` to every block with the gated path, and each block updates it, so one forward applies
one decay-and-accumulate round **per injected layer** (6 with `WORLD_MODEL_INJECT_EVERY_N_LAYERS = 1`
and 6 layers). The fix should update memory once per forward, at the decoder or model level.

**Update (October 11, 2026, from [EncoderDecoderModel.md](../reference/source/EncoderDecoderModel.md)):**
`generate_response()` writes the response into hippocampal memory after every CPU generation, including
`ChatbotTrainer`'s label-backfill and generation-quality calls on the live model, so training-time
evaluation also writes episodic memories. Only serving generations should write.

Action Items:

- [ ] Separate scoring from state update: let the caller commit coverage/association updates once per decode step (and never during training/validation unless intended).
- [ ] Test: beam search with width k updates coverage once per step.

Location in code: `src/DecoderBlock.cpp` (`forward()` hippocampal coverage/association update); tagged `TODO: See TD-343`.

Files to Modify:

- `src/DecoderBlock.{hpp,cpp}`
- `src/EncoderDecoderModel.cpp`
- `src/HippocampalMemory.{hpp,cpp}`

---

### TD-344: `DecoderBlock` Dropout Is Stored but Never Applied

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Transformer / DecoderBlock | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-341, by inspection. The constructor takes `dropout` (default 0.1, which
`LLMDecoder` passes), stores it in `dropout_rate`, and `save()`/`load()` persist it, but no forward
path applies dropout. Training gets no dropout regularization despite the parameter.

**Update (October 11, 2026, from [EncoderBlock.md](../reference/source/EncoderBlock.md)):** `EncoderBlock`
has the same gap: `dropout` (default 0.1) is stored, saved, loaded and printed but never applied, so
none of the encoders (chatbot, RAG, LeJEPA world model) get dropout either. Fix both blocks together.

Action Items:

- [ ] Either implement dropout (training-mode only, on sub-layer outputs) or remove the parameter and its persistence (keeping load() compatible).

Location in code: `src/DecoderBlock.cpp` (constructor); tagged `TODO: See TD-344`.

Files to Modify:

- `src/DecoderBlock.{hpp,cpp}`
- `src/Decoder.cpp`

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

Location in code: `src/DecoderBlock.cpp` (`load()`); tagged `TODO: See TD-345`.

Files to Modify:

- `src/DecoderBlock.{hpp,cpp}`
- `src/ModelSerializer.cpp`

---

### TD-346: `LLMDecoder::update_weights()` Never Updates the Final Norm

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Transformer / Decoder | October 11, 2026 | 1 hour |

Description:
Found while writing the code-traced reference [Decoder.md](../reference/source/Decoder.md) and confirmed by running the real class.
`update_weights(float lr)` calls the token embedding's and every block's `update_weights()`, but the
final norm's update is commented out ("For now, we'll skip if the method doesn't exist"), although
`LayerNorm::update_weights()` exists; the `lr` argument is ignored. Twenty steps through this path
moved the token embeddings (L1 0.22) while the final norm's gamma and beta stayed exactly unchanged.
`EncoderDecoderModel::update_weights()`/`train_step()` use this path; the shipped `ChatbotTrainer`
calls `optimizer->step()` and is unaffected.

Action Items:

- [ ] Call `final_norm->update_weights()` (in the no-optimizer case; with an optimizer, fold into TD-297's single top-level `step()`); remove or honour the unused `lr` parameter.
- [ ] Test: `update_weights()` changes the final norm's parameters.

Location in code: `src/Decoder.cpp` (`update_weights()`); tagged `TODO: See TD-346`.

Files to Modify:

- `src/Decoder.{hpp,cpp}`
- `tests/decoder_test.cpp`

---

### TD-347: Cached and Full Decoder Forwards Diverge Past `max_seq_length`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Transformer / Decoder | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-346 and confirmed by running the real class. Within `max_seq_length` the KV-cached
incremental forward matches the full forward (|Δ| = 0.000002). Past it they split:
`PositionalEncoding::forward()` leaves rows beyond `max_len` without positional encoding (stderr
warning), while `forward_with_cache()` and `gpu_decode_step()` compute positional encoding inline for
any position. With `max_seq_length = 4` the last token of a length-6 sequence differed by 13.6 (L1 over
`d_model = 16`). Over-length generation therefore produces different hidden states depending on the
strategy (KV-cached greedy/sampling vs full-recompute beam), and neither has positions the model was
trained on.

Action Items:

- [ ] Cap decoder input + generated length at `max_seq_length` in generation (truncate the oldest context or stop), and make the cached paths refuse or clamp positions ≥ `max_seq_length` consistently with `PositionalEncoding`.
- [ ] Test: generation never exceeds `max_seq_length`; cached and full forwards agree at the boundary.

Location in code: `src/Decoder.cpp` (`forward_with_cache()`, `gpu_decode_step()` positional encoding); tagged `TODO: See TD-347`.

Files to Modify:

- `src/Decoder.cpp`
- `src/EncoderDecoderModel.cpp`
- `src/PositionalEncoding.cpp`

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

Location in code: `src/Decoder.cpp` (`set_training()`, `forward_with_encoder()`, `load_weights()`, `gpu_backward()`); tagged `TODO: See TD-348`.

Files to Modify:

- `src/Decoder.{hpp,cpp}`

---

### TD-349: RAG Indexes Whole Files but Embeds Only Their First `max_seq_length` Tokens

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | RAG / DocumentStore | October 11, 2026 | 2-4 hours |

Description:
Found while writing the code-traced reference [DocumentStore.md](../reference/source/DocumentStore.md) and confirmed by running the real class.
`chatbot_api_server` adds each top-level `.txt` file under `RAG_DOCS_PATH` as **one** `DocumentStore`
document (no chunking), and `LLMEncoder::encode()` truncates input to `max_seq_length` tokens before
embedding. A long document is therefore represented by its opening only. With `max_seq_length = 16`,
two documents sharing their first 16 tokens got **bit-identical embeddings** and tied on every query,
including one matching the tail of one of them exactly. At the default `MAX_SEQ_LENGTH = 512`,
content past roughly the first page of each file is unretrievable, yet the whole file text is what
`RAGInference` puts in the prompt.

Action Items:

- [ ] Chunk documents at index time (e.g. overlapping windows within `max_seq_length` tokens), each chunk its own entry with a parent-document reference, and return chunks rather than whole files.
- [ ] Test: content past the first window is retrievable.

Location in code: `src/DocumentStore.cpp` (`addDocument()`), `src/ChatbotAPIServer.cpp` (RAG indexing loop); tagged `TODO: See TD-349`.

Files to Modify:

- `src/DocumentStore.{hpp,cpp}`
- `src/ChatbotAPIServer.cpp`
- `src/RAGInference.cpp`
- `tests/documentstore_test.cpp`

---

### TD-350: `DocumentStore::retrieve()` Isn't Safe to Call Concurrently

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | RAG / DocumentStore | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-349, by inspection. `retrieve()` is `const` but calls `encoder->encode()`, which
mutates the encoder's caches (`requires_grad` defaults to true) and prints `Encoding N tokens...` to
stdout on every query; `DocumentStore` has no lock. Today RAG generation runs under
`ChatbotAPI::config_mutex_` (TD-230), so requests are serialized and there's no race. Fixing TD-230
by releasing that lock, as its own action item suggests, would expose concurrent `retrieve()` calls
to data races inside the encoder.

Action Items:

- [ ] Before (or with) TD-230: give `DocumentStore` a mutex around encode+score, or put the RAG encoder in inference mode (`set_requires_grad(false)`) and make `encode()` side-effect free; drop the stdout print.
- [ ] Test: concurrent `retrieve()` under TSan.

Location in code: `src/DocumentStore.cpp` (`retrieve()`); tagged `TODO: See TD-350`.

Files to Modify:

- `src/DocumentStore.{hpp,cpp}`
- `src/LLMEncoder.cpp`
- `src/ChatbotAPIServer.cpp`

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

Location in code: `src/EncoderBlock.cpp` (`update_weights()`, `load_weights()`, `save_weights()`, `print_config()`); tagged `TODO: See TD-357`.

Files to Modify:

- `src/EncoderBlock.{hpp,cpp}`
- `src/LLMEncoder.cpp`

---

### TD-358: GPU Training Feeds Unshifted Targets, Teaching the Model to Copy Instead of Predict

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Models / EncoderDecoderModel | October 11, 2026 | 1-2 hours (plus retraining every GPU-trained checkpoint) |

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

Action Items:

- [ ] In `gpu_forward()`, build the decoder input exactly as `forward()` does (`[BOS] + target[:-1]`, via one shared helper) while keeping `targets` as the labels.
- [ ] Add a CPU-vs-GPU equivalence test (same weights → same logits and loss for `forward()` and `gpu_forward()`), run on ai-machine; until GPU CI exists, ship a run package for the remote host.
- [ ] Retrain affected models; consider bumping the checkpoint format version so GPU-trained pre-fix checkpoints are flagged.

Location in code: `src/EncoderDecoderModel.cpp` (`gpu_forward()`); tagged `TODO: See TD-358`.

Files to Modify:

- `src/EncoderDecoderModel.{hpp,cpp}`
- `tests/encoderdecoder_test.cpp`

---

### TD-359: `generate_response()` and `gpu_generate_response()` Ignore `max_length`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Models / EncoderDecoderModel | October 11, 2026 | 1 hour |

Description:
Found alongside TD-358 and confirmed. Neither method forwards its `max_length` argument to
`TextGenerator`, which uses its stored `max_length` (initially `max_seq_length`):
`generate_response("hello world", 5)` produced 9 tokens with the generator at 64. The GPU version
documents it ("kept for interface parity"). `ChatbotTrainer`'s generation-quality metrics
(`generation_quality_max_tokens = 50`) and label backfill call `generate_response()`, so their
generations can run to `max_seq_length`, slowing metric computation. The `_with_strategy` variants do
honour `max_length`.

Action Items:

- [ ] Apply `max_length` for the call (temporarily set it on the generator config and restore it, or pass it through), on both CPU and GPU.
- [ ] Test: `generate_response(…, 5)` returns at most 5 new tokens.

Location in code: `src/EncoderDecoderModel.cpp` (`generate_response()`, `gpu_generate_response()`); tagged `TODO: See TD-359`.

Files to Modify:

- `src/EncoderDecoderModel.cpp`
- `tests/encoderdecoder_test.cpp`

---

### TD-360: `EncoderDecoderModel` No-Optimizer Path Never Updates the Encoder

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Models / EncoderDecoderModel | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-358 and confirmed. `update_weights()` and `set_learning_rate()` skip the encoder
(the calls are commented out: "LLMEncoder doesn't have this method"), and the LM head's LR isn't set
either. **Corrected October 11, 2026** ([LLMEncoder.md](../reference/source/LLMEncoder.md)): that
comment is stale for `set_learning_rate()`, which `LLMEncoder` has had since the TD-093 fix; only
`LLMEncoder::update_weights()` is missing. Ten `train_step()` calls with no optimizer left the encoder's weights exactly unchanged while
the decoder's moved. Combined with TD-346 (final decoder norm), `train_step()` trains only the decoder
blocks, embeddings and head. `ChatbotTrainer` registers an optimizer and is unaffected.

Action Items:

- [ ] Add `LLMEncoder::update_weights()`, call it and the existing `LLMEncoder::set_learning_rate()`; set the head's LR; fold into TD-297's single-step design for the optimizer case.

Location in code: `src/EncoderDecoderModel.cpp` (`update_weights()`, `set_learning_rate()`); tagged `TODO: See TD-360`.

Files to Modify:

- `src/EncoderDecoderModel.cpp`
- `src/LLMEncoder.cpp`
- `src/encoder.hpp`

---

### TD-361: Training Targets Get a Double `BOS`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Models / EncoderDecoderModel | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-358 and confirmed. Targets are tokenized with `encode(response, true)` =
`[BOS, w1…wn, EOS]` (in `train_step()` and `ChatbotTrainer`'s tokenization), and `forward()` prepends
another `BOS`, so the decoder input is `[BOS, BOS, w1…wn]` and position 0 is trained to predict `BOS`
from `BOS`. Generation follows suit: from `[BOS]`, a trained model first emits a second `BOS`
(`2 2 12 23 …` in the TD-358 experiment), wasting a step and a `max_length` slot.

Action Items:

- [ ] Either tokenize targets without the leading `BOS` (keeping `EOS`) or stop prepending in `forward()`/the GPU fix, consistently for CPU and GPU; this changes training alignment, so do it together with TD-358's retraining.

Location in code: `src/EncoderDecoderModel.cpp` (`forward()` decoder input, `train_step()`); tagged `TODO: See TD-361`.

Files to Modify:

- `src/EncoderDecoderModel.cpp`
- `src/ChatbotTrainer.cpp`

---

### TD-362: CPU and GPU Generation Paths Behave Differently

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Models / EncoderDecoderModel | October 11, 2026 | 2-4 hours |

Description:
Found alongside TD-358, by inspection. The four generation entry points diverge:
- Invalid-vocab masking (IDs ≥ the tokenizer's vocab) happens only in the CPU strategy path and the
  CPU beam paths; the CPU plain cached path and every GPU path don't mask.
- Only CPU paths write hippocampal memory or attach the world model (beam only).
- An unknown strategy goes to `generate()` on CPU but to nucleus sampling on GPU.
- `num_beams > 1` forces beam on CPU (TD-261) but is ignored by the GPU non-beam strategies.

The same request can therefore produce different output depending on whether the server runs on a
GPU build.

Action Items:

- [ ] Factor one shared model_fn/strategy-dispatch for CPU and GPU (masking, defaults, beam handling, memory write), differing only in the forward call.

Location in code: `src/EncoderDecoderModel.cpp` (`generate_response()`, `gpu_generate_response()`, `gpu_generate_response_with_strategy()`); tagged `TODO: See TD-362`.

Files to Modify:

- `src/EncoderDecoderModel.cpp`

---

### TD-363: `EncoderDecoderModel` Clone, Load and World-Model Input Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Models / EncoderDecoderModel | October 11, 2026 | 1-2 hours |

Description:
Found alongside TD-358, by inspection:
- `clone()` round-trips `save_model()`/`load_model()`, so the copy lacks the world model, hippocampal
  memory, generation config, repetition params and LoRA adapters. `ChatbotTrainer`'s async BLEU/ROUGE
  snapshots therefore generate differently from the live model.
- `load_model()` doesn't check its config reads, and restores special-token IDs onto the model but not
  into the generator's config until `sync_special_tokens()` runs.
- `forward()` re-derives the world model's input by decoding token ids back to text, which is lossy
  under the tokenizer's normalization (TD-224).

Action Items:

- [ ] Copy the extra state in `clone()` (or document the snapshot's scope); check reads and sync the generator in `load_model()`; pass the original text to the world model where available.

Location in code: `src/EncoderDecoderModel.cpp` (`clone()`, `load_model()`, `forward()` world-model block); tagged `TODO: See TD-363`.

Files to Modify:

- `src/EncoderDecoderModel.{hpp,cpp}`

---

### TD-364: `LLMEncoder::encode()` Input Doesn't Match the Model's Encoder Input

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Transformer / LLMEncoder | October 11, 2026 | 2-4 hours |

Description:
Found while tracing `LLMEncoder` and verified. `encode(text)` tokenizes with the encoder's own
tokenizer and `add_special_tokens = true`, so `"hello world"` becomes `[BOS, 80, 77, EOS]`.
`EncoderDecoderModel` always tokenizes encoder input with `encode(input_text, false)` (`[80, 77]`), in
training and in generation, so the trained encoder has never seen `BOS`/`EOS` at its input and every
real token sits one position later under `encode()`. The first output row differed (1.398 vs 0.980).
`PipelineInferenceEngine`'s encoder stage calls `encode()` on the trained chat encoder and hands the
result to the decoder, so pipeline-mode answers come from out-of-distribution encoder input. For RAG
(`DocumentStore`) it only needs to be consistent, which it is.

Also verified: when `encode()` truncates to `max_seq_length` it uses `resize()`, dropping the `EOS` it
just added (32 ids cut to 8, last id no longer `EOS`).

Action Items:

- [ ] Give `encode()` an `add_special_tokens` parameter (default matching the model: `false`), or have `PipelineInferenceEngine` tokenize with the model's tokenizer and call `encode_with_mask()`.
- [ ] When truncating with special tokens, keep the final `EOS`.
- [ ] Add a test that pipeline and plain generation produce the same encoder output for the same text.

Location in code: `src/LLMEncoder.cpp` (`encode()`), `src/PipelineInferenceEngine.hpp` (encoder stage); tagged `TODO: See TD-364`.

Files to Modify:

- `src/LLMEncoder.cpp`
- `src/encoder.hpp`
- `src/PipelineInferenceEngine.hpp`
- `tests/llmencoder_test.cpp`

---

### TD-365: Encoder Input Is Never Truncated to `max_seq_length` When Serving

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Transformer / LLMEncoder | October 11, 2026 | 2-4 hours |

Description:
Found while tracing `LLMEncoder` and verified. Only `encode(text)` truncates. `encode_with_mask()` and
`gpu_encode()` take any length, and neither `EncoderDecoderModel`'s generation entry points nor
`ChatbotAPI` truncate the input. A 60-token input on a `max_seq_length = 16` model completed generation
after `PositionalEncoding` warned *"Positions beyond max_len will not receive positional encodings"*, so
44 positions had no order information. Attention cost also grows as O(n²) with no bound, so a long chat
message costs more CPU or GPU time than the configured model size suggests. Related: TD-347 (decoder
side past `max_seq_length`) and TD-294 (the conversation token budget isn't tied to `max_seq_length`).

Action Items:

- [ ] Decide on keep-the-tail truncation or rejection, and apply it once where `EncoderDecoderModel` tokenizes encoder input (CPU and GPU, training and generation).
- [ ] Add a `max_seq_length` check (assert or throw) in `encode_with_mask()`/`gpu_encode()`.
- [ ] Coordinate with TD-294 so the conversation context fits the encoder.

Location in code: `src/EncoderDecoderModel.cpp` (input tokenization in `generate_response()`, `generate_response_with_strategy()`, both GPU generation paths), `src/LLMEncoder.cpp` (`encode_with_mask()`, `gpu_encode()`); tagged `TODO: See TD-365`.

Files to Modify:

- `src/EncoderDecoderModel.cpp`
- `src/LLMEncoder.cpp`
- `tests/llmencoder_test.cpp`

---

### TD-366: Serving Encoders Cache Activations for Backward; Dead State and Duplicate API

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Transformer / LLMEncoder | October 11, 2026 | 1-2 hours |

Description:
Found while tracing `LLMEncoder`, by inspection. `requires_grad` defaults to `true` and nothing clears it
for the serving encoders (the chat encoder in `chatbot_api_server` and the RAG encoder). So every
inference `encode*()` call copies `(num_layers + 1) × seq × d_model` floats into
`cached_encoder_outputs`, which `backward()` never reads (it relies on each block's own caches).
`cached_embeddings` and `cached_pos_encoded` are declared and never used.
`get_sentence_embedding()` and `get_sentence_embedding_trainable()` are identical, and neither has a
production caller (`DocumentStore` pools for itself).

Action Items:

- [ ] Remove `cached_encoder_outputs`, `cached_embeddings` and `cached_pos_encoded`, or at least stop filling the first.
- [ ] Clear `requires_grad` for serving instances (`EncoderDecoderModel::set_training(false)` already forwards it; make the server and RAG setup call it).
- [ ] Merge the two sentence-embedding functions.

Location in code: `src/encoder.hpp` (members), `src/LLMEncoder.cpp` (`get_sentence_embedding_trainable()`); tagged `TODO: See TD-366`.

Files to Modify:

- `src/encoder.hpp`
- `src/LLMEncoder.cpp`
- `src/ChatbotAPIServer.cpp`

---

### TD-367: `LLMEncoder` Console Output and Unchecked Weight I/O

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Transformer / LLMEncoder | October 11, 2026 | 1-2 hours |

Description:
Found while tracing `LLMEncoder`, by inspection. Library code writes to `std::cout`: a 7-line banner in
the constructor, `"Encoding N tokens..."` on **every** `encode()` call (every RAG add or search and
every pipeline request), save/load, the tokenizer methods and `print_config()`. `load_weights()`
doesn't check its header reads, so a short file produces a misleading "architecture mismatch"
message, and it reads `max_seq_length` but doesn't compare it. `save_weights()` doesn't check its
writes. Same class as TD-357 (`EncoderBlock`).

Action Items:

- [ ] Switch to `adai::Logger` (debug level for the per-call message).
- [ ] Check header reads and writes, and decide whether `max_seq_length` must match (harmless for sinusoidal encoding, so a warning is enough).

Location in code: `src/LLMEncoder.cpp` (constructor, `encode()`, `save_weights()`, `load_weights()`); tagged `TODO: See TD-367`.

Files to Modify:

- `src/LLMEncoder.cpp`

---

### TD-368: Weight Initialization Can't Be Seeded or Reproduced

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Transformer / FeedForward | October 12, 2026 | 2-4 hours |

Description:
Found while tracing `FeedForward` and verified. Every `FeedForward` constructor seeds a fresh
`std::mt19937` from `std::random_device`, and `Matrix::randomize()` (used by the other layers'
initialization) does the same with a function-static generator. Two `FeedForward(4, 8)` instances got
different weights, and there's no seed setting anywhere in `src/` (no config key, no global seed).
So a training run can't be replayed from initialization, a reported divergence can't be reproduced
bit-for-bit, and a CPU-vs-GPU equivalence check (as TD-358's fix needs) must copy weights explicitly
rather than construct two identical models.

Action Items:

- [ ] Add one process-wide seedable initialization RNG (for example `adai::init_rng()` with a `SEED` config key / `--seed` flag; `random_device` when unset, and log the seed actually used).
- [ ] Route `FeedForward`, `Matrix::randomize()` and the other layer initializers through it.
- [ ] Test: two models built with the same seed have identical weights.

Location in code: `src/FeedForward.cpp` (constructor), `src/Matrix.cpp` (`randomize()`); tagged `TODO: See TD-368`.

Files to Modify:

- `src/FeedForward.cpp`
- `src/Matrix.cpp`
- `src/Config.{hpp,cpp}`
- layer constructors that initialize weights

---

### TD-369: `FeedForward::load_weights()` Loads Truncated Files as a Silent Mix of Old and New Weights

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Transformer / FeedForward | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `FeedForward` and verified. `load_weights()` checks the two dimension ints but
never checks the stream afterwards. A file cut off after 10 floats loaded **without an exception**:
the start of `W1` came from the file, and the rest of `W1`, all of `W2` and both biases silently kept
their previous values. This is the per-block `<block>_feedforward.bin` file that
`EncoderDecoderModel::save_model()`/`load_model()` use through the block and encoder save paths, so a
damaged checkpoint becomes a silent mix of old (often random) and new weights. `save_weights()`
doesn't check its writes. Same class as TD-298, TD-345, TD-357 and TD-367.

Action Items:

- [ ] Check the file size up front (`8 + 4·(2·d_model·d_ff + d_ff + d_model)` bytes), read into temporaries, check the stream, and assign only on success.
- [ ] Check `save_weights()`'s stream before returning; write each matrix in one call rather than one float at a time.
- [ ] Test: truncated and empty files throw and leave the layer unchanged.

Location in code: `src/FeedForward.cpp` (`load_weights()`, `save_weights()`); tagged `TODO: See TD-369`.

Files to Modify:

- `src/FeedForward.cpp`
- `tests/feedforward_test.cpp`

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

### TD-371: Residual-Branch Output Isn't Scaled for Depth

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open — owner decision | Transformer / FeedForward | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `FeedForward` and measured. With unit-variance input (what `LayerNorm` produces),
a freshly initialized FFN's output has a standard deviation of about 1.35 (1.38 at d=64, 1.32 at
d=256, 1.37 at d=512, `d_ff = 4·d_model`). Every Pre-LN block adds this, plus its attention output,
to the residual stream unscaled, so the stream's variance grows with depth from the first step.
GPT-2-style models initialize the residual branch's output projections (`W2` here, the attention
`W_o`) with an extra `1/√(2·num_layers)` factor to keep it bounded. This isn't a bug; it's an
initialization choice that bears on the open hypothesis that Pre-LN residual growth caused the
epoch-5 divergence on ai-machine.

Action Items:

- [ ] Decide whether to scale `W2` (and the attention output projections) by `1/√(2·num_layers)` at initialization, which needs the layer count passed down.
- [ ] If adopted, compare a short training run against the current initialization before switching, since it changes training dynamics (existing checkpoints are unaffected).

Location in code: `src/FeedForward.cpp` (constructor `W2` init); tagged `TODO: See TD-371`.

Files to Modify:

- `src/FeedForward.{hpp,cpp}`
- `src/EncoderBlock.cpp`
- `src/DecoderBlock.cpp`
- `src/MultiHeadAttention.cpp`

---

### TD-372: FTP Data Connections Are Accepted From Any Host, So Files Can Be Stolen

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Data / FtpDataServer | October 12, 2026 | 2-4 hours |

Description:
Found while tracing `FtpDataServer` and verified against a live server. `cmd_retr()` calls
`accept()` on the PASV socket and takes the **first connection from any address**; it never checks
that the peer matches the control connection's client IP. A third socket that connected to the
advertised data port before the real client received the entire file (300,000 of 300,000 bytes), and
the server logged `226 Transfer complete` and marked the token consumed. **The same worked under FTPS
with `PROT P`**: the data-channel `SSL_accept()` accepts any TLS client (no session-reuse or peer
binding), so a TLS-capable thief also received all 300,000 bytes. PASV ports come lowest-first from a
small fixed range (default 50000–50099), so an attacker on the network only has to keep connecting
to them. Consequences: dataset confidentiality is lost, and the real trainer's download fails while
its token is consumed, so every download can be denied as well. The per-file token and TLS design is
bypassed entirely at the data channel.

Action Items:

- [ ] Record the control connection's peer address; after `accept()` on the PASV socket, compare the data peer's address and reject (close, keep listening until the timeout) any mismatch.
- [ ] Under `PROT P`, additionally require TLS session resumption from the control channel (as vsftpd's `require_ssl_reuse` does) or at least the peer-IP check above.
- [ ] Don't consume the token unless the transfer went to the authenticated peer.
- [ ] Add a live-session test: a foreign connection to the PASV port gets nothing and the real client still succeeds.

Location in code: `src/FtpDataServer.hpp` (`cmd_retr()`, `ConnState`); tagged `TODO: See TD-372`.

Files to Modify:

- `src/FtpDataServer.hpp`
- `tests/FtpDataServerTests.cpp`

---

### TD-373: `--ftps` Doesn't Require Encryption

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / FtpDataServer | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `FtpDataServer` and verified. With `--ftps` enabled, nothing requires
`AUTH TLS` before `USER`/`PASS` or `PROT P` before `RETR`: a plaintext client logged in and downloaded
the full file from the FTPS server. Credentials and data can therefore cross the network unencrypted
on a server operators believe is "FTPS". `DataTransport` requests TLS (`CURLUSESSL_ALL`), so today's
trainer is protected unless the connection is downgraded or a client is misconfigured.

Action Items:

- [ ] When FTPS is enabled, refuse `USER`/`PASS` with `530`/`534` until `AUTH TLS` has completed, and refuse `RETR` unless `PROT P` is set.
- [ ] Do this together with TD-339 (client certificate verification) and TD-284 (FTPS keys from config).

Location in code: `src/FtpDataServer.hpp` (`cmd_pass()`, `cmd_retr()`); tagged `TODO: See TD-373`.

Files to Modify:

- `src/FtpDataServer.hpp`
- `tests/FtpDataServerTests.cpp`

---

### TD-374: FTP Re-Login Leaks Rate-Limit Slots

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / FtpDataServer | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `FtpDataServer` and verified. `PASS` claims a per-`run_id` rate-limit slot,
but `USER` resets the connection's login without releasing it, and the end-of-session cleanup releases
only the latest `run_id`. After one connection logged in twice and quit, a run with
`max_sessions_per_run = 2` and **one** open session was refused a second (`421 Too many concurrent
sessions`). Each such connection leaks a slot until `registry_server` restarts; after
`max_sessions_per_run` leaks, that `run_id` can't download anything. libcurl doesn't re-login on one
connection, so this takes a misbehaving or hostile client.

Action Items:

- [ ] In `cmd_user()` (and before a second successful `PASS`), release the held slot and clear the login.
- [ ] Test: repeated `USER`/`PASS` on one connection, then quit, leaves no slots held.

Location in code: `src/FtpDataServer.hpp` (`cmd_user()`, `cmd_pass()`); tagged `TODO: See TD-374`.

Files to Modify:

- `src/FtpDataServer.hpp`
- `tests/FtpDataServerTests.cpp`

---

### TD-375: `FtpDataServer` Has No Connection Limits or Timeouts

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Data / FtpDataServer | October 12, 2026 | 2-4 hours |

Description:
Found while tracing `FtpDataServer`; the read behaviour was verified, the rest is by inspection.
`listener_loop()` spawns one **detached** thread per connection with no cap. `read_line()` reads a byte
at a time with **no idle timeout and no length limit**: a 2 MB line with no newline was accepted and
the connection stayed open. So any client that can reach the control port can exhaust
`registry_server`'s threads and memory before logging in. The detached threads aren't tracked, so
`stop()` (and the destructor) can't end them; a connection open at shutdown keeps running against a
destroyed `FtpDataServer` (use-after-free).

Action Items:

- [ ] Cap concurrent connections (reply `421` beyond the cap); set a control-socket receive timeout (for example 60 s pre-auth, a few minutes after login); limit command lines to 512 bytes as RFC 959 implementations commonly do.
- [ ] Track connection threads (or their sockets) so `stop()` can shut them down and join before destruction.

Location in code: `src/FtpDataServer.hpp` (`listener_loop()`, `read_line()`, `stop()`); tagged `TODO: See TD-375`.

Files to Modify:

- `src/FtpDataServer.hpp`

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

### TD-378: BLEU Smoothing Inflates Unrelated Answers and Zeroes Correct Short Ones

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Metrics / GenerationQuality | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `GenerationQualityMetrics` and verified against a textbook (Papineni et al.)
BLEU. `compute_corpus_bleu()` uses `(matches + 1) / (hypothesis n-grams + 1)` for **every** order,
including unigrams, and a fixed `log = -20` when the hypothesis has no n-grams of some order. The
header labels this "add-1 smoothing (Lin & Och 2004, method 1)", but Lin & Och's add-one smoothing
applies only to n ≥ 2. Single-pair results:

| Case | This code BLEU-4 | Textbook BLEU-4 |
|---|---|---|
| Unrelated 5-word answer (no words in common) | 0.230 (BLEU-1 0.167) | 0 |
| Same 5 words, scrambled | 0.359 | 0 |
| Exactly correct 3-word answer | 0.0067 | 0 (no 4-grams) |

So an unrelated answer's BLEU-4 exceeds its BLEU-1, and a correct short reply scores far below an
unrelated long one. The per-epoch metric (10 pairs by default) is less distorted (0.015 for fully
unrelated output) but still biased against short targets, which dominate conversational data such as
daily_dialog. The per-sample backfill (TD-273) gets the full sentence-level distortion. The tests
(`CompletelyDifferentHypothesisScoresNearZero`, `BLEU1AtLeastBLEU2AtLeastBLEU4`) pass only on their
particular examples.

Action Items:

- [ ] Use unsmoothed corpus BLEU for the per-epoch metric, or a documented standard smoothing (Chen & Cherry 2014 method 1 — epsilon for zero-match orders — or add-1 for n ≥ 2 only); keep sentence-level BLEU only behind an explicitly smoothed function.
- [ ] Handle hypotheses shorter than `n` the standard way, and fix the header's citation.
- [ ] Add tests against known reference values (for example, sacreBLEU/NLTK outputs recorded as constants): unrelated → 0, scrambled → low, short exact match.

Location in code: `src/GenerationQualityMetrics.cpp` (`compute_corpus_bleu()`), `src/GenerationQualityMetrics.hpp` (class comment); tagged `TODO: See TD-378`.

Files to Modify:

- `src/GenerationQualityMetrics.{hpp,cpp}`
- `tests/generation_quality_test.cpp`

---

### TD-379: `bleu1`/`bleu2` Are Computed but Never Stored

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Metrics / GenerationQuality | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `GenerationQualityMetrics`, by inspection. `evaluate()` computes `bleu1` and
`bleu2`, but `IMetricsReporter::update_generation_quality_metrics()` takes only BLEU-4 and the three
ROUGE values. `TrainingMetricsService` then builds a fresh `GenerationQualityScore` from those four,
so the SQLite and Postgres `bleu1`/`bleu2` columns always store `-1` for real training runs: columns
that look like data but never hold any.

Action Items:

- [ ] Pass the whole `GenerationQualityScore` through the reporter interface (and the push client/API if the dashboards should show them), or drop the two fields and columns.

Location in code: `src/TrainingMetricsService.cpp` (`update_generation_quality_metrics()`); tagged `TODO: See TD-379`.

Files to Modify:

- `src/TrainingMetricsService.{hpp,cpp}`
- `src/ChatbotTrainer.cpp`
- the `IMetricsReporter` interface

---

### TD-380: Generation-Quality Scores Aren't Comparable or Well-Sampled

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Metrics / GenerationQuality | October 12, 2026 | 1-2 hours |

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

- [ ] Document the tokenization and scale in the header, metrics API and dashboards (or report 0–100).
- [ ] Add a note to the lesson that the training metric isn't standard corpus BLEU.
- [ ] Raise the default sample size (or report a confidence interval) once TD-359 makes generation length bounded.

Location in code: `src/GenerationQualityMetrics.hpp` (`tokenize()`), `src/ChatbotTrainer.hpp` (`generation_quality_sample_size`); tagged `TODO: See TD-380`.

Files to Modify:

- `src/GenerationQualityMetrics.hpp`
- `src/ChatbotTrainer.hpp`
- `docs/lessons/reading-bleu-rouge.md`

---

### TD-381: `chatbot_api_server` Ignores `GPU_STRATEGY`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | GPU / GPUManager | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `GPUUtils.hpp`, by inspection (definite). `incremental_trainer` maps
`GPU_STRATEGY` (`background` → low-priority stream, `full` → high) into
`Matrix::gpu_try_initialize()`'s third argument. `chatbot_api_server` calls
`gpu_try_initialize(config.gpu_device_id, config.gpu_memory_fraction)` with only two arguments, so it
always gets the default low-priority stream. `config.chatbot.conf` ships `GPU_STRATEGY=full`, and
CLAUDE.md documents the key for both binaries, but on the server it has no effect: serving GPU work
always yields to other GPU tenants (including a trainer on the same card).

Action Items:

- [ ] Pass `config.gpu_strategy == GPUStrategy::BACKGROUND` as the third argument, and log the mode as `incremental_trainer` does.

Location in code: `src/ChatbotAPIServer.cpp` (GPU initialization); tagged `TODO: See TD-381`.

Files to Modify:

- `src/ChatbotAPIServer.cpp`

---

### TD-382: GPU Work From Any Thread Shares Unsynchronized `GPUManager` State

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | GPU / GPUManager | October 12, 2026 | 2-4 hours |

Description:
Found while tracing `GPUUtils.hpp`, by inspection. `GPUManager`'s state is process-wide
`inline static` with no locking: `allocated_bytes_` is a plain `size_t`, and there's one cuBLAS handle
and one stream. `Matrix::operator*` sends every product with inner dimensions of at least 32 to
`multiply_gpu()` whenever `GPUManager::is_available()`, **from any thread**. `ChatbotTrainer`'s async
BLEU/ROUGE path is commented as "deliberately still CPU-only" because it calls `generate_response()`
on a `clone()`d model, but that model's `Matrix` products still go to cuBLAS on the shared stream,
concurrently with the training thread's own GPU work. That means unsynchronized updates to the budget
counter (lost updates, wrong refusals) and one cuBLAS handle used from two host threads at once, which
NVIDIA advises against. The same applies to any other thread doing large `Matrix` multiplies while a
model runs on the GPU. CUDA-side analysis only; the SYCL manager will be checked in its own reference.

Action Items:

- [ ] Decide on the threading model: either a mutex (or atomic counter plus a per-thread cuBLAS handle) in `GPUManager`, or a thread-local "CPU only" override that the async quality thread sets so its `Matrix` math really stays on the CPU.
- [ ] Correct the async-path comment in `ChatbotTrainer::compute_generation_quality_metrics()`.
- [ ] Test (on a GPU host): concurrent `Matrix` multiplies from two threads keep the budget counter exact.

**SYCL amendment (October 12, 2026, [GPUUtils_SYCL.md](../reference/source/GPUUtils_SYCL.md)):** the
SYCL manager already guards `max_memory_bytes_`/`allocated_bytes_` with `memory_mutex_`, and
`sycl::queue` is thread-safe for submission, so on SYCL (ai-machine) this item reduces to the
cross-thread GPU use itself (the async quality thread sharing the training queue), not a counter race.

Location in code: `src/gpu/GPUUtils.hpp` (`GPUManager` state), `src/Matrix.cpp` (`operator*` dispatch), `src/ChatbotTrainer.cpp` (async quality comment); tagged `TODO: See TD-382`.

Files to Modify:

- `src/gpu/GPUUtils.hpp`
- `src/gpu/sycl/GPUUtils_SYCL.hpp`
- `src/Matrix.cpp`
- `src/ChatbotTrainer.cpp`

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
| **HIGH** | Open | GPU / Kernels | October 12, 2026 | 1-2 hours |

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

Action Items:

- [ ] Add a barrier (`__syncthreads()` / `item.barrier(local_space)`) after each `scratch[0]` read that's followed by another reduction, in all four kernels in both backends (or reduce into a second buffer).
- [ ] On a GPU host, run `compute-sanitizer --tool racecheck` (CUDA) over a training step to confirm no other shared-memory races remain.
- [ ] Do this with TD-358, before the retraining runs.

Location in code: `src/gpu/MatrixGPU.cu` and `src/gpu/sycl/MatrixGPU_SYCL.cpp` (softmax rows, LayerNorm forward, cross-entropy loss and grad); tagged `TODO: See TD-384` at all eight sites.

Files to Modify:

- `src/gpu/MatrixGPU.cu`
- `src/gpu/sycl/MatrixGPU_SYCL.cpp`

---

### TD-385: `matrix_sum_gpu` Allocates and Synchronizes on Every Call

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | GPU / Kernels | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `MatrixGPU.cu`, by inspection. `matrix_sum_gpu()` reduces recursively and
allocates a `GPUMemory` buffer at every level (`cudaMalloc`/`cudaFree`, both synchronizing), ending
with a blocking one-float download. `matrix_cross_entropy_loss_gpu()` calls it every training step,
so each step stalls the CPU on the GPU and churns the allocator. `matrix_count_below_threshold_gpu()`
(the saturation hook) has the same shape. **SYCL (October 12, 2026):** both SYCL versions have the same
shape plus a `.wait()` on the submit; the SYCL `count_below_threshold` also lacks a `size <= 0` guard.

Action Items:

- [ ] Reduce in a single launch with a grid-stride loop and atomics (or a two-level fixed buffer reused across calls), and keep the loss on the device until it's actually needed on the host.

Location in code: `src/gpu/MatrixGPU.cu` (`matrix_sum_gpu()`); tagged `TODO: See TD-385`.

Files to Modify:

- `src/gpu/MatrixGPU.cu`
- `src/gpu/sycl/MatrixGPU_SYCL.cpp`

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
| MEDIUM | Open | GPU / GPUMatrix | October 12, 2026 | 1-2 hours |

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

- [ ] Truncate encoder input once where the model tokenizes (TD-365), which removes the overflow; or size the cross-attention caches from the actual encoder length.
- [ ] Test (on a GPU host): an over-length prompt generates without throwing and matches the CPU path.

**Verified October 12, 2026** ([MatrixGPU_SYCL.md](../reference/source/MatrixGPU_SYCL.md)): on the
oneAPI CPU device, appending a 12-row encoder output to a `GPUDecoderKVCache(2, max_seq_length = 8,
4)` cross-attention cache throws `"GPUKVCache::append: exceeds max_seq_length capacity"`.

Location in code: `src/gpu/MatrixGPU.hpp` (`GPUDecoderKVCache` constructor), `src/EncoderDecoderModel.cpp` (both GPU generation paths); tagged `TODO: See TD-387`.

Files to Modify:

- `src/EncoderDecoderModel.cpp`
- `src/gpu/MatrixGPU.hpp`

---

### TD-388: `GPUKVCache` Copies the Whole Cache Every Decode Step

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | GPU / GPUMatrix | October 12, 2026 | 2-4 hours |

Description:
Found while tracing `MatrixGPU.hpp`, by inspection. `GPUKVCache::get_keys()`/`get_values()` allocate a
new `GPUMatrix` and copy the whole populated prefix on every call. Cached self-attention and
cross-attention call them once per layer per decode step, so a generation of `n` tokens copies
O(n²·d_model) floats per layer and performs two allocations per layer per step. That works against
TD-050's goal (no per-step device malloc/free churn) and will show up in TD-033/TD-050's still-open
GPU latency benchmark.

Action Items:

- [ ] Return a non-owning view (pointer, rows, cols) of the populated prefix and have attention consume it directly (cuBLAS/oneMKL take a leading dimension, so no copy is needed).

Location in code: `src/gpu/MatrixGPU.hpp` (`GPUKVCache::materialize()`); tagged `TODO: See TD-388`.

Files to Modify:

- `src/gpu/MatrixGPU.hpp`
- `src/MultiHeadAttention.cpp`
- `src/CrossAttention.cpp`

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

- `src/gpu/MatrixGPU.hpp`
- `src/gpu/sycl/MatrixGPU_SYCL.hpp`
- `tests/matrixgpu_td003_test.cpp`

---

### TD-390: SYCL Deferred Frees and Blocking Copies Stall the In-Order Queue

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | GPU / GPUManager | October 12, 2026 | 4-8 hours |

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
- [ ] Replace per-free `host_task`s with a caching allocator: keep freed blocks in a size-bucketed pool tied to the in-order queue (safe to reuse immediately, since later kernels on the same queue run after earlier ones), and return memory to USM only at `cleanup()` or under budget pressure.
- [ ] Make uploads asynchronous where the host buffer outlives the copy (or stage through a pinned buffer), keeping `.wait()` only on downloads.

Location in code: `src/gpu/sycl/GPUUtils_SYCL.hpp` (`GPUMemory::defer_free()`, `copy_from_host()`/`copy_to_host()`); tagged `TODO: See TD-390`.

Files to Modify:

- `src/gpu/sycl/GPUUtils_SYCL.hpp`
- `src/ChatbotTrainer.cpp` (remove the accounting workaround syncs afterwards)

---

### TD-391: SYCL Lists the Same GPU Once per Platform and Doesn't Log the Backend

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | GPU / GPUManager | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `GPUUtils_SYCL.hpp`, by inspection. `enumerate_gpu_devices()` collects
`device_type::gpu` devices from **every** SYCL platform. With both the Level Zero and OpenCL runtimes
installed (the usual oneAPI setup, and what ai-machine's driver packages provide), one Intel Arc card
appears once per platform, so `device_count()` is 2 and `GPU_DEVICE_ID=0` means "whichever platform
enumerates first". The startup log prints the device name via `get_device_info()`, but not the
platform or backend, so nobody can tell whether training runs on Level Zero (the expected, faster path
for oneMKL) or OpenCL. Given ai-machine's driver-instability history (SIGSEGV in `libigc`, see
[ONEAPI_SYCL_DRIVER_SEGFAULT.md](../../operations/guides/troubleshooting/ONEAPI_SYCL_DRIVER_SEGFAULT.md)), knowing the backend matters for diagnosis too.

Action Items:

- [ ] Prefer one backend (Level Zero first, OpenCL as fallback) and de-duplicate physical devices, or document `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` in the deployment config.
- [ ] Log the platform and backend in `get_device_info()`/the startup line.
- [ ] Ask the user to check ai-machine's current selection (`sycl-ls`, or the new log line).

Location in code: `src/gpu/sycl/GPUUtils_SYCL.hpp` (`enumerate_gpu_devices()`, `get_device_info()`); tagged `TODO: See TD-391`.

Files to Modify:

- `src/gpu/sycl/GPUUtils_SYCL.hpp`

---

### TD-392: SYCL `GPUMemory(0)` Throws Where CUDA Succeeds

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | GPU / GPUManager | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `GPUUtils_SYCL.hpp` and verified: `sycl::malloc_device<float>(0, q)` returns
`nullptr`, which `GPUMemory`'s constructor treats as an allocation failure and throws
(`"sycl::malloc_device failed: allocation returned null"`). CUDA's `cudaMalloc(0)` succeeds, so a
zero-row `GPUMatrix` (for example, an empty token sequence reaching a GPU path) throws only on SYCL,
with a misleading message.

Action Items:

- [ ] Treat `count == 0` as a valid empty allocation (skip `malloc_device`, keep `device_ptr_` null) in the SYCL `GPUMemory`, matching CUDA.

Location in code: `src/gpu/sycl/GPUUtils_SYCL.hpp` (`GPUMemory` constructor); tagged `TODO: See TD-392`.

Files to Modify:

- `src/gpu/sycl/GPUUtils_SYCL.hpp`

---

### TD-393: SYCL Sub-Group Reduction Is Wrong When the Sub-Group Size Is 8

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | GPU / Kernels | October 12, 2026 | 1-2 hours |

Description:
Found while tracing `MatrixGPU_SYCL.cpp` and verified on the oneAPI CPU device.
`workgroup_reduce_sum()` (used by `matrix_sum_gpu` and `matrix_count_below_threshold_gpu`) reduces
within each sub-group, stores one partial per sub-group, and then has **only sub-group 0** combine the
`256 / S` partials. That's correct only if `S ≥ 16`. No `reqd_sub_group_size` is set, so the compiler
chooses `S`; the CPU device chose 8, so 8 of 32 partials were summed per level: 256 ones → 64,
1,000 → 256, 1,000,000 → 65,536; the saturation count came out 427 instead of 1,667. The
cross-entropy loss also goes through `matrix_sum_gpu`, so it's wrong once `seq_len > 64`.

ai-machine's Arc B60 (Xe2/Battlemage) supports only sub-group sizes 16 and 32, so it's probably
latent there, but nothing in the code guarantees it. Arc A-series (Alchemist) and CPU devices can use
8, and the CUDA version (warp 32) is unaffected.

Action Items:

- [ ] Have sub-group 0 loop over all partials (`for (i = lane; i < num_subgroups; i += S)`), or pin `[[sycl::reqd_sub_group_size(16)]]` (or 32) on the reduction kernels.
- [ ] Cover it with TD-394's CPU-device test mode (sizes 255, 256, 257, 1e6).

Location in code: `src/gpu/sycl/MatrixGPU_SYCL.cpp` (`workgroup_reduce_sum()`); tagged `TODO: See TD-393`.

Files to Modify:

- `src/gpu/sycl/MatrixGPU_SYCL.cpp`

---

### TD-394: GPU Kernel Tests Can't Run Without a GPU

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | GPU / Testing | October 12, 2026 | 2-4 hours |

Description:
Found while tracing `MatrixGPU_SYCL.cpp`. The GPU kernel tests (`gpuutilsTests`,
`matrixgpuTd003Tests`, the GPU cases in `feedforward_test.cpp`/`multiheadattention_test.cpp`) skip
whenever `GPUManager::probe()` finds no GPU, and the SYCL `GPUManager` enumerates GPU devices only. No
development or CI environment used so far has a GPU, so **no GPU kernel had ever been run** by the
test suite. A scratch harness that let the SYCL backend use the oneAPI CPU device ran every kernel in
minutes and found two real bugs: TD-384 (verified deterministically) and TD-393. A CPU-device mode
can't prove GPU timing behaviour, but it checks numerics and catches races that a sequential CPU
schedule exposes.

Action Items:

- [ ] Add an opt-in CPU fallback to the SYCL `GPUManager` (for example `ADAI_SYCL_ALLOW_CPU=1`, test-only), and run `gpuutilsTests`/`matrixgpuTd003Tests` plus a kernel-vs-CPU-reference suite (every `matrix_*_gpu`) in `build/sycl-gputest`.
- [ ] Add LayerNorm forward/backward, softmax, cross-entropy and reduction-size cases (the harness from MatrixGPU_SYCL_cpp.md §2 is a ready template).
- [ ] Optionally add a CI job using the oneAPI CPU runtime.

Location in code: `src/gpu/sycl/GPUUtils_SYCL.hpp` (`enumerate_gpu_devices()`); tagged `TD-394` alongside TD-391's TODO.

Files to Modify:

- `src/gpu/sycl/GPUUtils_SYCL.hpp`
- `tests/matrixgpu_td003_test.cpp`
- `tests/CMakeLists.txt`

---

### TD-395: `GPUMatrix` Is Duplicated Between the CUDA and SYCL Headers

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | GPU / GPUMatrix | October 12, 2026 | 2-4 hours |

Description:
Found while tracing `MatrixGPU_SYCL.hpp`, by inspection. `GPUMatrix` is defined twice: in the CUDA
block of `src/gpu/MatrixGPU.hpp` and again in `src/gpu/sycl/MatrixGPU_SYCL.hpp`. The two are the same
member for member except `copy()` (CUDA: async device copy; SYCL: `q.memcpy(...).wait()`) and `zero()`
(`cudaMemsetAsync` vs. `q.memset`). Every `GPUMatrix` change has to be made twice, and a missed one
makes the backends diverge silently. `GPUKVCache` already shows the alternative: one definition after
the backend split, using only the shared API. The SYCL `copy()`'s `.wait()` is also an unnecessary
full-queue drain on an in-order queue, against CLAUDE.md's "no `.wait()` on intermediate work" rule
(compare TD-390). The SYCL header's status note ("capped by TD-050") is stale, since TD-050 is mostly
resolved.

Action Items:

- [ ] Add a backend `matrix_zero_gpu()` (and use `matrix_copy_device_to_device_gpu()` for `copy()`), then move `GPUMatrix` below the backend split in `MatrixGPU.hpp` as a single definition and delete the SYCL copy.
- [ ] Drop the `.wait()` from `copy()`; refresh the SYCL header's status note.

Location in code: `src/gpu/MatrixGPU.hpp` and `src/gpu/sycl/MatrixGPU_SYCL.hpp` (`GPUMatrix`); tagged `TODO: See TD-395`.

Files to Modify:

- `src/gpu/MatrixGPU.hpp`
- `src/gpu/sycl/MatrixGPU_SYCL.hpp`
- `src/gpu/MatrixGPU.cu`
- `src/gpu/sycl/MatrixGPU_SYCL.cpp`

---

### TD-396: Hippocampal Memory Eviction Starves New Memories

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Memory / HippocampalMemory | October 13, 2026 | 2-4 hours |

Description:
Found while tracing `HippocampalMemory` and verified by simulation with the real class. TD-193's
least-used eviction removes the slot with the lowest `coverage_`, but a newly written slot starts at
coverage 0, while every older slot carries accumulated, decayed coverage. Under `DecoderBlock`'s
bookkeeping (each forward step: `decay_coverage(0.95)`, then add each slot's mean attention weight),
a new slot needs about `1/(1 − γ)` = 20 steps to approach the others. With near-uniform attention it
never catches up before the next write, so it is the slot evicted next. Simulation (capacity 8, one
write per "response" followed by 50 forward steps, 40 responses): with uniform attention the
survivors were writes **0–6** plus the newest, so the first `capacity − 1` memories are permanent and
nothing new persists. With peaked random attention turnover was normal (20, 22, 26, 33, 35, 37, 38,
39). Near-uniform attention is the likely case in shipped models: the hippocampal cross-attention gets
no gradient while its gate is stuck at 0 (TD-341), and keys are mean-pooled world-model embeddings of
whole responses. The failure is masked today only because TD-341 zeroes the path's output.

Action Items:

- [ ] Give new slots a protection window (not evictable for N steps/writes), or start them at the current mean coverage, or evict by `coverage / age` (or LRU on last significant attention).
- [ ] Add a test that simulates the decay-and-accumulate bookkeeping under uniform attention and asserts recent writes survive.
- [ ] Do this together with TD-341.

Location in code: `src/HippocampalMemory.cpp` (`evict_least_used()`, `insert_slot()`); tagged `TODO: See TD-396`.

Files to Modify:

- `src/HippocampalMemory.{hpp,cpp}`
- `tests/hippocampalmemory_test.cpp`

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

### TD-401: `incremental_trainer` Flag Parsing Crashes or Silently Misreads Bad Values

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / CLI | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainerArgs` and verified against `build/debug/bin/incremental_trainer`:
- `--admin-port abc` **aborts the process**: `std::stoi` throws and `main()` calls
  `parse_incremental_trainer_global_args()` outside any `try` (`terminate called … what(): stoi`, exit
  134). An out-of-range value throws `out_of_range` the same way.
- `--admin-port 8084x` is accepted as 8084, and no port range is checked.
- `--objective=` isn't validated: `main()` only tests `== "lejepa"` and `resolve_dataset_kind()` maps
  anything else to `chatbot`, so `--objective=lejpa` was accepted silently. With `train`, that runs
  chatbot training (on the chatbot pool) instead of world-model pretraining. The test
  `UnrecognizedObjectiveFallsBackToChatbot` asserts this fallback.
- A flag missing its value (for example a trailing `--config`) is appended to the positional
  arguments instead of rejected (`status --config` carried on); the tests specify this too.

Action Items:

- [ ] Parse `--admin-port` with full-string, range-checked parsing (1–65535) and report a usage error; wrap the parse in `main()`'s error handling.
- [ ] Reject unknown `--objective` values (`chatbot`/`lejepa` only) and flags missing their value, with a clear message; update the tests that assert the permissive behaviour.

Location in code: `src/IncrementalTrainerArgs.cpp` (`parse_incremental_trainer_global_args()`); tagged `TODO: See TD-401`.

Files to Modify:

- `src/IncrementalTrainerArgs.cpp`
- `src/IncrementalTrainingTool.cpp`
- `tests/incremental_trainer_args_test.cpp`

---

### TD-402: `--dataset-kind` Is Not Validated

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / CLI | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainerArgs`, by inspection. `--dataset-kind <k>` is stored as given
and `resolve_dataset_kind()` returns it unchanged. With `RemoteTransport` it becomes a URL segment,
which `registry_server` rejects (404) unless it's `encoder`/`decoder`/`world_model`/`chatbot`. With
`LocalTransport`, `DatasetRegistry` joins it as `session_dir + "/" + kind`, so a typo silently selects
a new, empty sub-pool ("no pending data") and a value containing `..` or `/` puts the local registry
files outside `session_dir`.

Action Items:

- [ ] Validate the value against the four kinds (shared with `dataset_manager --kind` and `DATASET_KIND`) and reject anything else.

Location in code: `src/IncrementalTrainerArgs.cpp` (`--dataset-kind`); tagged `TODO: See TD-402`.

Files to Modify:

- `src/IncrementalTrainerArgs.cpp`
- `src/DatasetRegistry.cpp`

---

### TD-403: `trainer_service` Drops `--objective` and `--dataset-kind`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / CLI | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainerArgs`, by inspection. `trainer_service` parses its command
line with the same parser but forwards only `--config`, `--model` and `--gpu-strategy` to each child
(plus `--foreground --admin-port N resume`). `--objective` and `--dataset-kind` given to
`trainer_service` are silently ignored, so the always-on service can only run chatbot training, never
LeJEPA world-model pretraining.

Action Items:

- [ ] Decide whether `trainer_service` should support `--objective=lejepa` (forward both flags) or reject them with a clear message.

Location in code: `src/TrainerServiceMain.cpp` (child argument list); tagged `TODO: See TD-403`.

Files to Modify:

- `src/TrainerServiceMain.cpp`

---

### TD-404: `derive_run_id()` Can Collide Across Hosts

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / CLI | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainerArgs`, by inspection. `derive_run_id()` returns
`RUN_ID` if set, else the first 8 characters of the hostname plus `_` plus `pid % 10000`. The LeJEPA
pretraining path always uses it (it never calls `begin_run()`), and `train`/`retrain` use it when MNS
doesn't allocate a run id. With a shared remote registry, run ids decide who owns claimed files, so
two hosts with the same 8-character prefix (for example `ai-machi…`) and a coinciding PID tail would
share ownership.

Action Items:

- [ ] Use the full hostname plus PID plus a random suffix (or a UUID), or allocate LeJEPA run ids from MNS like the chatbot path.

Location in code: `src/IncrementalTrainerArgs.cpp` (`derive_run_id()`); tagged `TODO: See TD-404`.

Files to Modify:

- `src/IncrementalTrainerArgs.cpp`
- `src/IncrementalTrainingTool.cpp`

---

### TD-405: Each Training Pass Restarts From the Best-Ever Checkpoint, Not the Latest

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Training / IncrementalTrainer | October 13, 2026 | 2-4 hours |

Description:
Found while tracing `IncrementalTrainer` and verified with the real class. All three constructors,
and `resume_last_session()`, load the session with the **lowest `final_validation_loss` across the
whole history**. But each pass's validation loss comes from a validation split of that pass's own
files, so losses from different passes aren't comparable. When a later pass trains on new data whose
validation loss is higher, the next pass starts from the older checkpoint again and the later
learning is discarded, while the registry has already marked those files trained. Verified: with
session 0 (val 1.20) and a newer session 1 (val 2.00), construction logged `Best checkpoint:
…/session_0_checkpoint.bin` and loaded it. In incremental training over a stream of datasets, the
model can stay pinned to one old checkpoint indefinitely; this could explain apparent lack of
progress across ai-machine passes.

Action Items:

- [ ] Resume from the most recent completed session by default; keep "best" selection as an explicit option (or compare losses only on a fixed, shared validation set).
- [ ] Apply the same rule in `resume_last_session()`, and log which checkpoint was chosen and why.
- [ ] Test: two sessions where the newer has higher validation loss → the newer is resumed.

Location in code: `src/IncrementalTrainer.cpp` (three constructors' best-selection loops, `resume_last_session()`); tagged `TODO: See TD-405`.

Files to Modify:

- `src/IncrementalTrainer.cpp`
- `tests/incrementaltrainer_test.cpp`

---

### TD-406: A Paused or Crashed Pass's Checkpoint Is Deleted on the Next Start

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Training / IncrementalTrainer | October 13, 2026 | 2-4 hours |

Description:
Found while tracing `IncrementalTrainer` and verified with the real class. `perform_auto_save()` (sample
cadence, forced `/admin/checkpoint`, and the `/admin/pause` drain path) writes `auto_save_session_N.bin`
and persists session N to `session_history.txt` with `final_validation_loss = 0`. On the next
construction:
1. `current_session_id` becomes N+1, so the in-progress check looks for `session_<N+1>_best.bin`
   and misses session N's best snapshot.
2. `cleanup_dead_sessions()` deletes session N's auto-save as an "orphaned snapshot from a
   crashed/superseded session".
3. It drops session N from the history ("Removing broken session") because its val loss is 0.

Verified: history with session 0 complete and session 1 paused (auto-save plus `session_1_best` on
disk) → the restart loaded session 0, removed session 1 and its auto-save, and left `session_1_best`
on disk unused forever. So TD-173's "pause drains the pass and checkpoints it" produces a checkpoint
the next pass deletes; the released files are retrained from older weights, and crash recovery from
auto-saves never happens.

Action Items:

- [ ] On startup, detect an interrupted session (val loss 0 with an existing auto-save or best snapshot) and resume from its newest snapshot instead of deleting it; keep `current_session_id` aligned or look up the snapshot by the interrupted session's id.
- [ ] Only treat snapshots as orphaned when no history entry references their session.
- [ ] Test: pause → restart resumes from the paused pass's snapshot.

Location in code: `src/IncrementalTrainer.cpp` (`perform_auto_save()`, `resume_last_session()`, `cleanup_dead_sessions()`); tagged `TODO: See TD-406`.

Files to Modify:

- `src/IncrementalTrainer.cpp`
- `tests/incremental_trainer_control_test.cpp`

---

### TD-407: Checkpoint Deletion Leaks the Per-Layer Weight Files

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Training / IncrementalTrainer | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainer` and measured. `EncoderDecoderModel::save_model()` writes
the top-level sidecars (`.config`, `.vocab`, `.encoder`, `.decoder`, `.lm_head`) plus per-layer files
(`_token_emb.bin`, `_encoder_block_<i>.bin` + `_attention.bin`/`_feedforward.bin`,
`_decoder_block_<i>.bin` + `.self_attn`/`.cross_attn`/`.ff`, `_final_norm.bin`), and the per-layer files
hold almost all the weights. `remove_model_files()` deletes only the five top-level sidecars plus
`.metrics.csv`: on a d_model 256, 4 + 4 layer checkpoint it removed **0.1 of 29.6 MB (0.2%)**. Every
retention rotation (`MAX_SESSIONS_TO_KEEP`), superseded best snapshot, broken-session cleanup and
`reset` therefore leaves almost a whole checkpoint behind. `cleanup_dead_sessions()` and
`reset_all()` scan for `*.bin` base files that `save_model()` never writes, so they only match
per-layer files by accident (partially deleting snapshots) and never the attention/FF files. Disk use
in `SESSION_DIR` grows by roughly one checkpoint per pass (plus snapshots and auto-saves) with no
bound.

Action Items:

- [ ] Have `EncoderDecoderModel` expose (or return from `save_model()`) the list of files a checkpoint consists of, and delete exactly those in `remove_model_files()`, `cleanup_dead_sessions()` and `reset_all()`; or save each checkpoint into its own directory and remove the directory.
- [ ] Add a test that save → remove leaves no files behind; offer a one-off cleanup command for existing session directories (ask the user to check ai-machine's `SESSION_DIR` size).

Location in code: `src/IncrementalTrainer.cpp` (`remove_model_files()`, `cleanup_dead_sessions()`, `reset_all()`); tagged `TODO: See TD-407`.

Files to Modify:

- `src/IncrementalTrainer.cpp`
- `src/EncoderDecoderModel.{hpp,cpp}`

---

### TD-408: `latest`/`best` Checkpoint Symlinks Always Dangle

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / IncrementalTrainer | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainer`, by inspection. `create_or_update_symlink()` links
`latest_checkpoint.bin`/`best_checkpoint.bin` (in the working directory) to the **bare** checkpoint path,
which `save_model()` never creates, so the links always dangle. Loading through a link looks for
`best_checkpoint.bin.config`, which isn't a link, so neither can be used as `MODEL_PATH`. On Windows
the fallback `copy_file()` copies the same nonexistent bare file and always fails.

Action Items:

- [ ] Link a checkpoint directory (see TD-407) or create links for every sidecar; or replace the links with a small text file naming the checkpoint base path that tools read.

Location in code: `src/IncrementalTrainer.cpp` (`create_or_update_symlink()`); tagged `TODO: See TD-408`.

Files to Modify:

- `src/IncrementalTrainer.cpp`

---

### TD-409: Tokenized-Data Cache Key Has No Preprocessing Version

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / IncrementalTrainer | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainer`, by inspection. `compute_tokenized_cache_key()` hashes the
files' content checksums and segment ranges, the vocab checksum, the tokenizer mode and
`max_seq_length`, but no version of `ChatbotTrainer`'s preprocessing. A code fix that changes the cached
token IDs (for example TD-361's double `BOS`, or TD-267's skipped pairs) would silently reuse old cached
data for every file already cached, so the fix wouldn't take effect until the cache directory was
cleared by hand.

Action Items:

- [ ] Add a `kTokenizedCacheFormatVersion` constant to the key and bump it with any preprocessing change; clear the cache as part of the TD-358/TD-361 retraining.

Location in code: `src/IncrementalTrainer.cpp` (`compute_tokenized_cache_key()`); tagged `TODO: See TD-409`.

Files to Modify:

- `src/IncrementalTrainer.cpp`
- `src/ChatbotTrainer.{hpp,cpp}`

---

### TD-410: `IncrementalTrainer` Session Bookkeeping and Hard-Coded Training Policy

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / IncrementalTrainer | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainer`, by inspection:
- `run_training()` calls `finalize_session(…, num_epochs, …)` with the **requested** epoch count, not the
  completed one, although early stopping is always on.
- `make_incremental_config()` hard-codes `enable_early_stopping = true`, `patience = 5` and
  `restore_best_weights = true`, and the config-file constructor forces `LRSchedule::WARMUP_COSINE`;
  none is configurable.
- `IncrementalConfig::save_incremental_checkpoints` and `checkpoint_dir` are never read.
- `update_best_checkpoint()` returns early when symlinks are disabled, so in-process best tracking
  isn't updated in that mode.
- `retrain` (`reset_model_for_config()`) clears and saves an empty history, orphaning every earlier
  checkpoint (which TD-407 then never removes).

Action Items:

- [ ] Pass `epochs_fully_completed`; expose early-stopping and schedule settings through `ServiceConfig`; remove the unused fields; separate best tracking from symlink updates; archive or delete old checkpoints on `retrain`.

Location in code: `src/IncrementalTrainer.cpp` (`run_training()`, `make_incremental_config()`), `src/IncrementalTrainer.hpp` (`IncrementalConfig`); tagged `TODO: See TD-410`.

Files to Modify:

- `src/IncrementalTrainer.{hpp,cpp}`
- `src/Config.{hpp,cpp}`

---

### TD-411: `resume` (Every `trainer_service` Pass) Never Downloads Over FTP

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Training / IncrementalTrainingTool | October 13, 2026 | 2-4 hours |

Description:
Found while tracing `IncrementalTrainingTool`, by inspection (definite). `trainer_service` runs
`incremental_trainer --foreground --admin-port N resume` for every pass, and `resume` calls
`IncrementalTrainer::resume_last_session()`. That function acquires pending files and then trains on
`resp.registry_paths()` directly, under the comment "resume_last_session runs on the same machine as
the registry … ftp_server_host is always empty here". That's only true for a local registry. With
`REGISTRY_SERVER_URL` pointing at a remote `registry_server` that delivers files over FTP (TD-040's
transport), these are the registry host's paths; the `DataTransport::fetch_all()` download that `train`
performs never happens, so the trainer can't open the files. With TD-412 the pass then "succeeds" and
marks them trained.

Action Items:

- [ ] Factor `train`'s acquire → (FTP download) → train → mark/release → cleanup sequence into one function and use it from `resume_last_session()` (with the segment ranges).
- [ ] Ask the user whether ai-machine's `trainer_service` uses a remote FTP registry; if so, audit and re-queue files marked trained since it went live.

Location in code: `src/IncrementalTrainer.cpp` (`resume_last_session()`), `src/IncrementalTrainingTool.cpp` (`resume` command); tagged `TODO: See TD-411`.

Files to Modify:

- `src/IncrementalTrainer.cpp`
- `src/IncrementalTrainingTool.cpp`

---

### TD-412: A Training Pass That Loads Zero Samples Marks Its Files Trained

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open | Training / IncrementalTrainingTool | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainingTool` and verified with `IncrementalTrainer` in a scratch
program. `ChatbotTrainer::train()` has no empty-data check, and neither do `train_on_files()` or
`retrain_on_files()`. Given a nonexistent file, the pass logged `Cannot open file`, `Total samples
loaded: 0`, `Training for 2 epochs...` and `Incremental training session completed successfully`, and
`train_on_files()` **returned `true`**. Every caller then calls `mark_trained()`, so a missing,
unreadable or empty file (or every file in a TD-411 remote-FTP `resume` pass) is recorded as trained
without any training, and a zero-sample session is added to the history.

Action Items:

- [ ] Treat zero loaded (or zero tokenized) samples as failure in `train_on_files()`/`retrain_on_files()` (and guard in `ChatbotTrainer::train()`), so callers release the files; log which files failed to load.
- [ ] Test: a nonexistent file → `false` → released, not marked trained.

Location in code: `src/IncrementalTrainer.cpp` (`train_on_files()`, `retrain_on_files()`), `src/ChatbotTrainer.cpp` (`train()`); tagged `TODO: See TD-412`.

Files to Modify:

- `src/IncrementalTrainer.cpp`
- `src/ChatbotTrainer.cpp`
- `tests/incrementaltrainer_test.cpp`

---

### TD-413: Standalone `train`/`retrain` Pre-Check Reads the Wrong Pending Pool

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / IncrementalTrainingTool | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainingTool` and verified with the built binaries. Without
`REGISTRY_SERVER_URL`, `train`'s pre-fork check (and `retrain`'s count) reads
`SESSION_DIR/pending_files.txt`. But `incremental_trainer` always sets `DATASET_KIND` (TD-204), so
`LocalTransport` keeps the pool in `SESSION_DIR/<kind>/pending_files.txt` (TD-202). After `dataset_manager
--kind chatbot add d.jsonl`, `incremental_trainer train 1` printed "No pending data. Use DatasetManager to
queue training data." Queuing without `--kind` passes the check, but the child's acquire looks in the
kind pool and finds nothing. Standalone training can't start from a correctly queued pool.

Action Items:

- [ ] Do the pre-check through `DatasetRegistry` (which applies the kind) in both modes, counting only files this model/kind can acquire.

Location in code: `src/IncrementalTrainingTool.cpp` (`train` pre-fork check, `retrain` count); tagged `TODO: See TD-413`.

Files to Modify:

- `src/IncrementalTrainingTool.cpp`

---

### TD-414: First-Run Vocabulary Bootstrap Is Unreachable From the CLI

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / IncrementalTrainingTool | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainingTool` and verified with the built binary. Every command
constructs `IncrementalTrainer(vocab, model, cfg)`, whose three-argument constructor calls
`build_model()` → `load_vocab()` unconditionally. The first-run vocabulary bootstrap
(`pending_vocab_build_` → `bootstrap_vocab()`) exists only in the config-file constructor, which
`main()` never uses. On a fresh `SESSION_DIR` with data queued and no `vocab.txt`, `train 1` aborted
with `terminate called … VocabularyFileError: Could not open vocabulary file: vocab.txt` (exit 134), and
so did `status` and `history`. A vocabulary has to be built separately first (`vocab_builder`), which
the usage text doesn't mention; this matches the "missing `VOCAB_PATH`" crash from ai-machine's first
deployment.

Action Items:

- [ ] Move the missing-vocab handling into the three-argument constructor (defer `build_model()` until `bootstrap_vocab()`), and make `status`/`history` work without a model.
- [ ] Catch exceptions in `main()`'s command dispatch with a clear message.

Location in code: `src/IncrementalTrainingTool.cpp` (`train` construction, `status`), `src/IncrementalTrainer.cpp` (constructors); tagged `TODO: See TD-414`.

Files to Modify:

- `src/IncrementalTrainer.cpp`
- `src/IncrementalTrainingTool.cpp`

---

### TD-415: LeJEPA Pass Ignores Segment Ranges and Strands Claims After a Crash

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / IncrementalTrainingTool | October 13, 2026 | 2-4 hours |

Description:
Found while tracing `IncrementalTrainingTool`, by inspection. `run_lejepa_training_pass()` loads each
acquired file with `load_conversation_pairs(path, pairs)`, without the TD-205 segment range the chatbot
path passes, so a file queued as N segments is loaded in full N times when its segments are claimed.
Its run id is always `derive_run_id()` (hostname prefix + PID tail, TD-404), never MNS-allocated, and
claimed files are only reclaimable by the same run id. After a crash, kill, or SIGTERM (whose handler
exits without releasing, TD-416), the claimed files stay claimed by a run id no later process will have,
until released by hand.

Action Items:

- [ ] Pass `file_ranges()` through the LeJEPA loader; allocate a stable run id for the world model (MNS, or a persisted one per world-model directory) so a restarted pass can reclaim its own files.

Location in code: `src/IncrementalTrainingTool.cpp` (`run_lejepa_training_pass()`); tagged `TODO: See TD-415`.

Files to Modify:

- `src/IncrementalTrainingTool.cpp`

---

### TD-416: `incremental_trainer` Command Hygiene

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / IncrementalTrainingTool | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IncrementalTrainingTool`; the first item verified:
- `train abc`/`retrain abc`: the epoch argument goes through an uncaught `std::stoi`, so the process
  aborts (exit 134).
- The SIGTERM/SIGINT handler logs through spdlog and calls `std::exit(0)` (not async-signal-safe; runs
  static destructors under the training thread), doesn't checkpoint or release claimed files, and exits
  0.
- `init` and `retrain` override `AUTO_SAVE_*` with hard-coded 30 min / 1,000 samples.
- The LeJEPA path's tokenizer is always ASCII (`TOKENIZER_MODE` ignored).
- `status`/`history` build a full `IncrementalTrainer`, which loads the best checkpoint and runs the
  deleting `cleanup_dead_sessions()` (TD-406/TD-407), just to print.
- The interactive MNS model pick reads stdin (EOF under systemd, which is the intended failure but
  undocumented).

Action Items:

- [ ] Parse epochs with validation; have the signal handler only set a flag that the training loop polls (then checkpoint, release, exit non-zero); use the configured auto-save values; thread `TOKENIZER_MODE` to LeJEPA; give `status`/`history` a read-only path.

Location in code: `src/IncrementalTrainingTool.cpp` (`signal_handler()`, `train`/`retrain`/`init`/`status`, LeJEPA vocab probe); tagged `TODO: See TD-416`.

Files to Modify:

- `src/IncrementalTrainingTool.cpp`

---

### TD-417: `--integrated-inference` Answers Differ From Normal `/chat`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / IntegratedInferenceEngine | October 13, 2026 | 2-4 hours |

Description:
Found while tracing `IntegratedInferenceEngine` and verified against a small real model.
`generate_from_encoder_output()` always picks the argmax; the request's `strategy` (passed through by
`ChatbotAPI`) and its temperature/top-k/top-p/beam settings are never consulted: `submit(..., "top_k")`
returned identical greedy output twice. The batcher tokenizes with `tokenizer_->encode(text)`, whose
default adds `BOS`/`EOS` (the model's encoder is trained without them, TD-364). The loop also has no
invalid-vocab masking, no repetition handling, no hippocampal/world-model inputs, and doesn't cap
`max_length` at `max_seq_length` (TD-347). On a small untrained model, 3 of 3 prompts produced different
text from `generate_response_with_strategy(…, "greedy", …)`. So `--integrated-inference` serves
different answers than the normal `/chat` path, for reasons unrelated to performance.

Action Items:

- [ ] Generate through `EncoderDecoderModel`'s own strategy dispatch (or `TextGenerator` with the request's config) and tokenize with `encode(text, false)`, so every engine path produces the same answers as plain `/chat`.
- [ ] Test: integrated vs. plain generation agree for the same request.

Location in code: `src/IntegratedInferenceEngine.hpp` (`generate_from_encoder_output()`, batcher tokenization); tagged `TODO: See TD-417`.

Files to Modify:

- `src/IntegratedInferenceEngine.hpp`
- `tests/integratedinferenceengine_test.cpp`

---

### TD-418: `IntegratedInferenceEngine` Worker Exceptions Terminate the Server; Shutdown Drops Requests

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / IntegratedInferenceEngine | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IntegratedInferenceEngine`; the shutdown part verified. The encoder thread calls
`encode_with_mask()` and the decoder thread calls `forward_with_encoder()`/`lm_head_->forward()` with no
`try`. Any exception (for example a token id outside the model's vocab when the tokenizer and model
disagree) escapes the `std::thread` and calls `std::terminate()`, taking `chatbot_api_server` down, and
the rest of the batch's promises are never fulfilled. `shutdown()` stops the queues and joins the
threads without failing queued or in-flight requests: destroying an engine with three queued requests
left all three futures throwing `std::future_error` (broken promise). Same class as TD-212 for
`BatchedInferenceEngine`.

Action Items:

- [ ] Catch per request in the encoder and decoder stages and deliver the exception to that request's promise; on shutdown, drain the queues and fail pending promises with a clear "shutting down" error.

Location in code: `src/IntegratedInferenceEngine.hpp` (`encoder_worker()`, `generate_from_encoder_output()`, `shutdown()`); tagged `TODO: See TD-418`.

Files to Modify:

- `src/IntegratedInferenceEngine.hpp`

---

### TD-419: `IntegratedInferenceEngine` Uses the Model Without Its Mutex

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / IntegratedInferenceEngine | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IntegratedInferenceEngine`, by inspection. The engine holds raw pointers to the
model's encoder, decoder and LM head and runs them from its own encoder and decoder threads without
`EncoderDecoderModel::model_mutex_`. `/chat/batch` bypasses the engine dispatcher (TD-234) and calls
`model_->forward()`, which takes that mutex, so a batch request and an integrated `/chat` request can run
the **same layer objects** at the same time, racing on their per-instance forward caches (the TD-156
class of bug that `model_mutex_` exists to prevent).

Action Items:

- [ ] Have the engine take `model_mutex_` (via an accessor) around each encoder and decoder call, or route `/chat/batch` through the same engine (fixing TD-234 at the same time).

Location in code: `src/IntegratedInferenceEngine.hpp` (constructor and worker calls); tagged `TODO: See TD-419`.

Files to Modify:

- `src/IntegratedInferenceEngine.hpp`
- `src/EncoderDecoderModel.hpp`
- `src/ChatbotAPI.cpp`

---

### TD-420: `IntegratedInferenceEngine` Claims, Dead Config and Fictional Stats

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / IntegratedInferenceEngine | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `IntegratedInferenceEngine`, by inspection:
- The header promises "all Priority 1–5 optimizations … Target: 50–100× improvement", and
  `PARALLEL_OPTIMIZATIONS_QUICK_REFERENCE.md` lists it as "Production Ready, 5.5×+", but nothing is batched
  through the model: one `encode_with_mask()` per request, one greedy decode per request (full recompute
  each step, no KV cache). The only overlap is decoding one request while encoding the next (TD-211's gap).
- `enable_pipeline`, `parallel_attention`, `max_tokens_per_batch` and `enable_stats` are never read;
  `use_openmp` only gates a **process-wide** `omp_set_num_threads()`.
- `avg_openmp_speedup`, `avg_attention_speedup`, `cumulative_speedup`, `avg_queue_depth` and
  `requests_dropped` are never set; `pipeline_efficiency` is a formula, not a measurement; `avg_batch_size`
  divides completed requests by batches formed.
- `submit()` blocks when `max_queue_size` is reached; `prepare_encoder_inputs()` is unused.

Action Items:

- [ ] Correct the header and quick-reference claims (or retire the engine in favour of the v2 real-batching plan); remove the dead fields and stats or implement them; don't set OpenMP threads process-wide from a constructor.

Location in code: `src/IntegratedInferenceEngine.hpp` (header banner, config, stats, constructor); tagged `TODO: See TD-420`.

Files to Modify:

- `src/IntegratedInferenceEngine.hpp`
- `docs/development/guides/quick-reference/PARALLEL_OPTIMIZATIONS_QUICK_REFERENCE.md`

---

### TD-421: `KVCache::append()` Has No Shape Checks

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Inference / KVCache | October 13, 2026 | 1-2 hours |

Description:
Found while tracing `KVCache` and verified under AddressSanitizer. `KVCache::append()` never compares
`new_keys` with `new_values`, or the new columns with the cached ones:
- Appending wider rows (`Matrix(1,8)` onto a cache of width 4) writes 8 columns into 4-column rows:
  **heap-buffer-overflow WRITE**.
- Appending fewer value rows than key rows (`append(Matrix(3,4), Matrix(1,4))`) indexes `new_values` by
  `new_keys.rows`: **heap-buffer-overflow READ**.
- An inconsistent first append (`keys 3×4`, `values 1×8`) is accepted silently.

Today's callers always pass matching shapes, so this is latent, but the failure mode is memory
corruption. `DecoderKVCache::get_self_attention_cache()`/`get_cross_attention_cache()` don't
bounds-check the layer index, and `current_length()` reads only layer 0.

Action Items:

- [ ] Validate shapes in `append()` (throw `invalid_argument`), bounds-check layer indices, and add a direct `KVCache` test file.

Location in code: `src/KVCache.hpp` (`append()`, `get_*_cache()`); tagged `TODO: See TD-421`.

Files to Modify:

- `src/KVCache.hpp`
- `tests/` (new KVCache test)

---

### TD-422: `KVCache` Copies the Whole Cache on Every Append

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Inference / KVCache | October 13, 2026 | 2-4 hours |

Description:
Found while tracing `KVCache` and measured. `append()` allocates new `[current_length + new_rows ×
d]` matrices and copies the old contents on every call. Self-attention caches get one append per decode
step per layer, so a generation of *n* tokens does O(n²·d) copying per layer before any attention work.
Measured for one cache at d_model 512, one row per step: 128 steps 15.6 ms, 256 steps 53.8 ms, 512 steps
255 ms (quadratic), which is about 6 s of pure copying for a 24-layer decoder generating 512 tokens on the
CPU serving path. `GPUKVCache` already pre-allocates to `max_seq_length`; this one doesn't. (The old
`kvcache.md` claimed O(d²) per cached step.)

Action Items:

- [ ] Pre-allocate `keys`/`values` to `max_seq_length` (or grow geometrically) and track the populated length; give `MultiHeadAttention`/`CrossAttention` a view of the populated rows instead of a full copy.
- [ ] Do this with TD-388 and TD-398, which have the same per-step copy pattern.

Location in code: `src/KVCache.hpp` (`append()`); tagged `TODO: See TD-422`.

Files to Modify:

- `src/KVCache.hpp`
- `src/MultiHeadAttention.cpp`
- `src/CrossAttention.cpp`

---

### TD-423: A Truncated `.lm_head` File Loads as All-`NaN` Logits

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Models / LanguageModelHead | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `LanguageModelHead` and verified. `load()` checks the two dimension ints, then
pre-fills each weight and bias value with `NaN` and reads without checking the stream. A `.lm_head`
file cut in half **loaded without error**: 751 of 1,600 weights were `NaN`, and all 100 logits of a
forward pass were `NaN`. Every checkpoint written by `EncoderDecoderModel::save_model()` has this
sidecar and `load_model()` reads it, so a checkpoint damaged by a full disk or a crash during save
produces a server that generates garbage instead of refusing to load. `save()` doesn't check its writes.
Same defect as TD-298 (`CrossAttention::load()`).

Action Items:

- [ ] Check the file size (or the stream after each read) and throw on short files, leaving the head unchanged; check `save()`'s stream.
- [ ] Do this with the other legacy loaders (TD-298, TD-345, TD-357, TD-367, TD-369).

Location in code: `src/LanguageModelHead.cpp` (`load()`, `save()`); tagged `TODO: See TD-423`.

Files to Modify:

- `src/LanguageModelHead.cpp`
- `tests/languagemodelhead_test.cpp`

---

### TD-424: Uncached Generation Projects Every Position Through the LM Head

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Models / LanguageModelHead | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `LanguageModelHead` and measured. The uncached generation callbacks run
`lm_head->forward(decoder_out)` over the **whole** decoder output every step: `EncoderDecoderModel`'s
beam `beam_model_fn` (two sites, commented "last position of output"), and `ChatbotAPI`'s CPU
`model_fn`/`/chat/batch`, which call `EncoderDecoderModel::forward()`. `TextGenerator` uses only the last
row. At d_model 512, vocab 8,000 and 100 positions, the projection takes 147 ms for all positions against
4.9 ms for the last one, about 30× the LM-head work per step (and per beam hypothesis). The KV-cached
path and the pipeline/integrated engines already project only what they need.

Action Items:

- [ ] In generation callbacks, project only the last decoder row (keep `forward()`'s full projection for training/loss), and fix the misleading comment.

Location in code: `src/EncoderDecoderModel.cpp` (both beam `beam_model_fn`s, `forward()`); tagged `TODO: See TD-424`.

Files to Modify:

- `src/EncoderDecoderModel.cpp`
- `src/ChatbotAPI.cpp`

---

### TD-425: `LanguageModelHead` Hygiene

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Models / LanguageModelHead | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `LanguageModelHead`, by inspection: the constructor prints four lines to
`std::cout` (library code) and computes its parameter count in `int`; `forward()` doesn't check
`input.cols`; `set_W_output()`/`set_bias()` don't check shapes or refresh the GPU mirror; `gpu_backward()`
dereferences `gpu_` without a check; `get_probabilities()` has no production caller.

Action Items:

- [ ] Switch to `adai::Logger`; add shape checks; invalidate the GPU mirror in the setters; guard `gpu_backward()`; remove or test `get_probabilities()`.

Location in code: `src/LanguageModelHead.{hpp,cpp}`; tagged `TODO: See TD-425`.

Files to Modify:

- `src/LanguageModelHead.{hpp,cpp}`

---

### TD-426: `LayerNorm` Silent Setters, Missing Shape Checks and Unchecked Loads

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Layers / LayerNorm | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `LayerNorm`; the first two items verified:
- `set_gamma()`/`set_beta()` print to `std::cerr` and **return without changing anything** on a shape
  mismatch: `set_gamma(1×5)` on a dim-4 norm left γ at 1 with no exception. `ModelSerializer` validates
  sizes first (`require()`), so it's latent there, but callers can't detect a failed set. Neither setter
  refreshes the GPU mirror.
- `forward()` doesn't check `input.cols == dim`, and `backward()` doesn't check its row count against the
  cached forward (or that a forward happened). Both surface as `std::out_of_range("Matrix index out of
  bounds")` from `Matrix::operator()`, not a clear dimension error.
- `load_weights()` doesn't check its reads (a truncated file keeps the old values for the rest), reads the
  saved ε and discards it, and `save_weights()` doesn't check its writes; both print to `std::cout`, as
  does `print_config()` (no production caller).
- `gpu_backward()` dereferences `gpu_` without a check.

Action Items:

- [ ] Throw `invalid_argument` from the setters and on shape mismatches; check reads and writes (with the other legacy loaders, Tier 55); switch to `adai::Logger`; guard `gpu_backward()`; refresh the GPU mirror in the setters.

Location in code: `src/LayerNorm.cpp` (`forward()`, `set_gamma()`/`set_beta()`, `load_weights()`, `gpu_backward()`); tagged `TODO: See TD-426`.

Files to Modify:

- `src/LayerNorm.{hpp,cpp}`

---

### TD-427: LeJEPA `Predictor` Is Never Saved, So Each Pass Restarts It

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | World Model / LeJEPAEncoder | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `LeJEPAEncoder` and verified. `save()` writes `config.bin`, the token embedding,
the encoder blocks and the final norm, but not the `Predictor` (which has no save/load of its own).
`incremental_trainer --objective=lejepa` resumes the encoder from `<SESSION_DIR>/world_model` every pass
(TD-184) but constructs a fresh, randomly initialized predictor. After 60 epochs, the trained instance's
predictor loss was 0.0017; the same weights reloaded through `save()`/`load()` gave 2.053. Each pass
re-learns the predictor from scratch, and until it does, the random predictor's gradients flow into the
resumed encoder through the context view, disturbing the learned representation. The dashboard's
predictor-loss curve (stable `lejepa-<model>` session) restarts from about 2.0 each pass for this reason.

Action Items:

- [ ] Add `Predictor::save()/load()` (it's a `FeedForward`) and include it in `LeJEPAEncoder::save()/load()`, tolerating its absence in older checkpoints with a warning.
- [ ] Test: predictor loss is unchanged across save/load.

Location in code: `src/LeJEPAEncoder.cpp` (`save()`, `load()`); tagged `TODO: See TD-427`.

Files to Modify:

- `src/LeJEPAEncoder.cpp`
- `src/Predictor.{hpp,cpp}`
- `tests/lejepaencoder_test.cpp`

---

### TD-428: LeJEPA Two-Update Workaround Is Obsolete After TD-195

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | World Model / LeJEPAEncoder | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `LeJEPAEncoder`, by inspection. `train_step()` applies the target-view gradient
and the context-view gradient as two separate `backward()` + `update_weights()` pairs, re-running the
context forward pass in between. The in-code TD-189 comment justifies this by saying
`LayerNorm`/`FeedForward`/`MultiHeadAttention::backward()` overwrite their gradient members, but TD-195
made all three accumulate. The workaround now only costs: Adam's step counter advances twice per sample,
every registered parameter (including the predictor, whose gradient is zero during step 1) is moved by
momentum twice, and the objective is optimized as two sequential half-steps.

Action Items:

- [ ] Accumulate both views' gradients (target backward, then a context re-forward and backward) and call `update_weights()` once; update the stale comment; compare a short pretraining run before and after.

Location in code: `src/LeJEPAEncoder.cpp` (`train_step()`); tagged `TODO: See TD-428`.

Files to Modify:

- `src/LeJEPAEncoder.cpp`

---

### TD-429: `LeJEPAEncoder` Exception Escape, Console Output and Unchecked Loads

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | World Model / LeJEPAEncoder | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `LeJEPAEncoder`, by inspection: `TokenizerEncodingError` derives from
`runtime_error`, so it escapes `run_lejepa_training_pass()`'s `catch (const std::invalid_argument&)` and
ends the whole pass; the constructor, save/load and tokenizer methods print to `std::cout`; `load()`
doesn't compare `max_seq_length` and inherits the sub-loaders' unchecked reads (TD-357/TD-426 class).

Action Items:

- [ ] Catch tokenizer exceptions per sample in the LeJEPA loop; switch to `adai::Logger`; validate `max_seq_length` and check reads on load.

Location in code: `src/LeJEPAEncoder.cpp` (`load()`), `src/IncrementalTrainingTool.cpp` (LeJEPA sample loop); tagged `TODO: See TD-429`.

Files to Modify:

- `src/LeJEPAEncoder.cpp`
- `src/IncrementalTrainingTool.cpp`

---

### TD-430: `Logger` Lazy Initialization Races and Can Throw From a Log Call

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Core / Logger | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `Logger` and verified. `Logger::get()` lazily calls `init(INFO)` when `logger_` is
null, without synchronization, and `logger_` is a plain static `shared_ptr`. When several threads log
before `init()`, more than one calls `init()`, and `spdlog::stderr_color_mt()` throws
`spdlog_ex("logger with name 'adai' already exists")` from inside the logging call: 3 of 8 threads threw
in a fresh process (every one of 200 forked trials saw a throw). The concurrent `shared_ptr` reads and
writes are a data race, and calling `init()` again while other threads log is unsafe the same way. The
binaries call `init()` early in `main()`, so tests and code that logs from threads during startup are
what's exposed. Also: `init(level, name)` drops only the **new** name (a logger registered under a
previous name lingers), `FileConfig::compress` only changes startup log lines, and `Level::ERROR`
collides with the Windows `<wingdi.h>` `ERROR` macro.

Action Items:

- [ ] Guard initialization with `std::call_once` (or a mutex) and make `logger_` an atomic `shared_ptr`/`spdlog::default_logger()`; never throw from a log call; drop the previous logger's own name on re-init; rename `Level::ERROR` (or `#undef` guard).

Location in code: `src/Logger.{hpp,cpp}` (`get()`, `init()`, `Level`); tagged `TODO: See TD-430`.

Files to Modify:

- `src/Logger.{hpp,cpp}`
- `tests/logger_test.cpp`

---

### TD-431: Trainer Processes Share One Rotating Log File and Ignore `LOG_LEVEL`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Core / Logger | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `Logger` and verified. When `LOG_FILE_PATH` is empty, `trainer_service`
(`TrainerServiceMain.cpp`) and `incremental_trainer`'s `train`/`retrain`/`resume` fall back to
`chatbot_server.log` in the working directory, so `trainer_service` and every child pass it launches write
the **same file through separate `rotating_file_sink_mt`s**. spdlog's rotating sink isn't multi-process
safe: each process rotates on its own size count and renames files out from under the other. Two processes
writing 30,000 lines each to a 1 MB × 3 log retained 34,162 lines (2.80 MB), against 47,031 lines (3.85 MB)
for one process writing the same 60,000 lines, about 27% less history than configured (no corrupted
lines). This contradicts LOG_FILE_STANDARDS.md (empty path should mean console-only), and the name
suggests the chatbot server's log. Separately, both trainer binaries always initialize at `INFO`; only
`chatbot_api_server` applies `LOG_LEVEL`.

Action Items:

- [ ] Give each process its own file (for example `<LOG_FILE_PATH>.child` or `incremental_trainer.log`), or log the children to stderr and let `trainer_service` capture it; use console-only when `LOG_FILE_PATH` is empty.
- [ ] Apply `LOG_LEVEL` in both trainer binaries.

Location in code: `src/TrainerServiceMain.cpp` (log path), `src/IncrementalTrainingTool.cpp` (`run_training_pipeline()`'s `Logger::init`); tagged `TODO: See TD-431`.

Files to Modify:

- `src/TrainerServiceMain.cpp`
- `src/IncrementalTrainingTool.cpp`

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

### TD-434: `Matrix::operator*` Runs Short Products Single-Threaded and Portable Builds Have No SIMD

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Core / Matrix | October 14, 2026 | 2-4 hours |

Description:
Found while tracing `Matrix` and verified by timing (8-core dev machine, AVX2 build):
- **Threading gate on `rows` only.** The AVX2 and NEON paths use `#pragma omp parallel for
  schedule(dynamic, 16) if (rows > 64)` (and the OpenMP fallback `rows > 64 && other.cols > 64`). Training
  samples and decode steps usually have ≤ 64 rows, so their weight-matrix products run on one thread however
  large the inner and output dimensions are. A work-based gate (`rows·cols·other.cols > 2¹⁸`, static
  schedule) measured 64×512×512: ~3 → ~0.8 ms; 64×512×2048: ~16 → ~7.5 ms; 8×512×8000 (LM head, decode):
  ~12.5 → ~2.2 ms; already-parallel 300×512×2048 unchanged or slightly faster. Callers already inside an
  OpenMP region (`MultiHeadAttention`'s per-head loop) get a serialized inner region under the default
  `OMP_MAX_ACTIVE_LEVELS`, so this doesn't oversubscribe.
- **No SIMD in `portable` builds.** `ADAI_SIMD_AVX2` comes from the compiler's `__AVX2__`
  (`MatrixSIMD.hpp`), which the `portable` preset (`-march=x86-64`) deliberately doesn't set, so
  `operator*` falls to the strided OpenMP ijk loop: 7–10× slower (64×512×2048: 162 ms vs 16 ms;
  8×512×8000: 93 ms vs 12.7 ms). CLAUDE.md recommends `portable` for binaries deployed to other machines;
  that's harmless for the daemons but costly for a CPU `chatbot_api_server` or trainer.

Action Items:

- [ ] Replace the `rows > 64` gates in `operator*` with a work-based threshold (and a static schedule); benchmark before/after on a CPU training pass.
- [ ] Add runtime ISA dispatch for the multiply (an AVX2/FMA kernel compiled with `__attribute__((target("avx2,fma")))`, selected once via the existing `adai::simd` CPUID helpers), so `portable` binaries use it on capable CPUs; extend the same to the element-wise ops if worthwhile.
- [ ] Add a test that runs the multiply with OpenMP threads on a ≤ 64-row shape and compares against the scalar reference.

Location in code: `src/Matrix.cpp` (`operator*`'s OpenMP pragmas); `src/MatrixSIMD.hpp` (compile-time detection); tagged `TODO: See TD-434`.

Files to Modify:

- `src/Matrix.cpp`
- `src/MatrixSIMD.hpp`
- `tests/matrix_simd_test.cpp`

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
- [ ] Make `randomize()` take a generator (with TD-368) or guard the static one.
- [ ] Reject negative sizes in the constructor; decide whether to keep the unused `*_gpu()` wrappers.

Location in code: `src/Matrix.cpp` (`operator()`, `sum()`, `randomize()`, `Matrix(int, int)`); `src/Matrix.hpp` (public members); tagged `TODO: See TD-435`.

Files to Modify:

- `src/Matrix.hpp`
- `src/Matrix.cpp`

---

### TD-436: SYCL (`icpx`) Builds Get No `-march` or SIMD Flags

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Build / SIMD | October 14, 2026 | 1-2 hours |

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

- [ ] Add `IntelLLVM` to both compiler-ID checks (or use `-march=native` for it), keeping the `-fno-finite-math-only` guard; confirm with the `SIMD:` configure message and `compile_commands.json`.
- [ ] Measure a training pass's wall time on ai-machine before/after.
- [ ] Mind that `-march=native` ties the binary to the build host's CPU (TD-437's startup check); build on ai-machine or pick an explicit `-march`.

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

---

### TD-438: `metrics_api_server` Accepts Invalid Numbers and Exits 0 on Bad Arguments

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Metrics / Server CLI | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MetricsApiServerArgs` and verified (scratch program against the real parser, and the
real `metrics_api_server`):
- **Unchecked numbers.** `parse_metrics_api_server_args()` uses `std::atoi`/`std::strtoul` with no range
  checks. `--port abc` → 0, and **the real server listened on a random ephemeral port** (39915) while
  printing `Server starting on http://0.0.0.0:0`; `--port 99999` printed `:99999` and **listened on 34463**
  (mod 65536). `--max-live-sessions -1` wraps to 18,446,744,073,709,551,615 (no cap);
  `--sweep-interval-seconds 0` disables `MetricsSessionRegistry`'s eviction sweep; `--persist-seconds 1e3`
  → 1 and `8081x` → 8081 (truncated); negative ports, pool sizes and intervals are accepted.
- **Saved admin overrides.** `apply_metrics_admin_overrides()` re-applies `daemon_config.db` values at
  startup with the same gaps (`stoull("-1")` wraps; `"50abc"` → 50; negative TTLs applied), so a bad
  `PUT /admin/config` (TD-306) persists across restarts.
- **Exit code.** `main()` prints usage and **exits 0** on any parse error (verified with `--bogus`), so a
  mistyped flag in a unit's `ExecStart` looks like a clean stop and `Restart=on-failure` doesn't retry. A
  value-taking flag with no value is reported as "Unknown option".

Action Items:

- [ ] Parse numbers with a checked helper (whole string, in range: port 1–65535, counts and intervals ≥ 1 or a documented 0 meaning) shared with TD-240; reject with a clear message.
- [ ] Validate the saved admin overrides the same way (skip and warn on out-of-range values).
- [ ] Exit non-zero on parse errors; report "missing value for --flag".
- [ ] Add tests for non-numeric, partial, negative and out-of-range values.

Location in code: `src/MetricsApiServerArgs.cpp` (`parse_metrics_api_server_args()`, `apply_metrics_admin_overrides()`); `src/TrainingMetricsAPIServer.cpp` (`main()`'s error exit); tagged `TODO: See TD-438`.

Files to Modify:

- `src/MetricsApiServerArgs.cpp`
- `src/TrainingMetricsAPIServer.cpp`
- `tests/metrics_api_server_args_test.cpp`

---

### TD-439: `metrics_api_server` Backend Name and Admin-Store Location Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Metrics / Server CLI | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MetricsApiServerArgs`:
- **Unknown backend accepted (verified in the parser).** `--storage-backend postgress` (or `postgres` in a
  build without `ADAI_ENABLE_POSTGRES`) is accepted; `MetricsDatabaseFactory::create()` logs a warning and
  returns no database, so the server runs file-only while startup prints `Storage backend: postgress`.
- **Admin store location (by inspection).** `main()` puts `daemon_config.db` in the directory of the
  **file-configured** `db_path`, computed before CLI flags. With `scripts/install_metrics_service.sh`'s
  defaults that's the same directory as `--db-path`, but with a custom `--metrics-dir` it's
  `<install>/training_sessions`, which the unit's `ProtectSystem=strict` + `ReadWritePaths=<metrics dir>`
  makes read-only, so the store fails to open and admin settings can't persist.
- **Precedence (by inspection).** CLI flags beat saved admin settings, so an admin-mutable flag added to
  `ExecStart` would silently undo every `PUT /admin/config` on restart. Latent with the installer's flags.
- **Hygiene.** The header names the test `MetricsApiServerArgs_test.cpp` (it's
  `metrics_api_server_args_test.cpp`).

Action Items:

- [ ] Validate `--storage-backend` against the compiled-in backends and fail at startup on an unknown one.
- [ ] Locate the admin store from the final `db_path` (or a dedicated `--admin-dir`), and warn when a CLI flag overrides a saved admin value.
- [ ] Fix the header comment.

Location in code: `src/MetricsApiServerArgs.{hpp,cpp}`; `src/TrainingMetricsAPIServer.cpp` (`admin_config_db_dir`); tagged `TODO: See TD-439`.

Files to Modify:

- `src/MetricsApiServerArgs.hpp`
- `src/MetricsApiServerArgs.cpp`
- `src/TrainingMetricsAPIServer.cpp`

---

### TD-440: Metrics Storage Has No Retention and `max_records_on_disk` Is Never Applied

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Metrics / Database | October 14, 2026 | 2-4 hours |

Description:
Found while tracing `MetricsDatabase` and verified by search:
- **No retention.** `IMetricsDatabase` has no delete/prune operation, and neither `SQLiteMetricsDatabase`
  nor `PostgresMetricsDatabase` deletes rows except while renaming them in `archive_session()`. The JSONL
  file (`TrainingMetricsService::persist_metrics()`, `std::ios::app`) is never rotated or trimmed.
- **A dead setting.** `MetricsServiceConfig::max_records_on_disk` (default 100,000; `--max-disk-records`,
  `config.metrics.conf`, admin-mutable via `PUT /admin/config`, reported by `GET /admin/config`) is read by
  no persistence code; only `max_records_in_memory` is enforced (on the in-memory ring buffer).
  `compress_old_records` is likewise never read. TRAINING_METRICS_API.md documents "Max records on disk" as
  a real limit.
- **Write-only tables.** `gradient_variance_history` (one row per optimizer step per session, unthrottled)
  and `generation_quality` have insert methods but no query method; no C++ code or repository tool reads
  them (`tools/abnormal_review/` reads only `abnormal_samples`). The dashboards use the in-memory snapshot.

So the metrics host's DB and JSONL grow for as long as `metrics_api_server` runs, fastest in the table
nothing reads.

Action Items:

- [ ] Owner decision: retention policy (per-session row cap, age, or both) and whether `gradient_variance_history`/`generation_quality` get readers (an API endpoint) or are dropped/throttled.
- [ ] Add a prune operation to `IMetricsDatabase` (both backends) and apply `max_records_on_disk` (per session) on persist or in the registry sweep; trim or rotate the JSONL to match.
- [ ] Remove `compress_old_records` or implement it; correct TRAINING_METRICS_API.md.

Location in code: `src/MetricsDatabase.hpp` (`IMetricsDatabase`); `src/TrainingMetricsService.hpp` (`max_records_on_disk`, `compress_old_records`); `src/TrainingMetricsService.cpp` (JSONL append); tagged `TODO: See TD-440`.

Files to Modify:

- `src/MetricsDatabase.hpp`
- `src/SQLiteMetricsDatabase.{hpp,cpp}`
- `src/PostgresMetricsDatabase.{hpp,cpp}`
- `src/TrainingMetricsService.{hpp,cpp}`
- `docs/development/TRAINING_METRICS_API.md`

---

### TD-441: `IMetricsDatabase` Contract Gaps: Step Counter, `limit` Order, Unbounded Reads

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Metrics / Database | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MetricsDatabase`, by inspection:
- **Step counter restarts.** `insert_gradient_variance_sample()`'s `step` is `TrainingMetricsService`'s
  `gradient_variance_step_counter_`, which starts at 0 whenever a session's service is recreated (server
  restart, restored session), giving duplicate `(session_key, step)` pairs; the interface comment calls it
  "monotonic".
- **`limit` keeps the oldest rows.** Both backends' `query_history()` sort `recorded_at ASC` then apply
  `LIMIT`, so `GET .../metrics/db-history?limit=500` returns a run's first 500 records, not the latest.
- **Unbounded reads.** `TrainingMetricsAPI`'s compare and export call `query_history()` with no limit,
  materializing whole histories per request; the interface has no paging.
- **Undocumented contract.** Throwing on error, thread safety, ascending order, `limit = 0` = all, and the
  `FLT_MAX` "no best loss" sentinel are only discoverable from the implementations;
  `MetricsDatabaseFactory::create()` is declared here but defined in `SQLiteMetricsDatabase.cpp`.

Action Items:

- [ ] Seed the step counter from the DB (`MAX(step)`) on restore, or drop the "monotonic" claim.
- [ ] Add an order/"latest N" option to `query_history()` and use it for `db-history`; document `limit`.
- [ ] Cap or page compare/export reads; document the interface contract; move the factory to its own file.

Location in code: `src/MetricsDatabase.hpp`; `src/TrainingMetricsService.cpp` (step counter); `src/SQLiteMetricsDatabase.cpp`/`src/PostgresMetricsDatabase.cpp` (`query_history()`); tagged `TODO: See TD-441`.

Files to Modify:

- `src/MetricsDatabase.hpp`
- `src/TrainingMetricsService.cpp`
- `src/SQLiteMetricsDatabase.cpp`
- `src/PostgresMetricsDatabase.cpp`
- `src/TrainingMetricsAPI.cpp`

---

### TD-442: `MetricsPushClient::end_session()` Drain Can Delay a Pass's Final Save by an Hour

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Metrics / Push Client | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MetricsPushClient` and verified. `end_session()` enqueues `/end`, sets the stop flag and
joins the push thread, which first sends **every** queued event. Each failing event costs up to 3 attempts plus
1.2 s of back-off (plus `timeout_ms` per attempt when the connection times out rather than being refused). With
the server down (connection refused), 10 queued epoch events plus `/end` took **13.2 s**. If the server goes
away mid-pass the queue fills to 1,024 step events, so the drain takes about 20 minutes (refused) or about 70
minutes (timeouts, default 1 s). The destructor drains the same way.

`IncrementalTrainer` calls `push_client_->end_session()` **before** saving the final checkpoint and calling
`mark_trained()`, so a metrics outage near the end of a chatbot pass delays its results by that long, and a stop
in that window (`systemctl stop`, `trainer_service`'s 10 s SIGKILL escalation, a GPU-driver crash) loses the
final save and leaves the files claimed. (`run_lejepa_training_pass()` saves first and is only delayed.)

Action Items:

- [ ] Bound the drain: a deadline for `end_session()`, and once stopping, drop `Sample` events and skip retries/back-off.
- [ ] In `IncrementalTrainer`, save the final checkpoint and `mark_trained()` before `end_session()`.
- [ ] Add a test: `end_session()` with the server down returns within the deadline.

Location in code: `src/MetricsPushClient.cpp` (`end_session()`, `push_loop()`); `src/IncrementalTrainer.cpp` (`end_session()` before the final save); tagged `TODO: See TD-442`.

Files to Modify:

- `src/MetricsPushClient.{hpp,cpp}`
- `src/IncrementalTrainer.cpp`
- `tests/metrics_push_client_test.cpp`

---

### TD-443: Metrics Push Payloads Send `nan`/`inf`, Round Small Values to 0 and Omit Adaptive Clipping

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Metrics / Push Client | October 14, 2026 | 2-4 hours |

Description:
Found while tracing `MetricsPushClient` and verified against a real `metrics_api_server`:
- **Non-finite values.** Payloads are built with `operator<<`, so NaN/infinity are sent as `nan`/`inf` (not
  JSON). The server's hand-written parser (`std::stof`) accepts them and its output re-emits them: after pushing
  an epoch with a NaN loss and an infinite validation loss, `GET /api/sessions/{key}/epochs` returned
  `"epoch_losses":[nan,2.5],"epoch_validation_losses":[inf,2.6]`, which Python's `json` rejects. The epoch view
  breaks exactly when a run diverges. `end_epoch()`'s derived perplexity `exp(loss)` is `inf` above loss ≈ 88.
- **Rounding.** `end_epoch()` uses `std::fixed` (6 decimals): a learning rate of 1e-7 was stored as **0**
  (`"epoch_learning_rates":[0,0]`), as are epoch `gradient_variance`/`weight_update_ratio` below 5e-7 (typical
  update ratios are 1e-4–1e-6). `update_generation_quality_metrics()` also uses `std::fixed` (harmless there).
- **Adaptive clipping never sent.** `update_adaptive_clip_metrics()`/`update_adaptive_clip_epoch()` store into
  `buf_adaptive_clip_*`, which `end_epoch()` never writes, and no server endpoint accepts them; the probe
  session's `epoch_adaptive_clip_thresholds` was `[]`, as it is for every pushed session.

Action Items:

- [ ] Serialize floats with a helper that writes non-finite values as `null` (or a documented sentinel) and uses enough significant digits (`std::defaultfloat` with `max_digits10`, or `%.9g`).
- [ ] Make the server reject or normalize non-finite input and never emit `nan`/`inf`.
- [ ] Add the adaptive-clip fields to the `end_epoch` payload and its server handler.
- [ ] Tests for NaN/inf/tiny values and adaptive-clip fields, end to end.

Location in code: `src/MetricsPushClient.cpp` (`end_epoch()`, `update_adaptive_clip_*()`); `src/TrainingMetricsAPI.cpp` (`/epoch/end` parsing); tagged `TODO: See TD-443`.

Files to Modify:

- `src/MetricsPushClient.cpp`
- `src/TrainingMetricsAPI.cpp`
- `tests/metrics_push_client_test.cpp`

---

### TD-444: `MetricsPushClient` Transport Gaps: No HTTPS, a Connection per Event, Silent Failures

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Metrics / Push Client | October 14, 2026 | 2-4 hours |

Description:
Found while tracing `MetricsPushClient`, by inspection:
- **No HTTPS.** `ParsedUrl` recognizes `https://` and switches to port 443, but `attempt_post()` always uses a
  plain `httplib::Client` (no `CPPHTTPLIB_OPENSSL_SUPPORT`), so it sends plain HTTP to a TLS port and
  `start_session()` returns 0; the run trains without metrics. There's no way to add headers (Cloudflare Access
  tokens), so metrics can't go through the tunnel. Today's `config.trainer.conf` uses a LAN `http://` URL.
- **A connection per event.** A new `httplib::Client` (TCP connection) per attempt, including per optimizer step
  for `/metrics/sample` and `/metrics/advanced`.
- **Silent failures.** `push_loop()` ignores every status except 404 (e.g. a 400 for a malformed body, a 503
  "server full"); failures are logged at debug level only.
- **Race and naming.** `reconnect_body_` is written by `start_session()` while the push thread runs (read only after
  a 404); the heartbeat can fire before `start_session()`. `detect_hostname_fragment()` cuts hostnames to 8
  characters, so `ai-machine`/`ai-machine2` share keys (409 retry handles it, with confusing keys).

Action Items:

- [ ] Support `https://` (`httplib::SSLClient` with OpenSSL) and configurable extra headers, or reject `https://` with a clear error.
- [ ] Reuse one keep-alive client in the push thread.
- [ ] Log non-2xx results at warn (rate-limited); guard `reconnect_body_`; consider a longer host fragment or a hash suffix.

Location in code: `src/MetricsPushClient.cpp` (`ParsedUrl`, `attempt_post()`, `push_loop()`); `src/MetricsPushClient.hpp` (`detect_hostname_fragment()`, `reconnect_body_`); tagged `TODO: See TD-444`.

Files to Modify:

- `src/MetricsPushClient.{hpp,cpp}`
- `src/CMakeLists.txt`

---

### TD-445: Finished Metrics Sessions Hold `max_live_sessions` Slots for an Hour

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Metrics / Session Registry | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MetricsSessionRegistry` and verified with the real registry. `start_session_or_conflict()`
(and `create_or_get_session()`) refuse a new key when `sessions_.size() >= max_live_sessions_`, and the map holds
**every** session, including ones that completed normally or went stale. A finished session leaves only when
idle for `completed_ttl_seconds` (3,600 in `config.metrics.conf`) or when its own key is reused; nothing evicts
one to make room. With a cap of 2, two started-and-ended sessions left the registry full and a third key got no
service (the API answers `503 metrics_server_full`).

Every chatbot pass registers a new key (`<session_id>-<host>`; the session number increments per pass), and
`trainer_service` launches the next pass immediately after one that did work; crashed passes' stale sessions count
too. So with the shipped cap of 16, more than 16 passes within an hour fills the server, and later passes log
"training will proceed WITHOUT metrics reporting" until the oldest finished session ages out.

Action Items:

- [ ] Before refusing, evict (archive) the least recently updated finished or stale session; count only live sessions against the cap.
- [ ] Add a test: cap N, N finished sessions, a new key still starts.
- [ ] Update TRAINING_METRICS_API.md's capacity note.

Location in code: `src/MetricsSessionRegistry.hpp` (`start_session_or_conflict()`, `create_or_get_session()`); tagged `TODO: See TD-445`.

Files to Modify:

- `src/MetricsSessionRegistry.hpp`
- `tests/metrics_session_registry_test.cpp`
- `docs/development/TRAINING_METRICS_API.md`

---

### TD-446: `MetricsSessionRegistry` Sweep-Interval Changes, Miss Logging and Lock-Held I/O

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Metrics / Session Registry | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MetricsSessionRegistry`:
- **Runtime sweep interval (verified).** `set_sweep_interval_seconds()` (reachable via `PUT /admin/config`, no range
  check, TD-306) to ≤ 0 while the thread runs makes `sweep_loop()`'s `wait_for(0)` return immediately, so it spins:
  0.51 CPU-seconds in 2 s with an empty registry, taking the exclusive registry lock each iteration. Going from 0
  (thread never started by the constructor) to a positive value does nothing; a finished session with a 1 s TTL
  was still present after 3 s while `GET /admin/config` reported the new interval.
- **Miss logging (by inspection).** `get_session()` logs a warning plus one line per live key for every unknown key.
  `handle_prometheus_aggregate()` calls it for each session in `list_sessions()`, which includes DB-only completed
  sessions, so each aggregate request logs `(DB sessions) × (1 + live)` lines; any 404 for a completed key does too.
- **I/O under the lock (by inspection).** Eviction archiving (a DB transaction and file renames) runs under the
  exclusive registry lock on every start and sweep; `list_sessions()` queries the DB under the shared lock.
- **Late writes (by inspection).** An evicted service still referenced by an in-flight request can persist after its
  files were renamed, recreating them under the original key, which a new session with that key then shares.

Action Items:

- [ ] Reject or clamp intervals ≤ 0 in the setter (or treat as "disabled" without spinning); start the thread when moving from 0 to positive.
- [ ] Log misses at debug (or once), without listing every key.
- [ ] Do archiving I/O and DB listing outside the registry lock; mark evicted services so late persists are dropped.

Location in code: `src/MetricsSessionRegistry.hpp` (`sweep_loop()`, `set_sweep_interval_seconds()`, constructor, `get_session()`, `archive_and_remove_locked()`, `list_sessions()`); tagged `TODO: See TD-446`.

Files to Modify:

- `src/MetricsSessionRegistry.hpp`
- `tests/metrics_session_registry_test.cpp`

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

### TD-448: `mns_cli` Ignores Unknown Flags, Misreads `set-training` Arguments and Aborts on Bad Numbers

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | MNS / CLI | October 14, 2026 | 2-4 hours |

Description:
Found while tracing `MnsCliCommands` and verified (scratch program against the builders, and the real `mns_cli`):
- **Unknown flags ignored.** Every `build_*_request()` skips arguments it doesn't recognize, and a value-taking flag
  at the end of the line is dropped. `register m chat --dmodel 256` built a request with the config file's
  `d_model` (512) and no error. Architecture is immutable after `register` (CLAUDE.md), so the only fix is delete
  and re-register. `list`, `link-world-model` and `set-candidate` behave the same way.
- **`set-training`'s stray arguments.** Anything other than `--new-run` becomes the metrics session key:
  `set-training m --new_run` built `{"state":"training","new_run":false,"metrics_session_key":"--new_run"}`,
  continuing the current run and linking a bogus metrics session.
- **Unchecked numbers.** `std::stoul`/`std::stoi`/`std::stof` exceptions are caught nowhere (not in the builders,
  not in `MnsCliTool.cpp`'s `main()`): `mns_cli register foo chatbot --d-model abc` printed `terminate called after
  throwing an instance of 'std::invalid_argument'` and aborted with a core dump (exit 134). `--d-model -1` was sent as
  18,446,744,073,709,551,615 (`stoul` wraps); `512x` → 512.

Action Items:

- [ ] Reject unknown flags and flags missing their value with a usage error in every builder.
- [ ] `set-training`: accept at most one positional session key and reject `-`-prefixed unknowns.
- [ ] Parse numbers with a checked helper (whole string, positive, in range), returning `make_error()`; catch any remaining exception in `main()` with exit 2.
- [ ] Tests for misspelled flags, stray `set-training` arguments, malformed/negative numbers.

Location in code: `src/MnsCliCommands.cpp` (every builder's flag loop, `build_set_training_request()`); `src/MnsCliTool.cpp` (`main()`); tagged `TODO: See TD-448`.

Files to Modify:

- `src/MnsCliCommands.cpp`
- `src/MnsCliTool.cpp`
- `tests/mns_cli_commands_test.cpp`

---

### TD-449: `mns_cli` Request-Building Gaps: Layer Counts, Relink Resets, Encoding

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | MNS / CLI | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MnsCliCommands`:
- **Layer counts (verified).** For `--kind encoder`/`decoder`/`world_model`, the own-kind layer field is set and the other
  zeroed only when `--num-layers` is given; without it the request carries the config's `num_encoder_layers` **and**
  `num_decoder_layers` (6/6), which `mns_server` rejects with a layer-count error instead of a usage message.
- **Relink resets (by inspection).** `link-world-model` always sends all eight connection fields with defaults for
  omitted flags, and `handle_link_world_model()` replaces the whole connection, so re-running it to change
  `--inject-every-n-layers` turns hippocampal memory off and resets its tuning.
- **Encoding (verified).** Names and query values aren't URL-encoded: `get a/state` requested `/models/a/state`;
  `list --role "a b&c"` built `/models?role=a b&c`. `json_escape()` escapes only `"`, `\`, `\n`, `\r`, `\t`, so other
  control characters (e.g. `\x01` in a tag) produce invalid JSON.
- **Minor.** `--sigreg-*` silently ignored for kinds other than `world_model`; `update` requires the exact order
  `<name> --run-group <value>`; `parse_url()` accepts `https://` though `mns_cli` connects with plain HTTP.

Action Items:

- [ ] Require `--num-layers` for standalone kinds (or derive it from the config's matching field and zero the other).
- [ ] Make `link-world-model` send only the fields given (server-side merge), or document full replacement in its usage.
- [ ] URL-encode path segments and query values; escape all control characters as `\u00XX`; reject `--sigreg-*` for other kinds.

Location in code: `src/MnsCliCommands.cpp` (`build_register_request()`, `build_link_world_model_request()`, `json_escape()`, path builders); tagged `TODO: See TD-449`.

Files to Modify:

- `src/MnsCliCommands.cpp`
- `tests/mns_cli_commands_test.cpp`

---

### TD-450: `mns_cli register` Takes Architecture Defaults from `config.mns.conf`, Not the Trainer's Config

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | MNS / CLI | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MnsCliTool`, by inspection of the code, config files and guide. `main()` loads
`discover_config_path(--config, "config.mns.conf")`, the `mns_server` daemon's file, and `register` fills every
architecture field not given as a flag from it. The repository's `config.mns.conf` has `D_MODEL=512`, `NUM_HEADS=8`,
`D_FF=2048`, `MAX_SEQ_LENGTH=1024`; `config.trainer.conf`'s fallback block has `D_MODEL=768`, `NUM_HEADS=12`. So
`mns_cli register my-model chatbot` without architecture flags permanently registers 512/8, and since MNS is
authoritative (and architecture immutable), the trainer silently adopts it instead of the 768/12 its own config shows.
The usage text doesn't say where the defaults come from; `docs/operations/guides/mns-cli-guide.md` says they come
from `config.conf` (the pre-split file, now a fallback after `config.mns.conf`) "so registration stays consistent with
what the trainer will use", which the split configuration no longer guarantees.

Action Items:

- [ ] Require explicit architecture flags for `register` (or a `--from-config <file>` that names the trainer's config), and print the architecture being registered before sending.
- [ ] Decide whether `config.mns.conf` should keep an architecture block at all.
- [ ] Update `mns_cli`'s usage text and mns-cli-guide.md.

Location in code: `src/MnsCliTool.cpp` (`main()`'s config loading, `cmd_register()`); tagged `TODO: See TD-450`.

Files to Modify:

- `src/MnsCliTool.cpp`
- `src/MnsCliCommands.cpp`
- `config.mns.conf`
- `docs/operations/guides/mns-cli-guide.md`

---

### TD-451: `mns_cli` Has No HTTPS and Minor Option Handling Gaps

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | MNS / CLI | October 14, 2026 | 2-4 hours |

Description:
Found while tracing `MnsCliTool`:
- **No HTTPS (verified).** `parse_url()` strips `https://` but keeps the default port 8083, and `make_http()` always
  builds a plain `httplib::Client`: `mns_cli --url https://127.0.0.1 health` reported `connection to 127.0.0.1:8083
  failed`. `mns_cli` can't reach MNS through a TLS endpoint (the Cloudflare tunnel) and can't send Access headers.
  Same gap as TD-444 (`MetricsPushClient`).
- **`--help` (verified).** Only recognized as the first word; `mns_cli list --help` sent a `list` request.
- **By inspection.** `--url`/`--config` are taken from anywhere on the line, so they can't be command argument values;
  `ModelNameClient.hpp` is included but unused; the file comment still mentions `config.conf`; the four `http_*()`
  helpers are duplicates.

Action Items:

- [ ] Support `https://` (`httplib::SSLClient`, port 443 by default) and optional extra headers (shared helper with TD-444), or reject `https://` clearly.
- [ ] Recognize `--help` per command; stop global-option parsing at the command word.
- [ ] Remove the unused include, fix the comment, fold the `http_*()` helpers into one.

Location in code: `src/MnsCliTool.cpp` (`make_http()`, `main()`); `src/MnsCliCommands.cpp` (`parse_url()`); tagged `TODO: See TD-451`.

Files to Modify:

- `src/MnsCliTool.cpp`
- `src/MnsCliCommands.cpp`

---

### TD-452: `MnsJsonHelpers` Parsing Gaps and a GUI Register Body Without TD-196 Fields

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | MNS / Manager GUI | October 14, 2026 | 2-4 hours |

Description:
Found while tracing `MnsJsonHelpers` (used only by the optional Qt `mns_manager_gui`):
- **Verified.** `json_value()` doesn't unescape string values (`C:\\models\\v3.bin`, `caf\u00e9 \"x\"` displayed as-is);
  it matches only `"key":` with no space (`"state" : "training"` → empty) and returns a fragment for object values;
  `json_array_objects()` falls back to the first other array when the key is absent (e.g. an error body);
  `json_escape()` escapes only `"`, `\` and `\n`, so tabs, carriage returns and other control characters produce
  invalid JSON; `json_pretty()` renders `{}`/`[]` as three-line blocks.
- **By inspection.** `build_register_model_body()` has no `kind`, `run_group` or `connection`, so the GUI can register
  only legacy bundled `chatbot` records, not encoders, decoders, world models or linked pairs (TD-196), and can't set a
  run group. `ParsedUrl` accepts `https://` with port 8083 (as TD-451).

Action Items:

- [ ] Unescape string values; return `nullopt` for a missing array key; complete `json_escape()` (or adopt TD-453's shared library).
- [ ] Add kind, run group and encoder/decoder/world-model fields to the GUI's Register tab and body.

Location in code: `src/MnsJsonHelpers.hpp` (`json_value()`, `json_array_objects()`, `json_escape()`, `build_register_model_body()`); tagged `TODO: See TD-452`.

Files to Modify:

- `src/MnsJsonHelpers.hpp`
- `src/MnsManagerGUI.{hpp,cpp}`
- `tests/mnsjsonhelpers_test.cpp`

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

Action Items:

- [ ] Move the MNS (server, client, CLI, GUI) and metrics (server, push client) request/response code to `nlohmann/json`, one file at a time behind the existing tests.
- [ ] Delete the per-file escape/scan helpers once their callers are migrated.

Location in code: the helpers listed above; tagged `TODO: See TD-453`.

Files to Modify:

- `src/MnsJsonHelpers.hpp`, `src/MnsCliCommands.cpp`, `src/ModelNameClient.cpp`, `src/ModelNameService.cpp`, `src/MetricsPushClient.cpp`, `src/TrainingMetricsAPI.cpp`

---

### TD-454: MNS Manager GUI's Training → Candidate Workflow Fails Against Server-Allocated Run IDs

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | MNS / Manager GUI | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MnsManagerGUI` and verified by sending the GUI's exact request bodies to a real `mns_server`.
`onSetTraining()` requires a typed run ID and sends `{"state":"training","run_id":"…"}`. Since run IDs became
server-allocated (CLAUDE.md, "MNS/registry-authoritative run and session numbering"), the server ignores a client
`run_id` for this transition, allocates `run-NN`, and reads `new_run` to decide whether to start a fresh run:

```text
Set Training,  run ID "my-run"   → state training, run_id "run-01"
Set Candidate, run ID "my-run"   → 409 "run_id does not match the active run; this trainer's run has been superseded"
Set Training,  run ID "my-run-2" → run_id still "run-01" (the GUI never sends new_run)
```

So the typed run ID is silently discarded, Set Candidate fails with a misleading "superseded" error unless the user copies
the allocated `run-NN` out of the detail panel, and the GUI can't start a new run. `docs/operations/guides/mns-manager-gui-guide.md`'s
walkthrough (enter `run-42`, Set Training, Set Candidate) follows exactly this path, and its `mns_cli` table shows
`set-training <name> <run-id>`, a form `mns_cli` no longer has.

Action Items:

- [ ] Set Training: drop the run-ID requirement, add a "start a new run" checkbox (`new_run`), and show the allocated `run_id` from the reply.
- [ ] Set Candidate: default the run ID to the selected record's `run_id`.
- [ ] Correct the GUI guide's field table, walkthrough and `mns_cli` equivalence table.

Location in code: `src/MnsManagerGUI.cpp` (`onSetTraining()`, `onSetCandidate()`); tagged `TODO: See TD-454`.

Files to Modify:

- `src/MnsManagerGUI.cpp`
- `docs/operations/guides/mns-manager-gui-guide.md`

---

### TD-455: MNS Manager GUI Error Detection, UI-Thread HTTP and Register Defaults

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | MNS / Manager GUI | October 14, 2026 | 2-4 hours |

Description:
Found while tracing `MnsManagerGUI`, by inspection:
- **Error detection.** Every action treats an empty body as "Connection failed" and a body containing the text `"error"` as
  failure; the `http*()` helpers return only the body, so the HTTP status is never consulted. A non-2xx reply without that
  key reads as success; a successful reply containing that text (a tag or label) reads as failure.
- **UI-thread HTTP.** Every request is a synchronous `httplib` call (connect 5 s, read 10 s) on the UI thread, so a slow or
  unreachable server freezes the window on every selection or filter change (same class as TD-263).
- **Register defaults.** The spin boxes start at d_model 128, 4 heads, d_ff 512, 6/6 layers, max sequence 1024, matching
  neither `config.mns.conf` (512/8) nor `config.trainer.conf` (768/12), for an immutable architecture; the layer minimum of 1
  prevents standalone encoder/decoder records (plus TD-452's missing fields).
- **Minor.** The role filter is never populated from `/roles`; typed filter values aren't URL-encoded; `#ifndef` guard;
  four duplicated `http*()` helpers.

Action Items:

- [ ] Return status and body from the HTTP helpers and judge success by status.
- [ ] Run requests off the UI thread (`QtConcurrent`/`QNetworkAccessManager`).
- [ ] Load Register defaults from a chosen config (or leave them blank and required); allow 0 layers per kind.
- [ ] Populate the role filter from `/roles`; URL-encode filter values.

Location in code: `src/MnsManagerGUI.cpp` (`http*()`, action slots, `createRegisterTab()`, `createModelsTab()`); tagged `TODO: See TD-455`.

Files to Modify:

- `src/MnsManagerGUI.{hpp,cpp}`

---

### TD-456: `mns_server` Treats a Mistyped `--admin-enabled` as `true` and Doesn't Range-Check `--port`

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | MNS / Server CLI | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `MnsServerArgs` and verified with the real `mns_server`:
- **Admin flag typo.** `parse_bool_flag()` returns the default for unrecognized text, and `--admin-enabled`'s default is `true`.
  Started with `--admin-enabled flase`, the server kept `PUT /admin/config` enabled; an unauthenticated
  `PUT {"registry_group":"hijacked"}` returned 200 and the server logged `admin config updated`. The admin settings
  (`registry_url`/`registry_group`) control where MNS proxies dataset-registry requests. (MNS has no authentication on any
  endpoint; this is about honouring an explicit operator choice.)
- **Port.** `std::stoi` with no range check: `--port 99999` logged `listening on port 99999` and listened on **34463**
  (mod 65536); `8083x` → 8083; `0` → a random port; negatives accepted. Clients using `NAME_SERVICE_URL` then can't find MNS
  and fall back to local architecture.
- **Minor (by inspection).** A value-taking flag with no value is reported as "Unknown argument"; `--admin-enabled` has no
  config key (unlike `metrics_api_server`'s `METRICS_API_ALLOW_CONTROL`); the header names the test
  `MnsServerArgs_test.cpp` (it's `mns_server_args_test.cpp`).

Action Items:

- [ ] Make an explicit but unrecognized `--admin-enabled` value an error (keep the lenient default only when the flag is absent).
- [ ] Parse `--port` with the shared checked helper (whole string, 1–65535) from TD-438/TD-240.
- [ ] Report "missing value for --flag"; consider an `MNS_ADMIN_ENABLED` config key; fix the header comment.

Location in code: `src/MnsServerArgs.cpp` (`parse_bool_flag()`, `parse_mns_server_args()`); tagged `TODO: See TD-456`.

Files to Modify:

- `src/MnsServerArgs.{hpp,cpp}`
- `tests/mns_server_args_test.cpp`

---

### TD-457: `ModelNameClient` Retries Non-Idempotent Requests After a Timeout

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | MNS / Client | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `ModelNameClient` and verified with a mock server. `http_post()`/`http_put()` retry up to 3 times on a
network error, **timeout** or 5xx, even when the server already processed the request. With a mock that answered the first
`PUT` after 1.5 s and a 1 s client timeout, one `set_training("m", true)` call reached the server **twice** and returned
the second reply. Against `mns_server`:
- `set_training(new_run = true)` (`incremental_trainer retrain`) increments `current_run_number` on each delivery, so a slow
  MNS skips a run number (nothing is ever recorded under the first allocation).
- `register_model()`'s retry gets `409` (the name now exists) and throws although registration succeeded (no production
  caller today).
- `set_training(new_run = false)`, `set_candidate()` and `push_progress()` are idempotent enough to repeat.

Action Items:

- [ ] Retry only idempotent requests (GETs, and PUTs explicitly marked safe), or add an idempotency key the server deduplicates for `new_run` allocation and registration.
- [ ] Add a test with a slow mock server asserting a single delivery for non-idempotent calls.

Location in code: `src/ModelNameClient.cpp` (`http_post()`, `http_put()`, `set_training()`, `register_model()`); tagged `TODO: See TD-457`.

Files to Modify:

- `src/ModelNameClient.{hpp,cpp}`
- `tests/modelnameclient_test.cpp`

---

### TD-458: A NaN Training Loss Makes MNS's JSON Unparseable

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | MNS / Client | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `ModelNameClient` and verified end to end with a real `mns_server`. `push_progress()` writes the loss with
`operator<<`, so a diverged epoch's NaN is sent as `nan`. `ModelNameService::handle_progress_update()` accepts it
(`json_double()` → `stod`) and the record serializer writes it back verbatim: `GET /models/{name}` returned
`"progress":{…,"loss":nan,…}`, and both that record and the entire `GET /models` list were rejected by Python's `json` module.
It persists (SQLite) until the next progress push for that model, and when the run is superseded the snapshot is archived into
`training_history` with `final_loss` NaN permanently. Any strict JSON client of the MNS API breaks for every model when one run
diverges, the moment monitoring matters most. Same class as TD-443 (metrics).

Action Items:

- [ ] Client: send non-finite values as `null` (or omit them).
- [ ] Server: reject or normalize non-finite numbers on input, and never emit `nan`/`inf` (write `null`).
- [ ] Repair any stored NaN `progress_loss`/`training_history.final_loss` rows on startup.

Location in code: `src/ModelNameClient.cpp` (`push_progress()`); `src/ModelNameService.cpp` (`handle_progress_update()`, the record serializer's `progress` and history output); tagged `TODO: See TD-458`.

Files to Modify:

- `src/ModelNameClient.cpp`
- `src/ModelNameService.cpp`
- `tests/modelnameclient_test.cpp`

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

### TD-460: MNS Retirement Leaves Role Mappings and Is Undone by Running Trainers

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | MNS / Server | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `ModelNameService` and verified against a real `mns_server`:
- **Stale role.** `retired` is reachable from any state via `PUT /models/{n}/state`, but only `handle_promote()`/`handle_delete()`
  update `roles_`, and `handle_resolve_role()` doesn't check state. After promoting `prod-a` to `chatbot` and retiring it,
  `GET /roles/chatbot/production` still returned `prod-a` (`"state":"retired"`, with its artifact), so a `chatbot_api_server`
  with `MODEL_ROLE=chatbot` keeps loading the retired model.
- **Revival by a running trainer.** `candidate` may be entered from `retired` ("revival"), and the `run_id` ownership check
  applies only when leaving `training`. Retiring a model during `run-01`, then sending the trainer's normal
  `{"state":"candidate","run_id":"run-01",…}` returned 200 and state `candidate`: the operator's retirement is silently undone
  at the end of the pass.

Action Items:

- [ ] On `→ retired`, remove any role mapping to the model (or refuse to retire a `production` model without a replacement); make `resolve_role` return 404/409 for a non-`production` record.
- [ ] Require an explicit flag (or no `run_id`) for `retired → candidate` revival, and reject a trainer's `run_id`-bearing candidate call when the model isn't `training`.
- [ ] Live tests for both.

Location in code: `src/ModelNameService.cpp` (`handle_state_transition()`, `handle_resolve_role()`); tagged `TODO: See TD-460`.

Files to Modify:

- `src/ModelNameService.cpp`
- `tests/model_name_service_live_test.cpp`

---

### TD-461: MNS Accepts Unbuildable Architectures at Registration

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | MNS / Server | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `ModelNameService` and verified against a real `mns_server`. Architecture is immutable after
registration, but `handle_register()` validates it only for standalone `encoder`/`decoder`/`world_model` kinds (positive
fields, correct layer field). A legacy (inline) `chatbot` gets **no validation**: `POST /models {"model_name":"noarch","role":"chatbot"}`
returned 201 and stored every architecture field as 0 (which `ModelNameClient::get_architecture()` then hands to the trainer and chat
server, TD-459). No kind checks divisibility: `d_model 100, num_heads 8` returned 201 although `MultiHeadAttention`'s constructor
rejects `d_model % num_heads != 0`, so the model can never be built. The only remedy is delete and re-register.

Action Items:

- [ ] Require all six fields positive for a legacy chatbot, and `d_model % num_heads == 0` for every kind; return 400 with the offending field.
- [ ] Live tests for missing arch and non-divisible heads.

Location in code: `src/ModelNameService.cpp` (`handle_register()`); tagged `TODO: See TD-461`.

Files to Modify:

- `src/ModelNameService.cpp`
- `tests/model_name_service_live_test.cpp`

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

- [ ] Fix or remove the datasets proxy (record `model_id` on mark-trained, use the model's `run_group` and kind).
- [ ] Guard the registry settings with a mutex.
- [ ] Return 500 (and roll back the in-memory change) when persistence fails.
- [ ] Remove the old role mapping when a model is promoted to a different role.

Location in code: `src/ModelNameService.cpp` (`json_string()`, `handle_datasets()`, `handle_admin_put_config()`, `persist_model()`, `handle_promote()`); tagged `TODO: See TD-462`.

Files to Modify:

- `src/ModelNameService.{hpp,cpp}`
- `src/RegistryTransport.cpp`

---

### TD-463: `mns_server` Runs In-Memory and Reports Healthy When `models.db` Can't Be Opened

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | MNS / Server | October 14, 2026 | 1-2 hours |

Description:
Found while tracing `ModelNameServiceServer` and verified with the real `mns_server`. `ModelNameService::init_db()` logs
`cannot open SQLite DB …` and returns with no database handle; every `persist_*()` then returns early, and neither `start()` nor
`main()` treats it as fatal. With `<data_dir>/models.db` made unopenable, the server started, `/health` returned
`{"status":"ok"}`, and `POST /models` returned 201; after a restart the model was gone (`404`, `model_count: 0`). Architecture,
run numbering and roles, everything MNS is authoritative for, silently stop surviving restarts.

Action Items:

- [ ] Make a failed `init_db()` (and a failed schema creation) fatal at startup with a clear message and non-zero exit.
- [ ] Optionally report persistence status in `/health`.

Location in code: `src/ModelNameService.cpp` (`init_db()`, `start()`); tagged `TODO: See TD-463`.

Files to Modify:

- `src/ModelNameService.cpp`
- `src/ModelNameServiceServer.cpp`

---

### TD-464: `mns_server` Startup Crash, Early-SIGTERM Hang and Fixed Log Level

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | MNS / Server | October 14, 2026 | 1-2 hours |

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
- [ ] Only set a flag in the signal handler; after `stop()`, re-check the flag once `listen()` has started (or use `wait_until_ready()` before stopping).
- [ ] Honour `LOG_LEVEL`; update the endpoint lists.

Location in code: `src/ModelNameServiceServer.cpp` (`signal_handler()`, `main()`); `src/ModelNameService.cpp` (`start()`); tagged `TODO: See TD-464`.

Files to Modify:

- `src/ModelNameServiceServer.cpp`
- `src/ModelNameService.cpp`

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

Recomputed directly from the 256 `### TD-NNN` entries under [Active Technical Debt](#active-technical-debt) — re-derive this from that list rather than trusting it blindly once an item resolves or a new one is filed.

|Priority|Count|Percentage|
|----------|-------|------------|
|High|28|11%|
|Medium|111|43%|
|Low|117|46%|

**Total Active Items:** 256

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
|Inference / Serving|3|
|Inference / Serving / Observability|1|
|Core Math / Activation|2|
|Inference / Serving / API|15|
|Inference / Serving / CLI|6|
|Inference / Serving / Server|7|
|Client / CLI|7|
|Client / GUI|6|
|Core Model / Generation|1|
|Training / ChatbotTrainer|10|
|Training / Service Supervisor|6|
|Config|8|
|NLP / ConversationContext|6|
|Attention / CrossAttention|7|
|Core / DaemonConfigStore|3|
|Admin config API|2|
|Data / DataFetcher|6|
|Data / dataset_manager CLI|12|
|Data / DatasetRegistry|4|
|Data / DataTransport|4|
|Transformer / DecoderBlock|5|
|Transformer / Decoder|3|
|RAG / DocumentStore|3|
|Transformer / EncoderBlock|2|
|Models / EncoderDecoderModel|6|
|Transformer / LLMEncoder|4|
|Transformer / FeedForward|4|
|Data / FtpDataServer|6|
|Metrics / GenerationQuality|3|
|GPU / GPUManager|6|
|GPU / Kernels|4|
|GPU / Testing|1|
|Memory / HippocampalMemory|3|
|Metrics / Reporter|2|
|Training / CLI|4|
|Training / IncrementalTrainer|6|
|Training / IncrementalTrainingTool|6|
|Inference / IntegratedInferenceEngine|4|
|Inference / KVCache|2|
|Models / LanguageModelHead|3|
|Layers / LayerNorm|1|
|World Model / LeJEPAEncoder|3|
|Core / Logger|2|
|Attention / LoRA|2|
|Core / Matrix|2|
|Build / SIMD|2|
|Metrics / Server CLI|2|
|Metrics / Database|2|
|Metrics / Push Client|3|
|Metrics / Session Registry|2|
|Training / MetricsTracker|1|
|MNS / CLI|4|
|MNS / Manager GUI|3|
|MNS / Server CLI|1|
|MNS / Client|3|
|MNS / Server|5|
|Core / JSON|1|
|GPU / GPUMatrix|4|
|Core / Batching|1|
|NLP / Tokenizer|6|

### Effort Distribution

|Effort Range|Count|
|--------------|-------|
|0-2 hours|156|
|2-4 hours|76|
|4-8 hours|14|
|8+ hours|6|
|Not estimated|4|

**Total Estimated Effort (Active Items):** 470-842 hours (excludes TD-014, TD-039, TD-171, and TD-224, which have no effort estimate, and TD-211's real-batching option, which is blocked on TD-171; TD-211 is counted at its 2-4 hour re-scope estimate. The entire TD-174 through TD-186 LeJEPA world-model batch is now resolved — see Tier 10 in the Recommended Execution Order above — so it no longer contributes to this total at all.)

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
