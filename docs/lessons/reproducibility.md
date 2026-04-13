# Reproducibility

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

A training run that cannot be reproduced is an experiment that cannot be trusted. If you cannot rerun an experiment and get the same result, you cannot confidently attribute improvements to specific changes, debug regressions, or hand a working configuration to another engineer.

Reproducibility is not a single setting — it is a discipline applied across the entire pipeline: code, data, environment, and random state. Each of these must be controlled independently.

---

## The Four Sources of Non-Reproducibility

**1. Random state** — weight initialization, data shuffling, and dropout all draw from random number generators. Different seeds produce different results.

**2. Data ordering** — the sequence in which training examples are presented affects the optimizer's trajectory. An undocumented shuffle produces a valid but unrepeatable run.

**3. Non-deterministic operations** — GPU floating-point reductions (e.g., `atomicAdd`) perform operations in an undefined order. The same inputs can produce slightly different outputs across runs due to floating-point non-associativity.

**4. Environment drift** — library versions, compiler flags, CUDA versions, and hardware generation can all change the numerical behavior of a training run, even with identical code and seeds.

---

## Seeding

Every source of randomness must be seeded explicitly. For PyTorch-based training:

```python
import random, numpy as np, torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Call this before model construction, before data loader construction, and before any operation that draws from a random distribution.

**Seed scope**: the seed controls the RNG state from that point forward. Any operation that draws a random number before the seed is set is uncontrolled. Common oversights:

- Data augmentation applied in worker processes uses a separate RNG; seed each worker explicitly via the data loader's `worker_init_fn`.
- Dropout in the model uses the main RNG; seeding before the forward pass gives deterministic dropout patterns.
- `torch.randperm` for data shuffling should use an explicit generator object rather than the global RNG, so shuffling and model operations do not share state.

**Seed independence**: use different seeds for model initialization and data ordering. If both use the same seed, changing the model architecture changes the data order, making ablations uninterpretable.

---

## Deterministic GPU Operations

By default, CUDA operations involving parallel reductions are non-deterministic. The same model, same data, and same seed will produce results that differ in the last few bits across runs.

To enforce determinism:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**`cudnn.benchmark = False`**: cuDNN benchmarks several convolution algorithms on first use and selects the fastest. The fastest algorithm may vary across runs or hardware. Disabling benchmarking forces a fixed algorithm.

**`use_deterministic_algorithms(True)`**: raises an error if any operation lacks a deterministic implementation. Some operations (e.g., certain scatter operations, some attention backends) do not have deterministic GPU implementations. The error tells you where the non-determinism is rather than silently allowing it.

**Performance cost**: deterministic mode is typically 10–30% slower. Use it for ablations; accept non-determinism in production training runs where throughput matters.

---

## Data Pipeline Reproducibility

The data pipeline is one of the most commonly overlooked sources of non-reproducibility.

**Shuffling**: use a seeded shuffle with a documented seed. If the shuffle seed is not recorded, the data order cannot be reconstructed.

**Multi-process data loading**: PyTorch's `DataLoader` spawns worker processes. Each worker has its own RNG state. Without explicit seeding in `worker_init_fn`, each run produces a different worker RNG sequence:

```python
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

loader = DataLoader(dataset, worker_init_fn=worker_init_fn)
```

**Dataset versioning**: the dataset itself must be fixed. A dataset that changes between runs (e.g., a live database, a directory where files are added or removed) breaks reproducibility regardless of seeding.

**Tokenization**: tokenization must be deterministic and version-locked. A tokenizer library update that changes normalization behavior will silently alter token IDs across runs.

---

## Environment Reproducibility

Results that depend on a specific library version are not portable. The environment must be recorded so the run can be reconstructed.

**At minimum, record**:

``` text
Python version
PyTorch version + CUDA version
CUDA driver version
cuDNN version
numpy, transformers, tokenizers library versions
OS and Linux kernel version
GPU model and driver
```

Use a lock file (`requirements.txt` with pinned versions, or `conda env export`) to capture the full environment. A container image (Docker) is the strongest form of environment reproducibility — the entire software stack is frozen.

**CUDA and hardware**: different GPU generations (Volta, Ampere, Hopper) may produce different floating-point results for the same operation due to differences in the underlying hardware implementations of operations like fused multiply-add. Results obtained on an A100 may differ slightly from results on a V100, even with identical code and seeds.

---

## Code Versioning

Every training run must be associated with a specific, recoverable version of the code.

**Git commit hash**: record the exact commit hash in every experiment log. This allows the code state to be reconstructed exactly, including any uncommitted changes (which should be absent — never run an experiment on dirty code).

**Experiment configuration**: all hyperparameters must be recorded alongside the run. Do not rely on reconstructing the configuration from memory or documentation — store the actual config file or a serialized copy of the hyperparameter dict.

A minimal experiment log entry:

``` text
run_id:        exp_042
git_commit:    a3f8c21d
date:          2026-04-12
seed:          42
config:        {lr: 3e-4, batch_size: 512, max_steps: 10000, ...}
val_loss:      2.341
best_step:     8200
```

---

## Checkpoint Reproducibility

A checkpoint saves the model state but must also save the full training state to resume training reproducibly:

- Model weights
- Optimizer state (Adam moment estimates)
- LR scheduler state
- **RNG state** (CPU RNG, CUDA RNG for each device)
- Step number
- Data loader state (current position in the epoch, shuffle state)

Without the RNG state, a resumed run will draw different random numbers from the point of resumption onward, producing a different trajectory from a run that never paused.

```python
checkpoint = {
    'step': step,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'rng_cpu': torch.get_rng_state(),
    'rng_cuda': torch.cuda.get_rng_state_all(),
    'rng_numpy': np.random.get_state(),
    'val_loss': val_loss,
}
```

---

## Levels of Reproducibility

Not all reproducibility requirements are equal. Match the level to the need:

| Level | What is guaranteed | Use case |
| --- | --- | --- |
| **Bit-exact** | Identical outputs to the last bit | Debugging a specific numerical failure |
| **Statistical** | Same loss curve within noise bounds | Verifying a configuration works |
| **Qualitative** | Same final model quality and behavior | Confirming an improvement generalizes |

Bit-exact reproducibility requires deterministic mode, identical hardware generation, identical library versions, and complete RNG state capture. It is expensive to maintain and rarely necessary outside of debugging.

Statistical reproducibility — same loss curve within the noise of training — is the practical standard for research and production work. It requires seeding, version recording, and environment documentation.

---

## Common Mistakes

- **Setting the seed once but not before every random operation.** The RNG state drifts from the seed as operations consume random numbers; what matters is the state at the start of each reproducible unit (model init, data shuffle, training loop).
- **Not seeding DataLoader workers.** These run in separate processes and do not inherit the main process seed.
- **Running experiments on uncommitted code.** A git commit hash of a dirty repository is meaningless — the uncommitted changes are not recoverable.
- **Conflating non-determinism with randomness.** Non-determinism (different results from same seed) is a separate problem from stochasticity (different results from different seeds). Non-determinism must be eliminated for reproducibility; stochasticity is managed by seed control.
- **Recording only the final metric, not the full config.** A run that produced a good result but whose configuration cannot be reconstructed is effectively lost.

---

## Quick Decision Checklist

``` text
1. Set all seeds (Python, NumPy, PyTorch CPU, PyTorch CUDA) before model
   construction and before data loader construction.
2. Use a different seed for model initialization and data shuffling.
3. Seed DataLoader workers explicitly via worker_init_fn.
4. Enable deterministic mode for ablations; accept non-determinism for
   production runs.
5. Record git commit hash, full config, and environment (pip freeze or
   Docker image tag) with every experiment.
6. Never run an experiment on uncommitted code.
7. Save RNG state in checkpoints alongside model and optimizer state.
8. Version and freeze the dataset. Do not use a mutable data source.
```

---

*See also: [Evaluation and Checkpointing Strategy](evaluation-and-checkpointing-strategy.md) — checkpoints must save RNG state to resume training reproducibly. [Data Quality and Selection](data-quality-and-selection.md) — dataset versioning is part of reproducibility; a changing corpus invalidates experiment comparisons.*
