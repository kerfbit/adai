# `LLMEncoder` — Source File Reference

- **Files:** [`src/encoder.hpp`](../../../../src/encoder.hpp) (303 lines, declaration), [`src/LLMEncoder.cpp`](../../../../src/LLMEncoder.cpp) (455, implementation)
- **Namespace:** global (no namespace)
- **Built into:** `adai_transformer`; used by `EncoderDecoderModel` (the chatbot's encoder), `DocumentStore` (the RAG retrieval encoder), `ModelSerializer` and `RLHFTrainer`
- **Status tag:** `@adai-status: stable`; `encoder.hpp` `@adai-version: 1.6.0`, `LLMEncoder.cpp` `1.7.0`; both `@adai-reviewed: 2026-11-12`
- **Tests:** [`tests/llmencoder_test.cpp`](../../../../tests/llmencoder_test.cpp) → `llmencoderTests` (42 tests); `encoderdecoder_test.cpp` covers checkpoint loading and the update step through the model
- **Last traced against the code:** 2026-11-12

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class
> (linked against the debug libraries) in a scratch program.

---

## 1. What this file is

The **encoder stack**: token embedding, then sinusoidal positional encoding, then `num_layers` Pre-LN
[`EncoderBlock`](EncoderBlock.md)s, then a final `LayerNorm`. The final norm is required because Pre-LN
blocks leave the residual stream unnormalized. The output is a `[seq_len × d_model]` matrix.

```text
token ids ─→ TokenEmbedding ─→ + PositionalEncoding ─→ EncoderBlock × num_layers ─→ final LayerNorm ─→ [seq × d_model]
   ↑
encode(text) only: the encoder's own BPETokenizer, no special tokens by default
```

The file is named `encoder.hpp`, but the class is `LLMEncoder` and the implementation is
`LLMEncoder.cpp`. This doc covers both.

### Why it matters

There are two live instances with different roles:

| Instance | Created by | Entry point | Weights |
|---|---|---|---|
| The chatbot encoder | `EncoderDecoderModel` constructor | `encode_with_mask()` (CPU) / `gpu_encode()` (GPU), with the **model's** tokenizer, no special tokens, truncated by `keep_last_positions()` | Trained; saved as `<checkpoint>.encoder` + sidecars |
| The RAG retrieval encoder | `EncoderDecoderModel::copy_encoder()`, called by `ChatbotAPIServer.cpp` | `encode(text)` and `count_tokens(text)` via `DocumentStore` | A copy of the trained encoder and the model's vocabulary, in inference mode (TD-247) |

---

## 2. API

### Construction and state

| Member | Behaviour |
|---|---|
| `LLMEncoder(vocab_size, d_model = 512, num_layers = 6, num_heads = 8, d_ff = 2048, max_seq_length = 512)` | Builds an empty `BPETokenizer`, `TokenEmbedding(vocab_size, d_model)`, `PositionalEncoding(max_seq_length, d_model)`, the blocks (each scaling its residual output projections for depth, TD-371) and the final norm. **Prints a 7-line configuration banner to `std::cout`** on every construction (TD-367) |
| `requires_grad` | Defaults to **`true`**, so a fresh training run caches what `backward()` needs. When true, `encode*()` keep the token ids. Serving instances clear it: `EncoderDecoderModel::set_training(false)` and `copy_encoder()` (TD-366) |
| `learning_rate` | 0.001. `set_learning_rate(lr)` writes it to the token embedding, every block's public `learning_rate` and the final norm (the TD-093 fix). Only matters on the no-optimizer path |
| `set_dropout(rate, active)` | Every block's dropout (TD-344); `EncoderDecoderModel::set_dropout()` calls it |

### Encoding

| Method | Behaviour |
|---|---|
| `encode(text, add_special_tokens = false)` | Tokenizes with the encoder's own tokenizer. By default no special tokens, matching how `EncoderDecoderModel` tokenizes encoder input (TD-364), and an over-long input keeps its **last** `max_seq_length` tokens (`keep_last_positions()`, TD-365). With special tokens, BOS and EOS stay at the ends and the body keeps its last `max_seq_length − 2` tokens. Logs the token count at debug level. No attention mask (each block gets `nullptr`) |
| `keep_last_positions(ids)` | The one truncation rule for encoder input: the last `max_seq_length` ids, as `ChatbotTrainer` truncates training inputs (TD-365). `EncoderDecoderModel` and `RLHFTrainer` apply it before encoding |
| `count_tokens(text)` | The token count `encode()` starts from: the tokenizer's ids for `text`, no special tokens, **before** truncation; 0 for empty or all-whitespace text. `DocumentStore` sizes its chunks with it (TD-349) |
| `get_max_seq_length()` | The window `encode()` truncates to |
| `encode_with_mask(ids, mask)` | Pre-tokenized ids and an explicit mask, passed to every block. Throws `invalid_argument` for more than `max_seq_length` ids (TD-365: positions past it would get no positional encoding); callers truncate first |
| `get_sentence_embedding(text)` / `get_sentence_embedding_trainable(text)` | Mean-pool `encode(text)` over rows; the second calls the first (TD-366). Neither has a production caller: `DocumentStore` does its own pooling over `encode()` |
| `backward_sentence_embedding(grad)` | Spreads `grad / seq_len` over every row and calls `backward()`. Tests only |

### Gradients and training

| Method | Behaviour |
|---|---|
| `backward(grad)` | No-op unless `requires_grad`. Final norm, then blocks in reverse, then `token_embedding->backward(cached_token_ids, grad)`. Positional encoding has no parameters |
| `zero_grad()` | Embedding, every block, final norm |
| `register_parameters_with_optimizer(opt)` | `set_optimizer()` on the embedding and final norm, `register_parameters_with_optimizer()` on each block, and keeps the optimizer |
| `update_weights()` | TD-360: with a registered optimizer, one `step()` and `zero_grad()` (TD-297: once, not once per sub-layer); otherwise plain SGD in the embedding, every block and the final norm |

### Tokenizer

| Method | Behaviour |
|---|---|
| `load_tokenizer_vocab(path)` | Loads the encoder's **own** tokenizer. Caller: `copy_encoder()` (the RAG encoder, with the model's vocabulary). Prints to `std::cout` (TD-367) |
| `build_tokenizer(corpus, vocab_size = 10000)` | Tests only. Prints to `std::cout` |

Inside `EncoderDecoderModel` the encoder's own tokenizer is **empty**: the model tokenizes with its
own and calls `encode_with_mask()`/`gpu_encode()`. There's no accessor for the encoder's tokenizer, so
`load_tokenizer_vocab()` is the only way to fill it.

### Persistence

| Method | Behaviour |
|---|---|
| `save_weights(file)` | Writes a 6-int header (`vocab_size`, `d_model`, `num_layers`, `num_heads`, `d_ff`, `max_seq_length`) with checked writes (`adai::binio::Writer`, TD-423), then `<file minus extension>_encoder_token_emb.bin`, `_encoder_block_<i>.bin` (each block adds its own sub-layer files) and `_encoder_final_norm.bin`. Prints to `std::cout` |
| `load_weights(file)` | Reads the header with checked reads (a short or over-long header throws), and throws on a mismatch of five fields; **`max_seq_length` is read but not compared** (harmless for sinusoidal encoding, TD-367). Loads the encoder-specific embedding/norm files, or falls back with a warning to the old shared `_token_emb.bin`/`_final_norm.bin` that pre-TD-538 checkpoints have. Each sub-layer's loader checks its own file (TD-423). Prints to `std::cout` |

Before TD-538's fix (October 18, 2026) the embedding and norm used `_token_emb.bin`/`_final_norm.bin`,
the decoder's names too, so inside an `EncoderDecoderModel` checkpoint the decoder's copies overwrote
the encoder's (logits changed by up to 1.54 on a round trip).

`EncoderDecoderModel::save_model()`/`load_model()` call these with `<checkpoint>.encoder`;
`copy_encoder()` saves and reloads through them. `ModelSerializer` (SafeTensors) bypasses them and
goes through the accessors.

### Accessors

`get_num_layers()`, `get_encoder_block(i)` (throws `out_of_range`; used for LoRA and SafeTensors),
`get_token_embedding()`, `get_final_norm()` and `get_embedding_dim()` (returns `d_model`).
`print_config()` writes to `std::cout` and has no production caller.

### GPU (`ADAI_ENABLE_GPU`)

| Method | Behaviour |
|---|---|
| `gpu_upload_weights()` / `gpu_download_weights()` / `gpu_download_grads()` / `gpu_zero_grads()` / `gpu_collect_parameters()` | Token embedding, every block and the final norm. `gpu_collect_parameters()` hands the device buffers to `GPUOptimizer` (TD-468) |
| `gpu_encode(ids)` | Throws past `max_seq_length` (TD-365). Token embedding and positional encoding on the device (`TokenEmbedding::gpu_forward()` with the positional table uploaded once, TD-547), so only the ids cross; then the blocks with **no mask** and the final norm |
| `gpu_backward(dout)` | No-op unless `requires_grad`. Final norm and blocks in reverse, then the embedding's scatter-add into its device gradient (TD-547) |

The CPU training path passes an all-ones `seq × seq` mask to `encode_with_mask()`; the GPU path skips
the mask, so the two are equivalent.

---

## 3. Who calls what

| Caller | Uses |
|---|---|
| `EncoderDecoderModel` | Constructor; `keep_last_positions()`, `encode_with_mask()`, `backward()`, `zero_grad()`, `update_weights()`, `register_parameters_with_optimizer()`, `set_requires_grad()`, `set_learning_rate()`, `set_dropout()`, `save_weights()`/`load_weights()`, `get_encoder_block()`/`get_num_layers()`; the GPU methods. `copy_encoder()` builds the RAG copy |
| `DocumentStore` | `encode(text)`, `count_tokens(text)` and `get_max_seq_length()`, when adding documents and searching ([DocumentStore.md](DocumentStore.md)) |
| `ChatbotAPIServer.cpp` | Takes `copy_encoder()` for RAG |
| `RLHFTrainer` | `keep_last_positions()` and `encode_with_mask()` on the model's encoder |
| `ModelSerializer` | `get_token_embedding()`, `get_encoder_block()`, `get_final_norm()` for SafeTensors save/load |

`LeJEPAEncoder` (the world model) is a separate class with a similar shape; it doesn't use
`LLMEncoder`.

---

## 4. Tests

`llmencoder_test.cpp` has 42 tests:
- construction with default, custom and minimal configurations;
- tokenizer load/build, including a missing file and an empty corpus;
- encode shapes, long-text truncation, empty strings, single words, repeats, Unicode and special characters;
- `encode()` against the model's own tokenization, and over-long input keeping its tail, with and
  without special tokens (TD-364, TD-365);
- `encode_with_mask()`, and its refusal past `max_seq_length`;
- `count_tokens()` as the untruncated `encode()` length, 0 for whitespace (TD-349);
- sentence embeddings (consistency, distinctness, the trainable variant agreeing in either mode);
- `set_requires_grad()`, `set_learning_rate()` and its propagation to blocks;
- `zero_grad()`, backward passes, `backward_sentence_embedding()`, a training workflow;
- save/load, including a missing file, a mismatched architecture and an output-preserving round trip;
- `get_embedding_dim()`, `print_config()`, optimizer registration and workflow.

`encoderdecoder_test.cpp` checks every checkpoint file on load (`EveryCheckpointFileIsCheckedOnLoad`,
TD-423) and one update per step (`OneUpdateIsOneStepWithOrWithoutAnOptimizer`, TD-297/TD-360).
`DocumentStoreTests` exercise `count_tokens()` on real chunking.

**Not covered:** the GPU methods, except through `EncoderDecoderGPUTest` in GPU builds.

---

## 5. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `std::cout` in the constructor, save/load, the tokenizer methods and `print_config()`; `load_weights()` doesn't compare `max_seq_length` (§2) | Console noise from library code; a header field silently ignored | By inspection | [TD-367](../../guides/TECHNICAL_DEBT.md#td-367-llmencoder-console-output-and-unchecked-weight-io) |

---

## 6. History

- **TD-349 (November 12, 2026):** `count_tokens()`, for `DocumentStore`'s chunking.
- **TD-547 (October 19, 2026):** the token embedding moved to the device for `gpu_encode()`/`gpu_backward()`
  and the GPU weight/gradient methods; it used to be looked up on the host and uploaded every sample.
- **TD-538 (October 18, 2026):** encoder-specific embedding and final-norm file names (§2).
- **TD-364, TD-365 (October 26, 2026):** `encode()` defaults to no special tokens, keeps BOS/EOS
  through truncation when asked for them, and every encoder input keeps its last `max_seq_length`
  tokens; `encode_with_mask()`/`gpu_encode()` refuse longer input. Before, `encode()` added BOS/EOS the
  trained chat encoder had never seen (a shifted, out-of-distribution input for pipeline inference),
  truncation dropped the EOS it had just added, and over-length serving input ran with position-less
  tokens.
- **TD-423 (October 26, 2026):** checked header reads and writes; a failed load leaves the layer
  unchanged.
- **TD-297, TD-360 (October 26, 2026):** `update_weights()`, one optimizer step per update.
- **TD-366 (October 26, 2026):** the unread per-layer activation copies and two unused matrices
  removed; serving encoders run in inference mode; `get_sentence_embedding_trainable()` no longer a
  copy.
- **TD-350 (October 27, 2026):** `encode()` logs its token count at debug level instead of printing
  to stdout on every call.
- **TD-247 (October 26, 2026):** the RAG encoder is a copy of the trained one; it used to be a fresh
  `LLMEncoder` with random weights.
- **Retired engine modes:** `PipelineInferenceEngine` and `IntegratedInferenceEngine` (and
  `ChatbotAPI::enable_pipeline_inference()`) were removed in October 2026 (TD-474/TD-475,
  TD-417–TD-420); they called `encode()` and `encode_with_mask()`.

This file replaced `docs/development/api/transformer/encoder.md` (1,515 lines). That doc described
the block as Post-LN (the code is Pre-LN with a final norm), spent half its length on integrating with
a `NeuralNetwork` class that doesn't exist in `src/`, restated other classes' interfaces, and gave
optimization advice that is out of date. Its component hierarchy, save/load layout and
semantic-similarity use case were kept.
