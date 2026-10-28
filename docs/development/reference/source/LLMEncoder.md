# `LLMEncoder` — Source File Reference

- **Files:** [`src/encoder.hpp`](../../../../src/encoder.hpp) (declaration), [`src/LLMEncoder.cpp`](../../../../src/LLMEncoder.cpp) (implementation)
- **Namespace:** global (no namespace)
- **Built into:** `adai_transformer`; used by `EncoderDecoderModel` (the chatbot's encoder), `DocumentStore` (the RAG retrieval encoder), (until their retirement in October 2026) `PipelineInferenceEngine` and `IntegratedInferenceEngine`
- **Status tag:** `@adai-status: stable`, `LLMEncoder.cpp` `@adai-version: 1.1.0`, `@adai-reviewed: 2026-10-18`; `encoder.hpp` `1.0.1`, `2026-09-10`
- **Tests:** [`tests/llmencoder_test.cpp`](../../../../tests/llmencoder_test.cpp) → `llmencoderTests` (37 tests)
- **Last traced against the code:** 2026-10-19

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
encode(text) only: the encoder's own BPETokenizer, with BOS/EOS added
```

The file is named `encoder.hpp`, but the class is `LLMEncoder` and the implementation is
`LLMEncoder.cpp`. This doc covers both.

### Why it matters

There are three live instances with different roles:

| Instance | Created by | Entry point | Weights |
|---|---|---|---|
| The chatbot encoder | `EncoderDecoderModel` constructor | `encode_with_mask()` (CPU) / `gpu_encode()` (GPU), with the **model's** tokenizer, no special tokens | Trained; saved as `<checkpoint>.encoder` + sidecars |
| The pipeline inference encoder stage | Same object as above (`model_->get_encoder()`) | `encode(text)`: the **encoder's own** tokenizer, which `ChatbotAPI::enable_pipeline_inference()` loads with the same vocab | Trained |
| The RAG retrieval encoder | `ChatbotAPIServer.cpp` (`make_shared<LLMEncoder>`) | `encode(text)` via `DocumentStore` | **Never loaded** (random, TD-247) |

---

## 2. API

### Construction and state

| Member | Behaviour |
|---|---|
| `LLMEncoder(vocab_size, d_model = 512, num_layers = 6, num_heads = 8, d_ff = 2048, max_seq_length = 512)` | Builds an empty `BPETokenizer`, `TokenEmbedding(vocab_size, d_model)`, `PositionalEncoding(max_seq_length, d_model)`, the blocks and the final norm. **Prints a 7-line configuration banner to `std::cout`** on every construction, including inside `EncoderDecoderModel` and the RAG setup |
| `requires_grad` | Defaults to **`true`**. When true, `encode*()` cache token IDs and per-layer outputs for `backward()`. No production caller ever sets it false, so serving instances (RAG, chat) cache activations too (see `cached_encoder_outputs` below) |
| `learning_rate` | 0.001. `set_learning_rate(lr)` writes it to the token embedding, every block's public `learning_rate` and the final norm (the TD-093 fix). Only matters on a no-optimizer path; with an optimizer registered, the optimizer's LR rules |
| Unused members | `cached_embeddings` and `cached_pos_encoded` are declared and never written or read. `cached_encoder_outputs` is written (one matrix per layer, plus the input) on every `encode*()` with `requires_grad`, but **never read**: `backward()` relies on each block's own caches. That's `(num_layers + 1) × seq × d_model` floats of dead copies per call |

### Encoding

| Method | Behaviour |
|---|---|
| `encode(text)` | `tokenizer->encode(text, true)`, so **BOS and EOS are added**. If the result is longer than `max_seq_length`, it's cut with `resize()`, **which drops the trailing EOS** (verified: 32 ids cut to 8, no EOS). Prints `"Encoding N tokens..."` to `std::cout` **on every call**, so on every RAG document added, every RAG query and every pipeline request. No attention mask is passed (each block gets `nullptr`) |
| `encode_with_mask(ids, mask)` | Takes pre-tokenized IDs and an explicit mask, passed to every block. **No truncation**: a sequence longer than `max_seq_length` gets `PositionalEncoding`'s warning, and the positions beyond it get no positional encoding (verified below) |
| `get_sentence_embedding(text)` / `get_sentence_embedding_trainable(text)` | Mean-pool `encode(text)` over rows. **The two are identical** apart from loop style. Neither has a production caller; `DocumentStore` does its own pooling (`getSentenceEmbedding()`) over `encode()` |
| `backward_sentence_embedding(grad)` | Spreads `grad / seq_len` over every row and calls `backward()`. Tests only |

> **Train/serve mismatch in `encode()` (verified).** `EncoderDecoderModel` always tokenizes encoder
> input with `tokenizer->encode(input_text, false)` (no special tokens), in training (`train_step`,
> `compute_loss`, `gpu_train_step`) and in generation. `encode(text)` adds BOS and EOS. For
> `"hello world"`: the model's encoder sees `[80, 77]`; `encode()` produces `[2, 80, 77, 3]`. Every
> real token sits one position later, and the encoder sees two tokens it never saw at input during
> training. The first output row differed (1.398 vs 0.980). This matters for **pipeline inference**
> (`PipelineInferenceEngine.hpp:276` calls `encoder_->encode(text)` on the trained chat encoder and
> hands the result to the decoder's cross-attention). For RAG it's only a consistency issue, since
> both documents and queries go through the same path.

> **Over-length input is never truncated in serving (verified).** `ChatbotAPI` has no truncation of
> its own, and `generate_response_with_strategy()` passes the full token list to
> `encode_with_mask()` (CPU) or `gpu_encode()` (GPU). A 60-token input on a `max_seq_length = 16`
> model completed generation, and printed *"Input sequence length (60) exceeds max_len (16).
> Positions beyond max_len will not receive positional encodings."* So 44 positions were
> position-less, which makes them indistinguishable by order to attention. Attention cost also grows
> as O(n²) without bound. Truncation (or rejection) should happen once, where the model tokenizes.

### Gradients and training

| Method | Behaviour |
|---|---|
| `backward(grad)` | No-op unless `requires_grad`. Final norm, then blocks in reverse, then `token_embedding->backward(cached_token_ids, grad)`. Positional encoding has no parameters |
| `zero_grad()` | Embedding, every block, final norm |
| `register_parameters_with_optimizer(opt)` | `set_optimizer()` on the embedding and final norm; `register_parameters_with_optimizer()` on each block. Called from `EncoderDecoderModel::register_parameters_with_optimizer()` |
| *(no `update_weights()`)* | There's no method to apply gradients without an optimizer. `EncoderDecoderModel::update_weights()` therefore skips the encoder, and its `set_learning_rate()` has `encoder->set_learning_rate(lr)` commented out as *"LLMEncoder doesn't have this method"*. **That comment is stale**: `set_learning_rate()` exists (added by the TD-093 fix); only `update_weights()` is missing. See [EncoderDecoderModel.md](EncoderDecoderModel.md) and TD-360. The shipped trainers register an optimizer and step it once, so they're unaffected |

### Tokenizer

| Method | Behaviour |
|---|---|
| `load_tokenizer_vocab(path)` | Loads the encoder's **own** tokenizer. Callers: `ChatbotAPIServer.cpp` (the RAG encoder) and `ChatbotAPI::enable_pipeline_inference()` (the chat encoder, for pipeline mode only). Prints to `std::cout` |
| `build_tokenizer(corpus, vocab_size = 10000)` | Tests only. Prints to `std::cout` |

Inside `EncoderDecoderModel` the encoder's own tokenizer is **empty** unless pipeline inference is
enabled; the model uses its own tokenizer and `encode_with_mask()`. There's no accessor for the
encoder's tokenizer, so `load_tokenizer_vocab()` is the only way to fill it.

### Persistence

| Method | Behaviour |
|---|---|
| `save_weights(file)` | Writes a 6-int header (`vocab_size`, `d_model`, `num_layers`, `num_heads`, `d_ff`, `max_seq_length`), then `<file minus extension>_encoder_token_emb.bin`, `_encoder_block_<i>.bin` (each block adds its own `_attention.bin`/`_feedforward.bin`) and `_encoder_final_norm.bin`. Before TD-538's fix (October 18, 2026) the embedding and norm used `_token_emb.bin`/`_final_norm.bin`, the same names as `LLMDecoder`'s; inside an `EncoderDecoderModel` checkpoint both strip to the same base, so the decoder's copies overwrote the encoder's. Write errors aren't checked. Prints to `std::cout` |
| `load_weights(file)` | Reads the header and throws on a mismatch of five fields. Since TD-538's fix, it loads the encoder-specific embedding/norm files, or falls back with a warning to the old shared names (what pre-fix checkpoints hold, and always loaded). **`max_seq_length` is read but not compared**, which is harmless for sinusoidal encoding but undocumented. **Reads aren't checked**: a short header leaves zeros and produces a misleading "architecture mismatch" message, and a truncated sidecar is up to the sub-layer's loader (TD-298/TD-345/TD-357 class). Prints to `std::cout` |

`EncoderDecoderModel::save_model()`/`load_model()` call these with `<checkpoint>.encoder`.
`ModelSerializer` (SafeTensors) bypasses them and goes through the accessors.

### Accessors

`get_num_layers()`, `get_encoder_block(i)` (throws `out_of_range`; used for LoRA and SafeTensors),
`get_token_embedding()`, `get_final_norm()` and `get_embedding_dim()` (returns `d_model`).
`print_config()` writes to `std::cout` and has no production caller.

### GPU (`ADAI_ENABLE_GPU`)

| Method | Behaviour |
|---|---|
| `gpu_upload_weights()` / `gpu_download_grads()` / `gpu_zero_grads()` | Blocks and final norm only. **The token embedding is deliberately excluded**: it runs on the host |
| `gpu_encode(ids)` | Embedding and positional encoding on the host, upload, blocks' `gpu_forward()` with **no mask**, final norm. No truncation (same as `encode_with_mask()`). Called by `EncoderDecoderModel`'s GPU training step and both GPU generation paths |
| `gpu_backward(dout)` | Final norm and blocks on the device, then `Matrix::from_gpu()` and the CPU `token_embedding->backward()` |

The CPU training path passes an all-ones `seq × seq` mask to `encode_with_mask()`. That mask is
rebuilt by `EncoderDecoderModel` on every call (O(n²) allocation for a no-op). The GPU path skips the
mask, so the two are equivalent.

---

## 3. Who calls what

| Caller | Uses |
|---|---|
| `EncoderDecoderModel` | Constructor; `encode_with_mask()`, `backward()`, `zero_grad()`, `register_parameters_with_optimizer()`, `save_weights()`/`load_weights()`; `get_encoder_block()` for LoRA; the GPU methods |
| `ModelSerializer` | `get_token_embedding()`, `get_encoder_block()`, `get_final_norm()` for SafeTensors save/load |
| `DocumentStore` | `encode(text)` when adding documents and searching ([DocumentStore.md](DocumentStore.md)) |
| `ChatbotAPIServer.cpp` | Builds the RAG encoder and calls `load_tokenizer_vocab()`; never loads weights (TD-247) |
| `ChatbotAPI::enable_pipeline_inference()` (removed October 28, 2026) | `load_tokenizer_vocab()` on the chat encoder |
| `PipelineInferenceEngine` (retired October 28, 2026) | `encode(text)` in its encoder stage |
| `IntegratedInferenceEngine` (retired October 27, 2026) | `encode_with_mask()` with its own tokenizer |

`LeJEPAEncoder` (the world model) is a separate class with a similar shape; it doesn't use
`LLMEncoder`.

---

## 4. Tests

`llmencoder_test.cpp` has 37 tests:
- construction with default, custom and minimal configurations;
- tokenizer load/build, including a missing file and an empty corpus;
- encode shapes, long-text truncation, empty strings, single words, repeats, Unicode and special characters;
- `encode_with_mask()`;
- sentence embeddings (consistency, distinctness, trainable variant);
- `set_requires_grad()`, `set_learning_rate()` and its propagation to blocks;
- `zero_grad()`, backward passes, `backward_sentence_embedding()`, a training workflow;
- save/load, including a missing file, a mismatched architecture and an output-preserving round trip;
- `get_embedding_dim()`, `print_config()`, optimizer registration and workflow.

**Not covered:**
- that `encode()` keeps EOS after truncation (`EncodeLongTextTruncates` checks only the row count);
- `encode()` vs. the model's own tokenization (special tokens);
- over-length input to `encode_with_mask()`/`gpu_encode()`;
- truncated weight files.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `encode()` adds BOS/EOS, but the model trains its encoder without special tokens; pipeline inference uses `encode()` on the trained chat encoder (§2) | Pipeline-mode answers come from a shifted, out-of-distribution encoder input | **Verified** | [TD-364](../../guides/TECHNICAL_DEBT.md#td-364-llmencoderencode-input-doesnt-match-the-models-encoder-input) |
| No truncation for `encode_with_mask()`/`gpu_encode()`, and none in `ChatbotAPI`/`EncoderDecoderModel` serving (§2) | Inputs over `max_seq_length` get position-less tokens and unbounded O(n²) attention | **Verified** | [TD-365](../../guides/TECHNICAL_DEBT.md#td-365-encoder-input-is-never-truncated-to-max_seq_length-when-serving) |
| `encode()` truncation drops the EOS it just added (§2) | Long RAG documents and pipeline inputs lose their end marker | **Verified** | [TD-364](../../guides/TECHNICAL_DEBT.md#td-364-llmencoderencode-input-doesnt-match-the-models-encoder-input) |
| No `update_weights()`; `EncoderDecoderModel`'s "doesn't have this method" comment is stale for `set_learning_rate()` (§2) | No-optimizer training never updates the encoder | By inspection (overlaps TD-360) | [TD-360](../../guides/TECHNICAL_DEBT.md#td-360-encoderdecodermodel-no-optimizer-path-never-updates-the-encoder) (corrected) |
| `requires_grad` defaults to true and is never cleared for serving instances; `cached_encoder_outputs` is never read; `cached_embeddings`/`cached_pos_encoded` unused (§2) | Wasted memory per inference call; dead state | By inspection | [TD-366](../../guides/TECHNICAL_DEBT.md#td-366-serving-encoders-cache-activations-for-backward-dead-state-and-duplicate-api) |
| `std::cout` in the constructor, `encode()` (every call), save/load, tokenizer methods and `print_config()` (§2) | Console noise from library code in serving paths | By inspection | [TD-367](../../guides/TECHNICAL_DEBT.md#td-367-llmencoder-console-output-and-unchecked-weight-io) |
| Token embedding and final norm saved under the decoder's file names, so every `EncoderDecoderModel` checkpoint lost them (logits changed by up to 1.54 on a round trip) | Every load ran the encoder on the decoder's embedding and norm | **Fixed October 18, 2026** (was verified) | [TD-538](../../guides/TECHNICAL_DEBT.md#td-538-encoder-token-embedding-and-final-norm-are-overwritten-in-every-checkpoint) |
| `load_weights()` unchecked reads; `max_seq_length` not compared; `save_weights()` write errors unchecked (§2) | Truncated header gives a misleading error; silent partial saves | By inspection | [TD-367](../../guides/TECHNICAL_DEBT.md#td-367-llmencoder-console-output-and-unchecked-weight-io) |
| `get_sentence_embedding()` and `_trainable()` are duplicates with no production caller (§2) | Dead, duplicated API | By inspection | [TD-366](../../guides/TECHNICAL_DEBT.md#td-366-serving-encoders-cache-activations-for-backward-dead-state-and-duplicate-api) |
| RAG encoder never loads trained weights | Already tracked | — | TD-247 |

---

## 6. Merge notes

This file replaces `docs/development/api/transformer/encoder.md` (1,515 lines). Where they disagree,
this file wins:

- **Architecture.** The old doc gave the block as `x1 = LayerNorm(x + MultiHeadAttention(x))`
  (Post-LN) and said layer normalization is *"applied after residual connections"*. The code is
  Pre-LN, with a final norm (§1).
- **Integration.** About half the old doc was a design essay on integrating with a
  `NeuralNetwork` class (classifier heads, batch processing, "feature extraction modes") with
  sentiment- and intent-classification examples. `NeuralNetwork` doesn't exist in `src/`; that
  material was dropped.
- **Class listing.** It restated the interfaces of `Matrix`, `Activation`, `LayerNorm`,
  `PositionalEncoding`, `TokenEmbedding`, `MultiHeadAttention`, `FeedForward` and `EncoderBlock`.
  Those have, or will have, their own references.
- **Optimization advice.** Suggestions such as "use Eigen" and "add KV caching" were dropped as
  out of date (the encoder output is already cached per generation; a GPU path exists).

**Kept from the old doc:**
- the component hierarchy and information flow (§1);
- the save/load layout (§2);
- the semantic-similarity use case, which is what `DocumentStore` does.

---

## Addendum (October 19, 2026): device embedding (TD-547)

`LLMEncoder` keeps a device copy of its positional table (`gpu_positional_table()`, uploaded once), and `gpu_encode()` use
`TokenEmbedding::gpu_forward()` instead of a host lookup plus upload. The GPU backward calls `TokenEmbedding::gpu_backward()` instead of
downloading the gradient for a host backward. `gpu_upload_weights()`, `gpu_download_weights()`, `gpu_download_grads()`, `gpu_zero_grads()` and
`gpu_collect_parameters()` now include the embedding.

## Addendum (October 26, 2026): model-matching `encode()`, tail truncation (TD-364, TD-365)

- **`encode()`.** `encode(text, add_special_tokens = false)` defaults to the model's encoder input. With special tokens, BOS and EOS
  survive truncation.
- **Truncation.** `keep_last_positions(ids)` is the one truncation rule: the last `max_seq_length` ids, as `ChatbotTrainer` truncates
  training inputs. `EncoderDecoderModel`, `RLHFTrainer` and `IntegratedInferenceEngine` apply it before encoding.
- **Checks.** `encode_with_mask()` and `gpu_encode()` throw `invalid_argument` past `max_seq_length`.

Tests: `EncodeMatchesTheModelsEncoderInput`, `OverLongInputKeepsTheTail`, `EncodeWithMaskRejectsOverLongInput`.

## Addendum (October 26, 2026): checked checkpoint I/O (TD-423)

`save_weights()`/`load_weights()`'s header (TD-367's read/write item) use `adai::binio::Reader`/`Writer` (`src/BinaryIO.{hpp,cpp}`).
- **Loading.** A file that is short, empty or longer than this layer's shapes throws `std::runtime_error`. The whole file is read
  into temporaries first, so a failed load leaves the layer unchanged.
- **Saving.** A failed write or flush (a full disk) throws instead of leaving a short file.

Tests: `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`.

## Addendum (October 26, 2026): one update per step (TD-297, TD-346, TD-360)

New `update_weights()` (TD-360): one optimizer step when registered (TD-297), else plain SGD in the embedding, every block and the final norm. Test: `EncoderDecoderModelTest.OneUpdateIsOneStepWithOrWithoutAnOptimizer`.

## Addendum (October 26, 2026): no unread activation copies (TD-366)

`cached_encoder_outputs`, `cached_embeddings` and `cached_pos_encoded` are removed (`backward()` uses each block's own caches). Serving
models run in inference mode: `chatbot_api_server`'s model and draft, `chatbot_gui`'s model, and the RAG encoder.
`get_sentence_embedding_trainable()` calls `get_sentence_embedding()`. Test: `LLMEncoderTD366Test.SentenceEmbeddingFunctionsAgreeInEitherMode`.

## Addendum (October 28, 2026): engine modes retired

`PipelineInferenceEngine` (`--pipeline-inference`) and `IntegratedInferenceEngine` (`--integrated-inference`) were removed in October 2026 (TD-474/TD-475, TD-417–TD-420); mentions of them above describe the code as it was.
