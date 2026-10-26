# `Decoder` (`LLMDecoder`) — Source File Reference

- **Files:** [`src/Decoder.hpp`](../../../../src/Decoder.hpp), [`src/Decoder.cpp`](../../../../src/Decoder.cpp)
- **Class:** `LLMDecoder` (global namespace)
- **Built into:** `adai_models` (owned by `EncoderDecoderModel`)
- **Status tag:** `@adai-status: beta` (capped by TD-050), `@adai-version: 0.14.0`, `@adai-reviewed: 2026-09-18`
- **Tests:** [`tests/decoder_test.cpp`](../../../../tests/decoder_test.cpp) → `decoderTests` (59 tests)
- **Origin:** original decoder stack; TD-050 (GPU incremental decode), TD-181 (sparse world-model injection knob), TD-186 (world-model/memory threading, hippocampal allocation)
- **Last traced against the code:** 2026-10-11

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class
> (linked against the debug libraries) in a scratch program.

---

## 1. What this file is

The **decoder stack** of the encoder-decoder chatbot:

```text
token ids → TokenEmbedding → + PositionalEncoding → DecoderBlock × num_layers → final LayerNorm → [seq, d_model]
```

`EncoderDecoderModel` owns one `LLMDecoder`. It feeds it the encoder output for cross-attention,
then sends the decoder output to `LanguageModelHead` for logits. Per-layer behaviour is in
[DecoderBlock.md](DecoderBlock.md). This class adds:
- the embedding and positional encoding;
- the causal mask;
- layer iteration;
- the final normalization;
- the KV-cache and GPU entry points;
- persistence.

### Why it matters

Every training step and every generated token goes through this stack, through one of three
forward paths: full, KV-cached, or GPU. They're supposed to compute the same thing.

---

## 2. Construction

`LLMDecoder(vocab_size, d_model = 512, num_layers = 6, num_heads = 8, d_ff = 2048, max_seq_length = 512, world_model_inject_every_n_layers = 0)`:

- **Throws** if `world_model_inject_every_n_layers < 0`. Nothing else is validated.
- **Builds** `TokenEmbedding(vocab, d_model)`, `PositionalEncoding(max_seq_length, d_model)`,
  `num_layers` `DecoderBlock`s and the final `LayerNorm`.
- **TD-181/TD-186:** block `i` gets **both** gated paths (world model and hippocampal) when
  `N > 0 && i % N == 0`. The default `0` disables injection everywhere. Every block is also given
  `dropout = 0.1`, which is never applied (TD-344).

Accessors: `get_d_model()`, `get_vocab_size()`, `get_num_layers()`, `get_max_seq_length()`,
`get_world_model_inject_every_n_layers()`, `get_token_embedding()`,
`get_decoder_block(i)` (bounds-checked), `get_final_norm()`, and `get_last_output()`. Copy and move
are deleted.

---

## 3. Forward paths

| Method | Behaviour |
|---|---|
| `forward(token_ids)` | `forward_with_encoder(ids, Matrix(0, 0))`, i.e. decoder-only |
| `forward_with_encoder(ids, encoder_output, world_model_output, memory, …hippocampal knobs)` | Embed, add positional encoding, build a `[seq, seq]` causal mask, run every block, apply the final norm. With an empty `encoder_output` each block gets a zero `Matrix(1, d_model)` dummy (TD-303). `world_model_output` and `memory` go to **every** block; only blocks with the gated path allocated use them |
| `forward_with_mask(ids, mask, encoder_output)` | Same with a caller-supplied mask and **no** world-model/memory arguments. No production caller |
| `forward_with_cache(ids, DecoderKVCache&, encoder_output, use_cache)` | Embeds only the new tokens and adds positional encoding **inline** at offset `kv_cache.current_length()`; the formula matches `PositionalEncoding`'s. Builds a `[new, total]` causal mask and calls each block's `forward_with_cache()` with its own self/cross cache. No world model or memory (TD-342) |

All forwards cache what `backward()` needs. `cached_encoder_output` is copied on every forward but
never read.

> **Cached and full forward diverge past `max_seq_length` (verified).** Within range, incremental
> decoding matches the full forward (|Δ| = 0.000002 on the last token for length 4,
> `max_seq_length = 4`). Past it they split:
> - `PositionalEncoding::forward()` leaves rows beyond `max_len` **without** positional encoding
>   (with a stderr warning);
> - `forward_with_cache()`'s inline formula keeps encoding any position.
>
> At length 6 the last token differed by **13.6** (L1 over `d_model = 16`). So the same prompt plus
> continuation gives different hidden states depending on whether generation used the KV cache
> (greedy/sampling) or a full recompute (beam), once a sequence exceeds the model's
> `max_seq_length`. Neither path has positions the model was trained on. Generation should cap
> sequences at `max_seq_length`.

> **Hippocampal memory is mutated once per injected layer (by inspection).** The same `memory`
> pointer reaches every block with the gated path, and each such block decays and accumulates
> coverage and associations ([DecoderBlock.md](DecoderBlock.md) §3). With
> `WORLD_MODEL_INJECT_EVERY_N_LAYERS = 1` and 6 layers, one forward call applies **6** decay +
> accumulate rounds. That compounds TD-343's per-call over-counting.

---

## 4. Backward and optimization

- **`backward(grad_output, grad_encoder_output&)`** runs through the final norm and then the
  blocks in reverse. It **sums** each block's encoder-side gradient into `grad_encoder_output`
  (every block's cross-attention reads the same encoder output), then calls
  `token_embedding->backward()`. The positional encoding has no parameters. It returns immediately
  when `requires_grad` is false.
- **`register_parameters_with_optimizer(opt)`** registers the token embedding, every block (minus
  the gates, TD-341) and the final norm.
- **`zero_grad()`** zeroes the embedding, the blocks and the final norm.
- **`set_learning_rate(lr)`** propagates to the embedding, the blocks and the final norm (TD-093
  fix).
- **`set_training(mode)`** sets `requires_grad`, which gates only `backward()`/`update_weights()`.
  Its "propagate to all components" loop has an empty body.

> **`update_weights(float lr)` never updates the final norm and ignores `lr` (verified).** It
> calls `token_embedding->update_weights()` and each block's `update_weights()`. The final norm's
> update is commented out ("For now, we'll skip if the method doesn't exist"), although
> `LayerNorm::update_weights()` exists. Twenty steps through this path moved the token embeddings
> (L1 0.22) while the final norm's gamma and beta stayed **exactly unchanged**. This is the path
> `EncoderDecoderModel::update_weights()`/`train_step()` use. The shipped `ChatbotTrainer` calls
> `optimizer->step()`, which does update the registered final norm, so it's unaffected. With an
> optimizer registered, adding the missing call would also add another over-step (TD-297).

---

## 5. Persistence

`save_weights(path)` writes a header to `path`: vocab, `d_model`, layers, heads, `d_ff`,
`max_seq_length`. The weights go to sibling files named from `path` minus its last extension:
- `<base>_token_emb.bin`;
- `<base>_decoder_block_<i>.bin` (plus each block's own sibling files, [DecoderBlock.md](DecoderBlock.md) §5);
- `<base>_final_norm.bin`.

`EncoderDecoderModel` calls it with `<model>.decoder`, so the base is the model path.

`load_weights(path)`:
- reads the header **without checking the stream**;
- throws on a vocab, `d_model`, layer, head or `d_ff` mismatch (`max_seq_length` isn't compared);
- loads each component.

This is the legacy multi-file format; SafeTensors (`ModelSerializer`) is the main checkpoint path.

---

## 6. GPU path (`ADAI_ENABLE_GPU`)

| Method | Behaviour |
|---|---|
| `gpu_upload_weights()` / `gpu_download_grads()` / `gpu_zero_grads()` | Blocks + final norm. The **token embedding stays on the CPU** |
| `gpu_decode(ids, encoder_out)` | Embedding + positional encoding on the CPU, upload, a `[tgt, tgt]` causal mask on the GPU, each block's `gpu_forward()`, final norm. Used for the training forward pass |
| `gpu_decode_step(new_ids, GPUDecoderKVCache&, encoder_out)` | TD-050's incremental decode, the same algorithm as `forward_with_cache()` (inline positional encoding, so the `max_seq_length` divergence above applies too) |
| `gpu_backward(dout)` | Final norm and blocks in reverse, summing encoder gradients via `std::optional`. Downloads the input gradient for the CPU embedding backward. `*grad_encoder_sum` is dereferenced unconditionally, so `num_layers = 0` would be undefined behaviour |

No GPU path carries world-model or memory inputs (TD-342).

---

## 7. Tests

`decoder_test.cpp` has 59 `DecoderTest` cases:
- construction and shapes;
- decoder-only vs encoder-decoder forwards;
- causal masking;
- backward and encoder-gradient summation;
- save/load;
- the KV-cache path;
- injection-knob allocation.

**Not covered:**
- `update_weights()`'s effect on the final norm;
- cached vs full divergence beyond `max_seq_length`;
- memory mutation per injected layer.

---

## 8. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `update_weights(lr)` skips the final norm and ignores `lr` (§4) | Non-optimizer training (`EncoderDecoderModel::train_step()`/`update_weights()`) never trains the final norm | **Verified** | [TD-346](../../guides/TECHNICAL_DEBT.md#td-346-llmdecoderupdate_weights-never-updates-the-final-norm) |
| Cached (and GPU incremental) positional encoding keeps encoding past `max_seq_length` while the full forward stops (§3, §6) | Divergent hidden states (Δ 13.6 in a small test) between generation strategies on over-length sequences | **Verified** | [TD-347](../../guides/TECHNICAL_DEBT.md#td-347-cached-and-full-decoder-forwards-diverge-past-max_seq_length) |
| Hippocampal memory mutated once per injected layer per forward (§3) | Coverage/association updates multiplied by the number of injected layers | By inspection (compounds TD-343) | [TD-343](../../guides/TECHNICAL_DEBT.md#td-343-hippocampal-memory-is-updated-on-every-forward-pass) (amended) |
| Dead code and hygiene: `set_training()`'s empty loop, unused `cached_encoder_output` copy, unused `forward_with_mask()`, unchecked `load_weights()` header reads, unvalidated constructor arguments, `gpu_backward()` UB with zero layers (§2–6) | Wasted copies, misleading code, silent garbage on truncated files | By inspection | [TD-348](../../guides/TECHNICAL_DEBT.md#td-348-llmdecoder-dead-code-and-robustness-gaps) |
| Decoder-only dummy cross-attention | Already tracked | — | TD-303 |
| Gated paths absent from cached/GPU paths | Already tracked | — | TD-342 |

---

## 9. Merge notes

This file replaces `docs/development/api/transformer/decoder.md` (about 1,000 lines). Where they
disagree, this file wins:

- **`update_weights()`.** The old doc said it updates "final layer norm parameters (gamma, beta)".
  It doesn't (§4).
- **Persistence.** The old doc said save/load persists only the config and "does NOT save token
  embeddings, decoder block weights, layer norm parameters". That's stale: every component is
  saved to sibling files (§5).
- **TD features.** It had no TD-181/186 injection knob, no `gpu_decode_step()`, and none of the
  world-model/memory parameters.

**Kept from the old doc:** the architecture stack, the decoder-only vs encoder-decoder modes, and
the causal mask and KV-cache usage descriptions.

**Dropped:** generic usage examples, configuration guidelines and design-decision essays.

---

## Addendum (October 19, 2026): device embedding (TD-547)

`LLMDecoder` keeps a device copy of its positional table (`gpu_positional_table()`, uploaded once), and `gpu_decode()` and `gpu_decode_step()` use
`TokenEmbedding::gpu_forward()` instead of a host lookup plus upload. The GPU backward calls `TokenEmbedding::gpu_backward()` instead of
downloading the gradient for a host backward. `gpu_upload_weights()`, `gpu_download_weights()`, `gpu_download_grads()`, `gpu_zero_grads()` and
`gpu_collect_parameters()` now include the embedding.

`gpu_decode_step()` used to compute its positional row inline, including past `max_seq_length`. It now uses the table rows, so like the full
forward it adds none past the table (TD-347 on the GPU path; one of TD-478's formula copies removed).

---

## Addendum (October 20, 2026): gated inputs in cached and GPU decoding (TD-342)

`forward_with_cache()`, `gpu_decode()` and `gpu_decode_step()` take the world-model output, memory and hippocampal settings and pass
them to every block, as `forward_with_encoder()` does. Each block applies them only if its gated path exists.

---

## Addendum (October 21, 2026): `last_hippocampal_step_weights()` (TD-343)

Returns `[1, slots]`: the newest token's row of each block's `last_hippocampal_attention()`, averaged over the blocks whose hippocampal
path ran. That makes one memory update per decode step, not one per injected layer.

## Addendum (October 26, 2026): cached positional rows (TD-347)

`forward_with_cache()` adds positional rows from `PositionalEncoding`'s table, with none at or past `max_seq_length`, matching the full
forward and the GPU cached path. Test: `CachedMatchesFullPastMaxSeqLength`.
