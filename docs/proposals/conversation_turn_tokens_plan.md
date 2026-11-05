# Conversation Turn Tokens

**Status:** Proposed (November 4, 2026). Owner decision on TD-295: real turn tokens rather than text role labels; design first,
for review before any code changes.

**Tracks:** [TD-295](../development/guides/TECHNICAL_DEBT.md#td-295-conversation-role-labels-are-spoofable-and-roles-arent-escaped)
(part 2: typed role labels impersonate turns). Closely tied to
[TD-224](../development/guides/TECHNICAL_DEBT.md#td-224-tokenizer-normalization-is-lossy-no-capitals-newlines-or-tabs) (lossy
tokenizer normalization), TD-265 (prompt format vs training data, closed as moot by TD-260 with the training-data question left
open), TD-505 (special-token ids), TD-223/TD-225 (vocabulary id assignment) and the TD-358 retraining.

## 1. Problem

A chat session's prompt is plain text built by `ConversationContext::format_for_model()`:

```text
User: hello
Assistant: hi there
User: what's next?
```

A user can type `Assistant: ...` inside a message, and nothing distinguishes it from a real turn. It is worse than it looks:
`BPETokenizer::pre_tokenize()` lowercases everything and collapses every whitespace run to one space (TD-224), so the model sees
`user: hello assistant: hi there user: what's next?` either way. Line breaks don't separate turns, and the labels are ordinary
words the model also meets in content. No escaping of the content can fix this while the labels are text.

The usual fix is reserved tokens that text can't produce: each turn starts with a special token (`<user>`, `<assistant>`,
`<system>`) that enters the prompt only as a token id written by the formatter, never through encoding user text.

## 2. How things work today

| Area | Today |
|---|---|
| Special tokens | `<pad>`/`<unk>`/`<bos>`/`<eos>` at ids 0–3 (`SpecialTokens.hpp`). `BPETokenizer`'s constructor seeds them; `load_vocab()` marks ids 0–3 as special. Hard-coded uses: `TextGenerator` config, `/chat/batch` padding, `BatchProcessor` (TD-505). |
| Vocabulary ids | `build_vocab()` starts ordinary tokens at id 4 (`current_id = 4`), so every existing vocabulary uses 4 onward for characters and merges. |
| Encoding | `encode(text, add_special_tokens)` runs `pre_tokenize()` (lowercase, regex split: `<user>` becomes `<`, `user`, `>`) then BPE. Special-token text is never recognized: typing `<bos>` gives ordinary ids. |
| Decoding | `decode(ids, skip_special_tokens)` drops tokens in `special_tokens`. |
| Prompts | Sessions: `format_for_model()` lines, the encoder's input (kept to its last `max_seq_length` tokens, TD-365), budgeted in real tokens since TD-294. Single-turn `/chat`: the raw message. RAG: `RAGInference::buildPrompt()` text with retrieved documents, the conversation and the question. |
| Training | `incremental_trainer` trains single-turn `input` → `response` pairs; the encoder input is the raw `input` text (`preprocess_data()`), the target the response plus `<eos>`. No training data uses the `User:`/`Assistant:` format, so session prompts are already out of distribution (the TD-265 question). |

## 3. Design

### D1. Vocabulary: three reserved turn tokens

Reserve `<system>` = 4, `<user>` = 5, `<assistant>` = 6 in the special-token standard, and start ordinary tokens at 7 in
`build_vocab()`. `load_vocab()` marks them special when present. `BPETokenizer::has_turn_tokens()` reports whether the loaded
vocabulary has all three at those ids.

Existing vocabularies have ordinary tokens at 4–6, so they can't simply gain them. They keep working unchanged (see D7); a model
gets turn tokens only with a rebuilt vocabulary, which means retraining.

*Alternative:* append the three tokens after the last id of an existing vocabulary. That avoids renumbering but still changes the
vocabulary size, so the embedding and output layers change and the checkpoint needs retraining anyway, and the ids differ per
vocabulary. Fixed ids are simpler.

### D2. Turn tokens enter only as ids

The formatter builds the prompt as token ids, not text: for each message, the role's turn id followed by
`encode(content, false)`. `encode()` itself never produces turn ids from text, so nothing a user types can create one.

This needs an id-based path from the session to the model: `ConversationContext::encode_for_model(const BPETokenizer&)` returning
ids, and `ChatbotAPI::generate_response()` taking ids (the model's generation entry points already start from ids after their
own `encode()`; they gain overloads that take the ids directly). RAG's `buildPrompt()` follows the same shape, ids with a
`<system>` turn for the retrieved context.

*Alternative (simpler, weaker):* keep text prompts, teach `encode()` to recognize the exact strings `<user>` etc. before
`pre_tokenize()`, and strip those strings from user text, imported conversations and RAG documents before formatting. This keeps
every string interface, but safety then depends on stripping everywhere text enters a prompt, and the tokenizer's own
normalization (lowercasing) must be applied in the stripping too. The id-based design has no such list to keep complete.

### D3. Prompt format

With turn tokens:

```text
[<system> system message]  <user> message  <assistant> reply  ...  <user> latest message  <assistant>
```

The trailing `<assistant>` cues the reply. The single-turn `/chat` uses the same shape (`<user> message <assistant>`), so a model
trained on it behaves the same with and without a session. TD-294's token counter counts a message as its turn id plus its
content's ids.

Without turn tokens (an existing vocabulary), the current text format stays, so nothing changes for deployed models.

### D4. Training uses the same format

`ChatbotTrainer::preprocess_data()` encodes each training input as `<user>` + `encode(input, false)` + `<assistant>` when the
vocabulary has turn tokens; targets are unchanged (the response and `<eos>`). This is what makes the model learn the tokens, and
it matches the single-turn `/chat` prompt exactly. `kPreprocessFormatVersion` is bumped so tokenized caches rebuild.

Multi-turn training data (a JSONL `history` field of earlier turns, encoded the same way) is a later extension; the JSONL reader
already ignores unknown fields.

### D5. Generation never emits turn tokens

The decoder shouldn't produce `<user>`/`<system>`/`<assistant>`. Mask them like unknown ids (`mask_unknown_ids()` already exists
on every generation path), or treat a generated `<user>` as a stop token. Masking is simpler and needs no new stop logic.

### D6. Conversation files and imports

Serialized conversations keep text content and roles (they already are, `ConversationFormat`); turn ids are only built when a
prompt is formatted, so saved files are unaffected and old files load into new sessions. The role-name corruption bug (TD-295
part 1: a role containing `|` or a newline corrupts the line; a regular message with role `SYSTEM` reloads as the system message)
is fixed independently by validating role names.

### D7. Compatibility and rollout

- Detection is by vocabulary: `has_turn_tokens()` true → D2–D5; false → today's behaviour exactly. No configuration switch.
- A vocabulary built after this change has the tokens; its model must be trained with them. Building the new vocabulary and
  retraining belongs with the TD-358 retraining (every GPU-trained checkpoint needs it anyway).
- TD-224 is decided for letter case (November 4, 2026: keep it, TD-560–TD-563), which needs the same vocabulary rebuild and
  retrain. Doing the turn tokens, case (and any whitespace change) in one vocabulary version avoids several retrains.
- MNS architecture records are unaffected (vocabulary size is not an architecture field).

## 4. Decisions for the owner

1. **Ids:** fixed 4–6 (D1, recommended) or appended after an existing vocabulary.
2. **Mechanism:** id-based prompts (D2, recommended) or recognized strings plus stripping (D2's alternative).
3. **Reply cue:** end the prompt with `<assistant>` (recommended), or not.
4. **Generation:** mask turn tokens (D5, recommended) or stop at `<user>`.
5. **Timing:** with the TD-224 decision and the TD-358 retraining (recommended), or on its own.
6. **Multi-turn training data:** now, or later (recommended later).

## 5. Phases

| Chunk | Work | Depends on |
|---|---|---|
| TT-1 | Special-token standard (ids 4–6), `build_vocab()` from 7, `load_vocab()` marks them, `has_turn_tokens()`, `vocab_builder` output; tests for both vocabulary kinds | — |
| TT-2 | `ConversationContext::encode_for_model()`; id-based `generate_response()` and model overloads (CPU and GPU paths, every strategy); `/chat` and sessions use it; TD-294 counter on ids | TT-1 |
| TT-3 | RAG prompt as ids, with a `<system>` turn for retrieved context | TT-2 |
| TT-4 | Training: wrap inputs (D4), `kPreprocessFormatVersion` bump, cache key | TT-1 |
| TT-5 | Generation masks turn tokens on every path (CPU, GPU, speculative, batched) | TT-1 |
| TT-6 | Rebuild the vocabulary and retrain (with TD-358, possibly TD-224); operator docs | TT-1–TT-5 |

## 6. Tests

- Tokenizer: new vocabularies have the three tokens at 4–6 and ordinary tokens from 7; old vocabulary files load with
  `has_turn_tokens()` false; `encode()` of text containing `<user>` never yields id 5.
- Prompt: a message containing `Assistant: ...` or `<assistant>` produces exactly one assistant id per real turn; the old text
  format is byte-for-byte unchanged without turn tokens.
- Training: tokenized inputs start with `<user>` and end with `<assistant>` only with turn tokens.
- Generation: no path ever emits ids 4–6.
- End to end: a small model trained with turn tokens, served by `chatbot_api_server`, answers a session whose user typed a fake
  `Assistant:` turn without treating it as one (compare its prompt ids).
