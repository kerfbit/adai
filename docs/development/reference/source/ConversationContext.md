# `ConversationContext` — Source File Reference

- **Files:** [`src/ConversationContext.hpp`](../../../../src/ConversationContext.hpp), [`src/ConversationContext.cpp`](../../../../src/ConversationContext.cpp)
- **Library:** `adai_nlp`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.2.0`, `@adai-reviewed: 2026-09-13`
- **Tests:** [`tests/conversationcontext_test.cpp`](../../../../tests/conversationcontext_test.cpp) (70 tests)
- **Last traced against the code:** 2026-10-09

> Code-traced reference. Every behaviour flagged in §5–§7 was confirmed by running the real class
> (a scratch program compiled against `ConversationContext.cpp`). This page supersedes the older
> `docs/development/api/nlp/conversation-context.md` (now deleted); its accurate content is merged
> here, see [§10](#10-merge-notes).

---

## 1. What this file is

A self-contained, dependency-free **multi-turn conversation history**: an ordered list of
role-tagged messages plus an optional system message, with message-count and estimated-token limits
enforced by evicting the oldest messages first. It formats the history into a model prompt, and
serializes it for persistence or transfer.

```text
add_user/assistant/message ──► deque<Message> ──► truncate_to_limits()  (oldest first)
set_system_message         ──► optional<Message> (exempt from eviction by default)
format_for_model()         ──► "System: …\nUser: …\nAssistant: …\n"
serialize()/deserialize()  ──► line format (TD-053 HTTP export/import, save/load files)
```

### Why it matters

It's the server-side memory of every multi-turn chat. `chatbot_api_server` keeps one per session,
and `/chat/session/export`/`import` (the CLI's `/save`/`/load`) are just its `serialize()` and
`deserialize()`. It decides which earlier turns the model still sees, and since import accepts its
format from clients, its parser is part of the server's attack surface.

---

## 2. Where it's used

| Caller | How |
|---|---|
| `src/ChatbotAPI.hpp/.cpp` | `Session` holds `unique_ptr<ConversationContext>(10 messages, 2048 tokens)`. `handle_chat_session()` and batch sessions: `add_user_message`, `format_for_model()`, `add_assistant_message`. `/clear-session` → `clear()`; export → `serialize()`; import → `deserialize()`; reply → `get_message_count()` |
| `src/ChatbotGUI.cpp` | `ConversationContext(20, 480)`. Add, `format_with_special_tokens()` (see TD-265), `clear()`, `save_to_file()`/`load_from_file()` |
| `src/ChatbotTrainer.cpp` | Includes the header but doesn't use it |
| `src/ChatbotCLI.*` | Comments only (the CLI keeps no local history; see [ChatbotCLI.md](ChatbotCLI.md)) |

Because the CLI exports via `serialize()` and the GUI saves via `save_to_file()` (which writes
`serialize()`), their files are interchangeable.

---

## 3. Data model

```cpp
struct Message { std::string role; std::string content; int token_count; };

std::deque<Message>    messages;          // history; deque so the oldest is cheap to remove
std::optional<Message> system_message;    // TD-079: value type (was a raw owning pointer)
int  max_messages, max_tokens;            // 0 = unlimited
bool keep_system_message;                 // system message exempt from clear() and token eviction
int  total_tokens;                        // cached sum, system message included
```

Copy and move are defaulted and correct (TD-079 fixed a double-free from the old raw pointer).
**Not thread-safe**; `ChatbotAPI` must synchronize access (see TD-227).

Roles are free-form strings. `user`, `assistant` and `system` are conventional, and others (e.g.
`function`) are allowed: they're formatted as-is and counted as "Other" in statistics.

---

## 4. API

| Method | Behaviour |
|---|---|
| `ConversationContext(max_messages = 20, max_tokens = 2048, keep_system_message = true)` | Limits; 0 = unlimited |
| `add_user_message` / `add_assistant_message` / `add_message(role, …)` (`token_count` 0 = estimate) | Append, add to the total, then `truncate_to_limits()` |
| `set_system_message(content, token_count = 0)` | Replace the system message (adjusting the total); doesn't truncate |
| `format_for_model(include_system = true, sep = "\n")` | `System: …` then `Role: content` per message (role first letter capitalised) |
| `format_with_special_tokens(bos, eos, sep)` | `<bos> [SYSTEM] …<sep> [USER] …<sep> … <eos>`. These are literal strings, which `BPETokenizer` treats as ordinary text, not special tokens |
| `get_last_user_message()` / `get_last_assistant_message()` | Searches backwards; **throws** `runtime_error` if none |
| `get_messages()` (copy), `get_system_message()` (`""` if none), `get_total_tokens()`, `get_message_count()` (excludes system), `is_empty()` | Accessors |
| `clear()` | Drops all messages; keeps the system message only if `keep_system_message` (TD-125) |
| `clear_all()` | Drops everything, unconditionally |
| `truncate_to_limits()` | See §5 |
| `set_max_messages(n)` / `set_max_tokens(n)` | Change a limit and truncate immediately |
| `serialize()` / `deserialize(data)` | See §6 |
| `save_to_file(path)` / `load_from_file(path)` | Thin wrappers (TD-053); throw `runtime_error` if the file can't be opened |
| `get_statistics()` | Multi-line summary: counts, tokens, limits, per-role counts |
| `create_summarized(keep_recent = 5, summary = "")` | New context: same limits and system message, plus (if a summary is given and there are more than `keep_recent` messages) a `system`-role message `"[Summary of earlier conversation: …]"`, then the last `keep_recent` messages. The caller supplies the summary text |

`estimate_tokens(text)` = ⌈bytes ÷ 4⌉ + spaces ÷ 2, minimum 1. It counts **bytes**, so non-ASCII
text is overestimated, and it has no relationship to `BPETokenizer`'s real counts. Callers can pass
real counts instead. `update_token_count()` is never called (dead code).

---

## 5. Truncation

```text
while messages > max_messages:                     drop oldest
while total_tokens > max_tokens and messages remain:
    if only one message left and total ≤ 1.2 × max_tokens: stop   (keep at least one, with 20% slack)
    drop oldest
if !keep_system_message and still over budget:     drop the system message too (TD-125)
```

The system message's tokens count toward the budget but, by default, it's never evicted, so a large
system message shrinks the room for history.

> **A single long message wipes the whole conversation (verified).** The newest message isn't
> protected: if it alone exceeds 1.2 × `max_tokens`, the loop evicts **everything, including that
> message**. With `ChatbotAPI`'s 2048-token session budget, a 12,000-character user message left 0
> messages. The server then formats an empty prompt, and `generate_response("")` fails because
> `BPETokenizer::encode("")` throws, so the user gets an HTTP 500 and has lost their history.

> **The token budget isn't tied to the model.** `max_tokens` counts the byte-based estimate, not
> real tokens, and `ChatbotAPI`'s 2048 has no link to the model's `MAX_SEQ_LENGTH` (e.g. 512 or
> 1024). Over-long prompts don't crash: `PositionalEncoding` warns and leaves positions beyond its
> length unencoded, so quality quietly degrades. The GUI's 480 is hardcoded against a
> 512-token model (TD-266).

---

## 6. Serialization format

```text
MAX_MESSAGES:<int>
MAX_TOKENS:<int>
KEEP_SYSTEM:<0|1>
---
SYSTEM|<token_count>|<escaped content>         (if a system message exists)
<role>|<token_count>|<escaped content>         (one line per message, in order)
```

Content escapes `\` → `\\`, newline → `\n`, CR → `\r` (TD-071 fixed silent loss of multi-line
messages). Roles are written raw (a role containing `|` or a newline would corrupt the line).

`deserialize(data)` **first calls `clear_all()`**, then reads metadata lines until `---`, then
message lines. A `SYSTEM` role becomes the system message; anything else goes through
`add_message()` (so truncation applies as messages are added). Lines without two `|` are skipped.

> **A malformed import wipes the existing conversation (verified).** Because state is cleared before
> parsing, a bad line (e.g. a non-numeric token count, so `std::stoi` throws) leaves the context
> empty. `ChatbotAPI::handle_import_session()` reports "Failed to import conversation", but the
> session's previous history is already gone.

> **Imports override the server's limits and budget (verified).** The metadata lines **replace**
> the instance's limits, and per-message token counts are trusted as written:
> - `MAX_MESSAGES:0` / `MAX_TOKENS:0` make the session **unlimited**. 500 imported messages were all
>   kept (a 2.5 MB prompt), and the session stayed unlimited for later messages;
> - a token count of `1` on huge messages, or a **negative** count (e.g. `-100000`, making
>   `total_tokens` negative), defeats the token budget.
>
> Through `POST /chat/session/import`, any client can therefore give its session unbounded history,
> memory and prompt size, regardless of `ChatbotAPI`'s `Session(10, 2048)` limits.

Note too that `format_for_model()`'s `User:`/`Assistant:` labels are plain text, so message content
containing `\nAssistant: …` can impersonate a turn. `BPETokenizer` collapses the newlines into spaces
anyway (TD-224), which also blurs turn boundaries.

---

## 7. Persistence notes

`save_to_file()` truncates and writes without checking the stream afterwards (a failed write after
opening goes unreported) and isn't atomic. `load_from_file()` has the same clear-first exposure as
`deserialize()`.

---

## 8. Tests

`conversationcontext_test.cpp` (70) covers construction, adding and auto-estimating, both
formatters, retrieval (including throws), message and token truncation, system-message handling
(TD-125), save/load and serialize/deserialize round-trips including escaping (TD-071),
statistics, summarization, and copy/move (TD-079). Not covered: oversized single messages,
malformed imports, imported limits or token counts.

---

## 9. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| A single message over 1.2 × `max_tokens` evicts everything, itself included (§5) | Long user message → empty prompt → HTTP 500 and lost history | **Verified** | [TD-292](../../guides/TECHNICAL_DEBT.md#td-292-one-long-message-wipes-the-whole-conversation) |
| `deserialize()` clears before parsing (§6) | A malformed import wipes the live session | **Verified** | [TD-293](../../guides/TECHNICAL_DEBT.md#td-293-a-malformed-conversation-import-wipes-the-live-session) |
| Imported metadata replaces limits; imported token counts trusted, including 0/1/negative (§6) | Client-controlled unbounded session history/memory/prompt via `/chat/session/import` | **Verified** | [TD-291](../../guides/TECHNICAL_DEBT.md#td-291-conversation-imports-can-lift-session-limits-and-budgets) |
| Token estimate is byte-based and unrelated to the model's `max_seq_length` (§4–5) | Prompts can exceed the model's positions; budget only approximate | By inspection | [TD-294](../../guides/TECHNICAL_DEBT.md#td-294-conversation-token-budget-is-unrelated-to-the-models-sequence-length) |
| Role labels are plain text; raw roles in the file format (§6) | Turn impersonation; corrupt lines for odd roles | By inspection | [TD-295](../../guides/TECHNICAL_DEBT.md#td-295-conversation-role-labels-are-spoofable-and-roles-arent-escaped) |
| `save_to_file()` doesn't check writes and isn't atomic; `update_token_count()` unused; `ChatbotTrainer` includes the header for nothing (§4, §7) | Minor | By inspection | [TD-296](../../guides/TECHNICAL_DEBT.md#td-296-conversationcontext-hygiene) |
| Not thread-safe | Already tracked | — | TD-227 |
| `format_with_special_tokens()` strings aren't real special tokens; GUI vs server format | Already tracked | — | TD-265 |

---

## 10. Merge notes

This file absorbed `docs/development/api/nlp/conversation-context.md`.

**Corrected:**
- *Class structure* showed `Message* system_message`; it's been `std::optional<Message>` since TD-079
  (which fixed the double-free the raw pointer caused).
- *"Status: Complete and Production-Ready"* and *"Integration status: ✅ Ready to integrate"*: it's
  integrated (ChatbotAPI sessions, GUI) and has the defects in §9.
- *"Unit tests to write" / "Next steps: write comprehensive unit tests"*: 70 tests exist (§8).
- *File format* lacked the TD-071 escaping and the clear-first import semantics (§6).
- *"System message never truncated (configurable)"*: true by default, but with
  `keep_system_message = false` it can be evicted (TD-125).
- *Truncation*: the doc didn't mention the 1.2× single-message slack or that an oversized newest
  message is itself evicted (§5).
- *Token estimation*: "~4 characters per token" is actually bytes ÷ 4 + spaces ÷ 2 (§4).
- The *"Advanced Integration with Token Counting"* example called `format_with_special_tokens()`
  for an encoder-decoder model; those tags aren't real special tokens to `BPETokenizer`.

**Dropped:** the per-message memory estimates, the production deployment checklist and use-case
limit table (unverified guidance), the "Future Enhancements" list (generic), and the illustrative
`Chatbot`/`TokenAwareChatbot` classes, since `ChatbotAPI` is the real integration (§2).

**Kept and merged:** design rationale (deque, separate system message, estimation with optional exact
counts, automatic truncation), the API surface and examples, both output formats, the `get_statistics()`
format, summarization semantics, custom roles, and the trade-offs (FIFO eviction, estimation vs.
exact counting, not thread-safe).
