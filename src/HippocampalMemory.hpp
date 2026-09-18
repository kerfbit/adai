#pragma once

// @adai-status: experimental
// @adai-version: 0.5.0
// @adai-reviewed: 2026-09-18

#include <deque>
#include <string>
#include <utility>
#include "Matrix.hpp"

/**
 * HippocampalMemory (LeJEPA world-model plan, HM-1 / TD-179; TD-193 least-used eviction + swap)
 *
 * Fast episodic memory — the "hippocampal" half of the Complementary Learning Systems pairing
 * this plan's LeJEPAEncoder plays the "cortical" half of (see
 * docs/proposals/lejepa_world_model_gated_injection_plan.md's Background section). Structurally
 * *not* a transformer encoder like LeJEPAEncoder: a bounded, continuously-updated ring buffer of
 * (key, value) pairs, written to as generation happens and read from via the gated hippocampal
 * cross-attention path (Component 6, TD-180). No pretraining phase — starts empty and
 * accumulates content during actual use, matching the hippocampus's own one-shot encoding, in
 * contrast to the world model's slow, batch-pretrained consolidation.
 *
 * Design note — coverage storage differs from the plan's own illustrative Slot struct: the
 * plan's snippet embeds `coverage` as a per-Slot float inside `std::deque<Slot> slots`, but also
 * declares a public `std::vector<float>& coverage_vector()` accessor returning a *mutable
 * reference* — the two are incompatible as literally written (a `std::vector<float>&` cannot
 * reference data embedded inside deque elements of a different type). TD-180's own Component 6
 * pseudocode needs genuine live, indexable, mutable access
 * (`memory->coverage_vector()[i] += hm_attn.attention_weights[i]`), so this implementation keeps
 * coverage in a separate `coverage_` container kept in lockstep with `slots_` (same size, same
 * insertion/eviction order) instead of embedded per-slot — everything else matches the plan's
 * interface exactly. `coverage_` is a `std::deque<float>`, not a `std::vector`, so both
 * containers move together identically regardless of which index gets removed on eviction.
 *
 * TD-193 eviction policy — least-used, not FIFO: at capacity, `write()` evicts whichever
 * currently-stored slot has the *lowest* `coverage_` value (ties broken by whichever is oldest,
 * i.e. lowest index) rather than always the single oldest slot. `coverage_` already tracks how
 * much attention-level use a slot has received (TD-180's own repetition-penalty bookkeeping), so
 * reusing it as the eviction signal means a frequently-relevant memory survives regardless of
 * age, while a stale, never-attended one is the first to go — a real (if simple) salience signal,
 * superseding the plan's own "salience-gated writing is a documented future extension" framing
 * with a salience-gated *eviction* policy on the existing write-everything path instead. A slot
 * evicted this way is never simply discarded when a swap file is configured (see below) — it's
 * appended there first, so this is genuinely a superset of the old FIFO behavior (which is what
 * this policy degenerates to when nothing's coverage has ever been touched, since every slot
 * then ties at 0.0f and the oldest-first tiebreak wins).
 *
 * TD-193 swap file (true swap, reloadable) — when constructed with a non-empty `swap_filepath`,
 * every slot evicted by `write()` (or by `recall_from_swap()` itself making room) is appended to
 * that file rather than lost, and `recall_from_swap()` can bring the *most recently evicted* one
 * back into the live ring buffer (LIFO — matches how OS swap treats recently-paged-out memory as
 * the most likely to be needed again soon), with its coverage reset to 0.0f (treated as freshly
 * relevant again, the same starting point any newly write()'d slot gets). The swap file is a
 * flat, append-only log — a one-time `d_model` header, then fixed-size (key, value, coverage)
 * records — entirely separate from `save()`/`load()`'s own concern (the live ring buffer's
 * current contents): `swap_filepath` is supplied once at construction like `capacity`/`d_model`,
 * not part of the serialized session state those two methods persist. Passing an empty
 * `swap_filepath` (the default) disables swap entirely — evicted slots are simply discarded,
 * exactly matching this class's original (pre-TD-193) behavior, so every existing caller that
 * doesn't ask for swap is unaffected.
 *
 * TD-194 cross-reference layer — a `size() x size()` association matrix (`association_`), one
 * entry per pair of currently-stored slots, kept in lockstep with `slots_`/`coverage_` (grows a
 * new zero row+column on insert, drops the evicted slot's row+column on eviction, matching
 * `coverage_vector()`'s own "mutable reference, caller does the actual bookkeeping" convention —
 * `association_matrix()`/`decay_associations()` mirror `coverage_vector()`/`decay_coverage()`
 * exactly). `DecoderBlock`'s own gated hippocampal cross-attention (TD-180) is both the writer and
 * the reader: after each forward pass, any two slots that received meaningful attention weight in
 * the *same* query step get their association Hebbian-strengthened ("used close together"),
 * decaying the same self-bounding way `coverage_` already does; before the next attention
 * computation, a slot's own association with other *recently used* slots (their current
 * `coverage_`) is folded into `score_bias` — the same injection point the repetition penalty
 * already uses — through `tanh` (the same gating activation `gate_h` already blends the whole
 * hippocampal path with), so a memory strongly cross-referenced with something just used gets
 * pulled toward attention, bounded rather than unbounded. See `DecoderBlock::forward()`'s own
 * implementation comment for the exact formula. Persisted separately from `save()`/`load()`'s own
 * concern via `save_associations()`/`load_associations()` — a fresh (all-zero) association matrix
 * is otherwise assumed after `load()`, since a saved session's own slot count/order must already
 * match before a saved association matrix means anything.
 *
 * Naming note: the member is `slots_`, not `slots` as in the plan's own snippet — `slots` is a
 * Qt macro (expanding to `Q_SLOTS` unless `QT_NO_KEYWORDS` is defined) and collides in any
 * translation unit that also includes Qt headers, discovered when this class's header is pulled
 * transitively into ChatbotGUI.cpp via DecoderBlock.hpp (TD-180).
 */
class HippocampalMemory {
   private:
    struct Slot {
        Matrix key;    // [1, d_model], typically from LeJEPAEncoder::encode()
        Matrix value;  // [1, d_model] — may equal key, or a separately stored payload
    };

    std::deque<Slot> slots_;
    std::deque<float> coverage_;  // coverage_[i] corresponds to slots_[i]; see class doc above
    std::deque<std::deque<float>> association_;  // association_[i][j]: slots_[i]<->slots_[j]
                                                  // cross-reference strength; see TD-194 class doc
    int capacity;
    int d_model;
    std::string swap_filepath_;  // empty = swap disabled; see class doc's TD-193 section

    /** Evicts the least-used entry (lowest coverage; ties broken by oldest/lowest index) from
     *  the given containers in place, appending it to swap_filepath_ first if one is configured.
     *  Shared by insert_slot() (the live buffer, association non-null) and load()'s own
     *  excess-trim loop (a local pair of containers being reconciled down to this instance's
     *  capacity, association null — load() resets association_ separately, see its own doc). When
     *  association is non-null, the evicted slot's row and column are dropped from it too. */
    void evict_least_used(std::deque<Slot>& slots, std::deque<float>& coverage,
                          std::deque<std::deque<float>>* association) const;

    /** Evicts (see above) if already at capacity, then appends the new (key, value, coverage) to
     *  the live buffer. Shared by write() and recall_from_swap(). */
    void insert_slot(const Matrix& key, const Matrix& value, float coverage);

    /** Appends one evicted slot's (key, value, coverage) to swap_filepath_, writing the
     *  one-time d_model header first if the file doesn't already exist or is empty. */
    void append_to_swap(const Matrix& key, const Matrix& value, float coverage) const;

   public:
    /**
     * @param d_model Embedding dimension — must match whatever encoder's encode() output feeds
     *                write()'s own key/value arguments (typically LeJEPAEncoder::encode()).
     * @param capacity Ring-buffer bound; the least-used slot is evicted once a write() would
     *                 exceed this (TD-193 — see class doc's own eviction-policy section).
     * @param swap_filepath Optional; when non-empty, every evicted slot is appended here instead
     *                      of discarded, and recall_from_swap() can bring the most recently
     *                      evicted one back (TD-193). Empty (the default) disables swap.
     */
    explicit HippocampalMemory(int d_model, int capacity = 512,
                               std::string swap_filepath = std::string());

    /**
     * Write a new episode. Least-used eviction at capacity (TD-193): the currently-stored slot
     * with the lowest coverage (and its corresponding coverage entry) is evicted — appended to
     * the swap file if one is configured, discarded otherwise — before the new one is appended,
     * so size() never exceeds capacity. The new slot's coverage starts at 0.0f.
     *
     * @param key   [1, d_model]
     * @param value [1, d_model]
     * @throws std::invalid_argument if either matrix isn't exactly [1, d_model]
     */
    void write(const Matrix& key, const Matrix& value);

    /**
     * Materialize all currently-stored keys/values as K/V matrices, oldest slot first — the
     * [num_slots, d_model] shape CrossAttention::forward() already expects. Both matrices are
     * [0, d_model] when empty rather than throwing, mirroring SIGReg's own empty-batch handling.
     */
    std::pair<Matrix, Matrix> read_all() const;

    /**
     * Mutable reference into this instance's own per-slot coverage — index i corresponds to the
     * i-th oldest currently-stored slot (same order read_all() returns). Invalidated by any
     * write() (eviction/append), recall_from_swap(), clear(), or load() call that changes size(),
     * the same rule as any other container-invalidating operation — do not hold onto this
     * reference across one.
     */
    std::deque<float>& coverage_vector();

    /** coverage[i] *= gamma for every currently-stored slot, called once per decode step. */
    void decay_coverage(float gamma);

    /**
     * Mutable reference into this instance's own cross-reference association matrix (TD-194) —
     * association[i][j] is the strength between the i-th and j-th currently-stored slots (same
     * order read_all()/coverage_vector() use), symmetric, diagonal unused. Same invalidation rule
     * as coverage_vector() — do not hold onto this reference across a write()/recall_from_swap()/
     * clear()/load() call.
     */
    std::deque<std::deque<float>>& association_matrix();

    /** association[i][j] *= gamma for every currently-stored pair, mirroring decay_coverage(). */
    void decay_associations(float gamma);

    int size() const {
        return static_cast<int>(slots_.size());
    }

    int get_capacity() const {
        return capacity;
    }

    int get_d_model() const {
        return d_model;
    }

    /** Whether this instance was constructed with a non-empty swap_filepath. */
    bool swap_enabled() const {
        return !swap_filepath_.empty();
    }

    /**
     * Bring the most recently evicted slot back from the swap file into the live ring buffer
     * (TD-193), resetting its coverage to 0.0f. LIFO order: the last slot evicted is the first
     * recalled. If the live buffer is already at capacity, making room evicts (and, in turn,
     * swaps out) the current least-used slot exactly as write() would.
     *
     * @return false if swap is disabled, the swap file doesn't exist, or it holds no records
     *         (nothing to recall) — true if a slot was successfully recalled.
     * @throws std::runtime_error if the swap file's own saved d_model doesn't match this
     *         instance's own (see load()'s own doc comment for the same class of check).
     */
    bool recall_from_swap();

    /** New conversation / new session boundary — drops every slot and its coverage entry. Does
     *  NOT touch the swap file — that's a separate, durable archive, not live session state. */
    void clear();

    /**
     * Persist the current session's slots (and their coverage) to a single file — session
     * state, not model-checkpoint versioning (see the plan's Compatibility section: deliberately
     * not wired into MNS/architecture-checkpoint machinery the way LeJEPAEncoder::save() is).
     * Does not touch the swap file (a separate, independently-configured archive — see the class
     * doc's TD-193 section).
     */
    void save(const std::string& filepath) const;

    /**
     * @throws std::runtime_error if the saved d_model doesn't match this instance's own, or if the
     *         file's own header slot count doesn't fit the file's actual remaining size (a
     *         corrupted or wrong-format file) — checked before trusting the count for any
     *         allocation, so a bad file fails clearly here rather than surfacing as an unrelated
     *         std::length_error/std::bad_alloc. A saved session larger than this instance's own
     *         current capacity is accepted — the least-used excess slots are evicted (to the
     *         swap file, if one is configured) down to this instance's capacity rather than
     *         throwing, since capacity is a runtime tuning knob, not an architectural constant —
     *         same policy write()'s own eviction uses (TD-193). Resets association_matrix() to a
     *         fresh all-zero size() x size() matrix (TD-194) — call load_associations() after,
     *         with a file saved against this exact post-load slot count/order, to restore it.
     */
    void load(const std::string& filepath);

    /**
     * Persist the current cross-reference association matrix (TD-194) to its own file, separate
     * from save()'s own concern (the live ring buffer's slots/coverage) — see the class doc for
     * why. A `size() x size()` matrix means nothing without also knowing which slots/order it
     * belongs to, so this is meant to be called alongside save(), against the same instance.
     */
    void save_associations(const std::string& filepath) const;

    /**
     * @throws std::runtime_error if the file's own saved slot count doesn't match this instance's
     *         *current* size() — call load() with the matching main state file first, since the
     *         association matrix's own indices only mean something relative to a specific
     *         slots_/coverage_ ordering — or if the file is corrupt/truncated relative to its own
     *         claimed slot count (same discipline load()'s own num_slots check uses, TD-191).
     */
    void load_associations(const std::string& filepath);
};
