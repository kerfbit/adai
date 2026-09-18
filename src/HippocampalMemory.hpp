#pragma once

// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-17

#include <deque>
#include <string>
#include <utility>
#include <vector>
#include "Matrix.hpp"

/**
 * HippocampalMemory (LeJEPA world-model plan, HM-1 / TD-179)
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
 * coverage in a separate `coverage_` vector kept in lockstep with `slots_` (same size, same
 * insertion/eviction order) instead of embedded per-slot — everything else matches the plan's
 * interface exactly.
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
    std::vector<float> coverage_;  // coverage_[i] corresponds to slots_[i]; see class doc above
    int capacity;
    int d_model;

   public:
    /**
     * @param d_model Embedding dimension — must match whatever encoder's encode() output feeds
     *                write()'s own key/value arguments (typically LeJEPAEncoder::encode()).
     * @param capacity Ring-buffer bound; the oldest slot is evicted once a write() would exceed
     *                 this (FIFO — the simplest possible write policy, per this item's own
     *                 Action Items; salience-gated writing is a documented future extension, not
     *                 implemented here).
     */
    explicit HippocampalMemory(int d_model, int capacity = 512);

    /**
     * Write a new episode. FIFO eviction at capacity: the single oldest slot (and its
     * corresponding coverage entry) is dropped before the new one is appended, so size() never
     * exceeds capacity. The new slot's coverage starts at 0.0f.
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
     * write() (eviction/append), clear(), or load() call that changes size(), the same rule as
     * any other vector-invalidating operation — do not hold onto this reference across one.
     */
    std::vector<float>& coverage_vector();

    /** coverage[i] *= gamma for every currently-stored slot, called once per decode step. */
    void decay_coverage(float gamma);

    int size() const {
        return static_cast<int>(slots_.size());
    }

    int get_capacity() const {
        return capacity;
    }

    int get_d_model() const {
        return d_model;
    }

    /** New conversation / new session boundary — drops every slot and its coverage entry. */
    void clear();

    /**
     * Persist the current session's slots (and their coverage) to a single file — session
     * state, not model-checkpoint versioning (see the plan's Compatibility section: deliberately
     * not wired into MNS/architecture-checkpoint machinery the way LeJEPAEncoder::save() is).
     */
    void save(const std::string& filepath) const;

    /**
     * @throws std::runtime_error if the saved d_model doesn't match this instance's own, or if the
     *         file's own header slot count doesn't fit the file's actual remaining size (a
     *         corrupted or wrong-format file) — checked before trusting the count for any
     *         allocation, so a bad file fails clearly here rather than surfacing as an unrelated
     *         std::length_error/std::bad_alloc. A saved session larger than this instance's own
     *         current capacity is accepted — the oldest excess slots are evicted down to this
     *         instance's capacity rather than throwing, since capacity is a runtime tuning knob,
     *         not an architectural constant.
     */
    void load(const std::string& filepath);
};
