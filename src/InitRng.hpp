#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-09

#include <cstdint>

namespace adai {

/**
 * TD-368: the one seedable source behind weight initialization and the trainer's random choices
 * (epoch shuffles, dropout masks, LeJEPA spans and SIGReg directions), so a run can be replayed.
 *
 * Each user takes one value from next_init_seed() to seed its own generator, so no generator is
 * shared between threads. Values depend only on the seed and the order of the calls. Without
 * set_init_seed() the first call picks a random seed and logs it, so any run can be replayed by
 * setting that value (`SEED` or `incremental_trainer --seed`).
 */

/// Restarts the sequence from @p seed (at least 1; std::invalid_argument otherwise).
void set_init_seed(int seed);

/// The next value of the sequence. Thread-safe.
std::uint32_t next_init_seed();

}  // namespace adai
