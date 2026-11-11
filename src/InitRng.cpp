// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10

#include "InitRng.hpp"
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include "Logger.hpp"

namespace adai {

namespace {
std::mutex seed_mutex;
std::mt19937_64 sequence;
bool seeded = false;
}  // namespace

void set_init_seed(int seed) {
    if (seed < 1) {
        throw std::invalid_argument("seed must be at least 1, got " + std::to_string(seed));
    }
    std::lock_guard<std::mutex> lock(seed_mutex);
    sequence.seed(static_cast<std::uint64_t>(seed));
    seeded = true;
}

std::uint32_t next_init_seed() {
    std::lock_guard<std::mutex> lock(seed_mutex);
    if (!seeded) {
        // In the range SEED accepts, so the logged value can be set.
        const int seed = static_cast<int>(std::random_device{}() % 2147483646u) + 1;
        sequence.seed(static_cast<std::uint64_t>(seed));
        seeded = true;
        Logger::info("Random seed {} (set SEED={} to replay this run)", seed, seed);
    }
    return static_cast<std::uint32_t>(sequence() >> 32);
}

}  // namespace adai
