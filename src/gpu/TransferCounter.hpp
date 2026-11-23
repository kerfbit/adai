#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-22

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace adai {
namespace gpu {

/**
 * TD-590: counts of host<->device traffic, so tests can hold GPU training to the owner rule (October 19,
 * 2026) that it moves nothing between host and device without a critical reason, and passes can log what
 * they moved. Counted where the copies are made (GPUMemory::copy_from_host()/copy_to_host(),
 * matrix_download_gpu() and the backends' other host copies) and where a host Matrix operation runs on the
 * device as a round trip (Matrix::multiply_gpu() and the other *_gpu() helpers). Device-to-device copies
 * aren't counted. Process-wide and thread-safe (relaxed atomics: counts, not ordering).
 */
struct TransferStats {
    std::uint64_t uploads = 0;           ///< host -> device copies
    std::uint64_t upload_bytes = 0;
    std::uint64_t downloads = 0;         ///< device -> host copies
    std::uint64_t download_bytes = 0;
    std::uint64_t host_round_trips = 0;  ///< host Matrix operations sent to the device and back
};

namespace detail {
inline std::atomic<std::uint64_t> g_uploads{0};
inline std::atomic<std::uint64_t> g_upload_bytes{0};
inline std::atomic<std::uint64_t> g_downloads{0};
inline std::atomic<std::uint64_t> g_download_bytes{0};
inline std::atomic<std::uint64_t> g_host_round_trips{0};
}  // namespace detail

inline void note_upload(std::size_t bytes) {
    detail::g_uploads.fetch_add(1, std::memory_order_relaxed);
    detail::g_upload_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

inline void note_download(std::size_t bytes) {
    detail::g_downloads.fetch_add(1, std::memory_order_relaxed);
    detail::g_download_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

inline void note_host_round_trip() {
    detail::g_host_round_trips.fetch_add(1, std::memory_order_relaxed);
}

inline TransferStats transfer_stats() {
    TransferStats s;
    s.uploads = detail::g_uploads.load(std::memory_order_relaxed);
    s.upload_bytes = detail::g_upload_bytes.load(std::memory_order_relaxed);
    s.downloads = detail::g_downloads.load(std::memory_order_relaxed);
    s.download_bytes = detail::g_download_bytes.load(std::memory_order_relaxed);
    s.host_round_trips = detail::g_host_round_trips.load(std::memory_order_relaxed);
    return s;
}

inline void reset_transfer_stats() {
    detail::g_uploads.store(0, std::memory_order_relaxed);
    detail::g_upload_bytes.store(0, std::memory_order_relaxed);
    detail::g_downloads.store(0, std::memory_order_relaxed);
    detail::g_download_bytes.store(0, std::memory_order_relaxed);
    detail::g_host_round_trips.store(0, std::memory_order_relaxed);
}

/// The traffic between two snapshots (later minus earlier).
inline TransferStats transfer_delta(const TransferStats& later, const TransferStats& earlier) {
    TransferStats d;
    d.uploads = later.uploads - earlier.uploads;
    d.upload_bytes = later.upload_bytes - earlier.upload_bytes;
    d.downloads = later.downloads - earlier.downloads;
    d.download_bytes = later.download_bytes - earlier.download_bytes;
    d.host_round_trips = later.host_round_trips - earlier.host_round_trips;
    return d;
}

}  // namespace gpu
}  // namespace adai
