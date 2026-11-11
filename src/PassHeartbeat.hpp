#pragma once

// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-11

// TD-569: how trainer_service tells a training pass that is working from one that has stopped (a
// GPU call that never returns after a driver engine reset fails no sample, so TD-566's device check
// never runs, and the supervisor used to wait for such a pass forever).
//
//  - The pass's working code calls note_pass_progress() as it goes: each training and validation
//  sample,
//    each tokenized pair, each downloaded chunk, each BPE merge, each LeJEPA batch, each
//    checkpoint. It is one relaxed atomic increment, safe from any thread, so OpenMP and download
//    workers call it too.
//  - PassHeartbeatWriter, in the pass (`incremental_trainer --heartbeat-file <path>`), copies the
//  counter
//    and the process id into the file every couple of seconds. It only copies; a pass whose working
//    code stops keeps writing the same count, which is what lets a hang be seen.
//  - PassStallWatch, in trainer_service, reads the file and times how long the count has stayed the
//  same
//    on its own steady clock, so the two processes never compare clocks. A file from another
//    process (an earlier pass) is ignored by its pid.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace adai {

/// Records that the pass made progress. Cheap enough for a per-sample or per-chunk call.
void note_pass_progress() noexcept;

/// The number of note_pass_progress() calls so far in this process.
std::uint64_t pass_progress_count() noexcept;

struct PassHeartbeat {
    long long pid = 0;
    std::uint64_t count = 0;
};

/// The heartbeat in @p path ("<pid> <count>"), or nullopt if the file is missing or doesn't parse
/// (a write in progress is replaced by a rename, so a reader sees the old or the new file, never
/// half of one).
std::optional<PassHeartbeat> read_pass_heartbeat(const std::string& path);

/// Writes "<pid> <count>" to @p path through a temporary file and a rename. Returns false on
/// failure.
bool write_pass_heartbeat(const std::string& path, const PassHeartbeat& heartbeat);

/// The pass side: a thread that writes this process's id and pass_progress_count() to the file at
/// once and then every @p interval while the count changes. The destructor stops the thread and
/// removes the file.
class PassHeartbeatWriter {
   public:
    explicit PassHeartbeatWriter(
        std::string path, std::chrono::milliseconds interval = std::chrono::milliseconds(2000));
    ~PassHeartbeatWriter();
    PassHeartbeatWriter(const PassHeartbeatWriter&) = delete;
    PassHeartbeatWriter& operator=(const PassHeartbeatWriter&) = delete;

   private:
    void run();

    std::string path_;
    std::chrono::milliseconds interval_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;  // guarded by mutex_
    std::thread thread_;
};

/// The supervisor side: decides whether the pass with a given pid has gone too long without
/// progress. A limit of zero turns the check off.
class PassStallWatch {
   public:
    using Clock = std::chrono::steady_clock;

    explicit PassStallWatch(std::chrono::seconds limit) : limit_(limit) {}

    /// A new pass was launched at @p now with process id @p pid; its time without progress starts
    /// here.
    void start(long long pid, Clock::time_point now);

    /// Takes the latest heartbeat read (nullopt when there is none) and returns true if the pass
    /// has now gone longer than the limit without its count changing. A heartbeat from another pid
    /// is ignored.
    bool observe(const std::optional<PassHeartbeat>& heartbeat, Clock::time_point now);

    /// Time since the pass last made progress (or was started), as of @p now.
    std::chrono::seconds since_progress(Clock::time_point now) const;

    std::chrono::seconds limit() const {
        return limit_;
    }

   private:
    std::chrono::seconds limit_;
    long long pid_ = 0;
    std::optional<std::uint64_t> last_count_;
    Clock::time_point last_progress_{};
};

}  // namespace adai
