// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-11

#include "PassHeartbeat.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "Logger.hpp"

namespace adai {

namespace {

std::atomic<std::uint64_t> g_pass_progress{0};

long long current_pid() {
#ifdef _WIN32
    return static_cast<long long>(_getpid());
#else
    return static_cast<long long>(getpid());
#endif
}

}  // namespace

void note_pass_progress() noexcept {
    g_pass_progress.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t pass_progress_count() noexcept {
    return g_pass_progress.load(std::memory_order_relaxed);
}

std::optional<PassHeartbeat> read_pass_heartbeat(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        return std::nullopt;
    std::string line;
    if (!std::getline(in, line))
        return std::nullopt;
    std::istringstream fields(line);
    PassHeartbeat heartbeat;
    std::string rest;
    if (!(fields >> heartbeat.pid >> heartbeat.count) || (fields >> rest) || heartbeat.pid <= 0) {
        return std::nullopt;
    }
    return heartbeat;
}

bool write_pass_heartbeat(const std::string& path, const PassHeartbeat& heartbeat) {
    namespace fs = std::filesystem;
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        out << heartbeat.pid << ' ' << heartbeat.count << '\n';
        out.flush();
        if (!out)
            return false;
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

PassHeartbeatWriter::PassHeartbeatWriter(std::string path, std::chrono::milliseconds interval)
    : path_(std::move(path)), interval_(interval) {
    std::error_code ec;
    const std::filesystem::path parent = std::filesystem::path(path_).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent, ec);
    thread_ = std::thread([this] { run(); });
}

PassHeartbeatWriter::~PassHeartbeatWriter() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable())
        thread_.join();
    std::error_code ec;
    std::filesystem::remove(path_, ec);
}

void PassHeartbeatWriter::run() {
    const long long pid = current_pid();
    std::optional<std::uint64_t> written;
    bool warned = false;
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
        const std::uint64_t count = pass_progress_count();
        if (!written || *written != count) {
            if (write_pass_heartbeat(path_, {pid, count})) {
                written = count;
            } else if (!warned) {
                warned = true;  // once: the supervisor will see no progress and stop the pass
                Logger::warn("Could not write the pass heartbeat file {}", path_);
            }
        }
        cv_.wait_for(lock, interval_, [this] { return stop_; });
    }
}

void PassStallWatch::start(long long pid, Clock::time_point now) {
    pid_ = pid;
    last_count_.reset();
    last_progress_ = now;
}

bool PassStallWatch::observe(const std::optional<PassHeartbeat>& heartbeat, Clock::time_point now) {
    if (heartbeat && heartbeat->pid == pid_ && (!last_count_ || *last_count_ != heartbeat->count)) {
        last_count_ = heartbeat->count;
        last_progress_ = now;
    }
    return limit_.count() > 0 && now - last_progress_ > limit_;
}

std::chrono::seconds PassStallWatch::since_progress(Clock::time_point now) const {
    return std::chrono::duration_cast<std::chrono::seconds>(now - last_progress_);
}

}  // namespace adai
