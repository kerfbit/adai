// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-13

// TD-574: see GPUDebugSync.hpp.

#include "GPUDebugSync.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>

namespace adai {
namespace gpu {

namespace {

std::string lower_trimmed(const std::string& s) {
    const auto b = s.find_first_not_of(" \t");
    if (b == std::string::npos) {
        return "";
    }
    const auto e = s.find_last_not_of(" \t");
    std::string out = s.substr(b, e - b + 1);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

struct State {
    std::mutex mutex;  // guards settings replacement and the trace file
    std::unique_ptr<GPUDebugSettings> settings;
    std::FILE* trace = nullptr;
    std::atomic<const char*> kernel{"none"};
    std::atomic<const char*> phase{"none"};
    std::atomic<unsigned long long> seq{0};
    std::atomic<unsigned long long> host_runs{0};
};

State& state() {
    static State s;
    return s;
}

// Rewrites the trace file's single record in place; a fixed width keeps an older, longer record
// from showing through.
void write_trace(State& s, unsigned long long seq) {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.trace) {
        return;
    }
    char line[160];
    std::snprintf(line, sizeof line, "%llu %s %s", seq, s.phase.load(), s.kernel.load());
    char record[160];
    std::snprintf(record, sizeof record, "%-158.158s\n", line);
    std::fseek(s.trace, 0, SEEK_SET);
    std::fwrite(record, 1, 159, s.trace);
    std::fflush(s.trace);
}

void open_trace(State& s) {
    if (s.trace) {
        std::fclose(s.trace);
        s.trace = nullptr;
    }
    if (s.settings && !s.settings->trace_file.empty()) {
        s.trace = std::fopen(s.settings->trace_file.c_str(), "w");
        if (!s.trace) {
            s.settings->problems.push_back("ADAI_GPU_TRACE_FILE: cannot open " +
                                           s.settings->trace_file);
        }
    }
}

}  // namespace

GPUDebugSettings parse_gpu_debug_settings(const char* sync, const char* host_kernels,
                                          const char* trace_file) {
    GPUDebugSettings out;
    if (sync) {
        const std::string v = lower_trimmed(sync);
        if (v.empty() || v == "off" || v == "0") {
            out.sync = DebugSyncMode::Off;
        } else if (v == "step") {
            out.sync = DebugSyncMode::Step;
        } else if (v == "kernel") {
            out.sync = DebugSyncMode::Kernel;
        } else if (v.rfind("kernel:", 0) == 0) {
            const std::string n = v.substr(7);
            char* end = nullptr;
            const unsigned long every = std::strtoul(n.c_str(), &end, 10);
            if (!n.empty() && std::isdigit(static_cast<unsigned char>(n[0])) && *end == '\0' &&
                every >= 1 && every <= 1000000) {
                out.sync = DebugSyncMode::Kernel;
                out.sync_every = static_cast<unsigned>(every);
            } else {
                out.problems.push_back(std::string("ADAI_GPU_SYNC: '") + sync +
                                       "' needs a whole number from 1 to 1000000 after kernel:");
            }
        } else {
            out.problems.push_back(std::string("ADAI_GPU_SYNC: unknown mode '") + sync +
                                   "' (off, step, kernel or kernel:N)");
        }
    }
    if (host_kernels) {
        std::stringstream list(host_kernels);
        std::string item;
        while (std::getline(list, item, ',')) {
            const std::string v = lower_trimmed(item);
            if (v.empty()) {
                continue;
            } else if (v == "embedding") {
                out.host_kernels |= kHostEmbedding;
            } else if (v == "optimizer") {
                out.host_kernels |= kHostOptimizer;
            } else if (v == "dropout") {
                out.host_kernels |= kHostDropout;
            } else {
                out.problems.push_back("ADAI_GPU_HOST_KERNELS: unknown family '" + v +
                                       "' (embedding, optimizer or dropout)");
            }
        }
    }
    if (trace_file) {
        out.trace_file = trace_file;
    }
    return out;
}

const GPUDebugSettings& gpu_debug_settings() {
    State& s = state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.settings) {
            return *s.settings;
        }
        s.settings = std::make_unique<GPUDebugSettings>(parse_gpu_debug_settings(
            std::getenv("ADAI_GPU_SYNC"), std::getenv("ADAI_GPU_HOST_KERNELS"),
            std::getenv("ADAI_GPU_TRACE_FILE")));
        open_trace(s);
    }
    return *s.settings;
}

void set_gpu_debug_settings_for_testing(const GPUDebugSettings& settings) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.settings = std::make_unique<GPUDebugSettings>(settings);
    open_trace(s);
    s.kernel = "none";
    s.phase = "none";
    s.seq = 0;
    s.host_runs = 0;
}

bool wait_before_gpu_operation() {
    const GPUDebugSettings& g = gpu_debug_settings();
    if (g.sync != DebugSyncMode::Kernel) {
        return false;
    }
    // seq counts the operations already queued, so this waits before operations 1, N+1, 2N+1, …
    return g.sync_every <= 1 || state().seq.load(std::memory_order_relaxed) % g.sync_every == 0;
}

void note_gpu_kernel(const char* kernel) {
    State& s = state();
    s.kernel.store(kernel, std::memory_order_relaxed);
    const unsigned long long n = s.seq.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!gpu_debug_settings().trace_file.empty()) {
        write_trace(s, n);
    }
}

void note_gpu_phase(const char* phase) {
    State& s = state();
    s.phase.store(phase, std::memory_order_relaxed);
    if (!gpu_debug_settings().trace_file.empty()) {
        write_trace(s, s.seq.load(std::memory_order_relaxed));
    }
}

void note_host_kernel_run() {
    state().host_runs.fetch_add(1, std::memory_order_relaxed);
}

unsigned long long host_kernel_runs() {
    return state().host_runs.load(std::memory_order_relaxed);
}

std::string describe_last_gpu_work() {
    State& s = state();
    std::string out = std::string("kernel ") + s.kernel.load() + " in phase " + s.phase.load() +
                      " (GPU work #" + std::to_string(s.seq.load()) + ")";
    switch (gpu_debug_settings().sync) {
        case DebugSyncMode::Kernel:
            if (gpu_debug_settings().sync_every <= 1) {
                out +=
                    "; ADAI_GPU_SYNC=kernel: the queue was drained before it, so this kernel "
                    "failed";
            } else {
                out += "; ADAI_GPU_SYNC=kernel:" + std::to_string(gpu_debug_settings().sync_every) +
                       ": the failure is in this operation or one of the ones queued since the "
                       "last wait";
            }
            break;
        case DebugSyncMode::Step:
            out +=
                "; ADAI_GPU_SYNC=step: the failure is in this phase, the kernel is the last one "
                "it queued";
            break;
        case DebugSyncMode::Off:
            out += "; the last work queued, which the GPU may not have reached";
            break;
    }
    return out;
}

std::string gpu_debug_summary() {
    const GPUDebugSettings& g = gpu_debug_settings();
    if (!g.any() && g.problems.empty()) {
        return "";
    }
    std::string out = "GPU diagnostics (TD-574, not for production): ADAI_GPU_SYNC=";
    if (g.sync == DebugSyncMode::Kernel) {
        out += g.sync_every <= 1 ? std::string("kernel") : "kernel:" + std::to_string(g.sync_every);
    } else {
        out += g.sync == DebugSyncMode::Step ? "step" : "off";
    }
    out += ", ADAI_GPU_HOST_KERNELS=";
    std::string fams;
    if (g.host_kernels & kHostEmbedding)
        fams += "embedding,";
    if (g.host_kernels & kHostOptimizer)
        fams += "optimizer,";
    if (g.host_kernels & kHostDropout)
        fams += "dropout,";
    if (!fams.empty())
        fams.pop_back();
    out += fams.empty() ? "none" : fams;
    out += ", ADAI_GPU_TRACE_FILE=" + (g.trace_file.empty() ? std::string("none") : g.trace_file);
    for (const auto& p : g.problems) {
        out += "; ignored: " + p;
    }
    return out;
}

}  // namespace gpu
}  // namespace adai
