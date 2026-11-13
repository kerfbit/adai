#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-13

// TD-574: diagnostics for finding which GPU work an Arc engine reset interrupts. Off by default; a
// run turns them on through the environment, read once:
//
//   ADAI_GPU_SYNC=off|step|kernel|kernel:N
//                                   step: wait for the queue at each training phase (forward,
//                                   backward, optimizer, validation); kernel: also before every
//                                   kernel, so a failure is attributed to the one kernel that ran;
//                                   kernel:N: before every Nth GPU operation only, so at most N
//                                   are queued (how deep a chain the resets need)
//   ADAI_GPU_HOST_KERNELS=<list>    run these kernel families on the host instead (download,
//                                   compute, upload): embedding, optimizer, dropout
//   ADAI_GPU_TRACE_FILE=<path>      keep "<seq> <phase> <kernel>" of the last GPU work in this
//                                   file, rewritten in place, so a pass killed in a hang leaves it
//
// Every kernel entry point records its name (one atomic store), whatever the settings, so a lost
// device can always report the last work queued. Without kernel mode that is the last work
// *queued*, which an asynchronous queue may not have reached yet.
//
// Plain C++ (no SYCL or CUDA types): compiled into adai_gpu, below adai_core, so it can't use
// adai::Logger; callers log describe_last_gpu_work() and gpu_debug_summary().

#include <string>
#include <vector>

namespace adai {
namespace gpu {

enum class DebugSyncMode { Off, Step, Kernel };

/// Kernel families ADAI_GPU_HOST_KERNELS can move to the host (bit flags).
enum HostKernelFamily : unsigned {
    kHostEmbedding = 1u << 0,  ///< embedding_gather_add_pe_gpu, embedding_scatter_add_gpu (atomics)
    kHostOptimizer =
        1u << 1,             ///< optimizer_update_gpu, matrix_sum_squares_accumulate_gpu (atomics)
    kHostDropout = 1u << 2,  ///< dropout_forward_gpu (the mask generated on the device)
};

struct GPUDebugSettings {
    DebugSyncMode sync = DebugSyncMode::Off;
    /// With DebugSyncMode::Kernel: wait before every Nth GPU operation (1 = every one).
    unsigned sync_every = 1;
    unsigned host_kernels = 0;  ///< HostKernelFamily bits
    std::string trace_file;
    /// Values that couldn't be read; they are ignored (the setting keeps its default).
    std::vector<std::string> problems;

    bool any() const {
        return sync != DebugSyncMode::Off || host_kernels != 0 || !trace_file.empty();
    }
};

/// Reads the three settings from their values (null = unset). Pure, for tests.
GPUDebugSettings parse_gpu_debug_settings(const char* sync, const char* host_kernels,
                                          const char* trace_file);

/// The process's settings, read from the environment on first use.
const GPUDebugSettings& gpu_debug_settings();

/// Replaces the process's settings (tests only); closes any open trace file.
void set_gpu_debug_settings_for_testing(const GPUDebugSettings& settings);

inline bool host_kernel(HostKernelFamily family) {
    return (gpu_debug_settings().host_kernels & family) != 0;
}

/// Whether the next GPU operation should first wait for the queue: always in kernel mode, before
/// every sync_every-th operation with kernel:N, never otherwise.
bool wait_before_gpu_operation();

/// Records @p kernel (a string literal) as the latest GPU work. Called by each kernel entry point.
void note_gpu_kernel(const char* kernel);

/// Records @p phase (a string literal) as the training phase now queuing work.
void note_gpu_phase(const char* phase);

/// Counts one kernel call run on the host (ADAI_GPU_HOST_KERNELS), so a run can show that the
/// setting took effect.
void note_host_kernel_run();
unsigned long long host_kernel_runs();

/// "kernel <k> in phase <p> (GPU work #<n>)", plus what that means in the current sync mode.
std::string describe_last_gpu_work();

/// One line describing the active settings and their problems; empty when none is set.
std::string gpu_debug_summary();

}  // namespace gpu
}  // namespace adai
