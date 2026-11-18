#pragma once

// @adai-status: experimental
// @adai-version: 0.3.0
// @adai-reviewed: 2026-11-14

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

/// TD-574: the default bound on the GPU queue: a wait before every 8th GPU operation. Over the
/// second to fourth rounds on ai-machine it cut the Arc's engine resets by 2 to 12 times without
/// stopping them. The passes that seemed to hang in the bound's wait were resets the runtime never
/// reported, which happen unbounded too; the wait timeout below ends them.
constexpr unsigned kDefaultGpuQueueDepth = 8;

/// GPU_QUEUE_DEPTH's value: a whole number from 0 to 1000000, else (unset or malformed)
/// kDefaultGpuQueueDepth. The bound starts from the environment's value, so a binary that reads
/// no config file (gpu_encode_soak, gpu_decode_bench, the GPU tests) can still be bounded.
unsigned parse_gpu_queue_depth(const char* value);

/// The queue bound: wait for the queue before every @p depth-th GPU operation, so at most @p depth
/// are queued; 0 = unbounded (the behaviour before TD-574). Set from GPU_QUEUE_DEPTH by
/// incremental_trainer and chatbot_api_server. An explicit ADAI_GPU_SYNC=kernel or kernel:N
/// diagnostic takes precedence.
void set_gpu_queue_depth(unsigned depth);
unsigned gpu_queue_depth();

/// TD-574: how long a wait for GPU work may take before the work counts as stuck. In the third
/// round on ai-machine the Arc stopped running work without an engine reset, and the waits that
/// were waiting on it never returned. With a timeout the wait fails instead, the queue is marked
/// stuck, and the caller's device-loss handling (TD-566) takes over.
constexpr unsigned kDefaultGpuWaitTimeoutSeconds = 60;

/// GPU_WAIT_TIMEOUT_SECONDS's value: a whole number from 0 to 86400, else (unset or malformed)
/// kDefaultGpuWaitTimeoutSeconds. Read from the environment first, as for GPU_QUEUE_DEPTH.
unsigned parse_gpu_wait_timeout(const char* value);

/// The wait timeout in seconds; 0 = wait as long as it takes (the behaviour before). Set from
/// GPU_WAIT_TIMEOUT_SECONDS by incremental_trainer and chatbot_api_server. SYCL only: the CUDA
/// backend's waits don't use it.
void set_gpu_wait_timeout(unsigned seconds);
unsigned gpu_wait_timeout();

/// TD-574: records that a wait for GPU work timed out in this process (the SYCL backend's
/// GPUManager::wait_for()). It stays set for the life of the process: the stuck work may still
/// hold the device, and on the Level Zero v1 adapter the oneAPI runtime's own teardown at exit then
/// blocks on it, so a binary checks gpu_work_stuck() before returning from main() and ends with
/// exit_without_gpu_teardown() instead.
void note_gpu_work_stuck();
bool gpu_work_stuck();
void clear_gpu_work_stuck_for_testing();

/// Flushes the C streams and ends the process with @p code at once (std::_Exit), running no static
/// destructors or atexit handlers, so the GPU runtime's teardown can't block on stuck work. Flush
/// the logger and the C++ streams first.
[[noreturn]] void exit_without_gpu_teardown(int code);

/// Whether ADAI_GPU_KEEP_TEARDOWN=1 asks a binary to return from main() normally even after its GPU
/// work got stuck, running the runtime's teardown (TD-574 diagnostics: to take a stack of the Level
/// Zero v1 adapter's hang at exit). Read from the environment on each call.
bool keep_gpu_teardown_requested();

/// Whether the next GPU operation should first wait for the queue: always in kernel mode, before
/// every sync_every-th operation with kernel:N, otherwise before every gpu_queue_depth()-th one
/// (never when it is 0).
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
