// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-15

// TD-574: a standalone reproducer for the Arc Pro B60 compute engine resets, for the report to
// Intel. It uses only SYCL (and oneMKL with --mkl), none of this project's code, so it builds on
// its own:
//
//   icpx -fsycl -O2 SyclEngineResetRepro.cpp -o sycl_engine_reset_repro
//   icpx -fsycl -qmkl -O2 -DREPRO_WITH_MKL SyclEngineResetRepro.cpp -o sycl_engine_reset_repro (for
//   --mkl)
//
//   sycl_engine_reset_repro [--seconds N] [--layers N] [--wait-every N] [--mkl] [--normal-priority]
//                           [--timeout N] [--blocking-wait]
//
// It does what our training pass does to the GPU, without the model: one in-order queue (low
// priority unless --normal-priority), and per step an upload of a small input, then <layers> times
// a 16x64 by 64x64 matrix multiply (a SYCL kernel, or oneMKL with --mkl), a LayerNorm and a softmax
// with work-group reductions, a GELU, a float-atomic sum of squares and a device-to-device copy,
// and at the end of the step a download of one float the host waits for (our per-sample loss). With
// --wait-every N the host also waits for the queue before every Nth operation (our
// GPU_QUEUE_DEPTH). Every wait polls its event and gives up after --timeout seconds (default 60):
// an engine reset the runtime never reports leaves it pending forever (observation 3 of the
// report). With --blocking-wait every wait is a plain sycl::event::wait() instead, so a stack of a
// process stuck on such a reset shows the runtime's wait; a watchdog thread then ends the process
// (exit code 2) after --timeout seconds with no wait finishing.
//
// Each second it prints a timestamped progress line (UTC, milliseconds), so the time from the last
// one to the kernel's "Engine reset" can be measured. Exit codes: 0 ran the whole time, 1 bad
// arguments, 2 a wait timed out (stuck), 3 the runtime reported an error (device lost), 4 no GPU.

#include <sycl/sycl.hpp>
#ifdef REPRO_WITH_MKL
#include <oneapi/mkl/blas.hpp>
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kRows = 16;
constexpr int kCols = 64;
constexpr int kElems = kRows * kCols;

struct Options {
    int seconds = 60;
    int layers = 24;
    long wait_every = 0;
    bool mkl = false;
    bool normal_priority = false;
    int timeout = 60;
    bool blocking_wait = false;
};

// For --blocking-wait's watchdog: when a wait last finished, and the counts at that time.
std::atomic<int64_t> g_last_wait_done_ns{0};
std::atomic<long> g_ops_done{0};
std::atomic<long> g_steps_done{0};

int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string now_utc() {
    using namespace std::chrono;
    const auto t = system_clock::now();
    const auto ms = duration_cast<milliseconds>(t.time_since_epoch()).count() % 1000;
    const std::time_t secs = system_clock::to_time_t(t);
    std::tm tm{};
    gmtime_r(&secs, &tm);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%02d:%02d:%02d.%03d", tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<int>(ms));
    return buf;
}

// Prints one line prefixed with the UTC time.
__attribute__((format(printf, 1, 2))) void say(const char* fmt, ...) {
    std::printf("[%s] ", now_utc().c_str());
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
    std::printf("\n");
    std::fflush(stdout);
}

bool parse_long(const char* s, long lo, long hi, long& out) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < lo || v > hi) {
        return false;
    }
    out = v;
    return true;
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        long v = 0;
        auto next = [&](long lo, long hi) {
            return i + 1 < argc && parse_long(argv[++i], lo, hi, v);
        };
        if (a == "--seconds" && next(1, 86400)) {
            o.seconds = static_cast<int>(v);
        } else if (a == "--layers" && next(1, 1000)) {
            o.layers = static_cast<int>(v);
        } else if (a == "--wait-every" && next(0, 1000000)) {
            o.wait_every = v;
        } else if (a == "--timeout" && next(1, 86400)) {
            o.timeout = static_cast<int>(v);
        } else if (a == "--mkl") {
            o.mkl = true;
        } else if (a == "--normal-priority") {
            o.normal_priority = true;
        } else if (a == "--blocking-wait") {
            o.blocking_wait = true;
        } else {
            std::fprintf(stderr, "bad argument: %s\n", a.c_str());
            return false;
        }
    }
#ifndef REPRO_WITH_MKL
    if (o.mkl) {
        std::fprintf(stderr, "--mkl: built without oneMKL (-DREPRO_WITH_MKL)\n");
        return false;
    }
#endif
    return true;
}

// Waits for @p e by polling its status, so a wait the runtime never completes can give up; with
// @p blocking, a plain sycl::event::wait() (the watchdog gives up instead).
bool wait_or_give_up(sycl::event e, int timeout_s, bool blocking) {
    if (blocking) {
        e.wait();
        g_last_wait_done_ns.store(steady_ns());
        return true;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    for (;;) {
        if (e.get_info<sycl::info::event::command_execution_status>() ==
            sycl::info::event_command_status::complete) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// Row-wise LayerNorm (in place) with work-group reductions: one work-group of kCols per row.
sycl::event layer_norm(sycl::queue& q, float* x) {
    return q.parallel_for(sycl::nd_range<1>(kElems, kCols), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        const auto g = it.get_group();
        const float v = x[i];
        const float mean = sycl::reduce_over_group(g, v, sycl::plus<float>()) / kCols;
        const float d = v - mean;
        const float var = sycl::reduce_over_group(g, d * d, sycl::plus<float>()) / kCols;
        x[i] = d / sycl::sqrt(var + 1e-5f);
    });
}

// Row-wise softmax (in place) with work-group reductions.
sycl::event softmax(sycl::queue& q, float* x) {
    return q.parallel_for(sycl::nd_range<1>(kElems, kCols), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        const auto g = it.get_group();
        const float v = x[i];
        const float m = sycl::reduce_over_group(g, v, sycl::maximum<float>());
        const float e = sycl::exp(v - m);
        const float s = sycl::reduce_over_group(g, e, sycl::plus<float>());
        x[i] = e / s;
    });
}

sycl::event gelu(sycl::queue& q, float* x) {
    return q.parallel_for(sycl::range<1>(kElems), [=](sycl::id<1> i) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + sycl::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    });
}

sycl::event sum_squares(sycl::queue& q, const float* x, float* acc) {
    return q.parallel_for(sycl::range<1>(kElems), [=](sycl::id<1> i) {
        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            a(*acc);
        a.fetch_add(x[i] * x[i]);
    });
}

// y = x (kRows x kCols) times w (kCols x kCols).
sycl::event matmul(sycl::queue& q, const float* x, const float* w, float* y, bool mkl) {
#ifdef REPRO_WITH_MKL
    if (mkl) {
        return oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans,
                                                  oneapi::mkl::transpose::nontrans, kRows, kCols,
                                                  kCols, 1.0f, x, kCols, w, kCols, 0.0f, y, kCols);
    }
#else
    (void)mkl;
#endif
    return q.parallel_for(sycl::range<2>(kRows, kCols), [=](sycl::id<2> id) {
        const size_t r = id[0], c = id[1];
        float s = 0.0f;
        for (int k = 0; k < kCols; ++k) {
            s += x[r * kCols + k] * w[k * kCols + c];
        }
        y[r * kCols + c] = s;
    });
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse(argc, argv, o)) {
        return 1;
    }

    bool async_error = false;
    auto on_async = [&](sycl::exception_list errors) {
        for (const auto& e : errors) {
            try {
                std::rethrow_exception(e);
            } catch (const sycl::exception& ex) {
                say("asynchronous SYCL error: %s", ex.what());
                async_error = true;
            }
        }
    };

    sycl::device dev;
    try {
        dev = sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception&) {
        const char* allow = std::getenv("ADAI_SYCL_ALLOW_CPU_DEVICE");
        if (allow == nullptr || std::string(allow) != "1") {
            say("no GPU");
            return 4;
        }
        dev = sycl::device(sycl::cpu_selector_v);
    }
    sycl::property_list props =
        o.normal_priority ? sycl::property_list{sycl::property::queue::in_order{}}
                          : sycl::property_list{sycl::property::queue::in_order{},
                                                sycl::ext::oneapi::property::queue::priority_low{}};
    sycl::queue q(dev, on_async, props);
    say("device: %s", dev.get_info<sycl::info::device::name>().c_str());
    say("backend: %s",
        dev.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "other");
    say("settings: %d s, %d layers per step, wait every %ld operations, %s multiply, %s priority, "
        "%d s wait timeout, %s waits",
        o.seconds, o.layers, o.wait_every, o.mkl ? "oneMKL" : "SYCL kernel",
        o.normal_priority ? "normal" : "low", o.timeout, o.blocking_wait ? "blocking" : "polled");

    if (o.blocking_wait) {
        // The main thread stays in sycl::event::wait() on a reset the runtime never reports; this
        // thread ends the process once no wait has finished for --timeout seconds.
        g_last_wait_done_ns.store(steady_ns());
        std::thread([timeout = o.timeout] {
            const int64_t limit = static_cast<int64_t>(timeout) * 1000000000;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (steady_ns() - g_last_wait_done_ns.load() >= limit) {
                    say("STUCK: a blocking wait for the GPU did not finish within %d s, after %ld "
                        "operations (step %ld)",
                        timeout, g_ops_done.load(), g_steps_done.load());
                    std::fflush(nullptr);
                    std::_Exit(2);  // the runtime's teardown can block on the stuck work
                }
            }
        }).detach();
    }

    long ops = 0, steps = 0;
    try {
        float* x = sycl::malloc_device<float>(kElems, q);
        float* y = sycl::malloc_device<float>(kElems, q);
        float* w = sycl::malloc_device<float>(kCols * kCols, q);
        float* acc = sycl::malloc_device<float>(1, q);
        float* host_in = sycl::malloc_host<float>(kElems, q);
        float* host_out = sycl::malloc_host<float>(1, q);
        std::vector<float> wh(kCols * kCols);
        uint32_t seed = 12345;
        auto rnd = [&] {
            seed = seed * 1664525u + 1013904223u;
            return (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 0.25f;
        };
        for (auto& v : wh) {
            v = rnd();
        }
        q.memcpy(w, wh.data(), wh.size() * sizeof(float)).wait();

        // queue_op: counts an operation and, with --wait-every, waits before every Nth one.
        bool stuck = false;
        auto before_op = [&] {
            if (o.wait_every > 0 && ops > 0 && ops % o.wait_every == 0) {
                if (!wait_or_give_up(q.ext_oneapi_submit_barrier(), o.timeout, o.blocking_wait)) {
                    stuck = true;
                }
            }
            ++ops;
            return !stuck;
        };

        const auto start = std::chrono::steady_clock::now();
        auto last_report = start;
        while (std::chrono::steady_clock::now() - start < std::chrono::seconds(o.seconds)) {
            for (int i = 0; i < kElems; ++i) {
                host_in[i] = rnd();
            }
            if (!before_op())
                break;
            q.memcpy(x, host_in, kElems * sizeof(float));
            if (!before_op())
                break;
            q.memset(acc, 0, sizeof(float));
            for (int l = 0; l < o.layers && !stuck; ++l) {
                if (!before_op())
                    break;
                matmul(q, x, w, y, o.mkl);
                if (!before_op())
                    break;
                layer_norm(q, y);
                if (!before_op())
                    break;
                gelu(q, y);
                if (!before_op())
                    break;
                softmax(q, y);
                if (!before_op())
                    break;
                sum_squares(q, y, acc);
                if (!before_op())
                    break;
                q.memcpy(x, y, kElems * sizeof(float));
            }
            if (stuck || !before_op())
                break;
            sycl::event down = q.memcpy(host_out, acc, sizeof(float));
            if (!wait_or_give_up(down, o.timeout, o.blocking_wait)) {
                stuck = true;
                break;
            }
            q.throw_asynchronous();
            if (async_error)
                break;
            ++steps;
            g_ops_done.store(ops);
            g_steps_done.store(steps);
            const auto now = std::chrono::steady_clock::now();
            if (now - last_report >= std::chrono::seconds(1)) {
                last_report = now;
                say("progress: %ld steps, %ld operations", steps, ops);
            }
        }
        if (stuck) {
            say("STUCK: a wait for the GPU did not finish within %d s, after %ld operations (step "
                "%ld)",
                o.timeout, ops, steps);
            std::fflush(nullptr);
            std::_Exit(2);  // the runtime's teardown can block on the stuck work
        }
        if (async_error) {
            say("FAILED: the runtime reported an error after %ld operations (step %ld)", ops,
                steps);
            return 3;
        }
        q.wait_and_throw();
        sycl::free(x, q);
        sycl::free(y, q);
        sycl::free(w, q);
        sycl::free(acc, q);
        sycl::free(host_in, q);
        sycl::free(host_out, q);
    } catch (const sycl::exception& ex) {
        say("FAILED: %s after %ld operations (step %ld)", ex.what(), ops, steps);
        return 3;
    }
    say("done: %ld steps, %ld operations", steps, ops);
    return 0;
}
