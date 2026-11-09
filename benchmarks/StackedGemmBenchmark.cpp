// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-09

/**
 * TD-171, phase B0 of docs/proposals/real_batching_v2_plan.md: is one matrix multiply over B stacked
 * sequences faster than B separate ones? Measures bare dense multiplies at the model's real shapes,
 * then one decoder layer's decode step, where the attention score products can't be stacked (each
 * sequence attends only within itself). Runs on the CPU, and on the GPU in a GPU build.
 *
 *   stacked_gemm_benchmark [d_model] [d_ff] [num_heads] [repeats]
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>
#include "Matrix.hpp"
#ifdef ADAI_ENABLE_GPU
#include "gpu/MatrixGPU.hpp"
#endif

namespace {

std::vector<float> random_values(int rows, int cols) {
    static std::mt19937 rng(171);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> values(static_cast<std::size_t>(rows) * cols);
    for (auto& v : values) {
        v = normal(rng);
    }
    return values;
}

struct CpuBackend {
    using Mat = Matrix;
    static const char* name() {
        return "CPU";
    }
    static Mat make(int rows, int cols) {
        const auto values = random_values(rows, cols);
        Mat m(rows, cols);
        std::size_t next = 0;
        for (auto& row : m.data) {
            for (auto& x : row) {
                x = values[next++];
            }
        }
        return m;
    }
    static void finish() {}
};

#ifdef ADAI_ENABLE_GPU
struct GpuBackend {
    using Mat = adai::gpu::GPUMatrix;
    static const char* name() {
        return "GPU";
    }
    static Mat make(int rows, int cols) {
        const auto values = random_values(rows, cols);
        Mat m(rows, cols);
        m.upload(values.data(), rows * cols);
        return m;
    }
    static void finish() {
        adai::gpu::GPUManager::synchronize();
    }
};
#endif

double median_ms(int repeats, const std::function<void()>& work) {
    std::vector<double> samples;
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        work();
        samples.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

const std::vector<int> kBatches = {1, 4, 8, 16, 32};

/// One [B*L, d_in] x [d_in, d_out] multiply against B multiplies of [L, d_in] x [d_in, d_out].
template <typename Backend>
void dense_table(int rows_per_sequence, int d_in, int d_out, int repeats) {
    using Mat = typename Backend::Mat;
    Mat weights = Backend::make(d_in, d_out);
    std::printf("\n%s dense [B*%d, %d] x [%d, %d]\n", Backend::name(), rows_per_sequence, d_in, d_in,
                d_out);
    std::printf("%4s %12s %12s %8s\n", "B", "separate ms", "stacked ms", "speedup");
    for (int batch : kBatches) {
        Mat single = Backend::make(rows_per_sequence, d_in);
        Mat stacked = Backend::make(batch * rows_per_sequence, d_in);
        const double separate_ms = median_ms(repeats, [&] {
            for (int b = 0; b < batch; ++b) {
                Mat out = single * weights;
                (void)out;
            }
            Backend::finish();
        });
        const double stacked_ms = median_ms(repeats, [&] {
            Mat out = stacked * weights;
            (void)out;
            Backend::finish();
        });
        std::printf("%4d %12.3f %12.3f %7.2fx\n", batch, separate_ms, stacked_ms,
                    separate_ms / stacked_ms);
    }
}

/**
 * One decoder layer's KV-cached decode step for B sequences of `context` tokens. Dense: the
 * feed-forward's two projections and the self-attention Q, K, V, output and cross-attention Q,
 * output projections (K and V are cached). Not stackable: per head, the row of scores against
 * `context` keys and the product with `context` values, for self- and cross-attention.
 */
template <typename Backend>
void decode_step_table(int d_model, int d_ff, int num_heads, int context, int repeats) {
    using Mat = typename Backend::Mat;
    const int d_head = d_model / num_heads;
    Mat w_in = Backend::make(d_model, d_ff), w_out = Backend::make(d_ff, d_model);
    Mat w_attn = Backend::make(d_model, d_model);
    Mat keys = Backend::make(d_head, context), values = Backend::make(context, d_head);
    Mat query = Backend::make(1, d_head), scores = Backend::make(1, context);

    std::printf("\n%s decoder layer decode step, context %d\n", Backend::name(), context);
    std::printf("%4s %12s %12s %8s %14s\n", "B", "separate ms", "stacked ms", "speedup",
                "attention share");
    for (int batch : kBatches) {
        Mat row = Backend::make(1, d_model), row_ff = Backend::make(1, d_ff);
        Mat rows = Backend::make(batch, d_model), rows_ff = Backend::make(batch, d_ff);

        const auto attention = [&](int sequences) {
            for (int b = 0; b < sequences * 2 * num_heads; ++b) {
                Mat s = query * keys;
                Mat v = scores * values;
                (void)s;
                (void)v;
            }
        };
        const double attention_ms = median_ms(repeats, [&] {
            attention(batch);
            Backend::finish();
        });
        const double separate_ms = median_ms(repeats, [&] {
            for (int b = 0; b < batch; ++b) {
                Mat a = row * w_in;
                Mat c = row_ff * w_out;
                (void)a;
                (void)c;
                for (int p = 0; p < 6; ++p) {
                    Mat q = row * w_attn;
                    (void)q;
                }
            }
            attention(batch);
            Backend::finish();
        });
        const double stacked_ms = median_ms(repeats, [&] {
            Mat a = rows * w_in;
            Mat c = rows_ff * w_out;
            (void)a;
            (void)c;
            for (int p = 0; p < 6; ++p) {
                Mat q = rows * w_attn;
                (void)q;
            }
            attention(batch);
            Backend::finish();
        });
        std::printf("%4d %12.3f %12.3f %7.2fx %13.0f%%\n", batch, separate_ms, stacked_ms,
                    separate_ms / stacked_ms, 100.0 * attention_ms / separate_ms);
    }
}

template <typename Backend>
void run(int d_model, int d_ff, int num_heads, int repeats) {
    for (int rows_per_sequence : {1, 32, 256}) {
        dense_table<Backend>(rows_per_sequence, d_model, d_ff, repeats);
    }
    for (int context : {32, 256, 512}) {
        decode_step_table<Backend>(d_model, d_ff, num_heads, context, repeats);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int d_model = argc > 1 ? std::atoi(argv[1]) : 768;
    const int d_ff = argc > 2 ? std::atoi(argv[2]) : 3072;
    const int num_heads = argc > 3 ? std::atoi(argv[3]) : 12;
    const int repeats = argc > 4 ? std::atoi(argv[4]) : 15;
    run<CpuBackend>(d_model, d_ff, num_heads, repeats);
#ifdef ADAI_ENABLE_GPU
    if (adai::gpu::GPUManager::probe() && adai::gpu::GPUManager::initialize(0, 0.5f)) {
        std::printf("\n%s\n", adai::gpu::GPUManager::get_device_info().c_str());
        run<GpuBackend>(d_model, d_ff, num_heads, repeats);
    } else {
        std::printf("\nNo GPU device: GPU tables skipped\n");
    }
#endif
    return 0;
}
