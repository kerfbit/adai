// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-13

// TD-050/TD-033: validates and times the GPU KV-cached decode on a real device.
//
// Builds one randomly initialized EncoderDecoderModel and decodes greedily, without stop tokens,
// through four paths:
//   gpu_cached  LLMDecoder::gpu_decode_step() with one GPUDecoderKVCache per generation (what
//               gpu_generate_response*() run for every strategy except beam search)
//   gpu_full    LLMDecoder::gpu_decode() over the whole sequence every step (the path before
//               TD-050, still used for beam search)
//   cpu_cached  LLMDecoder::forward_with_cache() with Matrix math kept on the CPU (ScopedCpuOnly)
//   cpu_dispatch the same, with Matrix::operator* sending large products to the GPU one call at a
//               time (chatbot_api_server's path on a GPU host before TD-033)
//
// Validation first: the paths run in lockstep on gpu_full's tokens, and every step's last-row
// logits are compared. The cached and full GPU paths must agree within 1e-3 * (1 + the largest
// |logit|) at every step and choose the same token (a choice may differ only where gpu_full's top
// two logits are closer than twice the difference).
// Then each path generates on its own, timed: one warm-up and <repeats> timed runs per length.
//
//   gpu_decode_bench [--lengths 50,100] [--repeats 3] [--d-model 768] [--heads 12] [--d-ff 3072]
//                    [--encoder-layers 8] [--decoder-layers 24] [--max-seq 512] [--vocab 16000]
//                    [--input-length 32] [--seed 1] [--paths gpu_cached,gpu_full,cpu_cached,
//                    cpu_dispatch]
//
// Prints "VALIDATE ..." and "RESULT path=<p> length=<n> ..." lines. Exit codes: 0 passed,
// 1 bad arguments or a validation failure, 2 the GPU stopped running work (a lost device), 3 no
// GPU.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "EncoderDecoderModel.hpp"
#include "InitRng.hpp"
#include "KVCache.hpp"
#include "Matrix.hpp"
#include "SpecialTokens.hpp"
#include "gpu/GPUDebugSync.hpp"
#include "gpu/GPUUtils.hpp"
#include "gpu/MatrixGPU.hpp"

namespace {

struct Options {
    std::vector<int> lengths{50, 100};
    int repeats = 3;
    int d_model = 768;
    int heads = 12;
    int d_ff = 3072;
    int encoder_layers = 8;
    int decoder_layers = 24;
    int max_seq = 512;
    int vocab = 16000;
    int input_length = 32;
    int seed = 1;
    std::vector<std::string> paths{"gpu_cached", "gpu_full", "cpu_cached", "cpu_dispatch"};
};

int parse_positive(const std::string& flag, const std::string& s) {
    size_t used = 0;
    int v = 0;
    try {
        v = std::stoi(s, &used);
    } catch (...) {
        used = 0;
    }
    if (used != s.size() || v < 1) {
        throw std::invalid_argument(flag + " needs a whole number of at least 1, got '" + s + "'");
    }
    return v;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, ',')) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int a = 1; a < argc; ++a) {
        const std::string flag = argv[a];
        if (flag == "--help" || flag == "-h") {
            throw std::invalid_argument("");
        }
        if (a + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + flag);
        }
        const std::string value = argv[++a];
        if (flag == "--lengths") {
            o.lengths.clear();
            for (const auto& item : split(value)) {
                o.lengths.push_back(parse_positive(flag, item));
            }
        } else if (flag == "--paths") {
            o.paths = split(value);
            for (const auto& p : o.paths) {
                if (p != "gpu_cached" && p != "gpu_full" && p != "cpu_cached" &&
                    p != "cpu_dispatch") {
                    throw std::invalid_argument("unknown path '" + p + "'");
                }
            }
        } else if (flag == "--repeats") {
            o.repeats = parse_positive(flag, value);
        } else if (flag == "--d-model") {
            o.d_model = parse_positive(flag, value);
        } else if (flag == "--heads") {
            o.heads = parse_positive(flag, value);
        } else if (flag == "--d-ff") {
            o.d_ff = parse_positive(flag, value);
        } else if (flag == "--encoder-layers") {
            o.encoder_layers = parse_positive(flag, value);
        } else if (flag == "--decoder-layers") {
            o.decoder_layers = parse_positive(flag, value);
        } else if (flag == "--max-seq") {
            o.max_seq = parse_positive(flag, value);
        } else if (flag == "--vocab") {
            o.vocab = parse_positive(flag, value);
        } else if (flag == "--input-length") {
            o.input_length = parse_positive(flag, value);
        } else if (flag == "--seed") {
            o.seed = parse_positive(flag, value);
        } else {
            throw std::invalid_argument("Unknown argument " + flag);
        }
    }
    if (o.lengths.empty()) {
        throw std::invalid_argument("--lengths is empty");
    }
    for (int len : o.lengths) {
        if (len > o.max_seq) {
            throw std::invalid_argument("length " + std::to_string(len) + " is past --max-seq " +
                                        std::to_string(o.max_seq));
        }
    }
    if (o.input_length > o.max_seq) {
        throw std::invalid_argument("--input-length is past --max-seq");
    }
    if (o.d_model % o.heads != 0) {
        throw std::invalid_argument("--d-model must be a multiple of --heads");
    }
    if (o.vocab < 21) {
        throw std::invalid_argument("--vocab must be at least 21");
    }
    return o;
}

int argmax(const std::vector<float>& row) {
    return static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin());
}

// The gap between the largest and second-largest value.
float top_two_gap(const std::vector<float>& row) {
    float first = -INFINITY, second = -INFINITY;
    for (float v : row) {
        if (v > first) {
            second = first;
            first = v;
        } else if (v > second) {
            second = v;
        }
    }
    return first - second;
}

#ifdef ADAI_ENABLE_GPU

// One decode path: reset() starts a generation, next() returns the last row of logits after
// feeding the whole sequence so far (each path takes only what it hasn't seen).
struct DecodePath {
    std::function<void()> reset;
    std::function<std::vector<float>(const std::vector<int>&)> next;
};

std::vector<float> last_row_gpu(LanguageModelHead& head, const adai::gpu::GPUMatrix& dec_out,
                                int vocab) {
    adai::gpu::GPUMatrix logits = head.gpu_forward(dec_out);
    std::vector<float> row(static_cast<size_t>(vocab));
    adai::gpu::matrix_download_gpu(
        logits.device_ptr() + static_cast<size_t>(logits.rows - 1) * vocab, row.data(), vocab);
    return row;
}

std::vector<float> last_row_cpu(LanguageModelHead& head, const Matrix& dec_out) {
    Matrix logits = head.forward(dec_out);
    return std::vector<float>(logits.data[logits.rows - 1].begin(),
                              logits.data[logits.rows - 1].end());
}

#endif

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    try {
        opt = parse_args(argc, argv);
    } catch (const std::invalid_argument& e) {
        if (e.what()[0] != '\0') {
            std::fprintf(stderr, "%s\n", e.what());
        }
        std::fprintf(stderr,
                     "usage: %s [--lengths 50,100] [--repeats 3] [--d-model 768] [--heads 12] "
                     "[--d-ff 3072] [--encoder-layers 8] [--decoder-layers 24] [--max-seq 512] "
                     "[--vocab 16000] [--input-length 32] [--seed 1] "
                     "[--paths gpu_cached,gpu_full,cpu_cached,cpu_dispatch]\n",
                     argv[0]);
        return e.what()[0] == '\0' ? 0 : 1;
    }

#ifndef ADAI_ENABLE_GPU
    std::fprintf(stderr, "built without GPU support\n");
    return 3;
#else
    if (!Matrix::gpu_try_initialize(0, 0.5f, true)) {
        std::fprintf(stderr, "no GPU device\n");
        return 3;
    }
    std::printf("%s\n", Matrix::gpu_info().c_str());
    if (const std::string debug = adai::gpu::gpu_debug_summary(); !debug.empty()) {
        std::printf("%s\n", debug.c_str());
    }

    adai::set_init_seed(opt.seed);
    EncoderDecoderModel model(opt.vocab, opt.d_model, opt.encoder_layers, opt.decoder_layers,
                              opt.heads, opt.d_ff, opt.max_seq);
    model.set_training(false);
    LLMEncoder& encoder = *model.get_encoder();
    LLMDecoder& decoder = *model.get_decoder();
    LanguageModelHead& head = *model.get_lm_head();
    const int vocab = opt.vocab;
    const int bos = adai::SpecialTokenIDs::BOS;

    std::mt19937 rng(static_cast<std::uint32_t>(opt.seed));
    std::uniform_int_distribution<int> pick(20, vocab - 1);
    std::vector<int> input(static_cast<size_t>(opt.input_length));
    for (int& id : input) {
        id = pick(rng);
    }
    std::printf(
        "model: vocab %d, d_model %d, heads %d, d_ff %d, encoder layers %d, decoder layers %d, "
        "max_seq %d; input %d tokens; seed %d\n",
        vocab, opt.d_model, opt.heads, opt.d_ff, opt.encoder_layers, opt.decoder_layers,
        opt.max_seq, opt.input_length, opt.seed);

    Matrix encoder_mask(opt.input_length, opt.input_length);
    for (auto& row : encoder_mask.data) {
        std::fill(row.begin(), row.end(), 1.0f);
    }
    Matrix cpu_encoder_out;
    std::unique_ptr<adai::gpu::GPUMatrix> gpu_encoder_out;
    std::unique_ptr<adai::gpu::GPUDecoderKVCache> gpu_cache;
    std::unique_ptr<DecoderKVCache> cpu_cache;
    size_t processed = 0;

    DecodePath gpu_cached{
        [&] {
            gpu_cache = std::make_unique<adai::gpu::GPUDecoderKVCache>(opt.decoder_layers,
                                                                       opt.max_seq, opt.d_model);
            processed = 0;
        },
        [&](const std::vector<int>& tokens) {
            std::vector<int> fresh(tokens.begin() + static_cast<long>(processed), tokens.end());
            processed = tokens.size();
            return last_row_gpu(head, decoder.gpu_decode_step(fresh, *gpu_cache, *gpu_encoder_out),
                                vocab);
        }};
    DecodePath gpu_full{[] {},
                        [&](const std::vector<int>& tokens) {
                            return last_row_gpu(head, decoder.gpu_decode(tokens, *gpu_encoder_out),
                                                vocab);
                        }};
    size_t cpu_processed = 0;
    DecodePath cpu_cached{
        [&] {
            cpu_cache = std::make_unique<DecoderKVCache>(opt.decoder_layers);
            cpu_processed = 0;
        },
        [&](const std::vector<int>& tokens) {
            std::vector<int> fresh(tokens.begin() + static_cast<long>(cpu_processed), tokens.end());
            cpu_processed = tokens.size();
            return last_row_cpu(head,
                                decoder.forward_with_cache(fresh, *cpu_cache, &cpu_encoder_out));
        }};

    auto encode = [&](bool on_gpu) {
        if (on_gpu) {
            gpu_encoder_out = std::make_unique<adai::gpu::GPUMatrix>(encoder.gpu_encode(input));
        } else {
            cpu_encoder_out = encoder.encode_with_mask(input, encoder_mask);
        }
    };

    try {
        // Validation: lockstep on gpu_full's tokens.
        int failures = 0;
        {
            const int length = *std::max_element(opt.lengths.begin(), opt.lengths.end());
            encode(true);
            {
                const adai::gpu::ScopedCpuOnly cpu_only;
                encode(false);
            }
            gpu_cached.reset();
            cpu_cached.reset();
            std::vector<int> tokens{bos};
            float max_cached_diff = 0.0f, max_cpu_diff = 0.0f;
            int cached_agree = 0, cpu_agree = 0, cached_near_ties = 0;
            for (int step = 0; step < length; ++step) {
                const std::vector<float> full = gpu_full.next(tokens);
                const std::vector<float> cached = gpu_cached.next(tokens);
                std::vector<float> cpu;
                {
                    const adai::gpu::ScopedCpuOnly cpu_only;
                    cpu = cpu_cached.next(tokens);
                }
                float cached_diff = 0.0f, cpu_diff = 0.0f, scale = 0.0f;
                for (int v = 0; v < vocab; ++v) {
                    if (!std::isfinite(cached[v]) || !std::isfinite(full[v])) {
                        throw std::runtime_error("non-finite logits at step " +
                                                 std::to_string(step));
                    }
                    cached_diff = std::max(cached_diff, std::fabs(cached[v] - full[v]));
                    cpu_diff = std::max(cpu_diff, std::fabs(cpu[v] - full[v]));
                    scale = std::max(scale, std::fabs(full[v]));
                }
                max_cached_diff = std::max(max_cached_diff, cached_diff);
                max_cpu_diff = std::max(max_cpu_diff, cpu_diff);
                // Rounding differences only: the two paths sum in different orders, nothing more.
                const float tolerance = 1e-3f * (1.0f + scale);
                const int choice = argmax(full);
                if (cached_diff > tolerance) {
                    ++failures;
                    std::printf("MISMATCH step %d: logits differ by %.6g (tolerance %.6g)\n", step,
                                cached_diff, tolerance);
                } else if (argmax(cached) == choice) {
                    ++cached_agree;
                } else if (top_two_gap(full) <= 2.0f * cached_diff) {
                    ++cached_near_ties;
                } else {
                    ++failures;
                    std::printf(
                        "MISMATCH step %d: gpu_full chose %d, gpu_cached %d (gap %.6g, "
                        "diff %.6g)\n",
                        step, choice, argmax(cached), top_two_gap(full), cached_diff);
                }
                if (argmax(cpu) == choice) {
                    ++cpu_agree;
                }
                tokens.push_back(choice);
            }
            std::printf(
                "VALIDATE steps=%d gpu_cached_agree=%d gpu_cached_near_ties=%d "
                "gpu_cached_max_logit_diff=%.3g cpu_agree=%d cpu_max_logit_diff=%.3g %s\n",
                length, cached_agree, cached_near_ties, max_cached_diff, cpu_agree, max_cpu_diff,
                failures == 0 ? "PASS" : "FAIL");
            std::fflush(stdout);
        }

        // Timing: each path generates on its own.
        for (const std::string& name : opt.paths) {
            const bool gpu = name.rfind("gpu_", 0) == 0;
            const bool cpu_only = name == "cpu_cached";
            DecodePath& path = name == "gpu_cached" ? gpu_cached
                               : name == "gpu_full" ? gpu_full
                                                    : cpu_cached;
            for (int length : opt.lengths) {
                std::vector<double> runs;
                for (int r = 0; r <= opt.repeats; ++r) {
                    std::unique_ptr<adai::gpu::ScopedCpuOnly> guard;
                    if (cpu_only) {
                        guard = std::make_unique<adai::gpu::ScopedCpuOnly>();
                    }
                    const auto start = std::chrono::steady_clock::now();
                    encode(gpu);
                    path.reset();
                    std::vector<int> tokens{bos};
                    for (int step = 0; step < length; ++step) {
                        tokens.push_back(argmax(path.next(tokens)));
                    }
                    const double ms = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - start)
                                          .count();
                    if (r > 0) {  // run 0 warms up
                        runs.push_back(ms);
                    }
                }
                std::sort(runs.begin(), runs.end());
                const double median = runs[runs.size() / 2];
                std::printf(
                    "RESULT path=%s length=%d median_ms=%.1f min_ms=%.1f max_ms=%.1f "
                    "per_token_ms=%.2f\n",
                    name.c_str(), length, median, runs.front(), runs.back(), median / length);
                std::fflush(stdout);
            }
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("FAILED: %s\n", e.what());
        std::printf("Last GPU work: %s\n", adai::gpu::describe_last_gpu_work().c_str());
        const bool usable = adai::gpu::GPUManager::device_usable();
        std::printf("%s\n", usable ? "the GPU still runs work"
                                   : "the GPU stopped running work (device lost)");
        return usable ? 1 : 2;
    }
#endif
}
