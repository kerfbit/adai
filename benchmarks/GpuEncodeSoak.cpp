// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-13

// TD-574: a forward-only GPU soak. Runs LeJEPAEncoder::gpu_encode() over a set of texts for a given
// time: our own SYCL kernels (embedding gather, LayerNorm, attention, softmax, GELU) and oneMKL
// GEMMs, queued without waiting except every <wait_every> encodes, but no backward pass, no
// atomics and no optimizer. Used by the GPU test package to tell whether Arc engine resets need the
// training path or happen with forward kernels alone.
//
//   gpu_encode_soak <texts.jsonl> <seconds> [wait_every=64] [d_model=64] [heads=2] [d_ff=128]
//                   [layers=1] [max_seq=16]
//
// Each JSONL line's "input", "response" and "text" fields are texts; any other line is one text.
// Exit codes: 0 ran the whole time, 1 failed with the GPU still usable, 2 the GPU stopped running
// work (a lost device), 3 no GPU.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "BPETokenizer.hpp"
#include "LeJEPAEncoder.hpp"
#include "Matrix.hpp"
#include "gpu/GPUDebugSync.hpp"
#include "gpu/MatrixGPU.hpp"
#include "nlohmann/json.hpp"

namespace {

bool parse_int(const char* s, int& out) {
    try {
        size_t used = 0;
        const int v = std::stoi(s, &used);
        if (used != std::string(s).size() || v < 1) {
            return false;
        }
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<std::string> read_texts(const std::string& path) {
    std::vector<std::string> texts;
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_object()) {
            for (const char* key : {"input", "response", "text"}) {
                if (j.contains(key) && j[key].is_string()) {
                    texts.push_back(j[key].get<std::string>());
                }
            }
        } else {
            texts.push_back(line);
        }
    }
    return texts;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <texts.jsonl> <seconds> [wait_every=64] [d_model=64] [heads=2] "
                     "[d_ff=128] [layers=1] [max_seq=16]\n",
                     argv[0]);
        return 1;
    }
    int seconds = 0;
    int dims[6] = {64, 64, 2, 128, 1, 16};  // wait_every, d_model, heads, d_ff, layers, max_seq
    if (!parse_int(argv[2], seconds)) {
        std::fprintf(stderr, "seconds must be a whole number of at least 1\n");
        return 1;
    }
    for (int a = 3; a < argc && a < 9; ++a) {
        if (!parse_int(argv[a], dims[a - 3])) {
            std::fprintf(stderr, "argument %d must be a whole number of at least 1\n", a);
            return 1;
        }
    }
    const int wait_every = dims[0], d_model = dims[1], heads = dims[2], d_ff = dims[3],
              layers = dims[4], max_seq = dims[5];

    std::vector<std::string> texts;
    try {
        texts = read_texts(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (texts.empty()) {
        std::fprintf(stderr, "no texts in %s\n", argv[1]);
        return 1;
    }

#ifndef ADAI_ENABLE_GPU
    std::fprintf(stderr, "built without GPU support\n");
    return 3;
#else
    if (!Matrix::gpu_try_initialize(0, 0.3f, true)) {
        std::fprintf(stderr, "no GPU device\n");
        return 3;
    }
    std::printf("%s\n", Matrix::gpu_info().c_str());
    if (const std::string debug = adai::gpu::gpu_debug_summary(); !debug.empty()) {
        std::printf("%s\n", debug.c_str());
    }

    BPETokenizer tokenizer;
    tokenizer.build_vocab(texts, 200);
    const std::string vocab_file = "gpu_encode_soak.vocab";
    tokenizer.save_vocab(vocab_file);
    LeJEPAEncoder encoder(static_cast<int>(tokenizer.get_vocab_size()), d_model, layers, heads,
                          d_ff, max_seq, 16);
    encoder.load_tokenizer_vocab(vocab_file);
    std::printf(
        "encoder: vocab %zu, d_model %d, heads %d, d_ff %d, layers %d, max_seq %d; "
        "%zu texts; a wait every %d encodes; %d s\n",
        tokenizer.get_vocab_size(), d_model, heads, d_ff, layers, max_seq, texts.size(), wait_every,
        seconds);

    adai::gpu::GPUManager::debug_sync_point("encode");
    const auto start = std::chrono::steady_clock::now();
    const auto until = start + std::chrono::seconds(seconds);
    long long encodes = 0;
    long long waits = 0;
    try {
        while (std::chrono::steady_clock::now() < until) {
            for (int k = 0; k < wait_every; ++k) {
                adai::gpu::GPUMatrix x = encoder.gpu_encode(texts[encodes % texts.size()]);
                ++encodes;
                if (k == wait_every - 1) {
                    std::vector<float> h(static_cast<size_t>(x.rows) * x.cols);
                    x.download(h.data(), static_cast<int>(h.size()));
                    ++waits;
                    for (float v : h) {
                        if (!std::isfinite(v)) {
                            throw std::runtime_error("non-finite encoder output");
                        }
                    }
                }
            }
            if (waits % 100 == 0) {
                std::printf("%lld encodes, %.1f s\n", encodes,
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                                .count());
                std::fflush(stdout);
            }
        }
    } catch (const std::exception& e) {
        std::printf("FAILED after %lld encodes: %s\n", encodes, e.what());
        std::printf("Last GPU work: %s\n", adai::gpu::describe_last_gpu_work().c_str());
        const bool usable = adai::gpu::GPUManager::device_usable();
        std::printf("%s\n", usable ? "the GPU still runs work"
                                   : "the GPU stopped running work (device lost)");
        return usable ? 1 : 2;
    }
    std::printf("OK: %lld encodes, %lld waits in %d s\n", encodes, waits, seconds);
    return 0;
#endif
}
