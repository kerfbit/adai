#include <cmath>
#include <iostream>
#include "src/DecoderBlock.hpp"
#include "src/EncoderBlock.hpp"
#include "src/LanguageModelHead.hpp"
#include "src/LayerNorm.hpp"
#include "src/TokenEmbedding.hpp"

bool test_layernorm_save_load() {
    std::cout << "\n=== Testing LayerNorm save/load ===" << std::endl;

    // Create and initialize LayerNorm
    LayerNorm ln1(128);

    // Modify some values
    Matrix gamma = ln1.get_gamma();
    Matrix beta = ln1.get_beta();
    for (int i = 0; i < 10; i++) {
        gamma(0, i) = 1.5f + i * 0.1f;
        beta(0, i) = -0.5f + i * 0.05f;
    }
    ln1.set_gamma(gamma);
    ln1.set_beta(beta);

    // Save
    ln1.save_weights("test_layernorm.bin");

    // Create new LayerNorm and load
    LayerNorm ln2(128);
    ln2.load_weights("test_layernorm.bin");

    // Verify
    const Matrix& loaded_gamma = ln2.get_gamma();
    const Matrix& loaded_beta = ln2.get_beta();

    bool success = true;
    for (int i = 0; i < 10; i++) {
        if (std::abs(gamma(0, i) - loaded_gamma(0, i)) > 1e-6f) {
            std::cout << "❌ Gamma mismatch at " << i << ": " << gamma(0, i) << " vs "
                      << loaded_gamma(0, i) << std::endl;
            success = false;
        }
        if (std::abs(beta(0, i) - loaded_beta(0, i)) > 1e-6f) {
            std::cout << "❌ Beta mismatch at " << i << ": " << beta(0, i) << " vs "
                      << loaded_beta(0, i) << std::endl;
            success = false;
        }
    }

    if (success) {
        std::cout << "✅ LayerNorm save/load successful" << std::endl;
    }
    return success;
}

bool test_tokenembedding_save_load() {
    std::cout << "\n=== Testing TokenEmbedding save/load ===" << std::endl;

    // Create and initialize TokenEmbedding
    TokenEmbedding te1(1000, 128);

    // Get embedding matrix and modify some values
    const Matrix& emb1 = te1.get_embeddings();

    // Save
    te1.save_weights("test_token_emb.bin");

    // Create new TokenEmbedding and load
    TokenEmbedding te2(1000, 128);
    te2.load_weights("test_token_emb.bin");

    // Verify
    const Matrix& emb2 = te2.get_embeddings();

    bool success = true;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (std::abs(emb1(i, j) - emb2(i, j)) > 1e-6f) {
                std::cout << "❌ Embedding mismatch at (" << i << "," << j << "): " << emb1(i, j)
                          << " vs " << emb2(i, j) << std::endl;
                success = false;
                goto done;
            }
        }
    }
done:
    if (success) {
        std::cout << "✅ TokenEmbedding save/load successful" << std::endl;
    }
    return success;
}

bool test_languagemodelhead_save_load() {
    std::cout << "\n=== Testing LanguageModelHead save/load ===" << std::endl;

    // Create and initialize LanguageModelHead
    LanguageModelHead lmh1(128, 1000);

    // Save
    lmh1.save_weights("test_lmh.bin");

    // Create new LanguageModelHead and load
    LanguageModelHead lmh2(128, 1000);
    lmh2.load_weights("test_lmh.bin");

    // Test with same input
    Matrix input(5, 128);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 128; j++) {
            input(i, j) = (i + j) * 0.01f;
        }
    }

    Matrix out1 = lmh1.forward(input);
    Matrix out2 = lmh2.forward(input);

    bool success = true;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 100; j++) {
            if (std::abs(out1(i, j) - out2(i, j)) > 1e-4f) {
                std::cout << "❌ Output mismatch at (" << i << "," << j << "): " << out1(i, j)
                          << " vs " << out2(i, j) << std::endl;
                success = false;
                goto done2;
            }
        }
    }
done2:
    if (success) {
        std::cout << "✅ LanguageModelHead save/load successful" << std::endl;
    }
    return success;
}

int main() {
    std::cout << "╔═══════════════════════════════════════╗" << std::endl;
    std::cout << "║  Weight Save/Load Functionality Test ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════╝" << std::endl;

    bool all_passed = true;

    all_passed &= test_layernorm_save_load();
    all_passed &= test_tokenembedding_save_load();
    all_passed &= test_languagemodelhead_save_load();

    std::cout << "\n═══════════════════════════════════════" << std::endl;
    if (all_passed) {
        std::cout << "🎉 All tests passed!" << std::endl;
    } else {
        std::cout << "❌ Some tests failed!" << std::endl;
    }
    std::cout << "═══════════════════════════════════════" << std::endl;

    return all_passed ? 0 : 1;
}
