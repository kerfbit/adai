#include <cmath>
#include <iostream>
#include <vector>
#include "BPETokenizer.hpp"
#include "Matrix.hpp"
#include "TextGenerator.hpp"

/**
 * TextGenerator Example
 *
 * Demonstrates various text generation strategies using the TextGenerator class.
 * This example uses a mock language model for demonstration purposes.
 */

// Mock language model for demonstration
// In production, this would be replaced with a real decoder/LLM
class MockLanguageModel {
   private:
    int vocab_size;
    int d_model;

   public:
    MockLanguageModel(int vocab, int model_dim) : vocab_size(vocab), d_model(model_dim) {}

    // Simplified forward pass - returns random logits for demonstration
    // In a real model, this would use actual decoder layers
    Matrix forward(const std::vector<int>& input_tokens) {
        int seq_len = static_cast<int>(input_tokens.size());
        Matrix logits(seq_len, vocab_size);

        // Generate pseudo-realistic logits based on token context
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < vocab_size; ++j) {
                // Simple pattern: favor certain tokens based on context
                float logit = -5.0f;  // Base (low) probability

                // Favor lowercase letters (tokens 4-29)
                if (j >= 4 && j < 30) {
                    logit = -2.0f + (std::sin(i * 0.5f + j * 0.3f) * 1.0f);
                }

                // Higher probability for space after several tokens
                if (j == 30 && i % 4 == 3) {
                    logit = 0.5f;
                }

                // Favor <eos> token after 10+ tokens
                if (j == 3 && seq_len > 10) {
                    logit = 1.0f;
                }

                // Context-dependent: if last token was 'h', favor 'e'
                if (i > 0 && input_tokens[i - 1] == 10 && j == 7) {
                    logit = 2.0f;
                }

                logits.data[i][j] = logit;
            }
        }

        return logits;
    }
};

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

void print_tokens(const std::vector<int>& tokens, const std::string& label) {
    std::cout << label << ": [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "TextGenerator Example - Autoregressive Text Generation\n";
    std::cout << "======================================================\n\n";

    // Setup
    const int VOCAB_SIZE = 100;
    const int D_MODEL = 64;

    MockLanguageModel model(VOCAB_SIZE, D_MODEL);

    // Create model forward function
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    // Initialize tokenizer (simplified for demo)
    BPETokenizer tokenizer;

    // ========================================================================
    // Example 1: Greedy Decoding
    // ========================================================================
    print_section("Example 1: Greedy Decoding");
    std::cout << "Strategy: Always select the highest probability token\n";
    std::cout << "Characteristics: Deterministic, fast, may be repetitive\n\n";

    {
        TextGenerator::GenerationConfig config;
        config.max_length = 20;
        config.temperature = 0.0f;  // Greedy

        TextGenerator generator(config, 42);

        std::vector<int> prompt = {2, 10, 5};  // <bos>, some tokens
        std::vector<int> generated = generator.generate_greedy(model_fn, prompt);

        print_tokens(prompt, "Prompt tokens");
        print_tokens(generated, "Generated (greedy)");
        std::cout << "Length: " << generated.size() << " tokens\n";
    }

    // ========================================================================
    // Example 2: Temperature Sampling
    // ========================================================================
    print_section("Example 2: Temperature Sampling");
    std::cout << "Strategy: Sample from probability distribution with temperature scaling\n";
    std::cout << "  - Low temp (0.3): Conservative, focused\n";
    std::cout << "  - Medium temp (1.0): Balanced\n";
    std::cout << "  - High temp (2.0): Creative, random\n\n";

    std::vector<int> base_prompt = {2, 10, 5, 8};

    for (float temp : {0.3f, 1.0f, 2.0f}) {
        TextGenerator::GenerationConfig config;
        config.max_length = 15;
        config.temperature = temp;

        TextGenerator generator(config, 42);

        std::vector<int> generated = generator.generate_sampling(model_fn, base_prompt, temp);

        std::cout << "Temperature " << temp << ": ";
        print_tokens(generated, "");
    }

    // ========================================================================
    // Example 3: Top-k Sampling
    // ========================================================================
    print_section("Example 3: Top-k Sampling");
    std::cout << "Strategy: Sample from only the k most likely tokens\n";
    std::cout << "Prevents sampling from very low probability tokens\n\n";

    for (int k : {5, 10, 20}) {
        TextGenerator::GenerationConfig config;
        config.max_length = 15;
        config.temperature = 1.0f;
        config.top_k = k;

        TextGenerator generator(config, 42);

        std::vector<int> generated = generator.generate_top_k(model_fn, base_prompt, k);

        std::cout << "Top-k (k=" << k << "): ";
        print_tokens(generated, "");
    }

    // ========================================================================
    // Example 4: Nucleus (Top-p) Sampling
    // ========================================================================
    print_section("Example 4: Nucleus (Top-p) Sampling");
    std::cout << "Strategy: Sample from smallest set with cumulative prob >= p\n";
    std::cout << "Dynamically adapts to probability distribution\n\n";

    for (float p : {0.5f, 0.9f, 0.95f}) {
        TextGenerator::GenerationConfig config;
        config.max_length = 15;
        config.temperature = 1.0f;
        config.top_p = p;

        TextGenerator generator(config, 42);

        std::vector<int> generated = generator.generate_nucleus(model_fn, base_prompt, p);

        std::cout << "Nucleus (p=" << p << "): ";
        print_tokens(generated, "");
    }

    // ========================================================================
    // Example 5: Beam Search
    // ========================================================================
    print_section("Example 5: Beam Search");
    std::cout << "Strategy: Maintain multiple hypotheses, select best overall sequence\n";
    std::cout << "Good for tasks requiring high quality (translation, summarization)\n\n";

    for (int num_beams : {1, 3, 5}) {
        TextGenerator::GenerationConfig config;
        config.max_length = 15;
        config.num_beams = num_beams;
        config.length_penalty = true;
        config.length_penalty_alpha = 0.6f;

        TextGenerator generator(config, 42);

        std::vector<int> generated =
            generator.generate_beam_search(model_fn, base_prompt, num_beams);

        std::cout << "Beam search (beams=" << num_beams << "): ";
        print_tokens(generated, "");
    }

    // ========================================================================
    // Example 6: Combined Sampling (Top-k + Top-p + Temperature)
    // ========================================================================
    print_section("Example 6: Combined Sampling");
    std::cout << "Strategy: Apply multiple filters sequentially\n";
    std::cout << "Temperature -> Top-k -> Top-p -> Repetition Penalty -> Sample\n\n";

    {
        TextGenerator::GenerationConfig config;
        config.max_length = 20;
        config.temperature = 0.8f;
        config.top_k = 10;
        config.top_p = 0.9f;
        config.repetition_penalty = 1.2f;

        TextGenerator generator(config, 42);

        std::vector<int> generated = generator.generate(model_fn, base_prompt);

        std::cout << "Combined sampling:\n";
        std::cout << "  - Temperature: " << config.temperature << "\n";
        std::cout << "  - Top-k: " << config.top_k << "\n";
        std::cout << "  - Top-p: " << config.top_p << "\n";
        std::cout << "  - Repetition penalty: " << config.repetition_penalty << "\n\n";
        print_tokens(generated, "Generated");
    }

    // ========================================================================
    // Example 7: Repetition Penalty Comparison
    // ========================================================================
    print_section("Example 7: Repetition Penalty");
    std::cout << "Strategy: Penalize tokens that have already been generated\n";
    std::cout << "Reduces repetitive output\n\n";

    for (float penalty : {1.0f, 1.2f, 1.5f}) {
        TextGenerator::GenerationConfig config;
        config.max_length = 20;
        config.temperature = 0.8f;
        config.repetition_penalty = penalty;

        TextGenerator generator(config, 42);

        std::vector<int> generated = generator.generate(model_fn, base_prompt);

        std::cout << "Repetition penalty " << penalty << ": ";
        print_tokens(generated, "");
    }

    // ========================================================================
    // Example 8: Length Control
    // ========================================================================
    print_section("Example 8: Length Control");
    std::cout << "Strategy: Control generation length with max_length and min_length\n\n";

    for (int max_len : {10, 20, 30}) {
        TextGenerator::GenerationConfig config;
        config.max_length = max_len;
        config.temperature = 0.8f;

        TextGenerator generator(config, 42);

        std::vector<int> generated = generator.generate(model_fn, base_prompt);

        std::cout << "Max length " << max_len << " -> Generated " << generated.size()
                  << " tokens: ";
        print_tokens(generated, "");
    }

    // ========================================================================
    // Example 9: Unconditional Generation
    // ========================================================================
    print_section("Example 9: Unconditional Generation");
    std::cout << "Strategy: Generate from scratch (no prompt)\n";
    std::cout << "Model starts with <bos> token only\n\n";

    {
        TextGenerator::GenerationConfig config;
        config.max_length = 15;
        config.temperature = 1.0f;

        TextGenerator generator(config, 42);

        std::vector<int> generated = generator.generate(model_fn, {});

        print_tokens(generated, "Unconditional generation");
        std::cout << "Note: Started with <bos> token (ID=" << config.bos_token_id << ")\n";
    }

    // ========================================================================
    // Example 10: Multiple Samples (Diversity)
    // ========================================================================
    print_section("Example 10: Multiple Samples with Same Prompt");
    std::cout << "Strategy: Generate multiple diverse outputs from same prompt\n";
    std::cout << "Using sampling (temperature=1.0) for diversity\n\n";

    {
        TextGenerator::GenerationConfig config;
        config.max_length = 15;
        config.temperature = 1.0f;
        config.top_p = 0.9f;

        std::cout << "Generating 3 samples from same prompt:\n\n";

        for (int i = 0; i < 3; ++i) {
            // Use different seed for each sample
            TextGenerator generator(config, 42 + i);

            std::vector<int> generated = generator.generate(model_fn, base_prompt);

            std::cout << "Sample " << (i + 1) << ": ";
            print_tokens(generated, "");
        }
    }

    // ========================================================================
    // Example 11: Configuration Management
    // ========================================================================
    print_section("Example 11: Configuration Management");
    std::cout << "Strategy: Update configuration dynamically\n\n";

    {
        TextGenerator generator;

        std::cout << "Default configuration:\n";
        auto default_config = generator.get_config();
        std::cout << "  Max length: " << default_config.max_length << "\n";
        std::cout << "  Temperature: " << default_config.temperature << "\n";
        std::cout << "  Top-k: " << default_config.top_k << "\n";
        std::cout << "  Top-p: " << default_config.top_p << "\n\n";

        // Update configuration
        TextGenerator::GenerationConfig new_config;
        new_config.max_length = 25;
        new_config.temperature = 0.7f;
        new_config.top_k = 15;
        new_config.top_p = 0.85f;

        generator.set_config(new_config);

        std::cout << "Updated configuration:\n";
        auto updated = generator.get_config();
        std::cout << "  Max length: " << updated.max_length << "\n";
        std::cout << "  Temperature: " << updated.temperature << "\n";
        std::cout << "  Top-k: " << updated.top_k << "\n";
        std::cout << "  Top-p: " << updated.top_p << "\n";
    }

    // ========================================================================
    // Summary
    // ========================================================================
    print_section("Summary of Generation Strategies");

    std::cout << "1. GREEDY DECODING\n";
    std::cout << "   - Best for: Fast inference, deterministic output\n";
    std::cout << "   - Trade-off: May be repetitive or boring\n\n";

    std::cout << "2. TEMPERATURE SAMPLING\n";
    std::cout << "   - Best for: Controlling creativity vs. coherence\n";
    std::cout << "   - Trade-off: High temp = creative but incoherent\n\n";

    std::cout << "3. TOP-K SAMPLING\n";
    std::cout << "   - Best for: Preventing unlikely tokens\n";
    std::cout << "   - Trade-off: Fixed k may be too restrictive/permissive\n\n";

    std::cout << "4. NUCLEUS (TOP-P) SAMPLING\n";
    std::cout << "   - Best for: Adaptive filtering based on distribution\n";
    std::cout << "   - Trade-off: More computation than top-k\n\n";

    std::cout << "5. BEAM SEARCH\n";
    std::cout << "   - Best for: High-quality output (translation, summarization)\n";
    std::cout << "   - Trade-off: Slower, can be less diverse\n\n";

    std::cout << "6. COMBINED SAMPLING\n";
    std::cout << "   - Best for: Production chatbots, creative writing\n";
    std::cout << "   - Trade-off: Many hyperparameters to tune\n\n";

    std::cout << "\nRECOMMENDATIONS:\n";
    std::cout << "  - Chatbots: temp=0.7-0.9, top_p=0.9, repetition_penalty=1.1-1.3\n";
    std::cout << "  - Translation: beam_search with num_beams=4-5\n";
    std::cout << "  - Creative writing: temp=1.0-1.2, top_p=0.95\n";
    std::cout << "  - Code generation: temp=0.2-0.5 (more deterministic)\n";
    std::cout << "  - Question answering: greedy or temp=0.1 (factual)\n\n";

    std::cout << "Example completed successfully!\n";

    return 0;
}
