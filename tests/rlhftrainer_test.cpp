// TD-038: RLHFTrainer is the first real integration driving RewardModel/PPOOptimizer against a
// live EncoderDecoderModel policy. These tests use a tiny model (small d_model/vocab/layers) so
// they run fast, and deliberately use "greedy" generation for determinism where the test needs
// to know exactly what response a prompt will produce.
#include <gtest/gtest.h>
#include <cmath>
#include "../src/EncoderDecoderModel.hpp"
#include "../src/RLHFTrainer.hpp"

namespace {

void build_test_vocab(BPETokenizer* tokenizer, int vocab_size) {
    std::vector<std::string> corpus = {
        "hello world",  "how are you",  "I am fine",     "thank you",
        "good morning", "good evening", "see you later", "goodbye",
        "this is good", "that is bad",  "very good job", "quite bad indeed",
    };
    tokenizer->build_vocab(corpus, vocab_size);
}

// Sum of log P(response_tokens[t] | prompt, response_tokens[0..t-1]) under the model's CURRENT
// weights -- a real, deterministic measure of "how likely is the model to produce this exact
// response", computed via the same teacher-forced forward() RLHFTrainer itself uses.
float total_log_prob(EncoderDecoderModel& model, const std::vector<int>& prompt_tokens,
                     const std::vector<int>& response_tokens) {
    Matrix logits = model.forward(prompt_tokens, response_tokens);
    int vocab_size = model.get_vocab_size();
    int steps = std::min(static_cast<int>(response_tokens.size()), logits.rows);
    float total = 0.0f;
    for (int t = 0; t < steps; ++t) {
        float max_logit = logits(t, 0);
        for (int v = 1; v < vocab_size; ++v) {
            max_logit = std::max(max_logit, logits(t, v));
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            sum_exp += std::exp(logits(t, v) - max_logit);
        }
        float log_prob = (logits(t, response_tokens[t]) - max_logit) - std::log(sum_exp);
        total += log_prob;
    }
    return total;
}

}  // namespace

TEST(RLHFTrainerTest, TrainRewardModelReducesLossOverEpochs) {
    int vocab_size = 100;
    EncoderDecoderModel model(vocab_size, /*d_model=*/32, /*encoder_layers=*/1, /*decoder_layers=*/1,
                              /*num_heads=*/2, /*d_ff=*/64);
    build_test_vocab(model.get_tokenizer(), vocab_size);

    RLHFConfig config;
    config.reward_model_learning_rate = 0.05f;
    RLHFTrainer trainer(model, config);

    std::vector<PreferenceExample> examples;
    for (int i = 0; i < 8; ++i) {
        examples.push_back({"how are you", "very good job", "quite bad indeed"});
    }

    float first_epoch_loss = trainer.train_reward_model_on_batch(examples, 1);
    float later_loss = trainer.train_reward_model_on_batch(examples, 30);

    EXPECT_GT(first_epoch_loss, 0.0f);
    EXPECT_LT(later_loss, first_epoch_loss)
        << "Bradley-Terry loss should decrease as the reward model learns chosen > rejected";
}

TEST(RLHFTrainerTest, RunIterationChangesPolicyWeights) {
    int vocab_size = 100;
    EncoderDecoderModel model(vocab_size, 32, 1, 1, 2, 64);
    build_test_vocab(model.get_tokenizer(), vocab_size);

    RLHFConfig config;
    config.generation_strategy = "greedy";
    config.max_response_length = 8;
    config.policy_learning_rate = 0.01f;
    RLHFTrainer trainer(model, config);
    // A minimally-trained reward model still produces a real (non-constant-zero) scalar signal.
    trainer.train_reward_model_on_batch({{"hello", "very good job", "quite bad indeed"}}, 5);

    std::vector<int> prompt_tokens = model.get_tokenizer()->encode("hello", true);
    std::string response_before =
        model.generate_response_with_strategy("hello", config.max_response_length, "greedy");
    std::vector<int> response_tokens = model.get_tokenizer()->encode(response_before, true);
    ASSERT_FALSE(response_tokens.empty());
    float log_prob_before = total_log_prob(model, prompt_tokens, response_tokens);

    RLHFStepResult result = trainer.run_iteration({"hello"});

    EXPECT_EQ(result.num_rollouts, 1);
    EXPECT_GT(result.num_response_tokens, 0);

    float log_prob_after = total_log_prob(model, prompt_tokens, response_tokens);
    EXPECT_NE(log_prob_before, log_prob_after)
        << "run_iteration() must have actually changed the policy's weights, not silently no-op'd";
}

// The core correctness claim: a POSITIVE advantage should make the response MORE likely under
// the policy, and a NEGATIVE advantage should make it LESS likely -- the opposite-sign check that
// proves apply_policy_gradient()'s gradient direction genuinely tracks the advantage's sign.
//
// This deliberately bypasses run_iteration()'s RewardModel/ValueFunction/GAE pipeline: those
// components are independently, randomly initialized, so on a cold start whether a given rollout
// nets a positive or negative GAE advantage is not reliably tied to which of two responses the
// reward model prefers (compute_gae() normalizes per-trajectory, so with gamma=lambda=1 and an
// end-loaded reward, advantage[t] = R - value[t] for an UNCALIBRATED value baseline -- its sign
// is arbitrary at init). Testing apply_policy_gradient() directly with caller-supplied, uniform
// advantages isolates exactly the mechanism this test cares about: does the gradient actually
// applied move log-prob in the direction the advantage's sign says it should.
TEST(RLHFTrainerTest, PositiveAdvantageIncreasesLogProbNegativeAdvantageDecreasesIt) {
    int vocab_size = 100;

    auto build_model = [vocab_size]() {
        auto model = std::make_unique<EncoderDecoderModel>(vocab_size, 32, 1, 1, 2, 64);
        build_test_vocab(model->get_tokenizer(), vocab_size);
        return model;
    };

    RLHFConfig config;
    config.generation_strategy = "greedy";
    config.max_response_length = 6;
    config.policy_learning_rate = 0.05f;

    // -- Positive-advantage case: a uniform +1.0 advantage at every response position should
    // increase the response's total log-prob.
    {
        auto model = build_model();
        RLHFTrainer trainer(*model, config);

        std::vector<int> prompt_tokens = model->get_tokenizer()->encode("hello", true);
        std::string response =
            model->generate_response_with_strategy("hello", config.max_response_length, "greedy");
        std::vector<int> response_tokens = model->get_tokenizer()->encode(response, true);
        ASSERT_FALSE(response_tokens.empty());

        std::vector<float> advantages(response_tokens.size(), 1.0f);

        float before = total_log_prob(*model, prompt_tokens, response_tokens);
        trainer.apply_policy_gradient(prompt_tokens, response_tokens, advantages);
        float after = total_log_prob(*model, prompt_tokens, response_tokens);

        EXPECT_GT(after, before) << "a positive advantage should make its response more likely "
                                    "after one policy-gradient step";
    }

    // -- Negative-advantage case: a uniform -1.0 advantage at every response position should
    // decrease the response's total log-prob.
    {
        auto model = build_model();
        RLHFTrainer trainer(*model, config);

        std::vector<int> prompt_tokens = model->get_tokenizer()->encode("hello", true);
        std::string response =
            model->generate_response_with_strategy("hello", config.max_response_length, "greedy");
        std::vector<int> response_tokens = model->get_tokenizer()->encode(response, true);
        ASSERT_FALSE(response_tokens.empty());

        std::vector<float> advantages(response_tokens.size(), -1.0f);

        float before = total_log_prob(*model, prompt_tokens, response_tokens);
        trainer.apply_policy_gradient(prompt_tokens, response_tokens, advantages);
        float after = total_log_prob(*model, prompt_tokens, response_tokens);

        EXPECT_LT(after, before) << "a negative advantage should make its response less likely "
                                    "after one policy-gradient step";
    }
}

TEST(RLHFTrainerTest, RunIterationOnEmptyPromptListIsANoOp) {
    int vocab_size = 100;
    EncoderDecoderModel model(vocab_size, 32, 1, 1, 2, 64);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    RLHFTrainer trainer(model);

    RLHFStepResult result = trainer.run_iteration({});

    EXPECT_EQ(result.num_rollouts, 0);
    EXPECT_EQ(result.num_response_tokens, 0);
}
