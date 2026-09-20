// Tests for src/IncrementalTrainerArgs.{hpp,cpp} — incremental_trainer's global argv parsing and
// a few small pure helpers (TD-035), extracted from IncrementalTrainingTool.cpp so they're
// testable without touching a real IncrementalTrainer (real training, forking, GPU init).
// IncrementalTrainer itself is already covered by IncrementalTrainerTests/
// IncrementalTrainerBackgroundTests/IncrementalTrainerControlTests/TrainerControlStateTests/
// TrainerAdminAPITests.

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "IncrementalTrainerArgs.hpp"

using adai::derive_run_id;
using adai::incremental_trainer_command_defers_gpu_init;
using adai::parse_incremental_trainer_global_args;
using adai::parse_reset_command_args;
using adai::resolve_dataset_kind;

namespace {
std::vector<char*> make_argv(std::vector<std::string>& storage) {
    std::vector<char*> argv;
    for (auto& s : storage)
        argv.push_back(s.data());
    return argv;
}
}  // namespace

TEST(ParseIncrementalTrainerGlobalArgs, NoArgsLeavesEverythingUnsetAndCommandListEmpty) {
    std::vector<std::string> raw = {"incremental_trainer"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_FALSE(r.config_path.has_value());
    EXPECT_FALSE(r.gpu_strategy.has_value());
    EXPECT_FALSE(r.model_name.has_value());
    EXPECT_FALSE(r.foreground);
    EXPECT_FALSE(r.admin_port.has_value());
    EXPECT_EQ(r.objective, "chatbot");
    EXPECT_FALSE(r.dataset_kind.has_value());
    EXPECT_TRUE(r.args.empty());
}

TEST(ParseIncrementalTrainerGlobalArgs, StripsGlobalFlagsAndKeepsCommandArgs) {
    std::vector<std::string> raw = {"incremental_trainer",
                                    "--config",
                                    "/tmp/c.conf",
                                    "--gpu-strategy",
                                    "full",
                                    "--model",
                                    "my-model",
                                    "--foreground",
                                    "--admin-port",
                                    "18432",
                                    "--objective=lejepa",
                                    "--dataset-kind",
                                    "world_model",
                                    "train",
                                    "5"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());

    ASSERT_TRUE(r.config_path.has_value());
    EXPECT_EQ(*r.config_path, "/tmp/c.conf");
    ASSERT_TRUE(r.gpu_strategy.has_value());
    EXPECT_EQ(*r.gpu_strategy, "full");
    ASSERT_TRUE(r.model_name.has_value());
    EXPECT_EQ(*r.model_name, "my-model");
    EXPECT_TRUE(r.foreground);
    ASSERT_TRUE(r.admin_port.has_value());
    EXPECT_EQ(*r.admin_port, 18432);
    EXPECT_EQ(r.objective, "lejepa");
    ASSERT_TRUE(r.dataset_kind.has_value());
    EXPECT_EQ(*r.dataset_kind, "world_model");
    ASSERT_EQ(r.args.size(), 2u);
    EXPECT_EQ(r.args[0], "train");
    EXPECT_EQ(r.args[1], "5");
}

TEST(ParseIncrementalTrainerGlobalArgs, DatasetKindFlagUnsetWhenAbsent) {
    // TD-202: absent --dataset-kind means "use the automatic objective->kind default computed
    // in IncrementalTrainingTool.cpp's main()" — this parser itself has no default to fall back
    // to, hence std::optional rather than a plain string.
    std::vector<std::string> raw = {"incremental_trainer", "train"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_FALSE(r.dataset_kind.has_value());
}

TEST(ParseIncrementalTrainerGlobalArgs, DatasetKindMissingItsValueFallsThroughAsPositional) {
    std::vector<std::string> raw = {"incremental_trainer", "train", "--dataset-kind"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_FALSE(r.dataset_kind.has_value());
    ASSERT_EQ(r.args.size(), 2u);
    EXPECT_EQ(r.args[0], "train");
    EXPECT_EQ(r.args[1], "--dataset-kind");
}

TEST(ParseIncrementalTrainerGlobalArgs, ObjectiveFlagDefaultsToChatbotWhenAbsent) {
    // TD-183 Action Item 3: the existing chatbot teacher-forcing objective must be unaffected
    // when --objective is never mentioned at all.
    std::vector<std::string> raw = {"incremental_trainer", "train"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_EQ(r.objective, "chatbot");
    ASSERT_EQ(r.args.size(), 1u);
    EXPECT_EQ(r.args[0], "train");
}

TEST(ParseIncrementalTrainerGlobalArgs, ObjectiveFlagUsesEqualsSyntaxNotSpaceSeparated) {
    // Deliberately different from every other flag's "--flag value" convention — matches the
    // literal "--objective=lejepa" syntax named in TD-183's own title.
    std::vector<std::string> raw = {"incremental_trainer", "--objective=lejepa", "train"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_EQ(r.objective, "lejepa");
    ASSERT_EQ(r.args.size(), 1u);
    EXPECT_EQ(r.args[0], "train");
}

TEST(ParseIncrementalTrainerGlobalArgs, ObjectiveFlagCanAppearInAnyOrderRelativeToTheCommand) {
    std::vector<std::string> raw = {"incremental_trainer", "train", "--objective=lejepa"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_EQ(r.objective, "lejepa");
    ASSERT_EQ(r.args.size(), 1u);
    EXPECT_EQ(r.args[0], "train");
}

TEST(ParseIncrementalTrainerGlobalArgs, BareObjectiveFlagWithNoEqualsSignFallsThroughAsPositional) {
    // No "=" at all -- doesn't match the "--objective=" prefix, so (like every other malformed
    // flag usage in this parser) it's collected as a plain positional instead of crashing or
    // silently consuming the next token.
    std::vector<std::string> raw = {"incremental_trainer", "--objective", "lejepa"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_EQ(r.objective, "chatbot");
    ASSERT_EQ(r.args.size(), 2u);
    EXPECT_EQ(r.args[0], "--objective");
    EXPECT_EQ(r.args[1], "lejepa");
}

TEST(ParseIncrementalTrainerGlobalArgs, AdminPortMissingItsValueFallsThroughAsPositional) {
    // Same "no following value" contract as --config (see
    // FlagMissingItsValueFallsThroughAsPositional below) — TD-172's supervisor always passes a
    // value, but a malformed manual invocation shouldn't crash on std::stoi with no argument.
    std::vector<std::string> raw = {"incremental_trainer", "--admin-port"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_FALSE(r.admin_port.has_value());
    ASSERT_EQ(r.args.size(), 1u);
    EXPECT_EQ(r.args[0], "--admin-port");
}

TEST(ParseIncrementalTrainerGlobalArgs, FlagsCanAppearInAnyOrderRelativeToTheCommand) {
    std::vector<std::string> raw = {"incremental_trainer", "status", "--foreground"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(r.foreground);
    ASSERT_EQ(r.args.size(), 1u);
    EXPECT_EQ(r.args[0], "status");
}

TEST(ParseIncrementalTrainerGlobalArgs, FlagMissingItsValueFallsThroughAsPositional) {
    // "--config" as the very last argument has no following value — the original inline loop's
    // `i + 1 < argc` guard means it falls through to the else branch and is collected as a plain
    // positional arg instead of consuming (nonexistent) input.
    std::vector<std::string> raw = {"incremental_trainer", "--config"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_FALSE(r.config_path.has_value());
    ASSERT_EQ(r.args.size(), 1u);
    EXPECT_EQ(r.args[0], "--config");
}

TEST(IncrementalTrainerCommandDefersGpuInit, TrainRetrainResumeDefer) {
    EXPECT_TRUE(incremental_trainer_command_defers_gpu_init("train"));
    EXPECT_TRUE(incremental_trainer_command_defers_gpu_init("retrain"));
    EXPECT_TRUE(incremental_trainer_command_defers_gpu_init("resume"));
}

TEST(IncrementalTrainerCommandDefersGpuInit, EveryOtherCommandInitsImmediately) {
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init("init"));
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init("reset"));
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init("status"));
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init("history"));
    // TD-172: "serve" removed — incremental_trainer no longer has an always-on service command.
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init("serve"));
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init("bogus"));
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init(""));
}

TEST(DeriveRunId, ReturnsConfiguredValueVerbatimWhenNonEmpty) {
    EXPECT_EQ(derive_run_id("my-configured-run"), "my-configured-run");
}

TEST(DeriveRunId, FallsBackToHostnamePidWhenConfiguredIsEmpty) {
    // Exact hostname/pid are environment-dependent — just verify the documented shape: a
    // non-empty string containing a single "_" separator, with the host portion capped at 8
    // characters (see the implementation's own truncation).
    auto id = derive_run_id("");
    EXPECT_FALSE(id.empty());
    auto underscore = id.find('_');
    ASSERT_NE(underscore, std::string::npos);
    EXPECT_LE(underscore, 8u) << "host portion must be truncated to at most 8 chars";
    EXPECT_GT(id.size(), underscore + 1) << "must have a non-empty pid suffix after the '_'";
}

TEST(DeriveRunId, IsStableAcrossRepeatedCallsInTheSameProcess) {
    // Same process -> same pid and hostname -> same derived id both times.
    EXPECT_EQ(derive_run_id(""), derive_run_id(""));
}

// TD-202/TD-204: resolve_dataset_kind() is the extracted, directly-testable form of the
// objective->kind mapping that was previously inlined in IncrementalTrainingTool.cpp's main()
// (untestable there) — the exact spot where a TD-203 "fix" introduced a regression (a
// config-file DATASET_KIND silently overriding the automatic per-objective default) that TD-204
// then reverted. These tests exist so that regression can't recur silently a third time.

TEST(ResolveDatasetKind, DefaultObjectiveMapsToChatbot) {
    EXPECT_EQ(resolve_dataset_kind("chatbot", std::nullopt), "chatbot");
}

TEST(ResolveDatasetKind, LejepaObjectiveMapsToWorldModel) {
    EXPECT_EQ(resolve_dataset_kind("lejepa", std::nullopt), "world_model");
}

TEST(ResolveDatasetKind, UnrecognizedObjectiveFallsBackToChatbot) {
    // Mirrors IncrementalTrainerGlobalArgs::objective's own "chatbot" default — only the
    // literal string "lejepa" selects world_model, matching TD-183's own contract.
    EXPECT_EQ(resolve_dataset_kind("something-else", std::nullopt), "chatbot");
}

TEST(ResolveDatasetKind, CliOverrideAlwaysWinsRegardlessOfObjective) {
    // Also the regression guard for the actual TD-203 bug: this function's signature has no
    // parameter through which a static config-file/env DATASET_KIND could leak in and silently
    // override the objective-based default — only a real, per-invocation `cli_override` can.
    EXPECT_EQ(resolve_dataset_kind("chatbot", std::optional<std::string>("encoder")), "encoder");
    EXPECT_EQ(resolve_dataset_kind("lejepa", std::optional<std::string>("chatbot")), "chatbot");
}

TEST(ParseResetCommandArgs, DefaultsBothFalse) {
    auto r = parse_reset_command_args({});
    EXPECT_FALSE(r.yes);
    EXPECT_FALSE(r.keep_data);
}

TEST(ParseResetCommandArgs, RecognizesBothFlagsInAnyOrder) {
    auto r1 = parse_reset_command_args({"--yes", "--keep-data"});
    EXPECT_TRUE(r1.yes);
    EXPECT_TRUE(r1.keep_data);

    auto r2 = parse_reset_command_args({"--keep-data", "--yes"});
    EXPECT_TRUE(r2.yes);
    EXPECT_TRUE(r2.keep_data);
}

TEST(ParseResetCommandArgs, UnrecognizedTokensAreIgnored) {
    auto r = parse_reset_command_args({"--bogus", "--yes"});
    EXPECT_TRUE(r.yes);
    EXPECT_FALSE(r.keep_data);
}
