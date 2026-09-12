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
    ASSERT_EQ(r.args.size(), 2u);
    EXPECT_EQ(r.args[0], "train");
    EXPECT_EQ(r.args[1], "5");
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

TEST(IncrementalTrainerCommandDefersGpuInit, TrainRetrainResumeServeDefer) {
    EXPECT_TRUE(incremental_trainer_command_defers_gpu_init("train"));
    EXPECT_TRUE(incremental_trainer_command_defers_gpu_init("retrain"));
    EXPECT_TRUE(incremental_trainer_command_defers_gpu_init("resume"));
    EXPECT_TRUE(incremental_trainer_command_defers_gpu_init("serve"));
}

TEST(IncrementalTrainerCommandDefersGpuInit, EveryOtherCommandInitsImmediately) {
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init("init"));
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init("reset"));
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init("status"));
    EXPECT_FALSE(incremental_trainer_command_defers_gpu_init("history"));
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
