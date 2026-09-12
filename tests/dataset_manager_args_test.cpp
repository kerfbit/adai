// Tests for src/DatasetManagerArgs.{hpp,cpp} — dataset_manager's per-command argument parsing
// (TD-035), extracted from DatasetManagerTool.cpp so it's testable without a real DatasetRegistry
// (real file I/O) or DataFetcher (real network downloads). DatasetRegistry's own behavior is
// already covered by DatasetRegistryTests/DatasetRegistryLiveTests — this file only covers
// parsing each command's own argv slice (NOT including the command name itself, same convention
// as MnsCliCommands.hpp).

#include <gtest/gtest.h>
#include "DatasetManagerArgs.hpp"

using namespace adai;

TEST(ParseAddArgs, RequiresDataFile) {
    EXPECT_TRUE(parse_add_args({}).error);
}

TEST(ParseAddArgs, ParsesDataFile) {
    auto r = parse_add_args({"myfile.jsonl"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.data_file, "myfile.jsonl");
}

TEST(ParseGutenbergArgs, RequiresBookId) {
    EXPECT_TRUE(parse_gutenberg_args({}).error);
}

TEST(ParseGutenbergArgs, InvalidBookIdIsAnErrorNotACrash) {
    // Regression: the original inline code called std::stoi(args[1]) completely unguarded — a
    // non-numeric book id would throw std::invalid_argument uncaught, crashing the whole
    // process instead of printing a normal usage/error message like every other bad-input path.
    auto r = parse_gutenberg_args({"not-a-number"});
    EXPECT_TRUE(r.error);
    EXPECT_NE(r.error_message.find("not-a-number"), std::string::npos);
}

TEST(ParseGutenbergArgs, DefaultsNumPairsTo500AndModelEmpty) {
    auto r = parse_gutenberg_args({"1342"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.book_id, 1342);
    EXPECT_EQ(r.num_pairs, 500);
    EXPECT_EQ(r.model, "");
}

TEST(ParseGutenbergArgs, ParsesNumPairsAndModel) {
    auto r = parse_gutenberg_args({"1342", "300", "my-model"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.book_id, 1342);
    EXPECT_EQ(r.num_pairs, 300);
    EXPECT_EQ(r.model, "my-model");
}

TEST(ParseGutenbergBatchArgs, RequiresIdList) {
    EXPECT_TRUE(parse_gutenberg_batch_args({}).error);
}

TEST(ParseGutenbergBatchArgs, SplitsCommaSeparatedIds) {
    auto r = parse_gutenberg_batch_args({"1342,11,84,1661"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.book_ids, (std::vector<int>{1342, 11, 84, 1661}));
    EXPECT_EQ(r.num_pairs_each, 500);
}

TEST(ParseGutenbergBatchArgs, InvalidIdInListIsAnError) {
    auto r = parse_gutenberg_batch_args({"1342,not-a-number,84"});
    EXPECT_TRUE(r.error);
}

TEST(ParseGutenbergBatchArgs, ParsesNumPairsEachAndModel) {
    auto r = parse_gutenberg_batch_args({"1342,11", "200", "my-model"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.num_pairs_each, 200);
    EXPECT_EQ(r.model, "my-model");
}

TEST(ParseHuggingfaceArgs, RequiresDatasetId) {
    EXPECT_TRUE(parse_huggingface_args({}).error);
}

TEST(ParseHuggingfaceArgs, DefaultsMatchOriginalBehavior) {
    auto r = parse_huggingface_args({"daily_dialog"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.dataset_id, "daily_dialog");
    EXPECT_EQ(r.num_pairs, 500);
    EXPECT_EQ(r.split, "train");
    EXPECT_EQ(r.input_field, "");
    EXPECT_EQ(r.output_field, "");
    EXPECT_EQ(r.model, "");
}

TEST(ParseHuggingfaceArgs, ParsesAllSixPositionalArgs) {
    auto r = parse_huggingface_args(
        {"tatsu-lab/alpaca", "300", "validation", "instruction", "output", "my-model"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.dataset_id, "tatsu-lab/alpaca");
    EXPECT_EQ(r.num_pairs, 300);
    EXPECT_EQ(r.split, "validation");
    EXPECT_EQ(r.input_field, "instruction");
    EXPECT_EQ(r.output_field, "output");
    EXPECT_EQ(r.model, "my-model");
}

TEST(ParseRemoveArgs, RequiresTarget) {
    EXPECT_TRUE(parse_remove_args({}).error);
}

TEST(ParseRemoveArgs, ParsesTarget) {
    auto r = parse_remove_args({"some/file.jsonl"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.target, "some/file.jsonl");
}

TEST(ParseAssignArgs, RequiresModelName) {
    EXPECT_TRUE(parse_assign_args({}).error);
}

TEST(ParseAssignArgs, NoFilesOrCountMeansAssignAll) {
    auto r = parse_assign_args({"my-model"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.model_name, "my-model");
    EXPECT_TRUE(r.targets.empty());
    EXPECT_EQ(r.count, 0);
}

TEST(ParseAssignArgs, ExplicitFilesAreCollectedAsTargets) {
    auto r = parse_assign_args({"my-model", "file1.jsonl", "file2.jsonl"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.targets, (std::vector<std::string>{"file1.jsonl", "file2.jsonl"}));
}

TEST(ParseAssignArgs, CountFlagIsParsed) {
    auto r = parse_assign_args({"my-model", "--count", "5"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.count, 5);
    EXPECT_TRUE(r.targets.empty());
}

TEST(ParseAssignArgs, InvalidCountValueIsAnError) {
    auto r = parse_assign_args({"my-model", "--count", "not-a-number"});
    EXPECT_TRUE(r.error);
}

TEST(ParseUnassignArgs, RequiresModelName) {
    EXPECT_TRUE(parse_unassign_args({}).error);
}

TEST(ParseUnassignArgs, ForceFlagAndTargetsBothParsed) {
    auto r = parse_unassign_args({"my-model", "file1.jsonl", "--force"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.model_name, "my-model");
    EXPECT_EQ(r.targets, (std::vector<std::string>{"file1.jsonl"}));
    EXPECT_TRUE(r.force);
}

TEST(ParseUnassignArgs, DefaultsForceFalseAndNoTargets) {
    auto r = parse_unassign_args({"my-model"});
    ASSERT_FALSE(r.error);
    EXPECT_FALSE(r.force);
    EXPECT_TRUE(r.targets.empty());
}

TEST(ParseDeleteArgs, RequiresAtLeastOneTarget) {
    EXPECT_TRUE(parse_delete_args({}).error);
    EXPECT_TRUE(parse_delete_args({"--force"}).error) << "flags alone are not a target";
}

TEST(ParseDeleteArgs, ParsesFlagsAndTargetsInAnyOrder) {
    auto r = parse_delete_args({"file1.jsonl", "--force", "file2.jsonl", "--delete-files"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.targets, (std::vector<std::string>{"file1.jsonl", "file2.jsonl"}));
    EXPECT_TRUE(r.force);
    EXPECT_TRUE(r.delete_files);
}

TEST(ParseDeleteArgs, DefaultsFlagsFalse) {
    auto r = parse_delete_args({"file1.jsonl"});
    ASSERT_FALSE(r.error);
    EXPECT_FALSE(r.force);
    EXPECT_FALSE(r.delete_files);
}
