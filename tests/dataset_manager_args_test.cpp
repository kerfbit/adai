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

// TD-321: a count of 0 or less used to select every pending file, so `--count -2` assigned the
// whole pool; a partial number like "3x" was read as 3.
TEST(ParseAssignArgs, CountBelowOneOrPartialIsAnError) {
    for (const char* value : {"0", "-2", "3x", ""}) {
        auto r = parse_assign_args({"my-model", "--count", value});
        EXPECT_TRUE(r.error) << value;
        EXPECT_NE(r.error_message.find("omit --count"), std::string::npos) << r.error_message;
    }
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

// ============================================================================
// TD-202: is_valid_dataset_kind / parse_migrate_args
// ============================================================================

TEST(IsValidDatasetKind, AcceptsExactlyMnsKindVocabulary) {
    EXPECT_TRUE(is_valid_dataset_kind("encoder"));
    EXPECT_TRUE(is_valid_dataset_kind("decoder"));
    EXPECT_TRUE(is_valid_dataset_kind("world_model"));
    EXPECT_TRUE(is_valid_dataset_kind("chatbot"));
}

TEST(IsValidDatasetKind, RejectsEmptyAndUnknownValues) {
    EXPECT_FALSE(is_valid_dataset_kind(""));
    EXPECT_FALSE(is_valid_dataset_kind("Encoder"))
        << "case-sensitive, matching MNS's own kind check";
    EXPECT_FALSE(is_valid_dataset_kind("not_a_kind"));
}

TEST(ParseMigrateArgs, RequiresKind) {
    EXPECT_TRUE(parse_migrate_args({}).error);
}

TEST(ParseMigrateArgs, RejectsInvalidKind) {
    auto r = parse_migrate_args({"not_a_kind"});
    EXPECT_TRUE(r.error);
    EXPECT_NE(r.error_message.find("not_a_kind"), std::string::npos);
}

TEST(ParseMigrateArgs, NoFilesOrCountMeansMigrateAll) {
    auto r = parse_migrate_args({"world_model"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.kind, "world_model");
    EXPECT_TRUE(r.targets.empty());
    EXPECT_EQ(r.count, 0);
}

TEST(ParseMigrateArgs, ExplicitFilesAreCollectedAsTargets) {
    auto r = parse_migrate_args({"chatbot", "file1.jsonl", "file2.jsonl"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.kind, "chatbot");
    EXPECT_EQ(r.targets, (std::vector<std::string>{"file1.jsonl", "file2.jsonl"}));
}

TEST(ParseMigrateArgs, CountFlagIsParsed) {
    auto r = parse_migrate_args({"encoder", "--count", "5"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.kind, "encoder");
    EXPECT_EQ(r.count, 5);
    EXPECT_TRUE(r.targets.empty());
}

TEST(ParseMigrateArgs, InvalidCountValueIsAnError) {
    auto r = parse_migrate_args({"decoder", "--count", "not-a-number"});
    EXPECT_TRUE(r.error);
}

// TD-321: as for assign, `migrate chatbot --count 0` used to migrate every legacy entry.
TEST(ParseMigrateArgs, CountBelowOneOrPartialIsAnError) {
    for (const char* value : {"0", "-2", "3x", ""}) {
        auto r = parse_migrate_args({"chatbot", "--count", value});
        EXPECT_TRUE(r.error) << value;
        EXPECT_NE(r.error_message.find("omit --count"), std::string::npos) << r.error_message;
    }
}

// ============================================================================
// TD-205: parse_segment_args
// ============================================================================

TEST(ParseSegmentArgs, RequiresPath) {
    EXPECT_TRUE(parse_segment_args({}).error);
}

TEST(ParseSegmentArgs, RequiresEitherCountOrRanges) {
    auto r = parse_segment_args({"data.jsonl"});
    EXPECT_TRUE(r.error);
}

TEST(ParseSegmentArgs, CountAndRangesAreMutuallyExclusive) {
    auto r = parse_segment_args({"data.jsonl", "--count", "3", "--ranges", "0-9"});
    EXPECT_TRUE(r.error);
}

TEST(ParseSegmentArgs, ParsesCount) {
    auto r = parse_segment_args({"data.jsonl", "--count", "3"});
    ASSERT_FALSE(r.error);
    EXPECT_EQ(r.path, "data.jsonl");
    EXPECT_EQ(r.split_count, 3);
    EXPECT_TRUE(r.ranges.empty());
}

TEST(ParseSegmentArgs, InvalidCountValueIsAnError) {
    auto r = parse_segment_args({"data.jsonl", "--count", "not-a-number"});
    EXPECT_TRUE(r.error);
}

TEST(ParseSegmentArgs, ParsesSingleRange) {
    auto r = parse_segment_args({"data.jsonl", "--ranges", "0-99"});
    ASSERT_FALSE(r.error);
    ASSERT_EQ(r.ranges.size(), 1u);
    EXPECT_EQ(r.ranges[0].start, 0);
    EXPECT_EQ(r.ranges[0].count, 100);
}

TEST(ParseSegmentArgs, ParsesMultipleCommaSeparatedRanges) {
    auto r = parse_segment_args({"data.jsonl", "--ranges", "0-99,100-149,150-150"});
    ASSERT_FALSE(r.error);
    ASSERT_EQ(r.ranges.size(), 3u);
    EXPECT_EQ(r.ranges[0].start, 0);
    EXPECT_EQ(r.ranges[0].count, 100);
    EXPECT_EQ(r.ranges[1].start, 100);
    EXPECT_EQ(r.ranges[1].count, 50);
    EXPECT_EQ(r.ranges[2].start, 150);
    EXPECT_EQ(r.ranges[2].count, 1);
}

TEST(ParseSegmentArgs, RejectsRangeWithEndBeforeStart) {
    auto r = parse_segment_args({"data.jsonl", "--ranges", "10-5"});
    EXPECT_TRUE(r.error);
}

TEST(ParseSegmentArgs, RejectsRangeWithNegativeStart) {
    auto r = parse_segment_args({"data.jsonl", "--ranges", "-1-5"});
    EXPECT_TRUE(r.error);
}

TEST(ParseSegmentArgs, RejectsMalformedRangeToken) {
    auto r = parse_segment_args({"data.jsonl", "--ranges", "not-a-range-at-all-just-text"});
    EXPECT_TRUE(r.error);
}

TEST(ParseSegmentArgs, RejectsUnrecognizedArgument) {
    auto r = parse_segment_args({"data.jsonl", "--bogus"});
    EXPECT_TRUE(r.error);
}

// requeue (--all | <file> ...) [--model <name>]
TEST(DatasetManagerArgsTest, RequeueArgs) {
    auto all = parse_requeue_args({"--all", "--model", "chat-1"});
    ASSERT_FALSE(all.error) << all.error_message;
    EXPECT_TRUE(all.all);
    EXPECT_EQ(all.model_name, "chat-1");
    EXPECT_TRUE(all.targets.empty());

    auto files = parse_requeue_args({"a.jsonl", "b.jsonl"});
    ASSERT_FALSE(files.error) << files.error_message;
    EXPECT_EQ(files.targets, (std::vector<std::string>{"a.jsonl", "b.jsonl"}));
    EXPECT_FALSE(files.all);

    EXPECT_TRUE(parse_requeue_args({}).error) << "needs --all or files";
    EXPECT_TRUE(parse_requeue_args({"--all", "a.jsonl"}).error) << "not both";
    EXPECT_TRUE(parse_requeue_args({"--all", "--model"}).error) << "--model needs a value";
    EXPECT_TRUE(parse_requeue_args({"--all", "--modle", "x"}).error) << "unknown flag";
}

// TD-322: an unknown option, or --count with no value, used to become a file target, so a typo'd
// flag was silently treated as a file name.
TEST(DatasetManagerArgsOptions, UnknownOptionsAreErrorsNotFileTargets) {
    const auto del = parse_delete_args({"f.jsonl", "--delete-file"});
    EXPECT_TRUE(del.error) << "typo of --delete-files";
    EXPECT_NE(del.error_message.find("--delete-file"), std::string::npos) << del.error_message;
    EXPECT_TRUE(parse_unassign_args({"m", "--froce"}).error);
    EXPECT_TRUE(parse_assign_args({"m", "--cuont", "3"}).error);
    EXPECT_TRUE(parse_assign_args({"m", "--count=3"}).error);
    EXPECT_TRUE(parse_migrate_args({"chatbot", "--all"}).error);
    EXPECT_TRUE(parse_assign_args({"--count", "3"}).error) << "a flag is not a model name";
}

TEST(DatasetManagerArgsOptions, CountNeedsAValue) {
    const auto assign = parse_assign_args({"m", "--count"});
    EXPECT_TRUE(assign.error);
    EXPECT_NE(assign.error_message.find("needs a value"), std::string::npos)
        << assign.error_message;
    EXPECT_TRUE(parse_migrate_args({"chatbot", "--count"}).error);
    EXPECT_TRUE(parse_segment_args({"f.jsonl", "--count"}).error);
    EXPECT_TRUE(parse_segment_args({"f.jsonl", "--ranges"}).error);
}

TEST(DatasetManagerArgsOptions, DoubleDashEndsOptions) {
    const auto del = parse_delete_args({"--force", "--", "--odd-name.jsonl"});
    ASSERT_FALSE(del.error) << del.error_message;
    EXPECT_TRUE(del.force);
    EXPECT_EQ(del.targets, (std::vector<std::string>{"--odd-name.jsonl"}));
    const auto assign = parse_assign_args({"m", "--count", "2", "--", "-x.jsonl"});
    ASSERT_FALSE(assign.error) << assign.error_message;
    EXPECT_EQ(assign.count, 2);
    EXPECT_EQ(assign.targets, (std::vector<std::string>{"-x.jsonl"}));
}

// TD-323: bare std::stoi kept the leading digits ("1342abc" -> 1342) and let 0 and negatives in.
TEST(DatasetManagerArgsNumbers, PartialAndNonPositiveNumbersAreErrors) {
    EXPECT_TRUE(parse_gutenberg_args({"1342abc"}).error);
    EXPECT_TRUE(parse_gutenberg_args({"1342", "500x"}).error);
    EXPECT_TRUE(parse_gutenberg_args({"-5"}).error);
    EXPECT_TRUE(parse_gutenberg_args({"0"}).error);
    EXPECT_TRUE(parse_gutenberg_args({"1342", "-10"}).error);
    EXPECT_TRUE(parse_gutenberg_args({"1342", "0"}).error);
    EXPECT_TRUE(parse_gutenberg_batch_args({"1342,0"}).error);
    EXPECT_TRUE(parse_gutenberg_batch_args({"1342,11x"}).error);
    EXPECT_TRUE(parse_gutenberg_batch_args({"1342,11", "-1"}).error);
    EXPECT_TRUE(parse_huggingface_args({"x", "-1"}).error);
    EXPECT_TRUE(parse_huggingface_args({"x", "10k"}).error);
    EXPECT_TRUE(parse_segment_args({"f.jsonl", "--count", "3abc"}).error);
    EXPECT_TRUE(parse_segment_args({"f.jsonl", "--count", "0"}).error);
    EXPECT_TRUE(parse_segment_args({"f.jsonl", "--ranges", "0-9x"}).error);
    EXPECT_TRUE(parse_segment_args({"f.jsonl", "--ranges", "0x-9"}).error);
}

TEST(DatasetManagerArgsNumbers, WholePositiveNumbersStillParse) {
    const auto g = parse_gutenberg_args({"1342", "250"});
    ASSERT_FALSE(g.error) << g.error_message;
    EXPECT_EQ(g.book_id, 1342);
    EXPECT_EQ(g.num_pairs, 250);
    const auto b = parse_gutenberg_batch_args({"1342,11,84", "20"});
    ASSERT_FALSE(b.error) << b.error_message;
    EXPECT_EQ(b.book_ids, (std::vector<int>{1342, 11, 84}));
    const auto s = parse_segment_args({"f.jsonl", "--ranges", "0-9,10-19"});
    ASSERT_FALSE(s.error) << s.error_message;
    EXPECT_EQ(s.ranges.size(), 2u);
}
