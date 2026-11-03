// Tests for src/IncrementalTrainerArgs.{hpp,cpp} — incremental_trainer's global argv parsing and
// a few small pure helpers (TD-035), extracted from IncrementalTrainingTool.cpp so they're
// testable without touching a real IncrementalTrainer (real training, forking, GPU init).
// IncrementalTrainer itself is already covered by IncrementalTrainerTests/
// IncrementalTrainerBackgroundTests/IncrementalTrainerControlTests/TrainerControlStateTests/
// TrainerAdminAPITests.

#include <gtest/gtest.h>
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "IncrementalTrainerArgs.hpp"

using adai::derive_run_id;
using adai::incremental_trainer_command_defers_gpu_init;
using adai::parse_incremental_trainer_global_args;
using adai::parse_reset_command_args;
using adai::parse_trainer_service_args;
using adai::trainer_service_child_argv;
using adai::persistent_run_id;
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

TEST(ParseIncrementalTrainerGlobalArgs, FlagsCanAppearInAnyOrderRelativeToTheCommand) {
    std::vector<std::string> raw = {"incremental_trainer", "status", "--foreground"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(r.foreground);
    ASSERT_EQ(r.args.size(), 1u);
    EXPECT_EQ(r.args[0], "status");
}

// TD-401: a global flag missing its value used to become a positional argument (`status --config`
// ran with normal config discovery); now it's an error naming the flag.
TEST(ParseIncrementalTrainerGlobalArgs, AGlobalFlagMissingItsValueIsAnError) {
    for (const std::string flag : {"--config", "--gpu-strategy", "--model", "--admin-port", "--dataset-kind"}) {
        std::vector<std::string> trailing = {"incremental_trainer", "status", flag};
        auto argv = make_argv(trailing);
        auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_TRUE(r.error) << flag;
        EXPECT_EQ(r.error_message, "Missing value for " + flag);

        std::vector<std::string> followed = {"incremental_trainer", flag, "--foreground", "resume"};
        argv = make_argv(followed);
        r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_TRUE(r.error) << flag << " took the next flag as its value";
    }
}

// TD-401: std::stoi aborted the process on "abc" and took "8084x" as 8084; no range was checked.
TEST(ParseIncrementalTrainerGlobalArgs, AdminPortMustBeAWholePortNumber) {
    for (const std::string bad : {"abc", "8084x", "0", "-1", "65536", "99999", "", "99999999999"}) {
        std::vector<std::string> raw = {"incremental_trainer", "--admin-port", bad, "resume"};
        auto argv = make_argv(raw);
        auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_TRUE(r.error) << "'" << bad << "'";
        EXPECT_NE(r.error_message.find("--admin-port"), std::string::npos);
        EXPECT_FALSE(r.admin_port.has_value());
    }
    for (const std::string good : {"1", "8084", "65535"}) {
        std::vector<std::string> raw = {"incremental_trainer", "--admin-port", good, "resume"};
        auto argv = make_argv(raw);
        auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_FALSE(r.error) << good << ": " << r.error_message;
        ASSERT_TRUE(r.admin_port.has_value());
        EXPECT_EQ(*r.admin_port, std::stoi(good));
    }
}

// TD-401: a mistyped objective (`--objective=lejpa`) silently ran chatbot training.
TEST(ParseIncrementalTrainerGlobalArgs, ObjectiveMustBeChatbotOrLejepa) {
    for (const std::string bad : {"--objective=lejpa", "--objective=", "--objective=LeJEPA", "--objective=world_model"}) {
        std::vector<std::string> raw = {"incremental_trainer", bad, "train"};
        auto argv = make_argv(raw);
        auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_TRUE(r.error) << bad;
        EXPECT_NE(r.error_message.find("chatbot or lejepa"), std::string::npos) << r.error_message;
    }
    std::vector<std::string> raw = {"incremental_trainer", "--objective=chatbot", "train"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(r.objective, "chatbot");
}

// A bare --objective (no "=") used to become a positional, so `--objective lejepa train` ran the
// command "--objective"; now it says how to write the flag.
TEST(ParseIncrementalTrainerGlobalArgs, BareObjectiveFlagSaysToUseEquals) {
    std::vector<std::string> raw = {"incremental_trainer", "--objective", "lejepa", "train"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(r.error);
    EXPECT_NE(r.error_message.find("--objective=lejepa"), std::string::npos) << r.error_message;
}

// Command flags (reset's --yes/--keep-data) aren't global flags and still reach the command.
TEST(ParseIncrementalTrainerGlobalArgs, CommandFlagsPassThroughToTheCommand) {
    std::vector<std::string> raw = {"incremental_trainer", "reset", "--yes", "--keep-data"};
    auto argv = make_argv(raw);
    auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    EXPECT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(r.args, (std::vector<std::string>{"reset", "--yes", "--keep-data"}));
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

// TD-404: the host was cut to 8 characters and the PID to its last 4 digits, so two hosts sharing a prefix
// (ai-machine-1, ai-machine-2) could derive the same id. It's now the full hostname, the PID and 8 random hex
// digits, so even two calls differ; callers that need a stable id use persistent_run_id().
TEST(DeriveRunId, IsTheFullHostnamePidAndARandomPart) {
    char host[256] = {};
    ASSERT_EQ(::gethostname(host, sizeof(host) - 1), 0);
    std::string expected_host = host;
    for (char& c : expected_host)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '.') c = '-';
    const std::string id = derive_run_id("");
    const std::string prefix = expected_host + "_" + std::to_string(::getpid()) + "_";
    ASSERT_EQ(id.rfind(prefix, 0), 0u) << id;
    const std::string random = id.substr(prefix.size());
    EXPECT_EQ(random.size(), 8u) << id;
    EXPECT_EQ(random.find_first_not_of("0123456789abcdef"), std::string::npos) << id;
    EXPECT_NE(derive_run_id(""), derive_run_id(""));
}

TEST(PersistentRunId, TheChatbotIdHasItsOwnFile) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("adai_td404_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    const std::string chatbot = persistent_run_id("", dir.string(), "chatbot");
    EXPECT_EQ(chatbot.rfind("chatbot-", 0), 0u) << chatbot;
    EXPECT_TRUE(fs::exists(dir / "chatbot_run_id"));
    EXPECT_EQ(persistent_run_id("", dir.string(), "chatbot"), chatbot);
    EXPECT_NE(persistent_run_id("", dir.string()), chatbot) << "the LeJEPA id is kept separately";
    fs::remove_all(dir);
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
    // literal string "lejepa" selects world_model, matching TD-183's own contract. Since TD-401
    // the parser refuses any other --objective value, so this fallback is never reached from argv.
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

// TD-523: trainer_service's reading of a pass's exit code. Exit 1 used to mean both "nothing
// pending" and "failed"; now 3 is idle and 1 a failure.
TEST(ClassifyPassExit, MapsResumeCodesSignalsAndOtherFailures) {
    EXPECT_EQ(adai::classify_pass_exit(adai::kResumeExitTrained), adai::PassOutcome::DidWork);
    EXPECT_EQ(adai::classify_pass_exit(adai::kResumeExitNothingPending), adai::PassOutcome::Idle);
    EXPECT_EQ(adai::classify_pass_exit(adai::kResumeExitFailed), adai::PassOutcome::Failed);
    EXPECT_EQ(adai::classify_pass_exit(2), adai::PassOutcome::Failed);
    EXPECT_EQ(adai::classify_pass_exit(127), adai::PassOutcome::Failed);  // exec failure
    EXPECT_EQ(adai::classify_pass_exit(128), adai::PassOutcome::Crashed);
    EXPECT_EQ(adai::classify_pass_exit(139), adai::PassOutcome::Crashed);
}

TEST(FailureBackoffSeconds, DoublesFromTheBaseAndCaps) {
    EXPECT_EQ(adai::failure_backoff_seconds(1, 45, 900), 45);
    EXPECT_EQ(adai::failure_backoff_seconds(2, 45, 900), 90);
    EXPECT_EQ(adai::failure_backoff_seconds(3, 45, 900), 180);
    EXPECT_EQ(adai::failure_backoff_seconds(5, 45, 900), 720);
    EXPECT_EQ(adai::failure_backoff_seconds(6, 45, 900), 900);
    EXPECT_EQ(adai::failure_backoff_seconds(1000, 45, 900), 900);
}

// TD-557: trainer_service ignored --help (it started the service), the parser's own errors,
// --foreground/--admin-port and stray positionals.
TEST(ParseTrainerServiceArgs, HelpAnywhereIsHelp) {
    for (const std::vector<std::string>& extra : std::vector<std::vector<std::string>>{
             {"--help"}, {"-h"}, {"--config", "c.conf", "--help"}, {"bogus", "-h"}}) {
        std::vector<std::string> raw = {"trainer_service"};
        raw.insert(raw.end(), extra.begin(), extra.end());
        auto argv = make_argv(raw);
        EXPECT_TRUE(parse_trainer_service_args(static_cast<int>(argv.size()), argv.data()).help) << extra.back();
    }
}

TEST(ParseTrainerServiceArgs, AcceptsThePassThroughFlags) {
    std::vector<std::string> raw = {"trainer_service", "--config", "c.conf", "--model", "m", "--gpu-strategy", "full"};
    auto argv = make_argv(raw);
    const auto r = parse_trainer_service_args(static_cast<int>(argv.size()), argv.data());
    ASSERT_FALSE(r.global.error) << r.global.error_message;
    EXPECT_FALSE(r.help);
    EXPECT_EQ(r.global.config_path.value_or(""), "c.conf");
    EXPECT_EQ(r.global.model_name.value_or(""), "m");
    EXPECT_EQ(r.global.gpu_strategy.value_or(""), "full");
}

TEST(ParseTrainerServiceArgs, RejectsChildFlagsPositionalsAndParserErrors) {
    const std::vector<std::pair<std::vector<std::string>, std::string>> cases = {
        {{"--foreground"}, "--foreground is set by trainer_service"},
        {{"--admin-port", "8085"}, "--admin-port is set by trainer_service"},
        {{"resume"}, "Unexpected argument 'resume' (trainer_service takes no command)"},
        {{"--config"}, "Missing value for --config"},
        {{"--admin-port", "abc"}, "Invalid value for --admin-port"},
    };
    for (const auto& [extra, message] : cases) {
        std::vector<std::string> raw = {"trainer_service"};
        raw.insert(raw.end(), extra.begin(), extra.end());
        auto argv = make_argv(raw);
        const auto r = parse_trainer_service_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_TRUE(r.global.error) << extra.front();
        EXPECT_EQ(r.global.error_message.rfind(message, 0), 0u) << r.global.error_message;
    }
}

// TD-402: --dataset-kind was taken as given; LocalTransport joins it into a path, so "chatbt" selected a
// new, empty pool and "../x" put the registry files outside session_dir.
TEST(ParseIncrementalTrainerGlobalArgs, DatasetKindMustBeOneOfTheFourKinds) {
    for (const std::string kind : {"encoder", "decoder", "world_model", "chatbot"}) {
        std::vector<std::string> raw = {"incremental_trainer", "--dataset-kind", kind, "status"};
        auto argv = make_argv(raw);
        const auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_FALSE(r.error) << kind << ": " << r.error_message;
        EXPECT_EQ(r.dataset_kind.value_or(""), kind);
    }
    for (const std::string kind : {"chatbt", "../escape", "a/b", "Chatbot", ""}) {
        std::vector<std::string> raw = {"incremental_trainer", "--dataset-kind", kind, "status"};
        auto argv = make_argv(raw);
        const auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_TRUE(r.error) << "'" << kind << "' was accepted";
        EXPECT_EQ(r.error_message, "Invalid value for --dataset-kind: '" + kind +
                                       "' (expected encoder, decoder, world_model or chatbot)");
    }
}

// TD-403: every pass gets --objective and --dataset-kind; they used to be dropped.
TEST(TrainerServiceChildArgv, AlwaysPassesTheObjectiveAndTheDatasetKind) {
    adai::IncrementalTrainerGlobalArgs cli;  // defaults: chatbot, no kind
    EXPECT_EQ(trainer_service_child_argv("/opt/adai/incremental_trainer", cli, 8085),
              (std::vector<std::string>{"/opt/adai/incremental_trainer", "--objective=chatbot", "--dataset-kind",
                                        "chatbot", "--foreground", "--admin-port", "8085", "resume"}));

    cli.config_path = "/etc/adai/config.trainer.conf";
    cli.model_name = "m";
    cli.gpu_strategy = "full";
    cli.objective = "lejepa";
    EXPECT_EQ(trainer_service_child_argv("it", cli, 9),
              (std::vector<std::string>{"it", "--config", "/etc/adai/config.trainer.conf", "--model", "m",
                                        "--gpu-strategy", "full", "--objective=lejepa", "--dataset-kind",
                                        "world_model", "--foreground", "--admin-port", "9", "resume"}));

    cli.dataset_kind = "encoder";
    const auto argv = trainer_service_child_argv("it", cli, 9);
    const auto kind = std::find(argv.begin(), argv.end(), "--dataset-kind");
    ASSERT_NE(kind, argv.end());
    ASSERT_NE(kind + 1, argv.end());
    EXPECT_EQ(*(kind + 1), "encoder");  // an explicit kind beats the objective's default
}

// TD-415: the LeJEPA pass's run id was PID-based, so after a crash its claimed files stayed claimed by an id
// no later process had. It's now kept in <dir>/lejepa_run_id.
TEST(PersistentRunId, ConfiguredWinsElseTheIdIsCreatedOnceAndReused) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("adai_td415_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    EXPECT_EQ(persistent_run_id("configured-run", dir.string()), "configured-run");
    EXPECT_FALSE(fs::exists(dir / "lejepa_run_id")) << "RUN_ID needs no file";

    const std::string first = persistent_run_id("", dir.string());  // creates the directory too
    EXPECT_EQ(first.rfind("lejepa-", 0), 0u) << first;
    EXPECT_EQ(persistent_run_id("", dir.string()), first) << "a later pass must get the same id";
    std::ifstream in(dir / "lejepa_run_id");
    std::string saved;
    std::getline(in, saved);
    EXPECT_EQ(saved, first);
    fs::remove_all(dir);

    // A directory that can't be created (a file is in the way): still an id, no throw.
    const fs::path blocker = fs::temp_directory_path() / ("adai_td415_file_" + std::to_string(::getpid()));
    std::ofstream(blocker) << "x";
    EXPECT_EQ(persistent_run_id("", (blocker / "sub").string()).rfind("lejepa-", 0), 0u);
    fs::remove(blocker);
}

// TD-416: `train abc` aborted through an uncaught std::stoi, and `train 5x` trained 5 epochs.
TEST(ParseEpochsArg, AWholePositiveNumberOrTheDefault) {
    std::string error;
    EXPECT_EQ(adai::parse_epochs_arg({"train"}, 7, error), 7);
    EXPECT_EQ(adai::parse_epochs_arg({"train", "3"}, 7, error), 3);
    for (const std::string bad : {"abc", "5x", "0", "-2", ""}) {
        error.clear();
        EXPECT_FALSE(adai::parse_epochs_arg({"retrain", bad}, 7, error).has_value()) << bad;
        EXPECT_EQ(error, "Invalid epoch count '" + bad + "' (expected a whole number, at least 1)");
    }
    error.clear();
    EXPECT_FALSE(adai::parse_epochs_arg({"train", "3", "4"}, 7, error).has_value());
    EXPECT_EQ(error, "Unexpected argument '4' (train takes one epoch count)");
}

// TD-524: --admin-port only when the admin API is on.
TEST(TrainerServiceChildArgv, NoAdminPortWithoutTheAdminApi) {
    adai::IncrementalTrainerGlobalArgs cli;
    EXPECT_EQ(trainer_service_child_argv("it", cli, std::nullopt),
              (std::vector<std::string>{"it", "--objective=chatbot", "--dataset-kind", "chatbot", "--foreground",
                                        "resume"}));
}

// TD-524: the sibling comes from the running executable's own directory, not argv[0] or PATH.
TEST(SiblingExecutablePath, UsesTheRunningExecutablesDirectory) {
    namespace fs = std::filesystem;
    const fs::path self = fs::read_symlink("/proc/self/exe");
    EXPECT_EQ(adai::sibling_executable_path("bare_name", "incremental_trainer"),
              (self.parent_path() / "incremental_trainer").string());
}

// TD-431: each trainer_service pass logs to its own file; trainer_service and every pass used to share and rotate one.
TEST(PassLogPath, AddsPassBeforeTheExtension) {
    EXPECT_EQ(adai::pass_log_path("/var/log/adai/trainer.log"), "/var/log/adai/trainer.pass.log");
    EXPECT_EQ(adai::pass_log_path("trainer"), "trainer.pass");
    EXPECT_EQ(adai::pass_log_path("logs/trainer.out.log"), "logs/trainer.out.pass.log");
}

TEST(TrainerServiceChildArgv, PassesThePassLogFile) {
    adai::IncrementalTrainerGlobalArgs cli;
    const auto argv = trainer_service_child_argv("it", cli, std::nullopt, "/var/log/adai/trainer.pass.log");
    const auto flag = std::find(argv.begin(), argv.end(), "--log-file");
    ASSERT_NE(flag, argv.end());
    ASSERT_NE(flag + 1, argv.end());
    EXPECT_EQ(*(flag + 1), "/var/log/adai/trainer.pass.log");
    const auto without = trainer_service_child_argv("it", cli, std::nullopt);
    EXPECT_EQ(std::find(without.begin(), without.end(), "--log-file"), without.end());
}

TEST(ParseIncrementalTrainerGlobalArgs, LogFileFlag) {
    std::vector<std::string> raw = {"incremental_trainer", "--log-file", "/tmp/x.log", "status"};
    auto argv = make_argv(raw);
    const auto r = parse_incremental_trainer_global_args(static_cast<int>(argv.size()), argv.data());
    ASSERT_FALSE(r.error) << r.error_message;
    EXPECT_EQ(r.log_file.value_or(""), "/tmp/x.log");

    std::vector<std::string> service = {"trainer_service", "--log-file", "/tmp/x.log"};
    auto service_argv = make_argv(service);
    const auto s = parse_trainer_service_args(static_cast<int>(service_argv.size()), service_argv.data());
    EXPECT_TRUE(s.global.error) << "trainer_service takes LOG_FILE_PATH, not --log-file";
}

TEST(DefaultBackgroundLogPath, UsesTheXdgStateDirectory) {
    namespace fs = std::filesystem;
    const fs::path state = fs::temp_directory_path() / ("adai_td431_" + std::to_string(::getpid()));
    const char* saved = std::getenv("XDG_STATE_HOME");
    const std::string saved_value = saved ? saved : "";
    ::setenv("XDG_STATE_HOME", state.c_str(), 1);
    EXPECT_EQ(adai::default_background_log_path(), (state / "adai" / "log" / "incremental_trainer.log").string());
    EXPECT_TRUE(fs::is_directory(state / "adai" / "log"));
    if (saved) ::setenv("XDG_STATE_HOME", saved_value.c_str(), 1); else ::unsetenv("XDG_STATE_HOME");
    fs::remove_all(state);
}
