#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

// TD-035: dataset_manager's per-command argument parsing, pulled out of
// DatasetManagerTool.cpp so it's testable without touching a real DatasetRegistry/DataFetcher
// (real file I/O, network downloads) — see DatasetManagerArgs_test.cpp. The registry/fetcher
// calls themselves stay in DatasetManagerTool.cpp's main(), unchanged; DatasetRegistry's own
// behavior is already covered by DatasetRegistryTests/DatasetRegistryLiveTests.

#include <optional>
#include <string>
#include <vector>

namespace adai {

struct AddArgs {
    std::string data_file;
    bool error = false;
    std::string error_message;
};
AddArgs parse_add_args(const std::vector<std::string>& args);

struct GutenbergArgs {
    int book_id = 0;
    int num_pairs = 500;
    std::string model;
    bool error = false;
    std::string error_message;
};
GutenbergArgs parse_gutenberg_args(const std::vector<std::string>& args);

struct GutenbergBatchArgs {
    std::vector<int> book_ids;
    int num_pairs_each = 500;
    std::string model;
    bool error = false;
    std::string error_message;
};
GutenbergBatchArgs parse_gutenberg_batch_args(const std::vector<std::string>& args);

struct HuggingfaceArgs {
    std::string dataset_id;
    int num_pairs = 500;
    std::string split = "train";
    std::string input_field;
    std::string output_field;
    std::string model;
    bool error = false;
    std::string error_message;
};
HuggingfaceArgs parse_huggingface_args(const std::vector<std::string>& args);

struct RemoveArgs {
    std::string target;
    bool error = false;
    std::string error_message;
};
RemoveArgs parse_remove_args(const std::vector<std::string>& args);

struct AssignArgs {
    std::string model_name;
    std::vector<std::string> targets;
    int count = 0;
    bool error = false;
    std::string error_message;
};
AssignArgs parse_assign_args(const std::vector<std::string>& args);

struct UnassignArgs {
    std::string model_name;
    std::vector<std::string> targets;
    bool force = false;
    bool error = false;
    std::string error_message;
};
UnassignArgs parse_unassign_args(const std::vector<std::string>& args);

struct DeleteArgs {
    std::vector<std::string> targets;
    bool force = false;
    bool delete_files = false;
    bool error = false;
    std::string error_message;
};
DeleteArgs parse_delete_args(const std::vector<std::string>& args);

}  // namespace adai
