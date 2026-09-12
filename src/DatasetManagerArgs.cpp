// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

// Calling convention note: every parse_*_args() here takes the command's OWN argument slice —
// NOT including the command name itself (main() strips that off before dispatching, the same
// convention MnsCliCommands.hpp uses). E.g. for `dataset_manager assign my-model file1 --count 3`,
// parse_assign_args() receives {"my-model", "file1", "--count", "3"}.

#include "DatasetManagerArgs.hpp"
#include <sstream>

namespace adai {

namespace {
template <typename T>
T make_error(std::string message) {
    T r;
    r.error = true;
    r.error_message = std::move(message);
    return r;
}
}  // namespace

AddArgs parse_add_args(const std::vector<std::string>& args) {
    if (args.empty())
        return make_error<AddArgs>("Usage: dataset_manager add <data_file>");
    AddArgs r;
    r.data_file = args[0];
    return r;
}

GutenbergArgs parse_gutenberg_args(const std::vector<std::string>& args) {
    if (args.empty()) {
        return make_error<GutenbergArgs>(
            "Usage: dataset_manager gutenberg <book_id> [num_pairs] [model]");
    }
    GutenbergArgs r;
    try {
        r.book_id = std::stoi(args[0]);
    } catch (const std::exception&) {
        return make_error<GutenbergArgs>("Invalid book_id: " + args[0]);
    }
    if (args.size() >= 2) {
        try {
            r.num_pairs = std::stoi(args[1]);
        } catch (const std::exception&) {
            return make_error<GutenbergArgs>("Invalid num_pairs: " + args[1]);
        }
    }
    if (args.size() >= 3)
        r.model = args[2];
    return r;
}

GutenbergBatchArgs parse_gutenberg_batch_args(const std::vector<std::string>& args) {
    if (args.empty()) {
        return make_error<GutenbergBatchArgs>(
            "Usage: dataset_manager gutenberg-batch <id1,id2,id3,...> [num_pairs_each] [model]");
    }
    GutenbergBatchArgs r;
    std::stringstream ss(args[0]);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        try {
            r.book_ids.push_back(std::stoi(tok));
        } catch (const std::exception&) {
            return make_error<GutenbergBatchArgs>("Invalid book id in list: " + tok);
        }
    }
    if (args.size() >= 2) {
        try {
            r.num_pairs_each = std::stoi(args[1]);
        } catch (const std::exception&) {
            return make_error<GutenbergBatchArgs>("Invalid num_pairs_each: " + args[1]);
        }
    }
    if (args.size() >= 3)
        r.model = args[2];
    return r;
}

HuggingfaceArgs parse_huggingface_args(const std::vector<std::string>& args) {
    if (args.empty()) {
        return make_error<HuggingfaceArgs>(
            "Usage: dataset_manager huggingface <dataset_id> [num_pairs] [split] [input_field] "
            "[output_field] [model]");
    }
    HuggingfaceArgs r;
    r.dataset_id = args[0];
    if (args.size() >= 2) {
        try {
            r.num_pairs = std::stoi(args[1]);
        } catch (const std::exception&) {
            return make_error<HuggingfaceArgs>("Invalid num_pairs: " + args[1]);
        }
    }
    if (args.size() >= 3)
        r.split = args[2];
    if (args.size() >= 4)
        r.input_field = args[3];
    if (args.size() >= 5)
        r.output_field = args[4];
    if (args.size() >= 6)
        r.model = args[5];
    return r;
}

RemoveArgs parse_remove_args(const std::vector<std::string>& args) {
    if (args.empty())
        return make_error<RemoveArgs>("Usage: dataset_manager remove <data_file>");
    RemoveArgs r;
    r.target = args[0];
    return r;
}

AssignArgs parse_assign_args(const std::vector<std::string>& args) {
    if (args.empty()) {
        return make_error<AssignArgs>(
            "Usage: dataset_manager assign <model_name> [file1 file2 ...] [--count N]");
    }
    AssignArgs r;
    r.model_name = args[0];
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--count" && i + 1 < args.size()) {
            try {
                r.count = std::stoi(args[++i]);
            } catch (const std::exception&) {
                return make_error<AssignArgs>("Invalid --count value");
            }
        } else {
            r.targets.push_back(args[i]);
        }
    }
    return r;
}

UnassignArgs parse_unassign_args(const std::vector<std::string>& args) {
    if (args.empty()) {
        return make_error<UnassignArgs>(
            "Usage: dataset_manager unassign <model_name> [file1 file2 ...] [--force]");
    }
    UnassignArgs r;
    r.model_name = args[0];
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--force") {
            r.force = true;
        } else {
            r.targets.push_back(args[i]);
        }
    }
    return r;
}

DeleteArgs parse_delete_args(const std::vector<std::string>& args) {
    DeleteArgs r;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--force") {
            r.force = true;
        } else if (args[i] == "--delete-files") {
            r.delete_files = true;
        } else {
            r.targets.push_back(args[i]);
        }
    }
    if (r.targets.empty()) {
        return make_error<DeleteArgs>(
            "Usage: dataset_manager delete <file1> [file2 ...] [--force] [--delete-files]");
    }
    return r;
}

}  // namespace adai
