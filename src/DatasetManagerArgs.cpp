// @adai-status: experimental
// @adai-version: 0.7.0
// @adai-reviewed: 2026-11-07

// Calling convention note: every parse_*_args() here takes the command's OWN argument slice —
// NOT including the command name itself (main() strips that off before dispatching, the same
// convention MnsCliCommands.hpp uses). E.g. for `dataset_manager assign my-model file1 --count 3`,
// parse_assign_args() receives {"my-model", "file1", "--count", "3"}.

#include "DatasetManagerArgs.hpp"
#include "Config.hpp"  // is_dataset_kind() (TD-402)
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

// TD-321: assign's and migrate's --count. A count of 0 or less used to fall through to "every
// entry", so `--count -2` or `--count 0` assigned (or migrated) the whole pending pool; only
// omitting --count means all now. The value must be a whole number (parse_strict_int()).
bool parse_count(const std::string& value, int& out) {
    const auto n = parse_strict_int(value);
    if (!n || *n < 1)
        return false;
    out = *n;
    return true;
}

std::string count_error(const std::string& value, const char* all_action) {
    return "Invalid --count value '" + value +
           "': it must be a whole number of at least 1 (omit --count to " + all_action + ")";
}

// TD-323: book ids and pair counts. Bare std::stoi kept the leading digits of "1342abc" and let 0
// and negative numbers through, which in local mode just wrote zero pairs.
bool parse_positive(const std::string& value, int& out) {
    const auto n = parse_strict_int(value);
    if (!n || *n < 1)
        return false;
    out = *n;
    return true;
}

// TD-322: an argument that looks like an option but isn't one of the command's own is an error;
// it used to become a file target, so `delete f --delete-file` (typo) deleted the entry and kept
// the file without a word. "--" ends the options, for a file whose name starts with "-".
bool looks_like_option(const std::string& arg) {
    return arg.size() > 1 && arg[0] == '-';
}

template <typename T>
T unknown_option(const std::string& arg, const char* usage) {
    return make_error<T>("Unknown option " + arg + ". " + usage);
}

template <typename T>
T missing_value(const char* flag, const char* usage) {
    return make_error<T>(std::string(flag) + " needs a value. " + usage);
}
}  // namespace

// TODO: See TD-325 in TECHNICAL_DEBT.md - extra arguments silently ignored
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
    if (!parse_positive(args[0], r.book_id)) {
        return make_error<GutenbergArgs>("Invalid book_id '" + args[0] +
                                         "': it must be a whole number of at least 1");
    }
    if (args.size() >= 2 && !parse_positive(args[1], r.num_pairs)) {
        return make_error<GutenbergArgs>("Invalid num_pairs '" + args[1] +
                                         "': it must be a whole number of at least 1");
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
        int id = 0;
        if (!parse_positive(tok, id)) {
            return make_error<GutenbergBatchArgs>("Invalid book id in list: '" + tok +
                                                  "' (each must be a whole number of at least 1)");
        }
        r.book_ids.push_back(id);
    }
    if (args.size() >= 2 && !parse_positive(args[1], r.num_pairs_each)) {
        return make_error<GutenbergBatchArgs>("Invalid num_pairs_each '" + args[1] +
                                              "': it must be a whole number of at least 1");
    }
    if (args.size() >= 3)
        r.model = args[2];
    return r;
}

// TODO: See TD-325 in TECHNICAL_DEBT.md - purely positional optionals shift silently (e.g. model lands in split)
HuggingfaceArgs parse_huggingface_args(const std::vector<std::string>& args) {
    if (args.empty()) {
        return make_error<HuggingfaceArgs>(
            "Usage: dataset_manager huggingface <dataset_id> [num_pairs] [split] [input_field] "
            "[output_field] [model]");
    }
    HuggingfaceArgs r;
    r.dataset_id = args[0];
    if (args.size() >= 2 && !parse_positive(args[1], r.num_pairs)) {
        return make_error<HuggingfaceArgs>("Invalid num_pairs '" + args[1] +
                                           "': it must be a whole number of at least 1");
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
    static const char* usage =
        "Usage: dataset_manager assign <model_name> [file1 file2 ...] [--count N]";
    if (args.empty() || looks_like_option(args[0])) {
        return make_error<AssignArgs>(usage);
    }
    AssignArgs r;
    r.model_name = args[0];
    bool options = true;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (options && args[i] == "--") {
            options = false;
        } else if (options && args[i] == "--count") {
            if (i + 1 >= args.size()) {
                return missing_value<AssignArgs>("--count", usage);
            }
            if (!parse_count(args[++i], r.count)) {
                return make_error<AssignArgs>(count_error(args[i], "assign every pending file"));
            }
        } else if (options && looks_like_option(args[i])) {
            return unknown_option<AssignArgs>(args[i], usage);  // TD-322
        } else {
            r.targets.push_back(args[i]);
        }
    }
    return r;
}

UnassignArgs parse_unassign_args(const std::vector<std::string>& args) {
    static const char* usage =
        "Usage: dataset_manager unassign <model_name> [file1 file2 ...] [--force]";
    if (args.empty() || looks_like_option(args[0])) {
        return make_error<UnassignArgs>(usage);
    }
    UnassignArgs r;
    r.model_name = args[0];
    bool options = true;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (options && args[i] == "--") {
            options = false;
        } else if (options && args[i] == "--force") {
            r.force = true;
        } else if (options && looks_like_option(args[i])) {
            return unknown_option<UnassignArgs>(args[i], usage);  // TD-322
        } else {
            r.targets.push_back(args[i]);
        }
    }
    return r;
}

DeleteArgs parse_delete_args(const std::vector<std::string>& args) {
    static const char* usage =
        "Usage: dataset_manager delete <file1> [file2 ...] [--force] [--delete-files]";
    DeleteArgs r;
    bool options = true;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (options && args[i] == "--") {
            options = false;
        } else if (options && args[i] == "--force") {
            r.force = true;
        } else if (options && args[i] == "--delete-files") {
            r.delete_files = true;
        } else if (options && looks_like_option(args[i])) {
            return unknown_option<DeleteArgs>(args[i], usage);  // TD-322
        } else {
            r.targets.push_back(args[i]);
        }
    }
    if (r.targets.empty()) {
        return make_error<DeleteArgs>(usage);
    }
    return r;
}

RequeueArgs parse_requeue_args(const std::vector<std::string>& args) {
    static const char* usage =
        "Usage: dataset_manager requeue (--all | <file1> [file2 ...]) [--model <name>]";
    RequeueArgs r;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--all") {
            r.all = true;
        } else if (args[i] == "--model") {
            if (i + 1 >= args.size() || args[i + 1].rfind("--", 0) == 0) {
                return make_error<RequeueArgs>(std::string("--model needs a value. ") + usage);
            }
            r.model_name = args[++i];
        } else if (args[i].rfind("--", 0) == 0) {
            return make_error<RequeueArgs>("Unknown option " + args[i] + ". " + usage);
        } else {
            r.targets.push_back(args[i]);
        }
    }
    if (r.all == !r.targets.empty()) {
        return make_error<RequeueArgs>(std::string("Give either --all or file(s), not both. ") +
                                       usage);
    }
    return r;
}

bool is_valid_dataset_kind(const std::string& kind) {
    return is_dataset_kind(kind);  // TD-402: one list, shared with the trainer
}

MigrateArgs parse_migrate_args(const std::vector<std::string>& args) {
    if (args.empty()) {
        return make_error<MigrateArgs>(
            "Usage: dataset_manager migrate <encoder|decoder|world_model|chatbot> "
            "[file1 file2 ...] [--count N]");
    }
    static const char* usage =
        "Usage: dataset_manager migrate <encoder|decoder|world_model|chatbot> [file1 file2 ...] "
        "[--count N]";
    MigrateArgs r;
    r.kind = args[0];
    if (!is_valid_dataset_kind(r.kind)) {
        return make_error<MigrateArgs>("Invalid kind '" + r.kind +
                                       "' — must be one of: encoder, decoder, world_model, "
                                       "chatbot");
    }
    bool options = true;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (options && args[i] == "--") {
            options = false;
        } else if (options && args[i] == "--count") {
            if (i + 1 >= args.size()) {
                return missing_value<MigrateArgs>("--count", usage);
            }
            if (!parse_count(args[++i], r.count)) {
                return make_error<MigrateArgs>(
                    count_error(args[i], "migrate every legacy pending file"));
            }
        } else if (options && looks_like_option(args[i])) {
            return unknown_option<MigrateArgs>(args[i], usage);  // TD-322
        } else {
            r.targets.push_back(args[i]);
        }
    }
    return r;
}

SegmentArgs parse_segment_args(const std::vector<std::string>& args) {
    if (args.empty()) {
        return make_error<SegmentArgs>(
            "Usage: dataset_manager segment <path> [--count N | --ranges A-B,C-D,...]");
    }
    SegmentArgs r;
    r.path = args[0];
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--count" || args[i] == "--ranges") {
            if (i + 1 >= args.size()) {
                return make_error<SegmentArgs>(args[i] + " needs a value");
            }
        }
        if (args[i] == "--count") {
            if (!parse_positive(args[++i], r.split_count)) {  // TD-323
                return make_error<SegmentArgs>("Invalid --count value '" + args[i] +
                                               "': it must be a whole number of at least 1");
            }
        } else if (args[i] == "--ranges") {
            const std::string& spec = args[++i];
            std::stringstream ss(spec);
            std::string token;
            while (std::getline(ss, token, ',')) {
                const auto dash = token.find('-');
                if (dash == std::string::npos || dash == 0) {
                    return make_error<SegmentArgs>(
                        "Invalid --ranges token (expected A-B, 0-based inclusive pair "
                        "indices): " +
                        token);
                }
                // TD-323: both ends whole numbers ("0-9x" used to read as 0-9).
                const auto a = parse_strict_int(token.substr(0, dash));
                const auto b = parse_strict_int(token.substr(dash + 1));
                if (!a || !b) {
                    return make_error<SegmentArgs>(
                        "Invalid --ranges token (expected A-B, integers): " + token);
                }
                if (*a < 0 || *b < *a) {
                    return make_error<SegmentArgs>("Invalid --ranges token (need 0 <= A <= B): " +
                                                   token);
                }
                // TODO: See TD-324 in TECHNICAL_DEBT.md - b - a + 1 overflows near INT_MAX; overlapping/duplicate ranges accepted
                r.ranges.push_back({*a, *b - *a + 1});
            }
        } else {
            return make_error<SegmentArgs>("Unrecognized argument: " + args[i]);
        }
    }
    if (r.split_count <= 0 && r.ranges.empty()) {
        return make_error<SegmentArgs>("Either --count N or --ranges A-B,... is required");
    }
    // TODO: See TD-324 in TECHNICAL_DEBT.md - a non-positive --count alongside --ranges is silently ignored
    if (r.split_count > 0 && !r.ranges.empty()) {
        return make_error<SegmentArgs>("--count and --ranges are mutually exclusive");
    }
    return r;
}

}  // namespace adai
