#include <fstream>
#include <iostream>
#include <sstream>
#include "Config.hpp"
#include "DataFetcher.hpp"
#include "DatasetRegistry.hpp"
#include "Logger.hpp"
#ifdef BUILD_MNS_SERVER
#include "ModelNameClient.hpp"
#endif

int main(int argc, char* argv[]) {
    // -----------------------------------------------------------------------
    // Parse global --config <path> flag; build clean args list.
    // Usage:  dataset_manager [--config <path>] <command> [args...]
    // -----------------------------------------------------------------------
    std::string config_path;
    std::vector<std::string> args;  // args[0] = command, args[1..] = its args

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else {
            args.push_back(a);
        }
    }

    // Discovery: --config > ./config.registry.conf > /etc/adai/config.registry.conf
    // > ./config.conf (legacy) > /etc/adai/config.conf (legacy).
    config_path = adai::ConfigLoader::discover_config_path(config_path, "config.registry.conf");
    adai::ServiceConfig svc_config = adai::ConfigLoader::load(config_path);

    if (args.empty()) {
        std::cout << "Usage: " << argv[0] << " [--config <path>] <command> [options]\n\n";
        std::cout << "Global options:\n";
        std::cout << "  --config <path>              Path to config.registry.conf\n";
        std::cout << "                               Search order: --config > "
                     "./config.registry.conf >\n";
        std::cout << "                               /etc/adai/config.registry.conf > "
                     "./config.conf (legacy) >\n";
        std::cout << "                               /etc/adai/config.conf (legacy)\n\n";
        std::cout << "Commands:\n";
        std::cout
            << "  add <data_file>              Add a local training file to the pending queue\n";
        std::cout << "  gutenberg <id> [pairs] [model]  Download & queue a Gutenberg book "
                     "(default: 500 pairs)\n";
        std::cout << "  gutenberg-batch <id1,id2...> [pairs] [model]  Download multiple books\n";
        std::cout << "  huggingface <id> [pairs] [split] [in_field] [out_field] [model]\n";
        std::cout << "                               Download a HuggingFace dataset (default: 500 "
                     "pairs, train split).\n";
        std::cout << "                               [model] (remote mode only) rotates through a "
                     "different slice\n";
        std::cout << "                               of the dataset per model name instead of "
                     "always the same rows.\n";
        std::cout << "\nNote: when REGISTRY_SERVER_URL is configured, add/gutenberg*/huggingface "
                     "run\n";
        std::cout << "      on the registry server itself (bytes land in its data_dir and are\n";
        std::cout << "      served to trainers over the existing FTP transport) instead of "
                     "locally.\n";
        std::cout
            << "  status                       Show pending/trained file counts and registry\n";
        std::cout << "  list-pending                 List all pending files\n";
        std::cout << "  list-trained                 List all trained files\n";
        std::cout << "  remove <file>                Remove a single file from the pending queue\n";
        std::cout << "                               (local mode only — use `delete` against a "
                     "live registry_server)\n";
        std::cout << "  clear-pending                Remove all files from the pending queue\n";
        std::cout << "                               (local mode only — use `delete` against a "
                     "live registry_server)\n";
        std::cout << "  assign <model> [file ...] [--count N]\n";
        std::cout << "                               Assign pending file(s) to a model (omit "
                     "files/--count = all)\n";
        std::cout << "  unassign <model> [file ...] [--force]\n";
        std::cout << "                               Clear model assignment (omit files = every "
                     "file assigned to it)\n";
        std::cout << "  delete <file> [...] [--force] [--delete-files]\n";
        std::cout << "                               Permanently purge file(s) from the pending "
                     "queue and registry\n";
        std::cout << "  models                       List registered models from name service\n";
        std::cout << "\nPopular Gutenberg Books:\n";
        std::cout << "  1342  - Pride and Prejudice (Jane Austen)\n";
        std::cout << "  11    - Alice in Wonderland (Lewis Carroll)\n";
        std::cout << "  84    - Frankenstein (Mary Shelley)\n";
        std::cout << "  1661  - Sherlock Holmes (Arthur Conan Doyle)\n";
        std::cout << "  2701  - Moby Dick (Herman Melville)\n";
        std::cout << "  16328 - Beowulf\n";
        std::cout << "  1260  - Jane Eyre (Charlotte Bronte)\n";
        std::cout << "  98    - A Tale of Two Cities (Charles Dickens)\n";
        std::cout << "\nPopular HuggingFace Datasets:\n";
        std::cout
            << "  daily_dialog              - Daily conversation pairs (dialog array format)\n";
        std::cout
            << "  tatsu-lab/alpaca          - Instruction-following (instruction/output fields)\n";
        std::cout
            << "  databricks/databricks-dolly-15k - Instruction dataset (instruction/response)\n";
        std::cout << "  Open-Orca/OpenOrca        - Chain-of-thought Q&A (question/response)\n";
        return 0;
    }

    const std::string command = args[0];

    if (command == "add") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0] << " add <data_file>\n";
            return 1;
        }

        std::string data_file = args[1];

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();

        bool ok;
        if (!svc_config.registry_server_url.empty()) {
            // Remote mode: upload the bytes to registry_server so they land on
            // its data_dir, where the FTP delivery pipeline can serve them out.
            ok = !reg.remote_upload(data_file).empty();
        } else {
            ok = reg.add_file(data_file);
        }

        if (ok) {
            std::cout << "✅ Data file added to pending queue\n";
            std::cout << "📊 Pending files: " << reg.pending_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add data file\n";
            return 1;
        }

    } else if (command == "gutenberg") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0] << " gutenberg <book_id> [num_pairs] [model]\n";
            std::cerr << "Example: " << argv[0] << " gutenberg 1342 500\n";
            std::cerr << "  [model] is only meaningful when REGISTRY_SERVER_URL is set: the "
                        "registry\n";
            std::cerr << "  serves a different rotating slice of the book per model name "
                        "instead of\n";
            std::cerr << "  always the same sentences. Ignored entirely in local mode.\n";
            return 1;
        }

        int book_id = std::stoi(args[1]);
        int num_pairs = (args.size() >= 3) ? std::stoi(args[2]) : 500;
        std::string model = (args.size() >= 4) ? args[3] : "";

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();

        bool ok;
        if (!svc_config.registry_server_url.empty()) {
            std::cout << "📚 Requesting registry server download Project Gutenberg book #"
                      << book_id << "...\n";
            ok = !reg.remote_fetch_gutenberg(book_id, num_pairs, model).empty();
        } else {
            std::cout << "📚 Downloading Project Gutenberg book #" << book_id << "...\n";
            DataFetcher fetcher;
            std::string path = fetcher.fetch_gutenberg(book_id, num_pairs);
            ok = !path.empty() && reg.add_file(path);
        }

        if (ok) {
            std::cout << "✅ Book added to training queue (" << num_pairs << " pairs)\n";
            std::cout << "📊 Pending files: " << reg.pending_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add Gutenberg book\n";
            return 1;
        }

    } else if (command == "gutenberg-batch") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0]
                      << " gutenberg-batch <id1,id2,id3,...> [num_pairs_each] [model]\n";
            std::cerr << "Example: " << argv[0] << " gutenberg-batch 1342,11,84,1661 300\n";
            return 1;
        }

        std::string ids_str = args[1];
        int num_pairs_each = (args.size() >= 3) ? std::stoi(args[2]) : 500;
        std::string model = (args.size() >= 4) ? args[3] : "";

        std::vector<int> book_ids;
        std::stringstream ss(ids_str);
        std::string tok;
        while (std::getline(ss, tok, ','))
            book_ids.push_back(std::stoi(tok));

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();

        int added = 0;
        if (!svc_config.registry_server_url.empty()) {
            std::cout << "📚 Requesting registry server download " << book_ids.size()
                      << " Project Gutenberg books...\n";
            for (int book_id : book_ids) {
                if (!reg.remote_fetch_gutenberg(book_id, num_pairs_each, model).empty())
                    ++added;
            }
        } else {
            std::cout << "📚 Downloading " << book_ids.size() << " Project Gutenberg books...\n";
            DataFetcher fetcher;
            auto paths = fetcher.fetch_gutenberg_batch(book_ids, num_pairs_each);
            for (const auto& p : paths) {
                if (!p.empty() && reg.add_file(p))
                    ++added;
            }
        }
        if (added > 0) {
            std::cout << "✅ " << added << "/" << book_ids.size()
                      << " books added to training queue\n";
            std::cout << "📊 Pending files: " << reg.pending_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add Gutenberg books\n";
            return 1;
        }

    } else if (command == "huggingface") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0]
                      << " huggingface <dataset_id> [num_pairs] [split] [input_field] "
                        "[output_field] [model]\n";
            std::cerr << "Example: " << argv[0] << " huggingface daily_dialog 500\n";
            std::cerr << "Example: " << argv[0]
                      << " huggingface tatsu-lab/alpaca 300 train instruction output\n";
            std::cerr << "  [model] is only meaningful when REGISTRY_SERVER_URL is set: the "
                        "registry\n";
            std::cerr << "  serves a different rotating slice of the dataset per model name "
                        "instead of\n";
            std::cerr << "  always the same rows; omit it (or reuse it) to keep advancing the "
                        "same model's\n";
            std::cerr << "  cursor. Ignored entirely in local mode.\n";
            return 1;
        }

        std::string dataset_id = args[1];
        int num_pairs = (args.size() >= 3) ? std::stoi(args[2]) : 500;
        std::string split = (args.size() >= 4) ? args[3] : "train";
        std::string input_field = (args.size() >= 5) ? args[4] : "";
        std::string output_field = (args.size() >= 6) ? args[5] : "";
        std::string model = (args.size() >= 7) ? args[6] : "";

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();

        bool ok;
        if (!svc_config.registry_server_url.empty()) {
            std::cout << "🤖 Requesting registry server download HuggingFace dataset '"
                      << dataset_id << "'...\n";
            ok = !reg.remote_fetch_huggingface(dataset_id, num_pairs, split, input_field,
                                               output_field, model)
                      .empty();
        } else {
            DataFetcher fetcher;
            std::string path =
                fetcher.fetch_huggingface(dataset_id, num_pairs, split, input_field, output_field);
            ok = !path.empty() && reg.add_file(path);
        }

        if (ok) {
            std::cout << "✅ Dataset added to training queue (" << num_pairs << " pairs)\n";
            std::cout << "📊 Pending files: " << reg.pending_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add HuggingFace dataset\n";
            return 1;
        }

    } else if (command == "status") {
        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();

        auto pending = reg.pending_files();
        auto trained = reg.trained_files();

        std::cout << "📊 Dataset status\n";
        std::cout << "   Pending  : " << pending.size() << " file(s)\n";
        std::cout << "   Trained  : " << trained.size() << " file(s)\n";
        std::cout << "   Samples  : " << reg.total_samples_trained() << " total trained\n";

        reg.print_registry();

        if (!pending.empty()) {
            auto entries = reg.pending_entries();
            std::cout << "\n📋 Pending files:\n";
            for (const auto& e : entries) {
                std::cout << "  - " << e.path;
                if (!e.model_name.empty())
                    std::cout << "  [model: " << e.model_name << "]";
                std::cout << "\n";
            }
        }

    } else if (command == "list-pending") {
        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_pending_list();

        auto entries = reg.pending_entries();
        if (entries.empty()) {
            std::cout << "No pending files.\n";
        } else {
            for (const auto& e : entries) {
                std::cout << e.path;
                if (!e.model_name.empty())
                    std::cout << "\t" << e.model_name;
                std::cout << "\n";
            }
        }

    } else if (command == "list-trained") {
        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();

        auto trained = reg.trained_files();
        if (trained.empty()) {
            std::cout << "No trained files.\n";
        } else {
            for (const auto& f : trained)
                std::cout << f << "\n";
        }

    } else if (command == "remove") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0] << " remove <data_file>\n";
            return 1;
        }

        std::string target = args[1];

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_pending_list();

        if (reg.remove_pending(target)) {
            std::cout << "✅ Removed from pending queue: " << target << "\n";
            std::cout << "📊 Pending files: " << reg.pending_files().size() << "\n";
        } else {
            std::cerr << "❌ File not found in pending queue: " << target << "\n";
            return 1;
        }

    } else if (command == "clear-pending") {
        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_pending_list();

        auto pending = reg.pending_files();
        if (pending.empty()) {
            std::cout << "Pending queue is already empty.\n";
            return 0;
        }

        reg.clear_pending();
        if (reg.save_pending_list()) {
            std::cout << "✅ Cleared " << pending.size() << " file(s) from pending queue\n";
        } else {
            std::cerr << "❌ Failed to save pending list\n";
            return 1;
        }

    } else if (command == "assign") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0]
                      << " assign <model_name> [file1 file2 ...] [--count N]\n";
            std::cerr << "  Omit files and --count to assign all pending files to the model.\n";
            std::cerr << "  --count is ignored when explicit files are given.\n";
            return 1;
        }

        std::string model_name = args[1];

#ifdef BUILD_MNS_SERVER
        std::string mns_url = svc_config.name_service_url;
        if (!mns_url.empty()) {
            try {
                adai::ModelNameClient client(mns_url, svc_config.name_service_timeout_ms);
                auto resolved = client.resolve_model(model_name);
                std::cout << "✅ Verified model: " << resolved.model_name
                          << " (id: " << resolved.model_id << ", state: " << resolved.state
                          << ")\n";
            } catch (const std::exception& e) {
                std::cerr << "❌ Model '" << model_name
                          << "' not found in name service: " << e.what() << "\n";
                return 1;
            }
        }
#endif

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();

        auto pending = reg.pending_files();
        if (pending.empty()) {
            std::cerr << "No pending files to assign.\n";
            return 1;
        }

        std::vector<std::string> targets;
        int count = 0;
        for (std::size_t i = 2; i < args.size(); ++i) {
            if (args[i] == "--count" && i + 1 < args.size()) {
                try {
                    count = std::stoi(args[++i]);
                } catch (...) {
                    std::cerr << "❌ Invalid --count value\n";
                    return 1;
                }
            } else {
                targets.push_back(args[i]);
            }
        }

        auto result = reg.assign_model(model_name, targets, count);
        if (result.assigned > 0) {
            std::cout << "✅ Assigned " << result.assigned << " file(s) to model '" << model_name
                      << "'\n";
            if (targets.empty()) {
                // count-based or "assign all" mode — the caller couldn't
                // otherwise know which specific files were picked.
                for (const auto& p : result.paths) {
                    std::cout << "   " << p << "\n";
                }
            }
        } else {
            std::cerr << "❌ No matching pending files found\n";
            return 1;
        }

    } else if (command == "unassign") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0] << " unassign <model_name> [file1 file2 ...] [--force]\n";
            std::cerr << "  Omit files to clear every entry currently assigned to the model.\n";
            std::cerr << "  --force also clears entries actively claimed by a training run.\n";
            return 1;
        }

        const std::string model_name = args[1];
        std::vector<std::string> targets;
        bool force = false;
        for (std::size_t i = 2; i < args.size(); ++i) {
            if (args[i] == "--force") {
                force = true;
            } else {
                targets.push_back(args[i]);
            }
        }

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_pending_list();

        auto result = reg.unassign_model(model_name, targets, force);
        std::cout << "✅ Unassigned " << result.unassigned << " file(s) from model '" << model_name
                  << "'\n";
        if (result.skipped > 0) {
            std::cout << "   (" << result.skipped
                      << " skipped: actively claimed by a run; use --force to override)\n";
        }
        if (result.unassigned == 0 && result.skipped == 0) {
            return 1;
        }

    } else if (command == "delete") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0]
                      << " delete <file1> [file2 ...] [--force] [--delete-files]\n";
            std::cerr << "  Purges entries from the pending queue and trained registry.\n";
            std::cerr << "  --force overrides the active-run-claim guard on pending entries.\n";
            std::cerr << "  --delete-files also unlinks the underlying file when the registry\n";
            std::cerr << "                 owns it (server-fetched/uploaded datasets only).\n";
            return 1;
        }

        std::vector<std::string> targets;
        bool force = false;
        bool delete_files = false;
        for (std::size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--force") {
                force = true;
            } else if (args[i] == "--delete-files") {
                delete_files = true;
            } else {
                targets.push_back(args[i]);
            }
        }
        if (targets.empty()) {
            std::cerr << "❌ At least one file is required\n";
            return 1;
        }

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();

        auto result = reg.delete_entries(targets, force, delete_files);
        std::cout << "✅ " << result.deleted << " deleted, " << result.skipped << " skipped, "
                  << result.not_found << " not found\n";
        for (const auto& d : result.details) {
            std::cout << "   " << d.path << ": " << d.status;
            if (delete_files && d.status == "deleted") {
                std::cout << (d.file_deleted ? " [file removed]" : " [file kept]");
            }
            std::cout << "\n";
        }
        if (result.deleted == 0) {
            return 1;
        }

    } else if (command == "models") {
#ifdef BUILD_MNS_SERVER
        std::string mns_url = svc_config.name_service_url;
        if (mns_url.empty()) {
            mns_url = "http://localhost:8083";
        }
        try {
            adai::ModelNameClient client(mns_url, svc_config.name_service_timeout_ms);
            // Use resolve_role to test connectivity; list via HTTP directly
            std::string host = "localhost";
            int port = 8083;
            std::string url = mns_url;
            if (url.rfind("http://", 0) == 0)
                url = url.substr(7);
            auto colon = url.find(':');
            if (colon != std::string::npos) {
                host = url.substr(0, colon);
                try {
                    port = std::stoi(url.substr(colon + 1));
                } catch (...) {
                }
            }
            // Query /models endpoint directly
            std::cout << "Querying name service at " << mns_url << "...\n";
            // ModelNameClient doesn't expose list; use resolve_model for known names
            // or just report the configured model
            if (!svc_config.model_name.empty()) {
                auto resolved = client.resolve_model(svc_config.model_name);
                std::cout << "  Model: " << resolved.model_name << "  State: " << resolved.state
                          << "  ID: " << resolved.model_id << "\n";
                if (!resolved.artifact.path.empty()) {
                    std::cout << "  Artifact: " << resolved.artifact.path << "\n";
                }
            } else {
                std::cout
                    << "No MODEL_NAME configured. Set NAME_SERVICE_URL and MODEL_NAME in config.\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Name service query failed: " << e.what() << "\n";
            return 1;
        }
#else
        std::cerr << "Name service support not compiled (requires BUILD_MNS_SERVER)\n";
        return 1;
#endif

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        std::cerr << "Run '" << argv[0] << "' without arguments for help\n";
        return 1;
    }

    return 0;
}
