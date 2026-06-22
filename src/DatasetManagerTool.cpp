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

    // Auto-discover config.conf: explicit --config > CWD/config.conf > /etc/adai/config.conf.
    if (config_path.empty()) {
        std::ifstream local_check("config.conf");
        if (local_check.good()) config_path = "config.conf";
    }
    adai::ServiceConfig svc_config =
        config_path.empty() ? adai::ConfigLoader::load()
                            : adai::ConfigLoader::load(config_path);

    if (args.empty()) {
        std::cout << "Usage: " << argv[0] << " [--config <path>] <command> [options]\n\n";
        std::cout << "Global options:\n";
        std::cout << "  --config <path>              Path to config.conf\n";
        std::cout << "                               Search order: --config > ./config.conf > "
                     "/etc/adai/config.conf\n\n";
        std::cout << "Commands:\n";
        std::cout << "  add <data_file>              Add a local training file to the pending queue\n";
        std::cout << "  gutenberg <id> [pairs]       Download & queue a Gutenberg book (default: 500 pairs)\n";
        std::cout << "  gutenberg-batch <id1,id2...> [pairs]  Download multiple books\n";
        std::cout << "  huggingface <id> [pairs] [split] [in_field] [out_field]\n";
        std::cout << "                               Download a HuggingFace dataset (default: 500 pairs, train split)\n";
        std::cout << "  status                       Show pending/trained file counts and registry\n";
        std::cout << "  list-pending                 List all pending files\n";
        std::cout << "  list-trained                 List all trained files\n";
        std::cout << "  clear-pending                Remove all files from the pending queue\n";
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
        std::cout << "  daily_dialog              - Daily conversation pairs (dialog array format)\n";
        std::cout << "  tatsu-lab/alpaca          - Instruction-following (instruction/output fields)\n";
        std::cout << "  databricks/databricks-dolly-15k - Instruction dataset (instruction/response)\n";
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

        if (reg.add_file(data_file)) {
            std::cout << "✅ Data file added to pending queue\n";
            std::cout << "📊 Pending files: " << reg.pending_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add data file\n";
            return 1;
        }

    } else if (command == "gutenberg") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0] << " gutenberg <book_id> [num_pairs]\n";
            std::cerr << "Example: " << argv[0] << " gutenberg 1342 500\n";
            return 1;
        }

        int book_id = std::stoi(args[1]);
        int num_pairs = (args.size() >= 3) ? std::stoi(args[2]) : 500;

        DataFetcher fetcher;
        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();

        std::cout << "📚 Downloading Project Gutenberg book #" << book_id << "...\n";

        std::string path = fetcher.fetch_gutenberg(book_id, num_pairs);
        if (!path.empty() && reg.add_file(path)) {
            std::cout << "✅ Book added to training queue (" << num_pairs << " pairs)\n";
            std::cout << "📊 Pending files: " << reg.pending_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add Gutenberg book\n";
            return 1;
        }

    } else if (command == "gutenberg-batch") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0]
                      << " gutenberg-batch <id1,id2,id3,...> [num_pairs_each]\n";
            std::cerr << "Example: " << argv[0] << " gutenberg-batch 1342,11,84,1661 300\n";
            return 1;
        }

        std::string ids_str = args[1];
        int num_pairs_each = (args.size() >= 3) ? std::stoi(args[2]) : 500;

        std::vector<int> book_ids;
        std::stringstream ss(ids_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) book_ids.push_back(std::stoi(tok));

        DataFetcher fetcher;
        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();

        std::cout << "📚 Downloading " << book_ids.size() << " Project Gutenberg books...\n";

        auto paths = fetcher.fetch_gutenberg_batch(book_ids, num_pairs_each);
        int added = 0;
        for (const auto& p : paths) {
            if (!p.empty() && reg.add_file(p)) ++added;
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
                      << " huggingface <dataset_id> [num_pairs] [split] [input_field] [output_field]\n";
            std::cerr << "Example: " << argv[0] << " huggingface daily_dialog 500\n";
            std::cerr << "Example: " << argv[0]
                      << " huggingface tatsu-lab/alpaca 300 train instruction output\n";
            return 1;
        }

        std::string dataset_id  = args[1];
        int num_pairs           = (args.size() >= 3) ? std::stoi(args[2]) : 500;
        std::string split       = (args.size() >= 4) ? args[3] : "train";
        std::string input_field = (args.size() >= 5) ? args[4] : "";
        std::string output_field= (args.size() >= 6) ? args[5] : "";

        DataFetcher fetcher;
        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        reg.load_pending_list();

        std::string path =
            fetcher.fetch_huggingface(dataset_id, num_pairs, split, input_field, output_field);
        if (!path.empty() && reg.add_file(path)) {
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
            std::cout << "\n📋 Pending files:\n";
            for (const auto& f : pending) std::cout << "  - " << f << "\n";
        }

    } else if (command == "list-pending") {
        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_pending_list();

        auto pending = reg.pending_files();
        if (pending.empty()) {
            std::cout << "No pending files.\n";
        } else {
            for (const auto& f : pending) std::cout << f << "\n";
        }

    } else if (command == "list-trained") {
        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();

        auto trained = reg.trained_files();
        if (trained.empty()) {
            std::cout << "No trained files.\n";
        } else {
            for (const auto& f : trained) std::cout << f << "\n";
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
            if (url.rfind("http://", 0) == 0) url = url.substr(7);
            auto colon = url.find(':');
            if (colon != std::string::npos) {
                host = url.substr(0, colon);
                try { port = std::stoi(url.substr(colon + 1)); } catch (...) {}
            }
            // Query /models endpoint directly
            std::cout << "Querying name service at " << mns_url << "...\n";
            // ModelNameClient doesn't expose list; use resolve_model for known names
            // or just report the configured model
            if (!svc_config.model_name.empty()) {
                auto resolved = client.resolve_model(svc_config.model_name);
                std::cout << "  Model: " << resolved.model_name
                          << "  State: " << resolved.state
                          << "  ID: " << resolved.model_id << "\n";
                if (!resolved.artifact.path.empty()) {
                    std::cout << "  Artifact: " << resolved.artifact.path << "\n";
                }
            } else {
                std::cout << "No MODEL_NAME configured. Set NAME_SERVICE_URL and MODEL_NAME in config.\n";
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
