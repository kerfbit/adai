#include "IncrementalTrainer.hpp"
#include "Config.hpp"
#include <iostream>
#include <sstream>

// ---------------------------------------------------------------------------
// Helper: apply model architecture + training hyper-params from ServiceConfig
// into an IncrementalConfig so no values are hard-coded in this file.
// ---------------------------------------------------------------------------
static void apply_service_config(IncrementalConfig& cfg,
                                  const adai::ServiceConfig& svc) {
    // Model architecture
    cfg.base_config.d_model            = static_cast<int>(svc.d_model);
    cfg.base_config.num_heads          = static_cast<int>(svc.num_heads);
    cfg.base_config.d_ff               = static_cast<int>(svc.d_ff);
    cfg.base_config.num_encoder_layers = static_cast<int>(svc.num_encoder_layers);
    cfg.base_config.num_decoder_layers = static_cast<int>(svc.num_decoder_layers);
    cfg.base_config.max_seq_length     = static_cast<int>(svc.max_seq_length);

    // Training hyperparameters
    cfg.base_config.learning_rate      = svc.learning_rate;
    cfg.base_config.num_epochs         = svc.num_epochs;
    cfg.base_config.weight_decay       = svc.weight_decay;
    cfg.base_config.gradient_clip_norm = svc.gradient_clip;
    cfg.base_config.batch_size         = svc.batch_size;
}

int main(int argc, char* argv[]) {
    // -----------------------------------------------------------------------
    // Strip the optional global flag  --config <path>  from argv so that the
    // rest of the command-dispatch logic sees a clean args list.
    // Usage:  incremental_trainer [--config <path>] <command> [args...]
    // -----------------------------------------------------------------------
    std::string config_path;
    std::vector<std::string> args;   // args[0] = command, args[1..] = its args

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else {
            args.push_back(a);
        }
    }

    // Load model architecture + training params from config file.
    // Priority: file < environment variables (ConfigLoader already handles this).
    adai::ServiceConfig svc_config = config_path.empty()
        ? adai::ConfigLoader::load()          // tries /etc/adai/config.conf + env
        : adai::ConfigLoader::load(config_path);

    if (args.empty()) {
        std::cout << "Usage: " << argv[0] << " [--config <path>] <command> [options]\n\n";
        std::cout << "Global options:\n";
        std::cout << "  --config <path>              Path to config.conf (default: /etc/adai/config.conf)\n";
        std::cout << "                               Sets model architecture, training params, vocab/model paths\n\n";
        std::cout << "Commands:\n";
        std::cout << "  init [vocab] [model]         Initialize incremental trainer\n";
        std::cout << "  add <data_file>              Add new training data\n";
        std::cout << "  gutenberg <book_id> [pairs]  Download & add Gutenberg book (default: 500 pairs)\n";
        std::cout << "  gutenberg-batch <id1,id2...> Download multiple books\n";
        std::cout << "  train [epochs]               Train on pending data\n";
        std::cout << "  retrain [epochs]             Full retrain on all data\n";
        std::cout << "  reset                        Remove all checkpoints and rebuild model from config\n";
        std::cout << "  resume                       Resume from last session\n";
        std::cout << "  status                       Show training status\n";
        std::cout << "  history                      Show session history\n";
        std::cout << "\nreset options:\n";
        std::cout << "  --yes                        Skip confirmation prompt\n";
        std::cout << "  --keep-data                  Preserve data registry (mark entries untrained)\n";
        std::cout << "\nPopular Gutenberg Books:\n";
        std::cout << "  1342  - Pride and Prejudice (Jane Austen)\n";
        std::cout << "  11    - Alice in Wonderland (Lewis Carroll)\n";
        std::cout << "  84    - Frankenstein (Mary Shelley)\n";
        std::cout << "  1661  - Sherlock Holmes (Arthur Conan Doyle)\n";
        std::cout << "  2701  - Moby Dick (Herman Melville)\n";
        std::cout << "  16328 - Beowulf\n";
        std::cout << "  1260  - Jane Eyre (Charlotte Bronte)\n";
        std::cout << "  98    - A Tale of Two Cities (Charles Dickens)\n";
        std::cout << "\nExample workflow:\n";
        std::cout << "  # Initial training with custom config\n";
        std::cout << "  " << argv[0] << " --config config.conf init\n";
        std::cout << "  " << argv[0] << " --config config.conf gutenberg 1342 500\n";
        std::cout << "  " << argv[0] << " --config config.conf train 10\n";
        std::cout << "\n  # Add multiple classic books\n";
        std::cout << "  " << argv[0] << " --config config.conf gutenberg-batch 11,84,1661,2701\n";
        std::cout << "  " << argv[0] << " --config config.conf train 5\n";
        return 1;
    }
    
    // -----------------------------------------------------------------------
    // Resolve vocab/model paths: prefer ServiceConfig paths, fall back to
    // conventional local file names so the tool still works without a config.
    // -----------------------------------------------------------------------
    std::string default_vocab = svc_config.vocab_path.empty()
                                    ? "vocab.txt"
                                    : svc_config.vocab_path;
    std::string default_model = svc_config.model_path.empty()
                                    ? "chatbot_model.bin"
                                    : svc_config.model_path;

    std::string command = args[0];

    if (command == "init") {
        // init <vocab> <model>  -- paths from args override config
        std::string vocab_path = (args.size() >= 2) ? args[1] : default_vocab;
        std::string model_path = (args.size() >= 3) ? args[2] : default_model;

        IncrementalConfig config;
        apply_service_config(config, svc_config);
        config.base_config.lr_schedule       = LRSchedule::WARMUP_COSINE;
        config.auto_save_enabled             = true;
        config.auto_save_every_minutes       = 30;
        config.auto_save_every_samples       = 1000;

        IncrementalTrainer trainer(vocab_path, model_path, config);

        std::cout << "✅ Incremental trainer initialized\n";
        std::cout << "📁 Session directory: " << config.session_dir << "\n";
        std::cout << "   d_model=" << config.base_config.d_model
                  << "  d_ff=" << config.base_config.d_ff
                  << "  heads=" << config.base_config.num_heads
                  << "  enc_layers=" << config.base_config.num_encoder_layers
                  << "  dec_layers=" << config.base_config.num_decoder_layers
                  << "  max_seq=" << config.base_config.max_seq_length << "\n";

    } else if (command == "add") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0] << " add <data_file>\n";
            return 1;
        }

        std::string data_file = args[1];

        IncrementalConfig config;
        apply_service_config(config, svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);

        if (trainer.add_new_data(data_file)) {
            std::cout << "✅ Data file added to pending queue\n";
            std::cout << "📊 Pending files: " << trainer.get_pending_data_files().size() << "\n";
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

        int book_id      = std::stoi(args[1]);
        int num_pairs    = (args.size() >= 3) ? std::stoi(args[2]) : 500;

        IncrementalConfig config;
        apply_service_config(config, svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);

        std::cout << "📚 Downloading Project Gutenberg book #" << book_id << "...\n";

        if (trainer.add_gutenberg_book(book_id, num_pairs)) {
            std::cout << "✅ Book added to training queue (" << num_pairs << " pairs)\n";
            std::cout << "📊 Pending files: " << trainer.get_pending_data_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add Gutenberg book\n";
            return 1;
        }

    } else if (command == "gutenberg-batch") {
        if (args.size() < 2) {
            std::cerr << "Usage: " << argv[0] << " gutenberg-batch <id1,id2,id3,...> [num_pairs_each]\n";
            std::cerr << "Example: " << argv[0] << " gutenberg-batch 1342,11,84,1661 300\n";
            return 1;
        }

        std::string ids_str    = args[1];
        int num_pairs_each     = (args.size() >= 3) ? std::stoi(args[2]) : 500;

        std::vector<int> book_ids;
        std::stringstream ss(ids_str);
        std::string id;
        while (std::getline(ss, id, ',')) {
            book_ids.push_back(std::stoi(id));
        }

        IncrementalConfig config;
        apply_service_config(config, svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);

        std::cout << "📚 Downloading " << book_ids.size() << " Project Gutenberg books...\n";

        if (trainer.add_gutenberg_books(book_ids, num_pairs_each)) {
            std::cout << "✅ Books added to training queue\n";
            std::cout << "📊 Pending files: " << trainer.get_pending_data_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add Gutenberg books\n";
            return 1;
        }

    } else if (command == "train") {
        // Epoch count can come from args or from config
        int epochs = (args.size() >= 2)
                         ? std::stoi(args[1])
                         : svc_config.num_epochs;

        IncrementalConfig config;
        apply_service_config(config, svc_config);
        config.base_config.num_epochs  = epochs;
        config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
        IncrementalTrainer trainer(default_vocab, default_model, config);

        if (trainer.get_pending_data_files().empty()) {
            std::cout << "⚠️  No pending data. Use 'add' command first.\n";
            return 1;
        }

        std::cout << "🚀 Starting incremental training for " << epochs << " epochs...\n";

        if (trainer.train_incremental(epochs)) {
            std::cout << "✅ Training completed successfully!\n";
            trainer.print_training_summary();
        } else {
            std::cerr << "❌ Training failed\n";
            return 1;
        }

    } else if (command == "retrain") {
        // Epoch count can come from args or from config
        int epochs = (args.size() >= 2)
                         ? std::stoi(args[1])
                         : svc_config.num_epochs;

        IncrementalConfig config;
        apply_service_config(config, svc_config);
        config.base_config.num_epochs  = epochs;
        config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
        config.auto_save_enabled        = true;
        config.auto_save_every_minutes  = 30;
        config.auto_save_every_samples  = 1000;

        IncrementalTrainer trainer(default_vocab, default_model, config);
        trainer.reset_model_for_config();

        std::cout << "🔄 Starting full retrain for " << epochs << " epochs...\n";
        std::cout << "   d_model=" << config.base_config.d_model
                  << "  d_ff=" << config.base_config.d_ff
                  << "  heads=" << config.base_config.num_heads
                  << "  enc_layers=" << config.base_config.num_encoder_layers
                  << "  dec_layers=" << config.base_config.num_decoder_layers
                  << "  lr=" << config.base_config.learning_rate << "\n";

        if (trainer.train_full_retrain(epochs)) {
            std::cout << "✅ Full retrain completed!\n";
            trainer.print_training_summary();
        } else {
            std::cerr << "❌ Full retrain failed\n";
            return 1;
        }

    } else if (command == "reset") {
        // Parse reset-specific flags from remaining args
        bool auto_yes    = false;
        bool keep_data   = false;
        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--yes")        auto_yes  = true;
            else if (args[i] == "--keep-data") keep_data = true;
        }

        // Show what will happen
        std::cout << "\n⚠️  RESET will:\n";
        std::cout << "   • Remove all checkpoint .bin files from the session directory\n";
        std::cout << "   • Back up '" << default_model << "' to '" << default_model << ".bak'\n";
        std::cout << "   • Clear session history and pending-data list\n";
        if (keep_data) {
            std::cout << "   • Preserve data registry (entries marked untrained for re-training)\n";
        } else {
            std::cout << "   • Delete the data registry\n";
        }
        std::cout << "   • Rebuild model architecture from config:\n";
        std::cout << "       d_model="      << svc_config.d_model
                  << "  d_ff="             << svc_config.d_ff
                  << "  heads="            << svc_config.num_heads
                  << "  enc_layers="       << svc_config.num_encoder_layers
                  << "  dec_layers="       << svc_config.num_decoder_layers
                  << "  max_seq="          << svc_config.max_seq_length << "\n\n";

        if (!auto_yes) {
            std::cout << "Continue? [y/N] ";
            std::string answer;
            std::getline(std::cin, answer);
            if (answer.empty() || (answer[0] != 'y' && answer[0] != 'Y')) {
                std::cout << "Reset cancelled.\n";
                return 0;
            }
        }

        IncrementalConfig config;
        apply_service_config(config, svc_config);
        config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
        IncrementalTrainer trainer(default_vocab, default_model, config);

        if (trainer.reset_all(keep_data)) {
            std::cout << "✅ Reset complete. Model rebuilt from config.\n";
            if (!keep_data) {
                std::cout << "   Use 'add' or 'gutenberg' to queue new training data.\n";
            } else {
                std::cout << "   All previous data marked untrained — run 'retrain' to use it.\n";
            }
        } else {
            std::cerr << "❌ Reset failed\n";
            return 1;
        }

    } else if (command == "resume") {
        IncrementalConfig config;
        apply_service_config(config, svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);

        if (trainer.resume_last_session()) {
            std::cout << "✅ Resumed from last session\n";
            std::cout << "📂 Latest checkpoint: " << trainer.get_latest_checkpoint() << "\n";
        } else {
            std::cerr << "❌ Failed to resume session\n";
            return 1;
        }

    } else if (command == "status") {
        IncrementalConfig config;
        apply_service_config(config, svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);

        trainer.print_training_summary();

        std::cout << "\n📋 Pending data files:\n";
        for (const auto& file : trainer.get_pending_data_files()) {
            std::cout << "  - " << file << "\n";
        }

    } else if (command == "history") {
        IncrementalConfig config;
        apply_service_config(config, svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);

        trainer.print_session_history();
        trainer.print_data_registry();

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        std::cerr << "Run '" << argv[0] << "' without arguments for help\n";
        return 1;
    }

    return 0;
}
