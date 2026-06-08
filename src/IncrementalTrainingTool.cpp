#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include "Config.hpp"
#include "DataFetcher.hpp"
#include "DatasetRegistry.hpp"
#include "IncrementalTrainer.hpp"
#include "Logger.hpp"
#include "Matrix.hpp"

#ifndef _WIN32
#  include <fcntl.h>
#  include <unistd.h>
#endif
#ifdef _WIN32
#  include <windows.h>
#endif

namespace {

// Detaches the calling process into the background.
//
// POSIX: fork()s; the parent gets back the child PID (> 0) and should print
//        the startup banner then exit(0).  The child gets 0, calls setsid(),
//        and redirects stdin/stdout/stderr to /dev/null before returning so
//        training output flows only through the Logger file sink.
// Windows: re-launches the current command line with --background-child via
//          CreateProcess(DETACHED_PROCESS); the launcher gets back the new
//          process ID (> 0) and should print the banner then exit(0).  The
//          re-launched child detects --background-child in argv[] and returns 0.
//
// Returns -1 if the fork/CreateProcess call fails (caller falls back to
// running in the foreground).
long long launch_background(int argc, char* argv[]) {
#ifndef _WIN32
    (void)argc;
    (void)argv;
    pid_t pid = ::fork();
    if (pid < 0) return -1LL;   // fork failed
    if (pid > 0) return static_cast<long long>(pid);  // parent: return child PID

    // --- child ---
    ::setsid();  // become session leader; fully detach from controlling terminal

    // Redirect stdin to /dev/null
    int nr = ::open("/dev/null", O_RDONLY);
    if (nr >= 0) { ::dup2(nr, STDIN_FILENO);  ::close(nr); }

    // Redirect stdout/stderr to /dev/null (training output goes through Logger)
    int nw = ::open("/dev/null", O_WRONLY);
    if (nw >= 0) {
        ::dup2(nw, STDOUT_FILENO);
        ::dup2(nw, STDERR_FILENO);
        ::close(nw);
    }
    return 0LL;

#else  // _WIN32
    // If this process was already re-launched as the background child, signal
    // that to the caller so it continues with normal training.
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--background-child") return 0LL;

    // Re-launch the same command line with the sentinel flag appended.
    std::string cmd = ::GetCommandLineA();
    cmd += " --background-child";

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    if (!::CreateProcessA(nullptr, cmd.data(), nullptr, nullptr,
                          FALSE, DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                          nullptr, nullptr, &si, &pi))
        return -1LL;

    const long long child_pid = static_cast<long long>(::GetProcessId(pi.hProcess));
    ::CloseHandle(pi.hProcess);
    ::CloseHandle(pi.hThread);
    return child_pid;
#endif
}

std::string derive_run_id(const std::string& configured) {
    if (!configured.empty()) return configured;
    std::string host = "host";
#ifdef _WIN32
    if (const char* env_host = std::getenv("COMPUTERNAME")) host = env_host;
    const int pid_tail = static_cast<int>(_getpid() % 10000);
#else
    std::array<char, 256> buf{};
    if (gethostname(buf.data(), buf.size() - 1) == 0) host = buf.data();
    const int pid_tail = static_cast<int>(getpid() % 10000);
#endif
    if (host.size() > 8) host = host.substr(0, 8);
    return host + "_" + std::to_string(pid_tail);
}

}  // namespace

int main(int argc, char* argv[]) {
    // -----------------------------------------------------------------------
    // Strip the optional global flag  --config <path>  from argv so that the
    // rest of the command-dispatch logic sees a clean args list.
    // Usage:  incremental_trainer [--config <path>] <command> [args...]
    // -----------------------------------------------------------------------
    std::string config_path;
    std::string gpu_strategy_override;
    std::vector<std::string> args;  // args[0] = command, args[1..] = its args

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (a == "--gpu-strategy" && i + 1 < argc) {
            gpu_strategy_override = argv[++i];
        } else if (a == "--background-child") {
            // Windows sentinel flag — already handled inside launch_background();
            // strip here so it never reaches the command dispatcher.
            (void)0;
        } else {
            args.push_back(a);
        }
    }

    // Load model architecture + training params from config file.
    // Priority: file < environment variables (ConfigLoader already handles this).
    // Auto-discover config.conf: explicit --config > CWD/config.conf > /etc/adai/config.conf.
    if (config_path.empty()) {
        // Check the current working directory first so running from the project
        // root always picks up the local config.conf without needing --config.
        std::ifstream local_check("config.conf");
        if (local_check.good()) {
            config_path = "config.conf";
        }
    }
    adai::ServiceConfig svc_config =
        config_path.empty()
            ? adai::ConfigLoader::load()  // falls back to /etc/adai/config.conf + env
            : adai::ConfigLoader::load(config_path);

    // CLI --gpu-strategy overrides the config file value.
    if (!gpu_strategy_override.empty()) {
        svc_config.gpu_strategy = adai::gpu_strategy_from_string(gpu_strategy_override);
    }

    // GPU auto-detect: attempt initialisation when requested; silently use CPU if unavailable.
    if (svc_config.gpu_enabled) {
        const bool low_priority = (svc_config.gpu_strategy == adai::GPUStrategy::BACKGROUND);
        if (Matrix::gpu_try_initialize(svc_config.gpu_device_id, svc_config.gpu_memory_fraction,
                                       low_priority)) {
            const char* mode = low_priority ? "background (low-priority stream)"
                                            : "full (high-priority stream)";
            adai::Logger::info("[GPU] GPU ready — strategy: {}. {}", mode, Matrix::gpu_info());
        } else {
            adai::Logger::warn(
                "[GPU] No CUDA device found or initialisation failed — running on CPU");
        }
    }

    if (args.empty()) {
        std::cout << "Usage: " << argv[0] << " [--config <path>] <command> [options]\n\n";
        std::cout << "Global options:\n";
        std::cout << "  --config <path>              Path to config.conf\n";
        std::cout << "                               Search order: --config > ./config.conf > "
                     "/etc/adai/config.conf\n";
        std::cout << "                               Sets model architecture, training params, "
                     "vocab/model paths\n";
        std::cout << "  --gpu-strategy <mode>        GPU scheduling strategy: background (default) or full\n";
        std::cout << "                               background: low-priority stream, yields to other GPU work\n";
        std::cout << "                               full:       high-priority stream, maximises throughput\n";
        std::cout << "                               Tip: pair 'full' with GPU_MEMORY_FRACTION=0.9\n\n";
        std::cout << "Commands:\n";
        std::cout << "  init [vocab] [model]         Initialize incremental trainer\n";
        std::cout << "  add <data_file>              Add new training data\n";
        std::cout << "  gutenberg <book_id> [pairs]  Download & add Gutenberg book (default: 500 "
                     "pairs)\n";
        std::cout << "  gutenberg-batch <id1,id2...> Download multiple books\n";
        std::cout << "  huggingface <dataset_id> [pairs] [split] [in_field] [out_field]\n";
        std::cout << "                               Download a HuggingFace dataset (default: 500 "
                     "pairs, train split)\n";
        std::cout << "  train [epochs]               Train on pending data\n";
        std::cout << "  retrain [epochs]             Full retrain on all data\n";
        std::cout << "  reset                        Remove all checkpoints and rebuild model from "
                     "config\n";
        std::cout << "  resume                       Resume from last session\n";
        std::cout << "  status                       Show training status\n";
        std::cout << "  history                      Show session history\n";
        std::cout << "\nreset options:\n";
        std::cout << "  --yes                        Skip confirmation prompt\n";
        std::cout
            << "  --keep-data                  Preserve data registry (mark entries untrained)\n";
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
        std::cout << "  HuggingFaceH4/ultrachat_200k - Multi-turn chat (requires HF_TOKEN for some "
                     "splits)\n";
        std::cout << "\nnote: Set HF_TOKEN env var to access gated datasets\n";
        std::cout << "\nExample workflow:\n";
        std::cout << "  # Initial training with custom config\n";
        std::cout << "  " << argv[0] << " --config config.conf init\n";
        std::cout << "  " << argv[0] << " --config config.conf gutenberg 1342 500\n";
        std::cout << "  " << argv[0] << " --config config.conf train 10\n";
        std::cout << "\n  # Add multiple classic books\n";
        std::cout << "  " << argv[0] << " --config config.conf gutenberg-batch 11,84,1661,2701\n";
        std::cout << "  " << argv[0] << " --config config.conf train 5\n";
        std::cout << "\n  # Add a HuggingFace dataset (auto-detect fields)\n";
        std::cout << "  " << argv[0] << " --config config.conf huggingface daily_dialog 500\n";
        std::cout
            << "  " << argv[0]
            << " --config config.conf huggingface tatsu-lab/alpaca 300 train instruction output\n";
        std::cout << "  " << argv[0] << " --config config.conf train 5\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // Resolve vocab/model paths: prefer ServiceConfig paths, fall back to
    // conventional local file names so the tool still works without a config.
    // -----------------------------------------------------------------------
    std::string default_vocab = svc_config.vocab_path.empty() ? "vocab.txt" : svc_config.vocab_path;
    std::string default_model =
        svc_config.model_path.empty() ? "chatbot_model.bin" : svc_config.model_path;

    std::string command = args[0];

    if (command == "init") {
        // init <vocab> <model>  -- paths from args override config
        std::string vocab_path = (args.size() >= 2) ? args[1] : default_vocab;
        std::string model_path = (args.size() >= 3) ? args[2] : default_model;

        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
        config.auto_save_enabled = true;
        config.auto_save_every_minutes = 30;
        config.auto_save_every_samples = 1000;

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
        std::string id;
        while (std::getline(ss, id, ',')) {
            book_ids.push_back(std::stoi(id));
        }

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
            std::cerr
                << "Usage: " << argv[0]
                << " huggingface <dataset_id> [num_pairs] [split] [input_field] [output_field]\n";
            std::cerr << "Example: " << argv[0] << " huggingface daily_dialog 500\n";
            std::cerr << "Example: " << argv[0]
                      << " huggingface tatsu-lab/alpaca 300 train instruction output\n";
            return 1;
        }

        std::string dataset_id = args[1];
        int num_pairs = (args.size() >= 3) ? std::stoi(args[2]) : 500;
        std::string split = (args.size() >= 4) ? args[3] : "train";
        std::string input_field = (args.size() >= 5) ? args[4] : "";
        std::string output_field = (args.size() >= 6) ? args[5] : "";

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

    } else if (command == "train") {
        // Epoch count can come from args or from config
        int epochs = (args.size() >= 2) ? std::stoi(args[1]) : svc_config.num_epochs;

        // Count pending files before fork (cheap single-file read; avoids a
        // full IncrementalTrainer construction just to query the list).
        std::vector<std::string> pre_fork_pending;
        {
            std::ifstream pf(svc_config.session_dir + "/pending_files.txt");
            std::string line;
            while (std::getline(pf, line))
                if (!line.empty()) pre_fork_pending.push_back(line);
        }
        if (pre_fork_pending.empty()) {
            std::cout << "⚠️  No pending data. Use 'add' command first.\n";
            return 1;
        }

        const std::string log_path =
            svc_config.log_file_path.empty() ? "chatbot_server.log" : svc_config.log_file_path;

        const long long child_pid = launch_background(argc, argv);
        if (child_pid > 0) {
            // Parent: print structured startup banner and exit so the shell
            // prompt returns to the user immediately.
            std::cout << "[ADAI] Training started in background — PID " << child_pid << "\n"
                      << "       Model  : " << default_model << "\n"
                      << "       Data   : " << pre_fork_pending.size() << " pending file(s)\n"
                      << "       Epochs : " << epochs << "\n"
                      << "       Log    : " << log_path << "\n"
                      << "       Stop   : kill " << child_pid << "\n";
            return 0;
        }
        if (child_pid < 0) {
            adai::Logger::warn("[background] fork failed — running in foreground");
        }

        // Child (or foreground fallback): run normal incremental training.
        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        config.base_config.num_epochs = epochs;
        config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
        IncrementalTrainer trainer(default_vocab, default_model, config);

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        const std::string run_id = derive_run_id(svc_config.run_id);
        auto pending = reg.acquire_pending(run_id);
        if (pending.empty()) {
            adai::Logger::warn("No pending data files to train on");
            return 1;
        }
        if (trainer.train_on_files(pending, epochs)) {
            std::vector<int> counts(pending.size(), 0);
            reg.mark_trained(run_id, pending, counts);
            trainer.print_training_summary();
        } else {
            reg.release_pending(run_id, pending);
            return 1;
        }

    } else if (command == "retrain") {
        // Epoch count can come from args or from config
        int epochs = (args.size() >= 2) ? std::stoi(args[1]) : svc_config.num_epochs;

        // Count all registered data files before fork (pending + trained) so
        // the startup banner can report the full dataset size.
        int data_file_count = 0;
        {
            std::ifstream pf(svc_config.session_dir + "/pending_files.txt");
            std::string line;
            while (std::getline(pf, line))
                if (!line.empty()) ++data_file_count;
        }

        const std::string log_path =
            svc_config.log_file_path.empty() ? "chatbot_server.log" : svc_config.log_file_path;

        const long long child_pid = launch_background(argc, argv);
        if (child_pid > 0) {
            std::cout << "[ADAI] Full retrain started in background — PID " << child_pid << "\n"
                      << "       Model  : " << default_model << "\n"
                      << "       Data   : " << data_file_count << " file(s)\n"
                      << "       Epochs : " << epochs << "\n"
                      << "       Log    : " << log_path << "\n"
                      << "       Stop   : kill " << child_pid << "\n";
            return 0;
        }
        if (child_pid < 0) {
            adai::Logger::warn("[background] fork failed — running in foreground");
        }

        // Child (or foreground fallback): run full retrain.
        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        config.base_config.num_epochs = epochs;
        config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
        config.auto_save_enabled = true;
        config.auto_save_every_minutes = 30;
        config.auto_save_every_samples = 1000;

        IncrementalTrainer trainer(default_vocab, default_model, config);
        trainer.reset_model_for_config();

        DatasetRegistry reg(DatasetRegistry::make_config(svc_config));
        reg.load_registry();
        const std::string run_id = derive_run_id(svc_config.run_id);
        auto trained_fs = reg.trained_files();
        auto pending_fs = reg.acquire_pending(run_id);
        std::vector<std::string> all_files;
        all_files.insert(all_files.end(), trained_fs.begin(), trained_fs.end());
        all_files.insert(all_files.end(), pending_fs.begin(), pending_fs.end());
        if (all_files.empty()) {
            adai::Logger::warn("No data files to retrain on");
            return 1;
        }
        adai::Logger::info("Starting full retrain on {} data file(s)", all_files.size());
        if (trainer.retrain_on_files(all_files, epochs)) {
            if (!pending_fs.empty()) {
                std::vector<int> counts(pending_fs.size(), 0);
                reg.mark_trained(run_id, pending_fs, counts);
            }
            trainer.print_training_summary();
        } else {
            if (!pending_fs.empty()) reg.release_pending(run_id, pending_fs);
            std::cerr << "❌ Full retrain failed\n";
            return 1;
        }

    } else if (command == "reset") {
        // Parse reset-specific flags from remaining args
        bool auto_yes = false;
        bool keep_data = false;
        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--yes") {
                auto_yes = true;
            } else if (args[i] == "--keep-data") {
                keep_data = true;
            }
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
        std::cout << "       d_model=" << svc_config.d_model << "  d_ff=" << svc_config.d_ff
                  << "  heads=" << svc_config.num_heads
                  << "  enc_layers=" << svc_config.num_encoder_layers
                  << "  dec_layers=" << svc_config.num_decoder_layers
                  << "  max_seq=" << svc_config.max_seq_length << "\n\n";

        if (!auto_yes) {
            std::cout << "Continue? [y/N] ";
            std::string answer;
            std::getline(std::cin, answer);
            if (answer.empty() || (answer[0] != 'y' && answer[0] != 'Y')) {
                std::cout << "Reset cancelled.\n";
                return 0;
            }
        }

        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
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
        const std::string log_path =
            svc_config.log_file_path.empty() ? "chatbot_server.log" : svc_config.log_file_path;

        const long long child_pid = launch_background(argc, argv);
        if (child_pid > 0) {
            std::cout << "[ADAI] Resume started in background — PID " << child_pid << "\n"
                      << "       Model  : " << default_model << "\n"
                      << "       Log    : " << log_path << "\n"
                      << "       Stop   : kill " << child_pid << "\n";
            return 0;
        }
        if (child_pid < 0) {
            adai::Logger::warn("[background] fork failed — running in foreground");
        }

        // Child (or foreground fallback): run resume.
        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);

        if (trainer.resume_last_session()) {
            adai::Logger::info("Resumed from last session; latest checkpoint: {}",
                               trainer.get_latest_checkpoint());
        } else {
            return 1;
        }

    } else if (command == "status") {
        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);
        trainer.load_data_registry();
        trainer.load_pending_data_list();

        trainer.print_training_summary();

        std::cout << "\n📋 Pending data files:\n";
        for (const auto& file : trainer.get_pending_data_files()) {
            std::cout << "  - " << file << "\n";
        }

    } else if (command == "history") {
        IncrementalConfig config = IncrementalTrainer::make_incremental_config(svc_config);
        IncrementalTrainer trainer(default_vocab, default_model, config);
        trainer.load_data_registry();
        trainer.load_pending_data_list();

        trainer.print_session_history();
        trainer.print_data_registry();

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        std::cerr << "Run '" << argv[0] << "' without arguments for help\n";
        return 1;
    }

    return 0;
}
