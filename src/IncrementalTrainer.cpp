#include "IncrementalTrainer.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <ctime>
#include <regex>
#include <cstdlib>

// ANSI color codes
#define COLOR_RESET "\033[0m"
#define COLOR_INFO "\033[1;36m"
#define COLOR_SUCCESS "\033[1;32m"
#define COLOR_WARNING "\033[1;33m"
#define COLOR_ERROR "\033[1;31m"
#define COLOR_PROGRESS "\033[1;35m"

namespace fs = std::filesystem;

IncrementalTrainer::IncrementalTrainer(const std::string& vocab_path, 
                                       const std::string& model_path)
    : current_session_id(0), samples_since_last_save(0), 
      best_validation_loss(std::numeric_limits<float>::max()), 
      best_checkpoint_path("") {
    
    std::cout << COLOR_INFO << "🔧 Initializing Incremental Training System..." << COLOR_RESET << std::endl;
    
    // Load tokenizer
    tokenizer = std::make_unique<BPETokenizer>();
    tokenizer->load_vocab(vocab_path);
    std::cout << COLOR_SUCCESS << "✅ Tokenizer loaded (vocab size: " 
              << tokenizer->get_vocab_size() << ")" << COLOR_RESET << std::endl;
    
    // Load or create model
    model = std::make_unique<EncoderDecoderModel>(
        tokenizer->get_vocab_size(),
        config.base_config.d_model,
        config.base_config.num_encoder_layers,
        config.base_config.num_decoder_layers,
        config.base_config.num_heads,
        config.base_config.d_ff,
        config.base_config.max_seq_length
    );
    
    model->set_tokenizer(tokenizer.release());
    
    // Try to load existing model
    std::ifstream model_file(model_path);
    if (model_file.good()) {
        try {
            model->load_model(model_path);
            std::cout << COLOR_SUCCESS << "✅ Model loaded from: " << model_path << COLOR_RESET << std::endl;
        } catch (...) {
            std::cout << COLOR_WARNING << "⚠️  Could not load model, using fresh initialization" 
                      << COLOR_RESET << std::endl;
        }
    }
    
    // Load session history and data registry
    ensure_directories_exist();
    load_session_history();
    load_data_registry();
    load_pending_data_list();
    
    // Initialize best checkpoint from history (TD-005)
    if (!session_history.empty()) {
        for (const auto& session : session_history) {
            if (session.final_validation_loss < best_validation_loss) {
                best_validation_loss = session.final_validation_loss;
                best_checkpoint_path = session.checkpoint_path;
            }
        }
        if (!best_checkpoint_path.empty()) {
            std::cout << COLOR_INFO << "🏆 Best checkpoint: " << best_checkpoint_path 
                      << " (val loss: " << best_validation_loss << ")" << COLOR_RESET << std::endl;
        }
    }
    
    last_save_time = std::chrono::system_clock::now();
}

IncrementalTrainer::IncrementalTrainer(const std::string& vocab_path, 
                                       const std::string& model_path,
                                       const IncrementalConfig& cfg)
    : IncrementalTrainer(vocab_path, model_path) {
    config = cfg;
}

void IncrementalTrainer::set_config(const IncrementalConfig& cfg) {
    config = cfg;
}

IncrementalConfig& IncrementalTrainer::get_config() {
    return config;
}

bool IncrementalTrainer::add_new_data(const std::string& data_file) {
    if (!fs::exists(data_file)) {
        std::cerr << COLOR_ERROR << "❌ Data file not found: " << data_file << COLOR_RESET << std::endl;
        return false;
    }
    
    // Check if already trained
    if (is_data_trained(data_file)) {
        std::cout << COLOR_WARNING << "⚠️  Data file already trained: " << data_file << COLOR_RESET << std::endl;
        return false;
    }
    
    pending_data_files.push_back(data_file);
    std::cout << COLOR_SUCCESS << "✅ Added new data file: " << data_file << COLOR_RESET << std::endl;
    
    // Save pending files list
    save_pending_data_list();
    
    return true;
}

bool IncrementalTrainer::add_new_data_batch(const std::vector<std::string>& data_files) {
    int added = 0;
    for (const auto& file : data_files) {
        if (add_new_data(file)) {
            added++;
        }
    }
    std::cout << COLOR_INFO << "📦 Added " << added << "/" << data_files.size() 
              << " new data files" << COLOR_RESET << std::endl;
    return added > 0;
}

void IncrementalTrainer::clear_pending_data() {
    pending_data_files.clear();
}

std::vector<std::string> IncrementalTrainer::get_pending_data_files() const {
    return pending_data_files;
}

std::vector<std::string> IncrementalTrainer::get_trained_data_files() const {
    return std::vector<std::string>(trained_data_files.begin(), trained_data_files.end());
}

bool IncrementalTrainer::train_incremental(int num_epochs) {
    if (pending_data_files.empty()) {
        std::cout << COLOR_WARNING << "⚠️  No pending data files to train on" << COLOR_RESET << std::endl;
        return false;
    }
    
    std::cout << COLOR_INFO << "\n🚀 Starting Incremental Training Session #" << current_session_id + 1 
              << COLOR_RESET << std::endl;
    std::cout << COLOR_INFO << "📊 Pending data files: " << pending_data_files.size() << COLOR_RESET << std::endl;
    
    initialize_session();
    
    // Load all pending data
    std::vector<ConversationPair> training_pairs;
    std::vector<ConversationPair> validation_pairs;
    
    for (const auto& data_file : pending_data_files) {
        std::vector<ConversationPair> pairs;
        int loaded = load_conversation_pairs(data_file, pairs);
        
        if (loaded > 0) {
            // Split into training/validation
            int val_size = loaded / config.base_config.validation_split;
            for (int i = 0; i < loaded; ++i) {
                if (i < val_size) {
                    validation_pairs.push_back(pairs[i]);
                } else {
                    training_pairs.push_back(pairs[i]);
                }
            }
            
            // Mark as trained
            DataVersion dv;
            dv.data_file = data_file;
            dv.checksum = compute_data_checksum(data_file);
            dv.num_samples = loaded;
            dv.added_time = std::chrono::system_clock::now();
            dv.trained = true;
            data_registry.push_back(dv);
            trained_data_files.insert(data_file);
        }
    }
    
    std::cout << COLOR_INFO << "📈 Total training samples: " << training_pairs.size() << COLOR_RESET << std::endl;
    std::cout << COLOR_INFO << "📉 Total validation samples: " << validation_pairs.size() << COLOR_RESET << std::endl;
    
    // Create trainer and configure
    ChatbotTrainer trainer(config.base_config);
    
    // Model already owns tokenizer from construction - just transfer model
    trainer.set_model(std::move(model));
    
    // Load data into trainer
    for (const auto& pair : training_pairs) {
        trainer.add_training_pair(pair.input, pair.response);
    }
    
    // Train
    std::cout << COLOR_PROGRESS << "\n⚡ Training for " << num_epochs << " epochs..." << COLOR_RESET << std::endl;
    bool success = trainer.train(num_epochs);
    
    // Get model back (model owns the tokenizer)
    model = trainer.release_model();
    
    if (success) {
        // TODO: TD-004 - Retrieve per-epoch losses from ChatbotTrainer with get_epoch_losses() method
        // TODO: TD-004 - Retrieve per-epoch validation losses from ChatbotTrainer
        // TODO: TD-004 - Calculate per-epoch training times from epoch start/end timestamps
        // TODO: TD-004 - Retrieve per-epoch learning rates from optimizer's LR scheduler history
        // TODO: TD-004 - Retrieve per-epoch gradient norms from optimizer's gradient tracking
        
        // Save checkpoint
        std::string checkpoint_path = generate_session_checkpoint_path();
        save_model(checkpoint_path);
        
        // Finalize session
        float final_loss = trainer.get_final_training_loss();
        float final_val_loss = trainer.get_final_validation_loss();
        finalize_session(training_pairs.size(), num_epochs, final_loss, final_val_loss);
        
        // Clear pending data
        pending_data_files.clear();
        save_pending_data_list();
        
        // Save registry
        save_data_registry();
        save_session_history();
        
        std::cout << COLOR_SUCCESS << "\n✅ Incremental training session completed!" << COLOR_RESET << std::endl;
        print_training_summary();
    }
    
    return success;
}

bool IncrementalTrainer::train_on_new_data_only(int num_epochs) {
    return train_incremental(num_epochs);
}

bool IncrementalTrainer::train_full_retrain(int num_epochs) {
    std::cout << COLOR_INFO << "\n🔄 Starting Full Retrain on All Data" << COLOR_RESET << std::endl;
    
    // Collect all trained data files
    std::vector<std::string> all_data_files;
    for (const auto& dv : data_registry) {
        if (dv.trained) {
            all_data_files.push_back(dv.data_file);
        }
    }
    
    // Add pending files
    all_data_files.insert(all_data_files.end(), pending_data_files.begin(), pending_data_files.end());
    
    if (all_data_files.empty()) {
        std::cout << COLOR_WARNING << "⚠️  No data files to train on" << COLOR_RESET << std::endl;
        return false;
    }
    
    std::cout << COLOR_INFO << "📦 Retraining on " << all_data_files.size() << " data files" << COLOR_RESET << std::endl;
    
    initialize_session();
    
    // Load all data
    std::vector<ConversationPair> all_pairs;
    for (const auto& data_file : all_data_files) {
        std::vector<ConversationPair> pairs;
        load_conversation_pairs(data_file, pairs);
        all_pairs.insert(all_pairs.end(), pairs.begin(), pairs.end());
    }
    
    std::cout << COLOR_INFO << "📊 Total samples: " << all_pairs.size() << COLOR_RESET << std::endl;
    
    // Create trainer
    ChatbotTrainer trainer(config.base_config);
    
    // Model already owns tokenizer from construction - just transfer model
    trainer.set_model(std::move(model));
    
    for (const auto& pair : all_pairs) {
        trainer.add_training_pair(pair.input, pair.response);
    }
    
    bool success = trainer.train(num_epochs);
    
    // Get model back (model owns the tokenizer)
    model = trainer.release_model();
    
    if (success) {
        std::string checkpoint_path = generate_session_checkpoint_path();
        save_model(checkpoint_path);
        
        float final_loss = trainer.get_final_training_loss();
        float final_val_loss = trainer.get_final_validation_loss();
        finalize_session(all_pairs.size(), num_epochs, final_loss, final_val_loss);
        
        pending_data_files.clear();
        save_data_registry();
        save_session_history();
    }
    
    return success;
}

bool IncrementalTrainer::resume_last_session() {
    if (session_history.empty()) {
        std::cout << COLOR_WARNING << "⚠️  No previous sessions to resume" << COLOR_RESET << std::endl;
        return false;
    }
    
    const auto& last_session = session_history.back();
    
    // Validate checkpoint path
    if (last_session.checkpoint_path.empty()) {
        std::cerr << COLOR_ERROR << "❌ Invalid session: checkpoint path is empty" << COLOR_RESET << std::endl;
        return false;
    }
    
    // Check if checkpoint files exist (check for .config which should always be present)
    if (!fs::exists(last_session.checkpoint_path + ".config")) {
        std::cerr << COLOR_ERROR << "❌ Checkpoint file not found: " << last_session.checkpoint_path << ".config" << COLOR_RESET << std::endl;
        return false;
    }
    
    std::cout << COLOR_INFO << "🔄 Resuming from session #" << last_session.session_id << COLOR_RESET << std::endl;
    
    return load_model(last_session.checkpoint_path);
}

bool IncrementalTrainer::load_session_history() {
    std::string history_file = get_session_dir() + "/session_history.txt";
    
    if (!fs::exists(history_file)) {
        return false;
    }
    
    std::ifstream file(history_file);
    if (!file.is_open()) {
        return false;
    }
    
    session_history.clear();
    std::string line;
    
    // TODO: TD-004 - Check for version marker to determine if per-epoch metrics are available
    // TODO: TD-004 - Parse per-epoch loss values from extended format (comma-separated after checkpoint_path)
    // TODO: TD-004 - Parse per-epoch validation loss values with backward compatibility for old format
    // TODO: TD-004 - Parse per-epoch training times with fallback to zero if not present
    // TODO: TD-004 - Parse per-epoch learning rates with fallback to config default if not present
    // TODO: TD-004 - Parse per-epoch gradient norms for advanced debugging capabilities
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        TrainingSession session;
        
        iss >> session.session_id >> session.samples_trained >> session.epochs_completed
            >> session.final_loss >> session.final_validation_loss >> session.checkpoint_path;
        
        // Validate that parsing was successful
        if (iss.fail() || session.checkpoint_path.empty()) {
            std::cerr << COLOR_WARNING << "⚠️  Skipping malformed session history line: " << line << COLOR_RESET << std::endl;
            continue;
        }
        
        session_history.push_back(session);
        
        if (session.session_id >= current_session_id) {
            current_session_id = session.session_id + 1;
        }
    }
    
    std::cout << COLOR_INFO << "📜 Loaded " << session_history.size() << " previous sessions" << COLOR_RESET << std::endl;
    
    return true;
}

bool IncrementalTrainer::save_session_history() {
    std::string history_file = get_session_dir() + "/session_history.txt";
    
    std::ofstream file(history_file);
    if (!file.is_open()) {
        std::cerr << COLOR_ERROR << "❌ Failed to save session history" << COLOR_RESET << std::endl;
        return false;
    }
    
    // TODO: TD-004 - Add version header ("# VERSION 2") to indicate extended format with per-epoch metrics
    // TODO: TD-004 - Update header comment to include per-epoch metric columns
    file << "# Session History: session_id samples_trained epochs final_loss final_val_loss checkpoint_path\n";
    
    // TODO: TD-004 - Serialize per-epoch losses as comma-separated values after checkpoint_path
    // TODO: TD-004 - Serialize per-epoch validation losses in separate column or embedded JSON
    // TODO: TD-004 - Serialize per-epoch training times in ISO 8601 duration format or seconds
    // TODO: TD-004 - Serialize per-epoch learning rates with scientific notation for precision
    // TODO: TD-004 - Serialize per-epoch gradient norms for debugging training instability
    // TODO: TD-004 - Add optional JSON serialization mode for complex nested data structures
    
    for (const auto& session : session_history) {
        file << session.session_id << " " 
             << session.samples_trained << " "
             << session.epochs_completed << " "
             << session.final_loss << " "
             << session.final_validation_loss << " "
             << session.checkpoint_path << "\n";
    }
    
    return true;
}

TrainingSession IncrementalTrainer::get_current_session() const {
    if (session_history.empty()) {
        return TrainingSession();
    }
    return session_history.back();
}

std::vector<TrainingSession> IncrementalTrainer::get_session_history() const {
    return session_history;
}

void IncrementalTrainer::cleanup_old_sessions() {
    if (session_history.size() <= config.max_sessions_to_keep) {
        return;
    }
    
    int to_remove = session_history.size() - config.max_sessions_to_keep;
    
    for (int i = 0; i < to_remove; ++i) {
        const auto& session = session_history[i];
        
        // TD-005: Check if we're deleting the best checkpoint
        bool deleting_best = (session.checkpoint_path == best_checkpoint_path);
        
        // Delete checkpoint file
        if (fs::exists(session.checkpoint_path)) {
            fs::remove(session.checkpoint_path);
            std::cout << COLOR_INFO << "🗑️  Removed old checkpoint: " << session.checkpoint_path << COLOR_RESET << std::endl;
        }
        
        // If we deleted the best checkpoint, find the next best from remaining sessions
        if (deleting_best && config.enable_checkpoint_symlinks) {
            best_validation_loss = std::numeric_limits<float>::max();
            best_checkpoint_path = "";
            
            // Search through remaining sessions (after the ones we're removing)
            for (size_t j = to_remove; j < session_history.size(); ++j) {
                const auto& remaining_session = session_history[j];
                if (remaining_session.final_validation_loss < best_validation_loss && 
                    fs::exists(remaining_session.checkpoint_path)) {
                    best_validation_loss = remaining_session.final_validation_loss;
                    best_checkpoint_path = remaining_session.checkpoint_path;
                }
            }
            
            // Update the best_checkpoint symlink to new best (or remove if no sessions remain)
            if (!best_checkpoint_path.empty()) {
                update_best_checkpoint(best_validation_loss, best_checkpoint_path);
            } else {
                // No valid checkpoints remain, remove the symlink
                remove_symlink_if_exists(config.best_symlink_name);
            }
        }
    }
    
    // Remove from history
    session_history.erase(session_history.begin(), session_history.begin() + to_remove);
    
    // TD-005: Update latest checkpoint symlink to point to the newest remaining session
    if (config.enable_checkpoint_symlinks && !session_history.empty()) {
        const auto& latest_session = session_history.back();
        if (fs::exists(latest_session.checkpoint_path)) {
            update_checkpoint_symlinks(latest_session.checkpoint_path);
        }
    }
}

bool IncrementalTrainer::load_data_registry() {
    std::string registry_file = get_session_dir() + "/" + config.data_registry_file;
    
    if (!fs::exists(registry_file)) {
        return false;
    }
    
    std::ifstream file(registry_file);
    if (!file.is_open()) {
        return false;
    }
    
    data_registry.clear();
    trained_data_files.clear();
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        DataVersion dv;
        int trained_int;
        
        iss >> dv.data_file >> dv.checksum >> dv.num_samples >> trained_int;
        dv.trained = (trained_int == 1);
        
        data_registry.push_back(dv);
        
        if (dv.trained) {
            trained_data_files.insert(dv.data_file);
        }
    }
    
    std::cout << COLOR_INFO << "📋 Loaded data registry: " << data_registry.size() 
              << " files (" << trained_data_files.size() << " trained)" << COLOR_RESET << std::endl;
    
    return true;
}

bool IncrementalTrainer::save_data_registry() {
    std::string registry_file = get_session_dir() + "/" + config.data_registry_file;
    
    std::ofstream file(registry_file);
    if (!file.is_open()) {
        std::cerr << COLOR_ERROR << "❌ Failed to save data registry" << COLOR_RESET << std::endl;
        return false;
    }
    
    file << "# Data Registry: data_file checksum num_samples trained\n";
    
    for (const auto& dv : data_registry) {
        file << dv.data_file << " "
             << dv.checksum << " "
             << dv.num_samples << " "
             << (dv.trained ? 1 : 0) << "\n";
    }
    
    return true;
}

bool IncrementalTrainer::is_data_trained(const std::string& data_file) {
    return trained_data_files.find(data_file) != trained_data_files.end();
}

std::string IncrementalTrainer::compute_data_checksum(const std::string& data_file) {
    // Simple checksum: file size + modification time
    if (!fs::exists(data_file)) {
        return "MISSING";
    }
    
    auto size = fs::file_size(data_file);
    auto ftime = fs::last_write_time(data_file);
    
    std::ostringstream oss;
    oss << size << "_" << ftime.time_since_epoch().count();
    
    return oss.str();
}

bool IncrementalTrainer::save_model(const std::string& path) {
    try {
        model->save_model(path);
        std::cout << COLOR_SUCCESS << "💾 Model saved to: " << path << COLOR_RESET << std::endl;
        
        // TODO: TD-005 - Update "latest_checkpoint.bin" symlink after successful save
        // TODO: TD-005 - Check if this is best checkpoint and update "best_checkpoint.bin" symlink if needed
        // TODO: TD-005 - Use std::filesystem::create_symlink() with force overwrite for existing links
        // TODO: TD-005 - Add platform detection with #ifdef _WIN32 for Windows compatibility
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << COLOR_ERROR << "❌ Failed to save model: " << e.what() << COLOR_RESET << std::endl;
        return false;
    }
}

bool IncrementalTrainer::load_model(const std::string& path) {
    try {
        model->load_model(path);
        std::cout << COLOR_SUCCESS << "📂 Model loaded from: " << path << COLOR_RESET << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << COLOR_ERROR << "❌ Failed to load model: " << e.what() << COLOR_RESET << std::endl;
        return false;
    }
}

std::string IncrementalTrainer::get_latest_checkpoint() const {
    if (session_history.empty()) {
        return "";
    }
    return session_history.back().checkpoint_path;
}

void IncrementalTrainer::print_training_summary() const {
    std::cout << "\n" << COLOR_INFO << "╔══════════════════════════════════════════════╗" << COLOR_RESET << std::endl;
    std::cout << COLOR_INFO << "║       Incremental Training Summary           ║" << COLOR_RESET << std::endl;
    std::cout << COLOR_INFO << "╚══════════════════════════════════════════════╝" << COLOR_RESET << std::endl;
    
    std::cout << COLOR_SUCCESS << "📊 Total Sessions: " << session_history.size() << COLOR_RESET << std::endl;
    std::cout << COLOR_SUCCESS << "📦 Total Data Files Trained: " << trained_data_files.size() << COLOR_RESET << std::endl;
    std::cout << COLOR_SUCCESS << "📈 Total Samples Trained: " << get_total_samples_trained() << COLOR_RESET << std::endl;
    std::cout << COLOR_SUCCESS << "⏱️  Total Training Time: " << std::fixed << std::setprecision(2)
              << get_total_training_time_hours() << " hours" << COLOR_RESET << std::endl;
    
    // TODO: TD-004 - Display per-epoch loss progression graph using ASCII art or sparklines
    // TODO: TD-004 - Show validation loss trend to detect overfitting visually
    // TODO: TD-004 - Display learning rate schedule visualization across epochs
    // TODO: TD-004 - Show min/max/avg gradient norms for training stability assessment
    
    // TD-005: Display checkpoint symlink information
    if (config.enable_checkpoint_symlinks) {
        std::cout << "\n" << COLOR_INFO << "📎 Checkpoint Links:" << COLOR_RESET << std::endl;
        
        // Show latest checkpoint link
        if (fs::exists(config.latest_symlink_name)) {
            std::cout << "  Latest: " << config.latest_symlink_name;
            if (!is_windows_platform() && fs::is_symlink(config.latest_symlink_name)) {
                std::cout << " -> " << fs::read_symlink(config.latest_symlink_name).string();
            }
            std::cout << std::endl;
        }
        
        // Show best checkpoint link
        if (fs::exists(config.best_symlink_name)) {
            std::cout << "  Best: " << config.best_symlink_name;
            if (!is_windows_platform() && fs::is_symlink(config.best_symlink_name)) {
                std::cout << " -> " << fs::read_symlink(config.best_symlink_name).string();
            }
            std::cout << " (val loss: " << best_validation_loss << ")" << std::endl;
        }
    }
    
    if (!session_history.empty()) {
        const auto& last = session_history.back();
        std::cout << COLOR_INFO << "\nLast Session (#" << last.session_id << "):" << COLOR_RESET << std::endl;
        std::cout << "  Samples: " << last.samples_trained << std::endl;
        std::cout << "  Epochs: " << last.epochs_completed << std::endl;
        std::cout << "  Final Loss: " << last.final_loss << std::endl;
        std::cout << "  Validation Loss: " << last.final_validation_loss << std::endl;
        std::cout << "  Checkpoint: " << last.checkpoint_path << std::endl;
    }
    
    std::cout << std::endl;
}

void IncrementalTrainer::print_session_history() const {
    std::cout << COLOR_INFO << "\n📜 Session History:" << COLOR_RESET << std::endl;
    std::cout << "Session | Samples | Epochs | Loss   | Val Loss | Checkpoint" << std::endl;
    std::cout << "--------|---------|--------|--------|----------|------------" << std::endl;
    
    for (const auto& session : session_history) {
        std::cout << std::setw(7) << session.session_id << " | "
                  << std::setw(7) << session.samples_trained << " | "
                  << std::setw(6) << session.epochs_completed << " | "
                  << std::setw(6) << std::fixed << std::setprecision(3) << session.final_loss << " | "
                  << std::setw(8) << session.final_validation_loss << " | "
                  << session.checkpoint_path << std::endl;
    }
}

void IncrementalTrainer::print_data_registry() const {
    std::cout << COLOR_INFO << "\n📋 Data Registry:" << COLOR_RESET << std::endl;
    std::cout << "Trained | Samples | Data File" << std::endl;
    std::cout << "--------|---------|----------" << std::endl;
    
    for (const auto& dv : data_registry) {
        std::cout << std::setw(7) << (dv.trained ? "✓" : " ") << " | "
                  << std::setw(7) << dv.num_samples << " | "
                  << dv.data_file << std::endl;
    }
}

float IncrementalTrainer::get_total_training_time_hours() const {
    float total_hours = 0.0f;
    for (const auto& session : session_history) {
        auto duration = std::chrono::duration_cast<std::chrono::hours>(
            session.end_time - session.start_time);
        total_hours += duration.count();
    }
    return total_hours;
}

int IncrementalTrainer::get_total_samples_trained() const {
    int total = 0;
    for (const auto& dv : data_registry) {
        if (dv.trained) {
            total += dv.num_samples;
        }
    }
    return total;
}

bool IncrementalTrainer::initialize_session() {
    TrainingSession session;
    session.session_id = current_session_id;
    session.start_time = std::chrono::system_clock::now();
    session.samples_trained = 0;
    session.epochs_completed = 0;
    session.final_loss = 0.0f;
    session.final_validation_loss = 0.0f;
    
    session_history.push_back(session);
    last_save_time = std::chrono::system_clock::now();
    samples_since_last_save = 0;
    
    return true;
}

// See TD-004 in TECHNICAL_DEBT.md - Enhanced metrics tracking for training sessions
// See TD-005 in TECHNICAL_DEBT.md - Checkpoint management and symbolic links
bool IncrementalTrainer::finalize_session(int samples_trained, int epochs_completed, float final_loss, float final_val_loss) {
    if (session_history.empty()) {
        return false;
    }
    
    auto& session = session_history.back();
    session.end_time = std::chrono::system_clock::now();
    session.samples_trained = samples_trained;
    session.epochs_completed = epochs_completed;
    session.final_loss = final_loss;
    session.final_validation_loss = final_val_loss;
    session.checkpoint_path = generate_session_checkpoint_path();
    
    // TODO: TD-004 - Store collected per-epoch metrics in session.per_epoch_losses vector
    // TODO: TD-004 - Store per-epoch validation losses in session.per_epoch_validation_losses vector
    // TODO: TD-004 - Store per-epoch training times in session.per_epoch_training_times vector
    // TODO: TD-004 - Store per-epoch learning rates in session.per_epoch_learning_rates vector
    
    // TD-005: Checkpoint symlink management
    update_checkpoint_symlinks(session.checkpoint_path);
    update_best_checkpoint(final_val_loss, session.checkpoint_path);
    
    current_session_id++;
    
    // Cleanup old sessions
    cleanup_old_sessions();
    
    return true;
}

bool IncrementalTrainer::should_auto_save() {
    if (!config.auto_save_enabled) {
        return false;
    }
    
    // Check sample count
    if (config.auto_save_every_samples > 0 && 
        samples_since_last_save >= config.auto_save_every_samples) {
        return true;
    }
    
    // Check time elapsed
    if (config.auto_save_every_minutes > 0) {
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - last_save_time);
        if (elapsed.count() >= config.auto_save_every_minutes) {
            return true;
        }
    }
    
    return false;
}

void IncrementalTrainer::perform_auto_save() {
    std::string auto_save_path = get_session_dir() + "/auto_save_session_" + 
                                 std::to_string(current_session_id) + ".bin";
    
    if (save_model(auto_save_path)) {
        std::cout << COLOR_SUCCESS << "💾 Auto-saved checkpoint" << COLOR_RESET << std::endl;
        last_save_time = std::chrono::system_clock::now();
        samples_since_last_save = 0;
    }
}

std::string IncrementalTrainer::generate_session_checkpoint_path() {
    std::ostringstream oss;
    oss << get_session_dir() << "/session_" << current_session_id << "_checkpoint.bin";
    return oss.str();
}

std::string IncrementalTrainer::get_session_dir() const {
    return config.session_dir;
}

void IncrementalTrainer::ensure_directories_exist() {
    if (!fs::exists(config.session_dir)) {
        fs::create_directories(config.session_dir);
    }
    
    if (config.cache_tokenized_data && !fs::exists(config.tokenized_cache_dir)) {
        fs::create_directories(config.tokenized_cache_dir);
    }
}

bool IncrementalTrainer::save_pending_data_list() {
    std::string pending_file = get_session_dir() + "/pending_files.txt";
    std::ofstream file(pending_file);
    if (!file.is_open()) {
        return false;
    }
    
    for (const auto& data_file : pending_data_files) {
        file << data_file << "\n";
    }
    
    return true;
}

bool IncrementalTrainer::load_pending_data_list() {
    std::string pending_file = get_session_dir() + "/pending_files.txt";
    if (!fs::exists(pending_file)) {
        return false;
    }
    
    std::ifstream file(pending_file);
    if (!file.is_open()) {
        return false;
    }
    
    pending_data_files.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            pending_data_files.push_back(line);
        }
    }
    
    if (!pending_data_files.empty()) {
        std::cout << COLOR_INFO << "📋 Loaded " << pending_data_files.size() 
                  << " pending data files" << COLOR_RESET << std::endl;
    }
    
    return true;
}

// ============================================================================
// Symlink Management (TD-005)
// ============================================================================

bool IncrementalTrainer::is_windows_platform() const {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

bool IncrementalTrainer::create_or_update_symlink(const std::string& target, const std::string& link_path) {
    if (!config.enable_checkpoint_symlinks) {
        return false;
    }
    
    // Remove existing symlink/file at link_path
    remove_symlink_if_exists(link_path);
    
    try {
        if (is_windows_platform()) {
            // Windows fallback: copy file instead of symlink
            // Use std::filesystem::copy with overwrite
            std::error_code ec;
            fs::copy_file(target, link_path, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                std::cerr << COLOR_WARNING << "⚠️  Failed to copy checkpoint file: " 
                          << ec.message() << COLOR_RESET << std::endl;
                return false;
            }
            std::cout << COLOR_INFO << "📋 Copied checkpoint to: " << link_path 
                      << COLOR_RESET << std::endl;
        } else {
            // Unix/Linux: create symbolic link
            std::error_code ec;
            fs::create_symlink(target, link_path, ec);
            if (ec) {
                std::cerr << COLOR_WARNING << "⚠️  Failed to create symlink: " 
                          << ec.message() << COLOR_RESET << std::endl;
                return false;
            }
            std::cout << COLOR_INFO << "🔗 Created symlink: " << link_path 
                      << " -> " << target << COLOR_RESET << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << COLOR_WARNING << "⚠️  Failed to create/update checkpoint link: " 
                  << e.what() << COLOR_RESET << std::endl;
        return false;
    }
}

bool IncrementalTrainer::remove_symlink_if_exists(const std::string& link_path) {
    try {
        if (fs::exists(link_path)) {
            std::error_code ec;
            fs::remove(link_path, ec);
            if (ec) {
                std::cerr << COLOR_WARNING << "⚠️  Failed to remove existing link: " 
                          << ec.message() << COLOR_RESET << std::endl;
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << COLOR_WARNING << "⚠️  Error removing link: " 
                  << e.what() << COLOR_RESET << std::endl;
        return false;
    }
}

void IncrementalTrainer::update_checkpoint_symlinks(const std::string& checkpoint_path) {
    if (!config.enable_checkpoint_symlinks) {
        return;
    }
    
    // Create/update "latest_checkpoint.bin" symlink in root directory
    std::string latest_link = config.latest_symlink_name;
    if (!create_or_update_symlink(checkpoint_path, latest_link)) {
        std::cerr << COLOR_ERROR << "❌ Failed to update latest checkpoint symlink!" << COLOR_RESET << std::endl;
        std::cerr << COLOR_ERROR << "   Target: " << checkpoint_path << COLOR_RESET << std::endl;
        std::cerr << COLOR_ERROR << "   Link: " << latest_link << COLOR_RESET << std::endl;
    }
}

void IncrementalTrainer::update_best_checkpoint(float validation_loss, const std::string& checkpoint_path) {
    if (!config.enable_checkpoint_symlinks) {
        return;
    }
    
    // Check if this is the best validation loss so far
    bool is_best = false;
    
    if (session_history.size() <= 1) {
        // First session - always best
        is_best = true;
    } else {
        // Compare with previous best
        if (validation_loss < best_validation_loss) {
            is_best = true;
        }
    }
    
    if (is_best) {
        best_validation_loss = validation_loss;
        best_checkpoint_path = checkpoint_path;
        
        // Create/update "best_checkpoint.bin" symlink in root directory
        std::string best_link = config.best_symlink_name;
        if (!create_or_update_symlink(checkpoint_path, best_link)) {
            std::cerr << COLOR_ERROR << "❌ Failed to update best checkpoint symlink!" << COLOR_RESET << std::endl;
            std::cerr << COLOR_ERROR << "   Target: " << checkpoint_path << COLOR_RESET << std::endl;
            std::cerr << COLOR_ERROR << "   Link: " << best_link << COLOR_RESET << std::endl;
        } else {
            std::cout << COLOR_SUCCESS << "🏆 New best checkpoint! Validation loss: " 
                      << validation_loss << COLOR_RESET << std::endl;
        }
    }
}

std::string IncrementalTrainer::get_best_checkpoint_path() const {
    return best_checkpoint_path;
}

// ============================================================================

int IncrementalTrainer::load_conversation_pairs(const std::string& filepath,
                                                std::vector<ConversationPair>& pairs) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << COLOR_ERROR << "❌ Cannot open file: " << filepath << COLOR_RESET << std::endl;
        return 0;
    }
    
    std::string line;
    std::string current_input;
    std::string current_response;
    int pair_count = 0;
    
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        
        if (line.empty()) {
            // End of pair
            if (!current_input.empty() && !current_response.empty()) {
                pairs.emplace_back(current_input, current_response);
                pair_count++;
                current_input.clear();
                current_response.clear();
            }
            continue;
        }
        
        if (line.substr(0, 6) == "INPUT:") {
            current_input = line.substr(6);
            current_input.erase(0, current_input.find_first_not_of(" \t"));
        } else if (line.substr(0, 9) == "RESPONSE:") {
            current_response = line.substr(9);
            current_response.erase(0, current_response.find_first_not_of(" \t"));
        }
    }
    
    // Don't forget last pair
    if (!current_input.empty() && !current_response.empty()) {
        pairs.emplace_back(current_input, current_response);
        pair_count++;
    }
    
    file.close();
    
    std::cout << COLOR_INFO << "📖 Loaded " << pair_count << " pairs from: " << filepath << COLOR_RESET << std::endl;
    
    return pair_count;
}

// Project Gutenberg integration

std::string IncrementalTrainer::get_gutenberg_url(int book_id) const {
    // Project Gutenberg uses a tiered directory structure
    // Book 12345 is at: https://www.gutenberg.org/files/12345/12345-0.txt
    // Try UTF-8 version first (-0.txt), fallback to plain ASCII (.txt)
    std::ostringstream oss;
    oss << "https://www.gutenberg.org/files/" << book_id << "/" << book_id << "-0.txt";
    return oss.str();
}

bool IncrementalTrainer::download_file(const std::string& url, const std::string& output_path) {
    std::cout << COLOR_INFO << "📥 Downloading: " << url << COLOR_RESET << std::endl;
    
    // Use curl to download
    std::ostringstream cmd;
    cmd << "curl -L -f -s -o \"" << output_path << "\" \"" << url << "\"";
    
    int result = std::system(cmd.str().c_str());
    
    if (result == 0 && fs::exists(output_path) && fs::file_size(output_path) > 0) {
        std::cout << COLOR_SUCCESS << "✅ Downloaded to: " << output_path << COLOR_RESET << std::endl;
        return true;
    }
    
    // Try fallback URL (plain ASCII)
    if (url.find("-0.txt") != std::string::npos) {
        std::string fallback_url = url;
        size_t pos = fallback_url.find("-0.txt");
        fallback_url.replace(pos, 6, ".txt");
        
        std::cout << COLOR_INFO << "⚠️  Trying fallback URL..." << COLOR_RESET << std::endl;
        std::ostringstream fallback_cmd;
        fallback_cmd << "curl -L -f -s -o \"" << output_path << "\" \"" << fallback_url << "\"";
        
        result = std::system(fallback_cmd.str().c_str());
        
        if (result == 0 && fs::exists(output_path) && fs::file_size(output_path) > 0) {
            std::cout << COLOR_SUCCESS << "✅ Downloaded to: " << output_path << COLOR_RESET << std::endl;
            return true;
        }
    }
    
    std::cerr << COLOR_ERROR << "❌ Failed to download: " << url << COLOR_RESET << std::endl;
    return false;
}

bool IncrementalTrainer::download_gutenberg_book(int book_id, const std::string& output_dir) {
    // Create output directory if needed
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }
    
    std::string url = get_gutenberg_url(book_id);
    std::ostringstream output_path;
    output_path << output_dir << "/gutenberg_" << book_id << ".txt";
    
    return download_file(url, output_path.str());
}

bool IncrementalTrainer::download_gutenberg_books(const std::vector<int>& book_ids, 
                                                  const std::string& output_dir) {
    int success_count = 0;
    
    for (int book_id : book_ids) {
        if (download_gutenberg_book(book_id, output_dir)) {
            success_count++;
        }
    }
    
    std::cout << COLOR_INFO << "📚 Downloaded " << success_count << "/" << book_ids.size() 
              << " books" << COLOR_RESET << std::endl;
    
    return success_count > 0;
}

std::string IncrementalTrainer::clean_gutenberg_text(const std::string& raw_text) {
    std::string cleaned = raw_text;
    
    // Remove Project Gutenberg header (before "*** START OF")
    size_t start_pos = cleaned.find("*** START OF");
    if (start_pos != std::string::npos) {
        start_pos = cleaned.find("\n", start_pos);
        if (start_pos != std::string::npos) {
            cleaned = cleaned.substr(start_pos + 1);
        }
    }
    
    // Remove Project Gutenberg footer (after "*** END OF")
    size_t end_pos = cleaned.find("*** END OF");
    if (end_pos != std::string::npos) {
        cleaned = cleaned.substr(0, end_pos);
    }
    
    // Remove excessive whitespace
    cleaned = std::regex_replace(cleaned, std::regex("[ \\t]+"), " ");
    cleaned = std::regex_replace(cleaned, std::regex("\\n{3,}"), "\n\n");
    
    return cleaned;
}

std::vector<std::string> IncrementalTrainer::extract_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    
    // Simple sentence splitting on . ! ?
    std::regex sentence_regex("[^.!?]+[.!?]+");
    
    auto sentences_begin = std::sregex_iterator(text.begin(), text.end(), sentence_regex);
    auto sentences_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = sentences_begin; i != sentences_end; ++i) {
        std::string sentence = i->str();
        
        // Trim whitespace
        sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
        sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);
        
        // Skip very short or very long sentences
        if (sentence.length() > 20 && sentence.length() < 500) {
            sentences.push_back(sentence);
        }
    }
    
    return sentences;
}

std::string IncrementalTrainer::generate_question_from_sentence(const std::string& sentence) {
    // Simple question generation patterns
    std::vector<std::string> question_templates = {
        "What does this mean: ",
        "Can you explain: ",
        "Tell me about: ",
        "What is this about: ",
        "Explain this: ",
        "What does this say: "
    };
    
    // Select a random template
    int idx = rand() % question_templates.size();
    return question_templates[idx] + sentence;
}

std::vector<std::pair<std::string, std::string>> 
IncrementalTrainer::create_qa_pairs_from_text(const std::vector<std::string>& sentences, int max_pairs) {
    std::vector<std::pair<std::string, std::string>> pairs;
    
    // Create pairs from consecutive sentences
    for (size_t i = 0; i < sentences.size() - 1 && pairs.size() < max_pairs; i += 2) {
        std::string question = generate_question_from_sentence(sentences[i]);
        std::string answer = sentences[i + 1];
        
        // Also create reverse pairs (context + question)
        if (pairs.size() < max_pairs) {
            pairs.emplace_back(question, answer);
        }
        
        // Create "summarize" style pairs
        if (i + 2 < sentences.size() && pairs.size() < max_pairs) {
            std::string context = sentences[i] + " " + sentences[i + 1];
            std::string summary = sentences[i + 2];
            pairs.emplace_back("Summarize: " + context, summary);
        }
    }
    
    return pairs;
}

// See TD-006 in TECHNICAL_DEBT.md - Fill-in-the-Middle (FIM) training data generation
bool IncrementalTrainer::convert_gutenberg_to_training_data(const std::string& text_file,
                                                            const std::string& output_file,
                                                            int max_pairs) {
    std::cout << COLOR_INFO << "🔄 Converting Gutenberg text to training pairs..." << COLOR_RESET << std::endl;
    
    // Read the file
    std::ifstream file(text_file);
    if (!file.is_open()) {
        std::cerr << COLOR_ERROR << "❌ Cannot open: " << text_file << COLOR_RESET << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string raw_text = buffer.str();
    file.close();
    
    // Clean the text
    std::string cleaned_text = clean_gutenberg_text(raw_text);
    
    // Extract sentences
    std::vector<std::string> sentences = extract_sentences(cleaned_text);
    std::cout << COLOR_INFO << "📝 Extracted " << sentences.size() << " sentences" << COLOR_RESET << std::endl;
    
    if (sentences.empty()) {
        std::cerr << COLOR_ERROR << "❌ No valid sentences found" << COLOR_RESET << std::endl;
        return false;
    }
    
    // Create Q&A pairs
    auto pairs = create_qa_pairs_from_text(sentences, max_pairs);
    std::cout << COLOR_INFO << "💬 Created " << pairs.size() << " conversation pairs" << COLOR_RESET << std::endl;
    
    // Write to output file in INPUT/RESPONSE format
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << COLOR_ERROR << "❌ Cannot create: " << output_file << COLOR_RESET << std::endl;
        return false;
    }
    
    for (const auto& pair : pairs) {
        out << "INPUT: " << pair.first << "\n";
        out << "RESPONSE: " << pair.second << "\n";
        out << "\n";
    }
    
    out.close();
    
    std::cout << COLOR_SUCCESS << "✅ Training data saved to: " << output_file << COLOR_RESET << std::endl;
    
    return true;
}

bool IncrementalTrainer::add_gutenberg_book(int book_id, int num_pairs) {
    std::string output_dir = "gutenberg_data";
    
    // Download the book
    if (!download_gutenberg_book(book_id, output_dir)) {
        return false;
    }
    
    // Convert to training data
    std::ostringstream text_file, training_file;
    text_file << output_dir << "/gutenberg_" << book_id << ".txt";
    training_file << output_dir << "/gutenberg_" << book_id << "_training.txt";
    
    if (!convert_gutenberg_to_training_data(text_file.str(), training_file.str(), num_pairs)) {
        return false;
    }
    
    // Add to pending data
    return add_new_data(training_file.str());
}

bool IncrementalTrainer::add_gutenberg_books(const std::vector<int>& book_ids, int num_pairs_per_book) {
    int success_count = 0;
    
    for (int book_id : book_ids) {
        if (add_gutenberg_book(book_id, num_pairs_per_book)) {
            success_count++;
        }
    }
    
    std::cout << COLOR_INFO << "📚 Added " << success_count << "/" << book_ids.size() 
              << " books to training queue" << COLOR_RESET << std::endl;
    
    return success_count > 0;
}
