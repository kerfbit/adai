#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <set>
#include "ChatbotTrainer.hpp"
#include "BPETokenizer.hpp"
#include "EncoderDecoderModel.hpp"

/**
 * @brief Training session information
 */
struct TrainingSession {
    int session_id = 0;
    int samples_trained = 0;
    int epochs_completed = 0;
    float final_loss = 0.0f;
    float final_validation_loss = 0.0f;
    std::string checkpoint_path;
    std::chrono::system_clock::time_point timestamp;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
};

/**
 * @brief Data file version tracking
 */
struct DataVersion {
    std::string data_file;
    std::string checksum;
    int num_samples = 0;
    std::chrono::system_clock::time_point added_time;
    bool trained = false;
};

/**
 * @brief Incremental training configuration
 */
struct IncrementalConfig {
    TrainingConfig base_config;  // Base training configuration
    
    // Session management
    std::string session_dir = "training_sessions";
    int max_sessions_to_keep = 50;
    
    // Data management
    std::string data_registry_file = "data_registry.txt";
    bool cache_tokenized_data = false;
    std::string tokenized_cache_dir = "tokenized_cache";
    
    // Auto-save settings
    bool auto_save_enabled = true;
    int auto_save_every_samples = 1000;
    int auto_save_every_minutes = 30;
    
    // Checkpointing
    bool save_incremental_checkpoints = true;
    std::string checkpoint_dir = "checkpoints";
    
    // Checkpoint symlink management (TD-005)
    bool enable_checkpoint_symlinks = true;  // Create latest/best symlinks
    std::string latest_symlink_name = "latest_checkpoint.bin";  // Name for latest checkpoint symlink
    std::string best_symlink_name = "best_checkpoint.bin";  // Name for best checkpoint symlink
};

/**
 * @brief Incremental Training Manager
 * 
 * Manages ongoing training sessions with:
 * - Session-based training history
 * - Data versioning and tracking
 * - Automatic checkpointing
 * - Incremental data addition
 * - Resume capability
 * - Project Gutenberg integration
 */
class IncrementalTrainer {
public:
    /**
     * @brief Constructor
     * @param vocab_path Path to vocabulary file
     * @param model_path Path to model checkpoint
     */
    IncrementalTrainer(const std::string& vocab_path, const std::string& model_path);
    
    /**
     * @brief Constructor with configuration
     * @param vocab_path Path to vocabulary file
     * @param model_path Path to model checkpoint
     * @param cfg Configuration settings
     */
    IncrementalTrainer(const std::string& vocab_path, const std::string& model_path,
                      const IncrementalConfig& cfg);
    
    // Configuration
    void set_config(const IncrementalConfig& cfg);
    IncrementalConfig& get_config();
    
    // Data management
    bool add_new_data(const std::string& data_file);
    bool add_new_data_batch(const std::vector<std::string>& data_files);
    void clear_pending_data();
    std::vector<std::string> get_pending_data_files() const;
    std::vector<std::string> get_trained_data_files() const;
    
    // Training
    bool train_incremental(int num_epochs);
    bool train_on_new_data_only(int num_epochs);
    bool train_full_retrain(int num_epochs);
    
    // Session management
    bool resume_last_session();
    bool load_session_history();
    bool save_session_history();
    TrainingSession get_current_session() const;
    std::vector<TrainingSession> get_session_history() const;
    void cleanup_old_sessions();
    
    // Data registry
    bool load_data_registry();
    bool save_data_registry();
    bool is_data_trained(const std::string& data_file);
    std::string compute_data_checksum(const std::string& data_file);
    
    // Model operations
    bool save_model(const std::string& path);
    bool load_model(const std::string& path);
    std::string get_latest_checkpoint() const;
    
    // Status and reporting
    void print_training_summary() const;
    void print_session_history() const;
    void print_data_registry() const;
    int get_total_samples_trained() const;
    float get_total_training_time_hours() const;
    
    // Project Gutenberg integration
    bool add_gutenberg_book(int book_id, int num_pairs = 500);
    bool add_gutenberg_books(const std::vector<int>& book_ids, int num_pairs_per_book = 300);
    
private:
    // Training components
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<EncoderDecoderModel> model;
    IncrementalConfig config;
    
    // Session tracking
    std::vector<TrainingSession> session_history;
    int current_session_id;
    
    // Data tracking
    std::vector<DataVersion> data_registry;
    std::set<std::string> trained_data_files;
    std::vector<std::string> pending_data_files;
    
    // Auto-save state
    std::chrono::system_clock::time_point last_save_time;
    int samples_since_last_save;
    
    // Best checkpoint tracking (TD-005)
    float best_validation_loss;
    std::string best_checkpoint_path;
    
    // Helper methods
    bool initialize_session();
    bool finalize_session(int samples_trained, int epochs_completed, float final_loss, float final_val_loss);
    bool should_auto_save();
    void perform_auto_save();
    std::string generate_session_checkpoint_path();
    std::string get_session_dir() const;
    void ensure_directories_exist();
    bool save_pending_data_list();
    bool load_pending_data_list();
    int load_conversation_pairs(const std::string& filepath, std::vector<ConversationPair>& pairs);
    
    // Symlink management helpers (TD-005)
    void update_checkpoint_symlinks(const std::string& checkpoint_path);
    void update_best_checkpoint(float validation_loss, const std::string& checkpoint_path);
    std::string get_best_checkpoint_path() const;
    bool is_windows_platform() const;
    bool create_or_update_symlink(const std::string& target, const std::string& link_path);
    bool remove_symlink_if_exists(const std::string& link_path);
    
    // Project Gutenberg helpers
    std::string get_gutenberg_url(int book_id) const;
    bool download_file(const std::string& url, const std::string& output_path);
    bool download_gutenberg_book(int book_id, const std::string& output_dir);
    bool download_gutenberg_books(const std::vector<int>& book_ids, const std::string& output_dir);
    std::string clean_gutenberg_text(const std::string& raw_text);
    std::vector<std::string> extract_sentences(const std::string& text);
    std::string generate_question_from_sentence(const std::string& sentence);
    std::vector<std::pair<std::string, std::string>> create_qa_pairs_from_text(
        const std::vector<std::string>& sentences, int max_pairs);
    bool convert_gutenberg_to_training_data(const std::string& text_file,
                                           const std::string& output_file, int max_pairs);
};

