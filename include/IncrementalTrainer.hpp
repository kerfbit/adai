#pragma once

#include <chrono>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "BPETokenizer.hpp"
#include "ChatbotTrainer.hpp"
#include "EncoderDecoderModel.hpp"

/**
 * @brief Training session metadata for incremental learning
 */
struct TrainingSession {
    int session_id;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    int samples_trained;
    int epochs_completed;
    float final_loss;
    float final_validation_loss;
    std::string checkpoint_path;
    std::vector<std::string> data_files_used;
    
    // TODO: TD-004 - Add per-epoch loss tracking vector for training visualization
    // TODO: TD-004 - Add per-epoch validation loss tracking vector for overfitting detection
    // TODO: TD-004 - Add per-epoch training time tracking for performance analysis
    // TODO: TD-004 - Add per-epoch learning rate tracking for schedule effectiveness analysis
    // TODO: TD-004 - Add per-epoch gradient norm tracking for training stability monitoring
};

/**
 * @brief Data versioning for tracking trained samples
 */
struct DataVersion {
    std::string data_file;
    std::string checksum;  // MD5 or simple hash
    int num_samples;
    std::chrono::system_clock::time_point added_time;
    bool trained;
};

/**
 * @brief Configuration for incremental training
 */
struct IncrementalConfig {
    // Base training config
    TrainingConfig base_config;
    
    // Incremental-specific settings
    std::string session_dir = "training_sessions";
    std::string data_registry_file = "data_registry.txt";
    int max_sessions_to_keep = 10;  // Keep last N session checkpoints
    
    // Training modes
    bool accumulate_all_data = false;  // Train on all data each session vs only new data
    bool periodic_full_retrain = false;  // Periodically retrain on all data
    int full_retrain_every_n_sessions = 10;
    
    // Efficiency settings
    bool cache_tokenized_data = true;
    std::string tokenized_cache_dir = "tokenized_cache";
    int micro_batch_size = 1;  // For very long training, save every N samples
    
    // Auto-save during long training
    bool auto_save_enabled = true;
    int auto_save_every_minutes = 30;  // Save checkpoint every N minutes
    int auto_save_every_samples = 1000;  // Save checkpoint every N samples
};

/**
 * @brief Incremental Training System
 * 
 * Supports ongoing training with new data without retraining from scratch:
 * - Session-based training with automatic checkpointing
 * - Data versioning and tracking
 * - Resume from interruption
 * - Add new training data incrementally
 * - Efficient caching of tokenized data
 * - Time-based and sample-based auto-save
 * - Training history and statistics
 * 
 * Example usage:
 * ```
 * IncrementalTrainer trainer("vocab.txt", "chatbot_model.bin");
 * trainer.add_new_data("new_conversations.txt");
 * trainer.train_incremental(5);  // 5 epochs on new data
 * ```
 */
class IncrementalTrainer {
private:
    IncrementalConfig config;
    std::unique_ptr<EncoderDecoderModel> model;
    
    // Session management
    int current_session_id;
    std::vector<TrainingSession> session_history;
    std::vector<DataVersion> data_registry;
    
    // Data tracking
    std::set<std::string> trained_data_files;
    std::vector<std::string> pending_data_files;
    
    // Internal state
    std::chrono::system_clock::time_point last_save_time;
    int samples_since_last_save;
    
public:
    IncrementalTrainer(const std::string& vocab_path, const std::string& model_path);
    IncrementalTrainer(const std::string& vocab_path, const std::string& model_path, 
                      const IncrementalConfig& cfg);
    
    // Setup and configuration
    void set_config(const IncrementalConfig& cfg);
    IncrementalConfig& get_config();
    
    // Data management
    bool add_new_data(const std::string& data_file);
    bool add_new_data_batch(const std::vector<std::string>& data_files);
    void clear_pending_data();
    std::vector<std::string> get_pending_data_files() const;
    std::vector<std::string> get_trained_data_files() const;
    
    // Project Gutenberg integration
    bool download_gutenberg_book(int book_id, const std::string& output_dir = "gutenberg_data");
    bool download_gutenberg_books(const std::vector<int>& book_ids, const std::string& output_dir = "gutenberg_data");
    bool add_gutenberg_book(int book_id, int num_pairs = 500);
    bool add_gutenberg_books(const std::vector<int>& book_ids, int num_pairs_per_book = 500);
    std::string get_gutenberg_url(int book_id) const;
    bool convert_gutenberg_to_training_data(const std::string& text_file, 
                                            const std::string& output_file,
                                            int max_pairs = 500);
    
    // Training operations
    bool train_incremental(int num_epochs = 1);
    bool train_on_new_data_only(int num_epochs = 1);
    bool train_full_retrain(int num_epochs = 1);
    bool resume_last_session();
    
    // Session management
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
    
    // Statistics and reporting
    void print_training_summary() const;
    void print_session_history() const;
    void print_data_registry() const;
    float get_total_training_time_hours() const;
    int get_total_samples_trained() const;
    
private:
    // Internal helpers
    bool initialize_session();
    bool finalize_session(int samples_trained, int epochs_completed, float final_loss, float final_val_loss);
    bool should_auto_save();
    void perform_auto_save();
    std::string generate_session_checkpoint_path();
    std::string get_session_dir() const;
    void ensure_directories_exist();
    int load_conversation_pairs(const std::string& filepath, 
                               std::vector<ConversationPair>& pairs);
    
    // TODO: TD-005 - Add update_checkpoint_symlinks(checkpoint_path) to manage latest symlink
    // TODO: TD-005 - Add update_best_checkpoint(validation_loss, checkpoint_path) to track best model
    // TODO: TD-005 - Add get_best_checkpoint_path() to retrieve best model across all sessions
    // TODO: TD-005 - Add is_windows_platform() helper for platform-specific symlink handling
    // TODO: TD-005 - Add create_or_update_symlink(target, link_path) with Windows fallback logic
    // TODO: TD-005 - Add remove_symlink_if_exists(link_path) for cleanup operations
    
    // Gutenberg helpers
    bool download_file(const std::string& url, const std::string& output_path);
    std::string clean_gutenberg_text(const std::string& raw_text);
    std::vector<std::string> extract_sentences(const std::string& text);
    std::vector<std::pair<std::string, std::string>> create_qa_pairs_from_text(
        const std::vector<std::string>& sentences, int max_pairs);
    std::string generate_question_from_sentence(const std::string& sentence);
};
