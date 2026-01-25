#ifndef CHECKPOINT_MANAGER_HPP
#define CHECKPOINT_MANAGER_HPP

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file CheckpointManager.hpp
 * @brief Enhanced checkpoint management for model training
 * 
 * Features:
 * - Automatic checkpoint rotation (keep N best models)
 * - Best model tracking by validation loss
 * - Checkpoint metadata (epoch, loss, timestamp)
 * - Directory-based organization
 * - Automatic cleanup of old checkpoints
 * 
 * @version 1.0
 * @date January 2026
 */

/**
 * @brief Checkpoint metadata
 */
struct CheckpointInfo {
    std::string filepath;
    int epoch;
    float train_loss;
    float validation_loss;
    std::time_t timestamp;
    bool is_best;
    
    CheckpointInfo()
        : filepath(""), epoch(0), train_loss(0.0f), 
          validation_loss(0.0f), timestamp(0), is_best(false) {}
    
    CheckpointInfo(const std::string& path, int ep, float train_l, float val_l, 
                   std::time_t ts, bool best = false)
        : filepath(path), epoch(ep), train_loss(train_l), 
          validation_loss(val_l), timestamp(ts), is_best(best) {}
};

/**
 * @brief Checkpoint manager for model training
 * 
 * Manages model checkpoints with automatic rotation and best model tracking.
 * 
 * Features:
 * - Save checkpoints with metadata
 * - Automatic rotation (keep N best checkpoints)
 * - Track best model by validation loss
 * - Load checkpoint metadata
 * - Clean up old checkpoints
 * 
 * Example usage:
 * @code
 * CheckpointManager manager("checkpoints/", 5);  // Keep 5 best checkpoints
 * 
 * for (int epoch = 0; epoch < num_epochs; epoch++) {
 *     float train_loss = train_epoch();
 *     float val_loss = validate();
 *     
 *     manager.save_checkpoint(epoch, train_loss, val_loss);
 * }
 * 
 * std::string best_model = manager.get_best_checkpoint_path();
 * @endcode
 */
class CheckpointManager {
private:
    std::string checkpoint_dir_;
    int max_checkpoints_;
    std::vector<CheckpointInfo> checkpoints_;
    std::string best_checkpoint_path_;
    float best_validation_loss_;
    
    /**
     * @brief Generate checkpoint filename
     */
    std::string generate_checkpoint_filename(int epoch) const {
        std::ostringstream oss;
        oss << "checkpoint_epoch_" << std::setw(4) << std::setfill('0') << epoch << ".bin";
        return oss.str();
    }
    
    /**
     * @brief Generate metadata filename for a checkpoint
     */
    std::string get_metadata_filename(const std::string& checkpoint_path) const {
        return checkpoint_path + ".meta";
    }
    
    /**
     * @brief Save checkpoint metadata
     */
    bool save_metadata(const CheckpointInfo& info) const {
        std::string meta_path = get_metadata_filename(info.filepath);
        std::ofstream file(meta_path);
        if (!file.is_open()) {
            return false;
        }
        
        file << "epoch=" << info.epoch << "\n";
        file << "train_loss=" << info.train_loss << "\n";
        file << "validation_loss=" << info.validation_loss << "\n";
        file << "timestamp=" << info.timestamp << "\n";
        file << "is_best=" << (info.is_best ? "true" : "false") << "\n";
        
        file.close();
        return true;
    }
    
    /**
     * @brief Load checkpoint metadata
     */
    CheckpointInfo load_metadata(const std::string& checkpoint_path) const {
        std::string meta_path = get_metadata_filename(checkpoint_path);
        std::ifstream file(meta_path);
        
        CheckpointInfo info;
        info.filepath = checkpoint_path;
        
        if (!file.is_open()) {
            return info;  // Return empty info if metadata not found
        }
        
        std::string line;
        while (std::getline(file, line)) {
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);
                
                if (key == "epoch") {
                    info.epoch = std::stoi(value);
                } else if (key == "train_loss") {
                    info.train_loss = std::stof(value);
                } else if (key == "validation_loss") {
                    info.validation_loss = std::stof(value);
                } else if (key == "timestamp") {
                    info.timestamp = std::stol(value);
                } else if (key == "is_best") {
                    info.is_best = (value == "true");
                }
            }
        }
        
        file.close();
        return info;
    }
    
    /**
     * @brief Delete checkpoint and its metadata
     */
    bool delete_checkpoint(const std::string& filepath) {
        bool success = true;
        
        // Delete model file
        try {
            if (std::filesystem::exists(filepath)) {
                std::filesystem::remove(filepath);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error deleting checkpoint: " << e.what() << std::endl;
            success = false;
        }
        
        // Delete metadata file
        std::string meta_path = get_metadata_filename(filepath);
        try {
            if (std::filesystem::exists(meta_path)) {
                std::filesystem::remove(meta_path);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error deleting metadata: " << e.what() << std::endl;
            success = false;
        }
        
        return success;
    }
    
    /**
     * @brief Rotate checkpoints (keep only N best)
     */
    void rotate_checkpoints() {
        if (checkpoints_.size() <= static_cast<size_t>(max_checkpoints_)) {
            return;  // No rotation needed
        }
        
        // Sort by validation loss (lower is better)
        std::sort(checkpoints_.begin(), checkpoints_.end(),
                 [](const CheckpointInfo& a, const CheckpointInfo& b) {
                     // Handle case where validation loss might be 0 (not computed)
                     if (a.validation_loss == 0.0f && b.validation_loss == 0.0f) {
                         return a.train_loss < b.train_loss;
                     }
                     if (a.validation_loss == 0.0f) return false;
                     if (b.validation_loss == 0.0f) return true;
                     return a.validation_loss < b.validation_loss;
                 });
        
        // Delete checkpoints beyond max_checkpoints_
        for (size_t i = max_checkpoints_; i < checkpoints_.size(); ++i) {
            if (!checkpoints_[i].is_best) {  // Don't delete best checkpoint
                std::cout << "Rotating out checkpoint: " << checkpoints_[i].filepath << std::endl;
                delete_checkpoint(checkpoints_[i].filepath);
            }
        }
        
        // Keep only the best checkpoints
        checkpoints_.erase(checkpoints_.begin() + max_checkpoints_, checkpoints_.end());
    }
    
public:
    /**
     * @brief Constructor
     * @param checkpoint_dir Directory to store checkpoints
     * @param max_checkpoints Maximum number of checkpoints to keep (default: 5)
     */
    CheckpointManager(const std::string& checkpoint_dir = "checkpoints/", 
                     int max_checkpoints = 5)
        : checkpoint_dir_(checkpoint_dir),
          max_checkpoints_(max_checkpoints),
          best_checkpoint_path_(""),
          best_validation_loss_(std::numeric_limits<float>::max()) {
        
        // Create checkpoint directory if it doesn't exist
        try {
            std::filesystem::create_directories(checkpoint_dir_);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not create checkpoint directory: " 
                     << e.what() << std::endl;
        }
        
        // Load existing checkpoints
        load_existing_checkpoints();
    }
    
    /**
     * @brief Load existing checkpoints from directory
     */
    void load_existing_checkpoints() {
        checkpoints_.clear();
        
        try {
            if (!std::filesystem::exists(checkpoint_dir_)) {
                return;
            }
            
            for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir_)) {
                if (entry.is_regular_file() && 
                    entry.path().extension() == ".bin" &&
                    entry.path().filename().string().find("checkpoint_") == 0) {
                    
                    CheckpointInfo info = load_metadata(entry.path().string());
                    if (info.epoch > 0 || info.train_loss > 0.0f) {  // Valid metadata
                        checkpoints_.push_back(info);
                        
                        // Track best checkpoint
                        if (info.is_best || 
                            (info.validation_loss > 0.0f && 
                             info.validation_loss < best_validation_loss_)) {
                            best_validation_loss_ = info.validation_loss;
                            best_checkpoint_path_ = info.filepath;
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Error loading existing checkpoints: " 
                     << e.what() << std::endl;
        }
        
        std::cout << "Loaded " << checkpoints_.size() << " existing checkpoints" << std::endl;
    }
    
    /**
     * @brief Save a checkpoint
     * 
     * Note: This only manages metadata. The caller must actually save the model.
     * 
     * @param epoch Current epoch
     * @param train_loss Training loss
     * @param validation_loss Validation loss
     * @return Path to checkpoint file
     */
    std::string save_checkpoint(int epoch, float train_loss, float validation_loss = 0.0f) {
        std::string filename = generate_checkpoint_filename(epoch);
        std::string filepath = checkpoint_dir_ + filename;
        
        bool is_best = false;
        if (validation_loss > 0.0f && validation_loss < best_validation_loss_) {
            best_validation_loss_ = validation_loss;
            best_checkpoint_path_ = filepath;
            is_best = true;
            
            // Mark previous best as not best
            for (auto& ckpt : checkpoints_) {
                ckpt.is_best = false;
            }
        }
        
        CheckpointInfo info(filepath, epoch, train_loss, validation_loss, 
                           std::time(nullptr), is_best);
        
        checkpoints_.push_back(info);
        save_metadata(info);
        
        // Rotate if needed
        rotate_checkpoints();
        
        return filepath;
    }
    
    /**
     * @brief Get path to best checkpoint
     * @return Path to best checkpoint file
     */
    std::string get_best_checkpoint_path() const {
        return best_checkpoint_path_;
    }
    
    /**
     * @brief Get best validation loss
     * @return Best validation loss value
     */
    float get_best_validation_loss() const {
        return best_validation_loss_;
    }
    
    /**
     * @brief Get all checkpoint info
     * @return Vector of checkpoint information
     */
    const std::vector<CheckpointInfo>& get_checkpoints() const {
        return checkpoints_;
    }
    
    /**
     * @brief Get checkpoint info for specific epoch
     * @param epoch Epoch number
     * @return Checkpoint info (or empty if not found)
     */
    CheckpointInfo get_checkpoint_info(int epoch) const {
        for (const auto& ckpt : checkpoints_) {
            if (ckpt.epoch == epoch) {
                return ckpt;
            }
        }
        return CheckpointInfo();
    }
    
    /**
     * @brief Check if a checkpoint exists for an epoch
     * @param epoch Epoch number
     * @return True if checkpoint exists
     */
    bool has_checkpoint(int epoch) const {
        for (const auto& ckpt : checkpoints_) {
            if (ckpt.epoch == epoch) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Delete all checkpoints
     */
    void clear_all() {
        for (const auto& ckpt : checkpoints_) {
            delete_checkpoint(ckpt.filepath);
        }
        checkpoints_.clear();
        best_checkpoint_path_.clear();
        best_validation_loss_ = std::numeric_limits<float>::max();
    }
    
    /**
     * @brief Print checkpoint summary
     */
    void print_summary() const {
        std::cout << "\n╔══════════════════════════════════════════════╗\n";
        std::cout << "║         Checkpoint Summary                  ║\n";
        std::cout << "╚══════════════════════════════════════════════╝\n";
        std::cout << "  Total checkpoints: " << checkpoints_.size() << "\n";
        std::cout << "  Max checkpoints: " << max_checkpoints_ << "\n";
        
        if (!best_checkpoint_path_.empty()) {
            std::cout << "  Best checkpoint: " << best_checkpoint_path_ << "\n";
            std::cout << "  Best val loss: " << best_validation_loss_ << "\n";
        }
        
        if (!checkpoints_.empty()) {
            std::cout << "\n  Recent checkpoints:\n";
            std::cout << std::fixed << std::setprecision(4);
            
            // Show last 5 checkpoints
            size_t start = checkpoints_.size() > 5 ? checkpoints_.size() - 5 : 0;
            for (size_t i = start; i < checkpoints_.size(); ++i) {
                const auto& ckpt = checkpoints_[i];
                std::cout << "    Epoch " << ckpt.epoch 
                         << " - Train: " << ckpt.train_loss;
                if (ckpt.validation_loss > 0.0f) {
                    std::cout << " - Val: " << ckpt.validation_loss;
                }
                if (ckpt.is_best) {
                    std::cout << " ⭐";
                }
                std::cout << "\n";
            }
        }
        
        std::cout << "══════════════════════════════════════════════\n\n";
    }
    
    /**
     * @brief Set maximum number of checkpoints to keep
     * @param max_checkpoints Maximum number
     */
    void set_max_checkpoints(int max_checkpoints) {
        max_checkpoints_ = max_checkpoints;
        rotate_checkpoints();
    }
    
    /**
     * @brief Get checkpoint directory
     * @return Directory path
     */
    const std::string& get_checkpoint_dir() const {
        return checkpoint_dir_;
    }
};

#endif  // CHECKPOINT_MANAGER_HPP
