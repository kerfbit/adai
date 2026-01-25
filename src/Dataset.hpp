#ifndef DATASET_HPP
#define DATASET_HPP

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

/**
 * @file Dataset.hpp
 * @brief Dataset abstraction for training data management
 * 
 * Provides efficient data loading, batching, and splitting functionality
 * for training transformer models.
 * 
 * @version 1.0
 * @date January 2026
 */

/**
 * @brief Training data sample (input, target response)
 */
struct DataSample {
    std::string input;
    std::string target;
    
    DataSample() = default;
    DataSample(const std::string& in, const std::string& tgt) 
        : input(in), target(tgt) {}
};

/**
 * @brief Dataset split types
 */
enum class SplitType {
    TRAIN,      // Training data
    VALIDATION, // Validation data
    TEST        // Test data
};

/**
 * @brief Dataset statistics
 */
struct DatasetStats {
    size_t total_samples;
    size_t train_samples;
    size_t validation_samples;
    size_t test_samples;
    size_t avg_input_length;
    size_t avg_target_length;
    size_t max_input_length;
    size_t max_target_length;
    
    DatasetStats() 
        : total_samples(0), train_samples(0), validation_samples(0), test_samples(0),
          avg_input_length(0), avg_target_length(0), 
          max_input_length(0), max_target_length(0) {}
};

/**
 * @brief Dataset abstraction for (input, response) pairs
 * 
 * Features:
 * - Load data from various file formats
 * - Automatic train/validation/test splitting
 * - Data shuffling and batching
 * - Dataset statistics and analysis
 * - Memory-efficient iteration
 * - Support for large datasets
 * 
 * Example usage:
 * @code
 * Dataset dataset;
 * dataset.load_from_file("conversations.txt");
 * dataset.split(0.8, 0.1, 0.1);  // 80% train, 10% val, 10% test
 * dataset.shuffle();
 * 
 * auto train_data = dataset.get_split(SplitType::TRAIN);
 * for (const auto& sample : train_data) {
 *     // Train on sample.input and sample.target
 * }
 * @endcode
 */
class Dataset {
private:
    std::vector<DataSample> data_;
    std::vector<size_t> train_indices_;
    std::vector<size_t> validation_indices_;
    std::vector<size_t> test_indices_;
    
    DatasetStats stats_;
    bool is_split_;
    std::mt19937 rng_;
    
    /**
     * @brief Parse a line-based conversation format
     * 
     * Format:
     * INPUT: <user message>
     * RESPONSE: <bot response>
     * (blank line between pairs)
     */
    bool parse_conversation_format(std::ifstream& file) {
        std::string line;
        std::string current_input;
        std::string current_response;
        size_t pair_count = 0;
        
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\n\r"));
            line.erase(line.find_last_not_of(" \t\n\r") + 1);
            
            if (line.empty()) {
                // End of pair
                if (!current_input.empty() && !current_response.empty()) {
                    data_.emplace_back(current_input, current_response);
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
            data_.emplace_back(current_input, current_response);
            pair_count++;
        }
        
        return pair_count > 0;
    }
    
    /**
     * @brief Parse a tab-separated format
     * 
     * Format: <input>\t<target>\n
     */
    bool parse_tsv_format(std::ifstream& file) {
        std::string line;
        size_t pair_count = 0;
        
        while (std::getline(file, line)) {
            size_t tab_pos = line.find('\t');
            if (tab_pos != std::string::npos) {
                std::string input = line.substr(0, tab_pos);
                std::string target = line.substr(tab_pos + 1);
                
                // Trim
                input.erase(0, input.find_first_not_of(" \t\n\r"));
                input.erase(input.find_last_not_of(" \t\n\r") + 1);
                target.erase(0, target.find_first_not_of(" \t\n\r"));
                target.erase(target.find_last_not_of(" \t\n\r") + 1);
                
                if (!input.empty() && !target.empty()) {
                    data_.emplace_back(input, target);
                    pair_count++;
                }
            }
        }
        
        return pair_count > 0;
    }
    
    /**
     * @brief Calculate dataset statistics
     */
    void calculate_statistics() {
        stats_.total_samples = data_.size();
        stats_.train_samples = train_indices_.size();
        stats_.validation_samples = validation_indices_.size();
        stats_.test_samples = test_indices_.size();
        
        if (data_.empty()) {
            return;
        }
        
        size_t total_input_length = 0;
        size_t total_target_length = 0;
        stats_.max_input_length = 0;
        stats_.max_target_length = 0;
        
        for (const auto& sample : data_) {
            size_t input_len = sample.input.length();
            size_t target_len = sample.target.length();
            
            total_input_length += input_len;
            total_target_length += target_len;
            
            stats_.max_input_length = std::max(stats_.max_input_length, input_len);
            stats_.max_target_length = std::max(stats_.max_target_length, target_len);
        }
        
        stats_.avg_input_length = total_input_length / data_.size();
        stats_.avg_target_length = total_target_length / data_.size();
    }
    
public:
    /**
     * @brief Constructor
     * @param seed Random seed for shuffling (default: 42)
     */
    Dataset(unsigned int seed = 42) 
        : is_split_(false), rng_(seed) {}
    
    /**
     * @brief Load data from file
     * 
     * Automatically detects format:
     * - Conversation format (INPUT:/RESPONSE:)
     * - TSV format (input\ttarget)
     * 
     * @param filepath Path to data file
     * @return True if successful
     */
    bool load_from_file(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file: " << filepath << std::endl;
            return false;
        }
        
        // Try to detect format from first line
        std::string first_line;
        std::getline(file, first_line);
        file.seekg(0);  // Reset to beginning
        
        bool success = false;
        if (first_line.find("INPUT:") != std::string::npos) {
            success = parse_conversation_format(file);
        } else if (first_line.find('\t') != std::string::npos) {
            success = parse_tsv_format(file);
        } else {
            // Default to conversation format
            success = parse_conversation_format(file);
        }
        
        file.close();
        
        if (success) {
            calculate_statistics();
        }
        
        return success;
    }
    
    /**
     * @brief Add a single sample to the dataset
     * @param input Input text
     * @param target Target text
     */
    void add_sample(const std::string& input, const std::string& target) {
        data_.emplace_back(input, target);
        calculate_statistics();
    }
    
    /**
     * @brief Add multiple samples to the dataset
     * @param samples Vector of data samples
     */
    void add_samples(const std::vector<DataSample>& samples) {
        data_.insert(data_.end(), samples.begin(), samples.end());
        calculate_statistics();
    }
    
    /**
     * @brief Split dataset into train/validation/test sets
     * 
     * @param train_ratio Ratio of training data (0.0-1.0)
     * @param val_ratio Ratio of validation data (0.0-1.0)
     * @param test_ratio Ratio of test data (0.0-1.0)
     * 
     * Note: train_ratio + val_ratio + test_ratio should equal 1.0
     */
    void split(float train_ratio = 0.8f, float val_ratio = 0.1f, float test_ratio = 0.1f) {
        if (data_.empty()) {
            std::cerr << "Warning: Cannot split empty dataset" << std::endl;
            return;
        }
        
        // Validate ratios
        float total_ratio = train_ratio + val_ratio + test_ratio;
        if (std::abs(total_ratio - 1.0f) > 1e-6) {
            std::cerr << "Warning: Ratios sum to " << total_ratio 
                     << ", normalizing to 1.0" << std::endl;
            train_ratio /= total_ratio;
            val_ratio /= total_ratio;
            test_ratio /= total_ratio;
        }
        
        // Clear existing splits
        train_indices_.clear();
        validation_indices_.clear();
        test_indices_.clear();
        
        // Create index array
        std::vector<size_t> indices(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            indices[i] = i;
        }
        
        // Shuffle indices
        std::shuffle(indices.begin(), indices.end(), rng_);
        
        // Calculate split points
        size_t train_end = static_cast<size_t>(data_.size() * train_ratio);
        size_t val_end = train_end + static_cast<size_t>(data_.size() * val_ratio);
        
        // Assign indices to splits
        train_indices_.assign(indices.begin(), indices.begin() + train_end);
        validation_indices_.assign(indices.begin() + train_end, indices.begin() + val_end);
        test_indices_.assign(indices.begin() + val_end, indices.end());
        
        is_split_ = true;
        calculate_statistics();
    }
    
    /**
     * @brief Shuffle the data samples
     * 
     * Note: This shuffles the underlying data, not the split indices.
     * Call split() again after shuffling if you need to re-split.
     */
    void shuffle() {
        std::shuffle(data_.begin(), data_.end(), rng_);
    }
    
    /**
     * @brief Shuffle indices within a split
     * @param split_type Which split to shuffle
     */
    void shuffle_split(SplitType split_type) {
        std::vector<size_t>* indices = nullptr;
        
        switch (split_type) {
            case SplitType::TRAIN:
                indices = &train_indices_;
                break;
            case SplitType::VALIDATION:
                indices = &validation_indices_;
                break;
            case SplitType::TEST:
                indices = &test_indices_;
                break;
        }
        
        if (indices) {
            std::shuffle(indices->begin(), indices->end(), rng_);
        }
    }
    
    /**
     * @brief Get samples from a specific split
     * @param split_type Which split to get
     * @return Vector of data samples
     */
    std::vector<DataSample> get_split(SplitType split_type) const {
        const std::vector<size_t>* indices = nullptr;
        
        switch (split_type) {
            case SplitType::TRAIN:
                indices = &train_indices_;
                break;
            case SplitType::VALIDATION:
                indices = &validation_indices_;
                break;
            case SplitType::TEST:
                indices = &test_indices_;
                break;
        }
        
        std::vector<DataSample> result;
        if (indices) {
            result.reserve(indices->size());
            for (size_t idx : *indices) {
                result.push_back(data_[idx]);
            }
        }
        
        return result;
    }
    
    /**
     * @brief Get all data samples
     * @return Vector of all data samples
     */
    const std::vector<DataSample>& get_all() const {
        return data_;
    }
    
    /**
     * @brief Get dataset statistics
     * @return Dataset statistics
     */
    const DatasetStats& get_stats() const {
        return stats_;
    }
    
    /**
     * @brief Get number of samples in dataset
     * @return Total number of samples
     */
    size_t size() const {
        return data_.size();
    }
    
    /**
     * @brief Get number of samples in a split
     * @param split_type Which split to count
     * @return Number of samples in split
     */
    size_t size(SplitType split_type) const {
        switch (split_type) {
            case SplitType::TRAIN:
                return train_indices_.size();
            case SplitType::VALIDATION:
                return validation_indices_.size();
            case SplitType::TEST:
                return test_indices_.size();
        }
        return 0;
    }
    
    /**
     * @brief Check if dataset is empty
     * @return True if empty
     */
    bool empty() const {
        return data_.empty();
    }
    
    /**
     * @brief Check if dataset has been split
     * @return True if split() has been called
     */
    bool is_split() const {
        return is_split_;
    }
    
    /**
     * @brief Clear all data and splits
     */
    void clear() {
        data_.clear();
        train_indices_.clear();
        validation_indices_.clear();
        test_indices_.clear();
        is_split_ = false;
        stats_ = DatasetStats();
    }
    
    /**
     * @brief Save dataset to file
     * @param filepath Output file path
     * @param format "conversation" or "tsv"
     * @return True if successful
     */
    bool save_to_file(const std::string& filepath, const std::string& format = "conversation") const {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << filepath << std::endl;
            return false;
        }
        
        if (format == "tsv") {
            for (const auto& sample : data_) {
                file << sample.input << "\t" << sample.target << "\n";
            }
        } else {  // conversation format
            for (const auto& sample : data_) {
                file << "INPUT: " << sample.input << "\n";
                file << "RESPONSE: " << sample.target << "\n\n";
            }
        }
        
        file.close();
        return true;
    }
    
    /**
     * @brief Print dataset statistics
     */
    void print_stats() const {
        std::cout << "Dataset Statistics:\n";
        std::cout << "  Total samples: " << stats_.total_samples << "\n";
        if (is_split_) {
            std::cout << "  Train samples: " << stats_.train_samples << "\n";
            std::cout << "  Validation samples: " << stats_.validation_samples << "\n";
            std::cout << "  Test samples: " << stats_.test_samples << "\n";
        }
        std::cout << "  Average input length: " << stats_.avg_input_length << " chars\n";
        std::cout << "  Average target length: " << stats_.avg_target_length << " chars\n";
        std::cout << "  Max input length: " << stats_.max_input_length << " chars\n";
        std::cout << "  Max target length: " << stats_.max_target_length << " chars\n";
    }
};

#endif  // DATASET_HPP
