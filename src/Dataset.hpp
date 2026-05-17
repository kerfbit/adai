#ifndef DATASET_HPP
#define DATASET_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "BatchProcessor.hpp"

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
    DataSample(const std::string& in, const std::string& tgt) : input(in), target(tgt) {}
};

/**
 * @brief Dataset split types
 */
enum class SplitType {
    TRAIN,       // Training data
    VALIDATION,  // Validation data
    TEST         // Test data
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
        : total_samples(0),
          train_samples(0),
          validation_samples(0),
          test_samples(0),
          avg_input_length(0),
          avg_target_length(0),
          max_input_length(0),
          max_target_length(0) {}
};

/**
 * @brief Dataset abstraction for (input, response) pairs
 *
 * Features:
 * - Load data from various file formats (conversation, TSV, JSON, CSV)
 * - Automatic train/validation/test splitting (random and stratified)
 * - K-fold cross-validation support
 * - Data shuffling and batching
 * - Dataset statistics and analysis
 * - Memory-efficient iteration with iterators
 * - Batch iteration for mini-batch training
 * - Data augmentation hooks
 * - Data filtering and preprocessing
 * - Lazy loading for large datasets
 * - Support for very large datasets
 *
 * Example usage:
 * @code
 * Dataset dataset;
 * dataset.load_from_file("conversations.txt");
 * dataset.split(0.8, 0.1, 0.1);  // 80% train, 10% val, 10% test
 * dataset.shuffle();
 *
 * // Iterator access
 * for (const auto& sample : dataset) {
 *     // Iterate over all samples
 * }
 *
 * // Batch iteration
 * for (auto batch : dataset.get_batch_iterator(SplitType::TRAIN, 32)) {
 *     // Train on batches of 32 samples
 * }
 * @endcode
 */
class Dataset {
   public:
    // Iterator support
    using iterator = std::vector<DataSample>::iterator;
    using const_iterator = std::vector<DataSample>::const_iterator;

    /**
     * @brief Batch iterator for mini-batch training
     */
    class BatchIterator {
       private:
        const Dataset* dataset_;
        const std::vector<size_t>* indices_;
        std::shared_ptr<std::vector<size_t>> owned_indices_;  // For when we own the indices
        size_t batch_size_;
        size_t current_pos_;

       public:
        BatchIterator(const Dataset* dataset, const std::vector<size_t>* indices, size_t batch_size,
                      size_t pos = 0)
            : dataset_(dataset), indices_(indices), batch_size_(batch_size), current_pos_(pos) {}

        // Constructor for owned indices
        BatchIterator(const Dataset* dataset, std::shared_ptr<std::vector<size_t>> indices,
                      size_t batch_size, size_t pos = 0)
            : dataset_(dataset),
              owned_indices_(indices),
              batch_size_(batch_size),
              current_pos_(pos) {
            indices_ = owned_indices_.get();
        }

        std::vector<DataSample> operator*() const {
            std::vector<DataSample> batch;
            size_t end_pos = std::min(current_pos_ + batch_size_, indices_->size());
            if (end_pos > current_pos_) {
                batch.reserve(end_pos - current_pos_);

                for (size_t i = current_pos_; i < end_pos; ++i) {
                    batch.push_back(dataset_->data_[(*indices_)[i]]);
                }
            }
            return batch;
        }

        BatchIterator& operator++() {
            current_pos_ += batch_size_;
            return *this;
        }

        bool operator!=(const BatchIterator& other) const {
            // If current position is at or beyond end, we're done
            if (current_pos_ >= indices_->size() && other.current_pos_ >= indices_->size()) {
                return false;
            }
            return current_pos_ != other.current_pos_;
        }
    };

   private:
    std::vector<DataSample> data_;
    std::vector<size_t> train_indices_;
    std::vector<size_t> validation_indices_;
    std::vector<size_t> test_indices_;
    std::vector<std::vector<size_t>> fold_indices_;  // For k-fold CV

    DatasetStats stats_;
    bool is_split_;
    int num_folds_;
    std::mt19937 rng_;

    // Data augmentation hooks
    std::function<DataSample(const DataSample&)> augmentation_fn_;

    // Preprocessing hooks
    std::function<std::string(const std::string&)> preprocess_fn_;

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
     * @brief Parse a JSON format
     *
     * Format: [{"input": "...", "target": "..."}, ...]
     * or line-delimited JSON: {"input": "...", "target": "..."}\n
     */
    bool parse_json_format(std::ifstream& file) {
        std::string line;
        std::string content;
        size_t pair_count = 0;

        // Read entire file or line by line
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#')
                continue;

            // Simple JSON parsing (assumes well-formed JSON)
            size_t input_pos = line.find("\"input\"");
            size_t target_pos = line.find("\"target\"");

            if (input_pos != std::string::npos && target_pos != std::string::npos) {
                // Extract input value
                size_t input_start = line.find(":", input_pos) + 1;
                size_t input_quote_start = line.find("\"", input_start) + 1;
                size_t input_quote_end = line.find("\"", input_quote_start);
                std::string input =
                    line.substr(input_quote_start, input_quote_end - input_quote_start);

                // Extract target value
                size_t target_start = line.find(":", target_pos) + 1;
                size_t target_quote_start = line.find("\"", target_start) + 1;
                size_t target_quote_end = line.find("\"", target_quote_start);
                std::string target =
                    line.substr(target_quote_start, target_quote_end - target_quote_start);

                if (!input.empty() && !target.empty()) {
                    data_.emplace_back(input, target);
                    pair_count++;
                }
            }
        }

        return pair_count > 0;
    }

    /**
     * @brief Parse a CSV format
     *
     * Format: input,target\n (with optional header)
     */
    bool parse_csv_format(std::ifstream& file, char delimiter = ',') {
        std::string line;
        size_t pair_count = 0;
        bool first_line = true;

        while (std::getline(file, line)) {
            // Skip empty lines
            if (line.empty())
                continue;

            // Skip header if it looks like one
            if (first_line) {
                first_line = false;
                if (line.find("input") != std::string::npos ||
                    line.find("Input") != std::string::npos) {
                    continue;
                }
            }

            size_t delim_pos = line.find(delimiter);
            if (delim_pos != std::string::npos) {
                std::string input = line.substr(0, delim_pos);
                std::string target = line.substr(delim_pos + 1);

                // Trim quotes if present
                auto trim_quotes = [](std::string& s) {
                    if (s.length() >= 2 && s.front() == '"' && s.back() == '"') {
                        s = s.substr(1, s.length() - 2);
                    }
                };

                trim_quotes(input);
                trim_quotes(target);

                // Trim whitespace
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
        : is_split_(false),
          num_folds_(0),
          rng_(seed),
          augmentation_fn_(nullptr),
          preprocess_fn_(nullptr) {}

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
        } else if (first_line.find('{') != std::string::npos &&
                   (first_line.find("\"input\"") != std::string::npos ||
                    first_line.find("\"Input\"") != std::string::npos)) {
            success = parse_json_format(file);
        } else if (first_line.find(',') != std::string::npos) {
            success = parse_csv_format(file, ',');
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
            std::cerr << "Warning: Ratios sum to " << total_ratio << ", normalizing to 1.0"
                      << std::endl;
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
    bool save_to_file(const std::string& filepath,
                      const std::string& format = "conversation") const {
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
        if (num_folds_ > 0) {
            std::cout << "  K-fold CV: " << num_folds_ << " folds\n";
        }
        std::cout << "  Average input length: " << stats_.avg_input_length << " chars\n";
        std::cout << "  Average target length: " << stats_.avg_target_length << " chars\n";
        std::cout << "  Max input length: " << stats_.max_input_length << " chars\n";
        std::cout << "  Max target length: " << stats_.max_target_length << " chars\n";
    }

    // ============================================================
    // Iterator Interface
    // ============================================================

    /**
     * @brief Get iterator to beginning of dataset
     */
    iterator begin() {
        return data_.begin();
    }

    /**
     * @brief Get iterator to end of dataset
     */
    iterator end() {
        return data_.end();
    }

    /**
     * @brief Get const iterator to beginning of dataset
     */
    const_iterator begin() const {
        return data_.begin();
    }

    /**
     * @brief Get const iterator to end of dataset
     */
    const_iterator end() const {
        return data_.end();
    }

    /**
     * @brief Get const iterator to beginning of dataset
     */
    const_iterator cbegin() const {
        return data_.cbegin();
    }

    /**
     * @brief Get const iterator to end of dataset
     */
    const_iterator cend() const {
        return data_.cend();
    }

    // ============================================================
    // Batch Iteration
    // ============================================================

    /**
     * @brief Get batch iterator for a split
     * @param split_type Which split to iterate over
     * @param batch_size Size of each batch
     * @return BatchIterator for iterating in batches
     */
    BatchIterator get_batch_iterator(SplitType split_type, size_t batch_size) const {
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

        return BatchIterator(this, indices, batch_size, 0);
    }

    /**
     * @brief Get batch end iterator for a split
     * @param split_type Which split to iterate over
     * @param batch_size Size of each batch
     * @return BatchIterator end position
     */
    BatchIterator get_batch_end(SplitType split_type, size_t batch_size) const {
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

        return BatchIterator(this, indices, batch_size, indices->size());
    }

    // Wrapper class for range-based for loop
    class BatchRange {
       private:
        BatchIterator begin_;
        BatchIterator end_;

       public:
        BatchRange(BatchIterator begin, BatchIterator end) : begin_(begin), end_(end) {}
        BatchIterator begin() const {
            return begin_;
        }
        BatchIterator end() const {
            return end_;
        }
    };

    /**
     * @brief Get batch range for a split (for range-based for loop)
     * @param split_type Which split to iterate over
     * @param batch_size Size of each batch
     * @return BatchRange for iterating in batches
     */
    BatchRange get_batch_range(SplitType split_type, size_t batch_size) const {
        return BatchRange(get_batch_iterator(split_type, batch_size),
                          get_batch_end(split_type, batch_size));
    }

    /**
     * @brief Get batch iterator for entire dataset
     * @param batch_size Size of each batch
     * @return BatchRange for iterating in batches
     */
    BatchRange get_batch_range(size_t batch_size) const {
        auto all_indices = std::make_shared<std::vector<size_t>>(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            (*all_indices)[i] = i;
        }
        BatchIterator begin_it(this, all_indices, batch_size, 0);
        BatchIterator end_it(this, all_indices, batch_size, data_.size());
        return BatchRange(begin_it, end_it);
    }

    // ============================================================
    // Stratified Splitting
    // ============================================================

    /**
     * @brief Split dataset with stratification by sample length
     *
     * Ensures balanced distribution of sample lengths across splits
     *
     * @param train_ratio Ratio of training data
     * @param val_ratio Ratio of validation data
     * @param test_ratio Ratio of test data
     * @param num_bins Number of length bins for stratification
     */
    void split_stratified(float train_ratio = 0.8f, float val_ratio = 0.1f, float test_ratio = 0.1f,
                          int num_bins = 5) {
        if (data_.empty()) {
            std::cerr << "Warning: Cannot split empty dataset" << std::endl;
            return;
        }

        // Validate ratios
        float total_ratio = train_ratio + val_ratio + test_ratio;
        if (std::abs(total_ratio - 1.0f) > 1e-6) {
            train_ratio /= total_ratio;
            val_ratio /= total_ratio;
            test_ratio /= total_ratio;
        }

        // Clear existing splits
        train_indices_.clear();
        validation_indices_.clear();
        test_indices_.clear();

        // Compute length bins
        std::vector<std::vector<size_t>> bins(num_bins);
        size_t max_len = 0;
        for (const auto& sample : data_) {
            max_len = std::max(max_len, sample.input.length() + sample.target.length());
        }

        // Assign samples to bins
        for (size_t i = 0; i < data_.size(); ++i) {
            size_t total_len = data_[i].input.length() + data_[i].target.length();
            int bin =
                std::min(static_cast<int>(total_len * num_bins / (max_len + 1)), num_bins - 1);
            bins[bin].push_back(i);
        }

        // Split each bin proportionally
        for (auto& bin : bins) {
            if (bin.empty())
                continue;

            std::shuffle(bin.begin(), bin.end(), rng_);

            size_t train_end = static_cast<size_t>(bin.size() * train_ratio);
            size_t val_end = train_end + static_cast<size_t>(bin.size() * val_ratio);

            train_indices_.insert(train_indices_.end(), bin.begin(), bin.begin() + train_end);
            validation_indices_.insert(validation_indices_.end(), bin.begin() + train_end,
                                       bin.begin() + val_end);
            test_indices_.insert(test_indices_.end(), bin.begin() + val_end, bin.end());
        }

        is_split_ = true;
        calculate_statistics();
    }

    // ============================================================
    // K-Fold Cross-Validation
    // ============================================================

    /**
     * @brief Setup k-fold cross-validation
     * @param k Number of folds
     */
    void setup_k_fold(int k) {
        if (data_.empty() || k < 2) {
            std::cerr << "Warning: Invalid k-fold setup" << std::endl;
            return;
        }

        num_folds_ = k;
        fold_indices_.clear();
        fold_indices_.resize(k);

        // Create shuffled indices
        std::vector<size_t> indices(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), rng_);

        // Distribute into folds
        for (size_t i = 0; i < indices.size(); ++i) {
            fold_indices_[i % k].push_back(indices[i]);
        }
    }

    /**
     * @brief Get training and validation sets for a specific fold
     * @param fold_idx Fold index (0 to k-1)
     * @param train_data Output: training samples
     * @param val_data Output: validation samples
     */
    void get_fold(int fold_idx, std::vector<DataSample>& train_data,
                  std::vector<DataSample>& val_data) const {
        if (fold_idx < 0 || fold_idx >= num_folds_) {
            std::cerr << "Error: Invalid fold index" << std::endl;
            return;
        }

        train_data.clear();
        val_data.clear();

        // Validation fold
        val_data.reserve(fold_indices_[fold_idx].size());
        for (size_t idx : fold_indices_[fold_idx]) {
            val_data.push_back(data_[idx]);
        }

        // Training folds (all others)
        size_t train_size = 0;
        for (int i = 0; i < num_folds_; ++i) {
            if (i != fold_idx)
                train_size += fold_indices_[i].size();
        }
        train_data.reserve(train_size);

        for (int i = 0; i < num_folds_; ++i) {
            if (i != fold_idx) {
                for (size_t idx : fold_indices_[i]) {
                    train_data.push_back(data_[idx]);
                }
            }
        }
    }

    /**
     * @brief Get number of folds
     */
    int get_num_folds() const {
        return num_folds_;
    }

    // ============================================================
    // Data Augmentation
    // ============================================================

    /**
     * @brief Set data augmentation function
     * @param fn Augmentation function that transforms a sample
     */
    void set_augmentation(std::function<DataSample(const DataSample&)> fn) {
        augmentation_fn_ = fn;
    }

    /**
     * @brief Apply augmentation to dataset (creates new samples)
     * @param num_augmented Number of augmented samples per original sample
     */
    void augment_data(int num_augmented = 1) {
        if (!augmentation_fn_) {
            std::cerr << "Warning: No augmentation function set" << std::endl;
            return;
        }

        size_t original_size = data_.size();
        data_.reserve(original_size * (1 + num_augmented));

        for (size_t i = 0; i < original_size; ++i) {
            for (int j = 0; j < num_augmented; ++j) {
                data_.push_back(augmentation_fn_(data_[i]));
            }
        }

        calculate_statistics();
    }

    // ============================================================
    // Preprocessing and Filtering
    // ============================================================

    /**
     * @brief Set preprocessing function for text
     * @param fn Preprocessing function (e.g., lowercasing, normalization)
     */
    void set_preprocessing(std::function<std::string(const std::string&)> fn) {
        preprocess_fn_ = fn;
    }

    /**
     * @brief Apply preprocessing to all samples
     */
    void apply_preprocessing() {
        if (!preprocess_fn_) {
            std::cerr << "Warning: No preprocessing function set" << std::endl;
            return;
        }

        for (auto& sample : data_) {
            sample.input = preprocess_fn_(sample.input);
            sample.target = preprocess_fn_(sample.target);
        }

        calculate_statistics();
    }

    /**
     * @brief Filter samples by length
     * @param min_length Minimum total length (input + target)
     * @param max_length Maximum total length (input + target)
     */
    void filter_by_length(size_t min_length, size_t max_length) {
        auto new_end = std::remove_if(
            data_.begin(), data_.end(), [min_length, max_length](const DataSample& sample) {
                size_t total_len = sample.input.length() + sample.target.length();
                return total_len < min_length || total_len > max_length;
            });
        data_.erase(new_end, data_.end());

        // Clear splits as indices are now invalid
        train_indices_.clear();
        validation_indices_.clear();
        test_indices_.clear();
        is_split_ = false;

        calculate_statistics();
    }

    /**
     * @brief Filter samples by pattern
     * @param pattern Pattern to search for
     * @param keep_matching If true, keep matching samples; if false, remove them
     */
    void filter_by_pattern(const std::string& pattern, bool keep_matching = true) {
        auto new_end = std::remove_if(
            data_.begin(), data_.end(), [&pattern, keep_matching](const DataSample& sample) {
                bool matches = (sample.input.find(pattern) != std::string::npos ||
                                sample.target.find(pattern) != std::string::npos);
                return keep_matching ? !matches : matches;
            });
        data_.erase(new_end, data_.end());

        // Clear splits as indices are now invalid
        train_indices_.clear();
        validation_indices_.clear();
        test_indices_.clear();
        is_split_ = false;

        calculate_statistics();
    }

    /**
     * @brief Lowercase all text in dataset
     */
    void lowercase() {
        for (auto& sample : data_) {
            std::transform(sample.input.begin(), sample.input.end(), sample.input.begin(),
                           ::tolower);
            std::transform(sample.target.begin(), sample.target.end(), sample.target.begin(),
                           ::tolower);
        }
    }

    // ============================================================
    // Batch Processing Integration
    // ============================================================

    /**
     * @brief Get batch of tokenized sequences with padding
     *
     * Retrieves samples from a split, tokenizes them, and creates a padded batch
     * ready for model processing.
     *
     * @param split_type Which split to get batch from
     * @param batch_start Start index within the split
     * @param batch_size Number of samples in batch
     * @param tokenizer_fn Function to tokenize a string into token IDs
     * @param pad_token_id Token ID to use for padding (default: PAD token)
     * @return TokenBatch with padded input sequences
     *
     * @code
     * auto tokenizer_fn = [&tokenizer](const std::string& text) {
     *     return tokenizer.encode(text);
     * };
     * TokenBatch batch = dataset.get_batch_with_padding(
     *     SplitType::TRAIN, 0, 32, tokenizer_fn, 0);
     * @endcode
     */
    TokenBatch get_batch_with_padding(
        SplitType split_type, size_t batch_start, size_t batch_size,
        std::function<std::vector<int>(const std::string&)> tokenizer_fn,
        int pad_token_id = adai::SpecialTokenIDs::PAD) const {
        // Get indices for split
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

        if (!indices || batch_start >= indices->size()) {
            return TokenBatch();  // Empty batch
        }

        // Calculate actual batch size
        size_t actual_batch_size = std::min(batch_size, indices->size() - batch_start);

        // Tokenize all inputs in batch
        std::vector<std::vector<int>> sequences;
        sequences.reserve(actual_batch_size);

        for (size_t i = batch_start; i < batch_start + actual_batch_size; ++i) {
            size_t data_idx = (*indices)[i];
            sequences.push_back(tokenizer_fn(data_[data_idx].input));
        }

        // Create padded batch
        return create_batch(sequences, pad_token_id);
    }

    /**
     * @brief Get batch of tokenized target sequences with padding
     *
     * Similar to get_batch_with_padding but for target sequences.
     *
     * @param split_type Which split to get batch from
     * @param batch_start Start index within the split
     * @param batch_size Number of samples in batch
     * @param tokenizer_fn Function to tokenize a string into token IDs
     * @param pad_token_id Token ID to use for padding (default: PAD token)
     * @return TokenBatch with padded target sequences
     */
    TokenBatch get_target_batch_with_padding(
        SplitType split_type, size_t batch_start, size_t batch_size,
        std::function<std::vector<int>(const std::string&)> tokenizer_fn,
        int pad_token_id = adai::SpecialTokenIDs::PAD) const {
        // Get indices for split
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

        if (!indices || batch_start >= indices->size()) {
            return TokenBatch();  // Empty batch
        }

        // Calculate actual batch size
        size_t actual_batch_size = std::min(batch_size, indices->size() - batch_start);

        // Tokenize all targets in batch
        std::vector<std::vector<int>> sequences;
        sequences.reserve(actual_batch_size);

        for (size_t i = batch_start; i < batch_start + actual_batch_size; ++i) {
            size_t data_idx = (*indices)[i];
            sequences.push_back(tokenizer_fn(data_[data_idx].target));
        }

        // Create padded batch
        return create_batch(sequences, pad_token_id);
    }

    /**
     * @brief Get dynamic batches optimized for sequence length
     *
     * Creates multiple batches where sequences of similar length are grouped
     * together to minimize padding waste.
     *
     * @param split_type Which split to get batches from
     * @param tokenizer_fn Function to tokenize a string into token IDs
     * @param max_batch_size Maximum number of sequences per batch (default: 32)
     * @param length_tolerance Maximum length difference within a batch (default: 10)
     * @param pad_token_id Token ID to use for padding (default: PAD token)
     * @return Vector of TokenBatch objects
     *
     * @code
     * auto tokenizer_fn = [&tokenizer](const std::string& text) {
     *     return tokenizer.encode(text);
     * };
     * auto batches = dataset.get_dynamic_batches(
     *     SplitType::TRAIN, tokenizer_fn, 32, 10, 0);
     * for (const auto& batch : batches) {
     *     // Process each batch
     *     std::cout << "Batch size: " << batch.batch_size()
     *               << ", max_length: " << batch.max_length << std::endl;
     * }
     * @endcode
     */
    std::vector<TokenBatch> get_dynamic_batches(
        SplitType split_type, std::function<std::vector<int>(const std::string&)> tokenizer_fn,
        int max_batch_size = 32, int length_tolerance = 10,
        int pad_token_id = adai::SpecialTokenIDs::PAD) const {
        // Get indices for split
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

        if (!indices || indices->empty()) {
            return std::vector<TokenBatch>();  // Empty
        }

        // Tokenize all sequences
        std::vector<std::vector<int>> all_sequences;
        all_sequences.reserve(indices->size());

        for (size_t idx : *indices) {
            all_sequences.push_back(tokenizer_fn(data_[idx].input));
        }

        // Create dynamic batches using BatchProcessor
        return create_dynamic_batches(all_sequences, max_batch_size, length_tolerance,
                                      pad_token_id);
    }

    /**
     * @brief Process a batch through a model and collect outputs
     *
     * Convenience method that handles batch padding, processing, and output collection.
     *
     * @param samples Batch of data samples to process
     * @param tokenizer_fn Function to tokenize a string into token IDs
     * @param model_fn Function that processes a batch of token sequences and returns outputs
     * @param pad_token_id Token ID to use for padding (default: PAD token)
     * @return Vector of model outputs (one per sample)
     *
     * @code
     * auto batch = dataset.get_split(SplitType::TRAIN).slice(0, 32);
     * auto outputs = dataset.process_batch(
     *     batch,
     *     [&tokenizer](const std::string& text) { return tokenizer.encode(text); },
     *     [&model](const std::vector<std::vector<int>>& seqs) {
     *         // Process batch through model
     *         return model.forward_batch(seqs);
     *     },
     *     0
     * );
     * @endcode
     */
    template <typename OutputType>
    std::vector<OutputType> process_batch(
        const std::vector<DataSample>& samples,
        std::function<std::vector<int>(const std::string&)> tokenizer_fn,
        std::function<std::vector<OutputType>(const TokenBatch&)> model_fn,
        int pad_token_id = adai::SpecialTokenIDs::PAD) const {
        if (samples.empty()) {
            return std::vector<OutputType>();
        }

        // Tokenize all inputs
        std::vector<std::vector<int>> sequences;
        sequences.reserve(samples.size());

        for (const auto& sample : samples) {
            sequences.push_back(tokenizer_fn(sample.input));
        }

        // Create padded batch
        TokenBatch batch = create_batch(sequences, pad_token_id);

        // Process through model
        return model_fn(batch);
    }

    /**
     * @brief Compute batch statistics for a split
     *
     * Analyzes the padding efficiency for batches in a split.
     *
     * @param split_type Which split to analyze
     * @param tokenizer_fn Function to tokenize a string into token IDs
     * @param batch_size Size of batches to analyze
     * @return BatchStats with efficiency metrics
     *
     * @code
     * auto stats = dataset.get_batch_statistics(
     *     SplitType::TRAIN,
     *     [&tokenizer](const std::string& text) { return tokenizer.encode(text); },
     *     32
     * );
     * std::cout << "Padding efficiency: "
     *           << (stats.efficiency_percentage * 100) << "%" << std::endl;
     * @endcode
     */
    BatchStats get_batch_statistics(
        SplitType split_type, std::function<std::vector<int>(const std::string&)> tokenizer_fn,
        size_t batch_size) const {
        // Get indices for split
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

        if (!indices || indices->empty()) {
            return BatchStats();  // Empty stats
        }

        // Sample a batch
        size_t actual_batch_size = std::min(batch_size, indices->size());
        std::vector<std::vector<int>> sequences;
        sequences.reserve(actual_batch_size);

        for (size_t i = 0; i < actual_batch_size; ++i) {
            size_t data_idx = (*indices)[i];
            sequences.push_back(tokenizer_fn(data_[i].input));
        }

        // Create batch and compute stats
        TokenBatch batch = create_batch(sequences, 0);
        std::vector<TokenBatch> batch_vec = {batch};
        return compute_batch_stats(batch_vec);
    }
};

/**
 * @brief Lazy-loading dataset for very large files
 *
 * Reads samples on-demand instead of loading entire dataset into memory.
 * Useful for datasets that don't fit in RAM.
 */
class LazyDataset {
   private:
    std::string filepath_;
    std::vector<std::streampos> sample_positions_;  // File positions of samples
    size_t num_samples_;
    DatasetStats stats_;

    /**
     * @brief Index the file to find sample positions
     */
    void index_file() {
        std::ifstream file(filepath_);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file: " << filepath_ << std::endl;
            return;
        }

        sample_positions_.clear();
        std::string line;

        while (file) {
            std::streampos pos = file.tellg();
            if (!std::getline(file, line))
                break;

            // Check if this is the start of a sample
            if (line.find("INPUT:") != std::string::npos || line.find('{') != std::string::npos) {
                sample_positions_.push_back(pos);
            }
        }

        num_samples_ = sample_positions_.size();
        file.close();
    }

   public:
    /**
     * @brief Constructor
     * @param filepath Path to dataset file
     */
    LazyDataset(const std::string& filepath) : filepath_(filepath), num_samples_(0) {
        index_file();
    }

    /**
     * @brief Get a sample by index (loads from disk)
     * @param index Sample index
     * @return Data sample
     */
    DataSample get_sample(size_t index) const {
        if (index >= num_samples_) {
            return DataSample();
        }

        std::ifstream file(filepath_);
        file.seekg(sample_positions_[index]);

        std::string line;
        std::string input, target;

        // Read sample (simplified - assumes conversation format)
        while (std::getline(file, line)) {
            if (line.find("INPUT:") != std::string::npos) {
                input = line.substr(7);
            } else if (line.find("RESPONSE:") != std::string::npos) {
                target = line.substr(10);
                break;
            }
        }

        file.close();
        return DataSample(input, target);
    }

    /**
     * @brief Get number of samples
     */
    size_t size() const {
        return num_samples_;
    }

    /**
     * @brief Load a range of samples into memory
     * @param start Start index
     * @param count Number of samples to load
     * @return Vector of data samples
     */
    std::vector<DataSample> load_range(size_t start, size_t count) const {
        std::vector<DataSample> samples;
        samples.reserve(count);

        for (size_t i = start; i < std::min(start + count, num_samples_); ++i) {
            samples.push_back(get_sample(i));
        }

        return samples;
    }
};

#endif  // DATASET_HPP
