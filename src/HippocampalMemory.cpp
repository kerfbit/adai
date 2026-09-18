// @adai-status: experimental
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-18

#include "HippocampalMemory.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
// Same validate-and-pass-through helper TD-175's SIGReg introduced, called directly inside the
// member-initializer list so a non-positive value is rejected with a clear std::invalid_argument
// rather than silently producing a zero-or-negative-capacity/dimension instance.
int require_positive(int value, const char* name) {
    if (value <= 0) {
        throw std::invalid_argument(std::string("HippocampalMemory: ") + name +
                                     " must be positive");
    }
    return value;
}
}  // namespace

HippocampalMemory::HippocampalMemory(int d_model, int capacity, std::string swap_filepath)
    : capacity(require_positive(capacity, "capacity")),
      d_model(require_positive(d_model, "d_model")),
      swap_filepath_(std::move(swap_filepath)) {}

void HippocampalMemory::evict_least_used(std::deque<Slot>& slots,
                                          std::deque<float>& coverage) const {
    size_t evict_idx = 0;
    float min_cov = coverage[0];
    for (size_t i = 1; i < coverage.size(); ++i) {
        if (coverage[i] < min_cov) {
            min_cov = coverage[i];
            evict_idx = i;
        }
    }

    if (!swap_filepath_.empty()) {
        append_to_swap(slots[evict_idx].key, slots[evict_idx].value, coverage[evict_idx]);
    }

    slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(evict_idx));
    coverage.erase(coverage.begin() + static_cast<std::ptrdiff_t>(evict_idx));
}

void HippocampalMemory::insert_slot(const Matrix& key, const Matrix& value, float coverage) {
    if (static_cast<int>(slots_.size()) >= capacity) {
        evict_least_used(slots_, coverage_);
    }

    slots_.push_back(Slot{key, value});
    coverage_.push_back(coverage);
}

void HippocampalMemory::write(const Matrix& key, const Matrix& value) {
    if (key.rows != 1 || key.cols != d_model) {
        throw std::invalid_argument("HippocampalMemory::write: key must be [1, d_model]");
    }
    if (value.rows != 1 || value.cols != d_model) {
        throw std::invalid_argument("HippocampalMemory::write: value must be [1, d_model]");
    }

    insert_slot(key, value, 0.0f);
}

std::pair<Matrix, Matrix> HippocampalMemory::read_all() const {
    const int n = static_cast<int>(slots_.size());
    Matrix keys(n, d_model);
    Matrix values(n, d_model);

    int i = 0;
    for (const auto& slot : slots_) {
        for (int j = 0; j < d_model; ++j) {
            keys(i, j) = slot.key(0, j);
            values(i, j) = slot.value(0, j);
        }
        ++i;
    }

    return {keys, values};
}

std::deque<float>& HippocampalMemory::coverage_vector() {
    return coverage_;
}

void HippocampalMemory::decay_coverage(float gamma) {
    for (float& c : coverage_) {
        c *= gamma;
    }
}

void HippocampalMemory::clear() {
    slots_.clear();
    coverage_.clear();
}

void HippocampalMemory::append_to_swap(const Matrix& key, const Matrix& value,
                                        float coverage) const {
    std::error_code ec;
    const bool need_header =
        !std::filesystem::exists(swap_filepath_, ec) || std::filesystem::is_empty(swap_filepath_, ec);

    std::ofstream file(swap_filepath_, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open swap file for writing: " + swap_filepath_);
    }

    if (need_header) {
        file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    }
    for (int j = 0; j < d_model; ++j) {
        file.write(reinterpret_cast<const char*>(&key(0, j)), sizeof(float));
    }
    for (int j = 0; j < d_model; ++j) {
        file.write(reinterpret_cast<const char*>(&value(0, j)), sizeof(float));
    }
    file.write(reinterpret_cast<const char*>(&coverage), sizeof(float));
}

bool HippocampalMemory::recall_from_swap() {
    if (swap_filepath_.empty() || !std::filesystem::exists(swap_filepath_)) {
        return false;
    }

    std::error_code ec;
    const auto file_size = static_cast<long long>(std::filesystem::file_size(swap_filepath_, ec));
    if (ec) {
        return false;
    }

    constexpr long long kHeaderBytes = sizeof(int);
    if (file_size < kHeaderBytes) {
        return false;  // no header yet — nothing has ever been swapped out
    }

    std::fstream file(swap_filepath_, std::ios::binary | std::ios::in | std::ios::out);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open swap file for reading: " + swap_filepath_);
    }

    // Validate the header BEFORE using this instance's own d_model to size a "does the file hold
    // a full record" check below — otherwise a mismatched reader (e.g. a larger d_model than the
    // file was actually written with) could make that size check fail first and silently return
    // false instead of surfacing the real mismatch.
    int swap_d_model = 0;
    file.read(reinterpret_cast<char*>(&swap_d_model), sizeof(int));
    if (swap_d_model != d_model) {
        throw std::runtime_error("HippocampalMemory: swap file dimension mismatch: saved d_model=" +
                                  std::to_string(swap_d_model) +
                                  " vs current d_model=" + std::to_string(d_model));
    }

    const long long record_bytes = (2LL * d_model + 1) * static_cast<long long>(sizeof(float));
    const long long num_records = (file_size - kHeaderBytes) / record_bytes;
    if (num_records <= 0) {
        return false;
    }

    const long long last_record_offset = kHeaderBytes + (num_records - 1) * record_bytes;
    file.seekg(last_record_offset);

    Matrix key(1, d_model);
    Matrix value(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&key(0, j)), sizeof(float));
    }
    for (int j = 0; j < d_model; ++j) {
        file.read(reinterpret_cast<char*>(&value(0, j)), sizeof(float));
    }
    // The evicted coverage value itself isn't reused — a recalled slot is treated as freshly
    // relevant again, the same starting point any newly write()'d slot gets (see class doc).
    file.close();

    // Drop the just-read record from the swap file by truncating it away — cheap (no rewrite of
    // the remaining records needed) since it was always the LAST record in the file.
    std::filesystem::resize_file(swap_filepath_, static_cast<std::uintmax_t>(last_record_offset));

    insert_slot(key, value, 0.0f);
    return true;
}

void HippocampalMemory::save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    const int num_slots = static_cast<int>(slots_.size());
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<const char*>(&capacity), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_slots), sizeof(int));

    int idx = 0;
    for (const auto& slot : slots_) {
        for (int j = 0; j < d_model; ++j) {
            file.write(reinterpret_cast<const char*>(&slot.key(0, j)), sizeof(float));
        }
        for (int j = 0; j < d_model; ++j) {
            file.write(reinterpret_cast<const char*>(&slot.value(0, j)), sizeof(float));
        }
        file.write(reinterpret_cast<const char*>(&coverage_[idx]), sizeof(float));
        ++idx;
    }
}

void HippocampalMemory::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    int loaded_d_model = 0;
    int loaded_capacity = 0;  // read for format completeness; see this method's own doc comment
                              // on why a capacity mismatch is accepted rather than rejected.
    int num_slots = 0;

    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_capacity), sizeof(int));
    file.read(reinterpret_cast<char*>(&num_slots), sizeof(int));
    (void)loaded_capacity;

    if (loaded_d_model != d_model) {
        throw std::runtime_error("HippocampalMemory dimension mismatch: saved d_model=" +
                                  std::to_string(loaded_d_model) +
                                  " vs current d_model=" + std::to_string(d_model));
    }

    // Validate num_slots against the file's own actual remaining size before trusting it for
    // anything — a corrupted or wrong-format file could otherwise hand a huge or negative value
    // to the read loop below, throwing an unexpected std::length_error/std::bad_alloc (or, worse,
    // silently reading past EOF into zero-initialized garbage) instead of the clear
    // std::runtime_error this method otherwise uses consistently for malformed data (see the
    // d_model check above).
    const std::streampos data_start = file.tellg();
    file.seekg(0, std::ios::end);
    const std::streampos file_end = file.tellg();
    file.seekg(data_start);

    const long long remaining_bytes =
        (data_start >= 0 && file_end >= 0)
            ? static_cast<long long>(file_end) - static_cast<long long>(data_start)
            : -1;
    const long long bytes_per_slot = (2LL * d_model + 1) * static_cast<long long>(sizeof(float));
    const long long expected_bytes = static_cast<long long>(num_slots) * bytes_per_slot;

    if (num_slots < 0 || remaining_bytes < 0 || expected_bytes > remaining_bytes) {
        throw std::runtime_error("HippocampalMemory: corrupt or truncated file '" + filepath +
                                  "' — header claims " + std::to_string(num_slots) +
                                  " slot(s), which needs " + std::to_string(expected_bytes) +
                                  " byte(s), but only " + std::to_string(remaining_bytes) +
                                  " remain");
    }

    std::deque<Slot> loaded_slots;
    std::deque<float> loaded_coverage;

    for (int i = 0; i < num_slots; ++i) {
        Matrix key(1, d_model);
        Matrix value(1, d_model);
        for (int j = 0; j < d_model; ++j) {
            file.read(reinterpret_cast<char*>(&key(0, j)), sizeof(float));
        }
        for (int j = 0; j < d_model; ++j) {
            file.read(reinterpret_cast<char*>(&value(0, j)), sizeof(float));
        }
        float coverage_val = 0.0f;
        file.read(reinterpret_cast<char*>(&coverage_val), sizeof(float));

        loaded_slots.push_back(Slot{key, value});
        loaded_coverage.push_back(coverage_val);
    }

    // Saved capacity is a runtime tuning knob, not an architectural constant (unlike d_model) —
    // accept a mismatch by respecting THIS instance's own current capacity, evicting the
    // least-used excess slots (to swap, if configured) down to it rather than throwing — same
    // policy write()'s own eviction uses (TD-193).
    while (static_cast<int>(loaded_slots.size()) > capacity) {
        evict_least_used(loaded_slots, loaded_coverage);
    }

    slots_ = std::move(loaded_slots);
    coverage_ = std::move(loaded_coverage);
}
