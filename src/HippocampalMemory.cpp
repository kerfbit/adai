// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-17

#include "HippocampalMemory.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

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

HippocampalMemory::HippocampalMemory(int d_model, int capacity)
    : capacity(require_positive(capacity, "capacity")),
      d_model(require_positive(d_model, "d_model")) {}

void HippocampalMemory::write(const Matrix& key, const Matrix& value) {
    if (key.rows != 1 || key.cols != d_model) {
        throw std::invalid_argument("HippocampalMemory::write: key must be [1, d_model]");
    }
    if (value.rows != 1 || value.cols != d_model) {
        throw std::invalid_argument("HippocampalMemory::write: value must be [1, d_model]");
    }

    if (static_cast<int>(slots_.size()) >= capacity) {
        slots_.pop_front();
        coverage_.erase(coverage_.begin());
    }

    slots_.push_back(Slot{key, value});
    coverage_.push_back(0.0f);
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

std::vector<float>& HippocampalMemory::coverage_vector() {
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
    // loaded_coverage.reserve() below — a corrupted or wrong-format file could otherwise hand
    // reserve() a huge or negative value (negative converts to an enormous size_t), throwing an
    // unexpected std::length_error/std::bad_alloc instead of the clear std::runtime_error this
    // method otherwise uses consistently for malformed data (see the d_model check above).
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
    std::vector<float> loaded_coverage;
    loaded_coverage.reserve(num_slots);

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
    // accept a mismatch by respecting THIS instance's own current capacity, evicting the oldest
    // loaded slots down to it rather than throwing.
    while (static_cast<int>(loaded_slots.size()) > capacity) {
        loaded_slots.pop_front();
        loaded_coverage.erase(loaded_coverage.begin());
    }

    slots_ = std::move(loaded_slots);
    coverage_ = std::move(loaded_coverage);
}
