// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-26
#include "BinaryIO.hpp"

#include <stdexcept>

namespace adai::binio {

Reader::Reader(const std::string& path) : path_(path), file_(path, std::ios::binary) {
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
}

void Reader::bytes(void* dst, std::size_t count, const char* what) {
    file_.read(static_cast<char*>(dst), static_cast<std::streamsize>(count));
    if (static_cast<std::size_t>(file_.gcount()) != count) {
        throw std::runtime_error("Truncated or unreadable file " + path_ + " (reading " + what +
                                 ")");
    }
}

Matrix Reader::matrix(int rows, int cols, const char* what) {
    if (rows < 0 || cols < 0) {
        throw std::runtime_error("Invalid matrix shape in " + path_ + " (" + what + ")");
    }
    Matrix m(rows, cols);
    for (int r = 0; r < rows; ++r) {
        bytes(m.data[r].data(), sizeof(float) * static_cast<std::size_t>(cols), what);
    }
    return m;
}

void Reader::expect_end() {
    if (file_.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Unexpected trailing data in " + path_ +
                                 " (the file doesn't match this model's shapes)");
    }
}

Writer::Writer(const std::string& path) : path_(path), file_(path, std::ios::binary) {
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
}

void Writer::bytes(const void* src, std::size_t count, const char* what) {
    file_.write(static_cast<const char*>(src), static_cast<std::streamsize>(count));
    if (!file_) {
        throw std::runtime_error("Failed to write " + path_ + " (writing " + what + ")");
    }
}

void Writer::matrix(const Matrix& m, const char* what) {
    for (int r = 0; r < m.rows; ++r) {
        bytes(m.data[r].data(), sizeof(float) * static_cast<std::size_t>(m.cols), what);
    }
}

void Writer::close() {
    file_.flush();
    const bool ok = static_cast<bool>(file_);
    file_.close();
    if (!ok || file_.fail()) {
        throw std::runtime_error("Failed to write " + path_ + " (flush/close)");
    }
}

}  // namespace adai::binio
