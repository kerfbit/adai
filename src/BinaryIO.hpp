// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-26
#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <type_traits>
#include "Matrix.hpp"

/**
 * @file BinaryIO.hpp
 * @brief Checked reads and writes for the binary checkpoint sidecars
 *
 * TD-423: every checkpoint loader used to read without checking the stream, so a file cut short by a
 * full disk or a crash during save loaded silently: NaN or stale weights, and a server that
 * generates garbage instead of refusing to start. Writes weren't checked either. Reader throws
 * std::runtime_error on a short read (and, through expect_end(), on trailing bytes); Writer throws when
 * a write or the final flush fails. Loaders read a whole file into temporaries before changing the
 * object, so a failed load leaves that object as it was.
 */
namespace adai::binio {

class Reader {
   public:
    /** @throws std::runtime_error if the file can't be opened */
    explicit Reader(const std::string& path);

    /** Reads one trivially copyable value. @throws std::runtime_error on a short read */
    template <typename T>
    T value(const char* what) {
        static_assert(std::is_trivially_copyable_v<T>, "binio::Reader::value needs a POD type");
        T out{};
        bytes(&out, sizeof(T), what);
        return out;
    }

    /** Reads a rows × cols float matrix, row by row. @throws std::runtime_error on a short read */
    Matrix matrix(int rows, int cols, const char* what);

    /** @throws std::runtime_error if the file has bytes left */
    void expect_end();

    const std::string& path() const {
        return path_;
    }

   private:
    void bytes(void* dst, std::size_t count, const char* what);

    std::string path_;
    std::ifstream file_;
};

class Writer {
   public:
    /** @throws std::runtime_error if the file can't be opened */
    explicit Writer(const std::string& path);

    template <typename T>
    void value(const T& v, const char* what) {
        static_assert(std::is_trivially_copyable_v<T>, "binio::Writer::value needs a POD type");
        bytes(&v, sizeof(T), what);
    }

    /** Writes a matrix row by row (the layout Reader::matrix() reads). */
    void matrix(const Matrix& m, const char* what);

    /** Flushes and closes. @throws std::runtime_error if any write or the flush failed */
    void close();

   private:
    void bytes(const void* src, std::size_t count, const char* what);

    std::string path_;
    std::ofstream file_;
};

}  // namespace adai::binio
