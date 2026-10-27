// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-27

#include "SecureRandom.hpp"

#include <cerrno>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <bcrypt.h>
#elif defined(__linux__)
#include <sys/random.h>
#else
#include <stdlib.h>  // arc4random_buf (macOS, BSD)
#endif

namespace adai {

void secure_random_bytes(unsigned char* out, std::size_t size) {
#if defined(_WIN32)
    while (size > 0) {
        const ULONG chunk = size > 0x7fffffffu ? 0x7fffffffu : static_cast<ULONG>(size);
        if (BCryptGenRandom(nullptr, out, chunk, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
            throw std::runtime_error("secure_random_bytes: BCryptGenRandom failed");
        }
        out += chunk;
        size -= chunk;
    }
#elif defined(__linux__)
    while (size > 0) {
        const ssize_t got = ::getrandom(out, size, 0);
        if (got < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("secure_random_bytes: getrandom failed (errno " +
                                     std::to_string(errno) + ")");
        }
        out += got;
        size -= static_cast<std::size_t>(got);
    }
#else
    ::arc4random_buf(out, size);
#endif
}

std::string secure_random_hex(std::size_t bytes) {
    static const char hex[] = "0123456789abcdef";
    std::vector<unsigned char> buf(bytes);
    if (bytes > 0) secure_random_bytes(buf.data(), bytes);
    std::string out;
    out.reserve(bytes * 2);
    for (unsigned char b : buf) {
        out += hex[b >> 4];
        out += hex[b & 0xf];
    }
    return out;
}

}  // namespace adai
