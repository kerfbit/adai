#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-27

/**
 * SecureRandom — bytes from the operating system's CSPRNG (TD-233).
 *
 * For anything that works as a credential: ChatbotAPI session ids (the only thing protecting a
 * conversation's export) and FtpDataServer's per-session FTP user/password. Those used to come
 * from std::mt19937, whose output can be predicted after enough of it has been seen.
 *
 * Linux: getrandom(); macOS/BSD: arc4random_buf(); Windows: BCryptGenRandom(). There is no weak
 * fallback: if the OS source fails, these throw std::runtime_error.
 */

#include <cstddef>
#include <string>

namespace adai {

/// Fills `out[0, size)` from the OS CSPRNG. Throws std::runtime_error on failure.
void secure_random_bytes(unsigned char* out, std::size_t size);

/// `bytes` random bytes as 2 * `bytes` lowercase hex digits. Throws std::runtime_error on failure.
std::string secure_random_hex(std::size_t bytes);

}  // namespace adai
