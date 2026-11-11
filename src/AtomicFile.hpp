#pragma once

// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10

#include <string>

namespace adai {

/**
 * @brief Replace `path`'s contents with `data` so that a reader sees either the old file or the
 *        whole new one, never part of it (TD-495).
 *
 * Writes a temporary file beside `path`, flushes and fsyncs it, renames it over `path`, then
 * fsyncs the directory (POSIX), so a failed write (disk full, file-size limit) or a crash leaves
 * the old file as it was. Creates the parent directory if needed.
 *
 * @return an empty string on success, otherwise what failed (the old file is untouched and the
 *         temporary file removed).
 */
std::string write_file_atomically(const std::string& path, const std::string& data);

}  // namespace adai
