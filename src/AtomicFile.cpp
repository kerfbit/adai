// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10

#include "AtomicFile.hpp"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>
#include "SecureRandom.hpp"

#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace adai {

namespace fs = std::filesystem;

namespace {
#ifndef _WIN32
// fsync the directory holding `path`, so the rename itself survives a crash. Best effort: some
// filesystems refuse to fsync a directory.
void sync_directory(const fs::path& dir) {
    const int fd = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
}
#endif
}  // namespace

std::string write_file_atomically(const std::string& path, const std::string& data) {
    const fs::path target(path);
    std::error_code ec;
    if (target.has_parent_path()) {
        fs::create_directories(target.parent_path(), ec);
        if (ec) {
            return "could not create '" + target.parent_path().string() + "': " + ec.message();
        }
    }
    const std::string tmp = path + ".tmp-" + secure_random_hex(6);

    std::FILE* file = std::fopen(tmp.c_str(), "wb");
    if (!file) {
        return "could not create '" + tmp + "': " + std::strerror(errno);
    }
    auto fail = [&](const std::string& what) {
        const std::string reason = what + ": " + std::strerror(errno);
        std::fclose(file);
        std::remove(tmp.c_str());
        return reason;
    };
    if (!data.empty() && std::fwrite(data.data(), 1, data.size(), file) != data.size()) {
        return fail("could not write '" + tmp + "'");
    }
    if (std::fflush(file) != 0) {
        return fail("could not write '" + tmp + "'");
    }
#ifdef _WIN32
    if (_commit(_fileno(file)) != 0) {
        return fail("could not flush '" + tmp + "' to disk");
    }
#else
    if (::fsync(fileno(file)) != 0) {
        return fail("could not flush '" + tmp + "' to disk");
    }
#endif
    if (std::fclose(file) != 0) {
        const std::string reason = "could not close '" + tmp + "': " + std::strerror(errno);
        std::remove(tmp.c_str());
        return reason;
    }

    fs::rename(tmp, target, ec);
    if (ec) {
        std::error_code ignored;
        fs::remove(tmp, ignored);
        return "could not replace '" + path + "': " + ec.message();
    }
#ifndef _WIN32
    sync_directory(target.parent_path());
#endif
    return {};
}

}  // namespace adai
