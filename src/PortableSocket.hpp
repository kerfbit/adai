#pragma once

// @adai-status: beta        (TD-161 — new, no test coverage yet beyond the callers it replaces)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13

/**
 * @file PortableSocket.hpp
 * @brief Portable BSD-socket/Winsock abstraction for the one file in this codebase that speaks
 * raw sockets directly (src/FtpDataServer.hpp — TD-161). `::socket()`/`::bind()`/`::listen()`/
 * `::accept()`/`::shutdown()`/`::inet_ntop()` themselves are already identical across POSIX and
 * Winsock (Winsock deliberately mirrors the BSD socket API for these), so this header only wraps
 * the handful of genuine differences: the socket descriptor's own type (`SOCKET` — an unsigned,
 * pointer-sized type on 64-bit Windows, not interchangeable with a plain `int` — vs POSIX's
 * plain `int`), `closesocket()` vs `::close()`, a per-process `WSAStartup()`/`WSACleanup()`
 * lifecycle that nothing else in this codebase's own code establishes independently, a
 * millisecond `DWORD` vs a `struct timeval` for `SO_RCVTIMEO`, `WSAGetLastError()` vs `errno`,
 * and the `int`-vs-`size_t` length parameter `send()`/`recv()` take on Windows vs POSIX.
 * Mirrors PortableTime.hpp's approach for TD-160.
 *
 * Windows/MinGW only — this codebase's only supported non-POSIX target (see
 * cmake/toolchains/mingw-w64.cmake). `socklen_t` and `ssize_t` are assumed available from
 * MinGW-w64's own headers (`<ws2tcpip.h>` / `<sys/types.h>`) rather than redefined here, since raw
 * MSVC (which lacks both) is out of scope for this project.
 */

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <cstddef>
#include <string>

namespace adai {

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
constexpr int kShutdownBoth = SD_BOTH;
#else
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
constexpr int kShutdownBoth = SHUT_RDWR;
#endif

/**
 * One-time process-wide Winsock startup/shutdown, held for as long as the owning object needs
 * working sockets. No-op on non-Windows platforms. `WSAStartup()`/`WSACleanup()` are refcounted
 * by the OS (each paired call increments/decrements an internal count; the Winsock DLL stays
 * loaded until the last `WSACleanup()`), so constructing more than one of these — or this one
 * coexisting with cpp-httplib's own internal `WSAStartup` call, which every httplib-consuming
 * binary in this codebase already makes — is safe. This guard exists so the socket code in
 * FtpDataServer.hpp is correct standalone, not dependent on some other component in the same
 * process having already initialized Winsock first.
 */
class WinsockGuard {
   public:
    WinsockGuard() {
#ifdef _WIN32
        WSADATA wsa_data;
        WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif
    }
    ~WinsockGuard() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
    WinsockGuard(const WinsockGuard&) = delete;
    WinsockGuard& operator=(const WinsockGuard&) = delete;
};

/** Portable socket close — `closesocket()` on Windows (a `SOCKET` is not a CRT file descriptor,
 *  `::close()` does not accept one), `::close()` elsewhere. */
inline void close_socket(socket_t s) {
#ifdef _WIN32
    ::closesocket(s);
#else
    ::close(s);
#endif
}

/** Portable `setsockopt(SOL_SOCKET, SO_REUSEADDR)`. Wrapped only for the `optval` pointer type —
 *  Windows' `setsockopt()` wants `const char*`, POSIX's wants `const void*` — rather than relying
 *  on an `int*` happening to convert to both, as the call sites this replaces did. */
inline void set_reuse_addr(socket_t s) {
    int opt = 1;
#ifdef _WIN32
    ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
}

/**
 * Portable receive-timeout setter for `SO_RCVTIMEO`. POSIX takes a `struct timeval`; Windows
 * takes a `DWORD` of milliseconds directly — passing a `struct timeval`'s raw bytes as a `DWORD`
 * on Windows compiles (both are just pointers to bytes as far as `setsockopt()` is concerned) but
 * silently sets a nonsense timeout instead of failing loudly, exactly the "compiles fine, wrong
 * at runtime" trap this TD's own writeup calls out for sockets generally.
 */
inline void set_recv_timeout(socket_t s, int seconds) {
#ifdef _WIN32
    DWORD timeout_ms = static_cast<DWORD>(seconds) * 1000;
    ::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                 sizeof(timeout_ms));
#else
    struct timeval tv {
        seconds, 0
    };
    ::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

/**
 * Portable send()/recv(). Windows' length parameter is `int`, not `size_t`/`ssize_t` — every
 * buffer this file ever passes (FTP control lines, one 64KB data-transfer chunk) fits comfortably
 * in an `int`, so the narrowing cast is safe. Applies POSIX's `MSG_NOSIGNAL` automatically
 * (prevents a `SIGPIPE` on a peer that closed its read side mid-write); Windows' `send()` never
 * raises that signal in the first place, so no equivalent flag is needed there.
 */
inline long send_bytes(socket_t s, const char* data, std::size_t len) {
#ifdef _WIN32
    return ::send(s, data, static_cast<int>(len), 0);
#else
    return ::send(s, data, len, MSG_NOSIGNAL);
#endif
}

inline long recv_bytes(socket_t s, char* data, std::size_t len) {
#ifdef _WIN32
    return ::recv(s, data, static_cast<int>(len), 0);
#else
    return ::recv(s, data, len, 0);
#endif
}

/**
 * Portable "describe the last socket error" — `errno`/`strerror()` on POSIX, `WSAGetLastError()`
 * on Windows (Winsock errors are never reported via `errno`). Returns just the numeric code on
 * Windows rather than a `FormatMessage`-derived string: sufficient for this codebase's
 * log-and-move-on error handling, and avoids pulling in `FormatMessageA`'s ANSI/wide-string
 * handling for a feature this file only ever uses in a log line, never for control flow.
 */
inline std::string last_socket_error() {
#ifdef _WIN32
    return "WSA error " + std::to_string(WSAGetLastError());
#else
    return std::strerror(errno);
#endif
}

}  // namespace adai
