// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-24

#pragma once

#ifndef _WIN32
#include <sys/socket.h>
#endif

namespace adai {

/**
 * TD-551: make an httplib server's listening port exclusive to this process.
 *
 * cpp-httplib's default socket options set SO_REUSEPORT on POSIX. A second instance started on a
 * port that is already in use (a manual start next to the systemd unit, a leftover test server, an
 * overlapping restart) therefore bound without error, and the kernel spread new connections across
 * both processes, each with its own locks and memory over the same data. Verified with two
 * registry_server instances: serialized fetches came back as duplicate slices.
 *
 * This replaces the options with SO_REUSEADDR alone, which still allows an immediate restart over
 * old TIME_WAIT connections but makes a second listener fail to bind, so listen() returns false.
 * Windows is left alone: httplib already sets SO_EXCLUSIVEADDRUSE there.
 *
 * Call it before listen()/bind_to_port(). Templated on the server type so this header needn't
 * include httplib.h.
 */
template <typename Server>
void use_exclusive_listen_port(Server& server) {
#ifndef _WIN32
    server.set_socket_options([](auto sock) {
        int yes = 1;
        ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    });
#else
    (void)server;
#endif
}

}  // namespace adai
