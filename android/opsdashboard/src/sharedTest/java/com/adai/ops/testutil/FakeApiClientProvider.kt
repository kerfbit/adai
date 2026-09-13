package com.adai.ops.testutil

// @adai-status: beta        (in-memory fake backing both plain-JVM and instrumented tests, TD-048)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import com.adai.ops.network.ApiClientProvider

data class ServiceForCall(
    val host: String,
    val port: Int,
    val serviceClass: Class<*>,
    val useHttps: Boolean,
    val accessClientId: String,
    val accessClientSecret: String,
)

/** Always returns the given fake service instance, regardless of host/port/serviceClass. */
class FakeApiClientProvider(private val service: Any) : ApiClientProvider() {
    val calls = mutableListOf<ServiceForCall>()

    @Suppress("UNCHECKED_CAST")
    override fun <T : Any> serviceFor(
        host: String,
        port: Int,
        serviceClass: Class<T>,
        useHttps: Boolean,
        accessClientId: String,
        accessClientSecret: String,
    ): T {
        calls += ServiceForCall(host, port, serviceClass, useHttps, accessClientId, accessClientSecret)
        return service as T
    }
}
