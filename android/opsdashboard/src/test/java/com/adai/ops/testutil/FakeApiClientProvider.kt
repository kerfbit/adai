package com.adai.ops.testutil

import com.adai.ops.network.ApiClientProvider

/** Always returns the given fake service instance, regardless of host/port/serviceClass. */
class FakeApiClientProvider(private val service: Any) : ApiClientProvider() {
    @Suppress("UNCHECKED_CAST")
    override fun <T : Any> serviceFor(
        host: String,
        port: Int,
        serviceClass: Class<T>,
        useHttps: Boolean,
        accessClientId: String,
        accessClientSecret: String,
    ): T = service as T
}
