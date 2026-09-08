package com.adai.ops.network

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import com.adai.ops.BuildConfig
import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import java.util.concurrent.TimeUnit

/**
 * All three backends (metrics/MNS/registry) have independently configurable host:port
 * pairs, so Retrofit clients are built lazily per (service interface, base URL) and
 * cached — mirrors the chatbot app's ApiClientProvider, generalized to three services.
 */
open class ApiClientProvider {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        explicitNulls = false
    }

    private val baseOkHttpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(15, TimeUnit.SECONDS)
            .writeTimeout(15, TimeUnit.SECONDS)
            .retryOnConnectionFailure(false)
            .apply {
                if (BuildConfig.DEBUG) {
                    addInterceptor(HttpLoggingInterceptor().apply {
                        level = HttpLoggingInterceptor.Level.BASIC
                    })
                }
            }
            .build()
    }

    private val cache = mutableMapOf<Pair<Class<*>, String>, Any>()

    @Synchronized
    open fun <T : Any> serviceFor(
        host: String,
        port: Int,
        serviceClass: Class<T>,
        useHttps: Boolean = false,
        accessClientId: String = "",
        accessClientSecret: String = "",
    ): T {
        // Relay mode (Cloudflare Tunnel via kerfbit.dev) terminates on 443 behind
        // Cloudflare Access, so no port is needed; LAN-direct mode keeps http://host:port.
        val baseUrl = if (useHttps) "https://$host/" else "http://$host:$port/"
        val key = serviceClass to "$baseUrl|$accessClientId|$accessClientSecret"

        @Suppress("UNCHECKED_CAST")
        (cache[key] as? T)?.let { return it }

        val client = if (accessClientId.isNotBlank() && accessClientSecret.isNotBlank()) {
            baseOkHttpClient.newBuilder()
                .addInterceptor(CloudflareAccessInterceptor(accessClientId, accessClientSecret))
                .build()
        } else {
            baseOkHttpClient
        }

        val service = Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(client)
            .addConverterFactory(json.asConverterFactory("application/json".toMediaType()))
            .build()
            .create(serviceClass)

        cache[key] = service
        return service
    }
}
