package com.adai.chatbot.network

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-10


import com.adai.chatbot.BuildConfig
import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import java.util.concurrent.TimeUnit

/**
 * The server's base URL is user-configurable at runtime (Settings screen), so Retrofit
 * can't be built once at app startup with a fixed baseUrl. This caches a single
 * Retrofit/ChatApiService pair and only rebuilds it when the host:port actually changes.
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
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(15, TimeUnit.SECONDS)
            .retryOnConnectionFailure(false)
            .apply {
                if (BuildConfig.DEBUG) {
                    addInterceptor(HttpLoggingInterceptor().apply {
                        level = HttpLoggingInterceptor.Level.BODY
                    })
                }
            }
            .build()
    }

    private var cachedKey: String? = null
    private var cachedService: ChatApiService? = null

    @Synchronized
    open fun serviceFor(
        host: String,
        port: Int,
        useHttps: Boolean = false,
        accessClientId: String = "",
        accessClientSecret: String = "",
    ): ChatApiService {
        // Relay mode (Cloudflare Tunnel via kerfbit.dev) terminates on 443 behind
        // Cloudflare Access, so no port is needed; LAN-direct mode keeps http://host:port.
        val baseUrl = if (useHttps) "https://$host/" else "http://$host:$port/"
        val key = "$baseUrl|$accessClientId|$accessClientSecret"
        cachedService?.let { if (cachedKey == key) return it }

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
            .create(ChatApiService::class.java)

        cachedKey = key
        cachedService = service
        return service
    }
}
