package com.adai.ops.network

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import okhttp3.Interceptor
import okhttp3.Response

/**
 * Attaches Cloudflare Access service-token headers so requests routed through
 * the kerfbit.dev relay pass Access's edge auth check before ever reaching
 * cloudflared or the LAN. Only ever attached to a client when both values are
 * non-blank (see ApiClientProvider) — LAN-direct requests never carry it.
 */
class CloudflareAccessInterceptor(
    private val clientId: String,
    private val clientSecret: String,
) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request().newBuilder()
            .addHeader("CF-Access-Client-Id", clientId)
            .addHeader("CF-Access-Client-Secret", clientSecret)
            .build()
        return chain.proceed(request)
    }
}
