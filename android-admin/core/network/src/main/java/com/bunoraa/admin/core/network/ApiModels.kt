package com.bunoraa.admin.core.network

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

@Serializable
 data class ApiEnvelope<T>(
    val success: Boolean,
    val message: String? = null,
    val data: T? = null,
    val meta: JsonObject? = null,
)

@Serializable
 data class AuthTokenResponse(
    val access: String? = null,
    val refresh: String? = null,
    @SerialName("mfa_required") val mfaRequired: Boolean = false,
    @SerialName("mfa_token") val mfaToken: String? = null,
    val methods: List<String> = emptyList(),
)

@Serializable
 data class SocialLoginRequest(
    val provider: String,
    @SerialName("access_token") val accessToken: String,
)

@Serializable
 data class PasswordLoginRequest(
    val email: String,
    val password: String,
)

@Serializable
 data class MfaVerifyRequest(
    @SerialName("mfa_token") val mfaToken: String,
    val method: String,
    val code: String? = null,
    val credential: String? = null,
)

@Serializable
 data class HealthCheck(
    val status: String,
    val service: String,
    val version: String? = null,
    val environment: String? = null,
    val checks: Map<String, Map<String, String>> = emptyMap(),
    val timestamp: String? = null,
)

@Serializable
 data class DashboardTotals(
    val users: Long = 0,
    val products: Long = 0,
    val orders: Long = 0,
    @SerialName("orders_pending") val ordersPending: Long = 0,
    @SerialName("revenue_30d") val revenue30d: String = "0",
)

@Serializable
 data class DashboardPayload(
    @SerialName("generated_at") val generatedAt: String,
    @SerialName("window_days") val windowDays: Int,
    val totals: DashboardTotals,
)

@Serializable
data class PushTokenRequest(
    val token: String,
    @SerialName("device_type") val deviceType: String,
    @SerialName("device_name") val deviceName: String? = null,
    val platform: String? = null,
    @SerialName("app_version") val appVersion: String? = null,
    val locale: String? = null,
    val timezone: String? = null,
)

@Serializable
data class PushTokenResponse(
    @SerialName("token_id") val tokenId: String? = null,
)
