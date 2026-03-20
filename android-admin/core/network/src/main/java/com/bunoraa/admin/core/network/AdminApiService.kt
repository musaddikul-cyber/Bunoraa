package com.bunoraa.admin.core.network

import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

interface AdminApiService {
    @POST("auth/token/")
    suspend fun passwordLogin(@Body request: PasswordLoginRequest): ApiEnvelope<AuthTokenResponse>

    @POST("accounts/mfa/verify/")
    suspend fun verifyMfa(@Body request: MfaVerifyRequest): ApiEnvelope<AuthTokenResponse>

    @POST("admin/auth/social/")
    suspend fun socialLogin(@Body request: SocialLoginRequest): ApiEnvelope<AuthTokenResponse>

    @GET("admin/health/")
    suspend fun getHealth(): ApiEnvelope<HealthCheck>

    @GET("admin/dashboard/")
    suspend fun getDashboard(): ApiEnvelope<DashboardPayload>

    @POST("notifications/push-tokens/")
    suspend fun registerPushToken(@Body request: PushTokenRequest): ApiEnvelope<PushTokenResponse>
}
