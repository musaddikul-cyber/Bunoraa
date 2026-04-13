package com.bunoraa.admin.feature.auth

import com.bunoraa.admin.core.common.Result
import com.bunoraa.admin.core.network.AdminApiService
import com.bunoraa.admin.core.network.AuthTokenResponse
import com.bunoraa.admin.core.network.MfaVerifyRequest
import com.bunoraa.admin.core.network.PasswordLoginRequest
import com.bunoraa.admin.core.network.SocialLoginRequest
import com.bunoraa.admin.core.datastore.TokenStore

class AuthRepository(
    private val api: AdminApiService,
    private val tokenStore: TokenStore,
) {
    suspend fun loginWithPassword(email: String, password: String): Result<AuthTokenResponse> {
        return try {
            val response = api.passwordLogin(PasswordLoginRequest(email, password))
            val payload = response.data
            if (response.success && payload != null) {
                val accessToken = payload.access
                val refreshToken = payload.refresh
                if (!payload.mfaRequired && !accessToken.isNullOrBlank() && !refreshToken.isNullOrBlank()) {
                    tokenStore.saveTokens(accessToken, refreshToken)
                }
                Result.Ok(payload)
            } else {
                Result.Err(response.message ?: "Login failed")
            }
        } catch (exc: Exception) {
            Result.Err("Login error", exc)
        }
    }

    suspend fun loginWithSocial(provider: String, accessToken: String): Result<AuthTokenResponse> {
        return try {
            val response = api.socialLogin(SocialLoginRequest(provider, accessToken))
            val payload = response.data
            if (response.success && payload != null) {
                val accessTokenValue = payload.access
                val refreshTokenValue = payload.refresh
                if (!payload.mfaRequired && !accessTokenValue.isNullOrBlank() && !refreshTokenValue.isNullOrBlank()) {
                    tokenStore.saveTokens(accessTokenValue, refreshTokenValue)
                }
                Result.Ok(payload)
            } else {
                Result.Err(response.message ?: "Social login failed")
            }
        } catch (exc: Exception) {
            Result.Err("Social login error", exc)
        }
    }

    suspend fun verifyMfa(mfaToken: String, method: String, code: String): Result<AuthTokenResponse> {
        return try {
            val response = api.verifyMfa(
                MfaVerifyRequest(
                    mfaToken = mfaToken,
                    method = method,
                    code = code,
                )
            )
            val payload = response.data
            if (response.success && payload != null) {
                val accessToken = payload.access
                val refreshToken = payload.refresh
                if (!accessToken.isNullOrBlank() && !refreshToken.isNullOrBlank()) {
                    tokenStore.saveTokens(accessToken, refreshToken)
                }
                Result.Ok(payload)
            } else {
                Result.Err(response.message ?: "MFA verification failed")
            }
        } catch (exc: Exception) {
            Result.Err("MFA error", exc)
        }
    }
}
