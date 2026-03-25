package com.bunoraa.admin.auth

import android.content.Context
import android.content.Intent
import android.net.Uri
import com.bunoraa.admin.BuildConfig
import com.bunoraa.admin.core.common.Result
import net.openid.appauth.AuthorizationException
import net.openid.appauth.AuthorizationRequest
import net.openid.appauth.AuthorizationResponse
import net.openid.appauth.AuthorizationService
import net.openid.appauth.AuthorizationServiceConfiguration
import net.openid.appauth.CodeVerifierUtil
import net.openid.appauth.ResponseTypeValues

enum class SsoProvider(val id: String) {
    GOOGLE("google-oauth2"),
    MICROSOFT("microsoft-graph");

    companion object {
        fun fromId(id: String): SsoProvider? = entries.firstOrNull { it.id == id }
    }
}

class SsoAuthManager(
    context: Context,
) {
    private val authService = AuthorizationService(context)

    fun createAuthorizationIntent(provider: SsoProvider): Result<Intent> {
        val clientId = clientId(provider)
        if (clientId.isBlank()) {
            return Result.Err("SSO client ID is missing for ${provider.name.lowercase()}.")
        }
        val redirectUri = redirectUri()
        if (redirectUri.scheme.isNullOrBlank()) {
            return Result.Err("SSO redirect scheme is missing.")
        }

        val request = AuthorizationRequest.Builder(
            serviceConfig(provider),
            clientId,
            ResponseTypeValues.CODE,
            redirectUri,
        )
            .setScopes(*scopes(provider).toTypedArray())
            .setCodeVerifier(CodeVerifierUtil.generateRandomCodeVerifier())
            .build()

        return Result.Ok(authService.getAuthorizationRequestIntent(request))
    }

    fun handleAuthorizationResult(
        data: Intent?,
        provider: SsoProvider,
        onSuccess: (String) -> Unit,
        onError: (String) -> Unit,
    ) {
        val response = AuthorizationResponse.fromIntent(data)
        val exception = AuthorizationException.fromIntent(data)

        if (response == null) {
            onError(exception?.errorDescription ?: "SSO cancelled.")
            return
        }

        val clientId = clientId(provider)
        if (clientId.isBlank()) {
            onError("SSO client ID is missing for ${provider.name.lowercase()}.")
            return
        }

        val tokenRequest = response.createTokenExchangeRequest()
        authService.performTokenRequest(tokenRequest) { tokenResponse, tokenException ->
            val accessToken = tokenResponse?.accessToken
            if (!accessToken.isNullOrBlank()) {
                onSuccess(accessToken)
            } else {
                onError(tokenException?.errorDescription ?: "Token exchange failed.")
            }
        }
    }

    fun dispose() {
        authService.dispose()
    }

    private fun serviceConfig(provider: SsoProvider): AuthorizationServiceConfiguration {
        return AuthorizationServiceConfiguration(
            authorizationEndpoint(provider),
            tokenEndpoint(provider),
        )
    }

    private fun authorizationEndpoint(provider: SsoProvider): Uri {
        return when (provider) {
            SsoProvider.GOOGLE -> Uri.parse("https://accounts.google.com/o/oauth2/v2/auth")
            SsoProvider.MICROSOFT -> Uri.parse(
                "https://login.microsoftonline.com/${microsoftTenant()}/oauth2/v2.0/authorize"
            )
        }
    }

    private fun tokenEndpoint(provider: SsoProvider): Uri {
        return when (provider) {
            SsoProvider.GOOGLE -> Uri.parse("https://oauth2.googleapis.com/token")
            SsoProvider.MICROSOFT -> Uri.parse(
                "https://login.microsoftonline.com/${microsoftTenant()}/oauth2/v2.0/token"
            )
        }
    }

    private fun clientId(provider: SsoProvider): String {
        return when (provider) {
            SsoProvider.GOOGLE -> BuildConfig.OIDC_GOOGLE_CLIENT_ID
            SsoProvider.MICROSOFT -> BuildConfig.OIDC_MICROSOFT_CLIENT_ID
        }
    }

    private fun microsoftTenant(): String {
        return if (BuildConfig.OIDC_MICROSOFT_TENANT.isBlank()) {
            "common"
        } else {
            BuildConfig.OIDC_MICROSOFT_TENANT
        }
    }

    private fun scopes(provider: SsoProvider): List<String> {
        return when (provider) {
            SsoProvider.GOOGLE -> listOf("openid", "email", "profile")
            SsoProvider.MICROSOFT -> listOf("openid", "email", "profile", "offline_access")
        }
    }

    private fun redirectUri(): Uri {
        if (BuildConfig.OIDC_REDIRECT_URI.isNotBlank()) {
            return Uri.parse(BuildConfig.OIDC_REDIRECT_URI)
        }
        return Uri.parse("${BuildConfig.OIDC_REDIRECT_SCHEME}:/oauth2redirect")
    }
}
