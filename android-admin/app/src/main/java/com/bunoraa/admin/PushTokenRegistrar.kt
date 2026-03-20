package com.bunoraa.admin

import android.content.Context
import android.os.Build
import com.bunoraa.admin.core.datastore.TokenStore
import com.bunoraa.admin.core.network.AdminApiService
import com.bunoraa.admin.core.network.PushTokenRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.TimeZone

class PushTokenRegistrar(
    private val context: Context,
    private val api: AdminApiService,
    private val tokenStore: TokenStore,
) {
    private val prefs = context.getSharedPreferences("bunoraa_admin_push", Context.MODE_PRIVATE)

    fun savePending(token: String) {
        prefs.edit().putString(KEY_PENDING, token).apply()
    }

    suspend fun registerIfPossible() {
        val accessToken = tokenStore.accessToken()
        val token = prefs.getString(KEY_PENDING, null)
        if (accessToken.isNullOrBlank() || token.isNullOrBlank()) return

        val request = PushTokenRequest(
            token = token,
            deviceType = "android",
            deviceName = Build.MODEL ?: "Android",
            platform = "android",
            appVersion = BuildConfig.VERSION_NAME,
            locale = Locale.getDefault().toString(),
            timezone = TimeZone.getDefault().id,
        )

        withContext(Dispatchers.IO) {
            try {
                val response = api.registerPushToken(request)
                if (response.success) {
                    prefs.edit().remove(KEY_PENDING).apply()
                }
            } catch (_: Exception) {
                // Keep token stored for retry after next login.
            }
        }
    }

    private companion object {
        private const val KEY_PENDING = "pending_token"
    }
}
