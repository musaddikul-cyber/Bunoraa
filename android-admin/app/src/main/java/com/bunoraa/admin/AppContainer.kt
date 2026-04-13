package com.bunoraa.admin

import android.content.Context
import com.bunoraa.admin.core.database.AdminDatabase
import com.bunoraa.admin.core.database.createAdminDatabase
import com.bunoraa.admin.core.datastore.SecureTokenStore
import com.bunoraa.admin.core.network.ApiClient
import com.bunoraa.admin.core.network.RealtimeClient
import com.bunoraa.admin.core.network.TokenProvider
import com.bunoraa.admin.feature.auth.AuthRepository
import com.bunoraa.admin.feature.dashboard.DashboardRepository

class AppContainer(context: Context) {
    private val tokenStore = SecureTokenStore(context)
    private val tokenProvider = object : TokenProvider {
        override fun accessToken(): String? = tokenStore.accessToken()
    }

    private val api = ApiClient.create(
        baseUrl = BuildConfig.API_BASE_URL,
        tokenProvider = tokenProvider,
    )

    private val database: AdminDatabase = createAdminDatabase(context)

    private val realtimeClient = RealtimeClient(tokenProvider)

    val authRepository = AuthRepository(api, tokenStore)
    val dashboardRepository = DashboardRepository(
        api,
        database.dashboardDao(),
        realtimeClient,
        BuildConfig.WS_BASE_URL,
    )
    val pushTokenRegistrar = PushTokenRegistrar(context, api, tokenStore)
}
