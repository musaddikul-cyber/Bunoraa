package com.bunoraa.admin

import android.content.Context
import androidx.room.Room
import com.bunoraa.admin.core.database.AdminDatabase
import com.bunoraa.admin.core.datastore.SecureTokenStore
import com.bunoraa.admin.core.network.ApiClient
import com.bunoraa.admin.feature.auth.AuthRepository
import com.bunoraa.admin.feature.dashboard.DashboardRepository

class AppContainer(context: Context) {
    private val tokenStore = SecureTokenStore(context)

    private val api = ApiClient.create(
        baseUrl = BuildConfig.API_BASE_URL,
        tokenProvider = object : com.bunoraa.admin.core.network.TokenProvider {
            override fun accessToken(): String? = tokenStore.accessToken()
        },
    )

    private val database = Room.databaseBuilder(
        context,
        AdminDatabase::class.java,
        "bunoraa_admin.db",
    ).fallbackToDestructiveMigration().build()

    val authRepository = AuthRepository(api, tokenStore)
    val dashboardRepository = DashboardRepository(api, database.dashboardDao())
}
