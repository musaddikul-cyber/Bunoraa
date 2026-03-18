package com.bunoraa.admin

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.compose.rememberNavController
import com.bunoraa.admin.auth.SsoAuthManager
import com.bunoraa.admin.auth.SsoProvider
import com.bunoraa.admin.core.common.Result
import com.bunoraa.admin.core.designsystem.BunoraaAdminTheme
import com.bunoraa.admin.feature.auth.AuthViewModel
import com.bunoraa.admin.feature.dashboard.DashboardViewModel

class MainActivity : ComponentActivity() {
    private val container by lazy { (application as BunoraaAdminApp).container }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            BunoraaAdminTheme(darkTheme = false) {
                val context = LocalContext.current
                val navController = rememberNavController()
                val authViewModel = remember { AuthViewModel(container.authRepository) }
                val dashboardViewModel = remember { DashboardViewModel(container.dashboardRepository) }
                val ssoManager = remember { SsoAuthManager(context) }
                var pendingProvider by remember { mutableStateOf<SsoProvider?>(null) }
                val ssoLauncher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.StartActivityForResult(),
                ) { result ->
                    val provider = pendingProvider ?: return@rememberLauncherForActivityResult
                    pendingProvider = null
                    ssoManager.handleAuthorizationResult(
                        data = result.data,
                        provider = provider,
                        onSuccess = { token -> authViewModel.loginWithSocial(provider.id, token) },
                        onError = { message -> authViewModel.setError(message) },
                    )
                }

                DisposableEffect(Unit) {
                    onDispose { ssoManager.dispose() }
                }

                AppNavHost(
                    navController = navController,
                    authViewModel = authViewModel,
                    dashboardViewModel = dashboardViewModel,
                    onSsoRequested = { providerId ->
                        val provider = SsoProvider.fromId(providerId)
                        if (provider == null) {
                            authViewModel.setError("Unsupported SSO provider.")
                            return@AppNavHost
                        }
                        when (val result = ssoManager.createAuthorizationIntent(provider)) {
                            is Result.Ok -> {
                                pendingProvider = provider
                                ssoLauncher.launch(result.value)
                            }
                            is Result.Err -> authViewModel.setError(result.message)
                        }
                    },
                )
            }
        }
    }
}
