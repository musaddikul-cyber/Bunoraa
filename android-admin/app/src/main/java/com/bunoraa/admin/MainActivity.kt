package com.bunoraa.admin

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.navigation.compose.rememberNavController
import com.bunoraa.admin.auth.SsoAuthManager
import com.bunoraa.admin.auth.SsoProvider
import com.bunoraa.admin.core.common.Result
import com.bunoraa.admin.core.designsystem.BunoraaAdminTheme
import com.bunoraa.admin.feature.auth.AuthViewModel
import com.bunoraa.admin.feature.dashboard.DashboardViewModel

class MainActivity : ComponentActivity() {
    private val container by lazy { (application as BunoraaAdminApp).container }
    private val deepLinkState = mutableStateOf<AdminDeepLink?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        deepLinkState.value = AdminDeepLinkParser.fromIntent(intent)

        setContent {
            BunoraaAdminTheme(darkTheme = false) {
                val context = LocalContext.current
                val navController = rememberNavController()
                val authViewModel = remember { AuthViewModel(container.authRepository) }
                val dashboardViewModel = remember { DashboardViewModel(container.dashboardRepository) }
                val ssoManager = remember { SsoAuthManager(context) }
                var pendingProvider by remember { mutableStateOf<SsoProvider?>(null) }
                val authState by authViewModel.state.collectAsState()
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
                val notificationPermissionLauncher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.RequestPermission(),
                ) { }

                DisposableEffect(Unit) {
                    onDispose { ssoManager.dispose() }
                }

                LaunchedEffect(Unit) {
                    if (
                        Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                        ContextCompat.checkSelfPermission(
                            context,
                            Manifest.permission.POST_NOTIFICATIONS,
                        ) != PackageManager.PERMISSION_GRANTED
                    ) {
                        notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                    }
                }

                LaunchedEffect(authState.isAuthenticated) {
                    if (authState.isAuthenticated) {
                        container.pushTokenRegistrar.registerIfPossible()
                    }
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
                    deepLink = deepLinkState.value,
                    onDeepLinkHandled = { deepLinkState.value = null },
                )
            }
        }
    }

    override fun onNewIntent(intent: android.content.Intent?) {
        super.onNewIntent(intent)
        deepLinkState.value = AdminDeepLinkParser.fromIntent(intent)
    }
}
