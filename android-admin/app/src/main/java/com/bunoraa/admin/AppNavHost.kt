package com.bunoraa.admin

import android.net.Uri
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import androidx.navigation.NavType
import com.bunoraa.admin.feature.auth.AuthRoute
import com.bunoraa.admin.feature.auth.AuthViewModel
import com.bunoraa.admin.feature.dashboard.DashboardRoute
import com.bunoraa.admin.feature.dashboard.DashboardViewModel

object AdminRoutes {
    const val Auth = "auth"
    const val Dashboard = "dashboard"
    const val Module = "module"
    const val ModuleRoute = "module/{module}"

    fun moduleRoute(module: String): String = "module/${Uri.encode(module)}"
}

@Composable
fun AppNavHost(
    navController: NavHostController,
    authViewModel: AuthViewModel,
    dashboardViewModel: DashboardViewModel,
    onSsoRequested: (String) -> Unit,
    deepLink: AdminDeepLink? = null,
    onDeepLinkHandled: () -> Unit = {},
) {
    val authState by authViewModel.state.collectAsState()
    val adminBaseUrl = rememberAdminBaseUrl()

    DisposableEffect(authState.isAuthenticated) {
        if (authState.isAuthenticated) {
            dashboardViewModel.startRealtime()
        }
        onDispose {
            dashboardViewModel.stopRealtime()
        }
    }

    LaunchedEffect(authState.isAuthenticated, deepLink) {
        if (!authState.isAuthenticated || deepLink == null) return@LaunchedEffect
        val target = when (deepLink.route.lowercase()) {
            AdminRoutes.Dashboard -> AdminRoutes.Dashboard
            AdminRoutes.Auth -> AdminRoutes.Dashboard
            else -> AdminRoutes.moduleRoute(deepLink.route.lowercase())
        }
        if (navController.currentDestination?.route != target) {
            navController.navigate(target) {
                popUpTo(AdminRoutes.Dashboard) { inclusive = false }
                launchSingleTop = true
            }
        }
    }

    NavHost(navController = navController, startDestination = AdminRoutes.Auth) {
        composable(AdminRoutes.Auth) {
            AuthRoute(
                viewModel = authViewModel,
                onAuthenticated = { navController.navigate(AdminRoutes.Dashboard) },
                onSsoRequested = onSsoRequested,
            )
        }
        composable(AdminRoutes.Dashboard) {
            DashboardRoute(
                viewModel = dashboardViewModel,
                deepLink = deepLink,
                onDeepLinkHandled = onDeepLinkHandled,
            )
        }
        composable(
            AdminRoutes.ModuleRoute,
            arguments = listOf(navArgument("module") { type = NavType.StringType }),
        ) { backStackEntry ->
            val rawModule = backStackEntry.arguments?.getString("module") ?: AdminRoutes.Dashboard
            val module = Uri.decode(rawModule)
            ModuleRoute(
                module = module,
                adminBaseUrl = adminBaseUrl,
                deepLink = deepLink,
                onDeepLinkHandled = onDeepLinkHandled,
                onBack = { navController.popBackStack() },
            )
        }
    }
}

@Composable
private fun rememberAdminBaseUrl(): String {
    val apiBase = BuildConfig.API_BASE_URL.trim()
    val trimmed = apiBase.trimEnd('/')
    val withoutApi = if (trimmed.endsWith("/api/v1")) {
        trimmed.removeSuffix("/api/v1")
    } else {
        trimmed
    }
    return "${withoutApi.trimEnd('/')}/admin/"
}
