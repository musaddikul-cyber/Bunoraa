package com.bunoraa.admin

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.bunoraa.admin.feature.auth.AuthRoute
import com.bunoraa.admin.feature.auth.AuthViewModel
import com.bunoraa.admin.feature.dashboard.DashboardRoute
import com.bunoraa.admin.feature.dashboard.DashboardViewModel

object AdminRoutes {
    const val Auth = "auth"
    const val Dashboard = "dashboard"
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
    }
}
