package com.bunoraa.admin.feature.dashboard

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.bunoraa.admin.AdminDeepLink
import com.bunoraa.admin.core.network.RealtimeStatus

@Composable
fun DashboardRoute(
    viewModel: DashboardViewModel,
    deepLink: AdminDeepLink? = null,
    onDeepLinkHandled: () -> Unit = {},
) {
    val state by viewModel.state.collectAsState()
    val snapshot = state.latest ?: state.cached

    LaunchedEffect(deepLink) {
        if (deepLink != null) {
            viewModel.handleDeepLink(deepLink)
            onDeepLinkHandled()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Admin Dashboard", style = MaterialTheme.typography.headlineSmall)
        Text("Last updated: ${snapshot?.generatedAt ?: "-"}")
        RealtimeBadge(state.realtimeStatus)

        if (state.lastDeepLink != null) {
            NotificationCard(
                title = "${state.lastDeepLink.title} (${state.lastDeepLink.route})",
                message = state.lastDeepLink.message,
            )
        }
        if (state.lastRealtimeEvent != null) {
            NotificationCard(
                title = "Realtime event: ${state.lastRealtimeEvent.type}",
                message = state.lastRealtimeEvent.payload.toString(),
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            MetricCard("Users", snapshot?.users?.toString() ?: "-")
            MetricCard("Products", snapshot?.products?.toString() ?: "-")
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            MetricCard("Orders", snapshot?.orders?.toString() ?: "-")
            MetricCard("Pending", snapshot?.ordersPending?.toString() ?: "-")
        }
        MetricCard("Revenue (30d)", snapshot?.revenue30d ?: "-")

        Button(onClick = { viewModel.refresh() }) {
            Text(if (state.isLoading) "Refreshing..." else "Refresh")
        }
        if (state.error != null) {
            Text(state.error ?: "", color = MaterialTheme.colorScheme.error)
        }
    }
}

@Composable
private fun RealtimeBadge(status: RealtimeStatus) {
    val label = when (status) {
        RealtimeStatus.Connecting -> "Realtime: connecting"
        is RealtimeStatus.Reconnecting -> "Realtime: reconnecting (${status.attempt})"
        RealtimeStatus.Connected -> "Realtime: connected"
        RealtimeStatus.Disconnected -> "Realtime: disconnected"
        is RealtimeStatus.Error -> "Realtime error: ${status.message}"
    }
    Text(label, style = MaterialTheme.typography.labelMedium)
}

@Composable
private fun NotificationCard(title: String, message: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Spacer(modifier = Modifier.height(8.dp))
            Text(message, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun MetricCard(title: String, value: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(title, style = MaterialTheme.typography.labelMedium)
            Spacer(modifier = Modifier.height(6.dp))
            Text(value, style = MaterialTheme.typography.headlineMedium)
        }
    }
}
