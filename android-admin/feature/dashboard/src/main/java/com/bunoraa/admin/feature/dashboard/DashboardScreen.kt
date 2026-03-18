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
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun DashboardRoute(viewModel: DashboardViewModel) {
    val state by viewModel.state.collectAsState()
    val snapshot = state.latest ?: state.cached

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Admin Dashboard", style = MaterialTheme.typography.headlineSmall)
        Text("Last updated: ${snapshot?.generatedAt ?: "-"}")

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
private fun MetricCard(title: String, value: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(title, style = MaterialTheme.typography.labelMedium)
            Spacer(modifier = Modifier.height(6.dp))
            Text(value, style = MaterialTheme.typography.headlineMedium)
        }
    }
}
