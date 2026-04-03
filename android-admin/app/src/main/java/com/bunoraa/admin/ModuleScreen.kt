package com.bunoraa.admin

import android.content.Intent
import android.net.Uri
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
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

data class AdminModuleInfo(
    val id: String,
    val title: String,
    val description: String,
    val adminPath: String? = null,
)

private val ADMIN_MODULES = listOf(
    AdminModuleInfo(
        id = "dashboard",
        title = "Dashboard",
        description = "Operational overview and KPIs.",
        adminPath = "",
    ),
    AdminModuleInfo(
        id = "orders",
        title = "Orders",
        description = "Orders, fulfillment, and status changes.",
        adminPath = "orders/order/",
    ),
    AdminModuleInfo(
        id = "catalog",
        title = "Catalog",
        description = "Products, categories, variants, and inventory.",
        adminPath = "catalog/product/",
    ),
    AdminModuleInfo(
        id = "pricing",
        title = "Pricing",
        description = "Currencies, exchange rates, and price rules.",
        adminPath = "i18n/currency/",
    ),
    AdminModuleInfo(
        id = "promotions",
        title = "Promotions",
        description = "Coupons, banners, and active sales.",
        adminPath = "promotions/coupon/",
    ),
    AdminModuleInfo(
        id = "cms",
        title = "CMS",
        description = "Pages, FAQ, and site settings.",
        adminPath = "pages/page/",
    ),
    AdminModuleInfo(
        id = "reviews",
        title = "Reviews",
        description = "Customer reviews and moderation queue.",
        adminPath = "reviews/review/",
    ),
    AdminModuleInfo(
        id = "shipping",
        title = "Shipping",
        description = "Carriers, methods, rates, and shipments.",
        adminPath = "shipping/shippingmethod/",
    ),
    AdminModuleInfo(
        id = "payments",
        title = "Payments",
        description = "Payments, methods, and transactions.",
        adminPath = "payments/payment/",
    ),
    AdminModuleInfo(
        id = "subscriptions",
        title = "Subscriptions",
        description = "Plans, subscribers, and renewals.",
        adminPath = "subscriptions/subscription/",
    ),
    AdminModuleInfo(
        id = "notifications",
        title = "Notifications",
        description = "Templates, deliveries, and preferences.",
        adminPath = "notifications/notification/",
    ),
    AdminModuleInfo(
        id = "analytics",
        title = "Analytics",
        description = "Dashboards, daily stats, and trends.",
        adminPath = "analytics/dailystat/",
    ),
    AdminModuleInfo(
        id = "support",
        title = "Support",
        description = "Conversations and agent tooling.",
        adminPath = "chat/conversation/",
    ),
    AdminModuleInfo(
        id = "system-health",
        title = "System Health",
        description = "Service health checks and operational status.",
        adminPath = null,
    ),
)

@Composable
fun ModuleRoute(
    module: String,
    adminBaseUrl: String,
    deepLink: AdminDeepLink? = null,
    onDeepLinkHandled: () -> Unit = {},
    onBack: () -> Unit = {},
) {
    val info = remember(module) { resolveModuleInfo(module) }
    val context = LocalContext.current
    var activeDeepLink by remember { mutableStateOf<AdminDeepLink?>(null) }
    val adminUrl = remember(adminBaseUrl, info) { resolveAdminUrl(adminBaseUrl, info) }
    val relatedUrl = remember(adminBaseUrl, activeDeepLink) {
        activeDeepLink?.url?.let { resolveRelativeUrl(adminBaseUrl, it) }
    }

    LaunchedEffect(deepLink) {
        if (deepLink != null) {
            activeDeepLink = deepLink
            onDeepLinkHandled()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(info.title, style = MaterialTheme.typography.headlineSmall)
        Text(info.description, style = MaterialTheme.typography.bodyMedium)

        if (activeDeepLink != null) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(activeDeepLink!!.title, style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(activeDeepLink!!.message, style = MaterialTheme.typography.bodyMedium)
                    if (!activeDeepLink!!.type.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            "Type: ${activeDeepLink!!.type}",
                            style = MaterialTheme.typography.labelMedium,
                        )
                    }
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(
                onClick = { openUrl(context, adminUrl) },
                modifier = Modifier.weight(1f),
            ) {
                Text("Open Admin Web")
            }
            OutlinedButton(
                onClick = onBack,
                modifier = Modifier.weight(1f),
            ) {
                Text("Back to Dashboard")
            }
        }

        if (!relatedUrl.isNullOrBlank()) {
            OutlinedButton(onClick = { openUrl(context, relatedUrl) }) {
                Text("Open Related Link")
            }
        }
    }
}

private fun resolveModuleInfo(module: String): AdminModuleInfo {
    val key = module.lowercase()
    return ADMIN_MODULES.firstOrNull { it.id == key }
        ?: AdminModuleInfo(
            id = key,
            title = key.replace('-', ' ').replaceFirstChar { it.uppercase() },
            description = "Module updates and actions.",
            adminPath = null,
        )
}

private fun resolveAdminUrl(adminBaseUrl: String, info: AdminModuleInfo): String {
    val base = adminBaseUrl.trimEnd('/')
    val path = info.adminPath?.trimStart('/')
    return if (path.isNullOrBlank()) {
        "$base/"
    } else {
        "$base/$path"
    }
}

private fun resolveRelativeUrl(adminBaseUrl: String, url: String): String {
    val trimmed = url.trim()
    if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
        return trimmed
    }
    val adminRoot = adminBaseUrl.trimEnd('/').removeSuffix("/admin")
    val prefix = adminRoot.ifBlank { adminBaseUrl.trimEnd('/') }
    return if (trimmed.startsWith("/")) {
        "$prefix$trimmed"
    } else {
        "$prefix/$trimmed"
    }
}

private fun openUrl(context: android.content.Context, url: String) {
    runCatching {
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
    }
}
