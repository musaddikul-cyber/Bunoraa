package com.bunoraa.admin

import android.content.Intent
import com.bunoraa.admin.core.common.AdminDeepLink

object AdminDeepLinkParser {
    private const val EXTRA_ROUTE = "admin_route"
    private const val EXTRA_TITLE = "admin_title"
    private const val EXTRA_MESSAGE = "admin_message"
    private const val EXTRA_TYPE = "admin_type"
    private const val EXTRA_URL = "admin_url"
    private const val EXTRA_MODULE = "admin_module"

    fun toIntent(intent: Intent, deepLink: AdminDeepLink): Intent {
        return intent.apply {
            putExtra(EXTRA_ROUTE, deepLink.route)
            putExtra(EXTRA_TITLE, deepLink.title)
            putExtra(EXTRA_MESSAGE, deepLink.message)
            putExtra(EXTRA_TYPE, deepLink.type)
            putExtra(EXTRA_URL, deepLink.url)
            putExtra(EXTRA_MODULE, deepLink.route)
        }
    }

    fun fromIntent(intent: Intent?): AdminDeepLink? {
        if (intent == null) return null
        val route = intent.getStringExtra(EXTRA_ROUTE) ?: return null
        val title = intent.getStringExtra(EXTRA_TITLE) ?: "Bunoraa Admin"
        val message = intent.getStringExtra(EXTRA_MESSAGE) ?: ""
        val type = intent.getStringExtra(EXTRA_TYPE)
        val url = intent.getStringExtra(EXTRA_URL)
        val module = intent.getStringExtra(EXTRA_MODULE)
        return AdminDeepLink(
            route = module ?: route,
            title = title,
            message = message,
            type = type,
            url = url,
        )
    }

    fun fromPayload(payload: Map<String, String>, title: String, message: String): AdminDeepLink {
        val type = payload["type"] ?: payload["notification_type"] ?: payload["event_type"]
        val url = payload["url"]
        val route = resolveRoute(payload = payload, type = type)
        return AdminDeepLink(
            route = route,
            title = title,
            message = message,
            type = type,
            url = url,
        )
    }

    private fun resolveRoute(payload: Map<String, String>, type: String?): String {
        val explicit = payload["route"] ?: payload["module"]
        if (!explicit.isNullOrBlank()) return explicit
        return when {
            type?.startsWith("order_") == true -> "orders"
            type?.startsWith("payment_") == true -> "payments"
            type?.startsWith("review_") == true -> "reviews"
            type?.contains("coupon", ignoreCase = true) == true -> "promotions"
            type?.contains("promo", ignoreCase = true) == true -> "promotions"
            type?.contains("stock", ignoreCase = true) == true -> "catalog"
            type?.contains("price", ignoreCase = true) == true -> "pricing"
            type?.contains("subscription", ignoreCase = true) == true -> "subscriptions"
            type?.contains("chat", ignoreCase = true) == true -> "support"
            type?.contains("health", ignoreCase = true) == true -> "system-health"
            else -> AdminRoutes.Dashboard
        }
    }
}
