package com.bunoraa.admin

import android.content.Intent

data class AdminDeepLink(
    val route: String,
    val title: String,
    val message: String,
    val type: String? = null,
    val url: String? = null,
)

object AdminDeepLinkParser {
    private const val EXTRA_ROUTE = "admin_route"
    private const val EXTRA_TITLE = "admin_title"
    private const val EXTRA_MESSAGE = "admin_message"
    private const val EXTRA_TYPE = "admin_type"
    private const val EXTRA_URL = "admin_url"

    fun toIntent(intent: Intent, deepLink: AdminDeepLink): Intent {
        return intent.apply {
            putExtra(EXTRA_ROUTE, deepLink.route)
            putExtra(EXTRA_TITLE, deepLink.title)
            putExtra(EXTRA_MESSAGE, deepLink.message)
            putExtra(EXTRA_TYPE, deepLink.type)
            putExtra(EXTRA_URL, deepLink.url)
        }
    }

    fun fromIntent(intent: Intent?): AdminDeepLink? {
        if (intent == null) return null
        val route = intent.getStringExtra(EXTRA_ROUTE) ?: return null
        val title = intent.getStringExtra(EXTRA_TITLE) ?: "Bunoraa Admin"
        val message = intent.getStringExtra(EXTRA_MESSAGE) ?: ""
        val type = intent.getStringExtra(EXTRA_TYPE)
        val url = intent.getStringExtra(EXTRA_URL)
        return AdminDeepLink(route = route, title = title, message = message, type = type, url = url)
    }

    fun fromPayload(payload: Map<String, String>, title: String, message: String): AdminDeepLink {
        val type = payload["type"] ?: payload["notification_type"]
        val url = payload["url"]
        val route = when {
            type?.startsWith("order_") == true -> AdminRoutes.Dashboard
            type?.startsWith("review_") == true -> AdminRoutes.Dashboard
            type?.contains("chat", ignoreCase = true) == true -> AdminRoutes.Dashboard
            else -> AdminRoutes.Dashboard
        }
        return AdminDeepLink(
            route = route,
            title = title,
            message = message,
            type = type,
            url = url,
        )
    }
}
