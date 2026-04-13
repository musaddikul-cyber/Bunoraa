package com.bunoraa.admin.core.common

data class AdminDeepLink(
    val route: String,
    val title: String,
    val message: String,
    val type: String? = null,
    val url: String? = null,
)
