package com.bunoraa.admin.core.network

import kotlinx.serialization.json.JsonObject

data class RealtimeEvent(
    val type: String,
    val payload: JsonObject,
    val receivedAtMillis: Long = System.currentTimeMillis(),
)

sealed class RealtimeStatus {
    data object Connected : RealtimeStatus()
    data object Disconnected : RealtimeStatus()
    data class Error(val message: String) : RealtimeStatus()
}
