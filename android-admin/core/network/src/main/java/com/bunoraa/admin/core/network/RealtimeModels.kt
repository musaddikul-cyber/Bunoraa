package com.bunoraa.admin.core.network

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

data class RealtimeEvent(
    val type: String,
    val payload: JsonObject,
    val receivedAtMillis: Long = System.currentTimeMillis(),
)

@Serializable
data class RealtimePollingPayload(
    val events: List<RealtimePollingEvent> = emptyList(),
    @SerialName("next_since") val nextSince: String? = null,
    @SerialName("server_time") val serverTime: String? = null,
)

@Serializable
data class RealtimePollingEvent(
    val type: String,
    val module: String? = null,
    @SerialName("entity_type") val entityType: String? = null,
    @SerialName("entity_id") val entityId: String? = null,
    val timestamp: String? = null,
    val payload: JsonObject = JsonObject(emptyMap()),
)

sealed class RealtimeStatus {
    data object Connecting : RealtimeStatus()
    data class Reconnecting(val attempt: Int, val delayMillis: Long) : RealtimeStatus()
    data object Connected : RealtimeStatus()
    data object Disconnected : RealtimeStatus()
    data class Error(val message: String) : RealtimeStatus()
}
