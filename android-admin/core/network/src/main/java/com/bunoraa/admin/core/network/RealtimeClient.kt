package com.bunoraa.admin.core.network

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit

class RealtimeClient(
    private val tokenProvider: TokenProvider,
) {
    private val json = Json {
        ignoreUnknownKeys = true
    }
    private val client = OkHttpClient.Builder()
        .pingInterval(30, TimeUnit.SECONDS)
        .build()
    private var socket: WebSocket? = null

    private val _events = MutableSharedFlow<RealtimeEvent>(extraBufferCapacity = 64)
    val events: SharedFlow<RealtimeEvent> = _events

    private val _status = MutableStateFlow<RealtimeStatus>(RealtimeStatus.Disconnected)
    val status: StateFlow<RealtimeStatus> = _status

    fun connect(baseUrl: String, path: String) {
        disconnect()
        val token = tokenProvider.accessToken()
        val wsBase = baseUrl.trimEnd('/')
        val wsPath = if (path.startsWith("/")) path else "/$path"
        val url = buildString {
            append(wsBase)
            append(wsPath)
            if (!token.isNullOrBlank()) {
                append(if (wsPath.contains("?")) "&" else "?")
                append("token=")
                append(token)
            }
        }

        val request = Request.Builder().url(url).build()
        socket = client.newWebSocket(
            request,
            object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                    _status.tryEmit(RealtimeStatus.Connected)
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    try {
                        val element = json.parseToJsonElement(text)
                        val obj = element.jsonObject
                        val type = obj["type"]?.toString()?.trim('"') ?: "message"
                        emitEvent(type, obj)
                    } catch (_: Exception) {
                        // Ignore malformed payloads.
                    }
                }

                override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                    _status.tryEmit(RealtimeStatus.Disconnected)
                    webSocket.close(code, reason)
                }

                override fun onFailure(
                    webSocket: WebSocket,
                    t: Throwable,
                    response: okhttp3.Response?,
                ) {
                    _status.tryEmit(RealtimeStatus.Error(t.message ?: "WebSocket error"))
                }
            },
        )
    }

    fun disconnect() {
        socket?.close(1000, "closing")
        socket = null
        _status.tryEmit(RealtimeStatus.Disconnected)
    }

    private fun emitEvent(type: String, payload: JsonObject) {
        _events.tryEmit(RealtimeEvent(type = type, payload = payload))
    }
}
