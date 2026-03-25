package com.bunoraa.admin.core.network

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
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
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private var socket: WebSocket? = null
    private var reconnectJob: Job? = null
    private var reconnectAttempts = 0
    private var desiredConnection = false
    private var connectionBaseUrl: String = ""
    private var connectionPath: String = ""

    private val _events = MutableSharedFlow<RealtimeEvent>(extraBufferCapacity = 64)
    val events: SharedFlow<RealtimeEvent> = _events

    private val _status = MutableStateFlow<RealtimeStatus>(RealtimeStatus.Disconnected)
    val status: StateFlow<RealtimeStatus> = _status

    fun connect(baseUrl: String, path: String) {
        connectionBaseUrl = baseUrl
        connectionPath = path
        desiredConnection = true
        reconnectAttempts = 0
        reconnectJob?.cancel()
        reconnectJob = null
        _status.tryEmit(RealtimeStatus.Connecting)
        openSocket()
    }

    fun disconnect() {
        desiredConnection = false
        reconnectAttempts = 0
        reconnectJob?.cancel()
        reconnectJob = null
        socket?.close(1000, "closing")
        socket = null
        _status.tryEmit(RealtimeStatus.Disconnected)
    }

    private fun openSocket() {
        socket?.close(1000, "reconnecting")
        socket = null

        val token = tokenProvider.accessToken()
        if (token.isNullOrBlank()) {
            _status.tryEmit(RealtimeStatus.Error("Missing access token for realtime connection."))
            return
        }

        val wsBase = connectionBaseUrl.trimEnd('/')
        val wsPath = if (connectionPath.startsWith("/")) connectionPath else "/$connectionPath"
        val url = "$wsBase$wsPath"

        val request = Request.Builder()
            .url(url)
            .addHeader("Authorization", "Bearer $token")
            .build()

        socket = client.newWebSocket(
            request,
            object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                    reconnectAttempts = 0
                    reconnectJob?.cancel()
                    reconnectJob = null
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
                    webSocket.close(code, reason)
                    handleSocketClosed(reason)
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    handleSocketClosed(reason)
                }

                override fun onFailure(
                    webSocket: WebSocket,
                    t: Throwable,
                    response: okhttp3.Response?,
                ) {
                    _status.tryEmit(RealtimeStatus.Error(t.message ?: "WebSocket error"))
                    scheduleReconnect()
                }
            },
        )
    }

    private fun handleSocketClosed(reason: String?) {
        socket = null
        if (!desiredConnection) {
            _status.tryEmit(RealtimeStatus.Disconnected)
            return
        }
        if (!reason.isNullOrBlank()) {
            _status.tryEmit(RealtimeStatus.Error(reason))
        }
        scheduleReconnect()
    }

    private fun scheduleReconnect() {
        if (!desiredConnection) return
        if (reconnectJob != null) return

        reconnectAttempts += 1
        val backoffFactor = 1 shl (reconnectAttempts - 1).coerceAtMost(5)
        val delayMillis = minOf(30_000L, 1_000L * backoffFactor)
        _status.tryEmit(RealtimeStatus.Reconnecting(reconnectAttempts, delayMillis))

        reconnectJob = scope.launch {
            delay(delayMillis)
            reconnectJob = null
            if (desiredConnection) {
                openSocket()
            }
        }
    }

    private fun emitEvent(type: String, payload: JsonObject) {
        _events.tryEmit(RealtimeEvent(type = type, payload = payload))
    }
}
