package com.bunoraa.admin.feature.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.bunoraa.admin.AdminDeepLink
import com.bunoraa.admin.core.network.RealtimeEvent
import com.bunoraa.admin.core.network.RealtimeStatus
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class DashboardViewModel(
    private val repository: DashboardRepository,
) : ViewModel() {
    private val _state = MutableStateFlow(DashboardUiState())
    val state: StateFlow<DashboardUiState> = _state.asStateFlow()
    private var realtimeJob: Job? = null
    private var pollingJob: Job? = null

    init {
        viewModelScope.launch {
            repository.observeDashboard().collect { cached ->
                _state.update { it.copy(cached = cached) }
            }
        }
        viewModelScope.launch {
            repository.realtimeStatus.collectLatest { status ->
                _state.update { it.copy(realtimeStatus = status) }
            }
        }
    }

    fun refresh() {
        _state.update { it.copy(isLoading = true, error = null) }
        viewModelScope.launch {
            val result = repository.refresh()
            result.fold(
                onSuccess = { fresh -> _state.update { it.copy(isLoading = false, latest = fresh) } },
                onFailure = { error -> _state.update { it.copy(isLoading = false, error = error.message) } },
            )
        }
    }

    fun startRealtime() {
        if (realtimeJob != null) return
        repository.connectRealtime()
        realtimeJob = viewModelScope.launch {
            repository.realtimeEvents.collect { event ->
                handleRealtimeEvent(event)
            }
        }
        startPolling()
    }

    fun stopRealtime() {
        realtimeJob?.cancel()
        realtimeJob = null
        pollingJob?.cancel()
        pollingJob = null
        repository.disconnectRealtime()
    }

    fun handleDeepLink(deepLink: AdminDeepLink) {
        _state.update { it.copy(lastDeepLink = deepLink) }
        refresh()
    }

    private fun handleRealtimeEvent(event: RealtimeEvent) {
        _state.update { it.copy(lastRealtimeEvent = event) }
        if (event.type in setOf("notification", "order_update", "price_update", "stock_update")) {
            refresh()
        }
    }

    private fun startPolling() {
        if (pollingJob != null) return
        pollingJob = viewModelScope.launch {
            while (true) {
                delay(60_000)
                refresh()
            }
        }
    }
}

data class DashboardUiState(
    val isLoading: Boolean = false,
    val error: String? = null,
    val cached: DashboardUiModel? = null,
    val latest: DashboardUiModel? = null,
    val realtimeStatus: RealtimeStatus = RealtimeStatus.Disconnected,
    val lastRealtimeEvent: RealtimeEvent? = null,
    val lastDeepLink: AdminDeepLink? = null,
)
