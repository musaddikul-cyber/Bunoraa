package com.bunoraa.admin.feature.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class DashboardViewModel(
    private val repository: DashboardRepository,
) : ViewModel() {
    private val _state = MutableStateFlow(DashboardUiState())
    val state: StateFlow<DashboardUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            repository.observeDashboard().collect { cached ->
                _state.update { it.copy(cached = cached) }
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
}

data class DashboardUiState(
    val isLoading: Boolean = false,
    val error: String? = null,
    val cached: DashboardUiModel? = null,
    val latest: DashboardUiModel? = null,
)
