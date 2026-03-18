package com.bunoraa.admin.feature.auth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.bunoraa.admin.core.common.Result
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class AuthViewModel(
    private val repository: AuthRepository,
) : ViewModel() {
    private val _state = MutableStateFlow(AuthUiState())
    val state: StateFlow<AuthUiState> = _state.asStateFlow()

    fun login(email: String, password: String) {
        _state.value = _state.value.copy(isLoading = true, error = null)
        viewModelScope.launch {
            when (val result = repository.loginWithPassword(email, password)) {
                is Result.Ok -> handleAuthResponse(result.value)
                is Result.Err -> _state.value = _state.value.copy(isLoading = false, error = result.message)
            }
        }
    }

    fun loginWithSocial(provider: String, accessToken: String) {
        _state.value = _state.value.copy(isLoading = true, error = null)
        viewModelScope.launch {
            when (val result = repository.loginWithSocial(provider, accessToken)) {
                is Result.Ok -> handleAuthResponse(result.value)
                is Result.Err -> _state.value = _state.value.copy(isLoading = false, error = result.message)
            }
        }
    }

    fun verifyMfa(code: String, method: String) {
        val mfaToken = _state.value.mfaToken ?: return
        _state.value = _state.value.copy(isLoading = true, error = null)
        viewModelScope.launch {
            when (val result = repository.verifyMfa(mfaToken, method, code)) {
                is Result.Ok -> handleAuthResponse(result.value)
                is Result.Err -> _state.value = _state.value.copy(isLoading = false, error = result.message)
            }
        }
    }

    fun setError(message: String) {
        _state.value = _state.value.copy(isLoading = false, error = message)
    }

    private fun handleAuthResponse(payload: com.bunoraa.admin.core.network.AuthTokenResponse) {
        if (payload.mfaRequired) {
            _state.value = _state.value.copy(
                isLoading = false,
                mfaToken = payload.mfaToken,
                mfaMethods = payload.methods,
                isAuthenticated = false,
            )
            return
        }
        _state.value = _state.value.copy(
            isLoading = false,
            isAuthenticated = !payload.access.isNullOrBlank(),
            mfaToken = null,
            mfaMethods = emptyList(),
        )
    }
}

data class AuthUiState(
    val isLoading: Boolean = false,
    val error: String? = null,
    val mfaToken: String? = null,
    val mfaMethods: List<String> = emptyList(),
    val isAuthenticated: Boolean = false,
)
