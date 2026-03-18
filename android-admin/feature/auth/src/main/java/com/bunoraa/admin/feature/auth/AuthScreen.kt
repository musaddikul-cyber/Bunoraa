package com.bunoraa.admin.feature.auth

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp

@Composable
fun AuthRoute(
    viewModel: AuthViewModel,
    onAuthenticated: () -> Unit,
    onSsoRequested: (String) -> Unit,
) {
    val state by viewModel.state.collectAsState()

    LaunchedEffect(state.isAuthenticated) {
        if (state.isAuthenticated) {
            onAuthenticated()
        }
    }

    if (state.mfaToken != null) {
        MfaScreen(
            state = state,
            onVerify = { method, code -> viewModel.verifyMfa(code, method) },
        )
        return
    }

    LoginScreen(
        state = state,
        onLogin = { email, password -> viewModel.login(email, password) },
        onSsoRequested = onSsoRequested,
    )
}

@Composable
private fun LoginScreen(
    state: AuthUiState,
    onLogin: (String, String) -> Unit,
    onSsoRequested: (String) -> Unit,
) {
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Bunoraa Admin", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))
        OutlinedTextField(
            modifier = Modifier.fillMaxWidth(),
            value = email,
            onValueChange = { email = it },
            label = { Text("Admin Email") },
        )
        Spacer(modifier = Modifier.height(12.dp))
        OutlinedTextField(
            modifier = Modifier.fillMaxWidth(),
            value = password,
            onValueChange = { password = it },
            label = { Text("Password") },
            visualTransformation = PasswordVisualTransformation(),
        )
        Spacer(modifier = Modifier.height(16.dp))
        Button(
            modifier = Modifier.fillMaxWidth(),
            onClick = { onLogin(email, password) },
            enabled = !state.isLoading,
        ) {
            Text(if (state.isLoading) "Signing in..." else "Sign In")
        }
        Spacer(modifier = Modifier.height(12.dp))
        TextButton(onClick = { onSsoRequested("google-oauth2") }) {
            Text("Sign in with Google")
        }
        TextButton(onClick = { onSsoRequested("microsoft-graph") }) {
            Text("Sign in with Microsoft")
        }
        if (state.error != null) {
            Spacer(modifier = Modifier.height(12.dp))
            Text(state.error ?: "", color = MaterialTheme.colorScheme.error)
        }
    }
}

@Composable
private fun MfaScreen(
    state: AuthUiState,
    onVerify: (String, String) -> Unit,
) {
    var code by remember { mutableStateOf("") }
    val method = state.mfaMethods.firstOrNull() ?: "totp"

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("MFA Required", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(12.dp))
        OutlinedTextField(
            modifier = Modifier.fillMaxWidth(),
            value = code,
            onValueChange = { code = it },
            label = { Text("Verification Code") },
        )
        Spacer(modifier = Modifier.height(16.dp))
        Button(
            modifier = Modifier.fillMaxWidth(),
            onClick = { onVerify(method, code) },
            enabled = !state.isLoading,
        ) {
            Text(if (state.isLoading) "Verifying..." else "Verify")
        }
        if (state.error != null) {
            Spacer(modifier = Modifier.height(12.dp))
            Text(state.error ?: "", color = MaterialTheme.colorScheme.error)
        }
    }
}
