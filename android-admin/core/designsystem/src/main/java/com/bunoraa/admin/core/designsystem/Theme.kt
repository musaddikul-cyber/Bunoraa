package com.bunoraa.admin.core.designsystem

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColors = lightColorScheme(
    primary = Color(0xFF1B5E20),
    onPrimary = Color.White,
    secondary = Color(0xFF005B9F),
    onSecondary = Color.White,
    background = Color(0xFFF7F7F9),
    surface = Color.White,
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF7EE081),
    onPrimary = Color(0xFF0E1F0E),
    secondary = Color(0xFF8AB4F8),
    onSecondary = Color(0xFF0D1B2A),
    background = Color(0xFF0F1115),
    surface = Color(0xFF151821),
)

@Composable
fun BunoraaAdminTheme(
    darkTheme: Boolean,
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        typography = MaterialTheme.typography,
        content = content,
    )
}
