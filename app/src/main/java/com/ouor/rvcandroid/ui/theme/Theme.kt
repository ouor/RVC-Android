package com.ouor.rvcandroid.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

// minSdk is 31, so dynamic colors (API 31+) are always available — no
// fallback color scheme needed.
@Composable
fun RvcTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val context = LocalContext.current
    val colors = if (darkTheme) dynamicDarkColorScheme(context)
    else dynamicLightColorScheme(context)
    MaterialTheme(colorScheme = colors, content = content)
}
