package com.ouor.rvcandroid

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.ouor.rvcandroid.ui.ConversionScreen
import com.ouor.rvcandroid.ui.theme.RvcTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            RvcTheme {
                ConversionScreen()
            }
        }
    }
}
