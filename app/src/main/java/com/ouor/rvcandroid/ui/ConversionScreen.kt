package com.ouor.rvcandroid.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversionScreen(vm: ConversionViewModel = viewModel()) {
    val state by vm.state.collectAsState()

    val pickModel = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::setModel) }

    val pickInput = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::setInput) }

    val createOutput = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("audio/wav")
    ) { uri: Uri? -> uri?.let(vm::setOutput) }

    Scaffold(
        topBar = { TopAppBar(title = { Text("RVC Android") }) },
        modifier = Modifier.fillMaxSize(),
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            FileRow(
                label = "Model (.onnx)",
                selection = state.model?.displayName,
                onPick = { pickModel.launch(arrayOf("*/*")) },
            )
            FileRow(
                label = "Input audio (.wav)",
                selection = state.input?.displayName,
                onPick = { pickInput.launch(arrayOf("audio/*")) },
            )
            FileRow(
                label = "Output destination",
                selection = state.output?.displayName,
                onPick = { createOutput.launch("rvc-output.wav") },
            )

            Spacer(Modifier.height(8.dp))

            Button(
                onClick = vm::convert,
                enabled = state.model != null &&
                    state.input != null &&
                    state.output != null &&
                    state.stage != Stage.RUNNING,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (state.stage == Stage.RUNNING) "Converting…" else "Convert")
            }

            if (state.stage == Stage.RUNNING || state.progress > 0f) {
                LinearProgressIndicator(
                    progress = { state.progress },
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            state.message?.let { msg ->
                Text(
                    text = msg,
                    color = if (state.stage == Stage.ERROR)
                        MaterialTheme.colorScheme.error
                    else MaterialTheme.colorScheme.primary,
                )
            }
        }
    }
}

@Composable
private fun FileRow(
    label: String,
    selection: String?,
    onPick: () -> Unit,
) {
    Column {
        Text(text = label, style = MaterialTheme.typography.labelLarge)
        Spacer(Modifier.height(4.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(onClick = onPick) {
                Text(if (selection == null) "Select" else "Change")
            }
            Spacer(Modifier.width(12.dp))
            Text(
                text = selection ?: "(not selected)",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}
