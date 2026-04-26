package com.ouor.rvcandroid.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversionScreen(vm: ConversionViewModel = viewModel()) {
    val state by vm.state.collectAsState()

    val pickModel = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::setModel) }
    val pickHubert = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::setHubert) }
    val pickRmvpe = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::setRmvpe) }
    val pickInput = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::setInput) }
    val createOutput = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("audio/wav")
    ) { uri: Uri? -> uri?.let(vm::setOutput) }

    val canConvert = state.model != null &&
        state.hubert != null &&
        state.input != null &&
        state.output != null &&
        state.stage != Stage.RUNNING

    Scaffold(
        topBar = { TopAppBar(title = { Text("RVC Android") }) },
        bottomBar = {
            BottomConvertBar(
                stage = state.stage,
                progress = state.progress,
                message = state.message,
                enabled = canConvert,
                onConvert = vm::convert,
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            ModelsCard(
                model = state.model,
                hubert = state.hubert,
                rmvpe = state.rmvpe,
                onPickModel = { pickModel.launch(arrayOf("*/*")) },
                onPickHubert = { pickHubert.launch(arrayOf("*/*")) },
                onPickRmvpe = { pickRmvpe.launch(arrayOf("*/*")) },
            )

            IoCard(
                input = state.input,
                output = state.output,
                onPickInput = { pickInput.launch(arrayOf("audio/*")) },
                onPickOutput = { createOutput.launch("rvc-output.wav") },
            )

            OptionsCard(
                f0UpKey = state.f0UpKey,
                speakerId = state.speakerId,
                onF0Change = vm::setF0UpKey,
                onSpeakerIdChange = vm::setSpeakerId,
            )
        }
    }
}

@Composable
private fun ModelsCard(
    model: FileSelection?,
    hubert: FileSelection?,
    rmvpe: FileSelection?,
    onPickModel: () -> Unit,
    onPickHubert: () -> Unit,
    onPickRmvpe: () -> Unit,
) {
    // RMVPE is optional at the picker level (only required at runtime for f0
    // models), so the auto-collapse trigger is "core models filled" — the
    // user can still tap the header to re-expand and add RMVPE.
    val coreFilled = model != null && hubert != null
    var expanded by rememberSaveable { mutableStateOf(true) }
    LaunchedEffect(coreFilled) {
        if (coreFilled) expanded = false
    }

    val total = 3
    val filled = listOf(model, hubert, rmvpe).count { it != null }

    Card {
        Column {
            CardHeader(
                title = "Models",
                subtitle = "$filled / $total selected",
                expanded = expanded,
                onToggle = { expanded = !expanded },
            )
            AnimatedVisibility(expanded) {
                Column(
                    modifier = Modifier.padding(start = 16.dp, end = 16.dp, bottom = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    FileRow("Synthesizer (.onnx)", model?.displayName, onPickModel)
                    FileRow("ContentVec / HuBERT (.onnx)", hubert?.displayName, onPickHubert)
                    FileRow("RMVPE (.onnx) — required for f0 models", rmvpe?.displayName, onPickRmvpe)
                }
            }
        }
    }
}

@Composable
private fun IoCard(
    input: FileSelection?,
    output: FileSelection?,
    onPickInput: () -> Unit,
    onPickOutput: () -> Unit,
) {
    Card {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            FileRow("Input audio (.wav)", input?.displayName, onPickInput)
            FileRow("Output destination", output?.displayName, onPickOutput)
        }
    }
}

@Composable
private fun OptionsCard(
    f0UpKey: Int,
    speakerId: Long,
    onF0Change: (Int) -> Unit,
    onSpeakerIdChange: (Long) -> Unit,
) {
    Card {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            PitchShiftRow(value = f0UpKey, onChange = onF0Change)
            SpeakerIdRow(value = speakerId, onChange = onSpeakerIdChange)
        }
    }
}

@Composable
private fun CardHeader(
    title: String,
    subtitle: String,
    expanded: Boolean,
    onToggle: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onToggle)
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text(
            text = if (expanded) "▴" else "▾",
            style = MaterialTheme.typography.titleMedium,
        )
    }
}

@Composable
private fun BottomConvertBar(
    stage: Stage,
    progress: Float,
    message: String?,
    enabled: Boolean,
    onConvert: () -> Unit,
) {
    Surface(tonalElevation = 3.dp) {
        Column(modifier = Modifier.fillMaxWidth()) {
            if (stage == Stage.RUNNING || progress > 0f) {
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            message?.let { msg ->
                Text(
                    text = msg,
                    color = if (stage == Stage.ERROR)
                        MaterialTheme.colorScheme.error
                    else MaterialTheme.colorScheme.primary,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                )
            }
            Button(
                onClick = onConvert,
                enabled = enabled,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
            ) {
                Text(if (stage == Stage.RUNNING) "Converting…" else "Convert")
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

@Composable
private fun PitchShiftRow(value: Int, onChange: (Int) -> Unit) {
    Column {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Pitch shift", style = MaterialTheme.typography.labelLarge)
            Text("$value semitones", style = MaterialTheme.typography.bodyMedium)
        }
        Slider(
            value = value.toFloat(),
            onValueChange = { onChange(it.toInt()) },
            valueRange = -12f..12f,
            steps = 23,
        )
    }
}

@Composable
private fun SpeakerIdRow(value: Long, onChange: (Long) -> Unit) {
    var text by remember { mutableStateOf(value.toString()) }
    OutlinedTextField(
        value = text,
        onValueChange = { txt ->
            text = txt
            if (txt.isEmpty()) onChange(0L) else txt.toLongOrNull()?.let(onChange)
        },
        label = { Text("Speaker ID") },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}
