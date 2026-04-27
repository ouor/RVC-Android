package com.ouor.rvcandroid.ui

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContract
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.IconButton
import androidx.compose.material3.LargeTopAppBar
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberModalBottomSheetState
import com.ouor.rvcandroid.audio.AudioFormat
import com.ouor.rvcandroid.audio.HistoryEntry
import com.ouor.rvcandroid.inference.ModelLoadStatus
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversionScreen(vm: ConversionViewModel = viewModel()) {
    val state by vm.state.collectAsState()
    val ctx = LocalContext.current
    var showPermissionRationale by rememberSaveable { mutableStateOf(false) }
    val recordPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) vm.startRecording() else showPermissionRationale = true
    }
    val launchRecorder: () -> Unit = {
        if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED) {
            vm.startRecording()
        } else {
            recordPermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

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
    val saveAs = rememberLauncherForActivityResult(
        CreateAudioDocument()
    ) { uri: Uri? -> uri?.let(vm::saveCurrentPreview) }

    val blockReason = state.convertBlockReason()
    val canConvert = blockReason == null && state.stage != Stage.RUNNING

    val convertLabel = when (state.stage) {
        Stage.RUNNING -> "Converting…"
        else -> "Convert"
    }

    val scrollBehavior = TopAppBarDefaults.exitUntilCollapsedScrollBehavior()

    Scaffold(
        modifier = Modifier.nestedScroll(scrollBehavior.nestedScrollConnection),
        topBar = {
            LargeTopAppBar(
                title = { Text("RVC Android") },
                scrollBehavior = scrollBehavior,
            )
        },
        bottomBar = {
            BottomConvertBar(
                stage = state.stage,
                runningStep = state.runningStep,
                message = state.message,
                enabled = canConvert,
                buttonLabel = convertLabel,
                blockReason = blockReason,
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
                synthStatus = state.synthStatus,
                hubertStatus = state.hubertStatus,
                rmvpeStatus = state.rmvpeStatus,
                allLoaded = state.allRequiredModelsLoaded,
                onPickModel = { pickModel.launch(arrayOf("*/*")) },
                onPickHubert = { pickHubert.launch(arrayOf("*/*")) },
                onPickRmvpe = { pickRmvpe.launch(arrayOf("*/*")) },
            )

            IoCard(
                input = state.input,
                inputError = state.inputError,
                inputWaveform = state.inputWaveform,
                onPickInput = { pickInput.launch(arrayOf("audio/*")) },
                onRecord = launchRecorder,
            )

            OptionsCard(
                f0UpKey = state.f0UpKey,
                speakerId = state.speakerId,
                outputFormat = state.outputFormat,
                onF0Change = vm::setF0UpKey,
                onSpeakerIdChange = vm::setSpeakerId,
                onOutputFormatChange = vm::setOutputFormat,
            )

            if (state.history.isNotEmpty()) {
                HistoryCard(
                    entries = state.history,
                    activeFile = state.preview.file,
                    onPlay = vm::openHistoryEntry,
                    onSaveAs = { entry ->
                        vm.loadHistoryEntry(entry)
                        saveAs.launch(
                            CreateAudioDocumentRequest(
                                mime = entry.format.mime,
                                filename = entry.file.name,
                            )
                        )
                    },
                    onDelete = vm::deleteHistoryEntry,
                )
            }
        }
    }

    val recording = state.recording
    if (recording is RecordingState.Active) {
        RecordingSheet(
            elapsedMs = recording.elapsedMs,
            amplitude = recording.amplitude,
            onStop = vm::stopRecording,
            onCancel = vm::cancelRecording,
        )
    }

    if (state.showResultSheet && state.preview.file != null) {
        ResultSheet(
            preview = state.preview,
            elapsedMs = state.elapsedMs,
            onDismiss = vm::dismissResultSheet,
            onTogglePlay = vm::togglePlay,
            onSeek = vm::seekTo,
            onSaveAs = {
                val format = state.preview.format ?: state.outputFormat
                saveAs.launch(
                    CreateAudioDocumentRequest(
                        mime = format.mime,
                        filename = "rvc-output.${format.ext}",
                    )
                )
            },
        )
    }

    if (showPermissionRationale) {
        PermissionRationaleDialog(
            onDismiss = { showPermissionRationale = false },
            onOpenSettings = {
                showPermissionRationale = false
                ctx.startActivity(
                    Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                        data = Uri.fromParts("package", ctx.packageName, null)
                    }
                )
            },
        )
    }
}

@Composable
private fun PermissionRationaleDialog(onDismiss: () -> Unit, onOpenSettings: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Microphone access") },
        text = {
            Text(
                "RVC Android needs the microphone permission to record input audio. " +
                    "Enable it in system settings to use the in-app recorder."
            )
        },
        confirmButton = { TextButton(onClick = onOpenSettings) { Text("Open settings") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun RecordingSheet(
    elapsedMs: Long,
    amplitude: Float,
    onStop: () -> Unit,
    onCancel: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        onDismissRequest = onCancel,
        sheetState = sheetState,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Recording", style = MaterialTheme.typography.titleMedium)
            Text(
                text = formatElapsed(elapsedMs),
                style = MaterialTheme.typography.headlineMedium,
            )
            // Amplitude lives in [0..1]; LinearProgressIndicator maps that
            // to a level meter without dragging in another graph library.
            LinearProgressIndicator(
                progress = { amplitude.coerceIn(0f, 1f) },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp),
            )
            // Same 60-second cap that gates picked input.
            LinearProgressIndicator(
                progress = { (elapsedMs / 60_000f).coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
                color = MaterialTheme.colorScheme.tertiary,
            )
            Text(
                text = "Auto-stop at 60s",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                OutlinedButton(
                    onClick = onCancel,
                    modifier = Modifier.weight(1f),
                ) { Text("Cancel") }
                Button(
                    onClick = onStop,
                    modifier = Modifier.weight(1f),
                ) { Text("Stop") }
            }
        }
    }
}

private fun formatElapsed(ms: Long): String {
    val s = ms / 1000
    val mm = s / 60
    val ss = s % 60
    val tenths = (ms % 1000) / 100
    return "%d:%02d.%d".format(mm, ss, tenths)
}

@Composable
private fun ModelsCard(
    model: FileSelection?,
    hubert: FileSelection?,
    rmvpe: FileSelection?,
    synthStatus: ModelLoadStatus,
    hubertStatus: ModelLoadStatus,
    rmvpeStatus: ModelLoadStatus,
    allLoaded: Boolean,
    onPickModel: () -> Unit,
    onPickHubert: () -> Unit,
    onPickRmvpe: () -> Unit,
) {
    // Collapse only after every required model has finished loading — that
    // way the user can watch the load spinners advance without the card
    // disappearing the moment they tap a file.
    var expanded by rememberSaveable { mutableStateOf(true) }
    LaunchedEffect(allLoaded) {
        if (allLoaded) expanded = false
    }

    val subtitle = subtitleFor(synthStatus, hubertStatus, rmvpeStatus)

    ElevatedCard(elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)) {
        Column {
            CardHeader(
                title = "Models",
                subtitle = subtitle,
                expanded = expanded,
                onToggle = { expanded = !expanded },
            )
            AnimatedVisibility(expanded) {
                Column(
                    modifier = Modifier.padding(start = 16.dp, end = 16.dp, bottom = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    ModelRow("Synthesizer (.onnx)", model?.displayName, synthStatus, onPickModel)
                    ModelRow("ContentVec / HuBERT (.onnx)", hubert?.displayName, hubertStatus, onPickHubert)
                    ModelRow("RMVPE (.onnx)", rmvpe?.displayName, rmvpeStatus, onPickRmvpe)
                }
            }
        }
    }
}

private fun subtitleFor(
    synth: ModelLoadStatus,
    hubert: ModelLoadStatus,
    rmvpe: ModelLoadStatus,
): String {
    val statuses = listOf(synth, hubert, rmvpe)
    val total = statuses.size
    val loaded = statuses.count { it is ModelLoadStatus.Loaded }
    val loading = statuses.count { it is ModelLoadStatus.Loading }
    val failed = statuses.count { it is ModelLoadStatus.Failed }
    return when {
        failed > 0 -> "$failed failed · $loaded / $total ready"
        loading > 0 -> "Loading… $loaded / $total ready"
        loaded == total -> "Ready · ${summaryFor(synth)}"
        else -> "$loaded / $total ready"
    }
}

private fun summaryFor(status: ModelLoadStatus): String {
    val s = (status as? ModelLoadStatus.Loaded)?.summary ?: return ""
    val rate = if (s.sampleRate >= 1000) "${s.sampleRate / 1000}kHz" else "${s.sampleRate}Hz"
    val pitch = if (s.f0) "f0" else "no-f0"
    return "$rate · $pitch · ${s.embChannels}d"
}

@Composable
private fun IoCard(
    input: FileSelection?,
    inputError: String?,
    inputWaveform: FloatArray?,
    onPickInput: () -> Unit,
    onRecord: () -> Unit,
) {
    ElevatedCard(elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            InputRow(
                input = input,
                error = inputError,
                waveform = inputWaveform,
                onPick = onPickInput,
                onRecord = onRecord,
            )
        }
    }
}

@Composable
private fun InputRow(
    input: FileSelection?,
    error: String?,
    waveform: FloatArray?,
    onPick: () -> Unit,
    onRecord: () -> Unit,
) {
    Column {
        InputPickRow(input = input, onPick = onPick, onRecord = onRecord)
        input?.meta?.let { meta ->
            Text(
                text = formatMeta(meta),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
        if (input != null && waveform != null && waveform.isNotEmpty()) {
            WaveformThumbnail(
                buckets = waveform,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(44.dp)
                    .padding(top = 6.dp),
            )
        }
        if (error != null) {
            Text(
                text = error,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
    }
}

@Composable
private fun InputPickRow(
    input: FileSelection?,
    onPick: () -> Unit,
    onRecord: () -> Unit,
) {
    Column {
        Text("Input audio", style = MaterialTheme.typography.labelLarge)
        Spacer(Modifier.height(6.dp))
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            FilledTonalButton(onClick = onPick) {
                Text(if (input == null) "Select" else "Change")
            }
            // Mic acts as an alternate "select" — picking a file or recording
            // both end up in the same state slot, so we don't show two
            // selection chips.
            IconButton(onClick = onRecord) {
                Text("●", color = MaterialTheme.colorScheme.error)
            }
            FileStatusChip(
                selection = input?.displayName,
                onClick = onPick,
                modifier = Modifier.weight(1f),
            )
        }
    }
}

@Composable
private fun WaveformThumbnail(buckets: FloatArray, modifier: Modifier = Modifier) {
    val color = MaterialTheme.colorScheme.primary
    Canvas(modifier = modifier) {
        if (buckets.isEmpty()) return@Canvas
        val bw = size.width / buckets.size
        val barWidth = (bw * 0.7f).coerceAtLeast(1f)
        val centerY = size.height / 2f
        val maxBar = size.height
        buckets.forEachIndexed { i, amp ->
            // RMS thumbnails on quiet audio look like a flat line; lift the
            // floor so silence still has a visible 1 px tick.
            val h = (amp.coerceIn(0f, 1f) * maxBar).coerceAtLeast(1f)
            drawLine(
                color = color,
                start = Offset(i * bw + bw / 2f, centerY - h / 2f),
                end = Offset(i * bw + bw / 2f, centerY + h / 2f),
                strokeWidth = barWidth,
            )
        }
    }
}

private fun formatMeta(meta: com.ouor.rvcandroid.audio.AudioMeta): String {
    val seconds = meta.durationMs / 1000
    val mm = seconds / 60
    val ss = seconds % 60
    val rate = if (meta.sampleRate >= 1000) "${meta.sampleRate / 1000f}kHz".replace(".0kHz", "kHz")
    else "${meta.sampleRate}Hz"
    val ch = when (meta.channels) {
        1 -> "Mono"
        2 -> "Stereo"
        else -> "${meta.channels}ch"
    }
    return "%d:%02d · %s · %s".format(mm, ss, rate, ch)
}

@Composable
private fun OptionsCard(
    f0UpKey: Int,
    speakerId: Long,
    outputFormat: AudioFormat,
    onF0Change: (Int) -> Unit,
    onSpeakerIdChange: (Long) -> Unit,
    onOutputFormatChange: (AudioFormat) -> Unit,
) {
    // A non-default speaker id or non-MP3 format is meaningful, so force
    // the section open in that case — otherwise the user could "lose" the
    // value behind the toggle.
    var manualOpen by rememberSaveable { mutableStateOf(false) }
    val nonDefaultAdvanced = speakerId != 0L || outputFormat != AudioFormat.MP3
    val advancedOpen = manualOpen || nonDefaultAdvanced

    ElevatedCard(elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            PitchShiftRow(value = f0UpKey, onChange = onF0Change)

            AdvancedToggle(
                open = advancedOpen,
                forced = nonDefaultAdvanced,
                onToggle = { manualOpen = !manualOpen },
            )
            AnimatedVisibility(advancedOpen) {
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    OutputFormatRow(value = outputFormat, onChange = onOutputFormatChange)
                    SpeakerIdRow(value = speakerId, onChange = onSpeakerIdChange)
                }
            }
        }
    }
}

@Composable
private fun OutputFormatRow(value: AudioFormat, onChange: (AudioFormat) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    Column {
        Text("Output format", style = MaterialTheme.typography.labelLarge)
        Spacer(Modifier.height(6.dp))
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            FilledTonalButton(onClick = { expanded = true }) {
                Text("${value.displayName} (.${value.ext}) ▾")
            }
            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false },
            ) {
                AudioFormat.entries.forEach { fmt ->
                    DropdownMenuItem(
                        text = { Text("${fmt.displayName} (.${fmt.ext})") },
                        onClick = {
                            onChange(fmt)
                            expanded = false
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun AdvancedToggle(open: Boolean, forced: Boolean, onToggle: () -> Unit) {
    val label = when {
        forced -> "Advanced"
        open -> "Hide advanced ▴"
        else -> "Show advanced ▾"
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .let { if (forced) it else it.clickable(onClick = onToggle) },
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
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
    runningStep: Step?,
    message: String?,
    enabled: Boolean,
    buttonLabel: String,
    blockReason: String?,
    onConvert: () -> Unit,
) {
    Surface(tonalElevation = 3.dp) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.navigationBars),
        ) {
            when {
                stage == Stage.RUNNING && runningStep != null -> {
                    StepRow(
                        current = runningStep,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                    )
                }
                stage == Stage.ERROR -> ErrorBanner(message)
                blockReason != null -> BlockReasonBanner(blockReason)
                else -> {}
            }
            Button(
                onClick = onConvert,
                enabled = enabled,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
            ) {
                Text(buttonLabel)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ResultSheet(
    preview: PreviewState,
    elapsedMs: Long?,
    onDismiss: () -> Unit,
    onTogglePlay: () -> Unit,
    onSeek: (Long) -> Unit,
    onSaveAs: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 8.dp)
                .padding(bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "✓",
                    color = MaterialTheme.colorScheme.primary,
                    style = MaterialTheme.typography.titleLarge,
                )
                Column(modifier = Modifier.weight(1f)) {
                    val title = if (elapsedMs != null) "Conversion complete" else "Saved conversion"
                    Text(title, style = MaterialTheme.typography.titleMedium)
                    val subtitle = buildString {
                        if (elapsedMs != null) append("Took ${formatDuration(elapsedMs)}")
                        preview.format?.let {
                            if (isNotEmpty()) append(" · ")
                            append(it.displayName)
                        }
                    }
                    if (subtitle.isNotEmpty()) {
                        Text(
                            text = subtitle,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
            PlayerControls(
                preview = preview,
                onTogglePlay = onTogglePlay,
                onSeek = onSeek,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                OutlinedButton(
                    onClick = onDismiss,
                    modifier = Modifier.weight(1f),
                ) { Text("Close") }
                Button(
                    onClick = onSaveAs,
                    modifier = Modifier.weight(1f),
                ) {
                    val label = preview.format?.let { "Save as ${it.displayName}" } ?: "Save as…"
                    Text(label)
                }
            }
        }
    }
}

@Composable
private fun PlayerControls(
    preview: PreviewState,
    onTogglePlay: () -> Unit,
    onSeek: (Long) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        FilledTonalButton(onClick = onTogglePlay) {
            Text(if (preview.isPlaying) "Pause" else "Play")
        }
        Column(modifier = Modifier.weight(1f)) {
            val duration = preview.durationMs.coerceAtLeast(1L)
            Slider(
                value = preview.positionMs.toFloat().coerceIn(0f, duration.toFloat()),
                onValueChange = { onSeek(it.toLong()) },
                valueRange = 0f..duration.toFloat(),
            )
            Text(
                text = "${formatTime(preview.positionMs)} / ${formatTime(preview.durationMs)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

private fun formatTime(ms: Long): String {
    val s = ms / 1000
    return "%d:%02d".format(s / 60, s % 60)
}

@Composable
private fun BlockReasonBanner(reason: String) {
    Text(
        text = reason,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
    )
}

@Composable
private fun StepRow(current: Step, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Step.entries.forEach { step ->
            val state = when {
                step == current -> StepUiState.Active
                step.ordinal < current.ordinal -> StepUiState.Done
                else -> StepUiState.Pending
            }
            StepChip(
                step = step,
                state = state,
                modifier = Modifier.weight(1f),
            )
        }
    }
}

private enum class StepUiState { Done, Active, Pending }

@Composable
private fun StepChip(step: Step, state: StepUiState, modifier: Modifier = Modifier) {
    val cs = MaterialTheme.colorScheme
    val (bg, fg) = when (state) {
        StepUiState.Done -> cs.primary to cs.onPrimary
        StepUiState.Active -> cs.primaryContainer to cs.onPrimaryContainer
        StepUiState.Pending -> cs.surfaceVariant to cs.onSurfaceVariant
    }
    Surface(
        color = bg,
        shape = MaterialTheme.shapes.small,
        modifier = modifier,
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp, Alignment.CenterHorizontally),
        ) {
            when (state) {
                StepUiState.Done -> Text(
                    text = "✓",
                    color = fg,
                    style = MaterialTheme.typography.labelMedium,
                )
                StepUiState.Active -> CircularProgressIndicator(
                    modifier = Modifier.size(12.dp),
                    strokeWidth = 1.5.dp,
                    color = fg,
                )
                StepUiState.Pending -> Spacer(Modifier.size(12.dp))
            }
            Text(
                text = step.label,
                color = fg,
                style = MaterialTheme.typography.labelMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun HistoryCard(
    entries: List<HistoryEntry>,
    activeFile: java.io.File?,
    onPlay: (HistoryEntry) -> Unit,
    onSaveAs: (HistoryEntry) -> Unit,
    onDelete: (HistoryEntry) -> Unit,
) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    ElevatedCard(elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)) {
        Column {
            CardHeader(
                title = "Recent conversions",
                subtitle = "${entries.size} saved · LRU 10",
                expanded = expanded,
                onToggle = { expanded = !expanded },
            )
            AnimatedVisibility(expanded) {
                Column(
                    modifier = Modifier.padding(start = 16.dp, end = 16.dp, bottom = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    entries.forEach { entry ->
                        HistoryRow(
                            entry = entry,
                            isActive = entry.file == activeFile,
                            onPlay = { onPlay(entry) },
                            onSaveAs = { onSaveAs(entry) },
                            onDelete = { onDelete(entry) },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun HistoryRow(
    entry: HistoryEntry,
    isActive: Boolean,
    onPlay: () -> Unit,
    onSaveAs: () -> Unit,
    onDelete: () -> Unit,
) {
    val cs = MaterialTheme.colorScheme
    Surface(
        color = if (isActive) cs.secondaryContainer else cs.surfaceVariant,
        shape = MaterialTheme.shapes.small,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            Text(
                text = entry.file.name,
                style = MaterialTheme.typography.bodyMedium,
                color = if (isActive) cs.onSecondaryContainer else cs.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                text = "${entry.format.displayName} · ${formatSize(entry.sizeBytes)} · ${formatRelative(entry.createdAt)}",
                style = MaterialTheme.typography.labelSmall,
                color = if (isActive) cs.onSecondaryContainer else cs.onSurfaceVariant,
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FilledTonalButton(onClick = onPlay) { Text("Open") }
                OutlinedButton(onClick = onSaveAs) { Text("Save as") }
                Spacer(Modifier.weight(1f))
                IconButton(onClick = onDelete) {
                    Text("✕", color = cs.error, style = MaterialTheme.typography.labelLarge)
                }
            }
        }
    }
}

private fun formatSize(bytes: Long): String = when {
    bytes >= 1_000_000L -> "%.1f MB".format(bytes / 1_000_000f)
    bytes >= 1_000L -> "%d KB".format(bytes / 1_000L)
    else -> "$bytes B"
}

private fun formatRelative(epochMs: Long): String {
    val deltaSec = (System.currentTimeMillis() - epochMs) / 1000
    return when {
        deltaSec < 60 -> "${deltaSec}s ago"
        deltaSec < 3600 -> "${deltaSec / 60}m ago"
        deltaSec < 86_400 -> "${deltaSec / 3600}h ago"
        else -> "${deltaSec / 86_400}d ago"
    }
}

@Composable
private fun ErrorBanner(message: String?) {
    if (message == null) return
    Text(
        text = message,
        color = MaterialTheme.colorScheme.error,
        style = MaterialTheme.typography.bodyMedium,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 10.dp),
    )
}

private fun formatDuration(ms: Long): String =
    if (ms < 60_000L) String.format(java.util.Locale.US, "%.1fs", ms / 1000f)
    else "${ms / 60_000L}m ${(ms % 60_000L) / 1000L}s"

@Composable
private fun FileRow(
    label: String,
    selection: String?,
    onPick: () -> Unit,
) {
    Column {
        Text(text = label, style = MaterialTheme.typography.labelLarge)
        Spacer(Modifier.height(6.dp))
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            FilledTonalButton(onClick = onPick) {
                Text(if (selection == null) "Select" else "Change")
            }
            FileStatusChip(
                selection = selection,
                onClick = onPick,
                modifier = Modifier.weight(1f),
            )
        }
    }
}

@Composable
private fun ModelRow(
    label: String,
    selection: String?,
    status: ModelLoadStatus,
    onPick: () -> Unit,
) {
    Column {
        Text(text = label, style = MaterialTheme.typography.labelLarge)
        Spacer(Modifier.height(6.dp))
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            FilledTonalButton(onClick = onPick) {
                Text(if (selection == null) "Select" else "Change")
            }
            ModelStatusChip(
                selection = selection,
                status = status,
                onClick = onPick,
                modifier = Modifier.weight(1f),
            )
        }
        // Failed loads need a full-width line for the error message; the chip
        // alone truncates it past a few words.
        if (status is ModelLoadStatus.Failed) {
            Text(
                text = status.error,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
    }
}

@Composable
private fun ModelStatusChip(
    selection: String?,
    status: ModelLoadStatus,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val cs = MaterialTheme.colorScheme
    val (containerColor, contentColor) = when (status) {
        is ModelLoadStatus.Loaded -> cs.secondaryContainer to cs.onSecondaryContainer
        is ModelLoadStatus.Failed -> cs.errorContainer to cs.onErrorContainer
        else -> cs.surfaceVariant to cs.onSurfaceVariant
    }
    AssistChip(
        onClick = onClick,
        modifier = modifier,
        label = {
            Text(
                text = selection ?: "Not selected",
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        },
        leadingIcon = {
            when (status) {
                ModelLoadStatus.Empty -> {}
                ModelLoadStatus.Loading -> CircularProgressIndicator(
                    modifier = Modifier.size(14.dp),
                    strokeWidth = 1.5.dp,
                    color = contentColor,
                )
                is ModelLoadStatus.Loaded -> Text(
                    text = "✓",
                    color = contentColor,
                    style = MaterialTheme.typography.labelLarge,
                )
                is ModelLoadStatus.Failed -> Text(
                    text = "✕",
                    color = contentColor,
                    style = MaterialTheme.typography.labelLarge,
                )
            }
        },
        colors = AssistChipDefaults.assistChipColors(
            containerColor = containerColor,
            labelColor = contentColor,
            leadingIconContentColor = contentColor,
        ),
    )
}

@Composable
private fun FileStatusChip(
    selection: String?,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val filled = selection != null
    AssistChip(
        onClick = onClick,
        modifier = modifier,
        label = {
            Text(
                text = selection ?: "Not selected",
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        },
        leadingIcon = if (filled) {
            { Text("✓", style = MaterialTheme.typography.labelLarge) }
        } else null,
        colors = if (filled) {
            AssistChipDefaults.assistChipColors(
                containerColor = MaterialTheme.colorScheme.secondaryContainer,
                labelColor = MaterialTheme.colorScheme.onSecondaryContainer,
                leadingIconContentColor = MaterialTheme.colorScheme.onSecondaryContainer,
            )
        } else AssistChipDefaults.assistChipColors(),
    )
}

@Composable
private fun PitchShiftRow(value: Int, onChange: (Int) -> Unit) {
    val signed = if (value > 0) "+$value" else value.toString()
    Column {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Pitch shift", style = MaterialTheme.typography.labelLarge)
            Surface(
                color = MaterialTheme.colorScheme.secondaryContainer,
                shape = MaterialTheme.shapes.small,
            ) {
                Text(
                    text = "$signed semitones",
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                    style = MaterialTheme.typography.labelMedium,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                )
            }
        }
        Slider(
            value = value.toFloat(),
            onValueChange = { onChange(it.toInt()) },
            valueRange = -12f..12f,
            steps = 23,
        )
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 4.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            val tickStyle = MaterialTheme.typography.labelSmall
            val tickColor = MaterialTheme.colorScheme.onSurfaceVariant
            Text("−12", style = tickStyle, color = tickColor)
            Text("0", style = tickStyle, color = tickColor)
            Text("+12", style = tickStyle, color = tickColor)
        }
    }
}

/**
 * SAF CreateDocument with a per-launch MIME + filename. The standard
 * [ActivityResultContracts.CreateDocument] bakes the MIME at construction,
 * which can't react to a Compose state change like the output-format
 * dropdown — this contract reads both at launch() time.
 */
private data class CreateAudioDocumentRequest(val mime: String, val filename: String)

private class CreateAudioDocument :
    ActivityResultContract<CreateAudioDocumentRequest, Uri?>() {
    override fun createIntent(context: Context, input: CreateAudioDocumentRequest): Intent =
        Intent(Intent.ACTION_CREATE_DOCUMENT)
            .addCategory(Intent.CATEGORY_OPENABLE)
            .setType(input.mime)
            .putExtra(Intent.EXTRA_TITLE, input.filename)

    override fun parseResult(resultCode: Int, intent: Intent?): Uri? =
        intent.takeIf { resultCode == android.app.Activity.RESULT_OK }?.data
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
