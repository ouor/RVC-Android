package com.ouor.rvcandroid.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContract
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.material3.LargeTopAppBar
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import com.ouor.rvcandroid.audio.AudioFormat
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
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
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
        CreateAudioDocument()
    ) { uri: Uri? -> uri?.let(vm::setOutput) }

    val blockReason = state.convertBlockReason()
    val canConvert = blockReason == null && state.stage != Stage.RUNNING

    val convertLabel = when {
        state.stage == Stage.RUNNING -> "Converting…"
        state.stage == Stage.DONE -> "Convert again"
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
                outputName = state.output?.displayName,
                elapsedMs = state.elapsedMs,
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
                requiresRmvpe = state.requiresRmvpe,
                allLoaded = state.allRequiredModelsLoaded,
                onPickModel = { pickModel.launch(arrayOf("*/*")) },
                onPickHubert = { pickHubert.launch(arrayOf("*/*")) },
                onPickRmvpe = { pickRmvpe.launch(arrayOf("*/*")) },
            )

            IoCard(
                input = state.input,
                output = state.output,
                outputFormat = state.outputFormat,
                inputError = state.inputError,
                onPickInput = { pickInput.launch(arrayOf("audio/*")) },
                onPickOutput = {
                    createOutput.launch(
                        CreateAudioDocumentRequest(
                            mime = state.outputFormat.mime,
                            filename = "rvc-output.${state.outputFormat.ext}",
                        )
                    )
                },
            )

            OptionsCard(
                f0UpKey = state.f0UpKey,
                speakerId = state.speakerId,
                outputFormat = state.outputFormat,
                onF0Change = vm::setF0UpKey,
                onSpeakerIdChange = vm::setSpeakerId,
                onOutputFormatChange = vm::setOutputFormat,
            )
        }
    }
}

@Composable
private fun ModelsCard(
    model: FileSelection?,
    hubert: FileSelection?,
    rmvpe: FileSelection?,
    synthStatus: ModelLoadStatus,
    hubertStatus: ModelLoadStatus,
    rmvpeStatus: ModelLoadStatus,
    requiresRmvpe: Boolean,
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

    val rmvpeLabel = if (requiresRmvpe) "RMVPE (.onnx) — required" else "RMVPE (.onnx) — optional"
    val subtitle = subtitleFor(synthStatus, hubertStatus, rmvpeStatus, requiresRmvpe)

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
                    ModelRow(rmvpeLabel, rmvpe?.displayName, rmvpeStatus, onPickRmvpe)
                }
            }
        }
    }
}

private fun subtitleFor(
    synth: ModelLoadStatus,
    hubert: ModelLoadStatus,
    rmvpe: ModelLoadStatus,
    requiresRmvpe: Boolean,
): String {
    val statuses = buildList {
        add(synth)
        add(hubert)
        if (requiresRmvpe) add(rmvpe)
    }
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
    output: FileSelection?,
    outputFormat: AudioFormat,
    inputError: String?,
    onPickInput: () -> Unit,
    onPickOutput: () -> Unit,
) {
    ElevatedCard(elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            InputRow(input = input, error = inputError, onPick = onPickInput)
            FileRow("Output (${outputFormat.displayName})", output?.displayName, onPickOutput)
        }
    }
}

@Composable
private fun InputRow(input: FileSelection?, error: String?, onPick: () -> Unit) {
    Column {
        FileRow("Input audio", input?.displayName, onPick)
        input?.meta?.let { meta ->
            Text(
                text = formatMeta(meta),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp),
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
    // A non-default speaker id is meaningful, so force the section open in
    // that case — otherwise the user could "lose" the value behind the toggle.
    var manualOpen by rememberSaveable { mutableStateOf(false) }
    val advancedOpen = manualOpen || speakerId != 0L

    ElevatedCard(elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            PitchShiftRow(value = f0UpKey, onChange = onF0Change)

            OutputFormatRow(value = outputFormat, onChange = onOutputFormatChange)

            AdvancedToggle(
                open = advancedOpen,
                forced = speakerId != 0L,
                onToggle = { manualOpen = !manualOpen },
            )
            AnimatedVisibility(advancedOpen) {
                SpeakerIdRow(value = speakerId, onChange = onSpeakerIdChange)
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
    outputName: String?,
    elapsedMs: Long?,
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
                stage == Stage.DONE -> ResultBanner(outputName, elapsedMs)
                stage == Stage.ERROR -> ErrorBanner(message)
                // Tell the user what's missing instead of just dimming the
                // button — most "why is this disabled?" moments come from
                // the model loads still running in the background.
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
private fun ResultBanner(outputName: String?, elapsedMs: Long?) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(
            text = "✓",
            color = MaterialTheme.colorScheme.primary,
            style = MaterialTheme.typography.titleMedium,
        )
        Column(modifier = Modifier.weight(1f)) {
            val headline = buildString {
                append("Saved")
                if (elapsedMs != null) append(" in ${formatDuration(elapsedMs)}")
            }
            Text(
                text = headline,
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.bodyMedium,
            )
            outputName?.let {
                Text(
                    text = it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
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
