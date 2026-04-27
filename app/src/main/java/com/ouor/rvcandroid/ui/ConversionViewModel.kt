package com.ouor.rvcandroid.ui

import ai.onnxruntime.OrtSession
import android.app.Application
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ouor.rvcandroid.audio.AudioFormat
import com.ouor.rvcandroid.audio.AudioIo
import com.ouor.rvcandroid.audio.AudioMeta
import com.ouor.rvcandroid.audio.AudioMetaProbe
import com.ouor.rvcandroid.audio.HistoryCache
import com.ouor.rvcandroid.audio.HistoryEntry
import com.ouor.rvcandroid.audio.PcmRecorder
import com.ouor.rvcandroid.audio.PreviewPlayer
import com.ouor.rvcandroid.audio.Resampler
import com.ouor.rvcandroid.audio.Waveform
import androidx.core.net.toUri
import androidx.media3.common.Player
import java.io.File
import com.ouor.rvcandroid.inference.ModelLoadStatus
import com.ouor.rvcandroid.inference.ModelMetadata
import com.ouor.rvcandroid.inference.ModelSummary
import com.ouor.rvcandroid.inference.OrtRuntime
import com.ouor.rvcandroid.inference.RvcPipeline
import com.ouor.rvcandroid.inference.RvcPipelineFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "Rvc.Conv"
private const val HUBERT_SAMPLE_RATE = 16000
private const val MAX_INPUT_DURATION_MS = 60_000L
private const val RECORDING_AMPLITUDE_POLL_MS = 50L

data class FileSelection(val uri: Uri, val displayName: String, val meta: AudioMeta? = null)

enum class Stage { IDLE, RUNNING, DONE, ERROR }

enum class Step(val label: String) {
    LOADING("Load"),
    READING("Read"),
    CONVERTING("Convert"),
    WRITING("Write"),
}

private enum class Slot { SYNTH, HUBERT, RMVPE }

sealed class RecordingState {
    object Idle : RecordingState()
    data class Active(val elapsedMs: Long, val amplitude: Float) : RecordingState()
}

/**
 * Local-cache preview of the most recently converted clip. The file lives
 * inside [HistoryCache] (so the LRU keeps it alive) and the player polls
 * its position into [positionMs] for the UI scrubber.
 */
data class PreviewState(
    val file: File? = null,
    val format: AudioFormat? = null,
    val isPlaying: Boolean = false,
    val positionMs: Long = 0L,
    val durationMs: Long = 0L,
)

data class ConversionUiState(
    val model: FileSelection? = null,
    val hubert: FileSelection? = null,
    val rmvpe: FileSelection? = null,
    val input: FileSelection? = null,
    val outputFormat: AudioFormat = AudioFormat.WAV,
    val inputError: String? = null,
    val inputWaveform: FloatArray? = null,
    val recording: RecordingState = RecordingState.Idle,
    val f0UpKey: Int = 0,
    val speakerId: Long = 0L,
    val synthStatus: ModelLoadStatus = ModelLoadStatus.Empty,
    val hubertStatus: ModelLoadStatus = ModelLoadStatus.Empty,
    val rmvpeStatus: ModelLoadStatus = ModelLoadStatus.Empty,
    val stage: Stage = Stage.IDLE,
    val runningStep: Step? = null,
    val elapsedMs: Long? = null,
    val message: String? = null,
    val preview: PreviewState = PreviewState(),
    val history: List<HistoryEntry> = emptyList(),
    val showResultSheet: Boolean = false,
) {
    /**
     * Whether RMVPE is mandatory. Determined by the synthesizer's metadata
     * once it has loaded; until then we don't yet know whether the user can
     * skip RMVPE selection.
     */
    val requiresRmvpe: Boolean
        get() = (synthStatus as? ModelLoadStatus.Loaded)?.summary?.f0 == true

    val allRequiredModelsLoaded: Boolean
        get() {
            if (synthStatus !is ModelLoadStatus.Loaded) return false
            if (hubertStatus !is ModelLoadStatus.Loaded) return false
            if (requiresRmvpe && rmvpeStatus !is ModelLoadStatus.Loaded) return false
            return true
        }

    /**
     * Why the convert button is disabled, surfaced to the user as a hint.
     * Returns null when convert is allowed.
     */
    fun convertBlockReason(): String? {
        if (stage == Stage.RUNNING) return null
        if (model == null) return "Pick a synthesizer model"
        if (hubert == null) return "Pick a HuBERT / ContentVec model"
        if (synthStatus is ModelLoadStatus.Failed) return "Synthesizer failed: ${synthStatus.error}"
        if (hubertStatus is ModelLoadStatus.Failed) return "HuBERT failed: ${hubertStatus.error}"
        if (rmvpeStatus is ModelLoadStatus.Failed) return "RMVPE failed: ${rmvpeStatus.error}"
        if (synthStatus is ModelLoadStatus.Loading) return "Loading synthesizer…"
        if (hubertStatus is ModelLoadStatus.Loading) return "Loading HuBERT…"
        if (synthStatus is ModelLoadStatus.Empty) return "Waiting for synthesizer"
        if (hubertStatus is ModelLoadStatus.Empty) return "Waiting for HuBERT"
        if (requiresRmvpe) {
            if (rmvpe == null) return "f0 model needs RMVPE"
            if (rmvpeStatus is ModelLoadStatus.Loading) return "Loading RMVPE…"
            if (rmvpeStatus is ModelLoadStatus.Empty) return "Waiting for RMVPE"
        }
        if (input == null) return "Pick or record an input audio"
        return null
    }
}

class ConversionViewModel(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(ConversionUiState())
    val state = _state.asStateFlow()

    /**
     * ORT sessions kept warm so a pipeline assembly during convert is just
     * wrapper construction, not another 5-second model open.
     */
    private val sessions = mutableMapOf<Slot, OrtSession>()
    private val sessionUris = mutableMapOf<Slot, Uri>()
    private val loadJobs = mutableMapOf<Slot, Job>()
    private var synthMetadata: ModelMetadata? = null

    private var pipeline: RvcPipeline? = null
    private var pipelineKey: List<Uri?>? = null
    private var convertJob: Job? = null

    private var recorder: PcmRecorder? = null
    private var recorderJob: Job? = null
    private var recorderFile: File? = null

    private val player: PreviewPlayer by lazy {
        val p = PreviewPlayer(getApplication())
        p.addListener(playerListener)
        p
    }
    private var playerPollJob: Job? = null
    private val playerListener = object : Player.Listener {
        override fun onIsPlayingChanged(isPlaying: Boolean) {
            _state.update { it.copy(preview = it.preview.copy(isPlaying = isPlaying)) }
            if (isPlaying) startPlayerPoll() else playerPollJob?.cancel()
        }
        override fun onPlaybackStateChanged(state: Int) {
            // STATE_READY = 3 — duration is finally known.
            if (state == Player.STATE_READY) {
                _state.update {
                    it.copy(preview = it.preview.copy(durationMs = player.durationMs))
                }
            }
        }
    }

    private fun startPlayerPoll() {
        playerPollJob?.cancel()
        playerPollJob = viewModelScope.launch {
            while (player.isPlaying) {
                _state.update {
                    it.copy(preview = it.preview.copy(positionMs = player.positionMs))
                }
                delay(50L)
            }
        }
    }

    fun setModel(uri: Uri) = handleSlotSelection(Slot.SYNTH, uri)
    fun setHubert(uri: Uri) = handleSlotSelection(Slot.HUBERT, uri)
    fun setRmvpe(uri: Uri) = handleSlotSelection(Slot.RMVPE, uri)
    init {
        // Surface whatever's already in the LRU on first compose so the
        // history card is populated across process restarts.
        refreshHistory()
    }

    fun setInput(uri: Uri) {
        val ctx: Context = getApplication()
        val sel = resolve(uri)
        // Reject overly long inputs upfront so the user sees the limit
        // before they wait for a doomed conversion. Files we can't probe at
        // all are treated optimistically (no meta = no rejection).
        viewModelScope.launch(Dispatchers.IO) {
            val meta = AudioMetaProbe.probe(ctx, uri)
            if (meta != null && meta.durationMs > MAX_INPUT_DURATION_MS) {
                _state.update {
                    it.copy(
                        input = null,
                        inputWaveform = null,
                        inputError = "Input too long (${meta.durationMs / 1000}s). Max 60s.",
                    )
                }
                return@launch
            }
            _state.update {
                it.copy(input = sel.copy(meta = meta), inputError = null, inputWaveform = null)
            }
            // Waveform generation actually decodes the file, so we kick it
            // off only after the cheap probe accepted the input — avoids
            // burning a full ffmpeg decode on a file we just rejected.
            val wave = Waveform.generate(ctx, uri)
            _state.update { st ->
                if (st.input?.uri == uri) st.copy(inputWaveform = wave) else st
            }
        }
    }
    fun setF0UpKey(value: Int) = _state.update { it.copy(f0UpKey = value) }
    fun setSpeakerId(value: Long) = _state.update { it.copy(speakerId = value) }
    fun setOutputFormat(format: AudioFormat) {
        _state.update { it.copy(outputFormat = format) }
    }

    private fun handleSlotSelection(slot: Slot, uri: Uri) {
        val sel = resolve(uri)
        // Drop any cached pipeline that was assembled against the previous
        // session — its wrappers reference an OrtSession we're about to close.
        invalidatePipeline()
        // Same Uri re-selected? Just update display state and skip the reload.
        if (sessionUris[slot] == uri && sessions[slot] != null) {
            _state.update { it.applySelection(slot, sel) }
            return
        }
        loadJobs[slot]?.cancel()
        loadJobs.remove(slot)
        closeSession(slot)
        if (slot == Slot.SYNTH) synthMetadata = null
        _state.update { it.applySelection(slot, sel).withStatus(slot, ModelLoadStatus.Loading) }
        startLoad(slot, sel)
    }

    private fun startLoad(slot: Slot, sel: FileSelection) {
        val ctx: Context = getApplication()
        loadJobs[slot] = viewModelScope.launch(Dispatchers.IO) {
            try {
                val session = OrtRuntime.openSession(ctx, sel.uri)
                val summary: ModelSummary? = if (slot == Slot.SYNTH) {
                    val meta = ModelMetadata.fromSession(session)
                        ?: error("synth has no embedded metadata")
                    synthMetadata = meta
                    ModelSummary(
                        sampleRate = meta.samplingRate,
                        f0 = meta.f0,
                        embedder = meta.embedder,
                        embChannels = meta.embChannels,
                    )
                } else null
                sessions[slot] = session
                sessionUris[slot] = sel.uri
                _state.update { it.withStatus(slot, ModelLoadStatus.Loaded(summary)) }
            } catch (t: Throwable) {
                Log.e(TAG, "load $slot failed", t)
                _state.update { it.withStatus(slot, ModelLoadStatus.Failed(t.message ?: "load failed")) }
            } finally {
                loadJobs.remove(slot)
            }
        }
    }

    /**
     * Begin a 44.1 kHz mono mic capture into the app cacheDir. Caller is
     * required to have RECORD_AUDIO granted before invoking — the
     * permission flow lives in the Compose layer where it can show the
     * Settings rationale.
     */
    fun startRecording() {
        if (recorder != null) return
        val ctx: Context = getApplication()
        val dir = File(ctx.cacheDir, "recordings").apply { mkdirs() }
        val file = File(dir, "rec_${System.currentTimeMillis()}.wav")
        val rec = PcmRecorder()
        try {
            rec.start(file)
        } catch (t: Throwable) {
            Log.e(TAG, "startRecording failed", t)
            _state.update { it.copy(inputError = "Recording failed: ${t.message}") }
            return
        }
        recorder = rec
        recorderFile = file
        _state.update { it.copy(recording = RecordingState.Active(0L, 0f), inputError = null) }
        val startedAt = System.currentTimeMillis()
        recorderJob = viewModelScope.launch(Dispatchers.Default) {
            while (isActive && rec.isRecording) {
                val elapsed = System.currentTimeMillis() - startedAt
                _state.update {
                    val cur = it.recording
                    if (cur is RecordingState.Active) it.copy(
                        recording = cur.copy(elapsedMs = elapsed, amplitude = rec.currentAmplitude),
                    ) else it
                }
                if (elapsed >= MAX_INPUT_DURATION_MS) {
                    stopRecording()
                    break
                }
                delay(RECORDING_AMPLITUDE_POLL_MS)
            }
        }
    }

    fun stopRecording() {
        val rec = recorder ?: return
        recorder = null
        recorderJob?.cancel()
        recorderJob = null
        val file = runCatching { rec.stop() }.getOrNull()
        _state.update { it.copy(recording = RecordingState.Idle) }
        if (file != null) {
            // Treat the recording exactly like a picked file so it goes
            // through the same probe + waveform pipeline as user input.
            setInput(file.toUri())
        }
        recorderFile = null
    }

    fun cancelRecording() {
        val rec = recorder ?: return
        recorder = null
        recorderJob?.cancel()
        recorderJob = null
        runCatching { rec.stop() }
        recorderFile?.let { runCatching { it.delete() } }
        recorderFile = null
        _state.update { it.copy(recording = RecordingState.Idle) }
    }

    fun convert() {
        val s = _state.value
        if (s.convertBlockReason() != null) return
        val input = s.input ?: return
        val ctx: Context = getApplication()
        convertJob = viewModelScope.launch(Dispatchers.Default) {
            val t0 = System.currentTimeMillis()
            Log.i(
                TAG,
                "convert: start model=${s.model?.displayName} hubert=${s.hubert?.displayName} " +
                    "rmvpe=${s.rmvpe?.displayName ?: "-"} input=${input.displayName} " +
                    "f0UpKey=${s.f0UpKey} sid=${s.speakerId} outFmt=${s.outputFormat}",
            )
            _state.update {
                it.copy(
                    stage = Stage.RUNNING,
                    runningStep = Step.LOADING,
                    elapsedMs = null,
                    message = null,
                )
            }
            try {
                val pipe = obtainPipeline()
                _state.update { it.copy(runningStep = Step.READING) }

                val src = AudioIo.decode(ctx, input.uri)
                val audio16k = Resampler.resample(src.samples, src.sampleRate, HUBERT_SAMPLE_RATE)
                _state.update { it.copy(runningStep = Step.CONVERTING) }

                val out = pipe.convert(
                    audio16k = audio16k,
                    f0UpKey = s.f0UpKey,
                    speakerId = s.speakerId,
                )

                _state.update { it.copy(runningStep = Step.WRITING) }
                // Stage to a scratch file and copy into the history LRU; the
                // user's "Save as…" later just copies from there to the SAF
                // destination they pick.
                val scratch = File(ctx.cacheDir, "convert_out.${s.outputFormat.ext}")
                AudioIo.encodeToFile(scratch, s.outputFormat, out, pipe.outputSampleRate)
                val baseName = input.displayName.substringBeforeLast('.', input.displayName)
                val cached = HistoryCache.add(ctx, scratch, s.outputFormat, baseName)
                runCatching { scratch.delete() }

                val elapsed = System.currentTimeMillis() - t0
                Log.i(TAG, "convert: done in ${elapsed}ms → ${cached.name}")

                withContext(Dispatchers.Main) { player.load(cached) }

                _state.update {
                    it.copy(
                        stage = Stage.DONE,
                        runningStep = null,
                        elapsedMs = elapsed,
                        message = null,
                        preview = PreviewState(
                            file = cached,
                            format = s.outputFormat,
                            isPlaying = false,
                            positionMs = 0L,
                            durationMs = 0L,
                        ),
                        history = HistoryCache.list(ctx),
                        showResultSheet = true,
                    )
                }
            } catch (t: Throwable) {
                Log.e(TAG, "convert: failed", t)
                _state.update {
                    it.copy(
                        stage = Stage.ERROR,
                        runningStep = null,
                        message = "${t.javaClass.simpleName}: ${t.message ?: "unknown error"}",
                    )
                }
            }
        }
    }

    fun dismissResultSheet() {
        // Pause when the modal hides, otherwise the player keeps running
        // with no controls visible.
        if (player.isPlaying) player.pause()
        _state.update { it.copy(showResultSheet = false) }
    }

    fun showResultSheet() {
        if (_state.value.preview.file != null) {
            _state.update { it.copy(showResultSheet = true) }
        }
    }

    fun togglePlay() {
        if (_state.value.preview.file == null) return
        player.togglePlay()
    }

    fun seekTo(ms: Long) {
        if (_state.value.preview.file == null) return
        player.seekTo(ms)
        _state.update { it.copy(preview = it.preview.copy(positionMs = ms)) }
    }

    fun loadHistoryEntry(entry: HistoryEntry) {
        player.load(entry.file)
        _state.update {
            it.copy(
                preview = PreviewState(
                    file = entry.file,
                    format = entry.format,
                    isPlaying = false,
                    positionMs = 0L,
                    durationMs = 0L,
                ),
            )
        }
    }

    /**
     * Copy the cached preview file into the user-picked SAF destination. The
     * cache file stays around so a subsequent "Save as…" — say, exporting
     * the same conversion to a second location — works without re-running
     * the pipeline. [HistoryCache] owns the eventual cleanup.
     */
    fun saveCurrentPreview(uri: Uri) {
        val src = _state.value.preview.file ?: return
        val ctx: Context = getApplication()
        viewModelScope.launch(Dispatchers.IO) {
            try {
                ctx.contentResolver.openOutputStream(uri)?.use { sink ->
                    src.inputStream().use { it.copyTo(sink) }
                } ?: error("cannot open $uri")
                Log.i(TAG, "saveCurrentPreview: copied ${src.name} → $uri")
            } catch (t: Throwable) {
                Log.e(TAG, "saveCurrentPreview failed", t)
                _state.update { it.copy(message = "Save failed: ${t.message}") }
            }
        }
    }

    fun deleteHistoryEntry(entry: HistoryEntry) {
        // If the user is currently previewing the entry they're deleting,
        // tear down the player so we don't keep a dangling file handle.
        val cur = _state.value.preview.file
        if (cur == entry.file) {
            player.stop()
            _state.update { it.copy(preview = PreviewState()) }
        }
        HistoryCache.delete(entry)
        refreshHistory()
    }

    fun refreshHistory() {
        val ctx: Context = getApplication()
        viewModelScope.launch(Dispatchers.IO) {
            val list = HistoryCache.list(ctx)
            _state.update { it.copy(history = list) }
        }
    }

    private suspend fun obtainPipeline(): RvcPipeline {
        val synth = sessions[Slot.SYNTH] ?: error("synth session not loaded")
        val hubert = sessions[Slot.HUBERT] ?: error("hubert session not loaded")
        val rmvpe = sessions[Slot.RMVPE]
        val meta = synthMetadata ?: error("synth metadata missing")
        val key = listOf(sessionUris[Slot.SYNTH], sessionUris[Slot.HUBERT], sessionUris[Slot.RMVPE])
        val cached = pipeline
        if (cached != null && pipelineKey == key) {
            Log.d(TAG, "obtainPipeline: reusing cached pipeline")
            return cached
        }
        Log.i(TAG, "obtainPipeline: assembling from cached sessions")
        val built = withContext(Dispatchers.Default) {
            RvcPipelineFactory.assemble(synth, meta, hubert, rmvpe)
        }
        pipeline = built
        pipelineKey = key
        return built
    }

    private fun invalidatePipeline() {
        pipeline = null
        pipelineKey = null
    }

    private fun closeSession(slot: Slot) {
        sessions.remove(slot)?.let { runCatching { it.close() } }
        sessionUris.remove(slot)
    }

    override fun onCleared() {
        super.onCleared()
        Log.d(TAG, "onCleared: closing sessions")
        cancelRecording()
        playerPollJob?.cancel()
        runCatching { player.removeListener(playerListener) }
        runCatching { player.release() }
        loadJobs.values.forEach { it.cancel() }
        loadJobs.clear()
        Slot.entries.forEach { closeSession(it) }
        invalidatePipeline()
    }

    private fun resolve(uri: Uri): FileSelection =
        FileSelection(uri, queryDisplayName(getApplication(), uri))

    private fun ConversionUiState.applySelection(slot: Slot, sel: FileSelection): ConversionUiState =
        when (slot) {
            Slot.SYNTH -> copy(model = sel)
            Slot.HUBERT -> copy(hubert = sel)
            Slot.RMVPE -> copy(rmvpe = sel)
        }

    private fun ConversionUiState.withStatus(slot: Slot, status: ModelLoadStatus): ConversionUiState =
        when (slot) {
            Slot.SYNTH -> copy(synthStatus = status)
            Slot.HUBERT -> copy(hubertStatus = status)
            Slot.RMVPE -> copy(rmvpeStatus = status)
        }
}

private fun queryDisplayName(ctx: Context, uri: Uri): String {
    ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { c -> if (c.moveToFirst()) return c.getString(0) }
    return uri.lastPathSegment ?: "(unknown)"
}
