package com.ouor.rvcandroid.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ouor.rvcandroid.audio.AudioFormat
import com.ouor.rvcandroid.audio.AudioIo
import com.ouor.rvcandroid.audio.Resampler
import com.ouor.rvcandroid.inference.RvcPipeline
import com.ouor.rvcandroid.inference.RvcPipelineFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "Rvc.Conv"
private const val HUBERT_SAMPLE_RATE = 16000

data class FileSelection(val uri: Uri, val displayName: String)

enum class Stage { IDLE, RUNNING, DONE, ERROR }

enum class Step(val label: String) {
    LOADING("Load"),
    READING("Read"),
    CONVERTING("Convert"),
    WRITING("Write"),
}

data class ConversionUiState(
    val model: FileSelection? = null,
    val hubert: FileSelection? = null,
    val rmvpe: FileSelection? = null,
    val input: FileSelection? = null,
    val output: FileSelection? = null,
    val f0UpKey: Int = 0,
    val speakerId: Long = 0L,
    val stage: Stage = Stage.IDLE,
    val runningStep: Step? = null,
    val elapsedMs: Long? = null,
    val message: String? = null,
)

class ConversionViewModel(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(ConversionUiState())
    val state = _state.asStateFlow()

    private var pipeline: RvcPipeline? = null
    private var pipelineKey: List<Uri?>? = null
    private var job: Job? = null

    fun setModel(uri: Uri) = _state.update { it.copy(model = resolve(uri)) }
    fun setHubert(uri: Uri) = _state.update { it.copy(hubert = resolve(uri)) }
    fun setRmvpe(uri: Uri) = _state.update { it.copy(rmvpe = resolve(uri)) }
    fun setInput(uri: Uri) = _state.update { it.copy(input = resolve(uri)) }
    fun setOutput(uri: Uri) = _state.update { it.copy(output = resolve(uri)) }
    fun setF0UpKey(value: Int) = _state.update { it.copy(f0UpKey = value) }
    fun setSpeakerId(value: Long) = _state.update { it.copy(speakerId = value) }

    fun convert() {
        val s = _state.value
        if (s.model == null || s.hubert == null || s.input == null || s.output == null) return
        if (s.stage == Stage.RUNNING) return
        val ctx: Context = getApplication()
        job = viewModelScope.launch(Dispatchers.Default) {
            val t0 = System.currentTimeMillis()
            Log.i(
                TAG,
                "convert: start model=${s.model.displayName} hubert=${s.hubert.displayName} " +
                    "rmvpe=${s.rmvpe?.displayName ?: "-"} input=${s.input.displayName} " +
                    "f0UpKey=${s.f0UpKey} sid=${s.speakerId}",
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
                val pipe = obtainPipeline(ctx, s.model.uri, s.hubert.uri, s.rmvpe?.uri)
                _state.update { it.copy(runningStep = Step.READING) }

                val src = AudioIo.decode(ctx, s.input.uri)
                val audio16k = Resampler.resample(src.samples, src.sampleRate, HUBERT_SAMPLE_RATE)
                _state.update { it.copy(runningStep = Step.CONVERTING) }

                val out = pipe.convert(
                    audio16k = audio16k,
                    f0UpKey = s.f0UpKey,
                    speakerId = s.speakerId,
                )

                _state.update { it.copy(runningStep = Step.WRITING) }
                val outFormat = AudioFormat.detect(ctx, s.output.uri) ?: AudioFormat.WAV
                AudioIo.encode(ctx, s.output.uri, outFormat, out, pipe.outputSampleRate)

                val elapsed = System.currentTimeMillis() - t0
                Log.i(TAG, "convert: done in ${elapsed}ms")
                _state.update {
                    it.copy(
                        stage = Stage.DONE,
                        runningStep = null,
                        elapsedMs = elapsed,
                        message = null,
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

    private suspend fun obtainPipeline(
        ctx: Context,
        model: Uri,
        hubert: Uri,
        rmvpe: Uri?,
    ): RvcPipeline {
        val key = listOf(model, hubert, rmvpe)
        val cached = pipeline
        if (cached != null && pipelineKey == key) {
            Log.d(TAG, "obtainPipeline: reusing cached pipeline")
            return cached
        }
        Log.i(TAG, "obtainPipeline: building new pipeline")
        cached?.let { runCatching { it.close() } }
        pipeline = null
        pipelineKey = null
        val built = withContext(Dispatchers.IO) {
            RvcPipelineFactory.create(ctx, model, hubert, rmvpe)
        }
        pipeline = built
        pipelineKey = key
        return built
    }

    override fun onCleared() {
        super.onCleared()
        Log.d(TAG, "onCleared: closing pipeline")
        pipeline?.let { runCatching { it.close() } }
        pipeline = null
    }

    private fun resolve(uri: Uri): FileSelection =
        FileSelection(uri, queryDisplayName(getApplication(), uri))
}

private fun queryDisplayName(ctx: Context, uri: Uri): String {
    ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { c -> if (c.moveToFirst()) return c.getString(0) }
    return uri.lastPathSegment ?: "(unknown)"
}
