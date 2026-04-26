package com.ouor.rvcandroid.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

private const val TAG = "Rvc.Conv"

data class FileSelection(val uri: Uri, val displayName: String)

enum class Stage { IDLE, RUNNING, DONE, ERROR }

data class ConversionUiState(
    val model: FileSelection? = null,
    val input: FileSelection? = null,
    val output: FileSelection? = null,
    val stage: Stage = Stage.IDLE,
    val progress: Float = 0f,
    val message: String? = null,
)

class ConversionViewModel(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(ConversionUiState())
    val state = _state.asStateFlow()

    private var job: Job? = null

    fun setModel(uri: Uri) = _state.update { it.copy(model = resolve(uri)) }
    fun setInput(uri: Uri) = _state.update { it.copy(input = resolve(uri)) }
    fun setOutput(uri: Uri) = _state.update { it.copy(output = resolve(uri)) }

    fun convert() {
        val s = _state.value
        if (s.model == null || s.input == null || s.output == null) return
        if (s.stage == Stage.RUNNING) return
        val ctx: Context = getApplication()
        job = viewModelScope.launch {
            Log.i(TAG, "convert: start model=${s.model.displayName} input=${s.input.displayName} output=${s.output.displayName}")
            val t0 = System.currentTimeMillis()
            _state.update { it.copy(stage = Stage.RUNNING, progress = 0f, message = null) }
            try {
                stubConvert(ctx, s.input.uri, s.output.uri) { p ->
                    _state.update { it.copy(progress = p) }
                }
                val elapsed = System.currentTimeMillis() - t0
                Log.i(TAG, "convert: done in ${elapsed}ms")
                _state.update {
                    it.copy(
                        stage = Stage.DONE,
                        progress = 1f,
                        message = "Saved to ${s.output.displayName}",
                    )
                }
            } catch (t: Throwable) {
                Log.e(TAG, "convert: failed", t)
                _state.update {
                    it.copy(stage = Stage.ERROR, message = t.message ?: "failed")
                }
            }
        }
    }

    private fun resolve(uri: Uri): FileSelection =
        FileSelection(uri, queryDisplayName(getApplication(), uri))
}

private fun queryDisplayName(ctx: Context, uri: Uri): String {
    ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { c -> if (c.moveToFirst()) return c.getString(0) }
    return uri.lastPathSegment ?: "(unknown)"
}

// Phase 2 placeholder: copies input bytes to output and fakes progress so the
// UI flow can be validated end-to-end before any real RVC inference exists.
private suspend fun stubConvert(
    ctx: Context,
    input: Uri,
    output: Uri,
    onProgress: (Float) -> Unit,
) {
    onProgress(0f)
    val bytes = ctx.contentResolver.openInputStream(input)
        ?.use { it.readBytes() }
        ?: error("cannot open input")
    repeat(10) { i ->
        delay(80)
        onProgress((i + 1) / 10f)
    }
    ctx.contentResolver.openOutputStream(output)
        ?.use { it.write(bytes) }
        ?: error("cannot open output")
}
