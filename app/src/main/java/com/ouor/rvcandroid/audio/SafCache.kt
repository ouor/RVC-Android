package com.ouor.rvcandroid.audio

import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.File

private const val TAG = "Rvc.SafCache"
private const val DIR_NAME = "audio-tmp"

/**
 * ffmpeg-kit operates on file paths, but Android's Storage Access Framework
 * hands us [Uri]s that may live behind a content provider. This helper
 * stages the bytes into the app's cacheDir so ffmpeg can read them, and
 * gives back a `File` that the caller is responsible for releasing.
 */
object SafCache {
    fun stageInput(ctx: Context, uri: Uri, ext: String): File {
        val dir = File(ctx.cacheDir, DIR_NAME).apply { mkdirs() }
        val out = File.createTempFile("in_", ".${ext.ifEmpty { "bin" }}", dir)
        ctx.contentResolver.openInputStream(uri)?.use { input ->
            out.outputStream().use { input.copyTo(it) }
        } ?: run {
            out.delete()
            error("cannot open input uri: $uri")
        }
        Log.d(TAG, "stageInput: $uri -> ${out.absolutePath} (${out.length()} bytes)")
        return out
    }

    fun newScratchFile(ctx: Context, prefix: String, ext: String): File {
        val dir = File(ctx.cacheDir, DIR_NAME).apply { mkdirs() }
        return File.createTempFile(prefix, ".$ext", dir)
    }

    fun deleteQuietly(file: File?) {
        if (file == null) return
        runCatching { file.delete() }.onFailure { Log.w(TAG, "delete failed: ${file.absolutePath}", it) }
    }
}
