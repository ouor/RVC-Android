package com.ouor.rvcandroid.audio

import android.content.Context
import android.util.Log
import java.io.File

private const val TAG = "Rvc.History"
private const val DIR_NAME = "history"
private const val MAX_ENTRIES = 10

data class HistoryEntry(
    val file: File,
    val format: AudioFormat,
    val createdAt: Long,
    val sizeBytes: Long,
)

/**
 * Tiny LRU-by-mtime cache for converted output files. Anything past
 * [MAX_ENTRIES] is deleted on the next [add] — newest first ordering, so
 * the most recent conversion always wins room.
 */
object HistoryCache {
    fun add(ctx: Context, source: File, format: AudioFormat, displayBase: String): File {
        val dir = dirOf(ctx)
        val ts = System.currentTimeMillis()
        val safe = displayBase.replace(Regex("[^A-Za-z0-9._-]"), "_").take(40)
        val dest = File(dir, "${ts}_${safe}.${format.ext}")
        source.inputStream().use { input ->
            dest.outputStream().use { input.copyTo(it) }
        }
        Log.i(TAG, "add: ${dest.name} (${dest.length()} bytes)")
        prune(dir)
        return dest
    }

    fun list(ctx: Context): List<HistoryEntry> {
        val dir = dirOf(ctx)
        val files = dir.listFiles() ?: return emptyList()
        return files.mapNotNull { f ->
            val format = AudioFormat.fromExtension(f.extension) ?: return@mapNotNull null
            HistoryEntry(
                file = f,
                format = format,
                createdAt = f.lastModified(),
                sizeBytes = f.length(),
            )
        }.sortedByDescending { it.createdAt }
    }

    fun delete(entry: HistoryEntry) {
        runCatching { entry.file.delete() }
    }

    fun clear(ctx: Context) {
        dirOf(ctx).listFiles()?.forEach { runCatching { it.delete() } }
    }

    private fun dirOf(ctx: Context): File = File(ctx.cacheDir, DIR_NAME).apply { mkdirs() }

    private fun prune(dir: File) {
        val files = dir.listFiles()?.sortedByDescending { it.lastModified() } ?: return
        if (files.size <= MAX_ENTRIES) return
        files.drop(MAX_ENTRIES).forEach {
            Log.d(TAG, "prune: ${it.name}")
            runCatching { it.delete() }
        }
    }
}
