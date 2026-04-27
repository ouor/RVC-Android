package com.ouor.rvcandroid.audio

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.core.net.toUri
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import java.io.File

private const val TAG = "Rvc.Player"

/**
 * Thin ExoPlayer wrapper for previewing converted output. Owns one
 * [ExoPlayer] for the lifetime of the ViewModel and exposes the bits the
 * UI cares about: playing flag, position, duration. The position has to
 * be polled from a coroutine — Media3 doesn't emit a Flow for it.
 */
class PreviewPlayer(ctx: Context) {
    private val player: ExoPlayer = ExoPlayer.Builder(ctx).build()

    val isPlaying: Boolean get() = player.isPlaying
    val positionMs: Long get() = player.currentPosition.coerceAtLeast(0L)
    val durationMs: Long get() = player.duration.let { if (it < 0L) 0L else it }

    fun load(file: File) {
        load(file.toUri())
    }

    fun load(uri: Uri) {
        Log.d(TAG, "load $uri")
        player.setMediaItem(MediaItem.fromUri(uri))
        player.prepare()
    }

    fun play() {
        player.playWhenReady = true
    }

    fun pause() {
        player.playWhenReady = false
    }

    fun togglePlay() {
        if (player.isPlaying) pause() else play()
    }

    fun seekTo(ms: Long) {
        player.seekTo(ms)
    }

    fun stop() {
        player.stop()
        player.clearMediaItems()
    }

    fun release() {
        player.release()
    }

    fun addListener(listener: Player.Listener) {
        player.addListener(listener)
    }

    fun removeListener(listener: Player.Listener) {
        player.removeListener(listener)
    }
}
