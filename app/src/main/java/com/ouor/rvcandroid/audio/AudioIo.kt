package com.ouor.rvcandroid.audio

import android.content.Context
import android.net.Uri
import android.util.Log

private const val TAG = "Rvc.AudioIo"

/**
 * Format-aware decode/encode dispatcher.
 *
 * WAV goes through the in-process [WavIo] for precision and zero-copy where
 * possible; everything else routes to FFmpeg (added in Phase 1.3/1.4).
 */
object AudioIo {
    fun decode(ctx: Context, uri: Uri): AudioData {
        val format = AudioFormat.detect(ctx, uri)
        Log.i(TAG, "decode: uri=$uri format=$format")
        return when (format) {
            AudioFormat.WAV, null -> decodeWav(ctx, uri)
            else -> error("decoder for $format not yet implemented")
        }
    }

    fun encode(
        ctx: Context,
        uri: Uri,
        format: AudioFormat,
        samples: FloatArray,
        sampleRate: Int,
    ) {
        Log.i(TAG, "encode: uri=$uri format=$format samples=${samples.size} sr=$sampleRate")
        when (format) {
            AudioFormat.WAV -> encodeWav(ctx, uri, samples, sampleRate)
            else -> error("encoder for $format not yet implemented")
        }
    }

    private fun decodeWav(ctx: Context, uri: Uri): AudioData =
        ctx.contentResolver.openInputStream(uri)?.use { WavIo.read(it) }
            ?: error("cannot open input: $uri")

    private fun encodeWav(ctx: Context, uri: Uri, samples: FloatArray, sampleRate: Int) {
        ctx.contentResolver.openOutputStream(uri)?.use { WavIo.write(it, samples, sampleRate) }
            ?: error("cannot open output: $uri")
    }
}
