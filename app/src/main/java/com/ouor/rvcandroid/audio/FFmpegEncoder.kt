package com.ouor.rvcandroid.audio

import android.content.Context
import android.net.Uri
import android.util.Log
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.ReturnCode
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "Rvc.FFmpegEnc"

/**
 * Encodes a mono float32 sample buffer into a compressed audio container
 * (MP3 via libmp3lame, AAC inside an M4A container, FLAC, OGG/Vorbis).
 *
 * The pipeline writes the raw f32le samples to a scratch file, runs ffmpeg
 * once with the matching codec args, then streams the produced container
 * back out to the user's chosen SAF [Uri].
 */
object FFmpegEncoder {
    fun encode(
        ctx: Context,
        uri: Uri,
        format: AudioFormat,
        samples: FloatArray,
        sampleRate: Int,
    ) {
        require(format != AudioFormat.WAV) { "use WavIo for WAV output" }
        val out = SafCache.newScratchFile(ctx, "enc_out_", format.ext)
        try {
            encodeToFile(out, format, samples, sampleRate)
            ctx.contentResolver.openOutputStream(uri)?.use { sink ->
                out.inputStream().use { it.copyTo(sink) }
            } ?: error("cannot open output uri: $uri")
            Log.i(TAG, "encode: wrote ${out.length()} bytes as ${format.displayName} → $uri")
        } finally {
            SafCache.deleteQuietly(out)
        }
    }

    /**
     * Encode straight to a local [target] file. The PCM scratch is still
     * needed because ffmpeg only reads from a path, but we skip the SAF
     * round-trip the [encode] caller pays.
     */
    fun encodeToFile(
        target: File,
        format: AudioFormat,
        samples: FloatArray,
        sampleRate: Int,
    ) {
        require(format != AudioFormat.WAV) { "use WavIo for WAV output" }
        val pcm = File.createTempFile("enc_in_", ".f32le", target.parentFile ?: target)
        try {
            writeF32Le(pcm, samples)
            val cmd = buildCommand(pcm.absolutePath, sampleRate, format, target.absolutePath)
            Log.d(TAG, "encodeToFile cmd: $cmd")
            val session = FFmpegKit.execute(cmd)
            if (!ReturnCode.isSuccess(session.returnCode)) {
                error("ffmpeg encode failed (rc=${session.returnCode}): ${session.allLogsAsString}")
            }
            Log.i(TAG, "encodeToFile: wrote ${target.length()} bytes as ${format.displayName}")
        } finally {
            runCatching { pcm.delete() }
        }
    }

    private fun buildCommand(
        pcmPath: String,
        sampleRate: Int,
        format: AudioFormat,
        outPath: String,
    ): String {
        // ffmpeg is told the raw PCM layout up-front (-f f32le, -ac 1, -ar)
        // so it doesn't try to probe the headerless input. The codec choice
        // is the one place each format diverges.
        val codecArgs = when (format) {
            AudioFormat.MP3 -> "-c:a libmp3lame -q:a 2" // VBR ~190 kbps, transparent
            AudioFormat.AAC, AudioFormat.M4A -> "-c:a aac -b:a 192k"
            AudioFormat.FLAC -> "-c:a flac -compression_level 5"
            AudioFormat.OGG -> "-c:a libvorbis -q:a 5" // ~160 kbps
            AudioFormat.WAV -> error("unreachable")
        }
        return buildString {
            append("-y -hide_banner -loglevel error ")
            append("-f f32le -ac 1 -ar ").append(sampleRate).append(' ')
            append("-i ").append(quote(pcmPath)).append(' ')
            append(codecArgs).append(' ')
            append(quote(outPath))
        }
    }

    private fun writeF32Le(file: File, samples: FloatArray) {
        val buf = ByteBuffer.allocate(samples.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        buf.asFloatBuffer().put(samples)
        file.writeBytes(buf.array())
    }

    private fun quote(path: String): String = "\"" + path.replace("\"", "\\\"") + "\""
}
