package com.ouor.rvcandroid.audio

import android.content.Context
import android.net.Uri
import android.util.Log
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.FFprobeKit
import com.arthenica.ffmpegkit.ReturnCode
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "Rvc.FFmpegDec"

/**
 * Decodes any audio format ffmpeg-kit-audio knows (MP3/AAC/M4A/FLAC/OGG/Opus
 * etc.) into a mono float32 [AudioData] at the source's native sample rate.
 *
 * We deliberately preserve the original sample rate instead of asking ffmpeg
 * to resample to 16 kHz inline — Phase 3A needs the source rate for the
 * input metadata badge, and the existing [Resampler] already handles the
 * 16 kHz step downstream.
 */
object FFmpegDecoder {
    fun decode(ctx: Context, uri: Uri): AudioData {
        val ext = AudioFormat.detect(ctx, uri)?.ext ?: "bin"
        val src = SafCache.stageInput(ctx, uri, ext)
        val pcm = SafCache.newScratchFile(ctx, "out_", "f32le")
        try {
            val sampleRate = probeSampleRate(src.absolutePath)
                ?: error("ffprobe could not determine sample rate for $uri")

            val cmd = buildString {
                append("-y -hide_banner -loglevel error ")
                append("-i ").append(quote(src.absolutePath)).append(' ')
                append("-vn -ac 1 -f f32le ")
                append(quote(pcm.absolutePath))
            }
            Log.d(TAG, "decode cmd: $cmd")

            val session = FFmpegKit.execute(cmd)
            if (!ReturnCode.isSuccess(session.returnCode)) {
                error("ffmpeg decode failed (rc=${session.returnCode}): ${session.allLogsAsString}")
            }

            val samples = readF32Le(pcm)
            Log.i(TAG, "decode: ${samples.size} samples @ ${sampleRate}Hz")
            return AudioData(samples, sampleRate)
        } finally {
            SafCache.deleteQuietly(src)
            SafCache.deleteQuietly(pcm)
        }
    }

    private fun probeSampleRate(path: String): Int? {
        val session = FFprobeKit.getMediaInformation(path)
        val info = session.mediaInformation ?: return null
        // FFprobeKit exposes the sample rate on the audio stream as a string
        // ("44100" etc.). Skip non-audio streams (video poster art etc.).
        val streams = info.streams ?: return null
        for (s in streams) {
            if (s.type != "audio") continue
            val sr = s.sampleRate?.toIntOrNull() ?: continue
            return sr
        }
        return null
    }

    private fun readF32Le(file: File): FloatArray {
        val bytes = file.readBytes()
        require(bytes.size % 4 == 0) { "f32le size not multiple of 4: ${bytes.size}" }
        val out = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    private fun quote(path: String): String = "\"" + path.replace("\"", "\\\"") + "\""
}
