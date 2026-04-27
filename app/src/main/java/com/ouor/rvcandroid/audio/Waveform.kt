package com.ouor.rvcandroid.audio

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlin.math.sqrt

private const val TAG = "Rvc.Waveform"

object Waveform {
    /**
     * Decode the audio at [uri] and reduce it to [buckets] RMS amplitude
     * values in [0f..1f] for thumbnail rendering. Heavy — call from
     * Dispatchers.IO, not the main thread.
     */
    fun generate(ctx: Context, uri: Uri, buckets: Int = 200): FloatArray? {
        return try {
            val audio = AudioIo.decode(ctx, uri)
            rms(audio.samples, buckets)
        } catch (t: Throwable) {
            Log.w(TAG, "generate failed for $uri: ${t.message}")
            null
        }
    }

    fun rms(samples: FloatArray, buckets: Int): FloatArray {
        if (samples.isEmpty() || buckets <= 0) return FloatArray(0)
        val bucketSize = (samples.size + buckets - 1) / buckets
        val out = FloatArray(buckets)
        var max = 0f
        for (i in 0 until buckets) {
            val start = i * bucketSize
            if (start >= samples.size) break
            val end = minOf(start + bucketSize, samples.size)
            var sumSq = 0.0
            for (j in start until end) {
                val s = samples[j]
                sumSq += s * s
            }
            val v = sqrt(sumSq / (end - start)).toFloat()
            out[i] = v
            if (v > max) max = v
        }
        // Normalise to [0..1] so the renderer doesn't have to think about
        // absolute amplitude — quiet recordings still look reasonable.
        if (max > 0f) {
            for (i in out.indices) out[i] = out[i] / max
        }
        return out
    }
}
