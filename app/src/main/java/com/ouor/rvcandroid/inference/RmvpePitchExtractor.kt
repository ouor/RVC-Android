package com.ouor.rvcandroid.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.Closeable
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.roundToInt

private const val TAG = "Rvc.RMVPE"

private const val F0_MIN = 50f
private const val F0_MAX = 1100f
private val F0_MEL_MIN = 1127f * ln(1f + F0_MIN / 700f)
private val F0_MEL_MAX = 1127f * ln(1f + F0_MAX / 700f)

class PitchData(val pitchf: FloatArray, val pitchCoarse: LongArray)

// Two backends share this interface: OrtRmvpePitchExtractor (CPU/NNAPI
// via onnxruntime, accepts any audio length) and QnnRmvpePitchExtractor
// (NPU via a spawned PIE runner, requires exactly staticAudioLen
// samples per call). The pipeline branches on staticAudioLen to
// decide between upfront extract on the full padded clip and per-
// chunk extracts aligned with synth chunks.
interface PitchExtractor : Closeable {
    val staticAudioLen: Int?
    fun extract(audio16k: FloatArray, f0UpKey: Int = 0, threshold: Float = 0.3f): PitchData
}

class OrtRmvpePitchExtractor(private val session: OrtSession) : PitchExtractor {

    override val staticAudioLen: Int? = null

    init {
        Log.i(TAG, "init: backend=ORT")
    }

    override fun extract(
        audio16k: FloatArray,
        f0UpKey: Int,
        threshold: Float,
    ): PitchData {
        val tStart = System.nanoTime()
        val env = OrtRuntime.env

        env.floatTensor(audio16k, longArrayOf(1L, audio16k.size.toLong())).use { wav ->
            env.floatTensor(floatArrayOf(threshold), longArrayOf(1L)).use { thr ->
                val tBuilt = System.nanoTime()
                session.run(mapOf("waveform" to wav, "threshold" to thr)).use { result ->
                    val tRan = System.nanoTime()
                    val pitchTensor = result.iterator().next().value as OnnxTensor
                    val raw = pitchTensor.copyFloats()
                    val shifted = if (f0UpKey == 0) raw else shiftPitch(raw, f0UpKey)
                    val coarse = melQuantize(shifted)
                    val tDone = System.nanoTime()
                    Log.i(
                        TAG,
                        "extract: audio=${audio16k.size} samples → pitchf[${shifted.size}] " +
                            "(voiced=${coarse.count { it > 1 }}/${coarse.size}) " +
                            "(build=${(tBuilt - tStart) / 1_000_000}ms run=${(tRan - tBuilt) / 1_000_000}ms decode=${(tDone - tRan) / 1_000_000}ms)",
                    )
                    return PitchData(shifted, coarse)
                }
            }
        }
    }

    override fun close() {
        Log.d(TAG, "close")
        session.close()
    }
}

internal fun shiftPitch(pitchf: FloatArray, semitones: Int): FloatArray {
    val factor = 2.0.pow(semitones / 12.0).toFloat()
    return FloatArray(pitchf.size) { pitchf[it] * factor }
}

// Match voice-changer's f0_coarse formula bit-for-bit so the synthesizer
// sees the same pitch-class indices it was trained against.
internal fun melQuantize(pitchf: FloatArray): LongArray {
    val span = F0_MEL_MAX - F0_MEL_MIN
    val out = LongArray(pitchf.size)
    for (i in pitchf.indices) {
        val f = pitchf[i]
        val scaled = if (f > 0f) {
            val mel = 1127f * ln(1f + f / 700f)
            if (mel > 0f) (mel - F0_MEL_MIN) * 254f / span + 1f else 1f
        } else 1f
        out[i] = scaled.coerceIn(1f, 255f).roundToInt().toLong()
    }
    return out
}
