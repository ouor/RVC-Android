package com.ouor.rvcandroid.inference

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Log
import java.io.Closeable

private const val TAG = "Rvc.Synth"
private const val AUDIO_OUTPUT = "audio"

class RvcSynthesizer(
    private val session: OrtSession,
    private val hasF0: Boolean,
) : Closeable {

    private val featsIsFp16: Boolean = featsType(session) == OnnxJavaType.FLOAT16
    private val audioIsFp16: Boolean = audioType(session) == OnnxJavaType.FLOAT16

    init {
        Log.i(TAG, "init: hasF0=$hasF0 featsFp16=$featsIsFp16 audioFp16=$audioIsFp16")
    }

    fun infer(
        feats: FloatArray,
        framesT: Int,
        channels: Int,
        pitch: LongArray? = null,
        pitchf: FloatArray? = null,
        speakerId: Long = 0L,
    ): FloatArray {
        val t0 = System.currentTimeMillis()
        require(feats.size == framesT * channels) {
            "feats size ${feats.size} != $framesT * $channels"
        }
        if (hasF0) {
            requireNotNull(pitch) { "f0 model requires pitch" }
            requireNotNull(pitchf) { "f0 model requires pitchf" }
            require(pitch.size == framesT) { "pitch size ${pitch.size} != $framesT" }
            require(pitchf.size == framesT) { "pitchf size ${pitchf.size} != $framesT" }
        }
        val env = OrtRuntime.env
        val featsShape = longArrayOf(1L, framesT.toLong(), channels.toLong())
        val tShape = longArrayOf(1L, framesT.toLong())
        val inputs = mutableMapOf<String, OnnxTensor>()
        try {
            inputs["feats"] =
                if (featsIsFp16) env.float16Tensor(feats, featsShape)
                else env.floatTensor(feats, featsShape)
            inputs["p_len"] = env.longTensor(longArrayOf(framesT.toLong()), longArrayOf(1L))
            inputs["sid"] = env.longTensor(longArrayOf(speakerId), longArrayOf(1L))
            if (hasF0) {
                inputs["pitch"] = env.longTensor(pitch!!, tShape)
                inputs["pitchf"] = env.floatTensor(pitchf!!, tShape)
            }
            // Some voice-changer exports surface debug intermediates (Mul_*,
            // Slice_*, RandomNormalLike_*, …) in addition to "audio" — pulling
            // only what we need keeps inference cheap and the result mapping
            // robust to schema additions.
            session.run(inputs, setOf(AUDIO_OUTPUT)).use { result ->
                val tensor = result.iterator().next().value as OnnxTensor
                val audio =
                    if (audioIsFp16) tensor.copyFloats16() else tensor.copyFloats()
                clipInPlace(audio)
                val elapsed = System.currentTimeMillis() - t0
                Log.i(
                    TAG,
                    "infer: feats[$framesT,$channels] f0=$hasF0 sid=$speakerId → audio[${audio.size}] in ${elapsed}ms",
                )
                return audio
            }
        } finally {
            inputs.values.forEach { runCatching { it.close() } }
        }
    }

    override fun close() {
        Log.d(TAG, "close")
        session.close()
    }

    private fun clipInPlace(out: FloatArray) {
        for (i in out.indices) {
            val v = out[i]
            if (v > 1f) out[i] = 1f else if (v < -1f) out[i] = -1f
        }
    }

    private companion object {
        fun featsType(session: OrtSession): OnnxJavaType? =
            (session.inputInfo["feats"]?.info as? TensorInfo)?.type

        fun audioType(session: OrtSession): OnnxJavaType? =
            (session.outputInfo[AUDIO_OUTPUT]?.info as? TensorInfo)?.type
    }
}
