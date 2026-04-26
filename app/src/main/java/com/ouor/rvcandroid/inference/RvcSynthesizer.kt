package com.ouor.rvcandroid.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.Closeable

private const val TAG = "Rvc.Synth"

class RvcSynthesizer(
    private val session: OrtSession,
    private val hasF0: Boolean,
) : Closeable {

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
        val inputs = mutableMapOf<String, OnnxTensor>()
        try {
            inputs["feats"] = env.floatTensor(
                feats,
                longArrayOf(1L, framesT.toLong(), channels.toLong()),
            )
            inputs["p_len"] = env.longTensor(longArrayOf(framesT.toLong()), longArrayOf(1L))
            inputs["sid"] = env.longTensor(longArrayOf(speakerId), longArrayOf(1L))
            if (hasF0) {
                inputs["pitch"] = env.longTensor(pitch!!, longArrayOf(1L, framesT.toLong()))
                inputs["pitchf"] = env.floatTensor(pitchf!!, longArrayOf(1L, framesT.toLong()))
            }
            session.run(inputs).use { result ->
                val tensor = result.iterator().next().value as OnnxTensor
                val audio = tensor.copyFloats()
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

    // Older voice-changer ONNX exports (pre-v2.1) emit unclipped audio; newer
    // ones (SynthesizerTrnMs768NSFsid_ONNX.py:69) bake torch.clip(-1,1) in.
    // Clipping unconditionally here keeps the wrapper agnostic to which
    // export version the user loaded.
    private fun clipInPlace(out: FloatArray) {
        for (i in out.indices) {
            val v = out[i]
            if (v > 1f) out[i] = 1f else if (v < -1f) out[i] = -1f
        }
    }
}
