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
    declaredStaticT: Int? = null,
) : RvcSynth {

    private val featsIsFp16: Boolean = featsType(session) == OnnxJavaType.FLOAT16
    private val audioIsFp16: Boolean = audioType(session) == OnnxJavaType.FLOAT16

    // Static models pin the synth's T axis (feats[1, T, C]). Prefer the
    // declared value from ModelMetadata.staticT, but also infer from the
    // session schema so an ONNX exported without metadata.staticT (older
    // tooling) still gets routed through the static path.
    override val staticT: Int? = declaredStaticT ?: detectStaticTFromSchema(session)

    init {
        Log.i(
            TAG,
            "init: hasF0=$hasF0 featsFp16=$featsIsFp16 audioFp16=$audioIsFp16 " +
                "T=${staticT?.toString() ?: "dynamic"}",
        )
    }

    override fun infer(
        feats: FloatArray,
        framesT: Int,
        channels: Int,
        pitch: LongArray?,
        pitchf: FloatArray?,
        speakerId: Long,
    ): FloatArray {
        val tStart = System.nanoTime()
        require(feats.size == framesT * channels) {
            "feats size ${feats.size} != $framesT * $channels"
        }
        if (staticT != null) {
            require(framesT == staticT) {
                "static synth expects T=$staticT, got T=$framesT — caller must chunk to fixed T"
            }
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
            val tBuilt = System.nanoTime()
            // Some voice-changer exports surface debug intermediates (Mul_*,
            // Slice_*, RandomNormalLike_*, …) in addition to "audio" — pulling
            // only what we need keeps inference cheap and the result mapping
            // robust to schema additions.
            session.run(inputs, setOf(AUDIO_OUTPUT)).use { result ->
                val tRan = System.nanoTime()
                val tensor = result.iterator().next().value as OnnxTensor
                val audio =
                    if (audioIsFp16) tensor.copyFloats16() else tensor.copyFloats()
                clipInPlace(audio)
                val tDone = System.nanoTime()
                Log.i(
                    TAG,
                    "infer: feats[$framesT,$channels] f0=$hasF0 sid=$speakerId → audio[${audio.size}] " +
                        "(build=${(tBuilt - tStart) / 1_000_000}ms run=${(tRan - tBuilt) / 1_000_000}ms decode=${(tDone - tRan) / 1_000_000}ms)",
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

        // Reads feats:[1, T, C] and returns T iff it's a concrete value.
        // Dynamic exports show T=-1; static exports pin it (e.g. 192).
        fun detectStaticTFromSchema(session: OrtSession): Int? {
            val info = (session.inputInfo["feats"]?.info as? TensorInfo) ?: return null
            val shape = info.shape
            if (shape.size != 3) return null
            return if (shape[1] > 0) shape[1].toInt() else null
        }
    }
}
