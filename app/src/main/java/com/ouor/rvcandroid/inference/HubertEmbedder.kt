package com.ouor.rvcandroid.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Log
import java.io.Closeable

private const val TAG = "Rvc.HuBERT"

class EmbeddingData(val feats: FloatArray, val frames: Int, val channels: Int) {
    val shape: LongArray get() = longArrayOf(1L, frames.toLong(), channels.toLong())
}

class HubertEmbedder(private val session: OrtSession) : Closeable {

    fun extract(audio16k: FloatArray): EmbeddingData {
        val t0 = System.currentTimeMillis()
        val env = OrtRuntime.env

        env.floatTensor(audio16k, longArrayOf(1L, audio16k.size.toLong())).use { source ->
            paddingMask(env, audio16k.size).use { mask ->
                session.run(mapOf("source" to source, "padding_mask" to mask)).use { result ->
                    val tensor = result.iterator().next().value as OnnxTensor
                    val shape = (tensor.info as TensorInfo).shape
                    require(shape.size == 3 && shape[0] == 1L) {
                        "expected feats [1, T, C], got ${shape.contentToString()}"
                    }
                    val frames = shape[1].toInt()
                    val channels = shape[2].toInt()
                    val feats = tensor.copyFloats()
                    val elapsed = System.currentTimeMillis() - t0
                    Log.i(
                        TAG,
                        "extract: audio=${audio16k.size} → feats[1, $frames, $channels] in ${elapsed}ms",
                    )
                    return EmbeddingData(feats, frames, channels)
                }
            }
        }
    }

    override fun close() {
        Log.d(TAG, "close")
        session.close()
    }

    // FairseqHubert always passes an all-false mask (FairseqHubert.py:28); the
    // exported ONNX still requires the input. We mirror by submitting one full
    // window of zeros — no real padding ever happens at our call sites.
    private fun paddingMask(env: OrtEnvironment, length: Int): OnnxTensor {
        val mask = Array(1) { BooleanArray(length) }
        return OnnxTensor.createTensor(env, mask)
    }
}
