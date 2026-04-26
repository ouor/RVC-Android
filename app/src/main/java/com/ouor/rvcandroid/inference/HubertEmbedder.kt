package com.ouor.rvcandroid.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Log
import java.io.Closeable

private const val TAG = "Rvc.HuBERT"

class EmbeddingData(val feats: FloatArray, val frames: Int, val channels: Int) {
    val shape: LongArray get() = longArrayOf(1L, frames.toLong(), channels.toLong())
}

// Two backends share this interface: OrtHubertEmbedder (CPU/NNAPI via
// onnxruntime, accepts any audio length) and QnnHubertEmbedder (NPU via
// a spawned PIE runner, requires exactly staticAudioLen samples per
// call). The pipeline branches on staticAudioLen to decide whether to
// embed the full clip up front or per-synth-chunk.
interface HubertEmbedder : Closeable {
    val staticAudioLen: Int?
    fun extract(audio16k: FloatArray): EmbeddingData
}

class OrtHubertEmbedder(
    private val session: OrtSession,
    private val outputName: String,
) : HubertEmbedder {

    override val staticAudioLen: Int? = null

    init {
        require(outputName in session.outputInfo.keys) {
            "hubert ONNX has no '$outputName' output (available: ${session.outputInfo.keys})"
        }
        Log.i(TAG, "init: backend=ORT outputName=$outputName")
    }

    override fun extract(audio16k: FloatArray): EmbeddingData {
        val tStart = System.nanoTime()
        val env = OrtRuntime.env

        env.floatTensor(audio16k, longArrayOf(1L, audio16k.size.toLong())).use { audio ->
            val tBuilt = System.nanoTime()
            session.run(mapOf("audio" to audio), setOf(outputName)).use { result ->
                val tRan = System.nanoTime()
                val tensor = result.iterator().next().value as OnnxTensor
                val shape = (tensor.info as TensorInfo).shape
                require(shape.size == 3 && shape[0] == 1L) {
                    "expected feats [1, T, C], got ${shape.contentToString()}"
                }
                val frames = shape[1].toInt()
                val channels = shape[2].toInt()
                val feats = tensor.copyFloats()
                val tDone = System.nanoTime()
                Log.i(
                    TAG,
                    "extract: audio=${audio16k.size} → $outputName[1, $frames, $channels] " +
                        "(build=${(tBuilt - tStart) / 1_000_000}ms run=${(tRan - tBuilt) / 1_000_000}ms decode=${(tDone - tRan) / 1_000_000}ms)",
                )
                return EmbeddingData(feats, frames, channels)
            }
        }
    }

    override fun close() {
        Log.d(TAG, "close")
        session.close()
    }
}
