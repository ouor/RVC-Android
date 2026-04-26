package com.ouor.rvcandroid.inference

import ai.onnxruntime.OrtSession
import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.Closeable

private const val TAG = "Rvc.Pipe"

class RvcPipeline(
    val metadata: ModelMetadata,
    private val embedder: HubertEmbedder,
    private val pitchExtractor: RmvpePitchExtractor?,
    private val synthesizer: RvcSynthesizer,
) : Closeable {

    val outputSampleRate: Int get() = metadata.samplingRate

    fun convert(
        audio16k: FloatArray,
        f0UpKey: Int = 0,
        speakerId: Long = 0L,
        onProgress: (Float) -> Unit = {},
    ): FloatArray {
        val t0 = System.currentTimeMillis()
        Log.i(
            TAG,
            "convert: input=${audio16k.size} samples (${audio16k.size / 16}ms), " +
                "f0UpKey=$f0UpKey, sid=$speakerId, target=${metadata.samplingRate}Hz",
        )
        onProgress(0f)

        val emb = embedder.extract(audio16k)
        onProgress(0.4f)

        val pitch = if (metadata.f0) {
            requireNotNull(pitchExtractor) { "f0 model requires pitch extractor" }
            pitchExtractor.extract(audio16k, f0UpKey)
        } else null
        onProgress(0.6f)

        val frames2x = emb.frames * 2
        val feats2x = upsample2xNearest(emb.feats, emb.frames, emb.channels)

        val targetT = if (pitch != null) minOf(frames2x, pitch.pitchf.size) else frames2x
        val feats = if (frames2x == targetT) feats2x else feats2x.copyOf(targetT * emb.channels)
        val coarse = pitch?.pitchCoarse?.copyOf(targetT)
        val pitchf = pitch?.pitchf?.copyOf(targetT)

        Log.d(TAG, "  aligned T=$targetT, feats=${feats.size}, pitch=${coarse?.size}")
        onProgress(0.7f)

        val audio = synthesizer.infer(
            feats = feats,
            framesT = targetT,
            channels = emb.channels,
            pitch = coarse,
            pitchf = pitchf,
            speakerId = speakerId,
        )
        onProgress(1f)

        val elapsed = System.currentTimeMillis() - t0
        Log.i(
            TAG,
            "convert: done audio=${audio.size} samples @ ${metadata.samplingRate}Hz in ${elapsed}ms",
        )
        return audio
    }

    override fun close() {
        Log.d(TAG, "close")
        runCatching { embedder.close() }
        runCatching { pitchExtractor?.close() }
        runCatching { synthesizer.close() }
    }

    // HuBERT emits at ~50fps (one frame per 320 samples of 16 kHz audio); the
    // synthesizer expects ~100fps (window=160). Voice-changer's Pipeline.py:211
    // bridges the two with F.interpolate(scale_factor=2) at PyTorch's default
    // mode='nearest' — each input frame replicated to two output frames.
    private fun upsample2xNearest(feats: FloatArray, frames: Int, channels: Int): FloatArray {
        val out = FloatArray(frames * 2 * channels)
        for (t in 0 until frames) {
            val src = t * channels
            val dst = t * 2 * channels
            System.arraycopy(feats, src, out, dst, channels)
            System.arraycopy(feats, src, out, dst + channels, channels)
        }
        return out
    }
}

object RvcPipelineFactory {
    fun create(
        ctx: Context,
        modelUri: Uri,
        hubertUri: Uri,
        rmvpeUri: Uri?,
    ): RvcPipeline {
        var synth: OrtSession? = null
        var hubert: OrtSession? = null
        var rmvpe: OrtSession? = null
        try {
            Log.i(TAG, "factory: loading synthesizer")
            synth = OrtRuntime.openSession(ctx, modelUri)
            val metadata = ModelMetadata.fromSession(synth)
                ?: error("synthesizer has no embedded metadata; export it via voice-changer")

            Log.i(TAG, "factory: loading hubert")
            hubert = OrtRuntime.openSession(ctx, hubertUri)

            if (metadata.f0) {
                requireNotNull(rmvpeUri) { "f0 model selected but no rmvpe uri provided" }
                Log.i(TAG, "factory: loading rmvpe")
                rmvpe = OrtRuntime.openSession(ctx, rmvpeUri)
            }

            return RvcPipeline(
                metadata = metadata,
                embedder = HubertEmbedder(hubert),
                pitchExtractor = rmvpe?.let { RmvpePitchExtractor(it) },
                synthesizer = RvcSynthesizer(synth, metadata.f0),
            )
        } catch (t: Throwable) {
            Log.e(TAG, "factory: aborting; closing partial sessions", t)
            runCatching { synth?.close() }
            runCatching { hubert?.close() }
            runCatching { rmvpe?.close() }
            throw t
        }
    }
}
