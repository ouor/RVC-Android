package com.ouor.rvcandroid.inference

import ai.onnxruntime.OrtSession
import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.Closeable
import kotlin.math.abs

private const val TAG = "Rvc.Pipe"
private const val SR = 16000
private const val WINDOW = 160

class RvcPipeline(
    val metadata: ModelMetadata,
    private val embedder: HubertEmbedder,
    private val pitchExtractor: RmvpePitchExtractor?,
    private val synthesizer: RvcSynthesizer,
) : Closeable {

    val outputSampleRate: Int get() = metadata.samplingRate

    // Chunking thresholds (seconds). The synthesizer runs super-linear in
    // sequence length (measured ~O(N^1.3) on this device's CPU+NNAPI mix),
    // so very long audio benefits from being split. We don't gain much
    // below ~20s once you account for the 2 × xPad seconds of overlap each
    // chunk pays, so xMax acts as a regression guard for short clips.
    private val xPad = 1
    private val xQuery = 3
    private val xCenter = 15
    private val xMax = 18

    private val tPad = SR * xPad
    private val tPadTgt = metadata.samplingRate * xPad
    private val tQuery = SR * xQuery
    private val tCenter = SR * xCenter
    private val tMax = SR * xMax

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

        val out = if (audio16k.size <= tMax) convertSingle(audio16k, f0UpKey, speakerId, onProgress)
        else convertChunked(audio16k, f0UpKey, speakerId, onProgress)

        Log.i(
            TAG,
            "convert: done audio=${out.size} samples @ ${metadata.samplingRate}Hz in ${System.currentTimeMillis() - t0}ms",
        )
        return out
    }

    override fun close() {
        Log.d(TAG, "close")
        runCatching { embedder.close() }
        runCatching { pitchExtractor?.close() }
        runCatching { synthesizer.close() }
    }

    // Single-shot path: no padding, no chunking. Used when the input fits
    // in tMax — the cost of a 2 × xPad reflect pad on each chunk would
    // dominate any savings from splitting at this length.
    private fun convertSingle(
        audio: FloatArray,
        f0UpKey: Int,
        sid: Long,
        onProgress: (Float) -> Unit,
    ): FloatArray {
        Log.i(TAG, "convertSingle: 1 chunk, no padding")
        val pitch = if (metadata.f0) {
            requireNotNull(pitchExtractor) { "f0 model requires pitch extractor" }
            pitchExtractor.extract(audio, f0UpKey)
        } else null
        onProgress(0.4f)
        val out = runStages(audio, pitch, sid)
        onProgress(1f)
        return out
    }

    // Chunked path: split at quiet points, run each chunk through the
    // full HuBERT → synthesizer stack with 2 × xPad seconds of overlap on
    // each side, then trim that overlap from the synth output before
    // concatenating. Pitch is extracted once on the full padded audio
    // since RMVPE scales linearly and slicing its output is cheap.
    private fun convertChunked(
        audio: FloatArray,
        f0UpKey: Int,
        sid: Long,
        onProgress: (Float) -> Unit,
    ): FloatArray {
        val splits = findSplitPoints(audio)
        val boundaries = listOf(0) + splits + listOf(audio.size)
        val numChunks = boundaries.size - 1
        Log.i(
            TAG,
            "convertChunked: $numChunks chunks, splits at ${splits.joinToString { "%.2fs".format(it / SR.toFloat()) }}",
        )

        val padded = reflectPad(audio, tPad)
        val pitch = if (metadata.f0) {
            requireNotNull(pitchExtractor) { "f0 model requires pitch extractor" }
            pitchExtractor.extract(padded, f0UpKey)
        } else null
        onProgress(0.2f)

        val parts = mutableListOf<FloatArray>()
        for (i in 0 until numChunks) {
            // Boundaries are in original-audio coordinates. Padded coords for
            // chunk i are [boundaries[i], boundaries[i+1] + 2*tPad), which
            // covers original time [boundaries[i] - tPad, boundaries[i+1] +
            // tPad]. Trimming tPadTgt from each side of the synth output then
            // recovers exactly [boundaries[i], boundaries[i+1]] in target time.
            val pStart = boundaries[i]
            val pEnd = (boundaries[i + 1] + 2 * tPad).coerceAtMost(padded.size)
            val chunkAudio = padded.copyOfRange(pStart, pEnd)
            val chunkPitch = pitch?.let { sliceChunkPitch(it, pStart, pEnd) }

            val synthOut = runStages(chunkAudio, chunkPitch, sid)
            val trimmed = trimChunkPad(synthOut, tPadTgt)
            parts += trimmed

            Log.d(
                TAG,
                "  chunk ${i + 1}/$numChunks: padded=${chunkAudio.size} synth=${synthOut.size} trimmed=${trimmed.size}",
            )
            onProgress(0.2f + 0.78f * (i + 1) / numChunks)
        }
        onProgress(1f)
        return concat(parts)
    }

    private fun runStages(audio: FloatArray, pitch: PitchData?, sid: Long): FloatArray {
        val emb = embedder.extract(audio)
        val frames2x = emb.frames * 2
        val feats2x = upsample2xNearest(emb.feats, emb.frames, emb.channels)
        val targetT = if (pitch != null) minOf(frames2x, pitch.pitchf.size) else frames2x
        val feats = if (frames2x == targetT) feats2x else feats2x.copyOf(targetT * emb.channels)
        val coarse = pitch?.pitchCoarse?.copyOf(targetT)
        val pitchf = pitch?.pitchf?.copyOf(targetT)
        Log.d(TAG, "  runStages: T=$targetT (hubert2x=${frames2x}, pitch=${pitch?.pitchf?.size ?: 0})")
        return synthesizer.infer(feats, targetT, emb.channels, coarse, pitchf, sid)
    }

    // rvc-nano's split-point picker (pipeline.py ~line 196). Slide an
    // abs-value moving sum of length WINDOW across the audio; for each
    // tCenter checkpoint, snap to the local minimum of that sum within
    // ±tQuery samples — i.e. the quietest spot near the target boundary.
    // A boundary in silence is far less audible after concatenation than
    // a boundary mid-syllable.
    private fun findSplitPoints(audio: FloatArray): List<Int> {
        val halfWin = WINDOW / 2
        val padded = reflectPad(audio, halfWin)
        val audioSum = FloatArray(audio.size)
        var sum = 0f
        for (i in 0 until WINDOW) sum += abs(padded[i])
        audioSum[0] = sum
        for (i in 1 until audio.size) {
            sum -= abs(padded[i - 1])
            sum += abs(padded[i + WINDOW - 1])
            audioSum[i] = sum
        }

        val splits = mutableListOf<Int>()
        var t = tCenter
        while (t < audio.size) {
            val lo = (t - tQuery).coerceAtLeast(0)
            val hi = (t + tQuery).coerceAtMost(audio.size)
            var minIdx = lo
            var minVal = audioSum[lo]
            for (i in lo + 1 until hi) {
                if (audioSum[i] < minVal) {
                    minVal = audioSum[i]
                    minIdx = i
                }
            }
            splits += (minIdx / WINDOW) * WINDOW
            t += tCenter
        }
        return splits
    }

    // numpy-style reflect padding: padded[pad-1] = audio[1], not audio[0].
    // The boundary sample itself is not duplicated, which matches what the
    // PyTorch reference uses inside F.pad(..., mode='reflect').
    private fun reflectPad(audio: FloatArray, pad: Int): FloatArray {
        if (pad == 0 || audio.isEmpty()) return audio
        require(pad < audio.size) { "pad ($pad) must be < audio length (${audio.size})" }
        val n = audio.size
        val out = FloatArray(n + 2 * pad)
        for (i in 0 until pad) out[i] = audio[pad - i]
        System.arraycopy(audio, 0, out, pad, n)
        for (i in 0 until pad) out[pad + n + i] = audio[n - 2 - i]
        return out
    }

    private fun sliceChunkPitch(pitch: PitchData, pStart: Int, pEnd: Int): PitchData {
        val fStart = pStart / WINDOW
        val fEnd = (pEnd / WINDOW).coerceAtMost(pitch.pitchf.size)
        return PitchData(
            pitch.pitchf.copyOfRange(fStart, fEnd),
            pitch.pitchCoarse.copyOfRange(fStart, fEnd),
        )
    }

    private fun trimChunkPad(chunk: FloatArray, pad: Int): FloatArray {
        if (pad == 0) return chunk
        if (chunk.size <= 2 * pad) return FloatArray(0)
        return chunk.copyOfRange(pad, chunk.size - pad)
    }

    private fun concat(parts: List<FloatArray>): FloatArray {
        val total = parts.sumOf { it.size }
        val out = FloatArray(total)
        var off = 0
        for (p in parts) {
            System.arraycopy(p, 0, out, off, p.size)
            off += p.size
        }
        return out
    }

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
            val hubertOutput = chooseHubertOutput(metadata)

            if (metadata.f0) {
                requireNotNull(rmvpeUri) { "f0 model selected but no rmvpe uri provided" }
                Log.i(TAG, "factory: loading rmvpe")
                rmvpe = OrtRuntime.openSession(ctx, rmvpeUri)
            }

            return RvcPipeline(
                metadata = metadata,
                embedder = HubertEmbedder(hubert, hubertOutput),
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

    private fun chooseHubertOutput(meta: ModelMetadata): String = when {
        meta.embOutputLayer == 12 && !meta.useFinalProj -> "unit12"
        meta.embOutputLayer == 9 && meta.useFinalProj -> "units9"
        meta.embOutputLayer == 12 && meta.useFinalProj -> "unit12s"
        else -> error(
            "unsupported embedder config: layer=${meta.embOutputLayer}, finalProj=${meta.useFinalProj}",
        )
    }
}
