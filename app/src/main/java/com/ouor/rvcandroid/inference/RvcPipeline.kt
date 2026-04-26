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
    private val pitchExtractor: PitchExtractor?,
    private val synthesizer: RvcSynth,
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

        val staticT = synthesizer.staticT
        val out = when {
            staticT != null -> convertStaticChunked(audio16k, f0UpKey, speakerId, onProgress, staticT)
            audio16k.size <= tMax -> convertSingle(audio16k, f0UpKey, speakerId, onProgress)
            else -> convertChunked(audio16k, f0UpKey, speakerId, onProgress)
        }

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

    // Static-T path: the synthesizer's T axis is pinned (e.g. 192).
    // RMVPE runs once on the full padded audio (CNN+GRU is fine on the
    // long sequence and slicing per-chunk is cheap). HuBERT either
    // runs once upfront (ORT path — gives each frame the full clip's
    // attention context) or per-chunk (QNN path — the .bin baked a
    // static audio_samples shape, so we have no choice). The final
    // concat is trimmed back to the original duration so the appended
    // zero-pad is discarded.
    private fun convertStaticChunked(
        audio: FloatArray,
        f0UpKey: Int,
        sid: Long,
        onProgress: (Float) -> Unit,
        staticT: Int,
    ): FloatArray {
        if (audio.isEmpty()) return FloatArray(0)

        val chunkSamples = staticT * WINDOW
        val originalLen = audio.size
        val paddedLen = ((originalLen + chunkSamples - 1) / chunkSamples) * chunkSamples
        val padded = if (paddedLen == originalLen) audio else FloatArray(paddedLen).also {
            System.arraycopy(audio, 0, it, 0, originalLen)
        }
        val numChunks = paddedLen / chunkSamples

        val perChunkHubert = embedder.staticAudioLen != null
        val hubertSubLen: Int
        val hubertCallsPerChunk: Int
        if (perChunkHubert) {
            hubertSubLen = embedder.staticAudioLen!!
            require(chunkSamples % hubertSubLen == 0) {
                "embedder.staticAudioLen=$hubertSubLen must divide chunkSamples=$chunkSamples " +
                    "(static HuBERT compiled at a length the synth chunk isn't a multiple of)"
            }
            hubertCallsPerChunk = chunkSamples / hubertSubLen
        } else {
            hubertSubLen = 0
            hubertCallsPerChunk = 0
        }

        val perChunkRmvpe = pitchExtractor?.staticAudioLen != null
        if (perChunkRmvpe) {
            require(pitchExtractor!!.staticAudioLen == chunkSamples) {
                "pitchExtractor.staticAudioLen=${pitchExtractor.staticAudioLen} != " +
                    "chunkSamples=$chunkSamples (RMVPE .bin compiled at the wrong length)"
            }
        }

        Log.i(
            TAG,
            "convertStaticChunked: T=$staticT, chunkSamples=$chunkSamples, " +
                "audio padded ${originalLen}→${paddedLen}, numChunks=$numChunks, " +
                "perChunkHubert=$perChunkHubert (subLen=$hubertSubLen, " +
                "$hubertCallsPerChunk calls/chunk), perChunkRmvpe=$perChunkRmvpe",
        )

        // Upfront RMVPE — runs once on the full padded audio when the
        // ORT path is in play (CPU scales near-linearly so doing it
        // once is fine). The QNN path defers to per-chunk extracts
        // inside the loop because the .bin baked a fixed audio_samples.
        val pitch = if (metadata.f0 && !perChunkRmvpe) {
            requireNotNull(pitchExtractor) { "f0 model requires pitch extractor" }
            pitchExtractor.extract(padded, f0UpKey)
        } else null
        onProgress(0.2f)

        // Upfront HuBERT (ORT path) computes once on the full padded
        // audio and we slice. QNN path defers to per-chunk extract().
        var fullFeats2x: FloatArray? = null
        var fullChannels = 0
        var totalFrames2x = numChunks * staticT
        if (!perChunkHubert) {
            val emb = embedder.extract(padded)
            fullChannels = emb.channels
            fullFeats2x = upsample2xNearest(emb.feats, emb.frames, emb.channels)
            totalFrames2x = emb.frames * 2
            Log.d(
                TAG,
                "  upfront: feats[$totalFrames2x, $fullChannels] " +
                    "pitch[${pitch?.pitchf?.size ?: 0}] → $numChunks chunks",
            )
            onProgress(0.55f)
        }

        val parts = mutableListOf<FloatArray>()
        for (i in 0 until numChunks) {
            val frameStart = i * staticT
            val frameEnd = (frameStart + staticT).coerceAtMost(totalFrames2x)

            val featsSlice: FloatArray
            val channels: Int
            if (perChunkHubert) {
                // staticAudioLen may be smaller than chunkSamples when
                // the .bin was compiled at a length the V79 VTCM
                // budget allowed (e.g. 5120 vs 30720). We make
                // hubertCallsPerChunk runs and concat their frames so
                // the synth still sees one staticT-sized window.
                val subParts = ArrayList<EmbeddingData>(hubertCallsPerChunk)
                for (s in 0 until hubertCallsPerChunk) {
                    val subStart = i * chunkSamples + s * hubertSubLen
                    val subAudio = padded.copyOfRange(subStart, subStart + hubertSubLen)
                    subParts += embedder.extract(subAudio)
                }
                channels = subParts[0].channels
                val totalFrames = subParts.sumOf { it.frames }
                val combined = FloatArray(totalFrames * channels)
                var off = 0
                for (p in subParts) {
                    System.arraycopy(p.feats, 0, combined, off, p.feats.size)
                    off += p.feats.size
                }
                // 2x nearest upsample 50 Hz → 100 Hz. ContentVec's
                // conv stack drops one frame per call to receptive
                // field, so totalFrames is typically (numCalls × N) - 0
                // and totalFrames × 2 may fall a few frames short of
                // staticT (e.g. 6 × 15 = 90 → 180, vs staticT=192).
                // sliceFeatsToStaticT handles this by replicating the
                // last frame to fill — same handling the upfront path
                // uses on its tail chunk.
                val combined2x = upsample2xNearest(combined, totalFrames, channels)
                featsSlice = sliceFeatsToStaticT(
                    combined2x, 0, totalFrames * 2, staticT, channels,
                )
            } else {
                channels = fullChannels
                featsSlice = sliceFeatsToStaticT(
                    fullFeats2x!!, frameStart, frameEnd, staticT, channels,
                )
            }

            val coarseSlice: LongArray?
            val pitchfSlice: FloatArray?
            if (perChunkRmvpe) {
                val chunkAudio = padded.copyOfRange(i * chunkSamples, (i + 1) * chunkSamples)
                val chunkPitch = pitchExtractor!!.extract(chunkAudio, f0UpKey)
                // QNN RMVPE returns exactly staticT frames per call,
                // but we still go through the slicer so a short tail
                // (rare; static-T → exact match) gets padded with the
                // standard fill values rather than crashing on size.
                coarseSlice = sliceLong(chunkPitch.pitchCoarse, 0, chunkPitch.pitchCoarse.size, staticT, fill = 1L)
                pitchfSlice = sliceFloat(chunkPitch.pitchf, 0, chunkPitch.pitchf.size, staticT, fill = 0f)
            } else {
                coarseSlice = pitch?.let { sliceLong(it.pitchCoarse, frameStart, frameEnd, staticT, fill = 1L) }
                pitchfSlice = pitch?.let { sliceFloat(it.pitchf, frameStart, frameEnd, staticT, fill = 0f) }
            }

            val synthOut = synthesizer.infer(
                featsSlice, staticT, channels, coarseSlice, pitchfSlice, sid,
            )
            parts += synthOut
            onProgress(0.55f + 0.43f * (i + 1) / numChunks)
        }

        val concatenated = concat(parts)
        val expectedTotalSamples = (originalLen.toLong() * metadata.samplingRate / SR).toInt()
        onProgress(1f)
        Log.d(
            TAG,
            "  concat: ${concatenated.size} samples → trim to $expectedTotalSamples (original duration)",
        )
        return if (concatenated.size > expectedTotalSamples) {
            concatenated.copyOfRange(0, expectedTotalSamples)
        } else concatenated
    }

    private fun sliceFeatsToStaticT(
        feats2x: FloatArray,
        frameStart: Int,
        frameEnd: Int,
        staticT: Int,
        channels: Int,
    ): FloatArray {
        val valid = frameEnd - frameStart
        if (valid == staticT) {
            return feats2x.copyOfRange(frameStart * channels, frameEnd * channels)
        }
        // Tail chunk: copy what's there, replicate the last valid frame
        // for the rest. Replicating beats zero-padding because zeros at
        // feats input would confuse the synth's TextEncoder.
        val out = FloatArray(staticT * channels)
        System.arraycopy(feats2x, frameStart * channels, out, 0, valid * channels)
        val lastFrameOff = (frameEnd - 1).coerceAtLeast(0) * channels
        for (t in valid until staticT) {
            System.arraycopy(feats2x, lastFrameOff, out, t * channels, channels)
        }
        return out
    }

    private fun sliceLong(arr: LongArray, frameStart: Int, frameEnd: Int, target: Int, fill: Long): LongArray {
        val safeStart = frameStart.coerceAtMost(arr.size)
        val safeEnd = frameEnd.coerceAtMost(arr.size)
        val raw = arr.copyOfRange(safeStart, safeEnd)
        if (raw.size == target) return raw
        if (raw.size > target) return raw.copyOf(target)
        val out = LongArray(target) { fill }
        System.arraycopy(raw, 0, out, 0, raw.size)
        return out
    }

    private fun sliceFloat(arr: FloatArray, frameStart: Int, frameEnd: Int, target: Int, fill: Float): FloatArray {
        val safeStart = frameStart.coerceAtMost(arr.size)
        val safeEnd = frameEnd.coerceAtMost(arr.size)
        val raw = arr.copyOfRange(safeStart, safeEnd)
        if (raw.size == target) return raw
        if (raw.size > target) return raw.copyOf(target)
        val out = FloatArray(target).also { it.fill(fill) }
        System.arraycopy(raw, 0, out, 0, raw.size)
        return out
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
    // Hard-coded metadata for QNN .bin loads. AI Hub strips the ONNX-
    // embedded metadata when emitting the context binary, so we cannot
    // round-trip the same JSON we put in via export_static_synthesizer.
    // These values are tied to the Tsukuyomi v2 40k test model that
    // tools/compile_synth_aihub.py compiled. Different models will need
    // either matching constants or a sidecar metadata file (a follow-up
    // can have the compile script emit one alongside the .bin).
    private const val QNN_DEFAULT_SR = 40000
    private const val QNN_DEFAULT_T = 192
    private const val QNN_DEFAULT_INTER_CHANNELS = 192

    // HuBERT QNN .bin compiled by tools/compile_hubert_aihub.py.
    // We can't use chunkSamples=30720 directly: V79 has 8 MiB VTCM
    // and the conv_layers.1 op alone needs ~48 MiB at that input
    // length, so AI Hub compile fails. 5120 fits cleanly (chunkSamples
    // = 6 × 5120) — the pipeline runs HuBERT 6× per synth chunk and
    // concatenates the 16-frame outputs into the 96-frame block synth
    // expects. Trade-off: HuBERT's transformer attention only sees
    // 0.32 s of context per call instead of 1.92 s. Audible boundary
    // artifacts haven't been a problem in tests so far.
    private const val QNN_HUBERT_AUDIO_SAMPLES = 5120

    fun create(
        ctx: Context,
        modelUri: Uri,
        hubertUri: Uri,
        rmvpeUri: Uri?,
    ): RvcPipeline {
        val synthName = queryDisplayName(ctx, modelUri)
        val hubertName = queryDisplayName(ctx, hubertUri)
        val rmvpeName = rmvpeUri?.let { queryDisplayName(ctx, it) }
        val synthIsQnn = synthName.endsWith(".bin", ignoreCase = true)
        val hubertIsQnn = hubertName.endsWith(".bin", ignoreCase = true)
        val rmvpeIsQnn = rmvpeName?.endsWith(".bin", ignoreCase = true) == true
        Log.i(
            TAG,
            "factory: synth='$synthName' (${if (synthIsQnn) "QNN" else "ORT"}), " +
                "hubert='$hubertName' (${if (hubertIsQnn) "QNN" else "ORT"}), " +
                "rmvpe='${rmvpeName ?: "(none)"}' " +
                "(${if (rmvpeName == null) "skip" else if (rmvpeIsQnn) "QNN" else "ORT"})",
        )

        var ortSynth: OrtSession? = null
        var ortHubert: OrtSession? = null
        var ortRmvpe: OrtSession? = null
        var qnnSynth: QnnRvcSynthesizer? = null
        var qnnHubert: QnnHubertEmbedder? = null
        var qnnRmvpe: QnnRmvpePitchExtractor? = null
        try {
            // Pick a metadata source. ORT synth carries the embedded
            // JSON; QNN synth has no ONNX to read so we fall back to
            // the hardcoded constants and the QNN HuBERT will be
            // expected to match (768-channel v2, layer 12).
            val metadata: ModelMetadata
            val synth: RvcSynth
            if (synthIsQnn) {
                metadata = ModelMetadata(
                    samplingRate = QNN_DEFAULT_SR,
                    f0 = true,
                    embChannels = 768,
                    embedder = "hubert_base",
                    embOutputLayer = 12,
                    useFinalProj = false,
                    modelType = "QnnContextBinary",
                    version = "static-1.0",
                    staticT = QNN_DEFAULT_T,
                )
                Log.i(TAG, "factory: caching synth .bin")
                val cachedBin = cacheBinUri(ctx, modelUri)
                Log.i(TAG, "factory: loading QNN synthesizer from ${cachedBin.name}")
                qnnSynth = QnnRvcSynthesizer(
                    ctx = ctx,
                    binPath = cachedBin.absolutePath,
                    staticT = QNN_DEFAULT_T,
                    interChannels = QNN_DEFAULT_INTER_CHANNELS,
                    outputSampleRate = QNN_DEFAULT_SR,
                )
                synth = qnnSynth
            } else {
                Log.i(TAG, "factory: loading synthesizer (ORT)")
                ortSynth = OrtRuntime.openSession(ctx, modelUri)
                metadata = ModelMetadata.fromSession(ortSynth)
                    ?: error("synthesizer has no embedded metadata; export it via voice-changer")
                synth = RvcSynthesizer(
                    session = ortSynth,
                    hasF0 = metadata.f0,
                    declaredStaticT = metadata.staticT,
                )
            }

            val embedder: HubertEmbedder
            if (hubertIsQnn) {
                Log.i(TAG, "factory: caching hubert .bin")
                val cachedBin = cacheBinUri(ctx, hubertUri)
                Log.i(TAG, "factory: loading QNN hubert from ${cachedBin.name}")
                qnnHubert = QnnHubertEmbedder(
                    ctx = ctx,
                    binPath = cachedBin.absolutePath,
                    staticAudioLen = QNN_HUBERT_AUDIO_SAMPLES,
                )
                embedder = qnnHubert
            } else {
                Log.i(TAG, "factory: loading hubert (ORT)")
                ortHubert = OrtRuntime.openSession(ctx, hubertUri)
                embedder = OrtHubertEmbedder(ortHubert, chooseHubertOutput(metadata))
            }

            val pitchExtractor: PitchExtractor? = if (metadata.f0) {
                requireNotNull(rmvpeUri) { "f0 model selected but no rmvpe uri provided" }
                if (rmvpeIsQnn) {
                    Log.i(TAG, "factory: caching rmvpe .bin")
                    val cachedBin = cacheBinUri(ctx, rmvpeUri)
                    Log.i(TAG, "factory: loading QNN rmvpe from ${cachedBin.name}")
                    qnnRmvpe = QnnRmvpePitchExtractor(
                        ctx = ctx,
                        binPath = cachedBin.absolutePath,
                        staticAudioLen = QNN_DEFAULT_T * 160,  // = chunkSamples
                    )
                    qnnRmvpe
                } else {
                    Log.i(TAG, "factory: loading rmvpe (ORT)")
                    ortRmvpe = OrtRuntime.openSession(ctx, rmvpeUri)
                    OrtRmvpePitchExtractor(ortRmvpe)
                }
            } else null

            return RvcPipeline(
                metadata = metadata,
                embedder = embedder,
                pitchExtractor = pitchExtractor,
                synthesizer = synth,
            )
        } catch (t: Throwable) {
            Log.e(TAG, "factory: aborting; closing partial sessions", t)
            runCatching { qnnSynth?.close() }
            runCatching { qnnHubert?.close() }
            runCatching { qnnRmvpe?.close() }
            runCatching { ortSynth?.close() }
            runCatching { ortHubert?.close() }
            runCatching { ortRmvpe?.close() }
            throw t
        }
    }

    private fun cacheBinUri(ctx: Context, uri: Uri): java.io.File {
        val expected = ctx.contentResolver.openFileDescriptor(uri, "r")
            ?.use { it.statSize } ?: -1L
        val key = uri.toString().hashCode().toLong().let { kotlin.math.abs(it) }
        val cached = java.io.File(ctx.cacheDir, "qnn-$key-$expected.bin")
        if (cached.exists() && cached.length() == expected && expected > 0L) {
            Log.d(TAG, "cacheBinUri: reusing ${cached.name}")
            return cached
        }
        ctx.cacheDir.listFiles { f -> f.name.startsWith("qnn-$key-") }
            ?.forEach { if (it != cached) runCatching { it.delete() } }
        Log.i(TAG, "cacheBinUri: copying $uri → ${cached.name} ($expected bytes)")
        ctx.contentResolver.openInputStream(uri)?.use { input ->
            cached.outputStream().use { input.copyTo(it) }
        } ?: error("cannot open .bin uri: $uri")
        return cached
    }

    private fun queryDisplayName(ctx: Context, uri: Uri): String {
        ctx.contentResolver.query(
            uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
            null, null, null,
        )?.use { c -> if (c.moveToFirst()) return c.getString(0) }
        return uri.lastPathSegment ?: "(unknown)"
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
