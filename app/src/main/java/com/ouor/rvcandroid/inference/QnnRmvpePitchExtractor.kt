package com.ouor.rvcandroid.inference

import android.content.Context
import android.util.Log

private const val TAG = "Rvc.QnnRMVPE"

// Static-shape RMVPE pitch extractor backed by a child-process QNN
// runtime. Compiled at audio_samples=staticAudioLen (typically 30720
// = one synth chunk @ 16 kHz), giving 192 pitch frames per call —
// exactly the synth's static T axis. The pipeline calls this once
// per synth chunk.
//
// Boundary trade-off: RMVPE's GRU keeps state across the input, so
// chunked extraction loses the cross-chunk hidden state that the
// upfront-on-full-audio path enjoys. Pitch may have small jumps at
// chunk seams; for PoC quality this has been acceptable.
class QnnRmvpePitchExtractor(
    ctx: Context,
    binPath: String,
    override val staticAudioLen: Int,
) : PitchExtractor {

    private val runner = QnnRmvpeRunner(ctx, binPath)

    init {
        Log.i(TAG, "init: staticAudioLen=$staticAudioLen")
    }

    override fun extract(
        audio16k: FloatArray,
        f0UpKey: Int,
        threshold: Float,
    ): PitchData {
        require(audio16k.size == staticAudioLen) {
            "QNN RMVPE requires audio length $staticAudioLen, got ${audio16k.size}"
        }
        val t0 = System.nanoTime()
        val out = runner.infer(audio16k, threshold)
        val tRan = System.nanoTime()
        val shifted = if (f0UpKey == 0) out.pitchf else shiftPitch(out.pitchf, f0UpKey)
        val coarse = melQuantize(shifted)
        val tDone = System.nanoTime()
        Log.i(
            TAG,
            "extract: audio=$staticAudioLen → pitchf[${out.numFrames}] " +
                "(voiced=${coarse.count { it > 1 }}/${coarse.size}) " +
                "in ${(tRan - t0) / 1_000_000}ms (incl IPC) decode=${(tDone - tRan) / 1_000_000}ms",
        )
        return PitchData(shifted, coarse)
    }

    override fun close() {
        Log.d(TAG, "close")
        runner.close()
    }
}
