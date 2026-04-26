package com.ouor.rvcandroid.inference

import android.content.Context
import android.util.Log

private const val TAG = "Rvc.QnnHuBERT"

// Static-shape HuBERT (ContentVec) embedder backed by a child-process
// QNN runtime. The .bin baked an audio_samples=staticAudioLen input,
// so the pipeline must call this with exactly that many samples per
// invocation. AI Hub kept the graph's IO at fp32 for this compile, so
// no Half packing is needed on either side.
class QnnHubertEmbedder(
    ctx: Context,
    binPath: String,
    override val staticAudioLen: Int,
) : HubertEmbedder {

    private val runner = QnnHubertRunner(ctx, binPath)

    init {
        Log.i(TAG, "init: staticAudioLen=$staticAudioLen")
    }

    override fun extract(audio16k: FloatArray): EmbeddingData {
        require(audio16k.size == staticAudioLen) {
            "QNN HuBERT requires audio length $staticAudioLen, got ${audio16k.size}"
        }
        val t0 = System.nanoTime()
        val out = runner.infer(audio16k)
        val elapsed = (System.nanoTime() - t0) / 1_000_000
        Log.i(
            TAG,
            "extract: audio=$staticAudioLen → feats[1, ${out.numFrames}, ${out.channels}] " +
                "in ${elapsed}ms (incl IPC)",
        )
        return EmbeddingData(out.features, out.numFrames, out.channels)
    }

    override fun close() {
        Log.d(TAG, "close")
        runner.close()
    }
}
