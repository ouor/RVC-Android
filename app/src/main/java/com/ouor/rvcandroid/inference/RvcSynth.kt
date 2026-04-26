package com.ouor.rvcandroid.inference

import java.io.Closeable

// Common surface for the synthesizer stage so RvcPipeline can hold either
// the ORT-backed RvcSynthesizer or the QNN-backed QnnRvcSynthesizer
// without knowing which one is loaded. The static-T detection it exposes
// is the only signal the pipeline branches on — same chunking code runs
// against both backends.
sealed interface RvcSynth : Closeable {
    val staticT: Int?

    fun infer(
        feats: FloatArray,
        framesT: Int,
        channels: Int,
        pitch: LongArray? = null,
        pitchf: FloatArray? = null,
        speakerId: Long = 0L,
    ): FloatArray
}
