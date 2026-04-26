package com.ouor.rvcandroid.inference

import android.util.Log
import java.util.Random

private const val TAG = "Rvc.QnnSynth"

// AI Hub-compiled context binary for the synthesizer. The binary expects
// a fixed T axis (192 for Tsukuyomi v2 40k) and a rand_noise input that
// replaces the synth's torch.randn_like(m_p) — the runtime synthesises
// that noise on CPU per chunk and hands it to the NPU alongside feats.
class QnnRvcSynthesizer(
    binPath: String,
    override val staticT: Int,
    private val interChannels: Int,
    private val outputSampleRate: Int,
) : RvcSynth {

    private var handle: Long = QnnNative.nativeLoad(binPath)
    private val random = Random()
    // Fixed graph T → fixed audio output length. The synth upsamples
    // 100fps post-2x-upsample frames to outputSampleRate samples per
    // second, i.e. (outputSampleRate / 100) samples per frame.
    private val outAudioSize: Int = staticT * (outputSampleRate / 100)

    init {
        require(handle != 0L) { "QnnNative.nativeLoad returned 0 for $binPath" }
        Log.i(
            TAG,
            "init: T=$staticT IC=$interChannels sr=$outputSampleRate → audio[$outAudioSize] handle=0x${java.lang.Long.toHexString(handle)}",
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
        require(framesT == staticT) {
            "QNN synth requires T=$staticT, got T=$framesT — pipeline must chunk to match"
        }
        require(feats.size == framesT * channels) {
            "feats size ${feats.size} != $framesT * $channels"
        }
        requireNotNull(pitch) { "QNN synth (f0=true) requires pitch" }
        requireNotNull(pitchf) { "QNN synth (f0=true) requires pitchf" }
        require(pitch.size == framesT && pitchf.size == framesT) {
            "pitch/pitchf size must equal T=$framesT"
        }

        // Truncate int64 → int32. The AI-Hub-compiled binary already
        // expects int32 inputs (--truncate_64bit_io) and the values
        // (frame counts, mel bin indices in [1,255], speaker id) are
        // safely within int32 range for any RVC config.
        val pitchInt = IntArray(pitch.size) { pitch[it].toInt() }

        // Synthesise the noise the synth's z_p sample needs. Hexagon HTP
        // can't generate random numbers, so the export-time forward()
        // patch swaps torch.randn_like(m_p) for an external rand_noise
        // input — we feed CPU-generated Gaussian samples here.
        val noise = FloatArray(interChannels * staticT)
        for (i in noise.indices) noise[i] = random.nextGaussian().toFloat()

        return QnnNative.nativeInfer(
            handle,
            feats,
            framesT,
            pitchInt,
            pitchf,
            speakerId.toInt(),
            noise,
            outAudioSize,
        ) ?: error("QnnNative.nativeInfer returned null")
    }

    override fun close() {
        if (handle != 0L) {
            Log.d(TAG, "close: handle=0x${java.lang.Long.toHexString(handle)}")
            QnnNative.nativeRelease(handle)
            handle = 0L
        }
    }
}
