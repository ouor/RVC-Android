package com.ouor.rvcandroid.inference

import android.content.Context
import android.util.Half
import android.util.Log
import java.util.Random

private const val TAG = "Rvc.QnnSynth"

// Static-shape RVC synthesizer backed by a child-process QNN runtime
// (see QnnSynthRunner). The Kotlin layer here is responsible for
// generating the per-call rand_noise (Hexagon HTP can't generate
// randoms; the export-time forward() patch in
// tools/export_static_synthesizer.py replaced torch.randn_like with an
// external rand_noise input), packing the fp16 inputs the graph wants
// (so the runner doesn't redo the conversion and the wire payload is
// half the fp32 size), and truncating int64 pitch values down to int32
// to match what AI Hub's --truncate_64bit_io baked into the binary.
class QnnRvcSynthesizer(
    ctx: Context,
    binPath: String,
    override val staticT: Int,
    private val interChannels: Int,
    private val outputSampleRate: Int,
) : RvcSynth {

    private val runner = QnnSynthRunner(ctx, binPath)
    private val random = Random()
    // Fixed graph T → fixed audio output length. Synth upsamples
    // 100fps post-2x-upsample frames to outputSampleRate samples per
    // second, i.e. (outputSampleRate / 100) samples per frame.
    private val outAudioSize: Int = staticT * (outputSampleRate / 100)

    init {
        Log.i(
            TAG,
            "init: T=$staticT IC=$interChannels sr=$outputSampleRate → audio[$outAudioSize]",
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

        val pitchInt = IntArray(pitch.size) { pitch[it].toInt() }

        // Pack feats fp32 → fp16 once on the host. Generating noise
        // straight into a ShortArray skips the fp32 intermediate.
        val featsHalf = ShortArray(feats.size) { Half.toHalf(feats[it]) }
        val noiseHalf = ShortArray(interChannels * staticT) {
            Half.toHalf(random.nextGaussian().toFloat())
        }

        val t0 = System.nanoTime()
        val audioHalf = runner.infer(
            feats = featsHalf,
            framesT = framesT,
            channels = channels,
            pitch = pitchInt,
            pitchf = pitchf,
            sid = speakerId.toInt(),
            noise = noiseHalf,
        )
        val audio = FloatArray(audioHalf.size) { Half.toFloat(audioHalf[it]) }
        val elapsed = (System.nanoTime() - t0) / 1_000_000
        Log.i(
            TAG,
            "infer: feats[$framesT,$channels] sid=$speakerId → audio[${audio.size}] in ${elapsed}ms (incl IPC, fp16)",
        )
        return audio
    }

    override fun close() {
        Log.d(TAG, "close")
        runner.close()
    }
}
