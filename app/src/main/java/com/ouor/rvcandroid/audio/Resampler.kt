package com.ouor.rvcandroid.audio

import android.util.Log

private const val TAG = "Rvc.Resamp"

object Resampler {
    fun resample(input: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        require(srcRate > 0 && dstRate > 0) { "invalid sample rate" }
        if (srcRate == dstRate || input.isEmpty()) return input
        val ratio = srcRate.toDouble() / dstRate
        val outLen = ((input.size.toLong() * dstRate) / srcRate).toInt()
        Log.d(TAG, "resample: ${srcRate}Hz -> ${dstRate}Hz, ${input.size} -> $outLen samples")
        val out = FloatArray(outLen)
        val lastIdx = input.size - 1
        for (i in 0 until outLen) {
            val srcPos = i * ratio
            val idx = srcPos.toInt()
            if (idx >= lastIdx) {
                out[i] = input[lastIdx]
            } else {
                val frac = (srcPos - idx).toFloat()
                val a = input[idx]
                val b = input[idx + 1]
                out[i] = a + (b - a) * frac
            }
        }
        return out
    }
}
