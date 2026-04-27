package com.ouor.rvcandroid.audio

data class AudioData(val samples: FloatArray, val sampleRate: Int) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is AudioData) return false
        return sampleRate == other.sampleRate && samples.contentEquals(other.samples)
    }

    override fun hashCode(): Int = 31 * sampleRate + samples.contentHashCode()
}
