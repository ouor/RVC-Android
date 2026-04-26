package com.ouor.rvcandroid.audio

import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class WavData(val samples: FloatArray, val sampleRate: Int) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is WavData) return false
        return sampleRate == other.sampleRate && samples.contentEquals(other.samples)
    }

    override fun hashCode(): Int = 31 * sampleRate + samples.contentHashCode()
}

object WavIo {
    private const val FORMAT_PCM = 1
    private const val FORMAT_FLOAT = 3
    private const val FORMAT_EXTENSIBLE = 0xFFFE

    fun read(input: InputStream): WavData = parse(input.readBytes())

    fun write(output: OutputStream, samples: FloatArray, sampleRate: Int) {
        output.write(encodePcm16Mono(samples, sampleRate))
    }

    private fun parse(bytes: ByteArray): WavData {
        require(bytes.size >= 44) { "wav too small: ${bytes.size}" }
        require(bytes.matches(0, "RIFF")) { "not a RIFF file" }
        require(bytes.matches(8, "WAVE")) { "not a WAVE file" }

        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        var pos = 12
        var format = 0
        var channels = 0
        var sampleRate = 0
        var bitsPerSample = 0
        var dataOffset = -1
        var dataSize = 0

        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val size = bb.getInt(pos + 4)
            pos += 8
            when (id) {
                "fmt " -> {
                    format = bb.getShort(pos).toInt() and 0xFFFF
                    channels = bb.getShort(pos + 2).toInt() and 0xFFFF
                    sampleRate = bb.getInt(pos + 4)
                    bitsPerSample = bb.getShort(pos + 14).toInt() and 0xFFFF
                    if (format == FORMAT_EXTENSIBLE && size >= 40) {
                        format = bb.getShort(pos + 24).toInt() and 0xFFFF
                    }
                }
                "data" -> {
                    dataOffset = pos
                    dataSize = size
                }
            }
            pos += size + (size and 1)
        }

        require(dataOffset >= 0) { "no data chunk" }
        require(channels in 1..8) { "unsupported channels: $channels" }
        require(sampleRate > 0) { "invalid sample rate" }

        val bytesPerSample = bitsPerSample / 8
        val frames = dataSize / (channels * bytesPerSample)
        val samples = FloatArray(frames)

        when {
            format == FORMAT_PCM && bitsPerSample == 16 -> {
                for (i in 0 until frames) {
                    var sum = 0
                    val base = dataOffset + i * channels * 2
                    for (c in 0 until channels) sum += bb.getShort(base + c * 2).toInt()
                    samples[i] = (sum.toFloat() / channels) / 32768f
                }
            }
            format == FORMAT_PCM && bitsPerSample == 24 -> {
                for (i in 0 until frames) {
                    var sum = 0
                    val base = dataOffset + i * channels * 3
                    for (c in 0 until channels) {
                        val o = base + c * 3
                        val b0 = bytes[o].toInt() and 0xFF
                        val b1 = bytes[o + 1].toInt() and 0xFF
                        val b2 = bytes[o + 2].toInt()
                        sum += (b2 shl 16) or (b1 shl 8) or b0
                    }
                    samples[i] = (sum.toFloat() / channels) / 8388608f
                }
            }
            format == FORMAT_PCM && bitsPerSample == 32 -> {
                for (i in 0 until frames) {
                    var sum = 0L
                    val base = dataOffset + i * channels * 4
                    for (c in 0 until channels) sum += bb.getInt(base + c * 4)
                    samples[i] = (sum.toFloat() / channels) / 2147483648f
                }
            }
            format == FORMAT_FLOAT && bitsPerSample == 32 -> {
                for (i in 0 until frames) {
                    var sum = 0f
                    val base = dataOffset + i * channels * 4
                    for (c in 0 until channels) sum += bb.getFloat(base + c * 4)
                    samples[i] = sum / channels
                }
            }
            else -> error("unsupported wav: format=$format bits=$bitsPerSample")
        }

        return WavData(samples, sampleRate)
    }

    private fun encodePcm16Mono(samples: FloatArray, sampleRate: Int): ByteArray {
        val dataSize = samples.size * 2
        val totalSize = 44 + dataSize
        val buf = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray(Charsets.US_ASCII))
        buf.putInt(totalSize - 8)
        buf.put("WAVE".toByteArray(Charsets.US_ASCII))
        buf.put("fmt ".toByteArray(Charsets.US_ASCII))
        buf.putInt(16)
        buf.putShort(FORMAT_PCM.toShort())
        buf.putShort(1)
        buf.putInt(sampleRate)
        buf.putInt(sampleRate * 2)
        buf.putShort(2)
        buf.putShort(16)
        buf.put("data".toByteArray(Charsets.US_ASCII))
        buf.putInt(dataSize)
        for (s in samples) {
            val v = (s.coerceIn(-1f, 1f) * 32767f).toInt()
            buf.putShort(v.toShort())
        }
        return buf.array()
    }

    private fun ByteArray.matches(offset: Int, ascii: String): Boolean {
        if (offset + ascii.length > size) return false
        for (i in ascii.indices) if (this[offset + i].toInt() != ascii[i].code) return false
        return true
    }
}
