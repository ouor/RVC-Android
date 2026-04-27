package com.ouor.rvcandroid.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

private const val TAG = "Rvc.Recorder"

/**
 * 16-bit PCM mic capture writing a WAV header on stop.
 *
 * 44.1 kHz mono matches what the existing pipeline downsamples from a
 * regular phone mic anyway; using 48 kHz on top of that just adds a
 * resample without buying the user any quality. Each [start] writes
 * directly to the supplied [File] so the caller can hand the result back
 * to AudioIo.decode without an extra copy.
 */
class PcmRecorder(
    val sampleRate: Int = 44_100,
) {
    private var record: AudioRecord? = null
    private var thread: Thread? = null
    @Volatile private var running = false
    @Volatile private var lastAmplitude = 0f
    private var outFile: File? = null
    private var bytesWritten = 0L

    val isRecording: Boolean get() = running
    val currentAmplitude: Float get() = lastAmplitude

    @SuppressLint("MissingPermission") // Caller is required to obtain RECORD_AUDIO before invoking start().
    fun start(target: File) {
        if (running) error("already recording")
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        require(minBuf > 0) { "audio record not available" }
        val bufSize = (minBuf * 2).coerceAtLeast(8192)
        val raf = RandomAccessFile(target, "rw")
        // Reserve room for the WAV header; we'll backfill it on stop once
        // we know the real data size.
        raf.setLength(0)
        raf.write(ByteArray(44))
        raf.close()

        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize,
        )
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            error("AudioRecord init failed")
        }
        rec.startRecording()

        record = rec
        outFile = target
        bytesWritten = 0L
        running = true
        thread = Thread({ readLoop(rec, target, bufSize) }, "PcmRecorder").also { it.start() }
        Log.i(TAG, "start: $target @ ${sampleRate}Hz, bufSize=$bufSize")
    }

    fun stop(): File? {
        if (!running) return outFile
        running = false
        thread?.join(500)
        thread = null
        record?.let {
            runCatching { it.stop() }
            runCatching { it.release() }
        }
        record = null
        val file = outFile ?: return null
        finaliseWavHeader(file, bytesWritten.toInt())
        Log.i(TAG, "stop: ${file.length()} bytes total ($bytesWritten data)")
        return file
    }

    private fun readLoop(rec: AudioRecord, target: File, bufSize: Int) {
        val buf = ByteArray(bufSize)
        // The 44-byte WAV header is already reserved by start(); appending
        // here writes the PCM payload right after it. We backfill the
        // header in stop() once we know the final data size.
        java.io.FileOutputStream(target, /* append = */ true).buffered().use { tail ->
            while (running) {
                val n = rec.read(buf, 0, buf.size)
                if (n <= 0) continue
                tail.write(buf, 0, n)
                bytesWritten += n
                lastAmplitude = peakAmplitude(buf, n)
            }
        }
    }

    private fun peakAmplitude(buf: ByteArray, n: Int): Float {
        var peak = 0
        var i = 0
        while (i + 1 < n) {
            val s = (buf[i].toInt() and 0xFF) or (buf[i + 1].toInt() shl 8)
            val v = abs(s.toShort().toInt())
            if (v > peak) peak = v
            i += 2
        }
        return peak / 32768f
    }

    private fun finaliseWavHeader(file: File, dataSize: Int) {
        val totalSize = 36 + dataSize
        val byteRate = sampleRate * 2 // mono 16-bit
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray(Charsets.US_ASCII))
            putInt(totalSize)
            put("WAVE".toByteArray(Charsets.US_ASCII))
            put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16)
            putShort(1)            // PCM
            putShort(1)            // mono
            putInt(sampleRate)
            putInt(byteRate)
            putShort(2)            // block align (mono * 2 bytes)
            putShort(16)           // bits per sample
            put("data".toByteArray(Charsets.US_ASCII))
            putInt(dataSize)
        }.array()
        RandomAccessFile(file, "rw").use { raf ->
            raf.seek(0)
            raf.write(header)
        }
    }
}
