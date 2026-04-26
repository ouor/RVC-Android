package com.ouor.rvcandroid.inference

import android.content.Context
import android.util.Log
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.Closeable
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "Rvc.QnnHubRunner"

// Magic numbers reused from the synth wire protocol — INFR/RESP/REDY/QUIT
// have the same byte values, only the body shape differs per runner.
private const val MAGIC_INFR: Int = 0x52464E49
private const val MAGIC_RESP: Int = 0x50534552
private const val MAGIC_REDY: Int = 0x59444552
private const val MAGIC_QUIT: Int = 0x54495551

// Sibling of QnnSynthRunner. Spawns librvc_hubert_runner.so as a
// separate child process so HuBERT gets its own fastrpc PD on the
// DSP, isolated from the synth runner. Each infer() pumps one fp32
// audio chunk in and reads back fp32 features — AI Hub's compile of
// the trimmed ContentVec graph kept fp32 IO (input is naturally fp32
// audio, and the graph's outputs stayed fp32 even though dispatch
// runs fp16 internally).
class QnnHubertRunner(
    ctx: Context,
    binPath: String,
) : Closeable {
    private val process: Process
    private val out: DataOutputStream
    private val inp: DataInputStream

    init {
        val nativeLibDir = ctx.applicationInfo.nativeLibraryDir
        val runner = File(nativeLibDir, "librvc_hubert_runner.so")
        require(runner.exists()) {
            "hubert runner binary not found: ${runner.absolutePath} — extractNativeLibs may be off"
        }

        val skelDir = ensureSkelDir(ctx)

        Log.i(TAG, "spawning ${runner.name} --bin ${File(binPath).name}")
        val pb = ProcessBuilder(runner.absolutePath, "--bin", binPath)
            .redirectErrorStream(false)
        pb.environment().apply {
            val adspPath = buildAdspLibraryPath(skelDir)
            put("ADSP_LIBRARY_PATH", adspPath)
            put("DSP_LIBRARY_PATH", adspPath)
            val existing = get("LD_LIBRARY_PATH").orEmpty()
            put(
                "LD_LIBRARY_PATH",
                if (existing.isEmpty()) nativeLibDir else "$nativeLibDir:$existing",
            )
        }
        process = pb.start()
        // 256 KiB buffers — INFR body for 30720-sample chunk is
        // 4 + (4 + 30720*4) = ~120 KiB; RESP is 4*4 + 96*768*2 = ~144 KiB.
        // One buffer flush per call coalesces the per-byte writes from
        // DataOutputStream into a single pipe write.
        out = DataOutputStream(BufferedOutputStream(process.outputStream, 256 shl 10))
        inp = DataInputStream(BufferedInputStream(process.inputStream, 256 shl 10))

        Thread {
            process.errorStream.bufferedReader().forEachLine { line ->
                Log.w(TAG, "stderr: $line")
            }
        }.apply {
            isDaemon = true
            name = "QnnHubertRunner-stderr"
            start()
        }

        val magic = readIntLE(inp)
        require(magic == MAGIC_REDY) {
            "hubert runner did not send READY (got 0x${magic.toUInt().toString(16)})"
        }
        Log.i(TAG, "runner READY")
    }

    // Returns features (fp32), numFrames, channels. numFrames may be
    // less than naively expected (samples / 320) because ContentVec's
    // stride-5/3 conv stack loses one frame to receptive field — at
    // 5120 samples we get 15 frames not 16.
    data class HubertOutput(
        val features: FloatArray,
        val numFrames: Int,
        val channels: Int,
    )

    fun infer(audio: FloatArray): HubertOutput {
        writeIntLE(out, MAGIC_INFR)
        writeIntLE(out, audio.size)            // numSamples
        writeFloatArray(out, audio)
        out.flush()

        val magic = readIntLE(inp)
        require(magic == MAGIC_RESP) {
            "expected RESP, got 0x${magic.toUInt().toString(16)}"
        }
        val status = readIntLE(inp)
        val numFrames = readIntLE(inp)
        val channels = readIntLE(inp)
        val features = readFloatArrayWire(inp)
        val errMsg = readString(inp)
        if (status != 0) error("hubert runner inference failed (status=$status): $errMsg")
        return HubertOutput(features, numFrames, channels)
    }

    override fun close() {
        runCatching {
            writeIntLE(out, MAGIC_QUIT)
            out.flush()
        }
        runCatching { process.waitFor() }
        Log.d(TAG, "runner exited (code=${runCatching { process.exitValue() }.getOrNull()})")
    }

    private fun ensureSkelDir(ctx: Context): File {
        val skelDir = File(ctx.filesDir, "qnn_skel").apply {
            mkdirs()
            setReadable(true, /* ownerOnly = */ false)
            setExecutable(true, false)
        }
        val assetNames = ctx.assets.list("qnn_skel") ?: emptyArray()
        if (assetNames.isEmpty()) {
            Log.w(TAG, "ensureSkelDir: assets/qnn_skel/ empty — run tools/setup_qnn_libs.sh")
        }
        for (name in assetNames) {
            val target = File(skelDir, name)
            if (!(target.exists() && target.length() > 0)) {
                Log.i(TAG, "ensureSkelDir: extracting $name")
                ctx.assets.open("qnn_skel/$name").use { input ->
                    target.outputStream().use { input.copyTo(it) }
                }
            }
            target.setReadable(true, /* ownerOnly = */ false)
        }
        return skelDir
    }

    private fun buildAdspLibraryPath(skelDir: File): String = listOf(
        skelDir.absolutePath,
        "/system/lib/rfsa/adsp",
        "/vendor/lib/rfsa/adsp",
        "/vendor/lib64/rfs/dsp/snap",
        "/vendor/dsp",
        "/dsp",
    ).joinToString(separator = ";")
}

private fun writeIntLE(out: DataOutputStream, v: Int) {
    out.write(v and 0xff)
    out.write((v ushr 8) and 0xff)
    out.write((v ushr 16) and 0xff)
    out.write((v ushr 24) and 0xff)
}

private fun readIntLE(inp: DataInputStream): Int {
    val b0 = inp.read()
    val b1 = inp.read()
    val b2 = inp.read()
    val b3 = inp.read()
    if (b0 < 0 || b1 < 0 || b2 < 0 || b3 < 0) error("unexpected EOF on hubert runner stream")
    return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
}

private fun writeFloatArray(out: DataOutputStream, arr: FloatArray) {
    writeIntLE(out, arr.size)
    if (arr.isEmpty()) return
    val buf = ByteBuffer.allocate(arr.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buf.asFloatBuffer().put(arr)
    out.write(buf.array())
}

private fun readFloatArrayWire(inp: DataInputStream): FloatArray {
    val n = readIntLE(inp)
    if (n == 0) return FloatArray(0)
    val bytes = ByteArray(n * 4)
    inp.readFully(bytes)
    val out = FloatArray(n)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
    return out
}

private fun readString(inp: DataInputStream): String {
    val n = readIntLE(inp)
    if (n == 0) return ""
    val bytes = ByteArray(n)
    inp.readFully(bytes)
    return String(bytes, Charsets.UTF_8)
}
