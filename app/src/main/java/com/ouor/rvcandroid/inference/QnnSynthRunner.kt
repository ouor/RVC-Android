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

private const val TAG = "Rvc.QnnRunner"

// Wire-protocol magic numbers, little-endian byte order on disk:
// 'I','N','F','R' = 0x52,0x46,0x4E,0x49 → 0x52464E49 as LE u32
private const val MAGIC_INFR: Int = 0x52464E49
private const val MAGIC_RESP: Int = 0x50534552
private const val MAGIC_REDY: Int = 0x59444552
private const val MAGIC_QUIT: Int = 0x54495551

// Spawns librvc_synth_runner.so (a PIE executable shipped via jniLibs)
// as a child process. The child inherits the app's uid but transitions
// to a different SELinux domain, which is what lets it open an
// unsigned process domain on the Hexagon DSP via fastrpc — the policy
// that blocked the in-process JNI version (Phase η). Each infer() is
// one stdin/stdout round-trip; the binary stays alive across calls so
// session setup (load .bin, retrieve graph) is paid once.
class QnnSynthRunner(
    ctx: Context,
    binPath: String,
) : Closeable {
    private val process: Process
    private val out: DataOutputStream
    private val inp: DataInputStream

    init {
        val nativeLibDir = ctx.applicationInfo.nativeLibraryDir
        val runner = File(nativeLibDir, "librvc_synth_runner.so")
        require(runner.exists()) {
            "runner binary not found: ${runner.absolutePath} — extractNativeLibs may be off"
        }

        val skelDir = ensureSkelDir(ctx)

        Log.i(TAG, "spawning ${runner.name} --bin ${File(binPath).name}")
        val pb = ProcessBuilder(runner.absolutePath, "--bin", binPath)
            .redirectErrorStream(false)
        pb.environment().apply {
            // ADSP_LIBRARY_PATH / DSP_LIBRARY_PATH are read inside the
            // Hexagon driver to locate the V79 skel; setting them on
            // the child's environment is the only way to pass them
            // through, since Os.setenv in the parent doesn't propagate.
            val adspPath = buildAdspLibraryPath(skelDir)
            put("ADSP_LIBRARY_PATH", adspPath)
            put("DSP_LIBRARY_PATH", adspPath)
            // libQnnSystem/libQnnHtp are in the app's nativeLibraryDir;
            // Android already includes that on LD_LIBRARY_PATH for
            // child processes, but pinning it explicitly makes the
            // search deterministic.
            val existing = get("LD_LIBRARY_PATH").orEmpty()
            put(
                "LD_LIBRARY_PATH",
                if (existing.isEmpty()) nativeLibDir else "$nativeLibDir:$existing",
            )
        }
        process = pb.start()
        // 1 MiB buffers — a single chunk request is ~370 KiB now that
        // feats + noise are fp16; one buffer fits the whole INFR body
        // so DataOutputStream's per-byte writes coalesce into a single
        // pipe write.
        out = DataOutputStream(BufferedOutputStream(process.outputStream, 1 shl 20))
        inp = DataInputStream(BufferedInputStream(process.inputStream, 1 shl 20))

        // Drain stderr → logcat asynchronously so the runner doesn't
        // block when its stderr buffer fills.
        Thread {
            process.errorStream.bufferedReader().forEachLine { line ->
                Log.w(TAG, "stderr: $line")
            }
        }.apply {
            isDaemon = true
            name = "QnnRunner-stderr"
            start()
        }

        val magic = readIntLE(inp)
        require(magic == MAGIC_REDY) {
            "runner did not send READY (got 0x${magic.toUInt().toString(16)})"
        }
        Log.i(TAG, "runner READY")
    }

    // feats and noise are fp16 raw bits (IEEE 754 binary16) packed by
    // the caller; audio is returned as fp16 raw bits. The caller does
    // the fp32↔fp16 conversion (android.util.Half) — sending fp16 over
    // the wire halves the payload (~720 KiB → ~370 KiB per call) and
    // removes the runner's old f32→f16 staging copies.
    fun infer(
        feats: ShortArray,
        framesT: Int,
        channels: Int,
        pitch: IntArray,
        pitchf: FloatArray,
        sid: Int,
        noise: ShortArray,
    ): ShortArray {
        writeIntLE(out, MAGIC_INFR)
        writeIntLE(out, framesT)
        writeIntLE(out, channels)
        writeShortArray(out, feats)
        writeIntLE(out, framesT)            // p_len mirrors framesT
        writeIntArray(out, pitch)
        writeFloatArray(out, pitchf)
        writeIntLE(out, sid)
        writeShortArray(out, noise)
        out.flush()

        val magic = readIntLE(inp)
        require(magic == MAGIC_RESP) {
            "expected RESP, got 0x${magic.toUInt().toString(16)}"
        }
        val status = readIntLE(inp)
        val audio = readShortArray(inp)
        val errMsg = readString(inp)
        if (status != 0) error("runner inference failed (status=$status): $errMsg")
        return audio
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
    if (b0 < 0 || b1 < 0 || b2 < 0 || b3 < 0) error("unexpected EOF on runner stream")
    return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
}

private fun writeFloatArray(out: DataOutputStream, arr: FloatArray) {
    writeIntLE(out, arr.size)
    if (arr.isEmpty()) return
    val buf = ByteBuffer.allocate(arr.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buf.asFloatBuffer().put(arr)
    out.write(buf.array())
}

private fun writeIntArray(out: DataOutputStream, arr: IntArray) {
    writeIntLE(out, arr.size)
    if (arr.isEmpty()) return
    val buf = ByteBuffer.allocate(arr.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buf.asIntBuffer().put(arr)
    out.write(buf.array())
}

private fun writeShortArray(out: DataOutputStream, arr: ShortArray) {
    writeIntLE(out, arr.size)
    if (arr.isEmpty()) return
    val buf = ByteBuffer.allocate(arr.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    buf.asShortBuffer().put(arr)
    out.write(buf.array())
}

private fun readShortArray(inp: DataInputStream): ShortArray {
    val n = readIntLE(inp)
    if (n == 0) return ShortArray(0)
    val bytes = ByteArray(n * 2)
    inp.readFully(bytes)
    val out = ShortArray(n)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(out)
    return out
}

private fun readString(inp: DataInputStream): String {
    val n = readIntLE(inp)
    if (n == 0) return ""
    val bytes = ByteArray(n)
    inp.readFully(bytes)
    return String(bytes, Charsets.UTF_8)
}
