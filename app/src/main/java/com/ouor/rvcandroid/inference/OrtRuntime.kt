package com.ouor.rvcandroid.inference

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.net.Uri
import android.system.Os
import android.util.Log
import java.io.File
import kotlin.math.absoluteValue

private const val TAG = "Rvc.Ort"
private const val ORT_LOG_ID = "rvc-ort"

object OrtRuntime {
    val env: OrtEnvironment = OrtEnvironment.getEnvironment(
        OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO,
        ORT_LOG_ID,
    ).also {
        Log.i(TAG, "init: env created at INFO (log_id=$ORT_LOG_ID)")
    }

    @Volatile
    private var nativeLibDir: String? = null

    @Volatile
    private var qnnPreloaded: Boolean = false

    // Three things have to be in place before ORT can stand up a QNN HTP
    // session on a Snapdragon device:
    //
    //   1. host-side libQnn*.so loaded into our linker namespace, so ORT's
    //      C++ dlopen("libQnnHtp.so") resolves by basename.
    //   2. Hexagon-side skel libs (libQnnHtpV79Skel.so etc.) extracted to a
    //      directory the DSP can read. They can't live in jniLibs — Android
    //      refuses to load Hexagon binaries from an aarch64-v8a slot — so
    //      we bundle them in /assets/qnn_skel and copy to filesDir at
    //      first run.
    //   3. ADSP_LIBRARY_PATH env var pointing at that filesDir, so the QNN
    //      HTP host driver tells the DSP where to dlopen the skel.
    fun ensureInitialized(ctx: Context) {
        if (qnnPreloaded) return
        nativeLibDir = ctx.applicationInfo.nativeLibraryDir
        Log.i(TAG, "ensureInitialized: nativeLibDir=$nativeLibDir")
        extractHexagonSkels(ctx)
        for (name in listOf("QnnSystem", "QnnHtp")) {
            runCatching { System.loadLibrary(name) }
                .onSuccess { Log.i(TAG, "ensureInitialized: preloaded lib$name.so") }
                .onFailure { e ->
                    Log.w(TAG, "ensureInitialized: preload lib$name.so failed: ${e.message}")
                }
        }
        qnnPreloaded = true
    }

    private fun extractHexagonSkels(ctx: Context) {
        val skelDir = File(ctx.filesDir, "qnn_skel").also { it.mkdirs() }
        val assetNames = ctx.assets.list("qnn_skel") ?: emptyArray()
        if (assetNames.isEmpty()) {
            Log.w(
                TAG,
                "extractHexagonSkels: no assets/qnn_skel/ contents — run tools/setup_qnn_libs.sh",
            )
        }
        for (name in assetNames) {
            val out = File(skelDir, name)
            if (out.exists() && out.length() > 0) continue
            Log.i(TAG, "extractHexagonSkels: $name → ${out.absolutePath}")
            ctx.assets.open("qnn_skel/$name").use { input ->
                out.outputStream().use { input.copyTo(it) }
            }
        }
        // Default ADSP search path varies by OEM, but /system/lib/rfsa/adsp +
        // /vendor/lib/rfsa/adsp + /dsp covers what every Snapdragon Android
        // device exposes. Prepending our extracted dir gets the QNN HTP
        // driver to use OUR skel for V79 instead of any older system one.
        val current = runCatching { System.getenv("ADSP_LIBRARY_PATH") }.getOrNull().orEmpty()
        val newPath = buildString {
            append(skelDir.absolutePath)
            append(';')
            if (current.isNotEmpty()) append(current).append(';')
            append("/system/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/dsp")
        }
        runCatching { Os.setenv("ADSP_LIBRARY_PATH", newPath, true) }
            .onSuccess { Log.i(TAG, "ADSP_LIBRARY_PATH=$newPath") }
            .onFailure { e -> Log.w(TAG, "setenv ADSP_LIBRARY_PATH failed: ${e.message}") }
    }

    fun openSession(file: File): OrtSession {
        val t0 = System.currentTimeMillis()
        val (session, backend) = createWithFallback(file)
        val elapsed = System.currentTimeMillis() - t0
        Log.i(
            TAG,
            "openSession: ${file.length() / (1024 * 1024)} MiB loaded from ${file.name} via $backend in ${elapsed}ms",
        )
        Log.d(TAG, "  inputs:  ${describe(session.inputInfo)}")
        Log.d(TAG, "  outputs: ${describe(session.outputInfo)}")
        runCatching {
            val providers = OrtSession::class.java.getMethod("getProviders").invoke(session)
            Log.i(TAG, "  providers: $providers")
        }
        return session
    }

    fun openSession(ctx: Context, uri: Uri): OrtSession {
        ensureInitialized(ctx)
        return openSession(ensureCachedFile(ctx, uri))
    }

    // QNN HTP dispatches op kernels to the Hexagon NPU. Fallback chain is
    // QNN → CPU. NNAPI is dropped from the chain: on this generation of
    // Snapdragon it partitioned 0 ops on our dynamic-shape graphs (verified
    // via BFCArena allocation logs and the absence of NNAPI partition
    // reports in INFO-level ORT logs), so it added load latency for no
    // throughput gain. With static-shape models from Phase γ onward, QNN's
    // partitioner has a real chance of claiming the synthesizer.
    private fun createWithFallback(file: File): Pair<OrtSession, String> {
        val qnnOpts = newOptions()
        try {
            qnnOpts.addQnn(qnnHtpOptions())
            Log.i(TAG, "createWithFallback: trying QNN HTP for ${file.name}")
            return env.createSession(file.absolutePath, qnnOpts) to "QNN"
        } catch (t: Throwable) {
            Log.w(
                TAG,
                "createWithFallback: QNN rejected ${file.name} (${t.javaClass.simpleName}: ${t.message}); falling back to CPU",
            )
            runCatching { qnnOpts.close() }
        }
        return env.createSession(file.absolutePath, newOptions()) to "CPU"
    }

    // With the SDK 2.45 lib swap (tools/setup_qnn_libs.sh + packaging
    // pickFirsts in build.gradle.kts) ORT 1.22's QNN EP calls into a
    // driver that actually knows about Hexagon V79. We don't pass htp_arch
    // here because ORT 1.22's ParseHtpArchitecture enum doesn't include 79
    // — leaving it unset lets the newer driver auto-detect from the
    // running chip, which is what we want anyway.
    private fun qnnHtpOptions(): Map<String, String> {
        return mapOf(
            "backend_path" to "libQnnHtp.so",
        )
    }

    private fun newOptions(): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
        setSessionLogVerbosityLevel(0)
    }

    // SAF URIs cannot be opened as filesystem paths, but ORT's mmap-backed
    // createSession(filePath) can. Streaming each ONNX once into cacheDir
    // and feeding ORT the resulting path avoids loading 350+ MB models
    // through the Java heap, which previously OOM'd at the contentvec step.
    // Subsequent calls with the same URI reuse the cached copy.
    private fun ensureCachedFile(ctx: Context, uri: Uri): File {
        val expectedSize = ctx.contentResolver.openFileDescriptor(uri, "r")
            ?.use { it.statSize } ?: -1L
        val key = uri.toString().hashCode().toLong().absoluteValue
        val cached = File(ctx.cacheDir, "ort-$key-$expectedSize.onnx")
        if (cached.exists() && cached.length() == expectedSize && expectedSize > 0L) {
            Log.d(TAG, "ensureCachedFile: reusing ${cached.name}")
            return cached
        }
        ctx.cacheDir.listFiles { f -> f.name.startsWith("ort-$key-") }
            ?.forEach { if (it != cached) runCatching { it.delete() } }
        Log.i(
            TAG,
            "ensureCachedFile: copying $uri → ${cached.name} ($expectedSize bytes)",
        )
        val t0 = System.currentTimeMillis()
        ctx.contentResolver.openInputStream(uri)?.use { input ->
            cached.outputStream().use { input.copyTo(it) }
        } ?: error("cannot open model uri: $uri")
        Log.i(TAG, "ensureCachedFile: copied in ${System.currentTimeMillis() - t0}ms")
        return cached
    }

    private fun describe(info: Map<String, NodeInfo>): String =
        info.entries.joinToString(prefix = "{", postfix = "}") { (name, ni) ->
            val v = ni.info
            if (v is TensorInfo) "$name:${v.type}${v.shape.contentToString()}"
            else "$name:$v"
        }
}
