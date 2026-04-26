package com.ouor.rvcandroid.inference

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.net.Uri
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

    // Android's runtime linker only resolves bare-name dlopen against libs
    // that live in the calling APK's namespace, and ORT's C++ side dlopens
    // the QNN backend (libQnnHtp.so etc.) by basename. System.loadLibrary
    // pulls the SOs into our namespace ahead of time so ORT's later dlopen
    // hits the namespace cache instead of failing on path resolution.
    fun ensureInitialized(ctx: Context) {
        if (nativeLibDir == null) {
            nativeLibDir = ctx.applicationInfo.nativeLibraryDir
            Log.i(TAG, "ensureInitialized: nativeLibDir=$nativeLibDir")
        }
        if (!qnnPreloaded) {
            for (name in listOf("QnnSystem", "QnnHtp")) {
                runCatching { System.loadLibrary(name) }
                    .onSuccess { Log.i(TAG, "ensureInitialized: preloaded lib$name.so") }
                    .onFailure { e ->
                        Log.w(TAG, "ensureInitialized: preload lib$name.so failed: ${e.message}")
                    }
            }
            qnnPreloaded = true
        }
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

    private fun qnnHtpOptions(): Map<String, String> {
        // backend_path is what ORT dlopens to enter QNN. The bare name works
        // because ensureInitialized already loaded libQnnHtp.so into the app
        // namespace. htp_performance_mode=burst gives best inference latency
        // (highest clock, no thermal pacing), and finalization mode 3 lets
        // QNN spend more time at session-creation tuning the graph for the
        // specific HTP chip — fine for an offline pipeline where load
        // latency is amortized across a whole conversion.
        return mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "htp_graph_finalization_optimization_mode" to "3",
            "enable_htp_fp16_precision" to "1",
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
