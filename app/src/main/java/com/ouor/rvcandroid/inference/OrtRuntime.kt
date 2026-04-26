package com.ouor.rvcandroid.inference

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.File
import java.util.EnumSet
import kotlin.math.absoluteValue

private const val TAG = "Rvc.Ort"
private const val ORT_LOG_ID = "rvc-ort"

object OrtRuntime {
    // INFO-level env unlocks NNAPI partition reports ("Number of partitions
    // supported by NNAPI: X / Y") and the per-op fallback warnings that tell
    // us whether the EP is actually dispatching to silicon. ORT's native
    // logger routes to logcat under tag "onnxruntime" on Android — filter
    // with `adb logcat -v brief onnxruntime:V *:S` alongside Rvc.* tags.
    val env: OrtEnvironment = OrtEnvironment.getEnvironment(
        OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO,
        ORT_LOG_ID,
    ).also {
        Log.i(TAG, "init: env created at INFO (log_id=$ORT_LOG_ID)")
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
        // Newer ORT Java APIs expose getProviders(); reflective lookup avoids
        // hard-failing if the bundled binary doesn't have it.
        runCatching {
            val providers = OrtSession::class.java.getMethod("getProviders").invoke(session)
            Log.i(TAG, "  providers: $providers")
        }
        return session
    }

    fun openSession(ctx: Context, uri: Uri): OrtSession {
        return openSession(ensureCachedFile(ctx, uri))
    }

    // NNAPI dispatches to whatever vendor accelerator the device exposes
    // (Hexagon NPU on Snapdragon, NPU on Exynos, etc.). It can refuse a
    // graph for many reasons — unsupported op, dynamic shape it can't
    // partition, version mismatch — so we always carry a CPU-only retry
    // path. USE_FP16 lets NNAPI run FP32 graphs in FP16 internally where
    // the driver supports it; precision loss is bounded by RVC's already
    // lossy synthesis path and the speedup on NPU silicon is large.
    private fun createWithFallback(file: File): Pair<OrtSession, String> {
        val nnapiOpts = newOptions()
        try {
            nnapiOpts.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            Log.i(TAG, "createWithFallback: trying NNAPI (USE_FP16) for ${file.name}")
            return env.createSession(file.absolutePath, nnapiOpts) to "NNAPI"
        } catch (t: Throwable) {
            Log.w(
                TAG,
                "createWithFallback: NNAPI rejected ${file.name} (${t.javaClass.simpleName}: ${t.message}); falling back to CPU",
            )
            runCatching { nnapiOpts.close() }
        }
        return env.createSession(file.absolutePath, newOptions()) to "CPU"
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
